import argparse, ast, io, json, os, sys, time, textwrap, multiprocessing as mp
import concurrent.futures as cf
from pathlib import Path
import math
import numpy as np
from termcolor import cprint
from tqdm import tqdm
from omegaconf import DictConfig, ListConfig, OmegaConf


def get_config():
    cli_conf   = OmegaConf.from_cli()
    yaml_conf  = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)

# ----------------- function-style -----------------
def _run_one(snippet: str, tests: list[str], t_limit: int, q):
    try:
        ns = {}
        exec(textwrap.dedent(snippet), ns, ns)
        for stmt in tests:
            exec(stmt, ns, ns)
        q.put(True)
    except Exception:
        q.put(False)

def _check_snippet(snippet: str, tests: list[str], t_limit: int) -> bool:
    q = mp.Queue()
    p = mp.Process(target=_run_one, args=(snippet, tests, t_limit, q))
    p.start()
    p.join(t_limit)
    if p.is_alive():
        p.terminate(); p.join()
        return False
    return q.get_nowait() if not q.empty() else False

from concurrent.futures import ProcessPoolExecutor, as_completed

def evaluate_function_dataset(data: list[dict],
                              n_workers: int | None = None):
    """
    run function-style in parallel, output 2D execution_result/correctness
    """
    n_workers = n_workers or mp.cpu_count()

    for item in data:
        m_code = len(item["extracted_output"])
        m_test = len(item["test_list"])
        item["execution_result"] = [[None]  * m_test for _ in range(m_code)]
        item["correctness"]      = [[False] * m_test for _ in range(m_code)]
        item.setdefault("step_map",         [])

    tasks = []
    for idx, item in enumerate(data):
        t_limit = item.get("test_time_limit", 1)
        for i, snippet in enumerate(item["extracted_output"]):
            for j, test_stmt in enumerate(item["test_list"]):
                tasks.append((idx, i, j, snippet, test_stmt, t_limit))

    # parallel
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_check_snippet, snippet, [test_stmt], t_limit): (idx, i, j)
            for idx, i, j, snippet, test_stmt, t_limit in tasks
        }

        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc=f"Function 2D tests ({n_workers} w)"):
            idx, i, j = futures[fut]
            ok = fut.result()
            data[idx]["execution_result"][i][j] = ok
            data[idx]["correctness"][i][j]      = ok

    return data


def _evaluate_item(item: dict) -> list[bool]:
    t_limit  = item.get("test_time_limit", 1)
    snippets = item["extracted_output"]
    tests    = item["test_list"]
    return [_check_snippet(s, tests, t_limit) for s in snippets]





def worker_stdio(script, input_val, output_queue):
    # Create an iterator over the input lines.
    input_lines = iter(input_val.splitlines())

    # Override the input() function in the exec context.
    def fake_input(prompt=""):
        try:
            return next(input_lines)
        except StopIteration:
            raise EOFError("No more input")
    
    # Redirect sys.stdout to capture printed output.
    stdout_capture = io.StringIO()
    original_stdout = sys.stdout
    original_stdin = sys.stdin  # Save original stdin
    sys.stdout = stdout_capture
    sys.stdin = io.StringIO(input_val)  # Simulate stdin with input_val

    context = {
        "__name__": "__main__",   # Ensures that `if __name__ == "__main__": ...` will fire
        "input": fake_input
    }

    try:
        exec(script, context)
        printed_output = stdout_capture.getvalue()
        output_queue.put(printed_output)

    except SystemExit:
        printed_output = stdout_capture.getvalue()
        output_queue.put(printed_output)

    except Exception as e:
        output_queue.put(f"error: {e}")

    finally:
        sys.stdout = original_stdout
        sys.stdin = original_stdin



def run_scripts_with_timeout(scripts, inputs, time_limits, worker):
    results = [None] * len(scripts)
    processes = []
    queues = []
    deadlines = []

    for i in range(len(scripts)):
        q = mp.Queue()
        p = mp.Process(target=worker, args=(scripts[i], inputs[i], q))
        processes.append(p)
        queues.append(q)
        p.start()
        deadlines.append(time.time() + time_limits[i])

    while any(p.is_alive() for p in processes):
        now = time.time()
        for i, p in enumerate(processes):
            if p.is_alive() and now >= deadlines[i]:
                p.terminate()
                results[i] = "Timeout Error"
        time.sleep(0.001)

    for i, p in enumerate(processes):
        if results[i] is None:
            try:
                results[i] = queues[i].get_nowait()
            except Exception as e:
                results[i] = f"Execution Error: {e}"

    return results

def test_if_eq(x, y): 
    return " ".join(x.split()) == " ".join(y.split())

def get_chunk_indices(n, num_chunks):
    size, rem = divmod(n, num_chunks)
    idx, start = [], 0
    for i in range(num_chunks):
        extra = 1 if i < rem else 0
        end   = start + size + extra
        idx.append((start, end)); start = end
    return idx







from tqdm import tqdm   

def run_scripts_with_chunk(code_list, test_input_list, time_limit_list,
                           worker, num_chunks):
    chunks = get_chunk_indices(len(code_list), num_chunks)

    exe_results = []
    pbar = tqdm(total=len(code_list), desc=f"STDIO tests ({num_chunks} ch)")  

    for start, end in chunks:
        sub_code_list       = code_list[start:end]
        sub_test_input_list = test_input_list[start:end]
        sub_time_limit_list = time_limit_list[start:end]

        sub_exe_results = run_scripts_with_timeout(
            sub_code_list,
            sub_test_input_list,
            sub_time_limit_list,
            worker
        )
        exe_results.extend(sub_exe_results)
        pbar.update(end - start)  

    pbar.close()          
    return exe_results


def evaluate_stdio_dataset(data: list[dict], num_chunks: int):
    
    idx_code, idx_case = [], []
    code_list, inp_list, tl_list = [], [], []

    for idx, item in enumerate(data):
        tl = item.get("test_time_limit", 1)
        m_code = len(item["extracted_output"])
        m_case = len(item["test_input"])

        data[idx]["execution_result"] = [[] for _ in range(m_code)]
        data[idx]["correctness"] = [[] for _ in range(m_code)]
        item.setdefault("step_map",           [])

        for c_idx, code in enumerate(item["extracted_output"]):
            for k in range(m_case):
                idx_code.append((idx, c_idx)) 
                idx_case.append(k)           
                code_list.append(code)
                inp_list.append(item["test_input"][k])
                tl_list.append(tl)

    exe_results = run_scripts_with_chunk(
        code_list, inp_list, tl_list, worker_stdio, num_chunks
    )

    for i, res in enumerate(exe_results):
        idx, c_idx = idx_code[i]
        k          = idx_case[i]
        item       = data[idx]

        while len(item["execution_result"][c_idx]) < k + 1:
            item["execution_result"][c_idx].append("")
            item["correctness"][c_idx].append(False)
        item["execution_result"][c_idx][k] = res
        exp_out = item["test_output"][k]
        item["correctness"][c_idx][k]      = test_if_eq(res, exp_out)

    return data




def main():
    cfg          = get_config()
    project_name = cfg.experiment.project
    outputs_name = "eval-" + cfg.model.replace("/", ".") + "-" + cfg.dataset.eval_dataset

    num_node = cfg.experiment.num_node
    node_index = cfg.experiment.node_index

    if num_node > 1:
        file_name    = f"../{project_name}/temp_data/outputs-{node_index}-{outputs_name}.json"
    else:
        file_name    = f"../{project_name}/temp_data/outputs-{outputs_name}.json"

    with open(file_name, 'r') as f:
        data = json.load(f)

    func_items  = [itm for itm in data if itm.get("test_method","function") == "function"]
    stdio_items = [itm for itm in data if itm.get("test_method") == "stdio"]

    # --- 1) function ---
    if func_items:
        updated_func = evaluate_function_dataset(func_items, n_workers=cfg.execute.num_chunk)
        
        func_iter = iter(updated_func)
        for i,it in enumerate(data):
            if it.get("test_method","function") == "function":
                data[i] = next(func_iter)


    # --- 2) stdio ---
    if stdio_items:
        total_scripts = sum(len(it["extracted_output"]) for it in stdio_items)
        num_chunks    = max(1, math.ceil(total_scripts / cfg.execute.num_chunk))
        updated_stdio = evaluate_stdio_dataset(stdio_items, num_chunks=num_chunks)
        it_stdio = iter(updated_stdio)
        for i, it in enumerate(data):
            if it.get("test_method") == "stdio":
                data[i] = next(it_stdio)
    
    

    # --- svae JSON ---
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding="utf-8", errors="surrogatepass") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    

    

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  
    main()
