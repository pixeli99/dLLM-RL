import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed



from models import DreamTokenizer, DreamModel
from train.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

SYSTEM_PROMPT_LEN = 28

from train.utils import get_config, flatten_omega_conf, AverageMeter, maybe_add_special_tokens

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")







class TrainDataset(Dataset):
    def __init__(self, inputs, labels, pmasks):
        self.inputs = inputs
        self.labels = labels
        self.pmasks = pmasks

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            self.inputs[idx],
            self.labels[idx],
            self.pmasks[idx]
        )


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    project_name = config.experiment.project
    pretrained_model = config.model.pretrained_model

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.project) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.project,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint", None)

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.project, exist_ok=True)
        config_path = Path(config.experiment.project) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    tokenizer = DreamTokenizer.from_pretrained(pretrained_model)
    uni_prompting = UniversalPrompting(tokenizer, max_prompt_len=config.training.max_prompt_len,
                                       max_gen_length=config.training.max_gen_length,
                                       ignore_id=-100)


    model = DreamModel.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16)

    # Add custom special tokens if provided and resize embeddings
    try:
        added_n = maybe_add_special_tokens(tokenizer, model, config)
        if added_n > 0:
            logger.info(f"Added {added_n} special tokens; resized embeddings to {len(tokenizer)}")
    except Exception as e:
        logger.warning(f"Failed to add special tokens: {e}")

    

    if config.training.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    else:
        model = model.to(accelerator.device)

    mask_id = model.config.mask_token_id
    pad_id = model.config.pad_token_id


    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")




    def collapse_k_unique(lst, k: int):
        if k <= 0:
            raise ValueError("k must be > 0")
        uniq = sorted(set(lst))

        mapping = {}
        n = len(uniq)
        for idx, val in enumerate(uniq):
            group = idx // k
            end_idx = min((group + 1) * k - 1, n - 1)
            rep = uniq[end_idx]
            mapping[val] = rep
        return [mapping[x] for x in lst]
    
    


    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    
    @torch.no_grad()
    def parse_structure_blocks(input_ids, tokenizer):
        """Parse think blocks and summary block from input_ids"""
        # Get special token ids
        think_open_ids = []
        think_close_ids = []
        for i in range(1, 7):  # think 1-6
            open_token = f"<think {i}>"
            close_token = f"</think {i}>"
            if open_token in tokenizer.get_added_vocab():
                think_open_ids.append(tokenizer.convert_tokens_to_ids(open_token))
                think_close_ids.append(tokenizer.convert_tokens_to_ids(close_token))

        summary_open_id = tokenizer.convert_tokens_to_ids("<summary>") if "<summary>" in tokenizer.get_added_vocab() else None
        summary_close_id = tokenizer.convert_tokens_to_ids("</summary>") if "</summary>" in tokenizer.get_added_vocab() else None

        B, L = input_ids.shape
        structure_info = []

        for b in range(B):
            seq = input_ids[b].tolist()
            blocks = {"think_blocks": [], "summary_block": None}

            # Find think blocks
            for open_id, close_id in zip(think_open_ids, think_close_ids):
                try:
                    start_idx = seq.index(open_id)
                    end_idx = seq.index(close_id, start_idx) + 1  # Include the closing tag
                    blocks["think_blocks"].append((start_idx, end_idx))
                except ValueError:
                    continue  # This think block not found

            # Find summary block
            if summary_open_id and summary_close_id:
                try:
                    start_idx = seq.index(summary_open_id)
                    end_idx = seq.index(summary_close_id, start_idx) + 1
                    blocks["summary_block"] = (start_idx, end_idx)
                except ValueError:
                    pass  # Summary not found

            structure_info.append(blocks)

        return structure_info

    def block_level_mask(input_ids, tokenizer, start_pos, lower_p=0.1, upper_p=0.9, mask_id=mask_id):
        """Apply block-level masking to structured think data"""
        B, L = input_ids.shape
        device = input_ids.device

        # Parse structure
        structure_info = parse_structure_blocks(input_ids, tokenizer)

        noisy_list, label_list, pmask_list = [], [], []

        for b in range(B):
            blocks = structure_info[b]
            base_ids = input_ids[b]

            # If no structure found, fallback to random masking
            if not blocks["think_blocks"] and not blocks["summary_block"]:
                # Simple random masking fallback
                p = torch.empty(L, device=device).uniform_(lower_p, upper_p)
                mask_pos = (torch.rand(L, device=device) < p)
                mask_pos[:start_pos] = False  # Don't mask prompt

                noisy_ids = base_ids.clone()
                noisy_ids[mask_pos] = mask_id

                noisy_list.append(noisy_ids)
                label_list.append(base_ids)
                pmask_list.append(mask_pos)
                continue

            # Strategy 1: Forward thinking (mask summary based on visible thinks)
            # Strategy 2: Backward reasoning (keep summary, mask some thinks)
            # Strategy 3: Partial paths (mask some complete think blocks)
            strategies = ["forward", "backward", "partial", "mixed"]
            weights = [0.35, 0.25, 0.25, 0.15]

            for _ in range(3):  # Generate multiple masks per sample
                strategy = torch.tensor(weights, device=device).multinomial(1).item()
                strategy = strategies[strategy]

                mask_pos = torch.zeros(L, dtype=torch.bool, device=device)

                if strategy == "forward":
                    # Mask summary with high prob, keep most thinks
                    for start, end in blocks["think_blocks"]:
                        if torch.rand(1, device=device) < 0.3:  # 30% chance to mask a think block
                            mask_pos[start:end] = True

                    if blocks["summary_block"]:
                        start, end = blocks["summary_block"]
                        if torch.rand(1, device=device) < 0.8:  # 80% chance to mask summary
                            mask_pos[start:end] = True

                elif strategy == "backward":
                    # Keep summary, mask most thinks
                    for start, end in blocks["think_blocks"]:
                        if torch.rand(1, device=device) < 0.7:  # 70% chance to mask a think block
                            mask_pos[start:end] = True

                    # Keep summary visible (low mask prob)
                    if blocks["summary_block"]:
                        start, end = blocks["summary_block"]
                        if torch.rand(1, device=device) < 0.2:  # Only 20% chance to mask
                            mask_pos[start:end] = True

                elif strategy == "partial":
                    # Randomly mask complete think blocks
                    num_blocks = len(blocks["think_blocks"])
                    if num_blocks > 0:
                        # Decide how many blocks to mask
                        num_to_mask = torch.randint(1, max(2, num_blocks), (1,), device=device).item()
                        indices = torch.randperm(num_blocks, device=device)[:num_to_mask]

                        for idx in indices:
                            start, end = blocks["think_blocks"][idx]
                            mask_pos[start:end] = True

                    # Summary: 50/50 chance
                    if blocks["summary_block"]:
                        start, end = blocks["summary_block"]
                        if torch.rand(1, device=device) < 0.5:
                            mask_pos[start:end] = True

                else:  # mixed
                    # Apply partial random masking within blocks
                    for start, end in blocks["think_blocks"]:
                        block_mask_prob = torch.rand(1, device=device).item() * 0.7 + 0.1  # 0.1-0.8
                        block_mask = torch.rand(end - start, device=device) < block_mask_prob
                        mask_pos[start:end] = block_mask

                    if blocks["summary_block"]:
                        start, end = blocks["summary_block"]
                        summary_mask_prob = torch.rand(1, device=device).item() * 0.6 + 0.2  # 0.2-0.8
                        summary_mask = torch.rand(end - start, device=device) < summary_mask_prob
                        mask_pos[start:end] = summary_mask

                # Never mask the prompt
                mask_pos[:start_pos] = False

                # Skip if no masking at all
                if not mask_pos.any():
                    continue

                noisy_ids = base_ids.clone()
                noisy_ids[mask_pos] = mask_id

                noisy_list.append(noisy_ids)
                label_list.append(base_ids)
                pmask_list.append(mask_pos)

        if len(noisy_list) == 0:
            # Fallback if no valid masks generated
            return input_ids, input_ids, torch.ones_like(input_ids, dtype=torch.bool)

        noisy_batch = torch.stack(noisy_list)
        labels_lm = torch.stack(label_list)
        p_mask = torch.stack(pmask_list)

        return noisy_batch, labels_lm, p_mask

    def prepare_inputs_and_labels_for_text(
        prompt, response, step_map, eps=1e-3, mask_id=mask_id
    ):
        input_ids_lm, labels_lm, start_pos, drop_num = uni_prompting((prompt, response))

        B, L = input_ids_lm.shape
        max_gen_len = config.training.max_gen_length
        if max_gen_len + start_pos < L:
            L_after = start_pos + max_gen_len
        else:
            L_after = L
        input_ids_lm = input_ids_lm[:, :L_after]
        labels_lm = labels_lm[:, :L_after]


        lower = config.training.lower_p
        upper = config.training.upper_p
        



        if config.training.method == "semi-ar":
            # Check if we should use block-level masking for structured data
            use_block_mask = config.training.get("use_block_mask", False)

            if use_block_mask:
                # Use the new block-level masking for structured think data
                noisy_batch, labels_lm, p_mask = block_level_mask(
                    input_ids_lm, tokenizer, start_pos,
                    lower_p=lower, upper_p=upper, mask_id=mask_id
                )
            else:
                # Original semi-AR method
                noisy_list, label_list, pmask_list = [], [], []

                device = input_ids_lm.device
                B, L   = input_ids_lm.shape


                for b in range(B):
                    # 1) transform step_map
                    order_list = list(step_map[b])
                    order_list = collapse_k_unique(order_list, config.training.block_size)
                    order = torch.as_tensor(order_list, device=device)
                    order_full = torch.full((L_after,), -1, device=device)
                    order_full[start_pos:] = order[: L_after - start_pos]

                    uniq_steps = torch.unique(order_full[start_pos:], sorted=True)

                    base_ids = input_ids_lm[b]  # (L,)

                    if config.training.post_num is not None:
                        pad_mask_b = (base_ids == pad_id)
                        pad_mask_b[:start_pos] = False
                        keep_first_pad_b = pad_mask_b & (torch.cumsum(pad_mask_b.int(), dim=0) <= config.training.post_num)
                        tail_pad_b       = pad_mask_b & ~keep_first_pad_b
                    else:
                        keep_first_pad_b = torch.zeros(L, dtype=torch.bool, device=device)
                        tail_pad_b       = torch.zeros(L, dtype=torch.bool, device=device)


                    for i in range(0, len(uniq_steps)):

                        block_mask = (order_full == uniq_steps[i])
                        p = torch.empty(L, device=device).uniform_(lower, upper)
                        block_mask = (torch.rand(L, device=device) < p) & block_mask

                        noisy_ids = base_ids.clone()
                        mask_pos  = (order_full > uniq_steps[i]) | block_mask
                        noisy_ids[mask_pos] = mask_id

                        pmask_this = block_mask & ~tail_pad_b

                        if not pmask_this.any():
                            continue

                        noisy_list.append(noisy_ids)
                        label_list.append(labels_lm[b])
                        pmask_list.append(pmask_this)

                    del order, order_full, uniq_steps

                noisy_batch = torch.stack(noisy_list)
                labels_lm   = torch.stack(label_list)
                p_mask      = torch.stack(pmask_list)
        

            
        
        elif config.training.method == "ar":
            
            noisy_batch = input_ids_lm
            labels_lm   = input_ids_lm
            p_mask = torch.zeros_like(input_ids_lm, dtype=torch.bool)
            p_mask[:, start_pos:] = True
        
            if config.training.post_num is not None:

                pad_mask = (input_ids_lm == pad_id)
                pad_mask[:, :start_pos] = False
                keep_first_pad = pad_mask & (torch.cumsum(pad_mask.int(), dim=1) <= config.training.post_num)
                p_mask   = p_mask & (~pad_mask | keep_first_pad)
            
        


        elif config.training.method == "random_masking":
            m = config.training.mask_times_per_sample
            B, L = input_ids_lm.shape
            device = input_ids_lm.device

            noisy_list, label_list, pmask_list = [], [], []
            for b in range(B):
                base_ids  = input_ids_lm[b]
                label_ids = labels_lm[b]

                if config.training.post_num is not None:
                    pad_mask_b = (base_ids == pad_id)
                    pad_mask_b[:start_pos] = False
                    keep_first_pad_b = pad_mask_b & (torch.cumsum(pad_mask_b.int(), dim=0) <= config.training.post_num)
                    tail_pad_b       = pad_mask_b & ~keep_first_pad_b
                else:
                    keep_first_pad_b = torch.zeros(L, dtype=torch.bool, device=device)
                    tail_pad_b       = torch.zeros(L, dtype=torch.bool, device=device)

                for _ in range(m):
                    t = (upper - lower) * torch.rand(1, device=device) + lower
                    rand_mask = torch.rand(L, device=device) < t
                    rand_mask[:start_pos] = False
                    rand_mask = rand_mask & ~tail_pad_b

                    if not rand_mask.any():
                        continue

                    noisy_ids = base_ids.clone()
                    noisy_ids[rand_mask]   = mask_id
                    noisy_ids[tail_pad_b]  = mask_id

                    noisy_list.append(noisy_ids)
                    label_list.append(label_ids)
                    pmask_list.append(rand_mask)

            noisy_batch = torch.stack(noisy_list)    # (B*m, L)
            labels_lm   = torch.stack(label_list)
            p_mask      = torch.stack(pmask_list)
        


        valid_rows = p_mask.any(dim=1)
        noisy_batch = noisy_batch[valid_rows]
        labels_lm   = labels_lm[valid_rows]
        p_mask      = p_mask[valid_rows]
        
        


            
        
        return noisy_batch, labels_lm, p_mask, start_pos, drop_num
        

    def simple_collate(batch):
        inp, lbl, msk = zip(*batch)
        return {
            "input_ids":  torch.stack(inp),
            "labels":     torch.stack(lbl),
            "p_mask_lm":  torch.stack(msk)
        }
    


    
    with open("./data/" + config.dataset.optimization_data + ".json", 'r') as f:
        dataset_load = json.load(f)
    #dataset_load = dataset_load[:2000]
    
    prompt_list = []
    response_list = []
    step_map_list = []
    for x in dataset_load:
        prompt_list.append(x["prompt"])
        response_list.append(x["response"])
        if "step_map" not in x.keys():
            step_map_list.append([j for j in range(config.training.max_gen_length)])
        else:
            step_map_list.append(x["step_map"])
    input_ids, labels, p_mask_lm, start_pos, drop_num = prepare_inputs_and_labels_for_text(prompt_list, response_list, step_map_list)

    


    dataset_lm = TrainDataset(input_ids, labels, p_mask_lm)

    total_batch_size_lm = config.training.batch_size_lm * accelerator.num_processes * config.training.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(dataset_lm) / total_batch_size_lm)
    num_train_epochs = config.training.num_train_epochs
    max_train_steps = num_update_steps_per_epoch * num_train_epochs + 1

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )

    train_dataloader_lm = DataLoader(
        dataset_lm,
        batch_size=config.training.batch_size_lm,
        sampler=None,
        collate_fn=simple_collate,
        num_workers=0
    )





    

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler, train_dataloader_lm = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader_lm
    )

    

    #################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    
    logger.info(f"  Num response = {len(dataset_load)}")
    logger.info(f"  Num sample dropped = {drop_num}")
    logger.info(f"  Num training data = {input_ids.shape[0]}")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    first_epoch = 0
    data_time_m = AverageMeter()
    end = time.time()

    import torch.nn.functional as F



    def make_causal_attention_mask(input_ids, pad_id, start_pos):
        B, T = input_ids.shape
        device = input_ids.device

        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
        ).view(1, 1, T, T)

        bias = torch.zeros((B, 1, T, T), device=device)
        bias.masked_fill_(causal_mask, float("-inf"))

        idx = torch.arange(T, device=device)
        pad_prefix_mask = (input_ids == pad_id) & (idx < start_pos)
        bias.masked_fill_(pad_prefix_mask.view(B, 1, 1, T), float("-inf"))

        return bias 



    def make_attention_mask(input_ids, pad_id, pos):
        B, T = input_ids.shape
        device = input_ids.device
        dtype = input_ids.dtype
        idx = torch.arange(T, device=device)
        keep = ~input_ids.ne(pad_id) & (idx[None, :] <= pos)  # shape (B, T)
        # Allocate bias of shape (B,1,1,T)
        bias = torch.zeros(B, 1, 1, T, device=device)
        bias.masked_fill_(keep[:, None, None, :], float("-inf"))
        return bias
    

    
    def forward_process(input_ids, labels, p_mask_lm, start_pos):
        if config.training.method == "ar":
            #if config.training.batch_size_lm == 1:
            #    logits = model(input_ids, is_causal=True).logits
            attn_mask = make_causal_attention_mask(input_ids, pad_id, start_pos)
            logits = model(input_ids, attention_mask=attn_mask, is_causal=False).logits
        else:
            attention_mask = make_attention_mask(input_ids, pad_id, start_pos)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, is_casual=False).logits
        
        B, T, V = logits.shape

        shift_mask   = p_mask_lm[:, 1:]
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]


        log_probs = F.log_softmax(shift_logits, dim=-1)                             # (B, T-1, V)
        logp_tok  = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)     # (B, T-1)
        loss_lm = - (logp_tok * shift_mask).sum(dim=1)

        mask_num = (shift_mask).sum(dim=1).clamp(min=1)
        loss_lm = loss_lm / mask_num
        loss_lm = loss_lm.sum() / B
        
        return loss_lm


    

    







    from tqdm.auto import tqdm

    global_step = 0
    last_log_time = time.time()
    accum_tokens = 0
    accum_samples = 0

    for epoch in range(first_epoch, num_train_epochs):
        
        model.train()
        
        progress_bar = tqdm(
            train_dataloader_lm,
            desc=f"Epoch {epoch+1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True, 
            leave=True        
        )
        
        

        for step, batch in enumerate(progress_bar, start=1):

            data_time_m.update(time.time() - end)
            
            input_ids = batch["input_ids"].to(accelerator.device)
            labels    = batch["labels"].to(accelerator.device)
            p_mask_lm = batch["p_mask_lm"].to(accelerator.device)

            loss_lm = forward_process(
                    input_ids=input_ids,
                    labels=labels,
                    p_mask_lm=p_mask_lm,
                    start_pos=start_pos
                )
            loss_lm = loss_lm / accelerator.gradient_accumulation_steps
            if step <= 10:
                print(loss_lm)
            accelerator.backward(loss_lm)

            # accumulate tokens and samples for throughput stats
            try:
                accum_tokens += int(p_mask_lm.sum().item())
                accum_samples += int(input_ids.size(0))
            except Exception:
                pass

            if (step + 1) % accelerator.gradient_accumulation_steps == 0:
                if config.training.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                
                # basic metrics logging
                global_step += 1
                now = time.time()
                dt = max(now - last_log_time, 1e-8)
                try:
                    lr = optimizer.param_groups[0]["lr"]
                except Exception:
                    lr = None
                metrics = {
                    "train/loss": float(loss_lm.detach().item()),
                    "train/lr": float(lr) if lr is not None else 0.0,
                    "train/samples": accum_samples,
                    "train/tokens": accum_tokens,
                    "train/sps": accum_samples / dt,
                    "train/tps": accum_tokens / dt,
                    "train/epoch": epoch + 1,
                }
                accelerator.log(metrics, step=global_step)
                last_log_time = now
                accum_tokens = 0
                accum_samples = 0

                optimizer.zero_grad(set_to_none=True)

                del input_ids, labels, p_mask_lm
                torch.cuda.empty_cache()


                


    accelerator.wait_for_everyone()

    # save checkpoint at the end of training
    save_checkpoint(model, tokenizer, config, accelerator, config.model.optimized_name)

    accelerator.end_training()






def save_checkpoint(model, tokenizer, config, accelerator, name):
    output_dir = Path(config.experiment.project)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        ckpts = sorted(
            [d for d in output_dir.iterdir() if d.name.startswith("checkpoint")],
            key=lambda p: int(p.name.split("-")[1]),
        )
        if len(ckpts) >= checkpoints_total_limit:
            to_remove = ckpts[: len(ckpts) - checkpoints_total_limit + 1]
            logger.info(f"removing checkpoints: {', '.join(p.name for p in to_remove)}")
            for p in to_remove:
                shutil.rmtree(p, ignore_errors=True)

    save_base = output_dir / "ckpt"
    save_base.mkdir(exist_ok=True)

    model_to_save = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)

    if accelerator.is_main_process:
        model_to_save.save_pretrained(
            save_base / name,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(str(save_base / name))

        metadata = {
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with (save_base / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model + tokenizer to {save_base / name}")
    















if __name__ == "__main__":
    main()
