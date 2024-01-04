
from functools import partial

import numpy as np
import torch
from megatron import get_args, get_timers, initialize_megatron, mpu
from megatron.model import BertModel, GPTModel
import torch.nn.functional as F

def compute_gpt_parameter_count(num_layers, hidden_size, vocab_size):
    return num_layers * (
        # self-attention
        hidden_size * (3 * hidden_size + 1) + hidden_size * (hidden_size + 1) +
        # mlp
        hidden_size * (4 * hidden_size + 1) + hidden_size * 4 *
        (hidden_size + 1) +
        # layer norm
        hidden_size * 4) + vocab_size * (hidden_size + 1)

def compute_gpt_total_flops(batch_size,
                       seq_len,
                       num_layers,
                       hidden_size,
                       vocab_size,
                       num_gpus,
                       latency,
                       backward=True,
                       checkpoint_activations=False):
    factor = 24
    if backward:
        factor += 48
    if checkpoint_activations:
        factor += 24

    total_flop = factor * batch_size * seq_len * (hidden_size ** 2) * num_layers * \
          (1 + seq_len / (6 * hidden_size)) \
          + 6 * batch_size * seq_len * hidden_size * vocab_size
    # Note: The above formula does not count the first embedding table lookup
    # because it is a sparse operation.
    # If we use dense dot to compute the first embedding table lookup,
    # then the last term in total_flops should be
    # "+ 10 * batch_size * seq_len * hidden_size * vocab_size".
    total_flops = total_flop
    return total_flops


def compute_gpt_bert_statistics(benchmark_case, latencies, num_devices):
    (batch_size, model_config, num_micro_batches, parallel_mode,
     parallel_args) = benchmark_case
    (seq_len, hidden_size, num_layers, num_heads,
     vocab_size) = model_config
    (prefer_reduce_scatter, use_remat, dp, op, pp,
     force_batch_dim_mapping) = parallel_args

    total_flops = compute_gpt_total_flops(batch_size,
                                seq_len,
                                num_layers,
                                hidden_size,
                                vocab_size,
                                num_devices,
                                np.mean(latencies),
                                checkpoint_activations=use_remat)
    parameter_count = compute_gpt_parameter_count(num_layers, hidden_size,
                                                  vocab_size)
    return total_flops, parameter_count

def get_bert_functions():
    args = get_args()
    micro_batch_size = args.micro_batch_size
    seq_len = args.encoder_seq_length

    def model_provider(pre_process=True, post_process=True):
        num_tokentypes = 2 if args.bert_binary_head else 0
        model = BertModel(num_tokentypes=num_tokentypes,
                          add_binary_head=args.bert_binary_head,
                          parallel_output=True,
                          pre_process=pre_process,
                          post_process=post_process)

        return model

    def loss_func(loss_mask, sentence_order, output_tensor):
        lm_loss_, sop_logits = output_tensor

        lm_loss_ = lm_loss_.float()
        loss_mask = loss_mask.float()
        lm_loss = torch.sum(
            lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

        if sop_logits is not None:
            sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(),
                                       sentence_order.view(-1),
                                       ignore_index=-1)
            sop_loss = sop_loss.float()
            loss = lm_loss + sop_loss
            #averaged_losses = average_losses_across_data_parallel_group(
            #    [lm_loss, sop_loss])
            averaged_losses = [0, 0]
            return loss, {
                'lm loss': averaged_losses[0],
                'sop loss': averaged_losses[1]
            }
        else:
            loss = lm_loss
            #averaged_losses = average_losses_across_data_parallel_group(
            #    [lm_loss])
            averaged_losses = [0]
            return loss, {'lm loss': averaged_losses[0]}

    tokens = torch.ones((micro_batch_size, seq_len)).cuda().long()
    padding_mask = \
        torch.ones(micro_batch_size, seq_len).cuda().bool()
    types = torch.ones((micro_batch_size, seq_len)).cuda().long()
    lm_labels = torch.ones((micro_batch_size, seq_len)).cuda().long()
    loss_mask = torch.ones((micro_batch_size, seq_len)).cuda().int()
    sentence_order = None

    def forward_step(data_iterator, model):
        if not args.bert_binary_head:
            types = None

        output_tensor = model(tokens,
                              padding_mask,
                              tokentype_ids=types,
                              lm_labels=lm_labels)
        return output_tensor, partial(loss_func, loss_mask, sentence_order)

    return model_provider, loss_func, forward_step

def get_gpt_functions(micro_batch_size,seq_len):
    def model_provider(pre_process=True, post_process=True):
        model = GPTModel(num_tokentypes=0,
                         parallel_output=True,
                         pre_process=pre_process,
                         post_process=post_process)
        return model

    def loss_func(loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

        # Reduce loss for logging.
        #averaged_loss = average_losses_across_data_parallel_group([loss])
        averaged_loss = [0]
        return loss, {'lm loss': averaged_loss[0]}

    tokens = torch.ones((micro_batch_size, seq_len)).cuda().long()
    labels = torch.ones((micro_batch_size, seq_len)).cuda().long()
    loss_mask = torch.ones((micro_batch_size, seq_len)).cuda().int()
    attention_mask = \
        torch.ones(micro_batch_size, 1, seq_len, seq_len).cuda().bool()
    position_ids = torch.ones((micro_batch_size, seq_len)).cuda().long()

    def forward_step(data_iterator, model):
        output_tensor = model(tokens,
                              position_ids,
                              attention_mask,
                              labels=labels)
        return output_tensor, partial(loss_func, loss_mask)

    return model_provider, loss_func, forward_step