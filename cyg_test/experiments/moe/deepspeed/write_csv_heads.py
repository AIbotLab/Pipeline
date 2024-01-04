from cyg_test.experiments.experiment_util import write_csv_row
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_name",type=str,default="none")
    args = parser.parse_args()
    heads = [
            "Type", "Method", "enable_overlapping", "seq_len", "hidden_size", "num_layers", "num_heads", "vocab_size", "num_expert", "expert_group_size",
            "GPU", "BatchSize", "Microbatch", 
            "prefer_reduce_scatter", "use_remat", "dp_size", "tensor_mp_size", "pipeline_mp_size", "ep_size" "force_batch_dim_mapping",
            "#Params (Billion)", "TFLOPs", "Peak Mem (GB)",
            "Time1 (s)", "Time2 (s)", "Time3 (s)", "Time4 (s)", "Time5 (s)", "Time6 (s)", "Time7 (s)", "Time8 (s)", "Time9 (s)", "Time10 (s)", 
        ]
    write_csv_row(args.output_name,heads)
