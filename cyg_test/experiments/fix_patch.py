import argparse
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal_send_recv",action='store_true')
    args = parser.parse_args()
    if args.signal_send_recv:
        shutil.copyfile("cyg_test/experiments/patch_file/signal_global_env.py","alpa/global_env.py")
    else:
        shutil.copyfile("cyg_test/experiments/patch_file/global_env.py","alpa/global_env.py")