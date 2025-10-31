import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="test", help="run name")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda device to use")
    parser.add_argument("--split_file", type=str, help="Path to split json")
    parser.add_argument("--data_root", type=str, help="Path to dataset")
    parser.add_argument("--data_keys", type=str, help="Dataset specifications")

    parser.add_argument("--task", type=str, help="Unused")
    parser.add_argument("--num_threads", type=int, help="Unused")
    parser.add_argument("--num_epoches", type=int, help="Unused")

    return parser