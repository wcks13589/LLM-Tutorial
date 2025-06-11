import argparse
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="下載預訓練資料集")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="要下載的資料集名稱")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="輸出資料集的路徑")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # 下載中文維基新聞資料集
    dataset = load_dataset(args.dataset_name)['train']
    dataset.to_json(args.output_dir, force_ascii=False)

if __name__ == "__main__":
    main()