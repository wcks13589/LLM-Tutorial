#!/usr/bin/env python3
"""
合併 LoRA 權重到基底模型的腳本

使用範例:
    python merge_lora.py \
        --nemo_lora_model /path/to/lora/checkpoint \
        --output_path /path/to/merged/model
"""

import argparse
from nemo.collections import llm


def main():
    parser = argparse.ArgumentParser(description="合併 LoRA 權重到基底模型")
    parser.add_argument(
        "--nemo_lora_model", 
        type=str, 
        required=True,
        help="LoRA checkpoint 的路徑"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="合併後模型的輸出路徑"
    )
    
    args = parser.parse_args()
    
    print(f"開始合併 LoRA 權重...")
    print(f"LoRA checkpoint 路徑: {args.nemo_lora_model}")
    print(f"輸出路徑: {args.output_path}")
    
    # 執行 LoRA 權重合併
    llm.peft.merge_lora(
        lora_checkpoint_path=args.nemo_lora_model,
        output_path=args.output_path,
    )
    
    print(f"LoRA 權重合併完成！合併後的模型已儲存至: {args.output_path}")


if __name__ == '__main__':
    main() 