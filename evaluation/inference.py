import argparse
import nemo_run as run
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.recipes.precision.mixed_precision import fp16_mixed
from megatron.core.inference.common_inference_params import CommonInferenceParams

def parse_args():
    parser = argparse.ArgumentParser(description='Configure and run NeMo inference.')
    parser.add_argument('--peft_ckpt_path', type=str, required=True,
                        help='Path to the PEFT checkpoint.')
    parser.add_argument('--input_dataset', type=str, required=True,
                        help='Path to the test dataset.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the output predictions.')
    return parser.parse_args()

def create_trainer() -> run.Config[nl.Trainer]:
    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=1
    )
    return run.Config(
        nl.Trainer,
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        strategy=strategy,
        plugins=fp16_mixed(),
    )

def configure_inference(args):
    return run.Partial(
        llm.generate,
        path=args.peft_ckpt_path,
        trainer=create_trainer(),
        input_dataset=args.input_dataset,
        max_batch_size=1,
        inference_params=CommonInferenceParams(num_tokens_to_generate=20, top_k=1),
        output_path=args.output_path,
    )

def create_local_executor(nodes: int = 1, devices: int = 1) -> run.LocalExecutor:
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
    }
    return run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

if __name__ == '__main__':
    args = parse_args()
    run.run(configure_inference(args), executor=create_local_executor())
