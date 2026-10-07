import os
import argparse
import pandas as pd
import ray
import torch
from PIL import Image
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

from lmdeploy import (
    pipeline,
    TurbomindEngineConfig,
    ChatTemplateConfig,
    GenerationConfig
)
from lmdeploy.vl import load_image

system_message = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
"""


# ================= 配置区域 =================
LINGOQA_TEST_PATH = "/path/to/val.parquet"
IMAGE_ROOT = "/path/to/LingoQA/Evaluation/images/val"
MODEL_PATH = "/path/to/Recogdrive_DriveLM/"

def parse_args():
    parser = argparse.ArgumentParser(description="LingoQA Inference with InternVL and Ray")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--parquet_path", type=str, default=LINGOQA_TEST_PATH)
    parser.add_argument("--image_root", type=str, default=IMAGE_ROOT)
    parser.add_argument("--output_path", type=str, default="./predictions.csv")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()

def load_single_image(path):
    return load_image(path)


def load_aligned_images_fast(image_root: str, segment_id: str, num_frames: int = 5) -> List[Image.Image]:
    """多线程读取图片"""
    segment_path = os.path.join(image_root, segment_id)
    image_paths = []
    
    for i in range(num_frames):
        image_paths.append(os.path.join(segment_path, f"{i}.jpg"))

    with ThreadPoolExecutor(max_workers=num_frames) as executor:
        images = list(executor.map(load_single_image, image_paths))
    return images

class LingoQAPredictor:
    def __init__(self, model_path: str, image_root: str):
        self.image_root = image_root
        pid = os.getpid()
        print(f"[Init] Worker initialized on PID {pid}.")
        
        self.pipe = pipeline(
            model_path,
            backend_config=TurbomindEngineConfig(session_len=32000, tp=1),
            chat_template_config=ChatTemplateConfig(model_name='internvl2_5', meta_instruction=system_message)
        )

        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            min_new_tokens=1,
            do_sample=True,
            temperature=0.01,
            top_p=0.001,
            top_k=1
        )

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        segment_ids = batch['segment_id']
        questions = batch['question']
        
        batch_prompts = []
        valid_indices = []
        
        for i in range(len(questions)):
            seg_id = segment_ids[i]
            images = load_aligned_images_fast(self.image_root, seg_id, num_frames=5)
            
            image_prefix = "".join([f"<image>\n" for _ in range(len(images))])
            prompt_text = f"{image_prefix}{questions[i]}\nAnswer the question in a single short sentence." 

            batch_prompts.append((prompt_text, images))
            valid_indices.append(i)
            
        if not batch_prompts:
            batch['answer'] = [""] * len(questions)
            return batch

        responses = self.pipe(batch_prompts, gen_config=self.generation_config)
        
        answers = [""] * len(questions)
        for idx, resp in zip(valid_indices, responses):
            answers[idx] = resp.text

        batch['answer'] = answers
        return batch

def main():
    args = parse_args()
    
    if not ray.is_initialized():
        ray.init()

    print(f"Loading metadata from {args.parquet_path}...")
    pd_df = pd.read_parquet(args.parquet_path)
    # 确保读取了 question 列
    pd_df = pd_df[['question_id', 'segment_id', 'question']]
    
    ds = ray.data.from_pandas(pd_df)
    
    # 重分区以利用所有 GPU
    num_blocks = args.num_gpus * 4
    ds = ds.repartition(num_blocks)

    print(f"Starting distributed inference on {args.num_gpus} GPUs...")
    ds = ds.map_batches(
        LingoQAPredictor,
        fn_constructor_args=(args.model_path, args.image_root),
        num_gpus=1,
        concurrency=args.num_gpus,
        batch_size=args.batch_size
    )

    print("Executing pipeline...")
    ds = ds.materialize()
    
    print("Collecting results...")
    final_pd = ds.to_pandas()
    
    final_pd = final_pd[['question_id', 'segment_id', 'question', 'answer']]
    # =========================================
    
    print(f"Saving {len(final_pd)} predictions to {args.output_path}...")
    final_pd.to_csv(args.output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()