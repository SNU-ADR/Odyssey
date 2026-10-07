import os
import json
import re
from PIL import Image
from tqdm import tqdm
import torch
import torch.distributed as dist
from transformers import AutoModel, AutoTokenizer
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig, GenerationConfig
from lmdeploy.vl import load_image

MODEL_PATH = '/path/to/Rcogdrive_DriveLM'  
JSON_INPUT_PATH = './v1_1_val_nus_q_only.json' 
JSON_OUTPUT_PATH = './output.json'  
IMAGE_ROOT = '/path/tp/nuscenes' 
BATCH_SIZE = 8

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


def get_image_sizes(image_paths, root_dir):
    """Loads images to get their original sizes."""
    cam_order = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
    image_sizes = {}
    for cam in cam_order:
        if cam in image_paths:
            img_path = os.path.join(root_dir, image_paths[cam])
            try:
                with Image.open(img_path) as img:
                    image_sizes[cam] = img.size
            except FileNotFoundError:
                if dist.get_rank() == 0:
                    print(f"Warning: Image not found at {img_path}. Cannot get size.")
    return image_sizes

def process_question(question, image_sizes):
    """Processes the question to include bounding box coordinates and formats the multi-image prompt."""
    cam_mapping = {
        'CAM_FRONT': 'FRONT VIEW', 'CAM_FRONT_LEFT': 'FRONT LEFT VIEW',
        'CAM_FRONT_RIGHT': 'FRONT RIGHT VIEW', 'CAM_BACK': 'BACK VIEW',
        'CAM_BACK_LEFT': 'BACK LEFT VIEW', 'CAM_BACK_RIGHT': 'BACK RIGHT VIEW'
    }

    prompt_header = "The following images are captured simultaneously from different cameras mounted on the same ego vehicle:\n"
    image_prompts = []
    cam_order = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
    for cam in cam_order:
        view_name = cam_mapping.get(cam)
        if view_name:
            image_prompts.append(f"<{view_name}>:\n<image>")

    final_prompt = prompt_header + "\n".join(image_prompts) + f"\n{question}"
    return final_prompt

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}'

    # 2. Load and flatten JSON data
    if rank == 0:
        print("Loading and preprocessing JSON data...")
    with open(JSON_INPUT_PATH, 'r') as f:
        data = json.load(f)

    flat_qa_list = []
    for scene_id, scene_data in data.items():
        for frame_id, frame_data in scene_data.get('key_frames', {}).items():
            question_idx = 0
            image_paths = frame_data.get('image_paths', {})
            for section_name, qa_list in frame_data.get('QA', {}).items():
                for qa_pair in qa_list:
                    unique_id = f"{scene_id}_{frame_id}_{question_idx}"
                    flat_qa_list.append({
                        "id": unique_id,
                        "question": qa_pair['Q'],
                        "image_paths": image_paths
                    })
                    question_idx += 1

    data_chunk = flat_qa_list[rank::world_size]
    print(f"Rank {rank}/{world_size} started on {device}, processing {len(data_chunk)} items.")

    backend_config = TurbomindEngineConfig(session_len=30000)
    pipe = pipeline(MODEL_PATH, backend_config=backend_config, chat_template_config=ChatTemplateConfig(model_name='internvl2_5', meta_instruction=system_message))

    generation_config = GenerationConfig(
                max_new_tokens=4096,
                min_new_tokens=50,
                do_sample=True,
                temperature=0.01,
                top_p=0.001,
                top_k=1
        )
    # 5. Execute batch inference on each rank's data chunk
    results = []
    cam_order = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
    
    for i in tqdm(range(0, len(data_chunk), BATCH_SIZE), desc=f"GPU {rank}"):
        batch_items = data_chunk[i:i + BATCH_SIZE]
        
        batch_prompts = []
        batch_ids = []
        
        for item in batch_items:
            item_id, raw_question, image_paths = item['id'], item['question'], item['image_paths']
            
            try:
                images_for_sample = []
                for cam in cam_order:
                    if cam in image_paths:
                        img_path = os.path.join(IMAGE_ROOT, image_paths[cam])
                        images_for_sample.append(load_image(img_path))
                
                if not images_for_sample:
                    continue

                image_sizes = get_image_sizes(image_paths, IMAGE_ROOT)
                question = process_question(raw_question, image_sizes)
                
                batch_ids.append(item_id)

                batch_prompts.append((question, images_for_sample))
                
            except Exception as e:
                print(f"Rank {rank} encountered a preprocessing error on item {item_id}: {e}")
                results.append({"id": item_id, "question": raw_question, "answer": f"Error in preprocessing: {e}"})

        if not batch_prompts:
            continue

        responses = pipe(batch_prompts, gen_config=generation_config)
        for item_id, prompt_tuple, response in zip(batch_ids, batch_prompts, responses):
            results.append({"id": item_id, "question": prompt_tuple[0], "answer": response.text})



    temp_output_path = f"output_part_{rank}.json"
    with open(temp_output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Rank {rank} finished and saved results to {temp_output_path}")

    dist.barrier()

    if rank == 0:
        print("All workers finished. Merging results...")
        all_results = []
        for r in range(world_size):
            temp_file = f"output_part_{r}.json"
            try:
                with open(temp_file, 'r') as f:
                    all_results.extend(json.load(f))
                os.remove(temp_file)
            except FileNotFoundError:
                print(f"Warning: Temporary file {temp_file} not found.")

        all_results.sort(key=lambda x: x['id'])
        
        with open(JSON_OUTPUT_PATH, 'w') as f:
            json.dump(all_results, f, indent=4)
            
        print(f"Successfully processed {len(all_results)} items.")
        print(f"Final output saved to {JSON_OUTPUT_PATH}")

    dist.destroy_process_group()

if __name__ == '__main__':
    main()