import os
import sys
from tqdm import tqdm
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

disable_torch_init()

dataset_path = "/root/autodl-tmp/dataset/train-video"
output_file_path = "video_llama_output_2.txt"
model_path = '/root/autodl-tmp/VideoLLaMA2-7B'

model, processor, tokenizer = model_init(model_path)

def process_video(video_path, model, processor, tokenizer):
    instruct = "Describe what is happening in this video in a single, simple sentence of no more than 15 words. Focus only on the main action."
    output = mm_infer(
        processor['video'](video_path),
        instruct, 
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        modal='video'
    )
    return output

total_videos = sum(1 for root, _, files in os.walk(dataset_path) 
                  for file in files if file.endswith(".mp4"))
print(f"Found {total_videos} video files to process.")

video_count = 0
max_videos = 9999999999

pbar = tqdm(total=min(total_videos, max_videos),
            desc="Processing videos",
            unit="video",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

with open(output_file_path, "w") as output_file:
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".mp4"):
                if video_count >= max_videos:
                    break
                    
                video_path = os.path.join(root, file)
                generated_text = process_video(video_path, model, processor, tokenizer)
                
                output_file.write(f"Video File: {file}\nGenerated Text: {generated_text}\n\n")
                output_file.flush()
                
                pbar.set_postfix({'Current': file}, refresh=True)
                pbar.update(1)
                
                video_count += 1
                
            if video_count >= max_videos:
                break

pbar.close()
print("\nVideo processing completed, results saved to video_llama_output_2.txt file.")