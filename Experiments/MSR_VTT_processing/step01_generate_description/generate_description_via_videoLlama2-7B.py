import os
import sys
from tqdm import tqdm
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

# Disable torch initialization for better efficiency
disable_torch_init()

# Configure paths
dataset_path = "/root/autodl-tmp/dataset/train-video"
output_file_path = "video_llama_output_2.txt"
model_path = '/root/autodl-tmp/VideoLLaMA2-7B'

# Initialize the model
model, processor, tokenizer = model_init(model_path)

def process_video(video_path, model, processor, tokenizer):
    """
    Use the VideoLlama2 model to infer and generate a textual description of the video.
    """
    instruct = ("Describe what is happening in this video in a single, simple sentence of no more "
                "than 15 words. Focus only on the main action.")  # Instruction for generating the description
    output = mm_infer(
        processor['video'](video_path),  # Process video input
        instruct, 
        model=model, 
        tokenizer=tokenizer, 
        do_sample=False, 
        modal='video'
    )
    return output

# Count the total number of videos
total_videos = sum(1 for root, _, files in os.walk(dataset_path) 
                  for file in files if file.endswith(".mp4"))
print(f"Found {total_videos} video files to process.")

# Process video files and record the results
video_count = 0
max_videos = 9999999999

# Create a progress bar
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
                
                # Write the file name and generated text to the output file
                output_file.write(f"Video File: {file}\nGenerated Text: {generated_text}\n\n")
                output_file.flush()  # Ensure immediate writing to file
                
                # Update the progress bar
                pbar.set_postfix({'Current': file}, refresh=True)
                pbar.update(1)
                
                video_count += 1
                
        if video_count >= max_videos:
            break

pbar.close()
print("\nVideo processing completed. Results have been saved to video_llama_output_2.txt.")