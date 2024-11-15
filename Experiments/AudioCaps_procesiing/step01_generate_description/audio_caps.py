import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
import argparse
import os
from tqdm import tqdm
import json
from pathlib import Path
import torch

def process_audio_files(args):
   model, processor, tokenizer = model_init(args.model_path)
   model.model.vision_tower = None
   
   output_dir = Path(args.output_dir)
   output_dir.mkdir(parents=True, exist_ok=True)
   
   audio_files = sorted(Path(args.input_dir).glob('*.wav'))
   audio_processor = processor['audio']
   
   output_csv = output_dir / "results.csv"
   with open(output_csv, "w", encoding="utf-8") as f:
       f.write("filename,description\n")
   
   for audio_file in tqdm(audio_files):
       try:
           audio_tensor = audio_processor(str(audio_file))
           question = "Please describe the audio content briefly and concisely in one short sentence."
           output = mm_infer(
               audio_tensor,
               question,
               model=model,
               tokenizer=tokenizer,
               modal='audio',
               do_sample=False,
           )
           
           description = output.strip().replace('"', '""') 
           
           with open(output_csv, "a", encoding="utf-8") as f:
               f.write(f'"{audio_file.name}","{description}"\n')
               
           print(f"Processed: {audio_file.name}")
           
       except Exception as e:
           print(f"Error processing {audio_file.name}: {str(e)}")
           with open(output_csv, "a", encoding="utf-8") as f:
               f.write(f'"{audio_file.name}","ERROR: {str(e)}"\n')
   
   print(f"\nResults saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', 
                       default='/root/autodl-tmp/VideoLLaMA2.1-7B-AV',
                       help='Path to the model')
    parser.add_argument('--input-dir', 
                       default='/root/autodl-tmp/AudioCaps/Amphion___AudioCaps/mnt/data2/wangyuancheng/TTA/processed_data/AudioCaps/wav',
                       help='Input directory containing audio files')
    parser.add_argument('--output-dir', 
                       default='audio_descriptions',
                       help='Output directory for results')
    
    args = parser.parse_args()
    process_audio_files(args)