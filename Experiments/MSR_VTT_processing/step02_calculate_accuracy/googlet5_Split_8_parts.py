import torch
import pandas as pd
import json
import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import os
import numpy as np
from torch.cuda.amp import autocast
import faiss
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 

@dataclass
class SimilarityResult:
    video_files: List[str]
    similarity_scores: List[float]
    generated_texts: List[str]
    query_time: float
    match_positions: List[int]

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = min(512, max(len(self.tokenizer.encode(text)) for text in texts))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        text = f"encode text: {text}"
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}

class MultiPartT5Analyzer:
    def __init__(self, model_name: str = 'google-t5/t5-base', batch_size: int = 16):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            mirror='https://hf-mirror.com'
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            mirror='https://hf-mirror.com'
        )
        self.model.to(self.device)
        self.batch_size = batch_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging()

    def setup_logging(self):
        log_file = f't5_multipart_analysis_{self.timestamp}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_embeddings(self, model_output):
        last_hidden_state = model_output.last_hidden_state
        embeddings = torch.mean(last_hidden_state, dim=1)
        return embeddings

    @torch.no_grad()
    def encode_texts(self, texts: List[str], desc: str = "Encoding") -> torch.Tensor:
        dataset = TextDataset(texts, self.tokenizer)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2
        )
        embeddings = []
        self.model.eval()
        for batch in tqdm(dataloader, desc=desc):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast('cuda'):
                outputs = self.model.encoder(**batch, return_dict=True)
                batch_embeddings = self.get_embeddings(outputs).detach()
                embeddings.append(batch_embeddings.cpu())
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        final_embeddings = torch.cat(embeddings, dim=0)
        return final_embeddings

    def split_datasets(self, df: pd.DataFrame, video_text_map: Dict[str, str], num_parts: int = 2, random_state: int = 42) -> Tuple[List[Dict], List[pd.DataFrame]]:
        print(f"\nOriginal Data Statistics:")
        print(f"Total captions: {len(df)}")
        print(f"Total unique video_ids: {len(df['video_id'].unique())}")
        print(f"Total video texts: {len(video_text_map)}")
        video_ids = df['video_id'].unique()
        test_size = 1.0 / num_parts
        part_dfs = []
        part_video_text_maps = []
        remaining_video_ids = video_ids
        for i in range(num_parts - 1):
            current_test_size = 1.0 / (num_parts - i)
            part_video_ids, remaining_video_ids = train_test_split(
                remaining_video_ids,
                test_size=(1 - current_test_size),
                random_state=random_state + i
            )
            part_df = df[df['video_id'].isin(part_video_ids)]
            part_video_text_map = {
                k: v for k, v in video_text_map.items() 
                if any(vid in k for vid in part_video_ids)
            }
            part_dfs.append(part_df)
            part_video_text_maps.append(part_video_text_map)
            print(f"\nPart {i+1} Statistics:")
            print(f"Captions: {len(part_df)}")
            print(f"Unique videos: {len(part_video_ids)}")
            print(f"Video texts: {len(part_video_text_map)}")
        last_part_df = df[df['video_id'].isin(remaining_video_ids)]
        last_part_video_text_map = {
            k: v for k, v in video_text_map.items() 
            if any(vid in k for vid in remaining_video_ids)
        }
        part_dfs.append(last_part_df)
        part_video_text_maps.append(last_part_video_text_map)
        print(f"\nPart {num_parts} Statistics:")
        print(f"Captions: {len(last_part_df)}")
        print(f"Unique videos: {len(remaining_video_ids)}")
        print(f"Video texts: {len(last_part_video_text_map)}")
        total_queries = sum(len(part_df) for part_df in part_dfs)
        print(f"\nVerification:")
        print(f"Original total: {len(df)}")
        print(f"Split total: {total_queries}")
        if total_queries != len(df):
            raise ValueError(f"Query count mismatch! Original: {len(df)}, After split: {total_queries}")
        print("\nDetailed Statistics:")
        for i, (part_df, part_map) in enumerate(zip(part_dfs, part_video_text_maps)):
            print(f"\nPart {i+1}:")
            print(f"  Total Queries: {len(part_df)}")
            print(f"  Unique Videos: {len(part_df['video_id'].unique())}")
            print(f"  Text Coverage: {len(part_map)}")
            print(f"  Queries per Video: {len(part_df) / len(part_df['video_id'].unique()):.2f}")
        return part_video_text_maps, part_dfs

    # Continuation of the class...

def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['OMP_NUM_THREADS'] = '4'
    torch.backends.cudnn.benchmark = True
    data_dir = "/root/autodl-tmp/dataprocessing/prepared_data"
    np.random.seed(42)
    torch.manual_seed(42)
    num_parts = 8
    analyzer = MultiPartT5Analyzer(batch_size=24)
    comparison_dir = analyzer.analyze_queries(data_dir, num_parts=num_parts)
    print(f"\nAnalysis results saved in: {comparison_dir}")

if __name__ == "__main__":
    main()