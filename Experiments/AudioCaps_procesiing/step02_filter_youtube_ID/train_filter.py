import pandas as pd
import os

def filter_train_data():
    # file path
    train_path = "/root/autodl-tmp/AudioCaps/train.csv"
    matched_results_path = "/root/autodl-tmp/VideoLLama2-audio/VideoLLaMA2/audio_descriptions/matched_results_filtered.csv"
    output_dir = "/root/autodl-tmp/AudioCaps"
    
    # loading data file
    print("loading data file...")
    train_df = pd.read_csv(train_path)
    matched_results_df = pd.read_csv(matched_results_path)
    
    # get common youtube_id
    common_ids = set(train_df['youtube_id']).intersection(set(matched_results_df['youtube_id']))
    
    # print information
    print(f"Train.csv origin line: {len(train_df)}")
    print(f"Matched Results line: {len(matched_results_df)}")
    print(f"common youtube_id: {len(common_ids)}")
    print(f"Remove line: {len(train_df) - len(train_df[train_df['youtube_id'].isin(common_ids)])}")
    
    filtered_train_df = train_df[train_df['youtube_id'].isin(common_ids)]
    
    # save after filter
    output_path = os.path.join(output_dir, 'train_filtered.csv')
    filtered_train_df.to_csv(output_path, index=False)
    
    print(f"\output path: {output_path}")
    print(f"final line: {len(filtered_train_df)}")
    

if __name__ == "__main__":
    filter_train_data()
