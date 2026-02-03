import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
from collections import Counter

class MultiSourceDataset(Dataset):
    def __init__(self, math_data, chat_data):
        self.data = []
        self.sources = []
        
        # Add math data
        for item in math_data:
            self.data.append(item)
            self.sources.append('math')
        
        # Add chat data  
        for item in chat_data:
            self.data.append(item)
            self.sources.append('chat')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.sources[idx]

def load_data():
    """Load your parquet files"""
    math_df = pd.read_parquet('/root/paddlejob/workspace/env_run/output/work/verl/data/simplerl_level3to5/train_clean.parquet')
    chat_df = pd.read_parquet('/root/paddlejob/workspace/env_run/output/work/RLMT/training/ppo_grpo/data/wildchat-if_train.parquet')
    
    # Convert to lists - adjust based on your actual data structure
    math_data = math_df.to_dict('records')
    chat_data = chat_df.to_dict('records')
    
    return math_data, chat_data

def check_batch_proportions(dataloader, num_batches=5):
    """Check the actual proportions in batches"""
    print(f"Checking first {num_batches} batches:")
    print("-" * 50)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
            
        data, sources = batch
        total = len(sources)
        source_counts = Counter(sources)
        proportions = {source: count/total for source, count in source_counts.items()}
        
        print(f"Batch {batch_idx}:")
        print(f"  Total samples: {total}")
        print(f"  Source counts: {dict(source_counts)}")
        print(f"  Proportions: {proportions}")
        
        # Show a few samples from each source in this batch
        math_samples = [data[i] for i in range(len(data)) if sources[i] == 'math']
        chat_samples = [data[i] for i in range(len(data)) if sources[i] == 'chat']
        
        if math_samples:
            print(f"  Math sample preview: {math_samples[0].get('question', 'No question field')[:100]}...")
        if chat_samples:
            print(f"  Chat sample preview: {str(chat_samples[0])[:100]}...")
        print()

# Load your data
print("Loading data...")
math_data, chat_data = load_data()

print(f"Math samples: {len(math_data)}")
print(f"Chat samples: {len(chat_data)}")
print(f"Total samples: {len(math_data) + len(chat_data)}")

# Create dataset
dataset = MultiSourceDataset(math_data, chat_data)

# Define desired proportions
desired_proportions = {
    'math': 0.7,  # 70% math data in each batch
    'chat': 0.3   # 30% chat data in each batch
}

print(f"\nDesired proportions: {desired_proportions}")

# Calculate weights for each sample
weights = []
for i in range(len(dataset)):
    source = dataset.sources[i]
    source_count = dataset.sources.count(source)
    total_samples = len(dataset)
    weight = desired_proportions[source] / (source_count / total_samples)
    weights.append(weight)

weights = torch.DoubleTensor(weights)

# Create sampler
sampler = WeightedRandomSampler(
    weights, 
    num_samples=min(1000, len(dataset)),  # Limit for testing
    replacement=True
)

# Create dataloader
dataloader = DataLoader(
    dataset, 
    batch_size=16,  # Smaller batch size for clearer inspection
    sampler=sampler,
    collate_fn=lambda batch: tuple(zip(*batch))
)

# Check the first 5 batches
check_batch_proportions(dataloader, num_batches=5)

# Also check overall statistics
print("\n" + "="*50)
print("OVERALL STATISTICS:")
print("="*50)
total_math = dataset.sources.count('math')
total_chat = dataset.sources.count('chat')
total_samples = len(dataset)
print(f"Math samples: {total_math} ({total_math/total_samples:.1%})")
print(f"Chat samples: {total_chat} ({total_chat/total_samples:.1%})")
print(f"Total dataset size: {total_samples}")

# Check if desired proportions are achievable
expected_math_per_batch = int(16 * 0.7)  # 70% of batch size 16
expected_chat_per_batch = 16 - expected_math_per_batch
print(f"\nExpected per batch: Math={expected_math_per_batch}, Chat={expected_chat_per_batch}")