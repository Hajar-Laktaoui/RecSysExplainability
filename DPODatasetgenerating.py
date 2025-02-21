import os
import json
import pickle
import torch
import time
import argparse
from transformers import GPT2TokenizerFast
from collections import OrderedDict
from utils import DataLoader2, Batchify3, now_time, ids2tokensReal

def load_pickle_data(data_path):
    """
    Load a pickle file while preserving its original order.
    """
    with open(data_path, 'rb') as f:
        return pickle.load(f)

class OrderedDataLoader:
    """
    Loads data in the original order and provides batches for inference or generation.
    """
    def __init__(self, data_path, tokenizer, batch_size):
        self.data = load_pickle_data(data_path)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.current_idx = 0
        
        # Create ordered mappings for users/items
        self.user_dict = self._create_entity_dict([item['user'] for item in self.data])
        self.item_dict = self._create_entity_dict([item['item'] for item in self.data])
        
    def _create_entity_dict(self, entities):
        """
        Create ordered mappings (entity->index, index->entity) to preserve entity order.
        """
        unique_entities = OrderedDict.fromkeys(entities)
        entity2idx = {entity: idx for idx, entity in enumerate(unique_entities)}
        idx2entity = {idx: entity for entity, idx in entity2idx.items()}
        return {'entity2idx': entity2idx, 'idx2entity': idx2entity}
    
    def __len__(self):
        return len(self.data)
    
    def get_batch(self):
        """
        Fetch a batch of data in the original order, converting them to tensors.
        """
        if self.current_idx >= len(self.data):
            return None
            
        batch_data = self.data[self.current_idx:self.current_idx + self.batch_size]
        
        users = torch.tensor([
            self.user_dict['entity2idx'][item['user']] for item in batch_data
        ])
        items = torch.tensor([
            self.item_dict['entity2idx'][item['item']] for item in batch_data
        ])
        
        texts = [item['template'][2] for item in batch_data]
        encoded = self.tokenizer(texts, padding=True, return_tensors='pt')
        
        self.current_idx += self.batch_size
        return {
            'user': users,
            'item': items,
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'original_data': batch_data
        }

def ensure_directory_exists(file_path):
    """
    Create the directory for the given file if it doesn't already exist.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(now_time() + f" Created directory: {directory}")

def save_checkpoint(data, output_path, batch_num, total_items):
    """
    Save checkpoint data (JSON) after each batch, ensuring the directory exists.
    """
    ensure_directory_exists(output_path)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(now_time() + f" Checkpoint saved: {batch_num}/{total_items} items processed")
        return True
    except Exception as e:
        print(now_time() + f" Error saving checkpoint: {str(e)}")
        return False

def generate_sentences(model, data_loader, device, output_path, max_length=20):
    """
    Generate sentences from the model, preserving original order, and save output after each batch.
    """
    model.eval()
    generated_data = []
    batch_num = 0
    total_batches = len(data_loader) // data_loader.batch_size

    ensure_directory_exists(output_path)

    with torch.no_grad():
        while True:
            batch = data_loader.get_batch()
            if batch is None:
                break

            user = batch['user'].to(device)
            item = batch['item'].to(device)
            input_ids = batch['input_ids'].to(device)

            # Autoregressive generation
            text = input_ids[:, :1]
            for _ in range(max_length):
                outputs = model(user, item, text, None, None)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                text = torch.cat([text, next_token], dim=1)
                if (next_token == data_loader.tokenizer.eos_token_id).all():
                    break

            for idx, (gen_ids, orig_data) in enumerate(zip(text, batch['original_data'])):
                generated_text = data_loader.tokenizer.decode(gen_ids[1:], skip_special_tokens=True)
                original_text = orig_data['template'][2]
                generated_data.append({
                    "user_id": orig_data['user'],
                    "item_id": orig_data['item'],
                    "chosen": original_text,
                    "rejected": generated_text
                })

            batch_num += 1
            save_checkpoint(generated_data, output_path, batch_num, total_batches)

            if batch_num % 10 == 0:
                print(f"Processed {batch_num}/{total_batches} batches")

    return generated_data

def main(args):
    """
    Main entry point for generating sentences with a pretrained model.
    """
    ensure_directory_exists(args.output_json)
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    print(now_time() + ' Loading tokenizer...')
    tokenizer = GPT2TokenizerFast.from_pretrained(
        'gpt2', 
        bos_token='<bos>', 
        eos_token='<eos>', 
        pad_token='<pad>'
    )
    
    print(now_time() + ' Loading dataset...')
    data_loader = OrderedDataLoader(args.data_path, tokenizer, args.batch_size)
    
    print(now_time() + ' Loading model...')
    model = torch.load(args.model_path, map_location=device)
    model.to(device)
    
    print(now_time() + ' Generating sentences...')
    start_time = time.time()
    generated_data = generate_sentences(model, data_loader, device, args.output_json)
    print(f"Generation time: {time.time() - start_time:.2f} seconds")
    print(now_time() + f' Generated sentences saved to {args.output_json}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Sentences')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output_json', type=str, required=True, help='Output JSON file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    
    args = parser.parse_args()
    main(args)
