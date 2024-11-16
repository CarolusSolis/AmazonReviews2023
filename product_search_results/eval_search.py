import os
import argparse
import numpy as np
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from recbole.evaluator.metrics import NDCG
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def set_device(gpu_id):
    if gpu_id == -1:
        return torch.device('cpu')
    else:
        return torch.device(
            'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='McAuley-Lab/Amazon-C4', help='dataset')
    parser.add_argument('--suffix', type=str, default='blair-baseCLS', help='suffix of the embs')
    parser.add_argument('-k', type=int, default=100, help='top k')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('-bs', type=int, default=64, help='batch size')
    parser.add_argument('--domain', action='store_true', help='whether to output results of each domain')
    parser.add_argument('--data_path', type=str, default='./cache', help='path to the data')
    parser.add_argument('--categories', type=str, nargs='+', help='specific categories to evaluate (e.g., --categories Pet Care)')

    args = parser.parse_args()

    if args.categories:
        category_suffix = '_' + ''.join(args.categories)
        args.suffix = args.suffix.replace('CLS', '') + category_suffix + 'CLS'

    args.dataset_name = args.dataset.split('/')[-1]
    args.plm_size = 1024 if 'large' in args.suffix else 768

    device = set_device(args.gpu_id)
    args.device = device

    return args


def load_items(args, categories_of_interest=None):
    id2item = []
    item2id = {}

    if args.dataset == 'McAuley-Lab/Amazon-C4':
        # Load category mapping
        cat_filepath = hf_hub_download(
            repo_id='McAuley-Lab/Amazon-Reviews-2023',
            filename='asin2category.json',
            repo_type='dataset'
        )
        with open(cat_filepath, 'r') as file:
            asin2category = json.loads(file.read())

        filepath = hf_hub_download(
            repo_id=args.dataset,
            filename='sampled_item_metadata_1M.jsonl',
            repo_type='dataset'
        )

        with open(filepath, 'r') as file:
            idx = 0
            for line in file:
                item_data = json.loads(line.strip())
                item = item_data['item_id']
                
                # Skip if not in categories of interest
                if args.categories and item_data['category'] not in args.categories:
                    continue
                    
                id2item.append(item)
                item2id[item] = idx
                idx += 1
                assert len(id2item) == len(item2id)
                assert len(id2item) == idx
    else:
        raise NotImplementedError('Dataset not supported')

    return id2item, item2id


def load_queries(args, item2id):
    query2target = []
    dataset = load_dataset(args.dataset)['test']
    
    if args.categories:
        # Load category mapping
        filepath = hf_hub_download(
            repo_id=args.dataset,
            filename='sampled_item_metadata_1M.jsonl',
            repo_type='dataset'
        )
        
        item_categories = {}
        with open(filepath, 'r') as file:
            for line in file:
                item_data = json.loads(line.strip())
                item_categories[item_data['item_id']] = item_data['category']
        
        # Only include queries whose target items are in the specified categories
        for query, target_item in zip(dataset['query'], dataset['item_id']):
            if target_item in item2id and target_item in item_categories and \
               item_categories[target_item] in args.categories:
                target_id = item2id[target_item]
                query2target.append(target_id)
    else:
        for target_item in dataset['item_id']:
            if target_item in item2id:  # Only include if target exists in filtered set
                target_id = item2id[target_item]
                query2target.append(target_id)
                
    return query2target


def load_plm_embedding(args, suffix):
    feat_path = os.path.join(args.data_path, args.dataset_name, f'{args.dataset_name}.{suffix}')
    loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, args.plm_size)
    return torch.FloatTensor(loaded_feat)


if __name__ == '__main__':
    # Arguments
    args = load_args()

    # Load index
    id2item, item2id = load_items(args)
    query2target = torch.LongTensor(load_queries(args, item2id)).to(args.device)

    # Load embeddings
    item_embs = load_plm_embedding(args, args.suffix)
    item_embs = F.normalize(item_embs, dim=-1).to(args.device)
    query_embs = load_plm_embedding(args, 'q_' + args.suffix)
    query_embs = F.normalize(query_embs, dim=-1).to(args.device)
    assert item_embs.shape[0] == len(id2item)
    assert query_embs.shape[0] == len(query2target)

    # topk
    metric = NDCG({
        'metric_decimal_place': 4,
        'topk': args.k
    })
    results = []
    with torch.no_grad():
        for pr in tqdm(range(0, query_embs.shape[0], args.bs)):
            batch_queries = query_embs[pr:pr+args.bs]
            batch_target = query2target[pr:pr+args.bs]
            scores = batch_queries @ item_embs.T    # [bs, n_items]
            topk_scores, topk_indices = torch.topk(scores, args.k, dim=-1)
            pos_index = (batch_target.unsqueeze(-1).expand(-1, args.k) == topk_indices).cpu()
            pos_len = torch.ones_like(batch_target).cpu().numpy()
            ndcg = metric.metric_info(pos_index, pos_len)
            ndcg = ndcg[:,-1]
            results.append(ndcg)
    
    results = np.concatenate(results)
    print(args)
    print(f'Overall NDCG@{args.k}: {results.mean()}')

    if args.domain:
        filepath = hf_hub_download(
            repo_id='McAuley-Lab/Amazon-Reviews-2023',
            filename='asin2category.json',
            repo_type='dataset'
        )
        with open(filepath, 'r') as file:
            asin2category = json.loads(file.read())

        cat2results = defaultdict(list)
        for i in range(query2target.shape[0]):
            target_id = query2target[i].cpu().item()
            item_token = id2item[target_id]
            cat = asin2category[item_token]
            cat2results[cat].append(results[i])

        for d in cat2results:
            print(f'  {d}: {np.mean(cat2results[d])}')
