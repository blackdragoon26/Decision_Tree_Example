from .utils import *
import re
import numpy as np
from itertools import product

def get_rf_feature_thres(model_file, keys, tree_num):
    feat_dict = {key: [] for key in keys}
    for i in range(tree_num):
        with open(f"{model_file}.dot", 'r') as f:
            lines = f.readlines()
        for line in lines:
            if "[" in line:
                m = re.search(r".*\[label=[\"'](.*?)\s*<=\s*([\d.]+)\\n", line.strip())
                if m and m.group(1) in feat_dict:
                    feat_dict[m.group(1)].append(float(m.group(2)))
    
    for key in feat_dict:
        feat_dict[key] = [int(x)+1 for x in feat_dict[key]]  # rounding down +1
        feat_dict[key] = sorted(list(set(feat_dict[key])))  # unique and sorted
    return feat_dict

def get_rf_trees_table_entries(model_file, keys, feat_dict, key_encode_bits, tree_num, pkts=None):
    tree_data = []
    tree_leaves = []
    trees = [0]  # Initialize with starting index 0
    leaf_info = []
    
    for tree_idx in range(tree_num):
        start_count = len(tree_leaves)
        with open(f"{model_file}.dot", 'r') as f:
            lines = f.readlines()
        
        nodes = {}
        connections = []
        
        # First pass: Process all nodes
        for line in lines:
            line = line.strip()
            if not line or line.startswith(('digraph', 'node', 'edge', '}')):
                continue
                
            # Process decision nodes
            if 'label="' in line and '<=' in line and '->' not in line:
                m = re.search(r"(\d+)\s*\[label=\"(.*?)\s*<=\s*([\d.]+)\\n", line)
                if m:
                    node_id = m.group(1)
                    feat = m.group(2).strip()
                    threshold = m.group(3)
                    nodes[node_id] = {
                        'type': 'decision',
                        'feat': feat,
                        'thre': threshold,
                        'path': [1000, 0] * len(keys),
                        'has_left': False
                    }
            
            # Process leaf nodes
            elif ('gini' in line or 'entropy' in line) and 'value = ' in line:
                m = re.search(r"(\d+)\s*\[label=\".*value\s*=\s*(\[[^\]]+\]).*?class", line)
                if m:
                    node_id = m.group(1)
                    nodes[node_id] = {
                        'type': 'leaf',
                        'path': [1000, 0] * len(keys),
                        'value': m.group(2)
                    }
                    leaf_info.append(list_to_proba(m.group(2)))
            
            # Collect connections
            elif '->' in line:
                connections.append(line)
        
        # Second pass: Process connections
        for line in connections:
            m = re.search(r"(\d+)\s*->\s*(\d+)", line)
            if not m or m.group(1) not in nodes or m.group(2) not in nodes:
                continue
                
            parent = m.group(1)
            child = m.group(2)
            
            if nodes[parent]['type'] == 'decision':
                feat = nodes[parent]['feat']
                thre = int(float(nodes[parent]['thre'])) + 1
                
                # Copy parent path to child
                nodes[child]['path'] = nodes[parent]['path'].copy()
                
                # Update path based on direction
                if not nodes[parent]['has_left']:  # Left child
                    idx = feat_dict[feat].index(thre) + 1
                    nodes[child]['path'][keys.index(feat)*2] = min(
                        nodes[child]['path'][keys.index(feat)*2],
                        idx
                    )
                    nodes[parent]['has_left'] = True
                else:  # Right child
                    idx = -feat_dict[feat].index(thre) - 1
                    nodes[child]['path'][keys.index(feat)*2+1] = min(
                        nodes[child]['path'][keys.index(feat)*2+1],
                        idx
                    )
                
                # Register leaf nodes
                if nodes[child]['type'] == 'leaf':
                    tree_leaves.append(nodes[child]['path'])
        
        trees.append(len(tree_leaves))
        print(f"Tree {tree_idx} processed: {len(tree_leaves)-start_count} leaves, {len(nodes)} nodes")
    
    # Generate final tree data
    for i in range(len(trees)-1):
        for leaf_idx in range(trees[i], trees[i+1]):
            valid = True
            entry = [pkts] if pkts is not None else []
            
            for f, key in enumerate(keys):
                a = tree_leaves[leaf_idx][f*2]
                b = tree_leaves[leaf_idx][f*2+1]
                
                if a + b <= 0:  # Conflict check
                    valid = False
                    break
                    
                te = get_model_table_range_mark(key_encode_bits[key], a, b, len(feat_dict[key]))
                val, mask = get_value_mask(te, key_encode_bits[key])
                entry.extend([int(val, 2), int(mask, 2)])
            
            if valid:
                entry.append(leaf_info[leaf_idx])
                tree_data.append(entry)
    
    return tree_data
