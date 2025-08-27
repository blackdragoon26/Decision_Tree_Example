import re
import json
import os

# You can adjust the root dir if needed
#dir_path = os.getcwd()
#current_dir = os.path.join(dir_path, 'models', 'dot_models')
current_dir="/home/motherfunder/IamWorking/parvezMaam/sampledecisiontrees/e2"
# Operation map for known features
OPERATION_MAP = {
    "f1": "sum",
    "f2": "min",
    "f3": "max"
}

def parse_dot(dot_text):
    nodes, edges = {}, []
    for line in dot_text.splitlines():
        line = line.strip()
        node_match = re.match(r'(\d+) \[label="([^"]+)"\] ;', line)
        edge_match = re.match(r'(\d+) -> (\d+) ;', line)

        if node_match:
            nodes[node_match.group(1)] = node_match.group(2)
        elif edge_match:
            edges.append((edge_match.group(1), edge_match.group(2)))
    return nodes, edges

def remap_features(nodes):
    feature_alias = {}       # Original feature → f1/f2
    reverse_alias = {}       # f1/f2 → Original feature
    json_map = {}            # f1/f2 → { original_feature, operation }
    valid_node_ids = set()
    alias_index = 1

    for node_id, label in nodes.items():
        match = re.search(r'(f\d+)', label)
        if match:
            original_feature = match.group(1)

            if original_feature not in feature_alias:
                if alias_index > 2:
                    raise ValueError("❌ More than 2 unique features found. Aborting.")
                assigned_alias = f"f{alias_index}"
                feature_alias[original_feature] = assigned_alias
                reverse_alias[assigned_alias] = original_feature

                op = OPERATION_MAP.get(original_feature, "unknown")
                json_map[assigned_alias] = {
                    "original_feature": original_feature,
                    "operation": op
                }
                alias_index += 1

            alias = feature_alias[original_feature]
            new_label = label.replace(original_feature, alias)
            nodes[node_id] = new_label
            valid_node_ids.add(node_id)
        else:
            valid_node_ids.add(node_id)
    return nodes, json_map, valid_node_ids

def rebuild_dot(nodes, edges, valid_node_ids):
    lines = [
        "digraph Tree {",
        'node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;',
        'edge [fontname=helvetica] ;\n'
    ]
    for node_id in nodes:
        if node_id in valid_node_ids:
            lines.append(f'{node_id} [label="{nodes[node_id]}"] ;')

    for src, dst in edges:
        if src in valid_node_ids and dst in valid_node_ids:
            lines.append(f"{src} -> {dst} ;")

    lines.append("}")
    return "\n".join(lines)

def process_dot_file(dot_path):
    try:
        with open(dot_path, 'r') as f:
            dot_text = f.read()

        nodes, edges = parse_dot(dot_text)
        nodes, feature_map, valid_node_ids = remap_features(nodes)

        # Output paths
        base = os.path.splitext(dot_path)[0]
        out_dot = base + "_filtered.dot"
        out_json = base + "_mapping.json"

        with open(out_dot, 'w') as f:
            f.write(rebuild_dot(nodes, edges, valid_node_ids))

        with open(out_json, 'w') as f:
            json.dump(feature_map, f, indent=2)

        print(f"Processed: {os.path.basename(dot_path)}")
        print(f"Saved: {os.path.basename(out_dot)}, {os.path.basename(out_json)}")
    except ValueError as ve:
        print(f"Skipped {os.path.basename(dot_path)}: {ve}")

def main():
    print(f"Scanning directory: {current_dir}")
    for filename in os.listdir(current_dir):
        if filename.endswith(".dot"):
            full_path = os.path.join(current_dir, filename)
            process_dot_file(full_path)

if __name__ == "__main__":
    main()

