from tree_to_table.rf import get_rf_feature_thres,get_rf_trees_table_entries
from tree_to_table.xgb import get_xgb_feature_thres,get_xgb_trees_table_entries
from tree_to_table.utils import *
import os

# current_path = os.getcwd()
dir_path = os.path.abspath(os.path.dirname(os.getcwd())) #


def get_class_flow(model_idx):
    print(dir_path)
    ## multi-phase classification
    pkt_flow_feat = ['f1', 'f2']
    pkt_flow_feat_bit = [16, 16]

    #convert the aggregate feature
    # bin_table = get_bin_table(pkt_flow_feat,16) #16 is the packet size bits

    #current_dir = os.path.join(dir_path, 'splidt', 'mininet', 'gsoc', 'models')
    current_dir="/home/motherfunder/IamWorking/parvezMaam/sampledecisiontrees/e2/NetBeacon/model_generation/models"

    class_flow_model_files = [
                current_dir + f'/samp_{model_idx}_filtered']
    class_flow_tree_nums = [1]

    max_feat_thres={}
    for i in range(len(pkt_flow_feat)):
        max_feat_thres[pkt_flow_feat[i]]=0
    count=0
    feat_dicts = []
    for model_file in class_flow_model_files:
        tree_num = class_flow_tree_nums[count]
        feat_dict = get_rf_feature_thres(model_file,pkt_flow_feat,tree_num)
        for key in feat_dict.keys():
            print(key,len(feat_dict[key]),feat_dict[key])
            if max_feat_thres[key]<len(feat_dict[key]):
                max_feat_thres[key]=len(feat_dict[key])
        count+=1
        feat_dicts.append(feat_dict)
    print(max_feat_thres)


    pkt_flow_mark_bit = [10, 10] #max_feat_thres
    feat_key_bits={}
    range_mark_bits = {}
    for i in range(len(pkt_flow_feat)):
        feat_key_bits[pkt_flow_feat[i]] = pkt_flow_feat_bit[i]
        range_mark_bits[pkt_flow_feat[i]] = pkt_flow_mark_bit[i]

    # pkt_flow_model_pkts = [2,4,8,32,256,512,2048] #model phases
    feat_table_data_all  = {}
    for i in range(len(pkt_flow_feat)):
        feat_table_data_all[pkt_flow_feat[i]] = []
    tree_data_all = []

    for i in range(len(class_flow_model_files)):
        tree_num = class_flow_tree_nums[i]
        model_file = class_flow_model_files[i]
        # pkts = pkt_flow_model_pkts[i]
        feat_dict = feat_dicts[i]
        feat_table_datas = get_feature_table_entries(feat_dict,feat_key_bits,range_mark_bits,pkts=None)
        sum_e = 0
        for key in feat_table_datas.keys():
            sum_e+=len(feat_table_datas[key])
        print("feature table entries: ",sum_e)

        tree_data = get_rf_trees_table_entries(model_file,pkt_flow_feat,feat_dict,range_mark_bits,tree_num,pkts=None)
        
        print("tree table entries: ",len(tree_data))
        print("all table entries: ",sum_e+len(tree_data))
        for i in range(len(pkt_flow_feat)):
            feat_table_data_all[pkt_flow_feat[i]].extend(feat_table_datas[pkt_flow_feat[i]])
        tree_data_all.extend(tree_data)

    for key in pkt_flow_feat:
        print(key,len(feat_table_data_all[key]))
    print(len(tree_data_all))

    
    
    # os.makedirs(output_dir, exist_ok=True)

    output_file = f'pkl_models/class_flow_model_{model_idx}_filtered.pkl'
    output_path = os.path.join(current_dir, output_file)

    with open(output_path, 'wb') as f:
        pickle.dump([feat_table_data_all, tree_data_all], f, protocol=2)

    print(f"Model data saved to {output_path}")

# Fix this to iterate over all the dot files in the directory
get_class_flow(0)# Example model index, change as needed
get_class_flow(1)
get_class_flow(2)
get_class_flow(3)
#main
