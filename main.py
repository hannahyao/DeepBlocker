#GiG

import numpy as np
import pandas as pd
from pathlib import Path

from deep_blocker import DeepBlocker
from tuple_embedding_models import  AutoEncoderTupleEmbedding, CTTTupleEmbedding, HybridTupleEmbedding, SIFEmbedding
from vector_pairing_models import ExactTopKVectorPairing
import blocking_utils
from configurations import * 
import pickle

def ctt_train_score_with_pred(folder_root, golden_set,left_table_fname, right_table_fname, cols_to_block, tuple_embedding_model, vector_pairing_model):
    folder_root = Path(folder_root)
    left_table_name_csv = left_table_fname+'.csv'
    right_table_name_csv = right_table_fname+'.csv'
    left_df = pd.read_csv(folder_root / left_table_name_csv)
    right_df = pd.read_csv(folder_root / right_table_name_csv)

    db = DeepBlocker(tuple_embedding_model, vector_pairing_model)
    candidate_set_df,predictions = db.block_datasets(left_df, right_df, cols_to_block,True)
    predictions = pd.DataFrame(predictions,columns=['ltable_id','rtable_id','value'])
    # print(predictions)
    # predictions = predictions[predictions.prediction > 0.1]
    # golden_df = pd.read_csv(Path(folder_root) /  "matches.csv")

    golden_df = filter_golden_set(golden_set,left_table_fname, right_table_fname)
    statistics_dict_binary = blocking_utils.compute_blocking_statistics((left_table_fname,right_table_fname), predictions, golden_df, left_df, right_df,CTT_BINARY_THRESHOLD)
    statistics_dict = blocking_utils.compute_blocking_statistics((left_table_fname,right_table_fname), candidate_set_df, golden_df, left_df, right_df,CTT_EMBEDDING_THRESHOLD)
    
    return statistics_dict,statistics_dict_binary

def do_blocking(folder_root, golden_set, left_table_fname, right_table_fname, cols_to_block, tuple_embedding_model, vector_pairing_model,threshold):
    folder_root = Path(folder_root)
    left_table_name_csv = left_table_fname+'.csv'
    right_table_name_csv = right_table_fname+'.csv'
    left_df = pd.read_csv(folder_root / left_table_name_csv)
    right_df = pd.read_csv(folder_root / right_table_name_csv)

    db = DeepBlocker(tuple_embedding_model, vector_pairing_model)
    candidate_set_df = db.block_datasets(left_df, right_df, cols_to_block,False)
    # golden_df = pd.read_csv(Path(folder_root) /  "matches.csv")

    golden_df = filter_golden_set(golden_set,left_table_fname, right_table_fname)
    statistics_dict = blocking_utils.compute_blocking_statistics((left_table_fname,right_table_fname), candidate_set_df, golden_df, left_df, right_df,threshold)
 
    return statistics_dict

def get_golden_set(left_table_fname, right_table_fname):
    output_file = 'nyc_output/'+ left_table_fname + '-output.txt'
    with open(output_file) as f:
        lines = f.readlines()
    line_df = pd.DataFrame(lines,columns=['full'])
    line_df = line_df['full'].str.split("JOIN", n = 1, expand = True)
    line_df = line_df.replace('\n',' ', regex=True)
    line_df.columns = ['ltable_id','rtable_id']
    
    golden_df = line_df[line_df['ltable_id'].str.contains(left_table_fname)]
    golden_df = line_df[line_df['rtable_id'].str.contains(right_table_fname)]
    golden_df.ltable_id = golden_df.ltable_id.str.strip()
    golden_df.rtable_id = golden_df.rtable_id.str.strip()

    golden_df['ltable_id'] = golden_df['ltable_id'].astype('str')

    return golden_df

def get_golden_set_full():
    output_file = 'nyc_output/'+ 'merged-output.txt'
    with open(output_file) as f:
        lines = f.readlines()
    line_df = pd.DataFrame(lines,columns=['full'])
    line_df = line_df['full'].str.split("JOIN", n = 1, expand = True)
    line_df = line_df.replace('\n',' ', regex=True)
    line_df.columns = ['ltable_id','rtable_id']
    
    line_df.ltable_id = line_df.ltable_id.str.strip()
    line_df.rtable_id = line_df.rtable_id.str.strip()
    line_df['left_table'] = line_df['ltable_id'].str.split('.').str[0]
    line_df['right_table'] = line_df['rtable_id'].str.split('.').str[0]
    line_df = line_df.drop_duplicates()
    tables = line_df[['left_table','right_table']].drop_duplicates()
    records = tables.to_records(index=False)
    table_pairs = list(records)
    return line_df, table_pairs
    

def filter_golden_set(golden_set,left_table_fname, right_table_fname):
    golden_df = golden_set[golden_set['ltable_id'].str.contains(left_table_fname)]
    golden_df = golden_set[golden_set['rtable_id'].str.contains(right_table_fname)]
    golden_df.ltable_id = golden_df.ltable_id.str.strip()
    golden_df.rtable_id = golden_df.rtable_id.str.strip()

    golden_df['ltable_id'] = golden_df['ltable_id'].astype('str')

    return golden_df

if __name__ == "__main__":
    # folder_root = "data/Structured/Amazon-Google"
    # left_table_fname, right_table_fname = "tableA.csv", "tableB.csv"
    # cols_to_block = ["title", "manufacturer", "price"]
    golden_set, table_pairs = get_golden_set_full()
    output = []
    for pair in table_pairs:
            
        folder_root = "nyc_cleaned"
        left_table_fname, right_table_fname = pair[0],pair[1]
        cols_to_block = [None]
        # print("using AutoEncoder embedding")
        # tuple_embedding_model = AutoEncoderTupleEmbedding()
        # topK_vector_pairing_model = ExactTopKVectorPairing(K=50)
        # statistics_dict = do_blocking(folder_root, left_table_fname, right_table_fname, cols_to_block, tuple_embedding_model, topK_vector_pairing_model)
        # print(statistics_dict)

        print("using SIF embedding")
        tuple_embedding_model = SIFEmbedding()
        topK_vector_pairing_model = ExactTopKVectorPairing(K=1)
        statistics_dict = do_blocking(folder_root, golden_set, left_table_fname, right_table_fname, cols_to_block, tuple_embedding_model, topK_vector_pairing_model,SIF_EMBEDDING_THRESHOLD)
        statistics_dict['mode'] = 'SIF'
        output.append(statistics_dict)
        # print(statistics_dict)

        print("using CTT embedding")
        tuple_embedding_model = CTTTupleEmbedding(synth_tuples_per_tuple=100)
        topK_vector_pairing_model = ExactTopKVectorPairing(K=1)
        statistics_dict , statistics_dict_binary= ctt_train_score_with_pred(folder_root, golden_set, left_table_fname, right_table_fname, cols_to_block, tuple_embedding_model, topK_vector_pairing_model)
        
        statistics_dict['mode'] = 'CTT_embedding'
        output.append(statistics_dict)
        statistics_dict_binary['mode'] = 'CTT_classifier'
        output.append(statistics_dict_binary)

        print(output)
        # print("using Hybrid embedding")
        # tuple_embedding_model = CTTTupleEmbedding()
        # topK_vector_pairing_model = ExactTopKVectorPairing(K=50)
        # statistics_dict = do_blocking(folder_root, left_table_fname, right_table_fname, cols_to_block, tuple_embedding_model, topK_vector_pairing_model)
        # print(statistics_dict)
    print(output)
    with open('output_stats.pkl', 'wb') as f:
        pickle.dump(output, f)
