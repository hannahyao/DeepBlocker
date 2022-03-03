#GiG

import numpy as np
import pandas as pd
from pathlib import Path

from deep_blocker import DeepBlocker
from tuple_embedding_models import  AutoEncoderTupleEmbedding, CTTTupleEmbedding, HybridTupleEmbedding
from vector_pairing_models import ExactTopKVectorPairing
import blocking_utils

def do_blocking(folder_root, left_table_fname, right_table_fname, cols_to_block, tuple_embedding_model, vector_pairing_model):
    folder_root = Path(folder_root)
    left_df = pd.read_csv(folder_root / left_table_fname)
    right_df = pd.read_csv(folder_root / right_table_fname)

    db = DeepBlocker(tuple_embedding_model, vector_pairing_model)
    candidate_set_df = db.block_datasets(left_df, right_df, cols_to_block)

    # golden_df = pd.read_csv(Path(folder_root) /  "matches.csv")

    golden_df = get_golden_set(left_table_fname)

    statistics_dict = blocking_utils.compute_blocking_statistics(candidate_set_df, golden_df, left_df, right_df)
    return statistics_dict

def get_golden_set(left_table_fname, right_table_fname,):
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

if __name__ == "__main__":
    # folder_root = "data/Structured/Amazon-Google"
    # left_table_fname, right_table_fname = "tableA.csv", "tableB.csv"
    # cols_to_block = ["title", "manufacturer", "price"]

    folder_root = "nyc_cleaned"
    left_table_fname, right_table_fname = "myrx-addi", "8vqd-3345"
    cols_to_block = ["title", "manufacturer", "price"]
    # print("using AutoEncoder embedding")
    # tuple_embedding_model = AutoEncoderTupleEmbedding()
    # topK_vector_pairing_model = ExactTopKVectorPairing(K=50)
    # statistics_dict = do_blocking(folder_root, left_table_fname, right_table_fname, cols_to_block, tuple_embedding_model, topK_vector_pairing_model)
    # print(statistics_dict)

    print("using CTT embedding")
    tuple_embedding_model = CTTTupleEmbedding(synth_tuples_per_tuple=100)
    topK_vector_pairing_model = ExactTopKVectorPairing(K=1)
    statistics_dict = do_blocking(folder_root, left_table_fname, right_table_fname, cols_to_block, tuple_embedding_model, topK_vector_pairing_model)
    print(statistics_dict)

    print("using Hybrid embedding")
    tuple_embedding_model = CTTTupleEmbedding()
    topK_vector_pairing_model = ExactTopKVectorPairing(K=50)
    statistics_dict = do_blocking(folder_root, left_table_fname, right_table_fname, cols_to_block, tuple_embedding_model, topK_vector_pairing_model)
    print(statistics_dict)
