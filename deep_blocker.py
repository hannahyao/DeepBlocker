#GiG
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
import blocking_utils

class DeepBlocker:
    def __init__(self, tuple_embedding_model, vector_pairing_model):
        self.tuple_embedding_model = tuple_embedding_model
        self.vector_pairing_model = vector_pairing_model

    def validate_columns(self):
        #Assumption: id column is named as id
        if "id" not in self.cols_to_block:
            self.cols_to_block.append("id")
        self.cols_to_block_without_id = [col for col in self.cols_to_block if col != "id"]

        #Check if all required columns are in left_df
        check = all([col in self.left_df.columns for col in self.cols_to_block])
        if not check:
            raise Exception("Not all columns in cols_to_block are present in the left dataset")

        #Check if all required columns are in right_df
        check = all([col in self.right_df.columns for col in self.cols_to_block])
        if not check:
            raise Exception("Not all columns in cols_to_block are present in the right dataset")


    def preprocess_datasets(self):
        self.left_df = self.left_df[self.cols_to_block]
        self.right_df = self.right_df[self.cols_to_block]

        self.left_df.fillna(' ', inplace=True)
        self.right_df.fillna(' ', inplace=True)

        self.left_df = self.left_df.astype(str)
        self.right_df = self.right_df.astype(str)


        self.left_df["_merged_text"] = self.left_df[self.cols_to_block_without_id].agg(' '.join, axis=1)
        self.right_df["_merged_text"] = self.right_df[self.cols_to_block_without_id].agg(' '.join, axis=1)

        #Drop the other columns
        self.left_df = self.left_df.drop(columns=self.cols_to_block_without_id)
        self.right_df = self.right_df.drop(columns=self.cols_to_block_without_id)


    def preprocess_columns(self):
  
        self.left_df = self.left_df.astype('str')
        self.right_df = self.right_df.astype('str')
        self.left_df.fillna(' ', inplace=True)
        self.right_df.fillna(' ', inplace=True)

        n=1 # lenght of dataset
        k = 10 #length of tuple
        left_list = []
        for cols in self.left_df.columns:
            left_df_tuple = [self.left_df.sample(n=k,random_state=1)[cols].T for x in range(0,n)]
            left_tuple_list = [','.join(i) for i in left_df_tuple]
            left_list.extend(left_tuple_list)
        right_list = [] 
        for cols in self.right_df.columns:
            right_df_tuple = [self.right_df.sample(n=k,random_state=1)[cols].T for x in range(0,n)]
            right_df_tuple_list = [','.join(i) for i in right_df_tuple]
            right_list.extend(right_df_tuple_list)

        self.left_df = pd.DataFrame(left_list,columns=['_merged_text'])
        self.left_df['id'] = np.arange(len(self.left_df))
        self.right_df = pd.DataFrame(right_list,columns=['_merged_text'])
        self.right_df['id'] = np.arange(len(self.right_df))
        
        # return self.left_df,self.right_df

    def block_datasets(self, left_df, right_df, cols_to_block,predict=False):
        self.left_df = left_df
        self.right_df = right_df
        self.cols_to_block = cols_to_block

        # self.validate_columns()
        # self.preprocess_datasets()

        self.preprocess_columns()

        print("Performing pre-processing for tuple embeddings ")
        all_merged_text = pd.concat([self.left_df["_merged_text"], self.right_df["_merged_text"]], ignore_index=True)
        self.tuple_embedding_model.preprocess(all_merged_text)

        print("Obtaining tuple embeddings for left table")
        self.left_tuple_embeddings = self.tuple_embedding_model.get_tuple_embedding(self.left_df["_merged_text"])
        print("Obtaining tuple embeddings for right table")
        self.right_tuple_embeddings = self.tuple_embedding_model.get_tuple_embedding(self.right_df["_merged_text"])
        
        if predict:
            prediction = self.tuple_embedding_model.get_prediction(self.left_df, self.right_df)

        print("Indexing the embeddings from the right dataset")
        self.vector_pairing_model.index(self.right_tuple_embeddings)

        print("Querying the embeddings from left dataset")
        topK_neighbors ,topK_values= self.vector_pairing_model.query(self.left_tuple_embeddings)

        self.candidate_set_df = blocking_utils.topK_neighbors_to_candidate_set(topK_neighbors,topK_values)
        
        if predict:
            return self.candidate_set_df,prediction
        return self.candidate_set_df
