import os

import numpy as np
import pandas as pd
from PIL import Image

from hu_moments import hu_moments


class Preprocessing:
    # TODO @Waldek As I understood, this is dataset-specific preprocessing. 
    # It might be worthy to describe which dataset you are capable of preprocessing with this class
    def __init__(self,
                train_path = "Datasets\TsignRecgTrain4170Annotation.txt",
                test_path = "Datasets\TsignRecgTest1994Annotation.txt"):
        self.train_path = train_path
        self.test_path = test_path
        self.df_train_ant =  []
        self.df_test_ant = []
    
    def get_datasets_annotations(self):
        df_train_ant = pd.read_csv(self.train_path,header=None,names=["file_name","width","height","x1","y1","x2","y2","category","none"], sep=';')
        df_train_ant.drop(columns="none",inplace=True)
        # 5 most common categories
        top_categories = df_train_ant["category"].value_counts()[2:7].index
        df_train_ant = df_train_ant.loc[df_train_ant["category"].isin(top_categories)].reset_index(drop=True).copy()
        df_train_ant = df_train_ant.sort_values(by=["category","file_name"],ignore_index=True)
        df_train_ant= df_train_ant.set_index("file_name")

        df_test_ant = pd.read_csv(self.test_path,header=None,names=["file_name","width","height","x1","y1","x2","y2","category","none"], sep=';')
        df_test_ant.drop(columns="none",inplace=True)
        df_test_ant = df_test_ant.loc[df_test_ant["category"].isin(top_categories)].reset_index(drop=True).copy()
        df_test_ant = df_test_ant.sort_values(by=["category","file_name"],ignore_index=True)
        df_test_ant= df_test_ant.set_index("file_name")
        self.df_train_ant =  df_train_ant
        self.df_test_ant = df_test_ant

        return df_train_ant, df_test_ant

    def crop_image(self,idx,dataset="Train", show=False):
        # TODO @Waldek is this method used? I cannot find it's usage in your code
        image = Image.open(os.path.join("Datasets",dataset,idx))
        if dataset =="Test":
            cropped_image = image.crop(self.df_test_ant[["x1","y1","x2","y2"]].loc[idx].to_list())
        else:
            cropped_image = image.crop(self.df_train_ant[["x1","y1","x2","y2"]].loc[idx].to_list())
        cropped_image = cropped_image.convert("L")
        if show:
            return cropped_image
        return np.array(cropped_image)

    def get_dataset(self):
        train_moments = []
        for idx in self.df_train_ant.index:
            image = self.crop_image(idx)
            moments = hu_moments(image)
            train_moments.append(moments)
        X_train = pd.DataFrame(train_moments, index=self.df_train_ant.index)
        y_train = np.array(self.df_train_ant["category"]).reshape(-1,1)

        test_moments = []
        for idx in self.df_test_ant.index:
            image = self.crop_image(idx,dataset="Test")
            moments = hu_moments(image)
            test_moments.append(moments)
        X_test = pd.DataFrame(test_moments, index=self.df_test_ant.index)
        y_test = np.array(self.df_test_ant["category"]).reshape(-1,1)

        return np.array(X_train), y_train, np.array(X_test), y_test
