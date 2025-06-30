import torch

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from os import path
   

class GenericDataset(Dataset):
    """
    Generic dataset class for tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
    - tensors (Tensor): tensors that have the same size of the first dimension.

    Returns:
    - dataset (Torch.utils.data.Dataset): dataset with provided tensors in order
    """

    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        for tensor in self.tensors:
            item = tensor[index]
        return item

    def __len__(self):
        return self.tensors[0].size(0)



def get_generic_tensor_dataloader(train_data,test_data=None, batch_size=32, num_workers=0, valid_frac=0):
    """
    Create PyTorch DataLoader instances for training, validation, and testing datasets.

    This function creates DataLoader instances for the training, validation, and testing datasets
    based on the provided input data. It supports creating a validation set with a specified fraction
    of the training data and provides options for shuffling and setting the batch size.

    Args:
    - train_data (torch.Tensor): Training data.
    - test_data (torch.Tensor, optional): Testing data. Default is None.
    - batch_size (int, optional): Batch size for the DataLoader. Default is 32.
    - num_workers (int, optional): Number of workers for data loading. Default is 0.
    - valid_frac (float, optional): Fraction of training data to use for validation (0 to 1). Default is 0.

    Returns:
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    - valid_loader (torch.utils.data.DataLoader, optional): DataLoader for the validation dataset.
    - test_loader (torch.utils.data.DataLoader, optional): DataLoader for the testing dataset.
    """
    
    train_dataset = GenericDataset(train_data)

    if valid_frac != 0:
        valid_dataset = GenericDataset(train_data)

    if test_data != None:
        test_dataset = GenericDataset(test_data)

    if valid_frac != 0:
        num = int(valid_frac * len(train_dataset))
        train_indices = torch.arange(0,len(train_dataset - num))
        valid_indices = torch.arange(len(train_dataset) - num, len(train_dataset))
        
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        
        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=valid_sampler)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  sampler=train_sampler)
    else:
        train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True)

    if test_data != None: 
        test_loader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False)
    
    if valid_frac == 0 and test_data == None:
        return train_loader
    elif valid_frac == 0 and test_data != None:
        return train_loader, test_loader
    else:
        return train_loader, valid_loader, test_loader
    


class FeatureLabelDataset(Dataset):

    def __init__(self,df,fnames,lnames):
        ftensors = torch.FloatTensor(df[fnames].to_numpy())
        ltensors = torch.FloatTensor(df[lnames].to_numpy())

        assert all(ftensors[0].size(0) == tensor.size(0) for tensor in ftensors)
        assert all(ltensors[0].size(0) == tensor.size(0) for tensor in ltensors)
        assert ftensors.size(0) == ltensors.size(0)

        self.ftensors = ftensors
        self.ltensors = ltensors

    def __getitem__(self, index):

        ftensor = self.ftensors[index]
        ltensor = self.ltensors[index]

        return ftensor,ltensor

    def __len__(self):
        return self.ftensors.size(0)
    


def get_feature_label_tensor_dataloader(df,feature_columns,label_columns,test_df=None, batch_size=32, num_workers=0, valid_frac=0):

    if valid_frac != 0:
        train_df,valid_df = train_test_split(df,test_size=valid_frac,shuffle=True)
        train_dataset = FeatureLabelDataset(train_df,feature_columns,label_columns)
        valid_dataset = FeatureLabelDataset(valid_df,feature_columns,label_columns)
        
        valid_loader = DataLoader(dataset=valid_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                shuffle=False)
    else:
        train_dataset = FeatureLabelDataset(df,feature_columns,label_columns)

    train_loader = DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            shuffle = True,
                                            drop_last=False)

    if test_df != None:
        test_dataset = FeatureLabelDataset(test_df,feature_columns,label_columns)
        test_loader = DataLoader(dataset=test_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                shuffle=False)
                    
    if valid_frac == 0 and test_df == None:
        return train_loader
    elif valid_frac == 0 and test_df != None:
        return train_loader, test_loader
    elif valid_frac != 0 and test_df == None:
        return train_loader, valid_loader
    else:
        return train_loader, valid_loader, test_loader
    

def get_dataset_and_scaler(df,valid_frac,feature_columns,label_columns,batch_size=32,num_workers=0):
        ## Divide Dataset
        train_df,valid_df = train_test_split(df,test_size=valid_frac,shuffle=True)

        ## Fit scaler
        feature_scaler,label_scaler = StandardScaler(),StandardScaler()
        feature_scaler.fit(train_df[feature_columns].to_numpy())
        label_scaler.fit(train_df[label_columns].to_numpy())

        ## Generate Dataset
        train_dataset = FeatureLabelDataset(train_df,feature_columns,label_columns)
        valid_dataset = FeatureLabelDataset(valid_df,feature_columns,label_columns)
        
        ## Generate Loaders
        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=False)
    
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle = True,
                                  drop_last=False)

        return train_loader,valid_loader,feature_scaler,label_scaler

 
 # DECRAPED use scaled NN instead
def get_scale_feature_label_tensor_dataloader(df,feature_columns,label_columns,test_df=None, batch_size=32, num_workers=0, valid_frac=0):

    std_scaler = StandardScaler()

    if valid_frac != 0:
        train_df,valid_df = train_test_split(df,test_size=valid_frac,shuffle=True)
        
        train_df_scaled = std_scaler.fit_transform(train_df.to_numpy())
        train_df_scaled = pd.DataFrame(train_df_scaled, columns=[df.columns])

        valid_df_scaled = std_scaler.transform(valid_df.to_numpy())
        valid_df_scaled = pd.DataFrame(valid_df_scaled, columns=[df.columns])

        train_dataset = FeatureLabelDataset(train_df_scaled,feature_columns,label_columns)
        valid_dataset = FeatureLabelDataset(valid_df_scaled,feature_columns,label_columns)
        
        valid_loader = DataLoader(dataset=valid_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                shuffle=False)
    else:
        df_scaled = std_scaler.fit_transform(df.to_numpy())
        df_scaled = pd.DataFrame(df_scaled, columns=[df.columns])

        train_dataset = FeatureLabelDataset(df_scaled,feature_columns,label_columns)

    train_loader = DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            shuffle = True,
                                            drop_last=False)

    if test_df != None:
        test_df_scaled = std_scaler.transform(test_df.to_numpy())
        test_df_scaled = pd.DataFrame(test_df_scaled, columns=[df.columns])
        test_dataset = FeatureLabelDataset(test_df,feature_columns,label_columns)
        test_loader = DataLoader(dataset=test_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                shuffle=False)
                    
    if valid_frac == 0 and test_df == None:
        return train_loader,std_scaler
    elif valid_frac == 0 and test_df != None:
        return train_loader, test_loader,std_scaler
    elif valid_frac != 0 and test_df == None:
        return train_loader, valid_loader, std_scaler
    else:
        return train_loader, valid_loader, test_loader, std_scaler