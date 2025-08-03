from langchain_community.document_loaders import DirectoryLoader, TextLoader
from utils import get_parent_dir, save_pickle
import os
import pickle
from utils import data_file_exists

class DataLoader():
    def __init__(self, dataset_name:str, data_folder :str = "data"):
        '''
        Initializes the DataLoader class.

        NOTE: The data folder is expected to be one level above the data_loader.py file.
        The data folder is expected to contain a subfolder with the name of the dataset.
        The subfolders there are expected to contain the data files.
        In the dataset folder, the data once loaded for a first time is saved to a pickle file.
        The next time the data is loaded, the pickle file is used instead of loading the data from the subfolders.
        This is to avoid loading the data from the subfolders every time the data is loaded.

        Args:
            dataset_name (str): The name of the dataset (this will also be the name of the subfolder in the data folder,
             and the pickle file).
            data_folder (str): The name of the data folder.
        
        '''
        self.data_folder = data_folder
        self.dataset_name = dataset_name
        self.data_path = None
        self.loaded_data = None
        

    def load_data(self):
        '''
        Loads data from the data folder and saves it to a pickle file if it doesn't exist.
        '''

        base_path = get_parent_dir(__file__)
        self.data_path = os.path.join(base_path, self.data_folder, self.dataset_name)

        exists = data_file_exists(self.data_path, self.dataset_name)

        if exists:
            with open(os.path.join(self.data_path, f"{self.ataset_name}.pkl"), "rb") as f:
                loaded_data = pickle.load(f)
        else:
            directory_loader = DirectoryLoader(self.data_path, glob="**/*.txt", loader_cls=TextLoader)
            loaded_data = directory_loader.load()
            save_pickle(loaded_data, os.path.join(self.data_path, f"{self.dataset_name}.pkl"))

        return loaded_data
    
    def get_loaded_data(self):
        return self.loaded_data






    