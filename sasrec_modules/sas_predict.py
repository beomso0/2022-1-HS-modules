from sasrec.model import SASREC
from sasrec.sampler import WarpSampler
from sasrec.util import SASRecDataSet, load_model
import pickle

class SASRec_predictor():

    def __init__(self, batch_size=4096,**kwargs):
        self.model = kwargs.get("model", None)
        self.data = kwargs.get("data", None)        
        self.user_map = kwargs.get("user_map", None)        
        self.item_map = kwargs.get("item_map", None)  
        self.batch_size = batch_size 


    def load_data(self, path, exp_name):
        # load data
        data = SASRecDataSet(filename=f'{path}/{exp_name}/SASRec_dataset_{exp_name}.txt', col_sep="\t")
        data.split()
        
        # load maps
        with open(f'{path}/{exp_name}/SASRec_user_item_map_{exp_name}.pkl','rb') as f:
            user_item_map = pickle.load(f)
        
        self.data = data
        self.user_map = user_item_map[0]
        self.item_map = user_item_map[1]
        self.inv_user_map = {v: k for k, v in self.user_map.items()}


    def load_model(self, path, exp_name):
        self.model = load_model(path, exp_name)


    def predict(self, user_list=None, item_list=None):

        # if user_list is not passed, make user_list (None means all)
        if user_list == None:
            self.model.sample_val_users(self.data,self.data.usernum)
            user_list = [self.inv_user_map[u] for u in self.model.val_users]


        # check user_list
        not_found_users = set(user_list).difference(self.user_map.keys())
        
        if len(not_found_users)==0:
            pass
        else:
            raise Exception(f'these users are not found in user_map \n{not_found_users}')


        #check item_list
        not_found_items = set(item_list).difference(self.item_map.keys())

        if len(not_found_items)==0:
            pass
        else:
            raise Exception(f'these items are not found in item_map \n{not_found_items}')

        
        score_df = self.model.get_user_item_score(self.data,
                                                user_list, 
                                                item_list,
                                                self.user_map,
                                                self.item_map,   
                                                self.batch_size if self.batch_size <= len(user_list) else len(user_list)-1
                                              )

        return score_df