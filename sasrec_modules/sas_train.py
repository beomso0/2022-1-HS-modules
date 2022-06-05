from sasrec.model import SASREC
from sasrec.sampler import WarpSampler
from sasrec.util import SASRecDataSet, load_model
import multiprocessing

def SASRec_train(
    data_path,
    save_path,
    exp_name,
    num_epochs=30,
    batch_size=2048,
    lr = 0.001,
    maxlen=100,
    num_blocks = 2,
    hidden_units = 256,
    num_heads = 2,
    dropout_rate = 0.5,
    l2_reg = 0.00001,
    save_last_only=True # If True, model evaluation will be executed only after the last epoch. 
                        # If False, model evaluation will be executed after every epoch, and better model will be saved.
):
    
    # load data
    data = SASRecDataSet(filename=data_path, col_sep="\t")
    data.split()

    # print train information
    num_steps = int(len(data.user_train) / batch_size)
    cc = 0.0
    for u in data.user_train:
        cc += len(data.user_train[u])
    print('%g Users and %g items' % (data.usernum, data.itemnum))
    print('average sequence length: %.2f' % (cc / len(data.user_train)))
    print('num_steps: ', num_steps)
    print('------------------------------\n')


    # make SASRec model object
    model = SASREC(item_num=data.itemnum,
                   seq_max_len=maxlen,
                   num_blocks=num_blocks,
                   embedding_dim=hidden_units,
                   attention_dim=hidden_units,
                   attention_num_heads=num_heads,
                   dropout_rate=dropout_rate,
                   conv_dims = [hidden_units, hidden_units],
                   l2_reg=l2_reg
    )

    # make batch sampler
    sampler = WarpSampler(data.User, data.usernum, data.itemnum, batch_size=batch_size, maxlen=maxlen, n_workers=multiprocessing.cpu_count())

    # start train
    print('Training start')
    print('------------------------------\n')
    model.train(
          data,
          sampler,
          num_epochs=num_epochs, 
          batch_size=batch_size, 
          lr=lr, 
          val_epoch=num_epochs if save_last_only else 1,
          val_target_user_n=1000, 
          target_item_n=-1,
          auto_save=True,
          path=save_path,
          exp_name=exp_name,
        )
    
    print('------------------------------')
    print('Training done')

    return model
