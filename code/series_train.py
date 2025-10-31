import os

train_list = ['2TS2WSMS_vary4', '2TS2WSMS_vary10', '2TS2WSMS_vary40', '2TS2WSMS_vary50',
              '2TS2WSMS_vary60', '2TS2WSMS_vary80', '2TS2WSMS_vary90', '2TS2WSMS_vary96', 
              '2TS2WSMS_vary98']
for train_name in train_list:
    os.system(f'python3 code/training.py --load_data_type {train_name}')
