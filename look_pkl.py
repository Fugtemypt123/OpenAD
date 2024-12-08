import pickle
path='/network_space/server127/shared/qinyiming/openad/data/full_shape_train_data.pkl'
# path='/network_space/server126/shared/yinshaofeng/Assignment/Cognition/openad/data/ysf_full_shape_val_data.pkl'
# path = '/network_space/server127/shared/qinyiming/openad/data/full_shape_val_data_10.pkl'

f=open(path,'rb')
data=pickle.load(f, encoding='latin1')

print(len(data))

# print(f'the length of data is {len(data)}')
# print(data[300]['semantic class'])

# new_data = []
# for i in range(6):
#     new_data.append(data[i*100])

# for i in range(11, 15):
#     new_data.append(data[i*100])

# for i in range(20, 22):
#     new_data.append(data[i*100])

# # save new_data
# path='/network_space/server127/shared/qinyiming/openad/data/full_shape_val_data_10.pkl'
# f=open(path,'wb')
# pickle.dump(new_data,f)
# f.close()
