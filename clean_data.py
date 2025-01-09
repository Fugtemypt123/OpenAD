import pickle

path = './data/ysf_full_shape_val_data.pkl'

f=open(path,'rb')
data=pickle.load(f, encoding='latin1')

clean_data = []

for i in range(len(data)):
    new_line = {}
    new_line['shape_id'] = data[i]['shape_id']
    new_line['semantic class'] = data[i]['semantic class']
    obj_class = data[i]['semantic class']
    new_line['affordance'] = data[i]['affordance']
    data_info = {}
    data_info['coordinate'] = data[i]['data_info']['coordinate']
    data_info['label'] = data[i]['data_info']['label']
    new_line['data_info'] = data_info
    func = []
    for f in data[i]['functionality']:
        sentence = f.replace("The "+obj_class, "It").replace(obj_class, "it")
        func.append(sentence)
    new_line['functionality'] = func

    clean_data.append(new_line)

# save clean_data
path='./data/ysf_full_shape_val_data_clean.pkl'
f=open(path,'wb')
pickle.dump(clean_data,f)
f.close()
