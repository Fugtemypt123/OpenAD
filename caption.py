import os
import numpy as np
import pickle as pkl
from utils import get_completion

if __name__ == "__main__":
    train_dir = "data/full_shape_train_data.pkl"
    # val_dir = "data/full_shape_val_data.pkl"
    with open(train_dir, 'rb') as f:
        train_data = pkl.load(f)
    # with open(val_dir, 'rb') as f:
    #     val_data = pkl.load(f)
    train_affordance = ['grasp', 'contain', 'lift', 'openable', 'layable', 'sittable',
               'support', 'wrap_grasp', 'pourable', 'move', 'displaY', 'pushable', 'pull',
               'listen', 'wear', 'press', 'cut', 'stab', 'none']
    # val_affordance = ['grab', 'accommodate', 'raise', 'unlock', 'rest', 'take a seat', 'bear',
    #             'wrap', 'pour', 'reposition', 'demonstrate', 'push', 'drag', 'hear',
    #             'clothe', 'thumb', 'slice', 'jab', 'none']
    
    # with open("data/ysf_full_shape_train_data.pkl", 'rb') as f:
    #     ysf_data = pkl.load(f)
    ysf_data = []
    # train_data = train_data[len(ysf_data):]
    
    for _, info in enumerate(train_data):
        
        temp_info = {}
        temp_info["shape_id"] = info["shape_id"]
        temp_info["semantic class"] = info["semantic class"]
        temp_info["affordance"] = info["affordance"]
        temp_info["data_info"] = info["full_shape"]
        labels = temp_info["data_info"]["label"]

        affordance_dict = {}
        for i in range(len(labels)):
            affordance_id = int(labels[i][0])
            affordance = train_affordance[affordance_id]
            if affordance not in affordance_dict:
                affordance_dict[affordance] = 1
            else:
                affordance_dict[affordance] += 1

        affordance_dict = dict(sorted(affordance_dict.items(), key=lambda x: x[1], reverse=True)) 
        if affordance_dict.get("none") is not None:
            affordance_dict.pop("none")
        sum_value = sum(affordance_dict.values())

        for key in affordance_dict:
            affordance_dict[key] = round(affordance_dict[key] / sum_value, 1)
            affordance_dict[key] = str(affordance_dict[key]*100)   

        prompt = "Now I have an object {} and its affordances with their importance score: ".format(temp_info["semantic class"])
        for affordance, score in affordance_dict.items():
            prompt += affordance + "(" + score + "%), "
        prompt += '\n'
        prompt += "Please write two short sentences about different functionalities that you think this object has. In the format of ```\n1. [functionality sentence 1]\n2.[functionality sentence 2]\n```"
        
        retry = 0
        response = None
        while retry < 3:
            try:
                response = get_completion(prompt)
                responses = response.split("\n")
                functionality = [responses[0][2:].strip(), responses[1][2:].strip()]
                break
            except:
                retry += 1

        if response is None:
            continue
        temp_info["functionality"] = functionality
        ysf_data.append(temp_info)

        with open("data/ysf_full_shape_val_data.pkl", 'wb') as f:
            pkl.dump(ysf_data, f)
        print("Successfully processed {} datas".format(len(ysf_data)))
