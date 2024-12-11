import os
from os.path import join as opj
import numpy as np
from torch.utils.data import Dataset
import pickle as pkl


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m

class TestDataset(Dataset):
    def __init__(self, data_dir, cand_num=5, test_num=50, partial=False, split="test", ysf=True, seed=1):
        '''
        Build test dataset for evaluating CLPP trained model.
        TODO: Currently use the validation dataset as the test dataset, should switch to test dataset later.
        @params:
            - cand_num: number of candidate for a given query
            - partial: whether use partial view data
            - split: the split of the dataset
            - ysf: whether use YSF dataset
        @returns:
            - test_dataset: the test dataset
                - pc: (cand_num, N, 3)
                - query: str
                - gt: int
        '''
        np.random.seed(seed)
        print(f"Set the np.ramdom.seed in TestDataset to {seed}")
        self.split = split
        test_dataset = []
        assert split == "test", "Build testdataset should be used for test split"
        assert ysf is True, "Currently only support YSF dataset"
        assert partial is False, "Currently only support full shape data"
        with open(opj(data_dir, 'ysf_full_shape_val_data.pkl'), 'rb') as f:
            temp_data = pkl.load(f)

        label_groups = {}
        for infor in temp_data:
            label = infor["semantic class"]
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(infor)

        assert cand_num < len(label_groups.keys()), "Cannot sample more than the number of classes"

        for label, items in label_groups.items():
            print(f"Label: {label}, Number of items: {len(items)}")
        
        for _ in range(test_num): # len of the test dataset
            sampled_labels = np.random.choice(list(label_groups.keys()), cand_num, replace=False)
            sampled_items = [np.random.choice(label_groups[l]) for l in sampled_labels]
            pcs = [item["data_info"]["coordinate"].astype(np.float32) for item in sampled_items]
            pcs = [pc_normalize(pc)[0] for pc in pcs]
            pcs = np.stack(pcs, axis=0)
            gt_index = np.random.randint(cand_num)
            gt_item = sampled_items[gt_index]
            query = gt_item["functionality"][0]
            gt = gt_index
            test_dataset.append({"pc": pcs, "query": query, "gt_index": gt, "semantic class": gt_item["semantic class"]})
            # print(pcs.shape, query, gt, gt_item["semantic class"])
            # break
        self.test_dataset = test_dataset
    
    def __getitem__(self, index):
        data_dict = self.test_dataset[index]
        pc = data_dict["pc"]
        query = data_dict["query"]
        gt_index = data_dict["gt_index"]
        model_class = data_dict["semantic class"]
        return pc, query, gt_index, model_class
    
    def __len__(self):
        return len(self.test_dataset)

class AffordNetDataset(Dataset):
    def __init__(self, data_dir, split, partial=False, ysf=False):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.partial = partial
        self.ysf = ysf

        self.load_data()
        self.affordances = self.all_data[0]["affordance"]
        return

    def load_data(self):
        self.all_data = []

        if self.ysf:
            print("Use YSF dataset")
            if self.partial:
                with open(opj(self.data_dir, 'ysf_partial_view_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data = pkl.load(f)
            else:
                with open(opj(self.data_dir, 'ysf_full_shape_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data = pkl.load(f)
            for _, info in enumerate(temp_data):
                # print(info)
                # print(info.keys())
                # break
                if self.partial:
                    partial_info = info["partial"]
                    for view, data_info in partial_info.items():
                        temp_info = {}
                        temp_info["shape_id"] = info["shape_id"]
                        temp_info["semantic class"] = info["semantic class"]
                        temp_info["affordance"] = info["affordance"]
                        temp_info["view_id"] = view
                        temp_info["data_info"] = data_info
                        temp_info["functionality"] = info["functionality"][0]
                        self.all_data.append(temp_info)
                else:
                    temp_info = {}
                    temp_info["shape_id"] = info["shape_id"]
                    temp_info["semantic class"] = info["semantic class"]
                    temp_info["affordance"] = info["affordance"]
                    temp_info["data_info"] = info["data_info"]
                    temp_info["functionality"] = info["functionality"][0]
                    self.all_data.append(temp_info)

        else:
            print("Use non-YSF dataset")
            if self.partial:
                with open(opj(self.data_dir, 'partial_view_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data = pkl.load(f)
            else:
                with open(opj(self.data_dir, 'full_shape_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data = pkl.load(f)
            for _, info in enumerate(temp_data):
                if self.partial:
                    partial_info = info["partial"]
                    for view, data_info in partial_info.items():
                        temp_info = {}
                        temp_info["shape_id"] = info["shape_id"]
                        temp_info["semantic class"] = info["semantic class"]
                        temp_info["affordance"] = info["affordance"]
                        temp_info["view_id"] = view
                        temp_info["data_info"] = data_info
                        self.all_data.append(temp_info)
                else:
                    temp_info = {}
                    temp_info["shape_id"] = info["shape_id"]
                    temp_info["semantic class"] = info["semantic class"]
                    temp_info["affordance"] = info["affordance"]
                    temp_info["data_info"] = info["full_shape"]
                    self.all_data.append(temp_info)

    def __getitem__(self, index):

        data_dict = self.all_data[index]
        modelid = data_dict["shape_id"]
        modelcat = data_dict["semantic class"]

        data_info = data_dict["data_info"]
        model_data = data_info["coordinate"].astype(np.float32)
        labels = data_info["label"]
        
        temp = labels.astype(np.float32).reshape(-1, 1)
        model_data = np.concatenate((model_data, temp), axis=1)

        datas = model_data[:, :3]
        targets = model_data[:, 3:]

        datas, _, _ = pc_normalize(datas)

        if self.ysf:
            functionality = data_dict["functionality"]
            return datas, datas, targets, modelid, modelcat, functionality

        return datas, datas, targets, modelid, modelcat

    def __len__(self):
        return len(self.all_data)