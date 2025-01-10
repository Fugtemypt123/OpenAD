# Learn the Concept of a Daily Object (DALF: Dual-stage Affordance Learning Framework for Concept Learning)


Official code for "DALF: Dual-stage Affordance Learning Framework for Concept Learning".

We present DALF for to enable a comprehensive understanding of object affordances.



## 1. Getting Started
We strongly encourage you to create a separate CONDA environment.
```
conda create -n openad python=3.8
conda activate openad
pip install -r requirements.txt
```
(shaofeng: 后面缺什么下什么就行)

## 2. Data
Download data from [this drive folder](https://drive.google.com/drive/folders/1f-_V_iA6POMYlBe2byuplJfdKmV72BHu?usp=sharing).

Currently, we support 2 models (OpenAD with backbones of [PointNet++](https://proceedings.neurips.cc/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf) and [DGCNN](https://dl.acm.org/doi/pdf/10.1145/3326362)) and 2 settings (full-shape and partial-view).

(shaofeng: 目前的数据放 `./data` 目录下，可以改)

## 3. Training
Please train the model on a single GPU for the best performance. Below are the steps for training the model with PointNet++ backbone on the full-shape setting, those of other combinations are equivalent.

* In ```config/openad_pn2/full_shape_cfg.py```, change the value of ```data_root``` to your downloaded data folder, and change the path to class weights to the path of the file ```full_shape_weights.npy``` (contained in the data folder).
* Assume you use the GPU 0, then run the following command to start training:

		CUDA_VISIBLE_DEVICES=0 python3 train.py --config ./config/openad_pn2/full_shape_cfg.py --work_dir ./log/openad_pn2/OPENAD_PN2_FULL_SHAPE_Release/ --gpu 0

## 4. Open-Vocabulary Testing
The followings are steps for open-vocabulary testing a trained model with PointNet++ backbone on the full-shape setting, those of other combinations are equivalent.

* Change the value of ```data_root``` in ```config/openad_pn2/full_shape_open_vocab_cfg.py``` to your downloaded data folder.
* Run the following command:

		CUDA_VISIBLE_DEVICES=0 python3 test_open_vocab.py --config ./config/openad_pn2/full_shape_open_vocab_cfg.py --checkpoint <path to your checkpoint model> --gpu 0
	Where ```<path to your checkpoint model>``` is your trained model.

We provide the pretrained models at [this drive](https://drive.google.com/drive/folders/17895vwgGHfIlDj3q0a7BOg6cotH5RTjm?usp=sharing).

(shaofeng: 目前的ckpt放 `./pretrain` 目录下，可以改)

## 5. Generate new data for the next steps (Completed by ysf)

To generate new data for the following steps, please paste your api key [here](https://github.com/Fugtemypt123/OpenAD/blob/main/utils/gpt.py#L7). and run:

```
python caption.py
```

It creates the pair of ```<point cloud, functionality (text prompt)>```. The data can be download [here](https://www.dropbox.com/scl/fo/288smvgyfjkce410ki2c4/ANthSlfqpMN_lpZvfkI1-Fs?rlkey=tvsr4qihekdi7yspp1eaydlgz&st=k0ur9df3&dl=0) and put under dir ```./data```.

## 6. Train CLPP (Contrastive Language-PointCloud Pre-trained)

To arrive the model that can align the semantics of a text prompt with the semantics of a set of point clouds of an object, the following steps leverage contrastive learning to finetune the pointnet++ encoder.

```
CUDA_VISIBLE_DEVICES=0 python3 train_clpp.py --config ./config/openad_pn2_clpp/clpp_full_shape_cfg.py --work_dir ./log/openad_pn2_clpp/OPENAD_PN2_CLPP/ --checkpoint <path to your checkpoint model> --gpu 0
```

Where ```<path to your checkpoint model>``` is your trained model in step 3.

## 7. Training-free method to rank multiple objects based on a query.

Following step provides a training-free method to rank multiple objects based on a query.

```
CUDA_VISIBLE_DEVICES=0 python3 rank_multi_obj.py --config ./config/openad_pn2/full_shape_open_vocab_cfg.py --checkpoint <path to your checkpoint model> --gpu 0 --query "It can contain some objects or water"
```

## 8. Multi-object evaluation.

Evaluate the model on test dataset created from `ysf_full_shape_val`, which gives the model `cand_num` objects and one ground-truth `functionality` prompt and asks the model which object best suits the prompt.

Each object is sampled from different categories to avoid ambiguity.

```bash
CUDA_VISIBLE_DEVICES=0 python3 eval_global.py --config ./config/openad_pn2_clpp/clpp_full_shape_open_vocab_cfg.py --checkpoint ${ckpt_path} --gpu 0 --test_num 50 --cand_num 5
```

## Citation

If you find our work useful for your research, please cite:
```
@inproceedings{Nguyen2023open,
      title={Open-vocabulary affordance detection in 3d point clouds},
      author={Nguyen, Toan and Vu, Minh Nhat and Vuong, An and Nguyen, Dzung and Vo, Thieu and Le, Ngan and Nguyen, Anh},
      booktitle = IROS,
      year      = 2023
}
```

## 9. Acknowledgement

Our source code is built with the heavy support from [3D AffordaceNet](https://github.com/Gorilla-Lab-SCUT/AffordanceNet). We express a huge thank to them.
