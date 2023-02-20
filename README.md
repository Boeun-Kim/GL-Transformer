# GL-Transformer (ECCV 2022)

This is the official implementation of "Global-local Motion Transformer for Unsupervised Skeleton-based Action Learning (ECCV 2022)". [[paper]](https://arxiv.org/abs/2207.06101) [[project]](https://boeun-kim.github.io/)




![framework](https://github.com/Boeun-Kim/GL-Transformer/blob/main/figures/framework.jpg)



 ## Dependencies

We tested our code on the following environment.

- CUDA 11.3
- python 3.8.10
- pytorch 1.12.0

Install python libraries with:

```
pip install -r requirements.txt
```



## Data preparation

1. Download raw skeleton data from https://github.com/shahroudy/NTURGB-D to `./data/preprocessing/raw`

   - nturgbd_skeletons_s001_to_s017.zip
   - nturgbd_skeletons_s018_to_s032.zip

2. Download incomplete data list from  https://github.com/shahroudy/NTURGB-D to `./data/preprocessing/raw`

   - NTU_RGBD_samples_with_missing_skeletons.txt
   - NTU_RGBD120_samples_with_missing_skeletons.txt

3. Unzip the data

   ```
   cd ./data/preprocessing/raw
   unzip nturgbd_skeletons_s001_to_s017.zip
   unzip nturgbd_skeletons_s018_to_s032.zip -d nturgb+d120_skeletons
   ```

4. Preprocess the data

   ```
   cd ..
   python ntu60_gendata.py
   python ntu120_gendata.py
   python preprocess_ntu.py
   ```

 

## Unsupervised Pretraining

Sample arguments for unsupervised pretraining:

(please refer to `arguments.py` for detailed arguments.)

```
python learn_PTmodel.py \
    --train_data_path [train data path] --eval_data_path [eval data path] \
    --train_label_path [train label path] --eval_label_path [eval label path] \
    --save_path [save path] \
    --depth 4 --num_heads 8 \
    --intervals 1 5 10
```

Pretraining weights (weights-ntu*) can be downloaded via 

https://drive.google.com/drive/folders/10-UZ9BaijCJZZkB2R5rBxmj5cgXmRJ2E?usp=sharing



## Linear Evaluation Protocol

Sample arguments for training and evaluating a linear classifier:

(please refer to `arguments.py` for detailed arguments.)

```
python linear_eval_protocol.py \
    --train_data_path [train data path] --eval_data_path [eval data path] \
    --train_label_path [train label path] --eval_label_path [eval label path] \
    --save_path [save path] \
    --depth 4 --num_heads 8 \
    --pretrained_model [pretrained weight path]
```

Pretraining weights (w_classifier-ntu*) can be downloaded via 

https://drive.google.com/drive/folders/10-UZ9BaijCJZZkB2R5rBxmj5cgXmRJ2E?usp=sharing

Those files include weights of "GL_Transformer + linear classifier".



## Test for Action Recognition

Sample arguments for testing whole framework:

(please refer to `arguments.py` for detailed arguments.)

```
python test_actionrecog.py \
    --eval_data_path [eval data path] \
    --eval_label_path [eval label path] \
    --depth 4 --num_heads 8 \
    --pretrained_model_w_classifier [pretrained weight path(w. linear classifier)]
```



## Reference

Part of our code is based on [MS-G3D](https://github.com/kenziyuliu/MS-G3D), [CrosSCLR](https://github.com/LinguoLi/CrosSCLR), and [PoseFormer](https://github.com/zczcwh/PoseFormer).

Thanks to the great resources.



## Citation

Please cite our work if you find it useful.

```
@inproceedings{kim2022global,
  title={Global-local motion transformer for unsupervised skeleton-based action learning},
  author={Kim, Boeun and Chang, Hyung Jin and Kim, Jungho and Choi, Jin Young},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part IV},
  pages={209--225},
  year={2022},
  organization={Springer}
}
```
