# Exploring Object Relation in Mean Teacher for Cross-Domain Detection
This is the implementation of CVPR 2019 '**Exploring Object Relation in Mean Teacher for Cross-Domain Detection**'. The original paper can be found [here](https://arxiv.org/pdf/1904.11245.pdf). If you find it helpful for your research, please consider citing:

    @inproceedings{cai2019exploring,
      title={Exploring Object Relation in Mean Teacher for Cross-Domain Detection},
      author={Cai, Qi and Pan, Yingwei and Ngo, Chong-Wah and Tian, Xinmei and Duan, Lingyu and Yao, Ting},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      pages={11457--11466},
      year={2019}
    }

## Usage
1. Install [mxnet](https://github.com/apache/incubator-mxnet). The version we use is 1.4.0.
2. Prepare the dataset. We mainly follow the steps in [da-faster-rcnn](https://github.com/yuhuayc/da-faster-rcnn). 
3. Download the pre-trained models and put them into models-foggy and models-sim10k. 
4. Train Foggy Cityscape domain adaptation or SIM-10k domain adaptation:
    ```Shell
    ./train_foggy_final.sh  or ./train_sim10k_final.sh
5. The trained models for Foggy Cityscape and SIM-10k are available at.