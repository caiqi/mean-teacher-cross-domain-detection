# Exploring Object Relation in Mean Teacher for Cross-Domain Detection
This is the implementation of '**Exploring Object Relation in Mean Teacher for Cross-Domain Detection**' [CVPR 2019]. The original paper can be found [here](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cai_Exploring_Object_Relation_in_Mean_Teacher_for_Cross-Domain_Detection_CVPR_2019_paper.pdf).

## Usage
1. Install [mxnet](https://github.com/apache/incubator-mxnet). The version we use is 1.4.0.
2. Prepare the dataset. We mainly follow the steps in [da-faster-rcnn](https://github.com/yuhuayc/da-faster-rcnn). 
3. Download the pre-trained models for [Foggy Cityscapes](https://github.com/caiqi/mean-teacher-cross-domain-detection/releases/download/v0.1/foggy_pretrain.params) and [SIM10k](https://github.com/caiqi/mean-teacher-cross-domain-detection/releases/download/v0.1/sim10k_pretrain.params). Then put them into models-foggy and models-sim10k. 
4. Train Foggy Cityscapes domain adaptation or SIM-10k domain adaptation:
    ```Shell
    ./train_foggy_final.sh  or ./train_sim10k_final.sh
5. The trained models for Foggy Cityscapes and SIM-10k are available at [foggy_final](https://github.com/caiqi/mean-teacher-cross-domain-detection/releases/download/v0.1/foggy_final.params) (mAP=0.351)and [sim10k_final](https://github.com/caiqi/mean-teacher-cross-domain-detection/releases/download/v0.1/sim10k_final.params) (mAP=0.466).

## Citation
If you find this code or model useful for your research, please cite our paper:

    @inproceedings{cai2019exploring,
      title={Exploring Object Relation in Mean Teacher for Cross-Domain Detection},
      author={Cai, Qi and Pan, Yingwei and Ngo, Chong-Wah and Tian, Xinmei and Duan, Lingyu and Yao, Ting},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      pages={11457--11466},
      year={2019}
    }
