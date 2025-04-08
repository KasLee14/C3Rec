# Breaking Knowledge Boundaries: Cognitive Distillation-enhanced Cross-Behavior Course Recommendation Model
This is our Pytorch implementation for the paper:

## Requirements
* OS: Ubuntu 18.04 or higher version
* python == 3.7.3 or above
* supported(tested) CUDA versions: 10.2
* Pytorch == 1.9.0 or above

## Code Structure
1. The entry script for training and evaluation is: [train.py](https://github.com/mysbupt/CrossCBR/blob/master/train.py).
2. The config file is: [config.yaml](https://github.com/mysbupt/CrossCBR/blob/master/config.yaml).
3. The script for data preprocess and dataloader: [utility.py](https://github.com/mysbupt/CrossCBR/blob/master/utility.py).
4. The model folder: [./models](https://github.com/mysbupt/CrossCBR/tree/master/models).
5. The experimental logs in tensorboard-format are saved in ./runs.
6. The experimental logs in txt-format are saved in ./log.
7. The best model and associate config file for each experimental setting is saved in ./checkpoints.

## How to run the code
Train C<sup>3</sup>Rec on the datasets with GPU 0: 

   > python train.py -g 0 -m CrossCBR -d MOOCCubeX -t YES
   > python train.py -g 0 -m CrossCBR -d Ednet -t YES

   You can specify the gpu id and the used dataset by cmd line arguments, while you can tune the hyper-parameters by revising the configy file [config.yaml](https://github.com/mysbupt/CrossCBR/blob/master/config.yaml). The detailed introduction of the hyper-parameters can be seen in the config file, and you are highly encouraged to read the paper to better understand the effects of some key hyper-parameters.
