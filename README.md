## ResNet (Deep Residual Networks) in TensorFlow

This code is modified from [tensorflow/models/resnet](https://github.com/tensorflow/models/tree/master/resnet), so that it can be applied to dataset composed of raw images instead of binary format. 

This network architecture is designed for cifar-10/cifar-100 dataset. If you want to apply it to your own dataset, you may need to modify network architecture as well as hyperparameters. See [Dataset](#dataset) for more datails.


### Related papers

- [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027v2.pdf)

- [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385v1.pdf)

- [Wide Residual Networks](https://arxiv.org/pdf/1605.07146v1.pdf)


### Dataset

If you want to apply this code to your own dataset, make sure bellow requirements for files are met.

- X_train.txt: Each line containing a path to an image for training.

- y_train.txt: Each line containing a label for corresponding example in X_train.txt.
&nbsp;

- X_val.txt (optional): Each line containing a path to an image for validation.

- y_val.txt (optional): Each line containing a label for corresponding example in X_val.txt.
&nbsp;

- infer.txt: Each line containing a path to an image for test.
&nbsp;

If you have dataset for testing, it's the same as validation process.

**You may find it help to have a look at these files in [data](https://github.com/watsonyanghx/ResNet_TensorFlow/tree/master/data) folder.**

> I used this code for Kaggle competition [CIFAR-10 - Object Recognition in Images | Kaggle](https://www.kaggle.com/c/cifar-10/data).
>
> Instead of using [Official CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html), the dataset from [CIFAR-10 - Object Recognition in Images | Kaggle](https://www.kaggle.com/c/cifar-10/data) is used.
> 
>   - Examples in train.7z is used for training and validation. 
> 
>   - Examples in test.7z is used for inference. 
> 
>   - Finally, the score given by this competition is taken as a measure of generalization ability of the trained model. 
> 
> Finally, I got about 92% accracy after about 40k training steps in the competition.


### Settings

* Random split training set into train/eval split.
* Pad and random crop. Horizontal flip. Per-image whitenting. 
* Momentum optimizer 0.9.
* Learning rate schedule: 0.1 (40k), 0.01 (60k), 0.001 (>60k).
* L2 weight decay: 0.002.
* Batch size: 128. (28-10 wide and 1001 layer bottleneck use 64)



**Note:** Change the code bellow in [resnet_main.py](https://github.com/watsonyanghx/ResNet_TensorFlow/blob/master/resnet/resnet_main.py#L241) based on your own setting.

```shell
 # Change values bellow based on your own setting.
  hps = resnet_model.HParams(batch_size=batch_size,
                              image_size=32,
                              depth=3,
                              num_classes=10,
                              min_lrn_rate=0.0001,
                              lrn_rate=0.1,
                              num_residual_units=5,
                              use_bottleneck=False,
                              weight_decay_rate=0.0002,
                              relu_leakiness=0.1,
                              optimizer='mom')
```


### Results

For more performance specifications, please visit [tensorflow/models/resnet](https://github.com/tensorflow/models/tree/master/resnet).

As mentioned, about 92% test accracy is achieved after about 40k training steps in [CIFAR-10 - Object Recognition in Images | Kaggle](https://www.kaggle.com/c/cifar-10/data).


### Prerequisite

1. Install TensorFlow.


### How to run

```shell
# cd to the your workspace.
# It contains resnet codes and cifar10 dataset.
# Note: User can split 5k from train set for eval set.
ls -R
  .:
  data  resnet

  ./data:
  train  infer  X_train.txt  X_val.txt  y_train.txt  y_val.txt  test.txt

  ./data/train:
  1.png  2.png  ...  50000.png

  ./data/infer:
  1.png  2.png  ...  300000.png

  ./resnet:
  cifar_input.py  helper.py  README.md  resnet_main.py  resnet_model.py


# Train the model.
python ./resnet/resnet_main.py --train_data_path=./data/X_train.txt \
                               --train_labels_path=./data/y_train.txt \
                               --log_root=./tmp/resnet_model \
                               --train_dir=./tmp/resnet_model/train \
                               --num_gpus=1


# While the model is training, you can also check on its progress using tensorboard:
tensorboard --logdir=./tmp/resnet_model

visit: http://127.0.0.1:6006/


# Evaluate the model.
# Avoid running on the same GPU as the training job at the same time,
# otherwise, you might run out of memory.
python ./resnet/resnet_main.py --eval_data_path=./data/X_val.txt \
                               --eval_labels_path=./data/y_val.txt \
                               --log_root=./tmp/resnet_model \
                               --eval_dir=./tmp/resnet_model/test \
                               --mode='eval' \
                               --num_gpus=1


# Inference
# Avoid running on the same GPU as the training job at the same time,
# otherwise, you might run out of memory.
python ./resnet/resnet_main.py --infer_data_path=./data/infer.txt \
                               --log_root=./tmp/resnet_model \
                               --mode='infer' \
                               --num_gpus=1
```

