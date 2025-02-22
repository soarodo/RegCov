# RegCov
This is an code implementation of the paper **Fast Normalization for Bilinear Pooling via Eigenvalue Regularization**

## Usage

### Download
Please donwnload the checkpoint(link) at the first stage and put it in the ~/

### Data preparation
Download and extract ImageNet train and val images from http://image-net.org/. The directory structure is the standard layout for the torchvision datasets.ImageFolder, and the training and validation data is expected to be in the train/ folder and val/ folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

### Evaluation
To evaluate the Dropcov method, run **train.sh**

