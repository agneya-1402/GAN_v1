# GAN from Scratch in PyTorch

This project implements a Generative Adversarial Network (GAN) from scratch following the seminal research paper [Generative Adversarial Networks (arXiv:1406.2661v1)](https://arxiv.org/pdf/1406.2661v1). The goal is to train a GAN on the MNIST dataset to generate realistic handwritten digit images.

## Overview

A GAN consists of two neural networks competing in a zero-sum game:
- **Generator:** Learns to produce images from a latent noise vector.
- **Discriminator:** Learns to distinguish between real images (from the dataset) and fake images (produced by the generator).

This implementation uses PyTorch and follows the training procedure detailed in the research paper.

## Project Structure

- **main_1.py**: The main script for training the GAN.
- **images/**: Directory where generated image samples are saved during training.
- **data/mnist/**: Directory where the MNIST dataset is downloaded.
- **README.md**: This documentation file.

## Requirements

- Python 3.x
- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- numpy

Install the dependencies using pip:

```bash
pip install torch torchvision numpy
```
## Usage
Run the training script with default settings:

```bash
python train.py --n_epochs 200 --batch_size 64 --lr 0.0002 --latent_dim 100 --img_size 28 --channels 1 --sample_interval 400
```

### You can adjust the command-line arguments:

* --n_epochs: Number of epochs for training.
* --batch_size: Batch size.
* --lr: Learning rate for the Adam optimizer.
* --b1 and --b2: Adam optimizer momentum decay parameters.
* --latent_dim: Dimensionality of the noise vector.
* --img_size: Dimensions of each generated image.
* --channels: Number of image channels (1 for grayscale, as in MNIST).
* --sample_interval: Interval (in batches) at which generated image samples are saved.

## Implementation Details
### Generator
Architecture:
The generator is composed of a series of fully connected (linear) layers. Each block includes an optional batch normalization layer (except the first layer) and a LeakyReLU activation. The final layer outputs a vector that is reshaped into an image and passed through a Tanh activation to produce pixel values in the range 
[
−
1
,
1
]
[−1,1].

### Discriminator
Architecture:
The discriminator flattens the input image and processes it through a series of linear layers with LeakyReLU activations. It outputs a single probability value via a Sigmoid function indicating whether the input is a real image or a fake one.

### Loss Function
Both networks are optimized using Binary Cross-Entropy Loss (BCE Loss):

### Generator Loss: 
Measures how effectively the generator can fool the discriminator.

### Discriminator Loss: 
Measures the discriminator's ability to distinguish between real and generated images.


## Training Process
### Generator Training:

Sample noise from a normal distribution.
Generate images using the generator.
Calculate the loss by evaluating how well these images are classified as real by the discriminator.
Update the generator’s parameters to improve its ability to fool the discriminator.

### Discriminator Training:

Evaluate a batch of real images and the generated images.
Compute the loss for both sets (real images should be classified as real, generated ones as fake).
Update the discriminator’s parameters to enhance its classification performance.
Image Saving:

At regular intervals (as specified by --sample_interval), a set of generated images is saved to the images/ directory to monitor training progress.
## Results
During training, generated images are saved in the images/ directory. Over time, as training progresses, the generator should produce images that become increasingly realistic, mirroring the quality of the MNIST dataset.

## References
Generative Adversarial Networks (Goodfellow et al., 2014) – This paper introduces the concept of GANs and lays the foundation for the adversarial training process used in this project.
## License
This project is licensed under the MIT License.
