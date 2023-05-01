# EcgenGAN
GANs for generating 2s of ECG data


# How to use
Each subarchitecture has its own subdirectory. simply navigate into one of them and call the python script.
Just make sure to provide a dataset under the specified paths. Any physionet database should do, provided
they are cut into 2s intervals. For reproducibility, I can offer the synthetic database I used, just shoot me a message.

# Architectures and Outputs

## Base GAN
![BaseGAN Generator](https://github.com/jackilion/EcgenGAN/blob/master/img/BaseGAN_Generator.png)
![BaseGAN Discriminator](https://github.com/jackilion/EcgenGAN/blob/master/img/BaseGAN_Discriminator.png)
![BaseGAN output](https://github.com/jackilion/EcgenGAN/blob/master/img/baseGAN_output.png)

## infoGAN
![infoGAN Generator](https://github.com/jackilion/EcgenGAN/blob/master/img/InfoGAN_Generator.png)
![infoGAN Discriminator](https://github.com/jackilion/EcgenGAN/blob/master/img/InfoGAN_Discriminator.png)
![infoGAN output](https://github.com/jackilion/EcgenGAN/blob/master/img/epoch_60_cat_1.png)
![infoGAN output](https://github.com/jackilion/EcgenGAN/blob/master/img/epoch_60_cat_2.png)
![infoGAN output](https://github.com/jackilion/EcgenGAN/blob/master/img/epoch_60_cat_3.png)

## cGAN
![cGAN Generator](https://github.com/jackilion/EcgenGAN/blob/master/img/cGAN_Generator.png)
![cGAN Discriminator](https://github.com/jackilion/EcgenGAN/blob/master/img/cGAN_Discriminator.png)
![cGAN output](https://github.com/jackilion/EcgenGAN/blob/master/img/cGAN_output.png)
