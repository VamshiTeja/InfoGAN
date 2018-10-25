# InfoGAN
Implementation of InfoGAN in Tensorflow

Generative Adversarial Netowrks for Unsupervised Learning
The goal is to unsupervised classification using Info-GAN's

### Environment Setup:
1) python 2.7
2) tensorflow 1.3
3) numpy
4) opencv
5) pandas
6) matplotlib
7) pickle
8) tqdm

## TRAINING
### 1)For Info-GAN:

run: python train_infogan_plus_w.py

This file doesnt take any arguments. You can change the hyperparameters by editing the file. This file supports training for both datasets.
You need to just change dataset argument(either 'kaggle' or 'mammo')


### 2)For VAE:

run: python train_vae.py

This file takes several optional arguments. 

example execution is 

python train_vae.py \
--training_epochs 100 \
--display_step 5 \
--checkpoint_step 5 \
--batch_size 32 \
--z_dim 1000 \
--learning_rate 0.0002 \
--dataset 'mammo' \
--imsize 256 \
--num_channels 3 \
--std 0.0 \
--restore 0


## TESTING 
To test resuts 
run python tester.py

This function takes no command line arguments. FUnctionality can be edited via hyperparameters defined at the beginning of the file.
