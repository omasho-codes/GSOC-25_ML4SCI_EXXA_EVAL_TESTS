# <p align="center">ML4SCI_25 EXXA </p>

I had trouble installing astropy in Colab , so I am sharing my notebook in ipynb and Kaggle Notebooks

## General Task & Images Based Task :  Unsupervised Clustering of Disks & Autoencoder with accesible latent space

### QUICK LINKS   

| Resource Type                          | Description                                       | Link |
|----------------------------------------|---------------------------------------------------|------|
| **Jupyter Notebook for General Test**  | Code and analysis in a Notebook format.         | [Notebook](https://www.kaggle.com/code/ujjwallal/general-test) |
| **Jupyter Notebook for Image Test**    | PDF of the notebook.                             | [Notebook](https://www.kaggle.com/code/suryatrainer/image-based-test) |
| **Images & Model Weights**             | Images & Model weights for replication/testing.  | [Drive](https://drive.google.com/file/d/15tQuOqEWjS2GBQO5SOuGE7M6UHyGDg63/view?usp=drive_link) |

## NOTE: I have used same model(s) for the both tests, so I'll be discussing in one go only.


### How I approached the problem :

- Firstly, I tried many preprocessing techniques rather relying on pretrained Deep Vision Models for clustering, given a small dataset, which includes 

    - **Radial Intensity Profile** (see [here](https://github.com/user-attachments/assets/8fc3955b-3f49-4649-88f9-5d8caa4608ab)) , I computed elliptical radial intensity profiles of these images which did provided information about these kinks/turbulences and whole problem was bring down much smaller space from 2D to 1D frequency domain then Conv1d network with constrastive learning.

    - **Fourier Laplacian Filtering** (see [here](https://github.com/user-attachments/assets/c2ae6e32-ea82-450a-8266-affdd2369c6f))
 , also called Spectral Laplacian filtering and Difference of Gaussians to highlight edges and        globs.

- for extracting relevant information like kinks/disturbances in the images which carries sufficient information about presence of a planet while narrowing down my input domain / filtering out information like shape, disk structure, orientation which was "not good", However discontinued working on this as I felt this approach is a more "local forced solution" hovering to planet finding only rather than a more generalized global solution. 

- Then I moved on to a Deep Learning Approach, I used some slim CNN based autoencoders which on training from scratch performed really bad, no wonder why, but encoders pretrained on Image-Net (**Resnet-18, EfficientNet-B0, ConvNeXt-Tiny** based) did quite well but dropped down this approach, because:
    - Firstly, these extremly deep networks with such huge parameters had high chances of memorising global patterns over learning good latent representations - collapsing the latent space - overfitting (their clustering of latent vectors didn't seemed much uniform likely due to casual representations)
    
    - Lastly, they were **originally made for classfication** meaning they have learned to create **invariant feature maps** (to E(2) transformations) which would result in       less informative latent representations , using these architectures with heavy augmentations did reduced MSE but they felt less generasible and didn't offer a global        more  broader solution". Clustering using these gave results which looked similar to human eye rather capturing information about planets across differnt vareity of         disks.

    - For very few Images, when didnt use Perceptual Loss (see below for implementation) , I got few poor reconstructions , probably first image is a result of very grainy        image which was minority (so simply using mse would preseve overall structure) and secondly because most of the images had fading effect model mdemorised this global        pattern and apply in all images which in overall resulted a drop in MSE, choosing loss smartly was very much needed.
 
      ![image](https://github.com/user-attachments/assets/edb165df-755c-4a97-9d64-e24185e308ff)

### Final Selection of Architecture :

- Made a shallow & light **3-layer D(8)-equivariant encoder and decoder** with minimal number of feature channels with [escnn](https://github.com/QUVA-Lab/escnn) library to avoid collapsing latent space and result decoder memorising.

- This gave similar results in terms of reconstruction but now model was truly equivarint to D(8)-transformations (Flips+*rotations) while mimicking E(2)-equivariance providing much better latent representations.  

- Using group pooling and global average pooling I got latent vectors which are invariant to E(2) transformations while also towards affine group (probably because of agumentations i used), they gave much better latent vectors which on clustering gave more broader clsuters centred on planet presence rather than similar looking images.

- Tried adding residual connections, making multi-scale feature pyramid (like in IncpetionNet) but didn't help probably because of 
small dataset.

- For entire training (Self-Supervised based pretraining didn't help) used MSE and pretrained ResNet-18 perceptual loss for reconstruction and added a term (inspired from [ViCReg](https://arxiv.org/abs/2105.04906) paper) to encourage orthogonality of latent vectors to make full use of latent space.

- Equivariant AE , explicitly devises kernels for D(8)-group transformed version of feature maps not relying on self-organising nature of traditional vision networks when trained on huge diverse data with augmentations hence perfect for small datasets with known possible symmetries.


  ![image](https://github.com/user-attachments/assets/e10d2466-a620-4179-a1dc-f9948c2d943e)
  <p align="center">A p4 group Convolution

## Why Ensembling:

- Due to extremely small size of dataset , model was proned to overfit if right architectures and losses were not chosen , single models were finding global patterns that minimised overall loss over the dataset hence losing to capture all variety of disks. With ensembling , each model was an expert to its own training set , meaning an expert in his domain , so in average voting effects overfitting were nullified due to a collection of models (like a jury) giving much better results.

  ![image](https://github.com/user-attachments/assets/9911175d-dc1d-4d5d-ae07-c193925f6412)


- In Future, I will be trying training equivariant architectures on large datasets then using those trained models here, these may provide better results like ImageNet pretained non-equivariant models did, as now they more sense of processing of images.

# Results & Obeservations:

# Clustering

## Method :

- Using the group pooled vector and applying global average pooling to squeeze out the spatial dimensions to get latent vectors which I clustered through Spectral             Clustering from sk-learn, I got clusters which did to some extent cluster images that had clear planets and images with clearly no planets separately.

- One thing interesting to note was on affine transformation of any image , its cluster index did not change indicating invariant nature of latent vector with respect to affine transformations.

![image](https://github.com/user-attachments/assets/2468a806-998e-4f3d-9090-e41fa15b063a)

## Cluster 0 : 

![image](https://github.com/user-attachments/assets/a760effb-32c0-4d31-be8c-be30d6682e52)


## Analysis through Visual Inspection:

- We see all disks of type which follows keplerian motion are together irrespective of orientation angles and shape.
- Other than that we see images that are either bright and have bright spots probably planets.
- We also see few dark and ring like disks (outliers).
   
  
## CLUSTER 1 : mostly all images with kinks/disturbances (possible embedded planets / self-gravitating disk or other instabilites)

![image](https://github.com/user-attachments/assets/d296029f-b6a5-447d-8394-a42c1d543edd)


## Analysis through Visual Inspection:

- Dark Images with almost all containing kinks/turbulences and many containing planet looking like spots.
- We see very few keplerian motion following disks (outliers) in this cluster too
  
# Reconstructions :

## Note: My losses were focal losses instead of traditional MSE, instead I just to see all results in the notebook by reconstructing all outputs.
by main focus was on structure preservation, focusing on spots and eliminating some poor reconstrcutions (see above image). 
               
## Metrics :
            
                → MSE : 0.091, 0.043 (single, ensemble x10) 
                → MSSSIM : 0.971 (ensemble x10)
                → Perceptual Loss : 0.251

## Visualisation :

## t-SNE plot :

![image](https://github.com/user-attachments/assets/0c64d29e-52f8-4553-a8a0-543e2695d78f)


## Some Non-Trivial Reconstructions :

- One Important thing to note , when transforming input image with E(2) tranformations , their reconstructed images didn't predictably transformed as with equivariant architecture highlighting generalisbility of equivariance over augmentation with small datasets.

  
### <p align="center">By ConvNext-Tiny based autoencoder
![image](https://github.com/user-attachments/assets/c2bb62de-0d27-4166-9a93-1fa85e65c606)

### <p align="center">By slim D(8) equivariant architecture

![image](https://github.com/user-attachments/assets/c3e4b83c-3bf8-43e0-b605-1553304e2783)

### <p align="center">Clearly shows model prefer reconstructing spots , meaning they must be have been captured in latent representations.
![image](https://github.com/user-attachments/assets/860ce98e-040d-4f2a-aecc-e54bd7871ae7)


# Feature Maps Visualisation:

### Clearly highlighting , model does captures the kinks and disturbances present in these disks

![image](https://github.com/user-attachments/assets/22e435a1-9f8c-44c4-9f93-0dc77966a93d)

### <p align="center">In first few layers, model is capturing similar information like Fourier Laplacian Filtering got 

![image](https://github.com/user-attachments/assets/0b991d88-cda5-4b89-8c7c-12a904086460)

### <p align="center">In lower layers, it clearly shows spots are of great importance hence this information is travelling to latent space.

![image](https://github.com/user-attachments/assets/653adf53-79d5-4d2c-9a0d-49ca0a560cf6)



# Loss Curves

## The training curves indicate that careful monitoring of both train and validation losses is crucial to prevent overfitting and to choose the optimal model state for deployment.

![image](https://github.com/user-attachments/assets/b242c2e4-0824-4696-afbf-017693904a1b)


![image](https://github.com/user-attachments/assets/8be610a8-11c2-4a6b-a4ec-1b11b3f41a49)


![image](https://github.com/user-attachments/assets/0a7f7fe6-5ff6-4424-b8ee-373026f364af)


![image](https://github.com/user-attachments/assets/7ce36e3e-5939-4440-9c93-69bbad4031ce)



# Training Hyperparameters, Augmentations & losses

## Choice of Augmentations :

- To simulate, all possible camera orientations while imaging these disks in 3D, I tried below transformations but in training
  removed perspective transformation and non-uniform scaling because its effect was given by shearing.Also, added gaussian smoothing, to smoothen out some very grainy images
  
| **Camera Movement**                                      | **Transformation**         |
|---------------------------------------------------------|---------------------------|
| Rotate Camera                        | Rotation                  |
| Move Camera Forward/Backward                            | Scaling (Uniform)         |
| Move Camera in an Arc Around Object (Keeping Focus on Object) | Projective Transformation |
| Tilt Camera While Keeping It in the Same Position (Slanting View) | Shear               |
| Zoom Lens Unevenly (Stretching One Axis More Than the Other) | Non-Uniform Scaling  |


![image](https://github.com/user-attachments/assets/28de4446-f9ac-4d18-8080-94c3001b8bed)


## Losses Used :

    def focal_mse_loss(x_recon, x, gamma=2.0):
        mse = (x_recon - x) ** 2
        weights = torch.exp(-gamma * x.abs())
        return (weights * mse).mean()
  
  
    def orthogonality_loss(z):
        batch_size, dim = z.shape
        z = z - z.mean(dim=0, keepdim=True)
        cov_matrix = (z.T @ z) / (batch_size - 1)
        identity = torch.eye(dim, device=z.device)
        loss = torch.norm(cov_matrix * (1 - identity), p="fro")
        return loss
  
    class PerceptualLoss(nn.Module):
        def __init__(self):
            super(PerceptualLoss, self).__init__()
            self.feature_extractor = resnet18_feature_extractor()
            self.criterion = nn.L1Loss()
    
        def forward(self, x, y):
            x = x.repeat(1, 3, 1, 1)
            features_x = self.feature_extractor(x)
            y = y.repeat(1, 3, 1, 1)
            features_y = self.feature_extractor(y)
            loss = self.criterion(features_x, features_y)
            return loss

    → Loss =   orthogonality_loss + 2*focal_perceptual_Loss + focal_mse_reconstruction    
  
  
------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Hyperparameters :

                → Train/Test Split : 10 Kfolds (1:9 - cv:train)
                → Loss = orthogonality_loss + 2*focal_perceptual_Loss + focal_mse_reconstruction                         
                → optimizer = optim.Adam()
                → number of epochs= 100
                → batch_size = 16
                → image_size = 256*256
                → learning_rate =  0.01
                → scheduler = ReduceLROnPlateau
                → latent dim = 16

------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Model Architecture :

     →EquivariantAE(
          (conv1): R2Conv([D8_on_R2[(3.141592653589793, 8)]: {irrep_0,0 (x1)}(1)], [D8_on_R2[(3.141592653589793, 8)]: {regular (x4)}(64)], kernel_size=3, stride=2, padding=1)
          (relu1): ELU(alpha=1.0, inplace=True, type=[D8_on_R2[(3.141592653589793, 8)]: {regular (x4)}(64)])
          (conv2): R2Conv([D8_on_R2[(3.141592653589793, 8)]: {regular (x4)}(64)], [D8_on_R2[(3.141592653589793, 8)]: {regular (x8)}(128)], kernel_size=3, stride=2, padding=1)
          (relu2): ELU(alpha=1.0, inplace=True, type=[D8_on_R2[(3.141592653589793, 8)]: {regular (x8)}(128)])
          (conv3): R2Conv([D8_on_R2[(3.141592653589793, 8)]: {regular (x8)}(128)], [D8_on_R2[(3.141592653589793, 8)]: {regular (x16)}(256)], kernel_size=3, stride=2, padding=1)
          (relu3): ELU(alpha=1.0, inplace=True, type=[D8_on_R2[(3.141592653589793, 8)]: {regular (x16)}(256)])
          (group_pool): GroupPooling([D8_on_R2[(3.141592653589793, 8)]: {regular (x16)}(256)])
          (deconv1): R2ConvTransposed([D8_on_R2[(3.141592653589793, 8)]: {regular (x16)}(256)], [D8_on_R2[(3.141592653589793, 8)]: {regular (x8)}(128)], kernel_size=3, stride=2, padding=1)
          (drop1): FieldDropout()
          (relu4): ELU(alpha=1.0, inplace=True, type=[D8_on_R2[(3.141592653589793, 8)]: {regular (x8)}(128)])
          (deconv2): R2ConvTransposed([D8_on_R2[(3.141592653589793, 8)]: {regular (x8)}(128)], [D8_on_R2[(3.141592653589793, 8)]: {regular (x4)}(64)], kernel_size=3, stride=2, padding=1)
          (drop2): FieldDropout()
          (relu5): ELU(alpha=1.0, inplace=True, type=[D8_on_R2[(3.141592653589793, 8)]: {regular (x4)}(64)])
          (deconv3): R2ConvTransposed([D8_on_R2[(3.141592653589793, 8)]: {regular (x4)}(64)], [D8_on_R2[(3.141592653589793, 8)]: {irrep_0,0 (x1)}(1)], kernel_size=3, stride=2, padding=1)
          )


------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Other things that I tried :

- I tried contrastive learning for pre training because euclidean distances in latent space were large between similar disks at different inclination angles probably because they largely differ in shapes but it didn't help maybe because of maybe wrong hyperparameters (choice of augmentations, temperature, latent dim, small dataset etc), so in future I would like to try equivariances more groups like SIM(2) , PGL(3) etc to generalize even more because with augmentations to achieve equivariance is hard with small less diverse datasets and also low resolution images suffer loss in characteristics due to interpolations with such transformations like perspective transformations.

![image](https://github.com/user-attachments/assets/bfddc1d5-8291-48e0-8724-4a85b4f03342)

- Imagenet-pretrained encoders for autoencoder architecure, were slow to train but did perform better in reconstruction, but clustering on their latent vectors seemed non-    uniform , so I choose to have better latent representations over reconstructions. (MSE with ConvNext-Tiny and 4 layer decoder got me 0.0004 MSE)

- NOTE : Using AE with skip connections is totally destroying the concept of using bottleneck information but it ofcourse gives best MSEs , with skip connections it doesnt even need any information coming from reduced representations

## Future Work:

- Making pretrained equivariant architecutres rather than training from scracth.
  
- Focus on attention based architectures to seek performance gains to capture information most relevant to the task.

- Using Conditional Generative models to create more synthetic samples rather than relying on running expensive hydrodynamical simulations using Phantom code.


## References 

[Equivariance versus Augmentation for Spherical Images](http://arxiv.org/abs/2202.03990)

[VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906)

[General E(2)-Equivariant Steerable CNNs](https://arxiv.org/abs/1911.08251)

[Generalizing Convolutional Neural Networks for Equivariance to Lie Groups on Arbitrary Continuous Data](https://arxiv.org/abs/2002.12880)
  

