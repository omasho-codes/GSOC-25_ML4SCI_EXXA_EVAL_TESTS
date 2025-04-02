# GSOC'25 ML4SCI EXXA1 EVALUATION TASKS

## Project - Equivariant Vision Networks for Predicting Planetary Systems' Architectures 

This repo contains solutions to evaluation tests of EXXA1 - (General Test + Images Based Test) 

I had trouble installing astropy in Colab , so I am sharing my notebook in ipynb and Kaggle Notebooks

## For detailed solution [here](https://github.com/omasho-codes/GSOC-25_ML4SCI_EXXA_EVAL_TESTS/blob/main/Detailed_solution%20(1).pdf)

## General-Test  [here](https://www.kaggle.com/code/ujjwallal/general-test)

## Image-Based-Test  [here](https://www.kaggle.com/code/suryatrainer/image-based-test)

## Brief Overview - 

Architecture Discussion :

              Simple 3 layer D(8) Equivariant Encoder-Decoder with
              ELU as non-linearity, 
              Group-pooling & Global Average Pooling for latent vectors, 
              Batch-Norm/MLPs, Multi-Scale Pyramid Feature Maps, LieConvs, didn't helped probably data limitaton.
              (Traditional AE failed miserably, here because even with augmentations, self organising property of 
              filters in deep CNNs is seen only with large , diverse datasets (here they were prone to overfit to 
              certain types of disks leading collapse of latent space - memorising over learning here I myself 
              introduced organised features maps with respect to group transformations of D(8) group which handled 
              pretty well given this small dataset. One thing to note though ImageNet Architectures (ConvNxt,EfficientNet) 
              did gave similar results but I discarded them as their equivariant properties come from mere augmentations of 
              the data that they have seen, meaning not true equivariance hence finetuning on this small data that they 
              havent seen would have much higher chanes of overfitting.


  ------------------------------------------------------------------------------------------------------------------------------------------------------------------
            
Model Architecure:

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
HyperParameters:  

                → criterions :  def focal_mse_loss(x_recon, x, gamma=2.0):
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
                                        
                → Train/Test Split : 10 Kfolds (1:9 - cv:train)
                → Loss =   orthogonality_loss + 2*focal_perceptual_Loss + focal_mse_reconstruction                         
                → optimizer = optim.Adam()
                → number of epochs= 100
                → batch_size = 16
                → image_size = 256*256
                → learning_rate =  0.01
                → scheduler = ReduceLROnPlateau

------------------------------------------------------------------------------------------------------------------------------------------------------------------
Results:

                →MSE : 0.091, 0.043 (single, ensemble x10) 
                →MSSSIM : 0.971 (ensemble x10)
                →For Cluster : Visual Inspection + t-SNE plot visualization

    
## LOSS CURVES

![WhatsApp Image 2025-04-01 at 11 24 36_e24d9cbd](https://github.com/user-attachments/assets/6eaaecd5-eab4-4a6e-8459-644894fa55ad)

![image](https://github.com/user-attachments/assets/f187cad8-49c5-4fd5-af75-3258caec4f54)

![image](https://github.com/user-attachments/assets/34a3b581-9988-48e2-bb16-71d7dc6ee111)

![WhatsApp Image 2025-04-01 at 11 23 54_919f4eca](https://github.com/user-attachments/assets/83ec2ce1-e508-4c5d-93aa-7ceb8af457b0)


## CLUSTER 0 : almost all keplerian motion following disks with some containing planets

![output_cluster_0 (5)](https://github.com/user-attachments/assets/9460422b-9183-492f-a91a-43dd45424299)




## CLUSTER 1 : mostly all images with kinks/disturbances (possible embedded planet / self-gravitating disk or other instabilites)   

![output_cluster_1 (7)](https://github.com/user-attachments/assets/92c2cf35-bba6-4c3e-9d66-65293cb73198)




## Some Reconstructions :

![image](https://github.com/user-attachments/assets/52f597c3-84f9-4f23-8636-2cb2f15595d8)


![image](https://github.com/user-attachments/assets/985a3a97-610e-4564-8875-5ab65520f723)


![image](https://github.com/user-attachments/assets/500fdfd7-1b96-4532-b4a0-5ca7f67a1703)


## Activation Maps of R2Conv 1:

Clearly highlighting , model does captures the kinks and disturbances present in these disks

![image](https://github.com/user-attachments/assets/7ceb1b86-002d-46fb-a427-05be366fe4eb)


 ## ॐ

    
  	


