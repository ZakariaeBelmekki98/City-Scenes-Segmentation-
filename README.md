# City-Scenes-Segmentation-
This repo contains the code for a U-Net model with Redsidual Blocks for segmenting city scenes from the Cityscapes dataset. The model is based on the architecture adopted in the paper "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" [link]. 

## Model Description
The model uses Contracting blocks in the beginning, then a series of Residual Blocks and then a Expanding blocks. This architecture seems to be working better in this setting than a standard U-Net []. 

## Results
These images are sampled from the validation set (the model did not see them before)


![sample](https://github.com/ZakariaeBelmekki98/City-Scenes-Segmentation-/assets/110834462/cb32b082-c89a-4881-8e5f-4a5fb0a18627)
