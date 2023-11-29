# City-Scenes-Segmentation-
This repo contains the code for a U-Net model with Redsidual Blocks for segmenting city scenes from the Cityscapes dataset. The model is based on the architecture adopted in the paper "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" [link]. 

## Model Description and Setting
The model uses Contracting blocks in the beginning, then a series of Residual Blocks and then a Expanding blocks. This architecture seems to be working better in this setting than a standard U-Net []. 

The model was trained for 100 epochs using a batch size of 4. The Adam optimizer was used with a learning rate of 0.0002.

## Results
These images are sampled from the validation set (the model did not see them before). The first row has the input images, the second the ground truth segmentations, and the third contains the predictions of the model.


![sample](https://github.com/ZakariaeBelmekki98/City-Scenes-Segmentation-/assets/110834462/cb32b082-c89a-4881-8e5f-4a5fb0a18627)

## Running the Script
After cloning the code, one can run it using 

`
python Training.py <dataset_directory> 
`

it also comes with optional arguments such as the batch size and the learning rate. 
