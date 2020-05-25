# Road-Crack-Detection

Classification of road damage using the pre-trained networks like ResNet, AlexNet and VGG16.
Also implemented two other CNN architectures from the following papers to compare the results
with pre-trained networks:

1) https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7533052
2) Deep on Edge: Opportunistic Road Damage Detection with Official City Vehicles

Latest Update (25-05-20):
 --> Resnet_Fusion.py
 Uses pre-trained Resnet weights and fuses the feature maps from each block of layer to gather lower and high level features. 
