# HF-LCZC
Hybrid Fusion for Local Climate Zone Classification

This repository contains two folders, namely, fusion-models and data.

The folder "fusion-models" contains different CNN models used for the fusion of Synthetic Aperture Radar (SAR) and Multispectral (MS) image data. Our fusion approaches, based on the CNN architecture, are applied to the Sen2LCZ42 dataset. After fusion, the fused features are run on the LCZ classifier, namely, the Sen2LCZ model. 

The folder "data" contains the final classification outputs from the Sen2LCZ model. This data has the classification output obtained from the fused Sentinel-1 and Sentinel-2 input from the Sen2LCZ42 dataset using the fusion models 

fusion-models/
-- model-fusion-feature.py           
-- model-fusion-hybrid-GaussianSmooth.py
-- model-fusion-hybrid-attention.py  
-- model-fusion-hybrid.py

data/
--  attent    
-- feat  
-- GF    
-- hybrid    
-- nofusion

a) model-fusion-feature.py: Feature-level fusion      

b) model-fusion-hybrid.py: Hybrid-level fusion (Pixel + Feature)    

c) model-fusion-hybrid-GaussianSmooth.py: Gaussian filtered smoothing integrated with hybrid-level fusion  

d) model-fusion-hybrid-attention.py: Hybrid-level fusion coupled with attention mechanisms
~
~
