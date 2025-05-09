# HF-LCZC
Hybrid Fusion for Local Climate Zone Classification

This repository contains a folder 'Model'. The folder contains different fusion approaches for the fusion of Synthetic Aperture Radar (SAR) and Multispectral (MS) image data. Based on the CNN architecture, our fusion approaches are applied to the Sen2LCZ42 dataset. After fusion, the fused features are run on the LCZ classifier, namely, the Sen2LCZ model. 

The final classification output from the Sen2LCZ model is uploaded here https://huggingface.co/datasets/ancythomas/output/tree/main.  This output has the classification output obtained from the fused Sentinel-1 and Sentinel-2 input from the Sen2LCZ42 dataset using different fusion models.   

Model/ 

-- model-fusion-feature.py  

-- model-fusion-hybrid-GaussianSmooth.py

-- model-fusion-hybrid-attention.py  

-- model-fusion-hybrid.py

output/

-- _32_weights.best_nofusion.hdf5

-- _32_weights.best_feature.hdf5

-- _32_weights.best_hybrid.hdf5

-- _32_weights.best_attention.hdf5

--_32_weights.best_GF.hdf5



a) model-fusion-feature.py: Feature-level fusion      

b) model-fusion-hybrid.py: Hybrid-level fusion (Pixel + Feature)    

c) model-fusion-hybrid-GaussianSmooth.py: Gaussian filtered smoothing integrated with hybrid-level fusion  

d) model-fusion-hybrid-attention.py: Hybrid-level fusion coupled with attention mechanisms
~
~
