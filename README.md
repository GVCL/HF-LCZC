# HF-LCZC
Hybrid Fusion for Local Climate Zone Classification

There will be two directories; named fusion-models and data.
The directory fusion-models contains the different fusion techniques used for the fusion of SAR & MS data. All fusion approaches are based on the CNN architecture. The fusion approaches are applied on the Sen2LCZ42 dataset. The final LCZ classifier used is Sen2LCZ model. The directory data contains the final classification oitput from Sen2LCZ model. The data will have the classification output obtained from fused Sentinel-1 and Sentinel-2 input using different fusion approaches. The LCZ classiifcation accuracy metrics for each of the method is given.


ls fusion-models/

model-fusion-feature.py           model-fusion-hybrid-GaussianSmooth.py
model-fusion-hybrid-attention.py  model-fusion-hybrid.py

ls data/

  attent    feat  GF    hybrid    nofusion

a)model-fusion-feature.py : Feature level fusion
b)model-fusion-hybrid.py  : Hybrid level fusion (Pixel + Feature)
c)model-fusion-hybrid-GaussianSmooth.py : Gaussian filtered smoothing integrated with Hybrid level fusion
d)model-fusion-hybrid-attention.py : Hybrid level fusion coupled with attention mechnanisms
~
~
