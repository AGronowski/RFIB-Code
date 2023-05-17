# Code for "Classification Utility, Fairness, and Compactness via Tunable Information Bottleneck and Rényi Measures"

This is the code for the paper "Classification Utility, Fairness, and Compactness
via Tunable Information Bottleneck and Rényi Measures"

Results can be reproduced by running main.py. This runs experiments using the RFIB method on the CelebA, FairFace, EyePACS, Adult, and COMPAS datasets.

CSV files with the labels and partitionings used are included but the image datasets need to be downloaded before the code can run.
After downloading the datasets, name the directories "eyepacs", "celeba", and "fairface" and place them in the "data" directory.

EyePACS dataset can be found at: https://www.kaggle.com/c/diabetic-retinopathy-detection/data

CelebA dataset can be found at: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

FairFace dataset can be found at: https://github.com/joojs/fairface

To install requirements
```
pip install -r requirements.txt
```

