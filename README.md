# Code for "Achieving Utility, Fairness and Compactness via Tunable Information Bottleneck Measures"

This is the code for the paper "Achieving Utility, Fairness and Compactness via Tunable Information Bottleneck Measures."

Results can be reproduced by running main.py that runs experiments using the RFIB method on the CelebA, FairFace, and EyePACs datasets.

CSV files with the labels and partitionings used are included but the datasets need to be downloaded before the code can run.
After downloading the datasets, name the directories "eyepacs", "celeba", and "fairface" and place them in the "data" directory.

EyePACs dataset can be found at: https://www.kaggle.com/c/diabetic-retinopathy-detection/data

CelebA dataset can be found at: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

FairFace dataset can be found at: https://github.com/joojs/fairface

To install requirements
```
pip install -r requirements.txt
```

