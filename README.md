## MFMSNet: A Multi-frequency and Multi-scale Interactive CNN-Transformer Hybrid Network for Breast Ultrasound Image Segmentation
### 

### preparation of environment
We have tested our code in following environment：
* torch == 1.10.0
* torchvision == 0.12.0
* python == 3.7

### dataset
* BUSI: W. Al-Dhabyani, M. Gomaa, H. Khaled, and A. Fahmy, “Dataset of breast ultrasound images,” Data in brief, vol. 28, p. 104863, 2020.
* BUI:  A. A. Ardakani, A. Mohammadi, M. Mirza-Aghazadeh-Attari, and
U. R. Acharya, “An open-access breast lesion ultrasound image
database: Applicable in artificial intelligence studies,” Computers in
Biology and Medicine, vol. 152, p. 106438, 2023.
* DDTI:  L. Pedraza, C. Vargas, F. Narváez, O. Durán, E. Muñoz, and
E. Romero, “An open access thyroid ultrasound image database,” in
10th International symposium on medical information processing and
analysis, vol. 9287. SPIE, 2015, pp. 188–193.

### create model
My model is in `Oct_ResNet/ocytoc.py`, and model creation can be done using `oct_resnet50_fusion(pretrained=True,num_classes=1)`

Backbone's pre-training address (https://drive.google.com/file/d/18AZSJkgAe3SUh0AAAA7PPDSd8E30_Tsb/view?usp=sharing)

MCRUNet architecture overview.
![image](https://github.com/wrc990616/MFMSNet/blob/main/pic/figure1.pdf)

## Acknowledgments
Thanks to the open access BUSI、BUI and DDTI dataset for providing ultrasound data and annotation of breast nodules.
