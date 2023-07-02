# MEnKF_Ensembler_DualGCN_DeepCDR
Matrix Ensemble Kalman Filter Model Averaging for DualGCN and DeepCDR Features

Please follow the following steps to reproduce the results in the manuscript

1. Download the [DualGCN Embeddings](https://drive.google.com/drive/folders/1Cree-pkbQ_UxBF4pXaNoaf6TnYnQyAKr?usp=drive_link) and [DeepCDR Embeddings](https://drive.google.com/drive/folders/1W8aIdWcW_yeaXwajWOcWXMH2RQHP0FWc?usp=drive_link) and place them in the folders [DualGCN](https://github.com/Ved-Piyush/MEnKF_Ensembler_DualGCN_DeepCDR/tree/main/DualGCN) and [DeepCDR](https://github.com/Ved-Piyush/MEnKF_Ensembler_DualGCN_DeepCDR/tree/main/DeepCDR), respectively.

2. Run the script [Read_Data_DeepCDR_DualGCN.ipynb](https://github.com/Ved-Piyush/MEnKF_Ensembler_DualGCN_DeepCDR/blob/main/Data_Preprocessing_Scripts/Read_Data_DeepCDR_DualGCN.ipynb) that will reindex the [DualGCN Embeddings](https://drive.google.com/drive/folders/1Cree-pkbQ_UxBF4pXaNoaf6TnYnQyAKr?usp=drive_link) and [DeepCDR Embeddings](https://drive.google.com/drive/folders/1W8aIdWcW_yeaXwajWOcWXMH2RQHP0FWc?usp=drive_link) to have same `Cell_Line` IDs and `Drug_ID` along the rows for the training and the testing datasets. The reindexed files will be placed in the [Data](https://github.com/Ved-Piyush/MEnKF_Ensembler_DualGCN_DeepCDR/tree/main/Data) folder.

3. Run the script [Reduce_Dimensions.ipynb](https://github.com/Ved-Piyush/MEnKF_Ensembler_DualGCN_DeepCDR/blob/main/Data_Preprocessing_Scripts/Reduce_Dimensions.ipynb) which reduces the dimensionality of the embeddings by using Principal Component Analysis. The Principal Component Embeddings will be placed in the [Data](https://github.com/Ved-Piyush/MEnKF_Ensembler_DualGCN_DeepCDR/tree/main/Data) folder. These Principal Component Embeddings would be used as features to the MEnKF method. 
