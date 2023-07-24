# MEnKF_Ensembler_DualGCN_DeepCDR
Matrix Ensemble Kalman Filter Model Averaging for DualGCN and DeepCDR Features

Please follow the following steps to reproduce the results in the manuscript

1. Download the [DualGCN Embeddings](https://drive.google.com/drive/folders/1Cree-pkbQ_UxBF4pXaNoaf6TnYnQyAKr?usp=drive_link) and [DeepCDR Embeddings](https://drive.google.com/drive/folders/1W8aIdWcW_yeaXwajWOcWXMH2RQHP0FWc?usp=drive_link) and place them in the folders [DualGCN](https://github.com/Ved-Piyush/MEnKF_Ensembler_DualGCN_DeepCDR/tree/main/DualGCN) and [DeepCDR](https://github.com/Ved-Piyush/MEnKF_Ensembler_DualGCN_DeepCDR/tree/main/DeepCDR), respectively.

2. Run the script [Read_Data_DeepCDR_DualGCN.ipynb](https://github.com/Ved-Piyush/MEnKF_Ensembler_DualGCN_DeepCDR/blob/main/Data_Preprocessing_Scripts/Read_Data_DeepCDR_DualGCN.ipynb) that will reindex the [DualGCN Embeddings](https://drive.google.com/drive/folders/1Cree-pkbQ_UxBF4pXaNoaf6TnYnQyAKr?usp=drive_link) and [DeepCDR Embeddings](https://drive.google.com/drive/folders/1W8aIdWcW_yeaXwajWOcWXMH2RQHP0FWc?usp=drive_link) to have same `Cell_Line` IDs and `Drug_ID` along the rows for the training and the testing datasets. The reindexed files will be placed in the [Data](https://github.com/Ved-Piyush/MEnKF_Ensembler_DualGCN_DeepCDR/tree/main/Data) folder.

3. Run the script [Reduce_Dimensions_Higher_PCs.ipynb](https://github.com/Ved-Piyush/MEnKF_Ensembler_DualGCN_DeepCDR/blob/main/Data_Preprocessing_Scripts/Reduce_Dimensions_Higher_PCs.ipynb) which reduces the dimensionality of the embeddings by using Principal Component Analysis. The Principal Component Embeddings will be placed in the [Data](https://github.com/Ved-Piyush/MEnKF_Ensembler_DualGCN_DeepCDR/tree/main/Data) folder. These Principal Component Embeddings would be used as features to the MEnKF method.

4. The MEnKF method is evaluated using embeddings from individual DualGCN, DeepCDR, and combined DeepCDR +DualGCN models, respectively. The MEnKF algorithm is run till the training RMSE does not improve for 10 successive updates. The scripts to run MEnKF with the various embedding configurations can be found below:
5. 
   a. DeepCDR drug and multi-omics embeddings - [MEnKF_DeepCDR_Sequential_Injection_Plots_All_Data.ipynb
](https://github.com/Ved-Piyush/MEnKF_Ensembler_DualGCN_DeepCDR/blob/main/MEnKF_Scripts/MEnKF_DeepCDR_Sequential_Injection_Plots_All_Data.ipynb) \

   b. DualGCN drug and multi-omics embeddings - [MEnKF_DualGCN_Sequential_Injection_Plots_All_Data.ipynb
](https://github.com/Ved-Piyush/MEnKF_Ensembler_DualGCN_DeepCDR/blob/main/MEnKF_Scripts/MEnKF_DualGCN_Sequential_Injection_Plots_All_Data.ipynb) \

   c. Combined DualGCN and DeepCDR drug and multi-omic embeddings - [MEnKF_DualGCN_DeepCDR_Sequential_Injection_Plots_All_Train.ipynb](https://github.com/Ved-Piyush/MEnKF_Ensembler_DualGCN_DeepCDR/blob/main/MEnKF_Scripts/MEnKF_DualGCN_DeepCDR_Sequential_Injection_Plots_All_Train.ipynb) \

6. The script for making the plots in the paper is at [Make_Plots_All_Train.ipynb
](https://github.com/Ved-Piyush/MEnKF_Ensembler_DualGCN_DeepCDR/blob/main/MEnKF_Scripts/Make_Plots_All_Train.ipynb)
