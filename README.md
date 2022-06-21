# NSGNN
## Paper "NSGNN： Predicting Drug-disease Associations Based on Neighborhood Subgraphs of Drug-disease Association Pairs"

### 'raw_dataset' directory
Contains Cdataset, DNdataset, Fdataset, LRSSL and Ndataset
(1) Fdataset
Wrname: the DrugBank IDs of drugs;
Wdname: the OMIM IDs of diseases;
drug: drug similarity matrix;
disease: disease similarity matrix;
didr: disease-drug association matrix.
1. Contain the prediction results of each baseline models is training on B-Dataset and F-Dataset
2. Please refer the code of LAGCN [here](https://github.com/storyandwine/LAGCN); please refer the code of DTINet [here](https://github.com/luoyunan/DTINet); please refer the code of deepDR [here](https://github.com/ChengF-Lab/deepDR).