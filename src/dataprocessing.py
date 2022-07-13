import pickle
import numpy as np

import pandas as pd
# data = pd.read_csv('didr.csv')

data = pd.read_csv('C:\\Users\\Administrator\\Downloads\\CTD_chemicals_diseases.csv')
# data=data.loc[(data['DirectEvidence']=='therapeutic') | (data['DirectEvidence']=='marker/mechanism')]
data=data.loc[(data['DirectEvidence']=='therapeutic')]
data=data.drop(labels=['CasRN','InferenceGeneSymbol','InferenceScore','OmimIDs','DirectEvidence','PubMedIDs','ChemicalName','DiseaseName'],axis=1)


data.to_csv('C:\\Users\\Administrator\\Desktop\\Chemical_Disease.csv', sep=" ",index=False)

with open('C:\\Users\\Administrator\\Desktop\\Chemical_Disease.csv', encoding='utf8') as f:
    drug_name = {}
    disease_name = {}
    drug = {}
    disease = {}
    m = 0
    n = 0
    for i in f:
        # line = i.replace('\n','').split("*")
        line = i.replace('\n','').split(" ")
        if line[0] not in drug.values():
            drug[m] = line[0]
            # drug[line[1]+'*'+line[0]] = m
            # drug_name[m] = line[0]
            # drug[m] = line[1]+'*'+line[0]
            m += 1
        if line[1] not in disease.values():
            disease[n] = line[1]
            # disease_name[n] = line[2]
            # disease[n] = line[3]+'*'+line[2]
            n += 1
    flag=0

