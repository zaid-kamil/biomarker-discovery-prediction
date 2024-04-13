import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(data):
    # Dropping unnecessary columns
    data.drop(["Patient Id","Patient First Name","Family Name","Father's name","Location of Institute",
               "Institute Name","Test 1","Test 2","Test 3","Test 4","Test 5","Symptom 1","Symptom 2",
               "Symptom 3","Symptom 4","Symptom 5"], inplace=True, axis=1)

    # Handling missing values
    data.replace(["No record", "Not available", "None", "Not applicable", "-"], np.nan, inplace=True)
    data.fillna(data.mode().iloc[0], inplace=True)

    # Renaming columns
    data.rename(columns={"Patient Age": "Patient_Age", "Genes in mother's side": "Genes_Mother_Side",
                         "Paternal gene": "Paternal_Gene", "Blood cell count (mcL)": "Blood_Cell_mcL",
                         "Mother's age": "Mother_Age", "Father's age": "Father_Age",
                         "Respiratory Rate (breaths/min)": "Respiratory_Rate_Breaths_Min",
                         "Heart Rate (rates/min": "Heart_Rates_Min", "Parental consent": "Parental_Consent",
                         "Follow-up": "Follow_Up", "Birth asphyxia": "Birth_Asphyxia",
                         "Autopsy shows birth defect (if applicable)": "Autopsy_Birth_Defect",
                         "Place of birth": "Place_Birth", "Folic acid details (peri-conceptional)": "Folic_Acid",
                         "H/O serious maternal illness": "Maternal_Illness",
                         "H/O radiation exposure (x-ray)": "Radiation_Exposure",
                         "H/O substance abuse": "Substance_Abuse",
                         "Assisted conception IVF/ART": "Assisted_Conception",
                         "History of anomalies in previous pregnancies": "History_Previous_Pregnancies",
                         "No. of previous abortion": "Previous_Abortion", "Birth defects": "Birth_Defects",
                         "White Blood cell count (thousand per microliter)": "White_Blood_Cell",
                         "Blood test result": "Blood_Test_Result", "Genetic Disorder": "Genetic_Disorder",
                         "Disorder Subclass": "Disorder_Subclass"}, inplace=True)
    data.rename(columns={"Inherited from father": "Inherited_Father", "Maternal gene": "Maternal_Gene"}, inplace=True)
    data.sort_values(by=["Patient_Age"], inplace=True)
    return data


# save preprocess data
data = load_data("data/Train.csv")
data = preprocess_data(data)
print(data.shape)
data.to_csv("data/train_preprocessed.csv", index=False)

