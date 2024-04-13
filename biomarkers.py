import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import missingno as msno
import statsmodels.stats.api as sms
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import levene
from scipy.stats import shapiro
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import scale
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.manifold import Isomap,TSNE
from sklearn.feature_selection import mutual_info_classif
from tqdm.notebook import tqdm
from scipy.stats import ttest_ind
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
import scipy.stats as stats
from dataprep.eda import *
from dataprep.eda import plot
from dataprep.eda import plot_diff
from dataprep.eda import plot_correlation
from dataprep.eda import plot_missing
import plotly.figure_factory as ff
from collections import Counter
import pandas_profiling as pp
# basemap
from mpl_toolkits.basemap import Basemap
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go

filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=UserWarning)

path = "data/train.csv"

GENOM_GENETICS = pd.read_csv(path)
Data = GENOM_GENETICS.copy()

Data

Data.drop(["Patient Id","Patient First Name","Family Name","Father's name","Location of Institute",
          "Institute Name","Test 1","Test 2","Test 3","Test 4","Test 5","Symptom 1","Symptom 2",
           "Symptom 3","Symptom 4","Symptom 5"],inplace=True,axis=1)

Data["Birth asphyxia"] = Data["Birth asphyxia"].replace("No record",np.NaN)
Data["Birth asphyxia"] = Data["Birth asphyxia"].replace("Not available",np.NaN)

Data["Autopsy shows birth defect (if applicable)"] = Data["Autopsy shows birth defect (if applicable)"].replace("None",np.NaN)
Data["Autopsy shows birth defect (if applicable)"] = Data["Autopsy shows birth defect (if applicable)"].replace("Not applicable",np.NaN)

Data["H/O radiation exposure (x-ray)"] = Data["H/O radiation exposure (x-ray)"].replace("Not applicable",np.NaN)
Data["H/O radiation exposure (x-ray)"] = Data["H/O radiation exposure (x-ray)"].replace("-",np.NaN)

Data["H/O substance abuse"] = Data["H/O substance abuse"].replace("Not applicable",np.NaN)
Data["H/O substance abuse"] = Data["H/O substance abuse"].replace("-",np.NaN)

Data.rename(columns={"Patient Age":"Patient_Age",
                    "Genes in mother's side":"Genes_Mother_Side",
                    "Paternal gene":"Paternal_Gene",
                    "Blood cell count (mcL)":"Blood_Cell_mcL",
                    "Mother's age":"Mother_Age",
                    "Father's age":"Father_Age",
                    "Respiratory Rate (breaths/min)":"Respiratory_Rate_Breaths_Min",
                    "Heart Rate (rates/min":"Heart_Rates_Min",
                    "Parental consent":"Parental_Consent",
                    "Follow-up":"Follow_Up",
                    "Birth asphyxia":"Birth_Asphyxia",
                    "Autopsy shows birth defect (if applicable)":"Autopsy_Birth_Defect",
                    "Place of birth":"Place_Birth",
                    "Folic acid details (peri-conceptional)":"Folic_Acid",
                    "H/O serious maternal illness":"Maternal_Illness",
                    "H/O radiation exposure (x-ray)":"Radiation_Exposure",
                    "H/O substance abuse":"Substance_Abuse",
                    "Assisted conception IVF/ART":"Assisted_Conception",
                    "History of anomalies in previous pregnancies":"History_Previous_Pregnancies",
                    "No. of previous abortion":"Previous_Abortion",
                    "Birth defects":"Birth_Defects",
                    "White Blood cell count (thousand per microliter)":"White_Blood_Cell",
                    "Blood test result":"Blood_Test_Result",
                    "Genetic Disorder":"Genetic_Disorder",
                    "Disorder Subclass":"Disorder_Subclass"},inplace=True)

Data.rename(columns={"Inherited from father":"Inherited_Father",
                    "Maternal gene":"Maternal_Gene"},inplace=True)

Data.sort_values(by=["Patient_Age"],inplace=True)

SAMPLE_ONE = Data[:1000]
SAMPLE_TWO = Data[4400:5600]
SAMPLE_THREE = Data[3453:6000]
SAMPLE_FOUR = Data[10000:13000]

SAMPLE_ONE = SAMPLE_ONE.reset_index(drop=True)
SAMPLE_TWO = SAMPLE_TWO.reset_index(drop=True)
SAMPLE_THREE = SAMPLE_THREE.reset_index(drop=True)
SAMPLE_FOUR = SAMPLE_FOUR.reset_index(drop=True)

SAMPLE_ONE["Inherited_Father"].fillna(SAMPLE_ONE["Inherited_Father"].mode()[0], inplace=True)
SAMPLE_ONE["Maternal_Gene"].fillna(SAMPLE_ONE["Maternal_Gene"].mode()[0], inplace=True)
SAMPLE_ONE["Respiratory_Rate_Breaths_Min"].fillna(SAMPLE_ONE["Respiratory_Rate_Breaths_Min"].mode()[0], inplace=True)
SAMPLE_ONE["Heart_Rates_Min"].fillna(SAMPLE_ONE["Heart_Rates_Min"].mode()[0], inplace=True)
SAMPLE_ONE["Follow_Up"].fillna(SAMPLE_ONE["Follow_Up"].mode()[0], inplace=True)
SAMPLE_ONE["Gender"].fillna(SAMPLE_ONE["Gender"].mode()[0], inplace=True)
SAMPLE_ONE["Birth_Asphyxia"].fillna(SAMPLE_ONE["Birth_Asphyxia"].mode()[0], inplace=True)
SAMPLE_ONE["Autopsy_Birth_Defect"].fillna(SAMPLE_ONE["Autopsy_Birth_Defect"].mode()[0], inplace=True)
SAMPLE_ONE["Place_Birth"].fillna(SAMPLE_ONE["Place_Birth"].mode()[0], inplace=True)
SAMPLE_ONE["Folic_Acid"].fillna(SAMPLE_ONE["Folic_Acid"].mode()[0], inplace=True)
SAMPLE_ONE["Maternal_Illness"].fillna(SAMPLE_ONE["Maternal_Illness"].mode()[0], inplace=True)
SAMPLE_ONE["Radiation_Exposure"].fillna(SAMPLE_ONE["Radiation_Exposure"].mode()[0], inplace=True)
SAMPLE_ONE["Substance_Abuse"].fillna(SAMPLE_ONE["Substance_Abuse"].mode()[0], inplace=True)
SAMPLE_ONE["Assisted_Conception"].fillna(SAMPLE_ONE["Assisted_Conception"].mode()[0], inplace=True)
SAMPLE_ONE["History_Previous_Pregnancies"].fillna(SAMPLE_ONE["History_Previous_Pregnancies"].mode()[0], inplace=True)
SAMPLE_ONE["Birth_Defects"].fillna(SAMPLE_ONE["Birth_Defects"].mode()[0], inplace=True)
SAMPLE_ONE["Blood_Test_Result"].fillna(SAMPLE_ONE["Blood_Test_Result"].mode()[0], inplace=True)


SAMPLE_ONE["Mother_Age"].fillna(SAMPLE_ONE.groupby(["Disorder_Subclass"])["Mother_Age"].transform("mean"),inplace=True)
SAMPLE_ONE["Father_Age"].fillna(SAMPLE_ONE.groupby(["Disorder_Subclass"])["Father_Age"].transform("mean"),inplace=True)
SAMPLE_ONE["Previous_Abortion"].fillna(SAMPLE_ONE.groupby(["Disorder_Subclass"])["Previous_Abortion"].transform("mean"),inplace=True)
SAMPLE_ONE["White_Blood_Cell"].fillna(SAMPLE_ONE.groupby(["Disorder_Subclass"])["White_Blood_Cell"].transform("mean"),inplace=True)

SAMPLE_ONE.dropna(inplace=True,axis=0)

SAMPLE_TWO["Inherited_Father"].fillna(SAMPLE_TWO["Inherited_Father"].mode()[0], inplace=True)
SAMPLE_TWO["Maternal_Gene"].fillna(SAMPLE_TWO["Maternal_Gene"].mode()[0], inplace=True)
SAMPLE_TWO["Respiratory_Rate_Breaths_Min"].fillna(SAMPLE_TWO["Respiratory_Rate_Breaths_Min"].mode()[0], inplace=True)
SAMPLE_TWO["Heart_Rates_Min"].fillna(SAMPLE_TWO["Heart_Rates_Min"].mode()[0], inplace=True)
SAMPLE_TWO["Follow_Up"].fillna(SAMPLE_TWO["Follow_Up"].mode()[0], inplace=True)
SAMPLE_TWO["Gender"].fillna(SAMPLE_TWO["Gender"].mode()[0], inplace=True)
SAMPLE_TWO["Birth_Asphyxia"].fillna(SAMPLE_TWO["Birth_Asphyxia"].mode()[0], inplace=True)
SAMPLE_TWO["Autopsy_Birth_Defect"].fillna(SAMPLE_TWO["Autopsy_Birth_Defect"].mode()[0], inplace=True)
SAMPLE_TWO["Place_Birth"].fillna(SAMPLE_TWO["Place_Birth"].mode()[0], inplace=True)
SAMPLE_TWO["Folic_Acid"].fillna(SAMPLE_TWO["Folic_Acid"].mode()[0], inplace=True)
SAMPLE_TWO["Maternal_Illness"].fillna(SAMPLE_TWO["Maternal_Illness"].mode()[0], inplace=True)
SAMPLE_TWO["Radiation_Exposure"].fillna(SAMPLE_TWO["Radiation_Exposure"].mode()[0], inplace=True)
SAMPLE_TWO["Substance_Abuse"].fillna(SAMPLE_TWO["Substance_Abuse"].mode()[0], inplace=True)
SAMPLE_TWO["Assisted_Conception"].fillna(SAMPLE_TWO["Assisted_Conception"].mode()[0], inplace=True)
SAMPLE_TWO["History_Previous_Pregnancies"].fillna(SAMPLE_TWO["History_Previous_Pregnancies"].mode()[0], inplace=True)
SAMPLE_TWO["Birth_Defects"].fillna(SAMPLE_TWO["Birth_Defects"].mode()[0], inplace=True)
SAMPLE_TWO["Blood_Test_Result"].fillna(SAMPLE_TWO["Blood_Test_Result"].mode()[0], inplace=True)

SAMPLE_TWO["Mother_Age"].fillna(SAMPLE_TWO.groupby(["Disorder_Subclass"])["Mother_Age"].transform("mean"),inplace=True)
SAMPLE_TWO["Father_Age"].fillna(SAMPLE_TWO.groupby(["Disorder_Subclass"])["Father_Age"].transform("mean"),inplace=True)
SAMPLE_TWO["Previous_Abortion"].fillna(SAMPLE_TWO.groupby(["Disorder_Subclass"])["Previous_Abortion"].transform("mean"),inplace=True)
SAMPLE_TWO["White_Blood_Cell"].fillna(SAMPLE_TWO.groupby(["Disorder_Subclass"])["White_Blood_Cell"].transform("mean"),inplace=True)

SAMPLE_TWO.dropna(inplace=True,axis=0)

SAMPLE_THREE["Inherited_Father"].fillna(SAMPLE_THREE["Inherited_Father"].mode()[0], inplace=True)
SAMPLE_THREE["Maternal_Gene"].fillna(SAMPLE_THREE["Maternal_Gene"].mode()[0], inplace=True)
SAMPLE_THREE["Respiratory_Rate_Breaths_Min"].fillna(SAMPLE_THREE["Respiratory_Rate_Breaths_Min"].mode()[0], inplace=True)
SAMPLE_THREE["Heart_Rates_Min"].fillna(SAMPLE_THREE["Heart_Rates_Min"].mode()[0], inplace=True)
SAMPLE_THREE["Follow_Up"].fillna(SAMPLE_THREE["Follow_Up"].mode()[0], inplace=True)
SAMPLE_THREE["Gender"].fillna(SAMPLE_THREE["Gender"].mode()[0], inplace=True)
SAMPLE_THREE["Birth_Asphyxia"].fillna(SAMPLE_THREE["Birth_Asphyxia"].mode()[0], inplace=True)
SAMPLE_THREE["Autopsy_Birth_Defect"].fillna(SAMPLE_THREE["Autopsy_Birth_Defect"].mode()[0], inplace=True)
SAMPLE_THREE["Place_Birth"].fillna(SAMPLE_THREE["Place_Birth"].mode()[0], inplace=True)
SAMPLE_THREE["Folic_Acid"].fillna(SAMPLE_THREE["Folic_Acid"].mode()[0], inplace=True)
SAMPLE_THREE["Maternal_Illness"].fillna(SAMPLE_THREE["Maternal_Illness"].mode()[0], inplace=True)
SAMPLE_THREE["Radiation_Exposure"].fillna(SAMPLE_THREE["Radiation_Exposure"].mode()[0], inplace=True)
SAMPLE_THREE["Substance_Abuse"].fillna(SAMPLE_THREE["Substance_Abuse"].mode()[0], inplace=True)
SAMPLE_THREE["Assisted_Conception"].fillna(SAMPLE_THREE["Assisted_Conception"].mode()[0], inplace=True)
SAMPLE_THREE["History_Previous_Pregnancies"].fillna(SAMPLE_THREE["History_Previous_Pregnancies"].mode()[0], inplace=True)
SAMPLE_THREE["Birth_Defects"].fillna(SAMPLE_THREE["Birth_Defects"].mode()[0], inplace=True)
SAMPLE_THREE["Blood_Test_Result"].fillna(SAMPLE_THREE["Blood_Test_Result"].mode()[0], inplace=True)


SAMPLE_THREE["Mother_Age"].fillna(SAMPLE_THREE.groupby(["Disorder_Subclass"])["Mother_Age"].transform("mean"),inplace=True)
SAMPLE_THREE["Father_Age"].fillna(SAMPLE_THREE.groupby(["Disorder_Subclass"])["Father_Age"].transform("mean"),inplace=True)
SAMPLE_THREE["Previous_Abortion"].fillna(SAMPLE_THREE.groupby(["Disorder_Subclass"])["Previous_Abortion"].transform("mean"),inplace=True)
SAMPLE_THREE["White_Blood_Cell"].fillna(SAMPLE_THREE.groupby(["Disorder_Subclass"])["White_Blood_Cell"].transform("mean"),inplace=True)

SAMPLE_THREE.dropna(inplace=True,axis=0)

SAMPLE_FOUR["Inherited_Father"].fillna(SAMPLE_FOUR["Inherited_Father"].mode()[0], inplace=True)
SAMPLE_FOUR["Maternal_Gene"].fillna(SAMPLE_FOUR["Maternal_Gene"].mode()[0], inplace=True)
SAMPLE_FOUR["Respiratory_Rate_Breaths_Min"].fillna(SAMPLE_FOUR["Respiratory_Rate_Breaths_Min"].mode()[0], inplace=True)
SAMPLE_FOUR["Heart_Rates_Min"].fillna(SAMPLE_FOUR["Heart_Rates_Min"].mode()[0], inplace=True)
SAMPLE_FOUR["Follow_Up"].fillna(SAMPLE_FOUR["Follow_Up"].mode()[0], inplace=True)
SAMPLE_FOUR["Gender"].fillna(SAMPLE_FOUR["Gender"].mode()[0], inplace=True)
SAMPLE_FOUR["Birth_Asphyxia"].fillna(SAMPLE_FOUR["Birth_Asphyxia"].mode()[0], inplace=True)
SAMPLE_FOUR["Autopsy_Birth_Defect"].fillna(SAMPLE_FOUR["Autopsy_Birth_Defect"].mode()[0], inplace=True)
SAMPLE_FOUR["Place_Birth"].fillna(SAMPLE_FOUR["Place_Birth"].mode()[0], inplace=True)
SAMPLE_FOUR["Folic_Acid"].fillna(SAMPLE_FOUR["Folic_Acid"].mode()[0], inplace=True)
SAMPLE_FOUR["Maternal_Illness"].fillna(SAMPLE_FOUR["Maternal_Illness"].mode()[0], inplace=True)
SAMPLE_FOUR["Radiation_Exposure"].fillna(SAMPLE_FOUR["Radiation_Exposure"].mode()[0], inplace=True)
SAMPLE_FOUR["Substance_Abuse"].fillna(SAMPLE_FOUR["Substance_Abuse"].mode()[0], inplace=True)
SAMPLE_FOUR["Assisted_Conception"].fillna(SAMPLE_FOUR["Assisted_Conception"].mode()[0], inplace=True)
SAMPLE_FOUR["History_Previous_Pregnancies"].fillna(SAMPLE_FOUR["History_Previous_Pregnancies"].mode()[0], inplace=True)
SAMPLE_FOUR["Birth_Defects"].fillna(SAMPLE_FOUR["Birth_Defects"].mode()[0], inplace=True)
SAMPLE_FOUR["Blood_Test_Result"].fillna(SAMPLE_FOUR["Blood_Test_Result"].mode()[0], inplace=True)


SAMPLE_FOUR["Mother_Age"].fillna(SAMPLE_FOUR.groupby(["Disorder_Subclass"])["Mother_Age"].transform("mean"),inplace=True)
SAMPLE_FOUR["Father_Age"].fillna(SAMPLE_FOUR.groupby(["Disorder_Subclass"])["Father_Age"].transform("mean"),inplace=True)
SAMPLE_FOUR["Previous_Abortion"].fillna(SAMPLE_FOUR.groupby(["Disorder_Subclass"])["Previous_Abortion"].transform("mean"),inplace=True)
SAMPLE_FOUR["White_Blood_Cell"].fillna(SAMPLE_FOUR.groupby(["Disorder_Subclass"])["White_Blood_Cell"].transform("mean"),inplace=True)

SAMPLE_FOUR.dropna(inplace=True,axis=0)

print("NAN VALUES:\n")
print(SAMPLE_ONE.isna().sum())
print("\n")
print("NAN VALUES:\n")
print(SAMPLE_TWO.isna().sum())
print("\n")
print("NAN VALUES:\n")
print(SAMPLE_THREE.isna().sum())
print("\n")
print("NAN VALUES:\n")
print(SAMPLE_FOUR.isna().sum())
print("\n")

Pre_Two_Data = Data.copy()

Data.dropna(inplace=True,axis=0)

Data = Data.reset_index(drop=True)

print("NAN VALUES:\n")
print(Data.isna().sum())

Encode_Data = Data.copy()

Encode_Func = LabelEncoder()

Encode_Data["Genes_Mother_Side"] = Encode_Func.fit_transform(Encode_Data["Genes_Mother_Side"])
Encode_Data["Inherited_Father"] = Encode_Func.fit_transform(Encode_Data["Inherited_Father"])
Encode_Data["Maternal_Gene"] = Encode_Func.fit_transform(Encode_Data["Maternal_Gene"])
Encode_Data["Paternal_Gene"] = Encode_Func.fit_transform(Encode_Data["Paternal_Gene"])
Encode_Data["Status"] = Encode_Func.fit_transform(Encode_Data["Status"])
Encode_Data["Respiratory_Rate_Breaths_Min"] = Encode_Func.fit_transform(Encode_Data["Respiratory_Rate_Breaths_Min"])
Encode_Data["Heart_Rates_Min"] = Encode_Func.fit_transform(Encode_Data["Heart_Rates_Min"])
Encode_Data["Follow_Up"] = Encode_Func.fit_transform(Encode_Data["Follow_Up"])
Encode_Data["Gender"] = Encode_Func.fit_transform(Encode_Data["Gender"])
Encode_Data["Birth_Asphyxia"] = Encode_Func.fit_transform(Encode_Data["Birth_Asphyxia"])
Encode_Data["Autopsy_Birth_Defect"] = Encode_Func.fit_transform(Encode_Data["Autopsy_Birth_Defect"])
Encode_Data["Place_Birth"] = Encode_Func.fit_transform(Encode_Data["Place_Birth"])
Encode_Data["Folic_Acid"] = Encode_Func.fit_transform(Encode_Data["Folic_Acid"])
Encode_Data["Maternal_Illness"] = Encode_Func.fit_transform(Encode_Data["Maternal_Illness"])
Encode_Data["Radiation_Exposure"] = Encode_Func.fit_transform(Encode_Data["Radiation_Exposure"])
Encode_Data["Substance_Abuse"] = Encode_Func.fit_transform(Encode_Data["Substance_Abuse"])
Encode_Data["Assisted_Conception"] = Encode_Func.fit_transform(Encode_Data["Assisted_Conception"])
Encode_Data["History_Previous_Pregnancies"] = Encode_Func.fit_transform(Encode_Data["History_Previous_Pregnancies"])
Encode_Data["Birth_Defects"] = Encode_Func.fit_transform(Encode_Data["Birth_Defects"])
Encode_Data["Blood_Test_Result"] = Encode_Func.fit_transform(Encode_Data["Blood_Test_Result"])
Encode_Data["Genetic_Disorder"] = Encode_Func.fit_transform(Encode_Data["Genetic_Disorder"])
Encode_Data["Disorder_Subclass"] = Encode_Func.fit_transform(Encode_Data["Disorder_Subclass"])
Encode_Data["Parental_Consent"] = Encode_Func.fit_transform(Encode_Data["Parental_Consent"])
Encode_Data = Encode_Data.astype("float32")

Encode_Data.info()

Encode_Data

"""PREDICTION PROCESS I"""

Genetic_Disorder_Data = Encode_Data.drop("Disorder_Subclass",axis=1)
Disorder_Subclass_Data = Encode_Data.drop("Genetic_Disorder",axis=1)
Genetic_Disorder_Data

Disorder_Subclass_Data

"""SPLITTING
-WE WILL ONLY ACCOMPLISH GENETIC FORECAST AS AN EXAMPLE. IF YOU WANT, YOU CAN DO THE SAME METHOD FOR DISORDER SUBCLASS
"""

GENETIC_X =  Genetic_Disorder_Data.drop("Genetic_Disorder",axis=1)
GENETIC_Y = Genetic_Disorder_Data["Genetic_Disorder"]

GX_Train,GX_Test,GY_Train,GY_Test = train_test_split(GENETIC_X,GENETIC_Y,test_size=0.2,random_state=42,shuffle=True)

print("X TRAIN SHAPE: ",GX_Train.shape)
print("X TEST SHAPE: ",GX_Test.shape)
print("Y TRAIN SHAPE: ",GY_Train.shape)
print("Y TEST SHAPE: ",GY_Test.shape)

"""NORMALIZING"""

Scaler_Function = StandardScaler()

GX_Train = Scaler_Function.fit_transform(GX_Train)
GX_Test = Scaler_Function.fit_transform(GX_Test)

"""MODEL"""

lj = LogisticRegression(solver="liblinear").fit(GX_Train,GY_Train)
gnb = GaussianNB().fit(GX_Train,GY_Train)
knnc = KNeighborsClassifier().fit(GX_Train,GY_Train)
cartc = DecisionTreeClassifier(random_state=42).fit(GX_Train,GY_Train)
rfc = RandomForestClassifier(random_state=42,verbose=False).fit(GX_Train,GY_Train)
gbmc = GradientBoostingClassifier(verbose=False).fit(GX_Train,GY_Train)
xgbc = XGBClassifier().fit(GX_Train,GY_Train)
lgbmc = LGBMClassifier().fit(GX_Train,GY_Train)
catbc = CatBoostClassifier(verbose=False).fit(GX_Train,GY_Train)

"""RESULTS"""

modelsc = [lj,gnb,knnc,cartc,rfc,gbmc,xgbc,lgbmc,catbc]

for model in modelsc:
    name = model.__class__.__name__
    predict = model.predict(GX_Test)
    R2CV = cross_val_score(model,GX_Test,GY_Test,cv=10,verbose=False).mean()
    error = -cross_val_score(model,GX_Test,GY_Test,cv=10,scoring="neg_mean_squared_error",verbose=False).mean()
    print(name + ": ")
    print("-" * 10)
    print("ACC-->",accuracy_score(GY_Test,predict))
    print("R2CV-->",R2CV)
    print("MEAN SQUARED ERROR-->",np.sqrt(error))
    print("-" * 30)

"""ACCURACY VISUALIZATION"""

r = pd.DataFrame(columns=["MODELS","R2CV"])
for model in modelsc:
    name = model.__class__.__name__
    R2CV = cross_val_score(model,GX_Test,GY_Test,cv=10,verbose=False).mean()
    result = pd.DataFrame([[name,R2CV*100]],columns=["MODELS","R2CV"])
    r = r.append(result)

figure = plt.figure(figsize=(20,8))
sns.barplot(x="R2CV",y="MODELS",data=r,color="k")
plt.xlabel("R2CV")
plt.ylabel("MODELS")
plt.xlim(0,100)
plt.title("MODEL ACCURACY COMPARISON")
plt.show()

"""PREDICTION PROCESS 2
#NOW LET'S CREATE A PREDICTION PROCESS WITHOUT DELETING ANY ROW BY FILLING THE NAN VALUES IN THE MAIN DATA
"""

Pre_Two_Data = Pre_Two_Data.reset_index(drop=True)

Pre_Two_Data

Pre_Two_Data["Inherited_Father"].fillna(Pre_Two_Data["Inherited_Father"].mode()[0], inplace=True)
Pre_Two_Data["Maternal_Gene"].fillna(Pre_Two_Data["Maternal_Gene"].mode()[0], inplace=True)
Pre_Two_Data["Respiratory_Rate_Breaths_Min"].fillna(Pre_Two_Data["Respiratory_Rate_Breaths_Min"].mode()[0], inplace=True)
Pre_Two_Data["Heart_Rates_Min"].fillna(Pre_Two_Data["Heart_Rates_Min"].mode()[0], inplace=True)
Pre_Two_Data["Follow_Up"].fillna(Pre_Two_Data["Follow_Up"].mode()[0], inplace=True)
Pre_Two_Data["Gender"].fillna(Pre_Two_Data["Gender"].mode()[0], inplace=True)
Pre_Two_Data["Birth_Asphyxia"].fillna(Pre_Two_Data["Birth_Asphyxia"].mode()[0], inplace=True)
Pre_Two_Data["Autopsy_Birth_Defect"].fillna(Pre_Two_Data["Autopsy_Birth_Defect"].mode()[0], inplace=True)
Pre_Two_Data["Place_Birth"].fillna(Pre_Two_Data["Place_Birth"].mode()[0], inplace=True)
Pre_Two_Data["Folic_Acid"].fillna(Pre_Two_Data["Folic_Acid"].mode()[0], inplace=True)
Pre_Two_Data["Maternal_Illness"].fillna(Pre_Two_Data["Maternal_Illness"].mode()[0], inplace=True)
Pre_Two_Data["Radiation_Exposure"].fillna(Pre_Two_Data["Radiation_Exposure"].mode()[0], inplace=True)
Pre_Two_Data["Substance_Abuse"].fillna(Pre_Two_Data["Substance_Abuse"].mode()[0], inplace=True)
Pre_Two_Data["Assisted_Conception"].fillna(Pre_Two_Data["Assisted_Conception"].mode()[0], inplace=True)
Pre_Two_Data["History_Previous_Pregnancies"].fillna(Pre_Two_Data["History_Previous_Pregnancies"].mode()[0], inplace=True)
Pre_Two_Data["Birth_Defects"].fillna(Pre_Two_Data["Birth_Defects"].mode()[0], inplace=True)
Pre_Two_Data["Blood_Test_Result"].fillna(Pre_Two_Data["Blood_Test_Result"].mode()[0], inplace=True)
Pre_Two_Data["Mother_Age"].fillna(Pre_Two_Data.groupby(["Disorder_Subclass"])["Mother_Age"].transform("mean"),inplace=True)
Pre_Two_Data["Father_Age"].fillna(Pre_Two_Data.groupby(["Disorder_Subclass"])["Father_Age"].transform("mean"),inplace=True)
Pre_Two_Data["Previous_Abortion"].fillna(Pre_Two_Data.groupby(["Disorder_Subclass"])["Previous_Abortion"].transform("mean"),inplace=True)
Pre_Two_Data["White_Blood_Cell"].fillna(Pre_Two_Data.groupby(["Disorder_Subclass"])["White_Blood_Cell"].transform("mean"),inplace=True)
Pre_Two_Data.dropna(inplace=True,axis=0)

print("NAN VALUES:\n")
print(Pre_Two_Data.isna().sum())

print("INFO:\n")
print(Pre_Two_Data.info())

Main_Encoded_Data = Pre_Two_Data.copy()

Main_Encoded_Data["Genes_Mother_Side"] = Encode_Func.fit_transform(Main_Encoded_Data["Genes_Mother_Side"])
Main_Encoded_Data["Inherited_Father"] = Encode_Func.fit_transform(Main_Encoded_Data["Inherited_Father"])
Main_Encoded_Data["Maternal_Gene"] = Encode_Func.fit_transform(Main_Encoded_Data["Maternal_Gene"])
Main_Encoded_Data["Paternal_Gene"] = Encode_Func.fit_transform(Main_Encoded_Data["Paternal_Gene"])
Main_Encoded_Data["Status"] = Encode_Func.fit_transform(Main_Encoded_Data["Status"])
Main_Encoded_Data["Respiratory_Rate_Breaths_Min"] = Encode_Func.fit_transform(Main_Encoded_Data["Respiratory_Rate_Breaths_Min"])
Main_Encoded_Data["Heart_Rates_Min"] = Encode_Func.fit_transform(Main_Encoded_Data["Heart_Rates_Min"])
Main_Encoded_Data["Follow_Up"] = Encode_Func.fit_transform(Main_Encoded_Data["Follow_Up"])
Main_Encoded_Data["Gender"] = Encode_Func.fit_transform(Main_Encoded_Data["Gender"])
Main_Encoded_Data["Birth_Asphyxia"] = Encode_Func.fit_transform(Main_Encoded_Data["Birth_Asphyxia"])
Main_Encoded_Data["Autopsy_Birth_Defect"] = Encode_Func.fit_transform(Main_Encoded_Data["Autopsy_Birth_Defect"])
Main_Encoded_Data["Place_Birth"] = Encode_Func.fit_transform(Main_Encoded_Data["Place_Birth"])
Main_Encoded_Data["Folic_Acid"] = Encode_Func.fit_transform(Main_Encoded_Data["Folic_Acid"])
Main_Encoded_Data["Maternal_Illness"] = Encode_Func.fit_transform(Main_Encoded_Data["Maternal_Illness"])
Main_Encoded_Data["Radiation_Exposure"] = Encode_Func.fit_transform(Main_Encoded_Data["Radiation_Exposure"])
Main_Encoded_Data["Substance_Abuse"] = Encode_Func.fit_transform(Main_Encoded_Data["Substance_Abuse"])
Main_Encoded_Data["Assisted_Conception"] = Encode_Func.fit_transform(Main_Encoded_Data["Assisted_Conception"])
Main_Encoded_Data["History_Previous_Pregnancies"] = Encode_Func.fit_transform(Main_Encoded_Data["History_Previous_Pregnancies"])
Main_Encoded_Data["Birth_Defects"] = Encode_Func.fit_transform(Main_Encoded_Data["Birth_Defects"])
Main_Encoded_Data["Blood_Test_Result"] = Encode_Func.fit_transform(Main_Encoded_Data["Blood_Test_Result"])
Main_Encoded_Data["Genetic_Disorder"] = Encode_Func.fit_transform(Main_Encoded_Data["Genetic_Disorder"])
Main_Encoded_Data["Disorder_Subclass"] = Encode_Func.fit_transform(Main_Encoded_Data["Disorder_Subclass"])
Main_Encoded_Data["Parental_Consent"] = Encode_Func.fit_transform(Main_Encoded_Data["Parental_Consent"])

Main_Encoded_Data = Main_Encoded_Data.astype("float32")

print("INFO:\n")
print(Main_Encoded_Data.info())

Genetic_Main_Data = Main_Encoded_Data.drop("Disorder_Subclass",axis=1)
Disorder_Main_Data = Main_Encoded_Data.drop("Genetic_Disorder",axis=1)

"""WE WILL ONLY ACCOMPLISH GENETIC FORECAST AS AN EXAMPLE. IF YOU WANT, YOU CAN DO THE SAME METHOD FOR DISORDER SUBCLASS"""

X =  Genetic_Main_Data.drop("Genetic_Disorder",axis=1)
Y = Genetic_Main_Data["Genetic_Disorder"]

X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y,test_size=0.2,random_state=42,shuffle=True)

print("X TRAIN SHAPE: ",X_Train.shape)
print("X TEST SHAPE: ",X_Test.shape)
print("Y TRAIN SHAPE: ",Y_Train.shape)
print("Y TEST SHAPE: ",Y_Test.shape)

Scaler_Function = StandardScaler()

X_Train = Scaler_Function.fit_transform(X_Train)
X_Test = Scaler_Function.fit_transform(X_Test)

#lj_m = LogisticRegression(solver="liblinear").fit(X_Train,Y_Train)
gnb_m = GaussianNB().fit(X_Train,Y_Train)
knnc_m = KNeighborsClassifier().fit(X_Train,Y_Train)
cartc_m = DecisionTreeClassifier(random_state=42).fit(X_Train,Y_Train)
rfc_m = RandomForestClassifier(random_state=42,verbose=False).fit(X_Train,Y_Train)
gbmc_m = GradientBoostingClassifier(verbose=False).fit(X_Train,Y_Train)
xgbc_m = XGBClassifier().fit(X_Train,Y_Train)
lgbmc_m = LGBMClassifier().fit(X_Train,Y_Train)
catbc_m = CatBoostClassifier(verbose=False).fit(X_Train,Y_Train)

model_m = [gnb_m,knnc_m,cartc_m,rfc_m,gbmc_m,xgbc_m,lgbmc_m,catbc_m]

for model in model_m:
    name = model.__class__.__name__
    predict = model.predict(X_Test)
    R2CV = cross_val_score(model,X_Test,Y_Test,cv=10,verbose=False).mean()
    error = -cross_val_score(model,X_Test,Y_Test,cv=10,scoring="neg_mean_squared_error",verbose=False).mean()
    print(name + ": ")
    print("-" * 10)
    print("ACC-->",accuracy_score(Y_Test,predict))
    print("R2CV-->",R2CV)
    print("MEAN SQUARED ERROR-->",np.sqrt(error))
    print("-" * 30)

r = pd.DataFrame(columns=["MODELS","R2CV"])
for model in model_m:
    name = model.__class__.__name__
    R2CV = cross_val_score(model,X_Test,Y_Test,cv=10,verbose=False).mean()
    result = pd.DataFrame([[name,R2CV*100]],columns=["MODELS","R2CV"])
    r = r.append(result)

figure = plt.figure(figsize=(20,8))
sns.barplot(x="R2CV",y="MODELS",data=r,color="k")
plt.xlabel("R2CV")
plt.ylabel("MODELS")
plt.xlim(0,100)
plt.title("MODEL ACCURACY COMPARISON")
plt.show()

r = pd.DataFrame(columns=["MODELS","error"])
for model in model_m:
    name = model.__class__.__name__
    error = -cross_val_score(model,X_Test,Y_Test,cv=10,scoring="neg_mean_squared_error").mean()
    result = pd.DataFrame([[name,np.sqrt(error)]],columns=["MODELS","error"])
    r = r.append(result)

figure = plt.figure(figsize=(20,8))
sns.barplot(x="error",y="MODELS",data=r,color="r")
plt.xlabel("ERROR")
plt.ylabel("MODELS")
plt.xlim(0,2)
plt.title("MODEL ERROR COMPARISON")
plt.show()

"""PREDICTION PROCESS II
NOW LET'S CREATE A PREDICTION PROCESS WITHOUT DELETING ANY ROW BY FILLING THE NAN VALUES IN THE MAIN DATA
"""

Pre_Two_Data = Pre_Two_Data.reset_index(drop=True)

Pre_Two_Data

Pre_Two_Data["Inherited_Father"].fillna(Pre_Two_Data["Inherited_Father"].mode()[0], inplace=True)
Pre_Two_Data["Maternal_Gene"].fillna(Pre_Two_Data["Maternal_Gene"].mode()[0], inplace=True)
Pre_Two_Data["Respiratory_Rate_Breaths_Min"].fillna(Pre_Two_Data["Respiratory_Rate_Breaths_Min"].mode()[0], inplace=True)
Pre_Two_Data["Heart_Rates_Min"].fillna(Pre_Two_Data["Heart_Rates_Min"].mode()[0], inplace=True)
Pre_Two_Data["Follow_Up"].fillna(Pre_Two_Data["Follow_Up"].mode()[0], inplace=True)
Pre_Two_Data["Gender"].fillna(Pre_Two_Data["Gender"].mode()[0], inplace=True)
Pre_Two_Data["Birth_Asphyxia"].fillna(Pre_Two_Data["Birth_Asphyxia"].mode()[0], inplace=True)
Pre_Two_Data["Autopsy_Birth_Defect"].fillna(Pre_Two_Data["Autopsy_Birth_Defect"].mode()[0], inplace=True)
Pre_Two_Data["Place_Birth"].fillna(Pre_Two_Data["Place_Birth"].mode()[0], inplace=True)
Pre_Two_Data["Folic_Acid"].fillna(Pre_Two_Data["Folic_Acid"].mode()[0], inplace=True)
Pre_Two_Data["Maternal_Illness"].fillna(Pre_Two_Data["Maternal_Illness"].mode()[0], inplace=True)
Pre_Two_Data["Radiation_Exposure"].fillna(Pre_Two_Data["Radiation_Exposure"].mode()[0], inplace=True)
Pre_Two_Data["Substance_Abuse"].fillna(Pre_Two_Data["Substance_Abuse"].mode()[0], inplace=True)
Pre_Two_Data["Assisted_Conception"].fillna(Pre_Two_Data["Assisted_Conception"].mode()[0], inplace=True)
Pre_Two_Data["History_Previous_Pregnancies"].fillna(Pre_Two_Data["History_Previous_Pregnancies"].mode()[0], inplace=True)
Pre_Two_Data["Birth_Defects"].fillna(Pre_Two_Data["Birth_Defects"].mode()[0], inplace=True)
Pre_Two_Data["Blood_Test_Result"].fillna(Pre_Two_Data["Blood_Test_Result"].mode()[0], inplace=True)
Pre_Two_Data["Mother_Age"].fillna(Pre_Two_Data.groupby(["Disorder_Subclass"])["Mother_Age"].transform("mean"),inplace=True)
Pre_Two_Data["Father_Age"].fillna(Pre_Two_Data.groupby(["Disorder_Subclass"])["Father_Age"].transform("mean"),inplace=True)
Pre_Two_Data["Previous_Abortion"].fillna(Pre_Two_Data.groupby(["Disorder_Subclass"])["Previous_Abortion"].transform("mean"),inplace=True)
Pre_Two_Data["White_Blood_Cell"].fillna(Pre_Two_Data.groupby(["Disorder_Subclass"])["White_Blood_Cell"].transform("mean"),inplace=True)
Pre_Two_Data.dropna(inplace=True,axis=0)


print("NAN VALUES:\n")
print(Pre_Two_Data.isna().sum())

print("INFO:\n")
print(Pre_Two_Data.info())

Main_Encoded_Data = Pre_Two_Data.copy()

Main_Encoded_Data["Genes_Mother_Side"] = Encode_Func.fit_transform(Main_Encoded_Data["Genes_Mother_Side"])
Main_Encoded_Data["Inherited_Father"] = Encode_Func.fit_transform(Main_Encoded_Data["Inherited_Father"])
Main_Encoded_Data["Maternal_Gene"] = Encode_Func.fit_transform(Main_Encoded_Data["Maternal_Gene"])
Main_Encoded_Data["Paternal_Gene"] = Encode_Func.fit_transform(Main_Encoded_Data["Paternal_Gene"])
Main_Encoded_Data["Status"] = Encode_Func.fit_transform(Main_Encoded_Data["Status"])
Main_Encoded_Data["Respiratory_Rate_Breaths_Min"] = Encode_Func.fit_transform(Main_Encoded_Data["Respiratory_Rate_Breaths_Min"])
Main_Encoded_Data["Heart_Rates_Min"] = Encode_Func.fit_transform(Main_Encoded_Data["Heart_Rates_Min"])
Main_Encoded_Data["Follow_Up"] = Encode_Func.fit_transform(Main_Encoded_Data["Follow_Up"])
Main_Encoded_Data["Gender"] = Encode_Func.fit_transform(Main_Encoded_Data["Gender"])
Main_Encoded_Data["Birth_Asphyxia"] = Encode_Func.fit_transform(Main_Encoded_Data["Birth_Asphyxia"])
Main_Encoded_Data["Autopsy_Birth_Defect"] = Encode_Func.fit_transform(Main_Encoded_Data["Autopsy_Birth_Defect"])
Main_Encoded_Data["Place_Birth"] = Encode_Func.fit_transform(Main_Encoded_Data["Place_Birth"])
Main_Encoded_Data["Folic_Acid"] = Encode_Func.fit_transform(Main_Encoded_Data["Folic_Acid"])
Main_Encoded_Data["Maternal_Illness"] = Encode_Func.fit_transform(Main_Encoded_Data["Maternal_Illness"])
Main_Encoded_Data["Radiation_Exposure"] = Encode_Func.fit_transform(Main_Encoded_Data["Radiation_Exposure"])
Main_Encoded_Data["Substance_Abuse"] = Encode_Func.fit_transform(Main_Encoded_Data["Substance_Abuse"])
Main_Encoded_Data["Assisted_Conception"] = Encode_Func.fit_transform(Main_Encoded_Data["Assisted_Conception"])
Main_Encoded_Data["History_Previous_Pregnancies"] = Encode_Func.fit_transform(Main_Encoded_Data["History_Previous_Pregnancies"])
Main_Encoded_Data["Birth_Defects"] = Encode_Func.fit_transform(Main_Encoded_Data["Birth_Defects"])
Main_Encoded_Data["Blood_Test_Result"] = Encode_Func.fit_transform(Main_Encoded_Data["Blood_Test_Result"])
Main_Encoded_Data["Genetic_Disorder"] = Encode_Func.fit_transform(Main_Encoded_Data["Genetic_Disorder"])
Main_Encoded_Data["Disorder_Subclass"] = Encode_Func.fit_transform(Main_Encoded_Data["Disorder_Subclass"])
Main_Encoded_Data["Parental_Consent"] = Encode_Func.fit_transform(Main_Encoded_Data["Parental_Consent"])

Main_Encoded_Data = Main_Encoded_Data.astype("float32")

print("INFO:\n")
print(Main_Encoded_Data.info())

Genetic_Main_Data = Main_Encoded_Data.drop("Disorder_Subclass",axis=1)
Disorder_Main_Data = Main_Encoded_Data.drop("Genetic_Disorder",axis=1)

"""WE WILL ONLY ACCOMPLISH GENETIC FORECAST AS AN EXAMPLE. IF YOU WANT, YOU CAN DO THE SAME METHOD FOR DISORDER SUBCLASS"""

X =  Genetic_Main_Data.drop("Genetic_Disorder",axis=1)
Y = Genetic_Main_Data["Genetic_Disorder"]

X.head(2)

X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y,test_size=0.2,random_state=42,shuffle=True)

print("X TRAIN SHAPE: ",X_Train.shape)
print("X TEST SHAPE: ",X_Test.shape)
print("Y TRAIN SHAPE: ",Y_Train.shape)
print("Y TEST SHAPE: ",Y_Test.shape)

Scaler_Function = StandardScaler()

X_Train = Scaler_Function.fit_transform(X_Train)
X_Test = Scaler_Function.fit_transform(X_Test)

gnb_m = GaussianNB().fit(X_Train,Y_Train)
knnc_m = KNeighborsClassifier().fit(X_Train,Y_Train)
cartc_m = DecisionTreeClassifier(random_state=42).fit(X_Train,Y_Train)
rfc_m = RandomForestClassifier(random_state=42,verbose=False).fit(X_Train,Y_Train)
gbmc_m = GradientBoostingClassifier(verbose=False).fit(X_Train,Y_Train)
xgbc_m = XGBClassifier().fit(X_Train,Y_Train)
lgbmc_m = LGBMClassifier().fit(X_Train,Y_Train)
catbc_m = CatBoostClassifier(verbose=False).fit(X_Train,Y_Train)

model_m = [gnb_m,knnc_m,cartc_m,rfc_m,gbmc_m,xgbc_m,lgbmc_m,catbc_m]

for model in model_m:
    name = model.__class__.__name__
    predict = model.predict(X_Test)
    R2CV = cross_val_score(model,X_Test,Y_Test,cv=10,verbose=False).mean()
    error = -cross_val_score(model,X_Test,Y_Test,cv=10,scoring="neg_mean_squared_error",verbose=False).mean()
    print(name + ": ")
    print("-" * 10)
    print("ACC-->",accuracy_score(Y_Test,predict))
    print("R2CV-->",R2CV)
    print("MEAN SQUARED ERROR-->",np.sqrt(error))
    print("-" * 30)

r = pd.DataFrame(columns=["MODELS","R2CV"])
for model in model_m:
    name = model.__class__.__name__
    R2CV = cross_val_score(model,X_Test,Y_Test,cv=10,verbose=False).mean()
    result = pd.DataFrame([[name,R2CV*100]],columns=["MODELS","R2CV"])
    r = r.append(result)

figure = plt.figure(figsize=(20,8))
sns.barplot(x="R2CV",y="MODELS",data=r,color="k")
plt.xlabel("R2CV")
plt.ylabel("MODELS")
plt.xlim(0,100)
plt.title("MODEL ACCURACY COMPARISON")
plt.show()

r = pd.DataFrame(columns=["MODELS","error"])
for model in model_m:
    name = model.__class__.__name__
    error = -cross_val_score(model,X_Test,Y_Test,cv=10,scoring="neg_mean_squared_error").mean()
    result = pd.DataFrame([[name,np.sqrt(error)]],columns=["MODELS","error"])
    r = r.append(result)

figure = plt.figure(figsize=(20,8))
sns.barplot(x="error",y="MODELS",data=r,color="r")
plt.xlabel("ERROR")
plt.ylabel("MODELS")
plt.xlim(0,2)
plt.title("MODEL ERROR COMPARISON")
plt.show()

"""ADDITIONAL METHOD
-WE WOULD LIKE TO SHOW YOU NORMALITY, HOMOGENEITY, OUTLIER METHODS ADDITIONALLY.
-These approaches are not normally used for this data. Because the data is insufficient and unstable for these methods. There is a lot of missing data.
"""

for col in Main_Encoded_Data.columns:
    print(col)
    print("---"*5)
    print("%.4f - %.4f" % shapiro(Main_Encoded_Data[col]))
    print("---"*15)

print("%.4f - %.4f" % levene(Main_Encoded_Data["Maternal_Illness"],Main_Encoded_Data["White_Blood_Cell"],
                            Main_Encoded_Data["Status"],Main_Encoded_Data["Genetic_Disorder"]))
# check for others

"""OUTLIER"""

Data_For_Outlier = Main_Encoded_Data.copy()

CLF_Function = LocalOutlierFactor()

CLF_Function.fit_predict(Data_For_Outlier)

#checking outlier, look where the biggest jump took place
General_Score = CLF_Function.negative_outlier_factor_

Sorted_Score = np.sort(General_Score)

print(Sorted_Score[:150])

"""AS WE SAID, THIS IS NOT SUITABLE FOR DATA. WE CAN'T SEE A BIG JUMP.
but when it does not, you can follow the process below
"""

point = Sorted_Score[6] # it just for example
print(point)
print("---"*10)
print(Data_For_Outlier[General_Score == point])

outliers = General_Score < point
print(Main_Encoded_Data[outliers])
print("---"*20)
print(Main_Encoded_Data[outliers].index)

Outliers_Index_List = [Main_Encoded_Data[outliers].index]

for d_i in Outliers_Index_List:
    Main_Encoded_Data.drop(index=d_i,inplace=True)

"""BIOLOGICAL INFORMATION AND PROCESS
------------------------------------------
1) In general, the reference ranges are: White blood cells: 4,500 to 11,000 cells per microliter (cells/mcL) Red blood cells: 4.5 million to 5.9 million cells/mcL for men; 4.1 million to 5.1 million cells/mcL for women.
Risk factors for pre-term birth were found to be: being indigenous, single, a smoker [adjusted odds ratio (AOR) 1.28, 95% confidence interval 1.17-1.41], age 40 years or older, reproductive technology assistance, threatened miscarriage, antepartum haemorrhage, urinary tract infection, pregnancy hypertension and suspected intra-uterine growth restriction. A previous spontaneous abortion was of borderline statistical significance, whereas a previous induced abortion (AOR 1.25, 1.13-1.40) was an independent risk factor.


--> WE WILL MAKE ARRANGEMENTS IN THE LIGHT OF SCIENTIFIC DATA ABOUT TWO NUMERICAL COLUMNS.
1) White_Blood_Cell
2)Previous_Abortion
"""

print("MODE: ", Main_Encoded_Data["White_Blood_Cell"].mode(),print("\n"))
print("MAX: ", Main_Encoded_Data["White_Blood_Cell"].max(),print("\n"))
print("MIN: ", Main_Encoded_Data["White_Blood_Cell"].min(),print("\n"))
print("MEAN: ", Main_Encoded_Data["White_Blood_Cell"].mean(),print("\n"))

print("MODE: ", Main_Encoded_Data["Previous_Abortion"].mode(),print("\n"))
print("MAX: ", Main_Encoded_Data["Previous_Abortion"].max(),print("\n"))
print("MIN: ", Main_Encoded_Data["Previous_Abortion"].min(),print("\n"))
print("MEAN: ", Main_Encoded_Data["Previous_Abortion"].mean(),print("\n"))

print("Confidence interval is based on normal distribution:\n",sms.DescrStatsW(Main_Encoded_Data["White_Blood_Cell"]).tconfint_mean())

print("Confidence interval is based on normal distribution:\n",sms.DescrStatsW(Main_Encoded_Data["Previous_Abortion"]).tconfint_mean())

Main_Encoded_Data["White_Blood_Cell"]

Main_Encoded_Data["Previous_Abortion"]

Abortion_Risk_Line = (1.25 + 1.40) / 2
print(Abortion_Risk_Line)

def std_values_abortion(x):
    if x >= 1.325:
        return "DANGEROUS" # based on Abortion Risk Factor
    else:
        return "ACCEPTABLE"

def std_values_blood(x):
    if x <= 4.5:
        return "DANGEROUS" # based on global WHITE BLOOD CELL Risk Factor
    elif x >= 11.:
        return "DANGEROUS" # based on global WHITE BLOOD CELL Risk Factor
    else:
        return "ACCEPTABLE"

Main_Encoded_Data["Previous_Abortion"] = Main_Encoded_Data["Previous_Abortion"].apply(lambda x: std_values_abortion(x))

Main_Encoded_Data["White_Blood_Cell"] = Main_Encoded_Data["White_Blood_Cell"].apply(lambda x: std_values_blood(x))

print("VALUE:\n")
print(Main_Encoded_Data["Previous_Abortion"].value_counts())

print("VALUE:\n")
print(Main_Encoded_Data["White_Blood_Cell"].value_counts())

Main_Encoded_Data["Previous_Abortion"] = Encode_Func.fit_transform(Main_Encoded_Data["Previous_Abortion"])
Main_Encoded_Data["White_Blood_Cell"] = Encode_Func.fit_transform(Main_Encoded_Data["White_Blood_Cell"])

Genetic_Main_Data = Main_Encoded_Data.drop("Disorder_Subclass",axis=1)
# save the data
Genetic_Main_Data.to_csv("data/biomarkers.csv",index=False)
X =  Genetic_Main_Data.drop("Genetic_Disorder",axis=1)
Y = Genetic_Main_Data["Genetic_Disorder"]
print(X.head())

X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y,test_size=0.2,random_state=42,shuffle=True)

X_Train = Scaler_Function.fit_transform(X_Train)
X_Test = Scaler_Function.fit_transform(X_Test)

gnb_m = GaussianNB().fit(X_Train,Y_Train)
knnc_m = KNeighborsClassifier().fit(X_Train,Y_Train)
cartc_m = DecisionTreeClassifier(random_state=42).fit(X_Train,Y_Train)
rfc_m = RandomForestClassifier(random_state=42,verbose=False).fit(X_Train,Y_Train)
gbmc_m = GradientBoostingClassifier(verbose=False).fit(X_Train,Y_Train)
xgbc_m = XGBClassifier().fit(X_Train,Y_Train)
lgbmc_m = LGBMClassifier().fit(X_Train,Y_Train)
catbc_m = CatBoostClassifier(verbose=False).fit(X_Train,Y_Train)


model_m = [gnb_m,knnc_m,cartc_m,rfc_m,gbmc_m,xgbc_m,lgbmc_m,catbc_m]
import joblib

for model in model_m:
    name = model.__class__.__name__
    predict = model.predict(X_Test)
    R2CV = cross_val_score(model,X_Test,Y_Test,cv=10,verbose=False).mean()
    error = -cross_val_score(model,X_Test,Y_Test,cv=10,scoring="neg_mean_squared_error",verbose=False).mean()
    print(name + ": ")
    print("-" * 10)
    print("ACC-->",accuracy_score(Y_Test,predict))
    print("R2CV-->",R2CV)
    print("MEAN SQUARED ERROR-->",np.sqrt(error))
    print("-" * 30)
    with open(f'{name}.jb', 'wb') as f:
        joblib.dump(model, f)

import pandas as pd
df = pd.read_csv('data/train.csv', nrows=1)
tuple(df.iloc[0].items())

