from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib
from sklearn.compose import ColumnTransformer


def train_evaluate_visualize_models(data, models):
    model_accuracies = []
    X = data.drop('Genetic_Disorder', axis=1)
    y = data['Genetic_Disorder']
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns
    # encode y
    y_enc = LabelEncoder()
    y = y_enc.fit_transform(y)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OrdinalEncoder(), cat_cols)
        ])
                                                                                        
    for model in models:
        pipeline_steps = [
            ('preprocessor', preprocessor),
            ('model', model)
        ]
        pipeline = Pipeline(steps=pipeline_steps)
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
        
            accuracy = accuracy_score(y_test, y_pred)
            cv_score = cross_val_score(pipeline, X_test, y_test, cv=10).mean()
            
            model_accuracies.append(accuracy)
            print(f"{model.__class__.__name__}:")
            print(f"\tAccuracy: {accuracy}")
            print(f"\tCross-Validation Score: {cv_score}")
            # save model
            joblib.dump(pipeline, f"models/{model.__class__.__name__}.joblib")
        except Exception as e:
            print(f"Error: {e}")
            accuracy, cv_score = 0, 0
            model_accuracies.append(accuracy)

    plt.figure(figsize=(10, 6))
    plt.barh([model.__class__.__name__ for model in models], model_accuracies, color='skyblue', edgecolor='black')
    # add text
    for i, model in enumerate(models):
        plt.text(model_accuracies[i], i, f'{model_accuracies[i]:.2f}', ha = 'left', va = 'center', fontsize=10)
    plt.xlabel('Accuracy')
    plt.ylabel('Models')
    plt.title('Model Accuracy Comparison')
    plt.xlim(0, 1)
    plt.show()
    # save encoded y
    joblib.dump(y_enc, "models/y_encoder.joblib")

# Load data
data = load_data("data/train_preprocessed.csv")
# Define models
models = [
    GaussianNB(),
    # RandomForestClassifier(random_state=42, verbose=False), # too slow
    GradientBoostingClassifier(verbose=False),
    XGBClassifier(),
    LGBMClassifier(verbose=-1,),
    CatBoostClassifier(verbose=False)
]

# Train, evaluate, and visualize models
train_evaluate_visualize_models(data, models)