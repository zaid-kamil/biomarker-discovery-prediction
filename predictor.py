import joblib

def predict(dataframe):
    # load the best model GradientBoostingClassifier
    model = joblib.load("models/GradientBoostingClassifier.joblib")
    return model.predict(dataframe)

# single prediction
def predict_record(dataframe):
    predictions = predict(dataframe)
    print(f"Prediction: {predictions}")
    # decode prediction
    y_enc = joblib.load("models/y_encoder.joblib")
    predictions = y_enc.inverse_transform(predictions)
    return [
        {"idx": i, "prediction": pred} for i, pred in enumerate(predictions)
    ]