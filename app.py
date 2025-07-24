import os
import gradio as gr
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

SAVED_MODEL_DIR = r"D:\ML_Projects\cancer_prediction\models"

scaler = joblib.load(os.path.join(SAVED_MODEL_DIR,'scaler.pkl'))
scaler_clf  = joblib.load(os.path.join(SAVED_MODEL_DIR,'scaler_clf.pkl'))
linear_model = joblib.load(os.path.join(SAVED_MODEL_DIR,'linear_model.pkl'))
raf_regressor=joblib.load(os.path.join(SAVED_MODEL_DIR,'raf_regressor.pkl'))
logic_clf=joblib.load(os.path.join(SAVED_MODEL_DIR,'logic_clf.pkl'))

def preprocess(perimeter_mean,area_mean,concavity_mean,compactness_mean,concave_points_mean,radius_worst,area_worst,perimeter_worst,concavity_worst,concave_points_worst):
    input_data = pd.DataFrame([[perimeter_mean,area_mean,concavity_mean,compactness_mean,concave_points_mean,radius_worst,area_worst,perimeter_worst,concavity_worst,concave_points_worst]],
                              columns=['perimeter_mean', 'area_mean', 'concavity_mean', 'compactness_mean', 'concave points_mean', 'radius_worst', 'area_worst', 'perimeter_worst', 'concavity_worst', 'concave points_worst'])
    input_scaled = scaler.transform(input_data)


    pred_linear = linear_model.predict(input_scaled)[0]
    pred_rf = raf_regressor.predict(input_scaled)[0]
    pred_avg = (pred_linear + pred_rf) /2

    input_data['pred_radius']=pred_avg
    input_clf_scaled = scaler_clf.transform(input_data)

    logic_class = logic_clf.predict(input_clf_scaled)[0]

    models = ['Linear','RandomForest']
    prediction = [pred_linear,pred_rf]
    plt.figure(figsize=(8,5))
    sns.barplot(x=models,y=prediction)
    plt.title("Radius Prediction by Models")
    plt.ylabel("Predicted Radius")
    plt.savefig("Radius_plot.png")
    plt.close()

    output_text = f"""
<div style='font-size: 16px; line-height: 1.5'>
    <span style='color: #1f77b4;'><b>Linear Regression Radius:</b> {pred_linear:.2f}</span><br>
    <span style='color: #ff7f0e;'><b>RandomForest Regression Radius:</b> {pred_rf:.2f}</span><br>
    <span style='color: {"red" if logic_class == 1 else "green"};'><b>Logistic Classification:</b> {'Malignant Detected' if logic_class == 1 else 'Benign Detected'}</span>
</div>
"""

    return output_text,"Radius_plot.png"
if __name__ == '__main__':
    iface = gr.Interface(
    fn = preprocess,
    inputs = [gr.Slider(0,200,label="Perimeter Mean"),
    gr.Slider(1,2000,1,label="Area Mean"),
    gr.Slider(0,2,label="Concavity Mean"),
    gr.Slider(0,2,label="Compactness Mean"),
    gr.Slider(0,2,label="Concave Points Mean"),
    gr.Slider(0,100,label="Radius Worst"),
    gr.Slider(1,3000,label="Area Worst"),
    gr.Slider(1,250,label="Perimeter Worst"),
    gr.Slider(0,2,label="Concavity Worst"),
    gr.Slider(0,2,label="Concave Points Worst")],

    outputs=[gr.HTML('Predicted Result'),gr.Image("Regression Model Radius Comparison Graph")],
    title="🧪Breast Cancer Prediction System",
    description="Enter tumor characteristics to get an instant prediction of whether it's likely benign or malignant.This System uses Linear Reagression and Random Forest" \
    "regreesion to predict the radius of the tumor cell and uses Logistic regression to classify the tumor benign or malignant based on those prediction.")

    iface.launch()

