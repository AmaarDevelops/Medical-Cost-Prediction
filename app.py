from flask import Flask,request,render_template,jsonify
import pandas as pd
import joblib


app = Flask(__name__)

try:
    model = joblib.load('best_model_rf.joblib')
    features = joblib.load('feature_columns.joblib')

except FileNotFoundError:
    print('Model or Feature files not found')
    model = None
    features = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if not model or not features:
        return jsonify({'error' : 'Model not Loaded'}) , 500
    try:
        data = request.get_json(force=True)
        input_data = pd.DataFrame(data,index=[0])

        def get_bmi_category(bmi_value):
            if bmi_value <= 18.5:
                return 'UnderWeight'
            elif 18.5 < bmi_value <= 24.9:
                return 'Normal Weight'
            elif 24.9 < bmi_value <= 29.9:
                return 'Overweight'
            else:
                return 'Obese'
            
        input_data['bmi_category'] = input_data['bmi'].apply(get_bmi_category)   
            


        input_data = input_data[features]
        prediction = model.predict(input_data)[0]

        return jsonify({'prediction' : prediction})
    
    except Exception as e:
        return jsonify({'error':str(e)}) , 400

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
