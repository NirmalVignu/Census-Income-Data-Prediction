from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline
from src.pipeline.training_pipeline import train_model

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='GET':
        return render_template('form.html')
    else:
        data=CustomData(
            age=int(request.form['age']),
            workclass=request.form['workclass'],
            education_num=int(request.form['education_num']),
            capital_gain=int(request.form['capital_gain']),
            capital_loss=int(request.form['capital_loss']),
            hours_per_week=int(request.form['hours_per_week']),
            marital_status=request.form['marital_status'],
            occupation=request.form['occupation'],
            relationship=request.form['relationship'],
            race=request.form['race'],
            native_country=request.form['native_country'],
            sex=request.form['sex']
)
        final_new_data=data.get_data_as_dataframe()
        if None in final_new_data.values:
            print('None values found in data')
            print(final_new_data.values)
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html',final_result=results)

@app.route('/train',methods=['GET','POST'])
def train():
    report=train_model()
    model_name='LogisticRegression'
    final={'report':report,'model_name':model_name}
    return render_template('train_model.html',final=final)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)

