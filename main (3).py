from flask import Flask,render_template,url_for,request
import pickle
import pandas as pd

app=Flask(__name__)

xgb_filename = 'classifier.pkl'
classifier = pickle.load(open(xgb_filename, 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction',methods=['POST','GET'])
def predict():
    if request.method=="POST":  

        duration=request.form['duration']
        service=request.form['service']
        flag=request.form['flag']
        src_bytes=request.form['src_bytes']
        dst_bytes=request.form['dst_bytes']
        land=request.form['land']
        wrong_fragment=request.form['wrong_fragment']
        urgent=request.form['urgent']
        hot=request.form['hot']
        num_failed_logins=request.form['num_failed_logins']

        DATAS=[duration,service,flag,src_bytes,dst_bytes,land,wrong_fragment,urgent,hot,num_failed_logins]

        new_data_df = pd.DataFrame([DATAS], columns=['duration', 'service', 'flag', 'src_bytes', 'dst_bytes',
                                               'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins'])

# Use the trained classifier to make predictions on the new data
        y_pred_new_data = classifier.predict(new_data_df)

        if y_pred_new_data[0] == "normal":
            value="The data is predicted as normal."
        elif y_pred_new_data[0] =="dos":
            value="The data is predicted as dos."
        elif y_pred_new_data[0]=="probe":
            value="The data is predicted as probe."
        elif y_pred_new_data[0]=="r2l":
            value="The data is predicted as r2l."

        return render_template('predict.html',value=value)
    return render_template('predict.html')



if __name__=="__main__":
    app.run(debug=True)

