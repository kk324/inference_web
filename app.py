import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request, make_response



# initialize our Flask application and pre-trained model
app = Flask(__name__)



@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/uploads', methods=['POST'])
def wait():
    file = request.files['file']
    file.save("./static/" + file.filename)

# 　推論
    df_train_x=pd.read_csv("train_x.csv")
    df_test_x=pd.read_csv(file.filename)

    # 列内に欠損がある列を削除
    train_X=df_train_x.dropna(how='any', axis=1)
    test_X=df_test_x.dropna(how='any', axis=1)
    

    # テストデータと訓練データのcolumnsの内容を合わせる
    test_X=test_X.drop(columns=['（派遣先）職場の雰囲気', '（派遣先）配属先部署'])  

    # 列内に文字列が含まれている列を削除
    train_X=train_X.drop(columns=train_X.select_dtypes(include='object').columns)
    test_X=test_X.drop(columns=test_X.select_dtypes(include='object').columns)

    # テストデータと訓練データのcolumnsの順序を合わせる
    train_X=train_X[test_X.columns]

    model = pickle.load(open('model.h5', 'rb'))

    y_pred = model.predict(test_X)

    df=pd.Series(y_pred, name="応募数 合計")
    df=pd.DataFrame(df)

    df_submit=pd.concat([test_X["お仕事No."], df], axis=1)

    df_submit.to_csv("./submit/submit_" + file.filename, index=False)

    # 推論後のファイル名を取得する
    global file_name 
    file_name = file.filename


    return render_template('download.html')

@app.route('/get', methods=['POST'])
def get():
    downloadFileName = file_name
    response = make_response()

    #ファイル名を与える
    response.data = open("./submit/submit_" + downloadFileName, "rb").read()
    response.minetype = "text/csv"
    
    response.headers['Content-Disposition'] = 'attachment; filename=submit_' + downloadFileName
    return response


if __name__ == "__main__":
    app.debug = True
    app.run()
