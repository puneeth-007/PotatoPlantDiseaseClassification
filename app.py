import streamlit as st
import pandas as pd
import numpy as np
import pickle
import cv2
from PIL import Image# Saves

st.markdown('<h1 style="text-align: center; color: white;">Potato Disease Classification</h1>',unsafe_allow_html=True)

lr=pickle.load(open(r'lr_pickle.pkl','rb'))
knn=pickle.load(open(r'kn_pickle.pkl','rb'))
dt=pickle.load(open(r'dt_pickle.pkl','rb'))
gnb=pickle.load(open(r'gnb_pickle.pkl','rb'))
xgb=pickle.load(open(r'xgb_pickle.pkl','rb'))
rf=pickle.load(open(r'rf_pickle.pkl','rb'))
svc=pickle.load(open(r'svc_pickle.pkl','rb'))


enable = st.checkbox("Enable camera")
picture = st.camera_input("Take a picture", disabled=not enable)

upload=st.file_uploader('upload an image file',type=['jpg','png'])

if upload is not None:
    with open(upload.name,'wb') as f:
        f.write(upload.getbuffer())  # here we are saving the file locally as upload.name 

if picture is not None:
    with open(picture.name,'wb') as f:
        f.write(picture.getbuffer())

def sh(st):
    if st==0:
        return 'Early_blight'
    elif st==1:
        return 'Healthy'
    else:
        return 'Late_blight'


if upload is not None:
    img = Image.open(upload)
    img = img.save("img.jpg")
    im=cv2.imread(upload.name,0)
    im=cv2.resize(im,(100,100))
    im=im.flatten()
    st.header('Output for photo uploaded')
    st.write(f'Predicted class in :blue[logistic regression] is :blue[{sh(lr.predict([im])[0])}]')
    st.write(f'Predicted class in :blue[knn] is :blue[{sh(knn.predict([im])[0])}]')
    st.write(f'Predicted class in :blue[decision tree] is :blue[{sh(dt.predict([im])[0])}]')
    st.write(f'Predicted class in :blue[navie bayes] is :blue[{sh(gnb.predict([im])[0])}]')
    st.write(f'Predicted class in :blue[xgboost] is :blue[{sh(xgb.predict([im])[0])}]')
    st.write(f'Predicted class in :blue[random forest] is :blue[{sh(rf.predict([im])[0])}]')
    st.write(f'Predicted class in :blue[svc] is :blue[{sh(svc.predict([im])[0])}]')
    result=sh(svc.predict([im])[0])
    st.markdown(f'<h1 style="text-align: center; color: white;">SVC has highest accuracy and result is <span style="color: blue;">{result}</span> </h1>',unsafe_allow_html=True)

if picture is not None:
    img = Image.open(picture)
    img = img.save("pic.jpg")
    im=cv2.imread(picture.name,0)
    im=cv2.resize(im,(100,100))
    im=im.flatten()
    st.header('Output for photo taken')
    st.write(f'Predicted class in :blue[logistic regression] is :blue[{sh(lr.predict([im])[0])}]')
    st.write(f'Predicted class in :blue[knn] is :blue[{sh(knn.predict([im])[0])}]')
    st.write(f'Predicted class in :blue[decision tree] is :blue[{sh(dt.predict([im])[0])}]')
    st.write(f'Predicted class in :blue[navie bayes] is :blue[{sh(gnb.predict([im])[0])}]')
    st.write(f'Predicted class in :blue[xgboost] is :blue[{sh(xgb.predict([im])[0])}]')
    st.write(f'Predicted class in :blue[random forest] is :blue[{sh(rf.predict([im])[0])}]')
    st.write(f'Predicted class in :blue[svc] is :blue[{sh(svc.predict([im])[0])}]')
    result=sh(svc.predict([im])[0])
    st.markdown(f'<h1 style="text-align: center; color: white;">SVC has highest accuracy and result is <span style="color: blue;">{result}</span> </h1>',unsafe_allow_html=True)


