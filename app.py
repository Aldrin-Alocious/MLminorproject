import pandas as pd
import streamlit as st
st.title('IRIS CLASSIFIER')
st.subheader('Sepal Length')
ip1=st.slider('Enter a value between 0 & 10:', min_value=0.0, max_value=10.0, step=0.1, key=1)
st.subheader('Sepal Width')
ip2=st.slider('Enter a value between 0 & 10:', min_value=0.0, max_value=10.0, step=0.1, key=2)
st.subheader('Petal Length')
ip3=st.slider('Enter a value between 0 & 10:', min_value=0.0, max_value=10.0, step=0.1, key=3)
st.subheader('Petal Width')
ip4=st.slider('Enter a value between 0 & 10:', min_value=0.0, max_value=10.0, step=0.1, key=4)
df=pd.read_csv("Book1.csv")
arr = df.to_numpy()
ndata=arr[:,:4]
ntarget=arr[:,4:]
lis=[]
for i in range(0,150):
  if ntarget[i][0]=='I.\xa0setosa':
    lis.append(0)
  elif ntarget[i][0]=='I.\xa0versicolor':
    lis.append(1)
  elif ntarget[i][0]=='I.\xa0virginica':
    lis.append(2)
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(ndata,lis)
op=model.predict([[ip1,ip2,ip3,ip4]])
if op==0:
  st.write("Iris Setosa")
elif op==1:
  st.write("Iris Versicolor")
elif op==2:
  st.write("Iris Virginica")
