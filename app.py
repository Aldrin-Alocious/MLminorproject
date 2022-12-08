import pandas as pd
import streamlit as st
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
op=model.predict([[5.7,4.4,1.5,0.4]])
if op==0:
  print("Iris Setosa")
elif op==1:
  print("Iris Versicolor")
elif op==2:
  print("Iris Virginica")
