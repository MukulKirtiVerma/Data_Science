# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:12:07 2021

@author: Mukul Kirti Verma
"""
"""
Series is a one-dimensional labeled array capable of holding data of any type (integer, string, float, python objects, etc.). The axis labels are collectively called index.
"""

import pandas as pd
s=pd.Series()

pandas.Series( data, index, dtype)
s=pd.Series(data=[1,2,3],index=['a','b','c'],dtype=float)

1	
data

data takes various forms like ndarray, list, constants

2	
index

Index values must be unique and hashable, same length as data. Default np.arrange(n) if no index is passed.

3	
dtype

dtype is for data type. If None, data type will be inferred


#==============================================
Create an Empty Series
A basic series, which can be created is an Empty Series.

import pandas as pd
s = pd.Series()
print (s)


#========================================
Create a Series from ndarray

data = np.array(['a','b','c','d'])
s = pd.Series(data)

with index:
import numpy as np
data = np.array(['a','b','c','d'])
s = pd.Series(data,index=[100,101,102,103,105])


Create a Series from dict
data = {'a' : 0., 'b' : 1., 'c' : 2.}

s = pd.Series(data,index=['b','c','d','a'])


Create a Series from Scalar
s = pd.Series(5, index=[0, 1, 2, 3])


Accessing Data from Series with Position

s = pd.Series([1,2,3,4,5],index = ['a','b','c','d','e'])

#retrieve the first element
print (s[0])

#retrieve the first three element
print (s[:3])

Retrieve the last three elements.
print (s[-3:])


Retrieve Data Using Label (Index)

s = pd.Series([1,2,3,4,5],index = ['a','b','c','d','e'])

#retrieve a single element
print (s['a'])

Retrieve multiple elements using a list of index label values.


s = pd.Series([1,2,3,4,5],index = ['a','b','c','d','e'])

#retrieve multiple elements
print (s[['a','c','d']])



If a label is not contained, an exception is raised.
print( s['f'])



DataFrame
#can be created from :
Lists
dict
Series
Numpy ndarrays
Another DataFrame


#empty
import pandas as pd
df = pd.DataFrame()
print( df)


Create a DataFrame from Lists
data = [[1,2,3,4,5],[6,7,8,9,0]]
df = pd.DataFrame(data)
print (df)


data = [['Alex',10],['Bob',12],['Clarke',13]]
df = pd.DataFrame(data,columns=['Name','Age'])
print (df)

df.columns=['First_Name','Age']
df.index=[100,200,300]


df = pd.DataFrame(data,columns=['Name','Age'],dtype=str)
print (df)
int
float
str
bolean


Create a DataFrame from Dict of ndarrays / Lists
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42,30]}
df = pd.DataFrame(data)
print (df)


create an indexed DataFrame using arrays.
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df = pd.DataFrame(data, index=['rank1','rank2','rank3','rank4'])
print (df)


Create a DataFrame from List of Dicts
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data)
print (df)


data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]

#With two column indices, values same as dictionary keys
df1 = pd.DataFrame(data, index=['first', 'second'], columns=['a', 'b'])

#With two column indices with one index with other name
df2 = pd.DataFrame(data, index=['first', 'second'], columns=['a', 'b1'])


Create a DataFrame from Dict of Series

d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)



Column Selection
d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
print (df ['one'])



Column Addition


df['three']=pd.Series([10,20,30],index=['a','b','d'])
print (df)

print ("Adding a new column using the existing columns in DataFrame:")
df['four']=df['one']+df['three']

print (df)



Column Deletion
#del func
del df['one']

#pop
df.pop('two')

Row Selection, Addition, and Deletion
We will now understand row selection, addition and deletion through examples. Let us begin with the concept of selection.

Selection by Label
Rows can be selected by passing row label to a loc function.


d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']), 
   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
print( df.loc['b'])


df['one']['d']
df['one']['d']=3

df.loc[3]

Selection by integer location
print (df.iloc[2])


print (df.iloc[1,4])



Slice Rows
Multiple rows can be selected using ‘ : ’ operator.
df = pd.DataFrame(d)
print (df[2:4])



Addition of Rows
df = pd.DataFrame([[1, 2], [3, 4]], columns = ['a','b'])
df2 = pd.DataFrame([[5, 6], [7, 8]], columns = ['a','b'])

df = df.append(df2)
print (df)


Deletion of Rows
df = df.drop(0)



Series Basic Functionality
s = pd.Series([1,2,3,4,5,7,8,9])
axes

print ("The axes are:")
print (s.axes)


Empty

print ("Is the Object empty?")
print (s.empty)


ndim
print ("The dimensions of the object:")
print (s.ndim)

size
print (s.size)


values
print(s.values)

print ("The first 5 rows of the data series:")
print (s.head())
print ("The first two rows of the data series:")
print(s.head(6))


tail
print ("The last 5 rows of the data series:")
print (s.tail())
print ("The last 2 rows of the data series:")
print (s.tail(2))




DataFrame basic function
s = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack']),
   'Age':pd.Series([25,26,25,23,30,29,23]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}
s=pd.DataFrame(s)

axes

print ("The axes are:")
print (s.axes)

Transpose
print (s.T)


Empty

print ("Is the Object empty?")
print (s.empty)

dtype
print (d.dtypes)

ndim
print ("The dimensions of the object:")
print (s.ndim)

size
print (s.size)

	
shape
print(s.shape)


values
print(s.values)

print ("The first 5 rows of the data series:")
print (s.head())
print ("The first two rows of the data series:")
print(s.head(2))


tail
print ("The last 5 rows of the data series:")
print (s.tail())
print ("The last 2 rows of the data series:")
print (s.tail(2))





import pandas as pd
import numpy as np

#read excel file===============================================
df = pd.read_excel(r"C:\Users\Mukul Kirti Verma\Downloads\ExcelTestData1.xlsx")
#display only first 5 rows=====================================
df.head()
#display first 10 rows=========================================
df.head(10)
#display last 5 rows===========================================
df.tail()

#Assign a value in row column==================================
df.loc[0:1,'MD']=np.nan
print(df)
#replace nan to zero in dataframe=============================
df=df.fillna(0)
#replace set of values with other set of valus================
df2=df.replace([0,2], [5,6])
#replace a range
df.replace(list(range(0,1000)), 0)
#replace set of value with  dectionary==========================
df.replace({0: 10, 2: 100})
df.replace({'MD': 0, 'DT1': 66}, 100)
df.replace({'MD': {0: 100, 4: 400}})
#replace with condition========================================
df[df<=2]=1

#list all column name==========================================
print(list(df.columns))
#list all index================================================
print(list(df.index))

r=list(df.index)
c=list(df.columns)
#iterate dataframe ============================================
for i in df.index:
    x=list(df.loc[i][:])
    print(x)

#adding new column having sum of other columns=================
df["total"] = df["MD"] + df["DT1"] + df["RHOB1"]
df.head()

#some functions================================================
df["MD"].sum(), #sum ofMD column
df["MD"].mean(),
df["MD"].min(),
df["MD"].max()

df[["MD","DT1"]].max()#return seriese with max of both columns

#add row having sum of DT1 and RHOB1 column====================
sum_row=df[["DT1","RHOB1"]].sum()
sum_row

df_sum=pd.DataFrame(data=sum_row).T
df_sum

df_sum=df_sum.reindex(columns=df.columns)
df_sum


df_final=df.append(df_sum,ignore_index=True)
df_final.tail()


#inserting a col in particular position
df_final.insert(4, "abb",2)
df_final.insert(4, "abb22", np.nan)
df_final.insert(4, "abb1", df['MD'])
df_final.insert(4, "abb2", "hi")
df_final.head()

#some grouping example
ddd=df.sort_values(by=['DT1'])
ddd=df.groupby(['DT1']).count()
ddd=df.groupby(['DT1']).first()
ddd=df.groupby(['DT1']).max()
ddd=df.groupby(['DT1']).min()

df_sub=df[["MD","RHOB1","DT1"]].groupby('DT1').sum()
df_sub

#writing dataframe to excel file with diffent sheet
writer = pd.ExcelWriter(r'D:\tut\output.xlsx')
ddd.to_excel(writer,'Sheet1')
df_final.to_excel(writer,'Sheet2')
writer.save()

#parsing diffrent sheet of excel file
xl = pd.ExcelFile(r"D:\tut\output.xlsx")
xl2=xl.sheet_names
df = xl.parse("Sheet1")
df.head()


subham = pd.read_excel(r"D:\tut\output.xlsx",sheetname='Sheet2')
df
dframe = pd.read_excel(r"D:\tut\output.xlsx", sheetname='Sheet1')
df
df = pd.read_excel(r"D:\tut\output.xlsx",sheetname=1)

df= pd.read_excel(r"D:\tut\output.xlsx",sheetname=1,header=None)

dframe = pd.read_excel(r"D:\tut\output.xlsx", sheetname=1,header=1)
dframe1 = pd.read_excel(r"D:\tut\output.xlsx", sheetname=1)

dframe = pd.read_excel(r"D:\tut\output.xlsx",sheetname=1, index_col=2)

dframe = pd.read
dframe = pd.read_excel(r"D:\tut\output.xlsx",sheetname=1, skiprows=2)

df=pd.read_excel(r"D:\tut\output.xlsx", sheetname=1,skip_footer=2)
dframe = pd.read_excel(r"D:\tut\output.xlsx", skip_footer=2)
dframe = pd.read_excel(r"D:\tut\output.xlsx", sheetname=1, skiprows=2,skip_footer=2)
print(str(list(df)))
for i in list(df.index):
    for j in list(df):
      print(df.loc[i,j],end="       ")
    print()
for i in df['MD']:
    print(i)



