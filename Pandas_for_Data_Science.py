# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:12:07 2021

@author: Mukul Kirti Verma
"""
"""
Series is a one-dimensional labeled array capable of holding data of any type (integer, string, float, python objects, etc.). The axis labels are collectively called index.
"""

pandas.Series( data, index, dtype)

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

with index
data = np.array(['a','b','c','d'])
s = pd.Series(data,index=[100,101,102,103])


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
data = [1,2,3,4,5]
df = pd.DataFrame(data)
print (df)


data = [['Alex',10],['Bob',12],['Clarke',13]]
df = pd.DataFrame(data,columns=['Name','Age'])
print (df)
df.columns
df.index

df = pd.DataFrame(data,columns=['Name','Age'],dtype=float)
print (df)


Create a DataFrame from Dict of ndarrays / Lists
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
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

d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)

# Adding a new column to an existing DataFrame object with column label by passing new series

print ("Adding a new column by passing as Series:")
df['three']=pd.Series([10,20,30],index=['a','b','c'])
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


Selection by integer location
print (df.iloc[2])


print (df.iloc[2,2])



Slice Rows
Multiple rows can be selected using ‘ : ’ operator.
df = pd.DataFrame(d)
print df[2:4]



Addition of Rows
df = pd.DataFrame([[1, 2], [3, 4]], columns = ['a','b'])
df2 = pd.DataFrame([[5, 6], [7, 8]], columns = ['a','b'])

df = df.append(df2)
print (df)


Deletion of Rows
df = df.drop(0)



Series Basic Functionality
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
print(s.head(2))


tail
print ("The last 5 rows of the data series:")
print (s.tail())
print ("The last 2 rows of the data series:")
print (s.tail(2))




DataFrame basic function
s = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack']),
   'Age':pd.Series([25,26,25,23,30,29,23]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}

axes

print ("The axes are:")
print (s.axes)

Transpose
print (df.T)


Empty

print ("Is the Object empty?")
print (s.empty)


dtype
print (df.dtypes)

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


axes

print ("The axes are:")
print (s.axes)
s.view
dir(s)
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
print(s.head(2))


tail
print ("The last 5 rows of the data series:")
print (s.tail())
print ("The last 2 rows of the data series:")
print (s.tail(2))



Descriptive Statistics

import pandas as pd
import numpy as np

#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky',\
                       'Vin','Steve','Smith','Jack',])
   }

#Create a DataFrame
df = pd.DataFrame(d)
print df

print (df.sum())
axis=1
print (df.sum(1))


print (df.mean())
df.mean(1)


df.std()


df=pd.DataFrame([1,1,2,2,2,3,4,5,6])
df.count()
df.cumsum()

1   count()	Number of non-null observations
2	sum()	Sum of values
3	mean()	Mean of Values
4	median()	Median of Values
5	mode()	Mode of values
6	std()	Standard Deviation of the Values
7	min()	Minimum Value
8	max()	Maximum Value
9	abs()	Absolute Value
10	prod()	Product of Values
11	cumsum()	Cumulative Sum

df.cumsum()

only numeric

df.describe()
df=pd.DataFrame(['b','c','a'])

df.describe(include=['object'])

Fucntion
Table wise Function Application: pipe()
Row or Column Wise Function Application: apply()
Element wise Function Application: applymap()


df=pd.DataFrame(['1','2','3'])
df-'a'


def adder(ele1,ele2):
   return ele1+ele2

df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
df.pipe(adder,2)
df=df.applymap(lambda  x: x+np.mean(df.values))

df.applymap(lambda x:x*100)


import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
print (df.apply(np.mean))



df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
df['result']=df.apply(lambda x: x.max() - x.min(),1)
print (df.apply(np.mean))
df+1

Element Wise Function Application
df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])

# My custom function
df['col1'].map(lambda x:x*100)
print df.apply(np.mean)


Reindexing
Reorder the existing data to match a new set of labels.
Insert missing value (NA) markers in label locations where no data for the label existed.

N=20

df = pd.DataFrame({
   'A': pd.date_range(start='2016-01-01',periods=N,freq='D'),
   'x': np.linspace(0,stop=N-1,num=N),
   'y': np.random.rand(N),
   'C': np.random.choice(['Low','Medium''High'],N).tolist(),
   'D': np.random.normal(100, 10, size=(N)).tolist()
})
df.reindex(index=[0,1,3],columns=['A','C','f'])
#reindex the DataFrame
df_reindexed = df.reindex(index=[0,2,5], columns=['A', 'C', 'B'])

print( df_reindexed)

df1 = pd.DataFrame(np.random.randn(10,3),columns=['col1','col2','col3'])
df2 = pd.DataFrame(np.random.randn(7,3),columns=['col1','col2','col3'])

df1 = df2.reindex_like(df1)
print (df1)



# Padding NAN's
print df2.reindex_like(df1)

# Now Fill the NAN's with preceding Values
print ("Data Frame with Forward Fill:")
print (df2.reindex_like(df1,method='ffill'))


# Padding NAN's
print df2.reindex_like(df1)

# Now Fill the NAN's with preceding Values
print ("Data Frame with Forward Fill limiting to 1:")
print (df2.reindex_like(df1,method='ffill',limit=1))

df=pd.DataFrame([1,2,3,4,5])
df.index=[100,102,103,104,105]
df.columns=['a']
df1.columns=['a','b']

df1 = pd.DataFrame(np.random.randn(6,3),columns=['col1','col2','col3'])
print (df1)

print ("After renaming the rows and columns:")
print( df1.rename(columns={'col1' : 'c1', 'col2' : 'c2'},index = {0 : 'apple', 1 : 'banana', 2 : 'durian'}))



N=20
df = pd.DataFrame({
   'A': pd.date_range(start='2016-01-01',periods=N,freq='D'),
   'x': np.linspace(0,stop=N-1,num=N),
   'y': np.random.rand(N),
   'C': np.random.choice(['Low','Medium','High'],N).tolist(),
   'D': np.random.normal(100, 10, size=(N)).tolist()
   })



for col in df:
   print( col)

for i in df.columns:
    print(df[i])


   
   
import pandas as pd
import numpy as np

#read excel file===============================================
df = pd.read_excel(r"C:\Users\Mukul Kirti Verma\Downloads\ExcelTestData1.xlsx")
#display only first 5 rows=====================================


df.describe()
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
df['MD'][df['MD']==1]=2

#list all column name==========================================
print(list(df.columns))
#list all index================================================
print(list(df.index))

r=list(df.index)
c=list(df.columns)
#iterate dataframe ============================================
for i in list(df.index):
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
