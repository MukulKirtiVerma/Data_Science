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
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
   'Lee','David','Gasper','Betina','Andres']),
   'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])
}

#Create a DataFrame
df = pd.DataFrame(d)
print df

print df.sum()
axis=1
print df.sum(1)


print df.mean()
df.mean(1)


df.std()


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


df.describe(include=['object'])

Fucntion
Table wise Function Application: pipe()
Row or Column Wise Function Application: apply()
Element wise Function Application: applymap()


def adder(ele1,ele2):
   return ele1+ele2

df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
df.pipe(adder,2)
df.applymap(lambda x:x*100)


import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
print (df.apply(np.mean))



df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
df.apply(lambda x: x.max() - x.min())
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
   'C': np.random.choice(['Low','Medium','High'],N).tolist(),
   'D': np.random.normal(100, 10, size=(N)).tolist()
})

#reindex the DataFrame
df_reindexed = df.reindex(index=[0,2,5], columns=['A', 'C', 'B'])

print( df_reindexed)

df1 = pd.DataFrame(np.random.randn(10,3),columns=['col1','col2','col3'])
df2 = pd.DataFrame(np.random.randn(7,3),columns=['col1','col2','col3'])

df1 = df1.reindex_like(df2)
print (df1)



# Padding NAN's
print df2.reindex_like(df1)

# Now Fill the NAN's with preceding Values
print ("Data Frame with Forward Fill:")
print df2.reindex_like(df1,method='ffill')


# Padding NAN's
print df2.reindex_like(df1)

# Now Fill the NAN's with preceding Values
print ("Data Frame with Forward Fill limiting to 1:")
print df2.reindex_like(df1,method='ffill',limit=1)


df1 = pd.DataFrame(np.random.randn(6,3),columns=['col1','col2','col3'])
print (df1)

print ("After renaming the rows and columns:")
print( df1.rename(columns={'col1' : 'c1', 'col2' : 'c2'},index = {0 : 'apple', 1 : 'banana', 2 : 'durian'}))




Indexing and Selecting Data
three way

Sr.No	Indexing & Description
1	
.loc()

Label based

2	
.iloc()

Integer based

3	
.ix()

.loc()
Pandas provide various methods to have purely label based indexing. 
When slicing, the start bound is also included. Integers are valid labels, 
but they refer to the label and not the position.

.loc() has multiple access methods like −

A single scalar label
A list of labels
A slice object
A Boolean array
loc takes two single/list/range operator separated by ','. 
The first one indicates the row and the second one indicates columns.


df = pd.DataFrame(np.random.randn(8, 4),
index = ['a','b','c','d','e','f','g','h'], columns = ['A', 'B', 'C', 'D'])

#select all rows for a specific column
print df.loc[:,'A']

# Select all rows for multiple columns, say list[]
print df.loc[:,['A','C']]


# Select few rows for multiple columns, say list[]
print df.loc[['a','b','f','h'],['A','C']]



# Select range of rows for all columns
print df.loc['a':'h']

# for getting values with a boolean array
print df.loc['a']>0


.iloc()
Pandas provide various methods in order to get purely integer based indexing. Like python and numpy, these are 0-based indexing.

The various access methods are as follows −

An Integer
A list of integers
A range of values



df = pd.DataFrame(np.random.randn(8, 4), columns = ['A', 'B', 'C', 'D'])

# select all rows for a specific column
print df.iloc[:4]


# Integer slicing
print df.iloc[:4]
print df.iloc[1:5, 2:4]



# Slicing through list of values
print df.iloc[[1, 3, 5], [1, 3]]
print df.iloc[1:3, :]
print df.iloc[:,1:3]



# Integer slicing
print df.ix[:4]


df = pd.DataFrame(np.random.randn(8, 4), columns = ['A', 'B', 'C', 'D'])
# Index slicing
print df.ix[:,'A']



#print single column
df = pd.DataFrame(np.random.randn(8, 4), columns = ['A', 'B', 'C', 'D'])
print df['A']


#print Multiple column
print df[['A','B']]


#with .column
print df.A

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
df.replace(list(range(0,1001)), 0)
#replace set of value with  dectionary==========================
df.replace({0: 10, 2: 100})
df.replace({'MD': 0, 'DT1': 66}, 100)
df.replace({'MD': {0: 100, 2000: 400}})
#replace with condition========================================

df[df<=2].fillna(0)
df[df<=2]=1

#list all column name==========================================
print(list(df.columns))
#list all index================================================
print(list(df.index))



#adding new column having sum of other columns=================
df["total"] = df["MD"] + df["DT1"] + df["RHOB1"]
df.head()

#some functions================================================
df["MD"].sum(), #sum ofMD column
df["MD"].mean(),
df["MD"].min(),
df["MD"].max()

df[["MD","DT1"]].max()#return seriese with max of both columns
df[list(df.columns)].max()
#add row having sum of DT1 and RHOB1 column====================
sum_row=df[["DT1","RHOB1"]].sum()
sum_row

df_sum=pd.DataFrame(data=sum_row).T
df_sum


df_sum=df_sum.reindex(columns=df.columns)
df_sum


df.append(df_sum)


df_final=df.append(df_sum,ignore_index=True)
df_final.tail()


#inserting a col in particular position
df_final.insert(4, "abb",2)
df_final.insert(4, "abb22", np.nan)
df['abb1']=df['MD']
df_final.insert(2, "abb1", df['MD'])
df_final.insert(4, "abb2", "hi")
df_final.head()

#some grouping example
ddd=df.sort_values(by=['DT1'])
ddd=df.groupby(['DT1']).count()

dd=pd.DataFrame([[1,1],[1,12],[1,2],[2,2],[2,2],[3,3]],columns=['a','b'])
dd.groupby(by=['b']).count()

ddd=df.groupby(['DT1']).first()
ddd=df.groupby(['DT1']).max()
ddd=df.groupby(['DT1']).min()

df_sub=df[["MD","RHOB1","DT1"]].groupby('DT1').sum()
df_sub

#writing dataframe to excel file with diffent sheet
writer = pd.ExcelWriter(r'D:\output.xlsx')
df_sub.to_excel(writer,'Sheet3')
ddd.to_excel(writer,'Sheet1')
df_final.to_excel(writer,'Sheet2')

writer.save()

#parsing diffrent sheet of excel file
xl = pd.ExcelFile(r"D:\output.xlsx")
xl2=xl.sheet_names
df = xl.parse(xl2[0])
df.head()


df= pd.read_excel(r"D:\output.xlsx",sheet_name='Sheet2')
df
dframe = pd.read_excel(r"D:\tut\output.xlsx", sheetname='Sheet1')
df
df = pd.read_excel(r"D:\output.xlsx",sheet_name=1)
df
df= pd.read_excel(r"D:\output.xlsx",sheet_name=2,header=None)
df.head(6)
dframe = pd.read_excel(r"D:\output.xlsx", sheet_name=2)
dframe
dframe1 = pd.read_excel(r"D:\output.xlsx", sheet_name=1)

dframe = pd.read_excel(r"D:\output.xlsx",sheet_name=2, index_col=4)

dframe = pd.read
dframe = pd.read_excel(r"D:\output.xlsx",sheet_name=2, skiprows=4)

df=pd.read_excel(r"D:\output.xlsx", sheet_name=2,skipfooter=5)
dframe = pd.read_excel(r"D:\output.xlsx", skip_footer=2)
dframe = pd.read_excel(r"D:\output.xlsx", sheet_name=2, skiprows=2,skipfooter=2)



print(str(list(df)))
for i in list(df.index):
    for j in list(df):
      print(df.loc[i,j],end="       ")
    print()
for i in df['MD']:
    print(i)
    
    
    
To iterate over the rows of the DataFrame, 
we can use the following functions −

iteritems() − to iterate over the (key,value) pairs

iterrows() − iterate over the rows as (index,series) pairs

itertuples() − iterate over the rows as namedtuples


import pandas as pd
import numpy as np
 
df = pd.DataFrame(np.random.randn(4,3),columns=['col1','col2','col3'])
for key,value in df.iteritems():
   print (key,value)
   
   

for row_index,row in df.iterrows():
   print( row_index,row)
   
   
for row in df.itertuples():
    print (row)



Sorting

There are two kinds of sorting available in Pandas. 
They are −

By label
By Actual Value

unsorted_df=pd.DataFrame(np.random.randn(10,2),index=[1,4,6,2,3,5,9,8,0,7],columns=['col2','col1'])
print (unsorted_df)


By Label
Using the sort_index() method, by passing the axis arguments and the order of sorting, 
DataFrame can be sorted. By default, sorting is done on row labels in ascending order.



sorted_df=unsorted_df.sort_index()
print( sorted_df)


unsorted_df = pd.DataFrame(np.random.randn(10,2),index=[1,4,6,2,3,5,9,8,0,7],columns = ['col2','col1'])

sorted_df = unsorted_df.sort_index(ascending=False)
print (sorted_df)


unsorted_df = pd.DataFrame(np.random.randn(10,2),index=[1,4,6,2,3,5,9,8,0,7],columns = ['col2','col1'])
 
sorted_df=unsorted_df.sort_index(axis=1,ascending=False)

print (sorted_df)


By Value
Like index sorting, sort_values() is the method for sorting by values. It accepts a 'by' argument which will use the column name of the DataFrame with which the values are to be sorted.


unsorted_df = pd.DataFrame({'col1':[2,1,1,1],'col2':[1,3,2,4]})
sorted_df = unsorted_df.sort_values(by='col1')

print( sorted_df)


By

unsorted_df = pd.DataFrame({'col1':[2,1,1,1],'col2':[1,3,2,4]})   
sorted_df = unsorted_df.sort_values(by=['col1','col2'])

print (sorted_df)


kind
unsorted_df = pd.DataFrame({'col1':[2,1,1,1],'col2':[1,3,2,4]})
sorted_df = unsorted_df.sort_values(by='col1' ,kind='mergesort')

print (sorted_df)


"""

string function
Sr.No	Function & Description
1	
lower()

Converts strings in the Series/Index to lower case.

2	
upper()

Converts strings in the Series/Index to upper case.

3	
len()

Computes String length().

4	
strip()

Helps strip whitespace(including newline) from each string in the Series/index from both the sides.

5	
split(' ')

Splits each string with the given pattern.

6	
cat(sep=' ')

Concatenates the series/index elements with given separator.

7	
get_dummies()

Returns the DataFrame with One-Hot Encoded values.

8	
contains(pattern)

Returns a Boolean value True for each element if the substring contains in the element, else False.

9	
replace(a,b)

Replaces the value a with the value b.

10	
repeat(value)

Repeats each element with specified number of times.

11	
count(pattern)

Returns count of appearance of pattern in each element.

12	
startswith(pattern)

Returns true if the element in the Series/Index starts with the pattern.

13	
endswith(pattern)

Returns true if the element in the Series/Index ends with the pattern.

14	
find(pattern)

Returns the first position of the first occurrence of the pattern.

15	
findall(pattern)

Returns a list of all occurrence of the pattern.

16	
swapcase

Swaps the case lower/upper.

17	
islower()

Checks whether all characters in each string in the Series/Index in lower case or not. Returns Boolean

18	
isupper()

Checks whether all characters in each string in the Series/Index in upper case or not. Returns Boolean.

19	
isnumeric()

Checks whether all characters in each string in the Series/Index are numeric. Returns Boolean.
"""

s = pd.Series(['Tom', 'William Rick', 'John', 'Alber@t', np.nan, '1234','SteveSmith'])

print (s.str.lower())





Some more Analytical Function

Percent_change
Series, DatFrames and Panel, all have the function pct_change(). 
This function compares every element with its prior element and computes the change percentage.


import pandas as pd
import numpy as np
s = pd.Series([1,2,3,4,5,4])
print (s.pct_change())

df = pd.DataFrame(np.random.randn(5, 2))
print (df.pct_change())



Covariance
Covariance is applied on series data.
The Series object has a method cov to compute 
covariance between series objects. NA will be excluded automatically.


import pandas as pd
import numpy as np
s1 = pd.Series([1,2,3,4,5,6])
s2 = pd.Series([1,2,3,4,5,6])

print (s1.cov(s2))
s1.plot()


frame = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', 'c', 'd', 'e'])
print (frame['a'].cov(frame['b']))
print (frame.cov())


Correlation
Correlation shows the linear relationship between 
any two array of values (series).
There are multiple methods to compute the correlation 
like pearson(default),
spearman and kendall.



frame = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', 'c', 'd', 'e'])

print (frame['a'].corr(frame['b']))
print (frame.corr())


Window Functions

rolling
df = pd.DataFrame(np.random.randn(10, 4),
   index = pd.date_range('1/1/2000', periods=10),
   columns = ['A', 'B', 'C', 'D'])
print (df.rolling(window=3).sum())

df=pd.DataFrame([[1,1],[2,2],[3,3],[4,4],[5,5]])

.expanding() Function


print( df.expanding(min_periods=3).sum())



Missing Data
 
df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f',
'h'],columns=['one', 'two', 'three'])

df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])

df

print (df['one'].isnull())



print( df['one'].notnull())


When summing data, NA will be treated as Zero
If the data are all NA, then the result will be NA

df['one'].sum()

df = pd.DataFrame(index=[0,1,2,3,4,5],columns=['one','two'])
print( df.sum())



Cleaning / Filling Missing Data

Replace NaN with a Scalar Value
print( df.fillna(0))


Fill NA Forward and Backward
Using the concepts of filling discussed in the ReIndexing Chapter we will fill the missing values.

1	
pad/fill

Fill methods Forward

2	
bfill/backfill

Fill methods Backward

df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f','h'],columns=['one', 'two', 'three'])

df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])

print (df.fillna(method='pad'))

print (df.fillna(method='backfill'))


Drop Missing Values

df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f',
'h'],columns=['one', 'two', 'three'])

df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
print( df.dropna())


df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f',
'h'],columns=['one', 'two', 'three'])

print (df.dropna(axis=1))





Any groupby operation involves one of the following 
operations on the original object.
They are −

1. Splitting the Object

2. Applying a function

3. Combining the results

In many situations, we split the data into sets and we apply some functionality on each subset. 
In the apply functionality, we can perform the following operations −

1. Aggregation − computing a summary statistic

2. Transformation − perform some group-specific operation

3. Filtration − discarding the data with some condition


import pandas as pd

ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
   'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
   'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
   'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
   'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data)

print( df )




Split Data into Groups
Pandas object can be split into any of their objects. 
There are multiple ways to split an object like −

x


# import the pandas library
import pandas as pd

ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
   'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
   'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
   'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
   'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data)

print (df.groupby('Team'))


print (df.groupby('Team').groups)
Its output is as follows −


Group by with multiple columns −





print( df.groupby(['Team','Year']).groups)
Its output is as follows −

{('Kings', 2014): Int64Index([4], dtype='int64'),
 ('Royals', 2014): Int64Index([9], dtype='int64'),
 ('Riders', 2014): Int64Index([0], dtype='int64'),
 ('Riders', 2015): Int64Index([1], dtype='int64'),
 ('Kings', 2016): Int64Index([6], dtype='int64'),
 ('Riders', 2016): Int64Index([8], dtype='int64'),
 ('Riders', 2017): Int64Index([11], dtype='int64'),
 ('Devils', 2014): Int64Index([2], dtype='int64'),
 ('Devils', 2015): Int64Index([3], dtype='int64'),
 ('kings', 2015): Int64Index([5], dtype='int64'),
 ('Royals', 2015): Int64Index([10], dtype='int64'),
 ('Kings', 2017): Int64Index([7], dtype='int64')}
Iterating through Groups
With the groupby object in hand, we can iterate through the object similar to itertools.obj.


grouped = df.groupby('Year')

for name,group in grouped:
   #print (name)
   print (group)


By default, the groupby object has the same label name as the group name.

Select a Group


grouped = df.groupby('Year')
print (grouped.get_group(2014))


Aggregations
An aggregated function returns a single aggregated value for each group. 
Once the group by object is created, several aggregation operations can be performed on 
the grouped data.


grouped = df.groupby('Year')
print (grouped['Points'].agg(np.mean))


Another way to see the size of each group is by applying the size() function −


ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
   'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
   'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
   'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
   'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data)

grouped = df.groupby('Team')
print (grouped.agg(np.size))


Applying Multiple Aggregation Functions at Once
With grouped Series, you can also pass a list or dict of functions to 
do aggregation with, and generate DataFrame as output −



grouped = df.groupby('Team')
print (grouped['Points'].agg([np.sum, np.mean, np.std]))
Its output is as follows −

Team      sum      mean          std
Devils   1536   768.000000   134.350288
Kings    2285   761.666667    24.006943
Riders   3049   762.250000    88.567771
Royals   1505   752.500000    72.831998
kings     812   812.000000          NaN


Transformations
Transformation on a group or a column returns an object that is indexed the same 
size of that is being grouped. Thus, the transform should return a result that is 
the same size as that of a group chunk.


grouped = df.groupby('Team')
score = lambda x: x-max(x)
print( grouped.transform(score))

  
Filtration
Filtration filters the data on a defined criteria and returns the subset of data. 
The filter() function is used to filter the data.


print (df.groupby('Team').filter(lambda x: len(x) >= 3))


"""
Merging/Joining
Pandas has full-featured, high performance in-memory join operations idiomatically very similar to relational databases like SQL.

Pandas provides a single function, merge, as the entry point for all standard database join operations between DataFrame objects −

pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,
left_index=False, right_index=False, sort=True)


Here, we have used the following parameters −

left − A DataFrame object.

right − Another DataFrame object.

on − Columns (names) to join on. Must be found in both the left and right DataFrame objects.

left_on − Columns from the left DataFrame to use as keys. Can either be column names or arrays with length equal to the length of the DataFrame.

right_on − Columns from the right DataFrame to use as keys. Can either be column names or arrays with length equal to the length of the DataFrame.

left_index − If True, use the index (row labels) from the left DataFrame as its join key(s). In case of a DataFrame with a MultiIndex (hierarchical), the number of levels must match the number of join keys from the right DataFrame.

right_index − Same usage as left_index for the right DataFrame.

how − One of 'left', 'right', 'outer', 'inner'. Defaults to inner. Each method has been described below.

sort − Sort the result DataFrame by the join keys in lexicographical order. Defaults to True, setting to False will improve the performance substantially in many cases.
"""

import pandas as pd
left = pd.DataFrame({
   'id':[1,2,3,4,5],
   'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
   'subject_id':['sub1','sub2','sub4','sub6','sub5']})
right = pd.DataFrame(
   {'id':[1,1,2,3,4,5],
   'Name': ['krish','Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
   'subject_id':['sub3''sub2','sub4','sub3','sub6','sub5']})
print (left)
print (right)


print (pd.merge(left,right,on='id'))

print (pd.merge(left,right,on=['id','subject_id']))


Merge Using 'how' Argument

Merge Method	SQL Equivalent	Description
left	        LEFT OUTER     JOIN	Use keys from left object
right	       RIGHT OUTER     JOIN	Use keys from right object
outer	        FULL OUTER     JOIN	Use union of keys
inner	       INNER JOIN	   Use intersection of keys

print (pd.merge(left, right, on='subject_id', how='left'))


print (pd.merge(left, right, on='subject_id', how='right'))

print (pd.merge(left, right, how='outer', on='subject_id'))

print( pd.merge(left, right, on='subject_id', how='inner'))



Concatenation

Pandas provides various facilities for easily combining together Series, DataFrame, and Panel objects.

 pd.concat(objs,axis=0,join='outer',join_axes=None,
ignore_index=False)
objs − This is a sequence or mapping of Series, DataFrame, or Panel objects.

axis − {0, 1, ...}, default 0. This is the axis to concatenate along.

join − {‘inner’, ‘outer’}, default ‘outer’. How to handle indexes on other axis(es). Outer for union and inner for intersection.

ignore_index − boolean, default False. If True, do not use the index values on the concatenation axis. The resulting axis will be labeled 0, ..., n - 1.

join_axes − This is the list of Index objects. Specific indexes to use for the other (n-1) axes instead of performing inner/outer set logic.


import pandas as pd

one = pd.DataFrame({
   'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
   'subject_id':['sub1','sub2','sub4','sub6','sub5'],
   'Marks_scored':[98,90,87,69,78]},
   index=[1,2,3,4,5])

two = pd.DataFrame({
   'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
   'subject_id':['sub2','sub4','sub3','sub6','sub5'],
   'Marks_scored':[89,80,79,97,88]},
   index=[1,2,3,4,5])

pd.concat([one,two])


print (pd.concat([one,two,two]))


Suppose we wanted to associate specific keys with 
each of the pieces of the chopped up DataFrame. 
We can do this by using the keys argument −


print (pd.concat([one,two],keys=['x','y']))


ignore index

print (pd.concat([one,two],keys=['x','y'],ignore_index=True))


axis

print( pd.concat([one,two],axis=1))

append

print (one.append([two,one,two]))




Time Series
Pandas provide a robust tool for working time with Time series data, especially in the financial sector. While working with time series data, we frequently come across the following −

Generating sequence of time
Convert the time series to different frequencies
Pandas provides a relatively compact and self-contained set of tools for performing the above tasks.

Get Current Time
datetime.now() gives you the current date and time.



import pandas as pd

print (pd.datetime.now())



df = pd.DataFrame({'year': [2015, 2016],
                   'month': [2, 3],
                   'day': [4, 5]})
x=pd.to_datetime(df)






Create a TimeStamp

pd.Timestamp('2017-03-01')

pd.Timestamp(25,unit='h')


Create a Range of Time

range(0,10)

pd.date_range(start='1/1/2018', end='1/08/2018')

pd.date_range(start='1/1/2018', periods=8)

pd.date_range(end='1/1/2018', periods=8)

pd.date_range(start='2018-04-24', end='2018-04-27', periods=3)

pd.date_range("11:00", "13:30", periods=5).time


pd.date_range("11:00", "13:30", freq="30min").time


frequency
pd.date_range(start='1/1/2018', periods=5, freq='M')

pd.date_range(start='1/1/2018', periods=5, freq='3M')


pd.date_range(start='1/1/2018', periods=5, tz='Asia/Kolkata')

#inclued start and end
pd.date_range(start='2017-01-01', end='2017-01-04', closed=None)


pd.date_range(start='2017-01-01', end='2017-01-04', closed='left')

pd.date_range(start='2017-01-01', end='2017-01-04', closed='right')

time delta

pd.to_timedelta('1 days 06:05:01.00003')


pd.to_timedelta(['1 days 06:05:01.00003', '15.5us', 'nan'])

Converting numbers by specifying the unit keyword argument:

    
import numpy as np
pd.to_timedelta(np.arange(5), unit='d')

pd.to_timedelta(np.arange(5), unit='d')+\
pd.to_timedelta(np.arange(5), unit='h')


pd.bdate_range(start='6/9/2021',periods=5)


Period Range
pd.period_range(start='2017-01-01', end='2018-01-01', freq='Y')
pd.period_range(start='2017-01-01', end='2018-01-01', freq='M')
pd.period_range(start='2017-01-01', end='2018-01-01', freq='D')
pd.period_range(start='2017-01-01', end='2018-01-01', freq='H')
pd.period_range(start='2017-01-01', end='2018-01-01', freq='S')


timedelta_range
pd.timedelta_range(start='1 day', periods=4)
#TimedeltaIndex(['1 days', '2 days', '3 days', '4 days'],dtype='timedelta64[ns]', freq='D')

pd.timedelta_range(start='1 day', periods=4, closed='right')
#TimedeltaIndex(['2 days', '3 days', '4 days'],dtype='timedelta64[ns]', freq='D')


pd.timedelta_range(start='1 day', end='2 days', freq='6H')
#TimedeltaIndex(['1 days 00:00:00', '1 days 06:00:00', '1 days 12:00:00','1 days 18:00:00', '2 days 00:00:00']
#     dtype='timedelta64[ns]', freq='6H')


#interval range
pd.interval_range(start=0, end=5)



pd.interval_range(start=pd.Timestamp('2017-01-01'),
                  end=pd.Timestamp('2017-01-04'))


pd.interval_range(start=0, end=6, periods=4)


pd.interval_range(end=5, periods=4, closed='both')


Categorical object can be created in multiple ways. 
The different ways have been described below −


category
By specifying the dtype as "category" in pandas object creation.

s = pd.Series(["a","b","c","a"], dtype="category")
print (s)

The number of elements passed to the series object is four, 
but the categories are only three. Observe the same in the output Categories.

pd.Categorical
Using the standard pandas Categorical constructor, we can create a category object.

pandas.Categorical(values, categories, ordered)

import pandas as pd
cat = pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'])
print( cat)
print( cat.categories)

specify category
cat = cat=pd.Categorical(['a','b','c','a','b','c','d'], ['c', 'b', 'a'])
print( cat.)

[a, b, c, a, b, c, NaN]
Categories (3, object): [c, b, a]
Here, the second argument signifies the categories. 
Thus, any value which is not present in the categories will be treated as NaN.

order
cat1=pd.Categorical(['a','b','c','a','b','c','c'], ['c', 'b', 'a'],ordered=True)
cat = pd.Categorical(['a','b','c','b','b','c','c'], ['c', 'b', 'a'],ordered=True)
print(cat)
print (cat.ordered)
cat==cat1

comparing equality (== and !=) to a list-like object (list, Series, array, ...) of the same length as the categorical data.

all comparisons (==, !=, >, >=, <, and <=) of categorical data to another categorical Series, when ordered==True and the categories are the same.



[a, b, c, a, b, c, NaN]
Categories (3, object): [c < b < a]



return list of category
import numpy as np
s = pd.Categorical(["a", "c", "c", np.nan], categories=["b", "a", "c"])
print (s.categories)

output
Index([u'b', u'a', u'c'], dtype='object')




s = pd.Series(["a","b","c","a"], dtype="category")

s.cat.categories = ["Group {}".format(g) for g in s.cat.categories]
print (s.cat.categories)

Appending New Categories
Using the Categorical.add.categories() method, new categories can be appended.


s = pd.Series(["a","b","c","a"], dtype="category")
s = s.cat.add_categories(['e'])
print (s.cat.categories)


Removing Categories
Using the Categorical.remove_categories() method, unwanted categories can be removed.


print( s.cat.remove_categories("a"))



Visualization
 
 
Basic Plotting: plot
This functionality on Series and DataFrame is just a simple wrapper 
around the matplotlib libraries plot() method.

import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(10,4),index=pd.date_range('1/1/2000',
   periods=10), columns=list('ABCD'))
df['A'].plot()



bar or barh for bar plots
hist for histogram
box for boxplot
area for area plots
scatter for scatter plots



1. Bar Plot
Let us now see what a Bar Plot is by creating one.
 A bar plot can be created in the following way −



df = pd.DataFrame(np.random.rand(10,4),columns=['a','b','c','d'])
df.plot.bar()


To produce a stacked bar plot, pass stacked=True −

import pandas as pd
df = pd.DataFrame(np.random.rand(10,4),columns=['a','b','c','d'])
df.plot.bar(stacked=True)



To get horizontal bar plots, use the barh method −

import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.rand(10,4),columns=['a','b','c','d')

df.plot.barh(stacked=True)

[1,2,1,2,4,1,2,4,5,1,3,4,5,2,5,6,2,1]
a=1,2,1,2,1,1,1,3,2,2,1
b=4,4,5,4,5,5,6

Histograms
Histograms can be plotted using the plot.hist() method. We can specify number of bins.

import pandas as pd
import numpy as np

df = pd.DataFrame({'a':np.random.randn(1000)+1,'b':np.random.randn(1000),'c':
np.random.randn(1000) - 1}, columns=['a', 'b', 'c'])

df.plot.hist(bins=20)


To plot different histograms for each column, use the following code −


df=pd.DataFrame({'a':np.random.randn(1000)+1,'b':np.random.randn(1000),'c':np.random.randn(1000) - 1}, columns=['a', 'b', 'c'])



Box Plots
Boxplot can be drawn calling Series.box.plot() and DataFrame.box.plot(),
 or DataFrame.boxplot() to visualize the distribution of values within each column.

For instance, here is a boxplot representing five trials of 10 observations
 of a uniform random variable on [0,1).


df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
df.plot.box()


Area Plot
Area plot can be created using the Series.plot.area() or the DataFrame.plot.area() methods.

df = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
df['a'].plot.area()


Scatter Plot
Scatter plot can be created using the DataFrame.plot.scatter() methods.

import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
df.plot.scatter(x='a', y='b')


Pie Chart
Pie chart can be created using the DataFrame.plot.pie() method.


df = pd.DataFrame(3 * np.random.rand(2), index=['a', 'b', ], columns=['x'])
df.plot.pie(subplots=True)


Using If/Truth Statement with Pandas
Pandas follows the numpy convention of raising an error when you try 
to convert something to a bool. 
This happens in an if or when using the Boolean 
operations, and, or, or not. It is not clear what the result should be.
 Should it be True because it is not zerolength? False because there are False values? 
 It is unclear, so instead, Pandas raises a ValueError −
 
import pandas as pd
a=pd.Series([False, True, True])
if pd.Series([False, True, False]).all():
   print ('I am True')
   
a=pd.Series(['hi'])
Use 
a.empty, 
a.bool

a.any()
a.all()




df Query
df = pd.DataFrame({'A': range(1, 6),
                   'B': range(10, 0, -2),
                   'C C': range(10, 5, -1)})
df.query('A > B')

df.query('A < B')

or
df[df.A > df.B]


For columns with spaces in their name, you can use backtick quoting.

df.query('B == `C C`')
or

df[df.B == df['C C']]



