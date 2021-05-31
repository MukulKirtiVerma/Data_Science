# -*- coding: utf-8 -*-
"""
Created on Mon May 31 18:37:34 2021

@author: Mukul Kirti Verma
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 13:14:34 2018
@author: Mukul Kirti Verma
"""
#Numpy lib is beautiful library for mathmatical  operations .

# to install numpy
!pip install numpy

import numpy
x=numpy.array([1,2,3])




import numpy as np 
#convert list to array=========

a = np.array([1, 2, 3, 4]) 

#create multi dim array===========
a = np.array([[1, 2], [3,4],[5,6]])

#define data type or array=========
x = [1,2,3] 
a = np.asarray(x, dtype = 'int64') 
print( a )
a = np.asarray(x, dtype = str) 

#tuple to array
x = (1,2,3) 
a = np.asarray(x) 




#check the shape of array==========
a = np.array([[1,2,3],[4,5,6]],) 
print (a.shape) 

a=a.reshape(1,6)
#change the shape of array=========
a.shape=(1,5)

print( a )




#create sequential array===============
#np.arange(start,end,linsoace)
#default start=0 and linspac =1
a = np.arange(5) 
a=np.arange(1,6)

print( a )



a = np.arange(-8,1,-2) 
print( a )

#reshape array========================
a = np.arange(24) 
b=a.reshape(2,2,,2) 
print (b)

#itemsize gives the total byte for each element in array
x = np.array([1,2,3,4,5], dtype = np.int64) 
print (x.itemsize)

#create a array of zeros=======================
x = np.zeros(6) 
print (x)

#create array of ones=========================
x = np.ones((2,5)) 
print (x)

#$Random numbers in ndarrays
#Another very commonly used method to create ndarrays is np.random.rand() method. It creates an array of a given shape with random values from [0,1):

# random 
np.random.rand(2,3)



#An array of your choice
#Or, in fact, you can create an array filled with any given value using the np.full() method. Just pass in the shape of the desired array and the value you want:

np.full((2,2),7)


#Imatrix in NumPy

#Note: A square matrix has an N x N shape. This means it has the same number of rows and columns.

# identity matrix
np.eye(3)


#However, NumPy gives you the flexibility to change the diagonal along which 
#the values have to be 1s. You can either move it above the main diagonal:

# not an identity matrix
np.eye(3,k=1)
array([[0., 1., 0.],
       [0., 0., 1.],
       [0., 0., 0.]])

#Or move it below the main diagonal:

np.eye(3,k=-1)
array([[0., 0., 0.],
       [0., 0., 0.],
       [1., 0., 0.]])
#Note: A matrix is called the Identity matrix only when the 1s are along 
#the main diagonal and not any other diagonal!





#create array of equal distribution===========
#np.linspace(start,end,linspace) 
np.linspace(5)
x = np.linspace(10,20,5) 
print(x)


#endpoint = False then end will not include in array========
x = np.linspace(10,20, 5, endpoint = False)
print( x )

#retstep = True show the diffrence b/w array=================
x = np.linspace(1,10,4, retstep = True) 
print (x)



#log of element of array with equal space===================
a = np.logspace(1,10,num = 10) 
print (a)


#access element of array 1 dimention
a = np.arange(10) 
b = a[4] 
print (b)


#acces element in 2d========================
#syntax arr[start row : end row : increment by , start col : end col : increment by]
a = np.arange(10) 
a.shape=(5,2)
print (a)

#print 1st col==========================
b = a[:,1:2] 
print (b)


#print 1st row==========================
b = a[1:2,:] 
print (b)

#from 1st row print all====================
b = a[1:,:] 
print (b)


#from 1st col print all=====================
b = a[:,1:] 
print (b)

#increment slice by 2======================
a = np.arange(10) 
b = a[2:7:2]
print(b)

#row only slice=============================
a = np.array([[1,2,3],[3,4,5],[4,5,6]]) 
a[0:2]
print( a[1:])
#a[row,col]

# slice single row or col=====================
#slice single col=============================
a = np.array([[1,2,3],[3,4,5],[4,5,6]]) 
print( a[...,2])


#slice row only==============================
a = np.array([[1,2,3,1],[3,4,5,1],[4,5,6,1]]) 

print( a[2,...])

#slice both=====================================
a = np.array([[1,2,3,1],[3,4,5,1],[4,5,6,1]]) 

print( a[1:3,2:4])
 
a = np.arange(10) 
b = a[2:7:3] 


a = np.arange(6) 
print (a[:5:1])




#Expanding and Squeezing a NumPy array
#Expanding a NumPy array
#You can add a new axis to an array using the expand_dims() 
#method by providing the array and the axis along which to expand:

# expand dimensions
a = np.array([1,2,3])
b = np.expand_dims(a,axis=0)
c = np.expand_dims(a,axis=1)
print('Original:','\n','Shape',a.shape,'\n',a)
print('Expand along columns:','\n','Shape',b.shape,'\n',b)
print('Expand along rows:','\n','Shape',c.shape,'\n',c)


#Squeezing a NumPy array
#On the other hand, if you instead want to reduce the axis of the array,
# use the squeeze() method. It removes the axis that has a single entry. This means if you have created a 2 x 2 x 1 matrix, squeeze() will remove the third dimension from the matrix:

# squeeze
a = np.array([[[1,2,3],
[4,5,6]]])
b = np.squeeze(a, axis=0)
print('Original','\n','Shape',a.shape,'\n',a)
print('Squeeze array:','\n','Shape',b.shape,'\n',b)


#However, if you already had a 2 x 2 matrix, using squeeze() in that case
# would give you an error:

# squeeze
a = np.array([[1,2,3],
[4,5,6]])
b = np.squeeze(a, axis=0)
print('Original','\n','Shape',a.shape,'\n',a)
print('Squeeze array:','\n','Shape',b.shape,'\n',b)


#create run time array by user define input===========
r=int(input())#enter row's
c=int(input())#enter cols
a=np.zeros(r*c)#create array or zeros=================
for i in range(r*c):
    a[i]=int(input("enter no."))#input array element=====
a.shape=(r,c)


# or create list of user input and convert to array===========

a = list(np.array([[1,2,3],[3,4,5],[4,5,6]]))
print (a[0])
a[0]=[7,8,9]
print (a[1:2])
print (a[1,1])



l=[]
x=int(input())
y=int(input())
for i in range(0,x):
    l1=[]
    for j in range(0,y):
        l1.append(int(input()))
    l.append(l1)
 
#transpose=========================================
    
import numpy as np
    
i=int(input())
j=int(input())
a=np.zeros(i*j)
a.shape=(i,j)
for ii in range (i):
    for jj in range(j):
        a[ii][jj]=int(input())



a.shape=(1,i*j);
k=0;
b=np.zeros(i*j)
b.shape=(j,i)
for ii in range(i):
    for jj in range(j):
        b[jj][ii]=a[0][k]
        k=k+1
        



#Trans==============================================        
i=int(input())
j=int(input())
a=np.zeros(i*j)
a.shape=(i,j)        
        
#eg     
l2=[]
b=np.zeros(i*j)
b.shape=(j,i)
for k in range(0,i):
    for kk in range(0,j):
        a[k][kk]=int(input())
        
 #eg       
for k in range(0,i):
    for kk in range(0,j):
        if(k==kk):
            l2.append(a[k][kk])
        if(k+kk==i-1):
            if(k!=kk):
                l2.append(a[k][kk])
       
"""        
Some numpy operations        
1. Elementwise operations
2. Basic operations
3. With scalars:
"""
import numpy as np
a = np.array([1, 2, 3, 4])
b=a + 1


#All arithmetic operates elementwise====================

b = np.ones(4) + 1
print(b)
#====================================
c=a - b
print(c)
#====================================
d=a * b
print(d)
#======================================
j = np.arange(5)
2**(j + 1) - j

#These operations are of course much faster than if you did them in pure python:


a = np.arange(10000)
a + 1  
#or
l = range(10000)
[i+1 for i in l] 


# Array multiplication is not matrix multiplication:========
c = np.ones((3, 3))
d=c * c    
print(d)               # NOT matrix multiplication!


# Matrix multiplication:
d=c.dot(c)
print(d)

"""
#Elementwise operations
Try simple arithmetic elementwise operations: add even elements with odd elements
Time them against their pure python counterparts using %timeit.
Generate:
"""
    
a = np.array([1, 2, 3, 4])
b = np.array([4, 2, 2, 4])
a == b

print(a > b)
"""
Array-wise comparisons: >>>>>> a = np.array([1, 2,...
Logical operations:
"""

a = np.array([1, 1, 0, 0], dtype=bool)
b = np.array([1, 0, 1, 0], dtype=bool)
print(np.logical_or(a, b))
#====================================

print(np.logical_and(a, b))
#====================================

print(np.logical_xor(a, b))




#Transcendental functions:=======================


a = np.arange(5)
#sin===============================================
print(np.sin(a))
#log===============================================
print(np.log(a))
#e pow x ===========================================
print(np.exp(a))


#Shape mismatches===================================

a = np.arange(4)
a + np.array([1, 2])  


#Broadcasting? ===============================


#create triangular matrix============================
a = np.triu(np.ones((3, 3)), 1)   # see help(np.triu)
print(a)
#transpose==========================================
print(a.T)


"""
The transposition is a view
As a results, the following code is wrong and will not make a matrix symmetric:
"""

a += a.T
print(a)



#sum func==========================================
x = np.array([1, 2, 3, 4])
print(x.sum())


x = np.array([[1, 1], [2, 2]])
print(x.sum())


print(x.sum(axis=0) )  # columns sum (first dimension)
#col 0 and col 1  indivisual sum===============================
print(x[:, 0].sum(), x[:, 1].sum())

#row wise sum==================================================
print(x.sum(axis=1))  # rows (second dimension)

print(x[0, :].sum(), x[1, :].sum())




#some oter func
x = np.array([1, 3, 2])
#min of array======================
print(x.min())
#max of array=====================
print(x.max())


print(x.argmin())  # index of minimum

print(x.argmax())  # index of maximum

#===============================================
#Logical operations:

#check all are True or not
print(np.all([True, True, False]))
#check any is True or not
print(np.any([True, True, False]))

#Note Can be used for array comparisons:
#!= operator============================
a = np.zeros((100, 100))
print(np.any(a != 0))
#== operator
print(np.all(a == a))

#<= and >= operator
a = np.array([1, 2, 3, 2])
b = np.array([2, 2, 3, 2])
c = np.array([6, 4, 4, 5])
print(np.all(((a <= b) & (b <= c))))

 

# mean median,mode ,std
x = np.array([1, 2, 3, 1])
y = np.array([[1, 2, 3], [5, 6, 1]])
print(x.mean())
print(np.median(x))
print(x.std())         # full population standard dev.


#eg
k=[]
for i in range(1,6):
    l=[]
    for j in range(1,11):
        l.append(i*j)
    k.append(l)
import numpy as np
x=np.asarray(k)
h=int(input())
x=x.T
print(x[...,h])



Indexing and Slicing of NumPy array
So far, we have seen how to create a NumPy array and how to play around with its shape. In this section, we will see how to extract specific values from the array using indexing and slicing.

 

Slicing 1-D NumPy arrays
Slicing means retrieving elements from one index to another index. All we have to do is to pass the starting and ending point in the index like this: [start: end].

However, you can even take it up a notch by passing the step-size. What is that? Well, suppose you wanted to print every other element from the array, you would define your step-size as 2, meaning get the element 2 places away from the present index.

Incorporating all this into a single index would look something like this: [start:end:step-size].

a = np.array([1,2,3,4,5,6])
print(a[1:5:2])
[2 4]
Notice that the last element did not get considered. This is because slicing includes the start index but excludes the end index.

A way around this is to write the next higher index to the final index value you want to retrieve:

a = np.array([1,2,3,4,5,6])
print(a[1:6:2])
[2 4 6]
If you don’t specify the start or end index, it is taken as 0 or array size, respectively, as default. And the step-size by default is 1.

a = np.array([1,2,3,4,5,6])
print(a[:6:2])
print(a[1::2])
print(a[1:6:])
[1 3 5]
[2 4 6]
[2 3 4 5 6]
 

Slicing 2-D NumPy arrays
Now, a 2-D array has rows and columns so it can get a little tricky to slice 2-D arrays. But once you understand it, you can slice any dimension array!

Before learning how to slice a 2-D array, let’s have a look at how to retrieve an element from a 2-D array:

a = np.array([[1,2,3],
[4,5,6]])
print(a[0,0])
print(a[1,2])
print(a[1,0])
1
6
4
Here, we provided the row value and column value to identify the element we wanted to extract. While in a 1-D array, we were only providing the column value since there was only 1 row.

So, to slice a 2-D array, you need to mention the slices for both, the row and the column:

a = np.array([[1,2,3],[4,5,6]])
# print first row values
print('First row values :','\n',a[0:1,:])
# with step-size for columns
print('Alternate values from first row:','\n',a[0:1,::2])
# 
print('Second column values :','\n',a[:,1::2])
print('Arbitrary values :','\n',a[0:1,1:3])
First row values : 
 [[1 2 3]]
Alternate values from first row: 
 [[1 3]]
Second column values : 
 [[2]
 [5]]
Arbitrary values : 
 [[2 3]]
 

Slicing 3-D NumPy arrays
So far we haven’t seen a 3-D array. Let’s first visualize how a 3-D array looks like:

NumPy 3-D array

a = np.array([[[1,2],[3,4],[5,6]],# first axis array
[[7,8],[9,10],[11,12]],# second axis array
[[13,14],[15,16],[17,18]]])# third axis array
# 3-D array
print(a)
[[[ 1  2]
  [ 3  4]
  [ 5  6]]

 [[ 7  8]
  [ 9 10]
  [11 12]]

 [[13 14]
  [15 16]
  [17 18]]]
In addition to the rows and columns, as in a 2-D array, a 3-D array also has a depth axis where it stacks one 2-D array behind the other. So, when you are slicing a 3-D array, you also need to mention which 2-D array you are slicing. This usually comes as the first value in the index:

# value
print('First array, first row, first column value :','\n',a[0,0,0])
print('First array last column :','\n',a[0,:,1])
print('First two rows for second and third arrays :','\n',a[1:,0:2,0:2])
First array, first row, first column value : 
 1
First array last column : 
 [2 4 6]
First two rows for second and third arrays : 
 [[[ 7  8]
  [ 9 10]]

 [[13 14]
  [15 16]]]
If in case you wanted the values as a single dimension array, you can always use the flatten() method to do the job!

print('Printing as a single array :','\n',a[1:,0:2,0:2].flatten())
Printing as a single array : 
 [ 7  8  9 10 13 14 15 16]
 

Negative slicing of NumPy arrays
An interesting way to slice your array is to use negative slicing. Negative slicing prints elements from the end rather than the beginning. Have a look below:

a = np.array([[1,2,3,4,5],
[6,7,8,9,10]])
print(a[:,-1])
[ 5 10]
Here, the last values for each row were printed. If, however, we wanted to extract from the end, we would have to explicitly provide a negative step-size otherwise the result would be an empty list.

print(a[:,-1:-3:-1])
[[ 5  4]
 [10  9]]
Having said that, the basic logic of slicing remains the same, i.e. the end index is never included in the output.

An interesting use of negative slicing is to reverse the original array.

a = np.array([[1,2,3,4,5],
[6,7,8,9,10]])
print('Original array :','\n',a)
print('Reversed array :','\n',a[::-1,::-1])
Original array : 
 [[ 1  2  3  4  5]
 [ 6  7  8  9 10]]
Reversed array : 
 [[10  9  8  7  6]
 [ 5  4  3  2  1]]
You can also use the flip() method to reverse an ndarray.

a = np.array([[1,2,3,4,5],
[6,7,8,9,10]])
print('Original array :','\n',a)
print('Reversed array vertically :','\n',np.flip(a,axis=1))
print('Reversed array horizontally :','\n',np.flip(a,axis=0))
Original array : 
 [[ 1  2  3  4  5]
 [ 6  7  8  9 10]]
Reversed array vertically : 
 [[ 5  4  3  2  1]
 [10  9  8  7  6]]
Reversed array horizontally : 
 [[ 6  7  8  9 10]
 [ 1  2  3  4  5]]
Stacking and Concatenating NumPy arrays
Stacking ndarrays
You can create a new array by combining existing arrays. This you can do in two ways:

Either combine the arrays vertically (i.e. along the rows) using the vstack() method, thereby increasing the number of rows in the resulting array
Or combine the arrays in a horizontal fashion (i.e. along the columns) using the hstack(), thereby increasing the number of columns in the resultant array
Numpy stacking

a = np.arange(0,5)
b = np.arange(5,10)
print('Array 1 :','\n',a)
print('Array 2 :','\n',b)
print('Vertical stacking :','\n',np.vstack((a,b)))
print('Horizontal stacking :','\n',np.hstack((a,b)))
Array 1 : 
 [0 1 2 3 4]
Array 2 : 
 [5 6 7 8 9]
Vertical stacking : 
 [[0 1 2 3 4]
 [5 6 7 8 9]]
Horizontal stacking : 
 [0 1 2 3 4 5 6 7 8 9]
A point to note here is that the axis along which you are combining the array should have the same size otherwise you are bound to get an error!

a = np.arange(0,5)
b = np.arange(5,9)
print('Array 1 :','\n',a)
print('Array 2 :','\n',b)
print('Vertical stacking :','\n',np.vstack((a,b)))
print('Horizontal stacking :','\n',np.hstack((a,b)))
NumPy stacking error

Another interesting way to combine arrays is using the dstack() method. It combines array elements index by index and stacks them along the depth axis:

a = [[1,2],[3,4]]
b = [[5,6],[7,8]]
c = np.dstack((a,b))
print('Array 1 :','\n',a)
print('Array 2 :','\n',b)
print('Dstack :','\n',c)
print(c.shape)
Array 1 : 
 [[1, 2], [3, 4]]
Array 2 : 
 [[5, 6], [7, 8]]
Dstack : 
 [[[1 5]
  [2 6]]

 [[3 7]
  [4 8]]]
(2, 2, 2)


#sorting
a = np.array([1,4,2,5,3,6,8,7,9])
np.sort(a, kind='quicksort')


#for image
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

# read image
im = misc.imread('./original.jpg')
# image
im



print(im.shape)
print(type(type))


plt.imshow(np.flip(im, axis=1))


Or you could normalize or change the range of values of the pixels. This is sometimes useful for faster computations.

im/255