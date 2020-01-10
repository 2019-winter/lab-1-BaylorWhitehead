---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Name(s)
**Baylor Whitehead**


**Instructions:** This is an individual assignment, but you may discuss your code with your neighbors.


# Python and NumPy

While other IDEs exist for Python development and for data science related activities, one of the most popular environments is Jupyter Notebooks.

This lab is not intended to teach you everything you will use in this course. Instead, it is designed to give you exposure to some critical components from NumPy that we will rely upon routinely.

## Exercise 0
Please read and reference the following as your progress through this course. 

* [What is the Jupyter Notebook?](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.ipynb#)
* [Notebook Tutorial](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)
* [Notebook Basics](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb)

**In the space provided below, what are three things that still remain unclear or need further explanation?**


**YOUR ANSWER HERE**


## Exercises 1-7
For the following exercises please read the Python appendix in the Marsland textbook and answer problems A.1-A.7 in the space provided below.


## Exercise 1

```python
import numpy as np

array = np.full((6,4), 2)
print(array)
```

## Exercise 2

```python
import numpy as np

array = np.full((6,4), 1)
np.fill_diagonal(array, 3)
print(array)
```

## Exercise 3

```python
import numpy as np

array1 = np.full((6,4), 2)
array2 = np.full((6,4), 1)
np.fill_diagonal(array2, 3)

print(np.multiply(array1, array2))
array1*array2 #also works

try:
    np.dot(array1, array2)
except:
    print("Dot does not work")
```

Multiplying the matrices (a * b) works because they are the same size. So each value is multiplied with the matching one in the other matrix.

Matrix dot product on these matrices (dot(a,b)) does not work because the n of a must be the same length as the m of b or else the result is undefined. This is because the rows of a need columns of b to be paired with.


## Exercise 4

```python
import numpy as np

array1 = np.full((6,4), 2)

array2 = np.full((6,4), 1)
np.fill_diagonal(array2, 3)

print(np.dot(array1.transpose(), array2))
print(np.dot(array1, array2.transpose()))
```

The shape of dotted matrices (dot(a,b)) is the m of the a and the n of b. The resultant size is the sides that do not have to match for a matrix dot product to be computed.

So for np.dot(array1.transpose(), array2) the columns of array1 now match the rows of array2. Then the resultant matrix has the same number of rows as array1 and the same number of columns as array2.

For np.dot(array1.transpose(), array2) there are 6 columns for array1 and 6 rows for array2 so the resulting matrix is the other sides of array1 and array2 so a 4x4.

For np.dot(array1.transpose(), array2) there are 4 columns for array1 and 4 rows for array2 so the resulting matrix is the other sides of array1 and array2 so a 6x6.


## Exercise 5

```python
def printFunc(a):
    print(a)
    
printFunc("Hello World!")
```

## Exercise 6

```python
import numpy as np

array = np.random.randint(0,30,5)
sum = np.sum(array)
mean = np.mean(array)
print(sum)
print(mean)
```

## Exercise 7

```python
import numpy as np

array = np.full((6,4), 2)
np.fill_diagonal(array, 1)

def numOnes(inputArray):
    count = 0
    for row in inputArray:
        for num in row:
            if(num == 1):
                count+=1
    return count, np.where(array == 1)[0].size

c1, c2 = numOnes(array)
print(c1)
print(c2)
```

## Excercises 8-???
While the Marsland book avoids using another popular package called Pandas, we will use it at times throughout this course. Please read and study [10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) before proceeding to any of the exercises below.


## Exercise 8
Repeat exercise A.1 from Marsland, but create a Pandas DataFrame instead of a NumPy array.

```python
import pandas as pd

frame = pd.DataFrame(2, index=range(6), columns=range(4))
print(frame)
```

## Exercise 9
Repeat exercise A.2 using a DataFrame instead.

```python
import pandas as pd
import numpy as np

frame = pd.DataFrame(1, index=range(6), columns=range(4))
np.fill_diagonal(frame.values, 3)
print(frame)
```

## Exercise 10
Repeat exercise A.3 using DataFrames instead.

```python
import pandas as pd
import numpy as np

frame1 = pd.DataFrame(2, index=range(6), columns=range(4))
frame2 = pd.DataFrame(1, index=range(6), columns=range(4))
np.fill_diagonal(frame2.values, 3)

print(pd.DataFrame.multiply(frame1,frame2))

try:
    print(pd.DataFrame.dot(frame1,frame2))
except:
    print("Dotting does not work")
```

Multiplying the matrices (a * b) works because they are the same size. So each value is multiplied with the matching one in the other matrix.

Matrix dot product on these matrices (dot(a,b)) does not work because the n of a must be the same length as the m of b or else the result is undefined. This is because the rows of a need columns of b to be paired with.


## Exercise 11
Repeat exercise A.7 using a dataframe.

```python
import pandas as pd

frame = pd.DataFrame(1, index=range(6), columns=range(4))
np.fill_diagonal(frame.values, 3)


count = 0
def countFrameOnes(df):
    count = 0
    for row in frame.values:
        for num in row:
            if (num == 1):
                count+=1
    return count, np.where(frame==1)[0].size

c1, c2 = countFrameOnes(array)
print(c1)
print(c2)     

```

## Exercises 12-14
Now let's look at a real dataset, and talk about ``.loc``. For this exercise, we will use the popular Titanic dataset from Kaggle. Here is some sample code to read it into a dataframe.

```python
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv"
)
titanic_df
```

Notice how we have nice headers and mixed datatypes? That is one of the reasons we might use Pandas. Please refresh your memory by looking at the 10 minutes to Pandas again, but then answer the following.


## Exercise 12
How do you select the ``name`` column without using .iloc?

```python
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv"
)
titanic_df["name"]
```

## Exercise 13
After setting the index to ``sex``, how do you select all passengers that are ``female``? And how many female passengers are there?

```python
## YOUR SOLUTION HERE
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv"
)
titanic_df.set_index('sex',inplace=True)
females_df = titanic_df.loc["female"]
print(len(females_df))
```

466 female passengers


## Exercise 14
How do you reset the index?

```python
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv"
)
titanic_df.set_index('sex',inplace=True)
titanic_df.reset_index(inplace=True)
titanic_df
```

using reset_index()
