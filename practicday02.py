import numpy as np
#find data type of an array
arr = np.array([1, 2, 3, 4, 5])
print("Data type of the array:", arr.dtype)
print("-------------------------------------------------")
#convert integer array to float
arr.astype(float)
print("Array after converting to float:", arr.astype(float))
print("-------------------------------------------------")
#intermidiate level
#findmsum off all elements in an array
arr = np.array([10,20,30])
sum_of_elements = arr.sum()
print("Sum of all elements in the array:", sum_of_elements)
print("-------------------------------------------------")
#find mean of an array
mean_value = np.mean(arr)
print("Mean of the array:", mean_value)
print("-------------------------------------------------")
#find maximum and minimum value in an array
np.max(arr)
print("Maximum value in the array:", np.max(arr))
np.min(arr)
print("Minimum value in the array:", np.min(arr))
print("-------------------------------------------------")
#find index of maximum and minimum value in an array
index_of_max = np.argmax(arr)
print("Index of maximum value in the array:", index_of_max)
index_of_min = np.argmin(arr)
print("Index of minimum value in the array:", index_of_min)
print("-------------------------------------------------")
#reverse a numpy
arr[::-1]
print("-------------------------------------------------")
#extract elements > 25
arr = np.array([10, 30, 40,20])
greater_than_25 = arr[arr > 25]
print("Elements greater than 25:", greater_than_25)
print("-------------------------------------------------")
#replace elements >30 with 30
arr[arr > 30] = 30
print("Array after replacing elements greater than 30 with 30:", arr)
print("-------------------------------------------------")
#reshape array size 12 to 3*4
arr = np.arange(12)
reshaped_arr = arr.reshape(3, 4)
print("Reshaped array (3*4):")
print(reshaped_arr)
print("-------------------------------------------------")
#flatten a 2D array to 1D
flattened_arr = np.array([[1, 2], [3, 4]])
flattened_arr = reshaped_arr.flatten()
print("Flattened array (1D):")
print(flattened_arr) 
print("-------------------------------------------------")
#sort an array
sort_arr = np.sort(arr)
print("Sorted array:")
print(sort_arr)
print("-------------------------------------------------")
#concatenate two arrays
arr1 = np.concatenate((np.array([1, 2]), np.array([4,5])))
print("Concatenated array:")
print(arr1)
print("-------------------------------------------------")
#count non-zero elements in an array
arr = np.array([0, 1, 2, 3, 0, 4])
non_zero_count = np.count_nonzero(arr)
print("Number of non-zero elements in the array:", non_zero_count)
print("-------------------------------------------------")
#create 2D array (3*3) with random integers between 1 and 10
random_array = np.random.randint(1, 11, size=(3, 3))
print("2D array with random integers between 1 and 10:")
print(random_array)
print("-------------------------------------------------")
#find row-wise and column-wise sum of a 2D array
row_sum = random_array.sum(axis=1)
column_sum = random_array.sum(axis=0)
print("Row-wise sum of the 2D array:", row_sum)
print("Column-wise sum of the 2D array:", column_sum)
print("-------------------------------------------------")
#find transpose of a matrix
arr.T
print("Transpose of the array:")
print(arr.T)
print("-------------------------------------------------")
#multiply all elements by 5
arr_multiplied = arr * 5
print("Array after multiplying all elements by 5:")
print(arr_multiplied)
print("-------------------------------------------------")
#square root of all elements in an array
sqrt_arr = np.sqrt(arr)
print("Square root of all elements in the array:")
print(sqrt_arr)
print("-------------------------------------------------")
#find unique elements in an array
unique_elements = np.unique([1, 2, 2, 3, 4, 4, 5])
print("Unique elements in the array:", unique_elements)
print("-------------------------------------------------")
#check if any element in an array is greater than 50
arr = np.array([10, 20, 30, 40, 50])
any_greater_than_50 = np.any(arr > 50)
print("Is any element in the array greater than 50?", any_greater_than_50)
print("-------------------------------------------------")
#advanced level
#check if all values > 0 
arr = np.array([1, 2, 3, 4, 5])
all_greater_than_0 = np.all(arr > 0)
print("Are all values in the array greater than 0?", all_greater_than_0)
print("-------------------------------------------------")
#matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
product = np.dot(A,B)
print("Matrix multiplication of A and B:")
print(product)
print("-------------------------------------------------")
#element-wise multiplication
elementwise_product = A * B
print("Element-wise multiplication of A and B:")
print(elementwise_product)
print("-------------------------------------------------")
#replace NaN/ not a  number  with 0
arr = np.array([1,np.nan, 2])
np.nan_to_num(arr)
print("Array after replacing NaN with 0:")
print(np.nan_to_num(arr))
print("-------------------------------------------------")
#find missing values

