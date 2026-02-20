import numpy as np

# 1. Create a numpy array from 1 to 20
arr = np.arange(1, 21)

print("NumPy array from 1 to 20:")
print(arr)

# 2. Find even numbers
even_numbers = arr[arr % 2 == 0]
print("\nEven numbers:")
print(even_numbers)

# 3. Convert 1D array to 4*5 matrix
matrix = arr.reshape(4, 5)
print("\n4*5 Matrix:")
print(matrix)

# 4. Find max & min value
max_value = arr.max()
min_value = arr.min()
print(f"\nMax value: {max_value}")
print(f"Min value: {min_value}")

# 5. Add 5 to all elements
arr_plus_5 = arr + 5
print("\nArray after adding 5 to all elements:")
print(arr_plus_5)
