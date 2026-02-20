numbers = []
print("Enter 5 numbers:")

for i in range(5):
    num = int(input(f"Number {i+1}: "))
    numbers.append(num)
even_numbers = [num for num in numbers if num % 2 == 0]

print(f"\nYour numbers: {numbers}")
print(f"Even numbers: {even_numbers}")
