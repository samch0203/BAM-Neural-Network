import numpy as np

# Bipolar activation function
def activation(x):
    return np.where(x >= 0, 1, -1)

# Number of vector pairs
n = 2

# Length of X and Y vectors
x_len = int(input("Enter length of X vectors: "))
y_len = int(input("Enter length of Y vectors: "))

X = []
Y = []

print("\nEnter X vectors (use space between values, only 1 or -1):")
for i in range(n):
    x = list(map(int, input(f"Enter X{i+1}: ").split()))
    X.append(x)

print("\nEnter Y vectors (use space between values, only 1 or -1):")
for i in range(n):
    y = list(map(int, input(f"Enter Y{i+1}: ").split()))
    Y.append(y)

X = np.array(X)
Y = np.array(Y)

# Step 1: Calculate weight matrix
W = np.zeros((x_len, y_len))

for i in range(n):
    W += np.outer(X[i], Y[i])

print("\nWeight Matrix W:")
print(W)

# Recall Y from X
x_test = np.array(list(map(int, input("\nEnter X vector to recall Y: ").split())))
y_result = activation(np.dot(x_test, W))

print("Recalled Y vector:", y_result)

# Recall X from Y
y_test = np.array(list(map(int, input("\nEnter Y vector to recall X: ").split())))
x_result = activation(np.dot(y_test, W.T))

print("Recalled X vector:", x_result)