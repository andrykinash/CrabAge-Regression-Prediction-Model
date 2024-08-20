import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

file_path = 'C:/Users/andry/Downloads/CrabAgePrediction.csv'
crab_data = pd.read_csv(file_path)

print("Initial data:")
print(crab_data.head())

crab_data['Sex'] = crab_data['Sex'].map({'F': 0, 'M': 1, 'I': 2})

print("Data after label encoding:")
print(crab_data.head())

#Calculate the mean and standard deviation then normalize
mean_values = crab_data.mean()
std_values = crab_data.std()
crab_data_normalized = (crab_data - mean_values) / std_values

X = crab_data_normalized.drop(['Age'], axis=1)
y = crab_data_normalized['Age']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

# ----------------------------------------------------------------------------------------------------------------------
#                                       SGD, BGD, Normal Equation, MSE functions
# ----------------------------------------------------------------------------------------------------------------------

def cost_func(y_true, y_pred):
    return np.mean((y_true - y_pred)**2) / 2

#used for LWR
def calculate_weight(X, x_query, tau=1.0):  
    return np.exp(-np.sum((X - x_query)**2, axis=1) / (2 * tau ** 2))

def SGD(X, y, alpha=0.01, epochs=100, random_state=None): 
    if random_state is not None:
        np.random.seed(random_state)

    X = np.c_[np.ones(X.shape[0]), X]
    theta = np.random.randn(X.shape[1]) 
    J_theta = [] #aka cost history
    
    for epoch in range(epochs):
        indices = np.random.permutation(len(y))  # shuffle the rows
        X = X[indices]
        y = y[indices]
        
        for i in range(len(y)):
            prediction = np.dot(X[i], theta)  # Linear Prediction
            error = prediction - y[i]  # Error Calculation
            gradient = X[i] * error
            theta -= gradient * alpha  
        
        cost = cost_func(y, np.dot(X, theta)) 
        J_theta.append(cost) 
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Cost: {cost}")
            
    return theta, J_theta  

def BGD(X, y, alpha=0.01, epochs=100, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    X = np.c_[np.ones(X.shape[0]), X]
    theta = np.random.randn(X.shape[1])
    J_theta = []
    
    for epoch in range(epochs):
        predictions = np.dot(X, theta)
        errors = predictions - y
        gradient = np.dot(X.T, errors) / len(y)
        
        theta -= alpha * gradient
        cost = cost_func(y, np.dot(X, theta))
        J_theta.append(cost)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Cost: {cost}") #prints every 20 epochs

    plt.figure()  # Create a new figure for BGD
    plt.plot(J_theta)
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Cost History during BGD')       
    return theta, J_theta

#LWR SGD
def LWR_SGD(X, y, x_query, alpha=0.01, epochs=100, tau=1.0, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    X = np.c_[np.ones(X.shape[0]), X]
    theta = np.random.randn(X.shape[1])
    weights = calculate_weight(X, np.hstack([1, x_query]), tau)
    
    for epoch in range(epochs):
        indices = np.random.permutation(len(y))  # shuffle the rows
        X = X[indices]
        y = y[indices]
        weights = weights[indices]
        
        for i in range(len(y)):
            prediction = np.dot(X[i], theta)  # Linear Prediction
            error = prediction - y[i]  # Error Calculation
            gradient = X[i] * error * weights[i]
            theta -= alpha * gradient  
    return theta

#bonus LWR normal equation
def LWR_Normal_Equation(X, y, x_query, tau=1.0):
    X = np.c_[np.ones(X.shape[0]), X]
    weights = calculate_weight(X, np.hstack([1, x_query]), tau)
    W = np.diag(weights)
    
    theta = np.linalg.pinv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(y)
    return theta

# compute normal equation function
def normal_equation(X, y):
    X = np.c_[np.ones(X.shape[0]), X] 
    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)  # Compute theta using the Normal Equation aka (X^T*X)inverse X^T*y
    return theta

# compute the MSE function
def compute_mse(X, y, theta):
    X = np.c_[np.ones(X.shape[0]), X]
    y_pred = X.dot(theta)
    mse = np.mean((y - y_pred)**2)
    return mse
#----------------------------------------------------------------------------------------------------

# Convert to Numpy arrays
X_train_np = X_train.to_numpy().astype(np.float64)
y_train_np = y_train.to_numpy().astype(np.float64)
X_test_np = X_test.to_numpy().astype(np.float64)
y_test_np = y_test.to_numpy().astype(np.float64)

# Run SGD
theta_sgd, J_theta_sgd = SGD(X_train_np, y_train_np, alpha=0.0001, epochs=100, random_state=2)

plt.figure()  # Create a new figure for SGD
print("Final theta from SGD:", theta_sgd)
plt.plot(J_theta_sgd)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost History during SGD')

# Run BGD 
theta_bgd, J_theta_bgd = BGD(X_train_np, y_train_np, alpha=0.0001, epochs=100, random_state=5)
print("Final theta from BGD:", theta_bgd)

# Run Normal Equation
theta_ne = normal_equation(X_train_np, y_train_np)
print("Final theta from Normal Equation:", theta_ne)

# LWR
x_query = X_test_np[0] 
theta_lwr_sgd = LWR_SGD(X_train_np, y_train_np, x_query, alpha=0.0001, epochs=100, tau=0.5, random_state=2)
print("Final theta from LWR_SGD:", theta_lwr_sgd)
mse_lwr_sgd = compute_mse(X_test_np, y_test_np, theta_lwr_sgd)
print(f'MSE for LWR with SGD: {mse_lwr_sgd}')

# Compute MSE for SGD, BGD, and Normal Equation
mse_sgd = compute_mse(X_test_np, y_test_np, theta_sgd)
mse_bgd = compute_mse(X_test_np, y_test_np, theta_bgd)
mse_ne = compute_mse(X_test_np, y_test_np, theta_ne)
print(f'MSE for SGD: {mse_sgd}')
print(f'MSE for BGD: {mse_bgd}')
print(f'MSE for Normal Equation: {mse_ne}')
theta_lwr_ne = LWR_Normal_Equation(X_train_np, y_train_np, x_query, tau=0.5)
print("Final theta from LWR with Normal Equation:", theta_lwr_ne)
mse_lwr_ne = compute_mse(X_test_np, y_test_np, theta_lwr_ne)
print(f'MSE for LWR with Normal Equation: {mse_lwr_ne}')

# Plot the MSE values for comparison
labels = ['SGD', 'BGD', 'Normal Equation']
mse_values = [mse_sgd, mse_bgd, mse_ne]
labels.extend(['LWR SGD'])
mse_values.extend([mse_lwr_sgd])
labels.extend(['LWR NE'])
mse_values.extend([mse_lwr_ne])
plt.figure()
plt.bar(labels, mse_values, color=['red', 'green', 'blue', 'purple', 'orange']) 
plt.xlabel('Method')
plt.ylabel('MSE')
plt.title('MSE Comparison Among Methods')
plt.show()