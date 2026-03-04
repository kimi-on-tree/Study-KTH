
import numpy as np
import matplotlib.pyplot as plt

# basic functions
def generate_dataset(n, mA, mB, sigmaA, sigmaB):
    classA = np.zeros((2,n))
    classB = np.zeros((2,n))
    
    classA = np.random.randn(2,n)*sigmaA + np.array(mA).reshape(2,1)
    classB = np.random.randn(2,n)*sigmaB + np.array(mB).reshape(2,1)
    
    data = np.hstack([classA, classB])
    label = np.hstack([-np.ones(n), np.ones(n)])
    
    perm = np.random.permutation(2*n)
    
    return data[:,perm], label[perm], classA, classB

def plot_dataset(cA, cB, filename):
    plt.figure()
    plt.scatter(cA[0,:], cA[1,:], c='r')
    plt.scatter(cB[0,:], cB[1,:], c='b')
    plt.title("Generated Dataset")
    plt.savefig(filename)
    plt.close()  

def bias(X):
    return np.vstack([X, np.ones((1, X.shape[1]))])

def plot_boundary(W, X, T, title, filename):
    plt.figure(figsize=(6,5))
    plt.scatter(X[0,T==-1],X[1,T==-1],c='r',label='Class A')
    plt.scatter(X[0,T==1],X[1,T==1],c='b',label='Class B')
    
    x_range = np.array([np.min(X[0,:])-1, np.max(X[0,:])+1])
    
    w = W.flatten()
    if len(w) == 3:
        y_range = -(w[0]*x_range + w[2])/w[1]
    else:
        y_range = -(w[0]*x_range)/w[1]
        
    plt.plot(x_range, y_range, 'g--', label='Decision Boundary')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    
def plot_two_boundaries(W1, W2, X, T, title, filename):
    plt.figure(figsize=(6,5))
    plt.scatter(X[0,T==-1], X[1,T==-1], c='r', label='Class A')
    plt.scatter(X[0,T==1],  X[1,T==1],  c='b', label='Class B')

    x_range = np.array([np.min(X[0,:])-1, np.max(X[0,:])+1])

    for W, name in [(W1, "Perceptron"), (W2, "Delta")]:
        w = W.flatten()
        y_range = -(w[0]*x_range + w[2]) / w[1]
        plt.plot(x_range, y_range, '--', label=name)

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=200)
    plt.close()

    
# Task1: Perceptron vs Delta (Online)
def perceptron_vs_delta():
    X_origin, T, classA, classB = generate_dataset(n=100, mA=[1.0,2.0], mB=[-2.5,-1.0], sigmaA=0.5, sigmaB=0.5)
    
    plot_dataset(classA, classB, "task1_dataset")
    
    X = bias(X_origin)
    lr_per = 0.0005
    lr_del = 0.0005
    epochs = 20
    
    W = np.random.randn(1,3)*0.01
    
    # Classical perceptron
    W_p = W.copy()
    for k in range(epochs):
        for i in range(X.shape[1]):
            x_i = X[:,i:i+1]
            t_i = T[i]
            
            a_i = np.dot(W_p,x_i)
            y_i = 1 if a_i >= 0 else -1
            
            if y_i != t_i:
                W_p += lr_per*t_i*x_i.T
    
    # Delta Rule(Online)
    W_d = W.copy()
    for k in range(epochs):
        for i in range(X.shape[1]):
            x_i = X[:, i:i+1]
            t_i = T[i]
            
            error = t_i - np.dot(W_d, x_i)
            W_d += lr_del*error*x_i.T
            
    plot_two_boundaries(W_p, W_d, X, T, "Perceptron vs Delta", f"task1_compare_lr={lr_del}.png")


# Task2: Online vs Batch
def online_vs_batch():
    X_origin, T, classA, classB = generate_dataset(n=100, mA=[2.0,2.0], mB=[-0.5,-1.0], sigmaA=0.5, sigmaB=0.5)
    
    plot_dataset(classA, classB, "task2_dataset")
    
    X = bias(X_origin)
    N = X.shape[1]
    lr_batch = 0.1
    lr_online = lr_batch/N
    epochs = 100 
    
    initial = 35
    np.random.seed(initial)
    W = np.random.randn(1, 3)*0.1
    
    #Batch Mode
    W_batch = W.copy()
    mse_batch = []
    T_r = T.reshape(1, -1)
    for k in range(epochs):
        output = np.dot(W_batch, X)
        error = T_r - output
        mse_batch.append(np.mean(error ** 2))
        W_batch += lr_batch*np.dot(error, X.T)/N     
        
    #Online Mode
    W_online = W.copy()
    mse_online = []
    for k in range(epochs):
        output = np.dot(W_online, X)
        error_sum = T_r - output
        mse = np.mean(error_sum ** 2)
        mse_online.append(mse)
        for i in range(X.shape[1]):
            x_i = X[:,i:i+1]
            t_i = T[i]
            error_i = t_i - np.dot(W_online, x_i)
            W_online += lr_online*error_i*x_i.T
            
    plot_two_boundaries(W_online, W_batch, X, T, "Online vs Batch", f"task2_compare_lr={lr_batch}.png")
            
    
    plt.figure(figsize=(10, 6))
    plt.plot(mse_batch, label='Batch Learning', linewidth=2)
    plt.plot(mse_online, label='Online Learning', linestyle='--', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(f"Delta Rule: Batch vs Online (initial seed={initial})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"task2_mse_initialseed={initial}.png", dpi=200)
    plt.close()


# Task3: Remove bias and perform on different datasets
def remove_bias():
    Xs, Ts, cAs, cBs = generate_dataset(n=100, mA=[1.5,1.5], mB=[-1.5,-1.5], sigmaA=0.5, sigmaB=0.5)
    plot_dataset(cAs, cBs, "task3_dataset_symmetrical")
    
    Xa, Ta, cAa, cBa = generate_dataset(n=100, mA=[3.0,5.0], mB=[1.0,1.0], sigmaA=0.5, sigmaB=0.5)
    plot_dataset(cAa, cBa, "task3_dataset_same_quadrant ")
    
    Xn, Tn, cAn, cBn = generate_dataset(n=100, mA=[3.0,2.0], mB=[1.0,-2.5], sigmaA=0.5, sigmaB=0.5)
    plot_dataset(cAn, cBn, "task3_dataset_normal")
    
    datasets = [(Xs, Ts, "Symmetric"), (Xa, Ta, "Same Quadrant"), (Xn, Tn, "Normal")]
    for X, T, name in datasets:
        W = np.random.randn(1, 2) * 0.01 
        lr = 0.001

        for _ in range(100):
            error = T.reshape(1, -1) - np.dot(W, X)
            W += lr * np.dot(error, X.T)
        
        plot_boundary(W, X, T, f"No Bias Experiment ({name})", f"task3_nobias_{name}")
    
if __name__ == "__main__":
    np.random.seed(42)
    perceptron_vs_delta()
    online_vs_batch()
    remove_bias()