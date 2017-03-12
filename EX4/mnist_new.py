
# import matplotlib.pyplot as plt
# import displaydata.displayData as dsp
# import numpy as np
# import scipy.optimize as op
# import scipy.io as scio
# import random
# import os
import displaydata.displayData as dsp
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
import scipy.io as scio
import random
import os

def Sigmoid(z):
    return 1/(1+np.exp(-z));

def sigmoidGradient(z):
    g = np.zeros(np.size(z));
    g = Sigmoid(z) * (1-Sigmoid(z));
    return g;

def  h_theta_x(Theta1, Theta2, X):
    m = np.size(X,0);
    n = np.size(X,1);
    num_labels = np.size(Theta2,0);
    a1 = X;
    a1 = np.c_[np.ones(m),a1];
    a2 = Sigmoid(a1.dot(Theta1.T));
    a2 = np.c_[np.ones(m),a2];
    a3 = Sigmoid(a2.dot(Theta2.T));
    return a3

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, lamda):
    #computing Theta1 Theta2
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1)));
    Theta2 = np.reshape(nn_params[((hidden_layer_size * (input_layer_size + 1))):],(num_labels, (hidden_layer_size + 1)));
    # print(Theta1.shape, Theta2.shape)
    m = np.size(X,0);
    J = 0;
    Theta1_grad = np.zeros(Theta1.shape);
    Theta2_grad = np.zeros(Theta2.shape);
    # print(Theta1_grad.shape, Theta2_grad.shape)
    #将y向量化
    E = np.eye(num_labels);
    Y = np.reshape(E[y-1],(m,num_labels));
    a3 = h_theta_x(Theta1,Theta2,X);
    # print(a3[0],Y[0])
    temp1 = Theta1[::,1:];   #% 先把theta(1)拿掉，不参与正则化
    temp2 = Theta2[::,1:];
    # print(temp1.shape,temp2.shape)
    temp1 = sum(temp1**2);     #% 计算每个参数的平方，再就求和
    temp2 = sum(temp2**2);

    cost = Y * np.log(a3) + (1 - Y ) * np.log((1 - a3)); # % cost是m*K(5000*10)的结果矩阵  sum(cost(:))全部求和
    J= -1  * sum(sum(cost[::,::]))/ m + lamda * ( sum(temp1[:])+ sum(temp2[:]))/(2*m);
    print("cost function:::",J)
    return J;

def randInitializeWeights(L_in, L_out):
    W = np.zeros((L_out,L_in+1));
    epsilon_init = 0.12;
    W = np.random.rand(L_out,L_in+1) * 2 * epsilon_init - epsilon_init;
    return W;

def trans(a):
    m = np.size(a,0);
    n = np.size(a,1);
    a.shape = (n,m);
    b = np.transpose(a);
    return b;

def gradient(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, lamda):
    print("computing......")
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1)));
    Theta2 = np.reshape(nn_params[((hidden_layer_size * (input_layer_size + 1))):],(num_labels, (hidden_layer_size + 1)));
    # print(Theta1.shape, Theta2.shape)
    m = np.size(X,0);
    #计算delta
    # 计算 Gradient
    delta_1 = np.zeros(Theta1.shape);#25,401
    delta_2 = np.zeros(Theta2.shape);#10,26
    X = np.c_[np.ones(m),X];
    for t in range(m):
        a1 = X[t,:].T;#401,1
        a1 = a1.reshape(len(a1),1);
        # print("a1",a1.shape)
        z2 = Theta1.dot(a1);#25,1
        z2 = z2.reshape((len(z2),1));
        # print("z2",z2.shape)
        a2 = Sigmoid(z2);#25,1
        a2 = np.r_[[[1]],a2];#26,1
        # print("a2",a2.shape)
        z3 = Theta2.dot(a2);#10,1
        a3 = Sigmoid(z3);#10,1
        # print("a3",a3.shape)
        err3 = np.zeros((num_labels,1));#10,1
        for k in range(num_labels):
            err3[k] = a3[k] - (y[t]==k+1)
        err2 = (Theta2.T).dot(err3);#26,1
        # print(err2.shape)
        err2 = err2[1:,:] * sigmoidGradient(z2);
        # print(err2.shape)
        delta_2 = delta_2 + err3.dot(a2.reshape((1,26)));
        delta_1 = delta_1 + err2.dot(a1.reshape((1,401)));

    Theta1_temp = np.c_[np.zeros(np.size(Theta1,0)),Theta1[:,1:]];
    Theta2_temp = np.c_[np.zeros(np.size(Theta2,0)),Theta2[:,1:]];
    Theta1_grad = delta_1 / m + lamda * Theta1_temp / m;
    Theta2_grad = delta_2 / m + lamda * Theta2_temp / m;

    Theta1_grad_temp = Theta1_grad.flatten();
    Theta2_grad_temp = Theta2_grad.flatten();
    grad = np.append(Theta1_grad_temp,Theta2_grad_temp);
    return grad;

input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   #25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10

print('Loading and Visualizing Data ...\n')
data = scio.loadmat('ex4data1.mat');
x = data['X']
y = data['y']
# print(X.shape, y.shape)
m = np.size(x,0);#行，训练集大小
n = np.size(x,1);#列，特征数量
#相当于随机100个数字
sel = np.arange(m);
random.shuffle(sel);
sel = sel[100:200];
#选择100个数据显示
dsp.displayData(x[sel,:],20);
###########part2#################
print('\nLoading Saved Neural Network Parameters ...\n')

#Load the weights into variables Theta1 and Theta2
th = scio.loadmat('ex4weights.mat');
Theta1 = th['Theta1']
Theta2 = th['Theta2']



list_theta1 = Theta1.flatten();
list_theta2 = Theta2.flatten();
# nn_params = np.append(list_theta1,list_theta2);
nn_params = np.append(list_theta1,list_theta2);
# print(nn_params.shape)
# nnCostFunction(theta_temp,input_layer_size,hidden_layer_size,num_labels,x,y,1)
# print(Theta1.shape, Theta2.shape)
initial_Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_Theta1_temp = initial_Theta1.flatten();
initial_Theta2_temp = initial_Theta2.flatten();
initial_nn_params = np.append(initial_Theta1_temp,initial_Theta2_temp);
print(initial_nn_params.shape)
lamda = 3
Result = op.minimize(fun = nnCostFunction,x0 = initial_nn_params,args = ( input_layer_size, hidden_layer_size,
                   num_labels, x, y, lamda), method = 'L-BFGS-B',jac=gradient)
rr = Result.x;
print(rr.shape)
np.save("nn_theta2",rr);
np.savetxt("nn_theta2.txt",rr);