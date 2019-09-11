import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import minimize

def initialize_weights(layer_in,layer_out):
    """
    Randomly initialises the theta matrices based on value of layers passed
    :param layer_in: input layer
    :param layer_out: output layer
    :return: randomly initialised theta matrix
    """
    epsilon = 0.12
    theta = np.random.rand(layer_in,layer_out) * (2* epsilon) - epsilon
    return theta

def sigmoid(mat):
    """
    Calculates the sigmoid of given matrix
    :param mat: matrix of any dimension
    :return: sigmoid
    """
    sigmoid = (1 + np.exp(-mat))
    sigmoid = np.divide(1,sigmoid)
    return sigmoid

def forward_prop(X,thetas):
    """
    Calculates a1, z2, a2, z3 and h using forward propagation algorithm
    :param X: matrix containing train features
    :param thetas: dictionary of theta having theta1 and theta2
    :return: dictionary containing values - a1, z2, a2, z3, h
    """
    m = int(X.shape[0])
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = np.dot(thetas.get("Theta1"),a1.T)  # 25x401 * 401x5000 = 25x5000

    n = int(z2.shape[1])
    a2 = sigmoid(z2)
    a2 = np.row_stack((np.ones(n),a2))

    z3 = np.dot(thetas.get("Theta2"),a2)
    a3 = sigmoid(z3)

    # print(a1.shape)
    # print(np.transpose(z2).shape)
    # print(a2.shape)
    # print(z3.shape)
    # print(a3.shape)

    forward_params = { "A1":a1,
                        "Z2":np.transpose(z2),
                        "A2":np.transpose(a2),
                       "Z3":np.transpose(z3),
                       "h":np.transpose(a3)
                    }
    return forward_params

def backward_prop(params, input_size, hidden_size, num_labels,X, y, lambda_reg):
#def backward_prop(forward_params,thetas,X,y):
    m = int(X.shape[0])
    X_train = np.matrix(X)
    y = np.matrix(y)


    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    thetas = {"Theta1": theta1, "Theta2": theta2}
    forward_params = forward_prop(X_train,thetas)


    # Some initializations
    J = 0
    capital_delta1 = np.zeros(theta1.shape)  # (25, 401)
    capital_delta2 = np.zeros(theta2.shape)  # (10, 26)

    # Calculating cost
    J = compute_cost(thetas,X,y)
    J = add_regularizarion_cost(X,thetas,J,lambda_reg)

    # Starting logic for back propagation algorithm

    for i in range(m):
        a1t = forward_params.get("A1")[i, :]                    # (1, 401)
        z2t = forward_params.get("Z2")[i, :]                    # (1, 25)
        a2t = forward_params.get("A2")[i, :]                    # (1, 26)
        ht = forward_params.get("h")[i, :]                      # (1, 10)
        yt = y[i, :]                                            # (1, 10)

        # Calculating delta 3
        delta3 = ht - yt                                        # (1, 10)

        # Calculating delta 2 but adding bias term of 1's here
        z2t = np.insert(z2t, 0, values=np.ones(1))               # (1, 26)
        delta2 = np.multiply(np.transpose(np.dot(np.transpose(thetas.get("Theta2")),np.transpose(delta3))),sigmoid_gradient(z2t)) # (1,26)

        # Calculating capital deltas
        # capital_delta1 = capital_delta1 + np.dot(delta2[1:, :], forward_params["A1"])
        # capital_delta2 = capital_delta2 + np.dot(delta3, forward_params["A2"].T)
        capital_delta1 = capital_delta1 + np.dot(np.transpose(np.matrix((delta2)))[1:,:],np.matrix(a1t))
        capital_delta2 = capital_delta2 + np.dot(np.matrix(delta3).T,np.matrix(a2t))

    # Calculating unregularized gradient
    capital_delta1 = capital_delta1 / m
    capital_delta2 = capital_delta2 / m

    # Adding the gradient regularization term
    capital_delta1[:,1:] = capital_delta1[:,1:] + (thetas.get("Theta1")[:,1:] * lambda_reg) / m
    capital_delta2[:,1:] = capital_delta2[:,1:] + (thetas.get("Theta2")[:,1:] * lambda_reg) / m

    # # Adding regularization term to the gradients
    # for i in range(unreg_gradient_theta_1.shape[0]):
    #     for j in range(1,unreg_gradient_theta_1.shape[1]):
    #         unreg_gradient_theta_1[i,j] = unreg_gradient_theta_1[i,j] + ((lambda_reg/m)* thetas.get("Theta1")[i,j] )
    #
    # for i in range(unreg_gradient_theta_2.shape[0]):
    #     for j in range(1,unreg_gradient_theta_2.shape[1]):
    #         unreg_gradient_theta_2[i,j] = unreg_gradient_theta_2[i,j] + ((lambda_reg/m)* thetas.get("Theta2")[i,j] )

    grad = np.concatenate((np.ravel(capital_delta1), np.ravel(capital_delta2)))

    # backward_params= { "J":J,
    #                    "delta1":unreg_gradient_theta_1,
    #                    "delta2":unreg_gradient_theta_2
    #                 }
    return J, grad

#def compute_cost(h,train_data):
def compute_cost(thetas, X, y):
    """
    Computes cost  to evaluate the loss for a given set of network parameters
    :param thetas: dictionary of theta having theta1 and theta2
    :param X: matrix containing train features
    :param y: y matrix containing values from 0 to 9
    :return: computed cost
    """
    m = int(X.shape[0])
    theta1 = thetas.get("Theta1")
    theta2 = thetas.get("Theta2")

    forward_params = forward_prop(X,thetas)
    h = forward_params.get("h")

    term1 = -1 * np.multiply(y,np.log(h))
    term2 =  np.multiply((1-y),np.log(1-h))

    J = (np.sum(np.sum(term1 - term2)))/m

    return J

def add_regularizarion_cost(X,thetas,J,lambda_reg):
    """
    Add regularisation term to the cost function given value of lambda
    :param X: matrix containing train features
    :param thetas: dictionary of theta having theta1 and theta2
    :param J: cost without regularisation
    :param lambda_reg: specified lambda value
    :return: regularised cost
    """
    m = int(X.shape[0])
    theta1 = np.square(thetas.get("Theta1").ravel())
    theta2 = np.square(thetas.get("Theta2").ravel())
    sum = np.sum(theta1) + np.sum(theta2)
    temp_reg = (lambda_reg/(2*m)) * sum
    J = J + temp_reg
    return J

def sigmoid_gradient(mat):
    """
    Calculates sigmoid gradient of a matrix given by
        g'(z) =d/dz(g(z))= g(z)(1 - g(z))
        where
        g(z) =1 / (1 + e^-z)
    :param mat: takes in a matrix of any shape
    :return: the sigmoid gradient of that matrix
    """
    sig = sigmoid(mat)
    sigmoid_grad = np.multiply(sig,(1-sig))
    return sigmoid_grad

def model_nn(train_data,test_data,layers,y_single_column):
    X_train = train_data[0]
    y_train = train_data[1]

    input_layer_size = layers[0]
    hidden_layer_size = layers[1]
    output_labels = layers[2]
    m = int(X_train.shape[0])

    # bias_input_layer = np.ones(m)
    # train_data[0] = np.column_stack((bias_input_layer, train_data[0]))

    print("Units in input layer : " + str(input_layer_size))
    print("Units in hidden layer : " + str(hidden_layer_size))
    print("Units in output layer : " + str(output_labels)),print()

    weights = loadmat("E:/machine-learning-coursera/machine-learning-ex4/ex4/ex4weights.mat")
    theta_1 = weights['Theta1']
    theta_2 = weights['Theta2']
    print("Shape of pre-defined Theta 1 : " + str(theta_1.shape))
    print("Shape of pre-defined Theta 2 : " + str(theta_2.shape)), print()
    predefined_thetas = {"Theta1": theta_1, "Theta2": theta_2}

    # J = compute_cost(predefined_thetas,X_train,y_train)

    # regularised_J = add_regularizarion_cost(X_train,predefined_thetas,J,1)
    # print(regularised_J)

    #sigmoid_grad = sigmoid_gradient(np.array([-1,-0.5,0,0.5,1]))

    # We need to randonly initialize weights in order to break the neural network symmetry
    # initial_theta_1 = initialize_weights(layers[1], layers[0] + 1)
    # initial_theta_2 = initialize_weights(layers[2], layers[1] + 1)
    # initial_thetas = {"Theta1": initial_theta_1, "Theta2": initial_theta_2}
    # print("Shape of initial Theta 1 : " + str(initial_theta_1.shape))
    # print("Shape of initial Theta 2 : " + str(initial_theta_2.shape))

    params = params = (np.random.random(size=hidden_layer_size * (input_layer_size + 1) + output_labels * (hidden_layer_size + 1)) - 0.5) * 0.25


    #initial_thetas = {"Theta1": theta1, "Theta2": theta2}
    #forward_parameters = forward_prop(X_train,initial_thetas)

    J,grad = backward_prop(params,input_layer_size,hidden_layer_size,output_labels,X_train,y_train,lambda_reg = 1)
    print(J)

    learning_rate = 1
    params = params = (np.random.random(size=hidden_layer_size * (input_layer_size + 1) + output_labels * (hidden_layer_size + 1)) - 0.5) * 0.25
    # fmin = minimize(fun=backward_prop, x0=params, args=(input_layer_size, hidden_layer_size, output_labels,
    #                         X_train, y_train, learning_rate),
    #                 method='TNC', jac=True, options={'maxiter': 250})
    fmin = minimize(fun=backward_prop, x0=params, args=(input_layer_size,hidden_layer_size,output_labels,X_train,y_train,learning_rate),
                    method='TNC', jac=True, options={'maxiter': 250})
    print(fmin)

    # Calculating predictions
    X = np.matrix(X_train)
    theta1 = np.matrix(np.reshape(fmin.x[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1))))
    theta2 = np.matrix(np.reshape(fmin.x[hidden_layer_size * (input_layer_size + 1):], (output_labels, (hidden_layer_size + 1))))

    thetas = {"Theta1": theta1, "Theta2": theta2}
    pred_params = forward_prop(X, thetas)
    y_pred = np.array(np.argmax(pred_params.get("h"), axis=1) + 1)

    # Computing accuracy of our network
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y_single_column)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('accuracy = {0}%'.format(accuracy * 100))

    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.to_csv("y_pred")

    y_single_column_df = pd.DataFrame(y_single_column)
    y_single_column_df.to_csv("single columns")


    # print(J)
    # print(grad.shape)
    # for i in range(0,m):
    #
    #     forward_parameters = forward_prop(train_data[0][i:i+1,:],initial_thetas)
    #
    #     backward_parameters = backward_prop(forward_parameters,initial_thetas,train_data[0][i:i+1,:],train_data[1][i:i+1,:])
    #
    #     # Initialising capital_delta2
    #     capital_delta1 = np.zeros(initial_thetas["Theta1"].shape)
    #     capital_delta2 = np.zeros(initial_thetas["Theta2"].shape)
    #
    #     # Calculating capital deltas
    #     capital_delta1 = capital_delta1 + np.dot(backward_parameters["D2"][1:,:],forward_parameters["A1"])
    #     capital_delta2 = capital_delta2 + np.dot(backward_parameters["D3"],forward_parameters["A2"].T)
    #
    # # Calculating unregularized gradient
    # unreg_gradient_theta_1 = capital_delta1/m
    # unreg_gradient_theta_2 = capital_delta2/m
    #
    # # Adding regularization term to the gradients
    # lambda1 = 3
    # for i in range(unreg_gradient_theta_1.shape[0]):
    #     for j in range(1,unreg_gradient_theta_1.shape[1]):
    #         unreg_gradient_theta_1[i,j] = unreg_gradient_theta_1[i,j] + ((lambda1/m)* theta_1[i,j] )
    #
    # for i in range(unreg_gradient_theta_2.shape[0]):
    #     for j in range(1,unreg_gradient_theta_2.shape[1]):
    #         unreg_gradient_theta_2[i,j] = unreg_gradient_theta_2[i,j] + ((lambda1/m)* theta_2[i,j] )

