import numpy as np
from scipy.io import loadmat

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

    :param mat:
    :return:
    """
    sigmoid = (1 + np.exp(-mat))
    sigmoid = np.divide(1,sigmoid)
    return sigmoid

def add_bias_term(mat,new_cols):
    """

    :param mat:
    :return:
    """
    return 0

def forward_prop(train_data,thetas):
    """

    :return:
    """
    # Adding bias unit of 1's - we will add a columns of 1's to our X matrix
    # m = int(train_data[0].shape[0])
    # bias_input_layer = np.ones(m)
    # new_X = np.column_stack((bias_input_layer,train_data[0]))
    # print(new_X.shape)
    # Performing forward propagation step by step
    a1 = train_data
    z2 = np.dot(thetas.get("Theta1"),np.transpose(a1))  # 25x401 * 401x5000 = 25x5000

    n = int(z2.shape[1])
    a2 = sigmoid(z2)
    a2 = np.row_stack((np.ones(n),a2))

    z3 = np.dot(thetas.get("Theta2"),a2)
    a3 = sigmoid(z3)

    forward_params = { "Z2":z2,
                        "A2":a2,
                       "Z3":z3,
                       "A3":a3
                    }
    return forward_params

def backward_prop(forward_params,thetas,X,y):

    # Calculating delta 3
    delta3 = forward_params["A3"] - np.transpose(y)

    # Calculating delta 2 but adding bias term of 1's here
    n = int(forward_params["Z2"].shape[1])
    z2 = np.row_stack((np.ones(n), forward_params["Z2"]))
    delta2 = np.dot(np.transpose(thetas.get("Theta2")),delta3) * sigmoid_gradient(z2)

    

    backward_params= { "D3": delta3,
                        "D2": delta2
                    }
    return backward_params

def compute_cost(h,train_data):

    m = int(train_data[0].shape[0])
    term1 = -1 * np.transpose(train_data[1]) * np.log(h)
    term2 = (1-np.transpose(train_data[1])) * np.log(1-h)

    J = (np.sum(np.sum(term1 - term2)))/m
    #J = -1 * (1 / m) * np.sum((np.log(h.T) * (train_data[1]) + np.log(1 - h).T * (1 - train_data[1]) ))

    print("Cost associated is : " + str(J))
    return J

def add_regularizarion(train_data,thetas,J,lambda_reg):
    m = int(train_data[0].shape[0])
    theta1 = np.square(thetas.get("Theta1").ravel())
    theta2 = np.square(thetas.get("Theta2").ravel())
    sum = np.sum(theta1) + np.sum(theta2)
    temp_reg = (lambda_reg/(2*m)) * sum
    J = J + temp_reg
    print("Regularised Cost is :" + str(J) + " with lambda value as : " + str(lambda_reg)),print()
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

def model_nn(train_data,test_data,layers):
    """

    :param train_data:
    :param test_data:
    :param layers:
    :return:
    """
    train_data = list(train_data)
    input_layer_size = layers[0]
    hidden_layer_size = layers[1]
    output_labels = layers[2]
    m = int(train_data[0].shape[0])
    bias_input_layer = np.ones(m)
    train_data[0] = np.column_stack((bias_input_layer, train_data[0]))

    # We need to randonly initialize weights in order to break the neural network symmetry
    print("Units in input layer : " + str(input_layer_size))
    print("Units in hidden layer : " + str(hidden_layer_size))
    print("Units in output layer : " + str(output_labels)),print()

    weights = loadmat("E:/machine-learning-coursera/machine-learning-ex4/ex4/ex4weights.mat")
    theta_1 = weights['Theta1']
    theta_2 = weights['Theta2']
    print("Shape of pre-defined Theta 1 : " + str(theta_1.shape))
    print("Shape of pre-defined Theta 2 : " + str(theta_2.shape)), print()
    predefined_thetas = {"Theta1": theta_1, "Theta2": theta_2}

    #h = forward_prop(train_data,predefined_thetas)

    #J = compute_cost(h,train_data)

    #regularised_J = add_regularizarion(train_data,predefined_thetas,J,1)

    sigmoid_grad = sigmoid_gradient(np.array([-1,-0.5,0,0.5,1]))

    initial_theta_1 = initialize_weights(layers[1], layers[0] + 1)
    initial_theta_2 = initialize_weights(layers[2], layers[1] + 1)
    initial_thetas = {"Theta1": initial_theta_1, "Theta2": initial_theta_2}
    print("Shape of initial Theta 1 : " + str(initial_theta_1.shape))
    print("Shape of initial Theta 2 : " + str(initial_theta_2.shape))

    for i in range(0,m):

        forward_parameters = forward_prop(train_data[0][i:i+1,:],initial_thetas)

        backward_parameters = backward_prop(forward_parameters,initial_thetas,train_data[0][i:i+1,:],train_data[1][i:i+1,:])
        break
