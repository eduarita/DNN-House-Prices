import numpy as np
import json
import matplotlib.pyplot as plt

def initialize_parameters_deep(layer_dims):
    """
    Argumentos:
    layer_dims -- lista que contiene el tamaño de cada capa de la red neuronal
    
    Retorna:
    parameters -- diccionario python que contiene los parámetros "W1", "b1", ..., "WL", "bL":
                    Wl -- matriz de pesos de dimensiones (layer_dims[l], layer_dims[l-1])
                    bl -- vector de sesgos de dimensiones (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # número de capas de la red neuronal

    for l in range(1, L):
        parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))

        assert(parameters[f"W{l}"].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters[f"b{l}"].shape == (layer_dims[l], 1))

    return parameters

def linear_forward(A, W, b):
    """
    Implementación de la parte lineal de la propagación hacia adelante.

    Args:
    A -- Activación de la capa anterior (o datos de entrada): (tamaño de la capa anterior, número de ejemplos)
    W -- Matriz de pesos: (tamaño de la capa actual, tamaño de la capa anterior)
    b -- Vector bias: (tamaño de la capa actual, 1)

    Returns:
    Z -- la entrada de la función de activación
    cache -- una tupla que contiene "A", "W" y "b" ; almacenado para la propagación hacia atrás
    """

    Z = np.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))

    cache = (A, W, b)

    return Z, cache

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)

    return A, cache

def compute_cost(Y_hat, Y):
    """
    Calcula el error absoluto medio entre la salida predicha Y_hat y los valores reales Y.

    Argumentos:
    Y_hat -- vector de tamaño (1, número de ejemplos)
    Y -- vector de tamaño (1, número de ejemplos)

    Returna:
    cost -- error absoluto medio entre Y_hat y Y
    """
    m = Y_hat.shape[1]
    cost = np.sum(np.abs(Y_hat - Y)) / m

    return cost

def linear_backward(dZ, cache):
    """
    Implementa la parte de retropropagacion para la capa lineal

    Argumentos:
    dZ -- Gradiente del costo con respecto a la salida lineal (dL/dZ)
    cache -- tupla de valores (A_prev, W, b) obtenidos en la propagacion hacia adelante en la capa actual

    Devuelve:
    dA_prev -- Gradiente del costo con respecto a la activacion (dL/dA) calculado en la capa anterior
    dW -- Gradiente del costo con respecto a los pesos (dL/dW)
    db -- Gradiente del costo con respecto a los sesgos (dL/db)
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def sigmoid_backward(dA, cache):
    Z = cache
    A, _ = sigmoid(Z)
    dZ = dA * A * (1 - A)
    return dZ

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    return dA_prev, dW, db

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters[f"W{l}"], parameters[f"b{l}"], activation="relu")
        caches.append(cache)
        
    AL, cache = linear_activation_forward(A, parameters[f"W{L}"], parameters[f"b{L}"], activation="sigmoid")
    caches.append(cache)
    
    return AL, caches

def L_model_backward(AL, Y, caches):
    """
    Implementa la retropropagación en una red neuronal con L capas ocultas.

    Argumentos:
    AL -- salida de la red neuronal, array de numpy de tamaño (1, número de ejemplos)
    Y -- variable de salida, array de numpy de tamaño (1, número de ejemplos)
    caches -- caché de activaciones y linealidades almacenadas en la propagación hacia adelante

    Retorna:
    grads -- un diccionario con los gradientes
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches) # número de capas
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # Inicialización de la retropropagación
    dAL = np.sign(AL-Y)
    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    # Retropropagación a través de las capas restantes
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "sigmoid")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Actualiza los parámetros utilizando la regla de descenso de gradiente

    Arguments:
    parameters -- diccionario que contiene los parámetros
    grads -- diccionario que contiene los gradientes, output de la backward_propagate
    learning_rate -- tasa de aprendizaje

    Returns:
    parameters -- diccionario que contiene los parámetros actualizados
    """
    
    # Número de capas en la red neuronal
    L = len(parameters) // 2 

    # Actualizar cada parámetro
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters

def model(X, Y, layers_dims, learning_rate=0.01, num_iterations=1000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->REGRESSION.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector of shape (number of examples, 1)
    layers_dims -- list containing the input size and each layer size, of length (L+1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []

    # Initialize parameters
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation
        AL, caches = L_model_forward(X, parameters)

        # Compute cost
        cost = compute_cost(AL, Y)

        # Backward propagation
        grads = L_model_backward(AL, Y, caches)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, cost))

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

    # Plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

#Obtener datos de entrenamiento
X_train = np.load("x.npy").T #Transpuesta; Row: Attribute, Col: Houses
Y_train = np.load("y.npy")
Y_train = Y_train.reshape((Y_train.shape[0],1))
print("Datos de entrenamiento:")
print("Caractersticas:", X_train)
print()
print("Ejemplos:",Y_train)

# Luego, normalice X restando la media y dividiendo por la desviación estándar
# Calculamos la media y la desviación estándar de las características
mean = X_train.mean(axis=1, keepdims=True)
std = X_train.std(axis=1, keepdims=True)

# Aplicamos la regularización de varianza
epsilon = 1e-8
std[std == 0] = epsilon

# Normalizamos las características
X_norm = (X_train - mean) / std
print()
print(X_norm)
print(X_norm.shape)
print(Y_train.shape)

# Definir la arquitectura de la red neuronal
layers_dims = [X_norm.shape[0], 20, 10, 5, 1]

# Entrenar el modelo
costs = model(X_norm, Y_train, layers_dims, learning_rate=0.001, num_iterations=2000, print_cost=True)

# Inicializar los parámetros
parameters = initialize_parameters_deep(layers_dims)

# Crear la lista de capas
dnn_layers = []

for l in range(1, len(layers_dims)):
    layer = {}
    layer["n"] = layers_dims[l]
    layer["w"] = parameters[f"W{l}"].tolist()
    layer["b"] = parameters[f"b{l}"].reshape(-1).tolist()
    
    if l == len(layers_dims) - 1:
        layer["activation"] = "linear"
    else:
        layer["activation"] = "relu"
    
    dnn_layers.append(layer)

# Crear el diccionario final y guardar el archivo
params = {"dnn_layers": dnn_layers}
with open("params.json", "w") as f:
    json.dump(params, f)

# Graficar el costo durante el entrenamiento
plt.plot(costs)
plt.ylabel('costo')
plt.xlabel('iteraciones')
plt.title("Tasa de aprendizaje =" + str(0.01))
plt.show()
