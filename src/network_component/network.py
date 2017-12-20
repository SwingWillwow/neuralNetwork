import random
import numpy as np
import time


class Network(object):
    def __init__(self, layer_size):
        self.layer_size = layer_size  # a list contain the how many layers we have and how many node in each layer
        self.layer_number = len(layer_size)  # the number of layer
        self.weights = [0.1*np.random.randn(y, x)
                        for x, y in zip(layer_size[:-1], layer_size[1:])]
        self.bias = [0.1*np.random.randn(x, 1) for x in layer_size[1:]]

    def __forward_step(self, a):
        for b, w in list(zip(self.bias, self.weights)):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def __back_propagation_with_batch(self, mini_batch, batch_size):
        # unzip the mini_batch
        mini_batch = list(zip(* mini_batch))
        mini_batch = np.array(mini_batch)
        x = np.zeros((784, 0))
        # stack  vector xs (784,1) into a matrix (784,batch_size)
        for single_x in mini_batch[0]:
            x = np.hstack((x, single_x))
        y = np.zeros((10, 0))
        # stack vector ys (10,1) into a matrix (10,batch_size)
        for single_y in mini_batch[1]:
            y = np.hstack((y, single_y))
        # init the nabla matrix this record the partial derivative of ws and bs
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        # activation's. also the step of feed forward
        activation = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.bias):
            z = np.dot(w, activation) + b  # add b to each cols in (w dot activation)
            zs.append(z)
            activation = sigmoid(z)  # apply sigmoid function to all element in activation
            activations.append(activation)  # record all the activations
        # count the delta of the output layer
        delta = -(y - activations[-1])*sigmoid_prime(zs[-1])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = np.sum(delta, axis=1)  # sum all the col of delta, to get the sum delta
        nabla_b[-1] = nabla_b[-1].reshape(nabla_b[-1].size, 1)
        for l in range(2, self.layer_number):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(zs[-l])
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            nabla_b[-l] = np.sum(delta, axis=1)
            nabla_b[-l] = nabla_b[-l].reshape(nabla_b[-l].size, 1)
        return nabla_w, nabla_b

    def __back_propagation(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # initialize nabla_w and nabla_b
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        activation = x
        activations = [x]
        zs = []
        # counting z's and a's i think this could be done in forward_step. change letter
        for w, b in zip(self.weights, self.bias):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = delta
        for l in range(2, self.layer_number):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(zs[-l])
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
            nabla_b[-l] = delta
        return nabla_w, nabla_b

    def __mini_batch_gradient_descent(self, mini_batch, eta):
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.bias]
        for x, y in mini_batch:
            nabla_w, nabla_b = self.__back_propagation(x, y)
            delta_w = [dw + nw for dw, nw in zip(delta_w, nabla_w)]
            delta_b = [db + nb for db, nb in zip(delta_b, nabla_b)]
        self.weights = [w - (eta / len(mini_batch)) * dw for w, dw in zip(self.weights, delta_w)]
        self.bias = [b - (eta / len(mini_batch)) * db for b, db in zip(self.bias, delta_b)]

    def __mini_batch_gradient_descent_better(self, mini_batch, eta):
        nabla_w,nabla_b = self.__back_propagation_with_batch(mini_batch, len(mini_batch))
        # use nabla_w and nabla_b to update weights and biases
        self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.bias = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.bias, nabla_b)]

    def __batch_gradient_descent(self, training_data, eta):
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.bias]
        for training_input, training_output in training_data:
            nabla_w, nabla_b = self.__back_propagation(training_input, training_output)
            delta_w = [dw + nw for dw, nw in zip(delta_w, nabla_w)]
            delta_b = [db + nb for db, nb in zip(delta_b, nabla_b)]
        self.weights = [w - (eta / len(training_data)) * dw for w, dw in zip(self.weights, delta_w)]
        self.bias = [b - (eta / len(training_data)) * db for b, db in zip(self.bias, delta_b)]

    def __evaluate(self, test_data):
        # count how many test_data can be recognized by the network
        test_result = [(np.argmax(self.__forward_step(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in test_result)

    def train_mini_batch(self, training_data, epochs,  mini_batch_size, eta, test_data=None):
        training_data_len = len(training_data)
        if test_data:
            test_data_len = len(test_data)
        for i in range(epochs):
            random.shuffle(training_data)
            # get mini_batches base the mini_batch_size
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range(0, training_data_len, mini_batch_size)]
            # for each mini_batch use gradient descent method to update ws and bs
            for mini_batch in mini_batches:
                self.__mini_batch_gradient_descent(mini_batch, eta)
                # self.__mini_batch_gradient_descent_better(mini_batch, eta)
            if test_data:
                print("epoch {} : {} / {}".format(i+1, self.__evaluate(test_data), test_data_len))
            else:
                print("epoch {} complete.".format(i+1))
        # print how many time this program used
        print(change_time(time.process_time()))

    def train(self, training_data, epochs, eta, test_data=None):
        if test_data:
            test_data_len = len(test_data)
        for i in range(epochs):
            self.__batch_gradient_descent(training_data, eta)
            if test_data:
                print("epoch {} : {} / {}".format(i+1, self.__evaluate(test_data), test_data_len))
            else:
                print("epoch {} complete.".format(i+1))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def change_time(t):
    t = int(t)
    return '{} minutes {} seconds'.format(t // 60, t % 60)
