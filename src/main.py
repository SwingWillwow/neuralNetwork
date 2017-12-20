from network_component.data_loader import load_data
from network_component import network

training_data, validation_data, test_data = load_data()
layer_num = input('please input the number of layer:')
layer = []
for i in range(int(layer_num)):
    j = input('please input the number of nodes in layer {}:'.format(i))
    layer.append(int(j))
net = network.Network(layer)
epoch = int(input('how many epochs to iterate?'))
eta = float(input('what\'s the learning rate?'))
mini_batch_size = int(input('how many data in the mini-batch?'))
# net.train(training_data,epoch,eta,test_data)
net.train_mini_batch(training_data, epoch, mini_batch_size, eta, test_data)

