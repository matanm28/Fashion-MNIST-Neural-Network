import sys

from random import shuffle
from typing import Dict, List, Callable

import numpy as np
from numpy import ndarray


def create_one_hot(train_y: ndarray):
    min_label, max_label = train_y.min(), train_y.max()
    vector_size = max_label - min_label + 1
    one_hot_y = np.zeros((train_y.shape[0], vector_size))
    one_hot_y[np.arange(train_y.shape[0]), train_y - min_label] = 1
    return one_hot_y


def shuffle_in_pairs(a: ndarray, b: ndarray) -> (ndarray, ndarray):
    c = list(zip(a, b))
    shuffle(c)
    e, f = zip(*c)
    return np.array(e), np.array(f)


def normalize_data_std(data_set: ndarray):
    mean = data_set.mean(axis=0)
    std_dev = data_set.std(axis=0)
    return np.nan_to_num((data_set - mean) / std_dev)


def normalize_data_min_max(data_set: ndarray, new_min: int = 0, new_max: int = 1):
    data_min = data_set.min(axis=0)
    data_max = data_set.max(axis=0)
    return ((data_set - data_min) / (data_max - data_min)) * (new_max - new_min) + new_min


def sigmoid(x: ndarray):
    return 1. / (1 + np.exp(-x))


def relu(x: ndarray):
    return np.maximum(0, x)


def sigmoid_derivative(dh: ndarray, z: ndarray):
    sig = sigmoid(z)
    return dh * sig * (1 - sig)


def relu_derivative(dh: ndarray, z: ndarray):
    dZ = np.array(dh, copy=True)
    dZ[z <= 0] = 0
    return dZ


def softmax(x: ndarray):
    ret = np.zeros_like(x)
    for i in range(x.shape[0]):
        e_x = np.exp(x[i])
        sum = e_x.sum(axis=0)
        if sum == 0:
            ret[i] = 0
        else:
            ret[i] = e_x / sum
    return ret


def softmax_derivative(dh: ndarray, z: ndarray):
    return softmax(z)


def sigmoid_and_softmax(x: ndarray):
    z = sigmoid(x)
    return softmax(z)


def sigmoid_and_softmax_derivative(dh: ndarray, z: ndarray):
    return softmax(z) * dh


def softmax_single(x: ndarray):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_data_answers_from_file(file_path: str, num_of_rows: int = None):
    if num_of_rows is not None:
        answers = np.loadtxt(file_path, max_rows=num_of_rows, dtype=np.int16)
    else:
        answers = np.loadtxt(file_path, dtype=np.int16)
    return answers


def get_data_from_file(file_path: str, num_of_rows: int = None):
    if num_of_rows is not None:
        data = np.loadtxt(file_path, max_rows=num_of_rows)
    else:
        data = get_data_set_from_file(file_path)
    return data


def get_data_set_from_file(training_set_examples_path: str):
    examples_list = []
    with open(training_set_examples_path, 'r') as training_examples_file:
        lines = training_examples_file.readlines()
        for line in lines:
            data = line.split()
            examples_list.append(data)
    return np.array(examples_list, dtype=np.float)


def initialize_layers(network_architecture: List[Dict], seed: int = 89):
    np.random.seed(seed)
    param_values = {}
    for i, layer in enumerate(network_architecture):
        layer_index = i + 1
        layer_input_size = layer[IN_DIM]
        layer_output_size = layer[OUT_DIM]
        param_values[f'w{layer_index}'] = np.random.randn(layer_output_size, layer_input_size) * 0.1
        param_values[f'b{layer_index}'] = np.ones((1, layer_output_size)) * 0.1
    return param_values


def network_forward_propagation(x, params):
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    z1 = x @ w1.T + b1
    h1 = sigmoid(z1)
    z2 = h1 @ w2.T + b2
    h2 = softmax(z2)
    ret = {'x': x, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2}
    for key in params:
        ret[key] = params[key]
    return h2, ret


def network_back_propagation(cache):
    x, y, z1, h1, z2, h2 = [cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2')]
    dz2 = (h2 - y)
    dw2 = dz2.T @ h1
    db2 = np.sum(dz2, axis=0, keepdims=True)
    dz1 = dz2 @ cache['w2'] * h1 * (1 - h1)
    dw1 = dz1.T @ x
    db1 = np.sum(dz1, axis=0, keepdims=True)
    return {'db1': db1, 'dw1': dw1, 'db2': db2, 'dw2': dw2}


def update(params_value: Dict, grads_value: Dict, network_architecture: List[Dict]):
    for i, layer in enumerate(network_architecture):
        layer_index = i + 1
        params_value[f'w{layer_index}'] -= LEARNING_RATE * grads_value[f'dw{layer_index}']
        params_value[f'b{layer_index}'] -= LEARNING_RATE * grads_value[f'db{layer_index}']
    return params_value


def get_cost_value(y_hat, y_true):
    m = y_hat.shape[1]
    cost = -1 / m * (np.dot(y_true, np.log(y_hat).T) + np.dot(1 - y_true, np.log(1 - y_hat).T))
    return np.squeeze(cost)


def convert_prob_into_label(y_probs: ndarray) -> ndarray:
    return np.argmax(y_probs, axis=1)


def get_accuracy_value(y_hat: ndarray, y_true: ndarray):
    y_hat_labels = convert_prob_into_label(y_hat)
    return (y_hat_labels == y_true).mean()


def count_accuracy_hit(y_hat: ndarray, y_true: ndarray):
    y_hat_labels = convert_prob_into_label(y_hat)
    return np.count_nonzero(y_hat_labels == y_true)


def train(train_x: ndarray, train_y: ndarray, num_of_epochs: int, params_value: Dict[str, ndarray] = None):
    if params_value is None:
        params_value = initialize_layers(NETWORK_ARCHITECTURE)
    accuracy_history = []
    step_size = 1
    for i in reversed(range(1, 51)):
        if train_x.shape[0] % i == 0:
            step_size = i
            break
    for i in range(num_of_epochs):
        train_x_shuffled, train_y_shuffled = shuffle_in_pairs(train_x, train_y)
        train_y_one_hot = create_one_hot(train_y_shuffled)
        temp_accuracy = 0
        for j in range(0, train_x.shape[0], step_size):
            x, y = train_x_shuffled[j:j + step_size - 1], train_y_one_hot[j:j + step_size - 1]
            y_hat_probabilities, cache = network_forward_propagation(x, params_value)
            cache['y'] = y
            temp_accuracy += count_accuracy_hit(y_hat_probabilities, train_y_shuffled[j:j + step_size - 1])
            grad_values = network_back_propagation(cache)
            params_value = update(params_value, grad_values, NETWORK_ARCHITECTURE)
        accuracy_history.append(temp_accuracy / train_x.shape[0])
    return params_value, np.array(accuracy_history)


def neural_network(test_x: ndarray, train_x: ndarray, train_y: ndarray):
    params_value, accuracy_history = train(train_x, train_y, NEURAL_NETWORK_EPOCHS)
    for i in range(NUM_OF_EXTRA_TRAININGS):
        print(f'Average Accuracy: {np.average(accuracy_history[-5:])*100}%')
        if np.average(accuracy_history[-5:]) > 0.91:
            print('Good Enough!')
            break
        print('Not so great...')
        print('Pushing through another epoch')
        params_value, accuracy_history = train(train_x, train_y, NEURAL_NETWORK_EPOCHS, params_value)
    y_hat, cache = network_forward_propagation(test_x, params_value)
    y_hat_labels = convert_prob_into_label(y_hat)
    return y_hat_labels


def k_fold(training_set: ndarray, y_true: ndarray, num_of_folds: int,
           func: Callable[[ndarray, ndarray, ndarray], ndarray]):
    from sklearn.metrics import recall_score, precision_score
    folds = np.array_split(training_set, num_of_folds)
    size = 0
    percentage_list = []
    precision_score_list = []
    recall_score_list = []
    average_type = 'macro'
    i = 1
    for fold in folds:
        print(f'starting fold {i} out of {num_of_folds}')
        i += 1
        mask = np.zeros(training_set.shape[0], dtype=bool)
        mask[size:size + fold.shape[0]] = True
        validation = training_set[mask]
        training = training_set[~mask]
        ans = y_true[~mask]
        y = y_true[mask]
        y_pred = func(validation, training, ans)
        percentage_list.append(np.count_nonzero(y == y_pred) / y_pred.size)
        precision_score_list.append(
            precision_score(y, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], average=average_type, zero_division=1))
        recall_score_list.append(
            recall_score(y, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], average=average_type, zero_division=1))
        size += fold.shape[0]
    print(f'Average Accuracy: {sum(percentage_list) * 100 / len(percentage_list)}%')
    print(f'Average Precision: {sum(precision_score_list) * 100 / len(precision_score_list)}%')
    print(f'Average Recall: {sum(recall_score_list) * 100 / len(recall_score_list)}%')


def main(train_x_file: str, train_y_file: str, test_x_file: str, num_of_folds: int = None, num_of_rows: int = None):
    train_y = get_data_answers_from_file(train_y_file, num_of_rows)
    train_x = normalize_data_std(get_data_from_file(train_x_file, num_of_rows))
    test_x = normalize_data_std(get_data_from_file(test_x_file))
    if num_of_folds is not None:
        k_fold(train_x, train_y, num_of_folds, neural_network)
    else:
        y_pred = neural_network(test_x, train_x, train_y)
        with open('test_y', 'w') as file:
            for i in range(y_pred.size):
                file.write(f'{y_pred[i]}\n')


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Not enough args", file=sys.stderr)
        exit(1)
    IN_DIM = 'in_dim'
    OUT_DIM = 'out_dim'
    NEURAL_NETWORK_EPOCHS = 50
    NUM_OF_EXTRA_TRAININGS = 3
    LEARNING_RATE = 0.01
    NETWORK_ARCHITECTURE = [
        {IN_DIM: 784, OUT_DIM: 120},
        {IN_DIM: 120, OUT_DIM: 10}
    ]
    LABELS = {
        0: 'T-shirt/top',
        1: 'Trousers',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }
    main(sys.argv[1], sys.argv[2], sys.argv[3])
