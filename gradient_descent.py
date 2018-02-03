import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def error_vector(y_vector, y_hat_vector):
    errors = [y_vector[i] * np.log(y_hat_vector[i]) +
              (1 - y_vector[i]) * np.log(1 - y_hat_vector[i])
              for i in range(len(y_vector))]
    return errors


def error(y_vector, y_hat_vector):
    errors = error_vector(y_vector, y_hat_vector)
    return -sum(errors) / len(y_vector)


def prediction(X_mat, w_vector, b_scalar):
    y_hat_vector = sigmoid(np.matmul(X_mat, w_vector) + b_scalar)
    return y_hat_vector


def gradients(X_mat, y_vector, y_hat_vector):
    dw1 = [-(y_vector[i] - y_hat_vector[i]) * X_mat[i][0] for i in range(len(y_vector))]
    dw2 = [-(y_vector[i] - y_hat_vector[i]) * X_mat[i][1] for i in range(len(y_vector))]
    db = [-(y_vector[i] - y_hat_vector[i]) for i in range(len(y_vector))]

    return dw1, dw2, db


def gradient_descent(X_mat, y_vector, w_vector, b_scalar, learn_rate=0.01):
    y_hat_vector = prediction(X_mat, w_vector, b_scalar)
    dw1, dw2, db = gradients(X_mat, y_vector, y_hat_vector)

    w_vector[0] -= learn_rate * sum(dw1)
    w_vector[1] -= learn_rate * sum(dw2)
    b_scalar -= learn_rate * sum(db)

    e = error(y_vector, y_hat_vector)

    return w_vector, b_scalar, e


def __main__():
    X_mat = []
    y_vector = []
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []

    with open("./data.txt", 'r') as f:
        for line in f:
            rows = line.split(',')
            num0 = float(rows[0])
            num1 = float(rows[1])
            num2 = int(rows[2])

            if num2 == 0:
                type1_x.append(num0)
                type1_y.append(num1)
            elif num2 == 1:
                type2_x.append(num0)
                type2_y.append(num1)

            X_mat.append([num0, num1])
            y_vector.append(num2)

    X_mat = np.array(X_mat)
    y_vector = np.array(y_vector)

    w_vector = np.array(np.random.rand(2, 1)) * 2 -1
    b_scalar = np.random.rand(1)[0] * 2 - 1

    line_args = [] # wx + b
    errors = []

    while True:
        w_vector, b_scalar, error = gradient_descent(X_mat, y_vector, w_vector, b_scalar)
        # w1x1 + w2x2 + b == x2 = -w1x1/w2 - b/w2
        line_args.append((-w_vector[0] / w_vector[1], -b_scalar / w_vector[1]))
        errors.append(error)
        if error < 0.15:
            break

    print(line_args[len(line_args) - 1])
    print(errors)

    plt.figure(figsize=(8, 5), dpi=80)
    axes = plt.subplot()

    type1 = axes.scatter(type1_x, type1_y, s=40, c='red')
    type2 = axes.scatter(type2_x, type2_y, s=40, c='green')

    plt.xlabel('x1')
    plt.ylabel('x2')
    axes.legend((type1, type2), ('0', '1'), loc=1)

    x = np.linspace(0, 1.0)
    y = line_args[len(line_args) - 1][0] * x + line_args[len(line_args) - 1][1]
    axes.plot(x, y)

    # for args in line_args:
    #     x = np.linspace(0, 1.0)
    #     y = args[0] * x + args[1]
    #     axes.plot(x, y)

    plt.show()


if __name__ == "__main__":
    __main__()

