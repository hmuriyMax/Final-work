# Импортируем библиотеки
import json
import random
import time
import itertools
from concurrent.futures import ThreadPoolExecutor

from keras.layers import Dense
from keras.models import Sequential


def generate_train(length):
    inp = []
    out = []
    for i in range(length):
        testCase = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
        inp.append(testCase)
        if testCase[0] * testCase[1] > 0:
            out.append([1, 0])
        else:
            out.append([0, 1])
    return inp, out


def main(neuros: int, iters: int, x_train, y_train):
    # Создаем модель
    model = Sequential()

    # Добавляем слои
    model.add(Dense(units=neuros, input_dim=2, activation='sigmoid'))
    model.add(Dense(units=2))

    # Компилируем модель
    model.compile(loss='binary_crossentropy',
                  metrics=['accuracy'])

    x_test, y_test = generate_train(100)
    # Обучаем модель
    model.fit(x_train, y_train, epochs=iters)

    # Оцениваем модель
    loss_and_metrics = model.evaluate(x_test, y_test)
    return loss_and_metrics


def worker(args):
    n = args[0]
    j = args[1]
    iters = 2 ** j
    acSum = 0
    for i in range(attempts):
        res = main(n, iters, x_train, y_train)
        acSum += res[1]
    experiments.append({
        "Neurons": n,
        "Iterations": iters,
        "Accuracy": acSum * 100 / attempts
    })


attempts = 1

if __name__ == '__main__':
    x_train, y_train = generate_train(1000)
    experiments = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:

        args = [[i, j] for i in range(1, 21) for j in range(0, 7)]
        # executor.map(worker, args)
        for argList in args:
            worker(argList)
    res = {
        'TotalTime': time.time() - start,
        'Experiments': experiments,
    }
    json.dump(res, open('output.json', 'w'), indent=4)
