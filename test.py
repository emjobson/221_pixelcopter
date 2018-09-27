# test.py
import matplotlib.pyplot as plt

if __name__ == '__main__':
    my_list = []

    """
    for i in range(500):
        my_list.append(float(i - 250))
    """

    my_list = [1.0, 2.0, 3.0]
    plt.plot([1, 2, 3], [10, 1, 2])
    plt.show()
