import numpy as np

if __name__ == '__main__':
    x = np.random.random(10)
    print(x)
    min_distance_in_row = x.min()
    min_end = int(np.argwhere(x == min_distance_in_row)[0])
    print(min_end)