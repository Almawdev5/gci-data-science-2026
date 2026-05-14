import numpy as np

def homework(a):
    my_result = a[(a % 5 == 0) & (a % 2 == 1)]
    return my_result

a = np.array([1, 5, 10, 3, 4, 25, 30])

print(homework(a))