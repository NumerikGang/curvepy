import random
import numpy as np
from curvepy.utilities import *


class A:
    xs = [*range(1,11)]

    def __getitem__(self, item):
        return self.xs.__getitem__(item)


if __name__ == '__main__':
    a = A()
    print(a[2:4])