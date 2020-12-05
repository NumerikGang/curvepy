import matplotlib.pyplot as plt

class BezierCurve():
    def __init__(self, bs):
        bs = [lambda t: b for b in bs]
        self.curves = [bs]
        self.n = len(bs) - 1

        self._gen_all_curves()

        self.curve = self.curves[-1][0]

    def _gen_all_curves(self):
        for step in range(1, self.n + 1):
            self.curves.append(
                [self._gen_single_curve(step, k) for k in range(len(self.curves[step - 1]) - 1)]
            )

    def _gen_single_curve(self, step, k):
        '''Expecting not to be base case'''
        return lambda t: (1 - t) * self.curves[step - 1][k](t) + t * self.curves[step - 1][k + 1](t)


if __name__ == '__main__':
    x = BezierCurve([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    print(x.curves)