import matplotlib.pyplot as plt

# https://stackoverflow.com/a/7152649/9958281
class DeCasteljau():
    class _BezierCurve():
        def __init__(self, curves, step, k):
            self.step = step
            self.k = k
            if step == 0:
                self.func = lambda t: curves[0][k]
            else:
                self.func = lambda t: (1 - t) * curves[step - 1][k](t) + t * curves[step - 1][k + 1](t)

        def __str__(self):
            pass

        def __call__(self, *args, **kwargs):
            pass

        def __repr__(self):
            pass

    def __init__(self, bs):
        base_case = [self._BezierCurve(bs, 0, k) for k in range(len(bs))]
        self.curves = [base_case]
        self.n = len(bs) - 1

        self._rec_gen_curves()

        self.curve = self.curves[-1][0]

    def _rec_gen_curves(self):
        for step in range(1, self.n + 1):
            self.curves.append(
                [self._BezierCurve(self.curves, step, k) for k in range(len(self.curves[step - 1]) - 1)]
            )


if __name__ == '__main__':
    pass
