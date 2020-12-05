import matplotlib.pyplot as plt


# https://stackoverflow.com/a/7152649/9958281
class DeCasteljau():
    class Inner(object):
        pass

    def __init__(self, bs):
        self.curves = [] # that we have the reference when setting it
        self._BezierCurve = self.set_bezier_class()
        base_case = [self._BezierCurve(0, k, curves=bs) for k in range(len(bs))]
        self.curves.append(base_case)
        self.n = len(bs) - 1

        self._rec_gen_curves()

        self.curve = self.curves[-1][0]

    def set_bezier_class(self):
        parent = self

        class _BezierCurve(DeCasteljau.Inner):
            def __init__(self, step, k, curves=parent.curves):
                self.step = step
                self.k = k
                if step == 0:
                    self.func = lambda t: curves[0][k]
                else:
                    # TODO: Replace .func when __call__ is implemented
                    self.func = lambda t: (1 - t) * curves[step - 1][k].func(t) + t * curves[step - 1][k + 1].func(t)

            def __str__(self):
                pass

            def __call__(self, *args, **kwargs):
                pass

            def __repr__(self):
                pass

        return _BezierCurve

    def _rec_gen_curves(self):
        for step in range(1, self.n + 1):
            self.curves.append(
                [self._BezierCurve(step, k) for k in range(len(self.curves[step - 1]) - 1)]
            )


if __name__ == '__main__':
    DeCasteljau([0.2, 0.4, 0.6, 0.8])
