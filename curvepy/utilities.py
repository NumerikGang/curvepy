import numpy as np
import enum


def csv_read(file_path: str) -> np.ndarray:
    try:
        with open(file_path, 'r') as csv_file:
            xs, ys, zs = [], [], []
            for line in csv_file:
                try:
                    x, y, z = line.split(',')
                    zs.append(float(z))
                except ValueError:
                    try:
                        x, y = line.split(',')
                    except ValueError:
                        print('Expected two or three values per line')
                        return np.array([])
                xs.append(float(x))
                ys.append(float(y))
        return np.array([xs, ys], dtype=float) if not zs else np.array([xs, ys, zs], dtype=float)
    except FileNotFoundError:
        print(f'File: {file_path} does not exist.')
        return np.array([])
