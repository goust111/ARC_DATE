import random
import numpy as np


def _uniform(row):
    return bool(np.all(row == row[0]))


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    sep = int(np.where(np.all(grid == 5, axis=0))[0][0])
    left = grid[:, :sep]
    right = grid[:, sep + 1:]
    h, w = left.shape
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        lu = _uniform(left[r])
        ru = _uniform(right[r])
        if lu and not ru:
            out[r] = right[r]
        elif ru and not lu:
            out[r] = left[r]
        elif lu and ru:
            for c in range(w):
                out[r, c] = left[r, c] if left[r, c] != 0 else right[r, c]
        else:
            out[r] = 0
    return out


def _mixed(width, colors):
    row = np.array([random.choice(colors) for _ in range(width)], dtype=int)
    while np.all(row == row[0]):
        row = np.array([random.choice(colors) for _ in range(width)], dtype=int)
    return row


def generate_input() -> np.ndarray:
    h, w = 10, 11
    p = 5
    g = np.zeros((h, w), dtype=int)
    g[:, p] = 5
    colors = [0, 1, 2, 3, 4, 7]
    cats = ["ln_ru", "lu_rn", "both_u", "both_n"]
    cats += [random.choice(cats) for _ in range(h - len(cats))]
    random.shuffle(cats)
    for r, cat in enumerate(cats):
        if cat == "ln_ru":
            g[r, :p] = _mixed(p, colors)
            g[r, p + 1:] = random.choice(colors)
        elif cat == "lu_rn":
            g[r, :p] = random.choice(colors)
            g[r, p + 1:] = _mixed(p, colors)
        elif cat == "both_u":
            color = random.choice(colors)
            g[r, :p] = color
            g[r, p + 1:] = color if random.random() < 0.5 else 0
        else:
            g[r, :p] = _mixed(p, colors)
            g[r, p + 1:] = _mixed(p, colors)
    return g
