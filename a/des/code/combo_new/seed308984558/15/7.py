import random

import numpy as np


def _expand(patch: np.ndarray, dirs):
    body = patch.copy()
    for d in dirs:
        if d == "right":
            body = np.hstack([body, np.fliplr(body)])
        elif d == "left":
            body = np.hstack([np.fliplr(body), body])
        elif d == "down":
            body = np.vstack([body, np.flipud(body)])
        else:
            body = np.vstack([np.flipud(body), body])
    return body


def main(grid: np.ndarray) -> np.ndarray:
    """Match each seed color to a legend arrow set and mirror-extend the corresponding left-panel shape."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    sep = int(np.argmax(np.all(g == 9, axis=0)))
    legend = {}
    for color in sorted(set(map(int, g[:, sep + 1 :].ravel())) - {0, 9}):
        rs, cs = np.where(g[:, sep + 1 :] == color)
        rs, cs = rs + 0, cs + sep + 1
        r0, r1, c0, c1 = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())
        dirs = []
        if c1 - c0 > 1:
            dirs.append("right" if np.sum(cs == c1) > np.sum(cs == c0) else "left")
        if r1 - r0 > 1:
            dirs.append("down" if np.sum(rs == r1) > np.sum(rs == r0) else "up")
        legend[color] = dirs
    left = g[:, :sep]
    for color in sorted(set(map(int, left.ravel())) - {0}):
        if color in legend:
            continue
        rs, cs = np.where(left == color)
        if len(rs) < 2:
            continue
        r0, r1, c0, c1 = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())
        patch = left[r0 : r1 + 1, c0 : c1 + 1].copy()
        seed_colors = sorted(set(map(int, left[r0 : r1 + 1, c0 : c1 + 1].ravel())) - {0, color})
        if not seed_colors:
            continue
        seed = seed_colors[0]
        patch[patch == seed] = color
        body = _expand((patch == color).astype(int) * color, legend.get(seed, []))
        out[r0 : r0 + body.shape[0], c0 : c0 + body.shape[1]] = np.maximum(out[r0 : r0 + body.shape[0], c0 : c0 + body.shape[1]], body)
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 24), random.randint(22, 28)
    g = np.zeros((h, w), dtype=int)
    sep = w // 2
    g[:, sep] = 9
    seed_colors = random.sample([1, 2, 3, 4], random.randint(2, 3))
    for i, seed in enumerate(seed_colors):
        r0 = 2 + i * 5
        c0 = sep + 2
        g[r0 : r0 + 2, c0 : c0 + 2] = seed
        if random.choice([True, False]):
            g[r0, c0 + 2 : c0 + 5] = seed
        else:
            g[r0 + 2 : r0 + 5, c0] = seed
    body_colors = random.sample([5, 6, 7, 8], len(seed_colors))
    for seed, body in zip(seed_colors, body_colors):
        r0 = random.randint(2, h - 6)
        c0 = random.randint(1, sep - 5)
        shape = np.array([[body, body, 0], [body, seed, body], [0, body, 0]])
        g[r0 : r0 + 3, c0 : c0 + 3] = np.maximum(g[r0 : r0 + 3, c0 : c0 + 3], shape)
    return g
