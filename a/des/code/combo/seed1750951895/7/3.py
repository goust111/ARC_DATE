from __future__ import annotations

import random

import numpy as np


def _decode_layers(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert w == 4 and h >= 2 and h % 2 == 0

    layers: list[tuple[int, int, int, int]] = []
    for r in range(h - 2, -1, -2):
        for c in range(w - 2, -1, -2):
            block = g[r : r + 2, c : c + 2]
            layers.append(
                (
                    int(block[0, 0]),
                    int(block[0, 1]),
                    int(block[1, 0]),
                    int(block[1, 1]),
                )
            )
    assert layers
    return layers


def _corner_fill(colors: list[int], quadrant: int) -> np.ndarray:
    n = len(colors)
    out = np.empty((n, n), dtype=int)
    rr, cc = np.indices((n, n))

    if quadrant == 0:
        depth = np.maximum(rr, cc)
    elif quadrant == 1:
        depth = np.maximum(rr, n - 1 - cc)
    elif quadrant == 2:
        depth = np.maximum(n - 1 - rr, cc)
    else:
        depth = np.maximum(n - 1 - rr, n - 1 - cc)

    palette = np.asarray(colors, dtype=int)
    out[:] = palette[depth]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode 2x2 blocks in reverse row-major order into four nested corner-fill quadrants.
    """
    layers = _decode_layers(grid)
    n = len(layers)

    quadrant_layers = [[layer[q] for layer in layers] for q in range(4)]
    tl = _corner_fill(quadrant_layers[0], quadrant=0)
    tr = _corner_fill(quadrant_layers[1], quadrant=1)
    bl = _corner_fill(quadrant_layers[2], quadrant=2)
    br = _corner_fill(quadrant_layers[3], quadrant=3)

    top = np.concatenate([tl, tr], axis=1)
    bottom = np.concatenate([bl, br], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _random_layers(n: int) -> list[tuple[int, int, int, int]]:
    all_colors = list(range(1, 10))
    mode = random.choice(["independent", "shared", "paired"])
    layers: list[tuple[int, int, int, int]] = []

    if mode == "shared":
        palette_size = random.randint(2, min(6, 9))
        palette = random.sample(all_colors, palette_size)
        for _ in range(n):
            vals = [random.choice(palette) for _ in range(4)]
            layers.append(tuple(vals))
    elif mode == "paired":
        left_palette = random.sample(all_colors, random.randint(2, 4))
        right_candidates = [c for c in all_colors if c not in left_palette]
        right_palette = random.sample(right_candidates, random.randint(2, min(4, len(right_candidates))))
        for _ in range(n):
            layers.append(
                (
                    random.choice(left_palette),
                    random.choice(right_palette),
                    random.choice(left_palette),
                    random.choice(right_palette),
                )
            )
    else:
        for _ in range(n):
            distinct = random.randint(1, 4)
            palette = random.sample(all_colors, distinct)
            vals = [random.choice(palette) for _ in range(4)]
            layers.append(tuple(vals))

    if len({layer for layer in layers}) == 1 and n > 1:
        idx = random.randrange(n)
        q = random.randrange(4)
        replacement = random.choice([c for c in all_colors if c != layers[idx][q]])
        layer = list(layers[idx])
        layer[q] = replacement
        layers[idx] = tuple(layer)

    return layers


def _encode_layers(layers: list[tuple[int, int, int, int]]) -> np.ndarray:
    n = len(layers)
    assert n % 2 == 0 and n >= 2
    grid = np.zeros((n, 4), dtype=int)
    idx = 0
    for r in range(n - 2, -1, -2):
        for c in range(2, -1, -2):
            a, b, c1, d = layers[idx]
            grid[r : r + 2, c : c + 2] = np.array([[a, b], [c1, d]], dtype=int)
            idx += 1
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        n = random.choice([2, 4, 6, 8, 10])
        layers = _random_layers(n)
        grid = _encode_layers(layers)
        out = main(grid)

        assert grid.shape == (n, 4)
        assert out.shape == (2 * n, 2 * n)
        assert np.all(grid > 0)
        assert len(np.unique(out)) >= min(2, n)
        return grid
    raise RuntimeError("failed to generate valid input")
