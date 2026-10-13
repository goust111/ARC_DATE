import itertools
import random

import numpy as np


BG = 0
SEP = 5


def _split_panels(g: np.ndarray):
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEP)]
    assert len(sep_cols) == 4, "expected four separator columns between five panels"
    starts = [0] + [c + 1 for c in sep_cols]
    ends = sep_cols + [g.shape[1]]
    panels = [g[:, a:b].copy() for a, b in zip(starts, ends)]
    shapes = {p.shape for p in panels}
    assert len(shapes) == 1
    return panels


def _compose(masks, order):
    out = np.zeros_like(next(iter(masks.values())), dtype=int)
    for color in order:
        mask = masks[color]
        out[mask] = color
    return out


def _solve_order(masks, composite):
    colors = list(masks)
    winners = []
    for order in itertools.permutations(colors):
        if np.array_equal(_compose(masks, order), composite):
            winners.append(order)
    assert len(winners) == 1, "layer evidence must imply one order"
    return winners[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the four-mask layer order from the fifth composite panel, including
    the fully hidden mask, then re-compose with that order reversed.
    """
    g = np.asarray(grid, dtype=int)
    panels = _split_panels(g)
    masks = {}
    for p in panels[:4]:
        vals = set(map(int, np.unique(p))) - {BG}
        assert len(vals) == 1
        color = vals.pop()
        masks[color] = p == color
    order = _solve_order(masks, panels[4])
    return _compose(masks, tuple(reversed(order)))


def _random_mask(h, w, density):
    return np.random.random((h, w)) < density


def _gen_attempt():
    h = random.randint(5, 7)
    w = random.randint(4, 5)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 4)
    order = tuple(random.sample(colors, 4))
    hidden = order[0]
    masks = {}
    for color in colors:
        masks[color] = _random_mask(h, w, random.uniform(0.28, 0.45))
    upper_union = np.zeros((h, w), dtype=bool)
    for color in order[1:]:
        upper_union |= masks[color]
    if int(upper_union.sum()) < 5:
        return None
    masks[hidden] = upper_union & _random_mask(h, w, random.uniform(0.45, 0.75))
    if not masks[hidden].any():
        masks[hidden][tuple(np.argwhere(upper_union)[0])] = True

    composite = _compose(masks, order)
    if hidden in set(map(int, np.unique(composite))):
        return None
    try:
        solved = _solve_order(masks, composite)
    except Exception:
        return None
    if solved != order:
        return None

    panels = []
    for color in colors:
        p = np.zeros((h, w), dtype=int)
        p[masks[color]] = color
        panels.append(p)
    panels.append(composite)
    sep = np.full((h, 1), SEP, dtype=int)
    grid = panels[0]
    for p in panels[1:]:
        grid = np.concatenate([grid, sep, p], axis=1)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(1000):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
