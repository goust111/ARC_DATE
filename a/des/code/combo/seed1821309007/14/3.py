import random
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
GREY = 5
PALETTE = (1, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r] == GREY)]
    assert len(sep_rows) == 2, "expected exactly two grey separator rows"
    assert sep_rows[0] + 1 < sep_rows[1], "separator rows must leave room for the middle panel"

    top = g[: sep_rows[0], :]
    middle = g[sep_rows[0] + 1 : sep_rows[1], :]
    bottom = g[sep_rows[1] + 1 :, :]
    assert top.shape == middle.shape == bottom.shape, "all three panels must have the same shape"
    assert top.size > 0, "panels must be non-empty"
    return top, middle, bottom


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the erased-pixel mask from the example pair and paint it red on the test panel."""
    full, broken, test = _split_panels(grid)

    erased_mask = (full != BLACK) & (broken == BLACK)
    assert np.any(erased_mask), "example pair must reveal a non-empty missing mask"
    assert np.array_equal(broken[~erased_mask], full[~erased_mask]), "broken sample may only erase pixels"

    out = test.copy()
    out[erased_mask] = RED
    return out


def _random_rect_coords(h: int, w: int) -> tuple[int, int, int, int]:
    rh = random.randint(1, max(1, min(4, h)))
    rw = random.randint(1, max(1, min(4, w)))
    r0 = random.randint(0, h - rh)
    c0 = random.randint(0, w - rw)
    return r0, c0, r0 + rh, c0 + rw


def _panel_density(panel: np.ndarray) -> float:
    return float(np.count_nonzero(panel != BLACK)) / float(panel.size)


def _gen_rect_panel(h: int, w: int, palette: tuple[int, ...]) -> np.ndarray:
    panel = np.zeros((h, w), dtype=int)
    n_rects = random.randint(4, 8)
    for _ in range(n_rects):
        r0, c0, r1, c1 = _random_rect_coords(h, w)
        color = random.choice(palette)
        panel[r0:r1, c0:c1] = color
    return panel


def _gen_banded_panel(h: int, w: int, palette: tuple[int, ...]) -> np.ndarray:
    panel = np.zeros((h, w), dtype=int)
    if random.random() < 0.5:
        cuts = sorted(random.sample(range(1, h), k=random.randint(1, min(3, h - 1))))
        starts = [0] + cuts
        ends = cuts + [h]
        for r0, r1 in zip(starts, ends, strict=False):
            panel[r0:r1, :] = random.choice(palette)
        for _ in range(random.randint(2, 4)):
            r0, c0, r1, c1 = _random_rect_coords(h, w)
            panel[r0:r1, c0:c1] = random.choice(palette)
    else:
        cuts = sorted(random.sample(range(1, w), k=random.randint(1, min(3, w - 1))))
        starts = [0] + cuts
        ends = cuts + [w]
        for c0, c1 in zip(starts, ends, strict=False):
            panel[:, c0:c1] = random.choice(palette)
        for _ in range(random.randint(2, 4)):
            r0, c0, r1, c1 = _random_rect_coords(h, w)
            panel[r0:r1, c0:c1] = random.choice(palette)
    if random.random() < 0.5:
        holes = random.randint(1, max(1, (h * w) // 8))
        for _ in range(holes):
            panel[random.randrange(h), random.randrange(w)] = BLACK
    return panel


def _gen_walk_panel(h: int, w: int, palette: tuple[int, ...]) -> np.ndarray:
    panel = np.zeros((h, w), dtype=int)
    target = random.randint(max(6, (h * w) // 4), max(7, (2 * h * w) // 3))
    r = random.randrange(h)
    c = random.randrange(w)
    for step in range(target):
        panel[r, c] = palette[step % len(palette)]
        if random.random() < 0.35:
            panel[r, c] = random.choice(palette)
        dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        r = min(max(r + dr, 0), h - 1)
        c = min(max(c + dc, 0), w - 1)
        if random.random() < 0.2:
            rr0 = max(0, r - 1)
            rr1 = min(h, r + 2)
            cc0 = max(0, c - 1)
            cc1 = min(w, c + 2)
            panel[rr0:rr1, cc0:cc1] = random.choice(palette)
    return panel


def _generate_panel(h: int, w: int) -> np.ndarray:
    palette = tuple(random.sample(PALETTE, k=random.randint(3, min(5, len(PALETTE)))))
    mode = random.choice(("rects", "bands", "walk"))
    if mode == "rects":
        panel = _gen_rect_panel(h, w, palette)
    elif mode == "bands":
        panel = _gen_banded_panel(h, w, palette)
    else:
        panel = _gen_walk_panel(h, w, palette)

    density = _panel_density(panel)
    if not 0.25 <= density <= 0.9:
        return None
    if len(set(map(int, panel[panel != BLACK].ravel()))) < 2:
        return None
    return panel


def _choose_mask(full: np.ndarray) -> np.ndarray | None:
    occupied = np.argwhere(full != BLACK)
    if len(occupied) < 4:
        return None

    mode = random.choice(("subset", "rect", "checker"))
    mask = np.zeros_like(full, dtype=bool)

    if mode == "subset":
        take = random.randint(max(3, len(occupied) // 6), max(3, len(occupied) // 3))
        picks = occupied[random.sample(range(len(occupied)), k=min(take, len(occupied)))]
        mask[picks[:, 0], picks[:, 1]] = True
    elif mode == "rect":
        rows = occupied[:, 0]
        cols = occupied[:, 1]
        r0 = random.randint(int(rows.min()), int(rows.max()))
        r1 = random.randint(r0, int(rows.max()))
        c0 = random.randint(int(cols.min()), int(cols.max()))
        c1 = random.randint(c0, int(cols.max()))
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
        mask &= full != BLACK
    else:
        parity = random.randint(0, 1)
        row_stride = random.choice((2, 3))
        col_stride = random.choice((2, 3))
        for r, c in occupied.tolist():
            if ((r % row_stride) + (c % col_stride)) % 2 == parity:
                mask[r, c] = True

    count = int(mask.sum())
    if count < 3 or count >= int(np.count_nonzero(full != BLACK)) * 3 // 4:
        return None
    return mask


def _ensure_mask_visible(panel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = panel.copy()
    missing = np.argwhere(mask & (out == BLACK))
    if len(missing) == 0:
        return out

    palette = [color for color in PALETTE if color != RED]
    for r, c in missing.tolist():
        out[r, c] = random.choice(palette)
    return out


def _build_case() -> np.ndarray | None:
    h = random.randint(6, 12)
    w = random.randint(6, 12)

    full = _generate_panel(h, w)
    test = _generate_panel(h, w)
    if full is None or test is None:
        return None

    mask = _choose_mask(full)
    if mask is None:
        return None

    broken = full.copy()
    broken[mask] = BLACK
    test = _ensure_mask_visible(test, mask)

    grid = np.full((3 * h + 2, w), BLACK, dtype=int)
    grid[:h, :] = full
    grid[h, :] = GREY
    grid[h + 1 : 2 * h + 1, :] = broken
    grid[2 * h + 1, :] = GREY
    grid[2 * h + 2 :, :] = test

    expected = test.copy()
    expected[mask] = RED
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, test):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
