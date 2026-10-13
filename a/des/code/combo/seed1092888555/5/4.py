import random
from typing import List, Tuple

import numpy as np


BG = 0
GREEN = 3
MAROON = 9
PALETTE = (1, 2, 4, 5, 6, 7, 8)
AXIAL_SIZES = (4, 6, 8, 9, 12)
ORTH_SIZES = (3,)
MAX_GEN_ATTEMPTS = 400


def _find_separator_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    for color in colors:
        full_rows = int(np.sum(np.all(grid == color, axis=1)))
        full_cols = int(np.sum(np.all(grid == color, axis=0)))
        if full_rows >= 2 and full_cols >= 2:
            return color
    raise AssertionError("separator color not found")


def _panel_spans(sep_ids: np.ndarray) -> List[Tuple[int, int]]:
    spans = []
    for a, b in zip(sep_ids[:-1], sep_ids[1:]):
        if int(b) - int(a) > 1:
            spans.append((int(a) + 1, int(b) - 1))
    return spans


def _swap_colors(arr: np.ndarray, a: int, b: int) -> np.ndarray:
    out = arr.copy()
    out[arr == a] = b
    out[arr == b] = a
    return out


def _fill_panel(sub: np.ndarray, direction: str) -> np.ndarray:
    out = sub.copy()
    out[out == GREEN] = BG

    seed_mask = (out != BG)
    coords = np.argwhere(seed_mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    seed = out[r0 : r1 + 1, c0 : c1 + 1].copy()
    colors = [int(v) for v in np.unique(seed) if int(v) != BG]
    assert len(colors) == 2
    a, b = colors

    ph, pw = out.shape
    sh, sw = seed.shape

    if direction == "down":
        assert r0 == 0 and ph % sh == 0
        for k, rr in enumerate(range(sh, ph, sh), start=1):
            patch = _swap_colors(seed, a, b) if k % 2 == 1 else seed
            out[rr : rr + sh, c0 : c0 + sw] = patch
    elif direction == "up":
        assert r1 == ph - 1 and ph % sh == 0
        for k, rr in enumerate(range(ph - 2 * sh, -1, -sh), start=1):
            patch = _swap_colors(seed, a, b) if k % 2 == 1 else seed
            out[rr : rr + sh, c0 : c0 + sw] = patch
    elif direction == "right":
        assert c0 == 0 and pw % sw == 0
        for k, cc in enumerate(range(sw, pw, sw), start=1):
            patch = _swap_colors(seed, a, b) if k % 2 == 1 else seed
            out[r0 : r0 + sh, cc : cc + sw] = patch
    elif direction == "left":
        assert c1 == pw - 1 and pw % sw == 0
        for k, cc in enumerate(range(pw - 2 * sw, -1, -sw), start=1):
            patch = _swap_colors(seed, a, b) if k % 2 == 1 else seed
            out[r0 : r0 + sh, cc : cc + sw] = patch
    else:
        raise AssertionError(direction)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = BG
    sep = _find_separator_color(g, bg)

    sep_rows = np.flatnonzero(np.all(g == sep, axis=1))
    sep_cols = np.flatnonzero(np.all(g == sep, axis=0))
    row_spans = _panel_spans(sep_rows)
    col_spans = _panel_spans(sep_cols)
    assert row_spans and col_spans

    heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(heights) == 1 and len(widths) == 1

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            marker_pts = np.argwhere(sub == GREEN)
            assert marker_pts.shape[0] == 1
            mr, mc = map(int, marker_pts[0])
            if mr == 0:
                direction = "down"
            elif mr == sub.shape[0] - 1:
                direction = "up"
            elif mc == 0:
                direction = "right"
            elif mc == sub.shape[1] - 1:
                direction = "left"
            else:
                raise AssertionError("marker must sit on a panel edge")
            out[r0 : r1 + 1, c0 : c1 + 1] = _fill_panel(sub, direction)
    return out


def _generate_seed(shape: Tuple[int, int], direction: str, a: int, b: int) -> np.ndarray:
    h, w = shape
    start_side = {
        "down": [(0, c) for c in range(1, w - 1)],
        "up": [(h - 1, c) for c in range(1, w - 1)],
        "right": [(r, 0) for r in range(1, h - 1)],
        "left": [(r, w - 1) for r in range(1, h - 1)],
    }[direction]

    for _ in range(200):
        seed = np.random.choice([BG, a, b], size=shape, p=[0.28, 0.36, 0.36]).astype(int)
        for r in range(h):
            if not np.any(seed[r] != BG):
                seed[r, random.randrange(w)] = random.choice((a, b))
        for c in range(w):
            if not np.any(seed[:, c] != BG):
                seed[random.randrange(h), c] = random.choice((a, b))
        if a not in seed or b not in seed:
            continue
        edge_holes = [(r, c) for r, c in start_side if seed[r, c] == BG]
        if not edge_holes:
            continue
        if np.count_nonzero(seed != BG) < max(3, (h * w) // 2):
            continue
        return seed
    raise RuntimeError("failed to build seed")


def _stamp_seed(panel: np.ndarray, seed: np.ndarray, direction: str) -> None:
    ph, pw = panel.shape
    sh, sw = seed.shape
    if direction == "down":
        panel[:sh, :sw] = seed
    elif direction == "up":
        panel[ph - sh :, :sw] = seed
    elif direction == "right":
        panel[:sh, :sw] = seed
    elif direction == "left":
        panel[:sh, pw - sw :] = seed
    else:
        raise AssertionError(direction)


def _place_marker(panel: np.ndarray, seed: np.ndarray, direction: str) -> None:
    ph, pw = panel.shape
    sh, sw = seed.shape
    if direction == "down":
        holes = [c for c in range(sw) if seed[0, c] == BG]
        panel[0, random.choice(holes)] = GREEN
    elif direction == "up":
        holes = [c for c in range(sw) if seed[-1, c] == BG]
        panel[ph - 1, random.choice(holes)] = GREEN
    elif direction == "right":
        holes = [r for r in range(sh) if seed[r, 0] == BG]
        panel[random.choice(holes), 0] = GREEN
    elif direction == "left":
        holes = [r for r in range(sh) if seed[r, -1] == BG]
        panel[random.choice(holes), pw - 1] = GREEN
    else:
        raise AssertionError(direction)


def _build_example() -> Tuple[np.ndarray, np.ndarray]:
    orientation = random.choice(("horizontal", "vertical"))

    if orientation == "horizontal":
        panel_h = random.choice(ORTH_SIZES)
        panel_w = random.choice(AXIAL_SIZES)
        valid_seed_ws = [d for d in (2, 3) if panel_w % d == 0 and panel_w // d >= 2]
        if not valid_seed_ws:
            raise RuntimeError("no valid horizontal seed width")
        dir_choices = ("right", "left")
    else:
        panel_w = random.choice(ORTH_SIZES)
        panel_h = random.choice(AXIAL_SIZES)
        valid_seed_hs = [d for d in (2, 3) if panel_h % d == 0 and panel_h // d >= 2]
        if not valid_seed_hs:
            raise RuntimeError("no valid vertical seed height")
        dir_choices = ("down", "up")

    layouts = []
    for nr in range(1, 5):
        for nc in range(1, 5):
            total_h = nr * panel_h + nr + 1
            total_w = nc * panel_w + nc + 1
            if 8 <= total_h <= 30 and 8 <= total_w <= 30 and 2 <= nr * nc <= 8:
                layouts.append((nr, nc))
    if not layouts:
        raise RuntimeError("no valid panel layout")
    n_rows, n_cols = random.choice(layouts)

    grid_h = n_rows * panel_h + n_rows + 1
    grid_w = n_cols * panel_w + n_cols + 1
    inp = np.full((grid_h, grid_w), BG, dtype=int)
    out = np.full((grid_h, grid_w), BG, dtype=int)

    for r in range(0, grid_h, panel_h + 1):
        inp[r, :] = MAROON
        out[r, :] = MAROON
    for c in range(0, grid_w, panel_w + 1):
        inp[:, c] = MAROON
        out[:, c] = MAROON

    for pr in range(n_rows):
        for pc in range(n_cols):
            r0 = 1 + pr * (panel_h + 1)
            c0 = 1 + pc * (panel_w + 1)
            panel_in = inp[r0 : r0 + panel_h, c0 : c0 + panel_w]
            direction = random.choice(dir_choices)
            a, b = random.sample(PALETTE, 2)

            if orientation == "horizontal":
                seed_w = random.choice(valid_seed_ws)
                seed_shape = (panel_h, seed_w)
            else:
                seed_h = random.choice(valid_seed_hs)
                seed_shape = (seed_h, panel_w)

            seed = _generate_seed(seed_shape, direction, a, b)
            _stamp_seed(panel_in, seed, direction)
            _place_marker(panel_in, seed, direction)
            out[r0 : r0 + panel_h, c0 : c0 + panel_w] = _fill_panel(panel_in, direction)

    solved = main(inp)
    assert np.array_equal(solved, out)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            inp, _out = _build_example()
        except Exception:
            continue
        return inp
    raise RuntimeError("failed to generate valid input")
