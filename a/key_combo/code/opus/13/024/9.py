import itertools
from collections import defaultdict, deque

import numpy as np


BG = 0
LINE_COLOR = 1
SPRITE_SIZE = 3
STRIDE = 4
VISIBLE_BANDS = 3
TOTAL_BANDS = 4
LATTICE_HEIGHT = 1 + STRIDE * VISIBLE_BANDS + SPRITE_SIZE
GEN_COLORS = np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_connected(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not coords:
        return True
    seen = {coords[0]}
    q = deque([coords[0]])
    cells = set(coords)
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(coords)


def _shape_library() -> list[np.ndarray]:
    cells = [(r, c) for r in range(SPRITE_SIZE) for c in range(SPRITE_SIZE)]
    library: list[np.ndarray] = []
    for area in (3, 4, 5):
        for subset in itertools.combinations(cells, area):
            if (1, 1) not in subset:
                continue
            mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
            for r, c in subset:
                mask[r, c] = True
            if _is_connected(mask):
                library.append(mask)
    return library


SHAPE_LIBRARY = _shape_library()


def _parse_lattice(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    for n_cols in (3, 4):
        lattice_w = STRIDE * n_cols - 1
        if h < LATTICE_HEIGHT or w < lattice_w:
            continue
        sep_cols = [3 + STRIDE * k for k in range(n_cols - 1)]
        for top in range(h - LATTICE_HEIGHT + 1):
            for left in range(w - lattice_w + 1):
                ok = True
                for sep_r in (0, 4, 8, 12):
                    if not np.all(g[top + sep_r, left : left + lattice_w] == BG):
                        ok = False
                        break
                if not ok:
                    continue
                for sep_c in sep_cols:
                    if not np.all(g[top : top + LATTICE_HEIGHT, left + sep_c] == BG):
                        ok = False
                        break
                if not ok:
                    continue
                if not np.all(g[top + 13 : top + 16, left : left + lattice_w] == BG):
                    continue

                masks: list[list[np.ndarray]] = []
                block_colors: list[list[int | None]] = []
                for band in range(VISIBLE_BANDS):
                    band_masks = []
                    band_colors = []
                    band_has_any = False
                    r0 = top + 1 + STRIDE * band
                    for col in range(n_cols):
                        c0 = left + STRIDE * col
                        block = g[r0 : r0 + SPRITE_SIZE, c0 : c0 + SPRITE_SIZE]
                        colors = {int(v) for v in block.reshape(-1) if int(v) != BG}
                        if len(colors) > 1:
                            ok = False
                            break
                        band_masks.append(block != BG)
                        if colors:
                            band_has_any = True
                            band_colors.append(colors.pop())
                        else:
                            band_colors.append(None)
                    if not ok or not band_has_any:
                        ok = False
                        break
                    masks.append(band_masks)
                    block_colors.append(band_colors)
                if not ok:
                    continue

                col_colors = []
                for col in range(n_cols):
                    seen = {block_colors[band][col] for band in range(VISIBLE_BANDS)}
                    seen.discard(None)
                    if len(seen) != 1:
                        ok = False
                        break
                    col_colors.append(seen.pop())
                if not ok:
                    continue

                for band in range(1, VISIBLE_BANDS):
                    for col in range(n_cols):
                        if not np.array_equal(masks[band][col], masks[band - 1][(col + 1) % n_cols]):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue

                predicted_cols = [
                    col for col in range(n_cols) if masks[0][(col + VISIBLE_BANDS) % n_cols].any()
                ]
                if len(predicted_cols) < 2:
                    continue

                mass = int(np.count_nonzero(g[top : top + LATTICE_HEIGHT, left : left + lattice_w]))
                candidates.append(
                    {
                        "top": top,
                        "left": left,
                        "n_cols": n_cols,
                        "lattice_w": lattice_w,
                        "col_colors": col_colors,
                        "masks": masks,
                        "predicted_cols": predicted_cols,
                        "mass": mass,
                    }
                )

    assert candidates, "could not locate a valid sprite lattice"
    candidates.sort(key=lambda item: item["mass"], reverse=True)
    return candidates[0]


def _paint_predicted_connections(grid: np.ndarray, anchors: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    rows: dict[int, list[int]] = defaultdict(list)
    cols: dict[int, list[int]] = defaultdict(list)
    for r, c in anchors:
        rows[r].append(c)
        cols[c].append(r)

    for r, cols_in_row in rows.items():
        ordered = sorted(cols_in_row)
        for c0, c1 in zip(ordered, ordered[1:]):
            out[r, c0 : c1 + 1] = LINE_COLOR

    for c, rows_in_col in cols.items():
        ordered = sorted(rows_in_col)
        for r0, r1 in zip(ordered, ordered[1:]):
            out[r0 : r1 + 1, c] = LINE_COLOR

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Predict the next cyclic sprite row and draw blue links between its invisible anchor centers.

    Input structure:
    - The grid has a black background and contains one sprite lattice, optionally surrounded by black margins.
    - The lattice has 3 visible sprite bands plus one empty bottom band reserved for the predicted next frame.
    - Sprite slots are 3x3, separated by full black rows and columns; each column keeps one fixed non-black color.
    - Every visible slot contains either a single-color 4-connected 3-5 cell shape or is empty.

    Rule:
    - Read each 3x3 slot as a binary mask, ignoring its color.
    - Verify that each lower visible band is the previous band cyclically shifted one column to the left.
    - Predict which slots in the empty bottom band would be non-empty in the next step.
    - Use the centers of those predicted non-empty slots as anchors, then draw blue horizontal or vertical segments between adjacent anchors sharing a row or a column.
    - Preserve all original pixels; do not draw the predicted sprites themselves.

    Key insight:
    - The blue endpoints are not present in the input at all: they are the centers of sprite slots inferred only after recovering the cyclic shift pattern of the visible lattice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    info = _parse_lattice(g)
    anchor_row = info["top"] + 14
    anchors = [(anchor_row, info["left"] + STRIDE * col + 1) for col in info["predicted_cols"]]
    return _paint_predicted_connections(g, anchors)


def _sample_base_masks(rng: np.random.Generator, n_cols: int) -> list[np.ndarray]:
    nonempty_count = int(rng.integers(2, n_cols + 1))
    chosen_idx = rng.choice(len(SHAPE_LIBRARY), size=nonempty_count, replace=False)
    masks = [SHAPE_LIBRARY[int(idx)].copy() for idx in chosen_idx]
    while len(masks) < n_cols:
        masks.append(np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool))
    rng.shuffle(masks)
    return masks


def _build_input(
    rng: np.random.Generator,
    n_cols: int,
    top: int,
    left: int,
    bottom: int,
    right: int,
    base_masks: list[np.ndarray],
) -> np.ndarray:
    lattice_w = STRIDE * n_cols - 1
    height = top + LATTICE_HEIGHT + bottom
    width = left + lattice_w + right
    grid = np.zeros((height, width), dtype=int)
    col_colors = list(map(int, rng.choice(GEN_COLORS, size=n_cols, replace=False)))

    for band in range(VISIBLE_BANDS):
        r0 = top + 1 + STRIDE * band
        for col in range(n_cols):
            c0 = left + STRIDE * col
            mask = base_masks[(col + band) % n_cols]
            if not mask.any():
                continue
            block = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=int)
            block[mask] = col_colors[col]
            grid[r0 : r0 + SPRITE_SIZE, c0 : c0 + SPRITE_SIZE] = block
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_cols = int(rng.choice([3, 4]))
    lattice_w = STRIDE * n_cols - 1
    top = int(rng.integers(0, 6))
    bottom = int(rng.integers(0, 7))
    left = int(rng.integers(0, 6))
    right = int(rng.integers(0, 7))
    if top + LATTICE_HEIGHT + bottom > 30 or left + lattice_w + right > 30:
        return None

    base_masks = _sample_base_masks(rng, n_cols)
    predicted_nonempty = sum(int(base_masks[(col + VISIBLE_BANDS) % n_cols].any()) for col in range(n_cols))
    if predicted_nonempty < 2:
        return None

    grid = _build_input(rng, n_cols, top, left, bottom, right, base_masks)
    try:
        info = _parse_lattice(grid)
    except AssertionError:
        return None

    if info["top"] != top or info["left"] != left or info["n_cols"] != n_cols:
        return None
    if len(info["predicted_cols"]) != predicted_nonempty:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
