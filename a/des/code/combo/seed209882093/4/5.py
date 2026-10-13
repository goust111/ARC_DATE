from __future__ import annotations

import numpy as np


BG = 0
GEN_RETRIES = 500
GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (18, 29)
N_COLORS_RANGE = (2, 5)
CLUSTER_PROB = 0.6


BASE_SHAPES = (
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
)


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(int(r), int(c))]
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _bbox_center(coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    return (r0 + r1) // 2, (c0 + c1) // 2


def _foreground_colors(grid: np.ndarray, bg: int = BG) -> list[int]:
    return [int(color) for color in np.unique(grid) if int(color) != bg]


def _first_appearance_order(grid: np.ndarray, bg: int = BG) -> list[int]:
    seen: dict[int, int] = {}
    flat = np.asarray(grid, dtype=int).ravel()
    for idx, value in enumerate(flat):
        color = int(value)
        if color != bg and color not in seen:
            seen[color] = int(idx)
    return [color for color, _ in sorted(seen.items(), key=lambda item: item[1])]


def _extract_color_parts(grid: np.ndarray, color: int) -> tuple[int, int, np.ndarray]:
    mask = np.asarray(grid == color, dtype=bool)
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)

    guide_row = int(np.argmax(row_counts))
    guide_col = int(np.argmax(col_counts))

    assert row_counts[guide_row] > 0
    assert col_counts[guide_col] > 0
    assert int(np.count_nonzero(row_counts == row_counts[guide_row])) == 1
    assert int(np.count_nonzero(col_counts == col_counts[guide_col])) == 1

    stamp_mask = mask.copy()
    stamp_mask[guide_row, :] = False
    stamp_mask[:, guide_col] = False
    comps = _components4(stamp_mask)
    assert len(comps) == 1
    stamp = comps[0]
    r0, c0, r1, c1 = _bbox(stamp)
    assert stamp.shape[0] >= 2
    assert r1 - r0 + 1 <= 4
    assert c1 - c0 + 1 <= 4
    return guide_row, guide_col, stamp


def _translated_stamp(coords: np.ndarray, target_r: int, target_c: int) -> np.ndarray:
    center_r, center_c = _bbox_center(coords)
    shift = np.array([target_r - center_r, target_c - center_c], dtype=int)
    return np.asarray(coords, dtype=int) + shift


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the guides and restamp each monochrome sprite at the intersection of
    its same-color horizontal and vertical guides.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    colors = _foreground_colors(g)
    assert colors
    order = _first_appearance_order(g)
    assert set(order) == set(colors)

    parts: dict[int, tuple[int, int, np.ndarray]] = {}
    for color in colors:
        guide_row, guide_col, stamp = _extract_color_parts(g, color)
        parts[color] = (guide_row, guide_col, stamp)

    out = np.full((h, w), BG, dtype=int)
    for color in order:
        guide_row, guide_col, stamp = parts[color]
        moved = _translated_stamp(stamp, guide_row, guide_col)
        assert bool(np.all((0 <= moved[:, 0]) & (moved[:, 0] < h)))
        assert bool(np.all((0 <= moved[:, 1]) & (moved[:, 1] < w)))
        out[moved[:, 0], moved[:, 1]] = int(color)
    return out


def _random_shape(rng: np.random.Generator) -> np.ndarray:
    patch = np.array(BASE_SHAPES[int(rng.integers(0, len(BASE_SHAPES)))], copy=True)
    patch = np.rot90(patch, int(rng.integers(0, 4)))
    if bool(rng.integers(0, 2)):
        patch = np.fliplr(patch)
    if bool(rng.integers(0, 2)):
        patch = np.flipud(patch)
    return np.asarray(patch, dtype=bool)


def _shape_coords(patch: np.ndarray) -> np.ndarray:
    return np.argwhere(np.asarray(patch, dtype=bool))


def _sample_axis_positions(limit: int, n: int, rng: np.random.Generator) -> list[int] | None:
    pool = list(range(1, limit - 1))
    if len(pool) < n:
        return None

    if n >= 2 and rng.random() < CLUSTER_PROB:
        close_pairs = [(a, b) for a in pool for b in pool if a < b and b - a in (2, 3)]
        if close_pairs:
            a, b = close_pairs[int(rng.integers(0, len(close_pairs)))]
            chosen = [a, b]
            if n > 2:
                remaining = [value for value in pool if value not in (a, b)]
                extra = rng.choice(np.asarray(remaining, dtype=int), size=n - 2, replace=False)
                chosen.extend(int(value) for value in extra.tolist())
            rng.shuffle(chosen)
            return [int(value) for value in chosen]

    picked = rng.choice(np.asarray(pool, dtype=int), size=n, replace=False)
    return [int(value) for value in picked.tolist()]


def _coords_clear_of_guides(coords: np.ndarray, rows: list[int], cols: list[int]) -> bool:
    row_set = set(int(v) for v in rows)
    col_set = set(int(v) for v in cols)
    for r, c in np.asarray(coords, dtype=int):
        if any(abs(int(r) - row) <= 1 for row in row_set):
            return False
        if any(abs(int(c) - col) <= 1 for col in col_set):
            return False
    return True


def _place_stamps(
    h: int,
    w: int,
    shapes: list[np.ndarray],
    guide_rows: list[int],
    guide_cols: list[int],
    rng: np.random.Generator,
) -> list[np.ndarray] | None:
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[np.ndarray | None] = [None] * len(shapes)
    order = sorted(
        range(len(shapes)),
        key=lambda idx: int(np.count_nonzero(shapes[idx])),
        reverse=True,
    )

    for idx in order:
        patch = np.asarray(shapes[idx], dtype=bool)
        coords0 = _shape_coords(patch)
        ph, pw = patch.shape
        candidates: list[np.ndarray] = []

        for r0 in range(h - ph + 1):
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + ph + 1)
            for c0 in range(w - pw + 1):
                cc0 = max(0, c0 - 1)
                cc1 = min(w, c0 + pw + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                coords = coords0 + np.array([r0, c0], dtype=int)
                if not _coords_clear_of_guides(coords, guide_rows, guide_cols):
                    continue
                candidates.append(coords)

        if not candidates:
            return None
        pick = candidates[int(rng.integers(0, len(candidates)))]
        placements[idx] = pick
        occupied[pick[:, 0], pick[:, 1]] = True

    return [np.asarray(coords, dtype=int) for coords in placements if coords is not None]


def _draw_guides(
    h: int,
    w: int,
    colors: list[int],
    guide_rows: list[int],
    guide_cols: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    layers: list[tuple[str, int, int]] = []
    for color, row, col in zip(colors, guide_rows, guide_cols):
        layers.append(("h", int(color), int(row)))
        layers.append(("v", int(color), int(col)))
    rng.shuffle(layers)

    for kind, color, pos in layers:
        if kind == "h":
            grid[pos, :] = int(color)
        else:
            grid[:, pos] = int(color)
    return grid


def _build_expected(
    shape_coords: dict[int, np.ndarray],
    guide_rows: dict[int, int],
    guide_cols: dict[int, int],
    order: list[int],
    shape: tuple[int, int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for color in order:
        moved = _translated_stamp(shape_coords[color], guide_rows[color], guide_cols[color])
        out[moved[:, 0], moved[:, 1]] = int(color)
    return out


def _visible_signature_ok(
    grid: np.ndarray,
    color: int,
    guide_row: int,
    guide_col: int,
    expected_stamp: np.ndarray,
) -> bool:
    mask = grid == int(color)
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)

    if int(row_counts[guide_row]) <= 4 or int(col_counts[guide_col]) <= 4:
        return False
    if int(row_counts[guide_row]) != int(row_counts.max()):
        return False
    if int(col_counts[guide_col]) != int(col_counts.max()):
        return False
    if int(np.count_nonzero(row_counts == row_counts[guide_row])) != 1:
        return False
    if int(np.count_nonzero(col_counts == col_counts[guide_col])) != 1:
        return False

    stamp_mask = mask.copy()
    stamp_mask[guide_row, :] = False
    stamp_mask[:, guide_col] = False
    comps = _components4(stamp_mask)
    if len(comps) != 1:
        return False

    got = {tuple(map(int, cell)) for cell in comps[0]}
    want = {tuple(map(int, cell)) for cell in np.asarray(expected_stamp, dtype=int)}
    return got == want


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1]))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1]))
    n_colors = int(rng.integers(N_COLORS_RANGE[0], N_COLORS_RANGE[1]))
    colors = [int(value) for value in rng.choice(np.arange(1, 10), size=n_colors, replace=False).tolist()]

    guide_rows = _sample_axis_positions(h, n_colors, rng)
    guide_cols = _sample_axis_positions(w, n_colors, rng)
    if guide_rows is None or guide_cols is None:
        return None

    shapes = [_random_shape(rng) for _ in range(n_colors)]
    placements = _place_stamps(h, w, shapes, guide_rows, guide_cols, rng)
    if placements is None or len(placements) != n_colors:
        return None

    grid = _draw_guides(h, w, colors, guide_rows, guide_cols, rng)
    shape_coords: dict[int, np.ndarray] = {}
    row_map: dict[int, int] = {}
    col_map: dict[int, int] = {}

    for color, row, col, coords in zip(colors, guide_rows, guide_cols, placements):
        grid[coords[:, 0], coords[:, 1]] = int(color)
        shape_coords[int(color)] = np.asarray(coords, dtype=int)
        row_map[int(color)] = int(row)
        col_map[int(color)] = int(col)

    for color in colors:
        if not _visible_signature_ok(grid, color, row_map[color], col_map[color], shape_coords[color]):
            return None

    order = _first_appearance_order(grid)
    if set(order) != set(colors):
        return None

    expected = _build_expected(shape_coords, row_map, col_map, order, grid.shape)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
