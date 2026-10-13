from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
PLACEHOLDER = 5
TIE_COLOR = 0
ROW_PALETTE = (1, 2, 4, 6)
COL_PALETTE = (3, 7, 8, 9)
GEN_RETRIES = 500


def _border_background(grid: np.ndarray) -> int:
    border = np.concatenate((grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]))
    counts = Counter(int(v) for v in border if int(v) != PLACEHOLDER)
    assert counts, "expected at least one non-placeholder border pixel"
    return int(counts.most_common(1)[0][0])


def _extract_keys(
    grid: np.ndarray, bg: int
) -> tuple[dict[int, tuple[str, int]], dict[int, tuple[str, int]]]:
    h, w = grid.shape
    row_keys: dict[int, tuple[str, int]] = {}
    col_keys: dict[int, tuple[str, int]] = {}

    edge_mask = np.zeros_like(grid, dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    colored = (grid != bg) & (grid != PLACEHOLDER)
    assert not np.any(colored & ~edge_mask), "only edge cells may be colored keys"

    for r in range(h):
        found = []
        if int(grid[r, 0]) not in (bg, PLACEHOLDER):
            found.append(("left", int(grid[r, 0])))
        if int(grid[r, w - 1]) not in (bg, PLACEHOLDER):
            found.append(("right", int(grid[r, w - 1])))
        assert len(found) <= 1, "a row may have at most one edge key"
        if found:
            row_keys[r] = found[0]

    for c in range(w):
        found = []
        if int(grid[0, c]) not in (bg, PLACEHOLDER):
            found.append(("top", int(grid[0, c])))
        if int(grid[h - 1, c]) not in (bg, PLACEHOLDER):
            found.append(("bottom", int(grid[h - 1, c])))
        assert len(found) <= 1, "a column may have at most one edge key"
        if found:
            col_keys[c] = found[0]

    return row_keys, col_keys


def _row_distance(side: str, c: int, w: int) -> int:
    return c if side == "left" else w - 1 - c


def _col_distance(side: str, r: int, h: int) -> int:
    return r if side == "top" else h - 1 - r


def _apply_competition(
    grid: np.ndarray,
    *,
    assume_row_side: str | None = None,
    assume_col_side: str | None = None,
    row_first: bool = False,
    col_first: bool = False,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert np.any(g == PLACEHOLDER), "expected grey placeholder pixels"

    bg = _border_background(g)
    row_keys, col_keys = _extract_keys(g, bg)
    h, w = g.shape
    out = g.copy()

    for r_raw, c_raw in np.argwhere(g == PLACEHOLDER):
        r = int(r_raw)
        c = int(c_raw)
        row_key = row_keys.get(r)
        col_key = col_keys.get(c)
        assert row_key or col_key, "every grey pixel must be governed by at least one key"

        if row_key and not col_key:
            out[r, c] = row_key[1]
            continue
        if col_key and not row_key:
            out[r, c] = col_key[1]
            continue

        assert row_key is not None and col_key is not None
        if row_first:
            out[r, c] = row_key[1]
            continue
        if col_first:
            out[r, c] = col_key[1]
            continue

        row_side = assume_row_side or row_key[0]
        col_side = assume_col_side or col_key[0]
        row_d = _row_distance(row_side, c, w)
        col_d = _col_distance(col_side, r, h)

        if row_d < col_d:
            out[r, c] = row_key[1]
        elif col_d < row_d:
            out[r, c] = col_key[1]
        else:
            out[r, c] = TIE_COLOR

    return out


def _analyze_grid(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = _border_background(g)
    row_keys, col_keys = _extract_keys(g, bg)
    h, w = g.shape

    stats = {
        "grey": 0,
        "row_only": 0,
        "col_only": 0,
        "both": 0,
        "row_win": 0,
        "col_win": 0,
        "tie": 0,
        "left_rows": 0,
        "right_rows": 0,
        "top_cols": 0,
        "bottom_cols": 0,
    }

    for side, _color in row_keys.values():
        stats["left_rows" if side == "left" else "right_rows"] += 1
    for side, _color in col_keys.values():
        stats["top_cols" if side == "top" else "bottom_cols"] += 1

    for r_raw, c_raw in np.argwhere(g == PLACEHOLDER):
        r = int(r_raw)
        c = int(c_raw)
        stats["grey"] += 1
        row_key = row_keys.get(r)
        col_key = col_keys.get(c)
        assert row_key or col_key
        if row_key and not col_key:
            stats["row_only"] += 1
            continue
        if col_key and not row_key:
            stats["col_only"] += 1
            continue

        assert row_key is not None and col_key is not None
        stats["both"] += 1
        row_d = _row_distance(row_key[0], c, w)
        col_d = _col_distance(col_key[0], r, h)
        if row_d < col_d:
            stats["row_win"] += 1
        elif col_d < row_d:
            stats["col_win"] += 1
        else:
            stats["tie"] += 1

    return stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey placeholder pixel by the nearer edge key in its row or column, with ties turning black.

    Input structure:
    - The grid has a black background, grey placeholder pixels, and colored control keys on the border.
    - Any row containing grey may have one row key on its left or right edge, and any column containing grey may have one column key on its top or bottom edge.
    - Keys stay on the border and grey pixels stay in the interior.

    Rule:
    - For each grey pixel, compare the row distance to its row key and the column distance to its column key.
    - If only one of those keys exists, paint the pixel with that key's color.
    - If both exist, use the color of the nearer key; if the two distances are equal, paint the pixel black.
    - Keep the background and all key pixels unchanged.

    Key insight:
    - The grey shape is not recolored as a whole: every placeholder pixel independently plays a tug-of-war between its own row key and column key, and the true border side of each key determines the arm length.
    """
    return _apply_competition(grid)


def _choose_consecutive_containing(
    center: int, count: int, lo: int, hi: int, rng: random.Random
) -> list[int]:
    start_lo = max(lo, center - count + 1)
    start_hi = min(center, hi - count + 1)
    assert start_lo <= start_hi
    start = rng.randint(start_lo, start_hi)
    return list(range(start, start + count))


def _choose_subsegment(values: list[int], min_len: int, rng: random.Random) -> list[int]:
    length = rng.randint(min_len, len(values))
    start = rng.randint(0, len(values) - length)
    return values[start : start + length]


def _tie_coord(h: int, w: int, row_side: str, col_side: str, d: int) -> tuple[int, int]:
    r = d if col_side == "top" else h - 1 - d
    c = d if row_side == "left" else w - 1 - d
    return r, c


def _assign_sides(
    indices: list[int], forced_index: int, forced_side: str, options: tuple[str, str], rng: random.Random
) -> dict[int, str]:
    sides = {idx: rng.choice(options) for idx in indices}
    sides[forced_index] = forced_side
    if len(indices) >= 2 and len(set(sides.values())) == 1:
        opposite = options[0] if forced_side == options[1] else options[1]
        for idx in indices:
            if idx != forced_index:
                sides[idx] = opposite
                break
    return sides


def _assign_colors(indices: list[int], palette: tuple[int, ...], rng: random.Random) -> dict[int, int]:
    choices = list(palette)
    rng.shuffle(choices)
    distinct = min(len(indices), min(3, len(choices)))
    base = choices[:distinct]
    shuffled = list(indices)
    rng.shuffle(shuffled)

    colors: dict[int, int] = {}
    for i, idx in enumerate(shuffled):
        if i < len(base):
            colors[idx] = base[i]
        else:
            colors[idx] = rng.choice(base)
    return colors


def _build_mask(
    h: int,
    w: int,
    core_rows: list[int],
    core_cols: list[int],
    row_ext_cols: list[int],
    row_ext_rows: list[int],
    col_ext_rows: list[int],
    col_ext_cols: list[int],
    tie_cell: tuple[int, int],
    rng: random.Random,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[np.ix_(core_rows, core_cols)] = True
    mask[np.ix_(row_ext_rows, row_ext_cols)] = True
    mask[np.ix_(col_ext_rows, col_ext_cols)] = True

    if rng.random() < 0.75:
        max_notch_h = min(2, len(core_rows) - 1)
        max_notch_w = min(2, len(core_cols) - 1)
        if max_notch_h >= 1 and max_notch_w >= 1:
            notch_h = rng.randint(1, max_notch_h)
            notch_w = rng.randint(1, max_notch_w)
            corners = [
                (core_rows[:notch_h], core_cols[:notch_w]),
                (core_rows[:notch_h], core_cols[-notch_w:]),
                (core_rows[-notch_h:], core_cols[:notch_w]),
                (core_rows[-notch_h:], core_cols[-notch_w:]),
            ]
            rng.shuffle(corners)
            for notch_rows, notch_cols in corners:
                cells = {(r, c) for r in notch_rows for c in notch_cols}
                if tie_cell in cells:
                    continue
                for r, c in cells:
                    mask[r, c] = False
                break

    return mask


def _generate_candidate(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(13, 20)
    w = rng.randint(13, 20)
    max_d = min(h - 4, w - 4)
    if max_d < 3:
        return None

    row_side = rng.choice(("left", "right"))
    col_side = rng.choice(("top", "bottom"))
    tie_r, tie_c = _tie_coord(h, w, row_side, col_side, rng.randint(3, max_d))

    core_row_count = rng.randint(3, 5)
    core_col_count = rng.randint(3, 5)
    core_rows = _choose_consecutive_containing(tie_r, core_row_count, 2, h - 3, rng)
    core_cols = _choose_consecutive_containing(tie_c, core_col_count, 2, w - 3, rng)

    if rng.random() < 0.5 and len(core_rows) > 3:
        core_rows = _choose_subsegment(core_rows, 3, rng)
    if rng.random() < 0.5 and len(core_cols) > 3:
        core_cols = _choose_subsegment(core_cols, 3, rng)
    if tie_r not in core_rows or tie_c not in core_cols:
        return None

    left_gap = min(core_cols) - 1
    right_gap = w - 2 - max(core_cols)
    up_gap = min(core_rows) - 1
    down_gap = h - 2 - max(core_rows)
    if min(left_gap, right_gap, up_gap, down_gap) < 1:
        return None

    row_ext_side = rng.choice(("left", "right"))
    row_ext_count = rng.randint(1, min(2, left_gap if row_ext_side == "left" else right_gap))
    if row_ext_side == "left":
        row_ext_cols = list(range(min(core_cols) - row_ext_count, min(core_cols)))
    else:
        row_ext_cols = list(range(max(core_cols) + 1, max(core_cols) + 1 + row_ext_count))
    row_ext_rows = _choose_subsegment(core_rows, 2, rng)

    col_ext_side = rng.choice(("top", "bottom"))
    col_ext_count = rng.randint(1, min(2, up_gap if col_ext_side == "top" else down_gap))
    if col_ext_side == "top":
        col_ext_rows = list(range(min(core_rows) - col_ext_count, min(core_rows)))
    else:
        col_ext_rows = list(range(max(core_rows) + 1, max(core_rows) + 1 + col_ext_count))
    col_ext_cols = _choose_subsegment(core_cols, 2, rng)

    row_sides = _assign_sides(core_rows, tie_r, row_side, ("left", "right"), rng)
    col_sides = _assign_sides(core_cols, tie_c, col_side, ("top", "bottom"), rng)
    row_colors = _assign_colors(core_rows, ROW_PALETTE, rng)
    col_colors = _assign_colors(core_cols, COL_PALETTE, rng)

    tie_cell = (tie_r, tie_c)
    mask = _build_mask(
        h,
        w,
        core_rows,
        core_cols,
        row_ext_cols,
        row_ext_rows,
        col_ext_rows,
        col_ext_cols,
        tie_cell,
        rng,
    )

    if not mask[tie_cell]:
        return None

    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[mask] = PLACEHOLDER

    for r in core_rows:
        key_col = 0 if row_sides[r] == "left" else w - 1
        grid[r, key_col] = row_colors[r]
    for c in core_cols:
        key_row = 0 if col_sides[c] == "top" else h - 1
        grid[key_row, c] = col_colors[c]

    return grid


def _valid_generated_grid(grid: np.ndarray) -> bool:
    if grid is None:
        return False

    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if not np.any(g == PLACEHOLDER):
        return False

    stats = _analyze_grid(g)
    if stats["grey"] < 12:
        return False
    if stats["row_only"] == 0 or stats["col_only"] == 0:
        return False
    if stats["both"] < 6:
        return False
    if stats["row_win"] == 0 or stats["col_win"] == 0 or stats["tie"] == 0:
        return False
    if stats["left_rows"] == 0 or stats["right_rows"] == 0:
        return False
    if stats["top_cols"] == 0 or stats["bottom_cols"] == 0:
        return False

    correct = main(g)
    if np.array_equal(correct, _apply_competition(g, row_first=True)):
        return False
    if np.array_equal(correct, _apply_competition(g, col_first=True)):
        return False
    if np.array_equal(correct, _apply_competition(g, assume_row_side="left", assume_col_side="top")):
        return False
    if np.array_equal(correct, _apply_competition(g, assume_row_side="right", assume_col_side="bottom")):
        return False

    grey_output_colors = set(map(int, correct[g == PLACEHOLDER]))
    if len(grey_output_colors) < 3:
        return False
    if TIE_COLOR not in grey_output_colors:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        seed = random.randrange(1 << 30)
        rng = random.Random(seed)
        grid = _generate_candidate(rng)
        if _valid_generated_grid(grid):
            return grid
    raise RuntimeError("failed to generate a valid tug-of-war grid")
