import random
from collections import Counter

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9
    ALL = list(range(10))


GEN_RETRIES = 400
N_CHOICES = (2, 3, 4)


def _crop_bbox(grid, bbox):
    r0, c0, r1, c1 = [int(x) for x in bbox]
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _cell_region(lines, idx):
    return slice(int(lines[idx]) + 1, int(lines[idx + 1]))


def _cell_bbox(lines, i, j):
    return (int(lines[i]) + 1, int(lines[j]) + 1, int(lines[i + 1]) - 1, int(lines[j + 1]) - 1)


def _pattern_key(arr):
    return tuple(int(v) for v in np.asarray(arr).ravel())


def _infer_separator_grid(grid):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"
    h, w = g.shape

    candidates = []
    for color in np.unique(g):
        color = int(color)
        rows = [r for r in range(h) if np.all(g[r, :] == color)]
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if len(rows) != len(cols) or len(rows) < 3:
            continue
        if rows[0] != 0 or cols[0] != 0 or rows[-1] != h - 1 or cols[-1] != w - 1:
            continue
        row_steps = np.diff(rows)
        col_steps = np.diff(cols)
        if len(row_steps) == 0:
            continue
        if len(set(int(v) for v in row_steps)) != 1 or len(set(int(v) for v in col_steps)) != 1:
            continue
        step = int(row_steps[0])
        if step != int(col_steps[0]):
            continue
        n = len(rows) - 1
        inner = step - 1
        if inner != n:
            continue
        if h != n * step + 1 or w != n * step + 1:
            continue
        candidates.append((color, tuple(rows), tuple(cols), n))

    assert len(candidates) == 1, "expected one separator-grid interpretation"
    return candidates[0]


def _parse_grid(grid):
    g = np.asarray(grid, dtype=int)
    sep, rows, cols, n = _infer_separator_grid(g)

    cells = {}
    for i in range(n):
        for j in range(n):
            bbox = _cell_bbox(rows, i, j)
            cell = _crop_bbox(g, bbox)
            assert cell.shape == (n, n)
            cells[(i, j)] = cell

    pat_counts = Counter()
    pat_arrays = {}
    pat_positions = {}
    for pos, cell in cells.items():
        pat = _pattern_key(cell)
        pat_counts[pat] += 1
        pat_arrays.setdefault(pat, cell.copy())
        pat_positions.setdefault(pat, []).append(pos)

    blank_candidates = [
        pat
        for pat, count in pat_counts.items()
        if count == 1 and len(set(pat)) == 1 and int(pat[0]) != sep
    ]
    assert len(blank_candidates) == 1, "expected exactly one uniform blank cell"
    blank_pat = blank_candidates[0]
    bg = int(blank_pat[0])

    nonblank = [pat for pat in pat_counts if pat != blank_pat]
    assert len(nonblank) == 2, "expected exactly two non-blank cell patterns"
    assert all(bg in pat for pat in nonblank), "background color must also appear inside the repeated patterns"

    majority_pat = max(nonblank, key=lambda pat: pat_counts[pat])
    minority_pat = min(nonblank, key=lambda pat: pat_counts[pat])
    assert pat_counts[majority_pat] > pat_counts[minority_pat] >= 1, "expected strict majority/minority counts"

    minority_arr = pat_arrays[minority_pat]
    non_bg_minority = [int(v) for v in minority_arr.ravel() if int(v) != bg]
    assert non_bg_minority, "minority pattern must contain foreground"
    minority_freq = Counter(non_bg_minority).most_common()
    assert len(minority_freq) == 1 or minority_freq[0][1] > minority_freq[1][1], "minority main color must be unique"
    minority_color = int(minority_freq[0][0])

    return {
        "grid": g,
        "n": int(n),
        "sep": int(sep),
        "bg": int(bg),
        "rows": rows,
        "cols": cols,
        "cells": cells,
        "blank_pos": pat_positions[blank_pat][0],
        "majority_arr": pat_arrays[majority_pat],
        "minority_arr": minority_arr,
        "majority_positions": list(pat_positions[majority_pat]),
        "minority_positions": list(pat_positions[minority_pat]),
    }


def _thumbnail(info):
    n = info["n"]
    thumb = np.full((n, n), int(info["bg"]), dtype=int)
    for r, c in info["minority_positions"]:
        thumb[int(r), int(c)] = int(info["minority_color"] if "minority_color" in info else 0)
    br, bc = info["blank_pos"]
    thumb[int(br), int(bc)] = int(info["sep"])
    return thumb


def _enrich_info(info):
    info = dict(info)
    non_bg_minority = [int(v) for v in info["minority_arr"].ravel() if int(v) != info["bg"]]
    info["minority_color"] = Counter(non_bg_minority).most_common(1)[0][0]
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the unique blank cell with an N×N thumbnail of where the minority cell pattern appears.

    Input structure:
    - A single-color separator grid divides the board into N×N equal cells.
    - Each cell interior is exactly N×N pixels, so the whole board is a self-similar cell lattice.
    - All non-blank cells show one of two repeated interior patterns: a majority pattern and a minority pattern.
    - Exactly one cell interior is entirely background.

    Rule:
    - Read the cell-grid layout itself as an N×N coarse map.
    - Build an N×N thumbnail whose minority-pattern cells become the minority pattern's main color, whose majority-pattern cells stay background, and whose blank cell position becomes the separator color.
    - Paste that thumbnail into the unique blank cell interior and keep every separator line and every other cell unchanged.

    Key insight:
    - The missing cell does not want another copy of a local motif; it wants a downsampled portrait of the whole board, with its own coarse-grid position explicitly marked in separator color.
    """
    info = _enrich_info(_parse_grid(grid))
    out = np.asarray(grid, dtype=int).copy()

    br, bc = info["blank_pos"]
    rs = _cell_region(info["rows"], br)
    cs = _cell_region(info["cols"], bc)
    out[rs, cs] = _thumbnail(info)
    return out


def _mask_library(n):
    rr, cc = np.indices((n, n))
    masks = []
    seen = set()

    def add(mask):
        mask = np.asarray(mask, dtype=bool)
        if not mask.any() or mask.all():
            return
        key = tuple(bool(v) for v in mask.ravel())
        if key not in seen:
            seen.add(key)
            masks.append(mask.copy())

    add(np.eye(n, dtype=bool))
    add(np.fliplr(np.eye(n, dtype=bool)))

    mask = np.zeros((n, n), dtype=bool)
    mask[0, :] = True
    add(mask)

    mask = np.zeros((n, n), dtype=bool)
    mask[-1, :] = True
    add(mask)

    mask = np.zeros((n, n), dtype=bool)
    mask[:, 0] = True
    add(mask)

    mask = np.zeros((n, n), dtype=bool)
    mask[:, -1] = True
    add(mask)

    mask = np.zeros((n, n), dtype=bool)
    mask[0, :] = True
    mask[:, 0] = True
    add(mask)

    mask = np.zeros((n, n), dtype=bool)
    mask[-1, :] = True
    mask[:, -1] = True
    add(mask)

    add((rr + cc) % 2 == 0)
    add(np.eye(n, dtype=bool) | np.fliplr(np.eye(n, dtype=bool)))

    mask = np.zeros((n, n), dtype=bool)
    mask[n // 2, :] = True
    add(mask)

    mask = np.zeros((n, n), dtype=bool)
    mask[:, n // 2] = True
    add(mask)

    if n >= 3:
        mask = np.zeros((n, n), dtype=bool)
        mask[n // 2, :] = True
        mask[:, n // 2] = True
        add(mask)

        mask = np.zeros((n, n), dtype=bool)
        mask[1:-1, 1:-1] = True
        add(mask)

        add((rr == 0) | (rr == n - 1) | (cc == 0) | (cc == n - 1))

    return masks


def _random_transform(arr):
    out = np.asarray(arr).copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    k = random.randrange(4)
    if k:
        out = np.rot90(out, k)
    return out.copy()


def _paint_pattern(mask, bg, primary, accent, role):
    arr = np.full(mask.shape, int(bg), dtype=int)
    arr[mask] = int(primary)

    active = list(zip(*np.where(mask)))
    inactive = list(zip(*np.where(~mask)))
    roll = random.random()

    if role == "minority":
        if accent is not None and len(active) >= 3 and roll < 0.65:
            r, c = random.choice(active)
            arr[int(r), int(c)] = int(accent)
    else:
        if accent is not None and len(active) >= 2 and roll < 0.35:
            r, c = random.choice(active)
            arr[int(r), int(c)] = int(accent)
        elif accent is not None and inactive and roll < 0.7:
            r, c = random.choice(inactive)
            arr[int(r), int(c)] = int(accent)

    return arr


def _cell_bounds_from_n(n, pos):
    step = n + 1
    r0 = int(pos[0]) * step + 1
    c0 = int(pos[1]) * step + 1
    return slice(r0, r0 + n), slice(c0, c0 + n)


def _write_cell(grid, n, pos, patch):
    rs, cs = _cell_bounds_from_n(n, pos)
    grid[rs, cs] = np.asarray(patch, dtype=int)


def _swap_cells(grid, n, a, b):
    out = np.asarray(grid, dtype=int).copy()
    ars, acs = _cell_bounds_from_n(n, a)
    brs, bcs = _cell_bounds_from_n(n, b)
    patch_a = out[ars, acs].copy()
    patch_b = out[brs, bcs].copy()
    out[ars, acs] = patch_b
    out[brs, bcs] = patch_a
    return out


def _semantic_checks(grid):
    info = _enrich_info(_parse_grid(grid))
    out = main(grid)
    thumb = _thumbnail(info)

    br, bc = info["blank_pos"]
    rs = _cell_region(info["rows"], br)
    cs = _cell_region(info["cols"], bc)
    expected = np.asarray(grid, dtype=int).copy()
    expected[rs, cs] = thumb
    assert np.array_equal(out, expected), "output must only replace the blank cell"
    assert not np.array_equal(thumb, info["majority_arr"]), "thumbnail must not collapse to the majority pattern"
    assert not np.array_equal(thumb, info["minority_arr"]), "thumbnail must not collapse to the minority pattern"

    swap_grid = _swap_cells(grid, info["n"], info["majority_positions"][0], info["minority_positions"][0])
    _parse_grid(swap_grid)
    assert not np.array_equal(main(swap_grid), out), "output must depend on global minority placement"

    moved_blank = _swap_cells(grid, info["n"], info["blank_pos"], info["majority_positions"][0])
    moved_info = _enrich_info(_parse_grid(moved_blank))
    assert moved_info["blank_pos"] == info["majority_positions"][0], "blank-cell swap must remain semantically valid"
    assert not np.array_equal(main(moved_blank), out), "output must depend on the blank cell position"


def _gen_attempt():
    n = random.choice(N_CHOICES)
    bg, sep, maj_primary, min_primary, maj_accent, min_accent = random.sample(Color.ALL, 6)

    masks = _mask_library(n)
    maj_mask = _random_transform(random.choice(masks))
    min_mask = _random_transform(random.choice(masks))

    majority = _paint_pattern(maj_mask, bg, maj_primary, maj_accent, role="majority")
    minority = _paint_pattern(min_mask, bg, min_primary, min_accent, role="minority")

    if np.array_equal(majority, minority):
        return None
    if np.all(majority == bg) or np.all(minority == bg):
        return None

    minority_non_bg = [int(v) for v in minority.ravel() if int(v) != bg]
    freq = Counter(minority_non_bg).most_common()
    if not freq or (len(freq) > 1 and freq[0][1] == freq[1][1]):
        return None

    positions = [(i, j) for i in range(n) for j in range(n)]
    blank_pos = random.choice(positions)
    nonblank_positions = [pos for pos in positions if pos != blank_pos]
    max_minority = (len(nonblank_positions) - 1) // 2
    if max_minority < 1:
        return None
    minority_count = random.randint(1, max_minority)
    minority_positions = set(random.sample(nonblank_positions, minority_count))

    grid = np.full((n * (n + 1) + 1, n * (n + 1) + 1), int(sep), dtype=int)
    blank_patch = np.full((n, n), int(bg), dtype=int)
    for pos in positions:
        if pos == blank_pos:
            patch = blank_patch
        elif pos in minority_positions:
            patch = minority
        else:
            patch = majority
        _write_cell(grid, n, pos, patch)

    try:
        info = _enrich_info(_parse_grid(grid))
        thumb = _thumbnail(info)
        if np.array_equal(thumb, majority) or np.array_equal(thumb, minority):
            return None
        _semantic_checks(grid)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid missing-thumbnail input")
