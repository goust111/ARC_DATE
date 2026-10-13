from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS) not in sys.path:
    sys.path.append(str(_UTILS))

from arc_utils import crop_bbox, find_components


BG = 0
GREY = 5
NON_GREY = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400

HOLE_SHAPES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
]


def _pair_candidate(grid: np.ndarray, a: int, b: int) -> dict | None:
    mask = (grid == int(a)) | (grid == int(b))
    if not mask.any() or not np.any(grid == int(a)) or not np.any(grid == int(b)):
        return None
    for color in (a, b):
        color_rows = np.flatnonzero(np.any(grid == int(color), axis=1))
        color_cols = np.flatnonzero(np.any(grid == int(color), axis=0))
        if len(color_rows) < 2 or len(color_cols) < 2:
            return None

    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    sub = grid[np.ix_(rows, cols)]
    allowed = {BG, int(a), int(b)}
    if not set(map(int, np.unique(sub))) <= allowed:
        return None
    if not np.any(sub == BG):
        return None
    return {
        "colors": tuple(sorted((int(a), int(b)))),
        "rows": tuple(map(int, rows)),
        "cols": tuple(map(int, cols)),
        "subgrid": sub.copy(),
    }


def _find_target(grid: np.ndarray) -> dict:
    colors = [int(c) for c in np.unique(grid) if int(c) != BG]
    candidates = []
    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            cand = _pair_candidate(grid, a, b)
            if cand is not None:
                candidates.append(cand)
    assert len(candidates) == 1
    return candidates[0]


def _legend_fill(grid: np.ndarray, rows: tuple[int, ...], cols: tuple[int, ...]) -> int:
    h, w = grid.shape
    row_mask = np.zeros(h, dtype=bool)
    col_mask = np.zeros(w, dtype=bool)
    row_mask[list(rows)] = True
    col_mask[list(cols)] = True
    remainder_mask = (~row_mask)[:, None] | (~col_mask)[None, :]
    remainder = np.where(remainder_mask, grid, BG)

    candidates = []
    for comp in find_components(
        remainder,
        bg=BG,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    ):
        r0, c0, r1, c1 = comp["bbox"]
        crop = crop_bbox(remainder, comp["bbox"])
        hh, ww = crop.shape
        if int(comp["area"]) != hh * ww:
            continue
        vals = sorted(map(int, np.unique(crop)))
        if len(vals) != 2 or GREY not in vals:
            continue
        fill = vals[0] if vals[1] == GREY else vals[1]
        if fill == GREY:
            continue
        if hh == 2 and ww >= 3 and np.all(~row_mask[r0 : r1 + 1]):
            if len(set(map(int, crop[0]))) == 1 and len(set(map(int, crop[1]))) == 1:
                row_colors = {int(crop[0, 0]), int(crop[1, 0])}
                if row_colors == {GREY, fill}:
                    candidates.append(fill)
        if ww == 2 and hh >= 3 and np.all(~col_mask[c0 : c1 + 1]):
            if len(set(map(int, crop[:, 0]))) == 1 and len(set(map(int, crop[:, 1]))) == 1:
                col_colors = {int(crop[0, 0]), int(crop[0, 1])}
                if col_colors == {GREY, fill}:
                    candidates.append(fill)
    assert len(candidates) == 1
    return int(candidates[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the unique complete bicolor row-column product, recover the hidden legend color from discarded rows or columns, and fill the extracted holes.

    Input structure:
    - The background is black.
    - Exactly one pair of non-black colors forms a bicolor image on some active row set and active column set; within their row-column product, every cell is either one of those two colors or a black hole.
    - Outside that active row-column product, there is exactly one full 2xN or Nx2 legend rectangle made of one grey stripe and one stripe of a single non-grey color.
    - Additional non-black distractor pixels may appear elsewhere but do not satisfy the same row-column-product condition.

    Rule:
    - Find the unique color pair whose occupied row set and occupied column set define a product region containing only those two colors plus at least one black hole.
    - Extract the subgrid obtained by keeping exactly those active rows and active columns, in their original order.
    - In the cells not belonging to both active rows and active columns, locate the unique 2xN or Nx2 grey/non-grey legend and read its non-grey stripe color.
    - Recolor every black hole in the extracted subgrid with that legend color.

    Key insight:
    The legend is only identifiable after the correct active rows and columns are known, because it is hidden specifically in the rows or columns discarded by the extraction step.
    """
    g = np.asarray(grid, dtype=int)
    target = _find_target(g)
    fill = _legend_fill(g, target["rows"], target["cols"])
    out = target["subgrid"].copy()
    out[out == BG] = fill
    return out


def _run_bits(length: int) -> np.ndarray:
    n_runs = random.randint(2, min(4, length))
    cuts = sorted(random.sample(range(1, length), n_runs - 1))
    bits = np.zeros(length, dtype=int)
    start = 0
    bit = random.randint(0, 1)
    for end in cuts + [length]:
        bits[start:end] = bit
        bit ^= 1
        start = end
    return bits


def _bicolor_pattern(h: int, w: int, colors: tuple[int, int]) -> np.ndarray:
    a, b = map(int, colors)
    for _ in range(80):
        mode = random.choice(["xor", "rows", "cols", "checker"])
        if mode == "xor":
            rows = _run_bits(h)[:, None]
            cols = _run_bits(w)[None, :]
            bits = rows ^ cols ^ random.randint(0, 1)
        elif mode == "rows":
            bits = np.repeat(_run_bits(h)[:, None], w, axis=1)
        elif mode == "cols":
            bits = np.repeat(_run_bits(w)[None, :], h, axis=0)
        else:
            rr, cc = np.indices((h, w))
            bits = (rr + cc + random.randint(0, 1)) % 2
        pattern = np.where(bits == 0, a, b).astype(int)
        ok = True
        for color in (a, b):
            color_rows = np.flatnonzero(np.any(pattern == color, axis=1))
            color_cols = np.flatnonzero(np.any(pattern == color, axis=0))
            if len(color_rows) < 2 or len(color_cols) < 2:
                ok = False
                break
        if ok:
            return pattern
    raise RuntimeError("failed to build a valid bicolor pattern")


def _place_holes(h: int, w: int) -> np.ndarray | None:
    hole_mask = np.zeros((h, w), dtype=bool)
    target_area = random.randint(3, min(8, (h - 2) * (w - 2)))
    attempts = 0
    while hole_mask.sum() < target_area and attempts < 200:
        attempts += 1
        shape = random.choice(HOLE_SHAPES)
        sh, sw = shape.shape
        if sh > h - 2 or sw > w - 2:
            continue
        max_r = h - sh - 1
        max_c = w - sw - 1
        r0 = random.randint(1, max_r)
        c0 = random.randint(1, max_c)
        block = hole_mask[r0 : r0 + sh, c0 : c0 + sw]
        if np.any(block & shape):
            continue
        expanded = hole_mask[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)]
        exp_shape = np.pad(shape, 1, constant_values=False)
        er, ec = expanded.shape
        if np.any(expanded & exp_shape[:er, :ec]):
            continue
        add_area = int(shape.sum())
        if int(hole_mask.sum()) + add_area > target_area:
            continue
        hole_mask[r0 : r0 + sh, c0 : c0 + sw] |= shape
    if not (3 <= int(hole_mask.sum()) <= 8):
        return None
    if np.any(hole_mask[0]) or np.any(hole_mask[-1]) or np.any(hole_mask[:, 0]) or np.any(hole_mask[:, -1]):
        return None
    return hole_mask


def _legend_patch(fill: int, horizontal: bool) -> np.ndarray:
    if horizontal:
        width = random.randint(3, 6)
        top_fill = random.choice([True, False])
        rows = [fill, GREY] if top_fill else [GREY, fill]
        return np.vstack(
            [
                np.full(width, rows[0], dtype=int),
                np.full(width, rows[1], dtype=int),
            ]
        )
    height = random.randint(3, 6)
    left_fill = random.choice([True, False])
    cols = [fill, GREY] if left_fill else [GREY, fill]
    return np.column_stack(
        [
            np.full(height, cols[0], dtype=int),
            np.full(height, cols[1], dtype=int),
        ]
    )


def _mixed_patch(colors: list[int]) -> np.ndarray:
    h = random.choice([3, 4])
    w = random.choice([3, 4])
    patch = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            patch[r, c] = colors[(r + 2 * c) % len(colors)]
    return patch


def _can_place(mask: np.ndarray, top: int, left: int, patch: np.ndarray, margin: int = 1) -> bool:
    h, w = patch.shape
    hh, ww = mask.shape
    if top < 0 or left < 0 or top + h > hh or left + w > ww:
        return False
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(hh, top + h + margin)
    c1 = min(ww, left + w + margin)
    return not mask[r0:r1, c0:c1].any()


def _reserve(mask: np.ndarray, top: int, left: int, patch: np.ndarray) -> None:
    h, w = patch.shape
    mask[top : top + h, left : left + w] = True


def _build_case() -> tuple[np.ndarray, np.ndarray, dict] | None:
    out_h = random.randint(5, 8)
    out_w = random.randint(5, 8)
    span_h = out_h + 2
    span_w = out_w + 2
    grid_h = random.randint(max(14, span_h + 4), min(22, 30))
    grid_w = random.randint(max(14, span_w + 4), min(22, 30))

    top = random.randint(0, grid_h - span_h)
    left = random.randint(0, grid_w - span_w)
    gap_r = random.randint(1, span_h - 3)
    gap_c = random.randint(1, span_w - 3)
    rows = tuple(top + i for i in range(span_h) if i not in (gap_r, gap_r + 1))
    cols = tuple(left + j for j in range(span_w) if j not in (gap_c, gap_c + 1))

    target_colors = tuple(random.sample(NON_GREY, 2))
    fill = random.choice([c for c in NON_GREY if c not in target_colors])
    distractor_colors = random.sample([c for c in NON_GREY if c not in (*target_colors, fill)], 3)

    pattern = _bicolor_pattern(out_h, out_w, target_colors)
    holes = _place_holes(out_h, out_w)
    if holes is None:
        return None
    target = pattern.copy()
    target[holes] = BG
    expected = pattern.copy()
    expected[holes] = fill

    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    for rr, r in enumerate(rows):
        for cc, c in enumerate(cols):
            grid[r, c] = int(target[rr, cc])
            occupied[r, c] = True

    horizontal = random.choice([True, False])
    legend = _legend_patch(fill, horizontal)
    if horizontal:
        lr = top + gap_r
        lc_min = left
        lc_max = left + span_w - legend.shape[1]
        if lc_max < lc_min:
            return None
        lc = random.randint(lc_min, lc_max)
        if not _can_place(occupied, lr, lc, legend, margin=0):
            return None
        grid[lr : lr + 2, lc : lc + legend.shape[1]] = legend
        _reserve(occupied, lr, lc, legend)
        legend_meta = {"orientation": "h", "top": lr, "left": lc, "shape": legend.shape}
    else:
        lc = left + gap_c
        lr_min = top
        lr_max = top + span_h - legend.shape[0]
        if lr_max < lr_min:
            return None
        lr = random.randint(lr_min, lr_max)
        if not _can_place(occupied, lr, lc, legend, margin=0):
            return None
        grid[lr : lr + legend.shape[0], lc : lc + 2] = legend
        _reserve(occupied, lr, lc, legend)
        legend_meta = {"orientation": "v", "top": lr, "left": lc, "shape": legend.shape}

    for _ in range(random.randint(1, 2)):
        patch = _mixed_patch(distractor_colors)
        placed = False
        for _attempt in range(80):
            pr = random.randint(0, grid_h - patch.shape[0])
            pc = random.randint(0, grid_w - patch.shape[1])
            if not _can_place(occupied, pr, pc, patch, margin=1):
                continue
            grid[pr : pr + patch.shape[0], pc : pc + patch.shape[1]] = patch
            _reserve(occupied, pr, pc, patch)
            placed = True
            break
        if not placed:
            return None

    meta = {
        "rows": rows,
        "cols": cols,
        "fill": fill,
        "target_colors": tuple(sorted(target_colors)),
        "expected": expected,
        "legend": legend_meta,
    }
    return grid, expected, meta


def _validate_case(grid: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    rows = meta["rows"]
    cols = meta["cols"]
    fill = int(meta["fill"])

    target = _find_target(grid)
    if target["rows"] != rows or target["cols"] != cols:
        return False
    if target["colors"] != tuple(sorted(meta["target_colors"])):
        return False
    if _legend_fill(grid, rows, cols) != fill:
        return False
    if not np.array_equal(main(grid), expected):
        return False
    if expected.shape != (len(rows), len(cols)):
        return False
    if not np.any(target["subgrid"] == BG):
        return False
    if not np.all(expected[target["subgrid"] == BG] == fill):
        return False
    if not any(b - a > 1 for a, b in zip(rows, rows[1:])):
        return False
    if not any(b - a > 1 for a, b in zip(cols, cols[1:])):
        return False
    bbox_shape = (max(rows) - min(rows) + 1, max(cols) - min(cols) + 1)
    if bbox_shape == expected.shape:
        return False
    legend = meta["legend"]
    if legend["orientation"] == "h":
        lr = legend["top"]
        if lr <= min(rows) or lr + 1 >= max(rows):
            return False
    else:
        lc = legend["left"]
        if lc <= min(cols) or lc + 1 >= max(cols):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_case()
        if built is None:
            continue
        grid, expected, meta = built
        if _validate_case(grid, expected, meta):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
