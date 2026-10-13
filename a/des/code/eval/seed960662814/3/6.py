from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))

PAIR_COUNT_POOL = (1, 2, 2, 3, 3, 4)
LINE_LEN_RANGE = (3, 8)
GAP_RANGE = (2, 6)
PLACEMENT_PAD = 1
PLACEMENT_TRIES = 200
GEN_RETRIES = 400

CASE_OPTIONS = (
    ("left", "above"),
    ("left", "below"),
    ("right", "above"),
    ("right", "below"),
)

GRID_RANGES = {
    1: ((12, 24), (12, 24)),
    2: ((18, 28), (18, 28)),
    3: ((22, 30), (22, 30)),
    4: ((26, 30), (26, 30)),
}


def _background_color(grid: np.ndarray) -> int:
    if bool((grid == BG).any()):
        return BG
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _parse_strip(points: list[tuple[int, int]], color: int) -> dict[str, int]:
    rows = sorted({int(r) for r, _ in points})
    cols = sorted({int(c) for _, c in points})
    if len(rows) == 1:
        row = int(rows[0])
        assert cols == list(range(cols[0], cols[-1] + 1)), "horizontal strip must be contiguous"
        assert 3 <= len(cols) <= 8, "horizontal strip length out of range"
        return {"color": int(color), "orientation": 0, "row": row, "c0": int(cols[0]), "c1": int(cols[-1])}
    if len(cols) == 1:
        col = int(cols[0])
        assert rows == list(range(rows[0], rows[-1] + 1)), "vertical strip must be contiguous"
        assert 3 <= len(rows) <= 8, "vertical strip length out of range"
        return {"color": int(color), "orientation": 1, "col": col, "r0": int(rows[0]), "r1": int(rows[-1])}
    raise AssertionError("each component must be a one-pixel-wide strip")


def _extract_pairs(grid: np.ndarray) -> list[dict[str, dict[str, int] | int]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    pairs: list[dict[str, dict[str, int] | int]] = []
    for color in (int(v) for v in np.unique(g) if int(v) != bg):
        comps = _components_mask(g == int(color))
        assert len(comps) == 2, "each foreground color must appear in exactly two strips"
        strips = [_parse_strip(comp, color) for comp in comps]
        h_strips = [strip for strip in strips if int(strip["orientation"]) == 0]
        v_strips = [strip for strip in strips if int(strip["orientation"]) == 1]
        assert len(h_strips) == 1 and len(v_strips) == 1, "each color needs one horizontal and one vertical strip"
        h_strip = h_strips[0]
        v_strip = v_strips[0]
        row = int(h_strip["row"])
        col = int(v_strip["col"])
        assert col < int(h_strip["c0"]) or col > int(h_strip["c1"]), "the projected crossing must be off the horizontal strip"
        assert row < int(v_strip["r0"]) or row > int(v_strip["r1"]), "the projected crossing must be off the vertical strip"
        pairs.append({"color": int(color), "h": h_strip, "v": v_strip})
    pairs.sort(
        key=lambda pair: (
            min(int(pair["h"]["row"]), int(pair["v"]["r0"])),
            min(int(pair["h"]["c0"]), int(pair["v"]["col"])),
        )
    )
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored horizontal and vertical strips, weave the blank corner between
    their facing ends, then extend both strips toward the projected crossing point.

    The woven rectangle is the box whose corners are:
    - the horizontal strip end nearest the vertical strip, and
    - the vertical strip end nearest the horizontal strip,
      together with the strips' row/column projection intersection.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()

    for pair in _extract_pairs(g):
        color = int(pair["color"])
        h_strip = pair["h"]
        v_strip = pair["v"]

        row = int(h_strip["row"])
        col = int(v_strip["col"])
        h_face = int(h_strip["c1"]) if int(h_strip["c1"]) < col else int(h_strip["c0"])
        v_face = int(v_strip["r1"]) if int(v_strip["r1"]) < row else int(v_strip["r0"])

        r0, r1 = sorted((row, v_face))
        c0, c1 = sorted((h_face, col))

        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if int(out[r, c]) == bg and (r + c) % 2 == 0:
                    out[r, c] = color

        step_c = 1 if col > h_face else -1
        c = h_face + step_c
        while (step_c == 1 and c <= col) or (step_c == -1 and c >= col):
            value = int(out[row, c])
            if value not in (bg, color):
                break
            out[row, c] = color
            c += step_c

        step_r = 1 if row > v_face else -1
        r = v_face + step_r
        while (step_r == 1 and r <= row) or (step_r == -1 and r >= row):
            value = int(out[r, col])
            if value not in (bg, color):
                break
            out[r, col] = color
            r += step_r

    return out


def _choose_grid_shape(pair_count: int) -> tuple[int, int]:
    (h_min, h_max), (w_min, w_max) = GRID_RANGES[int(pair_count)]
    return random.randint(h_min, h_max), random.randint(w_min, w_max)


def _choose_cases(pair_count: int) -> list[tuple[str, str]]:
    if pair_count <= len(CASE_OPTIONS):
        return random.sample(CASE_OPTIONS, k=pair_count)
    chosen = list(CASE_OPTIONS)
    while len(chosen) < pair_count:
        chosen.append(random.choice(CASE_OPTIONS))
    random.shuffle(chosen)
    return chosen


def _make_pair_patch(color: int, case: tuple[str, str]) -> np.ndarray:
    h_dir, v_dir = case
    h_len = random.randint(*LINE_LEN_RANGE)
    v_len = random.randint(*LINE_LEN_RANGE)
    gap_cols = random.randint(*GAP_RANGE)
    gap_rows = random.randint(*GAP_RANGE)

    patch_h = gap_rows + v_len
    patch_w = gap_cols + h_len
    patch = np.full((patch_h, patch_w), BG, dtype=int)

    if v_dir == "above":
        row = patch_h - 1
        vr0, vr1 = 0, v_len - 1
    else:
        row = 0
        vr0, vr1 = gap_rows, gap_rows + v_len - 1

    if h_dir == "left":
        col = patch_w - 1
        hc0, hc1 = 0, h_len - 1
    else:
        col = 0
        hc0, hc1 = gap_cols, gap_cols + h_len - 1

    patch[row, hc0 : hc1 + 1] = int(color)
    patch[vr0 : vr1 + 1, col] = int(color)
    return patch


def _can_place(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rr0 = max(0, int(r0) - pad)
    cc0 = max(0, int(c0) - pad)
    rr1 = min(mask.shape[0], int(r0) + int(h) + pad)
    cc1 = min(mask.shape[1], int(c0) + int(w) + pad)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _reserve(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, int(r0) - pad)
    cc0 = max(0, int(c0) - pad)
    rr1 = min(mask.shape[0], int(r0) + int(h) + pad)
    cc1 = min(mask.shape[1], int(c0) + int(w) + pad)
    mask[rr0:rr1, cc0:cc1] = True


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_POOL)
    height, width = _choose_grid_shape(pair_count)
    grid = np.full((height, width), BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    colors = random.sample(COLORS, k=pair_count)
    cases = _choose_cases(pair_count)

    for color, case in zip(colors, cases):
        placed = False
        for _ in range(PLACEMENT_TRIES):
            patch = _make_pair_patch(color, case)
            patch_h, patch_w = map(int, patch.shape)
            if patch_h > height or patch_w > width:
                continue
            r0 = random.randint(0, height - patch_h)
            c0 = random.randint(0, width - patch_w)
            if not _can_place(reserved, r0, c0, patch_h, patch_w, PLACEMENT_PAD):
                continue

            window = grid[r0 : r0 + patch_h, c0 : c0 + patch_w]
            window[patch != BG] = patch[patch != BG]
            _reserve(reserved, r0, c0, patch_h, patch_w, PLACEMENT_PAD)
            placed = True
            break
        if not placed:
            return None

    try:
        pairs = _extract_pairs(grid)
        if len(pairs) != pair_count:
            return None
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid woven-strip ARC input")
