import random
from collections import Counter

import numpy as np


BG = 0
SEPARATOR_COLOR = 5
FRAME_COLOR = 8
PAIR_COLORS = [1, 2, 3, 4, 6, 7, 9]

PAIR_COUNT_RANGE = (3, 6)
SLOT_SIZE = 4
MARGIN_RANGE = (1, 2)
GEN_RETRIES = 400
MAX_DIM = 30


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _find_separator(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    h, w = grid.shape
    candidates: list[tuple[str, int, int]] = []
    for r in range(h):
        if int(grid[r, 0]) != bg and bool((grid[r] == grid[r, 0]).all()):
            candidates.append(("row", int(r), int(grid[r, 0])))
    for c in range(w):
        if int(grid[0, c]) != bg and bool((grid[:, c] == grid[0, c]).all()):
            candidates.append(("col", int(c), int(grid[0, c])))
    assert len(candidates) == 1, "expected exactly one solid separator line"
    return candidates[0]


def _split_regions(
    grid: np.ndarray, separator_kind: str, separator_idx: int
) -> list[tuple[np.ndarray, int, int]]:
    if separator_kind == "row":
        return [
            (grid[:separator_idx, :], 0, 0),
            (grid[separator_idx + 1 :, :], separator_idx + 1, 0),
        ]
    return [
        (grid[:, :separator_idx], 0, 0),
        (grid[:, separator_idx + 1 :], 0, separator_idx + 1),
    ]


def _frame_bbox(region: np.ndarray, bg: int) -> tuple[int, int, int, int] | None:
    candidates: list[tuple[int, int, int, int]] = []
    for color in np.unique(region):
        color = int(color)
        if color == bg:
            continue
        pts = np.argwhere(region == color)
        if len(pts) < 8:
            continue
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh < 3 or ww < 3:
            continue
        if len(pts) != 2 * (hh + ww) - 4:
            continue
        sub = region[r0 : r1 + 1, c0 : c1 + 1]
        if not bool((sub[0, :] == color).all()):
            continue
        if not bool((sub[-1, :] == color).all()):
            continue
        if not bool((sub[:, 0] == color).all()):
            continue
        if not bool((sub[:, -1] == color).all()):
            continue
        if sub.shape[0] > 2 and sub.shape[1] > 2 and not bool((sub[1:-1, 1:-1] == bg).all()):
            continue
        candidates.append((r0, c0, r1, c1))
    if len(candidates) != 1:
        return None
    return candidates[0]


def _locate_template(
    grid: np.ndarray, separator_kind: str, separator_idx: int, bg: int
) -> tuple[tuple[int, int, int, int], int]:
    matches: list[tuple[tuple[int, int, int, int], int]] = []
    for region_idx, (region, r_off, c_off) in enumerate(_split_regions(grid, separator_kind, separator_idx)):
        bbox = _frame_bbox(region, bg)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        matches.append(((r0 + r_off, c0 + c_off, r1 + r_off, c1 + c_off), int(region_idx)))
    assert len(matches) == 1, "expected exactly one template frame"
    return matches[0]


def _instruction_region(
    grid: np.ndarray, separator_kind: str, separator_idx: int, template_region_idx: int
) -> np.ndarray:
    for region_idx, (region, _r_off, _c_off) in enumerate(_split_regions(grid, separator_kind, separator_idx)):
        if int(region_idx) != int(template_region_idx):
            return region
    raise AssertionError("instruction region not found")


def _sorted_pairs(region: np.ndarray, bg: int) -> list[tuple[int, str]]:
    pairs: list[tuple[int, int, int, str]] = []
    for color in np.unique(region):
        color = int(color)
        if color == bg:
            continue
        pts = np.argwhere(region == color)
        assert len(pts) == 2, "each instruction color must appear exactly twice"
        (r0, c0), (r1, c1) = [tuple(map(int, p)) for p in pts]
        if r0 == r1:
            kind = "horizontal"
        elif c0 == c1:
            kind = "vertical"
        else:
            assert abs(r0 - r1) == abs(c0 - c1), "diagonal pairs must lie on a 45-degree line"
            kind = "diagonal"
        pairs.append((r0 + r1, c0 + c1, color, kind))
    pairs.sort(key=lambda item: (item[0], item[1]))
    return [(color, kind) for _rr, _cc, color, kind in pairs]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw colored template stripes from the directions of scattered same-color pixel pairs.

    Input structure:
    - bg = most common color (black).
    - One solid grey separator line splits the grid into an instruction region and a template region.
    - The instruction region contains color pairs; each color appears exactly twice, aligned horizontally, vertically, or diagonally.
    - The template region contains one azure single-pixel-thick rectangular frame whose interior is blank.

    Rule:
    - Sort the instruction pairs by their midpoint position, top-to-bottom then left-to-right.
    - The k-th pair controls the k-th row and k-th column of the template interior.
    - Horizontal pairs fill the whole k-th row, vertical pairs fill the whole k-th column, and diagonal pairs place a single pixel at position (k, k).
    - When a row fill overlaps a column fill, the row color takes priority; the azure frame stays unchanged.

    Key insight:
    - The geometry of each color pair is the instruction type, while the spatial order of pair midpoints determines where that instruction is written inside the blank template.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    separator_kind, separator_idx, _separator_color = _find_separator(g, bg)
    template_bbox, template_region_idx = _locate_template(g, separator_kind, separator_idx, bg)
    instruction = _instruction_region(g, separator_kind, separator_idx, template_region_idx)
    pairs = _sorted_pairs(instruction, bg)

    r0, c0, r1, c1 = template_bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    assert inner_h >= len(pairs) and inner_w >= len(pairs), "template interior is too small"

    canvas = np.full((inner_h, inner_w), bg, dtype=int)

    for k, (color, kind) in enumerate(pairs):
        if kind == "vertical":
            canvas[:, k] = int(color)
    for k, (color, kind) in enumerate(pairs):
        if kind == "diagonal":
            canvas[k, k] = int(color)
    for k, (color, kind) in enumerate(pairs):
        if kind == "horizontal":
            canvas[k, :] = int(color)

    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = canvas
    return out


def _margin() -> int:
    return random.randint(*MARGIN_RANGE)


def _balanced_kinds(count: int) -> list[str]:
    base = ["horizontal", "vertical", "diagonal"]
    start = random.randrange(len(base))
    kinds = [base[(start + i) % len(base)] for i in range(count)]
    random.shuffle(kinds)
    return kinds


def _pair_pixels(kind: str, base_r: int, base_c: int) -> list[tuple[int, int]]:
    if kind == "horizontal":
        return [(base_r + 1, base_c + 0), (base_r + 1, base_c + 2)]
    if kind == "vertical":
        return [(base_r + 0, base_c + 1), (base_r + 2, base_c + 1)]
    if random.random() < 0.5:
        return [(base_r + 0, base_c + 0), (base_r + 2, base_c + 2)]
    return [(base_r + 0, base_c + 2), (base_r + 2, base_c + 0)]


def _make_instruction_block(pair_count: int) -> np.ndarray:
    layout_options = [(rows, cols) for rows in range(1, 4) for cols in range(1, 4) if rows * cols >= pair_count]
    slot_rows, slot_cols = random.choice(layout_options)
    chosen_slots = sorted(random.sample(range(slot_rows * slot_cols), pair_count))
    colors = random.sample(PAIR_COLORS, pair_count)
    kinds = _balanced_kinds(pair_count)

    block = np.full((slot_rows * SLOT_SIZE, slot_cols * SLOT_SIZE), BG, dtype=int)
    for color, kind, slot_idx in zip(colors, kinds, chosen_slots):
        sr = (slot_idx // slot_cols) * SLOT_SIZE + random.randint(0, 1)
        sc = (slot_idx % slot_cols) * SLOT_SIZE + random.randint(0, 1)
        for r, c in _pair_pixels(kind, sr, sc):
            block[r, c] = int(color)
    return block


def _make_template_block(pair_count: int) -> np.ndarray:
    block = np.full((pair_count + 2, pair_count + 2), BG, dtype=int)
    block[0, :] = FRAME_COLOR
    block[-1, :] = FRAME_COLOR
    block[:, 0] = FRAME_COLOR
    block[:, -1] = FRAME_COLOR
    return block


def _compose_vertical(left_obj: np.ndarray, right_obj: np.ndarray) -> np.ndarray | None:
    lh, lw = map(int, left_obj.shape)
    rh, rw = map(int, right_obj.shape)

    left_top_min, left_bottom_min = _margin(), _margin()
    right_top_min, right_bottom_min = _margin(), _margin()
    total_h = max(lh + left_top_min + left_bottom_min, rh + right_top_min + right_bottom_min)

    left_outer, left_inner = _margin(), _margin()
    right_inner, right_outer = _margin(), _margin()
    total_w = lw + left_outer + left_inner + 1 + rw + right_inner + right_outer
    if total_h > MAX_DIM or total_w > MAX_DIM:
        return None

    grid = np.full((total_h, total_w), BG, dtype=int)
    sep_c = left_outer + lw + left_inner
    grid[:, sep_c] = SEPARATOR_COLOR

    left_top = left_top_min + random.randint(0, total_h - (lh + left_top_min + left_bottom_min))
    right_top = right_top_min + random.randint(0, total_h - (rh + right_top_min + right_bottom_min))

    grid[left_top : left_top + lh, left_outer : left_outer + lw] = left_obj
    right_c0 = sep_c + 1 + right_inner
    grid[right_top : right_top + rh, right_c0 : right_c0 + rw] = right_obj
    return grid


def _compose_horizontal(top_obj: np.ndarray, bottom_obj: np.ndarray) -> np.ndarray | None:
    th, tw = map(int, top_obj.shape)
    bh, bw = map(int, bottom_obj.shape)

    top_left_min, top_right_min = _margin(), _margin()
    bottom_left_min, bottom_right_min = _margin(), _margin()
    total_w = max(tw + top_left_min + top_right_min, bw + bottom_left_min + bottom_right_min)

    top_outer, top_inner = _margin(), _margin()
    bottom_inner, bottom_outer = _margin(), _margin()
    total_h = th + top_outer + top_inner + 1 + bh + bottom_inner + bottom_outer
    if total_h > MAX_DIM or total_w > MAX_DIM:
        return None

    grid = np.full((total_h, total_w), BG, dtype=int)
    sep_r = top_outer + th + top_inner
    grid[sep_r, :] = SEPARATOR_COLOR

    top_left = top_left_min + random.randint(0, total_w - (tw + top_left_min + top_right_min))
    bottom_left = bottom_left_min + random.randint(0, total_w - (bw + bottom_left_min + bottom_right_min))

    grid[top_outer : top_outer + th, top_left : top_left + tw] = top_obj
    bottom_r0 = sep_r + 1 + bottom_inner
    grid[bottom_r0 : bottom_r0 + bh, bottom_left : bottom_left + bw] = bottom_obj
    return grid


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    instruction = _make_instruction_block(pair_count)
    template = _make_template_block(pair_count)

    separator_kind = random.choice(["row", "col"])
    template_first = random.random() < 0.5

    if separator_kind == "col":
        left_obj, right_obj = (template, instruction) if template_first else (instruction, template)
        grid = _compose_vertical(left_obj, right_obj)
    else:
        top_obj, bottom_obj = (template, instruction) if template_first else (instruction, template)
        grid = _compose_horizontal(top_obj, bottom_obj)

    if grid is None:
        return None

    assert grid.ndim == 2 and max(grid.shape) <= MAX_DIM
    out = main(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid), "template should receive at least one painted cell"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
