from __future__ import annotations

from importlib.machinery import SourceFileLoader

import numpy as np

_ARC_UTILS = SourceFileLoader(
    "_arc_utils_fable_1_024_7",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
bbox = _ARC_UTILS.bbox
colors_present = _ARC_UTILS.colors_present
crop_bbox = _ARC_UTILS.crop_bbox
get_bg = _ARC_UTILS.get_bg
translate = _ARC_UTILS.translate


BG = Color.BLACK
ANCHOR = Color.GREY
PALETTE = tuple(c for c in Color.ALL_COLORS if c not in {BG, ANCHOR})
RNG = np.random.default_rng()
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

PANEL_H_RANGE = (9, 15)  # upper-exclusive
PANEL_W_RANGE = (9, 15)  # upper-exclusive
BASE_AREA_RANGE = (5, 10)  # upper-exclusive
EXTRA_AREA_RANGE = (2, 6)  # upper-exclusive
MAX_REL = 4
MAX_ATTEMPTS = 400


def _uniform_color(line: np.ndarray) -> int | None:
    return int(line[0]) if np.all(line == line[0]) else None


def _find_separator(g: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = g.shape
    candidates = []
    for c in range(1, w - 1):
        if c != w - c - 1:
            continue
        col = g[:, c]
        sep_color = _uniform_color(col)
        if sep_color is None or sep_color == bg or sep_color == ANCHOR:
            continue
        candidates.append((int(c), int(sep_color)))
    assert len(candidates) == 1, "expected one solid separator column between equal-width panels"
    return candidates[0]


def _panel_info(panel: np.ndarray, bg: int) -> dict:
    anchors = np.argwhere(panel == ANCHOR)
    assert len(anchors) == 1, "each panel must contain exactly one grey anchor"
    anchor = tuple(map(int, anchors[0]))

    fg_colors = colors_present(panel, exclude={bg, ANCHOR})
    assert len(fg_colors) == 1, "each panel must contain exactly one mask color"
    color = int(fg_colors[0])
    mask = np.asarray(panel == color, dtype=bool)
    area = int(mask.sum())
    assert area > 0
    return {"anchor": anchor, "color": color, "mask": mask, "area": area}


def _choose_large_small(a: dict, b: dict) -> tuple[dict, dict]:
    assert int(a["area"]) != int(b["area"]), "the two masks must have different sizes"
    return (a, b) if int(a["area"]) > int(b["area"]) else (b, a)


def _shift_mask(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    shifted = translate(mask.astype(int), int(dr), int(dc), background=0)
    return np.asarray(shifted, dtype=bool)


def _cut_output(large: dict, small: dict, bg: int) -> np.ndarray:
    dr = int(large["anchor"][0]) - int(small["anchor"][0])
    dc = int(large["anchor"][1]) - int(small["anchor"][1])
    shifted_small = _shift_mask(small["mask"], dr, dc)

    out = np.full(large["mask"].shape, int(bg), dtype=int)
    out[large["mask"] & ~shifted_small] = int(large["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the smaller mask to the larger one by their grey anchors and cut that overlap out of the larger mask.

    Input structure:
    - A single solid separator column splits the grid into equal-width left and right panels.
    - Each panel contains background, exactly one colored foreground mask, and exactly one grey anchor pixel.
    - The two mask colors differ, and the two mask areas are not equal.

    Rule:
    - Identify the panel with fewer foreground cells as the cutting mask and the other panel as the target mask.
    - Translate the smaller mask so that its grey anchor lands on the larger panel's grey anchor.
    - Output only the larger panel: keep its foreground cells except where the translated smaller mask covers them, and do not draw the separator or either anchor.

    Key insight:
    - The alignment reference is the grey anchor, not the panel frame or the mask bounding box, so the transformation is a directed set subtraction after anchor-based translation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    bg = int(get_bg(g))
    sep_col, _sep_color = _find_separator(g, bg)
    left = crop_bbox(g, (0, 0, g.shape[0] - 1, sep_col - 1))
    right = crop_bbox(g, (0, sep_col + 1, g.shape[0] - 1, g.shape[1] - 1))
    assert left.shape == right.shape

    left_info = _panel_info(left, bg)
    right_info = _panel_info(right, bg)
    large, small = _choose_large_small(left_info, right_info)
    return _cut_output(large, small, bg)


def _bounds(offsets: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in offsets]
    cols = [c for _r, c in offsets]
    return min(rows), max(rows), min(cols), max(cols)


def _grow_connected(candidates: set[tuple[int, int]], area: int) -> set[tuple[int, int]]:
    preferred = [cell for cell in candidates if abs(cell[0]) + abs(cell[1]) <= 2]
    seed_pool = preferred or list(candidates)
    start = seed_pool[int(RNG.integers(0, len(seed_pool)))]
    cells = {start}

    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in N4:
                nb = (r + dr, c + dc)
                if nb in candidates and nb not in cells:
                    frontier.add(nb)
        if not frontier:
            return set()
        frontier_list = list(frontier)
        cells.add(frontier_list[int(RNG.integers(0, len(frontier_list)))])
    return cells


def _sample_base_offsets() -> set[tuple[int, int]] | None:
    for _ in range(120):
        top = int(RNG.integers(1, 4))
        bottom = int(RNG.integers(1, 4))
        left = int(RNG.integers(1, 4))
        right = int(RNG.integers(1, 4))
        candidates = {
            (dr, dc)
            for dr in range(-top, bottom + 1)
            for dc in range(-left, right + 1)
            if (dr, dc) != (0, 0)
        }
        max_area = min(len(candidates), BASE_AREA_RANGE[1] - 1)
        if max_area < BASE_AREA_RANGE[0]:
            continue
        area = int(RNG.integers(BASE_AREA_RANGE[0], max_area + 1))
        offsets = _grow_connected(candidates, area)
        if not offsets:
            continue
        min_r, max_r, min_c, max_c = _bounds(offsets)
        if min_r < 0 < max_r and min_c < 0 < max_c:
            return offsets
    return None


def _expand_amount(side: str, min_r: int, max_r: int, min_c: int, max_c: int) -> int:
    if side == "top":
        return int(MAX_REL - abs(min_r))
    if side == "bottom":
        return int(MAX_REL - abs(max_r))
    if side == "left":
        return int(MAX_REL - abs(min_c))
    if side == "right":
        return int(MAX_REL - abs(max_c))
    raise ValueError(f"unknown side: {side}")


def _sample_large_offsets(base: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
    base = set(base)
    base_min_r, base_max_r, base_min_c, base_max_c = _bounds(base)
    target_extra = int(RNG.integers(EXTRA_AREA_RANGE[0], EXTRA_AREA_RANGE[1]))

    primary_sides = [
        side
        for side in ("top", "left")
        if _expand_amount(side, base_min_r, base_max_r, base_min_c, base_max_c) >= 1
    ]
    if not primary_sides:
        return None

    for _ in range(120):
        expand = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        primary = primary_sides[int(RNG.integers(0, len(primary_sides)))]
        primary_max = _expand_amount(primary, base_min_r, base_max_r, base_min_c, base_max_c)
        expand[primary] = int(RNG.integers(1, primary_max + 1))

        other_sides = []
        for side in ("top", "bottom", "left", "right"):
            if side == primary:
                continue
            side_max = _expand_amount(side, base_min_r, base_max_r, base_min_c, base_max_c)
            if side_max >= 1 and RNG.random() < 0.45:
                expand[side] = int(RNG.integers(1, side_max + 1))
                other_sides.append(side)

        min_r = base_min_r - expand["top"]
        max_r = base_max_r + expand["bottom"]
        min_c = base_min_c - expand["left"]
        max_c = base_max_c + expand["right"]

        candidates = {
            (dr, dc)
            for dr in range(min_r, max_r + 1)
            for dc in range(min_c, max_c + 1)
            if (dr, dc) != (0, 0)
        }

        occupied = set(base)
        extras: set[tuple[int, int]] = set()
        while len(extras) < target_extra:
            frontier = set()
            for r, c in occupied:
                for dr, dc in N4:
                    nb = (r + dr, c + dc)
                    if nb != (0, 0) and nb in candidates and nb not in occupied:
                        frontier.add(nb)
            if not frontier:
                break
            frontier_list = list(frontier)
            chosen = frontier_list[int(RNG.integers(0, len(frontier_list)))]
            occupied.add(chosen)
            extras.add(chosen)

        if len(extras) != target_extra:
            continue
        if not any(r < base_min_r or c < base_min_c for r, c in extras):
            continue

        large = occupied
        large_min_r, _large_max_r, large_min_c, _large_max_c = _bounds(large)
        if large_min_r == base_min_r and large_min_c == base_min_c:
            continue
        return large

    return None


def _anchor_positions(shape: tuple[int, int], offsets: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = shape
    min_r, max_r, min_c, max_c = _bounds(offsets)
    row_lo = max(1, -min_r)
    row_hi = min(h - 2, h - 1 - max_r)
    col_lo = max(1, -min_c)
    col_hi = min(w - 2, w - 1 - max_c)
    if row_lo > row_hi or col_lo > col_hi:
        return []
    return [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]


def _sample_anchor_pair(
    shape: tuple[int, int],
    small_offsets: set[tuple[int, int]],
    large_offsets: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    small_positions = _anchor_positions(shape, small_offsets)
    large_positions = _anchor_positions(shape, large_offsets)
    if not small_positions or not large_positions:
        return None

    mode = str(RNG.choice(np.array(["row", "col", "both"], dtype=object), p=[0.3, 0.3, 0.4]))
    valid = []
    fallback = []
    for small_anchor in small_positions:
        for large_anchor in large_positions:
            dr = int(large_anchor[0] - small_anchor[0])
            dc = int(large_anchor[1] - small_anchor[1])
            if abs(dr) + abs(dc) < 2:
                continue
            fallback.append((small_anchor, large_anchor))
            if mode == "row" and dr != 0 and dc == 0:
                valid.append((small_anchor, large_anchor))
            elif mode == "col" and dr == 0 and dc != 0:
                valid.append((small_anchor, large_anchor))
            elif mode == "both" and dr != 0 and dc != 0:
                valid.append((small_anchor, large_anchor))

    choices = valid or fallback
    if not choices:
        return None
    return choices[int(RNG.integers(0, len(choices)))]


def _render_panel(
    shape: tuple[int, int],
    offsets: set[tuple[int, int]],
    anchor: tuple[int, int],
    color: int,
    bg: int,
) -> np.ndarray:
    panel = np.full(shape, int(bg), dtype=int)
    for dr, dc in offsets:
        rr = int(anchor[0] + dr)
        cc = int(anchor[1] + dc)
        panel[rr, cc] = int(color)
    assert int(panel[anchor]) == int(bg)
    panel[anchor] = ANCHOR
    return panel


def _same_coords_cut(large: dict, small: dict, bg: int) -> np.ndarray:
    out = np.full(large["mask"].shape, int(bg), dtype=int)
    out[large["mask"] & ~small["mask"]] = int(large["color"])
    return out


def _bbox_align_cut(large: dict, small: dict, bg: int) -> np.ndarray:
    large_box = bbox(large["mask"])
    small_box = bbox(small["mask"])
    assert large_box is not None and small_box is not None
    dr = int(large_box[0] - small_box[0])
    dc = int(large_box[1] - small_box[1])
    shifted_small = _shift_mask(small["mask"], dr, dc)
    out = np.full(large["mask"].shape, int(bg), dtype=int)
    out[large["mask"] & ~shifted_small] = int(large["color"])
    return out


def _semantic_checks(inp: np.ndarray) -> bool:
    bg = int(get_bg(inp))
    sep_col, _sep_color = _find_separator(inp, bg)
    left = crop_bbox(inp, (0, 0, inp.shape[0] - 1, sep_col - 1))
    right = crop_bbox(inp, (0, sep_col + 1, inp.shape[0] - 1, inp.shape[1] - 1))
    left_info = _panel_info(left, bg)
    right_info = _panel_info(right, bg)
    large, small = _choose_large_small(left_info, right_info)

    true_out = _cut_output(large, small, bg)
    same_out = _same_coords_cut(large, small, bg)
    bbox_out = _bbox_align_cut(large, small, bg)

    fg = int(large["color"])
    visible = int(np.count_nonzero(true_out == fg))
    removed = int(large["area"]) - visible
    dr = int(large["anchor"][0] - small["anchor"][0])
    dc = int(large["anchor"][1] - small["anchor"][1])

    return (
        abs(dr) + abs(dc) >= 2
        and visible >= 2
        and removed >= 2
        and np.array_equal(main(inp), true_out)
        and not np.array_equal(true_out, same_out)
        and not np.array_equal(true_out, bbox_out)
    )


def _gen_attempt() -> np.ndarray | None:
    panel_h = int(RNG.integers(PANEL_H_RANGE[0], PANEL_H_RANGE[1]))
    panel_w = int(RNG.integers(PANEL_W_RANGE[0], PANEL_W_RANGE[1]))
    panel_shape = (panel_h, panel_w)

    small_offsets = _sample_base_offsets()
    if small_offsets is None:
        return None
    large_offsets = _sample_large_offsets(small_offsets)
    if large_offsets is None:
        return None

    anchors = _sample_anchor_pair(panel_shape, small_offsets, large_offsets)
    if anchors is None:
        return None
    small_anchor, large_anchor = anchors

    colors = [int(c) for c in RNG.choice(np.array(PALETTE, dtype=int), size=3, replace=False)]
    large_color, small_color, sep_color = colors
    large_left = bool(RNG.integers(0, 2))

    large_panel = _render_panel(panel_shape, large_offsets, large_anchor, large_color, BG)
    small_panel = _render_panel(panel_shape, small_offsets, small_anchor, small_color, BG)
    left_panel = large_panel if large_left else small_panel
    right_panel = small_panel if large_left else large_panel

    grid = np.full((panel_h, panel_w * 2 + 1), BG, dtype=int)
    grid[:, :panel_w] = left_panel
    grid[:, panel_w] = sep_color
    grid[:, panel_w + 1 :] = right_panel

    return grid if _semantic_checks(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
