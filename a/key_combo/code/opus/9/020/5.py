from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox_h, colors_present, find_components, get_bg
except ModuleNotFoundError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import bbox_h, colors_present, find_components, get_bg


BG = 0
GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (24, 28)
HALF_H_RANGE = (2, 4)
D_RANGE = (2, 4)
LINE_COUNT_RANGE = (2, 4)
FRAG_LEN_RANGE = (2, 4)
GEN_RETRIES = 400


def _swap_values(values: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    out = values.copy()
    out[values == int(color_a)] = int(color_b)
    out[values == int(color_b)] = int(color_a)
    return out


def _pick_object_component(grid: np.ndarray, bg: int) -> tuple[dict, list[int], np.ndarray, int]:
    candidates = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=False):
        coords = comp["coords"]
        values = sorted({int(grid[r, c]) for r, c in coords})
        if len(values) != 2:
            continue
        if bbox_h(comp["bbox"]) % 2 != 0:
            continue
        mask = np.zeros_like(grid, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        r0, _c0, r1, _c1 = comp["bbox"]
        split_row = int(r0 + (r1 - r0 + 1) // 2)
        if not np.any(mask[:split_row]) or not np.any(mask[split_row:]):
            continue
        if not np.any(mask[split_row - 1]) or not np.any(mask[split_row]):
            continue
        candidates.append((comp, values, mask, split_row))
    assert len(candidates) == 1, "expected exactly one even-height two-color object"
    return candidates[0]


def _object_shift(mask: np.ndarray, split_row: int) -> int:
    upper_boundary_cols = np.flatnonzero(mask[split_row - 1])
    lower_boundary_cols = np.flatnonzero(mask[split_row])
    assert upper_boundary_cols.size > 0 and lower_boundary_cols.size > 0
    shift = int(lower_boundary_cols.max() + 1 - upper_boundary_cols.min())
    assert shift > 0
    return shift


def _straight_orientation(comp: dict) -> str:
    r0, c0, r1, c1 = comp["bbox"]
    if r0 == r1 and c1 > c0:
        return "h"
    if c0 == c1 and r1 > r0:
        return "v"
    raise AssertionError("line fragments must be nontrivial straight segments")


def _extract_broken_lines(grid: np.ndarray, line_color: int, bg: int) -> list[dict]:
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[line_color],
    )
    assert len(comps) >= 4 and len(comps) % 2 == 0

    orientations = [_straight_orientation(comp) for comp in comps]
    candidates = []
    for i in range(len(comps)):
        b1 = comps[i]["bbox"]
        for j in range(i + 1, len(comps)):
            b2 = comps[j]["bbox"]
            if orientations[i] != orientations[j]:
                continue
            if orientations[i] == "h" and b1[0] == b2[0]:
                left_i, right_i = (i, j) if b1[1] < b2[1] else (j, i)
                left, right = comps[left_i]["bbox"], comps[right_i]["bbox"]
                gap = int(right[1] - left[3] - 1)
                if gap >= 1 and np.all(grid[left[0], left[3] + 1 : right[1]] == int(bg)):
                    candidates.append(
                        {
                            "a": left_i,
                            "b": right_i,
                            "orientation": "h",
                            "gap": gap,
                            "anchor": int(left[0]),
                            "start": int(left[3] + 1),
                            "stop": int(right[1] - 1),
                        }
                    )
            if orientations[i] == "v" and b1[1] == b2[1]:
                top_i, bot_i = (i, j) if b1[0] < b2[0] else (j, i)
                top, bottom = comps[top_i]["bbox"], comps[bot_i]["bbox"]
                gap = int(bottom[0] - top[2] - 1)
                if gap >= 1 and np.all(grid[top[2] + 1 : bottom[0], top[1]] == int(bg)):
                    candidates.append(
                        {
                            "a": top_i,
                            "b": bot_i,
                            "orientation": "v",
                            "gap": gap,
                            "anchor": int(top[1]),
                            "start": int(top[2] + 1),
                            "stop": int(bottom[0] - 1),
                        }
                    )

    adjacency = [[] for _ in comps]
    for idx, cand in enumerate(candidates):
        adjacency[cand["a"]].append(idx)
        adjacency[cand["b"]].append(idx)
    assert candidates and all(len(edges) == 1 for edges in adjacency), "line pairing must be unique"

    unique = {adjacency[k][0] for k in range(len(adjacency))}
    assert len(unique) * 2 == len(comps)
    lines = [candidates[idx] for idx in sorted(unique)]
    return sorted(lines, key=lambda item: (item["orientation"], item["anchor"], item["start"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the two-color object, slide its swapped upper half, and use that slide distance to gate broken-line repair.

    Input structure:
    - Black is the background.
    - The upper area contains one 4-connected two-color object whose bounding-box height is even.
    - The lower area contains several broken one-cell-thick horizontal or vertical line segments in a third color.

    Rule:
    - Identify the unique two-color object and split it across the horizontal midpoint of its bounding box.
    - Swap the object's two colors in both halves, keep the lower half fixed, and shift the upper half right until its
      leftmost split-boundary pixel becomes diagonally adjacent to the lower half's rightmost split-boundary pixel.
    - Let that horizontal shift be D.
    - For each broken lower-area line, measure the length G of its single continuous gap; fill the gap with the line color
      iff G <= D, and leave it empty otherwise.

    Key insight:
    - The broken lines are not repaired by a fixed threshold: the threshold is the exact slide distance forced by the
      geometry of the split two-color object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    obj_comp, obj_colors, obj_mask, split_row = _pick_object_component(g, bg)
    color_a, color_b = obj_colors
    shift = _object_shift(obj_mask, split_row)

    fg_colors = set(colors_present(g, exclude={bg}))
    line_colors = fg_colors - set(obj_colors)
    assert len(line_colors) == 1, "expected exactly one line color"
    line_color = int(next(iter(line_colors)))

    row_ids = np.arange(g.shape[0])[:, None]
    upper_mask = obj_mask & (row_ids < split_row)
    lower_mask = obj_mask & ~upper_mask

    out = g.copy()
    out[obj_mask] = int(bg)

    lower_rows, lower_cols = np.where(lower_mask)
    lower_values = _swap_values(g[lower_rows, lower_cols], color_a, color_b)
    out[lower_rows, lower_cols] = lower_values

    upper_rows, upper_cols = np.where(upper_mask)
    shifted_cols = upper_cols + int(shift)
    assert int(shifted_cols.max()) < g.shape[1]
    upper_values = _swap_values(g[upper_rows, upper_cols], color_a, color_b)
    out[upper_rows, shifted_cols] = upper_values

    for line in _extract_broken_lines(g, line_color, bg):
        if line["gap"] > shift:
            continue
        if line["orientation"] == "h":
            out[line["anchor"], line["start"] : line["stop"] + 1] = int(line_color)
        else:
            out[line["start"] : line["stop"] + 1, line["anchor"]] = int(line_color)

    return out


def _grow_half_rows(height: int, boundary_left: int, boundary_width: int, local_w: int, reverse: bool) -> list[tuple[int, int]]:
    rows = [None] * height
    index = height - 1 if reverse else 0
    rows[index] = (boundary_left, boundary_width)
    left, width = boundary_left, boundary_width
    steps = range(height - 2, -1, -1) if reverse else range(1, height)
    for idx in steps:
        options = []
        for new_width in sorted({width - 1, width, width + 1}):
            if not 2 <= new_width <= local_w:
                continue
            lo = max(0, left - 1)
            hi = min(local_w - new_width, left + 1)
            for new_left in range(lo, hi + 1):
                if max(new_left, left) <= min(new_left + new_width - 1, left + width - 1):
                    options.append((new_left, new_width))
        assert options
        left, width = random.choice(options)
        rows[idx] = (left, width)
    return rows


def _build_object_patch(color_a: int, color_b: int) -> tuple[np.ndarray, int]:
    half_h = random.randint(*HALF_H_RANGE)
    shift = random.randint(*D_RANGE)
    upper_left = random.randint(0, 2)
    upper_width = random.randint(2, 4)

    lower_width_min = max(2, shift - upper_width + 1)
    lower_width_max = min(5, upper_left + shift)
    assert lower_width_min <= lower_width_max
    lower_width = random.randint(lower_width_min, lower_width_max)
    lower_left = upper_left + shift - lower_width

    local_w = max(upper_left + upper_width, lower_left + lower_width) + random.randint(1, 2)
    upper_rows = _grow_half_rows(half_h, upper_left, upper_width, local_w, reverse=True)
    lower_rows = _grow_half_rows(half_h, lower_left, lower_width, local_w, reverse=False)

    mask = np.zeros((half_h * 2, local_w), dtype=bool)
    for r, (left, width) in enumerate(upper_rows):
        mask[r, left : left + width] = True
    for r, (left, width) in enumerate(lower_rows, start=half_h):
        mask[r, left : left + width] = True

    cols = np.flatnonzero(mask.any(axis=0))
    mask = mask[:, cols[0] : cols[-1] + 1]

    rr, cc = np.indices(mask.shape)
    phase = random.randint(0, 1)
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask & (((rr + cc + phase) % 2) == 0)] = int(color_a)
    patch[mask & (((rr + cc + phase) % 2) == 1)] = int(color_b)

    top_colors = set(colors_present(patch[:half_h], exclude={BG}))
    bot_colors = set(colors_present(patch[half_h:], exclude={BG}))
    assert top_colors == {color_a, color_b}
    assert bot_colors == {color_a, color_b}

    comp, _colors, comp_mask, split_row = _pick_object_component(patch, BG)
    assert comp["bbox"][0] == 0 and comp["bbox"][2] == patch.shape[0] - 1
    parsed_shift = _object_shift(comp_mask, split_row)
    assert parsed_shift == shift
    return patch, shift


def _object_output_patch(patch: np.ndarray, shift: int, color_a: int, color_b: int) -> np.ndarray:
    half_h = patch.shape[0] // 2
    mask = patch != BG
    out = np.zeros((patch.shape[0], patch.shape[1] + shift), dtype=int)

    lower_rows, lower_cols = np.where(mask & (np.arange(patch.shape[0])[:, None] >= half_h))
    out[lower_rows, lower_cols] = _swap_values(patch[lower_rows, lower_cols], color_a, color_b)

    upper_rows, upper_cols = np.where(mask & (np.arange(patch.shape[0])[:, None] < half_h))
    out[upper_rows, upper_cols + shift] = _swap_values(patch[upper_rows, upper_cols], color_a, color_b)
    return out


def _line_cells(orientation: str, anchor: int, start: int, frag_a: int, gap: int, frag_b: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int, int, int]]:
    if orientation == "h":
        left = [(anchor, c) for c in range(start, start + frag_a)]
        right_start = start + frag_a + gap
        right = [(anchor, c) for c in range(right_start, right_start + frag_b)]
        bbox = (anchor, start, anchor, right_start + frag_b - 1)
    else:
        top = [(r, anchor) for r in range(start, start + frag_a)]
        bot_start = start + frag_a + gap
        bottom = [(r, anchor) for r in range(bot_start, bot_start + frag_b)]
        left = top
        right = bottom
        bbox = (start, anchor, bot_start + frag_b - 1, anchor)
    return left, right, bbox


def _expanded_ok(forbidden: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(forbidden.shape[0] - 1, r1 + 1)
    cc1 = min(forbidden.shape[1] - 1, c1 + 1)
    return not np.any(forbidden[rr0 : rr1 + 1, cc0 : cc1 + 1])


def _mark_expanded(forbidden: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(forbidden.shape[0] - 1, r1 + 1)
    cc1 = min(forbidden.shape[1] - 1, c1 + 1)
    forbidden[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _orientation_plan() -> list[str]:
    count = random.randint(*LINE_COUNT_RANGE)
    mode = random.choice(["h", "v", "mix"])
    if mode == "h":
        return ["h"] * count
    if mode == "v":
        return ["v"] * count
    plan = ["h", "v"]
    while len(plan) < count:
        plan.append(random.choice(["h", "v"]))
    random.shuffle(plan)
    return plan


def _place_lines(
    grid: np.ndarray,
    expected: np.ndarray,
    line_color: int,
    shift: int,
    band_c0: int,
    band_c1: int,
    row_start: int,
) -> list[int] | None:
    H, _W = grid.shape
    band_w = band_c1 - band_c0 + 1
    lower_h = H - row_start - 1
    orientations = _orientation_plan()
    max_gaps = [band_w - 4 if orient == "h" else lower_h - 4 for orient in orientations]
    if min(max_gaps) < 1 or max(max_gaps) <= shift:
        return None

    small_idx = random.randrange(len(orientations))
    large_choices = [i for i, max_gap in enumerate(max_gaps) if i != small_idx and max_gap > shift]
    if not large_choices:
        return None
    large_idx = random.choice(large_choices)

    chosen_gaps = []
    for idx, max_gap in enumerate(max_gaps):
        if idx == small_idx:
            chosen_gaps.append(random.randint(1, min(shift, max_gap)))
        elif idx == large_idx:
            chosen_gaps.append(random.randint(shift + 1, max_gap))
        else:
            chosen_gaps.append(random.randint(1, max_gap))

    forbidden = np.zeros_like(grid, dtype=bool)
    line_bboxes = []
    for idx, (orient, gap) in enumerate(zip(orientations, chosen_gaps)):
        candidates = []
        for frag_a in range(FRAG_LEN_RANGE[0], FRAG_LEN_RANGE[1] + 1):
            for frag_b in range(FRAG_LEN_RANGE[0], FRAG_LEN_RANGE[1] + 1):
                total = frag_a + gap + frag_b
                if orient == "h":
                    if total > band_w:
                        continue
                    for anchor in range(row_start, H - 1):
                        for start in range(band_c0, band_c1 - total + 2):
                            left, right, bbox = _line_cells(orient, anchor, start, frag_a, gap, frag_b)
                            if _expanded_ok(forbidden, bbox):
                                candidates.append((left, right, bbox))
                else:
                    if total > lower_h:
                        continue
                    for anchor in range(band_c0, band_c1 + 1):
                        for start in range(row_start, H - total):
                            left, right, bbox = _line_cells(orient, anchor, start, frag_a, gap, frag_b)
                            if _expanded_ok(forbidden, bbox):
                                candidates.append((left, right, bbox))
        if not candidates:
            return None

        left_cells, right_cells, bbox = random.choice(candidates)
        for r, c in left_cells + right_cells:
            grid[r, c] = int(line_color)
            expected[r, c] = int(line_color)
        if gap <= shift:
            if orientations[idx] == "h":
                expected[bbox[0], left_cells[-1][1] + 1 : right_cells[0][1]] = int(line_color)
            else:
                expected[left_cells[-1][0] + 1 : right_cells[0][0], bbox[1]] = int(line_color)
        _mark_expanded(forbidden, bbox)
        line_bboxes.append(bbox)

    gaps = sorted(chosen_gaps)
    assert gaps[0] <= shift < gaps[-1]
    return chosen_gaps


def _validate_generated(inp: np.ndarray, expected: np.ndarray, shift: int, gap_list: list[int]) -> bool:
    try:
        bg = int(get_bg(inp))
        obj_comp, obj_colors, obj_mask, split_row = _pick_object_component(inp, bg)
        if _object_shift(obj_mask, split_row) != shift:
            return False
        line_color_set = set(colors_present(inp, exclude={bg})) - set(obj_colors)
        if len(line_color_set) != 1:
            return False
        line_color = int(next(iter(line_color_set)))
        parsed_gaps = sorted(line["gap"] for line in _extract_broken_lines(inp, line_color, bg))
        if parsed_gaps != sorted(gap_list):
            return False
        if not (parsed_gaps[0] <= shift < parsed_gaps[-1]):
            return False
        out = main(inp)
        if not np.array_equal(out, expected):
            return False
        if np.array_equal(inp, out):
            return False
        if np.count_nonzero((out == line_color) & (inp != line_color)) == 0:
            return False
        if np.count_nonzero((expected == line_color) & (inp == BG)) == 0:
            return False
        return obj_comp["bbox"][2] < min(line["anchor"] if line["orientation"] == "h" else line["start"] for line in _extract_broken_lines(inp, line_color, bg))
    except Exception:
        return False


def _attempt_generate() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    color_a, color_b, line_color = random.sample(range(1, 10), 3)

    patch, shift = _build_object_patch(color_a, color_b)
    transformed_patch = _object_output_patch(patch, shift, color_a, color_b)

    side = random.choice(["left", "right"])
    obj_r0 = random.randint(1, 3)
    row_start = obj_r0 + patch.shape[0] + 4
    if row_start > H - 7:
        return None

    if side == "left":
        obj_c0 = 1
        band_c0 = obj_c0 + transformed_patch.shape[1] + 4
        band_c1 = W - 2
    else:
        band_c1 = W - transformed_patch.shape[1] - 5
        band_c0 = 1
        obj_c0 = band_c1 + 4
    if obj_c0 < 1 or obj_c0 + transformed_patch.shape[1] > W - 1:
        return None
    if band_c1 - band_c0 + 1 < 8:
        return None

    inp = np.zeros((H, W), dtype=int)
    expected = np.zeros((H, W), dtype=int)

    patch_mask = patch != BG
    rr, cc = np.where(patch_mask)
    inp[obj_r0 + rr, obj_c0 + cc] = patch[rr, cc]

    expected[obj_r0 : obj_r0 + transformed_patch.shape[0], obj_c0 : obj_c0 + transformed_patch.shape[1]] = transformed_patch

    gaps = _place_lines(inp, expected, line_color, shift, band_c0, band_c1, row_start)
    if gaps is None:
        return None
    if not _validate_generated(inp, expected, shift, gaps):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid split-slide-gated line-repair input")
