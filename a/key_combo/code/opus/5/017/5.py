from __future__ import annotations

from collections import Counter
import random

import numpy as np
from scipy.ndimage import binary_dilation, label


STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
STRUCT8 = np.ones((3, 3), dtype=int)

BG = 0
PATH_COLOR = 7
RECT_COLORS = [1, 2, 3, 4, 6, 8, 9]
SHAPES = [(1, 1), (1, 2), (2, 1), (2, 2)]
MAX_GEN_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def label_mask(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    struct = STRUCT4 if int(connectivity) == 4 else STRUCT8
    lab, n = label(mask.astype(bool), structure=struct)
    return lab, int(n)


def find_components_of_color(grid: np.ndarray, color: int, connectivity: int = 4) -> list[dict]:
    comps = []
    labels, n = label_mask(grid == int(color), connectivity=connectivity)
    for cid in range(1, n + 1):
        coords = np.argwhere(labels == cid)
        comps.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return comps


def solid_rectangle_box(grid: np.ndarray, color: int, component: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, component["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert int(component["area"]) == h * w, "each non-path object must be a solid rectangle"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color))
    return r0, c0, r1, c1


def facing_corners(
    upper: tuple[int, int, int, int],
    lower: tuple[int, int, int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    _ur0, uc0, ur1, uc1 = upper
    lr0, lc0, _lr1, lc1 = lower
    assert ur1 < lr0, "paired rectangles must be vertically separated"

    if uc1 < lc0:
        return (ur1, uc1), (lr0, lc0)
    if lc1 < uc0:
        return (ur1, uc0), (lr0, lc1)
    raise AssertionError("paired rectangles must be diagonally separated")


def compute_channel(path_mask: np.ndarray) -> np.ndarray:
    labels, n_paths = label_mask(path_mask, connectivity=4)
    assert n_paths == 2, "expected exactly two orange path components"
    channel = np.ones_like(path_mask, dtype=bool)
    for cid in range(1, n_paths + 1):
        channel &= binary_dilation(labels == cid, structure=STRUCT8)
    channel &= ~path_mask
    assert np.any(channel), "expected a non-empty overlap channel"
    return channel


def union_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw only the channel-filtered parts of diagonal connectors between same-color rectangle pairs.

    Input structure:
    - The background is black.
    - Orange (7) forms exactly two 4-connected one-cell-wide path components that run roughly downward with a narrow gap between them.
    - Every other foreground color appears as exactly two separated solid rectangles of size 1x1 to 2x2, with the lower rectangle placed diagonally from the upper rectangle.

    Rule:
    - Dilate each orange path component by one cell in the 8-neighborhood, then intersect the two dilations to obtain the middle channel.
    - For each non-orange color, locate the upper and lower rectangle and take the 45-degree diagonal from the upper rectangle's downward-facing corner to the lower rectangle's upward-facing corner.
    - Paint only the open diagonal cells that lie inside the channel; keep rectangle cells, path cells, and channel cells outside those diagonals unchanged.
    - If two colors want to paint the same channel cell, keep the color whose two-rectangle union bounding box has smaller area.

    Key insight:
    - The diagonal connection is not drawn directly from the rectangle geometry alone: the orange paths first create a derived channel mask, and that mask filters which diagonal pixels become visible.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    path_mask = g == PATH_COLOR
    channel = compute_channel(path_mask)
    out = g.copy()

    pair_infos = []
    for color in sorted(set(map(int, g.ravel())) - {bg, PATH_COLOR}):
        components = find_components_of_color(g, color, connectivity=4)
        assert len(components) == 2, "each rectangle color must appear as exactly two components"

        rectangles = [solid_rectangle_box(g, color, comp) for comp in components]
        rectangles.sort(key=lambda box: (box[0], box[1]))
        upper, lower = rectangles
        start, end = facing_corners(upper, lower)

        row_delta = int(end[0] - start[0])
        col_delta = int(end[1] - start[1])
        assert row_delta > 0 and abs(row_delta) == abs(col_delta), "corner pair must define a 45-degree diagonal"
        dc = 1 if col_delta > 0 else -1

        visible = []
        r = int(start[0] + 1)
        c = int(start[1] + dc)
        while (r, c) != end:
            if bool(channel[r, c]) and int(g[r, c]) == bg:
                visible.append((r, c))
            r += 1
            c += dc

        pair_infos.append(
            {
                "color": int(color),
                "bbox_area": bbox_area(union_bbox(upper, lower)),
                "visible": visible,
            }
        )

    for info in sorted(pair_infos, key=lambda item: int(item["bbox_area"])):
        color = int(info["color"])
        for r, c in info["visible"]:
            if int(out[r, c]) == bg:
                out[r, c] = color
    return out


def _partition_total(total: int, count: int, minimum: int, maximum: int) -> list[int] | None:
    if count * minimum > total or count * maximum < total:
        return None
    out = []
    remaining = int(total)
    for i in range(count - 1):
        rest = count - i - 1
        lo = max(minimum, remaining - maximum * rest)
        hi = min(maximum, remaining - minimum * rest)
        if lo > hi:
            return None
        val = random.randint(lo, hi)
        out.append(val)
        remaining -= val
    out.append(remaining)
    return out


def _cells_to_mask(shape: tuple[int, int], cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if cells:
        rows, cols = zip(*cells)
        mask[rows, cols] = True
    return mask


def _buffered_cells(shape: tuple[int, int], cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    mask = _cells_to_mask(shape, cells)
    dilated = binary_dilation(mask, structure=STRUCT8)
    return set(map(tuple, np.argwhere(dilated)))


def _walk_segment(path: list[tuple[int, int]], dr: int, dc: int, steps: int) -> None:
    r, c = path[-1]
    for _ in range(steps):
        r += dr
        c += dc
        path.append((r, c))


def _gen_directions(hdir: int, turns: int) -> list[tuple[int, int]]:
    directions = [(1, 0)]
    for i in range(turns):
        directions.append((0, hdir) if i % 2 == 0 else (1, 0))
    return directions


def _left_normal(dr: int, dc: int) -> tuple[int, int]:
    return -dc, dr


def _wall_sets(channel_path: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    left = set()
    right = set()
    channel_cells = set(channel_path)
    for i, (r, c) in enumerate(channel_path):
        dirs = []
        if i > 0:
            pr, pc = channel_path[i - 1]
            dirs.append((r - pr, c - pc))
        if i + 1 < len(channel_path):
            nr, nc = channel_path[i + 1]
            dirs.append((nr - r, nc - c))

        for dr, dc in dirs:
            lr, lc = _left_normal(dr, dc)
            if (r + lr, c + lc) not in channel_cells:
                left.add((r + lr, c + lc))
            if (r - lr, c - lc) not in channel_cells:
                right.add((r - lr, c - lc))

        if len(dirs) == 2 and dirs[0] != dirs[1]:
            for side in (1, -1):
                offsets = []
                for dr, dc in dirs:
                    lr, lc = _left_normal(dr, dc)
                    offsets.append((side * lr, side * lc))
                direct = [(r + dr0, c + dc0) for dr0, dc0 in offsets]
                if all(cell not in channel_cells for cell in direct):
                    er = r + offsets[0][0] + offsets[1][0]
                    ec = c + offsets[0][1] + offsets[1][1]
                    if side == 1:
                        left.add((er, ec))
                    else:
                        right.add((er, ec))
    return left, right


def _is_thin_component(mask: np.ndarray) -> bool:
    labels, n = label_mask(mask, connectivity=4)
    if n != 1:
        return False
    cells = set(map(tuple, np.argwhere(labels == 1)))
    for r, c in cells:
        degree = sum((rr, cc) in cells for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
        if degree > 2:
            return False
    return True


def _generate_channel_path(h: int, w: int) -> list[tuple[int, int]] | None:
    total_down = h - 3
    if total_down < 10 or w < 16:
        return None

    turns = random.randint(3, 5)
    hdir = random.choice([-1, 1])
    directions = _gen_directions(hdir, turns)
    v_count = sum(1 for dr, dc in directions if (dr, dc) == (1, 0))
    h_count = sum(1 for dr, dc in directions if dr == 0)

    v_steps = _partition_total(total_down, v_count, 2, max(6, total_down - 2 * (v_count - 1)))
    if v_steps is None:
        return None
    h_steps = [random.randint(1, 3) for _ in range(h_count)]
    total_h = sum(h_steps)

    min_col = 3
    max_col = w - 4
    if hdir == 1:
        if min_col + total_h > max_col:
            return None
        start_col = random.randint(min_col, max_col - total_h)
    else:
        if min_col > max_col - total_h:
            return None
        start_col = random.randint(min_col + total_h, max_col)

    path = [(1, start_col)]
    vi = 0
    hi = 0
    for dr, dc in directions:
        steps = v_steps[vi] if dr else h_steps[hi]
        if dr:
            vi += 1
        else:
            hi += 1
        _walk_segment(path, dr, dc, steps)

    if any(r < 1 or r > h - 2 or c < min_col or c > max_col for r, c in path):
        return None
    if len({c for _r, c in path}) < 4:
        return None
    return path


def _build_paths(h: int, w: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]], np.ndarray] | None:
    channel_path = _generate_channel_path(h, w)
    if channel_path is None:
        return None

    left, right = _wall_sets(channel_path)
    channel_cells = set(channel_path)
    if left & right or left & channel_cells or right & channel_cells:
        return None
    if any(r < 0 or r >= h or c < 0 or c >= w for r, c in left | right):
        return None

    left_mask = _cells_to_mask((h, w), left)
    right_mask = _cells_to_mask((h, w), right)
    if not _is_thin_component(left_mask) or not _is_thin_component(right_mask):
        return None

    path_mask = left_mask | right_mask
    channel = compute_channel(path_mask)
    if int(channel.sum()) < max(8, h // 2):
        return None
    if len({c for _r, c in np.argwhere(channel)}) < 4:
        return None
    return left, right, channel


def _pair_candidate(
    h: int,
    w: int,
    shape: tuple[int, int],
    direction: int,
    gap: int,
    ur: int,
    uc: int,
) -> dict | None:
    rh, rw = shape
    upper = (ur, uc, ur + rh - 1, uc + rw - 1)
    if direction == 1:
        start = (upper[2], upper[3])
        end = (start[0] + gap + 1, start[1] + gap + 1)
        lower = (end[0], end[1], end[0] + rh - 1, end[1] + rw - 1)
    else:
        start = (upper[2], upper[1])
        end = (start[0] + gap + 1, start[1] - (gap + 1))
        lower = (end[0], end[1] - rw + 1, end[0] + rh - 1, end[1])

    if min(*upper, *lower) < 0:
        return None
    if max(upper[2], lower[2]) >= h or max(upper[3], lower[3]) >= w:
        return None

    diag = [(start[0] + step, start[1] + direction * step) for step in range(1, gap + 1)]
    box_cells = {
        (r, c)
        for r in range(upper[0], upper[2] + 1)
        for c in range(upper[1], upper[3] + 1)
    } | {
        (r, c)
        for r in range(lower[0], lower[2] + 1)
        for c in range(lower[1], lower[3] + 1)
    }
    return {
        "upper": upper,
        "lower": lower,
        "diag": diag,
        "diag_set": set(diag),
        "box_cells": box_cells,
        "bbox_area": bbox_area(union_bbox(upper, lower)),
        "shape": shape,
        "direction": int(direction),
    }


def _classify_visible(diag: list[tuple[int, int]], channel: np.ndarray) -> tuple[list[tuple[int, int]], str]:
    visible_idx = [idx for idx, (r, c) in enumerate(diag) if bool(channel[r, c])]
    visible = [diag[idx] for idx in visible_idx]
    if not visible:
        return [], "outside"
    if len(visible) == len(diag):
        return visible, "full"
    if any(b - a > 1 for a, b in zip(visible_idx, visible_idx[1:])):
        return visible, "partial_discontinuous"
    return visible, "partial_contiguous"


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(ar0 - br1 - 1, br0 - ar1 - 1, 0)
    col_gap = max(ac0 - bc1 - 1, bc0 - ac1 - 1, 0)
    return int(max(row_gap, col_gap))


def _enumerate_candidates(channel: np.ndarray, path_mask: np.ndarray) -> list[dict]:
    h, w = channel.shape
    reserved = path_mask | channel
    path_cells = set(map(tuple, np.argwhere(path_mask)))
    out = []
    for shape in SHAPES:
        for direction in (-1, 1):
            for gap in range(2, min(h, w) - 4):
                for ur in range(0, h - shape[0] + 1):
                    for uc in range(0, w - shape[1] + 1):
                        cand = _pair_candidate(h, w, shape, direction, gap, ur, uc)
                        if cand is None:
                            continue
                        if any(bool(reserved[r, c]) for r, c in cand["box_cells"]):
                            continue
                        if cand["diag_set"] & path_cells:
                            continue
                        visible, kind = _classify_visible(cand["diag"], channel)
                        cand["visible"] = visible
                        cand["visible_set"] = set(visible)
                        cand["kind"] = kind
                        if kind == "partial_discontinuous" and len(visible) < 2:
                            continue
                        out.append(cand)
    random.shuffle(out)
    return out


def _compatible(a: dict, b: dict) -> bool:
    if a["box_cells"] & b["box_cells"]:
        return False
    for box_a in (a["upper"], a["lower"]):
        for box_b in (b["upper"], b["lower"]):
            if _bbox_gap(box_a, box_b) < 1:
                return False
    if a["diag_set"] & b["box_cells"]:
        return False
    if b["diag_set"] & a["box_cells"]:
        return False
    return True


def _choose_candidates(candidates: list[dict]) -> tuple[list[dict], dict] | None:
    partials = [cand for cand in candidates if cand["kind"] == "partial_discontinuous"]
    outsides = [cand for cand in candidates if cand["kind"] == "outside"]
    extras = [cand for cand in candidates if cand["kind"] in {"full", "partial_contiguous", "partial_discontinuous"}]

    if not partials or not outsides:
        return None

    for primary in partials:
        overlap_pool = [
            cand
            for cand in extras
            if cand is not primary
            and _compatible(primary, cand)
            and primary["visible_set"] & cand["visible_set"]
            and int(primary["bbox_area"]) != int(cand["bbox_area"])
        ]
        random.shuffle(overlap_pool)
        for overlap in overlap_pool:
            outside_pool = [
                cand
                for cand in outsides
                if cand is not primary
                and cand is not overlap
                and _compatible(primary, cand)
                and _compatible(overlap, cand)
            ]
            if not outside_pool:
                continue
            outside = random.choice(outside_pool)
            selected = [primary, overlap, outside]

            visible_union = set().union(*(cand["visible_set"] for cand in selected))
            extra_pool = [
                cand
                for cand in candidates
                if cand not in selected
                and all(_compatible(cand, other) for other in selected)
                and not (cand["visible_set"] & visible_union)
            ]
            chosen_extra = None
            visible_extra = [cand for cand in extra_pool if cand["kind"] in {"full", "partial_contiguous"} and cand["visible"]]
            if visible_extra and random.random() < 0.65:
                chosen_extra = random.choice(visible_extra)
            elif extra_pool and random.random() < 0.45:
                chosen_extra = random.choice(extra_pool)
            if chosen_extra is not None:
                selected.append(chosen_extra)

            smaller = min(primary, overlap, key=lambda cand: int(cand["bbox_area"]))
            larger = overlap if smaller is primary else primary
            overlap_cells = primary["visible_set"] & overlap["visible_set"]
            return selected, {
                "partial": primary,
                "outside": outside,
                "smaller_overlap": smaller,
                "larger_overlap": larger,
                "overlap_cells": overlap_cells,
            }
    return None


def _paint_boxes(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _descending_lines(shape: tuple[int, int]) -> list[list[tuple[int, int]]]:
    h, w = shape
    lines = []
    for c0 in range(w):
        line = []
        r, c = 0, c0
        while 0 <= r < h and 0 <= c < w:
            line.append((r, c))
            r += 1
            c += 1
        lines.append(line)
    for r0 in range(1, h):
        line = []
        r, c = r0, 0
        while 0 <= r < h and 0 <= c < w:
            line.append((r, c))
            r += 1
            c += 1
        lines.append(line)

    for c0 in range(w):
        line = []
        r, c = 0, c0
        while 0 <= r < h and 0 <= c < w:
            line.append((r, c))
            r += 1
            c -= 1
        lines.append(line)
    for r0 in range(1, h):
        line = []
        r, c = r0, w - 1
        while 0 <= r < h and 0 <= c < w:
            line.append((r, c))
            r += 1
            c -= 1
        lines.append(line)
    random.shuffle(lines)
    return lines


def _contiguous_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _make_single_cell_pair(line: list[tuple[int, int]], start_idx: int, end_idx: int) -> dict:
    assert start_idx < end_idx
    upper = line[start_idx]
    lower = line[end_idx]
    diag = line[start_idx + 1 : end_idx]
    return {
        "upper": (upper[0], upper[1], upper[0], upper[1]),
        "lower": (lower[0], lower[1], lower[0], lower[1]),
        "box_cells": {upper, lower},
        "diag": diag,
        "diag_set": set(diag),
        "bbox_area": bbox_area(union_bbox((upper[0], upper[1], upper[0], upper[1]), (lower[0], lower[1], lower[0], lower[1]))),
    }


def _find_primary_and_overlap(channel: np.ndarray, path_mask: np.ndarray) -> tuple[dict, dict, tuple[int, int], set[tuple[int, int]]] | None:
    path_cells = set(map(tuple, np.argwhere(path_mask)))
    for line in _descending_lines(channel.shape):
        hit_indices = [idx for idx, cell in enumerate(line) if bool(channel[cell])]
        for run_start, run_end in _contiguous_runs(hit_indices):
            if run_end - run_start + 1 < 4:
                continue
            if run_start < 2 or run_end + 2 >= len(line):
                continue
            a = _make_single_cell_pair(line, run_start - 2, run_end + 2)
            if a["box_cells"] & path_cells:
                continue

            run_indices = list(range(run_start, run_end + 1))
            block_idx = run_indices[len(run_indices) // 2]
            overlap_choices = [idx for idx in run_indices if idx != block_idx]
            random.shuffle(overlap_choices)
            for overlap_idx in overlap_choices:
                for left_pad, right_pad in ((1, 2), (2, 1), (2, 2), (1, 3), (3, 1)):
                    c_start = overlap_idx - left_pad
                    c_end = overlap_idx + right_pad
                    if c_start < 0 or c_end >= len(line) or c_start >= c_end:
                        continue
                    c = _make_single_cell_pair(line, c_start, c_end)
                    if c["box_cells"] & path_cells:
                        continue
                    if a["box_cells"] & c["box_cells"]:
                        continue
                    if line[block_idx] in c["box_cells"]:
                        continue
                    overlap_cells = (set(a["diag"]) & set(c["diag"]) & set(line[run_start : run_end + 1])) - {line[block_idx]}
                    if not overlap_cells:
                        continue
                    left_remaining = [
                        idx for idx in run_indices if idx < block_idx and line[idx] not in overlap_cells
                    ]
                    right_remaining = [
                        idx for idx in run_indices if idx > block_idx and line[idx] not in overlap_cells
                    ]
                    if left_remaining and right_remaining:
                        return a, c, line[block_idx], overlap_cells
    return None


def _find_occluder_pair(
    target: tuple[int, int],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    channel: np.ndarray,
    path_mask: np.ndarray,
) -> dict | None:
    h, w = shape
    tr, tc = target
    for direction in (-1, 1):
        for gap in (2, 3, 4, 5):
            end = (tr + gap, tc + direction * gap)
            if not (0 <= end[0] < h and 0 <= end[1] < w):
                continue
            diag = [(tr + step, tc + direction * step) for step in range(1, gap)]
            if any(bool(channel[r, c]) for r, c in diag):
                continue
            if bool(path_mask[end]) or end in occupied:
                continue
            pair = {
                "upper": (tr, tc, tr, tc),
                "lower": (end[0], end[1], end[0], end[1]),
                "box_cells": {target, end},
                "diag": diag,
                "diag_set": set(diag),
                "bbox_area": bbox_area(union_bbox((tr, tc, tr, tc), (end[0], end[1], end[0], end[1]))),
            }
            return pair
    return None


def _expected_output(inp: np.ndarray, selected: list[dict], channel: np.ndarray) -> np.ndarray:
    out = inp.copy()
    claims: dict[tuple[int, int], tuple[int, int, int]] = {}
    for order, cand in enumerate(selected):
        for r, c in cand["diag"]:
            if not bool(channel[r, c]) or int(inp[r, c]) != BG:
                continue
            key = (int(cand["bbox_area"]), int(order), int(cand["color"]))
            current = claims.get((r, c))
            if current is None or key < current:
                claims[(r, c)] = key
    for (r, c), (_area, _order, color) in claims.items():
        out[r, c] = int(color)
    return out


def _validate_generated(inp: np.ndarray, expected: np.ndarray, selected: list[dict], info: dict) -> bool:
    if not np.array_equal(main(inp), expected):
        return False
    if np.array_equal(inp, expected):
        return False

    path_labels, n_paths = label_mask(inp == PATH_COLOR, connectivity=4)
    if n_paths != 2:
        return False
    del path_labels

    partial = info["partial"]
    outside = info["outside"]
    overlap_cells = info["overlap_cells"]
    if not overlap_cells:
        return False

    painted = [idx for idx, cell in enumerate(partial["diag"]) if int(expected[cell]) == int(partial["color"])]
    if not painted or len(painted) == len(partial["diag"]):
        return False
    if any(bool(info["channel"][r, c]) and int(inp[r, c]) == BG for r, c in outside["diag"]):
        return False

    winner = info["smaller_overlap"]
    loser = info["larger_overlap"]
    if int(winner["bbox_area"]) >= int(loser["bbox_area"]):
        return False
    sample_cell = min(overlap_cells)
    if int(expected[sample_cell]) != int(winner["color"]):
        return False

    colors = [int(cand["color"]) for cand in selected]
    if len(colors) != len(set(colors)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 26)

    built = _build_paths(h, w)
    if built is None:
        return None
    left, right, channel = built
    path_mask = _cells_to_mask((h, w), left | right)
    chosen = _find_primary_and_overlap(channel, path_mask)
    if chosen is None:
        return None
    primary, overlap, blocker_cell, overlap_cells = chosen
    occupied = set().union(primary["box_cells"], overlap["box_cells"], left, right)
    outside = _find_occluder_pair(blocker_cell, (h, w), occupied, channel, path_mask)
    if outside is None:
        return None

    selected = [primary, overlap, outside]
    colors = random.sample(RECT_COLORS, len(selected))
    inp = np.full((h, w), BG, dtype=int)
    for r, c in left | right:
        inp[r, c] = PATH_COLOR
    for cand, color in zip(selected, colors):
        cand["color"] = int(color)
        _paint_boxes(inp, cand["upper"], color)
        _paint_boxes(inp, cand["lower"], color)

    expected = _expected_output(inp, selected, channel)
    smaller, larger = sorted((primary, overlap), key=lambda item: int(item["bbox_area"]))
    info = {
        "partial": primary,
        "outside": outside,
        "smaller_overlap": smaller,
        "larger_overlap": larger,
        "overlap_cells": overlap_cells,
        "channel": channel,
    }
    if not _validate_generated(inp, expected, selected, info):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 5")
