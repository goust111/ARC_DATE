from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists() and str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask


BG = Color.BLACK
FRAME = Color.GREY
CONTROL_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK)
MAX_ATTEMPTS = 400


def _bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def _box_area(box: tuple[int, int, int, int]) -> int:
    return _bbox_h(box) * _bbox_w(box)


def _contains(parent: tuple[int, int, int, int], child: tuple[int, int, int, int]) -> bool:
    return (
        parent[0] < child[0]
        and parent[1] < child[1]
        and child[2] < parent[2]
        and child[3] < parent[3]
    )


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _ring_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _detect_boxes(grid: np.ndarray) -> list[dict]:
    boxes: list[dict] = []
    for coords in components_mask(grid == FRAME, connectivity=4):
        box = bbox_coords(coords)
        h = _bbox_h(box)
        w = _bbox_w(box)
        assert h >= 3 and w >= 3, "frames must be at least 3x3"
        assert len(coords) == 2 * h + 2 * w - 4, "grey object must be a 1-cell rectangular ring"

        comp_mask = np.zeros((h, w), dtype=bool)
        rel = coords - np.array([[box[0], box[1]]], dtype=int)
        comp_mask[rel[:, 0], rel[:, 1]] = True
        ring = _ring_mask(h, w)
        assert np.array_equal(comp_mask, ring), "grey component must exactly match a rectangular ring"
        boxes.append({"bbox": box})

    assert boxes, "expected at least one grey frame"

    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes):
            if i >= j:
                continue
            box_a = a["bbox"]
            box_b = b["bbox"]
            if _bboxes_overlap(box_a, box_b) and not (_contains(box_a, box_b) or _contains(box_b, box_a)):
                raise AssertionError("frames must be disjoint or strictly nested")

    for box in boxes:
        containers = [other["bbox"] for other in boxes if _contains(other["bbox"], box["bbox"])]
        box["depth"] = len(containers)
    return boxes


def _markers_from_grid(grid: np.ndarray) -> dict[int, int]:
    assert np.all((grid[1:, :] == BG) | (grid[1:, :] == FRAME)), "only control pixels may use non-grey foreground colors"
    marker_cols = np.where((grid[0] != BG) & (grid[0] != FRAME))[0]
    assert len(marker_cols) > 0, "expected top-row control pixels"
    colors = [int(grid[0, c]) for c in marker_cols]
    assert len(colors) == len(set(colors)), "control colors must be distinct"
    return {int(c): int(grid[0, c]) for c in marker_cols}


def _select_marker_col(col_lo: int, col_hi: int, marker_cols: list[int], mode: str = "nearest") -> int:
    matches = [int(c) for c in marker_cols if col_lo <= c <= col_hi]
    assert matches, "each frame span must contain at least one control column"
    center2 = col_lo + col_hi
    if mode == "nearest":
        return min(matches, key=lambda c: (abs(2 * c - center2), c))
    if mode == "leftmost":
        return min(matches)
    if mode == "rightmost":
        return max(matches)
    raise ValueError(f"unknown marker selection mode: {mode}")


def _render_with_strategy(
    grid: np.ndarray,
    boxes: list[dict],
    markers: dict[int, int],
    *,
    marker_mode: str = "nearest",
    order_mode: str = "outer_first",
) -> np.ndarray:
    out = np.array(grid, copy=True)
    marker_cols = sorted(markers)
    reverse = order_mode == "inner_first"
    if order_mode not in {"outer_first", "inner_first"}:
        raise ValueError(f"unknown order mode: {order_mode}")

    order = sorted(range(len(boxes)), key=lambda i: (boxes[i]["depth"], _box_area(boxes[i]["bbox"])), reverse=reverse)
    for idx in order:
        r0, c0, r1, c1 = boxes[idx]["bbox"]
        fill_col = _select_marker_col(c0, c1, marker_cols, mode=marker_mode)
        color = markers[fill_col]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        mask = interior != FRAME
        interior[mask] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill nested hollow gray frames using the top-row control color nearest to each frame's center column.

    Input structure:
    - The grid is a black background with several gray 1-cell-thick hollow rectangular frames.
    - Some frames are strictly nested inside larger frames, while others may be separate.
    - The top row contains distinct single-pixel control markers in non-gray colors.

    Rule:
    - Detect every gray hollow rectangle and compute its horizontal span from left border to right border.
    - For each frame, collect all control pixels whose columns lie inside that span.
    - Choose the control pixel whose column is closest to the frame center column, breaking ties toward the smaller column.
    - Fill the frame interior with that control color from outer depth to inner depth, so deeper nested frames overwrite earlier outer fills while gray borders and the top-row controls stay unchanged.

    Key insight:
    - Column-span containment only identifies candidate controls; the actual fill color is resolved by center-column proximity, and nested depth determines overwrite priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert int(np.max(g)) <= 9 and int(np.min(g)) >= 0

    boxes = _detect_boxes(g)
    markers = _markers_from_grid(g)
    return _render_with_strategy(g, boxes, markers, marker_mode="nearest", order_mode="outer_first")


def _sample_group_lengths(total_boxes: int) -> list[int]:
    assert 3 <= total_boxes <= 5
    groups: list[int] = []
    remaining = total_boxes
    if remaining >= 3 and random.random() < 0.6:
        first = 3
    else:
        first = 2
    groups.append(first)
    remaining -= first

    while remaining > 0:
        if remaining >= 2 and random.random() < 0.3:
            groups.append(2)
            remaining -= 2
        else:
            groups.append(1)
            remaining -= 1
    random.shuffle(groups)
    return groups


def _build_chain(length: int) -> list[dict] | None:
    assert 1 <= length <= 3
    boxes = [{"r0": 0, "c0": 0, "h": random.randint(3, 5), "w": random.randint(3, 5)}]
    for _ in range(1, length):
        child_h = boxes[-1]["h"]
        child_w = boxes[-1]["w"]
        parent_h = child_h + 4 + random.randint(0, 3)
        parent_w = child_w + 4 + random.randint(0, 3)
        top = random.randint(2, parent_h - child_h - 2)
        left = random.randint(2, parent_w - child_w - 2)
        bottom = parent_h - child_h - top
        right = parent_w - child_w - left
        if min(top, left, bottom, right) < 2:
            return None
        for box in boxes:
            box["r0"] += top
            box["c0"] += left
        boxes.append({"r0": 0, "c0": 0, "h": parent_h, "w": parent_w})

    chain = list(reversed(boxes))
    for box in chain:
        box["r1"] = box["r0"] + box["h"] - 1
        box["c1"] = box["c0"] + box["w"] - 1
    return chain


def _candidate_cols_for_chain(chain: list[dict]) -> list[list[int]]:
    candidates: list[list[int]] = []
    for idx, box in enumerate(chain):
        cols = list(range(box["c0"], box["c1"] + 1))
        if idx < len(chain) - 1:
            child = chain[idx + 1]
            cols = [c for c in cols if not (child["c0"] <= c <= child["c1"])]
        else:
            interior = list(range(box["c0"] + 1, box["c1"]))
            cols = interior or cols
        candidates.append(cols)
    return candidates


def _choose_marker_columns(chain: list[dict], require_nearest_difference: bool) -> list[int] | None:
    candidate_lists = _candidate_cols_for_chain(chain)
    all_choices: list[tuple[int, int, int, tuple[int, ...]]] = []
    for combo in itertools.product(*candidate_lists):
        if len(set(combo)) != len(combo):
            continue

        chosen_cols: list[int] = []
        leftmost_diff = 0
        distinct_adjacent = 0
        own_hits = 0
        for idx, box in enumerate(chain):
            matches = [int(c) for c in combo if box["c0"] <= c <= box["c1"]]
            chosen = _select_marker_col(box["c0"], box["c1"], matches, mode="nearest")
            chosen_cols.append(chosen)
            leftmost_diff += int(chosen != min(matches))
            distinct_adjacent += int(idx > 0 and chosen != chosen_cols[idx - 1])
            own_hits += int(chosen == combo[idx])

        if require_nearest_difference and leftmost_diff == 0:
            continue
        score = (leftmost_diff, distinct_adjacent, own_hits)
        all_choices.append((score[0], score[1], score[2], tuple(int(c) for c in combo)))

    if not all_choices:
        return None

    best = max(choice[:3] for choice in all_choices)
    pool = [choice[3] for choice in all_choices if choice[:3] == best]
    return list(random.choice(pool))


def _place_groups(groups: list[list[dict]]) -> tuple[int, int, list[list[dict]]] | None:
    widths = [group[0]["w"] for group in groups]
    heights = [group[0]["h"] for group in groups]
    gaps = [random.randint(2, 4) for _ in range(len(groups) + 1)]
    width = gaps[0] + sum(widths) + sum(gaps[1:-1]) + gaps[-1]
    if width > 26:
        return None

    height = max(15, max(heights) + 4 + random.randint(0, 2))
    if height > 24:
        return None

    placed: list[list[dict]] = []
    cursor = gaps[0]
    for group, group_h, group_w, gap_after in zip(groups, heights, widths, gaps[1:]):
        top = random.randint(2, height - group_h - 1)
        shifted: list[dict] = []
        for box in group:
            shifted_box = {
                "bbox": (
                    int(top + box["r0"]),
                    int(cursor + box["c0"]),
                    int(top + box["r1"]),
                    int(cursor + box["c1"]),
                ),
            }
            shifted.append(shifted_box)
        placed.append(shifted)
        cursor += group_w + gap_after
    return height, width, placed


def _spec_markers(boxes: list[dict], markers: dict[int, int], mode: str = "nearest") -> dict[int, int]:
    out: dict[int, int] = {}
    marker_cols = sorted(markers)
    for idx, box in enumerate(boxes):
        c0 = box["bbox"][1]
        c1 = box["bbox"][3]
        out[idx] = markers[_select_marker_col(c0, c1, marker_cols, mode=mode)]
    return out


def _draw_input(height: int, width: int, boxes: list[dict], marker_pairs: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for box in boxes:
        r0, c0, r1, c1 = box["bbox"]
        grid[r0, c0 : c1 + 1] = FRAME
        grid[r1, c0 : c1 + 1] = FRAME
        grid[r0 : r1 + 1, c0] = FRAME
        grid[r0 : r1 + 1, c1] = FRAME
    for col, color in marker_pairs:
        grid[0, col] = color
    return grid


def _validate_spec(inp: np.ndarray, boxes: list[dict], markers: dict[int, int]) -> bool:
    try:
        detected = _detect_boxes(inp)
        if len(detected) != len(boxes):
            return False
    except Exception:
        return False

    for box in boxes:
        box["depth"] = sum(_contains(other["bbox"], box["bbox"]) for other in boxes if other is not box)

    marker_cols = sorted(markers)
    leaf_boxes = [box for box in boxes if not any(_contains(box["bbox"], other["bbox"]) for other in boxes if other is not box)]
    if not leaf_boxes:
        return False

    multi_match = False
    for box in boxes:
        c0 = box["bbox"][1]
        c1 = box["bbox"][3]
        matches = [c for c in marker_cols if c0 <= c <= c1]
        if not matches:
            return False
        if len(matches) > 1:
            multi_match = True
    if not multi_match:
        return False

    for leaf in leaf_boxes:
        c0 = leaf["bbox"][1]
        c1 = leaf["bbox"][3]
        matches = [c for c in marker_cols if c0 <= c <= c1]
        if len(matches) != 1:
            return False

    expected = _render_with_strategy(inp, boxes, markers, marker_mode="nearest", order_mode="outer_first")
    leftmost = _render_with_strategy(inp, boxes, markers, marker_mode="leftmost", order_mode="outer_first")
    inner_first = _render_with_strategy(inp, boxes, markers, marker_mode="nearest", order_mode="inner_first")
    if np.array_equal(expected, leftmost):
        return False
    if np.array_equal(expected, inner_first):
        return False
    if len(set(_spec_markers(boxes, markers, mode="nearest").values())) < 2:
        return False

    try:
        return bool(np.array_equal(main(inp), expected))
    except Exception:
        return False


def _attempt() -> np.ndarray | None:
    total_boxes = random.randint(3, 5)
    group_lengths = _sample_group_lengths(total_boxes)
    local_groups: list[list[dict]] = []

    want_nearest_difference = True
    for length in group_lengths:
        chain = _build_chain(length)
        if chain is None:
            return None
        marker_cols = _choose_marker_columns(chain, require_nearest_difference=want_nearest_difference and length >= 2)
        if marker_cols is None:
            return None
        for box, marker_col in zip(chain, marker_cols):
            box["marker_col_local"] = int(marker_col)
        local_groups.append(chain)
        if length >= 2:
            want_nearest_difference = False

    placement = _place_groups(local_groups)
    if placement is None:
        return None
    height, width, placed_groups = placement

    boxes: list[dict] = []
    marker_cols_global: list[int] = []
    for local_group, placed_group in zip(local_groups, placed_groups):
        for local_box, placed_box in zip(local_group, placed_group):
            global_col = placed_box["bbox"][1] + (local_box["marker_col_local"] - local_box["c0"])
            placed_box["marker_col"] = int(global_col)
            boxes.append(placed_box)
            marker_cols_global.append(int(global_col))

    if len(set(marker_cols_global)) != len(marker_cols_global):
        return None

    colors = random.sample(CONTROL_COLORS, len(boxes))
    marker_pairs = sorted((box["marker_col"], color) for box, color in zip(boxes, colors))
    markers = {int(col): int(color) for col, color in marker_pairs}
    inp = _draw_input(height, width, boxes, marker_pairs)

    if not _validate_spec(inp, boxes, markers):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame ARC input")
