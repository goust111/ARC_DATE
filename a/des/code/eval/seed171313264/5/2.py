from __future__ import annotations

from collections import deque
import random
import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)

MAX_GENERATION_ATTEMPTS = 400
PAIR_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 250
GRID_MIN = 18
GRID_MAX = 30
GAP_CHOICES = (2, 3, 4, 5, 6)
PAIR_COUNT_CHOICES = (2, 3)
SHARED_SPAN_RANGE = (2, 4)
SEGMENT_RANGE = (2, 4)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    assert rows.size and cols.size
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _find_components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(g[r0, c0]) != color:
                continue

            seen[r0, c0] = True
            q = deque([(r0, c0)])
            coords: list[tuple[int, int]] = []
            min_r = max_r = r0
            min_c = max_c = c0

            while q:
                r, c = q.popleft()
                coords.append((r, c))
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(g[rr, cc]) == color:
                        seen[rr, cc] = True
                        q.append((rr, cc))

            mask = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=bool)
            for r, c in coords:
                mask[r - min_r, c - min_c] = True
            comps.append(
                {
                    "bbox": (int(min_r), int(min_c), int(max_r), int(max_c)),
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )

    return comps


def _pair_orientation(a: dict, b: dict) -> str:
    same_rows = a["bbox"][0] == b["bbox"][0] and a["bbox"][2] == b["bbox"][2]
    same_cols = a["bbox"][1] == b["bbox"][1] and a["bbox"][3] == b["bbox"][3]
    assert same_rows ^ same_cols, "same-color fragments must align in exactly one axis"
    return "H" if same_rows else "V"


def _merge_binary_masks(mask_a: np.ndarray, mask_b: np.ndarray, orientation: str) -> np.ndarray:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)

    if orientation == "H":
        assert a.shape[0] == b.shape[0]
        out = np.zeros((a.shape[0], a.shape[1] + b.shape[1]), dtype=bool)
        out[:, : a.shape[1]] = a
        out[:, a.shape[1] :] = b
        return out

    assert orientation == "V"
    assert a.shape[1] == b.shape[1]
    out = np.zeros((a.shape[0] + b.shape[0], a.shape[1]), dtype=bool)
    out[: a.shape[0], :] = a
    out[a.shape[0] :, :] = b
    return out


def _colorize(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[np.asarray(mask, dtype=bool)] = int(color)
    return out


def _stack_centered(shapes: list[np.ndarray], bg: int) -> np.ndarray:
    assert shapes
    out_h = sum(int(shape.shape[0]) for shape in shapes) + len(shapes) - 1
    out_w = max(int(shape.shape[1]) for shape in shapes)
    out = np.full((out_h, out_w), bg, dtype=int)

    top = 0
    for shape in shapes:
        h, w = map(int, shape.shape)
        left = (out_w - w) // 2
        patch = out[top : top + h, left : left + w]
        mask = shape != bg
        patch[mask] = shape[mask]
        top += h + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair fragments by color, slide each pair together along its shared axis,
    then stack the merged shapes from largest area to smallest area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one foreground color"

    merged: list[tuple[int, tuple[int, int], np.ndarray]] = []
    for color in colors:
        comps = _find_components_of_color(g, color)
        assert len(comps) == 2, "each foreground color must form exactly two fragments"
        orientation = _pair_orientation(comps[0], comps[1])

        if orientation == "H":
            left, right = sorted(comps, key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
            merged_mask = _merge_binary_masks(left["mask"], right["mask"], "H")
        else:
            top, bottom = sorted(comps, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
            merged_mask = _merge_binary_masks(top["mask"], bottom["mask"], "V")

        anchor = (
            min(int(comps[0]["bbox"][0]), int(comps[1]["bbox"][0])),
            min(int(comps[0]["bbox"][1]), int(comps[1]["bbox"][1])),
        )
        merged.append((int(merged_mask.sum()), anchor, _colorize(merged_mask, color, bg)))

    merged.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))
    return _stack_centered([shape for _, _, shape in merged], bg)


def _choose_interval(width: int, prev: tuple[int, int] | None, force_side: str | None) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for start in range(width):
        for end in range(start, width):
            if prev is not None and (end < prev[0] or start > prev[1]):
                continue
            if force_side == "left" and start != 0:
                continue
            if force_side == "right" and end != width - 1:
                continue
            candidates.append((start, end))
    assert candidates
    return random.choice(candidates)


def _build_interval_fragment(height: int, width: int, seam_row: int, touch_side: str) -> np.ndarray:
    intervals: list[tuple[int, int] | None] = [None] * height
    intervals[seam_row] = _choose_interval(width, None, touch_side)

    for r in range(seam_row - 1, -1, -1):
        intervals[r] = _choose_interval(width, intervals[r + 1], None)
    for r in range(seam_row + 1, height):
        intervals[r] = _choose_interval(width, intervals[r - 1], None)

    mask = np.zeros((height, width), dtype=bool)
    for r, interval in enumerate(intervals):
        start, end = interval
        mask[r, start : end + 1] = True
    return _crop_mask(mask)


def _random_pair_masks(orientation: str) -> tuple[np.ndarray, np.ndarray]:
    shared = random.randint(*SHARED_SPAN_RANGE)
    seg_a = random.randint(*SEGMENT_RANGE)
    seg_b = random.randint(*SEGMENT_RANGE)
    seam_index = random.randrange(shared)

    left = _build_interval_fragment(shared, seg_a, seam_index, "right")
    right = _build_interval_fragment(shared, seg_b, seam_index, "left")
    if orientation == "H":
        return left, right
    assert orientation == "V"
    return left.T, right.T


def _sample_orientations(count: int) -> list[str]:
    assert count in PAIR_COUNT_CHOICES
    if count == 2:
        orientations = ["H", "V"]
    else:
        orientations = ["H", "V", random.choice(["H", "V"])]
    random.shuffle(orientations)
    return orientations


def _make_pair(color: int, orientation: str) -> dict | None:
    for _ in range(PAIR_ATTEMPTS):
        mask_a, mask_b = _random_pair_masks(orientation)
        merged_mask = _merge_binary_masks(mask_a, mask_b, orientation)
        gap = random.choice(GAP_CHOICES)
        if orientation == "H":
            input_h = int(mask_a.shape[0])
            input_w = int(mask_a.shape[1] + gap + mask_b.shape[1])
        else:
            input_h = int(mask_a.shape[0] + gap + mask_b.shape[0])
            input_w = int(mask_a.shape[1])
        return {
            "color": int(color),
            "orientation": orientation,
            "a": mask_a,
            "b": mask_b,
            "merged": merged_mask,
            "area": int(merged_mask.sum()),
            "gap": int(gap),
            "input_h": input_h,
            "input_w": input_w,
        }
    return None


def _expected_output(pairs: list[dict]) -> np.ndarray:
    ordered = sorted(pairs, key=lambda pair: -int(pair["area"]))
    shapes = [_colorize(pair["merged"], int(pair["color"]), BG) for pair in ordered]
    return _stack_centered(shapes, BG)


def _expand_bbox(bbox: tuple[int, int, int, int], pad: int, height: int, width: int) -> tuple[int, int, int, int]:
    return (
        max(0, int(bbox[0]) - pad),
        max(0, int(bbox[1]) - pad),
        min(height - 1, int(bbox[2]) + pad),
        min(width - 1, int(bbox[3]) + pad),
    )


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _paint_mask(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = map(int, mask.shape)
    patch = grid[top : top + h, left : left + w]
    assert np.all(patch[mask] == BG)
    patch[mask] = int(color)


def _choose_grid_shape(pairs: list[dict], out_h: int, out_w: int) -> tuple[int, int] | None:
    max_pair_h = max(int(pair["input_h"]) for pair in pairs)
    max_pair_w = max(int(pair["input_w"]) for pair in pairs)
    reserved_area = sum((int(pair["input_h"]) + 2) * (int(pair["input_w"]) + 2) for pair in pairs)
    side_hint = int(np.ceil(np.sqrt(reserved_area))) + 1

    min_h = max(GRID_MIN, out_h + 1, max_pair_h + 2, side_hint)
    min_w = max(GRID_MIN, out_w + 1, max_pair_w + 2, side_hint)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None
    return random.randint(min_h, GRID_MAX), random.randint(min_w, GRID_MAX)


def _place_pairs(pairs: list[dict], height: int, width: int) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    reserved: list[tuple[int, int, int, int]] = []
    order = sorted(
        pairs,
        key=lambda pair: max(int(pair["input_h"]), int(pair["input_w"])),
        reverse=True,
    )

    for pair in order:
        placed = False
        color = int(pair["color"])
        gap = int(pair["gap"])
        a = pair["a"]
        b = pair["b"]

        for _ in range(PLACEMENT_ATTEMPTS):
            if pair["orientation"] == "H":
                h = int(a.shape[0])
                total_w = int(a.shape[1] + gap + b.shape[1])
                if h > height or total_w > width:
                    break
                top = random.randint(0, height - h)
                left = random.randint(0, width - total_w)
                bbox = (top, left, top + h - 1, left + total_w - 1)
                expanded = _expand_bbox(bbox, 1, height, width)
                if any(_intersects(expanded, other) for other in reserved):
                    continue
                _paint_mask(grid, a, color, top, left)
                _paint_mask(grid, b, color, top, left + int(a.shape[1]) + gap)
            else:
                w = int(a.shape[1])
                total_h = int(a.shape[0] + gap + b.shape[0])
                if total_h > height or w > width:
                    break
                top = random.randint(0, height - total_h)
                left = random.randint(0, width - w)
                bbox = (top, left, top + total_h - 1, left + w - 1)
                expanded = _expand_bbox(bbox, 1, height, width)
                if any(_intersects(expanded, other) for other in reserved):
                    continue
                _paint_mask(grid, a, color, top, left)
                _paint_mask(grid, b, color, top + int(a.shape[0]) + gap, left)

            reserved.append(expanded)
            placed = True
            break

        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    colors = random.sample(PALETTE, pair_count)
    orientations = _sample_orientations(pair_count)

    pairs: list[dict] = []
    used_areas: set[int] = set()
    for color, orientation in zip(colors, orientations):
        pair = _make_pair(color, orientation)
        if pair is None or pair["area"] in used_areas:
            return None
        used_areas.add(int(pair["area"]))
        pairs.append(pair)

    expected = _expected_output(pairs)
    out_h, out_w = map(int, expected.shape)
    shape = _choose_grid_shape(pairs, out_h, out_w)
    if shape is None:
        return None

    grid = _place_pairs(pairs, shape[0], shape[1])
    if grid is None:
        return None

    actual = main(grid)
    if actual.shape[0] >= grid.shape[0] or actual.shape[1] >= grid.shape[1]:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
