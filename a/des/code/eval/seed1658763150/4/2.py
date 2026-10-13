from __future__ import annotations

import random

import numpy as np


BG = 0
BRIDGE = 5
SEPARATOR = 9
OBJECT_COLORS = tuple(color for color in range(10) if color not in {BG, BRIDGE, SEPARATOR})

GRID_MIN = 18
GRID_MAX = 30
OUTER_MARGIN = 1
PAIR_PADDING = 1
GEN_RETRIES = 400

SEGMENT_RANGE = (2, 4)
STRIPE_RANGE = (2, 4)
RECT_H_RANGE = (2, 5)
RECT_W_RANGE = (2, 5)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _count_grey_runs(values: np.ndarray) -> int:
    runs = 0
    in_run = False
    for value in np.asarray(values, dtype=int).tolist():
        if value == BRIDGE:
            if not in_run:
                runs += 1
                in_run = True
        else:
            in_run = False
    assert runs > 0, "bridge must contain at least one grey segment"
    return runs


def _rectangle_info(grid: np.ndarray, coords: np.ndarray) -> dict[str, int]:
    color = int(grid[tuple(coords[0])])
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    area = int(len(coords))
    assert area == (r1 - r0 + 1) * (c1 - c0 + 1), "object must be a solid rectangle"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be monochrome"
    return {"color": color, "r0": r0, "c0": c0, "r1": r1, "c1": c1}


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    height, width = g.shape
    out = g.copy()
    out[g == SEPARATOR] = BG

    rect_mask = (g != BG) & (g != BRIDGE) & (g != SEPARATOR)
    rects: list[dict[str, int]] = []
    rect_id = np.full((height, width), -1, dtype=int)
    for coords in _components(rect_mask):
        info = _rectangle_info(g, coords)
        idx = len(rects)
        rects.append(info)
        rect_id[info["r0"] : info["r1"] + 1, info["c0"] : info["c1"] + 1] = idx
    assert rects, "expected at least one rectangle pair"

    line_mask = (g == BRIDGE) | (g == SEPARATOR)
    for coords in _components(line_mask):
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))

        horizontal = r0 == r1 and c0 < c1
        vertical = c0 == c1 and r0 < r1
        if r0 == r1 and c0 == c1:
            horizontal = c0 > 0 and c0 + 1 < width and rect_id[r0, c0 - 1] >= 0 and rect_id[r0, c0 + 1] >= 0
            vertical = r0 > 0 and r0 + 1 < height and rect_id[r0 - 1, c0] >= 0 and rect_id[r0 + 1, c0] >= 0
        assert horizontal ^ vertical, "bridge must be a straight horizontal or vertical line"

        if horizontal:
            assert np.all(np.isin(g[r0, c0 : c1 + 1], (BRIDGE, SEPARATOR))), "bridge row must be contiguous"
            left_idx = int(rect_id[r0, c0 - 1]) if c0 > 0 else -1
            right_idx = int(rect_id[r0, c1 + 1]) if c1 + 1 < width else -1
            assert left_idx >= 0 and right_idx >= 0, "horizontal bridge must touch rectangles at both ends"
            left_rect = rects[left_idx]
            right_rect = rects[right_idx]
            assert left_rect["color"] == right_rect["color"], "bridge endpoints must connect same-color rectangles"
            assert left_rect["r0"] == right_rect["r0"] and left_rect["r1"] == right_rect["r1"], "horizontal pair must share row span"
            assert left_rect["c1"] + 1 == c0 and right_rect["c0"] - 1 == c1, "bridge must exactly span the gap"

            parts = _count_grey_runs(g[r0, c0 : c1 + 1])
            gap = c1 - c0 + 1
            assert gap % parts == 0, "gap must split evenly into stripes"
            stripe = gap // parts
            for idx in range(parts):
                value = left_rect["color"] if idx % 2 == 0 else BG
                s0 = c0 + idx * stripe
                s1 = s0 + stripe
                out[left_rect["r0"] : left_rect["r1"] + 1, s0:s1] = value
        else:
            assert np.all(np.isin(g[r0 : r1 + 1, c0], (BRIDGE, SEPARATOR))), "bridge column must be contiguous"
            top_idx = int(rect_id[r0 - 1, c0]) if r0 > 0 else -1
            bottom_idx = int(rect_id[r1 + 1, c0]) if r1 + 1 < height else -1
            assert top_idx >= 0 and bottom_idx >= 0, "vertical bridge must touch rectangles at both ends"
            top_rect = rects[top_idx]
            bottom_rect = rects[bottom_idx]
            assert top_rect["color"] == bottom_rect["color"], "bridge endpoints must connect same-color rectangles"
            assert top_rect["c0"] == bottom_rect["c0"] and top_rect["c1"] == bottom_rect["c1"], "vertical pair must share column span"
            assert top_rect["r1"] + 1 == r0 and bottom_rect["r0"] - 1 == r1, "bridge must exactly span the gap"

            parts = _count_grey_runs(g[r0 : r1 + 1, c0])
            gap = r1 - r0 + 1
            assert gap % parts == 0, "gap must split evenly into stripes"
            stripe = gap // parts
            for idx in range(parts):
                value = top_rect["color"] if idx % 2 == 0 else BG
                s0 = r0 + idx * stripe
                s1 = s0 + stripe
                out[s0:s1, top_rect["c0"] : top_rect["c1"] + 1] = value

    out[g == BRIDGE] = BRIDGE
    out[g == SEPARATOR] = BG
    return out


def _random_composition(total: int, parts: int) -> list[int]:
    assert parts >= 1
    assert total >= parts
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    result: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        result.append(cut - prev)
        prev = cut
    return result


def _bridge_pattern(length: int, segments: int) -> np.ndarray:
    grey_total = length - (segments - 1)
    assert grey_total >= segments, "need positive grey length for every segment"
    grey_lengths = _random_composition(grey_total, segments)
    values: list[int] = []
    for idx, grey_len in enumerate(grey_lengths):
        values.extend([BRIDGE] * grey_len)
        if idx + 1 < segments:
            values.append(SEPARATOR)
    return np.array(values, dtype=int)


def _horizontal_choices(height: int, width: int) -> list[tuple[int, int, int, int]]:
    choices: list[tuple[int, int, int, int]] = []
    max_h = min(RECT_H_RANGE[1], height - 2 * OUTER_MARGIN)
    max_w = min(RECT_W_RANGE[1], (width - 2 * OUTER_MARGIN) // 2)
    for rect_h in range(RECT_H_RANGE[0], max_h + 1):
        for rect_w in range(RECT_W_RANGE[0], max_w + 1):
            for segments in range(SEGMENT_RANGE[0], SEGMENT_RANGE[1] + 1):
                for stripe in range(STRIPE_RANGE[0], STRIPE_RANGE[1] + 1):
                    gap = segments * stripe
                    total_w = 2 * rect_w + gap
                    if total_w <= width - 2 * OUTER_MARGIN:
                        choices.append((rect_h, rect_w, segments, stripe))
    return choices


def _vertical_choices(height: int, width: int) -> list[tuple[int, int, int, int]]:
    choices: list[tuple[int, int, int, int]] = []
    max_h = min(RECT_H_RANGE[1], (height - 2 * OUTER_MARGIN) // 2)
    max_w = min(RECT_W_RANGE[1], width - 2 * OUTER_MARGIN)
    for rect_h in range(RECT_H_RANGE[0], max_h + 1):
        for rect_w in range(RECT_W_RANGE[0], max_w + 1):
            for segments in range(SEGMENT_RANGE[0], SEGMENT_RANGE[1] + 1):
                for stripe in range(STRIPE_RANGE[0], STRIPE_RANGE[1] + 1):
                    gap = segments * stripe
                    total_h = 2 * rect_h + gap
                    if total_h <= height - 2 * OUTER_MARGIN:
                        choices.append((rect_h, rect_w, segments, stripe))
    return choices


def _sample_pair(color: int, height: int, width: int) -> dict[str, object] | None:
    h_choices = _horizontal_choices(height, width)
    v_choices = _vertical_choices(height, width)
    orientations = []
    if h_choices:
        orientations.append("h")
    if v_choices:
        orientations.append("v")
    if not orientations:
        return None

    orientation = random.choice(orientations)
    if orientation == "h":
        rect_h, rect_w, segments, stripe = random.choice(h_choices)
        gap = segments * stripe
        return {
            "orientation": "h",
            "color": int(color),
            "rect_h": rect_h,
            "rect_w": rect_w,
            "segments": segments,
            "stripe": stripe,
            "gap": gap,
            "bridge_offset": random.randrange(rect_h),
            "pattern": _bridge_pattern(gap, segments),
            "total_h": rect_h,
            "total_w": 2 * rect_w + gap,
        }

    rect_h, rect_w, segments, stripe = random.choice(v_choices)
    gap = segments * stripe
    return {
        "orientation": "v",
        "color": int(color),
        "rect_h": rect_h,
        "rect_w": rect_w,
        "segments": segments,
        "stripe": stripe,
        "gap": gap,
        "bridge_offset": random.randrange(rect_w),
        "pattern": _bridge_pattern(gap, segments),
        "total_h": 2 * rect_h + gap,
        "total_w": rect_w,
    }


def _can_place(reserved: np.ndarray, top: int, left: int, box_h: int, box_w: int) -> bool:
    height, width = reserved.shape
    r0 = max(0, top - PAIR_PADDING)
    c0 = max(0, left - PAIR_PADDING)
    r1 = min(height, top + box_h + PAIR_PADDING)
    c1 = min(width, left + box_w + PAIR_PADDING)
    return not reserved[r0:r1, c0:c1].any()


def _reserve(reserved: np.ndarray, top: int, left: int, box_h: int, box_w: int) -> None:
    height, width = reserved.shape
    r0 = max(0, top - PAIR_PADDING)
    c0 = max(0, left - PAIR_PADDING)
    r1 = min(height, top + box_h + PAIR_PADDING)
    c1 = min(width, left + box_w + PAIR_PADDING)
    reserved[r0:r1, c0:c1] = True


def _place_spec(spec: dict[str, object], reserved: np.ndarray) -> tuple[int, int] | None:
    height, width = reserved.shape
    box_h = int(spec["total_h"])
    box_w = int(spec["total_w"])
    candidates: list[tuple[int, int]] = []
    for top in range(OUTER_MARGIN, height - OUTER_MARGIN - box_h + 1):
        for left in range(OUTER_MARGIN, width - OUTER_MARGIN - box_w + 1):
            if _can_place(reserved, top, left, box_h, box_w):
                candidates.append((top, left))
    if not candidates:
        return None
    return random.choice(candidates)


def _draw_pair(grid: np.ndarray, spec: dict[str, object]) -> None:
    color = int(spec["color"])
    top = int(spec["top"])
    left = int(spec["left"])
    rect_h = int(spec["rect_h"])
    rect_w = int(spec["rect_w"])
    gap = int(spec["gap"])
    offset = int(spec["bridge_offset"])
    pattern = np.asarray(spec["pattern"], dtype=int)

    if spec["orientation"] == "h":
        grid[top : top + rect_h, left : left + rect_w] = color
        right_left = left + rect_w + gap
        grid[top : top + rect_h, right_left : right_left + rect_w] = color
        row = top + offset
        grid[row, left + rect_w : left + rect_w + gap] = pattern
        return

    grid[top : top + rect_h, left : left + rect_w] = color
    bottom_top = top + rect_h + gap
    grid[bottom_top : bottom_top + rect_h, left : left + rect_w] = color
    col = left + offset
    grid[top + rect_h : top + rect_h + gap, col] = pattern


def _render_expected(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == SEPARATOR] = BG

    for spec in specs:
        color = int(spec["color"])
        top = int(spec["top"])
        left = int(spec["left"])
        rect_h = int(spec["rect_h"])
        rect_w = int(spec["rect_w"])
        gap = int(spec["gap"])
        stripe = int(spec["stripe"])
        offset = int(spec["bridge_offset"])
        pattern = np.asarray(spec["pattern"], dtype=int)

        if spec["orientation"] == "h":
            gap_left = left + rect_w
            for idx in range(int(spec["segments"])):
                value = color if idx % 2 == 0 else BG
                s0 = gap_left + idx * stripe
                s1 = s0 + stripe
                out[top : top + rect_h, s0:s1] = value
            row = top + offset
            out[row, gap_left : gap_left + gap] = np.where(pattern == BRIDGE, BRIDGE, BG)
            continue

        gap_top = top + rect_h
        for idx in range(int(spec["segments"])):
            value = color if idx % 2 == 0 else BG
            s0 = gap_top + idx * stripe
            s1 = s0 + stripe
            out[s0:s1, left : left + rect_w] = value
        col = left + offset
        out[gap_top : gap_top + gap, col] = np.where(pattern == BRIDGE, BRIDGE, BG)

    return out


def _sample_pair_count(height: int, width: int) -> int:
    area = height * width
    if area < 430:
        max_pairs = 2
    elif area < 620:
        max_pairs = 3
    else:
        max_pairs = 4
    return random.randint(1, max_pairs)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    pair_count = _sample_pair_count(height, width)
    colors = random.sample(OBJECT_COLORS, pair_count)

    specs: list[dict[str, object]] = []
    for color in colors:
        spec = _sample_pair(color, height, width)
        if spec is None:
            return None
        specs.append(spec)

    reserved = np.zeros((height, width), dtype=bool)
    for spec in sorted(specs, key=lambda item: int(item["total_h"]) * int(item["total_w"]), reverse=True):
        position = _place_spec(spec, reserved)
        if position is None:
            return None
        spec["top"], spec["left"] = map(int, position)
        _reserve(reserved, int(spec["top"]), int(spec["left"]), int(spec["total_h"]), int(spec["total_w"]))

    grid = np.full((height, width), BG, dtype=int)
    for spec in specs:
        _draw_pair(grid, spec)

    expected = _render_expected(grid, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
