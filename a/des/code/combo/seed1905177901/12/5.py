from __future__ import annotations

from functools import lru_cache
import random

import numpy as np


def _components_4(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coordinate arrays."""
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(np.asarray(coords, dtype=int))

    return components


def _extract_rectangles(grid: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    """Extract monochrome solid-rectangle foreground objects."""
    colors = [int(color) for color in np.unique(grid) if int(color) != 0]
    assert len(colors) == 1, "expected exactly one foreground color"
    color = colors[0]

    rects: list[dict[str, int]] = []
    for coords in _components_4(grid == color):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        area = int(coords.shape[0])
        assert area == height * width, "each object must be a solid rectangle"
        assert bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color))
        rects.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "h": height,
                "w": width,
            }
        )

    assert len(rects) >= 2, "expected a sequence of at least two rectangles"
    return color, rects


def _constant(values: list[int]) -> bool:
    return len(set(values)) == 1


def _positive_step(values: list[int]) -> int | None:
    if len(values) < 2:
        return None
    diffs = [int(b - a) for a, b in zip(values, values[1:])]
    if any(diff <= 0 for diff in diffs):
        return None
    if len(set(diffs)) != 1:
        return None
    return diffs[0]


def _horizontal_candidate(rects: list[dict[str, int]]) -> dict[str, object] | None:
    ordered = sorted(rects, key=lambda rect: (rect["c0"], rect["r0"]))
    spacing = _positive_step([rect["c0"] for rect in ordered])
    if spacing is None:
        return None

    top_aligned = _constant([rect["r0"] for rect in ordered])
    bottom_aligned = _constant([rect["r1"] for rect in ordered])
    if not top_aligned and not bottom_aligned:
        return None

    widths = [rect["w"] for rect in ordered]
    heights = [rect["h"] for rect in ordered]
    width_step = _positive_step(widths)
    height_step = _positive_step(heights)

    if _constant(heights) and width_step is not None:
        return {
            "axis": "horizontal",
            "grow": "width",
            "align": "top" if top_aligned else "bottom",
            "spacing": spacing,
            "delta": width_step,
            "ordered": ordered,
        }
    if _constant(widths) and height_step is not None:
        return {
            "axis": "horizontal",
            "grow": "height",
            "align": "top" if top_aligned else "bottom",
            "spacing": spacing,
            "delta": height_step,
            "ordered": ordered,
        }
    return None


def _vertical_candidate(rects: list[dict[str, int]]) -> dict[str, object] | None:
    ordered = sorted(rects, key=lambda rect: (rect["r0"], rect["c0"]))
    spacing = _positive_step([rect["r0"] for rect in ordered])
    if spacing is None:
        return None

    left_aligned = _constant([rect["c0"] for rect in ordered])
    right_aligned = _constant([rect["c1"] for rect in ordered])
    if not left_aligned and not right_aligned:
        return None

    widths = [rect["w"] for rect in ordered]
    heights = [rect["h"] for rect in ordered]
    width_step = _positive_step(widths)
    height_step = _positive_step(heights)

    if _constant(heights) and width_step is not None:
        return {
            "axis": "vertical",
            "grow": "width",
            "align": "left" if left_aligned else "right",
            "spacing": spacing,
            "delta": width_step,
            "ordered": ordered,
        }
    if _constant(widths) and height_step is not None:
        return {
            "axis": "vertical",
            "grow": "height",
            "align": "left" if left_aligned else "right",
            "spacing": spacing,
            "delta": height_step,
            "ordered": ordered,
        }
    return None


def _infer_sequence(rects: list[dict[str, int]]) -> dict[str, object]:
    candidates = [cand for cand in (_horizontal_candidate(rects), _vertical_candidate(rects)) if cand is not None]
    assert len(candidates) == 1, "sequence orientation should be uniquely determined"

    seq = candidates[0]
    delta = int(seq["delta"])
    assert 1 <= delta <= 3, "expected a 1-to-3 cell growth step"
    ordered = list(seq["ordered"])
    last = ordered[-1]

    new_h = int(last["h"])
    new_w = int(last["w"])
    if seq["grow"] == "width":
        new_w += delta
    else:
        new_h += delta

    if seq["axis"] == "horizontal":
        c0 = int(last["c0"]) + int(seq["spacing"])
        c1 = c0 + new_w - 1
        if seq["align"] == "top":
            r0 = int(ordered[0]["r0"])
            r1 = r0 + new_h - 1
        else:
            r1 = int(ordered[0]["r1"])
            r0 = r1 - new_h + 1
    else:
        r0 = int(last["r0"]) + int(seq["spacing"])
        r1 = r0 + new_h - 1
        if seq["align"] == "left":
            c0 = int(ordered[0]["c0"])
            c1 = c0 + new_w - 1
        else:
            c1 = int(ordered[0]["c1"])
            c0 = c1 - new_w + 1

    seq["next_bbox"] = (r0, c0, r1, c1)
    return seq


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend an equally spaced increasing rectangle sequence by one object.

    Input structure:
    - Black background.
    - Several same-color 4-connected foreground objects.
    - The objects are solid rectangles laid out horizontally or vertically.
    - One rectangle dimension stays fixed while the other grows by a constant
      1-to-3 cell step.

    Rule:
    - Infer the ordering direction, spacing, growth dimension, and growth step.
    - Draw the next rectangle at the next equally spaced position, keeping the
      same color and alignment.
    - Preserve all existing objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    color, rects = _extract_rectangles(g)
    seq = _infer_sequence(rects)
    r0, c0, r1, c1 = map(int, seq["next_bbox"])

    height, width = g.shape
    assert 0 <= r0 <= r1 < height
    assert 0 <= c0 <= c1 < width
    assert bool(np.all(g[r0 : r1 + 1, c0 : c1 + 1] == 0)), "next rectangle must fit in empty space"

    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


# --------------- generator (o3) ---------------

BG = 0
COLORS = tuple(range(1, 10))
AXES = ("horizontal", "vertical")
GROW_DIMS = ("width", "height")
ALIGN_BY_AXIS = {
    "horizontal": ("top", "bottom"),
    "vertical": ("left", "right"),
}

COUNT_RANGE = (3, 5)
BASE_LENGTH_RANGE = (1, 5)
DELTA_RANGE = (1, 3)
GAP_RANGE = (1, 4)
MARGIN_RANGE = (0, 4)
GRID_SIDE_MAX = 40
LOCAL_SIDE_MAX = GRID_SIDE_MAX - 2 * MARGIN_RANGE[1]
GENERATION_RETRIES = 400


def _sequence_lengths(count: int, base: int, delta: int) -> list[int]:
    return [base + delta * i for i in range(count + 1)]


def _local_bboxes(
    axis: str,
    grow: str,
    align: str,
    count: int,
    base: int,
    delta: int,
    spacing: int,
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
    lengths = _sequence_lengths(count, base, delta)
    max_length = max(lengths)
    bboxes: list[tuple[int, int, int, int]] = []

    if axis == "horizontal":
        for idx, length in enumerate(lengths):
            h = length if grow == "height" else 1
            w = length if grow == "width" else 1
            c0 = idx * spacing
            c1 = c0 + w - 1
            if align == "top":
                r0 = 0
                r1 = h - 1
            else:
                r1 = max_length - 1
                r0 = r1 - h + 1
            bboxes.append((r0, c0, r1, c1))
    else:
        for idx, length in enumerate(lengths):
            h = length if grow == "height" else 1
            w = length if grow == "width" else 1
            r0 = idx * spacing
            r1 = r0 + h - 1
            if align == "left":
                c0 = 0
                c1 = w - 1
            else:
                c1 = max_length - 1
                c0 = c1 - w + 1
            bboxes.append((r0, c0, r1, c1))

    local_h = max(r1 for _, _, r1, _ in bboxes) + 1
    local_w = max(c1 for _, _, _, c1 in bboxes) + 1

    return bboxes, (local_h, local_w)


def _paint_rectangles(
    shape: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
    color: int,
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    return grid


def _translate_rects(
    rects: list[tuple[int, int, int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int, int, int]]:
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in rects]


@lru_cache(maxsize=None)
def _feasible_options(axis: str, grow: str) -> tuple[tuple[int, int, int, int], ...]:
    options: list[tuple[int, int, int, int]] = []
    for count in range(COUNT_RANGE[0], COUNT_RANGE[1] + 1):
        for base in range(BASE_LENGTH_RANGE[0], BASE_LENGTH_RANGE[1] + 1):
            for delta in range(DELTA_RANGE[0], DELTA_RANGE[1] + 1):
                for gap in range(GAP_RANGE[0], GAP_RANGE[1] + 1):
                    lengths = _sequence_lengths(count, base, delta)
                    if axis == "horizontal":
                        max_axis_size = max(lengths[:-1]) if grow == "width" else 1
                    else:
                        max_axis_size = max(lengths[:-1]) if grow == "height" else 1
                    spacing = max_axis_size + gap + 1
                    _, (local_h, local_w) = _local_bboxes(axis, grow, ALIGN_BY_AXIS[axis][0], count, base, delta, spacing)
                    if local_h <= LOCAL_SIDE_MAX and local_w <= LOCAL_SIDE_MAX:
                        options.append((count, base, delta, gap))

    assert options, "each branch should have at least one feasible parameter set"
    return tuple(options)


def _gen_attempt() -> np.ndarray | None:
    axis = random.choice(AXES)
    grow = random.choice(GROW_DIMS)
    align = random.choice(ALIGN_BY_AXIS[axis])
    count, base, delta, gap = random.choice(_feasible_options(axis, grow))
    lengths = _sequence_lengths(count, base, delta)

    if axis == "horizontal":
        max_axis_size = max(lengths[:-1]) if grow == "width" else 1
    else:
        max_axis_size = max(lengths[:-1]) if grow == "height" else 1
    spacing = max_axis_size + gap + 1

    rects_all, (local_h, local_w) = _local_bboxes(axis, grow, align, count, base, delta, spacing)
    top_margin = random.randint(*MARGIN_RANGE)
    left_margin = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)
    full_h = local_h + top_margin + bottom_margin
    full_w = local_w + left_margin + right_margin
    assert full_h <= GRID_SIDE_MAX and full_w <= GRID_SIDE_MAX

    color = random.choice(COLORS)
    shifted = _translate_rects(rects_all, top_margin, left_margin)
    inp_rects = shifted[:-1]
    out_rects = shifted
    inp = _paint_rectangles((full_h, full_w), inp_rects, color)
    expected = _paint_rectangles((full_h, full_w), out_rects, color)

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid increasing-sequence input")
