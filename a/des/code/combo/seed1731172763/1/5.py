from __future__ import annotations

import random

import numpy as np


BG = 0


def _draw_border(grid: np.ndarray, top: int, left: int, bottom: int, right: int, color: int) -> None:
    grid[top, left : right + 1] = int(color)
    grid[bottom, left : right + 1] = int(color)
    grid[top : bottom + 1, left] = int(color)
    grid[top : bottom + 1, right] = int(color)


def _is_outer_marker_cell(r: int, c: int, top: int, left: int, bottom: int, right: int) -> bool:
    return (
        (r == top - 1 and left <= c <= right)
        or (r == bottom + 1 and left <= c <= right)
        or (c == left - 1 and top <= r <= bottom)
        or (c == right + 1 and top <= r <= bottom)
    )


def _border_size(top: int, left: int, bottom: int, right: int) -> int:
    height = int(bottom - top + 1)
    width = int(right - left + 1)
    return 2 * height + 2 * width - 4


def _find_frame_for_color(grid: np.ndarray, color: int) -> dict:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0, "each frame color must appear at least once"

    rows = coords[:, 0]
    cols = coords[:, 1]
    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())
    coord_set = {(int(r), int(c)) for r, c in coords}

    candidates: list[dict] = []
    for top in (rmin, rmin + 1):
        for left in (cmin, cmin + 1):
            for bottom in (rmax, rmax - 1):
                for right in (cmax, cmax - 1):
                    if bottom - top + 1 < 3 or right - left + 1 < 3:
                        continue
                    if top < 0 or left < 0 or bottom >= grid.shape[0] or right >= grid.shape[1]:
                        continue
                    if not np.all(grid[top, left : right + 1] == int(color)):
                        continue
                    if not np.all(grid[bottom, left : right + 1] == int(color)):
                        continue
                    if not np.all(grid[top : bottom + 1, left] == int(color)):
                        continue
                    if not np.all(grid[top : bottom + 1, right] == int(color)):
                        continue

                    valid = True
                    marker_count = 0
                    for r, c in coord_set:
                        on_border = (
                            (r == top and left <= c <= right)
                            or (r == bottom and left <= c <= right)
                            or (c == left and top <= r <= bottom)
                            or (c == right and top <= r <= bottom)
                        )
                        if on_border:
                            continue
                        if not _is_outer_marker_cell(r, c, top, left, bottom, right):
                            valid = False
                            break
                        marker_count += 1
                    if not valid:
                        continue

                    candidates.append(
                        {
                            "color": int(color),
                            "bbox": (int(top), int(left), int(bottom), int(right)),
                            "marker_count": int(marker_count),
                            "area": int((bottom - top + 1) * (right - left + 1)),
                            "border_cells": int(_border_size(top, left, bottom, right)),
                        }
                    )

    assert candidates, f"failed to recover a frame for color {color}"
    best_area = max(candidate["area"] for candidate in candidates)
    best = [candidate for candidate in candidates if candidate["area"] == best_area]
    assert len(best) == 1, f"frame for color {color} is ambiguous"

    frame = best[0]
    assert frame["border_cells"] + frame["marker_count"] == int(coords.shape[0])
    return frame


def _sorted_frames(grid: np.ndarray) -> list[dict]:
    colors = [int(color) for color in np.unique(grid) if int(color) != BG]
    assert colors, "expected at least one colored frame"

    frames = [_find_frame_for_color(grid, color) for color in colors]
    frames.sort(key=lambda frame: -frame["area"])

    for outer, inner in zip(frames, frames[1:]):
        ot, ol, ob, or_ = outer["bbox"]
        it, il, ib, ir = inner["bbox"]
        assert ot < it and ol < il and ib < ob and ir < or_, "frames must be strictly nested"

    return frames


def _render_output(frames: list[dict]) -> np.ndarray:
    outer_top, outer_left, _outer_bottom, outer_right = frames[0]["bbox"]
    del outer_top
    width = int(outer_right - outer_left + 1)
    total_height = sum(int(frame["marker_count"]) + 1 for frame in frames) + max(0, len(frames) - 1)
    out = np.full((total_height, width), BG, dtype=int)

    row = 0
    for idx, frame in enumerate(frames):
        top, left, _bottom, right = frame["bbox"]
        del top
        band_height = int(frame["marker_count"]) + 1
        band_bottom = row + band_height - 1
        rel_left = int(left - outer_left)
        rel_right = int(right - outer_left)
        _draw_border(out, row, rel_left, band_bottom, rel_right, frame["color"])
        row = band_bottom + 1
        if idx + 1 < len(frames):
            row += 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split a stack of nested rectangular frames into separate output bands.

    Input structure:
    - Background is black.
    - Each non-black color contributes one single-pixel-thick rectangular frame.
    - Same-colored extra pixels may sit just outside the frame and touch it by one
      orthogonal step; those pixels are markers, not extra frames.

    Rule:
    - Recover every frame's true rectangle and sort the frames from outermost to
      innermost.
    - Count the external same-color markers of each frame; if a frame has k markers,
      its output band has height k + 1.
    - Draw one border per band, preserving the frame's original width and its
      horizontal span relative to the outer frame. Leave one black row between bands.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert (g >= 0).all(), "grid colors must be non-negative"
    assert BG in g, "expected black background"

    frames = _sorted_frames(g)
    return _render_output(frames)


# --------------- generator (o5x) ---------------

COLOR_POOL = tuple(range(1, 10))
DEPTH_RANGE = (2, 5)
INNER_HEIGHT_RANGE = (5, 7)
INNER_WIDTH_RANGE = (5, 8)
INSET_RANGE = (2, 3)
OUTER_MARGIN_RANGE = (1, 4)
MARKER_CAP = 4
MAX_CANVAS_SIZE = 35
MAX_GENERATION_ATTEMPTS = 400


def _sample_structure() -> list[tuple[int, int, int, int]] | None:
    depth = random.randint(*DEPTH_RANGE)
    inner_height = random.randint(*INNER_HEIGHT_RANGE)
    inner_width = random.randint(*INNER_WIDTH_RANGE)
    insets = [
        tuple(random.randint(*INSET_RANGE) for _ in range(4))
        for _ in range(depth - 1)
    ]

    outer_height = inner_height + sum(int(top + bottom) for top, bottom, _left, _right in insets)
    outer_width = inner_width + sum(int(left + right) for _top, _bottom, left, right in insets)

    margin_top = random.randint(*OUTER_MARGIN_RANGE)
    margin_bottom = random.randint(*OUTER_MARGIN_RANGE)
    margin_left = random.randint(*OUTER_MARGIN_RANGE)
    margin_right = random.randint(*OUTER_MARGIN_RANGE)

    height = outer_height + margin_top + margin_bottom
    width = outer_width + margin_left + margin_right
    if height > MAX_CANVAS_SIZE or width > MAX_CANVAS_SIZE:
        return None

    frames = [(margin_top, margin_left, margin_top + outer_height - 1, margin_left + outer_width - 1)]
    for top, bottom, left, right in insets:
        prev_top, prev_left, prev_bottom, prev_right = frames[-1]
        frames.append(
            (
                int(prev_top + top),
                int(prev_left + left),
                int(prev_bottom - bottom),
                int(prev_right - right),
            )
        )

    return frames


def _outer_ring_cells(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    top, left, bottom, right = bbox
    cells: list[tuple[int, int]] = []

    if top > 0:
        for c in range(left, right + 1):
            if int(grid[top - 1, c]) == BG:
                cells.append((int(top - 1), int(c)))
    if bottom + 1 < grid.shape[0]:
        for c in range(left, right + 1):
            if int(grid[bottom + 1, c]) == BG:
                cells.append((int(bottom + 1), int(c)))
    if left > 0:
        for r in range(top, bottom + 1):
            if int(grid[r, left - 1]) == BG:
                cells.append((int(r), int(left - 1)))
    if right + 1 < grid.shape[1]:
        for r in range(top, bottom + 1):
            if int(grid[r, right + 1]) == BG:
                cells.append((int(r), int(right + 1)))

    return cells


def _sample_marker_counts(capacities: list[int]) -> list[int]:
    counts = [random.randint(0, capacity) for capacity in capacities]
    if all(count == 0 for count in counts):
        idx = random.randrange(len(counts))
        counts[idx] = random.randint(1, capacities[idx])
    if all(count > 0 for count in counts) and len(counts) >= 2 and random.random() < 0.5:
        counts[random.randrange(len(counts))] = 0
    return counts


def _build_expected(specs: list[dict]) -> np.ndarray:
    ordered = sorted(specs, key=lambda spec: -spec["area"])
    return _render_output(ordered)


def _gen_attempt() -> np.ndarray | None:
    bboxes = _sample_structure()
    if bboxes is None:
        return None

    colors = random.sample(COLOR_POOL, len(bboxes))
    height = max(bottom for _top, _left, bottom, _right in bboxes) + 1 + random.randint(*OUTER_MARGIN_RANGE)
    width = max(right for _top, _left, _bottom, right in bboxes) + 1 + random.randint(*OUTER_MARGIN_RANGE)
    if height > MAX_CANVAS_SIZE or width > MAX_CANVAS_SIZE:
        return None

    grid = np.full((height, width), BG, dtype=int)
    specs: list[dict] = []
    for color, bbox in zip(colors, bboxes):
        _draw_border(grid, *bbox, color)
        top, left, bottom, right = bbox
        specs.append(
            {
                "color": int(color),
                "bbox": tuple(map(int, bbox)),
                "marker_count": 0,
                "area": int((bottom - top + 1) * (right - left + 1)),
            }
        )

    ring_cells = [_outer_ring_cells(grid, bbox) for bbox in bboxes]
    capacities = [min(MARKER_CAP, len(cells)) for cells in ring_cells]
    if any(capacity == 0 for capacity in capacities):
        return None

    marker_counts = _sample_marker_counts(capacities)
    for spec, color, cells, marker_count in zip(specs, colors, ring_cells, marker_counts):
        for r, c in random.sample(cells, marker_count):
            grid[r, c] = int(color)
        spec["marker_count"] = int(marker_count)

    expected = _build_expected(specs)
    try:
        produced = main(grid)
    except Exception:
        return None
    if np.array_equal(produced, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid nested-frame input")
