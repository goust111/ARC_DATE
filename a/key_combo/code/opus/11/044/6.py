from __future__ import annotations

import math
import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
GRID_HEIGHT_RANGE = (12, 18)
GRID_WIDTH_RANGE = (12, 18)
FRAME_COUNT_CHOICES = (3, 4, 5)
FRAME_COUNT_WEIGHTS = (3, 4, 3)
FRAME_SIDE_CHOICES = (3, 4, 5, 6)
FRAME_GAP = 1
MIN_RADIUS_FROM_CENTER = 2.0
MIN_ANGLE_GAP = math.radians(22)
MAX_GENERATION_ATTEMPTS = 600
_GENERATION_CALLS = 0


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_border_mask(height: int, width: int) -> np.ndarray:
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _clockwise_angle_from_up(point: tuple[float, float], center: tuple[float, float]) -> float:
    dr = float(point[0] - center[0])
    dc = float(point[1] - center[1])
    angle = math.atan2(dc, -dr)
    return angle % (2 * math.pi)


def _quadrant(point: tuple[float, float], center: tuple[float, float]) -> tuple[int, int]:
    dr = float(point[0] - center[0])
    dc = float(point[1] - center[1])
    return (0 if dr < 0 else 1, 0 if dc < 0 else 1)


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(np.min(g)) >= 0 and int(np.max(g)) <= 9, "expected ARC palette ids"
    assert np.any(g != BACKGROUND), "expected at least one frame"
    assert int(np.bincount(g.ravel()).argmax()) == BACKGROUND, "expected black background to dominate"

    h, w = g.shape
    center = ((h - 1) / 2.0, (w - 1) / 2.0)
    frames: list[dict] = []

    for color in sorted(int(c) for c in np.unique(g) if int(c) != BACKGROUND):
        comps = _components(g == color)
        assert len(comps) == 1, "each non-black color must form exactly one frame"
        coords = comps[0]
        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3, "frames must have non-empty interiors"

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        border = _frame_border_mask(height, width)
        assert np.all(sub[border] == color), "expected a complete monochrome rectangular border"
        assert np.all(sub[~border] == BACKGROUND), "expected a black frame interior"
        assert int(coords.shape[0]) == int(border.sum()), "expected a one-cell-thick hollow frame"

        interior_rows, interior_cols = np.nonzero(~border)
        interior_rows = interior_rows + r0
        interior_cols = interior_cols + c0
        centroid = (float(interior_rows.mean()), float(interior_cols.mean()))
        radius = math.hypot(centroid[0] - center[0], centroid[1] - center[1])
        assert radius > 0, "frame interior centroid must not coincide with the grid center"

        frames.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "interior_rows": interior_rows,
                "interior_cols": interior_cols,
                "centroid": centroid,
                "angle": _clockwise_angle_from_up(centroid, center),
                "radius": radius,
            }
        )

    assert 3 <= len(frames) <= 5, "expected 3 to 5 frames"

    ordered_angles = sorted(frame["angle"] for frame in frames)
    for left, right in zip(ordered_angles, ordered_angles[1:]):
        assert not math.isclose(left, right, abs_tol=1e-9), "frame angles must be unique"

    return frames


def _render_from_order(shape: tuple[int, int], ordered_frames: list[dict], *, shift: int) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    n = len(ordered_frames)
    for i, frame in enumerate(ordered_frames):
        source_color = int(ordered_frames[(i + shift) % n]["color"])
        out[frame["interior_rows"], frame["interior_cols"]] = source_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each hollow rectangle frame and refill its interior with the previous frame's color in clockwise angular order.

    Input structure:
    - The grid has a black background and 3 to 5 non-overlapping one-cell-thick hollow rectangular frames.
    - Every frame is monochrome, axis-aligned, uniquely colored, and its strict interior is black.
    - The frame interiors lie at different angles around the grid center, so their clockwise order is unambiguous.

    Rule:
    - Identify every hollow rectangular frame, record its color, and record the coordinates strictly inside its border.
    - Compute each interior centroid and sort the frames by angle around the grid center, taking straight up as 0 and increasing clockwise.
    - Erase all frame-border pixels to black.
    - Fill each interior with the color of the previous frame in that angular order, wrapping cyclically.

    Key insight:
    - The disappearing frame borders are also the color palette: their colors must be captured before erasure, then reassigned according to the angular order of the enclosed regions.
    """
    frames = _extract_frames(np.asarray(grid, dtype=int))
    ordered_frames = sorted(frames, key=lambda frame: frame["angle"])
    return _render_from_order(tuple(grid.shape), ordered_frames, shift=-1)


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = FRAME_GAP) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _sample_grid_shape() -> tuple[int, int]:
    return random.randint(*GRID_HEIGHT_RANGE), random.randint(*GRID_WIDTH_RANGE)


def _sample_frame_count() -> int:
    global _GENERATION_CALLS
    index = _GENERATION_CALLS
    _GENERATION_CALLS += 1
    if random.random() < 0.8:
        return FRAME_COUNT_CHOICES[index % len(FRAME_COUNT_CHOICES)]
    return random.choices(FRAME_COUNT_CHOICES, weights=FRAME_COUNT_WEIGHTS, k=1)[0]


def _sample_frame_specs(count: int, profile: int) -> list[dict]:
    colors = random.sample(PALETTE, count)
    specs = []
    for i, color in enumerate(colors):
        height = random.choice(FRAME_SIDE_CHOICES)
        width = random.choice(FRAME_SIDE_CHOICES)
        if i == 0 and profile == 0:
            if random.random() < 0.5:
                height = 3
            else:
                width = 3
        elif i == 0 and profile == 1:
            if random.random() < 0.5:
                height = 6
            else:
                width = 6
        elif i == 0 and profile == 2 and height == width:
            width = 3 if width != 3 else 6
        specs.append({"color": color, "height": height, "width": width})
    specs.sort(key=lambda spec: (spec["height"] * spec["width"], spec["height"] + spec["width"]), reverse=True)
    return specs


def _try_place_frames(shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    h, w = shape
    center = ((h - 1) / 2.0, (w - 1) / 2.0)
    placed: list[dict] = []

    for spec in specs:
        fh = int(spec["height"])
        fw = int(spec["width"])
        candidates = []
        for r0 in range(0, h - fh + 1):
            for c0 in range(0, w - fw + 1):
                bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
                if any(not _boxes_separated(bbox, other["bbox"]) for other in placed):
                    continue

                centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
                radius = math.hypot(centroid[0] - center[0], centroid[1] - center[1])
                if radius < MIN_RADIUS_FROM_CENTER:
                    continue

                angle = _clockwise_angle_from_up(centroid, center)
                candidates.append(
                    {
                        "bbox": bbox,
                        "color": int(spec["color"]),
                        "height": fh,
                        "width": fw,
                        "centroid": centroid,
                        "angle": angle,
                        "radius": radius,
                    }
                )

        random.shuffle(candidates)
        if not candidates:
            return None
        placed.append(candidates[0])

    return placed


def _angular_quality_ok(frames: list[dict], center: tuple[float, float]) -> bool:
    ordered = sorted(frames, key=lambda frame: frame["angle"])
    angles = [frame["angle"] for frame in ordered]
    gaps = [
        (angles[(i + 1) % len(angles)] - angles[i]) % (2 * math.pi)
        for i in range(len(angles))
    ]
    if min(gaps) < MIN_ANGLE_GAP:
        return False

    occupied_quadrants = {_quadrant(frame["centroid"], center) for frame in frames}
    if len(occupied_quadrants) < 3:
        return False

    ids = [frame["bbox"] for frame in ordered]
    alternate_orders = [
        [frame["bbox"] for frame in sorted(frames, key=lambda frame: (frame["centroid"][0], frame["centroid"][1]))],
        [frame["bbox"] for frame in sorted(frames, key=lambda frame: (frame["centroid"][1], frame["centroid"][0]))],
    ]
    for seq in alternate_orders:
        if ids == seq or ids == list(reversed(seq)):
            return False

    return True


def _make_input_from_placed(shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for frame in placed:
        _draw_frame(grid, frame["bbox"], int(frame["color"]))
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        frames = _extract_frames(grid)
    except AssertionError:
        return False

    center = ((grid.shape[0] - 1) / 2.0, (grid.shape[1] - 1) / 2.0)
    if not _angular_quality_ok(frames, center):
        return False

    ordered = sorted(frames, key=lambda frame: frame["angle"])
    expected = _render_from_order(grid.shape, ordered, shift=-1)
    own_color_fill = _render_from_order(grid.shape, ordered, shift=0)
    next_color_fill = _render_from_order(grid.shape, ordered, shift=1)

    try:
        actual = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, own_color_fill):
        return False
    if np.array_equal(actual, next_color_fill):
        return False
    if np.any(actual[grid != BACKGROUND] != BACKGROUND):
        return False

    input_colors = {frame["color"] for frame in frames}
    output_colors = set(int(c) for c in np.unique(actual) if int(c) != BACKGROUND)
    if input_colors != output_colors:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    shape = _sample_grid_shape()
    count = _sample_frame_count()
    specs = _sample_frame_specs(count, profile=(count - 3) % 3)
    placed = _try_place_frames(shape, specs)
    if placed is None:
        return None

    center = ((shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0)
    if not _angular_quality_ok(placed, center):
        return None

    grid = _make_input_from_placed(shape, placed)
    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
