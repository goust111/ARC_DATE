import random
from collections import deque

import numpy as np


BG = 0
YELLOW = 4
DEPTH_FILL_COLORS = (5, 6, 7, 8)  # grey, magenta, orange, azure
FRAME_COLOR_POOL = (1, 2, 3, 9)
FRAME_GAP_IN_OUTPUT = 2
MAX_GENERATION_ATTEMPTS = 300
MAX_GRID_SIZE = 30
MIN_MARKER_FRAMES_PROB = 0.75


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unnest layered rectangular frames into a left-to-right strip.

    Input structure:
    - black background
    - 1 to 4 nested hollow rectangular frames of distinct colors
    - optional yellow marker(s) attached just outside some frame borders

    Rule:
    - extract the frames from outermost to innermost
    - keep each frame's size and border color
    - fill its interior according to depth:
      outer->inner = grey, magenta, orange, azure
    - restore each frame's yellow marker pixels at the same position relative
      to that frame after expansion
    """
    g = np.asarray(grid, dtype=int)
    layers = _extract_layers(g)
    assert 1 <= len(layers) <= 4, "expected 1-4 nested frames"

    row_shift = 0
    max_rel_row = 0
    layer_extents = []
    for layer in layers:
        min_r, max_r, min_c, max_c = _layer_extents(layer)
        row_shift = max(row_shift, -min_r)
        max_rel_row = max(max_rel_row, max_r)
        layer_extents.append((min_r, max_r, min_c, max_c))

    placements = []
    next_object_left = 0
    for layer, (_min_r, _max_r, min_c, max_c) in zip(layers, layer_extents):
        frame_left = next_object_left - min_c
        placements.append((row_shift, frame_left))
        object_width = max_c - min_c + 1
        next_object_left += object_width + FRAME_GAP_IN_OUTPUT

    out_h = row_shift + max_rel_row + 1
    out_w = next_object_left - FRAME_GAP_IN_OUTPUT
    out = np.full((out_h, out_w), BG, dtype=int)

    for depth, (layer, (top, left)) in enumerate(zip(layers, placements)):
        h, w = layer["height"], layer["width"]
        color = layer["color"]
        fill_color = DEPTH_FILL_COLORS[depth]

        _paint_frame(out, top, left, h, w, color)
        if h > 2 and w > 2:
            out[top + 1 : top + h - 1, left + 1 : left + w - 1] = fill_color

        for dr, dc in layer["markers"]:
            out[top + dr, left + dc] = YELLOW

    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layer_count = rng.randint(1, 4)
    gaps = [rng.randint(1, 2) for _ in range(layer_count - 1)]

    top = rng.randint(1, 3)
    bottom = rng.randint(1, 3)
    left = rng.randint(1, 3)
    right = rng.randint(1, 3)

    min_outer_h = _minimum_outer_size(layer_count, gaps)
    min_outer_w = _minimum_outer_size(layer_count, gaps)
    max_outer_h = MAX_GRID_SIZE - top - bottom
    max_outer_w = MAX_GRID_SIZE - left - right
    if min_outer_h > max_outer_h or min_outer_w > max_outer_w:
        return None

    outer_h = rng.randint(min_outer_h, min(max_outer_h, min_outer_h + 8))
    outer_w = rng.randint(min_outer_w, min(max_outer_w, min_outer_w + 8))

    grid = np.full((top + outer_h + bottom, left + outer_w + right), BG, dtype=int)
    frame_colors = rng.sample(FRAME_COLOR_POOL, layer_count)
    bboxes = _nested_bboxes(top, left, outer_h, outer_w, gaps)

    for color, (r0, r1, c0, c1) in zip(frame_colors, bboxes):
        _paint_frame(grid, r0, c0, r1 - r0 + 1, c1 - c0 + 1, color)

    eligible = [0]
    for idx in range(1, layer_count):
        if gaps[idx - 1] == 2:
            eligible.append(idx)

    marker_layers = [idx for idx in eligible if rng.random() < MIN_MARKER_FRAMES_PROB]
    if not marker_layers:
        marker_layers = [rng.choice(eligible)]

    generated_markers = {}
    for idx in marker_layers:
        marker = _place_marker(grid, bboxes[idx], rng)
        if marker is None:
            return None
        r0, _r1, c0, _c1 = bboxes[idx]
        generated_markers[idx] = sorted((r - r0, c - c0) for r, c in marker)

    try:
        parsed = _extract_layers(grid)
        if len(parsed) != layer_count:
            return None
        if [layer["color"] for layer in parsed] != frame_colors:
            return None

        for idx, layer in enumerate(parsed):
            bbox = layer["bbox"]
            if bbox != bboxes[idx]:
                return None
            expected = generated_markers.get(idx, [])
            if sorted(layer["markers"]) != expected:
                return None

        out = main(grid)
        if out.size == 0 or np.array_equal(out, grid):
            return None
    except AssertionError:
        return None

    return grid


def _minimum_outer_size(layer_count: int, gaps: list[int]) -> int:
    size = 3
    for gap in reversed(gaps):
        size += 2 * (gap + 1)
    return size


def _nested_bboxes(
    top: int,
    left: int,
    outer_h: int,
    outer_w: int,
    gaps: list[int],
) -> list[tuple[int, int, int, int]]:
    bboxes = []
    r0 = top
    c0 = left
    r1 = top + outer_h - 1
    c1 = left + outer_w - 1
    bboxes.append((r0, r1, c0, c1))
    for gap in gaps:
        inset = gap + 1
        r0 += inset
        c0 += inset
        r1 -= inset
        c1 -= inset
        bboxes.append((r0, r1, c0, c1))
    return bboxes


def _place_marker(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    r0, r1, c0, c1 = bbox
    side = rng.choice(("top", "bottom", "left", "right"))
    length = rng.randint(1, 2)

    if side in ("top", "bottom"):
        if c1 - c0 + 1 < length:
            return None
        start = rng.randint(c0, c1 - length + 1)
        row = r0 - 1 if side == "top" else r1 + 1
        cells = [(row, start + offset) for offset in range(length)]
    else:
        if r1 - r0 + 1 < length:
            return None
        start = rng.randint(r0, r1 - length + 1)
        col = c0 - 1 if side == "left" else c1 + 1
        cells = [(start + offset, col) for offset in range(length)]

    for r, c in cells:
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            return None
        if grid[r, c] != BG:
            return None

    for r, c in cells:
        grid[r, c] = YELLOW
    return cells


def _extract_layers(grid: np.ndarray) -> list[dict]:
    frame_mask = (grid != BG) & (grid != YELLOW)
    frame_comps = _components(frame_mask)
    frames = []

    for coords in frame_comps:
        color = int(grid[coords[0][0], coords[0][1]])
        bbox = _bbox(coords)
        if not _is_rectangle_frame(grid, coords, color, bbox):
            continue
        r0, r1, c0, c1 = bbox
        frames.append(
            {
                "bbox": bbox,
                "color": color,
                "height": r1 - r0 + 1,
                "width": c1 - c0 + 1,
                "markers": [],
            }
        )

    assert frames, "no frame detected"
    frames.sort(key=lambda layer: _bbox_area(layer["bbox"]), reverse=True)
    _assert_nested_chain(frames)

    marker_mask = grid == YELLOW
    for coords in _components(marker_mask):
        matched = []
        for idx, frame in enumerate(frames):
            rel = _marker_relative_coords(coords, frame["bbox"])
            if rel is not None:
                matched.append((idx, rel))
        assert len(matched) == 1, "yellow marker must match exactly one frame"
        idx, rel = matched[0]
        frames[idx]["markers"].extend(rel)

    for frame in frames:
        frame["markers"] = sorted(set(frame["markers"]))

    return frames


def _layer_extents(layer: dict) -> tuple[int, int, int, int]:
    h, w = layer["height"], layer["width"]
    rows = [0, h - 1]
    cols = [0, w - 1]
    for dr, dc in layer["markers"]:
        rows.append(dr)
        cols.append(dc)
    return min(rows), max(rows), min(cols), max(cols)


def _assert_nested_chain(frames: list[dict]) -> None:
    for outer, inner in zip(frames, frames[1:]):
        or0, or1, oc0, oc1 = outer["bbox"]
        ir0, ir1, ic0, ic1 = inner["bbox"]
        assert or0 < ir0 < ir1 < or1, "frames must nest vertically"
        assert oc0 < ic0 < ic1 < oc1, "frames must nest horizontally"


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), max(rows), min(cols), max(cols)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, r1, c0, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _is_rectangle_frame(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    color: int,
    bbox: tuple[int, int, int, int],
) -> bool:
    r0, r1, c0, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))

    coord_set = set(coords)
    if coord_set != expected:
        return False
    for r, c in expected:
        if int(grid[r, c]) != color:
            return False
    return True


def _marker_relative_coords(
    coords: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]] | None:
    coords = sorted(coords)
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1, c0, c1 = bbox

    if len(coords) not in (1, 2):
        return None

    if len(set(rows)) == 1:
        row = rows[0]
        ordered_cols = sorted(cols)
        if len(ordered_cols) == 2 and ordered_cols[1] != ordered_cols[0] + 1:
            return None
        if row == r0 - 1 and c0 <= ordered_cols[0] and ordered_cols[-1] <= c1:
            return [(row - r0, c - c0) for _r, c in coords]
        if row == r1 + 1 and c0 <= ordered_cols[0] and ordered_cols[-1] <= c1:
            return [(row - r0, c - c0) for _r, c in coords]

    if len(set(cols)) == 1:
        col = cols[0]
        ordered_rows = sorted(rows)
        if len(ordered_rows) == 2 and ordered_rows[1] != ordered_rows[0] + 1:
            return None
        if col == c0 - 1 and r0 <= ordered_rows[0] and ordered_rows[-1] <= r1:
            return [(r - r0, col - c0) for r, _c in coords]
        if col == c1 + 1 and r0 <= ordered_rows[0] and ordered_rows[-1] <= r1:
            return [(r - r0, col - c0) for r, _c in coords]

    return None


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _paint_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color
