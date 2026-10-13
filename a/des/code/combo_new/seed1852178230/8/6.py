import random
from collections import defaultdict, deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
SIDES = ("top", "bottom", "left", "right")
PAIR_COUNT_CHOICES = (2, 3)
OUTER_SIZE_MIN = 3
OUTER_SIZE_MAX = 5
GEN_ATTEMPTS = 400
PAIR_PLACEMENT_ATTEMPTS = 120


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _components_by_color(grid: np.ndarray, bg: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return components


def _frame_from_component(component):
    r0, c0, r1, c1 = component["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if not (OUTER_SIZE_MIN <= height <= OUTER_SIZE_MAX and OUTER_SIZE_MIN <= width <= OUTER_SIZE_MAX):
        return None
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0 + 1, r1):
        expected.add((r, c0))
        expected.add((r, c1))
    coords = set(component["coords"])
    if coords != expected:
        return None
    return {
        "color": int(component["color"]),
        "bbox": (r0, c0, r1, c1),
        "height": height,
        "width": width,
    }


def _extract_frames(grid: np.ndarray, bg: int):
    frames = []
    for component in _components_by_color(grid, bg):
        frame = _frame_from_component(component)
        if frame is not None:
            frames.append(frame)
    return frames


def _interior_pattern(grid: np.ndarray, frame, bg: int):
    r0, c0, r1, c1 = frame["bbox"]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    return [tuple(map(int, rc)) for rc in np.argwhere(interior != bg)]


def _adjacent_markers(grid: np.ndarray, frame, bg: int):
    h, w = grid.shape
    r0, c0, r1, c1 = frame["bbox"]
    markers = []

    if r0 > 0:
        for c in range(c0, c1 + 1):
            color = int(grid[r0 - 1, c])
            if color != bg:
                markers.append((r0 - 1, c, color))
    if r1 + 1 < h:
        for c in range(c0, c1 + 1):
            color = int(grid[r1 + 1, c])
            if color != bg:
                markers.append((r1 + 1, c, color))
    if c0 > 0:
        for r in range(r0, r1 + 1):
            color = int(grid[r, c0 - 1])
            if color != bg:
                markers.append((r, c0 - 1, color))
    if c1 + 1 < w:
        for r in range(r0, r1 + 1):
            color = int(grid[r, c1 + 1])
            if color != bg:
                markers.append((r, c1 + 1, color))
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair matching hollow frames, rotate each source pattern by 180 degrees, and teleport it into the empty twin using the target marker color.

    Input structure:
    - The background is black.
    - Several non-black one-pixel-thick rectangular hollow frames appear on the grid; each frame color appears exactly twice.
    - Within each same-color pair, one frame contains a small non-black interior pattern and the other has an empty black interior plus one adjacent exterior marker pixel.

    Rule:
    - Detect all hollow rectangular frames and pair them by frame color.
    - In each pair, treat the framed pattern as the source and the empty frame with the exterior marker as the target.
    - Read the source pattern coordinates relative to the source interior, rotate those coordinates by 180 degrees around the interior center, and draw the result inside the target interior.
    - Recolor the teleported pattern with the target marker color, clear the original source pattern to black, and leave all frame borders and marker pixels unchanged.

    Key insight:
    - Same-color frames define the portal pairing, while the easy-to-miss single marker beside the empty twin determines the teleported pattern's new color rather than the source pattern color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames = _extract_frames(g, bg)
    by_color = defaultdict(list)
    for frame in frames:
        by_color[int(frame["color"])].append(frame)

    out = g.copy()

    for color, pair in by_color.items():
        assert len(pair) == 2, f"frame color {color} must occur exactly twice"

        annotated = []
        for frame in pair:
            pattern = _interior_pattern(g, frame, bg)
            markers = _adjacent_markers(g, frame, bg)
            annotated.append((frame, pattern, markers))

        sources = [entry for entry in annotated if entry[1]]
        targets = [entry for entry in annotated if not entry[1]]
        assert len(sources) == 1 and len(targets) == 1, f"frame color {color} must have one source and one target"

        source, pattern, source_markers = sources[0]
        target, _, target_markers = targets[0]

        assert not source_markers, f"source frame color {color} should not have exterior markers"
        assert len(target_markers) == 1, f"target frame color {color} should have exactly one exterior marker"

        src_inner_h = int(source["height"]) - 2
        src_inner_w = int(source["width"]) - 2
        dst_inner_h = int(target["height"]) - 2
        dst_inner_w = int(target["width"]) - 2
        assert (src_inner_h, src_inner_w) == (dst_inner_h, dst_inner_w), "paired portal interiors must match"

        sr0, sc0, _, _ = source["bbox"]
        tr0, tc0, _, _ = target["bbox"]

        for pr, pc in pattern:
            out[sr0 + 1 + pr, sc0 + 1 + pc] = bg

        marker_color = int(target_markers[0][2])
        for pr, pc in pattern:
            rr = src_inner_h - 1 - pr
            cc = src_inner_w - 1 - pc
            out[tr0 + 1 + rr, tc0 + 1 + cc] = marker_color

    return out


# --------------- generator ---------------

def _draw_frame(grid: np.ndarray, bbox, color: int):
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _random_pattern(inner_h: int, inner_w: int, rng: random.Random):
    total = inner_h * inner_w
    max_cells = min(3, total)
    cell_count = rng.randint(1, max_cells)

    if total == 1:
        return [(0, 0)]

    for _ in range(80):
        start = (rng.randrange(inner_h), rng.randrange(inner_w))
        cells = {start}
        while len(cells) < cell_count:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < inner_h and 0 <= nc < inner_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != cell_count:
            continue
        rotated = {(inner_h - 1 - r, inner_w - 1 - c) for r, c in cells}
        if cell_count == 1 or cells != rotated or rng.random() < 0.25:
            return sorted(cells)

    coords = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    return sorted(coords[:cell_count])


def _marker_cell(side: str, bbox, rng: random.Random):
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0 - 1, rng.randint(c0, c1)
    if side == "bottom":
        return r1 + 1, rng.randint(c0, c1)
    if side == "left":
        return rng.randint(r0, r1), c0 - 1
    return rng.randint(r0, r1), c1 + 1


def _place_object(grid: np.ndarray, blocked: np.ndarray, outer_h: int, outer_w: int, rng: random.Random):
    h, w = grid.shape
    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        r0 = rng.randint(1, h - outer_h - 1)
        c0 = rng.randint(1, w - outer_w - 1)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        if blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
            continue
        blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
        return r0, c0, r1, c1
    return None


def _gen_attempt(rng: random.Random):
    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    if pair_count == 2:
        height = rng.randint(12, 24)
        width = rng.randint(12, 24)
    else:
        height = rng.randint(18, 30)
        width = rng.randint(18, 30)

    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    colors = rng.sample(PALETTE, 3 * pair_count)
    frame_colors = colors[:pair_count]
    pattern_colors = colors[pair_count : 2 * pair_count]
    marker_colors = colors[2 * pair_count :]

    for idx in range(pair_count):
        frame_color = frame_colors[idx]
        pattern_color = pattern_colors[idx]
        marker_color = marker_colors[idx]
        outer_h = rng.randint(OUTER_SIZE_MIN, OUTER_SIZE_MAX)
        outer_w = rng.randint(OUTER_SIZE_MIN, OUTER_SIZE_MAX)

        source_bbox = _place_object(grid, blocked, outer_h, outer_w, rng)
        if source_bbox is None:
            return None
        target_bbox = _place_object(grid, blocked, outer_h, outer_w, rng)
        if target_bbox is None:
            return None

        _draw_frame(grid, source_bbox, frame_color)
        _draw_frame(grid, target_bbox, frame_color)

        inner_h = outer_h - 2
        inner_w = outer_w - 2
        pattern = _random_pattern(inner_h, inner_w, rng)
        sr0, sc0, _, _ = source_bbox
        for pr, pc in pattern:
            grid[sr0 + 1 + pr, sc0 + 1 + pc] = pattern_color

        side = rng.choice(SIDES)
        mr, mc = _marker_cell(side, target_bbox, rng)
        grid[mr, mc] = marker_color

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
