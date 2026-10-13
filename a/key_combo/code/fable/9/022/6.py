from __future__ import annotations

import numpy as np

BACKGROUND = 0
PROBE_COLOR = 1
FRAME_COLOR = 3
INPUT_COLORS = (2, 4, 5, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

MIN_STACKS = 2
MAX_STACKS = 3
MIN_OBJECTS = 3
MAX_OBJECTS = 6
MIN_SIDE = 2
MAX_SIDE = 6
MIN_CLEARANCE = 1
MAX_CLEARANCE = 6
MIN_GAP_BETWEEN_STACKS = 2
MAX_GAP_BETWEEN_STACKS = 4
MAX_GENERATION_ATTEMPTS = 600


def _components_by_color(grid: np.ndarray) -> list[dict]:
    """Return 4-connected monochromatic components on a black background."""
    h, w = map(int, grid.shape)
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BACKGROUND or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )

    return components


def _find_square_objects(grid: np.ndarray) -> list[dict]:
    """Extract solid monochromatic square objects from the grid."""
    squares: list[dict] = []
    for comp in _components_by_color(grid):
        r0, c0, r1, c1 = comp["bbox"]
        side_r = r1 - r0 + 1
        side_c = c1 - c0 + 1
        if side_r != side_c:
            raise ValueError("all foreground components must be squares")
        if comp["area"] != side_r * side_c:
            raise ValueError("square components must be solid")
        block = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(block == comp["color"]):
            raise ValueError("square bounding boxes must be fully filled")

        squares.append(
            {
                "color": int(comp["color"]),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "side": int(side_r),
            }
        )

    squares.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["side"], item["color"]))
    return squares


def _column_probe_lengths(grid: np.ndarray, square: dict) -> list[int]:
    """Measure how many black cells lie below each bottom-edge column."""
    h, _w = map(int, grid.shape)
    _r0, c0, r1, c1 = square["bbox"]
    lengths: list[int] = []

    for c in range(c0, c1 + 1):
        rr = r1 + 1
        while rr < h and int(grid[rr, c]) == BACKGROUND:
            rr += 1
        lengths.append(int(rr - (r1 + 1)))

    return lengths


def _paint_frame(mask: np.ndarray, bbox: tuple[int, int, int, int], distance: int) -> None:
    """Mark the clipped outer square frame expanded by `distance`."""
    if distance <= 0:
        return

    h, w = map(int, mask.shape)
    r0, c0, r1, c1 = bbox
    top = r0 - distance
    left = c0 - distance
    bottom = r1 + distance
    right = c1 + distance

    clip_r0 = max(0, top)
    clip_c0 = max(0, left)
    clip_r1 = min(h - 1, bottom)
    clip_c1 = min(w - 1, right)

    if 0 <= top < h:
        mask[top, clip_c0 : clip_c1 + 1] = True
    if 0 <= bottom < h:
        mask[bottom, clip_c0 : clip_c1 + 1] = True
    if 0 <= left < w:
        mask[clip_r0 : clip_r1 + 1, left] = True
    if 0 <= right < w:
        mask[clip_r0 : clip_r1 + 1, right] = True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Measure each square's downward clearance, then draw a green frame at that radius while preserving the input squares.

    Input structure:
    - The grid has a black background and several solid monochromatic square objects.
    - Input colors never use blue or green; those are reserved for constructed marks.
    - Some squares have another square directly below their full width, while others only see the bottom edge.

    Rule:
    - For every square, probe downward from each column along its bottom edge until the first non-black cell or the grid bottom, painting that black run blue.
    - Let d be the smallest probe length for that square; expand the square's bounding box outward by d and draw the clipped green outer frame.
    - Paint probes and frames only onto cells that were black in the input, keeping every original square unchanged.

    Key insight:
    - The green frame size is not tied to the square's side length; it is determined by the downward empty space measured first by the blue probes.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    squares = _find_square_objects(g)
    out = g.copy()
    bg_mask = g == BACKGROUND
    blue_mask = np.zeros_like(bg_mask)
    green_mask = np.zeros_like(bg_mask)

    for square in squares:
        r0, c0, r1, c1 = square["bbox"]
        lengths = _column_probe_lengths(g, square)
        distance = min(lengths) if lengths else 0

        for c, length in zip(range(c0, c1 + 1), lengths):
            if length > 0:
                blue_mask[r1 + 1 : r1 + 1 + length, c] = True

        _paint_frame(green_mask, square["bbox"], distance)

    out[blue_mask & bg_mask] = PROBE_COLOR
    out[green_mask & bg_mask] = FRAME_COLOR
    return out


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _random_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    """Split a non-negative integer into `parts` non-negative summands."""
    if parts <= 1:
        return [int(total)]
    if total == 0:
        return [0] * parts
    cuts = sorted(int(x) for x in rng.integers(0, total + 1, size=parts - 1))
    values = [cuts[0]]
    for i in range(1, len(cuts)):
        values.append(cuts[i] - cuts[i - 1])
    values.append(total - cuts[-1])
    return values


def _sample_stack_levels(rng: np.random.Generator) -> list[int]:
    while True:
        stack_count = _randint(rng, MIN_STACKS, MAX_STACKS)
        levels = [_randint(rng, 1, 3) for _ in range(stack_count)]
        total = sum(levels)
        if MIN_OBJECTS <= total <= MAX_OBJECTS and any(level > 1 for level in levels):
            return levels


def _sample_stack_geometry(levels: int, rng: np.random.Generator) -> dict:
    """Sample one vertical stack where each lower square fully covers the one above."""
    bottom_side = _randint(rng, max(3, MIN_SIDE), MAX_SIDE)
    sizes_bottom = [bottom_side]
    for _ in range(levels - 1):
        sizes_bottom.append(_randint(rng, MIN_SIDE, sizes_bottom[-1]))
    sizes_top = list(reversed(sizes_bottom))

    clearances = [_randint(rng, MIN_CLEARANCE, MAX_CLEARANCE) for _ in range(levels)]
    if levels >= 2 and rng.random() < 0.6:
        idx = _randint(rng, 0, levels - 1)
        clearances[idx] = sizes_top[idx]

    cols_bottom = [0]
    for lower_side, upper_side in zip(sizes_bottom, sizes_bottom[1:]):
        lower_c0 = cols_bottom[-1]
        cols_bottom.append(_randint(rng, lower_c0, lower_c0 + lower_side - upper_side))
    cols_top = list(reversed(cols_bottom))

    objects = []
    rel_r0 = 0
    for side, clearance, rel_c0 in zip(sizes_top, clearances, cols_top):
        objects.append({"r0": rel_r0, "c0": rel_c0, "side": side, "d": clearance})
        rel_r0 += side + clearance

    min_frame_c = min(obj["c0"] - obj["d"] for obj in objects)
    max_frame_c = max(obj["c0"] + obj["side"] - 1 + obj["d"] for obj in objects)
    shift = -min_frame_c
    for obj in objects:
        obj["c0"] += shift

    return {
        "objects": objects,
        "height": rel_r0,
        "width": int(max_frame_c - min_frame_c + 1),
    }


def _build_candidate_input(rng: np.random.Generator) -> np.ndarray | None:
    levels = _sample_stack_levels(rng)
    stacks = [_sample_stack_geometry(level, rng) for level in levels]

    max_height = max(stack["height"] for stack in stacks)
    if max_height > 30:
        return None
    height = _randint(rng, max_height, min(30, max_height + 3))

    base_gaps = [_randint(rng, MIN_GAP_BETWEEN_STACKS, MAX_GAP_BETWEEN_STACKS) for _ in range(len(stacks) - 1)]
    min_width = sum(stack["width"] for stack in stacks) + sum(base_gaps)
    if min_width > 30:
        return None
    width = _randint(rng, min_width, min(30, min_width + 4))

    extra_slots = _random_composition(width - min_width, len(stacks) + 1, rng)
    x = extra_slots[0]
    stack_offsets = []
    for idx, stack in enumerate(stacks):
        stack_offsets.append(x)
        x += stack["width"]
        if idx < len(base_gaps):
            x += base_gaps[idx] + extra_slots[idx + 1]

    colors = rng.choice(np.array(INPUT_COLORS, dtype=int), size=sum(levels), replace=False)
    color_iter = iter(int(color) for color in colors.tolist())

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for stack, x0 in zip(stacks, stack_offsets):
        top_margin = height - stack["height"]
        for obj in stack["objects"]:
            color = next(color_iter)
            r0 = top_margin + obj["r0"]
            c0 = x0 + obj["c0"]
            side = obj["side"]
            grid[r0 : r0 + side, c0 : c0 + side] = color

    return grid


def _sample_probe_and_frame_masks(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    squares = _find_square_objects(grid)
    blue_mask = np.zeros_like(grid, dtype=bool)
    green_mask = np.zeros_like(grid, dtype=bool)
    info: list[dict] = []
    h, _w = map(int, grid.shape)

    for square in squares:
        r0, c0, r1, c1 = square["bbox"]
        lengths = _column_probe_lengths(grid, square)
        if not lengths:
            raise ValueError("empty square footprint")
        if len(set(lengths)) != 1:
            raise ValueError("generated squares must have uniform probe lengths")
        d = lengths[0]

        stop_row = r1 + d + 1
        blocked_by_square = stop_row < h
        if blocked_by_square:
            blocker_color = int(grid[stop_row, c0])
            if blocker_color == BACKGROUND:
                raise ValueError("probe stop must be caused by a square or the border")
        else:
            blocker_color = BACKGROUND

        for c in range(c0, c1 + 1):
            blue_mask[r1 + 1 : r1 + 1 + d, c] = True
        _paint_frame(green_mask, square["bbox"], d)
        info.append(
            {
                "bbox": square["bbox"],
                "side": square["side"],
                "distance": d,
                "blocked": blocked_by_square,
                "blocker_color": blocker_color,
            }
        )

    return blue_mask, green_mask, info


def _validate_generated_input(grid: np.ndarray) -> bool:
    if grid is None:
        return False
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    colors = set(int(v) for v in np.unique(grid))
    if PROBE_COLOR in colors or FRAME_COLOR in colors:
        return False

    try:
        squares = _find_square_objects(grid)
        blue_mask, green_mask, info = _sample_probe_and_frame_masks(grid)
    except ValueError:
        return False

    if not (MIN_OBJECTS <= len(squares) <= MAX_OBJECTS):
        return False

    distances = [item["distance"] for item in info]
    sides = [item["side"] for item in info]
    blocked_count = sum(1 for item in info if item["blocked"])
    floor_count = len(info) - blocked_count

    if any(distance <= 0 for distance in distances):
        return False
    if blocked_count < 1 or floor_count < 1:
        return False
    if len(set(sides)) < 2:
        return False
    if len(set(distances)) < 2:
        return False
    if not any(distance == side for distance, side in zip(distances, sides)):
        return False
    if not any(distance != side for distance, side in zip(distances, sides)):
        return False

    out = main(grid)
    blue_count = int(np.sum(out == PROBE_COLOR))
    green_count = int(np.sum(out == FRAME_COLOR))
    if blue_count <= 0 or green_count <= 0:
        return False
    if int(np.sum((grid != BACKGROUND) & ((out == PROBE_COLOR) | (out == FRAME_COLOR)))) != 0:
        return False

    frame_only = green_mask & (grid == BACKGROUND)
    probe_only = blue_mask & (grid == BACKGROUND) & ~green_mask
    if int(frame_only.sum()) < len(info):
        return False
    if int(probe_only.sum()) < len(info):
        return False

    if blue_count + green_count > int(grid.size * 0.45):
        return False

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate_input(rng)
        if _validate_generated_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid non-degenerate ARC input")
