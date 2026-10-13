from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 2
PATTERN_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)

PATTERN_SIZE_OPTIONS = (2, 3, 4, 5)
FRAME_INNER_MIN = 4
FRAME_INNER_MAX = 12
MAX_SCALE = 4

COUNT_OPTIONS = (2, 3, 4)
COUNT_WEIGHTS = (2, 4, 3)
GRID_MIN_BY_COUNT = {2: 18, 3: 22, 4: 26}
GRID_MAX = 30
PLACEMENT_PAD = 1
MAX_GENERATION_ATTEMPTS = 500


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _find_monochrome_components(grid: np.ndarray, colors: set[int], connectivity: int = 8) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if visited[r, c] or color not in colors:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                if connectivity == 4:
                    neighbors = ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1))
                else:
                    neighbors = _neighbors8(rr, cc)
                for nr, nc in neighbors:
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            bbox = (min(rows), min(cols), max(rows), max(cols))
            components.append({"color": color, "coords": coords, "bbox": bbox})

    return components


def _bbox_shape(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((int(factor), int(factor)), dtype=int)).astype(int)


def _is_valid_frame(grid: np.ndarray, component: dict) -> bool:
    bbox = component["bbox"]
    h, w = _bbox_shape(bbox)
    if h != w or h < 4:
        return False
    sprite = _crop(grid, bbox)
    if not np.all(sprite[0, :] == FRAME_COLOR):
        return False
    if not np.all(sprite[-1, :] == FRAME_COLOR):
        return False
    if not np.all(sprite[:, 0] == FRAME_COLOR):
        return False
    if not np.all(sprite[:, -1] == FRAME_COLOR):
        return False
    if not np.all(sprite[1:-1, 1:-1] == BG_COLOR):
        return False
    expected_area = 4 * h - 4
    return len(component["coords"]) == expected_area


def _is_valid_pattern(grid: np.ndarray, component: dict) -> bool:
    bbox = component["bbox"]
    h, w = _bbox_shape(bbox)
    if h != w or h < 2:
        return False
    color = int(component["color"])
    if color in (BG_COLOR, FRAME_COLOR):
        return False
    sprite = _crop(grid, bbox)
    colors = set(map(int, np.unique(sprite)))
    if not colors <= {BG_COLOR, color}:
        return False
    mask = sprite == color
    if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
        return False
    return True


def _matching_info(pattern_sizes: list[int], frame_inners: list[int]):
    n = len(pattern_sizes)
    adj = [tuple(j for j, inner in enumerate(frame_inners) if inner % size == 0) for size in pattern_sizes]
    if any(len(options) == 0 for options in adj):
        return 0, None, adj, [0] * n

    used = [False] * n
    assignment = [-1] * n
    solutions: list[tuple[int, ...]] = []

    def rec():
        if len(solutions) > 1:
            return
        remaining = [i for i, a in enumerate(assignment) if a == -1]
        if not remaining:
            solutions.append(tuple(assignment))
            return
        idx = min(remaining, key=lambda i: sum(not used[j] for j in adj[i]))
        options = [j for j in adj[idx] if not used[j]]
        if not options:
            return
        for frame_idx in options:
            used[frame_idx] = True
            assignment[idx] = frame_idx
            rec()
            assignment[idx] = -1
            used[frame_idx] = False

    rec()
    frame_degrees = [sum(frame_idx in options for options in adj) for frame_idx in range(n)]
    unique_assignment = list(solutions[0]) if len(solutions) == 1 else None
    return len(solutions), unique_assignment, adj, frame_degrees


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale each small pattern into the uniquely compatible red frame whose interior side is an integer multiple of that pattern's side.

    Input structure:
    - The input is a black background containing several red hollow square frames with 1-cell-thick borders.
    - Elsewhere on the canvas there are the same number of separate non-red monochrome square patterns,
      each with a tight square bounding box and no positional correspondence to any frame.
    - Frame interiors are empty in the input, and matching information comes only from side-length divisibility.

    Rule:
    - A pattern may be placed only into a frame whose interior side length is divisible by that pattern's side length.
    - Across all patterns and frames there is exactly one perfect one-to-one matching under that divisibility rule.
    - Erase the original small patterns, enlarge each matched pattern by its own integer factor, and stamp it into the matched frame interior while keeping the red border unchanged.

    Key insight:
    - Local choices can be ambiguous because one frame may accept several pattern sizes, so the assignment must be solved globally as a unique perfect matching in the divisibility graph.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    frame_components = _find_monochrome_components(g, {FRAME_COLOR}, connectivity=8)
    assert frame_components, "expected at least one red frame"
    frames = [comp for comp in frame_components if _is_valid_frame(g, comp)]
    assert len(frames) == len(frame_components), "all red components must be hollow square frames"

    pattern_colors = set(map(int, np.unique(g))) - {BG_COLOR, FRAME_COLOR}
    pattern_components = _find_monochrome_components(g, pattern_colors, connectivity=8)
    patterns = [comp for comp in pattern_components if _is_valid_pattern(g, comp)]
    assert len(patterns) == len(pattern_components), "all non-red components must be valid square patterns"
    assert len(patterns) == len(frames), "frame and pattern counts must match"

    pattern_sizes = [_bbox_shape(comp["bbox"])[0] for comp in patterns]
    frame_inners = [_bbox_shape(comp["bbox"])[0] - 2 for comp in frames]
    solution_count, assignment, _adj, _frame_deg = _matching_info(pattern_sizes, frame_inners)
    assert solution_count == 1 and assignment is not None, "expected a unique divisibility matching"

    out = g.copy()
    for comp in patterns:
        rows, cols = zip(*comp["coords"])
        out[rows, cols] = BG_COLOR

    for pattern_idx, frame_idx in enumerate(assignment):
        pattern = patterns[pattern_idx]
        frame = frames[frame_idx]
        sprite = _crop(g, pattern["bbox"])
        side = sprite.shape[0]
        inner = _bbox_shape(frame["bbox"])[0] - 2
        factor = inner // side
        scaled = _scale_sprite(sprite, factor)
        r0, c0, r1, c1 = frame["bbox"]
        assert scaled.shape == (inner, inner)
        out[r0 + 1 : r1, c0 + 1 : c1] = scaled

    return out


def _connected8(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, cells[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _draw_path(mask: np.ndarray, start: tuple[int, int], target: tuple[int, int]) -> None:
    r, c = start
    mask[r, c] = True
    while (r, c) != target:
        dr = 0 if target[0] == r else (1 if target[0] > r else -1)
        dc = 0 if target[1] == c else (1 if target[1] > c else -1)
        if dr != 0 and dc != 0 and random.random() < 0.45:
            if random.random() < 0.5:
                dc = 0
            else:
                dr = 0
        r += dr
        c += dc
        mask[r, c] = True


def _random_pattern_mask(size: int) -> np.ndarray:
    min_area = max(size + 1, (size * size + 1) // 2)
    max_area = size * size - 1
    for _ in range(200):
        mask = np.zeros((size, size), dtype=bool)
        anchors = {
            (0, random.randrange(size)),
            (size - 1, random.randrange(size)),
            (random.randrange(size), 0),
            (random.randrange(size), size - 1),
        }
        center = (random.randrange(size), random.randrange(size))
        mask[center] = True
        for anchor in random.sample(list(anchors), len(anchors)):
            _draw_path(mask, center, anchor)
            if random.random() < 0.5:
                center = anchor

        target_area = random.randint(min_area, max_area)
        while int(mask.sum()) < target_area:
            frontier = {
                (nr, nc)
                for r, c in np.argwhere(mask)
                for nr, nc in _neighbors8(int(r), int(c))
                if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]
            }
            if not frontier:
                break
            rr, cc = random.choice(list(frontier))
            mask[rr, cc] = True

        if int(mask.sum()) >= min_area and int(mask.sum()) <= max_area and _connected8(mask):
            if mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any():
                return mask
    raise RuntimeError("failed to generate a connected square pattern")


def _render_pattern(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, int(color), BG_COLOR).astype(int)


def _render_frame(inner: int) -> np.ndarray:
    outer = inner + 2
    frame = np.full((outer, outer), FRAME_COLOR, dtype=int)
    frame[1:-1, 1:-1] = BG_COLOR
    return frame


def _sample_sizes(count: int):
    for _ in range(400):
        pattern_sizes = random.sample(PATTERN_SIZE_OPTIONS, count)
        candidate_inners = sorted(
            {
                size * scale
                for size in pattern_sizes
                for scale in range(1, MAX_SCALE + 1)
                if FRAME_INNER_MIN <= size * scale <= FRAME_INNER_MAX
            }
        )
        if len(candidate_inners) < count:
            continue
        frame_inners = random.sample(candidate_inners, count)
        solution_count, assignment, adj, frame_degrees = _matching_info(pattern_sizes, frame_inners)
        pattern_degrees = [len(options) for options in adj]
        if solution_count != 1 or assignment is None:
            continue
        if 1 not in pattern_degrees or max(pattern_degrees) <= 1:
            continue
        if max(frame_degrees) <= 1:
            continue
        scales = [frame_inners[assignment[i]] // pattern_sizes[i] for i in range(count)]
        if max(scales) <= 1:
            continue
        return pattern_sizes, frame_inners
    return None


def _pad_slice(start: int, length: int, limit: int) -> tuple[int, int]:
    return max(0, start - PLACEMENT_PAD), min(limit, start + length + PLACEMENT_PAD)


def _place_rectangles(shape: tuple[int, int], items: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(items)
    order = sorted(range(len(items)), key=lambda i: items[i][0] * items[i][1], reverse=True)

    for idx in order:
        ih, iw = items[idx]
        candidates = [(r, c) for r in range(1, h - ih) for c in range(1, w - iw)]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            rr0, rr1 = _pad_slice(r0, ih, h)
            cc0, cc1 = _pad_slice(c0, iw, w)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            positions[idx] = (r0, c0)
            occupied[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None

    return [pos for pos in positions if pos is not None]


def _center(top_left: tuple[int, int], size: int) -> tuple[float, float]:
    r0, c0 = top_left
    return (r0 + (size - 1) / 2.0, c0 + (size - 1) / 2.0)


def _has_proximity_decoy(
    pattern_specs: list[dict], frame_specs: list[dict], assignment: list[int]
) -> bool:
    frame_centers = [_center(frame["pos"], frame["inner"] + 2) for frame in frame_specs]
    for pattern_idx, pattern in enumerate(pattern_specs):
        pr, pc = _center(pattern["pos"], pattern["size"])
        dists = [abs(pr - fr) + abs(pc - fc) for fr, fc in frame_centers]
        best = min(dists)
        nearest = {idx for idx, dist in enumerate(dists) if dist == best}
        if assignment[pattern_idx] not in nearest:
            return True
    return False


def _build_expected(shape: tuple[int, int], pattern_specs: list[dict], frame_specs: list[dict], assignment: list[int]):
    inp = np.full(shape, BG_COLOR, dtype=int)
    out = np.full(shape, BG_COLOR, dtype=int)

    for frame in frame_specs:
        sprite = _render_frame(frame["inner"])
        r0, c0 = frame["pos"]
        inp[r0 : r0 + sprite.shape[0], c0 : c0 + sprite.shape[1]] = sprite
        out[r0 : r0 + sprite.shape[0], c0 : c0 + sprite.shape[1]] = sprite

    for pattern in pattern_specs:
        sprite = _render_pattern(pattern["mask"], pattern["color"])
        r0, c0 = pattern["pos"]
        inp[r0 : r0 + sprite.shape[0], c0 : c0 + sprite.shape[1]] = sprite

    for pattern_idx, frame_idx in enumerate(assignment):
        pattern = pattern_specs[pattern_idx]
        frame = frame_specs[frame_idx]
        scaled = _scale_sprite(_render_pattern(pattern["mask"], pattern["color"]), frame["inner"] // pattern["size"])
        r0, c0 = frame["pos"]
        out[r0 + 1 : r0 + 1 + frame["inner"], c0 + 1 : c0 + 1 + frame["inner"]] = scaled

    return inp, out


def _generate_attempt() -> np.ndarray | None:
    count = random.choices(COUNT_OPTIONS, weights=COUNT_WEIGHTS, k=1)[0]
    size_data = _sample_sizes(count)
    if size_data is None:
        return None
    pattern_sizes, frame_inners = size_data

    colors = random.sample(PATTERN_COLORS, count)
    pattern_specs = []
    for size, color in zip(pattern_sizes, colors):
        pattern_specs.append({"size": size, "color": color, "mask": _random_pattern_mask(size)})
    random.shuffle(pattern_specs)

    frame_specs = [{"inner": inner} for inner in frame_inners]
    random.shuffle(frame_specs)

    pattern_sizes_now = [spec["size"] for spec in pattern_specs]
    frame_inners_now = [spec["inner"] for spec in frame_specs]
    solution_count, assignment, adj, frame_degrees = _matching_info(pattern_sizes_now, frame_inners_now)
    if solution_count != 1 or assignment is None:
        return None

    pattern_degrees = [len(options) for options in adj]
    if 1 not in pattern_degrees or max(pattern_degrees) <= 1 or max(frame_degrees) <= 1:
        return None

    min_side = GRID_MIN_BY_COUNT[count]
    outer_sizes = [spec["inner"] + 2 for spec in frame_specs]
    side_low = max(min_side, max(outer_sizes) + 3)
    if side_low > GRID_MAX:
        return None
    shape = (
        random.randint(side_low, GRID_MAX),
        random.randint(side_low, GRID_MAX),
    )

    items = [(spec["size"], spec["size"]) for spec in pattern_specs] + [
        (spec["inner"] + 2, spec["inner"] + 2) for spec in frame_specs
    ]
    positions = _place_rectangles(shape, items)
    if positions is None:
        return None

    for spec, pos in zip(pattern_specs, positions[:count]):
        spec["pos"] = pos
    for spec, pos in zip(frame_specs, positions[count:]):
        spec["pos"] = pos

    if not any(int(spec["mask"].sum()) < spec["size"] * spec["size"] for spec in pattern_specs):
        return None
    if not _has_proximity_decoy(pattern_specs, frame_specs, assignment):
        return None

    inp, expected = _build_expected(shape, pattern_specs, frame_specs, assignment)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid divisibility-matching frame puzzle")
