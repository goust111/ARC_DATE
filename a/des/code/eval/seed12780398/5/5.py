import random
from typing import Iterable

import numpy as np


MARKER_FLIP = 1
MARKER_ROTATE = 2
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NON_MARKER_COLORS = tuple(c for c in range(10) if c not in (MARKER_FLIP, MARKER_ROTATE))
MAX_GENERATION_ATTEMPTS = 300


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, exclude: Iterable[int] = ()) -> list[int]:
    excluded = set(int(x) for x in exclude)
    return [int(v) for v in np.unique(grid) if int(v) not in excluded]


def component_coords(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def same_array(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def is_hollow_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * (h + w) - 4
    if len(coords) != expected:
        return False
    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if local[1:-1, 1:-1].any():
        return False
    return bool(
        np.all(local[0, :])
        and np.all(local[-1, :])
        and np.all(local[:, 0])
        and np.all(local[:, -1])
    )


def extract_frames(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames = []
    for color in colors_present(g, exclude={bg}):
        for coords in component_coords(g == color):
            if is_hollow_frame(coords):
                frames.append(
                    {
                        "color": int(color),
                        "bbox": bbox_coords(coords),
                    }
                )
    return frames


def frame_contains(outer: dict, inner: dict) -> bool:
    r0, c0, r1, c1 = outer["bbox"]
    ir0, ic0, ir1, ic1 = inner["bbox"]
    return r0 + 1 < ir0 and c0 + 1 < ic0 and ir1 < r1 - 1 and ic1 < c1 - 1


def build_chains(frames: list[dict]) -> list[list[dict]]:
    if not frames:
        return []
    n = len(frames)
    areas = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        areas.append((r1 - r0 + 1) * (c1 - c0 + 1))

    parents: list[int | None] = [None] * n
    children: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        candidates = [j for j in range(n) if i != j and frame_contains(frames[j], frames[i])]
        if not candidates:
            continue
        parent = min(candidates, key=lambda j: areas[j])
        parents[i] = parent
        children[parent].append(i)

    roots = [i for i in range(n) if parents[i] is None]
    roots.sort(key=lambda i: frames[i]["bbox"])

    chains = []
    for root in roots:
        chain = []
        cur = root
        while True:
            chain.append(frames[cur])
            if not children[cur]:
                break
            assert len(children[cur]) == 1, "expected linear nesting"
            cur = children[cur][0]
        chains.append(chain)
    return chains


def locate_marker(interior: np.ndarray) -> tuple[int, int, int]:
    h, w = map(int, interior.shape)
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    found = []
    for r, c in corners:
        color = int(interior[r, c])
        if color in (MARKER_FLIP, MARKER_ROTATE):
            found.append((r, c, color))
    assert len(found) == 1, "expected one corner marker"
    return found[0]


def extract_payload(interior: np.ndarray, bg: int, marker_rc: tuple[int, int]) -> np.ndarray:
    work = np.asarray(interior, dtype=int).copy()
    work[marker_rc[0], marker_rc[1]] = bg
    coords = np.argwhere(work != bg)
    assert len(coords) > 0, "payload must be non-empty"
    r0, c0, r1, c1 = bbox_coords(coords)
    return work[r0 : r1 + 1, c0 : c1 + 1].copy()


def apply_transform_sequence(payload: np.ndarray, depth: int, start_color: int) -> np.ndarray:
    arr = np.asarray(payload, dtype=int).copy()
    do_rotate = start_color == MARKER_ROTATE
    for _ in range(depth):
        arr = np.rot90(arr, -1) if do_rotate else np.fliplr(arr)
        do_rotate = not do_rotate
    return arr


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify nested hollow-rectangle groups, use the innermost corner marker to
    choose the first payload transform, then alternate rotate/flip one step per
    nesting level from inner to outer.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frames = extract_frames(g, bg)
    chains = build_chains(frames)

    out = g.copy()
    for chain in chains:
        inner = chain[-1]
        r0, c0, r1, c1 = inner["bbox"]
        inner_view = out[r0 + 1 : r1, c0 + 1 : c1]
        marker_r, marker_c, marker_color = locate_marker(inner_view)
        payload = extract_payload(inner_view, bg, (marker_r, marker_c))
        transformed = apply_transform_sequence(payload, len(chain), marker_color)
        ih, iw = map(int, inner_view.shape)
        th, tw = map(int, transformed.shape)
        assert th <= ih and tw <= iw, "transformed payload must fit inside innermost frame"
        inner_view[:, :] = bg
        top = (ih - th) // 2
        left = (iw - tw) // 2
        inner_view[top : top + th, left : left + tw] = transformed
    return out


def trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def random_connected_mask(height: int, width: int, cells: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = set()
    for dr, dc in NEIGHBORS4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((nr, nc))
    while mask.sum() < cells and frontier:
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        mask[r, c] = True
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))
    return trim_mask(mask)


def is_asymmetric(mask: np.ndarray) -> bool:
    variants = [np.fliplr(mask), np.rot90(mask, 1), np.rot90(mask, 2), np.rot90(mask, 3)]
    return all(not same_array(mask, var) for var in variants)


def make_payload(bg: int, color: int, depth: int, start_color: int) -> tuple[np.ndarray, np.ndarray]:
    require_change = depth != 4
    for _ in range(400):
        base_h = random.randint(2, 5)
        base_w = random.randint(2, 5)
        min_cells = max(3, (base_h * base_w + 3) // 4)
        max_cells = min(base_h * base_w - 1, min_cells + 5)
        if min_cells > max_cells:
            continue
        mask = random_connected_mask(base_h, base_w, random.randint(min_cells, max_cells))
        if min(mask.shape) < 2:
            continue
        if not is_asymmetric(mask):
            continue
        payload = np.full(mask.shape, bg, dtype=int)
        payload[mask] = color
        transformed = apply_transform_sequence(payload, depth, start_color)
        if require_change and same_array(payload, transformed):
            continue
        return payload, transformed
    fallback_masks = [
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
        np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
    ]
    for _ in range(40):
        mask = random.choice(fallback_masks).copy()
        for _ in range(random.randint(0, 3)):
            mask = np.rot90(mask, random.randint(1, 3))
            if random.random() < 0.5:
                mask = np.fliplr(mask)
        if not is_asymmetric(mask):
            continue
        payload = np.full(mask.shape, bg, dtype=int)
        payload[mask] = color
        transformed = apply_transform_sequence(payload, depth, start_color)
        if require_change and same_array(payload, transformed):
            continue
        return payload, transformed
    raise RuntimeError("failed to generate asymmetric payload")


def paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def make_group(bg: int, group_count: int) -> np.ndarray:
    max_depth = 4 if group_count == 1 else 3 if group_count == 2 else 2
    depth = random.randint(2, max_depth)
    start_color = random.choice((MARKER_ROTATE, MARKER_FLIP))

    pool = [c for c in range(10) if c not in (bg, MARKER_FLIP, MARKER_ROTATE)]
    random.shuffle(pool)
    frame_colors = pool[:depth]
    payload_color = pool[depth]

    payload, transformed = make_payload(bg, payload_color, depth, start_color)
    max_side = max(int(payload.shape[0]), int(payload.shape[1]), int(transformed.shape[0]), int(transformed.shape[1]))
    interior_h = random.randint(max_side + 2, max_side + 5)
    interior_w = random.randint(max_side + 2, max_side + 5)
    inner_h = interior_h + 2
    inner_w = interior_w + 2
    pads = [random.randint(2, 3) for _ in range(depth - 1)]

    total_pad = sum(pads)
    group_h = inner_h + 2 * total_pad
    group_w = inner_w + 2 * total_pad
    group = np.full((group_h, group_w), bg, dtype=int)

    prefix = 0
    for layer in range(depth):
        if layer > 0:
            prefix += pads[layer - 1]
        suffix = sum(pads[layer:]) if layer < depth - 1 else 0
        h = inner_h + 2 * suffix
        w = inner_w + 2 * suffix
        bbox = (prefix, prefix, prefix + h - 1, prefix + w - 1)
        paint_frame(group, bbox, frame_colors[layer])

    inner_offset = total_pad
    top = inner_offset + 1 + (interior_h - payload.shape[0]) // 2
    left = inner_offset + 1 + (interior_w - payload.shape[1]) // 2
    group[top : top + payload.shape[0], left : left + payload.shape[1]] = payload

    corners = [
        (inner_offset + 1, inner_offset + 1),
        (inner_offset + 1, inner_offset + interior_w),
        (inner_offset + interior_h, inner_offset + 1),
        (inner_offset + interior_h, inner_offset + interior_w),
    ]
    marker_r, marker_c = random.choice(corners)
    assert group[marker_r, marker_c] == bg, "marker corner must stay separate from payload"
    group[marker_r, marker_c] = start_color
    return group


def choose_canvas_shapes(groups: list[np.ndarray]) -> tuple[int, int] | None:
    heights = [int(g.shape[0]) for g in groups]
    widths = [int(g.shape[1]) for g in groups]
    total_area = sum((h + 2) * (w + 2) for h, w in zip(heights, widths))
    side = int(np.ceil(np.sqrt(total_area * 1.6)))
    min_h = max(max(heights) + 2, side - 2)
    min_w = max(max(widths) + 2, side - 2)
    if min_h > 30 or min_w > 30:
        return None
    max_h = min(30, min_h + 6)
    max_w = min(30, min_w + 6)
    return random.randint(min_h, max_h), random.randint(min_w, max_w)


def place_groups(canvas_shape: tuple[int, int], groups: list[np.ndarray], bg: int) -> np.ndarray | None:
    h, w = canvas_shape
    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    order = list(range(len(groups)))
    random.shuffle(order)
    for idx in order:
        arr = groups[idx]
        gh, gw = map(int, arr.shape)
        if gh + 2 > h or gw + 2 > w:
            return None
        placed = False
        for _ in range(250):
            top = random.randint(1, h - gh - 1)
            left = random.randint(1, w - gw - 1)
            r0 = max(0, top - 1)
            c0 = max(0, left - 1)
            r1 = min(h, top + gh + 1)
            c1 = min(w, left + gw + 1)
            if occupied[r0:r1, c0:c1].any():
                continue
            grid[top : top + gh, left : left + gw] = arr
            occupied[top : top + gh, left : left + gw] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(NON_MARKER_COLORS)
    group_count = random.randint(1, 3)
    groups = [make_group(bg, group_count) for _ in range(group_count)]
    canvas_shape = choose_canvas_shapes(groups)
    if canvas_shape is None:
        return None
    grid = place_groups(canvas_shape, groups, bg)
    if grid is None:
        return None
    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
