import random
from collections import deque

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _interior_mask(component: dict) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in component["coords"]:
        mask[r - r0, c - c0] = True

    outside = np.zeros((h, w), dtype=bool)
    queue = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))

    while queue:
        cr, cc = queue.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                queue.append((nr, nc))

    interior = (~mask) & (~outside)
    assert interior.any(), "each region must be closed"
    hole_count = 0
    seen = np.zeros_like(interior, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not interior[r, c] or seen[r, c]:
                continue
            hole_count += 1
            queue = deque([(r, c)])
            seen[r, c] = True
            while queue:
                cr, cc = queue.popleft()
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and interior[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
    return interior, (r0, c0, r1, c1), hole_count


def _analyze(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all((0 <= g) & (g <= 9)), "ARC colors must be digits"
    assert int(np.count_nonzero(g == BG)) > 0, "background required"

    components = _components(g)
    assert len(components) in (3, 4), "expected 3 or 4 nested regions"
    assert len({component["color"] for component in components}) == len(components), "layer colors must be distinct"

    h, w = g.shape
    for component in components:
        interior_local, bbox, hole_count = _interior_mask(component)
        assert hole_count == 1, "each boundary must enclose exactly one interior region"
        r0, c0, r1, c1 = bbox
        interior_full = np.zeros((h, w), dtype=bool)
        interior_full[r0 : r1 + 1, c0 : c1 + 1] = interior_local
        component["interior"] = interior_full

    for component in components:
        sample = tuple(map(int, component["coords"][0]))
        depth = 1 + sum(bool(other["interior"][sample]) for other in components if other is not component)
        component["depth"] = depth

    depths = sorted(component["depth"] for component in components)
    assert depths == list(range(1, len(components) + 1)), "regions must form a strict nesting chain"

    ordered = sorted(components, key=lambda component: component["depth"])
    for outer, inner in zip(ordered, ordered[1:]):
        sample = tuple(map(int, inner["coords"][0]))
        assert outer["interior"][sample], "every deeper region must lie inside the previous layer"
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate nested region colors according to depth order and foreground-parity direction.

    Input structure:
    - The grid has black background and 3 or 4 distinct non-black closed boundary components.
    - The components are 4-connected, each encloses exactly one interior region, and the regions form a strict nested chain.
    - The black space between consecutive layers stays empty, so nesting depth is readable from containment rather than from left-right position.

    Rule:
    - Sort the boundary components by nesting depth from outermost to innermost.
    - Read their colors in that depth order and count the total number of non-background pixels T.
    - If T is odd, shift the depth-ordered color list one step inward; if T is even, shift it one step outward.
    - Repaint each boundary component with its shifted color and keep all background pixels unchanged.

    Key insight:
    - The slots are defined by topological containment depth, while the direction of the cyclic recolor is chosen by the global foreground-pixel parity.
    """
    g = np.asarray(grid, dtype=int)
    ordered = _analyze(g)
    colors = [component["color"] for component in ordered]
    total = sum(component["area"] for component in ordered)
    rotated = colors[1:] + colors[:1] if total % 2 == 1 else colors[-1:] + colors[:-1]

    out = g.copy()
    for component, color in zip(ordered, rotated):
        coords = component["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)

    assert not np.array_equal(out, g), "distinct colors and cyclic shift must change the grid"
    return out


def _draw_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, plug: str | None) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    if plug == "tl":
        grid[r0 + 1, c0 + 1] = color
    elif plug == "tr":
        grid[r0 + 1, c1 - 1] = color
    elif plug == "bl":
        grid[r1 - 1, c0 + 1] = color
    elif plug == "br":
        grid[r1 - 1, c1 - 1] = color


def _ring_area(bbox: tuple[int, int, int, int], plug: str | None) -> int:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return 2 * (h + w) - 4 + int(plug is not None)


def _sample_boxes(rng: random.Random, layers: int) -> list[tuple[int, int, int, int]]:
    inner_h = rng.randint(4, 5 if layers == 4 else 6)
    inner_w = rng.randint(4, 5 if layers == 4 else 6)
    boxes = [(0, 0, inner_h - 1, inner_w - 1)]
    current = boxes[0]
    for _ in range(layers - 1):
        r0, c0, r1, c1 = current
        top = rng.randint(2, 3)
        bottom = rng.randint(2, 3)
        left = rng.randint(2, 3)
        right = rng.randint(2, 3)
        current = (r0 - top, c0 - left, r1 + bottom, c1 + right)
        boxes.append(current)
    return list(reversed(boxes))


def _compose_grid(
    rng: random.Random,
    boxes: list[tuple[int, int, int, int]],
    colors: list[int],
    target_parity: int,
) -> np.ndarray:
    layers = len(boxes)
    plug_bits = [rng.randint(0, 1) for _ in range(layers - 1)]
    plug_bits.append((target_parity - sum(plug_bits)) % 2)
    rng.shuffle(plug_bits)

    plugs = []
    plug_names = ["tl", "tr", "bl", "br"]
    for box, bit in zip(boxes, plug_bits):
        r0, c0, r1, c1 = box
        if bit == 1 and (r1 - r0 >= 3) and (c1 - c0 >= 3):
            plugs.append(rng.choice(plug_names))
        else:
            plugs.append(None)

    min_r = min(box[0] for box in boxes)
    min_c = min(box[1] for box in boxes)
    max_r = max(box[2] for box in boxes)
    max_c = max(box[3] for box in boxes)
    top_margin = rng.randint(1, 3)
    left_margin = rng.randint(1, 3)
    bottom_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)

    shift_r = top_margin - min_r
    shift_c = left_margin - min_c
    shifted_boxes = []
    for r0, c0, r1, c1 in boxes:
        shifted_boxes.append((r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c))

    height = max_r - min_r + 1 + top_margin + bottom_margin
    width = max_c - min_c + 1 + left_margin + right_margin
    assert max(height, width) <= 30, "grid must stay within ARC limits"

    grid = np.zeros((height, width), dtype=int)
    for box, color, plug in zip(shifted_boxes, colors, plugs):
        _draw_ring(grid, box, color, plug)

    total = sum(_ring_area(box, plug) for box, plug in zip(shifted_boxes, plugs))
    assert total % 2 == target_parity
    return grid


def _valid_generation(grid: np.ndarray, target_parity: int) -> bool:
    try:
        ordered = _analyze(grid)
    except AssertionError:
        return False

    total = sum(component["area"] for component in ordered)
    if total % 2 != target_parity:
        return False

    output = main(grid)
    if np.array_equal(output, grid):
        return False

    depths = [component["depth"] for component in ordered]
    if depths != list(range(1, len(ordered) + 1)):
        return False

    for component in ordered:
        if component["area"] < 8:
            return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        layers = rng.choice([3, 4])
        target_parity = rng.randint(0, 1)
        colors = rng.sample(PALETTE, layers)
        boxes = _sample_boxes(rng, layers)
        try:
            grid = _compose_grid(rng, boxes, colors, target_parity)
        except AssertionError:
            continue
        if _valid_generation(grid, target_parity):
            return grid
    raise RuntimeError("failed to generate a valid nested-region input")
