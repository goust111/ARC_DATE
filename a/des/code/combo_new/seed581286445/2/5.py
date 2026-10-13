from __future__ import annotations

import random

import numpy as np


DIRS = ("up", "down", "left", "right")
BG = 0
GEN_RETRIES = 400
PLACE_RETRIES = 250
RING_COUNTS = (1, 2)
RING_SIZES = (3, 5, 7, 9)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _classify_ring(local_mask: np.ndarray) -> str:
    h, w = local_mask.shape
    assert h >= 3 and w >= 3, "rings must have an interior span"
    assert not bool(local_mask[1:-1, 1:-1].any()), "ring body must stay on the bbox border"

    top_gap = not bool(local_mask[0, 1:-1].any())
    bottom_gap = not bool(local_mask[-1, 1:-1].any())
    left_gap = not bool(local_mask[1:-1, 0].any())
    right_gap = not bool(local_mask[1:-1, -1].any())

    gaps = [name for name, missing in zip(DIRS, (top_gap, bottom_gap, left_gap, right_gap)) if missing]
    assert len(gaps) == 1, "expected exactly one missing side"
    gap = gaps[0]

    assert bool(local_mask[0, 0] and local_mask[0, -1] and local_mask[-1, 0] and local_mask[-1, -1]), "corners must exist"
    if gap != "up":
        assert bool(local_mask[0, 1:-1].all())
    if gap != "down":
        assert bool(local_mask[-1, 1:-1].all())
    if gap != "left":
        assert bool(local_mask[1:-1, 0].all())
    if gap != "right":
        assert bool(local_mask[1:-1, -1].all())
    return gap


def _draw_ring(canvas: np.ndarray, center: tuple[int, int], height: int, width: int, direction: str, color: int) -> None:
    h, w = canvas.shape
    cr, cc = center
    top = int(cr - height // 2)
    left = int(cc - width // 2)
    bottom = int(top + height - 1)
    right = int(left + width - 1)

    def paint(r: int, c: int) -> None:
        if 0 <= r < h and 0 <= c < w:
            canvas[int(r), int(c)] = int(color)

    if direction != "up":
        for c in range(left, right + 1):
            paint(int(top), int(c))
    if direction != "down":
        for c in range(left, right + 1):
            paint(int(bottom), int(c))
    if direction != "left":
        for r in range(top, bottom + 1):
            paint(int(r), int(left))
    if direction != "right":
        for r in range(top, bottom + 1):
            paint(int(r), int(right))


def _ring_patch(height: int, width: int, direction: str, color: int) -> np.ndarray:
    patch = np.full((int(height), int(width)), BG, dtype=int)
    _draw_ring(patch, (int(height // 2), int(width // 2)), int(height), int(width), direction, color)
    return patch


def _find_markers(grid: np.ndarray, bg: int) -> dict[str, tuple[tuple[int, int], int]]:
    h, w = grid.shape
    markers: dict[str, tuple[tuple[int, int], int]] = {}
    for comp in _components(grid != int(bg)):
        if int(comp.shape[0]) != 1:
            continue
        r, c = map(int, comp[0])
        if r == 0 and 0 < c < w - 1:
            side = "up"
        elif r == h - 1 and 0 < c < w - 1:
            side = "down"
        elif c == 0 and 0 < r < h - 1:
            side = "left"
        elif c == w - 1 and 0 < r < h - 1:
            side = "right"
        else:
            continue
        assert side not in markers, "each side can have at most one marker"
        markers[side] = ((int(r), int(c)), int(grid[r, c]))
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group broken rectangular rings by opening direction and rebuild each group as a concentric stack at the matching border marker.

    Input structure:
    - The background is black.
    - Foreground objects are monochrome one-cell-thick rectangular frames with exactly one missing side, plus isolated single-pixel boundary markers.
    - Each missing side points up, down, left, or right; the markers sit on the corresponding top, bottom, left, or right border.

    Rule:
    - Detect the isolated boundary singletons as anchors and remove them temporarily from object detection.
    - Classify every remaining frame by which border side is missing, then sort each direction-group by bounding-box area from largest to smallest.
    - Clear the original frames and redraw each group at its matching border anchor so that all frames in the group share that anchor as their center, while keeping the same missing-side direction; out-of-bounds pixels are clipped.
    - Restore the boundary markers unchanged in the output.

    Key insight:
    - The singleton marker identifies which border should host a direction-group, and the frame bounding boxes determine the nesting order inside that centered stack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(BG)

    markers = _find_markers(g, int(bg))
    marker_cells = {tuple(pos) for pos, _color in markers.values()}

    work = g.copy()
    for r, c in marker_cells:
        work[int(r), int(c)] = int(bg)

    groups: dict[str, list[dict[str, int]]] = {direction: [] for direction in DIRS}
    for comp in _components(work != int(bg)):
        r0, c0, r1, c1 = _bbox(comp)
        patch = work[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        colors = sorted(set(map(int, patch[patch != int(bg)].flatten())))
        assert len(colors) == 1, "each ring must be monochrome"

        local = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
        for r, c in comp:
            local[int(r - r0), int(c - c0)] = True

        direction = _classify_ring(local)
        groups[direction].append(
            {
                "height": int(local.shape[0]),
                "width": int(local.shape[1]),
                "area": int(local.shape[0] * local.shape[1]),
                "color": int(colors[0]),
            }
        )

    out = np.full_like(g, int(bg))
    for side, (pos, color) in markers.items():
        _ = side
        out[int(pos[0]), int(pos[1])] = int(color)

    for direction in DIRS:
        if direction not in markers:
            continue
        center = markers[direction][0]
        items = sorted(groups[direction], key=lambda item: int(item["area"]), reverse=True)
        for item in items:
            _draw_ring(out, center, int(item["height"]), int(item["width"]), direction, int(item["color"]))

    for pos, color in markers.values():
        out[int(pos[0]), int(pos[1])] = int(color)
    return out


def _sample_group_sizes(count: int) -> list[tuple[int, int]]:
    assert count in RING_COUNTS
    min_size = 3 + 2 * (int(count) - 1)
    candidates = [size for size in RING_SIZES if size >= min_size]
    largest_h = random.choice(candidates)
    largest_w = random.choice(candidates)
    return [(int(largest_h - 2 * i), int(largest_w - 2 * i)) for i in range(int(count))]


def _ring_cells(center: tuple[int, int], height: int, width: int, direction: str, shape: tuple[int, int]) -> set[tuple[int, int]]:
    canvas = np.full(shape, BG, dtype=int)
    _draw_ring(canvas, center, int(height), int(width), direction, 1)
    return {tuple(map(int, rc)) for rc in np.argwhere(canvas == 1)}


def _build_expected(
    shape: tuple[int, int],
    ring_color: int,
    marker_color: int,
    anchors: dict[str, tuple[int, int]],
    groups: dict[str, list[tuple[int, int]]],
) -> np.ndarray | None:
    h, w = shape
    out = np.full((int(h), int(w)), BG, dtype=int)
    marker_cells = set()
    for side, pos in anchors.items():
        _ = side
        marker_cells.add((int(pos[0]), int(pos[1])))
        out[int(pos[0]), int(pos[1])] = int(marker_color)

    occupied: set[tuple[int, int]] = set()
    for direction in DIRS:
        for height, width in sorted(groups[direction], key=lambda item: item[0] * item[1], reverse=True):
            cells = _ring_cells(anchors[direction], int(height), int(width), direction, (int(h), int(w)))
            if cells & marker_cells:
                return None
            if cells & occupied:
                return None
            occupied |= cells
            for r, c in cells:
                out[int(r), int(c)] = int(ring_color)
    return out


def _separated(candidate: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], pad: int = 1) -> bool:
    r0, c0, r1, c1 = candidate
    for rr0, cc0, rr1, cc1 in others:
        if not (r1 + pad < rr0 or rr1 + pad < r0 or c1 + pad < cc0 or cc1 + pad < c0):
            return False
    return True


def _sample_anchors(shape: tuple[int, int], groups: dict[str, list[tuple[int, int]]]) -> dict[str, tuple[int, int]] | None:
    h, w = shape
    top_half_w = max(width // 2 for width in [size[1] for size in groups["up"]])
    bottom_half_w = max(width // 2 for width in [size[1] for size in groups["down"]])
    left_half_h = max(height // 2 for height in [size[0] for size in groups["left"]])
    right_half_h = max(height // 2 for height in [size[0] for size in groups["right"]])

    top_cols = list(range(int(top_half_w), int(w - top_half_w)))
    bottom_cols = list(range(int(bottom_half_w), int(w - bottom_half_w)))
    left_rows = list(range(int(left_half_h), int(h - left_half_h)))
    right_rows = list(range(int(right_half_h), int(h - right_half_h)))
    if not top_cols or not bottom_cols or not left_rows or not right_rows:
        return None

    anchors = {
        "up": (0, random.choice(top_cols)),
        "down": (int(h - 1), random.choice(bottom_cols)),
        "left": (random.choice(left_rows), 0),
        "right": (random.choice(right_rows), int(w - 1)),
    }
    return anchors


def _sample_layout() -> tuple[dict[str, list[tuple[int, int]]], dict[str, tuple[int, int]]] | None:
    h = random.randint(20, 28)
    w = random.randint(20, 28)

    counts = {direction: random.randint(*RING_COUNTS) for direction in DIRS}
    if sum(1 for count in counts.values() if count == 2) < 2:
        return None

    groups = {direction: _sample_group_sizes(counts[direction]) for direction in DIRS}
    anchors = _sample_anchors((int(h), int(w)), groups)
    if anchors is None:
        return None
    return groups, anchors


def _place_input_rings(
    shape: tuple[int, int],
    ring_color: int,
    anchors: dict[str, tuple[int, int]],
    groups: dict[str, list[tuple[int, int]]],
) -> np.ndarray | None:
    h, w = shape
    grid = np.full((int(h), int(w)), BG, dtype=int)
    reserved: set[tuple[int, int]] = set()
    for pos in anchors.values():
        r, c = map(int, pos)
        reserved.add((int(r), int(c)))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                reserved.add((rr, cc))

    specs: list[tuple[int, int, str]] = []
    for direction in DIRS:
        for height, width in groups[direction]:
            specs.append((int(height), int(width), direction))
    random.shuffle(specs)
    specs.sort(key=lambda item: item[0] * item[1], reverse=True)

    placed_boxes: list[tuple[int, int, int, int]] = []
    for height, width, direction in specs:
        patch = _ring_patch(int(height), int(width), direction, int(ring_color))
        cells = np.argwhere(patch != BG)
        placed = False
        for _ in range(PLACE_RETRIES):
            top = random.randint(1, int(h - height - 1))
            left = random.randint(1, int(w - width - 1))
            box = (int(top), int(left), int(top + height - 1), int(left + width - 1))
            if not _separated(box, placed_boxes, pad=1):
                continue
            absolute_cells = {(int(top + r), int(left + c)) for r, c in cells}
            if absolute_cells & reserved:
                continue
            for r, c in absolute_cells:
                grid[int(r), int(c)] = int(ring_color)
            placed_boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    return grid


def _attempt_generate() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    groups, anchors = layout

    colors = list(range(1, 10))
    random.shuffle(colors)
    ring_color = int(colors[0])
    marker_color = int(colors[1])

    shape = (int(max(pos[0] for pos in anchors.values()) + 1), int(max(pos[1] for pos in anchors.values()) + 1))
    if shape[0] < 20 or shape[1] < 20:
        return None

    expected = _build_expected(shape, int(ring_color), int(marker_color), anchors, groups)
    if expected is None:
        return None

    inp = _place_input_rings(shape, int(ring_color), anchors, groups)
    if inp is None:
        return None

    for r, c in anchors.values():
        inp[int(r), int(c)] = int(marker_color)

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
