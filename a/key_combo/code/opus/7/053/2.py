from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
KEY_COLORS = [4, 5, 6, 7, 8, 9]

FRAME_COUNT_RANGE = (3, 5)
FRAME_INNER_H_RANGE = (3, 5)
FRAME_INNER_W_RANGE = (3, 5)
GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (18, 28)
MAX_ATTEMPTS = 400
MIN_GAP = 1


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shape_patch(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    h = max(rows) + 1
    w = max(cols) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _frame_candidates(g: np.ndarray) -> list[dict]:
    comps = _components(g == BLUE)
    frames = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue
        local = np.zeros((h, w), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        if not np.array_equal(local, _frame_mask(h, w)):
            continue
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        black_coords = np.argwhere(interior == BLACK)
        assert black_coords.size > 0, "each frame must contain black interior cells"
        assert len(_components(interior == BLACK)) == 1, "black gap must be a single 4-connected shape"
        colors = set(map(int, np.unique(interior)))
        assert colors <= {BLACK, RED, GREEN}
        assert RED in colors and GREEN in colors
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap_shape": _normalize(black_coords),
            }
        )
    assert len(frames) >= 3
    return sorted(frames, key=lambda item: item["bbox"])


def _key_shapes(g: np.ndarray) -> list[dict]:
    key_mask = np.isin(g, KEY_COLORS)
    comps = _components(key_mask)
    shapes = []
    for coords in comps:
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each key shape must be monochromatic"
        color = next(iter(colors))
        assert color in KEY_COLORS
        shapes.append({"coords": coords, "color": color, "shape": _normalize(coords)})
    return shapes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match scattered key-shapes to frame interior gaps, then swap red and green only inside matched frames.

    Input structure:
    - The background is black.
    - Several one-cell-thick blue rectangular frames contain interiors made only of black, red, and green.
    - Separate small monochromatic 4-connected key shapes use colors outside black/blue/red/green.

    Rule:
    - For each frame, take the black cells strictly inside the blue border and normalize that gap shape by translation.
    - Normalize every scattered key shape the same way; a frame is activated iff exactly one key has the same shape.
    - In activated frames swap red and green inside the border, leave unmatched frames unchanged, and erase all key shapes to black.

    Key insight:
    - The scattered shapes are not pasted into the frames; they act as geometric keys that gate whether the frame's internal red/green pattern should be swapped.
    """
    g = np.asarray(grid, dtype=int)
    frames = _frame_candidates(g)
    keys = _key_shapes(g)

    by_shape: dict[tuple[tuple[int, int], ...], list[dict]] = {}
    for key in keys:
        by_shape.setdefault(key["shape"], []).append(key)
    for entries in by_shape.values():
        assert len(entries) == 1, "key shapes must be unique by geometry"

    out = g.copy()
    for key in keys:
        out[key["coords"][:, 0], key["coords"][:, 1]] = BLACK

    for frame in frames:
        if frame["gap_shape"] not in by_shape:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1].copy()
        red_mask = interior == RED
        green_mask = interior == GREEN
        interior[red_mask] = GREEN
        interior[green_mask] = RED
        out[r0 + 1 : r1, c0 + 1 : c1] = interior

    return out


def _random_shape(max_h: int, max_w: int, area: int) -> tuple[tuple[int, int], ...] | None:
    shape = {(0, 0)}
    while len(shape) < area:
        frontier = set()
        for cell in shape:
            frontier.update(_neighbors4(*cell))
        frontier -= shape
        candidates = []
        for nr, nc in frontier:
            cells = shape | {(nr, nc)}
            rows = [r for r, _c in cells]
            cols = [c for _r, c in cells]
            if max(rows) - min(rows) + 1 <= max_h and max(cols) - min(cols) + 1 <= max_w:
                candidates.append((nr, nc))
        if not candidates:
            return None
        shape.add(random.choice(candidates))
    return _normalize(np.asarray(sorted(shape), dtype=int))


def _sample_unique_shapes(count: int) -> list[tuple[tuple[int, int], ...]] | None:
    shapes: list[tuple[tuple[int, int], ...]] = []
    seen = set()
    for _ in range(count):
        for _ in range(80):
            inner_h = random.randint(*FRAME_INNER_H_RANGE)
            inner_w = random.randint(*FRAME_INNER_W_RANGE)
            max_area = min(8, inner_h * inner_w - 2)
            if max_area < 3:
                continue
            area = random.randint(3, max_area)
            shape = _random_shape(inner_h, inner_w, area)
            if shape is None or shape in seen:
                continue
            seen.add(shape)
            shapes.append(shape)
            break
        else:
            return None
    return shapes


def _inflate_bbox(shape: tuple[tuple[int, int], ...], top: int, left: int) -> tuple[int, int, int, int]:
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    return top, left, top + max(rows), left + max(cols)


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    if a[2] < b[0]:
        dr = b[0] - a[2] - 1
    elif b[2] < a[0]:
        dr = a[0] - b[2] - 1
    else:
        dr = 0
    if a[3] < b[1]:
        dc = b[1] - a[3] - 1
    elif b[3] < a[1]:
        dc = a[1] - b[3] - 1
    else:
        dc = 0
    return dr + dc


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    occupied_boxes: list[tuple[int, int, int, int]],
    row_range: tuple[int, int],
    col_range: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = patch.shape
    r_start, r_stop = row_range
    c_start, c_stop = col_range
    candidates = []
    mask = patch != BLACK
    for r0 in range(r_start, r_stop - h + 1):
        for c0 in range(c_start, c_stop - w + 1):
            if np.any(grid[r0 : r0 + h, c0 : c0 + w][mask] != BLACK):
                continue
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_bbox_gap(box, other) < MIN_GAP for other in occupied_boxes):
                continue
            candidates.append((r0, c0, box))
    if not candidates:
        return None
    r0, c0, box = random.choice(candidates)
    grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
    occupied_boxes.append(box)
    return r0, c0


def _make_frame_patch(
    inner_h: int,
    inner_w: int,
    gap_shape: tuple[tuple[int, int], ...],
) -> np.ndarray | None:
    patch = np.full((inner_h + 2, inner_w + 2), BLUE, dtype=int)
    interior = patch[1:-1, 1:-1]
    interior[:, :] = RED

    shape_h = max(r for r, _c in gap_shape) + 1
    shape_w = max(c for _r, c in gap_shape) + 1
    max_dr = inner_h - shape_h
    max_dc = inner_w - shape_w
    if max_dr < 0 or max_dc < 0:
        return None
    off_r = random.randint(0, max_dr)
    off_c = random.randint(0, max_dc)
    for r, c in gap_shape:
        interior[off_r + r, off_c + c] = BLACK

    remaining = [(r, c) for r in range(inner_h) for c in range(inner_w) if interior[r, c] != BLACK]
    if len(remaining) < 2:
        return None
    green_count = random.randint(1, len(remaining) - 1)
    green_cells = set(random.sample(remaining, green_count))
    for r, c in remaining:
        interior[r, c] = GREEN if (r, c) in green_cells else RED
    return patch


def _frame_gap_shape(frame_patch: np.ndarray) -> tuple[tuple[int, int], ...]:
    interior = frame_patch[1:-1, 1:-1]
    return _normalize(np.argwhere(interior == BLACK))


def _expected_output(grid: np.ndarray) -> np.ndarray:
    return main(grid)


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    shapes = _sample_unique_shapes(frame_count)
    if shapes is None:
        return None

    active_count = random.randint(2, frame_count - 1)
    active_indices = set(random.sample(range(frame_count), active_count))

    frame_dims = [
        (
            random.randint(*FRAME_INNER_H_RANGE),
            random.randint(*FRAME_INNER_W_RANGE),
        )
        for _ in range(frame_count)
    ]
    frame_patches = []
    for (inner_h, inner_w), shape in zip(frame_dims, shapes):
        patch = _make_frame_patch(inner_h, inner_w, shape)
        if patch is None:
            return None
        if _frame_gap_shape(patch) != shape:
            return None
        frame_patches.append(patch)

    frame_widths = [patch.shape[1] for patch in frame_patches]
    frame_heights = [patch.shape[0] for patch in frame_patches]
    key_patches = []
    for idx in sorted(active_indices):
        key_color = KEY_COLORS[len(key_patches)]
        key_patches.append(_shape_patch(shapes[idx], key_color))

    min_grid_h = max(
        GRID_H_RANGE[0],
        max(frame_heights) + 2,
        sum(sorted(frame_heights)[:2]) + 3,
        max((patch.shape[0] for patch in key_patches), default=1) + 2,
    )
    min_grid_w = max(
        GRID_W_RANGE[0],
        max(frame_widths) + max((patch.shape[1] for patch in key_patches), default=1) + 7,
    )
    if min_grid_h > GRID_H_RANGE[1] or min_grid_w > GRID_W_RANGE[1]:
        return None

    height = random.randint(min_grid_h, GRID_H_RANGE[1])
    width = random.randint(min_grid_w, GRID_W_RANGE[1])
    grid = np.zeros((height, width), dtype=int)
    occupied_boxes: list[tuple[int, int, int, int]] = []

    left_limit = width // 2 + random.randint(-1, 1)
    left_limit = max(max(frame_widths) + 1, min(left_limit, width - 5))
    if left_limit <= max(frame_widths):
        return None
    right_start = left_limit + 2
    if right_start >= width - 1:
        return None

    frames_info = []
    order = list(range(frame_count))
    random.shuffle(order)
    for idx in order:
        patch = frame_patches[idx]
        pos = _place_patch(
            grid,
            patch,
            occupied_boxes,
            (1, height - 1),
            (1, left_limit),
        )
        if pos is None:
            return None
        r0, c0 = pos
        frames_info.append((idx, r0, c0))

    keys_info = []
    key_boxes: list[tuple[int, int, int, int]] = []
    for key_patch in key_patches:
        pos = _place_patch(
            grid,
            key_patch,
            occupied_boxes + key_boxes,
            (1, height - 1),
            (right_start, width - 1),
        )
        if pos is None:
            return None
        r0, c0 = pos
        box = (r0, c0, r0 + key_patch.shape[0] - 1, c0 + key_patch.shape[1] - 1)
        key_boxes.append(box)
        keys_info.append((r0, c0))

    if len(_frame_candidates(grid)) != frame_count:
        return None
    keys = _key_shapes(grid)
    if len(keys) != active_count:
        return None

    frame_shapes = Counter(frame["gap_shape"] for frame in _frame_candidates(grid))
    key_shapes = Counter(key["shape"] for key in keys)
    if any(count != 1 for count in frame_shapes.values()):
        return None
    if any(count != 1 for count in key_shapes.values()):
        return None
    if not set(key_shapes).issubset(set(frame_shapes)):
        return None
    if len(key_shapes) >= len(frame_shapes):
        return None

    out = _expected_output(grid)
    if np.array_equal(grid, out):
        return None
    if np.any(np.isin(out, KEY_COLORS)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
