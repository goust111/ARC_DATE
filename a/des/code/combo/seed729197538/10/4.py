import math
import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components from a boolean mask."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rr, cc = np.argwhere(mask).T
    canvas[top + rr, left + cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort same-shaped objects by effective area (total area minus grey markers),
    then rotate their colors one step along that order and remove grey markers.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BLACK)

    objects = []
    for coords in _connected_components(g != BLACK):
        vals = [int(g[r, c]) for r, c in coords]
        non_grey = [v for v in vals if v != GREY]
        if not non_grey:
            continue

        base_color, _ = Counter(non_grey).most_common(1)[0]
        grey_count = sum(v == GREY for v in vals)
        objects.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "base_color": int(base_color),
                "effective_area": int(len(coords) - grey_count),
            }
        )

    if not objects:
        return out

    order = sorted(
        range(len(objects)),
        key=lambda i: (
            objects[i]["effective_area"],
            objects[i]["bbox"][0],
            objects[i]["bbox"][1],
        ),
    )

    for pos, idx in enumerate(order):
        prev_idx = order[pos - 1]
        new_color = int(objects[prev_idx]["base_color"])
        for r, c in objects[idx]["coords"]:
            out[r, c] = new_color

    return out


GEN_ATTEMPTS = 400
TEMPLATE_ATTEMPTS = 120
GRID_MAX = 30
GAP = 2
JITTER = 1
MARGIN_RANGE = (1, 3)
OBJ_COUNT_RANGE = (3, 6)
TEMPLATE_H_RANGE = (4, 7)
TEMPLATE_W_RANGE = (4, 7)
IRREGULAR_TEMPLATE_PROB = 0.8


def _boundary_cells(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)]


def _interior_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    inside: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]:
                inside.append((r, c))
    return inside


def _sample_template(min_interior: int) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    for _ in range(TEMPLATE_ATTEMPTS):
        h = random.randint(*TEMPLATE_H_RANGE)
        w = random.randint(*TEMPLATE_W_RANGE)
        if (h - 2) * (w - 2) < min_interior:
            continue

        mask = np.ones((h, w), dtype=bool)
        want_irregular = random.random() < IRREGULAR_TEMPLATE_PROB
        edge_left = {"top": w, "bottom": w, "left": h, "right": h}
        max_remove = max(0, len(_boundary_cells(h, w)) // 3)
        remove_target = random.randint(1, max_remove) if want_irregular and max_remove > 0 else 0

        cells = _boundary_cells(h, w)
        random.shuffle(cells)
        removed = 0
        for r, c in cells:
            if removed >= remove_target:
                break
            touches = []
            if r == 0:
                touches.append("top")
            if r == h - 1:
                touches.append("bottom")
            if c == 0:
                touches.append("left")
            if c == w - 1:
                touches.append("right")
            if any(edge_left[name] <= 1 for name in touches):
                continue
            mask[r, c] = False
            for name in touches:
                edge_left[name] -= 1
            removed += 1

        interior = _interior_cells(mask)
        if len(_connected_components(mask)) != 1:
            continue
        if len(interior) < min_interior:
            continue
        if want_irregular and remove_target > 0 and removed == 0:
            continue
        return mask, interior
    return None


def _sample_grey_counts(n_obj: int, max_count: int) -> list[int]:
    choices = list(range(max_count + 1))
    picked = {0, random.randint(1, max_count)}
    while len(picked) < n_obj:
        picked.add(random.choice(choices))
    counts = list(picked)
    random.shuffle(counts)
    return counts


def _feasible_layouts(n_obj: int, h: int, w: int) -> list[tuple[int, int]]:
    layouts = []
    slot_h = h + GAP + JITTER
    slot_w = w + GAP + JITTER
    for rows in range(1, min(3, n_obj) + 1):
        cols = math.ceil(n_obj / rows)
        max_h = max(MARGIN_RANGE) + (rows - 1) * slot_h + JITTER + h + max(MARGIN_RANGE)
        max_w = max(MARGIN_RANGE) + (cols - 1) * slot_w + JITTER + w + max(MARGIN_RANGE)
        if max_h <= GRID_MAX and max_w <= GRID_MAX:
            layouts.append((rows, cols))
    return layouts


def _sample_positions(n_obj: int, h: int, w: int) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    layouts = _feasible_layouts(n_obj, h, w)
    if not layouts:
        return None

    rows, cols = random.choice(layouts)
    slot_h = h + GAP + JITTER
    slot_w = w + GAP + JITTER
    top_margin = random.randint(*MARGIN_RANGE)
    left_margin = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)

    slots = [(r, c) for r in range(rows) for c in range(cols)]
    chosen = random.sample(slots, n_obj)
    random.shuffle(chosen)

    positions = []
    for sr, sc in chosen:
        top = top_margin + sr * slot_h + random.randint(0, JITTER)
        left = left_margin + sc * slot_w + random.randint(0, JITTER)
        positions.append((top, left))

    H = max(top for top, _ in positions) + h + bottom_margin
    W = max(left for _, left in positions) + w + right_margin
    return (H, W), positions


def _make_object_patch(
    mask: np.ndarray,
    interior: list[tuple[int, int]],
    color: int,
    grey_count: int,
) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    if grey_count:
        for r, c in random.sample(interior, grey_count):
            patch[r, c] = GREY
    return patch


def _build_expected(
    shape: tuple[int, int],
    mask: np.ndarray,
    objects: list[dict],
) -> np.ndarray:
    expected = np.zeros(shape, dtype=int)
    order = sorted(range(len(objects)), key=lambda i: objects[i]["effective_area"])
    for pos, idx in enumerate(order):
        prev_idx = order[pos - 1]
        new_color = int(objects[prev_idx]["base_color"])
        _paint_mask(expected, mask, objects[idx]["top"], objects[idx]["left"], new_color)
    return expected


def _gen_attempt() -> np.ndarray | None:
    n_obj = random.randint(*OBJ_COUNT_RANGE)
    template = _sample_template(n_obj - 1)
    if template is None:
        return None
    mask, interior = template
    h, w = mask.shape

    layout = _sample_positions(n_obj, h, w)
    if layout is None:
        return None
    shape, positions = layout

    grey_counts = _sample_grey_counts(n_obj, len(interior))
    colors = random.sample(OBJECT_COLORS, n_obj)

    grid = np.zeros(shape, dtype=int)
    objects = []
    total_area = int(mask.sum())
    rr, cc = np.argwhere(mask).T
    for (top, left), color, grey_count in zip(positions, colors, grey_counts):
        patch = _make_object_patch(mask, interior, color, grey_count)
        grid[top + rr, left + cc] = patch[rr, cc]
        objects.append(
            {
                "top": int(top),
                "left": int(left),
                "base_color": int(color),
                "effective_area": int(total_area - grey_count),
            }
        )

    if len({obj["effective_area"] for obj in objects}) != n_obj:
        return None

    expected = _build_expected(shape, mask, objects)
    comps = _connected_components(grid != BLACK)
    if len(comps) != n_obj:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
