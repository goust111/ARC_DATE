from __future__ import annotations

import random

import numpy as np


BG_COLOR = 3
FRAME_COLOR = 9
MARKER_COLOR = 7
ITEM_COLORS = (0, 1, 2, 4, 5, 6, 8)

DIRECTION_ORDER = ("up", "down", "left", "right")
DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

CONTAINER_COUNT_CHOICES = (2, 2, 3, 3, 4)
OUTER_SIZE_CHOICES = (5, 7, 9)
DISTINCT_COUNT_CHOICES = (2, 3, 4, 5)
RESERVE_PAD = 1
GEN_ATTEMPTS = 400


def _frame_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    mask = grid == FRAME_COLOR
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    bboxes: list[tuple[int, int, int, int]] = []

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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            box_h = r1 - r0 + 1
            box_w = c1 - c0 + 1
            assert box_h >= 3 and box_w >= 3, "frame must be at least 3x3"

            border = np.zeros((box_h, box_w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            assert len(coords) == int(border.sum()), "frame component must match a hollow rectangle"
            assert np.array_equal(sub == FRAME_COLOR, border), "frame component must be a rectangular ring"
            bboxes.append((r0, c0, r1, c1))

    bboxes.sort()
    return bboxes


def _marker_direction(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    h, w = map(int, grid.shape)

    candidates: list[tuple[str, tuple[int, int]]] = []
    checks = {
        "up": (r0 - 1, mid_c, r0, mid_c),
        "down": (r1 + 1, mid_c, r1, mid_c),
        "left": (mid_r, c0 - 1, mid_r, c0),
        "right": (mid_r, c1 + 1, mid_r, c1),
    }
    for direction, (mr, mc, sr, sc) in checks.items():
        if 0 <= mr < h and 0 <= mc < w and int(grid[mr, mc]) == MARKER_COLOR:
            candidates.append((direction, (sr, sc)))

    assert len(candidates) == 1, "each frame must have exactly one adjacent marker"
    return candidates[0]


def _distinct_item_count(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    colors = {
        int(color)
        for color in np.unique(interior)
        if int(color) not in {BG_COLOR, FRAME_COLOR, MARKER_COLOR}
    }
    return len(colors)


def _draw_tentacle(
    out: np.ndarray,
    start: tuple[int, int],
    direction: str,
    length: int,
) -> None:
    dr, dc = DELTAS[direction]
    h, w = map(int, out.shape)
    sr, sc = start
    for step in range(1, int(length) + 1):
        rr = sr + dr * step
        cc = sc + dc * step
        if not (0 <= rr < h and 0 <= cc < w):
            break
        if int(out[rr, cc]) == MARKER_COLOR:
            continue
        out[rr, cc] = FRAME_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count distinct item colors inside each frame and grow a maroon tentacle in the marked direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    frames = _frame_bboxes(g)
    assert frames, "expected at least one frame"
    for bbox in frames:
        direction, start = _marker_direction(g, bbox)
        length = _distinct_item_count(g, bbox)
        _draw_tentacle(out, start, direction, length)
    return out


def _reserve_bounds(
    rows: int,
    cols: int,
    r0: int,
    c0: int,
    outer_h: int,
    outer_w: int,
    direction: str,
    length: int,
) -> tuple[int, int, int, int]:
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    path_r0, path_r1 = r0, r1
    path_c0, path_c1 = c0, c1
    if direction == "up":
        path_r0 = r0 - length
    elif direction == "down":
        path_r1 = r1 + length
    elif direction == "left":
        path_c0 = c0 - length
    else:
        path_c1 = c1 + length

    return (
        max(0, path_r0 - RESERVE_PAD),
        min(rows - 1, path_r1 + RESERVE_PAD),
        max(0, path_c0 - RESERVE_PAD),
        min(cols - 1, path_c1 + RESERVE_PAD),
    )


def _choose_total_items(interior_area: int, distinct_count: int) -> int:
    total_max = min(interior_area, max(distinct_count + 1, interior_area // 2), distinct_count + 3)
    return random.randint(distinct_count + 1, total_max)


def _sample_items(
    outer_h: int,
    outer_w: int,
    distinct_count: int,
) -> list[tuple[int, int, int]]:
    interior_cells = [(r, c) for r in range(1, outer_h - 1) for c in range(1, outer_w - 1)]
    total_items = _choose_total_items(len(interior_cells), distinct_count)
    colors = random.sample(list(ITEM_COLORS), distinct_count)
    counts = [1] * distinct_count
    for _ in range(total_items - distinct_count):
        counts[random.randrange(distinct_count)] += 1

    picked_cells = random.sample(interior_cells, total_items)
    random.shuffle(picked_cells)
    items: list[tuple[int, int, int]] = []
    idx = 0
    for color, count in zip(colors, counts):
        for _ in range(count):
            rr, cc = picked_cells[idx]
            items.append((rr, cc, int(color)))
            idx += 1
    return items


def _can_use(occupied: np.ndarray, bounds: tuple[int, int, int, int]) -> bool:
    r0, r1, c0, c1 = bounds
    return not bool(np.any(occupied[r0 : r1 + 1, c0 : c1 + 1]))


def _mark_used(occupied: np.ndarray, bounds: tuple[int, int, int, int]) -> None:
    r0, r1, c0, c1 = bounds
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _sample_position(
    rows: int,
    cols: int,
    outer_h: int,
    outer_w: int,
    direction: str,
    length: int,
) -> tuple[int, int] | None:
    r_min = length if direction == "up" else 0
    r_max = rows - outer_h - (length if direction == "down" else 0)
    c_min = length if direction == "left" else 0
    c_max = cols - outer_w - (length if direction == "right" else 0)
    if r_min > r_max or c_min > c_max:
        return None
    return random.randint(r_min, r_max), random.randint(c_min, c_max)


def _direction_cells(
    r0: int,
    c0: int,
    outer_h: int,
    outer_w: int,
    direction: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if direction == "up":
        return (r0, mid_c), (r0 - 1, mid_c)
    if direction == "down":
        return (r1, mid_c), (r1 + 1, mid_c)
    if direction == "left":
        return (mid_r, c0), (mid_r, c0 - 1)
    return (mid_r, c1), (mid_r, c1 + 1)


def _draw_case(
    rows: int,
    cols: int,
    specs: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((rows, cols), BG_COLOR, dtype=int)
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        direction = str(spec["direction"])
        items = list(spec["items"])

        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        inp[r0 : r1 + 1, c0] = FRAME_COLOR
        inp[r0 : r1 + 1, c1] = FRAME_COLOR
        inp[r0, c0 : c1 + 1] = FRAME_COLOR
        inp[r1, c0 : c1 + 1] = FRAME_COLOR

        _start, marker = _direction_cells(r0, c0, outer_h, outer_w, direction)
        inp[marker] = MARKER_COLOR
        for rr, cc, color in items:
            inp[r0 + rr, c0 + cc] = int(color)

    out = inp.copy()
    for spec in specs:
        start, _marker = _direction_cells(
            int(spec["r0"]),
            int(spec["c0"]),
            int(spec["outer_h"]),
            int(spec["outer_w"]),
            str(spec["direction"]),
        )
        _draw_tentacle(out, start, str(spec["direction"]), int(spec["k"]))
    return inp, out


def _sample_case() -> np.ndarray | None:
    count = random.choice(CONTAINER_COUNT_CHOICES)
    rows = random.randint(16 + count, 30)
    cols = random.randint(16 + count, 30)
    directions = random.sample(list(DIRECTION_ORDER), count)
    lengths = random.sample(list(DISTINCT_COUNT_CHOICES), count)
    occupied = np.zeros((rows, cols), dtype=bool)
    specs: list[dict[str, object]] = []

    size_choices = OUTER_SIZE_CHOICES if count <= 3 else OUTER_SIZE_CHOICES[:2]
    for direction, length in zip(directions, lengths):
        placed = False
        for _ in range(120):
            outer_h = random.choice(size_choices)
            outer_w = random.choice(size_choices)
            if (outer_h - 2) * (outer_w - 2) < length + 1:
                continue

            pos = _sample_position(rows, cols, outer_h, outer_w, direction, length)
            if pos is None:
                continue
            r0, c0 = pos
            bounds = _reserve_bounds(rows, cols, r0, c0, outer_h, outer_w, direction, length)
            if not _can_use(occupied, bounds):
                continue

            items = _sample_items(outer_h, outer_w, length)
            specs.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "outer_h": outer_h,
                    "outer_w": outer_w,
                    "direction": direction,
                    "k": length,
                    "items": items,
                }
            )
            _mark_used(occupied, bounds)
            placed = True
            break
        if not placed:
            return None

    if len({int(spec["k"]) for spec in specs}) < 2:
        return None

    inp, expected = _draw_case(rows, cols, specs)
    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _sample_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
