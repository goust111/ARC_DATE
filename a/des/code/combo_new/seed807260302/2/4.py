from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREY = 5
SEED_COLORS = (1, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500


def _room_components(g: np.ndarray) -> list[dict]:
    """Return 4-connected non-fence rectangular rooms."""
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    rooms: list[dict] = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(g[sr, sc]) == GREY:
                continue

            q: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(g[rr, cc]) != GREY:
                        seen[rr, cc] = True
                        q.append((rr, cc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            expected_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
            assert int(arr.shape[0]) == expected_area, "fences must split the grid into rectangular rooms"
            assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] != GREY), "room interior cannot contain fence pixels"
            rooms.append({"coords": arr, "bbox": (r0, c0, r1, c1)})

    return rooms


def _paint_cross(out: np.ndarray, fence_mask: np.ndarray, center: tuple[int, int], color: int, arm: int) -> None:
    h, w = map(int, out.shape)
    r, c = map(int, center)
    assert 0 <= r < h and 0 <= c < w and not bool(fence_mask[r, c])
    out[r, c] = int(color)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        for dist in range(1, int(arm) + 1):
            rr = r + dr * dist
            cc = c + dc * dist
            if rr < 0 or rr >= h or cc < 0 or cc >= w or bool(fence_mask[rr, cc]):
                break
            out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each room's scattered seeds and red marker with a seed-count-sized cross.

    Input structure:
    - Black background with one-pixel grey horizontal and/or vertical fence lines splitting the grid into rectangular rooms.
    - Each room contains exactly one red marker pixel and one to four scattered seed pixels of a single non-black, non-red, non-grey color.
    - Seed positions are arbitrary within their room; the red marker gives the intended cross center.

    Rule:
    - For each room, count its seed pixels N and read their shared seed color.
    - Clear all non-fence pixels back to black, removing the original seeds and red marker.
    - From the red marker, draw a cross in the seed color with arms of length N in the four cardinal directions, including the center.
    - Stop each arm before it would enter a grey fence or leave the grid; all grey fences remain unchanged.

    Key insight:
    - The fence partition defines independent rooms, seed count encodes size, seed color encodes paint color, and the red marker encodes position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fence_mask = g == GREY
    out = np.full(g.shape, BG, dtype=int)
    out[fence_mask] = GREY

    rooms = _room_components(g)
    assert rooms, "expected at least one room"
    for room in rooms:
        coords = room["coords"]
        vals = g[coords[:, 0], coords[:, 1]]

        red_hits = np.argwhere(vals == RED)
        assert int(red_hits.shape[0]) == 1, "each room must contain exactly one red center marker"
        center_idx = int(red_hits[0, 0])
        center = tuple(map(int, coords[center_idx]))

        seed_mask = (vals != BG) & (vals != RED) & (vals != GREY)
        seed_vals = vals[seed_mask]
        assert 1 <= int(seed_vals.shape[0]) <= 4, "each room must contain one to four seeds"
        seed_colors = set(map(int, seed_vals))
        assert len(seed_colors) == 1, "all seeds in one room must share a color"
        seed_color = int(next(iter(seed_colors)))

        _paint_cross(out, fence_mask, center, seed_color, int(seed_vals.shape[0]))

    return out


# --------------- generator ---------------


def _sample_sizes(count: int) -> list[int] | None:
    max_size = 14 if count == 1 else 9
    for _ in range(100):
        sizes = [random.randint(3, max_size) for _ in range(count)]
        if sum(sizes) + count - 1 <= 30:
            return sizes
    return None


def _room_grid() -> tuple[list[int], list[int]] | None:
    choices = [(r, c) for r in range(1, 5) for c in range(1, 5) if 4 <= r * c <= 9]
    random.shuffle(choices)
    for room_rows, room_cols in choices:
        heights = _sample_sizes(room_rows)
        widths = _sample_sizes(room_cols)
        if heights is None or widths is None:
            continue
        if sum(heights) + room_rows - 1 <= 30 and sum(widths) + room_cols - 1 <= 30:
            return heights, widths
    return None


def _bounds_from_sizes(heights: list[int], widths: list[int]) -> list[tuple[int, int, int, int]]:
    row_bounds: list[tuple[int, int]] = []
    r = 0
    for height in heights:
        row_bounds.append((r, r + int(height) - 1))
        r += int(height) + 1

    col_bounds: list[tuple[int, int]] = []
    c = 0
    for width in widths:
        col_bounds.append((c, c + int(width) - 1))
        c += int(width) + 1

    return [(r0, c0, r1, c1) for r0, r1 in row_bounds for c0, c1 in col_bounds]


def _draw_fences(grid: np.ndarray, heights: list[int], widths: list[int]) -> None:
    r = 0
    for height in heights[:-1]:
        r += int(height)
        grid[r, :] = GREY
        r += 1

    c = 0
    for width in widths[:-1]:
        c += int(width)
        grid[:, c] = GREY
        c += 1


def _would_truncate(center: tuple[int, int], bbox: tuple[int, int, int, int], arm: int) -> bool:
    r, c = map(int, center)
    r0, c0, r1, c1 = map(int, bbox)
    return min(r - r0, r1 - r, c - c0, c1 - c) < int(arm)


def _center_candidates(
    bbox: tuple[int, int, int, int], arm: int, want_truncated: bool
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    candidates: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if _would_truncate((r, c), bbox, int(arm)) == bool(want_truncated):
                candidates.append((r, c))
    return candidates


def _arm_counts(room_count: int) -> list[int]:
    counts = [1, 2, 3, 4]
    while len(counts) < room_count:
        counts.append(random.randint(1, 4))
    random.shuffle(counts)
    return counts[:room_count]


def _paint_cross_in_bbox(
    out: np.ndarray, bbox: tuple[int, int, int, int], center: tuple[int, int], color: int, arm: int
) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    cr, cc = map(int, center)
    out[cr, cc] = int(color)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        for dist in range(1, int(arm) + 1):
            rr = cr + dr * dist
            cc2 = cc + dc * dist
            if not (r0 <= rr <= r1 and c0 <= cc2 <= c1):
                break
            out[rr, cc2] = int(color)


def _populate_rooms(
    grid: np.ndarray, rooms: list[tuple[int, int, int, int]]
) -> list[dict] | None:
    random.shuffle(rooms)
    counts = _arm_counts(len(rooms))

    colors = [int(random.choice(SEED_COLORS)) for _ in rooms]
    if len(set(colors)) == 1 and len(colors) > 1:
        alternatives = [c for c in SEED_COLORS if c != colors[0]]
        colors[-1] = int(random.choice(alternatives))

    safe_possible = [
        i for i, (bbox, arm) in enumerate(zip(rooms, counts)) if _center_candidates(bbox, int(arm), False)
    ]
    if not safe_possible:
        return None
    safe_idx = int(random.choice(safe_possible))
    trunc_choices = [i for i in range(len(rooms)) if i != safe_idx]
    if not trunc_choices:
        return None
    trunc_idx = int(random.choice(trunc_choices))

    records: list[dict] = []
    for i, (bbox, arm, color) in enumerate(zip(rooms, counts, colors)):
        if i == safe_idx:
            want_truncated = False
        elif i == trunc_idx:
            want_truncated = True
        else:
            want_truncated = random.random() < 0.5

        candidates = _center_candidates(bbox, int(arm), want_truncated)
        if not candidates:
            candidates = _center_candidates(bbox, int(arm), True)
        if not candidates:
            return None
        center = random.choice(candidates)

        r0, c0, r1, c1 = map(int, bbox)
        seed_candidates = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) != center]
        if len(seed_candidates) < int(arm):
            return None
        seed_cells = random.sample(seed_candidates, int(arm))

        grid[center] = RED
        for cell in seed_cells:
            grid[cell] = int(color)

        records.append(
            {
                "bbox": bbox,
                "center": center,
                "color": int(color),
                "arm": int(arm),
                "truncated": _would_truncate(center, bbox, int(arm)),
            }
        )

    if not any(rec["truncated"] for rec in records):
        return None
    if not any(not rec["truncated"] for rec in records):
        return None
    return records


def _expected_output(inp: np.ndarray, records: list[dict]) -> np.ndarray:
    out = np.full(inp.shape, BG, dtype=int)
    out[inp == GREY] = GREY
    for rec in records:
        _paint_cross_in_bbox(out, rec["bbox"], rec["center"], rec["color"], rec["arm"])
    return out


def _gen_attempt() -> np.ndarray | None:
    sampled = _room_grid()
    if sampled is None:
        return None
    heights, widths = sampled

    rows = sum(heights) + len(heights) - 1
    cols = sum(widths) + len(widths) - 1
    if max(rows, cols) > 30:
        return None

    grid = np.zeros((rows, cols), dtype=int)
    _draw_fences(grid, heights, widths)
    rooms = _bounds_from_sizes(heights, widths)

    records = _populate_rooms(grid, rooms)
    if records is None:
        return None

    expected = _expected_output(grid, records)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    if max(grid.shape) > 30 or max(actual.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fenced-room cross input")
