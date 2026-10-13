import random
from collections import deque

import numpy as np


BG = 0


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_palette_segments(grid: np.ndarray) -> list[dict[str, int]]:
    row = np.asarray(grid[0], dtype=int)
    segments: list[dict[str, int]] = []
    w = row.shape[0]
    c = 0
    while c < w:
        color = int(row[c])
        if color == BG:
            c += 1
            continue
        start = c
        while c < w and int(row[c]) == color:
            c += 1
        segments.append({"color": color, "start": start, "width": c - start})
    return segments


def _closed_black_regions(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    enclosed: list[list[tuple[int, int]]] = []
    for comp in _components(g == BG):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        enclosed.append(comp)
    return enclosed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match palette-block width rank to enclosed-region area rank.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    segments = sorted(
        _parse_palette_segments(g),
        key=lambda seg: (int(seg["width"]), int(seg["start"])),
    )
    regions = sorted(
        _closed_black_regions(g),
        key=lambda comp: (
            len(comp),
            min(r for r, _c in comp),
            min(c for _r, c in comp),
        ),
    )

    for region, segment in zip(regions, segments, strict=False):
        color = int(segment["color"])
        for r, c in region:
            out[r, c] = color

    return out


GEN_RETRIES = 500
GRID_H_RANGE = (15, 24)
GRID_W_RANGE = (20, 30)
ROOM_COUNT_RANGE = (2, 5)
TOP_GAP_RANGE = (1, 2)
BOTTOM_GAP_RANGE = (1, 2)
SIDE_GAP_RANGE = (1, 2)
MIN_ROOM_HEIGHT = 2
MIN_ROOM_WIDTH = 2
STUB_COUNT_RANGE = (0, 2)
STUB_LEN_RANGE = (2, 4)


def _randint(rng: random.Random, bounds: tuple[int, int]) -> int:
    return rng.randint(int(bounds[0]), int(bounds[1]))


def _sample_distinct_widths(total: int, count: int, rng: random.Random) -> list[int] | None:
    if total < count * 2:
        return None
    for _ in range(300):
        cuts = sorted(rng.sample(range(1, total), count - 1))
        prev = 0
        widths = []
        for cut in cuts + [total]:
            widths.append(cut - prev)
            prev = cut
        if min(widths) >= 2 and len(set(widths)) == count:
            return widths
    return None


def _split_options(room: tuple[int, int, int, int]) -> list[tuple[str, list[int]]]:
    r0, r1, c0, c1 = room
    options: list[tuple[str, list[int]]] = []

    min_col = c0 + 1 + MIN_ROOM_WIDTH
    max_col = c1 - 1 - MIN_ROOM_WIDTH
    if min_col <= max_col:
        options.append(("v", list(range(min_col, max_col + 1))))

    min_row = r0 + 1 + MIN_ROOM_HEIGHT
    max_row = r1 - 1 - MIN_ROOM_HEIGHT
    if min_row <= max_row:
        options.append(("h", list(range(min_row, max_row + 1))))

    return options


def _generate_partition(
    height: int,
    width: int,
    room_count: int,
    rng: random.Random,
) -> tuple[np.ndarray, list[int]] | None:
    top_gap = _randint(rng, TOP_GAP_RANGE)
    bottom_gap = _randint(rng, BOTTOM_GAP_RANGE)
    left_gap = _randint(rng, SIDE_GAP_RANGE)
    right_gap = _randint(rng, SIDE_GAP_RANGE)

    r0 = 1 + top_gap
    r1 = height - 1 - bottom_gap
    c0 = left_gap
    c1 = width - 1 - right_gap

    if r1 - r0 - 1 < MIN_ROOM_HEIGHT or c1 - c0 - 1 < MIN_ROOM_WIDTH:
        return None

    rooms = [(r0, r1, c0, c1)]
    walls: list[tuple[str, int, int, int]] = []

    while len(rooms) < room_count:
        feasible: list[tuple[int, tuple[int, int, int, int], list[tuple[str, list[int]]]]] = []
        for idx, room in enumerate(rooms):
            options = _split_options(room)
            if options:
                feasible.append((idx, room, options))
        if not feasible:
            return None

        idx, room, options = rng.choice(feasible)
        orient, positions = rng.choice(options)
        pos = rng.choice(positions)
        rr0, rr1, cc0, cc1 = room

        if orient == "v":
            walls.append(("v", pos, rr0, rr1))
            children = [(rr0, rr1, cc0, pos), (rr0, rr1, pos, cc1)]
        else:
            walls.append(("h", pos, cc0, cc1))
            children = [(rr0, pos, cc0, cc1), (pos, rr1, cc0, cc1)]

        rooms.pop(idx)
        rooms.extend(children)

    areas = [(r1_ - r0_ - 1) * (c1_ - c0_ - 1) for r0_, r1_, c0_, c1_ in rooms]
    if min(areas) <= 0 or len(set(areas)) != room_count:
        return None

    wall_color = int(rng.choice([c for c in range(1, 10)]))
    grid = np.zeros((height, width), dtype=int)

    grid[r0, c0 : c1 + 1] = wall_color
    grid[r1, c0 : c1 + 1] = wall_color
    grid[r0 : r1 + 1, c0] = wall_color
    grid[r0 : r1 + 1, c1] = wall_color
    for orient, pos, start, end in walls:
        if orient == "v":
            grid[start : end + 1, pos] = wall_color
        else:
            grid[pos, start : end + 1] = wall_color

    # Open wall stubs add line irregularity without creating extra enclosed regions.
    for _ in range(_randint(rng, STUB_COUNT_RANGE)):
        orient = rng.choice(["v", "h"])
        stub_len = _randint(rng, STUB_LEN_RANGE)
        if orient == "v":
            col_choices = [c for c in range(width) if int(grid[1:, c].sum()) == 0]
            if not col_choices:
                continue
            c = rng.choice(col_choices)
            max_r = height - stub_len
            if max_r <= 1:
                continue
            r = rng.randint(1, max_r)
            if np.any(grid[r : r + stub_len, c] != 0):
                continue
            grid[r : r + stub_len, c] = wall_color
        else:
            row_choices = [r for r in range(1, height) if int(grid[r].sum()) == 0]
            if not row_choices:
                continue
            r = rng.choice(row_choices)
            max_c = width - stub_len
            if max_c <= 0:
                continue
            c = rng.randint(0, max_c)
            if np.any(grid[r, c : c + stub_len] != 0):
                continue
            grid[r, c : c + stub_len] = wall_color

    return grid, areas


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = _randint(rng, GRID_H_RANGE)
    width = _randint(rng, GRID_W_RANGE)
    room_count = _randint(rng, ROOM_COUNT_RANGE)

    palette_widths = _sample_distinct_widths(width, room_count, rng)
    if palette_widths is None:
        return None

    part = _generate_partition(height, width, room_count, rng)
    if part is None:
        return None
    grid, _areas = part

    wall_color = int(np.max(grid))
    palette_colors = [c for c in range(1, 10) if c != wall_color]
    if len(palette_colors) < room_count:
        return None
    palette_colors = rng.sample(palette_colors, room_count)

    col = 0
    for seg_width, color in zip(palette_widths, palette_colors, strict=False):
        grid[0, col : col + seg_width] = int(color)
        col += seg_width

    segments = _parse_palette_segments(grid)
    regions = _closed_black_regions(grid)
    if len(segments) != room_count or len(regions) != room_count:
        return None
    if len({int(seg["width"]) for seg in segments}) != room_count:
        return None
    if len({len(region) for region in regions}) != room_count:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
