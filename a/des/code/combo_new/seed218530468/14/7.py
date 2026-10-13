import random
from collections import Counter

import numpy as np


_RNG = random.Random()
_COLORS = tuple(range(10))
_GEN_RETRIES = 400
_MIN_ROOM_SIDE = 2
_MAX_PALETTE = 6


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _is_palette_row(row: np.ndarray, grey: int) -> bool:
    vals = {int(v) for v in row if int(v) != grey}
    return len(vals) > 1


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    h = grid.shape[0]
    for r in range(1, h - 1):
        val = int(grid[r, 0])
        if not np.all(grid[r] == val):
            continue
        if not np.any(grid[r + 1 :] == val):
            continue
        if any(_is_palette_row(grid[rr], val) for rr in range(r)):
            return r, val
    raise AssertionError("separator row not found")


def _read_palette(grid: np.ndarray, sep: int, grey: int) -> tuple[int, list[int]]:
    palette_row = None
    for r in range(sep - 1, -1, -1):
        if _is_palette_row(grid[r], grey):
            palette_row = grid[r]
            break
    assert palette_row is not None, "palette row not found"

    non_grey = [int(v) for v in palette_row if int(v) != grey]
    bg = Counter(non_grey).most_common(1)[0][0]

    palette = [int(v) for v in palette_row if int(v) not in (bg, grey)]
    assert palette, "palette must be non-empty"
    assert len(set(palette)) == len(palette), "palette colors must be unique"
    return bg, palette


def _region_data(grid: np.ndarray, sep: int, bg: int) -> tuple[list[list[tuple[int, int]]], np.ndarray]:
    h, w = grid.shape
    lower_mask = grid[sep + 1 :] == bg
    comps = _components(lower_mask)

    labels = np.full((h, w), -1, dtype=int)
    regions: list[list[tuple[int, int]]] = []
    for comp in comps:
        abs_comp = [(r + sep + 1, c) for r, c in comp]
        if any(r == h - 1 or c == 0 or c == w - 1 for r, c in abs_comp):
            continue
        idx = len(regions)
        regions.append(abs_comp)
        for r, c in abs_comp:
            labels[r, c] = idx
    return regions, labels


def _adjacency_sets(grid: np.ndarray, sep: int, grey: int, labels: np.ndarray, n_regions: int) -> list[set[int]]:
    h, w = grid.shape
    adj = [set() for _ in range(n_regions)]
    for r in range(sep + 1, h):
        for c in range(w):
            if int(grid[r, c]) != grey:
                continue
            up = labels[r - 1, c] if r - 1 >= sep + 1 else -1
            down = labels[r + 1, c] if r + 1 < h else -1
            left = labels[r, c - 1] if c - 1 >= 0 else -1
            right = labels[r, c + 1] if c + 1 < w else -1
            if up >= 0 and down >= 0 and up != down:
                adj[up].add(down)
                adj[down].add(up)
            if left >= 0 and right >= 0 and left != right:
                adj[left].add(right)
                adj[right].add(left)
    return adj


def _analyze(grid: np.ndarray) -> tuple[int, int, int, list[int], list[list[tuple[int, int]]], list[set[int]]]:
    sep, grey = _find_separator(grid)
    bg, palette = _read_palette(grid, sep, grey)
    regions, labels = _region_data(grid, sep, bg)
    assert regions, "expected enclosed regions"
    adj = _adjacency_sets(grid, sep, grey, labels, len(regions))
    return sep, grey, bg, palette, regions, adj


def main(grid: np.ndarray) -> np.ndarray:
    """
    Refill each enclosed region using the palette color indexed by its number of neighboring regions.

    Input structure:
    - The grid is split by a full horizontal grey row.
    - Above the separator, one palette row places isolated colored pixels from left to right on the background.
    - Below the separator, 1-pixel grey walls partition the background into closed 4-connected regions.

    Rule:
    - Identify every enclosed background region below the grey separator.
    - Count how many other regions share a grey wall with that region, using opposite sides of the same wall pixel as the adjacency test.
    - Use the K-th palette color from left to right to fill a region with K neighbors; if K is outside the palette, leave that region as background.
    - Keep the palette row and all grey walls unchanged.

    Key insight:
    - The fill color is determined by the region-adjacency graph, while the palette row is only an ordered decoder from neighbor count to color.
    """
    g = np.asarray(grid, dtype=int)
    sep, grey, bg, palette, regions, adj = _analyze(g)

    out = g.copy()
    for idx, cells in enumerate(regions):
        degree = len(adj[idx])
        if 1 <= degree <= len(palette):
            color = palette[degree - 1]
            for r, c in cells:
                out[r, c] = color
        else:
            for r, c in cells:
                out[r, c] = bg
    return out


def _splittable(room: tuple[int, int, int, int]) -> tuple[bool, bool]:
    r0, r1, c0, c1 = room
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    can_h = height >= 2 * _MIN_ROOM_SIDE + 1
    can_v = width >= 2 * _MIN_ROOM_SIDE + 1
    return can_h, can_v


def _split_room(
    room: tuple[int, int, int, int], lower: np.ndarray, grey: int, rng: random.Random
) -> list[tuple[int, int, int, int]] | None:
    r0, r1, c0, c1 = room
    can_h, can_v = _splittable(room)
    choices = []
    if can_h:
        choices.append("h")
    if can_v:
        choices.append("v")
    if not choices:
        return None

    orient = rng.choice(choices)
    if orient == "h":
        cut = rng.randint(r0 + _MIN_ROOM_SIDE, r1 - _MIN_ROOM_SIDE)
        lower[cut, c0 : c1 + 1] = grey
        return [(r0, cut - 1, c0, c1), (cut + 1, r1, c0, c1)]

    cut = rng.randint(c0 + _MIN_ROOM_SIDE, c1 - _MIN_ROOM_SIDE)
    lower[r0 : r1 + 1, cut] = grey
    return [(r0, r1, c0, cut - 1), (r0, r1, cut + 1, c1)]


def _sample_palette_positions(width: int, count: int, rng: random.Random) -> list[int] | None:
    if count > (width + 1) // 2:
        return None
    for _ in range(200):
        cols = sorted(rng.sample(range(width), count))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _build_partition(width: int, room_h: int, grey: int, bg: int, rng: random.Random):
    lower = np.full((room_h + 1, width), bg, dtype=int)
    lower[:, 0] = grey
    lower[:, -1] = grey
    lower[-1, :] = grey

    rooms = [(0, room_h - 1, 1, width - 2)]
    max_rooms = min(8, max(3, 2 + (room_h * (width - 2)) // 18))
    target_rooms = rng.randint(3, max_rooms)

    while len(rooms) < target_rooms:
        candidates = [i for i, room in enumerate(rooms) if any(_splittable(room))]
        if not candidates:
            break
        idx = rng.choice(candidates)
        room = rooms.pop(idx)
        pieces = _split_room(room, lower, grey, rng)
        if pieces is None:
            rooms.append(room)
            break
        rooms.extend(pieces)

    if len(rooms) < 3:
        return None
    return lower


def _choose_palette_len(
    degrees: list[int], width: int, rng: random.Random
) -> int | None:
    max_palette = min(_MAX_PALETTE, len(_COLORS) - 2, (width + 1) // 2)
    max_degree = max(degrees)
    if max_degree < 2 or max_palette < 2:
        return None

    short_branch = rng.random() < 0.5
    if short_branch:
        upper = min(max_palette, max_degree - 1)
        if upper < 1:
            return None
        choices = [
            k
            for k in range(1, upper + 1)
            if any(d <= k for d in degrees) and any(d > k for d in degrees)
        ]
        if not choices:
            return None
        return rng.choice(choices)

    if max_degree > max_palette:
        return None
    return rng.randint(max_degree, max_palette)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    width = rng.randint(11, 28)
    room_h = rng.randint(6, 24)
    if room_h + 3 > 30:
        return None

    bg, grey = rng.sample(_COLORS, 2)
    lower = _build_partition(width, room_h, grey, bg, rng)
    if lower is None:
        return None

    grid = np.full((room_h + 3, width), bg, dtype=int)
    grid[1, :] = grey
    grid[2:, :] = lower

    regions, labels = _region_data(grid, 1, bg)
    if len(regions) < 3:
        return None
    adj = _adjacency_sets(grid, 1, grey, labels, len(regions))
    degrees = [len(s) for s in adj]
    if min(degrees) < 1 or len(set(degrees)) < 2:
        return None

    pal_len = _choose_palette_len(degrees, width, rng)
    if pal_len is None:
        return None

    palette_colors = rng.sample([c for c in _COLORS if c not in (bg, grey)], pal_len)
    positions = _sample_palette_positions(width, pal_len, rng)
    if positions is None:
        return None

    for c, color in zip(positions, palette_colors):
        grid[0, c] = color

    out = main(grid)
    if out.shape != grid.shape or not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
