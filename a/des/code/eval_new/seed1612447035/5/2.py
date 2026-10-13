from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MIN_SEGMENTS = 3
MAX_SEGMENTS = 6
MIN_LENGTH = 3
MAX_LENGTH = 6
MAX_OUTPUT_WIDTH = 26
GEN_ATTEMPTS = 400
PATH_ATTEMPTS = 200


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _monochrome_components(grid: np.ndarray, bg: int = BG) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize_segment(comp: list[tuple[int, int]], color: int) -> dict[str, object]:
    rows = sorted({r for r, _ in comp})
    cols = sorted({c for _, c in comp})
    assert len(comp) >= MIN_LENGTH
    assert len(rows) == 1 or len(cols) == 1, "each component must be horizontal or vertical"
    if len(rows) == 1:
        ordered = sorted(comp, key=lambda p: p[1])
        assert cols == list(range(cols[0], cols[-1] + 1)), "horizontal segment must be contiguous"
    else:
        ordered = sorted(comp, key=lambda p: p[0])
        assert rows == list(range(rows[0], rows[-1] + 1)), "vertical segment must be contiguous"
    assert MIN_LENGTH <= len(ordered) <= MAX_LENGTH
    return {
        "color": int(color),
        "coords": tuple(ordered),
        "endpoints": (ordered[0], ordered[-1]),
        "length": int(len(ordered)),
    }


def _extract_segments(grid: np.ndarray) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    for comp in _monochrome_components(grid, bg=BG):
        color = int(grid[comp[0][0], comp[0][1]])
        segments.append(_normalize_segment(comp, color))
    assert segments, "input must contain at least one segment"
    return segments


def _chain_order(segments: list[dict[str, object]]) -> list[int]:
    coord_owner: dict[tuple[int, int], int] = {}
    endpoint_owner: dict[tuple[int, int], tuple[int, int]] = {}
    for idx, seg in enumerate(segments):
        for coord in seg["coords"]:
            coord_owner[coord] = idx
        a, b = seg["endpoints"]
        endpoint_owner[a] = (idx, 0)
        endpoint_owner[b] = (idx, 1)

    endpoint_links: dict[tuple[int, int], tuple[int, int]] = {}
    seg_neighbors = [set() for _ in segments]

    for coord, seg_idx in coord_owner.items():
        for nbr in _neighbors4(*coord):
            other_idx = coord_owner.get(nbr)
            if other_idx is None or other_idx == seg_idx or coord > nbr:
                continue
            assert coord in endpoint_owner and nbr in endpoint_owner, "only segment endpoints may touch"
            a = endpoint_owner[coord]
            b = endpoint_owner[nbr]
            assert a not in endpoint_links and b not in endpoint_links, "each endpoint may touch at most one endpoint"
            endpoint_links[a] = b
            endpoint_links[b] = a
            seg_neighbors[a[0]].add(b[0])
            seg_neighbors[b[0]].add(a[0])

    free_endpoints: list[tuple[int, int, tuple[int, int]]] = []
    for idx, seg in enumerate(segments):
        degree = len(seg_neighbors[idx])
        if len(segments) == 1:
            assert degree == 0
        else:
            assert degree in (1, 2), "segments must form a simple chain"
        for ep_idx, coord in enumerate(seg["endpoints"]):
            if (idx, ep_idx) not in endpoint_links:
                free_endpoints.append((idx, ep_idx, coord))
    assert len(free_endpoints) == 2, "a chain has exactly two free endpoints"

    seen = set()
    stack = [0]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(seg_neighbors[cur] - seen)
    assert len(seen) == len(segments), "segment graph must be connected"

    start_seg, start_ep, _ = min(free_endpoints, key=lambda item: (item[2][0], item[2][1]))
    order: list[int] = []
    cur_seg = start_seg
    cur_ep = start_ep
    visited = set()
    while True:
        assert cur_seg not in visited, "chain traversal must not revisit a segment"
        visited.add(cur_seg)
        order.append(cur_seg)
        exit_ep = 1 - cur_ep
        nxt = endpoint_links.get((cur_seg, exit_ep))
        if nxt is None:
            break
        cur_seg, cur_ep = nxt
    assert len(order) == len(segments)
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flatten a hidden chain of colored straight segments into one left-to-right color band.

    Input structure:
    - The background is black.
    - Several short monochrome line segments are scattered on the grid; each segment is horizontal or vertical and has a distinct color.
    - Segment endpoints touch other segment endpoints by 4-neighbor adjacency, forming one non-branching chain.
    - Exactly two segment endpoints are free: the chain's two outer ends.

    Rule:
    - Separate the monochrome line segments and identify their endpoints.
    - Reconstruct the chain order from endpoint-to-endpoint 4-neighbor adjacencies, choosing the start whose free endpoint is higher; if tied, further left.
    - Output a single-row grid whose cells are the chain's segment colors laid out in traversal order, with each segment repeated once per pixel in that segment.

    Key insight:
    - The important information is not the 2D placement of each segment by itself, but the hidden sequence encoded by which endpoints touch.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    segments = _extract_segments(g)
    order = _chain_order(segments)
    band: list[int] = []
    for idx in order:
        seg = segments[idx]
        band.extend([int(seg["color"])] * int(seg["length"]))
    out = np.asarray([band], dtype=int)
    assert out.ndim == 2 and out.shape[0] == 1 and out.shape[1] <= 30
    return out


def _sample_lengths(count: int) -> list[int]:
    total_min = count * MIN_LENGTH
    total_max = min(MAX_OUTPUT_WIDTH, count * MAX_LENGTH)
    total = random.randint(total_min, total_max)
    lengths: list[int] = []
    remaining = total
    for idx in range(count):
        left = count - idx - 1
        lo = max(MIN_LENGTH, remaining - left * MAX_LENGTH)
        hi = min(MAX_LENGTH, remaining - left * MIN_LENGTH)
        lengths.append(random.randint(lo, hi))
        remaining -= lengths[-1]
    return lengths


def _segment_fits(
    occupied: set[tuple[int, int]],
    cells: list[tuple[int, int]],
    predecessor: tuple[int, int] | None,
) -> bool:
    new_set = set(cells)
    if len(new_set) != len(cells) or occupied & new_set:
        return False
    for idx, cell in enumerate(cells):
        existing_neighbors = {nbr for nbr in _neighbors4(*cell) if nbr in occupied}
        allowed_existing = set()
        if idx == 0:
            if predecessor is not None:
                allowed_existing.add(predecessor)
        if existing_neighbors != allowed_existing:
            return False

        new_neighbors = {nbr for nbr in _neighbors4(*cell) if nbr in new_set}
        allowed_new = set()
        if idx > 0:
            allowed_new.add(cells[idx - 1])
        if idx + 1 < len(cells):
            allowed_new.add(cells[idx + 1])
        if new_neighbors != allowed_new:
            return False
    return True


def _build_path(lengths: list[int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    def dfs(
        idx: int,
        path: list[tuple[int, int]],
        occupied: set[tuple[int, int]],
        dirs: list[tuple[int, int]],
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
        if idx == len(lengths):
            orientations = {("H" if dc else "V") for _dr, dc in dirs}
            if len(orientations) < 2:
                return None
            return path, dirs

        options = DIRS4[:]
        random.shuffle(options)
        if idx > 0:
            prev_dr, prev_dc = dirs[-1]
            options = [d for d in options if d != (-prev_dr, -prev_dc)]

        anchor = path[-1] if path else (0, 0)
        for dr, dc in options:
            if idx == 0:
                cells = [(anchor[0] + step * dr, anchor[1] + step * dc) for step in range(lengths[idx])]
                predecessor = None
            else:
                cells = [(anchor[0] + step * dr, anchor[1] + step * dc) for step in range(1, lengths[idx] + 1)]
                predecessor = anchor
            if not _segment_fits(occupied, cells, predecessor):
                continue
            next_path = cells if idx == 0 else path + cells
            result = dfs(idx + 1, next_path, occupied | set(cells), dirs + [(dr, dc)])
            if result is not None:
                return result
        return None

    for _ in range(PATH_ATTEMPTS):
        built = dfs(0, [], set(), [])
        if built is not None:
            return built
    return None


def _apply_symmetry(coords: list[tuple[int, int]], variant: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r, c in coords:
        if variant == 0:
            nr, nc = r, c
        elif variant == 1:
            nr, nc = r, -c
        elif variant == 2:
            nr, nc = -r, c
        elif variant == 3:
            nr, nc = -r, -c
        elif variant == 4:
            nr, nc = c, r
        elif variant == 5:
            nr, nc = c, -r
        elif variant == 6:
            nr, nc = -c, r
        else:
            nr, nc = -c, -r
        out.append((int(nr), int(nc)))
    return out


def _compose_grid(
    path: list[tuple[int, int]],
    lengths: list[int],
    colors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    variant = random.randrange(8)
    path = _apply_symmetry(path, variant)

    rows = [r for r, _ in path]
    cols = [c for _, c in path]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    box_h = max_r - min_r + 1
    box_w = max_c - min_c + 1

    grid_h = random.randint(box_h + 2, min(30, box_h + 8))
    grid_w = random.randint(box_w + 2, min(30, box_w + 8))
    top = random.randint(1, grid_h - box_h - 1)
    left = random.randint(1, grid_w - box_w - 1)
    dr = top - min_r
    dc = left - min_c
    path = [(r + dr, c + dc) for r, c in path]

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    pos = 0
    for color, length in zip(colors, lengths):
        for _ in range(length):
            r, c = path[pos]
            grid[r, c] = int(color)
            pos += 1

    forward_band: list[int] = []
    for color, length in zip(colors, lengths):
        forward_band.extend([int(color)] * int(length))
    reverse_band: list[int] = []
    for color, length in zip(reversed(colors), reversed(lengths)):
        reverse_band.extend([int(color)] * int(length))
    expected = np.asarray([forward_band if path[0] <= path[-1] else reverse_band], dtype=int)
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(MIN_SEGMENTS, MAX_SEGMENTS)
    lengths = _sample_lengths(count)
    built = _build_path(lengths)
    if built is None:
        return None
    path, _dirs = built
    colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], count)
    grid, expected = _compose_grid(path, lengths, colors)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
