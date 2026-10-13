from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
PATH_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)  # main-area width; legend adds one more column
FRAGMENT_COUNT_OPEN = (3, 5)
FRAGMENT_COUNT_CLOSED = (4, 6)
OPEN_TOTAL_RANGE = (10, 18)  # unique cells in the final assembled chain
SCATTER_DISTANCE_RANGE = (2, 7)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _extract_fragment(area: np.ndarray, color: int) -> dict[str, object]:
    pts = [tuple(map(int, p)) for p in np.argwhere(area == int(color))]
    assert pts, f"missing fragment for color {color}"
    mask = area == int(color)
    start = pts[0]
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < area.shape[0] and 0 <= nc < area.shape[1] and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((int(nr), int(nc)))
                q.append((int(nr), int(nc)))
    assert len(seen) == len(pts), f"color {color} must form exactly one 4-connected fragment"

    coord_set = set(seen)
    endpoints = []
    for cell in coord_set:
        degree = sum((nbr in coord_set) for nbr in _neighbors4(*cell))
        assert degree <= 2, f"fragment color {color} must be a non-branching path"
        if degree == 1:
            endpoints.append(cell)
    assert len(endpoints) == 2, f"fragment color {color} must have exactly two endpoints"

    coords = np.array(sorted(coord_set), dtype=int)
    endpoints = sorted(endpoints)
    return {
        "color": int(color),
        "coords": coords,
        "coord_set": coord_set,
        "endpoints": endpoints,
    }


def _translate_fragment(
    fragment: dict[str, object],
    head_idx: int,
    anchor: tuple[int, int],
) -> dict[str, object]:
    head = tuple(fragment["endpoints"][head_idx])
    tail_src = tuple(fragment["endpoints"][1 - head_idx])
    dr = int(anchor[0] - head[0])
    dc = int(anchor[1] - head[1])
    delta = np.array([dr, dc], dtype=int)
    coords = np.asarray(fragment["coords"], dtype=int) + delta
    coord_list = [tuple(map(int, p)) for p in coords]
    return {
        "coords": coords,
        "coord_set": set(coord_list),
        "head": tuple(map(int, anchor)),
        "tail": (int(tail_src[0] + dr), int(tail_src[1] + dc)),
        "tail_source": tail_src,
        "head_source": head,
    }


def _candidate_rank(extra_overlap: int, dist: int, tie: tuple[int, ...]) -> tuple[int, ...]:
    if extra_overlap == 0:
        return (0, int(dist), *tie)
    return (1, int(extra_overlap), int(dist), *tie)


def _enclosed_regions(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if int(grid[r, c]) == BLACK and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((int(r), int(c)))
    for c in range(w):
        for r in (0, h - 1):
            if int(grid[r, c]) == BLACK and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((int(r), int(c)))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == BLACK and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                q.append((int(nr), int(nc)))

    regions: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BLACK or bool(seen[r, c]):
                continue
            region = []
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                region.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == BLACK and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            regions.append(region)
    return regions


def _fill_enclosed_regions(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for region in _enclosed_regions(out):
        for r, c in region:
            out[r, c] = GREY
    return out


def _assemble_main_area(main_area: np.ndarray, order: list[int]) -> np.ndarray:
    h, w = main_area.shape
    fragments = [_extract_fragment(main_area, color) for color in order]
    out = np.full((h, w), BLACK, dtype=int)

    first = fragments[0]
    occupied = set(first["coord_set"])
    for r, c in first["coord_set"]:
        out[r, c] = int(first["color"])

    if len(fragments) == 1:
        return _fill_enclosed_regions(out)

    second = fragments[1]
    best_pair = None
    for tail_idx, first_tail in enumerate(first["endpoints"]):
        for head_idx in range(2):
            placed = _translate_fragment(second, head_idx, tuple(first_tail))
            extra = len((placed["coord_set"] & occupied) - {tuple(first_tail)})
            dist = _manhattan(tuple(first_tail), tuple(second["endpoints"][head_idx]))
            rank = _candidate_rank(extra, dist, (int(tail_idx), int(head_idx)))
            candidate = (rank, int(tail_idx), int(head_idx), placed)
            if best_pair is None or candidate[0] < best_pair[0]:
                best_pair = candidate
    assert best_pair is not None

    _, _tail_idx, head_idx, placed = best_pair
    for r, c in placed["coord_set"]:
        out[r, c] = int(second["color"])
    occupied |= placed["coord_set"]
    current_tail_pos = tuple(placed["tail"])
    current_tail_source = tuple(placed["tail_source"])

    for fragment in fragments[2:]:
        best = None
        for head_idx in range(2):
            placed = _translate_fragment(fragment, head_idx, current_tail_pos)
            extra = len((placed["coord_set"] & occupied) - {current_tail_pos})
            dist = _manhattan(current_tail_source, tuple(fragment["endpoints"][head_idx]))
            rank = _candidate_rank(extra, dist, (int(head_idx),))
            candidate = (rank, int(head_idx), placed)
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None

        _, head_idx, placed = best
        for r, c in placed["coord_set"]:
            out[r, c] = int(fragment["color"])
        occupied |= placed["coord_set"]
        current_tail_pos = tuple(placed["tail"])
        current_tail_source = tuple(placed["tail_source"])

    return _fill_enclosed_regions(out)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble colored path fragments by legend order, then fill any enclosed black regions.

    Input structure:
    - The background is black.
    - Column 0 is a 1-pixel-wide legend strip whose non-black cells list distinct fragment colors from top to bottom.
    - The remaining columns form the main area, containing one 4-connected monochrome rectilinear path fragment for each legend color.
    - Every fragment is 1 pixel thick and has exactly two endpoints.

    Rule:
    - Read the legend colors from top to bottom to get the fragment concatenation order.
    - Keep the first fragment fixed in place.
    - For each later fragment, translate it so one of its endpoints lands on the current tail of the assembled path; prefer the endpoint choice that avoids extra overlap with the assembled path, and otherwise use the closer endpoint in the input layout.
    - At the shared endpoint, the later fragment's color overwrites the earlier one.
    - Remove the legend strip and fill every black region that becomes fully enclosed by the assembled path with grey.

    Key insight:
    - The left strip is an ordering instruction: once fragment colors are sequenced, endpoint alignment and enclosed-region flood fill determine the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 2

    legend_colors = [int(g[r, 0]) for r in range(g.shape[0]) if int(g[r, 0]) != BLACK]
    assert legend_colors, "legend strip must contain at least one color"
    assert len(set(legend_colors)) == len(legend_colors), "legend colors must be distinct"

    main_area = g[:, 1:]
    return _assemble_main_area(main_area, legend_colors)


def _sample_fragment_lengths(k: int, total_unique: int) -> list[int] | None:
    target = int(total_unique + k - 1)
    if not (3 * k <= target <= 8 * k):
        return None
    lengths: list[int] = []
    remaining = target
    for i in range(k):
        left = k - i - 1
        lo = max(3, remaining - 8 * left)
        hi = min(8, remaining - 3 * left)
        if lo > hi:
            return None
        value = random.randint(lo, hi)
        lengths.append(int(value))
        remaining -= int(value)
    if remaining != 0:
        return None
    return lengths


def _partition_path(path: list[tuple[int, int]], lengths: list[int]) -> list[list[tuple[int, int]]]:
    fragments: list[list[tuple[int, int]]] = []
    start = 0
    for length in lengths:
        end = start + int(length)
        fragments.append(path[start:end])
        start = end - 1
    assert start == len(path) - 1
    return fragments


def _generate_open_path(total_unique: int, coord_limit: int) -> list[tuple[int, int]] | None:
    for _ in range(300):
        path = [(0, 0)]
        occupied = {(0, 0)}
        prev_dir: tuple[int, int] | None = None
        while len(path) < total_unique:
            r, c = path[-1]
            candidates = []
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if abs(nr) > coord_limit or abs(nc) > coord_limit:
                    continue
                if (nr, nc) in occupied:
                    continue
                if any((nbr in occupied) and (nbr != (r, c)) for nbr in _neighbors4(nr, nc)):
                    continue
                candidates.append((int(nr), int(nc)))
            if not candidates:
                break
            if prev_dir is not None:
                straight = (r + prev_dir[0], c + prev_dir[1])
                if straight in candidates and random.random() < 0.45:
                    nxt = straight
                else:
                    nxt = random.choice(candidates)
            else:
                nxt = random.choice(candidates)
            prev_dir = (int(nxt[0] - r), int(nxt[1] - c))
            path.append(nxt)
            occupied.add(nxt)
        if len(path) == total_unique:
            return path
    return None


def _rectangle_ring_path(height: int, width: int) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for r in range(height):
        path.append((int(r), 0))
    for c in range(1, width):
        path.append((height - 1, int(c)))
    for r in range(height - 2, -1, -1):
        path.append((int(r), width - 1))
    for c in range(width - 2, 0, -1):
        path.append((0, int(c)))
    return path


def _generate_closed_path(k: int) -> tuple[list[tuple[int, int]], list[int]] | None:
    for _ in range(200):
        ring_h = random.randint(4, 8)
        ring_w = random.randint(4, 8)
        path = _rectangle_ring_path(ring_h, ring_w)
        total_unique = len(path)
        lengths = _sample_fragment_lengths(k, total_unique)
        if lengths is None:
            continue
        if random.random() < 0.5:
            path = list(reversed(path))
        shift = random.randrange(len(path))
        path = path[shift:] + path[:shift]
        return path, lengths
    return None


def _shift_path_into_grid(
    path: list[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    rs = [r for r, _ in path]
    cs = [c for _, c in path]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    bbox_h = rmax - rmin + 1
    bbox_w = cmax - cmin + 1
    if bbox_h > h - 4 or bbox_w > w - 4:
        return None

    top_lo = 2
    top_hi = h - bbox_h - 2
    left_lo = 2
    left_hi = w - bbox_w - 2
    if top_lo > top_hi or left_lo > left_hi:
        return None

    top = random.randint(top_lo, top_hi)
    left = random.randint(left_lo, left_hi)
    dr = top - rmin
    dc = left - cmin
    return [(int(r + dr), int(c + dc)) for r, c in path]


def _paint_fragments(
    h: int,
    w: int,
    fragments: list[list[tuple[int, int]]],
    colors: list[int],
) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for color, fragment in zip(colors, fragments):
        for r, c in fragment:
            grid[r, c] = int(color)
    return grid


def _touches_occupied(coords: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    coord_set = set(coords)
    for cell in coords:
        if cell in occupied:
            return True
        for nbr in _neighbors4(*cell):
            if nbr in occupied and nbr not in coord_set:
                return True
    return False


def _head_targets(anchor: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    lo, hi = SCATTER_DISTANCE_RANGE
    targets = []
    for r in range(h):
        for c in range(w):
            d = _manhattan(anchor, (int(r), int(c)))
            if lo <= d <= hi:
                targets.append((int(r), int(c)))
    random.shuffle(targets)
    return targets


def _scatter_fragments(
    h: int,
    w: int,
    assembled: list[list[tuple[int, int]]],
) -> list[list[tuple[int, int]]] | None:
    scattered = [list(assembled[0])]
    occupied = set(assembled[0])
    prev_tail = tuple(assembled[0][-1])
    first_head = tuple(assembled[0][0])
    first_tail = tuple(assembled[0][-1])

    for idx, fragment in enumerate(assembled[1:], start=1):
        head_src = tuple(fragment[0])
        tail_src = tuple(fragment[-1])
        placed_fragment = None
        for target_head in _head_targets(prev_tail, h, w):
            dr = int(target_head[0] - head_src[0])
            dc = int(target_head[1] - head_src[1])
            if dr == 0 and dc == 0:
                continue
            moved = [(int(r + dr), int(c + dc)) for r, c in fragment]
            if any(not (0 <= r < h and 0 <= c < w) for r, c in moved):
                continue
            if _touches_occupied(moved, occupied):
                continue

            d_head = _manhattan(prev_tail, target_head)
            moved_tail = tuple(moved[-1])
            d_tail = _manhattan(prev_tail, moved_tail)
            if d_head + 1 > d_tail:
                continue

            if idx == 1:
                good = _manhattan(first_tail, target_head)
                alternatives = (
                    _manhattan(first_head, target_head),
                    _manhattan(first_tail, moved_tail),
                    _manhattan(first_head, moved_tail),
                )
                if good + 1 > min(alternatives):
                    continue

            placed_fragment = moved
            break
        if placed_fragment is None:
            return None
        scattered.append(placed_fragment)
        occupied |= set(placed_fragment)
        prev_tail = tuple(placed_fragment[-1])

    return scattered


def _build_input_grid(
    h: int,
    w: int,
    fragments: list[list[tuple[int, int]]],
    colors: list[int],
) -> np.ndarray:
    main_area = np.full((h, w), BLACK, dtype=int)
    for color, fragment in zip(colors, fragments):
        for r, c in fragment:
            main_area[r, c] = int(color)

    grid = np.full((h, w + 1), BLACK, dtype=int)
    grid[:, 1:] = main_area
    legend_top = random.randint(0, h - len(colors))
    for i, color in enumerate(colors):
        grid[legend_top + i, 0] = int(color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    closed_branch = random.random() < 0.5
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    if w + 1 > 30:
        return None

    if closed_branch:
        k = random.randint(*FRAGMENT_COUNT_CLOSED)
        result = _generate_closed_path(k)
        if result is None:
            return None
        path, lengths = result
    else:
        k = random.randint(*FRAGMENT_COUNT_OPEN)
        total_lo = max(OPEN_TOTAL_RANGE[0], 2 * k + 1)
        total_hi = min(OPEN_TOTAL_RANGE[1], 7 * k + 1)
        if total_lo > total_hi:
            return None
        total_unique = random.randint(total_lo, total_hi)
        lengths = _sample_fragment_lengths(k, total_unique)
        if lengths is None:
            return None
        coord_limit = min((h - 5) // 2, (w - 5) // 2, 8)
        if coord_limit < 4:
            return None
        path = _generate_open_path(total_unique, coord_limit)
        if path is None:
            return None

    shifted = _shift_path_into_grid(path, h, w)
    if shifted is None:
        return None

    fragments = _partition_path(shifted, lengths)
    colors = random.sample(PATH_COLORS, k)

    target_main = _paint_fragments(h, w, fragments, colors)
    target = _fill_enclosed_regions(target_main)
    enclosed = _enclosed_regions(target_main)
    if closed_branch and not enclosed:
        return None
    if (not closed_branch) and enclosed:
        return None

    scattered = _scatter_fragments(h, w, fragments)
    if scattered is None:
        return None

    inp = _build_input_grid(h, w, scattered, colors)
    out = main(inp)
    if not np.array_equal(out, target):
        return None
    if closed_branch != bool(np.any(out == GREY)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
