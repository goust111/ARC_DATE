from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GENERATION_RETRIES = 400


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _path_endpoints(cells: list[Coord] | set[Coord]) -> list[Coord]:
    cell_set = set(cells)
    degrees = {
        cell: sum((neighbor in cell_set) for neighbor in _neighbors4(*cell))
        for cell in cell_set
    }
    assert all(degree in (1, 2) for degree in degrees.values()), "expected a non-branching path"
    endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
    assert len(endpoints) == 2, "expected exactly two endpoints"
    return endpoints


def _ordered_path(cells: list[Coord] | set[Coord], start: Coord | None = None) -> list[Coord]:
    cell_set = set(cells)
    endpoints = _path_endpoints(cell_set)
    cur = endpoints[0] if start is None else start
    assert cur in cell_set and cur in endpoints

    path: list[Coord] = []
    prev: Coord | None = None
    while True:
        path.append(cur)
        next_cells = [
            neighbor
            for neighbor in _neighbors4(*cur)
            if neighbor in cell_set and neighbor != prev
        ]
        if not next_cells:
            break
        assert len(next_cells) == 1, "path traversal must be unique"
        prev, cur = cur, next_cells[0]

    assert len(path) == len(cell_set), "path traversal must visit every cell"
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the source color sequence onto the marked placeholder path, repeating it cyclically.

    Input structure:
    - The background contains several 4-connected one-pixel-wide paths with no branches.
    - Exactly one marked source path is multicolored, and exactly one of its endpoints touches a gray pixel.
    - Exactly one marked target path is monochrome, one of its endpoints touches a gray pixel, and its length is an integer multiple of the source length.
    - Other monochrome paths without gray endpoint markers are distractors.

    Rule:
    - Read the colors of the source path starting from the endpoint adjacent to gray.
    - Traverse the target path starting from its gray-adjacent endpoint and overwrite its cells with the source sequence.
    - When the source sequence ends, restart it from the beginning until the whole target path is filled.

    Key insight:
    - The path geometry is only a carrier: gray marks the reading and writing endpoints, while the transferable information is the ordered color string along the source path.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    path_mask = (g != bg) & (g != GREY)
    components = _components(path_mask)

    marked_multicolor: list[tuple[list[Coord], Coord]] = []
    marked_monochrome: list[tuple[list[Coord], Coord]] = []

    for comp in components:
        endpoints = _path_endpoints(comp)
        marked = []
        for endpoint in endpoints:
            touches_grey = any(
                _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == GREY
                for nr, nc in _neighbors4(*endpoint)
            )
            if touches_grey:
                marked.append(endpoint)
        assert len(marked) <= 1, "a path may have at most one gray-marked endpoint"

        if not marked:
            continue

        colors = {int(g[r, c]) for r, c in comp}
        if len(colors) >= 2:
            marked_multicolor.append((comp, marked[0]))
        else:
            marked_monochrome.append((comp, marked[0]))

    assert len(marked_multicolor) == 1, "expected exactly one marked multicolor source path"
    source_cells, source_start = marked_multicolor[0]
    source_path = _ordered_path(source_cells, source_start)
    source_values = [int(g[r, c]) for r, c in source_path]
    assert len(set(source_values)) >= 2, "source path must be multicolor"

    target_candidates = [
        (cells, start)
        for cells, start in marked_monochrome
        if len(cells) % len(source_path) == 0
    ]
    assert len(target_candidates) == 1, "expected exactly one marked target path"
    target_cells, target_start = target_candidates[0]
    target_path = _ordered_path(target_cells, target_start)

    out = g.copy()
    for i, (r, c) in enumerate(target_path):
        out[r, c] = source_values[i % len(source_values)]
    return out


def _turn_count(path: list[Coord]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return turns


def _random_composition(total: int, parts: int) -> list[int]:
    assert parts >= 1 and total >= parts
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    out: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _turn_dirs(prev_dir: Coord | None) -> list[Coord]:
    if prev_dir is None:
        dirs = DIRS4[:]
    else:
        dr, dc = prev_dir
        dirs = [(-dc, dr), (dc, -dr)]
    random.shuffle(dirs)
    return dirs


def _extend_segment(
    path: list[Coord],
    used: set[Coord],
    direction: Coord,
    steps: int,
    shape: tuple[int, int],
) -> tuple[list[Coord], set[Coord]] | None:
    dr, dc = direction
    h, w = shape
    new_path = list(path)
    new_used = set(used)
    r, c = new_path[-1]

    for _ in range(steps):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return None
        if (nr, nc) in new_used:
            return None
        if any((ar, ac) in new_used and (ar, ac) != (r, c) for ar, ac in _neighbors4(nr, nc)):
            return None
        new_path.append((nr, nc))
        new_used.add((nr, nc))
        r, c = nr, nc
    return new_path, new_used


def _build_path(length: int, *, min_segments: int, max_segments: int) -> list[Coord]:
    steps = length - 1
    box_candidates = [
        (h, w)
        for h in range(3, 11)
        for w in range(3, 11)
        if h * w >= max(length, 2 * length - 2)
    ]

    def search(
        seg_i: int,
        prev_dir: Coord | None,
        path: list[Coord],
        used: set[Coord],
        run_lengths: list[int],
        shape: tuple[int, int],
    ) -> list[Coord] | None:
        if seg_i == len(run_lengths):
            return path
        for direction in _turn_dirs(prev_dir):
            extended = _extend_segment(path, used, direction, run_lengths[seg_i], shape)
            if extended is None:
                continue
            found = search(seg_i + 1, direction, extended[0], extended[1], run_lengths, shape)
            if found is not None:
                return found
        return None

    for _ in range(300):
        h, w = random.choice(box_candidates)
        n_segments = random.randint(min(min_segments, steps), min(max_segments, steps))
        run_lengths = _random_composition(steps, n_segments)
        starts = [(r, c) for r in range(h) for c in range(w)]
        random.shuffle(starts)
        for start in starts:
            found = search(0, None, [start], {start}, run_lengths, (h, w))
            if found is not None:
                return found
    raise RuntimeError("failed to build a valid path")


def _normalize_path(path: list[Coord]) -> list[Coord]:
    r0 = min(r for r, _ in path)
    c0 = min(c for _, c in path)
    return [(r - r0, c - c0) for r, c in path]


def _build_marked_patch(path: list[Coord], colors: list[int], mark_at_start: bool, bg: int) -> tuple[np.ndarray, list[Coord]]:
    norm = _normalize_path(path)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    shifted = [(r + 1, c + 1) for r, c in norm]
    patch = np.full((h + 2, w + 2), bg, dtype=int)

    ordered = shifted if mark_at_start else list(reversed(shifted))
    for (r, c), color in zip(ordered, colors):
        patch[r, c] = int(color)

    endpoint = ordered[0]
    inward = ordered[1]
    blocked_dir = (inward[0] - endpoint[0], inward[1] - endpoint[1])
    marker_dirs = [direction for direction in DIRS4 if direction != blocked_dir]
    dr, dc = random.choice(marker_dirs)
    patch[endpoint[0] + dr, endpoint[1] + dc] = GREY
    return patch, ordered


def _build_plain_patch(path: list[Coord], color: int, bg: int) -> np.ndarray:
    norm = _normalize_path(path)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    patch = np.full((h + 2, w + 2), bg, dtype=int)
    for r, c in norm:
        patch[r + 1, c + 1] = int(color)
    return patch


def _minimal_period(values: list[int]) -> int:
    n = len(values)
    for p in range(1, n + 1):
        if n % p != 0:
            continue
        if all(values[i] == values[i % p] for i in range(n)):
            return p
    return n


def _sample_source_sequence(length: int, forbidden: set[int]) -> list[int]:
    usable = [color for color in COLOR_POOL if color not in forbidden]
    min_distinct = min(3, length)
    distinct = random.randint(min_distinct, min(4, len(usable), length))
    for _ in range(200):
        palette = random.sample(usable, distinct)
        values = palette[:]
        while len(values) < length:
            values.append(random.choice(palette))
        random.shuffle(values)
        if len(set(values)) >= 2 and _minimal_period(values) == length:
            return values
    raise RuntimeError("failed to sample a full-period source sequence")


def _dilate8(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(radius):
        cur = out.copy()
        nxt = cur.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rs0 = max(0, -dr)
                rs1 = cur.shape[0] - max(0, dr)
                cs0 = max(0, -dc)
                cs1 = cur.shape[1] - max(0, dc)
                rd0 = max(0, dr)
                rd1 = rd0 + (rs1 - rs0)
                cd0 = max(0, dc)
                cd1 = cd0 + (cs1 - cs0)
                nxt[rd0:rd1, cd0:cd1] |= cur[rs0:rs1, cs0:cs1]
        out = nxt
    return out


def _place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    *,
    bg: int,
    padding: int = 1,
    border: int = 1,
) -> tuple[int, int]:
    h, w = patch.shape
    mask = patch != bg
    padded = _dilate8(occupied, padding)
    candidates: list[Coord] = []
    for r0 in range(border, grid.shape[0] - h - border + 1):
        for c0 in range(border, grid.shape[1] - w - border + 1):
            if not bool((padded[r0 : r0 + h, c0 : c0 + w] & mask).any()):
                candidates.append((r0, c0))
    if not candidates:
        raise ValueError("no free placement for patch")
    r0, c0 = random.choice(candidates)
    view = grid[r0 : r0 + h, c0 : c0 + w]
    view[mask] = patch[mask]
    occupied[r0 : r0 + h, c0 : c0 + w] |= mask
    return r0, c0


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    source_len = random.randint(4, 6)
    multiplier = random.randint(2, 4)
    target_len = source_len * multiplier
    target_color = random.choice(COLOR_POOL)
    source_values = _sample_source_sequence(source_len, {target_color})

    source_path = _build_path(source_len, min_segments=1, max_segments=4)
    target_path = _build_path(target_len, min_segments=2, max_segments=6)

    source_patch, source_local_order = _build_marked_patch(
        source_path, source_values, mark_at_start=bool(random.getrandbits(1)), bg=bg
    )
    target_patch, target_local_order = _build_marked_patch(
        target_path, [target_color] * target_len, mark_at_start=bool(random.getrandbits(1)), bg=bg
    )

    distractor_count = random.randint(1, 3)
    distractor_colors: list[int] = []
    if random.random() < 0.8:
        distractor_colors.append(target_color)
    while len(distractor_colors) < distractor_count:
        distractor_colors.append(random.choice(COLOR_POOL))

    distractor_specs: list[tuple[np.ndarray, int]] = []
    for color in distractor_colors:
        length = random.randint(3, min(9, target_len - 1))
        path = _build_path(length, min_segments=1, max_segments=4)
        distractor_specs.append((_build_plain_patch(path, color, bg), length))

    patches = [source_patch, target_patch] + [patch for patch, _length in distractor_specs]
    max_h = max(patch.shape[0] for patch in patches)
    max_w = max(patch.shape[1] for patch in patches)
    packed_area = sum((patch.shape[0] + 2) * (patch.shape[1] + 2) for patch in patches)
    side = max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(packed_area))) + 2)
    if side > 30:
        return None

    h = random.randint(side, 30)
    w = random.randint(side, 30)
    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    try:
        source_r0, source_c0 = _place_patch(grid, occupied, source_patch, bg=bg)
        target_r0, target_c0 = _place_patch(grid, occupied, target_patch, bg=bg)
        for patch, _length in distractor_specs:
            _place_patch(grid, occupied, patch, bg=bg)
    except ValueError:
        return None

    target_global_order = [(target_r0 + r, target_c0 + c) for r, c in target_local_order]
    expected = grid.copy()
    repeated = [source_values[i % len(source_values)] for i in range(len(target_global_order))]
    for (r, c), color in zip(target_global_order, repeated):
        expected[r, c] = color

    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    source_global_order = [(source_r0 + r, source_c0 + c) for r, c in source_local_order]
    grey_count = int(np.sum(grid == GREY))
    if grey_count != 2:
        return None
    if len(source_global_order) != source_len or len(target_global_order) != target_len:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid inputs for puzzle 5")
