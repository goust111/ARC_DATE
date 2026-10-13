from __future__ import annotations

import random
from collections import deque
from functools import lru_cache

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)
SPECIAL_COLORS = {BLACK, GREY, MAROON}
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


INTERIOR_OPTIONS = {
    2: [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2)],
    3: [(2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (3, 4), (4, 3)],
    4: [(2, 4), (4, 2), (3, 3), (2, 5), (5, 2), (3, 4), (4, 3), (4, 4)],
}

CONTAINER_COUNT_RANGE = (3, 5)
STACK_GAP_RANGE = (1, 3)
STACK_SPACING_RANGE = (1, 3)
SIDE_MARGIN_RANGE = (1, 3)
TOP_MARGIN_RANGE = (1, 3)
PRELEGEND_GAP_RANGE = (2, 4)
LEGEND_SPACING_RANGE = (1, 3)
PARTITION_RETRIES = 120
GEN_RETRIES = 300


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _monochrome_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _frame_info(pts: list[tuple[int, int]]) -> dict[str, int] | None:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    if set(pts) != border:
        return None
    return {"r0": r0, "c0": c0, "r1": r1, "c1": c1, "h": h, "w": w}


def _find_frames(grid: np.ndarray) -> list[dict[str, int]]:
    frames: list[dict[str, int]] = []
    for pts in _components_mask(grid == GREY):
        info = _frame_info(pts)
        if info is not None:
            frames.append(info)
    return frames


def _extract_legend(grid: np.ndarray) -> dict[int, int]:
    h, w = grid.shape
    mapping: dict[int, int] = {}
    for r, c in map(tuple, np.argwhere(grid == MAROON)):
        r = int(r)
        c = int(c)
        if r == 0:
            continue
        for start in range(c - 2, c + 1):
            if start < 0 or start + 2 >= w:
                continue
            row = grid[r - 1, start : start + 3]
            color = int(row[0])
            if color in SPECIAL_COLORS or not np.all(row == color):
                continue
            below = grid[r, start : start + 3]
            if int(np.count_nonzero(below == MAROON)) != 1:
                continue
            offset = c - start
            if int(below[offset]) != MAROON:
                continue
            if offset in mapping and mapping[offset] != color:
                raise AssertionError("legend offset is ambiguous")
            mapping[offset] = color
            break
    return mapping


def _pair_frames(grid: np.ndarray, frames: list[dict[str, int]]) -> list[tuple[dict[str, int], dict[str, int]]]:
    containers: list[dict[str, int]] = []
    projections: list[dict[str, int]] = []
    for frame in frames:
        inner = grid[frame["r0"] + 1 : frame["r1"], frame["c0"] + 1 : frame["c1"]]
        if np.any(inner != BLACK):
            containers.append(frame)
        else:
            projections.append(frame)

    pairs: list[tuple[dict[str, int], dict[str, int]]] = []
    used: set[int] = set()
    containers.sort(key=lambda item: (item["c0"], item["r0"]))
    projections.sort(key=lambda item: (item["c0"], item["r0"]))
    for container in containers:
        candidates = [
            (idx, frame)
            for idx, frame in enumerate(projections)
            if idx not in used
            and frame["c0"] == container["c0"]
            and frame["c1"] == container["c1"]
            and frame["h"] == container["h"]
            and frame["w"] == container["w"]
            and frame["r1"] < container["r0"]
        ]
        if not candidates:
            continue
        idx, projection = max(candidates, key=lambda item: item[1]["r1"])
        used.add(idx)
        pairs.append((container, projection))
    return pairs


def _normalize_piece(pts: list[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, c0 = min(rs), min(cs)
    shape = tuple(sorted((r - r0, c - c0) for r, c in pts))
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    return shape, h, w


def _extract_pieces(inner: np.ndarray) -> list[tuple[tuple[tuple[int, int], ...], int, int]]:
    pieces = [_normalize_piece(pts) for pts in _monochrome_components(inner)]
    pieces.sort(key=lambda item: (-len(item[0]), item[1] * item[2], item[0]))
    return pieces


def _can_tile_rectangle(
    pieces: list[tuple[tuple[tuple[int, int], ...], int, int]], h: int, w: int
) -> bool:
    if sum(len(shape) for shape, _, _ in pieces) != h * w:
        return False

    by_cell: list[dict[int, list[int]]] = []
    for shape, ph, pw in pieces:
        cell_map: dict[int, list[int]] = {}
        for r0 in range(h - ph + 1):
            for c0 in range(w - pw + 1):
                mask = 0
                touched: list[int] = []
                for dr, dc in shape:
                    idx = (r0 + dr) * w + (c0 + dc)
                    mask |= 1 << idx
                    touched.append(idx)
                for idx in touched:
                    cell_map.setdefault(idx, []).append(mask)
        by_cell.append(cell_map)

    full = (1 << (h * w)) - 1

    @lru_cache(maxsize=None)
    def dfs(used_mask: int, cover_mask: int) -> bool:
        if cover_mask == full:
            return used_mask == (1 << len(pieces)) - 1
        missing = full & ~cover_mask
        first = missing & -missing
        cell = first.bit_length() - 1
        for idx in range(len(pieces)):
            if (used_mask >> idx) & 1:
                continue
            for placement in by_cell[idx].get(cell, []):
                if placement & cover_mask:
                    continue
                if dfs(used_mask | (1 << idx), cover_mask | placement):
                    return True
        return False

    return dfs(0, 0)


def _paint_frame(grid: np.ndarray, frame: dict[str, int], color: int) -> None:
    r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the outline above each grey container using the legend color selected by
    the number of colored fragments that can tile the container interior by translation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    legend = _extract_legend(g)
    frames = _find_frames(g)
    pairs = _pair_frames(g, frames)

    out = g.copy()
    for container, projection in pairs:
        inner = g[container["r0"] + 1 : container["r1"], container["c0"] + 1 : container["c1"]]
        pieces = _extract_pieces(inner)
        if not (2 <= len(pieces) <= 4):
            continue
        if not _can_tile_rectangle(pieces, inner.shape[0], inner.shape[1]):
            continue
        color = legend.get(len(pieces) - 2)
        if color is None:
            continue
        _paint_frame(out, projection, color)
    return out


def _draw_border(patch: np.ndarray, color: int) -> None:
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color


def _random_piece_sizes(total: int, n_pieces: int, rng: random.Random) -> list[int]:
    sizes: list[int] = []
    remaining = total
    for i in range(n_pieces - 1):
        pieces_left = n_pieces - i - 1
        lo = max(2, remaining - 4 * pieces_left)
        hi = min(4, remaining - 2 * pieces_left)
        sizes.append(rng.randint(lo, hi))
        remaining -= sizes[-1]
    sizes.append(remaining)
    rng.shuffle(sizes)
    return sizes


def _grow_partition(h: int, w: int, sizes: list[int], rng: random.Random) -> list[list[tuple[int, int]]] | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(PARTITION_RETRIES):
        seeds = rng.sample(cells, len(sizes))
        regions = [{seed} for seed in seeds]
        assigned = {seed: idx for idx, seed in enumerate(seeds)}
        frontiers = [set() for _ in sizes]
        for idx, (r, c) in enumerate(seeds):
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in assigned:
                    frontiers[idx].add((nr, nc))

        while len(assigned) < h * w:
            choices = [idx for idx in range(len(sizes)) if len(regions[idx]) < sizes[idx] and frontiers[idx]]
            if not choices:
                break
            weights = [sizes[idx] - len(regions[idx]) for idx in choices]
            region_idx = rng.choices(choices, weights=weights, k=1)[0]
            cell = rng.choice(tuple(frontiers[region_idx]))
            regions[region_idx].add(cell)
            assigned[cell] = region_idx
            for frontier in frontiers:
                frontier.discard(cell)
            r, c = cell
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in assigned:
                    frontiers[region_idx].add((nr, nc))

        if len(assigned) != h * w:
            continue
        if any(len(region) != size for region, size in zip(regions, sizes)):
            continue
        return [sorted(region) for region in regions]
    return None


def _make_stack(n_pieces: int, legend_colors: set[int], rng: random.Random) -> np.ndarray | None:
    inner_h, inner_w = rng.choice(INTERIOR_OPTIONS[n_pieces])
    sizes = _random_piece_sizes(inner_h * inner_w, n_pieces, rng)
    regions = _grow_partition(inner_h, inner_w, sizes, rng)
    if regions is None:
        return None

    piece_colors = rng.sample([c for c in COLOR_POOL if c not in legend_colors], n_pieces)
    container = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
    projection = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
    _draw_border(container, GREY)
    _draw_border(projection, GREY)

    rng.shuffle(regions)
    for color, region in zip(piece_colors, regions):
        for r, c in region:
            container[r + 1, c + 1] = color

    gap = rng.randint(*STACK_GAP_RANGE)
    patch = np.full((projection.shape[0] + gap + container.shape[0], container.shape[1]), BLACK, dtype=int)
    patch[: projection.shape[0], :] = projection
    patch[projection.shape[0] + gap :, :] = container
    return patch


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    offset_to_color = {offset: color for offset, color in enumerate(rng.sample(COLOR_POOL, 3))}
    legend_colors = set(offset_to_color.values())

    n_containers = rng.randint(*CONTAINER_COUNT_RANGE)
    piece_counts = [2, 3, 4]
    while len(piece_counts) < n_containers:
        piece_counts.append(rng.choice([2, 3, 4]))
    rng.shuffle(piece_counts)

    stacks: list[np.ndarray] = []
    for n_pieces in piece_counts:
        stack = _make_stack(n_pieces, legend_colors, rng)
        if stack is None:
            return None
        stacks.append(stack)

    rng.shuffle(stacks)
    stack_gaps = [rng.randint(*STACK_SPACING_RANGE) for _ in range(max(0, len(stacks) - 1))]
    top_margin = rng.randint(*TOP_MARGIN_RANGE)
    left_margin = rng.randint(*SIDE_MARGIN_RANGE)
    right_margin = rng.randint(*SIDE_MARGIN_RANGE)
    prelegend_gap = rng.randint(*PRELEGEND_GAP_RANGE)
    row_height = max(stack.shape[0] for stack in stacks)
    stack_width = sum(stack.shape[1] for stack in stacks) + sum(stack_gaps)

    legend_blocks = [(color, offset) for offset, color in offset_to_color.items()]
    rng.shuffle(legend_blocks)
    legend_gaps = [rng.randint(*LEGEND_SPACING_RANGE) for _ in range(max(0, len(legend_blocks) - 1))]
    legend_width = 3 * len(legend_blocks) + sum(legend_gaps)

    width = max(left_margin + stack_width + right_margin, left_margin + legend_width + right_margin)
    height = top_margin + row_height + prelegend_gap + 2
    grid = np.full((height, width), BLACK, dtype=int)

    c = left_margin
    for idx, stack in enumerate(stacks):
        ph, pw = stack.shape
        r = top_margin + rng.randint(0, row_height - ph)
        grid[r : r + ph, c : c + pw] = stack
        if idx < len(stack_gaps):
            c += pw + stack_gaps[idx]

    legend_start = rng.randint(left_margin, width - right_margin - legend_width)
    legend_r = height - 2
    c = legend_start
    for idx, (color, offset) in enumerate(legend_blocks):
        grid[legend_r, c : c + 3] = color
        grid[legend_r + 1, c + offset] = MAROON
        if idx < len(legend_gaps):
            c += 3 + legend_gaps[idx]

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    try:
        if not np.array_equal(main(out), out):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
