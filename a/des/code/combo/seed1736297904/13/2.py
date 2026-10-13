from __future__ import annotations

import random

import numpy as np


BG = 0
QUADS = ("tl", "tr", "bl", "br")
PALETTE = tuple(range(1, 10))
RNG = random.Random()

GRID_H_RANGE = (13, 23)
GRID_W_RANGE = (13, 23)
CORNER_SIZE_MIN = 2
CORNER_SIZE_MAX = 5
ACTIVE_QUADRANT_RANGE = (2, 4)
SEEDS_PER_QUADRANT_MAX = 3
GEN_RETRIES = 400


def _quadrant_of(r: int, c: int, h: int, w: int) -> str | None:
    if h % 2 == 1 and r == h // 2:
        return None
    if w % 2 == 1 and c == w // 2:
        return None
    top = r < h // 2
    left = c < w // 2
    if top and left:
        return "tl"
    if top and not left:
        return "tr"
    if not top and left:
        return "bl"
    return "br"


def _edge_run(values: np.ndarray, color: int) -> int:
    count = 0
    for value in values:
        if int(value) != color:
            break
        count += 1
    return count


def _corner_slice(h: int, w: int, quad: str, size: int) -> tuple[slice, slice]:
    if quad == "tl":
        return slice(0, size), slice(0, size)
    if quad == "tr":
        return slice(0, size), slice(w - size, w)
    if quad == "bl":
        return slice(h - size, h), slice(0, size)
    return slice(h - size, h), slice(w - size, w)


def _detect_corners(grid: np.ndarray) -> dict[str, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    tl_color = int(g[0, 0])
    tr_color = int(g[0, w - 1])
    bl_color = int(g[h - 1, 0])
    br_color = int(g[h - 1, w - 1])

    tl_size = min(_edge_run(g[0, :], tl_color), _edge_run(g[:, 0], tl_color))
    tr_size = min(_edge_run(g[0, ::-1], tr_color), _edge_run(g[:, w - 1], tr_color))
    bl_size = min(_edge_run(g[h - 1, :], bl_color), _edge_run(g[::-1, 0], bl_color))
    br_size = min(_edge_run(g[h - 1, ::-1], br_color), _edge_run(g[::-1, w - 1], br_color))

    return {
        "tl": {"color": tl_color, "size": tl_size},
        "tr": {"color": tr_color, "size": tr_size},
        "bl": {"color": bl_color, "size": bl_size},
        "br": {"color": br_color, "size": br_size},
    }


def _seed_square_bounds(r: int, c: int, size: int, quad: str, h: int, w: int) -> tuple[int, int, int, int]:
    if quad == "tl":
        return max(0, r - size + 1), r, max(0, c - size + 1), c
    if quad == "tr":
        return max(0, r - size + 1), r, c, min(w - 1, c + size - 1)
    if quad == "bl":
        return r, min(h - 1, r + size - 1), max(0, c - size + 1), c
    return r, min(h - 1, r + size - 1), c, min(w - 1, c + size - 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each non-corner seed into a same-sized square toward the corner of its quadrant.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    corners = _detect_corners(g)

    out = np.zeros_like(g, dtype=int)
    corner_mask = np.zeros((h, w), dtype=bool)
    for quad in QUADS:
        size = int(corners[quad]["size"])
        color = int(corners[quad]["color"])
        rs, cs = _corner_slice(h, w, quad, size)
        out[rs, cs] = color
        corner_mask[rs, cs] = True

    seeds: list[tuple[int, int, int, str]] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or bool(corner_mask[r, c]):
                continue
            quad = _quadrant_of(r, c, h, w)
            if quad is None:
                continue
            seeds.append((r, c, color, quad))

    for r, c, color, quad in seeds:
        size = int(corners[quad]["size"])
        r0, r1, c0, c1 = _seed_square_bounds(r, c, size, quad, h, w)
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == BG] = color

    return out


def _quadrant_bounds(h: int, w: int, quad: str) -> tuple[int, int, int, int]:
    if quad in ("tl", "tr"):
        r0, r1 = 0, h // 2 - 1
    else:
        r0 = h // 2 if h % 2 == 0 else h // 2 + 1
        r1 = h - 1

    if quad in ("tl", "bl"):
        c0, c1 = 0, w // 2 - 1
    else:
        c0 = w // 2 if w % 2 == 0 else w // 2 + 1
        c1 = w - 1

    return r0, r1, c0, c1


def _touches_corner_buffer(r: int, c: int, h: int, w: int, quad: str, size: int) -> bool:
    if quad == "tl":
        return r <= size and c <= size
    if quad == "tr":
        return r <= size and c >= w - size - 1
    if quad == "bl":
        return r >= h - size - 1 and c <= size
    return r >= h - size - 1 and c >= w - size - 1


def _is_clipped_seed(r: int, c: int, h: int, w: int, quad: str, size: int) -> bool:
    if quad == "tl":
        return r < size - 1 or c < size - 1
    if quad == "tr":
        return r < size - 1 or c > w - size
    if quad == "bl":
        return r > h - size or c < size - 1
    return r > h - size or c > w - size


def _clear_of_occupied(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
    for rr, cc in occupied:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    return True


def _candidate_cells(
    h: int,
    w: int,
    quad: str,
    size: int,
    occupied: set[tuple[int, int]],
    clipped_only: bool | None = None,
) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = _quadrant_bounds(h, w, quad)
    cells: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (0, h - 1) or c in (0, w - 1):
                continue
            if _touches_corner_buffer(r, c, h, w, quad, size):
                continue
            if not _clear_of_occupied(r, c, occupied):
                continue
            clipped = _is_clipped_seed(r, c, h, w, quad, size)
            if clipped_only is True and not clipped:
                continue
            if clipped_only is False and clipped:
                continue
            cells.append((r, c))
    return cells


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2])


def _pick_one(cells: list[tuple[int, int]]) -> tuple[int, int]:
    return cells[RNG.randrange(len(cells))]


def _place_overlap_pair(
    h: int,
    w: int,
    quad: str,
    size: int,
    occupied: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    cells = _candidate_cells(h, w, quad, size, occupied)
    RNG.shuffle(cells)
    for first in cells:
        first_box = _seed_square_bounds(first[0], first[1], size, quad, h, w)
        partners: list[tuple[int, int]] = []
        for second in cells:
            if second == first:
                continue
            if max(abs(first[0] - second[0]), abs(first[1] - second[1])) <= 1:
                continue
            second_box = _seed_square_bounds(second[0], second[1], size, quad, h, w)
            if _boxes_overlap(first_box, second_box):
                partners.append(second)
        if partners:
            return first, _pick_one(partners)
    return None


def _valid_generated_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    corners = _detect_corners(g)
    corner_mask = np.zeros((h, w), dtype=bool)
    used_quads = set()
    for quad in QUADS:
        size = int(corners[quad]["size"])
        color = int(corners[quad]["color"])
        if size < CORNER_SIZE_MIN or color == BG:
            return False
        rs, cs = _corner_slice(h, w, quad, size)
        patch = g[rs, cs]
        if not bool((patch == color).all()):
            return False
        corner_mask[rs, cs] = True

    seeds: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or bool(corner_mask[r, c]):
                continue
            quad = _quadrant_of(r, c, h, w)
            if quad is None:
                return False
            if color != int(corners[quad]["color"]):
                return False
            used_quads.add(quad)
            for rr, cc in seeds:
                if max(abs(r - rr), abs(c - cc)) <= 1:
                    return False
            seeds.append((r, c))

    if len(seeds) < 2 or len(used_quads) < 2:
        return False

    out = main(g)
    corner_pixels = sum(int(corners[q]["size"]) ** 2 for q in QUADS)
    if int((out != BG).sum()) <= corner_pixels:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = RNG.randint(*GRID_H_RANGE)
    w = RNG.randint(*GRID_W_RANGE)
    max_size = min(CORNER_SIZE_MAX, h // 2 - 1, w // 2 - 1)
    if max_size < CORNER_SIZE_MIN:
        return None

    need_overlap = RNG.random() < 0.5
    need_clip = RNG.random() < 0.5

    sizes = {quad: RNG.randint(CORNER_SIZE_MIN, max_size) for quad in QUADS}
    special_quads = [quad for quad in QUADS if sizes[quad] >= 3]
    if (need_overlap or need_clip) and not special_quads:
        return None

    colors = dict(zip(QUADS, RNG.sample(PALETTE, len(QUADS))))
    counts = {quad: 0 for quad in QUADS}

    active_count = RNG.randint(*ACTIVE_QUADRANT_RANGE)
    for quad in RNG.sample(list(QUADS), active_count):
        counts[quad] = RNG.randint(1, SEEDS_PER_QUADRANT_MAX)

    overlap_quad = RNG.choice(special_quads) if need_overlap else None
    if need_clip:
        clip_choices = [quad for quad in special_quads if quad != overlap_quad] or special_quads
        clip_quad = RNG.choice(clip_choices)
    else:
        clip_quad = None

    for quad in QUADS:
        needed = 0
        if quad == overlap_quad:
            needed += 2
        if quad == clip_quad:
            needed += 1
        counts[quad] = max(counts[quad], needed)

    total_seeds = sum(counts.values())
    if total_seeds < 2 or total_seeds > 9:
        return None

    grid = np.zeros((h, w), dtype=int)
    for quad in QUADS:
        rs, cs = _corner_slice(h, w, quad, sizes[quad])
        grid[rs, cs] = colors[quad]

    occupied: set[tuple[int, int]] = set()
    placed: dict[str, list[tuple[int, int]]] = {quad: [] for quad in QUADS}

    if overlap_quad is not None:
        pair = _place_overlap_pair(h, w, overlap_quad, sizes[overlap_quad], occupied)
        if pair is None:
            return None
        for cell in pair:
            occupied.add(cell)
            placed[overlap_quad].append(cell)

    if clip_quad is not None:
        clip_cells = _candidate_cells(h, w, clip_quad, sizes[clip_quad], occupied, clipped_only=True)
        if not clip_cells:
            return None
        clip_cell = _pick_one(clip_cells)
        occupied.add(clip_cell)
        placed[clip_quad].append(clip_cell)

    for quad in QUADS:
        while len(placed[quad]) < counts[quad]:
            cells = _candidate_cells(h, w, quad, sizes[quad], occupied)
            if not cells:
                return None
            cell = _pick_one(cells)
            occupied.add(cell)
            placed[quad].append(cell)

    for quad in QUADS:
        color = colors[quad]
        for r, c in placed[quad]:
            grid[r, c] = color

    if not _valid_generated_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
