from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
RANK_COLORS = (BLUE, RED, GREEN, YELLOW)

DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
INWARD = {
    "tl": ("down", "right"),
    "tr": ("down", "left"),
    "bl": ("up", "right"),
    "br": ("up", "left"),
}
OUTWARD = {
    "tl": ("up", "left"),
    "tr": ("up", "right"),
    "bl": ("down", "left"),
    "br": ("down", "right"),
}
SLOTS = ("tl", "tr", "bl", "br")

GEN_ATTEMPTS = 600
ASSIGNMENT_ATTEMPTS = 240
PLACEMENT_ATTEMPTS = 80
LINE_ATTEMPTS = 40


@dataclass(frozen=True)
class Pocket:
    r: int
    c: int
    side: int
    direction: str
    offset: int
    slot: str = ""


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _outer_bbox(pocket: Pocket) -> tuple[int, int, int, int]:
    return pocket.r - 1, pocket.c - 1, pocket.r + pocket.side, pocket.c + pocket.side


def _gap_cell(pocket: Pocket) -> tuple[int, int]:
    if pocket.direction == "up":
        return pocket.r - 1, pocket.c + pocket.offset
    if pocket.direction == "down":
        return pocket.r + pocket.side, pocket.c + pocket.offset
    if pocket.direction == "left":
        return pocket.r + pocket.offset, pocket.c - 1
    return pocket.r + pocket.offset, pocket.c + pocket.side


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells = []
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if rr in (r0, r1) or cc in (c0, c1):
                cells.append((rr, cc))
    return cells


def _frame_cells(pocket: Pocket) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _outer_bbox(pocket)
    gap = _gap_cell(pocket)
    return [(rr, cc) for rr, cc in _border_cells(r0, c0, r1, c1) if (rr, cc) != gap]


def _interior_slices(pocket: Pocket) -> tuple[slice, slice]:
    return slice(pocket.r, pocket.r + pocket.side), slice(pocket.c, pocket.c + pocket.side)


def _paint_frame(grid: np.ndarray, pocket: Pocket) -> None:
    for rr, cc in _frame_cells(pocket):
        grid[rr, cc] = GREY


def _ray_path(grid: np.ndarray, pocket: Pocket) -> tuple[list[tuple[int, int]], bool]:
    dr, dc = DIRECTIONS[pocket.direction]
    h, w = grid.shape
    rr, cc = _gap_cell(pocket)
    path: list[tuple[int, int]] = []
    while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != GREY:
        path.append((rr, cc))
        rr += dr
        cc += dc
    blocked = 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == GREY
    return path, blocked


def _pocket_key(pocket: Pocket) -> tuple[int, int, int, str, int]:
    return (pocket.r, pocket.c, pocket.side, pocket.direction, pocket.offset)


def _candidate_from_window(grid: np.ndarray, r: int, c: int, side: int) -> Pocket | None:
    h, w = grid.shape
    if r <= 0 or c <= 0 or r + side >= h or c + side >= w:
        return None
    if not np.all(grid[r : r + side, c : c + side] == BLACK):
        return None

    r0, c0, r1, c1 = r - 1, c - 1, r + side, c + side
    gaps: list[tuple[int, int]] = []
    for rr, cc in _border_cells(r0, c0, r1, c1):
        value = int(grid[rr, cc])
        if value == BLACK:
            gaps.append((rr, cc))
        elif value != GREY:
            return None
    if len(gaps) != 1:
        return None

    gap_r, gap_c = gaps[0]
    if (gap_r, gap_c) in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None
    if gap_r == r0:
        return Pocket(r, c, side, "up", gap_c - c)
    if gap_r == r1:
        return Pocket(r, c, side, "down", gap_c - c)
    if gap_c == c0:
        return Pocket(r, c, side, "left", gap_r - r)
    if gap_c == c1:
        return Pocket(r, c, side, "right", gap_r - r)
    return None


def _find_candidates(grid: np.ndarray) -> list[Pocket]:
    h, w = grid.shape
    candidates: list[Pocket] = []
    for side in range(min(h, w) - 2, 1, -1):
        for r in range(1, h - side):
            for c in range(1, w - side):
                pocket = _candidate_from_window(grid, r, c, side)
                if pocket is not None:
                    candidates.append(pocket)
    return candidates


def _select_pockets(candidates: Iterable[Pocket], shape: tuple[int, int]) -> list[Pocket]:
    occupied = np.zeros(shape, dtype=bool)
    chosen: list[Pocket] = []
    order = sorted(candidates, key=lambda p: (-p.side, p.r, p.c, p.direction, p.offset))
    for pocket in order:
        rs, cs = _interior_slices(pocket)
        if occupied[rs, cs].any():
            continue
        occupied[rs, cs] = True
        chosen.append(pocket)
    return chosen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each black square room with a size-rank color and shoot a ray through its single frame gap.

    Input structure
    - The grid uses black background and grey obstacle pixels only.
    - Relevant objects are black axis-aligned square interiors of side at least 2.
    - Each relevant square is surrounded by a one-cell-thick grey frame whose border is missing exactly one non-corner cell.

    Rule
    - Detect every black square whose expanded outer border is grey except for one black gap cell.
    - Greedily keep non-overlapping candidates from larger side length to smaller side length.
    - Color the kept squares by size rank: largest blue, second red, third green, fourth yellow.
    - Starting from each gap cell, extend a one-cell-wide ray outward perpendicular to the missing side until a grey cell or the grid edge stops it.
    - If multiple rays use the same cell, keep the color of the larger square; preserve all grey cells and all untouched black cells.

    Key insight
    - The solver must combine local single-gap frame detection with global size ordering, because the square size determines both the fill color and the winner at ray intersections.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert set(map(int, np.unique(g))) <= {BLACK, GREY}, "expected only black and grey in the input"

    selected = _select_pockets(_find_candidates(g), g.shape)
    assert len(selected) <= 4, "expected at most four ranked squares"

    out = g.copy()
    priority = np.full(g.shape, -1, dtype=int)
    ray_color = np.full(g.shape, -1, dtype=int)

    for rank, pocket in enumerate(selected):
        color = RANK_COLORS[rank]
        path, _blocked = _ray_path(g, pocket)
        for rr, cc in path:
            if pocket.side > priority[rr, cc]:
                priority[rr, cc] = pocket.side
                ray_color[rr, cc] = color

    ray_mask = priority >= 0
    out[ray_mask] = ray_color[ray_mask]
    for rank, pocket in enumerate(selected):
        rs, cs = _interior_slices(pocket)
        out[rs, cs] = RANK_COLORS[rank]
    return out


def _slot_ranges(h: int, w: int, side: int, slot: str) -> tuple[range, range] | None:
    margin = 2
    center_gap = 2
    if slot[0] == "t":
        r0 = margin
        r1 = h // 2 - side - center_gap
    else:
        r0 = h // 2 + center_gap
        r1 = h - side - margin - 1
    if slot[1] == "l":
        c0 = margin
        c1 = w // 2 - side - center_gap
    else:
        c0 = w // 2 + center_gap
        c1 = w - side - margin - 1
    if r0 > r1 or c0 > c1:
        return None
    return range(r0, r1 + 1), range(c0, c1 + 1)


def _place_quadrant_pockets(
    rng: random.Random,
    h: int,
    w: int,
    slot_sizes: dict[str, int],
) -> dict[str, tuple[int, int, int]] | None:
    placed: dict[str, tuple[int, int, int]] = {}
    used_boxes: list[tuple[int, int, int, int]] = []
    for slot, side in slot_sizes.items():
        ranges = _slot_ranges(h, w, side, slot)
        if ranges is None:
            return None
        rows, cols = ranges
        options = [(r, c) for r in rows for c in cols]
        rng.shuffle(options)
        chosen = None
        for r, c in options[:PLACEMENT_ATTEMPTS]:
            r0, c0, r1, c1 = r - 1, c - 1, r + side, c + side
            if any(not (r1 < ar0 or ar1 < r0 or c1 < ac0 or ac1 < c0) for ar0, ac0, ar1, ac1 in used_boxes):
                continue
            chosen = (r, c, side)
            used_boxes.append((r0, c0, r1, c1))
            break
        if chosen is None:
            return None
        placed[slot] = chosen
    return placed


def _pairwise_overlaps(
    paths: list[list[tuple[int, int]]],
) -> list[tuple[int, int, tuple[int, int], int, int]]:
    overlaps = []
    sets = [set(path) for path in paths]
    for i in range(len(paths)):
        index_i = {cell: idx for idx, cell in enumerate(paths[i])}
        for j in range(i + 1, len(paths)):
            index_j = {cell: idx for idx, cell in enumerate(paths[j])}
            shared = sets[i] & sets[j]
            for cell in shared:
                overlaps.append((i, j, cell, index_i[cell], index_j[cell]))
    return overlaps


def _assign_directions_and_offsets(
    rng: random.Random,
    h: int,
    w: int,
    bases: dict[str, tuple[int, int, int]],
    cross_mode: bool,
) -> tuple[list[Pocket], list[list[tuple[int, int]]], list[tuple[int, int, tuple[int, int], int, int]]] | None:
    pocket_slots = sorted(bases)
    allowed_lookup = INWARD if cross_mode else OUTWARD
    for _ in range(ASSIGNMENT_ATTEMPTS):
        pockets: list[Pocket] = []
        for slot in pocket_slots:
            r, c, side = bases[slot]
            direction = rng.choice(allowed_lookup[slot])
            offset = rng.randrange(side)
            pockets.append(Pocket(r, c, side, direction, offset, slot))

        if len({p.direction for p in pockets}) < 2:
            continue

        grid = np.zeros((h, w), dtype=int)
        for pocket in pockets:
            _paint_frame(grid, pocket)

        paths: list[list[tuple[int, int]]] = []
        valid = True
        for pocket in pockets:
            path, blocked = _ray_path(grid, pocket)
            if blocked or len(path) < 4:
                valid = False
                break
            paths.append(path)
        if not valid:
            continue

        overlaps = _pairwise_overlaps(paths)
        if cross_mode and not overlaps:
            continue
        if not cross_mode and overlaps:
            continue
        return pockets, paths, overlaps
    return None


def _mark_halo(mask: np.ndarray, cells: Iterable[tuple[int, int]], radius: int = 1) -> None:
    h, w = mask.shape
    for rr, cc in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    mask[nr, nc] = True


def _install_stoppers(
    rng: random.Random,
    grid: np.ndarray,
    pockets: list[Pocket],
    full_paths: list[list[tuple[int, int]]],
    overlaps: list[tuple[int, int, tuple[int, int], int, int]],
    cross_mode: bool,
) -> tuple[list[list[tuple[int, int]]], tuple[int, int, tuple[int, int]] | None]:
    must_keep = [4] * len(pockets)
    chosen_overlap: tuple[int, int, tuple[int, int]] | None = None
    if cross_mode:
        i, j, cell, idx_i, idx_j = rng.choice(overlaps)
        must_keep[i] = max(must_keep[i], idx_i + 1)
        must_keep[j] = max(must_keep[j], idx_j + 1)
        chosen_overlap = (i, j, cell)

    shared_cells = set()
    for i, j, cell, _idx_i, _idx_j in overlaps:
        shared_cells.add(cell)

    for idx, path in enumerate(full_paths):
        stop_candidates = [k for k in range(must_keep[idx], len(path)) if path[k] not in shared_cells]
        if stop_candidates and rng.random() < 0.75:
            upper = min(len(stop_candidates), 6)
            cut_index = stop_candidates[rng.randrange(upper)]
            rr, cc = path[cut_index]
            grid[rr, cc] = GREY

    actual_paths = [_ray_path(grid, pocket)[0] for pocket in pockets]
    return actual_paths, chosen_overlap


def _area_free(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = mask.shape
    if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w):
        return False
    return not mask[r0 : r1 + 1, c0 : c1 + 1].any()


def _draw_room_frame(
    grid: np.ndarray,
    r: int,
    c: int,
    height: int,
    width: int,
    gaps: list[tuple[int, int]],
) -> None:
    r0, c0, r1, c1 = r - 1, c - 1, r + height, c + width
    gap_set = set(gaps)
    for rr, cc in _border_cells(r0, c0, r1, c1):
        if (rr, cc) not in gap_set:
            grid[rr, cc] = GREY


def _noncorner_border_positions(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = r - 1, c - 1, r + height, c + width
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    return [cell for cell in _border_cells(r0, c0, r1, c1) if cell not in corners]


def _place_invalid_square(rng: random.Random, grid: np.ndarray, reserved: np.ndarray) -> bool:
    h, w = grid.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        side = rng.choice((2, 3, 4))
        r = rng.randint(2, h - side - 3)
        c = rng.randint(2, w - side - 3)
        if not _area_free(reserved, r - 2, c - 2, r + side + 1, c + side + 1):
            continue
        positions = _noncorner_border_positions(r, c, side, side)
        if len(positions) < 2:
            continue
        if rng.random() < 0.5:
            gaps = rng.sample(positions, 2)
        else:
            r0, c0, r1, c1 = r - 1, c - 1, r + side, c + side
            gaps = [rng.choice([(r0, c0), (r0, c1), (r1, c0), (r1, c1)])]
        _draw_room_frame(grid, r, c, side, side, gaps)
        _mark_halo(reserved, [(rr, cc) for rr in range(r - 1, r + side + 1) for cc in range(c - 1, c + side + 1)], radius=1)
        return True
    return False


def _place_rectangle_distractor(rng: random.Random, grid: np.ndarray, reserved: np.ndarray) -> bool:
    h, w = grid.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        height = rng.choice((2, 3, 4, 5))
        width = rng.choice((2, 3, 4, 5))
        if height == width:
            continue
        r = rng.randint(2, h - height - 3)
        c = rng.randint(2, w - width - 3)
        if not _area_free(reserved, r - 2, c - 2, r + height + 1, c + width + 1):
            continue
        positions = _noncorner_border_positions(r, c, height, width)
        if not positions:
            continue
        gaps = [rng.choice(positions)]
        _draw_room_frame(grid, r, c, height, width, gaps)
        _mark_halo(reserved, [(rr, cc) for rr in range(r - 1, r + height + 1) for cc in range(c - 1, c + width + 1)], radius=1)
        return True
    return False


def _place_extra_segments(rng: random.Random, grid: np.ndarray, reserved: np.ndarray) -> int:
    h, w = grid.shape
    placed = 0
    target = rng.randint(2, 5)
    for _ in range(target * LINE_ATTEMPTS):
        if placed >= target:
            break
        length = rng.randint(3, 8)
        horizontal = rng.random() < 0.5
        if horizontal:
            rr = rng.randint(1, h - 2)
            cc = rng.randint(1, w - length - 1)
            cells = [(rr, cc + delta) for delta in range(length)]
        else:
            rr = rng.randint(1, h - length - 1)
            cc = rng.randint(1, w - 2)
            cells = [(rr + delta, cc) for delta in range(length)]
        if any(reserved[r, c] for r, c in cells):
            continue
        if any(int(grid[r, c]) == GREY for r, c in cells):
            continue
        for r, c in cells:
            grid[r, c] = GREY
        _mark_halo(reserved, cells, radius=0)
        placed += 1
    return placed


def _validate_instance(
    grid: np.ndarray,
    intended: list[Pocket],
    cross_mode: bool,
    chosen_overlap: tuple[int, int, tuple[int, int]] | None,
    placed_invalid_square: bool,
    placed_rectangle: bool,
) -> bool:
    detected = _select_pockets(_find_candidates(grid), grid.shape)
    if {_pocket_key(p) for p in detected} != {_pocket_key(p) for p in intended}:
        return False
    if len(detected) != len(intended):
        return False
    if len(detected) < 3 or len(detected) > 4:
        return False

    output = main(grid)
    paths = [_ray_path(grid, pocket)[0] for pocket in detected]
    overlaps = _pairwise_overlaps(paths)
    if cross_mode and not overlaps:
        return False
    if not cross_mode and overlaps:
        return False

    ordered = sorted(detected, key=lambda p: (-p.side, p.r, p.c, p.direction, p.offset))
    for rank, pocket in enumerate(ordered):
        rs, cs = _interior_slices(pocket)
        if not np.all(output[rs, cs] == RANK_COLORS[rank]):
            return False
        if len(_ray_path(grid, pocket)[0]) < 4:
            return False

    if chosen_overlap is not None:
        _oi, _oj, cell = chosen_overlap
        contenders = [pocket for pocket in detected if cell in set(_ray_path(grid, pocket)[0])]
        if len(contenders) < 2:
            return False
        winner = max(contenders, key=lambda pocket: pocket.side)
        winner_rank = ordered.index(winner)
        if int(output[cell]) != RANK_COLORS[winner_rank]:
            return False

    return placed_invalid_square and placed_rectangle


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(21, 26)
    w = rng.randint(21, 26)
    count = 4 if rng.random() < 0.45 else 3
    chosen_slots = rng.sample(SLOTS, count)
    chosen_sizes = rng.sample((2, 3, 4, 5), count)
    rng.shuffle(chosen_sizes)
    slot_sizes = {slot: size for slot, size in zip(chosen_slots, chosen_sizes)}

    bases = _place_quadrant_pockets(rng, h, w, slot_sizes)
    if bases is None:
        return None

    cross_mode = rng.random() < 0.6
    assignment = _assign_directions_and_offsets(rng, h, w, bases, cross_mode)
    if assignment is None:
        return None
    pockets, full_paths, overlaps = assignment

    grid = np.zeros((h, w), dtype=int)
    for pocket in pockets:
        _paint_frame(grid, pocket)

    actual_paths, chosen_overlap = _install_stoppers(rng, grid, pockets, full_paths, overlaps, cross_mode)
    for pocket, path in zip(pockets, actual_paths):
        if len(path) < 4:
            return None
        path_set = set(path)
        for other in pockets:
            if other == pocket:
                continue
            if path_set & set(_frame_cells(other)):
                return None

    reserved = np.zeros_like(grid, dtype=bool)
    for pocket in pockets:
        r0, c0, r1, c1 = _outer_bbox(pocket)
        _mark_halo(reserved, [(rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)], radius=1)
    for path in actual_paths:
        _mark_halo(reserved, path, radius=1)

    placed_invalid_square = _place_invalid_square(rng, grid, reserved)
    placed_rectangle = _place_rectangle_distractor(rng, grid, reserved)
    if not (placed_invalid_square and placed_rectangle):
        return None
    if _place_extra_segments(rng, grid, reserved) < 2:
        return None

    intended = sorted(pockets, key=lambda p: (-p.side, p.r, p.c, p.direction, p.offset))
    if not _validate_instance(
        grid,
        intended,
        cross_mode,
        chosen_overlap,
        placed_invalid_square,
        placed_rectangle,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
