from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, get_bg


WALL = Color.GREY
BG = Color.BLACK
SEED_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
DIR_BY_SIDE = {"L": (0, 1), "R": (0, -1), "T": (1, 0), "B": (-1, 0)}
OPPOSITE_SIDE = {"L": "R", "R": "L", "T": "B", "B": "T"}
TURN_PAIRS = (("L", "T"), ("L", "B"), ("R", "T"), ("R", "B"))


@dataclass(frozen=True)
class RectSpec:
    r0: int
    c0: int
    r1: int
    c1: int
    gaps: tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class SeedSpec:
    r: int
    c: int
    color: int


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _seed_direction(r: int, c: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    hits = []
    if r == 0:
        hits.append((1, 0))
    if r == h - 1:
        hits.append((-1, 0))
    if c == 0:
        hits.append((0, 1))
    if c == w - 1:
        hits.append((0, -1))
    assert len(hits) == 1, "boundary seeds must not be placed on corners"
    return hits[0]


def _rect_border_cells(rect: RectSpec) -> set[tuple[int, int]]:
    cells = set()
    for c in range(rect.c0, rect.c1 + 1):
        cells.add((rect.r0, c))
        cells.add((rect.r1, c))
    for r in range(rect.r0, rect.r1 + 1):
        cells.add((r, rect.c0))
        cells.add((r, rect.c1))
    return cells


def _gap_side(rect: RectSpec, gap: tuple[int, int]) -> str:
    r, c = gap
    if r == rect.r0 and rect.c0 < c < rect.c1:
        return "T"
    if r == rect.r1 and rect.c0 < c < rect.c1:
        return "B"
    if c == rect.c0 and rect.r0 < r < rect.r1:
        return "L"
    if c == rect.c1 and rect.r0 < r < rect.r1:
        return "R"
    raise AssertionError("gap must be a non-corner border cell")


def _entry_result(rect: RectSpec, gap: tuple[int, int], direction: tuple[int, int]) -> tuple[bool, tuple[int, int] | None]:
    side = _gap_side(rect, gap)
    other = rect.gaps[1] if rect.gaps[0] == gap else rect.gaps[0]
    other_side = _gap_side(rect, other)
    if direction == (0, 1) and side == "L":
        return True, other if other_side == "R" and other[0] == gap[0] else None
    if direction == (0, -1) and side == "R":
        return True, other if other_side == "L" and other[0] == gap[0] else None
    if direction == (1, 0) and side == "T":
        return True, other if other_side == "B" and other[1] == gap[1] else None
    if direction == (-1, 0) and side == "B":
        return True, other if other_side == "T" and other[1] == gap[1] else None
    return False, None


def _parse_rectangles(grid: np.ndarray) -> list[RectSpec]:
    h, w = grid.shape
    rects = []
    seen = set()
    for r0 in range(1, h - 3):
        for r1 in range(r0 + 3, h - 1):
            for c0 in range(1, w - 3):
                for c1 in range(c0 + 3, w - 1):
                    if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BG):
                        continue
                    rect = RectSpec(r0, c0, r1, c1, ())
                    border = sorted(_rect_border_cells(rect))
                    values = [int(grid[r, c]) for r, c in border]
                    if any(v not in (BG, WALL) for v in values):
                        continue
                    gaps = [cell for cell, v in zip(border, values) if v == BG]
                    if len(gaps) != 2:
                        continue
                    rect = RectSpec(r0, c0, r1, c1, tuple(sorted(gaps)))
                    try:
                        side0 = _gap_side(rect, rect.gaps[0])
                        side1 = _gap_side(rect, rect.gaps[1])
                    except AssertionError:
                        continue
                    if side0 == side1:
                        continue
                    key = (r0, c0, r1, c1)
                    if key not in seen:
                        seen.add(key)
                        rects.append(rect)
    for i, rect_a in enumerate(rects):
        for rect_b in rects[i + 1 :]:
            overlap_r = max(rect_a.r0, rect_b.r0) <= min(rect_a.r1, rect_b.r1)
            overlap_c = max(rect_a.c0, rect_b.c0) <= min(rect_a.c1, rect_b.c1)
            assert not (overlap_r and overlap_c), "rectangles must not overlap"
    assert rects, "expected at least one gray hollow rectangle"
    return rects


def _find_boundary_seeds(grid: np.ndarray) -> list[SeedSpec]:
    h, w = grid.shape
    coords = np.argwhere((grid != BG) & (grid != WALL))
    seeds = []
    for rr, cc in coords:
        r = int(rr)
        c = int(cc)
        assert r in (0, h - 1) or c in (0, w - 1), "colored non-wall pixels must be boundary seeds"
        _seed_direction(r, c, (h, w))
        seeds.append(SeedSpec(r, c, int(grid[r, c])))
    return sorted(seeds, key=lambda s: (s.r, s.c, s.color))


def _trace_seed_on_grid(
    grid: np.ndarray,
    rects: list[RectSpec],
    seed: SeedSpec,
) -> tuple[list[int], str]:
    shape = grid.shape
    direction = _seed_direction(seed.r, seed.c, shape)
    gap_lookup: dict[tuple[int, int], int] = {}
    for idx, rect in enumerate(rects):
        for gap in rect.gaps:
            assert gap not in gap_lookup
            gap_lookup[gap] = idx

    hits = []
    r, c = seed.r, seed.c
    while True:
        nr = r + direction[0]
        nc = c + direction[1]
        if not _in_bounds(shape, nr, nc):
            return hits, "edge"
        rect_idx = gap_lookup.get((nr, nc))
        if rect_idx is not None:
            entered, exit_gap = _entry_result(rects[rect_idx], (nr, nc), direction)
            if entered:
                hits.append(rect_idx)
                if exit_gap is None:
                    return hits, "inner_wall"
                r, c = exit_gap
                continue
        if int(grid[nr, nc]) == WALL:
            return hits, "wall"
        r, c = nr, nc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood-fill every hollow rectangle reached by a boundary seed through aligned entry and exit gaps.

    Input structure:
    - The background is black.
    - Gray one-cell-thick hollow rectangles each have exactly two one-cell gaps on their walls.
    - Colored singleton seeds sit on the outer canvas boundary and fire straight inward orthogonally.

    Rule:
    - Each seed sends an invisible straight ray in its inward direction; hitting any gray wall before a gap stops it.
    - If the ray crosses a rectangle through one wall gap, that rectangle's entire interior is filled with the seed color.
    - Inside the rectangle the ray keeps the same direction; it continues only if it exits through the second gap on the opposite aligned wall, allowing a chain into later rectangles.
    - Keep the gray walls, background, and seed pixels unchanged; only the interiors of reached rectangles are recolored.

    Key insight:
    - The output depends on line-of-sight reachability through gap-to-gap lock channels, not on proximity or simple row/column matching.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert get_bg(g) == BG

    rects = _parse_rectangles(g)
    seeds = _find_boundary_seeds(g)

    out = g.copy()
    filled_by: dict[int, int] = {}
    for seed in seeds:
        hit_rects, _termination = _trace_seed_on_grid(g, rects, seed)
        for rect_idx in hit_rects:
            prev = filled_by.get(rect_idx)
            assert prev in (None, seed.color), "a rectangle must be reached by at most one effective seed color"
            filled_by[rect_idx] = seed.color

    for rect_idx, color in filled_by.items():
        rect = rects[rect_idx]
        out[rect.r0 + 1 : rect.r1, rect.c0 + 1 : rect.c1] = color
    return out


def _can_place(occupied: np.ndarray, rect: RectSpec, pad: int = 1) -> bool:
    h, w = occupied.shape
    if rect.r0 < 1 or rect.c0 < 1 or rect.r1 > h - 2 or rect.c1 > w - 2:
        return False
    rr0 = max(0, rect.r0 - pad)
    cc0 = max(0, rect.c0 - pad)
    rr1 = min(h - 1, rect.r1 + pad)
    cc1 = min(w - 1, rect.c1 + pad)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_occupied(occupied: np.ndarray, rect: RectSpec, pad: int = 1) -> None:
    h, w = occupied.shape
    rr0 = max(0, rect.r0 - pad)
    cc0 = max(0, rect.c0 - pad)
    rr1 = min(h - 1, rect.r1 + pad)
    cc1 = min(w - 1, rect.c1 + pad)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _random_aligned_rect(h: int, w: int, orientation: str, lane: int, side: str, cursor: int) -> RectSpec | None:
    height = random.randint(4, 7)
    width = random.randint(4, 7)
    if orientation == "H":
        offset = random.randint(1, height - 2)
        r0 = lane - offset
        r1 = r0 + height - 1
        if side == "L":
            c0 = cursor
            c1 = c0 + width - 1
        else:
            c1 = cursor
            c0 = c1 - width + 1
        if not (1 <= r0 < r1 <= h - 2 and 1 <= c0 < c1 <= w - 2):
            return None
        gaps = ((lane, c0), (lane, c1))
        return RectSpec(r0, c0, r1, c1, gaps)

    offset = random.randint(1, width - 2)
    c0 = lane - offset
    c1 = c0 + width - 1
    if side == "T":
        r0 = cursor
        r1 = r0 + height - 1
    else:
        r1 = cursor
        r0 = r1 - height + 1
    if not (1 <= r0 < r1 <= h - 2 and 1 <= c0 < c1 <= w - 2):
        return None
    gaps = ((r0, lane), (r1, lane))
    return RectSpec(r0, c0, r1, c1, gaps)


def _place_chain(
    shape: tuple[int, int],
    occupied: np.ndarray,
    used_rows: set[int],
    used_cols: set[int],
    orientation: str,
    side: str,
    length: int,
) -> tuple[list[RectSpec], tuple[int, int]] | None:
    h, w = shape
    for _ in range(120):
        if orientation == "H":
            candidates = [r for r in range(2, h - 2) if r not in used_rows]
            if not candidates:
                return None
            lane = random.choice(candidates)
            cursor = random.randint(3, 6) if side == "L" else random.randint(w - 7, w - 4)
        else:
            candidates = [c for c in range(2, w - 2) if c not in used_cols]
            if not candidates:
                return None
            lane = random.choice(candidates)
            cursor = random.randint(3, 6) if side == "T" else random.randint(h - 7, h - 4)

        rects = []
        local_occ = occupied.copy()
        ok = True
        for _idx in range(length):
            rect = _random_aligned_rect(h, w, orientation, lane, side, cursor)
            if rect is None or not _can_place(local_occ, rect):
                ok = False
                break
            rects.append(rect)
            _mark_occupied(local_occ, rect)
            if orientation == "H":
                cursor = rect.c1 + random.randint(2, 5) if side == "L" else rect.c0 - random.randint(2, 5)
            else:
                cursor = rect.r1 + random.randint(2, 5) if side == "T" else rect.r0 - random.randint(2, 5)
        if not ok:
            continue
        for rect in rects:
            _mark_occupied(occupied, rect)
        if orientation == "H":
            used_rows.add(lane)
            seed_pos = (lane, 0) if side == "L" else (lane, w - 1)
        else:
            used_cols.add(lane)
            seed_pos = (0, lane) if side == "T" else (h - 1, lane)
        return rects, seed_pos
    return None


def _random_side_cell(rect: RectSpec, side: str) -> tuple[int, int]:
    if side == "L":
        return random.randint(rect.r0 + 1, rect.r1 - 1), rect.c0
    if side == "R":
        return random.randint(rect.r0 + 1, rect.r1 - 1), rect.c1
    if side == "T":
        return rect.r0, random.randint(rect.c0 + 1, rect.c1 - 1)
    return rect.r1, random.randint(rect.c0 + 1, rect.c1 - 1)


def _isolates_corner(rect: RectSpec, side_a: str, gap_a: tuple[int, int], side_b: str, gap_b: tuple[int, int]) -> bool:
    gap_by_side = {side_a: gap_a, side_b: gap_b}
    if {"L", "T"} == set(gap_by_side):
        return gap_by_side["L"][0] == rect.r0 + 1 and gap_by_side["T"][1] == rect.c0 + 1
    if {"L", "B"} == set(gap_by_side):
        return gap_by_side["L"][0] == rect.r1 - 1 and gap_by_side["B"][1] == rect.c0 + 1
    if {"R", "T"} == set(gap_by_side):
        return gap_by_side["R"][0] == rect.r0 + 1 and gap_by_side["T"][1] == rect.c1 - 1
    if {"R", "B"} == set(gap_by_side):
        return gap_by_side["R"][0] == rect.r1 - 1 and gap_by_side["B"][1] == rect.c1 - 1
    return False


def _place_passive_rect(
    shape: tuple[int, int],
    occupied: np.ndarray,
    active_rows: set[int],
    active_cols: set[int],
) -> RectSpec | None:
    h, w = shape
    for _ in range(160):
        height = random.randint(4, 7)
        width = random.randint(4, 7)
        r0 = random.randint(1, h - height - 1)
        c0 = random.randint(1, w - width - 1)
        rect = RectSpec(r0, c0, r0 + height - 1, c0 + width - 1, ())
        if any(r0 < row < rect.r1 for row in active_rows):
            continue
        if any(c0 < col < rect.c1 for col in active_cols):
            continue
        if not _can_place(occupied, rect):
            continue
        pattern = random.choice(["H", "V", "TURN"])
        if pattern == "H":
            gap_row = random.randint(rect.r0 + 1, rect.r1 - 1)
            gaps = ((gap_row, rect.c0), (gap_row, rect.c1))
        elif pattern == "V":
            gap_col = random.randint(rect.c0 + 1, rect.c1 - 1)
            gaps = ((rect.r0, gap_col), (rect.r1, gap_col))
        else:
            side_a, side_b = random.choice(TURN_PAIRS)
            gap_a = _random_side_cell(rect, side_a)
            gap_b = _random_side_cell(rect, side_b)
            if gap_a == gap_b:
                continue
            if _isolates_corner(rect, side_a, gap_a, side_b, gap_b):
                continue
            gaps = (gap_a, gap_b)
        rect = RectSpec(rect.r0, rect.c0, rect.r1, rect.c1, gaps)
        _mark_occupied(occupied, rect)
        return rect
    return None


def _make_turn_rect(
    rect: RectSpec,
    entry_side: str,
    exit_side: str,
    lane: int | None = None,
) -> RectSpec | None:
    if entry_side == "L":
        gap_entry = (lane if lane is not None else random.randint(rect.r0 + 1, rect.r1 - 1), rect.c0)
    elif entry_side == "R":
        gap_entry = (lane if lane is not None else random.randint(rect.r0 + 1, rect.r1 - 1), rect.c1)
    elif entry_side == "T":
        gap_entry = (rect.r0, lane if lane is not None else random.randint(rect.c0 + 1, rect.c1 - 1))
    else:
        gap_entry = (rect.r1, lane if lane is not None else random.randint(rect.c0 + 1, rect.c1 - 1))
    gap_exit = _random_side_cell(rect, exit_side)
    if gap_entry == gap_exit:
        return None
    if _isolates_corner(rect, entry_side, gap_entry, exit_side, gap_exit):
        return None
    return RectSpec(rect.r0, rect.c0, rect.r1, rect.c1, (gap_entry, gap_exit))


def _place_inner_wall_trap(
    shape: tuple[int, int],
    occupied: np.ndarray,
    rects: list[RectSpec],
    used_rows: set[int],
    used_cols: set[int],
    used_boundary: set[tuple[int, int]],
    color: int,
) -> tuple[RectSpec, SeedSpec] | None:
    h, w = shape
    entry_to_exits = {"L": ["T", "B"], "R": ["T", "B"], "T": ["L", "R"], "B": ["L", "R"]}
    for _ in range(240):
        height = random.randint(4, 7)
        width = random.randint(4, 7)
        r0 = random.randint(1, h - height - 1)
        c0 = random.randint(1, w - width - 1)
        base = RectSpec(r0, c0, r0 + height - 1, c0 + width - 1, ())
        if not _can_place(occupied, base):
            continue
        entry_side = random.choice(["L", "R", "T", "B"])
        if entry_side in ("L", "R"):
            lane_vals = range(base.r0 + 1, base.r1)
        else:
            lane_vals = range(base.c0 + 1, base.c1)
        lane = random.choice(list(lane_vals))
        if entry_side == "L" and lane in used_rows:
            continue
        if entry_side == "R" and lane in used_rows:
            continue
        if entry_side == "T" and lane in used_cols:
            continue
        if entry_side == "B" and lane in used_cols:
            continue
        exit_side = random.choice(entry_to_exits[entry_side])
        trap = _make_turn_rect(base, entry_side, exit_side, lane)
        if trap is None:
            continue
        entry_gap = next(g for g in trap.gaps if _gap_side(trap, g) == entry_side)
        if entry_side == "L":
            seed_pos = (entry_gap[0], 0)
        elif entry_side == "R":
            seed_pos = (entry_gap[0], w - 1)
        elif entry_side == "T":
            seed_pos = (0, entry_gap[1])
        else:
            seed_pos = (h - 1, entry_gap[1])
        if seed_pos in used_boundary:
            continue
        seed = SeedSpec(seed_pos[0], seed_pos[1], color)
        hit_rects, termination = _trace_seed_on_specs(shape, rects + [trap], seed)
        if hit_rects == [len(rects)] and termination == "inner_wall":
            _mark_occupied(occupied, trap)
            return trap, seed
    return None


def _render_scene(shape: tuple[int, int], rects: list[RectSpec], seeds: list[SeedSpec]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for rect in rects:
        for r, c in _rect_border_cells(rect):
            grid[r, c] = WALL
        for r, c in rect.gaps:
            grid[r, c] = BG
    for seed in seeds:
        assert grid[seed.r, seed.c] == BG
        grid[seed.r, seed.c] = seed.color
    return grid


def _trace_seed_on_specs(
    shape: tuple[int, int],
    rects: list[RectSpec],
    seed: SeedSpec,
) -> tuple[list[int], str]:
    wall_cells = set()
    gap_lookup: dict[tuple[int, int], int] = {}
    for idx, rect in enumerate(rects):
        wall_cells |= _rect_border_cells(rect) - set(rect.gaps)
        for gap in rect.gaps:
            gap_lookup[gap] = idx

    direction = _seed_direction(seed.r, seed.c, shape)
    hits = []
    r, c = seed.r, seed.c
    while True:
        nr = r + direction[0]
        nc = c + direction[1]
        if not _in_bounds(shape, nr, nc):
            return hits, "edge"
        rect_idx = gap_lookup.get((nr, nc))
        if rect_idx is not None:
            entered, exit_gap = _entry_result(rects[rect_idx], (nr, nc), direction)
            if entered:
                hits.append(rect_idx)
                if exit_gap is None:
                    return hits, "inner_wall"
                r, c = exit_gap
                continue
        if (nr, nc) in wall_cells:
            return hits, "wall"
        r, c = nr, nc


def _expected_output(shape: tuple[int, int], rects: list[RectSpec], seeds: list[SeedSpec]) -> tuple[np.ndarray, dict]:
    grid = _render_scene(shape, rects, seeds)
    out = grid.copy()
    fill_by_rect: dict[int, int] = {}
    traces = []
    for seed in sorted(seeds, key=lambda s: (s.r, s.c, s.color)):
        hit_rects, termination = _trace_seed_on_specs(shape, rects, seed)
        traces.append({"seed": seed, "hit_rects": hit_rects, "termination": termination})
        for rect_idx in hit_rects:
            prev = fill_by_rect.get(rect_idx)
            assert prev in (None, seed.color)
            fill_by_rect[rect_idx] = seed.color
    for rect_idx, color in fill_by_rect.items():
        rect = rects[rect_idx]
        out[rect.r0 + 1 : rect.r1, rect.c0 + 1 : rect.c1] = color
    meta = {
        "trace_info": traces,
        "hit_rect_count": len(fill_by_rect),
        "unhit_rect_count": len(rects) - len(fill_by_rect),
        "max_chain_len": max((len(t["hit_rects"]) for t in traces), default=0),
        "blocked_seed_count": sum(1 for t in traces if not t["hit_rects"] and t["termination"] == "wall"),
    }
    return out, meta


def _find_blocked_decoy_seed(
    shape: tuple[int, int],
    rects: list[RectSpec],
    used_boundary: set[tuple[int, int]],
    color: int,
) -> SeedSpec | None:
    h, w = shape
    candidates = []
    for r in range(1, h - 1):
        candidates.append((r, 0))
        candidates.append((r, w - 1))
    for c in range(1, w - 1):
        candidates.append((0, c))
        candidates.append((h - 1, c))
    random.shuffle(candidates)
    for r, c in candidates:
        if (r, c) in used_boundary:
            continue
        seed = SeedSpec(r, c, color)
        hit_rects, termination = _trace_seed_on_specs(shape, rects, seed)
        if not hit_rects and termination == "wall":
            return seed
    return None


def _generate_case(return_meta: bool = False):
    for _ in range(400):
        shape = (random.randint(18, 30), random.randint(18, 30))
        occupied = np.zeros(shape, dtype=bool)
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        used_boundary: set[tuple[int, int]] = set()
        rects: list[RectSpec] = []
        seeds: list[SeedSpec] = []
        colors = SEED_COLORS[:]
        random.shuffle(colors)

        chain_count = random.randint(1, 3)
        ok = True
        for _chain_idx in range(chain_count):
            orientation = random.choice(["H", "V"])
            side = random.choice(["L", "R"]) if orientation == "H" else random.choice(["T", "B"])
            length = random.choice([1, 2, 2, 3])
            placed = _place_chain(shape, occupied, used_rows, used_cols, orientation, side, length)
            if placed is None:
                ok = False
                break
            chain_rects, seed_pos = placed
            if seed_pos in used_boundary:
                ok = False
                break
            rects.extend(chain_rects)
            color = colors.pop()
            seeds.append(SeedSpec(seed_pos[0], seed_pos[1], color))
            used_boundary.add(seed_pos)
        if not ok:
            continue

        passive_target = random.randint(1, 3)
        for _ in range(passive_target):
            rect = _place_passive_rect(shape, occupied, used_rows, used_cols)
            if rect is not None:
                rects.append(rect)

        if colors:
            trap = _place_inner_wall_trap(shape, occupied, rects, used_rows, used_cols, used_boundary, colors[0])
            if trap is not None:
                trap_rect, trap_seed = trap
                rects.append(trap_rect)
                seeds.append(trap_seed)
                used_boundary.add((trap_seed.r, trap_seed.c))
                colors.pop(0)

        if len(rects) < 2:
            continue

        decoy_target = random.randint(1, min(3, len(colors)))
        for _ in range(decoy_target):
            color = colors.pop()
            decoy = _find_blocked_decoy_seed(shape, rects, used_boundary, color)
            if decoy is None:
                break
            seeds.append(decoy)
            used_boundary.add((decoy.r, decoy.c))

        try:
            expected, meta = _expected_output(shape, rects, seeds)
        except AssertionError:
            continue
        if meta["hit_rect_count"] < 1:
            continue
        if meta["unhit_rect_count"] < 1:
            continue
        if meta["blocked_seed_count"] < 1:
            continue
        if len({tuple(rect.gaps) for rect in rects}) == 1:
            continue

        grid = _render_scene(shape, rects, seeds)
        if return_meta:
            return grid, expected, {"rects": rects, "seeds": seeds, **meta}
        return grid
    raise RuntimeError("failed to generate a valid key-combo ARC input")


def generate_input() -> np.ndarray:
    return _generate_case(return_meta=False)
