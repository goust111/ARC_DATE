from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
FRAME_COLOR = 5
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_bool(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        out.append(np.array(coords, dtype=int))
    return out


def _foreground_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BLACK):
        for coords in _components_bool(grid == color):
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    comps.sort(key=lambda comp: comp["bbox"])
    return comps


def _is_hollow_rectangle(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False

    expected = np.zeros((height, width), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if int(expected.sum()) != int(comp["area"]):
        return False

    got = np.zeros((height, width), dtype=bool)
    for r, c in comp["coords"]:
        got[int(r) - r0, int(c) - c0] = True
    if not np.array_equal(got, expected):
        return False

    return bool(np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BLACK))


def _piece_from_component(comp: dict) -> dict:
    r0, c0, r1, c1 = comp["bbox"]
    local = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in comp["coords"]))
    return {
        "color": int(comp["color"]),
        "cells": local,
        "h": r1 - r0 + 1,
        "w": c1 - c0 + 1,
        "area": int(comp["area"]),
        "source": tuple(sorted((int(r), int(c)) for r, c in comp["coords"])),
    }


def _frame_from_component(comp: dict) -> dict:
    r0, c0, r1, c1 = comp["bbox"]
    return {
        "color": int(comp["color"]),
        "bbox": (r0, c0, r1, c1),
        "origin": (r0 + 1, c0 + 1),
        "shape": (r1 - r0 - 1, c1 - c0 - 1),
        "area": (r1 - r0 - 1) * (c1 - c0 - 1),
    }


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    comps = _foreground_components(grid)
    frames: list[dict] = []
    pieces: list[dict] = []
    for comp in comps:
        if _is_hollow_rectangle(grid, comp):
            frames.append(_frame_from_component(comp))
        else:
            pieces.append(_piece_from_component(comp))

    assert len(frames) == 2, "expected exactly two hollow rectangular frames"
    assert len(pieces) >= 4, "expected several scattered pieces"
    assert len({p["color"] for p in pieces}) == len(pieces), "piece colors must be unique"
    frames.sort(key=lambda frame: frame["bbox"])
    return frames, pieces


def _piece_placements(piece: dict, frame: dict) -> list[tuple[int, int, int]]:
    height, width = frame["shape"]
    if piece["h"] > height or piece["w"] > width:
        return []

    placements: list[tuple[int, int, int]] = []
    seen: set[int] = set()
    for pr in range(height - piece["h"] + 1):
        for pc in range(width - piece["w"] + 1):
            bits = 0
            for lr, lc in piece["cells"]:
                bits |= 1 << ((pr + lr) * width + pc + lc)
            if bits not in seen:
                seen.add(bits)
                placements.append((bits, pr, pc))
    return placements


def _subset_sum_possible(areas: list[int], target: int) -> bool:
    if target < 0:
        return False
    sums = {0}
    for area in areas:
        sums |= {s + int(area) for s in tuple(sums)}
    return int(target) in sums


def _solve_tiling(pieces: list[dict], frames: list[dict], max_solutions: int = 2) -> list[tuple[int, int, int, int]]:
    assert len(frames) == 2
    frame_areas = [int(frame["area"]) for frame in frames]
    assert sum(piece["area"] for piece in pieces) == sum(frame_areas), "piece areas must match both interiors"

    full_masks = [(1 << area) - 1 for area in frame_areas]
    all_options: list[list[tuple[int, int, int, int]]] = []
    for piece in pieces:
        opts: list[tuple[int, int, int, int]] = []
        for frame_idx, frame in enumerate(frames):
            for bits, pr, pc in _piece_placements(piece, frame):
                opts.append((frame_idx, bits, pr, pc))
        assert opts, "every piece must fit at least one frame"
        all_options.append(opts)

    solutions: list[list[tuple[int, int, int, int]]] = []
    assigned: list[tuple[int, int, int, int] | None] = [None] * len(pieces)

    def rec(unplaced: tuple[int, ...], occ0: int, occ1: int) -> None:
        if len(solutions) >= max_solutions:
            return

        free0 = frame_areas[0] - occ0.bit_count()
        free1 = frame_areas[1] - occ1.bit_count()
        if free0 < 0 or free1 < 0:
            return

        remaining_areas = [int(pieces[i]["area"]) for i in unplaced]
        if sum(remaining_areas) != free0 + free1:
            return
        if not _subset_sum_possible(remaining_areas, free0):
            return

        if not unplaced:
            if occ0 == full_masks[0] and occ1 == full_masks[1]:
                assert all(item is not None for item in assigned)
                solutions.append([item for item in assigned if item is not None])
            return

        best_piece = -1
        best_feasible: list[tuple[int, int, int, int]] | None = None
        best_score: tuple[int, int] | None = None
        for piece_idx in unplaced:
            feasible: list[tuple[int, int, int, int]] = []
            for frame_idx, bits, pr, pc in all_options[piece_idx]:
                occupied = occ0 if frame_idx == 0 else occ1
                if bits & occupied:
                    continue
                feasible.append((frame_idx, bits, pr, pc))
            if not feasible:
                return
            score = (len(feasible), -int(pieces[piece_idx]["area"]))
            if best_score is None or score < best_score:
                best_score = score
                best_piece = piece_idx
                best_feasible = feasible

        assert best_feasible is not None
        next_unplaced = tuple(i for i in unplaced if i != best_piece)
        for frame_idx, bits, pr, pc in best_feasible:
            assigned[best_piece] = (frame_idx, bits, pr, pc)
            if frame_idx == 0:
                new_occ0, new_occ1 = occ0 | bits, occ1
            else:
                new_occ0, new_occ1 = occ0, occ1 | bits

            next_areas = [int(pieces[i]["area"]) for i in next_unplaced]
            next_free0 = frame_areas[0] - new_occ0.bit_count()
            if _subset_sum_possible(next_areas, next_free0):
                rec(next_unplaced, new_occ0, new_occ1)
            assigned[best_piece] = None
            if len(solutions) >= max_solutions:
                return

    rec(tuple(range(len(pieces))), 0, 0)
    assert len(solutions) == 1, "the two-frame packing must have one global solution"
    return solutions[0]


def _paint_solution(grid: np.ndarray, frames: list[dict], pieces: list[dict], solution: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = grid.copy()
    for piece in pieces:
        for r, c in piece["source"]:
            out[r, c] = BLACK

    for piece, (frame_idx, _bits, pr, pc) in zip(pieces, solution):
        r0, c0 = frames[frame_idx]["origin"]
        for lr, lc in piece["cells"]:
            out[r0 + pr + lr, c0 + pc + lc] = int(piece["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate scattered monochrome pieces into two hollow frames so both interiors are exactly tiled.

    Input structure:
    - A black background contains exactly two one-cell-thick hollow rectangular frames with black interiors.
    - Outside the frames are several connected monochrome sprite pieces, all with distinct colors.
    - The pieces admit one globally unique assignment and translation-only tiling of the two frame interiors.

    Rule:
    - Erase every original outside piece position to black.
    - For the unique global solution, translate each piece into its assigned frame without rotation or overlap.
    - Preserve both frame borders and all untouched background cells.

    Key insight:
    - The frames must be solved together: area-plausible local choices exist, but only one joint assignment fills both interiors exactly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    frames, pieces = _parse_scene(g)
    solution = _solve_tiling(pieces, frames, max_solutions=2)
    return _paint_solution(g, frames, pieces, solution)


# ------------------------------ generator ------------------------------

GEN_ATTEMPTS = 700
PARTITION_ATTEMPTS = 220
SUBSET_ATTEMPTS = 260
MIN_CANVAS = 22
MAX_CANVAS = 30
MIN_PIECE_AREA = 3

_GEN_CALL_COUNT = 0


def _neighbors_in_mask(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            out.append((nr, nc))
    return out


def _is_connected_mask(mask: np.ndarray) -> bool:
    return int(mask.sum()) > 0 and len(_components_bool(mask)) == 1


def _random_connected_subset(mask: np.ndarray, target: int) -> np.ndarray | None:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not 0 < target < len(cells):
        return mask.copy() if target == len(cells) else None

    for _ in range(SUBSET_ATTEMPTS):
        seed = random.choice(cells)
        selected = {seed}
        frontier = set(_neighbors_in_mask(mask, *seed))
        while len(selected) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            selected.add(cell)
            for nb in _neighbors_in_mask(mask, *cell):
                if nb not in selected:
                    frontier.add(nb)
        if len(selected) != target:
            continue

        sub = np.zeros_like(mask, dtype=bool)
        for r, c in selected:
            sub[r, c] = True
        rest = mask & ~sub
        if _is_connected_mask(sub) and _is_connected_mask(rest):
            return sub
    return None


def _partition_mask(mask: np.ndarray, areas: list[int]) -> list[np.ndarray] | None:
    if len(areas) == 1:
        return [mask.copy()] if int(mask.sum()) == int(areas[0]) and _is_connected_mask(mask) else None

    first = int(areas[0])
    if first <= 0 or first >= int(mask.sum()):
        return None

    for _ in range(PARTITION_ATTEMPTS):
        sub = _random_connected_subset(mask, first)
        if sub is None:
            continue
        rest = mask & ~sub
        tail = _partition_mask(rest, areas[1:])
        if tail is not None:
            return [sub] + tail
    return None


def _random_composition(total: int, parts: int, min_value: int) -> list[int] | None:
    if parts == 0:
        return [] if total == 0 else None
    if total < parts * min_value:
        return None
    values = [min_value] * parts
    for _ in range(total - parts * min_value):
        values[random.randrange(parts)] += 1
    random.shuffle(values)
    return values


def _sample_area_lists(area0: int, area1: int, count0: int, count1: int) -> tuple[list[int], list[int]] | None:
    max_shared = min(area0 - MIN_PIECE_AREA * (count0 - 1), area1 - MIN_PIECE_AREA * (count1 - 1), 12)
    if max_shared < 4:
        return None
    shared = random.randint(4, max_shared)
    rest0 = _random_composition(area0 - shared, count0 - 1, MIN_PIECE_AREA)
    rest1 = _random_composition(area1 - shared, count1 - 1, MIN_PIECE_AREA)
    if rest0 is None or rest1 is None:
        return None
    return [shared] + rest0, [shared] + rest1


def _piece_from_tile(mask: np.ndarray, color: int, group: int) -> dict:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    cells = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in coords))
    true_cells = tuple(sorted((int(r), int(c)) for r, c in coords))
    return {
        "group": group,
        "color": int(color),
        "cells": cells,
        "true_cells": true_cells,
        "true_place": (r0, c0),
        "h": r1 - r0 + 1,
        "w": c1 - c0 + 1,
        "area": int(coords.shape[0]),
        "source": (),
    }


def _solid_rect_piece(piece: dict) -> bool:
    return int(piece["area"]) == int(piece["h"]) * int(piece["w"])


def _dummy_frame(height: int, width: int) -> dict:
    return {
        "color": FRAME_COLOR,
        "bbox": (0, 0, height + 1, width + 1),
        "origin": (1, 1),
        "shape": (height, width),
        "area": height * width,
    }


def _has_wrong_area_subset(pieces: list[dict], frames: list[dict]) -> bool:
    target = int(frames[0]["area"])
    true_group0 = frozenset(i for i, piece in enumerate(pieces) if int(piece["group"]) == 0)
    n = len(pieces)
    indices = range(n)
    for size in range(1, n):
        for subset in combinations(indices, size):
            if frozenset(subset) == true_group0:
                continue
            if sum(int(pieces[i]["area"]) for i in subset) == target:
                return True
    return False


def _design_is_semantic(pieces: list[dict], frames: list[dict]) -> bool:
    if len([p for p in pieces if not _solid_rect_piece(p)]) < 2:
        return False
    if not _has_wrong_area_subset(pieces, frames):
        return False

    cross_groups: set[int] = set()
    for piece in pieces:
        fits = [bool(_piece_placements(piece, frame)) for frame in frames]
        if all(fits):
            cross_groups.add(int(piece["group"]))
    if cross_groups != {0, 1}:
        return False

    try:
        solution = _solve_tiling(pieces, frames, max_solutions=2)
    except AssertionError:
        return False

    for piece, (frame_idx, _bits, pr, pc) in zip(pieces, solution):
        if int(frame_idx) != int(piece["group"]):
            return False
        if (int(pr), int(pc)) != tuple(map(int, piece["true_place"])):
            return False
    return True


def _sample_dimensions() -> tuple[int, int, int, int]:
    dims = [(h, w) for h in range(4, 8) for w in range(4, 8)]
    while True:
        h0, w0 = random.choice(dims)
        h1, w1 = random.choice(dims)
        if (h0, w0) != (h1, w1) and abs(h0 * w0 - h1 * w1) <= 12:
            return h0, w0, h1, w1


def _build_design(count0: int, count1: int) -> tuple[list[dict], list[dict]] | None:
    h0, w0, h1, w1 = _sample_dimensions()
    frames = [_dummy_frame(h0, w0), _dummy_frame(h1, w1)]
    sampled = _sample_area_lists(h0 * w0, h1 * w1, count0, count1)
    if sampled is None:
        return None
    areas0, areas1 = sampled

    part0 = _partition_mask(np.ones((h0, w0), dtype=bool), areas0)
    part1 = _partition_mask(np.ones((h1, w1), dtype=bool), areas1)
    if part0 is None or part1 is None:
        return None

    colors = random.sample(list(PIECE_COLORS), count0 + count1)
    pieces: list[dict] = []
    for mask, color in zip(part0, colors[:count0]):
        pieces.append(_piece_from_tile(mask, color, 0))
    for mask, color in zip(part1, colors[count0:]):
        pieces.append(_piece_from_tile(mask, color, 1))

    random.shuffle(pieces)
    if not _design_is_semantic(pieces, frames):
        return None
    return pieces, frames


def _random_free_rect(reserved: np.ndarray, height: int, width: int, pad: int) -> tuple[int, int] | None:
    h, w = reserved.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h - height + 1):
        for c in range(w - width + 1):
            r0 = max(0, r - pad)
            c0 = max(0, c - pad)
            r1 = min(h, r + height + pad)
            c1 = min(w, c + width + pad)
            if not bool(reserved[r0:r1, c0:c1].any()):
                candidates.append((r, c))
    return random.choice(candidates) if candidates else None


def _reserve_rect(reserved: np.ndarray, r: int, c: int, height: int, width: int, pad: int) -> None:
    h, w = reserved.shape
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(h, r + height + pad)
    c1 = min(w, c + width + pad)
    reserved[r0:r1, c0:c1] = True


def _draw_frame(grid: np.ndarray, r: int, c: int, height: int, width: int, color: int) -> dict:
    grid[r, c : c + width + 2] = color
    grid[r + height + 1, c : c + width + 2] = color
    grid[r : r + height + 2, c] = color
    grid[r : r + height + 2, c + width + 1] = color
    return {
        "color": color,
        "bbox": (r, c, r + height + 1, c + width + 1),
        "origin": (r + 1, c + 1),
        "shape": (height, width),
        "area": height * width,
    }


def _place_scene(pieces: list[dict], design_frames: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    rows = random.randint(MIN_CANVAS, MAX_CANVAS)
    cols = random.randint(MIN_CANVAS, MAX_CANVAS)
    grid = np.full((rows, cols), BLACK, dtype=int)
    reserved = np.zeros_like(grid, dtype=bool)

    actual_frames: list[dict] = []
    for design_frame in design_frames:
        height, width = design_frame["shape"]
        loc = _random_free_rect(reserved, height + 2, width + 2, pad=2)
        if loc is None:
            return None
        r, c = loc
        actual_frames.append(_draw_frame(grid, r, c, height, width, FRAME_COLOR))
        _reserve_rect(reserved, r, c, height + 2, width + 2, pad=2)

    placed: list[dict] = []
    for piece in random.sample(pieces, len(pieces)):
        loc = _random_free_rect(reserved, int(piece["h"]), int(piece["w"]), pad=1)
        if loc is None:
            return None
        r0, c0 = loc
        source: list[tuple[int, int]] = []
        for lr, lc in piece["cells"]:
            rr, cc = r0 + int(lr), c0 + int(lc)
            grid[rr, cc] = int(piece["color"])
            source.append((rr, cc))
        copied = dict(piece)
        copied["source"] = tuple(sorted(source))
        placed.append(copied)
        _reserve_rect(reserved, r0, c0, int(piece["h"]), int(piece["w"]), pad=1)

    expected = grid.copy()
    for piece in placed:
        for r, c in piece["source"]:
            expected[r, c] = BLACK
    for piece in placed:
        frame = actual_frames[int(piece["group"])]
        r0, c0 = frame["origin"]
        for lr, lc in piece["true_cells"]:
            expected[r0 + int(lr), c0 + int(lc)] = int(piece["color"])

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid, expected


def _gen_attempt(count0: int, count1: int) -> np.ndarray | None:
    design = _build_design(count0, count1)
    if design is None:
        return None
    pieces, frames = design
    placed = _place_scene(pieces, frames)
    if placed is None:
        return None
    grid, expected = placed
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    modes = ((2, 2), (2, 3), (3, 2), (3, 3))
    preferred = modes[_GEN_CALL_COUNT % len(modes)]
    _GEN_CALL_COUNT += 1

    for attempt in range(GEN_ATTEMPTS):
        if attempt < GEN_ATTEMPTS // 3:
            count0, count1 = preferred
        else:
            count0, count1 = random.choice(modes)
        grid = _gen_attempt(count0, count1)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-frame tiling input")
