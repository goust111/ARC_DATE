from __future__ import annotations

import random

import numpy as np

BLACK = 0
ORANGE = 7
PALETTE_POOL = [1, 2, 3, 4, 5, 6, 8, 9]

GRID_H_RANGE = (16, 28)
GRID_W_RANGE = (18, 32)
FRAME_COUNT_RANGE = (2, 5)
OUTER_H_RANGE = (5, 11)
OUTER_W_RANGE = (5, 13)
PLACEMENT_TRIES = 120
GENERATION_RETRIES = 400

def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _find_palette(grid: np.ndarray) -> tuple[int, list[int]]:
    h, _w = grid.shape
    col = grid[:, 0]
    candidates: list[tuple[int, list[int]]] = []
    r = 0
    while r < h:
        value = int(col[r])
        if value not in (BLACK, ORANGE):
            start = r
            while r < h and int(col[r]) not in (BLACK, ORANGE):
                r += 1
            run = [int(v) for v in col[start:r]]
            if len(run) == 6 and len(set(run)) == 6:
                candidates.append((start, run))
        else:
            r += 1
    assert len(candidates) == 1, "expected exactly one 6-color palette on the left edge"
    return candidates[0]


def _frame_from_bbox(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> tuple[int, int, int, int, int, frozenset[tuple[int, int]]] | None:
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    perimeter = _perimeter_coords(r0, c0, r1, c1)
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != BLACK):
        return None

    if any(int(grid[r, c]) not in (BLACK, ORANGE) for r, c in perimeter):
        return None

    side_counts = [
        int(np.sum(grid[r0, c0 : c1 + 1] == ORANGE)),
        int(np.sum(grid[r1, c0 : c1 + 1] == ORANGE)),
        int(np.sum(grid[r0 : r1 + 1, c0] == ORANGE)),
        int(np.sum(grid[r0 : r1 + 1, c1] == ORANGE)),
    ]
    if min(side_counts) == 0:
        return None

    holes = sum(int(grid[r, c]) != ORANGE for r, c in perimeter)
    if holes > 5:
        return None
    orange_cells = frozenset((r, c) for r, c in perimeter if int(grid[r, c]) == ORANGE)
    return r0, c0, r1, c1, holes, orange_cells


def _detect_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    orange = np.argwhere(grid == ORANGE)
    orange_set = {(int(r), int(c)) for r, c in orange}
    assert orange_set, "expected at least one orange frame"

    rows_to_cols: dict[int, list[int]] = {}
    cols_to_rows: dict[int, list[int]] = {}
    for r, c in orange_set:
        rows_to_cols.setdefault(r, []).append(c)
        cols_to_rows.setdefault(c, []).append(r)
    for cols in rows_to_cols.values():
        cols.sort()
    for rows in cols_to_rows.values():
        rows.sort()

    candidates: dict[tuple[int, int, int, int], tuple[int, int, int, int, int, frozenset[tuple[int, int]]]] = {}
    for r0, c0 in sorted(orange_set):
        for c1 in rows_to_cols[r0]:
            if c1 <= c0 + 1:
                continue
            for r1 in cols_to_rows[c0]:
                if r1 <= r0 + 1 or (r1, c1) not in orange_set:
                    continue
                frame = _frame_from_bbox(grid, r0, c0, r1, c1)
                if frame is not None:
                    candidates[(r0, c0, r1, c1)] = frame

    ordered = sorted(
        candidates.values(),
        key=lambda item: (-len(item[5]), -((item[2] - item[0] + 1) * (item[3] - item[1] + 1))),
    )
    frames: list[tuple[int, int, int, int, int]] = []
    covered: set[tuple[int, int]] = set()
    for r0, c0, r1, c1, holes, orange_cells in ordered:
        if orange_cells & covered:
            continue
        frames.append((r0, c0, r1, c1, holes))
        covered |= set(orange_cells)

    assert covered == orange_set, "failed to recover every orange frame"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the holes on each orange rectangular frame and use that count to index
    the left-edge palette. Fill the frame interior with the mapped color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    _palette_start, palette = _find_palette(g)
    out = g.copy()
    frames = _detect_frames(g)
    for r0, c0, r1, c1, hole_count in frames:
        fill_color = int(palette[hole_count])
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BLACK] = fill_color
    assert frames
    return out


def _eligible_holes(outer_h: int, outer_w: int) -> list[tuple[str, int, tuple[int, int]]]:
    choices: list[tuple[str, int, tuple[int, int]]] = []
    for c in range(2, outer_w - 2):
        choices.append(("top", c, (0, c)))
        choices.append(("bottom", c, (outer_h - 1, c)))
    for r in range(2, outer_h - 2):
        choices.append(("left", r, (r, 0)))
        choices.append(("right", r, (r, outer_w - 1)))
    return choices


def _max_holes(outer_h: int, outer_w: int) -> int:
    top_bottom = 2 * ((max(0, outer_w - 4) + 1) // 2)
    left_right = 2 * ((max(0, outer_h - 4) + 1) // 2)
    return top_bottom + left_right


def _choose_holes(outer_h: int, outer_w: int, hole_count: int) -> list[tuple[int, int]] | None:
    if hole_count == 0:
        return []
    choices = _eligible_holes(outer_h, outer_w)
    if hole_count > len(choices) or hole_count > _max_holes(outer_h, outer_w):
        return None
    for _ in range(200):
        picked = random.sample(choices, hole_count)
        side_positions: dict[str, list[int]] = {}
        for side, index, _pos in picked:
            side_positions.setdefault(side, []).append(index)
        if any(
            any(b - a == 1 for a, b in zip(sorted(indices), sorted(indices)[1:], strict=False))
            for indices in side_positions.values()
        ):
            continue
        return [pos for _side, _index, pos in picked]
    return None


def _paint_frame(grid: np.ndarray, r0: int, c0: int, outer_h: int, outer_w: int, holes: list[tuple[int, int]]) -> None:
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    grid[r0, c0 : c1 + 1] = ORANGE
    grid[r1, c0 : c1 + 1] = ORANGE
    grid[r0 : r1 + 1, c0] = ORANGE
    grid[r0 : r1 + 1, c1] = ORANGE
    for dr, dc in holes:
        grid[r0 + dr, c0 + dc] = BLACK


def _expanded_bbox(r0: int, c0: int, r1: int, c1: int, h: int, w: int, pad: int = 1) -> tuple[int, int, int, int]:
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _sample_hole_counts(frame_count: int) -> list[int]:
    if random.random() < 0.6:
        return random.sample(list(range(6)), k=frame_count)
    return [random.randint(0, 5) for _ in range(frame_count)]


def _attempt_case() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    palette_start = random.randint(0, height - 6)
    palette = random.sample(PALETTE_POOL, 6)

    inp = np.full((height, width), BLACK, dtype=int)
    inp[palette_start : palette_start + 6, 0] = palette
    out = inp.copy()

    blocked = np.zeros((height, width), dtype=bool)
    blocked[:, 0:2] = True
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    hole_counts = _sample_hole_counts(frame_count)

    placed = 0
    for hole_count in hole_counts:
        for _ in range(PLACEMENT_TRIES):
            outer_h = random.randint(*OUTER_H_RANGE)
            outer_w = random.randint(*OUTER_W_RANGE)
            holes = _choose_holes(outer_h, outer_w, hole_count)
            if holes is None:
                continue

            if outer_h >= height - 2 or outer_w >= width - 3:
                continue
            r0 = random.randint(1, height - outer_h - 1)
            c0 = random.randint(2, width - outer_w - 1)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            er0, ec0, er1, ec1 = _expanded_bbox(r0, c0, r1, c1, height, width, pad=1)
            if blocked[er0 : er1 + 1, ec0 : ec1 + 1].any():
                continue

            _paint_frame(inp, r0, c0, outer_h, outer_w, holes)
            _paint_frame(out, r0, c0, outer_h, outer_w, holes)
            blocked[er0 : er1 + 1, ec0 : ec1 + 1] = True
            out[r0 + 1 : r1, c0 + 1 : c1] = int(palette[hole_count])
            placed += 1
            break
        else:
            return None

    if placed != frame_count:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        case = _attempt_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input for puzzle 3")
