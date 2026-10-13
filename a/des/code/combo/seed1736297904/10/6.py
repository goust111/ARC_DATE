import random
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
GREY = 5
MOTIF_COLORS = (1, 3, 4, 6, 7, 8, 9)

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
RING_OFFSETS = tuple(
    (dr, dc)
    for dr in range(-3, 4)
    for dc in range(-3, 4)
    if max(abs(dr), abs(dc)) == 3 and not (abs(dr) == 3 and abs(dc) == 3)
)
MARK_OFFSETS = ((-4, 0), (0, 4), (4, 0), (0, -4))
POINTER_ZONE = tuple((dr, dc) for dr in range(-2, 3) for dc in range(-2, 3))
MAX_ATTEMPTS = 300
GRID_SIDE_RANGE = (18, 35)
MAX_CLOCKS = 5
BOX_SIZE = 9
BOX_GAP = 1


def _clock_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(4, h - 4):
        for c in range(4, w - 4):
            if not all(int(grid[r + dr, c + dc]) == GREY for dr, dc in RING_OFFSETS):
                continue
            if not all(int(grid[r + dr, c + dc]) == GREY for dr, dc in MARK_OFFSETS):
                continue
            centers.append((r, c))
    return centers


def _pointer_turns(grid: np.ndarray, center_r: int, center_c: int) -> int:
    scores = []
    for dr, dc in DIRS:
        score = 0
        if int(grid[center_r + dr, center_c + dc]) == RED:
            score += 1
        if int(grid[center_r + 2 * dr, center_c + 2 * dc]) == RED:
            score += 2
        scores.append(score)

    best = max(scores)
    assert best > 0, "clock pointer missing"
    assert scores.count(best) == 1, "clock pointer direction ambiguous"
    turns = scores.index(best)

    allowed = {(0, 0), DIRS[turns], (2 * DIRS[turns][0], 2 * DIRS[turns][1])}
    for dr, dc in POINTER_ZONE:
        if int(grid[center_r + dr, center_c + dc]) == RED:
            assert (dr, dc) in allowed, "unexpected red pixel inside clock"
    return turns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each clock hand as a clockwise rotation instruction for the central 3x3 motif.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    centers = _clock_centers(g)
    assert centers, "no clock detected"

    out = g.copy()
    for center_r, center_c in centers:
        turns = _pointer_turns(g, center_r, center_c)
        motif = g[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2].copy()
        motif[motif == RED] = BLACK
        rotated = np.rot90(motif, -turns)

        for dr, dc in POINTER_ZONE:
            if int(out[center_r + dr, center_c + dc]) == RED:
                out[center_r + dr, center_c + dc] = BLACK

        out[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2] = rotated

    return out


def _paint_clock(
    canvas: np.ndarray,
    center_r: int,
    center_c: int,
    motif: np.ndarray,
    turns: int,
    *,
    include_pointer: bool,
) -> None:
    for dr, dc in RING_OFFSETS:
        canvas[center_r + dr, center_c + dc] = GREY
    for dr, dc in MARK_OFFSETS:
        canvas[center_r + dr, center_c + dc] = GREY
    canvas[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2] = motif
    if include_pointer:
        dr, dc = DIRS[turns]
        canvas[center_r + dr, center_c + dc] = RED
        canvas[center_r + 2 * dr, center_c + 2 * dc] = RED


def _balanced_turns(count: int) -> list[int]:
    seq: list[int] = []
    while len(seq) < count:
        batch = [0, 1, 2, 3]
        random.shuffle(batch)
        seq.extend(batch)
    seq = seq[:count]
    random.shuffle(seq)
    return seq


def _motif_edge_cell(turns: int) -> tuple[int, int]:
    dr, dc = DIRS[turns]
    return 1 + dr, 1 + dc


def _random_motif(turns: int) -> np.ndarray | None:
    blocked = _motif_edge_cell(turns)
    positions = [(r, c) for r in range(3) for c in range(3) if (r, c) != blocked]

    for _ in range(60):
        motif = np.zeros((3, 3), dtype=int)
        color_count = random.randint(1, 2)
        colors = random.sample(MOTIF_COLORS, color_count)

        colored_total = random.randint(4, min(8, len(positions)))
        colored_positions = {(1, 1)}
        extras = [pos for pos in positions if pos != (1, 1)]
        colored_positions.update(random.sample(extras, colored_total - 1))

        if color_count == 1:
            for r, c in colored_positions:
                motif[r, c] = colors[0]
        else:
            assignments = [colors[0], colors[1]]
            assignments.extend(random.choice(colors) for _ in range(len(colored_positions) - 2))
            random.shuffle(assignments)
            for (r, c), color in zip(colored_positions, assignments):
                motif[r, c] = color

        if tuple(sorted(np.unique(motif[motif != BLACK]).tolist())) != tuple(sorted(colors)):
            continue
        if turns != 0 and np.array_equal(motif, np.rot90(motif, -turns)):
            continue
        if np.count_nonzero(motif == BLACK) < 2:
            continue
        return motif
    return None


def _boxes_with_gap(boxes: Iterable[tuple[int, int, int, int]], box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    for er0, ec0, er1, ec1 in boxes:
        if not (r1 + BOX_GAP < er0 or er1 + BOX_GAP < r0 or c1 + BOX_GAP < ec0 or ec1 + BOX_GAP < c0):
            return False
    return True


def _place_boxes(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    coords = [(r, c) for r in range(height - BOX_SIZE + 1) for c in range(width - BOX_SIZE + 1)]
    random.shuffle(coords)
    boxes: list[tuple[int, int, int, int]] = []
    placements: list[tuple[int, int]] = []
    for top, left in coords:
        box = (top, left, top + BOX_SIZE - 1, left + BOX_SIZE - 1)
        if not _boxes_with_gap(boxes, box):
            continue
        boxes.append(box)
        placements.append((top + 4, left + 4))
        if len(placements) == count:
            return placements
    return None


def _render_pair(height: int, width: int, specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)
    for spec in specs:
        center_r, center_c = spec["center"]
        motif = spec["motif"]
        turns = spec["turns"]
        _paint_clock(inp, center_r, center_c, motif, turns, include_pointer=True)
        _paint_clock(out, center_r, center_c, np.rot90(motif, -turns), turns, include_pointer=False)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    approx_capacity = max(1, (height * width) // 110)
    clock_count = random.randint(1, min(MAX_CLOCKS, approx_capacity))

    centers = _place_boxes(height, width, clock_count)
    if centers is None:
        return None

    turns_list = _balanced_turns(clock_count)
    specs = []
    for center, turns in zip(centers, turns_list):
        motif = _random_motif(turns)
        if motif is None:
            return None
        specs.append({"center": center, "motif": motif, "turns": turns})

    inp, expected = _render_pair(height, width, specs)
    if not np.array_equal(main(inp), expected):
        return None
    if not np.any(inp != expected):
        return None
    if len(_clock_centers(inp)) != clock_count:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid clock-rotation input")
