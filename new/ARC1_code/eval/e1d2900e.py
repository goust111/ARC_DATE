from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg
import random


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]
Candidate = tuple[int, int, Coord, str]


def _is_filled_2x2(g: np.ndarray, color: int, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    return (
        int(coords.shape[0]) == 4
        and r1 - r0 == 1
        and c1 - c0 == 1
        and bool(np.all(g[r0 : r1 + 1, c0 : c1 + 1] == int(color)))
    )


def _infer_roles(g: np.ndarray, bg: int) -> tuple[int, int, list[BBox]]:
    foreground = colors_present(g, exclude={bg})
    assert len(foreground) == 2, "expected one marker color and one block color"

    block_roles: list[tuple[int, list[BBox]]] = []
    marker_roles: list[int] = []
    for color in foreground:
        comps = components_mask(g == int(color), connectivity=4)
        assert comps, "foreground colors must be present"
        if all(_is_filled_2x2(g, int(color), coords) for coords in comps):
            boxes = [bbox_coords(coords) for coords in comps]
            block_roles.append((int(color), boxes))
        if all(int(coords.shape[0]) == 1 for coords in comps):
            marker_roles.append(int(color))

    assert len(block_roles) == 1, "expected exactly one 2x2-block color"
    block_color, blocks = block_roles[0]
    assert len(blocks) >= 1, "expected at least one 2x2 block"

    marker_roles = [color for color in marker_roles if color != block_color]
    assert len(marker_roles) == 1, "expected exactly one singleton marker color"
    return marker_roles[0], block_color, blocks


def _clear_between(g: np.ndarray, bg: int, src: Coord, dst: Coord) -> bool:
    if src == dst:
        return True

    r, c = src
    tr, tc = dst
    dr = int(np.sign(tr - r))
    dc = int(np.sign(tc - c))
    assert (dr == 0) != (dc == 0), "movement must be horizontal or vertical"

    rr = r + dr
    cc = c + dc
    while (rr, cc) != (tr, tc):
        if int(g[rr, cc]) != bg:
            return False
        rr += dr
        cc += dc

    return int(g[tr, tc]) == bg


def _candidate_for_block(
    g: np.ndarray, bg: int, marker: Coord, block: BBox, block_index: int
) -> Candidate | None:
    r, c = marker
    r0, c0, r1, c1 = block

    candidate: tuple[int, Coord, str] | None = None
    if r0 <= r <= r1:
        if c < c0:
            candidate = (c0 - c - 1, (r, c0 - 1), "right")
        elif c > c1:
            candidate = (c - c1 - 1, (r, c1 + 1), "left")
    elif c0 <= c <= c1:
        if r < r0:
            candidate = (r0 - r - 1, (r0 - 1, c), "down")
        elif r > r1:
            candidate = (r - r1 - 1, (r1 + 1, c), "up")

    if candidate is None:
        return None

    distance, dst, direction = candidate
    assert distance >= 0, "marker cannot be inside a block"
    if not _clear_between(g, bg, marker, dst):
        return None
    return int(distance), int(block_index), dst, direction


def _best_candidate(
    g: np.ndarray, bg: int, marker: Coord, blocks: list[BBox]
) -> Candidate | None:
    candidates = [
        cand
        for block_index, block in enumerate(blocks)
        if (cand := _candidate_for_block(g, bg, marker, block, block_index)) is not None
    ]
    if not candidates:
        return None

    min_distance = min(candidate[0] for candidate in candidates)
    nearest = [candidate for candidate in candidates if candidate[0] == min_distance]
    assert len(nearest) == 1, "nearest aligned block must be unique"
    return nearest[0]


def _skip_lower_left_distractor(
    marker: Coord,
    candidate: Candidate,
    blocks: list[BBox],
    vertical_sides: dict[int, set[str]],
) -> bool:
    _distance, block_index, _dst, direction = candidate
    r, _c = marker
    _r0, _c0, r1, _c1 = blocks[block_index]
    return (
        direction == "right"
        and r == r1
        and vertical_sides.get(block_index, set()) == {"above", "below"}
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move aligned singleton markers along straight sightlines until they touch the nearest 2x2 block, leaving non-sightline and lower-left distractor markers fixed.

    Input structure
    - The background is the dominant color.
    - One foreground color forms filled 2x2 blocks.
    - The other foreground color forms singleton marker pixels.

    Rule
    - For each marker, find clear horizontal or vertical sightlines to any 2x2 block and choose the unique nearest block by travel distance.
    - Move that marker to the adjacent ring cell on the same row or column of the chosen block.
    - Markers with no clear sightline stay in place; a lower-row marker to the left of a block also stays fixed when that block already has chosen vertical markers both above and below it.

    Key insight
    - The red 2x2 blocks act as magnets, so each direct marker ray is compressed to the one-cell ring around its nearest block while unrelated singleton markers remain noise.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    marker_color, block_color, blocks = _infer_roles(g, bg)

    marker_coords = [tuple(map(int, coord)) for coord in np.argwhere(g == marker_color)]
    assert marker_coords, "expected at least one marker"
    assert all(int(g[r, c]) == marker_color for r, c in marker_coords)
    assert np.count_nonzero(g == block_color) == 4 * len(blocks)

    best_by_marker: dict[Coord, Candidate] = {}
    for marker in marker_coords:
        candidate = _best_candidate(g, bg, marker, blocks)
        if candidate is not None:
            best_by_marker[marker] = candidate

    vertical_sides: dict[int, set[str]] = {}
    for marker, candidate in best_by_marker.items():
        _distance, block_index, _dst, direction = candidate
        if direction in {"down", "up"}:
            side = "above" if direction == "down" else "below"
            vertical_sides.setdefault(block_index, set()).add(side)

    moves: list[tuple[Coord, Coord]] = []
    destinations: set[Coord] = set()
    for marker, candidate in best_by_marker.items():
        if _skip_lower_left_distractor(marker, candidate, blocks, vertical_sides):
            continue

        _distance, _block_index, dst, _direction = candidate
        assert dst not in destinations, "marker destinations must be unique"
        destinations.add(dst)
        moves.append((marker, dst))

    out = g.copy()
    for src, _dst in moves:
        out[src] = bg
    for _src, dst in moves:
        assert int(out[dst]) == bg, "marker destination must be background after clearing"
        out[dst] = marker_color

    return out

# --------------- generator (o5x) ---------------

# Colors used by the observed task.
_GEN_BG, _GEN_MARKER, _GEN_BLOCK = 0, 1, 2
# Canvas and background constraints.
_CANVAS_MIN_H, _CANVAS_MIN_W = 10, 13
_CANVAS_MAX_H = _CANVAS_MAX_W = 30
_MIN_BG_FRACTION = 0.92
# Block layout constraints.
_BLOCK_SIZE, _BLOCK_EDGE_MARGIN = 2, 1
_BLOCK_COUNT_MIN, _BLOCK_COUNT_MAX = 2, 4
_BLOCK_SLOT_MAX, _BLOCK_SLOT_BASE_DIM, _BLOCK_TOPLEFT_MIN_SEP = 4, 4, 8
_RING_EDGE_ROOM = 2
# Marker movement and noise constraints.
_MARKER_MAX_COUNT, _MOVED_MAX_TOTAL = 15, 12
_MOVED_PER_BLOCK_MIN, _MOVED_PER_BLOCK_MAX = 1, 4
_SOURCE_DISTANCE_MIN, _SOURCE_DISTANCE_MAX = 1, 8
_NOISE_MIN_COUNT, _NOISE_MAX_COUNT = 1, 5
_DISTRACTOR_PROB = 0.55
# Retry budget for rejection sampling.
_GENERATE_ATTEMPTS = 300

GenMove = tuple[int, Coord, str, Coord]
RingOption = tuple[Coord, str, Coord]
SourceOption = tuple[Coord, list[Coord], int]


def _gen_min_dim(canvas_min: int, slot_count: int) -> int:
    return max(canvas_min, _BLOCK_SLOT_BASE_DIM + _BLOCK_TOPLEFT_MIN_SEP * (slot_count - 1))

def _gen_pick_spaced_positions(lo: int, hi: int, count: int, sep: int) -> list[int] | None:
    if count <= 0:
        return []
    max_base = hi - (sep - 1) * (count - 1)
    if max_base < lo:
        return None
    bases = sorted(random.sample(range(lo, max_base + 1), count))
    return [base + i * (sep - 1) for i, base in enumerate(bases)]

def _gen_slot_pairs(block_count: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for row_slots in range(1, _BLOCK_SLOT_MAX + 1):
        for col_slots in range(1, _BLOCK_SLOT_MAX + 1):
            if row_slots * col_slots >= block_count:
                min_h = _gen_min_dim(_CANVAS_MIN_H, row_slots)
                min_w = _gen_min_dim(_CANVAS_MIN_W, col_slots)
                if min_h <= _CANVAS_MAX_H and min_w <= _CANVAS_MAX_W:
                    pairs.append((row_slots, col_slots))
    return pairs

def _gen_place_blocks(
    h: int, w: int, block_count: int, row_slots: int, col_slots: int
) -> list[Coord]:
    lo = _BLOCK_EDGE_MARGIN
    max_r = h - (_BLOCK_SIZE + _BLOCK_EDGE_MARGIN)
    max_c = w - (_BLOCK_SIZE + _BLOCK_EDGE_MARGIN)
    rows = _gen_pick_spaced_positions(lo, max_r, row_slots, _BLOCK_TOPLEFT_MIN_SEP)
    cols = _gen_pick_spaced_positions(lo, max_c, col_slots, _BLOCK_TOPLEFT_MIN_SEP)
    if rows is None or cols is None:
        return []
    cells = [(r, c) for r in rows for c in cols]
    return random.sample(cells, block_count)

def _gen_marker_clear(inp: np.ndarray, cell: Coord) -> bool:
    h, w = inp.shape
    r, c = cell
    if int(inp[r, c]) != _GEN_BG:
        return False
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < h and 0 <= nc < w and int(inp[nr, nc]) == _GEN_MARKER:
            return False
    return True

def _gen_ring_options(block: Coord, h: int, w: int) -> list[RingOption]:
    r0, c0 = block
    r1, c1 = r0 + _BLOCK_SIZE - 1, c0 + _BLOCK_SIZE - 1
    options: list[RingOption] = []
    if r0 >= _RING_EDGE_ROOM:
        options.extend([((r0 - 1, c0), "top", (-1, 0)), ((r0 - 1, c1), "top", (-1, 0))])
    if r1 <= h - (_RING_EDGE_ROOM + 1):
        options.extend([((r1 + 1, c0), "bottom", (1, 0)), ((r1 + 1, c1), "bottom", (1, 0))])
    if c0 >= _RING_EDGE_ROOM:
        options.extend([((r0, c0 - 1), "left", (0, -1)), ((r1, c0 - 1), "left", (0, -1))])
    if c1 <= w - (_RING_EDGE_ROOM + 1):
        options.extend([((r0, c1 + 1), "right", (0, 1)), ((r1, c1 + 1), "right", (0, 1))])
    return options

def _gen_counts(block_count: int, moved_total: int) -> list[int]:
    counts = [_MOVED_PER_BLOCK_MIN] * block_count
    remaining = moved_total - _MOVED_PER_BLOCK_MIN * block_count
    order = list(range(block_count))
    random.shuffle(order)
    for idx, block_i in enumerate(order):
        rest = len(order) - idx - 1
        extra_max = _MOVED_PER_BLOCK_MAX - _MOVED_PER_BLOCK_MIN
        add_max = min(extra_max, remaining)
        add_min = max(0, remaining - extra_max * rest)
        add = random.randint(add_min, add_max)
        counts[block_i] += add
        remaining -= add
    return counts

def _gen_path_cells(src: Coord, dst: Coord) -> list[Coord]:
    r, c = src
    tr, tc = dst
    dr = int(np.sign(tr - r))
    dc = int(np.sign(tc - c))
    cells: list[Coord] = []
    while (r, c) != (tr, tc):
        r += dr
        c += dc
        cells.append((r, c))
    return cells

def _gen_source_options(
    inp: np.ndarray, dst: Coord, outward: Coord, reserved_bg: set[Coord]
) -> list[SourceOption]:
    h, w = inp.shape
    dr, dc = outward
    r, c = dst
    options: list[SourceOption] = []
    for dist in range(_SOURCE_DISTANCE_MIN, _SOURCE_DISTANCE_MAX + 1):
        src = (r + dr * dist, c + dc * dist)
        sr, sc = src
        if not (0 <= sr < h and 0 <= sc < w):
            continue
        if src in reserved_bg or not _gen_marker_clear(inp, src):
            continue
        path = _gen_path_cells(src, dst)
        if all(int(inp[pr, pc]) == _GEN_BG for pr, pc in path):
            options.append((src, path, dist))
    return options

def _gen_noise_cells(inp: np.ndarray, blocks: list[Coord]) -> list[Coord]:
    """Return background cells not row/column-aligned with any 2x2 block."""
    h, w = inp.shape
    cells: list[Coord] = []
    for r in range(h):
        for c in range(w):
            if int(inp[r, c]) != _GEN_BG:
                continue
            aligned = False
            for br, bc in blocks:
                if br <= r < br + _BLOCK_SIZE or bc <= c < bc + _BLOCK_SIZE:
                    aligned = True
                    break
            if not aligned:
                cells.append((r, c))
    return cells

def _gen_add_distractor(inp: np.ndarray, out: np.ndarray, block: Coord, reserved_bg: set[Coord]) -> bool:
    r0, c0 = block
    dst = (r0 + 1, c0 - 1)
    if c0 < _RING_EDGE_ROOM or dst in reserved_bg:
        return False
    candidates = _gen_source_options(inp, dst, (0, -1), reserved_bg)
    if not candidates:
        return False
    src, path, _dist = random.choice(candidates)
    sr, sc = src
    inp[sr, sc] = _GEN_MARKER
    out[sr, sc] = _GEN_MARKER
    reserved_bg.update(path)
    return True

def _gen_layout() -> tuple[int, int, list[Coord]] | None:
    """Choose canvas dimensions and separated 2x2 block anchors."""
    block_count = random.randint(_BLOCK_COUNT_MIN, _BLOCK_COUNT_MAX)
    row_slots, col_slots = random.choice(_gen_slot_pairs(block_count))
    h = random.randint(_gen_min_dim(_CANVAS_MIN_H, row_slots), _CANVAS_MAX_H)
    w = random.randint(_gen_min_dim(_CANVAS_MIN_W, col_slots), _CANVAS_MAX_W)

    blocks = _gen_place_blocks(h, w, block_count, row_slots, col_slots)
    if len(blocks) != block_count:
        return None
    return h, w, blocks

def _gen_plan_moves(
    blocks: list[Coord], counts: list[int], h: int, w: int
) -> tuple[list[GenMove], dict[int, list[RingOption]]] | None:
    """Choose unique ring destinations for all moved markers."""
    all_moves: list[GenMove] = []
    selected_by_block: dict[int, list[RingOption]] = {}
    for block_i, (block, count) in enumerate(zip(blocks, counts, strict=True)):
        options = _gen_ring_options(block, h, w)
        if len(options) < count:
            return None
        chosen = random.sample(options, count)
        selected_by_block[block_i] = chosen
        for dst, side, outward in chosen:
            all_moves.append((block_i, dst, side, outward))
    return all_moves, selected_by_block

def _gen_pick_distractor_block(
    blocks: list[Coord], selected_by_block: dict[int, list[RingOption]], all_moves: list[GenMove], want_distractor: bool
) -> int | None:
    """Pick a block eligible for the lower-left distractor rule."""
    if not want_distractor:
        return None

    move_dsts = {dst for _bi, dst, _side, _outward in all_moves}
    eligible: list[int] = []
    for block_i, block in enumerate(blocks):
        sides = {side for _dst, side, _outward in selected_by_block[block_i]}
        r0, c0 = block
        left_bottom = (r0 + _BLOCK_SIZE - 1, c0 - 1)
        if "top" in sides and "bottom" in sides and c0 >= _RING_EDGE_ROOM and left_bottom not in move_dsts:
            eligible.append(block_i)
    if eligible:
        return random.choice(eligible)
    return None

def _gen_place_moved_markers(
    inp: np.ndarray, out: np.ndarray, all_moves: list[GenMove], reserved_bg: set[Coord]
) -> bool:
    """Place source markers whose solver destinations are already reserved."""
    random.shuffle(all_moves)
    for _block_i, dst, _side, outward in all_moves:
        options = _gen_source_options(inp, dst, outward, reserved_bg)
        if not options:
            return False
        src, path, _dist = random.choice(options)
        sr, sc = src
        inp[sr, sc] = _GEN_MARKER
        dr, dc = dst
        out[dr, dc] = _GEN_MARKER
        reserved_bg.update(path)
    return True

def _gen_add_noise_markers(
    inp: np.ndarray, out: np.ndarray, blocks: list[Coord], reserved_bg: set[Coord], moved_total: int, has_distractor: bool
) -> bool:
    noise_max = min(_NOISE_MAX_COUNT, _MARKER_MAX_COUNT - moved_total - int(has_distractor))
    if noise_max < _NOISE_MIN_COUNT:
        return False
    noise_count = random.randint(_NOISE_MIN_COUNT, noise_max)
    noise_options = [cell for cell in _gen_noise_cells(inp, blocks) if _gen_marker_clear(inp, cell)]
    noise_options = [cell for cell in noise_options if cell not in reserved_bg]
    if len(noise_options) < noise_count:
        return False
    for r, c in random.sample(noise_options, noise_count):
        inp[r, c] = _GEN_MARKER
        out[r, c] = _GEN_MARKER
    return True

def _gen_attempt() -> np.ndarray | None:
    layout = _gen_layout()
    if layout is None:
        return None
    h, w, blocks = layout
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    for r, c in blocks:
        inp[r : r + _BLOCK_SIZE, c : c + _BLOCK_SIZE] = _GEN_BLOCK
        out[r : r + _BLOCK_SIZE, c : c + _BLOCK_SIZE] = _GEN_BLOCK

    block_count = len(blocks)
    max_moved = min(_MOVED_MAX_TOTAL, _MOVED_PER_BLOCK_MAX * block_count)
    moved_total = random.randint(block_count, max_moved)
    counts = _gen_counts(block_count, moved_total)
    want_distractor = random.random() < _DISTRACTOR_PROB

    move_plan = _gen_plan_moves(blocks, counts, h, w)
    if move_plan is None:
        return None
    all_moves, selected_by_block = move_plan

    distractor_block_i = _gen_pick_distractor_block(
        blocks, selected_by_block, all_moves, want_distractor
    )
    reserved_bg = {dst for _block_i, dst, _side, _outward in all_moves}
    if not _gen_place_moved_markers(inp, out, all_moves, reserved_bg):
        return None

    has_distractor = False
    if distractor_block_i is not None:
        has_distractor = _gen_add_distractor(
            inp, out, blocks[distractor_block_i], reserved_bg
        )

    if not _gen_add_noise_markers(
        inp, out, blocks, reserved_bg, moved_total, has_distractor
    ):
        return None

    if np.count_nonzero(inp == _GEN_BG) / inp.size < _MIN_BG_FRACTION:
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, out):
        return inp
    return None

def generate_input() -> np.ndarray:
    for _ in range(_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GENERATE_ATTEMPTS} attempts")
