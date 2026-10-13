from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.is_dir():
    utils_path = str(_UTILS_DIR)
    if utils_path not in sys.path:
        sys.path.append(utils_path)

try:
    from arc_utils import Color, in_bounds
except Exception:
    class Color:
        BLACK = 0
        RED = 2
        GREY = 5

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        h, w = map(int, shape)
        return 0 <= int(r) < h and 0 <= int(c) < w


BLACK = int(Color.BLACK)
RED = int(Color.RED)
GREY = int(Color.GREY)

DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
SQUARE_OFFSETS = ((0, 0), (0, 1), (1, 0), (1, 1))
SEED_OFFSETS = {
    (-1, -1): (-1, -1),
    (-1, 1): (-1, 2),
    (1, -1): (2, -1),
    (1, 1): (2, 2),
}
OPPOSITE = {direction: (-direction[0], -direction[1]) for direction in DIRS}


@dataclass(frozen=True)
class Block:
    origin: tuple[int, int]
    color: int
    directions: tuple[tuple[int, int], ...]
    square_cells: tuple[tuple[int, int], ...]
    all_cells: tuple[tuple[int, int], ...]


@dataclass
class NodeSpec:
    origin: tuple[int, int]
    directions: set[tuple[int, int]]


def _seed_pos(origin: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    r, c = origin
    dr, dc = SEED_OFFSETS[direction]
    return r + dr, c + dc


def _square_cells(origin: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = origin
    return tuple((r + dr, c + dc) for dr, dc in SQUARE_OFFSETS)


def _block_cells(origin: tuple[int, int], directions: set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    cells = set(_square_cells(origin))
    for direction in directions:
        cells.add(_seed_pos(origin, direction))
    return tuple(sorted(cells))


def _paint_block(grid: np.ndarray, block: Block, color: int) -> None:
    value = int(color)
    for r, c in block.all_cells:
        grid[r, c] = value


def _build_block(origin: tuple[int, int], color: int, directions: set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> Block:
    ordered_dirs = tuple(direction for direction in DIRS if direction in directions)
    return Block(
        origin=origin,
        color=int(color),
        directions=ordered_dirs,
        square_cells=_square_cells(origin),
        all_cells=_block_cells(origin, ordered_dirs),
    )


def _ray_step_to_cells(
    origin: tuple[int, int],
    direction: tuple[int, int],
    cells: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> int | None:
    seed_r, seed_c = _seed_pos(origin, direction)
    dr, dc = direction
    best: int | None = None
    for r, c in cells:
        step = (r - seed_r) * dr
        if step < 1:
            continue
        if c != seed_c + dc * step:
            continue
        if r != seed_r + dr * step:
            continue
        if best is None or step < best:
            best = int(step)
    return best


def _first_hit_label(
    specs: list[NodeSpec],
    source_idx: int,
    direction: tuple[int, int],
    extra_specs: list[NodeSpec] | None = None,
) -> tuple[tuple[str, int] | None, bool]:
    best_label: tuple[str, int] | None = None
    best_step: int | None = None
    tie = False

    for idx, spec in enumerate(specs):
        if idx == source_idx:
            continue
        step = _ray_step_to_cells(specs[source_idx].origin, direction, _block_cells(spec.origin, spec.directions))
        if step is None:
            continue
        if best_step is None or step < best_step:
            best_label = ("spec", idx)
            best_step = step
            tie = False
        elif step == best_step:
            tie = True

    if extra_specs is None:
        return best_label, tie

    for idx, spec in enumerate(extra_specs):
        step = _ray_step_to_cells(specs[source_idx].origin, direction, _block_cells(spec.origin, spec.directions))
        if step is None:
            continue
        if best_step is None or step < best_step:
            best_label = ("extra", idx)
            best_step = step
            tie = False
        elif step == best_step:
            tie = True

    return best_label, tie


def _cells_compatible(a: tuple[tuple[int, int], ...], b: tuple[tuple[int, int], ...]) -> bool:
    for ra, ca in a:
        for rb, cb in b:
            if max(abs(ra - rb), abs(ca - cb)) <= 1:
                return False
    return True


def _candidate_fits(specs: list[NodeSpec], candidate: NodeSpec) -> bool:
    candidate_cells = _block_cells(candidate.origin, candidate.directions)
    return all(_cells_compatible(candidate_cells, _block_cells(spec.origin, spec.directions)) for spec in specs)


def _parse_blocks(grid: np.ndarray) -> tuple[list[Block], dict[tuple[int, int], int], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert min(g.shape) > 0, "expected a nonempty grid"
    assert set(map(int, np.unique(g))) <= {BLACK, RED, GREY}, "expected only black, red, and grey"

    height, width = map(int, g.shape)
    candidates: list[tuple[int, int, int]] = []
    for r in range(height - 1):
        for c in range(width - 1):
            color = int(g[r, c])
            if color not in {RED, GREY}:
                continue
            if np.all(g[r : r + 2, c : c + 2] == color):
                candidates.append((r, c, color))

    assert candidates, "expected at least one 2x2 square"

    blocks: list[Block] = []
    square_owner: dict[tuple[int, int], int] = {}
    for r, c, color in candidates:
        square_cells = _square_cells((r, c))
        for cell in square_cells:
            assert cell not in square_owner, "expected non-overlapping 2x2 squares"
        directions = {
            direction
            for direction in DIRS
            if in_bounds(g.shape, *_seed_pos((r, c), direction)) and int(g[_seed_pos((r, c), direction)]) == color
        }
        assert directions, "each square must have at least one diagonal seed"
        block = _build_block((r, c), color, directions)
        block_idx = len(blocks)
        for cell in square_cells:
            square_owner[cell] = block_idx
        blocks.append(block)

    occupied_by: dict[tuple[int, int], int] = {}
    for idx, block in enumerate(blocks):
        for r, c in block.all_cells:
            assert in_bounds(g.shape, r, c), "block cells must stay in bounds"
            assert (r, c) not in occupied_by, "blocks may not overlap or share seeds"
            assert int(g[r, c]) == block.color, "every parsed block cell must match its block color"
            occupied_by[(r, c)] = idx

    actual_fg = {tuple(map(int, coord)) for coord in np.argwhere(g != BLACK)}
    assert actual_fg == set(occupied_by), "foreground must consist only of squares and diagonal seeds"

    red_blocks = [idx for idx, block in enumerate(blocks) if block.color == RED]
    assert len(red_blocks) == 1, "expected exactly one red ignition source"
    return blocks, occupied_by, red_blocks[0]


def _solve_from_blocks(
    grid: np.ndarray,
    blocks: list[Block],
    occupied_by: dict[tuple[int, int], int],
    source_idx: int,
) -> tuple[np.ndarray, list[bool]]:
    out = np.asarray(grid, dtype=int).copy()
    ignited = [False] * len(blocks)
    ignited[source_idx] = True
    _paint_block(out, blocks[source_idx], RED)

    queue: deque[int] = deque([source_idx])
    while queue:
        idx = queue.popleft()
        block = blocks[idx]
        for direction in block.directions:
            seed_r, seed_c = _seed_pos(block.origin, direction)
            dr, dc = direction
            r = seed_r + dr
            c = seed_c + dc
            while in_bounds(out.shape, r, c):
                hit_idx = occupied_by.get((r, c))
                if hit_idx is not None:
                    if not ignited[hit_idx]:
                        ignited[hit_idx] = True
                        _paint_block(out, blocks[hit_idx], RED)
                        queue.append(hit_idx)
                    break
                out[r, c] = RED
                r += dr
                c += dc

    return out, ignited


def main(grid: np.ndarray) -> np.ndarray:
    """
    Ignite the grey square network reached by diagonal rays from the unique red source block.

    Input structure
    - The grid has a black background plus several 2x2 solid squares.
    - Some corners of each square have same-color singleton seeds one diagonal step outside the square.
    - Exactly one square with its seeds is red; every other square and seed is grey.

    Rule
    - Starting from the unique red square, cast a ray for every one of its seed directions, beginning one cell beyond the seed and painting traversed background cells red.
    - Rays pass through earlier red trail cells but stop at the first square or seed pixel they meet.
    - If that hit belongs to an unignited grey square unit, recolor that whole square and all of its seeds red, then cast rays from the newly ignited square's own seeds.
    - Continue until no new square is reached; unreached grey squares and seeds stay grey.

    Key insight
    - The diagonal seeds only mark potential outgoing directions, while the output depends on the propagation network actually reachable from the initial red source.
    """
    blocks, occupied_by, source_idx = _parse_blocks(np.asarray(grid, dtype=int))
    out, _ignited = _solve_from_blocks(np.asarray(grid, dtype=int), blocks, occupied_by, source_idx)
    return out


# --------------- generator (o5x) ---------------

GRID_SIZE_MIN = 14
GRID_SIZE_MAX = 30
REACHED_COUNT_RANGE = (2, 7)
DISTRACTOR_COUNT_CHOICES = (0, 1, 1, 2, 2, 3)
EDGE_DELTA_MIN = 4
EDGE_DELTA_MAX = 7
EXTRA_SEED_PROB = 0.35
DISTRACTOR_SPAN = 10
GEN_RETRY_LIMIT = 400


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _scale(direction: tuple[int, int], amount: int) -> tuple[int, int]:
    return direction[0] * int(amount), direction[1] * int(amount)


def _occupied_bbox(specs: list[NodeSpec]) -> tuple[int, int, int, int]:
    rows = []
    cols = []
    for spec in specs:
        for r, c in _block_cells(spec.origin, spec.directions):
            rows.append(r)
            cols.append(c)
    return min(rows), min(cols), max(rows), max(cols)


def _random_subset(rng: np.random.Generator, min_size: int = 1) -> set[tuple[int, int]]:
    order = list(DIRS)
    rng.shuffle(order)
    max_size = len(order)
    size = int(rng.integers(min_size, max_size + 1))
    return set(order[:size])


def _build_reachable_specs(rng: np.random.Generator, reached_count: int) -> list[NodeSpec] | None:
    specs = [NodeSpec((0, 0), set())]

    while len(specs) < reached_count:
        options: list[tuple[int, tuple[int, int], NodeSpec]] = []
        for parent_idx, parent in enumerate(specs):
            unused_dirs = [direction for direction in DIRS if direction not in parent.directions]
            for direction in unused_dirs:
                hit_seed_order = [False, True]
                rng.shuffle(hit_seed_order)
                for hit_seed in hit_seed_order:
                    min_delta = EDGE_DELTA_MIN + int(hit_seed)
                    deltas = list(range(min_delta, EDGE_DELTA_MAX + 1))
                    rng.shuffle(deltas)
                    for delta in deltas:
                        child_origin = _add(parent.origin, _scale(direction, delta))
                        child_dirs = {OPPOSITE[direction]} if hit_seed else set()
                        child = NodeSpec(child_origin, child_dirs)
                        if not _candidate_fits(specs, child):
                            continue
                        label, tie = _first_hit_label(specs, parent_idx, direction, extra_specs=[child])
                        if tie or label != ("extra", 0):
                            continue

                        unintended = False
                        for other_idx, other in enumerate(specs):
                            for other_direction in other.directions:
                                other_label, other_tie = _first_hit_label(specs, other_idx, other_direction, extra_specs=[child])
                                if other_tie or other_label == ("extra", 0):
                                    unintended = True
                                    break
                            if unintended:
                                break
                        if unintended:
                            continue
                        options.append((parent_idx, direction, child))

        if not options:
            return None

        parent_idx, direction, child = options[int(rng.integers(len(options)))]
        specs[parent_idx].directions.add(direction)
        specs.append(child)

    for idx, spec in enumerate(specs):
        if rng.random() >= EXTRA_SEED_PROB:
            continue
        extras = [direction for direction in DIRS if direction not in spec.directions]
        rng.shuffle(extras)
        for direction in extras:
            label, tie = _first_hit_label(specs, idx, direction)
            if not tie and label is None:
                spec.directions.add(direction)
                break

    return specs


def _place_distractors(
    rng: np.random.Generator,
    specs: list[NodeSpec],
    reached_count: int,
    want_count: int,
) -> bool:
    for _ in range(want_count):
        r0, c0, r1, c1 = _occupied_bbox(specs)
        placed = False
        for _ in range(120):
            origin = (
                int(rng.integers(r0 - DISTRACTOR_SPAN, r1 + DISTRACTOR_SPAN + 1)),
                int(rng.integers(c0 - DISTRACTOR_SPAN, c1 + DISTRACTOR_SPAN + 1)),
            )
            candidate = NodeSpec(origin, _random_subset(rng, min_size=1))
            if not _candidate_fits(specs, candidate):
                continue

            safe = True
            for idx in range(reached_count):
                for direction in specs[idx].directions:
                    label, tie = _first_hit_label(specs, idx, direction, extra_specs=[candidate])
                    if tie or label == ("extra", 0):
                        safe = False
                        break
                if not safe:
                    break
            if not safe:
                continue

            specs.append(candidate)
            placed = True
            break
        if not placed:
            return False
    return True


def _translate_specs_to_grid(
    rng: np.random.Generator,
    specs: list[NodeSpec],
) -> tuple[np.ndarray, list[Block], dict[tuple[int, int], int]] | None:
    r0, c0, r1, c1 = _occupied_bbox(specs)
    bbox_h = r1 - r0 + 1
    bbox_w = c1 - c0 + 1

    min_h = max(GRID_SIZE_MIN, bbox_h)
    min_w = max(GRID_SIZE_MIN, bbox_w)
    max_h = GRID_SIZE_MAX
    max_w = GRID_SIZE_MAX
    if min_h > max_h or min_w > max_w:
        return None

    height = int(rng.integers(min_h, max_h + 1))
    width = int(rng.integers(min_w, max_w + 1))
    top_shift = int(rng.integers(0, height - bbox_h + 1))
    left_shift = int(rng.integers(0, width - bbox_w + 1))
    delta_r = top_shift - r0
    delta_c = left_shift - c0

    translated: list[Block] = []
    occupied_by: dict[tuple[int, int], int] = {}
    for idx, spec in enumerate(specs):
        origin = (spec.origin[0] + delta_r, spec.origin[1] + delta_c)
        color = RED if idx == 0 else GREY
        block = _build_block(origin, color, spec.directions)
        for r, c in block.all_cells:
            if not in_bounds((height, width), r, c):
                return None
            if (r, c) in occupied_by:
                return None
            occupied_by[(r, c)] = idx
        translated.append(block)

    grid = np.full((height, width), BLACK, dtype=int)
    for block in translated:
        _paint_block(grid, block, block.color)
    return grid, translated, occupied_by


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    reached_count = int(rng.integers(*REACHED_COUNT_RANGE))
    specs = _build_reachable_specs(rng, reached_count)
    if specs is None:
        return None

    distractor_count = int(DISTRACTOR_COUNT_CHOICES[int(rng.integers(len(DISTRACTOR_COUNT_CHOICES)))])
    if not _place_distractors(rng, specs, reached_count, distractor_count):
        return None

    translated = _translate_specs_to_grid(rng, specs)
    if translated is None:
        return None
    inp, blocks, occupied_by = translated

    expected, ignited = _solve_from_blocks(inp, blocks, occupied_by, 0)
    if sum(ignited) != reached_count:
        return None
    if distractor_count and sum(not flag for flag in ignited) < distractor_count:
        return None

    added_trail = int(np.count_nonzero((inp == BLACK) & (expected == RED)))
    if added_trail == 0:
        return None
    if added_trail > inp.size // 3:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ignition-network input")
