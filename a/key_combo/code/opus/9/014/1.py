from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, crop_bbox, in_bounds


BLOCK_SIZE = 6
GRID_PARTS = 3
SEPARATOR_COLOR = Color.GREY
BG_COLOR = Color.BLACK
BLOCKER_COLOR = Color.TEAL
ENTRY_COLOR = Color.GREEN
EXIT_COLOR = Color.RED
TURN_TO_COLOR = {
    0: Color.BLUE,
    1: Color.YELLOW,
    2: Color.ORANGE,
    3: Color.MAROON,
    4: Color.PINK,
}

_BLOCK_LIBRARY: dict[int, list[dict]] | None = None
_EMITTED_PATTERNS: set[tuple[int, ...]] = set()


def _split_segments(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for cut in sorted(separators):
        if start < cut:
            spans.append((int(start), int(cut - 1)))
        start = cut + 1
    if start < length:
        spans.append((int(start), int(length - 1)))
    return spans


def _render_block(entry_row: int, exit_row: int, obstacles: tuple[tuple[int, int], ...]) -> np.ndarray:
    block = np.full((BLOCK_SIZE, BLOCK_SIZE), BG_COLOR, dtype=int)
    block[int(entry_row), 0] = ENTRY_COLOR
    block[int(exit_row), BLOCK_SIZE - 1] = EXIT_COLOR
    for r, c in obstacles:
        if (r, c) not in ((int(entry_row), 0), (int(exit_row), BLOCK_SIZE - 1)):
            block[int(r), int(c)] = BLOCKER_COLOR
    return block


def _trace_block(block: np.ndarray) -> dict | None:
    entry_cells = np.argwhere(block == ENTRY_COLOR)
    exit_cells = np.argwhere(block == EXIT_COLOR)
    if entry_cells.shape[0] != 1 or exit_cells.shape[0] != 1:
        return None

    sr, sc = map(int, entry_cells[0])
    er, ec = map(int, exit_cells[0])
    if sc != 0 or ec != block.shape[1] - 1:
        return None

    r, c = sr, sc
    dr, dc = 0, 1
    turns = 0
    seen_states: set[tuple[int, int, int, int]] = set()
    path = [(int(r), int(c))]
    hit_types: list[str] = []

    while True:
        state = (int(r), int(c), int(dr), int(dc))
        if state in seen_states:
            return None
        seen_states.add(state)

        nr, nc = int(r + dr), int(c + dc)
        if in_bounds(block.shape, nr, nc) and int(block[nr, nc]) == EXIT_COLOR:
            path.append((int(nr), int(nc)))
            return {
                "turns": int(turns),
                "path": tuple(path),
                "hit_types": tuple(hit_types),
            }

        blocked = False
        hit_kind = ""
        if not in_bounds(block.shape, nr, nc):
            blocked = True
            hit_kind = "wall"
        elif int(block[nr, nc]) == BLOCKER_COLOR:
            blocked = True
            hit_kind = "obstacle"

        if not blocked:
            r, c = int(nr), int(nc)
            path.append((int(r), int(c)))
            continue

        if dr == 0:
            candidate_dirs = [(-1, 0), (1, 0)]
        else:
            candidate_dirs = [(0, -1), (0, 1)]

        choices: list[tuple[int, int]] = []
        for tdr, tdc in candidate_dirs:
            tr, tc = int(r + tdr), int(c + tdc)
            if in_bounds(block.shape, tr, tc) and int(block[tr, tc]) != BLOCKER_COLOR:
                choices.append((int(tdr), int(tdc)))

        if len(choices) != 1:
            return None

        dr, dc = choices[0]
        turns += 1
        hit_types.append(hit_kind)


def _candidate_from_layout(entry_row: int, exit_row: int, obstacles: tuple[tuple[int, int], ...]) -> dict | None:
    block = _render_block(entry_row, exit_row, obstacles)
    traced = _trace_block(block)
    if traced is None or traced["turns"] not in TURN_TO_COLOR:
        return None

    necessary = 0
    for i in range(len(obstacles)):
        smaller = obstacles[:i] + obstacles[i + 1 :]
        alt = _trace_block(_render_block(entry_row, exit_row, smaller))
        if alt != traced:
            necessary += 1

    return {
        "entry_row": int(entry_row),
        "exit_row": int(exit_row),
        "obstacles": tuple(sorted((int(r), int(c)) for r, c in obstacles)),
        "turns": int(traced["turns"]),
        "path": traced["path"],
        "hit_types": traced["hit_types"],
        "necessary_obstacles": int(necessary),
        "obstacle_count": int(len(obstacles)),
        "delta": int(abs(exit_row - entry_row)),
        "layout_key": (int(entry_row), int(exit_row), tuple(sorted(obstacles))),
    }


def _build_block_library() -> dict[int, list[dict]]:
    pools: dict[int, list[dict]] = {turns: [] for turns in TURN_TO_COLOR}
    seen_layouts: set[tuple[int, int, tuple[tuple[int, int], ...]]] = set()
    cells = [(r, c) for r in range(BLOCK_SIZE) for c in range(BLOCK_SIZE)]
    attempts = 0

    while attempts < 250000 and any(len(pools[t]) < 30 for t in TURN_TO_COLOR):
        attempts += 1
        entry_row = random.randrange(BLOCK_SIZE)
        exit_row = random.randrange(BLOCK_SIZE)
        obstacle_count = random.randint(2, 5)
        choices = [p for p in cells if p not in ((entry_row, 0), (exit_row, BLOCK_SIZE - 1))]
        obstacles = tuple(sorted(random.sample(choices, obstacle_count)))
        key = (int(entry_row), int(exit_row), obstacles)
        if key in seen_layouts:
            continue
        seen_layouts.add(key)

        candidate = _candidate_from_layout(entry_row, exit_row, obstacles)
        if candidate is None:
            continue
        turns = int(candidate["turns"])
        if turns > 0 and int(candidate["necessary_obstacles"]) == 0:
            continue
        pools[turns].append(candidate)

    if any(len(pools[t]) < 12 for t in TURN_TO_COLOR):
        raise RuntimeError("failed to build a diverse local block library")
    return pools


def _get_block_library() -> dict[int, list[dict]]:
    global _BLOCK_LIBRARY
    if _BLOCK_LIBRARY is None:
        _BLOCK_LIBRARY = _build_block_library()
    return _BLOCK_LIBRARY


def _choose_turn_pattern() -> tuple[int, ...]:
    counts = [0, 1, 2, 3, 4]
    counts.extend(random.choices(list(TURN_TO_COLOR), k=GRID_PARTS * GRID_PARTS - len(counts)))
    random.shuffle(counts)
    return tuple(int(x) for x in counts)


def _pick_candidates(pattern: tuple[int, ...], library: dict[int, list[dict]]) -> list[dict] | None:
    chosen: list[dict] = []
    used_layouts: set[tuple[int, int, tuple[tuple[int, int], ...]]] = set()
    for turns in pattern:
        options = [cand for cand in library[int(turns)] if cand["layout_key"] not in used_layouts]
        if not options:
            return None
        cand = random.choice(options)
        chosen.append(cand)
        used_layouts.add(cand["layout_key"])
    return chosen


def _candidate_pairs(blocks: list[dict]) -> list[tuple[dict, dict]]:
    pairs: list[tuple[dict, dict]] = []
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            pairs.append((blocks[i], blocks[j]))
    return pairs


def _validate_semantics(pattern: tuple[int, ...], blocks: list[dict]) -> bool:
    counts = np.array(pattern, dtype=int).reshape(GRID_PARTS, GRID_PARTS)
    if len(set(pattern)) != 5:
        return False
    if np.array_equal(counts, np.fliplr(counts)):
        return False
    if np.array_equal(counts, np.flipud(counts)):
        return False
    if np.array_equal(counts, counts.T):
        return False

    pairs = _candidate_pairs(blocks)
    if not any(a["obstacle_count"] == b["obstacle_count"] and a["turns"] != b["turns"] for a, b in pairs):
        return False
    if not any(a["delta"] == b["delta"] and a["turns"] != b["turns"] for a, b in pairs):
        return False
    if not any(block["turns"] == 1 and block["necessary_obstacles"] >= 1 for block in blocks):
        return False
    if not any(block["turns"] in (2, 3, 4) and block["necessary_obstacles"] >= 2 for block in blocks):
        return False
    if not any("obstacle" in block["hit_types"] for block in blocks if block["turns"] >= 2):
        return False
    return True


def _assemble_grid(blocks: list[dict]) -> np.ndarray:
    full_size = GRID_PARTS * BLOCK_SIZE + (GRID_PARTS - 1)
    grid = np.full((full_size, full_size), BG_COLOR, dtype=int)

    for idx in range(1, GRID_PARTS):
        cut = idx * BLOCK_SIZE + (idx - 1)
        grid[cut, :] = SEPARATOR_COLOR
        grid[:, cut] = SEPARATOR_COLOR

    for idx, block in enumerate(blocks):
        br, bc = divmod(idx, GRID_PARTS)
        r0 = br * (BLOCK_SIZE + 1)
        c0 = bc * (BLOCK_SIZE + 1)
        grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = _render_block(
            block["entry_row"],
            block["exit_row"],
            block["obstacles"],
        )
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each separator-delimited mini-maze, count its turns, and compress the partition grid to turn-coded colors.

    Input structure:
    - Grey full rows and columns partition the canvas into a 3x3 array of equal subgrids.
    - Each subgrid contains black corridor cells, teal blocker pixels, one green entry on the left edge, and one red exit on the right edge.
    - The subgrid boundary and the teal blockers act as local walls that constrain a single deterministic path.

    Rule:
    - For each subgrid, start at the green entry and move right.
    - Keep moving through black cells; if the next step is the red exit, stop for that subgrid.
    - Whenever the next step would hit a teal blocker or leave the subgrid, make the unique perpendicular turn allowed by the local walls.
    - Count the number of direction changes, map 0/1/2/3/4 turns to blue/yellow/orange/maroon/pink, and place that color in the corresponding cell of the 3x3 output.

    Key insight:
    - The output color depends on the traced local route through each partition, not on raw obstacle count or endpoint rows alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    separator_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == SEPARATOR_COLOR)]
    separator_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEPARATOR_COLOR)]
    row_spans = _split_segments(g.shape[0], separator_rows)
    col_spans = _split_segments(g.shape[1], separator_cols)
    assert len(row_spans) == GRID_PARTS and len(col_spans) == GRID_PARTS

    out = np.full((len(row_spans), len(col_spans)), BG_COLOR, dtype=int)
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            block = crop_bbox(g, (r0, c0, r1, c1))
            traced = _trace_block(block)
            assert traced is not None
            turns = int(traced["turns"])
            assert turns in TURN_TO_COLOR
            out[rr, cc] = TURN_TO_COLOR[turns]

    return out


def generate_input() -> np.ndarray:
    library = _get_block_library()
    for _ in range(500):
        pattern = _choose_turn_pattern()
        if pattern in _EMITTED_PATTERNS:
            continue
        blocks = _pick_candidates(pattern, library)
        if blocks is None:
            continue
        if not _validate_semantics(pattern, blocks):
            continue

        grid = _assemble_grid(blocks)
        expected = np.array(
            [[TURN_TO_COLOR[int(pattern[r * GRID_PARTS + c])] for c in range(GRID_PARTS)] for r in range(GRID_PARTS)],
            dtype=int,
        )
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue

        _EMITTED_PATTERNS.add(pattern)
        return grid
    raise RuntimeError("failed to generate a valid partition-maze input")
