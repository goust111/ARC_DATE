from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREY = 5
ORANGE = 7
CYAN = 8

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
LEFT = {0: 3, 1: 0, 2: 1, 3: 2}
RIGHT = {0: 1, 1: 2, 2: 3, 3: 0}
OPPOSITE_TURN = {"L": "R", "R": "L"}

MIN_SIZE = 18
MAX_SIZE = 24
SOURCE_MARGIN = 5
MIN_SEG_LEN = 2
MAX_SEG_LEN = 7
MAX_GENERATION_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rotate(dir_idx: int, lr: str) -> int:
    return LEFT[dir_idx] if lr == "L" else RIGHT[dir_idx]


def _missing_corner_for_turn(dir_idx: int, new_dir: int, spare_bit: int) -> tuple[int, int]:
    if dir_idx in (1, 3):
        row = 0 if new_dir == 0 else 1
        return row, int(spare_bit)
    col = 0 if new_dir == 3 else 1
    return int(spare_bit), col


def _turn_from_missing(dir_idx: int, missing: tuple[int, int]) -> int:
    mr, mc = missing
    if dir_idx in (1, 3):
        return 0 if mr == 0 else 2
    return 3 if mc == 0 else 1


def _bbox_cells(top: int, left: int) -> list[tuple[int, int]]:
    return [(top + dr, left + dc) for dr in range(2) for dc in range(2)]


def _expanded_bbox_cells(shape: tuple[int, int], top: int, left: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(top - 1, top + 3):
        for c in range(left - 1, left + 3):
            if _in_bounds(shape, r, c):
                cells.append((r, c))
    return cells


def _segment_cells(start: tuple[int, int], dir_idx: int, length: int) -> list[tuple[int, int]]:
    dr, dc = DIRS[dir_idx]
    r, c = start
    return [(r + dr * step, c + dc * step) for step in range(1, length + 1)]


def _ray_to_edge(start: tuple[int, int], dir_idx: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = DIRS[dir_idx]
    r, c = start
    cells = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return cells
        cells.append((r, c))


def _candidate_bboxes(turn_cell: tuple[int, int], dir_idx: int) -> list[tuple[int, int]]:
    r, c = turn_cell
    if dir_idx == 0:
        return [(r - 2, c - 1), (r - 2, c)]
    if dir_idx == 1:
        return [(r - 1, c + 1), (r, c + 1)]
    if dir_idx == 2:
        return [(r + 1, c - 1), (r + 1, c)]
    return [(r - 1, c - 2), (r, c - 2)]


def _build_turn_candidates(turn_cell: tuple[int, int], dir_idx: int, lr: str) -> list[dict]:
    new_dir = _rotate(dir_idx, lr)
    candidates = []
    for top, left in _candidate_bboxes(turn_cell, dir_idx):
        for spare_bit in random.sample([0, 1], 2):
            candidates.append(
                {
                    "block_type": "L",
                    "top": top,
                    "left": left,
                    "missing": _missing_corner_for_turn(dir_idx, new_dir, spare_bit),
                    "hit_dir": dir_idx,
                }
            )
    random.shuffle(candidates)
    return candidates


def _build_solid_candidates(turn_cell: tuple[int, int], dir_idx: int) -> list[dict]:
    candidates = [{"block_type": "solid", "top": top, "left": left} for top, left in _candidate_bboxes(turn_cell, dir_idx)]
    random.shuffle(candidates)
    return candidates


def _block_cells(block: dict) -> list[tuple[int, int]]:
    return _bbox_cells(int(block["top"]), int(block["left"]))


def _block_bbox_key(block: dict) -> tuple[int, int]:
    return int(block["top"]), int(block["left"])


def _valid_block_placement(
    shape: tuple[int, int],
    block: dict,
    reserved: set[tuple[int, int]],
    protected_paths: set[tuple[int, int]],
) -> bool:
    top = int(block["top"])
    left = int(block["left"])
    if top < 0 or left < 0 or top + 1 >= shape[0] or left + 1 >= shape[1]:
        return False
    bbox_cells = _block_cells(block)
    if any(cell in reserved or cell in protected_paths for cell in bbox_cells):
        return False
    return True


def _place_beam(
    shape: tuple[int, int],
    source: tuple[int, int],
    start_dir: int,
    spec: dict,
    reserved: set[tuple[int, int]],
    protected_paths: set[tuple[int, int]],
    existing_bbox_cells: set[tuple[int, int]],
):
    turns = list(spec["turns"])
    terminal = str(spec["terminal"])

    def rec(
        pos: tuple[int, int],
        dir_idx: int,
        turn_i: int,
        local_blocks: list[dict],
        local_reserved: set[tuple[int, int]],
        local_bbox_cells: set[tuple[int, int]],
        path_cells: list[tuple[int, int]],
        events: list[tuple[str, tuple[int, int]]],
    ):
        if turn_i == len(turns):
            if terminal == "edge":
                tail = _ray_to_edge(pos, dir_idx, shape)
                if len(tail) < MIN_SEG_LEN or any(cell in local_bbox_cells for cell in tail):
                    return None
                return {
                    "blocks": local_blocks,
                    "reserved": local_reserved,
                    "bbox_cells": local_bbox_cells,
                    "path_cells": path_cells + tail,
                    "events": events,
                }

            max_k = min(MAX_SEG_LEN, max(shape))
            k_choices = list(range(MIN_SEG_LEN, max_k + 1))
            random.shuffle(k_choices)
            for length in k_choices:
                segment = _segment_cells(pos, dir_idx, length)
                if not all(_in_bounds(shape, r, c) for r, c in segment):
                    continue
                if any(cell in local_bbox_cells for cell in segment):
                    continue
                turn_cell = segment[-1]
                for block in _build_solid_candidates(turn_cell, dir_idx):
                    if not _valid_block_placement(shape, block, local_reserved, protected_paths):
                        continue
                    new_blocks = local_blocks + [block]
                    new_reserved = set(local_reserved)
                    new_reserved.update(_expanded_bbox_cells(shape, int(block["top"]), int(block["left"])))
                    new_bbox_cells = set(local_bbox_cells)
                    new_bbox_cells.update(_block_cells(block))
                    return {
                        "blocks": new_blocks,
                        "reserved": new_reserved,
                        "bbox_cells": new_bbox_cells,
                        "path_cells": path_cells + segment,
                        "events": events + [("solid", _block_bbox_key(block))],
                    }
            return None

        lr = turns[turn_i]
        max_k = min(MAX_SEG_LEN, max(shape))
        k_choices = list(range(MIN_SEG_LEN, max_k + 1))
        random.shuffle(k_choices)
        for length in k_choices:
            segment = _segment_cells(pos, dir_idx, length)
            if not all(_in_bounds(shape, r, c) for r, c in segment):
                continue
            if any(cell in local_bbox_cells for cell in segment):
                continue
            turn_cell = segment[-1]
            for block in _build_turn_candidates(turn_cell, dir_idx, lr):
                if not _valid_block_placement(shape, block, local_reserved, protected_paths):
                    continue
                new_blocks = local_blocks + [block]
                new_reserved = set(local_reserved)
                new_reserved.update(_expanded_bbox_cells(shape, int(block["top"]), int(block["left"])))
                new_bbox_cells = set(local_bbox_cells)
                new_bbox_cells.update(_block_cells(block))
                new_events = events + [("turn", _block_bbox_key(block))]
                new_path = path_cells + segment
                next_dir = _turn_from_missing(dir_idx, block["missing"])
                result = rec(turn_cell, next_dir, turn_i + 1, new_blocks, new_reserved, new_bbox_cells, new_path, new_events)
                if result is not None:
                    return result
        return None

    return rec(source, start_dir, 0, [], set(reserved), set(existing_bbox_cells), [], [])


def _parse_scene(grid: np.ndarray) -> tuple[tuple[int, int], dict[tuple[int, int], dict], dict[tuple[int, int], tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    src = np.argwhere(g == RED)
    assert src.shape == (1, 2), "expected exactly one red source pixel"
    source = tuple(map(int, src[0]))

    used_colors = {int(v) for v in np.unique(g)}
    assert used_colors <= {BLACK, RED, GREY, ORANGE, CYAN}, "unexpected colors present"

    blocks: dict[tuple[int, int], dict] = {}
    occupancy: dict[tuple[int, int], tuple[int, int]] = {}

    for coords in _components_of_color(g, GREY):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "grey block must be 2x2"
        assert coords.shape[0] == 4
        key = (r0, c0)
        blocks[key] = {"block_type": "solid", "top": r0, "left": c0}
        for cell in _bbox_cells(r0, c0):
            assert cell not in occupancy, "block bboxes must not overlap"
            occupancy[cell] = key

    for coords in _components_of_color(g, ORANGE):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "orange marker must be a 2x2 L"
        assert coords.shape[0] == 3
        patch = np.zeros((2, 2), dtype=bool)
        for r, c in coords:
            patch[int(r - r0), int(c - c0)] = True
        missing = np.argwhere(~patch)
        assert missing.shape == (1, 2)
        key = (r0, c0)
        blocks[key] = {"block_type": "L", "top": r0, "left": c0, "missing": tuple(map(int, missing[0]))}
        for cell in _bbox_cells(r0, c0):
            assert cell not in occupancy, "block bboxes must not overlap"
            occupancy[cell] = key

    assert any(block["block_type"] == "L" for block in blocks.values()), "need at least one L block"
    assert any(block["block_type"] == "solid" for block in blocks.values()), "need at least one solid block"

    for key, block in blocks.items():
        br, bc = map(int, key)
        assert all(max(abs(r - source[0]), abs(c - source[1])) > 1 for r, c in _bbox_cells(br, bc)), "blocks must not be adjacent to the source"
        for other_key in blocks:
            if key >= other_key:
                continue
            ar, ac = key
            rr, cc = other_key
            assert abs(ar - rr) > 2 or abs(ac - cc) > 2, "blocks must not touch"

    return source, blocks, occupancy


def _simulate(grid: np.ndarray, *, return_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    source, blocks, occupancy = _parse_scene(g)
    out = g.copy()
    traces = []

    for start_dir in range(4):
        r, c = source
        dir_idx = start_dir
        seen: set[tuple[int, int, int]] = set()
        path_cells: list[tuple[int, int]] = []
        events: list[tuple[str, tuple[int, int]]] = []

        while True:
            state = (r, c, dir_idx)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = DIRS[dir_idx]
            nr, nc = r + dr, c + dc
            if not _in_bounds(out.shape, nr, nc):
                break

            block_key = occupancy.get((nr, nc))
            if block_key is not None:
                block = blocks[block_key]
                if block["block_type"] == "solid":
                    events.append(("solid", block_key))
                    break
                events.append(("turn", block_key))
                dir_idx = _turn_from_missing(dir_idx, block["missing"])
                continue

            r, c = nr, nc
            path_cells.append((r, c))
            if out[r, c] == BLACK:
                out[r, c] = CYAN

        traces.append({"events": events, "path": path_cells})

    if return_trace:
        return out, traces
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit cyan rays from the red source, turning at orange L-blocks and stopping at grey 2x2 squares.

    Input structure:
    - The grid has a black background, exactly one red source pixel, several orange 2x2 L-triomino blocks, and several grey 2x2 solid squares.
    - Blocks do not overlap, and the orange L blocks are interpreted by their full 2x2 bounding boxes rather than as passable missing-cell gaps.
    - The source and all blocks stay fixed in place.

    Rule:
    - Fire one ray from the red source in each cardinal direction.
    - Each ray advances cell by cell, painting only black cells cyan.
    - If the next cell lies in a grey 2x2 square's bounding box, stop on the current cell.
    - If the next cell lies in an orange L block's 2x2 bounding box, turn on the current cell:
      for a horizontal ray, the missing corner's top/bottom half chooses up/down; for a vertical ray, the missing corner's left/right half chooses left/right.

    Key insight:
    - The orange L block is a direction code stored in its missing corner, while the entire 2x2 box acts as the collision object.
    """
    return _simulate(grid, return_trace=False)


def _choose_source(shape: tuple[int, int]) -> tuple[int, int]:
    return (
        random.randint(SOURCE_MARGIN, shape[0] - SOURCE_MARGIN - 1),
        random.randint(SOURCE_MARGIN, shape[1] - SOURCE_MARGIN - 1),
    )


def _build_specs() -> dict[int, dict]:
    primary = random.randrange(4)
    first_turn = random.choice(["L", "R"])
    primary_terminal = random.choice(["edge", "solid"])
    specs: dict[int, dict] = {
        primary: {"turns": [first_turn, OPPOSITE_TURN[first_turn]], "terminal": primary_terminal}
    }

    secondary = random.choice([d for d in range(4) if d != primary])
    secondary_turn = random.choice(["L", "R"])
    specs[secondary] = {"turns": [secondary_turn], "terminal": "solid" if primary_terminal == "edge" else "edge"}

    remaining = [d for d in range(4) if d not in specs]
    random.shuffle(remaining)
    specs[remaining[0]] = {"turns": [], "terminal": "solid"}
    specs[remaining[1]] = {"turns": [], "terminal": "edge"}
    return specs


def _grid_from_blocks(shape: tuple[int, int], source: tuple[int, int], blocks: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[source] = RED
    for block in blocks:
        top = int(block["top"])
        left = int(block["left"])
        if block["block_type"] == "solid":
            grid[top : top + 2, left : left + 2] = GREY
        else:
            grid[top : top + 2, left : left + 2] = ORANGE
            mr, mc = block["missing"]
            grid[top + int(mr), left + int(mc)] = BLACK
    return grid


def _flip_l_orientation(grid: np.ndarray, block: dict) -> np.ndarray:
    alt = grid.copy()
    top = int(block["top"])
    left = int(block["left"])
    mr, mc = map(int, block["missing"])
    if int(block["hit_dir"]) in (1, 3):
        mr = 1 - mr
    else:
        mc = 1 - mc
    alt[top : top + 2, left : left + 2] = ORANGE
    alt[top + mr, left + mc] = BLACK
    return alt


def _remove_solid(grid: np.ndarray, block: dict) -> np.ndarray:
    alt = grid.copy()
    top = int(block["top"])
    left = int(block["left"])
    alt[top : top + 2, left : left + 2] = BLACK
    return alt


def _validate_semantics(grid: np.ndarray, planned: dict[int, list[tuple[str, tuple[int, int]]]], blocks: list[dict]) -> bool:
    try:
        out, traces = _simulate(grid, return_trace=True)
    except AssertionError:
        return False

    if np.array_equal(out, grid):
        return False

    used = set()
    for dir_idx in range(4):
        actual = traces[dir_idx]["events"]
        if actual != planned[dir_idx]:
            return False
        if len(traces[dir_idx]["path"]) < MIN_SEG_LEN:
            return False
        for kind, key in actual:
            used.add((kind, key))

    if sum(1 for events in planned.values() if any(kind == "turn" for kind, _ in events)) < 2:
        return False
    if not any(any(kind == "solid" for kind, _ in events) for events in planned.values()):
        return False
    if not any(len(events) == 0 or events[-1][0] == "turn" for events in planned.values()):
        return False
    if len({key for _kind, key in used}) != len(blocks):
        return False

    for block in blocks:
        if block["block_type"] == "L":
            alt = _flip_l_orientation(grid, block)
            if np.array_equal(main(alt), out):
                return False
        else:
            alt = _remove_solid(grid, block)
            if np.array_equal(main(alt), out):
                return False

    return True


def _attempt_generate() -> np.ndarray | None:
    shape = (random.randint(MIN_SIZE, MAX_SIZE), random.randint(MIN_SIZE, MAX_SIZE))
    source = _choose_source(shape)
    specs = _build_specs()

    reserved: set[tuple[int, int]] = set()
    for r in range(source[0] - 1, source[0] + 2):
        for c in range(source[1] - 1, source[1] + 2):
            if _in_bounds(shape, r, c):
                reserved.add((r, c))

    protected_paths: set[tuple[int, int]] = set()
    existing_bbox_cells: set[tuple[int, int]] = set()
    all_blocks: list[dict] = []
    planned: dict[int, list[tuple[str, tuple[int, int]]]] = {d: [] for d in range(4)}

    beam_order = sorted(range(4), key=lambda d: (len(specs[d]["turns"]), specs[d]["terminal"] == "solid"), reverse=True)
    for start_dir in beam_order:
        result = _place_beam(shape, source, start_dir, specs[start_dir], reserved, protected_paths, existing_bbox_cells)
        if result is None:
            return None
        reserved = result["reserved"]
        existing_bbox_cells = result["bbox_cells"]
        protected_paths.update(result["path_cells"])
        all_blocks.extend(result["blocks"])
        planned[start_dir] = result["events"]

    grid = _grid_from_blocks(shape, source, all_blocks)
    if not _validate_semantics(grid, planned, all_blocks):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid missing-corner compass task")
