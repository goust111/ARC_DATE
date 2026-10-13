from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, components_mask, get_bg, neighbors4, random_free_location


BG = int(Color.BLACK)
NEUTRAL = int(Color.GREY)
DIR_BY_MARKER = {
    int(Color.GREEN): (0, -1),
    int(Color.YELLOW): (1, 0),
    int(Color.PINK): (-1, 0),
    int(Color.TEAL): (0, 1),
}
MARKER_COLORS = tuple(DIR_BY_MARKER)
GENERATE_ATTEMPTS = 400


@dataclass
class Block:
    block_id: int
    cells: set[tuple[int, int]]
    direction: tuple[int, int] | None
    moving: bool


def _neutral_color(grid: np.ndarray, bg: int) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    foreground = [(int(color), int(count)) for color, count in zip(colors, counts) if int(color) != bg]
    assert foreground, "expected foreground cells"
    top_count = max(count for _color, count in foreground)
    candidates = [color for color, count in foreground if count == top_count]
    assert len(candidates) == 1, "neutral block color must be uniquely most frequent"
    return int(candidates[0])


def _shift(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _stop_reason(blockers: set[int], blocks_by_id: dict[int, Block]) -> str:
    kinds = {"mover" if blocks_by_id[bid].moving else "static" for bid in blockers}
    if kinds == {"mover"}:
        return "mover"
    if kinds == {"static"}:
        return "static"
    return "mixed"


def _simulate(
    blocks: list[Block],
    shape: tuple[int, int],
    *,
    with_trace: bool,
) -> tuple[list[Block], dict[int, dict[str, object]]]:
    h, w = map(int, shape)
    state = [
        Block(
            block_id=int(block.block_id),
            cells=set(block.cells),
            direction=None if block.direction is None else tuple(map(int, block.direction)),
            moving=bool(block.moving),
        )
        for block in blocks
    ]
    by_id = {block.block_id: block for block in state}
    trace: dict[int, dict[str, object]] = {
        block.block_id: {"steps": 0, "stop": None} for block in state if block.moving
    }
    active = {block.block_id for block in state if block.moving}

    while active:
        occupancy: dict[tuple[int, int], int] = {}
        for block in state:
            for cell in block.cells:
                occupancy[cell] = block.block_id

        moves: dict[int, set[tuple[int, int]]] = {}
        stopping: dict[int, str] = {}

        for block_id in sorted(active):
            block = by_id[block_id]
            dr, dc = block.direction
            shifted = _shift(block.cells, dr, dc)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in shifted):
                stopping[block_id] = "edge"
                continue

            entered = shifted - block.cells
            blockers = {occupancy[cell] for cell in entered if cell in occupancy}
            if blockers:
                stopping[block_id] = _stop_reason(blockers, by_id)
                continue
            moves[block_id] = shifted

        for block_id, shifted in moves.items():
            by_id[block_id].cells = shifted
            if with_trace:
                trace[block_id]["steps"] = int(trace[block_id]["steps"]) + 1

        for block_id, reason in stopping.items():
            active.remove(block_id)
            if with_trace:
                trace[block_id]["stop"] = reason

        for block_id in list(active):
            if block_id not in moves:
                active.remove(block_id)
                if with_trace:
                    trace[block_id]["stop"] = "unknown"

    if with_trace:
        for block_id, info in trace.items():
            assert info["stop"] in {"edge", "static", "mover", "mixed"}
    return state, trace


def _parse_blocks(grid: np.ndarray) -> tuple[int, int, list[Block], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    bg = int(get_bg(g))
    neutral = _neutral_color(g, bg)
    marker_cells = set(map(tuple, np.argwhere((g != bg) & (g != neutral))))
    marker_colors = {int(g[r, c]) for r, c in marker_cells}
    assert marker_colors <= set(MARKER_COLORS), "unknown marker color"

    blocks: list[Block] = []
    for block_id, coords in enumerate(components_mask(g == neutral, connectivity=4)):
        cells = {(int(r), int(c)) for r, c in coords}
        touching = set()
        for r, c in cells:
            for rr, cc in neighbors4(r, c):
                if (rr, cc) in marker_cells:
                    touching.add(int(g[rr, cc]))
        assert len(touching) <= 1, "each neutral component may touch at most one marker color"
        if touching:
            color = touching.pop()
            blocks.append(Block(block_id=block_id, cells=cells, direction=DIR_BY_MARKER[color], moving=True))
        else:
            blocks.append(Block(block_id=block_id, cells=cells, direction=None, moving=False))

    assert any(block.moving for block in blocks), "expected at least one moving block"
    return bg, neutral, blocks, marker_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the marker pixels and synchronously slide each marked neutral block until the current layout blocks it.

    Input structure:
    - The background is the most common color and the neutral block color is the uniquely most frequent foreground color.
    - The neutral foreground is split into one or more 4-connected components, and some component exteriors have one adjacent colored marker pixel.
    - Marker colors are fixed direction codes, while unmarked neutral components are static obstacles.

    Rule:
    - Remove every marker pixel.
    - For each marked neutral component, move the whole component one cell per step in the direction encoded by its marker color.
    - All marked components test the next step against the current layout at the same time: if the translated shape would leave the grid or any newly entered cell is currently occupied, that component stops permanently; otherwise it advances one cell.
    - Repeat until every moving component has stopped, then draw the final layout using only the neutral color on the original background.

    Key insight:
    - The colored pixel is an instruction, not part of the object: the output depends on synchronous blocking between moving neutral components, not on recoloring or on sliding each component independently to an edge.
    """
    g = np.asarray(grid, dtype=int)
    bg, neutral, blocks, _marker_cells = _parse_blocks(g)
    final_blocks, _trace = _simulate(blocks, g.shape, with_trace=False)

    out = np.full(g.shape, bg, dtype=int)
    for block in final_blocks:
        for r, c in block.cells:
            out[r, c] = neutral
    return out


def _rect_patch(height: int, width: int) -> np.ndarray:
    return np.full((int(height), int(width)), NEUTRAL, dtype=int)


def _random_extra_patch() -> np.ndarray:
    base_shapes = [
        np.array([[1]], dtype=int),
        np.array([[1, 1]], dtype=int),
        np.array([[1], [1]], dtype=int),
        np.array([[1, 1], [1, 1]], dtype=int),
        np.array([[1, 0], [1, 1]], dtype=int),
        np.array([[1, 1, 1]], dtype=int),
        np.array([[1], [1], [1]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    ]
    patch = random.choice(base_shapes)
    turns = random.randrange(4)
    patch = np.rot90(patch, k=turns)
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    return np.where(patch == 1, NEUTRAL, BG).astype(int)


def _component_touches(cell: tuple[int, int], components: list[set[tuple[int, int]]]) -> list[int]:
    r, c = cell
    touched = []
    for idx, cells in enumerate(components):
        if any((rr, cc) in cells for rr, cc in neighbors4(r, c)):
            touched.append(idx)
    return touched


def _marker_candidates(
    comp_index: int,
    components: list[set[tuple[int, int]]],
    shape: tuple[int, int],
    used_markers: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    occupied = set().union(*components)
    candidates = []
    for r, c in components[comp_index]:
        for rr, cc in neighbors4(r, c):
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if (rr, cc) in occupied or (rr, cc) in used_markers:
                continue
            if any((nr, nc) in used_markers for nr, nc in neighbors4(rr, cc)):
                continue
            if _component_touches((rr, cc), components) == [comp_index]:
                candidates.append((rr, cc))
    return sorted(set(candidates))


def _collect_components(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    return [{(int(r), int(c)) for r, c in coords} for coords in components_mask(grid == NEUTRAL, connectivity=4)]


def _build_base_scene() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(20, 26)
    grid = np.full((h, w), BG, dtype=int)

    barrier_w = random.randint(1, 2)
    barrier_h = random.randint(h - 8, h - 3)
    barrier_r0 = random.randint(1, h - barrier_h - 1)
    barrier_c0 = random.randint(w // 2 - 2, w // 2)
    grid[barrier_r0 : barrier_r0 + barrier_h, barrier_c0 : barrier_c0 + barrier_w] = NEUTRAL

    right_start = barrier_c0 + barrier_w + 2
    left_end = barrier_c0 - 2
    if right_start >= w - 3 or left_end <= 3:
        return None

    pair_h = random.randint(1, 2)
    teal_w = random.randint(1, 3)
    green_w = random.randint(1, 3)
    pair_r0 = random.randint(1, h - pair_h - 2)
    teal_c0 = right_start + random.randint(0, 1)
    green_c0 = w - green_w - random.randint(1, 2)
    gap = green_c0 - (teal_c0 + teal_w)
    if gap < 2:
        return None
    grid[pair_r0 : pair_r0 + pair_h, teal_c0 : teal_c0 + teal_w] = NEUTRAL
    grid[pair_r0 : pair_r0 + pair_h, green_c0 : green_c0 + green_w] = NEUTRAL

    up_mode = random.choice(["edge", "static"])
    down_mode = "static" if up_mode == "edge" else "edge"

    up_w = random.randint(1, 2)
    down_w = random.randint(1, 2)
    cols = list(range(1, max(2, left_end - 1)))
    random.shuffle(cols)
    up_c0 = down_c0 = None
    for c0 in cols:
        if c0 + up_w > left_end:
            continue
        up_c0 = c0
        break
    if up_c0 is None:
        return None
    for c0 in cols:
        if c0 + down_w > left_end:
            continue
        if c0 >= up_c0 + up_w + 2 or up_c0 >= c0 + down_w + 2:
            down_c0 = c0
            break
    if down_c0 is None:
        return None

    up_h = random.randint(1, 3)
    if up_mode == "edge":
        up_r0 = random.randint(h // 2, h - up_h - 2)
    else:
        obs_h = random.randint(1, 2)
        gap_up = random.randint(2, 5)
        obs_r0 = random.randint(2, max(2, h // 2 - 2))
        up_r0 = obs_r0 + obs_h + gap_up
        if up_r0 + up_h >= h - 1:
            return None
        grid[obs_r0 : obs_r0 + obs_h, up_c0 : up_c0 + up_w] = NEUTRAL
    grid[up_r0 : up_r0 + up_h, up_c0 : up_c0 + up_w] = NEUTRAL

    down_h = random.randint(1, 3)
    if down_mode == "edge":
        down_r0 = random.randint(1, max(1, h // 2 - down_h - 1))
    else:
        obs_h = random.randint(1, 2)
        gap_down = random.randint(2, 5)
        obs_r0 = random.randint(max(h // 2, 2), h - obs_h - 3)
        down_r0 = obs_r0 - gap_down - down_h
        if down_r0 <= 0:
            return None
        grid[obs_r0 : obs_r0 + obs_h, down_c0 : down_c0 + down_w] = NEUTRAL
    grid[down_r0 : down_r0 + down_h, down_c0 : down_c0 + down_w] = NEUTRAL

    extra_count = random.randint(1, 2)
    for _ in range(extra_count):
        patch = _random_extra_patch()
        try:
            r0, c0 = random_free_location(
                grid,
                patch,
                transparent=BG,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
        except ValueError:
            return None
        mask = patch != BG
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = NEUTRAL

    return grid


def _stamp_markers(base_grid: np.ndarray) -> np.ndarray | None:
    grid = base_grid.copy()
    components = _collect_components(grid)
    if len(components) < 5:
        return None

    moving_targets: dict[int, int] = {}
    sorted_ids = sorted(range(len(components)), key=lambda idx: min(components[idx]))

    left_candidates = [idx for idx in sorted_ids if max(c for _r, c in components[idx]) < grid.shape[1] // 2]
    right_candidates = [idx for idx in sorted_ids if min(c for _r, c in components[idx]) > grid.shape[1] // 2]

    if len(left_candidates) < 2 or len(right_candidates) < 2:
        return None

    moving_targets[right_candidates[0]] = int(Color.TEAL)
    moving_targets[right_candidates[1]] = int(Color.GREEN)
    moving_targets[left_candidates[-1]] = int(Color.PINK)
    moving_targets[left_candidates[0]] = int(Color.YELLOW)

    used_markers: set[tuple[int, int]] = set()
    for comp_index, color in moving_targets.items():
        candidates = _marker_candidates(comp_index, components, grid.shape, used_markers)
        if not candidates:
            return None
        cell = random.choice(candidates)
        used_markers.add(cell)
        grid[cell] = int(color)

    return grid


def _trace_input(grid: np.ndarray) -> tuple[list[Block], dict[int, dict[str, object]]]:
    _bg, _neutral, blocks, _markers = _parse_blocks(grid)
    return _simulate(blocks, grid.shape, with_trace=True)


def _is_valid_training_example(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
        final_blocks, trace = _trace_input(grid)
    except Exception:
        return False

    if grid.shape != out.shape:
        return False
    if set(np.unique(out)) - {BG, NEUTRAL}:
        return False

    moving_colors = {int(grid[r, c]) for r, c in np.argwhere((grid != BG) & (grid != NEUTRAL))}
    if moving_colors != set(MARKER_COLORS):
        return False

    reasons = {info["stop"] for info in trace.values()}
    if "mover" not in reasons or "edge" not in reasons or "static" not in reasons:
        return False
    if not any(int(info["steps"]) > 0 for info in trace.values()):
        return False
    if np.array_equal(grid, out):
        return False

    # The horizontal pair should merge the synchronous-interaction clue into the final shape.
    final_sizes = sorted(len(block.cells) for block in final_blocks)
    if max(final_sizes) < 2:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        base = _build_base_scene()
        if base is None:
            continue
        grid = _stamp_markers(base)
        if grid is None:
            continue
        if _is_valid_training_example(grid):
            return grid
    raise RuntimeError("failed to generate a valid walking-instruction-marker input")
