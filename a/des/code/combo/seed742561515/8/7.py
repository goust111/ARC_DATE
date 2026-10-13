from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 8  # azure
ARROW = 5  # grey
BLOCK_COLORS = (1, 2, 3, 4, 6, 7, 9)
DIR_VECTORS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

# Each 2x2 L-triomino admits two arrow interpretations.
SHAPE_CANDIDATES = {
    frozenset(((0, 1), (1, 0), (1, 1))): (("U", (0, 1)), ("L", (1, 0))),
    frozenset(((0, 0), (1, 0), (1, 1))): (("U", (0, 0)), ("R", (1, 1))),
    frozenset(((0, 0), (0, 1), (1, 1))): (("D", (1, 1)), ("L", (0, 0))),
    frozenset(((0, 0), (0, 1), (1, 0))): (("D", (1, 0)), ("R", (0, 1))),
}
TEMPLATES_BY_DIR: dict[str, list[dict[str, object]]] = {d: [] for d in DIR_VECTORS}
for offsets, candidates in SHAPE_CANDIDATES.items():
    for idx, (direction, head) in enumerate(candidates):
        alt_direction, alt_head = candidates[1 - idx]
        TEMPLATES_BY_DIR[direction].append(
            {
                "offsets": tuple(sorted(offsets)),
                "head": head,
                "alt_direction": alt_direction,
                "alt_head": alt_head,
            }
        )


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR_VECTORS.values():
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _find_blocks(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in {bg, ARROW})
    for color in colors:
        for comp in _components(grid == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            cells = {(r, c) for r, c in comp}
            if len(cells) == 4 and r1 == r0 + 1 and c1 == c0 + 1:
                full = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
                if cells == full:
                    blocks.append({"color": color, "cells": tuple(sorted(cells))})
    return blocks


def _find_arrows(grid: np.ndarray) -> list[dict[str, object]]:
    arrows: list[dict[str, object]] = []
    for comp in _components(grid == ARROW):
        cells = {(r, c) for r, c in comp}
        if len(cells) != 3:
            continue
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 != r0 + 1 or c1 != c0 + 1:
            continue
        norm = frozenset((r - r0, c - c0) for r, c in cells)
        if norm not in SHAPE_CANDIDATES:
            continue
        candidates = []
        for direction, head_off in SHAPE_CANDIDATES[norm]:
            candidates.append(
                {
                    "direction": direction,
                    "head": (r0 + head_off[0], c0 + head_off[1]),
                }
            )
        arrows.append({"cells": tuple(sorted(cells)), "candidates": tuple(candidates)})
    return arrows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each grey 3-pixel arrow to the nearest object on its ray.

    If the chain eventually reaches a 2x2 colored block, recolor every arrow on
    that chain with the block color. Arrows whose chain reaches no block stay grey.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = BG if np.any(g == BG) else _most_common_color(g)
    arrows = _find_arrows(g)
    blocks = _find_blocks(g, bg)

    arrow_by_cell: dict[tuple[int, int], int] = {}
    for idx, arrow in enumerate(arrows):
        for cell in arrow["cells"]:
            arrow_by_cell[cell] = idx

    block_by_cell: dict[tuple[int, int], int] = {}
    for idx, block in enumerate(blocks):
        for cell in block["cells"]:
            block_by_cell[cell] = idx

    def scan_from(head: tuple[int, int], direction: str) -> tuple[str, int] | None:
        dr, dc = DIR_VECTORS[direction]
        r, c = head[0] + dr, head[1] + dc
        while _in_bounds(g.shape, r, c):
            if int(g[r, c]) != bg:
                if (r, c) in arrow_by_cell:
                    return ("arrow", arrow_by_cell[(r, c)])
                if (r, c) in block_by_cell:
                    return ("block", block_by_cell[(r, c)])
                return None
            r += dr
            c += dc
        return None

    memo: dict[int, int | None] = {}
    visiting: set[int] = set()

    def resolve_arrow(idx: int) -> int | None:
        if idx in memo:
            return memo[idx]
        if idx in visiting:
            return None

        visiting.add(idx)
        colors: list[int] = []
        for cand in arrows[idx]["candidates"]:
            hit = scan_from(cand["head"], cand["direction"])
            if hit is None:
                continue
            kind, hit_idx = hit
            if kind == "block":
                colors.append(int(blocks[hit_idx]["color"]))
            else:
                child = resolve_arrow(hit_idx)
                if child is not None:
                    colors.append(int(child))
        visiting.remove(idx)

        if not colors:
            memo[idx] = None
        elif len(set(colors)) == 1:
            memo[idx] = colors[0]
        else:
            memo[idx] = None
        return memo[idx]

    out = g.copy()
    for idx, arrow in enumerate(arrows):
        color = resolve_arrow(idx)
        if color is None:
            continue
        for r, c in arrow["cells"]:
            out[r, c] = color
    return out


GRID_MIN = 20
GRID_MAX = 30
CHAIN_COUNT_RANGE = (2, 4)
MAX_CHAIN_LENGTH = 4
STRAY_OPTIONS = (0, 1, 1, 2)
MAX_LINK_DISTANCE = 7
PADDING = 1
GEN_ATTEMPTS = 400


def _expand(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = PADDING) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    h, w = shape
    for r, c in cells:
        for rr in range(max(0, r - radius), min(h, r + radius + 1)):
            for cc in range(max(0, c - radius), min(w, c + radius + 1)):
                out.add((rr, cc))
    return out


def _ray_cells(head: tuple[int, int], direction: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = DIR_VECTORS[direction]
    r, c = head[0] + dr, head[1] + dc
    cells: list[tuple[int, int]] = []
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _segment_cells(head: tuple[int, int], target: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    dr, dc = DIR_VECTORS[direction]
    r, c = head
    cells: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if (r, c) == target:
            break
        cells.append((r, c))
    return cells


def _entry_cells(cells: set[tuple[int, int]], direction: str) -> list[tuple[int, int]]:
    dr, dc = DIR_VECTORS[direction]
    return sorted(cell for cell in cells if (cell[0] - dr, cell[1] - dc) not in cells)


def _sample_chain_lengths(n: int) -> list[int]:
    mode = random.choice(("mixed", "direct", "indirect"))
    lengths: list[int]
    if mode == "mixed":
        lengths = [1, random.randint(2, MAX_CHAIN_LENGTH)]
        while len(lengths) < n:
            lengths.append(random.randint(1, MAX_CHAIN_LENGTH))
    elif mode == "direct":
        lengths = [random.choice((1, 1, 2)) for _ in range(n)]
        if all(length == 1 for length in lengths):
            lengths[random.randrange(n)] = 2
    else:
        lengths = [random.randint(2, MAX_CHAIN_LENGTH) for _ in range(n)]
    random.shuffle(lengths)
    return lengths[:n]


def _blank_scene(shape: tuple[int, int]) -> dict[str, object]:
    return {
        "grid": np.full(shape, BG, dtype=int),
        "occupied": set(),
        "padded": set(),
        "reserved": set(),
    }


def _cells_available(state: dict[str, object], cells: set[tuple[int, int]]) -> bool:
    grid = state["grid"]
    padded = state["padded"]
    reserved = state["reserved"]
    for r, c in cells:
        if not _in_bounds(grid.shape, r, c):
            return False
        if (r, c) in padded or (r, c) in reserved:
            return False
    return True


def _commit_object(
    state: dict[str, object],
    cells: set[tuple[int, int]],
    color: int,
    reserved: set[tuple[int, int]] | None = None,
) -> dict[str, object]:
    grid = state["grid"]
    for r, c in cells:
        grid[r, c] = color
    state["occupied"].update(cells)
    state["padded"].update(_expand(cells, grid.shape))
    if reserved:
        state["reserved"].update(reserved)
    return {"cells": tuple(sorted(cells)), "color": color}


def _try_place_block(state: dict[str, object], color: int) -> dict[str, object] | None:
    grid = state["grid"]
    choices = [(r, c) for r in range(1, grid.shape[0] - 2) for c in range(1, grid.shape[1] - 2)]
    random.shuffle(choices)
    for r0, c0 in choices:
        cells = {(r0, c0), (r0 + 1, c0), (r0, c0 + 1), (r0 + 1, c0 + 1)}
        if _cells_available(state, cells):
            return _commit_object(state, cells, color)
    return None


def _candidate_arrow_placements(
    state: dict[str, object],
    target_cells: set[tuple[int, int]],
) -> list[dict[str, object]]:
    grid = state["grid"]
    shape = grid.shape
    candidates: list[dict[str, object]] = []
    directions = list(DIR_VECTORS)
    random.shuffle(directions)
    for direction in directions:
        entry_cells = _entry_cells(target_cells, direction)
        if not entry_cells:
            continue
        random.shuffle(entry_cells)
        templates = list(TEMPLATES_BY_DIR[direction])
        random.shuffle(templates)
        distances = list(range(2, MAX_LINK_DISTANCE + 1))
        random.shuffle(distances)
        dr, dc = DIR_VECTORS[direction]
        for entry in entry_cells:
            for template in templates:
                head_off = template["head"]
                alt_head_off = template["alt_head"]
                alt_direction = template["alt_direction"]
                offsets = template["offsets"]
                for dist in distances:
                    head = (entry[0] - dr * dist, entry[1] - dc * dist)
                    anchor = (head[0] - head_off[0], head[1] - head_off[1])
                    cells = {(anchor[0] + rr, anchor[1] + cc) for rr, cc in offsets}
                    if not _cells_available(state, cells):
                        continue
                    segment = _segment_cells(head, entry, direction)
                    if any(cell in state["occupied"] or cell in state["reserved"] for cell in segment):
                        continue
                    alt_head = (anchor[0] + alt_head_off[0], anchor[1] + alt_head_off[1])
                    alt_ray = _ray_cells(alt_head, alt_direction, shape)
                    if any(cell in state["occupied"] or cell in state["reserved"] for cell in alt_ray):
                        continue
                    candidates.append(
                        {
                            "cells": cells,
                            "reserved": set(segment) | set(alt_ray),
                        }
                    )
    return candidates


def _try_place_arrow_to_target(
    state: dict[str, object],
    target: dict[str, object],
) -> dict[str, object] | None:
    target_cells = set(target["cells"])
    placements = _candidate_arrow_placements(state, target_cells)
    if not placements:
        return None
    choice = random.choice(placements)
    return _commit_object(state, choice["cells"], ARROW, reserved=choice["reserved"])


def _try_place_stray_arrow(state: dict[str, object]) -> dict[str, object] | None:
    grid = state["grid"]
    shape = grid.shape
    anchors = [(r, c) for r in range(shape[0] - 1) for c in range(shape[1] - 1)]
    random.shuffle(anchors)
    for anchor in anchors[:200]:
        directions = list(DIR_VECTORS)
        random.shuffle(directions)
        for direction in directions:
            templates = list(TEMPLATES_BY_DIR[direction])
            random.shuffle(templates)
            for template in templates:
                offsets = template["offsets"]
                cells = {(anchor[0] + rr, anchor[1] + cc) for rr, cc in offsets}
                if not _cells_available(state, cells):
                    continue
                head = (anchor[0] + template["head"][0], anchor[1] + template["head"][1])
                alt_head = (anchor[0] + template["alt_head"][0], anchor[1] + template["alt_head"][1])
                primary_ray = _ray_cells(head, direction, shape)
                alt_ray = _ray_cells(alt_head, template["alt_direction"], shape)
                if any(cell in state["occupied"] or cell in state["reserved"] for cell in primary_ray):
                    continue
                if any(cell in state["occupied"] or cell in state["reserved"] for cell in alt_ray):
                    continue
                return _commit_object(state, cells, ARROW, reserved=set(primary_ray) | set(alt_ray))
    return None


def _gen_attempt() -> np.ndarray | None:
    chain_count = random.randint(*CHAIN_COUNT_RANGE)
    chain_lengths = _sample_chain_lengths(chain_count)
    stray_count = random.choice(STRAY_OPTIONS)
    total_arrows = sum(chain_lengths) + stray_count

    side_min = max(GRID_MIN, 12 + total_arrows)
    side_max = max(side_min, GRID_MAX)
    h = random.randint(side_min, side_max)
    w = random.randint(side_min, side_max)
    state = _blank_scene((h, w))

    colors = random.sample(BLOCK_COLORS, chain_count)
    chains = sorted(zip(chain_lengths, colors), reverse=True)

    for length, color in chains:
        target = _try_place_block(state, color)
        if target is None:
            return None
        for _ in range(length):
            arrow = _try_place_arrow_to_target(state, target)
            if arrow is None:
                return None
            target = arrow

    for _ in range(stray_count):
        if _try_place_stray_arrow(state) is None:
            return None

    inp = state["grid"]
    try:
        out = main(inp)
    except Exception:
        return None

    # Ensure the sample actually exercises the rule set.
    if not np.any(inp == ARROW):
        return None
    if np.array_equal(inp, out):
        return None
    if stray_count > 0 and not np.any(out == ARROW):
        return None
    return inp.copy()


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
