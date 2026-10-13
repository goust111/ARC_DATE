from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ORTHOGONAL = {
    "up": ["left", "right"],
    "down": ["left", "right"],
    "left": ["up", "down"],
    "right": ["up", "down"],
}
SOURCE_POOL = [BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE, MAROON]
KNOWN_MIXES = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, YELLOW)): GREEN,
    frozenset((RED, GREEN)): MAROON,
    frozenset((BLUE, GREEN)): AZURE,
}
KNOWN_PAIRS = [
    (RED, BLUE),
    (RED, YELLOW),
    (BLUE, YELLOW),
    (RED, GREEN),
    (BLUE, GREEN),
]

GENERATION_ATTEMPTS = 300
COMPONENT_ATTEMPTS = 200


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _mix_colors(colors: list[int]) -> int:
    if len(colors) == 1:
        return int(colors[0])
    if len(colors) == 2:
        return int(KNOWN_MIXES.get(frozenset(map(int, colors)), GREY))
    return GREY


def _find_sources(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    sources: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (BLACK, GREY)):
        for comp in _components(g == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if hh not in (2, 3) or ww not in (2, 3):
                continue
            if len(comp) != hh * ww:
                continue
            if not bool((g[r0 : r1 + 1, c0 : c1 + 1] == color).all()):
                continue
            sources.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "cells": set(comp),
                }
            )
    sources.sort(key=lambda src: (src["bbox"][0], src["bbox"][1], src["color"]))
    return sources


def _source_adjacency(
    sources: list[dict], h: int, w: int
) -> dict[tuple[int, int], set[int]]:
    adj: dict[tuple[int, int], set[int]] = {}
    for idx, src in enumerate(sources):
        cells = src["cells"]
        for r, c in cells:
            for nr, nc in _neighbors4(r, c, h, w):
                if (nr, nc) in cells:
                    continue
                adj.setdefault((nr, nc), set()).add(idx)
    return adj


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each colored source rectangle through its grey pipe component and fill
    that component's unique terminal hole.

    Input structure:
    - Background is black.
    - Non-grey colored source objects are solid 2x2 or 3x3 rectangles.
    - Grey pixels form orthogonal pipe components.
    - Each valid pipe component has one or more source attachments and one free
      endpoint whose next cell in the same direction is a black terminal hole.

    Rule:
    - A terminal reached by one source gets that source color.
    - A terminal reached by two sources uses the fixed mix table.
    - Any other two-color pair, or three-plus sources, becomes grey.
    - Pipes and source rectangles remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    out = g.copy()
    sources = _find_sources(g)
    if not sources:
        return out

    adj = _source_adjacency(sources, h, w)
    for comp in _components(g == GREY):
        comp_set = set(comp)
        attached: set[int] = set()
        for cell in comp:
            attached.update(adj.get(cell, set()))
        if not attached:
            continue

        terminals: set[tuple[int, int]] = set()
        for r, c in comp:
            grey_neighbors = [(nr, nc) for nr, nc in _neighbors4(r, c, h, w) if (nr, nc) in comp_set]
            if len(grey_neighbors) != 1:
                continue
            if adj.get((r, c)):
                continue
            pr, pc = grey_neighbors[0]
            dr, dc = r - pr, c - pc
            tr, tc = r + dr, c + dc
            if 0 <= tr < h and 0 <= tc < w and int(g[tr, tc]) == BLACK:
                terminals.add((tr, tc))

        if len(terminals) != 1:
            continue

        colors = [int(sources[idx]["color"]) for idx in sorted(attached)]
        tr, tc = next(iter(terminals))
        out[tr, tc] = _mix_colors(colors)

    return out


def _max_source_length(hub_r: int, hub_c: int, direction: str, size: int, rh: int, rw: int) -> int:
    if direction == "up":
        return hub_r - size - 1
    if direction == "down":
        return rh - hub_r - size - 2
    if direction == "left":
        return hub_c - size - 1
    return rw - hub_c - size - 2


def _layout_source(
    hub_r: int,
    hub_c: int,
    direction: str,
    size: int,
    offset: int,
    length: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    block: set[tuple[int, int]] = set()
    branch: set[tuple[int, int]] = set()

    if direction == "up":
        r0 = hub_r - length - size
        c0 = hub_c - offset
        for rr in range(r0, r0 + size):
            for cc in range(c0, c0 + size):
                block.add((rr, cc))
        for step in range(1, length + 1):
            branch.add((hub_r - step, hub_c))
    elif direction == "down":
        r0 = hub_r + length + 1
        c0 = hub_c - offset
        for rr in range(r0, r0 + size):
            for cc in range(c0, c0 + size):
                block.add((rr, cc))
        for step in range(1, length + 1):
            branch.add((hub_r + step, hub_c))
    elif direction == "left":
        r0 = hub_r - offset
        c0 = hub_c - length - size
        for rr in range(r0, r0 + size):
            for cc in range(c0, c0 + size):
                block.add((rr, cc))
        for step in range(1, length + 1):
            branch.add((hub_r, hub_c - step))
    else:
        r0 = hub_r - offset
        c0 = hub_c + length + 1
        for rr in range(r0, r0 + size):
            for cc in range(c0, c0 + size):
                block.add((rr, cc))
        for step in range(1, length + 1):
            branch.add((hub_r, hub_c + step))

    return block, branch


def _max_straight_terminal_length(r: int, c: int, direction: str, rh: int, rw: int) -> int:
    if direction == "up":
        return r - 2
    if direction == "down":
        return rh - r - 3
    if direction == "left":
        return c - 2
    return rw - c - 3


def _max_bend_run(r: int, c: int, direction: str, rh: int, rw: int) -> int:
    if direction == "up":
        return r - 1
    if direction == "down":
        return rh - r - 2
    if direction == "left":
        return c - 1
    return rw - c - 2


def _advance(r: int, c: int, direction: str, steps: int) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    return r + dr * steps, c + dc * steps


def _straight_trunk(
    hub_r: int, hub_c: int, direction: str, length: int
) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for step in range(1, length + 1):
        cells.add(_advance(hub_r, hub_c, direction, step))
    tip = _advance(hub_r, hub_c, direction, length)
    terminal = _advance(hub_r, hub_c, direction, length + 1)
    return cells, tip, terminal


def _bent_trunk(
    hub_r: int,
    hub_c: int,
    direction0: str,
    length0: int,
    direction1: str,
    length1: int,
) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for step in range(1, length0 + 1):
        cells.add(_advance(hub_r, hub_c, direction0, step))
    bend_r, bend_c = _advance(hub_r, hub_c, direction0, length0)
    for step in range(1, length1 + 1):
        cells.add(_advance(bend_r, bend_c, direction1, step))
    tip = _advance(bend_r, bend_c, direction1, length1)
    terminal = _advance(bend_r, bend_c, direction1, length1 + 1)
    return cells, tip, terminal


def _choose_case_colors(rng: random.Random) -> list[tuple[str, list[int]]]:
    known_pair = list(rng.choice(KNOWN_PAIRS))
    remaining = [color for color in SOURCE_POOL if color not in known_pair]

    fallback_pairs = [
        pair for pair in combinations(remaining, 2) if frozenset(pair) not in KNOWN_MIXES
    ]
    other_pair = list(rng.choice(fallback_pairs))

    leftovers = [color for color in remaining if color not in other_pair]
    rng.shuffle(leftovers)
    multi = leftovers[:3]
    single = [leftovers[3]]

    cases = [
        ("single", single),
        ("known_pair", known_pair),
        ("other_pair", other_pair),
        ("multi", multi),
    ]
    rng.shuffle(cases)
    return cases


def _build_local_component(
    rh: int,
    rw: int,
    source_colors: list[int],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(COMPONENT_ATTEMPTS):
        colors = list(source_colors)
        rng.shuffle(colors)

        hub_r = rng.randint(3, rh - 4)
        hub_c = rng.randint(3, rw - 4)
        trunk_start = rng.choice(list(DIRS))
        source_dirs = rng.sample([name for name in DIRS if name != trunk_start], len(colors))
        bent = bool(rng.random() < 0.55)
        trunk_final = rng.choice(ORTHOGONAL[trunk_start]) if bent else trunk_start

        grey_cells: set[tuple[int, int]] = {(hub_r, hub_c)}
        source_blocks: list[tuple[int, set[tuple[int, int]]]] = []
        block_cells: set[tuple[int, int]] = set()
        ok = True

        for color, direction in zip(colors, source_dirs):
            size = rng.choice([2, 3])
            offset = rng.randrange(size)
            max_len = min(4, _max_source_length(hub_r, hub_c, direction, size, rh, rw))
            if max_len < 1:
                ok = False
                break
            length = rng.randint(1, max_len)
            block, branch = _layout_source(hub_r, hub_c, direction, size, offset, length)

            if any(rr < 1 or rr > rh - 2 or cc < 1 or cc > rw - 2 for rr, cc in block):
                ok = False
                break
            if block & block_cells:
                ok = False
                break
            if block & grey_cells:
                ok = False
                break
            if branch & block_cells:
                ok = False
                break
            if branch & grey_cells:
                ok = False
                break

            block_cells.update(block)
            grey_cells.update(branch)
            source_blocks.append((color, block))

        if not ok:
            continue

        if bent:
            max0 = min(4, _max_bend_run(hub_r, hub_c, trunk_start, rh, rw))
            if max0 < 1:
                continue
            length0 = rng.randint(1, max0)
            bend_r, bend_c = _advance(hub_r, hub_c, trunk_start, length0)
            max1 = min(4, _max_straight_terminal_length(bend_r, bend_c, trunk_final, rh, rw))
            if max1 < 1:
                continue
            length1 = rng.randint(1, max1)
            trunk_cells, _tip, terminal = _bent_trunk(
                hub_r, hub_c, trunk_start, length0, trunk_final, length1
            )
        else:
            max_trunk = min(5, _max_straight_terminal_length(hub_r, hub_c, trunk_start, rh, rw))
            if max_trunk < 1:
                continue
            length = rng.randint(1, max_trunk)
            trunk_cells, _tip, terminal = _straight_trunk(hub_r, hub_c, trunk_start, length)

        if terminal in grey_cells or terminal in block_cells:
            continue
        if any(rr < 1 or rr > rh - 2 or cc < 1 or cc > rw - 2 for rr, cc in trunk_cells):
            continue
        if trunk_cells & block_cells:
            continue
        if trunk_cells & grey_cells:
            continue

        grey_cells.update(trunk_cells)
        patch = np.full((rh, rw), BLACK, dtype=int)
        for rr, cc in grey_cells:
            patch[rr, cc] = GREY
        for color, block in source_blocks:
            for rr, cc in block:
                patch[rr, cc] = int(color)

        if int(patch[terminal]) != BLACK:
            continue
        terminal_grey_neighbors = sum(
            1 for nr, nc in _neighbors4(terminal[0], terminal[1], rh, rw) if int(patch[nr, nc]) == GREY
        )
        if terminal_grey_neighbors != 1:
            continue

        expected = patch.copy()
        expected[terminal] = _mix_colors(colors)
        try:
            if not np.array_equal(main(patch), expected):
                continue
        except Exception:
            continue
        return patch, expected
    return None


def _regions(h: int, w: int, split_r: int, split_c: int) -> list[tuple[int, int, int, int]]:
    return [
        (1, 1, split_r - 2, split_c - 2),
        (1, split_c + 1, split_r - 2, w - split_c - 2),
        (split_r + 1, 1, h - split_r - 2, split_c - 2),
        (split_r + 1, split_c + 1, h - split_r - 2, w - split_c - 2),
    ]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(24, 30)
    w = rng.randint(24, 30)
    split_r = rng.randint(12, h - 12)
    split_c = rng.randint(12, w - 12)

    regions = _regions(h, w, split_r, split_c)
    if any(rh < 10 or rw < 10 for _, _, rh, rw in regions):
        return None

    cases = _choose_case_colors(rng)
    grid = np.full((h, w), BLACK, dtype=int)
    expected = grid.copy()

    for (top, left, rh, rw), (_case_name, colors) in zip(regions, cases):
        built = _build_local_component(rh, rw, colors, rng)
        if built is None:
            return None
        patch, out_patch = built
        grid[top : top + rh, left : left + rw] = patch
        expected[top : top + rh, left : left + rw] = out_patch

    try:
        produced = main(grid)
    except Exception:
        return None
    if np.array_equal(produced, grid):
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-mixing ARC input")
