import random
from itertools import permutations

import numpy as np


BG = 0
RED = 2
GREEN = 3
YELLOW = 4

ARROW_SPECS = {
    "up": {
        "cells": ((0, 1), (1, 0), (1, 2)),
        "delta": (-1, 0),
        "shape": (2, 3),
    },
    "down": {
        "cells": ((0, 0), (0, 2), (1, 1)),
        "delta": (1, 0),
        "shape": (2, 3),
    },
    "left": {
        "cells": ((0, 1), (1, 0), (2, 1)),
        "delta": (0, -1),
        "shape": (3, 2),
    },
    "right": {
        "cells": ((0, 0), (1, 1), (2, 0)),
        "delta": (0, 1),
        "shape": (3, 2),
    },
}
ARROW_BY_MASK = {
    frozenset(spec["cells"]): spec["delta"] for spec in ARROW_SPECS.values()
}
LABEL_SIDES = ("top", "bottom", "left", "right")

GRID_HEIGHT_RANGE = (12, 24)
GRID_WIDTH_RANGE = (12, 24)
MAX_STEPS = 4
GEN_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 250


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components_of_color(grid: np.ndarray, color: int, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    neighbor_fn = _neighbors4 if connectivity == 4 else _neighbors8
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in neighbor_fn(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalized_shape(coords: list[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0, c0, _, _ = _bbox(coords)
    return frozenset((r - r0, c - c0) for r, c in coords)


def _min_distance(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    return min(abs(ar - br) + abs(ac - bc) for ar, ac in a for br, bc in b)


def _find_mover(grid: np.ndarray) -> tuple[int, int]:
    comps = _components_of_color(grid, RED, connectivity=4)
    movers = []
    for comp in comps:
        if len(comp) != 4:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            continue
        if len(set(comp)) != 4:
            continue
        if all((r, c) in comp for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)):
            movers.append((r0, c0))
    assert len(movers) == 1, "expected exactly one red 2x2 mover"
    return movers[0]


def _parse_arrows(grid: np.ndarray) -> list[dict]:
    arrows = []
    for comp in _components_of_color(grid, GREEN, connectivity=8):
        assert len(comp) == 3, "each green arrow should contain 3 pixels"
        mask = _normalized_shape(comp)
        assert mask in ARROW_BY_MASK, "unrecognized arrow shape"
        arrows.append({"cells": comp, "delta": ARROW_BY_MASK[mask]})
    assert 1 <= len(arrows) <= MAX_STEPS, "expected 1-4 arrow sprites"
    return arrows


def _parse_labels(grid: np.ndarray) -> list[dict]:
    labels = []
    for comp in _components_of_color(grid, YELLOW, connectivity=4):
        count = len(comp)
        assert 1 <= count <= MAX_STEPS, "yellow marker size should be 1-4"
        labels.append({"cells": comp, "count": count})
    assert labels, "expected yellow marker components"
    return labels


def _assign_labels(arrows: list[dict], labels: list[dict]) -> list[tuple[int, tuple[int, int]]]:
    assert len(arrows) == len(labels), "each arrow should have one label"
    n = len(arrows)
    distances = [
        [_min_distance(arrow["cells"], label["cells"]) for label in labels]
        for arrow in arrows
    ]

    best_score = None
    best_perm = None
    tied = False
    for perm in permutations(range(n)):
        score = sum(distances[i][perm[i]] for i in range(n))
        if best_score is None or score < best_score:
            best_score = score
            best_perm = perm
            tied = False
        elif score == best_score:
            tied = True

    assert best_perm is not None and not tied, "arrow-label pairing should be unique"
    instructions = []
    for arrow_idx, label_idx in enumerate(best_perm):
        count = labels[label_idx]["count"]
        instructions.append((count, arrows[arrow_idx]["delta"]))

    counts = [count for count, _delta in instructions]
    assert len(set(counts)) == len(counts), "yellow counts should be unique"
    return sorted(instructions)


def main(grid: np.ndarray) -> np.ndarray:
    """Translate the red 2x2 mover by following the numbered green arrow sprites.

    Input structure:
    - The background is Black.
    - Exactly one Red 2x2 solid square is the mover object.
    - Several Green 3-pixel arrow sprites are scattered on the canvas; each sprite is a head pixel with two trailing pixels and points up, down, left, or right.
    - Each arrow has a nearby Yellow connected marker made of 1-4 single pixels, whose size encodes that arrow's execution order.

    Rule:
    - Pair every Yellow marker with its nearest Green arrow sprite and sort the arrows by marker size from 1 upward.
    - For each arrow in that order, move the Red 2x2 square exactly 2 cells in the arrow's pointing direction.
    - Remove all arrows, markers, and the original mover position.
    - Return a same-sized Black grid containing only the Red 2x2 square at its final location.

    Key insight:
    - The Green sprite shape encodes direction while the Yellow pixel count encodes sequence, so the task is to decode both symbol systems and accumulate the fixed 2-cell translations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    start_r, start_c = _find_mover(g)
    instructions = _assign_labels(_parse_arrows(g), _parse_labels(g))

    r, c = start_r, start_c
    for _count, (dr, dc) in instructions:
        r += 2 * dr
        c += 2 * dc

    h, w = g.shape
    assert 0 <= r <= h - 2 and 0 <= c <= w - 2, "final mover must stay in bounds"
    out = np.full((h, w), BG, dtype=int)
    out[r : r + 2, c : c + 2] = RED
    return out


def _label_cells(
    arrow_r0: int,
    arrow_c0: int,
    arrow_h: int,
    arrow_w: int,
    count: int,
    side: str,
    rng: random.Random,
) -> list[tuple[int, int]]:
    if side == "top":
        row = arrow_r0 - 1
        start = rng.randint(arrow_c0 - count + 1, arrow_c0 + arrow_w - 1)
        return [(row, start + i) for i in range(count)]
    if side == "bottom":
        row = arrow_r0 + arrow_h
        start = rng.randint(arrow_c0 - count + 1, arrow_c0 + arrow_w - 1)
        return [(row, start + i) for i in range(count)]
    if side == "left":
        col = arrow_c0 - 1
        start = rng.randint(arrow_r0 - count + 1, arrow_r0 + arrow_h - 1)
        return [(start + i, col) for i in range(count)]
    col = arrow_c0 + arrow_w
    start = rng.randint(arrow_r0 - count + 1, arrow_r0 + arrow_h - 1)
    return [(start + i, col) for i in range(count)]


def _cells_fit(shape: tuple[int, int], cells: list[tuple[int, int]]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _cells_clear(occupied: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(not occupied[r, c] for r, c in cells)


def _mark_with_padding(occupied: np.ndarray, cells: list[tuple[int, int]], pad: int = 1) -> None:
    h, w = occupied.shape
    for r, c in cells:
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        occupied[r0:r1, c0:c1] = True


def _trajectory_offsets(directions: list[str]) -> list[tuple[int, int]]:
    offsets = [(0, 0)]
    r = 0
    c = 0
    for name in directions:
        dr, dc = ARROW_SPECS[name]["delta"]
        r += 2 * dr
        c += 2 * dc
        offsets.append((r, c))
    return offsets


def _build_expected(shape: tuple[int, int], top_left: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    r, c = top_left
    out[r : r + 2, c : c + 2] = RED
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_HEIGHT_RANGE)
    w = rng.randint(*GRID_WIDTH_RANGE)
    step_count = rng.randint(1, MAX_STEPS)
    directions = [rng.choice(tuple(ARROW_SPECS)) for _ in range(step_count)]
    offsets = _trajectory_offsets(directions)
    final_dr, final_dc = offsets[-1]
    if (final_dr, final_dc) == (0, 0):
        return None

    min_r = min(dr for dr, _dc in offsets)
    max_r = max(dr for dr, _dc in offsets)
    min_c = min(dc for _dr, dc in offsets)
    max_c = max(dc for _dr, dc in offsets)

    start_r_low = -min_r
    start_r_high = h - 2 - max_r
    start_c_low = -min_c
    start_c_high = w - 2 - max_c
    if start_r_low > start_r_high or start_c_low > start_c_high:
        return None

    start_r = rng.randint(start_r_low, start_r_high)
    start_c = rng.randint(start_c_low, start_c_high)

    grid = np.full((h, w), BG, dtype=int)
    grid[start_r : start_r + 2, start_c : start_c + 2] = RED

    occupied = np.zeros((h, w), dtype=bool)
    path_cells: list[tuple[int, int]] = []
    for dr, dc in offsets:
        r0 = start_r + dr
        c0 = start_c + dc
        for rr in range(r0, r0 + 2):
            for cc in range(c0, c0 + 2):
                path_cells.append((rr, cc))
    _mark_with_padding(occupied, path_cells, pad=1)

    placed = {}
    for order in range(step_count, 0, -1):
        name = directions[order - 1]
        spec = ARROW_SPECS[name]
        arrow_h, arrow_w = spec["shape"]
        arrow_offsets = spec["cells"]
        success = False
        for _ in range(PLACEMENT_ATTEMPTS):
            arrow_r0 = rng.randint(0, h - arrow_h)
            arrow_c0 = rng.randint(0, w - arrow_w)
            side = rng.choice(LABEL_SIDES)
            label_cells = _label_cells(arrow_r0, arrow_c0, arrow_h, arrow_w, order, side, rng)
            arrow_cells = [(arrow_r0 + dr, arrow_c0 + dc) for dr, dc in arrow_offsets]
            cluster_cells = arrow_cells + label_cells
            if len(set(cluster_cells)) != len(cluster_cells):
                continue
            if not _cells_fit((h, w), cluster_cells):
                continue
            if not _cells_clear(occupied, cluster_cells):
                continue
            for r, c in arrow_cells:
                grid[r, c] = GREEN
            for r, c in label_cells:
                grid[r, c] = YELLOW
            _mark_with_padding(occupied, cluster_cells, pad=1)
            placed[order] = name
            success = True
            break
        if not success:
            return None

    assert placed and len(placed) == step_count

    end_r = start_r + final_dr
    end_c = start_c + final_dc
    expected = _build_expected((h, w), (end_r, end_c))
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
