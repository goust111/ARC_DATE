from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg, in_bounds


BG = 0
PALETTE = list(range(1, 10))
NEIGH4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

CORE_SHAPES = [
    {(-1, 0), (0, -1), (1, -1), (1, 0), (1, 1)},
    {(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1)},
    {(-1, -1), (-1, 0), (0, -1), (1, -1), (1, 0), (1, 1)},
    {(-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)},
    {(-1, -1), (-1, 0), (0, 1), (1, 0), (1, 1)},
]

BASE_TARGET_KINDS = ["up_left", "up_right", "down_left", "down_right"]
ROW_TARGET_KINDS = ["same_row_left", "same_row_right"]
COL_TARGET_KINDS = ["same_col_up", "same_col_down"]

GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
MAX_ATTEMPTS = 400


def _to_coord_list(coords: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in coords]


def _touches8(cell: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    r, c = cell
    for rr, cc in cells:
        if max(abs(r - rr), abs(c - cc)) == 1:
            return True
    return False


def _identify_scene(grid: np.ndarray) -> tuple[int, int, tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)

    source_comps = [comp for comp in comps if int(comp["area"]) > 1]
    assert len(source_comps) == 1, "expected exactly one multi-pixel source component"
    source = source_comps[0]
    source_color = int(source["color"])

    same_color = [comp for comp in comps if int(comp["color"]) == source_color]
    assert len(same_color) == 1, "source color must belong only to the source component"

    source_cells = sorted(_to_coord_list(source["coords"]))
    source_set = set(source_cells)

    singleton_comps = [comp for comp in comps if int(comp["area"]) == 1 and int(comp["color"]) != source_color]
    assert singleton_comps, "expected an anchor pixel and at least one target marker"

    adjacent_singletons: list[tuple[int, int]] = []
    for comp in singleton_comps:
        cell = tuple(map(int, comp["coords"][0]))
        if _touches8(cell, source_set):
            adjacent_singletons.append(cell)

    assert len(adjacent_singletons) == 1, "expected exactly one singleton anchor touching the source"
    anchor = adjacent_singletons[0]

    targets = sorted(
        tuple(map(int, comp["coords"][0]))
        for comp in singleton_comps
        if tuple(map(int, comp["coords"][0])) != anchor
    )
    assert targets, "expected at least one target marker"
    return bg, source_color, anchor, source_cells, targets


def _relative_source(source_cells: list[tuple[int, int]], anchor: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = anchor
    return sorted((r - ar, c - ac) for r, c in source_cells)


def _stamp_cells(
    anchor: tuple[int, int],
    rel_source: list[tuple[int, int]],
    target: tuple[int, int],
) -> set[tuple[int, int]]:
    ar, ac = anchor
    tr, tc = target
    row_sign = -1 if tr < ar else 1
    col_sign = -1 if tc < ac else 1
    cells = {(tr, tc)}
    for dr, dc in rel_source:
        cells.add((tr + row_sign * dr, tc + col_sign * dc))
    return cells


def _render_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, source_color, anchor, source_cells, targets = _identify_scene(g)
    rel_source = _relative_source(source_cells, anchor)
    out = g.copy()
    anchor_color = int(g[anchor])

    painted: set[tuple[int, int]] = set()
    for target in targets:
        cells = _stamp_cells(anchor, rel_source, target)
        for rr, cc in cells:
            assert in_bounds(g.shape, rr, cc), "stamp must stay in bounds"
            current = int(g[rr, cc])
            if (rr, cc) != target:
                assert current != anchor_color, "stamps must not overwrite the anchor"
                assert current in {bg, source_color}, "stamps must avoid unrelated input objects"
            if (rr, cc) in painted:
                raise AssertionError("stamps must not conflict")
        painted |= cells
        for rr, cc in cells:
            out[rr, cc] = source_color

    out[anchor] = anchor_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the source shape onto every target marker, flipping the copied orientation by the target's side relative to the anchor.

    Input structure:
    - A dominant background contains one monochrome multi-pixel source object and one adjacent singleton anchor pixel of a globally unique color.
    - Elsewhere, several singleton target markers of colors different from both the background and the source are scattered on the canvas.

    Rule:
    - Read every source pixel as a relative offset from the anchor.
    - For each target marker independently, negate all relative row offsets if the target lies above the anchor row, and negate all relative column offsets if the target lies left of the anchor column; on the same row or column, that axis is not flipped.
    - Repaint the transformed source offsets in the source color with the target as the new origin, and also cover the target marker cell itself with the source color.
    - Keep the original source object and anchor in place; valid inputs ensure the stamped copies do not conflict.

    Key insight:
    - The anchor is a control origin: every target receives the same relative source mask, but each axis is mirrored independently according to which side of the anchor that target occupies.
    """
    return _render_output(grid)


def _connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGH4:
            nbr = (r + dr, c + dc)
            if nbr in cells and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return len(seen) == len(cells)


def _valid_shape(cells: set[tuple[int, int]]) -> bool:
    if not cells or (0, 0) in cells or not _connected(cells):
        return False

    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if not (min(rows) < 0 < max(rows) and min(cols) < 0 < max(cols)):
        return False

    row_flip = {(-r, c) for r, c in cells}
    col_flip = {(r, -c) for r, c in cells}
    both_flip = {(-r, -c) for r, c in cells}
    if cells in (row_flip, col_flip, both_flip):
        return False
    return True


def _transform_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    transformed = set(cells)
    if random.random() < 0.5:
        transformed = {(c, r) for r, c in transformed}
    if random.random() < 0.5:
        transformed = {(-r, c) for r, c in transformed}
    if random.random() < 0.5:
        transformed = {(r, -c) for r, c in transformed}
    return transformed


def _generate_relative_shape() -> list[tuple[int, int]]:
    for _ in range(200):
        cells = set(random.choice(CORE_SHAPES))
        goal = random.randint(len(cells), 7)
        while len(cells) < goal:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGH4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) == (0, 0) or max(abs(nr), abs(nc)) > 2 or (nr, nc) in cells:
                        continue
                    frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        cells = _transform_shape(cells)
        if _valid_shape(cells):
            return sorted(cells)
    raise RuntimeError("failed to generate a valid asymmetric source shape")


def _relation_ok(cell: tuple[int, int], anchor: tuple[int, int], kind: str) -> bool:
    r, c = cell
    ar, ac = anchor
    if kind == "up_left":
        return r < ar and c < ac
    if kind == "up_right":
        return r < ar and c > ac
    if kind == "down_left":
        return r > ar and c < ac
    if kind == "down_right":
        return r > ar and c > ac
    if kind == "same_row_left":
        return r == ar and c < ac
    if kind == "same_row_right":
        return r == ar and c > ac
    if kind == "same_col_up":
        return c == ac and r < ar
    if kind == "same_col_down":
        return c == ac and r > ar
    raise ValueError(f"unknown target kind: {kind}")


def _too_close(a: set[tuple[int, int]], b: set[tuple[int, int]], gap: int = 1) -> bool:
    for r1, c1 in a:
        for r2, c2 in b:
            if max(abs(r1 - r2), abs(c1 - c2)) <= gap:
                return True
    return False


def _choose_anchor(shape: tuple[int, int], rel_source: list[tuple[int, int]]) -> tuple[int, int]:
    h, w = shape
    min_dr = min(dr for dr, _ in rel_source)
    max_dr = max(dr for dr, _ in rel_source)
    min_dc = min(dc for _, dc in rel_source)
    max_dc = max(dc for _, dc in rel_source)

    row_lo = max(4, -min_dr + 1)
    row_hi = min(h - 5, h - 2 - max_dr)
    col_lo = max(4, -min_dc + 1)
    col_hi = min(w - 5, w - 2 - max_dc)
    if row_lo > row_hi or col_lo > col_hi:
        raise RuntimeError("grid too small for anchor placement")
    return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)


def _pick_target(
    grid_shape: tuple[int, int],
    anchor: tuple[int, int],
    rel_source: list[tuple[int, int]],
    kind: str,
    forbidden_regions: list[set[tuple[int, int]]],
) -> tuple[int, int] | None:
    h, w = grid_shape
    ar, ac = anchor
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if not _relation_ok(cell, anchor, kind):
                continue
            if abs(r - ar) + abs(c - ac) < 5:
                continue
            footprint = _stamp_cells(anchor, rel_source, cell)
            if any(not in_bounds(grid_shape, rr, cc) for rr, cc in footprint):
                continue
            if any(_too_close(footprint, region, gap=1) for region in forbidden_regions):
                continue
            candidates.append(cell)

    if not candidates:
        return None

    random.shuffle(candidates)
    candidates.sort(key=lambda cell: abs(cell[0] - ar) + abs(cell[1] - ac), reverse=True)
    top = candidates[: max(1, min(12, len(candidates)))]
    return random.choice(top)


def _build_target_kinds() -> list[str]:
    kinds = list(BASE_TARGET_KINDS)
    row_kind = random.choice(ROW_TARGET_KINDS)
    col_kind = random.choice(COL_TARGET_KINDS)
    if random.random() < 0.5:
        kinds.append(row_kind)
    else:
        kinds.append(col_kind)
    if random.random() < 0.45:
        kinds.append(col_kind if kinds[-1] in ROW_TARGET_KINDS else row_kind)
    random.shuffle(kinds)
    return kinds


def _expected_from_parts(
    inp: np.ndarray,
    source_color: int,
    anchor: tuple[int, int],
    rel_source: list[tuple[int, int]],
    targets: list[tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    for target in targets:
        for rr, cc in _stamp_cells(anchor, rel_source, target):
            out[rr, cc] = source_color
    return out


def _validate_generated(
    inp: np.ndarray,
    out: np.ndarray,
    anchor: tuple[int, int],
    source_cells: list[tuple[int, int]],
    targets: list[tuple[int, int]],
    target_kinds: list[str],
) -> bool:
    bg, source_color, got_anchor, got_source_cells, got_targets = _identify_scene(inp)
    if bg != BG or got_anchor != anchor:
        return False
    if got_source_cells != sorted(source_cells) or got_targets != sorted(targets):
        return False

    rel_source = _relative_source(source_cells, anchor)
    source_region = set(source_cells) | {anchor}
    used_regions = [source_region]
    for target in targets:
        footprint = _stamp_cells(anchor, rel_source, target)
        if any(not in_bounds(inp.shape, rr, cc) for rr, cc in footprint):
            return False
        if any(_too_close(footprint, region, gap=1) for region in used_regions):
            return False
        if int(out[target]) != source_color or int(inp[target]) == source_color:
            return False
        used_regions.append(footprint)

    if not all(any(_relation_ok(target, anchor, kind) for target in targets) for kind in BASE_TARGET_KINDS):
        return False
    if not any(kind in ROW_TARGET_KINDS + COL_TARGET_KINDS for kind in target_kinds):
        return False
    if not np.array_equal(out, _expected_from_parts(inp, source_color, anchor, rel_source, targets)):
        return False
    if not np.array_equal(out, main(inp)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    rel_source = _generate_relative_shape()
    anchor = _choose_anchor((h, w), rel_source)
    source_cells = [(anchor[0] + dr, anchor[1] + dc) for dr, dc in rel_source]

    target_kinds = _build_target_kinds()
    source_region = set(source_cells) | {anchor}
    forbidden_regions = [source_region]
    targets: list[tuple[int, int]] = []
    for kind in target_kinds:
        target = _pick_target((h, w), anchor, rel_source, kind, forbidden_regions)
        if target is None:
            return None
        targets.append(target)
        forbidden_regions.append(_stamp_cells(anchor, rel_source, target))

    colors = random.sample(PALETTE, 2 + len(targets))
    source_color = colors[0]
    anchor_color = colors[1]
    target_colors = colors[2:]

    inp = np.full((h, w), BG, dtype=int)
    for r, c in source_cells:
        inp[r, c] = source_color
    inp[anchor] = anchor_color
    for color, target in zip(target_colors, targets):
        inp[target] = color

    out = _expected_from_parts(inp, source_color, anchor, rel_source, targets)
    if not _validate_generated(inp, out, anchor, source_cells, targets, target_kinds):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid orientation-flip stamp input")
