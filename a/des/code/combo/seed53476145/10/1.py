from __future__ import annotations

from itertools import combinations
import numpy as np


_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in _DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "cells": cells})
    return comps


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _leaf_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    leaves: set[tuple[int, int]] = set()
    for r, c in cells:
        degree = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c, h, w))
        if degree == 1:
            leaves.add((r, c))
    return leaves


def _leaf_side(core: set[tuple[int, int]], leaves: set[tuple[int, int]]) -> str:
    r0, c0, r1, c1 = _bbox(core)
    candidates = []
    if all(r < r0 for r, _ in leaves):
        candidates.append("up")
    if all(r > r1 for r, _ in leaves):
        candidates.append("down")
    if all(c < c0 for _, c in leaves):
        candidates.append("left")
    if all(c > c1 for _, c in leaves):
        candidates.append("right")
    assert len(candidates) == 1, "leaf pixels must all lie on one side of the core"
    return candidates[0]


def _shift(side: str, amount: int) -> tuple[int, int]:
    if side == "up":
        return amount, 0
    if side == "down":
        return -amount, 0
    if side == "left":
        return 0, amount
    return 0, -amount


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each sprite's leaf pixels, then move the remaining core in the opposite direction.

    Input structure:
    - Background is black.
    - Each non-background object is a 4-connected monochrome sprite.
    - Every sprite has a core plus one or more leaf pixels (degree-1 cells).
    - All leaf pixels of a sprite lie on the same side of its core.

    Rule:
    - For each sprite, find all degree-1 foreground pixels.
    - Remove those leaf pixels.
    - Detect which side of the core those leaf pixels occupy.
    - Translate the remaining core in the opposite direction by exactly the number of removed leaf pixels.
    - Clear the original sprite position to background.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    out = np.full_like(g, bg)

    for comp in _find_components(g, bg):
        color = int(comp["color"])
        cells = {tuple(cell) for cell in comp["cells"]}
        leaves = _leaf_cells(cells, h, w)
        assert leaves, "each sprite must have at least one leaf pixel"

        core = cells - leaves
        assert core, "removing leaves must leave a non-empty core"

        side = _leaf_side(core, leaves)
        dr, dc = _shift(side, len(leaves))
        moved = {(r + dr, c + dc) for r, c in core}
        assert all(0 <= r < h and 0 <= c < w for r, c in moved), "translated core left the grid"
        assert all(int(out[r, c]) == int(bg) for r, c in moved), "translated cores must not overlap"
        for r, c in moved:
            out[r, c] = color

    return out


# --------------- generator ---------------

_GEN_ATTEMPTS = 400
_GRID_MIN = 14
_GRID_MAX = 23
_MAX_OBJECTS = 4
_MAX_LEAVES = 4
_PALETTE = np.arange(1, 10)


def _has_internal_leaves(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    for r, c in cells:
        degree = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c, h, w))
        if degree == 1:
            return True
    return False


def _core_shape(rng: np.random.Generator, side: str, leaf_count: int) -> tuple[set[tuple[int, int]], int, int]:
    along = 2 * leaf_count - 1
    for _ in range(80):
        kind = rng.choice(["solid", "hollow", "punctured"])

        if kind == "solid":
            min_h, min_w = 2, 2
        elif kind == "hollow":
            min_h, min_w = 3, 3
        else:
            min_h, min_w = 4, 4

        if side in ("up", "down"):
            h = int(rng.integers(min_h, 7))
            w = int(rng.integers(max(min_w, along), 8))
        else:
            h = int(rng.integers(max(min_h, along), 8))
            w = int(rng.integers(min_w, 7))

        cells = {(r, c) for r in range(h) for c in range(w)}
        if kind == "hollow":
            cells = {(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)}
        elif kind == "punctured":
            holes = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
            rng.shuffle(holes)
            hole_count = int(rng.integers(1, min(3, len(holes)) + 1))
            for hole in holes[:hole_count]:
                cells.remove(hole)

        if not _has_internal_leaves(cells, h, w):
            return cells, h, w

    raise RuntimeError("failed to generate a leafless core")


def _non_adjacent_indices(span: int, count: int, rng: np.random.Generator) -> list[int]:
    choices = [
        list(combo)
        for combo in combinations(range(span), count)
        if all(combo[i + 1] - combo[i] >= 2 for i in range(len(combo) - 1))
    ]
    assert choices, "shape side must support the requested number of leaves"
    return choices[int(rng.integers(0, len(choices)))]


def _leaf_offsets(side: str, h: int, w: int, indices: list[int]) -> set[tuple[int, int]]:
    if side == "up":
        return {(-1, idx) for idx in indices}
    if side == "down":
        return {(h, idx) for idx in indices}
    if side == "left":
        return {(idx, -1) for idx in indices}
    return {(idx, w) for idx in indices}


def _mark_forbidden(cells: set[tuple[int, int]], h: int, w: int, forbidden: set[tuple[int, int]]) -> None:
    for r, c in cells:
        for dr, dc in _DIR8:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                forbidden.add((rr, cc))


def _placeable(cells: set[tuple[int, int]], h: int, w: int, forbidden: set[tuple[int, int]]) -> bool:
    return all(0 <= r < h and 0 <= c < w and (r, c) not in forbidden for r, c in cells)


def _make_object(
    rng: np.random.Generator,
    grid_h: int,
    grid_w: int,
    input_forbidden: set[tuple[int, int]],
    output_forbidden: set[tuple[int, int]],
) -> tuple[dict[str, object], set[tuple[int, int]], set[tuple[int, int]]] | None:
    side = rng.choice(["up", "down", "left", "right"])
    leaf_count = int(rng.integers(1, _MAX_LEAVES + 1))
    core_rel, core_h, core_w = _core_shape(rng, side, leaf_count)

    span = core_w if side in ("up", "down") else core_h
    if span < 2 * leaf_count - 1:
        return None
    attach = _non_adjacent_indices(span, leaf_count, rng)
    leaves_rel = _leaf_offsets(side, core_h, core_w, attach)

    dr, dc = _shift(side, leaf_count)
    for _ in range(120):
        r0 = int(rng.integers(0, grid_h - core_h + 1))
        c0 = int(rng.integers(0, grid_w - core_w + 1))

        input_core = {(r0 + r, c0 + c) for r, c in core_rel}
        input_leaves = {(r0 + r, c0 + c) for r, c in leaves_rel}
        input_cells = input_core | input_leaves
        output_cells = {(r + dr, c + dc) for r, c in input_core}

        if not _placeable(input_cells, grid_h, grid_w, input_forbidden):
            continue
        if not _placeable(output_cells, grid_h, grid_w, output_forbidden):
            continue

        info = {
            "side": side,
            "leaf_count": leaf_count,
            "input_cells": input_cells,
            "output_cells": output_cells,
        }
        return info, input_cells, output_cells
    return None


def _paint(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    grid_h = int(rng.integers(_GRID_MIN, _GRID_MAX + 1))
    grid_w = int(rng.integers(_GRID_MIN, _GRID_MAX + 1))
    obj_count = int(rng.integers(1, _MAX_OBJECTS + 1))
    colors = [int(c) for c in rng.choice(_PALETTE, size=obj_count, replace=False)]

    inp = np.zeros((grid_h, grid_w), dtype=int)
    expected = np.zeros_like(inp)
    input_forbidden: set[tuple[int, int]] = set()
    output_forbidden: set[tuple[int, int]] = set()

    directions = ["up", "down", "left", "right"]
    rng.shuffle(directions)

    for i in range(obj_count):
        forced_side = directions[i % 4] if obj_count >= 4 else None
        placed = None
        for _ in range(60):
            candidate = _make_object(rng, grid_h, grid_w, input_forbidden, output_forbidden)
            if candidate is None:
                continue
            info, input_cells, output_cells = candidate
            if forced_side is not None and info["side"] != forced_side:
                continue
            placed = candidate
            break
        if placed is None:
            return None

        info, input_cells, output_cells = placed
        color = colors[i]
        _paint(inp, input_cells, color)
        _paint(expected, output_cells, color)
        _mark_forbidden(input_cells, grid_h, grid_w, input_forbidden)
        _mark_forbidden(output_cells, grid_h, grid_w, output_forbidden)

    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {_GEN_ATTEMPTS} attempts")
