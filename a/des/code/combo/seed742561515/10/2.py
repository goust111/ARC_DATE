from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 8
ARROW_COLOR = 9
RIBBON_COLORS = (1, 2, 3, 4, 5, 6, 7)
GEN_RETRIES = 300
OBJECT_RETRIES = 200


# Arrow = 1-cell shaft + 3-pixel head. The tip is the forward-most head-center cell.
_ARROW_OFFSETS = {
    "right": ((0, 0), (0, -1), (-1, 0), (1, 0)),
    "left": ((0, 0), (0, 1), (-1, 0), (1, 0)),
    "down": ((0, 0), (-1, 0), (0, -1), (0, 1)),
    "up": ((0, 0), (1, 0), (0, -1), (0, 1)),
}

_ARROW_SHAPES = {
    "right": frozenset({(0, 1), (1, 0), (1, 1), (2, 1)}),
    "left": frozenset({(0, 0), (1, 0), (1, 1), (2, 0)}),
    "down": frozenset({(0, 1), (1, 0), (1, 1), (1, 2)}),
    "up": frozenset({(0, 0), (0, 1), (0, 2), (1, 1)}),
}

_ARROW_TIP_OFFSETS = {
    "right": (1, 1),
    "left": (1, 0),
    "down": (1, 1),
    "up": (0, 1),
}

_STEP = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}

_PERP_CHOICES = {
    "right": [(-2, 0), (2, 0)],
    "left": [(-2, 0), (2, 0)],
    "down": [(0, -2), (0, 2)],
    "up": [(0, -2), (0, 2)],
}


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        neighbors = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in neighbors:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _detect_arrows(grid: np.ndarray, arrow_color: int) -> list[dict[str, object]]:
    arrows: list[dict[str, object]] = []
    for comp in _components(grid == arrow_color, connectivity=4):
        coords = {(int(r), int(c)) for r, c in comp}
        if len(coords) != 4:
            continue
        min_r = min(r for r, _c in coords)
        min_c = min(c for _r, c in coords)
        norm = frozenset((r - min_r, c - min_c) for r, c in coords)
        direction = None
        for name, shape in _ARROW_SHAPES.items():
            if norm == shape:
                direction = name
                break
        if direction is None:
            continue
        tip_dr, tip_dc = _ARROW_TIP_OFFSETS[direction]
        arrows.append(
            {
                "direction": direction,
                "coords": sorted(coords),
                "tip": (int(min_r + tip_dr), int(min_c + tip_dc)),
            }
        )
    return arrows


def _detect_ribbons(grid: np.ndarray, bg: int, arrow_color: int) -> list[dict[str, object]]:
    mask = (grid != bg) & (grid != arrow_color)
    ribbons: list[dict[str, object]] = []
    for comp in _components(mask, connectivity=4):
        rows = sorted({int(r) for r, _c in comp})
        cols = sorted({int(c) for _r, c in comp})
        if len(comp) < 2:
            continue
        if len(rows) == 1:
            row = int(rows[0])
            cols_sorted = sorted(int(c) for _r, c in comp)
            if cols_sorted != list(range(cols_sorted[0], cols_sorted[-1] + 1)):
                continue
            cells = [(row, int(c)) for c in cols_sorted]
            colors = [int(grid[row, c]) for c in cols_sorted]
            ribbons.append(
                {
                    "orientation": "h",
                    "cells": cells,
                    "colors": colors,
                    "row": row,
                    "c0": int(cols_sorted[0]),
                    "c1": int(cols_sorted[-1]),
                }
            )
        elif len(cols) == 1:
            col = int(cols[0])
            rows_sorted = sorted(int(r) for r, _c in comp)
            if rows_sorted != list(range(rows_sorted[0], rows_sorted[-1] + 1)):
                continue
            cells = [(int(r), col) for r in rows_sorted]
            colors = [int(grid[r, col]) for r in rows_sorted]
            ribbons.append(
                {
                    "orientation": "v",
                    "cells": cells,
                    "colors": colors,
                    "col": col,
                    "r0": int(rows_sorted[0]),
                    "r1": int(rows_sorted[-1]),
                }
            )
    return ribbons


def _ordered_ribbon_for_arrow(ribbon: dict[str, object], arrow: dict[str, object]) -> tuple[list[tuple[int, int]], list[int], int]:
    direction = str(arrow["direction"])
    tip_r, tip_c = map(int, arrow["tip"])
    cells = list(ribbon["cells"])
    colors = list(ribbon["colors"])
    n = len(cells)

    if direction == "right":
        ordered_cells = cells
        ordered_colors = colors
        k = int(tip_c - int(ribbon["c0"]) + 1)
    elif direction == "left":
        ordered_cells = list(reversed(cells))
        ordered_colors = list(reversed(colors))
        k = int(int(ribbon["c1"]) - tip_c + 1)
    elif direction == "down":
        ordered_cells = cells
        ordered_colors = colors
        k = int(tip_r - int(ribbon["r0"]) + 1)
    else:
        ordered_cells = list(reversed(cells))
        ordered_colors = list(reversed(colors))
        k = int(int(ribbon["r1"]) - tip_r + 1)

    assert 1 <= k < n
    return ordered_cells, ordered_colors, k


def _fold_ordered_ribbon(
    ordered_cells: list[tuple[int, int]], ordered_colors: list[int], k: int
) -> dict[tuple[int, int], int]:
    n = len(ordered_cells)
    dr = int(ordered_cells[1][0] - ordered_cells[0][0])
    dc = int(ordered_cells[1][1] - ordered_cells[0][1])
    base_r, base_c = map(int, ordered_cells[0])

    out: dict[tuple[int, int], int] = {}
    for t in range(k, n):
        rr = int(base_r + dr * t)
        cc = int(base_c + dc * t)
        out[(rr, cc)] = int(ordered_colors[t])
    for i in range(k):
        t = int(2 * k - 1 - i)
        rr = int(base_r + dr * t)
        cc = int(base_c + dc * t)
        out[(rr, cc)] = int(ordered_colors[i])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each straight multi-color ribbon across the arrow-indicated axis.

    The implementation treats each ribbon as a one-cell-thick straight line of
    2-5 differently colored unit segments. The arrow direction identifies which
    end is folded, and the arrow tip aligns with the last segment on the folded
    side. Folding reverses that side's visible order and overlays it onto the
    opposite side.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)

    ribbons = _detect_ribbons(g, bg, ARROW_COLOR)
    arrows = _detect_arrows(g, ARROW_COLOR)

    out = np.full((h, w), bg, dtype=int)
    used_ribbons: set[int] = set()

    for arrow in arrows:
        direction = str(arrow["direction"])
        tip_r, tip_c = map(int, arrow["tip"])
        arrow_rows = [int(r) for r, _c in arrow["coords"]]
        arrow_cols = [int(c) for _r, c in arrow["coords"]]

        candidates: list[tuple[int, int]] = []
        for idx, ribbon in enumerate(ribbons):
            if idx in used_ribbons:
                continue
            if direction in {"right", "left"} and ribbon["orientation"] == "h":
                row = int(ribbon["row"])
                if int(ribbon["c0"]) <= tip_c <= int(ribbon["c1"]):
                    if max(arrow_rows) < row or min(arrow_rows) > row:
                        dist = min(abs(row - rr) for rr in arrow_rows)
                        candidates.append((dist, idx))
            if direction in {"up", "down"} and ribbon["orientation"] == "v":
                col = int(ribbon["col"])
                if int(ribbon["r0"]) <= tip_r <= int(ribbon["r1"]):
                    if max(arrow_cols) < col or min(arrow_cols) > col:
                        dist = min(abs(col - cc) for cc in arrow_cols)
                        candidates.append((dist, idx))

        assert candidates, "each arrow must match a ribbon"
        candidates.sort()
        ribbon_idx = int(candidates[0][1])
        used_ribbons.add(ribbon_idx)

        ordered_cells, ordered_colors, k = _ordered_ribbon_for_arrow(ribbons[ribbon_idx], arrow)
        folded = _fold_ordered_ribbon(ordered_cells, ordered_colors, k)
        for (r, c), color in folded.items():
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = int(color)

    return out


def _sample_fold_count(length: int) -> int:
    branches: list[str] = []
    if any(k < length - k for k in range(1, length)):
        branches.append("short")
    if any(k == length - k for k in range(1, length)):
        branches.append("equal")
    if any(k > length - k for k in range(1, length)):
        branches.append("long")
    branch = random.choice(branches)
    if branch == "short":
        ks = [k for k in range(1, length) if k < length - k]
    elif branch == "equal":
        ks = [k for k in range(1, length) if k == length - k]
    else:
        ks = [k for k in range(1, length) if k > length - k]
    return int(random.choice(ks))


def _build_spec(
    start: tuple[int, int], direction: str, length: int, k: int, colors: list[int], perp: tuple[int, int]
) -> dict[str, object]:
    dr, dc = _STEP[direction]
    ordered_cells = [
        (int(start[0] + dr * t), int(start[1] + dc * t))
        for t in range(length)
    ]
    tip = (
        int(start[0] + dr * (k - 1) + perp[0]),
        int(start[1] + dc * (k - 1) + perp[1]),
    )
    arrow_cells = [
        (int(tip[0] + off_r), int(tip[1] + off_c))
        for off_r, off_c in _ARROW_OFFSETS[direction]
    ]
    folded = _fold_ordered_ribbon(ordered_cells, colors, k)
    return {
        "direction": direction,
        "ordered_cells": ordered_cells,
        "colors": list(colors),
        "k": int(k),
        "tip": tip,
        "arrow_cells": arrow_cells,
        "input_cells": set(ordered_cells) | set(arrow_cells),
        "output_cells": dict(folded),
    }


def _in_bounds(cells: list[tuple[int, int]] | set[tuple[int, int]] | dict[tuple[int, int], int], h: int, w: int) -> bool:
    items = cells.keys() if isinstance(cells, dict) else cells
    return all(0 <= int(r) < h and 0 <= int(c) < w for r, c in items)


def _expand(cells: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _render_input(h: int, w: int, specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for spec in specs:
        for (r, c), color in zip(spec["ordered_cells"], spec["colors"]):
            grid[int(r), int(c)] = int(color)
        for r, c in spec["arrow_cells"]:
            grid[int(r), int(c)] = ARROW_COLOR
    return grid


def _render_output(h: int, w: int, specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for spec in specs:
        for (r, c), color in spec["output_cells"].items():
            grid[int(r), int(c)] = int(color)
    return grid


def _sample_object(h: int, w: int, input_forbidden: set[tuple[int, int]], output_forbidden: set[tuple[int, int]]) -> dict[str, object] | None:
    for _ in range(OBJECT_RETRIES):
        direction = random.choice(["right", "left", "down", "up"])
        length = random.randint(2, 5)
        k = _sample_fold_count(length)
        colors = random.sample(RIBBON_COLORS, length)
        perp = random.choice(_PERP_CHOICES[direction])
        start = (random.randrange(h), random.randrange(w))
        spec = _build_spec(start, direction, length, k, colors, perp)

        if not _in_bounds(spec["input_cells"], h, w):
            continue
        if not _in_bounds(spec["output_cells"], h, w):
            continue
        input_cells = set(spec["input_cells"])
        output_cells = set(spec["output_cells"])
        if input_cells & input_forbidden:
            continue
        if output_cells & output_forbidden:
            continue
        return spec
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(14, 24)
        w = random.randint(14, 24)
        target_count = random.randint(2, 5)

        specs: list[dict[str, object]] = []
        input_forbidden: set[tuple[int, int]] = set()
        output_forbidden: set[tuple[int, int]] = set()

        for _idx in range(target_count):
            spec = _sample_object(h, w, input_forbidden, output_forbidden)
            if spec is None:
                specs = []
                break
            specs.append(spec)
            input_forbidden |= _expand(set(spec["input_cells"]), h, w, radius=1)
            output_forbidden |= _expand(set(spec["output_cells"]), h, w, radius=1)

        if len(specs) != target_count:
            continue

        dirs = Counter(str(spec["direction"]) for spec in specs)
        if len(specs) >= 3 and len(dirs) == 1:
            continue

        inp = _render_input(h, w, specs)
        expected = _render_output(h, w, specs)
        try:
            out = main(inp)
        except AssertionError:
            continue
        if not np.array_equal(out, expected):
            continue
        return inp

    raise RuntimeError("failed to generate valid ribbon-fold input")
