from __future__ import annotations

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5

SPECIAL_COLORS = {BLACK, RED, YELLOW, GREY}

_PATTERN_SPECS: dict[str, dict[str, tuple[tuple[int, int], ...]]] = {
    "horizontal": {
        "cells": ((1, 0), (1, 1), (1, 2)),
        "dirs": ((0, -1), (0, 1)),
    },
    "vertical": {
        "cells": ((0, 1), (1, 1), (2, 1)),
        "dirs": ((-1, 0), (1, 0)),
    },
    "plus": {
        "cells": ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        "dirs": ((-1, 0), (1, 0), (0, -1), (0, 1)),
    },
    "x": {
        "cells": ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2)),
        "dirs": ((-1, -1), (-1, 1), (1, -1), (1, 1)),
    },
    "full": {
        "cells": (
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2),
            (2, 0), (2, 1), (2, 2),
        ),
        "dirs": (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ),
    },
}

_PATTERN_LOOKUP = {
    frozenset(spec["cells"]): spec["dirs"] for spec in _PATTERN_SPECS.values()
}

_Seed = tuple[int, int, int, tuple[tuple[int, int], ...]]


def _decode_seeds(grid: np.ndarray) -> list[_Seed]:
    h, w = grid.shape
    seeds: list[_Seed] = []
    seen_positions: set[tuple[int, int]] = set()

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in SPECIAL_COLORS:
                continue
            if r < 4 or c < 1 or c + 1 >= w:
                continue
            block = grid[r - 4 : r - 1, c - 1 : c + 2]
            if block.shape != (3, 3):
                continue
            if not np.all(grid[r - 1, c - 1 : c + 2] == BLACK):
                continue
            if not np.all((block == GREY) | (block == color)):
                continue

            pattern_cells = frozenset(
                (rr, cc)
                for rr in range(3)
                for cc in range(3)
                if int(block[rr, cc]) == color
            )
            dirs = _PATTERN_LOOKUP.get(pattern_cells)
            if dirs is None:
                continue
            if (r, c) in seen_positions:
                continue
            seeds.append((r, c, color, dirs))
            seen_positions.add((r, c))

    seeds.sort(key=lambda item: (item[0], item[1], item[2]))
    assert 2 <= len(seeds) <= 4, "expected 2-4 seeds"
    assert len({seed[2] for seed in seeds}) == len(seeds), "seed colors must be unique"
    return seeds


def _empty_output(grid: np.ndarray) -> np.ndarray:
    out = np.full_like(grid, BLACK)
    out[grid == RED] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each seed's 3x3 direction glyph, then grow all seeds synchronously.

    Seeds expand only onto original black cells. Grey instruction blocks and red
    obstacles block growth. When multiple seeds first reach the same black cell
    on the same step, that cell becomes yellow and stops propagating further.
    The 3x3 instruction blocks are removed in the final output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    seeds = _decode_seeds(g)
    traversable = g == BLACK

    out = _empty_output(g)
    empty = -1
    tied = -2
    occupied = np.full((h, w), empty, dtype=int)

    frontiers: list[set[tuple[int, int]]] = []
    for seed_id, (r, c, color, _dirs) in enumerate(seeds):
        occupied[r, c] = seed_id
        out[r, c] = color
        frontiers.append({(r, c)})

    while True:
        proposals: dict[tuple[int, int], int] = {}
        for seed_id, frontier in enumerate(frontiers):
            _r0, _c0, _color, dirs = seeds[seed_id]
            for r, c in frontier:
                for dr, dc in dirs:
                    nr = r + dr
                    nc = c + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    if not traversable[nr, nc]:
                        continue
                    if occupied[nr, nc] != empty:
                        continue
                    key = (nr, nc)
                    prev = proposals.get(key)
                    if prev is None:
                        proposals[key] = seed_id
                    elif prev != seed_id:
                        proposals[key] = tied

        if not proposals:
            break

        next_frontiers = [set() for _ in seeds]
        for (r, c), owner in proposals.items():
            if owner == tied:
                occupied[r, c] = tied
                out[r, c] = YELLOW
                continue
            occupied[r, c] = owner
            out[r, c] = seeds[owner][2]
            next_frontiers[owner].add((r, c))
        frontiers = next_frontiers

    return out


# --------------- generator ---------------

_GEN_SEED_COLORS = np.array([1, 3, 6, 7, 8, 9], dtype=int)
_GEN_PATTERN_NAMES = tuple(_PATTERN_SPECS.keys())
_GEN_TIE_MODES = ("horizontal", "vertical", "diag_down", "diag_up")

_GEN_HEIGHT_RANGE = (18, 27)
_GEN_WIDTH_RANGE = (18, 27)
_GEN_SEED_COUNT_RANGE = (2, 5)
_GEN_OBSTACLE_COUNT_RANGE = (3, 10)

_GEN_ROW_MIN = 5
_GEN_COL_MIN = 2
_GEN_SEED_ROW_MARGIN = 2
_GEN_SEED_COL_MARGIN = 2
_GEN_PLACE_RETRIES = 120
_GEN_MAX_ATTEMPTS = 400


def _pattern_patch(color: int, pattern_name: str) -> np.ndarray:
    patch = np.full((3, 3), GREY, dtype=int)
    for r, c in _PATTERN_SPECS[pattern_name]["cells"]:
        patch[r, c] = color
    return patch


def _draw_seed_structure(grid: np.ndarray, seed: tuple[int, int, int, str]) -> None:
    r, c, color, pattern_name = seed
    grid[r - 4 : r - 1, c - 1 : c + 2] = _pattern_patch(color, pattern_name)
    grid[r, c] = color


def _structure_box(seed: tuple[int, int, int, str], pad: int = 1) -> tuple[int, int, int, int]:
    r, c, _color, _pattern_name = seed
    return (r - 4 - pad, c - 1 - pad, r + pad, c + 1 + pad)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _seed_ranges(h: int, w: int) -> tuple[range, range]:
    row_hi = h - _GEN_SEED_ROW_MARGIN
    col_hi = w - _GEN_SEED_COL_MARGIN
    return range(_GEN_ROW_MIN, row_hi), range(_GEN_COL_MIN, col_hi)


def _pick_pattern(rng: np.random.Generator, names: tuple[str, ...]) -> str:
    return str(names[int(rng.integers(0, len(names)))])


def _try_place_seed(
    h: int,
    w: int,
    color: int,
    pattern_name: str,
    boxes: list[tuple[int, int, int, int]],
    rng: np.random.Generator,
    *,
    fixed_r: int | None = None,
    fixed_c: int | None = None,
    banned_box: tuple[int, int, int, int] | None = None,
    unique_line: set[int] | None = None,
    line_axis: str | None = None,
) -> tuple[int, int, int, str] | None:
    row_range, col_range = _seed_ranges(h, w)
    for _ in range(_GEN_PLACE_RETRIES):
        r = fixed_r if fixed_r is not None else int(rng.choice(tuple(row_range)))
        c = fixed_c if fixed_c is not None else int(rng.choice(tuple(col_range)))
        if unique_line is not None:
            key = r if line_axis == "row" else c
            if key in unique_line:
                continue
        seed = (r, c, color, pattern_name)
        box = _structure_box(seed)
        if any(_boxes_overlap(box, other) for other in boxes):
            continue
        if banned_box is not None and _boxes_overlap(box, banned_box):
            continue
        return seed
    return None


def _corridor_box(
    a: tuple[int, int],
    b: tuple[int, int],
    h: int,
    w: int,
    pad: int,
) -> tuple[int, int, int, int]:
    r0 = max(0, min(a[0], b[0]) - pad)
    c0 = max(0, min(a[1], b[1]) - pad)
    r1 = min(h - 1, max(a[0], b[0]) + pad)
    c1 = min(w - 1, max(a[1], b[1]) + pad)
    return (r0, c0, r1, c1)


def _tie_pair(
    h: int,
    w: int,
    colors: list[int],
    mode: str,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int, int, str]], tuple[int, int, int, int], set[tuple[int, int]]] | None:
    if mode == "horizontal":
        compatible = ("horizontal", "plus", "full")
        distances = [d for d in (6, 8, 10) if d <= w - 6]
    elif mode == "vertical":
        compatible = ("vertical", "plus", "full")
        distances = [d for d in (8, 10) if d <= h - 8]
    else:
        compatible = ("x", "full")
        distances = [d for d in (8, 10) if d <= min(h - 8, w - 6)]
    if not distances:
        return None

    row_range, col_range = _seed_ranges(h, w)
    row_vals = tuple(row_range)
    col_vals = tuple(col_range)
    for _ in range(_GEN_PLACE_RETRIES):
        dist = int(distances[int(rng.integers(0, len(distances)))])
        pattern_a = _pick_pattern(rng, compatible)
        pattern_b = _pick_pattern(rng, compatible)

        if mode == "horizontal":
            r = int(row_vals[int(rng.integers(0, len(row_vals)))])
            c1_max = w - _GEN_SEED_COL_MARGIN - dist
            if c1_max <= _GEN_COL_MIN:
                continue
            c1 = int(rng.integers(_GEN_COL_MIN, c1_max))
            c2 = c1 + dist
            seed_a = (r, c1, colors[0], pattern_a)
            seed_b = (r, c2, colors[1], pattern_b)
            protected = {(r, c) for c in range(c1 + 1, c2)}
        elif mode == "vertical":
            c = int(col_vals[int(rng.integers(0, len(col_vals)))])
            r1_max = h - _GEN_SEED_ROW_MARGIN - dist
            if r1_max <= _GEN_ROW_MIN:
                continue
            r1 = int(rng.integers(_GEN_ROW_MIN, r1_max))
            r2 = r1 + dist
            seed_a = (r1, c, colors[0], pattern_a)
            seed_b = (r2, c, colors[1], pattern_b)
            protected = {(r, c) for r in range(r1 + 1, r2)}
        elif mode == "diag_down":
            r1_max = h - _GEN_SEED_ROW_MARGIN - dist
            c1_max = w - _GEN_SEED_COL_MARGIN - dist
            if r1_max <= _GEN_ROW_MIN or c1_max <= _GEN_COL_MIN:
                continue
            r1 = int(rng.integers(_GEN_ROW_MIN, r1_max))
            c1 = int(rng.integers(_GEN_COL_MIN, c1_max))
            seed_a = (r1, c1, colors[0], pattern_a)
            seed_b = (r1 + dist, c1 + dist, colors[1], pattern_b)
            protected = {(r1 + k, c1 + k) for k in range(1, dist)}
        else:
            r1_max = h - _GEN_SEED_ROW_MARGIN - dist
            c1_min = _GEN_COL_MIN + dist
            c1_max = w - _GEN_SEED_COL_MARGIN
            if r1_max <= _GEN_ROW_MIN or c1_min >= c1_max:
                continue
            r1 = int(rng.integers(_GEN_ROW_MIN, r1_max))
            c1 = int(rng.integers(c1_min, c1_max))
            seed_a = (r1, c1, colors[0], pattern_a)
            seed_b = (r1 + dist, c1 - dist, colors[1], pattern_b)
            protected = {(r1 + k, c1 - k) for k in range(1, dist)}

        box_a = _structure_box(seed_a)
        box_b = _structure_box(seed_b)
        if _boxes_overlap(box_a, box_b):
            continue
        banned_box = _corridor_box(
            (seed_a[0], seed_a[1]),
            (seed_b[0], seed_b[1]),
            h,
            w,
            pad=3,
        )
        return [seed_a, seed_b], banned_box, protected
    return None


def _add_parallel_extras(
    h: int,
    w: int,
    colors: list[int],
    specs: list[tuple[int, int, int, str]],
    boxes: list[tuple[int, int, int, int]],
    rng: np.random.Generator,
    banned_box: tuple[int, int, int, int] | None,
    family: str,
) -> list[tuple[int, int, int, str]] | None:
    assert family in {"horizontal", "vertical"}
    unique_line = {seed[0] if family == "horizontal" else seed[1] for seed in specs}
    for color in colors:
        pattern_name = family
        seed = _try_place_seed(
            h,
            w,
            color,
            pattern_name,
            boxes,
            rng,
            banned_box=banned_box,
            unique_line=unique_line,
            line_axis="row" if family == "horizontal" else "col",
        )
        if seed is None:
            return None
        specs.append(seed)
        boxes.append(_structure_box(seed))
        unique_line.add(seed[0] if family == "horizontal" else seed[1])
    return specs


def _render_input(
    h: int,
    w: int,
    specs: list[tuple[int, int, int, str]],
    obstacle_count: int,
    protected: set[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray | None:
    grid = np.full((h, w), BLACK, dtype=int)
    for seed in specs:
        _draw_seed_structure(grid, seed)

    protected_mask = np.zeros((h, w), dtype=bool)
    for r, c in protected:
        protected_mask[r, c] = True

    placed = 0
    for _ in range(obstacle_count):
        candidates: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                if grid[r, c] != BLACK or protected_mask[r, c]:
                    continue
                r0 = max(0, r - 1)
                r1 = min(h, r + 2)
                c0 = max(0, c - 1)
                c1 = min(w, c + 2)
                patch = grid[r0:r1, c0:c1]
                if np.count_nonzero(patch != BLACK) != 0:
                    continue
                candidates.append((r, c))
        if not candidates:
            break
        r, c = candidates[int(rng.integers(0, len(candidates)))]
        grid[r, c] = RED
        placed += 1

    if placed < 3:
        return None
    return grid


def _gen_no_tie_layout(
    h: int,
    w: int,
    colors: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    family = "horizontal" if int(rng.integers(0, 2)) == 0 else "vertical"
    specs: list[tuple[int, int, int, str]] = []
    boxes: list[tuple[int, int, int, int]] = []
    specs = _add_parallel_extras(h, w, colors, specs, boxes, rng, None, family)
    if specs is None:
        return None
    obstacle_count = int(rng.integers(*_GEN_OBSTACLE_COUNT_RANGE))
    return _render_input(h, w, specs, obstacle_count, set(), rng)


def _gen_tie_layout(
    h: int,
    w: int,
    colors: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    modes = list(_GEN_TIE_MODES)
    rng.shuffle(modes)
    for mode in modes:
        pair_info = _tie_pair(h, w, colors[:2], mode, rng)
        if pair_info is None:
            continue
        specs, banned_box, protected = pair_info
        boxes = [_structure_box(seed) for seed in specs]

        remaining_colors = colors[2:]
        if remaining_colors:
            if mode == "horizontal":
                family = "horizontal"
            elif mode == "vertical":
                family = "vertical"
            else:
                family = "horizontal" if int(rng.integers(0, 2)) == 0 else "vertical"
            specs_copy = list(specs)
            boxes_copy = list(boxes)
            specs_copy = _add_parallel_extras(
                h, w, remaining_colors, specs_copy, boxes_copy, rng, banned_box, family
            )
            if specs_copy is None:
                continue
            specs = specs_copy

        obstacle_count = int(rng.integers(*_GEN_OBSTACLE_COUNT_RANGE))
        grid = _render_input(h, w, specs, obstacle_count, protected, rng)
        if grid is not None:
            return grid
    return None


def _is_valid_example(grid: np.ndarray, expect_tie: bool) -> bool:
    try:
        seeds = _decode_seeds(grid)
        out = main(grid)
    except Exception:
        return False

    if out.shape != grid.shape:
        return False
    if np.count_nonzero(grid == RED) < 3:
        return False
    has_tie = bool(np.any(out == YELLOW))
    if has_tie != expect_tie:
        return False
    for _r, _c, color, _dirs in seeds:
        if int(np.count_nonzero(out == color)) <= 1:
            return False
    colored_growth = int(np.count_nonzero((out != BLACK) & (out != RED)))
    if colored_growth <= len(seeds):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_MAX_ATTEMPTS):
        h = int(rng.integers(*_GEN_HEIGHT_RANGE))
        w = int(rng.integers(*_GEN_WIDTH_RANGE))
        seed_count = int(rng.integers(*_GEN_SEED_COUNT_RANGE))
        colors = [int(v) for v in rng.choice(_GEN_SEED_COLORS, size=seed_count, replace=False)]
        expect_tie = bool(int(rng.integers(0, 2)))

        if expect_tie:
            grid = _gen_tie_layout(h, w, colors, rng)
        else:
            grid = _gen_no_tie_layout(h, w, colors, rng)
        if grid is None:
            continue
        if _is_valid_example(grid, expect_tie):
            return grid
    raise RuntimeError(f"generate_input failed after {_GEN_MAX_ATTEMPTS} attempts")
