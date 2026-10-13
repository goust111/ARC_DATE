from __future__ import annotations

import random
from collections import Counter

import numpy as np


GREEN = 3

BG_COLOR = 0
DIAGONAL_COLOR = 5
GEN_OBJECT_COLORS = [1, 2, 4, 6, 7, 8, 9]
GEN_GRID_MIN = 12
GEN_GRID_MAX = 20
GEN_RETRIES = 400

UPPER = "upper"
LOWER = "lower"

SHAPE_LIBRARY = [
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _normalize_cells(cells: set[tuple[int, int]] | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _shape_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True

    variants = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        variants.add(_normalize_cells([tuple(map(int, rc)) for rc in np.argwhere(rot)]))
        flip = np.fliplr(rot)
        variants.add(_normalize_cells([tuple(map(int, rc)) for rc in np.argwhere(flip)]))
    return list(variants)


ALL_SHAPES = [variant for base in SHAPE_LIBRARY for variant in _shape_variants(base)]


def _most_common(values: list[int]) -> int:
    return int(Counter(values).most_common(1)[0][0])


def _off_diagonal_bg(grid: np.ndarray) -> int:
    n = grid.shape[0]
    vals = [int(grid[r, c]) for r in range(n) for c in range(n) if r != c]
    return _most_common(vals)


def _side_of(cells: tuple[tuple[int, int], ...]) -> str:
    side = None
    for r, c in cells:
        if r == c:
            raise AssertionError("objects must stay off the diagonal")
        cur = UPPER if c > r else LOWER
        if side is None:
            side = cur
        elif side != cur:
            raise AssertionError("object may not cross the diagonal")
    assert side is not None
    return side


def _side_distance(cells: tuple[tuple[int, int], ...], side: str) -> int:
    if side == UPPER:
        return min(c - r for r, c in cells)
    return min(r - c for r, c in cells)


def _step(side: str, inward: bool) -> tuple[int, int]:
    if side == UPPER:
        return (1, -1) if inward else (-1, 1)
    return (-1, 1) if inward else (1, -1)


def _shift_cells(cells: tuple[tuple[int, int], ...] | set[tuple[int, int]], dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r + dr), int(c + dc)) for r, c in cells))


def _in_bounds(n: int, r: int, c: int) -> bool:
    return 0 <= r < n and 0 <= c < n


def _cells_on_side(cells: tuple[tuple[int, int], ...], side: str) -> bool:
    if side == UPPER:
        return all(c > r for r, c in cells)
    return all(r > c for r, c in cells)


def _extract_objects(grid: np.ndarray, bg: int, diagonal_color: int) -> list[dict]:
    n = grid.shape[0]
    visited = np.zeros_like(grid, dtype=bool)
    objects = []
    for r in range(n):
        for c in range(n):
            if r == c or visited[r, c]:
                continue
            color = int(grid[r, c])
            if color in (bg, diagonal_color):
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not _in_bounds(n, nr, nc) or nr == nc or visited[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            cells = tuple(sorted((int(rr), int(cc)) for rr, cc in coords))
            objects.append(
                {
                    "color": color,
                    "cells": cells,
                    "side": _side_of(cells),
                }
            )
    return objects


def _can_step(cells: tuple[tuple[int, int], ...], side: str, blocked: set[tuple[int, int]], n: int) -> bool:
    dr, dc = _step(side, inward=True)
    moved = []
    for r, c in cells:
        nr = r + dr
        nc = c + dc
        if not _in_bounds(n, nr, nc):
            return False
        moved.append((nr, nc))
    moved_cells = tuple(sorted(moved))
    if not _cells_on_side(moved_cells, side):
        return False
    return all(cell not in blocked for cell in moved_cells)


def _settle_side(objects: list[dict], n: int) -> list[dict]:
    blocked = {(i, i) for i in range(n)}
    settled = []
    for obj in sorted(
        objects,
        key=lambda item: (
            _side_distance(item["cells"], item["side"]),
            min(r for r, _ in item["cells"]),
            min(c for _, c in item["cells"]),
            len(item["cells"]),
        ),
    ):
        cells = obj["cells"]
        while _can_step(cells, obj["side"], blocked, n):
            dr, dc = _step(obj["side"], inward=True)
            cells = _shift_cells(cells, dr, dc)
        blocked.update(cells)
        settled.append(
            {
                "color": obj["color"],
                "side": obj["side"],
                "cells": cells,
            }
        )
    return settled


def _settle_objects(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"
    n = g.shape[0]
    diagonal = [int(g[i, i]) for i in range(n)]
    assert len(set(diagonal)) == 1, "expected a full main diagonal line"
    diagonal_color = diagonal[0]
    bg = _off_diagonal_bg(g)

    objects = _extract_objects(g, bg, diagonal_color)
    upper = [obj for obj in objects if obj["side"] == UPPER]
    lower = [obj for obj in objects if obj["side"] == LOWER]
    return bg, diagonal_color, _settle_side(upper, n) + _settle_side(lower, n)


def _paired_indices(objects: list[dict]) -> set[int]:
    lower_lookup = {
        frozenset(obj["cells"]): idx
        for idx, obj in enumerate(objects)
        if obj["side"] == LOWER
    }
    paired = set()
    for idx, obj in enumerate(objects):
        if obj["side"] != UPPER:
            continue
        mirrored = frozenset((c, r) for r, c in obj["cells"])
        mate = lower_lookup.get(mirrored)
        if mate is not None:
            paired.add(idx)
            paired.add(mate)
    return paired


def main(grid: np.ndarray) -> np.ndarray:
    """Slide every off-diagonal object toward the main diagonal, then recolor transposed cross-diagonal matches green.

    Input structure:
    - The grid is square and contains a complete gray main diagonal.
    - Monochrome foreground objects lie strictly on either side of that diagonal.
    - The remaining off-diagonal cells are background.

    Rule:
    - Settle upper-side objects by repeatedly moving them one step down-left, and lower-side objects one step up-right.
    - On each side, objects with smaller diagonal distance settle first and later objects stop when the next step would cross the diagonal or hit an earlier settled object.
    - After all objects settle, recolor every upper/lower pair whose settled cell sets match exactly under transpose across the main diagonal; keep all other colors and the diagonal unchanged.

    Key insight:
    - The task is a side-wise diagonal-gravity simulation followed by an exact transpose test on the settled objects, not a direct comparison of the scattered input positions.
    """
    bg, diagonal_color, settled = _settle_objects(grid)
    n = grid.shape[0]
    out = np.full((n, n), bg, dtype=int)
    out[np.arange(n), np.arange(n)] = diagonal_color

    paired = _paired_indices(settled)
    for idx, obj in enumerate(settled):
        color = GREEN if idx in paired else int(obj["color"])
        for r, c in obj["cells"]:
            out[r, c] = color
    return out


def _place_shape(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((r0 + r, c0 + c) for r, c in shape))


def _max_outward_shift(cells: tuple[tuple[int, int], ...], side: str, n: int) -> int:
    if side == UPPER:
        return min(min(r for r, _ in cells), min(n - 1 - c for _, c in cells))
    return min(min(n - 1 - r for r, _ in cells), min(c for _, c in cells))


def _blocked_by_diagonal(cells: tuple[tuple[int, int], ...], side: str) -> bool:
    dr, dc = _step(side, inward=True)
    moved = _shift_cells(cells, dr, dc)
    return not _cells_on_side(moved, side)


def _valid_front_cells(
    cells: tuple[tuple[int, int], ...],
    side: str,
    occupied: set[tuple[int, int]],
    n: int,
) -> bool:
    if not cells:
        return False
    if not all(_in_bounds(n, r, c) for r, c in cells):
        return False
    if not _cells_on_side(cells, side):
        return False
    if any(cell in occupied or cell[0] == cell[1] for cell in cells):
        return False
    if _max_outward_shift(cells, side, n) < 1:
        return False
    if not _blocked_by_diagonal(cells, side):
        return False
    dr, dc = _step(side, inward=True)
    moved = _shift_cells(cells, dr, dc)
    return all(cell not in occupied for cell in moved)


def _valid_stacked_cells(
    cells: tuple[tuple[int, int], ...],
    side: str,
    occupied: set[tuple[int, int]],
    n: int,
) -> bool:
    if not cells:
        return False
    if not all(_in_bounds(n, r, c) for r, c in cells):
        return False
    if not _cells_on_side(cells, side):
        return False
    if any(cell in occupied or cell[0] == cell[1] for cell in cells):
        return False
    if _max_outward_shift(cells, side, n) < 1:
        return False
    if _can_step(cells, side, occupied, n):
        return False
    dr, dc = _step(side, inward=True)
    moved = _shift_cells(cells, dr, dc)
    return any(cell in occupied for cell in moved)


def _transpose_cells(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((c, r) for r, c in cells))


def _sample_front_pair(
    n: int,
    occupied_upper: set[tuple[int, int]],
    occupied_lower: set[tuple[int, int]],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]] | None:
    for _ in range(250):
        shape = random.choice(ALL_SHAPES)
        h = max(r for r, _ in shape) + 1
        w = max(c for _, c in shape) + 1
        r0 = random.randint(0, n - h)
        c0 = random.randint(0, n - w)
        upper = _place_shape(shape, r0, c0)
        lower = _transpose_cells(upper)
        if _valid_front_cells(upper, UPPER, occupied_upper, n) and _valid_front_cells(lower, LOWER, occupied_lower, n):
            return upper, lower
    return None


def _sample_front_single(
    side: str,
    n: int,
    occupied: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...] | None:
    for _ in range(250):
        shape = random.choice(ALL_SHAPES)
        h = max(r for r, _ in shape) + 1
        w = max(c for _, c in shape) + 1
        r0 = random.randint(0, n - h)
        c0 = random.randint(0, n - w)
        cells = _place_shape(shape, r0, c0)
        if _valid_front_cells(cells, side, occupied, n):
            return cells
    return None


def _sample_stacked_from_base(
    base_cells: tuple[tuple[int, int], ...],
    side: str,
    occupied: set[tuple[int, int]],
    n: int,
) -> tuple[tuple[int, int], ...] | None:
    dr, dc = _step(side, inward=False)
    for dist in (1, 2):
        cells = _shift_cells(base_cells, dr * dist, dc * dist)
        if _valid_stacked_cells(cells, side, occupied, n):
            return cells
    return None


def _render_grid(n: int, objects: list[dict]) -> np.ndarray:
    grid = np.full((n, n), BG_COLOR, dtype=int)
    grid[np.arange(n), np.arange(n)] = DIAGONAL_COLOR
    for obj in objects:
        for r, c in obj["cells"]:
            grid[r, c] = int(obj["color"])
    return grid


def _count_stack_stops(settled: list[dict]) -> int:
    by_side = {UPPER: [], LOWER: []}
    for obj in settled:
        by_side[obj["side"]].append(obj)

    n = max(max(r, c) for obj in settled for r, c in obj["cells"]) + 1
    count = 0
    for side, items in by_side.items():
        blocked = {(i, i) for i in range(n)}
        prior_objects: set[tuple[int, int]] = set()
        ordered = sorted(
            items,
            key=lambda item: (
                _side_distance(item["cells"], item["side"]),
                min(r for r, _ in item["cells"]),
                min(c for _, c in item["cells"]),
            ),
        )
        dr, dc = _step(side, inward=True)
        for obj in ordered:
            moved = _shift_cells(obj["cells"], dr, dc)
            if any(cell in prior_objects for cell in moved):
                count += 1
            blocked.update(obj["cells"])
            prior_objects.update(obj["cells"])
    return count


def _build_final_layout() -> tuple[np.ndarray, int, bool, bool] | None:
    n = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    want_pair = random.random() < 0.65
    want_stack = random.random() < 0.55

    objects: list[dict] = []
    front_upper: list[tuple[tuple[int, int], ...]] = []
    front_lower: list[tuple[tuple[int, int], ...]] = []
    occupied_upper: set[tuple[int, int]] = set()
    occupied_lower: set[tuple[int, int]] = set()
    intended_pairs = 0

    if want_pair:
        pair = _sample_front_pair(n, occupied_upper, occupied_lower)
        if pair is None:
            return None
        upper, lower = pair
        front_upper.append(upper)
        front_lower.append(lower)
        occupied_upper.update(upper)
        occupied_lower.update(lower)
        objects.extend(
            [
                {"side": UPPER, "cells": upper},
                {"side": LOWER, "cells": lower},
            ]
        )
        intended_pairs += 1
    else:
        upper = _sample_front_single(UPPER, n, occupied_upper)
        lower = _sample_front_single(LOWER, n, occupied_lower)
        if upper is None or lower is None:
            return None
        front_upper.append(upper)
        front_lower.append(lower)
        occupied_upper.update(upper)
        occupied_lower.update(lower)
        objects.extend(
            [
                {"side": UPPER, "cells": upper},
                {"side": LOWER, "cells": lower},
            ]
        )

    extra_front = random.randint(0, 2)
    for _ in range(extra_front):
        side = random.choice([UPPER, LOWER])
        if side == UPPER:
            cells = _sample_front_single(UPPER, n, occupied_upper)
            if cells is None:
                continue
            front_upper.append(cells)
            occupied_upper.update(cells)
        else:
            cells = _sample_front_single(LOWER, n, occupied_lower)
            if cells is None:
                continue
            front_lower.append(cells)
            occupied_lower.update(cells)
        objects.append({"side": side, "cells": cells})

    if want_stack:
        if want_pair and random.random() < 0.65:
            candidates = list(zip(front_upper, front_lower))
            random.shuffle(candidates)
            added = False
            for upper_base, lower_base in candidates:
                upper = _sample_stacked_from_base(upper_base, UPPER, occupied_upper, n)
                lower = _sample_stacked_from_base(lower_base, LOWER, occupied_lower, n)
                if upper is None or lower is None:
                    continue
                occupied_upper.update(upper)
                occupied_lower.update(lower)
                objects.extend(
                    [
                        {"side": UPPER, "cells": upper},
                        {"side": LOWER, "cells": lower},
                    ]
                )
                intended_pairs += 1
                added = True
                break
            want_stack = added
        else:
            sides = [UPPER, LOWER]
            random.shuffle(sides)
            added = False
            for side in sides:
                bases = front_upper if side == UPPER else front_lower
                occ = occupied_upper if side == UPPER else occupied_lower
                random.shuffle(bases)
                for base in bases:
                    cells = _sample_stacked_from_base(base, side, occ, n)
                    if cells is None:
                        continue
                    occ.update(cells)
                    objects.append({"side": side, "cells": cells})
                    added = True
                    break
                if added:
                    break
            want_stack = added

    if len(objects) > len(GEN_OBJECT_COLORS):
        return None

    colors = random.sample(GEN_OBJECT_COLORS, len(objects))
    for obj, color in zip(objects, colors):
        obj["color"] = color

    final_grid = _render_grid(n, objects)
    bg, diagonal_color, settled = _settle_objects(final_grid)
    if bg != BG_COLOR or diagonal_color != DIAGONAL_COLOR:
        return None
    original_sets = sorted((obj["side"], tuple(sorted(obj["cells"]))) for obj in objects)
    settled_sets = sorted((obj["side"], tuple(sorted(obj["cells"]))) for obj in settled)
    if original_sets != settled_sets:
        return None

    pair_count = len(_paired_indices(settled)) // 2
    if want_pair != (pair_count > 0):
        return None
    if pair_count != intended_pairs:
        return None

    stack_count = _count_stack_stops(settled)
    if want_stack and stack_count == 0:
        return None
    if not want_stack and stack_count > 0:
        return None

    return final_grid, n, want_pair, want_stack


def _reverse_shift(final_grid: np.ndarray) -> np.ndarray | None:
    bg, diagonal_color, settled = _settle_objects(final_grid)
    n = final_grid.shape[0]
    by_side = {UPPER: [], LOWER: []}
    for obj in settled:
        by_side[obj["side"]].append(obj)

    if not by_side[UPPER] or not by_side[LOWER]:
        return None

    upper_shift = random.randint(1, min(_max_outward_shift(obj["cells"], UPPER, n) for obj in by_side[UPPER]))
    lower_shift = random.randint(1, min(_max_outward_shift(obj["cells"], LOWER, n) for obj in by_side[LOWER]))

    inp = np.full((n, n), bg, dtype=int)
    inp[np.arange(n), np.arange(n)] = diagonal_color
    for obj in settled:
        dr, dc = _step(obj["side"], inward=False)
        shift = upper_shift if obj["side"] == UPPER else lower_shift
        moved = _shift_cells(obj["cells"], dr * shift, dc * shift)
        if any(inp[r, c] != bg for r, c in moved):
            return None
        for r, c in moved:
            inp[r, c] = int(obj["color"])
    return inp


def _gen_attempt() -> np.ndarray | None:
    built = _build_final_layout()
    if built is None:
        return None
    final_grid, _n, want_pair, want_stack = built
    inp = _reverse_shift(final_grid)
    if inp is None:
        return None
    out = main(inp)
    if not np.array_equal(out, main(final_grid)):
        return None

    _, _, settled = _settle_objects(inp)
    pair_count = len(_paired_indices(settled)) // 2
    stack_count = _count_stack_stops(settled)
    if want_pair != (pair_count > 0):
        return None
    if want_stack and stack_count == 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
