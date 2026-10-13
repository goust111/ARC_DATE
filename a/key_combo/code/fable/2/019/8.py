from __future__ import annotations

import random
from collections import Counter

import numpy as np


OP_NAMES = (
    "id",
    "rot90",
    "rot180",
    "rot270",
    "mirror_v",
    "mirror_h",
    "mirror_diag",
    "mirror_anti",
)

GROUP_FAMILIES = [
    ("rot180", (0, 2)),
    ("mirror_v", (0, 4)),
    ("mirror_h", (0, 5)),
    ("mirror_diag", (0, 6)),
    ("mirror_anti", (0, 7)),
    ("quarter_turn", (0, 1, 2, 3)),
    ("axial_klein", (0, 2, 4, 5)),
    ("full_d4", tuple(range(8))),
]

MIN_MASK_CELLS = 3
TRANSFORM_CACHE: dict[int, tuple[tuple[np.ndarray, np.ndarray], ...]] = {}
ORBIT_CACHE: dict[tuple[int, tuple[int, ...]], tuple[tuple[tuple[int, int], ...], ...]] = {}
COMPOSITION_TABLE: tuple[tuple[int, ...], ...] | None = None
GROUP_BAG: list[tuple[str, tuple[int, ...]]] = []


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _apply_op(op: int, side: int, r: int, c: int) -> tuple[int, int]:
    last = int(side) - 1
    if op == 0:
        return int(r), int(c)
    if op == 1:
        return int(c), int(last - r)
    if op == 2:
        return int(last - r), int(last - c)
    if op == 3:
        return int(last - c), int(r)
    if op == 4:
        return int(r), int(last - c)
    if op == 5:
        return int(last - r), int(c)
    if op == 6:
        return int(c), int(r)
    if op == 7:
        return int(last - c), int(last - r)
    raise ValueError(f"unknown operation {op}")


def _transform_maps(side: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    side = int(side)
    cached = TRANSFORM_CACHE.get(side)
    if cached is not None:
        return cached

    rr, cc = np.indices((side, side))
    last = side - 1
    maps = (
        (rr, cc),
        (cc, last - rr),
        (last - rr, last - cc),
        (last - cc, rr),
        (rr, last - cc),
        (last - rr, cc),
        (cc, rr),
        (last - cc, last - rr),
    )
    TRANSFORM_CACHE[side] = maps
    return maps


def _build_composition_table() -> tuple[tuple[int, ...], ...]:
    global COMPOSITION_TABLE
    if COMPOSITION_TABLE is not None:
        return COMPOSITION_TABLE

    side = 5
    maps = _transform_maps(side)
    signatures = []
    for op in range(8):
        rr, cc = maps[op]
        signatures.append(tuple(zip(rr.ravel().tolist(), cc.ravel().tolist())))

    table = []
    for op_a in range(8):
        row = []
        for op_b in range(8):
            rr_b, cc_b = maps[op_b]
            rr_a, cc_a = maps[op_a]
            composed = tuple(
                zip(rr_a[rr_b, cc_b].ravel().tolist(), cc_a[rr_b, cc_b].ravel().tolist())
            )
            row.append(signatures.index(composed))
        table.append(tuple(row))

    COMPOSITION_TABLE = tuple(table)
    return COMPOSITION_TABLE


def _compatible_ops(patch: np.ndarray, mask_color: int | None) -> tuple[int, ...]:
    side = int(patch.shape[0])
    maps = _transform_maps(side)
    ops = []
    for op, (rr, cc) in enumerate(maps):
        transformed = patch[rr, cc]
        if mask_color is None:
            ok = np.array_equal(patch, transformed)
        else:
            visible = (patch != int(mask_color)) & (transformed != int(mask_color))
            ok = bool(np.all(patch[visible] == transformed[visible]))
        if ok:
            ops.append(op)
    return tuple(ops)


def _is_closed_group(ops: tuple[int, ...]) -> bool:
    if not ops or ops[0] != 0:
        return False
    table = _build_composition_table()
    ops_set = set(int(op) for op in ops)
    for op_a in ops_set:
        for op_b in ops_set:
            if table[op_a][op_b] not in ops_set:
                return False
    return True


def _group_orbits(side: int, ops: tuple[int, ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    key = (int(side), tuple(int(op) for op in ops))
    cached = ORBIT_CACHE.get(key)
    if cached is not None:
        return cached

    unseen = {(r, c) for r in range(side) for c in range(side)}
    orbits = []
    while unseen:
        r, c = min(unseen)
        orbit = tuple(sorted({_apply_op(op, side, r, c) for op in ops}))
        unseen.difference_update(orbit)
        orbits.append(orbit)
    ORBIT_CACHE[key] = tuple(orbits)
    return ORBIT_CACHE[key]


def _fill_from_group(patch: np.ndarray, mask_color: int, ops: tuple[int, ...]) -> np.ndarray | None:
    filled = patch.copy()
    has_mask = False
    for orbit in _group_orbits(int(patch.shape[0]), ops):
        values = {int(patch[r, c]) for r, c in orbit if int(patch[r, c]) != int(mask_color)}
        if len(values) != 1:
            return None
        color = next(iter(values))
        for r, c in orbit:
            if int(patch[r, c]) == int(mask_color):
                filled[r, c] = color
                has_mask = True
    if not has_mask:
        return None
    if _compatible_ops(filled, None) != tuple(ops):
        return None
    return filled


def _candidate_from_square(
    grid: np.ndarray, bg: int, mask_color: int, r0: int, c0: int, side: int
) -> dict | None:
    patch = grid[r0 : r0 + side, c0 : c0 + side]
    mask_count = int(np.sum(patch == int(mask_color)))
    if mask_count < MIN_MASK_CELLS:
        return None
    if np.any(patch == int(bg)):
        return None

    ops = _compatible_ops(patch, int(mask_color))
    if len(ops) <= 1 or not _is_closed_group(ops):
        return None

    filled = _fill_from_group(patch, int(mask_color), ops)
    if filled is None:
        return None

    return {
        "mask_color": int(mask_color),
        "bbox": (int(r0), int(c0), int(r0 + side - 1), int(c0 + side - 1)),
        "side": int(side),
        "group": tuple(int(op) for op in ops),
        "filled": filled,
        "mask_count": mask_count,
        "signal_count": int(np.sum((patch != int(mask_color)) & (patch != int(bg)))),
    }


def _enumerate_candidates(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    bg = get_bg(g)
    candidates: list[dict] = []

    for mask_color in sorted(c for c in np.unique(g) if int(c) != bg):
        coords = np.argwhere(g == int(mask_color))
        if int(coords.shape[0]) < MIN_MASK_CELLS:
            continue

        rmin, cmin = coords.min(axis=0)
        rmax, cmax = coords.max(axis=0)
        min_side = max(int(rmax - rmin + 1), int(cmax - cmin + 1))
        max_side = min(height, width)

        for side in range(min_side, max_side + 1):
            row_lo = max(0, int(rmax) - side + 1)
            row_hi = min(int(rmin), height - side)
            col_lo = max(0, int(cmax) - side + 1)
            col_hi = min(int(cmin), width - side)
            if row_lo > row_hi or col_lo > col_hi:
                continue
            for r0 in range(row_lo, row_hi + 1):
                for c0 in range(col_lo, col_hi + 1):
                    candidate = _candidate_from_square(g, bg, int(mask_color), r0, c0, side)
                    if candidate is not None:
                        candidates.append(candidate)
    return candidates


def _find_unique_candidate(grid: np.ndarray) -> dict:
    candidates = _enumerate_candidates(grid)
    assert candidates, "expected at least one valid square"

    best_signal = max(int(candidate["signal_count"]) for candidate in candidates)
    candidates = [candidate for candidate in candidates if int(candidate["signal_count"]) == best_signal]
    min_side = min(int(candidate["side"]) for candidate in candidates)
    candidates = [candidate for candidate in candidates if int(candidate["side"]) == min_side]

    assert len(candidates) == 1, f"expected one best square, found {len(candidates)}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Locate the hidden square whose masked visible cells fit one nontrivial square-symmetry subgroup, then restore the masked cells from that subgroup's orbits.

    Input structure:
    - A rectangular grid contains a noisy multicolor background plus one hidden square subregion with several clustered mask pixels of one special color.
    - All occurrences of the mask color lie inside that square, and outside noise breaks any global symmetry.
    - Inside the square, ignoring the mask color leaves visible cells that are compatible with exactly one maximal nontrivial subgroup of the square's eight symmetries.

    Rule:
    - Search every square that contains all pixels of each possible mask color.
    - For each candidate square, keep exactly the symmetries whose transformed visible cells never conflict; the surviving operations must form a nontrivial subgroup.
    - Replace each mask cell by the unique visible color in its subgroup orbit, and leave every cell outside the square unchanged.

    Key insight:
    - The mask does not tell you the boundary directly: first find the only square where ignoring that color yields a locally self-consistent symmetry subgroup, then use that subgroup's full orbit structure to fill the hidden cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    candidate = _find_unique_candidate(g)
    r0, c0, r1, c1 = candidate["bbox"]
    mask_color = int(candidate["mask_color"])
    filled = candidate["filled"]

    out = g.copy()
    patch = out[r0 : r1 + 1, c0 : c1 + 1]
    mask = patch == mask_color
    patch[mask] = filled[mask]
    return out


def _next_group() -> tuple[str, tuple[int, ...]]:
    global GROUP_BAG
    if not GROUP_BAG:
        GROUP_BAG = GROUP_FAMILIES.copy()
        random.shuffle(GROUP_BAG)
    return GROUP_BAG.pop()


def _render_group_patch(side: int, ops: tuple[int, ...], colors: list[int]) -> np.ndarray:
    patch = np.empty((side, side), dtype=int)
    orbits = list(_group_orbits(side, ops))
    palette_size = min(len(colors), random.randint(3, min(5, len(colors))))
    palette = random.sample(colors, palette_size)
    color_choices = palette + [random.choice(palette) for _ in range(len(orbits) - palette_size)]
    random.shuffle(color_choices)
    for orbit, color in zip(orbits, color_choices):
        for r, c in orbit:
            patch[r, c] = int(color)
    return patch


def _make_exact_group_patch(side: int, bg: int, mask_color: int, ops: tuple[int, ...]) -> np.ndarray:
    colors = [color for color in range(10) if color not in (int(bg), int(mask_color))]
    for _ in range(500):
        patch = _render_group_patch(side, ops, colors)
        if len(set(map(int, patch.ravel()))) < 3:
            continue
        if _compatible_ops(patch, None) == tuple(ops):
            return patch
    raise RuntimeError("failed to create a square with the requested exact symmetry group")


def _random_connected_mask(side: int) -> list[tuple[int, int]]:
    box_h = random.randint(2, min(4, side - 1))
    box_w = random.randint(2, min(4, side - 1))
    top = random.randint(0, side - box_h)
    left = random.randint(0, side - box_w)
    max_area = min(8, box_h * box_w)
    area = random.randint(MIN_MASK_CELLS, max_area)

    start = (top + random.randrange(box_h), left + random.randrange(box_w))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in sorted(cells):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if top <= nr < top + box_h and left <= nc < left + box_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))
    return sorted(cells)


def _mask_patch(full_patch: np.ndarray, mask_color: int, ops: tuple[int, ...]) -> np.ndarray:
    side = int(full_patch.shape[0])
    for _ in range(500):
        masked = full_patch.copy()
        coords = _random_connected_mask(side)
        for r, c in coords:
            masked[r, c] = int(mask_color)
        if _compatible_ops(masked, int(mask_color)) != tuple(ops):
            continue
        if _fill_from_group(masked, int(mask_color), ops) is not None:
            return masked
    raise RuntimeError("failed to place a concentrated mask without changing the inferred group")


def _fits_shape(occupied: np.ndarray, square_bbox: tuple[int, int, int, int], shape: list[tuple[int, int]], r0: int, c0: int) -> bool:
    height, width = occupied.shape
    r1 = r0 + max(r for r, _ in shape)
    c1 = c0 + max(c for _, c in shape)
    if r0 < 0 or c0 < 0 or r1 >= height or c1 >= width:
        return False
    sr0, sc0, sr1, sc1 = square_bbox
    for dr, dc in shape:
        rr, cc = r0 + dr, c0 + dc
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            return False
        if occupied[rr, cc]:
            return False
    return True


def _stamp_shape(grid: np.ndarray, occupied: np.ndarray, color: int, shape: list[tuple[int, int]], r0: int, c0: int) -> None:
    for dr, dc in shape:
        rr, cc = r0 + dr, c0 + dc
        grid[rr, cc] = int(color)
        occupied[rr, cc] = True


def _add_clutter(
    grid: np.ndarray,
    bg: int,
    square_bbox: tuple[int, int, int, int],
    colors: list[int],
) -> None:
    height, width = grid.shape
    occupied = grid != int(bg)
    sr0, _sc0, sr1, _sc1 = square_bbox

    left_rows = list(range(height))
    right_rows = list(range(height))
    random.shuffle(left_rows)
    random.shuffle(right_rows)
    for color in colors:
        while left_rows:
            row = left_rows.pop()
            if not occupied[row, 0]:
                grid[row, 0] = int(color)
                occupied[row, 0] = True
                break
        while right_rows:
            row = right_rows.pop()
            if not occupied[row, width - 1]:
                grid[row, width - 1] = int(color)
                occupied[row, width - 1] = True
                break

    shapes = [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (1, 0), (0, 1)],
    ]
    motif_count = random.randint(len(colors), len(colors) + 3)
    for _ in range(motif_count):
        color = random.choice(colors)
        shape = random.choice(shapes)
        for _ in range(80):
            row = random.randrange(height)
            col = random.randrange(width)
            if _fits_shape(occupied, square_bbox, shape, row, col):
                _stamp_shape(grid, occupied, color, shape, row, col)
                break

    if width > height:
        for color in colors:
            row = sr0 - 1 if sr0 > 0 else sr1 + 1
            if 0 <= row < height and not occupied[row, width // 2]:
                grid[row, width // 2] = int(color)
                occupied[row, width // 2] = True


def _semantic_ok(
    inp: np.ndarray,
    expected: np.ndarray,
    square_bbox: tuple[int, int, int, int],
    mask_color: int,
    ops: tuple[int, ...],
) -> bool:
    try:
        candidate = _find_unique_candidate(inp)
    except AssertionError:
        return False
    if tuple(candidate["bbox"]) != tuple(square_bbox):
        return False
    if int(candidate["mask_color"]) != int(mask_color):
        return False
    if tuple(candidate["group"]) != tuple(ops):
        return False
    solved = main(inp)
    return bool(np.array_equal(solved, expected))


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, 9)
    group_name, ops = _next_group()

    height = random.randint(16, 22)
    width = random.randint(height + 4, min(30, height + 8))
    max_side = min(10, height - 4, width - 6)
    if max_side < 5:
        return None
    side = random.randint(5, max_side)

    mask_color_choices = [color for color in range(10) if color != bg]
    mask_color = random.choice(mask_color_choices)

    full_patch = _make_exact_group_patch(side, bg, mask_color, ops)
    masked_patch = _mask_patch(full_patch, mask_color, ops)

    top = random.randint(2, height - side - 2)
    left = random.randint(2, width - side - 2)
    square_bbox = (top, left, top + side - 1, left + side - 1)

    base = np.full((height, width), bg, dtype=int)
    visible_colors = sorted(set(map(int, full_patch.ravel())))
    _add_clutter(base, bg, square_bbox, visible_colors)

    inp = base.copy()
    out = base.copy()
    inp[top : top + side, left : left + side] = masked_patch
    out[top : top + side, left : left + side] = full_patch

    if get_bg(inp) != bg:
        return None
    if not _semantic_ok(inp, out, square_bbox, mask_color, ops):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a unique hidden-repair scene")
