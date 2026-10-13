import random
from collections import Counter, deque

import numpy as np


BG = 0
GRAY = 5
ARROW_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

ARROW_RELS = {
    (-1, 0): [(0, 0), (0, -1), (0, 1), (1, 0), (2, 0)],
    (1, 0): [(0, 0), (0, -1), (0, 1), (-1, 0), (-2, 0)],
    (0, -1): [(0, 0), (-1, 0), (1, 0), (0, 1), (0, 2)],
    (0, 1): [(0, 0), (-1, 0), (1, 0), (0, -1), (0, -2)],
}

ARROW_TEMPLATES = {}
for direction, rels in ARROW_RELS.items():
    min_r = min(dr for dr, _ in rels)
    min_c = min(dc for _, dc in rels)
    normalized = frozenset((dr - min_r, dc - min_c) for dr, dc in rels)
    tip_rel = (-min_r, -min_c)
    ARROW_TEMPLATES[normalized] = (direction, tip_rel)

GRID_H_RANGE = (14, 29)
GRID_W_RANGE = (14, 29)
BLOCK_SIZE_CHOICES = [2, 3]
TARGET_PLACE_TRIES = 250
EXTRA_PLACE_TRIES = 250
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_for_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _square_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _steps_to_edge(coords: list[tuple[int, int]], direction: tuple[int, int], shape: tuple[int, int]) -> int:
    h, w = shape
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    dr, dc = direction
    if dr == -1:
        return min(rows)
    if dr == 1:
        return h - 1 - max(rows)
    if dc == -1:
        return min(cols)
    return w - 1 - max(cols)


def _translate(coords: list[tuple[int, int]], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(r + dr * steps, c + dc * steps) for r, c in coords]


def _match_arrow(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(coords) != 5:
        return None
    r0, c0, _, _ = _square_bbox(coords)
    normalized = frozenset((r - r0, c - c0) for r, c in coords)
    match = ARROW_TEMPLATES.get(normalized)
    if match is None:
        return None
    direction, tip_rel = match
    tip = (r0 + tip_rel[0], c0 + tip_rel[1])
    return direction, tip


def _paint(grid: np.ndarray, coords: list[tuple[int, int]] | set[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = color


def _square_cells(top: tuple[int, int], size: int) -> list[tuple[int, int]]:
    r0, c0 = top
    return [(r, c) for r in range(r0, r0 + size) for c in range(c0, c0 + size)]


def _expand4(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    expanded = set(cells)
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                expanded.add((nr, nc))
    return expanded


def _arrow_cells_from_tip(tip: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    tr, tc = tip
    return [(tr + dr, tc + dc) for dr, dc in ARROW_RELS[direction]]


def _sight_cells(tip: tuple[int, int], direction: tuple[int, int], target: set[tuple[int, int]]) -> set[tuple[int, int]]:
    dr, dc = direction
    r, c = tip[0] + dr, tip[1] + dc
    seen = set()
    while (r, c) not in target:
        seen.add((r, c))
        r += dr
        c += dc
    return seen


def _swept_cells(top: tuple[int, int], size: int, direction: tuple[int, int], steps: int) -> set[tuple[int, int]]:
    orig = set(_square_cells(top, size))
    swept = set()
    for step in range(1, steps + 1):
        swept.update(_translate(list(orig), direction, step))
    return swept - orig


def _target_tip(block_top: tuple[int, int], size: int, direction: tuple[int, int], lane: int, gap: int) -> tuple[int, int]:
    r0, c0 = block_top
    dr, dc = direction
    if dr == -1:
        return (r0 + size - 1 + gap, c0 + lane)
    if dr == 1:
        return (r0 - gap, c0 + lane)
    if dc == -1:
        return (r0 + lane, c0 + size - 1 + gap)
    return (r0 + lane, c0 - gap)


def _destination_top(block_top: tuple[int, int], size: int, direction: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    r0, c0 = block_top
    dr, dc = direction
    if dr == -1:
        return (0, c0)
    if dr == 1:
        return (h - size, c0)
    if dc == -1:
        return (r0, 0)
    return (r0, w - size)


def _in_bounds(cells: list[tuple[int, int]] | set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each ray-targeted gray square to the edge indicated by its colored arrow and leave a colored trail behind.

    Input structure:
    - The background is black.
    - Several solid gray squares of one common size appear on the canvas.
    - Several 5-pixel colored arrows appear in the four cardinal orientations; each arrow is a 3-cell bar with a 2-cell head.
    - Along each arrow's forward ray, the first gray square is its unique target.

    Rule:
    - Remove every arrow from the grid.
    - For each arrow, ray-cast forward from the center of its tip until the first gray square is reached.
    - Translate that whole gray square in the arrow direction until its front edge touches the grid boundary.
    - Fill the square's original footprint with the arrow's color as a same-size trail marker.
    - Gray squares that are not hit by any arrow stay where they are.

    Key insight:
    - The arrows are remote controls: they do not move themselves, but instead select distant gray squares by ray casting and determine both the push direction and the trail color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(Counter(map(int, g.flat)).most_common(1)[0][0])

    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        comps_by_color[color] = _components_for_color(g, color)

    block_candidates: list[tuple[int, int, int, int]] = []
    for color, comps in comps_by_color.items():
        dims = set()
        valid = bool(comps)
        for comp in comps:
            r0, c0, r1, c1 = _square_bbox(comp)
            side_r = r1 - r0 + 1
            side_c = c1 - c0 + 1
            if side_r != side_c or side_r < 2 or len(comp) != side_r * side_c:
                valid = False
                break
            dims.add((side_r, side_c))
        if valid and len(dims) == 1:
            side = next(iter(dims))[0]
            block_candidates.append((len(comps), len(comps) * side * side, color, side))

    assert block_candidates, "expected at least one gray-square color"
    _, _, block_color, block_size = max(block_candidates, key=lambda item: (item[0], item[1], -item[2]))
    block_comps = comps_by_color[block_color]

    blocks = []
    cell_to_block: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(block_comps):
        blocks.append({"coords": comp})
        for cell in comp:
            cell_to_block[cell] = idx

    arrows = []
    for color, comps in comps_by_color.items():
        if color == block_color:
            continue
        for comp in comps:
            matched = _match_arrow(comp)
            assert matched is not None, "every non-gray component must be an arrow"
            direction, tip = matched
            arrows.append({"color": color, "coords": comp, "direction": direction, "tip": tip})

    assert arrows, "expected at least one arrow"

    hits: dict[int, dict] = {}
    for arrow in arrows:
        dr, dc = arrow["direction"]
        r, c = arrow["tip"][0] + dr, arrow["tip"][1] + dc
        hit_idx = None
        while 0 <= r < h and 0 <= c < w:
            if (r, c) in cell_to_block:
                hit_idx = cell_to_block[(r, c)]
                break
            if int(g[r, c]) != bg:
                break
            r += dr
            c += dc
        assert hit_idx is not None, "every arrow must hit a gray square"
        assert hit_idx not in hits, "gray squares must be matched one-to-one with arrows"
        hits[hit_idx] = arrow

    out = np.full_like(g, bg)

    for idx, block in enumerate(blocks):
        if idx in hits:
            continue
        _paint(out, block["coords"], block_color)

    for idx, arrow in hits.items():
        block = blocks[idx]
        steps = _steps_to_edge(block["coords"], arrow["direction"], g.shape)
        if steps == 0:
            _paint(out, block["coords"], block_color)
            continue
        moved = _translate(block["coords"], arrow["direction"], steps)
        assert _in_bounds(moved, g.shape)
        assert all(int(out[r, c]) == bg for r, c in block["coords"])
        assert all(int(out[r, c]) == bg for r, c in moved)
        _paint(out, block["coords"], arrow["color"])
        _paint(out, moved, block_color)

    return out


def _try_place_target(
    shape: tuple[int, int],
    size: int,
    color: int,
    occupied_input: set[tuple[int, int]],
    must_be_bg: set[tuple[int, int]],
    grey_input_blocked: set[tuple[int, int]],
    grey_output_blocked: set[tuple[int, int]],
    output_used: set[tuple[int, int]],
) -> dict | None:
    h, w = shape
    for _ in range(TARGET_PLACE_TRIES):
        direction = random.choice(DIRS)
        dr, dc = direction

        if dr == -1:
            if h - size - 3 < 1:
                continue
            r0 = random.randint(1, h - size - 3)
            c0 = random.randint(0, w - size)
            max_gap = h - (r0 + size) - 2
        elif dr == 1:
            if h - size - 1 < 3:
                continue
            r0 = random.randint(3, h - size - 1)
            c0 = random.randint(0, w - size)
            max_gap = r0 - 2
        elif dc == -1:
            if w - size - 3 < 1:
                continue
            r0 = random.randint(0, h - size)
            c0 = random.randint(1, w - size - 3)
            max_gap = w - (c0 + size) - 2
        else:
            if w - size - 1 < 3:
                continue
            r0 = random.randint(0, h - size)
            c0 = random.randint(3, w - size - 1)
            max_gap = c0 - 2

        if max_gap < 1:
            continue

        lane = random.randint(0, size - 1)
        gap = random.randint(1, max_gap)
        block_top = (r0, c0)
        orig = set(_square_cells(block_top, size))
        tip = _target_tip(block_top, size, direction, lane, gap)
        arrow = set(_arrow_cells_from_tip(tip, direction))
        if not _in_bounds(orig | arrow, shape):
            continue

        steps = _steps_to_edge(list(orig), direction, shape)
        if steps <= 0:
            continue

        sight = _sight_cells(tip, direction, orig)
        if not _in_bounds(sight, shape):
            continue

        dest_top = _destination_top(block_top, size, direction, shape)
        dest = set(_square_cells(dest_top, size))
        swept = _swept_cells(block_top, size, direction, steps)

        if orig & occupied_input or arrow & occupied_input or sight & occupied_input or swept & occupied_input:
            continue
        if orig & must_be_bg or arrow & must_be_bg:
            continue
        if orig & grey_input_blocked:
            continue
        if orig & output_used or dest & output_used:
            continue
        if dest & grey_output_blocked:
            continue
        if orig & arrow or orig & dest or arrow & dest:
            continue

        occupied_input.update(orig | arrow)
        must_be_bg.update(sight | swept | dest)
        grey_input_blocked.update(_expand4(orig, shape))
        grey_output_blocked.update(_expand4(dest, shape))
        output_used.update(orig | dest)

        return {
            "color": color,
            "direction": direction,
            "block": sorted(orig),
            "dest": sorted(dest),
            "arrow": sorted(arrow),
        }

    return None


def _try_place_extra(
    shape: tuple[int, int],
    size: int,
    occupied_input: set[tuple[int, int]],
    must_be_bg: set[tuple[int, int]],
    grey_input_blocked: set[tuple[int, int]],
    grey_output_blocked: set[tuple[int, int]],
    output_used: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = shape
    for _ in range(EXTRA_PLACE_TRIES):
        r0 = random.randint(0, h - size)
        c0 = random.randint(0, w - size)
        cells = set(_square_cells((r0, c0), size))
        if cells & occupied_input or cells & must_be_bg:
            continue
        if cells & grey_input_blocked or cells & grey_output_blocked:
            continue
        if cells & output_used:
            continue

        occupied_input.update(cells)
        grey_input_blocked.update(_expand4(cells, shape))
        grey_output_blocked.update(_expand4(cells, shape))
        output_used.update(cells)
        return sorted(cells)
    return None


def _build_expected(shape: tuple[int, int], targets: list[dict], extras: list[list[tuple[int, int]]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for cells in extras:
        _paint(out, cells, GRAY)
    for rec in targets:
        _paint(out, rec["block"], rec["color"])
        _paint(out, rec["dest"], GRAY)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    size = random.choice(BLOCK_SIZE_CHOICES)
    shape = (h, w)

    if min(shape) < size + 6:
        return None

    if min(shape) < 16:
        max_targets = 2
    elif min(shape) < 22:
        max_targets = 3
    else:
        max_targets = 4

    target_count = random.randint(1, max_targets)
    extra_count = random.randint(0, min(2, max_targets))
    if target_count == 1 and extra_count == 0 and random.random() < 0.5:
        extra_count = 1

    occupied_input: set[tuple[int, int]] = set()
    must_be_bg: set[tuple[int, int]] = set()
    grey_input_blocked: set[tuple[int, int]] = set()
    grey_output_blocked: set[tuple[int, int]] = set()
    output_used: set[tuple[int, int]] = set()

    targets: list[dict] = []
    colors = random.sample(ARROW_COLORS, target_count)
    for color in colors:
        placed = _try_place_target(
            shape,
            size,
            color,
            occupied_input,
            must_be_bg,
            grey_input_blocked,
            grey_output_blocked,
            output_used,
        )
        if placed is None:
            return None
        targets.append(placed)

    extras: list[list[tuple[int, int]]] = []
    for _ in range(extra_count):
        extra = _try_place_extra(
            shape,
            size,
            occupied_input,
            must_be_bg,
            grey_input_blocked,
            grey_output_blocked,
            output_used,
        )
        if extra is None:
            return None
        extras.append(extra)

    inp = np.full(shape, BG, dtype=int)
    for rec in targets:
        _paint(inp, rec["block"], GRAY)
        _paint(inp, rec["arrow"], rec["color"])
    for cells in extras:
        _paint(inp, cells, GRAY)

    expected = _build_expected(shape, targets, extras)

    if int(Counter(map(int, inp.flat)).most_common(1)[0][0]) != BG:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
