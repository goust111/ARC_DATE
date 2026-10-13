from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, crop_bbox, get_bg


Cell = tuple[int, int]


def _cells_from_mask(mask: np.ndarray) -> frozenset[Cell]:
    return frozenset((int(r), int(c)) for r, c in np.argwhere(mask))


def _translate_cells(cells: frozenset[Cell], offset: Cell) -> frozenset[Cell]:
    dr, dc = map(int, offset)
    return frozenset((r + dr, c + dc) for r, c in cells)


def _candidate_positions(
    proto_cells: frozenset[Cell],
    visible_cells: frozenset[Cell],
    foreground_cells: frozenset[Cell],
    grid_shape: tuple[int, int],
) -> list[Cell]:
    h, w = map(int, grid_shape)
    ph = max(r for r, _ in proto_cells) + 1
    pw = max(c for _, c in proto_cells) + 1
    offsets: set[Cell] = set()
    for vr, vc in visible_cells:
        for pr, pc in proto_cells:
            offsets.add((vr - pr, vc - pc))

    candidates: list[Cell] = []
    for r0, c0 in offsets:
        if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
            continue
        placed = _translate_cells(proto_cells, (r0, c0))
        if not visible_cells.issubset(placed):
            continue
        if not placed.issubset(foreground_cells):
            continue
        candidates.append((int(r0), int(c0)))
    return sorted(set(candidates))


def _chain_order_from_full_shapes(
    grid: np.ndarray, full_by_color: dict[int, frozenset[Cell]]
) -> list[int]:
    colors = list(full_by_color)
    out_edges = {color: set() for color in colors}
    in_edges = {color: set() for color in colors}
    edge_count = 0

    for idx, color_a in enumerate(colors):
        for color_b in colors[idx + 1 :]:
            overlap = full_by_color[color_a] & full_by_color[color_b]
            if not overlap:
                continue
            seen = {int(grid[r, c]) for r, c in overlap}
            assert len(seen) == 1, "pairwise overlap must show exactly one visible color"
            upper = next(iter(seen))
            assert upper in (color_a, color_b), "triple overlap is not allowed"
            lower = color_b if upper == color_a else color_a
            out_edges[upper].add(lower)
            in_edges[lower].add(upper)
            edge_count += 1

    assert edge_count == len(colors) - 1, "overlap graph must be a single chain"
    assert all(len(v) <= 1 for v in out_edges.values())
    assert all(len(v) <= 1 for v in in_edges.values())

    starts = [color for color in colors if not in_edges[color]]
    ends = [color for color in colors if not out_edges[color]]
    assert len(starts) == 1 and len(ends) == 1, "chain must have one top and one bottom"

    order: list[int] = []
    seen_colors: set[int] = set()
    cur = starts[0]
    while True:
        assert cur not in seen_colors, "overlap chain must be acyclic"
        order.append(cur)
        seen_colors.add(cur)
        if not out_edges[cur]:
            break
        cur = next(iter(out_edges[cur]))

    assert len(order) == len(colors), "overlap chain must cover every stamp"
    return order


def _extract_scene_semantics(
    grid: np.ndarray,
) -> tuple[int, np.ndarray, list[int], dict[int, frozenset[Cell]], dict[int, frozenset[Cell]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(colors) >= 3, "expected at least three stamp colors"
    visible_by_color = {
        color: frozenset((int(r), int(c)) for r, c in np.argwhere(g == color))
        for color in colors
    }
    assert all(visible_by_color.values()), "every stamp color must remain visible"

    sizes = {color: len(cells) for color, cells in visible_by_color.items()}
    max_size = max(sizes.values())
    top_candidates = [color for color, size in sizes.items() if size == max_size]
    assert len(top_candidates) == 1, "top complete stamp must be unique"
    top_color = top_candidates[0]

    top_arr = np.array(sorted(visible_by_color[top_color]), dtype=int)
    top_bbox = bbox_coords(top_arr)
    proto_crop = crop_bbox(g, top_bbox)
    proto_mask = proto_crop == top_color
    proto_cells = _cells_from_mask(proto_mask)
    assert proto_cells, "prototype cannot be empty"

    foreground_cells = frozenset((int(r), int(c)) for r, c in np.argwhere(g != bg))
    full_by_color: dict[int, frozenset[Cell]] = {}
    for color, visible_cells in visible_by_color.items():
        candidates = _candidate_positions(proto_cells, visible_cells, foreground_cells, g.shape)
        assert len(candidates) == 1, "every partial stamp must map to one translated prototype"
        full_by_color[color] = _translate_cells(proto_cells, candidates[0])

    rebuilt_foreground = frozenset().union(*full_by_color.values())
    assert rebuilt_foreground == foreground_cells, "prototype placements must cover the foreground exactly"

    order = _chain_order_from_full_shapes(g, full_by_color)
    assert order[0] == top_color, "the complete stamp must be the chain top"
    assert visible_by_color[top_color] == full_by_color[top_color], "top stamp must be fully visible"
    for color in order[1:]:
        assert visible_by_color[color] < full_by_color[color], "every lower stamp must be partially hidden"

    return bg, proto_mask, order, visible_by_color, full_by_color


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the full prototype from the only complete stamp and repaint it with the bottom stamp color.

    Input structure:
    - The dominant background contains several same-shape stamp copies in distinct colors.
    - Every stamp is a pure translation of one prototype sprite.
    - Exactly one top-layer stamp is fully visible; every lower stamp is partially hidden by overlaps, and the overlaps form one top-to-bottom chain.

    Rule:
    - Find the unique complete visible stamp and use its tight crop as the prototype sprite mask.
    - Reconstruct each colored stamp as the unique translated copy of that prototype whose full cells all lie inside the observed foreground.
    - For every overlapping pair, the color visible on the shared cells is the higher layer; follow these pairwise relations to the chain end, then output the prototype crop recolored with the bottom stamp's color.

    Key insight:
    - The bottom stamp never appears complete in the scene, so its shape must be borrowed from the top full copy while its identity comes only from the occlusion chain.
    """
    bg, proto_mask, order, _visible_by_color, _full_by_color = _extract_scene_semantics(grid)
    bottom_color = int(order[-1])
    out = np.full(proto_mask.shape, bg, dtype=int)
    out[proto_mask] = bottom_color
    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 600
PROTO_ATTEMPTS = 120
CHAIN_ATTEMPTS = 120
BG_COLORS = tuple(range(10))
STAMP_COUNT_CHOICES = (3, 4)
PROTO_H_RANGE = (4, 6)
PROTO_W_RANGE = (4, 6)
PROTO_AREA_RANGE = (7, 12)
MARGIN_RANGE = (1, 4)
MAX_GRID_SIDE = 30


def _neighbors4(cell: Cell) -> list[Cell]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _tight_mask(cells: set[Cell]) -> np.ndarray:
    arr = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[int(r - r0), int(c - c0)] = True
    return mask


def _random_prototype_mask() -> np.ndarray | None:
    for _ in range(PROTO_ATTEMPTS):
        h = random.randint(*PROTO_H_RANGE)
        w = random.randint(*PROTO_W_RANGE)
        max_area = min(PROTO_AREA_RANGE[1], h * w - 1)
        min_area = min(PROTO_AREA_RANGE[0], max_area)
        if min_area < 5:
            continue
        target_area = random.randint(min_area, max_area)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < target_area:
            frontier = {
                nxt
                for cell in cells
                for nxt in _neighbors4(cell)
                if 0 <= nxt[0] < h and 0 <= nxt[1] < w and nxt not in cells
            }
            if not frontier:
                break
            cells.add(random.choice(sorted(frontier)))
        if len(cells) != target_area:
            continue

        mask = _tight_mask(cells)
        mh, mw = mask.shape
        area = int(mask.sum())
        if mh < 3 or mw < 3 or area < 7:
            continue
        if area == mh * mw:
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    return None


def _delta_catalog(proto_cells: frozenset[Cell]) -> list[dict[str, object]]:
    ph = max(r for r, _ in proto_cells) + 1
    pw = max(c for _, c in proto_cells) + 1
    area = len(proto_cells)
    catalog: list[dict[str, object]] = []
    for dr in range(-(ph - 1), ph):
        for dc in range(-(pw - 1), pw):
            if dr == 0 and dc == 0:
                continue
            hidden_local = {(r, c) for r, c in proto_cells if (r + dr, c + dc) in proto_cells}
            visible_local = proto_cells - hidden_local
            visible_count = len(visible_local)
            if visible_count < 3 or visible_count >= area:
                continue
            catalog.append(
                {
                    "delta": (int(dr), int(dc)),
                    "visible_count": int(visible_count),
                    "overlap_count": int(area - visible_count),
                }
            )
    return catalog


def _add(a: Cell, b: Cell) -> Cell:
    return int(a[0] + b[0]), int(a[1] + b[1])


def _positions_overlap(proto_cells: frozenset[Cell], pos_a: Cell, pos_b: Cell) -> bool:
    return bool(_translate_cells(proto_cells, pos_a) & _translate_cells(proto_cells, pos_b))


def _place_chain(proto_cells: frozenset[Cell], count: int) -> list[Cell] | None:
    area = len(proto_cells)
    catalog = _delta_catalog(proto_cells)
    heavy = [item for item in catalog if int(item["visible_count"]) <= max(4, area // 2)]
    light = [item for item in catalog if int(item["visible_count"]) >= max(5, area // 2 + 1)]
    if not heavy or not light:
        return None

    for _ in range(CHAIN_ATTEMPTS):
        positions = [(0, 0)]
        prior_visible_counts: list[int] = []
        ok = True
        for idx in range(1, count):
            if idx == 1:
                pool = heavy[:]
            elif idx == count - 1:
                threshold = min(prior_visible_counts)
                pool = [item for item in light if int(item["visible_count"]) > threshold]
                if not pool:
                    pool = [item for item in catalog if int(item["visible_count"]) > threshold]
            else:
                pool = catalog[:]
            random.shuffle(pool)

            chosen = None
            for item in pool:
                delta = item["delta"]
                new_pos = _add(positions[-1], delta)
                if any(_positions_overlap(proto_cells, new_pos, old) for old in positions[:-1]):
                    continue
                chosen = item
                positions.append(new_pos)
                prior_visible_counts.append(int(item["visible_count"]))
                break
            if chosen is None:
                ok = False
                break
        if not ok:
            continue

        bottom_visible = prior_visible_counts[-1]
        if bottom_visible <= min(prior_visible_counts[:-1]):
            continue
        return positions
    return None


def _normalize_positions(proto_cells: frozenset[Cell], positions: list[Cell]) -> tuple[list[Cell], tuple[int, int]]:
    all_cells = [cell for pos in positions for cell in _translate_cells(proto_cells, pos)]
    rows = [r for r, _ in all_cells]
    cols = [c for _, c in all_cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    span_h = max_r - min_r + 1
    span_w = max_c - min_c + 1

    top = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    bottom = random.randint(*MARGIN_RANGE)
    right = random.randint(*MARGIN_RANGE)
    grid_h = span_h + top + bottom
    grid_w = span_w + left + right
    if grid_h > MAX_GRID_SIDE or grid_w > MAX_GRID_SIDE:
        raise ValueError("grid too large")

    shift = (top - min_r, left - min_c)
    shifted = [_add(pos, shift) for pos in positions]
    return shifted, (grid_h, grid_w)


def _full_shapes(proto_cells: frozenset[Cell], order: list[int], positions: list[Cell]) -> dict[int, frozenset[Cell]]:
    return {
        int(color): _translate_cells(proto_cells, pos)
        for color, pos in zip(order, positions, strict=True)
    }


def _render_scene(
    proto_cells: frozenset[Cell],
    order: list[int],
    positions: list[Cell],
    bg: int,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    grid = np.full(grid_shape, int(bg), dtype=int)
    full_by_color = _full_shapes(proto_cells, order, positions)
    for color in reversed(order):
        for r, c in full_by_color[int(color)]:
            grid[int(r), int(c)] = int(color)
    return grid


def _visible_from_latent(
    proto_cells: frozenset[Cell], order: list[int], positions: list[Cell]
) -> dict[int, frozenset[Cell]]:
    full_by_color = _full_shapes(proto_cells, order, positions)
    covered: set[Cell] = set()
    visible_by_color: dict[int, frozenset[Cell]] = {}
    for color in order:
        visible = full_by_color[int(color)] - covered
        visible_by_color[int(color)] = frozenset(visible)
        covered |= full_by_color[int(color)]
    return visible_by_color


def _expected_output(proto_mask: np.ndarray, bottom_color: int, bg: int) -> np.ndarray:
    out = np.full(proto_mask.shape, int(bg), dtype=int)
    out[proto_mask] = int(bottom_color)
    return out


def _validate_generated_scene(
    grid: np.ndarray,
    proto_mask: np.ndarray,
    order: list[int],
    positions: list[Cell],
    bg: int,
) -> bool:
    if int(get_bg(grid)) != int(bg):
        return False

    proto_cells = _cells_from_mask(proto_mask)
    full_by_color = _full_shapes(proto_cells, order, positions)
    visible_by_color = _visible_from_latent(proto_cells, order, positions)

    if any(not visible_by_color[int(color)] for color in order):
        return False
    if visible_by_color[int(order[0])] != full_by_color[int(order[0])]:
        return False
    if any(visible_by_color[int(color)] == full_by_color[int(color)] for color in order[1:]):
        return False

    visible_counts = [len(visible_by_color[int(color)]) for color in order[1:]]
    if visible_counts[-1] <= min(visible_counts[:-1]):
        return False

    cover_counts: dict[Cell, int] = {}
    for cells in full_by_color.values():
        for cell in cells:
            cover_counts[cell] = cover_counts.get(cell, 0) + 1
    if max(cover_counts.values(), default=0) > 2:
        return False

    for idx, color_a in enumerate(order):
        for color_b in order[idx + 1 :]:
            overlap = full_by_color[int(color_a)] & full_by_color[int(color_b)]
            if abs(order.index(color_a) - order.index(color_b)) == 1:
                if not overlap:
                    return False
            elif overlap:
                return False

    try:
        solved_bg, solved_mask, solved_order, _solved_visible, solved_full = _extract_scene_semantics(grid)
    except AssertionError:
        return False

    if solved_bg != bg or solved_order != order or not np.array_equal(solved_mask, proto_mask):
        return False
    if solved_full != full_by_color:
        return False

    expected = _expected_output(proto_mask, int(order[-1]), bg)
    return np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    proto_mask = _random_prototype_mask()
    if proto_mask is None:
        return None
    proto_cells = _cells_from_mask(proto_mask)

    stamp_count = random.choice(STAMP_COUNT_CHOICES)
    positions = _place_chain(proto_cells, stamp_count)
    if positions is None:
        return None

    try:
        shifted_positions, grid_shape = _normalize_positions(proto_cells, positions)
    except ValueError:
        return None

    bg = random.choice(BG_COLORS)
    palette = [color for color in range(10) if color != bg]
    random.shuffle(palette)
    order = [int(color) for color in palette[:stamp_count]]
    grid = _render_scene(proto_cells, order, shifted_positions, bg, grid_shape)

    if not _validate_generated_scene(grid, proto_mask, order, shifted_positions, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
