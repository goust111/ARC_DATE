from __future__ import annotations

import random
from collections import Counter

import numpy as np


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
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_border(points: list[tuple[int, int]] | set[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in points)


def _most_common_color(arr: np.ndarray) -> int:
    vals, cnts = np.unique(arr, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _expanded_cells(cells: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _mask_to_cells(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        pts.add((top + int(r), left + int(c)))
    return pts


def _random_manhattan_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    path = [(r, c)]
    dr = 1 if tr > r else -1
    dc = 1 if tc > c else -1
    rem_r = abs(tr - r)
    rem_c = abs(tc - c)
    while rem_r > 0 or rem_c > 0:
        axes: list[str] = []
        if rem_r > 0:
            axes.append("r")
        if rem_c > 0:
            axes.append("c")
        axis = random.choice(axes)
        if axis == "r":
            r += dr
            rem_r -= 1
        else:
            c += dc
            rem_c -= 1
        path.append((r, c))
    return path


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)

    comps = _components(g != bg)
    assert comps, "expected at least one non-background component"
    interior = [comp for comp in comps if not _touches_border(comp, h, w)]
    shadow_comp = max(interior if interior else comps, key=len)
    shadow_set = set(shadow_comp)
    shadow_mask = np.zeros((h, w), dtype=bool)
    for r, c in shadow_comp:
        shadow_mask[r, c] = True

    shadow_colors = [int(g[r, c]) for r, c in shadow_comp]
    shadow_color = Counter(shadow_colors).most_common(1)[0][0]
    label_colors = [c for c in sorted(set(shadow_colors)) if c != shadow_color]
    assert label_colors, "expected colored labels inside the shadow"

    out = np.full((h, w), bg, dtype=int)
    covered = np.zeros((h, w), dtype=bool)
    cover_counts: dict[int, int] = {}

    for color in label_colors:
        comps_c = _components(g == color)
        label_comps = [comp for comp in comps_c if all((r, c) in shadow_set for r, c in comp)]
        frag_comps = [comp for comp in comps_c if all((r, c) not in shadow_set for r, c in comp)]
        assert len(label_comps) == 1, f"expected one in-shadow label for color {color}"
        assert len(frag_comps) == 1, f"expected one edge fragment for color {color}"

        label = label_comps[0]
        frag = frag_comps[0]
        lr0, lc0, _, _ = _bbox(label)
        fr0, fc0, _, _ = _bbox(frag)

        placed_cells: list[tuple[int, int]] = []
        for r, c in frag:
            nr = lr0 + (r - fr0)
            nc = lc0 + (c - fc0)
            assert 0 <= nr < h and 0 <= nc < w, "translated fragment left the grid"
            assert shadow_mask[nr, nc], "translated fragment must stay inside the shadow"
            placed_cells.append((nr, nc))

        cover_counts[color] = len(placed_cells)
        for nr, nc in placed_cells:
            out[nr, nc] = color
            covered[nr, nc] = True

    fill_color = max(cover_counts.items(), key=lambda kv: kv[1])[0]
    remainder = shadow_mask & ~covered
    out[remainder] = fill_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble edge fragments inside the central shadow using same-color labels as anchors,
    then fill the remaining shadow with the color of the largest fragment.
    """
    return _solve(grid)


# --------------- generator ---------------

BG_COLOR = 0
SHADOW_COLOR = 5
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (18, 27)
GRID_W_RANGE = (18, 27)
PIECE_COUNT_RANGE = (3, 5)
PIECE_BOX_SIDE_RANGE = (2, 4)
PIECE_AREA_RANGE = (3, 7)
SHADOW_MARGIN = 3
SHADOW_BOX_H_RANGE = (9, 14)
SHADOW_BOX_W_RANGE = (10, 16)
SHADOW_DENSITY_RANGE = (0.50, 0.72)
MAX_RETRIES = 400


def _gen_piece_mask() -> tuple[np.ndarray, int] | None:
    for _ in range(80):
        label_len = random.choice((1, 2))
        box_h = random.randint(*PIECE_BOX_SIDE_RANGE)
        box_w_min = 2 if label_len == 2 else 1
        box_w = random.randint(box_w_min, PIECE_BOX_SIDE_RANGE[1])
        lo = max(label_len + 1, PIECE_AREA_RANGE[0])
        hi = min(box_h * box_w, PIECE_AREA_RANGE[1])
        if lo > hi:
            continue
        target_area = random.randint(lo, hi)

        mask = np.zeros((box_h, box_w), dtype=bool)
        mask[0, 0] = True
        if label_len == 2:
            mask[0, 1] = True

        while int(mask.sum()) < target_area:
            frontier: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                rr = int(r)
                cc = int(c)
                for nr, nc in _neighbors4(rr, cc, box_h, box_w):
                    if not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(tuple(frontier))
            mask[nr, nc] = True

        if int(mask.sum()) != target_area:
            continue

        mask = _crop_mask(mask)
        if label_len == 2 and (mask.shape[1] < 2 or not mask[0, 1]):
            continue
        if not mask[0, 0]:
            continue
        if int(mask.sum()) == mask.size and random.random() < 0.7:
            continue
        return mask, label_len
    return None


def _place_pieces(
    h: int, w: int, masks: list[np.ndarray], box_r0: int, box_c0: int, box_h: int, box_w: int
) -> tuple[list[tuple[int, int]], list[set[tuple[int, int]]]] | None:
    anchors: list[tuple[int, int]] = []
    abs_cells: list[set[tuple[int, int]]] = []
    blocked: set[tuple[int, int]] = set()

    order = list(range(len(masks)))
    random.shuffle(order)
    anchor_slots: list[tuple[int, int] | None] = [None] * len(masks)
    cell_slots: list[set[tuple[int, int]] | None] = [None] * len(masks)

    for idx in order:
        mask = masks[idx]
        ph, pw = mask.shape
        row_lo = box_r0 + 1
        row_hi = box_r0 + box_h - ph - 1
        col_lo = box_c0 + 1
        col_hi = box_c0 + box_w - pw - 1
        if row_lo > row_hi or col_lo > col_hi:
            return None

        candidates = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
        random.shuffle(candidates)
        placed = False
        for ar, ac in candidates:
            cells = _mask_to_cells(mask, ar, ac)
            if cells & blocked:
                continue
            anchor_slots[idx] = (ar, ac)
            cell_slots[idx] = cells
            blocked |= _expanded_cells(cells, h, w, radius=1)
            placed = True
            break
        if not placed:
            return None

    for idx in range(len(masks)):
        assert anchor_slots[idx] is not None and cell_slots[idx] is not None
        anchors.append(anchor_slots[idx])  # type: ignore[arg-type]
        abs_cells.append(cell_slots[idx])  # type: ignore[arg-type]
    return anchors, abs_cells


def _grow_shadow(
    seeds: set[tuple[int, int]], box_r0: int, box_c0: int, box_h: int, box_w: int, target_area: int
) -> set[tuple[int, int]]:
    shadow = set(seeds)
    while len(shadow) < target_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in shadow:
            for nr, nc in _neighbors4(r, c, box_r0 + box_h, box_c0 + box_w):
                if box_r0 <= nr < box_r0 + box_h and box_c0 <= nc < box_c0 + box_w and (nr, nc) not in shadow:
                    frontier.add((nr, nc))
        if not frontier:
            break
        frontier_list = list(frontier)
        weights = []
        for r, c in frontier_list:
            adj = 0
            for nr, nc in _neighbors4(r, c, box_r0 + box_h, box_c0 + box_w):
                if (nr, nc) in shadow:
                    adj += 1
            weights.append(adj + 1)
        pick = random.choices(frontier_list, weights=weights, k=1)[0]
        shadow.add(pick)
    return shadow


def _place_edge_fragments(
    h: int, w: int, shadow: set[tuple[int, int]], masks: list[np.ndarray]
) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int] | None] = [None] * len(masks)
    blocked = _expanded_cells(shadow, h, w, radius=1)

    order = sorted(range(len(masks)), key=lambda i: int(masks[i].sum()), reverse=True)
    for idx in order:
        mask = masks[idx]
        ph, pw = mask.shape
        candidates: list[tuple[int, int]] = []
        for c in range(w - pw + 1):
            candidates.append((0, c))
            candidates.append((h - ph, c))
        for r in range(h - ph + 1):
            candidates.append((r, 0))
            candidates.append((r, w - pw))
        random.shuffle(candidates)

        placed = False
        for top, left in candidates:
            cells = _mask_to_cells(mask, top, left)
            if cells & blocked:
                continue
            placements[idx] = (top, left)
            blocked |= _expanded_cells(cells, h, w, radius=1)
            placed = True
            break
        if not placed:
            return None

    return [p for p in placements if p is not None]


def _build_expected_output(
    h: int,
    w: int,
    shadow: set[tuple[int, int]],
    piece_cells: list[set[tuple[int, int]]],
    colors: list[int],
) -> np.ndarray:
    areas = [len(cells) for cells in piece_cells]
    fill_color = colors[int(np.argmax(areas))]
    out = np.full((h, w), BG_COLOR, dtype=int)
    covered: set[tuple[int, int]] = set()
    for color, cells in zip(colors, piece_cells):
        for r, c in cells:
            out[r, c] = color
            covered.add((r, c))
    for r, c in shadow:
        if (r, c) not in covered:
            out[r, c] = fill_color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    colors = random.sample(list(PIECE_COLORS), n_pieces)

    piece_specs = [_gen_piece_mask() for _ in range(n_pieces)]
    if any(spec is None for spec in piece_specs):
        return None
    masks = [spec[0] for spec in piece_specs if spec is not None]
    label_lens = [int(spec[1]) for spec in piece_specs if spec is not None]
    piece_areas = [int(mask.sum()) for mask in masks]
    if piece_areas.count(max(piece_areas)) != 1:
        return None

    min_box_h = max(mask.shape[0] for mask in masks) + 4
    min_box_w = max(mask.shape[1] for mask in masks) + 6
    max_box_h = min(SHADOW_BOX_H_RANGE[1], h - 2 * SHADOW_MARGIN)
    max_box_w = min(SHADOW_BOX_W_RANGE[1], w - 2 * SHADOW_MARGIN)
    if min_box_h > max_box_h or min_box_w > max_box_w:
        return None
    box_h = random.randint(max(SHADOW_BOX_H_RANGE[0], min_box_h), max_box_h)
    box_w = random.randint(max(SHADOW_BOX_W_RANGE[0], min_box_w), max_box_w)
    box_r0 = random.randint(SHADOW_MARGIN, h - box_h - SHADOW_MARGIN)
    box_c0 = random.randint(SHADOW_MARGIN, w - box_w - SHADOW_MARGIN)

    placement = _place_pieces(h, w, masks, box_r0, box_c0, box_h, box_w)
    if placement is None:
        return None
    anchors, piece_cells = placement

    shadow = set().union(*piece_cells)
    connect_order = list(range(n_pieces))
    random.shuffle(connect_order)
    for i in range(1, n_pieces):
        a_idx = connect_order[i - 1]
        b_idx = connect_order[i]
        for cell in _random_manhattan_path(anchors[a_idx], anchors[b_idx]):
            shadow.add(cell)

    box_area = box_h * box_w
    min_target = max(len(shadow) + 6, int(box_area * SHADOW_DENSITY_RANGE[0]))
    max_target = max(min_target, int(box_area * SHADOW_DENSITY_RANGE[1]))
    target_area = random.randint(min_target, min(box_area - 1, max_target))
    shadow = _grow_shadow(shadow, box_r0, box_c0, box_h, box_w, target_area)

    if any(not cells <= shadow for cells in piece_cells):
        return None
    if len(shadow - set().union(*piece_cells)) == 0:
        return None

    edge_positions = _place_edge_fragments(h, w, shadow, masks)
    if edge_positions is None:
        return None

    grid = np.full((h, w), BG_COLOR, dtype=int)
    for r, c in shadow:
        grid[r, c] = SHADOW_COLOR

    for color, anchor, label_len in zip(colors, anchors, label_lens):
        ar, ac = anchor
        grid[ar, ac] = color
        if label_len == 2:
            grid[ar, ac + 1] = color

    for color, mask, (top, left) in zip(colors, masks, edge_positions):
        cells = _mask_to_cells(mask, top, left)
        for r, c in cells:
            grid[r, c] = color

    expected = _build_expected_output(h, w, shadow, piece_cells, colors)
    if not np.array_equal(_solve(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
