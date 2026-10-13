from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
SEPARATOR = 9

HOLE_COUNTS = (0, 1, 2)
LEGEND_COLORS = (1, 2, 3, 4, 6, 7, 8)

MAX_GRID_H = 30
MAX_GRID_W = 30
GEN_RETRIES = 400
MUTATION_RETRIES = 32
def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((int(r), int(c)))
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _trim(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg_mask = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg_mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            touches_border = False
            while stack:
                r, c = stack.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_border = True
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg_mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            if not touches_border:
                holes += 1
    return holes


def _find_separator(g: np.ndarray) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for c in range(g.shape[1]):
        vals = np.unique(g[:, c])
        if vals.size == 1 and int(vals[0]) != BG:
            candidates.append((int(c), int(vals[0])))
    assert len(candidates) == 1, "expected exactly one full-height non-background separator"
    return candidates[0]


def _legend_entries(right: np.ndarray) -> tuple[list[tuple[int, int, int]], int]:
    entries: list[tuple[int, int, int]] = []
    legend_end = -1
    for r in range(right.shape[0]):
        cols = np.flatnonzero(right[r] != BG)
        if cols.size == 0:
            continue
        colors = np.unique(right[r, cols])
        assert colors.size == 1, "each legend row must use one color"
        assert int(cols[0]) == 0, "legend square must be in the leftmost column of the right area"
        holes = int(cols.size) - 1
        entries.append((int(r), holes, int(colors[0])))
        legend_end = max(legend_end, int(cols[-1]))
    assert entries, "legend must exist on the right side"
    hole_counts = [holes for _, holes, _ in entries]
    assert len(set(hole_counts)) == len(hole_counts), "legend hole counts must be unique"
    return entries, legend_end + 2


def _grey_shapes(left: np.ndarray) -> list[dict[str, object]]:
    mask = left == GREY
    shapes: list[dict[str, object]] = []
    for comp in _components(mask):
        r0, c0, r1, c1 = _bbox(comp)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            sub[r - r0, c - c0] = True
        shapes.append(
            {
                "mask": sub,
                "area": int(sub.sum()),
                "holes": _count_holes(sub),
                "height": int(sub.shape[0]),
                "width": int(sub.shape[1]),
            }
        )
    return shapes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group grey shapes by hole count, decode the hole-count legend on the right,
    then recolor and repack each group onto the legend's target row.
    """
    g = np.asarray(grid, dtype=int)
    sep_col, sep_color = _find_separator(g)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]

    entries, _legend_limit = _legend_entries(right)
    groups: dict[int, list[dict[str, object]]] = {}
    for shape in _grey_shapes(left):
        groups.setdefault(int(shape["holes"]), []).append(shape)

    out = np.full_like(g, BG)
    out[:, sep_col] = int(sep_color)

    for row, holes, color in entries:
        shapes = sorted(
            groups.get(holes, []),
            key=lambda item: (
                int(item["area"]),
                int(item["height"]),
                int(item["width"]),
            ),
        )
        cursor = 0
        for shape in shapes:
            mask = shape["mask"]
            h, w = mask.shape
            c0 = sep_col + 1 + cursor
            assert row + h <= out.shape[0]
            assert c0 + w <= out.shape[1]
            out[row : row + h, c0 : c0 + w][mask] = int(color)
            cursor += w + 1
    return out


def _hole_rects(hole_count: int, outer_h: int, outer_w: int) -> list[tuple[int, int, int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []
    for _ in range(80):
        rects.clear()
        ok = True
        for _hole_idx in range(hole_count):
            choices: list[tuple[int, int]] = []
            for hh in (1, 2):
                for ww in (1, 2):
                    if outer_h - hh - 2 >= 1 and outer_w - ww - 2 >= 1:
                        choices.append((hh, ww))
            if not choices:
                return None
            hh, ww = random.choice(choices)
            placed = False
            for _ in range(80):
                r = random.randint(1, outer_h - hh - 1)
                c = random.randint(1, outer_w - ww - 1)
                good = True
                for er, ec, eh, ew in rects:
                    if not (
                        r + hh + 1 <= er
                        or er + eh + 1 <= r
                        or c + ww + 1 <= ec
                        or ec + ew + 1 <= c
                    ):
                        good = False
                        break
                if good:
                    rects.append((r, c, hh, ww))
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return list(rects)
    return None


def _mutate_shape(mask: np.ndarray, hole_count: int) -> np.ndarray:
    current = mask.copy()
    for _ in range(random.randint(8, 18)):
        accepted = False
        for _try in range(MUTATION_RETRIES):
            cand = current.copy()
            true_coords = np.argwhere(cand)
            r0, c0 = true_coords.min(axis=0)
            r1, c1 = true_coords.max(axis=0)
            if random.random() < 0.55:
                cells: list[tuple[int, int]] = []
                for r in range(max(0, int(r0) - 1), min(cand.shape[0], int(r1) + 2)):
                    for c in range(max(0, int(c0) - 1), min(cand.shape[1], int(c1) + 2)):
                        if cand[r, c]:
                            continue
                        if any(
                            0 <= rr < cand.shape[0] and 0 <= cc < cand.shape[1] and cand[rr, cc]
                            for rr, cc in _neighbors4(r, c)
                        ):
                            cells.append((r, c))
                if not cells:
                    continue
                rr, cc = random.choice(cells)
                cand[rr, cc] = True
            else:
                cells = []
                for r, c in map(tuple, true_coords):
                    if any(
                        not (0 <= rr < cand.shape[0] and 0 <= cc < cand.shape[1]) or not cand[rr, cc]
                        for rr, cc in _neighbors4(int(r), int(c))
                    ):
                        cells.append((int(r), int(c)))
                if len(cells) <= 1:
                    continue
                rr, cc = random.choice(cells)
                cand[rr, cc] = False
            if cand.sum() == 0:
                continue
            trimmed = _trim(cand)
            if _is_connected(trimmed) and _count_holes(trimmed) == hole_count:
                current = cand
                accepted = True
                break
        if not accepted:
            continue
    trimmed = _trim(current)
    if random.random() < 0.5:
        trimmed = np.fliplr(trimmed)
    if random.random() < 0.5:
        trimmed = np.flipud(trimmed)
    trimmed = np.rot90(trimmed, k=random.randint(0, 3))
    return trimmed.copy()


def _make_shape(hole_count: int) -> np.ndarray | None:
    if hole_count == 0:
        outer_h = random.randint(3, 4)
        outer_w = random.randint(3, 5)
    elif hole_count == 1:
        outer_h = random.randint(4, 6)
        outer_w = random.randint(4, 6)
    else:
        outer_h = random.randint(5, 6)
        outer_w = random.randint(5, 6)

    rects = _hole_rects(hole_count, outer_h, outer_w)
    if rects is None:
        return None

    pad = 1
    canvas = np.zeros((outer_h + 2 * pad, outer_w + 2 * pad), dtype=bool)
    canvas[pad : pad + outer_h, pad : pad + outer_w] = True
    for r, c, hh, ww in rects:
        canvas[pad + r : pad + r + hh, pad + c : pad + c + ww] = False

    shape = _mutate_shape(canvas, hole_count)
    if not _is_connected(shape) or _count_holes(shape) != hole_count:
        return None
    if hole_count == 0 and (shape.shape[0] > 5 or shape.shape[1] > 6):
        return None
    if hole_count == 1 and (shape.shape[0] > 7 or shape.shape[1] > 7):
        return None
    if hole_count == 2 and (shape.shape[0] > 7 or shape.shape[1] > 8):
        return None
    return shape


def _sample_group_shapes() -> dict[int, list[np.ndarray]] | None:
    counts = {holes: 1 for holes in HOLE_COUNTS}
    counts[random.choice((0, 1))] = 2

    groups: dict[int, list[np.ndarray]] = {}
    for holes in HOLE_COUNTS:
        shapes: list[np.ndarray] = []
        used_areas: set[int] = set()
        for _ in range(counts[holes]):
            for _try in range(100):
                shape = _make_shape(holes)
                if shape is None:
                    continue
                area = int(shape.sum())
                if area in used_areas:
                    continue
                used_areas.add(area)
                shapes.append(shape)
                break
            else:
                return None
        groups[holes] = shapes
    return groups


def _pack_width(shapes: list[np.ndarray]) -> int:
    return sum(int(shape.shape[1]) for shape in shapes) + max(0, len(shapes) - 1)


def _place_left(shapes: list[np.ndarray], grid_h: int, left_w: int) -> list[tuple[int, int]] | None:
    if not shapes:
        return []
    if any(grid_h < shape.shape[0] or left_w < shape.shape[1] for shape in shapes):
        return None

    order = list(range(len(shapes)))
    order.sort(
        key=lambda idx: (
            shapes[idx].shape[0] * shapes[idx].shape[1],
            int(shapes[idx].sum()),
        ),
        reverse=True,
    )

    blocked = np.zeros((grid_h, left_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(shapes)

    for idx in order:
        shape = shapes[idx]
        h, w = shape.shape
        spots: list[tuple[int, int]] = []
        for r in range(0, grid_h - h + 1):
            for c in range(0, left_w - w + 1):
                rr0 = max(0, r - 1)
                rr1 = min(grid_h, r + h + 1)
                cc0 = max(0, c - 1)
                cc1 = min(left_w, c + w + 1)
                if blocked[rr0:rr1, cc0:cc1].any():
                    continue
                spots.append((r, c))
        if not spots:
            return None
        r, c = random.choice(spots)
        placements[idx] = (r, c)
        rr0 = max(0, r - 1)
        rr1 = min(grid_h, r + h + 1)
        cc0 = max(0, c - 1)
        cc1 = min(left_w, c + w + 1)
        blocked[rr0:rr1, cc0:cc1] = True

    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _build_grid() -> np.ndarray | None:
    groups = _sample_group_shapes()
    if groups is None:
        return None

    row_order = random.sample(list(HOLE_COUNTS), len(HOLE_COUNTS))
    color_by_hole = dict(zip(HOLE_COUNTS, random.sample(LEGEND_COLORS, len(HOLE_COUNTS))))

    row_heights = {holes: max(shape.shape[0] for shape in groups[holes]) for holes in HOLE_COUNTS}
    cursor = random.randint(1, 2)
    row_by_hole: dict[int, int] = {}
    for i, holes in enumerate(row_order):
        row_by_hole[holes] = cursor
        cursor += row_heights[holes]
        if i + 1 < len(row_order):
            cursor += random.randint(1, 2)
    grid_h = cursor + random.randint(1, 2)
    if grid_h > MAX_GRID_H:
        return None

    legend_width = max(HOLE_COUNTS) + 1
    right_w = max(legend_width, max(_pack_width(groups[holes]) for holes in HOLE_COUNTS)) + random.randint(1, 2)

    all_shapes = [shape for holes in HOLE_COUNTS for shape in groups[holes]]
    max_left_w = MAX_GRID_W - 1 - right_w
    min_left_w = max(max(shape.shape[1] for shape in all_shapes) + 1, 7)
    if max_left_w < min_left_w:
        return None

    base_left_w = None
    placements = None
    for left_w in range(min_left_w, max_left_w + 1):
        placements = _place_left(all_shapes, grid_h, left_w)
        if placements is not None:
            base_left_w = left_w
            break
    if placements is None or base_left_w is None:
        return None

    extra = min(2, max_left_w - base_left_w)
    left_w = base_left_w + random.randint(0, extra)
    placements = _place_left(all_shapes, grid_h, left_w)
    if placements is None:
        left_w = base_left_w
        placements = _place_left(all_shapes, grid_h, left_w)
    if placements is None:
        return None

    grid = np.full((grid_h, left_w + 1 + right_w), BG, dtype=int)
    grid[:, left_w] = SEPARATOR

    shape_index = 0
    for holes in HOLE_COUNTS:
        for shape in groups[holes]:
            r, c = placements[shape_index]
            h, w = shape.shape
            grid[r : r + h, c : c + w][shape] = GREY
            shape_index += 1

    right0 = left_w + 1
    for holes in row_order:
        row = row_by_hole[holes]
        color = color_by_hole[holes]
        grid[row, right0 : right0 + holes + 1] = color

    out = main(grid)

    # Keep the generator aligned with the solver assumptions.
    right = grid[:, right0:]
    entries, _legend_limit = _legend_entries(right)
    assert len(entries) == len(HOLE_COUNTS)
    assert np.all(out[:, :left_w] != GREY)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
