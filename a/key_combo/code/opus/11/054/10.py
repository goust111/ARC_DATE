from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

RECT_COLORS = (BLUE, GREEN, YELLOW, ORANGE)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GENERATION_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc2 = stack.pop()
            comp.append((cr, cc2))
            for dr, dc in DIRS4:
                nr = cr + dr
                nc = cc2 + dc
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _border_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = bbox
    out[r0 : r1 + 1, c0] = True
    out[r0 : r1 + 1, c1] = True
    out[r0, c0 : c1 + 1] = True
    out[r1, c0 : c1 + 1] = True
    return out


def _is_hollow_rectangle(mask: np.ndarray) -> bool:
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    if not cells:
        return False
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return False
    return np.array_equal(mask, _border_mask(mask.shape, (r0, c0, r1, c1)))


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _find_rectangles(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    rectangles: list[dict[str, int]] = []
    colors = sorted({int(v) for v in g.flat if int(v) not in (BLACK, RED)})
    for color in colors:
        for comp in _components(g == color):
            mask = np.zeros_like(g, dtype=bool)
            for r, c in comp:
                mask[r, c] = True
            if not _is_hollow_rectangle(mask):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            rectangles.append(
                {
                    "color": int(color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                }
            )

    assert rectangles, "expected at least one hollow rectangle"
    for rect in rectangles:
        bbox_i = (rect["r0"], rect["c0"], rect["r1"], rect["c1"])
        rect["depth"] = 1 + sum(
            1
            for other in rectangles
            if other is not rect
            and _contains(
                (other["r0"], other["c0"], other["r1"], other["c1"]),
                bbox_i,
            )
        )
    return sorted(rectangles, key=lambda rect: (rect["depth"], rect["r0"], rect["c0"]))


def _growth_cells(rect: dict[str, int], shape: tuple[int, int], *, flip_parity: bool = False) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
    odd_depth = rect["depth"] % 2 == 1
    if flip_parity:
        odd_depth = not odd_depth

    cells: list[tuple[int, int]] = []
    if odd_depth:
        for cc in (c0 - 1, c1 + 1):
            if 0 <= cc < w:
                for rr in range(r0, r1 + 1):
                    cells.append((rr, cc))
    else:
        for rr in (r0 - 1, r1 + 1):
            if 0 <= rr < h:
                for cc in range(c0, c1 + 1):
                    cells.append((rr, cc))
    return cells


def _apply_growth(
    grid: np.ndarray,
    rectangles: list[dict[str, int]],
    *,
    flip_parity: bool = False,
    ignore_red: bool = False,
    force_horizontal: bool = False,
    force_vertical: bool = False,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    for rect in rectangles:
        if force_horizontal:
            cells = []
            for cc in (rect["c0"] - 1, rect["c1"] + 1):
                if 0 <= cc < w:
                    for rr in range(rect["r0"], rect["r1"] + 1):
                        cells.append((rr, cc))
        elif force_vertical:
            cells = []
            for rr in (rect["r0"] - 1, rect["r1"] + 1):
                if 0 <= rr < h:
                    for cc in range(rect["c0"], rect["c1"] + 1):
                        cells.append((rr, cc))
        else:
            cells = _growth_cells(rect, g.shape, flip_parity=flip_parity)

        for rr, cc in cells:
            if not ignore_red and int(g[rr, cc]) == RED:
                continue
            out[rr, cc] = int(rect["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each hollow rectangle outward along the axis chosen by its nesting depth, while leaving red protection pixels untouched.

    Input structure:
    - A black background contains one nested family of 3 or 4 one-cell-thick hollow rectangles in distinct non-red colors.
    - The rectangles are strictly nested with black gaps between neighboring borders, so each bbox defines one unambiguous depth layer.
    - Extra red singleton pixels may appear outside the rectangle borders, including on cells where a growth stripe would otherwise be drawn.

    Rule:
    - Detect every hollow rectangle and compute its nesting depth: the outermost frame has depth 1, the next has depth 2, and so on inward.
    - Odd-depth rectangles grow horizontally by adding one same-colored column immediately outside both their left and right borders.
    - Even-depth rectangles grow vertically by adding one same-colored row immediately outside both their top and bottom borders.
    - If a target growth cell is red, leave it red; all other target growth cells are filled with the rectangle's color.
    - Keep the original rectangle borders, the black interiors, and every non-target background cell unchanged.

    Key insight:
    - The nested order is not used for recoloring the layers themselves; its parity selects the growth axis, and the red pixels act only as local exemptions on those growth stripes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rectangles = _find_rectangles(g)
    return _apply_growth(g, rectangles)


def _draw_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)


def _make_nested_scene() -> tuple[np.ndarray, list[dict[str, int]]]:
    n_layers = random.choice([3, 4])

    max_inner_h = 7 if n_layers == 3 else 5
    max_inner_w = 8 if n_layers == 3 else 6
    inner_h = random.randint(3, max_inner_h)
    inner_w = random.randint(3, max_inner_w)

    row_insets = [random.choice([2, 3]) for _ in range(n_layers - 1)]
    col_insets = [random.choice([2, 3]) for _ in range(n_layers - 1)]
    outer_h = inner_h + 2 * sum(row_insets)
    outer_w = inner_w + 2 * sum(col_insets)

    top = random.choice([2, 3])
    bottom = random.choice([2, 3])
    left = random.choice([2, 3])
    right = random.choice([2, 3])

    h = outer_h + top + bottom
    w = outer_w + left + right
    if max(h, w) > 30:
        raise ValueError("scene too large")

    grid = np.zeros((h, w), dtype=int)
    colors = random.sample(list(RECT_COLORS), k=n_layers)
    rectangles: list[dict[str, int]] = []

    r0, c0 = top, left
    r1, c1 = top + outer_h - 1, left + outer_w - 1
    for depth in range(1, n_layers + 1):
        color = colors[depth - 1]
        rect = {
            "color": int(color),
            "r0": int(r0),
            "c0": int(c0),
            "r1": int(r1),
            "c1": int(c1),
            "depth": int(depth),
        }
        rectangles.append(rect)
        _draw_border(grid, (r0, c0, r1, c1), color)
        if depth < n_layers:
            r0 += row_insets[depth - 1]
            c0 += col_insets[depth - 1]
            r1 -= row_insets[depth - 1]
            c1 -= col_insets[depth - 1]

    return grid, rectangles


def _nearby_distractor_cells(
    grid: np.ndarray,
    rectangles: list[dict[str, int]],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    candidates: set[tuple[int, int]] = set()
    for rect in rectangles:
        r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
        for rr in range(max(0, r0 - 2), min(h, r1 + 3)):
            for cc in range(max(0, c0 - 2), min(w, c1 + 3)):
                if int(grid[rr, cc]) != BLACK or (rr, cc) in forbidden:
                    continue
                if r0 <= rr <= r1 and c0 <= cc <= c1:
                    continue
                manhattan_to_bbox = 0
                if rr < r0:
                    manhattan_to_bbox += r0 - rr
                elif rr > r1:
                    manhattan_to_bbox += rr - r1
                if cc < c0:
                    manhattan_to_bbox += c0 - cc
                elif cc > c1:
                    manhattan_to_bbox += cc - c1
                if 1 <= manhattan_to_bbox <= 2:
                    candidates.add((rr, cc))
    return list(candidates)


def _validate_input(grid: np.ndarray) -> None:
    rectangles = _find_rectangles(grid)
    assert 3 <= len(rectangles) <= 4
    assert [rect["depth"] for rect in rectangles] == list(range(1, len(rectangles) + 1))
    assert len({rect["color"] for rect in rectangles}) == len(rectangles)
    for rect in rectangles:
        mask = _border_mask(grid.shape, (rect["r0"], rect["c0"], rect["r1"], rect["c1"]))
        assert np.all(grid[mask] == rect["color"])

    active_growth = set()
    odd_open = 0
    even_open = 0
    for rect in rectangles:
        cells = _growth_cells(rect, grid.shape)
        for rr, cc in cells:
            if int(grid[rr, cc]) == RED:
                active_growth.add((rr, cc))
            else:
                if rect["depth"] % 2 == 1:
                    odd_open += 1
                else:
                    even_open += 1
    assert len(active_growth) >= 2
    assert odd_open > 0 and even_open > 0
    assert np.all(grid != RED) or len(active_growth) > 0

    out = main(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid)
    assert not np.array_equal(out, _apply_growth(grid, rectangles, flip_parity=True))
    assert not np.array_equal(out, _apply_growth(grid, rectangles, ignore_red=True))
    assert not np.array_equal(out, _apply_growth(grid, rectangles, force_horizontal=True))
    assert not np.array_equal(out, _apply_growth(grid, rectangles, force_vertical=True))


def _gen_attempt() -> np.ndarray | None:
    try:
        grid, rectangles = _make_nested_scene()
    except ValueError:
        return None

    growth_by_depth = {rect["depth"]: _growth_cells(rect, grid.shape) for rect in rectangles}
    odd_cells = sorted(
        {
            cell
            for rect in rectangles
            if rect["depth"] % 2 == 1
            for cell in growth_by_depth[rect["depth"]]
            if int(grid[cell]) == BLACK
        }
    )
    even_cells = sorted(
        {
            cell
            for rect in rectangles
            if rect["depth"] % 2 == 0
            for cell in growth_by_depth[rect["depth"]]
            if int(grid[cell]) == BLACK
        }
    )
    if not odd_cells or not even_cells:
        return None

    active_reds: set[tuple[int, int]] = set()
    active_reds.add(random.choice(odd_cells))
    active_reds.add(random.choice(even_cells))

    remaining_growth = [cell for cell in sorted(set(odd_cells) | set(even_cells)) if cell not in active_reds]
    random.shuffle(remaining_growth)
    extra_active = random.randint(0, min(3, len(remaining_growth)))
    for cell in remaining_growth[:extra_active]:
        active_reds.add(cell)

    forbidden = set(active_reds)
    forbidden.update({cell for rect in rectangles for cell in _growth_cells(rect, grid.shape)})
    distractor_candidates = _nearby_distractor_cells(grid, rectangles, forbidden)
    random.shuffle(distractor_candidates)
    distractor_count = random.randint(0, min(3, len(distractor_candidates)))
    distractor_reds = set(distractor_candidates[:distractor_count])

    for rr, cc in active_reds | distractor_reds:
        grid[rr, cc] = RED

    try:
        _validate_input(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-rectangle growth input")
