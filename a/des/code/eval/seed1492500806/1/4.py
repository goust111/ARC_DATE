import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (20, 32)
COLOR_COUNT_RANGE = (2, 4)
RECT_COUNT_RANGE = (1, 4)
RECT_H_VALUES = tuple(range(1, 11))
RECT_W_VALUES = tuple(range(1, 13))
LEGEND_INNER_WIDTHS = (1, 2, 3)
MAX_PLACE_TRIES = 250
GEN_RETRIES = 400


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, tuple[int, int, int, int]]] = []
    for comp in _components(grid == GREY, connectivity=4):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(sub[0, :] == GREY):
            continue
        if not np.all(sub[-1, :] == GREY):
            continue
        if not np.all(sub[:, 0] == GREY):
            continue
        if not np.all(sub[:, -1] == GREY):
            continue
        if h > 2 and w > 2 and np.any(sub[1:-1, 1:-1] == GREY):
            continue
        if int(np.count_nonzero(sub == GREY)) != 2 * (h + w) - 4:
            continue

        inner = sub[1:-1, 1:-1]
        inner_colors = inner[(inner != BLACK) & (inner != GREY)]
        if inner_colors.size == 0:
            continue

        candidates.append((r1, c1, h * w, (r0, c0, r1, c1)))

    assert candidates, "main: could not find the grey legend frame"
    return max(candidates)[-1]


def _extract_legend_order(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[int]:
    r0, c0, r1, c1 = bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    pixels = [
        (int(r), int(c), int(inner[r, c]))
        for r, c in np.argwhere((inner != BLACK) & (inner != GREY))
    ]
    assert pixels, "main: legend must contain at least one color pixel"
    assert len({c for _, c, _ in pixels}) == 1, "main: legend colors must form a single column"
    assert len({r for r, _, _ in pixels}) == len(pixels), "main: legend rows must be unique"

    pixels.sort(key=lambda item: (item[0], item[1]))
    order = [color for _, _, color in pixels]
    assert len(order) == len(set(order)), "main: legend colors must be unique"
    return order


def _collect_group_rectangles(
    grid: np.ndarray, legend_bbox: tuple[int, int, int, int], legend_order: list[int]
) -> dict[int, list[tuple[int, int, int]]]:
    outside = np.ones_like(grid, dtype=bool)
    r0, c0, r1, c1 = legend_bbox
    outside[r0 : r1 + 1, c0 : c1 + 1] = False

    visible_colors = set(np.unique(grid[outside]).tolist())
    unexpected = visible_colors - {BLACK, GREY} - set(legend_order)
    assert not unexpected, "main: found colors outside the legend that are not listed in the legend"

    groups: dict[int, list[tuple[int, int, int]]] = {}
    for color in legend_order:
        rects: list[tuple[int, int, int]] = []
        for comp in _components((grid == color) & outside, connectivity=4):
            rr0, cc0, rr1, cc1 = _bbox(comp)
            h = rr1 - rr0 + 1
            w = cc1 - cc0 + 1
            sub = grid[rr0 : rr1 + 1, cc0 : cc1 + 1]
            assert sub.shape == (h, w)
            assert int(len(comp)) == h * w, "main: each colored object must be a solid rectangle"
            assert np.all(sub == color), "main: each colored object must be monochrome"
            rects.append((h, w, h * w))

        assert rects, f"main: legend color {color} has no rectangles in the scene"
        rects.sort(key=lambda item: (item[2], item[0], item[1]), reverse=True)
        areas = [area for _, _, area in rects]
        assert len(areas) == len(set(areas)), "main: rectangle areas within a color group must be unique"
        groups[color] = rects

    return groups


def _stack_group(color: int, rects: list[tuple[int, int, int]]) -> np.ndarray:
    base_h, base_w, _ = rects[0]
    canvas = np.zeros((base_h, base_w), dtype=int)
    for layer_idx, (h, w, _area) in enumerate(rects):
        fill = color if layer_idx % 2 == 0 else GREY
        top = (base_h - h) // 2
        left = (base_w - w) // 2
        canvas[top : top + h, left : left + w] = fill
    return canvas


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered same-color rectangles, sort each group by area, and turn
    each group into a nested concentric stack. Legend order controls the
    left-to-right order of the output groups.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "main: expected a 2D grid"

    legend_bbox = _find_legend_bbox(g)
    legend_order = _extract_legend_order(g, legend_bbox)
    groups = _collect_group_rectangles(g, legend_bbox, legend_order)

    stacks = [_stack_group(color, groups[color]) for color in legend_order]
    out_h = max(stack.shape[0] for stack in stacks)
    out_w = sum(stack.shape[1] for stack in stacks) + len(stacks) - 1

    out = np.zeros((out_h, out_w), dtype=int)
    cursor = 0
    for stack in stacks:
        h, w = stack.shape
        out[:h, cursor : cursor + w] = stack
        cursor += w + 1
    return out


def _sample_group_dims(rng: random.Random, count: int) -> list[tuple[int, int]] | None:
    parity_h = rng.randrange(2)
    parity_w = rng.randrange(2)
    heights = [v for v in RECT_H_VALUES if v % 2 == parity_h]
    widths = [v for v in RECT_W_VALUES if v % 2 == parity_w]
    if len(heights) < count or len(widths) < count:
        return None

    chosen_h = sorted(rng.sample(heights, count), reverse=True)
    chosen_w = sorted(rng.sample(widths, count), reverse=True)
    return [(h, w) for h, w in zip(chosen_h, chosen_w)]


def _reserve(mask: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> None:
    h_total, w_total = mask.shape
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(h_total, r + h + pad)
    c1 = min(w_total, c + w + pad)
    mask[r0:r1, c0:c1] = True


def _can_place(mask: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> bool:
    h_total, w_total = mask.shape
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(h_total, r + h + pad)
    c1 = min(w_total, c + w + pad)
    return not bool(mask[r0:r1, c0:c1].any())


def _paint_legend(
    grid: np.ndarray,
    legend_order: list[int],
    legend_h: int,
    legend_w: int,
    top_pad: int,
    color_col: int,
) -> tuple[int, int, int, int]:
    h, w = grid.shape
    r0 = h - legend_h
    c0 = w - legend_w
    r1 = h - 1
    c1 = w - 1

    grid[r0:r1 + 1, c0:c1 + 1] = BLACK
    grid[r0, c0:c1 + 1] = GREY
    grid[r1, c0:c1 + 1] = GREY
    grid[r0:r1 + 1, c0] = GREY
    grid[r0:r1 + 1, c1] = GREY

    for idx, color in enumerate(legend_order):
        rr = r0 + 1 + top_pad + idx
        cc = c0 + 1 + color_col
        grid[rr, cc] = color

    return r0, c0, r1, c1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    color_count = rng.randint(*COLOR_COUNT_RANGE)
    colors = rng.sample(COLOR_POOL, color_count)
    legend_order = colors[:]
    rng.shuffle(legend_order)

    rect_counts = [rng.randint(*RECT_COUNT_RANGE) for _ in colors]
    if all(count == 1 for count in rect_counts):
        rect_counts[rng.randrange(color_count)] = rng.randint(2, RECT_COUNT_RANGE[1])

    groups: dict[int, list[tuple[int, int]]] = {}
    all_rects: list[tuple[int, int, int]] = []
    for color, count in zip(colors, rect_counts):
        dims = _sample_group_dims(rng, count)
        if dims is None:
            return None
        groups[color] = dims
        for h, w in dims:
            all_rects.append((h, w, color))

    inner_w = rng.choice(LEGEND_INNER_WIDTHS)
    top_pad = rng.randint(0, 1)
    bottom_pad = rng.randint(0, 1)
    color_col = rng.randrange(inner_w)
    legend_h = color_count + top_pad + bottom_pad + 2
    legend_w = inner_w + 2

    largest_h = max(h for h, _, _ in all_rects)
    largest_w = max(w for _, w, _ in all_rects)
    padded_area = sum((h + 2) * (w + 2) for h, w, _ in all_rects) + (legend_h + 2) * (legend_w + 2)

    min_h = max(GRID_H_RANGE[0], largest_h + 2, legend_h + 1)
    min_w = max(GRID_W_RANGE[0], largest_w + 2, legend_w + 1)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    for _ in range(12):
        h = rng.randint(min_h, GRID_H_RANGE[1])
        w = rng.randint(min_w, GRID_W_RANGE[1])
        if padded_area > int(0.48 * h * w):
            continue

        grid = np.zeros((h, w), dtype=int)
        blocked = np.zeros((h, w), dtype=bool)

        legend_bbox = _paint_legend(grid, legend_order, legend_h, legend_w, top_pad, color_col)
        lr0, lc0, lr1, lc1 = legend_bbox
        _reserve(blocked, lr0, lc0, lr1 - lr0 + 1, lc1 - lc0 + 1, pad=1)

        placed = True
        items = sorted(all_rects, key=lambda item: item[0] * item[1], reverse=True)
        for rect_h, rect_w, color in items:
            if h - rect_h - 2 < 1 or w - rect_w - 2 < 1:
                placed = False
                break

            done = False
            for _ in range(MAX_PLACE_TRIES):
                rr = rng.randint(1, h - rect_h - 1)
                cc = rng.randint(1, w - rect_w - 1)
                if not _can_place(blocked, rr, cc, rect_h, rect_w, pad=1):
                    continue
                grid[rr : rr + rect_h, cc : cc + rect_w] = color
                _reserve(blocked, rr, cc, rect_h, rect_w, pad=1)
                done = True
                break
            if not done:
                placed = False
                break

        if not placed:
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue

        if out.size == 0 or out.shape[0] < 1 or out.shape[1] < 1:
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to sample a valid grid")
