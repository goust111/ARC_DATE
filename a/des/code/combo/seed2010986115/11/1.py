from __future__ import annotations

import random
from typing import Iterable

import numpy as np


GREY = 5
BG = 0
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _most_common_color(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[counts.argmax()])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for rr, cc in _neighbors4(r, c, h, w):
                    if mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((int(rr), int(cc)))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_from_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, int]:
    r0, c0, r1, c1 = _bbox(coords)
    color = int(grid[coords[0]])
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    assert len(coords) == area, "each detected object must be a solid rectangle"
    assert bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color))
    return {
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
        "color": int(color),
    }


def _fill_rect(out: np.ndarray, rect: dict[str, int], color: int) -> None:
    out[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = int(color)


def _shared_edge_len(a: dict[str, int], b: dict[str, int]) -> int:
    best = 0
    if a["r1"] + 1 == b["r0"] or b["r1"] + 1 == a["r0"]:
        overlap = min(a["c1"], b["c1"]) - max(a["c0"], b["c0"]) + 1
        if overlap > 0:
            best = max(best, int(overlap))
    if a["c1"] + 1 == b["c0"] or b["c1"] + 1 == a["c0"]:
        overlap = min(a["r1"], b["r1"]) - max(a["r0"], b["r0"]) + 1
        if overlap > 0:
            best = max(best, int(overlap))
    return int(best)


def _extract_scene(
    g: np.ndarray,
) -> tuple[int, list[dict[str, int]], list[dict[str, int]]]:
    bg = _most_common_color(g)
    assert np.any(g == GREY), "the puzzle requires a grey template"

    grey_rects = [_rect_from_component(g, comp) for comp in _components(g == GREY)]
    assert grey_rects, "expected at least one visible grey template block"

    fragment_rects: list[dict[str, int]] = []
    for color in sorted(set(map(int, np.unique(g)))):
        if color in (bg, GREY):
            continue
        comps = _components(g == color)
        assert len(comps) == 1, "each fragment color must form exactly one component"
        fragment_rects.append(_rect_from_component(g, comps[0]))
    assert fragment_rects, "expected at least one scattered fragment"
    return bg, grey_rects, fragment_rects


def _reconstruct_missing(
    grey_rects: list[dict[str, int]],
    fragment_rects: list[dict[str, int]],
) -> list[dict[str, int]]:
    row_top_by_height: dict[int, int] = {}
    col_left_by_width: dict[int, int] = {}

    for rect in grey_rects:
        h = rect["h"]
        w = rect["w"]
        row_top_by_height.setdefault(h, rect["r0"])
        col_left_by_width.setdefault(w, rect["c0"])
        assert row_top_by_height[h] == rect["r0"], "same-height grey rows must align"
        assert col_left_by_width[w] == rect["c0"], "same-width grey columns must align"

    missing_rects: list[dict[str, int]] = []
    used_positions: set[tuple[int, int]] = set()
    for frag in fragment_rects:
        h = frag["h"]
        w = frag["w"]
        assert h in row_top_by_height, "every fragment height must appear in a visible grey row"
        assert w in col_left_by_width, "every fragment width must appear in a visible grey column"
        r0 = row_top_by_height[h]
        c0 = col_left_by_width[w]
        key = (int(r0), int(c0))
        assert key not in used_positions, "fragment sizes must map to distinct missing slots"
        used_positions.add(key)
        missing_rects.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r0 + h - 1),
                "c1": int(c0 + w - 1),
                "h": int(h),
                "w": int(w),
                "color": int(frag["color"]),
            }
        )
    return missing_rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match scattered rectangular fragments to the grey template's missing blocks,
    refill those blocks with fragment colors, then recolor every visible grey block
    using the adjacent filled fragment with the longest shared edge.
    """
    g = np.asarray(grid, dtype=int)
    bg, grey_rects, fragment_rects = _extract_scene(g)
    missing_rects = _reconstruct_missing(grey_rects, fragment_rects)

    out = np.full(g.shape, bg, dtype=int)
    for rect in missing_rects:
        _fill_rect(out, rect, rect["color"])

    for rect in grey_rects:
        choices: list[tuple[int, int, int, int]] = []
        for miss in missing_rects:
            edge = _shared_edge_len(rect, miss)
            if edge > 0:
                choices.append((int(edge), int(miss["r0"]), int(miss["c0"]), int(miss["color"])))
        assert choices, "each visible grey block must touch at least one filled fragment"
        best_edge = max(item[0] for item in choices)
        tied = [item for item in choices if item[0] == best_edge]
        _, _, _, color = min(tied, key=lambda item: (item[1], item[2]))
        _fill_rect(out, rect, color)

    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 26)
MACRO_COUNT_CHOICES = (2, 3, 4)
SIZE_POOL = (2, 3, 4, 5)
OCCUPIED_EXTRA_MAX = 3
MAX_SHAPE_CELLS = 9
GREY_COUNT_MAX = 6
MISSING_COUNT_RANGE = (2, 5)
SHAPE_RETRIES = 120
PIECE_PLACE_RETRIES = 250
GEN_RETRIES = 400
TIE_PROB = 0.4


def _macro_neighbors(cell: tuple[int, int], rows: int, cols: int) -> Iterable[tuple[int, int]]:
    r, c = cell
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            yield int(rr), int(cc)


def _starts(sizes: list[int], offset: int = 0) -> list[int]:
    cur = int(offset)
    out: list[int] = []
    for size in sizes:
        out.append(cur)
        cur += int(size)
    return out


def _random_connected_shape(rows: int, cols: int) -> set[tuple[int, int]] | None:
    min_cells = max(rows + cols - 1, 4)
    max_cells = min(rows * cols, rows + cols + OCCUPIED_EXTRA_MAX, MAX_SHAPE_CELLS)
    if min_cells > max_cells:
        return None

    target = random.randint(min_cells, max_cells)
    start = (random.randrange(rows), random.randrange(cols))
    shape = {start}
    frontier = set(_macro_neighbors(start, rows, cols))

    while len(shape) < target and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        shape.add(cell)
        for nb in _macro_neighbors(cell, rows, cols):
            if nb not in shape:
                frontier.add(nb)

    if len(shape) != target:
        return None
    if {r for r, _ in shape} != set(range(rows)):
        return None
    if {c for _, c in shape} != set(range(cols)):
        return None
    return shape


def _choose_grey_cells(
    shape: set[tuple[int, int]],
    rows: int,
    cols: int,
    row_heights: list[int],
    col_widths: list[int],
    want_tie: bool,
) -> tuple[set[tuple[int, int]], bool] | None:
    shape_list = list(shape)
    row_tops = _starts(row_heights, 0)
    col_lefts = _starts(col_widths, 0)
    candidates: list[tuple[set[tuple[int, int]], bool]] = []

    for mask in range(1, 1 << len(shape_list)):
        grey = {shape_list[i] for i in range(len(shape_list)) if (mask >> i) & 1}
        missing = shape - grey
        if not (2 <= len(grey) <= GREY_COUNT_MAX):
            continue
        if not (MISSING_COUNT_RANGE[0] <= len(missing) <= min(MISSING_COUNT_RANGE[1], len(PIECE_COLORS))):
            continue
        if {r for r, _ in grey} != set(range(rows)):
            continue
        if {c for _, c in grey} != set(range(cols)):
            continue

        independent = True
        for cell in grey:
            for nb in _macro_neighbors(cell, rows, cols):
                if nb in grey:
                    independent = False
                    break
            if not independent:
                break
        if not independent:
            continue

        tie_used = False
        valid = True
        for cell in grey:
            r, c = cell
            touching = [nb for nb in _macro_neighbors(cell, rows, cols) if nb in missing]
            if not touching:
                valid = False
                break

            scored: list[tuple[int, int, int]] = []
            for nr, nc in touching:
                if nr == r:
                    edge = row_heights[r]
                else:
                    edge = col_widths[c]
                scored.append((int(edge), int(row_tops[nr]), int(col_lefts[nc])))

            best_edge = max(edge for edge, _, _ in scored)
            best_group = [item for item in scored if item[0] == best_edge]
            topmost = min(item[1] for item in best_group)
            if sum(1 for _, top, _ in best_group if top == topmost) > 1:
                valid = False
                break
            if len(best_group) > 1:
                tie_used = True

        if valid:
            candidates.append((grey, tie_used))

    if not candidates:
        return None

    preferred = [item for item in candidates if item[1] == want_tie]
    return random.choice(preferred or candidates)


def _cell_rects(
    shape: set[tuple[int, int]],
    row_heights: list[int],
    col_widths: list[int],
    top: int,
    left: int,
) -> dict[tuple[int, int], dict[str, int]]:
    row_tops = _starts(row_heights, top)
    col_lefts = _starts(col_widths, left)
    rects: dict[tuple[int, int], dict[str, int]] = {}
    for r, c in shape:
        r0 = row_tops[r]
        c0 = col_lefts[c]
        h = row_heights[r]
        w = col_widths[c]
        rects[(r, c)] = {
            "macro_r": int(r),
            "macro_c": int(c),
            "r0": int(r0),
            "c0": int(c0),
            "r1": int(r0 + h - 1),
            "c1": int(c0 + w - 1),
            "h": int(h),
            "w": int(w),
        }
    return rects


def _boxes_conflict(a: dict[str, int], b: dict[str, int], margin: int = 1) -> bool:
    return not (
        a["r1"] + margin < b["r0"]
        or b["r1"] + margin < a["r0"]
        or a["c1"] + margin < b["c0"]
        or b["c1"] + margin < a["c0"]
    )


def _render_expected(
    grid_shape: tuple[int, int],
    shape: set[tuple[int, int]],
    grey: set[tuple[int, int]],
    rects: dict[tuple[int, int], dict[str, int]],
    row_heights: list[int],
    col_widths: list[int],
    colors_by_missing: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.full(grid_shape, BG, dtype=int)

    for cell, color in colors_by_missing.items():
        rect = rects[cell]
        out[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = int(color)

    for cell in grey:
        r, c = cell
        rect = rects[cell]
        scored: list[tuple[int, int, int, int]] = []
        for nr, nc in _macro_neighbors(cell, len(row_heights), len(col_widths)):
            nb = (nr, nc)
            if nb not in shape or nb in grey:
                continue
            edge = row_heights[r] if nr == r else col_widths[c]
            scored.append((int(edge), int(rects[nb]["r0"]), int(rects[nb]["c0"]), int(colors_by_missing[nb])))
        best_edge = max(item[0] for item in scored)
        tied = [item for item in scored if item[0] == best_edge]
        color = min(tied, key=lambda item: (item[1], item[2]))[3]
        out[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = int(color)

    return out


def _place_fragments(
    grid_h: int,
    grid_w: int,
    template_box: dict[str, int],
    missing_cells: list[tuple[int, int]],
    rects: dict[tuple[int, int], dict[str, int]],
    colors_by_missing: dict[tuple[int, int], int],
) -> dict[tuple[int, int], dict[str, int]] | None:
    placed: dict[tuple[int, int], dict[str, int]] = {}
    blockers = [template_box]
    order = list(missing_cells)
    random.shuffle(order)

    for cell in order:
        spec = rects[cell]
        done = False
        for _ in range(PIECE_PLACE_RETRIES):
            r0 = random.randint(0, grid_h - spec["h"])
            c0 = random.randint(0, grid_w - spec["w"])
            cand = {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r0 + spec["h"] - 1),
                "c1": int(c0 + spec["w"] - 1),
                "h": int(spec["h"]),
                "w": int(spec["w"]),
                "color": int(colors_by_missing[cell]),
            }
            if any(_boxes_conflict(cand, block, margin=1) for block in blockers):
                continue
            blockers.append(cand)
            placed[cell] = cand
            done = True
            break
        if not done:
            return None
    return placed


def _gen_attempt() -> np.ndarray | None:
    want_tie = random.random() < TIE_PROB
    rows = random.choice(MACRO_COUNT_CHOICES)
    cols = random.choice(MACRO_COUNT_CHOICES)
    row_heights = random.sample(SIZE_POOL, rows)
    col_widths = random.sample(SIZE_POOL, cols)

    shape = None
    grey: set[tuple[int, int]] | None = None
    for _ in range(SHAPE_RETRIES):
        shape = _random_connected_shape(rows, cols)
        if shape is None:
            continue
        choice = _choose_grey_cells(shape, rows, cols, row_heights, col_widths, want_tie)
        if choice is None:
            continue
        grey, _tie_used = choice
        break
    if shape is None or grey is None:
        return None

    missing_cells = sorted(shape - grey)
    colors = random.sample(PIECE_COLORS, len(missing_cells))
    colors_by_missing = {cell: color for cell, color in zip(missing_cells, colors, strict=False)}

    template_h = sum(row_heights)
    template_w = sum(col_widths)
    min_h = max(GRID_H_RANGE[0], template_h + 6)
    min_w = max(GRID_W_RANGE[0], template_w + 6)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None
    grid_h = random.randint(min_h, GRID_H_RANGE[1])
    grid_w = random.randint(min_w, GRID_W_RANGE[1])
    top = random.randint(1, grid_h - template_h - 1)
    left = random.randint(1, grid_w - template_w - 1)

    rects = _cell_rects(shape, row_heights, col_widths, top, left)
    template_box = {
        "r0": int(top),
        "c0": int(left),
        "r1": int(top + template_h - 1),
        "c1": int(left + template_w - 1),
    }

    fragment_positions = _place_fragments(
        grid_h=grid_h,
        grid_w=grid_w,
        template_box=template_box,
        missing_cells=missing_cells,
        rects=rects,
        colors_by_missing=colors_by_missing,
    )
    if fragment_positions is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for cell in grey:
        _fill_rect(grid, {**rects[cell], "color": GREY}, GREY)
    for cell, rect in fragment_positions.items():
        _fill_rect(grid, rect, colors_by_missing[cell])

    expected = _render_expected(
        grid_shape=grid.shape,
        shape=shape,
        grey=grey,
        rects=rects,
        row_heights=row_heights,
        col_widths=col_widths,
        colors_by_missing=colors_by_missing,
    )
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
