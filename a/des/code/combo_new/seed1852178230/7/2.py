from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
PAIR_COUNT_RANGE = (2, 4)
GRID_MAX = 30
GRID_MIN_BASE = 16
H_RECT_H_RANGE = (2, 6)
H_RECT_W_RANGE = (3, 8)
V_RECT_H_RANGE = (3, 8)
V_RECT_W_RANGE = (2, 6)
GAP_RANGE = (1, 5)
PADDING = 1
GENERATION_RETRIES = 400


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
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    box = _bbox(coords)
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, box


def _row_profile(mask: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    counts: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    for r in range(mask.shape[0]):
        cols = np.flatnonzero(mask[r])
        assert cols.size > 0, "every row in a horizontal fragment must contain pixels"
        start = int(cols[0])
        end = int(cols[-1])
        count = int(cols.size)
        assert end - start + 1 == count, "fragment rows must be contiguous"
        counts.append(count)
        starts.append(start)
        ends.append(end)
    return counts, starts, ends


def _col_profile(mask: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    counts: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    for c in range(mask.shape[1]):
        rows = np.flatnonzero(mask[:, c])
        assert rows.size > 0, "every column in a vertical fragment must contain pixels"
        start = int(rows[0])
        end = int(rows[-1])
        count = int(rows.size)
        assert end - start + 1 == count, "fragment columns must be contiguous"
        counts.append(count)
        starts.append(start)
        ends.append(end)
    return counts, starts, ends


def _find_legend(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    candidates: list[tuple[dict[int, int], list[tuple[int, int]]]] = []
    for coords in _components(grid != BG):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if c1 - c0 + 1 != 2:
            continue
        if len(coords) != sub.size:
            continue

        mapping: dict[int, int] = {}
        ok = True
        for r in range(sub.shape[0]):
            left = int(sub[r, 0])
            right = int(sub[r, 1])
            if left == BG or right == BG or left == right:
                ok = False
                break
            if left in mapping and mapping[left] != right:
                ok = False
                break
            mapping[left] = right
        if ok and len(mapping) == sub.shape[0] and len(set(mapping.values())) == len(mapping):
            candidates.append((mapping, coords))

    assert len(candidates) == 1, "expected a unique 2-column legend component"
    mapping, coords = candidates[0]
    mask = np.zeros_like(grid, dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mapping, mask


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each same-colored fragment pair together into a solid rectangle, recolor it via the 2-column legend, and erase the legend.

    Input structure:
    - The background is black.
    - Every non-background fragment color appears in exactly two disconnected pieces.
    - Each same-colored pair shares either one full row span or one full column span, and there is a separate legend component made of stacked 1x2 horizontal bars whose left cell is a fragment color and right cell is that pair's output color.

    Rule:
    - Identify the legend and read the color mapping from its rows, then remove the legend.
    - For each fragment-color pair, determine whether the pieces align horizontally or vertically.
    - Slide a horizontal pair together along the shared rows, or a vertical pair together along the shared columns, until the complementary pieces close their gap and form a solid rectangle anchored at the left/top fragment's outer edge.
    - Recolor the completed rectangle with the legend's mapped output color.

    Key insight:
    - The same-color pieces encode complementary partial row-runs or column-runs of one hidden rectangle, while the separate 2-column legend determines the final recoloring independently of the geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= GRID_MAX, "ARC grids must fit within 30x30"

    mapping, legend_mask = _find_legend(g)
    work = g.copy()
    work[legend_mask] = BG

    fragment_colors = [int(color) for color in np.unique(work) if int(color) != BG]
    assert fragment_colors, "expected at least one fragment pair"
    assert set(fragment_colors) == set(mapping), "legend rows must match fragment colors"

    out = np.full_like(g, BG)
    for color in fragment_colors:
        comps = _components(work == color)
        assert len(comps) == 2, "each fragment color must produce exactly two pieces"
        comp_a, comp_b = comps
        mask_a, box_a = _component_mask(comp_a)
        mask_b, box_b = _component_mask(comp_b)
        ar0, ac0, ar1, ac1 = box_a
        br0, bc0, br1, bc1 = box_b

        same_rows = ar0 == br0 and ar1 == br1 and (ac1 < bc0 or bc1 < ac0)
        same_cols = ac0 == bc0 and ac1 == bc1 and (ar1 < br0 or br1 < ar0)
        assert same_rows ^ same_cols, "each pair must align on exactly one axis"

        out_color = mapping[color]
        if same_rows:
            if ac0 <= bc0:
                left_mask, left_box = mask_a, box_a
                right_mask, right_box = mask_b, box_b
            else:
                left_mask, left_box = mask_b, box_b
                right_mask, right_box = mask_a, box_a

            counts_left, starts_left, _ends_left = _row_profile(left_mask)
            counts_right, _starts_right, ends_right = _row_profile(right_mask)
            assert all(start == 0 for start in starts_left), "left fragment must be flush to the hidden rectangle's left edge"
            assert all(end == right_mask.shape[1] - 1 for end in ends_right), "right fragment must be flush to the hidden rectangle's right edge"

            widths = [a + b for a, b in zip(counts_left, counts_right)]
            assert len(set(widths)) == 1, "paired horizontal row runs must sum to one rectangle width"
            full_w = widths[0]
            r0, c0, r1, _c1 = left_box
            out[r0 : r1 + 1, c0 : c0 + full_w] = out_color
        else:
            if ar0 <= br0:
                top_mask, top_box = mask_a, box_a
                bottom_mask, bottom_box = mask_b, box_b
            else:
                top_mask, top_box = mask_b, box_b
                bottom_mask, bottom_box = mask_a, box_a

            counts_top, starts_top, _ends_top = _col_profile(top_mask)
            counts_bottom, _starts_bottom, ends_bottom = _col_profile(bottom_mask)
            assert all(start == 0 for start in starts_top), "top fragment must be flush to the hidden rectangle's top edge"
            assert all(end == bottom_mask.shape[0] - 1 for end in ends_bottom), "bottom fragment must be flush to the hidden rectangle's bottom edge"

            heights = [a + b for a, b in zip(counts_top, counts_bottom)]
            assert len(set(heights)) == 1, "paired vertical column runs must sum to one rectangle height"
            full_h = heights[0]
            r0, c0, _r1, c1 = top_box
            out[r0 : r0 + full_h, c0 : c1 + 1] = out_color

    return out


def _random_profile(length: int, extent: int) -> list[int]:
    assert length >= 1 and extent >= 2
    values = [random.randint(1, extent - 1)]
    for _ in range(length - 1):
        prev = values[-1]
        choices = [prev]
        if prev > 1:
            choices.append(prev - 1)
        if prev < extent - 1:
            choices.append(prev + 1)
        values.append(random.choice(choices))
    if length > 1 and extent > 2 and len(set(values)) == 1:
        i = random.randrange(length)
        delta = random.choice((-1, 1))
        values[i] = min(max(values[i] + delta, 1), extent - 1)
    return values


def _reserve(occupied: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    r0 = max(0, r - PADDING)
    c0 = max(0, c - PADDING)
    r1 = min(occupied.shape[0], r + h + PADDING)
    c1 = min(occupied.shape[1], c + w + PADDING)
    occupied[r0:r1, c0:c1] = True


def _can_place(occupied: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    r0 = max(0, r - PADDING)
    c0 = max(0, c - PADDING)
    r1 = min(occupied.shape[0], r + h + PADDING)
    c1 = min(occupied.shape[1], c + w + PADDING)
    return not bool(np.any(occupied[r0:r1, c0:c1]))


def _place_boxes(grid_shape: tuple[int, int], boxes: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    occupied = np.zeros(grid_shape, dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(boxes)
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0] * boxes[i][1], reverse=True)

    for idx in order:
        h, w = boxes[idx]
        positions = [(r, c) for r in range(grid_shape[0] - h + 1) for c in range(grid_shape[1] - w + 1)]
        random.shuffle(positions)
        for r, c in positions:
            if _can_place(occupied, r, c, h, w):
                placements[idx] = (r, c)
                _reserve(occupied, r, c, h, w)
                break
        if placements[idx] is None:
            return None
    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _sample_pair_spec(piece_color: int, out_color: int) -> dict:
    orientation = random.choice(("horizontal", "vertical"))
    gap = random.randint(*GAP_RANGE)
    if orientation == "horizontal":
        rect_h = random.randint(*H_RECT_H_RANGE)
        rect_w = random.randint(*H_RECT_W_RANGE)
        split = _random_profile(rect_h, rect_w)
        footprint = (rect_h, rect_w + gap)
    else:
        rect_h = random.randint(*V_RECT_H_RANGE)
        rect_w = random.randint(*V_RECT_W_RANGE)
        split = _random_profile(rect_w, rect_h)
        footprint = (rect_h + gap, rect_w)
    return {
        "orientation": orientation,
        "piece_color": piece_color,
        "out_color": out_color,
        "rect_h": rect_h,
        "rect_w": rect_w,
        "gap": gap,
        "split": split,
        "footprint": footprint,
    }


def _draw_pair(inp: np.ndarray, out: np.ndarray, spec: dict, top: int, left: int) -> None:
    color = spec["piece_color"]
    out_color = spec["out_color"]
    rect_h = spec["rect_h"]
    rect_w = spec["rect_w"]
    gap = spec["gap"]
    split = spec["split"]

    if spec["orientation"] == "horizontal":
        for dr, cut in enumerate(split):
            inp[top + dr, left : left + cut] = color
            inp[top + dr, left + gap + cut : left + gap + rect_w] = color
        out[top : top + rect_h, left : left + rect_w] = out_color
    else:
        for dc, cut in enumerate(split):
            inp[top : top + cut, left + dc] = color
            inp[top + gap + cut : top + gap + rect_h, left + dc] = color
        out[top : top + rect_h, left : left + rect_w] = out_color


def _legend_patch(rows: list[tuple[int, int]]) -> np.ndarray:
    patch = np.zeros((len(rows), 2), dtype=int)
    for r, (left_color, right_color) in enumerate(rows):
        patch[r, 0] = int(left_color)
        patch[r, 1] = int(right_color)
    return patch


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(PALETTE, 2 * pair_count)
    piece_colors = colors[:pair_count]
    out_colors = colors[pair_count:]
    specs = [_sample_pair_spec(piece_color, out_color) for piece_color, out_color in zip(piece_colors, out_colors)]

    min_dim = min(GRID_MAX, GRID_MIN_BASE + 2 * pair_count)
    grid_h = random.randint(min_dim, GRID_MAX)
    grid_w = random.randint(min_dim, GRID_MAX)

    boxes = [tuple(spec["footprint"]) for spec in specs]
    boxes.append((pair_count, 2))
    placements = _place_boxes((grid_h, grid_w), boxes)
    if placements is None or len(placements) != len(boxes):
        return None

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    out = np.full((grid_h, grid_w), BG, dtype=int)

    for spec, (top, left) in zip(specs, placements[:-1]):
        _draw_pair(inp, out, spec, top, left)

    legend_rows = list(zip(piece_colors, out_colors))
    random.shuffle(legend_rows)
    legend = _legend_patch(legend_rows)
    legend_top, legend_left = placements[-1]
    inp[legend_top : legend_top + legend.shape[0], legend_left : legend_left + legend.shape[1]] = legend

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
