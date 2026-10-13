from __future__ import annotations

from collections import Counter, defaultdict, deque
import random
from typing import Iterable

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(box: BBox) -> int:
    return int(box[2] - box[0] + 1)


def bbox_w(box: BBox) -> int:
    return int(box[3] - box[1] + 1)


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _neighbor_deltas(connectivity: int) -> tuple[Coord, ...]:
    if int(connectivity) == 8:
        return (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        )
    return ((-1, 0), (1, 0), (0, -1), (0, 1))


def components_mask(mask: np.ndarray, *, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    deltas = _neighbor_deltas(connectivity)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int,
    colors: Iterable[int] | None = None,
    min_area: int = 1,
) -> list[dict]:
    if colors is None:
        color_values = sorted(set(map(int, grid.ravel())) - {int(bg)})
    else:
        color_values = sorted({int(c) for c in colors if int(c) != int(bg)})

    out: list[dict] = []
    for color in color_values:
        for coords in components_mask(grid == color, connectivity=connectivity):
            area = int(coords.shape[0])
            if area < int(min_area):
                continue
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": area,
                }
            )
    return out


def _row_span_interior(coords: np.ndarray) -> tuple[Coord, ...]:
    pixel_set = {tuple(map(int, rc)) for rc in coords.tolist()}
    rows: dict[int, list[int]] = defaultdict(list)
    for r, c in pixel_set:
        rows[r].append(c)

    cells: list[Coord] = []
    for r, cols in rows.items():
        cols = sorted(cols)
        if len(cols) < 2:
            continue
        left = cols[0]
        right = cols[-1]
        for c in range(left + 1, right):
            if (r, c) not in pixel_set:
                cells.append((r, c))
    return tuple(cells)


def _overlap_groups(records: list[tuple[int, int, int]]) -> tuple[dict[int, int], int]:
    ordered = sorted(records, key=lambda item: (item[1], item[2], item[0]))
    mapping: dict[int, int] = {}
    group = -1
    current_end = -1
    for idx, start, end in ordered:
        if group == -1 or start > current_end:
            group += 1
            current_end = end
        else:
            current_end = max(current_end, end)
        mapping[idx] = group
    return mapping, group + 1


def _valid_rectangles(grid: np.ndarray, bg: int) -> list[dict]:
    rectangles: list[dict] = []
    for comp in find_components(grid, bg=bg, connectivity=4, min_area=9):
        box = comp["bbox"]
        h = bbox_h(box)
        w = bbox_w(box)
        if h < 3 or w < 3:
            continue
        if int(comp["area"]) != h * w:
            continue
        rectangles.append(comp)
    return rectangles


def _outline_objects(grid: np.ndarray, bg: int) -> list[dict]:
    outlines: list[dict] = []
    for comp in find_components(grid, bg=bg, connectivity=8, min_area=3):
        interior = _row_span_interior(comp["coords"])
        if interior:
            outlines.append(
                {
                    "color": int(comp["color"]),
                    "coords": comp["coords"],
                    "bbox": comp["bbox"],
                    "interior": interior,
                    "area": len(interior),
                }
            )
    return outlines


def _compress_color_grid(rectangles: list[dict]) -> np.ndarray:
    row_by_rect, row_count = _overlap_groups(
        [(idx, rect["bbox"][0], rect["bbox"][2]) for idx, rect in enumerate(rectangles)]
    )
    col_by_rect, col_count = _overlap_groups(
        [(idx, rect["bbox"][1], rect["bbox"][3]) for idx, rect in enumerate(rectangles)]
    )
    out = np.full((row_count, col_count), -1, dtype=int)
    for idx, rect in enumerate(rectangles):
        rr = row_by_rect[idx]
        cc = col_by_rect[idx]
        assert out[rr, cc] == -1, "expected one rectangle per compact-grid slot"
        out[rr, cc] = int(rect["color"])
    assert np.all(out >= 0), "expected a complete rectangle lattice"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill outline interiors whose row-span area matches a color count from the compressed rectangle lattice.

    Input structure:
    - The grid has a dominant background plus two separated regions.
    - One region contains several monochrome filled rectangles at least 3x3 arranged on a row/column lattice, mixed with smaller clutter components.
    - The other region contains monochrome 8-connected outline objects with distinct row-span interior areas.

    Rule:
    - Keep only the 4-connected monochrome components that are solid rectangles of size at least 3x3, and erase the other foreground clutter.
    - Group the kept rectangles by overlapping bbox row intervals and overlapping bbox column intervals to compress them into a compact color grid.
    - Count how many compact-grid cells each color occupies.
    - For every outline object, compute its row-span interior cells: on each occupied row, all non-outline cells strictly between the leftmost and rightmost outline pixels.
    - If exactly one color count equals exactly one outline interior area, fill that outline interior with that color; otherwise leave it unchanged.

    Key insight:
    - The fill color is not encoded by the outline itself; it is encoded indirectly by how often that color appears after collapsing the remote rectangle lattice into a compact grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = get_bg(g)
    rectangles = _valid_rectangles(g, bg)
    outlines = _outline_objects(g, bg)
    assert rectangles, "expected filled rectangles"
    assert outlines, "expected outline objects"

    keep = np.zeros_like(g, dtype=bool)
    for rect in rectangles:
        rr, cc = rect["coords"].T
        keep[rr, cc] = True
    for outline in outlines:
        rr, cc = outline["coords"].T
        keep[rr, cc] = True

    out = g.copy()
    out[(g != bg) & ~keep] = bg

    compact = _compress_color_grid(rectangles)
    color_counts = Counter(map(int, compact.ravel()))

    count_to_colors: dict[int, list[int]] = defaultdict(list)
    for color, count in color_counts.items():
        count_to_colors[int(count)].append(int(color))

    area_to_outlines: dict[int, list[int]] = defaultdict(list)
    for idx, outline in enumerate(outlines):
        area_to_outlines[int(outline["area"])].append(idx)

    for value, colors in count_to_colors.items():
        targets = area_to_outlines.get(int(value), [])
        if len(colors) != 1 or len(targets) != 1:
            continue
        fill_color = int(colors[0])
        for r, c in outlines[targets[0]]["interior"]:
            out[r, c] = fill_color

    return out


RECT_COLOR_POOL = [1, 2, 3, 4, 7, 8, 9]
OUTLINE_COLOR_POOL = [5, 6]
CLUTTER_SHAPES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
]
COUNT_OPTIONS_BY_ROWS = {
    2: [(1, 2, 3)],
    3: [(1, 2, 6), (1, 3, 5), (2, 3, 4)],
}
MAX_ATTEMPTS = 500


def _sample_compact_grid(rng: random.Random) -> tuple[np.ndarray, dict[int, int]]:
    rows = rng.choice([2, 3])
    cols = 3
    counts = list(rng.choice(COUNT_OPTIONS_BY_ROWS[rows]))
    colors = rng.sample(RECT_COLOR_POOL, 3)
    color_to_count = {int(color): int(count) for color, count in zip(colors, counts, strict=True)}

    cells: list[int] = []
    for color, count in color_to_count.items():
        cells.extend([int(color)] * int(count))
    rng.shuffle(cells)
    return np.array(cells, dtype=int).reshape(rows, cols), color_to_count


def _band_extents(rng: random.Random, count: int) -> list[tuple[int, int]]:
    pos = rng.randint(0, 1)
    extents: list[tuple[int, int]] = []
    for idx in range(count):
        if idx > 0:
            pos += rng.randint(1, 2)
        length = rng.randint(3, 5)
        extents.append((pos, pos + length - 1))
        pos += length
    return extents


def _intervals_in_extent(rng: random.Random, extent: tuple[int, int], count: int) -> list[tuple[int, int]]:
    start, end = extent
    common = rng.randint(start, end)
    out: list[tuple[int, int]] = []
    for _ in range(count):
        max_len = end - start + 1
        length = rng.randint(3, max_len)
        lo_min = max(start, common - length + 1)
        lo_max = min(common, end - length + 1)
        lo = rng.randint(lo_min, lo_max)
        out.append((lo, lo + length - 1))
    return out


def _render_rectangle_region(rng: random.Random, compact: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = compact.shape
    row_extents = _band_extents(rng, rows)
    col_extents = _band_extents(rng, cols)
    row_intervals = [_intervals_in_extent(rng, extent, cols) for extent in row_extents]
    col_intervals = [_intervals_in_extent(rng, extent, rows) for extent in col_extents]

    top = rng.randint(1, 2)
    left = rng.randint(1, 2)
    bottom = rng.randint(1, 3)
    right = rng.randint(1, 3)
    h = top + row_extents[-1][1] + 1 + bottom
    w = left + col_extents[-1][1] + 1 + right

    clean = np.zeros((h, w), dtype=int)
    for rr in range(rows):
        for cc in range(cols):
            r0, r1 = row_intervals[rr][cc]
            c0, c1 = col_intervals[cc][rr]
            clean[top + r0 : top + r1 + 1, left + c0 : left + c1 + 1] = int(compact[rr, cc])

    noisy = clean.copy()
    clutter_goal = rng.randint(5, 10)
    clutter_colors = list({int(v) for v in compact.ravel().tolist()})
    attempts = 0
    placed = 0
    while placed < clutter_goal and attempts < 200:
        attempts += 1
        shape = rng.choice(CLUTTER_SHAPES)
        sh, sw = shape.shape
        r0 = rng.randint(0, h - sh)
        c0 = rng.randint(0, w - sw)
        if not _can_place_clutter(noisy, shape, r0, c0):
            continue
        color = int(rng.choice(clutter_colors))
        patch = np.where(shape, color, 0)
        paste(noisy, patch, r0, c0, transparent=0)
        placed += 1
    return noisy, clean


def _can_place_clutter(grid: np.ndarray, shape: np.ndarray, r0: int, c0: int) -> bool:
    h, w = grid.shape
    sh, sw = shape.shape
    coords = np.argwhere(shape)
    occupied = {(r0 + int(dr), c0 + int(dc)) for dr, dc in coords}
    for rr, cc in occupied:
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if grid[rr, cc] != 0:
            return False
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in occupied and grid[nr, nc] != 0:
                return False
    return True


def _factor_shapes(area: int) -> list[tuple[int, int]]:
    options: set[tuple[int, int]] = set()
    for h in range(1, area + 1):
        if area % h != 0:
            continue
        w = area // h
        if max(h, w) <= 8:
            options.add((h, w))
            options.add((w, h))
    return sorted(options)


def _make_frame(
    rng: random.Random,
    area: int,
    border_color: int,
    fill_color: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    interior_h, interior_w = rng.choice(_factor_shapes(area))
    patch = np.zeros((interior_h + 2, interior_w + 2), dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    out = patch.copy()
    if fill_color is not None:
        out[1:-1, 1:-1] = int(fill_color)
    return patch, out


def _render_outline_region(
    rng: random.Random,
    color_to_count: dict[int, int],
    *,
    stack_axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    matched_items = list(color_to_count.items())
    rng.shuffle(matched_items)

    used_areas = {int(count) for count in color_to_count.values()}
    area_pool = [1, 2, 3, 4, 5, 6, 7, 8]
    extra_area = None
    if rng.random() < 0.65:
        extras = [area for area in area_pool if area not in used_areas]
        if extras:
            extra_area = rng.choice(extras)

    frames: list[tuple[np.ndarray, np.ndarray]] = []
    for color, area in matched_items:
        border = rng.choice(OUTLINE_COLOR_POOL)
        frames.append(_make_frame(rng, int(area), int(border), int(color)))
    if extra_area is not None:
        border = rng.choice(OUTLINE_COLOR_POOL)
        frames.append(_make_frame(rng, int(extra_area), int(border), None))

    if stack_axis == "h":
        top = rng.randint(1, 2)
        bottom = rng.randint(1, 2)
        left = rng.randint(1, 2)
        right = rng.randint(1, 2)
        gaps = [rng.randint(2, 4) for _ in range(max(0, len(frames) - 1))]
        h = top + max(frame[0].shape[0] for frame in frames) + bottom
        w = left + sum(frame[0].shape[1] for frame in frames) + sum(gaps) + right
        inp = np.zeros((h, w), dtype=int)
        out = np.zeros((h, w), dtype=int)
        c = left
        for idx, (frame_inp, frame_out) in enumerate(frames):
            jitter = rng.randint(0, h - top - bottom - frame_inp.shape[0])
            r = top + jitter
            paste(inp, frame_inp, r, c)
            paste(out, frame_out, r, c)
            c += frame_inp.shape[1]
            if idx < len(gaps):
                c += gaps[idx]
        return inp, out

    top = rng.randint(1, 2)
    bottom = rng.randint(1, 2)
    left = rng.randint(1, 2)
    right = rng.randint(1, 2)
    gaps = [rng.randint(2, 4) for _ in range(max(0, len(frames) - 1))]
    h = top + sum(frame[0].shape[0] for frame in frames) + sum(gaps) + bottom
    w = left + max(frame[0].shape[1] for frame in frames) + right
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    r = top
    for idx, (frame_inp, frame_out) in enumerate(frames):
        jitter = rng.randint(0, w - left - right - frame_inp.shape[1])
        c = left + jitter
        paste(inp, frame_inp, r, c)
        paste(out, frame_out, r, c)
        r += frame_inp.shape[0]
        if idx < len(gaps):
            r += gaps[idx]
    return inp, out


def _compose_regions(
    rng: random.Random,
    region_a_in: np.ndarray,
    region_a_out: np.ndarray,
    region_b_in: np.ndarray,
    region_b_out: np.ndarray,
    *,
    split: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    if split == "tb":
        top_margin = rng.randint(0, 2)
        bottom_margin = rng.randint(0, 2)
        left_margin = rng.randint(0, 2)
        right_margin = rng.randint(0, 2)
        gap = rng.randint(2, 4)
        h = top_margin + region_a_in.shape[0] + gap + region_b_in.shape[0] + bottom_margin
        w = left_margin + max(region_a_in.shape[1], region_b_in.shape[1]) + right_margin
        if h > 30 or w > 30:
            return None
        inp = np.zeros((h, w), dtype=int)
        out = np.zeros((h, w), dtype=int)
        a_c = left_margin + rng.randint(0, w - left_margin - right_margin - region_a_in.shape[1])
        b_c = left_margin + rng.randint(0, w - left_margin - right_margin - region_b_in.shape[1])
        paste(inp, region_a_in, top_margin, a_c)
        paste(out, region_a_out, top_margin, a_c)
        b_r = top_margin + region_a_in.shape[0] + gap
        paste(inp, region_b_in, b_r, b_c)
        paste(out, region_b_out, b_r, b_c)
        return inp, out

    top_margin = rng.randint(0, 2)
    bottom_margin = rng.randint(0, 2)
    left_margin = rng.randint(0, 2)
    right_margin = rng.randint(0, 2)
    gap = rng.randint(2, 4)
    h = top_margin + max(region_a_in.shape[0], region_b_in.shape[0]) + bottom_margin
    w = left_margin + region_a_in.shape[1] + gap + region_b_in.shape[1] + right_margin
    if h > 30 or w > 30:
        return None
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    a_r = top_margin + rng.randint(0, h - top_margin - bottom_margin - region_a_in.shape[0])
    b_r = top_margin + rng.randint(0, h - top_margin - bottom_margin - region_b_in.shape[0])
    paste(inp, region_a_in, a_r, left_margin)
    paste(out, region_a_out, a_r, left_margin)
    b_c = left_margin + region_a_in.shape[1] + gap
    paste(inp, region_b_in, b_r, b_c)
    paste(out, region_b_out, b_r, b_c)
    return inp, out


def _generate_pair(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    compact, color_to_count = _sample_compact_grid(rng)
    rect_in, rect_out = _render_rectangle_region(rng, compact)

    split = rng.choice(["tb", "lr"])
    stack_axis = "h" if split == "tb" else "v"
    outline_in, outline_out = _render_outline_region(rng, color_to_count, stack_axis=stack_axis)

    if rng.random() < 0.5:
        first = (rect_in, rect_out)
        second = (outline_in, outline_out)
    else:
        first = (outline_in, outline_out)
        second = (rect_in, rect_out)

    pair = _compose_regions(rng, first[0], first[1], second[0], second[1], split=split)
    if pair is None:
        return None
    inp, out = pair
    if inp.shape[0] > 30 or inp.shape[1] > 30 or out.shape[0] > 30 or out.shape[1] > 30:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    seed = random.randrange(1 << 30)
    rng = random.Random(seed)
    for _ in range(MAX_ATTEMPTS):
        pair = _generate_pair(rng)
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid input for puzzle 8")
