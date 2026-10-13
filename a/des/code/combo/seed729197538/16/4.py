from __future__ import annotations

import random

import numpy as np


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _full_line_indices(grid: np.ndarray, color: int) -> tuple[list[int], list[int]]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == int(color))]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == int(color))]
    return rows, cols


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    seps = sorted(set(map(int, separators)))
    assert seps, "expected at least one separator line"
    assert seps[0] > 0 and seps[-1] < length - 1, "separator lines must be internal"

    spans: list[tuple[int, int]] = []
    start = 0
    for sep in seps:
        assert start < sep, "separator lines must leave nonempty regions"
        spans.append((int(start), int(sep - 1)))
        start = sep + 1
    assert start < length, "separator lines must leave a final nonempty region"
    spans.append((int(start), int(length - 1)))
    return spans


def _rotate_offset(dr: int, dc: int, turns_cw: int) -> tuple[int, int]:
    rr, cc = int(dr), int(dc)
    for _ in range(int(turns_cw) % 4):
        rr, cc = cc, -rr
    return int(rr), int(cc)


def _parse_region(region: np.ndarray, bg: int, marker: int) -> dict:
    h, w = map(int, region.shape)
    assert bool(np.any(region == int(bg))), "region must contain background"

    marker_count = int(np.sum(region == int(marker)))
    assert 1 <= marker_count <= 3, "each region must contain 1 to 3 markers"

    shape_mask = (region != int(bg)) & (region != int(marker))
    shape_area = int(shape_mask.sum())
    assert shape_area >= 3, "shape must be large enough to identify the anchor"

    comps = _components(shape_mask)
    assert len(comps) == 1, "expected exactly one 4-connected shape"

    colors, counts = np.unique(region[shape_mask], return_counts=True)
    palette = {int(color): int(count) for color, count in zip(colors, counts)}
    assert len(palette) == 2, "shape must have one body color and one anchor color"

    anchor_colors = [color for color, count in palette.items() if count == 1]
    body_colors = [color for color, count in palette.items() if count >= 2]
    assert len(anchor_colors) == 1 and len(body_colors) == 1, "anchor must be the unique singleton color"

    anchor_color = int(anchor_colors[0])
    anchor_positions = np.argwhere(region == anchor_color)
    assert len(anchor_positions) == 1, "anchor color must appear exactly once in the region"
    anchor = tuple(map(int, anchor_positions[0]))
    assert shape_mask[anchor], "anchor must lie inside the shape"

    pixels = [
        (int(r), int(c), int(region[r, c]))
        for r, c in np.argwhere(shape_mask)
    ]

    return {
        "shape": (h, w),
        "marker_count": marker_count,
        "anchor": anchor,
        "anchor_color": anchor_color,
        "body_color": int(body_colors[0]),
        "pixels": pixels,
    }


def _render_rotated_region(shape: tuple[int, int], bg: int, info: dict) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), int(bg), dtype=int)
    ar, ac = map(int, info["anchor"])
    turns_cw = int(info["marker_count"]) % 4
    for r, c, color in info["pixels"]:
        dr, dc = int(r - ar), int(c - ac)
        rr, cc = _rotate_offset(dr, dc, turns_cw)
        nr, nc = int(ar + rr), int(ac + cc)
        if 0 <= nr < h and 0 <= nc < w:
            out[nr, nc] = int(color)
    return out


def _candidate_configs(grid: np.ndarray) -> list[dict]:
    colors = list(map(int, np.unique(grid)))
    candidates: list[dict] = []

    for separator in colors:
        sep_rows, sep_cols = _full_line_indices(grid, separator)
        if not sep_rows or not sep_cols:
            continue
        try:
            row_spans = _spans(grid.shape[0], sep_rows)
            col_spans = _spans(grid.shape[1], sep_cols)
        except AssertionError:
            continue

        regions = []
        for r0, r1 in row_spans:
            for c0, c1 in col_spans:
                regions.append(((r0, r1, c0, c1), grid[r0 : r1 + 1, c0 : c1 + 1]))

        non_sep_colors = [color for color in colors if color != int(separator)]
        for bg in non_sep_colors:
            for marker in non_sep_colors:
                if marker == bg:
                    continue
                infos = []
                ok = True
                for bbox, region in regions:
                    try:
                        info = _parse_region(region, bg, marker)
                    except AssertionError:
                        ok = False
                        break
                    infos.append((bbox, info))
                if ok:
                    candidates.append(
                        {
                            "separator": int(separator),
                            "bg": int(bg),
                            "marker": int(marker),
                            "sep_rows": sep_rows,
                            "sep_cols": sep_cols,
                            "regions": infos,
                        }
                    )
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    In each separator-defined subregion, rotate the lone two-color shape clockwise
    by 90/180/270 degrees around its unique anchor pixel according to the number
    of marker pixels in that region, erase the markers, and clip the rotated
    result to the region bounds.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    candidates = _candidate_configs(g)
    assert len(candidates) == 1, "expected a unique interpretation of separators, background, and markers"
    cfg = candidates[0]

    out = np.full(g.shape, int(cfg["bg"]), dtype=int)
    out[cfg["sep_rows"], :] = int(cfg["separator"])
    out[:, cfg["sep_cols"]] = int(cfg["separator"])

    for (r0, r1, c0, c1), info in cfg["regions"]:
        out[r0 : r1 + 1, c0 : c1 + 1] = _render_rotated_region(
            (int(r1 - r0 + 1), int(c1 - c0 + 1)),
            int(cfg["bg"]),
            info,
        )

    return out


# --------------- generator (o5x, att2) ---------------

COLOR_POOL = tuple(range(10))
GRID_REGION_ROWS_RANGE = (2, 4)
GRID_REGION_COLS_RANGE = (2, 4)
REGION_SIDE_RANGE = (5, 9)
SEPARATOR_WIDTH = 1
MAX_SHAPE_AREA = 9
MIN_SHAPE_AREA = 3
CLIP_PROBABILITY = 0.45
MARKER_COUNTS = (1, 2, 3)
GENERATION_ATTEMPTS = 400


def _sample_spans(count: int) -> list[int]:
    return [random.randint(*REGION_SIDE_RANGE) for _ in range(int(count))]


def _separator_indices(lengths: list[int]) -> list[int]:
    idxs = []
    cursor = 0
    for length in lengths[:-1]:
        cursor += int(length)
        idxs.append(int(cursor))
        cursor += SEPARATOR_WIDTH
    return idxs


def _dist_to_edge(r: int, c: int, h: int, w: int) -> int:
    return min(int(r), int(c), int(h - 1 - r), int(w - 1 - c))


def _sample_anchor(h: int, w: int, want_clip: bool) -> tuple[int, int]:
    cells = [(r, c) for r in range(h) for c in range(w)]
    if want_clip:
        edgeish = [(r, c) for r, c in cells if _dist_to_edge(r, c, h, w) <= 1]
        if edgeish:
            return random.choice(edgeish)
    interior = [(r, c) for r, c in cells if _dist_to_edge(r, c, h, w) >= 1]
    if interior:
        return random.choice(interior)
    return random.choice(cells)


def _grow_shape(h: int, w: int, anchor: tuple[int, int], area: int) -> set[tuple[int, int]] | None:
    cells = {tuple(map(int, anchor))}
    frontier = set(_neighbors4(anchor[0], anchor[1], h, w))
    while len(cells) < int(area):
        frontier = {cell for cell in frontier if cell not in cells}
        if not frontier:
            return None
        pick = random.choice(tuple(frontier))
        cells.add(tuple(map(int, pick)))
        frontier.update(_neighbors4(pick[0], pick[1], h, w))
    return cells


def _choose_marker_cells(
    h: int,
    w: int,
    shape_cells: set[tuple[int, int]],
    marker_count: int,
) -> list[tuple[int, int]] | None:
    candidates = []
    for r in range(h):
        for c in range(w):
            if (r, c) in shape_cells:
                continue
            if any((nr, nc) in shape_cells for nr, nc in _neighbors4(r, c, h, w)):
                continue
            candidates.append((r, c))

    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for cell in candidates:
        if any(abs(cell[0] - rr) + abs(cell[1] - cc) == 1 for rr, cc in chosen):
            continue
        chosen.append(cell)
        if len(chosen) == int(marker_count):
            return chosen
    return None


def _rotated_cells(
    shape_cells: set[tuple[int, int]],
    anchor: tuple[int, int],
    turns_cw: int,
) -> list[tuple[int, int]]:
    ar, ac = map(int, anchor)
    coords = []
    for r, c in shape_cells:
        rr, cc = _rotate_offset(int(r - ar), int(c - ac), int(turns_cw))
        coords.append((int(ar + rr), int(ac + cc)))
    return coords


def _visible_rotated_cells(
    shape_cells: set[tuple[int, int]],
    anchor: tuple[int, int],
    turns_cw: int,
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r, c in _rotated_cells(shape_cells, anchor, turns_cw)
        if 0 <= r < h and 0 <= c < w
    }


def _shape_matches_rotation(
    shape_cells: set[tuple[int, int]],
    anchor: tuple[int, int],
    turns_cw: int,
    h: int,
    w: int,
) -> bool:
    return _visible_rotated_cells(shape_cells, anchor, turns_cw, h, w) == set(shape_cells)


def _make_region(
    h: int,
    w: int,
    bg: int,
    marker: int,
    body: int,
    anchor_color: int,
    marker_count: int,
    want_clip: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    max_area = min(MAX_SHAPE_AREA, h * w - marker_count - 3)
    if max_area < MIN_SHAPE_AREA:
        return None

    for _ in range(120):
        anchor = _sample_anchor(h, w, want_clip)
        area = random.randint(MIN_SHAPE_AREA, max_area)
        shape_cells = _grow_shape(h, w, anchor, area)
        if shape_cells is None:
            continue

        rotated = _rotated_cells(shape_cells, anchor, marker_count)
        clipped = any(not (0 <= r < h and 0 <= c < w) for r, c in rotated)
        if clipped != bool(want_clip):
            continue

        visible_rotated = {
            (r, c)
            for r, c in rotated
            if 0 <= r < h and 0 <= c < w
        }
        if len(visible_rotated) < MIN_SHAPE_AREA:
            continue
        if _shape_matches_rotation(shape_cells, anchor, marker_count, h, w):
            continue

        marker_cells = _choose_marker_cells(h, w, shape_cells, marker_count)
        if marker_cells is None:
            continue

        region_in = np.full((h, w), int(bg), dtype=int)
        region_out = np.full((h, w), int(bg), dtype=int)

        for r, c in shape_cells:
            region_in[r, c] = int(anchor_color) if (r, c) == anchor else int(body)
        for r, c in marker_cells:
            region_in[r, c] = int(marker)
        for r, c in visible_rotated:
            region_out[r, c] = int(anchor_color) if (r, c) == anchor else int(body)
        return region_in, region_out

    return None


def _marker_count_schedule(region_count: int) -> list[int]:
    counts = [MARKER_COUNTS[i % len(MARKER_COUNTS)] for i in range(int(region_count))]
    random.shuffle(counts)
    return counts


def _clip_schedule(region_count: int) -> list[bool]:
    flags = [random.random() < CLIP_PROBABILITY for _ in range(int(region_count))]
    if region_count >= 2 and all(flags):
        flags[random.randrange(region_count)] = False
    if region_count >= 2 and not any(flags):
        flags[random.randrange(region_count)] = True
    return flags


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    region_rows = random.randint(*GRID_REGION_ROWS_RANGE)
    region_cols = random.randint(*GRID_REGION_COLS_RANGE)
    heights = _sample_spans(region_rows)
    widths = _sample_spans(region_cols)

    bg, separator, marker = random.sample(COLOR_POOL, 3)
    payload_colors = [color for color in COLOR_POOL if color not in {bg, separator, marker}]

    total_h = sum(heights) + (region_rows - 1) * SEPARATOR_WIDTH
    total_w = sum(widths) + (region_cols - 1) * SEPARATOR_WIDTH
    inp = np.full((total_h, total_w), int(bg), dtype=int)
    out = inp.copy()

    sep_rows = _separator_indices(heights)
    sep_cols = _separator_indices(widths)
    inp[sep_rows, :] = int(separator)
    inp[:, sep_cols] = int(separator)
    out[sep_rows, :] = int(separator)
    out[:, sep_cols] = int(separator)

    row_starts = [0]
    for h in heights[:-1]:
        row_starts.append(row_starts[-1] + int(h) + SEPARATOR_WIDTH)
    col_starts = [0]
    for w in widths[:-1]:
        col_starts.append(col_starts[-1] + int(w) + SEPARATOR_WIDTH)

    region_count = region_rows * region_cols
    marker_counts = _marker_count_schedule(region_count)
    clip_flags = _clip_schedule(region_count)

    idx = 0
    for r0, h in zip(row_starts, heights):
        for c0, w in zip(col_starts, widths):
            body, anchor_color = random.sample(payload_colors, 2)
            built = _make_region(
                int(h),
                int(w),
                int(bg),
                int(marker),
                int(body),
                int(anchor_color),
                int(marker_counts[idx]),
                bool(clip_flags[idx]),
            )
            if built is None:
                return None
            region_in, region_out = built
            inp[r0 : r0 + h, c0 : c0 + w] = region_in
            out[r0 : r0 + h, c0 : c0 + w] = region_out
            idx += 1

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        built = _build_attempt()
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
