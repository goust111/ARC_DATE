from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _nonblack_runs(row: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    c = 0
    w = row.shape[0]
    while c < w:
        if int(row[c]) == BLACK:
            c += 1
            continue
        c0 = c
        color = int(row[c])
        while c < w and int(row[c]) == color:
            c += 1
        runs.append((c0, c, color))
    return runs


def _parse_legend(grid: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
    h, w = grid.shape
    rows = np.flatnonzero(np.any(grid != BLACK, axis=1))
    assert rows.size > 0, "expected a legend row"
    legend_row = int(rows[0])

    runs = _nonblack_runs(grid[legend_row])
    assert runs, "legend row must contain colored runs"

    mapping: dict[int, np.ndarray] = {}
    used = np.zeros((h, w), dtype=bool)
    for c0, c1, color in runs:
        assert color not in mapping, "legend colors must be unique"
        assert legend_row + 2 < h, "legend templates require at least two rows"

        template_h = 2
        if legend_row + 3 < h and np.any(grid[legend_row + 3, c0:c1] != BLACK):
            template_h = 3

        template = grid[legend_row + 1 : legend_row + 1 + template_h, c0:c1] != BLACK
        assert template.shape == (template_h, c1 - c0)
        assert np.any(template), "template must contain at least one on pixel"

        mapping[color] = template
        used[legend_row, c0:c1] = True
        used[legend_row + 1 : legend_row + 1 + template_h, c0:c1] = True

    return mapping, used


def _rectangle_ring_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None
    expected = 2 * height + 2 * width - 4
    if len(coords) != expected:
        return None
    coord_set = set(coords)
    for r, c in coords:
        if r not in (r0, r1) and c not in (c0, c1):
            return None
    for c in range(c0, c1 + 1):
        if (r0, c) not in coord_set or (r1, c) not in coord_set:
            return None
    for r in range(r0, r1 + 1):
        if (r, c0) not in coord_set or (r, c1) not in coord_set:
            return None
    return r0, c0, r1, c1


def _tile_template(template: np.ndarray, height: int, width: int, color: int) -> np.ndarray:
    rr, cc = np.indices((height, width))
    tiled_mask = template[rr % template.shape[0], cc % template.shape[1]]
    return np.where(tiled_mask, color, BLACK).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow rectangle by tiling the legend template keyed by its interior marker color.

    Input structure:
    - The background is black.
    - Near the top sits a horizontal legend row made of separated colored runs; directly below each run is a same-width binary template made only of grey and black.
    - Lower in the grid are several 1-pixel-thick closed rectangular borders whose interiors are black except for one colored marker pixel.

    Rule:
    - Read the top legend as a mapping from each run color to the binary template beneath it.
    - For every closed rectangle, inspect its lone interior marker pixel and select the template with the same color.
    - Repaint the whole rectangle interior from its top-left corner by periodically tiling that template: template on-cells become the marker color and off-cells stay black.

    Key insight:
    - The interior marker is not the fill color by itself; it is a lookup key that selects which binary tile pattern should be repeated inside the matching rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    templates, used = _parse_legend(g)
    out = g.copy()

    colors = [int(v) for v in np.unique(g) if int(v) != BLACK]
    found_any = False
    for color in colors:
        mask = (g == color) & (~used)
        for comp in _components(mask):
            bbox = _rectangle_ring_bbox(comp)
            if bbox is None:
                continue

            r0, c0, r1, c1 = bbox
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            marker_locs = np.argwhere(interior != BLACK)
            assert len(marker_locs) == 1, "each rectangle must contain exactly one marker"
            mr, mc = map(int, marker_locs[0])
            marker_color = int(interior[mr, mc])
            assert marker_color in templates, "marker color must exist in the legend"

            filled = _tile_template(templates[marker_color], interior.shape[0], interior.shape[1], marker_color)
            out[r0 + 1 : r1, c0 + 1 : c1] = filled
            found_any = True

    assert found_any, "expected at least one closed rectangle"
    return out


def _random_template(height: int, width: int) -> np.ndarray:
    for _ in range(120):
        style = random.choice(["checker", "stripes", "diag", "random", "columns"])
        mask = np.zeros((height, width), dtype=bool)

        if style == "checker":
            phase = random.randint(0, 1)
            rr, cc = np.indices((height, width))
            mask = ((rr + cc + phase) % 2) == 0
        elif style == "stripes":
            period = random.choice([2, 3])
            offset = random.randrange(period)
            rr, cc = np.indices((height, width))
            mask = ((cc + offset) % period) != 1
        elif style == "diag":
            shift = random.randrange(width)
            rr, cc = np.indices((height, width))
            mask = ((cc - rr - shift) % max(2, width - 1)) == 0
            mask |= ((cc + rr + shift) % max(2, width)) == 0
        elif style == "columns":
            cols = random.sample(range(width), k=random.randint(1, max(1, width - 1)))
            mask[:, cols] = True
            if random.random() < 0.5:
                mask[random.randrange(height), :] = ~mask[random.randrange(height), :]
        else:
            density = random.uniform(0.35, 0.7)
            mask = np.random.random((height, width)) < density

        for r in range(height):
            if not mask[r].any():
                mask[r, random.randrange(width)] = True
        if height == 3 and not mask[2].any():
            mask[2, random.randrange(width)] = True
        if mask.all() or not mask.any():
            continue
        if width >= 2 and all(mask[:, c].all() for c in range(width)):
            continue
        return mask

    fallback = np.zeros((height, width), dtype=bool)
    for r in range(height):
        for c in range(width):
            fallback[r, c] = (r + c) % 2 == 0
    fallback[-1, 0] = True
    fallback[0, -1] = False
    return fallback


def _choose_marker_colors(segment_colors: list[int], rect_count: int) -> list[int]:
    if rect_count == 1:
        return [random.choice(segment_colors)]

    if len(segment_colors) >= rect_count and random.random() < 0.5:
        return random.sample(segment_colors, rect_count)

    colors = [random.choice(segment_colors) for _ in range(rect_count)]
    if len(set(colors)) == rect_count:
        dup_idx = random.randrange(1, rect_count)
        colors[dup_idx] = colors[0]
    return colors


def _rand_inner_size(template: np.ndarray, max_h: int, max_w: int) -> tuple[int, int]:
    tmpl_h, tmpl_w = template.shape
    mode = random.choice(["small", "large", "mixed"])

    if mode == "small":
        inner_h = random.randint(1, min(max_h, tmpl_h))
        inner_w = random.randint(1, min(max_w, tmpl_w))
    elif mode == "large":
        lo_h = min(max_h, tmpl_h)
        lo_w = min(max_w, tmpl_w)
        inner_h = random.randint(lo_h, max_h) if lo_h <= max_h else random.randint(1, max_h)
        inner_w = random.randint(lo_w, max_w) if lo_w <= max_w else random.randint(1, max_w)
    else:
        inner_h = random.randint(1, max_h)
        inner_w = random.randint(1, max_w)

    return inner_h, inner_w


def _try_place_rectangles(
    grid_h: int,
    grid_w: int,
    start_row: int,
    rect_sizes: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    blocked[:start_row, :] = True

    placements: list[tuple[int, int] | None] = [None] * len(rect_sizes)
    order = sorted(range(len(rect_sizes)), key=lambda i: rect_sizes[i][0] * rect_sizes[i][1], reverse=True)

    for idx in order:
        rect_h, rect_w = rect_sizes[idx]
        row_lo = start_row
        row_hi = grid_h - rect_h - 1
        col_lo = 1
        col_hi = grid_w - rect_w - 1
        if row_lo > row_hi or col_lo > col_hi:
            return None

        placed = False
        for _ in range(250):
            r0 = random.randint(row_lo, row_hi)
            c0 = random.randint(col_lo, col_hi)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + rect_h + 1)
            cc1 = min(grid_w, c0 + rect_w + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            blocked[rr0:rr1, cc0:cc1] = True
            placements[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None

    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    segment_count = random.randint(2, 4)
    segment_widths = [random.randint(3, 5) for _ in range(segment_count)]
    template_heights = [random.randint(2, 3) for _ in range(segment_count)]

    usable_colors = COLOR_POOL[:]
    random.shuffle(usable_colors)
    segment_colors = usable_colors[:segment_count]
    border_color = usable_colors[segment_count]

    templates = {
        color: _random_template(height, width)
        for color, height, width in zip(segment_colors, template_heights, segment_widths, strict=False)
    }

    separators = [random.randint(1, 2) for _ in range(segment_count - 1)]
    legend_width = sum(segment_widths) + sum(separators)
    max_template_h = max(template_heights)
    top_margin = random.randint(0, 1)
    lower_gap = random.randint(1, 3)
    min_grid_h = top_margin + 1 + max_template_h + lower_gap + 5
    min_grid_w = max(legend_width, 12)
    if min_grid_h > 30 or min_grid_w > 30:
        return None

    grid_h = random.randint(min_grid_h, 30)
    grid_w = random.randint(min_grid_w, 30)
    legend_left = random.randint(0, grid_w - legend_width)
    content_start = top_margin + 1 + max_template_h + lower_gap

    lower_h = grid_h - content_start - 1
    if lower_h < 3:
        return None

    max_rects_by_height = max(1, min(4, lower_h // 4 + 1))
    rect_count = random.randint(1, max_rects_by_height)
    marker_colors = _choose_marker_colors(segment_colors, rect_count)

    rect_specs: list[dict[str, int]] = []
    rect_sizes: list[tuple[int, int]] = []
    for marker_color in marker_colors:
        template = templates[marker_color]
        max_inner_h = min(8, grid_h - content_start - 3)
        max_inner_w = min(10, grid_w - 4)
        if max_inner_h < 1 or max_inner_w < 1:
            return None
        inner_h, inner_w = _rand_inner_size(template, max_inner_h, max_inner_w)
        rect_specs.append(
            {
                "marker_color": marker_color,
                "inner_h": inner_h,
                "inner_w": inner_w,
            }
        )
        rect_sizes.append((inner_h + 2, inner_w + 2))

    placements = _try_place_rectangles(grid_h, grid_w, content_start, rect_sizes)
    if placements is None or len(placements) != len(rect_specs):
        return None

    inp = np.zeros((grid_h, grid_w), dtype=int)

    c = legend_left
    for i, color in enumerate(segment_colors):
        width = segment_widths[i]
        tmpl = templates[color]
        inp[top_margin, c : c + width] = color
        inp[top_margin + 1 : top_margin + 1 + tmpl.shape[0], c : c + width] = np.where(tmpl, GREY, BLACK)
        c += width
        if i < len(separators):
            c += separators[i]

    out = inp.copy()
    for spec, (r0, c0) in zip(rect_specs, placements, strict=False):
        inner_h = spec["inner_h"]
        inner_w = spec["inner_w"]
        marker_color = spec["marker_color"]

        inp[r0, c0 : c0 + inner_w + 2] = border_color
        inp[r0 + inner_h + 1, c0 : c0 + inner_w + 2] = border_color
        inp[r0 : r0 + inner_h + 2, c0] = border_color
        inp[r0 : r0 + inner_h + 2, c0 + inner_w + 1] = border_color

        marker_r = random.randint(r0 + 1, r0 + inner_h)
        marker_c = random.randint(c0 + 1, c0 + inner_w)
        inp[marker_r, marker_c] = marker_color

        out[r0, c0 : c0 + inner_w + 2] = border_color
        out[r0 + inner_h + 1, c0 : c0 + inner_w + 2] = border_color
        out[r0 : r0 + inner_h + 2, c0] = border_color
        out[r0 : r0 + inner_h + 2, c0 + inner_w + 1] = border_color
        out[r0 + 1 : r0 + inner_h + 1, c0 + 1 : c0 + inner_w + 1] = _tile_template(
            templates[marker_color], inner_h, inner_w, marker_color
        )

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
