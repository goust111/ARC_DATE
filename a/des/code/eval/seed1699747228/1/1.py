import random

import numpy as np


GEN_RETRIES = 400
ALL_COLORS = tuple(range(10))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _find_separator_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == grid[r, 0]) and int(grid[r, 0]) != bg]
    assert rows and rows[-1] < grid.shape[0] - 1, "expected a bottom grey separator row"
    sep = int(rows[-1])
    return sep, int(grid[sep, 0])


def _parse_templates(grid: np.ndarray, bg: int, grey: int, sep: int) -> list[np.ndarray]:
    area = grid[sep + 1 :]
    assert area.size > 0, "expected template rows below the separator"
    templates: list[np.ndarray] = []
    c = 0
    while c < area.shape[1]:
        if np.all(area[:, c] == grey):
            c += 1
            continue
        start = c
        while c < area.shape[1] and not np.all(area[:, c] == grey):
            c += 1
        template = area[:, start:c] != bg
        assert np.any(template) and np.any(~template), "each template needs foreground and background"
        templates.append(template)
    assert len(templates) == 4, "expected four bottom templates"
    return templates


def _region_seed_color(region: np.ndarray, bg: int, grey: int) -> int:
    coords = np.argwhere((region != bg) & (region != grey))
    assert len(coords) == 1, "each region must contain exactly one seed pixel"
    rr, cc = map(int, coords[0])
    return int(region[rr, cc])


def _grey_side_count(grid: np.ndarray, grey: int, bbox: tuple[int, int, int, int]) -> int:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    count = 0
    if r0 > 0 and np.all(grid[r0 - 1, c0 : c1 + 1] == grey):
        count += 1
    if r1 + 1 < h and np.all(grid[r1 + 1, c0 : c1 + 1] == grey):
        count += 1
    if c0 > 0 and np.all(grid[r0 : r1 + 1, c0 - 1] == grey):
        count += 1
    if c1 + 1 < w and np.all(grid[r0 : r1 + 1, c1 + 1] == grey):
        count += 1
    assert 1 <= count <= 4, "grey-side count must be between 1 and 4"
    return count


def _tile_mask(template: np.ndarray, height: int, width: int) -> np.ndarray:
    th, tw = template.shape
    reps = ((height + th - 1) // th, (width + tw - 1) // tw)
    return np.tile(template, reps)[:height, :width]


def _paint_region(out: np.ndarray, bbox: tuple[int, int, int, int], template: np.ndarray, color: int, bg: int) -> None:
    r0, c0, r1, c1 = bbox
    patch = out[r0 : r1 + 1, c0 : c1 + 1]
    patch[:] = bg
    patch[_tile_mask(template, patch.shape[0], patch.shape[1])] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select a bottom template from the seed region's grey-edge topology, then tile that
    template over the enclosing rectangle and recolor the template foreground with the
    seed color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    sep, grey = _find_separator_row(g, bg)
    templates = _parse_templates(g, bg, grey, sep)

    out = g.copy()
    for coords in _components(g[:sep] != grey):
        r0, c0, r1, c1 = _bbox(coords)
        assert coords.shape[0] == (r1 - r0 + 1) * (c1 - c0 + 1), "regions must be solid rectangles"
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        seed_color = _region_seed_color(region, bg, grey)
        k = _grey_side_count(g, grey, (r0, c0, r1, c1))
        _paint_region(out, (r0, c0, r1, c1), templates[k - 1], seed_color, bg)

    return out


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _template_bank(size: int) -> list[np.ndarray]:
    if size == 3:
        raw = [
            ["010", "111", "010"],
            ["100", "010", "001"],
            ["101", "010", "101"],
            ["111", "100", "100"],
            ["111", "010", "010"],
            ["110", "011", "001"],
            ["101", "111", "000"],
            ["011", "110", "011"],
        ]
    else:
        raw = [
            ["1010", "0101", "1010", "0101"],
            ["1001", "0110", "0110", "1001"],
            ["1111", "1001", "1001", "1111"],
            ["1111", "1000", "1000", "1000"],
            ["1111", "0110", "0110", "0110"],
            ["1100", "0110", "0011", "0001"],
            ["1100", "1100", "0011", "0011"],
            ["0010", "0111", "1111", "0111"],
        ]
    return [_mask_from_rows(rows) for rows in raw]


def _all_template_options(size: int) -> list[np.ndarray]:
    options: list[np.ndarray] = []
    seen: set[bytes] = set()
    for base in _template_bank(size):
        for turns in range(4):
            rotated = np.rot90(base, turns)
            for flipped in (rotated, np.fliplr(rotated)):
                key = flipped.tobytes()
                if key not in seen:
                    seen.add(key)
                    options.append(flipped.copy())
    return options


def _random_partition(total: int, parts: int, minimum: int) -> list[int]:
    extra = total - parts * minimum
    assert extra >= 0
    if parts == 1:
        return [total]
    bars = sorted(random.sample(range(extra + parts - 1), parts - 1))
    sizes: list[int] = []
    prev = -1
    for pos in bars + [extra + parts - 1]:
        sizes.append(pos - prev - 1 + minimum)
        prev = pos
    return sizes


def _choose_templates(size: int) -> list[np.ndarray]:
    options = _all_template_options(size)
    chosen = random.sample(options, 4)
    return [mask.copy() for mask in chosen]


def _gen_attempt() -> np.ndarray | None:
    size = random.choice((3, 4))
    width = 4 * size + 3
    top_h = random.randint(2, 5)
    mid_h = random.randint(2, 5)
    bottom_h = random.randint(2, 6)
    ncols = random.randint(3, 5)
    col_widths = _random_partition(width - (ncols - 1), ncols, 2)

    h1 = top_h
    h2 = top_h + 1 + mid_h
    sep = h2 + 1 + bottom_h
    height = sep + 1 + size

    colors = list(ALL_COLORS)
    random.shuffle(colors)
    region_count = 2 + ncols
    bg, grey, template_fg = colors[:3]
    seed_colors = colors[3 : 3 + region_count]

    grid = np.full((height, width), bg, dtype=int)
    grid[h1, :] = grey
    grid[h2, :] = grey
    grid[sep, :] = grey

    boundaries: list[int] = []
    cursor = col_widths[0]
    for width_part in col_widths[1:]:
        boundaries.append(cursor)
        cursor += 1 + width_part
    for c in boundaries:
        grid[h2:sep + 1, c] = grey

    templates = _choose_templates(size)
    cursor = 0
    for i, template in enumerate(templates):
        block = grid[sep + 1 :, cursor : cursor + size]
        block[:] = bg
        block[template] = template_fg
        cursor += size
        if i < 3:
            grid[sep + 1 :, cursor] = grey
            cursor += 1

    regions: list[tuple[tuple[int, int, int, int], int, int]] = [
        ((0, 0, top_h - 1, width - 1), 1, seed_colors[0]),
        ((h1 + 1, 0, h2 - 1, width - 1), 2, seed_colors[1]),
    ]

    left = 0
    for idx, span in enumerate(col_widths):
        right = left + span - 1
        grey_sides = 3 if idx in (0, len(col_widths) - 1) else 4
        regions.append(((h2 + 1, left, sep - 1, right), grey_sides, seed_colors[idx + 2]))
        left = right + 2

    for bbox, _k, seed_color in regions:
        r0, c0, r1, c1 = bbox
        sr = random.randint(r0, r1)
        sc = random.randint(c0, c1)
        grid[sr, sc] = seed_color

    expected = grid.copy()
    for bbox, k, seed_color in regions:
        _paint_region(expected, bbox, templates[k - 1], seed_color, bg)

    parsed_templates = _parse_templates(grid, bg, grey, sep)
    if not all(np.array_equal(a, b) for a, b in zip(parsed_templates, templates)):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
