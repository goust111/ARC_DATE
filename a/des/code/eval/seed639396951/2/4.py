import random
from collections import Counter, deque

import numpy as np


BG = 0
MARKER = 4
SEPARATOR = 5
PALETTE = [1, 2, 3, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

GEN_ATTEMPTS = 300
MAX_GRID_SIDE = 30


def _bg_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(Counter(map(int, values)).most_common(1)[0][0])


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIRS8 if connectivity == 8 else DIRS4
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(np.asarray(cells, dtype=int))
    return out


def _separator_info(grid: np.ndarray, bg: int) -> tuple[list[int], int]:
    separator_rows = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != bg and np.all(grid[r] == grid[r, 0])
    ]
    assert separator_rows, "expected at least one separator row"
    sep_colors = [int(grid[r, 0]) for r in separator_rows]
    sep_color = Counter(sep_colors).most_common(1)[0][0]
    assert all(color == sep_color for color in sep_colors), "separator rows must share one color"
    return separator_rows, int(sep_color)


def _band_spans(height: int, separator_rows: list[int]) -> list[tuple[int, int]]:
    sep = set(separator_rows)
    spans: list[tuple[int, int]] = []
    r = 0
    while r < height:
        while r < height and r in sep:
            r += 1
        if r >= height:
            break
        start = r
        while r < height and r not in sep:
            r += 1
        spans.append((start, r))
    return spans


def _marker_position(grid: np.ndarray, band_spans: list[tuple[int, int]], sep_rows: list[int], bg: int) -> tuple[int, int]:
    sep = set(sep_rows)
    candidates: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        if r in sep or int(grid[r, 0]) == bg:
            continue
        color = int(grid[r, 0])
        comps = _components(grid == color, connectivity=8)
        for comp in comps:
            pts = {tuple(map(int, cell)) for cell in comp}
            if (r, 0) in pts and len(pts) == 1:
                candidates.append((r, 0))
                break
    assert len(candidates) == 1, "expected exactly one singleton left-edge marker"
    marker = candidates[0]
    assert any(r0 <= marker[0] < r1 for r0, r1 in band_spans), "marker must lie inside a band"
    return marker


def _sprite_from_component(color: int, coords: np.ndarray) -> dict:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    rel = coords - np.array([r0, c0], dtype=int)
    return {
        "color": int(color),
        "cells": rel.astype(int),
        "height": int(r1 - r0 + 1),
        "width": int(c1 - c0 + 1),
        "left": int(c0),
        "right": int(c1),
    }


def _extract_band_sprites(
    grid: np.ndarray,
    span: tuple[int, int],
    bg: int,
    sep_color: int,
    marker: tuple[int, int] | None = None,
) -> dict[int, dict]:
    r0, r1 = span
    band = np.array(grid[r0:r1], dtype=int)
    if marker is not None and r0 <= marker[0] < r1:
        band[marker[0] - r0, marker[1]] = bg
    colors = [int(v) for v in np.unique(band) if int(v) not in (bg, sep_color)]
    sprites: dict[int, dict] = {}
    for color in colors:
        comps = _components(band == color, connectivity=8)
        assert len(comps) == 1, "each band color must form exactly one sprite"
        sprites[color] = _sprite_from_component(color, comps[0])
    return sprites


def _place_band(
    canvas: np.ndarray,
    span: tuple[int, int],
    sprites: dict[int, dict],
    order: list[int],
    gaps: list[int],
    start_col: int,
    bg: int,
) -> None:
    r0, r1 = span
    band_h = r1 - r0
    x = int(start_col)
    for index, color in enumerate(order):
        sprite = sprites[color]
        top = r0 + (band_h - sprite["height"]) // 2
        for dr, dc in sprite["cells"]:
            rr = top + int(dr)
            cc = x + int(dc)
            assert 0 <= rr < canvas.shape[0] and 0 <= cc < canvas.shape[1], "sprite placement out of bounds"
            assert int(canvas[rr, cc]) == bg, "sprites must not overlap"
            canvas[rr, cc] = int(color)
        x += sprite["width"]
        if index < len(gaps):
            x += int(gaps[index])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder each row band's sprites to match the prototype row's color order and spacing pattern.

    Input structure:
    - Black background with full-width grey separator rows splitting the canvas into equal-height bands.
    - Each band contains one sprite per color; one band's left edge has a singleton marker pixel.

    Rule:
    - The marker identifies the prototype band.
    - Read that band's sprite colors from left to right and record every adjacent horizontal gap.
    - Rebuild every band using its own sprite shapes, but in the prototype color order and with the same gaps.
    - Start each rebuilt sequence at column 0, vertically center the sprites in the band, and remove the marker.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    sep_rows, sep_color = _separator_info(g, bg)
    spans = _band_spans(g.shape[0], sep_rows)
    assert len(spans) >= 2, "expected multiple row bands"
    heights = {r1 - r0 for r0, r1 in spans}
    assert len(heights) == 1, "bands must have equal height"

    marker = _marker_position(g, spans, sep_rows, bg)
    proto_index = next(i for i, (r0, r1) in enumerate(spans) if r0 <= marker[0] < r1)

    band_sprites = [
        _extract_band_sprites(g, span, bg, sep_color, marker if i == proto_index else None)
        for i, span in enumerate(spans)
    ]
    template_sprites = band_sprites[proto_index]
    ordered = sorted(template_sprites.values(), key=lambda sprite: (sprite["left"], sprite["color"]))
    template_order = [sprite["color"] for sprite in ordered]
    template_gaps = [
        int(ordered[i + 1]["left"] - ordered[i]["right"] - 1)
        for i in range(len(ordered) - 1)
    ]

    out = np.full_like(g, bg)
    out[sep_rows, :] = sep_color
    template_colors = set(template_order)
    for span, sprites in zip(spans, band_sprites):
        assert set(sprites) == template_colors, "all bands must contain the same color set"
        _place_band(out, span, sprites, template_order, template_gaps, start_col=0, bg=bg)
    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> dict:
    arr = np.asarray(sorted(cells), dtype=int)
    r0 = int(arr[:, 0].min())
    c0 = int(arr[:, 1].min())
    arr = arr - np.array([r0, c0], dtype=int)
    return _sprite_from_component(-1, arr)


def _random_shape(band_h: int, max_w: int) -> dict | None:
    for _ in range(80):
        h = random.randint(1, band_h)
        w = random.randint(1, max_w)
        max_area = min(h * w, 5)
        if max_area < 2:
            continue
        target_area = random.randint(2, max_area)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < target_area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target_area:
            continue
        sprite = _normalize_cells(cells)
        sprite["color"] = -1
        if sprite["width"] <= max_w and sprite["height"] <= band_h:
            return sprite
    return None


def _sequence_width(sprites: dict[int, dict], order: list[int], gaps: list[int], start_col: int) -> int:
    width = int(start_col)
    for i, color in enumerate(order):
        width += int(sprites[color]["width"])
        if i < len(gaps):
            width += int(gaps[i])
    return width


def _layout_grid_rows(band_count: int, band_h: int, top_sep: bool, bottom_sep: bool) -> tuple[int, list[int], list[tuple[int, int]]]:
    row = 0
    sep_rows: list[int] = []
    spans: list[tuple[int, int]] = []
    if top_sep:
        sep_rows.append(row)
        row += 1
    for band in range(band_count):
        spans.append((row, row + band_h))
        row += band_h
        if band < band_count - 1 or bottom_sep:
            sep_rows.append(row)
            row += 1
    return row, sep_rows, spans


def _gen_attempt() -> np.ndarray | None:
    band_count = random.randint(3, 5)
    band_h = random.randint(3, 5)
    color_count = random.randint(3, 5)
    top_sep = random.choice([False, True])
    bottom_sep = random.choice([False, True])
    grid_h, sep_rows, spans = _layout_grid_rows(band_count, band_h, top_sep, bottom_sep)
    if grid_h > MAX_GRID_SIDE:
        return None

    colors = random.sample(PALETTE, color_count)
    proto_index = random.randrange(band_count)
    max_shape_w = 3 if color_count >= 4 else 4

    shape_rows: list[dict[int, dict]] = []
    for _ in range(band_count):
        row_shapes: dict[int, dict] = {}
        for color in colors:
            sprite = _random_shape(band_h, max_shape_w)
            if sprite is None:
                return None
            sprite["color"] = color
            row_shapes[color] = sprite
        shape_rows.append(row_shapes)

    template_order = random.sample(colors, color_count)
    small_gap = random.randint(1, 2)
    large_gap = random.randint(small_gap + 1, 4)
    template_gaps = [random.choice([small_gap, large_gap]) for _ in range(color_count - 1)]
    if len(set(template_gaps)) == 1:
        template_gaps[random.randrange(len(template_gaps))] = large_gap if template_gaps[0] == small_gap else small_gap

    row_specs: list[dict] = []
    max_width_needed = 0
    for band_index in range(band_count):
        if band_index == proto_index:
            order = list(template_order)
            gaps = list(template_gaps)
            left_margin = random.randint(2, 4)
        else:
            order = random.sample(colors, color_count)
            while order == template_order:
                order = random.sample(colors, color_count)
            uniform_gap = random.randint(1, 2)
            gaps = [uniform_gap] * (color_count - 1)
            left_margin = random.randint(0, 4)
        in_width = _sequence_width(shape_rows[band_index], order, gaps, left_margin)
        out_width = _sequence_width(shape_rows[band_index], template_order, template_gaps, 0)
        max_width_needed = max(max_width_needed, in_width, out_width)
        row_specs.append({
            "order": order,
            "gaps": gaps,
            "left_margin": left_margin,
        })

    width = max_width_needed + random.randint(0, 3)
    if width > MAX_GRID_SIDE:
        return None

    grid = np.full((grid_h, width), BG, dtype=int)
    grid[sep_rows, :] = SEPARATOR
    expected = np.full_like(grid, BG)
    expected[sep_rows, :] = SEPARATOR

    for band_index, span in enumerate(spans):
        spec = row_specs[band_index]
        _place_band(
            grid,
            span,
            shape_rows[band_index],
            spec["order"],
            spec["gaps"],
            spec["left_margin"],
            BG,
        )
        _place_band(
            expected,
            span,
            shape_rows[band_index],
            template_order,
            template_gaps,
            0,
            BG,
        )

    marker_row = random.randint(spans[proto_index][0], spans[proto_index][1] - 1)
    if grid[marker_row, 0] != BG:
        return None
    grid[marker_row, 0] = MARKER

    if np.array_equal(grid, expected):
        return None
    if not all(len(set(spec["gaps"])) == 1 for i, spec in enumerate(row_specs) if i != proto_index):
        return None
    if len(set(template_gaps)) < 2:
        return None
    if not any(spec["order"] != template_order for i, spec in enumerate(row_specs) if i != proto_index):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
