from __future__ import annotations

import random

import numpy as np


AZURE = 8
COLORS = tuple(range(10))
GEN_ATTEMPTS = 400
GEN_GRID_H_RANGE = (14, 30)
GEN_GRID_W_RANGE = (12, 30)
GEN_TOWER_COUNT_RANGE = (2, 4)
GEN_BODY_LEN_RANGE = (2, 5)
GEN_MIN_COL_GAP = 3


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(int(r), int(c))]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _extract_towers(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    bg = _most_common_color(g)
    assert bg != AZURE, "the antenna color must stay distinct from the background"

    towers: list[dict[str, object]] = []
    for coords in _components4(g != bg):
        rows = sorted(int(r) for r, _ in coords)
        cols = {int(c) for _, c in coords}
        assert len(cols) == 1, "each tower must occupy a single column"
        col = next(iter(cols))
        top = rows[0]
        bottom = rows[-1]
        assert rows == list(range(top, bottom + 1)), "tower pixels must be vertically contiguous"

        for rr in range(top, bottom + 1):
            if col > 0:
                assert int(g[rr, col - 1]) == bg, "tower body must remain one-pixel wide"
            if col + 1 < width:
                assert int(g[rr, col + 1]) == bg, "tower body must remain one-pixel wide"

        column = [int(v) for v in g[top : bottom + 1, col]]
        azure_offsets = [i for i, value in enumerate(column) if value == AZURE]
        assert len(azure_offsets) == 1, "each tower must contain exactly one azure antenna"
        antenna_offset = azure_offsets[0]
        assert antenna_offset in (0, len(column) - 1), "the antenna must sit at one end of the tower"

        if antenna_offset == 0:
            direction = -1
            antenna_row = top
            body_top_to_bottom = column[1:]
            band_colors = list(reversed(body_top_to_bottom))
        else:
            direction = 1
            antenna_row = bottom
            body_top_to_bottom = column[:-1]
            band_colors = body_top_to_bottom[:]

        assert len(body_top_to_bottom) >= 2, "tower body must contain at least two pixels"
        assert all(color != AZURE for color in body_top_to_bottom), "only the antenna may be azure"

        towers.append(
            {
                "top": int(top),
                "col": int(col),
                "antenna_row": int(antenna_row),
                "direction": int(direction),
                "body_top_to_bottom": tuple(int(v) for v in body_top_to_bottom),
                "band_colors": tuple(int(v) for v in band_colors),
            }
        )

    assert towers, "expected at least one signal tower"
    towers.sort(key=lambda tower: (int(tower["top"]), int(tower["col"])))
    return towers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Broadcast widening horizontal color bands outward from each single-column signal tower.

    Input structure:
    - The background is the most common color.
    - Each foreground object is a one-column vertical tower: a contiguous body of at least two non-background pixels plus exactly one azure antenna pixel attached at the top or bottom end.
    - Tower bodies may use multiple colors, and different towers may have different heights, colors, columns, and antenna directions.

    Rule:
    - Read each tower independently. An antenna on the top means broadcast upward; an antenna on the bottom means broadcast downward.
    - For a tower body, read its colors from the end farthest from the antenna toward the antenna, obtaining the sequence C1, C2, ..., Cn.
    - Starting one row beyond the antenna and moving along the broadcast direction, draw n horizontal bands centered on the tower column. Band i uses color Ci and width 2*i-1.
    - Clip bands at the grid boundary, keep the original towers unchanged, and only paint background cells. If towers or earlier bands block a cell, leave the existing pixel in place. Process towers from smaller top row to larger top row.

    Key insight:
    - The antenna gives the direction, the tower-body colors are reused in reversed near/far order, and the band widths grow as odd numbers centered on the tower column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    bg = _most_common_color(g)

    out = g.copy()
    for tower in _extract_towers(g):
        antenna_row = int(tower["antenna_row"])
        col = int(tower["col"])
        direction = int(tower["direction"])
        for i, color in enumerate(tower["band_colors"], start=1):
            row = antenna_row + direction * i
            if not (0 <= row < height):
                continue
            half = i - 1
            c0 = max(0, col - half)
            c1 = min(width - 1, col + half)
            for cc in range(c0, c1 + 1):
                if int(out[row, cc]) == bg:
                    out[row, cc] = int(color)
    return out


def _choose_columns(width: int, count: int) -> list[int] | None:
    cols = list(range(width))
    random.shuffle(cols)
    chosen: list[int] = []
    for col in cols:
        if all(abs(col - prev) >= GEN_MIN_COL_GAP for prev in chosen):
            chosen.append(int(col))
            if len(chosen) == count:
                chosen.sort()
                return chosen
    return None


def _placement_options(height: int, body_len: int, direction: int) -> list[tuple[int, int, bool]]:
    options: list[tuple[int, int, bool]] = []
    if direction == -1:
        for antenna_row in range(1, height - body_len):
            top = antenna_row
            clipped = antenna_row < body_len
            options.append((int(antenna_row), int(top), bool(clipped)))
    else:
        for antenna_row in range(body_len, height - 1):
            top = antenna_row - body_len
            clipped = antenna_row > height - body_len - 1
            options.append((int(antenna_row), int(top), bool(clipped)))
    return options


def _pick_body_colors(body_len: int, bg: int) -> list[int]:
    pool = [color for color in COLORS if color not in {bg, AZURE}]
    if random.random() < 0.25:
        return [int(random.choice(pool)) for _ in range(body_len)]

    palette_size = random.randint(2, min(4, len(pool)))
    palette = random.sample(pool, palette_size)
    colors = [int(random.choice(palette)) for _ in range(body_len)]
    if len(set(colors)) == 1:
        idx = random.randrange(body_len)
        replacement = random.choice([color for color in palette if color != colors[idx]])
        colors[idx] = int(replacement)
    return colors


def _balanced_directions(count: int) -> list[int]:
    half = count // 2
    directions = [-1] * half + [1] * half
    if count % 2 == 1:
        directions.append(random.choice([-1, 1]))
    random.shuffle(directions)
    return directions


def _spec_signature(spec: dict[str, object]) -> tuple[int, int, int, tuple[int, ...]]:
    body = list(spec["body_top_to_bottom"])
    direction = int(spec["direction"])
    if direction == -1:
        band_colors = tuple(reversed(body))
    else:
        band_colors = tuple(body)
    return (int(spec["top"]), int(spec["col"]), direction, band_colors)


def _parsed_signature(tower: dict[str, object]) -> tuple[int, int, int, tuple[int, ...]]:
    return (
        int(tower["top"]),
        int(tower["col"]),
        int(tower["direction"]),
        tuple(int(v) for v in tower["band_colors"]),
    )


def _render_input(height: int, width: int, bg: int, specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((height, width), int(bg), dtype=int)
    for spec in specs:
        col = int(spec["col"])
        antenna_row = int(spec["antenna_row"])
        direction = int(spec["direction"])
        body = list(spec["body_top_to_bottom"])
        if direction == -1:
            grid[antenna_row, col] = AZURE
            for i, color in enumerate(body, start=1):
                grid[antenna_row + i, col] = int(color)
        else:
            start = antenna_row - len(body)
            for i, color in enumerate(body):
                grid[start + i, col] = int(color)
            grid[antenna_row, col] = AZURE
    return grid


def _rich_enough(specs: list[dict[str, object]]) -> bool:
    directions = {int(spec["direction"]) for spec in specs}
    has_multicolor = any(len(set(spec["body_top_to_bottom"])) >= 2 for spec in specs)
    has_long = any(len(spec["body_top_to_bottom"]) >= 3 for spec in specs)
    has_clipped = any(bool(spec["clipped"]) for spec in specs)
    has_unclipped = any(not bool(spec["clipped"]) for spec in specs)
    return directions == {-1, 1} and has_multicolor and has_long and has_clipped and has_unclipped


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GEN_GRID_H_RANGE)
    width = random.randint(*GEN_GRID_W_RANGE)
    count = random.randint(*GEN_TOWER_COUNT_RANGE)
    bg = random.choice([color for color in COLORS if color != AZURE])

    columns = _choose_columns(width, count)
    if columns is None:
        return None

    specs: list[dict[str, object]] = []
    used_tops: set[int] = set()
    for col, direction in zip(columns, _balanced_directions(count)):
        body_len = random.randint(*GEN_BODY_LEN_RANGE)
        options = [opt for opt in _placement_options(height, body_len, direction) if opt[1] not in used_tops]
        if not options:
            return None
        clipped = [opt for opt in options if opt[2]]
        unclipped = [opt for opt in options if not opt[2]]
        if clipped and unclipped:
            antenna_row, top, is_clipped = random.choice(clipped if random.random() < 0.5 else unclipped)
        else:
            antenna_row, top, is_clipped = random.choice(options)
        used_tops.add(int(top))

        specs.append(
            {
                "top": int(top),
                "col": int(col),
                "direction": int(direction),
                "antenna_row": int(antenna_row),
                "body_top_to_bottom": tuple(_pick_body_colors(body_len, bg)),
                "clipped": bool(is_clipped),
            }
        )

    if not _rich_enough(specs):
        return None

    grid = _render_input(height, width, bg, specs)
    parsed = _extract_towers(grid)
    if sorted(_parsed_signature(tower) for tower in parsed) != sorted(_spec_signature(spec) for spec in specs):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(out != grid)) < count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower sample")
