from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
ARROW_COLOR = 2
SEPARATOR_COLOR = 5
SOURCE_COLORS = (1, 3, 4, 6, 7, 8, 9)
GENERATION_RETRIES = 400

ARROW_SHAPES = {
    "up": {(1, 1), (2, 0), (2, 1), (2, 2)},
    "down": {(0, 0), (0, 1), (0, 2), (1, 1)},
    "left": {(0, 2), (1, 1), (1, 2), (2, 2)},
    "right": {(0, 0), (1, 0), (1, 1), (2, 0)},
}

ARROW_COMPONENT_SHAPES = {
    "up": {(0, 1), (1, 0), (1, 1), (1, 2)},
    "down": {(0, 0), (0, 1), (0, 2), (1, 1)},
    "left": {(0, 1), (1, 0), (1, 1), (2, 1)},
    "right": {(0, 0), (1, 0), (1, 1), (2, 0)},
}

CENTER_OFFSETS = {
    "up": (0, 1),
    "down": (1, 1),
    "left": (1, 0),
    "right": (1, 1),
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _separator_info(grid: np.ndarray) -> tuple[str, int]:
    full_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEPARATOR_COLOR)]
    full_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEPARATOR_COLOR)]
    assert len(full_rows) + len(full_cols) == 1, "expected one grey separator line"
    if full_rows:
        return "horizontal", int(full_rows[0])
    return "vertical", int(full_cols[0])


def _arrow_direction(coords: list[tuple[int, int]]) -> tuple[str, int, int] | None:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, c0 = min(rows), min(cols)
    if len(coords) != 4:
        return None
    rel = {(r - r0, c - c0) for r, c in coords}
    for direction, shape in ARROW_COMPONENT_SHAPES.items():
        if rel == shape:
            return direction, r0, c0
    return None


def _find_arrows(grid: np.ndarray, axis: str, sep: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for comp in _color_components(grid, ARROW_COLOR):
        match = _arrow_direction(comp)
        if match is None:
            continue
        direction, r0, c0 = match
        r1 = r0 + 2
        c1 = c0 + 2
        if axis == "horizontal":
            if direction == "up" and r0 > sep:
                dr, dc = CENTER_OFFSETS[direction]
                records.append({"direction": direction, "center": (r0 + dr, c0 + dc), "coords": comp})
            elif direction == "down" and r1 < sep:
                dr, dc = CENTER_OFFSETS[direction]
                records.append({"direction": direction, "center": (r0 + dr, c0 + dc), "coords": comp})
        else:
            if direction == "left" and c0 > sep:
                dr, dc = CENTER_OFFSETS[direction]
                records.append({"direction": direction, "center": (r0 + dr, c0 + dc), "coords": comp})
            elif direction == "right" and c1 < sep:
                dr, dc = CENTER_OFFSETS[direction]
                records.append({"direction": direction, "center": (r0 + dr, c0 + dc), "coords": comp})

    assert records, "expected at least one red arrow marker"
    directions = {record["direction"] for record in records}
    assert len(directions) == 1, "all arrows must point toward the same separator"
    records.sort(key=lambda rec: rec["center"][1] if axis == "horizontal" else rec["center"][0])
    return records


def _slice_from_source(grid: np.ndarray, axis: str, sep: int, direction: str, center: tuple[int, int]) -> list[int]:
    h, w = grid.shape
    r, c = center
    values: list[int] = []
    if axis == "horizontal":
        if direction == "up":
            rows = range(sep - 1, -1, -1)
        else:
            rows = range(sep + 1, h)
        for rr in rows:
            value = int(grid[rr, c])
            if value != BACKGROUND:
                values.append(value)
    else:
        if direction == "left":
            cols = range(sep - 1, -1, -1)
        else:
            cols = range(sep + 1, w)
        for cc in cols:
            value = int(grid[r, cc])
            if value != BACKGROUND:
                values.append(value)
    return values


def _draw_slice(out: np.ndarray, direction: str, center: tuple[int, int], values: list[int]) -> None:
    r, c = center
    if direction == "up":
        dr, dc = 1, 0
    elif direction == "down":
        dr, dc = -1, 0
    elif direction == "left":
        dr, dc = 0, 1
    else:
        dr, dc = 0, -1

    for i, value in enumerate(values):
        rr = r + dr * i
        cc = c + dc * i
        out[rr, cc] = int(value)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use red arrowheads as projection instructions that copy source slices across a grey divider into the blank half.

    Input structure:
    - The background is black, and one full grey row or column splits the grid into a source region and a target region.
    - The source region contains one connected multi-color pattern made from non-background colors.
    - The target region is otherwise black, except for one or more red 3x3 arrowhead markers placed near the separator.
    - Each arrow points toward the separator and its center row/column identifies the source row or column to sample.

    Rule:
    - Detect whether the separator is horizontal or vertical, then determine which side is the source from the common arrow direction.
    - For each arrow, read the corresponding source column (horizontal separator) or source row (vertical separator), keeping only non-black pixels in order from the separator outward.
    - Remove all red arrow pixels from the target region.
    - Starting at the arrow's center cell, draw the sampled slice in the direction opposite the arrow, i.e. away from the separator.

    Key insight:
    - Each arrowhead simultaneously selects a source cross-section and specifies the projection direction, so the same marker encodes both "what to copy" and "where to print it".
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    axis, sep = _separator_info(g)
    arrows = _find_arrows(g, axis, sep)
    direction = str(arrows[0]["direction"])

    out = g.copy()
    for arrow in arrows:
        for r, c in arrow["coords"]:
            out[r, c] = BACKGROUND

    for arrow in arrows:
        center = tuple(arrow["center"])
        values = _slice_from_source(g, axis, sep, direction, center)
        _draw_slice(out, direction, center, values)

    return out


def _spaced_centers(length: int, count: int) -> list[int] | None:
    candidates = list(range(1, length - 1))
    for _ in range(120):
        picks = sorted(random.sample(candidates, count))
        if all(b - a >= 4 for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _pick_palette() -> tuple[int, ...]:
    size = random.randint(2, min(4, len(SOURCE_COLORS)))
    return tuple(random.sample(SOURCE_COLORS, size))


def _sample_sequences(count: int, max_len: int, palette: tuple[int, ...]) -> list[list[int]]:
    for _ in range(120):
        sequences = [
            [random.choice(palette) for _ in range(random.randint(1, max_len))]
            for _ in range(count)
        ]
        used = {value for seq in sequences for value in seq}
        if len(used) >= 2 and any(len(seq) > 1 for seq in sequences):
            return sequences
    raise RuntimeError("failed to sample diverse slice sequences")


def _paint_arrow(grid: np.ndarray, top: int, left: int, direction: str) -> None:
    for dr, dc in ARROW_SHAPES[direction]:
        grid[top + dr, left + dc] = ARROW_COLOR


def _expected_from_layout(
    grid: np.ndarray,
    axis: str,
    source_on_first_side: bool,
    sep: int,
    centers: list[int],
    sequences: list[list[int]],
) -> np.ndarray:
    out = grid.copy()

    if axis == "horizontal":
        direction = "up" if source_on_first_side else "down"
        arrow_top = sep + 1 if source_on_first_side else sep - 3
        for c in centers:
            for dr, dc in ARROW_SHAPES[direction]:
                out[arrow_top + dr, c - 1 + dc] = BACKGROUND
            start_r = sep + 2 if source_on_first_side else sep - 2
            step = 1 if source_on_first_side else -1
            for i, value in enumerate(sequences[centers.index(c)]):
                out[start_r + step * i, c] = int(value)
    else:
        direction = "left" if source_on_first_side else "right"
        arrow_left = sep + 1 if source_on_first_side else sep - 3
        for r in centers:
            for dr, dc in ARROW_SHAPES[direction]:
                out[r - 1 + dr, arrow_left + dc] = BACKGROUND
            start_c = sep + 2 if source_on_first_side else sep - 2
            step = 1 if source_on_first_side else -1
            for i, value in enumerate(sequences[centers.index(r)]):
                out[r, start_c + step * i] = int(value)

    return out


def _build_horizontal(source_above: bool) -> tuple[np.ndarray, np.ndarray] | None:
    width = random.randint(10, 26)
    max_arrows = min(4, (width + 1) // 4)
    if max_arrows < 2:
        return None
    arrow_count = random.randint(2, max_arrows)
    centers = _spaced_centers(width, arrow_count)
    if centers is None:
        return None

    max_len = random.randint(2, 6)
    source_depth = random.randint(max_len, max_len + 4)
    target_depth = random.randint(max_len + 1, max_len + 5)
    height = source_depth + 1 + target_depth
    if height > 30:
        return None

    palette = _pick_palette()
    sequences = _sample_sequences(arrow_count, max_len, palette)
    grid = np.full((height, width), BACKGROUND, dtype=int)

    sep = source_depth if source_above else target_depth
    grid[sep, :] = SEPARATOR_COLOR
    spine_r = sep - 1 if source_above else sep + 1
    bar_start = max(0, centers[0] - random.randint(0, 2))
    bar_end = min(width - 1, centers[-1] + random.randint(0, 2))
    for c in range(bar_start, bar_end + 1):
        grid[spine_r, c] = random.choice(palette)

    for c, seq in zip(centers, sequences):
        for i, value in enumerate(seq):
            rr = spine_r - i if source_above else spine_r + i
            grid[rr, c] = int(value)

    arrow_dir = "up" if source_above else "down"
    arrow_top = sep + 1 if source_above else sep - 3
    for c in centers:
        _paint_arrow(grid, arrow_top, c - 1, arrow_dir)

    expected = _expected_from_layout(grid, "horizontal", source_above, sep, centers, sequences)
    return grid, expected


def _build_vertical(source_left: bool) -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(10, 26)
    max_arrows = min(4, (height + 1) // 4)
    if max_arrows < 2:
        return None
    arrow_count = random.randint(2, max_arrows)
    centers = _spaced_centers(height, arrow_count)
    if centers is None:
        return None

    max_len = random.randint(2, 6)
    source_width = random.randint(max_len, max_len + 4)
    target_width = random.randint(max_len + 1, max_len + 5)
    width = source_width + 1 + target_width
    if width > 30:
        return None

    palette = _pick_palette()
    sequences = _sample_sequences(arrow_count, max_len, palette)
    grid = np.full((height, width), BACKGROUND, dtype=int)

    sep = source_width if source_left else target_width
    grid[:, sep] = SEPARATOR_COLOR
    spine_c = sep - 1 if source_left else sep + 1
    bar_start = max(0, centers[0] - random.randint(0, 2))
    bar_end = min(height - 1, centers[-1] + random.randint(0, 2))
    for r in range(bar_start, bar_end + 1):
        grid[r, spine_c] = random.choice(palette)

    for r, seq in zip(centers, sequences):
        for i, value in enumerate(seq):
            cc = spine_c - i if source_left else spine_c + i
            grid[r, cc] = int(value)

    arrow_dir = "left" if source_left else "right"
    arrow_left = sep + 1 if source_left else sep - 3
    for r in centers:
        _paint_arrow(grid, r - 1, arrow_left, arrow_dir)

    expected = _expected_from_layout(grid, "vertical", source_left, sep, centers, sequences)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        if random.random() < 0.5:
            sample = _build_horizontal(source_above=random.random() < 0.5)
        else:
            sample = _build_vertical(source_left=random.random() < 0.5)
        if sample is None:
            continue
        inp, expected = sample
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid projection puzzle")
