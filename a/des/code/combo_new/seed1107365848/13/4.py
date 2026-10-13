from __future__ import annotations

import random

import numpy as np


BLACK = 0
ALL_COLORS = tuple(range(1, 10))
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 160


def _parse_palette_row(row: np.ndarray) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    i = 0
    saw_pair = False
    values = [int(v) for v in row]
    while i < len(values):
        if values[i] == BLACK:
            i += 1
            continue
        j = i
        while j < len(values) and values[j] != BLACK:
            j += 1
        if j - i != 2:
            return None
        a, b = values[i], values[i + 1]
        if a == b or a in mapping or b in mapping:
            return None
        mapping[a] = b
        mapping[b] = a
        saw_pair = True
        i = j
    return mapping if saw_pair else None


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
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
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(coords)
    return parts


def _ordered_line(coords: list[tuple[int, int]]) -> tuple[str, list[tuple[int, int]]]:
    assert len(coords) >= 2, "paths must have length at least 2"
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    if len(rows) == 1:
        row = rows[0]
        ordered_cols = sorted(c for _, c in coords)
        assert ordered_cols == list(range(ordered_cols[0], ordered_cols[-1] + 1)), "horizontal path must be contiguous"
        return "horizontal", [(int(row), int(c)) for c in ordered_cols]
    if len(cols) == 1:
        col = cols[0]
        ordered_rows = sorted(r for r, _ in coords)
        assert ordered_rows == list(range(ordered_rows[0], ordered_rows[-1] + 1)), "vertical path must be contiguous"
        return "vertical", [(int(r), int(col)) for r in ordered_rows]
    raise AssertionError("each component must be a straight line")


def _solve_for_palette(
    grid: np.ndarray,
    palette_row: int,
    mapping: dict[int, int],
) -> np.ndarray:
    h, w = grid.shape
    out = grid.copy()
    paths: list[tuple[list[tuple[int, int]], list[tuple[int, int]], int]] = []
    dest_used: set[tuple[int, int]] = set()

    found_path = False
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BLACK):
        mask = (grid == color).copy()
        mask[palette_row, :] = False
        if not bool(mask.any()):
            continue
        assert color in mapping, "every path color must appear in the palette row"
        for comp in _components(mask):
            found_path = True
            orient, ordered = _ordered_line(comp)
            if orient == "horizontal":
                shifted = [(r + 1, c) for r, c in ordered]
            else:
                shifted = [(r, c + 1) for r, c in ordered]
            for rr, cc in shifted:
                assert 0 <= rr < h and 0 <= cc < w, "shifted path must stay in bounds"
                assert rr != palette_row, "shifted path cannot overwrite the palette row"
                assert (rr, cc) not in dest_used, "shifted paths must not overlap"
            dest_used.update(shifted)
            paths.append((ordered, shifted, int(mapping[color])))

    assert found_path, "expected at least one path"

    for ordered, _shifted, _new_color in paths:
        for r, c in ordered:
            out[r, c] = BLACK
    for _ordered, shifted, new_color in paths:
        for r, c in shifted:
            out[r, c] = int(new_color)
    return out


def _candidate_outputs(grid: np.ndarray) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    for palette_row in sorted({0, int(grid.shape[0] - 1)}):
        mapping = _parse_palette_row(grid[palette_row])
        if mapping is None:
            continue
        try:
            candidates.append(_solve_for_palette(grid, palette_row, mapping))
        except AssertionError:
            continue
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each line path with its palette partner color and shift it by one cell according to orientation.

    Input structure:
    - The background is black.
    - The first row or the last row is a palette row made of adjacent color pairs,
      with each pair separated from the next by black cells.
    - Away from the palette row, the non-black cells form disjoint horizontal or
      vertical straight paths, and every path color appears in exactly one palette pair.

    Rule:
    - Parse the palette row so that the two colors inside each adjacent pair become
      mutual replacements.
    - For every monochrome straight path, recolor the whole path to its paired palette color.
    - Shift each horizontal path one row downward and each vertical path one column
      to the right, clearing the original path cells to black.
    - Leave the palette row unchanged.

    Key insight:
    - The palette row encodes a bidirectional color swap, while the path geometry
      encodes the motion direction, so the final output combines an independent
      recoloring rule with an orientation-dependent translation rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 2, "expected a 2D ARC grid"
    candidates = _candidate_outputs(g)
    assert candidates, "expected a valid palette row on the first or last row"
    first = candidates[0]
    for other in candidates[1:]:
        assert np.array_equal(other, first), "palette-row detection must be unambiguous"
    return first


def _palette_sequence(pairs: list[tuple[int, int]]) -> list[int]:
    seq: list[int] = []
    for index, (a, b) in enumerate(pairs):
        if index:
            seq.append(BLACK)
        seq.extend((int(a), int(b)))
    return seq


def _shifted_cells(orientation: str, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if orientation == "horizontal":
        return [(r + 1, c) for r, c in cells]
    return [(r, c + 1) for r, c in cells]


def _try_place_path(
    height: int,
    width: int,
    palette_row: int,
    orientation: str,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    usable_top = 0 if palette_row != 0 else 1
    usable_bottom = height - 1 if palette_row != height - 1 else height - 2
    for _ in range(PLACEMENT_ATTEMPTS):
        if orientation == "horizontal":
            row_min = usable_top
            row_max = usable_bottom - 1
            if row_max < row_min:
                return None
            length = random.randint(2, max(2, min(width, 10)))
            row = random.randint(row_min, row_max)
            start_col = random.randint(0, width - length)
            cells = [(int(row), int(c)) for c in range(start_col, start_col + length)]
        else:
            max_len = usable_bottom - usable_top + 1
            if max_len < 2 or width < 2:
                return None
            length = random.randint(2, min(max_len, 10))
            col = random.randint(0, width - 2)
            start_row = random.randint(usable_top, usable_bottom - length + 1)
            cells = [(int(r), int(col)) for r in range(start_row, start_row + length)]

        shifted = _shifted_cells(orientation, cells)
        footprint = set(cells) | set(shifted)
        if any(r == palette_row for r, _ in cells):
            continue
        if any(r == palette_row for r, _ in shifted):
            continue
        if any(not (0 <= r < height and 0 <= c < width) for r, c in shifted):
            continue
        if footprint & occupied:
            continue
        return cells
    return None


def _build_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 4)
    palette_colors = random.sample(ALL_COLORS, 2 * pair_count)
    random.shuffle(palette_colors)
    pairs = [(palette_colors[2 * i], palette_colors[2 * i + 1]) for i in range(pair_count)]
    palette_top = random.choice([True, False])

    palette_width = len(_palette_sequence(pairs))
    width = random.randint(max(12, palette_width + 1), 24)
    height = random.randint(8, 22)
    if height < 4:
        return None

    grid = np.zeros((height, width), dtype=int)
    start = random.randint(0, width - palette_width)
    seq = _palette_sequence(pairs)
    row_index = 0 if palette_top else height - 1
    grid[row_index, start : start + len(seq)] = np.array(seq, dtype=int)

    path_count = random.randint(max(4, pair_count), 6)
    horizontal_count = random.randint(1, path_count - 1)
    vertical_count = path_count - horizontal_count
    orientations = ["horizontal"] * horizontal_count + ["vertical"] * vertical_count
    random.shuffle(orientations)

    path_colors = [random.choice(pair) for pair in pairs]
    while len(path_colors) < path_count:
        path_colors.append(random.choice(palette_colors))
    random.shuffle(path_colors)

    specs = list(zip(orientations, path_colors))
    specs.sort(key=lambda item: 0 if item[0] == "vertical" else 1)

    occupied: set[tuple[int, int]] = set()
    placed: list[tuple[list[tuple[int, int]], int]] = []
    for orientation, color in specs:
        cells = _try_place_path(height, width, row_index, orientation, occupied)
        if cells is None:
            return None
        shifted = _shifted_cells(orientation, cells)
        occupied.update(cells)
        occupied.update(shifted)
        placed.append((cells, int(color)))

    for cells, color in placed:
        for r, c in cells:
            grid[r, c] = int(color)

    other_edge = height - 1 if row_index == 0 else 0
    if _parse_palette_row(grid[other_edge]) is not None:
        return None

    candidates = _candidate_outputs(grid)
    if len(candidates) != 1:
        return None
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid palette-path ARC input")
