from __future__ import annotations

import random

import numpy as np


BG = 0
GEN_RETRIES = 500
GEN_LAYOUTS = ((2, 2), (2, 3), (3, 2), (3, 3))
GEN_CELL_SIZE_RANGE = (3, 7)
GEN_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _segments_from_lines(length: int, lines: list[int]) -> list[tuple[int, int]]:
    line_set = set(int(x) for x in lines)
    segments: list[tuple[int, int]] = []
    start = 0
    while start < length:
        while start < length and start in line_set:
            start += 1
        if start >= length:
            break
        end = start
        while end < length and end not in line_set:
            end += 1
        segments.append((int(start), int(end)))
        start = end
    return segments


def _background_color(grid: np.ndarray) -> int:
    if np.any(grid == BG):
        return BG
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(counts.argmax())])


def _find_structure(
    grid: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        rows = [r for r in range(h) if np.all(g[r, :] == color)]
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if not rows or not cols:
            continue

        row_segments = _segments_from_lines(h, rows)
        col_segments = _segments_from_lines(w, cols)
        if len(row_segments) < 2 or len(col_segments) < 2:
            continue

        row_sizes = {r1 - r0 for r0, r1 in row_segments}
        col_sizes = {c1 - c0 for c0, c1 in col_segments}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue

        ok = True
        for r0, r1 in row_segments:
            for c0, c1 in col_segments:
                patch = g[r0:r1, c0:c1]
                inside = {
                    int(v) for v in np.unique(patch) if int(v) not in (bg, color)
                }
                if len(inside) != 1:
                    ok = False
                    break
                cell_color = next(iter(inside))
                mask = patch == cell_color
                if not mask.any() or bool(mask.all()):
                    ok = False
                    break
            if not ok:
                break

        if ok:
            candidates.append((int(color), row_segments, col_segments))

    assert len(candidates) == 1, "expected a unique separator structure"
    separator, row_segments, col_segments = candidates[0]
    return bg, separator, row_segments, col_segments


def _extract_cells(
    grid: np.ndarray,
    bg: int,
    separator: int,
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    cells: list[dict[str, object]] = []
    for row_idx, (r0, r1) in enumerate(row_segments):
        for col_idx, (c0, c1) in enumerate(col_segments):
            patch = g[r0:r1, c0:c1]
            colors = {int(v) for v in np.unique(patch) if int(v) not in (bg, separator)}
            assert len(colors) == 1
            color = next(iter(colors))
            mask = patch == color
            assert mask.any() and not bool(mask.all())
            cells.append(
                {
                    "index": (int(row_idx), int(col_idx)),
                    "rows": (int(r0), int(r1)),
                    "cols": (int(c0), int(c1)),
                    "color": int(color),
                    "mask": mask.copy(),
                }
            )
    return cells


def _compatible(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    return bool(np.array_equal(np.rot90(mask_a, 2), ~mask_b))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge complementary same-color cell patterns inside a separator-partitioned grid.

    Input structure
    - Black background.
    - One non-background color forms complete separator rows and columns, splitting
      the canvas into equal rectangular cells.
    - Each cell contains exactly one monochrome pattern using a color distinct from
      the background and separators.

    Rule
    - Within each color group, find cell pairs whose masks become exact complements
      after a 180-degree rotation.
    - For each such pair, fill the lexicographically smaller cell with a solid
      rectangle of that color and clear the other cell to background.
    - Leave non-pairing distractors and separator lines unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg, separator, row_segments, col_segments = _find_structure(g)
    cells = _extract_cells(g, bg, separator, row_segments, col_segments)

    by_color: dict[int, list[dict[str, object]]] = {}
    for cell in cells:
        by_color.setdefault(int(cell["color"]), []).append(cell)

    out = g.copy()
    for color, items in by_color.items():
        neighbors: dict[int, list[int]] = {i: [] for i in range(len(items))}
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if _compatible(items[i]["mask"], items[j]["mask"]):
                    neighbors[i].append(j)
                    neighbors[j].append(i)

        assert all(len(adj) <= 1 for adj in neighbors.values()), "ambiguous pairing"
        used: set[int] = set()
        for i, adj in neighbors.items():
            if not adj or i in used:
                continue
            j = adj[0]
            if j in used:
                continue
            used.add(i)
            used.add(j)

            first, second = sorted(
                (items[i], items[j]), key=lambda cell: tuple(cell["index"])
            )
            fr0, fr1 = first["rows"]
            fc0, fc1 = first["cols"]
            sr0, sr1 = second["rows"]
            sc0, sc1 = second["cols"]
            out[int(fr0) : int(fr1), int(fc0) : int(fc1)] = int(color)
            out[int(sr0) : int(sr1), int(sc0) : int(sc1)] = int(bg)

    return out


def _axis_segments(
    count: int, cell_size: int, bordered: bool
) -> tuple[list[tuple[int, int]], int]:
    segments: list[tuple[int, int]] = []
    pos = 1 if bordered else 0
    for idx in range(count):
        segments.append((int(pos), int(pos + cell_size)))
        pos += cell_size
        if idx != count - 1 or bordered:
            pos += 1
    return segments, int(pos)


def _paint_separators(
    grid: np.ndarray,
    separator: int,
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
) -> None:
    row_mask = np.zeros(grid.shape[0], dtype=bool)
    col_mask = np.zeros(grid.shape[1], dtype=bool)
    for r0, r1 in row_segments:
        row_mask[r0:r1] = True
    for c0, c1 in col_segments:
        col_mask[c0:c1] = True
    grid[~row_mask, :] = int(separator)
    grid[:, ~col_mask] = int(separator)


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    seen: set[tuple[int, int]] = set()
    cells: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                pt = (int(nr), int(nc))
                if pt not in seen:
                    seen.add(pt)
                    cells.append(pt)
    return cells


def _random_connected_mask(height: int, width: int) -> np.ndarray | None:
    total = height * width
    min_area = max(2, total // 4)
    max_area = min(total - 2, total - min_area)
    if min_area > max_area:
        return None

    target = random.randint(min_area, max_area)
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True

    while int(mask.sum()) < target:
        choices = _frontier(mask)
        if not choices:
            return None
        mask[random.choice(choices)] = True

    if not mask.any() or bool(mask.all()):
        return None
    return mask


def _random_pair_masks(height: int, width: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(60):
        first = _random_connected_mask(height, width)
        if first is None:
            continue
        second = ~np.rot90(first, 2)
        if second.any() and not bool(second.all()):
            return first, second
    return None


def _random_distractor_mask(height: int, width: int) -> np.ndarray | None:
    for _ in range(60):
        mask = _random_connected_mask(height, width)
        if mask is not None:
            return mask
    return None


def _expected_output(
    grid: np.ndarray,
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
    pairs: list[tuple[tuple[int, int], tuple[int, int], int]],
) -> np.ndarray:
    out = grid.copy()
    for left, right, color in pairs:
        keep, clear = sorted((left, right))
        kr0, kr1 = row_segments[keep[0]]
        kc0, kc1 = col_segments[keep[1]]
        cr0, cr1 = row_segments[clear[0]]
        cc0, cc1 = col_segments[clear[1]]
        out[kr0:kr1, kc0:kc1] = int(color)
        out[cr0:cr1, cc0:cc1] = BG
    return out


def _attempt_generate() -> np.ndarray | None:
    rows, cols = random.choice(GEN_LAYOUTS)
    cell_h = random.randint(*GEN_CELL_SIZE_RANGE)
    cell_w = random.randint(*GEN_CELL_SIZE_RANGE)
    row_bordered = bool(random.getrandbits(1))
    col_bordered = bool(random.getrandbits(1))
    row_segments, height = _axis_segments(rows, cell_h, row_bordered)
    col_segments, width = _axis_segments(cols, cell_w, col_bordered)

    total_cells = rows * cols
    min_pairs = max(1, total_cells - 8)
    max_pairs = total_cells // 2
    pair_count = random.randint(min_pairs, max_pairs)
    distractor_count = total_cells - 2 * pair_count

    colors = random.sample(GEN_COLORS, 1 + pair_count + distractor_count)
    separator = int(colors[0])
    pair_colors = [int(c) for c in colors[1 : 1 + pair_count]]
    distractor_colors = [int(c) for c in colors[1 + pair_count :]]

    grid = np.full((height, width), BG, dtype=int)
    _paint_separators(grid, separator, row_segments, col_segments)

    cells = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(cells)
    cell_masks: dict[tuple[int, int], tuple[int, np.ndarray]] = {}
    pairs: list[tuple[tuple[int, int], tuple[int, int], int]] = []

    for color in pair_colors:
        first_cell = cells.pop()
        second_cell = cells.pop()
        pair_masks = _random_pair_masks(cell_h, cell_w)
        if pair_masks is None:
            return None
        mask_a, mask_b = pair_masks
        if random.random() < 0.5:
            mask_a, mask_b = mask_b, mask_a
        cell_masks[first_cell] = (int(color), mask_a)
        cell_masks[second_cell] = (int(color), mask_b)
        pairs.append((first_cell, second_cell, int(color)))

    for color, cell in zip(distractor_colors, cells, strict=False):
        mask = _random_distractor_mask(cell_h, cell_w)
        if mask is None:
            return None
        cell_masks[cell] = (int(color), mask)

    for (row_idx, col_idx), (color, mask) in cell_masks.items():
        r0, r1 = row_segments[row_idx]
        c0, c1 = col_segments[col_idx]
        patch = grid[r0:r1, c0:c1]
        patch[mask] = int(color)

    expected = _expected_output(grid, row_segments, col_segments, pairs)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
