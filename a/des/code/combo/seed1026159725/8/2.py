import random
from collections import Counter

import numpy as np


MAX_GENERATION_ATTEMPTS = 300
PALETTE = tuple(range(10))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_separator_layout(grid: np.ndarray) -> tuple[int, int, list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _mode_color(g)

    candidate_colors = set()
    row_color_counts = Counter(
        int(g[r, 0])
        for r in range(h)
        if np.all(g[r] == g[r, 0]) and int(g[r, 0]) != bg
    )
    col_color_counts = Counter(
        int(g[0, c])
        for c in range(w)
        if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg
    )
    candidate_colors.update(set(row_color_counts))
    candidate_colors.intersection_update(set(col_color_counts))

    for sep in sorted(candidate_colors):
        rows = [r for r in range(h) if np.all(g[r] == sep)]
        cols = [c for c in range(w) if np.all(g[:, c] == sep)]
        if len(rows) != 2 or len(cols) != 2:
            continue
        panel_heights = [rows[0], rows[1] - rows[0] - 1, h - rows[1] - 1]
        panel_widths = [cols[0], cols[1] - cols[0] - 1, w - cols[1] - 1]
        if min(panel_heights) <= 0 or min(panel_widths) <= 0:
            continue
        if len(set(panel_heights)) == 1 and len(set(panel_widths)) == 1:
            return bg, int(sep), rows, cols

    raise AssertionError("failed to infer 3x3 separator layout")


def _panel_slices(rows: list[int], cols: list[int], shape: tuple[int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = shape
    row_slices = [(0, rows[0]), (rows[0] + 1, rows[1]), (rows[1] + 1, h)]
    col_slices = [(0, cols[0]), (cols[0] + 1, cols[1]), (cols[1] + 1, w)]
    return row_slices, col_slices


def _signature_from_panel(panel: np.ndarray, bg: int, sep: int) -> tuple[tuple[int, int, int], ...]:
    mask = (panel != bg) & (panel != sep)
    coords = np.argwhere(mask)
    assert coords.size > 0, "each panel must contain a pattern"
    r0, c0 = coords.min(axis=0)
    items = []
    for r, c in coords:
        items.append((int(r - r0), int(c - c0), int(panel[int(r), int(c)])))
    return tuple(sorted(items))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the globally most frequent panel pattern in a 3x3 separator grid.

    Each panel contributes one translation-normalized colored pattern signature.
    The unique most frequent signature survives; all other panel patterns are
    erased to the background color while the separator lines remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg, sep, rows, cols = _find_separator_layout(g)
    row_slices, col_slices = _panel_slices(rows, cols, g.shape)

    panels = []
    signatures = []
    for rs in row_slices:
        for cs in col_slices:
            panel = g[rs[0] : rs[1], cs[0] : cs[1]]
            sig = _signature_from_panel(panel, bg, sep)
            panels.append((rs, cs))
            signatures.append(sig)

    counts = Counter(signatures)
    top_two = counts.most_common(2)
    assert top_two and (len(top_two) == 1 or top_two[0][1] > top_two[1][1]), "expected a unique most frequent signature"
    winner = top_two[0][0]

    out = g.copy()
    for (rs, cs), sig in zip(panels, signatures):
        if sig == winner:
            continue
        panel = out[rs[0] : rs[1], cs[0] : cs[1]]
        panel[(panel != bg) & (panel != sep)] = bg

    return out


def _normalize_coords(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return sorted((r - min_r, c - min_c) for r, c in coords)


def _random_connected_shape(max_h: int, max_w: int) -> list[tuple[int, int]]:
    for _ in range(80):
        box_h = random.randint(2, max(2, min(max_h, 4)))
        box_w = random.randint(2, max(2, min(max_w, 4)))
        area = random.randint(3, min(box_h * box_w, 6))

        cells = {(random.randrange(box_h), random.randrange(box_w))}
        while len(cells) < area:
            r, c = random.choice(tuple(cells))
            dr, dc = random.choice(NEIGHBORS4)
            nr, nc = r + dr, c + dc
            if 0 <= nr < box_h and 0 <= nc < box_w:
                cells.add((nr, nc))

        coords = _normalize_coords(cells)
        height = max(r for r, _ in coords) + 1
        width = max(c for _, c in coords) + 1
        if height <= max_h and width <= max_w and (height > 1 or width > 1):
            return coords

    raise RuntimeError("failed to build a connected shape")


def _make_signature(existing: set[tuple[tuple[int, int, int], ...]], panel_h: int, panel_w: int, colors: list[int]) -> tuple[tuple[int, int, int], ...]:
    for _ in range(200):
        coords = _random_connected_shape(panel_h - 1, panel_w - 1)
        color = random.choice(colors)
        sig = tuple((r, c, color) for r, c in coords)
        if sig not in existing:
            return sig
    raise RuntimeError("failed to create a unique signature")


def _random_other_counts(total: int, majority_count: int) -> list[int]:
    for _ in range(200):
        num_parts = random.randint(2, min(5, total))
        remaining = total
        parts = []
        valid = True
        for i in range(num_parts):
            min_needed = num_parts - i - 1
            hi = min(majority_count - 1, remaining - min_needed)
            if hi < 1:
                valid = False
                break
            part = random.randint(1, hi)
            parts.append(part)
            remaining -= part
        if valid and remaining == 0:
            random.shuffle(parts)
            return parts
    raise RuntimeError("failed to partition distractor counts")


def _place_signature(panel: np.ndarray, signature: tuple[tuple[int, int, int], ...]) -> None:
    max_r = max(r for r, _, _ in signature) + 1
    max_c = max(c for _, c, _ in signature) + 1
    top = random.randint(0, panel.shape[0] - max_r)
    left = random.randint(0, panel.shape[1] - max_c)
    for r, c, color in signature:
        panel[top + r, left + c] = color


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, sep, rows, cols = _find_separator_layout(g)
    row_slices, col_slices = _panel_slices(rows, cols, g.shape)
    panels = []
    signatures = []
    for rs in row_slices:
        for cs in col_slices:
            panels.append((rs, cs))
            signatures.append(_signature_from_panel(g[rs[0] : rs[1], cs[0] : cs[1]], bg, sep))

    winner = Counter(signatures).most_common(1)[0][0]
    out = g.copy()
    for (rs, cs), sig in zip(panels, signatures):
        if sig == winner:
            continue
        region = out[rs[0] : rs[1], cs[0] : cs[1]]
        region[(region != bg) & (region != sep)] = bg
    return out


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(5, 8)
    panel_w = random.randint(5, 8)
    grid_h = panel_h * 3 + 2
    grid_w = panel_w * 3 + 2

    bg, sep = random.sample(PALETTE, 2)
    pattern_colors = random.sample([c for c in PALETTE if c not in {bg, sep}], random.randint(3, 6))

    majority_count = random.randint(3, 5)
    other_counts = _random_other_counts(9 - majority_count, majority_count)

    existing = set()
    signatures = []
    for _ in range(1 + len(other_counts)):
        sig = _make_signature(existing, panel_h, panel_w, pattern_colors)
        existing.add(sig)
        signatures.append(sig)

    assignments = [signatures[0]] * majority_count
    for sig, count in zip(signatures[1:], other_counts):
        assignments.extend([sig] * count)
    random.shuffle(assignments)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    sep_rows = [panel_h, panel_h * 2 + 1]
    sep_cols = [panel_w, panel_w * 2 + 1]
    grid[sep_rows[0], :] = sep
    grid[sep_rows[1], :] = sep
    grid[:, sep_cols[0]] = sep
    grid[:, sep_cols[1]] = sep

    row_slices, col_slices = _panel_slices(sep_rows, sep_cols, grid.shape)
    idx = 0
    for rs in row_slices:
        for cs in col_slices:
            _place_signature(grid[rs[0] : rs[1], cs[0] : cs[1]], assignments[idx])
            idx += 1

    try:
        bg2, sep2, rows2, cols2 = _find_separator_layout(grid)
        row_slices2, col_slices2 = _panel_slices(rows2, cols2, grid.shape)
        sigs = [
            _signature_from_panel(grid[rs[0] : rs[1], cs[0] : cs[1]], bg2, sep2)
            for rs in row_slices2
            for cs in col_slices2
        ]
        counts = Counter(sigs)
        top_two = counts.most_common(2)
        if not top_two or (len(top_two) > 1 and top_two[0][1] <= top_two[1][1]):
            return None
        if top_two[0][1] != majority_count:
            return None
        expected = _expected_output(grid)
        actual = main(grid)
        if expected.shape != actual.shape or not np.array_equal(expected, actual):
            return None
        return grid
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
