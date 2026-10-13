from __future__ import annotations

import random

import numpy as np


PANEL_COLS = 4
SEPARATOR_COLOR = 5
SYMMETRY_TYPES = ("both", "h", "v", "none")
MAX_GENERATION_ATTEMPTS = 300


def _panel_bounds(panel_row: int, panel_col: int, panel_size: int) -> tuple[slice, slice]:
    stride = panel_size + 1
    r0 = panel_row * stride
    c0 = panel_col * stride
    return slice(r0, r0 + panel_size), slice(c0, c0 + panel_size)


def _majority_color(block: np.ndarray) -> int:
    counts = np.bincount(block.ravel(), minlength=10)
    return int(np.argmax(counts))


def _classify_mask(mask: np.ndarray) -> str:
    has_h = bool(np.array_equal(mask, np.flipud(mask)))
    has_v = bool(np.array_equal(mask, np.fliplr(mask)))
    if has_h and has_v:
        return "both"
    if has_h:
        return "h"
    if has_v:
        return "v"
    return "none"


def _infer_layout(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    assert (w - (PANEL_COLS - 1)) % PANEL_COLS == 0, "grid width must encode four equal panels"
    panel_size = (w - (PANEL_COLS - 1)) // PANEL_COLS
    assert panel_size >= 2, "panel size must be at least 2"

    stride = panel_size + 1
    assert (h + 1) % stride == 0, "grid height must fit whole panel rows"
    panel_rows = (h + 1) // stride
    assert panel_rows >= 2, "input needs at least one data row plus one legend row"

    sep_cols = [panel_size + i * stride for i in range(PANEL_COLS - 1)]
    sep_rows = [panel_size + i * stride for i in range(panel_rows - 1)]
    sep_color = int(grid[0, sep_cols[0]])

    for c in sep_cols:
        assert np.all(grid[:, c] == sep_color), "separator columns must be solid"
    for r in sep_rows:
        assert np.all(grid[r, :] == sep_color), "separator rows must be solid"

    return int(panel_size), int(panel_rows), int(sep_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each panel by the legend color matched to its reflection symmetry, then remove the legend row.

    Input structure:
    - The grid is split into equal square panels by one-cell-wide grey separator lines.
    - Each non-legend panel contains a foreground pattern drawn on a background color.
    - The last panel row is a four-panel solid-color legend ordered left-to-right as both-axis, horizontal-only, vertical-only, and none.

    Rule:
    - Read the four legend colors from the last row of solid panels.
    - For every earlier panel, treat all non-background cells as the pattern mask and test whether that mask is symmetric under top-bottom reflection, left-right reflection, both, or neither.
    - Recolor every foreground cell in that panel with the legend color assigned to its symmetry class while leaving background cells unchanged.
    - Return only the rows above the legend, removing the separator directly above the legend as well.

    Key insight:
    - The final row is a lookup table: it maps visible symmetry classes to output colors, so the task is legend reading plus per-panel symmetry classification.
    """
    g = np.asarray(grid, dtype=int)
    panel_size, panel_rows, _sep_color = _infer_layout(g)
    data_rows = panel_rows - 1
    out_height = data_rows * panel_size + max(0, data_rows - 1)
    out = g[:out_height, :].copy()

    legend_row = panel_rows - 1
    legend_colors: list[int] = []
    for panel_col in range(PANEL_COLS):
        rs, cs = _panel_bounds(legend_row, panel_col, panel_size)
        legend_panel = g[rs, cs]
        assert np.all(legend_panel == legend_panel[0, 0]), "legend panels must be solid"
        legend_colors.append(int(legend_panel[0, 0]))

    legend_map = dict(zip(SYMMETRY_TYPES, legend_colors))
    for panel_row in range(data_rows):
        for panel_col in range(PANEL_COLS):
            rs, cs = _panel_bounds(panel_row, panel_col, panel_size)
            panel = g[rs, cs]
            bg = _majority_color(panel)
            mask = panel != bg
            sym_type = _classify_mask(mask)
            out_panel = out[rs, cs]
            out_panel[mask] = legend_map[sym_type]

    return out


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _has_isolated_cell(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        neighbors = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = int(r) + dr, int(c) + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                neighbors += 1
        if neighbors == 0:
            return True
    return False


def _is_good_mask(mask: np.ndarray) -> bool:
    size = int(mask.shape[0])
    total = size * size
    fg = int(mask.sum())
    if fg < max(4, size - 1) or fg > (total - 1) // 2:
        return False
    if bool(mask[0, 0] or mask[0, -1] or mask[-1, 0] or mask[-1, -1]):
        return False
    if size > 2 and not bool(mask[1:-1, 1:-1].any()):
        return False
    if int(np.count_nonzero(mask.any(axis=0))) < 2 or int(np.count_nonzero(mask.any(axis=1))) < 2:
        return False
    if _count_components(mask) > 3:
        return False
    if _has_isolated_cell(mask):
        return False
    return True


def _orbits_for_type(size: int, sym_type: str) -> list[list[tuple[int, int]]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[list[tuple[int, int]]] = []
    for r in range(size):
        for c in range(size):
            if sym_type == "both":
                pts = {
                    (r, c),
                    (size - 1 - r, c),
                    (r, size - 1 - c),
                    (size - 1 - r, size - 1 - c),
                }
            elif sym_type == "h":
                pts = {(r, c), (size - 1 - r, c)}
            elif sym_type == "v":
                pts = {(r, c), (r, size - 1 - c)}
            else:
                pts = {(r, c)}
            key = tuple(sorted(pts))
            if key not in seen:
                seen.add(key)
                orbits.append(list(key))
    return orbits


def _sample_orbit_mask(size: int, sym_type: str) -> np.ndarray:
    min_fg = max(4, size - 1)
    max_fg = (size * size - 1) // 2
    orbits = _orbits_for_type(size, sym_type)
    interior = [
        i
        for i, orbit in enumerate(orbits)
        if any(0 < r < size - 1 and 0 < c < size - 1 for r, c in orbit)
    ]

    for _ in range(250):
        mask = np.zeros((size, size), dtype=bool)
        target = random.randint(min_fg, max_fg)
        if interior:
            for r, c in orbits[random.choice(interior)]:
                mask[r, c] = True

        include_prob = random.uniform(0.25, 0.6)
        order = list(range(len(orbits)))
        random.shuffle(order)
        for idx in order:
            orbit = orbits[idx]
            if any(mask[r, c] for r, c in orbit):
                continue
            orbit_size = len(orbit)
            current = int(mask.sum())
            remaining = target - current
            if current >= target and random.random() < 0.8:
                continue
            if current + orbit_size > max_fg:
                continue
            threshold = include_prob + (0.2 if orbit_size <= max(1, remaining) else 0.0)
            if random.random() < min(threshold, 0.95):
                for r, c in orbit:
                    mask[r, c] = True

        if int(mask.sum()) < min_fg:
            for idx in sorted(range(len(orbits)), key=lambda i: len(orbits[i])):
                orbit = orbits[idx]
                if any(mask[r, c] for r, c in orbit):
                    continue
                if int(mask.sum()) + len(orbit) > max_fg:
                    continue
                for r, c in orbit:
                    mask[r, c] = True
                if int(mask.sum()) >= min_fg:
                    break

        if _is_good_mask(mask) and _classify_mask(mask) == sym_type:
            return mask

    raise RuntimeError(f"failed to sample {sym_type} mask of size {size}")


def _make_none_mask(size: int) -> np.ndarray:
    min_fg = max(4, size - 1)
    max_fg = (size * size - 1) // 2
    corners = {(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)}
    candidates = [(r, c) for r in range(size) for c in range(size) if (r, c) not in corners]
    interior = [(r, c) for r, c in candidates if 0 < r < size - 1 and 0 < c < size - 1]

    for _ in range(400):
        target = random.randint(min_fg, max_fg)
        cells = {random.choice(interior)}
        while len(cells) < target:
            if random.random() < 0.85:
                base_r, base_c = random.choice(tuple(cells))
                neighbors = [
                    (base_r + dr, base_c + dc)
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    if 0 <= base_r + dr < size and 0 <= base_c + dc < size
                ]
                neighbors = [pos for pos in neighbors if pos not in corners]
                random.shuffle(neighbors)
                if neighbors:
                    cells.add(neighbors[0])
                    continue

            frontier = [
                pos
                for pos in candidates
                if pos not in cells
                and any(abs(pos[0] - r) + abs(pos[1] - c) == 1 for r, c in cells)
            ]
            if not frontier:
                frontier = [pos for pos in candidates if pos not in cells]
            cells.add(random.choice(frontier))

        mask = np.zeros((size, size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if (
            _is_good_mask(mask)
            and _classify_mask(mask) == "none"
            and not bool(np.array_equal(mask, np.rot90(mask, 2)))
        ):
            return mask

    raise RuntimeError(f"failed to sample none mask of size {size}")


def _make_mask(sym_type: str, size: int) -> np.ndarray:
    if sym_type == "none":
        return _make_none_mask(size)
    return _sample_orbit_mask(size, sym_type)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        panel_size = random.randint(4, 6)
        data_rows = random.randint(1, 3)
        total_rows = data_rows + 1
        h = total_rows * panel_size + (total_rows - 1)
        w = PANEL_COLS * panel_size + (PANEL_COLS - 1)

        available = [0, 1, 2, 3, 4, 6, 7, 8, 9]
        bg = random.choice(available)
        palette = [color for color in available if color != bg]
        random.shuffle(palette)
        legend_colors = palette[:4]
        pattern_colors = palette[4:]

        grid = np.full((h, w), bg, dtype=int)
        stride = panel_size + 1
        for k in range(total_rows - 1):
            grid[panel_size + k * stride, :] = SEPARATOR_COLOR
        for k in range(PANEL_COLS - 1):
            grid[:, panel_size + k * stride] = SEPARATOR_COLOR

        kinds = list(SYMMETRY_TYPES) * data_rows
        random.shuffle(kinds)
        legend_map = dict(zip(SYMMETRY_TYPES, legend_colors))
        masks: list[np.ndarray] = []
        colors: list[int] = []
        for kind in kinds:
            masks.append(_make_mask(kind, panel_size))
            colors.append(random.choice(pattern_colors))

        index = 0
        for panel_row in range(data_rows):
            for panel_col in range(PANEL_COLS):
                rs, cs = _panel_bounds(panel_row, panel_col, panel_size)
                panel = np.full((panel_size, panel_size), bg, dtype=int)
                panel[masks[index]] = colors[index]
                grid[rs, cs] = panel
                index += 1

        for panel_col, color in enumerate(legend_colors):
            rs, cs = _panel_bounds(data_rows, panel_col, panel_size)
            grid[rs, cs] = color

        out_height = data_rows * panel_size + max(0, data_rows - 1)
        expected = grid[:out_height, :].copy()
        index = 0
        for panel_row in range(data_rows):
            for panel_col in range(PANEL_COLS):
                rs, cs = _panel_bounds(panel_row, panel_col, panel_size)
                expected_panel = expected[rs, cs]
                expected_panel[masks[index]] = legend_map[kinds[index]]
                index += 1

        try:
            if np.array_equal(main(grid), expected):
                return grid
        except AssertionError:
            pass

    raise RuntimeError("failed to generate a valid panel-symmetry puzzle")
