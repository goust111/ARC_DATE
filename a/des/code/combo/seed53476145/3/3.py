from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 9
PALETTE = tuple(range(1, 9))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400
PANEL_HEIGHT_RANGE = (7, 11)


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

    return count


def _separator_columns(grid: np.ndarray) -> tuple[int, list[int]]:
    h, w = grid.shape
    solid_cols = [
        c
        for c in range(w)
        if int(grid[0, c]) != BG and bool(np.all(grid[:, c] == grid[0, c]))
    ]
    assert 2 <= len(solid_cols) <= 4, "expected 3-5 panels split by solid separator columns"

    sep_color = int(grid[0, solid_cols[0]])
    assert all(int(grid[0, c]) == sep_color for c in solid_cols), "separator color must be unique"
    assert all(solid_cols[i] + 1 < solid_cols[i + 1] for i in range(len(solid_cols) - 1))
    assert h >= 1
    return sep_color, solid_cols


def _split_panels(grid: np.ndarray, sep_cols: list[int]) -> list[np.ndarray]:
    panels = []
    widths = []
    start = 0
    for col in sep_cols:
        panels.append(grid[:, start:col].copy())
        widths.append(col - start)
        start = col + 1
    panels.append(grid[:, start:].copy())
    widths.append(grid.shape[1] - start)

    assert 3 <= len(panels) <= 5
    assert all(width > 0 for width in widths)
    assert len(set(widths)) == 1, "panels must have equal width"
    return panels


def _assemble_panels(panels: list[np.ndarray], sep_color: int) -> np.ndarray:
    assert panels
    panel_h, panel_w = panels[0].shape
    out_w = len(panels) * panel_w + (len(panels) - 1)
    out = np.full((panel_h, out_w), BG, dtype=int)

    col = 0
    for i, panel in enumerate(panels):
        assert panel.shape == (panel_h, panel_w)
        out[:, col : col + panel_w] = panel
        col += panel_w
        if i + 1 < len(panels):
            out[:, col] = sep_color
            col += 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder equal-width panels by foreground connectivity and flip only the separated ones.

    Input structure:
    - Background is black.
    - One-cell-wide full-height separator columns split the grid into 3-5 equal-width panels.
    - Inside each panel, non-black pixels form either one global 4-connected body or multiple
      mutually separated bodies.

    Rule:
    - Collect connected panels from left to right.
    - Then collect separated panels from right to left.
    - Connected panels stay unchanged.
    - Separated panels are vertically flipped before being written.
    - Keep the separator columns between output panels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_color, sep_cols = _separator_columns(g)
    panels = _split_panels(g, sep_cols)

    connected: list[np.ndarray] = []
    separated: list[np.ndarray] = []
    for panel in panels:
        n_components = _component_count(panel != BG)
        assert n_components >= 1, "each panel must contain foreground pixels"
        if n_components == 1:
            connected.append(panel)
        else:
            separated.append(np.flipud(panel))

    ordered = connected + separated[::-1]
    out = _assemble_panels(ordered, sep_color)
    assert out.shape == g.shape
    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _grow_connected_cells(height: int, width: int, area: int) -> set[tuple[int, int]]:
    assert 1 <= area <= height * width
    for _ in range(80):
        cells = {(random.randrange(height), random.randrange(width))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) == area:
            return _normalize_cells(cells)
    raise RuntimeError("failed to grow a connected shape")


def _partition_mask(mask: np.ndarray, parts: int) -> list[list[tuple[int, int]]] | None:
    cells = [(r, c) for r, c in zip(*np.where(mask), strict=False)]
    if len(cells) < parts * 2:
        return None

    for _ in range(60):
        seeds = random.sample(cells, parts)
        owners = {seed: idx for idx, seed in enumerate(seeds)}
        remaining = set(cells) - set(seeds)

        while remaining:
            candidates = []
            for cell in remaining:
                labels = {
                    owners[(cell[0] + dr, cell[1] + dc)]
                    for dr, dc in NEIGHBORS4
                    if (cell[0] + dr, cell[1] + dc) in owners
                }
                if labels:
                    candidates.append((cell, tuple(labels)))
            if not candidates:
                break

            cell, labels = random.choice(candidates)
            owners[cell] = random.choice(labels)
            remaining.remove(cell)

        if remaining:
            continue

        groups = [[] for _ in range(parts)]
        for cell, label in owners.items():
            groups[label].append(cell)
        if min(len(group) for group in groups) >= 2:
            return groups

    return None


def _make_connected_panel(height: int, width: int) -> np.ndarray:
    inner_h = height - 2
    inner_w = width - 2
    assert inner_h >= 3 and inner_w >= 3

    for _ in range(120):
        max_parts = min(4, max(2, inner_h * inner_w // 4))
        parts = random.randint(2, max_parts)
        min_area = max(2 * parts, 6)
        max_area = min(inner_h * inner_w - 1, (3 * inner_h * inner_w) // 4)
        if max_area < min_area:
            continue

        area = random.randint(min_area, max_area)
        cells = _grow_connected_cells(inner_h, inner_w, area)
        mask = np.zeros((inner_h, inner_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        groups = _partition_mask(mask, parts)
        if groups is None:
            continue

        panel = np.zeros((height, width), dtype=int)
        colors = random.sample(PALETTE, parts)
        for color, group in zip(colors, groups, strict=False):
            for r, c in group:
                panel[r + 1, c + 1] = color

        if _component_count(panel != BG) == 1:
            return panel

    raise RuntimeError("failed to generate a connected panel")


def _random_object_shape(inner_h: int, inner_w: int) -> set[tuple[int, int]]:
    box_h = random.randint(1, min(3, inner_h))
    box_w = random.randint(1, min(3, inner_w))
    max_area = min(box_h * box_w, 5)
    min_area = 2 if max_area >= 2 else 1
    area = random.randint(min_area, max_area)
    return _grow_connected_cells(box_h, box_w, area)


def _can_place_shape(
    shape: set[tuple[int, int]],
    top: int,
    left: int,
    occupied: set[tuple[int, int]],
    inner_h: int,
    inner_w: int,
) -> bool:
    abs_cells = [(top + r, left + c) for r, c in shape]
    for r, c in abs_cells:
        if not (0 <= r < inner_h and 0 <= c < inner_w):
            return False
        if (r, c) in occupied:
            return False
        for dr, dc in NEIGHBORS4:
            if (r + dr, c + dc) in occupied:
                return False
    return True


def _make_separated_panel(height: int, width: int) -> np.ndarray:
    inner_h = height - 2
    inner_w = width - 2
    assert inner_h >= 3 and inner_w >= 3

    for _ in range(160):
        panel = np.zeros((height, width), dtype=int)
        occupied: set[tuple[int, int]] = set()

        max_objects = 4 if inner_h * inner_w >= 24 and inner_w >= 4 else 3
        objects = random.randint(2, max_objects)
        colors = random.sample(PALETTE, objects)
        success = True

        for color in colors:
            shape = _random_object_shape(inner_h, inner_w)
            shape_h = max(r for r, _ in shape) + 1
            shape_w = max(c for _, c in shape) + 1

            positions = [
                (r, c)
                for r in range(inner_h - shape_h + 1)
                for c in range(inner_w - shape_w + 1)
            ]
            random.shuffle(positions)

            placed = False
            for top, left in positions:
                if not _can_place_shape(shape, top, left, occupied, inner_h, inner_w):
                    continue
                for r, c in shape:
                    rr = top + r
                    cc = left + c
                    panel[rr + 1, cc + 1] = color
                    occupied.add((rr, cc))
                placed = True
                break

            if not placed:
                success = False
                break

        if success and _component_count(panel != BG) == objects:
            return panel

    raise RuntimeError("failed to generate a separated panel")


def _sample_layout() -> tuple[int, int, int]:
    panel_count = random.randint(3, 5)
    width_options = [
        panel_width
        for panel_width in range(5, 10)
        if panel_count * panel_width + (panel_count - 1) <= 30
    ]
    assert width_options
    panel_height = random.randint(*PANEL_HEIGHT_RANGE)
    panel_width = random.choice(width_options)
    return panel_count, panel_height, panel_width


def _expected_output(panels: list[np.ndarray], is_connected: list[bool]) -> np.ndarray:
    connected = [panel.copy() for panel, flag in zip(panels, is_connected, strict=False) if flag]
    separated = [
        np.flipud(panel)
        for panel, flag in zip(reversed(panels), reversed(is_connected), strict=False)
        if not flag
    ]
    return _assemble_panels(connected + separated, SEP)


def _gen_attempt() -> np.ndarray | None:
    panel_count, panel_height, panel_width = _sample_layout()
    connected_count = random.randint(1, panel_count - 1)
    kinds = [True] * connected_count + [False] * (panel_count - connected_count)
    random.shuffle(kinds)

    panels = []
    for is_connected in kinds:
        panel = (
            _make_connected_panel(panel_height, panel_width)
            if is_connected
            else _make_separated_panel(panel_height, panel_width)
        )
        observed = _component_count(panel != BG) == 1
        if observed != is_connected:
            return None
        panels.append(panel)

    inp = _assemble_panels(panels, SEP)
    expected = _expected_output(panels, kinds)
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
