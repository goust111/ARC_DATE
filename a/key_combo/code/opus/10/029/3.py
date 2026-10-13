from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
MARKER = 2
PANEL = 8
MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_panels(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    panels = []

    for r in range(h):
        for c in range(w):
            if grid[r, c] == BG or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            r0 = r1 = r
            c0 = c1 = c
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] != BG:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            crop = grid[r0 : r1 + 1, c0 : c1 + 1]
            colors = set(map(int, np.unique(crop)))
            if not colors.issubset({BG, MARKER, PANEL}):
                continue
            if MARKER not in colors and PANEL not in colors:
                continue
            panels.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "coords": coords,
                    "crop": crop.copy(),
                    "marker_count": int(np.count_nonzero(crop == MARKER)),
                    "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                }
            )

    return panels


def _shift_selected_rows_right(crop: np.ndarray) -> np.ndarray:
    out = crop.copy()
    h, w = crop.shape
    for r in range(h):
        distance = h - 1 - r
        if distance >= 2 and distance % 2 == 0:
            out[r, 0] = BG
            out[r, 1:w] = crop[r, 0 : w - 1]
    return out


def _shift_selected_rows_left(crop: np.ndarray) -> np.ndarray:
    out = crop.copy()
    h, w = crop.shape
    for r in range(h):
        distance = h - 1 - r
        if distance >= 2 and distance % 2 == 0:
            out[r, 0 : w - 1] = crop[r, 1:w]
            out[r, w - 1] = BG
    return out


def _shift_top_anchored_right(crop: np.ndarray) -> np.ndarray:
    out = crop.copy()
    h, w = crop.shape
    for r in range(0, h, 2):
        out[r, 0] = BG
        out[r, 1:w] = crop[r, 0 : w - 1]
    return out


def _shift_wrong_bottom_phase(crop: np.ndarray) -> np.ndarray:
    out = crop.copy()
    h, w = crop.shape
    for r in range(h):
        distance = h - 1 - r
        if distance % 2 == 1:
            out[r, 0] = BG
            out[r, 1:w] = crop[r, 0 : w - 1]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the uniquely most-marked panel, then right-shift alternating rows anchored from the bottom.

    Input structure:
    - The grid uses black background, with 3-4 separated rectangular panels made of teal cells and embedded red marker pixels.
    - Each panel occupies one axis-aligned bounding box and every panel contains at least one red marker.
    - Panels do not touch each other, so each panel appears as one 4-connected non-black component.

    Rule:
    - Find every non-black panel and count the red markers inside its bounding box.
    - Select the unique panel with the largest red-marker count and crop its exact bounding box.
    - On that crop, keep the bottom row and the row just above it fixed.
    - Moving upward, shift every other remaining row one cell to the right; dropped right-edge pixels disappear and the new left edge becomes black.

    Key insight:
    - The red pixels first decide which panel is chosen, then those same red pixels are shifted together with their teal row content under a bottom-anchored row-selection pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    panels = _find_panels(g)
    assert 3 <= len(panels) <= 4

    best = max(panel["marker_count"] for panel in panels)
    winners = [panel for panel in panels if panel["marker_count"] == best]
    assert len(winners) == 1
    return _shift_selected_rows_right(winners[0]["crop"])


def _random_split(total: int, margin: int, gap: int, min_a: int, min_b: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    usable = total - 2 * margin - gap
    if usable < min_a + min_b:
        return None
    a = random.randint(min_a, usable - min_b)
    b = usable - a
    return (margin, margin + a), (margin + a + gap, total - margin)


def _choose_counts(panel_count: int, winner: int) -> list[int]:
    winner_count = random.randint(3, 5)
    runner_up = random.randint(1, winner_count - 1)
    loser_indices = [idx for idx in range(panel_count) if idx != winner]
    runner_idx = random.choice(loser_indices)
    counts = []
    for idx in range(panel_count):
        if idx == winner:
            counts.append(winner_count)
        elif idx == runner_idx:
            counts.append(runner_up)
        else:
            counts.append(random.randint(1, winner_count - 1))
    return counts


def _choose_winner_markers(height: int, width: int, count: int) -> list[tuple[int, int]]:
    shift_rows = [r for r in range(height) if (height - 1 - r) >= 2 and (height - 1 - r) % 2 == 0]
    fixed_rows = [r for r in range(height) if r not in shift_rows]
    assert shift_rows and fixed_rows and count >= 3

    cells = set()
    edge_cols = [0, width - 1]
    shift_row = random.choice(shift_rows)
    fixed_row = random.choice(fixed_rows)
    cells.add((shift_row, random.choice(edge_cols)))
    cells.add((fixed_row, random.choice(edge_cols)))

    if len(shift_rows) >= 2 and count >= 4:
        other_shift = random.choice([r for r in shift_rows if r != shift_row])
        cells.add((other_shift, random.randint(1, width - 2)))

    while len(cells) < count:
        cells.add((random.randrange(height), random.randrange(width)))

    return list(cells)


def _choose_random_markers(height: int, width: int, count: int) -> list[tuple[int, int]]:
    cells = random.sample([(r, c) for r in range(height) for c in range(width)], count)
    return [(int(r), int(c)) for r, c in cells]


def _build_panel(height: int, width: int, marker_count: int, *, is_winner: bool) -> np.ndarray:
    panel = np.full((height, width), PANEL, dtype=int)
    if is_winner:
        markers = _choose_winner_markers(height, width, marker_count)
    else:
        markers = _choose_random_markers(height, width, marker_count)
    for r, c in markers:
        panel[r, c] = MARKER
    return panel


def _panel_box_in_cell(cell: tuple[int, int, int, int], dims: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, r1, c0, c1 = cell
    height, width = dims
    top = random.randint(r0, r1 - height)
    left = random.randint(c0, c1 - width)
    return top, left, top + height - 1, left + width - 1


def _render_grid(shape: tuple[int, int], placements: list[tuple[tuple[int, int, int, int], np.ndarray]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for (r0, c0, r1, c1), panel in placements:
        grid[r0 : r1 + 1, c0 : c1 + 1] = panel
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        panels = _find_panels(grid)
        if len(panels) not in (3, 4):
            return False
        if any(panel["marker_count"] <= 0 for panel in panels):
            return False

        best = max(panel["marker_count"] for panel in panels)
        winners = [panel for panel in panels if panel["marker_count"] == best]
        if len(winners) != 1:
            return False

        winner = winners[0]
        if not any(panel["area"] > winner["area"] for panel in panels if panel is not winner):
            return False
        crop = winner["crop"]
        expected = _shift_selected_rows_right(crop)
        if not np.array_equal(main(grid), expected):
            return False

        if np.array_equal(expected, crop):
            return False
        if np.array_equal(expected, _shift_selected_rows_left(crop)):
            return False
        if np.array_equal(expected, _shift_top_anchored_right(crop)):
            return False
        if np.array_equal(expected, _shift_wrong_bottom_phase(crop)):
            return False

        shift_rows = [r for r in range(crop.shape[0]) if (crop.shape[0] - 1 - r) >= 2 and (crop.shape[0] - 1 - r) % 2 == 0]
        fixed_rows = [r for r in range(crop.shape[0]) if r not in shift_rows]
        if not any(np.any(crop[r] == MARKER) for r in shift_rows):
            return False
        if not any(np.any(crop[r] == MARKER) for r in fixed_rows):
            return False
        return True
    except Exception:
        return False


def _attempt() -> np.ndarray | None:
    rows = random.randint(12, 16)
    cols = random.randint(18, 26)
    row_bands = _random_split(rows, margin=1, gap=2, min_a=5, min_b=5)
    col_bands = _random_split(cols, margin=1, gap=2, min_a=7, min_b=7)
    if row_bands is None or col_bands is None:
        return None

    cells = []
    for rr in row_bands:
        for cc in col_bands:
            cells.append((rr[0], rr[1], cc[0], cc[1]))

    panel_count = random.randint(3, 4)
    chosen_cells = random.sample(range(4), panel_count)
    winner_idx = random.randrange(panel_count)

    dims = []
    winner_dims = None
    for local_idx, cell_idx in enumerate(chosen_cells):
        r0, r1, c0, c1 = cells[cell_idx]
        cell_h = r1 - r0
        cell_w = c1 - c0
        if local_idx == winner_idx:
            if cell_h < 5 or cell_w < 4:
                return None
            height = random.randint(5, min(7, cell_h))
            width = random.randint(4, min(7, cell_w))
            winner_dims = (height, width)
            dims.append((height, width))
        else:
            height = random.randint(4, min(7, cell_h))
            width = random.randint(4, min(8, cell_w))
            dims.append((height, width))

    assert winner_dims is not None
    winner_area = winner_dims[0] * winner_dims[1]
    bigger_candidates = []
    for local_idx, cell_idx in enumerate(chosen_cells):
        if local_idx == winner_idx:
            continue
        r0, r1, c0, c1 = cells[cell_idx]
        cell_h = r1 - r0
        cell_w = c1 - c0
        max_area = min(7, cell_h) * min(8, cell_w)
        if max_area > winner_area:
            bigger_candidates.append(local_idx)
    if not bigger_candidates:
        return None

    challenger_idx = random.choice(bigger_candidates)
    cell_idx = chosen_cells[challenger_idx]
    r0, r1, c0, c1 = cells[cell_idx]
    cell_h = r1 - r0
    cell_w = c1 - c0
    bigger_options = [
        (h, w)
        for h in range(4, min(7, cell_h) + 1)
        for w in range(4, min(8, cell_w) + 1)
        if h * w > winner_area
    ]
    if not bigger_options:
        return None
    dims[challenger_idx] = random.choice(bigger_options)

    counts = _choose_counts(panel_count, winner_idx)

    placements = []
    for local_idx, cell_idx in enumerate(chosen_cells):
        panel = _build_panel(dims[local_idx][0], dims[local_idx][1], counts[local_idx], is_winner=local_idx == winner_idx)
        box = _panel_box_in_cell(cells[cell_idx], dims[local_idx])
        placements.append((box, panel))

    grid = _render_grid((rows, cols), placements)
    if grid[0, 0] != BG:
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
