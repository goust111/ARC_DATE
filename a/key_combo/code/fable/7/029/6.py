from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, colors_present, components_mask, crop_bbox, get_bg, paste, random_free_location


GEN_MAX_ATTEMPTS = 400
GEN_PANEL_COUNT_OPTIONS = (4, 5)
GEN_PANEL_SIZE_BY_COUNT = {
    4: {"h": (5, 9), "w": (5, 9)},
    5: {"h": (5, 8), "w": (5, 8)},
}
GEN_BG = Color.BLACK
GEN_PANEL_COLORS = tuple(c for c in Color.ALL_COLORS if c != GEN_BG)
GEN_EDGE_BORDER = 1
GEN_PANEL_PADDING = 1


def _extract_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    panels: list[tuple[tuple[int, int, int, int], np.ndarray]] = []

    for coords in components_mask(g != bg, connectivity=4):
        box = bbox_coords(coords)
        patch = crop_bbox(g, box)
        assert patch.size > 0
        assert np.all(patch != bg), "each object must be a solid non-background rectangle"
        assert patch.shape[0] * patch.shape[1] == int(coords.shape[0]), "foreground objects must be filled rectangles"
        panels.append((box, patch))

    assert 4 <= len(panels) <= 5, "expected 4 or 5 same-sized panels"
    panels.sort(key=lambda item: (item[0][0], item[0][1]))
    shapes = {patch.shape for _box, patch in panels}
    assert len(shapes) == 1, "all panels must share one shape"
    return [patch for _box, patch in panels]


def _majority_color(values: list[int]) -> int:
    counts = Counter(values)
    ranked = counts.most_common()
    assert ranked, "empty color vote"
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1], "cellwise majority must be unique"
    return int(ranked[0][0])


def _reconstruct_defects(panels: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    assert panels
    shape = panels[0].shape
    assert all(panel.shape == shape for panel in panels)
    h, w = shape

    template = np.zeros(shape, dtype=int)
    for r in range(h):
        for c in range(w):
            template[r, c] = _majority_color([int(panel[r, c]) for panel in panels])

    out = np.full(shape, Color.BLACK, dtype=int)
    for panel in panels:
        diff = panel != template
        rr, cc = np.where(diff)
        for raw_r, raw_c in zip(rr, cc):
            r = int(raw_r)
            c = int(raw_c)
            color = int(panel[r, c])
            assert out[r, c] == Color.BLACK, "defect positions must not overlap across panels"
            out[r, c] = color
    return template, out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align same-sized filled pattern panels and keep only the minority defect pixels.

    Input structure:
    - A black background contains 4 or 5 separated filled rectangles of identical size.
    - Every rectangle is a full multicolor panel with no background holes.
    - The panels are copies of one hidden base pattern, except each panel has 1 to 3 altered pixels at local positions
      that do not overlap with the altered positions of the other panels.

    Rule:
    - Detect the filled rectangular panels and crop each one to its own bounding box.
    - Align the crops by local coordinates and, for every cell, take the unique majority color across panels to
      reconstruct the hidden base pattern.
    - For every panel cell that disagrees with this reconstructed base, write that panel's actual color into the same
      local position of a black output grid of panel size.

    Key insight:
    - The repeated panel pattern is pure noise for the answer; only the cellwise cross-panel majority reveals the hidden
      template, and the true output is the sparse union of the minority deviations.
    """
    panels = _extract_panels(np.asarray(grid, dtype=int))
    _template, out = _reconstruct_defects(panels)
    return out


def _make_template(h: int, w: int) -> np.ndarray:
    for _ in range(80):
        palette = random.sample(GEN_PANEL_COLORS, random.randint(3, 5))
        seed_count = random.randint(len(palette), min(h * w, len(palette) + 4))
        seed_positions = random.sample([(r, c) for r in range(h) for c in range(w)], seed_count)
        seed_colors = palette.copy()
        while len(seed_colors) < seed_count:
            seed_colors.append(random.choice(palette))
        random.shuffle(seed_colors)

        template = np.zeros((h, w), dtype=int)
        for r in range(h):
            for c in range(w):
                best_i = min(
                    range(seed_count),
                    key=lambda i: (
                        abs(r - seed_positions[i][0]) + abs(c - seed_positions[i][1]),
                        abs(r - seed_positions[i][0]) * abs(c - seed_positions[i][1]),
                        i,
                    ),
                )
                template[r, c] = seed_colors[best_i]

        for _ in range(random.randint(1, 3)):
            color = random.choice(palette)
            r0 = random.randint(0, h - 1)
            c0 = random.randint(0, w - 1)
            r1 = random.randint(r0, h - 1)
            c1 = random.randint(c0, w - 1)
            template[r0 : r1 + 1, c0 : c1 + 1] = color

        cols = colors_present(template)
        counts = Counter(map(int, template.ravel()))
        row_variants = {tuple(map(int, row)) for row in template.tolist()}
        col_variants = {tuple(map(int, col)) for col in template.T.tolist()}
        if len(cols) >= 3 and max(counts.values()) <= int(0.65 * h * w) and len(row_variants) >= 2 and len(col_variants) >= 2:
            return template

    raise RuntimeError("failed to synthesize a suitably irregular base template")


def _sample_defect_positions(h: int, w: int, total: int) -> list[tuple[int, int]]:
    all_cells = [(r, c) for r in range(h) for c in range(w)]
    border = [(r, c) for r, c in all_cells if r in {0, h - 1} or c in {0, w - 1}]
    interior = [cell for cell in all_cells if cell not in set(border)]
    chosen: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()

    if border and total >= 1 and random.random() < 0.75:
        cell = random.choice(border)
        chosen.append(cell)
        used.add(cell)
    if interior and total >= 2 and random.random() < 0.75:
        cell = random.choice(interior)
        if cell not in used:
            chosen.append(cell)
            used.add(cell)

    remaining = [cell for cell in all_cells if cell not in used]
    chosen.extend(random.sample(remaining, total - len(chosen)))
    random.shuffle(chosen)
    return chosen


def _apply_defects(template: np.ndarray, defect_counts: list[int]) -> tuple[list[np.ndarray], np.ndarray]:
    h, w = template.shape
    total = sum(defect_counts)
    positions = _sample_defect_positions(h, w, total)
    expected = np.full((h, w), Color.BLACK, dtype=int)
    panels = [template.copy() for _ in defect_counts]

    idx = 0
    for panel, defect_count in zip(panels, defect_counts):
        for _ in range(defect_count):
            r, c = positions[idx]
            idx += 1
            base_color = int(template[r, c])
            new_color = random.choice([color for color in GEN_PANEL_COLORS if color != base_color])
            panel[r, c] = new_color
            expected[r, c] = new_color

    assert idx == total
    assert int(np.count_nonzero(expected != Color.BLACK)) == total
    return panels, expected


def _grid_size_for_panels(panel_count: int, h: int, w: int) -> tuple[int, int]:
    min_rows = max(16, 2 * h + 5)
    max_rows = min(30, 2 * h + 9)
    if panel_count == 5:
        min_cols = max(22, 3 * w + 4)
    else:
        min_cols = max(16, 2 * w + 5)
    max_cols = min(30, 3 * w + 8)
    assert min_rows <= max_rows and min_cols <= max_cols
    return random.randint(min_rows, max_rows), random.randint(min_cols, max_cols)


def _bg_is_unique_mode(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    ranked = counts.most_common()
    return len(ranked) == 1 or ranked[0][1] > ranked[1][1]


def _semantic_checks(grid: np.ndarray, panels: list[np.ndarray], expected: np.ndarray) -> bool:
    try:
        extracted = _extract_panels(grid)
        if len(extracted) != len(panels):
            return False
        template, out = _reconstruct_defects(extracted)
    except Exception:
        return False

    if not np.array_equal(out, expected):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(expected, template):
        return False
    if any(np.array_equal(expected, panel) for panel in panels):
        return False

    nonblack = np.argwhere(expected != Color.BLACK)
    if len(nonblack) < len(panels):
        return False

    for raw_r, raw_c in nonblack:
        r = int(raw_r)
        c = int(raw_c)
        mismatches = [int(panel[r, c]) for panel in panels if int(panel[r, c]) != int(template[r, c])]
        if len(mismatches) != 1 or mismatches[0] != int(expected[r, c]):
            return False

    return _bg_is_unique_mode(grid)


def _gen_attempt() -> np.ndarray | None:
    panel_count = random.choice(GEN_PANEL_COUNT_OPTIONS)
    size_ranges = GEN_PANEL_SIZE_BY_COUNT[panel_count]
    h = random.randint(*size_ranges["h"])
    w = random.randint(*size_ranges["w"])

    defect_counts = [random.randint(1, 3) for _ in range(panel_count)]
    if sum(defect_counts) >= h * w:
        return None

    template = _make_template(h, w)
    panels, expected = _apply_defects(template, defect_counts)

    rows, cols = _grid_size_for_panels(panel_count, h, w)
    grid = np.full((rows, cols), GEN_BG, dtype=int)
    for panel in panels:
        try:
            r0, c0 = random_free_location(
                grid,
                panel,
                transparent=GEN_BG,
                border=GEN_EDGE_BORDER,
                padding=GEN_PANEL_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(grid, panel, r0, c0)

    return grid if _semantic_checks(grid, panels, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid deviation-overlay input")
