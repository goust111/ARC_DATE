from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 2
PANEL_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]

PANEL_HEIGHT_RANGE = (8, 10)
PANEL_WIDTH_RANGE = (8, 9)
TARGET_COUNT_RANGE = (6, 14)
MAX_GENERATION_ATTEMPTS = 500

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int, h: int, w: int) -> list[Coord]:
    out: list[Coord] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _is_lr_symmetric(mask: np.ndarray) -> bool:
    box = _bbox(mask)
    assert box is not None, "panel pattern must be non-empty"
    cropped = _crop_bbox(mask.astype(np.uint8), box).astype(bool)
    return bool(np.array_equal(cropped, np.fliplr(cropped)))


def _panel_info(panel: np.ndarray) -> dict:
    colors = sorted(set(map(int, np.unique(panel))) - {BACKGROUND})
    assert len(colors) == 1, "each panel must contain exactly one foreground color"
    color = colors[0]
    mask = panel == color
    return {
        "color": color,
        "mask": mask,
        "count": int(mask.sum()),
        "symmetric": _is_lr_symmetric(mask),
    }


def _overlay_order(panels: list[np.ndarray]) -> list[dict]:
    infos = [_panel_info(panel) for panel in panels]
    return sorted(infos, key=lambda item: (item["symmetric"], -item["count"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the three separated color panels by symmetry priority and transparently overlay them into one panel.

    Input structure:
    - A black background grid is split by exactly two solid red separator columns into three equal-width panels.
    - Each panel contains one monochrome pattern on black, and panel colors are not black or red.
    - The three panels have the same height and width, but their left/middle/right positions do not encode depth.

    Rule:
    - Remove the two separator columns and read the three panels.
    - For each panel, crop the non-black pattern to its bbox and test whether that shape is left-right mirror symmetric.
    - Symmetric panels have higher overlay priority than asymmetric panels; within the same class, the panel with fewer colored cells has higher priority.
    - Starting from the lowest priority panel and ending with the highest priority panel, draw each panel's non-black pixels onto a black canvas of single-panel size, with later pixels overwriting earlier ones.

    Key insight:
    - Spatial panel position is a distractor: depth comes from a geometric property (mirror symmetry) and is refined by a visible count statistic (pattern sparsity).
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    h, w = g.shape
    sep_cols = [c for c in range(w) if np.all(g[:, c] == SEPARATOR)]
    assert len(sep_cols) == 2, "expected exactly two separator columns"
    c0, c1 = sep_cols

    panel_widths = [c0, c1 - c0 - 1, w - c1 - 1]
    assert panel_widths[0] > 0 and len(set(panel_widths)) == 1, "separator columns must split the grid into three equal panels"
    panel_w = panel_widths[0]

    panels = [
        g[:, 0:panel_w],
        g[:, c0 + 1 : c1],
        g[:, c1 + 1 : w],
    ]
    assert all(panel.shape == (h, panel_w) for panel in panels)

    order = _overlay_order(panels)
    out = np.full((h, panel_w), BACKGROUND, dtype=int)
    for item in order:
        out[item["mask"]] = int(item["color"])
    return out


def _weighted_choice(cells: list[Coord], anchor: Coord) -> Coord:
    weights = []
    for r, c in cells:
        dist = abs(r - anchor[0]) + abs(c - anchor[1])
        weights.append(1.0 / (1.0 + dist))
    return random.choices(cells, weights=weights, k=1)[0]


def _symmetric_area(domain: set[Coord], axis_c: int) -> int:
    return sum(1 if c == axis_c else 2 for _r, c in domain)


def _symmetric_frontier(domain: set[Coord], h: int, w: int, axis_c: int) -> list[Coord]:
    frontier: set[Coord] = set()
    for r, c in domain:
        for rr, cc in _neighbors4(r, c, h, w):
            if cc > axis_c or (rr, cc) in domain:
                continue
            mirror_c = 2 * axis_c - cc
            if 0 <= mirror_c < w:
                frontier.add((rr, cc))
    return sorted(frontier)


def _build_symmetric_mask(h: int, w: int, axis_c: int, base: set[Coord], target: int) -> np.ndarray | None:
    if target < len(base):
        return None

    domain = set(base)
    anchor = min(base)
    while _symmetric_area(domain, axis_c) < target:
        remaining = target - _symmetric_area(domain, axis_c)
        candidates = [
            cell
            for cell in _symmetric_frontier(domain, h, w, axis_c)
            if (1 if cell[1] == axis_c else 2) <= remaining
        ]
        if not candidates:
            return None
        chosen = _weighted_choice(candidates, anchor)
        domain.add(chosen)

    if _symmetric_area(domain, axis_c) != target:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in domain:
        mask[r, c] = True
        mask[r, 2 * axis_c - c] = True
    return mask if _is_lr_symmetric(mask) else None


def _build_asymmetric_mask(h: int, w: int, base: set[Coord], target: int) -> np.ndarray | None:
    if target < len(base):
        return None

    cells = set(base)
    anchor = min(base)
    while len(cells) < target:
        frontier = sorted(
            {
                nb
                for r, c in cells
                for nb in _neighbors4(r, c, h, w)
                if nb not in cells
            }
        )
        if not frontier:
            return None
        cells.add(_weighted_choice(frontier, anchor))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    if _is_lr_symmetric(mask):
        return None
    return mask


def _sample_distinct_counts() -> list[int]:
    choices = list(range(TARGET_COUNT_RANGE[0], TARGET_COUNT_RANGE[1] + 1))
    for _ in range(100):
        counts = random.sample(choices, 3)
        if min(abs(a - b) for i, a in enumerate(counts) for b in counts[i + 1 :]) >= 2:
            return counts
    raise RuntimeError("failed to sample distinct counts")


def _render_panel(mask: np.ndarray, color: int) -> np.ndarray:
    panel = np.full(mask.shape, BACKGROUND, dtype=int)
    panel[mask] = int(color)
    return panel


def _attempt() -> np.ndarray | None:
    h = random.randint(*PANEL_HEIGHT_RANGE)
    w = random.randint(*PANEL_WIDTH_RANGE)
    axis_c = random.randint(2, w - 3)
    base_r = random.randint(1, h - 3)
    base = {(base_r, axis_c), (base_r + 1, axis_c)}

    colors = random.sample(PANEL_COLORS, 3)
    sym_count = random.choice((1, 2))
    symmetric_flags = [True] * sym_count + [False] * (3 - sym_count)
    random.shuffle(symmetric_flags)
    counts = _sample_distinct_counts()

    masks: list[np.ndarray] = []
    for is_symmetric, target in zip(symmetric_flags, counts):
        mask = None
        for _ in range(80):
            if is_symmetric:
                mask = _build_symmetric_mask(h, w, axis_c, base, target)
            else:
                mask = _build_asymmetric_mask(h, w, base, target)
            if mask is not None:
                break
        if mask is None:
            return None
        masks.append(mask)

    panels = [_render_panel(mask, color) for mask, color in zip(masks, colors)]
    order = _overlay_order(panels)

    expected = np.full((h, w), BACKGROUND, dtype=int)
    for item in order:
        expected[item["mask"]] = int(item["color"])

    visible = {color: int((expected == color).sum()) for color in colors}
    if any(count < 2 for count in visible.values()):
        return None
    if visible[order[0]["color"]] >= order[0]["count"]:
        return None
    if visible[order[1]["color"]] >= order[1]["count"]:
        return None

    grid = np.full((h, 3 * w + 2), BACKGROUND, dtype=int)
    grid[:, w] = SEPARATOR
    grid[:, 2 * w + 1] = SEPARATOR
    grid[:, 0:w] = panels[0]
    grid[:, w + 1 : 2 * w + 1] = panels[1]
    grid[:, 2 * w + 2 :] = panels[2]

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
