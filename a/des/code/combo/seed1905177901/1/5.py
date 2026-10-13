from __future__ import annotations

import random
from collections import deque

import numpy as np


BG_COLOR = 0
SEP_COLOR = 5
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
CATEGORY_ORDER = ("H", "V", "R")
GEN_ATTEMPTS = 400
PANEL_COUNT_RANGE = (3, 5)
PANEL_SIDE_RANGE = (7, 11)
SHAPE_ATTEMPTS = 200


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].astype(bool)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(int(v) for v in coords[0])
    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                queue.append((rr, cc))
    return len(seen) == int(mask.sum())


def _classify_mask(mask: np.ndarray) -> str:
    tight = _tight_mask(mask)
    h_sym = np.array_equal(tight, np.flipud(tight))
    v_sym = np.array_equal(tight, np.fliplr(tight))
    r_sym = np.array_equal(tight, np.rot90(tight, 2))

    if h_sym and not v_sym:
        return "H"
    if v_sym and not h_sym:
        return "V"
    if r_sym and not h_sym and not v_sym:
        return "R"
    if not h_sym and not v_sym and not r_sym:
        return "N"
    raise AssertionError("object does not belong to exactly one allowed symmetry class")


def _safe_classify_mask(mask: np.ndarray) -> str | None:
    try:
        return _classify_mask(mask)
    except AssertionError:
        return None


def _find_separator_layout(grid: np.ndarray, bg: int) -> tuple[int, int, list[int]]:
    h, w = grid.shape
    del h
    full_cols: dict[int, list[int]] = {}
    for c in range(w):
        column = grid[:, c]
        color = int(column[0])
        if color == bg:
            continue
        if np.all(column == color):
            full_cols.setdefault(color, []).append(c)

    candidates: list[tuple[int, int, list[int]]] = []
    for color, cols in full_cols.items():
        cols = sorted(cols)
        panel_count = len(cols) + 1
        if not 3 <= panel_count <= 5:
            continue
        panel_width = cols[0]
        if panel_width <= 0:
            continue
        expected_cols = [panel_width + i * (panel_width + 1) for i in range(len(cols))]
        if cols != expected_cols:
            continue
        if panel_count * panel_width + len(cols) != w:
            continue
        candidates.append((color, panel_width, cols))

    assert len(candidates) == 1, "expected a unique grey separator layout"
    return candidates[0]


def _split_panels(grid: np.ndarray) -> tuple[int, int, list[np.ndarray]]:
    bg = _dominant_color(grid)
    sep_color, panel_width, sep_cols = _find_separator_layout(grid, bg)

    panels: list[np.ndarray] = []
    start = 0
    for sep_col in sep_cols:
        panel = grid[:, start:sep_col]
        assert panel.shape[1] == panel_width
        panels.append(panel.copy())
        start = sep_col + 1
    panel = grid[:, start:]
    assert panel.shape[1] == panel_width
    panels.append(panel.copy())
    return bg, sep_color, panels


def _classify_panel(panel: np.ndarray, bg: int) -> str:
    fg_mask = panel != bg
    assert fg_mask.any(), "each panel must contain one object"
    colors = np.unique(panel[fg_mask])
    assert len(colors) == 1, "panel object must be monochromatic"
    assert _is_connected(fg_mask), "panel object must be 4-connected"
    return _classify_mask(fg_mask)


def _safe_classify_panel(panel: np.ndarray, bg: int) -> str | None:
    try:
        return _classify_panel(panel, bg)
    except AssertionError:
        return None


def _transform_panel(panel: np.ndarray, category: str) -> np.ndarray | None:
    if category == "H":
        return panel.copy()
    if category == "V":
        return np.fliplr(panel).copy()
    if category == "R":
        assert panel.shape[0] == panel.shape[1], "90-degree panel rotation requires square panels"
        return np.rot90(panel, -1).copy()
    if category == "N":
        return None
    raise AssertionError(f"unknown category: {category}")


def _assemble_panels(panels: list[np.ndarray], sep_color: int, bg: int) -> np.ndarray:
    assert panels
    height, width = panels[0].shape
    out_width = len(panels) * width + (len(panels) - 1)
    out = np.full((height, out_width), bg, dtype=int)
    for i, panel in enumerate(panels):
        assert panel.shape == (height, width)
        c0 = i * (width + 1)
        out[:, c0 : c0 + width] = panel
        if i + 1 < len(panels):
            out[:, c0 + width] = sep_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder panel objects by symmetry class after category-specific transforms.

    Panels are separated by full grey columns. Each panel contains exactly one
    monochromatic 4-connected object on a black background. The object's tight
    mask belongs to exactly one of four symmetry classes:
    horizontal-only, vertical-only, 180-rotation-only, or none.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg, sep_color, panels = _split_panels(g)
    buckets: dict[str, list[np.ndarray]] = {key: [] for key in CATEGORY_ORDER}
    for panel in panels:
        category = _classify_panel(panel, bg)
        transformed = _transform_panel(panel, category)
        if transformed is not None:
            buckets[category].append(transformed)

    kept = buckets["H"] + buckets["V"] + buckets["R"]
    assert kept, "at least one panel should survive"
    return _assemble_panels(kept, sep_color, bg)


def _grow_connected_mask(
    h: int,
    w: int,
    target_size: int,
    seed: tuple[int, int],
) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    mask[seed] = True
    chosen = {seed}
    frontier = {
        (seed[0] + dr, seed[1] + dc)
        for dr, dc in DIR4
        if 0 <= seed[0] + dr < h and 0 <= seed[1] + dc < w
    }

    while len(chosen) < target_size:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if cell in chosen:
            continue
        chosen.add(cell)
        mask[cell] = True
        r, c = cell
        for dr, dc in DIR4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in chosen:
                frontier.add((rr, cc))
    return mask


def _gen_horizontal_mask(max_dim: int) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(3, max_dim)
        w = random.randint(3, max_dim)
        top_h = (h + 1) // 2
        seed = (top_h - 1, random.randrange(w))
        size = random.randint(2, top_h * w)
        top = _grow_connected_mask(top_h, w, size, seed)
        if top is None:
            continue
        if h % 2 == 1:
            full = np.vstack([top, np.flipud(top[:-1])])
        else:
            full = np.vstack([top, np.flipud(top)])
        full = _tight_mask(full)
        if _is_connected(full) and _safe_classify_mask(full) == "H":
            return full
    return None


def _gen_vertical_mask(max_dim: int) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(3, max_dim)
        w = random.randint(3, max_dim)
        left_w = (w + 1) // 2
        seed = (random.randrange(h), left_w - 1)
        size = random.randint(2, h * left_w)
        left = _grow_connected_mask(h, left_w, size, seed)
        if left is None:
            continue
        if w % 2 == 1:
            full = np.hstack([left, np.fliplr(left[:, :-1])])
        else:
            full = np.hstack([left, np.fliplr(left)])
        full = _tight_mask(full)
        if _is_connected(full) and _safe_classify_mask(full) == "V":
            return full
    return None


def _gen_rot_mask(max_dim: int) -> np.ndarray | None:
    odd_dims = [dim for dim in range(3, max_dim + 1) if dim % 2 == 1]
    if not odd_dims:
        return None
    for _ in range(SHAPE_ATTEMPTS):
        h = random.choice(odd_dims)
        w = random.choice(odd_dims)
        center = (h // 2, w // 2)
        size = random.randint(2, max(2, (h * w + 1) // 2))
        seed_mask = _grow_connected_mask(h, w, size, center)
        if seed_mask is None:
            continue
        full = seed_mask | np.rot90(seed_mask, 2)
        full = _tight_mask(full)
        if _is_connected(full) and _safe_classify_mask(full) == "R":
            return full
    return None


def _gen_none_mask(max_dim: int) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(3, max_dim)
        w = random.randint(3, max_dim)
        size = random.randint(3, max(3, min(h * w - 1, (2 * h * w) // 3)))
        seed = (random.randrange(h), random.randrange(w))
        mask = _grow_connected_mask(h, w, size, seed)
        if mask is None:
            continue
        mask = _tight_mask(mask)
        if _is_connected(mask) and _safe_classify_mask(mask) == "N":
            return mask
    return None


def _generate_mask(category: str, max_dim: int) -> np.ndarray:
    generators = {
        "H": _gen_horizontal_mask,
        "V": _gen_vertical_mask,
        "R": _gen_rot_mask,
        "N": _gen_none_mask,
    }
    mask = generators[category](max_dim)
    if mask is None:
        raise RuntimeError(f"failed to generate a {category} mask")
    return mask


def _place_mask(mask: np.ndarray, side: int, color: int, category: str) -> np.ndarray | None:
    h, w = mask.shape
    if h > side - 2 or w > side - 2:
        return None
    row_choices = list(range(1, side - h))
    col_choices = list(range(1, side - w))
    random.shuffle(row_choices)
    random.shuffle(col_choices)

    for r0 in row_choices:
        for c0 in col_choices:
            panel = np.full((side, side), BG_COLOR, dtype=int)
            patch = panel[r0 : r0 + h, c0 : c0 + w]
            patch[mask] = color
            if _safe_classify_panel(panel, BG_COLOR) != category:
                continue
            if category == "V" and np.array_equal(panel, np.fliplr(panel)):
                continue
            if category == "R" and np.array_equal(panel, np.rot90(panel, -1)):
                continue
            return panel
    return None


def _sample_categories(panel_count: int) -> list[str]:
    options = ["H", "V", "R", "N"]
    while True:
        categories = [random.choice(options) for _ in range(panel_count)]
        if all(category == "N" for category in categories):
            continue
        if len(set(categories)) < 2:
            continue
        return categories


def _expected_output(panels: list[np.ndarray], categories: list[str]) -> np.ndarray:
    buckets: dict[str, list[np.ndarray]] = {key: [] for key in CATEGORY_ORDER}
    for panel, category in zip(panels, categories):
        transformed = _transform_panel(panel, category)
        if transformed is not None:
            buckets[category].append(transformed)
    kept = buckets["H"] + buckets["V"] + buckets["R"]
    return _assemble_panels(kept, SEP_COLOR, BG_COLOR)


def _gen_attempt() -> np.ndarray | None:
    panel_count = random.randint(*PANEL_COUNT_RANGE)
    side = random.randint(*PANEL_SIDE_RANGE)
    colors = random.sample(OBJECT_COLORS, panel_count)
    categories = _sample_categories(panel_count)

    panels: list[np.ndarray] = []
    for color, category in zip(colors, categories):
        mask = _generate_mask(category, side - 2)
        panel = _place_mask(mask, side, color, category)
        if panel is None:
            return None
        panels.append(panel)

    expected = _expected_output(panels, categories)
    inp = _assemble_panels(panels, SEP_COLOR, BG_COLOR)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    # Reject the rare degenerate case where no visible change happens at all.
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-symmetry input")
