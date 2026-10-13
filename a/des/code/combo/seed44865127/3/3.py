import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _render_row(sprites: list[np.ndarray], bg: int) -> np.ndarray:
    assert sprites
    out_h = max(sprite.shape[0] for sprite in sprites)
    out_w = sum(sprite.shape[1] for sprite in sprites) + len(sprites) - 1
    out = np.full((out_h, out_w), bg, dtype=int)
    c = 0
    for sprite in sprites:
        h, w = sprite.shape
        out[:h, c : c + w] = sprite
        c += w + 1
    return out


def _split_panels(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    full_cols_by_color: dict[int, list[int]] = {}
    for c in range(w):
        col = g[:, c]
        if np.all(col == col[0]):
            full_cols_by_color.setdefault(int(col[0]), []).append(c)

    candidates = []
    for color, cols in full_cols_by_color.items():
        if color == bg or len(cols) not in (2, 3):
            continue
        if np.any(g[:, [c for c in range(w) if c not in cols]] == color):
            continue
        starts = [0] + [c + 1 for c in cols]
        ends = [c - 1 for c in cols] + [w - 1]
        if any(s > e for s, e in zip(starts, ends)):
            continue
        widths = [e - s + 1 for s, e in zip(starts, ends)]
        if len(widths) not in (3, 4) or len(set(widths)) != 1:
            continue
        panels = [g[:, s : e + 1] for s, e in zip(starts, ends)]
        candidates.append((color, panels))

    assert len(candidates) == 1, "expected one separator layout"
    sep_color, panels = candidates[0]
    assert len(panels) in (3, 4)
    for c in full_cols_by_color[sep_color]:
        assert np.all(g[:, c] == sep_color)
    return bg, sep_color, panels


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the per-panel extras left after removing the shared skeleton.

    Input structure
    - The grid has 3-4 equal-width panels separated by full-height single-color columns.
    - Each panel contains one monochrome foreground pattern on a shared background.
    - All panel patterns share the same skeleton positions, plus each panel has 1-3 extra pixels.

    Rule
    - Split the grid into panels using the separator columns.
    - Intersect the foreground masks of all panels to get the shared skeleton.
    - For each panel, keep only its foreground pixels outside that skeleton.
    - Crop each extra-pixel set to its tight bounding box and arrange the crops left-to-right,
      with one background column between consecutive sprites.

    Key insight
    The meaningful common structure is the intersection across panels; the output keeps only
    the per-panel differences, while preserving each difference sprite's internal geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg, _, panels = _split_panels(g)
    masks = []
    panel_colors = []
    for panel in panels:
        colors = [int(c) for c in np.unique(panel) if int(c) != bg]
        assert len(colors) == 1, "each panel must be monochrome"
        panel_colors.append(colors[0])
        mask = panel == colors[0]
        assert np.any(mask), "each panel needs foreground"
        masks.append(mask)

    skeleton = masks[0].copy()
    for mask in masks[1:]:
        skeleton &= mask
    assert np.any(skeleton), "panels must share a non-empty skeleton"

    sprites = []
    for panel, mask, color in zip(panels, masks, panel_colors):
        extra = mask & ~skeleton
        box = _bbox(extra)
        assert box is not None, "each panel must contribute extra pixels"
        r0, c0, r1, c1 = box
        crop_mask = extra[r0 : r1 + 1, c0 : c1 + 1]
        sprite = np.full(crop_mask.shape, bg, dtype=int)
        sprite[crop_mask] = color
        sprites.append(sprite)

    return _render_row(sprites, bg)


# --------------- generator ---------------

GEN_ATTEMPTS = 300
PANEL_H_RANGE = (6, 10)
PANEL_W_RANGE = (6, 10)
SKELETON_SIZE_RANGE = (4, 9)
EXTRA_COUNT_RANGE = (1, 4)
ALL_COLORS = tuple(range(10))
ORTHO_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))

EXTRA_PATTERNS = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (1, 1)),
    ((0, 1), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 1), (0, 2)),
)


def _random_connected_mask(rng: np.random.Generator, h: int, w: int, size: int):
    for _ in range(100):
        start = (int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1)))
        cells = {start}
        while len(cells) < size:
            frontier = []
            for r, c in cells:
                for dr, dc in ORTHO_STEPS:
                    nr, nc = r + dr, c + dc
                    if 1 <= nr < h - 1 and 1 <= nc < w - 1 and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
        if len(cells) != size:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        box = _bbox(mask)
        assert box is not None
        r0, c0, r1, c1 = box
        if r1 == r0 or c1 == c0:
            continue
        return mask
    return None


def _pattern_size(pattern) -> int:
    return len(pattern)


def _place_extra(
    rng: np.random.Generator,
    h: int,
    w: int,
    skeleton: np.ndarray,
    used_positions: set[tuple[int, int]],
    count: int,
):
    patterns = [pattern for pattern in EXTRA_PATTERNS if _pattern_size(pattern) == count]
    for _ in range(200):
        pattern = patterns[int(rng.integers(len(patterns)))]
        max_r = max(r for r, _ in pattern)
        max_c = max(c for _, c in pattern)
        anchor_r = int(rng.integers(0, h - max_r))
        anchor_c = int(rng.integers(0, w - max_c))
        coords = [(anchor_r + dr, anchor_c + dc) for dr, dc in pattern]
        if any(skeleton[r, c] for r, c in coords):
            continue
        if any((r, c) in used_positions for r, c in coords):
            continue
        return coords
    return None


def _build_panel(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    panel = np.full(mask.shape, bg, dtype=int)
    panel[mask] = color
    return panel


def _gen_attempt(rng: np.random.Generator):
    num_panels = 3 + int(rng.integers(2))
    panel_h = int(rng.integers(*PANEL_H_RANGE))
    panel_w = int(rng.integers(*PANEL_W_RANGE))

    skeleton_size = int(rng.integers(SKELETON_SIZE_RANGE[0], min(SKELETON_SIZE_RANGE[1], (panel_h - 2) * (panel_w - 2)) + 1))
    skeleton = _random_connected_mask(rng, panel_h, panel_w, skeleton_size)
    if skeleton is None:
        return None

    palette = list(ALL_COLORS)
    rng.shuffle(palette)
    bg = int(palette[0])
    sep = int(palette[1])
    panel_colors = [int(c) for c in palette[2 : 2 + num_panels]]
    if len(panel_colors) != num_panels:
        return None

    used_extra_positions: set[tuple[int, int]] = set()
    panels = []
    sprites = []
    for color in panel_colors:
        extra_count = int(rng.integers(*EXTRA_COUNT_RANGE))
        coords = _place_extra(rng, panel_h, panel_w, skeleton, used_extra_positions, extra_count)
        if coords is None:
            return None

        extra = np.zeros((panel_h, panel_w), dtype=bool)
        for r, c in coords:
            extra[r, c] = True
            used_extra_positions.add((r, c))

        panel_mask = skeleton | extra
        if np.any(panel_mask.all(axis=0)):
            return None

        panels.append(_build_panel(panel_mask, color, bg))

        box = _bbox(extra)
        assert box is not None
        r0, c0, r1, c1 = box
        sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        sprite[extra[r0 : r1 + 1, c0 : c1 + 1]] = color
        sprites.append(sprite)

    grid_w = num_panels * panel_w + (num_panels - 1)
    grid = np.full((panel_h, grid_w), bg, dtype=int)
    c = 0
    for idx, panel in enumerate(panels):
        grid[:, c : c + panel_w] = panel
        c += panel_w
        if idx != num_panels - 1:
            grid[:, c] = sep
            c += 1

    if _get_bg(grid) != bg:
        return None

    expected = _render_row(sprites, bg)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
