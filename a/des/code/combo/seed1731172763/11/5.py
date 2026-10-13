import random

import numpy as np


BG = 0
SEP = 5
FILL_SEQUENCE = (2, 1, 3, 4, 6, 7, 8, 9)

# --------------- solver ---------------


def _panel_ranges(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    sep_cols = np.all(grid == SEP, axis=0)
    assert np.any(sep_cols), "expected at least one separator column"
    assert not sep_cols[0] and not sep_cols[-1], "separator columns cannot touch the outer border"

    ranges: list[tuple[int, int]] = []
    c = 0
    while c < w:
        if sep_cols[c]:
            while c < w and sep_cols[c]:
                c += 1
            continue
        start = c
        while c < w and not sep_cols[c]:
            c += 1
        ranges.append((start, c))

    assert len(ranges) >= 2, "expected multiple panels"
    widths = {c1 - c0 for c0, c1 in ranges}
    assert len(widths) == 1, "panels must have equal width"
    return ranges


def _count_pattern_cells(panel: np.ndarray) -> int:
    return int(np.count_nonzero(panel != BG))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each panel's missing subset pixels relative to the most complete panel.

    Input structure
    - Black background.
    - Full-height grey separator columns split the grid into equal-width panels.
    - Exactly one panel contains the most complete version of a shared pattern.
    - Every other panel is a strict subset of that pattern with some pixels removed to background.

    Rule
    - Use the panel with the most non-background pixels as the template.
    - For each other panel, find template pixels that are missing there.
    - Paint those missing pixels in left-to-right panel order with:
      Red, then Blue, then Green, then the remaining fixed sequence.
    - Leave the complete panel unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    ranges = _panel_ranges(g)
    panels = [g[:, c0:c1] for c0, c1 in ranges]
    counts = [_count_pattern_cells(panel) for panel in panels]
    max_count = max(counts)
    assert counts.count(max_count) == 1, "expected a unique most-complete panel"
    template_idx = counts.index(max_count)
    template = panels[template_idx]
    template_mask = template != BG

    assert len(panels) - 1 <= len(FILL_SEQUENCE), "not enough fill colors for incomplete panels"

    out = g.copy()
    fill_i = 0
    for panel_idx, (c0, c1) in enumerate(ranges):
        if panel_idx == template_idx:
            continue

        panel = g[:, c0:c1]
        panel_mask = panel != BG
        assert not np.any(panel_mask & (~template_mask)), "panel cannot add pixels outside the template"
        assert np.array_equal(panel[panel_mask], template[panel_mask]), "panel pixels must match the template colors"

        missing = template_mask & (~panel_mask)
        assert np.any(missing), "non-template panels must miss at least one template pixel"
        out[:, c0:c1][missing] = FILL_SEQUENCE[fill_i]
        fill_i += 1

    return out


# --------------- generator ---------------

PANEL_COUNT_RANGE = (3, 6)
PANEL_HEIGHT_RANGE = (5, 9)
PANEL_WIDTH_RANGE = (5, 9)
SEPARATOR_WIDTH_RANGE = (1, 2)
TEMPLATE_COLOR_COUNT_RANGE = (1, 3)
MASK_DENSITY_RANGE = (0.28, 0.62)
MAX_ATTEMPTS = 300


def _random_mask_noise(h: int, w: int) -> np.ndarray:
    density = random.uniform(*MASK_DENSITY_RANGE)
    return np.random.random((h, w)) < density


def _random_mask_rectangles(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for _ in range(random.randint(2, 4)):
        rh = random.randint(1, h)
        rw = random.randint(1, w)
        r0 = random.randint(0, h - rh)
        c0 = random.randint(0, w - rw)
        mask[r0 : r0 + rh, c0 : c0 + rw] = True
    return mask


def _random_mask_strokes(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for _ in range(random.randint(3, 6)):
        if random.choice((True, False)):
            r = random.randrange(h)
            c0 = random.randrange(w)
            c1 = random.randrange(c0, w)
            mask[r, c0 : c1 + 1] = True
        else:
            c = random.randrange(w)
            r0 = random.randrange(h)
            r1 = random.randrange(r0, h)
            mask[r0 : r1 + 1, c] = True
    return mask


def _generate_template_mask(h: int, w: int, panel_count: int) -> np.ndarray | None:
    min_pixels = panel_count + 3
    max_pixels = h * w - 1
    if min_pixels > max_pixels:
        return None

    for _ in range(100):
        mode = random.choice(("noise", "rectangles", "strokes"))
        if mode == "noise":
            mask = _random_mask_noise(h, w)
        elif mode == "rectangles":
            mask = _random_mask_rectangles(h, w)
        else:
            mask = _random_mask_strokes(h, w)

        count = int(mask.sum())
        if not (min_pixels <= count <= max_pixels):
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) < 2 or len(cols) < 2:
            continue
        if rows[-1] - rows[0] < 1 or cols[-1] - cols[0] < 1:
            continue
        return mask
    return None


def _colorize_template(mask: np.ndarray, colors: list[int]) -> np.ndarray:
    template = np.full(mask.shape, BG, dtype=int)
    cells = [tuple(cell) for cell in np.argwhere(mask)]
    seeds = random.sample(cells, min(len(colors), len(cells)))
    while len(seeds) < len(colors):
        seeds.append(random.choice(cells))

    for r, c in cells:
        best = min(
            range(len(colors)),
            key=lambda i: (abs(r - seeds[i][0]) + abs(c - seeds[i][1]), i),
        )
        template[r, c] = colors[best]
    return template


def _choose_missing_sets(mask: np.ndarray, panel_count: int) -> list[np.ndarray] | None:
    cells = [tuple(cell) for cell in np.argwhere(mask)]
    n_missing_panels = panel_count - 1
    max_remove = len(cells) - 2
    if max_remove < n_missing_panels:
        return None

    removal_counts = random.sample(range(1, max_remove + 1), n_missing_panels)
    random.shuffle(removal_counts)

    missing_masks: list[np.ndarray] = []
    seen: set[bytes] = set()
    for remove_count in removal_counts:
        for _ in range(100):
            removed = set(random.sample(cells, remove_count))
            missing = np.zeros(mask.shape, dtype=bool)
            for r, c in removed:
                missing[r, c] = True
            key = missing.tobytes()
            if key not in seen:
                seen.add(key)
                missing_masks.append(missing)
                break
        else:
            return None
    return missing_masks
def _build_input(
    template: np.ndarray,
    panel_count: int,
    complete_idx: int,
    missing_masks: list[np.ndarray],
    sep_w: int,
) -> np.ndarray:
    h, w = template.shape
    total_w = panel_count * w + (panel_count - 1) * sep_w
    grid = np.full((h, total_w), BG, dtype=int)

    miss_i = 0
    col = 0
    for panel_idx in range(panel_count):
        if panel_idx > 0:
            grid[:, col : col + sep_w] = SEP
            col += sep_w

        panel = template.copy()
        if panel_idx != complete_idx:
            panel[missing_masks[miss_i]] = BG
            miss_i += 1
        grid[:, col : col + w] = panel
        col += w

    return grid


def _build_expected_output(
    template: np.ndarray,
    panel_count: int,
    complete_idx: int,
    missing_masks: list[np.ndarray],
    sep_w: int,
) -> np.ndarray:
    out = _build_input(template, panel_count, complete_idx, missing_masks, sep_w)
    h, w = template.shape
    template_mask = template != BG

    miss_i = 0
    fill_i = 0
    col = 0
    for panel_idx in range(panel_count):
        if panel_idx > 0:
            col += sep_w

        if panel_idx != complete_idx:
            panel = out[:, col : col + w]
            missing = missing_masks[miss_i]
            assert np.array_equal(missing, template_mask & (panel == BG))
            panel[missing] = FILL_SEQUENCE[fill_i]
            miss_i += 1
            fill_i += 1
        col += w

    return out


def _gen_attempt() -> np.ndarray | None:
    panel_count = random.randint(*PANEL_COUNT_RANGE)
    sep_w = random.randint(*SEPARATOR_WIDTH_RANGE)
    h = random.randint(*PANEL_HEIGHT_RANGE)
    w = random.randint(*PANEL_WIDTH_RANGE)
    complete_idx = random.randrange(panel_count)

    mask = _generate_template_mask(h, w, panel_count)
    if mask is None:
        return None

    fill_colors = set(FILL_SEQUENCE[: panel_count - 1])
    available = [c for c in range(10) if c not in fill_colors and c not in (BG, SEP)]
    n_template_colors = random.randint(
        TEMPLATE_COLOR_COUNT_RANGE[0],
        min(TEMPLATE_COLOR_COUNT_RANGE[1], len(available)),
    )
    template_colors = random.sample(available, n_template_colors)
    template = _colorize_template(mask, template_colors)

    missing_masks = _choose_missing_sets(mask, panel_count)
    if missing_masks is None:
        return None

    grid = _build_input(template, panel_count, complete_idx, missing_masks, sep_w)
    expected = _build_expected_output(template, panel_count, complete_idx, missing_masks, sep_w)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
