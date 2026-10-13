import numpy as np

from arc_utils import colors_present, get_bg, label_mask, paste
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite four quadrant panels into one panel, resolving overlaps by a fixed positional layer order.

    Input structure
    - The input is a 2x2 array of equal-size panels on a shared background.
    - Each panel contains exactly one distinct foreground color.
    - Corresponding cells in the four panels map to the same output cell.

    Rule
    - Create one output panel with the same shape as a single input panel.
    - Paint the upper-left panel first, then lower-right, then lower-left,
      then upper-right.
    - Only non-background pixels are painted; later panels overwrite earlier
      pixels at the same position.

    Key insight
    - The four panels are transparent layers whose depth is determined by panel
      position, not by numeric color order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = g.shape
    assert H % 2 == 0 and W % 2 == 0, "grid must split evenly into four panels"
    panel_h = H // 2
    panel_w = W // 2
    assert panel_h > 0 and panel_w > 0, "panels must be non-empty"

    bg = get_bg(g)
    upper_left = g[:panel_h, :panel_w]
    upper_right = g[:panel_h, panel_w:]
    lower_left = g[panel_h:, :panel_w]
    lower_right = g[panel_h:, panel_w:]
    panels = (upper_left, upper_right, lower_left, lower_right)

    panel_colors = [colors_present(panel, exclude={bg}) for panel in panels]
    assert all(len(colors) == 1 for colors in panel_colors), "each panel must have exactly one foreground color"
    assert len({colors[0] for colors in panel_colors}) == 4, "panel foreground colors must be distinct"

    out = np.full((panel_h, panel_w), bg, dtype=int)
    for panel in (upper_left, lower_right, lower_left, upper_right):
        mask = panel != bg
        out[mask] = panel[mask]
    return out

# --------------- generator (o5x) ---------------

# Panel geometry and fixed quadrant placement.
_PANEL_SHAPE, _INPUT_SHAPE = (5, 5), (10, 10)
_PANEL_ORIGINS = ((0, 0), (0, 5), (5, 0), (5, 5))
_PANEL_COUNT = 4

# Fixed panel colors and overlay priority.
_BACKGROUND, _GREEN, _GREEN_PANEL = 0, 3, 1
_PANEL_COLORS, _PANEL_PRIORITY = (4, 3, 9, 8), (0, 3, 2, 1)  # UL, UR, LL, LR
_COLOR_TO_PANEL = {4: 0, 3: 1, 9: 2, 8: 3}
_COLOR_PRIORITY = {4: 0, 8: 1, 9: 2, 3: 3}
_NON_GREEN_PANELS = (0, 2, 3)
_FILL_PANEL_ORDER = (2, 3, 0)

# Coverage, density, and component constraints.
_UNION_COVERAGE_RANGE, _VISIBLE_COLOR_COUNTS = (20, 24), (3, 4)
_OPTIONAL_VISIBLE_COLORS = [4, 8, 9]
_PANEL_FG_RANGE, _COMPONENT_RANGE = (8, 17), (1, 6)
_PAIR_OVERLAP_RANGE = (1, 10)

# Sampling controls.
_LOW_OVERLAP_CHOICE_RATE, _GENERATION_RETRIES = 0.7, 300


def _gen_component_count(mask: np.ndarray) -> int:
    return label_mask(mask, connectivity=4)[1]


def _gen_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(mask)]


def _gen_composition(total: int, parts: int, low: int, high: int) -> list[int] | None:
    counts = []
    remaining = int(total)
    for i in range(parts):
        rest = parts - i - 1
        lo = max(low, remaining - high * rest)
        hi = min(high, remaining - low * rest)
        if lo > hi:
            return None
        val = random.randint(lo, hi)
        counts.append(val)
        remaining -= val
    random.shuffle(counts)
    return counts


def _gen_fill_mask(mask: np.ndarray, allowed: np.ndarray, target: int) -> np.ndarray | None:
    out = mask.copy()
    while int(out.sum()) < target:
        choices = []
        for r, c in _gen_coords(allowed & ~out):
            trial = out.copy()
            trial[r, c] = True
            if _gen_component_count(trial) <= _COMPONENT_RANGE[1]:
                choices.append((r, c))
        if not choices:
            return None
        r, c = random.choice(choices)
        out[r, c] = True
    return out


def _gen_pick_witness(masks: np.ndarray, green_mask: np.ndarray) -> tuple[int, int] | None:
    coords = _gen_coords(green_mask)
    random.shuffle(coords)
    for r, c in coords:
        ok = True
        for p in _NON_GREEN_PANELS:
            trial = masks[p].copy()
            trial[r, c] = True
            if _gen_component_count(trial) > _COMPONENT_RANGE[1]:
                ok = False
                break
        if ok:
            return r, c
    return None


def _gen_fill_panel(
    masks: np.ndarray,
    panel_i: int,
    allowed: np.ndarray,
    target: int,
) -> bool:
    while int(masks[panel_i].sum()) < target:
        low_overlap = []
        any_overlap = []
        for r, c in _gen_coords(allowed & ~masks[panel_i]):
            trial = masks[panel_i].copy()
            trial[r, c] = True
            if _gen_component_count(trial) > _COMPONENT_RANGE[1]:
                continue
            overlap_add = 0
            ok = True
            for other_i in range(_PANEL_COUNT):
                if other_i == panel_i:
                    continue
                if masks[other_i, r, c]:
                    overlap_add += 1
                if int((trial & masks[other_i]).sum()) > _PAIR_OVERLAP_RANGE[1]:
                    ok = False
                    break
            if ok:
                any_overlap.append((r, c))
                if overlap_add == 0:
                    low_overlap.append((r, c))
        choices = low_overlap if low_overlap and random.random() < _LOW_OVERLAP_CHOICE_RATE else any_overlap
        if not choices:
            return False
        r, c = random.choice(choices)
        masks[panel_i, r, c] = True
    return True


def _gen_make_output() -> np.ndarray | None:
    union_count = random.randint(*_UNION_COVERAGE_RANGE)
    visible_count = random.choice(_VISIBLE_COLOR_COUNTS)
    extra_colors = random.sample(_OPTIONAL_VISIBLE_COLORS, visible_count - 1)
    n_green = random.randint(_PANEL_FG_RANGE[0], min(_PANEL_FG_RANGE[1], union_count - len(extra_colors)))
    other_counts = _gen_composition(
        union_count - n_green,
        len(extra_colors),
        1,
        _PANEL_FG_RANGE[1],
    )
    if other_counts is None:
        return None

    out = np.zeros(_PANEL_SHAPE, dtype=int)
    available = np.ones(_PANEL_SHAPE, dtype=bool)
    green = _gen_fill_mask(np.zeros(_PANEL_SHAPE, dtype=bool), available, n_green)
    if green is None:
        return None
    out[green] = _GREEN
    available &= ~green

    color_counts = list(zip(extra_colors, other_counts))
    random.shuffle(color_counts)
    for color, count in color_counts:
        mask = _gen_fill_mask(np.zeros(_PANEL_SHAPE, dtype=bool), available, count)
        if mask is None:
            return None
        out[mask] = color
        available &= ~mask
    return out


def _gen_allowed_masks(out: np.ndarray) -> np.ndarray:
    visible_priority = np.full(_PANEL_SHAPE, -1, dtype=int)
    for color, priority in _COLOR_PRIORITY.items():
        visible_priority[out == color] = priority
    allowed = np.zeros((_PANEL_COUNT, *_PANEL_SHAPE), dtype=bool)
    for p, priority in enumerate(_PANEL_PRIORITY):
        allowed[p] = visible_priority >= priority
    return allowed


def _gen_seed_visible_masks(out: np.ndarray) -> np.ndarray:
    """Create panel masks containing all cells where that panel is visible."""
    masks = np.zeros((_PANEL_COUNT, *_PANEL_SHAPE), dtype=bool)
    for color, panel_i in _COLOR_TO_PANEL.items():
        masks[panel_i] = out == color
    return masks


def _gen_sample_targets(masks: np.ndarray, allowed: np.ndarray, out: np.ndarray) -> list[int] | None:
    """Sample final foreground counts for panels that still need filling."""
    targets = [0, int(masks[_GREEN_PANEL].sum()), 0, 0]
    for panel_i in _NON_GREEN_PANELS:
        current = int(masks[panel_i].sum())
        allowed_count = int(allowed[panel_i].sum())
        non_green_allowed = int((allowed[panel_i] & (out != _GREEN)).sum())
        hi = min(_PANEL_FG_RANGE[1], allowed_count, non_green_allowed + min(_PAIR_OVERLAP_RANGE[1], int((out == _GREEN).sum())))
        lo = max(_PANEL_FG_RANGE[0], current)
        if lo > hi:
            return None
        targets[panel_i] = random.randint(lo, hi)
    return targets


def _gen_valid_masks(masks: np.ndarray) -> bool:
    """Check the panel density, connectivity, and pairwise-overlap constraints."""
    if not all(_PANEL_FG_RANGE[0] <= int(masks[p].sum()) <= _PANEL_FG_RANGE[1] for p in range(_PANEL_COUNT)):
        return False
    if not all(_COMPONENT_RANGE[0] <= _gen_component_count(masks[p]) <= _COMPONENT_RANGE[1] for p in range(_PANEL_COUNT)):
        return False
    for a in range(_PANEL_COUNT):
        for b in range(a + 1, _PANEL_COUNT):
            overlap = int((masks[a] & masks[b]).sum())
            if not _PAIR_OVERLAP_RANGE[0] <= overlap <= _PAIR_OVERLAP_RANGE[1]:
                return False
    return True


def _gen_render_input(masks: np.ndarray) -> np.ndarray:
    """Render the four panel masks into the 10x10 solver input grid."""
    inp = np.zeros(_INPUT_SHAPE, dtype=int)
    for panel_i, (r0, c0) in enumerate(_PANEL_ORIGINS):
        panel = np.zeros(_PANEL_SHAPE, dtype=int)
        panel[masks[panel_i]] = _PANEL_COLORS[panel_i]
        paste(inp, panel, r0, c0, transparent=_BACKGROUND)
    return inp


def _gen_attempt() -> np.ndarray | None:
    out = _gen_make_output()
    if out is None:
        return None

    masks = _gen_seed_visible_masks(out)
    witness = _gen_pick_witness(masks, out == _GREEN)
    if witness is None:
        return None
    wr, wc = witness
    masks[:, wr, wc] = True

    allowed = _gen_allowed_masks(out)
    targets = _gen_sample_targets(masks, allowed, out)
    if targets is None:
        return None
    for panel_i in _FILL_PANEL_ORDER:
        if not _gen_fill_panel(masks, panel_i, allowed[panel_i], targets[panel_i]):
            return None
    if not _gen_valid_masks(masks):
        return None

    inp = _gen_render_input(masks)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GENERATION_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid ea9794b1 input")
