from __future__ import annotations

from collections import Counter
import random

import numpy as np


PANEL_NAMES = ("TL", "TR", "BL", "BR")
MAX_GENERATION_ATTEMPTS = 400
MOTIFS = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)


def _majority_color(panel: np.ndarray) -> int:
    counts = Counter(map(int, panel.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected non-empty panel"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "panel background must be a unique majority"
    return int(ranked[0][0])


def _extract_structure(grid: np.ndarray) -> tuple[np.ndarray, int, list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[1] >= 5, "expected main area plus gap and legend columns"

    gap_col = g[:, -2]
    global_bg = int(gap_col[0])
    assert np.all(gap_col == global_bg), "expected a single blank separator column before the legend"

    legend_col = g[:, -1]
    legend_colors = [int(v) for v in legend_col if int(v) != global_bg]
    assert 1 <= len(legend_colors) <= 4, "legend strip must list between 1 and 4 panel background colors"
    assert len(legend_colors) == len(set(legend_colors)), "legend colors must be unique"
    return g[:, :-2], global_bg, legend_colors


def _find_cross(main_area: np.ndarray) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for color in np.unique(main_area):
        rows = np.where(np.all(main_area == int(color), axis=1))[0]
        cols = np.where(np.all(main_area == int(color), axis=0))[0]
        for r in rows:
            for c in cols:
                candidates.append((int(color), int(r), int(c)))
    assert len(candidates) == 1, "expected exactly one full-row/full-column cross"
    return candidates[0]


def _split_panels(main_area: np.ndarray) -> tuple[dict[str, np.ndarray], int]:
    height, width = main_area.shape
    cross_color, cross_r, cross_c = _find_cross(main_area)
    assert height == 2 * cross_r + 1, "cross row must bisect the main area"
    assert width == 2 * cross_c + 1, "cross column must bisect the main area"

    panels = {
        "TL": main_area[:cross_r, :cross_c],
        "TR": main_area[:cross_r, cross_c + 1 :],
        "BL": main_area[cross_r + 1 :, :cross_c],
        "BR": main_area[cross_r + 1 :, cross_c + 1 :],
    }
    assert len({panel.shape for panel in panels.values()}) == 1, "quadrants must be equal-sized"
    for panel in panels.values():
        colors = set(map(int, np.unique(panel)))
        assert cross_color not in colors, "cross color must not appear inside panel interiors"
        assert len(colors) == 2, "each panel must have exactly one local background and one foreground color"
    return panels, cross_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the cross-split panels according to the side legend's background-color layer order.

    Input structure:
    - A left main area is split into four equal panels by one full row and one full column of a separator color.
    - Each panel contains exactly one local background color plus one foreground color.
    - To the right, after one blank separator column, a vertical legend strip lists some panel background colors from top to bottom.

    Rule:
    - Remove the cross and align the four panels by local coordinates.
    - Use each panel's majority color as its transparent lookup key; keep only the panels whose background colors appear in the legend.
    - Paint selected panels from the legend's bottom entry up to its top entry so higher listed layers overwrite lower ones, using only non-background panel pixels.

    Key insight:
    - The legend does not name the visible foreground colors; it names each panel's transparent local background, and that lookup both orders the layers and filters out unlisted panels.
    """
    main_area, global_bg, legend_colors = _extract_structure(np.asarray(grid, dtype=int))
    panels, _cross_color = _split_panels(main_area)

    panel_bgs = {name: _majority_color(panel) for name, panel in panels.items()}
    bg_to_panel = {bg: name for name, bg in panel_bgs.items()}
    assert len(bg_to_panel) == 4, "panel background colors must uniquely identify the four panels"
    for color in legend_colors:
        assert color in bg_to_panel, "every legend color must match a panel background"

    out = np.full(next(iter(panels.values())).shape, global_bg, dtype=int)
    for bg in reversed(legend_colors):
        panel = panels[bg_to_panel[bg]]
        mask = panel != bg
        out[mask] = panel[mask]
    return out


def _normalize_shape(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    shifted = sorted((r - min_r, c - min_c) for r, c in coords)
    return tuple((int(r), int(c)) for r, c in shifted)


def _transform_motif(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    transformed = tuple(coords)
    for _ in range(random.randint(0, 3)):
        transformed = tuple((c, -r) for r, c in transformed)
    if random.random() < 0.5:
        transformed = tuple((r, -c) for r, c in transformed)
    return _normalize_shape(transformed)


def _motif_mask(shape: tuple[tuple[int, int], ...], panel_h: int, panel_w: int) -> np.ndarray:
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    mask = np.zeros((panel_h, panel_w), dtype=bool)
    r0 = random.randint(0, panel_h - max_r - 1)
    c0 = random.randint(0, panel_w - max_c - 1)
    for r, c in shape:
        mask[r0 + r, c0 + c] = True
    return mask


def _add_shape(panel_masks: dict[str, np.ndarray], owners: tuple[str, ...], panel_h: int, panel_w: int) -> bool:
    for _ in range(80):
        shape = _transform_motif(random.choice(MOTIFS))
        mask = _motif_mask(shape, panel_h, panel_w)
        if any(np.any(panel_masks[name] & mask) for name in owners):
            continue
        for name in owners:
            panel_masks[name] |= mask
        return True
    return False


def _build_panel_masks(panel_h: int, panel_w: int, selected: tuple[str, ...]) -> dict[str, np.ndarray] | None:
    panel_masks = {name: np.zeros((panel_h, panel_w), dtype=bool) for name in PANEL_NAMES}

    for name in PANEL_NAMES:
        if not _add_shape(panel_masks, (name,), panel_h, panel_w):
            return None

    if len(selected) >= 2 and not _add_shape(panel_masks, selected, panel_h, panel_w):
        return None

    extra_shared = random.randint(1, 3)
    for _ in range(extra_shared):
        if len(selected) < 2:
            break
        subset_size = random.randint(2, len(selected))
        owners = tuple(sorted(random.sample(selected, subset_size), key=PANEL_NAMES.index))
        if not _add_shape(panel_masks, owners, panel_h, panel_w):
            return None

    omitted = tuple(name for name in PANEL_NAMES if name not in selected)
    if omitted:
        if not _add_shape(panel_masks, (random.choice(omitted),), panel_h, panel_w):
            return None

    area = panel_h * panel_w
    for name, mask in panel_masks.items():
        filled = int(mask.sum())
        if filled == 0 or filled >= area // 2:
            return None
    overlap = np.zeros((panel_h, panel_w), dtype=int)
    for name in selected:
        overlap += panel_masks[name].astype(int)
    if int(np.sum(overlap >= 2)) == 0:
        return None
    if omitted:
        omitted_unique = np.zeros((panel_h, panel_w), dtype=bool)
        for name in omitted:
            other_selected = np.zeros((panel_h, panel_w), dtype=bool)
            for selected_name in selected:
                other_selected |= panel_masks[selected_name]
            omitted_unique |= panel_masks[name] & ~other_selected
        if not omitted_unique.any():
            return None
    return panel_masks


def _render_input(
    panel_h: int,
    panel_w: int,
    global_bg: int,
    cross_color: int,
    panel_bgs: dict[str, int],
    panel_fgs: dict[str, int],
    panel_masks: dict[str, np.ndarray],
    legend_order: tuple[str, ...],
) -> np.ndarray:
    height = 2 * panel_h + 1
    width = 2 * panel_w + 3
    grid = np.full((height, width), global_bg, dtype=int)

    main_area = grid[:, : 2 * panel_w + 1]
    cross_r = panel_h
    cross_c = panel_w
    main_area[cross_r, :] = cross_color
    main_area[:, cross_c] = cross_color

    panel_slices = {
        "TL": (slice(0, panel_h), slice(0, panel_w)),
        "TR": (slice(0, panel_h), slice(panel_w + 1, 2 * panel_w + 1)),
        "BL": (slice(panel_h + 1, 2 * panel_h + 1), slice(0, panel_w)),
        "BR": (slice(panel_h + 1, 2 * panel_h + 1), slice(panel_w + 1, 2 * panel_w + 1)),
    }

    for name in PANEL_NAMES:
        rs, cs = panel_slices[name]
        panel = np.full((panel_h, panel_w), panel_bgs[name], dtype=int)
        panel[panel_masks[name]] = panel_fgs[name]
        main_area[rs, cs] = panel

    legend_len = len(legend_order)
    legend_start = random.randint(0, height - legend_len)
    legend_col = width - 1
    for offset, name in enumerate(legend_order):
        grid[legend_start + offset, legend_col] = panel_bgs[name]
    return grid


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(6, 9)
    panel_w = random.randint(6, 9)

    color_pool = random.sample(range(10), 10)
    global_bg = color_pool[0]
    cross_color = color_pool[1]
    panel_bgs = {name: color_pool[2 + i] for i, name in enumerate(PANEL_NAMES)}
    panel_fgs = {name: color_pool[6 + i] for i, name in enumerate(PANEL_NAMES)}

    legend_len = random.choices([2, 3, 4], weights=[4, 3, 2], k=1)[0]
    legend_order = tuple(random.sample(PANEL_NAMES, legend_len))

    panel_masks = _build_panel_masks(panel_h, panel_w, legend_order)
    if panel_masks is None:
        return None

    grid = _render_input(
        panel_h,
        panel_w,
        global_bg,
        cross_color,
        panel_bgs,
        panel_fgs,
        panel_masks,
        legend_order,
    )

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != (panel_h, panel_w):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
