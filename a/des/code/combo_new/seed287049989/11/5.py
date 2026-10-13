from __future__ import annotations

import random

import numpy as np


PALETTE = tuple(range(10))
SEPARATOR_COLOR = 5
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 240
RIGHT_PANEL_WIDTH_RANGE = (10, 14)
GRID_HEIGHT_RANGE = (10, 16)
MARGIN = 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _crop_color_mask(panel: np.ndarray, color: int) -> np.ndarray:
    coords = np.argwhere(panel == color)
    assert coords.size > 0, f"missing color {color}"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return panel[r0 : r1 + 1, c0 : c1 + 1] == color


def _classify_fragment(mask: np.ndarray) -> tuple[bool, bool]:
    fill_right = bool(mask[:, -1].all())
    fill_bottom = bool(mask[-1, :].all())
    assert fill_right or fill_bottom, "fragment must expose at least one fold edge"
    return fill_right, fill_bottom


def _mirror_right(mask: np.ndarray) -> np.ndarray:
    return np.concatenate([mask, mask[:, :-1][:, ::-1]], axis=1)


def _mirror_down(mask: np.ndarray) -> np.ndarray:
    return np.concatenate([mask, mask[:-1, :][::-1, :]], axis=0)


def _expand_fragment(mask: np.ndarray) -> np.ndarray:
    fill_right, fill_bottom = _classify_fragment(mask)
    expanded = mask.copy()
    if fill_right:
        expanded = _mirror_right(expanded)
    if fill_bottom:
        expanded = _mirror_down(expanded)
    center = (expanded.shape[0] // 2, expanded.shape[1] // 2)
    assert expanded.shape[0] % 2 == 1 and expanded.shape[1] % 2 == 1
    assert bool(expanded[center]), "the seed must lie on the completed pattern"
    return expanded


def _mask_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _place_masks(
    rng: random.Random,
    masks: list[np.ndarray],
    height: int,
    width: int,
    margin: int = MARGIN,
) -> list[tuple[int, int]] | None:
    cells_by_idx = [_mask_cells(mask) for mask in masks]
    order = sorted(
        range(len(masks)),
        key=lambda idx: (-len(cells_by_idx[idx]), -int(np.prod(masks[idx].shape))),
    )
    blocked: set[tuple[int, int]] = set()
    placements: list[tuple[int, int] | None] = [None] * len(masks)

    for idx in order:
        mask = masks[idx]
        mask_cells = cells_by_idx[idx]
        mh, mw = map(int, mask.shape)
        row_lo = margin
        row_hi = height - mh - margin
        col_lo = margin
        col_hi = width - mw - margin
        if row_lo > row_hi or col_lo > col_hi:
            return None

        candidates = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
        rng.shuffle(candidates)

        placed = False
        for top, left in candidates[:PLACEMENT_ATTEMPTS]:
            absolute = [(top + r, left + c) for r, c in mask_cells]
            if any(cell in blocked for cell in absolute):
                continue

            placements[idx] = (top, left)
            for rr, cc in absolute:
                for nr in range(rr - 1, rr + 2):
                    for nc in range(cc - 1, cc + 2):
                        if 0 <= nr < height and 0 <= nc < width:
                            blocked.add((nr, nc))
            placed = True
            break

        if not placed:
            return None

    assert all(pos is not None for pos in placements)
    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _make_left_fragment(rng: random.Random, max_dim: int) -> np.ndarray:
    width = rng.randint(2, max_dim)
    starts = [rng.randint(0, width - 1) for _ in range(3)]
    starts[2] = rng.randint(1, width - 1)
    if starts[0] > 0 and starts[1] > 0:
        starts[rng.randrange(2)] = 0

    mask = np.zeros((3, width), dtype=bool)
    for r, start in enumerate(starts):
        mask[r, start:] = True

    fill_right, fill_bottom = _classify_fragment(mask)
    assert fill_right and not fill_bottom
    return mask


def _make_top_fragment(rng: random.Random, max_dim: int) -> np.ndarray:
    height = rng.randint(2, max_dim)
    base = _make_left_fragment(rng, height)
    mask = base.T
    fill_right, fill_bottom = _classify_fragment(mask)
    assert fill_bottom and not fill_right
    return mask


def _make_quarter_fragment(rng: random.Random, max_dim: int) -> np.ndarray:
    height = rng.randint(2, max_dim)
    width = rng.randint(2, max_dim)
    starts = [rng.randint(0, width - 1) for _ in range(height)]
    starts[-1] = 0

    mask = np.zeros((height, width), dtype=bool)
    for r, start in enumerate(starts):
        mask[r, start:] = True

    if height * width > 4 and bool(mask[:-1, :-1].all()) and rng.random() < 0.8:
        row = rng.randrange(0, height - 1)
        col = rng.randrange(0, width - 1)
        while col >= starts[row]:
            starts[row] += 1
            col = -1
        mask[:] = False
        for r, start in enumerate(starts):
            mask[r, start:] = True

    fill_right, fill_bottom = _classify_fragment(mask)
    assert fill_right and fill_bottom
    return mask


def _make_fragment(kind: str, rng: random.Random, max_dim: int) -> np.ndarray:
    if kind == "quarter":
        return _make_quarter_fragment(rng, max_dim)
    if kind == "left":
        return _make_left_fragment(rng, max_dim)
    assert kind == "top"
    return _make_top_fragment(rng, max_dim)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unfold each left-panel fragment into its full symmetric pattern and redraw it around the matching right-panel seed.

    Input structure:
    - The grid has a dominant background color and one full-height separator column at the exact middle.
    - Left of the separator, each non-background color forms one small monochrome fragment inside a box no larger than 4x4.
    - Right of the separator, each fragment color appears exactly once as a singleton seed pixel marking the center of the completed pattern.

    Rule:
    - For every left-panel fragment, inspect its cropped bounding box.
    - If the rightmost column is full, mirror the fragment across that edge; if the bottom row is full, mirror it across that edge; if both are full, apply both mirrors.
    - This shared-edge mirroring keeps the fold edge as the symmetry axis, so each completed pattern has odd height and odd width.
    - Place the completed pattern in the right panel so its geometric center coincides with the matching seed pixel, and return only the right panel.

    Key insight:
    - The left shapes are not complete objects: the filled boundary edges reveal whether each one is a left half, a top half, or a top-left quarter of a larger symmetric figure, while the matching seed color tells where that figure must be reconstructed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = g.shape
    assert w % 2 == 1

    bg = _most_common_color(g)
    sep = w // 2
    sep_color = int(g[0, sep])
    assert sep_color != bg and bool(np.all(g[:, sep] == sep_color))

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    out = np.full(right.shape, bg, dtype=int)

    colors = [int(color) for color in np.unique(left) if int(color) != bg]
    for color in colors:
        fragment = _crop_color_mask(left, color)
        full = _expand_fragment(fragment)
        seed_cells = np.argwhere(right == color)
        assert seed_cells.shape[0] == 1, f"expected one seed for color {color}"
        seed_r, seed_c = map(int, seed_cells[0])
        top = seed_r - full.shape[0] // 2
        left_col = seed_c - full.shape[1] // 2

        for r, c in _mask_cells(full):
            rr = top + r
            cc = left_col + c
            assert 0 <= rr < right.shape[0] and 0 <= cc < right.shape[1], "completed pattern must fit the panel"
            assert int(out[rr, cc]) in (bg, color), "completed patterns must not overlap"
            out[rr, cc] = color

    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    object_kinds = ["quarter", "left", "top"]
    if rng.random() < 0.5:
        object_kinds.append(rng.choice(["quarter", "left", "top"]))
    rng.shuffle(object_kinds)

    max_dim = 4 if len(object_kinds) == 3 else 3
    fragments = [_make_fragment(kind, rng, max_dim) for kind in object_kinds]
    full_masks = [_expand_fragment(fragment) for fragment in fragments]

    panel_w = rng.randint(*RIGHT_PANEL_WIDTH_RANGE)
    if len(object_kinds) == 4:
        panel_w = max(panel_w, 12)
    panel_h = rng.randint(*GRID_HEIGHT_RANGE)
    if len(object_kinds) == 4:
        panel_h = max(panel_h, 13)

    if 2 * panel_w + 1 > 30 or panel_h > 30:
        return None

    right_positions = _place_masks(rng, full_masks, panel_h, panel_w)
    if right_positions is None:
        return None
    left_positions = _place_masks(rng, fragments, panel_h, panel_w)
    if left_positions is None:
        return None

    bg_choices = [color for color in PALETTE if color != SEPARATOR_COLOR]
    bg = rng.choice(bg_choices)
    object_colors = rng.sample(
        [color for color in PALETTE if color not in (bg, SEPARATOR_COLOR)],
        len(object_kinds),
    )

    right_panel = np.full((panel_h, panel_w), bg, dtype=int)
    seeds: list[tuple[int, int, int]] = []
    for color, mask, (top, left_col) in zip(object_colors, full_masks, right_positions):
        center_r = top + mask.shape[0] // 2
        center_c = left_col + mask.shape[1] // 2
        seeds.append((color, center_r, center_c))
        for r, c in _mask_cells(mask):
            right_panel[top + r, left_col + c] = color

    grid = np.full((panel_h, panel_w * 2 + 1), bg, dtype=int)
    grid[:, panel_w] = SEPARATOR_COLOR

    for color, fragment, (top, left_col) in zip(object_colors, fragments, left_positions):
        for r, c in _mask_cells(fragment):
            grid[top + r, left_col + c] = color

    for color, seed_r, seed_c in seeds:
        grid[seed_r, panel_w + 1 + seed_c] = color

    expected = right_panel
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
