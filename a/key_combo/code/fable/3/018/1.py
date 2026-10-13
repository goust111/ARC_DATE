from __future__ import annotations

import random

import numpy as np


PANEL_ORIGINS = (0, 1, 2, 3)  # UL, UR, LL, LR


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _count_components4(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    total = 0
    H, W = mask.shape
    for r0, c0 in np.argwhere(mask):
        r0 = int(r0)
        c0 = int(c0)
        if seen[r0, c0]:
            continue
        total += 1
        stack = [(r0, c0)]
        seen[r0, c0] = True
        while stack:
            r, c = stack.pop()
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < H and 0 <= cc < W and bool(mask[rr, cc]) and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
    return int(total)


def _find_separator(g: np.ndarray) -> tuple[int, int, int]:
    H, W = g.shape
    for color in map(int, np.unique(g)):
        full_rows = [r for r in range(H) if bool(np.all(g[r, :] == color))]
        full_cols = [c for c in range(W) if bool(np.all(g[:, c] == color))]
        if not full_rows or not full_cols:
            continue
        color_mask = g == color
        for r in full_rows:
            for c in full_cols:
                if 2 * r + 1 != H or 2 * c + 1 != W:
                    continue
                line_mask = np.zeros_like(color_mask, dtype=bool)
                line_mask[r, :] = True
                line_mask[:, c] = True
                if np.array_equal(color_mask, line_mask):
                    return int(r), int(c), int(color)
    raise AssertionError("expected one complete separator row and one complete separator column")


def _panel_views(g: np.ndarray, sep_r: int, sep_c: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        g[:sep_r, :sep_c],
        g[:sep_r, sep_c + 1 :],
        g[sep_r + 1 :, :sep_c],
        g[sep_r + 1 :, sep_c + 1 :],
    )


def _background_from_panels(panels: tuple[np.ndarray, ...]) -> int:
    common = set(map(int, np.unique(panels[0])))
    for panel in panels[1:]:
        common &= set(map(int, np.unique(panel)))
    assert common, "expected a shared panel background color"

    candidates: list[tuple[int, int]] = []
    for color in common:
        ok = True
        total_area = 0
        for panel in panels:
            other = set(map(int, np.unique(panel))) - {int(color)}
            if len(other) != 1:
                ok = False
                break
            total_area += int(np.count_nonzero(panel == int(color)))
        if ok:
            candidates.append((total_area, int(color)))

    assert candidates, "expected one background plus one foreground color in each panel"
    candidates.sort(reverse=True)
    return int(candidates[0][1])


def _panel_specs(panels: tuple[np.ndarray, ...], bg: int) -> list[dict]:
    specs = []
    for panel in panels:
        colors = set(map(int, np.unique(panel))) - {int(bg)}
        assert len(colors) == 1, "each panel must contain exactly one foreground color"
        color = int(next(iter(colors)))
        mask = panel == color
        count = _count_components4(mask)
        assert count > 0, "each panel must contain at least one object"
        specs.append({"panel": panel, "color": color, "mask": mask, "count": count})
    counts = [spec["count"] for spec in specs]
    assert len(set(counts)) == len(counts), "panel object counts must be distinct"
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite four separator-defined panels by object-count depth into one transparent overlay panel.

    Input structure:
    - One complete separator row and one complete separator column split the grid into four equal panels.
    - The separator color appears only on those two lines.
    - Each panel shares one background color and has exactly one foreground color forming 4-connected objects.
    - The four panels have distinct foreground object counts and aligned foreground positions overlap.

    Rule:
    - Remove the separators and align the four panels in the coordinate frame of a single panel.
    - Draw the panel with the most 4-connected foreground objects first, then continue in decreasing object count.
    - A later non-background pixel overwrites any earlier pixel at the same aligned coordinate.

    Key insight:
    - Layer depth is encoded by object count, not by panel position or color: fewer objects mean a later, higher layer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_r, sep_c, _sep_color = _find_separator(g)
    panels = _panel_views(g, sep_r, sep_c)
    shape = panels[0].shape
    assert shape[0] > 0 and shape[1] > 0, "panels must be nonempty"
    assert all(panel.shape == shape for panel in panels), "the separator must create four equal panels"

    bg = _background_from_panels(panels)
    specs = _panel_specs(panels, bg)

    out = np.full(shape, bg, dtype=int)
    for spec in sorted(specs, key=lambda s: int(s["count"]), reverse=True):
        mask = spec["mask"]
        out[mask] = spec["panel"][mask]
    return out


# ---------------- generator ----------------


_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)


def _sample_witnesses(h: int, w: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) >= 3 for rr, cc in chosen):
            chosen.append((int(r), int(c)))
            if len(chosen) == 4:
                return chosen
    return None


def _reserved_neighborhood(cells: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    reserved: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    reserved.add((int(rr), int(cc)))
    return reserved


def _can_place(mask: np.ndarray, shape: tuple[tuple[int, int], ...], r0: int, c0: int, reserved: set[tuple[int, int]]) -> bool:
    H, W = mask.shape
    for dr, dc in shape:
        r = int(r0 + dr)
        c = int(c0 + dc)
        if not (0 <= r < H and 0 <= c < W):
            return False
        if (r, c) in reserved or bool(mask[r, c]):
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < H and 0 <= cc < W and bool(mask[rr, cc]):
                    return False
    return True


def _place_extra_component(mask: np.ndarray, reserved: set[tuple[int, int]]) -> bool:
    H, W = mask.shape
    shapes = list(_SHAPES)
    for _ in range(300):
        shape = random.choice(shapes)
        max_r = max(r for r, _c in shape)
        max_c = max(c for _r, c in shape)
        r0 = random.randint(0, H - max_r - 1)
        c0 = random.randint(0, W - max_c - 1)
        if not _can_place(mask, shape, r0, c0, reserved):
            continue
        for dr, dc in shape:
            mask[r0 + dr, c0 + dc] = True
        return True
    return False


def _make_masks(
    h: int,
    w: int,
    counts_by_panel: list[int],
    rank_to_panel: list[int],
    witnesses: list[tuple[int, int]],
) -> list[np.ndarray] | None:
    masks = [np.zeros((h, w), dtype=bool) for _ in range(4)]
    for rank, panel_i in enumerate(rank_to_panel):
        for r, c in witnesses[: 4 - rank]:
            masks[panel_i][r, c] = True

    reserved = _reserved_neighborhood(witnesses, h, w)
    for panel_i, mask in enumerate(masks):
        target = int(counts_by_panel[panel_i])
        while _count_components4(mask) < target:
            if not _place_extra_component(mask, reserved):
                return None
        if _count_components4(mask) != target:
            return None
    return masks


def _overlay_from_masks(bg: int, masks: list[np.ndarray], colors: list[int], counts_by_panel: list[int]) -> np.ndarray:
    out = np.full(masks[0].shape, int(bg), dtype=int)
    order = sorted(range(4), key=lambda i: int(counts_by_panel[i]), reverse=True)
    for panel_i in order:
        out[masks[panel_i]] = int(colors[panel_i])
    return out


def _render_input(
    h: int,
    w: int,
    bg: int,
    sep: int,
    colors: list[int],
    masks: list[np.ndarray],
) -> np.ndarray:
    grid = np.full((2 * h + 1, 2 * w + 1), int(bg), dtype=int)
    grid[h, :] = int(sep)
    grid[:, w] = int(sep)
    origins = ((0, 0), (0, w + 1), (h + 1, 0), (h + 1, w + 1))
    for panel_i, (r0, c0) in enumerate(origins):
        panel = grid[r0 : r0 + h, c0 : c0 + w]
        panel[masks[panel_i]] = int(colors[panel_i])
    return grid


def _validate_generated(
    inp: np.ndarray,
    expected: np.ndarray,
    bg: int,
    sep: int,
    colors: list[int],
    counts_by_panel: list[int],
    rank_to_panel: list[int],
    witnesses: list[tuple[int, int]],
) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    try:
        sep_r, sep_c, found_sep = _find_separator(inp)
        panels = _panel_views(inp, sep_r, sep_c)
        found_bg = _background_from_panels(panels)
        specs = _panel_specs(panels, found_bg)
        produced = main(inp)
    except Exception:
        return False

    if int(found_bg) != int(bg) or int(found_sep) != int(sep):
        return False
    if not np.array_equal(produced, expected):
        return False
    if [int(spec["color"]) for spec in specs] != [int(c) for c in colors]:
        return False
    if [int(spec["count"]) for spec in specs] != [int(c) for c in counts_by_panel]:
        return False

    # Witnesses force all four layers to be necessary: all four overlap at the
    # first witness, then progressively fewer lower layers overlap at each next one.
    for rank, panel_i in enumerate(rank_to_panel):
        present_count = 4 - rank
        mask = specs[panel_i]["mask"]
        for witness_i, (r, c) in enumerate(witnesses):
            should_exist = witness_i < present_count
            if bool(mask[r, c]) != should_exist:
                return False
    for witness_i, (r, c) in enumerate(witnesses):
        visible_rank = 3 - witness_i
        visible_panel = rank_to_panel[visible_rank]
        if int(expected[r, c]) != int(colors[visible_panel]):
            return False

    visible_colors = set(map(int, np.unique(expected))) - {int(bg)}
    return visible_colors == set(map(int, colors))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(9, 14)
    w = random.randint(9, 14)
    witnesses = _sample_witnesses(h, w)
    if witnesses is None:
        return None

    palette = list(range(10))
    bg = random.choice(palette)
    remaining = [c for c in palette if c != bg]
    sep = random.choice(remaining)
    panel_colors = random.sample([c for c in remaining if c != sep], 4)

    counts_desc = sorted(random.sample(range(1, 7), 4), reverse=True)
    fixed_orders = {
        (0, 1, 2, 3),
        (0, 3, 2, 1),
        (0, 2, 1, 3),
        (3, 2, 1, 0),
    }
    rank_to_panel = random.sample(list(PANEL_ORIGINS), 4)
    if tuple(rank_to_panel) in fixed_orders:
        return None

    counts_by_panel = [0, 0, 0, 0]
    for rank, panel_i in enumerate(rank_to_panel):
        counts_by_panel[panel_i] = int(counts_desc[rank])

    masks = _make_masks(h, w, counts_by_panel, rank_to_panel, witnesses)
    if masks is None:
        return None

    expected = _overlay_from_masks(bg, masks, panel_colors, counts_by_panel)
    inp = _render_input(h, w, bg, sep, panel_colors, masks)
    if not _validate_generated(inp, expected, bg, sep, panel_colors, counts_by_panel, rank_to_panel, witnesses):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid count-depth overlay input")
