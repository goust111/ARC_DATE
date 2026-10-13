from __future__ import annotations

from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()


def _component_count(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        count += 1
        stack = [(r, c)]
        seen[r, c] = True
        while stack:
            cr, cc = stack.pop()
            for dr, dc in DIR4:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return count


def _bbox_area(mask: np.ndarray) -> int:
    rows, cols = np.where(mask)
    return int((rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1))


def _separator_row(grid: np.ndarray) -> int:
    rows = np.where(np.all(grid == GREY, axis=1))[0]
    assert len(rows) == 1, f"expected exactly one grey separator row, got {rows.tolist()}"
    return int(rows[0])


def _base_and_motif(panel: np.ndarray) -> tuple[int, int, np.ndarray]:
    colors = [int(c) for c in np.unique(panel)]
    assert len(colors) == 2, f"panel must contain exactly two colors, got {colors}"
    counts = Counter(map(int, panel.flat))
    (base, base_count), (motif, motif_count) = counts.most_common(2)
    assert base_count > motif_count, "panel base color must be dominant"
    mask = panel == motif
    assert _component_count(mask) == 1, "motif mask must be a single 4-connected component"
    return int(base), int(motif), mask


def _swap_panels(top: np.ndarray, bottom: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    top_base, _, top_mask = _base_and_motif(top)
    bottom_base, _, bottom_mask = _base_and_motif(bottom)
    assert top_base != bottom_base, "panel bases must differ so the swap stays visible"

    new_top = np.full_like(top, top_base)
    new_bottom = np.full_like(bottom, bottom_base)
    new_top[bottom_mask] = bottom_base
    new_bottom[top_mask] = top_base

    p = int(np.count_nonzero(new_top != top_base))
    assert p > 0
    return new_top, new_bottom, p


def main(grid: np.ndarray) -> np.ndarray:
    """Swap the two panel motif masks, count the swapped top foreground, and tile the swapped bottom that many times.

    Input structure:
    - A 2D grid is split by exactly one full grey separator row into two same-size rectangular panels.
    - Each panel contains exactly two non-black colors: one dominant base color and one minority motif color.
    - The minority color in each panel forms a single irregular 4-connected motif mask.

    Rule:
    - Remove the grey separator and treat the regions above and below it as the top and bottom panels.
    - Clear each panel to its own base color, then paste the other panel's motif mask into it using the other panel's base color.
    - Let P be the number of non-base pixels in the swapped top panel.
    - Output a rectangular grid whose top half is the swapped top panel placed at the left of a black canvas, and whose bottom half is the swapped bottom panel tiled horizontally P times.

    Key insight:
    - The repetition count is not read directly from the input; it is created by the cross-panel swap, because the swapped top foreground count equals the exchanged bottom motif size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    sep = _separator_row(g)
    top = g[:sep, :]
    bottom = g[sep + 1 :, :]
    assert top.shape == bottom.shape and top.size > 0, "expected two equal-size panels around the separator"

    swapped_top, swapped_bottom, p = _swap_panels(top, bottom)
    h, w = top.shape
    out_w = int(w * p)
    out = np.full((2 * h, out_w), BLACK, dtype=int)
    out[:h, :w] = swapped_top
    out[h:, :] = np.tile(swapped_bottom, (1, p))
    return out


def _random_connected_mask(h: int, w: int, size: int, *, prefer_border_touch: bool) -> np.ndarray | None:
    margin = 0 if prefer_border_touch else 1
    if h - 2 * margin <= 0 or w - 2 * margin <= 0:
        return None

    valid = {(r, c) for r in range(margin, h - margin) for c in range(margin, w - margin)}
    if len(valid) < size:
        return None

    for _ in range(80):
        start = tuple(list(valid)[int(RNG.integers(len(valid)))])
        cells = {start}
        frontier = set()
        for dr, dc in DIR4:
            cand = (start[0] + dr, start[1] + dc)
            if cand in valid:
                frontier.add(cand)

        while len(cells) < size and frontier:
            idx = int(RNG.integers(len(frontier)))
            cell = list(frontier)[idx]
            frontier.remove(cell)
            cells.add(cell)
            for dr, dc in DIR4:
                cand = (cell[0] + dr, cell[1] + dc)
                if cand in valid and cand not in cells:
                    frontier.add(cand)

        if len(cells) != size:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if _component_count(mask) != 1:
            continue
        if _bbox_area(mask) == size:
            continue
        rows, cols = np.where(mask)
        if len(set(map(int, rows))) == 1 or len(set(map(int, cols))) == 1:
            continue
        return mask
    return None


def _make_panel(h: int, w: int, base: int, motif: int, size: int, *, prefer_border_touch: bool) -> np.ndarray | None:
    mask = _random_connected_mask(h, w, size, prefer_border_touch=prefer_border_touch)
    if mask is None:
        return None
    panel = np.full((h, w), int(base), dtype=int)
    panel[mask] = int(motif)
    return panel


def _build_wrong_count_output(swapped_top: np.ndarray, swapped_bottom: np.ndarray, count: int) -> np.ndarray:
    h, w = swapped_top.shape
    out = np.full((2 * h, w * count), BLACK, dtype=int)
    out[:h, :w] = swapped_top
    out[h:, :] = np.tile(swapped_bottom, (1, count))
    return out


def _validate_example(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30

    sep = _separator_row(grid)
    top = grid[:sep, :]
    bottom = grid[sep + 1 :, :]
    assert top.shape == bottom.shape
    assert top.shape[0] in (4, 5)
    assert top.shape[1] in (6, 7, 8)

    top_base, top_motif, top_mask = _base_and_motif(top)
    bottom_base, bottom_motif, bottom_mask = _base_and_motif(bottom)
    assert len({top_base, top_motif, bottom_base, bottom_motif}) == 4

    top_area = int(top_mask.sum())
    bottom_area = int(bottom_mask.sum())
    assert 3 <= top_area <= 6
    assert 3 <= bottom_area <= 4
    assert top_area != bottom_area

    swapped_top, swapped_bottom, p = _swap_panels(top, bottom)
    assert p == bottom_area
    assert p in (3, 4)

    out = main(grid)
    assert out.ndim == 2 and max(out.shape) <= 30
    assert out.shape == (2 * top.shape[0], p * top.shape[1])
    assert np.array_equal(out[: top.shape[0], : top.shape[1]], swapped_top)
    assert np.all(out[: top.shape[0], top.shape[1] :] == BLACK)
    assert np.array_equal(out[top.shape[0] :, :], np.tile(swapped_bottom, (1, p)))

    wrong = _build_wrong_count_output(swapped_top, swapped_bottom, top_area)
    assert not np.array_equal(out, wrong), "example degenerates to counting the original top motif"


def generate_input() -> np.ndarray:
    for _ in range(400):
        bottom_area = int(RNG.choice([3, 4]))
        h = int(RNG.choice([4, 5]))
        w_max = min(8, 30 // bottom_area)
        w = int(RNG.integers(6, w_max + 1))

        top_area_choices = [area for area in (3, 4, 5, 6) if area != bottom_area]
        top_area = int(RNG.choice(top_area_choices))

        colors = RNG.choice(PALETTE, size=4, replace=False)
        top_base, bottom_base, top_motif, bottom_motif = map(int, colors)

        top = _make_panel(h, w, top_base, top_motif, top_area, prefer_border_touch=bool(RNG.integers(2)))
        bottom = _make_panel(h, w, bottom_base, bottom_motif, bottom_area, prefer_border_touch=bool(RNG.integers(2)))
        if top is None or bottom is None:
            continue

        grid = np.full((2 * h + 1, w), BLACK, dtype=int)
        grid[:h, :] = top
        grid[h, :] = GREY
        grid[h + 1 :, :] = bottom

        try:
            _validate_example(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle input")
