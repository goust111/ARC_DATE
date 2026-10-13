from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Tuple

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
SEP = 5
ORANGE = 7
BASE_COLORS = (BLUE, RED, GREEN, YELLOW)
PANEL_ORDER = ("tl", "tr", "bl", "br")
PANEL_TO_RC = {"tl": (0, 0), "tr": (0, 1), "bl": (1, 0), "br": (1, 1)}
CORNER_COORDS = {
    "tl": (0, 0),
    "tr": (0, -1),
    "bl": (-1, 0),
    "br": (-1, -1),
}
DIAGONALS = (("tl", "br"), ("tr", "bl"))
GEN_ATTEMPTS = 400


def _bands(idxs: Iterable[int]) -> List[Tuple[int, int]]:
    vals = sorted(int(x) for x in idxs)
    if not vals:
        return []
    out: List[Tuple[int, int]] = []
    start = prev = vals[0]
    for x in vals[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append((start, prev))
        start = prev = x
    out.append((start, prev))
    return out


def _bbox(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _colors_present(grid: np.ndarray, exclude: Iterable[int] = ()) -> List[int]:
    cols = set(map(int, grid.flatten()))
    cols -= set(map(int, exclude))
    return sorted(cols)


def _panel_signature(mask: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    return tuple(map(tuple, np.argwhere(mask)))


def _corner_value(grid: np.ndarray, corner: str) -> int:
    r, c = CORNER_COORDS[corner]
    return int(grid[r, c])


def _set_corner(grid: np.ndarray, corner: str, color: int) -> None:
    r, c = CORNER_COORDS[corner]
    grid[r, c] = int(color)


def _extract_guides(grid: np.ndarray) -> List[Tuple[str, int]]:
    guides: List[Tuple[str, int]] = []
    for corner in PANEL_ORDER:
        color = _corner_value(grid, corner)
        if color != BG:
            guides.append((corner, color))
    return guides


def _panel_layout(grid: np.ndarray) -> Dict[str, Tuple[int, int, int, int]]:
    work = np.asarray(grid, dtype=int).copy()
    for corner, _color in _extract_guides(work):
        _set_corner(work, corner, BG)

    box = _bbox(work != BG)
    assert box is not None, "expected a non-empty panel assembly"
    r0, c0, r1, c1 = box
    crop = work[r0 : r1 + 1, c0 : c1 + 1]

    sep_rows = [i for i in range(crop.shape[0]) if np.all(crop[i, :] == SEP)]
    sep_cols = [j for j in range(crop.shape[1]) if np.all(crop[:, j] == SEP)]
    assert sep_rows and sep_cols, "expected grey separator row and column"

    row_bands = _bands(i for i in range(crop.shape[0]) if i not in set(sep_rows))
    col_bands = _bands(j for j in range(crop.shape[1]) if j not in set(sep_cols))
    assert len(row_bands) == 2 and len(col_bands) == 2, "expected a 2x2 panel layout"

    (rt0, rt1), (rb0, rb1) = row_bands
    (cl0, cl1), (cr0, cr1) = col_bands
    top_h = rt1 - rt0
    bot_h = rb1 - rb0
    left_w = cl1 - cl0
    right_w = cr1 - cr0
    assert top_h == bot_h and left_w == right_w, "panels must be equal sized"

    return {
        "tl": (r0 + rt0, c0 + cl0, r0 + rt1, c0 + cl1),
        "tr": (r0 + rt0, c0 + cr0, r0 + rt1, c0 + cr1),
        "bl": (r0 + rb0, c0 + cl0, r0 + rb1, c0 + cl1),
        "br": (r0 + rb0, c0 + cr0, r0 + rb1, c0 + cr1),
    }


def _crop(grid: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _panel_base_and_mask(panel: np.ndarray) -> Tuple[int, np.ndarray]:
    cols = _colors_present(panel, exclude={BG})
    assert ORANGE in cols and len(cols) == 2, "panel must contain one base color and one orange motif"
    base = next(col for col in cols if col != ORANGE)
    mask = panel == ORANGE
    assert mask.any(), "motif must be non-empty"
    return int(base), mask


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    guides = _extract_guides(g)
    assert len(guides) == 2, "expected exactly two guide pixels"
    guide_colors = [color for _corner, color in guides]
    assert len(set(guide_colors)) == 2, "guide colors must be distinct"
    assert all(color in BASE_COLORS for color in guide_colors), "guide colors must match panel base colors"

    layout = _panel_layout(g)
    bases: Dict[str, int] = {}
    masks: Dict[str, np.ndarray] = {}
    for pos, box in layout.items():
        base, mask = _panel_base_and_mask(_crop(g, box))
        bases[pos] = base
        masks[pos] = mask

    assert sorted(bases.values()) == sorted(BASE_COLORS), "panel base colors must be blue/red/green/yellow once each"
    base_to_panel = {base: pos for pos, base in bases.items()}
    selected = tuple(base_to_panel[color] for _corner, color in guides)
    assert len(set(selected)) == 2, "guides must target different panels"
    p0, p1 = selected
    pr0, pc0 = PANEL_TO_RC[p0]
    pr1, pc1 = PANEL_TO_RC[p1]
    assert pr0 != pr1 and pc0 != pc1, "selected panels must be diagonal"

    return {
        "grid": g,
        "guides": guides,
        "layout": layout,
        "bases": bases,
        "masks": masks,
        "selected": selected,
    }


def _render_swap(grid: np.ndarray, selected: Tuple[str, str]) -> np.ndarray:
    info = _analyze(grid)
    g = info["grid"]
    out = g.copy()
    for corner, _color in info["guides"]:
        _set_corner(out, corner, BG)

    a, b = selected
    panels = info["layout"]
    bases = info["bases"]
    masks = info["masks"]
    ar0, ac0, ar1, ac1 = panels[a]
    br0, bc0, br1, bc1 = panels[b]
    out[ar0 : ar1 + 1, ac0 : ac1 + 1] = bases[a]
    out[br0 : br1 + 1, bc0 : bc1 + 1] = bases[b]
    out[ar0 : ar1 + 1, ac0 : ac1 + 1][masks[b]] = bases[b]
    out[br0 : br1 + 1, bc0 : bc1 + 1][masks[a]] = bases[a]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the orange motif masks between two color-matched diagonal panels, then erase the corner guides.

    Input structure:
    - A black background contains a 2x2 block of equal-sized rectangular panels separated by full grey bars.
    - The four panels use the four base colors blue, red, green, and yellow exactly once, and each panel contains one orange motif.
    - Exactly two outer canvas corners contain single guide pixels whose colors match two panel base colors.

    Rule:
    - Read the two non-black corner guide pixels and identify the two panels whose base colors match those guide colors.
    - Extract the orange motif mask from each matched panel using panel-local coordinates.
    - Clear each matched panel back to its own base color, then paste the other matched panel's mask into it and recolor that pasted mask with the source panel's base color.
    - Leave the two unselected panels and all grey separator bars unchanged, and erase both guide pixels to black.

    Key insight:
    - The corner pixels are disposable routing instructions: their colors choose which two diagonal panels exchange motif shapes, while the exchanged motifs keep the source panels' base colors.
    """
    info = _analyze(grid)
    return _render_swap(np.asarray(grid, dtype=int), tuple(info["selected"]))


def _neighbors4(cell: Tuple[int, int], h: int, w: int) -> List[Tuple[int, int]]:
    r, c = cell
    out: List[Tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(coords[0])
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in _neighbors4(cur, *mask.shape):
            if nxt not in seen and mask[nxt]:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == int(mask.sum())


def _random_mask(panel_h: int, panel_w: int) -> np.ndarray | None:
    inner_h = panel_h - 2
    inner_w = panel_w - 2
    if inner_h <= 0 or inner_w <= 0:
        return None
    max_cells = min(10, inner_h * inner_w)
    if max_cells < 4:
        return None

    for _ in range(120):
        target = int(np.random.randint(4, max_cells + 1))
        mask = np.zeros((panel_h, panel_w), dtype=bool)
        start = (int(np.random.randint(1, panel_h - 1)), int(np.random.randint(1, panel_w - 1)))
        mask[start] = True
        frontier = {start}
        while int(mask.sum()) < target and frontier:
            cell = list(frontier)[int(np.random.randint(len(frontier)))]
            candidates = [n for n in _neighbors4(cell, panel_h, panel_w) if 0 < n[0] < panel_h - 1 and 0 < n[1] < panel_w - 1 and not mask[n]]
            if not candidates:
                frontier.remove(cell)
                continue
            nxt = candidates[int(np.random.randint(len(candidates)))]
            mask[nxt] = True
            frontier.add(nxt)
            frontier.add(cell)

        if int(mask.sum()) != target or not _connected(mask):
            continue
        if np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1]):
            continue
        box = _bbox(mask)
        assert box is not None
        r0, c0, r1, c1 = box
        box_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if box_area == int(mask.sum()):
            continue
        if (r1 - r0 + 1) == 1 or (c1 - c0 + 1) == 1:
            continue
        return mask
    return None


def _place_panel(
    grid: np.ndarray,
    box: Tuple[int, int, int, int],
    base: int,
    mask: np.ndarray,
) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(base)
    grid[r0 : r1 + 1, c0 : c1 + 1][mask] = ORANGE


def _build_grid(
    panel_h: int,
    panel_w: int,
    sep_h: int,
    sep_w: int,
    margins: Tuple[int, int, int, int],
    base_by_panel: Dict[str, int],
    mask_by_panel: Dict[str, np.ndarray],
    guide_mapping: Dict[str, int],
) -> np.ndarray:
    top, bottom, left, right = margins
    height = top + panel_h + sep_h + panel_h + bottom
    width = left + panel_w + sep_w + panel_w + right
    grid = np.zeros((height, width), dtype=int)

    row_top = top
    row_bottom = top + panel_h + sep_h
    col_left = left
    col_right = left + panel_w + sep_w

    grid[top + panel_h : top + panel_h + sep_h, left : left + panel_w + sep_w + panel_w] = SEP
    grid[top : top + panel_h + sep_h + panel_h, left + panel_w : left + panel_w + sep_w] = SEP

    boxes = {
        "tl": (row_top, col_left, row_top + panel_h - 1, col_left + panel_w - 1),
        "tr": (row_top, col_right, row_top + panel_h - 1, col_right + panel_w - 1),
        "bl": (row_bottom, col_left, row_bottom + panel_h - 1, col_left + panel_w - 1),
        "br": (row_bottom, col_right, row_bottom + panel_h - 1, col_right + panel_w - 1),
    }
    for pos in PANEL_ORDER:
        _place_panel(grid, boxes[pos], base_by_panel[pos], mask_by_panel[pos])
    for corner, color in guide_mapping.items():
        _set_corner(grid, corner, color)
    return grid


def _guide_pair_for_selected(selected: Tuple[str, str]) -> Tuple[str, str]:
    if set(selected) == {"tl", "br"}:
        return ("tr", "bl")
    return ("tl", "br")


def _validate_example(grid: np.ndarray) -> bool:
    info = _analyze(grid)
    out = main(grid)

    sigs = [_panel_signature(info["masks"][pos]) for pos in PANEL_ORDER]
    if len(set(sigs)) != 4:
        return False

    selected = tuple(info["selected"])
    if set(selected) not in ({"tl", "br"}, {"tr", "bl"}):
        return False
    guide_corners = tuple(corner for corner, _color in info["guides"])
    if set(guide_corners) == set(selected):
        return False

    selected_areas = [int(info["masks"][pos].sum()) for pos in selected]
    if selected_areas[0] == selected_areas[1]:
        return False

    wrong = _render_swap(grid, guide_corners)
    if np.array_equal(out, wrong):
        return False
    if np.array_equal(out, grid):
        return False

    for pos in PANEL_ORDER:
        in_panel = _crop(grid, info["layout"][pos])
        out_panel = _crop(out, info["layout"][pos])
        if pos not in selected and not np.array_equal(in_panel, out_panel):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    panel_h = int(np.random.randint(5, 9))
    panel_w = int(np.random.randint(5, 9))
    sep_h = int(np.random.randint(1, 3))
    sep_w = int(np.random.randint(1, 3))
    margins = tuple(int(np.random.randint(1, 4)) for _ in range(4))
    top, bottom, left, right = margins
    height = top + panel_h + sep_h + panel_h + bottom
    width = left + panel_w + sep_w + panel_w + right
    if max(height, width) > 30:
        return None

    masks: Dict[str, np.ndarray] = {}
    seen = set()
    for pos in PANEL_ORDER:
        mask = _random_mask(panel_h, panel_w)
        if mask is None:
            return None
        sig = _panel_signature(mask)
        if sig in seen:
            return None
        seen.add(sig)
        masks[pos] = mask

    base_by_panel = {pos: color for pos, color in zip(PANEL_ORDER, np.random.permutation(BASE_COLORS))}
    selected = DIAGONALS[int(np.random.randint(2))]
    guide_corners = _guide_pair_for_selected(selected)

    selected_colors = [base_by_panel[pos] for pos in selected]
    np.random.shuffle(selected_colors)
    guide_mapping = {guide_corners[0]: int(selected_colors[0]), guide_corners[1]: int(selected_colors[1])}

    grid = _build_grid(panel_h, panel_w, sep_h, sep_w, margins, base_by_panel, masks, guide_mapping)
    if not _validate_example(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
