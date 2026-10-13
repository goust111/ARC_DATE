from __future__ import annotations

import numpy as np
import random
from arc_utils import paste


def _bands(indices: np.ndarray) -> list[tuple[int, int]]:
    assert indices.size > 0, "expected at least one non-separator row/column"
    vals = [int(x) for x in indices]
    bands: list[tuple[int, int]] = []
    start = prev = vals[0]
    for x in vals[1:]:
        if x == prev + 1:
            prev = x
        else:
            bands.append((start, prev))
            start = prev = x
    bands.append((start, prev))
    return bands


def _separator_color(g: np.ndarray) -> int:
    candidates = []
    for color in np.unique(g):
        has_full_row = bool(np.any(np.all(g == int(color), axis=1)))
        has_full_col = bool(np.any(np.all(g == int(color), axis=0)))
        if has_full_row and has_full_col:
            candidates.append(int(color))
    assert len(candidates) == 1, f"expected unique row/column separator color, got {candidates}"
    return candidates[0]


def _base_and_motif_mask(panel: np.ndarray) -> tuple[int, np.ndarray]:
    colors, counts = np.unique(panel, return_counts=True)
    assert len(colors) == 2, f"expected a two-color panel, got {colors.tolist()}"
    max_count = int(counts.max())
    assert int(np.sum(counts == max_count)) == 1, "panel base color must be structurally unique"
    base = int(colors[int(np.argmax(counts))])
    return base, panel != base


def main(grid: np.ndarray) -> np.ndarray:
    """
    Within each horizontal pair of two-color panels, swap the motif masks and recolor each pasted mask with the source panel's base color.

    Input structure:
    - A separator color forms full blank rows and at least one full blank column.
    - The non-separator area is split into row bands, each containing exactly two same-shaped rectangular panels.
    - Each panel has a dominant base color and one minority motif color.

    Rule:
    - For every row band, clear each panel to its own base color.
    - Paste the right panel's motif mask into the left panel using the right base color.
    - Paste the left panel's motif mask into the right panel using the left base color.

    Key insight:
    - The motif colors themselves are distractors; only each motif's shape mask and the two panel base colors are exchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    sep = _separator_color(g)
    row_bands = _bands(np.where(~np.all(g == sep, axis=1))[0])
    col_bands = _bands(np.where(~np.all(g == sep, axis=0))[0])
    assert len(col_bands) == 2, f"expected exactly two panel columns, got {len(col_bands)}"

    out = g.copy()
    (lc0, lc1), (rc0, rc1) = col_bands
    for r0, r1 in row_bands:
        left = g[r0 : r1 + 1, lc0 : lc1 + 1]
        right = g[r0 : r1 + 1, rc0 : rc1 + 1]
        assert left.shape == right.shape, "paired panels must have the same shape"

        left_base, left_motif = _base_and_motif_mask(left)
        right_base, right_motif = _base_and_motif_mask(right)

        new_left = np.full(left.shape, left_base, dtype=int)
        new_right = np.full(right.shape, right_base, dtype=int)
        new_left[right_motif] = right_base
        new_right[left_motif] = left_base

        out[r0 : r1 + 1, lc0 : lc1 + 1] = new_left
        out[r0 : r1 + 1, rc0 : rc1 + 1] = new_right

    return out

# --------------- generator (att2, o5x) ---------------

# Canvas and separator geometry.
NUM_BANDS_MIN, NUM_BANDS_MAX = 2, 4
PANEL_H_MIN, PANEL_H_MAX = 6, 8
PANEL_W_MIN, PANEL_W_MAX = 7, 8
SEPARATOR_H = 2
SEPARATOR_W_MIN, SEPARATOR_W_MAX = 1, 2
MAX_GRID_SIZE = 30

# Motif size and component growth.
MOTIF_MARGIN = 1
MOTIF_FILL_MIN, MOTIF_FILL_MAX = 0.10, 0.30
MAX_MOTIF_COMPONENTS = 4
COMPONENT_WEIGHTS = (0.68, 0.16, 0.10, 0.06)
PART_MIN_SIZE, PART_WEIGHT_OFFSET = 2, 0.2

# Color sampling and retry limits.
PALETTE = tuple(range(1, 10))
SAME_BASE_PROB = 0.25
GENERATION_RETRIES = 300


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if MOTIF_MARGIN <= rr < h - MOTIF_MARGIN and MOTIF_MARGIN <= cc < w - MOTIF_MARGIN:
            out.append((rr, cc))
    return out


def _partition(total: int, parts: int) -> list[int]:
    mins = [PART_MIN_SIZE] * parts if total >= PART_MIN_SIZE * parts else [1] * parts
    remaining = total - sum(mins)
    weights = [random.random() + PART_WEIGHT_OFFSET for _ in range(parts)]
    sizes = mins[:]
    for _ in range(remaining):
        i = random.choices(range(parts), weights=weights, k=1)[0]
        sizes[i] += 1
    random.shuffle(sizes)
    return sizes


def _touches_other(label: np.ndarray, r: int, c: int, own: int) -> bool:
    h, w = label.shape
    for rr, cc in _neighbors4(r, c, h, w):
        v = int(label[rr, cc])
        if v not in (0, own):
            return True
    return False


def _grow_mask(h: int, w: int, target: int, n_comp: int) -> np.ndarray | None:
    label = np.zeros((h, w), dtype=int)
    cells = [(r, c) for r in range(MOTIF_MARGIN, h - MOTIF_MARGIN) for c in range(MOTIF_MARGIN, w - MOTIF_MARGIN)]
    random.shuffle(cells)

    seeds: list[tuple[int, int]] = []
    for r, c in cells:
        if all(abs(r - sr) + abs(c - sc) > 1 for sr, sc in seeds):
            seeds.append((r, c))
            if len(seeds) == n_comp:
                break
    if len(seeds) != n_comp:
        return None

    sizes = _partition(target, n_comp)
    comps: list[set[tuple[int, int]]] = []
    for i, (r, c) in enumerate(seeds, start=1):
        label[r, c] = i
        comps.append({(r, c)})

    for i, size in enumerate(sizes, start=1):
        while len(comps[i - 1]) < size:
            frontier = set()
            for rr, cc in comps[i - 1]:
                frontier.update(_neighbors4(rr, cc, h, w))
            choices = [
                cell
                for cell in frontier
                if label[cell] == 0 and not _touches_other(label, cell[0], cell[1], i)
            ]
            if not choices:
                return None
            nr, nc = random.choice(choices)
            label[nr, nc] = i
            comps[i - 1].add((nr, nc))

    return label > 0


def _random_motif_mask(h: int, w: int) -> np.ndarray | None:
    area = h * w
    lo = max(1, int(np.ceil(MOTIF_FILL_MIN * area)))
    hi = min((h - 2 * MOTIF_MARGIN) * (w - 2 * MOTIF_MARGIN), int(np.floor(MOTIF_FILL_MAX * area)))
    if lo > hi:
        return None
    target = random.randint(lo, hi)

    max_comp = min(MAX_MOTIF_COMPONENTS, target)
    n_comp = random.choices(range(1, max_comp + 1), weights=COMPONENT_WEIGHTS[:max_comp], k=1)[0]
    return _grow_mask(h, w, target, n_comp)


def _pair_colors() -> tuple[int, int, int, int]:
    colors = list(PALETTE)
    if random.random() < SAME_BASE_PROB:
        base = random.choice(colors)
        left_motif, right_motif = random.sample([c for c in colors if c != base], 2)
        return base, base, left_motif, right_motif
    left_base, right_base, left_motif, right_motif = random.sample(colors, 4)
    return left_base, right_base, left_motif, right_motif


def _choose_layout() -> tuple[int, int, int, int, int]:
    num_bands = random.randint(NUM_BANDS_MIN, NUM_BANDS_MAX)
    panel_h = random.choice([h for h in range(PANEL_H_MIN, PANEL_H_MAX + 1) if num_bands * h + SEPARATOR_H * (num_bands - 1) <= MAX_GRID_SIZE])
    panel_w = random.randint(PANEL_W_MIN, PANEL_W_MAX)
    sep_w = random.randint(SEPARATOR_W_MIN, SEPARATOR_W_MAX)
    return num_bands, panel_h, panel_w, SEPARATOR_H, sep_w


def _make_pair_panels(panel_h: int, panel_w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Build the input and output panels for one paired row band."""
    left_base, right_base, left_motif_color, right_motif_color = _pair_colors()
    left_mask = _random_motif_mask(panel_h, panel_w)
    right_mask = _random_motif_mask(panel_h, panel_w)
    if left_mask is None or right_mask is None:
        return None

    in_left = np.full((panel_h, panel_w), left_base, dtype=int)
    in_right = np.full((panel_h, panel_w), right_base, dtype=int)
    out_left = np.full((panel_h, panel_w), left_base, dtype=int)
    out_right = np.full((panel_h, panel_w), right_base, dtype=int)
    in_left[left_mask] = left_motif_color
    in_right[right_mask] = right_motif_color
    out_left[right_mask] = right_base
    out_right[left_mask] = left_base
    return in_left, in_right, out_left, out_right


def _gen_attempt() -> np.ndarray | None:
    num_bands, panel_h, panel_w, sep_h, sep_w = _choose_layout()
    height = num_bands * panel_h + (num_bands - 1) * sep_h
    width = 2 * panel_w + sep_w
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)
    right_c0 = panel_w + sep_w

    for band in range(num_bands):
        r0 = band * (panel_h + sep_h)
        panels = _make_pair_panels(panel_h, panel_w)
        if panels is None:
            return None
        in_left, in_right, out_left, out_right = panels

        paste(inp, in_left, r0, 0)
        paste(inp, in_right, r0, right_c0)
        paste(out, out_left, r0, 0)
        paste(out, out_right, r0, right_c0)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5a719d11 input")
