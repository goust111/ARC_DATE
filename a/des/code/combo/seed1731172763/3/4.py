from __future__ import annotations

from fractions import Fraction
import random

import numpy as np


BG = 0
SEP = 9
FG_COLORS = tuple(range(1, 9))

N_CHOICES = (3, 4, 5)
PANEL_SIZE_RANGES = {
    3: ((4, 8), (4, 8)),
    4: ((4, 6), (4, 6)),
    5: ((3, 4), (3, 4)),
}

MAX_MASK_ATTEMPTS = 60
MAX_PANEL_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 400


def _find_layout(g: np.ndarray) -> tuple[int, int, int, int]:
    height, width = map(int, g.shape)
    candidates: list[tuple[int, int, int, int]] = []

    for color in map(int, np.unique(g)):
        if color == BG:
            continue

        sep_rows = [int(r) for r in np.where(np.all(g == color, axis=1))[0]]
        sep_cols = [int(c) for c in np.where(np.all(g == color, axis=0))[0]]
        if not sep_rows or len(sep_rows) != len(sep_cols):
            continue

        n = len(sep_rows) + 1
        if (height - len(sep_rows)) % n != 0 or (width - len(sep_cols)) % n != 0:
            continue

        panel_h = (height - len(sep_rows)) // n
        panel_w = (width - len(sep_cols)) // n
        if panel_h <= 0 or panel_w <= 0:
            continue

        expect_rows = [panel_h + i * (panel_h + 1) for i in range(n - 1)]
        expect_cols = [panel_w + i * (panel_w + 1) for i in range(n - 1)]
        if sep_rows == expect_rows and sep_cols == expect_cols:
            candidates.append((color, n, panel_h, panel_w))

    assert len(candidates) == 1, "expected a unique separator layout"
    return candidates[0]


def _panel_slices(n: int, panel_h: int, panel_w: int) -> list[tuple[slice, slice]]:
    return [
        (
            slice(r * (panel_h + 1), r * (panel_h + 1) + panel_h),
            slice(c * (panel_w + 1), c * (panel_w + 1) + panel_w),
        )
        for r in range(n)
        for c in range(n)
    ]


def _centroid_key(panel: np.ndarray) -> tuple[Fraction, Fraction]:
    coords = np.argwhere(panel != BG)
    assert len(coords) > 0, "non-empty panel expected"

    count = len(coords)
    row_sum = int(coords[:, 0].sum())
    col_sum = int(coords[:, 1].sum())
    return Fraction(row_sum, count), Fraction(col_sum, count)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rearrange non-empty separator panels by the centroid of their interior colored pixels.

    Input structure:
    - Black background.
    - One-color full separator rows and full separator columns divide the canvas
      into an N x N array of equal-size panels.
    - Exactly N panels are non-empty; the rest are pure background.

    Rule:
    - For each non-empty panel, compute the centroid of all non-black pixels.
    - Sort panels by centroid row, then centroid column.
    - Output the sorted panels in a single horizontal strip, separated by one
      separator-colored column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep, n, panel_h, panel_w = _find_layout(g)
    nonempty: list[tuple[tuple[Fraction, Fraction], np.ndarray]] = []

    for rs, cs in _panel_slices(n, panel_h, panel_w):
        panel = g[rs, cs]
        if np.any(panel != BG):
            nonempty.append((_centroid_key(panel), panel.copy()))

    assert len(nonempty) == n, "expected exactly N non-empty panels"
    nonempty.sort(key=lambda item: item[0])

    out = np.full((panel_h, n * panel_w + (n - 1)), BG, dtype=int)
    for i, (_key, panel) in enumerate(nonempty):
        c0 = i * (panel_w + 1)
        out[:, c0 : c0 + panel_w] = panel
        if i < n - 1:
            out[:, c0 + panel_w] = sep
    return out


def _random_walk_mask(height: int, width: int) -> np.ndarray | None:
    target = random.randint(2, min(7, max(2, (height * width) // 2)))
    for _ in range(MAX_MASK_ATTEMPTS):
        cells = {(random.randrange(height), random.randrange(width))}
        while len(cells) < target:
            base_r, base_c = random.choice(tuple(cells))
            neighbors = [
                (base_r - 1, base_c),
                (base_r + 1, base_c),
                (base_r, base_c - 1),
                (base_r, base_c + 1),
            ]
            neighbors = [
                cell
                for cell in neighbors
                if 0 <= cell[0] < height and 0 <= cell[1] < width and cell not in cells
            ]
            if not neighbors:
                break
            cells.add(random.choice(neighbors))
        if len(cells) == target:
            mask = np.zeros((height, width), dtype=bool)
            for r, c in cells:
                mask[r, c] = True
            return mask
    return None


def _rect_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    rect_h = random.randint(1, min(3, height))
    rect_w = random.randint(1, min(3, width))
    r0 = random.randint(0, height - rect_h)
    c0 = random.randint(0, width - rect_w)
    mask[r0 : r0 + rect_h, c0 : c0 + rect_w] = True

    if mask.sum() >= 4 and random.random() < 0.75:
        corners = [
            (r0, c0),
            (r0, c0 + rect_w - 1),
            (r0 + rect_h - 1, c0),
            (r0 + rect_h - 1, c0 + rect_w - 1),
        ]
        rr, cc = random.choice(corners)
        mask[rr, cc] = False

    if random.random() < 0.7:
        frontier: list[tuple[int, int]] = []
        for r, c in zip(*np.where(mask), strict=False):
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if frontier:
            rr, cc = random.choice(frontier)
            mask[rr, cc] = True

    return mask


def _bars_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)

    v_len = random.randint(2, height)
    v_col = random.randrange(width)
    v_row = random.randint(0, height - v_len)
    mask[v_row : v_row + v_len, v_col] = True

    h_len = random.randint(2, width)
    h_row = random.randrange(height)
    h_col = random.randint(0, width - h_len)
    mask[h_row, h_col : h_col + h_len] = True

    return mask


def _valid_mask(mask: np.ndarray) -> bool:
    area = int(mask.sum())
    total = int(mask.size)
    return 2 <= area < total and area <= max(6, (2 * total) // 3)


def _make_panel(height: int, width: int) -> tuple[np.ndarray, tuple[Fraction, Fraction]] | None:
    builders = (_random_walk_mask, _rect_mask, _bars_mask)
    for _ in range(MAX_PANEL_ATTEMPTS):
        mask = random.choice(builders)(height, width)
        if mask is None or not _valid_mask(mask):
            continue

        panel = np.full((height, width), BG, dtype=int)
        coords = list(zip(*np.where(mask), strict=False))
        palette_size = min(len(coords), random.choices((1, 2, 3), weights=(5, 3, 1))[0])
        palette = random.sample(FG_COLORS, palette_size)

        colors = palette[:]
        while len(colors) < len(coords):
            colors.append(random.choice(palette))
        random.shuffle(colors)

        for (r, c), color in zip(coords, colors, strict=True):
            panel[r, c] = int(color)

        return panel, _centroid_key(panel)
    return None


def _assemble_input(
    n: int,
    panel_h: int,
    panel_w: int,
    slots: list[tuple[int, int]],
    panels: list[np.ndarray],
) -> np.ndarray:
    grid = np.full((n * panel_h + (n - 1), n * panel_w + (n - 1)), BG, dtype=int)

    for i in range(n - 1):
        grid[panel_h + i * (panel_h + 1), :] = SEP
        grid[:, panel_w + i * (panel_w + 1)] = SEP

    for (slot_r, slot_c), panel in zip(slots, panels, strict=True):
        r0 = slot_r * (panel_h + 1)
        c0 = slot_c * (panel_w + 1)
        grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel

    return grid


def _assemble_output(sorted_panels: list[np.ndarray], panel_w: int) -> np.ndarray:
    panel_h = int(sorted_panels[0].shape[0])
    n = len(sorted_panels)
    out = np.full((panel_h, n * panel_w + (n - 1)), BG, dtype=int)
    for i, panel in enumerate(sorted_panels):
        c0 = i * (panel_w + 1)
        out[:, c0 : c0 + panel_w] = panel
        if i < n - 1:
            out[:, c0 + panel_w] = SEP
    return out


def _sample_slots(n: int) -> list[tuple[int, int]]:
    all_slots = [(r, c) for r in range(n) for c in range(n)]
    for _ in range(MAX_PANEL_ATTEMPTS):
        slots = random.sample(all_slots, n)
        if len({r for r, _c in slots}) >= 2 and len({c for _r, c in slots}) >= 2:
            return slots
    return random.sample(all_slots, n)


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(N_CHOICES)
    (h_lo, h_hi), (w_lo, w_hi) = PANEL_SIZE_RANGES[n]
    panel_h = random.randint(h_lo, h_hi)
    panel_w = random.randint(w_lo, w_hi)

    seen_keys: list[tuple[Fraction, Fraction]] = []
    seen_signatures: set[bytes] = set()
    panel_infos: list[tuple[np.ndarray, tuple[Fraction, Fraction]]] = []
    for _ in range(n):
        panel_info = None
        for _ in range(MAX_PANEL_ATTEMPTS):
            candidate = _make_panel(panel_h, panel_w)
            if candidate is None:
                continue
            panel, key = candidate
            signature = panel.tobytes()
            if signature in seen_signatures or key in seen_keys:
                continue
            panel_info = (panel, key)
            seen_signatures.add(signature)
            seen_keys.append(key)
            break
        if panel_info is None:
            return None
        panel_infos.append(panel_info)

    slots = _sample_slots(n)
    random.shuffle(panel_infos)
    row_major_keys = [
        key for (_slot, (_panel, key)) in sorted(zip(slots, panel_infos, strict=True), key=lambda item: item[0])
    ]
    if row_major_keys == sorted(row_major_keys):
        return None

    grid = _assemble_input(
        n,
        panel_h,
        panel_w,
        slots,
        [panel for panel, _key in panel_infos],
    )
    expected = _assemble_output(
        [panel for panel, _key in sorted(panel_infos, key=lambda item: item[1])],
        panel_w,
    )

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-centroid input")
