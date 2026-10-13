from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 9
SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator bounds.
GRID_HEIGHT_RANGE = (9, 15)
PANEL_COUNT_RANGE = (2, 5)
PANEL_WIDTH_RANGE = (7, 13)
MASK_ATTEMPTS = 80
GENERATION_ATTEMPTS = 300
KINDS = ("lr", "ud", "both", "none")


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _bands(width: int, separators: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = 0
    for col in separators:
        if start < col:
            out.append((start, col))
        start = col + 1
    if start < width:
        out.append((start, width))
    return out


def _symmetry_flags(mask: np.ndarray) -> tuple[bool, bool]:
    return bool(np.array_equal(mask, np.fliplr(mask))), bool(np.array_equal(mask, np.flipud(mask)))


def _fold_mask(mask: np.ndarray) -> np.ndarray:
    keep = mask.copy()
    h, w = mask.shape
    lr, ud = _symmetry_flags(mask)

    if lr:
        keep &= np.arange(w)[None, :] < (w + 1) // 2
    if ud:
        keep &= np.arange(h)[:, None] < (h + 1) // 2
    return keep


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold symmetric sprites inside separator-defined panels.

    Input structure
    - Background is black.
    - One separator color forms one-cell-wide full-height columns.
    - Each panel between separators contains exactly one 4-connected monochrome sprite.

    Rule
    - Left-right symmetric sprites keep only their left half.
    - Up-down symmetric sprites keep only their top half.
    - Sprites with both symmetries keep only their top-left quadrant.
    - Non-symmetric sprites stay unchanged.
    - Separator columns remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape

    sep_cols = [c for c in range(w) if int(g[0, c]) != BG and np.all(g[:, c] == g[0, c])]
    sep_colors = {int(g[0, c]) for c in sep_cols}
    assert len(sep_colors) == 1, "expected exactly one separator color"
    sep = next(iter(sep_colors))
    assert sep != BG, "separator color must differ from background"

    panels = _bands(w, sep_cols)
    assert len(panels) >= 2, "expected multiple panels"

    out = g.copy()
    for c0, c1 in panels:
        panel = g[:, c0:c1]
        fg = panel != BG
        assert fg.any(), "each panel must contain a sprite"
        colors = set(map(int, np.unique(panel[fg])))
        assert len(colors) == 1, "each panel sprite must be monochrome"
        color = next(iter(colors))
        assert color != sep, "sprite color must differ from separator color"

        comps = _components(fg)
        assert len(comps) == 1, "each panel must contain exactly one 4-connected sprite"
        r0, lc0, r1, lc1 = _bbox(comps[0])
        mask = fg[r0 : r1 + 1, lc0 : lc1 + 1]
        kept = _fold_mask(mask)

        cut = mask & ~kept
        patch = out[r0 : r1 + 1, c0 + lc0 : c0 + lc1 + 1]
        patch[cut] = BG

    return out


def _grow_connected(shape: tuple[int, int], target: int, seeds: set[tuple[int, int]]) -> np.ndarray | None:
    h, w = shape
    cells = set(seeds)
    frontier: set[tuple[int, int]] = set()

    for r, c in cells:
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    while len(cells) < target:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        r, c = cell
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _sample_odd(lo: int, hi: int) -> int | None:
    values = [v for v in range(lo, hi + 1) if v % 2 == 1]
    return random.choice(values) if values else None


def _make_lr_mask(panel_h: int, panel_w: int) -> np.ndarray | None:
    full_w = _sample_odd(3, panel_w - 2)
    if full_w is None:
        return None
    h = random.randint(3, panel_h - 2)
    half_w = full_w // 2 + 1
    area = h * half_w
    lo = max(2, area // 4)
    hi = max(lo, (2 * area) // 3)
    seed = {(random.randrange(h), half_w - 1)}
    half = _grow_connected((h, half_w), random.randint(lo, hi), seed)
    if half is None:
        return None
    full = np.hstack([half, half[:, :-1][:, ::-1]])
    return full if _symmetry_flags(full) == (True, False) else None


def _make_ud_mask(panel_h: int, panel_w: int) -> np.ndarray | None:
    full_h = _sample_odd(3, panel_h - 2)
    if full_h is None:
        return None
    w = random.randint(3, panel_w - 2)
    half_h = full_h // 2 + 1
    area = half_h * w
    lo = max(2, area // 4)
    hi = max(lo, (2 * area) // 3)
    seed = {(half_h - 1, random.randrange(w))}
    top = _grow_connected((half_h, w), random.randint(lo, hi), seed)
    if top is None:
        return None
    full = np.vstack([top, top[:-1, :][::-1, :]])
    return full if _symmetry_flags(full) == (False, True) else None


def _make_both_mask(panel_h: int, panel_w: int) -> np.ndarray | None:
    full_h = _sample_odd(3, panel_h - 2)
    full_w = _sample_odd(3, panel_w - 2)
    if full_h is None or full_w is None:
        return None
    half_h = full_h // 2 + 1
    half_w = full_w // 2 + 1
    area = half_h * half_w
    lo = max(2, area // 4)
    hi = max(lo, (3 * area) // 4)
    seed = {(half_h - 1, half_w - 1)}
    quarter = _grow_connected((half_h, half_w), random.randint(lo, hi), seed)
    if quarter is None:
        return None
    top = np.hstack([quarter, quarter[:, :-1][:, ::-1]])
    full = np.vstack([top, top[:-1, :][::-1, :]])
    return full if _symmetry_flags(full) == (True, True) else None


def _make_none_mask(panel_h: int, panel_w: int) -> np.ndarray | None:
    h = random.randint(3, panel_h - 2)
    w = random.randint(3, panel_w - 2)
    area = h * w
    lo = max(4, area // 4)
    hi = max(lo, (2 * area) // 3)
    seed = {(random.randrange(h), random.randrange(w))}
    mask = _grow_connected((h, w), random.randint(lo, hi), seed)
    if mask is None:
        return None
    return mask if _symmetry_flags(mask) == (False, False) else None


def _make_mask(kind: str, panel_h: int, panel_w: int) -> np.ndarray | None:
    for _ in range(MASK_ATTEMPTS):
        if kind == "lr":
            mask = _make_lr_mask(panel_h, panel_w)
        elif kind == "ud":
            mask = _make_ud_mask(panel_h, panel_w)
        elif kind == "both":
            mask = _make_both_mask(panel_h, panel_w)
        else:
            mask = _make_none_mask(panel_h, panel_w)
        if mask is not None:
            return mask
    return None


def _draw_mask(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = canvas[top : top + h, left : left + w]
    patch[mask] = color


def _make_panel(kind: str, panel_h: int, panel_w: int, color: int) -> tuple[np.ndarray, np.ndarray] | None:
    mask = _make_mask(kind, panel_h, panel_w)
    if mask is None:
        return None

    h, w = mask.shape
    top = random.randint(1, panel_h - h - 1)
    left = random.randint(1, panel_w - w - 1)

    inp = np.full((panel_h, panel_w), BG, dtype=int)
    out = np.full((panel_h, panel_w), BG, dtype=int)
    _draw_mask(inp, mask, top, left, color)
    _draw_mask(out, _fold_mask(mask), top, left, color)
    return inp, out


def _build_grid(panel_height: int, widths: list[int], kinds: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    total_w = sum(widths) + len(widths) - 1
    inp = np.full((panel_height, total_w), BG, dtype=int)
    out = np.full((panel_height, total_w), BG, dtype=int)

    cursor = 0
    for panel_w, kind in zip(widths, kinds, strict=False):
        color = random.choice(SPRITE_COLORS)
        panel_pair = _make_panel(kind, panel_height, panel_w, color)
        if panel_pair is None:
            return None
        panel_inp, panel_out = panel_pair
        inp[:, cursor : cursor + panel_w] = panel_inp
        out[:, cursor : cursor + panel_w] = panel_out
        cursor += panel_w
        if cursor < total_w:
            inp[:, cursor] = SEP
            out[:, cursor] = SEP
            cursor += 1

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(*GRID_HEIGHT_RANGE)
    panel_count = random.randint(*PANEL_COUNT_RANGE)
    widths = [random.randint(*PANEL_WIDTH_RANGE) for _ in range(panel_count)]
    kinds = [random.choice(KINDS) for _ in range(panel_count)]

    pair = _build_grid(panel_h, widths, kinds)
    if pair is None:
        return None
    inp, expected = pair

    try:
        solved = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(solved, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
