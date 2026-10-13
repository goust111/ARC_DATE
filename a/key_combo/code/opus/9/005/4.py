from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    cols = set(map(int, np.unique(grid)))
    if exclude is not None:
        cols -= set(map(int, exclude))
    return sorted(cols)


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> np.ndarray:
    r0 = int(r0)
    c0 = int(c0)
    h, w = map(int, patch.shape)
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]
    return dst


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _is_four_connected(mask: np.ndarray) -> bool:
    area = int(mask.sum())
    if area == 0:
        return False
    return len(components_mask(mask.astype(bool))) == 1


def _find_hidden_subgrid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    for pair in itertools.combinations(colors_present(g), 2):
        mask = np.isin(g, pair)
        rows = mask.any(axis=1)
        cols = mask.any(axis=0)
        rh = int(rows.sum())
        cw = int(cols.sum())
        if rh < 4 or cw < 4 or rh >= h or cw >= w:
            continue
        if int(mask.sum()) != rh * cw:
            continue
        sub = g[np.ix_(rows, cols)]
        if set(map(int, np.unique(sub))) != set(map(int, pair)):
            continue
        candidates.append(
            {
                "pair": tuple(map(int, pair)),
                "rows": rows.copy(),
                "cols": cols.copy(),
                "subgrid": sub.copy(),
            }
        )
    assert len(candidates) == 1, "expected exactly one row/column-product two-color image"
    return candidates[0]


def _find_sprite(grid: np.ndarray, forbidden_colors: set[int]) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            window = g[r0 : r0 + 3, c0 : c0 + 3]
            cols = set(map(int, np.unique(window)))
            if 0 not in cols:
                continue
            fg_colors = cols - {0}
            if len(fg_colors) != 1:
                continue
            fg = next(iter(fg_colors))
            if fg in forbidden_colors:
                continue
            fg_mask = window == fg
            area = int(fg_mask.sum())
            if not (2 <= area <= 4):
                continue
            if int(np.sum(g == fg)) != area:
                continue
            if not _is_four_connected(fg_mask):
                continue
            candidates.append(
                {
                    "top": int(r0),
                    "left": int(c0),
                    "fg": int(fg),
                    "count": area,
                    "sprite": window.copy(),
                }
            )
    assert len(candidates) == 1, "expected exactly one valid 3x3 sprite stamp"
    return candidates[0]


def _overlay_copies(tile: np.ndarray, sprite: np.ndarray, copies: int) -> np.ndarray:
    tile = np.asarray(tile, dtype=int)
    sprite = np.asarray(sprite, dtype=int)
    h, w = tile.shape
    out = np.tile(tile, (1, int(copies)))
    r0 = h // 2 - 1
    c0 = w // 2 - 1
    for k in range(int(copies)):
        paste(out, sprite, r0, c0 + k * w, transparent=0)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the hidden two-color image, tile it by the sprite's foreground-pixel count, and stamp the sprite onto each copy.

    Input structure:
    - The grid contains exactly one pair of colors whose pixels fill the full cross product of their active rows and active columns.
    - Elsewhere there is exactly one 3x3 sprite made of black background and one non-pair foreground color, with 2-4 connected foreground pixels.
    - Other colors are filler or separators and do not belong to either the hidden two-color image or the sprite.

    Rule:
    - Find the unique color pair whose mask fills its active row/column product, then extract the induced two-color subgrid.
    - Find the unique 3x3 sprite, and count its non-black pixels to get F.
    - Repeat the extracted subgrid horizontally F times.
    - On each copy, overlay the 3x3 sprite with black treated as transparent, aligning the sprite center to the copy center cell.

    Key insight:
    - The small sprite has two simultaneous roles: its pixel count determines how many copies to create, and its shape itself is the transparent stamp applied to every copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    hidden = _find_hidden_subgrid(g)
    sprite = _find_sprite(g, set(hidden["pair"]))
    return _overlay_copies(hidden["subgrid"], sprite["sprite"], sprite["count"])


_SPRITE_BASE_SHAPES = {
    2: (
        ((1, 0), (1, 1)),
    ),
    3: (
        ((0, 1), (1, 1), (2, 1)),
        ((0, 0), (1, 0), (1, 1)),
    ),
    4: (
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (0, 2), (1, 0), (1, 1)),
    ),
}


def _transform_coords(coords: tuple[tuple[int, int], ...], kind: int) -> tuple[tuple[int, int], ...]:
    pts = list(coords)
    out = []
    for r, c in pts:
        if kind == 0:
            rr, cc = r, c
        elif kind == 1:
            rr, cc = r, 2 - c
        elif kind == 2:
            rr, cc = 2 - r, c
        elif kind == 3:
            rr, cc = 2 - r, 2 - c
        elif kind == 4:
            rr, cc = c, r
        elif kind == 5:
            rr, cc = c, 2 - r
        elif kind == 6:
            rr, cc = 2 - c, r
        else:
            rr, cc = 2 - c, 2 - r
        out.append((int(rr), int(cc)))
    return tuple(sorted(set(out)))


def _build_sprite(shape_cells: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    sprite = np.zeros((3, 3), dtype=int)
    for r, c in shape_cells:
        sprite[int(r), int(c)] = int(color)
    return sprite


def _random_tile(m: int, n: int, colors: tuple[int, int]) -> np.ndarray:
    a, b = map(int, colors)
    style = random.randrange(4)
    tile = np.full((m, n), a, dtype=int)

    if style == 0:
        row_bits = np.random.randint(0, 2, size=m)
        col_bits = np.random.randint(0, 2, size=n)
        mask = (row_bits[:, None] ^ col_bits[None, :]).astype(bool)
        if random.random() < 0.5:
            mask ^= np.random.randint(0, 2, size=(m, n)).astype(bool)
    elif style == 1:
        mask = np.random.rand(m, n) < random.uniform(0.3, 0.7)
        band_r0 = random.randint(0, m - 2)
        band_r1 = random.randint(band_r0 + 1, m - 1)
        mask[band_r0 : band_r1 + 1, random.randrange(n)] = True
        mask[random.randrange(m), :] ^= True
    elif style == 2:
        mask = np.zeros((m, n), dtype=bool)
        for _ in range(random.randint(2, 4)):
            r0 = random.randint(0, m - 1)
            c0 = random.randint(0, n - 1)
            r1 = random.randint(r0, m - 1)
            c1 = random.randint(c0, n - 1)
            mask[r0 : r1 + 1, c0 : c1 + 1] ^= True
    else:
        rr, cc = np.indices((m, n))
        mask = ((rr + 2 * cc + random.randint(0, 3)) % random.choice((2, 3))) == 0
        if random.random() < 0.5:
            mask |= rr == random.randrange(m)
        if random.random() < 0.5:
            mask ^= cc == random.randrange(n)

    if int(mask.sum()) == 0 or int(mask.sum()) == m * n:
        return _random_tile(m, n, colors)

    tile[mask] = b
    if len(set(map(tuple, tile))) < 2:
        return _random_tile(m, n, colors)
    if len(set(map(tuple, tile.T))) < 2:
        return _random_tile(m, n, colors)
    if np.all(tile[:, :-1] == tile[:, 1:]) or np.all(tile[:-1, :] == tile[1:, :]):
        return _random_tile(m, n, colors)
    for period in range(1, n):
        if n % period == 0 and np.array_equal(tile, np.tile(tile[:, :period], (1, n // period))):
            return _random_tile(m, n, colors)
    return tile


def _corner_reserved_ranges(h: int, w: int, corner: str) -> tuple[set[int], set[int], tuple[int, int]]:
    if corner == "tl":
        return set(range(4)), set(range(4)), (0, 0)
    if corner == "tr":
        return set(range(4)), set(range(w - 4, w)), (0, w - 3)
    if corner == "bl":
        return set(range(h - 4, h)), set(range(4)), (h - 3, 0)
    return set(range(h - 4, h)), set(range(w - 4, w)), (h - 3, w - 3)


def _sample_active_indices(total: int, count: int, forbidden: set[int]) -> list[int] | None:
    allowed = [i for i in range(total) if i not in forbidden]
    if len(allowed) < count:
        return None
    for _ in range(30):
        picks = sorted(random.sample(allowed, count))
        gaps = sum(1 for a, b in zip(picks, picks[1:]) if b - a > 1)
        if gaps >= 1 or len(allowed) == count or random.random() < 0.25:
            return picks
    return sorted(random.sample(allowed, count))


def _paint_filler(grid: np.ndarray, rows: list[int], cols: list[int], filler_colors: list[int]) -> None:
    h, w = grid.shape
    active_rows = np.zeros(h, dtype=bool)
    active_cols = np.zeros(w, dtype=bool)
    active_rows[rows] = True
    active_cols[cols] = True
    outside = ~(active_rows[:, None] & active_cols[None, :])

    if len(filler_colors) >= 2:
        internal_rows = active_rows.copy()
        internal_cols = active_cols.copy()
        r_idx = np.flatnonzero(active_rows)
        c_idx = np.flatnonzero(active_cols)
        internal_rows[: r_idx[0]] = False
        internal_rows[r_idx[-1] + 1 :] = False
        internal_cols[: c_idx[0]] = False
        internal_cols[c_idx[-1] + 1 :] = False
        stripe = (internal_rows[:, None] ^ internal_cols[None, :]) & outside
        grid[stripe] = filler_colors[1]

    for color in filler_colors[2:]:
        if random.random() < 0.5:
            r = random.randrange(h)
            grid[r, outside[r]] = int(color)
        else:
            c = random.randrange(w)
            grid[outside[:, c], c] = int(color)


def _semantic_check(inp: np.ndarray) -> None:
    hidden = _find_hidden_subgrid(inp)
    sprite = _find_sprite(inp, set(hidden["pair"]))
    tile = hidden["subgrid"]
    stamped = _overlay_copies(tile, sprite["sprite"], sprite["count"])
    solved = main(inp)

    assert np.array_equal(solved, stamped), "main() must match the decomposed rule"
    assert not np.array_equal(solved, np.tile(tile, (1, sprite["count"]))), "sprite overlay must matter"
    wrong_copies = 2 if sprite["count"] != 2 else 3
    if wrong_copies != sprite["count"]:
        wrong = _overlay_copies(tile, sprite["sprite"], wrong_copies)
        assert solved.shape != wrong.shape or not np.array_equal(solved, wrong), "sprite count must matter"
    assert sprite["sprite"].shape == (3, 3)
    assert 2 <= sprite["count"] <= 4
    assert tile.shape[0] >= 4 and tile.shape[1] >= 4


def _generate_attempt() -> np.ndarray | None:
    copies = random.randint(2, 4)
    tile_h = random.randint(4, 7)
    tile_w = random.randint(4, min(7, 30 // copies))
    h = tile_h + random.randint(4, 8)
    w = tile_w + random.randint(4, 9)
    if max(h, w) > 30:
        return None

    all_colors = list(range(1, 10))
    pair_colors = tuple(random.sample(all_colors, 2))
    remaining = [c for c in all_colors if c not in pair_colors]
    sprite_color = random.choice(remaining)
    filler_colors = random.sample([c for c in remaining if c != sprite_color], random.randint(2, 4))
    grid = np.full((h, w), filler_colors[0], dtype=int)

    base_shape = random.choice(_SPRITE_BASE_SHAPES[copies])
    shape = _transform_coords(base_shape, random.randrange(8))
    sprite = _build_sprite(shape, sprite_color)

    corner = random.choice(("tl", "tr", "bl", "br"))
    forbidden_rows, forbidden_cols, (sr, sc) = _corner_reserved_ranges(h, w, corner)
    rows = _sample_active_indices(h, tile_h, forbidden_rows)
    cols = _sample_active_indices(w, tile_w, forbidden_cols)
    if rows is None or cols is None:
        return None

    tile = _random_tile(tile_h, tile_w, pair_colors)
    _paint_filler(grid, rows, cols, filler_colors)

    row_mask = np.zeros(h, dtype=bool)
    col_mask = np.zeros(w, dtype=bool)
    row_mask[rows] = True
    col_mask[cols] = True
    grid[np.ix_(rows, cols)] = tile
    paste(grid, sprite, sr, sc)

    try:
        _semantic_check(grid)
    except AssertionError:
        return None

    out = main(grid)
    if out.shape[0] > 30 or out.shape[1] > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
