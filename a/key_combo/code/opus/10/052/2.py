from __future__ import annotations

import random

import numpy as np


BG = 0
GUIDE = 8
PLACEHOLDER = 5
TILE_COLORS = (1, 2, 3, 4, 6, 7, 9)
GEN_RETRIES = 400


def _find_guide_row(grid: np.ndarray) -> tuple[bool, np.ndarray]:
    top_has = bool(np.any(grid[0] == GUIDE))
    bottom_has = bool(np.any(grid[-1] == GUIDE))
    assert top_has ^ bottom_has, "expected exactly one edge guide row"
    guide_top = top_has
    guide_row = grid[0] if guide_top else grid[-1]
    assert guide_row[0] == GUIDE and guide_row[-1] == GUIDE, "guide gaps cannot touch endpoints"
    assert np.all((guide_row == BG) | (guide_row == GUIDE))
    return guide_top, guide_row


def _find_placeholder_suffix(content: np.ndarray) -> tuple[int, int, int]:
    h, w = content.shape
    candidates: list[tuple[int, int, int]] = []
    for color in map(int, np.unique(content)):
        mask = content == color
        rows_all = np.all(mask, axis=1)
        cols_all = np.all(mask, axis=0)
        if not rows_all[-1] or not cols_all[-1]:
            continue

        r_cut = h
        while r_cut > 0 and rows_all[r_cut - 1]:
            r_cut -= 1
        c_cut = w
        while c_cut > 0 and cols_all[c_cut - 1]:
            c_cut -= 1
        if r_cut == 0 or c_cut == 0:
            continue

        expected = np.zeros((h, w), dtype=bool)
        expected[r_cut:, :] = True
        expected[:, c_cut:] = True
        if np.array_equal(mask, expected):
            candidates.append((color, r_cut, c_cut))

    assert len(candidates) == 1, f"expected one placeholder suffix, got {candidates}"
    return candidates[0]


def _minimal_periods_any(block: np.ndarray) -> tuple[int, int]:
    h, w = block.shape

    row_period = None
    for period in range(1, h + 1):
        if all(block[r, c] == block[r % period, c] for r in range(h) for c in range(w)):
            row_period = period
            break

    col_period = None
    for period in range(1, w + 1):
        if all(block[r, c] == block[r, c % period] for r in range(h) for c in range(w)):
            col_period = period
            break

    assert row_period is not None and col_period is not None
    return int(row_period), int(col_period)


def _minimal_periods(block: np.ndarray) -> tuple[int, int]:
    row_period, col_period = _minimal_periods_any(block)
    assert row_period < block.shape[0] and col_period < block.shape[1]
    return row_period, col_period


def _tile_extend(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    rr = np.arange(h)[:, None] % tile.shape[0]
    cc = np.arange(w)[None, :] % tile.shape[1]
    return tile[rr, cc]


def _pack_columns(columns: np.ndarray, guide_top: bool) -> np.ndarray:
    h, w = columns.shape
    out = np.zeros((h, w), dtype=int)
    for c in range(w):
        vals = columns[:, c][columns[:, c] != BG]
        if vals.size == 0:
            continue
        if guide_top:
            out[h - vals.size :, c] = vals
        else:
            out[: vals.size, c] = vals
    return out


def _expected_from_completed(completed: np.ndarray, keep_cols: np.ndarray, guide_top: bool) -> np.ndarray:
    kept = completed[:, keep_cols]
    return _pack_columns(kept, guide_top)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the hidden periodic tile, erase columns under guide-bar gaps, and pack the surviving colored cells away from the guide.

    Input structure:
    - A one-pixel teal guide bar lies on the top or bottom edge, spans the full width, and contains internal black gaps.
    - After removing that guide row, the remaining area has a top-left visible block cut from a repeating rectangular tile.
    - A single gray placeholder color fills an L-shaped suffix made of bottom rows and right columns, hiding the continuation of the tile.

    Rule:
    - Remove the guide row and infer the minimal repeating tile from the visible top-left block.
    - Periodically extend that tile over the whole non-guide area.
    - Keep only the completed columns whose guide-row cell is teal; erase columns whose guide-row cell is a gap.
    - Drop the erased columns, then within each surviving column compress non-black cells toward the edge opposite the guide while preserving their top-to-bottom order.

    Key insight:
    - The gray area is not output content but a masked continuation of the same tile, and the guide-row gaps act as a binary column filter before per-column compression away from the guide.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    guide_top, guide_row = _find_guide_row(g)
    content = g[1:, :] if guide_top else g[:-1, :]

    placeholder, r_cut, c_cut = _find_placeholder_suffix(content)
    visible = content[:r_cut, :c_cut]
    assert placeholder not in set(map(int, visible.flat))

    row_period, col_period = _minimal_periods(visible)
    tile = visible[:row_period, :col_period]
    completed = _tile_extend(tile, content.shape)

    keep_cols = guide_row == GUIDE
    assert np.any(keep_cols)
    return _expected_from_completed(completed, keep_cols, guide_top)


def _sample_exact_tile() -> np.ndarray:
    for _ in range(200):
        tile_h = random.randint(3, 4)
        tile_w = random.randint(3, 4)
        color_count = random.randint(2, 3)
        palette = [BG] + random.sample(list(TILE_COLORS), color_count)
        tile = np.array(
            [[random.choice(palette) for _ in range(tile_w)] for _ in range(tile_h)],
            dtype=int,
        )

        if len(set(map(int, tile.flat)) - {BG}) < 2:
            continue
        if not np.any(tile == BG):
            continue
        if np.any(np.all(tile == BG, axis=0)) or np.any(np.all(tile == BG, axis=1)):
            continue
        if not any(np.any(tile[:, c] == BG) and np.any(tile[:, c] != BG) for c in range(tile_w)):
            continue
        if not any(np.any(tile[r, :] == BG) and np.any(tile[r, :] != BG) for r in range(tile_h)):
            continue
        if _minimal_periods_any(tile) != (tile_h, tile_w):
            continue
        return tile
    raise RuntimeError("failed to sample an exact-period tile")


def _sample_guide_mask(width: int) -> np.ndarray:
    for _ in range(200):
        keep = np.ones(width, dtype=bool)
        used = np.zeros(width, dtype=bool)
        gap_count = random.randint(1, min(3, max(1, width // 4)))
        gaps: list[tuple[int, int]] = []
        ok = True
        for _ in range(gap_count):
            gap_w = random.randint(1, min(3, width - 2))
            starts = list(range(1, width - gap_w))
            random.shuffle(starts)
            placed = False
            for start in starts:
                stop = start + gap_w
                lo = max(0, start - 1)
                hi = min(width, stop + 1)
                if np.any(used[lo:hi]):
                    continue
                keep[start:stop] = False
                used[start:stop] = True
                gaps.append((start, stop))
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok and np.any(~keep) and np.any(keep):
            return keep
    raise RuntimeError("failed to sample guide gaps")


def _gen_attempt() -> np.ndarray | None:
    tile = _sample_exact_tile()
    tile_h, tile_w = tile.shape

    rep_h = random.randint(2, 4)
    rep_w = random.randint(3, 5)
    r_cut = tile_h * rep_h
    c_cut = tile_w * rep_w
    width = c_cut + random.randint(2, 5)
    keep_cols = _sample_guide_mask(width)
    guide_top = bool(random.getrandbits(1))

    content_h = r_cut + random.randint(1, 4)
    if max(content_h + 1, width) > 30:
        return None

    completed = _tile_extend(tile, (content_h, width))
    placeholder_mask = np.zeros((content_h, width), dtype=bool)
    placeholder_mask[r_cut:, :] = True
    placeholder_mask[:, c_cut:] = True

    content = completed.copy()
    content[placeholder_mask] = PLACEHOLDER
    guide_row = np.where(keep_cols, GUIDE, BG)
    inp = np.vstack([guide_row[None, :], content]) if guide_top else np.vstack([content, guide_row[None, :]])

    expected = _expected_from_completed(completed, keep_cols, guide_top)
    visible_only = completed.copy()
    visible_only[placeholder_mask] = BG
    visible_expected = _expected_from_completed(visible_only, keep_cols, guide_top)
    selected = completed[:, keep_cols]

    if not np.any((completed != BG) & placeholder_mask & keep_cols[None, :]):
        return None
    if not np.any((completed != BG) & (~keep_cols)[None, :]):
        return None
    if np.array_equal(expected, visible_expected):
        return None
    if np.array_equal(expected, selected):
        return None
    if not np.all(np.any(completed[:, keep_cols] != BG, axis=0)):
        return None

    try:
        parsed_top, parsed_guide = _find_guide_row(inp)
        parsed_content = inp[1:, :] if parsed_top else inp[:-1, :]
        placeholder, parsed_r_cut, parsed_c_cut = _find_placeholder_suffix(parsed_content)
        row_period, col_period = _minimal_periods(parsed_content[:parsed_r_cut, :parsed_c_cut])
        out = main(inp)
    except Exception:
        return None

    if placeholder != PLACEHOLDER:
        return None
    if parsed_top != guide_top:
        return None
    if not np.array_equal(parsed_guide == GUIDE, keep_cols):
        return None
    if (parsed_r_cut, parsed_c_cut) != (r_cut, c_cut):
        return None
    if (row_period, col_period) != tile.shape:
        return None
    if not np.array_equal(out, expected):
        return None
    if np.array_equal(out, np.zeros_like(out)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
