import random

import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 600
MAX_MASK_ATTEMPTS = 200


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == SEP)]
    assert len(rows) == 1, "expected exactly one full-width grey separator row"
    return int(rows[0])


def _overlay(top: np.ndarray, bottom_aligned: np.ndarray) -> np.ndarray:
    overlap = (top != BG) & (bottom_aligned != BG)
    assert np.all(top[overlap] == bottom_aligned[overlap]), "folded overlaps must agree"
    merged = top.copy()
    fill = (merged == BG) & (bottom_aligned != BG)
    merged[fill] = bottom_aligned[fill]
    return merged


def _candidate_tile(masked: np.ndarray, tile_h: int, tile_w: int):
    tile = np.full((tile_h, tile_w), -1, dtype=int)
    for r, c in np.argwhere(masked != BG):
        pr = int(r) % tile_h
        pc = int(c) % tile_w
        value = int(masked[r, c])
        if tile[pr, pc] == -1:
            tile[pr, pc] = value
        elif tile[pr, pc] != value:
            return None
    return tile


def _infer_unique_minimal_tile(masked: np.ndarray) -> np.ndarray:
    h, w = masked.shape
    best_area = None
    best = []
    for tile_h in range(1, h + 1):
        for tile_w in range(1, w + 1):
            tile = _candidate_tile(masked, tile_h, tile_w)
            if tile is None:
                continue
            area = tile_h * tile_w
            if best_area is None or area < best_area:
                best_area = area
                best = [tile]
            elif area == best_area:
                best.append(tile)

    assert best_area is not None, "no periodic tile is consistent with the visible cells"
    assert len(best) == 1, "minimal-area tile is not unique"
    tile = best[0]
    assert np.all(tile != -1), "minimal-area tile is underdetermined"
    return tile


def _has_unique_minimal_tile(masked: np.ndarray) -> bool:
    try:
        _infer_unique_minimal_tile(masked)
    except AssertionError:
        return False
    return True


def _tile_wallpaper(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = np.arange(height)[:, None] % tile.shape[0]
    cols = np.arange(width)[None, :] % tile.shape[1]
    return tile[rows, cols]


def _minimal_period(tile: np.ndarray) -> tuple[int, int]:
    h, w = tile.shape
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    for tile_h in range(1, h + 1):
        for tile_w in range(1, w + 1):
            if np.array_equal(tile, tile[rows % tile_h, cols % tile_w]):
                return tile_h, tile_w
    return h, w


def main(grid: np.ndarray) -> np.ndarray:
    """Fold two incomplete wallpaper views together, then infer and restore the smallest repeating tile.

    Input structure:
    - The background is black, and a single full-width grey separator row splits the grid into equal-height top and bottom panels.
    - Each panel is a partially erased view of the same non-black periodic wallpaper; the bottom panel is stored upside-down relative to the top panel.
    - Some coordinates are black holes, and matching non-black overlaps agree in color after the bottom panel is flipped vertically.

    Rule:
    - Remove the grey separator row.
    - Flip the bottom panel vertically and overlay it onto the top panel, keeping the top color when both sides are visible.
    - Treat the remaining black cells as missing data, infer the unique minimal-area periodic tile consistent with all visible colors, and fill every hole from that tile.

    Key insight:
    - Neither half alone reveals enough tile phases, but folding the two complementary partial views into one coordinate system exposes enough periodic evidence to recover the full wallpaper.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"

    sep = _find_separator_row(g)
    top = g[:sep]
    bottom = g[sep + 1 :]
    assert top.shape == bottom.shape and top.shape[0] > 0, "separator must split equal-height panels"

    merged = _overlay(top, np.flipud(bottom))
    tile = _infer_unique_minimal_tile(merged)
    return _tile_wallpaper(tile, *merged.shape)


def _sample_tile() -> np.ndarray:
    for _ in range(200):
        tile_h = random.randint(2, 3)
        tile_w = random.randint(2, 3)
        n_colors = random.randint(2, min(3, tile_h * tile_w))
        colors = random.sample(PALETTE, n_colors)
        tile = np.array(
            [[random.choice(colors) for _ in range(tile_w)] for _ in range(tile_h)],
            dtype=int,
        )
        if len(set(map(int, tile.flat))) < 2:
            continue
        if _minimal_period(tile) != (tile_h, tile_w):
            continue
        return tile
    raise RuntimeError("failed to sample a full-period tile")


def _phase_groups(height: int, width: int, tile_h: int, tile_w: int):
    groups = {}
    for r in range(height):
        for c in range(width):
            groups.setdefault((r % tile_h, c % tile_w), []).append((r, c))
    return groups


def _allowed_categories(phase, top_missing, bottom_missing):
    if phase == top_missing:
        return ("bottom", "none")
    if phase == bottom_missing:
        return ("top", "none")
    return ("top", "bottom", "both", "none")


def _build_masks(height: int, width: int, tile_h: int, tile_w: int):
    total = height * width
    min_visible = int(np.ceil(0.40 * total))
    max_visible = int(np.floor(0.60 * total))
    groups = _phase_groups(height, width, tile_h, tile_w)
    phases = list(groups)

    for _ in range(MAX_MASK_ATTEMPTS):
        top_missing, bottom_missing = random.sample(phases, 2)
        top_target = random.randint(min_visible, max_visible)
        bottom_target = random.randint(min_visible, max_visible)

        both_lo = max(1, top_target + bottom_target - (total - 1))
        both_hi = min(top_target - 1, bottom_target - 1)
        if both_lo > both_hi:
            continue
        both_target = random.randint(both_lo, both_hi)

        remaining = {
            "top": top_target - both_target,
            "bottom": bottom_target - both_target,
            "both": both_target,
        }
        none_target = total - remaining["top"] - remaining["bottom"] - remaining["both"]
        if min(remaining.values()) < 1 or none_target < 1:
            continue
        remaining["none"] = none_target

        assignment = {}
        ok = True

        for phase in random.sample(phases, len(phases)):
            cells = [cell for cell in groups[phase] if cell not in assignment]
            allowed = [cat for cat in _allowed_categories(phase, top_missing, bottom_missing) if cat != "none"]
            options = [cat for cat in allowed if remaining[cat] > 0]
            if not cells or not options:
                ok = False
                break
            weights = [remaining[cat] for cat in options]
            cat = random.choices(options, weights=weights, k=1)[0]
            cell = random.choice(cells)
            assignment[cell] = cat
            remaining[cat] -= 1

        if not ok:
            continue

        unassigned = [(r, c) for r in range(height) for c in range(width) if (r, c) not in assignment]
        random.shuffle(unassigned)
        for cell in unassigned:
            phase = (cell[0] % tile_h, cell[1] % tile_w)
            options = [cat for cat in _allowed_categories(phase, top_missing, bottom_missing) if remaining[cat] > 0]
            if not options:
                ok = False
                break
            weights = [remaining[cat] for cat in options]
            cat = random.choices(options, weights=weights, k=1)[0]
            assignment[cell] = cat
            remaining[cat] -= 1

        if not ok or any(remaining.values()):
            continue

        top = np.zeros((height, width), dtype=bool)
        bottom = np.zeros((height, width), dtype=bool)
        for (r, c), cat in assignment.items():
            if cat in ("top", "both"):
                top[r, c] = True
            if cat in ("bottom", "both"):
                bottom[r, c] = True

        if np.any(~top & ~bottom) and np.any(top & bottom):
            return top, bottom

    return None


def _bad_shortcut_matches_output(top_view: np.ndarray, bottom_input: np.ndarray, expected: np.ndarray) -> bool:
    overlap = (top_view != BG) & (bottom_input != BG)
    if np.any(top_view[overlap] != bottom_input[overlap]):
        return False
    wrong_merged = _overlay(top_view, bottom_input)
    if not _has_unique_minimal_tile(wrong_merged):
        return False
    return np.array_equal(_tile_wallpaper(_infer_unique_minimal_tile(wrong_merged), *expected.shape), expected)


def _valid_instance(tile: np.ndarray, output: np.ndarray, top_mask: np.ndarray, bottom_mask: np.ndarray) -> bool:
    top_view = np.where(top_mask, output, BG)
    bottom_aligned = np.where(bottom_mask, output, BG)
    merged = _overlay(top_view, bottom_aligned)

    if _has_unique_minimal_tile(top_view):
        return False
    if _has_unique_minimal_tile(bottom_aligned):
        return False
    if not np.any(merged == BG):
        return False
    if not _has_unique_minimal_tile(merged):
        return False
    if not np.array_equal(_tile_wallpaper(_infer_unique_minimal_tile(merged), *output.shape), output):
        return False

    groups = _phase_groups(output.shape[0], output.shape[1], tile.shape[0], tile.shape[1])
    for cells in groups.values():
        if not any(merged[r, c] != BG for r, c in cells):
            return False

    bottom_input = np.flipud(bottom_aligned)
    if _bad_shortcut_matches_output(top_view, bottom_input, output):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        tile = _sample_tile()
        panel_h = random.randint(max(4, tile.shape[0] + 2), min(7, tile.shape[0] + 4))
        panel_w = random.randint(max(5, tile.shape[1] + 3), min(9, tile.shape[1] + 5))
        output = _tile_wallpaper(tile, panel_h, panel_w)

        masks = _build_masks(panel_h, panel_w, tile.shape[0], tile.shape[1])
        if masks is None:
            continue
        top_mask, bottom_mask = masks
        if not _valid_instance(tile, output, top_mask, bottom_mask):
            continue

        top_view = np.where(top_mask, output, BG)
        bottom_input = np.flipud(np.where(bottom_mask, output, BG))
        return np.vstack([top_view, np.full((1, panel_w), SEP, dtype=int), bottom_input])

    raise RuntimeError("failed to generate a valid folded-wallpaper instance")
