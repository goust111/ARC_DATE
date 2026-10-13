import random

import numpy as np


BLACK = 0
NON_BLACK = tuple(range(1, 10))
GEN_ATTEMPTS = 400


def _colors_present(grid: np.ndarray) -> list[int]:
    return [int(v) for v in np.unique(grid) if int(v) != BLACK]


def _tile_fill(shape: tuple[int, int], tile: np.ndarray) -> np.ndarray:
    """Mirror the shared arc_utils.tile_fill helper without importing it."""
    h, w = map(int, shape)
    th, tw = tile.shape
    rr = np.arange(h)[:, None] % th
    cc = np.arange(w)[None, :] % tw
    return tile[rr, cc]


def _complete_axis_period(tile: np.ndarray, axis: int) -> int:
    h, w = tile.shape
    limit = h if axis == 0 else w
    for period in range(1, limit + 1):
        ok = True
        if axis == 0:
            for r in range(h):
                if not np.array_equal(tile[r], tile[r % period]):
                    ok = False
                    break
        else:
            for c in range(w):
                if not np.array_equal(tile[:, c], tile[:, c % period]):
                    ok = False
                    break
        if ok:
            return int(period)
    raise AssertionError("period search failed")


def _minimal_visible_period(block: np.ndarray, axis: int) -> int:
    h, w = block.shape
    limit = h if axis == 0 else w
    for period in range(1, limit + 1):
        ok = True
        if axis == 0:
            for r in range(period, h):
                row = block[r]
                prev = block[r - period]
                mask = (row != BLACK) & (prev != BLACK)
                if np.any(row[mask] != prev[mask]):
                    ok = False
                    break
        else:
            for c in range(period, w):
                col = block[:, c]
                prev = block[:, c - period]
                mask = (col != BLACK) & (prev != BLACK)
                if np.any(col[mask] != prev[mask]):
                    ok = False
                    break
        if ok:
            return int(period)
    raise AssertionError("visible-period search failed")


def _infer_tile_from_block(block: np.ndarray) -> np.ndarray | None:
    if block.size == 0:
        return None
    colors = _colors_present(block)
    if len(colors) != 2:
        return None

    pr = _minimal_visible_period(block, axis=0)
    pc = _minimal_visible_period(block, axis=1)

    residue_sets = [[set() for _ in range(pc)] for _ in range(pr)]
    coords = np.argwhere(block != BLACK)
    for r, c in coords:
        residue_sets[int(r) % pr][int(c) % pc].add(int(block[r, c]))

    if any(len(cell) != 1 for row in residue_sets for cell in row):
        return None

    tile = np.empty((pr, pc), dtype=int)
    for r in range(pr):
        for c in range(pc):
            tile[r, c] = next(iter(residue_sets[r][c]))

    pr2 = _complete_axis_period(tile, axis=0)
    pc2 = _complete_axis_period(tile, axis=1)
    tile = tile[:pr2, :pc2]
    if len(set(map(int, tile.flat))) != 2:
        return None

    expected = _tile_fill(block.shape, tile)
    visible = block != BLACK
    if np.any(block[visible] != expected[visible]):
        return None
    return tile


def _is_patch_band(band: np.ndarray) -> bool:
    if band.size == 0:
        return False
    mask = band != BLACK
    if np.any(np.count_nonzero(mask, axis=0) > 1):
        return False
    coords = np.argwhere(mask)
    for r, c in coords:
        r0 = max(0, int(r) - 1)
        r1 = min(band.shape[0], int(r) + 2)
        c0 = max(0, int(c) - 1)
        c1 = min(band.shape[1], int(c) + 2)
        if int(np.count_nonzero(mask[r0:r1, c0:c1])) != 1:
            return False
    return True


def _locate_pattern(grid: np.ndarray) -> tuple[int, np.ndarray]:
    h, _w = grid.shape
    for top in range(h - 1, 0, -1):
        band = grid[:top]
        if not _is_patch_band(band):
            continue
        tile = _infer_tile_from_block(grid[top:])
        if tile is not None:
            return int(top), tile
    raise AssertionError("failed to isolate patch band and periodic pattern")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair only those periodic holes whose columns contain a matching-color patch above them.

    Input structure:
    - The lower part is a full-width pattern zone made by repeating one unique minimal rectangular tile in exactly two non-black colors, with some cells blacked out as holes.
    - Above it is a black patch band containing isolated one-pixel patches, with at most one patch per column and patch colors drawn from the same two pattern colors.

    Rule:
    - Infer the periodic tile from the visible non-black cells in the lower pattern zone.
    - For each patch, scan downward within that same column and repair only the first black hole whose expected tile color matches the patch color.
    - Patches vanish in the output, unmatched holes stay black, and the rest of the pattern remains unchanged.

    Key insight:
    - The tile only tells you what every hole wants to be; the actual repaint is gated by a same-column, same-color single-use patch, so many holes must remain unfilled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pattern_top, tile = _locate_pattern(g)
    pattern = g[pattern_top:]
    expected = _tile_fill(pattern.shape, tile)

    out = g.copy()
    out[:pattern_top, :] = BLACK

    patches = np.argwhere(g[:pattern_top] != BLACK)
    for patch_r, patch_c in patches:
        color = int(g[int(patch_r), int(patch_c)])
        col = int(patch_c)
        for r in range(pattern.shape[0]):
            if int(pattern[r, col]) != BLACK:
                continue
            if int(expected[r, col]) != color:
                continue
            out[pattern_top + r, col] = color
            break

    return out


def _bruteforce_minimal_tiles(block: np.ndarray) -> list[np.ndarray]:
    h, w = block.shape
    valid: list[np.ndarray] = []
    for th in range(1, h + 1):
        for tw in range(1, w + 1):
            residue_sets = [[set() for _ in range(tw)] for _ in range(th)]
            ok = True
            for r, c in np.argwhere(block != BLACK):
                cell = residue_sets[int(r) % th][int(c) % tw]
                cell.add(int(block[r, c]))
                if len(cell) > 1:
                    ok = False
                    break
            if not ok:
                continue
            if any(len(cell) != 1 for row in residue_sets for cell in row):
                continue
            tile = np.empty((th, tw), dtype=int)
            for r in range(th):
                for c in range(tw):
                    tile[r, c] = next(iter(residue_sets[r][c]))
            if len(set(map(int, tile.flat))) != 2:
                continue
            if _complete_axis_period(tile, axis=0) != th or _complete_axis_period(tile, axis=1) != tw:
                continue
            expected = _tile_fill(block.shape, tile)
            visible = block != BLACK
            if np.any(block[visible] != expected[visible]):
                continue
            valid.append(tile.copy())
    if not valid:
        return []
    best_area = min(int(tile.size) for tile in valid)
    return [tile for tile in valid if int(tile.size) == best_area]


def _make_tile(tile_h: int, tile_w: int, colors: tuple[int, int]) -> np.ndarray | None:
    for _ in range(120):
        bits = np.random.randint(0, 2, size=(tile_h, tile_w))
        tile = np.where(bits == 0, colors[0], colors[1]).astype(int)
        if len(set(map(int, tile.flat))) != 2:
            continue
        if _complete_axis_period(tile, axis=0) != tile_h or _complete_axis_period(tile, axis=1) != tile_w:
            continue
        if not any(len(set(map(int, tile[:, c]))) == 2 for c in range(tile_w)):
            continue
        return tile
    return None


def _columns_with_gap(width: int, chosen: list[int], candidates: list[int]) -> list[int]:
    return [c for c in candidates if all(abs(c - other) >= 2 for other in chosen) and 0 <= c < width]


def _choose_role_columns(expected: np.ndarray, tile_h: int) -> dict[str, int] | None:
    pattern_h, width = expected.shape
    mixed_cols = [c for c in range(width) if len(set(map(int, expected[:, c]))) == 2]
    if not mixed_cols:
        return None

    for _ in range(80):
        role_cols: dict[str, int] = {}
        chosen: list[int] = []

        col_c_options = _columns_with_gap(width, chosen, mixed_cols)
        if not col_c_options:
            continue
        role_cols["C"] = random.choice(col_c_options)
        chosen.append(role_cols["C"])

        col_d_options = _columns_with_gap(width, chosen, list(range(width)))
        if not col_d_options:
            continue
        role_cols["D"] = random.choice(col_d_options)
        chosen.append(role_cols["D"])

        col_a_options = _columns_with_gap(width, chosen, list(range(width)))
        if not col_a_options:
            continue
        role_cols["A"] = random.choice(col_a_options)
        chosen.append(role_cols["A"])

        col_b_options = _columns_with_gap(width, chosen, list(range(width)))
        if not col_b_options:
            continue
        role_cols["B"] = random.choice(col_b_options)
        chosen.append(role_cols["B"])

        if pattern_h > tile_h + 1:
            return role_cols
    return None


def _pick_diff_pair(expected: np.ndarray, col: int) -> tuple[int, int] | None:
    col_vals = expected[:, col]
    for r0 in range(len(col_vals) - 1):
        for r1 in range(r0 + 1, len(col_vals)):
            if int(col_vals[r0]) != int(col_vals[r1]):
                return int(r0), int(r1)
    return None


def _plan_instance() -> tuple[np.ndarray, np.ndarray] | None:
    width = random.randint(10, 18)
    band_h = random.randint(3, 5)
    pattern_h = random.randint(7, min(14, 30 - band_h))
    colors = tuple(random.sample(NON_BLACK, 2))

    tile_h = random.randint(2, min(4, pattern_h - 2))
    tile_w = random.randint(2, min(5, width - 2))
    tile = _make_tile(tile_h, tile_w, colors)
    if tile is None:
        return None

    expected = _tile_fill((pattern_h, width), tile)
    role_cols = _choose_role_columns(expected, tile_h)
    if role_cols is None:
        return None

    diff_pair = _pick_diff_pair(expected, role_cols["C"])
    if diff_pair is None:
        return None

    hole_mask = np.zeros((pattern_h, width), dtype=bool)
    patch_specs: dict[int, int] = {}
    semantic_targets: dict[str, tuple[int, int]] = {}

    # A: simple successful repair.
    row_a = random.randint(0, pattern_h - 1)
    col_a = role_cols["A"]
    hole_mask[row_a, col_a] = True
    patch_specs[col_a] = int(expected[row_a, col_a])
    semantic_targets["A"] = (row_a, col_a)

    # B: patch vanishes because the column has only the opposite-color hole.
    row_b = random.randint(0, pattern_h - 1)
    col_b = role_cols["B"]
    hole_mask[row_b, col_b] = True
    patch_specs[col_b] = colors[0] if int(expected[row_b, col_b]) == colors[1] else colors[1]
    semantic_targets["B"] = (row_b, col_b)

    # C: the first hole in the column is wrong-colored; patch repairs the deeper match.
    row_c0, row_c1 = diff_pair
    col_c = role_cols["C"]
    hole_mask[row_c0, col_c] = True
    hole_mask[row_c1, col_c] = True
    patch_specs[col_c] = int(expected[row_c1, col_c])
    semantic_targets["C_top"] = (row_c0, col_c)
    semantic_targets["C_fix"] = (row_c1, col_c)

    # D: one patch with two same-color holes repairs only the upper one.
    col_d = role_cols["D"]
    row_d0 = random.randint(0, pattern_h - tile_h - 1)
    row_d1 = row_d0 + tile_h
    hole_mask[row_d0, col_d] = True
    hole_mask[row_d1, col_d] = True
    patch_specs[col_d] = int(expected[row_d0, col_d])
    semantic_targets["D_fix"] = (row_d0, col_d)
    semantic_targets["D_keep"] = (row_d1, col_d)

    # Extra unpatched holes keep the scene from collapsing into a full repair task.
    safe_cols = [c for c in range(width) if c not in patch_specs]
    random.shuffle(safe_cols)
    extra_holes = random.randint(1, 4)
    placed_extra = 0
    for col in safe_cols:
        row = random.randint(0, pattern_h - 1)
        if hole_mask[row, col]:
            continue
        hole_mask[row, col] = True
        placed_extra += 1
        if placed_extra == extra_holes:
            break
    if placed_extra == 0:
        return None

    pattern_in = expected.copy()
    pattern_in[hole_mask] = BLACK

    if np.any(np.count_nonzero(pattern_in, axis=1) < max(3, width // 2)):
        return None

    for tr in range(tile_h):
        for tc in range(tile_w):
            residue = pattern_in[tr::tile_h, tc::tile_w]
            if not np.any(residue != BLACK):
                return None

    if pattern_in[0].tolist().count(BLACK) > width // 3:
        return None

    band = np.zeros((band_h, width), dtype=int)
    patch_row_choices = list(range(band_h - 1))
    used_rows: dict[int, int] = {}
    for col in sorted(patch_specs):
        row = random.choice(patch_row_choices)
        used_rows[col] = row
        band[row, col] = int(patch_specs[col])

    if not _is_patch_band(band):
        return None

    inp = np.vstack([band, pattern_in])
    out = np.vstack([np.zeros_like(band), pattern_in.copy()])
    for col, color in patch_specs.items():
        for row in range(pattern_h):
            if not hole_mask[row, col]:
                continue
            if int(expected[row, col]) != int(color):
                continue
            out[band_h + row, col] = int(color)
            break

    # Independent semantic validation.
    tiles = _bruteforce_minimal_tiles(pattern_in)
    if len(tiles) != 1 or not np.array_equal(tiles[0], tile):
        return None
    if _infer_tile_from_block(pattern_in) is None or not np.array_equal(_infer_tile_from_block(pattern_in), tile):
        return None

    row, col = semantic_targets["A"]
    if int(out[band_h + row, col]) != int(expected[row, col]):
        return None
    row, col = semantic_targets["B"]
    if int(out[band_h + row, col]) != BLACK:
        return None
    row, col = semantic_targets["C_top"]
    if int(out[band_h + row, col]) != BLACK:
        return None
    row, col = semantic_targets["C_fix"]
    if int(out[band_h + row, col]) != int(expected[row, col]):
        return None
    row, col = semantic_targets["D_fix"]
    if int(out[band_h + row, col]) != int(expected[row, col]):
        return None
    row, col = semantic_targets["D_keep"]
    if int(out[band_h + row, col]) != BLACK:
        return None
    if not np.any((expected != BLACK) & (out[band_h:] == BLACK)):
        return None

    try:
        solver_out = main(inp)
    except Exception:
        return None
    if not np.array_equal(solver_out, out):
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        pair = _plan_instance()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid periodic-patch instance")
