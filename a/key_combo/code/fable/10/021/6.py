from __future__ import annotations

import random

import numpy as np


GEN_PERIOD_MIN = 2
GEN_PERIOD_MAX = 6
GEN_GRID_MIN = 14
GEN_GRID_MAX = 28
GEN_WINDOW_MIN = 4
GEN_RETRY_LIMIT = 500
GEN_WINDOW_TRIES = 250
GEN_TILE_TRIES = 120
GEN_COLORS = tuple(range(1, 10))


def _find_window_marker(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    values, counts = np.unique(grid, return_counts=True)
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for value, count in zip(values.tolist(), counts.tolist()):
        if int(count) != 4:
            continue
        coords = np.argwhere(grid == int(value))
        rows = sorted({int(r) for r, _c in coords})
        cols = sorted({int(c) for _r, c in coords})
        if len(rows) != 2 or len(cols) != 2:
            continue
        corners = {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}
        if {tuple(map(int, rc)) for rc in coords.tolist()} != corners:
            continue
        candidates.append((int(value), (rows[0], cols[0], rows[1], cols[1])))
    assert len(candidates) == 1, "expected exactly one 4-corner window marker color"
    return candidates[0]


def _infer_periodic_tile(grid: np.ndarray, visible_mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = grid.shape
    coords = np.argwhere(visible_mask)
    assert len(coords) > 0, "need visible evidence outside the window"

    best_key: tuple[int, int, int, int] | None = None
    best_tile: np.ndarray | None = None

    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            if len(coords) < pr * pc:
                continue
            tile = np.full((pr, pc), -1, dtype=int)
            seen = np.zeros((pr, pc), dtype=bool)
            ok = True
            for r, c in coords:
                rr = int(r) % pr
                cc = int(c) % pc
                value = int(grid[r, c])
                if seen[rr, cc]:
                    if int(tile[rr, cc]) != value:
                        ok = False
                        break
                else:
                    tile[rr, cc] = value
                    seen[rr, cc] = True
            if ok and bool(seen.all()):
                key = (pr * pc, pr + pc, pr, pc)
                if best_key is None or key < best_key:
                    best_key = key
                    best_tile = tile.copy()

    assert best_tile is not None and best_key is not None, "could not infer a complete smallest periodic tile"
    return best_tile, (best_key[2], best_key[3])


def _tile_full_grid(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    pr, pc = tile.shape
    rows = np.arange(h)[:, None] % pr
    cols = np.arange(w)[None, :] % pc
    return tile[rows, cols].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the true periodic contents hidden behind a marked rectangular window and output only that window.

    Input structure:
    - The whole grid is intended to be a repetition of one small rectangular tile.
    - Four same-colored corner marker pixels, using a color that appears nowhere else, mark the corners of one axis-aligned window.
    - Every cell inside that window has been tampered with, while every cell outside the window still follows the original periodic tiling.

    Rule:
    - Use only the pixels outside the marked window to infer the smallest repeating tile that is consistent with the visible data.
    - Reconstruct the full periodic image from that tile.
    - Return only the marked window area, restored to its true periodic colors, including the four marked corners.

    Key insight:
    - The window contents are deliberate lies, so cropping the visible window fails; the only trustworthy evidence is the untouched periodic structure outside it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _marker, (r0, c0, r1, c1) = _find_window_marker(g)
    visible = np.ones(g.shape, dtype=bool)
    visible[r0 : r1 + 1, c0 : c1 + 1] = False

    tile, _period = _infer_periodic_tile(g, visible)
    restored = _tile_full_grid(tile, g.shape)
    assert np.array_equal(restored[visible], g[visible]), "outside-window evidence must match the inferred tile"
    return restored[r0 : r1 + 1, c0 : c1 + 1]


def _count_residues(shape: tuple[int, int], period: tuple[int, int], window: tuple[int, int, int, int] | None = None) -> np.ndarray:
    h, w = shape
    pr, pc = period
    counts = np.zeros((pr, pc), dtype=int)
    for r in range(h):
        for c in range(w):
            if window is not None:
                r0, c0, r1, c1 = window
                if r0 <= r <= r1 and c0 <= c <= c1:
                    continue
            counts[r % pr, c % pc] += 1
    return counts


def _minimal_period_of_full_grid(grid: np.ndarray) -> tuple[int, int]:
    visible = np.ones(grid.shape, dtype=bool)
    _, period = _infer_periodic_tile(grid, visible)
    return period


def _make_random_tile(pr: int, pc: int) -> np.ndarray | None:
    for _ in range(GEN_TILE_TRIES):
        color_count = random.randint(3, min(6, len(GEN_COLORS) - 2))
        palette = random.sample(GEN_COLORS, color_count)
        tile = np.array([[random.choice(palette) for _ in range(pc)] for _ in range(pr)], dtype=int)
        if len(set(map(int, tile.flatten()))) < 3:
            continue
        if len({tuple(map(int, row)) for row in tile.tolist()}) < 2:
            continue
        if len({tuple(map(int, col)) for col in tile.T.tolist()}) < 2:
            continue
        if _minimal_period_of_full_grid(tile) != (pr, pc):
            continue
        return tile
    return None


def _window_duplicate_exists(full: np.ndarray, window: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = window
    patch = full[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    H, W = full.shape
    for rr in range(H - h + 1):
        for cc in range(W - w + 1):
            if rr == r0 and cc == c0:
                continue
            if not (rr + h - 1 < r0 or rr > r1 or cc + w - 1 < c0 or cc > c1):
                continue
            if np.array_equal(full[rr : rr + h, cc : cc + w], patch):
                return True
    return False


def _best_fake_patch(tile: np.ndarray, full: np.ndarray, period: tuple[int, int], window: tuple[int, int, int, int]) -> np.ndarray | None:
    pr, pc = period
    r0, c0, r1, c1 = window
    rows = np.arange(r0, r1 + 1)
    cols = np.arange(c0, c1 + 1)
    truth = full[r0 : r1 + 1, c0 : c1 + 1]
    best_diff = -1
    best_patch: np.ndarray | None = None

    for dr in range(pr):
        for dc in range(pc):
            if dr == 0 and dc == 0:
                continue
            patch = tile[(rows[:, None] + dr) % pr, (cols[None, :] + dc) % pc]
            diff = int(np.count_nonzero(patch != truth))
            if diff > best_diff:
                best_diff = diff
                best_patch = patch.copy()

    if best_patch is None:
        return None

    threshold = max(4, truth.size // 2)
    if best_diff < threshold:
        return None

    return best_patch


def _add_extra_corruption(fake: np.ndarray, truth: np.ndarray, allowed_colors: list[int]) -> np.ndarray:
    if fake.shape[0] < 3 or fake.shape[1] < 3 or not allowed_colors:
        return fake

    out = fake.copy()
    accent = random.choice(allowed_colors)
    h, w = fake.shape

    if random.random() < 0.5:
        rr = random.randint(1, h - 2)
        out[rr, 1 : w - 1] = accent
    else:
        cc = random.randint(1, w - 2)
        out[1 : h - 1, cc] = accent

    if np.array_equal(out[1:-1, 1:-1], truth[1:-1, 1:-1]):
        return fake
    return out


def _all_marker_candidates(grid: np.ndarray) -> list[tuple[int, tuple[int, int, int, int]]]:
    values, counts = np.unique(grid, return_counts=True)
    found: list[tuple[int, tuple[int, int, int, int]]] = []
    for value, count in zip(values.tolist(), counts.tolist()):
        if int(count) != 4:
            continue
        coords = np.argwhere(grid == int(value))
        rows = sorted({int(r) for r, _c in coords})
        cols = sorted({int(c) for _r, c in coords})
        if len(rows) != 2 or len(cols) != 2:
            continue
        corners = {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}
        if {tuple(map(int, rc)) for rc in coords.tolist()} == corners:
            found.append((int(value), (rows[0], cols[0], rows[1], cols[1])))
    return found


def _build_attempt() -> np.ndarray | None:
    pr = random.randint(GEN_PERIOD_MIN, GEN_PERIOD_MAX)
    pc = random.randint(GEN_PERIOD_MIN, GEN_PERIOD_MAX)
    tile = _make_random_tile(pr, pc)
    if tile is None:
        return None

    h_min = max(GEN_GRID_MIN, pr * 3)
    h_max = max(h_min, min(GEN_GRID_MAX, pr * 4 + 6))
    w_min = max(GEN_GRID_MIN, pc * 3)
    w_max = max(w_min, min(GEN_GRID_MAX, pc * 4 + 6))
    h = random.randint(h_min, h_max)
    w = random.randint(w_min, w_max)
    full = _tile_full_grid(tile, (h, w))

    tile_colors = sorted({int(v) for v in tile.flatten()})
    marker_options = [color for color in GEN_COLORS if color not in tile_colors]
    if len(marker_options) < 2:
        return None
    marker = random.choice(marker_options)
    accent_options = [color for color in marker_options if color != marker]

    for _ in range(GEN_WINDOW_TRIES):
        win_h = random.randint(max(GEN_WINDOW_MIN, pr + 1), h - 2)
        win_w = random.randint(max(GEN_WINDOW_MIN, pc + 1), w - 2)
        if h - win_h - 1 < 1 or w - win_w - 1 < 1:
            continue
        r0 = random.randint(1, h - win_h - 1)
        c0 = random.randint(1, w - win_w - 1)
        window = (r0, c0, r0 + win_h - 1, c0 + win_w - 1)

        outside_counts = _count_residues((h, w), (pr, pc), window)
        if np.any(outside_counts == 0):
            continue
        if _window_duplicate_exists(full, window):
            continue

        fake = _best_fake_patch(tile, full, (pr, pc), window)
        if fake is None:
            continue
        truth = full[r0 : r0 + win_h, c0 : c0 + win_w]
        fake = _add_extra_corruption(fake, truth, accent_options)

        inp = full.copy()
        inp[r0 : r0 + win_h, c0 : c0 + win_w] = fake
        inp[r0, c0] = marker
        inp[r0, c0 + win_w - 1] = marker
        inp[r0 + win_h - 1, c0] = marker
        inp[r0 + win_h - 1, c0 + win_w - 1] = marker

        if np.array_equal(inp[r0 : r0 + win_h, c0 : c0 + win_w], truth):
            continue
        if np.count_nonzero(inp[r0 : r0 + win_h, c0 : c0 + win_w] != truth) < max(4, (win_h * win_w) // 2):
            continue
        if _all_marker_candidates(inp) != [(marker, window)]:
            continue

        visible = np.ones(inp.shape, dtype=bool)
        visible[r0 : r0 + win_h, c0 : c0 + win_w] = False
        inferred_tile, inferred_period = _infer_periodic_tile(inp, visible)
        if inferred_period != (pr, pc):
            continue
        if not np.array_equal(inferred_tile, tile):
            continue
        if not np.array_equal(main(inp), truth):
            continue
        return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid window-lie puzzle")
