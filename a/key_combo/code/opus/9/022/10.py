from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


MISSING = 0
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_SIZE_RANGE = (10, 12)
TILE_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]
MISSING_FRACTION_RANGE = (0.20, 0.40)
BLOB_COUNT_CHOICES = (2, 3, 4)
MIN_BLOB_SIZE = 4
MAX_GEN_ATTEMPTS = 600
SHAPE_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 120
COLOR_POOL = tuple(range(1, 10))


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox_of_cells(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _minimal_period(tile: np.ndarray) -> tuple[int, int]:
    tile = np.asarray(tile, dtype=int)
    h, w = map(int, tile.shape)
    row_ids = np.arange(h)[:, None]
    col_ids = np.arange(w)[None, :]
    for th in range(1, h + 1):
        for tw in range(1, w + 1):
            if np.array_equal(tile, tile[row_ids % th, col_ids % tw]):
                return int(th), int(tw)
    return h, w


def _valid_tile_candidates(grid: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    known = g != MISSING
    known_cells = np.argwhere(known)
    candidates: list[tuple[int, int, np.ndarray]] = []
    for th in range(1, h + 1):
        for tw in range(1, w + 1):
            tile = np.full((th, tw), -1, dtype=int)
            ok = True
            for r, c in known_cells:
                rr = int(r) % th
                cc = int(c) % tw
                value = int(g[r, c])
                if tile[rr, cc] == -1:
                    tile[rr, cc] = value
                elif tile[rr, cc] != value:
                    ok = False
                    break
            if ok and np.all(tile != -1):
                candidates.append((int(th), int(tw), tile.copy()))
    return candidates


def _infer_unique_minimal_tile(grid: np.ndarray) -> np.ndarray:
    candidates = _valid_tile_candidates(grid)
    assert candidates, "expected at least one consistent repeating tile"
    best_area = min(int(th * tw) for th, tw, _tile in candidates)
    best = [(th, tw, tile) for th, tw, tile in candidates if int(th * tw) == best_area]
    assert len(best) == 1, "expected a unique minimal repeating tile"
    tile = best[0][2]
    colors = sorted(int(x) for x in np.unique(tile))
    assert len(colors) == 2 and MISSING not in colors, "expected exactly two non-black tile colors"
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the hidden two-color wallpaper tile and fill black holes with its color-swapped prediction.

    Input structure:
    - The whole grid is covered by a repeating two-color wallpaper; black(0) cells are missing pixels, not background.
    - The latent minimal tile has size 2x2 to 3x3, and every tile phase remains visible at least once in a non-black cell.
    - Missing pixels form several connected black blobs, while all original non-black pixels stay in place.

    Rule:
    - Ignore black cells and infer the unique minimal-area repeating tile consistent with all visible pixels.
    - Read the predicted wallpaper color for each black cell from that tile by phase `(r mod tile_h, c mod tile_w)`.
    - Fill each black cell with the other tile color instead of the predicted one.
    - Keep every original non-black cell unchanged.

    Key insight:
    - The missing cells are not repaired with the wallpaper itself; they are repaired with the wallpaper's phase-correct opposite color, so periodic inference and color swapping are both necessary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    tile = _infer_unique_minimal_tile(g)
    tile_h, tile_w = map(int, tile.shape)
    color_a, color_b = sorted(int(x) for x in np.unique(tile))

    row_ids = np.arange(g.shape[0])[:, None] % tile_h
    col_ids = np.arange(g.shape[1])[None, :] % tile_w
    predicted = tile[row_ids, col_ids]

    out = g.copy()
    missing = g == MISSING
    swapped = np.where(predicted == color_a, color_b, color_a)
    out[missing] = swapped[missing]

    assert np.array_equal(out[~missing], g[~missing]), "visible pixels must stay unchanged"
    return out


def _build_wallpaper(tile: np.ndarray, h: int, w: int) -> np.ndarray:
    row_ids = np.arange(h)[:, None] % tile.shape[0]
    col_ids = np.arange(w)[None, :] % tile.shape[1]
    return tile[row_ids, col_ids]


def _sample_full_period_tile(th: int, tw: int, colors: tuple[int, int]) -> np.ndarray:
    color_a, color_b = map(int, colors)
    for _ in range(200):
        bits = np.random.randint(0, 2, size=(th, tw), dtype=int)
        zeros = int((bits == 0).sum())
        ones = int((bits == 1).sum())
        if zeros == 0 or ones == 0:
            continue
        if th * tw >= 6 and min(zeros, ones) < 2:
            continue
        if _minimal_period(bits) != (th, tw):
            continue
        return np.where(bits == 0, color_a, color_b)
    raise RuntimeError("failed to sample a full-period tile")


def _partition_total(total: int, parts: int, minimum: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total - parts * minimum + parts), parts - 1))
    shifted = [0] + cuts + [total - parts * minimum + parts]
    result = []
    for i in range(parts):
        result.append(int(shifted[i + 1] - shifted[i] - 1 + minimum))
    random.shuffle(result)
    return result


def _sample_blob_shape(size: int, max_h: int, max_w: int) -> np.ndarray | None:
    dims = []
    for h in range(2, max_h + 1):
        for w in range(2, max_w + 1):
            area = h * w
            if size >= area or size < max(4, math.ceil(area * 0.45)):
                continue
            dims.append((h, w))
    if not dims:
        return None

    for _ in range(SHAPE_ATTEMPTS):
        h, w = random.choice(dims)
        seed = (random.randrange(h), random.randrange(w))
        cells = {seed}
        while len(cells) < size:
            frontier = []
            for r, c in list(cells):
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != size:
            continue

        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if len(rows) < 2 or len(cols) < 2:
            continue

        row_counts = [sum(1 for rr, _ in cells if rr == r) for r in range(h)]
        col_counts = [sum(1 for _, cc in cells if cc == c) for c in range(w)]
        if len({count for count in row_counts if count > 0}) == 1 and len({count for count in col_counts if count > 0}) == 1:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if len(_components4(mask)) != 1:
            continue
        return mask
    return None


def _touches_existing(mask: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = map(int, mask.shape)
    for r, c in cells:
        if bool(mask[r, c]):
            return True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]):
                return True
    return False


def _phase_counts(h: int, w: int, th: int, tw: int) -> np.ndarray:
    counts = np.zeros((th, tw), dtype=int)
    for r in range(h):
        for c in range(w):
            counts[r % th, c % tw] += 1
    return counts


def _place_blob(
    missing: np.ndarray,
    shape: np.ndarray,
    phase_counts: np.ndarray,
    th: int,
    tw: int,
) -> bool:
    h, w = map(int, missing.shape)
    sh, sw = map(int, shape.shape)
    offsets = [(int(r), int(c)) for r, c in np.argwhere(shape)]
    choices = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            cells = [(int(r0 + r), int(c0 + c)) for r, c in offsets]
            if _touches_existing(missing, cells):
                continue
            next_counts = phase_counts.copy()
            for r, c in cells:
                next_counts[r % th, c % tw] -= 1
            if np.any(next_counts <= 0):
                continue
            touches_border = any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)
            bbox = _bbox_of_cells(cells)
            span = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
            choices.append((touches_border, span, r0, c0, next_counts))
    if not choices:
        return False

    border_choices = [item for item in choices if item[0]]
    if border_choices and random.random() < 0.35:
        chosen = random.choice(border_choices)
    else:
        chosen = random.choice(choices)
    _touches_border, _span, r0, c0, next_counts = chosen
    for r, c in offsets:
        missing[r0 + r, c0 + c] = True
    phase_counts[:, :] = next_counts
    return True


def _expected_output(wallpaper: np.ndarray, missing: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    color_a, color_b = map(int, colors)
    out = wallpaper.copy()
    out[missing] = np.where(wallpaper[missing] == color_a, color_b, color_a)
    return out


def _validate_example(inp: np.ndarray, wallpaper: np.ndarray, tile: np.ndarray, missing: np.ndarray) -> bool:
    h, w = map(int, inp.shape)
    th, tw = map(int, tile.shape)
    colors = tuple(sorted(int(x) for x in np.unique(tile)))
    if len(colors) != 2 or MISSING in colors:
        return False

    if set(map(int, np.unique(inp))) != {MISSING, colors[0], colors[1]}:
        return False

    total = h * w
    missing_count = int(missing.sum())
    if not (math.ceil(total * MISSING_FRACTION_RANGE[0]) <= missing_count <= math.floor(total * MISSING_FRACTION_RANGE[1])):
        return False

    comps = _components4(missing)
    if len(comps) < 2:
        return False

    phase_seen = np.zeros((th, tw), dtype=bool)
    for r in range(h):
        for c in range(w):
            if int(inp[r, c]) != MISSING:
                phase_seen[r % th, c % tw] = True
    if not bool(phase_seen.all()):
        return False

    hidden_colors = {int(wallpaper[r, c]) for r, c in np.argwhere(missing)}
    if len(hidden_colors) != 2:
        return False

    for comp in comps:
        if len(comp) < MIN_BLOB_SIZE:
            return False
        r0, c0, r1, c1 = _bbox_of_cells(comp)
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if bbox_area == len(comp):
            return False

    try:
        inferred = _infer_unique_minimal_tile(inp)
    except AssertionError:
        return False
    if inferred.shape != tile.shape or not np.array_equal(inferred, tile):
        return False

    candidates = _valid_tile_candidates(inp)
    min_area = min(int(a * b) for a, b, _ in candidates)
    if min_area != int(th * tw):
        return False

    expected = _expected_output(wallpaper, missing, colors)
    produced = main(inp)
    if not np.array_equal(produced, expected):
        return False
    if np.array_equal(produced, wallpaper):
        return False
    if np.any(produced == MISSING):
        return False
    if not np.array_equal(produced[~missing], wallpaper[~missing]):
        return False
    if np.any(produced[missing] == wallpaper[missing]):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    th, tw = random.choice(TILE_DIMS)
    color_a, color_b = random.sample(COLOR_POOL, 2)
    tile = _sample_full_period_tile(th, tw, (color_a, color_b))
    wallpaper = _build_wallpaper(tile, h, w)

    total = h * w
    min_missing = math.ceil(total * MISSING_FRACTION_RANGE[0])
    max_missing = math.floor(total * MISSING_FRACTION_RANGE[1])
    blob_count = random.choice(BLOB_COUNT_CHOICES)
    if max_missing < blob_count * MIN_BLOB_SIZE:
        return None
    target_missing = random.randint(min_missing, max_missing)
    if target_missing < blob_count * MIN_BLOB_SIZE:
        target_missing = blob_count * MIN_BLOB_SIZE
    sizes = _partition_total(target_missing, blob_count, MIN_BLOB_SIZE)

    missing = np.zeros((h, w), dtype=bool)
    phase_counts = _phase_counts(h, w, th, tw)

    for size in sizes:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            shape = _sample_blob_shape(size, min(h, 7), min(w, 7))
            if shape is None:
                continue
            if _place_blob(missing, shape, phase_counts, th, tw):
                placed = True
                break
        if not placed:
            return None

    inp = wallpaper.copy()
    inp[missing] = MISSING

    if not _validate_example(inp, wallpaper, tile, missing):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic color-swap example")
