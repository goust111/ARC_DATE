from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _render_from_tile(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    row_idx = np.arange(h, dtype=int)[:, None] % tile.shape[0]
    col_idx = np.arange(w, dtype=int)[None, :] % tile.shape[1]
    return tile[row_idx, col_idx]


def _majority_tile(
    grid: np.ndarray, row_period: int, col_period: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    counts = [[Counter() for _ in range(col_period)] for _ in range(row_period)]
    class_sizes = np.zeros((row_period, col_period), dtype=int)
    h, w = grid.shape
    for r in range(h):
        rr = r % row_period
        for c in range(w):
            cc = c % col_period
            color = int(grid[r, c])
            counts[rr][cc][color] += 1
            class_sizes[rr, cc] += 1

    tile = np.zeros((row_period, col_period), dtype=int)
    top_counts = np.zeros((row_period, col_period), dtype=int)
    for rr in range(row_period):
        for cc in range(col_period):
            ordered = counts[rr][cc].most_common(2)
            if not ordered:
                return None
            majority_color, majority_count = ordered[0]
            if majority_count * 2 <= class_sizes[rr, cc]:
                return None
            tile[rr, cc] = int(majority_color)
            top_counts[rr, cc] = int(majority_count)
    return tile, class_sizes, top_counts


def _candidate_shifts(
    tile: np.ndarray, grid: np.ndarray, patch_coords: np.ndarray
) -> list[tuple[int, int]]:
    row_period, col_period = tile.shape
    shifts: list[tuple[int, int]] = []
    for sr in range(row_period):
        for sc in range(col_period):
            if sr == 0 and sc == 0:
                continue
            ok = True
            for r, c in patch_coords:
                if int(tile[(r + sr) % row_period, (c + sc) % col_period]) != int(grid[r, c]):
                    ok = False
                    break
            if ok:
                shifts.append((sr, sc))
    return shifts


def _find_models(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    max_row_period = max(1, h // 3)
    max_col_period = max(1, w // 3)
    models: list[dict[str, object]] = []

    for row_period in range(1, max_row_period + 1):
        for col_period in range(1, max_col_period + 1):
            majority = _majority_tile(g, row_period, col_period)
            if majority is None:
                continue
            tile, class_sizes, _top_counts = majority
            standard = _render_from_tile(tile, g.shape)
            mismatch = g != standard
            if not np.any(mismatch):
                continue

            components = _mask_components(mismatch)
            if len(components) != 1:
                continue
            patch_coords = components[0]

            patch_counts = np.zeros((row_period, col_period), dtype=int)
            for r, c in patch_coords:
                patch_counts[r % row_period, c % col_period] += 1
            if np.any(patch_counts * 2 >= class_sizes):
                continue

            shifts = _candidate_shifts(tile, g, patch_coords)
            for shift_r, shift_c in shifts:
                models.append(
                    {
                        "row_period": row_period,
                        "col_period": col_period,
                        "shift": (shift_r, shift_c),
                        "tile": tile,
                        "standard": standard,
                        "patch_mask": mismatch,
                        "patch_coords": patch_coords,
                    }
                )
    return models


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal the phase-shifted patch hidden inside a periodic wallpaper and crop it out.

    Input structure:
    - The whole grid is filled with a non-black multi-color wallpaper that repeats by some row and column period.
    - One 4-connected patch has been overwritten by the same wallpaper seen at one different global phase.
    - Inside the patch, every cell still looks locally periodic, but its color disagrees with the wallpaper phase used by the rest of the grid.

    Rule:
    - Search the allowed period range and recover the wallpaper's standard color for each residue class by strict majority vote.
    - The cells that disagree with that recovered standard wallpaper are exactly the patch cells; they must form one 4-connected component and share one non-zero phase shift.
    - Crop to the patch bounding box and keep only the patch's shifted colors there; paint all other cells in that box black.

    Key insight:
    - The patch is not detectable by local texture, only by a global residue-coordinate system: once the true period is established, a single coherent phase conflict exposes the hidden component.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    models = _find_models(g)
    assert len(models) == 1, f"expected exactly one valid wallpaper model, found {len(models)}"
    patch_coords = models[0]["patch_coords"]
    assert isinstance(patch_coords, np.ndarray)
    r0, c0, r1, c1 = _bbox_from_coords(patch_coords)

    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in patch_coords:
        out[r - r0, c - c0] = int(g[r, c])
    return out


GEN_MAX_TRIES = 400
GEN_COLORS = tuple(range(1, 10))
GEN_ROW_PERIOD_RANGE = (3, 8)
GEN_COL_PERIOD_RANGE = (2, 8)
GEN_HEIGHT_MARGIN = (0, 8)
GEN_WIDTH_MARGIN = (0, 8)
GEN_PATCH_BBOX_MIN = 3
GEN_PATCH_BBOX_MAX = 11
GEN_PATCH_FILL_MIN_FRAC = 0.35
GEN_PATCH_FILL_MAX_FRAC = 0.72


def _has_subperiod(values: list[int]) -> bool:
    n = len(values)
    for p in range(1, n):
        if n % p == 0 and all(values[i] == values[i % p] for i in range(n)):
            return True
    return False


def _wallpaper_color(
    r: int, c: int, palette: list[int], offsets: list[int]
) -> int:
    row_period = len(palette)
    col_period = len(offsets)
    return int(palette[(r - offsets[c % col_period]) % row_period])


def _make_wallpaper(
    height: int, width: int, palette: list[int], offsets: list[int]
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            grid[r, c] = _wallpaper_color(r, c, palette, offsets)
    return grid


def _valid_shift_options(offsets: list[int], row_period: int) -> dict[str, list[tuple[int, int]]]:
    col_period = len(offsets)
    groups = {"row_only": [], "col_only": [], "mixed": []}
    for shift_r in range(row_period):
        for shift_c in range(col_period):
            if shift_r == 0 and shift_c == 0:
                continue
            if all(
                shift_r != (offsets[(c + shift_c) % col_period] - offsets[c]) % row_period
                for c in range(col_period)
            ):
                if shift_c == 0:
                    groups["row_only"].append((shift_r, shift_c))
                elif shift_r == 0:
                    groups["col_only"].append((shift_r, shift_c))
                else:
                    groups["mixed"].append((shift_r, shift_c))
    return groups


def _sample_offsets(row_period: int, col_period: int) -> tuple[list[int], tuple[int, int]] | None:
    for _ in range(120):
        offsets = [random.randrange(row_period) for _ in range(col_period)]
        if _has_subperiod(offsets):
            continue
        if len(set(offsets)) < min(col_period, 3):
            continue

        options = _valid_shift_options(offsets, row_period)
        categories = [name for name, vals in options.items() if vals]
        if not categories:
            continue
        category = random.choice(categories)
        shift = random.choice(options[category])
        return offsets, shift
    return None


def _grow_patch_mask(height: int, width: int) -> np.ndarray | None:
    area = height * width
    min_fill = max(5, int(np.ceil(area * GEN_PATCH_FILL_MIN_FRAC)))
    max_fill = min(area - 1, int(np.floor(area * GEN_PATCH_FILL_MAX_FRAC)))
    if min_fill > max_fill:
        return None

    target = random.randint(min_fill, max_fill)
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((nr, nc))

    while len(cells) < target and frontier:
        candidates = list(frontier)
        weights = []
        for rr, cc in candidates:
            degree = sum(
                (nr, nc) in cells
                for nr, nc in _neighbors4(rr, cc)
                if 0 <= nr < height and 0 <= nc < width
            )
            edge_bonus = int(rr in (0, height - 1) or cc in (0, width - 1))
            weights.append(1 + degree + edge_bonus)
        choice = random.choices(candidates, weights=weights, k=1)[0]
        frontier.discard(choice)
        cells.add(choice)
        for nr, nc in _neighbors4(*choice):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    if len(cells) != target:
        return None

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
    if cropped.all():
        return None
    if cropped.shape[0] < 2 or cropped.shape[1] < 2:
        return None
    return cropped


def _expected_output(grid: np.ndarray, patch_mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch_mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        out[r - r0, c - c0] = int(grid[r, c])
    return out


def _matches_known_model(
    models: list[dict[str, object]],
    row_period: int,
    col_period: int,
    shift: tuple[int, int],
    patch_mask: np.ndarray,
    wallpaper: np.ndarray,
) -> bool:
    if len(models) != 1:
        return False
    model = models[0]
    if model["row_period"] != row_period or model["col_period"] != col_period:
        return False
    if tuple(model["shift"]) != tuple(shift):
        return False
    if not np.array_equal(model["patch_mask"], patch_mask):
        return False
    return np.array_equal(model["standard"], wallpaper)


def _attempt() -> np.ndarray | None:
    row_period = random.randint(*GEN_ROW_PERIOD_RANGE)
    col_period = random.randint(*GEN_COL_PERIOD_RANGE)

    height = random.randint(3 * row_period, min(30, 3 * row_period + GEN_HEIGHT_MARGIN[1]))
    width = random.randint(3 * col_period, min(30, 3 * col_period + GEN_WIDTH_MARGIN[1]))

    sampled = _sample_offsets(row_period, col_period)
    if sampled is None:
        return None
    offsets, shift = sampled
    shift_r, shift_c = shift

    palette = random.sample(GEN_COLORS, row_period)
    wallpaper = _make_wallpaper(height, width, palette, offsets)

    patch_box_h = random.randint(
        GEN_PATCH_BBOX_MIN, min(GEN_PATCH_BBOX_MAX, height)
    )
    patch_box_w = random.randint(
        GEN_PATCH_BBOX_MIN, min(GEN_PATCH_BBOX_MAX, width)
    )
    patch_shape = _grow_patch_mask(patch_box_h, patch_box_w)
    if patch_shape is None:
        return None

    patch_h, patch_w = patch_shape.shape
    top = random.randint(0, height - patch_h)
    left = random.randint(0, width - patch_w)
    patch_mask = np.zeros((height, width), dtype=bool)
    patch_mask[top : top + patch_h, left : left + patch_w] = patch_shape

    patch_coords = np.argwhere(patch_mask)
    if len(_mask_components(patch_mask)) != 1:
        return None

    residue_totals = np.zeros((row_period, col_period), dtype=int)
    residue_patch = np.zeros((row_period, col_period), dtype=int)
    for r in range(height):
        for c in range(width):
            residue_totals[r % row_period, c % col_period] += 1
    for r, c in patch_coords:
        residue_patch[r % row_period, c % col_period] += 1
    if np.any(residue_patch * 2 >= residue_totals):
        return None

    inp = wallpaper.copy()
    for r, c in patch_coords:
        shifted = _wallpaper_color(r + shift_r, c + shift_c, palette, offsets)
        if shifted == int(wallpaper[r, c]):
            return None
        inp[r, c] = shifted

    if 0 in set(map(int, inp.ravel())):
        return None

    models = _find_models(inp)
    if not _matches_known_model(models, row_period, col_period, shift, patch_mask, wallpaper):
        return None

    expected = _expected_output(inp, patch_mask)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_TRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a unique phase-shifted wallpaper patch")
