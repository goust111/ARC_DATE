import random
from typing import Iterable

import numpy as np


ORANGE = 7
BLUE = 1
BLACK = 0
MARK_COLORS = (2, 3, 4, 6)

GEN_ATTEMPTS = 400
LAYOUTS = ((1, 3), (3, 1), (2, 2))
MIN_DENSITY = 0.30
MAX_DENSITY = 0.50
MIN_MISSING = 0.10
MAX_MISSING = 0.20


def _crop_non_color(grid: np.ndarray, color: int) -> np.ndarray:
    coords = np.argwhere(grid != int(color))
    assert coords.size > 0, "expected a non-background arrangement"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.array(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)


def _bands(length: int, separators: Iterable[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for idx in sorted(int(x) for x in separators):
        if start < idx:
            bands.append((int(start), int(idx)))
        start = int(idx) + 1
    if start < int(length):
        bands.append((int(start), int(length)))
    return bands


def _extract_panels(grid: np.ndarray) -> list[np.ndarray]:
    crop = _crop_non_color(np.asarray(grid, dtype=int), ORANGE)
    sep_rows = [r for r in range(crop.shape[0]) if np.all(crop[r] == BLUE)]
    sep_cols = [c for c in range(crop.shape[1]) if np.all(crop[:, c] == BLUE)]
    row_bands = _bands(crop.shape[0], sep_rows)
    col_bands = _bands(crop.shape[1], sep_cols)
    panels = [crop[r0:r1, c0:c1].copy() for r0, r1 in row_bands for c0, c1 in col_bands]
    assert 3 <= len(panels) <= 4, "expected 3 or 4 panels"
    shapes = {tuple(panel.shape) for panel in panels}
    assert len(shapes) == 1, "panels must be equal-sized"
    for panel in panels:
        assert panel.size > 0
        assert not np.any(panel == ORANGE)
        assert not np.any(panel == BLUE)
    return panels


def _overlay_panels(panels: list[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(panels[0], dtype=int)
    for panel in panels:
        mask = panel != BLACK
        clash = mask & (out != BLACK) & (out != panel)
        assert not np.any(clash), "panel overlay conflict"
        out[mask] = panel[mask]
    return out


def _tile_to_shape(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows = np.arange(shape[0])[:, None] % tile.shape[0]
    cols = np.arange(shape[1])[None, :] % tile.shape[1]
    return np.asarray(tile[rows, cols], dtype=int)


def _candidate_tile(observed: np.ndarray, period_h: int, period_w: int) -> np.ndarray | None:
    tile = np.zeros((period_h, period_w), dtype=int)
    seen = np.zeros((period_h, period_w), dtype=bool)
    for r, c in np.argwhere(observed != BLACK):
        rr = int(r) % int(period_h)
        cc = int(c) % int(period_w)
        color = int(observed[int(r), int(c)])
        if not seen[rr, cc]:
            tile[rr, cc] = color
            seen[rr, cc] = True
        elif int(tile[rr, cc]) != color:
            return None
    if not np.all(seen):
        return None
    return tile


def _infer_minimal_tile(observed: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed, dtype=int)
    assert observed.ndim == 2
    best_area = None
    best: list[np.ndarray] = []
    H, W = map(int, observed.shape)
    for period_h in range(1, H + 1):
        for period_w in range(1, W + 1):
            area = int(period_h * period_w)
            if best_area is not None and area > best_area:
                continue
            tile = _candidate_tile(observed, period_h, period_w)
            if tile is None:
                continue
            if best_area is None or area < best_area:
                best_area = area
                best = [tile.copy()]
            elif area == best_area:
                best.append(tile.copy())
    assert best, "no fully observed periodic tile fits the data"
    unique: list[np.ndarray] = []
    for tile in best:
        if not any(tile.shape == other.shape and np.array_equal(tile, other) for other in unique):
            unique.append(tile)
    assert len(unique) == 1, "minimal periodic explanation must be unique"
    return unique[0]


def _phase_index(r: int, c: int, period: int) -> int:
    return int(r % period) * int(period) + int(c % period)


def _panel_is_independently_sufficient(panel: np.ndarray, target: np.ndarray) -> bool:
    try:
        tile = _infer_minimal_tile(panel)
    except AssertionError:
        return False
    rebuilt = _tile_to_shape(tile, target.shape)
    return bool(np.array_equal(rebuilt, target))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay sparse same-sized panels, infer the hidden repeating tile, and fill the remaining gaps.

    Input structure:
    - Orange background surrounds a 3-panel or 4-panel arrangement separated by full blue rows and/or columns.
    - Each panel has the same size, uses black as its local blank color, and contains sparse colored marks.
    - All non-black marks come from the same latent periodic pattern at the same panel-relative coordinates.

    Rule:
    - Remove the orange frame and blue separator lines, then extract every same-sized panel.
    - Transparently overlay all panels into one black canvas by copying every non-black pixel to its matching coordinate.
    - Treat the black cells left after overlay as missing data, infer the unique minimal repeating tile consistent with all non-black cells, and use that tile to fill every missing cell.

    Key insight:
    - No single panel is complete enough on its own; only after overlaying their partial observations does the periodic structure become inferable and usable to repair the final holes.
    """
    panels = _extract_panels(np.asarray(grid, dtype=int))
    overlay = _overlay_panels(panels)
    tile = _infer_minimal_tile(overlay)
    out = _tile_to_shape(tile, overlay.shape)
    assert np.array_equal(out[overlay != BLACK], overlay[overlay != BLACK])
    return out


def _random_tile(period: int) -> np.ndarray:
    for _ in range(300):
        n_colors = random.randint(2, len(MARK_COLORS))
        palette = random.sample(MARK_COLORS, n_colors)
        tile = np.array(
            [[random.choice(palette) for _ in range(period)] for _ in range(period)],
            dtype=int,
        )
        if len(set(map(int, tile.flat))) < 2:
            continue
        if _infer_minimal_tile(tile).shape == (period, period):
            return tile
    raise RuntimeError("failed to build a minimal tile")


def _sample_missing_mask(shape: tuple[int, int], period: int) -> np.ndarray | None:
    H, W = map(int, shape)
    area = int(H * W)
    min_missing = max(1, int(np.ceil(area * MIN_MISSING)))
    max_missing = max(min_missing, int(np.floor(area * MAX_MISSING)))
    positions = [(r, c) for r in range(H) for c in range(W)]
    phase_counts = [0] * (period * period)
    for r, c in positions:
        phase_counts[_phase_index(r, c, period)] += 1
    for _ in range(200):
        target_missing = random.randint(min_missing, max_missing)
        random.shuffle(positions)
        remaining = phase_counts[:]
        missing = np.zeros((H, W), dtype=bool)
        picked = 0
        for r, c in positions:
            phase = _phase_index(r, c, period)
            if remaining[phase] <= 1:
                continue
            if picked >= target_missing:
                break
            missing[r, c] = True
            remaining[phase] -= 1
            picked += 1
        if picked != target_missing:
            continue
        if all(count >= 1 for count in remaining):
            return missing
    return None


def _allocate_masks(shape: tuple[int, int], period: int, panel_count: int, missing: np.ndarray) -> list[np.ndarray] | None:
    H, W = map(int, shape)
    area = int(H * W)
    min_count = int(np.ceil(area * MIN_DENSITY))
    max_count = int(np.floor(area * MAX_DENSITY))
    if min_count > max_count:
        return None
    phases = list(range(period * period))
    for _ in range(250):
        forbidden = [random.choice(phases) for _ in range(panel_count)]
        if len(set(forbidden)) == 1:
            continue
        allowed = np.zeros((panel_count, H, W), dtype=bool)
        for j in range(panel_count):
            for r in range(H):
                for c in range(W):
                    allowed[j, r, c] = (not missing[r, c]) and (_phase_index(r, c, period) != forbidden[j])
        targets = [random.randint(min_count, max_count) for _ in range(panel_count)]
        masks = np.zeros((panel_count, H, W), dtype=bool)
        counts = [0] * panel_count
        coords = [(r, c) for r in range(H) for c in range(W) if not missing[r, c]]
        random.shuffle(coords)
        ok = True
        for r, c in coords:
            candidates = [j for j in range(panel_count) if allowed[j, r, c] and counts[j] < max_count]
            if not candidates:
                ok = False
                break
            candidates.sort(key=lambda j: (counts[j] / max(1, targets[j]), counts[j], random.random()))
            chosen = candidates[0]
            masks[chosen, r, c] = True
            counts[chosen] += 1
        if not ok:
            continue
        order = list(range(panel_count))
        random.shuffle(order)
        for j in order:
            extras = [(r, c) for r in range(H) for c in range(W) if allowed[j, r, c] and not masks[j, r, c]]
            random.shuffle(extras)
            for r, c in extras:
                if counts[j] >= targets[j]:
                    break
                masks[j, r, c] = True
                counts[j] += 1
        if any(count < min_count or count > max_count for count in counts):
            continue
        return [masks[j].copy() for j in range(panel_count)]
    return None


def _build_input_from_masks(
    target: np.ndarray,
    masks: list[np.ndarray],
    layout: tuple[int, int],
) -> np.ndarray:
    panel_rows, panel_cols = map(int, layout)
    panel_h, panel_w = map(int, target.shape)
    arr_h = panel_rows * panel_h + (panel_rows - 1)
    arr_w = panel_cols * panel_w + (panel_cols - 1)
    grid = np.full((arr_h + 2, arr_w + 2), ORANGE, dtype=int)
    crop = np.zeros((arr_h, arr_w), dtype=int)
    if panel_rows > 1:
        for idx in range(1, panel_rows):
            crop[idx * panel_h + (idx - 1), :] = BLUE
    if panel_cols > 1:
        for idx in range(1, panel_cols):
            crop[:, idx * panel_w + (idx - 1)] = BLUE
    for idx, mask in enumerate(masks):
        rr = idx // panel_cols
        cc = idx % panel_cols
        r0 = rr * (panel_h + 1)
        c0 = cc * (panel_w + 1)
        panel = np.zeros_like(target, dtype=int)
        panel[mask] = target[mask]
        crop[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel
    grid[1 : 1 + arr_h, 1 : 1 + arr_w] = crop
    return grid


def _validate_generated(
    grid: np.ndarray,
    target: np.ndarray,
    tile: np.ndarray,
    panel_count: int,
) -> bool:
    try:
        panels = _extract_panels(grid)
        if len(panels) != panel_count:
            return False
        if any(not (MIN_DENSITY <= np.count_nonzero(panel) / panel.size <= MAX_DENSITY) for panel in panels):
            return False
        overlay = _overlay_panels(panels)
        missing_ratio = np.count_nonzero(overlay == BLACK) / overlay.size
        if not (MIN_MISSING <= missing_ratio <= MAX_MISSING):
            return False
        inferred = _infer_minimal_tile(overlay)
        if inferred.shape != tile.shape or not np.array_equal(_tile_to_shape(inferred, target.shape), target):
            return False
        for pr in range(tile.shape[0]):
            for pc in range(tile.shape[1]):
                coords = overlay[pr::tile.shape[0], pc::tile.shape[1]]
                if not np.any(coords != BLACK):
                    return False
        if np.array_equal(overlay, target):
            return False
        if any(_panel_is_independently_sufficient(panel, target) for panel in panels):
            return False
        if not np.array_equal(main(grid), target):
            return False
    except AssertionError:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    layout = random.choice(LAYOUTS)
    panel_count = int(layout[0] * layout[1])
    period = random.choice((2, 3))
    if layout == (2, 2):
        panel_h = random.randint(5, 8)
        panel_w = random.randint(5, 8)
    elif layout == (3, 1):
        panel_h = random.randint(4, 8)
        panel_w = random.randint(5, 8)
    else:
        panel_h = random.randint(5, 8)
        panel_w = random.randint(4, 8)

    tile = _random_tile(period)
    target = _tile_to_shape(tile, (panel_h, panel_w))
    if _infer_minimal_tile(target).shape != (period, period):
        return None

    missing = _sample_missing_mask(target.shape, period)
    if missing is None:
        return None

    masks = _allocate_masks(target.shape, period, panel_count, missing)
    if masks is None:
        return None

    panels = []
    for mask in masks:
        panel = np.zeros_like(target, dtype=int)
        panel[mask] = target[mask]
        panels.append(panel)

    phase_total = period * period
    for panel in panels:
        observed_phases = {
            _phase_index(int(r), int(c), period)
            for r, c in np.argwhere(panel != BLACK)
        }
        if len(observed_phases) == phase_total:
            return None
        if _panel_is_independently_sufficient(panel, target):
            return None

    grid = _build_input_from_masks(target, masks, layout)
    if max(grid.shape) > 30:
        return None
    if not _validate_generated(grid, target, tile, panel_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-overlay puzzle")
