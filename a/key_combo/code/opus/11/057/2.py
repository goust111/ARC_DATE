from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(pts, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _crop_component(grid: np.ndarray, component: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    coords = component["coords"]
    out[coords[:, 0] - r0, coords[:, 1] - c0] = int(component["color"])
    return out


def _hole_count(mask: np.ndarray) -> int:
    padded = np.pad(~mask.astype(bool), 1, constant_values=True)
    h, w = padded.shape
    seen = np.zeros_like(padded, dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not padded[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            touches_border = False
            while q:
                rr, cc = q.popleft()
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and padded[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _analyze(grid: np.ndarray) -> tuple[dict, list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert components, "expected at least one foreground component"

    max_area = max(int(comp["area"]) for comp in components)
    largest = [comp for comp in components if int(comp["area"]) == max_area]
    assert len(largest) == 1, "expected a unique largest foreground component"

    signal = largest[0]
    noise = [comp for comp in components if comp is not signal]
    n_noise = len(noise)
    assert n_noise >= 1
    crop = _crop_component(g, signal, bg)
    return signal, noise, crop


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the unique largest foreground component and tile it by the noise-component count.

    Input structure:
    - The background is the dominant color, typically black.
    - One uniquely largest 4-connected monochrome foreground component acts as the signal object.
    - Several smaller 4-connected monochrome foreground components act as separated noise objects.

    Rule:
    - Find all non-background 4-connected monochrome components.
    - Select the unique largest component, then crop its tight bounding box while keeping only that component and background inside the crop.
    - Count the remaining smaller components as N, and tile the cropped signal patch N times vertically and N times horizontally.

    Key insight:
    - The small components are not disposable distractors: after the largest object is isolated, their count becomes the tiling factor for the final output.
    """
    signal, noise, crop = _analyze(grid)
    _ = signal
    n = len(noise)
    return np.tile(crop, (n, n))


_BG = 0
_COLORS = tuple(range(1, 10))
_NEIGHBORS8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
_MAX_ATTEMPTS = 500
_GEN_CALL_COUNT = 0


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _cells_to_patch(cells: set[tuple[int, int]], color: int) -> np.ndarray:
    norm = _normalize_cells(cells)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in norm:
        patch[r, c] = int(color)
    return patch


def _random_connected_cells(area: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(tuple(frontier)))
    return _normalize_cells(cells)


def _random_signal_patch_no_hole(color: int, min_area: int, max_area: int) -> np.ndarray | None:
    for _ in range(200):
        area = random.randint(min_area, max_area)
        patch = _cells_to_patch(_random_connected_cells(area), color)
        h, w = patch.shape
        if not (3 <= h <= 7 and 3 <= w <= 7):
            continue
        if area >= h * w:
            continue
        if _hole_count(patch == color) != 0:
            continue
        return patch
    return None


def _random_signal_patch_with_hole(color: int, min_area: int, max_area: int) -> np.ndarray | None:
    dims = [(3, 4), (4, 4), (4, 5), (5, 4), (5, 5)]
    random.shuffle(dims)
    for h, w in dims:
        mask = np.ones((h, w), dtype=bool)
        mask[1 : h - 1, 1 : w - 1] = False
        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        rr, cc = random.choice(corners)
        mask[rr, cc] = False
        area = int(mask.sum())
        if not (min_area <= area <= max_area):
            continue
        if _hole_count(mask) != 1:
            continue
        patch = np.zeros((h, w), dtype=int)
        patch[mask] = int(color)
        return patch
    return None


def _random_noise_patch(color: int, area: int) -> np.ndarray:
    if area == 1:
        return np.array([[int(color)]], dtype=int)
    if area == 2:
        if random.random() < 0.5:
            return np.array([[int(color), int(color)]], dtype=int)
        return np.array([[int(color)], [int(color)]], dtype=int)

    shapes = [
        np.array([[color, color, color]], dtype=int),
        np.array([[color], [color], [color]], dtype=int),
        np.array([[color, 0], [color, color]], dtype=int),
        np.array([[0, color], [color, color]], dtype=int),
    ]
    return random.choice(shapes).copy()


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    coords = np.argwhere(mask)
    for r, c in coords:
        for dr, dc in _NEIGHBORS8:
            nr, nc = int(r) + dr, int(c) + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _place_patch(canvas: np.ndarray, occ: np.ndarray, patch: np.ndarray) -> bool:
    h, w = canvas.shape
    ph, pw = patch.shape
    mask = patch != 0
    padded_occ = _dilate8(occ)
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            if np.any(padded_occ[r0 : r0 + ph, c0 : c0 + pw] & mask):
                continue
            candidates.append((r0, c0))
    if not candidates:
        return False
    r0, c0 = random.choice(candidates)
    canvas[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
    occ[r0 : r0 + ph, c0 : c0 + pw] |= mask
    return True


def _sample_noise_colors(signal_color: int, n_noise: int) -> list[int]:
    pool = [color for color in _COLORS if color != signal_color]
    distinct = random.randint(1, min(3, n_noise, len(pool)))
    chosen = random.sample(pool, distinct)
    colors = [random.choice(chosen) for _ in range(n_noise)]
    random.shuffle(colors)
    return colors


def _build_candidate() -> np.ndarray | None:
    global _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1

    n_noise = 2 + ((_GEN_CALL_COUNT - 1) % 3)
    use_hole = (_GEN_CALL_COUNT % 3 == 0)

    noise_areas = [random.randint(1, 3) for _ in range(n_noise)]
    noise_sum = sum(noise_areas)
    min_signal_area = max(8, noise_sum + 1)
    max_signal_area = 15
    if min_signal_area > max_signal_area:
        return None

    signal_color = random.choice(_COLORS)
    if use_hole:
        signal = _random_signal_patch_with_hole(signal_color, min_signal_area, max_signal_area)
    else:
        signal = _random_signal_patch_no_hole(signal_color, min_signal_area, max_signal_area)
        if signal is None:
            signal = _random_signal_patch_with_hole(signal_color, min_signal_area, max_signal_area)
    if signal is None:
        return None

    sh, sw = signal.shape
    if max(sh, sw) * n_noise > 30:
        return None

    canvas_h = random.randint(max(13, sh + 6), 18)
    canvas_w = random.randint(max(13, sw + 6), 18)
    canvas = np.zeros((canvas_h, canvas_w), dtype=int)
    occ = np.zeros((canvas_h, canvas_w), dtype=bool)

    if not _place_patch(canvas, occ, signal):
        return None

    noise_colors = _sample_noise_colors(signal_color, n_noise)
    order = list(range(n_noise))
    random.shuffle(order)
    order.sort(key=lambda idx: noise_areas[idx], reverse=True)
    for idx in order:
        patch = _random_noise_patch(noise_colors[idx], noise_areas[idx])
        if not _place_patch(canvas, occ, patch):
            return None
    return canvas


def _expected_output(grid: np.ndarray) -> np.ndarray:
    signal, noise, crop = _analyze(grid)
    _ = signal
    return np.tile(crop, (len(noise), len(noise)))


def _validate_example(grid: np.ndarray) -> bool:
    try:
        signal, noise, crop = _analyze(grid)
    except AssertionError:
        return False

    areas = [int(signal["area"])] + [int(comp["area"]) for comp in noise]
    n = len(noise)
    if n not in (2, 3, 4):
        return False
    if int(signal["area"]) <= sum(int(comp["area"]) for comp in noise):
        return False
    if any(area < 1 or area > 3 for area in areas[1:]):
        return False
    if crop.size <= int(signal["area"]):
        return False
    if max(crop.shape) > 7:
        return False

    bg = _get_bg(grid)
    crop_components = _find_components(crop, bg)
    if len(crop_components) != 1:
        return False
    if int(crop_components[0]["area"]) != int(signal["area"]):
        return False

    out = main(grid)
    expected = np.tile(crop, (n, n))
    if not np.array_equal(out, expected):
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _build_candidate()
        if grid is None:
            continue
        if _validate_example(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for opus/11/057/2")
