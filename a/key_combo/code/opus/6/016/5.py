from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
RAY = 5
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
TILE_SHAPES = [(2, 2), (2, 3), (3, 2), (3, 3)]
RUN_MIN_LEN = 4
GEN_ATTEMPTS = 500


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _candidate_runs(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[list[tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for r in range(h):
        for start in range(w):
            if int(grid[r, start]) in (BG, RAY):
                continue
            used: set[int] = set()
            for end in range(start, w):
                color = int(grid[r, end])
                if color in (BG, RAY) or color in used:
                    break
                used.add(color)
                if end - start + 1 >= RUN_MIN_LEN:
                    run = tuple((r, cc) for cc in range(start, end + 1))
                    if run not in seen:
                        seen.add(run)
                        candidates.append(list(run))
    return candidates


def _cast_rays(
    shape: tuple[int, int],
    run: list[tuple[int, int]],
    colors: list[int],
    *,
    reverse: bool = False,
) -> np.ndarray:
    h, w = shape
    canvas = np.zeros((h, w), dtype=int)
    ordered = list(zip(run, colors))
    if reverse:
        ordered = ordered[::-1]
    for (r, c), color in ordered:
        canvas[r, c] = int(color)
    for (r, c), _color in ordered:
        for dr, dc in DIAG_DIRS:
            nr, nc = r + dr, c + dc
            while 0 <= nr < h and 0 <= nc < w and canvas[nr, nc] == BG:
                canvas[nr, nc] = RAY
                nr += dr
                nc += dc
    return canvas


def _region_components(shape: tuple[int, int], run: list[tuple[int, int]], rays: np.ndarray) -> list[np.ndarray]:
    mask = rays == BG
    for r, c in run:
        mask[r, c] = False
    return _components(mask)


def _infer_periodic_tile(
    grid: np.ndarray,
    coords: np.ndarray,
    *,
    max_period: int = 3,
) -> tuple[np.ndarray, tuple[int, int, int, int], int] | None:
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    bbox_h = int(r1 - r0 + 1)
    bbox_w = int(c1 - c0 + 1)
    values = np.array([int(grid[r, c]) for r, c in coords], dtype=int)
    rel = coords - np.array([[r0, c0]], dtype=int)

    best: tuple[tuple[int, int, int, int], np.ndarray, int] | None = None
    for ph in range(1, min(max_period, bbox_h) + 1):
        if bbox_h < 2 * ph:
            continue
        for pw in range(1, min(max_period, bbox_w) + 1):
            if bbox_w < 2 * pw:
                continue
            tile = np.zeros((ph, pw), dtype=int)
            ok = True
            for rr in range(ph):
                for cc in range(pw):
                    mask = (rel[:, 0] % ph == rr) & (rel[:, 1] % pw == cc)
                    if not np.any(mask):
                        ok = False
                        break
                    tile[rr, cc] = Counter(map(int, values[mask])).most_common(1)[0][0]
                if not ok:
                    break
            if not ok:
                continue
            pred = tile[rel[:, 0] % ph, rel[:, 1] % pw]
            mismatches = int(np.count_nonzero(pred != values))
            key = (mismatches, ph * pw, ph, pw)
            if best is None or key < best[0]:
                best = (key, tile, mismatches)

    if best is None:
        return None
    key, tile, mismatches = best
    return tile, key, mismatches


def _paint_region(out: np.ndarray, coords: np.ndarray, tile: np.ndarray) -> None:
    r0, c0, _r1, _c1 = _bbox(coords)
    ph, pw = tile.shape
    for r, c in coords:
        out[r, c] = int(tile[(r - r0) % ph, (c - c0) % pw])


def _evaluate_run(grid: np.ndarray, run: list[tuple[int, int]]) -> dict | None:
    colors = [int(grid[r, c]) for r, c in run]
    if len(set(colors)) != len(colors):
        return None
    rays = _cast_rays(grid.shape, run, colors)
    regions = _region_components(grid.shape, run, rays)
    if len(regions) < 2:
        return None

    total_mismatches = 0
    tiles: list[np.ndarray] = []
    for coords in regions:
        inferred = _infer_periodic_tile(grid, coords)
        if inferred is None:
            return None
        tile, key, mismatches = inferred
        r0, c0, r1, c1 = _bbox(coords)
        bbox_h = int(r1 - r0 + 1)
        bbox_w = int(c1 - c0 + 1)
        if bbox_h < 2 * tile.shape[0] or bbox_w < 2 * tile.shape[1]:
            return None
        tiles.append(tile)
        total_mismatches += mismatches

    return {
        "run": run,
        "colors": colors,
        "rays": rays,
        "regions": regions,
        "tiles": tiles,
        "score": (int(total_mismatches), -int(len(regions)), -int(len(run))),
    }


def _solve_details(grid: np.ndarray) -> dict:
    candidates = _candidate_runs(grid)
    assert candidates, "expected at least one multicolor horizontal run"
    scored = [item for item in (_evaluate_run(grid, run) for run in candidates) if item is not None]
    assert scored, "no candidate run yields valid periodic partitions"
    scored.sort(key=lambda item: item["score"])
    best = scored[0]
    best_score = best["score"]
    tied = [item for item in scored if item["score"] == best_score]
    assert len(tied) == 1, "main run is ambiguous"
    return {"best": best, "all": scored}


def _render_solution(grid: np.ndarray, details: dict) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    best = details["best"]
    out = g.copy()
    for coords, tile in zip(best["regions"], best["tiles"]):
        _paint_region(out, coords, tile)
    out[best["rays"] == RAY] = RAY
    for (r, c), color in zip(best["run"], best["colors"]):
        out[r, c] = int(color)
    return out


def _minimal_period(tile: np.ndarray) -> tuple[int, int]:
    h, w = tile.shape
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            ok = True
            for r in range(h):
                for c in range(w):
                    if int(tile[r, c]) != int(tile[r % ph, c % pw]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return int(ph), int(pw)
    raise AssertionError("unreachable")


def _random_irreducible_tile(shape: tuple[int, int], colors: tuple[int, int]) -> np.ndarray | None:
    ph, pw = map(int, shape)
    a, b = map(int, colors)
    for _ in range(200):
        tile = np.random.choice(np.array([a, b], dtype=int), size=(ph, pw))
        if len(set(map(int, tile.flat))) < 2:
            continue
        if _minimal_period(tile) != (ph, pw):
            continue
        return tile.astype(int)
    return None


def _sample_region_variant(
    shape: tuple[int, int],
    coords: np.ndarray,
    used_signatures: set[tuple[tuple[int, int], tuple[int, ...]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    h, w = shape
    r0, c0, r1, c1 = _bbox(coords)
    bbox_h = int(r1 - r0 + 1)
    bbox_w = int(c1 - c0 + 1)
    valid_shapes = [dims for dims in TILE_SHAPES if bbox_h >= 2 * dims[0] and bbox_w >= 2 * dims[1]]
    if not valid_shapes:
        return None

    area = int(coords.shape[0])
    for _ in range(300):
        shape_choice = random.choice(valid_shapes)
        colors = tuple(random.sample(PALETTE, 2))
        tile = _random_irreducible_tile(shape_choice, colors)
        if tile is None:
            continue
        signature = (tile.shape, tuple(map(int, tile.flat)))
        if signature in used_signatures:
            continue

        clean = np.zeros((h, w), dtype=int)
        noisy = np.zeros((h, w), dtype=int)
        _paint_region(clean, coords, tile)
        _paint_region(noisy, coords, tile)

        inferred_clean = _infer_periodic_tile(clean, coords)
        if inferred_clean is None:
            continue
        clean_tile, _clean_key, clean_mismatches = inferred_clean
        if clean_mismatches != 0 or not np.array_equal(clean_tile, tile):
            continue

        max_noise = min(3, max(1, area // 8), max(1, int(area * 0.2)))
        noise_count = random.randint(1, max_noise)
        noise_cells = random.sample(range(area), noise_count)
        for idx in noise_cells:
            r, c = map(int, coords[idx])
            current = int(noisy[r, c])
            choices = [color for color in PALETTE if color != current]
            noisy[r, c] = int(random.choice(choices))

        inferred_noisy = _infer_periodic_tile(noisy, coords)
        if inferred_noisy is None:
            continue
        noisy_tile, _noisy_key, noisy_mismatches = inferred_noisy
        if noisy_mismatches < 1 or not np.array_equal(noisy_tile, tile):
            continue

        used_signatures.add(signature)
        return tile, clean, noisy
    return None


def _global_partition_mismatch(grid: np.ndarray, run: list[tuple[int, int]]) -> int:
    pattern_mask = grid != BG
    for r, c in run:
        pattern_mask[r, c] = False
    coords = np.argwhere(pattern_mask)
    inferred = _infer_periodic_tile(grid, coords)
    assert inferred is not None
    _tile, _key, mismatches = inferred
    return int(mismatches)


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(20, 28)
    w = random.randint(13, 16)
    run_len = random.randint(5, min(7, w - 6))
    max_left = w - run_len - 3
    if max_left < 3:
        return None
    left_gap = random.randint(3, max_left)
    right_gap = w - run_len - left_gap
    if right_gap < 3:
        return None

    upper_lo = max(6, h // 4)
    upper_hi = max(upper_lo, h // 2 - 3)
    lower_lo = min(h - 7, h // 2 + 2)
    lower_hi = h - 7
    if lower_lo > lower_hi:
        return None
    row = random.randint(upper_lo, upper_hi) if random.random() < 0.5 else random.randint(lower_lo, lower_hi)

    run_colors = random.sample(PALETTE, run_len)
    run = [(row, c) for c in range(left_gap, left_gap + run_len)]
    rays = _cast_rays((h, w), run, run_colors)
    reverse_rays = _cast_rays((h, w), run, run_colors, reverse=True)
    if np.array_equal(rays == RAY, reverse_rays == RAY):
        return None

    regions = _region_components((h, w), run, rays)
    if len(regions) < 4:
        return None
    for coords in regions:
        r0, c0, r1, c1 = _bbox(coords)
        if (r1 - r0 + 1) < 4 or (c1 - c0 + 1) < 4:
            return None

    clean = np.zeros((h, w), dtype=int)
    noisy = np.zeros((h, w), dtype=int)
    for (r, c), color in zip(run, run_colors):
        clean[r, c] = int(color)
        noisy[r, c] = int(color)

    used_signatures: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    for coords in regions:
        sampled = _sample_region_variant((h, w), coords, used_signatures)
        if sampled is None:
            return None
        _tile, clean_region, noisy_region = sampled
        clean[clean_region != BG] = clean_region[clean_region != BG]
        noisy[noisy_region != BG] = noisy_region[noisy_region != BG]

    expected = clean.copy()
    expected[rays == RAY] = RAY

    try:
        details = _solve_details(noisy)
    except AssertionError:
        return None
    best = details["best"]
    if tuple(best["run"]) != tuple(run):
        return None
    if not np.array_equal(_render_solution(noisy, details), expected):
        return None

    global_mismatch = _global_partition_mismatch(clean, run)
    if global_mismatch < max(8, int(np.count_nonzero(clean) // 10)):
        return None

    return noisy, expected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair noisy periodic tiles independently inside the diagonal ray partitions cast from an ordered multicolor row.

    Input structure:
    - Black background with one horizontal multicolor run near the middle; the run uses distinct non-black, non-grey colors and spans most of the row.
    - The grey diagonal rays are absent from the input, but if cast from that run they divide the surrounding space into several 4-connected regions.
    - Each region contains its own small repeating tile pattern with 1-3 corrupted noise pixels.

    Rule:
    - Consider every plausible horizontal multicolor run candidate, cast four 45-degree grey rays from its cells from left to right on a blank canvas, and keep the candidate whose induced regions admit the best low-noise periodic explanation.
    - For the selected run, use the resulting grey barriers and the run itself to define the regions.
    - In each region, infer the smallest repeating tile that still repeats at least twice along both bbox axes by majority vote over residue classes, then repaint the region with that clean tile.
    - Overlay the unchanged run and the grey rays onto the repaired regions.

    Key insight:
    - The left-to-right ray order does not just add decoration: it determines the partition geometry, and those partitions in turn determine which pixels belong to each local periodic model.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    details = _solve_details(g)
    return _render_solution(g, details)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_example()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid ray-partition tile-repair input")
