from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbor_count(cell: tuple[int, int], pset: set[tuple[int, int]]) -> int:
    r, c = cell
    return sum((r + dr, c + dc) in pset for dr, dc in DIR4)


def _trace_path(grid: np.ndarray) -> list[tuple[int, int]]:
    bg = _get_bg(grid)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid != bg)]
    assert coords, "expected a non-empty path"

    pset = set(coords)
    degrees = {cell: _neighbor_count(cell, pset) for cell in coords}
    assert all(deg in (1, 2) for deg in degrees.values()), "path must be a simple chain"

    endpoints = sorted(cell for cell, deg in degrees.items() if deg == 1)
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    start = endpoints[0]

    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        order.append(cur)
        next_cells = []
        for dr, dc in DIR4:
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in pset and nxt != prev:
                next_cells.append(nxt)
        if not next_cells:
            break
        assert len(next_cells) == 1, "simple path traversal must be unique"
        prev, cur = cur, next_cells[0]

    assert len(order) == len(coords), "path must be connected"
    return order


def _primitive_period(seq: list[int]) -> int:
    n = len(seq)
    for p in range(1, n):
        if n % p != 0:
            continue
        if all(seq[i] == seq[i % p] for i in range(n)):
            return p
    return n


def _residue_majority(seq: list[int], period: int) -> tuple[list[int], int, bool]:
    clean_band: list[int] = []
    mismatches = 0
    strict_majority = True
    for offset in range(period):
        residue = seq[offset::period]
        counts = Counter(residue)
        color, freq = counts.most_common(1)[0]
        clean_band.append(int(color))
        mismatches += len(residue) - int(freq)
        if int(freq) * 2 <= len(residue):
            strict_majority = False
    return clean_band, int(mismatches), strict_majority


def _infer_period(seq: list[int]) -> tuple[int, list[int], int]:
    n = len(seq)
    candidates: list[tuple[int, int, list[int]]] = []
    for period in range(2, n // 3 + 1):
        clean_band, mismatches, strict_majority = _residue_majority(seq, period)
        if strict_majority:
            candidates.append((mismatches, period, clean_band))
    assert candidates, "expected at least one valid periodic explanation"
    mismatches, period, clean_band = min(candidates, key=lambda item: (item[0], item[1]))
    return int(period), clean_band, int(mismatches)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair corrupted colors on a snake path by recovering its shortest periodic color band.

    Input structure:
    - The most common color is the background.
    - All non-background cells form one 4-connected simple path with exactly two endpoints.
    - Reading colors along that path yields a short periodic band repeated at least three times, with a few corrupted cells.

    Rule:
    - Find the two endpoints and start from the top-left endpoint.
    - Trace the whole snake cell by cell to obtain a 1D color sequence.
    - Infer the shortest period length that best explains the sequence by residue-wise majority vote, then repaint every mismatching path cell with its periodic majority color.

    Key insight:
    - The visible object is a 2D embedding of a 1D periodic string, so denoising must happen in traversal order rather than by local 2D neighborhoods.
    """
    g = np.asarray(grid, dtype=int).copy()
    order = _trace_path(g)
    seq = [int(g[r, c]) for r, c in order]
    period, clean_band, _mismatches = _infer_period(seq)

    out = g.copy()
    for idx, (r, c) in enumerate(order):
        out[r, c] = int(clean_band[idx % period])
    return out


def _valid_next(
    nxt: tuple[int, int],
    cur: tuple[int, int],
    pset: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    r, c = nxt
    h, w = shape
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if nxt in pset:
        return False
    for dr, dc in DIR4:
        adj = (r + dr, c + dc)
        if adj in pset and adj != cur:
            return False
    return True


def _path_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(2, len(path)):
        dr1 = path[i - 1][0] - path[i - 2][0]
        dc1 = path[i - 1][1] - path[i - 2][1]
        dr2 = path[i][0] - path[i - 1][0]
        dc2 = path[i][1] - path[i - 1][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return turns


def _build_path(h: int, w: int, length: int) -> list[tuple[int, int]] | None:
    for _ in range(400):
        start = (random.randint(1, h - 2), random.randint(1, w - 2))
        path = [start]
        pset = {start}
        prev_dir: tuple[int, int] | None = None

        while len(path) < length:
            cur = path[-1]
            valid_dirs = []
            for dr, dc in DIR4:
                nxt = (cur[0] + dr, cur[1] + dc)
                if _valid_next(nxt, cur, pset, (h, w)):
                    weight = 1
                    if prev_dir == (dr, dc):
                        weight = 3
                    valid_dirs.extend([(dr, dc)] * weight)
            if not valid_dirs:
                break

            dr, dc = random.choice(valid_dirs)
            nxt = (cur[0] + dr, cur[1] + dc)
            path.append(nxt)
            pset.add(nxt)
            prev_dir = (dr, dc)

        if len(path) != length:
            continue

        rows = [r for r, _ in path]
        cols = [c for _, c in path]
        if _path_turns(path) < max(3, length // 7):
            continue
        if max(rows) - min(rows) < 3 or max(cols) - min(cols) < 3:
            continue
        order = _trace_path(_path_to_grid(path, 0, 1, h, w))
        assert set(order) == set(path)
        return path

    return None


def _path_to_grid(
    order: list[tuple[int, int]],
    bg: int,
    path_color: int,
    h: int,
    w: int,
) -> np.ndarray:
    grid = np.full((h, w), int(bg), dtype=int)
    for r, c in order:
        grid[r, c] = int(path_color)
    return grid


def _generate_band(period: int, palette: list[int]) -> list[int]:
    for _ in range(300):
        band = [random.choice(palette) for _ in range(period)]
        if len(set(band)) < 2:
            continue
        if _primitive_period(band) != period:
            continue
        return band
    raise RuntimeError("failed to generate a primitive color band")


def _all_candidate_scores(seq: list[int]) -> list[tuple[int, int]]:
    scores = []
    for period in range(2, len(seq) // 3 + 1):
        _band, mismatches, strict_majority = _residue_majority(seq, period)
        if strict_majority:
            scores.append((period, mismatches))
    return scores


def _pick_noise_positions(length: int, period: int) -> list[int]:
    residue_sizes = [len(range(offset, length, period)) for offset in range(period)]
    capacities = [(size - 1) // 2 for size in residue_sizes]
    available_residues = [idx for idx, cap in enumerate(capacities) if cap > 0]
    assert available_residues

    min_noise = 1
    max_noise = min(sum(capacities), max(2, length // 6))
    target_noise = random.randint(min_noise, max_noise)

    chosen: list[int] = []
    used_by_residue = [0] * period
    candidate_positions = list(range(length))
    random.shuffle(candidate_positions)
    for pos in candidate_positions:
        residue = pos % period
        if used_by_residue[residue] >= capacities[residue]:
            continue
        chosen.append(pos)
        used_by_residue[residue] += 1
        if len(chosen) == target_noise:
            break

    assert chosen
    return sorted(chosen)


def _make_instance() -> tuple[np.ndarray, np.ndarray, int]:
    h = random.randint(12, 20)
    w = random.randint(12, 20)
    length = random.randint(18, min(28, h * w // 3))
    max_period = min(7, length // 3)
    period = random.randint(2, max_period)

    path = _build_path(h, w, length)
    if path is None:
        raise RuntimeError("failed to build a valid snake path")
    ordered_path = path if min(path[0], path[-1]) == path[0] else list(reversed(path))

    bg = random.randint(0, 9)
    non_bg_colors = [c for c in range(10) if c != bg]
    random.shuffle(non_bg_colors)
    palette_size = random.randint(2, min(4, period, len(non_bg_colors)))
    band_palette = non_bg_colors[:palette_size]
    band = _generate_band(period, band_palette)

    clean_seq = [band[i % period] for i in range(length)]
    clean = np.full((h, w), bg, dtype=int)
    for idx, (r, c) in enumerate(ordered_path):
        clean[r, c] = int(clean_seq[idx])

    noisy = clean.copy()
    extra_noise_pool = [c for c in non_bg_colors if c not in band_palette][:3]
    wrong_color_pool = band_palette + extra_noise_pool
    noise_positions = _pick_noise_positions(length, period)
    for idx in noise_positions:
        true_color = clean_seq[idx]
        choices = [c for c in wrong_color_pool if c != true_color]
        noisy[ordered_path[idx]] = random.choice(choices)

    return noisy, clean, period


def _validate_instance(noisy: np.ndarray, clean: np.ndarray, planted_period: int) -> bool:
    if noisy.shape != clean.shape or noisy.ndim != 2 or max(noisy.shape) > 30:
        return False
    if np.array_equal(noisy, clean):
        return False

    path = _trace_path(noisy)
    noisy_seq = [int(noisy[r, c]) for r, c in path]
    clean_seq = [int(clean[r, c]) for r, c in path]
    if _primitive_period([clean_seq[i] for i in range(planted_period)]) != planted_period:
        return False

    inferred_period, _band, inferred_mismatches = _infer_period(noisy_seq)
    if inferred_period != planted_period:
        return False

    planted_band, planted_mismatches, strict_majority = _residue_majority(noisy_seq, planted_period)
    if not strict_majority or planted_mismatches != inferred_mismatches:
        return False
    if any(clean_seq[i] != planted_band[i % planted_period] for i in range(len(clean_seq))):
        return False

    for period, mismatches in _all_candidate_scores(noisy_seq):
        if period < planted_period and mismatches <= inferred_mismatches:
            return False

    if not np.array_equal(main(noisy), clean):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        noisy, clean, period = _make_instance()
        if _validate_instance(noisy, clean, period):
            return noisy
    raise RuntimeError("failed to generate a valid noisy periodic snake input")
