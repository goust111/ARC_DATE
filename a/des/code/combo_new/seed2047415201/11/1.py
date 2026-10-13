from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
GREY = 5
COLOR_POOL = [1, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            seen[sr, sc] = True
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            pts: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                pts.append((int(r), int(c)))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [p[0] for p in pts]
    cols = [p[1] for p in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _find_rect_frames(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for pts in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(pts)
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue
        expected: set[tuple[int, int]] = set()
        for r in range(r0, r1 + 1):
            expected.add((r, c0))
            expected.add((r, c1))
        for c in range(c0, c1 + 1):
            expected.add((r0, c))
            expected.add((r1, c))
        if set(pts) == expected:
            frames.append((int(r0), int(c0), int(r1), int(c1)))
    assert frames, "expected at least one grey rectangular frame"
    frames.sort()
    return frames


def _frame_region_mask(shape: tuple[int, int], frames: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in frames:
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _pattern_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in mask.reshape(-1).tolist())


def _read_legend(
    g: np.ndarray, frame_region: np.ndarray
) -> tuple[dict[tuple[int, ...], int], np.ndarray]:
    h, w = g.shape
    candidates: list[tuple[int, int, tuple[int, ...], int]] = []

    for r0 in range(h - 3):
        for c0 in range(w - 2):
            if c0 < w // 2:
                continue
            if frame_region[r0 : r0 + 4, c0 : c0 + 3].any():
                continue
            pattern = g[r0 : r0 + 3, c0 : c0 + 3]
            if not np.isin(pattern, [BG, RED]).all() or not np.any(pattern == RED):
                continue
            bottom = g[r0 + 3, c0 : c0 + 3]
            filled = [i for i, val in enumerate(bottom.tolist()) if int(val) != BG]
            if len(filled) != 1:
                continue
            color = int(bottom[filled[0]])
            if color in (BG, RED, GREY):
                continue
            candidates.append((int(r0), int(c0), _pattern_key(pattern == RED), color))

    assert candidates, "expected a right-side fingerprint legend"

    by_col: dict[int, list[tuple[int, int, tuple[int, ...], int]]] = {}
    for cand in candidates:
        by_col.setdefault(cand[1], []).append(cand)
    legend_c0 = max(by_col, key=lambda c: (len(by_col[c]), c))
    entries = sorted(by_col[legend_c0])

    mapping: dict[tuple[int, ...], int] = {}
    legend_mask = np.zeros(g.shape, dtype=bool)
    for r0, c0, pattern, color in entries:
        if pattern in mapping:
            assert mapping[pattern] == color, "duplicate fingerprint maps to different colors"
            continue
        mapping[pattern] = color
        legend_mask[r0 : r0 + 4, c0 : c0 + 3] = True

    assert mapping, "legend must contain at least one pattern"
    top = min(r for r, _c, _p, _color in entries)
    bottom = max(r + 4 for r, _c, _p, _color in entries)
    legend_mask[top:bottom, legend_c0 : legend_c0 + 3] = True
    legend_mask &= ~frame_region
    return mapping, legend_mask


def _fingerprint_for_frame(g: np.ndarray, frame: tuple[int, int, int, int]) -> tuple[int, ...]:
    r0, c0, r1, c1 = frame
    ir0, ic0 = r0 + 1, c0 + 1
    ih, iw = r1 - r0 - 1, c1 - c0 - 1
    assert ih >= 3 and iw >= 3, "frame interior must support a 3x3 normalization grid"
    mask = np.zeros((3, 3), dtype=bool)
    interior = g[ir0:r1, ic0:c1]
    for lr, lc in zip(*np.where(interior == RED), strict=False):
        zr = min(2, int(lr) * 3 // ih)
        zc = min(2, int(lc) * 3 // iw)
        mask[zr, zc] = True
    assert np.any(mask), "each frame must contain red markers"
    return _pattern_key(mask)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill framed interiors by matching normalized red-marker fingerprints to a right-side legend.

    Input structure:
    - The background is black, with several non-overlapping one-pixel-thick grey rectangular frames.
    - Each frame interior contains red marker pixels whose exact coordinates may vary.
    - A right-side legend column contains entries made from a 3x3 red pattern on black, with one colored square directly below the pattern.

    Rule:
    - For each grey frame, divide its interior evenly into a 3x3 grid and mark every coarse cell that contains at least one red pixel.
    - Match that normalized 3x3 fingerprint to the legend entry with the same red pattern.
    - Fill all black, non-marker pixels inside the frame with the matched legend square color, while preserving the grey border and red markers.
    - Erase the right-side legend area back to black.

    Key insight: The precise red marker coordinates are noise; only their coarse 3x3 distribution selects the fill color through the legend.
    """
    g = np.asarray(grid, dtype=int)
    frames = _find_rect_frames(g)
    frame_region = _frame_region_mask(g.shape, frames)
    mapping, legend_mask = _read_legend(g, frame_region)

    out = g.copy()
    for frame in frames:
        key = _fingerprint_for_frame(g, frame)
        assert key in mapping, "frame fingerprint must be present in the legend"
        fill_color = mapping[key]
        r0, c0, r1, c1 = frame
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BG] = fill_color

    out[legend_mask] = BG
    return out


def _valid_fingerprints() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1, 1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        count = int(mask.sum())
        if not (3 <= count <= 6):
            continue
        if not mask.any(axis=0).all():
            continue
        if int(mask.any(axis=1).sum()) < 2:
            continue
        masks.append(mask)
    return masks


FINGERPRINTS = _valid_fingerprints()


def _zone_indices(length: int, zone: int) -> list[int]:
    return [i for i in range(length) if min(2, i * 3 // length) == zone]


def _draw_frame_with_markers(
    grid: np.ndarray,
    r0: int,
    c0: int,
    ih: int,
    iw: int,
    pattern: np.ndarray,
    rng: random.Random,
) -> None:
    r1 = r0 + ih + 1
    c1 = c0 + iw + 1
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY

    used: set[tuple[int, int]] = set()
    active = [(int(r), int(c)) for r, c in zip(*np.where(pattern), strict=False)]
    for zr, zc in active:
        rr = rng.choice(_zone_indices(ih, zr))
        cc = rng.choice(_zone_indices(iw, zc))
        used.add((r0 + 1 + rr, c0 + 1 + cc))

    if active and rng.random() < 0.35:
        for _ in range(rng.randint(1, 2)):
            zr, zc = rng.choice(active)
            choices = [
                (r0 + 1 + rr, c0 + 1 + cc)
                for rr in _zone_indices(ih, zr)
                for cc in _zone_indices(iw, zc)
                if (r0 + 1 + rr, c0 + 1 + cc) not in used
            ]
            if choices:
                used.add(rng.choice(choices))

    for r, c in used:
        grid[r, c] = RED


def _place_legend(
    grid: np.ndarray,
    patterns: list[np.ndarray],
    colors: list[int],
    rng: random.Random,
) -> None:
    h, w = grid.shape
    c0 = w - 3
    available_gap = h - 4 * len(patterns)
    base_between = len(patterns) - 1
    assert available_gap >= base_between
    gaps = [0 for _ in range(len(patterns) + 1)]
    for i in range(1, len(patterns)):
        gaps[i] = 1
    for _ in range(available_gap - base_between):
        gaps[rng.randrange(len(gaps))] += 1

    r = gaps[0]
    for idx, (pattern, color) in enumerate(zip(patterns, colors, strict=True)):
        patch = grid[r : r + 3, c0 : c0 + 3]
        patch[pattern] = RED
        grid[r + 3, c0 + rng.randrange(3)] = color
        r += 4 + gaps[idx + 1]


def _free_frame_positions(occupied: np.ndarray, fh: int, fw: int, max_c1: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - fh + 1):
        for c0 in range(min(w - fw, max_c1 - fw + 1) + 1):
            r_slice = slice(max(0, r0 - 1), min(h, r0 + fh + 1))
            c_slice = slice(max(0, c0 - 1), min(w, c0 + fw + 1))
            if not occupied[r_slice, c_slice].any():
                positions.append((int(r0), int(c0)))
    return positions


def _make_case(rng: random.Random) -> np.ndarray | None:
    entry_count = rng.randint(2, 5)
    min_h = 4 * entry_count + (entry_count - 1)
    height = rng.randint(max(16, min_h), 30)
    width = rng.randint(18, 30)

    patterns = rng.sample(FINGERPRINTS, entry_count)
    colors = rng.sample(COLOR_POOL, entry_count)
    assignments = list(range(entry_count))
    extra_frames = rng.randint(0, min(2, entry_count))
    assignments.extend(rng.randrange(entry_count) for _ in range(extra_frames))
    rng.shuffle(assignments)

    grid = np.full((height, width), BG, dtype=int)
    _place_legend(grid, patterns, colors, rng)

    occupied = np.zeros((height, width), dtype=bool)
    occupied[:, width - 4 : width] = True

    max_frame_right = width - 5
    for pat_idx in assignments:
        placed = False
        for _ in range(40):
            ih = rng.randint(3, 8)
            iw = rng.randint(3, 8)
            fh, fw = ih + 2, iw + 2
            positions = _free_frame_positions(occupied, fh, fw, max_frame_right)
            if not positions:
                continue
            r0, c0 = rng.choice(positions)
            _draw_frame_with_markers(grid, r0, c0, ih, iw, patterns[pat_idx], rng)
            occupied[max(0, r0 - 1) : min(height, r0 + fh + 1),
                     max(0, c0 - 1) : min(width, c0 + fw + 1)] = True
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _make_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fingerprint-legend input")
