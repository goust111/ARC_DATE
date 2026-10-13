from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

GEN_ATTEMPTS = 400
FRAME_GAP = 2
PAIR_MIN_STEPS = 4


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _blue_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != BLUE:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == BLUE:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _blue_components(grid):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            raise AssertionError("blue component is too small to be a hollow frame")
        expected = set()
        for c in range(c0, c1 + 1):
            expected.add((r0, c))
            expected.add((r1, c))
        for r in range(r0, r1 + 1):
            expected.add((r, c0))
            expected.add((r, c1))
        if set(comp) != expected:
            raise AssertionError("all blue components must be 1-cell-thick hollow rectangles")
        frames.append((r0, c0, r1, c1))
    frames.sort()
    if not frames:
        raise AssertionError("expected at least one blue frame")
    return frames


def _canonical_pair(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _pair_points(points: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(points) % 2:
        raise AssertionError("red endpoints must be even in number")

    by_diff: dict[int, list[tuple[int, int]]] = {}
    by_sum: dict[int, list[tuple[int, int]]] = {}
    for pt in points:
        by_diff.setdefault(pt[0] - pt[1], []).append(pt)
        by_sum.setdefault(pt[0] + pt[1], []).append(pt)

    pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for groups in (by_diff, by_sum):
        for group in groups.values():
            if len(group) == 2:
                pairs.add(_canonical_pair(group[0], group[1]))

    usage = Counter()
    for a, b in pairs:
        usage[a] += 1
        usage[b] += 1

    if len(pairs) != len(points) // 2:
        raise AssertionError("red endpoints do not form a complete set of unique diagonal pairs")
    for pt in points:
        if usage[pt] != 1:
            raise AssertionError("each red endpoint must belong to exactly one valid pair")

    return sorted(pairs)


def _pair_red_pixels(grid: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    points = [tuple(map(int, rc)) for rc in np.argwhere(grid == RED)]
    if not 4 <= len(points) <= 8:
        raise AssertionError("expected 4-8 red endpoint pixels")
    return _pair_points(points)


def _diag_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if dr == 0 or abs(dr) != abs(dc):
        raise AssertionError("paired red pixels must share a 45-degree diagonal")
    step_r = 1 if dr > 0 else -1
    step_c = 1 if dc > 0 else -1
    steps = abs(dr)
    return [(a[0] + k * step_r, a[1] + k * step_c) for k in range(steps + 1)]


def _inside_interior(pt: tuple[int, int], frame: tuple[int, int, int, int]) -> bool:
    r, c = pt
    r0, c0, r1, c1 = frame
    return r0 < r < r1 and c0 < c < c1


def _intersection_counts(
    frames: list[tuple[int, int, int, int]],
    paths: list[list[tuple[int, int]]],
) -> list[int]:
    counts = [0] * len(frames)
    for idx, frame in enumerate(frames):
        for path in paths:
            if any(_inside_interior(pt, frame) for pt in path):
                counts[idx] += 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow blue frame according to how many red diagonal connections cross its interior.

    Input structure:
    - A black background contains several blue one-cell-thick hollow rectangular frames.
    - Four to eight red single pixels act as diagonal endpoints, with each valid pair lying on one shared 45-degree or 135-degree diagonal.
    - Red endpoints start outside every frame, while the implied diagonal connections may pass through frame interiors.

    Rule:
    - Pair red endpoints whenever exactly two red pixels share the same `r-c` or `r+c` diagonal invariant.
    - Draw each pair's red diagonal segment between its endpoints, but do not overwrite blue frame-border pixels.
    - For every blue frame, count how many distinct red segments have at least one pixel strictly inside that frame.
    - Keep count-0 interiors black, fill only black interior cells green for count 1, and yellow for count 2 or more.

    Key insight:
    - The blue rectangles are passive detectors: their fill color depends on external diagonal events, not on the frames' own geometry.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _find_frames(g)
    pairs = _pair_red_pixels(g)
    paths = [_diag_path(a, b) for a, b in pairs]

    for path in paths:
        for r, c in path:
            if int(out[r, c]) != BLUE:
                out[r, c] = RED

    counts = _intersection_counts(frames, paths)
    for (r0, c0, r1, c1), count in zip(frames, counts):
        fill = BLACK if count == 0 else GREEN if count == 1 else YELLOW
        if fill == BLACK:
            continue
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if int(out[r, c]) == BLACK:
                    out[r, c] = fill

    return out


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = FRAME_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _draw_frame(grid: np.ndarray, frame: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame
    grid[r0, c0 : c1 + 1] = BLUE
    grid[r1, c0 : c1 + 1] = BLUE
    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = BLUE


def _place_frames(rng: random.Random, h: int, w: int, n_frames: int) -> list[tuple[int, int, int, int]] | None:
    frames: list[tuple[int, int, int, int]] = []
    for _ in range(250):
        inner_h = rng.randint(3, 5)
        inner_w = rng.randint(3, 5)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        r0 = rng.randint(1, h - outer_h - 1)
        c0 = rng.randint(1, w - outer_w - 1)
        frame = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        if any(_boxes_too_close(frame, other) for other in frames):
            continue
        frames.append(frame)
        if len(frames) == n_frames:
            return sorted(frames)
    return None


def _line_coords(h: int, w: int, slope: int, key: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    if slope == 1:
        for r in range(h):
            c = r - key
            if 0 <= c < w:
                coords.append((r, c))
    else:
        for r in range(h):
            c = key - r
            if 0 <= c < w:
                coords.append((r, c))
    return coords


def _frame_maps(
    h: int,
    w: int,
    frames: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    blocked = np.zeros((h, w), dtype=bool)
    interior_bits = np.zeros((h, w), dtype=np.int64)
    for idx, (r0, c0, r1, c1) in enumerate(frames):
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True
        interior_bits[r0 + 1 : r1, c0 + 1 : c1] |= 1 << idx
    return blocked, interior_bits


def _enumerate_candidates(
    h: int,
    w: int,
    frames: list[tuple[int, int, int, int]],
) -> list[dict]:
    blocked, interior_bits = _frame_maps(h, w, frames)
    candidates: list[dict] = []
    for slope in (1, -1):
        key_values = range(-(w - 1), h) if slope == 1 else range(h + w - 1)
        for key in key_values:
            coords = _line_coords(h, w, slope, key)
            if len(coords) < PAIR_MIN_STEPS + 1:
                continue
            open_idx = [i for i, pt in enumerate(coords) if not blocked[pt]]
            for ia, ib in itertools.combinations(open_idx, 2):
                if ib - ia < PAIR_MIN_STEPS:
                    continue
                path = coords[ia : ib + 1]
                mask = 0
                for r, c in path:
                    mask |= int(interior_bits[r, c])
                candidates.append(
                    {
                        "pair": _canonical_pair(coords[ia], coords[ib]),
                        "path": path,
                        "mask": mask,
                        "frames": tuple(idx for idx in range(len(frames)) if (mask >> idx) & 1),
                        "slope": slope,
                        "length": len(path),
                    }
                )
    return candidates


def _selected_pairs_valid(cands: list[dict]) -> bool:
    points: list[tuple[int, int]] = []
    expected = set()
    for cand in cands:
        a, b = cand["pair"]
        points.extend([a, b])
        expected.add(cand["pair"])
    if len(set(points)) != len(points):
        return False
    try:
        actual = set(_pair_points(points))
    except AssertionError:
        return False
    return actual == expected


def _counts_from_candidates(n_frames: int, cands: list[dict]) -> list[int]:
    counts = [0] * n_frames
    for cand in cands:
        for idx in cand["frames"]:
            counts[idx] += 1
    return counts


def _coverage_classes(counts: list[int]) -> set[int]:
    return {0 if count == 0 else 1 if count == 1 else 2 for count in counts}


def _choose_segments(
    rng: random.Random,
    h: int,
    w: int,
    frames: list[tuple[int, int, int, int]],
) -> list[dict] | None:
    all_candidates = _enumerate_candidates(h, w, frames)
    if not all_candidates:
        return None

    by_mask: dict[int, list[dict]] = {}
    rng.shuffle(all_candidates)
    for cand in all_candidates:
        by_mask.setdefault(cand["mask"], []).append(cand)

    pool: list[dict] = []
    mask_keys = list(by_mask)
    rng.shuffle(mask_keys)
    for mask in mask_keys:
        group = by_mask[mask]
        group.sort(key=lambda cand: (cand["length"], len(cand["frames"])), reverse=True)
        take = 6 if mask else 4
        pool.extend(group[:take])

    nonzero_pool = [cand for cand in pool if cand["mask"] != 0]
    rng.shuffle(nonzero_pool)
    nonzero_pool = nonzero_pool[:36]

    for target in rng.sample([3, 4], 2):
        if len(nonzero_pool) < target:
            continue
        for combo in itertools.combinations(nonzero_pool, target):
            chosen = list(combo)
            if not _selected_pairs_valid(chosen):
                continue
            counts = _counts_from_candidates(len(frames), chosen)
            if _coverage_classes(counts) != {0, 1, 2}:
                continue
            selected = chosen
            if len(selected) < 4 and rng.random() < 0.45:
                extras = pool[:]
                rng.shuffle(extras)
                for extra in extras:
                    if extra in selected:
                        continue
                    candidate = selected + [extra]
                    if len(candidate) > 4 or not _selected_pairs_valid(candidate):
                        continue
                    counts = _counts_from_candidates(len(frames), candidate)
                    if _coverage_classes(counts) == {0, 1, 2}:
                        selected = candidate
                        break
            return selected
    return None


def _build_input_grid(
    h: int,
    w: int,
    frames: list[tuple[int, int, int, int]],
    segments: list[dict],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for frame in frames:
        _draw_frame(grid, frame)
    for seg in segments:
        a, b = seg["pair"]
        grid[a] = RED
        grid[b] = RED
    return grid


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        frames = _find_frames(grid)
        pairs = _pair_red_pixels(grid)
        paths = [_diag_path(a, b) for a, b in pairs]
        counts = _intersection_counts(frames, paths)
        if _coverage_classes(counts) != {0, 1, 2}:
            return False
        out = main(grid)
        if GREEN not in out or YELLOW not in out:
            return False
        zero_ok = False
        for frame, count in zip(frames, counts):
            r0, c0, r1, c1 = frame
            interior_in = grid[r0 + 1 : r1, c0 + 1 : c1]
            interior_out = out[r0 + 1 : r1, c0 + 1 : c1]
            if count == 0:
                zero_ok = zero_ok or np.any(interior_out == BLACK)
            elif count == 1:
                if not np.any(interior_out == GREEN):
                    return False
            else:
                if not np.any(interior_out == YELLOW):
                    return False
            if np.any(interior_in == RED):
                return False
        return zero_ok
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        h = rng.randint(18, 24)
        w = rng.randint(18, 24)
        n_frames = rng.randint(3, 5)
        frames = _place_frames(rng, h, w, n_frames)
        if frames is None:
            continue
        segments = _choose_segments(rng, h, w, frames)
        if segments is None:
            continue
        grid = _build_input_grid(h, w, frames, segments)
        if _semantic_check(grid):
            return grid
    raise RuntimeError("failed to generate a valid diagonal-frame counting input")
