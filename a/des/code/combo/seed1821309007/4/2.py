from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 8
PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]
AXES = {
    "h": np.array([0, 1], dtype=int),
    "v": np.array([1, 0], dtype=int),
    "d1": np.array([1, 1], dtype=int),
    "d2": np.array([1, -1], dtype=int),
}
MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = cr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _choose_bg(g: np.ndarray) -> int:
    if np.count_nonzero(g == BG):
        return BG
    return int(Counter(map(int, g.ravel())).most_common(1)[0][0])


def _axis_score(coords: np.ndarray, values: np.ndarray, axis: np.ndarray) -> tuple[float, float, tuple[int, ...]]:
    proj = coords @ axis
    groups = []
    for color in sorted(set(map(int, values.tolist()))):
        vals = proj[values == color]
        groups.append((float(vals.mean()), color, int(vals.min()), int(vals.max())))
    groups.sort()

    penalty = 0.0
    for i in range(len(groups) - 1):
        _m0, _c0, lo0, hi0 = groups[i]
        _m1, _c1, lo1, hi1 = groups[i + 1]
        overlap = max(0, hi0 - lo1 + 1)
        penalty += 1000.0 * overlap

    compactness = sum(hi - lo + 1 for _mean, _color, lo, hi in groups)
    order = tuple(color for _mean, color, _lo, _hi in groups)
    return penalty, float(compactness), order


def _color_sequence(coords: np.ndarray, values: np.ndarray) -> list[int]:
    best_score = None
    best_order = None
    for axis in AXES.values():
        score = _axis_score(coords, values, axis)
        if best_score is None or score < best_score:
            best_score = score
            best_order = list(score[2])
    assert best_order is not None
    return best_order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn each multi-color object into a horizontal color strip ordered by the object's internal gradient direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _choose_bg(g)
    comps = _components(g != bg)
    assert comps, "expected at least one foreground object"

    objects = []
    for comp in comps:
        values = np.array([int(g[r, c]) for r, c in comp], dtype=int)
        colors = sorted(set(map(int, values.tolist())))
        assert 2 <= len(colors) <= 4, "expected each object to use 2-4 colors"
        seq = _color_sequence(comp, values)
        r0 = int(comp[:, 0].min())
        c0 = int(comp[:, 1].min())
        objects.append((c0, r0, seq))

    objects.sort(key=lambda item: (item[0], item[1]))
    strips = [seq for _c0, _r0, seq in objects]
    out = np.full((len(strips) * 2 - 1, max(len(seq) for seq in strips)), bg, dtype=int)
    for i, seq in enumerate(strips):
        out[2 * i, : len(seq)] = np.array(seq, dtype=int)
    return out


def _banded_patch(height: int, width: int, colors: list[int], axis_name: str) -> np.ndarray:
    patch = np.full((height, width), BG, dtype=int)
    axis = AXES[axis_name]
    coords = np.array([(r, c) for r in range(height) for c in range(width)], dtype=int)
    levels = coords @ axis
    unique_levels = sorted(set(map(int, levels.tolist())))
    assert len(unique_levels) >= len(colors)

    cuts = sorted(random.sample(range(1, len(unique_levels)), len(colors) - 1))
    bounds = [0] + cuts + [len(unique_levels)]
    band_of_level: dict[int, int] = {}
    for i in range(len(colors)):
        for level_idx in range(bounds[i], bounds[i + 1]):
            band_of_level[unique_levels[level_idx]] = i

    for r in range(height):
        for c in range(width):
            patch[r, c] = int(colors[band_of_level[int(r * axis[0] + c * axis[1])]])
    return patch


def _placeable(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > gh or c0 + w > gw:
        return False
    for rr in range(max(0, r0 - 1), min(gh, r0 + h + 1)):
        for cc in range(max(0, c0 - 1), min(gw, c0 + w + 1)):
            if grid[rr, cc] != BG:
                return False
    return True


def _expected_from_specs(specs: list[tuple[int, int, list[int]]]) -> np.ndarray:
    specs = sorted(specs, key=lambda item: (item[1], item[0]))
    seqs = [seq for _r0, _c0, seq in specs]
    out = np.full((len(seqs) * 2 - 1, max(len(seq) for seq in seqs)), BG, dtype=int)
    for i, seq in enumerate(seqs):
        out[2 * i, : len(seq)] = seq
    return out


def _sample_object_count() -> int:
    return random.randint(2, 5)


def _sample_axes(obj_count: int) -> list[str]:
    axis_names = list(AXES)
    picks: list[str] = []
    while len(picks) < obj_count:
        shuffled = axis_names[:]
        random.shuffle(shuffled)
        picks.extend(shuffled)
    return picks[:obj_count]


def _sample_patch_shape(axis_name: str, color_count: int) -> tuple[int, int]:
    if axis_name == "h":
        return random.randint(3, 6), random.randint(max(color_count, 5), 9)
    if axis_name == "v":
        return random.randint(max(color_count, 5), 9), random.randint(3, 6)
    side_min = max(color_count + 1, 4)
    return random.randint(side_min, 8), random.randint(side_min, 8)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(14, 28)
    width = random.randint(14, 28)
    grid = np.full((height, width), BG, dtype=int)

    obj_count = _sample_object_count()
    axis_names = _sample_axes(obj_count)
    specs: list[tuple[int, int, list[int]]] = []

    for axis_name in axis_names:
        color_count = random.randint(2, 4)
        colors = random.sample(PALETTE, color_count)
        if random.random() < 0.5:
            colors.reverse()

        ph, pw = _sample_patch_shape(axis_name, color_count)
        patch = _banded_patch(ph, pw, colors, axis_name)

        choices = [(r0, c0) for r0 in range(1, height - ph) for c0 in range(1, width - pw)]
        random.shuffle(choices)
        for r0, c0 in choices:
            if not _placeable(grid, patch, r0, c0):
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
            specs.append((r0, c0, colors[:]))
            break
        else:
            return None

    expected = _expected_from_specs(specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
