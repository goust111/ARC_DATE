from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FRAME_COLOR = 6
PATTERN_COLOR = 7
DICT_SIZE = 13
DIR_ORDER = ("up", "right", "down", "left")
DIR_VECS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
ROT_K = {
    "up": 0,
    "right": 3,
    "down": 2,
    "left": 1,
}
DICT_WINDOWS = {
    "up": (slice(1, 4), slice(5, 8)),
    "left": (slice(5, 8), slice(1, 4)),
    "right": (slice(5, 8), slice(9, 12)),
    "down": (slice(9, 12), slice(5, 8)),
}
WALL_CHOICES = (1, 2, 3, 4, 5, 8, 9)
SEED_CHOICES = (1, 2, 3, 4, 8, 9)
GRID_MIN = 24
GRID_MAX = 32
REGION_COUNT_CHOICES = (2, 3, 4)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 200


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _same_color_component(g: np.ndarray, start: tuple[int, int], color: int) -> list[tuple[int, int]]:
    h, w = g.shape
    seen = {start}
    q = deque([start])
    comp: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        comp.append((int(r), int(c)))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(g[nr, nc]) == color:
                seen.add((nr, nc))
                q.append((nr, nc))
    return comp


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _ray_distance(mask: np.ndarray, r: int, c: int, direction: str) -> int:
    dr, dc = DIR_VECS[direction]
    steps = 0
    rr, cc = r, c
    h, w = mask.shape
    while True:
        rr += dr
        cc += dc
        steps += 1
        if not (0 <= rr < h and 0 <= cc < w) or not mask[rr, cc]:
            return steps


def _mode(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _coalesce(indices: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    for idx in indices:
        if not groups or idx != groups[-1][-1] + 1:
            groups.append([idx])
        else:
            groups[-1].append(idx)
    return groups


def _bands_from_dividers(inner: np.ndarray, frame_color: int, axis: int) -> list[tuple[int, int]]:
    length = inner.shape[axis]
    if axis == 0:
        divider_rows = [i for i in range(length) if np.mean(inner[i, :] == frame_color) >= 0.8]
    else:
        divider_rows = [i for i in range(length) if np.mean(inner[:, i] == frame_color) >= 0.8]

    if divider_rows:
        cuts = [grp[len(grp) // 2] for grp in _coalesce(divider_rows)]
        bands: list[tuple[int, int]] = []
        start = 0
        for cut in cuts:
            if start < cut:
                bands.append((start, cut))
            start = cut + 1
        if start < length:
            bands.append((start, length))
        if len(bands) == 3 and all(end - start > 0 for start, end in bands):
            return bands

    assert length % 3 == 0, "dictionary interior must split into three bands"
    step = length // 3
    return [(0, step), (step, 2 * step), (2 * step, length)]


def _find_dictionary(g: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    h, w = g.shape
    corner_points = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    candidates: list[tuple[int, tuple[int, int, int, int], int]] = []

    for r, c in corner_points:
        color = int(g[r, c])
        if color == BG:
            continue
        comp = _same_color_component(g, (r, c), color)
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        if r1 - r0 + 1 < 9 or c1 - c0 + 1 < 9:
            continue
        box = g[r0 : r1 + 1, c0 : c1 + 1]
        border = np.concatenate([box[0, :], box[-1, :], box[1:-1, 0], box[1:-1, -1]])
        if np.mean(border == color) < 0.8:
            continue
        candidates.append((len(comp), bbox, color))

    assert len(candidates) == 1, "expected exactly one dictionary frame touching a grid corner"
    _, bbox, frame_color = max(candidates)
    return bbox, int(frame_color)


def _extract_patterns(g: np.ndarray, bbox: tuple[int, int, int, int], frame_color: int) -> dict[str, np.ndarray]:
    r0, c0, r1, c1 = bbox
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    row_bands = _bands_from_dividers(inner, frame_color, axis=0)
    col_bands = _bands_from_dividers(inner, frame_color, axis=1)

    windows = {
        "up": inner[row_bands[0][0] : row_bands[0][1], col_bands[1][0] : col_bands[1][1]],
        "left": inner[row_bands[1][0] : row_bands[1][1], col_bands[0][0] : col_bands[0][1]],
        "right": inner[row_bands[1][0] : row_bands[1][1], col_bands[2][0] : col_bands[2][1]],
        "down": inner[row_bands[2][0] : row_bands[2][1], col_bands[1][0] : col_bands[1][1]],
    }

    patterns: dict[str, np.ndarray] = {}
    for name, window in windows.items():
        pattern = (window != BG) & (window != frame_color)
        assert pattern.any(), f"dictionary sector {name} must contain a pattern"
        patterns[name] = pattern
    return patterns


def _seed_direction(component: list[tuple[int, int]], seed: tuple[int, int]) -> str:
    comp_mask = np.zeros((max(r for r, _ in component) + 1, max(c for _, c in component) + 1), dtype=bool)
    for r, c in component:
        comp_mask[r, c] = True
    distances = {name: _ray_distance(comp_mask, seed[0], seed[1], name) for name in DIR_ORDER}
    best = min(distances.values())
    winners = [name for name, dist in distances.items() if dist == best]
    assert len(winners) == 1, "seed direction must be unambiguous"
    return winners[0]


def _apply_pattern(
    out: np.ndarray,
    original: np.ndarray,
    component: list[tuple[int, int]],
    seed: tuple[int, int],
    seed_color: int,
    pattern: np.ndarray,
) -> None:
    r0, c0, _, _ = _bbox(component)
    ph, pw = pattern.shape
    for r, c in component:
        if (r, c) == seed or int(original[r, c]) != BG:
            continue
        if bool(pattern[(r - r0) % ph, (c - c0) % pw]):
            out[r, c] = seed_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the seed's nearest wall direction to pick the matching dictionary pattern,
    rotate that pattern to the seed orientation, and tile it inside each enclosed region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    dict_bbox, frame_color = _find_dictionary(g)
    patterns = _extract_patterns(g, dict_bbox, frame_color)

    outside_dict = np.ones(g.shape, dtype=bool)
    r0, c0, r1, c1 = dict_bbox
    outside_dict[r0 : r1 + 1, c0 : c1 + 1] = False

    nonblack_outside = g[outside_dict & (g != BG)]
    assert nonblack_outside.size > 0, "expected walls and seeds outside the dictionary"
    wall_color = _mode(nonblack_outside)

    passable = outside_dict & (g != wall_color)
    out = g.copy()
    h, w = g.shape

    for component in _components(passable):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in component):
            continue
        seeds = [(r, c) for r, c in component if int(g[r, c]) != BG]
        assert len(seeds) == 1, "each enclosed region must contain exactly one seed"
        seed = seeds[0]
        direction = _seed_direction(component, seed)
        rotated = np.rot90(patterns[direction], ROT_K[direction])
        _apply_pattern(out, g, component, seed, int(g[seed]), rotated)

    return out


def _random_pattern() -> np.ndarray:
    for _ in range(200):
        pattern = np.array([[random.random() < 0.48 for _ in range(3)] for _ in range(3)], dtype=bool)
        count = int(pattern.sum())
        if count < 3 or count > 6:
            continue
        if not _is_connected(pattern):
            continue
        rotations = {np.rot90(pattern, k).tobytes() for k in range(4)}
        if len(rotations) < 4:
            continue
        return pattern
    raise AssertionError("failed to sample a non-symmetric dictionary pattern")


def _make_dictionary_patterns() -> dict[str, np.ndarray]:
    for _ in range(200):
        raw = {name: _random_pattern() for name in DIR_ORDER}
        rotated_keys = {np.rot90(raw[name], ROT_K[name]).tobytes() for name in DIR_ORDER}
        if len(rotated_keys) == 4:
            return raw
    raise AssertionError("failed to build distinct dictionary patterns")


def _build_dictionary_patch(patterns: dict[str, np.ndarray]) -> np.ndarray:
    patch = np.full((DICT_SIZE, DICT_SIZE), BG, dtype=int)
    patch[[0, 4, 8, DICT_SIZE - 1], :] = FRAME_COLOR
    patch[:, [0, 4, 8, DICT_SIZE - 1]] = FRAME_COLOR
    for name, (rs, cs) in DICT_WINDOWS.items():
        patch[rs, cs][patterns[name]] = PATTERN_COLOR
    return patch


def _random_shape() -> np.ndarray | None:
    h = random.randint(5, 8)
    w = random.randint(5, 8)
    mask = np.ones((h, w), dtype=bool)
    notch_count = random.randint(0, 3)
    changed = False

    for _ in range(notch_count):
        side = random.choice(DIR_ORDER)
        if side in ("up", "down"):
            if h < 5 or w < 4:
                continue
            depth = random.randint(1, min(2, h - 2))
            span = random.randint(1, w - 2)
            start = random.randint(0, w - span)
            rows = slice(0, depth) if side == "up" else slice(h - depth, h)
            before = mask.copy()
            mask[rows, start : start + span] = False
            if mask.any() and _is_connected(mask):
                changed = True
            else:
                mask = before
        else:
            if w < 5 or h < 4:
                continue
            depth = random.randint(1, min(2, w - 2))
            span = random.randint(1, h - 2)
            start = random.randint(0, h - span)
            cols = slice(0, depth) if side == "left" else slice(w - depth, w)
            before = mask.copy()
            mask[start : start + span, cols] = False
            if mask.any() and _is_connected(mask):
                changed = True
            else:
                mask = before

    if not changed and random.random() < 0.8:
        return None

    mask = _crop_mask(mask)
    if mask.shape[0] < 4 or mask.shape[1] < 4:
        return None
    if int(mask.sum()) < 10 or not _is_connected(mask):
        return None
    return mask


def _shape_with_seed(direction: str) -> tuple[np.ndarray, tuple[int, int]] | None:
    for _ in range(200):
        mask = _random_shape()
        if mask is None:
            continue
        candidates: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            distances = {name: _ray_distance(mask, rr, cc, name) for name in DIR_ORDER}
            best = min(distances.values())
            if distances[direction] != best:
                continue
            if sum(dist == best for dist in distances.values()) != 1:
                continue
            if distances[direction] > 2:
                continue
            if max(distances.values()) < distances[direction] + 2:
                continue
            candidates.append((rr, cc))
        if candidates:
            return mask, random.choice(candidates)
    return None


def _region_patch(mask: np.ndarray, seed: tuple[int, int], seed_color: int, wall_color: int) -> np.ndarray:
    h, w = mask.shape
    patch = np.full((h + 2, w + 2), BG, dtype=int)
    for r, c in np.argwhere(mask):
        rr = int(r) + 1
        cc = int(c) + 1
        for nr, nc in _neighbors4(rr, cc):
            inside_mask = 1 <= nr <= h and 1 <= nc <= w and bool(mask[nr - 1, nc - 1])
            if not inside_mask:
                patch[nr, nc] = wall_color
    patch[seed[0] + 1, seed[1] + 1] = seed_color
    return patch


def _expanded_slice(start: int, size: int, limit: int) -> slice:
    return slice(max(0, start - 1), min(limit, start + size + 1))


def _direction_multiset(count: int) -> list[str]:
    bag = list(DIR_ORDER)
    random.shuffle(bag)
    if count <= 4:
        return bag[:count]
    extra = [random.choice(DIR_ORDER) for _ in range(count - 4)]
    dirs = bag + extra
    random.shuffle(dirs)
    return dirs


def _render_expected(
    inp: np.ndarray,
    patterns: dict[str, np.ndarray],
    regions: list[dict[str, object]],
) -> np.ndarray:
    out = inp.copy()
    for region in regions:
        direction = str(region["direction"])
        rotated = np.rot90(patterns[direction], ROT_K[direction])
        _apply_pattern(
            out,
            inp,
            list(region["cells"]),
            tuple(region["seed"]),
            int(region["seed_color"]),
            rotated,
        )
    return out


def _valid_output(expected: np.ndarray, regions: list[dict[str, object]]) -> bool:
    for region in regions:
        cells = list(region["cells"])
        seed = tuple(region["seed"])
        seed_color = int(region["seed_color"])
        colored = sum(int(expected[r, c]) == seed_color for r, c in cells)
        blacks = sum(int(expected[r, c]) == BG for r, c in cells if (r, c) != seed)
        if colored < 3 or blacks == 0:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(GRID_MIN, GRID_MAX)
        w = random.randint(GRID_MIN, GRID_MAX)
        grid = np.full((h, w), BG, dtype=int)

        patterns = _make_dictionary_patterns()
        dict_patch = _build_dictionary_patch(patterns)
        dict_h, dict_w = dict_patch.shape
        corner = random.choice(("tl", "tr", "bl", "br"))
        if corner == "tl":
            dict_r, dict_c = 0, 0
        elif corner == "tr":
            dict_r, dict_c = 0, w - dict_w
        elif corner == "bl":
            dict_r, dict_c = h - dict_h, 0
        else:
            dict_r, dict_c = h - dict_h, w - dict_w
        grid[dict_r : dict_r + dict_h, dict_c : dict_c + dict_w] = dict_patch

        reserved = np.zeros((h, w), dtype=bool)
        reserved[_expanded_slice(dict_r, dict_h, h), _expanded_slice(dict_c, dict_w, w)] = True

        wall_color = random.choice(WALL_CHOICES)
        seed_pool = [color for color in SEED_CHOICES if color != wall_color]
        region_count = random.choice(REGION_COUNT_CHOICES)
        if region_count > len(seed_pool):
            continue
        seed_colors = random.sample(seed_pool, region_count)
        directions = _direction_multiset(region_count)

        regions: list[dict[str, object]] = []
        ok = True
        for direction, seed_color in zip(directions, seed_colors, strict=False):
            shape_info = _shape_with_seed(direction)
            if shape_info is None:
                ok = False
                break
            mask, seed = shape_info
            patch = _region_patch(mask, seed, seed_color, wall_color)
            ph, pw = patch.shape

            placed = False
            for _attempt in range(PLACEMENT_RETRIES):
                r0 = random.randint(1, h - ph - 1)
                c0 = random.randint(1, w - pw - 1)
                if reserved[_expanded_slice(r0, ph, h), _expanded_slice(c0, pw, w)].any():
                    continue
                occupied = patch != BG
                if np.any(grid[r0 : r0 + ph, c0 : c0 + pw][occupied] != BG):
                    continue
                grid[r0 : r0 + ph, c0 : c0 + pw][occupied] = patch[occupied]
                reserved[_expanded_slice(r0, ph, h), _expanded_slice(c0, pw, w)] = True

                cells = [(r0 + int(r) + 1, c0 + int(c) + 1) for r, c in np.argwhere(mask)]
                regions.append(
                    {
                        "cells": cells,
                        "seed": (r0 + seed[0] + 1, c0 + seed[1] + 1),
                        "seed_color": seed_color,
                        "direction": direction,
                    }
                )
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok or not regions:
            continue

        expected = _render_expected(grid, patterns, regions)
        if np.array_equal(expected, grid):
            continue
        if not _valid_output(expected, regions):
            continue

        try:
            actual = main(grid)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid

    raise AssertionError("failed to generate a valid ARC sample")
