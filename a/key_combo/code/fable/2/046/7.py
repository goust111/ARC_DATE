from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
HEIGHT_RANGE = (8, 15)
WIDTH_RANGE = (8, 15)
PIXEL_COUNT_CHOICES = (2, 3, 4)
PIXEL_COUNT_WEIGHTS = (1, 4, 3)
MAX_GENERATION_ATTEMPTS = 800


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _corners(shape: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    h, w = map(int, shape)
    return ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _profile(pos: tuple[int, int], shape: tuple[int, int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    dists = tuple(_manhattan(pos, corner) for corner in _corners(shape))
    assert len(set(dists)) == 4, "every pixel must have four distinct corner distances"
    order = tuple(sorted(range(4), key=lambda idx: dists[idx]))
    return dists, order


def _extract_pixels(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], list[int]]:
    coords = np.argwhere(grid != bg)
    assert 2 <= int(coords.shape[0]) <= 4, "expected between two and four lone pixels"

    positions: list[tuple[int, int]] = []
    colors: list[int] = []
    seen_colors: set[int] = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        color = int(grid[rr, cc])
        assert color not in seen_colors, "foreground colors must be distinct"
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                assert int(grid[nr, nc]) == bg, "foreground pixels must be 4-isolated"
        positions.append((rr, cc))
        colors.append(color)
        seen_colors.add(color)
    return positions, colors


def _greedy_matching(
    positions: list[tuple[int, int]], shape: tuple[int, int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    pair_list: list[tuple[int, int, int]] = []
    pair_dists: list[int] = []
    for pixel_idx, pos in enumerate(positions):
        for corner_idx, corner in enumerate(_corners(shape)):
            dist = _manhattan(pos, corner)
            pair_list.append((int(dist), int(pixel_idx), int(corner_idx)))
            pair_dists.append(int(dist))

    assert len(set(pair_dists)) == len(pair_dists), "all pixel-corner distances must be unique"

    chosen_corners = [-1] * len(positions)
    chosen_dists = [-1] * len(positions)
    used_corners: set[int] = set()
    for dist, pixel_idx, corner_idx in sorted(pair_list):
        if chosen_corners[pixel_idx] != -1 or corner_idx in used_corners:
            continue
        chosen_corners[pixel_idx] = int(corner_idx)
        chosen_dists[pixel_idx] = int(dist)
        used_corners.add(int(corner_idx))
        if len(used_corners) == len(positions):
            break

    assert all(corner_idx != -1 for corner_idx in chosen_corners), "every pixel must claim a corner"
    return tuple(chosen_corners), tuple(chosen_dists)


def _render(
    positions: list[tuple[int, int]],
    colors: list[int],
    shape: tuple[int, int],
    corner_ids: tuple[int, ...],
    pair_dists: tuple[int, ...],
    bg: int,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    corners = _corners(shape)
    order = sorted(range(len(positions)), key=lambda idx: pair_dists[idx], reverse=True)
    for idx in order:
        r, c = positions[idx]
        corner_r, corner_c = corners[corner_ids[idx]]
        r0, r1 = sorted((int(r), int(corner_r)))
        c0, c1 = sorted((int(c), int(corner_c)))
        out[r0 : r1 + 1, c0 : c1 + 1] = int(colors[idx])
    return out


def _solve(
    positions: list[tuple[int, int]], colors: list[int], shape: tuple[int, int], bg: int
) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray]:
    corner_ids, pair_dists = _greedy_matching(positions, shape)
    out = _render(positions, colors, shape, corner_ids, pair_dists, bg)
    return corner_ids, pair_dists, out


def _independent_nearest_output(
    positions: list[tuple[int, int]], colors: list[int], shape: tuple[int, int], bg: int
) -> np.ndarray:
    profiles = [_profile(pos, shape) for pos in positions]
    corner_ids = tuple(profile[1][0] for profile in profiles)
    pair_dists = tuple(profile[0][profile[1][0]] for profile in profiles)
    return _render(positions, colors, shape, corner_ids, pair_dists, bg)


def _available_corner_output(
    positions: list[tuple[int, int]],
    colors: list[int],
    shape: tuple[int, int],
    bg: int,
    pixel_order: list[int],
) -> np.ndarray:
    profiles = [_profile(pos, shape) for pos in positions]
    available = set(range(4))
    corner_ids = [-1] * len(positions)
    pair_dists = [-1] * len(positions)
    for idx in pixel_order:
        dists, order = profiles[idx]
        for corner_idx in order:
            if corner_idx in available:
                corner_ids[idx] = int(corner_idx)
                pair_dists[idx] = int(dists[corner_idx])
                available.remove(int(corner_idx))
                break
    assert all(corner_idx != -1 for corner_idx in corner_ids)
    return _render(positions, colors, shape, tuple(corner_ids), tuple(pair_dists), bg)


def _coverage_counts(positions: list[tuple[int, int]], shape: tuple[int, int], corner_ids: tuple[int, ...]) -> np.ndarray:
    counts = np.zeros(shape, dtype=int)
    corners = _corners(shape)
    for pos, corner_idx in zip(positions, corner_ids):
        r, c = pos
        corner_r, corner_c = corners[int(corner_idx)]
        r0, r1 = sorted((int(r), int(corner_r)))
        c0, c1 = sorted((int(c), int(corner_c)))
        counts[r0 : r1 + 1, c0 : c1 + 1] += 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Greedily match lone pixels to unique corners, then fill each claimed corner rectangle.

    Input structure:
    - The background is a uniform field containing between two and four foreground pixels.
    - Every foreground object is a single 4-isolated pixel, and the foreground colors are all distinct.
    - Across all foreground pixels and all four grid corners, every Manhattan distance is different.
    - At least two pixels are nearest to the same corner, so corner competition is visible.

    Rule:
    - Repeatedly choose the smallest Manhattan-distance pair among unmatched pixels and still-unclaimed corners.
    - Each matched pixel claims that corner, and no later pixel may use it.
    - After all pixels are matched, fill the inclusive axis-aligned rectangle between each pixel and its claimed corner with that pixel's color.
    - If rectangles overlap, the pixel with the smaller matching distance wins on the overlap.

    Key insight:
    - This is not "each pixel goes to its own nearest corner": once a corner is claimed, losers must fall back to later choices, creating a greedy preemption chain.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    positions, colors = _extract_pixels(g, bg)

    nearest_corners = [_profile(pos, g.shape)[1][0] for pos in positions]
    assert Counter(nearest_corners).most_common(1)[0][1] >= 2, "expected real corner competition"

    _corner_ids, _pair_dists, out = _solve(positions, colors, g.shape, bg)
    return out


def _non_adjacent(pos: tuple[int, int], placed: list[dict]) -> bool:
    return all(abs(pos[0] - info["pos"][0]) + abs(pos[1] - info["pos"][1]) != 1 for info in placed)


def _candidate_infos(shape: tuple[int, int]) -> list[dict]:
    h, w = map(int, shape)
    blocked = set(_corners(shape))
    infos: list[dict] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in blocked:
                continue
            dists = tuple(_manhattan((r, c), corner) for corner in _corners(shape))
            if len(set(dists)) != 4:
                continue
            order = tuple(sorted(range(4), key=lambda idx: dists[idx]))
            infos.append({"pos": (int(r), int(c)), "dists": dists, "order": order})
    return infos


def _pick_preemptors(victim: dict, infos: list[dict], pixel_count: int) -> list[dict] | None:
    target_corners = list(victim["order"][: pixel_count - 1])
    chosen: list[dict] = [victim]
    used_dists = set(victim["dists"])

    def rec(step: int) -> list[dict] | None:
        if step == len(target_corners):
            return [dict(info) for info in chosen]

        target_corner = int(target_corners[step])
        candidates = [
            info
            for info in infos
            if info["order"][0] == target_corner
            and int(info["dists"][target_corner]) < int(victim["dists"][target_corner])
            and info["pos"] != victim["pos"]
            and _non_adjacent(info["pos"], chosen)
            and set(info["dists"]).isdisjoint(used_dists)
        ]
        random.shuffle(candidates)
        for info in candidates:
            chosen.append(info)
            used_dists.update(info["dists"])
            result = rec(step + 1)
            if result is not None:
                return result
            chosen.pop()
            used_dists.difference_update(info["dists"])
        return None

    return rec(0)


def _semantic_instance_ok(selected: list[dict], shape: tuple[int, int]) -> bool:
    positions = [info["pos"] for info in selected]
    colors = list(range(1, len(selected) + 1))
    all_dists = [dist for info in selected for dist in info["dists"]]

    if len(set(all_dists)) != len(all_dists):
        return False
    if Counter(info["order"][0] for info in selected).most_common(1)[0][1] < 2:
        return False

    corner_ids, pair_dists, out = _solve(positions, colors, shape, BACKGROUND)
    victim_rank = selected[0]["order"].index(corner_ids[0])
    if victim_rank != len(selected) - 1:
        return False

    coverage = _coverage_counts(positions, shape, corner_ids)
    painted_fraction = float(np.mean(coverage > 0))
    overlap_fraction = float(np.mean(coverage > 1))
    if overlap_fraction == 0.0:
        return False
    if not (0.12 <= painted_fraction <= 0.72):
        return False
    if overlap_fraction > 0.22:
        return False

    independent = _independent_nearest_output(positions, colors, shape, BACKGROUND)
    if np.array_equal(out, independent):
        return False

    row_major = sorted(range(len(positions)), key=lambda idx: positions[idx])
    nearest_first = sorted(
        range(len(positions)),
        key=lambda idx: (
            selected[idx]["dists"][selected[idx]["order"][0]],
            positions[idx][0],
            positions[idx][1],
        ),
    )
    row_major_out = _available_corner_output(positions, colors, shape, BACKGROUND, row_major)
    nearest_first_out = _available_corner_output(positions, colors, shape, BACKGROUND, nearest_first)
    if np.array_equal(out, row_major_out) and np.array_equal(out, nearest_first_out):
        return False

    grid = np.full(shape, BACKGROUND, dtype=int)
    for pos, color in zip(positions, colors):
        grid[pos] = int(color)
    return np.array_equal(main(grid), out)


def _build_candidate_grid(shape: tuple[int, int], pixel_count: int) -> np.ndarray | None:
    infos = _candidate_infos(shape)
    victim_pool = [
        info
        for info in infos
        if 0 < info["pos"][0] < shape[0] - 1 and 0 < info["pos"][1] < shape[1] - 1
    ]
    if not victim_pool:
        return None

    random.shuffle(victim_pool)
    for victim in victim_pool:
        selected = _pick_preemptors(victim, infos, pixel_count)
        if selected is None or not _semantic_instance_ok(selected, shape):
            continue
        colors = random.sample(COLOR_POOL, k=pixel_count)
        grid = np.full(shape, BACKGROUND, dtype=int)
        for info, color in zip(selected, colors):
            grid[info["pos"]] = int(color)
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        pixel_count = random.choices(PIXEL_COUNT_CHOICES, weights=PIXEL_COUNT_WEIGHTS, k=1)[0]
        min_side = 12 if pixel_count == 4 else 10 if pixel_count == 3 else HEIGHT_RANGE[0]
        h = random.randint(min_side, HEIGHT_RANGE[1])
        w = random.randint(min_side, WIDTH_RANGE[1])
        grid = _build_candidate_grid((h, w), int(pixel_count))
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate corner-claim puzzle input")
