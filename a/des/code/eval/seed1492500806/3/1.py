from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
AXIS = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_HEIGHT_RANGE = (10, 18)
GRID_WIDTH_RANGE = (10, 18)
AXIS_COUNT_CHOICES = (1, 2, 3)
AXIS_MIN_GAP = 4
SEED_COUNT_RANGE = (2, 4)
MAX_ATTEMPTS = 400


def _turn_ccw(dr: int, dc: int) -> tuple[int, int]:
    # The prompt fixes a 90-degree bend but leaves the handedness ambiguous.
    # Use one consistent "mirror bend" convention across solver and generator.
    return -dc, dr


def _find_axes(g: np.ndarray) -> tuple[list[int], list[int]]:
    rows = [int(r) for r in range(g.shape[0]) if np.all(g[r, :] == AXIS)]
    cols = [int(c) for c in range(g.shape[1]) if np.all(g[:, c] == AXIS)]
    assert rows or cols, "expected at least one grey symmetry axis"
    return rows, cols


def _nearest_axis(
    seed: tuple[int, int], horizontal_axes: list[int], vertical_axes: list[int]
) -> tuple[str, int]:
    r, c = seed
    candidates: list[tuple[int, int, int, str, int]] = []
    for axis in vertical_axes:
        candidates.append((abs(c - axis), 0, axis, "vertical", axis))
    for axis in horizontal_axes:
        candidates.append((abs(r - axis), 1, axis, "horizontal", axis))
    assert candidates, "expected at least one axis candidate"
    candidates.sort()
    _dist, _kind_order, _axis_order, kind, axis = candidates[0]
    return kind, axis


def _ideal_path_for_axis(
    shape: tuple[int, int],
    seed: tuple[int, int],
    axis_kind: str,
    axis_pos: int,
    seed_set: set[tuple[int, int]],
    grey_mask: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = shape
    r, c = seed
    if axis_kind == "vertical":
        assert c != axis_pos
        dr0, dc0 = (0, 1) if c < axis_pos else (0, -1)
        pre_steps = abs(c - axis_pos) - 1
    else:
        assert axis_kind == "horizontal"
        assert r != axis_pos
        dr0, dc0 = (1, 0) if r < axis_pos else (-1, 0)
        pre_steps = abs(r - axis_pos) - 1

    path: list[tuple[int, int]] = []
    rr, cc = r, c

    for _ in range(pre_steps):
        rr += dr0
        cc += dc0
        if (rr, cc) in seed_set:
            return path
        if not grey_mask[rr, cc]:
            path.append((rr, cc))

    rr += dr0
    cc += dc0
    if not (0 <= rr < h and 0 <= cc < w):
        return path
    assert grey_mask[rr, cc], "approach must cross the chosen axis"

    rr += dr0
    cc += dc0
    if not (0 <= rr < h and 0 <= cc < w):
        return path
    if (rr, cc) in seed_set:
        return path
    if not grey_mask[rr, cc]:
        path.append((rr, cc))

    dr, dc = _turn_ccw(dr0, dc0)
    while True:
        rr += dr
        cc += dc
        if not (0 <= rr < h and 0 <= cc < w):
            break
        if (rr, cc) in seed_set:
            break
        if not grey_mask[rr, cc]:
            path.append((rr, cc))

    return path


def _ray_specs_from_grid(
    g: np.ndarray,
) -> list[tuple[int, tuple[int, int], int, list[tuple[int, int]]]]:
    horizontal_axes, vertical_axes = _find_axes(g)
    grey_mask = g == AXIS
    seeds = [tuple(map(int, rc)) for rc in np.argwhere((g != BG) & (g != AXIS))]
    assert seeds, "expected at least one colored seed"
    seed_set = set(seeds)

    rays = []
    for seed in seeds:
        axis_kind, axis_pos = _nearest_axis(seed, horizontal_axes, vertical_axes)
        path = _ideal_path_for_axis(g.shape, seed, axis_kind, axis_pos, seed_set, grey_mask)
        rays.append((len(path), seed, int(g[seed]), path))
    return rays


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored crystal arms toward grey axes, bend them by 90 degrees after crossing,
    and truncate longer arms when they collide with shorter settled rays.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g == AXIS), "expected grey axis pixels"

    rays = _ray_specs_from_grid(g)
    occupied = {seed for _length, seed, _color, _path in rays}

    out = g.copy()
    for _length, seed, color, path in sorted(rays, key=lambda item: (item[0], item[1][0], item[1][1])):
        for rr, cc in path:
            if (rr, cc) in occupied:
                break
            occupied.add((rr, cc))
            out[rr, cc] = color
    return out


def _sample_spaced_positions(limit: int, count: int) -> list[int] | None:
    positions = list(range(2, limit - 2))
    valid = []
    for combo in itertools.combinations(positions, count):
        if all(combo[i + 1] - combo[i] >= AXIS_MIN_GAP for i in range(len(combo) - 1)):
            valid.append(combo)
    if not valid:
        return None
    return list(map(int, random.choice(valid)))


def _enumerate_candidates(
    shape: tuple[int, int], axis_kind: str, axis_positions: list[int]
) -> list[dict]:
    h, w = shape
    axis_positions = sorted(axis_positions)
    grey_mask = np.zeros(shape, dtype=bool)
    if axis_kind == "vertical":
        grey_mask[:, axis_positions] = True
    else:
        grey_mask[axis_positions, :] = True

    candidates: list[dict] = []
    if axis_kind == "vertical":
        for i, axis in enumerate(axis_positions):
            left_lo = 1 if i == 0 else (axis_positions[i - 1] + axis) // 2 + 1
            left_hi = axis - 2
            right_lo = axis + 2
            right_hi = w - 2 if i == len(axis_positions) - 1 else (axis + axis_positions[i + 1] - 1) // 2

            if left_lo <= left_hi:
                for r in range(1, h - 1):
                    for c in range(left_lo, left_hi + 1):
                        seed = (r, c)
                        path = _ideal_path_for_axis(shape, seed, "vertical", axis, {seed}, grey_mask)
                        if len(path) >= 3:
                            candidates.append(
                                {
                                    "seed": seed,
                                    "axis_kind": "vertical",
                                    "axis_pos": axis,
                                    "side": "left",
                                    "path": path,
                                }
                            )

            if right_lo <= right_hi:
                for r in range(1, h - 1):
                    for c in range(right_lo, right_hi + 1):
                        seed = (r, c)
                        path = _ideal_path_for_axis(shape, seed, "vertical", axis, {seed}, grey_mask)
                        if len(path) >= 3:
                            candidates.append(
                                {
                                    "seed": seed,
                                    "axis_kind": "vertical",
                                    "axis_pos": axis,
                                    "side": "right",
                                    "path": path,
                                }
                            )
    else:
        for i, axis in enumerate(axis_positions):
            above_lo = 1 if i == 0 else (axis_positions[i - 1] + axis) // 2 + 1
            above_hi = axis - 2
            below_lo = axis + 2
            below_hi = h - 2 if i == len(axis_positions) - 1 else (axis + axis_positions[i + 1] - 1) // 2

            if above_lo <= above_hi:
                for r in range(above_lo, above_hi + 1):
                    for c in range(1, w - 1):
                        seed = (r, c)
                        path = _ideal_path_for_axis(shape, seed, "horizontal", axis, {seed}, grey_mask)
                        if len(path) >= 3:
                            candidates.append(
                                {
                                    "seed": seed,
                                    "axis_kind": "horizontal",
                                    "axis_pos": axis,
                                    "side": "above",
                                    "path": path,
                                }
                            )

            if below_lo <= below_hi:
                for r in range(below_lo, below_hi + 1):
                    for c in range(1, w - 1):
                        seed = (r, c)
                        path = _ideal_path_for_axis(shape, seed, "horizontal", axis, {seed}, grey_mask)
                        if len(path) >= 3:
                            candidates.append(
                                {
                                    "seed": seed,
                                    "axis_kind": "horizontal",
                                    "axis_pos": axis,
                                    "side": "below",
                                    "path": path,
                                }
                            )
    return candidates


def _build_input_grid(
    shape: tuple[int, int], axis_kind: str, axis_positions: list[int], specs: list[dict]
) -> np.ndarray:
    h, w = shape
    g = np.full((h, w), BG, dtype=int)
    if axis_kind == "vertical":
        g[:, axis_positions] = AXIS
    else:
        g[axis_positions, :] = AXIS

    colors = random.sample(PALETTE, len(specs))
    random.shuffle(colors)
    for spec, color in zip(specs, colors):
        r, c = spec["seed"]
        g[r, c] = color
    return g


def _paths_match_specs(g: np.ndarray, specs: list[dict]) -> tuple[bool, list[int], list[set[tuple[int, int]]]]:
    grey_mask = g == AXIS
    seed_set = {tuple(map(int, rc)) for rc in np.argwhere((g != BG) & (g != AXIS))}
    lengths: list[int] = []
    path_sets: list[set[tuple[int, int]]] = []
    for spec in specs:
        path = _ideal_path_for_axis(
            g.shape,
            spec["seed"],
            spec["axis_kind"],
            spec["axis_pos"],
            seed_set,
            grey_mask,
        )
        if len(path) < 3 or len(path) != len(spec["path"]):
            return False, [], []
        lengths.append(len(path))
        path_sets.append(set(path))
    return True, lengths, path_sets


def _final_added_cells(inp: np.ndarray, out: np.ndarray) -> int:
    return int(np.count_nonzero((inp == BG) & (out != BG)))


def _select_specs(
    candidates: list[dict], n_seeds: int, collision_mode: bool
) -> list[dict] | None:
    if len(candidates) < n_seeds:
        return None

    for _ in range(120):
        chosen: list[dict] = []
        chosen_lengths: set[int] = set()
        occupied_seeds: set[tuple[int, int]] = set()
        occupied_paths: set[tuple[int, int]] = set()

        if collision_mode:
            groups: dict[tuple[int, str], list[dict]] = {}
            for cand in candidates:
                groups.setdefault((cand["axis_pos"], cand["side"]), []).append(cand)
            viable = []
            for group in groups.values():
                if len(group) < 2:
                    continue
                pairs = []
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        a = group[i]
                        b = group[j]
                        if len(a["path"]) == len(b["path"]):
                            continue
                        if set(a["path"]) & set(b["path"]):
                            pairs.append((a, b))
                if pairs:
                    viable.append(pairs)
            if not viable:
                return None
            a, b = random.choice(random.choice(viable))
            chosen.extend([a, b])
            chosen_lengths.update((len(a["path"]), len(b["path"])))
            occupied_seeds.update((a["seed"], b["seed"]))

        pool = candidates[:]
        random.shuffle(pool)
        for cand in pool:
            if len(chosen) == n_seeds:
                break
            if cand["seed"] in occupied_seeds:
                continue
            if len(cand["path"]) in chosen_lengths:
                continue
            cand_path = set(cand["path"])
            if any(spec["seed"] in cand_path or cand["seed"] in set(spec["path"]) for spec in chosen):
                continue
            if not collision_mode and (cand_path & occupied_paths):
                continue
            if collision_mode and any(
                cand["seed"] == spec["seed"] or cand_path & set(spec["path"]) for spec in chosen
            ):
                continue

            chosen.append(cand)
            chosen_lengths.add(len(cand["path"]))
            occupied_seeds.add(cand["seed"])
            occupied_paths |= cand_path

        if len(chosen) == n_seeds:
            return chosen
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    axis_kind = random.choice(("vertical", "horizontal"))
    collision_mode = random.random() < 0.5
    n_seeds = random.randint(*SEED_COUNT_RANGE)

    max_axes = min(
        len(AXIS_COUNT_CHOICES),
        1 + max(0, ((w if axis_kind == "vertical" else h) - 6) // AXIS_MIN_GAP),
    )
    if max_axes <= 0:
        return None
    axis_count = random.choice(AXIS_COUNT_CHOICES[:max_axes])
    axis_positions = _sample_spaced_positions(w if axis_kind == "vertical" else h, axis_count)
    if axis_positions is None:
        return None

    candidates = _enumerate_candidates((h, w), axis_kind, axis_positions)
    specs = _select_specs(candidates, n_seeds, collision_mode)
    if specs is None:
        return None

    inp = _build_input_grid((h, w), axis_kind, axis_positions, specs)
    ok, lengths, path_sets = _paths_match_specs(inp, specs)
    if not ok or len(set(lengths)) != len(lengths):
        return None

    overlap = any(path_sets[i] & path_sets[j] for i in range(len(path_sets)) for j in range(i + 1, len(path_sets)))
    if overlap != collision_mode:
        return None

    out = main(inp)
    added = _final_added_cells(inp, out)
    if collision_mode:
        if added >= sum(lengths):
            return None
    else:
        if added != sum(lengths):
            return None

    if added < n_seeds * 2:
        return None
    if np.count_nonzero(out != BG) > (h * w) // 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid crystal-arm input")
