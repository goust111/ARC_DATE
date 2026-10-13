from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
SEED_COLORS = (1, 2, 3, 4, 7)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_non_wall(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == WALL:
                continue
            queue = deque([(r0, c0)])
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != WALL:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            colors = {int(grid[r, c]) for r, c in coords if int(grid[r, c]) not in (BG, WALL)}
            if len(colors) > 1:
                raise ValueError("each region must contain at most one seed color")
            seed = 0 if not colors else next(iter(colors))
            components.append(
                {
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "seed": int(seed),
                    "centroid_c": float(arr[:, 1].mean()),
                }
            )
    return components


def _pair_regions(components: list[dict], shape: tuple[int, int]) -> list[int]:
    h, w = shape
    labels = np.full((h, w), -1, dtype=int)
    for idx, comp in enumerate(components):
        labels[comp["coords"][:, 0], comp["coords"][:, 1]] = idx

    n = len(components)
    overlaps = np.zeros((n, n), dtype=int)
    for i, comp in enumerate(components):
        for r, c in comp["coords"]:
            rr = int(r)
            cc = w - 1 - int(c)
            j = int(labels[rr, cc])
            if j >= 0:
                overlaps[i, j] += 1

    scores: list[tuple[int, int, int, int]] = []
    for i in range(n):
        for j in range(i, n):
            score = int(overlaps[i, j]) if i == j else int(overlaps[i, j] + overlaps[j, i])
            if score <= 0:
                continue
            mirror_gap = abs((components[i]["centroid_c"] + components[j]["centroid_c"]) - (w - 1))
            scores.append((score, -int(round(mirror_gap * 1000)), -abs(i - j), i, j))

    scores.sort(reverse=True)
    pair_of = [-1] * n
    for _score, _gap, _idx_gap, i, j in scores:
        if i == j:
            if pair_of[i] == -1:
                pair_of[i] = i
            continue
        if pair_of[i] == -1 and pair_of[j] == -1:
            pair_of[i] = j
            pair_of[j] = i

    for i in range(n):
        if pair_of[i] != -1:
            continue
        j = int(np.argmax(overlaps[i]))
        pair_of[i] = j
        if pair_of[j] == -1:
            pair_of[j] = i

    for i, j in enumerate(pair_of):
        if pair_of[j] != i:
            raise ValueError("region partnering is not mutual")
    return pair_of


def _left_index(i: int, j: int, components: list[dict]) -> int:
    return i if components[i]["centroid_c"] <= components[j]["centroid_c"] else j


def _apply_rule(
    grid: np.ndarray,
    components: list[dict],
    pair_of: list[int],
    *,
    double_policy: str = "area",
    propagate_single: bool = True,
) -> np.ndarray:
    out = np.where(grid == WALL, WALL, BG).astype(int)
    done: set[int] = set()

    for i, j in enumerate(pair_of):
        if i in done:
            continue
        done.add(i)
        done.add(j)
        comp_i = components[i]
        comp_j = components[j]
        seed_i = int(comp_i["seed"])
        seed_j = int(comp_j["seed"])

        if i == j:
            fill = seed_i if seed_i else BG
            out[comp_i["coords"][:, 0], comp_i["coords"][:, 1]] = int(fill)
            continue

        if seed_i and not seed_j:
            fill = seed_i
            out[comp_i["coords"][:, 0], comp_i["coords"][:, 1]] = fill
            out[comp_j["coords"][:, 0], comp_j["coords"][:, 1]] = fill if propagate_single else BG
            continue

        if seed_j and not seed_i:
            fill = seed_j
            out[comp_j["coords"][:, 0], comp_j["coords"][:, 1]] = fill
            out[comp_i["coords"][:, 0], comp_i["coords"][:, 1]] = fill if propagate_single else BG
            continue

        if not seed_i and not seed_j:
            out[comp_i["coords"][:, 0], comp_i["coords"][:, 1]] = BG
            out[comp_j["coords"][:, 0], comp_j["coords"][:, 1]] = BG
            continue

        if double_policy == "keep_own":
            out[comp_i["coords"][:, 0], comp_i["coords"][:, 1]] = seed_i
            out[comp_j["coords"][:, 0], comp_j["coords"][:, 1]] = seed_j
            continue

        if double_policy == "left":
            left = _left_index(i, j, components)
            fill = int(components[left]["seed"])
        elif double_policy == "right":
            left = _left_index(i, j, components)
            fill = int(components[j if left == i else i]["seed"])
        else:
            if comp_i["area"] == comp_j["area"]:
                left = _left_index(i, j, components)
                fill = int(components[left]["seed"])
            else:
                fill = seed_i if comp_i["area"] > comp_j["area"] else seed_j

        out[comp_i["coords"][:, 0], comp_i["coords"][:, 1]] = fill
        out[comp_j["coords"][:, 0], comp_j["coords"][:, 1]] = fill

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill opposite-side partner regions according to seed presence, then arbitrate double-seed conflicts by region area.

    Input structure:
    - The grid contains black space separated by one-cell-wide grey walls.
    - Non-wall cells form 4-connected regions; some regions contain exactly one colored seed pixel.
    - Regions are paired by reflecting them across the vertical midline; a central region may pair with itself.

    Rule:
    - Keep grey walls unchanged and detect all non-wall 4-connected regions.
    - For each region, find the reflected partner region reached by mirroring its cells across the vertical axis.
    - If exactly one partner has a seed, flood-fill both regions with that seed color; if both partners have different seeds, compare the two region areas and flood-fill both with the larger region's seed color; a self-paired central region keeps its own seed color or stays black if unseeded.

    Key insight:
    - The output color is decided per mirrored region pair rather than per pixel: symmetry identifies the partner, while area only matters when both partners compete with different seeds.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    components = _components_non_wall(g)
    pair_of = _pair_regions(components, g.shape)
    return _apply_rule(g, components, pair_of, double_policy="area", propagate_single=True)


def _rand_composition(total: int, parts: int, low: int, high: int) -> list[int] | None:
    if total < parts * low or total > parts * high:
        return None
    out: list[int] = []
    remain = total
    for idx in range(parts):
        slots = parts - idx - 1
        lo = max(low, remain - slots * high)
        hi = min(high, remain - slots * low)
        if lo > hi:
            return None
        val = random.randint(lo, hi)
        out.append(val)
        remain -= val
    return out


def _bands_from_heights(heights: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    r = 1
    for h in heights:
        bands.append((r, r + h - 1))
        r += h + 1
    return bands


def _row_overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def _best_overlap_permutation(left_bands: list[tuple[int, int]], right_bands: list[tuple[int, int]]) -> bool:
    mat = np.array([[_row_overlap(lb, rb) for rb in right_bands] for lb in left_bands], dtype=int)
    return all(int(np.argmax(mat[i])) == i for i in range(3)) and all(int(np.argmax(mat[:, j])) == j for j in range(3))


def _place_seed(grid: np.ndarray, rows: tuple[int, int], cols: tuple[int, int], color: int) -> None:
    r = random.randint(rows[0], rows[1])
    c = random.randint(cols[0], cols[1])
    grid[r, c] = int(color)


def _make_candidate() -> np.ndarray | None:
    wl = random.randint(2, 4)
    wr = random.randint(2, 5)
    if wl == wr:
        return None
    wc = random.randint(max(3, abs(wl - wr)), 5)
    width = wl + wc + wr + 4
    if not (13 <= width <= 19):
        return None

    total_side_rows = random.randint(9, 15)
    left_heights = _rand_composition(total_side_rows, 3, 2, 7)
    right_heights = _rand_composition(total_side_rows, 3, 2, 7)
    if left_heights is None or right_heights is None:
        return None

    left_bands = _bands_from_heights(left_heights)
    right_bands = _bands_from_heights(right_heights)
    if not _best_overlap_permutation(left_bands, right_bands):
        return None

    area_diffs = [wl * left_heights[i] - wr * right_heights[i] for i in range(3)]
    pos = [i for i, diff in enumerate(area_diffs) if diff > 0]
    neg = [i for i, diff in enumerate(area_diffs) if diff < 0]
    if not pos or not neg:
        return None

    double_left = random.choice(pos)
    double_right = random.choice(neg)
    single_candidates = [i for i in range(3) if i not in {double_left, double_right}]
    if not single_candidates:
        return None
    single_pair = single_candidates[0]
    single_on_left = bool(random.getrandbits(1))

    height = total_side_rows + 4
    c_left_wall = wl + 1
    c_right_wall = c_left_wall + wc + 1
    grid = np.zeros((height, width), dtype=int)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL
    grid[:, c_left_wall] = WALL
    grid[:, c_right_wall] = WALL

    for boundary in (left_bands[0][1] + 1, left_bands[1][1] + 1):
        grid[boundary, : c_left_wall + 1] = WALL
    for boundary in (right_bands[0][1] + 1, right_bands[1][1] + 1):
        grid[boundary, c_right_wall:] = WALL

    palette = random.sample(SEED_COLORS, k=5)
    pair_colors = {
        double_left: (palette[0], palette[1]),
        double_right: (palette[2], palette[3]),
        single_pair: (palette[4], 0) if single_on_left else (0, palette[4]),
    }

    left_cols = (1, wl)
    right_cols = (c_right_wall + 1, width - 2)
    for idx in range(3):
        seed_left, seed_right = pair_colors.get(idx, (0, 0))
        if seed_left:
            _place_seed(grid, left_bands[idx], left_cols, seed_left)
        if seed_right:
            _place_seed(grid, right_bands[idx], right_cols, seed_right)

    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    components = _components_non_wall(grid)
    if not (6 <= len(components) <= 10):
        return False

    pair_of = _pair_regions(components, grid.shape)
    pair_sizes = sorted(sum(1 for j in pair_of if j == i) for i in range(len(pair_of)) if pair_of[i] == i)
    if pair_sizes != [1]:
        return False
    off_axis_pairs = {(min(i, j), max(i, j)) for i, j in enumerate(pair_of) if i != j}
    if len(off_axis_pairs) != 3:
        return False

    seeds = [int(comp["seed"]) for comp in components if int(comp["seed"])]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        return False

    pair_records: list[tuple[int, int, int, int]] = []
    used: set[int] = set()
    for i, j in enumerate(pair_of):
        if i in used or i == j:
            used.add(i)
            continue
        used.add(i)
        used.add(j)
        left = _left_index(i, j, components)
        right = j if left == i else i
        pair_records.append((left, right, components[left]["area"], components[right]["area"]))

    if len(pair_records) != 3:
        return False
    if not any(la > ra for _l, _r, la, ra in pair_records):
        return False
    if not any(la < ra for _l, _r, la, ra in pair_records):
        return False
    if not any((components[l]["seed"] == 0) ^ (components[r]["seed"] == 0) for l, r, _la, _ra in pair_records):
        return False

    truth = _apply_rule(grid, components, pair_of, double_policy="area", propagate_single=True)
    wrong_left = _apply_rule(grid, components, pair_of, double_policy="left", propagate_single=True)
    wrong_right = _apply_rule(grid, components, pair_of, double_policy="right", propagate_single=True)
    wrong_single = _apply_rule(grid, components, pair_of, double_policy="keep_own", propagate_single=False)
    if np.array_equal(truth, wrong_left):
        return False
    if np.array_equal(truth, wrong_right):
        return False
    if np.array_equal(truth, wrong_single):
        return False

    center_regions = [i for i, j in enumerate(pair_of) if i == j]
    if len(center_regions) != 1:
        return False
    center = center_regions[0]
    if components[center]["seed"] != 0:
        return False
    center_coords = components[center]["coords"]
    if np.any(truth[center_coords[:, 0], center_coords[:, 1]] != BG):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _make_candidate()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape != grid.shape:
            continue
        if not _semantic_checks(grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
