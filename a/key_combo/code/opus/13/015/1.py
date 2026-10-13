import itertools
from typing import Any

import numpy as np


BG = 0
MASK = 5
MARKER = 7
PAINT_COLORS = (1, 2, 3, 4)
MASK_CORNERS = ((0, 0), (0, 2), (2, 0), (2, 2))
DIR_ORDER = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _build_mask_patterns() -> dict[int, list[np.ndarray]]:
    patterns = {k: [] for k in range(5)}
    cells = [(r, c) for r in range(3) for c in range(3)]
    for chosen in itertools.combinations(cells, 5):
        arr = np.zeros((3, 3), dtype=int)
        for r, c in chosen:
            arr[r, c] = 1
        corner_count = sum(int(arr[r, c]) for r, c in MASK_CORNERS)
        patterns[corner_count].append(arr)
    return patterns


MASK_PATTERNS_BY_K = _build_mask_patterns()


def _directions_for_k(k: int) -> tuple[tuple[int, int], ...]:
    if not 0 <= int(k) <= 4:
        raise AssertionError(f"invalid corner count: {k}")
    return DIR_ORDER[: int(k)]


def _paint_clipped_square(grid: np.ndarray, top: int, left: int, color: int) -> bool:
    h, w = grid.shape
    r0 = max(int(top), 0)
    c0 = max(int(left), 0)
    r1 = min(int(top) + 2, h)
    c1 = min(int(left) + 2, w)
    if r0 >= r1 or c0 >= c1:
        return False
    grid[r0:r1, c0:c1] = int(color)
    return True


def _visible_stamp_count(shape: tuple[int, int], top: int, left: int, dr: int, dc: int) -> int:
    h, w = shape
    count = 0
    rr = int(top)
    cc = int(left)
    while rr < h and cc < w and rr + 1 >= 0 and cc + 1 >= 0:
        count += 1
        rr += int(dr)
        cc += int(dc)
    return count


def _seed_paint_color(seed: np.ndarray) -> int | None:
    orange_count = int(np.count_nonzero(seed == MARKER))
    if not 1 <= orange_count <= 2:
        return None
    if np.any(seed == BG):
        return None
    colors = {int(v) for v in np.unique(seed)}
    paint_colors = colors - {MARKER}
    if len(paint_colors) != 1:
        return None
    paint = next(iter(paint_colors))
    if paint in {BG, MASK, MARKER}:
        return None
    return int(paint)


def _find_groups(grid: np.ndarray) -> list[dict[str, Any]]:
    h, w = grid.shape
    groups = []
    used_seeds: set[tuple[int, int]] = set()
    for mr in range(h - 2):
        for mc in range(w - 5):
            mask_block = grid[mr : mr + 3, mc : mc + 3]
            if np.count_nonzero(mask_block == MASK) != 5:
                continue
            if np.any((mask_block != BG) & (mask_block != MASK)):
                continue
            if not np.all(grid[mr : mr + 3, mc + 3] == BG):
                continue

            matches = []
            for seed_top in (mr, mr + 1):
                if seed_top + 1 >= h:
                    continue
                seed_left = mc + 4
                seed = grid[seed_top : seed_top + 2, seed_left : seed_left + 2]
                if seed.shape != (2, 2):
                    continue
                paint = _seed_paint_color(seed)
                if paint is not None:
                    matches.append((seed_top, seed_left, paint))

            if len(matches) != 1:
                continue

            seed_top, seed_left, paint = matches[0]
            if (seed_top, seed_left) in used_seeds:
                continue
            used_seeds.add((seed_top, seed_left))
            corner_count = sum(int(mask_block[r, c] == MASK) for r, c in MASK_CORNERS)
            groups.append(
                {
                    "mask_top": int(mr),
                    "mask_left": int(mc),
                    "seed_top": int(seed_top),
                    "seed_left": int(seed_left),
                    "paint": int(paint),
                    "k": int(corner_count),
                }
            )
    groups.sort(key=lambda g: (g["mask_top"], g["mask_left"], g["seed_top"], g["seed_left"]))
    return groups


def _apply_groups(base: np.ndarray, groups: list[dict[str, Any]]) -> np.ndarray:
    out = np.array(base, dtype=int, copy=True)
    for group in groups:
        sr = int(group["seed_top"])
        sc = int(group["seed_left"])
        paint = int(group["paint"])
        out[sr : sr + 2, sc : sc + 2] = paint
        for dr, dc in _directions_for_k(int(group["k"])):
            rr = sr
            cc = sc
            while _paint_clipped_square(out, rr, cc, paint):
                rr += dr
                cc += dc
    return out


def _ray_mask(shape: tuple[int, int], group: dict[str, Any]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    sr = int(group["seed_top"])
    sc = int(group["seed_left"])
    for dr, dc in _directions_for_k(int(group["k"])):
        rr = sr
        cc = sc
        while rr < h and cc < w and rr + 1 >= 0 and cc + 1 >= 0:
            r0 = max(rr, 0)
            c0 = max(cc, 0)
            r1 = min(rr + 2, h)
            c1 = min(cc + 2, w)
            if r0 < r1 and c0 < c1:
                mask[r0:r1, c0:c1] = True
            rr += dr
            cc += dc
    return mask


def _bbox_gap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    row_gap = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    col_gap = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return max(int(row_gap), int(col_gap))


def _candidate_positions(h: int, w: int, k: int) -> list[dict[str, int]]:
    min_stamps = 4
    candidates = []
    for mask_top in range(1, h - 2):
        for mask_left in range(1, w - 5):
            seed_left = mask_left + 4
            for seed_offset in (0, 1):
                seed_top = mask_top + seed_offset
                if seed_top + 1 >= h:
                    continue
                if any(
                    _visible_stamp_count((h, w), seed_top, seed_left, dr, dc) < min_stamps
                    for dr, dc in _directions_for_k(k)
                ):
                    continue
                candidates.append(
                    {
                        "mask_top": int(mask_top),
                        "mask_left": int(mask_left),
                        "seed_top": int(seed_top),
                        "seed_left": int(seed_left),
                        "bbox": (int(mask_top), int(mask_left), int(mask_top + 2), int(mask_left + 5)),
                    }
                )
    return candidates


def _build_input_from_groups(shape: tuple[int, int], groups: list[dict[str, Any]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for group in groups:
        mask_top = int(group["mask_top"])
        mask_left = int(group["mask_left"])
        seed_top = int(group["seed_top"])
        seed_left = int(group["seed_left"])
        paint = int(group["paint"])
        pattern = group["pattern"]
        marker_positions = group["marker_positions"]

        grid[mask_top : mask_top + 3, mask_left : mask_left + 3] = pattern * MASK
        grid[seed_top : seed_top + 2, seed_left : seed_left + 2] = paint
        for local_r, local_c in marker_positions:
            grid[seed_top + local_r, seed_left + local_c] = MARKER
    return grid


def _validate_layout(grid: np.ndarray, groups: list[dict[str, Any]]) -> bool:
    h, w = grid.shape
    ordered = sorted(groups, key=lambda g: (g["mask_top"], g["mask_left"], g["seed_top"], g["seed_left"]))

    masks = []
    for group in ordered:
        core = np.zeros((h, w), dtype=bool)
        mr = int(group["mask_top"])
        mc = int(group["mask_left"])
        sr = int(group["seed_top"])
        sc = int(group["seed_left"])
        core[mr : mr + 3, mc : mc + 3] = True
        core[sr : sr + 2, sc : sc + 2] = True
        masks.append(core)

    ray_only = []
    for group in ordered:
        ray = _ray_mask((h, w), group)
        sr = int(group["seed_top"])
        sc = int(group["seed_left"])
        ray[sr : sr + 2, sc : sc + 2] = False
        ray_only.append(ray)

    for i in range(len(ordered)):
        for j in range(len(ordered)):
            if i == j:
                continue
            if np.any(ray_only[i] & masks[j]):
                return False

    overlap_counter = np.zeros((h, w), dtype=int)
    for ray in ray_only:
        overlap_counter += ray.astype(int)
    if int(np.count_nonzero(overlap_counter >= 2)) > 10:
        return False

    out = _apply_groups(grid, ordered)
    if int(np.count_nonzero(out != BG)) > int(h * w * 0.45):
        return False

    found = _find_groups(grid)
    expected = [
        (g["mask_top"], g["mask_left"], g["seed_top"], g["seed_left"], g["paint"], g["k"])
        for g in ordered
    ]
    actual = [
        (g["mask_top"], g["mask_left"], g["seed_top"], g["seed_left"], g["paint"], g["k"])
        for g in found
    ]
    return actual == expected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify each grey 3x3 mask by corner occupancy, then use that class to choose diagonal seed-ray directions.

    Input structure:
    - The grid has a black background containing 2-4 separated mask-seed pairs.
    - Each pair has a 3x3 grey mask with exactly five grey cells and a fully occupied 2x2 seed one column to its right.
    - Each seed contains one paint color plus 1-2 orange marker cells; the seed may be top-aligned or bottom-aligned with the mask.

    Rule:
    - For each 3x3 mask, count how many of its four corners are grey; call that number k.
    - Replace the orange cells in the paired seed with the seed's paint color.
    - Use k to select diagonal ray directions in fixed order: 0 -> none, 1 -> down-right, 2 -> down-right and down-left, 3 -> add up-right, 4 -> add up-left as well.
    - Along every selected diagonal, repeatedly stamp a solid 2x2 paint-color block starting from the seed position until the block is fully outside the grid; later pairs in top-left-to-bottom-right order overwrite earlier rays on overlap.

    Key insight:
    - The mask does not map to a label cell; its corner count implicitly encodes the seed's ray pattern, while the seed's orange cells only mark pixels to recolor before propagation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    groups = _find_groups(g)
    assert groups, "expected at least one mask-seed pair"
    return _apply_groups(g, groups)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    group_count = int(rng.integers(2, 5))
    min_size = {2: 18, 3: 21, 4: 23}[group_count]
    h = int(rng.integers(min_size, 30))
    w = int(rng.integers(min_size, 30))

    ks = list(map(int, rng.choice(np.arange(5), size=group_count, replace=False)))
    paints = list(map(int, rng.choice(np.array(PAINT_COLORS), size=group_count, replace=False)))

    group_specs = []
    for k, paint in zip(ks, paints):
        pattern = MASK_PATTERNS_BY_K[k][int(rng.integers(len(MASK_PATTERNS_BY_K[k])))]
        marker_count = int(rng.integers(1, 3))
        marker_ids = rng.choice(4, size=marker_count, replace=False)
        marker_positions = [divmod(int(idx), 2) for idx in marker_ids]
        group_specs.append(
            {
                "k": int(k),
                "paint": int(paint),
                "pattern": np.array(pattern, dtype=int, copy=True),
                "marker_positions": marker_positions,
            }
        )

    candidates_by_k = {k: _candidate_positions(h, w, k) for k in set(ks)}
    if any(not candidates_by_k[k] for k in candidates_by_k):
        return None

    placed = []
    ordered_specs = sorted(group_specs, key=lambda g: (-g["k"], rng.random()))
    for spec in ordered_specs:
        candidates = candidates_by_k[spec["k"]]
        indices = rng.permutation(len(candidates))
        chosen = None
        for idx in indices[: min(len(indices), 250)]:
            cand = candidates[int(idx)]
            if all(_bbox_gap(cand["bbox"], other["bbox"]) >= 5 for other in placed):
                chosen = cand
                break
        if chosen is None:
            return None
        group = dict(spec)
        group.update(chosen)
        placed.append(group)

    grid = _build_input_from_groups((h, w), placed)
    if not _validate_layout(grid, placed):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.ndim != 2 or max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
