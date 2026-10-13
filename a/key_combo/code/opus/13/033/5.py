import random
from collections import Counter

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
VALUE_COLORS = (1, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_H_RANGE = (14, 18)
GRID_W_RANGE = (16, 18)
REGION_AREA_RANGE = (20, 35)
GEN_RETRY_LIMIT = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _find_palette(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, dict[int, int], np.ndarray]] = []
    for r0 in range(h - 1):
        block = grid[r0 : r0 + 2, w - 2 : w]
        if block.shape != (2, 2):
            continue
        if int(block[0, 0]) != RED or int(block[1, 0]) != GREEN:
            continue
        if int(block[0, 1]) in (BLACK, GREY) or int(block[1, 1]) in (BLACK, GREY):
            continue
        if w >= 3 and not np.all(grid[r0 : r0 + 2, w - 3] == BLACK):
            continue
        if r0 > 0 and not np.all(grid[r0 - 1, w - 2 : w] == BLACK):
            continue
        if r0 + 2 < h and not np.all(grid[r0 + 2, w - 2 : w] == BLACK):
            continue
        mask = np.zeros_like(grid, dtype=bool)
        mask[r0 : r0 + 2, w - 2 : w] = True
        mapping = {RED: int(block[0, 1]), GREEN: int(block[1, 1])}
        candidates.append((r0, mapping, mask))
    assert len(candidates) == 1, "expected one unique palette block on the right edge"
    _, mapping, mask = candidates[0]
    return mapping, mask


def _denoise(grid: np.ndarray, protected: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if protected[r, c]:
                continue
            color = int(grid[r, c])
            if color in (BLACK, GREY):
                continue
            if not any(
                _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == color
                for nr, nc in _neighbors4(r, c)
            ):
                out[r, c] = BLACK
    return out


def _dent_host(
    clean: np.ndarray, r: int, c: int, mapping: dict[int, int], protected: np.ndarray
) -> int | None:
    if protected[r, c] or clean[r, c] != BLACK:
        return None
    h, w = clean.shape
    if r == 0 or c == 0 or r == h - 1 or c == w - 1:
        return None

    sub = clean[r - 1 : r + 2, c - 1 : c + 2]
    counts = {color: int(np.sum(sub == color)) for color in mapping}
    total = sum(counts.values())
    if total == 0:
        return None
    host, host_count = max(counts.items(), key=lambda item: item[1])
    if host_count * 2 <= total:
        return None

    orth = []
    for nr, nc in _neighbors4(r, c):
        orth.append(int(clean[nr, nc]))
    if any(v not in (BLACK, host) for v in orth):
        return None
    if sum(v == host for v in orth) < 3:
        return None
    return int(host)


def _repair_plan(
    clean: np.ndarray, mapping: dict[int, int], protected: np.ndarray
) -> dict[tuple[int, int], int]:
    plan: dict[tuple[int, int], int] = {}
    h, w = clean.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            host = _dent_host(clean, r, c, mapping, protected)
            if host is not None:
                plan[(r, c)] = host
    return plan


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise two colored regions, then repair their black dents using a palette-defined color remapping.

    Input structure:
    - A black background contains exactly two large irregular foreground regions, one red and one green.
    - Each region has 1-3 small black dents inside it, plus a few isolated one-cell noise specks near the regions.
    - A 2x2 palette block sits on the right edge: the left column is the fixed key colors red and green, and the right column gives their repair colors.

    Rule:
    - First remove isolated non-black, non-grey noise pixels, but keep the right-edge palette block unchanged.
    - Read the palette as a mapping from region color to dent-repair color.
    - For each non-boundary black cell outside the palette, if its cleaned 3x3 neighborhood has a strict red-or-green majority and its orthogonal neighbors show that it is an enclosed dent, recolor it with the mapped repair color.
    - Leave all other cells, including the palette itself, unchanged.

    Key insight:
    - The black defects belong to the surrounding red or green region after denoising, but the final fill color is not that region color and not the opposite region color; it must be looked up from the palette.
    """
    g = np.asarray(grid, dtype=int)
    mapping, protected = _find_palette(g)
    clean = _denoise(g, protected)

    out = clean.copy()
    for (r, c), host in _repair_plan(clean, mapping, protected).items():
        out[r, c] = mapping[host]
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[np.ix_(rows, cols)]


def _make_region_patch(rng: random.Random) -> np.ndarray | None:
    for _ in range(200):
        core_h, core_w = rng.choice([(5, 5), (5, 6), (6, 5), (4, 6), (6, 4), (4, 5), (5, 4)])
        canvas = np.zeros((core_h + 2, core_w + 2), dtype=bool)
        canvas[1 : 1 + core_h, 1 : 1 + core_w] = True
        bulges = []
        for c in range(1, core_w + 1):
            bulges.append((0, c))
            bulges.append((core_h + 1, c))
        for r in range(1, core_h + 1):
            bulges.append((r, 0))
            bulges.append((r, core_w + 1))
        rng.shuffle(bulges)
        for r, c in bulges[: rng.randint(1, 3)]:
            canvas[r, c] = True
        patch = _crop_mask(canvas)
        area = int(np.sum(patch))
        if not (REGION_AREA_RANGE[0] <= area <= REGION_AREA_RANGE[1]):
            continue
        if patch.shape[0] < 5 or patch.shape[1] < 5:
            continue
        if area == patch.shape[0] * patch.shape[1]:
            continue
        return patch
    return None


def _all_true(mask: np.ndarray, coords: tuple[tuple[int, int], ...]) -> bool:
    return all(0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and mask[r, c] for r, c in coords)


def _dent_candidates(mask: np.ndarray) -> tuple[list[tuple[tuple[int, int], ...]], list[tuple[tuple[int, int], ...]]]:
    h, w = mask.shape
    singles: list[tuple[tuple[int, int], ...]] = []
    pairs: list[tuple[tuple[int, int], ...]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            orth = tuple((nr, nc) for nr, nc in _neighbors4(r, c))
            if _all_true(mask, orth):
                singles.append(((r, c),))
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            for dr, dc in ((0, 1), (1, 0)):
                cells = ((r, c), (r + dr, c + dc))
                if not _all_true(mask, cells):
                    continue
                ok = True
                for cell in cells:
                    rr, cc = cell
                    for nr, nc in _neighbors4(rr, cc):
                        if (nr, nc) in cells:
                            continue
                        if not _in_bounds(mask.shape, nr, nc) or not mask[nr, nc]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    pairs.append(cells)
    return singles, pairs


def _cluster_far_enough(
    cluster: tuple[tuple[int, int], ...], taken: set[tuple[int, int]]
) -> bool:
    return all(max(abs(r - tr), abs(c - tc)) >= 2 for r, c in cluster for tr, tc in taken)


def _choose_dents(
    patch: np.ndarray, rng: random.Random
) -> list[tuple[tuple[int, int], ...]] | None:
    singles, pairs = _dent_candidates(patch)
    if not singles:
        return None
    for _ in range(100):
        target_clusters = rng.randint(1, 3)
        selected: list[tuple[tuple[int, int], ...]] = []
        taken: set[tuple[int, int]] = set()
        use_pair = rng.random() < 0.4
        if use_pair and pairs:
            pair_pool = pairs[:]
            rng.shuffle(pair_pool)
            for cluster in pair_pool:
                if _cluster_far_enough(cluster, taken):
                    selected.append(cluster)
                    taken.update(cluster)
                    break
        pool = singles + ([] if use_pair else pairs)
        rng.shuffle(pool)
        for cluster in pool:
            if len(selected) >= target_clusters:
                break
            if any(cell in taken for cell in cluster):
                continue
            if not _cluster_far_enough(cluster, taken):
                continue
            selected.append(cluster)
            taken.update(cluster)
        if 1 <= len(selected) <= 3:
            return selected
    return None


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, color: int) -> list[tuple[int, int]]:
    coords = []
    for r, c in np.argwhere(patch):
        rr = r0 + int(r)
        cc = c0 + int(c)
        grid[rr, cc] = color
        coords.append((rr, cc))
    return coords


def _min_manhattan(coord: tuple[int, int], cells: list[tuple[int, int]]) -> int:
    r, c = coord
    return min(abs(r - rr) + abs(c - cc) for rr, cc in cells)


def _noise_positions_for_region(
    grid: np.ndarray,
    region_cells: list[tuple[int, int]],
    forbidden: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 3):
            if forbidden[r, c] or grid[r, c] != BLACK:
                continue
            d = _min_manhattan((r, c), region_cells)
            if 2 <= d <= 3:
                candidates.append((r, c))
    return candidates


def _can_place_noise(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    if grid[r, c] != BLACK:
        return False
    return all(
        not _in_bounds(grid.shape, nr, nc) or int(grid[nr, nc]) != color
        for nr, nc in _neighbors4(r, c)
    )


def _region_noise_count(total: int, rng: random.Random) -> tuple[int, int]:
    if total == 2:
        return 1, 1
    if total == 3:
        return (2, 1) if rng.random() < 0.5 else (1, 2)
    return (2, 2) if rng.random() < 0.7 else ((3, 1) if rng.random() < 0.5 else (1, 3))


def _build_sample(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    base = np.zeros((h, w), dtype=int)

    palette_row = rng.randint(1, h - 3)
    palette_values = rng.sample(VALUE_COLORS, 2)
    mapping = {RED: palette_values[0], GREEN: palette_values[1]}
    base[palette_row, w - 2] = RED
    base[palette_row + 1, w - 2] = GREEN
    base[palette_row, w - 1] = mapping[RED]
    base[palette_row + 1, w - 1] = mapping[GREEN]
    palette_mask = np.zeros_like(base, dtype=bool)
    palette_mask[palette_row : palette_row + 2, w - 2 : w] = True

    left_patch = _make_region_patch(rng)
    right_patch = _make_region_patch(rng)
    if left_patch is None or right_patch is None:
        return None

    ph1, pw1 = left_patch.shape
    ph2, pw2 = right_patch.shape
    left_c0 = rng.randint(1, 2)
    gap_to_palette = rng.randint(2, 3)
    right_c0 = w - 2 - gap_to_palette - pw2
    if right_c0 - (left_c0 + pw1) < 2:
        return None

    left_r0 = rng.randint(1, h - ph1 - 1)
    right_r0 = rng.randint(1, h - ph2 - 1)

    colors = [RED, GREEN]
    rng.shuffle(colors)
    left_color, right_color = colors

    left_cells = _place_patch(base, left_patch, left_r0, left_c0, left_color)
    right_cells = _place_patch(base, right_patch, right_r0, right_c0, right_color)

    left_dents_local = _choose_dents(left_patch, rng)
    right_dents_local = _choose_dents(right_patch, rng)
    if left_dents_local is None or right_dents_local is None:
        return None

    dent_map: dict[tuple[int, int], int] = {}
    for cluster in left_dents_local:
        for r, c in cluster:
            dent_map[(left_r0 + r, left_c0 + c)] = left_color
    for cluster in right_dents_local:
        for r, c in cluster:
            dent_map[(right_r0 + r, right_c0 + c)] = right_color

    expected = base.copy()
    for (r, c), host in dent_map.items():
        expected[r, c] = mapping[host]

    inp = base.copy()
    for r, c in dent_map:
        inp[r, c] = BLACK

    noise_total = rng.randint(2, 4)
    left_noise_n, right_noise_n = _region_noise_count(noise_total, rng)
    forbidden = palette_mask.copy()
    for r, c in dent_map:
        forbidden[r, c] = True

    region_info = [
        (left_cells, left_color, right_color, left_noise_n),
        (right_cells, right_color, left_color, right_noise_n),
    ]
    cross_noise_used = False
    for region_cells, host, other, need in region_info:
        candidates = _noise_positions_for_region(inp, region_cells, forbidden)
        rng.shuffle(candidates)
        placed = 0
        for r, c in candidates:
            if placed >= need:
                break
            noise_color = other if (placed == 0 and rng.random() < 0.5) or (need == 1 and not cross_noise_used and rng.random() < 0.8) else host
            if noise_color == other:
                cross_noise_used = True
            if not _can_place_noise(inp, r, c, noise_color):
                continue
            inp[r, c] = noise_color
            forbidden[r, c] = True
            placed += 1
        if placed != need:
            return None

    clean = _denoise(inp, palette_mask)
    repairs = _repair_plan(clean, mapping, palette_mask)
    if repairs != dent_map:
        return None
    if not np.array_equal(main(inp), expected):
        return None

    changed_to_black = set(map(tuple, np.argwhere((inp != BLACK) & (clean == BLACK))))
    if len(changed_to_black) != noise_total:
        return None
    if not cross_noise_used and noise_total >= 3:
        return None
    if any(
        np.sum(expected == mapping[color]) <= np.sum(base == mapping[color]) for color in (RED, GREEN)
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRY_LIMIT):
        sample = _build_sample(rng)
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid input")
