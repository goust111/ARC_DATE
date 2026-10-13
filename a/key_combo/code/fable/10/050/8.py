from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
SEED_COLOR = 5
MASK_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _component_count(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        count += 1
        stack = [(r, c)]
        seen[r, c] = True
        while stack:
            rr, cc = stack.pop()
            for nr, nc in _neighbors4(rr, cc):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return count


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    b = _bbox(mask)
    assert b is not None
    r0, c0, r1, c1 = b
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _spiral_offsets(length: int) -> tuple[list[tuple[int, int]], list[int]]:
    offsets: list[tuple[int, int]] = []
    segment_ends: list[int] = []
    r = 0
    c = 0
    step_len = 1
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    direction_index = 0

    while len(offsets) < length:
        for _ in range(2):
            dr, dc = directions[direction_index % 4]
            for _ in range(step_len):
                r += dr
                c += dc
                offsets.append((r, c))
                if len(offsets) >= length:
                    break
            segment_ends.append(len(offsets))
            direction_index += 1
            if len(offsets) >= length:
                break
        step_len += 1
    return offsets, segment_ends


def _spiral_path(seed: tuple[int, int], length: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    sr, sc = seed
    offsets, _ = _spiral_offsets(length)
    for dr, dc in offsets:
        r = sr + dr
        c = sc + dc
        assert _in_bounds(shape, r, c), "spiral path leaves the grid"
        path.append((r, c))
    return path


def _render_output(
    shape: tuple[int, int],
    bg: int,
    seed: tuple[int, int],
    ranked_counts: list[tuple[int, int]],
) -> np.ndarray:
    total = sum(count for _, count in ranked_counts)
    path = _spiral_path(seed, total, shape)
    out = np.full(shape, bg, dtype=int)
    out[seed] = SEED_COLOR

    start = 0
    for color, count in ranked_counts:
        for r, c in path[start : start + count]:
            out[r, c] = int(color)
        start += count
    assert start == total
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Turn mask-size histogram into a clockwise square spiral anchored at the gray seed.

    Input structure:
    - Black background with several non-gray foreground masks of arbitrary shape and position.
    - Exactly one gray seed pixel appears by itself and is not part of any mask.

    Rule:
    - Count the pixels of every non-background, non-gray color and sort those colors by descending count.
    - Erase the masks, keep the seed fixed, and from the seed trace a clockwise square spiral with step lengths
      1, 1, 2, 2, 3, 3, ... in up, right, down, left order.
    - Color the spiral cell-by-cell: use the largest mask's color for exactly its original pixel count, then
      switch immediately to the next color, until all counts are exhausted.

    Key insight:
    - Shape and location are distractions; only each color's pixel total survives, and that histogram is rewound
      into one continuous spiral strip whose color boundaries can fall mid-segment rather than at corners.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    seed_coords = np.argwhere(g == SEED_COLOR)
    assert seed_coords.shape == (1, 2), "expected exactly one gray seed pixel"
    seed = tuple(map(int, seed_coords[0]))

    colors = [int(color) for color in np.unique(g) if int(color) not in (bg, SEED_COLOR)]
    assert colors, "expected at least one non-gray mask color"

    ranked_counts = [(color, int(np.count_nonzero(g == color))) for color in colors]
    assert all(count > 0 for _, count in ranked_counts)
    assert len({count for _, count in ranked_counts}) == len(ranked_counts), "mask sizes must be distinct"
    ranked_counts.sort(key=lambda item: item[1], reverse=True)

    return _render_output(g.shape, bg, seed, ranked_counts)


def _split_total(total: int, parts: int, min_part: int) -> list[int]:
    sizes = [min_part] * parts
    for _ in range(total - min_part * parts):
        sizes[random.randrange(parts)] += 1
    random.shuffle(sizes)
    return sizes


def _sample_mask_dims(count: int, components: int) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    max_side = min(10, count + 2)
    for h in range(2, max_side + 1):
        for w in range(2, max_side + 1):
            area = h * w
            fill = count / area
            if area < count or area > count * 3:
                continue
            if not (0.28 <= fill <= 0.82):
                continue
            if area < count + components - 1:
                continue
            candidates.append((h, w))
    return random.choice(candidates) if candidates else None


def _sample_seeds(h: int, w: int, k: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(120):
        random.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        for cell in cells:
            if all(abs(cell[0] - rr) + abs(cell[1] - cc) > 1 for rr, cc in chosen):
                chosen.append(cell)
                if len(chosen) == k:
                    return chosen
    return None


def _grow_mask(count: int, components: int) -> np.ndarray | None:
    dims = _sample_mask_dims(count, components)
    if dims is None:
        return None
    h, w = dims
    min_part = 2 if count >= components * 2 else 1
    if min_part * components > count:
        return None
    targets = _split_total(count, components, min_part)
    seeds = _sample_seeds(h, w, components)
    if seeds is None:
        return None

    labels = np.full((h, w), -1, dtype=int)
    for idx, (r, c) in enumerate(seeds):
        labels[r, c] = idx
    sizes = [1] * components

    while sizes != targets:
        progressed = False
        order = list(range(components))
        random.shuffle(order)
        for idx in order:
            if sizes[idx] >= targets[idx]:
                continue
            frontier: list[tuple[int, int]] = []
            for rr, cc in np.argwhere(labels == idx):
                rr = int(rr)
                cc = int(cc)
                for nr, nc in _neighbors4(rr, cc):
                    if not _in_bounds(labels.shape, nr, nc) or labels[nr, nc] != -1:
                        continue
                    touching_other = False
                    for ar, ac in _neighbors4(nr, nc):
                        if _in_bounds(labels.shape, ar, ac) and labels[ar, ac] not in (-1, idx):
                            touching_other = True
                            break
                    if not touching_other:
                        frontier.append((nr, nc))
            if frontier:
                nr, nc = random.choice(frontier)
                labels[nr, nc] = idx
                sizes[idx] += 1
                progressed = True
        if not progressed:
            return None

    mask = labels >= 0
    if int(mask.sum()) != count or _component_count(mask) != components:
        return None
    return _crop_mask(mask)


def _generate_color_patch(count: int) -> tuple[np.ndarray, int] | None:
    choices = [k for k in (1, 2, 3) if k <= count]
    random.shuffle(choices)
    if 1 in choices and random.random() < 0.65:
        choices.sort(key=lambda k: (k == 1, random.random()))
    for components in choices:
        for _ in range(40):
            mask = _grow_mask(count, components)
            if mask is None:
                continue
            return mask, components
    return None


def _place_patch(
    grid: np.ndarray,
    reserved: np.ndarray,
    mask: np.ndarray,
    color: int,
) -> tuple[int, int, int, int] | None:
    h, w = mask.shape
    H, W = grid.shape
    positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(positions)
    for r, c in positions:
        r0 = max(0, r - 1)
        c0 = max(0, c - 1)
        r1 = min(H, r + h + 1)
        c1 = min(W, c + w + 1)
        if reserved[r0:r1, c0:c1].any():
            continue
        block = grid[r : r + h, c : c + w]
        if np.any(block[mask]):
            continue
        block[mask] = color
        reserved[r0:r1, c0:c1] = True
        return r, c, h, w
    return None


def _pick_counts(n_colors: int) -> list[int] | None:
    for _ in range(120):
        counts = random.sample(range(4, 19), n_colors)
        counts.sort(reverse=True)
        total = sum(counts)
        if not (24 <= total <= 70):
            continue
        _, segment_ends = _spiral_offsets(total)
        boundaries = np.cumsum(counts[:-1]).tolist()
        if not any(boundary not in set(segment_ends) for boundary in boundaries):
            continue
        return counts
    return None


def _pick_shape_and_seed(total: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    offsets, _ = _spiral_offsets(total)
    rows = [0] + [dr for dr, _ in offsets]
    cols = [0] + [dc for _, dc in offsets]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    path_h = max_r - min_r + 1
    path_w = max_c - min_c + 1

    for _ in range(120):
        H = random.randint(max(path_h + 2, 14), 30)
        W = random.randint(max(path_w + 2, 14), 30)
        if H * W < total * 3:
            continue
        seed_r_min = -min_r
        seed_r_max = H - 1 - max_r
        seed_c_min = -min_c
        seed_c_max = W - 1 - max_c
        if seed_r_min > seed_r_max or seed_c_min > seed_c_max:
            continue
        seed = (
            random.randint(seed_r_min, seed_r_max),
            random.randint(seed_c_min, seed_c_max),
        )
        return (H, W), seed
    return None


def _counterexample_ok(grid: np.ndarray, colors: list[int], ranked_counts: list[tuple[int, int]]) -> bool:
    bbox_areas: dict[int, int] = {}
    multi_component = False
    for color in colors:
        mask = grid == color
        multi_component |= _component_count(mask) > 1
        b = _bbox(mask)
        assert b is not None
        r0, c0, r1, c1 = b
        bbox_areas[color] = (r1 - r0 + 1) * (c1 - c0 + 1)

    count_order = [color for color, _ in ranked_counts]
    area_order = [color for color, _ in sorted(bbox_areas.items(), key=lambda item: item[1], reverse=True)]
    return multi_component or area_order != count_order


def _validate_instance(
    inp: np.ndarray,
    seed: tuple[int, int],
    ranked_counts: list[tuple[int, int]],
) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if int(np.count_nonzero(inp == SEED_COLOR)) != 1 or int(inp[seed]) != SEED_COLOR:
        return False

    bg = _most_common_color(inp)
    if bg != BG:
        return False

    colors = [int(color) for color in np.unique(inp) if int(color) not in (BG, SEED_COLOR)]
    if len(colors) != len(ranked_counts):
        return False

    observed_counts = [(color, int(np.count_nonzero(inp == color))) for color in colors]
    observed_counts.sort(key=lambda item: item[1], reverse=True)
    if observed_counts != ranked_counts:
        return False

    total = sum(count for _, count in ranked_counts)
    try:
        path = _spiral_path(seed, total, inp.shape)
    except AssertionError:
        return False

    out = main(inp)
    if out.shape != inp.shape:
        return False
    if int(np.count_nonzero(out == SEED_COLOR)) != 1 or int(out[seed]) != SEED_COLOR:
        return False

    expected = _render_output(inp.shape, BG, seed, ranked_counts)
    if not np.array_equal(out, expected):
        return False

    painted = {(int(r), int(c)) for r, c in np.argwhere(out != BG) if (int(r), int(c)) != seed}
    if painted != set(path):
        return False

    _, segment_ends = _spiral_offsets(total)
    boundaries = np.cumsum([count for _, count in ranked_counts[:-1]]).tolist()
    if not any(boundary not in set(segment_ends) for boundary in boundaries):
        return False

    if not _counterexample_ok(inp, colors, ranked_counts):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n_colors = random.choice([3, 4, 5])
    counts = _pick_counts(n_colors)
    if counts is None:
        return None

    layout = _pick_shape_and_seed(sum(counts))
    if layout is None:
        return None
    shape, seed = layout

    colors = random.sample(MASK_COLOR_POOL, n_colors)
    ranked_counts = list(zip(colors, counts))
    grid = np.full(shape, BG, dtype=int)
    grid[seed] = SEED_COLOR

    reserved = np.zeros(shape, dtype=bool)
    sr, sc = seed
    reserved[max(0, sr - 1) : min(shape[0], sr + 2), max(0, sc - 1) : min(shape[1], sc + 2)] = True

    patches: list[tuple[int, int, np.ndarray]] = []
    for color, count in ranked_counts:
        generated = _generate_color_patch(count)
        if generated is None:
            return None
        mask, area_components = generated
        patches.append((mask.size, color, mask))

    patches.sort(reverse=True, key=lambda item: item[0])
    for _bbox_area, color, mask in patches:
        placed = _place_patch(grid, reserved, mask, color)
        if placed is None:
            return None

    if not _validate_instance(grid, seed, ranked_counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
