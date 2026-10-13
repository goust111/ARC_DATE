from __future__ import annotations

import random

import numpy as np


BG = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400
RECT_SIZES = ((3, 5), (3, 7), (5, 7), (3, 11), (3, 13), (5, 11))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}

    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("mask must be non-empty")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0, r1, c1)


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if g[r, c] == BG or seen[r, c]:
                continue

            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                assert int(g[rr, cc]) == color
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            r0 = min(rows)
            c0 = min(cols)
            r1 = max(rows)
            c1 = max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True

            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": len(coords),
                }
            )

    return comps


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for h in range(1, int(area**0.5) + 1):
        if area % h != 0:
            continue
        w = area // h
        pairs.append((h, w))
        if h != w:
            pairs.append((w, h))
    return pairs


def _pair_options(comp_a: dict, comp_b: dict) -> list[dict]:
    if int(comp_a["color"]) == int(comp_b["color"]):
        return []
    if int(comp_a["area"]) == int(comp_b["area"]):
        return []

    if int(comp_a["area"]) > int(comp_b["area"]):
        larger, smaller = comp_a, comp_b
    else:
        larger, smaller = comp_b, comp_a

    large_mask = np.asarray(larger["mask"], dtype=bool)
    small_mask = np.asarray(smaller["mask"], dtype=bool)
    large_h, large_w = map(int, large_mask.shape)
    small_h, small_w = map(int, small_mask.shape)
    total = int(larger["area"]) + int(smaller["area"])

    out: list[dict] = []
    seen = set()

    if large_mask.shape == small_mask.shape and total == int(np.prod(large_mask.shape)):
        if np.array_equal(~large_mask, small_mask):
            key = (large_h, large_w, 0, 0)
            seen.add(key)
            out.append(
                {
                    "larger": larger,
                    "smaller": smaller,
                    "shape": (large_h, large_w),
                    "small_offset": (0, 0),
                }
            )

    large_canvas = None
    large_area = int(larger["area"])
    small_area = int(smaller["area"])
    assert large_area + small_area == total

    for h, w in _factor_pairs(total):
        if h < large_h or w < large_w or h < small_h or w < small_w:
            continue

        large_canvas = np.zeros((h, w), dtype=bool)
        large_canvas[:large_h, :large_w] = large_mask

        for sr in range(h - small_h + 1):
            for sc in range(w - small_w + 1):
                overlap = large_canvas[sr : sr + small_h, sc : sc + small_w] & small_mask
                if bool(overlap.any()):
                    continue
                key = (h, w, sr, sc)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "larger": larger,
                        "smaller": smaller,
                        "shape": (h, w),
                        "small_offset": (sr, sc),
                    }
                )

    return out


def _unique_matching(components: list[dict]) -> list[dict]:
    n = len(components)
    assert n % 2 == 0 and n > 0

    options_by_index: dict[int, list[dict]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            for opt in _pair_options(components[i], components[j]):
                entry = {
                    "i": i,
                    "j": j,
                    "larger": opt["larger"],
                    "smaller": opt["smaller"],
                    "shape": opt["shape"],
                    "small_offset": opt["small_offset"],
                }
                options_by_index[i].append(entry)
                options_by_index[j].append(entry)

    solutions: list[list[dict]] = []

    def dfs(used: set[int], current: list[dict]) -> None:
        if len(solutions) > 1:
            return
        if len(used) == n:
            solutions.append(current.copy())
            return

        remaining = [i for i in range(n) if i not in used]
        pivot = min(remaining, key=lambda idx: len([o for o in options_by_index[idx] if o["i"] not in used and o["j"] not in used]))
        choices = [o for o in options_by_index[pivot] if o["i"] not in used and o["j"] not in used]
        for opt in choices:
            current.append(opt)
            used.add(opt["i"])
            used.add(opt["j"])
            dfs(used, current)
            used.remove(opt["i"])
            used.remove(opt["j"])
            current.pop()

    dfs(set(), [])
    assert len(solutions) == 1, "expected a unique complementary pairing"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered colored fragments by complementary shape, rebuild each full rectangle, and repaint it with the smaller fragment's color.

    Input structure:
    - The background is black.
    - The grid contains an even number of scattered colored fragments, each a 4-connected monochrome component.
    - Each true pair consists of two differently colored fragments cut from one rectangle.
    - Within a pair, the smaller fragment is uniquely determined by pixel count.

    Rule:
    - Detect every fragment and pair fragments whose shapes are complementary pieces of the same rectangle.
    - Reconstruct each rectangle by translating the two paired fragments into one shared placement.
    - Draw the completed rectangle at the larger fragment's original top-left anchor.
    - Repaint the whole reconstructed rectangle in the smaller fragment's color, and erase the smaller fragment's original cells to black.

    Key insight:
    - The pairing key is geometric complementarity rather than color, and the repaint color comes from the smaller member of each pair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _find_components(g)
    matches = _unique_matching(components)

    out = g.copy()
    for match in matches:
        smaller = match["smaller"]
        larger = match["larger"]
        h, w = map(int, match["shape"])
        r0, c0, _r1, _c1 = larger["bbox"]
        color = int(smaller["color"])

        for rr, cc in smaller["coords"]:
            out[rr, cc] = BG

        out[r0 : r0 + h, c0 : c0 + w] = color

    return out


def _random_connected_mask(h: int, w: int, target: int) -> np.ndarray:
    seed = (random.randrange(h), random.randrange(w))
    cells = {seed}

    while len(cells) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _flip_variant(mask: np.ndarray) -> list[np.ndarray]:
    variants = [mask]
    variants.append(np.flipud(mask))
    variants.append(np.fliplr(mask))
    variants.append(np.flipud(np.fliplr(mask)))
    unique: list[np.ndarray] = []
    seen = set()
    for variant in variants:
        key = variant.tobytes()
        if key not in seen:
            seen.add(key)
            unique.append(variant.copy())
    return unique


def _generate_pair(used_totals: set[int], used_areas: set[int]) -> dict | None:
    choices = [size for size in RECT_SIZES if int(size[0] * size[1]) not in used_totals]
    random.shuffle(choices)

    for h, w in choices:
        total = int(h * w)
        small_max = total // 2 - 1
        if small_max < 3:
            continue

        for _ in range(120):
            target = random.randint(3, small_max)
            if target in used_areas or (total - target) in used_areas:
                continue

            small_full = _random_connected_mask(h, w, target)
            large_full = ~small_full
            if not _connected(large_full):
                continue

            for small_variant in _flip_variant(small_full):
                large_variant = ~small_variant
                if not _connected(large_variant):
                    continue
                if not bool(large_variant[0, :].any()) or not bool(large_variant[:, 0].any()):
                    continue

                large_patch, large_bbox = _crop_mask(large_variant)
                small_patch, small_bbox = _crop_mask(small_variant)
                if int(large_patch.shape[1]) <= h:
                    continue

                return {
                    "full_shape": (h, w),
                    "large_patch": large_patch,
                    "small_patch": small_patch,
                    "large_bbox_full": large_bbox,
                    "small_bbox_full": small_bbox,
                    "areas": (int(large_patch.sum()), int(small_patch.sum())),
                }

    return None


def _stamp_patch(grid: np.ndarray, patch: np.ndarray, color: int, top: int, left: int) -> None:
    rows, cols = np.where(patch)
    grid[top + rows, left + cols] = int(color)


def _box_free(blocked: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    return not bool(blocked[top : top + h, left : left + w].any())


def _block_box(blocked: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    h_total, w_total = blocked.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h_total, top + h + 1)
    c1 = min(w_total, left + w + 1)
    blocked[r0:r1, c0:c1] = True


def _place_specs(specs: list[dict], colors: list[int], rows: int, cols: int) -> np.ndarray | None:
    grid = np.zeros((rows, cols), dtype=int)
    blocked = np.zeros((rows, cols), dtype=bool)

    indexed = list(enumerate(specs))
    indexed.sort(key=lambda item: int(np.prod(item[1]["full_shape"])), reverse=True)

    placements: dict[int, dict] = {}
    for idx, spec in indexed:
        fh, fw = map(int, spec["full_shape"])
        lh, lw = map(int, spec["large_patch"].shape)
        sh, sw = map(int, spec["small_patch"].shape)

        large_positions = [
            (r, c)
            for r in range(rows - fh + 1)
            for c in range(cols - fw + 1)
            if _box_free(blocked, r, c, fh, fw)
        ]
        random.shuffle(large_positions)
        if not large_positions:
            return None

        placed = False
        for large_top, large_left in large_positions[:80]:
            local_blocked = blocked.copy()
            local_grid = grid.copy()
            _stamp_patch(local_grid, spec["large_patch"], colors[2 * idx], large_top, large_left)
            _block_box(local_blocked, large_top, large_left, fh, fw)

            small_positions = [
                (r, c)
                for r in range(rows - sh + 1)
                for c in range(cols - sw + 1)
                if _box_free(local_blocked, r, c, sh, sw)
            ]
            random.shuffle(small_positions)
            if not small_positions:
                continue

            small_top, small_left = small_positions[0]
            _stamp_patch(local_grid, spec["small_patch"], colors[2 * idx + 1], small_top, small_left)
            _block_box(local_blocked, small_top, small_left, sh, sw)

            grid = local_grid
            blocked = local_blocked
            placements[idx] = {
                "large_top": large_top,
                "large_left": large_left,
                "small_top": small_top,
                "small_left": small_left,
            }
            placed = True
            break

        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice((1, 2, 3))
    used_totals: set[int] = set()
    used_areas: set[int] = set()
    specs: list[dict] = []

    for _ in range(pair_count):
        spec = _generate_pair(used_totals, used_areas)
        if spec is None:
            return None
        specs.append(spec)
        total = int(np.prod(spec["full_shape"]))
        used_totals.add(total)
        used_areas.update(spec["areas"])

    palette = list(range(1, 10))
    random.shuffle(palette)
    if 2 * pair_count > len(palette):
        return None
    colors = palette[: 2 * pair_count]

    min_rows = max(12, sum(spec["full_shape"][0] for spec in specs) + 2)
    min_cols = max(14, max(spec["full_shape"][1] for spec in specs) + 6)
    rows = random.randint(min_rows, min(30, min_rows + 8))
    cols = random.randint(min_cols, min(30, min_cols + 10))

    grid = _place_specs(specs, colors, rows, cols)
    if grid is None:
        return None

    comps = _find_components(grid)
    if len(comps) != 2 * pair_count:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragments input")
