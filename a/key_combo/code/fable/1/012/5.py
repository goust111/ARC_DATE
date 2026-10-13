from __future__ import annotations

import random

import numpy as np


PALETTE = tuple(range(10))
QUADRANTS = ((0, 0), (0, 1), (1, 0), (1, 1))
MAX_RETRIES = 500

BBox = tuple[int, int, int, int]


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rot_cw_coord(r: int, c: int, n: int) -> tuple[int, int]:
    return int(c), int(n - 1 - r)


def _rot_ccw_coord(r: int, c: int, n: int) -> tuple[int, int]:
    return int(n - 1 - c), int(r)


def _rotate_bbox_ccw(bbox: BBox, n: int) -> BBox:
    r0, c0, r1, c1 = bbox
    corners = np.array(
        [
            _rot_ccw_coord(r0, c0, n),
            _rot_ccw_coord(r0, c1, n),
            _rot_ccw_coord(r1, c0, n),
            _rot_ccw_coord(r1, c1, n),
        ],
        dtype=int,
    )
    return _bbox_coords(corners)


def _apply_color_map(grid: np.ndarray, color_map: dict[int, int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    src = np.array(grid, dtype=int, copy=False)
    for old, new in color_map.items():
        out[src == int(old)] = int(new)
    return out


def _find_occluder_bbox(grid: np.ndarray) -> tuple[int, BBox]:
    candidates: list[tuple[int, BBox]] = []
    for color in np.unique(grid):
        coords = np.argwhere(grid == int(color))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox_coords(coords)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 2 or h != w:
            continue
        if int(coords.shape[0]) != h * w:
            continue
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)):
            candidates.append((h * w, (r0, c0, r1, c1)))

    if len(candidates) != 1:
        raise ValueError(f"expected exactly one solid square occluder, found {len(candidates)}")
    return candidates[0]


def _is_single_cycle(color_map: dict[int, int]) -> bool:
    if len(color_map) < 2:
        return False
    keys = set(color_map)
    vals = set(color_map.values())
    if keys != vals:
        return False
    if any(int(src) == int(dst) for src, dst in color_map.items()):
        return False

    start = next(iter(keys))
    seen = set()
    cur = start
    while cur not in seen:
        seen.add(cur)
        cur = color_map[cur]
    return cur == start and seen == keys


def _infer_cycle_map(grid: np.ndarray, occ_mask: np.ndarray) -> dict[int, int]:
    n = int(grid.shape[0])
    counts: dict[int, dict[int, int]] = {}
    for r in range(n):
        for c in range(n):
            rr, cc = _rot_cw_coord(r, c, n)
            if bool(occ_mask[r, c]) or bool(occ_mask[rr, cc]):
                continue
            src = int(grid[r, c])
            dst = int(grid[rr, cc])
            bucket = counts.setdefault(src, {})
            bucket[dst] = bucket.get(dst, 0) + 1

    color_map: dict[int, int] = {}
    for src, bucket in counts.items():
        if len(bucket) != 1:
            raise ValueError(f"ambiguous color successor for {src}: {bucket}")
        dst = next(iter(bucket))
        color_map[int(src)] = int(dst)

    if not _is_single_cycle(color_map):
        raise ValueError(f"observed mapping is not a single cycle: {color_map}")
    return color_map


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a square occlusion by combining 90-degree rotation with an inferred color cycle.

    Input structure:
    - The input is a square grid whose visible pattern is invariant under: rotate the whole grid 90 degrees clockwise, then advance every pattern color to the next color in one hidden cycle.
    - One square region is covered by a solid monochrome occluder block, and its three rotated counterparts remain visible.

    Rule:
    - Find the solid square occluder.
    - Compare visible cells with their clockwise-rotated partners to infer the hidden color-successor cycle.
    - For each occluded cell, take the cell at the counterclockwise-corresponding location, then advance that source color by one step of the inferred cycle and write it into the occluder.

    Key insight:
    - Geometry alone is insufficient: the pattern repeats only after a rotation and a synchronized color permutation, so the missing patch must be borrowed from the counterclockwise counterpart and recolored by the inferred successor map.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError("expected a square 2D grid")

    n = int(g.shape[0])
    _area, occ_bbox = _find_occluder_bbox(g)
    r0, c0, r1, c1 = occ_bbox
    occ_mask = np.zeros_like(g, dtype=bool)
    occ_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    color_map = _infer_cycle_map(g, occ_mask)

    out = g.copy()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            sr, sc = _rot_ccw_coord(r, c, n)
            if bool(occ_mask[sr, sc]):
                raise ValueError("occluder overlaps its counterclockwise source")
            out[r, c] = int(color_map[int(g[sr, sc])])
    return out


def _make_rect_patch(size: int, colors: list[int]) -> np.ndarray:
    rects: list[tuple[int, int, int, int]] = [(0, 0, size, size)]
    for _ in range(random.randint(4, 8)):
        splittable = []
        for idx, (r0, c0, r1, c1) in enumerate(rects):
            if r1 - r0 >= 4 or c1 - c0 >= 4:
                splittable.append(idx)
        if not splittable:
            break

        idx = random.choice(splittable)
        r0, c0, r1, c1 = rects.pop(idx)
        axes = []
        if r1 - r0 >= 4:
            axes.append("h")
        if c1 - c0 >= 4:
            axes.append("v")
        axis = random.choice(axes)
        if axis == "h":
            cut = random.randint(r0 + 2, r1 - 2)
            rects.extend([(r0, c0, cut, c1), (cut, c0, r1, c1)])
        else:
            cut = random.randint(c0 + 2, c1 - 2)
            rects.extend([(r0, c0, r1, cut), (r0, cut, r1, c1)])

    if len(rects) < len(colors):
        return np.empty((0, 0), dtype=int)

    assignments = colors.copy()
    assignments.extend(random.choices(colors, k=len(rects) - len(colors)))
    random.shuffle(assignments)

    patch = np.empty((size, size), dtype=int)
    for (r0, c0, r1, c1), color in zip(rects, assignments, strict=False):
        patch[r0:r1, c0:c1] = int(color)
    return patch


def _make_voronoi_patch(size: int, colors: list[int]) -> np.ndarray:
    n_seeds = random.randint(5, 8)
    seeds: list[tuple[int, int]] = []
    for _ in range(200):
        if len(seeds) >= n_seeds:
            break
        r = random.randrange(size)
        c = random.randrange(size)
        if all(abs(r - sr) + abs(c - sc) >= 2 for sr, sc in seeds):
            seeds.append((r, c))
    if len(seeds) < n_seeds:
        return np.empty((0, 0), dtype=int)

    seed_colors = colors.copy()
    seed_colors.extend(random.choices(colors, k=n_seeds - len(colors)))
    random.shuffle(seed_colors)

    patch = np.empty((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            best_idx = min(
                range(n_seeds),
                key=lambda idx: (
                    abs(r - seeds[idx][0]) + abs(c - seeds[idx][1]),
                    (r - seeds[idx][0]) ** 2 + (c - seeds[idx][1]) ** 2,
                    idx,
                ),
            )
            patch[r, c] = int(seed_colors[best_idx])
    return patch


def _valid_base_patch(patch: np.ndarray, colors: list[int]) -> bool:
    if patch.shape[0] == 0:
        return False
    if patch.shape[0] != patch.shape[1]:
        return False
    used = set(map(int, patch.reshape(-1)))
    if used != set(map(int, colors)):
        return False
    if any(int(np.sum(patch == color)) < 3 for color in colors):
        return False
    if np.array_equal(patch, np.rot90(patch, -1)):
        return False
    if len({tuple(row.tolist()) for row in patch}) < 2:
        return False
    if len({tuple(col.tolist()) for col in patch.T}) < 2:
        return False
    return True


def _make_base_patch(size: int, colors: list[int]) -> np.ndarray | None:
    for _ in range(80):
        if random.random() < 0.5:
            patch = _make_rect_patch(size, colors)
        else:
            patch = _make_voronoi_patch(size, colors)
        if _valid_base_patch(patch, colors):
            return patch
    return None


def _build_full_pattern(base: np.ndarray, color_map: dict[int, int]) -> np.ndarray:
    q = int(base.shape[0])
    tl = np.array(base, dtype=int, copy=True)
    tr = _apply_color_map(np.rot90(tl, -1), color_map)
    br = _apply_color_map(np.rot90(tr, -1), color_map)
    bl = _apply_color_map(np.rot90(br, -1), color_map)

    full = np.empty((2 * q, 2 * q), dtype=int)
    full[:q, :q] = tl
    full[:q, q:] = tr
    full[q:, q:] = br
    full[q:, :q] = bl
    return full


def _choose_occluder_bbox(full: np.ndarray, q: int) -> BBox | None:
    n = int(full.shape[0])
    options: list[tuple[int, int, BBox]] = []
    max_side = min(5, q)

    for qr, qc in QUADRANTS:
        base_r = qr * q
        base_c = qc * q
        for side in range(2, max_side + 1):
            for dr in range(q - side + 1):
                for dc in range(q - side + 1):
                    bbox = (
                        base_r + dr,
                        base_c + dc,
                        base_r + dr + side - 1,
                        base_c + dc + side - 1,
                    )
                    src_bbox = _rotate_bbox_ccw(bbox, n)
                    r0, c0, r1, c1 = bbox
                    sr0, sc0, sr1, sc1 = src_bbox
                    patch = full[r0 : r1 + 1, c0 : c1 + 1]
                    source = full[sr0 : sr1 + 1, sc0 : sc1 + 1]
                    if len(np.unique(patch)) < 2 or len(np.unique(source)) < 2:
                        continue
                    if np.array_equal(patch, np.rot90(source, -1)):
                        continue
                    margin = min(dr, dc, q - (dr + side), q - (dc + side))
                    options.append((margin, side, bbox))

    if not options:
        return None

    weights = [1 + margin + side for margin, side, _bbox in options]
    choices = [bbox for _margin, _side, bbox in options]
    return random.choices(choices, weights=weights, k=1)[0]


def _gen_attempt() -> np.ndarray | None:
    q = random.randint(6, 15)
    pattern_colors = random.sample(PALETTE, 4)
    cycle_order = pattern_colors.copy()
    random.shuffle(cycle_order)
    color_map = {
        int(cycle_order[i]): int(cycle_order[(i + 1) % len(cycle_order)])
        for i in range(len(cycle_order))
    }
    occluder_color = random.choice([color for color in PALETTE if color not in pattern_colors])

    base = _make_base_patch(q, cycle_order)
    if base is None:
        return None

    full = _build_full_pattern(base, color_map)
    if np.array_equal(full, np.rot90(full, -1)):
        return None

    occ_bbox = _choose_occluder_bbox(full, q)
    if occ_bbox is None:
        return None

    r0, c0, r1, c1 = occ_bbox
    inp = full.copy()
    inp[r0 : r1 + 1, c0 : c1 + 1] = int(occluder_color)

    try:
        _area, found_bbox = _find_occluder_bbox(inp)
        if found_bbox != occ_bbox:
            return None
        if not np.array_equal(main(inp), full):
            return None
    except Exception:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
