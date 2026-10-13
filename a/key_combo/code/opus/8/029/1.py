import math
import random
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_PATH) not in sys.path:
    sys.path.append(str(_UTILS_PATH))

from arc_utils import Color, find_components, in_bounds, neighbors4


BLUE = Color.BLUE
ORANGE = Color.ORANGE
BLACK = Color.BLACK
REGION_COLORS = [Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK]
FRAME_SIZES = range(4, 9)
SHAPE_LIBRARY = [
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0)],
    [(0, 0), (0, 1), (0, 2), (1, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)],
    [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (2, 2)],
]
SIDE_TO_STEP = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _normalize_shape(coords):
    arr = np.array(coords, dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    arr = np.unique(arr, axis=0)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return [tuple(map(int, rc)) for rc in arr[order]]


def _transform_shape(coords, rot, flip):
    arr = np.array(coords, dtype=int)
    if flip:
        arr[:, 1] *= -1
    for _ in range(rot):
        arr = np.stack([-arr[:, 1], arr[:, 0]], axis=1)
    return _normalize_shape(arr)


_ALL_SHAPE_VARIANTS = []
for _base in SHAPE_LIBRARY:
    for _rot in range(4):
        for _flip in (0, 1):
            _variant = _transform_shape(_base, _rot, _flip)
            if _variant not in _ALL_SHAPE_VARIANTS:
                _ALL_SHAPE_VARIANTS.append(_variant)

SHAPE_VARIANTS = {}
for _min_rows in (1, 2):
    for _min_cols in (1, 2):
        SHAPE_VARIANTS[(_min_rows, _min_cols)] = [
            variant
            for variant in _ALL_SHAPE_VARIANTS
            if max(r for r, _ in variant) + 1 >= _min_rows and max(c for _, c in variant) + 1 >= _min_cols
        ]


def _random_shape(min_rows=1, min_cols=1):
    pool = SHAPE_VARIANTS[(min_rows, min_cols)]
    return random.choice(pool)


def _shape_bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _shift(coords, dr, dc):
    return [(r + dr, c + dc) for r, c in coords]


def _expanded_cells(coords, shape, radius=1):
    h, w = shape
    out = set()
    for r, c in coords:
        for rr in range(max(0, r - radius), min(h, r + radius + 1)):
            for cc in range(max(0, c - radius), min(w, c + radius + 1)):
                out.add((rr, cc))
    return out


def _frame_border_cells(bbox, gap):
    r0, c0, r1, c1 = bbox
    cells = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                if (r, c) != gap:
                    cells.append((r, c))
    return cells


def _frame_gap_side(gap, bbox):
    r, c = gap
    r0, c0, r1, c1 = bbox
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    return "right"


def _frame_gap_candidates(bbox, side):
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _cast_ray(grid, gap, side):
    dr, dc = SIDE_TO_STEP[side]
    r, c = gap
    ray = []
    while in_bounds(grid.shape, r, c) and int(grid[r, c]) in (BLACK, ORANGE):
        ray.append((r, c))
        r += dr
        c += dc
    blocker = None
    if in_bounds(grid.shape, r, c):
        blocker = (r, c)
    endpoint = ray[-1] if ray else None
    return ray, endpoint, blocker


def _region_components(grid):
    colors = sorted(set(map(int, np.unique(grid))) - {BLACK, BLUE, ORANGE})
    if not colors:
        return []
    return find_components(
        grid,
        bg=BLACK,
        colors=colors,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )


def _blue_frames(grid):
    return find_components(
        grid,
        bg=BLACK,
        colors=[BLUE],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )


def _angle_clockwise(center_r, center_c, coords):
    centroid_r, centroid_c = coords.mean(axis=0)
    angle = math.atan2(float(centroid_r - center_r), float(centroid_c - center_c))
    if angle < 0:
        angle += 2 * math.pi
    dist2 = float((centroid_r - center_r) ** 2 + (centroid_c - center_c) ** 2)
    return angle, dist2, float(centroid_r), float(centroid_c)


def _analyze(grid):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    frames = _blue_frames(g)
    assert 3 <= len(frames) <= 5
    regions = _region_components(g)
    assert 3 <= len(regions) <= 5

    region_by_cell = {}
    for idx, region in enumerate(regions):
        assert int(region["color"]) in REGION_COLORS
        for r, c in region["coords"]:
            region_by_cell[(int(r), int(c))] = idx

    out = g.copy()
    rays = []
    hit_counts = [0] * len(regions)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        assert 4 <= r1 - r0 + 1 <= 8
        assert 4 <= c1 - c0 + 1 <= 8
        coords = {tuple(map(int, rc)) for rc in frame["coords"]}
        missing = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                if not on_border:
                    assert g[r, c] == BLACK
                    continue
                if (r, c) in coords:
                    continue
                assert g[r, c] == BLACK
                assert r not in (r0, r1) or c not in (c0, c1)
                missing.append((r, c))
        assert len(missing) == 1
        gap = missing[0]
        side = _frame_gap_side(gap, frame["bbox"])
        ray, endpoint, blocker = _cast_ray(g, gap, side)
        assert ray
        for r, c in ray:
            out[r, c] = ORANGE

        adjacent_regions = set()
        if endpoint is not None:
            for nr, nc in neighbors4(*endpoint):
                if (nr, nc) in region_by_cell:
                    adjacent_regions.add(region_by_cell[(nr, nc)])
        if blocker is not None:
            br, bc = blocker
            if (br, bc) in region_by_cell:
                adjacent_regions.add(region_by_cell[(br, bc)])
        assert len(adjacent_regions) <= 1
        hit_region = next(iter(adjacent_regions)) if adjacent_regions else None
        if hit_region is not None:
            hit_counts[hit_region] += 1
        rays.append(
            {
                "gap": gap,
                "side": side,
                "ray": ray,
                "endpoint": endpoint,
                "blocker": blocker,
                "hit_region": hit_region,
                "frame_bbox": frame["bbox"],
            }
        )

    center_r = (g.shape[0] - 1) / 2.0
    center_c = (g.shape[1] - 1) / 2.0
    hit_region_ids = [idx for idx, count in enumerate(hit_counts) if count > 0]
    ordered_hit_regions = []
    for idx in hit_region_ids:
        region = regions[idx]
        angle, dist2, centroid_r, centroid_c = _angle_clockwise(center_r, center_c, region["coords"])
        ordered_hit_regions.append((angle, dist2, centroid_r, centroid_c, idx))
    ordered_hit_regions.sort()

    ring_ids = [idx for _angle, _dist2, _cr, _cc, idx in ordered_hit_regions]
    ring_colors = [int(regions[idx]["color"]) for idx in ring_ids]
    for pos, region_id in enumerate(ring_ids):
        step = hit_counts[region_id] % len(ring_ids)
        new_color = ring_colors[(pos + step) % len(ring_ids)]
        coords = regions[region_id]["coords"]
        out[coords[:, 0], coords[:, 1]] = new_color

    return {
        "output": out,
        "frames": frames,
        "regions": regions,
        "rays": rays,
        "hit_counts": hit_counts,
        "ring_ids": ring_ids,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor separated solid regions by rotating their colors along a centroid-angle ring, with each step count set by ray hits from blue frame gaps.

    Input structure:
    - The grid has a black background, 3-5 blue one-cell-thick hollow rectangular frames, and 3-5 separated monochrome solid regions in non-blue colors.
    - Each blue frame is an axis-aligned rectangle with side lengths 4-8 and exactly one missing non-corner border pixel.
    - The solid regions are 4-connected, mutually non-touching, and do not touch any frame.

    Rule:
    - For each blue frame, find its unique border gap and emit a one-cell-wide ray outward, perpendicular to that side; paint the traversed black cells orange and stop before the first non-black non-orange pixel or at the grid edge.
    - Count, for every non-blue non-orange region, how many rays end 4-neighbor-adjacent to that region.
    - Sort only the hit regions by centroid angle around the grid center, with 0 degrees pointing right and increasing clockwise; break equal-angle ties by smaller centroid distance to the center.
    - Replace each hit region's color with the color found that many steps later in the cyclic sorted color list, while leaving unhit regions, blue frames, and orange rays unchanged.

    Key insight:
    - The orange rays are not fill signals; they are visibility-constrained votes, and the vote totals determine how far each region advances around the spatial color cycle.
    """
    return _analyze(grid)["output"]


def _place_region(grid, coords, color):
    for r, c in coords:
        grid[r, c] = color


def _region_fits(grid, coords):
    h, w = grid.shape
    if any(not in_bounds(grid.shape, r, c) for r, c in coords):
        return False
    if any(grid[r, c] != BLACK for r, c in coords):
        return False
    expanded = _expanded_cells(coords, (h, w), radius=1)
    for r, c in expanded:
        if grid[r, c] != BLACK and (r, c) not in coords:
            return False
    return True


def _sample_center_region(shape, grid_shape):
    h, w = grid_shape
    sr0, sc0, sr1, sc1 = _shape_bbox(shape)
    sh = sr1 - sr0 + 1
    sw = sc1 - sc0 + 1
    r_center = h // 2 - sh // 2 + random.randint(-1, 1)
    c_center = w // 2 - sw // 2 + random.randint(-1, 1)
    r_center = max(4, min(r_center, h - sh - 5))
    c_center = max(4, min(c_center, w - sw - 5))
    return _shift(shape, r_center, c_center)


def _sample_blocker_region(shared_coords, side, grid):
    h, w = grid.shape
    shape = _random_shape()
    sr0, sc0, sr1, sc1 = _shape_bbox(shape)
    sh = sr1 - sr0 + 1
    sw = sc1 - sc0 + 1
    rr0, cc0, rr1, cc1 = _shape_bbox(shared_coords)
    for _ in range(200):
        gap = random.randint(2, 5)
        if side == "left":
            c0_max = cc0 - gap - sw
            if c0_max < 3:
                return None
            c0 = random.randint(3, c0_max)
            r0_min = max(3, rr0 - sh + 1)
            r0_max = min(rr1, h - sh - 3)
            if r0_min > r0_max:
                continue
            r0 = random.randint(r0_min, r0_max)
        elif side == "right":
            c0_min = cc1 + gap + 1
            c0_max = w - sw - 4
            if c0_min > c0_max:
                return None
            c0 = random.randint(c0_min, c0_max)
            r0_min = max(3, rr0 - sh + 1)
            r0_max = min(rr1, h - sh - 3)
            if r0_min > r0_max:
                continue
            r0 = random.randint(r0_min, r0_max)
        elif side == "top":
            r0_max = rr0 - gap - sh
            if r0_max < 3:
                return None
            r0 = random.randint(3, r0_max)
            c0_min = max(3, cc0 - sw + 1)
            c0_max = min(cc1, w - sw - 3)
            if c0_min > c0_max:
                continue
            c0 = random.randint(c0_min, c0_max)
        else:
            r0_min = rr1 + gap + 1
            r0_max = h - sh - 4
            if r0_min > r0_max:
                return None
            r0 = random.randint(r0_min, r0_max)
            c0_min = max(3, cc0 - sw + 1)
            c0_max = min(cc1, w - sw - 3)
            if c0_min > c0_max:
                continue
            c0 = random.randint(c0_min, c0_max)
        coords = _shift(shape, r0, c0)
        if _region_fits(grid, coords):
            return coords
    return None


def _sample_free_region(grid):
    h, w = grid.shape
    for _ in range(400):
        shape = _random_shape()
        sr0, sc0, sr1, sc1 = _shape_bbox(shape)
        sh = sr1 - sr0 + 1
        sw = sc1 - sc0 + 1
        r0 = random.randint(2, h - sh - 3)
        c0 = random.randint(2, w - sw - 3)
        coords = _shift(shape, r0, c0)
        if _region_fits(grid, coords):
            return coords
    return None


def _bbox_expanded_clear(grid, bbox):
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    if not in_bounds(grid.shape, r0, c0) or not in_bounds(grid.shape, r1, c1):
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    return bool(np.all(grid[rr0 : rr1 + 1, cc0 : cc1 + 1] == BLACK))


def _path_clear_to_region(grid, gap, side, region_id_grid, target_idx):
    ray, endpoint, blocker = _cast_ray(grid, gap, side)
    if endpoint is None or blocker is None:
        return False
    br, bc = blocker
    if int(region_id_grid[br, bc]) != target_idx:
        return False
    adjacent = set()
    for nr, nc in neighbors4(*endpoint):
        if in_bounds(grid.shape, nr, nc) and int(region_id_grid[nr, nc]) >= 0:
            adjacent.add(int(region_id_grid[nr, nc]))
    return adjacent == {target_idx}


def _enumerate_hit_candidates(grid, region_id_grid, target_idx, side_filter=None):
    coords = np.argwhere(region_id_grid == target_idx)
    candidates = []
    sides = [side_filter] if side_filter else list(SIDE_TO_STEP)
    for _ in range(900):
        side = random.choice(sides)
        fh = random.choice(list(FRAME_SIZES))
        fw = random.choice(list(FRAME_SIZES))
        cell_r, cell_c = map(int, coords[random.randrange(len(coords))])
        if side in ("left", "right"):
            off = random.randint(1, fh - 2)
            r0 = cell_r - off
            r1 = r0 + fh - 1
            if r0 < 0 or r1 >= grid.shape[0]:
                continue
            if side == "left":
                c0_min = cell_c + 2
                c0_max = grid.shape[1] - fw - 1
                if c0_min > c0_max:
                    continue
                c0 = random.randint(c0_min, c0_max)
                bbox = (r0, c0, r1, c0 + fw - 1)
                gap = (cell_r, c0)
            else:
                c1_min = fw
                c1_max = cell_c - 2
                if c1_min > c1_max:
                    continue
                c1 = random.randint(c1_min, c1_max)
                bbox = (r0, c1 - fw + 1, r1, c1)
                gap = (cell_r, c1)
        else:
            off = random.randint(1, fw - 2)
            c0 = cell_c - off
            c1 = c0 + fw - 1
            if c0 < 0 or c1 >= grid.shape[1]:
                continue
            if side == "top":
                r0_min = cell_r + 2
                r0_max = grid.shape[0] - fh - 1
                if r0_min > r0_max:
                    continue
                r0 = random.randint(r0_min, r0_max)
                bbox = (r0, c0, r0 + fh - 1, c1)
                gap = (r0, cell_c)
            else:
                r1_min = fh
                r1_max = cell_r - 2
                if r1_min > r1_max:
                    continue
                r1 = random.randint(r1_min, r1_max)
                bbox = (r1 - fh + 1, c0, r1, c1)
                gap = (r1, cell_c)
        if not _bbox_expanded_clear(grid, bbox):
            continue
        if _path_clear_to_region(grid, gap, side, region_id_grid, target_idx):
            candidates.append({"bbox": bbox, "gap": gap, "side": side, "target": target_idx})
    random.shuffle(candidates)
    unique = {}
    for cand in candidates:
        key = (cand["bbox"], cand["gap"], cand["side"], cand["target"])
        unique[key] = cand
    return list(unique.values())


def _enumerate_edge_candidates(grid):
    h, w = grid.shape
    candidates = []
    for _ in range(400):
        side = random.choice(list(SIDE_TO_STEP))
        fh = random.choice(list(FRAME_SIZES))
        fw = random.choice(list(FRAME_SIZES))
        if h - fh - 1 <= 1 or w - fw - 1 <= 1:
            continue
        r0 = random.randint(1, h - fh - 1)
        c0 = random.randint(1, w - fw - 1)
        bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        if not _bbox_expanded_clear(grid, bbox):
            continue
        gaps = _frame_gap_candidates(bbox, side)
        random.shuffle(gaps)
        for gap in gaps:
            ray, endpoint, blocker = _cast_ray(grid, gap, side)
            if blocker is None and endpoint is not None:
                candidates.append({"bbox": bbox, "gap": gap, "side": side, "target": None})
                break
    random.shuffle(candidates)
    unique = {}
    for cand in candidates:
        key = (cand["bbox"], cand["gap"], cand["side"])
        unique[key] = cand
    return list(unique.values())


def _apply_frame(grid, candidate):
    for r, c in _frame_border_cells(candidate["bbox"], candidate["gap"]):
        grid[r, c] = BLUE


def _make_region_id_grid(grid):
    region_id_grid = np.full(grid.shape, -1, dtype=int)
    regions = _region_components(grid)
    for idx, region in enumerate(regions):
        coords = region["coords"]
        region_id_grid[coords[:, 0], coords[:, 1]] = idx
    return region_id_grid, regions


def _same_or_overlap_ray(frame_a, frame_b):
    ray_a = set(frame_a["ray"])
    ray_b = set(frame_b["ray"])
    return bool(ray_a & ray_b)


def _has_behind_region(analysis, required_side):
    for ray_info in analysis["rays"]:
        if ray_info["hit_region"] is None or ray_info["side"] != required_side:
            continue
        dr, dc = SIDE_TO_STEP[ray_info["side"]]
        r, c = ray_info["blocker"]
        r += dr
        c += dc
        while in_bounds(analysis["output"].shape, r, c):
            if int(analysis["output"][r, c]) in REGION_COLORS:
                return True
            if int(analysis["output"][r, c]) == BLUE:
                break
            r += dr
            c += dc
    return False


def _validate_instance(grid, blocker_side):
    analysis = _analyze(grid)
    assert np.any(analysis["output"] != grid)
    hit_counts = analysis["hit_counts"]
    hit_regions = [idx for idx, count in enumerate(hit_counts) if count > 0]
    assert len(hit_regions) >= 2
    assert max(hit_counts) >= 2
    assert sum(1 for ray in analysis["rays"] if ray["hit_region"] is None) <= 1
    assert _has_behind_region(analysis, blocker_side)
    assert len(set(int(region["color"]) for region in analysis["regions"])) == len(analysis["regions"])
    ring = analysis["ring_ids"]
    if ring:
        changed = False
        for region_id in ring:
            coords = analysis["regions"][region_id]["coords"]
            if not np.all(grid[coords[:, 0], coords[:, 1]] == analysis["output"][coords[:, 0], coords[:, 1]]):
                changed = True
        assert changed
    return analysis


def generate_input() -> np.ndarray:
    for _ in range(500):
        h = random.randint(18, 24)
        w = random.randint(18, 24)
        frame_count = random.choices([3, 4, 5], weights=[1, 3, 4])[0]
        min_regions = 3 if frame_count <= 4 else 4
        region_count = random.randint(min_regions, 5)
        grid = np.zeros((h, w), dtype=int)

        shared_shape = _random_shape(min_rows=2, min_cols=2)
        shared_coords = _sample_center_region(shared_shape, grid.shape)
        if not _region_fits(grid, shared_coords):
            continue
        blocker_side = random.choice(["left", "right", "top", "bottom"])
        blocker_coords = _sample_blocker_region(shared_coords, blocker_side, grid)
        if blocker_coords is None:
            continue

        region_shapes = [shared_coords, blocker_coords]
        colors = random.sample(REGION_COLORS, region_count)
        for coords, color in zip(region_shapes, colors[:2]):
            _place_region(grid, coords, color)

        ok = True
        while len(region_shapes) < region_count:
            coords = _sample_free_region(grid)
            if coords is None:
                ok = False
                break
            region_shapes.append(coords)
            _place_region(grid, coords, colors[len(region_shapes) - 1])
        if not ok:
            continue

        region_id_grid, regions = _make_region_id_grid(grid)
        shared_idx = next(
            idx for idx, region in enumerate(regions) if set(map(tuple, region["coords"])) == set(shared_coords)
        )
        blocker_idx = next(
            idx for idx, region in enumerate(regions) if set(map(tuple, region["coords"])) == set(blocker_coords)
        )
        other_ids = [idx for idx in range(len(regions)) if idx not in (shared_idx,)]
        random.shuffle(other_ids)

        specs = [("hit", shared_idx, blocker_side), ("hit", shared_idx, None)]
        if frame_count >= 4:
            extra_targets = [idx for idx in other_ids if idx != shared_idx][:2]
            if len(extra_targets) < 2:
                continue
            specs.append(("hit", extra_targets[0], None))
            specs.append(("hit", extra_targets[1], None))
        else:
            non_shared = [idx for idx in other_ids if idx != shared_idx]
            if not non_shared:
                continue
            specs.append(("hit", non_shared[0], None))
        if frame_count == 5:
            specs.append(("edge", None, None))

        frame_grid = grid.copy()
        chosen = []
        failed = False
        for kind, target_idx, side_pref in specs:
            if kind == "hit":
                region_id_grid, _regions_now = _make_region_id_grid(frame_grid)
                candidates = _enumerate_hit_candidates(frame_grid, region_id_grid, target_idx, side_pref)
            else:
                candidates = _enumerate_edge_candidates(frame_grid)
            if not candidates:
                failed = True
                break
            picked = None
            for cand in candidates:
                if any(cand["bbox"] == prev["bbox"] for prev in chosen):
                    continue
                picked = cand
                break
            if picked is None:
                failed = True
                break
            _apply_frame(frame_grid, picked)
            chosen.append(picked)
        if failed:
            continue

        try:
            analysis = _validate_instance(frame_grid, blocker_side)
        except AssertionError:
            continue

        if frame_count == 5 and not any(ray["hit_region"] is None for ray in analysis["rays"]):
            continue
        if frame_count < 5 and any(ray["hit_region"] is None for ray in analysis["rays"]):
            continue
        if blocker_idx == shared_idx:
            continue
        return frame_grid

    raise RuntimeError("failed to generate a valid puzzle instance")


if __name__ == "__main__":
    path = Path(__file__)
    mod = SourceFileLoader("self_arc_task", str(path)).load_module()
    sigs = set()
    for _ in range(20):
        inp = mod.generate_input()
        out = mod.main(inp)
        sigs.add((inp.shape, inp.tobytes(), out.tobytes()))
    print("samples", len(sigs))
