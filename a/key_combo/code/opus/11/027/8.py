import itertools
import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GUIDE_COLORS = [BLUE, RED]
GRID_SIDE_RANGE = (17, 24)
GEN_RETRIES = 400


def _bbox_from_coords(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_h(bbox):
    return bbox[2] - bbox[0] + 1


def _bbox_w(bbox):
    return bbox[3] - bbox[1] + 1


def _bbox_center(bbox):
    r0, c0, r1, c1 = bbox
    rr = r0 + (_bbox_h(bbox) - 1) / 2
    cc = c0 + (_bbox_w(bbox) - 1) / 2
    return int(round(rr)), int(round(cc))


def _crop_bbox(grid, bbox):
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst, patch, r0, c0, transparent=None):
    h, w = patch.shape
    assert 0 <= r0 and 0 <= c0 and r0 + h <= dst.shape[0] and c0 + w <= dst.shape[1]
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]
    return dst


def _connected(mask):
    coords = list(map(tuple, np.argwhere(mask)))
    if not coords:
        return False
    seen = {coords[0]}
    q = deque([coords[0]])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == len(coords)


def _find_components(grid, colors):
    colors = set(map(int, colors))
    seen = np.zeros(grid.shape, dtype=bool)
    comps = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if seen[r, c] or color not in colors:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return comps


def _local_coords(mask):
    return list(map(tuple, np.argwhere(mask)))


def _translated_cells(mask, step, repeats):
    coords = _local_coords(mask)
    dr, dc = step
    cells = set()
    for k in range(1, repeats + 1):
        for r, c in coords:
            cells.add((r + k * dr, c + k * dc))
    return cells


def _ray_is_self_nonoverlapping(mask, step, repeats):
    coords = set(_local_coords(mask))
    seen = set(coords)
    dr, dc = step
    for k in range(1, repeats + 1):
        shifted = {(r + k * dr, c + k * dc) for r, c in coords}
        if shifted & seen:
            return False
        seen |= shifted
    return True


def _bbox_in_bounds(grid_shape, bbox, step, repeats):
    h = _bbox_h(bbox)
    w = _bbox_w(bbox)
    r0, c0, _, _ = bbox
    dr, dc = step
    rr = r0 + repeats * dr
    cc = c0 + repeats * dc
    return 0 <= rr and 0 <= cc and rr + h <= grid_shape[0] and cc + w <= grid_shape[1]


def _render(base_grid, center_bbox, center_mask, guides):
    out = np.array(base_grid, dtype=int, copy=True)
    patch_shape = center_mask.shape
    for guide in sorted(guides, key=lambda g: (g["color"], g["step"][0], g["step"][1])):
        color_patch = np.where(center_mask, int(guide["color"]), BG)
        dr, dc = guide["step"]
        for k in range(1, int(guide["count"]) + 1):
            _paste(out, color_patch, center_bbox[0] + k * dr, center_bbox[1] + k * dc, transparent=BG)
    grey_patch = np.where(center_mask, GREY, BG)
    _paste(out, grey_patch, center_bbox[0], center_bbox[1], transparent=BG)
    return out


def _parse_input(grid):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    colors = set(map(int, np.unique(g)))
    assert colors <= {BG, BLUE, RED, GREY}

    comps = _find_components(g, colors - {BG})
    center_candidates = [comp for comp in comps if comp["color"] == GREY]
    assert len(center_candidates) == 1
    center = center_candidates[0]
    center_bbox = center["bbox"]
    assert 4 <= center["area"] <= 7
    assert _bbox_h(center_bbox) in (3, 4) and _bbox_w(center_bbox) in (3, 4)
    assert _bbox_h(center_bbox) == _bbox_w(center_bbox)

    center_patch = _crop_bbox(g, center_bbox)
    center_mask = center_patch == GREY
    assert _connected(center_mask)

    guides = [comp for comp in comps if comp["color"] in (BLUE, RED)]
    assert 1 <= len(guides) <= 2
    assert len({guide["color"] for guide in guides}) == len(guides)

    center_anchor = _bbox_center(center_bbox)
    axis_dirs = set()
    parsed = []
    for guide in guides:
        assert 1 <= guide["area"] <= 3
        guide_anchor = _bbox_center(guide["bbox"])
        dr = int(guide_anchor[0] - center_anchor[0])
        dc = int(guide_anchor[1] - center_anchor[1])
        assert (dr == 0) ^ (dc == 0)
        axis_dirs.add((0 if dr == 0 else int(dr / abs(dr)), 0 if dc == 0 else int(dc / abs(dc))))

        for r, c in guide["coords"]:
            rr = r - center_bbox[0] - dr
            cc = c - center_bbox[1] - dc
            assert 0 <= rr < center_mask.shape[0] and 0 <= cc < center_mask.shape[1]
            assert center_mask[rr, cc]

        parsed.append(
            {
                "color": int(guide["color"]),
                "coords": guide["coords"],
                "bbox": guide["bbox"],
                "step": (dr, dc),
                "count": int(guide["area"]),
            }
        )

    assert len(axis_dirs) == len(parsed)
    return {
        "grid": g,
        "center_bbox": center_bbox,
        "center_mask": center_mask,
        "guides": parsed,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the central grey sprite repeatedly along each guide direction, using the guide size as the loop count.

    Input structure:
    - A black background contains one central grey sprite and one or two smaller non-grey guide objects.
    - The grey sprite is a 4-connected shape with a 3x3 or 4x4 bounding box.
    - Each guide is a blue or red partial copy of the grey sprite, placed purely above, below, left, or right of the center.

    Rule:
    - Crop the central grey sprite by its bounding box.
    - For each guide, compute the displacement vector from the center bounding-box center to the guide bounding-box center.
    - Let N be the guide's pixel count. Recolor the cropped central sprite to the guide color and stamp it at center + k*V for k = 1..N.
    - Paste only non-background pixels and keep the original central sprite grey.

    Key insight:
    - Each partial copy encodes two parameters at once: its location gives the stamp direction and step, while its pixel count gives how many translated copies to place.
    """
    parsed = _parse_input(grid)
    return _render(parsed["grid"], parsed["center_bbox"], parsed["center_mask"], parsed["guides"])


def _enumerate_center_masks(size):
    cells = [(r, c) for r in range(size) for c in range(size)]
    out = []
    for area in range(4, 8):
        for comb in itertools.combinations(range(size * size), area):
            mask = np.zeros((size, size), dtype=bool)
            for idx in comb:
                mask[cells[idx]] = True
            if not _connected(mask):
                continue
            if _bbox_from_coords(_local_coords(mask)) != (0, 0, size - 1, size - 1):
                continue
            out.append(mask)
    return out


def _guide_masks_for_center(center_mask):
    coords = _local_coords(center_mask)
    center_anchor = _bbox_center((0, 0, center_mask.shape[0] - 1, center_mask.shape[1] - 1))
    guide_masks = []
    for area in range(1, min(3, len(coords)) + 1):
        for subset in itertools.combinations(coords, area):
            mask = np.zeros_like(center_mask, dtype=bool)
            for r, c in subset:
                mask[r, c] = True
            if not _connected(mask):
                continue
            local_anchor = _bbox_center(_bbox_from_coords(subset))
            if local_anchor != center_anchor:
                continue
            guide_masks.append(mask)
    return guide_masks


def _build_catalog():
    catalog = []
    for size in (3, 4):
        for mask in _enumerate_center_masks(size):
            guides = _guide_masks_for_center(mask)
            big_guides = [g for g in guides if int(g.sum()) > 1]
            if not guides or not big_guides:
                continue
            catalog.append({"mask": mask, "guides": guides, "big_guides": big_guides})
    return catalog


CENTER_CATALOG = _build_catalog()


def _sample_center_top_left(height, width, size):
    ideal_r = (height - size) // 2
    ideal_c = (width - size) // 2
    rs = list(range(max(0, ideal_r - 2), min(height - size, ideal_r + 2) + 1))
    cs = list(range(max(0, ideal_c - 2), min(width - size, ideal_c + 2) + 1))
    random.shuffle(rs)
    random.shuffle(cs)
    return [(r, c) for r in rs for c in cs]


def _distance_candidates(center_bbox, center_mask, grid_shape, direction, count, min_step=1):
    out = []
    for dist in range(min_step, max(grid_shape) + 1):
        step = (direction[0] * dist, direction[1] * dist)
        if not _bbox_in_bounds(grid_shape, center_bbox, step, count):
            continue
        if not _ray_is_self_nonoverlapping(center_mask, step, count):
            continue
        out.append(dist)
    return out


def _footprints_disjoint(center_mask, guides):
    per_guide = []
    for guide in guides:
        coords = set()
        dr, dc = guide["step"]
        for k in range(1, guide["count"] + 1):
            for r, c in _local_coords(center_mask):
                coords.add((r + k * dr, c + k * dc))
        per_guide.append(coords)
    for i in range(len(per_guide)):
        for j in range(i + 1, len(per_guide)):
            if per_guide[i] & per_guide[j]:
                return False
    return True


def _build_input_grid(height, width, center_tl, center_mask, guides):
    grid = np.zeros((height, width), dtype=int)
    grey_patch = np.where(center_mask, GREY, BG)
    _paste(grid, grey_patch, center_tl[0], center_tl[1], transparent=BG)
    for guide in guides:
        guide_patch = np.where(guide["mask"], guide["color"], BG)
        dr, dc = guide["step"]
        _paste(grid, guide_patch, center_tl[0] + dr, center_tl[1] + dc, transparent=BG)
    return grid


def _guide_extra_possible(center_bbox, center_mask, grid_shape, guide):
    next_count = guide["count"] + 1
    return _bbox_in_bounds(grid_shape, center_bbox, guide["step"], next_count) and _ray_is_self_nonoverlapping(center_mask, guide["step"], next_count)


def _validate_instance(grid):
    parsed = _parse_input(grid)
    out = main(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid)

    guides = parsed["guides"]
    center_bbox = parsed["center_bbox"]
    center_mask = parsed["center_mask"]

    minus_ok = False
    plus_ok = False
    for idx, guide in enumerate(guides):
        if guide["count"] > 1:
            alt_guides = [dict(g) for g in guides]
            alt_guides[idx]["count"] -= 1
            alt = _render(parsed["grid"], center_bbox, center_mask, alt_guides)
            if not np.array_equal(alt, out):
                minus_ok = True
        if _guide_extra_possible(center_bbox, center_mask, grid.shape, guide):
            alt_guides = [dict(g) for g in guides]
            alt_guides[idx]["count"] += 1
            alt = _render(parsed["grid"], center_bbox, center_mask, alt_guides)
            if not np.array_equal(alt, out):
                plus_ok = True

    assert minus_ok
    assert plus_ok
    return True


def _gen_attempt():
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    n_guides = 2 if random.random() < 0.55 else 1
    entry = random.choice(CENTER_CATALOG)
    center_mask = entry["mask"]
    size = center_mask.shape[0]
    for center_tl in _sample_center_top_left(height, width, size):
        center_bbox = (center_tl[0], center_tl[1], center_tl[0] + size - 1, center_tl[1] + size - 1)
        directions = random.sample(DIRS4, n_guides)
        primary_idx = random.randrange(n_guides)

        for _ in range(80):
            guides = []
            ok = True
            for i, direction in enumerate(directions):
                mask_pool = entry["big_guides"] if i == primary_idx else entry["guides"]
                guide_mask = random.choice(mask_pool)
                count = int(guide_mask.sum())
                min_step = 2
                dists = _distance_candidates(center_bbox, center_mask, (height, width), direction, count, min_step=min_step)
                if i == primary_idx:
                    dists = [
                        dist
                        for dist in dists
                        if _bbox_in_bounds((height, width), center_bbox, (direction[0] * dist, direction[1] * dist), count + 1)
                        and _ray_is_self_nonoverlapping(center_mask, (direction[0] * dist, direction[1] * dist), count + 1)
                    ]
                if not dists:
                    ok = False
                    break
                dist = random.choice(dists)
                guides.append(
                    {
                        "mask": guide_mask,
                        "count": count,
                        "step": (direction[0] * dist, direction[1] * dist),
                    }
                )
            if not ok or not _footprints_disjoint(center_mask, guides):
                continue

            colors = GUIDE_COLORS[:]
            random.shuffle(colors)
            for guide, color in zip(guides, colors):
                guide["color"] = color

            grid = _build_input_grid(height, width, center_tl, center_mask, guides)
            try:
                _validate_instance(grid)
            except Exception:
                continue
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid displacement-loop-stamp input")
