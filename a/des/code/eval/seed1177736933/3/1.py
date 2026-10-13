from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
OBJECT_COLOR = 4
LEGEND_FRAME_COLOR = 5
LEGEND_COLOR_POOL = [1, 2, 3, 6, 7, 8, 9]
MAX_GEN_ATTEMPTS = 300

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

RNG = np.random.default_rng()

HOLE_LIBRARY = {
    "dot": np.array([[1]], dtype=bool),
    "domino": np.array([[1, 1]], dtype=bool),
    "line3": np.array([[1, 1, 1]], dtype=bool),
    "square2": np.array([[1, 1], [1, 1]], dtype=bool),
    "L3": np.array([[1, 0], [1, 1]], dtype=bool),
    "T4": np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
}


def _components(mask: np.ndarray, conn8: bool = False) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR8 if conn8 else DIR4
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    out[rel[:, 0], rel[:, 1]] = True
    return out


def _all_unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    base = _trim_mask(mask)
    seen = set()
    variants = []
    for flip in (False, True):
        flipped = np.fliplr(base) if flip else base
        for k in range(4):
            arr = np.rot90(flipped, k)
            key = (int(arr.shape[0]), int(arr.shape[1]), arr.astype(np.uint8).tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(arr.copy())
    return variants


def _canonical_shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    keys = []
    for arr in _all_unique_transforms(mask):
        keys.append((int(arr.shape[0]), int(arr.shape[1]), arr.astype(np.uint8).tobytes()))
    assert keys
    return min(keys)


def _exact_shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    arr = _trim_mask(mask)
    return int(arr.shape[0]), int(arr.shape[1]), arr.astype(np.uint8).tobytes()


def _single_hole_mask(obj_mask: np.ndarray) -> np.ndarray:
    bg_mask = ~obj_mask
    holes = []
    h, w = obj_mask.shape
    for comp in _components(bg_mask, conn8=False):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp):
            continue
        holes.append(comp)
    assert len(holes) == 1, "expected exactly one enclosed hole"
    return _mask_from_coords(holes[0])


def _legend_info(grid: np.ndarray) -> tuple[dict[tuple[int, int, bytes], int], tuple[int, int, int, int], np.ndarray]:
    assert int(grid[0, 0]) == LEGEND_FRAME_COLOR, "legend frame must occupy top-left corner"
    frame_mask = grid == LEGEND_FRAME_COLOR
    comp = None
    for coords in _components(frame_mask, conn8=False):
        if any(r == 0 and c == 0 for r, c in coords):
            comp = coords
            break
    assert comp is not None, "legend frame not found"
    bbox = _bbox(comp)
    r0, c0, r1, c1 = bbox
    legend_area = np.zeros_like(grid, dtype=bool)
    legend_area[r0 : r1 + 1, c0 : c1 + 1] = True

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    exemplars = _components((sub != BG) & (sub != LEGEND_FRAME_COLOR), conn8=False)
    mapping: dict[tuple[int, int, bytes], int] = {}
    for coords in exemplars:
        comp_mask = _mask_from_coords(coords)
        hole_mask = _single_hole_mask(comp_mask)
        key = _canonical_shape_key(hole_mask)
        color = int(sub[coords[0, 0], coords[0, 1]])
        assert key not in mapping, "legend hole shapes must be unique up to rotation/flip"
        mapping[key] = color
    assert mapping, "legend must contain at least one exemplar"
    return mapping, bbox, legend_area


def _outer_ring_mask(
    grid: np.ndarray,
    coords: np.ndarray,
    legend_area: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    h, w = grid.shape
    r0, c0, r1, c1 = _bbox(coords)
    er0 = max(0, r0 - 1)
    ec0 = max(0, c0 - 1)
    er1 = min(h - 1, r1 + 1)
    ec1 = min(w - 1, c1 + 1)

    local_obj = np.zeros((er1 - er0 + 1, ec1 - ec0 + 1), dtype=bool)
    rel = coords - np.array([er0, ec0], dtype=int)
    local_obj[rel[:, 0], rel[:, 1]] = True

    local_bg = (grid[er0 : er1 + 1, ec0 : ec1 + 1] == BG) & (~legend_area[er0 : er1 + 1, ec0 : ec1 + 1])

    outside = np.zeros_like(local_obj, dtype=bool)
    q = deque()
    lh, lw = local_obj.shape
    for rr in range(lh):
        for cc in range(lw):
            if rr not in (0, lh - 1) and cc not in (0, lw - 1):
                continue
            if local_bg[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))
    while q:
        rr, cc = q.popleft()
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < lh and 0 <= nc < lw and local_bg[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    ring = np.zeros_like(local_obj, dtype=bool)
    obj_pts = np.argwhere(local_obj)
    for rr, cc in obj_pts:
        for dr, dc in DIR8:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < lh and 0 <= nc < lw and local_bg[nr, nc]:
                ring[nr, nc] = True
    ring &= outside
    return ring, er0, ec0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each yellow object's hole against the legend's hole shape, then draw
    a one-pixel outer 8-neighborhood border in the matched color and erase the legend.
    """
    g = np.asarray(grid, dtype=int)
    mapping, legend_bbox, legend_area = _legend_info(g)
    r0, c0, r1, c1 = legend_bbox

    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = BG

    object_mask = (g == OBJECT_COLOR) & (~legend_area)
    objects = _components(object_mask, conn8=False)
    assert objects, "expected at least one yellow object"

    for coords in objects:
        local_mask = _mask_from_coords(coords)
        hole_mask = _single_hole_mask(local_mask)
        color = mapping[_canonical_shape_key(hole_mask)]
        ring, er0, ec0 = _outer_ring_mask(g, coords, legend_area)
        for rr, cc in np.argwhere(ring):
            gr, gc = er0 + int(rr), ec0 + int(cc)
            if out[gr, gc] == BG:
                out[gr, gc] = color

    return out


def _build_ring_patch(hole_mask: np.ndarray, min_pad: int, max_pad: int) -> np.ndarray:
    hole = _trim_mask(hole_mask)
    hh, hw = hole.shape
    top = int(RNG.integers(min_pad, max_pad + 1))
    bottom = int(RNG.integers(min_pad, max_pad + 1))
    left = int(RNG.integers(min_pad, max_pad + 1))
    right = int(RNG.integers(min_pad, max_pad + 1))
    patch = np.ones((hh + top + bottom, hw + left + right), dtype=bool)
    patch[top : top + hh, left : left + hw][hole] = False
    return patch


def _build_object_patch(hole_mask: np.ndarray) -> np.ndarray:
    base = _build_ring_patch(hole_mask, min_pad=1, max_pad=3)
    bh, bw = base.shape
    margin = 2
    canvas = np.zeros((bh + 2 * margin, bw + 2 * margin), dtype=bool)
    canvas[margin : margin + bh, margin : margin + bw] = base

    side_count = int(RNG.integers(0, 4))
    for side in RNG.permutation(4)[:side_count]:
        if side == 0:
            thick = int(RNG.integers(1, margin + 1))
            span = int(RNG.integers(max(1, bw // 2), bw + 1))
            start = int(RNG.integers(0, bw - span + 1))
            canvas[margin - thick : margin, margin + start : margin + start + span] = True
        elif side == 1:
            thick = int(RNG.integers(1, margin + 1))
            span = int(RNG.integers(max(1, bw // 2), bw + 1))
            start = int(RNG.integers(0, bw - span + 1))
            canvas[margin + bh : margin + bh + thick, margin + start : margin + start + span] = True
        elif side == 2:
            thick = int(RNG.integers(1, margin + 1))
            span = int(RNG.integers(max(1, bh // 2), bh + 1))
            start = int(RNG.integers(0, bh - span + 1))
            canvas[margin + start : margin + start + span, margin - thick : margin] = True
        else:
            thick = int(RNG.integers(1, margin + 1))
            span = int(RNG.integers(max(1, bh // 2), bh + 1))
            start = int(RNG.integers(0, bh - span + 1))
            canvas[margin + start : margin + start + span, margin + bw : margin + bw + thick] = True

    return _trim_mask(canvas)


def _legend_layout_spec(patches: list[np.ndarray], cols: int) -> tuple[int, int, list[int], list[int], int, int]:
    n = len(patches)
    rows = (n + cols - 1) // cols
    row_heights = [0] * rows
    col_widths = [0] * cols
    for idx, patch in enumerate(patches):
        rr = idx // cols
        cc = idx % cols
        row_heights[rr] = max(row_heights[rr], int(patch.shape[0]))
        col_widths[cc] = max(col_widths[cc], int(patch.shape[1]))
    total_h = sum(row_heights) + (rows + 1) + 2
    total_w = sum(col_widths) + (cols + 1) + 2
    return rows, cols, row_heights, col_widths, total_h, total_w


def _render_legend(
    hole_specs: list[dict[str, object]],
    colors: list[int],
) -> np.ndarray:
    patches = []
    for spec, color in zip(hole_specs, colors):
        patch = _build_ring_patch(spec["legend_hole"], min_pad=1, max_pad=2).astype(int) * color
        patches.append(patch)

    n = len(patches)
    candidate_cols = sorted(set([1, min(2, n), n]))
    specs = [_legend_layout_spec(patches, cols) for cols in candidate_cols]
    feasible = [spec for spec in specs if spec[4] <= 24 and spec[5] <= 24]
    if not feasible:
        feasible = [spec for spec in specs if spec[4] <= 28 and spec[5] <= 28]
    if not feasible:
        feasible = specs
    rows, cols, row_heights, col_widths, total_h, total_w = feasible[int(RNG.integers(0, len(feasible)))]

    inner_h = total_h - 2
    inner_w = total_w - 2
    legend = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    legend[0, :] = LEGEND_FRAME_COLOR
    legend[-1, :] = LEGEND_FRAME_COLOR
    legend[:, 0] = LEGEND_FRAME_COLOR
    legend[:, -1] = LEGEND_FRAME_COLOR

    row_y = 1
    for rr in range(rows):
        col_x = 1
        for cc in range(cols):
            idx = rr * cols + cc
            if idx >= n:
                break
            patch = patches[idx]
            cell_h = row_heights[rr]
            cell_w = col_widths[cc]
            off_r = int(RNG.integers(0, cell_h - patch.shape[0] + 1))
            off_c = int(RNG.integers(0, cell_w - patch.shape[1] + 1))
            pr = row_y + 1 + off_r
            pc = col_x + 1 + off_c
            mask = patch != BG
            legend[pr : pr + patch.shape[0], pc : pc + patch.shape[1]][mask] = patch[mask]
            col_x += cell_w + 1
        row_y += row_heights[rr] + 1

    return legend


def _can_place(
    occupied: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
    pad: int,
) -> bool:
    h, w = occupied.shape
    ph, pw = patch.shape
    coords = np.argwhere(patch)
    for rr, cc in coords:
        gr, gc = top + int(rr), left + int(cc)
        if gr < 0 or gc < 0 or gr >= h or gc >= w:
            return False
        if occupied[gr, gc]:
            return False
        r0 = max(0, gr - pad)
        c0 = max(0, gc - pad)
        r1 = min(h, gr + pad + 1)
        c1 = min(w, gc + pad + 1)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> None:
    mask = patch.astype(bool)
    grid[top : top + patch.shape[0], left : left + patch.shape[1]][mask] = color


def _transformed_family(name: str, exact_key_to_avoid: tuple[int, int, bytes] | None = None) -> np.ndarray:
    variants = _all_unique_transforms(HOLE_LIBRARY[name])
    if exact_key_to_avoid is not None:
        filtered = [v for v in variants if _exact_shape_key(v) != exact_key_to_avoid]
        if filtered:
            variants = filtered
    return variants[int(RNG.integers(0, len(variants)))]


def _gen_attempt() -> np.ndarray | None:
    legend_count = int(RNG.integers(3, 5))
    shape_names = list(HOLE_LIBRARY)
    chosen_names = list(RNG.choice(shape_names, size=legend_count, replace=False))
    if not any(len(_all_unique_transforms(HOLE_LIBRARY[name])) > 1 for name in chosen_names):
        chosen_names[0] = "L3"
        chosen_names = list(dict.fromkeys(chosen_names))
        while len(chosen_names) < legend_count:
            extra = str(RNG.choice(shape_names))
            if extra not in chosen_names:
                chosen_names.append(extra)

    legend_colors = list(RNG.choice(LEGEND_COLOR_POOL, size=len(chosen_names), replace=False))
    hole_specs = []
    for name in chosen_names:
        legend_hole = _transformed_family(name)
        hole_specs.append(
            {
                "name": name,
                "legend_hole": legend_hole,
                "legend_exact_key": _exact_shape_key(legend_hole),
            }
        )

    used_count = int(RNG.integers(2, min(len(chosen_names), 4) + 1))
    used_names = list(RNG.choice(chosen_names, size=used_count, replace=False))
    object_count = int(RNG.integers(max(2, used_count), 6))

    special_name = None
    transform_sensitive = [name for name in used_names if len(_all_unique_transforms(HOLE_LIBRARY[name])) > 1]
    if transform_sensitive:
        special_name = str(RNG.choice(transform_sensitive))

    object_names = used_names.copy()
    while len(object_names) < object_count:
        object_names.append(str(RNG.choice(used_names)))
    RNG.shuffle(object_names)

    spec_by_name = {spec["name"]: spec for spec in hole_specs}
    object_patches = []
    forced_diff_used = False
    for name in object_names:
        avoid = None
        if name == special_name and not forced_diff_used:
            avoid = spec_by_name[name]["legend_exact_key"]
            forced_diff_used = True
        hole = _transformed_family(name, exact_key_to_avoid=avoid)
        object_patches.append((name, _build_object_patch(hole)))

    legend = _render_legend(hole_specs, legend_colors)

    min_h = max(14, legend.shape[0] + 2)
    min_w = max(16, legend.shape[1] + 2)
    if min_h >= 31 or min_w >= 31:
        return None
    grid_h = int(RNG.integers(min_h, 31))
    grid_w = int(RNG.integers(min_w, 31))
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[: legend.shape[0], : legend.shape[1]] = legend

    occupied = np.zeros_like(grid, dtype=bool)
    occupied[: legend.shape[0], : legend.shape[1]] = True

    order = sorted(object_patches, key=lambda item: int(item[1].shape[0] * item[1].shape[1]), reverse=True)
    for _name, patch in order:
        ph, pw = patch.shape
        if ph + 2 >= grid_h or pw + 2 >= grid_w:
            return None
        placed = False
        for _ in range(120):
            top = int(RNG.integers(1, grid_h - ph - 1))
            left = int(RNG.integers(1, grid_w - pw - 1))
            if not _can_place(occupied, patch, top, left, pad=2):
                continue
            _place_patch(grid, patch, top, left, OBJECT_COLOR)
            coords = np.argwhere(patch)
            occupied[top + coords[:, 0], left + coords[:, 1]] = True
            placed = True
            break
        if not placed:
            return None

    if len(set(object_names)) < 2:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if not np.all(out[: legend.shape[0], : legend.shape[1]] == BG):
        return None
    if np.count_nonzero((out != grid) & (out != BG)) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
