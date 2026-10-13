from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FRAME_COLOR = 5
PALETTE = tuple(color for color in range(1, 10) if color != FRAME_COLOR)
MARKER_DIMS = ((1, 2), (2, 1))
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    shifted = coords - np.array([r0, c0])
    h, w = shifted.max(axis=0) + 1
    mask = np.zeros((int(h), int(w)), dtype=bool)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _find_components(grid: np.ndarray, *, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _rect_frame_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    box = _bbox(mask)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    expected = np.zeros_like(mask, dtype=bool)
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    return box if np.array_equal(mask, expected) else None


def _scaled_index(i: int, src_len: int, dst_len: int) -> int:
    if src_len == 1:
        return 0
    return int(round(i * (dst_len - 1) / (src_len - 1)))


def _draw_hv_line(canvas: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        canvas[r0, lo : hi + 1] = True
    elif c0 == c1:
        lo, hi = sorted((r0, r1))
        canvas[lo : hi + 1, c0] = True
    else:
        raise AssertionError("stretched adjacency must stay horizontal or vertical")


def _fill_enclosed(stroke: np.ndarray) -> np.ndarray:
    h, w = stroke.shape
    open_space = ~stroke.astype(bool)
    padded = np.pad(open_space, 1, constant_values=True)
    seen = np.zeros_like(padded, dtype=bool)
    q = deque()
    ph, pw = padded.shape
    for r in range(ph):
        for c in (0, pw - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(pw):
        for r in (0, ph - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, ph, pw):
            if padded[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    holes = open_space & ~seen[1 : h + 1, 1 : w + 1]
    return stroke.astype(bool) | holes


def _stretch_mask(mask: np.ndarray, target_shape: tuple[int, int], *, return_stroke: bool = False):
    src_h, src_w = mask.shape
    dst_h, dst_w = target_shape
    stroke = np.zeros((dst_h, dst_w), dtype=bool)
    coords = np.argwhere(mask)
    mapped: dict[tuple[int, int], tuple[int, int]] = {}
    for sr, sc in coords:
        tr = _scaled_index(int(sr), src_h, dst_h)
        tc = _scaled_index(int(sc), src_w, dst_w)
        mapped[(int(sr), int(sc))] = (tr, tc)
        stroke[tr, tc] = True
    for sr, sc in coords:
        sr = int(sr)
        sc = int(sc)
        if (sr + 1, sc) in mapped:
            _draw_hv_line(stroke, mapped[(sr, sc)], mapped[(sr + 1, sc)])
        if (sr, sc + 1) in mapped:
            _draw_hv_line(stroke, mapped[(sr, sc)], mapped[(sr, sc + 1)])
    filled = _fill_enclosed(stroke)
    return (filled, stroke) if return_stroke else filled


def _mask_symmetry(mask: np.ndarray) -> bool:
    variants = [
        np.fliplr(mask),
        np.flipud(mask),
        np.rot90(mask, 1),
        np.rot90(mask, 2),
        np.rot90(mask, 3),
    ]
    return any(mask.shape == other.shape and np.array_equal(mask, other) for other in variants)


def _is_marker_component(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return int(comp["area"]) == 2 and (h, w) in MARKER_DIMS


def _component_mid2(comp: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    return int(r0 + r1), int(c0 + c1)


def _identify_cluster(markers: list[dict], center: dict) -> bool:
    if len(markers) != 4:
        return False
    r0, c0, r1, c1 = center["bbox"]
    roles = {}
    for marker in markers:
        mr0, mc0, mr1, mc1 = marker["bbox"]
        if mr1 < r0 and mc1 < c0:
            role = "NW"
        elif mr1 < r0 and mc0 > c1:
            role = "NE"
        elif mr0 > r1 and mc1 < c0:
            role = "SW"
        elif mr0 > r1 and mc0 > c1:
            role = "SE"
        else:
            return False
        if role in roles:
            return False
        roles[role] = marker
    if set(roles) != {"NW", "NE", "SW", "SE"}:
        return False
    cr2, cc2 = _component_mid2(center)
    nw_r2, nw_c2 = _component_mid2(roles["NW"])
    ne_r2, ne_c2 = _component_mid2(roles["NE"])
    sw_r2, sw_c2 = _component_mid2(roles["SW"])
    se_r2, se_c2 = _component_mid2(roles["SE"])
    return (
        nw_r2 < cr2 < sw_r2
        and ne_r2 < cr2 < se_r2
        and nw_c2 < cc2 < ne_c2
        and sw_c2 < cc2 < se_c2
    )


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    frame_mask = g == FRAME_COLOR
    frame_box = _rect_frame_bbox(frame_mask)
    assert frame_box is not None, "expected one hollow gray frame"
    comps = [comp for comp in _find_components(g, bg=BG) if comp["color"] != FRAME_COLOR]
    by_color: dict[int, list[dict]] = {}
    for comp in comps:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    clusters = []
    for marker_color, same_color in by_color.items():
        markers = [comp for comp in same_color if _is_marker_component(comp)]
        if len(markers) != 4:
            continue
        marker_ids = {id(marker) for marker in markers}
        centers = []
        for center in comps:
            if id(center) in marker_ids:
                continue
            if _is_marker_component(center):
                continue
            if _identify_cluster(markers, center):
                centers.append(center)
        assert len(centers) == 1, "each marker-color set must enclose one unique center"
        clusters.append({"marker_color": marker_color, "markers": markers, "center": centers[0]})

    assert len(clusters) == 3, "expected exactly three clusters"
    targets = [cluster for cluster in clusters if cluster["center"]["color"] == cluster["marker_color"]]
    assert len(targets) == 1, "expected exactly one self-colored cluster"
    return {"frame_bbox": frame_box, "clusters": clusters, "target": targets[0]}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the self-colored marker cluster, crop its center sprite, and stretch it into the gray frame.

    Input structure:
    - The background is black.
    - Three separated clusters each contain four same-color 1x2 or 2x1 corner markers
      around one small monochrome 4-connected center sprite.
    - A separate gray one-pixel-thick hollow rectangle appears elsewhere on the grid.

    Rule:
    - Find the unique cluster whose center sprite has the same color as its four corner markers.
    - Crop that center sprite to its bounding box.
    - Stretch the cropped sprite into the frame interior by mapping bbox corners to
      interior corners, connecting mapped 4-neighbor pairs with horizontal or vertical
      line segments, and filling enclosed background regions.
    - Erase all three original clusters and keep only the gray frame plus the stretched result.

    Key insight:
    - The decisive cue is not size or position but the self-color match between a cluster's
      center sprite and its four markers; only that chosen sprite should be stretched.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_grid(g)
    target = info["target"]
    tr0, tc0, tr1, tc1 = target["center"]["bbox"]
    center_color = int(target["center"]["color"])
    sprite = g[tr0 : tr1 + 1, tc0 : tc1 + 1] == center_color

    fr0, fc0, fr1, fc1 = info["frame_bbox"]
    ir0, ic0 = fr0 + 1, fc0 + 1
    ir1, ic1 = fr1 - 1, fc1 - 1
    assert ir0 <= ir1 and ic0 <= ic1, "frame must have a non-empty interior"
    stretched = _stretch_mask(sprite, (ir1 - ir0 + 1, ic1 - ic0 + 1))

    out = g.copy()
    out[(out != BG) & (out != FRAME_COLOR)] = BG
    out[ir0 : ir1 + 1, ic0 : ic1 + 1][stretched] = center_color
    return out


def _enumerate_center_masks() -> list[np.ndarray]:
    masks = []
    for h in range(2, 5):
        for w in range(2, 5):
            area = h * w
            for bits in range(1 << area):
                pop = bits.bit_count()
                if pop < 3 or pop > 6:
                    continue
                mask = np.zeros((h, w), dtype=bool)
                for idx in range(area):
                    if (bits >> idx) & 1:
                        mask[idx // w, idx % w] = True
                if not _connected4(mask):
                    continue
                box = _bbox(mask)
                if box != (0, 0, h - 1, w - 1):
                    continue
                masks.append(mask)
    return masks


CENTER_MASKS = _enumerate_center_masks()
ASYMMETRIC_MASKS = [mask for mask in CENTER_MASKS if not _mask_symmetry(mask)]


def _place_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int = BG) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent == BG:
        mask = patch != transparent
        view[mask] = patch[mask]
    else:
        view[:, :] = patch


def _make_cluster_patch(center_mask: np.ndarray, marker_color: int, center_color: int) -> np.ndarray:
    ch, cw = center_mask.shape
    marker_dims = [random.choice(MARKER_DIMS) for _ in range(4)]
    top_gap = random.randint(1, 2)
    bottom_gap = random.randint(1, 2)
    left_gap = random.randint(1, 2)
    right_gap = random.randint(1, 2)
    top_h = max(marker_dims[0][0], marker_dims[1][0])
    bot_h = max(marker_dims[2][0], marker_dims[3][0])
    left_w = max(marker_dims[0][1], marker_dims[2][1])
    right_w = max(marker_dims[1][1], marker_dims[3][1])

    center_r0 = top_h + top_gap
    center_c0 = left_w + left_gap
    h = center_r0 + ch + bottom_gap + bot_h
    w = center_c0 + cw + right_gap + right_w
    patch = np.full((h, w), BG, dtype=int)

    positions = [
        (center_r0 - top_gap - marker_dims[0][0], center_c0 - left_gap - marker_dims[0][1]),
        (center_r0 - top_gap - marker_dims[1][0], center_c0 + cw + right_gap),
        (center_r0 + ch + bottom_gap, center_c0 - left_gap - marker_dims[2][1]),
        (center_r0 + ch + bottom_gap, center_c0 + cw + right_gap),
    ]
    for (mh, mw), (mr, mc) in zip(marker_dims, positions):
        patch[mr : mr + mh, mc : mc + mw] = marker_color

    patch[center_r0 : center_r0 + ch, center_c0 : center_c0 + cw][center_mask] = center_color
    return patch


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR


def _random_layout(shapes: list[tuple[int, int]], grid_shape: tuple[int, int], border: int = 1):
    h, w = grid_shape
    occupied = np.zeros(grid_shape, dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(shapes)
    order = list(range(len(shapes)))
    random.shuffle(order)
    order.sort(key=lambda idx: shapes[idx][0] * shapes[idx][1], reverse=True)
    for idx in order:
        ph, pw = shapes[idx]
        candidates = []
        for r in range(h - ph + 1):
            for c in range(w - pw + 1):
                rr0 = max(0, r - border)
                cc0 = max(0, c - border)
                rr1 = min(h, r + ph + border)
                cc1 = min(w, c + pw + border)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((r, c))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements[idx] = (r0, c0)
        occupied[r0 : r0 + ph, c0 : c0 + pw] = True
    return placements


def _choose_masks(interior_shape: tuple[int, int]):
    want_fill = bool(random.getrandbits(1))
    target_options = []
    for mask in ASYMMETRIC_MASKS:
        filled, stroke = _stretch_mask(mask, interior_shape, return_stroke=True)
        if bool(filled.sum() > stroke.sum()) == want_fill:
            target_options.append(mask)
    if not target_options:
        return None

    for _ in range(200):
        target = random.choice(target_options)
        distractors = random.sample(CENTER_MASKS, 2)
        masks = [target, *distractors]
        areas = [int(mask.sum()) for mask in masks]
        if areas[0] == max(areas) and areas.count(max(areas)) == 1:
            continue
        if areas[0] == min(areas) and areas.count(min(areas)) == 1:
            continue
        stretched = [_stretch_mask(mask, interior_shape) for mask in masks]
        if len({st.tobytes() + str(st.shape).encode() for st in stretched}) != 3:
            continue
        if len({mask.tobytes() + str(mask.shape).encode() for mask in masks}) != 3:
            continue
        return masks
    return None


def _validate_generated(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        info = _analyze_grid(grid)
        out = main(grid)
    except Exception:
        return False
    if not np.array_equal(out, expected):
        return False

    fr0, fc0, fr1, fc1 = info["frame_bbox"]
    inside = np.zeros_like(grid, dtype=bool)
    inside[fr0 + 1 : fr1, fc0 + 1 : fc1] = True
    if np.any((out != BG) & (out != FRAME_COLOR) & ~inside):
        return False
    if not np.array_equal(out[grid == FRAME_COLOR], np.full(np.count_nonzero(grid == FRAME_COLOR), FRAME_COLOR)):
        return False

    target = info["target"]
    color = int(target["center"]["color"])
    tr0, tc0, tr1, tc1 = target["center"]["bbox"]
    sprite = grid[tr0 : tr1 + 1, tc0 : tc1 + 1] == color
    expected_stretch = _stretch_mask(sprite, (fr1 - fr0 - 1, fc1 - fc0 - 1))
    if not np.array_equal(out[fr0 + 1 : fr1, fc0 + 1 : fc1] == color, expected_stretch):
        return False

    cluster_masks = []
    for cluster in info["clusters"]:
        cr0, cc0, cr1, cc1 = cluster["center"]["bbox"]
        cluster_masks.append(grid[cr0 : cr1 + 1, cc0 : cc1 + 1] == cluster["center"]["color"])
    distractor_stretches = [
        _stretch_mask(mask, (fr1 - fr0 - 1, fc1 - fc0 - 1))
        for cluster, mask in zip(info["clusters"], cluster_masks)
        if cluster is not target
    ]
    if any(np.array_equal(expected_stretch, other) for other in distractor_stretches):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    inner_h = random.randint(5, 8)
    inner_w = random.randint(5, 8)
    masks = _choose_masks((inner_h, inner_w))
    if masks is None:
        return None

    target_idx = random.randrange(3)
    ordered_masks = [None, None, None]
    ordered_masks[target_idx] = masks[0]
    other_masks = iter(masks[1:])
    for idx in range(3):
        if ordered_masks[idx] is None:
            ordered_masks[idx] = next(other_masks)

    marker_colors = random.sample(PALETTE, 3)
    remaining = [color for color in PALETTE if color not in marker_colors]
    distractor_centers = random.sample(remaining, 2)
    center_colors = []
    extra_iter = iter(distractor_centers)
    for idx, marker_color in enumerate(marker_colors):
        center_colors.append(marker_color if idx == target_idx else next(extra_iter))

    cluster_patches = [
        _make_cluster_patch(mask, marker_colors[idx], center_colors[idx])
        for idx, mask in enumerate(ordered_masks)
    ]
    frame_shape = (inner_h + 2, inner_w + 2)
    shapes = [patch.shape for patch in cluster_patches] + [frame_shape]

    max_h = max(shape[0] for shape in shapes)
    max_w = max(shape[1] for shape in shapes)
    grid_h = random.randint(max(17, max_h + 2), min(26, max_h + 12))
    grid_w = random.randint(max(17, max_w + 2), min(26, max_w + 12))
    placements = _random_layout(shapes, (grid_h, grid_w), border=1)
    if placements is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for patch, pos in zip(cluster_patches, placements[:3]):
        assert pos is not None
        _place_patch(grid, patch, pos[0], pos[1])

    frame_pos = placements[3]
    assert frame_pos is not None
    fr0, fc0 = frame_pos
    _draw_frame(grid, (fr0, fc0, fr0 + frame_shape[0] - 1, fc0 + frame_shape[1] - 1))

    try:
        expected = main(grid)
    except Exception:
        return None
    return grid if _validate_generated(grid, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid self-color stretch input")
