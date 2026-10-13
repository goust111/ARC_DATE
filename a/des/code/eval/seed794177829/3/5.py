import random
from collections import Counter

import numpy as np


BG = 0
MAROON = 9
SHAPE_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
SIDES = ["left", "right", "top", "bottom"]


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in dirs:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def round_half_away(x: float) -> int:
    if x >= 0:
        return int(np.floor(x + 0.5))
    return int(np.ceil(x - 0.5))


def boundary_from_filled(filled: np.ndarray) -> np.ndarray:
    h, w = filled.shape
    out = np.zeros_like(filled, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not filled[r, c]:
                continue
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w or not filled[nr, nc]:
                    out[r, c] = True
                    break
    return out


def mask_to_coords(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    return [(int(top + r), int(left + c)) for r, c in pts]


def detect_maroon_rectangles(grid: np.ndarray, maroon: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    comps = components_mask(grid == maroon, connectivity=4)
    solid_bbox = None
    frame_bbox = None
    for comp in comps:
        r0, c0, r1, c1 = bbox_coords(comp)
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        area = len(comp)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if area == bh * bw and np.all(sub == maroon):
            solid_bbox = (r0, c0, r1, c1)
            continue
        if bh >= 3 and bw >= 3:
            border = np.zeros((bh, bw), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if area == int(border.sum()) and np.all(sub[border] == maroon) and np.all(sub[~border] != maroon):
                frame_bbox = (r0, c0, r1, c1)
    assert solid_bbox is not None, "missing solid maroon mask"
    assert frame_bbox is not None, "missing hollow maroon target frame"
    return solid_bbox, frame_bbox


def color_components(grid: np.ndarray, colors: list[int], connectivity: int = 8) -> list[tuple[int, np.ndarray]]:
    comps: list[tuple[int, np.ndarray]] = []
    for color in colors:
        for comp in components_mask(grid == color, connectivity=connectivity):
            comps.append((color, comp))
    return comps


def component_touch_side(coords: np.ndarray, mask: np.ndarray) -> str:
    counts = Counter()
    h, w = mask.shape
    for r, c in coords:
        for name, dr, dc in (
            ("top", -1, 0),
            ("bottom", 1, 0),
            ("left", 0, -1),
            ("right", 0, 1),
        ):
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                counts[name] += 1
    assert counts, "masked shape must touch the mask"
    return counts.most_common(1)[0][0]


def reconstruct_component(coords: np.ndarray, side: str) -> np.ndarray:
    pts = {(int(r), int(c)) for r, c in coords}
    if side == "right":
        axis = max(c for _, c in pts)
        for r, c in list(pts):
            pts.add((r, 2 * axis - c))
    elif side == "left":
        axis = min(c for _, c in pts)
        for r, c in list(pts):
            pts.add((r, 2 * axis - c))
    elif side == "bottom":
        axis = max(r for r, _ in pts)
        for r, c in list(pts):
            pts.add((2 * axis - r, c))
    else:
        axis = min(r for r, _ in pts)
        for r, c in list(pts):
            pts.add((2 * axis - r, c))
    arr = np.array(sorted(pts), dtype=int)
    return arr


def centroid_of_components(parts: list[tuple[int, np.ndarray]]) -> tuple[float, float]:
    all_pts = np.concatenate([coords for _, coords in parts], axis=0)
    return float(np.mean(all_pts[:, 0])), float(np.mean(all_pts[:, 1]))


def translated_components(
    parts: list[tuple[int, np.ndarray]], target_bbox: tuple[int, int, int, int]
) -> tuple[list[tuple[int, np.ndarray]], int, int]:
    cr, cc = centroid_of_components(parts)
    tr0, tc0, tr1, tc1 = target_bbox
    target_center_r = (tr0 + tr1) / 2.0
    target_center_c = (tc0 + tc1) / 2.0
    dr = round_half_away(target_center_r - cr)
    dc = round_half_away(target_center_c - cc)
    moved: list[tuple[int, np.ndarray]] = []
    for color, coords in parts:
        shifted = coords + np.array([[dr, dc]], dtype=int)
        moved.append((color, shifted))
    return moved, dr, dc


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solid_bbox, frame_bbox = detect_maroon_rectangles(g, MAROON)
    sr0, sc0, sr1, sc1 = solid_bbox
    solid_mask = np.zeros_like(g, dtype=bool)
    solid_mask[sr0 : sr1 + 1, sc0 : sc1 + 1] = True

    non_special = [int(c) for c in np.unique(g) if int(c) not in (bg, MAROON)]
    comps = color_components(g, non_special, connectivity=8)

    occluded_visible: list[tuple[int, np.ndarray]] = []
    reconstructed: list[tuple[int, np.ndarray]] = []
    for color, comp in comps:
        touches = False
        for r, c in comp:
            for dr, dc in DIR4:
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and solid_mask[nr, nc]:
                    touches = True
                    break
            if touches:
                break
        if not touches:
            continue
        side = component_touch_side(comp, solid_mask)
        occluded_visible.append((color, comp))
        reconstructed.append((color, reconstruct_component(comp, side)))

    moved, _, _ = translated_components(reconstructed, frame_bbox)

    out = g.copy()
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = bg
    fr0, fc0, fr1, fc1 = frame_bbox
    out[fr0 : fr1 + 1, fc0 : fc1 + 1][0, :] = bg
    out[fr0 : fr1 + 1, fc0 : fc1 + 1][-1, :] = bg
    out[fr0 : fr1 + 1, fc0 : fc1 + 1][:, 0] = bg
    out[fr0 : fr1 + 1, fc0 : fc1 + 1][:, -1] = bg
    for _, comp in occluded_visible:
        out[comp[:, 0], comp[:, 1]] = bg

    for color, coords in moved:
        for r, c in coords:
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = color
    return out


def make_symmetric_outline(rng: random.Random) -> np.ndarray:
    for _ in range(100):
        h = rng.randrange(5, 12, 2)
        w = rng.randrange(5, 12, 2)
        ry = max(1.0, h // 2)
        rx = max(1.0, w // 2)
        yy, xx = np.mgrid[0:h, 0:w]
        dy = np.abs(yy - h // 2) / ry
        dx = np.abs(xx - w // 2) / rx
        family = rng.choice(["rectangle", "diamond", "ellipse", "octagon"])
        if family == "rectangle":
            filled = np.ones((h, w), dtype=bool)
        elif family == "diamond":
            filled = (dy + dx) <= 1.0 + 1e-9
        elif family == "ellipse":
            filled = (dy * dy + dx * dx) <= 1.0 + 1e-9
        else:
            major = np.maximum(dy, dx)
            minor = np.minimum(dy, dx)
            filled = (major + 0.45 * minor) <= 1.0 + 1e-9
        if not np.any(filled):
            continue
        outline = boundary_from_filled(filled)
        if outline.sum() < 8:
            continue
        if len(components_mask(outline, connectivity=8)) != 1:
            continue
        return outline
    raise RuntimeError("failed to build a symmetric outline")


def place_occ_shapes(
    rng: random.Random, side: str, masks: list[np.ndarray]
) -> tuple[list[list[tuple[int, int]]], tuple[int, int, int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    if side in ("left", "right"):
        radii = [(mask.shape[1] - 1) // 2 for mask in masks]
        common_axis = max(radii)
        rows = []
        top = 0
        for idx, mask in enumerate(masks):
            rx = (mask.shape[1] - 1) // 2
            left = common_axis - rx
            rows.append((top, left))
            top += mask.shape[0]
            if idx + 1 < len(masks):
                top += rng.randint(2, 4)
        group_h = top
        group_w = 2 * common_axis + 1
        mask_span = common_axis + rng.randint(1, 3)
        ext_top = rng.randint(0, 2)
        ext_bottom = rng.randint(0, 2)
        cluster_h = group_h + ext_top + ext_bottom
        cluster_w = common_axis + 1 + mask_span

        target_inner_h = group_h + rng.randint(4, 7)
        target_inner_w = group_w + rng.randint(4, 7)
        target_outer_h = target_inner_h + 2
        target_outer_w = target_inner_w + 2
        gap = rng.randint(3, 6)
        margin = rng.randint(1, 3)

        grid_h = max(cluster_h, target_outer_h) + 2 * margin + rng.randint(0, 6)
        grid_w = cluster_w + target_outer_w + gap + 2 * margin + rng.randint(0, 4)

        cluster_top = margin + rng.randint(0, grid_h - cluster_h - margin)
        target_top = margin + rng.randint(0, grid_h - target_outer_h - margin)

        if side == "right":
            cluster_left = margin + rng.randint(0, 2)
            target_left = cluster_left + cluster_w + gap + rng.randint(0, 2)
            group_top = cluster_top + ext_top
            group_left = cluster_left
            axis = group_left + common_axis
            mask_bbox = (
                cluster_top,
                axis + 1,
                cluster_top + cluster_h - 1,
                axis + mask_span,
            )
        else:
            target_left = margin + rng.randint(0, 2)
            cluster_left = target_left + target_outer_w + gap + rng.randint(0, 2)
            group_top = cluster_top + ext_top
            group_left = cluster_left + mask_span - common_axis
            axis = group_left + common_axis
            mask_bbox = (
                cluster_top,
                axis - mask_span,
                cluster_top + cluster_h - 1,
                axis - 1,
            )

        cluster_right = max(group_left + group_w - 1, mask_bbox[3])
        cluster_bottom = max(group_top + group_h - 1, mask_bbox[2])
        grid_h = max(grid_h, cluster_bottom + margin + 1, target_top + target_outer_h + margin)
        grid_w = max(grid_w, cluster_right + margin + 1, target_left + target_outer_w + margin)
        shape_coords = []
        for (top0, left0), mask in zip(rows, masks):
            shape_coords.append(mask_to_coords(mask, group_top + top0, group_left + left0))
        return shape_coords, mask_bbox, (grid_h, grid_w), (target_inner_h, target_inner_w), (group_h, group_w)

    radii = [(mask.shape[0] - 1) // 2 for mask in masks]
    common_axis = max(radii)
    cols = []
    left = 0
    for idx, mask in enumerate(masks):
        ry = (mask.shape[0] - 1) // 2
        top = common_axis - ry
        cols.append((top, left))
        left += mask.shape[1]
        if idx + 1 < len(masks):
            left += rng.randint(2, 4)
    group_h = 2 * common_axis + 1
    group_w = left
    mask_span = common_axis + rng.randint(1, 3)
    ext_left = rng.randint(0, 2)
    ext_right = rng.randint(0, 2)
    cluster_h = common_axis + 1 + mask_span
    cluster_w = group_w + ext_left + ext_right

    target_inner_h = group_h + rng.randint(4, 7)
    target_inner_w = group_w + rng.randint(4, 7)
    target_outer_h = target_inner_h + 2
    target_outer_w = target_inner_w + 2
    gap = rng.randint(3, 6)
    margin = rng.randint(1, 3)

    grid_h = cluster_h + target_outer_h + gap + 2 * margin + rng.randint(0, 4)
    grid_w = max(cluster_w, target_outer_w) + 2 * margin + rng.randint(0, 6)

    cluster_left = margin + rng.randint(0, grid_w - cluster_w - margin)
    target_left = margin + rng.randint(0, grid_w - target_outer_w - margin)

    if side == "bottom":
        cluster_top = margin + rng.randint(0, 2)
        target_top = cluster_top + cluster_h + gap + rng.randint(0, 2)
        group_top = cluster_top
        group_left = cluster_left + ext_left
        axis = group_top + common_axis
        mask_bbox = (
            axis + 1,
            cluster_left,
            axis + mask_span,
            cluster_left + cluster_w - 1,
        )
    else:
        target_top = margin + rng.randint(0, 2)
        cluster_top = target_top + target_outer_h + gap + rng.randint(0, 2)
        group_top = cluster_top + mask_span - common_axis
        group_left = cluster_left + ext_left
        axis = group_top + common_axis
        mask_bbox = (
            axis - mask_span,
            cluster_left,
            axis - 1,
            cluster_left + cluster_w - 1,
        )

    cluster_right = max(group_left + group_w - 1, mask_bbox[3])
    cluster_bottom = max(group_top + group_h - 1, mask_bbox[2])
    grid_h = max(grid_h, cluster_bottom + margin + 1, target_top + target_outer_h + margin)
    grid_w = max(grid_w, cluster_right + margin + 1, target_left + target_outer_w + margin)
    shape_coords = []
    for (top0, left0), mask in zip(cols, masks):
        shape_coords.append(mask_to_coords(mask, group_top + top0, group_left + left0))
    return shape_coords, mask_bbox, (grid_h, grid_w), (target_inner_h, target_inner_w), (group_h, group_w)


def bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def all_points(parts: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    return [pt for part in parts for pt in part]


def rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def frame_border_points(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        pts.append((r, c1))
    return pts


def build_visibility_mask(shape_points: list[tuple[int, int]], mask_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = mask_bbox
    return [(r, c) for r, c in shape_points if not (r0 <= r <= r1 and c0 <= c <= c1)]


def parts_to_solver_format(parts: list[tuple[int, list[tuple[int, int]]]]) -> list[tuple[int, np.ndarray]]:
    return [(color, np.array(sorted(coords), dtype=int)) for color, coords in parts]


def can_place_outline(
    shape: np.ndarray,
    top: int,
    left: int,
    occupied: np.ndarray,
    blocked_rects: list[tuple[int, int, int, int]],
    pad: int = 1,
) -> bool:
    h, w = occupied.shape
    sh, sw = shape.shape
    if top < 0 or left < 0 or top + sh > h or left + sw > w:
        return False
    pts = mask_to_coords(shape, top, left)
    bbox = (top, left, top + sh - 1, left + sw - 1)
    for rect in blocked_rects:
        if rect_overlap(bbox, rect, pad=pad):
            return False
    for r, c in pts:
        rr0 = max(0, r - pad)
        rr1 = min(h - 1, r + pad)
        cc0 = max(0, c - pad)
        cc1 = min(w - 1, c + pad)
        if np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1]):
            return False
    return True


def paint_points(grid: np.ndarray, points: list[tuple[int, int]], color: int) -> None:
    for r, c in points:
        grid[r, c] = color


def add_free_shapes(
    rng: random.Random,
    grid: np.ndarray,
    occupied: np.ndarray,
    colors: list[int],
    blocked_rects: list[tuple[int, int, int, int]],
) -> None:
    n_free = rng.randint(0, min(2, len(colors)))
    chosen_colors = colors[:n_free]
    for color in chosen_colors:
        placed = False
        for _ in range(120):
            shape = make_symmetric_outline(rng)
            sh, sw = shape.shape
            top = rng.randint(0, grid.shape[0] - sh)
            left = rng.randint(0, grid.shape[1] - sw)
            if not can_place_outline(shape, top, left, occupied, blocked_rects, pad=1):
                continue
            pts = mask_to_coords(shape, top, left)
            paint_points(grid, pts, color)
            for r, c in pts:
                occupied[r, c] = True
            blocked_rects.append((top, left, top + sh - 1, left + sw - 1))
            placed = True
            break
        if not placed:
            return


def gen_attempt(rng: random.Random) -> np.ndarray | None:
    side = rng.choice(SIDES)
    n_occ = rng.randint(1, 3)
    occ_masks = [make_symmetric_outline(rng) for _ in range(n_occ)]
    occ_colors = rng.sample(SHAPE_COLORS, n_occ)

    shape_coords, mask_bbox, grid_shape, _, _ = place_occ_shapes(rng, side, occ_masks)
    h, w = grid_shape
    grid = np.full((h, w), BG, dtype=int)

    mr0, mc0, mr1, mc1 = mask_bbox
    if mr0 < 0 or mc0 < 0 or mr1 >= h or mc1 >= w:
        return None

    visible_parts: list[tuple[int, list[tuple[int, int]]]] = []
    full_parts: list[tuple[int, list[tuple[int, int]]]] = []
    source_bboxes = []
    for color, pts in zip(occ_colors, shape_coords):
        if any(r < 0 or r >= h or c < 0 or c >= w for r, c in pts):
            return None
        full_parts.append((color, pts))
        source_bboxes.append(bbox_from_points(pts))
        visible = build_visibility_mask(pts, mask_bbox)
        if not visible:
            return None
        visible_parts.append((color, visible))

    for bbox in source_bboxes:
        if rect_overlap(bbox, mask_bbox, pad=0) is False:
            return None

    all_visible = all_points([pts for _, pts in visible_parts])
    if not all_visible:
        return None
    src_group_bbox = bbox_from_points(all_points([pts for _, pts in full_parts]))

    group_for_target = parts_to_solver_format(full_parts)
    cr, cc = centroid_of_components(group_for_target)
    group_h = src_group_bbox[2] - src_group_bbox[0] + 1
    group_w = src_group_bbox[3] - src_group_bbox[1] + 1
    inner_h = group_h + rng.randint(4, 7)
    inner_w = group_w + rng.randint(4, 7)

    if side in ("left", "right"):
        gap = rng.randint(3, 6)
        if side == "right":
            frame_c0 = mc1 + gap + rng.randint(0, 2)
        else:
            frame_c0 = max(1, mc0 - gap - (inner_w + 2) - rng.randint(0, 2))
        max_c0 = w - inner_w - 2
        if frame_c0 > max_c0:
            return None
        frame_r0 = rng.randint(1, max(1, h - inner_h - 2))
    else:
        gap = rng.randint(3, 6)
        if side == "bottom":
            frame_r0 = mr1 + gap + rng.randint(0, 2)
        else:
            frame_r0 = max(1, mr0 - gap - (inner_h + 2) - rng.randint(0, 2))
        max_r0 = h - inner_h - 2
        if frame_r0 > max_r0:
            return None
        frame_c0 = rng.randint(1, max(1, w - inner_w - 2))

    frame_bbox = (frame_r0, frame_c0, frame_r0 + inner_h + 1, frame_c0 + inner_w + 1)
    if frame_bbox[2] >= h or frame_bbox[3] >= w:
        return None
    if rect_overlap(frame_bbox, mask_bbox, pad=1):
        return None
    for bbox in source_bboxes:
        if rect_overlap(frame_bbox, bbox, pad=1):
            return None

    translated, _, _ = translated_components(group_for_target, frame_bbox)
    ir0, ic0, ir1, ic1 = frame_r0 + 1, frame_c0 + 1, frame_bbox[2] - 1, frame_bbox[3] - 1
    translated_sets = []
    for _, coords in translated:
        r0, c0, r1, c1 = bbox_coords(coords)
        if r0 < ir0 or c0 < ic0 or r1 > ir1 or c1 > ic1:
            return None
        translated_sets.append({(int(r), int(c)) for r, c in coords})
    for i in range(len(translated_sets)):
        for j in range(i + 1, len(translated_sets)):
            if translated_sets[i] & translated_sets[j]:
                return None

    occupied = np.zeros_like(grid, dtype=bool)
    paint_points(grid, frame_border_points(frame_bbox), MAROON)
    occupied[frame_r0 : frame_bbox[2] + 1, frame_c0 : frame_bbox[3] + 1] = True

    for color, pts in full_parts:
        paint_points(grid, pts, color)
        for r, c in pts:
            occupied[r, c] = True

    grid[mr0 : mr1 + 1, mc0 : mc1 + 1] = MAROON
    occupied[mr0 : mr1 + 1, mc0 : mc1 + 1] = True

    blocked_rects = [mask_bbox, frame_bbox, src_group_bbox]
    remaining_colors = [c for c in SHAPE_COLORS if c not in occ_colors]
    rng.shuffle(remaining_colors)
    add_free_shapes(rng, grid, occupied, remaining_colors, blocked_rects)

    if get_bg(grid) != BG:
        return None
    if len(components_mask(grid == MAROON, connectivity=4)) != 2:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
