import itertools
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_ANGLES = (0, 1, 2, 3)
MAX_GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box):
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _extract_quadrants(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h % 2 == 1 and w % 2 == 1, "separator row/col require odd grid sizes"
    mid_r = h // 2
    mid_c = w // 2
    sep = int(g[mid_r, mid_c])
    assert np.all(g[mid_r, :] == sep), "expected a full separator row"
    assert np.all(g[:, mid_c] == sep), "expected a full separator column"
    qh = mid_r
    qw = mid_c
    assert qh == qw, "quadrants must be square to support 90-degree rotations"
    quads = [
        g[:mid_r, :mid_c].copy(),
        g[:mid_r, mid_c + 1 :].copy(),
        g[mid_r + 1 :, :mid_c].copy(),
        g[mid_r + 1 :, mid_c + 1 :].copy(),
    ]
    return quads, sep


def _color_union(parts, bg: int):
    out = np.full_like(parts[0], bg)
    for part in parts:
        vis = part != bg
        conflict = vis & (out != bg) & (out != part)
        if np.any(conflict):
            return None
        out[vis] = part[vis]
    return out


def _has_nontrivial_rot_symmetry(grid: np.ndarray):
    return any(np.array_equal(grid, np.rot90(grid, k)) for k in (1, 2, 3))


def _enumerate_solutions(quadrants, bg: int):
    occupied = [idx for idx, quad in enumerate(quadrants) if np.any(quad != bg)]
    blank = [idx for idx, quad in enumerate(quadrants) if not np.any(quad != bg)]
    if len(occupied) != 3 or len(blank) != 1:
        return []

    valid = []
    for angles in itertools.permutations(ALL_ANGLES, 3):
        aligned = [np.rot90(quadrants[idx], -angle) for idx, angle in zip(occupied, angles)]
        base = _color_union(aligned, bg)
        if base is None:
            continue
        pair_ok = True
        for a, b in ((0, 1), (0, 2), (1, 2)):
            overlap = (aligned[a] != bg) & (aligned[b] != bg)
            overlap_colors = {int(x) for x in aligned[a][overlap]}
            if int(overlap.sum()) < 2 or len(overlap_colors) < 2:
                pair_ok = False
                break
        if not pair_ok:
            continue
        if _has_nontrivial_rot_symmetry(base):
            continue
        if any(np.array_equal(part, base) for part in aligned):
            continue
        missing_angle = next(angle for angle in ALL_ANGLES if angle not in angles)
        output = np.rot90(base, missing_angle)
        valid.append(
            {
                "occupied": tuple(occupied),
                "blank": blank[0],
                "angles": dict(zip(occupied, angles)),
                "missing_angle": missing_angle,
                "base": base,
                "output": output,
            }
        )
    return valid


def _unique_outputs(solutions):
    unique = {}
    for sol in solutions:
        out = sol["output"]
        key = (out.shape, out.tobytes())
        unique.setdefault(key, out)
    return list(unique.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full motif from three damaged rotated quadrant copies, then emit the missing rotation.

    Input structure:
    - One separator row and one separator column split the grid into four equal square quadrants.
    - Three quadrants contain damaged copies of the same multicolor motif under three different 90-degree rotations.
    - The three damage masks are complementary: every true motif pixel is visible in at least one copy, but no copy is complete.
    - The remaining quadrant is entirely background.

    Rule:
    - Read the empty quadrant's color as the background and isolate the three non-empty quadrants.
    - Try the four rotation labels on those three copies and keep the unique output whose inverse-rotated copies agree cellwise on all visible colors.
    - Union those aligned visible pixels to recover the complete motif, then rotate that motif by the one unused angle.

    Key insight:
    - The answer is not any visible quadrant: the full motif exists only after rotating the three damaged copies back into one common frame and taking their colorwise union.
    """
    quadrants, _sep = _extract_quadrants(np.asarray(grid, dtype=int))
    blank = [quad for quad in quadrants if np.all(quad == quad[0, 0])]
    assert len(blank) == 1, "expected exactly one blank quadrant"
    bg = int(blank[0][0, 0])

    solutions = _enumerate_solutions(quadrants, bg)
    outputs = _unique_outputs(solutions)
    assert len(outputs) == 1, "output must be uniquely determined"
    return outputs[0].copy()


def _grow_connected_mask(h: int, w: int, area: int):
    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True
    while int(mask.sum()) < area:
        frontier = []
        for r, c in np.argwhere(mask):
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True
    return mask


def _generate_base(quadrant_size: int, bg: int, body: int, markers, accent: int):
    min_side = 4
    for _ in range(200):
        h = random.randint(min_side, quadrant_size - 1)
        w = random.randint(min_side, quadrant_size - 1)
        top = random.randint(0, quadrant_size - h)
        left = random.randint(0, quadrant_size - w)
        area_min = max(14, int(round(h * w * 0.35)))
        area_max = min(h * w - 1, int(round(h * w * 0.65)))
        if area_min > area_max:
            continue
        area = random.randint(area_min, area_max)
        local_mask = _grow_connected_mask(h, w, area)
        if local_mask is None or int(local_mask.sum()) < 14:
            continue

        cells = [tuple(map(int, coord)) for coord in np.argwhere(local_mask)]
        if len(cells) < 14:
            continue

        quad = np.full((quadrant_size, quadrant_size), bg, dtype=int)
        sub = quad[top : top + h, left : left + w]
        sub[local_mask] = body

        marker_cells = random.sample(cells, 3)
        marker_positions = []
        for color, (r, c) in zip(markers, marker_cells):
            sub[r, c] = color
            marker_positions.append((top + r, left + c))

        remaining = [cell for cell in cells if cell not in marker_cells]
        if len(remaining) >= 3 and random.random() < 0.8:
            accent_count = random.randint(2, min(4, len(remaining)))
            for r, c in random.sample(remaining, accent_count):
                sub[r, c] = accent

        if _has_nontrivial_rot_symmetry(quad):
            continue
        if len(set(int(x) for x in quad.ravel() if int(x) != bg)) < 4:
            continue
        return quad, marker_positions
    return None, None


def _visibility_masks(base: np.ndarray, bg: int, marker_positions):
    fg_coords = [tuple(map(int, coord)) for coord in np.argwhere(base != bg)]
    if len(fg_coords) < 14:
        return None

    marker_positions = [tuple(pos) for pos in marker_positions]
    marker_set = set(marker_positions)
    other_coords = [coord for coord in fg_coords if coord not in marker_set]
    if len(other_coords) < 9:
        return None

    visible = [set() for _ in range(3)]
    marker_pairs = [(0, 1), (0, 2), (1, 2)]
    for coord, (a, b) in zip(marker_positions, marker_pairs):
        visible[a].add(coord)
        visible[b].add(coord)

    unique_witness = random.sample(other_coords, 3)
    unique_set = set(unique_witness)
    for idx, coord in enumerate(unique_witness):
        visible[idx].add(coord)

    pool = [coord for coord in other_coords if coord not in unique_set]
    shared_pair = random.sample(pool, 3)
    shared_set = set(shared_pair)
    for coord, (a, b) in zip(shared_pair, marker_pairs):
        visible[a].add(coord)
        visible[b].add(coord)

    pool = [coord for coord in pool if coord not in shared_set]
    miss_witness = random.sample(pool, 3)
    miss_set = set(miss_witness)
    for idx, coord in enumerate(miss_witness):
        for copy_idx in range(3):
            if copy_idx != idx:
                visible[copy_idx].add(coord)

    patterns = [
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1, 2),
    ]
    weights = [1, 1, 1, 4, 4, 4, 3]
    for coord in fg_coords:
        if coord in marker_set or coord in unique_set or coord in shared_set or coord in miss_set:
            continue
        choice = random.choices(patterns, weights=weights, k=1)[0]
        for idx in choice:
            visible[idx].add(coord)

    for idx in range(3):
        colors = {int(base[r, c]) for r, c in visible[idx]}
        if len(visible[idx]) < 6 or len(colors) < 2:
            return None
        if len(visible[idx]) >= len(fg_coords) - 1:
            return None

    if any(not set(fg_coords) - visible[idx] for idx in range(3)):
        return None
    if any((visible[a] | visible[b]) == set(fg_coords) for a, b in ((0, 1), (0, 2), (1, 2))):
        return None
    for a, b in ((0, 1), (0, 2), (1, 2)):
        overlap = visible[a] & visible[b]
        overlap_colors = {int(base[r, c]) for r, c in overlap}
        if len(overlap) < 2 or len(overlap_colors) < 2:
            return None

    masks = []
    for idx in range(3):
        mask = np.zeros_like(base, dtype=bool)
        for r, c in visible[idx]:
            mask[r, c] = True
        masks.append(mask)
    return masks


def _build_input(base: np.ndarray, bg: int, sep: int, blank_idx: int, used_angles, masks):
    copies = []
    for angle, mask in zip(used_angles, masks):
        damaged = np.full_like(base, bg)
        damaged[mask] = base[mask]
        copies.append(np.rot90(damaged, angle))

    order = list(range(3))
    random.shuffle(order)
    placed = [np.full_like(base, bg) for _ in range(4)]
    actual_angles = {}
    occupied_slots = [idx for idx in range(4) if idx != blank_idx]
    for slot, copy_idx in zip(occupied_slots, order):
        placed[slot] = copies[copy_idx]
        actual_angles[slot] = used_angles[copy_idx]

    q = base.shape[0]
    grid = np.full((2 * q + 1, 2 * q + 1), sep, dtype=int)
    grid[:q, :q] = placed[0]
    grid[:q, q + 1 :] = placed[1]
    grid[q + 1 :, :q] = placed[2]
    grid[q + 1 :, q + 1 :] = placed[3]
    return grid, actual_angles


def _validate_instance(grid: np.ndarray, expected: np.ndarray, base: np.ndarray, bg: int, masks):
    quadrants, _sep = _extract_quadrants(grid)
    sols = _enumerate_solutions(quadrants, bg)
    outputs = _unique_outputs(sols)
    if len(outputs) != 1:
        return False
    if not np.array_equal(outputs[0], expected):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    if any(np.array_equal(np.rot90(base, angle), expected) for angle in ALL_ANGLES if angle != sols[0]["missing_angle"]):
        return False

    full_mask = base != bg
    for mask in masks:
        if np.array_equal(mask, full_mask):
            return False
    for a, b in ((0, 1), (0, 2), (1, 2)):
        pair_union = masks[a] | masks[b]
        if np.array_equal(pair_union, full_mask):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        quadrant_size = random.randint(7, 13)
        palette = list(range(10))
        bg = random.choice(palette)
        palette.remove(bg)
        sep = random.choice(palette)
        palette.remove(sep)
        random.shuffle(palette)
        body, marker1, marker2, marker3, accent = palette[:5]

        base, marker_positions = _generate_base(
            quadrant_size,
            bg,
            body,
            (marker1, marker2, marker3),
            accent,
        )
        if base is None:
            continue

        masks = _visibility_masks(base, bg, marker_positions)
        if masks is None:
            continue

        used_angles = random.sample(list(ALL_ANGLES), 3)
        missing_angle = next(angle for angle in ALL_ANGLES if angle not in used_angles)
        blank_idx = random.randrange(4)
        expected = np.rot90(base, missing_angle)
        grid, _actual_angles = _build_input(base, bg, sep, blank_idx, used_angles, masks)

        if _validate_instance(grid, expected, base, bg, masks):
            return grid

    raise RuntimeError("failed to generate a valid input for puzzle 6")
