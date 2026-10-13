import random
from collections import deque

import numpy as np


BG = 0
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_mask(coords: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    if shape is None:
        r0, c0, r1, c1 = _bbox(coords)
        shape = (r1 - r0 + 1, c1 - c0 + 1)
        base_r, base_c = r0, c0
    else:
        base_r, base_c = 0, 0
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0] - base_r, coords[:, 1] - base_c] = True
    return mask


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _trim_patch(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BG)
    if len(coords) == 0:
        return np.zeros((0, 0), dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    trimmed = _trim_mask(mask)
    return trimmed.shape, tuple(int(x) for x in trimmed.ravel())


def _rotation_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return min(_mask_key(np.rot90(mask, k)) for k in range(4))


def _has_hole(mask: np.ndarray) -> bool:
    bg = ~mask
    for comp in _components(bg):
        r0, c0, r1, c1 = _bbox(comp)
        if r0 > 0 and c0 > 0 and r1 < mask.shape[0] - 1 and c1 < mask.shape[1] - 1:
            return True
    return False


def _orientation_specific(mask: np.ndarray) -> bool:
    base = _mask_key(mask)
    for k in range(1, 4):
        if _mask_key(np.rot90(mask, k)) == base:
            return False
    return True


def _find_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]]:
    h, w = grid.shape
    candidates = []
    for coords in _components(grid != BG):
        r0, c0, r1, c1 = _bbox(coords)
        if (r1 - r0 + 1, c1 - c0 + 1) != (4, 2):
            continue
        if len(coords) != 8:
            continue
        if not (r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == BG):
            continue
        candidates.append(((r0, c0, r1, c1), patch.copy()))
    assert len(candidates) == 1, "expected exactly one 4x2 edge legend"
    bbox, patch = candidates[0]
    mapping = [(int(patch[r, 0]), int(patch[r, 1])) for r in range(4)]
    for src, dst in mapping:
        assert src != BG and dst != BG and src != dst
    return bbox, mapping


def _component_info(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "patch": patch,
        "mask": mask,
        "area": len(coords),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reinsert scattered fragments into the central missing holes, using rotation-dependent recoloring from an edge legend.

    Input structure:
    - Background is black.
    - A large connected multi-color main pattern sits near the center, with one or more enclosed black holes where pieces were removed.
    - Outside the main pattern are several smaller multi-color fragment components whose silhouettes match the holes under some rotation.
    - One edge contains a full 4x2 color legend: each row stores an input color in the left column and its replacement color in the right column for rotations 0°, 90°, 180°, 270° from top to bottom.

    Rule:
    - Identify the edge legend, the largest remaining foreground component as the main pattern, and every enclosed black component inside it as a hole.
    - For each scattered fragment, try the four quarter-turn rotations and find the unique hole whose shape matches the rotated fragment silhouette.
    - Once the matching rotation is known, recolor only the pixels equal to that rotation row's legend input color into the row's output color.
    - Paste the rotated, recolored fragment into the matching hole and erase the fragment's original location to black.

    Key insight:
    - The hole contour determines which rotation is correct, and that rotation index is then reused to choose the only color substitution that should be applied before the piece is restored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend_bbox, mapping = _find_legend(g)
    r0, c0, r1, c1 = legend_bbox
    work = g.copy()
    work[r0 : r1 + 1, c0 : c1 + 1] = BG

    comps = [_component_info(work, coords) for coords in _components(work != BG)]
    assert len(comps) >= 2, "expected a main pattern plus at least one fragment"
    main_comp = max(comps, key=lambda info: info["area"])
    fragments = [info for info in comps if info is not main_comp]
    assert fragments

    main_mask = np.zeros_like(g, dtype=bool)
    main_coords = main_comp["coords"]
    main_mask[main_coords[:, 0], main_coords[:, 1]] = True

    holes = []
    for bg_coords in _components(work == BG):
        rr0, cc0, rr1, cc1 = _bbox(bg_coords)
        if rr0 == 0 or cc0 == 0 or rr1 == g.shape[0] - 1 or cc1 == g.shape[1] - 1:
            continue
        comp_mask = np.zeros_like(g, dtype=bool)
        comp_mask[bg_coords[:, 0], bg_coords[:, 1]] = True
        ok = True
        touched_main = False
        for br, bc in bg_coords:
            for dr, dc in DIR4:
                nr, nc = int(br + dr), int(bc + dc)
                if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                    ok = False
                    break
                if comp_mask[nr, nc]:
                    continue
                if g[nr, nc] != BG:
                    if not main_mask[nr, nc]:
                        ok = False
                        break
                    touched_main = True
            if not ok:
                break
        if not ok or not touched_main:
            continue
        holes.append(
            {
                "coords": bg_coords,
                "bbox": (rr0, cc0, rr1, cc1),
                "mask": _coords_mask(bg_coords),
            }
        )

    assert len(holes) == len(fragments), "fragment and hole counts must match"

    assignments = []
    used_holes: set[int] = set()
    for frag in fragments:
        matches = []
        for hole_idx, hole in enumerate(holes):
            for k in range(4):
                rotated = _trim_patch(np.rot90(frag["patch"], k))
                if rotated.shape != hole["mask"].shape:
                    continue
                if np.array_equal(rotated != BG, hole["mask"]):
                    matches.append((hole_idx, k, rotated))
        assert len(matches) == 1, "each fragment must match exactly one hole+rotation"
        hole_idx, k, rotated = matches[0]
        assert hole_idx not in used_holes, "holes must be used exactly once"
        used_holes.add(hole_idx)
        assignments.append((frag, hole_idx, k, rotated))
    assert len(used_holes) == len(holes)

    out = g.copy()
    for frag in fragments:
        coords = frag["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    for frag, hole_idx, k, rotated in assignments:
        src, dst = mapping[k]
        placed = rotated.copy()
        placed[placed == src] = dst
        hole = holes[hole_idx]
        rr0, cc0, rr1, cc1 = hole["bbox"]
        region = out[rr0 : rr1 + 1, cc0 : cc1 + 1]
        assert np.all(region[hole["mask"]] == BG)
        assert np.all(placed[hole["mask"]] != BG)
        region[hole["mask"]] = placed[hole["mask"]]

    return out


# --------------- generator ---------------

GRID_MIN = 22
GRID_MAX = 30
MAIN_H_RANGE = (10, 16)
MAIN_W_RANGE = (10, 16)
HOLE_COUNT_RANGE = (2, 4)
HOLE_SIZE_RANGE = (2, 5)
HOLE_AREA_RANGE = (4, 8)
GEN_RETRIES = 500
FILL_MODES = ("rows", "cols", "checker", "diagonal")


def _ensure_colors(arr: np.ndarray, palette: list[int]) -> None:
    cells = random.sample(range(arr.size), len(palette))
    for idx, color in zip(cells, palette):
        r, c = divmod(idx, arr.shape[1])
        arr[r, c] = color


def _make_main_fill(h: int, w: int, palette: list[int]) -> np.ndarray:
    mode = random.choice(FILL_MODES)
    arr = np.zeros((h, w), dtype=int)
    if mode == "rows":
        for r in range(h):
            arr[r, :] = random.choice(palette)
    elif mode == "cols":
        for c in range(w):
            arr[:, c] = random.choice(palette)
    elif mode == "checker":
        step = random.randint(1, 3)
        rows, cols = np.indices((h, w))
        pal = np.array(palette, dtype=int)
        arr[:, :] = pal[((rows // step) + (cols // step)) % len(palette)]
    else:
        step = random.randint(1, 3)
        offset = random.randrange(len(palette))
        rows, cols = np.indices((h, w))
        pal = np.array(palette, dtype=int)
        arr[:, :] = pal[((rows + cols) // step + offset) % len(palette)]
    _ensure_colors(arr, palette)
    return arr


def _random_piece_mask() -> np.ndarray | None:
    h = random.randint(*HOLE_SIZE_RANGE)
    w = random.randint(*HOLE_SIZE_RANGE)
    max_area = min(HOLE_AREA_RANGE[1], h * w - 1)
    min_area = max(HOLE_AREA_RANGE[0], 4)
    if max_area < min_area:
        return None
    target = random.randint(min_area, max_area)
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))

    add_frontier(*start)
    while int(mask.sum()) < target and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        add_frontier(*cell)

    if int(mask.sum()) != target:
        return None
    mask = _trim_mask(mask)
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        return None
    if _has_hole(mask):
        return None
    if not _orientation_specific(mask):
        return None
    return mask


def _place_holes(main_h: int, main_w: int, masks: list[np.ndarray]) -> list[tuple[int, int]] | None:
    occupied = np.zeros((main_h, main_w), dtype=bool)
    placements = []
    for mask in masks:
        ph, pw = mask.shape
        candidates = []
        for r0 in range(1, main_h - ph):
            for c0 in range(1, main_w - pw):
                rr0 = max(0, r0 - 1)
                rr1 = min(main_h, r0 + ph + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(main_w, c0 + pw + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements.append((r0, c0))
        occupied[max(0, r0 - 1) : min(main_h, r0 + ph + 1), max(0, c0 - 1) : min(main_w, c0 + pw + 1)] = True
    return placements


def _make_piece_patch(mask: np.ndarray, dst: int, filler_pool: list[int]) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    dst_count = random.randint(1, len(coords) - 1)
    dst_cells = set(random.sample(coords, dst_count))
    filler_count = random.randint(1, min(2, len(filler_pool)))
    fillers = random.sample(filler_pool, filler_count)
    mode = random.choice(("parity", "row", "random"))

    patch = np.zeros(mask.shape, dtype=int)
    for r, c in coords:
        if (r, c) in dst_cells:
            patch[r, c] = dst
            continue
        if mode == "parity":
            patch[r, c] = fillers[(r + c) % len(fillers)]
        elif mode == "row":
            patch[r, c] = fillers[r % len(fillers)]
        else:
            patch[r, c] = random.choice(fillers)
    assert len(set(int(x) for x in patch[mask])) >= 2
    return patch


def _can_place_patch(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = occupied.shape
    ph, pw = patch.shape
    if r0 <= 0 or c0 <= 0 or r0 + ph >= h or c0 + pw >= w:
        return False
    rr0 = max(0, r0 - 1)
    rr1 = min(h, r0 + ph + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(w, c0 + pw + 1)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _scatter_fragments(
    shape: tuple[int, int],
    main_bbox: tuple[int, int, int, int],
    legend_bbox: tuple[int, int, int, int],
    fragments: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    mr0, mc0, mr1, mc1 = main_bbox
    lr0, lc0, lr1, lc1 = legend_bbox
    occupied[max(0, mr0 - 1) : min(h, mr1 + 2), max(0, mc0 - 1) : min(w, mc1 + 2)] = True
    occupied[max(0, lr0 - 1) : min(h, lr1 + 2), max(0, lc0 - 1) : min(w, lc1 + 2)] = True

    positions = []
    for patch in fragments:
        ph, pw = patch.shape
        candidates = []
        for r0 in range(1, h - ph):
            for c0 in range(1, w - pw):
                if _can_place_patch(occupied, patch, r0, c0):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        positions.append((r0, c0))
        occupied[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True
    return positions


def _gen_attempt() -> np.ndarray | None:
    colors = list(range(1, 10))
    random.shuffle(colors)
    src_colors = colors[:4]
    dst_colors = colors[4:8]
    accent = colors[8]
    main_palette = [accent] + random.sample(dst_colors, random.randint(2, 4))

    hole_count = random.randint(*HOLE_COUNT_RANGE)
    rotations = random.sample(range(4), hole_count)

    main_h = random.randint(*MAIN_H_RANGE)
    main_w = random.randint(*MAIN_W_RANGE)
    grid_h_min = max(GRID_MIN, main_h + 8)
    grid_w_min = max(GRID_MIN, main_w + 10)
    if grid_h_min > GRID_MAX or grid_w_min > GRID_MAX:
        return None
    grid_h = random.randint(grid_h_min, GRID_MAX)
    grid_w = random.randint(grid_w_min, GRID_MAX)

    legend_side = random.choice(("left", "right"))
    if legend_side == "left":
        main_c_min, main_c_max = 4, grid_w - main_w - 2
    else:
        main_c_min, main_c_max = 1, grid_w - main_w - 5
    main_r_min, main_r_max = 2, grid_h - main_h - 2
    if main_c_min > main_c_max or main_r_min > main_r_max:
        return None

    main_r0 = random.randint(main_r_min, main_r_max)
    main_c0 = random.randint(main_c_min, main_c_max)
    main_r1 = main_r0 + main_h - 1
    main_c1 = main_c0 + main_w - 1
    main_bbox = (main_r0, main_c0, main_r1, main_c1)

    legend_r0 = random.randint(0, grid_h - 4)
    legend_c0 = 0 if legend_side == "left" else grid_w - 2
    legend_bbox = (legend_r0, legend_c0, legend_r0 + 3, legend_c0 + 1)
    legend = np.array([[src_colors[i], dst_colors[i]] for i in range(4)], dtype=int)

    masks = []
    seen_shapes = set()
    for _ in range(hole_count):
        for _ in range(80):
            mask = _random_piece_mask()
            if mask is None:
                continue
            sig = _rotation_signature(mask)
            if sig in seen_shapes:
                continue
            seen_shapes.add(sig)
            masks.append(mask)
            break
        else:
            return None

    hole_positions = _place_holes(main_h, main_w, masks)
    if hole_positions is None:
        return None

    solved = np.zeros((grid_h, grid_w), dtype=int)
    solved[legend_r0 : legend_r0 + 4, legend_c0 : legend_c0 + 2] = legend
    main_fill = _make_main_fill(main_h, main_w, main_palette)
    solved[main_r0 : main_r1 + 1, main_c0 : main_c1 + 1] = main_fill

    input_grid = solved.copy()
    fragment_patches = []
    for mask, (local_r0, local_c0), rot in zip(masks, hole_positions, rotations):
        dst = dst_colors[rot]
        src = src_colors[rot]
        filler_pool = [color for color in main_palette if color != dst]
        if not filler_pool:
            return None
        piece_patch = _make_piece_patch(mask, dst, filler_pool)

        rr0 = main_r0 + local_r0
        cc0 = main_c0 + local_c0
        sub = solved[rr0 : rr0 + mask.shape[0], cc0 : cc0 + mask.shape[1]]
        sub[mask] = piece_patch[mask]

        input_grid[rr0 : rr0 + mask.shape[0], cc0 : cc0 + mask.shape[1]][mask] = BG

        fragment = piece_patch.copy()
        fragment[fragment == dst] = src
        fragment = _trim_patch(np.rot90(fragment, (4 - rot) % 4))
        fragment_patches.append(fragment)

    fragment_positions = _scatter_fragments((grid_h, grid_w), main_bbox, legend_bbox, fragment_patches)
    if fragment_positions is None:
        return None

    for patch, (r0, c0) in zip(fragment_patches, fragment_positions):
        target = input_grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        if np.any((target != BG) & (patch != BG)):
            return None
        target[patch != BG] = patch[patch != BG]

    try:
        if np.array_equal(main(input_grid), solved):
            return input_grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid ARC instance")
