from __future__ import annotations

import random

import numpy as np
from scipy.ndimage import binary_dilation, label


BG = 0
GREY = 5
STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
STRUCT8 = np.ones((3, 3), dtype=int)
KEY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400
PAIR_RETRIES = 200
PLACEMENT_RETRIES = 300


def _label(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    struct = STRUCT4 if int(connectivity) == 4 else STRUCT8
    lab, n = label(mask.astype(bool), structure=struct)
    return lab, int(n)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    lab, n = _label(mask, connectivity=connectivity)
    return [np.argwhere(lab == idx) for idx in range(1, n + 1)]


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _shape_key(mask: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    origin = coords.min(axis=0)
    norm = coords - origin
    height = int(norm[:, 0].max()) + 1
    width = int(norm[:, 1].max()) + 1
    cells = tuple(sorted((int(r), int(c)) for r, c in norm))
    return height, width, cells


def _mask_from_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if cells:
        rows, cols = zip(*cells)
        mask[list(rows), list(cols)] = True
    return mask


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    lab, n = _label(grid == int(color), connectivity=4)
    comps = []
    for idx in range(1, n + 1):
        coords = np.argwhere(lab == idx)
        if coords.size == 0:
            continue
        comps.append({"coords": coords, "bbox": _bbox_from_coords(coords)})
    return comps


def _crop_component_mask(comp: dict) -> np.ndarray:
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _single_hole_mask(obj_mask: np.ndarray) -> np.ndarray | None:
    bg_mask = ~obj_mask
    lab, n = _label(bg_mask, connectivity=4)
    holes = []
    for idx in range(1, n + 1):
        comp = lab == idx
        if comp[0, :].any() or comp[-1, :].any() or comp[:, 0].any() or comp[:, -1].any():
            continue
        holes.append(comp)
    if len(holes) != 1:
        return None
    return holes[0]


def _left_normal(dr: int, dc: int) -> tuple[int, int]:
    return -dc, dr


def _wall_sets(path: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    left: set[tuple[int, int]] = set()
    right: set[tuple[int, int]] = set()
    channel = set(path)
    for i, (r, c) in enumerate(path):
        dirs = []
        if i > 0:
            pr, pc = path[i - 1]
            dirs.append((r - pr, c - pc))
        if i + 1 < len(path):
            nr, nc = path[i + 1]
            dirs.append((nr - r, nc - c))
        for dr, dc in dirs:
            lr, lc = _left_normal(dr, dc)
            if (r + lr, c + lc) not in channel:
                left.add((r + lr, c + lc))
            if (r - lr, c - lc) not in channel:
                right.add((r - lr, c - lc))
        if len(dirs) == 2 and dirs[0] != dirs[1]:
            for side in (1, -1):
                offsets = []
                for dr, dc in dirs:
                    lr, lc = _left_normal(dr, dc)
                    offsets.append((side * lr, side * lc))
                direct = [(r + dr2, c + dc2) for dr2, dc2 in offsets]
                if all(cell not in channel for cell in direct):
                    er = r + offsets[0][0] + offsets[1][0]
                    ec = c + offsets[0][1] + offsets[1][1]
                    if side == 1:
                        left.add((er, ec))
                    else:
                        right.add((er, ec))
    return left, right


def _normalize_sets(*cell_sets: set[tuple[int, int]]) -> tuple[list[set[tuple[int, int]]], tuple[int, int]]:
    all_cells = set().union(*cell_sets)
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    shifted = [{(r - min_r, c - min_c) for r, c in cells} for cells in cell_sets]
    max_r = max(r for cells in shifted for r, _ in cells)
    max_c = max(c for cells in shifted for _, c in cells)
    return shifted, (max_r + 1, max_c + 1)


def _random_path() -> list[tuple[int, int]]:
    while True:
        h = random.randint(1, 4)
        w = random.randint(1, 4)
        if h == 1 and w == 1:
            continue
        if h + w - 1 < 3:
            continue
        moves = [(1, 0)] * (h - 1) + [(0, 1)] * (w - 1)
        random.shuffle(moves)
        coords = [(0, 0)]
        r = c = 0
        for dr, dc in moves:
            r += dr
            c += dc
            coords.append((r, c))

        arr = np.asarray(coords, dtype=int)
        if random.random() < 0.5:
            arr[:, 0] = h - 1 - arr[:, 0]
        if random.random() < 0.5:
            arr[:, 1] = w - 1 - arr[:, 1]
        if random.random() < 0.5:
            arr = arr[:, ::-1]
        return [(int(rr), int(cc)) for rr, cc in arr]


def _generate_pair_patch() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for _ in range(PAIR_RETRIES):
        path = _random_path()
        left_cells, right_cells = _wall_sets(path)
        if not left_cells or not right_cells:
            continue
        (path_cells, left_cells, right_cells), shape = _normalize_sets(set(path), left_cells, right_cells)
        left_mask = _mask_from_cells(left_cells, shape)
        right_mask = _mask_from_cells(right_cells, shape)
        if len(_components(left_mask, connectivity=4)) != 1:
            continue
        if len(_components(right_mask, connectivity=4)) != 1:
            continue

        padded_shape = (shape[0] + 2, shape[1] + 2)
        left_full = np.zeros(padded_shape, dtype=bool)
        right_full = np.zeros(padded_shape, dtype=bool)
        left_full[1:-1, 1:-1] = left_mask
        right_full[1:-1, 1:-1] = right_mask
        patch = np.zeros(padded_shape, dtype=int)
        patch[left_full | right_full] = GREY
        channel = (
            binary_dilation(left_full, structure=STRUCT8)
            & binary_dilation(right_full, structure=STRUCT8)
            & (patch == BG)
        )
        if len(_components(channel, connectivity=4)) != 1:
            continue
        channel_shape = _trim_mask(channel)
        if _shape_key(channel_shape) == _shape_key(_mask_from_cells(path_cells, shape)):
            continue
        return patch, channel, channel_shape
    raise RuntimeError("failed to generate a valid grey-object pair")


def _generate_hole_mask(used_keys: set[tuple[int, int, tuple[tuple[int, int], ...]]]) -> np.ndarray:
    for _ in range(PAIR_RETRIES):
        path = _random_path()
        coords = np.asarray(path, dtype=int)
        coords -= coords.min(axis=0)
        shape = (int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1)
        hole = np.zeros(shape, dtype=bool)
        hole[coords[:, 0], coords[:, 1]] = True
        key = _shape_key(hole)
        if key not in used_keys:
            return hole
    raise RuntimeError("failed to generate a distinct hole shape")


def _build_key_patch(hole_mask: np.ndarray, color: int) -> np.ndarray:
    hh, hw = hole_mask.shape
    pad_top = random.randint(1, 2)
    pad_bottom = random.randint(1, 2)
    pad_left = random.randint(1, 2)
    pad_right = random.randint(1, 2)

    base_h = hh + pad_top + pad_bottom
    base_w = hw + pad_left + pad_right
    extra_top = random.randint(0, 2)
    extra_bottom = random.randint(0, 2)
    extra_left = random.randint(0, 2)
    extra_right = random.randint(0, 2)

    out_h = base_h + extra_top + extra_bottom
    out_w = base_w + extra_left + extra_right
    mask = np.zeros((out_h, out_w), dtype=bool)
    base_r0 = extra_top
    base_c0 = extra_left
    mask[base_r0 : base_r0 + base_h, base_c0 : base_c0 + base_w] = True

    for _ in range(random.randint(1, 3)):
        side = random.choice("tblr")
        if side == "t" and extra_top > 0:
            width = random.randint(max(1, base_w // 3), base_w)
            c0 = base_c0 + random.randint(0, base_w - width)
            depth = random.randint(1, extra_top)
            mask[base_r0 - depth : base_r0, c0 : c0 + width] = True
        elif side == "b" and extra_bottom > 0:
            width = random.randint(max(1, base_w // 3), base_w)
            c0 = base_c0 + random.randint(0, base_w - width)
            depth = random.randint(1, extra_bottom)
            mask[base_r0 + base_h : base_r0 + base_h + depth, c0 : c0 + width] = True
        elif side == "l" and extra_left > 0:
            height = random.randint(max(1, base_h // 3), base_h)
            r0 = base_r0 + random.randint(0, base_h - height)
            depth = random.randint(1, extra_left)
            mask[r0 : r0 + height, base_c0 - depth : base_c0] = True
        elif side == "r" and extra_right > 0:
            height = random.randint(max(1, base_h // 3), base_h)
            r0 = base_r0 + random.randint(0, base_h - height)
            depth = random.randint(1, extra_right)
            mask[r0 : r0 + height, base_c0 + base_w : base_c0 + base_w + depth] = True

    hole_r0 = base_r0 + pad_top
    hole_c0 = base_c0 + pad_left
    mask[hole_r0 : hole_r0 + hh, hole_c0 : hole_c0 + hw][hole_mask] = False

    patch = np.zeros((out_h, out_w), dtype=int)
    patch[mask] = int(color)

    hole = _single_hole_mask(mask)
    assert hole is not None and _shape_key(hole) == _shape_key(hole_mask)
    return patch


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    return _bbox_from_coords(coords)


def _boxes_far(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != BG
    h, w = patch.shape
    view = grid[top : top + h, left : left + w]
    view[mask] = patch[mask]


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    placed_boxes: list[tuple[int, int, int, int]],
    *,
    pad: int,
) -> tuple[int, int] | None:
    h, w = patch.shape
    gh, gw = grid.shape
    if h > gh or w > gw:
        return None
    fg_box = _bbox_from_mask(patch != BG)
    fr0, fc0, fr1, fc1 = fg_box
    candidates = [(r, c) for r in range(gh - h + 1) for c in range(gw - w + 1)]
    random.shuffle(candidates)
    for top, left in candidates[:PLACEMENT_RETRIES]:
        box = (top + fr0, left + fc0, top + fr1, left + fc1)
        if all(_boxes_far(box, other, pad) for other in placed_boxes):
            _paste_patch(grid, patch, top, left)
            placed_boxes.append(box)
            return top, left
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey gap channel using the key whose enclosed hole has the same shape.

    Input structure:
    - Black background contains several grey components arranged in isolated pairs with a one-cell-wide background channel between the members of each pair.
    - Separate monochromatic colored key blocks appear elsewhere; each key is a connected solid block with exactly one enclosed black hole.
    - The outer silhouettes of the keys are arbitrary distractors, while the hole shapes are the only matching cue.

    Rule:
    - For every pair of grey components, dilate each component by one step in the 8-neighborhood and intersect the two dilations on background cells; that intersection is the channel shape between the pair.
    - For every colored key, extract its enclosed hole shape. If a key hole is translationally identical to a channel, paint that entire channel with the key color.
    - Erase all key blocks in the output, keep the grey components unchanged, and leave unmatched channels black.

    Key insight:
    - Both matched shapes are negative spaces: the channel is the overlap of two grey halos, and the key is identified by its internal hole rather than by its outer contour.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    grey_components = _find_color_components(g, GREY)
    grey_masks = []
    for comp in grey_components:
        mask = np.zeros_like(g, dtype=bool)
        coords = comp["coords"]
        mask[coords[:, 0], coords[:, 1]] = True
        grey_masks.append(mask)

    shape_to_color: dict[tuple[int, int, tuple[tuple[int, int], ...]], int] = {}
    for color in sorted(int(c) for c in np.unique(g) if int(c) not in {BG, GREY}):
        for comp in _find_color_components(g, color):
            obj_mask = _crop_component_mask(comp)
            hole = _single_hole_mask(obj_mask)
            if hole is None:
                continue
            key = _shape_key(hole)
            if key in shape_to_color:
                assert shape_to_color[key] == color
            else:
                shape_to_color[key] = color

    out = np.zeros_like(g)
    out[g == GREY] = GREY

    for i in range(len(grey_masks)):
        for j in range(i + 1, len(grey_masks)):
            channel = (
                binary_dilation(grey_masks[i], structure=STRUCT8)
                & binary_dilation(grey_masks[j], structure=STRUCT8)
                & (g == BG)
            )
            if not channel.any():
                continue
            key = _shape_key(channel)
            color = shape_to_color.get(key)
            if color is not None:
                out[channel] = color

    return out


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(22, 30)
    grid_w = random.randint(22, 30)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []

    n_pairs = random.choice([2, 3])
    matched_count = random.randint(1, n_pairs - 1)
    distractor_count = random.randint(1, 2)

    pair_infos = []
    used_shapes: set[tuple[int, int, tuple[tuple[int, int], ...]]] = set()

    for _ in range(n_pairs):
        patch = channel_full = channel_shape = None
        for _ in range(PAIR_RETRIES):
            patch, channel_full, channel_shape = _generate_pair_patch()
            key = _shape_key(channel_shape)
            if key not in used_shapes:
                used_shapes.add(key)
                break
        else:
            return None
        pair_infos.append(
            {
                "patch": patch,
                "channel_full": channel_full,
                "channel_shape": channel_shape,
                "shape_key": _shape_key(channel_shape),
            }
        )

    matched_indices = set(random.sample(range(n_pairs), matched_count))
    all_key_shapes = [pair_infos[idx]["channel_shape"] for idx in sorted(matched_indices)]
    while len(all_key_shapes) < matched_count + distractor_count:
        all_key_shapes.append(_generate_hole_mask(used_shapes))
        used_shapes.add(_shape_key(all_key_shapes[-1]))

    total_keys = matched_count + distractor_count
    colors = random.sample(KEY_COLORS, total_keys)
    key_specs = []
    for hole_mask, color in zip(all_key_shapes, colors):
        key_specs.append({"patch": _build_key_patch(hole_mask, color), "shape_key": _shape_key(hole_mask), "color": color})

    random.shuffle(key_specs)

    expected = np.zeros_like(grid)

    for info in pair_infos:
        pos = _place_patch(grid, info["patch"], placed_boxes, pad=2)
        if pos is None:
            return None
        top, left = pos
        expected[top : top + info["patch"].shape[0], left : left + info["patch"].shape[1]][info["patch"] == GREY] = GREY
        info["top"] = top
        info["left"] = left

    shape_to_color = {spec["shape_key"]: spec["color"] for spec in key_specs}
    for idx, info in enumerate(pair_infos):
        if idx not in matched_indices:
            continue
        color = shape_to_color[info["shape_key"]]
        h, w = info["patch"].shape
        expected[info["top"] : info["top"] + h, info["left"] : info["left"] + w][info["channel_full"]] = color

    for spec in key_specs:
        pos = _place_patch(grid, spec["patch"], placed_boxes, pad=2)
        if pos is None:
            return None

    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
