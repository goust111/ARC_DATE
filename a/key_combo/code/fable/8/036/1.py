from __future__ import annotations

from collections import Counter
import random

import numpy as np


BBox = tuple[int, int, int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return bbox_coords(coords)


def crop_bbox(grid: np.ndarray, b: BBox) -> np.ndarray:
    r0, c0, r1, c1 = b
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    out: list[dict] = []
    for color in colors_present(grid, exclude={bg}):
        for coords in components_mask(grid == int(color), connectivity=4):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def row_intervals(flags: list[bool]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(flags) and flags[j + 1]:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    b = bbox(mask)
    assert b is not None
    return crop_bbox(mask.astype(bool), b)


def _in_bbox(r: int, c: int, b: BBox) -> bool:
    r0, c0, r1, c1 = b
    return r0 <= r <= r1 and c0 <= c <= c1


def _scene_band(grid: np.ndarray, bg: int) -> tuple[int, int]:
    intervals = row_intervals([bool(np.any(grid[r] != bg)) for r in range(grid.shape[0])])
    assert len(intervals) >= 2, "expected one upper scene band and lower candidate rows"
    return intervals[0]


def _find_occluder(scene: np.ndarray, full_grid: np.ndarray, bg: int) -> dict:
    counts = Counter(map(int, full_grid.ravel()))
    candidates = []
    for comp in monochrome_components(scene, bg):
        r0, c0, r1, c1 = comp["bbox"]
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if comp["area"] < 4 or comp["area"] != area:
            continue
        if counts[int(comp["color"])] != comp["area"]:
            continue
        candidates.append(comp)
    assert len(candidates) == 1, "expected a unique solid-color occluder"
    return candidates[0]


def _axis_candidates(scene: np.ndarray, occluder_bbox: BBox, occluder_color: int) -> list[tuple[int, int]]:
    h, w = scene.shape
    r0, c0, r1, c1 = occluder_bbox
    good: list[tuple[int, int]] = []
    for k in range(0, 2 * (w - 1) + 1):
        mc0 = k - c1
        mc1 = k - c0
        if not (0 <= mc0 <= mc1 < w):
            continue
        patch = scene[r0 : r1 + 1, mc0 : mc1 + 1]
        if np.any(patch == occluder_color):
            continue

        compared = 0
        mismatch = 0
        for rr in range(h):
            for cc in range(w):
                mm = k - cc
                if mm < 0 or mm >= w or cc > mm:
                    continue
                if _in_bbox(rr, cc, occluder_bbox) or _in_bbox(rr, mm, occluder_bbox):
                    continue
                if scene[rr, cc] == scene[rr, mm]:
                    compared += 1
                else:
                    mismatch += 1
        if compared > 0 and mismatch == 0:
            good.append((compared, k))
    return good


def _recover_hidden_patch(scene: np.ndarray, occluder_bbox: BBox, axis_k: int) -> np.ndarray:
    r0, c0, r1, c1 = occluder_bbox
    mc0 = axis_k - c1
    mc1 = axis_k - c0
    return np.fliplr(scene[r0 : r1 + 1, mc0 : mc1 + 1])


def _key_shape(recovered_patch: np.ndarray, bg: int) -> np.ndarray:
    mask = recovered_patch != bg
    assert mask.any(), "recovered patch must contain a hidden sprite"
    return normalize_mask(mask)


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the hidden mirrored sprite, repair the scene, then recolor the matching candidate.

    Input structure:
    - The upper occupied band is a vertically mirror-symmetric scene whose axis is offset from the whole-grid center.
    - One unique solid monochrome rectangle appears only once and occludes one side of a non-symmetric sprite pair.
    - Lower occupied components are single-color candidate sprites separated from the scene by blank rows.

    Rule:
    - Detect the scene's vertical mirror axis from the still-visible symmetric evidence outside the occluder.
    - Mirror the occluder bbox to the opposite side, read that counterpart patch, and flip it horizontally to recover the hidden content.
    - Replace the occluder with the recovered patch, then find the lower candidate congruent to the recovered sprite shape and recolor that candidate with the occluder color.

    Key insight:
    - The true key shape is not the visible twin itself but its mirrored restoration under the occluder's local reference frame, so the unflipped twin candidate is only a decoy.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    scene_r0, scene_r1 = _scene_band(g, bg)
    scene_rows = g[scene_r0 : scene_r1 + 1]
    scene_box = bbox(scene_rows != bg)
    assert scene_box is not None
    _sr0, scene_c0, _sr1, scene_c1 = scene_box
    scene = scene_rows[:, scene_c0 : scene_c1 + 1]

    occluder = _find_occluder(scene, g, bg)
    axis_scores = _axis_candidates(scene, occluder["bbox"], int(occluder["color"]))
    assert axis_scores, "failed to locate a valid vertical mirror axis"
    best_score = max(score for score, _k in axis_scores)
    best_axes = [k for score, k in axis_scores if score == best_score]
    assert len(best_axes) == 1, "expected a unique mirror axis"
    axis_k = int(best_axes[0])

    recovered_patch = _recover_hidden_patch(scene, occluder["bbox"], axis_k)
    key_shape = _key_shape(recovered_patch, bg)

    out = g.copy()
    or0, oc0, or1, oc1 = occluder["bbox"]
    gr0 = scene_r0 + or0
    gc0 = scene_c0 + oc0
    out[gr0 : gr0 + recovered_patch.shape[0], gc0 : gc0 + recovered_patch.shape[1]] = recovered_patch

    candidate_region = g[scene_r1 + 1 :, :]
    matches = []
    for comp in monochrome_components(candidate_region, bg):
        comp_mask = np.zeros(candidate_region.shape, dtype=bool)
        comp_mask[comp["coords"][:, 0], comp["coords"][:, 1]] = True
        if np.array_equal(normalize_mask(comp_mask), key_shape):
            matches.append(comp)

    assert len(matches) == 1, "expected exactly one candidate matching the recovered sprite"
    for rr, cc in matches[0]["coords"]:
        out[scene_r1 + 1 + int(rr), int(cc)] = int(occluder["color"])
    return out


COLORS = list(range(10))
GEN_TRIES = 400


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = set()
    stack = [tuple(map(int, coords[0]))]
    cells = {tuple(map(int, rc)) for rc in coords}
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        r, c = cell
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                stack.append(nxt)
    return len(seen) == len(cells)


def _random_connected_mask(
    h_range: tuple[int, int],
    w_range: tuple[int, int],
    *,
    min_cells: int,
    max_cells: int | None = None,
) -> np.ndarray:
    for _ in range(300):
        h = random.randint(*h_range)
        w = random.randint(*w_range)
        cap = h * w
        cell_hi = cap - 1
        if max_cells is not None:
            cell_hi = min(cell_hi, max_cells)
        target = random.randint(min_cells, max(min_cells, cell_hi))

        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        cells = {start}
        frontier = set()

        def add_frontier(r: int, c: int) -> None:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))

        add_frontier(*start)
        while len(cells) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            mask[cell] = True
            cells.add(cell)
            add_frontier(*cell)

        if len(cells) != target:
            continue
        if not _mask_connected(mask):
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        return mask
    raise RuntimeError("failed to generate a connected mask")


def _random_key_mask() -> np.ndarray:
    for _ in range(300):
        mask = _random_connected_mask((3, 5), (3, 6), min_cells=4)
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    raise RuntimeError("failed to generate a non-symmetric key mask")


def _random_filler_mask() -> np.ndarray:
    if random.random() < 0.5:
        h = random.randint(1, 2)
        w = random.randint(1, 3)
        return np.ones((h, w), dtype=bool)
    return _random_connected_mask((2, 3), (2, 4), min_cells=3, max_cells=6)


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    view = canvas[r0 : r0 + h, c0 : c0 + w]
    view[mask] = int(color)


def _can_place(occupied: np.ndarray, h: int, w: int, r0: int, c0: int, pad: int = 1) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0] - 1, r1 + pad)
    cc1 = min(occupied.shape[1] - 1, c1 + pad)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _mark_occupied(occupied: np.ndarray, h: int, w: int, r0: int, c0: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _build_scene(
    bg: int,
    scene_h: int,
    scene_w: int,
    occluder_color: int,
) -> tuple[np.ndarray, dict]:
    axis_k = scene_w - 1
    palette = [c for c in COLORS if c not in {bg, occluder_color}]
    rail_color, key_color = random.sample(palette, 2)
    filler_colors = [c for c in palette if c not in {rail_color, key_color}]

    scene = np.full((scene_h, scene_w), bg, dtype=int)
    scene[:, 0] = rail_color
    scene[:, -1] = rail_color
    occupied = scene != bg

    key_mask = _random_key_mask()
    kh, kw = key_mask.shape
    while 2 * kw + 3 > scene_w:
        key_mask = _random_key_mask()
        kh, kw = key_mask.shape
    max_left_c0 = (scene_w - 2 * kw) // 2
    assert max_left_c0 >= 1
    key_r0 = random.randint(1, scene_h - kh - 1)
    left_c0 = random.randint(1, max_left_c0)
    right_c0 = axis_k - (left_c0 + kw - 1)
    hidden_on_left = random.random() < 0.5

    _paint_mask(scene, key_mask, key_r0, left_c0, key_color)
    _paint_mask(scene, np.fliplr(key_mask), key_r0, right_c0, key_color)
    _mark_occupied(occupied, kh, kw, key_r0, left_c0)
    _mark_occupied(occupied, kh, kw, key_r0, right_c0)

    filler_count = random.randint(1, 3)
    for _ in range(filler_count):
        placed = False
        for _try in range(150):
            filler = _random_filler_mask()
            fh, fw = filler.shape
            max_fill_left = (scene_w - 2 * fw) // 2
            if max_fill_left < 1:
                break
            r0 = random.randint(0, scene_h - fh)
            c0 = random.randint(1, max_fill_left)
            rc0 = axis_k - (c0 + fw - 1)
            if not _can_place(occupied, fh, fw, r0, c0):
                continue
            if not _can_place(occupied, fh, fw, r0, rc0):
                continue
            color = random.choice(filler_colors)
            _paint_mask(scene, filler, r0, c0, color)
            _paint_mask(scene, np.fliplr(filler), r0, rc0, color)
            _mark_occupied(occupied, fh, fw, r0, c0)
            _mark_occupied(occupied, fh, fw, r0, rc0)
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place mirrored filler")

    if scene_w % 2 == 1:
        center_c = scene_w // 2
        center_color = random.choice([c for c in filler_colors if c != rail_color])
        rows = sorted(random.sample(range(scene_h), random.randint(2, min(4, scene_h))))
        scene[rows, center_c] = center_color

    if hidden_on_left:
        hidden_bbox = (key_r0, left_c0, key_r0 + kh - 1, left_c0 + kw - 1)
        visible_bbox = (key_r0, right_c0, key_r0 + kh - 1, right_c0 + kw - 1)
    else:
        hidden_bbox = (key_r0, right_c0, key_r0 + kh - 1, right_c0 + kw - 1)
        visible_bbox = (key_r0, left_c0, key_r0 + kh - 1, left_c0 + kw - 1)

    full_scene = scene.copy()
    scene[hidden_bbox[0] : hidden_bbox[2] + 1, hidden_bbox[1] : hidden_bbox[3] + 1] = occluder_color

    return scene, {
        "full_scene": full_scene,
        "axis_k": axis_k,
        "occluder_bbox": hidden_bbox,
        "visible_bbox": visible_bbox,
        "key_mask": key_mask.copy(),
        "key_color": key_color,
        "occluder_color": occluder_color,
    }


def _distinct_shape(masks: list[np.ndarray], candidate: np.ndarray) -> bool:
    return all(not np.array_equal(candidate, existing) for existing in masks)


def _place_candidates(
    grid: np.ndarray,
    cand_top: int,
    bg: int,
    true_mask: np.ndarray,
    occluder_color: int,
) -> dict | None:
    h, w = grid.shape
    region_h = h - cand_top
    occupied = np.zeros((region_h, w), dtype=bool)

    masks = [true_mask.copy(), np.fliplr(true_mask).copy()]
    while len(masks) < random.randint(4, 6):
        distractor = _random_connected_mask((2, 5), (2, 6), min_cells=3)
        distractor = normalize_mask(distractor)
        if np.array_equal(distractor, np.fliplr(true_mask)):
            continue
        if not _distinct_shape(masks, distractor):
            continue
        masks.append(distractor)

    order = list(range(len(masks)))
    random.shuffle(order)
    candidate_colors = random.sample([c for c in COLORS if c not in {bg, occluder_color}], len(masks))

    placements: dict[int, dict] = {}
    for idx, mask_idx in enumerate(order):
        mask = masks[mask_idx]
        mh, mw = mask.shape
        color = candidate_colors[idx]
        placed = False
        for _ in range(250):
            r0 = random.randint(0, region_h - mh)
            c0 = random.randint(0, w - mw)
            if not _can_place(occupied, mh, mw, r0, c0, pad=1):
                continue
            _paint_mask(grid[cand_top:], mask, r0, c0, color)
            _mark_occupied(occupied, mh, mw, r0, c0)
            placements[mask_idx] = {
                "mask": mask.copy(),
                "color": color,
                "top": cand_top + r0,
                "left": c0,
            }
            placed = True
            break
        if not placed:
            return None

    true_info = placements[0]
    decoy_info = placements[1]
    return {
        "true": true_info,
        "decoy": decoy_info,
        "shapes": masks,
        "placements": placements,
    }


def _expected_output(inp: np.ndarray, meta: dict) -> np.ndarray:
    out = inp.copy()
    scene_top = meta["scene_top"]
    scene_left = meta["scene_left"]
    hr0, hc0, hr1, hc1 = meta["scene_meta"]["occluder_bbox"]
    key_color = int(meta["scene_meta"]["key_color"])
    hidden_mask = meta["scene_meta"]["key_mask"]
    hidden_patch = np.full((hr1 - hr0 + 1, hc1 - hc0 + 1), meta["bg"], dtype=int)
    hidden_patch[hidden_mask] = key_color
    out[
        scene_top + hr0 : scene_top + hr1 + 1,
        scene_left + hc0 : scene_left + hc1 + 1,
    ] = hidden_patch

    true = meta["candidates"]["true"]
    tr0 = int(true["top"])
    tc0 = int(true["left"])
    th, tw = true["mask"].shape
    true_view = out[tr0 : tr0 + th, tc0 : tc0 + tw]
    true_view[true["mask"]] = int(meta["scene_meta"]["occluder_color"])
    return out


def _validate_semantics(inp: np.ndarray, meta: dict) -> bool:
    bg = int(meta["bg"])
    out = main(inp)
    expected = _expected_output(inp, meta)
    if not np.array_equal(out, expected):
        return False

    scene_top = meta["scene_top"]
    scene_left = meta["scene_left"]
    full_scene = meta["scene_meta"]["full_scene"]
    axis_k = int(meta["scene_meta"]["axis_k"])
    scene_h, scene_w = full_scene.shape
    repaired = out[scene_top : scene_top + scene_h, scene_left : scene_left + scene_w]
    if not np.array_equal(repaired, full_scene):
        return False

    for r in range(scene_h):
        for c in range(scene_w):
            if repaired[r, c] != repaired[r, axis_k - c]:
                return False

    if np.array_equal(meta["scene_meta"]["key_mask"], np.fliplr(meta["scene_meta"]["key_mask"])):
        return False

    cand = meta["candidates"]
    if not np.array_equal(cand["decoy"]["mask"], np.fliplr(cand["true"]["mask"])):
        return False

    seen = []
    for mask in cand["shapes"]:
        if any(np.array_equal(mask, old) for old in seen):
            return False
        seen.append(mask)

    parsed_scene_r0, parsed_scene_r1 = _scene_band(inp, bg)
    if parsed_scene_r0 != scene_top or parsed_scene_r1 != scene_top + scene_h - 1:
        return False

    parsed_scene = inp[parsed_scene_r0 : parsed_scene_r1 + 1, scene_left : scene_left + scene_w]
    occ = _find_occluder(parsed_scene, inp, bg)
    axes = _axis_candidates(parsed_scene, occ["bbox"], int(occ["color"]))
    if len([k for score, k in axes if score == max(score for score, _ in axes)]) != 1:
        return False

    if np.array_equal(inp, out):
        return False
    return True


def _generate_instance() -> tuple[np.ndarray, dict] | None:
    bg = random.choice(COLORS)
    occluder_color = random.choice([c for c in COLORS if c != bg])

    width = random.randint(18, 26)
    scene_w = random.randint(11, min(16, width - 2))
    scene_h = random.randint(7, 11)
    scene_top = random.randint(1, 2)
    gap = random.randint(2, 3)
    candidate_h = random.randint(7, 11)
    bottom_margin = random.randint(1, 2)
    height = scene_top + scene_h + gap + candidate_h + bottom_margin
    if height > 30:
        return None

    possible_lefts = []
    for scene_left in range(1, width - scene_w):
        axis_global_num = 2 * scene_left + scene_w - 1
        grid_center_num = width - 1
        if axis_global_num != grid_center_num:
            possible_lefts.append(scene_left)
    if not possible_lefts:
        return None
    scene_left = random.choice(possible_lefts)

    grid = np.full((height, width), bg, dtype=int)
    try:
        scene, scene_meta = _build_scene(bg, scene_h, scene_w, occluder_color)
    except RuntimeError:
        return None
    grid[scene_top : scene_top + scene_h, scene_left : scene_left + scene_w] = scene

    cand_top = scene_top + scene_h + gap
    candidates = _place_candidates(grid, cand_top, bg, scene_meta["key_mask"], occluder_color)
    if candidates is None:
        return None

    meta = {
        "bg": bg,
        "scene_top": scene_top,
        "scene_left": scene_left,
        "scene_meta": scene_meta,
        "candidates": candidates,
    }

    if Counter(map(int, grid.ravel())).most_common(1)[0][0] != bg:
        return None
    try:
        if not _validate_semantics(grid, meta):
            return None
    except AssertionError:
        return None
    return grid, meta


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        built = _generate_instance()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid mirrored-key ARC input")
