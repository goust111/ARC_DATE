from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 500


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


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> np.ndarray:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]:
        raise ValueError("patch out of bounds")
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]
    return dst


def _components(mask: np.ndarray, *, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

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
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_monochromatic_components(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[dict]:
    exclude = set() if exclude is None else set(exclude)
    comps: list[dict] = []
    for color in sorted(set(map(int, grid.flatten())) - exclude):
        mask = grid == color
        for coords in _components(mask, connectivity=4):
            box = _bbox(mask_from_coords(coords, grid.shape))
            assert box is not None
            comps.append({"color": int(color), "coords": coords, "bbox": box, "area": int(coords.shape[0])})
    return comps


def mask_from_coords(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _tight_mask_from_component(comp: dict, shape: tuple[int, int]) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = comp["coords"]
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return patch


def _rotate_rel(coord: tuple[int, int], base_shape: tuple[int, int], k: int) -> tuple[int, int]:
    r, c = coord
    h, w = base_shape
    k %= 4
    if k == 0:
        return int(r), int(c)
    if k == 1:
        return int(w - 1 - c), int(r)
    if k == 2:
        return int(h - 1 - r), int(w - 1 - c)
    return int(c), int(h - 1 - r)


def _mask_rotations(mask: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(mask, k).astype(bool).copy() for k in range(4)]


def _is_rotationally_distinct(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _is_fully_asymmetric(mask: np.ndarray) -> bool:
    variants = [np.rot90(mask, k) for k in range(4)]
    if any(np.array_equal(mask, v) for v in variants[1:]):
        return False
    if any(np.array_equal(mask, np.fliplr(v)) or np.array_equal(mask, np.flipud(v)) for v in variants):
        return False
    return True


def _generate_connected_mask(
    sizes: list[tuple[int, int]],
    area_range: tuple[int, int],
    *,
    require_full_asymmetry: bool,
    require_rotational_distinct: bool,
) -> np.ndarray | None:
    for _ in range(200):
        h, w = random.choice(sizes)
        area = random.randint(area_range[0], min(area_range[1], h * w - 1))
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = {start}
        while len(cells) < area and frontier:
            rr, cc = random.choice(tuple(frontier))
            choices = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    choices.append((nr, nc))
            if not choices:
                frontier.discard((rr, cc))
                continue
            cell = random.choice(choices)
            cells.add(cell)
            frontier.add(cell)
        mask = np.zeros((h, w), dtype=bool)
        for rr, cc in cells:
            mask[rr, cc] = True
        box = _bbox(mask)
        assert box is not None
        mask = _crop_bbox(mask.astype(int), box).astype(bool)
        if min(mask.shape) < 2 or mask.sum() < 3:
            continue
        if not all(mask[r].any() for r in range(mask.shape[0])):
            continue
        if not all(mask[:, c].any() for c in range(mask.shape[1])):
            continue
        if len(_components(mask, connectivity=4)) != 1:
            continue
        if require_full_asymmetry and not _is_fully_asymmetric(mask):
            continue
        if require_rotational_distinct and not _is_rotationally_distinct(mask):
            continue
        return mask
    return None


def _build_demo_patch(
    a_mask: np.ndarray,
    b_mask: np.ndarray,
    b_offset: tuple[int, int],
    a_color: int,
    b_color: int,
) -> np.ndarray:
    a_coords = [tuple(x) for x in np.argwhere(a_mask)]
    b_coords = [tuple(x) for x in np.argwhere(b_mask)]
    br, bc = b_offset
    pair_coords = [(r, c, a_color) for r, c in a_coords] + [(br + r, bc + c, b_color) for r, c in b_coords]
    min_r = min(r for r, _, _ in pair_coords)
    max_r = max(r for r, _, _ in pair_coords)
    min_c = min(c for _, c, _ in pair_coords)
    max_c = max(c for _, c, _ in pair_coords)
    inner_h = max_r - min_r + 3
    inner_w = max_c - min_c + 3
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = FRAME
    patch[-1, :] = FRAME
    patch[:, 0] = FRAME
    patch[:, -1] = FRAME
    for r, c, color in pair_coords:
        patch[r - min_r + 2, c - min_c + 2] = color
    return patch


def _variant_patches(
    a_mask: np.ndarray,
    b_mask: np.ndarray,
    b_offset: tuple[int, int],
    a_color: int,
    b_color: int,
    rotation: int,
) -> dict:
    a_rot = np.rot90(a_mask, rotation).astype(bool).copy()
    h, w = a_mask.shape
    b_cells = [tuple(x) for x in np.argwhere(b_mask)]
    rel_b = [_rotate_rel((b_offset[0] + r, b_offset[1] + c), (h, w), rotation) for r, c in b_cells]
    a_cells = [tuple(x) for x in np.argwhere(a_rot)]
    all_cells = a_cells + rel_b
    min_r = min(r for r, _ in all_cells)
    max_r = max(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    max_c = max(c for _, c in all_cells)
    anchor = (-min_r, -min_c)
    patch_h = max_r - min_r + 1
    patch_w = max_c - min_c + 1

    inp = np.zeros((patch_h, patch_w), dtype=int)
    out = np.zeros((patch_h, patch_w), dtype=int)
    for r, c in a_cells:
        inp[anchor[0] + r, anchor[1] + c] = a_color
        out[anchor[0] + r, anchor[1] + c] = a_color
    for r, c in rel_b:
        out[anchor[0] + r, anchor[1] + c] = b_color
    return {
        "input_patch": inp,
        "output_patch": out,
        "footprint": out != BG,
        "anchor": anchor,
        "rotation": int(rotation),
        "a_shape": a_rot.shape,
    }


def _dilate(mask: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return mask.copy()
    out = np.zeros_like(mask, dtype=bool)
    coords = np.argwhere(mask)
    for r, c in coords:
        r0 = max(0, r - pad)
        r1 = min(mask.shape[0], r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(mask.shape[1], c + pad + 1)
        out[r0:r1, c0:c1] = True
    return out


def _random_location(occupied: np.ndarray, footprint: np.ndarray, *, border: int) -> tuple[int, int] | None:
    h, w = footprint.shape
    H, W = occupied.shape
    candidates: list[tuple[int, int]] = []
    for r in range(border, H - h - border + 1):
        for c in range(border, W - w - border + 1):
            if not np.any(occupied[r : r + h, c : c + w] & footprint):
                candidates.append((r, c))
    return random.choice(candidates) if candidates else None


def _reserve(occupied: np.ndarray, footprint: np.ndarray, r0: int, c0: int, *, pad: int) -> None:
    grown = _dilate(footprint, pad)
    h, w = grown.shape
    occupied[r0 : r0 + h, c0 : c0 + w] |= grown


def _frame_component(grid: np.ndarray) -> dict:
    grey_comps = [comp for comp in _find_monochromatic_components(grid, exclude={BG}) if comp["color"] == FRAME]
    valid: list[dict] = []
    for comp in grey_comps:
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1] == FRAME
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.array_equal(patch, border):
            valid.append(comp)
    assert len(valid) == 1
    return valid[0]


def _parse_rule(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    assert bg == BG
    frame = _frame_component(g)
    fr0, fc0, fr1, fc1 = frame["bbox"]

    all_comps = _find_monochromatic_components(g, exclude={bg, FRAME})
    inside: list[dict] = []
    outside: list[dict] = []
    for comp in all_comps:
        r0, c0, r1, c1 = comp["bbox"]
        if fr0 < r0 and r1 < fr1 and fc0 < c0 and c1 < fc1:
            inside.append(comp)
        else:
            outside.append(comp)

    assert len(inside) == 2
    inside_colors = {comp["color"] for comp in inside}
    outside_colors = {comp["color"] for comp in outside}
    shared = inside_colors & outside_colors
    assert len(shared) == 1
    a_color = next(iter(shared))
    b_color = next(iter(inside_colors - {a_color}))

    a_demo = next(comp for comp in inside if comp["color"] == a_color)
    b_demo = next(comp for comp in inside if comp["color"] == b_color)
    a_mask = _tight_mask_from_component(a_demo, g.shape)
    assert _is_fully_asymmetric(a_mask)
    a_box = a_demo["bbox"]
    rel_b = [(int(r - a_box[0]), int(c - a_box[1])) for r, c in b_demo["coords"]]
    a_rotations = _mask_rotations(a_mask)

    copies: list[dict] = []
    for comp in outside:
        assert comp["color"] == a_color
        mask = _tight_mask_from_component(comp, g.shape)
        matches = [k for k, rot in enumerate(a_rotations) if mask.shape == rot.shape and np.array_equal(mask, rot)]
        assert len(matches) == 1
        copies.append({"bbox": comp["bbox"], "rotation": matches[0]})
    assert copies

    return {
        "frame_bbox": frame["bbox"],
        "a_color": a_color,
        "b_color": b_color,
        "a_shape": a_mask.shape,
        "b_rel_coords": rel_b,
        "copies": copies,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the gray demo frame as a placement grammar, then stamp the companion shape beside every rotated copy of the demo shape.

    Input structure:
    - The background is black, and exactly one gray hollow rectangle encloses a demo area.
    - Inside the frame, one colored shape A and one differently colored shape B appear as separate 4-connected objects.
    - Outside the frame, several copies of A appear in the same color but rotated by quarter turns; B does not appear outside the frame.

    Rule:
    - Use the framed demo to identify A's base orientation, B's shape, and every B cell's position relative to A's tight bounding box.
    - For each outside copy of A, determine which quarter-turn rotation maps the demo A to that copy.
    - Apply that same rotation to B and to its relative coordinates, stamp the rotated B beside the copy, and preserve the original A pixels.
    - Erase the entire gray frame and everything inside it.

    Key insight:
    - The gray frame is not content to keep; it is a geometric syntax example whose relative layout must rotate covariantly with each outside copy of A.
    """
    g = np.asarray(grid, dtype=int)
    rule = _parse_rule(g)

    out = g.copy()
    fr0, fc0, fr1, fc1 = rule["frame_bbox"]
    out[fr0 : fr1 + 1, fc0 : fc1 + 1] = BG

    h, w = rule["a_shape"]
    for copy in rule["copies"]:
        r0, c0, _r1, _c1 = copy["bbox"]
        k = copy["rotation"]
        for rel in rule["b_rel_coords"]:
            rr, cc = _rotate_rel(rel, (h, w), k)
            tr = r0 + rr
            tc = c0 + cc
            assert 0 <= tr < out.shape[0] and 0 <= tc < out.shape[1]
            assert out[tr, tc] == BG
            out[tr, tc] = rule["b_color"]
    return out


def _choose_b_offset(a_mask: np.ndarray, b_mask: np.ndarray) -> tuple[int, int] | None:
    ah, aw = a_mask.shape
    bh, bw = b_mask.shape
    candidates: list[tuple[int, int]] = []
    for br in range(-bh - 2, ah + 3):
        for bc in range(-bw - 2, aw + 3):
            if (br <= -bh or br >= ah or bc <= -bw or bc >= aw):
                gap_ok = True
            else:
                gap_ok = False
            if not gap_ok:
                continue
            placed = np.zeros((max(ah, br + bh, bh - br, 1) + 8, max(aw, bc + bw, bw - bc, 1) + 8), dtype=int)
            origin = 4
            for r, c in np.argwhere(a_mask):
                placed[origin + r, origin + c] = 1
            overlap = False
            touch = False
            for r, c in np.argwhere(b_mask):
                rr = origin + br + r
                cc = origin + bc + c
                if placed[rr, cc]:
                    overlap = True
                    break
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        if placed[rr + dr, cc + dc]:
                            touch = True
            if overlap or touch:
                continue
            candidates.append((br, bc))
    return random.choice(candidates) if candidates else None


def _shortcut_unrotated(input_grid: np.ndarray, expected: np.ndarray, b_mask: np.ndarray, b_offset: tuple[int, int]) -> np.ndarray | None:
    info = _parse_rule(input_grid)
    out = input_grid.copy()
    fr0, fc0, fr1, fc1 = info["frame_bbox"]
    out[fr0 : fr1 + 1, fc0 : fc1 + 1] = BG
    for copy in info["copies"]:
        r0, c0, _r1, _c1 = copy["bbox"]
        for r, c in np.argwhere(b_mask):
            rr = r0 + b_offset[0] + int(r)
            cc = c0 + b_offset[1] + int(c)
            if not (0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]):
                return None
            if out[rr, cc] not in (BG, info["b_color"]):
                return None
            out[rr, cc] = info["b_color"]
    return out


def _shortcut_rotated_offset_only(
    input_grid: np.ndarray,
    b_mask: np.ndarray,
    b_offset: tuple[int, int],
    a_shape: tuple[int, int],
) -> np.ndarray | None:
    info = _parse_rule(input_grid)
    out = input_grid.copy()
    fr0, fc0, fr1, fc1 = info["frame_bbox"]
    out[fr0 : fr1 + 1, fc0 : fc1 + 1] = BG
    for copy in info["copies"]:
        r0, c0, _r1, _c1 = copy["bbox"]
        top = _rotate_rel(b_offset, a_shape, copy["rotation"])
        for r, c in np.argwhere(b_mask):
            rr = r0 + top[0] + int(r)
            cc = c0 + top[1] + int(c)
            if not (0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]):
                return None
            if out[rr, cc] not in (BG, info["b_color"]):
                return None
            out[rr, cc] = info["b_color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    a_mask = _generate_connected_mask(
        [(2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3)],
        (4, 7),
        require_full_asymmetry=True,
        require_rotational_distinct=True,
    )
    b_mask = _generate_connected_mask(
        [(2, 2), (2, 3), (3, 2), (3, 3)],
        (3, 5),
        require_full_asymmetry=False,
        require_rotational_distinct=True,
    )
    if a_mask is None or b_mask is None:
        return None

    a_color, b_color = random.sample(COLOR_POOL, 2)
    b_offset = _choose_b_offset(a_mask, b_mask)
    if b_offset is None:
        return None

    demo_patch = _build_demo_patch(a_mask, b_mask, b_offset, a_color, b_color)
    distinct_rotations = {0, random.choice((1, 3))}
    for k in range(1, 4):
        if random.random() < 0.35:
            distinct_rotations.add(k)
    rotations = list(distinct_rotations)
    while len(rotations) < random.randint(3, 5):
        rotations.append(random.choice(tuple(distinct_rotations)))
    random.shuffle(rotations)

    variants = [_variant_patches(a_mask, b_mask, b_offset, a_color, b_color, k) for k in rotations]
    if not any(v["rotation"] != 0 for v in variants):
        return None

    H = random.randint(20, 30)
    W = random.randint(20, 30)
    grid = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    demo_loc = _random_location(occupied, demo_patch != BG, border=1)
    if demo_loc is None:
        return None
    _paste(grid, demo_patch, demo_loc[0], demo_loc[1], transparent=BG)
    _reserve(occupied, demo_patch != BG, demo_loc[0], demo_loc[1], pad=1)

    placed_variants: list[tuple[tuple[int, int], dict]] = []
    order = list(range(len(variants)))
    order.sort(key=lambda idx: int(variants[idx]["footprint"].sum()), reverse=True)
    for idx in order:
        variant = variants[idx]
        loc = _random_location(occupied, variant["footprint"], border=1)
        if loc is None:
            return None
        _paste(grid, variant["input_patch"], loc[0], loc[1], transparent=BG)
        _reserve(occupied, variant["footprint"], loc[0], loc[1], pad=1)
        placed_variants.append((loc, variant))

    expected = grid.copy()
    expected[demo_loc[0] : demo_loc[0] + demo_patch.shape[0], demo_loc[1] : demo_loc[1] + demo_patch.shape[1]] = BG
    for loc, variant in placed_variants:
        _paste(expected, variant["output_patch"], loc[0], loc[1], transparent=BG)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    parsed = _parse_rule(grid)
    rotations_found = [copy["rotation"] for copy in parsed["copies"]]
    if len(rotations_found) < 3 or len(set(rotations_found)) < 2 or not any(k in (1, 3) for k in rotations_found):
        return None

    wrong1 = _shortcut_unrotated(grid, expected, b_mask, b_offset)
    if wrong1 is not None and np.array_equal(wrong1, expected):
        return None

    wrong2 = _shortcut_rotated_offset_only(grid, b_mask, b_offset, a_mask.shape)
    if wrong2 is not None and np.array_equal(wrong2, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
