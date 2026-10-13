from __future__ import annotations

from collections import Counter, deque
import itertools
import random

import numpy as np


BG = 0
CANVAS = 5
MARKER = 7
SPRITE_COLORS = (1, 2, 3, 4)
GEN_RETRIES = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1]).copy()


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int, *, transparent: int = BG) -> None:
    h, w = patch.shape
    dst_slice = dst[top : top + h, left : left + w]
    mask = patch != int(transparent)
    dst_slice[mask] = patch[mask]


def _component_data(grid: np.ndarray, coords: np.ndarray) -> dict:
    bbox = _bbox_from_coords(coords)
    patch = _crop_bbox(grid, bbox)
    return {"coords": coords, "bbox": bbox, "patch": patch, "area": int(coords.shape[0])}


def _count_holes_from_patch(patch: np.ndarray, *, bg: int = BG) -> int:
    solid = patch != int(bg)
    holes = 0
    for comp in _components(~solid):
        if comp.size == 0:
            continue
        rs = comp[:, 0]
        cs = comp[:, 1]
        touches_edge = (
            int(rs.min()) == 0
            or int(cs.min()) == 0
            or int(rs.max()) == patch.shape[0] - 1
            or int(cs.max()) == patch.shape[1] - 1
        )
        if not touches_edge:
            holes += 1
    return holes


def _unique_translation(src_marker: np.ndarray, dst_marker: np.ndarray, patch_shape: tuple[int, int]) -> tuple[int, int]:
    src_coords = [tuple(map(int, rc)) for rc in np.argwhere(src_marker)]
    dst_coords = [tuple(map(int, rc)) for rc in np.argwhere(dst_marker)]
    assert src_coords and dst_coords
    dst_set = set(dst_coords)
    h, w = patch_shape
    dh, dw = dst_marker.shape
    candidates: set[tuple[int, int]] = set()

    for sr, sc in src_coords:
        for dr, dc in dst_coords:
            top = dr - sr
            left = dc - sc
            if top < 0 or left < 0 or top + h > dh or left + w > dw:
                continue
            shifted = {(r + top, c + left) for r, c in src_coords}
            if shifted <= dst_set:
                candidates.add((int(top), int(left)))

    assert len(candidates) == 1, "marker pattern must determine one translation"
    return next(iter(candidates))


def _overlay_with_order(dst_patch: np.ndarray, sprites: list[dict], order: list[int]) -> np.ndarray:
    out = dst_patch.copy()
    for idx in order:
        sprite = sprites[idx]
        _paste(out, sprite["patch"], int(sprite["top"]), int(sprite["left"]), transparent=BG)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the largest canvas and overlay aligned sprites in ascending order of internal hole count.

    Input structure:
    - Black is the dominant background.
    - One unique largest non-background 4-connected object is a grey destination canvas containing orange marker pixels.
    - Two to four smaller multicolor source sprites sit outside the canvas; each sprite has one body color, orange marker pixels, and zero or more enclosed black holes.

    Rule:
    - Identify the largest non-background 4-connected object and crop the output to its bounding box.
    - For every smaller source sprite, count the number of enclosed 4-connected black regions inside its cropped patch.
    - Find the unique translation that makes the sprite's orange marker set coincide with a subset of the canvas orange markers.
    - Overlay the sprites onto the cropped canvas from fewer holes to more holes, treating black sprite pixels as transparent so later sprites overwrite earlier ones.

    Key insight:
    - The visible top layer is determined by counting enclosed holes inside each sprite, not by area, while the orange marker pattern fixes where each sprite must be pasted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    comps = [_component_data(g, comp) for comp in _components(g != bg)]
    assert len(comps) >= 2

    areas = [comp["area"] for comp in comps]
    max_area = max(areas)
    assert areas.count(max_area) == 1
    dst_idx = areas.index(max_area)
    dst = comps[dst_idx]
    dst_patch = dst["patch"]
    dst_marker = dst_patch == MARKER

    sprites: list[dict] = []
    for idx, comp in enumerate(comps):
        if idx == dst_idx:
            continue
        patch = comp["patch"]
        top, left = _unique_translation(patch == MARKER, dst_marker, patch.shape)
        sprites.append(
            {
                "patch": patch,
                "top": int(top),
                "left": int(left),
                "holes": int(_count_holes_from_patch(patch, bg=bg)),
                "bbox": comp["bbox"],
            }
        )

    sprites.sort(key=lambda item: (item["holes"], item["bbox"][0], item["bbox"][1]))
    out = dst_patch.copy()
    for sprite in sprites:
        _paste(out, sprite["patch"], sprite["top"], sprite["left"], transparent=bg)
    return out


def _mask_from_strings(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


_NO_HOLE_BASES = [
    _mask_from_strings(["1111", "1111", "1111", "1111"]),
    _mask_from_strings(["00100", "01110", "11111", "01110", "00100"]),
    _mask_from_strings(["11110", "11110", "11111", "01110"]),
    _mask_from_strings(["01110", "11111", "11111", "01110"]),
]


def _partition_base(holes: int) -> np.ndarray:
    assert 1 <= holes <= 3
    if holes == 1:
        row_col_choices = [(1, 1)]
    elif holes == 2:
        row_col_choices = [(1, 2), (2, 1)]
    else:
        row_col_choices = [(1, 3), (3, 1)]

    rows, cols = random.choice(row_col_choices)
    cell_h = random.randint(1, 2)
    cell_w = random.randint(1, 2)
    h = rows * cell_h + rows + 1
    w = cols * cell_w + cols + 1
    wall_rows = {0}
    wall_cols = {0}
    pos = 0
    for _ in range(rows):
        pos += cell_h + 1
        wall_rows.add(pos)
    pos = 0
    for _ in range(cols):
        pos += cell_w + 1
        wall_cols.add(pos)
    rr, cc = np.indices((h, w))
    return np.isin(rr, list(wall_rows)) | np.isin(cc, list(wall_cols))


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    return mask[coords[:, 0].min() : coords[:, 0].max() + 1, coords[:, 1].min() : coords[:, 1].max() + 1].copy()


def _exterior_zero_mask(mask: np.ndarray) -> np.ndarray:
    zeros = ~mask
    out = np.zeros_like(mask, dtype=bool)
    for comp in _components(zeros):
        rs = comp[:, 0]
        cs = comp[:, 1]
        touches_edge = (
            int(rs.min()) == 0
            or int(cs.min()) == 0
            or int(rs.max()) == mask.shape[0] - 1
            or int(cs.max()) == mask.shape[1] - 1
        )
        if touches_edge:
            out[rs, cs] = True
    return out


def _tab_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    exterior = _exterior_zero_mask(mask)
    candidates: list[tuple[int, int]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not exterior[r, c]:
                continue
            if any(0 <= nr < h and 0 <= nc < w and mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                candidates.append((r, c))
    return candidates


def _decorate_mask(mask: np.ndarray, holes: int) -> np.ndarray:
    pad_t = random.randint(0, 2)
    pad_b = random.randint(0, 2)
    pad_l = random.randint(0, 2)
    pad_r = random.randint(0, 2)
    if max(pad_t, pad_b, pad_l, pad_r) == 0:
        pad_t = 1
    work = np.pad(mask, ((pad_t, pad_b), (pad_l, pad_r)))
    for _ in range(random.randint(0, 6)):
        candidates = _tab_candidates(work)
        if not candidates:
            break
        r, c = random.choice(candidates)
        work[r, c] = True
        if _count_holes_from_patch(np.where(work, 1, 0)) != holes:
            work[r, c] = False
    work = _crop_mask(work)
    assert _count_holes_from_patch(np.where(work, 1, 0)) == holes
    return work


def _make_sprite_mask(holes: int) -> np.ndarray:
    if holes == 0:
        base = random.choice(_NO_HOLE_BASES).copy()
    else:
        base = _partition_base(holes)
    for _ in range(60):
        mask = _decorate_mask(base, holes)
        if holes == 0 or min(mask.shape) >= 3:
            return mask
    raise RuntimeError("failed to build sprite mask")


def _sample_output_placements(masks: list[np.ndarray], crop_h: int, crop_w: int) -> list[tuple[int, int]] | None:
    for _ in range(400):
        placements: list[tuple[int, int]] = []
        supports: list[set[tuple[int, int]]] = []
        ok = True
        for mask in masks:
            h, w = mask.shape
            top = random.randint(0, crop_h - h)
            left = random.randint(0, crop_w - w)
            placements.append((top, left))
            coords = {
                (top + int(r), left + int(c))
                for r, c in np.argwhere(mask)
            }
            supports.append(coords)
        overlap = any(len(supports[i] & supports[j]) > 0 for i in range(len(supports)) for j in range(i + 1, len(supports)))
        if not overlap:
            ok = False
        if ok:
            return placements
    return None


def _marker_subset_candidates(mask: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    combos: list[tuple[tuple[int, int], ...]] = []
    for k in (3, 2):
        all_combos = list(itertools.combinations(coords, k))
        random.shuffle(all_combos)
        for combo in all_combos:
            rs = [r for r, _ in combo]
            cs = [c for _, c in combo]
            if max(rs) - min(rs) + max(cs) - min(cs) < 2:
                continue
            if len(set(rs)) == 1 or len(set(cs)) == 1:
                continue
            combos.append(tuple(sorted(combo)))
            if len(combos) >= 40:
                return combos
    return combos


def _choose_markers(
    masks: list[np.ndarray], placements: list[tuple[int, int]], crop_shape: tuple[int, int]
) -> tuple[list[tuple[tuple[int, int], ...]], set[tuple[int, int]]] | None:
    candidate_lists = [_marker_subset_candidates(mask) for mask in masks]
    if not all(candidate_lists):
        return None

    for _ in range(1200):
        chosen = [random.choice(cands) for cands in candidate_lists]
        dst_markers: set[tuple[int, int]] = set()
        distinct = True
        for combo, (top, left) in zip(chosen, placements):
            shifted = {(top + r, left + c) for r, c in combo}
            if shifted & dst_markers:
                distinct = False
                break
            dst_markers |= shifted
        if not distinct:
            continue

        dst_marker_grid = np.zeros(crop_shape, dtype=bool)
        for r, c in dst_markers:
            dst_marker_grid[r, c] = True

        unique = True
        for mask, combo, (top, left) in zip(masks, chosen, placements):
            src_marker = np.zeros(mask.shape, dtype=bool)
            for r, c in combo:
                src_marker[r, c] = True
            try:
                got = _unique_translation(src_marker, dst_marker_grid, mask.shape)
            except AssertionError:
                unique = False
                break
            if got != (top, left):
                unique = False
                break
        if unique:
            return chosen, dst_markers
    return None


def _build_canvas_mask(crop_h: int, crop_w: int, reserved: set[tuple[int, int]]) -> np.ndarray:
    mask = np.ones((crop_h, crop_w), dtype=bool)
    perimeter = [
        (0, c) for c in range(crop_w)
    ] + [
        (crop_h - 1, c) for c in range(crop_w)
    ] + [
        (r, 0) for r in range(1, crop_h - 1)
    ] + [
        (r, crop_w - 1) for r in range(1, crop_h - 1)
    ]
    random.shuffle(perimeter)
    removed = 0
    for r, c in perimeter:
        if (r, c) in reserved:
            continue
        test = mask.copy()
        test[r, c] = False
        if _components(test).__len__() != 1:
            continue
        if not test[0].any() or not test[-1].any() or not test[:, 0].any() or not test[:, -1].any():
            continue
        mask = test
        removed += 1
        if removed >= random.randint(1, 4):
            break
    return mask


def _render_sprite(mask: np.ndarray, body_color: int, markers: tuple[tuple[int, int], ...]) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(body_color)
    for r, c in markers:
        patch[r, c] = MARKER
    return patch


def _order_by_holes_then_bbox(sprites: list[dict]) -> list[int]:
    return sorted(range(len(sprites)), key=lambda i: (sprites[i]["holes"], sprites[i]["input_bbox"][0], sprites[i]["input_bbox"][1]))


def _order_by_area_then_bbox(sprites: list[dict]) -> list[int]:
    return sorted(range(len(sprites)), key=lambda i: (sprites[i]["area"], sprites[i]["input_bbox"][0], sprites[i]["input_bbox"][1]))


def _order_by_bbox_area_then_bbox(sprites: list[dict]) -> list[int]:
    return sorted(
        range(len(sprites)),
        key=lambda i: (sprites[i]["patch"].shape[0] * sprites[i]["patch"].shape[1], sprites[i]["input_bbox"][0], sprites[i]["input_bbox"][1]),
    )


def _place_patch_outside(
    grid: np.ndarray, patch: np.ndarray, occupied: set[tuple[int, int]]
) -> tuple[int, int] | None:
    h, w = patch.shape
    solid = [tuple(map(int, rc)) for rc in np.argwhere(patch != BG)]
    candidates: list[tuple[int, int]] = []
    for top in range(grid.shape[0] - h + 1):
        for left in range(grid.shape[1] - w + 1):
            ok = True
            for r, c in solid:
                rr = top + r
                cc = left + c
                if (rr, cc) in occupied:
                    ok = False
                    break
                for nr in range(rr - 1, rr + 2):
                    for nc in range(cc - 1, cc + 2):
                        if (nr, nc) in occupied:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                candidates.append((top, left))
    if not candidates:
        return None
    return random.choice(candidates)


def _semantic_check(inp: np.ndarray, expected: np.ndarray, sprites: list[dict], dst_patch: np.ndarray) -> bool:
    got = main(inp)
    if not np.array_equal(got, expected):
        return False

    correct = _order_by_holes_then_bbox(sprites)
    reverse = list(reversed(correct))
    by_area = _order_by_area_then_bbox(sprites)
    by_bbox_area = _order_by_bbox_area_then_bbox(sprites)
    if correct == by_area or correct == by_bbox_area:
        return False

    out_reverse = _overlay_with_order(dst_patch, sprites, reverse)
    out_area = _overlay_with_order(dst_patch, sprites, by_area)
    out_bbox_area = _overlay_with_order(dst_patch, sprites, by_bbox_area)
    if np.array_equal(expected, out_reverse):
        return False
    if np.array_equal(expected, out_area):
        return False
    if np.array_equal(expected, out_bbox_area):
        return False

    overlap = False
    for i in range(len(sprites)):
        a = {
            (sprites[i]["top"] + int(r), sprites[i]["left"] + int(c))
            for r, c in np.argwhere(sprites[i]["patch"] != BG)
        }
        for j in range(i + 1, len(sprites)):
            b = {
                (sprites[j]["top"] + int(r), sprites[j]["left"] + int(c))
                for r, c in np.argwhere(sprites[j]["patch"] != BG)
            }
            if a & b:
                overlap = True
    if not overlap:
        return False

    comps = _components(inp != BG)
    areas = [comp.shape[0] for comp in comps]
    if areas.count(max(areas)) != 1:
        return False

    return True


def _generate_instance() -> np.ndarray | None:
    n_sprites = random.choice((2, 3, 4))
    hole_counts = sorted(random.sample((0, 1, 2, 3), n_sprites))
    random.shuffle(hole_counts)

    masks = [_make_sprite_mask(holes) for holes in hole_counts]
    max_h = max(mask.shape[0] for mask in masks)
    max_w = max(mask.shape[1] for mask in masks)
    lo_h = max(max_h + 2, 9)
    hi_h = max(lo_h, min(18, max_h + 6))
    lo_w = max(max_w + 2, 9)
    hi_w = max(lo_w, min(18, max_w + 6))
    crop_h = random.randint(lo_h, hi_h)
    crop_w = random.randint(lo_w, hi_w)

    placements = _sample_output_placements(masks, crop_h, crop_w)
    if placements is None:
        return None

    marker_choice = _choose_markers(masks, placements, (crop_h, crop_w))
    if marker_choice is None:
        return None
    marker_subsets, dst_markers = marker_choice

    reserved = set().union(
        *[
            {(top + int(r), left + int(c)) for r, c in np.argwhere(mask)}
            for mask, (top, left) in zip(masks, placements)
        ]
    )
    canvas_mask = _build_canvas_mask(crop_h, crop_w, reserved | dst_markers)
    if not all(canvas_mask[r, c] for r, c in dst_markers):
        return None

    dst_patch = np.full((crop_h, crop_w), BG, dtype=int)
    dst_patch[canvas_mask] = CANVAS
    for r, c in dst_markers:
        dst_patch[r, c] = MARKER

    colors = random.sample(SPRITE_COLORS, n_sprites)
    sprites: list[dict] = []
    for holes, mask, markers, color, (top, left) in zip(hole_counts, masks, marker_subsets, colors, placements):
        patch = _render_sprite(mask, color, markers)
        sprites.append(
            {
                "holes": int(holes),
                "patch": patch,
                "top": int(top),
                "left": int(left),
                "area": int((patch != BG).sum()),
            }
        )

    hole_order = sorted(range(n_sprites), key=lambda i: sprites[i]["holes"])
    expected = _overlay_with_order(dst_patch, sprites, hole_order)

    grid_h = random.randint(max(crop_h + 8, 18), 26)
    grid_w = random.randint(max(crop_w + 8, 18), 26)
    inp = np.full((grid_h, grid_w), BG, dtype=int)

    canvas_top = random.randint(2, grid_h - crop_h - 2)
    canvas_left = random.randint(2, grid_w - crop_w - 2)
    inp[canvas_top : canvas_top + crop_h, canvas_left : canvas_left + crop_w] = dst_patch

    occupied = {(canvas_top + int(r), canvas_left + int(c)) for r, c in np.argwhere(dst_patch != BG)}
    order = sorted(range(n_sprites), key=lambda i: -sprites[i]["area"])
    for idx in order:
        pos = _place_patch_outside(inp, sprites[idx]["patch"], occupied)
        if pos is None:
            return None
        top, left = pos
        patch = sprites[idx]["patch"]
        solid = np.argwhere(patch != BG)
        for r, c in solid:
            rr = top + int(r)
            cc = left + int(c)
            inp[rr, cc] = int(patch[int(r), int(c)])
            occupied.add((rr, cc))
        sprites[idx]["input_bbox"] = (int(top), int(left), int(top + patch.shape[0] - 1), int(left + patch.shape[1] - 1))

    if not _semantic_check(inp, expected, sprites, dst_patch):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_instance()
        if grid is not None and grid.ndim == 2 and max(grid.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid input")
