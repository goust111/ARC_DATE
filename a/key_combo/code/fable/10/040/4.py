from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
QUADRANTS = ((0, 0), (0, 1), (1, 0), (1, 1))
GEN_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = coords - np.array([r0, c0])
    out[shifted[:, 0], shifted[:, 1]] = True
    return out


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == int(color), dtype=bool)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    H, W = mask.shape
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    out = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        for coords in _color_components(grid, color):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                }
            )
    return out


def _quadrant_of_coords(coords: np.ndarray, shape: tuple[int, int]) -> tuple[int, int]:
    cr = shape[0] // 2
    cc = shape[1] // 2
    rows = coords[:, 0]
    cols = coords[:, 1]
    top = np.all(rows < cr)
    bottom = np.all(rows > cr)
    left = np.all(cols < cc)
    right = np.all(cols > cc)
    assert (top ^ bottom) and (left ^ right), "component must stay within one quadrant"
    return (0 if top else 1, 0 if left else 1)


def _apply_flips(mask: np.ndarray, flip_v: bool, flip_h: bool) -> np.ndarray:
    out = mask
    if flip_v:
        out = np.flipud(out)
    if flip_h:
        out = np.fliplr(out)
    return np.array(out, dtype=bool, copy=True)


def _mask_for_quadrant(template_mask: np.ndarray, template_q: tuple[int, int], object_q: tuple[int, int]) -> np.ndarray:
    flip_v = template_q[0] != object_q[0]
    flip_h = template_q[1] != object_q[1]
    return _apply_flips(template_mask, flip_v, flip_h)


def _embedding_positions(object_mask: np.ndarray, template_mask: np.ndarray) -> list[tuple[int, int]]:
    oh, ow = object_mask.shape
    th, tw = template_mask.shape
    if th > oh or tw > ow:
        return []
    coords = np.argwhere(template_mask)
    hits: list[tuple[int, int]] = []
    for r0 in range(oh - th + 1):
        for c0 in range(ow - tw + 1):
            shifted = coords + np.array([r0, c0])
            if np.all(object_mask[shifted[:, 0], shifted[:, 1]]):
                hits.append((int(r0), int(c0)))
    return hits


def _extract_template(grid: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int], int]:
    frames = []
    for coords in _color_components(grid, GREY):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        expected = np.zeros_like(patch, dtype=bool)
        expected[-1, :] = True
        expected[:, -1] = True
        if np.array_equal(patch == GREY, expected):
            frames.append((coords, (r0, c0, r1, c1)))
    assert len(frames) == 1, "expected exactly one grey L frame"

    _coords, frame_bbox = frames[0]
    r0, c0, r1, c1 = frame_bbox
    interior = grid[r0:r1, c0:c1]
    colors = sorted(set(map(int, np.unique(interior))) - {int(bg), GREY})
    assert len(colors) == 1, "expected one template color inside the frame"
    template_color = colors[0]
    template_mask = np.asarray(interior == template_color, dtype=bool)
    assert template_mask.any(), "template must be non-empty"
    template_mask = _crop_mask(template_mask)
    template_q = _quadrant_of_coords(np.array([[r0, c0], [r1, c1]], dtype=int), grid.shape)
    return template_mask, frame_bbox, template_q, int(template_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve the framed template into each quadrant object after mirroring it according to that object's quadrant relative to the frame.

    Input structure:
    - The background is black, and one grey bottom-right L-frame encloses a single-color template mask.
    - Outside the frame are several larger monochromatic objects, each fully contained in one quadrant of the grid split by the center row and center column.
    - The framed template is not mirror-symmetric, and each outside object contains exactly one embedding of the correctly mirrored template as a subshape.

    Rule:
    - Extract the template mask from inside the grey L-frame.
    - For each outside object, compare its quadrant with the frame's quadrant: crossing the horizontal centerline causes a vertical flip, and crossing the vertical centerline causes a horizontal flip.
    - Find the unique embedding of that flipped mask inside the object and recolor exactly those embedded object pixels to grey, leaving all other pixels unchanged.

    Key insight:
    - The template is searched as an internal subshape rather than as the whole object, and the required mirror is determined by quadrant displacement from the framed reference.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    template_mask, frame_bbox, template_q, _template_color = _extract_template(g, bg)
    fr0, fc0, fr1, fc1 = frame_bbox

    out = g.copy()
    carved = 0
    for comp in _find_components(g, bg):
        r0, c0, r1, c1 = comp["bbox"]
        if comp["color"] == GREY:
            assert (r0, c0, r1, c1) == frame_bbox, "unexpected grey component"
            continue
        if not (r1 < fr0 or r0 > fr1 or c1 < fc0 or c0 > fc1):
            continue

        object_q = _quadrant_of_coords(comp["coords"], g.shape)
        object_mask = _mask_from_coords(comp["coords"])
        target_mask = _mask_for_quadrant(template_mask, template_q, object_q)
        positions = _embedding_positions(object_mask, target_mask)
        assert len(positions) == 1, "each object must contain one unique target embedding"

        local_r, local_c = positions[0]
        target_cells = np.argwhere(target_mask) + np.array([local_r, local_c])
        out[r0 + target_cells[:, 0], c0 + target_cells[:, 1]] = GREY
        carved += int(target_cells.shape[0])

    assert carved > 0, "expected at least one object to carve"
    return out


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _make_template_mask() -> np.ndarray:
    for _ in range(400):
        h = random.randint(3, 4)
        w = random.randint(3, 4)
        area = random.randint(4, min(6, h * w - 1))
        mask = _random_connected_mask(h, w, area)
        if mask is None or not _touches_all_sides(mask):
            continue
        seen = {_mask_key(_apply_flips(mask, fv, fh)) for fv in (False, True) for fh in (False, True)}
        if len(seen) != 4:
            continue
        return mask
    raise RuntimeError("failed to sample an asymmetric template mask")


def _quadrant_bounds(shape: tuple[int, int], quadrant: tuple[int, int]) -> tuple[int, int, int, int]:
    H, W = shape
    cr = H // 2
    cc = W // 2
    if quadrant[0] == 0:
        r_lo, r_hi = 1, cr - 1
    else:
        r_lo, r_hi = cr + 1, H - 2
    if quadrant[1] == 0:
        c_lo, c_hi = 1, cc - 1
    else:
        c_lo, c_hi = cc + 1, W - 2
    return r_lo, r_hi, c_lo, c_hi


def _frame_top_left(shape: tuple[int, int], quadrant: tuple[int, int], frame_h: int, frame_w: int) -> tuple[int, int]:
    r_lo, r_hi, c_lo, c_hi = _quadrant_bounds(shape, quadrant)
    if quadrant[0] == 0:
        r0 = random.randint(r_lo, min(r_lo + 1, r_hi - frame_h + 1))
    else:
        r0 = random.randint(max(r_lo, r_hi - frame_h), r_hi - frame_h + 1)
    if quadrant[1] == 0:
        c0 = random.randint(c_lo, min(c_lo + 1, c_hi - frame_w + 1))
    else:
        c0 = random.randint(max(c_lo, c_hi - frame_w), c_hi - frame_w + 1)
    return int(r0), int(c0)


def _can_place(occupied: np.ndarray, patch_mask: np.ndarray, r0: int, c0: int, pad: int = 1) -> bool:
    H, W = occupied.shape
    coords = np.argwhere(patch_mask)
    for rr, cc in coords:
        gr = int(r0 + rr)
        gc = int(c0 + cc)
        if not (0 <= gr < H and 0 <= gc < W):
            return False
        for nr in range(gr - pad, gr + pad + 1):
            for nc in range(gc - pad, gc + pad + 1):
                if 0 <= nr < H and 0 <= nc < W and occupied[nr, nc]:
                    return False
    return True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    coords = np.argwhere(mask)
    grid[r0 + coords[:, 0], c0 + coords[:, 1]] = int(color)


def _mark_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _generate_object_mask(target_mask: np.ndarray, alt_masks: list[np.ndarray], max_h: int, max_w: int) -> np.ndarray | None:
    th, tw = target_mask.shape
    min_h = max(th + 1, 4)
    min_w = max(tw + 1, 4)
    if max_h < min_h or max_w < min_w:
        return None

    target_area = int(target_mask.sum())
    for _ in range(500):
        oh = random.randint(min_h, min(max_h, th + 3))
        ow = random.randint(min_w, min(max_w, tw + 3))
        er = random.randint(0, oh - th)
        ec = random.randint(0, ow - tw)
        mask = np.zeros((oh, ow), dtype=bool)
        mask[er : er + th, ec : ec + tw] |= target_mask

        target_coords = set(map(tuple, np.argwhere(mask)))
        cells = set(target_coords)
        extras = random.randint(3, 6)
        for _grow in range(extras):
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < oh and 0 <= nc < ow and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            outside = [p for p in frontier if not (er <= p[0] < er + th and ec <= p[1] < ec + tw)]
            choice = random.choice(outside or tuple(frontier))
            cells.add(choice)
            mask[choice] = True
            if random.random() < 0.35:
                r, c = choice
                step2 = [
                    (nr, nc)
                    for nr, nc in _neighbors4(r, c)
                    if 0 <= nr < oh and 0 <= nc < ow and (nr, nc) not in cells
                ]
                if step2:
                    choice2 = random.choice(step2)
                    cells.add(choice2)
                    mask[choice2] = True

        mask = _crop_mask(mask)
        if mask.shape == target_mask.shape:
            continue
        if int(mask.sum()) < target_area + 3:
            continue
        if len(_embedding_positions(mask, target_mask)) != 1:
            continue
        if any(_embedding_positions(mask, wrong) for wrong in alt_masks):
            continue
        return mask
    return None


def _placement_candidates(shape: tuple[int, int], quadrant: tuple[int, int], patch_shape: tuple[int, int]) -> list[tuple[int, int]]:
    r_lo, r_hi, c_lo, c_hi = _quadrant_bounds(shape, quadrant)
    ph, pw = patch_shape
    candidates = [
        (r0, c0)
        for r0 in range(r_lo, r_hi - ph + 2)
        for c0 in range(c_lo, c_hi - pw + 2)
    ]
    random.shuffle(candidates)
    return candidates


def _build_expected_output(
    inp: np.ndarray,
    object_specs: list[tuple[tuple[int, int], np.ndarray, int, int, np.ndarray]],
) -> np.ndarray:
    out = inp.copy()
    for _quadrant, mask, r0, c0, target in object_specs:
        pos = _embedding_positions(mask, target)
        assert len(pos) == 1
        pr, pc = pos[0]
        coords = np.argwhere(target) + np.array([pr, pc])
        out[r0 + coords[:, 0], c0 + coords[:, 1]] = GREY
    return out


def _validate_instance(inp: np.ndarray, expected: np.ndarray, template_q: tuple[int, int]) -> bool:
    bg = _get_bg(inp)
    try:
        template_mask, frame_bbox, seen_template_q, _ = _extract_template(inp, bg)
        if seen_template_q != template_q:
            return False
        seen_quadrants = set()
        for comp in _find_components(inp, bg):
            if comp["color"] == GREY:
                continue
            r0, c0, r1, c1 = comp["bbox"]
            fr0, fc0, fr1, fc1 = frame_bbox
            if not (r1 < fr0 or r0 > fr1 or c1 < fc0 or c0 > fc1):
                continue
            q = _quadrant_of_coords(comp["coords"], inp.shape)
            if q in seen_quadrants:
                return False
            seen_quadrants.add(q)
            obj_mask = _mask_from_coords(comp["coords"])
            target = _mask_for_quadrant(template_mask, template_q, q)
            if len(_embedding_positions(obj_mask, target)) != 1:
                return False
            for other_q in QUADRANTS:
                other = _mask_for_quadrant(template_mask, template_q, other_q)
                if _mask_key(other) == _mask_key(target):
                    continue
                if _embedding_positions(obj_mask, other):
                    return False
        if seen_quadrants != set(QUADRANTS):
            return False
        return np.array_equal(main(inp), expected)
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    H = random.choice([21, 23, 25, 27, 29])
    W = random.choice([21, 23, 25, 27, 29])
    grid = np.full((H, W), BLACK, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    cr = H // 2
    cc = W // 2
    occupied[cr, :] = True
    occupied[:, cc] = True

    template_mask = _make_template_mask()
    th, tw = template_mask.shape
    frame_h, frame_w = th + 1, tw + 1
    template_q = random.choice(QUADRANTS)
    frame_r0, frame_c0 = _frame_top_left(grid.shape, template_q, frame_h, frame_w)

    palette = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 5)
    template_color = palette[0]
    object_colors = palette[1:]

    grid[frame_r0 + th, frame_c0 : frame_c0 + frame_w] = GREY
    grid[frame_r0 : frame_r0 + frame_h, frame_c0 + tw] = GREY
    _paint_mask(grid, template_mask, frame_r0, frame_c0, template_color)
    _mark_bbox(occupied, frame_r0, frame_c0, frame_h, frame_w)

    all_targets = {_mask_key(_mask_for_quadrant(template_mask, template_q, q)): _mask_for_quadrant(template_mask, template_q, q) for q in QUADRANTS}
    object_specs: list[tuple[tuple[int, int], np.ndarray, int, int, np.ndarray]] = []

    for quadrant, color in zip(QUADRANTS, object_colors):
        target = _mask_for_quadrant(template_mask, template_q, quadrant)
        alt_masks = [mask for key, mask in all_targets.items() if key != _mask_key(target)]
        q_bounds = _quadrant_bounds(grid.shape, quadrant)
        max_h = min(8, q_bounds[1] - q_bounds[0] + 1)
        max_w = min(8, q_bounds[3] - q_bounds[2] + 1)
        obj_mask = _generate_object_mask(target, alt_masks, max_h=max_h, max_w=max_w)
        if obj_mask is None:
            return None

        placed = False
        for r0, c0 in _placement_candidates(grid.shape, quadrant, obj_mask.shape):
            if _can_place(occupied, obj_mask, r0, c0, pad=1):
                _paint_mask(grid, obj_mask, r0, c0, color)
                coords = np.argwhere(obj_mask)
                occupied[r0 + coords[:, 0], c0 + coords[:, 1]] = True
                object_specs.append((quadrant, obj_mask, r0, c0, target))
                placed = True
                break
        if not placed:
            return None

    expected = _build_expected_output(grid, object_specs)
    return grid if _validate_instance(grid, expected, template_q) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid quadrant-carving input")
