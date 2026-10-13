import random
from collections import Counter

import numpy as np


BG = 0
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GENERATION_RETRIES = 500
OBJECT_COUNT_OPTIONS = (2, 3)
HOLE_COUNT_POOL = (2, 3, 4, 5)
COLOR_POOL = tuple(range(1, 10))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
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
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _find_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        for coords in _component_coords(grid == color):
            bbox = _bbox_from_coords(coords)
            crop = _crop_bbox(grid, bbox)
            hole_mask, holes = _hole_mask_and_coords(crop, color, bg)
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox,
                    "crop": crop,
                    "hole_mask": hole_mask,
                    "holes": holes,
                    "scale": len(holes),
                }
            )
    return objects


def _hole_mask_and_coords(crop: np.ndarray, color: int, bg: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    h, w = crop.shape
    seen = np.zeros((h, w), dtype=bool)
    hole_mask = np.zeros((h, w), dtype=bool)
    holes: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(crop[r, c]) != bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_border = False
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_border = True
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and int(crop[nr, nc]) == bg and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if touches_border:
                continue
            assert len(coords) == 1, "every enclosed background component must be a 1-cell hole"
            hr, hc = coords[0]
            hole_mask[hr, hc] = True
            holes.append((hr, hc))

    fg_colors = set(map(int, np.unique(crop))) - {bg}
    assert fg_colors == {int(color)}, "each object must be monochrome"
    return hole_mask, holes


def _scale_int_array(arr: np.ndarray, factor: int) -> np.ndarray:
    k = int(factor)
    return np.kron(arr, np.ones((k, k), dtype=arr.dtype))


def _scale_bool_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(np.uint8), np.ones((factor, factor), dtype=np.uint8)).astype(bool)


def _transparent_paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _render_objects(objects: list[dict], shape: tuple[int, int], bg: int) -> tuple[np.ndarray, list[dict]]:
    out = np.full(shape, bg, dtype=int)
    ordered = sorted(objects, key=lambda obj: obj["scale"], reverse=True)
    layer_info: list[dict] = []
    for obj in ordered:
        r0, c0, _r1, _c1 = obj["bbox"]
        scaled_crop = _scale_int_array(obj["crop"], obj["scale"])
        scaled_holes = _scale_bool_mask(obj["hole_mask"], obj["scale"])
        h, w = scaled_crop.shape
        assert r0 + h <= shape[0] and c0 + w <= shape[1], "scaled object must fit in the canvas"
        _transparent_paste(out, scaled_crop, r0, c0, bg)

        abs_holes = np.zeros(shape, dtype=bool)
        abs_holes[r0 : r0 + h, c0 : c0 + w] = scaled_holes
        layer_info.append({"obj": obj, "hole_mask_abs": abs_holes})
    return out, layer_info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale each holed object by its hole count and composite the enlarged objects with transparent holes.

    Input structure:
    - The grid has a uniform background and several separated single-color foreground objects.
    - Each object may have border concavities, but every enclosed background component inside its bbox is a 1-cell hole.
    - Different objects have different hole counts, and each count is at least two.

    Rule:
    - For each object, count its enclosed 1-cell holes and use that count as the integer scale factor k.
    - Replace the original small objects by enlarging every cell of the object's bbox into a k x k block, anchored at the original bbox top-left.
    - Draw enlarged objects from larger k to smaller k; background cells inside the enlarged bbox stay transparent, so lower layers remain visible through holes.

    Key insight:
    - The same holes both encode the enlargement factor and remain transparent windows after scaling, while the reverse count order resolves layer priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    objects = _find_objects(g, bg)
    assert len(objects) >= 2, "expected multiple objects"

    scales = [obj["scale"] for obj in objects]
    assert all(scale >= 2 for scale in scales), "every object must have at least two holes"
    assert len(scales) == len(set(scales)), "hole counts must be unique"

    out, _info = _render_objects(objects, g.shape, bg)
    return out


def _interior_capacity(core_h: int, core_w: int) -> int:
    cells = max(0, core_h - 2) * max(0, core_w - 2)
    return (cells + 1) // 2


def _choose_nonadjacent_holes(core_h: int, core_w: int, hole_count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(1, core_h - 1) for c in range(1, core_w - 1)]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []

    def search(start: int) -> bool:
        if len(chosen) == hole_count:
            return True
        if len(candidates) - start < hole_count - len(chosen):
            return False
        for idx in range(start, len(candidates)):
            r, c = candidates[idx]
            if any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in chosen):
                continue
            chosen.append((r, c))
            if search(idx + 1):
                return True
            chosen.pop()
        return False

    return chosen.copy() if search(0) else None


def _random_span(length: int) -> tuple[int, int]:
    if length <= 2:
        return 0, length
    span_len = random.randint(max(2, length // 2), length)
    start = random.randint(0, length - span_len)
    return start, start + span_len


def _make_object_mask(hole_count: int) -> np.ndarray | None:
    configs: list[tuple[int, int, int, int, int, int]] = []
    for core_h in range(4, 7):
        for core_w in range(4, 7):
            if _interior_capacity(core_h, core_w) < hole_count:
                continue
            for top in (0, 1):
                for bottom in (0, 1):
                    for left in (0, 1):
                        for right in (0, 1):
                            final_h = core_h + top + bottom
                            final_w = core_w + left + right
                            if max(final_h, final_w) > 6:
                                continue
                            if hole_count >= 5 and (top + bottom + left + right) > 0:
                                continue
                            configs.append((core_h, core_w, top, bottom, left, right))
    random.shuffle(configs)

    for core_h, core_w, top, bottom, left, right in configs:
        mask = np.zeros((core_h + top + bottom, core_w + left + right), dtype=bool)
        r0, c0 = top, left
        mask[r0 : r0 + core_h, c0 : c0 + core_w] = True

        if top:
            s0, s1 = _random_span(core_w)
            mask[:top, c0 + s0 : c0 + s1] = True
        if bottom:
            s0, s1 = _random_span(core_w)
            mask[r0 + core_h :, c0 + s0 : c0 + s1] = True
        if left:
            s0, s1 = _random_span(core_h)
            mask[r0 + s0 : r0 + s1, :left] = True
        if right:
            s0, s1 = _random_span(core_h)
            mask[r0 + s0 : r0 + s1, c0 + core_w :] = True

        holes = _choose_nonadjacent_holes(core_h, core_w, hole_count)
        if holes is None:
            continue
        for hr, hc in holes:
            mask[r0 + hr, c0 + hc] = False

        if len(_component_coords(mask)) != 1:
            continue
        crop = mask.astype(int)
        hole_mask, hole_coords = _hole_mask_and_coords(crop, 1, 0)
        if len(hole_coords) != hole_count:
            continue
        return mask
    return None


def _mask_to_colored_crop(mask: np.ndarray, color: int) -> np.ndarray:
    crop = np.zeros(mask.shape, dtype=int)
    crop[mask] = int(color)
    return crop


def _scaled_shape(crop: np.ndarray, scale: int) -> tuple[int, int]:
    h, w = crop.shape
    return h * scale, w * scale


def _biased_coord(limit: int) -> int:
    if limit <= 0:
        return 0
    return min(limit, int((random.random() ** 1.8) * (limit + 1)))


def _can_place(mask: np.ndarray, blocked: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > blocked.shape[0] or c0 + w > blocked.shape[1]:
        return False
    coords = np.argwhere(mask)
    return not np.any(blocked[r0 + coords[:, 0], c0 + coords[:, 1]])


def _mark_blocked(blocked: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    coords = np.argwhere(mask)
    for rr, cc in coords:
        ar, ac = r0 + int(rr), c0 + int(cc)
        for dr, dc in NEIGHBORS8:
            nr, nc = ar + dr, ac + dc
            if 0 <= nr < blocked.shape[0] and 0 <= nc < blocked.shape[1]:
                blocked[nr, nc] = True


def _visible_window_on_top(layer_info: list[dict], final_grid: np.ndarray, bg: int) -> bool:
    top_layer = min(layer_info, key=lambda item: item["obj"]["scale"])
    return bool(np.any(final_grid[top_layer["hole_mask_abs"]] != bg))


def _build_generator_objects() -> list[dict] | None:
    obj_count = random.choice(OBJECT_COUNT_OPTIONS)
    counts = sorted(random.sample(HOLE_COUNT_POOL, obj_count), reverse=True)
    colors = random.sample(COLOR_POOL, obj_count)

    objects: list[dict] = []
    for hole_count, color in zip(counts, colors):
        mask = _make_object_mask(hole_count)
        if mask is None:
            return None
        crop = _mask_to_colored_crop(mask, color)
        hole_mask, holes = _hole_mask_and_coords(crop, color, BG)
        objects.append(
            {
                "color": color,
                "crop": crop,
                "mask": mask,
                "hole_mask": hole_mask,
                "holes": holes,
                "scale": hole_count,
            }
        )
    return objects


def _choose_canvas_shape(objects: list[dict]) -> tuple[int, int]:
    min_h = max(_scaled_shape(obj["crop"], obj["scale"])[0] for obj in objects)
    min_w = max(_scaled_shape(obj["crop"], obj["scale"])[1] for obj in objects)
    lo_h = max(18, min_h)
    lo_w = max(18, min_w)
    hi_h = min(30, lo_h + 6)
    hi_w = min(30, lo_w + 6)
    return random.randint(lo_h, hi_h), random.randint(lo_w, hi_w)


def _place_generator_objects(objects: list[dict], shape: tuple[int, int]) -> list[dict] | None:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []
    for obj in sorted(objects, key=lambda item: item["scale"], reverse=True):
        crop_h, crop_w = obj["crop"].shape
        scaled_h, scaled_w = _scaled_shape(obj["crop"], obj["scale"])
        max_r = h - scaled_h
        max_c = w - scaled_w
        if max_r < 0 or max_c < 0:
            return None

        candidates: list[tuple[int, int]] = []
        for _ in range(90):
            r0 = _biased_coord(max_r)
            c0 = _biased_coord(max_c)
            if _can_place(obj["mask"], blocked, r0, c0):
                candidates.append((r0, c0))
        if not candidates:
            for r0 in range(max_r + 1):
                for c0 in range(max_c + 1):
                    if _can_place(obj["mask"], blocked, r0, c0):
                        candidates.append((r0, c0))
        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        placed_obj = dict(obj)
        placed_obj["bbox"] = (r0, c0, r0 + crop_h - 1, c0 + crop_w - 1)
        placed.append(placed_obj)
        _mark_blocked(blocked, obj["mask"], r0, c0)
    return placed


def _render_input(objects: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in objects:
        r0, c0, _r1, _c1 = obj["bbox"]
        _transparent_paste(grid, obj["crop"], r0, c0, BG)
    return grid


def _gen_attempt() -> np.ndarray | None:
    objects = _build_generator_objects()
    if objects is None:
        return None

    shape = _choose_canvas_shape(objects)
    placed = _place_generator_objects(objects, shape)
    if placed is None:
        return None

    inp = _render_input(placed, shape)
    out, layer_info = _render_objects(placed, shape, BG)

    if not _visible_window_on_top(layer_info, out, BG):
        return None
    if float(np.count_nonzero(out)) / float(out.size) > 0.72:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
