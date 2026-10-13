import numpy as np


MAX_GEN_ATTEMPTS = 400
PALETTE = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == len(coords)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            coords = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            coords = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords_arr = np.array(coords, dtype=int)
            bbox = _bbox_from_coords(coords_arr)
            r0, c0, r1, c1 = bbox
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            mask = patch == color
            comps.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": bbox,
                    "mask": mask,
                    "area": int(mask.sum()),
                    "bbox_area": int(mask.size),
                }
            )
    return comps


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((factor, factor), dtype=int)).astype(bool)


def _scale_factor_if_match(large_mask: np.ndarray, small_mask: np.ndarray) -> int | None:
    lh, lw = large_mask.shape
    sh, sw = small_mask.shape
    if sh > lh or sw > lw:
        return None
    if lh % sh != 0 or lw % sw != 0:
        return None
    factor_h = lh // sh
    factor_w = lw // sw
    if factor_h != factor_w or factor_h <= 1:
        return None
    factor = int(factor_h)
    if np.array_equal(_scale_mask(small_mask, factor), large_mask):
        return factor
    return None


def _match_missing_to_sprite(missing_mask: np.ndarray, sprites: list[dict]) -> list[tuple[int, dict]]:
    matches = []
    for sprite in sprites:
        factor = _scale_factor_if_match(missing_mask, sprite["mask"])
        if factor is not None:
            matches.append((factor, sprite))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each incomplete rectangle with the color of the uniquely shape-matching smaller sprite.

    Input structure:
    - The grid has a dominant background plus several disjoint monochrome components.
    - Some components are large rectangles with one connected background gap somewhere inside their own bounding boxes.
    - Other components are small monochrome sprites; some are exact same-orientation shapes as a gap after integer downscaling, and others are distractors.

    Rule:
    - For each large component, compute the background region inside its bounding box; this is the missing shape.
    - Among the other components, find the unique sprite whose tight shape scales up by one integer factor to exactly that missing shape.
    - Fill the entire missing region with that sprite's color, leaving the original rectangle pixels, all sprites, and the remaining background unchanged.

    Key insight:
    - The missing negative-space shape determines where to act, but only cross-scale shape matching determines which color to paint with.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    components = _find_monochrome_components(g, bg)
    assert components, "expected foreground components"

    out = g.copy()
    filled_any = False
    for comp in components:
        if comp["bbox_area"] == comp["area"]:
            continue
        missing_mask = ~comp["mask"]
        if not missing_mask.any():
            continue
        if len(_components(missing_mask)) != 1:
            continue

        sprites = [other for other in components if other is not comp]
        missing_shape = _tighten(missing_mask)
        matches = _match_missing_to_sprite(missing_shape, sprites)
        if len(matches) != 1:
            continue

        _, sprite = matches[0]
        r0, c0, _r1, _c1 = comp["bbox"]
        rr, cc = np.argwhere(missing_mask).T
        out[r0 + rr, c0 + cc] = sprite["color"]
        filled_any = True

    assert filled_any, "expected at least one incomplete rectangle with a unique scaled sprite match"
    return out


def _tighten(mask: np.ndarray) -> np.ndarray:
    return _crop_mask(mask)


def _grow_connected_mask(h: int, w: int, cell_count: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    start = (int(np.random.randint(0, h)), int(np.random.randint(0, w)))
    mask[start] = True
    frontier = {start}
    while int(mask.sum()) < cell_count and frontier:
        rr, cc = list(frontier)[int(np.random.randint(0, len(frontier)))]
        neighbors = []
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                neighbors.append((nr, nc))
        if not neighbors:
            frontier.remove((rr, cc))
            continue
        choice = neighbors[int(np.random.randint(0, len(neighbors)))]
        mask[choice] = True
        frontier.add(choice)
    if int(mask.sum()) != cell_count:
        return None
    tight = _tighten(mask)
    if tight.size == tight.sum():
        return None
    if min(tight.shape) < 2:
        return None
    return tight


def _random_shape() -> np.ndarray:
    for _ in range(200):
        h = int(np.random.randint(2, 5))
        w = int(np.random.randint(2, 5))
        max_cells = min(h * w - 1, 7)
        min_cells = max(3, (h * w) // 2)
        if min_cells > max_cells:
            continue
        cell_count = int(np.random.randint(min_cells, max_cells + 1))
        mask = _grow_connected_mask(h, w, cell_count)
        if mask is None:
            continue
        if not _is_connected(mask):
            continue
        return mask
    raise RuntimeError("failed to generate a connected shape")


def _mutate_shape(shape: np.ndarray) -> np.ndarray | None:
    h, w = shape.shape
    filled = [tuple(x) for x in np.argwhere(shape)]
    empty = [tuple(x) for x in np.argwhere(~shape)]
    order_f = np.random.permutation(len(filled)) if filled else []
    order_e = np.random.permutation(len(empty)) if empty else []
    for fi in order_f:
        for ei in order_e:
            cand = shape.copy()
            rf, cf = filled[int(fi)]
            re, ce = empty[int(ei)]
            cand[rf, cf] = False
            cand[re, ce] = True
            if not _is_connected(cand):
                continue
            tight = _tighten(cand)
            if tight.shape != shape.shape:
                continue
            if np.array_equal(tight, shape):
                continue
            if tight.size == tight.sum():
                continue
            return tight
    for _ in range(120):
        cand = _random_shape()
        if cand.shape == shape.shape and not np.array_equal(cand, shape) and int(cand.sum()) == int(shape.sum()):
            return cand
    return None


def _paint_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = color
    return patch


def _make_big_patch(shape: np.ndarray, scale: int, obj_color: int, bg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = _scale_mask(shape, scale)
    mh, mw = missing.shape
    margin_top = int(np.random.randint(1, 3))
    margin_bottom = int(np.random.randint(1, 3))
    margin_left = int(np.random.randint(1, 3))
    margin_right = int(np.random.randint(1, 3))
    rect_h = mh + margin_top + margin_bottom
    rect_w = mw + margin_left + margin_right
    patch = np.full((rect_h, rect_w), obj_color, dtype=int)
    missing_mask = np.zeros((rect_h, rect_w), dtype=bool)
    missing_mask[margin_top : margin_top + mh, margin_left : margin_left + mw] = missing
    patch[missing_mask] = bg
    return patch, missing_mask, missing


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def _padded_free(occ: np.ndarray, r0: int, c0: int, h: int, w: int, gap: int = 1) -> bool:
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(occ.shape[0], r0 + h + gap)
    cc1 = min(occ.shape[1], c0 + w + gap)
    return not occ[rr0:rr1, cc0:cc1].any()


def _mark_occ(occ: np.ndarray, r0: int, c0: int, h: int, w: int):
    occ[r0 : r0 + h, c0 : c0 + w] = True


def _place_objects(objects: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    heights = [obj["patch"].shape[0] for obj in objects]
    widths = [obj["patch"].shape[1] for obj in objects]
    approx = sum((h + 2) * (w + 2) for h, w in zip(heights, widths))
    min_side = max(max(heights) + 2, max(widths) + 2, int(np.ceil(np.sqrt(approx * 1.7))))
    if min_side > 30:
        return None

    for _ in range(120):
        h = int(np.random.randint(min_side, 31))
        w = int(np.random.randint(min_side, 31))
        occ = np.zeros((h, w), dtype=bool)
        placed = []
        ok = True
        for obj in sorted(objects, key=lambda x: x["patch"].size, reverse=True):
            ph, pw = obj["patch"].shape
            found = False
            for _ in range(250):
                r0 = int(np.random.randint(0, h - ph + 1))
                c0 = int(np.random.randint(0, w - pw + 1))
                if not _padded_free(occ, r0, c0, ph, pw, gap=1):
                    continue
                found = True
                _mark_occ(occ, r0, c0, ph, pw)
                rec = dict(obj)
                rec["top_left"] = (r0, c0)
                rec["bbox"] = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                placed.append(rec)
                break
            if not found:
                ok = False
                break
        if ok:
            grid = np.full((h, w), objects[0]["bg"], dtype=int)
            for obj in placed:
                r0, c0 = obj["top_left"]
                patch = obj["patch"]
                mask = patch != obj["bg"]
                sub = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
                if np.any(sub[mask] != obj["bg"]):
                    ok = False
                    break
                sub[mask] = patch[mask]
            if ok:
                return grid, placed
    return None


def _build_expected(grid: np.ndarray, bigs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for big in bigs:
        r0, c0 = big["top_left"]
        rr, cc = np.argwhere(big["missing_mask"]).T
        out[r0 + rr, c0 + cc] = big["fill_color"]
    return out


def _validate_generated_case(grid: np.ndarray, bigs: list[dict], sprites: list[dict], distractors: list[dict], expected: np.ndarray) -> bool:
    if not np.array_equal(main(grid), expected):
        return False

    bg = _get_bg(grid)
    comps = _find_monochrome_components(grid, bg)
    solver_big_count = 0
    for comp in comps:
        if comp["bbox_area"] == comp["area"]:
            continue
        missing_mask = ~comp["mask"]
        if len(_components(missing_mask)) != 1:
            continue
        missing_shape = _tighten(missing_mask)
        matches = _match_missing_to_sprite(missing_shape, [other for other in comps if other is not comp])
        if len(matches) == 1:
            solver_big_count += 1
    if solver_big_count != len(bigs):
        return False

    if len(distractors) == 0:
        return False

    matched_centers = []
    for big in bigs:
        big_center = _bbox_center(big["bbox"])
        sprite_center = _bbox_center(big["match_bbox"])
        matched_centers.append(_distance(big_center, sprite_center))
        if big["obj_color"] == big["fill_color"]:
            return False

    closer_distractor_exists = False
    for big, match_dist in zip(bigs, matched_centers):
        big_center = _bbox_center(big["bbox"])
        for dist_obj in distractors:
            if _distance(big_center, _bbox_center(dist_obj["bbox"])) < match_dist:
                closer_distractor_exists = True
                break
    if not closer_distractor_exists:
        return False

    distractor_shape_counterexample = False
    for dist_obj in distractors:
        for sprite in sprites:
            if dist_obj["shape"].shape == sprite["shape"].shape and int(dist_obj["shape"].sum()) == int(sprite["shape"].sum()):
                distractor_shape_counterexample = True
                break
    return distractor_shape_counterexample


def _generate_case():
    bg = int(np.random.randint(0, 10))
    target_count = int(np.random.choice([2, 3]))
    distractor_count = int(np.random.choice([1, 2]))
    needed_colors = 1 + 2 * target_count + distractor_count
    colors = [c for c in PALETTE if c != bg]
    np.random.shuffle(colors)
    if len(colors) < needed_colors:
        return None

    target_shapes = []
    while len(target_shapes) < target_count:
        shape = _random_shape()
        if any(np.array_equal(shape, other) for other in target_shapes):
            continue
        target_shapes.append(shape)

    distractor_shapes = []
    while len(distractor_shapes) < distractor_count:
        base = target_shapes[len(distractor_shapes) % len(target_shapes)]
        if np.random.rand() < 0.35:
            rotated = np.rot90(base, k=int(np.random.randint(1, 4)))
            candidate = _tighten(rotated)
            if candidate.shape == base.shape and not np.array_equal(candidate, base):
                shape = candidate
            else:
                shape = _mutate_shape(base)
        else:
            shape = _mutate_shape(base)
        if shape is None:
            return None
        if any(np.array_equal(shape, s) for s in target_shapes + distractor_shapes):
            return None
        distractor_shapes.append(shape)

    objects = []
    bigs = []
    sprites = []
    distractors = []
    color_idx = 0
    scales = np.random.choice([2, 3], size=target_count, replace=True)
    if 2 in scales and 3 not in scales and np.random.rand() < 0.5:
        scales[int(np.random.randint(0, target_count))] = 3
    if 3 in scales and 2 not in scales and np.random.rand() < 0.5:
        scales[int(np.random.randint(0, target_count))] = 2

    for shape, scale in zip(target_shapes, scales):
        obj_color = colors[color_idx]
        fill_color = colors[color_idx + 1]
        color_idx += 2
        big_patch, missing_mask, missing_shape = _make_big_patch(shape, int(scale), obj_color, bg)
        sprite_patch = _paint_patch(shape, fill_color, bg)
        big_rec = {
            "kind": "big",
            "patch": big_patch,
            "bg": bg,
            "shape": shape,
            "scale": int(scale),
            "missing_mask": missing_mask,
            "missing_shape": missing_shape,
            "obj_color": obj_color,
            "fill_color": fill_color,
        }
        sprite_rec = {
            "kind": "sprite",
            "patch": sprite_patch,
            "bg": bg,
            "shape": shape,
            "color": fill_color,
        }
        bigs.append(big_rec)
        sprites.append(sprite_rec)
        objects.extend([big_rec, sprite_rec])

    for shape in distractor_shapes:
        color = colors[color_idx]
        color_idx += 1
        rec = {
            "kind": "distractor",
            "patch": _paint_patch(shape, color, bg),
            "bg": bg,
            "shape": shape,
            "color": color,
        }
        distractors.append(rec)
        objects.append(rec)

    placed = _place_objects(objects)
    if placed is None:
        return None
    grid, placed_objects = placed

    kind_lists = {"big": [], "sprite": [], "distractor": []}
    for obj in placed_objects:
        kind_lists[obj["kind"]].append(obj)

    if len(kind_lists["big"]) != target_count or len(kind_lists["sprite"]) != target_count or len(kind_lists["distractor"]) != distractor_count:
        return None

    unmatched_sprites = kind_lists["sprite"][:]
    for big in kind_lists["big"]:
        match = None
        match_idx = None
        for idx, sprite in enumerate(unmatched_sprites):
            factor = _scale_factor_if_match(big["missing_shape"], sprite["shape"])
            if factor is not None:
                if match is not None:
                    return None
                match = sprite
                match_idx = idx
        if match is None:
            return None
        big["match_bbox"] = match["bbox"]
        del unmatched_sprites[int(match_idx)]

    for dist_obj in kind_lists["distractor"]:
        for big in kind_lists["big"]:
            if _scale_factor_if_match(big["missing_shape"], dist_obj["shape"]) is not None:
                return None

    expected = _build_expected(grid, kind_lists["big"])
    if np.array_equal(expected, grid):
        return None
    if not _validate_generated_case(grid, kind_lists["big"], kind_lists["sprite"], kind_lists["distractor"], expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid incomplete-rectangle color-completion input")
