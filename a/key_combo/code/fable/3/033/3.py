import random
from collections import Counter, deque

import numpy as np


TRANSFORMS = (
    (0, False),
    (1, False),
    (2, False),
    (3, False),
    (0, True),
    (1, True),
    (2, True),
    (3, True),
)

MAX_GEN_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(arr)[r0 : r1 + 1, c0 : c1 + 1]


def _tight_crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    return _crop_bbox(np.asarray(mask, dtype=bool), _bbox_from_coords(coords))


def _apply_transform(arr: np.ndarray, rotation: int, flip: bool) -> np.ndarray:
    out = np.rot90(np.asarray(arr), int(rotation))
    if flip:
        out = np.fliplr(out)
    return out


def _mask_signature(mask: np.ndarray):
    mask = _tight_crop_mask(np.asarray(mask, dtype=bool))
    coords = tuple(map(tuple, np.argwhere(mask)))
    return (tuple(map(int, mask.shape)), coords)


def _canonical_shape_signature(mask: np.ndarray):
    mask = _tight_crop_mask(np.asarray(mask, dtype=bool))
    signatures = []
    for rotation, flip in TRANSFORMS:
        transformed = _tight_crop_mask(_apply_transform(mask, rotation, flip))
        signatures.append(_mask_signature(transformed))
    return min(signatures)


def _has_nontrivial_symmetry(mask: np.ndarray) -> bool:
    mask = _tight_crop_mask(np.asarray(mask, dtype=bool))
    for rotation, flip in TRANSFORMS:
        if rotation == 0 and not flip:
            continue
        transformed = _tight_crop_mask(_apply_transform(mask, rotation, flip))
        if np.array_equal(transformed, mask):
            return True
    return False


def _count_mask_holes(mask: np.ndarray) -> int:
    holes = _enclosed_regions(np.asarray(mask, dtype=bool))
    return len(holes)


def _enclosed_regions(component_mask: np.ndarray) -> list[np.ndarray]:
    component_mask = np.asarray(component_mask, dtype=bool)
    h, w = component_mask.shape
    exterior = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not component_mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not component_mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not component_mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    return _components_from_mask((~component_mask) & (~exterior))


def _extract_sprite_info(
    grid: np.ndarray, border_color: int, border_coords: np.ndarray, bg: int
) -> dict | None:
    r0, c0, r1, c1 = _bbox_from_coords(border_coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border_mask = np.zeros(sub.shape, dtype=bool)
    local_rows = border_coords[:, 0] - r0
    local_cols = border_coords[:, 1] - c0
    border_mask[local_rows, local_cols] = True

    holes = _enclosed_regions(border_mask)
    if len(holes) != 1:
        return None

    hole_coords = holes[0]
    hole_mask_full = np.zeros(sub.shape, dtype=bool)
    hole_mask_full[hole_coords[:, 0], hole_coords[:, 1]] = True
    hr0, hc0, hr1, hc1 = _bbox_from_coords(hole_coords)
    hole_shape = hole_mask_full[hr0 : hr1 + 1, hc0 : hc1 + 1]
    hole_contents = sub[hr0 : hr1 + 1, hc0 : hc1 + 1]

    payload = np.full(hole_shape.shape, bg, dtype=int)
    payload[hole_shape] = hole_contents[hole_shape]
    filled = bool(np.any(payload[hole_shape] != bg))

    return {
        "border_color": int(border_color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "hole_bbox": (int(r0 + hr0), int(c0 + hc0), int(r0 + hr1), int(c0 + hc1)),
        "shape": hole_shape.astype(bool),
        "payload": payload,
        "filled": filled,
    }


def _find_border_color(grid: np.ndarray, bg: int) -> int:
    best_color = None
    best_score = None
    for color in _colors_present(grid, exclude={bg}):
        comps = _components_from_mask(grid == color)
        sprite_count = 0
        hole_area = 0
        total_area = 0
        for comp in comps:
            total_area += int(comp.shape[0])
            info = _extract_sprite_info(grid, color, comp, bg)
            if info is None:
                continue
            sprite_count += 1
            hole_area += int(np.sum(info["shape"]))
        score = (sprite_count, hole_area, total_area)
        if best_score is None or score > best_score:
            best_score = score
            best_color = color
    if best_color is None or best_score is None or best_score[0] == 0:
        raise AssertionError("failed to identify the shared border color")
    return int(best_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every empty cavity by copying the patterned content from its shape-matched solved cavity under the same rigid transform.

    Input structure:
    - A dominant background contains several separated hollow sprites sharing one border color.
    - Each sprite has exactly one enclosed cavity; some cavities are empty background, others are completely filled with a multicolor pattern.
    - Filled cavity shapes are pairwise different, and cavity shapes have no rotational or reflection symmetry.

    Rule:
    - Identify the common border-color components and recover each sprite's enclosed cavity shape independently of whether that cavity is empty or already filled.
    - For every empty cavity, find the unique filled cavity whose shape matches after one rotation or reflection.
    - Apply that same rigid transform to the filled cavity's internal pattern and write the transformed pattern into the empty cavity, leaving already solved sprites unchanged.

    Key insight:
    - The solved sprites are the legend: the same transform that aligns cavity outlines must also be transferred to the multicolor content inside them.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    border_color = _find_border_color(g, bg)
    border_components = _components_from_mask(g == border_color)

    sprites = []
    for comp in border_components:
        info = _extract_sprite_info(g, border_color, comp, bg)
        if info is not None:
            sprites.append(info)

    assert sprites, "expected at least one hollow sprite"
    filled_sprites = [sprite for sprite in sprites if sprite["filled"]]
    empty_sprites = [sprite for sprite in sprites if not sprite["filled"]]
    assert filled_sprites, "expected at least one solved cavity"

    out = g.copy()
    for target in empty_sprites:
        matches = []
        for source in filled_sprites:
            for rotation, flip in TRANSFORMS:
                transformed_shape = _apply_transform(source["shape"], rotation, flip)
                if np.array_equal(transformed_shape, target["shape"]):
                    matches.append((source, rotation, flip))
        assert len(matches) == 1, "every empty cavity must have one unique solved match"

        source, rotation, flip = matches[0]
        transformed_payload = _apply_transform(source["payload"], rotation, flip)
        transformed_shape = _apply_transform(source["shape"], rotation, flip)
        assert np.array_equal(transformed_shape, target["shape"])

        r0, c0, r1, c1 = target["hole_bbox"]
        region = out[r0 : r1 + 1, c0 : c1 + 1].copy()
        region[target["shape"]] = transformed_payload[transformed_shape]
        out[r0 : r1 + 1, c0 : c1 + 1] = region

    return out


def _random_connected_shape(area: int, rng: random.Random) -> np.ndarray | None:
    board_size = 6
    for _ in range(120):
        start = (rng.randrange(board_size), rng.randrange(board_size))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < board_size and 0 <= nc < board_size and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != area:
            continue
        coords = np.array(sorted(cells), dtype=int)
        coords -= coords.min(axis=0)
        h = int(coords[:, 0].max()) + 1
        w = int(coords[:, 1].max()) + 1
        if h > 5 or w > 5:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        mask = _tight_crop_mask(mask)
        if _count_mask_holes(mask) != 0:
            continue
        if _has_nontrivial_symmetry(mask):
            continue
        return mask
    return None


def _random_shape_pair_same_area(rng: random.Random) -> list[np.ndarray] | None:
    for _ in range(160):
        area = rng.randint(5, 8)
        shapes = []
        signatures = set()
        while len(shapes) < 2:
            shape = _random_connected_shape(area, rng)
            if shape is None:
                break
            signature = _canonical_shape_signature(shape)
            if signature in signatures:
                continue
            signatures.add(signature)
            shapes.append(shape)
        if len(shapes) == 2:
            return shapes
    return None


def _random_pattern(mask: np.ndarray, palette: list[int], bg: int, rng: random.Random) -> np.ndarray:
    coords = [tuple(coord) for coord in np.argwhere(mask)]
    rng.shuffle(coords)
    color_count = rng.randint(2, min(4, len(coords), len(palette)))
    chosen_colors = rng.sample(palette, color_count)

    values = []
    for idx, _coord in enumerate(coords):
        if idx < color_count:
            values.append(chosen_colors[idx])
        else:
            values.append(rng.choice(chosen_colors))
    rng.shuffle(values)

    payload = np.full(mask.shape, bg, dtype=int)
    for (r, c), value in zip(coords, values):
        payload[r, c] = int(value)
    return payload


def _build_object(shape: np.ndarray, payload: np.ndarray | None, border_color: int, bg: int) -> np.ndarray:
    h, w = shape.shape
    obj = np.full((h + 2, w + 2), border_color, dtype=int)
    inner = obj[1 : h + 1, 1 : w + 1]
    if payload is None:
        inner[shape] = bg
    else:
        inner[shape] = payload[shape]
    return obj


def _paste(canvas: np.ndarray, obj: np.ndarray, top: int, left: int) -> None:
    h, w = obj.shape
    canvas[top : top + h, left : left + w] = obj


def _sample_non_identity_transform(
    rng: random.Random, *, need_flip: bool | None = None
) -> tuple[int, bool]:
    options = []
    for rotation, flip in TRANSFORMS:
        if rotation == 0 and not flip:
            continue
        if need_flip is None or flip == need_flip:
            options.append((rotation, flip))
    return rng.choice(options)


def _place_objects(
    objects: list[dict], bg: int, rng: random.Random
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    max_h = max(int(obj["array"].shape[0]) for obj in objects)
    max_w = max(int(obj["array"].shape[1]) for obj in objects)
    total_rect_area = sum(int(obj["array"].shape[0] * obj["array"].shape[1]) for obj in objects)
    min_side = max(max_h + 3, max_w + 3, int(np.ceil(np.sqrt(total_rect_area * 2.2))) + 2, 16)
    if min_side > 30:
        return None

    for _ in range(120):
        height = rng.randint(min_side, 30)
        width = rng.randint(min_side, 30)
        canvas = np.full((height, width), bg, dtype=int)
        occupied = np.zeros((height, width), dtype=bool)
        positions: list[tuple[int, int] | None] = [None] * len(objects)
        order = list(range(len(objects)))
        rng.shuffle(order)
        success = True

        for idx in order:
            obj = objects[idx]["array"]
            h, w = obj.shape
            if h + 2 > height or w + 2 > width:
                success = False
                break

            candidates = []
            for _try in range(200):
                top = rng.randint(1, height - h - 1)
                left = rng.randint(1, width - w - 1)
                r0 = max(0, top - 1)
                r1 = min(height, top + h + 1)
                c0 = max(0, left - 1)
                c1 = min(width, left + w + 1)
                if occupied[r0:r1, c0:c1].any():
                    continue
                candidates.append((top, left))
                if len(candidates) >= 24:
                    break
            if not candidates:
                success = False
                break

            top, left = rng.choice(candidates)
            _paste(canvas, obj, top, left)
            occupied[max(0, top - 1) : min(height, top + h + 1), max(0, left - 1) : min(width, left + w + 1)] = True
            positions[idx] = (top, left)

        if not success:
            continue
        if _get_bg(canvas) != bg:
            continue
        return canvas, [(int(r), int(c)) for r, c in positions]  # type: ignore[arg-type]

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    colors = list(range(10))
    bg = rng.choice(colors)
    border_color = rng.choice([color for color in colors if color != bg])
    pattern_palette = [color for color in colors if color not in {bg, border_color}]

    source_shapes = _random_shape_pair_same_area(rng)
    if source_shapes is None:
        return None

    source_payloads = [_random_pattern(shape, pattern_palette, bg, rng) for shape in source_shapes]
    source_objects = []
    for source_id, (shape, payload) in enumerate(zip(source_shapes, source_payloads)):
        source_objects.append(
            {
                "kind": "source",
                "source_id": source_id,
                "shape": shape,
                "payload": payload,
                "array": _build_object(shape, payload, border_color, bg),
            }
        )

    empty_specs = [
        (0, _sample_non_identity_transform(rng, need_flip=False)),
        (1, _sample_non_identity_transform(rng, need_flip=True)),
    ]
    for _ in range(rng.randint(0, 2)):
        source_id = rng.randrange(len(source_shapes))
        empty_specs.append((source_id, _sample_non_identity_transform(rng)))

    rng.shuffle(empty_specs)
    empty_objects = []
    for source_id, (rotation, flip) in empty_specs:
        transformed_shape = _apply_transform(source_shapes[source_id], rotation, flip)
        empty_objects.append(
            {
                "kind": "empty",
                "source_id": source_id,
                "rotation": int(rotation),
                "flip": bool(flip),
                "shape": transformed_shape,
                "array": _build_object(transformed_shape, None, border_color, bg),
            }
        )

    all_objects = source_objects + empty_objects
    placed = _place_objects(all_objects, bg, rng)
    if placed is None:
        return None
    inp, positions = placed

    out = inp.copy()
    for obj, (top, left) in zip(all_objects, positions):
        if obj["kind"] != "empty":
            continue
        source_id = int(obj["source_id"])
        source_payload = source_payloads[source_id]
        transformed_payload = _apply_transform(source_payload, obj["rotation"], obj["flip"])
        target_shape = obj["shape"]
        region = out[top + 1 : top + 1 + target_shape.shape[0], left + 1 : left + 1 + target_shape.shape[1]].copy()
        region[target_shape] = transformed_payload[target_shape]
        out[top + 1 : top + 1 + target_shape.shape[0], left + 1 : left + 1 + target_shape.shape[1]] = region

    if len(_components_from_mask(inp == border_color)) != len(all_objects):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")
