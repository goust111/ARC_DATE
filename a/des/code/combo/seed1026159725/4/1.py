from __future__ import annotations

import random

import numpy as np


def _neighbors(connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((1, 0), (-1, 0), (0, 1), (0, -1))
    if connectivity == 8:
        return (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )
    raise ValueError("unsupported connectivity")


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(cnts.argmax())])


def _bbox(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    pts = np.asarray(coords, dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(coords: list[tuple[int, int]] | np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _components_from_mask(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    nbrs = _neighbors(connectivity)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in nbrs:
                    r2, c2 = rr + dr, cc + dc
                    if 0 <= r2 < h and 0 <= c2 < w and mask[r2, c2] and not seen[r2, c2]:
                        seen[r2, c2] = True
                        stack.append((int(r2), int(c2)))
            comps.append(comp)
    return comps


def _monochrome_components(grid: np.ndarray, bg: int, connectivity: int = 4) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    nbrs = _neighbors(connectivity)
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in nbrs:
                    r2, c2 = rr + dr, cc + dc
                    if (
                        0 <= r2 < h
                        and 0 <= c2 < w
                        and not seen[r2, c2]
                        and int(grid[r2, c2]) == color
                    ):
                        seen[r2, c2] = True
                        stack.append((int(r2), int(c2)))
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    best = None
    base = mask.astype(bool)
    for flipped in (False, True):
        cur = np.fliplr(base) if flipped else base
        for k in range(4):
            ori = np.rot90(cur, k)
            key = (tuple(map(int, ori.shape)), ori.astype(np.uint8).tobytes())
            if best is None or key < best:
                best = key
    assert best is not None
    return best


def _all_orientations(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    base = mask.astype(bool)
    for flipped in (False, True):
        cur = np.fliplr(base) if flipped else base
        for k in range(4):
            ori = np.rot90(cur, k)
            key = (tuple(map(int, ori.shape)), ori.astype(np.uint8).tobytes())
            if key not in seen:
                seen.add(key)
                out.append(ori.copy())
    return out


def _extract_holes(component: dict) -> list[dict]:
    coords = component["coords"]
    r0, c0, r1, c1 = component["bbox"]
    occ = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        occ[r - r0, c - c0] = True

    h, w = occ.shape
    exterior = np.zeros_like(occ, dtype=bool)
    stack: list[tuple[int, int]] = []
    for rr in range(h):
        for cc in range(w):
            if rr not in (0, h - 1) and cc not in (0, w - 1):
                continue
            if occ[rr, cc] or exterior[rr, cc]:
                continue
            exterior[rr, cc] = True
            stack.append((rr, cc))
            while stack:
                r2, c2 = stack.pop()
                for dr, dc in _neighbors(4):
                    r3, c3 = r2 + dr, c2 + dc
                    if 0 <= r3 < h and 0 <= c3 < w and not occ[r3, c3] and not exterior[r3, c3]:
                        exterior[r3, c3] = True
                        stack.append((int(r3), int(c3)))

    holes_mask = (~occ) & (~exterior)
    holes: list[dict] = []
    for rel_coords in _components_from_mask(holes_mask, connectivity=4):
        rel_mask = _mask_from_coords(rel_coords)
        hr0, hc0, _, _ = _bbox(rel_coords)
        abs_coords = [(r0 + rr, c0 + cc) for rr, cc in rel_coords]
        holes.append(
            {
                "coords": abs_coords,
                "mask": rel_mask,
                "bbox": (r0 + hr0, c0 + hc0, r0 + hr0 + rel_mask.shape[0] - 1, c0 + hc0 + rel_mask.shape[1] - 1),
                "key": _shape_key(rel_mask),
            }
        )
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each internal hole shape to a same-shaped legend sprite and fill the hole
    with that sprite's color, then erase the legend sprites.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    comps = _monochrome_components(g, bg=bg, connectivity=4)
    assert comps, "expected at least one non-background component"

    objects: list[dict] = []
    sprites: list[dict] = []
    for comp in comps:
        holes = _extract_holes(comp)
        if holes:
            comp = dict(comp)
            comp["holes"] = holes
            objects.append(comp)
        else:
            sprites.append(comp)

    assert objects, "expected at least one holed object"
    assert sprites, "expected at least one legend sprite"

    color_votes: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}
    for spr in sprites:
        mask = _mask_from_coords(spr["coords"])
        key = _shape_key(mask)
        bucket = color_votes.setdefault(key, {})
        color = int(spr["color"])
        bucket[color] = bucket.get(color, 0) + 1

    out = g.copy()
    for obj in objects:
        for hole in obj["holes"]:
            cmap = color_votes.get(hole["key"])
            assert cmap, "every hole shape must be represented in the legend"
            fill = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            rr, cc = zip(*hole["coords"], strict=False)
            out[np.asarray(rr, dtype=int), np.asarray(cc, dtype=int)] = int(fill)

    for spr in sprites:
        rr, cc = zip(*spr["coords"], strict=False)
        out[np.asarray(rr, dtype=int), np.asarray(cc, dtype=int)] = int(bg)

    return out


_RAW_SHAPES = [
    [[1]],
    [[1, 1]],
    [[1, 1, 1]],
    [[1, 0], [1, 1]],
    [[1, 1], [1, 1]],
    [[0, 1, 0], [1, 1, 1]],
    [[1, 1, 0], [0, 1, 1]],
    [[1, 1, 1], [1, 0, 0]],
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
]


def _build_shape_library() -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    lib: list[np.ndarray] = []
    for raw in _RAW_SHAPES:
        mask = np.asarray(raw, dtype=bool)
        key = _shape_key(mask)
        if key not in seen:
            seen.add(key)
            lib.append(mask)
    return lib


SHAPE_LIBRARY = _build_shape_library()
GEN_RETRIES = 400


def _render_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _rect_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int
) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _choose_distinct_colors(count: int) -> list[int]:
    colors = list(range(10))
    random.shuffle(colors)
    return colors[:count]


def _make_object(bg: int, obj_color: int, shapes: list[np.ndarray]) -> tuple[np.ndarray, list[dict]] | None:
    for _ in range(120):
        oriented = [random.choice(_all_orientations(shape)) for shape in shapes]
        oriented.sort(key=lambda m: (-int(m.sum()), -m.shape[0] * m.shape[1]))
        max_h = max(mask.shape[0] for mask in oriented)
        max_w = max(mask.shape[1] for mask in oriented)
        height = random.randint(max(6, max_h + 2), max(9, max_h + 4 + len(oriented)))
        width = random.randint(max(6, max_w + 2), max(10, max_w + 5 + len(oriented)))
        occupied = np.zeros((height, width), dtype=bool)
        placed: list[dict] = []
        ok = True
        for mask in oriented:
            sh, sw = mask.shape
            candidates = [
                (r, c)
                for r in range(1, height - sh)
                for c in range(1, width - sw)
            ]
            random.shuffle(candidates)
            chosen = None
            for r0, c0 in candidates:
                valid = True
                cells = np.argwhere(mask)
                for dr, dc in cells:
                    rr = r0 + int(dr)
                    cc = c0 + int(dc)
                    for r1 in range(max(0, rr - 1), min(height, rr + 2)):
                        for c1 in range(max(0, cc - 1), min(width, cc + 2)):
                            if occupied[r1, c1]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    chosen = (r0, c0)
                    break
            if chosen is None:
                ok = False
                break
            r0, c0 = chosen
            cells_rel = [(int(r0 + dr), int(c0 + dc)) for dr, dc in np.argwhere(mask)]
            for rr, cc in cells_rel:
                occupied[rr, cc] = True
            placed.append({"coords": cells_rel, "key": _shape_key(mask), "mask": mask})
        if not ok:
            continue

        patch = np.full((height, width), int(obj_color), dtype=int)
        holes: list[dict] = []
        for hole in placed:
            for rr, cc in hole["coords"]:
                patch[rr, cc] = int(bg)
            holes.append({"coords": hole["coords"], "key": hole["key"]})
        return patch, holes
    return None


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> list[tuple[int, int]]:
    h, w = patch.shape
    coords: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(patch[r, c]) == bg:
                continue
            grid[top + r, left + c] = int(patch[r, c])
            coords.append((top + r, left + c))
    return coords


def _place_objects(
    grid: np.ndarray,
    bg: int,
    patches: list[np.ndarray],
    area: tuple[int, int, int, int],
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = area
    placed_boxes: list[tuple[int, int, int, int]] = []
    positions: list[tuple[int, int]] = [(-1, -1)] * len(patches)
    order = list(range(len(patches)))
    order.sort(key=lambda i: -(patches[i].shape[0] * patches[i].shape[1]))
    for idx in order:
        patch = patches[idx]
        h, w = patch.shape
        if h > r1 - r0 + 1 or w > c1 - c0 + 1:
            return None
        candidates = [
            (rr, cc)
            for rr in range(r0, r1 - h + 2)
            for cc in range(c0, c1 - w + 2)
        ]
        random.shuffle(candidates)
        found = None
        for rr, cc in candidates:
            box = (rr, cc, rr + h - 1, cc + w - 1)
            if all(_rect_separated(box, prev, gap=1) for prev in placed_boxes):
                found = (rr, cc)
                placed_boxes.append(box)
                positions[idx] = found
                break
        if found is None:
            return None
    return positions


def _place_legend_sprites(
    grid: np.ndarray,
    bg: int,
    sprites: list[np.ndarray],
    side: str,
    bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    positions: list[tuple[int, int]] = [(-1, -1)] * len(sprites)
    order = list(range(len(sprites)))
    random.shuffle(order)
    if side in ("left", "right"):
        needed = sum(sprites[i].shape[0] for i in order) + len(order) + 1
        free = (r1 - r0 + 1) - needed
        if free < 0:
            raise ValueError("legend strip too short")
        cur = r0 + 1 + random.randint(0, free)
        for idx in order:
            patch = sprites[idx]
            sh, sw = patch.shape
            x_free = (c1 - c0 + 1) - sw - 2
            if x_free < 0:
                raise ValueError("legend strip too narrow")
            left = c0 + 1 + random.randint(0, x_free)
            positions[idx] = (cur, left)
            cur += sh + 1
    else:
        needed = sum(sprites[i].shape[1] for i in order) + len(order) + 1
        free = (c1 - c0 + 1) - needed
        if free < 0:
            raise ValueError("legend strip too short")
        cur = c0 + 1 + random.randint(0, free)
        for idx in order:
            patch = sprites[idx]
            sh, sw = patch.shape
            y_free = (r1 - r0 + 1) - sh - 2
            if y_free < 0:
                raise ValueError("legend strip too narrow")
            top = r0 + 1 + random.randint(0, y_free)
            positions[idx] = (top, cur)
            cur += sw + 1
    for patch, (top, left) in zip(sprites, positions, strict=False):
        _paste_patch(grid, patch, top, left, bg)
    return positions


def _build_layout(side: str, object_patches: list[np.ndarray], sprite_patches: list[np.ndarray]) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]]:
    obj_heights = [patch.shape[0] for patch in object_patches]
    obj_widths = [patch.shape[1] for patch in object_patches]
    sprite_heights = [patch.shape[0] for patch in sprite_patches]
    sprite_widths = [patch.shape[1] for patch in sprite_patches]

    main_h = max(max(obj_heights) + 4, sum(obj_heights) // 2 + 8, 12)
    main_w = max(max(obj_widths) + 4, sum(obj_widths) // 2 + 8, 12)
    if side in ("left", "right"):
        legend_w = max(max(sprite_widths) + 3, 6)
        full_h = max(main_h, sum(sprite_heights) + len(sprite_heights) + 3)
        full_w = main_w + legend_w + 2
        if side == "left":
            legend_bbox = (0, 0, full_h - 1, legend_w - 1)
            main_bbox = (0, legend_w + 2, full_h - 1, full_w - 1)
        else:
            main_bbox = (0, 0, full_h - 1, main_w - 1)
            legend_bbox = (0, main_w + 2, full_h - 1, full_w - 1)
    else:
        legend_h = max(max(sprite_heights) + 3, 6)
        full_h = main_h + legend_h + 2
        full_w = max(main_w, sum(sprite_widths) + len(sprite_widths) + 3)
        if side == "top":
            legend_bbox = (0, 0, legend_h - 1, full_w - 1)
            main_bbox = (legend_h + 2, 0, full_h - 1, full_w - 1)
        else:
            main_bbox = (0, 0, main_h - 1, full_w - 1)
            legend_bbox = (main_h + 2, 0, full_h - 1, full_w - 1)
    return np.zeros((full_h, full_w), dtype=int), main_bbox, legend_bbox


def _generate_instance() -> tuple[np.ndarray, np.ndarray] | None:
    side = random.choice(["left", "right", "top", "bottom"])
    n_objects = random.randint(2, 4)
    n_shapes = random.randint(2, 4)

    palette = _choose_distinct_colors(1 + n_objects + n_shapes)
    bg = palette[0]
    object_colors = palette[1 : 1 + n_objects]
    fill_colors = palette[1 + n_objects :]

    chosen_shapes = random.sample(SHAPE_LIBRARY, n_shapes)
    shape_to_color = {_shape_key(shape): color for shape, color in zip(chosen_shapes, fill_colors, strict=False)}

    hole_counts = [random.randint(1, 3) for _ in range(n_objects)]
    while sum(hole_counts) < n_shapes:
        hole_counts[random.randrange(n_objects)] += 1

    assignments: list[list[np.ndarray]] = [[] for _ in range(n_objects)]
    for shape in chosen_shapes:
        assignments[random.randrange(n_objects)].append(shape)
    for idx in range(n_objects):
        while len(assignments[idx]) < hole_counts[idx]:
            assignments[idx].append(random.choice(chosen_shapes))

    object_patches: list[np.ndarray] = []
    object_holes: list[list[dict]] = []
    for obj_color, shapes in zip(object_colors, assignments, strict=False):
        made = _make_object(bg, obj_color, shapes)
        if made is None:
            return None
        patch, holes = made
        object_patches.append(patch)
        object_holes.append(holes)

    sprite_patches = [
        _render_sprite(random.choice(_all_orientations(shape)), shape_to_color[_shape_key(shape)], bg)
        for shape in chosen_shapes
    ]

    grid, main_bbox, legend_bbox = _build_layout(side, object_patches, sprite_patches)
    grid.fill(int(bg))
    object_positions = _place_objects(grid, bg, object_patches, main_bbox)
    if object_positions is None:
        return None

    expected = grid.copy()
    for patch, holes, (top, left) in zip(object_patches, object_holes, object_positions, strict=False):
        _paste_patch(grid, patch, top, left, bg)
        _paste_patch(expected, patch, top, left, bg)
        for hole in holes:
            fill = shape_to_color[hole["key"]]
            for rr, cc in hole["coords"]:
                expected[top + rr, left + cc] = int(fill)

    sprite_positions = _place_legend_sprites(grid, bg, sprite_patches, side, legend_bbox)
    for patch, (top, left) in zip(sprite_patches, sprite_positions, strict=False):
        for rr, cc in np.argwhere(patch != bg):
            expected[top + int(rr), left + int(cc)] = int(bg)

    if _most_common_color(grid) != bg:
        return None
    if not np.any(grid != expected):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        made = _generate_instance()
        if made is not None:
            return made[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
