import random
from collections import Counter, defaultdict, deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> BBox:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=grid.dtype)


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _find_components(grid: np.ndarray, bg: int, connectivity: int = 4) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        for coords in _components_mask(grid == color, connectivity=connectivity):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(len(coords)),
                }
            )
    return comps


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, BBox]:
    bbox = _bbox_coords(coords)
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, bbox


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return mask.shape, mask.tobytes()


def _paste(out: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, mask: np.ndarray | None = None) -> None:
    h, w = patch.shape
    if mask is None:
        out[r0 : r0 + h, c0 : c0 + w] = patch
        return
    target = out[r0 : r0 + h, c0 : c0 + w]
    target[mask] = patch[mask]


def _shape(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


_BASE_SHAPES = [
    _shape(("##",)),
    _shape(("###",)),
    _shape(("##", "#.")),
    _shape((".#", "##")),
    _shape(("###", ".#.")),
    _shape(("##.", ".##")),
    _shape(("#..", "###")),
    _shape(("###", "..#")),
    _shape((".#.", "###")),
    _shape(("##", ".#", ".#")),
    _shape(("#.", "#.", "##")),
    _shape(("####",)),
    _shape(("###", "##.")),
    _shape(("###", ".##")),
]


def _build_shape_library() -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    result: list[np.ndarray] = []
    for shape in _BASE_SHAPES:
        key = _mask_key(shape)
        if key not in seen:
            seen.add(key)
            result.append(shape)
    return result


SHAPE_LIBRARY = _build_shape_library()
MAX_ATTEMPTS = 400


def _detect_walls(grid: np.ndarray, bg: int) -> tuple[str, list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comps = _find_components(g, bg=bg, connectivity=4)

    left = []
    right = []
    top = []
    bottom = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        if r0 == 0 and r1 == h - 1:
            if c0 == 0:
                left.append(comp)
            if c1 == w - 1:
                right.append(comp)
        if c0 == 0 and c1 == w - 1:
            if r0 == 0:
                top.append(comp)
            if r1 == h - 1:
                bottom.append(comp)

    if len(left) == 1 and len(right) == 1 and not top and not bottom:
        return "vertical", [
            {"side": "left", **left[0]},
            {"side": "right", **right[0]},
        ]
    if len(top) == 1 and len(bottom) == 1 and not left and not right:
        return "horizontal", [
            {"side": "top", **top[0]},
            {"side": "bottom", **bottom[0]},
        ]
    raise AssertionError("expected exactly one opposite wall pair")


def _extract_notches(grid: np.ndarray, wall: dict) -> list[dict]:
    r0, c0, r1, c1 = wall["bbox"]
    sub = _crop_bbox(grid, wall["bbox"])
    notch_mask = sub != int(wall["color"])
    pieces = []
    if not np.any(notch_mask):
        return pieces

    for coords in _components_mask(notch_mask, connectivity=4):
        local_bbox = _bbox_coords(coords)
        lr0, lc0, _, _ = local_bbox
        hmask, _ = _mask_from_coords(coords)

        if wall["side"] == "left":
            assert np.any(coords[:, 1] == sub.shape[1] - 1)
        elif wall["side"] == "right":
            assert np.any(coords[:, 1] == 0)
        elif wall["side"] == "top":
            assert np.any(coords[:, 0] == sub.shape[0] - 1)
        else:
            assert np.any(coords[:, 0] == 0)

        pieces.append(
            {
                "side": wall["side"],
                "wall_color": int(wall["color"]),
                "bbox": (r0 + lr0, c0 + lc0, r0 + local_bbox[2], c0 + local_bbox[3]),
                "mask": hmask,
                "key": _mask_key(hmask),
                "top_left": (r0 + lr0, c0 + lc0),
            }
        )
    return pieces


def _extract_sprites(grid: np.ndarray, bg: int, walls: list[dict]) -> list[dict]:
    work = np.asarray(grid, dtype=int).copy()
    for wall in walls:
        r0, c0, r1, c1 = wall["bbox"]
        work[r0 : r1 + 1, c0 : c1 + 1] = int(bg)

    sprites = []
    for comp in _find_components(work, bg=bg, connectivity=4):
        mask, bbox = _mask_from_coords(comp["coords"])
        sprites.append(
            {
                "color": int(comp["color"]),
                "bbox": bbox,
                "mask": mask,
                "key": _mask_key(mask),
            }
        )
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each scattered sprite into a shape-complement notch so both opposite walls become flat inlaid strips.

    Input structure:
    - The background is the dominant floor color.
    - Exactly two opposite border strips are thick monochrome walls; their inner faces contain background-colored notches.
    - The central floor contains the same number of separated monochrome sprites as notches, and sprites never need rotation or reflection.

    Rule:
    - Detect the two edge walls and read every notch as a cropped boolean shape inside its wall strip.
    - Remove all floor sprites, then match sprites to notches by exact translated shape equality; every sprite is used once and every notch is filled once.
    - Repaint each wall as a solid strip, then fill every notch with the color of its matching sprite; the floor returns to background.

    Key insight:
    - The visible motion is a distraction: the real signal is exact notch/sprite shape complement across the whole set, not nearest-wall or straight-line contact.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    orientation, walls = _detect_walls(g, bg)
    assert orientation in {"vertical", "horizontal"}

    notches = []
    for wall in walls:
        notches.extend(_extract_notches(g, wall))
    assert notches, "expected at least one notch"

    sprites = _extract_sprites(g, bg, walls)
    assert len(sprites) == len(notches), "sprite/notch counts must match"

    notches_by_key: dict[tuple[tuple[int, int], bytes], list[dict]] = defaultdict(list)
    sprites_by_key: dict[tuple[tuple[int, int], bytes], list[dict]] = defaultdict(list)
    for notch in notches:
        notches_by_key[notch["key"]].append(notch)
    for sprite in sprites:
        sprites_by_key[sprite["key"]].append(sprite)

    assert set(notches_by_key) == set(sprites_by_key), "every notch shape must have matching sprites"
    out = np.full_like(g, bg)
    for wall in walls:
        r0, c0, r1, c1 = wall["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(wall["color"])

    for key in sorted(notches_by_key, key=lambda k: (k[0][0] * 31 + k[0][1], k[1])):
        group_notches = notches_by_key[key]
        group_sprites = sprites_by_key[key]
        assert len(group_notches) == len(group_sprites), "shape multiplicities must match"
        colors = {int(sprite["color"]) for sprite in group_sprites}
        assert len(colors) == 1, "ambiguous same-shape sprite colors would make the output non-unique"
        fill_color = next(iter(colors))
        for notch in group_notches:
            patch = np.full(notch["mask"].shape, fill_color, dtype=int)
            _paste(out, patch, notch["top_left"][0], notch["top_left"][1], mask=notch["mask"])
    return out


def _pick_palette(n_groups: int) -> tuple[int, int, int, list[int]]:
    colors = list(range(10))
    bg = random.choice(colors)
    remain = [c for c in colors if c != bg]
    random.shuffle(remain)
    wall_a = remain.pop()
    if random.random() < 0.35:
        wall_b = wall_a
    else:
        wall_b = remain.pop()
    sprite_cols = remain[:n_groups]
    return bg, wall_a, wall_b, sprite_cols


def _oriented_shapes(orientation: str, max_depth: int) -> list[np.ndarray]:
    shapes = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for base in SHAPE_LIBRARY:
        shape = base if orientation == "vertical" else np.rot90(base)
        key = _mask_key(shape)
        if key in seen:
            continue
        seen.add(key)
        if orientation == "vertical":
            if shape.shape[1] < max_depth and shape.shape[0] <= 4:
                shapes.append(shape.astype(bool))
        else:
            if shape.shape[0] < max_depth and shape.shape[1] <= 4:
                shapes.append(shape.astype(bool))
    return shapes


def _choose_canvas() -> dict:
    if random.random() < 0.5:
        orientation = "vertical"
        h = random.randint(14, 24)
        t1 = random.randint(4, 5)
        t2 = random.randint(4, 5)
        floor = random.randint(t1 + t2 + 1, min(18, 30 - t1 - t2))
        w = t1 + t2 + floor
        return {"orientation": orientation, "h": h, "w": w, "a": t1, "b": t2}
    orientation = "horizontal"
    w = random.randint(14, 24)
    t1 = random.randint(4, 5)
    t2 = random.randint(4, 5)
    floor = random.randint(t1 + t2 + 1, min(18, 30 - t1 - t2))
    h = t1 + t2 + floor
    return {"orientation": orientation, "h": h, "w": w, "a": t1, "b": t2}


def _group_counts(n_groups: int) -> list[int]:
    counts = [1] * n_groups
    repeated = random.sample(range(n_groups), k=random.randint(1, min(2, n_groups - 1)))
    for idx in repeated:
        counts[idx] = random.randint(2, 3)
    return counts


def _axis_span(shape: np.ndarray, orientation: str) -> int:
    return int(shape.shape[0] if orientation == "vertical" else shape.shape[1])


def _place_on_side(
    instances: list[dict],
    *,
    axis_len: int,
    orientation: str,
) -> list[dict] | None:
    occupied = np.zeros(axis_len, dtype=bool)
    placed: list[dict] = []
    order = sorted(instances, key=lambda item: (_axis_span(item["shape"], orientation), random.random()), reverse=True)
    for inst in order:
        span = _axis_span(inst["shape"], orientation)
        candidates = []
        for start in range(1, axis_len - span):
            lo = max(0, start - 1)
            hi = min(axis_len, start + span + 1)
            if not occupied[lo:hi].any():
                candidates.append(start)
        if not candidates:
            return None
        start = random.choice(candidates)
        occupied[start : start + span] = True
        placed.append({**inst, "axis_start": int(start)})
    return placed


def _build_notch_layout(spec: dict, shapes: list[np.ndarray], counts: list[int], group_colors: list[int]) -> dict | None:
    orientation = spec["orientation"]
    axis_len = spec["h"] if orientation == "vertical" else spec["w"]
    sides = ("left", "right") if orientation == "vertical" else ("top", "bottom")

    groups = []
    unique_groups = [idx for idx, count in enumerate(counts) if count == 1]
    if not unique_groups:
        return None
    decoy_group = random.choice(unique_groups)

    for gid, (shape, count, color) in enumerate(zip(shapes, counts, group_colors)):
        groups.append({"gid": gid, "shape": shape, "count": count, "color": color})

    instances = []
    for group in groups:
        if group["count"] == 1:
            side_choices = [random.choice(sides)]
        else:
            if random.random() < 0.5:
                first = random.choice(sides)
                side_choices = [first] + [sides[1] if first == sides[0] else sides[0]] + [
                    random.choice(sides) for _ in range(group["count"] - 2)
                ]
            else:
                side_choices = [random.choice(sides) for _ in range(group["count"])]
        for sid in side_choices:
            instances.append(
                {
                    "gid": group["gid"],
                    "shape": group["shape"],
                    "color": group["color"],
                    "side": sid,
                    "is_decoy": group["gid"] == decoy_group,
                }
            )

    if not any(inst["side"] == sides[0] for inst in instances) or not any(inst["side"] == sides[1] for inst in instances):
        return None

    by_side = {side: [] for side in sides}
    for inst in instances:
        by_side[inst["side"]].append(inst)

    placed_all = []
    for side in sides:
        placed_side = _place_on_side(by_side[side], axis_len=axis_len, orientation=orientation)
        if placed_side is None:
            return None
        placed_all.extend(placed_side)

    return {"instances": placed_all, "decoy_group": decoy_group}


def _notch_top_left(spec: dict, side: str, shape: np.ndarray, axis_start: int) -> Coord:
    h, w, a, b = spec["h"], spec["w"], spec["a"], spec["b"]
    if spec["orientation"] == "vertical":
        if side == "left":
            return axis_start, a - shape.shape[1]
        return axis_start, w - b
    if side == "top":
        return a - shape.shape[0], axis_start
    return h - b, axis_start


def _render_output(spec: dict, bg: int, wall_a: int, wall_b: int, layout: dict) -> tuple[np.ndarray, list[dict]]:
    h, w, a, b = spec["h"], spec["w"], spec["a"], spec["b"]
    out = np.full((h, w), bg, dtype=int)
    placements: list[dict] = []

    if spec["orientation"] == "vertical":
        out[:, :a] = wall_a
        out[:, w - b :] = wall_b
    else:
        out[:a, :] = wall_a
        out[h - b :, :] = wall_b

    for inst in layout["instances"]:
        shape = inst["shape"]
        r0, c0 = _notch_top_left(spec, inst["side"], shape, inst["axis_start"])
        patch = np.full(shape.shape, int(inst["color"]), dtype=int)
        _paste(out, patch, r0, c0, mask=shape)
        placements.append({**inst, "top_left": (r0, c0)})
    return out, placements


def _floor_bounds(spec: dict) -> tuple[int, int, int, int]:
    h, w, a, b = spec["h"], spec["w"], spec["a"], spec["b"]
    if spec["orientation"] == "vertical":
        return 0, a, h, w - b
    return a, 0, h - b, w


def _can_place_sprite(occupied: np.ndarray, floor: tuple[int, int, int, int], shape: np.ndarray, r0: int, c0: int) -> bool:
    fr0, fc0, fr1, fc1 = floor
    h, w = shape.shape
    if r0 < fr0 or c0 < fc0 or r0 + h > fr1 or c0 + w > fc1:
        return False
    for dr, dc in np.argwhere(shape):
        rr, cc = r0 + int(dr), c0 + int(dc)
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if 0 <= nr < occupied.shape[0] and 0 <= nc < occupied.shape[1] and occupied[nr, nc]:
                    return False
    return True


def _place_sprite_patch(inp: np.ndarray, occupied: np.ndarray, shape: np.ndarray, color: int, floor: tuple[int, int, int, int], desired: Coord | None = None) -> Coord | None:
    fr0, fc0, fr1, fc1 = floor
    candidates = []
    h, w = shape.shape
    for r0 in range(fr0, fr1 - h + 1):
        for c0 in range(fc0, fc1 - w + 1):
            if _can_place_sprite(occupied, floor, shape, r0, c0):
                candidates.append((r0, c0))
    if not candidates:
        return None

    if desired is None:
        pos = random.choice(candidates)
    else:
        pos = min(candidates, key=lambda rc: (abs(rc[0] - desired[0]) + abs(rc[1] - desired[1]), random.random()))

    patch = np.full(shape.shape, _get_bg(inp), dtype=int)
    patch[shape] = int(color)
    _paste(inp, patch, pos[0], pos[1], mask=shape)
    for dr, dc in np.argwhere(shape):
        occupied[pos[0] + int(dr), pos[1] + int(dc)] = True
    return pos


def _decoy_target(spec: dict, notch: dict) -> Coord:
    fr0, fc0, fr1, fc1 = _floor_bounds(spec)
    shape = notch["shape"]
    nr, nc = notch["top_left"]
    if spec["orientation"] == "vertical":
        if notch["side"] == "left":
            return nr, max(fc0, fc1 - shape.shape[1] - 1)
        return nr, min(fc1 - shape.shape[1], fc0 + 1)
    if notch["side"] == "top":
        return max(fr0, fr1 - shape.shape[0] - 1), nc
    return min(fr1 - shape.shape[0], fr0 + 1), nc


def _render_input(spec: dict, out: np.ndarray, placements: list[dict], bg: int, decoy_group: int) -> tuple[np.ndarray, dict] | None:
    inp = out.copy()
    for notch in placements:
        r0, c0 = notch["top_left"]
        patch = np.full(notch["shape"].shape, bg, dtype=int)
        _paste(inp, patch, r0, c0, mask=notch["shape"])

    floor = _floor_bounds(spec)
    occupied = np.zeros(inp.shape, dtype=bool)
    decoy = next((n for n in placements if n["gid"] == decoy_group), None)
    assert decoy is not None

    sprite_order = placements[:]
    random.shuffle(sprite_order)
    sprite_order.sort(key=lambda item: (item["gid"] != decoy_group, -int(item["shape"].sum()), random.random()))

    sprite_positions = []
    for notch in sprite_order:
        desired = _decoy_target(spec, notch) if notch["gid"] == decoy_group else None
        pos = _place_sprite_patch(inp, occupied, notch["shape"], notch["color"], floor, desired=desired)
        if pos is None:
            return None
        sprite_positions.append({**notch, "sprite_top_left": pos})
    return inp, {"sprites": sprite_positions, "decoy_group": decoy_group}


def _distance_to_side(spec: dict, side: str, sprite: dict) -> int:
    r0, c0 = sprite["sprite_top_left"]
    shape = sprite["shape"]
    if spec["orientation"] == "vertical":
        if side == "left":
            return c0 - spec["a"]
        return (spec["w"] - spec["b"]) - (c0 + shape.shape[1])
    if side == "top":
        return r0 - spec["a"]
    return (spec["h"] - spec["b"]) - (r0 + shape.shape[0])


def _semantic_ok(inp: np.ndarray, out: np.ndarray, spec: dict, placements: list[dict], sprite_meta: dict) -> bool:
    if not np.array_equal(main(inp), out):
        return False

    counts = Counter(item["gid"] for item in placements)
    if not any(count > 1 for count in counts.values()):
        return False
    if not any(count == 1 for count in counts.values()):
        return False

    sides = Counter(item["side"] for item in placements)
    if len(sides) != 2 or min(sides.values()) < 1:
        return False

    decoy_group = sprite_meta["decoy_group"]
    decoy_sprite = next(sprite for sprite in sprite_meta["sprites"] if sprite["gid"] == decoy_group)
    if spec["orientation"] == "vertical":
        other_side = "right" if decoy_sprite["side"] == "left" else "left"
    else:
        other_side = "bottom" if decoy_sprite["side"] == "top" else "top"

    if not (_distance_to_side(spec, other_side, decoy_sprite) < _distance_to_side(spec, decoy_sprite["side"], decoy_sprite)):
        return False

    solved = main(inp)
    colors_in_floor = set(np.unique(inp[_floor_bounds(spec)[0] : _floor_bounds(spec)[2], _floor_bounds(spec)[1] : _floor_bounds(spec)[3]])) - {_get_bg(inp)}
    if len(colors_in_floor) < 3:
        return False
    if np.array_equal(solved, np.full_like(solved, _get_bg(inp))):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    spec = _choose_canvas()
    n_groups = random.randint(3, 4)
    bg, wall_a, wall_b, sprite_cols = _pick_palette(n_groups)

    shapes = _oriented_shapes(spec["orientation"], max_depth=min(spec["a"], spec["b"]))
    if len(shapes) < n_groups:
        return None
    chosen_shapes = random.sample(shapes, n_groups)
    counts = _group_counts(n_groups)

    layout = _build_notch_layout(spec, chosen_shapes, counts, sprite_cols)
    if layout is None:
        return None

    out, placements = _render_output(spec, bg, wall_a, wall_b, layout)
    rendered = _render_input(spec, out, placements, bg, layout["decoy_group"])
    if rendered is None:
        return None
    inp, sprite_meta = rendered

    if not _semantic_ok(inp, out, spec, placements, sprite_meta):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-notch inlay input")
