import random
from collections import Counter

import numpy as np


SIDES = ("top", "right", "bottom", "left")
NEXT_SIDE = {"top": "right", "right": "bottom", "bottom": "left", "left": "top"}
WALL_THICKNESS = 5
GEN_RETRIES = 300
PLACEMENT_TRIES = 200


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _components_from_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [
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
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in nbrs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray, bg: int, connectivity: int = 4) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in nbrs:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_from_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _complement_crop(sprite_mask: np.ndarray) -> np.ndarray:
    comp = ~sprite_mask
    assert np.any(comp), "matched sprite must leave a non-empty negative shape"
    return _tight_crop(comp)


def _side_of_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    h, w = shape
    if r0 == 0:
        return "top"
    if r1 == h - 1:
        return "bottom"
    if c0 == 0:
        return "left"
    if c1 == w - 1:
        return "right"
    raise AssertionError("wall component must touch exactly one border")


def _extract_walls(grid: np.ndarray, bg: int) -> dict[str, dict]:
    comps = _monochrome_components(grid, bg=bg, connectivity=4)
    walls: dict[str, dict] = {}
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        if r0 != 0 and c0 != 0 and r1 != grid.shape[0] - 1 and c1 != grid.shape[1] - 1:
            continue
        side = _side_of_bbox(comp["bbox"], grid.shape)
        if side in walls:
            if comp["area"] > walls[side]["area"]:
                walls[side] = comp
        else:
            walls[side] = comp
    assert set(walls) == set(SIDES), "expected four colored wall components"
    return walls


def _extract_notches(grid: np.ndarray, wall: dict, bg: int) -> list[dict]:
    r0, c0, r1, c1 = wall["bbox"]
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    comps = _components_from_mask(crop == bg, connectivity=4)
    notches = []
    for coords in comps:
        local_bbox = _bbox_from_coords(coords)
        lr0, lc0, lr1, lc1 = local_bbox
        mask = np.zeros((lr1 - lr0 + 1, lc1 - lc0 + 1), dtype=bool)
        rel = coords - np.array([lr0, lc0], dtype=int)
        mask[rel[:, 0], rel[:, 1]] = True
        global_coords = coords + np.array([r0, c0], dtype=int)
        notches.append(
            {
                "coords": global_coords,
                "bbox": _bbox_from_coords(global_coords),
                "mask": mask,
                "key": _mask_key(mask),
            }
        )
    return notches


def _extract_sprites(grid: np.ndarray, walls: dict[str, dict], bg: int) -> list[dict]:
    work = np.array(grid, copy=True)
    for wall in walls.values():
        r0, c0, r1, c1 = wall["bbox"]
        work[r0 : r1 + 1, c0 : c1 + 1] = bg
    comps = _monochrome_components(work, bg=bg, connectivity=4)
    sprites = []
    for comp in comps:
        mask = _mask_from_coords(comp["coords"])
        sprites.append({**comp, "mask": mask, "key": _mask_key(mask)})
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Send each wall-colored sprite back to its home wall by filling the uniquely complementary notch and erasing the sprite.

    Input structure:
    - The background is the dominant color.
    - Four thick monochrome walls sit on the top, right, bottom, and left edges; the corner squares stay background.
    - Each wall contains several background notches, and the interior contains several solid monochrome sprites.

    Rule:
    - Identify the four wall colors from the border components.
    - For every interior sprite whose color matches one wall, compute the tight crop of the sprite's negative shape inside the sprite's own bounding box.
    - On that same-colored wall, find the unique notch whose shape equals that negative crop, fill that notch with the sprite color, and erase the sprite from its original location.
    - Sprites whose color matches no wall remain unchanged, and unmatched notches stay background.

    Key insight:
    - Shape alone is not enough: the correct notch is the sprite's cropped negative shape, and color restricts which wall is allowed to receive it.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    walls = _extract_walls(g, bg)
    wall_by_color = {int(wall["color"]): side for side, wall in walls.items()}
    notches_by_side = {side: _extract_notches(g, wall, bg) for side, wall in walls.items()}
    sprites = _extract_sprites(g, walls, bg)

    out = np.array(g, copy=True)
    for sprite in sprites:
        side = wall_by_color.get(int(sprite["color"]))
        if side is None:
            continue
        target_mask = _complement_crop(sprite["mask"])
        target_key = _mask_key(target_mask)
        matches = [notch for notch in notches_by_side[side] if notch["key"] == target_key]
        assert len(matches) == 1, "wall-colored sprite must have exactly one complementary notch on its wall"
        for r, c in sprite["coords"]:
            out[r, c] = bg
        for r, c in matches[0]["coords"]:
            out[r, c] = int(sprite["color"])
    return out


def _build_shape_library() -> list[dict]:
    entries = []
    for h, w in ((2, 3), (3, 2), (3, 3)):
        for bits in range(1, 1 << (h * w)):
            mask = np.array([(bits >> i) & 1 for i in range(h * w)], dtype=bool).reshape(h, w)
            if np.all(mask) or not np.any(mask):
                continue
            if _tight_crop(mask).shape != mask.shape:
                continue
            if len(_components_from_mask(mask, connectivity=4)) != 1:
                continue
            comp = _complement_crop(mask)
            if len(_components_from_mask(comp, connectivity=4)) != 1:
                continue
            if int(np.sum(mask)) < 3 or int(np.sum(comp)) < 2:
                continue
            entries.append(
                {
                    "sprite": mask,
                    "comp": comp,
                    "sprite_key": _mask_key(mask),
                    "comp_key": _mask_key(comp),
                }
            )
    entries.sort(key=lambda x: (x["sprite"].shape, int(np.sum(x["sprite"])), int(np.sum(x["comp"]))))
    return entries


SHAPE_LIBRARY = _build_shape_library()


def _unique_comp_shapes() -> list[dict]:
    unique = []
    seen = set()
    for entry in SHAPE_LIBRARY:
        if entry["comp_key"] in seen:
            continue
        seen.add(entry["comp_key"])
        unique.append(entry)
    return unique


ACTIVE_SHAPES = _unique_comp_shapes()
NOTCH_LIBRARY = []
_notch_seen = set()
for _entry in SHAPE_LIBRARY:
    for _mask in (_entry["sprite"], _entry["comp"]):
        _key = _mask_key(_mask)
        if _key not in _notch_seen:
            _notch_seen.add(_key)
            NOTCH_LIBRARY.append(_mask.copy())


def _paint_walls(grid: np.ndarray, side_colors: dict[str, int], t: int) -> None:
    h, w = grid.shape
    grid[:t, t : w - t] = side_colors["top"]
    grid[h - t :, t : w - t] = side_colors["bottom"]
    grid[t : h - t, :t] = side_colors["left"]
    grid[t : h - t, w - t :] = side_colors["right"]


def _wall_bbox(side: str, h: int, w: int, t: int) -> tuple[int, int, int, int]:
    if side == "top":
        return 0, t, t - 1, w - t - 1
    if side == "bottom":
        return h - t, t, h - 1, w - t - 1
    if side == "left":
        return t, 0, h - t - 1, t - 1
    return t, w - t, h - t - 1, w - 1


def _coords_from_mask(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(mask)
    coords[:, 0] += top
    coords[:, 1] += left
    return coords


def _place_mask_on_uniform_region(
    region: np.ndarray,
    mask: np.ndarray,
    fill_color: int,
    base_color: int,
    tries: int = PLACEMENT_TRIES,
) -> tuple[int, int] | None:
    rh, rw = region.shape
    mh, mw = mask.shape
    if rh < mh + 2 or rw < mw + 2:
        return None
    rows = list(range(1, rh - mh))
    cols = list(range(1, rw - mw))
    random.shuffle(rows)
    random.shuffle(cols)
    for _ in range(tries):
        r = random.choice(rows)
        c = random.choice(cols)
        if not np.all(region[r - 1 : r + mh + 1, c - 1 : c + mw + 1] == base_color):
            continue
        patch = region[r : r + mh, c : c + mw]
        patch[mask] = fill_color
        return int(r), int(c)
    return None


def _choose_active_shapes(count: int) -> list[dict]:
    choices = random.sample(ACTIVE_SHAPES, count)
    comp_keys = [entry["comp_key"] for entry in choices]
    assert len(set(comp_keys)) == len(comp_keys)
    return choices


def _build_expected(grid: np.ndarray, bg: int, active: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for spec in active:
        for r, c in spec["sprite_coords"]:
            out[r, c] = bg
        for r, c in spec["target_coords"]:
            out[r, c] = spec["color"]
    return out


def _validate_structure(grid: np.ndarray, bg: int, expected: np.ndarray, active: list[dict]) -> None:
    produced = main(grid)
    assert np.array_equal(produced, expected), "solver output must match generated ground truth"

    walls = _extract_walls(grid, bg)
    wall_by_color = {int(wall["color"]): side for side, wall in walls.items()}
    notches_by_side = {side: _extract_notches(grid, wall, bg) for side, wall in walls.items()}
    global_counts = Counter(notch["key"] for side in SIDES for notch in notches_by_side[side])

    for spec in active:
        side = wall_by_color[spec["color"]]
        assert side == spec["side"]
        local_notches = notches_by_side[side]
        comp_matches = [n for n in local_notches if n["key"] == spec["comp_key"]]
        same_shape = [n for n in local_notches if n["key"] == spec["sprite_key"]]
        assert len(comp_matches) == 1, "same-color wall must contain exactly one complementary notch"
        assert len(same_shape) >= 1, "same-color wall must contain a positive-shape decoy"
        assert global_counts[spec["comp_key"]] >= 2, "ignore-color shortcut must be invalidated by another wall"

    before_counts = Counter(int(x) for x in grid.flatten())
    after_counts = Counter(int(x) for x in expected.flatten())
    moved = sum(len(spec["sprite_coords"]) for spec in active)
    assert moved > 0
    assert before_counts[bg] != after_counts[bg], "filling and erasing must change the canvas"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(24, 30)
        w = random.randint(24, 30)
        bg = random.randint(0, 9)
        palette = [c for c in range(10) if c != bg]
        wall_colors = random.sample(palette, 4)
        side_colors = dict(zip(SIDES, wall_colors))
        other_colors = [c for c in palette if c not in wall_colors]
        if not other_colors:
            continue

        grid = np.full((h, w), bg, dtype=int)
        _paint_walls(grid, side_colors, WALL_THICKNESS)

        active_count = 4 if random.random() < 0.55 else 3
        active_sides = random.sample(list(SIDES), active_count)
        active_shapes = _choose_active_shapes(active_count)
        active_specs = []
        for side, entry in zip(active_sides, active_shapes):
            active_specs.append(
                {
                    "side": side,
                    "color": side_colors[side],
                    "sprite_mask": entry["sprite"],
                    "comp_mask": entry["comp"],
                    "sprite_key": entry["sprite_key"],
                    "comp_key": entry["comp_key"],
                }
            )

        wall_plans: dict[str, list[dict]] = {side: [] for side in SIDES}
        local_target_keys = {side: set() for side in SIDES}
        for spec in active_specs:
            wall_plans[spec["side"]].append({"mask": spec["comp_mask"], "role": "target", "owner": spec})
            wall_plans[spec["side"]].append({"mask": spec["sprite_mask"], "role": "positive", "owner": spec})
            local_target_keys[spec["side"]].add(spec["comp_key"])
            decoy_side = NEXT_SIDE[spec["side"]]
            wall_plans[decoy_side].append({"mask": spec["comp_mask"], "role": "cross", "owner": spec})

        for side in SIDES:
            minimum = max(2, len(wall_plans[side]))
            target_count = random.randint(minimum, max(minimum, 4))
            fill_options = [
                mask
                for mask in NOTCH_LIBRARY
                if _mask_key(mask) not in local_target_keys[side]
            ]
            while len(wall_plans[side]) < target_count:
                wall_plans[side].append({"mask": random.choice(fill_options), "role": "filler", "owner": None})

        notch_records = {side: [] for side in SIDES}
        ok = True
        for side in SIDES:
            r0, c0, r1, c1 = _wall_bbox(side, h, w, WALL_THICKNESS)
            region = grid[r0 : r1 + 1, c0 : c1 + 1]
            plans = sorted(wall_plans[side], key=lambda x: int(np.sum(x["mask"])), reverse=True)
            random.shuffle(plans)
            plans.sort(key=lambda x: int(np.sum(x["mask"])), reverse=True)
            for plan in plans:
                pos = _place_mask_on_uniform_region(
                    region=region,
                    mask=plan["mask"],
                    fill_color=bg,
                    base_color=side_colors[side],
                )
                if pos is None:
                    ok = False
                    break
                pr, pc = pos
                coords = _coords_from_mask(plan["mask"], r0 + pr, c0 + pc)
                record = {
                    "side": side,
                    "role": plan["role"],
                    "owner": plan["owner"],
                    "mask": plan["mask"],
                    "key": _mask_key(plan["mask"]),
                    "coords": coords,
                }
                notch_records[side].append(record)
                if plan["role"] == "target":
                    plan["owner"]["target_coords"] = coords
            if not ok:
                break
        if not ok:
            continue

        interior = grid[WALL_THICKNESS : h - WALL_THICKNESS, WALL_THICKNESS : w - WALL_THICKNESS]
        sprite_specs = list(active_specs)
        unmatched_count = random.randint(1, min(2, len(other_colors)))
        unmatched_colors = random.sample(other_colors, unmatched_count)
        for color in unmatched_colors:
            entry = random.choice(SHAPE_LIBRARY)
            sprite_specs.append(
                {
                    "side": None,
                    "color": color,
                    "sprite_mask": entry["sprite"],
                    "comp_mask": entry["comp"],
                    "sprite_key": entry["sprite_key"],
                    "comp_key": entry["comp_key"],
                }
            )

        random.shuffle(sprite_specs)
        sprite_specs.sort(key=lambda x: int(np.sum(x["sprite_mask"])), reverse=True)
        for spec in sprite_specs:
            pos = _place_mask_on_uniform_region(
                region=interior,
                mask=spec["sprite_mask"],
                fill_color=spec["color"],
                base_color=bg,
            )
            if pos is None:
                ok = False
                break
            pr, pc = pos
            coords = _coords_from_mask(
                spec["sprite_mask"],
                WALL_THICKNESS + pr,
                WALL_THICKNESS + pc,
            )
            spec["sprite_coords"] = coords
        if not ok:
            continue

        expected = _build_expected(grid, bg, active_specs)
        try:
            _validate_structure(grid, bg, expected, active_specs)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
