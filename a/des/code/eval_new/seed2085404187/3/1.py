import random
from collections import Counter, deque

import numpy as np


BG = 0
WALL = 5
SPRITE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match outside sprites to rooms by color keys, then project each sprite through the matched gap.

    Input structure:
    - The background is black, and one shared wall color forms several one-pixel-thick rectangular rooms.
    - Every room border is complete except for one single-cell gap on one side.
    - Each room has one monochrome sprite outside its own gap, and a separate monochrome key block inside some room.
    - Sprite colors are unique; each key shares its color with exactly one outside sprite.

    Rule:
    - Detect the room wall color geometrically and parse every room's bbox, interior, and gap direction.
    - For each room, the outside sprite touching the cell immediately beyond its gap defines that sprite's anchor cell.
    - The key inside a room chooses which colored sprite must enter that room, regardless of which room the sprite started beside.
    - Remove all keys and original outside sprites, then translate the chosen sprite into the keyed room.
    - The sprite is not rotated: its anchor keeps the matched gap's row/column alignment, and the whole shape slides straight to the opposite inner wall in the gap's direction.

    Key insight:
    - The task has two separate links: color keys decide the room-to-sprite pairing, while each target room's gap direction decides how the matched sprite is pushed across the room.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    wall_color, rooms = _find_rooms(g, bg)
    objects = _find_colored_components(g, exclude={bg, wall_color})

    object_at = {}
    for obj in objects:
        for cell in obj["coords"]:
            object_at[cell] = obj

    sprite_by_color = {}
    used_sprite_ids = set()
    for room in rooms:
        anchor = _outside_anchor(room)
        assert anchor in object_at, "gap-adjacent sprite anchor must exist"
        sprite = object_at[anchor]
        sprite_id = id(sprite)
        assert sprite_id not in used_sprite_ids, "each outside anchor should belong to a distinct sprite"
        used_sprite_ids.add(sprite_id)
        room["source_sprite"] = sprite
        room["source_anchor"] = (
            anchor[0] - sprite["bbox"][0],
            anchor[1] - sprite["bbox"][1],
        )
        sprite_by_color[sprite["color"]] = {
            "mask": sprite["mask"],
            "color": sprite["color"],
            "anchor": room["source_anchor"],
        }

    used_key_ids = set()
    for room in rooms:
        keys = []
        ir0, ic0, ir1, ic1 = room["interior"]
        for obj in objects:
            oid = id(obj)
            if oid in used_sprite_ids:
                continue
            if all(ir0 <= r <= ir1 and ic0 <= c <= ic1 for r, c in obj["coords"]):
                keys.append(obj)
        assert len(keys) == 1, "each room must contain exactly one key block"
        key = keys[0]
        used_key_ids.add(id(key))
        room["key_color"] = key["color"]
        room["key"] = key

    out = g.copy()
    for room in rooms:
        for r, c in room["source_sprite"]["coords"]:
            out[r, c] = bg
        for r, c in room["key"]["coords"]:
            out[r, c] = bg

    for room in rooms:
        sprite = sprite_by_color[room["key_color"]]
        mask = sprite["mask"]
        sh, sw = mask.shape
        ar, ac = sprite["anchor"]
        r0, c0, r1, c1 = room["bbox"]
        gr, gc = room["gap"]
        side = room["side"]
        if side == "top":
            dest_r0 = r1 - sh
            dest_c0 = gc - ac
        elif side == "bottom":
            dest_r0 = r0 + 1
            dest_c0 = gc - ac
        elif side == "left":
            dest_r0 = gr - ar
            dest_c0 = c1 - sw
        else:
            dest_r0 = gr - ar
            dest_c0 = c0 + 1

        ir0, ic0, ir1, ic1 = room["interior"]
        for rr, cc in np.argwhere(mask):
            r = int(dest_r0 + rr)
            c = int(dest_c0 + cc)
            assert ir0 <= r <= ir1 and ic0 <= c <= ic1, "projected sprite must stay inside the room"
            assert out[r, c] == bg, "projected sprite should only occupy cleared interior cells"
            out[r, c] = sprite["color"]

    return out


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts = []
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
            parts.append(coords)
    return parts


def _bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _parse_room(grid: np.ndarray, color: int, coords, bg: int):
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    cells = set(coords)
    if any(cell not in border for cell in cells):
        return None
    missing = sorted(border - cells)
    if len(missing) != 1:
        return None
    gap = missing[0]
    gr, gc = gap
    if (gr in (r0, r1)) and (gc in (c0, c1)):
        return None
    if gr == r0:
        side = "top"
    elif gr == r1:
        side = "bottom"
    elif gc == c0:
        side = "left"
    else:
        side = "right"
    interior = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    if interior[0] > interior[2] or interior[1] > interior[3]:
        return None
    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "side": side,
        "interior": interior,
    }


def _find_rooms(grid: np.ndarray, bg: int):
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    candidates = []
    for color in colors:
        comps = _components(grid == color)
        if not comps:
            continue
        rooms = []
        ok = True
        for comp in comps:
            room = _parse_room(grid, color, comp, bg)
            if room is None:
                ok = False
                break
            rooms.append(room)
        if ok:
            candidates.append((len(rooms), color, rooms))
    assert candidates, "failed to identify room wall color"
    candidates.sort(key=lambda item: (-item[0], item[1]))
    _, wall_color, rooms = candidates[0]
    rooms.sort(key=lambda room: room["bbox"])
    return wall_color, rooms


def _crop_mask(coords, bbox):
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _find_colored_components(grid: np.ndarray, exclude):
    objects = []
    for color in [int(c) for c in np.unique(grid) if int(c) not in exclude]:
        for coords in _components(grid == color):
            box = _bbox(coords)
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": box,
                    "mask": _crop_mask(coords, box),
                }
            )
    return objects


def _outside_anchor(room):
    r0, c0, r1, c1 = room["bbox"]
    gr, gc = room["gap"]
    if room["side"] == "top":
        return gr - 1, gc
    if room["side"] == "bottom":
        return gr + 1, gc
    if room["side"] == "left":
        return gr, gc - 1
    return gr, gc + 1


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-projection input")


def _gen_attempt(rng: random.Random):
    room_count = rng.randint(2, 4)
    colors = rng.sample(SPRITE_COLORS, room_count)
    source_sides = list(SIDES)
    rng.shuffle(source_sides)
    source_sides = source_sides[:room_count]

    room_sizes = [
        (rng.randint(4, 7), rng.randint(4, 7))
        for _ in range(room_count)
    ]
    target_for_sprite = _derangement(room_count, rng)
    key_color_in_room = [None] * room_count
    for sprite_idx, room_idx in enumerate(target_for_sprite):
        key_color_in_room[room_idx] = colors[sprite_idx]

    sprites = []
    for idx in range(room_count):
        target_idx = target_for_sprite[idx]
        max_h = min(room_sizes[idx][0], room_sizes[target_idx][0], 5)
        max_w = min(room_sizes[idx][1], room_sizes[target_idx][1], 5)
        sprite = _random_sprite(source_sides[idx], max_h, max_w, rng)
        sprites.append(sprite)

    patches = []
    for idx in range(room_count):
        incoming_idx = colors.index(key_color_in_room[idx])
        gap_offset = _shared_gap_offset(
            room_h=room_sizes[idx][0],
            room_w=room_sizes[idx][1],
            side=source_sides[idx],
            source_sprite=sprites[idx],
            incoming_sprite=sprites[incoming_idx],
            rng=rng,
        )
        if gap_offset is None:
            return None
        patch = _room_patch(
            room_h=room_sizes[idx][0],
            room_w=room_sizes[idx][1],
            side=source_sides[idx],
            sprite=sprites[idx],
            sprite_color=colors[idx],
            key_color=key_color_in_room[idx],
            gap_offset=gap_offset,
            rng=rng,
        )
        if patch is None:
            return None
        patches.append(patch)

    return _layout_patches(patches, rng)


def _derangement(n: int, rng: random.Random):
    if n == 2:
        return [1, 0]
    perm = list(range(n))
    for _ in range(200):
        rng.shuffle(perm)
        if all(i != perm[i] for i in range(n)):
            return perm[:]
    raise RuntimeError("failed to sample a derangement")


def _random_sprite(side: str, max_h: int, max_w: int, rng: random.Random):
    for _ in range(80):
        h = rng.randint(1, max_h)
        w = rng.randint(1, max_w)
        if side == "top":
            anchor = (h - 1, rng.randrange(w))
        elif side == "bottom":
            anchor = (0, rng.randrange(w))
        elif side == "left":
            anchor = (rng.randrange(h), w - 1)
        else:
            anchor = (rng.randrange(h), 0)

        area_cap = min(h * w, 6)
        if h * w == 1:
            target_area = 1
        else:
            target_area = rng.randint(2, area_cap)
        cells = {anchor}
        while len(cells) < target_area:
            frontier = []
            for r, c in list(cells):
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        rows = sorted({r for r, _ in cells})
        cols = sorted({c for _, c in cells})
        mask = np.zeros((rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1), dtype=bool)
        for r, c in cells:
            mask[r - rows[0], c - cols[0]] = True
        anchor2 = (anchor[0] - rows[0], anchor[1] - cols[0])
        if side == "top" and anchor2[0] != mask.shape[0] - 1:
            continue
        if side == "bottom" and anchor2[0] != 0:
            continue
        if side == "left" and anchor2[1] != mask.shape[1] - 1:
            continue
        if side == "right" and anchor2[1] != 0:
            continue
        return {"mask": mask, "anchor": anchor2}
    raise RuntimeError("failed to build a sprite")


def _shared_gap_offset(room_h: int, room_w: int, side: str, source_sprite, incoming_sprite, rng: random.Random):
    if side in ("top", "bottom"):
        s_low, s_high = _gap_offset_range(room_w, source_sprite["mask"].shape[1], source_sprite["anchor"][1])
        t_low, t_high = _gap_offset_range(room_w, incoming_sprite["mask"].shape[1], incoming_sprite["anchor"][1])
    else:
        s_low, s_high = _gap_offset_range(room_h, source_sprite["mask"].shape[0], source_sprite["anchor"][0])
        t_low, t_high = _gap_offset_range(room_h, incoming_sprite["mask"].shape[0], incoming_sprite["anchor"][0])
    low = max(s_low, t_low)
    high = min(s_high, t_high)
    if low > high:
        return None
    return rng.randint(low, high)


def _gap_offset_range(interior_len: int, sprite_len: int, anchor_pos: int):
    return anchor_pos, interior_len - sprite_len + anchor_pos


def _room_patch(
    room_h: int,
    room_w: int,
    side: str,
    sprite,
    sprite_color: int,
    key_color: int,
    gap_offset: int,
    rng: random.Random,
):
    mask = sprite["mask"]
    sh, sw = mask.shape
    ar, ac = sprite["anchor"]

    top_margin = 1 + (sh if side == "top" else 0)
    bottom_margin = 1 + (sh if side == "bottom" else 0)
    left_margin = 1 + (sw if side == "left" else 0)
    right_margin = 1 + (sw if side == "right" else 0)

    patch_h = top_margin + room_h + 2 + bottom_margin
    patch_w = left_margin + room_w + 2 + right_margin
    patch = np.full((patch_h, patch_w), BG, dtype=int)

    r0 = top_margin
    c0 = left_margin
    r1 = r0 + room_h + 1
    c1 = c0 + room_w + 1

    patch[r0:r1 + 1, c0] = WALL
    patch[r0:r1 + 1, c1] = WALL
    patch[r0, c0:c1 + 1] = WALL
    patch[r1, c0:c1 + 1] = WALL

    if side in ("top", "bottom"):
        gap_c = c0 + 1 + gap_offset
        gap_r = r0 if side == "top" else r1
        sprite_r0 = r0 - sh if side == "top" else r1 + 1
        sprite_c0 = gap_c - ac
    else:
        gap_r = r0 + 1 + gap_offset
        gap_c = c0 if side == "left" else c1
        sprite_r0 = gap_r - ar
        sprite_c0 = c0 - sw if side == "left" else c1 + 1
    patch[gap_r, gap_c] = BG

    patch[sprite_r0:sprite_r0 + sh, sprite_c0:sprite_c0 + sw][mask] = sprite_color

    key_mask = _random_key_mask(room_h, room_w, rng)
    kh, kw = key_mask.shape
    for _ in range(50):
        key_r0 = rng.randint(r0 + 1, r1 - kh)
        key_c0 = rng.randint(c0 + 1, c1 - kw)
        window = patch[key_r0:key_r0 + kh, key_c0:key_c0 + kw]
        if np.any(window[key_mask] != BG):
            continue
        window[key_mask] = key_color
        return patch
    return None


def _random_key_mask(room_h: int, room_w: int, rng: random.Random):
    choices = [np.array([[1]], dtype=bool)]
    if room_w >= 2:
        choices.append(np.array([[1, 1]], dtype=bool))
    if room_h >= 2:
        choices.append(np.array([[1], [1]], dtype=bool))
    return rng.choice(choices)


def _layout_patches(patches, rng: random.Random):
    n = len(patches)
    orders = []
    base = list(range(n))
    for _ in range(6):
        ids = base[:]
        rng.shuffle(ids)
        orders.append(ids)

    for ids in orders:
        row_splits = [None] + list(range(1, n))
        rng.shuffle(row_splits)
        for split in row_splits:
            rows = [ids] if split is None else [ids[:split], ids[split:]]
            top_margin = rng.randint(0, 3)
            bottom_margin = rng.randint(0, 3)
            left_margin = rng.randint(0, 3)
            right_margin = rng.randint(0, 3)
            row_gap = rng.randint(1, 3) if len(rows) == 2 else 0

            placements = []
            y = top_margin
            total_w = 0
            for row in rows:
                gap = rng.randint(1, 3)
                indent = rng.randint(0, 2)
                row_h = max(patches[i].shape[0] for i in row)
                row_w = indent + sum(patches[i].shape[1] for i in row) + gap * (len(row) - 1)
                total_w = max(total_w, row_w)
                x = left_margin + indent
                for idx in row:
                    ph, pw = patches[idx].shape
                    y_off = rng.randint(0, row_h - ph)
                    placements.append((idx, y + y_off, x))
                    x += pw + gap
                y += row_h + row_gap

            total_h = y - row_gap + bottom_margin
            total_w = left_margin + total_w + right_margin
            if total_h > 30 or total_w > 30:
                continue

            grid = np.full((total_h, total_w), BG, dtype=int)
            for idx, r0, c0 in placements:
                patch = patches[idx]
                ph, pw = patch.shape
                region = grid[r0:r0 + ph, c0:c0 + pw]
                if np.any(region[patch != BG] != BG):
                    break
                region[patch != BG] = patch[patch != BG]
            else:
                return grid
    return None
