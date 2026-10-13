from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("top", "right", "bottom", "left")
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _rotate_patch(patch: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return patch
    if side == "right":
        return np.rot90(patch, -1)
    if side == "bottom":
        return np.rot90(patch, 2)
    if side == "left":
        return np.rot90(patch, 1)
    raise ValueError(f"unknown side: {side}")


def _iter_border(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c1))
    if r1 > r0:
        for c in range(c1, c0 - 1, -1):
            cells.append((r1, c))
    if c1 > c0:
        for r in range(r1 - 1, r0, -1):
            cells.append((r, c0))
    return cells


def _validate_room(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    anchor: tuple[int, int],
    side: str,
) -> dict | None:
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    ar, ac = anchor
    if (ar in (r0, r1)) and (ac in (c0, c1)):
        return None

    anchor_color = int(grid[ar, ac])
    if anchor_color in (BLACK, GREY):
        return None

    seen_anchor = 0
    for rr, cc in _iter_border(r0, c0, r1, c1):
        value = int(grid[rr, cc])
        if (rr, cc) == anchor:
            seen_anchor += 1
            continue
        if value != GREY:
            return None
    if seen_anchor != 1:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or not np.all(interior == BLACK):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "anchor": anchor,
        "color": anchor_color,
        "side": side,
        "interior_shape": (r1 - r0 - 1, c1 - c0 - 1),
    }


def _room_from_anchor(grid: np.ndarray, r: int, c: int) -> dict | None:
    h, w = grid.shape
    up = r > 0 and int(grid[r - 1, c]) == GREY
    down = r + 1 < h and int(grid[r + 1, c]) == GREY
    left = c > 0 and int(grid[r, c - 1]) == GREY
    right = c + 1 < w and int(grid[r, c + 1]) == GREY

    if left and right and not up and not down:
        c0 = c
        while c0 > 0 and int(grid[r, c0 - 1]) == GREY:
            c0 -= 1
        c1 = c
        while c1 + 1 < w and int(grid[r, c1 + 1]) == GREY:
            c1 += 1
        if not (c0 < c < c1):
            return None

        extends_up = r > 0 and int(grid[r - 1, c0]) == GREY and int(grid[r - 1, c1]) == GREY
        extends_down = r + 1 < h and int(grid[r + 1, c0]) == GREY and int(grid[r + 1, c1]) == GREY
        if extends_up == extends_down:
            return None

        if extends_down:
            r0 = r
            r1 = r
            while r1 + 1 < h and int(grid[r1 + 1, c0]) == GREY and int(grid[r1 + 1, c1]) == GREY:
                r1 += 1
            return _validate_room(grid, r0, c0, r1, c1, (r, c), "top")

        r1 = r
        r0 = r
        while r0 > 0 and int(grid[r0 - 1, c0]) == GREY and int(grid[r0 - 1, c1]) == GREY:
            r0 -= 1
        return _validate_room(grid, r0, c0, r1, c1, (r, c), "bottom")

    if up and down and not left and not right:
        r0 = r
        while r0 > 0 and int(grid[r0 - 1, c]) == GREY:
            r0 -= 1
        r1 = r
        while r1 + 1 < h and int(grid[r1 + 1, c]) == GREY:
            r1 += 1
        if not (r0 < r < r1):
            return None

        extends_left = c > 0 and int(grid[r0, c - 1]) == GREY and int(grid[r1, c - 1]) == GREY
        extends_right = c + 1 < w and int(grid[r0, c + 1]) == GREY and int(grid[r1, c + 1]) == GREY
        if extends_left == extends_right:
            return None

        if extends_right:
            c0 = c
            c1 = c
            while c1 + 1 < w and int(grid[r0, c1 + 1]) == GREY and int(grid[r1, c1 + 1]) == GREY:
                c1 += 1
            return _validate_room(grid, r0, c0, r1, c1, (r, c), "left")

        c1 = c
        c0 = c
        while c0 > 0 and int(grid[r0, c0 - 1]) == GREY and int(grid[r1, c0 - 1]) == GREY:
            c0 -= 1
        return _validate_room(grid, r0, c0, r1, c1, (r, c), "right")

    return None


def _find_rooms(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    rooms: list[dict] = []
    used_anchors: set[tuple[int, int]] = set()
    used_colors: set[int] = set()

    for r in range(h):
        for c in range(w):
            value = int(grid[r, c])
            if value in (BLACK, GREY):
                continue
            grey_neighbors = [
                (nr, nc)
                for nr, nc in _neighbors4(r, c, h, w)
                if int(grid[nr, nc]) == GREY
            ]
            if len(grey_neighbors) != 2:
                continue
            room = _room_from_anchor(grid, r, c)
            if room is None:
                continue
            if room["anchor"] in used_anchors:
                continue
            if room["color"] in used_colors:
                raise AssertionError(f"duplicate room color {room['color']}")
            used_anchors.add(room["anchor"])
            used_colors.add(room["color"])
            rooms.append(room)

    assert rooms, "no valid rooms found"
    return rooms


def _extract_templates(grid: np.ndarray, rooms: list[dict]) -> dict[int, dict]:
    anchor_positions = {room["anchor"] for room in rooms}
    templates: dict[int, dict] = {}

    for room in rooms:
        color = int(room["color"])
        coords = [
            (r, c)
            for r, c in np.argwhere(grid == color)
            if (int(r), int(c)) not in anchor_positions
        ]
        assert coords, f"no template pixels found for color {color}"
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for rr, cc in coords:
            patch[rr - r0, cc - c0] = color
        templates[color] = {
            "bbox": (r0, c0, r1, c1),
            "patch": patch,
        }

    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each color-matched outside template into the room keyed by its border anchor.

    Input structure:
    - Black background contains several grey rectangular room borders whose interiors start empty.
    - Each room border has exactly one non-grey anchor pixel, and every anchor color is unique.
    - Outside the rooms, each anchor color appears again as a same-color rectangular template made of that color plus black cells.

    Rule:
    - Detect every room, read its anchor color, and locate the outside template with the same color.
    - The anchor side encodes the rotation: top keeps the template, right rotates it 90 degrees clockwise, bottom rotates it 180 degrees, and left rotates it 90 degrees counterclockwise.
    - Erase each outside template from the canvas, then place the rotated template inside the matching room interior, copying only its colored cells while keeping black cells black.

    Key insight:
    - Color identifies which template belongs to which room, while the anchor's side on the grey border independently specifies the template's orientation.
    """
    g = np.asarray(grid, dtype=int)
    rooms = _find_rooms(g)
    templates = _extract_templates(g, rooms)

    out = g.copy()
    for record in templates.values():
        r0, c0, r1, c1 = record["bbox"]
        patch = record["patch"]
        mask = patch != BLACK
        out[r0 : r1 + 1, c0 : c1 + 1][mask] = BLACK

    for room in rooms:
        color = int(room["color"])
        patch = templates[color]["patch"]
        rotated = _rotate_patch(patch, room["side"])
        expected_shape = tuple(room["interior_shape"])
        assert rotated.shape == expected_shape, (
            f"template shape {rotated.shape} does not fit room interior {expected_shape}"
        )

        r0, c0, r1, c1 = room["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[rotated != BLACK] = color

    return out


def _rotation_keys(mask: np.ndarray) -> set[tuple[tuple[int, int], bytes]]:
    keys: set[tuple[tuple[int, int], bytes]] = set()
    base = np.where(mask, 1, 0).astype(np.uint8)
    for k in range(4):
        rot = np.rot90(base, k)
        keys.add((rot.shape, rot.tobytes()))
    return keys


def _sample_template_mask(h: int, w: int) -> np.ndarray | None:
    area = h * w
    for _ in range(200):
        target = random.randint(max(2, area // 3), max(2, area - 1))
        cells = {(0, random.randrange(w))}
        frontier = set(cells)
        while len(cells) < target and frontier:
            fr, fc = random.choice(tuple(frontier))
            neighbors = []
            for nr, nc in ((fr - 1, fc), (fr + 1, fc), (fr, fc - 1), (fr, fc + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    neighbors.append((nr, nc))
            if neighbors:
                chosen = random.choice(neighbors)
                cells.add(chosen)
                frontier.add(chosen)
            else:
                frontier.remove((fr, fc))

        if len(cells) < 2:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for rr, cc in cells:
            mask[rr, cc] = True

        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if mask.all():
            continue
        if len(_rotation_keys(mask)) < 4:
            continue
        return mask
    return None


def _all_positions(
    occupied: np.ndarray,
    box_h: int,
    box_w: int,
    margin: int = 1,
    pad: int = 1,
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(margin, h - box_h - margin + 1):
        for c0 in range(margin, w - box_w - margin + 1):
            rr0 = max(0, r0 - pad)
            rr1 = min(h, r0 + box_h + pad)
            cc0 = max(0, c0 - pad)
            cc1 = min(w, c0 + box_w + pad)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _mark_box(occupied: np.ndarray, r0: int, c0: int, box_h: int, box_w: int, pad: int = 1) -> None:
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    rr1 = min(h, r0 + box_h + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(w, c0 + box_w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _draw_room(grid: np.ndarray, room: dict) -> None:
    r0, c0 = room["room_pos"]
    inner_h, inner_w = room["interior_shape"]
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1

    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY

    if room["side"] == "top":
        ac = random.randint(c0 + 1, c1 - 1)
        grid[r0, ac] = room["color"]
        room["anchor"] = (r0, ac)
    elif room["side"] == "bottom":
        ac = random.randint(c0 + 1, c1 - 1)
        grid[r1, ac] = room["color"]
        room["anchor"] = (r1, ac)
    elif room["side"] == "left":
        ar = random.randint(r0 + 1, r1 - 1)
        grid[ar, c0] = room["color"]
        room["anchor"] = (ar, c0)
    else:
        ar = random.randint(r0 + 1, r1 - 1)
        grid[ar, c1] = room["color"]
        room["anchor"] = (ar, c1)


def _draw_template(grid: np.ndarray, room: dict) -> None:
    r0, c0 = room["template_pos"]
    color = room["color"]
    mask = room["template_mask"]
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color


def _build_expected_output(inp: np.ndarray, rooms: list[dict]) -> np.ndarray:
    out = inp.copy()
    for room in rooms:
        tr0, tc0 = room["template_pos"]
        mask = room["template_mask"]
        out[tr0 : tr0 + mask.shape[0], tc0 : tc0 + mask.shape[1]][mask] = BLACK

    for room in rooms:
        rr0, rc0 = room["room_pos"]
        rotated = _rotate_patch(room["template_patch"], room["side"])
        out[rr0 + 1 : rr0 + 1 + rotated.shape[0], rc0 + 1 : rc0 + 1 + rotated.shape[1]][
            rotated != BLACK
        ] = room["color"]
    return out


def _sample_room_specs(n_rooms: int) -> list[dict] | None:
    colors = random.sample(COLOR_POOL, n_rooms)
    sides = list(SIDES)
    if n_rooms > 4:
        sides.append(random.choice(SIDES))
    random.shuffle(sides)

    specs: list[dict] = []
    for color, side in zip(colors, sides):
        base_h = random.randint(2, 5)
        base_w = random.randint(2, 5)
        mask = _sample_template_mask(base_h, base_w)
        if mask is None:
            return None
        patch = np.where(mask, color, BLACK).astype(int)

        if side in ("top", "bottom"):
            interior_shape = (base_h, base_w)
        else:
            interior_shape = (base_w, base_h)

        specs.append(
            {
                "color": color,
                "side": side,
                "template_mask": mask,
                "template_patch": patch,
                "interior_shape": interior_shape,
                "room_shape": (interior_shape[0] + 2, interior_shape[1] + 2),
            }
        )
    return specs


def _place_objects(grid_shape: tuple[int, int], specs: list[dict]) -> bool:
    occupied = np.zeros(grid_shape, dtype=bool)

    room_order = sorted(range(len(specs)), key=lambda i: specs[i]["room_shape"][0] * specs[i]["room_shape"][1], reverse=True)
    for idx in room_order:
        room_h, room_w = specs[idx]["room_shape"]
        positions = _all_positions(occupied, room_h, room_w, margin=1, pad=1)
        if not positions:
            return False
        r0, c0 = random.choice(positions)
        specs[idx]["room_pos"] = (r0, c0)
        _mark_box(occupied, r0, c0, room_h, room_w, pad=1)

    tpl_order = sorted(
        range(len(specs)),
        key=lambda i: specs[i]["template_mask"].shape[0] * specs[i]["template_mask"].shape[1],
        reverse=True,
    )
    for idx in tpl_order:
        tpl_h, tpl_w = specs[idx]["template_mask"].shape
        positions = _all_positions(occupied, tpl_h, tpl_w, margin=1, pad=1)
        if not positions:
            return False
        r0, c0 = random.choice(positions)
        specs[idx]["template_pos"] = (r0, c0)
        _mark_box(occupied, r0, c0, tpl_h, tpl_w, pad=1)

    return True


def _gen_attempt() -> np.ndarray | None:
    n_rooms = random.choice([4, 5])
    specs = _sample_room_specs(n_rooms)
    if specs is None:
        return None

    min_h = max(18, max(spec["room_shape"][0] for spec in specs) + 3)
    min_w = max(18, max(spec["room_shape"][1] for spec in specs) + 3)
    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)
    if grid_h < 8 or grid_w < 8:
        return None

    if not _place_objects((grid_h, grid_w), specs):
        return None

    inp = np.zeros((grid_h, grid_w), dtype=int)
    rooms = [dict(spec) for spec in specs]
    for room in rooms:
        _draw_room(inp, room)
    for room in rooms:
        _draw_template(inp, room)

    expected = _build_expected_output(inp, rooms)
    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
