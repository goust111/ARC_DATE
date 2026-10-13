import random
from collections import deque
from typing import Iterable

import numpy as np


SIDES = ("top", "right", "bottom", "left")
SIDE_TO_INDEX = {side: idx for idx, side in enumerate(SIDES)}
CARDINAL4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


GEN_BG = 0
GEN_FRAME = 5
DOOR_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
LAYOUT_CHOICES = ((1, 2), (2, 2), (2, 3))
OUTER_H_RANGE = (6, 9)
OUTER_W_RANGE = (6, 9)
GAP_RANGE = (2, 4)
MARGIN_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in CARDINAL4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _c in pts]
    cols = [c for _r, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    coords: list[tuple[int, int]] = []
    coords.extend((r0, c) for c in range(c0, c1 + 1))
    coords.extend((r, c1) for r in range(r0 + 1, r1))
    if r1 > r0:
        coords.extend((r1, c) for c in range(c1, c0 - 1, -1))
    if c1 > c0:
        coords.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return coords


def _side_cells(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    if side == "right":
        return [(r, c1) for r in range(r0 + 1, r1)]
    raise ValueError(side)


def _room_candidates(grid: np.ndarray, frame_color: int, bg: int) -> list[dict]:
    candidates: list[dict] = []
    for comp in _components_mask(grid == int(frame_color)):
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        if (r1 - r0 + 1) < 5 or (c1 - c0 + 1) < 5:
            continue
        border = _perimeter_coords(bbox)
        border_set = set(border)
        comp_set = set(comp)
        if not comp_set.issubset(border_set):
            continue
        if any(int(grid[r, c]) == int(bg) for r, c in border):
            continue
        if not any(r == r0 for r, _c in comp_set):
            continue
        if not any(r == r1 for r, _c in comp_set):
            continue
        if not any(c == c0 for _r, c in comp_set):
            continue
        if not any(c == c1 for _r, c in comp_set):
            continue
        candidates.append({"bbox": bbox, "cells": comp_set})

    filtered: list[dict] = []
    for idx, cand in enumerate(candidates):
        r0, c0, r1, c1 = cand["bbox"]
        nested = False
        for jdx, other in enumerate(candidates):
            if idx == jdx:
                continue
            or0, oc0, or1, oc1 = other["bbox"]
            if or0 < r0 and oc0 < c0 and r1 < or1 and c1 < oc1:
                nested = True
                break
        if not nested:
            filtered.append(cand)
    return filtered


def _door_interior_cells(
    bbox: tuple[int, int, int, int], side: str, cells: list[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return tuple((r0 + 1, c) for _r, c in cells)
    if side == "bottom":
        return tuple((r1 - 1, c) for _r, c in cells)
    if side == "left":
        return tuple((r, c0 + 1) for r, _c in cells)
    if side == "right":
        return tuple((r, c1 - 1) for r, _c in cells)
    raise ValueError(side)


def _extract_doors(
    grid: np.ndarray, bbox: tuple[int, int, int, int], frame_color: int, bg: int
) -> list[dict] | None:
    r0, c0, r1, c1 = bbox
    corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    if any(int(grid[r, c]) != int(frame_color) for r, c in corners):
        return None

    doors: list[dict] = []
    for side in SIDES:
        wall = _side_cells(bbox, side)
        idx = 0
        while idx < len(wall):
            r, c = wall[idx]
            color = int(grid[r, c])
            if color == int(frame_color):
                idx += 1
                continue
            if color == int(bg):
                return None
            end = idx
            while end < len(wall) and int(grid[wall[end]]) == color:
                end += 1
            run = wall[idx:end]
            if len(run) != 2:
                return None
            doors.append(
                {
                    "color": int(color),
                    "side": side,
                    "cells": tuple(run),
                    "interior_cells": _door_interior_cells(bbox, side, run),
                }
            )
            idx = end

    if len(doors) != 2:
        return None
    if len({door["color"] for door in doors}) != 2:
        return None
    return doors


def _extract_sprite(
    grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int
) -> dict | None:
    r0, c0, r1, c1 = bbox
    interior = np.asarray(grid[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
    mask = interior != int(bg)
    if not bool(mask.any()):
        return None

    rows, cols = np.where(mask)
    sr0, sc0 = int(rows.min()), int(cols.min())
    sr1, sc1 = int(rows.max()), int(cols.max())
    sprite = interior[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
    sprite_mask = sprite != int(bg)
    offset = (r0 + 1 + sr0, c0 + 1 + sc0)
    occupied = {
        (offset[0] + int(r), offset[1] + int(c))
        for r, c in zip(*np.where(sprite_mask))
    }
    return {
        "array": sprite,
        "mask": sprite_mask,
        "offset": offset,
        "occupied_abs": occupied,
    }


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert len(colors) > 0, "expected foreground colors"
    frame_color = int(colors[np.argmax(counts)])
    candidates = _room_candidates(g, int(frame_color), bg)
    assert candidates, "expected at least one framed room"

    rooms: list[dict] = []
    for room_index, cand in enumerate(sorted(candidates, key=lambda item: item["bbox"])):
        doors = _extract_doors(g, cand["bbox"], int(frame_color), int(bg))
        if doors is None:
            continue
        annotated_doors: list[dict] = []
        for door_index, door in enumerate(doors):
            door_copy = dict(door)
            door_copy["room_index"] = int(room_index)
            door_copy["door_index"] = int(door_index)
            annotated_doors.append(door_copy)
        rooms.append(
            {
                "bbox": cand["bbox"],
                "doors": annotated_doors,
                "sprite": _extract_sprite(g, cand["bbox"], int(bg)),
            }
        )

    assert rooms, "no valid rooms found"
    door_by_color: dict[int, list[dict]] = {}
    for room in rooms:
        for door in room["doors"]:
            door_by_color.setdefault(int(door["color"]), []).append(door)
    assert door_by_color and all(len(v) == 2 for v in door_by_color.values())

    return {
        "grid": g,
        "bg": int(bg),
        "frame_color": int(frame_color),
        "rooms": rooms,
        "door_by_color": door_by_color,
    }


def _fallback_anchor_cells(
    occupied_abs: set[tuple[int, int]], side: str, target_cells: tuple[tuple[int, int], ...]
) -> list[tuple[int, int]]:
    pts = sorted(occupied_abs)
    if side == "top":
        edge = min(r for r, _c in pts)
        candidates = [(r, c) for r, c in pts if r == edge]
        center = sum(c for _r, c in target_cells) / len(target_cells)
        candidates.sort(key=lambda item: (abs(item[1] - center), item[1]))
    elif side == "bottom":
        edge = max(r for r, _c in pts)
        candidates = [(r, c) for r, c in pts if r == edge]
        center = sum(c for _r, c in target_cells) / len(target_cells)
        candidates.sort(key=lambda item: (abs(item[1] - center), item[1]))
    elif side == "left":
        edge = min(c for _r, c in pts)
        candidates = [(r, c) for r, c in pts if c == edge]
        center = sum(r for r, _c in target_cells) / len(target_cells)
        candidates.sort(key=lambda item: (abs(item[0] - center), item[0]))
    else:
        edge = max(c for _r, c in pts)
        candidates = [(r, c) for r, c in pts if c == edge]
        center = sum(r for r, _c in target_cells) / len(target_cells)
        candidates.sort(key=lambda item: (abs(item[0] - center), item[0]))
    return sorted(candidates[:2])


def _choose_source_door(room: dict, sprite: dict) -> tuple[dict, np.ndarray]:
    occupied = sprite["occupied_abs"]
    best_door = None
    best_key = None
    for door in room["doors"]:
        overlap = sum(cell in occupied for cell in door["interior_cells"])
        distance = min(
            abs(r - ar) + abs(c - ac)
            for r, c in occupied
            for ar, ac in door["interior_cells"]
        )
        key = (int(overlap), -int(distance))
        if best_key is None or key > best_key:
            best_key = key
            best_door = door
    assert best_door is not None and best_key is not None

    anchor_abs = [cell for cell in best_door["interior_cells"] if cell in occupied]
    if len(anchor_abs) < 2:
        anchor_abs = _fallback_anchor_cells(occupied, best_door["side"], best_door["interior_cells"])
    assert len(anchor_abs) == 2

    off_r, off_c = sprite["offset"]
    anchor_mask = np.zeros_like(sprite["array"], dtype=bool)
    for ar, ac in anchor_abs:
        anchor_mask[ar - off_r, ac - off_c] = True
    return best_door, anchor_mask


def _placed_pixels(
    sprite: np.ndarray,
    anchor_mask: np.ndarray,
    room_bbox: tuple[int, int, int, int],
    door: dict,
    bg: int,
) -> list[tuple[int, int, int]] | None:
    rel_anchor = sorted(tuple(map(int, rc)) for rc in np.argwhere(anchor_mask))
    target_anchor = sorted(tuple(map(int, rc)) for rc in door["interior_cells"])
    if len(rel_anchor) != 2 or len(target_anchor) != 2:
        return None

    dr = target_anchor[0][0] - rel_anchor[0][0]
    dc = target_anchor[0][1] - rel_anchor[0][1]
    shifted_anchor = sorted((r + dr, c + dc) for r, c in rel_anchor)
    if shifted_anchor != target_anchor:
        return None

    r0, c0, r1, c1 = room_bbox
    pixels: list[tuple[int, int, int]] = []
    touching: set[tuple[int, int]] = set()
    sprite_mask = sprite != int(bg)
    for rr, cc in np.argwhere(sprite_mask):
        ar = int(rr) + dr
        ac = int(cc) + dc
        if not (r0 + 1 <= ar <= r1 - 1 and c0 + 1 <= ac <= c1 - 1):
            return None
        if ar in (r0 + 1, r1 - 1) or ac in (c0 + 1, c1 - 1):
            touching.add((ar, ac))
        pixels.append((ar, ac, int(sprite[int(rr), int(cc)])))

    if touching != set(target_anchor):
        return None
    return pixels


def _matching_door(door_by_color: dict[int, list[dict]], door: dict) -> dict:
    peers = door_by_color[int(door["color"])]
    for peer in peers:
        if peer["room_index"] != door["room_index"] or peer["door_index"] != door["door_index"]:
            return peer
    raise AssertionError("door color must appear in another room")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Teleport each room-local sprite through the door it touches, rotating the sprite so its
    source-door side becomes the target-door side, then place it just inside the paired door.
    """
    scene = _parse_scene(grid)
    g = scene["grid"]
    bg = scene["bg"]
    door_by_color = scene["door_by_color"]
    rooms = scene["rooms"]

    out = g.copy()
    placements: list[list[tuple[int, int, int]]] = []
    occupied_targets: set[tuple[int, int]] = set()

    for room in rooms:
        sprite = room["sprite"]
        if sprite is None:
            continue
        for r, c in sprite["occupied_abs"]:
            out[r, c] = bg

    for room in rooms:
        sprite = room["sprite"]
        if sprite is None:
            continue

        src_door, anchor_mask = _choose_source_door(room, sprite)
        dst_door = _matching_door(door_by_color, src_door)

        # The prompt's verbal example reverses clockwise/CCW, so use the geometric alignment rule:
        # rotate until the sprite side touching the source door lands on the target-door wall.
        k_ccw = (SIDE_TO_INDEX[src_door["side"]] - SIDE_TO_INDEX[dst_door["side"]]) % 4
        rotated_sprite = np.rot90(sprite["array"], k=int(k_ccw))
        rotated_anchor = np.rot90(anchor_mask, k=int(k_ccw))
        pixels = _placed_pixels(
            rotated_sprite,
            rotated_anchor,
            rooms[dst_door["room_index"]]["bbox"],
            dst_door,
            bg,
        )
        assert pixels is not None, "rotated sprite must fit its destination room"
        for r, c, _color in pixels:
            assert (r, c) not in occupied_targets, "teleports must not overlap"
            occupied_targets.add((r, c))
        placements.append(pixels)

    for pixels in placements:
        for r, c, color in pixels:
            out[r, c] = int(color)
    return out


def _paint_room_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _door_start_range(bbox: tuple[int, int, int, int], side: str) -> range:
    r0, c0, r1, c1 = bbox
    if side in ("top", "bottom"):
        return range(c0 + 1, c1 - 1)
    return range(r0 + 1, r1 - 1)


def _valid_starts(
    bbox: tuple[int, int, int, int], side: str, used: list[dict]
) -> list[int]:
    starts: list[int] = []
    for start in _door_start_range(bbox, side):
        ok = True
        for door in used:
            if door["side"] != side:
                continue
            if abs(int(start) - int(door["start"])) < 3:
                ok = False
                break
        if ok:
            starts.append(int(start))
    return starts


def _make_door(
    bbox: tuple[int, int, int, int], side: str, start: int, color: int
) -> dict:
    r0, c0, r1, c1 = bbox
    if side == "top":
        cells = ((r0, start), (r0, start + 1))
    elif side == "bottom":
        cells = ((r1, start), (r1, start + 1))
    elif side == "left":
        cells = ((start, c0), (start + 1, c0))
    elif side == "right":
        cells = ((start, c1), (start + 1, c1))
    else:
        raise ValueError(side)
    return {
        "color": int(color),
        "side": side,
        "start": int(start),
        "cells": tuple(cells),
        "interior_cells": _door_interior_cells(bbox, side, list(cells)),
    }


def _paint_door(grid: np.ndarray, door: dict) -> None:
    for r, c in door["cells"]:
        grid[r, c] = int(door["color"])


def _sample_layout(rng: random.Random) -> tuple[list[dict], tuple[int, int]]:
    rows, cols = rng.choice(LAYOUT_CHOICES)
    sizes = [
        (
            rng.randint(*OUTER_H_RANGE),
            rng.randint(*OUTER_W_RANGE),
        )
        for _ in range(rows * cols)
    ]
    row_heights = [
        max(sizes[row * cols + col][0] for col in range(cols))
        for row in range(rows)
    ]
    col_widths = [
        max(sizes[row * cols + col][1] for row in range(rows))
        for col in range(cols)
    ]
    gap_r = rng.randint(*GAP_RANGE)
    gap_c = rng.randint(*GAP_RANGE)
    margin_t = rng.randint(*MARGIN_RANGE)
    margin_b = rng.randint(*MARGIN_RANGE)
    margin_l = rng.randint(*MARGIN_RANGE)
    margin_r = rng.randint(*MARGIN_RANGE)

    h = margin_t + sum(row_heights) + gap_r * (rows - 1) + margin_b
    w = margin_l + sum(col_widths) + gap_c * (cols - 1) + margin_r
    rooms: list[dict] = []
    idx = 0
    row_origin = margin_t
    for row in range(rows):
        col_origin = margin_l
        for col in range(cols):
            rh, rw = sizes[idx]
            box_h = row_heights[row]
            box_w = col_widths[col]
            off_r = rng.randint(0, box_h - rh)
            off_c = rng.randint(0, box_w - rw)
            r0 = row_origin + off_r
            c0 = col_origin + off_c
            rooms.append({"bbox": (r0, c0, r0 + rh - 1, c0 + rw - 1), "doors": []})
            idx += 1
            col_origin += box_w + gap_c
        row_origin += row_heights[row] + gap_r
    return rooms, (h, w)


def _assign_pair_doors(
    rng: random.Random,
    rooms: list[dict],
    room_a: int,
    room_b: int,
    color_a: int,
    color_b: int,
) -> bool:
    for color in (color_a, color_b):
        placed = False
        for _ in range(80):
            side_a = rng.choice(SIDES)
            side_b = rng.choice(SIDES)
            starts_a = _valid_starts(rooms[room_a]["bbox"], side_a, rooms[room_a]["doors"])
            starts_b = _valid_starts(rooms[room_b]["bbox"], side_b, rooms[room_b]["doors"])
            if not starts_a or not starts_b:
                continue
            door_a = _make_door(rooms[room_a]["bbox"], side_a, rng.choice(starts_a), color)
            door_b = _make_door(rooms[room_b]["bbox"], side_b, rng.choice(starts_b), color)
            rooms[room_a]["doors"].append(door_a)
            rooms[room_b]["doors"].append(door_b)
            placed = True
            break
        if not placed:
            return False
    return True


def _crop_mask_pair(mask: np.ndarray, anchor_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.where(mask)
    r0, c0 = int(rows.min()), int(cols.min())
    r1, c1 = int(rows.max()), int(cols.max())
    return mask[r0 : r1 + 1, c0 : c1 + 1], anchor_mask[r0 : r1 + 1, c0 : c1 + 1]


def _boundary_cells(mask: np.ndarray, side: str) -> set[tuple[int, int]]:
    if side == "top":
        edge = 0
        return {(edge, c) for c in range(mask.shape[1]) if bool(mask[edge, c])}
    if side == "bottom":
        edge = mask.shape[0] - 1
        return {(edge, c) for c in range(mask.shape[1]) if bool(mask[edge, c])}
    if side == "left":
        edge = 0
        return {(r, edge) for r in range(mask.shape[0]) if bool(mask[r, edge])}
    edge = mask.shape[1] - 1
    return {(r, edge) for r in range(mask.shape[0]) if bool(mask[r, edge])}


def _random_sprite_candidate(
    rng: random.Random,
    palette: list[int],
    source_room: dict,
    source_door: dict,
    target_room: dict,
    target_door: dict,
    bg: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]], int] | None:
    target_rotation = (SIDE_TO_INDEX[source_door["side"]] - SIDE_TO_INDEX[target_door["side"]]) % 4

    for _ in range(200):
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        mask = np.zeros((h, w), dtype=bool)
        anchor_mask = np.zeros((h, w), dtype=bool)

        if source_door["side"] in ("top", "bottom"):
            start = rng.randint(0, w - 2)
            row = 0 if source_door["side"] == "top" else h - 1
            anchor_cells = {(row, start), (row, start + 1)}
            blocked = {
                (row, c)
                for c in range(w)
                if (row, c) not in anchor_cells
            }
        else:
            start = rng.randint(0, h - 2)
            col = 0 if source_door["side"] == "left" else w - 1
            anchor_cells = {(start, col), (start + 1, col)}
            blocked = {
                (r, col)
                for r in range(h)
                if (r, col) not in anchor_cells
            }

        for r, c in anchor_cells:
            mask[r, c] = True
            anchor_mask[r, c] = True

        target_area = rng.randint(3, min(h * w, 8))
        while int(mask.sum()) < target_area:
            frontier = set()
            for r, c in zip(*np.where(mask)):
                for dr, dc in CARDINAL4:
                    nr, nc = int(r) + dr, int(c) + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if bool(mask[nr, nc]) or (nr, nc) in blocked:
                        continue
                    frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = rng.choice(sorted(frontier))
            mask[nr, nc] = True

        if int(mask.sum()) < 3:
            continue

        mask, anchor_mask = _crop_mask_pair(mask, anchor_mask)
        if _boundary_cells(mask, source_door["side"]) != set(map(tuple, np.argwhere(anchor_mask))):
            continue
        if int(mask.sum()) == int(anchor_mask.sum()):
            continue

        color_count = 1 if len(palette) == 1 or rng.random() < 0.35 else min(len(palette), rng.choice((2, 2, 3)))
        colors = rng.sample(palette, k=color_count)
        sprite = np.full(mask.shape, bg, dtype=int)
        occupied = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        seed = rng.randrange(9)
        for r, c in occupied:
            sprite[r, c] = colors[(2 * r + 3 * c + seed) % len(colors)]
        if color_count > 1 and len({int(sprite[r, c]) for r, c in occupied}) == 1:
            rr, cc = occupied[-1]
            sprite[rr, cc] = colors[1]

        source_pixels = _placed_pixels(sprite, anchor_mask, source_room["bbox"], source_door, bg)
        if source_pixels is None:
            continue

        rotated_sprite = np.rot90(sprite, k=int(target_rotation))
        rotated_anchor = np.rot90(anchor_mask, k=int(target_rotation))
        target_pixels = _placed_pixels(
            rotated_sprite, rotated_anchor, target_room["bbox"], target_door, bg
        )
        if target_pixels is None:
            continue

        if target_rotation != 0 and np.array_equal(sprite, rotated_sprite):
            continue
        return sprite, anchor_mask, source_pixels, int(target_rotation)
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rooms, shape = _sample_layout(rng)
    room_count = len(rooms)
    if room_count > len(DOOR_COLOR_POOL):
        return None

    grid = np.full(shape, GEN_BG, dtype=int)
    for room in rooms:
        _paint_room_border(grid, room["bbox"], GEN_FRAME)

    room_order = list(range(room_count))
    rng.shuffle(room_order)
    pairs = [(room_order[i], room_order[i + 1]) for i in range(0, room_count, 2)]
    door_colors = rng.sample(DOOR_COLOR_POOL, k=room_count)

    for pair_index, (room_a, room_b) in enumerate(pairs):
        color_a = door_colors[2 * pair_index]
        color_b = door_colors[2 * pair_index + 1]
        if not _assign_pair_doors(rng, rooms, room_a, room_b, color_a, color_b):
            return None

    for room_index, room in enumerate(rooms):
        if len(room["doors"]) != 2:
            return None
        for door_index, door in enumerate(room["doors"]):
            door["room_index"] = int(room_index)
            door["door_index"] = int(door_index)
            _paint_door(grid, door)

    source_rooms: list[int] = []
    for room_a, room_b in pairs:
        mode = rng.choice(("none", "left", "right", "both"))
        if mode == "left":
            source_rooms.append(room_a)
        elif mode == "right":
            source_rooms.append(room_b)
        elif mode == "both":
            source_rooms.extend((room_a, room_b))
    if not source_rooms:
        room_a, room_b = rng.choice(pairs)
        source_rooms.append(rng.choice((room_a, room_b)))

    used_door_colors = {door["color"] for room in rooms for door in room["doors"]}
    sprite_palette = [c for c in range(10) if c not in {GEN_BG, GEN_FRAME} and c not in used_door_colors]
    if not sprite_palette:
        sprite_palette = [c for c in range(10) if c not in {GEN_BG, GEN_FRAME}]

    door_by_color: dict[int, list[dict]] = {}
    for room in rooms:
        for door in room["doors"]:
            door_by_color.setdefault(int(door["color"]), []).append(door)
    for color, entries in door_by_color.items():
        if len(entries) != 2:
            return None

    rotations: list[int] = []
    for room_index in source_rooms:
        room = rooms[room_index]
        source_door = rng.choice(room["doors"])
        target_candidates = door_by_color[int(source_door["color"])]
        target_door = next(
            door
            for door in target_candidates
            if door["room_index"] != room_index or door["door_index"] != source_door["door_index"]
        )
        sprite_result = _random_sprite_candidate(
            rng,
            sprite_palette,
            room,
            source_door,
            rooms[target_door["room_index"]],
            target_door,
            GEN_BG,
        )
        if sprite_result is None:
            return None
        _sprite, _anchor_mask, source_pixels, rot = sprite_result
        for r, c, color in source_pixels:
            if int(grid[r, c]) != GEN_BG:
                return None
            grid[r, c] = int(color)
        rotations.append(int(rot))

    if not rotations:
        return None
    if all(rot == 0 for rot in rotations) and rng.random() < 0.75:
        return None

    try:
        output = main(grid)
    except Exception:
        return None
    if np.array_equal(output, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal-room puzzle")
