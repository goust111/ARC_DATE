from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
RECT_COLORS = (2, 3, 4, 6, 7, 8, 9)
SIDE_PRIORITY = {"N": 0, "W": 1, "S": 2, "E": 3}
OPPOSITE_SIDE = {"N": "S", "S": "N", "W": "E", "E": "W"}
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        q = deque([(rr, cc)])
        seen[rr, cc] = True
        comp = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if not (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]):
                    continue
                if seen[nr, nc] or not mask[nr, nc]:
                    continue
                seen[nr, nc] = True
                q.append((nr, nc))
        out.append(np.array(comp, dtype=int))
    return out


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    rectangles = []
    for color in sorted(set(map(int, np.unique(g))) - {BLACK, BLUE, GREY}):
        for coords in _components_of_color(g, color):
            r0, c0, r1, c1 = _bbox_coords(coords)
            crop = g[r0 : r1 + 1, c0 : c1 + 1]
            h, w = crop.shape
            if h < 4 or w < 4:
                continue

            border_ok = (
                np.all(crop[0, :] == color)
                and np.all(crop[-1, :] == color)
                and np.all(crop[:, 0] == color)
                and np.all(crop[:, -1] == color)
            )
            if not border_ok:
                continue

            interior = crop[1:-1, 1:-1]
            valid = np.isin(interior, [color, GREY])
            if not bool(valid.all()):
                continue

            seed_local = np.argwhere(interior == GREY)
            if len(seed_local) > 1:
                continue
            if len(seed_local) == 1:
                sr, sc = seed_local[0]
                seed = (int(r0 + 1 + sr), int(c0 + 1 + sc))
            else:
                seed = None

            if int((crop == color).sum()) + int((crop == GREY).sum()) != h * w:
                continue

            rectangles.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "seed": seed,
                }
            )
    rectangles.sort(key=lambda item: item["bbox"])
    return rectangles


def _nearest_wall_side(bbox: tuple[int, int, int, int], seed: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    sr, sc = seed
    distances = {
        "N": int(sr - r0),
        "W": int(sc - c0),
        "S": int(r1 - sr),
        "E": int(c1 - sc),
    }
    return min(distances, key=lambda side: (distances[side], SIDE_PRIORITY[side]))


def _door_coord(
    bbox: tuple[int, int, int, int], seed: tuple[int, int], side: str | None = None
) -> tuple[int, int, str]:
    r0, c0, r1, c1 = bbox
    sr, sc = seed
    chosen = _nearest_wall_side(bbox, seed) if side is None else str(side)
    if chosen == "N":
        return int(r0), int(sc), chosen
    if chosen == "W":
        return int(sr), int(c0), chosen
    if chosen == "S":
        return int(r1), int(sc), chosen
    return int(sr), int(c1), chosen


def _apply_hollowing(
    grid: np.ndarray,
    rectangles: list[dict],
    *,
    forced_sides: dict[tuple[int, int, int, int], str] | None = None,
    erase_seed: bool = True,
) -> tuple[np.ndarray, dict[tuple[int, int, int, int], str]]:
    out = np.asarray(grid, dtype=int).copy()
    used_sides: dict[tuple[int, int, int, int], str] = {}
    for rect in rectangles:
        seed = rect["seed"]
        if seed is None:
            continue
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = BLACK
        if not erase_seed:
            out[seed] = GREY
        side_override = None if forced_sides is None else forced_sides.get(rect["bbox"])
        dr, dc, side = _door_coord(rect["bbox"], seed, side_override)
        out[dr, dc] = BLACK
        used_sides[rect["bbox"]] = side
    return out, used_sides


def _flood_blue(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    q = deque()
    for r, c in np.argwhere(out == BLUE):
        q.append((int(r), int(c)))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]):
                continue
            if int(out[nr, nc]) != BLACK:
                continue
            out[nr, nc] = BLUE
            q.append((nr, nc))
    return out


def _transform(
    grid: np.ndarray,
    *,
    do_hollow: bool = True,
    do_flood: bool = True,
    erase_seed: bool = True,
    forced_sides: dict[tuple[int, int, int, int], str] | None = None,
) -> np.ndarray:
    rectangles = _find_rectangles(grid)
    out = np.asarray(grid, dtype=int).copy()
    if do_hollow:
        out, _ = _apply_hollowing(out, rectangles, forced_sides=forced_sides, erase_seed=erase_seed)
    if do_flood:
        out = _flood_blue(out)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Hollow every rectangle that contains a grey interior seed, open a nearest-wall door, then flood blue through the new passages.

    Input structure:
    - The grid is mostly black background with several non-overlapping solid colored rectangles.
    - Some rectangles contain exactly one grey pixel strictly inside the border; blue pixels appear only in the black background.
    - Rectangle colors are distinct, and rectangles without grey seeds stay as solid blockers.

    Rule:
    - Detect each colored rectangle by its monochromatic border and bounding box.
    - If a rectangle contains a grey interior seed, keep its 1-pixel border, turn its interior black, remove the grey seed, and open one black door on the border at the wall nearest to the seed; ties prefer north, then west.
    - Leave rectangles without grey seeds unchanged.
    - Starting from every original blue seed, flood through 4-connected black cells and recolor all reached black cells blue.

    Key insight:
    Grey seeds are keys only when they are contained inside a rectangle: they choose which wall gets opened, changing background connectivity so the external blue flood can enter the hollow room.
    """
    return _transform(grid, do_hollow=True, do_flood=True, erase_seed=True)


def _seed_candidates_for_side(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    candidates = []
    for sr in range(r0 + 1, r1):
        for sc in range(c0 + 1, c1):
            if _nearest_wall_side(bbox, (sr, sc)) == side:
                candidates.append((sr, sc))
    return candidates


def _can_place(room_boxes: list[tuple[int, int, int, int]], gap: int = 2) -> bool:
    for i, (r0, c0, r1, c1) in enumerate(room_boxes):
        for rr0, cc0, rr1, cc1 in room_boxes[i + 1 :]:
            if not (r1 + gap < rr0 or rr1 + gap < r0 or c1 + gap < cc0 or cc1 + gap < c0):
                return False
    return True


def _make_module(shape: tuple[int, int], side: str) -> dict | None:
    h, w = shape
    for _ in range(120):
        room_h = random.randint(5, 7)
        room_w = random.randint(5, 7)
        pocket = random.randint(4, 6)
        block_a = random.randint(4, 6)
        block_b = random.randint(4, 6)

        if side == "W":
            if block_a + room_h + block_b > h:
                continue
            room_r0 = random.randint(block_a, h - room_h - block_b)
            room_c0 = pocket
            room = (room_r0, room_c0, room_r0 + room_h - 1, room_c0 + room_w - 1)
            blockers = [
                (room_r0 - block_a, 0, room_r0 - 1, pocket - 1),
                (room_r0 + room_h, 0, room_r0 + room_h + block_b - 1, pocket - 1),
            ]
            pocket_cells = [(r, c) for r in range(room_r0, room_r0 + room_h) for c in range(0, pocket)]
        elif side == "E":
            if block_a + room_h + block_b > h:
                continue
            room_r0 = random.randint(block_a, h - room_h - block_b)
            room_c0 = w - pocket - room_w
            room = (room_r0, room_c0, room_r0 + room_h - 1, room_c0 + room_w - 1)
            blockers = [
                (room_r0 - block_a, room[3] + 1, room_r0 - 1, w - 1),
                (room_r0 + room_h, room[3] + 1, room_r0 + room_h + block_b - 1, w - 1),
            ]
            pocket_cells = [(r, c) for r in range(room_r0, room_r0 + room_h) for c in range(room[3] + 1, w)]
        elif side == "N":
            if block_a + room_w + block_b > w:
                continue
            room_c0 = random.randint(block_a, w - room_w - block_b)
            room_r0 = pocket
            room = (room_r0, room_c0, room_r0 + room_h - 1, room_c0 + room_w - 1)
            blockers = [
                (0, room_c0 - block_a, pocket - 1, room_c0 - 1),
                (0, room[3] + 1, pocket - 1, room[3] + block_b),
            ]
            pocket_cells = [(r, c) for r in range(0, pocket) for c in range(room_c0, room_c0 + room_w)]
        else:
            if block_a + room_w + block_b > w:
                continue
            room_c0 = random.randint(block_a, w - room_w - block_b)
            room_r0 = h - pocket - room_h
            room = (room_r0, room_c0, room_r0 + room_h - 1, room_c0 + room_w - 1)
            blockers = [
                (room[2] + 1, room_c0 - block_a, h - 1, room_c0 - 1),
                (room[2] + 1, room[3] + 1, h - 1, room[3] + block_b),
            ]
            pocket_cells = [(r, c) for r in range(room[2] + 1, h) for c in range(room_c0, room_c0 + room_w)]

        seed_options = _seed_candidates_for_side(room, side)
        if not seed_options or not pocket_cells:
            continue
        return {
            "side": side,
            "room_bbox": room,
            "blockers": blockers,
            "grey_seed": random.choice(seed_options),
            "blue_seed": random.choice(pocket_cells),
        }
    return None


def _render_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _semantic_checks(grid: np.ndarray, modules: list[dict]) -> bool:
    out = main(grid)
    rectangles = _find_rectangles(grid)
    if len(rectangles) != 6:
        return False
    if sum(rect["seed"] is not None for rect in rectangles) != 2:
        return False
    if not np.array_equal(out, _transform(grid)):
        return False
    if np.array_equal(out, _flood_blue(grid)):
        return False
    if np.array_equal(out, _transform(grid, do_flood=False)):
        return False
    if np.array_equal(out, _transform(grid, erase_seed=False)):
        return False

    forced = {}
    for rect in rectangles:
        if rect["seed"] is not None:
            side = _nearest_wall_side(rect["bbox"], rect["seed"])
            forced[rect["bbox"]] = OPPOSITE_SIDE[side]
    if not forced or np.array_equal(out, _transform(grid, forced_sides=forced)):
        return False

    for module in modules:
        r0, c0, r1, c1 = module["room_bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or not bool(np.all(interior == BLUE)):
            return False

    for rect in rectangles:
        if rect["seed"] is None:
            r0, c0, r1, c1 = rect["bbox"]
            if not bool(np.all(out[r0 : r1 + 1, c0 : c1 + 1] == rect["color"])):
                return False

    if int((out == BLUE).sum()) <= int((grid == BLUE).sum()) + 8:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(("WE", "NS"))
    if mode == "WE":
        height = random.randint(18, 26)
        width = random.randint(22, 28)
        sides = ["W", "E"]
    else:
        height = random.randint(22, 28)
        width = random.randint(18, 26)
        sides = ["N", "S"]

    modules = []
    for side in sides:
        module = _make_module((height, width), side)
        if module is None:
            return None
        modules.append(module)

    room_boxes = [module["room_bbox"] for module in modules]
    if not _can_place(room_boxes, gap=3):
        return None

    all_boxes = []
    for module in modules:
        all_boxes.append(module["room_bbox"])
        all_boxes.extend(module["blockers"])
    if not _can_place(all_boxes, gap=0):
        return None

    colors = random.sample(RECT_COLORS, len(all_boxes))
    grid = np.zeros((height, width), dtype=int)

    idx = 0
    for module in modules:
        module["room_color"] = colors[idx]
        _render_rect(grid, module["room_bbox"], colors[idx])
        idx += 1
        module["blocker_colors"] = []
        for bbox in module["blockers"]:
            module["blocker_colors"].append(colors[idx])
            _render_rect(grid, bbox, colors[idx])
            idx += 1

    for module in modules:
        grid[module["grey_seed"]] = GREY
        grid[module["blue_seed"]] = BLUE

    if max(grid.shape) > 30:
        return None
    if int((grid == BLUE).sum()) != 2:
        return None
    if int((grid == GREY).sum()) != 2:
        return None
    if not _semantic_checks(grid, modules):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
