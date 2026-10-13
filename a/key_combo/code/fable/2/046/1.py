from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GRAY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    assert connectivity == 4, "this task uses 4-connected components only"
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((int(cr), int(cc)))
            for dr, dc in DIR4:
                nr = int(cr + dr)
                nc = int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.asarray(coords, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _interval_overlap(a: dict, b: dict) -> bool:
    return max(int(a["bbox"][1]), int(b["bbox"][1])) <= min(int(a["bbox"][3]), int(b["bbox"][3]))


def _dock_segments(grid: np.ndarray, bg: int) -> list[dict]:
    bottom = grid[-1]
    docks: list[dict] = []
    c = 0
    while c < grid.shape[1]:
        if int(bottom[c]) == bg:
            c += 1
            continue
        color = int(bottom[c])
        c0 = int(c)
        while c < grid.shape[1] and int(bottom[c]) == color:
            c += 1
        c1 = int(c - 1)
        docks.append(
            {
                "id": len(docks),
                "color": color,
                "c0": c0,
                "c1": c1,
                "length": int(c1 - c0 + 1),
            }
        )
    assert docks, "expected at least one bottom-row dock segment"
    assert len({dock["color"] for dock in docks}) == len(docks), "dock colors must be unique"
    assert len({dock["length"] for dock in docks}) == len(docks), "dock lengths must be unique"
    return docks


def _gray_objects(grid: np.ndarray, bg: int, dock_colors: set[int]) -> tuple[int, list[dict]]:
    object_colors = [color for color in colors_present(grid, exclude={bg}) if int(color) not in dock_colors]
    assert len(object_colors) == 1, "expected exactly one floating-object color"
    object_color = int(object_colors[0])
    objects: list[dict] = []
    for idx, coords in enumerate(components_mask(grid == object_color, connectivity=4)):
        r0, c0, r1, c1 = bbox_coords(coords)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
        objects.append(
            {
                "id": int(idx),
                "coords": coords.astype(int),
                "bbox": (r0, c0, r1, c1),
                "mask": patch,
                "area": int(coords.shape[0]),
            }
        )
    assert objects, "expected at least one gray object"
    assert len({obj["area"] for obj in objects}) == len(objects), "object areas must be unique"
    return object_color, objects


def _rank_color_map(objects: list[dict], docks: list[dict]) -> dict[int, int]:
    ranked_objects = sorted(objects, key=lambda obj: (-int(obj["area"]), int(obj["bbox"][0]), int(obj["bbox"][1])))
    ranked_docks = sorted(docks, key=lambda dock: (-int(dock["length"]), int(dock["c0"])))
    assert len(ranked_objects) == len(ranked_docks), "object and dock counts must match"
    return {
        int(obj["id"]): int(dock["color"])
        for obj, dock in zip(ranked_objects, ranked_docks)
    }


def _settle_objects(
    objects: list[dict],
    dock_mask: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    h, _w = dock_mask.shape
    order = sorted(
        objects,
        key=lambda obj: (-int(obj["bbox"][2]), -int(obj["bbox"][0]), int(obj["bbox"][1])),
    )
    occupied = dock_mask.copy()
    settled_mask = np.zeros_like(dock_mask, dtype=bool)
    placements: dict[int, np.ndarray] = {}
    supports: dict[int, str] = {}
    down = np.array([1, 0], dtype=int)

    for obj in order:
        coords = obj["coords"]
        shift = 0
        while True:
            candidate = coords + (shift + 1) * down
            if int(candidate[:, 0].max()) >= h:
                break
            if np.any(occupied[candidate[:, 0], candidate[:, 1]]):
                break
            shift += 1
        placed = coords + shift * down
        next_coords = coords + (shift + 1) * down
        if int(next_coords[:, 0].max()) >= h:
            support = "floor"
        elif np.any(settled_mask[next_coords[:, 0], next_coords[:, 1]]):
            support = "object"
        else:
            assert np.any(dock_mask[next_coords[:, 0], next_coords[:, 1]])
            support = "dock"
        occupied[placed[:, 0], placed[:, 1]] = True
        settled_mask[placed[:, 0], placed[:, 1]] = True
        placements[int(obj["id"])] = placed.astype(int)
        supports[int(obj["id"])] = support
    return placements, supports


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each floating gray object straight downward, then recolor it by matching area rank to dock-length rank.

    Input structure:
    - The background is black.
    - The bottom row contains several separated horizontal colored dock segments whose colors and lengths are all distinct.
    - Above them are the same number of gray 4-connected objects, with pairwise distinct pixel counts.

    Rule:
    - Remove the gray objects from their starting positions and process them from lower initial bottom edge to higher initial bottom edge.
    - Keep each object's shape and columns fixed, and drop it vertically until one more downward step would leave the grid or overlap a dock or an already-settled object.
    - Rank docks by length from longest to shortest and rank objects by area from largest to smallest; recolor the object of rank r to the color of the dock of rank r.
    - Keep the docks unchanged.

    Key insight:
    - Gravity decides where objects land, but color depends only on the separate area-vs-length ranking, not on the dock directly underneath.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    bg = get_bg(g)
    docks = _dock_segments(g, bg)
    dock_colors = {int(dock["color"]) for dock in docks}
    object_color, objects = _gray_objects(g, bg, dock_colors)
    color_map = _rank_color_map(objects, docks)

    out = g.copy()
    out[g == object_color] = bg
    dock_mask = out != bg
    placements, _supports = _settle_objects(objects, dock_mask)

    for obj in objects:
        color = int(color_map[int(obj["id"])])
        coords = placements[int(obj["id"])]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _grow_polyomino(area: int, max_w: int, rng: random.Random) -> np.ndarray | None:
    dims = []
    for h in range(1, min(5, area) + 1):
        for w in range(1, min(max_w, 5, area) + 1):
            if area <= h * w and h * w <= area + 4:
                dims.append((h, w))
    rng.shuffle(dims)
    for h, w in dims:
        for _ in range(80):
            seed = (rng.randrange(h), rng.randrange(w))
            cells = {seed}
            while len(cells) < area:
                frontier = []
                for r, c in cells:
                    for dr, dc in DIR4:
                        nr = int(r + dr)
                        nc = int(c + dc)
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                            frontier.append((nr, nc))
                if not frontier:
                    break
                cells.add(rng.choice(frontier))
            if len(cells) != area:
                continue
            mask = np.zeros((h, w), dtype=bool)
            for r, c in cells:
                mask[int(r), int(c)] = True
            coords = np.argwhere(mask)
            r0, c0, r1, c1 = bbox_coords(coords)
            mask = mask[r0 : r1 + 1, c0 : c1 + 1]
            if mask.shape[1] > max_w:
                continue
            if mask.shape[0] >= 2 and rng.random() < 0.5:
                mask = np.flipud(mask)
            if mask.shape[1] >= 2 and rng.random() < 0.5:
                mask = np.fliplr(mask)
            return mask.copy()
    return None


def _sample_docks(count: int, rng: random.Random) -> tuple[list[dict], int] | None:
    for _ in range(120):
        lengths = rng.sample(range(3, 8), count)
        colors = rng.sample(PALETTE, count)
        gaps = [rng.randint(1, 2) for _ in range(count - 1)]
        left_margin = rng.randint(0, 2)
        right_margin = rng.randint(0, 2)
        width = left_margin + sum(lengths) + sum(gaps) + right_margin
        if not 12 <= width <= 30:
            continue
        docks: list[dict] = []
        c = left_margin
        for idx, (length, color) in enumerate(zip(lengths, colors)):
            docks.append(
                {
                    "id": int(idx),
                    "color": int(color),
                    "length": int(length),
                    "c0": int(c),
                    "c1": int(c + length - 1),
                }
            )
            c += length
            if idx != count - 1:
                c += gaps[idx]
        return docks, int(width)
    return None


def _shape_to_coords(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(mask)
    coords[:, 0] += int(top)
    coords[:, 1] += int(left)
    return coords.astype(int)


def _build_scene(rng: random.Random) -> dict | None:
    count = rng.randint(2, 4)
    sampled = _sample_docks(count, rng)
    if sampled is None:
        return None
    docks, width = sampled

    stack_dock = rng.randrange(count)
    other_docks = [dock["id"] for dock in docks if int(dock["id"]) != stack_dock]
    rng.shuffle(other_docks)
    home_docks = [stack_dock, stack_dock] + other_docks[: count - 2]

    areas = rng.sample(range(3, 10), count)
    objects: list[dict] = []
    for obj_id in range(count):
        home = int(home_docks[obj_id])
        dock = docks[home]
        mask = _grow_polyomino(int(areas[obj_id]), min(5, int(dock["length"])), rng)
        if mask is None:
            return None
        objects.append(
            {
                "id": int(obj_id),
                "home_dock": int(home),
                "area": int(mask.sum()),
                "mask": mask,
                "h": int(mask.shape[0]),
                "w": int(mask.shape[1]),
            }
        )

    stack_ids = [0, 1]
    stack_order = stack_ids[:]
    rng.shuffle(stack_order)
    gap_between = rng.randint(1, 3)
    bottom_clearance = rng.randint(3, 6)
    top_margin = rng.randint(0, 3)
    stack_height = sum(int(objects[obj_id]["h"]) for obj_id in stack_order) + gap_between
    min_height = max(12, top_margin + stack_height + bottom_clearance + 1)
    if min_height > 30:
        return None
    height = rng.randint(min_height, min(30, min_height + 7))

    stack_dock_info = docks[int(stack_dock)]
    anchor_col = rng.randint(int(stack_dock_info["c0"]), int(stack_dock_info["c1"]))
    stack_lefts: dict[int, int] = {}
    for obj_id in stack_ids:
        w = int(objects[obj_id]["w"])
        lo = max(int(stack_dock_info["c0"]), int(anchor_col - w + 1))
        hi = min(int(anchor_col), int(stack_dock_info["c1"] - w + 1))
        if lo > hi:
            return None
        stack_lefts[int(obj_id)] = rng.randint(lo, hi)
    if stack_lefts[0] == stack_lefts[1] and int(objects[0]["w"]) == int(objects[1]["w"]):
        alternatives = []
        for obj_id in stack_ids:
            w = int(objects[obj_id]["w"])
            lo = max(int(stack_dock_info["c0"]), int(anchor_col - w + 1))
            hi = min(int(anchor_col), int(stack_dock_info["c1"] - w + 1))
            options = [left for left in range(lo, hi + 1) if left != stack_lefts[obj_id]]
            if options:
                alternatives.append((obj_id, options))
        if alternatives:
            obj_id, options = rng.choice(alternatives)
            stack_lefts[int(obj_id)] = rng.choice(options)

    bottoms: dict[int, int] = {}
    tops: dict[int, int] = {}
    current_bottom = int(height - 1 - bottom_clearance)
    for pos, obj_id in enumerate(stack_order):
        h = int(objects[obj_id]["h"])
        top = int(current_bottom - h + 1)
        if top < top_margin:
            return None
        bottoms[int(obj_id)] = int(current_bottom)
        tops[int(obj_id)] = int(top)
        current_bottom = int(top - gap_between - 1)

    used_bottoms = set(bottoms.values())
    available_bottoms = list(range(2, height - 3))
    rng.shuffle(available_bottoms)
    for obj_id in range(count):
        if obj_id in tops:
            continue
        h = int(objects[obj_id]["h"])
        candidates = [
            bottom
            for bottom in available_bottoms
            if bottom not in used_bottoms and bottom - h + 1 >= 0
        ]
        if not candidates:
            return None
        bottom = rng.choice(candidates)
        used_bottoms.add(int(bottom))
        bottoms[int(obj_id)] = int(bottom)
        tops[int(obj_id)] = int(bottom - h + 1)

    for obj in objects:
        home = int(obj["home_dock"])
        dock = docks[home]
        if int(obj["id"]) in stack_lefts:
            left = int(stack_lefts[int(obj["id"])])
        else:
            max_left = int(dock["c1"] - obj["w"] + 1)
            if max_left < int(dock["c0"]):
                return None
            left = rng.randint(int(dock["c0"]), max_left)
        obj["left"] = int(left)
        obj["top"] = int(tops[int(obj["id"])])
        obj["coords"] = _shape_to_coords(obj["mask"], int(obj["top"]), int(obj["left"]))
        obj["bbox"] = bbox_coords(obj["coords"])

    if len({int(obj["bbox"][2]) for obj in objects}) != count:
        return None
    for i in range(count):
        for j in range(i + 1, count):
            pts_i = {tuple(map(int, rc)) for rc in objects[i]["coords"]}
            pts_j = {tuple(map(int, rc)) for rc in objects[j]["coords"]}
            if pts_i & pts_j:
                return None

    if not _interval_overlap(objects[0], objects[1]):
        return None

    return {
        "height": int(height),
        "width": int(width),
        "docks": docks,
        "objects": objects,
        "stack_dock": int(stack_dock),
    }


def _scene_input(scene: dict) -> np.ndarray:
    grid = np.full((int(scene["height"]), int(scene["width"])), BG, dtype=int)
    for dock in scene["docks"]:
        grid[-1, int(dock["c0"]) : int(dock["c1"]) + 1] = int(dock["color"])
    for obj in scene["objects"]:
        coords = obj["coords"]
        grid[coords[:, 0], coords[:, 1]] = GRAY
    return grid


def _scene_output(scene: dict) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, str], dict[int, int]]:
    h = int(scene["height"])
    w = int(scene["width"])
    out = np.full((h, w), BG, dtype=int)
    dock_mask = np.zeros((h, w), dtype=bool)
    for dock in scene["docks"]:
        out[-1, int(dock["c0"]) : int(dock["c1"]) + 1] = int(dock["color"])
        dock_mask[-1, int(dock["c0"]) : int(dock["c1"]) + 1] = True
    placements, supports = _settle_objects(scene["objects"], dock_mask)
    color_map = _rank_color_map(scene["objects"], scene["docks"])
    for obj in scene["objects"]:
        coords = placements[int(obj["id"])]
        out[coords[:, 0], coords[:, 1]] = int(color_map[int(obj["id"])])
    return out, placements, supports, color_map


def _validate_scene(scene: dict) -> np.ndarray | None:
    inp = _scene_input(scene)
    if get_bg(inp) != BG:
        return None
    if inp.shape[0] > 30 or inp.shape[1] > 30:
        return None
    if len(scene["objects"]) != len(scene["docks"]) or len(scene["objects"]) < 2:
        return None
    if len({int(dock["length"]) for dock in scene["docks"]}) != len(scene["docks"]):
        return None
    if len({int(obj["area"]) for obj in scene["objects"]}) != len(scene["objects"]):
        return None
    if len({int(obj["bbox"][2]) for obj in scene["objects"]}) != len(scene["objects"]):
        return None
    if not _interval_overlap(scene["objects"][0], scene["objects"][1]):
        return None

    expected, placements, supports, color_map = _scene_output(scene)
    if not any(support == "object" for support in supports.values()):
        return None
    if not any(
        int(color_map[int(obj["id"])]) != int(scene["docks"][int(obj["home_dock"])]["color"])
        for obj in scene["objects"]
    ):
        return None

    for obj in scene["objects"]:
        placed = placements[int(obj["id"])]
        shifts = placed - obj["coords"]
        if len(set(map(int, shifts[:, 0]))) != 1:
            return None
        if not np.all(shifts[:, 1] == 0):
            return None
        if int(shifts[0, 0]) <= 0:
            return None

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        scene = _build_scene(random)
        if scene is None:
            continue
        grid = _validate_scene(scene)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dock-ranking gravity scene")
