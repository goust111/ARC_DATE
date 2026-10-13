from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import itertools
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_key_combo_12_036_1",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
label_mask = _ARC_UTILS.label_mask
paste = _ARC_UTILS.paste


BG = 0
SHELF = 5
INPUT_COLORS = (1, 2, 3, 4, 6, 7)
RECOLOR_BY_HOLES = {0: 1, 1: 2, 2: 3, 3: 4}
GEN_RETRIES = 400


TEMPLATES_BY_HOLES = {
    0: [
        np.array([[1, 1], [1, 0], [1, 1]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 1]], dtype=int),
    ],
    1: [
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
        np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
        np.array([[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]], dtype=int),
    ],
    2: [
        np.array([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
    ],
    3: [
        np.array(
            [[1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1]],
            dtype=int,
        ),
        np.array(
            [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
            dtype=int,
        ),
    ],
}


def _extract_shelves(grid: np.ndarray) -> list[dict]:
    shelves: list[dict] = []
    h, w = map(int, grid.shape)
    for r in range(h):
        c = 0
        while c < w:
            if int(grid[r, c]) != SHELF:
                c += 1
                continue
            c0 = c
            while c + 1 < w and int(grid[r, c + 1]) == SHELF:
                c += 1
            shelves.append({"row": int(r), "c0": int(c0), "c1": int(c)})
            c += 1
    shelves.sort(key=lambda shelf: (int(shelf["row"]), int(shelf["c0"])))
    return shelves


def _count_holes_in_patch(patch: np.ndarray, color: int) -> int:
    bg_mask = patch != int(color)
    labels, n = label_mask(bg_mask, connectivity=4)
    holes = 0
    for idx in range(1, int(n) + 1):
        coords = np.argwhere(labels == idx)
        if coords.size == 0:
            continue
        touches_border = bool(
            (coords[:, 0] == 0).any()
            or (coords[:, 0] == patch.shape[0] - 1).any()
            or (coords[:, 1] == 0).any()
            or (coords[:, 1] == patch.shape[1] - 1).any()
        )
        if not touches_border:
            holes += 1
    return int(holes)


def _drop_sprite(coords: np.ndarray, shelves: list[dict], width: int, height: int) -> tuple[int, dict]:
    best_shift = int(height)
    hit_shelves: list[dict] = []
    for r, c in coords:
        obstacle_row = int(height)
        obstacle_shelf = None
        for shelf in shelves:
            if int(shelf["row"]) > int(r) and int(shelf["c0"]) <= int(c) <= int(shelf["c1"]):
                obstacle_row = int(shelf["row"])
                obstacle_shelf = shelf
                break
        shift = int(obstacle_row - 1 - int(r))
        if shift < best_shift:
            best_shift = shift
            hit_shelves = [obstacle_shelf] if obstacle_shelf is not None else []
        elif shift == best_shift and obstacle_shelf is not None:
            hit_shelves.append(obstacle_shelf)

    if hit_shelves:
        support = min(hit_shelves, key=lambda shelf: (int(shelf["row"]), int(shelf["c0"])))
    else:
        support = {"row": int(height), "c0": 0, "c1": int(width - 1)}
    return int(best_shift), support


def _sprite_records(grid: np.ndarray) -> list[dict]:
    shelves = _extract_shelves(grid)
    comps = find_components(
        grid,
        bg=BG,
        colors=[color for color in range(10) if color not in (BG, SHELF)],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    sprites: list[dict] = []
    for comp in comps:
        patch = crop_bbox(grid, comp["bbox"])
        color = int(comp["color"])
        holes = _count_holes_in_patch(patch, color)
        shift, support = _drop_sprite(comp["coords"], shelves, grid.shape[1], grid.shape[0])
        bbox = tuple(map(int, comp["bbox"]))
        landed_bbox = (int(bbox[0] + shift), int(bbox[1]), int(bbox[2] + shift), int(bbox[3]))
        sprites.append(
            {
                "bbox": bbox,
                "landed_bbox": landed_bbox,
                "patch": patch,
                "color": color,
                "holes": holes,
                "support": support,
            }
        )
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop sprites onto the nearest grey shelf, then reorder and recolor each shelf group by hole count.

    Input structure:
    - The background is black, and 2-3 grey horizontal shelf segments sit on separate rows.
    - Several monochromatic 4-connected sprites of distinct non-grey colors appear above those shelves.
    - Each sprite may contain 0-3 enclosed black hole regions inside its own bounding box.

    Rule:
    - Treat each sprite as a rigid body and move it straight downward until one of its pixels would hit the nearest shelf below or the grid floor.
    - Count the sprite's enclosed 4-connected holes, then group landed sprites by the shelf row that stopped them.
    - Within each shelf group, sort sprites by increasing hole count, breaking ties by their pre-sort landed left-to-right order, pack them from the shelf's left edge with one black column between neighbors, and recolor them with 0->blue, 1->red, 2->green, 3->yellow while leaving holes black.

    Key insight:
    - The shelf row is a physical grouping key produced by gravity, but the within-group order and final colors come from a separate topological key: the number of enclosed holes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    shelves = _extract_shelves(g)
    shelf_groups: dict[tuple[int, int, int], list[dict]] = {}
    for sprite in _sprite_records(g):
        support = sprite["support"]
        key = (int(support["row"]), int(support["c0"]), int(support["c1"]))
        shelf_groups.setdefault(key, []).append(sprite)

    out = np.full_like(g, BG)
    out[g == SHELF] = SHELF

    for shelf in shelves:
        key = (int(shelf["row"]), int(shelf["c0"]), int(shelf["c1"]))
        sprites = shelf_groups.get(key, [])
        if not sprites:
            continue
        ordered = sorted(
            sprites,
            key=lambda item: (int(item["holes"]), int(item["landed_bbox"][1])),
        )
        cursor = int(shelf["c0"])
        for sprite in ordered:
            patch = np.array(sprite["patch"], dtype=int)
            new_color = int(RECOLOR_BY_HOLES[int(sprite["holes"])])
            recolored = np.where(patch == int(sprite["color"]), new_color, BG)
            top = int(shelf["row"] - recolored.shape[0])
            paste(out, recolored, top, cursor, transparent=BG)
            cursor += int(recolored.shape[1]) + 1

    return out


def _random_transform(mask: np.ndarray, *, prefer_short: bool) -> np.ndarray:
    out = np.array(mask, dtype=int)
    if prefer_short and out.shape[0] > out.shape[1]:
        out = np.rot90(out)
    elif (not prefer_short) and out.shape[1] > out.shape[0] and random.random() < 0.35:
        out = np.rot90(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    if out.shape[0] == out.shape[1] and random.random() < 0.5:
        out = np.rot90(out, k=random.randint(1, 3))
    return np.array(out, dtype=int)


def _shape_for_holes(holes: int, *, prefer_short: bool) -> np.ndarray:
    mask = random.choice(TEMPLATES_BY_HOLES[int(holes)])
    shape = _random_transform(mask, prefer_short=prefer_short)
    assert _count_holes_in_patch(shape, 1) == int(holes)
    return shape


def _partition_objects(total: int, shelf_count: int) -> list[int]:
    options: list[tuple[int, ...]] = []
    for parts in itertools.product((1, 2, 3), repeat=shelf_count):
        if sum(parts) == int(total):
            options.append(parts)
    return list(random.choice(options))


def _choose_hole_layout(shelf_counts: list[int]) -> tuple[list[list[int]], int] | None:
    total = int(sum(shelf_counts))
    require_all = total >= 4 and random.random() < 0.7
    require_tie = total >= 5 and any(count >= 2 for count in shelf_counts) and random.random() < 0.5

    for _ in range(200):
        flat = [random.choices([0, 1, 2, 3], weights=[4, 4, 3, 3], k=1)[0] for _ in range(total)]
        if require_all:
            for idx, hole in enumerate((0, 1, 2, 3)):
                flat[idx] = hole
        random.shuffle(flat)

        per_shelf: list[list[int]] = []
        cursor = 0
        for count in shelf_counts:
            per_shelf.append(flat[cursor : cursor + count])
            cursor += count

        candidates = [i for i, holes in enumerate(per_shelf) if len(holes) >= 2 and len(set(holes)) >= 2]
        if not candidates:
            continue
        if len(set(flat)) < min(3, total):
            continue
        if require_tie and not any(len(holes) >= 2 and len(set(holes)) < len(holes) for holes in per_shelf):
            continue
        return per_shelf, int(random.choice(candidates))

    return None


def _assign_input_colors(objects: list[dict]) -> bool:
    remaining = list(INPUT_COLORS)
    random.shuffle(remaining)
    for obj in sorted(objects, key=lambda item: int(item["holes"]), reverse=True):
        mapped = int(RECOLOR_BY_HOLES[int(obj["holes"])])
        choices = [color for color in remaining if int(color) != mapped]
        if not choices:
            return False
        color = random.choice(choices)
        remaining.remove(color)
        obj["input_color"] = int(color)
    return True


def _build_expected(
    shape: tuple[int, int],
    shelves: list[dict],
    objects_by_shelf: list[list[dict]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for shelf in shelves:
        out[int(shelf["row"]), int(shelf["c0"]) : int(shelf["c1"]) + 1] = SHELF

    for shelf, objects in zip(shelves, objects_by_shelf):
        ordered = sorted(objects, key=lambda item: (int(item["holes"]), int(item["landed_left"])))
        cursor = int(shelf["c0"])
        for obj in ordered:
            patch = np.where(np.array(obj["shape"], dtype=int) == 1, int(RECOLOR_BY_HOLES[int(obj["holes"])]), BG)
            top = int(shelf["row"] - patch.shape[0])
            paste(out, patch, top, cursor, transparent=BG)
            cursor += int(patch.shape[1]) + 1
    return out


def _gen_attempt() -> np.ndarray | None:
    shelf_count = random.choice((2, 3))
    total_objects = random.randint(4, 6)
    shelf_counts = _partition_objects(total_objects, shelf_count)
    hole_layout = _choose_hole_layout(shelf_counts)
    if hole_layout is None:
        return None
    holes_by_shelf, unsorted_shelf = hole_layout

    prefer_short = shelf_count == 3 or random.random() < 0.75
    objects_by_shelf: list[list[dict]] = []
    for holes in holes_by_shelf:
        shelf_objects: list[dict] = []
        for hole_count in holes:
            shape = _shape_for_holes(hole_count, prefer_short=prefer_short)
            shelf_objects.append(
                {
                    "holes": int(hole_count),
                    "shape": shape,
                    "height": int(shape.shape[0]),
                    "width": int(shape.shape[1]),
                }
            )
        objects_by_shelf.append(shelf_objects)

    flat_objects = [obj for shelf in objects_by_shelf for obj in shelf]
    if not _assign_input_colors(flat_objects):
        return None

    shelf_layouts: list[dict] = []
    for shelf_index, objects in enumerate(objects_by_shelf):
        count = len(objects)
        indices = list(range(count))
        if shelf_index == int(unsorted_shelf):
            perms = [
                perm
                for perm in itertools.permutations(indices)
                if [objects[i]["holes"] for i in perm] != sorted(objects[i]["holes"] for i in perm)
            ]
            perm = list(random.choice(perms))
        else:
            random.shuffle(indices)
            perm = indices

        gaps = [random.randint(1, 3) for _ in range(max(0, count - 1))]
        left_pad = random.randint(0, 3)
        right_pad = random.randint(0, 3)
        cursor = int(left_pad)
        positions: dict[int, int] = {}
        for pos, idx in enumerate(perm):
            positions[int(idx)] = int(cursor)
            cursor += int(objects[idx]["width"])
            if pos < count - 1:
                cursor += int(gaps[pos])

        output_width = sum(int(obj["width"]) for obj in objects) + max(0, count - 1)
        shelf_len = max(int(cursor + right_pad), int(output_width)) + random.randint(0, 1)
        shelf_layouts.append({"perm": perm, "positions": positions, "len": int(shelf_len)})

    max_shelf_len = max(int(layout["len"]) for layout in shelf_layouts)
    width = max(16, int(max_shelf_len + random.randint(2, 5)))
    if width > 30:
        return None

    max_heights = [max(int(obj["height"]) for obj in shelf) for shelf in objects_by_shelf]
    shelf_rows: list[int] = []
    band_tops: list[int] = []
    band_top = 0
    for max_height in max_heights:
        band_tops.append(int(band_top))
        shelf_row = int(band_top + max_height + random.randint(2, 4))
        shelf_rows.append(shelf_row)
        band_top = int(shelf_row + 2)
    height = int(shelf_rows[-1] + random.randint(4, 6))
    if height > 30:
        return None

    shelves: list[dict] = []
    for layout, row in zip(shelf_layouts, shelf_rows):
        slack = int(width - layout["len"])
        c0 = random.randint(0, slack) if slack > 0 else 0
        shelves.append({"row": int(row), "c0": int(c0), "c1": int(c0 + layout["len"] - 1)})

    for shelf, objects, layout in zip(shelves, objects_by_shelf, shelf_layouts):
        for idx, obj in enumerate(objects):
            obj["landed_left"] = int(shelf["c0"] + layout["positions"][int(idx)])

    for shelf_index, (band_top, shelf, objects) in enumerate(zip(band_tops, shelves, objects_by_shelf)):
        low = int(band_top)
        for obj in objects:
            high = int(shelf["row"] - obj["height"] - 1)
            if high < low:
                return None
            obj["top"] = random.randint(low, high)
            obj["shelf_index"] = int(shelf_index)

    grid = np.full((height, width), BG, dtype=int)
    for shelf in shelves:
        grid[int(shelf["row"]), int(shelf["c0"]) : int(shelf["c1"]) + 1] = SHELF

    for objects in objects_by_shelf:
        for obj in objects:
            patch = np.where(np.array(obj["shape"], dtype=int) == 1, int(obj["input_color"]), BG)
            paste(grid, patch, int(obj["top"]), int(obj["landed_left"]), transparent=BG)

    expected = _build_expected(grid.shape, shelves, objects_by_shelf)

    if np.array_equal(grid, expected):
        return None
    if len({int(obj["holes"]) for obj in flat_objects}) < 2:
        return None
    if sum(1 for objs in objects_by_shelf if len(objs) >= 2) == 0:
        return None
    if not any(
        [objects_by_shelf[idx][obj_idx]["holes"] for obj_idx in shelf_layouts[idx]["perm"]]
        != sorted(objects_by_shelf[idx][obj_idx]["holes"] for obj_idx in shelf_layouts[idx]["perm"])
        for idx in range(len(objects_by_shelf))
    ):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    landed_records = _sprite_records(grid)
    if len(landed_records) != total_objects:
        return None
    if any(int(record["holes"]) > 3 for record in landed_records):
        return None
    if sorted(int(record["support"]["row"]) for record in landed_records) != sorted(
        int(shelf_rows[obj["shelf_index"]]) for obj in flat_objects
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shelf-hole sorting puzzle")
