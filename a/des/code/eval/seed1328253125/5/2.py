import random
from collections import Counter

import numpy as np


ARROW_COLOR = 5
DIR_VECTORS = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}
DIR_SIDES = {
    "right": ("right", "left"),
    "left": ("left", "right"),
    "down": ("bottom", "top"),
    "up": ("top", "bottom"),
}
PERP = {
    "right": (1, 0),
    "left": (1, 0),
    "down": (0, 1),
    "up": (0, 1),
}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _connected_components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    steps = DIR4 if connectivity == 4 else DIR8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _parse_arrow(coords: np.ndarray) -> dict | None:
    cells = {tuple(map(int, rc)) for rc in coords}
    if len(cells) < 4:
        return None

    for name, (dr, dc) in DIR_VECTORS.items():
        pr, pc = PERP[name]
        for tail in cells:
            if (tail[0] - dr, tail[1] - dc) in cells:
                continue

            shaft = []
            rr, cc = tail
            while (rr, cc) in cells:
                shaft.append((rr, cc))
                rr += dr
                cc += dc

            if len(shaft) < 2:
                continue

            end = shaft[-1]
            head1 = (end[0] + dr + pr, end[1] + dc + pc)
            head2 = (end[0] + dr - pr, end[1] + dc - pc)
            expected = set(shaft)
            expected.add(head1)
            expected.add(head2)
            if expected != cells:
                continue

            return {
                "direction": name,
                "vec": (dr, dc),
                "perp": (pr, pc),
                "tail": tail,
                "shaft_end": end,
                "cells": expected,
            }
    return None


def _find_objects(grid: np.ndarray, bg: int, arrow_color: int) -> tuple[list[dict], dict[tuple[int, int], int]]:
    objects: list[dict] = []
    cell_to_obj: dict[tuple[int, int], int] = {}

    for color in map(int, np.unique(grid)):
        if color in {bg, arrow_color}:
            continue
        for coords in _connected_components(grid == color, connectivity=8):
            idx = len(objects)
            cell_set = {tuple(map(int, rc)) for rc in coords}
            for cell in cell_set:
                cell_to_obj[cell] = idx
            objects.append(
                {
                    "color": color,
                    "coords": cell_set,
                    "bbox": _bbox(coords),
                }
            )
    return objects, cell_to_obj


def _find_tail_object(arrow: dict, cell_to_obj: dict[tuple[int, int], int]) -> int:
    dr, dc = arrow["vec"]
    tr, tc = arrow["tail"]
    direct = (tr - dr, tc - dc)
    if direct in cell_to_obj:
        return cell_to_obj[direct]

    hits = set()
    for rr, cc in ((tr - 1, tc), (tr + 1, tc), (tr, tc - 1), (tr, tc + 1)):
        if (rr, cc) == (tr + dr, tc + dc):
            continue
        if (rr, cc) in cell_to_obj:
            hits.add(cell_to_obj[(rr, cc)])
    assert len(hits) == 1, "tail must identify a unique source object"
    return next(iter(hits))


def _find_target_object(
    arrow: dict,
    cell_to_obj: dict[tuple[int, int], int],
    grid_shape: tuple[int, int],
) -> int:
    h, w = grid_shape
    dr, dc = arrow["vec"]
    pr, pc = arrow["perp"]
    er, ec = arrow["shaft_end"]

    for step in range(1, max(h, w) + 1):
        cr = er + dr * step
        cc = ec + dc * step
        hits = set()
        for rr, cc2 in ((cr, cc), (cr + pr, cc + pc), (cr - pr, cc - pc)):
            if (rr, cc2) in cell_to_obj:
                hits.add(cell_to_obj[(rr, cc2)])
        if hits:
            assert len(hits) == 1, "arrow should point to a unique nearest target object"
            return next(iter(hits))
    raise AssertionError("arrow must point to a target object")


def _move_delta(direction: str, src_bbox: tuple[int, int, int, int], dst_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    ar0, ac0, ar1, ac1 = src_bbox
    br0, bc0, br1, bc1 = dst_bbox

    if direction == "right":
        gap = bc0 - ac1 - 1
        return 0, -gap
    if direction == "left":
        gap = ac0 - bc1 - 1
        return 0, gap
    if direction == "down":
        gap = br0 - ar1 - 1
        return -gap, 0
    gap = ar0 - br1 - 1
    return gap, 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify grey arrows between objects, pull each target object back toward the
    arrow tail object, and erase the arrows. Conflicting targets stay in place.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    if ARROW_COLOR == bg or int(np.sum(g == ARROW_COLOR)) == 0:
        return g.copy()

    arrow_components = _connected_components(g == ARROW_COLOR, connectivity=8)
    arrows: list[dict] = []
    for comp in arrow_components:
        parsed = _parse_arrow(comp)
        assert parsed is not None, "every grey component must be an arrow"
        arrows.append(parsed)

    objects, cell_to_obj = _find_objects(g, bg, ARROW_COLOR)
    target_counts: Counter[int] = Counter()
    arrow_links: list[tuple[dict, int, int]] = []

    for arrow in arrows:
        src_idx = _find_tail_object(arrow, cell_to_obj)
        dst_idx = _find_target_object(arrow, cell_to_obj, g.shape)
        assert src_idx != dst_idx, "source and target objects must differ"
        target_counts[dst_idx] += 1
        arrow_links.append((arrow, src_idx, dst_idx))

    out = g.copy()
    out[out == ARROW_COLOR] = bg

    moved_targets = {dst_idx for _arrow, _src_idx, dst_idx in arrow_links if target_counts[dst_idx] == 1}
    for idx in moved_targets:
        for r, c in objects[idx]["coords"]:
            out[r, c] = bg

    for arrow, src_idx, dst_idx in arrow_links:
        if target_counts[dst_idx] != 1:
            continue
        dr, dc = _move_delta(arrow["direction"], objects[src_idx]["bbox"], objects[dst_idx]["bbox"])
        for r, c in objects[dst_idx]["coords"]:
            nr, nc = r + dr, c + dc
            assert 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]
            out[nr, nc] = objects[dst_idx]["color"]

    return out


def _cells_from_mask(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    cells = set()
    for r, c in np.argwhere(mask):
        cells.add((int(top + r), int(left + c)))
    return cells


def _in_bounds(cells: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _padded(cells: set[tuple[int, int]], shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    out = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _shape_is_arrow(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    return _parse_arrow(coords) is not None


def _random_shape(required_side: str) -> dict:
    for _ in range(100):
        h = random.randint(1, 4)
        w = random.randint(1, 4)
        if required_side == "left":
            req = (random.randint(0, h - 1), 0)
        elif required_side == "right":
            req = (random.randint(0, h - 1), w - 1)
        elif required_side == "top":
            req = (0, random.randint(0, w - 1))
        else:
            req = (h - 1, random.randint(0, w - 1))

        target_area = random.randint(1, min(h * w, 6))
        cells = {req}
        frontier = {
            (req[0] + dr, req[1] + dc)
            for dr, dc in DIR4
            if 0 <= req[0] + dr < h and 0 <= req[1] + dc < w
        }

        while len(cells) < target_area and frontier:
            nr, nc = random.choice(tuple(frontier))
            frontier.remove((nr, nc))
            cells.add((nr, nc))
            for dr, dc in DIR4:
                rr, cc = nr + dr, nc + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.add((rr, cc))

        if len(cells) != target_area:
            continue

        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        r0, c0 = min(rows), min(cols)
        r1, c1 = max(rows), max(cols)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True
        req_rel = (req[0] - r0, req[1] - c0)

        if _shape_is_arrow(mask):
            continue

        return {"mask": mask, "req": req_rel}
    raise RuntimeError("failed to create a non-arrow object shape")


def _rectangle_shape(required_side: str) -> dict:
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    if required_side in {"left", "right"}:
        req = (random.randint(0, h - 1), 0 if required_side == "left" else w - 1)
    else:
        req = (0 if required_side == "top" else h - 1, random.randint(0, w - 1))
    return {"mask": np.ones((h, w), dtype=bool), "req": req}


def _arrow_cells(direction: str, line_a: int, line_b: int, shift: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if direction == "right":
        for c in range(line_b, line_b + shift - 1):
            cells.add((line_a, c))
        cells.add((line_a - 1, line_b + shift - 1))
        cells.add((line_a + 1, line_b + shift - 1))
    elif direction == "left":
        for c in range(line_b, line_b - shift + 1, -1):
            cells.add((line_a, c))
        cells.add((line_a - 1, line_b - shift + 1))
        cells.add((line_a + 1, line_b - shift + 1))
    elif direction == "down":
        for r in range(line_b, line_b + shift - 1):
            cells.add((r, line_a))
        cells.add((line_b + shift - 1, line_a - 1))
        cells.add((line_b + shift - 1, line_a + 1))
    else:
        for r in range(line_b, line_b - shift + 1, -1):
            cells.add((r, line_a))
        cells.add((line_b - shift + 1, line_a - 1))
        cells.add((line_b - shift + 1, line_a + 1))
    return cells


def _nonconflict_bundle(shape: tuple[int, int], a_color: int, b_color: int, direction: str) -> dict | None:
    a_side, b_side = DIR_SIDES[direction]
    a_shape = _random_shape(a_side)
    b_shape = _random_shape(b_side)
    h, w = shape

    for _ in range(200):
        shift = random.randint(3, 6)
        if direction in {"right", "left"}:
            row = random.randint(1, h - 2)
            tail = random.randint(0, w - 1)
            if direction == "right":
                a_world = (row, tail - 1)
                b_out_world = (row, tail)
                b_in_world = (row, tail + shift)
            else:
                a_world = (row, tail + 1)
                b_out_world = (row, tail)
                b_in_world = (row, tail - shift)

            a_top = a_world[0] - a_shape["req"][0]
            a_left = a_world[1] - a_shape["req"][1]
            b_out_top = b_out_world[0] - b_shape["req"][0]
            b_out_left = b_out_world[1] - b_shape["req"][1]
            b_in_top = b_in_world[0] - b_shape["req"][0]
            b_in_left = b_in_world[1] - b_shape["req"][1]
            arrow = _arrow_cells(direction, row, tail, shift)
        else:
            col = random.randint(1, w - 2)
            tail = random.randint(0, h - 1)
            if direction == "down":
                a_world = (tail - 1, col)
                b_out_world = (tail, col)
                b_in_world = (tail + shift, col)
            else:
                a_world = (tail + 1, col)
                b_out_world = (tail, col)
                b_in_world = (tail - shift, col)

            a_top = a_world[0] - a_shape["req"][0]
            a_left = a_world[1] - a_shape["req"][1]
            b_out_top = b_out_world[0] - b_shape["req"][0]
            b_out_left = b_out_world[1] - b_shape["req"][1]
            b_in_top = b_in_world[0] - b_shape["req"][0]
            b_in_left = b_in_world[1] - b_shape["req"][1]
            arrow = _arrow_cells(direction, col, tail, shift)

        a_cells = _cells_from_mask(a_shape["mask"], a_top, a_left)
        b_in_cells = _cells_from_mask(b_shape["mask"], b_in_top, b_in_left)
        b_out_cells = _cells_from_mask(b_shape["mask"], b_out_top, b_out_left)

        if not (_in_bounds(a_cells, shape) and _in_bounds(b_in_cells, shape) and _in_bounds(b_out_cells, shape) and _in_bounds(arrow, shape)):
            continue
        if a_cells & b_in_cells:
            continue
        if a_cells & arrow or b_in_cells & arrow:
            continue
        if a_cells & b_out_cells or b_out_cells == b_in_cells:
            continue

        return {
            "input_objects": [(a_color, a_cells), (b_color, b_in_cells)],
            "output_objects": [(a_color, a_cells), (b_color, b_out_cells)],
            "arrows": [arrow],
            "cells": a_cells | b_in_cells | b_out_cells | arrow,
        }
    return None


def _conflict_bundle(shape: tuple[int, int], colors: list[int], directions: list[str]) -> dict | None:
    h, w = shape
    b_color = colors[0]
    a_colors = colors[1:]
    required_side = random.choice(["left", "right", "top", "bottom"])
    b_shape = _rectangle_shape(required_side)
    bh, bw = b_shape["mask"].shape

    for _ in range(250):
        b_top = random.randint(0, h - bh)
        b_left = random.randint(0, w - bw)
        b_cells = _cells_from_mask(b_shape["mask"], b_top, b_left)
        input_objects = [(b_color, b_cells)]
        output_objects = [(b_color, b_cells)]
        arrows: list[set[tuple[int, int]]] = []
        all_cells = set(b_cells)
        ok = True

        for direction, a_color in zip(directions, a_colors):
            a_side, b_side = DIR_SIDES[direction]
            a_shape = _random_shape(a_side)
            shift = random.randint(3, 6)

            if direction == "right":
                row = b_top + random.randint(0, bh - 1)
                b_world = (row, b_left)
                tail = b_world[1] - shift
                a_world = (row, tail - 1)
                a_top = a_world[0] - a_shape["req"][0]
                a_left = a_world[1] - a_shape["req"][1]
                arrow = _arrow_cells(direction, row, tail, shift)
            elif direction == "left":
                row = b_top + random.randint(0, bh - 1)
                b_world = (row, b_left + bw - 1)
                tail = b_world[1] + shift
                a_world = (row, tail + 1)
                a_top = a_world[0] - a_shape["req"][0]
                a_left = a_world[1] - a_shape["req"][1]
                arrow = _arrow_cells(direction, row, tail, shift)
            elif direction == "down":
                col = b_left + random.randint(0, bw - 1)
                b_world = (b_top, col)
                tail = b_world[0] - shift
                a_world = (tail - 1, col)
                a_top = a_world[0] - a_shape["req"][0]
                a_left = a_world[1] - a_shape["req"][1]
                arrow = _arrow_cells(direction, col, tail, shift)
            else:
                col = b_left + random.randint(0, bw - 1)
                b_world = (b_top + bh - 1, col)
                tail = b_world[0] + shift
                a_world = (tail + 1, col)
                a_top = a_world[0] - a_shape["req"][0]
                a_left = a_world[1] - a_shape["req"][1]
                arrow = _arrow_cells(direction, col, tail, shift)

            a_cells = _cells_from_mask(a_shape["mask"], a_top, a_left)
            if not (_in_bounds(a_cells, shape) and _in_bounds(arrow, shape)):
                ok = False
                break
            if (a_cells | arrow) & all_cells:
                ok = False
                break

            input_objects.append((a_color, a_cells))
            output_objects.append((a_color, a_cells))
            arrows.append(arrow)
            all_cells |= a_cells | arrow

        if ok:
            return {
                "input_objects": input_objects,
                "output_objects": output_objects,
                "arrows": arrows,
                "cells": all_cells,
            }
    return None


def _distractor_bundle(shape: tuple[int, int], color: int) -> dict | None:
    h, w = shape
    side = random.choice(["left", "right", "top", "bottom"])
    obj_shape = _random_shape(side)
    oh, ow = obj_shape["mask"].shape

    for _ in range(100):
        top = random.randint(0, h - oh)
        left = random.randint(0, w - ow)
        cells = _cells_from_mask(obj_shape["mask"], top, left)
        if not _in_bounds(cells, shape):
            continue
        return {
            "input_objects": [(color, cells)],
            "output_objects": [(color, cells)],
            "arrows": [],
            "cells": cells,
        }
    return None


def _paint(shape: tuple[int, int], bg: int, objects: list[tuple[int, set[tuple[int, int]]]], arrows: list[set[tuple[int, int]]]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for color, cells in objects:
        for r, c in cells:
            grid[r, c] = color
    for arrow in arrows:
        for r, c in arrow:
            grid[r, c] = ARROW_COLOR
    return grid


def _scene_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    shape = (h, w)

    palette = [c for c in range(10) if c != ARROW_COLOR]
    bg = random.choice(palette)
    palette.remove(bg)
    random.shuffle(palette)

    include_conflict = random.random() < 0.5
    non_count = random.randint(1, 2) if include_conflict else random.randint(1, 3)
    conflict_dirs = []
    if include_conflict:
        conflict_dirs = random.sample(list(DIR_VECTORS), k=random.choice([2, 3]))

    needed = non_count * 2 + (1 + len(conflict_dirs) if include_conflict else 0)
    if needed > len(palette):
        return None

    objects_in: list[tuple[int, set[tuple[int, int]]]] = []
    objects_out: list[tuple[int, set[tuple[int, int]]]] = []
    arrows: list[set[tuple[int, int]]] = []
    reserved: set[tuple[int, int]] = set()

    def place_bundle(bundle: dict) -> bool:
        nonlocal reserved
        if _padded(bundle["cells"], shape) & reserved:
            return False
        for color, cells in bundle["input_objects"]:
            objects_in.append((color, cells))
        for color, cells in bundle["output_objects"]:
            objects_out.append((color, cells))
        arrows.extend(bundle["arrows"])
        reserved |= _padded(bundle["cells"], shape)
        return True

    for _ in range(non_count):
        direction = random.choice(list(DIR_VECTORS))
        a_color = palette.pop()
        b_color = palette.pop()
        placed = False
        for _attempt in range(80):
            bundle = _nonconflict_bundle(shape, a_color, b_color, direction)
            if bundle is not None and place_bundle(bundle):
                placed = True
                break
        if not placed:
            return None

    if include_conflict:
        use = 1 + len(conflict_dirs)
        group_colors = [palette.pop() for _ in range(use)]
        placed = False
        for _attempt in range(80):
            bundle = _conflict_bundle(shape, group_colors, conflict_dirs)
            if bundle is not None and place_bundle(bundle):
                placed = True
                break
        if not placed:
            return None

    max_distractors = min(2, len(palette))
    distractor_count = random.randint(0, max_distractors)
    for _ in range(distractor_count):
        color = palette.pop()
        placed = False
        for _attempt in range(50):
            bundle = _distractor_bundle(shape, color)
            if bundle is not None and place_bundle(bundle):
                placed = True
                break
        if not placed:
            return None

    inp = _paint(shape, bg, objects_in, arrows)
    out = _paint(shape, bg, objects_out, [])
    bg_count = int(np.sum(inp == bg))
    fg_count = int(inp.size - bg_count)
    if bg_count <= fg_count:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        scene = _scene_attempt()
        if scene is None:
            continue
        inp, out = scene
        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, out):
            return inp
    raise RuntimeError("failed to generate a valid input")
