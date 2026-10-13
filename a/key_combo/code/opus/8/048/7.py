import random
from collections import deque
from itertools import combinations

import numpy as np


BG = 0
RED = 2
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
FILL_COLORS = [1, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 500

HOLE_SHAPES = [
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
]


def _in_bounds(shape, r, c):
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components_from_mask(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_center(bbox):
    r0, c0, r1, c1 = bbox
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _normalize(coords):
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _coords_to_patch(coords, color):
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = color
    return patch


def _boundary_cells(mask):
    h, w = mask.shape
    out = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if not _in_bounds(mask.shape, nr, nc) or not mask[nr, nc]:
                out.add((rr, cc))
                break
    return out


def _holes_inside_mask(mask):
    comps = _components_from_mask(mask)
    if len(comps) != 1:
        return []
    r0, c0, r1, c1 = _bbox(comps[0])
    window = mask[r0 : r1 + 1, c0 : c1 + 1]
    empty = ~window
    external = np.zeros_like(empty, dtype=bool)
    q = deque()
    h, w = empty.shape
    for r in range(h):
        for c in (0, w - 1):
            if empty[r, c] and not external[r, c]:
                external[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if empty[r, c] and not external[r, c]:
                external[r, c] = True
                q.append((r, c))
    while q:
        rr, cc = q.popleft()
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not external[nr, nc]:
                external[nr, nc] = True
                q.append((nr, nc))
    hole_mask = empty & ~external
    holes = []
    for comp in _components_from_mask(hole_mask):
        holes.append({(r + r0, c + c0) for r, c in comp})
    return holes


def _find_large_object_and_pieces(grid):
    comps = _components_from_mask(grid != BG)
    candidates = []
    pieces = []
    for coords in comps:
        colors = {int(grid[r, c]) for r, c in coords}
        bbox = _bbox(coords)
        if colors == {RED}:
            mask = np.zeros_like(grid, dtype=bool)
            for r, c in coords:
                mask[r, c] = True
            holes = _holes_inside_mask(mask)
            if holes:
                candidates.append(
                    {
                        "coords": coords,
                        "bbox": bbox,
                        "mask": mask,
                        "holes": holes,
                        "area": len(coords),
                    }
                )
        elif RED in colors:
            anchor_cells = [(r, c) for r, c in coords if int(grid[r, c]) == RED]
            fill_cells = [(r, c) for r, c in coords if int(grid[r, c]) not in (BG, RED)]
            if anchor_cells and fill_cells:
                pieces.append(
                    {
                        "coords": coords,
                        "bbox": bbox,
                        "anchor_cells": anchor_cells,
                        "fill_cells": fill_cells,
                        "fill_values": [int(grid[r, c]) for r, c in fill_cells],
                    }
                )
    assert candidates, "expected one red holed object"
    obj = max(candidates, key=lambda item: item["area"])
    obj["boundary"] = _boundary_cells(obj["mask"])
    return obj, pieces


def _piece_candidates(piece, boundary, holes, grid_shape):
    h, w = grid_shape
    hole_map = {}
    for idx, hole in enumerate(holes):
        for cell in hole:
            hole_map[cell] = idx
    valid = []
    anchor_only = 0
    for dr in range(-h + 1, h):
        for dc in range(-w + 1, w):
            shifted_anchors = []
            good = True
            for r, c in piece["anchor_cells"]:
                nr, nc = r + dr, c + dc
                if not _in_bounds(grid_shape, nr, nc) or (nr, nc) not in boundary:
                    good = False
                    break
                shifted_anchors.append((nr, nc))
            if not good:
                continue
            anchor_only += 1
            shifted_fill = []
            hole_ids = set()
            for r, c in piece["fill_cells"]:
                nr, nc = r + dr, c + dc
                if not _in_bounds(grid_shape, nr, nc) or (nr, nc) not in hole_map:
                    good = False
                    break
                shifted_fill.append((nr, nc))
                hole_ids.add(hole_map[(nr, nc)])
            if not good or len(hole_ids) != 1:
                continue
            hole_idx = next(iter(hole_ids))
            if set(shifted_fill) != holes[hole_idx]:
                continue
            valid.append({"dr": dr, "dc": dc, "hole_idx": hole_idx})
    return valid, anchor_only


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each outer mixed-color fragment so its red anchor pixels align to the red object's boundary and its non-red pixels fill one hole.

    Input structure:
    - The background is black.
    - One large red object occupies the middle of the grid and contains enclosed black holes.
    - Several separate outer fragments are 4-connected mixed-color pieces containing red anchor pixels plus non-red fill pixels.

    Rule:
    - Identify the large red object, its boundary pixels, and its enclosed black holes.
    - For each outer fragment, try translations until its red pixels land on red boundary pixels of the large object and its non-red pixels land exactly on one hole.
    - Process fragments from nearer to farther relative to the large object's center, erase their source locations, and paint only the translated non-red pixels into the holes.
    - Keep the large red object unchanged.

    Key insight:
    - The red cells in each fragment are not output paint: they are alignment anchors whose only job is to reveal the unique translation that sends the fragment's colored cells into a hole.
    """
    g = np.asarray(grid, dtype=int)
    obj, pieces = _find_large_object_and_pieces(g)
    holes = obj["holes"]
    boundary = obj["boundary"]
    object_center = _bbox_center(obj["bbox"])

    solved = []
    for idx, piece in enumerate(pieces):
        candidates, _anchor_only = _piece_candidates(piece, boundary, holes, g.shape)
        assert len(candidates) == 1, "each piece must have a unique valid translation"
        solved.append((idx, piece, candidates[0]))

    def distance_key(item):
        _idx, piece, _placement = item
        pr, pc = _bbox_center(piece["bbox"])
        dist2 = (pr - object_center[0]) ** 2 + (pc - object_center[1]) ** 2
        return (dist2, piece["bbox"])

    out = g.copy()
    for _idx, piece, _placement in solved:
        for r, c in piece["coords"]:
            out[r, c] = BG

    for _idx, piece, placement in sorted(solved, key=distance_key):
        dr = placement["dr"]
        dc = placement["dc"]
        for (r, c), color in zip(piece["fill_cells"], piece["fill_values"]):
            nr, nc = r + dr, c + dc
            if out[nr, nc] == BG:
                out[nr, nc] = color
    return out


def _make_irregular_object(h, w):
    for _ in range(80):
        mask = np.ones((h, w), dtype=bool)
        sides = random.sample(["top", "bottom", "left", "right"], k=random.randint(2, 4))
        for side in sides:
            if side in ("top", "bottom"):
                max_len = max(1, min(w - 2, 4))
                length = random.randint(1, max_len)
                c0 = random.randint(1, w - length - 1)
                depth = random.randint(1, min(2, h - 2))
                rows = slice(0, depth) if side == "top" else slice(h - depth, h)
                mask[rows, c0 : c0 + length] = False
            else:
                max_len = max(1, min(h - 2, 4))
                length = random.randint(1, max_len)
                r0 = random.randint(1, h - length - 1)
                depth = random.randint(1, min(2, w - 2))
                cols = slice(0, depth) if side == "left" else slice(w - depth, w)
                mask[r0 : r0 + length, cols] = False
        if len(_components_from_mask(mask)) == 1 and not np.all(mask):
            return mask
    return None


def _can_carve_hole(mask, hole_cells):
    hole_set = set(hole_cells)
    for r, c in hole_cells:
        if not _in_bounds(mask.shape, r, c) or not mask[r, c]:
            return False
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if (nr, nc) in hole_set:
                continue
            if not _in_bounds(mask.shape, nr, nc) or not mask[nr, nc]:
                return False
    return True


def _choose_holes(mask):
    areas = {3: [], 4: [], 5: []}
    for shape in HOLE_SHAPES:
        areas[len(shape)].append(shape)
    picked = [
        random.choice(areas[3]),
        random.choice(areas[4]),
        random.choice(areas[5]),
    ]
    random.shuffle(picked)

    work = mask.copy()
    placed = []
    for shape in picked:
        sh = 1 + max(r for r, _ in shape)
        sw = 1 + max(c for _, c in shape)
        candidates = []
        for r0 in range(1, work.shape[0] - sh):
            for c0 in range(1, work.shape[1] - sw):
                coords = [(r0 + r, c0 + c) for r, c in shape]
                if _can_carve_hole(work, coords):
                    candidates.append(coords)
        if not candidates:
            return None
        coords = random.choice(candidates)
        for r, c in coords:
            work[r, c] = False
        placed.append(set(coords))
    if len(_components_from_mask(work)) != 1:
        return None
    if len(_holes_inside_mask(work)) != 3:
        return None
    return work, placed


def _pattern_occurrences(boundary, anchor_subset):
    pattern = _normalize(anchor_subset)
    boundary = set(boundary)
    count = 0
    for base_r, base_c in boundary:
        translated = {(base_r + dr, base_c + dc) for dr, dc in pattern}
        if translated <= boundary:
            count += 1
    return count


def _choose_anchors(hole, object_mask, boundary):
    perimeter = set()
    for r, c in hole:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if _in_bounds(object_mask.shape, nr, nc) and object_mask[nr, nc]:
                perimeter.add((nr, nc))
    perimeter = sorted(perimeter)
    options = []
    for size in (2, 3):
        if len(hole) + size > 8 or len(perimeter) < size:
            continue
        for combo in combinations(perimeter, size):
            if _pattern_occurrences(boundary, combo) > 1:
                options.append(combo)
    if not options:
        for size in (2, 3):
            if len(hole) + size > 8 or len(perimeter) < size:
                continue
            options.extend(combinations(perimeter, size))
    if not options:
        return None
    return set(random.choice(options))


def _build_piece_spec(hole, anchors, fill_color):
    union = set(hole) | set(anchors)
    r0, c0, r1, c1 = _bbox(list(union))
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in hole:
        patch[r - r0, c - c0] = fill_color
    for r, c in anchors:
        patch[r - r0, c - c0] = RED
    return {
        "patch": patch,
        "target_origin": (r0, c0),
        "target_union": union,
        "hole": set(hole),
        "anchors": set(anchors),
    }


def _draw_patch(grid, patch, top, left):
    for r, c in np.argwhere(patch != BG):
        grid[top + r, left + c] = int(patch[r, c])


def _can_place_patch(grid, patch, top, left):
    h, w = patch.shape
    if top < 0 or left < 0 or top + h > grid.shape[0] or left + w > grid.shape[1]:
        return False
    for rr, cc in np.argwhere(patch != BG):
        r = top + int(rr)
        c = left + int(cc)
        for dr, dc in DIR8 + [(0, 0)]:
            nr, nc = r + dr, c + dc
            if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] != BG:
                return False
    return True


def _piece_position_candidates(grid, patch):
    h, w = patch.shape
    candidates = []
    for top in range(grid.shape[0] - h + 1):
        for left in range(grid.shape[1] - w + 1):
            if _can_place_patch(grid, patch, top, left):
                candidates.append((top, left))
    random.shuffle(candidates)
    return candidates


def _input_output_pair():
    grid_h = random.randint(19, 26)
    grid_w = random.randint(19, 26)
    obj_h = random.randint(10, 13)
    obj_w = random.randint(10, 13)
    object_mask = _make_irregular_object(obj_h, obj_w)
    if object_mask is None:
        return None
    carved = _choose_holes(object_mask)
    if carved is None:
        return None
    object_mask, holes = carved
    boundary = _boundary_cells(object_mask)
    fill_colors = random.sample(FILL_COLORS, k=3)
    piece_specs = []
    for hole, fill_color in zip(holes, fill_colors):
        anchors = _choose_anchors(hole, object_mask, boundary)
        if anchors is None:
            return None
        piece_specs.append(_build_piece_spec(hole, anchors, fill_color))

    min_margin = 3
    if grid_h < obj_h + 2 * min_margin or grid_w < obj_w + 2 * min_margin:
        return None
    obj_top = random.randint(min_margin, grid_h - obj_h - min_margin)
    obj_left = random.randint(min_margin, grid_w - obj_w - min_margin)

    output = np.zeros((grid_h, grid_w), dtype=int)
    for r, c in np.argwhere(object_mask):
        output[obj_top + r, obj_left + c] = RED
    for spec in piece_specs:
        for r, c in spec["hole"]:
            output[obj_top + r, obj_left + c] = int(spec["patch"][r - spec["target_origin"][0], c - spec["target_origin"][1]])

    input_grid = output.copy()
    for spec in piece_specs:
        for r, c in spec["hole"]:
            input_grid[obj_top + r, obj_left + c] = BG

    placed_sources = []
    order = list(range(3))
    random.shuffle(order)
    for idx in order:
        spec = piece_specs[idx]
        patch = spec["patch"]
        candidates = _piece_position_candidates(input_grid, patch)
        if not candidates:
            return None
        chosen = None
        for top, left in candidates:
            translated_origin = (obj_top + spec["target_origin"][0], obj_left + spec["target_origin"][1])
            if (top, left) == translated_origin:
                continue
            chosen = (top, left)
            break
        if chosen is None:
            return None
        _draw_patch(input_grid, patch, chosen[0], chosen[1])
        placed_sources.append((idx, chosen))

    return input_grid, output


def _semantic_checks(inp, out):
    obj, pieces = _find_large_object_and_pieces(inp)
    if len(obj["holes"]) != 3 or len(pieces) != 3:
        return False
    boundary = obj["boundary"]
    anchor_ambiguity = 0
    object_center = _bbox_center(obj["bbox"])
    seen_dists = set()
    for piece in pieces:
        candidates, anchor_only = _piece_candidates(piece, boundary, obj["holes"], inp.shape)
        if len(candidates) != 1:
            return False
        if anchor_only > 1:
            anchor_ambiguity += 1
        pr, pc = _bbox_center(piece["bbox"])
        dist_key = round((pr - object_center[0]) ** 2 + (pc - object_center[1]) ** 2, 6)
        if dist_key in seen_dists:
            return False
        seen_dists.add(dist_key)
    if anchor_ambiguity < 2:
        return False
    solved = main(inp)
    return np.array_equal(solved, out)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        pair = _input_output_pair()
        if pair is None:
            continue
        inp, out = pair
        if _semantic_checks(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
