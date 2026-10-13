import random
from collections import deque

import numpy as np


BG = 0
GRAY = 5
AZURE = 8
MAROON = 9
TARGET_COLORS = [1, 2, 3, 4, 6, 7]
DIR_TO_SIDE = {0: "top", 1: "right", 2: "bottom", 3: "left"}
SIDE_TO_TURNS = {"top": 0, "right": 1, "bottom": 2, "left": 3}
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_RETRIES = 300
SHAPE_RETRIES = 200
PLACE_RETRIES = 250


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _coords_from_mask(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for rr, cc in np.argwhere(mask):
        coords.append((r0 + int(rr), c0 + int(cc)))
    return coords


def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


def _rotated_coords_from_mask(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    turns_cw: int,
) -> list[tuple[int, int]]:
    turns_cw %= 4
    if turns_cw == 0:
        return _coords_from_mask(mask, bbox[0], bbox[1])
    rot = np.rot90(mask, -turns_cw)
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    new_h, new_w = rot.shape
    new_r0 = _round_half_up(center_r - (new_h - 1) / 2.0)
    new_c0 = _round_half_up(center_c - (new_w - 1) / 2.0)
    return _coords_from_mask(rot, new_r0, new_c0)


def _rotation_signatures(mask: np.ndarray) -> list[tuple[int, int, tuple[int, ...]]]:
    sigs = []
    for k in range(4):
        rot = np.rot90(mask, k)
        sigs.append((int(rot.shape[0]), int(rot.shape[1]), tuple(rot.astype(int).ravel())))
    return sigs


def _canonical_signature(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    return min(_rotation_signatures(mask))


def _find_dictionary_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    gray_components = _components(grid == GRAY)
    if not gray_components:
        raise ValueError("no gray dictionary frame found")
    best = min((_bbox(comp) for comp in gray_components), key=lambda b: (b[0], b[1], b[2] - b[0], b[3] - b[1]))
    return best


def _parse_dictionary_entries(
    grid: np.ndarray, bbox: tuple[int, int, int, int]
) -> list[dict[str, object]]:
    r0, c0, r1, c1 = bbox
    separator_rows = [r for r in range(r0, r1 + 1) if np.all(grid[r, c0 : c1 + 1] == GRAY)]
    entries: list[dict[str, object]] = []
    for top, bottom in zip(separator_rows, separator_rows[1:]):
        if bottom - top <= 1:
            continue
        patch = grid[top + 1 : bottom, c0 + 1 : c1]
        azure_parts = _components(patch == AZURE)
        swatch_parts = _components((patch != BG) & (patch != AZURE))
        if len(azure_parts) != 1 or len(swatch_parts) != 1:
            continue
        sample_mask = _mask_from_coords(azure_parts[0])
        swatch_vals = {int(patch[r, c]) for r, c in swatch_parts[0]}
        if len(swatch_vals) != 1:
            continue
        entries.append({"mask": sample_mask, "color": swatch_vals.pop()})
    if not entries:
        raise ValueError("dictionary parsing found no entries")
    return entries


def _marker_side_and_cell(
    grid: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[str, tuple[int, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    found: list[tuple[str, tuple[int, int]]] = []
    if r0 - 1 >= 0:
        for c in range(c0, c1 + 1):
            if grid[r0 - 1, c] == MAROON:
                found.append(("top", (r0 - 1, c)))
    if c1 + 1 < w:
        for r in range(r0, r1 + 1):
            if grid[r, c1 + 1] == MAROON:
                found.append(("right", (r, c1 + 1)))
    if r1 + 1 < h:
        for c in range(c0, c1 + 1):
            if grid[r1 + 1, c] == MAROON:
                found.append(("bottom", (r1 + 1, c)))
    if c0 - 1 >= 0:
        for r in range(r0, r1 + 1):
            if grid[r, c0 - 1] == MAROON:
                found.append(("left", (r, c0 - 1)))
    if len(found) != 1:
        raise ValueError("each object must have exactly one adjacent maroon direction marker")
    return found[0]


def _match_dictionary(mask: np.ndarray, entries: list[dict[str, object]]) -> int:
    matches = []
    for idx, entry in enumerate(entries):
        sample = entry["mask"]
        for k in range(4):
            if np.array_equal(np.rot90(mask, k), sample):
                matches.append(idx)
                break
    if len(matches) != 1:
        raise ValueError("object shape must match exactly one dictionary entry")
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each azure object from the top-left shape dictionary, then rotate it by the adjacent maroon direction marker.

    Input structure:
    - The background is black.
    - A gray framed dictionary sits in the top-left; gray horizontal lines split it into rows.
    - Each dictionary row contains one azure binary sample shape and, after a one-cell black gap, one solid color swatch.
    - Elsewhere in the grid, several 4-connected azure shape objects appear, and each object has exactly one maroon marker pixel touching one side of its bounding box from outside.

    Rule:
    - Parse the dictionary rows into shape-to-color entries, using the azure sample mask and the swatch color.
    - For each azure object outside the dictionary, match its binary mask to the unique dictionary sample that agrees under some 0/90/180/270-degree rotation.
    - Recolor the whole object to that matched swatch color.
    - Read the maroon marker side: top means rotate 0 degrees, right means rotate 90 degrees clockwise, bottom means rotate 180 degrees, and left means rotate 270 degrees clockwise; erase the maroon marker in the output.
    - Rotate the recolored object around the center of its own bounding box and write it back onto the same grid while leaving the dictionary unchanged.

    Key insight:
    - The gray panel is a shape dictionary for color lookup, while the maroon pixel is an independent rotation code that must be applied after the shape has been identified and recolored.
    """
    g = np.asarray(grid, dtype=int)
    dict_bbox = _find_dictionary_bbox(g)
    entries = _parse_dictionary_entries(g, dict_bbox)

    object_mask = g == AZURE
    r0, c0, r1, c1 = dict_bbox
    object_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    objects = _components(object_mask)

    out = g.copy()
    out[g == MAROON] = BG
    out[object_mask] = BG

    for coords in objects:
        bbox = _bbox(coords)
        mask = _mask_from_coords(coords)
        entry_idx = _match_dictionary(mask, entries)
        color = int(entries[entry_idx]["color"])
        side, _marker = _marker_side_and_cell(g, bbox)
        turns = SIDE_TO_TURNS[side]
        rotated = _rotated_coords_from_mask(mask, bbox, turns)
        for rr, cc in rotated:
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = color
    return out


def _random_connected_shape(rng: random.Random) -> np.ndarray:
    dim_choices = [(2, 2), (2, 4), (4, 2), (3, 3), (4, 4)]
    for _ in range(SHAPE_RETRIES):
        h, w = rng.choice(dim_choices)
        target = rng.randint(3, min(6, h * w - 1))
        cells = {(rng.randrange(h), rng.randrange(w))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != target:
            continue
        raw = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            raw[r, c] = True
        comp = _components(raw)
        if len(comp) != 1:
            continue
        mask = _mask_from_coords(comp[0])
        if min(mask.shape) < 2:
            continue
        if (mask.shape[0] % 2) != (mask.shape[1] % 2):
            continue
        if len(set(_rotation_signatures(mask))) != 4:
            continue
        return mask
    raise RuntimeError("failed to build a non-ambiguous shape")


def _build_dictionary_patch(
    rng: random.Random, shapes: list[np.ndarray], colors: list[int]
) -> tuple[np.ndarray, list[dict[str, object]]]:
    max_w = max(int(shape.shape[1]) for shape in shapes)
    sample_field_w = max_w + 1
    swatch_h = 2
    swatch_w = 2
    row_heights = [max(int(shape.shape[0]), swatch_h) + 1 for shape in shapes]
    patch_h = 1 + sum(row_heights) + len(row_heights)
    patch_w = 2 + sample_field_w + 1 + swatch_w + 1
    patch = np.full((patch_h, patch_w), BG, dtype=int)
    patch[:, 0] = GRAY
    patch[:, -1] = GRAY
    separators = [0]
    cursor = 1
    entries: list[dict[str, object]] = []
    for row_h, shape, color in zip(row_heights, shapes, colors):
        sep = cursor + row_h
        separators.append(sep)
        row_top = cursor
        sh, sw = shape.shape
        sr0 = row_top + (row_h - sh) // 2
        sc0 = 1 + rng.randint(0, sample_field_w - sw)
        patch[sr0 : sr0 + sh, sc0 : sc0 + sw][shape] = AZURE

        swatch_r0 = row_top + (row_h - swatch_h) // 2
        swatch_c0 = 1 + sample_field_w + 1
        patch[swatch_r0 : swatch_r0 + swatch_h, swatch_c0 : swatch_c0 + swatch_w] = color

        entries.append({"mask": shape.copy(), "color": color})
        cursor = sep + 1
    for r in separators:
        patch[r, :] = GRAY
    return patch, entries


def _balanced_turns(rng: random.Random, n: int) -> list[int]:
    seq: list[int] = []
    while len(seq) < n:
        block = [0, 1, 2, 3]
        rng.shuffle(block)
        seq.extend(block)
    return seq[:n]


def _marker_candidates(
    bbox: tuple[int, int, int, int], side: str, grid_h: int, grid_w: int
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top" and r0 - 1 >= 0:
        return [(r0 - 1, c) for c in range(c0, c1 + 1)]
    if side == "right" and c1 + 1 < grid_w:
        return [(r, c1 + 1) for r in range(r0, r1 + 1)]
    if side == "bottom" and r1 + 1 < grid_h:
        return [(r1 + 1, c) for c in range(c0, c1 + 1)]
    if side == "left" and c0 - 1 >= 0:
        return [(r, c0 - 1) for r in range(r0, r1 + 1)]
    return []


def _cells_in_bbox(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _halo(cells: set[tuple[int, int]], grid_h: int, grid_w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_h and 0 <= nc < grid_w:
                    out.add((nr, nc))
    return out


def _inside(bounds_h: int, bounds_w: int, cells: set[tuple[int, int]]) -> bool:
    return all(0 <= r < bounds_h and 0 <= c < bounds_w for r, c in cells)


def _place_objects(
    rng: random.Random,
    grid_h: int,
    grid_w: int,
    dict_bbox: tuple[int, int, int, int],
    dict_entries: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    object_count = rng.randint(max(4, len(dict_entries)), 6)
    shape_ids = list(range(len(dict_entries)))
    while len(shape_ids) < object_count:
        shape_ids.append(rng.randrange(len(dict_entries)))
    rng.shuffle(shape_ids)

    input_turns = _balanced_turns(rng, object_count)
    output_turns = _balanced_turns(rng, object_count)

    reserved = _halo(_cells_in_bbox(dict_bbox), grid_h, grid_w)
    specs: list[dict[str, object]] = []

    for shape_id, obj_turn, dir_turn in zip(shape_ids, input_turns, output_turns):
        base_mask = dict_entries[shape_id]["mask"]
        obj_mask = np.rot90(base_mask, -obj_turn)
        oh, ow = obj_mask.shape
        placed = None
        for _ in range(PLACE_RETRIES):
            r0 = rng.randint(0, grid_h - oh)
            c0 = rng.randint(0, grid_w - ow)
            bbox = (r0, c0, r0 + oh - 1, c0 + ow - 1)
            input_cells = set(_coords_from_mask(obj_mask, r0, c0))
            output_cells = set(_rotated_coords_from_mask(obj_mask, bbox, dir_turn))
            if not _inside(grid_h, grid_w, output_cells):
                continue
            side = DIR_TO_SIDE[dir_turn]
            marker_options = _marker_candidates(bbox, side, grid_h, grid_w)
            rng.shuffle(marker_options)
            for marker in marker_options:
                union = set(input_cells) | set(output_cells) | {marker}
                if union & _cells_in_bbox(dict_bbox):
                    continue
                if _halo(union, grid_h, grid_w) & reserved:
                    continue
                placed = {
                    "input_mask": obj_mask,
                    "input_bbox": bbox,
                    "input_cells": input_cells,
                    "output_cells": output_cells,
                    "marker": marker,
                    "color": int(dict_entries[shape_id]["color"]),
                }
                reserved |= _halo(union, grid_h, grid_w)
                break
            if placed is not None:
                break
        if placed is None:
            return None
        specs.append(placed)
    return specs


def _render_pair(
    grid_h: int,
    grid_w: int,
    dict_patch: np.ndarray,
    object_specs: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((grid_h, grid_w), BG, dtype=int)
    out = np.full((grid_h, grid_w), BG, dtype=int)
    ph, pw = dict_patch.shape
    inp[:ph, :pw] = dict_patch
    out[:ph, :pw] = dict_patch
    for spec in object_specs:
        for r, c in spec["input_cells"]:
            inp[r, c] = AZURE
        mr, mc = spec["marker"]
        inp[mr, mc] = MAROON
        for r, c in spec["output_cells"]:
            out[r, c] = spec["color"]
    return inp, out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    dict_count = rng.randint(2, 4)
    shapes: list[np.ndarray] = []
    seen = set()
    while len(shapes) < dict_count:
        shape = _random_connected_shape(rng)
        sig = _canonical_signature(shape)
        if sig in seen:
            continue
        seen.add(sig)
        shapes.append(shape)
    colors = random.sample(TARGET_COLORS, dict_count)
    rng.shuffle(shapes)
    dict_patch, dict_entries = _build_dictionary_patch(rng, shapes, colors)
    dict_h, dict_w = dict_patch.shape

    min_h = max(dict_h, 16)
    min_w = max(dict_w + 10, 18)
    if min_h > 30 or min_w > 30:
        return None
    grid_h = rng.randint(min_h, 30)
    grid_w = rng.randint(min_w, 30)

    dict_bbox = (0, 0, dict_h - 1, dict_w - 1)
    object_specs = _place_objects(rng, grid_h, grid_w, dict_bbox, dict_entries)
    if object_specs is None:
        return None

    inp, expected = _render_pair(grid_h, grid_w, dict_patch, object_specs)
    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
