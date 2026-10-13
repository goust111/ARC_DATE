import random

import numpy as np


BG = 0
GREY = 5
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_RETRIES = 300
PLACEMENT_RETRIES = 200


def _shape(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


SHAPE_LIBRARY = [
    _shape(["11"]),
    _shape(["1", "1"]),
    _shape(["111"]),
    _shape(["1", "1", "1"]),
    _shape(["10", "11"]),
    _shape(["01", "11"]),
    _shape(["11", "10"]),
    _shape(["11", "01"]),
    _shape(["111", "010"]),
    _shape(["010", "111"]),
    _shape(["10", "11", "10"]),
    _shape(["01", "11", "01"]),
    _shape(["110", "011"]),
    _shape(["011", "110"]),
]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                if cr > 0 and mask[cr - 1, cc] and not seen[cr - 1, cc]:
                    seen[cr - 1, cc] = True
                    stack.append((cr - 1, cc))
                if cr + 1 < h and mask[cr + 1, cc] and not seen[cr + 1, cc]:
                    seen[cr + 1, cc] = True
                    stack.append((cr + 1, cc))
                if cc > 0 and mask[cr, cc - 1] and not seen[cr, cc - 1]:
                    seen[cr, cc - 1] = True
                    stack.append((cr, cc - 1))
                if cc + 1 < w and mask[cr, cc + 1] and not seen[cr, cc + 1]:
                    seen[cr, cc + 1] = True
                    stack.append((cr, cc + 1))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _shape_key_from_coords(coords: np.ndarray) -> tuple[tuple[int, int], bytes]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return (tuple(int(v) for v in mask.shape), mask.tobytes())


def _shape_key_from_mask(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return (tuple(int(v) for v in mask.shape), mask.astype(bool).tobytes())


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _ring_and_hole(hole: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded_hole = np.pad(hole.astype(bool), 1)
    wall = ~padded_hole
    return wall, padded_hole


def _anchor_for_side(ring: np.ndarray, side: str) -> tuple[int, int]:
    coords = np.argwhere(ring)
    if side == "top":
        coords = coords[coords[:, 0] == coords[:, 0].min()]
        coords = coords[np.argsort(coords[:, 1])]
    elif side == "bottom":
        coords = coords[coords[:, 0] == coords[:, 0].max()]
        coords = coords[np.argsort(coords[:, 1])]
    elif side == "left":
        coords = coords[coords[:, 1] == coords[:, 1].min()]
        coords = coords[np.argsort(coords[:, 0])]
    else:
        coords = coords[coords[:, 1] == coords[:, 1].max()]
        coords = coords[np.argsort(coords[:, 0])]
    chosen = coords[len(coords) // 2]
    return int(chosen[0]), int(chosen[1])


def _extract_sprite_map(grid: np.ndarray) -> dict[tuple[tuple[int, int], bytes], int]:
    sprite_map: dict[tuple[tuple[int, int], bytes], int] = {}
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, GREY):
            continue
        for comp in _components(grid == color):
            key = _shape_key_from_coords(comp)
            if key in sprite_map and sprite_map[key] != color:
                raise AssertionError("shape legend must be unambiguous")
            sprite_map[key] = color
    return sprite_map


def _hole_components_for_frame(grid: np.ndarray, comp: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = _bbox(comp)
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    local = np.zeros(crop.shape, dtype=bool)
    rel = comp - np.array([r0, c0])
    local[rel[:, 0], rel[:, 1]] = True

    outer_border = np.zeros_like(local)
    outer_border[0, :] = local[0, :]
    outer_border[-1, :] = local[-1, :]
    outer_border[:, 0] = local[:, 0]
    outer_border[:, -1] = local[:, -1]

    holes: list[np.ndarray] = []
    for black_comp in _components(crop == BG):
        if (
            np.any(black_comp[:, 0] == 0)
            or np.any(black_comp[:, 0] == crop.shape[0] - 1)
            or np.any(black_comp[:, 1] == 0)
            or np.any(black_comp[:, 1] == crop.shape[1] - 1)
        ):
            continue

        touches_outer = False
        for rr, cc in black_comp:
            if rr > 0 and outer_border[rr - 1, cc]:
                touches_outer = True
                break
            if rr + 1 < crop.shape[0] and outer_border[rr + 1, cc]:
                touches_outer = True
                break
            if cc > 0 and outer_border[rr, cc - 1]:
                touches_outer = True
                break
            if cc + 1 < crop.shape[1] and outer_border[rr, cc + 1]:
                touches_outer = True
                break
        if not touches_outer:
            holes.append(black_comp + np.array([r0, c0]))
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey rectangular enclosure by matching its internal hole shape to an outside sprite legend.

    Input structure:
    - The background is black.
    - Several grey rectangular enclosures appear in the grid; each one is a connected grey frame/wall structure whose interior contains one or more enclosed black hole regions.
    - Near the grid edges, small monochrome colored sprites appear on the background, and each sprite's normalized shape matches the hole shape used inside at least one grey enclosure.

    Rule:
    - Extract every non-grey legend sprite as a translation-invariant shape key paired with its color.
    - For each grey enclosure, find the enclosed black hole components inside its bounding box while ignoring the larger open interior background that touches the enclosure's outer rectangle.
    - Match each remaining hole's normalized shape against the outside sprite legend without rotation or reflection.
    - Recolor all grey pixels of that enclosure to the matched legend color, leaving the hole cells and the outside sprites unchanged.

    Key insight:
    - The outside sprites are shape-to-color keys: the enclosed hole geometry determines the enclosure's new border color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    sprite_map = _extract_sprite_map(g)

    for comp in _components(g == GREY):
        holes = _hole_components_for_frame(g, comp)
        if not holes:
            continue
        colors = {sprite_map[_shape_key_from_coords(hole)] for hole in holes}
        if len(colors) != 1:
            raise AssertionError("all holes in one enclosure must map to one legend color")
        out[comp[:, 0], comp[:, 1]] = next(iter(colors))

    return out


def _build_frame_patch(hole: np.ndarray, hole_count: int, side: str) -> np.ndarray:
    ring, padded_hole = _ring_and_hole(hole)
    tile_h, tile_w = ring.shape

    if side in ("top", "bottom"):
        left_clear = random.randint(1, 3)
        right_clear = random.randint(1, 3)
        gap = random.randint(2, 3)
        top_clear = random.randint(2, 4) if side == "top" else random.randint(1, 3)
        bottom_clear = random.randint(2, 4) if side == "bottom" else random.randint(1, 3)
        interior_h = top_clear + tile_h + bottom_clear
        interior_w = left_clear + hole_count * tile_w + (hole_count - 1) * gap + right_clear
        patch = np.zeros((interior_h + 2, interior_w + 2), dtype=int)
        patch[0, :] = GREY
        patch[-1, :] = GREY
        patch[:, 0] = GREY
        patch[:, -1] = GREY
        tile_r = 1 + top_clear
        tile_c = 1 + left_clear
        for _ in range(hole_count):
            patch[tile_r : tile_r + tile_h, tile_c : tile_c + tile_w][ring] = GREY
            anchor_r, anchor_c = _anchor_for_side(ring, side)
            gr = tile_r + anchor_r
            gc = tile_c + anchor_c
            if side == "top":
                patch[1:gr, gc] = GREY
            else:
                patch[gr + 1 : -1, gc] = GREY
            tile_c += tile_w + gap
    else:
        top_clear = random.randint(1, 3)
        bottom_clear = random.randint(1, 3)
        gap = random.randint(2, 3)
        left_clear = random.randint(2, 4) if side == "left" else random.randint(1, 3)
        right_clear = random.randint(2, 4) if side == "right" else random.randint(1, 3)
        interior_h = top_clear + hole_count * tile_h + (hole_count - 1) * gap + bottom_clear
        interior_w = left_clear + tile_w + right_clear
        patch = np.zeros((interior_h + 2, interior_w + 2), dtype=int)
        patch[0, :] = GREY
        patch[-1, :] = GREY
        patch[:, 0] = GREY
        patch[:, -1] = GREY
        tile_r = 1 + top_clear
        tile_c = 1 + left_clear
        for _ in range(hole_count):
            patch[tile_r : tile_r + tile_h, tile_c : tile_c + tile_w][ring] = GREY
            anchor_r, anchor_c = _anchor_for_side(ring, side)
            gr = tile_r + anchor_r
            gc = tile_c + anchor_c
            if side == "left":
                patch[gr, 1:gc] = GREY
            else:
                patch[gr, gc + 1 : -1] = GREY
            tile_r += tile_h + gap

    return patch


def _frame_patch_is_valid(patch: np.ndarray, hole_key: tuple[tuple[int, int], bytes], hole_count: int) -> bool:
    comps = _components(patch == GREY)
    if len(comps) != 1:
        return False
    holes = _hole_components_for_frame(patch, comps[0])
    if len(holes) != hole_count:
        return False
    return all(_shape_key_from_coords(hole) == hole_key for hole in holes)


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int, margin: int) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(h, top + height + margin)
    c1 = min(w, left + width + margin)
    return not bool(occupied[r0:r1, c0:c1].any())


def _stamp(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    non_bg = patch != BG
    view = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    view[non_bg] = patch[non_bg]
    occupied[top : top + patch.shape[0], left : left + patch.shape[1]] |= non_bg


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, near_edge: bool) -> bool:
    ph, pw = patch.shape
    h, w = grid.shape
    for _ in range(PLACEMENT_RETRIES):
        if near_edge:
            side = random.choice(("top", "bottom", "left", "right"))
            if side == "top":
                top = random.randint(0, min(1, h - ph))
                left = random.randint(0, w - pw)
            elif side == "bottom":
                top = random.randint(max(0, h - ph - 1), h - ph)
                left = random.randint(0, w - pw)
            elif side == "left":
                top = random.randint(0, h - ph)
                left = random.randint(0, min(1, w - pw))
            else:
                top = random.randint(0, h - ph)
                left = random.randint(max(0, w - pw - 1), w - pw)
        else:
            top = random.randint(0, h - ph)
            left = random.randint(0, w - pw)

        if _can_place(occupied, top, left, ph, pw, margin=1):
            _stamp(grid, occupied, patch, top, left)
            return True
    return False


def _sprite_patch(shape: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(shape.shape, dtype=int)
    patch[shape] = int(color)
    return patch


def _candidate_grid_size(frame_count: int) -> tuple[int, int]:
    low = 22 if frame_count >= 2 else 18
    return random.randint(low, 30), random.randint(low, 30)


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice((1, 2, 3))
    distinct_shape_count = random.randint(1, frame_count)
    chosen_shapes = random.sample(SHAPE_LIBRARY, distinct_shape_count)
    chosen_colors = random.sample(SPRITE_COLORS, distinct_shape_count)

    shape_entries = []
    for hole, color in zip(chosen_shapes, chosen_colors):
        shape_entries.append(
            {
                "hole": hole,
                "color": int(color),
                "key": _shape_key_from_mask(hole),
            }
        )

    frames = []
    for _ in range(frame_count):
        entry = random.choice(shape_entries)
        hole_count = random.choice((1, 2))
        side = random.choice(("top", "bottom", "left", "right"))
        patch = _build_frame_patch(entry["hole"], hole_count, side)
        if not _frame_patch_is_valid(patch, entry["key"], hole_count):
            return None
        frames.append({"patch": patch, "color": entry["color"], "key": entry["key"]})

    h, w = _candidate_grid_size(frame_count)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    for frame in sorted(frames, key=lambda item: item["patch"].size, reverse=True):
        if not _place_patch(grid, occupied, frame["patch"], near_edge=False):
            return None

    for entry in shape_entries:
        if not _place_patch(grid, occupied, _sprite_patch(entry["hole"], entry["color"]), near_edge=True):
            return None

    try:
        grey_comps = _components(grid == GREY)
        if len(grey_comps) != frame_count:
            return None
        sprite_map = _extract_sprite_map(grid)
        if {entry["key"] for entry in shape_entries} - set(sprite_map):
            return None
        if any(sprite_map[entry["key"]] != entry["color"] for entry in shape_entries):
            return None
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None

    for comp in grey_comps:
        vals = set(int(v) for v in out[comp[:, 0], comp[:, 1]])
        if len(vals) != 1 or next(iter(vals)) in (BG, GREY):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_RETRIES} attempts")
