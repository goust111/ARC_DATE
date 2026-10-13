import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
REGION_COLORS = (1, 2, 3, 4, 6, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
HOLE_SHAPES = (
    {(0, 0)},
    {(0, 0), (0, 1)},
    {(0, 0), (1, 0)},
    {(0, 0), (0, 1), (1, 0)},
    {(0, 0), (0, 1), (1, 1)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (2, 0)},
)
GEN_ATTEMPTS = 400


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _color_components(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    comp_id = np.full((h, w), -1, dtype=int)
    comps = []
    next_id = 0
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BLACK, ORANGE) or comp_id[r, c] != -1:
                continue
            q = deque([(r, c)])
            comp_id[r, c] = next_id
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and comp_id[nr, nc] == -1
                        and int(grid[nr, nc]) == color
                    ):
                        comp_id[nr, nc] = next_id
                        q.append((nr, nc))
            comps.append(
                {
                    "id": next_id,
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )
            next_id += 1
    return comps, comp_id


def _associate_markers(grid: np.ndarray, comps: list[dict], comp_id: np.ndarray) -> list[dict]:
    h, w = grid.shape
    by_id = {comp["id"]: comp for comp in comps}
    for comp in comps:
        comp["marker"] = None
    for r, c in np.argwhere(grid == ORANGE):
        region_ids = set()
        for dr, dc in DIRS4:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and comp_id[nr, nc] != -1:
                region_ids.add(int(comp_id[nr, nc]))
        assert len(region_ids) == 1, "each orange marker must touch exactly one region"
        rid = next(iter(region_ids))
        assert by_id[rid]["marker"] is None, "each region must have exactly one marker"
        by_id[rid]["marker"] = (int(r), int(c))
    for comp in comps:
        assert comp["marker"] is not None, "missing region marker"
    return comps


def _region_center(region: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = region["bbox"]
    return (r0 + r1) // 2, (c0 + c1) // 2


def _direction_from_marker(region: dict) -> str:
    r0, c0, r1, c1 = region["bbox"]
    mr, mc = region["marker"]
    cr, cc = _region_center(region)
    dr = mr - cr
    dc = mc - cc
    if abs(dr) > abs(dc):
        return "up" if dr < 0 else "down"
    if abs(dc) > abs(dr):
        return "left" if dc < 0 else "right"
    if mr == r0:
        return "up"
    if mr == r1:
        return "down"
    if mc == c0:
        return "left"
    return "right"


def _hole_cells(grid: np.ndarray, region: dict) -> list[tuple[int, int]]:
    coords = set(region["coords"])
    coords.add(region["marker"])
    r0, c0, r1, c1 = _bbox(coords)
    solid = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        solid[r - r0, c - c0] = True
    black = grid[r0 : r1 + 1, c0 : c1 + 1] == BLACK
    holes = []
    for comp in _mask_components(black):
        if any(rr in (0, black.shape[0] - 1) or cc in (0, black.shape[1] - 1) for rr, cc in comp):
            continue
        if any(not black[rr, cc] for rr, cc in comp):
            continue
        boundary_ok = True
        for rr, cc in comp:
            for dr, dc in DIRS4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < solid.shape[0] and 0 <= nc < solid.shape[1]:
                    if not black[nr, nc] and not solid[nr, nc]:
                        boundary_ok = False
                        break
            if not boundary_ok:
                break
        if boundary_ok:
            holes.extend((r0 + rr, c0 + cc) for rr, cc in comp)
    return holes


def _stamp_cells(hole_cells: list[tuple[int, int]], center: tuple[int, int], marker: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    if not hole_cells:
        return []
    cr, cc = center
    mr, mc = marker
    if direction in ("up", "down"):
        flipped = [(2 * cr - r, c) for r, c in hole_cells]
        dc = mc - cc
        shifted = [(r, c + dc) for r, c in flipped]
        rows = [r for r, _ in shifted]
        dr = (mr - 1) - max(rows) if direction == "up" else (mr + 1) - min(rows)
        final = [(r + dr, c) for r, c in shifted]
    else:
        flipped = [(r, 2 * cc - c) for r, c in hole_cells]
        dr = mr - cr
        shifted = [(r + dr, c) for r, c in flipped]
        cols = [c for _, c in shifted]
        dc = (mc - 1) - max(cols) if direction == "left" else (mc + 1) - min(cols)
        final = [(r, c + dc) for r, c in shifted]
    return sorted(set(final))


def _ray_color(grid: np.ndarray, marker: tuple[int, int], direction: str, own_color: int) -> int:
    h, w = grid.shape
    dr, dc = DIR_VECTORS[direction]
    r, c = marker[0] + dr, marker[1] + dc
    while 0 <= r < h and 0 <= c < w:
        value = int(grid[r, c])
        if value not in (BLACK, ORANGE, own_color):
            return value
        r += dr
        c += dc
    return GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each region's internal black holes outside the region on the marker side.

    The orange boundary pixel indicates which side to use. The hole set is mirrored
    around the region center along that axis, shifted so the stamp sits one cell
    outside the marked side, then recolored by the first region visible on that ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()
    out[out == ORANGE] = BLACK

    comps, comp_id = _color_components(g)
    comps = _associate_markers(g, comps, comp_id)

    for region in comps:
        holes = _hole_cells(g, region)
        if not holes:
            continue
        direction = _direction_from_marker(region)
        center = _region_center(region)
        stamp = _stamp_cells(holes, center, region["marker"], direction)
        color = _ray_color(g, region["marker"], direction, int(region["color"]))
        for r, c in stamp:
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                out[r, c] = color
    return out


def _shape_dims(shape: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _choose_slots() -> list[tuple[int, int]]:
    slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    count = random.randint(2, 4)
    return random.sample(slots, count)


def _pick_direction(slot: tuple[int, int], occupied: set[tuple[int, int]]) -> tuple[str, tuple[int, int] | None]:
    r, c = slot
    targets = {
        "left": (r, c - 1) if c - 1 >= 0 and (r, c - 1) in occupied else None,
        "right": (r, c + 1) if c + 1 <= 1 and (r, c + 1) in occupied else None,
        "up": (r - 1, c) if r - 1 >= 0 and (r - 1, c) in occupied else None,
        "down": (r + 1, c) if r + 1 <= 1 and (r + 1, c) in occupied else None,
    }
    hit_dirs = [d for d, t in targets.items() if t is not None]
    miss_dirs = [d for d, t in targets.items() if t is None]
    if hit_dirs and miss_dirs:
        pool = hit_dirs if random.random() < 0.5 else miss_dirs
    else:
        pool = hit_dirs or miss_dirs
    direction = random.choice(pool)
    return direction, targets[direction]


def _generate_holes(h: int, w: int) -> list[tuple[int, int]]:
    max_box_h = min(4, h - 2)
    max_box_w = min(4, w - 2)
    box_h = random.randint(2, max_box_h)
    box_w = random.randint(2, max_box_w)
    top = random.randint(1, h - box_h - 1)
    left = random.randint(1, w - box_w - 1)

    occ: set[tuple[int, int]] = set()
    comp_count = 1 if box_h * box_w <= 4 else random.choice([1, 2])
    for _ in range(comp_count):
        candidates = []
        for shape in HOLE_SHAPES:
            sh, sw = _shape_dims(shape)
            if sh > box_h or sw > box_w:
                continue
            for sr in range(box_h - sh + 1):
                for sc in range(box_w - sw + 1):
                    placed = {(top + sr + rr, left + sc + cc) for rr, cc in shape}
                    if placed & occ:
                        continue
                    separated = True
                    for r, c in placed:
                        for dr, dc in DIRS4:
                            if (r + dr, c + dc) in occ:
                                separated = False
                                break
                        if not separated:
                            break
                    if separated:
                        candidates.append(placed)
        if not candidates:
            break
        occ |= random.choice(candidates)
    if not occ:
        occ.add((top, left))
    return sorted(occ)


def _build_region(spec: dict) -> np.ndarray:
    patch = np.full((spec["h"], spec["w"]), spec["color"], dtype=int)
    for r, c in spec["holes_local"]:
        patch[r, c] = BLACK
    mr, mc = spec["marker_local"]
    patch[mr, mc] = ORANGE
    return patch


def _global_cells(top: int, left: int, coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in coords]


def _make_spec(slot: tuple[int, int], center: tuple[int, int], color: int) -> dict:
    h = random.choice([7, 9])
    w = random.choice([7, 9])
    top = center[0] - h // 2
    left = center[1] - w // 2
    holes_local = _generate_holes(h, w)
    return {
        "slot": slot,
        "center": center,
        "color": color,
        "h": h,
        "w": w,
        "top": top,
        "left": left,
        "holes_local": holes_local,
    }


def _pick_marker(spec: dict, direction: str, target: dict | None) -> tuple[int, int]:
    top, left, h, w = spec["top"], spec["left"], spec["h"], spec["w"]
    bottom = top + h - 1
    right = left + w - 1
    if direction in ("left", "right"):
        rows = list(range(top + 1, bottom))
        if target is not None:
            t_rows = range(target["top"], target["top"] + target["h"])
            rows = [r for r in rows if r in t_rows]
        row = random.choice(rows)
        col = left if direction == "left" else right
        return row, col
    cols = list(range(left + 1, right))
    if target is not None:
        t_cols = range(target["left"], target["left"] + target["w"])
        cols = [c for c in cols if c in t_cols]
    col = random.choice(cols)
    row = top if direction == "up" else bottom
    return row, col


def _expected_output(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    out[out == ORANGE] = BLACK
    for spec in specs:
        for r, c in spec["stamp_cells"]:
            out[r, c] = spec["stamp_color"]
    return out


def _region_patch_is_valid(spec: dict) -> bool:
    patch = _build_region(spec)
    comps, comp_id = _color_components(patch)
    if len(comps) != 1:
        return False
    try:
        _associate_markers(patch, comps, comp_id)
    except AssertionError:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    margin_r = random.randint(9, 11)
    margin_c = random.randint(9, 11)
    pitch_r = random.randint(14, 16)
    pitch_c = random.randint(14, 16)
    row_centers = [margin_r, margin_r + pitch_r]
    col_centers = [margin_c, margin_c + pitch_c]
    height = row_centers[-1] + margin_r + 1
    width = col_centers[-1] + margin_c + 1

    slots = _choose_slots()
    occupied = set(slots)
    colors = random.sample(REGION_COLORS, len(slots))
    specs = []
    by_slot = {}
    for slot, color in zip(slots, colors):
        center = (row_centers[slot[0]], col_centers[slot[1]])
        spec = _make_spec(slot, center, color)
        specs.append(spec)
        by_slot[slot] = spec

    for spec in specs:
        direction, target_slot = _pick_direction(spec["slot"], occupied)
        target = by_slot.get(target_slot) if target_slot is not None else None
        marker_global = _pick_marker(spec, direction, target)
        spec["direction"] = direction
        spec["target_slot"] = target_slot
        spec["marker_global"] = marker_global
        spec["marker_local"] = (marker_global[0] - spec["top"], marker_global[1] - spec["left"])
        spec["holes_global"] = _global_cells(spec["top"], spec["left"], spec["holes_local"])
        spec["region_cells"] = [
            (r, c)
            for r in range(spec["top"], spec["top"] + spec["h"])
            for c in range(spec["left"], spec["left"] + spec["w"])
            if (r, c) != marker_global and (r, c) not in spec["holes_global"]
        ]
        spec["stamp_cells"] = _stamp_cells(spec["holes_global"], spec["center"], marker_global, direction)
        spec["stamp_color"] = by_slot[target_slot]["color"] if target_slot is not None else GREY
        if not _region_patch_is_valid(spec):
            return None

    grid = np.full((height, width), BLACK, dtype=int)
    occupied_input = set()
    for spec in specs:
        patch = _build_region(spec)
        top, left = spec["top"], spec["left"]
        bottom = top + spec["h"] - 1
        right = left + spec["w"] - 1
        if top < 0 or left < 0 or bottom >= height or right >= width:
            return None
        for r in range(spec["h"]):
            for c in range(spec["w"]):
                value = int(patch[r, c])
                if value == BLACK:
                    continue
                gr, gc = top + r, left + c
                if (gr, gc) in occupied_input:
                    return None
                occupied_input.add((gr, gc))
                grid[gr, gc] = value

    stamp_union = set()
    for spec in specs:
        if any(not (0 <= r < height and 0 <= c < width) for r, c in spec["stamp_cells"]):
            return None
        for cell in spec["stamp_cells"]:
            if cell in occupied_input or cell in stamp_union:
                return None
            stamp_union.add(cell)

    for spec in specs:
        seen = _ray_color(grid, spec["marker_global"], spec["direction"], int(spec["color"]))
        if seen != spec["stamp_color"]:
            return None

    orange_count = int((grid == ORANGE).sum())
    if orange_count != len(specs):
        return None

    expected = _expected_output(grid, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
