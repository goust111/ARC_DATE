from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
TRAIL = 9
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
STRUCTURE_COLORS = (1, 3, 6, 7, 8)
GEN_RETRIES = 500
GRID_RANGE = (20, 28)
STRUCTURE_COUNT_CHOICES = (2, 3)
SLIDE_RANGE = (6, 8)
PLACE_PADDING = 1
EXTRA_SPRITE_PROB = 0.6

# Canonical east-facing logical masks.
# They all span both logical rows/cols and include the west-edge center cell,
# which is the pixel a horizontal ray sees first before the component is docked.
LOGICAL_MASKS = (
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
)
DIR_TO_ROT = {
    (0, 1): 0,
    (-1, 0): 1,
    (0, -1): 2,
    (1, 0): 3,
}


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int = BACKGROUND) -> None:
    h, w = patch.shape
    for rr in range(h):
        for cc in range(w):
            val = int(patch[rr, cc])
            if val != int(transparent):
                canvas[r0 + rr, c0 + cc] = val


def random_free_location(canvas: np.ndarray, patch: np.ndarray, *, transparent: int = BACKGROUND, padding: int = 0) -> tuple[int, int] | None:
    h, w = canvas.shape
    ph, pw = patch.shape
    mask = patch != transparent
    if not mask.any():
        return 0, 0
    occ = canvas != transparent
    if padding > 0:
        expanded = occ.copy()
        base = occ.copy()
        for dr in range(-padding, padding + 1):
            for dc in range(-padding, padding + 1):
                if dr == 0 and dc == 0:
                    continue
                src_r0 = max(0, -dr)
                src_r1 = min(h, h - dr)
                src_c0 = max(0, -dc)
                src_c1 = min(w, w - dc)
                dst_r0 = max(0, dr)
                dst_r1 = min(h, h + dr)
                dst_c0 = max(0, dc)
                dst_c1 = min(w, w + dc)
                expanded[dst_r0:dst_r1, dst_c0:dst_c1] |= base[src_r0:src_r1, src_c0:src_c1]
        occ = expanded
    candidates = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            if not (occ[r0 : r0 + ph, c0 : c0 + pw] & mask).any():
                candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def structure_patch(color: int) -> np.ndarray:
    patch = np.full((6, 6), int(color), dtype=int)
    patch[1:5, 1:5] = BACKGROUND
    patch[2:4, 5] = BACKGROUND
    return patch


def sparse_patch(logical_mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full((3, 3), BACKGROUND, dtype=int)
    for lr in range(2):
        for lc in range(2):
            if logical_mask[lr, lc]:
                patch[2 * lr, 2 * lc] = int(color)
    return patch


def dense_patch(logical_mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full((4, 4), BACKGROUND, dtype=int)
    for lr in range(2):
        for lc in range(2):
            if logical_mask[lr, lc]:
                patch[2 * lr : 2 * lr + 2, 2 * lc : 2 * lc + 2] = int(color)
    return patch


def canonical_patches(color: int, logical_mask: np.ndarray, slide: int) -> tuple[np.ndarray, np.ndarray]:
    struct = structure_patch(color)
    sparse = sparse_patch(logical_mask, color)
    dense = dense_patch(logical_mask, color)
    width = slide + 5
    inp = np.full((6, width), BACKGROUND, dtype=int)
    out = np.full((6, width), BACKGROUND, dtype=int)
    paste(inp, struct, 0, 0)
    paste(out, struct, 0, 0)
    paste(inp, sparse, 1, 1 + slide)
    for step in range(slide, 0, -1):
        paste(out, np.where(dense != BACKGROUND, TRAIL, BACKGROUND), 1, 1 + step)
    paste(out, dense, 1, 1)
    return inp, out


def directional_patches(color: int, logical_mask: np.ndarray, slide: int, direction: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rot_k = DIR_TO_ROT[direction]
    struct = np.rot90(structure_patch(color), rot_k).copy()
    sparse = np.rot90(sparse_patch(logical_mask, color), rot_k).copy()
    dense = np.rot90(dense_patch(logical_mask, color), rot_k).copy()
    cavity = largest_bg_rectangle(struct, BACKGROUND)
    assert cavity == (1, 1, 4, 4)
    cr0, cc0, _cr1, _cc1 = cavity
    dr, dc = direction
    sparse_r0 = cr0 + slide * dr
    sparse_c0 = cc0 + slide * dc
    offset_r = max(0, -sparse_r0)
    offset_c = max(0, -sparse_c0)
    final_r0 = offset_r + cr0
    final_c0 = offset_c + cc0
    sparse_r0 += offset_r
    sparse_c0 += offset_c
    height = max(offset_r + struct.shape[0], sparse_r0 + 4, final_r0 + 4)
    width = max(offset_c + struct.shape[1], sparse_c0 + 4, final_c0 + 4)
    inp = np.full((height, width), BACKGROUND, dtype=int)
    out = np.full((height, width), BACKGROUND, dtype=int)
    paste(inp, struct, offset_r, offset_c)
    paste(out, struct, offset_r, offset_c)
    paste(inp, sparse, sparse_r0, sparse_c0)
    for step in range(slide, 0, -1):
        paste(out, np.where(dense != BACKGROUND, TRAIL, BACKGROUND), final_r0 + step * dr, final_c0 + step * dc)
    paste(out, dense, final_r0, final_c0)
    return inp, out


def sparse_components(mask: np.ndarray) -> list[np.ndarray]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    remaining = set(coords)
    comps = []
    while remaining:
        start = remaining.pop()
        q = deque([start])
        pts = [start]
        while q:
            r, c = q.popleft()
            near = [
                (rr, cc)
                for rr, cc in list(remaining)
                if abs(rr - r) <= 2
                and abs(cc - c) <= 2
                and max(abs(rr - r), abs(cc - c)) == 2
            ]
            for nxt in near:
                remaining.remove(nxt)
                q.append(nxt)
                pts.append(nxt)
        comps.append(np.array(sorted(pts), dtype=int))
    return comps


def largest_bg_rectangle(subgrid: np.ndarray, bg: int) -> tuple[int, int, int, int] | None:
    h, w = subgrid.shape
    best = None
    best_key = None
    bg_mask = subgrid == int(bg)
    for r0 in range(h):
        for c0 in range(w):
            if not bg_mask[r0, c0]:
                continue
            for r1 in range(r0, h):
                for c1 in range(c0, w):
                    if not bg_mask[r0 : r1 + 1, c0 : c1 + 1].all():
                        continue
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    key = (area, r1 - r0 + 1, c1 - c0 + 1)
                    if best is None or key > best_key:
                        best = (r0, c0, r1, c1)
                        best_key = key
    return best


def detect_structures(grid: np.ndarray) -> tuple[int, list[dict], list[dict], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    structures = []
    coord_to_comp: dict[tuple[int, int], int] = {}
    structure_mask = np.zeros_like(g, dtype=bool)
    next_id = 0
    for color in sorted(int(c) for c in np.unique(g) if int(c) != int(bg)):
        for coords in components_mask(g == int(color)):
            comp = {
                "id": next_id,
                "color": int(color),
                "coords": coords,
                "bbox": bbox_coords(coords),
            }
            next_id += 1
            r0, c0, r1, c1 = comp["bbox"]
            if r1 - r0 + 1 < 6 or c1 - c0 + 1 < 6:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            cavity = largest_bg_rectangle(sub, bg)
            if cavity is None:
                continue
            cr0, cc0, cr1, cc1 = cavity
            if (cr1 - cr0 + 1, cc1 - cc0 + 1) != (4, 4):
                continue
            structure = {
                **comp,
                "cavity_bbox": (r0 + cr0, c0 + cc0, r0 + cr1, c0 + cc1),
            }
            structures.append(structure)
            for r, c in coords:
                structure_mask[int(r), int(c)] = True
                coord_to_comp[(int(r), int(c))] = int(structure["id"])
    sprites = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != int(bg)):
        sparse_mask = (g == int(color)) & ~structure_mask
        for coords in sparse_components(sparse_mask):
            sprite = {
                "id": next_id,
                "color": int(color),
                "coords": coords,
                "bbox": bbox_coords(coords),
            }
            sprites.append(sprite)
            for r, c in coords:
                coord_to_comp[(int(r), int(c))] = next_id
            next_id += 1
    return bg, structures, sprites, coord_to_comp


def cavity_centers(cavity_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = cavity_bbox
    rows = sorted({(r0 + r1) // 2, (r0 + r1 + 1) // 2})
    cols = sorted({(c0 + c1) // 2, (c0 + c1 + 1) // 2})
    return [(int(r), int(c)) for r in rows for c in cols]


def ray_hit_component(
    grid: np.ndarray,
    structure: dict,
    start: tuple[int, int],
    direction: tuple[int, int],
    coord_to_comp: dict[tuple[int, int], int],
    bg: int,
) -> int | None:
    h, w = grid.shape
    sr0, sc0, sr1, sc1 = structure["bbox"]
    rr, cc = start
    dr, dc = direction
    rr += dr
    cc += dc
    left_bbox = False
    while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(bg):
        if not (sr0 <= rr <= sr1 and sc0 <= cc <= sc1):
            left_bbox = True
        rr += dr
        cc += dc
    if not (0 <= rr < h and 0 <= cc < w):
        return None
    if not left_bbox:
        return None
    if int(grid[rr, cc]) != int(structure["color"]):
        return None
    comp_id = coord_to_comp.get((int(rr), int(cc)))
    if comp_id is None or comp_id == int(structure["id"]):
        return None
    return int(comp_id)


def densify_component(coords: np.ndarray) -> np.ndarray:
    pts = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        pts.add((rr, cc))
        pts.add((rr + 1, cc))
        pts.add((rr, cc + 1))
        pts.add((rr + 1, cc + 1))
    return np.array(sorted(pts), dtype=int)


def open_ray_cells(grid: np.ndarray, structure: dict, direction: tuple[int, int], bg: int) -> set[tuple[int, int]]:
    h, w = grid.shape
    sr0, sc0, sr1, sc1 = structure["bbox"]
    dr, dc = direction
    cells: set[tuple[int, int]] = set()
    for start in cavity_centers(structure["cavity_bbox"]):
        rr, cc = start[0] + dr, start[1] + dc
        left_bbox = False
        path = []
        while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(bg):
            if not (sr0 <= rr <= sr1 and sc0 <= cc <= sc1):
                left_bbox = True
                path.append((int(rr), int(cc)))
            rr += dr
            cc += dc
        if left_bbox:
            cells.update(path)
    return cells


def analyze_scene(grid: np.ndarray) -> tuple[list[dict], dict[int, dict], dict[int, tuple[int, int]]]:
    bg, structures, sprites, coord_to_comp = detect_structures(grid)
    sprite_by_id = {sp["id"]: sp for sp in sprites}
    assignments: dict[int, dict] = {}
    directions: dict[int, tuple[int, int]] = {}
    used_sprites: set[int] = set()
    for structure in structures:
        matches = []
        for center in cavity_centers(structure["cavity_bbox"]):
            for direction in DIRS:
                comp_id = ray_hit_component(grid, structure, center, direction, coord_to_comp, bg)
                if comp_id is None:
                    continue
                pair = (direction, comp_id)
                if pair not in matches:
                    matches.append(pair)
        assert len(matches) == 1, "each structure must see exactly one same-color external sprite"
        direction, comp_id = matches[0]
        assert comp_id not in used_sprites, "matched sprites must be one-to-one"
        used_sprites.add(comp_id)
        assignments[int(structure["id"])] = sprite_by_id[int(comp_id)]
        directions[int(structure["id"])] = direction
        dense_bbox = bbox_coords(densify_component(sprite_by_id[int(comp_id)]["coords"]))
        cavity_bbox = structure["cavity_bbox"]
        assert dense_bbox[2] - dense_bbox[0] == cavity_bbox[2] - cavity_bbox[0]
        assert dense_bbox[3] - dense_bbox[1] == cavity_bbox[3] - cavity_bbox[1]
    return structures, assignments, directions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Ray-capture each color-matched sparse sprite, densify it, and slide it into the matching cavity.
    
    Input structure:
    - The background is black and the foreground contains 2-3 large monochrome structures plus same-color sparse sprites outside them.
    - Each large structure has one rectangular internal cavity, while each sparse sprite sits on a two-step lattice with gaps between its pixels.
    - Exactly one ray that starts from a cavity center can leave each structure and hit a same-color external sprite.
    
    Rule:
    - For every large structure, find the 4x4 cavity inside its bbox and cast rays from the cavity center in the four cardinal directions.
    - The unique ray that exits the structure through background and first hits a same-color component identifies both the matching sparse sprite and its docking direction.
    - Densify that sparse sprite by replacing every sparse pixel with a 2x2 solid block of the same color, then slide the dense sprite straight toward the structure until its bbox matches the cavity bbox.
    - Paint every intermediate occupied cell of the moving dense sprite maroon, exclude the final placement from the trail, and leave all unmatched sparse sprites unchanged.
    
    Key insight:
    - Color alone is not enough: the correct component is whichever same-color sprite is actually caught by the cavity ray, and that same ray also determines the direction of the later docking move.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    structures, assignments, directions = analyze_scene(g)
    for structure in structures:
        sprite = assignments[int(structure["id"])]
        direction = directions[int(structure["id"])]
        dense_coords = densify_component(sprite["coords"])
        dr, dc = direction
        cavity_r0, cavity_c0, cavity_r1, cavity_c1 = structure["cavity_bbox"]
        dense_r0, dense_c0, dense_r1, dense_c1 = bbox_coords(dense_coords)
        if dr == 0:
            shift_r = 0
            shift_c = cavity_c0 - dense_c0
            assert dense_r0 == cavity_r0 and dense_r1 == cavity_r1
        else:
            shift_r = cavity_r0 - dense_r0
            shift_c = 0
            assert dense_c0 == cavity_c0 and dense_c1 == cavity_c1
        steps = abs(shift_r + shift_c)
        step_r = 0 if shift_r == 0 else (1 if shift_r > 0 else -1)
        step_c = 0 if shift_c == 0 else (1 if shift_c > 0 else -1)
        for step in range(steps):
            off_r = step * step_r
            off_c = step * step_c
            for r, c in dense_coords:
                rr = int(r + off_r)
                cc = int(c + off_c)
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    out[rr, cc] = TRAIL
        for r, c in dense_coords:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = int(sprite["color"])
    return out


def place_extra_sprite(inp: np.ndarray, out: np.ndarray, colors: list[int]) -> bool:
    try:
        structures, _assignments, directions = analyze_scene(inp)
    except AssertionError:
        return False
    structure_by_color = {int(s["color"]): s for s in structures}
    color = int(random.choice(colors))
    logical = random.choice(LOGICAL_MASKS)
    patch = sparse_patch(logical, color)
    blocked = inp.copy()
    for struct in structures:
        for r, c in open_ray_cells(inp, struct, directions[int(struct["id"])], BACKGROUND):
            blocked[int(r), int(c)] = max(1, int(struct["color"]))
    struct = structure_by_color[color]
    for r, c in open_ray_cells(inp, struct, directions[int(struct["id"])], BACKGROUND):
        blocked[int(r), int(c)] = max(1, color)
    for _ in range(80):
        loc = random_free_location(blocked, patch, padding=2)
        if loc is None:
            return False
        r0, c0 = loc
        trial_inp = inp.copy()
        trial_out = out.copy()
        paste(trial_inp, patch, r0, c0)
        paste(trial_out, patch, r0, c0)
        try:
            structures, assignments, _ = analyze_scene(trial_inp)
        except AssertionError:
            continue
        if len(structures) != len(colors) or len(assignments) != len(colors):
            continue
        inp[:, :] = trial_inp
        out[:, :] = trial_out
        return True
    return False


def semantic_checks(inp: np.ndarray, out: np.ndarray, expected_structures: int, expect_extra: bool) -> bool:
    try:
        structures, assignments, directions = analyze_scene(inp)
        solved = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(solved, out):
        return False
    if len(structures) != expected_structures or len(assignments) != expected_structures:
        return False
    matched_sprite_ids = {sprite["id"] for sprite in assignments.values()}
    bg, _, sprites, _ = detect_structures(inp)
    unmatched = [sp for sp in sprites if sp["id"] not in matched_sprite_ids]
    if expect_extra and len(unmatched) != 1:
        return False
    if not expect_extra and unmatched:
        return False
    for structure in structures:
        direction = directions[int(structure["id"])]
        sprite = assignments[int(structure["id"])]
        cavity_bbox = structure["cavity_bbox"]
        dense_bbox = bbox_coords(densify_component(sprite["coords"]))
        if (dense_bbox[2] - dense_bbox[0], dense_bbox[3] - dense_bbox[1]) != (
            cavity_bbox[2] - cavity_bbox[0],
            cavity_bbox[3] - cavity_bbox[1],
        ):
            return False
        if direction[0] == 0 and dense_bbox[0] != cavity_bbox[0]:
            return False
        if direction[1] == 0 and dense_bbox[1] != cavity_bbox[1]:
            return False
    for sprite in unmatched:
        for r, c in sprite["coords"]:
            if int(out[r, c]) != int(sprite["color"]):
                return False
    return True


def gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    n_structures = random.choice(STRUCTURE_COUNT_CHOICES)
    expect_extra = n_structures == 2 and random.random() < EXTRA_SPRITE_PROB
    colors = random.sample(STRUCTURE_COLORS, n_structures)
    specs = []
    for color in colors:
        logical = random.choice(LOGICAL_MASKS)
        slide = random.randint(*SLIDE_RANGE)
        direction = random.choice(DIRS)
        inp_patch, out_patch = directional_patches(color, logical, slide, direction)
        specs.append(
            {
                "input": inp_patch,
                "output": out_patch,
            }
        )
    specs.sort(key=lambda item: np.count_nonzero(item["output"]), reverse=True)
    inp = np.full((h, w), BACKGROUND, dtype=int)
    out = np.full((h, w), BACKGROUND, dtype=int)
    occ = np.zeros((h, w), dtype=int)
    for spec in specs:
        block = np.ones(spec["output"].shape, dtype=int)
        loc = random_free_location(occ, block, transparent=0, padding=PLACE_PADDING)
        if loc is None:
            return None
        paste(inp, spec["input"], loc[0], loc[1])
        paste(out, spec["output"], loc[0], loc[1])
        paste(occ, block, loc[0], loc[1], transparent=0)
    if expect_extra and not place_extra_sprite(inp, out, colors):
        return None
    if not semantic_checks(inp, out, n_structures, expect_extra):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
