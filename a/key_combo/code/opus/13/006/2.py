from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
CYAN = 8

BODY_COLORS = (BLUE, RED, GREEN, YELLOW)
CONNECTOR_COLORS = (PINK, ORANGE)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def crop_true(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def sprite_from_component(grid: np.ndarray, coords: np.ndarray) -> dict:
    box = bbox_coords(coords)
    r0, c0, r1, c1 = box
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    connector_locs = np.argwhere((patch == PINK) | (patch == ORANGE))
    assert connector_locs.shape[0] == 1
    conn_r, conn_c = map(int, connector_locs[0])
    colors = {int(v) for v in patch.flatten() if int(v) not in (BG, PINK, ORANGE)}
    assert len(colors) == 1
    body_color = int(next(iter(colors)))
    return {
        "bbox": box,
        "patch": patch,
        "connector_color": int(patch[conn_r, conn_c]),
        "connector": (conn_r, conn_c),
        "body_color": body_color,
    }


def dock_mask(base: dict, plug: dict) -> np.ndarray:
    base_coords = np.argwhere(base["patch"] != BG)
    plug_coords = np.argwhere(plug["patch"] != BG)

    br, bc = base["connector"]
    pr, pc = plug["connector"]
    dr = int(br + 1 - pr)
    dc = int(bc - pc)

    pts = [(int(r), int(c)) for r, c in base_coords]
    pts.extend((int(r + dr), int(c + dc)) for r, c in plug_coords)
    pts_arr = np.array(pts, dtype=int)
    r0, c0, r1, c1 = bbox_coords(pts_arr)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        out[r - r0, c - c0] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock sprite pairs by matching connector colors, then display the cropped pixelwise NOR of the two emergent composite masks.

    Input structure:
    - Black background with exactly four small 4-connected multicolor sprites above a full-width grey horizontal line.
    - Each sprite uses one body color from blue/red/green/yellow and contains exactly one connector pixel in pink or orange.
    - The two sprites sharing one connector color form one pair, and the other two sharing the other connector color form the second pair.

    Rule:
    - Pair sprites by connector color, and within each pair choose as the base the sprite whose tight-bbox top-left is lexicographically smallest.
    - Translate the plug so its connector lands exactly one cell below the base connector, then treat the docked result as one composite object.
    - Extract the tight binary masks of the two docked composites, compute their pixelwise NOR, and tight-crop the non-empty cyan result.
    - Clear all original sprites, keep the grey line, and place the cropped NOR result horizontally centered with one blank row between its bottom edge and the grey line.

    Key insight:
    - The NOR inputs are not directly visible in the input; they only appear after connector-color pairing and docking determine the two composite masks.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    line_rows = np.where(np.all(g == GREY, axis=1))[0]
    assert line_rows.size >= 1
    line_r = int(line_rows[-1])

    sprite_mask = (g != BG) & (g != GREY)
    components = find_components(sprite_mask)
    assert len(components) == 4

    sprites = [sprite_from_component(g, comp) for comp in components]
    by_connector: dict[int, list[dict]] = defaultdict(list)
    for sprite in sprites:
        by_connector[sprite["connector_color"]].append(sprite)
    assert sorted(by_connector) == [PINK, ORANGE]
    assert all(len(group) == 2 for group in by_connector.values())

    pair_masks = []
    for connector_color in (PINK, ORANGE):
        pair = sorted(by_connector[connector_color], key=lambda s: (s["bbox"][0], s["bbox"][1]))
        pair_masks.append(dock_mask(pair[0], pair[1]))

    assert pair_masks[0].shape == pair_masks[1].shape
    nor_mask = ~(pair_masks[0] | pair_masks[1])

    out = np.zeros_like(g)
    out[line_r, :] = GREY
    if nor_mask.any():
        cropped = crop_true(nor_mask)
        ph, pw = cropped.shape
        top = line_r - ph - 1
        left = (w - pw) // 2
        assert top >= 0
        out[top : top + ph, left : left + pw][cropped] = CYAN
    return out


def manhattan_path(start: tuple[int, int], end: tuple[int, int], horizontal_first: bool) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    pts = [(r, c)]
    if horizontal_first:
        while c != tc:
            c += 1 if tc > c else -1
            pts.append((r, c))
        while r != tr:
            r += 1 if tr > r else -1
            pts.append((r, c))
    else:
        while r != tr:
            r += 1 if tr > r else -1
            pts.append((r, c))
        while c != tc:
            c += 1 if tc > c else -1
            pts.append((r, c))
    return pts


def grow_connected(mask: np.ndarray, allowed: np.ndarray, target_size: int) -> bool:
    while int(mask.sum()) < target_size:
        frontier = set()
        for r, c in np.argwhere(mask):
            for nr, nc in neighbors4(int(r), int(c)):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and allowed[nr, nc] and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return False
        nr, nc = random.choice(list(frontier))
        mask[nr, nc] = True
    return True


def build_pair(body_a: int, body_b: int, connector_color: int, box_h: int, box_w: int) -> dict | None:
    seam = random.randint(1, box_h - 3)
    conn_c = random.randint(1, box_w - 2)

    base_mask = np.zeros((box_h, box_w), dtype=bool)
    plug_mask = np.zeros((box_h, box_w), dtype=bool)
    base_allowed = np.zeros((box_h, box_w), dtype=bool)
    plug_allowed = np.zeros((box_h, box_w), dtype=bool)
    base_allowed[: seam + 1, :] = True
    plug_allowed[seam + 1 :, :] = True

    base_conn = (seam, conn_c)
    plug_conn = (seam + 1, conn_c)
    base_mask[base_conn] = True
    plug_mask[plug_conn] = True

    if random.choice([True, False]):
        base_side = 0
        plug_side = box_w - 1
    else:
        base_side = box_w - 1
        plug_side = 0

    base_targets = [
        (0, random.randint(0, box_w - 1)),
        (random.randint(0, seam), base_side),
    ]
    plug_targets = [
        (box_h - 1, random.randint(0, box_w - 1)),
        (random.randint(seam + 1, box_h - 1), plug_side),
    ]

    for target in base_targets:
        for r, c in manhattan_path(base_conn, target, horizontal_first=random.choice([True, False])):
            base_mask[r, c] = True
    for target in plug_targets:
        for r, c in manhattan_path(plug_conn, target, horizontal_first=random.choice([True, False])):
            plug_mask[r, c] = True

    base_area = int(base_mask.sum())
    plug_area = int(plug_mask.sum())
    if not (3 <= base_area <= 6 and 3 <= plug_area <= 6):
        return None

    if random.choice([True, False]):
        extra_target = random.randint(base_area, 6)
        if not grow_connected(base_mask, base_allowed, extra_target):
            return None
    if random.choice([True, False]):
        extra_target = random.randint(plug_area, 6)
        if not grow_connected(plug_mask, plug_allowed, extra_target):
            return None

    if not (3 <= int(base_mask.sum()) <= 6 and 3 <= int(plug_mask.sum()) <= 6):
        return None

    composite = base_mask | plug_mask
    if crop_true(composite).shape != (box_h, box_w):
        return None

    base_coords = np.argwhere(base_mask)
    plug_coords = np.argwhere(plug_mask)
    base_box = bbox_coords(base_coords)
    plug_box = bbox_coords(plug_coords)

    base_patch = np.full((base_box[2] - base_box[0] + 1, base_box[3] - base_box[1] + 1), BG, dtype=int)
    plug_patch = np.full((plug_box[2] - plug_box[0] + 1, plug_box[3] - plug_box[1] + 1), BG, dtype=int)
    for r, c in base_coords:
        base_patch[r - base_box[0], c - base_box[1]] = body_a
    for r, c in plug_coords:
        plug_patch[r - plug_box[0], c - plug_box[1]] = body_b
    base_patch[base_conn[0] - base_box[0], base_conn[1] - base_box[1]] = connector_color
    plug_patch[plug_conn[0] - plug_box[0], plug_conn[1] - plug_box[1]] = connector_color

    return {
        "mask": composite,
        "base_patch": base_patch,
        "plug_patch": plug_patch,
    }


def can_place(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    ph, pw = patch.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False
    fg = np.argwhere(patch != BG)
    for dr, dc in fg:
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - pad)
        r2 = min(h, rr + pad + 1)
        c1 = max(0, cc - pad)
        c2 = min(w, cc + pad + 1)
        if occupied[r1:r2, c1:c2].any():
            return False
    return True


def stamp(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = patch.shape
    region = grid[r0 : r0 + ph, c0 : c0 + pw]
    mask = patch != BG
    region[mask] = patch[mask]
    occupied[r0 : r0 + ph, c0 : c0 + pw][mask] = True


def place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, max_row: int, after: tuple[int, int] | None) -> tuple[int, int] | None:
    ph, pw = patch.shape
    candidates = [(r, c) for r in range(max_row - ph + 1) for c in range(grid.shape[1] - pw + 1)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        if after is not None and (r0, c0) <= after:
            continue
        if can_place(occupied, patch, r0, c0, pad=1):
            stamp(grid, occupied, patch, r0, c0)
            return r0, c0
    return None


def generate_input() -> np.ndarray:
    for _ in range(1000):
        grid_h = random.randint(14, 18)
        grid_w = random.randint(14, 18)
        line_r = grid_h - 1

        box_h = random.randint(4, 5)
        box_w = random.randint(4, 5)

        body_colors = list(BODY_COLORS)
        random.shuffle(body_colors)
        connector_colors = list(CONNECTOR_COLORS)
        random.shuffle(connector_colors)

        pair1 = build_pair(body_colors[0], body_colors[1], connector_colors[0], box_h, box_w)
        pair2 = build_pair(body_colors[2], body_colors[3], connector_colors[1], box_h, box_w)
        if pair1 is None or pair2 is None:
            continue

        mask1 = pair1["mask"]
        mask2 = pair2["mask"]
        if mask1.shape != mask2.shape:
            continue
        if not np.any(mask1 != mask2):
            continue
        nor_count = int((~(mask1 | mask2)).sum())
        if nor_count < 3:
            continue

        grid = np.zeros((grid_h, grid_w), dtype=int)
        grid[line_r, :] = GREY
        occupied = np.zeros((grid_h, grid_w), dtype=bool)
        occupied[line_r, :] = True
        max_row = line_r - 3

        pair_order = [pair1, pair2]
        random.shuffle(pair_order)
        ok = True
        for pair in pair_order:
            base_pos = place_patch(grid, occupied, pair["base_patch"], max_row, after=None)
            if base_pos is None:
                ok = False
                break
            plug_pos = place_patch(grid, occupied, pair["plug_patch"], max_row, after=base_pos)
            if plug_pos is None:
                ok = False
                break
        if not ok:
            continue

        out = main(grid)
        if out.shape != grid.shape:
            continue
        if np.count_nonzero(out == CYAN) < 3:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
