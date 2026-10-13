from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GRAY = 5
ORANGE = 7

LEFT_TILE_COLORS = {BLUE, RED}
RIGHT_TILE_COLORS = {GREEN, ORANGE}
COLOR_MAP = {BG: BG, BLUE: ORANGE, RED: GREEN}
SIDES = ("up", "right", "down", "left")
DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
CW_SIDE = {"up": "right", "right": "down", "down": "left", "left": "up"}
CCW_ARM = {v: k for k, v in CW_SIDE.items()}


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:, :] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


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


def find_components(grid: np.ndarray, *, colors: set[int] | None = None) -> list[dict]:
    if colors is None:
        mask = grid != BG
    else:
        mask = np.isin(grid, list(colors))
    out = []
    for coords in components_mask(mask):
        box = bbox_coords(coords)
        vals = sorted({int(grid[r, c]) for r, c in coords})
        out.append(
            {
                "coords": coords,
                "bbox": box,
                "area": int(len(coords)),
                "colors": set(vals),
            }
        )
    return out


def occ_mask(coords: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def flip_for_side(patch: np.ndarray, side: str) -> np.ndarray:
    if side in {"left", "right"}:
        return np.fliplr(patch)
    return np.flipud(patch)


def recolor_patch(patch: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = patch.copy()
    for src, dst in mapping.items():
        out[patch == int(src)] = int(dst)
    return out


def place_box(box: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if side == "right":
        return r0, c1 + 1
    if side == "left":
        return r0, c0 - w
    if side == "down":
        return r1 + 1, c0
    return r0 - h, c0


def rect_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def render_cells(cells: set[tuple[int, int]], colors: dict[tuple[int, int], int]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for rc in cells:
        patch[rc[0] - r0, rc[1] - c0] = int(colors[rc])
    return patch


def grow_connected(area: int, *, limit_h: int, limit_w: int) -> set[tuple[int, int]]:
    while True:
        start = (random.randrange(limit_h), random.randrange(limit_w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < limit_h and 0 <= nc < limit_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        nr = min(rows)
        nc = min(cols)
        norm = {(r - nr, c - nc) for r, c in cells}
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if 2 <= h <= 4 and 2 <= w <= 4:
            return norm


def random_colored_patch(area_range: tuple[int, int], *, limit_h: int, limit_w: int) -> np.ndarray:
    while True:
        area = random.randint(*area_range)
        cells = grow_connected(area, limit_h=limit_h, limit_w=limit_w)
        colors: dict[tuple[int, int], int] = {}
        ordered = list(cells)
        random.shuffle(ordered)
        split = random.randint(1, len(ordered) - 1)
        first = set(ordered[:split])
        for rc in ordered:
            colors[rc] = BLUE if rc in first else RED
        patch = render_cells(cells, colors)
        if BLUE in patch and RED in patch:
            return patch


def make_tile_pair() -> tuple[np.ndarray, np.ndarray]:
    while True:
        left = random_colored_patch((4, 6), limit_h=3, limit_w=3)
        if left.shape[0] > 3 or left.shape[1] > 3:
            continue
        right = recolor_patch(left, COLOR_MAP)
        return left, right


def detect_cross(grid: np.ndarray) -> tuple[tuple[int, int], dict[str, int], str]:
    gray_comps = components_mask(grid == GRAY)
    assert len(gray_comps) == 1
    coords = gray_comps[0]
    center_candidates = []
    coord_set = {(int(r), int(c)) for r, c in coords}
    for r, c in coord_set:
        if all((r + dr, c + dc) in coord_set for dr, dc in DIRS.values()):
            center_candidates.append((r, c))
    assert len(center_candidates) == 1
    center = center_candidates[0]
    cr, cc = center
    for r, c in coord_set:
        assert r == cr or c == cc

    lengths: dict[str, int] = {}
    for name, (dr, dc) in DIRS.items():
        rr, cc2 = cr + dr, cc + dc
        n = 0
        while (rr, cc2) in coord_set:
            n += 1
            rr += dr
            cc2 += dc
        lengths[name] = n
        assert n >= 1

    longest = max(lengths.values())
    longest_dirs = [k for k, v in lengths.items() if v == longest]
    assert len(longest_dirs) == 1
    arm = longest_dirs[0]
    others = [v for k, v in lengths.items() if k != arm]
    assert all(longest >= v + 2 for v in others)
    return center, lengths, arm


def parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    colored = find_components(np.where(g == GRAY, BG, g))
    right_candidates = [c for c in colored if c["colors"] <= RIGHT_TILE_COLORS]
    assert len(right_candidates) == 1
    right = right_candidates[0]

    left_like = [c for c in colored if c["colors"] <= LEFT_TILE_COLORS]
    assert len(left_like) == 2

    right_mask = occ_mask(right["coords"], right["bbox"])
    tile_left = None
    sprite = None
    rr0, rc0, _rr1, _rc1 = right["bbox"]
    for comp in left_like:
        box = comp["bbox"]
        mask = occ_mask(comp["coords"], box)
        if mask.shape == right_mask.shape and np.array_equal(mask, right_mask):
            r0, c0, r1, c1 = box
            if r0 == rr0 and rc0 - c1 == 2:
                tile_left = comp
            else:
                sprite = comp
        else:
            sprite = comp
    assert tile_left is not None and sprite is not None

    left_box = tile_left["bbox"]
    right_box = right["bbox"]
    left_patch = crop_bbox(g, left_box)
    right_patch = crop_bbox(g, right_box)
    mapping = {BG: BG}
    for r in range(left_patch.shape[0]):
        for c in range(left_patch.shape[1]):
            lv = int(left_patch[r, c])
            rv = int(right_patch[r, c])
            if lv == BG:
                assert rv == BG
                continue
            if lv in mapping:
                assert mapping[lv] == rv
            else:
                mapping[lv] = rv
    assert mapping == COLOR_MAP

    _center, lengths, arm = detect_cross(g)
    side = CW_SIDE[arm]
    return {
        "mapping": mapping,
        "left_tile": tile_left,
        "right_tile": right,
        "sprite": sprite,
        "side": side,
        "arm": arm,
        "arm_lengths": lengths,
    }


def compose_output(shape: tuple[int, int], sprite_box: tuple[int, int, int, int], sprite_patch: np.ndarray, side: str) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    r0, c0, _r1, _c1 = sprite_box
    paste(out, sprite_patch, r0, c0, transparent=BG)
    mirrored = recolor_patch(flip_for_side(sprite_patch, side), COLOR_MAP)
    mr, mc = place_box(sprite_box, side)
    h, w = mirrored.shape
    assert 0 <= mr and 0 <= mc and mr + h <= shape[0] and mc + w <= shape[1]
    paste(out, mirrored, mr, mc, transparent=BG)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Attach a recolored mirror copy of the sprite on the side encoded by the cross arm.

    Input structure:
    - Black background contains a top pair of same-shape 3x3 example tiles separated by one empty column.
    - The left example tile uses blue/red, the right example tile uses orange/green at the same occupied positions.
    - Lower in the grid there is one blue/red 4-connected sprite and one gray cross whose unique longest arm encodes a direction.

    Rule:
    - Learn the color mapping from aligned positions of the two example tiles, including black->black.
    - Rotate the cross's unique longest-arm direction 90 degrees clockwise to choose the attachment side.
    - Mirror the sprite bbox crop horizontally for left/right attachment, or vertically for up/down attachment.
    - Apply the learned color mapping only to that mirrored copy, place it flush against the chosen side of the original sprite bbox, and erase the example tiles plus the cross.

    Key insight:
    - The top pair is a color legend, not an output template: the final added object is a mirror of the lower sprite whose colors come from the learned tile-to-tile mapping.
    """
    g = np.asarray(grid, dtype=int)
    parsed = parse_scene(g)
    sprite_box = parsed["sprite"]["bbox"]
    sprite_patch = crop_bbox(g, sprite_box)
    return compose_output(g.shape, sprite_box, sprite_patch, parsed["side"])


def _side_variants(shape: tuple[int, int], sprite_box: tuple[int, int, int, int], sprite_patch: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for side in SIDES:
        try:
            out[side] = compose_output(shape, sprite_box, sprite_patch, side)
        except AssertionError:
            continue
    return out


def _valid_sprite_positions(h: int, w: int, side: str, tile_bottom: int, H: int, W: int) -> list[tuple[int, int]]:
    positions = []
    for r0 in range(tile_bottom + 3, H - h + 1):
        for c0 in range(1, W - w):
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            mr, mc = place_box(box, side)
            if mr < 0 or mc < 0 or mr + h > H or mc + w > W:
                continue
            mirror_box = (mr, mc, mr + h - 1, mc + w - 1)
            if mirror_box[0] <= tile_bottom:
                continue
            positions.append((r0, c0))
    return positions


def _cross_boxes(center: tuple[int, int], lengths: dict[str, int]) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    cr, cc = center
    cells = [(cr, cc)]
    for name, (dr, dc) in DIRS.items():
        for step in range(1, lengths[name] + 1):
            cells.append((cr + dr * step, cc + dc * step))
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return cells, (min(rows), min(cols), max(rows), max(cols))


def _sample_cross(
    H: int,
    W: int,
    forbidden: list[tuple[int, int, int, int]],
    sprite_box: tuple[int, int, int, int],
    side: str,
) -> tuple[list[tuple[int, int]], dict[str, int], str]:
    arm = CCW_ARM[side]
    for _ in range(400):
        base = random.randint(1, 2)
        lengths = {name: random.randint(1, base) for name in SIDES}
        lengths[arm] = random.randint(base + 2, base + 3)
        for cr in range(1, H - 1):
            for cc in range(1, W - 1):
                cells, box = _cross_boxes((cr, cc), lengths)
                if any(not (0 <= r < H and 0 <= c < W) for r, c in cells):
                    continue
                if any(rect_intersects(box, other, pad=1) for other in forbidden):
                    continue
                sr0, sc0, sr1, sc1 = sprite_box
                if abs(cr - (sr0 + sr1) // 2) + abs(cc - (sc0 + sc1) // 2) > 10:
                    continue
                return cells, lengths, arm
    raise RuntimeError("failed to place cross")


def _build_input() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(600):
        tile_left, tile_right = make_tile_pair()
        side = random.choice(SIDES)
        sprite_patch = random_colored_patch((5, 8), limit_h=4, limit_w=4)
        if np.array_equal(sprite_patch, flip_for_side(sprite_patch, side)):
            continue

        h, w = sprite_patch.shape
        H = random.randint(max(14, h * 2 + 8), 24)
        W = random.randint(max(16, w * 2 + 8), 24)
        tile_r = random.randint(1, 2)
        if W < 10:
            continue
        tile_c = random.randint(1, W - 8)
        tile_bottom = tile_r + tile_left.shape[0] - 1

        sprite_positions = _valid_sprite_positions(h, w, side, tile_bottom, H, W)
        if not sprite_positions:
            continue
        sprite_r, sprite_c = random.choice(sprite_positions)
        sprite_box = (sprite_r, sprite_c, sprite_r + h - 1, sprite_c + w - 1)
        mirror_r, mirror_c = place_box(sprite_box, side)
        mirror_box = (mirror_r, mirror_c, mirror_r + h - 1, mirror_c + w - 1)

        left_box = (tile_r, tile_c, tile_r + tile_left.shape[0] - 1, tile_c + tile_left.shape[1] - 1)
        right_c = left_box[3] + 2
        right_box = (tile_r, right_c, tile_r + tile_right.shape[0] - 1, right_c + tile_right.shape[1] - 1)

        forbidden = [left_box, right_box, sprite_box, mirror_box]
        try:
            cross_cells, arm_lengths, arm = _sample_cross(H, W, forbidden, sprite_box, side)
        except RuntimeError:
            continue

        inp = np.full((H, W), BG, dtype=int)
        paste(inp, tile_left, tile_r, tile_c, transparent=BG)
        paste(inp, tile_right, tile_r, right_c, transparent=BG)
        paste(inp, sprite_patch, sprite_r, sprite_c, transparent=BG)
        for r, c in cross_cells:
            inp[r, c] = GRAY

        expected = compose_output(inp.shape, sprite_box, sprite_patch, side)
        if not _validate_instance(inp, expected, sprite_box, sprite_patch, side, arm, arm_lengths):
            continue
        return inp, expected
    raise RuntimeError("failed to generate valid puzzle instance")


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    sprite_box: tuple[int, int, int, int],
    sprite_patch: np.ndarray,
    side: str,
    arm: str,
    arm_lengths: dict[str, int],
) -> bool:
    try:
        parsed = parse_scene(inp)
        actual = main(inp)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False
    if parsed["side"] != side or parsed["arm"] != arm or parsed["arm_lengths"] != arm_lengths:
        return False
    if parsed["mapping"] != COLOR_MAP:
        return False
    if BLUE not in sprite_patch or RED not in sprite_patch:
        return False

    variants = _side_variants(inp.shape, sprite_box, sprite_patch)
    if any(np.array_equal(expected, grid) for other, grid in variants.items() if other != side):
        return False

    no_map = np.full(inp.shape, BG, dtype=int)
    r0, c0, _r1, _c1 = sprite_box
    paste(no_map, sprite_patch, r0, c0, transparent=BG)
    mr, mc = place_box(sprite_box, side)
    paste(no_map, flip_for_side(sprite_patch, side), mr, mc, transparent=BG)
    if np.array_equal(expected, no_map):
        return False

    no_mirror = np.full(inp.shape, BG, dtype=int)
    paste(no_mirror, sprite_patch, r0, c0, transparent=BG)
    paste(no_mirror, recolor_patch(sprite_patch, COLOR_MAP), mr, mc, transparent=BG)
    if np.array_equal(expected, no_mirror):
        return False

    return True


def generate_input() -> np.ndarray:
    return _build_input()[0]
