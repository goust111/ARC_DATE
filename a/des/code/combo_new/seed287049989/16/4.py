import random
from collections import Counter

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
FRAME_COLOR = 5
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            mask, box = _tight_mask(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": box,
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _mask_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    tight = _crop_mask(mask)
    return int(tight.shape[0]), int(tight.shape[1]), tight.astype(np.uint8).tobytes()


def _canonical_rot_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    tight = _crop_mask(mask)
    return min(_mask_key(np.rot90(tight, k)) for k in range(4))


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    grid[r0 : r0 + h, c0 : c0 + w][mask] = int(color)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _normalize_coords(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _rotate_coords(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    return tuple((w - 1 - c, r) for r, c in coords)


def _canonical_rot_coords(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    cur = _normalize_coords(coords)
    variants = []
    for _ in range(4):
        cur = _normalize_coords(cur)
        variants.append(cur)
        cur = _rotate_coords(cur)
    return min(variants)


def _hole_safe(mask: np.ndarray) -> bool:
    inv = ~mask
    for coords in _components(inv):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == inv.shape[0] - 1):
            continue
        if np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == inv.shape[1] - 1):
            continue
        return False
    return True


def _polyomino_masks(max_area: int = 8, max_side: int = 4) -> list[np.ndarray]:
    polys: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for area in range(1, max_area):
        nxt: set[tuple[tuple[int, int], ...]] = set()
        for poly in polys[area]:
            cells = set(poly)
            for r, c in list(cells):
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in cells:
                        continue
                    nxt.add(_canonical_rot_coords(tuple(cells | {(nr, nc)})))
        polys[area + 1] = nxt

    masks: list[np.ndarray] = []
    for area in range(3, max_area + 1):
        for poly in sorted(polys[area]):
            h = max(r for r, _ in poly) + 1
            w = max(c for _, c in poly) + 1
            if max(h, w) > max_side:
                continue
            mask = np.zeros((h, w), dtype=bool)
            for r, c in poly:
                mask[r, c] = True
            if not _hole_safe(mask):
                continue
            masks.append(mask)
    return masks


SHAPE_LIBRARY = _polyomino_masks()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hole in the grey mold with the color of its matching rotated puzzle piece.

    Input structure:
    - Background is the most common color, with a large monochrome mold component in the lower middle of the grid.
    - The mold contains several enclosed background holes; each hole is a distinct 4-connected shape.
    - Separate monochrome colored pieces are scattered away from the mold, and each piece matches one hole up to rotation.

    Rule:
    - Identify the large mold component and extract every enclosed background hole inside its bounding box.
    - For each colored piece, compare its shape to the hole shapes under 0/90/180/270 degree rotation.
    - Fill each matched hole with that piece's color, erase all loose pieces to background, and keep the mold unchanged.

    Key insight:
    - The black cavities inside the large monochrome region are the true targets; the loose colored objects are only shape carriers and must be matched by rotation, not by position.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    comps = _color_components(g, bg)
    assert comps, "expected a mold and some colored pieces"

    frame = max(comps, key=lambda comp: comp["area"])
    fr0, fc0, fr1, fc1 = frame["bbox"]

    sub = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    holes: list[dict] = []
    for coords in _components(sub == bg):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == sub.shape[0] - 1):
            continue
        if np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == sub.shape[1] - 1):
            continue
        mask, _ = _tight_mask(coords)
        holes.append(
            {
                "coords": coords,
                "key": _canonical_rot_key(mask),
            }
        )

    pieces = [comp for comp in comps if comp is not frame]
    assert len(holes) == len(pieces), "piece count must equal hole count"

    piece_by_key: dict[tuple[int, int, bytes], dict] = {}
    for piece in pieces:
        key = _canonical_rot_key(piece["mask"])
        assert key not in piece_by_key, "piece shapes must be unique up to rotation"
        piece_by_key[key] = piece

    out = g.copy()
    for piece in pieces:
        out[piece["coords"][:, 0], piece["coords"][:, 1]] = bg

    for hole in holes:
        piece = piece_by_key[hole["key"]]
        out[fr0 + hole["coords"][:, 0], fc0 + hole["coords"][:, 1]] = int(piece["color"])

    return out


def _random_frame_dims(height: int, width: int) -> tuple[int, int] | None:
    mold_h_max = min(12, height - 6)
    mold_w_max = min(16, width - 4)
    if mold_h_max < 8 or mold_w_max < 10:
        return None
    mold_h = random.randint(8, mold_h_max)
    mold_w = random.randint(10, mold_w_max)
    return mold_h, mold_w


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    grid[r0 : r0 + h, c0 : c0 + w] = FRAME_COLOR

    sides = []
    if c0 >= 2:
        sides.append("left")
    if c0 + w <= grid.shape[1] - 2:
        sides.append("right")
    if r0 + h <= grid.shape[0] - 2:
        sides.append("bottom")
    random.shuffle(sides)
    for side in sides[: random.randint(0, min(2, len(sides)))]:
        if side == "left":
            tab_h = random.randint(2, max(2, h - 2))
            tab_w = random.randint(1, 2)
            tab_r = random.randint(r0 + 1, r0 + h - tab_h - 1)
            grid[tab_r : tab_r + tab_h, c0 - tab_w : c0] = FRAME_COLOR
        elif side == "right":
            tab_h = random.randint(2, max(2, h - 2))
            tab_w = random.randint(1, 2)
            tab_r = random.randint(r0 + 1, r0 + h - tab_h - 1)
            grid[tab_r : tab_r + tab_h, c0 + w : c0 + w + tab_w] = FRAME_COLOR
        else:
            tab_h = random.randint(1, 2)
            tab_w = random.randint(2, max(2, w - 2))
            tab_c = random.randint(c0 + 1, c0 + w - tab_w - 1)
            grid[r0 + h : r0 + h + tab_h, tab_c : tab_c + tab_w] = FRAME_COLOR


def _pick_slot_count() -> int:
    return random.randint(2, 5)


def _choose_slot_shapes(count: int) -> list[np.ndarray]:
    shuffled = SHAPE_LIBRARY[:]
    random.shuffle(shuffled)
    chosen: list[np.ndarray] = []
    used: set[tuple[int, int, bytes]] = set()
    for mask in shuffled:
        key = _canonical_rot_key(mask)
        if key in used:
            continue
        chosen.append(mask)
        used.add(key)
        if len(chosen) == count:
            break
    return chosen


def _place_slots(inner_h: int, inner_w: int, masks: list[np.ndarray]) -> list[dict] | None:
    occupied = np.zeros((inner_h, inner_w), dtype=bool)
    placements: list[dict | None] = [None] * len(masks)
    order = sorted(
        range(len(masks)),
        key=lambda i: (masks[i].shape[0] * masks[i].shape[1], int(masks[i].sum())),
        reverse=True,
    )

    for idx in order:
        base = masks[idx]
        slot_mask = np.rot90(base, random.randint(0, 3))
        h, w = slot_mask.shape
        if h > inner_h or w > inner_w:
            return None
        blocked = _dilate8(occupied)
        candidates = []
        for r0 in range(inner_h - h + 1):
            for c0 in range(inner_w - w + 1):
                if not np.any(blocked[r0 : r0 + h, c0 : c0 + w] & slot_mask):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        occupied[r0 : r0 + h, c0 : c0 + w] |= slot_mask
        placements[idx] = {
            "mask": slot_mask,
            "r0": r0,
            "c0": c0,
            "key": _canonical_rot_key(slot_mask),
        }

    return [placement for placement in placements if placement is not None]


def _place_pieces(
    grid: np.ndarray,
    pieces: list[dict],
    band_h: int,
) -> bool:
    occupied = np.zeros(grid.shape, dtype=bool)
    occupied[grid != 0] = True
    order = sorted(
        range(len(pieces)),
        key=lambda i: pieces[i]["mask"].shape[0] * pieces[i]["mask"].shape[1],
        reverse=True,
    )

    for idx in order:
        piece = pieces[idx]
        mask = piece["mask"]
        ph, pw = mask.shape
        if ph > band_h:
            return False
        blocked = _dilate8(occupied)
        candidates = []
        for r0 in range(0, band_h - ph + 1):
            for c0 in range(0, grid.shape[1] - pw + 1):
                if not np.any(blocked[r0 : r0 + ph, c0 : c0 + pw] & mask):
                    candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        _paint_mask(grid, mask, r0, c0, piece["color"])
        occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
    return True


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(18, 28)
    width = random.randint(18, 29)
    dims = _random_frame_dims(height, width)
    if dims is None:
        return None
    mold_h, mold_w = dims

    min_r0 = max(6, height // 2 - 1)
    max_r0 = height - mold_h - 2
    if min_r0 > max_r0:
        return None
    mold_r0 = random.randint(min_r0, max_r0)
    center_c0 = (width - mold_w) // 2
    mold_c0 = max(1, min(width - mold_w - 1, center_c0 + random.randint(-2, 2)))

    slot_count = _pick_slot_count()
    slot_shapes = _choose_slot_shapes(slot_count)
    if len(slot_shapes) != slot_count:
        return None

    inner_h = mold_h - 2
    inner_w = mold_w - 2
    slot_layout = _place_slots(inner_h, inner_w, slot_shapes)
    if slot_layout is None or len(slot_layout) != slot_count:
        return None

    input_grid = np.zeros((height, width), dtype=int)
    _draw_frame(input_grid, mold_r0, mold_c0, mold_h, mold_w)

    expected = input_grid.copy()
    colors = random.sample(PIECE_COLORS, slot_count)
    pieces: list[dict] = []

    relative_rots = [random.randint(0, 3) for _ in range(slot_count)]
    if slot_count >= 3 and len(set(relative_rots)) == 1:
        relative_rots[0] = (relative_rots[0] + 1) % 4

    for idx, slot in enumerate(slot_layout):
        slot_mask = slot["mask"]
        abs_r0 = mold_r0 + 1 + int(slot["r0"])
        abs_c0 = mold_c0 + 1 + int(slot["c0"])
        input_grid[abs_r0 : abs_r0 + slot_mask.shape[0], abs_c0 : abs_c0 + slot_mask.shape[1]][slot_mask] = 0
        expected[abs_r0 : abs_r0 + slot_mask.shape[0], abs_c0 : abs_c0 + slot_mask.shape[1]][slot_mask] = colors[idx]

        piece_mask = np.rot90(slot_mask, relative_rots[idx])
        pieces.append({"mask": piece_mask, "color": colors[idx], "key": slot["key"]})

    if len(_components(input_grid == FRAME_COLOR)) != 1:
        return None

    band_h = mold_r0 - 1
    if band_h < max(piece["mask"].shape[0] for piece in pieces):
        return None
    if not _place_pieces(input_grid, pieces, band_h):
        return None

    if np.array_equal(input_grid, expected):
        return None
    if not np.array_equal(main(input_grid), expected):
        return None
    return input_grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _generate_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid mold-matching sample")
