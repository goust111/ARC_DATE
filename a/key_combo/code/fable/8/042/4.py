import random
from collections import Counter
from itertools import combinations

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_GEN_ATTEMPTS = 400
COMPONENT_PADDING = 1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            bbox = _bbox_from_coords(coords)
            comps.append({"color": color, "coords": coords, "bbox": bbox, "area": len(coords)})
    return comps


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _component_mask(grid: np.ndarray, comp: dict) -> np.ndarray:
    crop = _crop(grid, comp["bbox"])
    return crop == int(comp["color"])


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((factor, factor), dtype=int)).astype(bool)


def _match_scale_1_to_2(grid: np.ndarray, a: dict, b: dict) -> tuple[dict, dict, np.ndarray] | None:
    if int(a["color"]) != int(b["color"]):
        return None
    mask_a = _component_mask(grid, a)
    mask_b = _component_mask(grid, b)
    ha, wa = mask_a.shape
    hb, wb = mask_b.shape
    if hb == 2 * ha and wb == 2 * wa and np.array_equal(_scale_mask(mask_a, 2), mask_b):
        return a, b, mask_a
    if ha == 2 * hb and wa == 2 * wb and np.array_equal(_scale_mask(mask_b, 2), mask_a):
        return b, a, mask_b
    return None


def _find_target_pair(grid: np.ndarray) -> tuple[dict, dict, np.ndarray]:
    bg = _get_bg(grid)
    comps = _find_components(grid, bg)
    matches: list[tuple[dict, dict, np.ndarray]] = []
    for a, b in combinations(comps, 2):
        match = _match_scale_1_to_2(grid, a, b)
        if match is not None:
            matches.append(match)
    assert len(matches) == 1, "expected exactly one same-color cross-scale 1x/2x pair"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique 1x/2x same-shape sprite pair and draw its 3x continuation at the next arithmetic-progression position.

    Input structure:
    - The grid has a dominant background and several separated monochrome sprites.
    - Exactly one color has a pair of sprites where one bbox crop is the 2x block-scaled version of the other.
    - Other sprites may repeat, but only at the same scale.

    Rule:
    - Detect the unique same-color pair whose larger sprite equals the 2x pixel expansion of the smaller sprite.
    - Build the 3x expansion of that smaller sprite in the same color.
    - Place the new sprite so its bbox top-left is `2 * big_top_left - small_top_left`, while leaving all original pixels unchanged.

    Key insight:
    - Repetition alone is a decoy here; the real signal is the unique cross-scale pattern match, and the output needs both the next size and the next position in the same arithmetic progression.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    small, big, base_mask = _find_target_pair(g)

    sr0, sc0, _sr1, _sc1 = small["bbox"]
    br0, bc0, _br1, _bc1 = big["bbox"]
    target_r = 2 * br0 - sr0
    target_c = 2 * bc0 - sc0

    scaled3 = _scale_mask(base_mask, 3)
    th, tw = scaled3.shape
    assert 0 <= target_r and 0 <= target_c
    assert target_r + th <= g.shape[0] and target_c + tw <= g.shape[1]

    out = g.copy()
    region = out[target_r : target_r + th, target_c : target_c + tw]
    assert not np.any((region != bg) & scaled3), "target 3x placement must be empty in valid inputs"
    region[scaled3] = int(small["color"])
    return out


def _random_connected_mask() -> np.ndarray:
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        max_area = min(h * w - 1, 8)
        min_area = max(3, (h * w + 2) // 3)
        if min_area > max_area:
            min_area = max_area
        target_area = random.randint(min_area, max_area)

        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = set()
        for nb in _neighbors4(*start):
            if 0 <= nb[0] < h and 0 <= nb[1] < w:
                frontier.add(nb)

        while len(cells) < target_area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nb in _neighbors4(*cell):
                if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in cells:
                    frontier.add(nb)

        if len(cells) != target_area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows[0] != 0 or rows[-1] != h - 1 or cols[0] != 0 or cols[-1] != w - 1:
            continue
        if mask.all():
            continue
        if len(rows) < 2 or len(cols) < 2:
            continue
        return mask
    raise RuntimeError("failed to build a connected mask")


def _sprite_from_mask(mask: np.ndarray, color: int, scale: int) -> np.ndarray:
    scaled = _scale_mask(mask, scale)
    sprite = np.full(scaled.shape, BG, dtype=int)
    sprite[scaled] = int(color)
    return sprite


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _reserve_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int = COMPONENT_PADDING) -> bool:
    hh, ww = occupied.shape
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if r0 < 0 or c0 < 0 or r1 >= hh or c1 >= ww:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(hh, r1 + padding + 1)
    cc1 = min(ww, c1 + padding + 1)
    if occupied[rr0:rr1, cc0:cc1].any():
        return False
    occupied[rr0:rr1, cc0:cc1] = True
    return True


def _paste_sprite(grid: np.ndarray, sprite: np.ndarray, r0: int, c0: int) -> None:
    mask = sprite != BG
    h, w = sprite.shape
    region = grid[r0 : r0 + h, c0 : c0 + w]
    assert not np.any((region != BG) & mask)
    region[mask] = sprite[mask]


def _component_signatures(grid: np.ndarray) -> list[tuple[int, tuple[tuple[int, ...], ...], tuple[int, int]]]:
    bg = _get_bg(grid)
    out = []
    for comp in _find_components(grid, bg):
        mask = _component_mask(grid, comp)
        out.append((int(comp["color"]), _mask_signature(mask), mask.shape))
    return out


def _choose_delta(h: int, w: int) -> tuple[int, int]:
    mode = random.choice(["horizontal", "vertical", "diagonal", "skew"])
    max_step = 9
    if mode == "horizontal":
        dc = random.randint(2 * w + 1, max_step)
        return 0, random.choice((-dc, dc))
    if mode == "vertical":
        dr = random.randint(2 * h + 1, max_step)
        return random.choice((-dr, dr)), 0
    if mode == "diagonal":
        dr = random.randint(2 * h + 1, max_step)
        dc = random.randint(2 * w + 1, max_step)
        return random.choice((-dr, dr)), random.choice((-dc, dc))
    dr = random.randint(2 * h + 1, max_step)
    dc = random.randint(w + 2, max_step)
    return random.choice((-dr, dr)), random.choice((-dc, dc))


def _relative_union_bbox(base_mask: np.ndarray, delta: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = base_mask.shape
    dr, dc = delta
    rects = [
        (0, 0, h - 1, w - 1),
        (dr, dc, dr + 2 * h - 1, dc + 2 * w - 1),
        (2 * dr, 2 * dc, 2 * dr + 3 * h - 1, 2 * dc + 3 * w - 1),
    ]
    return (
        min(r0 for r0, _, _, _ in rects),
        min(c0 for _, c0, _, _ in rects),
        max(r1 for _, _, r1, _ in rects),
        max(c1 for _, _, _, c1 in rects),
    )


def _pick_canvas_and_positions(base_mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    h, w = base_mask.shape
    for _ in range(200):
        delta = _choose_delta(h, w)
        ur0, uc0, ur1, uc1 = _relative_union_bbox(base_mask, delta)
        union_h = ur1 - ur0 + 1
        union_w = uc1 - uc0 + 1
        min_h = max(20, union_h + 5)
        min_w = max(20, union_w + 5)
        if min_h > 30 or min_w > 30:
            continue
        H = random.randint(min_h, 30)
        W = random.randint(min_w, 30)
        margin = 1
        shift_r = random.randint(margin - ur0, H - margin - 1 - ur1)
        shift_c = random.randint(margin - uc0, W - margin - 1 - uc1)
        small = (shift_r, shift_c)
        big = (shift_r + delta[0], shift_c + delta[1])
        third = (shift_r + 2 * delta[0], shift_c + 2 * delta[1])
        return (H, W), small, big, third
    raise RuntimeError("failed to place target progression")


def _random_location(occupied: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    H, W = occupied.shape
    spots = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(spots)
    for r0, c0 in spots:
        if _reserve_bbox(occupied, r0, c0, h, w):
            return r0, c0
    return None


def _scaled_signatures(mask: np.ndarray) -> set[tuple[tuple[int, ...], ...]]:
    return {_mask_signature(_scale_mask(mask, factor)) for factor in (1, 2, 3)}


def _build_attempt() -> tuple[np.ndarray, np.ndarray, dict] | None:
    target_mask = _random_connected_mask()
    target_color = random.choice(PALETTE)
    (H, W), small_pos, big_pos, third_pos = _pick_canvas_and_positions(target_mask)

    inp = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    small_sprite = _sprite_from_mask(target_mask, target_color, 1)
    big_sprite = _sprite_from_mask(target_mask, target_color, 2)
    third_sprite = _sprite_from_mask(target_mask, target_color, 3)

    if not _reserve_bbox(occupied, small_pos[0], small_pos[1], *small_sprite.shape):
        return None
    if not _reserve_bbox(occupied, big_pos[0], big_pos[1], *big_sprite.shape):
        return None
    if not _reserve_bbox(occupied, third_pos[0], third_pos[1], *third_sprite.shape):
        return None

    _paste_sprite(inp, small_sprite, *small_pos)
    _paste_sprite(inp, big_sprite, *big_pos)

    target_sigs = _scaled_signatures(target_mask)

    used_colors = {target_color}
    placed_groups = []

    same_shape_color = random.choice([c for c in PALETTE if c != target_color])
    same_shape_scale = random.choice((1, 2))
    same_shape_sprite = _sprite_from_mask(target_mask, same_shape_color, same_shape_scale)
    placed_groups.append(("same_shape_other_color", same_shape_sprite, 2))
    used_colors.add(same_shape_color)

    for _ in range(80):
        decoy_mask = _random_connected_mask()
        if _mask_signature(decoy_mask) == _mask_signature(target_mask):
            continue
        if _scaled_signatures(decoy_mask) & target_sigs:
            continue
        same_color_scale = random.choice((1, 2, 3))
        same_color_sprite = _sprite_from_mask(decoy_mask, target_color, same_color_scale)
        placed_groups.append(("same_color_other_shape", same_color_sprite, 2))
        break
    else:
        return None

    extra_groups = random.randint(1, 3)
    for _ in range(extra_groups):
        for _ in range(80):
            mask = _random_connected_mask()
            sigs = _scaled_signatures(mask)
            if sigs & target_sigs:
                continue
            color = random.choice([c for c in PALETTE if c not in used_colors] or list(PALETTE))
            if color == target_color and _mask_signature(mask) == _mask_signature(target_mask):
                continue
            scale = random.choice((1, 2, 3))
            sprite = _sprite_from_mask(mask, color, scale)
            if sprite.shape[0] > H - 2 or sprite.shape[1] > W - 2:
                continue
            placed_groups.append(("duplicate_decoy", sprite, random.choice((2, 2, 3))))
            used_colors.add(color)
            break
        else:
            return None

    random.shuffle(placed_groups)
    for _label, sprite, copies in placed_groups:
        h, w = sprite.shape
        if h > H - 2 or w > W - 2:
            return None
        positions = []
        for _ in range(copies):
            pos = _random_location(occupied, h, w)
            if pos is None:
                return None
            positions.append(pos)
        for pos in positions:
            _paste_sprite(inp, sprite, *pos)

    out = inp.copy()
    _paste_sprite(out, third_sprite, *third_pos)

    meta = {
        "target_color": target_color,
        "target_mask_sig": _mask_signature(target_mask),
        "small_pos": small_pos,
        "big_pos": big_pos,
        "third_pos": third_pos,
        "duplicate_group_count": len(placed_groups),
    }
    return inp, out, meta


def _validate_attempt(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(inp, expected):
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    bg = _get_bg(inp)
    comps = _find_components(inp, bg)
    if len(comps) < 6:
        return False

    try:
        small, big, base_mask = _find_target_pair(inp)
    except Exception:
        return False

    if int(small["color"]) != int(meta["target_color"]):
        return False
    if _mask_signature(base_mask) != meta["target_mask_sig"]:
        return False
    if (small["bbox"][0], small["bbox"][1]) != meta["small_pos"]:
        return False
    if (big["bbox"][0], big["bbox"][1]) != meta["big_pos"]:
        return False

    third_mask = _scale_mask(base_mask, 3)
    tr, tc = meta["third_pos"]
    region = inp[tr : tr + third_mask.shape[0], tc : tc + third_mask.shape[1]]
    if region.shape != third_mask.shape:
        return False
    if np.any((region != bg) & third_mask):
        return False

    comp_sigs = _component_signatures(inp)
    counts = Counter(comp_sigs)
    duplicate_groups = sum(1 for n in counts.values() if n >= 2)
    if duplicate_groups < 2:
        return False

    target_sig = meta["target_mask_sig"]
    target_color = meta["target_color"]
    same_shape_other_color = any(color != target_color and sig == target_sig for color, sig, _shape in comp_sigs)
    same_color_other_shape = any(color == target_color and sig != target_sig for color, sig, _shape in comp_sigs)
    if not same_shape_other_color or not same_color_other_shape:
        return False

    added = solved != inp
    if not np.array_equal(solved[~added], inp[~added]):
        return False
    if int(np.sum(added)) != int(np.sum(third_mask)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        built = _build_attempt()
        if built is None:
            continue
        inp, expected, meta = built
        if _validate_attempt(inp, expected, meta):
            return inp
    raise RuntimeError("failed to generate a valid scale-series extrapolation input")
