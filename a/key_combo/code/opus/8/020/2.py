import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, crop_bbox, find_components, get_bg, in_bounds, paste


BG = Color.BLACK
BODY_A = Color.RED
BODY_B = Color.BLUE
HIDE = Color.GREEN
ANCHOR_A = Color.ORANGE
ANCHOR_B = Color.YELLOW

GRID_HEIGHTS = tuple(range(18, 25))
GRID_WIDTHS = tuple(range(18, 25))
CENTRAL_SIZES = (9, 11)
MAX_GEN_ATTEMPTS = 400


def _body_color(value: int) -> int:
    value = int(value)
    if value == ANCHOR_A:
        return BODY_A
    if value == ANCHOR_B:
        return BODY_B
    return value


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _connected_component(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    sr, sc = map(int, start)
    assert in_bounds(mask.shape, sr, sc) and bool(mask[sr, sc])
    stack = [(sr, sc)]
    seen = {(sr, sc)}
    coords = []
    while stack:
        r, c = stack.pop()
        coords.append((r, c))
        for rr, cc in _neighbors4(r, c):
            if in_bounds((h, w), rr, cc) and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return np.array(coords, dtype=int)


def _orbit_positions(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    return sorted(
        {
            (int(r), int(c)),
            (h - 1 - int(r), int(c)),
            (int(r), w - 1 - int(c)),
            (h - 1 - int(r), w - 1 - int(c)),
        }
    )


def _sprite_patch_from_coords(region: np.ndarray, coords: np.ndarray, anchor: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for rr, cc in coords:
        patch[int(rr) - r0, int(cc) - c0] = int(region[int(rr), int(cc)])
    ar, ac = anchor
    return patch, (int(ar) - r0, int(ac) - c0)


def _extract_sprite(region: np.ndarray, anchor_color: int, body_color: int) -> tuple[np.ndarray, tuple[int, int]]:
    anchors = np.argwhere(region == int(anchor_color))
    assert anchors.shape == (1, 2)
    anchor = tuple(map(int, anchors[0]))
    mask = np.isin(region, [int(body_color), int(anchor_color)])
    coords = _connected_component(mask, anchor)
    return _sprite_patch_from_coords(region, coords, anchor)


def _central_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    components = find_components(grid, bg=BG, connectivity=8, monochromatic=False, return_format="dict")
    central_parts = []
    for comp in components:
        coords = comp["coords"]
        colors = {int(grid[r, c]) for r, c in coords}
        if int(comp["area"]) > 1 or HIDE in colors:
            central_parts.append(coords)
    assert central_parts, "expected a non-trivial central region"
    return bbox_coords(np.concatenate(central_parts, axis=0))


def _repair_region(region: np.ndarray) -> np.ndarray:
    repaired = region.copy()
    h, w = repaired.shape
    visited: set[tuple[int, int]] = set()
    for rr, cc in np.argwhere(region == HIDE):
        rr = int(rr)
        cc = int(cc)
        if (rr, cc) in visited:
            continue
        orbit = _orbit_positions(h, w, rr, cc)
        visited.update(orbit)
        visible = {
            _body_color(region[r, c])
            for r, c in orbit
            if int(region[r, c]) not in (BG, HIDE)
        }
        assert len(visible) == 1, "each hidden orbit must reveal exactly one body color"
        fill = visible.pop()
        for r, c in orbit:
            if int(repaired[r, c]) == HIDE:
                repaired[r, c] = fill
    assert not np.any(repaired == HIDE)
    return repaired


def _stamp(out: np.ndarray, patch: np.ndarray, anchor: tuple[int, int], target: tuple[int, int]) -> None:
    tr, tc = target
    ar, ac = anchor
    paste(out, patch, int(tr) - int(ar), int(tc) - int(ac), transparent=BG, clip=True)


def _solve(grid: np.ndarray, *, repair: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == BG

    cr0, cc0, cr1, cc1 = _central_bbox(g)
    region = crop_bbox(g, (cr0, cc0, cr1, cc1))
    if repair:
        region = _repair_region(region)

    patch_a, anchor_a = _extract_sprite(region, ANCHOR_A, BODY_A)
    patch_b, anchor_b = _extract_sprite(region, ANCHOR_B, BODY_B)

    out = g.copy()
    out[cr0 : cr1 + 1, cc0 : cc1 + 1] = BG

    targets = []
    for color in (ANCHOR_A, ANCHOR_B):
        for rr, cc in np.argwhere(g == int(color)):
            rr = int(rr)
            cc = int(cc)
            if cr0 <= rr <= cr1 and cc0 <= cc <= cc1:
                continue
            targets.append((cc, rr, int(color)))

    for _col, row, color in sorted(targets):
        if color == ANCHOR_A:
            _stamp(out, patch_a, anchor_a, (row, _col))
        else:
            _stamp(out, patch_b, anchor_b, (row, _col))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repair the hidden bilateral template in the center, then stamp each recovered sprite to matching anchor markers.

    Input structure:
    - The background is black.
    - A central odd-sized square contains two incomplete sprites whose body masks are symmetric under both horizontal and vertical reflection.
    - Red body pixels with one orange anchor form sprite A, blue body pixels with one yellow anchor form sprite B, and some body pixels are replaced by green placeholders.
    - Outside the central square there are isolated orange and yellow single-pixel target anchors.

    Rule:
    - Use the horizontal/vertical reflection orbit of each green cell inside the central square to recover its missing body color, treating orange as red and yellow as blue for symmetry inference.
    - After repair, extract the 4-connected red/orange component rooted at the orange anchor as sprite A and the blue/yellow component rooted at the yellow anchor as sprite B.
    - Crop each recovered sprite to its tight bounding box, erase the entire central source square to black, then stamp sprite A onto every outside orange anchor and sprite B onto every outside yellow anchor.
    - Stamps are aligned by their anchor pixel, clipped at the grid boundary, and applied in anchor order from left to right, breaking ties top to bottom.

    Key insight:
    - The visible central fragments are not the true templates: the missing body pixels must first be reconstructed from dual-axis symmetry, and only the repaired anchor-rooted connected component reveals the stamp to copy.
    """
    return _solve(grid, repair=True)


def _quadrant(center: int, r: int, c: int) -> tuple[int, int]:
    return (0 if r < center else 1, 0 if c < center else 1)


def _frame_mask(size: int) -> np.ndarray:
    margin = random.choice([0, 1])
    mask = np.zeros((size, size), dtype=bool)
    r0 = c0 = margin
    r1 = c1 = size - 1 - margin
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    if random.random() < 0.7 and r0 + 1 < r1 and c0 + 1 < c1:
        inset = 1
        for rr in (r0 + inset, r1 - inset):
            for cc in (c0 + inset, c1 - inset):
                mask[rr, cc] = True
    return mask


def _cross_mask(size: int) -> np.ndarray:
    center = size // 2
    mask = np.zeros((size, size), dtype=bool)
    family = random.choice(["cross", "diamond"])
    if family == "diamond":
        radius = 2 if size == 9 else random.choice([2, 3])
        for r in range(size):
            for c in range(size):
                if abs(r - center) + abs(c - center) <= radius:
                    mask[r, c] = True
        if radius >= 3 and random.random() < 0.6:
            for dc in (-1, 1):
                mask[center, center + dc * radius] = False
                mask[center + dc * radius, center] = False
            mask[center, center] = True
        return mask

    stem_margin = random.choice([2, 3]) if size == 11 else 2
    mask[stem_margin : size - stem_margin, center] = True
    arm_half = random.choice([1, 2])
    mask[center, center - arm_half : center + arm_half + 1] = True
    shoulder_offset = random.choice([1, 2])
    shoulder_half = 1
    for sign in (-1, 1):
        row = center + sign * shoulder_offset
        mask[row, center - shoulder_half : center + shoulder_half + 1] = True
    if random.random() < 0.5:
        leg_half = 1
        for sign in (-1, 1):
            row = center + sign * (shoulder_offset + 1)
            if 0 <= row < size:
                mask[row, center - leg_half : center + leg_half + 1] = True
    return mask


def _mask_orbits(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen: set[tuple[int, int]] = set()
    out = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if (r, c) in seen:
            continue
        orbit = [pos for pos in _orbit_positions(h, w, r, c) if bool(mask[pos])]
        for pos in orbit:
            seen.add(pos)
        out.append(sorted(orbit))
    return out


def _pick_anchor(mask: np.ndarray, preferred: str) -> tuple[int, int]:
    size = mask.shape[0]
    center = size // 2
    coords = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    big_orbit = [coord for coord in coords if len(_orbit_positions(size, size, coord[0], coord[1])) == 4]
    pool = big_orbit or coords
    if preferred == "upper_left":
        ranked = [coord for coord in pool if coord[0] < center and coord[1] <= center]
    else:
        ranked = [coord for coord in pool if coord[0] > center and coord[1] >= center]
    return random.choice(ranked or pool)


def _hide_orbits(mask: np.ndarray, anchor: tuple[int, int]) -> tuple[set[tuple[int, int]], list[dict]]:
    size = mask.shape[0]
    center = size // 2
    all_orbits = [orbit for orbit in _mask_orbits(mask) if anchor not in orbit and len(orbit) > 1]
    four_orbits = [orbit for orbit in all_orbits if len(orbit) == 4]
    assert four_orbits, "generator expects at least one 4-cell reflection orbit"
    chosen = [random.choice(four_orbits)]
    remaining = [orbit for orbit in all_orbits if orbit not in chosen]
    if remaining:
        chosen.append(random.choice(remaining))
    if len(remaining) > 1 and random.random() < 0.5:
        extra = random.choice([orbit for orbit in remaining if orbit not in chosen])
        chosen.append(extra)

    hidden: set[tuple[int, int]] = set()
    meta = []
    for idx, orbit in enumerate(chosen):
        if len(orbit) == 4 and idx == 0:
            visible = random.choice(orbit)
            hidden_now = {pos for pos in orbit if pos != visible}
            hidden |= hidden_now
            meta.append({"orbit": orbit, "visible": [visible], "hidden": sorted(hidden_now)})
            continue
        keep = 1 if random.random() < 0.7 else max(1, len(orbit) - 2)
        visible = set(random.sample(orbit, k=keep))
        hidden_now = set(orbit) - visible
        hidden |= hidden_now
        meta.append({"orbit": orbit, "visible": sorted(visible), "hidden": sorted(hidden_now)})

    hidden_quadrants = {_quadrant(center, r, c) for r, c in hidden if r != center and c != center}
    assert len(hidden_quadrants) >= 2
    assert any(len(item["orbit"]) == 4 and len(item["visible"]) == 1 for item in meta)
    return hidden, meta


def _build_center(size: int) -> tuple[np.ndarray, dict]:
    for _ in range(80):
        a_mask = _cross_mask(size)
        b_mask = _frame_mask(size)
        if np.any(a_mask & b_mask):
            continue

        a_anchor = _pick_anchor(a_mask, "upper_left")
        b_anchor = _pick_anchor(b_mask, "lower_right")
        if a_anchor == b_anchor:
            continue

        try:
            a_hidden, a_meta = _hide_orbits(a_mask, a_anchor)
            b_hidden, b_meta = _hide_orbits(b_mask, b_anchor)
        except AssertionError:
            continue

        center = np.zeros((size, size), dtype=int)
        center[a_mask] = BODY_A
        center[b_mask] = BODY_B
        for r, c in a_hidden | b_hidden:
            center[r, c] = HIDE
        center[a_anchor] = ANCHOR_A
        center[b_anchor] = ANCHOR_B

        repaired = np.zeros((size, size), dtype=int)
        repaired[a_mask] = BODY_A
        repaired[b_mask] = BODY_B
        repaired[a_anchor] = ANCHOR_A
        repaired[b_anchor] = ANCHOR_B

        return center, {
            "repaired": repaired,
            "a_anchor": a_anchor,
            "b_anchor": b_anchor,
            "a_hidden_meta": a_meta,
            "b_hidden_meta": b_meta,
        }
    raise RuntimeError("failed to build a non-overlapping central template")


def _sprite_from_repaired(repaired: np.ndarray, anchor_color: int, body_color: int) -> tuple[np.ndarray, tuple[int, int], set[tuple[int, int]]]:
    patch, anchor = _extract_sprite(repaired, anchor_color, body_color)
    coords = {
        (int(r) - int(anchor[0]), int(c) - int(anchor[1]))
        for r, c in np.argwhere(patch != BG)
    }
    return patch, anchor, coords


def _candidate_targets(shape: tuple[int, int], central_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = central_bbox
    out = []
    for r in range(h):
        for c in range(w):
            if r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1:
                continue
            out.append((r, c))
    return out


def _clip_for_target(target: tuple[int, int], patch: np.ndarray, anchor: tuple[int, int], shape: tuple[int, int]) -> bool:
    tr, tc = target
    ar, ac = anchor
    top = tr - ar
    left = tc - ac
    return top < 0 or left < 0 or top + patch.shape[0] > shape[0] or left + patch.shape[1] > shape[1]


def _stamp_cells(target: tuple[int, int], patch: np.ndarray, anchor: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    tr, tc = target
    ar, ac = anchor
    coords = set()
    for rr, cc in np.argwhere(patch != BG):
        gr = int(tr + rr - ar)
        gc = int(tc + cc - ac)
        if in_bounds(shape, gr, gc):
            coords.add((gr, gc))
    return coords


def _place_targets(
    shape: tuple[int, int],
    central_bbox: tuple[int, int, int, int],
    patch_a: np.ndarray,
    anchor_a: tuple[int, int],
    patch_b: np.ndarray,
    anchor_b: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    candidates = _candidate_targets(shape, central_bbox)
    if not candidates:
        return None

    need_clip = random.random() < 0.45
    need_overlap = random.random() < 0.2
    count_a = random.choice([2, 3])
    count_b = random.choice([2, 3])

    occupied_markers: set[tuple[int, int]] = set()
    targets_a: list[tuple[int, int]] = []
    targets_b: list[tuple[int, int]] = []

    def can_use(pos: tuple[int, int]) -> bool:
        r, c = pos
        for rr, cc in occupied_markers:
            if max(abs(r - rr), abs(c - cc)) <= 1:
                return False
        return True

    def add_one(color: str, pos: tuple[int, int]) -> bool:
        if pos not in candidates or not can_use(pos):
            return False
        occupied_markers.add(pos)
        if color == "A":
            targets_a.append(pos)
        else:
            targets_b.append(pos)
        return True

    if need_overlap:
        color = random.choice(["A", "B"])
        patch = patch_a if color == "A" else patch_b
        base_candidates = candidates[:]
        random.shuffle(base_candidates)
        delta_options = [(0, 2), (2, 0), (1, 2), (2, 1), (0, 3), (3, 0)]
        placed = False
        for base in base_candidates:
            for dr, dc in delta_options:
                for sign_r, sign_c in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    other = (base[0] + sign_r * dr, base[1] + sign_c * dc)
                    if other not in candidates:
                        continue
                    if not can_use(base) or not can_use(other):
                        continue
                    anchor = anchor_a if color == "A" else anchor_b
                    first_cells = _stamp_cells(base, patch, anchor, shape)
                    second_cells = _stamp_cells(other, patch, anchor, shape)
                    overlap = first_cells & second_cells
                    if 0 < len(overlap) <= max(4, min(len(first_cells), len(second_cells)) // 3):
                        add_one(color, base)
                        add_one(color, other)
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    if need_clip:
        color = random.choice(["A", "B"])
        patch = patch_a if color == "A" else patch_b
        anchor = anchor_a if color == "A" else anchor_b
        border_candidates = [pos for pos in candidates if 0 in pos or pos[0] == shape[0] - 1 or pos[1] == shape[1] - 1]
        random.shuffle(border_candidates)
        placed = False
        for pos in border_candidates:
            if _clip_for_target(pos, patch, anchor, shape) and add_one(color, pos):
                placed = True
                break
        if not placed:
            return None

    pool = candidates[:]
    random.shuffle(pool)
    for pos in pool:
        if len(targets_a) >= count_a and len(targets_b) >= count_b:
            break
        choices = []
        if len(targets_a) < count_a:
            choices.append("A")
        if len(targets_b) < count_b:
            choices.append("B")
        if not choices:
            break
        color = random.choice(choices)
        add_one(color, pos)

    if len(targets_a) != count_a or len(targets_b) != count_b:
        return None

    placements = []
    for pos in targets_a:
        placements.append(("A", pos, _stamp_cells(pos, patch_a, anchor_a, shape)))
    for pos in targets_b:
        placements.append(("B", pos, _stamp_cells(pos, patch_b, anchor_b, shape)))
    placements.sort(key=lambda item: (item[1][1], item[1][0]))

    overlap_area = 0
    for i in range(len(placements)):
        for j in range(i + 1, len(placements)):
            overlap_area += len(placements[i][2] & placements[j][2])
    if need_overlap and overlap_area == 0:
        return None
    if not need_overlap and overlap_area != 0:
        return None
    if need_clip and not any(
        _clip_for_target(pos, patch_a if color == "A" else patch_b, anchor_a if color == "A" else anchor_b, shape)
        for color, pos, _cells in placements
    ):
        return None
    return targets_a, targets_b


def _expected_output(
    inp: np.ndarray,
    central_bbox: tuple[int, int, int, int],
    patch_a: np.ndarray,
    anchor_a: tuple[int, int],
    patch_b: np.ndarray,
    anchor_b: tuple[int, int],
) -> np.ndarray:
    out = inp.copy()
    r0, c0, r1, c1 = central_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = BG
    targets = []
    for rr, cc in np.argwhere(inp == ANCHOR_A):
        rr = int(rr)
        cc = int(cc)
        if not (r0 <= rr <= r1 and c0 <= cc <= c1):
            targets.append((cc, rr, ANCHOR_A))
    for rr, cc in np.argwhere(inp == ANCHOR_B):
        rr = int(rr)
        cc = int(cc)
        if not (r0 <= rr <= r1 and c0 <= cc <= c1):
            targets.append((cc, rr, ANCHOR_B))
    for col, row, color in sorted(targets):
        if color == ANCHOR_A:
            _stamp(out, patch_a, anchor_a, (row, col))
        else:
            _stamp(out, patch_b, anchor_b, (row, col))
    return out


def _validate_case(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    try:
        if not np.array_equal(main(inp), expected):
            return False
    except Exception:
        return False

    try:
        shortcut = _solve(inp, repair=False)
    except Exception:
        return False
    if np.array_equal(shortcut, expected):
        return False

    center = crop_bbox(inp, meta["central_bbox"])
    if int(np.sum(center == HIDE)) < 4:
        return False
    if np.count_nonzero(inp == ANCHOR_A) < 3 or np.count_nonzero(inp == ANCHOR_B) < 3:
        return False

    for key in ("a_hidden_meta", "b_hidden_meta"):
        items = meta[key]
        if not any(len(item["orbit"]) == 4 and len(item["visible"]) == 1 for item in items):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(GRID_HEIGHTS)
    w = random.choice(GRID_WIDTHS)
    size = random.choice(CENTRAL_SIZES)
    if h < size + 4 or w < size + 4:
        return None

    center_patch, center_meta = _build_center(size)
    repaired = center_meta["repaired"]
    patch_a, anchor_a, _coords_a = _sprite_from_repaired(repaired, ANCHOR_A, BODY_A)
    patch_b, anchor_b, _coords_b = _sprite_from_repaired(repaired, ANCHOR_B, BODY_B)

    top = random.randint(2, h - size - 2)
    left = random.randint(2, w - size - 2)
    central_bbox = (top, left, top + size - 1, left + size - 1)

    target_sets = _place_targets((h, w), central_bbox, patch_a, anchor_a, patch_b, anchor_b)
    if target_sets is None:
        return None
    targets_a, targets_b = target_sets

    inp = np.zeros((h, w), dtype=int)
    paste(inp, center_patch, top, left, clip=False)
    for r, c in targets_a:
        inp[r, c] = ANCHOR_A
    for r, c in targets_b:
        inp[r, c] = ANCHOR_B

    expected = _expected_output(inp, central_bbox, patch_a, anchor_a, patch_b, anchor_b)
    meta = {
        "central_bbox": central_bbox,
        "a_hidden_meta": center_meta["a_hidden_meta"],
        "b_hidden_meta": center_meta["b_hidden_meta"],
    }
    if not _validate_case(inp, expected, meta):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-repair stamping puzzle")
