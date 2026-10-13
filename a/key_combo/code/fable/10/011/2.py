import random
from collections import Counter

import numpy as np


BG = 0
SPINE = 1
GREY = 5
DONOR_COLORS = [2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(mask, dtype=np.uint8)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    flipped = np.fliplr(mask)
    a = _shape_key(mask)
    b = _shape_key(flipped)
    return a if a <= b else b


def _has_internal_hole(mask: np.ndarray) -> bool:
    inv = ~np.asarray(mask, dtype=bool)
    for comp in _components(inv):
        if (
            np.any(comp[:, 0] == 0)
            or np.any(comp[:, 0] == inv.shape[0] - 1)
            or np.any(comp[:, 1] == 0)
            or np.any(comp[:, 1] == inv.shape[1] - 1)
        ):
            continue
        return True
    return False


def _parse_receptors(grid: np.ndarray, bg: int) -> list[dict]:
    grey_comps = _components(grid == GREY)
    spine_comps = _components(grid == SPINE)
    assert grey_comps, "expected grey half-sprites"
    assert spine_comps, "expected blue spine axes"

    grey_id = -np.ones(grid.shape, dtype=int)
    for idx, coords in enumerate(grey_comps):
        grey_id[coords[:, 0], coords[:, 1]] = idx

    used_grey: set[int] = set()
    receptors: list[dict] = []
    for spine_coords in spine_comps:
        r0, c0, r1, c1 = _bbox(spine_coords)
        assert c0 == c1, "spine must be vertical"
        expected_rows = np.arange(r0, r1 + 1, dtype=int)
        assert np.array_equal(np.sort(spine_coords[:, 0]), expected_rows), "spine must be solid"
        s = c0

        left_ids = {
            int(grey_id[r, s - 1])
            for r in range(r0, r1 + 1)
            if s - 1 >= 0 and grey_id[r, s - 1] >= 0
        }
        right_ids = {
            int(grey_id[r, s + 1])
            for r in range(r0, r1 + 1)
            if s + 1 < grid.shape[1] and grey_id[r, s + 1] >= 0
        }
        assert bool(left_ids) ^ bool(right_ids), "grey half must appear on exactly one side"
        side = "left" if left_ids else "right"
        ids = left_ids if left_ids else right_ids
        assert len(ids) == 1, "each spine must touch exactly one grey component"
        gid = next(iter(ids))
        assert gid not in used_grey, "grey component cannot serve multiple spines"
        used_grey.add(gid)

        half_coords = grey_comps[gid]
        half_mask = _mask_from_coords(half_coords)
        assert not np.array_equal(half_mask, np.fliplr(half_mask)), "half-sprite may not be mirror-symmetric"
        if side == "left":
            assert np.all(half_coords[:, 1] < s)
        else:
            assert np.all(half_coords[:, 1] > s)
        receptors.append(
            {
                "side": side,
                "spine_col": int(s),
                "coords": half_coords,
                "mask": half_mask,
            }
        )

    assert len(receptors) == len(grey_comps) == len(spine_comps), "every grey half needs one spine"
    return receptors


def _single_hole_mask(grid: np.ndarray, coords: np.ndarray, color: int, bg: int) -> np.ndarray | None:
    r0, c0, r1, c1 = _bbox(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any((sub != color) & (sub != bg)):
        return None
    if np.any(sub[0, :] == bg) or np.any(sub[-1, :] == bg) or np.any(sub[:, 0] == bg) or np.any(sub[:, -1] == bg):
        return None
    hole_parts = _components(sub == bg)
    if len(hole_parts) != 1:
        return None
    return _mask_from_coords(hole_parts[0])


def _parse_donors(grid: np.ndarray, bg: int) -> list[dict]:
    donors: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {bg, SPINE, GREY}):
        for coords in _components(grid == color):
            hole_mask = _single_hole_mask(grid, coords, color, bg)
            assert hole_mask is not None, "non-receptor colored objects must be holed donor blocks"
            donors.append({"color": int(color), "coords": coords, "hole_mask": hole_mask})
    assert donors, "expected donor blocks"
    return donors


def _solve_assignments(grid: np.ndarray) -> list[tuple[dict, dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    receptors = _parse_receptors(g, bg)
    donors = _parse_donors(g, bg)

    donors_by_key: dict[tuple[tuple[int, int], bytes], list[dict]] = {}
    for donor in donors:
        donors_by_key.setdefault(_shape_key(donor["hole_mask"]), []).append(donor)

    pairs: list[tuple[dict, dict]] = []
    used_colors: set[int] = set()
    for receptor in receptors:
        mirror_key = _shape_key(np.fliplr(receptor["mask"]))
        matches = donors_by_key.get(mirror_key, [])
        assert len(matches) == 1, "each receptor must have exactly one mirror-hole donor"
        donor = matches[0]
        assert donor["color"] not in used_colors, "donor colors should be unique once matched"
        used_colors.add(int(donor["color"]))
        pairs.append((receptor, donor))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each grey half-sprite by reclaiming its missing mirrored half from a donor block's negative-space hole.

    Input structure:
    - The background is the most common color.
    - Several receptors each consist of a vertical blue spine line with one connected grey half-sprite attached on exactly one side.
    - Other colored objects are solid monochrome rectangles whose interiors contain one background-colored hole.

    Rule:
    - For each receptor, compare the grey half-sprite with the donor holes and find the unique hole whose shape is a left-right mirror of the visible half, not a plain translated copy.
    - Use that donor block's color to recolor the visible half and to draw its reflected counterpart across the blue spine on the empty side.
    - Erase every blue spine and erase every donor block, leaving only the completed colored symmetric sprites.

    Key insight:
    - The missing half never appears directly in the input; it is encoded as a background hole inside a donor rectangle, and only mirror congruence identifies the correct color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = np.full(g.shape, bg, dtype=int)

    for receptor, donor in _solve_assignments(g):
        color = int(donor["color"])
        spine_col = int(receptor["spine_col"])
        for r, c in receptor["coords"]:
            rr = int(r)
            cc = int(c)
            mirror_c = 2 * spine_col - cc
            assert 0 <= mirror_c < g.shape[1], "mirrored half must stay in bounds"
            out[rr, cc] = color
            out[rr, mirror_c] = color

    return out


def _random_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    tr, tc = end
    mask[r, c] = True
    while (r, c) != (tr, tc):
        moves = []
        if r < tr:
            moves.append((1, 0))
        elif r > tr:
            moves.append((-1, 0))
        if c < tc:
            moves.append((0, 1))
        elif c > tc:
            moves.append((0, -1))
        dr, dc = random.choice(moves)
        r += dr
        c += dc
        mask[r, c] = True


def _random_connected_shape(h: int, w: int, min_fill: float = 0.4, max_fill: float = 0.72) -> np.ndarray:
    area_min = max(4, int(np.ceil(h * w * min_fill)))
    area_max = max(area_min, int(np.floor(h * w * max_fill)))
    target = random.randint(area_min, area_max)
    for _ in range(200):
        mask = np.zeros((h, w), dtype=bool)
        border_points = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        order = border_points[:]
        random.shuffle(order)
        cursor = order[0]
        mask[cursor] = True
        for pt in order[1:]:
            _random_path(mask, cursor, pt)
            cursor = pt

        while int(mask.sum()) < target:
            cells = np.argwhere(mask)
            rr, cc = map(int, cells[random.randrange(len(cells))])
            nbrs = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    nbrs.append((nr, nc))
            if not nbrs:
                break
            nr, nc = random.choice(nbrs)
            mask[nr, nc] = True

        if int(mask.sum()) < area_min:
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if len(_components(mask)) != 1:
            continue
        if _has_internal_hole(mask):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    raise RuntimeError("failed to generate non-symmetric half-shape")


def _make_receptor_patches(visible_mask: np.ndarray, side: str, color: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = visible_mask.shape
    inp = np.zeros((h, 2 * w + 1), dtype=int)
    out = np.zeros_like(inp)
    if side == "left":
        left_mask = visible_mask
        right_mask = np.fliplr(visible_mask)
        inp[:, :w][left_mask] = GREY
        inp[:, w] = SPINE
    else:
        left_mask = np.fliplr(visible_mask)
        right_mask = visible_mask
        inp[:, w] = SPINE
        inp[:, w + 1 :][right_mask] = GREY
    out[:, :w][left_mask] = color
    out[:, w + 1 :][right_mask] = color
    return inp, out


def _make_donor_patch(hole_mask: np.ndarray, color: int) -> np.ndarray:
    h, w = hole_mask.shape
    top = random.randint(1, 2)
    bottom = random.randint(1, 2)
    left = random.randint(1, 2)
    right = random.randint(1, 2)
    patch = np.full((h + top + bottom, w + left + right), int(color), dtype=int)
    patch[top : top + h, left : left + w][hole_mask] = BG
    return patch


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0], r0 + h + 1)
    cc1 = min(occupied.shape[1], c0 + w + 1)
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def _place_items(rows: int, cols: int, items: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.zeros((rows, cols), dtype=int)
    out = np.zeros((rows, cols), dtype=int)
    occupied = np.zeros((rows, cols), dtype=bool)
    order = list(range(len(items)))
    random.shuffle(order)
    order.sort(key=lambda idx: items[idx]["input_patch"].size, reverse=True)

    for idx in order:
        item = items[idx]
        patch = item["input_patch"]
        out_patch = item["output_patch"]
        h, w = patch.shape
        if h > rows or w > cols:
            return None
        placed = False
        for _ in range(400):
            r0 = random.randint(0, rows - h)
            c0 = random.randint(0, cols - w)
            if not _can_place(occupied, r0, c0, h, w):
                continue
            view = inp[r0 : r0 + h, c0 : c0 + w]
            out_view = out[r0 : r0 + h, c0 : c0 + w]
            view[patch != BG] = patch[patch != BG]
            out_view[out_patch != BG] = out_patch[out_patch != BG]
            occupied[r0 : r0 + h, c0 : c0 + w] = True
            placed = True
            break
        if not placed:
            return None
    return inp, out


def _validate_scene(scene: dict) -> None:
    visible_keys = [_shape_key(rec["visible_mask"]) for rec in scene["receptors"]]
    mirror_keys = [_shape_key(np.fliplr(rec["visible_mask"])) for rec in scene["receptors"]]
    assert len({_canonical_key(rec["visible_mask"]) for rec in scene["receptors"]}) == len(scene["receptors"])
    for rec in scene["receptors"]:
        assert not np.array_equal(rec["visible_mask"], np.fliplr(rec["visible_mask"]))
        assert not _has_internal_hole(rec["visible_mask"])

    hole_keys = [_shape_key(d["hole_mask"]) for d in scene["donors"]]
    assert len(set(hole_keys)) == len(hole_keys), "all donor holes should be shape-distinct"
    assert len({int(d["color"]) for d in scene["donors"]}) == len(scene["donors"]), "donor colors must be unique"

    key_counts = Counter(hole_keys)
    trap_count = 0
    extra_count = 0
    for donor in scene["donors"]:
        kind = donor["kind"]
        key = _shape_key(donor["hole_mask"])
        if kind == "mirror":
            assert key in mirror_keys and key_counts[key] == 1
        elif kind == "trap":
            assert key in visible_keys and key_counts[key] == 1
            trap_count += 1
        else:
            assert key not in visible_keys and key not in mirror_keys
            extra_count += 1
        assert not _has_internal_hole(donor["hole_mask"])
    assert trap_count >= 1
    assert extra_count >= 1


def _build_scene() -> dict:
    receptor_count = random.choice([2, 3])
    sides = ["left", "right"]
    while len(sides) < receptor_count:
        sides.append(random.choice(["left", "right"]))
    random.shuffle(sides)

    donor_colors = random.sample(DONOR_COLORS, 2 * receptor_count + (1 if receptor_count == 3 else random.choice([1, 2])))
    color_cursor = 0
    used_canon: set[tuple[tuple[int, int], bytes]] = set()
    used_raw: set[tuple[tuple[int, int], bytes]] = set()
    receptors: list[dict] = []
    donors: list[dict] = []

    for idx in range(receptor_count):
        for _ in range(200):
            h = random.randint(4, 7)
            w = random.randint(2, 4)
            base = _random_connected_shape(h, w)
            visible = base if sides[idx] == "left" else np.fliplr(base)
            canon = _canonical_key(visible)
            if canon in used_canon:
                continue
            if _shape_key(visible) in used_raw or _shape_key(np.fliplr(visible)) in used_raw:
                continue
            used_canon.add(canon)
            break
        else:
            raise RuntimeError("failed to find unique half-shape")

        mirror_color = donor_colors[color_cursor]
        trap_color = donor_colors[color_cursor + 1]
        color_cursor += 2
        receptors.append(
            {
                "visible_mask": visible,
                "side": sides[idx],
                "mirror_color": mirror_color,
            }
        )
        donors.append({"hole_mask": np.fliplr(visible), "color": mirror_color, "kind": "mirror"})
        donors.append({"hole_mask": visible, "color": trap_color, "kind": "trap"})
        used_raw.add(_shape_key(visible))
        used_raw.add(_shape_key(np.fliplr(visible)))

    extra_count = len(donor_colors) - color_cursor
    for _ in range(extra_count):
        for _try in range(200):
            h = random.randint(3, 7)
            w = random.randint(2, 5)
            extra = _random_connected_shape(h, w, min_fill=0.35, max_fill=0.68)
            key = _shape_key(extra)
            if key in used_raw or _shape_key(np.fliplr(extra)) in used_raw:
                continue
            donors.append({"hole_mask": extra, "color": donor_colors[color_cursor], "kind": "extra"})
            color_cursor += 1
            used_raw.add(key)
            break
        else:
            raise RuntimeError("failed to create unrelated extra donor")

    scene = {"receptors": receptors, "donors": donors}
    _validate_scene(scene)
    return scene


def _render_scene(scene: dict) -> tuple[np.ndarray, np.ndarray] | None:
    items: list[dict] = []
    for rec in scene["receptors"]:
        inp_patch, out_patch = _make_receptor_patches(rec["visible_mask"], rec["side"], rec["mirror_color"])
        items.append({"input_patch": inp_patch, "output_patch": out_patch})
    for donor in scene["donors"]:
        inp_patch = _make_donor_patch(donor["hole_mask"], donor["color"])
        items.append({"input_patch": inp_patch, "output_patch": np.zeros_like(inp_patch)})

    for _ in range(40):
        rows = random.randint(22, 30)
        cols = random.randint(22, 30)
        rendered = _place_items(rows, cols, items)
        if rendered is not None:
            return rendered
    return None


def _gen_attempt() -> np.ndarray | None:
    scene = _build_scene()
    rendered = _render_scene(scene)
    if rendered is None:
        return None
    inp, expected = rendered
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid negative-donor ARC input")
