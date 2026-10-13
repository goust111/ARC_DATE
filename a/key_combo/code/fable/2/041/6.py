from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
PALETTE = tuple(range(1, 10))
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _get_bg(grid: np.ndarray) -> int:
    vals = Counter(map(int, grid.ravel()))
    return int(vals.most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=int)
    r0, c0, r1, c1 = _bbox_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    out[rel[:, 0], rel[:, 1]] = True
    return out


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_coords(coords)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    b = _bbox(mask)
    assert b is not None
    return _crop_bbox(mask.astype(bool), b).astype(bool)


def _mask_components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    deltas = N4 if connectivity == 4 else N8
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_mono_components(grid: np.ndarray, bg: int, connectivity: int = 4) -> list[dict]:
    out: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        for coords in _mask_components(grid == color, connectivity=connectivity):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = np.asarray(mask, dtype=bool)
    return tuple(map(int, m.shape)), m.tobytes()


def _distinct_transforms(mask: np.ndarray) -> list[np.ndarray]:
    base = np.asarray(mask, dtype=bool)
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    for flip in (False, True):
        cur = np.flipud(base) if flip else base
        for k in range(4):
            m = np.rot90(cur, k)
            key = _mask_key(m)
            if key not in seen:
                seen.add(key)
                out.append(m.copy())
    return out


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_mask_key(m) for m in _distinct_transforms(mask))


def _apply_dihedral(mask: np.ndarray, code: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    if code // 4:
        out = np.flipud(out)
    return np.rot90(out, code % 4).copy()


def _coords_set(mask: np.ndarray) -> set[Coord]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _subset_translations(complete: np.ndarray, damaged: np.ndarray) -> list[Coord]:
    ccoords = np.argwhere(complete)
    dcoords = np.argwhere(damaged)
    cset = {tuple(map(int, rc)) for rc in ccoords}
    translations: set[Coord] = set()
    for cr, cc in ccoords:
        for dr, dc in dcoords:
            shift = (int(cr - dr), int(cc - dc))
            shifted = {(int(r + shift[0]), int(c + shift[1])) for r, c in dcoords}
            if shifted <= cset and shifted != cset:
                translations.add(shift)
    return sorted(translations)


def _missing_from_alignment(
    complete: np.ndarray, damaged: np.ndarray, shift: Coord
) -> np.ndarray:
    cset = _coords_set(complete)
    dset = {
        (int(r + shift[0]), int(c + shift[1]))
        for r, c in np.argwhere(damaged)
    }
    missing = np.array(sorted(cset - dset), dtype=int)
    assert missing.size > 0
    return _coords_to_mask(missing)


def _infer_missing_mask(
    mask_a: np.ndarray, mask_b: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for order, (complete, damaged) in enumerate(((mask_a, mask_b), (mask_b, mask_a))):
        for shift in _subset_translations(complete, damaged):
            missing = _missing_from_alignment(complete, damaged, shift)
            candidates.append(
                {
                    "order": order,
                    "shift": shift,
                    "missing": missing,
                    "complete": complete,
                    "damaged": damaged,
                }
            )
    assert len(candidates) == 1, "expected exactly one complete/damaged alignment"
    info = candidates[0]
    return info["missing"], info


def _rand_choice(rng: np.random.Generator, seq):
    return seq[int(rng.integers(0, len(seq)))]


def _apply_cells(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    cells = np.argwhere(mask)
    for r, c in cells:
        grid[top + int(r), left + int(c)] = int(color)


def _mark_halo(blocked: np.ndarray, cells: np.ndarray, halo: int = 1) -> None:
    H, W = blocked.shape
    for r, c in cells:
        rr = int(r)
        cc = int(c)
        r0 = max(0, rr - halo)
        r1 = min(H, rr + halo + 1)
        c0 = max(0, cc - halo)
        c1 = min(W, cc + halo + 1)
        blocked[r0:r1, c0:c1] = True


def _can_place_mask(blocked: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    H, W = blocked.shape
    h, w = mask.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    for r, c in np.argwhere(mask):
        if blocked[top + int(r), left + int(c)]:
            return False
    return True


def _mask_touches(a: np.ndarray, b: np.ndarray) -> bool:
    aset = _coords_set(a)
    bset = _coords_set(b)
    for r, c in aset:
        for dr, dc in N4:
            if (r + dr, c + dc) in bset:
                return True
    return False


def _is_contiguous(mask: np.ndarray) -> bool:
    return len(_mask_components(mask.astype(bool), connectivity=4)) == 1


def _grow_polyomino(
    rng: np.random.Generator,
    area: int,
    *,
    min_orientations: int = 1,
    max_tries: int = 200,
) -> np.ndarray:
    for _ in range(max_tries):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in N4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.add(nxt)
            cells.add(_rand_choice(rng, sorted(frontier)))
        arr = np.array(sorted(cells), dtype=int)
        arr -= arr.min(axis=0)
        mask = np.zeros(tuple(arr.max(axis=0) + 1), dtype=bool)
        mask[arr[:, 0], arr[:, 1]] = True
        if len(_distinct_transforms(mask)) < min_orientations:
            continue
        return mask
    raise RuntimeError("failed to grow a suitable polyomino")


HOLE_LIBRARY = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1], [0, 1], [0, 1]], dtype=bool),
]
HOLE_KEY_SET = {_canonical_key(mask) for mask in HOLE_LIBRARY}


def _solve(grid: np.ndarray, return_details: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    comps = _find_mono_components(g, bg=bg, connectivity=8)
    assert comps, "expected a template frame and outside sprites"
    max_area = max(comp["area"] for comp in comps)
    frame_candidates = [comp for comp in comps if comp["area"] == max_area]
    assert len(frame_candidates) == 1, "expected one largest frame component"
    frame = frame_candidates[0]
    frame_color = int(frame["color"])
    frame_bbox = frame["bbox"]

    frame_crop = _crop_bbox(g, frame_bbox)
    hole_mask = frame_crop != frame_color
    holes: list[dict[str, object]] = []
    for coords in _mask_components(hole_mask, connectivity=4):
        hb = _bbox_coords(coords)
        hr0, hc0, _hr1, _hc1 = hb
        rel = coords - np.array([hr0, hc0], dtype=int)
        mask = np.zeros((rel[:, 0].max() + 1, rel[:, 1].max() + 1), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append(
            {
                "top": int(hr0),
                "left": int(hc0),
                "mask": mask,
                "key": _canonical_key(mask),
            }
        )
    assert holes, "expected at least one hole"

    outside = g.copy()
    r0, c0, r1, c1 = frame_bbox
    outside[r0 : r1 + 1, c0 : c1 + 1] = bg
    sprite_comps = _find_mono_components(outside, bg=bg, connectivity=8)
    assert sprite_comps, "expected outside sprites"

    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in sprite_comps:
        by_color[int(comp["color"])].append(comp)

    pair_info: dict[int, dict[str, object]] = {}
    for color, pair in by_color.items():
        assert len(pair) == 2, "each sprite color must appear exactly twice"
        masks = [_coords_to_mask(comp["coords"]) for comp in pair]
        missing, info = _infer_missing_mask(masks[0], masks[1])
        pair_info[color] = {
            "missing": missing,
            "key": _canonical_key(missing),
            "complete_key": _canonical_key(info["complete"]),
            "damaged_key": _canonical_key(info["damaged"]),
            "shift": info["shift"],
        }

    key_to_color: dict[tuple[tuple[int, int], bytes], int] = {}
    for color, info in pair_info.items():
        key = info["key"]
        assert key not in key_to_color, "hole keys must map to unique sprite colors"
        key_to_color[key] = int(color)
    assert set(key_to_color) == {hole["key"] for hole in holes}

    out = np.full(frame_crop.shape, frame_color, dtype=int)
    for hole in holes:
        color = key_to_color.get(hole["key"])
        assert color is not None, "every hole must match exactly one pair difference"
        top = int(hole["top"])
        left = int(hole["left"])
        mask = hole["mask"]
        out[top : top + mask.shape[0], left : left + mask.shape[1]][mask] = color

    if not return_details:
        return out
    return out, {
        "bg": bg,
        "frame_bbox": frame_bbox,
        "frame_color": frame_color,
        "holes": holes,
        "pairs": pair_info,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each template hole by subtracting the damaged sprite from its full same-color partner.

    Input structure:
    - The background contains one large monochrome template component whose bbox encloses several background-color holes.
    - Outside the template, every non-template color appears on exactly two monochrome sprites of that color.
    - Within each same-color pair, one sprite is a complete shape and the other is a damaged copy obtained by deleting cells, with a unique translation alignment back into the complete one.

    Rule:
    - Identify the largest monochrome component as the template and crop the output to its bounding box.
    - For each same-color outside pair, align the damaged sprite into the complete sprite by the unique valid translation and compute the missing-cell mask as complete minus damaged.
    - Fill the cropped template solidly with the template color, then recolor each hole with the color of the unique pair whose missing-cell mask matches that hole up to rotation or reflection.

    Key insight:
    - The color key is not any visible outside sprite itself; it is the hidden deficit shape revealed only after subtracting the damaged copy from the complete one.
    """
    return _solve(grid, return_details=False)


def _build_pair_for_missing(
    rng: np.random.Generator, missing: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    missing_key = _canonical_key(missing)
    missing_area = int(missing.sum())
    for _ in range(400):
        damaged_area = int(rng.integers(max(4, missing_area + 1), min(9, missing_area + 5)))
        damaged = _grow_polyomino(rng, damaged_area, min_orientations=2)
        if _canonical_key(damaged) == missing_key:
            continue

        candidates: list[tuple[np.ndarray, np.ndarray]] = []
        damaged_transforms = _distinct_transforms(damaged)
        rng.shuffle(damaged_transforms)
        missing_transforms = _distinct_transforms(missing)
        rng.shuffle(missing_transforms)

        for dmask in damaged_transforms:
            dcoords = np.argwhere(dmask)
            dset = {tuple(map(int, rc)) for rc in dcoords}
            for mmask in missing_transforms:
                mcoords = np.argwhere(mmask)
                for dr in range(-mmask.shape[0], dmask.shape[0] + 1):
                    for dc in range(-mmask.shape[1], dmask.shape[1] + 1):
                        shifted = {(int(r + dr), int(c + dc)) for r, c in mcoords}
                        if shifted & dset:
                            continue
                        touch = False
                        for r, c in shifted:
                            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                                if (rr, cc) in dset:
                                    touch = True
                                    break
                            if touch:
                                break
                        if not touch:
                            continue
                        union = np.array(sorted(dset | shifted), dtype=int)
                        union -= union.min(axis=0)
                        complete = np.zeros(tuple(union.max(axis=0) + 1), dtype=bool)
                        complete[union[:, 0], union[:, 1]] = True
                        if complete.shape[0] > 8 or complete.shape[1] > 8:
                            continue
                        if not _is_contiguous(complete):
                            continue
                        complete_key = _canonical_key(complete)
                        damaged_key = _canonical_key(dmask)
                        if complete_key == missing_key:
                            continue
                        if complete_key in HOLE_KEY_SET or damaged_key in HOLE_KEY_SET:
                            continue
                        try:
                            inferred, _info = _infer_missing_mask(complete, dmask)
                        except AssertionError:
                            continue
                        if _canonical_key(inferred) != missing_key:
                            continue
                        candidates.append((complete, dmask))
        if candidates:
            complete, damaged_mask = _rand_choice(rng, candidates)
            return complete.astype(bool), damaged_mask.astype(bool)
    raise RuntimeError("failed to construct a unique complete/damaged pair")


def _build_pair_library() -> dict[tuple[tuple[int, int], bytes], tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(0)
    out: dict[tuple[tuple[int, int], bytes], tuple[np.ndarray, np.ndarray]] = {}
    for hole in HOLE_LIBRARY:
        key = _canonical_key(hole)
        complete, damaged = _build_pair_for_missing(rng, hole)
        out[key] = (complete.copy(), damaged.copy())
    return out


PAIR_LIBRARY = _build_pair_library()


def _choose_holes(rng: np.random.Generator, count: int) -> list[np.ndarray]:
    order = list(range(len(HOLE_LIBRARY)))
    rng.shuffle(order)
    holes = [HOLE_LIBRARY[idx].copy() for idx in order[:count]]
    if all(len(_distinct_transforms(hole)) == 1 for hole in holes):
        holes[0] = HOLE_LIBRARY[0].copy()
    return holes


def _place_holes_in_frame(
    rng: np.random.Generator, hole_masks: list[np.ndarray]
) -> tuple[int, int, list[tuple[int, int, np.ndarray]]]:
    max_h = max(mask.shape[0] for mask in hole_masks)
    max_w = max(mask.shape[1] for mask in hole_masks)
    total_area = sum(int(mask.sum()) for mask in hole_masks)
    min_side = max(max_h + 3, max_w + 3, int(math.ceil(math.sqrt(total_area * 3))) + 2, 9)

    for _ in range(250):
        frame_h = int(rng.integers(min_side, min(19, min_side + 7)))
        frame_w = int(rng.integers(min_side, min(19, min_side + 7)))
        blocked = np.zeros((frame_h, frame_w), dtype=bool)
        placements: list[tuple[int, int, np.ndarray]] = []
        ok = True
        order = sorted(hole_masks, key=lambda m: int(m.sum()), reverse=True)
        for mask in order:
            h, w = mask.shape
            candidates: list[tuple[int, int]] = []
            for top in range(1, frame_h - h):
                for left in range(1, frame_w - w):
                    if _can_place_mask(blocked, mask, top, left):
                        candidates.append((top, left))
            if not candidates:
                ok = False
                break
            top, left = _rand_choice(rng, candidates)
            placements.append((top, left, mask))
            cells = np.argwhere(mask) + np.array([top, left], dtype=int)
            _mark_halo(blocked, cells, halo=1)
        if ok:
            return frame_h, frame_w, placements
    raise RuntimeError("failed to place holes inside the frame")


def _place_outside_sprites(
    rng: np.random.Generator,
    grid: np.ndarray,
    frame_bbox: BBox,
    sprites: list[tuple[np.ndarray, int]],
) -> bool:
    H, W = grid.shape
    blocked = np.zeros((H, W), dtype=bool)
    r0, c0, r1, c1 = frame_bbox
    blocked[max(0, r0 - 1) : min(H, r1 + 2), max(0, c0 - 1) : min(W, c1 + 2)] = True

    placed = sorted(sprites, key=lambda item: int(item[0].sum()), reverse=True)
    for mask, color in placed:
        candidates: list[tuple[int, int]] = []
        h, w = mask.shape
        for top in range(0, H - h + 1):
            for left in range(0, W - w + 1):
                if _can_place_mask(blocked, mask, top, left):
                    candidates.append((top, left))
        if not candidates:
            return False
        top, left = _rand_choice(rng, candidates)
        _apply_cells(grid, mask, top, left, color)
        cells = np.argwhere(mask) + np.array([top, left], dtype=int)
        _mark_halo(blocked, cells, halo=1)
    return True


def _validate_instance(inp: np.ndarray, expected: np.ndarray) -> None:
    out, details = _solve(inp, return_details=True)
    assert np.array_equal(out, expected)

    hole_keys = [hole["key"] for hole in details["holes"]]
    assert len(hole_keys) == len(set(hole_keys))
    assert len(hole_keys) == len(details["pairs"])

    for color, info in details["pairs"].items():
        assert info["key"] in hole_keys
        assert info["complete_key"] not in hole_keys
        assert info["damaged_key"] not in hole_keys
        assert info["key"] != info["complete_key"]
        assert info["key"] != info["damaged_key"]
        assert isinstance(color, int)

    for hole in details["holes"]:
        matches = [color for color, info in details["pairs"].items() if info["key"] == hole["key"]]
        assert len(matches) == 1


def _generate_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = int(rng.integers(2, 5))
    hole_bases = _choose_holes(rng, pair_count)

    pair_payloads = []
    seen_keys: set[tuple[tuple[int, int], bytes]] = set()
    colors = list(PALETTE)
    rng.shuffle(colors)

    for hole_base in hole_bases:
        hole_key = _canonical_key(hole_base)
        if hole_key in seen_keys:
            return None
        seen_keys.add(hole_key)
        base_complete, base_damaged = PAIR_LIBRARY[hole_key]
        complete = base_complete.copy()
        damaged = base_damaged.copy()

        transform_idx = int(rng.integers(0, 8))
        complete = _apply_dihedral(complete, transform_idx)
        damaged = _apply_dihedral(damaged, transform_idx)

        hole_variants = _distinct_transforms(hole_base)
        if len(hole_variants) > 1 and rng.random() < 0.8:
            non_identity = [m for m in hole_variants if _mask_key(m) != _mask_key(hole_base)]
            hole_mask = _rand_choice(rng, non_identity) if non_identity else hole_base
        else:
            hole_mask = _rand_choice(rng, hole_variants)

        color = int(colors.pop())
        pair_payloads.append(
            {
                "color": color,
                "hole_mask": hole_mask,
                "hole_key": hole_key,
                "complete_key": _canonical_key(complete),
                "damaged_key": _canonical_key(damaged),
                "complete": complete,
                "damaged": damaged,
            }
        )

    hole_key_set = {payload["hole_key"] for payload in pair_payloads}
    for payload in pair_payloads:
        if payload["complete_key"] in hole_key_set or payload["damaged_key"] in hole_key_set:
            return None

    frame_color = int(colors.pop())
    hole_masks = [item["hole_mask"] for item in pair_payloads]
    frame_h, frame_w, hole_places = _place_holes_in_frame(rng, hole_masks)

    grid_h = int(rng.integers(max(frame_h + 8, 22), 31))
    grid_w = int(rng.integers(max(frame_w + 8, 22), 31))
    if grid_h > 30 or grid_w > 30:
        return None
    frame_top = int(rng.integers(1, grid_h - frame_h))
    frame_left = int(rng.integers(1, grid_w - frame_w))
    frame_bbox = (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    inp[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = frame_color

    expected = np.full((frame_h, frame_w), frame_color, dtype=int)
    place_by_key = {
        _canonical_key(hole_mask): (top, left, hole_mask) for top, left, hole_mask in hole_places
    }
    assert len(place_by_key) == len(pair_payloads)
    for payload in pair_payloads:
        top, left, hole_mask = place_by_key[payload["hole_key"]]
        for r, c in np.argwhere(hole_mask):
            inp[frame_top + top + int(r), frame_left + left + int(c)] = BG
            expected[top + int(r), left + int(c)] = int(payload["color"])

    sprites: list[tuple[np.ndarray, int]] = []
    for payload in pair_payloads:
        pair_sprites = [(payload["complete"], payload["color"]), (payload["damaged"], payload["color"])]
        rng.shuffle(pair_sprites)
        sprites.extend(pair_sprites)
    if not _place_outside_sprites(rng, inp, frame_bbox, sprites):
        return None

    _validate_instance(inp, expected)
    return inp, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        try:
            result = _generate_attempt(rng)
        except (AssertionError, RuntimeError, ValueError):
            continue
        if result is not None:
            inp, _expected = result
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
