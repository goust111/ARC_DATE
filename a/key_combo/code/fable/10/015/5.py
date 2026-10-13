from __future__ import annotations

import random
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np


BG = 0
FRAME = 5
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 400


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, mask: np.ndarray | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if mask is None:
        view[:] = patch
    else:
        view[mask] = patch[mask]


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _largest_template_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    best_area = -1
    best_bbox: tuple[int, int, int, int] | None = None
    for color in colors:
        for coords in _components(g == color, connectivity=8):
            area = int(coords.shape[0])
            if area > best_area:
                best_area = area
                best_bbox = _bbox_from_coords(coords)
    assert best_bbox is not None
    return best_bbox


def _all_transforms(arr: np.ndarray) -> list[np.ndarray]:
    out = []
    seen = set()
    for flip in (False, True):
        base = np.fliplr(arr) if flip else arr
        for k in range(4):
            cur = np.rot90(base, k)
            key = (cur.shape, cur.tobytes())
            if key not in seen:
                seen.add(key)
                out.append(cur.copy())
    return out


def _all_patch_mask_transforms(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    seen = set()
    for flip in (False, True):
        base_patch = np.fliplr(patch) if flip else patch
        base_mask = np.fliplr(mask) if flip else mask
        for k in range(4):
            cur_patch = np.rot90(base_patch, k)
            cur_mask = np.rot90(base_mask, k)
            key = (cur_mask.shape, cur_mask.tobytes(), cur_patch.tobytes())
            if key not in seen:
                seen.add(key)
                out.append((cur_patch.copy(), cur_mask.copy()))
    return out


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    keys = [(cur.shape, cur.tobytes()) for cur in _all_transforms(mask.astype(bool))]
    return min(keys)


@dataclass
class Cavity:
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    full_mask: np.ndarray
    shape_key: tuple[tuple[int, int], bytes]


@dataclass
class Piece:
    patch: np.ndarray
    mask: np.ndarray
    shape_key: tuple[tuple[int, int], bytes]


def _extract_cavities(sub: np.ndarray) -> list[Cavity]:
    holes = []
    for coords in _components(sub == BG, connectivity=4):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        mask = np.zeros((h, w), dtype=bool)
        rel = coords - np.array([r0, c0])
        mask[rel[:, 0], rel[:, 1]] = True
        full_mask = np.zeros(sub.shape, dtype=bool)
        full_mask[r0 : r1 + 1, c0 : c1 + 1][mask] = True
        holes.append(Cavity((r0, c0, r1, c1), mask, full_mask, _canonical_key(mask)))
    return holes


def _extract_pieces(grid: np.ndarray, template_bbox: tuple[int, int, int, int]) -> list[Piece]:
    g = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = template_bbox
    g[r0 : r1 + 1, c0 : c1 + 1] = BG
    pieces = []
    for coords in _components(g != BG, connectivity=8):
        pr0, pc0, pr1, pc1 = _bbox_from_coords(coords)
        patch = g[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
        mask = patch != BG
        pieces.append(Piece(patch, mask, _canonical_key(mask)))
    return pieces


def _build_orbits(cavities: list[Cavity], shape: tuple[int, int]) -> list[list[int]]:
    h, w = shape
    lookup = {c.full_mask.tobytes(): i for i, c in enumerate(cavities)}
    used = set()
    orbits: list[list[int]] = []
    for i, cavity in enumerate(cavities):
        if i in used:
            continue
        refl = np.fliplr(cavity.full_mask)
        j = lookup[refl.tobytes()]
        if j == i:
            orbits.append([i])
            used.add(i)
        else:
            orbit = sorted([i, j], key=lambda idx: cavities[idx].bbox[1])
            orbits.append(orbit)
            used.update(orbit)
    return orbits


def _feasible_patches(piece: Piece, cavity: Cavity) -> list[np.ndarray]:
    out = []
    seen = set()
    for patch_t, mask_t in _all_patch_mask_transforms(piece.patch, piece.mask):
        if mask_t.shape != cavity.mask.shape or not np.array_equal(mask_t, cavity.mask):
            continue
        cur = patch_t.copy()
        cur[~mask_t] = BG
        key = cur.tobytes()
        if key not in seen:
            seen.add(key)
            out.append(cur)
    return out


def _solve_axis_orbit(cavity: Cavity, pieces: list[Piece]) -> np.ndarray:
    assert len(pieces) == 1, f"axis orbit expects 1 piece, got {len(pieces)}"
    candidates = []
    for patch in _feasible_patches(pieces[0], cavity):
        if np.array_equal(patch, np.fliplr(patch)):
            candidates.append(patch)
    uniq = {(cand.shape, cand.tobytes()): cand for cand in candidates}
    assert len(uniq) == 1, f"axis orbit must have exactly 1 symmetric fill, got {len(uniq)}"
    return next(iter(uniq.values()))


def _solve_pair_orbit(cavity_l: Cavity, cavity_r: Cavity, pieces: list[Piece]) -> tuple[np.ndarray, np.ndarray]:
    candidates: dict[tuple[bytes, bytes], tuple[np.ndarray, np.ndarray]] = {}
    if len(pieces) == 1:
        piece = pieces[0]
        for side, cavity in enumerate((cavity_l, cavity_r)):
            for patch in _feasible_patches(piece, cavity):
                if side == 0:
                    left = patch
                    right = np.fliplr(left)
                else:
                    right = patch
                    left = np.fliplr(right)
                candidates[(left.tobytes(), right.tobytes())] = (left.copy(), right.copy())
    else:
        assert len(pieces) == 2, f"pair orbit expects 1 or 2 pieces, got {len(pieces)}"
        for left_piece, right_piece in ((pieces[0], pieces[1]), (pieces[1], pieces[0])):
            for left in _feasible_patches(left_piece, cavity_l):
                need = np.fliplr(left)
                for right in _feasible_patches(right_piece, cavity_r):
                    if np.array_equal(right, need):
                        candidates[(left.tobytes(), right.tobytes())] = (left.copy(), right.copy())
    assert len(candidates) == 1, f"pair orbit must have exactly 1 mirrored fill, got {len(candidates)}"
    return next(iter(candidates.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit each outside patterned piece into its matching cavity, then use whole-image left-right symmetry to resolve the ambiguous pose and the one missing piece.

    Input structure:
    - A large gray rectangular template sits on a black background, with several black polyomino cavities cut out inside it.
    - The template and its cavity layout are mirror-symmetric about a vertical axis; some cavities come as mirrored left/right pairs and some may sit on the axis.
    - Outside the template are multicolored patterned pieces. Each piece has the same shape as exactly one cavity orbit up to rotation or reflection, and there is exactly one fewer piece than cavities.

    Rule:
    - Crop to the template bounding box and start from a solid gray rectangle.
    - Match each outside piece to the unique cavity orbit with the same shape modulo D4 transforms.
    - For asymmetric cavity shapes, the fitting transform is fixed by the contour. For symmetric cavity shapes, keep only the fitting pose that makes each left/right cavity pair become horizontal mirrors of each other. For an axis cavity, keep only a pose whose filled pattern is itself horizontally symmetric.
    - The one cavity without a piece is filled by horizontally mirroring its partner cavity's resolved pattern.

    Key insight:
    - Shape matching alone determines where a piece belongs but not always how it should face; the global completed template must be left-right symmetric, and that same constraint also reconstructs the missing piece.
    """
    g = np.asarray(grid, dtype=int)
    template_bbox = _largest_template_bbox(g)
    sub = _crop(g, template_bbox)
    cavities = _extract_cavities(sub)
    pieces = _extract_pieces(g, template_bbox)
    assert len(pieces) + 1 == len(cavities), f"expected exactly one missing piece: pieces={len(pieces)} cavities={len(cavities)}"

    orbits = _build_orbits(cavities, sub.shape)
    orbit_by_key: dict[tuple[tuple[int, int], bytes], list[int]] = {}
    for orbit in orbits:
        orbit_by_key[cavities[orbit[0]].shape_key] = orbit
    assert len(orbit_by_key) == len(orbits), "every cavity orbit must have a unique canonical shape key"

    grouped: dict[int, list[Piece]] = {orbit[0]: [] for orbit in orbits}
    for piece in pieces:
        assert piece.shape_key in orbit_by_key, "piece shape must match some cavity orbit"
        grouped[orbit_by_key[piece.shape_key][0]].append(piece)

    out = np.full(sub.shape, FRAME, dtype=int)
    for orbit in orbits:
        key = orbit[0]
        orbit_pieces = grouped[key]
        if len(orbit) == 1:
            patch = _solve_axis_orbit(cavities[orbit[0]], orbit_pieces)
            r0, c0, _r1, _c1 = cavities[orbit[0]].bbox
            _paste(out, patch, r0, c0, cavities[orbit[0]].mask)
        else:
            left_patch, right_patch = _solve_pair_orbit(cavities[orbit[0]], cavities[orbit[1]], orbit_pieces)
            lr0, lc0, _lr1, _lc1 = cavities[orbit[0]].bbox
            rr0, rc0, _rr1, _rc1 = cavities[orbit[1]].bbox
            _paste(out, left_patch, lr0, lc0, cavities[orbit[0]].mask)
            _paste(out, right_patch, rr0, rc0, cavities[orbit[1]].mask)

    assert np.array_equal(out, np.fliplr(out)), "completed template must be left-right symmetric"
    return out


ASYM_SHAPES = [
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1], [1, 1], [0, 1], [0, 1]], dtype=bool),
]

SYM_SHAPES = [
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
]


def _shape_key_list(shapes: list[np.ndarray]) -> list[tuple[tuple[int, int], bytes]]:
    return [_canonical_key(shape) for shape in shapes]


ASYM_KEYS = _shape_key_list(ASYM_SHAPES)
SYM_KEYS = _shape_key_list(SYM_SHAPES)


def _distinct_orientations(mask: np.ndarray) -> list[np.ndarray]:
    return _all_transforms(mask.astype(bool))


def _axis_orientations(mask: np.ndarray) -> list[np.ndarray]:
    return [cur for cur in _distinct_orientations(mask) if np.array_equal(cur, np.fliplr(cur))]


def _sample_colors(palette: list[int], count: int) -> list[int]:
    colors = palette[:]
    random.shuffle(colors)
    return colors[:count]


def _random_non_lr_patch(mask: np.ndarray, palette: list[int]) -> np.ndarray:
    h, w = mask.shape
    coords = np.argwhere(mask)
    for _ in range(200):
        patch = np.zeros((h, w), dtype=int)
        colors = _sample_colors(palette, 2 if coords.shape[0] >= 3 else 1)
        style = random.choice(["random", "checker", "row", "col", "diag"])
        for r, c in coords:
            if len(colors) == 1:
                val = colors[0]
            elif style == "checker":
                val = colors[(r + c) % 2]
            elif style == "row":
                val = colors[r % 2]
            elif style == "col":
                val = colors[c % 2]
            elif style == "diag":
                val = colors[(r <= c)]
            else:
                val = random.choice(colors)
            patch[r, c] = val
        vals = {int(v) for v in patch[mask]}
        if len(vals) < min(2, len(colors)):
            continue
        if np.array_equal(patch, np.fliplr(patch)):
            continue
        return patch
    raise RuntimeError("failed to build non-symmetric patch")


def _random_lr_patch(mask: np.ndarray, palette: list[int]) -> np.ndarray:
    h, w = mask.shape
    patch = np.zeros((h, w), dtype=int)
    seen = set()
    groups = []
    for r, c in np.argwhere(mask):
        mate = (r, w - 1 - c)
        group = tuple(sorted({(int(r), int(c)), mate}))
        if group not in seen:
            seen.add(group)
            groups.append(group)
    colors = _sample_colors(palette, min(3, len(palette)))
    for _ in range(200):
        patch[:] = BG
        used = []
        for group in groups:
            color = random.choice(colors)
            used.append(color)
            for rr, cc in group:
                if mask[rr, cc]:
                    patch[rr, cc] = color
        if len(set(used)) < 2 and len(groups) >= 2:
            continue
        if np.array_equal(patch, np.fliplr(patch)):
            return patch.copy()
    raise RuntimeError("failed to build LR-symmetric patch")


def _random_piece_from_target(patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    variants = [cur_patch for cur_patch, _cur_mask in _all_patch_mask_transforms(patch, mask)]
    return random.choice(variants).copy()


def _can_place_group(
    occ: np.ndarray,
    placements: list[tuple[np.ndarray, int, int]],
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    new_cells = set()
    for mask, r0, c0 in placements:
        mh, mw = mask.shape
        if r0 < 1 or c0 < 1 or r0 + mh > h - 1 or c0 + mw > w - 1:
            return False
        for dr, dc in np.argwhere(mask):
            rr = r0 + int(dr)
            cc = c0 + int(dc)
            new_cells.add((rr, cc))
    for rr, cc in new_cells:
        if occ[rr, cc]:
            return False
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < h and 0 <= nc < w and occ[nr, nc] and (nr, nc) not in new_cells:
                return False
    return True


def _place_group(occ: np.ndarray, placements: list[tuple[np.ndarray, int, int]]) -> None:
    for mask, r0, c0 in placements:
        mh, mw = mask.shape
        occ[r0 : r0 + mh, c0 : c0 + mw][mask] = True


def _scatter_pieces(
    grid_shape: tuple[int, int],
    template_bbox: tuple[int, int, int, int],
    patches: list[np.ndarray],
) -> list[tuple[np.ndarray, int, int]] | None:
    h, w = grid_shape
    r0, c0, r1, c1 = template_bbox
    placed: list[tuple[np.ndarray, int, int]] = []
    blocked: list[tuple[int, int, int, int]] = [(r0 - 1, c0 - 1, r1 + 1, c1 + 1)]
    order = sorted(patches, key=lambda p: -int((p != BG).sum()))
    for patch in order:
        ph, pw = patch.shape
        choices = []
        for rr in range(h - ph + 1):
            for cc in range(w - pw + 1):
                br0, bc0, br1, bc1 = rr - 1, cc - 1, rr + ph, cc + pw
                ok = True
                for ar0, ac0, ar1, ac1 in blocked:
                    if not (br1 < ar0 or ar1 < br0 or bc1 < ac0 or ac1 < bc0):
                        ok = False
                        break
                if ok:
                    choices.append((rr, cc))
        if not choices:
            return None
        rr, cc = random.choice(choices)
        placed.append((patch, rr, cc))
        blocked.append((rr - 1, cc - 1, rr + ph, cc + pw))
    return placed


def _unique_shape_choices(include_axis: bool) -> tuple[list[np.ndarray], np.ndarray | None]:
    pair_shapes = []
    remaining_asym = list(range(len(ASYM_SHAPES)))
    random.shuffle(remaining_asym)
    asym_count = random.randint(2, 3)
    for idx in remaining_asym[:asym_count]:
        pair_shapes.append(ASYM_SHAPES[idx])
    axis_shape = None
    if include_axis:
        axis_candidates = [SYM_SHAPES[0], SYM_SHAPES[1]]
        axis_shape = random.choice(axis_candidates)
    return pair_shapes, axis_shape


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    th = random.randint(10, 15)
    tw = random.choice([11, 13, 15, 17])
    include_axis = True
    pair_shapes, axis_shape = _unique_shape_choices(include_axis)

    occ = np.zeros((th, tw), dtype=bool)
    orbit_records = []

    pair_specs = []
    for base in pair_shapes:
        left_mask = random.choice(_distinct_orientations(base))
        right_mask = np.fliplr(left_mask)
        pair_specs.append((base, left_mask, right_mask))
    pair_specs.sort(key=lambda item: -int(item[1].sum()))

    for base, left_mask, right_mask in pair_specs:
        placed = False
        h0, w0 = left_mask.shape
        max_left = (tw - 2 * w0 - 1) // 2
        if max_left < 1:
            raise RuntimeError("template too narrow")
        for _ in range(200):
            r = random.randint(1, th - h0 - 1)
            c = random.randint(1, max_left)
            rc = tw - c - w0
            if c + w0 >= rc:
                continue
            placements = [(left_mask, r, c), (right_mask, r, rc)]
            if not _can_place_group(occ, placements, (th, tw)):
                continue
            _place_group(occ, placements)
            orbit_records.append(
                {
                    "kind": "pair",
                    "base_key": _canonical_key(base),
                    "left_mask": left_mask,
                    "right_mask": right_mask,
                    "left_pos": (r, c),
                    "right_pos": (r, rc),
                }
            )
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place mirrored pair")

    if axis_shape is not None:
        placed = False
        for axis_mask in _axis_orientations(axis_shape):
            ah, aw = axis_mask.shape
            if (tw - aw) % 2 != 0:
                continue
            c = (tw - aw) // 2
            for _ in range(120):
                r = random.randint(1, th - ah - 1)
                placements = [(axis_mask, r, c)]
                if not _can_place_group(occ, placements, (th, tw)):
                    continue
                _place_group(occ, placements)
                orbit_records.append(
                    {
                        "kind": "axis",
                        "base_key": _canonical_key(axis_shape),
                        "mask": axis_mask,
                        "pos": (r, c),
                    }
                )
                placed = True
                break
            if placed:
                break
        if not placed:
            raise RuntimeError("failed to place axis cavity")

    pair_orbits = [record for record in orbit_records if record["kind"] == "pair"]
    missing_record = random.choice(pair_orbits)
    missing_side = random.choice(["left", "right"])
    palette = random.sample(list(PIECE_COLORS), random.randint(4, 6))

    out_crop = np.full((th, tw), FRAME, dtype=int)
    cavity_total = 0
    source_pieces: list[np.ndarray] = []
    for record in orbit_records:
        if record["kind"] == "pair":
            cavity_total += 2
            left_patch = _random_non_lr_patch(record["left_mask"], palette)
            right_patch = np.fliplr(left_patch)
            lr, lc = record["left_pos"]
            rr, rc = record["right_pos"]
            _paste(out_crop, left_patch, lr, lc, record["left_mask"])
            _paste(out_crop, right_patch, rr, rc, record["right_mask"])

            left_missing = record is missing_record and missing_side == "left"
            right_missing = record is missing_record and missing_side == "right"
            if not left_missing:
                piece = _random_piece_from_target(left_patch, record["left_mask"])
                source_pieces.append(piece)
            if not right_missing:
                piece = _random_piece_from_target(right_patch, record["right_mask"])
                source_pieces.append(piece)
        else:
            cavity_total += 1
            axis_patch = _random_lr_patch(record["mask"], palette)
            ar, ac = record["pos"]
            _paste(out_crop, axis_patch, ar, ac, record["mask"])
            source_pieces.append(_random_piece_from_target(axis_patch, record["mask"]))

    if len(source_pieces) + 1 != cavity_total:
        raise RuntimeError("piece count mismatch")

    top = random.randint(1, 4)
    left = random.randint(1, 4)
    h = min(30, th + top + random.randint(6, 11))
    w = min(30, tw + left + random.randint(6, 11))
    if h - th - top < 2 or w - tw - left < 2:
        raise RuntimeError("grid too tight")
    top = random.randint(1, h - th - 1)
    left = random.randint(1, w - tw - 1)
    template_bbox = (top, left, top + th - 1, left + tw - 1)

    scattered = _scatter_pieces((h, w), template_bbox, source_pieces)
    if scattered is None:
        raise RuntimeError("failed to scatter pieces")

    inp = np.zeros((h, w), dtype=int)
    inp[top : top + th, left : left + tw] = FRAME
    for record in orbit_records:
        if record["kind"] == "pair":
            lr, lc = record["left_pos"]
            rr, rc = record["right_pos"]
            inp[top + lr : top + lr + record["left_mask"].shape[0], left + lc : left + lc + record["left_mask"].shape[1]][record["left_mask"]] = BG
            inp[top + rr : top + rr + record["right_mask"].shape[0], left + rc : left + rc + record["right_mask"].shape[1]][record["right_mask"]] = BG
        else:
            ar, ac = record["pos"]
            inp[top + ar : top + ar + record["mask"].shape[0], left + ac : left + ac + record["mask"].shape[1]][record["mask"]] = BG

    for patch, rr, cc in scattered:
        _paste(inp, patch, rr, cc, patch != BG)

    try:
        solved = main(inp)
    except AssertionError as exc:
        template_sub = _crop(inp, template_bbox)
        cavities_dbg = _extract_cavities(template_sub)
        pieces_dbg = _extract_pieces(inp, template_bbox)
        raise RuntimeError(
            f"solver rejected generated instance: {exc}; cavities={len(cavities_dbg)} pieces={len(pieces_dbg)} "
            f"crop={template_sub.shape} full={inp.shape}"
        ) from exc
    if not np.array_equal(solved, out_crop):
        raise RuntimeError("solver mismatch")
    if not np.array_equal(out_crop, np.fliplr(out_crop)):
        raise RuntimeError("output must be symmetric")
    if len({record["base_key"] for record in orbit_records}) != len(orbit_records):
        raise RuntimeError("orbit shape keys must be unique")

    template_sub = _crop(inp, template_bbox)
    cavities = _extract_cavities(template_sub)
    pieces = _extract_pieces(inp, template_bbox)
    orbits = _build_orbits(cavities, template_sub.shape)
    if len(orbits) < 2:
        raise RuntimeError("too few cavity orbits")

    grouped: dict[tuple[tuple[int, int], bytes], list[Piece]] = {}
    for piece in pieces:
        grouped.setdefault(piece.shape_key, []).append(piece)

    axis_orbit_ok = False
    for orbit in orbits:
        if len(orbit) != 1:
            continue
        cavity = cavities[orbit[0]]
        if not np.array_equal(cavity.mask, np.fliplr(cavity.mask)):
            continue
        r0, c0, _r1, _c1 = cavity.bbox
        patch = out_crop[r0 : r0 + cavity.mask.shape[0], c0 : c0 + cavity.mask.shape[1]].copy()
        patch[~cavity.mask] = BG
        if np.array_equal(patch, np.fliplr(patch)):
            axis_orbit_ok = True
            break
    if axis_shape is not None and not axis_orbit_ok:
        raise RuntimeError("missing symmetric ambiguous orbit")

    return inp, out_crop


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            inp, _out = _build_instance()
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
