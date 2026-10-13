from __future__ import annotations

import random

import numpy as np


BG = 0
ARROW = 2
LOCK = 9
MAIN_COLORS = (1, 3, 4, 5, 6, 7, 8)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

ARROW_PATTERNS = {
    3: (
        {(0, 0), (0, 1), (1, 0)},  # top-left bend -> clockwise
        {(0, 1), (1, 0), (1, 1)},  # bottom-right bend -> clockwise
    ),
    1: (
        {(0, 0), (0, 1), (1, 1)},  # top-right bend -> counterclockwise
        {(0, 0), (1, 0), (1, 1)},  # bottom-left bend -> counterclockwise
    ),
}

HOLE_SHAPES = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
)

GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _c in coords]
    cs = [c for _r, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _mask_from_coords(shape: tuple[int, int], coords: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for r, c in coords:
        out[int(r), int(c)] = True
    return out


def _crop_non_bg(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    assert coords.size > 0, "expected foreground pixels"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _erode4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            ok = True
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w or not mask[nr, nc]:
                    ok = False
                    break
            out[r, c] = ok
    return out


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _arrow_rotation_k(grid: np.ndarray) -> int:
    coords = [(int(r), int(c)) for r, c in np.argwhere(grid == ARROW)]
    assert len(coords) == 3, "expected a 3-cell red arrow"
    r0, c0, r1, c1 = _bbox(coords)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "arrow must fit in a 2x2 box"
    rel = {(r - r0, c - c0) for r, c in coords}
    for k, patterns in ARROW_PATTERNS.items():
        if rel in patterns:
            return int(k)
    raise AssertionError("arrow must be an L triomino with a supported bend")


def _extract_holes(main_patch: np.ndarray) -> list[dict]:
    h, w = main_patch.shape
    bg_comps = _components(main_patch == BG)
    lock_cells = {(int(r), int(c)) for r, c in np.argwhere(main_patch == LOCK)}
    holes: list[dict] = []
    for comp in bg_comps:
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        rel_coords = [(r - r0, c - c0) for r, c in comp]
        hole_mask = _mask_from_coords((r1 - r0 + 1, c1 - c0 + 1), rel_coords)
        comp_set = set(comp)
        locks = []
        for lr, lc in lock_cells:
            if any((lr + dr, lc + dc) in comp_set for dr, dc in DIRS4):
                locks.append((int(lr), int(lc)))
        holes.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "coords": comp,
                "coord_set": comp_set,
                "mask": hole_mask,
                "locks": sorted(locks),
            }
        )
    return holes


def _largest_component_patch(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    comps = _components(grid != BG)
    assert comps, "expected foreground components"
    main_coords = max(comps, key=len)
    r0, c0, r1, c1 = _bbox(main_coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy(), (int(r0), int(c0), int(r1), int(c1))


def _fragment_candidates(fragment: np.ndarray, holes: list[dict]) -> list[tuple[int, np.ndarray]]:
    seen: set[bytes] = set()
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(fragment, k)
        key = rotated.tobytes() + np.array(rotated.shape, dtype=np.int16).tobytes()
        if key in seen:
            continue
        seen.add(key)
        variants.append(rotated)

    candidates: list[tuple[int, np.ndarray]] = []
    for hole_index, hole in enumerate(holes):
        if len(hole["locks"]) != 1:
            continue
        lock = hole["locks"][0]
        for rotated in variants:
            if rotated.shape != hole["mask"].shape:
                continue
            if not np.array_equal(rotated != BG, hole["mask"]):
                continue
            lock_pixels = np.argwhere(rotated == LOCK)
            if len(lock_pixels) != 1:
                continue
            rr, cc = map(int, lock_pixels[0])
            abs_lock = (hole["bbox"][0] + rr, hole["bbox"][1] + cc)
            if _manhattan(abs_lock, lock) == 1:
                candidates.append((int(hole_index), rotated))
    return candidates


def _assign_fragments(fragments: list[np.ndarray], holes: list[dict]) -> dict[int, tuple[int, np.ndarray]]:
    options = {i: _fragment_candidates(fragment, holes) for i, fragment in enumerate(fragments)}
    assert all(options.values()), "every fragment must match at least one hole"

    order = sorted(options, key=lambda i: len({hole for hole, _patch in options[i]}))
    assignment: dict[int, tuple[int, np.ndarray]] = {}
    used_holes: set[int] = set()

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        frag_idx = order[pos]
        for hole_idx, rotated in options[frag_idx]:
            if hole_idx in used_holes:
                continue
            used_holes.add(int(hole_idx))
            assignment[frag_idx] = (int(hole_idx), rotated)
            if dfs(pos + 1):
                return True
            used_holes.remove(int(hole_idx))
            del assignment[frag_idx]
        return False

    assert dfs(0), "expected a perfect fragment-to-hole assignment"
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the locked fragments into the marked cavities, then rotate the recovered main sprite by the red bend cue.

    Input structure:
    - The background is black.
    - One large connected main sprite uses a single main color plus some black cavities,
      one red 3-cell L marker inside the sprite, and one or more maroon lock pixels on
      cavity borders.
    - Separate smaller components are loose fragments made from the same main color; each
      fragment contains exactly one maroon lock pixel on its edge.

    Rule:
    - Treat the largest non-black component as the main sprite and find its enclosed black cavities.
    - Match each loose fragment to a cavity by rotating the fragment until its shape equals the
      cavity shape and its maroon pixel lands 4-neighbor-adjacent to the cavity's maroon border lock.
    - Paste every matched fragment into its cavity, then replace all maroon pixels and all red arrow
      pixels with the main color.
    - Read the red L marker's bend: top-left or bottom-right bend means rotate the completed sprite
      clockwise by 90 degrees; top-right or bottom-left bend means rotate it counterclockwise by 90 degrees.
    - Output only the rotated completed main sprite, cropped to its non-black bounding box.

    Key insight:
    The maroon pixels are geometric locks that disambiguate which cavity each fragment belongs to and
    which rotation to use, while the red L marker is a second-stage instruction that rotates the whole
    repaired sprite after the local hole-filling step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    main_patch, main_bbox = _largest_component_patch(g)
    del main_bbox

    non_special = {int(v) for v in np.unique(main_patch) if int(v) not in {BG, ARROW, LOCK}}
    assert len(non_special) == 1, "expected exactly one main color"
    main_color = next(iter(non_special))
    rotation_k = _arrow_rotation_k(main_patch)

    holes = _extract_holes(main_patch)

    all_comps = _components(g != BG)
    main_area = int(np.sum(main_patch != BG))
    fragment_patches: list[np.ndarray] = []
    for comp in all_comps:
        if len(comp) == main_area:
            r0, c0, r1, c1 = _bbox(comp)
            if np.array_equal(g[r0 : r1 + 1, c0 : c1 + 1], main_patch):
                continue
        if len(comp) >= main_area:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        fragment_patches.append(g[r0 : r1 + 1, c0 : c1 + 1].copy())

    assert fragment_patches, "expected loose fragments"

    assignment = _assign_fragments(fragment_patches, holes)
    filled = main_patch.copy()
    for frag_idx, (hole_idx, rotated) in assignment.items():
        del frag_idx
        hole = holes[hole_idx]
        r0, c0, r1, c1 = hole["bbox"]
        target = filled[r0 : r1 + 1, c0 : c1 + 1]
        mask = rotated != BG
        assert np.array_equal(mask, hole["mask"])
        target[mask] = rotated[mask]

    filled[(filled == LOCK) | (filled == ARROW)] = int(main_color)
    rotated = np.rot90(filled, rotation_k)
    return _crop_non_bg(rotated)


def _generate_outer_mask() -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(10, 16)
        w = random.randint(10, 16)
        min_width = max(7, w - random.randint(3, 5))
        widths = [random.randint(min_width, w)]
        for _row in range(1, h):
            prev = widths[-1]
            lo = max(min_width, prev - 2)
            hi = min(w, prev + 2)
            widths.append(random.randint(lo, hi))

        lefts = [random.randint(0, w - widths[0])]
        ok = True
        for row in range(1, h):
            prev_left = lefts[-1]
            prev_right = prev_left + widths[row - 1] - 1
            width = widths[row]
            lo = max(0, prev_left - 2, prev_left - width + 2)
            hi = min(w - width, prev_left + 2, prev_right - 1)
            if lo > hi:
                ok = False
                break
            lefts.append(random.randint(lo, hi))
        if not ok:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, (left, width) in enumerate(zip(lefts, widths, strict=False)):
            mask[r, left : left + width] = True

        mask = _crop_non_bg(mask.astype(int)).astype(bool)
        interior = _erode4(mask)
        if int(interior.sum()) < 26:
            continue
        return mask
    return None


def _shape_positions(container: np.ndarray, shape: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    ch, cw = container.shape
    sh, sw = shape.shape
    shape_cells = [(int(r), int(c)) for r, c in np.argwhere(shape)]
    out: list[tuple[int, int]] = []
    for r0 in range(ch - sh + 1):
        for c0 in range(cw - sw + 1):
            ok = True
            for dr, dc in shape_cells:
                rr = r0 + dr
                cc = c0 + dc
                if not container[rr, cc]:
                    ok = False
                    break
                r1 = max(0, rr - 1)
                r2 = min(ch, rr + 2)
                c1 = max(0, cc - 1)
                c2 = min(cw, cc + 2)
                if np.any(blocked[r1:r2, c1:c2]):
                    ok = False
                    break
            if ok:
                out.append((int(r0), int(c0)))
    return out


def _place_holes(mask: np.ndarray, count: int) -> list[np.ndarray] | None:
    interior = _erode4(mask)
    blocked = np.zeros_like(mask, dtype=bool)
    holes: list[np.ndarray] = []
    for _ in range(count):
        placed = False
        bases = list(HOLE_SHAPES)
        random.shuffle(bases)
        for base in bases:
            rotations = list(range(4))
            random.shuffle(rotations)
            for k in rotations:
                shape = np.rot90(base, k)
                positions = _shape_positions(interior, shape, blocked)
                if not positions:
                    continue
                r0, c0 = random.choice(positions)
                hole = np.zeros_like(mask, dtype=bool)
                sh, sw = shape.shape
                hole[r0 : r0 + sh, c0 : c0 + sw] = shape
                holes.append(hole)
                blocked |= hole
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return holes


def _boundary_pairs(mask: np.ndarray, hole: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    hole_cells = {(int(r), int(c)) for r, c in np.argwhere(hole)}
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for hr, hc in hole_cells:
        for dr, dc in DIRS4:
            br = hr + dr
            bc = hc + dc
            if br < 0 or br >= mask.shape[0] or bc < 0 or bc >= mask.shape[1]:
                continue
            if not mask[br, bc] or (br, bc) in hole_cells:
                continue
            adj_holes = 0
            for dr2, dc2 in DIRS4:
                nr = br + dr2
                nc = bc + dc2
                if (nr, nc) in hole_cells:
                    adj_holes += 1
            if adj_holes == 1:
                pairs.append(((int(hr), int(hc)), (int(br), int(bc))))
    return pairs


def _arrow_positions(mask: np.ndarray, blocked: np.ndarray, pattern: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for r0 in range(h - 1):
        for c0 in range(w - 1):
            box = {(r0, c0), (r0 + 1, c0), (r0, c0 + 1), (r0 + 1, c0 + 1)}
            if any(not mask[r, c] or blocked[r, c] for r, c in box):
                continue
            arrow_cells = {(r0 + dr, c0 + dc) for dr, dc in pattern}
            ok = True
            for ar, ac in arrow_cells:
                for dr, dc in DIRS4:
                    nr = ar + dr
                    nc = ac + dc
                    if 0 <= nr < h and 0 <= nc < w and blocked[nr, nc]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                out.append((int(r0), int(c0)))
    return out


def _rect_can_place(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = reserved.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    r1 = max(0, r0 - pad)
    r2 = min(H, r0 + h + pad)
    c1 = max(0, c0 - pad)
    c2 = min(W, c0 + w + pad)
    return not bool(np.any(reserved[r1:r2, c1:c2]))


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    target = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    target[mask] = patch[mask]


def _build_fragment_from_hole(main_patch: np.ndarray, hole: dict, main_color: int) -> np.ndarray | None:
    assert len(hole["locks"]) == 1
    lock = hole["locks"][0]
    lock_targets = [cell for cell in hole["coords"] if _manhattan(cell, lock) == 1]
    if len(lock_targets) != 1:
        return None
    r0, c0, r1, c1 = hole["bbox"]
    fragment = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    fragment[hole["mask"]] = int(main_color)
    hr, hc = lock_targets[0]
    fragment[hr - r0, hc - c0] = LOCK
    return fragment


def _make_case() -> tuple[np.ndarray, np.ndarray] | None:
    main_color = random.choice(MAIN_COLORS)
    mask = _generate_outer_mask()
    if mask is None:
        return None

    fillable_count = random.randint(2, 4)
    extra_holes = random.randint(1, 2)
    holes = _place_holes(mask, fillable_count + extra_holes)
    if holes is None:
        return None

    fillable_indices = set(random.sample(range(len(holes)), fillable_count))
    lock_cells: set[tuple[int, int]] = set()
    hole_lock_map: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for index in fillable_indices:
        pairs = [pair for pair in _boundary_pairs(mask, holes[index]) if pair[1] not in lock_cells]
        if not pairs:
            return None
        chosen = random.choice(pairs)
        hole_lock_map[int(index)] = chosen
        lock_cells.add(chosen[1])

    blocked = np.zeros_like(mask, dtype=bool)
    for hole in holes:
        blocked |= hole
    for lr, lc in lock_cells:
        blocked[lr, lc] = True

    rotation_branch = random.choice((1, 3))
    pattern = random.choice(ARROW_PATTERNS[rotation_branch])
    positions = _arrow_positions(mask, blocked, pattern)
    if not positions:
        return None
    arrow_r0, arrow_c0 = random.choice(positions)
    arrow_cells = {(arrow_r0 + dr, arrow_c0 + dc) for dr, dc in pattern}

    main_patch = np.full(mask.shape, BG, dtype=int)
    main_patch[mask] = int(main_color)
    all_holes = np.zeros_like(mask, dtype=bool)
    permanent_holes = np.zeros_like(mask, dtype=bool)
    for index, hole in enumerate(holes):
        all_holes |= hole
        if index not in fillable_indices:
            permanent_holes |= hole
        main_patch[hole] = BG
    for _hole_cell, lock_cell in hole_lock_map.values():
        main_patch[lock_cell] = LOCK
    for r, c in arrow_cells:
        main_patch[r, c] = ARROW

    completed = np.full(mask.shape, BG, dtype=int)
    completed[mask] = int(main_color)
    completed[permanent_holes] = BG

    pre_rotate = random.randrange(4)
    main_patch = np.rot90(main_patch, pre_rotate)
    completed = np.rot90(completed, pre_rotate)

    arrow_k = _arrow_rotation_k(main_patch)
    expected = _crop_non_bg(np.rot90(completed, arrow_k))
    if np.array_equal(np.rot90(completed, 1), np.rot90(completed, 3)):
        return None

    extracted_holes = _extract_holes(main_patch)
    fillable_holes = [hole for hole in extracted_holes if len(hole["locks"]) == 1]
    if len(fillable_holes) != fillable_count:
        return None

    fragments = []
    for hole in fillable_holes:
        fragment = _build_fragment_from_hole(main_patch, hole, main_color)
        if fragment is None:
            return None
        fragments.append(np.rot90(fragment, random.randrange(4)))

    H = random.randint(max(main_patch.shape[0] + 6, 18), 30)
    W = random.randint(max(main_patch.shape[1] + 6, 18), 30)
    canvas = np.full((H, W), BG, dtype=int)
    reserved = np.zeros((H, W), dtype=bool)

    main_positions = [
        (r, c)
        for r in range(1, H - main_patch.shape[0])
        for c in range(1, W - main_patch.shape[1])
        if _rect_can_place(reserved, r, c, main_patch.shape[0], main_patch.shape[1], pad=0)
    ]
    if not main_positions:
        return None
    main_r0, main_c0 = random.choice(main_positions)
    _paste_patch(canvas, main_patch, main_r0, main_c0)
    reserved[main_r0 : main_r0 + main_patch.shape[0], main_c0 : main_c0 + main_patch.shape[1]] = True

    random.shuffle(fragments)
    for fragment in fragments:
        fh, fw = fragment.shape
        positions = [
            (r, c)
            for r in range(H - fh + 1)
            for c in range(W - fw + 1)
            if _rect_can_place(reserved, r, c, fh, fw, pad=1)
        ]
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        _paste_patch(canvas, fragment, r0, c0)
        reserved[r0 : r0 + fh, c0 : c0 + fw] = True

    return canvas, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sample = _make_case()
        if sample is None:
            continue
        inp, expected = sample
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
