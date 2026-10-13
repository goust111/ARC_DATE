from __future__ import annotations

import random

import numpy as np


BG = 0
MASK_COLOR = 5
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
GEN_RETRIES = 400
GRID_RANGE = (18, 28)
MASK_H_RANGE = (5, 8)
MASK_W_RANGE = (5, 8)
SHAPE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _rotate_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h, _ = _shape_dims(shape)
    return _normalize({(c, h - 1 - r) for r, c in shape})


def _flip_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    _, w = _shape_dims(shape)
    return _normalize({(r, w - 1 - c) for r, c in shape})


def _translate(shape: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in shape]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(sorted(comp))
    return out


def _connected(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> bool:
    pts = set(coords)
    if not pts:
        return False
    stack = [next(iter(pts))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(pts)


def _dilate(mask: np.ndarray, dirs: tuple[tuple[int, int], ...]) -> np.ndarray:
    out = mask.copy()
    for dr, dc in dirs:
        src_r0 = max(0, -dr)
        src_r1 = min(mask.shape[0], mask.shape[0] - dr)
        src_c0 = max(0, -dc)
        src_c1 = min(mask.shape[1], mask.shape[1] - dc)
        dst_r0 = src_r0 + dr
        dst_r1 = src_r1 + dr
        dst_c0 = src_c0 + dc
        dst_c1 = src_c1 + dc
        out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _touches_mask(coords: list[tuple[int, int]] | set[tuple[int, int]], mask: np.ndarray) -> bool:
    for r, c in coords:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]):
                return True
    return False


def _find_mask_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    if bool(np.any(g == MASK_COLOR)):
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == MASK_COLOR)]
        r0, c0, r1, c1 = _bbox(coords)
        if np.all(g[r0 : r1 + 1, c0 : c1 + 1] == MASK_COLOR):
            return int(r0), int(c0), int(r1), int(c1)

    candidates = []
    for color in sorted(set(map(int, g.ravel())) - {BG}):
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        if not coords:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color):
            candidates.append((len(coords), (int(r0), int(c0), int(r1), int(c1))))
    assert len(candidates) == 1, "expected a unique solid rectangular mask"
    return candidates[0][1]


def _recover_hidden_cells(
    visible: set[tuple[int, int]],
    templates: list[tuple[tuple[int, int], ...]],
    mask: np.ndarray,
    grid_shape: tuple[int, int],
) -> set[tuple[int, int]] | None:
    if not visible or not templates:
        return None

    visible_list = sorted(visible)
    hidden_variants: set[tuple[tuple[int, int], ...]] = set()

    for template in templates:
        for vr, vc in visible_list:
            for tr, tc in template:
                top = vr - tr
                left = vc - tc
                placed = []
                valid = True
                for sr, sc in template:
                    rr, cc = top + sr, left + sc
                    if not _in_bounds(grid_shape, rr, cc):
                        valid = False
                        break
                    placed.append((int(rr), int(cc)))
                if not valid:
                    continue

                outside = {cell for cell in placed if not bool(mask[cell])}
                if outside != visible:
                    continue

                hidden = tuple(sorted(cell for cell in placed if bool(mask[cell])))
                if hidden:
                    hidden_variants.add(hidden)

    if len(hidden_variants) != 1:
        return None
    return set(next(iter(hidden_variants)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the colored pixels hidden beneath the grey mask by matching each truncated pattern to a full translated copy.

    Input structure:
    - The background is black.
    - Several monochrome non-black pattern copies are scattered on the grid, and copies of the same pattern keep the same shape and orientation.
    - One solid grey rectangle overlays the scene and hides part of one or more patterns.

    Rule:
    - Find the grey rectangle and treat it as an occluder rather than as a foreground object.
    - For each color that has visible pixels touching the mask boundary, gather the visible remainder and compare it with complete same-color copies elsewhere in the grid.
    - Translate the matching complete copy so that all of its pixels outside the mask exactly match the visible remainder; any aligned pixels inside the mask are the hidden cells to restore.
    - Output only the mask rectangle's bounding box, filling restored hidden cells with their original colors and all untouched mask cells with black.

    Key insight:
    - The grey block does not define the output by itself; it only covers pixels whose missing values can be inferred from an unoccluded translated replica of the same pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mr0, mc0, mr1, mc1 = _find_mask_bbox(g)
    mask = np.zeros_like(g, dtype=bool)
    mask[mr0 : mr1 + 1, mc0 : mc1 + 1] = True
    out = np.full((mr1 - mr0 + 1, mc1 - mc0 + 1), BG, dtype=int)

    colors = sorted(set(map(int, g.ravel())) - {BG, MASK_COLOR})
    for color in colors:
        comps = _components(g == color)
        touching = [comp for comp in comps if _touches_mask(comp, mask)]
        if not touching:
            continue

        visible: set[tuple[int, int]] = set()
        for comp in touching:
            visible.update(comp)

        templates: list[tuple[tuple[int, int], ...]] = []
        seen: set[tuple[tuple[int, int], ...]] = set()
        for comp in comps:
            if _touches_mask(comp, mask):
                continue
            key = _normalize(comp)
            if key not in seen:
                seen.add(key)
                templates.append(key)

        hidden = _recover_hidden_cells(visible, templates, mask, g.shape)
        assert hidden is not None, f"failed to match occluded pattern for color {color}"
        for r, c in hidden:
            out[r - mr0, c - mc0] = int(color)

    return out


def _shape_l() -> tuple[tuple[int, int], ...]:
    h = random.randint(3, 5)
    w = random.randint(3, 5)
    coords = {(r, 0) for r in range(h)}
    coords.update((h - 1, c) for c in range(w))
    return _normalize(coords)


def _shape_t() -> tuple[tuple[int, int], ...]:
    h = random.randint(3, 5)
    w = random.randint(3, 5)
    stem = random.randint(1, w - 2)
    coords = {(0, c) for c in range(w)}
    coords.update((r, stem) for r in range(1, h))
    return _normalize(coords)


def _shape_c() -> tuple[tuple[int, int], ...]:
    h = random.randint(4, 5)
    w = random.randint(4, 5)
    coords = {(0, c) for c in range(w)}
    coords.update((h - 1, c) for c in range(w))
    coords.update((r, 0) for r in range(h))
    return _normalize(coords)


def _shape_f() -> tuple[tuple[int, int], ...]:
    h = random.randint(4, 5)
    w = random.randint(4, 5)
    mid = random.randint(1, h - 2)
    arm_w = random.randint(2, w - 1)
    coords = {(r, 0) for r in range(h)}
    coords.update((0, c) for c in range(w))
    coords.update((mid, c) for c in range(arm_w))
    return _normalize(coords)


def _shape_z() -> tuple[tuple[int, int], ...]:
    w = random.randint(4, 6)
    coords = {(0, c) for c in range(w - 1)}
    coords.add((1, w - 2))
    coords.update((2, c) for c in range(1, w))
    if random.random() < 0.5:
        coords.add((1, 1))
    return _normalize(coords)


def _random_shape() -> tuple[tuple[int, int], ...]:
    shape = random.choice((_shape_l, _shape_t, _shape_c, _shape_f, _shape_z))()
    if random.random() < 0.5:
        shape = _flip_shape(shape)
    for _ in range(random.randrange(4)):
        shape = _rotate_shape(shape)
    return _normalize(shape)


def _pick_mask_position(h: int, w: int) -> tuple[int, int, int, int] | None:
    mh = random.randint(*MASK_H_RANGE)
    mw = random.randint(*MASK_W_RANGE)
    row_choices = list(range(4, h - mh - 3))
    col_choices = list(range(4, w - mw - 3))
    if not row_choices or not col_choices:
        return None
    r0 = random.choice(row_choices)
    c0 = random.choice(col_choices)
    return int(r0), int(c0), int(r0 + mh - 1), int(c0 + mw - 1)


def _candidate_ok(cells: list[tuple[int, int]], blocked: np.ndarray) -> bool:
    return all(not bool(blocked[r, c]) for r, c in cells)


def _full_copy_candidates(
    shape: tuple[tuple[int, int], ...],
    occupied: np.ndarray,
    forbidden: np.ndarray,
) -> list[list[tuple[int, int]]]:
    h, w = _shape_dims(shape)
    occ_block = _dilate(occupied, DIRS8)
    out: list[list[tuple[int, int]]] = []
    for top in range(occupied.shape[0] - h + 1):
        for left in range(occupied.shape[1] - w + 1):
            cells = _translate(shape, top, left)
            if _candidate_ok(cells, occ_block) and _candidate_ok(cells, forbidden):
                out.append(cells)
    return out


def _inside_mask(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _side_visible_ok(
    visible: list[tuple[int, int]],
    side: str,
    bbox: tuple[int, int, int, int],
) -> bool:
    r0, c0, r1, c1 = bbox
    if side == "left":
        return all(c < c0 for _, c in visible)
    if side == "right":
        return all(c > c1 for _, c in visible)
    if side == "top":
        return all(r < r0 for r, _ in visible)
    return all(r > r1 for r, _ in visible)


def _occluded_copy_candidates(
    shape: tuple[tuple[int, int], ...],
    side: str,
    mask_bbox: tuple[int, int, int, int],
    occupied: np.ndarray,
) -> list[list[tuple[int, int]]]:
    h, w = _shape_dims(shape)
    mr0, mc0, mr1, mc1 = mask_bbox
    occ_block = _dilate(occupied, DIRS8)
    mask = np.zeros_like(occupied, dtype=bool)
    mask[mr0 : mr1 + 1, mc0 : mc1 + 1] = True
    out: list[list[tuple[int, int]]] = []

    if side in ("left", "right"):
        top_range = range(mr0 - h + 1, mr1 + 1)
        if side == "left":
            left_range = range(mc0 - w + 1, mc0 + 1)
        else:
            left_range = range(mc1 - w + 1, mc1 + 1)
    else:
        left_range = range(mc0 - w + 1, mc1 + 1)
        if side == "top":
            top_range = range(mr0 - h + 1, mr0 + 1)
        else:
            top_range = range(mr1 - h + 1, mr1 + 1)

    for top in top_range:
        for left in left_range:
            cells = _translate(shape, top, left)
            if any(not _in_bounds(occupied.shape, r, c) for r, c in cells):
                continue
            if not _candidate_ok(cells, occ_block):
                continue

            hidden = [cell for cell in cells if _inside_mask(cell, mask_bbox)]
            visible = [cell for cell in cells if not _inside_mask(cell, mask_bbox)]
            if not hidden or len(visible) < 2:
                continue
            if not _side_visible_ok(visible, side, mask_bbox):
                continue
            if not _connected(visible):
                continue
            if not _touches_mask(visible, mask):
                continue

            recovered = _recover_hidden_cells(set(visible), [shape], mask, occupied.shape)
            if recovered != set(hidden):
                continue
            out.append(cells)
    return out


def _paint(scene: np.ndarray, occupied: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        scene[r, c] = int(color)
        occupied[r, c] = True


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    mask_bbox = _pick_mask_position(h, w)
    if mask_bbox is None:
        return None
    mr0, mc0, mr1, mc1 = mask_bbox

    n_occ = random.choice((1, 2))
    n_families = random.randint(max(2, n_occ + 1), 4)
    colors = random.sample(list(SHAPE_COLORS), n_families)
    side_pool = ["left", "right", "top", "bottom"]
    occ_sides = random.sample(side_pool, n_occ)

    scene = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    mask = np.zeros((h, w), dtype=bool)
    mask[mr0 : mr1 + 1, mc0 : mc1 + 1] = True
    forbidden_full = _dilate(mask, DIRS4)

    families = []
    for idx, color in enumerate(colors):
        families.append(
            {
                "color": int(color),
                "shape": _random_shape(),
                "occluded": idx < n_occ,
                "side": occ_sides[idx] if idx < n_occ else None,
                "copies": random.choice((2, 3)),
            }
        )

    random.shuffle(families)
    families.sort(key=lambda item: (not item["occluded"], -len(item["shape"])))

    for family in families:
        if family["occluded"]:
            occluded_candidates = _occluded_copy_candidates(
                family["shape"], family["side"], mask_bbox, occupied
            )
            if not occluded_candidates:
                return None
            cells = random.choice(occluded_candidates)
            _paint(scene, occupied, cells, family["color"])
            family["placed"] = 1
        else:
            family["placed"] = 0

        needed_full = family["copies"] - family["placed"]
        for _ in range(needed_full):
            full_candidates = _full_copy_candidates(family["shape"], occupied, forbidden_full)
            if not full_candidates:
                return None
            cells = random.choice(full_candidates)
            _paint(scene, occupied, cells, family["color"])

    expected = scene[mr0 : mr1 + 1, mc0 : mc1 + 1].copy()
    if not bool(np.any(expected != BG)):
        return None
    if not bool(np.any(expected == BG)):
        return None

    inp = scene.copy()
    inp[mr0 : mr1 + 1, mc0 : mc1 + 1] = MASK_COLOR

    try:
        produced = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid occlusion-reconstruction ARC input")
