from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_TO_ROT = {
    (-1, 0): 0,   # up
    (0, 1): 1,    # right
    (1, 0): 2,    # down
    (0, -1): 3,   # left
}


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _coords_to_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_from_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _mask_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    coords = np.array(sorted(cells), dtype=int)
    return _mask_from_coords(coords)


def _mask_to_cells(mask: np.ndarray, top: int = 0, left: int = 0) -> set[tuple[int, int]]:
    return {
        (top + int(r), left + int(c))
        for r, c in np.argwhere(mask)
    }


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(comp, dtype=int))
    return out


def _is_connected_cells(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _rotate_cw(mask: np.ndarray, quarter_turns: int) -> np.ndarray:
    q = int(quarter_turns) % 4
    return np.rot90(mask, k=(-q) % 4).astype(bool)


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    rots: list[np.ndarray] = []
    for q in range(4):
        rot = _rotate_cw(mask, q)
        if not any(np.array_equal(rot, old) for old in rots):
            rots.append(rot)
    return rots


def _shape_preserves_blue_connectivity(mask: np.ndarray) -> bool:
    local_blue = ~mask
    if not local_blue.any():
        return True
    h, w = mask.shape
    for comp in _components(local_blue):
        touches_outer_blue = (
            np.any(comp[:, 0] == 0)
            or np.any(comp[:, 1] == 0)
            or np.any(comp[:, 1] == w - 1)
        )
        if not touches_outer_blue:
            return False
    return True


def _extract_pockets(grid: np.ndarray) -> list[dict]:
    blue_mask = grid == BLUE
    blue_components = _components(blue_mask)
    assert len(blue_components) == 1, "expected exactly one blue structure"
    r0, c0, r1, c1 = _bbox_from_coords(blue_components[0])
    crop = blue_mask[r0 : r1 + 1, c0 : c1 + 1]
    pockets = []
    for comp in _components(~crop):
        if np.any(comp[:, 0] == crop.shape[0] - 1):
            gr = comp[:, 0] + r0
            gc = comp[:, 1] + c0
            coords = np.column_stack([gr, gc]).astype(int)
            pockets.append(
                {
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "mask": _mask_from_coords(coords),
                }
            )
    assert pockets, "expected at least one pocket"
    return pockets


def _candidate_components(grid: np.ndarray) -> list[np.ndarray]:
    return _components(grid == GREY)


def _rod_is_external(
    rod_cells: set[tuple[int, int]],
    body_cells: set[tuple[int, int]],
    rod_dir: tuple[int, int],
) -> bool:
    if rod_dir == (-1, 0):
        return max(r for r, _ in rod_cells) < min(r for r, _ in body_cells)
    if rod_dir == (1, 0):
        return min(r for r, _ in rod_cells) > max(r for r, _ in body_cells)
    if rod_dir == (0, -1):
        return max(c for _, c in rod_cells) < min(c for _, c in body_cells)
    if rod_dir == (0, 1):
        return min(c for _, c in rod_cells) > max(c for _, c in body_cells)
    return False


def _parse_candidate(coords: np.ndarray) -> dict:
    cells = _coords_to_set(coords)
    candidates = []

    for tip in cells:
        tip_neighbors = [n for n in _neighbors4(*tip) if n in cells]
        if len(tip_neighbors) != 1:
            continue

        body_step = (tip_neighbors[0][0] - tip[0], tip_neighbors[0][1] - tip[1])
        prev = tip
        cur = tip_neighbors[0]
        rod = [tip]
        valid = True

        while True:
            forward = (cur[0] + body_step[0], cur[1] + body_step[1])
            cur_neighbors = [n for n in _neighbors4(*cur) if n in cells]
            allowed = {prev}
            if forward in cells:
                allowed.add(forward)

            if forward in cells and set(cur_neighbors).issubset(allowed):
                rod.append(cur)
                prev = cur
                cur = forward
                if len(rod) > 3:
                    valid = False
                    break
                continue
            break

        if not valid or not (1 <= len(rod) <= 3):
            continue

        rod_cells = set(rod)
        body_cells = cells - rod_cells
        if len(body_cells) < 3 or not _is_connected_cells(body_cells):
            continue

        rod_dir = (-body_step[0], -body_step[1])
        if rod_dir not in DIR_TO_ROT:
            continue
        if not _rod_is_external(rod_cells, body_cells, rod_dir):
            continue

        body_coords = np.array(sorted(body_cells), dtype=int)
        candidates.append(
            {
                "body_coords": body_coords,
                "body_mask": _mask_from_coords(body_coords),
                "rod_coords": np.array(sorted(rod_cells), dtype=int),
                "rotation_q": DIR_TO_ROT[rod_dir],
                "direction": rod_dir,
            }
        )

    unique = []
    seen = set()
    for cand in candidates:
        key = (
            cand["rotation_q"],
            tuple(map(tuple, cand["body_coords"])),
            tuple(map(tuple, cand["rod_coords"])),
        )
        if key not in seen:
            seen.add(key)
            unique.append(cand)

    assert len(unique) == 1, "each gray component must have exactly one valid rod interpretation"
    return unique[0]


def _match_pocket_index(rotated_body: np.ndarray, pockets: list[dict]) -> int | None:
    hits = [i for i, pocket in enumerate(pockets) if np.array_equal(rotated_body, pocket["mask"])]
    assert len(hits) <= 1, "one rotated body must not exactly equal multiple pockets"
    return hits[0] if hits else None


def _analyze_grid(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    pockets = _extract_pockets(grid)
    candidates = []
    for coords in _candidate_components(grid):
        parsed = _parse_candidate(coords)
        rotated = _rotate_cw(parsed["body_mask"], parsed["rotation_q"])
        directed_match = _match_pocket_index(rotated, pockets)
        any_rotation_matches = sorted(
            {
                idx
                for q in range(4)
                for idx, pocket in enumerate(pockets)
                if np.array_equal(_rotate_cw(parsed["body_mask"], q), pocket["mask"])
            }
        )
        candidates.append(
            {
                "coords": coords,
                "body_mask": parsed["body_mask"],
                "rotation_q": parsed["rotation_q"],
                "direction": parsed["direction"],
                "directed_match": directed_match,
                "any_rotation_matches": any_rotation_matches,
            }
        )
    return pockets, candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor rod-marked gray candidates red when their rod-directed rotation makes them exactly fit a bottom-opening pocket in the blue structure.

    Input structure:
    - A black background contains one upper blue 4-connected structure and several lower gray 4-connected candidate objects.
    - The blue structure occupies a short wide band; inside its bbox, some black 4-connected cavities touch the bbox bottom edge and serve as pockets.
    - Each gray candidate consists of an irregular gray body plus one straight 1-cell-wide gray rod extending 1-3 cells outward in one cardinal direction.

    Rule:
    - Read each rod direction as a rotation command for the body only: up means 0 degrees, right means 90 degrees clockwise, down means 180 degrees, and left means 270 degrees clockwise.
    - Remove the rod conceptually, rotate the body around its own tight bbox by that angle, and test whether the rotated body exactly matches one pocket shape.
    - If it matches a pocket, recolor the entire original gray candidate, including its rod, to red in place; otherwise leave it gray.

    Key insight:
    - The rod is not a movement path but a direction code that uniquely selects the body's rotation before shape matching against the blue structure's pockets.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    pockets, candidates = _analyze_grid(g)
    assert pockets and candidates

    out = g.copy()
    for cand in candidates:
        if cand["directed_match"] is not None:
            out[cand["coords"][:, 0], cand["coords"][:, 1]] = RED
    return out


SHAPE_ATTEMPTS = 120
GEN_ATTEMPTS = 400
POCKET_LIBRARY = [
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
]


def _random_connected_shape(max_h: int, max_w: int, area_min: int = 3, area_max: int = 6) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(2, max_h)
        w = random.randint(2, max_w)
        area = random.randint(area_min, min(area_max, h * w))
        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True
        cells = {seed}
        frontier = set()
        frontier.update({n for n in _neighbors4(*seed) if 0 <= n[0] < h and 0 <= n[1] < w})
        while len(cells) < area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            mask[cell] = True
            cells.add(cell)
            for nb in _neighbors4(*cell):
                if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in cells:
                    frontier.add(nb)
        if len(cells) != area:
            continue
        coords = np.argwhere(mask)
        norm = _mask_from_coords(coords)
        if norm.shape[0] < 2 or norm.shape[1] < 2:
            continue
        if len(_unique_rotations(norm)) != 4:
            continue
        if not _shape_preserves_blue_connectivity(norm):
            continue
        return norm
    return None


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(map(tuple, np.argwhere(mask)))


def _rotation_family_keys(mask: np.ndarray) -> set[tuple[tuple[int, int], ...]]:
    return {_shape_key(_rotate_cw(mask, q)) for q in range(4)}


def _choose_pocket_shapes(band_h: int, count: int) -> list[np.ndarray] | None:
    candidates = [mask for mask in POCKET_LIBRARY if mask.shape[0] <= band_h - 1]
    random.shuffle(candidates)
    shapes: list[np.ndarray] = []
    families: list[set[tuple[tuple[int, int], ...]]] = []
    for mask in candidates:
        fam = _rotation_family_keys(mask)
        if any(fam & old for old in families):
            continue
        shapes.append(mask.copy())
        families.append(fam)
        if len(shapes) == count:
            return shapes
    return None


def _add_non_pocket_notch(blue: np.ndarray, left: int, top: int, band_w: int, band_h: int, pocket_boxes: list[tuple[int, int, int, int]]) -> None:
    if random.random() >= 0.6:
        return
    for _ in range(40):
        notch_h = 1
        notch_w = random.randint(1, 2)
        c0 = random.randint(left + 1, left + band_w - notch_w - 2)
        r0 = top
        r1 = top + notch_h - 1
        c1 = c0 + notch_w - 1
        ok = True
        for pr0, pc0, pr1, pc1 in pocket_boxes:
            if not (c1 < pc0 - 1 or pc1 + 1 < c0 or r1 < pr0 - 1 or pr1 + 1 < r0):
                ok = False
                break
        if ok:
            blue[r0 : r1 + 1, c0 : c1 + 1] = BLACK
            return


def _build_blue_structure(h: int, w: int) -> tuple[np.ndarray, list[np.ndarray]] | None:
    band_h = 4
    pocket_count = random.choice([2, 3])
    shapes = _choose_pocket_shapes(band_h, pocket_count)
    if shapes is None:
        return None

    widths = [mask.shape[1] for mask in shapes]
    gaps = [random.randint(1, 2) for _ in range(pocket_count - 1)]
    side_margin = random.randint(1, 2)
    band_w = side_margin * 2 + sum(widths) + sum(gaps)
    if band_w > w - 2:
        return None

    left = random.randint(1, w - band_w - 1)
    top = random.randint(1, 3)
    if top + band_h + 6 >= h:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[top : top + band_h, left : left + band_w] = BLUE

    x = left + side_margin
    pocket_boxes = []
    for idx, mask in enumerate(shapes):
        ph, pw = mask.shape
        r0 = top + band_h - ph
        c0 = x
        coords = np.argwhere(mask)
        for rr, cc in coords:
            grid[r0 + rr, c0 + cc] = BLACK
        pocket_boxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
        x += pw
        if idx < len(gaps):
            x += gaps[idx]

    _add_non_pocket_notch(grid, left, top, band_w, band_h, pocket_boxes)
    if len(_components(grid == BLUE)) != 1:
        return None
    pockets = _extract_pockets(grid)
    if len(pockets) != pocket_count:
        return None
    return grid, [p["mask"] for p in pockets]


def _boundary_cells(mask: np.ndarray, direction: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    dr, dc = direction
    cells = []
    for r, c in np.argwhere(mask):
        nr = int(r) + dr
        nc = int(c) + dc
        if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
            cells.append((int(r), int(c)))
    return cells


def _make_candidate_patch(body_mask: np.ndarray, rod_dir: tuple[int, int], rod_len: int, attach: tuple[int, int]) -> np.ndarray:
    bh, bw = body_mask.shape
    if rod_dir == (-1, 0):
        patch = np.zeros((bh + rod_len, bw), dtype=int)
        patch[rod_len:, :] = body_mask.astype(int) * GREY
        ar, ac = attach
        for k in range(rod_len):
            patch[rod_len - 1 - k, ac] = GREY
    elif rod_dir == (1, 0):
        patch = np.zeros((bh + rod_len, bw), dtype=int)
        patch[:bh, :] = body_mask.astype(int) * GREY
        ar, ac = attach
        for k in range(rod_len):
            patch[bh + k, ac] = GREY
    elif rod_dir == (0, -1):
        patch = np.zeros((bh, bw + rod_len), dtype=int)
        patch[:, rod_len:] = body_mask.astype(int) * GREY
        ar, ac = attach
        for k in range(rod_len):
            patch[ar, rod_len - 1 - k] = GREY
    else:
        patch = np.zeros((bh, bw + rod_len), dtype=int)
        patch[:, :bw] = body_mask.astype(int) * GREY
        ar, ac = attach
        for k in range(rod_len):
            patch[ar, bw + k] = GREY
    return patch


def _candidate_from_body(body_mask: np.ndarray, rotation_q: int) -> np.ndarray | None:
    rod_dir = [(-1, 0), (0, 1), (1, 0), (0, -1)][rotation_q]
    boundary = _boundary_cells(body_mask, rod_dir)
    random.shuffle(boundary)
    rod_lens = [1, 2, 3]
    random.shuffle(rod_lens)

    for attach in boundary:
        for rod_len in rod_lens:
            patch = _make_candidate_patch(body_mask, rod_dir, rod_len, attach)
            coords = np.argwhere(patch == GREY)
            try:
                parsed = _parse_candidate(coords)
            except AssertionError:
                continue
            if parsed["rotation_q"] != rotation_q:
                continue
            if not np.array_equal(parsed["body_mask"], body_mask):
                continue
            return patch
    return None


def _place_patch(grid: np.ndarray, patch: np.ndarray, top_min: int) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    rows = list(range(top_min, h - ph + 1))
    cols = list(range(1, w - pw - 1 + 1))
    random.shuffle(rows)
    random.shuffle(cols)
    mask = patch == GREY

    for r0 in rows:
        for c0 in cols:
            ok = True
            for rr, cc in np.argwhere(mask):
                gr = r0 + int(rr)
                gc = c0 + int(cc)
                if grid[gr, gc] != BLACK:
                    ok = False
                    break
                for nr, nc in _neighbors4(gr, gc):
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] in (GREY, BLUE):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                for rr, cc in np.argwhere(mask):
                    grid[r0 + int(rr), c0 + int(cc)] = GREY
                return True
    return False


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        pockets, candidates = _analyze_grid(grid)
    except AssertionError:
        return False

    if len(pockets) not in (2, 3):
        return False
    if len(candidates) != len(pockets) + 1:
        return False

    matched = [cand for cand in candidates if cand["directed_match"] is not None]
    unmatched = [cand for cand in candidates if cand["directed_match"] is None]
    if len(unmatched) < 1:
        return False

    hit_counts = [0] * len(pockets)
    for cand in matched:
        hit_counts[cand["directed_match"]] += 1
    if any(count != 1 for count in hit_counts):
        return False

    if not any(cand["rotation_q"] != 0 and cand["directed_match"] is not None for cand in candidates):
        return False

    if not any(cand["directed_match"] is None and cand["any_rotation_matches"] for cand in candidates):
        return False

    out = main(grid)
    red_components = _components(out == RED)
    if len(red_components) != len(matched):
        return False

    for cand in matched:
        if not np.all(out[cand["coords"][:, 0], cand["coords"][:, 1]] == RED):
            return False
    for cand in unmatched:
        if not np.all(out[cand["coords"][:, 0], cand["coords"][:, 1]] == GREY):
            return False

    return True


def _build_instance() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    built = _build_blue_structure(h, w)
    if built is None:
        return None
    grid, pockets = built

    matched_patches = []
    nonzero_used = False
    for pocket_mask in pockets:
        choices = [0, 1, 2, 3]
        random.shuffle(choices)
        if not nonzero_used and all(q == 0 for q in choices[:-1]):
            choices = [1, 2, 3, 0]
        patch = None
        chosen_q = None
        for q in choices:
            body = np.rot90(pocket_mask, k=q).astype(bool)
            patch = _candidate_from_body(body, q)
            if patch is not None:
                chosen_q = q
                break
        if patch is None or chosen_q is None:
            return None
        nonzero_used = nonzero_used or chosen_q != 0
        matched_patches.append(patch)

    decoy_patch = None
    pocket_order = list(range(len(pockets)))
    random.shuffle(pocket_order)
    for pocket_idx in pocket_order:
        pocket_mask = pockets[pocket_idx]
        match_qs = [0, 1, 2, 3]
        random.shuffle(match_qs)
        for match_q in match_qs:
            body = np.rot90(pocket_mask, k=match_q).astype(bool)
            bad_qs = [q for q in range(4) if q != match_q]
            random.shuffle(bad_qs)
            for bad_q in bad_qs:
                rotated = _rotate_cw(body, bad_q)
                if any(np.array_equal(rotated, other) for other in pockets):
                    continue
                decoy_patch = _candidate_from_body(body, bad_q)
                if decoy_patch is not None:
                    break
            if decoy_patch is not None:
                break
        if decoy_patch is not None:
            break
    if decoy_patch is None:
        return None

    patches = matched_patches + [decoy_patch]
    random.shuffle(patches)
    top_min = max(r for r, _ in np.argwhere(grid == BLUE)) + 3
    for patch in patches:
        if not _place_patch(grid, patch, top_min):
            return None

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
