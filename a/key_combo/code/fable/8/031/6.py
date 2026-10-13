from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def get_bg(grid: np.ndarray) -> int:
    """Background color = most common value in the grid."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.ravel()))
    if exclude:
        colors -= set(map(int, exclude))
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int, shape: tuple[int, int]) -> list[Coord]:
    h, w = shape
    out: list[Coord] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, mask.shape):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in colors_present(grid, exclude={bg}):
        for coords in _mask_components(grid == color):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _normalize_coords(coords: np.ndarray) -> tuple[Coord, ...]:
    r0, c0, _, _ = bbox_coords(coords)
    shifted = [(int(r - r0), int(c - c0)) for r, c in coords]
    return tuple(sorted(shifted))


def _mask_to_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask).astype(int)


def _mask_to_local_cells(mask: np.ndarray) -> frozenset[Coord]:
    return frozenset((int(r), int(c)) for r, c in np.argwhere(mask))


def _is_connected(mask: np.ndarray) -> bool:
    comps = _mask_components(mask)
    return len(comps) == 1


def _piece_placements(piece_mask: np.ndarray, cavity_mask: np.ndarray) -> list[frozenset[Coord]]:
    piece_coords = _mask_to_coords(piece_mask)
    cavity_coords = _mask_to_coords(cavity_mask)
    h, w = cavity_mask.shape
    cavity_set = _mask_to_local_cells(cavity_mask)
    placements: set[frozenset[Coord]] = set()
    for pr, pc in piece_coords:
        for cr, cc in cavity_coords:
            dr = int(cr) - int(pr)
            dc = int(cc) - int(pc)
            placed: set[Coord] = set()
            ok = True
            for rr, cc2 in piece_coords:
                nr = int(rr) + dr
                nc = int(cc2) + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w or (nr, nc) not in cavity_set:
                    ok = False
                    break
                placed.add((nr, nc))
            if ok:
                placements.add(frozenset(placed))
    return sorted(placements, key=lambda s: (len(s), tuple(sorted(s))))


def _solve_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)

    comps = _find_components(g, bg)
    assert comps, "expected foreground objects"
    largest_area = max(comp["area"] for comp in comps)
    largest = [comp for comp in comps if comp["area"] == largest_area]
    assert len(largest) == 1, "host rectangle component must be uniquely largest"
    host = largest[0]
    base = int(host["color"])
    host_bbox = tuple(map(int, host["bbox"]))
    host_patch = crop_bbox(g, host_bbox)
    allowed_inside = {bg, base}
    assert set(map(int, host_patch.ravel())) <= allowed_inside, "host bbox must contain only base color and cavity"
    cavity_mask = host_patch == bg
    assert cavity_mask.any(), "expected a non-empty cavity"
    assert _is_connected(host_patch == base), "host remainder must stay 4-connected"

    masked = g.copy()
    r0, c0, r1, c1 = host_bbox
    masked[r0 : r1 + 1, c0 : c1 + 1] = bg
    fragments = _find_components(masked, bg)
    assert len(fragments) >= 2, "expected fragments outside the host"
    fragment_colors = [int(fragment["color"]) for fragment in fragments]
    assert len(set(fragment_colors)) == len(fragment_colors), "fragments must use distinct colors"

    cavity_area = int(cavity_mask.sum())
    cavity_cells = _mask_to_local_cells(cavity_mask)
    for fragment in fragments:
        fr0, fc0, fr1, fc1 = map(int, fragment["bbox"])
        patch = crop_bbox(masked, (fr0, fc0, fr1, fc1))
        fragment["local_mask"] = patch == int(fragment["color"])
        fragment["placements"] = _piece_placements(fragment["local_mask"], cavity_mask)

    fits: list[tuple[int, int, frozenset[Coord], frozenset[Coord]]] = []
    for i, left in enumerate(fragments):
        for j in range(i + 1, len(fragments)):
            right = fragments[j]
            if int(left["area"]) + int(right["area"]) != cavity_area:
                continue
            for placed_left in left["placements"]:
                for placed_right in right["placements"]:
                    if placed_left.isdisjoint(placed_right) and placed_left | placed_right == cavity_cells:
                        fits.append((i, j, placed_left, placed_right))

    assert len(fits) == 1, "expected exactly one translated fragment pair that fills the cavity"
    pair_i, pair_j, _, _ = fits[0]

    return {
        "bg": bg,
        "base": base,
        "host_bbox": host_bbox,
        "cavity_mask": cavity_mask,
        "cavity_area": cavity_area,
        "fragments": fragments,
        "pair": (pair_i, pair_j),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill one rectangular host cavity with the unique pair of translated fragments and erase those source fragments.

    Input structure:
    - A dominant background surrounds one large non-background host object whose bounding box is a solid rectangle except for one background-colored cavity.
    - Several small colored fragments lie elsewhere, each as its own 4-connected object with a distinct non-host color.
    - Exactly one pair of fragments can be translated, without rotation or reflection, so their union matches the cavity exactly.

    Rule:
    - Identify the uniquely largest component as the host and take the missing background cells inside its bounding box as the cavity.
    - Search all fragment pairs and keep the unique pair whose translated shapes fit inside the cavity with no overlap and exact full coverage.
    - Recolor the cavity with the host color and erase only that matched fragment pair to the background; keep all other fragments unchanged.

    Key insight:
    - Area matches alone are insufficient because one single fragment and one distractor pair have the same total area as the cavity; only an exact two-piece translation fit identifies the removable pair.
    """
    g = np.asarray(grid, dtype=int)
    info = _solve_instance(g)
    out = g.copy()
    bg = int(info["bg"])
    base = int(info["base"])
    r0, c0, r1, c1 = info["host_bbox"]
    cavity_mask = np.asarray(info["cavity_mask"], dtype=bool)
    out[r0 : r1 + 1, c0 : c1 + 1][cavity_mask] = base

    pair_i, pair_j = info["pair"]
    for idx in (pair_i, pair_j):
        coords = np.asarray(info["fragments"][idx]["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = bg
    return out


# --------------- generator ---------------

GRID_MIN = 22
GRID_MAX = 30
HOST_H_RANGE = (8, 13)
HOST_W_RANGE = (9, 14)
CAVITY_BOX_MIN = 4
CAVITY_BOX_MAX = 7
CAVITY_AREA_MIN = 6
CAVITY_AREA_MAX = 11
GEN_ATTEMPTS = 600


def _row_col_signature(mask: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(int(v) for v in mask.sum(axis=1) if int(v) > 0),
        tuple(int(v) for v in mask.sum(axis=0) if int(v) > 0),
    )


def _random_connected_mask(
    h: int,
    w: int,
    area: int,
    *,
    allowed: np.ndarray | None = None,
    seed_choices: list[Coord] | None = None,
) -> np.ndarray | None:
    if allowed is None:
        allowed = np.ones((h, w), dtype=bool)
    allowed = np.asarray(allowed, dtype=bool)
    if int(allowed.sum()) < area:
        return None
    all_cells = [(int(r), int(c)) for r, c in np.argwhere(allowed)]
    if seed_choices is None:
        seed_choices = all_cells
    seed_choices = [cell for cell in seed_choices if allowed[cell]]
    if not seed_choices:
        return None
    mask = np.zeros((h, w), dtype=bool)
    start = random.choice(seed_choices)
    mask[start] = True
    while int(mask.sum()) < area:
        frontier: set[Coord] = set()
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c), mask.shape):
                if allowed[nr, nc] and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        mask[random.choice(list(frontier))] = True
    return mask


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = _mask_to_coords(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _touch_sides(mask: np.ndarray) -> set[str]:
    h, w = mask.shape
    touched: set[str] = set()
    if bool(mask[0, :].any()):
        touched.add("top")
    if bool(mask[h - 1, :].any()):
        touched.add("bottom")
    if bool(mask[:, 0].any()):
        touched.add("left")
    if bool(mask[:, w - 1].any()):
        touched.add("right")
    return touched


def _generate_cavity_box(host_h: int, host_w: int) -> tuple[np.ndarray, str, str | None] | None:
    branch = random.choice(["internal", "edge"])
    side = random.choice(["top", "bottom", "left", "right"]) if branch == "edge" else None
    if branch == "internal":
        max_h = min(CAVITY_BOX_MAX, host_h - 2)
        max_w = min(CAVITY_BOX_MAX, host_w - 2)
    elif side in {"top", "bottom"}:
        max_h = min(CAVITY_BOX_MAX, host_h - 1)
        max_w = min(CAVITY_BOX_MAX, host_w - 2)
    else:
        max_h = min(CAVITY_BOX_MAX, host_h - 2)
        max_w = min(CAVITY_BOX_MAX, host_w - 1)
    if max_h < CAVITY_BOX_MIN or max_w < CAVITY_BOX_MIN:
        return None

    for _ in range(120):
        h = random.randint(CAVITY_BOX_MIN, max_h)
        w = random.randint(CAVITY_BOX_MIN, max_w)
        if branch == "internal":
            allowed = np.zeros((h, w), dtype=bool)
            allowed[1:-1, 1:-1] = True
            max_area = min(CAVITY_AREA_MAX, int(allowed.sum()) - 1)
            if max_area < CAVITY_AREA_MIN:
                continue
            area = random.randint(CAVITY_AREA_MIN, max_area)
            mask = _random_connected_mask(h, w, area, allowed=allowed)
            if mask is None:
                continue
            if int(mask.sum()) == h * w:
                continue
            return _crop_mask(mask), branch, side

        allowed = np.ones((h, w), dtype=bool)
        allowed[0, 0] = False
        allowed[0, w - 1] = False
        allowed[h - 1, 0] = False
        allowed[h - 1, w - 1] = False
        if side == "top":
            seeds = [(0, c) for c in range(1, w - 1)]
        elif side == "bottom":
            seeds = [(h - 1, c) for c in range(1, w - 1)]
        elif side == "left":
            seeds = [(r, 0) for r in range(1, h - 1)]
        else:
            seeds = [(r, w - 1) for r in range(1, h - 1)]
        max_area = min(CAVITY_AREA_MAX, int(allowed.sum()) - 1)
        if max_area < CAVITY_AREA_MIN:
            continue
        area = random.randint(CAVITY_AREA_MIN, max_area)
        mask = _random_connected_mask(h, w, area, allowed=allowed, seed_choices=seeds)
        if mask is None:
            continue
        if _touch_sides(mask) != {side}:
            continue
        if int(mask.sum()) == h * w:
            continue
        return _crop_mask(mask), branch, side
    return None


def _partition_cavity(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    area = int(mask.sum())
    if area < 4:
        return None
    for _ in range(200):
        target = random.randint(2, area - 2)
        part = np.zeros_like(mask, dtype=bool)
        seed = random.choice([(int(r), int(c)) for r, c in np.argwhere(mask)])
        part[seed] = True
        while int(part.sum()) < target:
            frontier: list[Coord] = []
            for r, c in np.argwhere(part):
                for nr, nc in _neighbors4(int(r), int(c), mask.shape):
                    if mask[nr, nc] and not part[nr, nc] and (nr, nc) not in frontier:
                        frontier.append((nr, nc))
            random.shuffle(frontier)
            chosen: Coord | None = None
            for nr, nc in frontier:
                test = part.copy()
                test[nr, nc] = True
                remainder = mask & ~test
                if remainder.any() and _is_connected(remainder):
                    chosen = (nr, nc)
                    break
            if chosen is None:
                break
            part[chosen] = True
        remainder = mask & ~part
        if int(part.sum()) == target and remainder.any() and _is_connected(part) and _is_connected(remainder):
            return _crop_mask(part), _crop_mask(remainder)
    return None


def _random_piece_shape(area: int, *, forbid: set[tuple[Coord, ...]] | None = None) -> np.ndarray | None:
    forbid = forbid or set()
    for _ in range(160):
        h = random.randint(2, min(area, 6))
        w_min = max(2, (area + h - 1) // h)
        w_max = min(area, 6)
        if w_min > w_max:
            continue
        w = random.randint(w_min, w_max)
        mask = _random_connected_mask(h, w, area)
        if mask is None:
            continue
        mask = _crop_mask(mask)
        norm = _normalize_coords(_mask_to_coords(mask))
        if norm in forbid:
            continue
        return mask
    return None


def _rect_gap_conflict(a: BBox, b: BBox, gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_patches(
    grid_shape: tuple[int, int],
    host_bbox: BBox,
    masks: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    placed_rects = [host_bbox]
    positions: list[tuple[int, int]] = []
    order = sorted(range(len(masks)), key=lambda i: -int(masks[i].shape[0] * masks[i].shape[1]))
    ordered_positions: dict[int, tuple[int, int]] = {}
    h_grid, w_grid = grid_shape
    for idx in order:
        mask = masks[idx]
        h, w = mask.shape
        found = None
        for _ in range(240):
            r0 = random.randint(0, h_grid - h)
            c0 = random.randint(0, w_grid - w)
            rect = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_rect_gap_conflict(rect, other, gap=1) for other in placed_rects):
                continue
            found = (r0, c0)
            placed_rects.append(rect)
            break
        if found is None:
            return None
        ordered_positions[idx] = found
    for idx in range(len(masks)):
        positions.append(ordered_positions[idx])
    return positions


def _paste_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = int(color)


def _build_input() -> np.ndarray | None:
    grid_h = random.randint(GRID_MIN, GRID_MAX)
    grid_w = random.randint(GRID_MIN, GRID_MAX)
    host_h = random.randint(HOST_H_RANGE[0], min(HOST_H_RANGE[1], grid_h - 6))
    host_w = random.randint(HOST_W_RANGE[0], min(HOST_W_RANGE[1], grid_w - 6))

    cavity_spec = _generate_cavity_box(host_h, host_w)
    if cavity_spec is None:
        return None
    cavity_box, branch, side = cavity_spec
    cavity_parts = _partition_cavity(cavity_box)
    if cavity_parts is None:
        return None
    true_a, true_b = cavity_parts
    cavity_area = int(cavity_box.sum())

    single_decoy = _random_piece_shape(cavity_area, forbid={_normalize_coords(_mask_to_coords(cavity_box))})
    if single_decoy is None:
        return None

    pair_decoy: tuple[np.ndarray, np.ndarray] | None = None
    for _ in range(120):
        split = random.randint(2, cavity_area - 2)
        decoy_a = _random_piece_shape(split)
        decoy_b = _random_piece_shape(cavity_area - split)
        if decoy_a is None or decoy_b is None:
            continue
        pair_decoy = (decoy_a, decoy_b)
        break
    if pair_decoy is None:
        return None
    decoy_a, decoy_b = pair_decoy

    top_limit = grid_h - host_h
    left_limit = grid_w - host_w
    host_top = random.randint(2, top_limit - 2)
    host_left = random.randint(2, left_limit - 2)
    host_bbox = (host_top, host_left, host_top + host_h - 1, host_left + host_w - 1)

    if branch == "internal":
        cav_top = random.randint(1, host_h - cavity_box.shape[0] - 1)
        cav_left = random.randint(1, host_w - cavity_box.shape[1] - 1)
    elif side == "top":
        cav_top = 0
        cav_left = random.randint(1, host_w - cavity_box.shape[1] - 1)
    elif side == "bottom":
        cav_top = host_h - cavity_box.shape[0]
        cav_left = random.randint(1, host_w - cavity_box.shape[1] - 1)
    elif side == "left":
        cav_top = random.randint(1, host_h - cavity_box.shape[0] - 1)
        cav_left = 0
    else:
        cav_top = random.randint(1, host_h - cavity_box.shape[0] - 1)
        cav_left = host_w - cavity_box.shape[1]

    host_mask = np.ones((host_h, host_w), dtype=bool)
    host_mask[cav_top : cav_top + cavity_box.shape[0], cav_left : cav_left + cavity_box.shape[1]] &= ~cavity_box
    if not _is_connected(host_mask):
        return None

    palette = list(range(10))
    bg, base = random.sample(palette, 2)
    remaining = [color for color in palette if color not in {bg, base}]
    if len(remaining) < 5:
        return None
    fragment_colors = random.sample(remaining, 5)
    fragment_masks = [true_a, true_b, single_decoy, decoy_a, decoy_b]
    fragment_positions = _place_patches((grid_h, grid_w), host_bbox, fragment_masks)
    if fragment_positions is None:
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    grid[host_top : host_top + host_h, host_left : host_left + host_w] = base
    cavity_view = grid[
        host_top + cav_top : host_top + cav_top + cavity_box.shape[0],
        host_left + cav_left : host_left + cav_left + cavity_box.shape[1],
    ]
    cavity_view[cavity_box] = bg

    for mask, (top, left), color in zip(fragment_masks, fragment_positions, fragment_colors):
        _paste_mask(grid, mask, top, left, color)

    return grid


def _semantic_accepts(grid: np.ndarray) -> bool:
    try:
        info = _solve_instance(grid)
    except AssertionError:
        return False

    fragments = info["fragments"]
    cavity_area = int(info["cavity_area"])
    pair = set(info["pair"])

    same_area = [i for i, fragment in enumerate(fragments) if int(fragment["area"]) == cavity_area]
    if len(same_area) != 1:
        return False
    if len(fragments[same_area[0]]["placements"]) != 0:
        return False

    sum_pairs = [
        (i, j)
        for i in range(len(fragments))
        for j in range(i + 1, len(fragments))
        if int(fragments[i]["area"]) + int(fragments[j]["area"]) == cavity_area
    ]
    if len(sum_pairs) != 2:
        return False
    bad_pairs = [set(pair_ids) for pair_ids in sum_pairs if set(pair_ids) != pair]
    if len(bad_pairs) != 1:
        return False
    i, j = tuple(sorted(bad_pairs[0]))
    if fragments[i]["placements"] and fragments[j]["placements"]:
        for placed_i in fragments[i]["placements"]:
            for placed_j in fragments[j]["placements"]:
                if placed_i.isdisjoint(placed_j) and len(placed_i | placed_j) == cavity_area:
                    return False

    base_count = int(np.sum(grid == int(info["base"])))
    bg_count = int(np.sum(grid == int(info["bg"])))
    if bg_count <= base_count:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_input()
        if grid is not None and _semantic_accepts(grid):
            return grid
    raise RuntimeError("failed to generate a valid missing-cavity kinship input")
