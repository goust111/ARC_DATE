from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
TEMPLATE_COLOR = 1
RING_COLORS = (2, 3, 4, 7)
GUIDE_COLORS = (5, 6, 8, 9)

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GUIDE_STEPS = [(-2, 0), (2, 0), (0, -2), (0, 2)]
LOCAL_MAIN_GUIDES = [(0, 1), (1, 1), (2, 1), (2, 0), (2, 2), (2, 3)]
LOCAL_REACHABLE_TARGETS = [
    {"center": (3, 0), "attach": (2, 0)},
    {"center": (3, 2), "attach": (2, 2)},
    {"center": (1, 3), "attach": (2, 3)},
]
LOCAL_DISC_GUIDES = [(0, 0), (0, 1), (1, 1)]
LOCAL_DISC_TARGETS = [
    {"center": (-1, 1), "attach": (0, 1)},
    {"center": (1, 2), "attach": (1, 1)},
]
GRID_CHOICES = (23, 25, 27, 29)
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components4(mask: np.ndarray) -> list[np.ndarray]:
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
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _adjacent_sites_from_topleft(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c + 1), (r + 3, c + 1), (r + 1, c - 1), (r + 1, c + 3)]


def _find_template(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    blue_mask = grid == TEMPLATE_COLOR
    candidates = []
    for comp in _components4(blue_mask):
        area = int(comp.shape[0])
        r0, c0, r1, c1 = _bbox(comp)
        if not (6 <= area <= 12 and r1 < 5 and c1 < 5):
            continue
        mask = np.zeros((5, 5), dtype=bool)
        for r, c in comp:
            mask[int(r), int(c)] = True
        if np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.fliplr(mask)):
            continue
        candidates.append((mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0, r1, c1)))
    assert len(candidates) == 1, "expected exactly one template component"
    return candidates[0]


def _find_checker(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    matches: list[tuple[int, int]] = []
    expected = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=int,
    )
    for r in range(h - 2):
        for c in range(w - 2):
            if np.array_equal(grid[r : r + 3, c : c + 3], expected):
                matches.append((r, c))
    assert len(matches) == 1, "expected exactly one checkerboard"
    return matches[0]


def _find_rings(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    rings: list[dict] = []
    for r in range(h - 2):
        for c in range(w - 2):
            window = grid[r : r + 3, c : c + 3]
            border = [
                int(window[0, 0]),
                int(window[0, 1]),
                int(window[0, 2]),
                int(window[1, 0]),
                int(window[1, 2]),
                int(window[2, 0]),
                int(window[2, 1]),
                int(window[2, 2]),
            ]
            if int(window[1, 1]) != BG and int(window[1, 1]) != _get_bg(grid):
                continue
            if len(set(border)) != 1:
                continue
            color = int(border[0])
            if color not in RING_COLORS:
                continue
            if int(window[1, 1]) != BG:
                continue
            rings.append({"topleft": (r, c), "center": (r + 1, c + 1), "color": color})
    return rings


def _build_object_mask(
    shape: tuple[int, int],
    template_bbox: tuple[int, int, int, int],
    checker_tl: tuple[int, int],
    rings: list[dict],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    tr0, tc0, tr1, tc1 = template_bbox
    mask[tr0 : tr1 + 1, tc0 : tc1 + 1] = True
    cr, cc = checker_tl
    mask[cr : cr + 3, cc : cc + 3] = True
    for ring in rings:
        rr, cc = ring["topleft"]
        mask[rr : rr + 3, cc : cc + 3] = True
    return mask


def _bfs_order(guide_nodes: set[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    assert start in guide_nodes
    seen = {start}
    frontier = [start]
    order: list[tuple[int, int]] = []
    while frontier:
        frontier = sorted(frontier)
        next_frontier: list[tuple[int, int]] = []
        for r, c in frontier:
            order.append((r, c))
            for dr, dc in GUIDE_STEPS:
                nb = (r + dr, c + dc)
                if nb in guide_nodes and nb not in seen:
                    seen.add(nb)
                    next_frontier.append(nb)
        frontier = next_frontier
    return order


def _rotate_template(template: np.ndarray, rotation: int) -> np.ndarray:
    return np.rot90(template, k=rotation)


def _stamp(
    out: np.ndarray,
    template: np.ndarray,
    center: tuple[int, int],
    color: int,
) -> None:
    h, w = template.shape
    r0 = center[0] - h // 2
    c0 = center[1] - w // 2
    for rr in range(h):
        for cc in range(w):
            if template[rr, cc]:
                out[r0 + rr, c0 + cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp rotated template copies onto the reachable ring targets selected by the guide path from the checkerboard.

    Input structure:
    - The background is black. The upper-left 5x5 area contains one irregular blue template component.
    - Elsewhere there is one 3x3 blue/black checkerboard, several colored 3x3 rings, and sparse guide pixels on a two-cell lattice.
    - The checkerboard has exactly one adjacent guide pixel, and only rings adjacent to the connected same-color guide network started there are targets.

    Rule:
    - Find the checkerboard, read the color of its unique adjacent guide pixel, and BFS through guide pixels of that color using two-cell orthogonal steps with row-major tie breaking inside each layer.
    - Any ring 8-adjacent to that reachable guide network is a target; its arrival side determines the template rotation: north=0, east=clockwise 90, south=180, west=counterclockwise 90.
    - Process targets in BFS discovery order, stamp the rotated template centered on each ring using the ring color, let later stamps overwrite earlier overlaps, erase the checkerboard, all guides, and all non-target rings, and keep the original template unchanged.

    Key insight:
    - The same guide network simultaneously filters reachable rings, encodes each stamp rotation by approach direction, and fixes overlap priority through BFS visit order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert bg == BG

    template, template_bbox = _find_template(g)
    checker_tl = _find_checker(g)
    rings = _find_rings(g)
    assert 4 <= len(rings) <= 6

    object_mask = _build_object_mask(g.shape, template_bbox, checker_tl, rings)

    connector_candidates = []
    for rr, cc in _adjacent_sites_from_topleft(*checker_tl):
        if not _in_bounds(g.shape, rr, cc):
            continue
        if object_mask[rr, cc]:
            continue
        if int(g[rr, cc]) != bg:
            connector_candidates.append((rr, cc, int(g[rr, cc])))
    assert len(connector_candidates) == 1, "checkerboard must have exactly one adjacent guide pixel"
    start_r, start_c, guide_color = connector_candidates[0]

    guide_nodes = {
        (int(r), int(c))
        for r, c in np.argwhere((g == guide_color) & ~object_mask)
    }
    order = _bfs_order(guide_nodes, (start_r, start_c))
    rank = {node: idx for idx, node in enumerate(order)}

    targets = []
    for ring in rings:
        center_r, center_c = ring["center"]
        arrivals = {
            (center_r - 2, center_c): 0,
            (center_r, center_c + 2): 3,
            (center_r + 2, center_c): 2,
            (center_r, center_c - 2): 1,
        }
        reachable = [(rank[node], rot) for node, rot in arrivals.items() if node in rank]
        if not reachable:
            continue
        reachable.sort()
        idx, rotation = reachable[0]
        targets.append((idx, center_r, center_c, ring["color"], rotation))

    assert len(targets) >= 2
    targets.sort()

    out = np.full_like(g, bg)
    tr0, tc0, tr1, tc1 = template_bbox
    out[tr0 : tr1 + 1, tc0 : tc1 + 1][template] = TEMPLATE_COLOR

    for _idx, center_r, center_c, color, rotation in targets:
        rotated = _rotate_template(template, rotation)
        _stamp(out, rotated, (center_r, center_c), color)

    return out


def _rot_offset(coord: tuple[int, int], rot: int) -> tuple[int, int]:
    r, c = coord
    if rot % 4 == 0:
        return r, c
    if rot % 4 == 1:
        return -c, r
    if rot % 4 == 2:
        return -r, -c
    return c, -r


def _apply_transform(coords: list[tuple[int, int]], rot: int, anchor: tuple[int, int]) -> list[tuple[int, int]]:
    out = []
    for coord in coords:
        dr, dc = _rot_offset(coord, rot)
        out.append((anchor[0] + dr, anchor[1] + dc))
    return out


def _apply_transform_targets(
    targets: list[dict], rot: int, anchor: tuple[int, int]
) -> list[dict]:
    out = []
    for target in targets:
        center = _rot_offset(target["center"], rot)
        attach = _rot_offset(target["attach"], rot)
        out.append(
            {
                "center": (anchor[0] + center[0], anchor[1] + center[1]),
                "attach": (anchor[0] + attach[0], anchor[1] + attach[1]),
            }
        )
    return out


def _lattice_to_grid(coord: tuple[int, int]) -> tuple[int, int]:
    return 2 * coord[0] + 1, 2 * coord[1] + 1


def _draw_template(grid: np.ndarray, mask: np.ndarray) -> None:
    grid[:5, :5][mask] = TEMPLATE_COLOR


def _draw_checker(grid: np.ndarray, center: tuple[int, int]) -> None:
    rr, cc = _lattice_to_grid(center)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            grid[rr + dr, cc + dc] = TEMPLATE_COLOR if (dr + dc) % 2 == 0 else BG


def _draw_ring(grid: np.ndarray, center: tuple[int, int], color: int) -> None:
    rr, cc = _lattice_to_grid(center)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                grid[rr + dr, cc + dc] = BG
            else:
                grid[rr + dr, cc + dc] = int(color)


def _draw_guides(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for coord in coords:
        rr, cc = _lattice_to_grid(coord)
        grid[rr, cc] = int(color)


def _mask_symmetries(mask: np.ndarray) -> bool:
    checks = [
        np.flipud(mask),
        np.fliplr(mask),
        mask.T,
        np.fliplr(mask.T),
    ]
    return any(np.array_equal(mask, other) for other in checks)


def _all_rotations_distinct(mask: np.ndarray) -> bool:
    rots = [np.rot90(mask, k) for k in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            if np.array_equal(rots[i], rots[j]):
                return False
    return True


def _gen_template_mask() -> np.ndarray:
    for _ in range(200):
        area = random.randint(6, 12)
        mask = np.zeros((5, 5), dtype=bool)
        start = (random.randint(0, 4), random.randint(0, 4))
        mask[start] = True
        cells = {start}
        frontier = set()
        for dr, dc in DIRS4:
            nr, nc = start[0] + dr, start[1] + dc
            if 0 <= nr < 5 and 0 <= nc < 5:
                frontier.add((nr, nc))
        while len(cells) < area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            if cell in cells:
                continue
            cells.add(cell)
            mask[cell] = True
            for dr, dc in DIRS4:
                nr, nc = cell[0] + dr, cell[1] + dc
                if 0 <= nr < 5 and 0 <= nc < 5 and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if len(cells) != area:
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if min(rows) != 0 or max(rows) != 4 or min(cols) != 0 or max(cols) != 4:
            continue
        if _mask_symmetries(mask):
            continue
        if not _all_rotations_distinct(mask):
            continue
        return mask
    raise RuntimeError("failed to generate asymmetric template")


def _anchor_range(
    coords: list[tuple[int, int]],
    lat_h: int,
    lat_w: int,
) -> tuple[range, range]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    min_use_r, max_use_r = 3, lat_h - 2
    min_use_c, max_use_c = 3, lat_w - 2
    r_lo = min_use_r - min(rows)
    r_hi = max_use_r - max(rows)
    c_lo = min_use_c - min(cols)
    c_hi = max_use_c - max(cols)
    return range(r_lo, r_hi + 1), range(c_lo, c_hi + 1)


def _chebyshev_far(a: list[tuple[int, int]], b: list[tuple[int, int]], gap: int) -> bool:
    for ar, ac in a:
        for br, bc in b:
            if max(abs(ar - br), abs(ac - bc)) < gap:
                return False
    return True


def _network_coords(guides: list[tuple[int, int]], targets: list[dict], checker: tuple[int, int] | None) -> list[tuple[int, int]]:
    coords = list(guides) + [target["center"] for target in targets]
    if checker is not None:
        coords.append(checker)
    return coords


def _target_rotation(center: tuple[int, int], attach: tuple[int, int]) -> int:
    dr = attach[0] - center[0]
    dc = attach[1] - center[1]
    if (dr, dc) == (-1, 0):
        return 0
    if (dr, dc) == (0, 1):
        return 3
    if (dr, dc) == (1, 0):
        return 2
    if (dr, dc) == (0, -1):
        return 1
    raise AssertionError((center, attach))


def _main_bfs_target_order(
    checker: tuple[int, int],
    guides: list[tuple[int, int]],
    targets: list[dict],
) -> list[dict]:
    start_candidates = [node for node in guides if max(abs(node[0] - checker[0]), abs(node[1] - checker[1])) == 1 and abs(node[0] - checker[0]) + abs(node[1] - checker[1]) == 1]
    assert len(start_candidates) == 1
    guide_set = set(guides)
    start = start_candidates[0]
    seen = {start}
    frontier = [start]
    order: list[tuple[int, int]] = []
    while frontier:
        frontier = sorted(frontier)
        nxt: list[tuple[int, int]] = []
        for node in frontier:
            order.append(node)
            for dr, dc in DIRS4:
                nb = (node[0] + dr, node[1] + dc)
                if nb in guide_set and nb not in seen:
                    seen.add(nb)
                    nxt.append(nb)
        frontier = nxt
    rank = {node: i for i, node in enumerate(order)}
    ordered = []
    for target in targets:
        ordered.append(
            {
                "rank": rank[target["attach"]],
                "center": target["center"],
                "rotation": _target_rotation(target["center"], target["attach"]),
                "color": target["color"],
            }
        )
    ordered.sort(key=lambda item: (item["rank"], item["center"][0], item["center"][1]))
    return ordered


def _render_output(
    shape: tuple[int, int],
    template_mask: np.ndarray,
    template_bbox: tuple[int, int, int, int],
    stamps: list[dict],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    tr0, tc0, tr1, tc1 = template_bbox
    out[tr0 : tr1 + 1, tc0 : tc1 + 1][template_mask] = TEMPLATE_COLOR
    for stamp in stamps:
        center_rc = _lattice_to_grid(stamp["center"])
        rotated = np.rot90(template_mask, k=stamp["rotation"])
        _stamp(out, rotated, center_rc, stamp["color"])
    return out


def _build_scene() -> tuple[np.ndarray, np.ndarray]:
    h = random.choice(GRID_CHOICES)
    w = random.choice(GRID_CHOICES)
    lat_h = (h - 1) // 2
    lat_w = (w - 1) // 2

    template_mask = _gen_template_mask()
    template_bbox = (0, 0, 4, 4)

    rot_main = random.randint(0, 3)
    main_local_coords = [(0, 0)] + LOCAL_MAIN_GUIDES + [item["center"] for item in LOCAL_REACHABLE_TARGETS]
    main_anchor_rows, main_anchor_cols = _anchor_range(
        [_rot_offset(coord, rot_main) for coord in main_local_coords], lat_h, lat_w
    )
    if not main_anchor_rows or not main_anchor_cols:
        raise RuntimeError("no room for main network")
    checker = (random.choice(tuple(main_anchor_rows)), random.choice(tuple(main_anchor_cols)))
    main_guides = _apply_transform(LOCAL_MAIN_GUIDES, rot_main, checker)
    reachable_targets = _apply_transform_targets(LOCAL_REACHABLE_TARGETS, rot_main, checker)

    same_count = random.randint(1, 2)
    other_count = random.randint(0, 1 if same_count == 2 else 2)
    total_rings = 3 + same_count + other_count
    if total_rings > 6:
        other_count = max(0, 6 - (3 + same_count))

    all_main_coords = _network_coords(main_guides, reachable_targets, checker)

    disc_same_rot = random.randint(0, 3)
    same_local_coords = LOCAL_DISC_GUIDES + [item["center"] for item in LOCAL_DISC_TARGETS[:same_count]]
    same_rows, same_cols = _anchor_range(
        [_rot_offset(coord, disc_same_rot) for coord in same_local_coords], lat_h, lat_w
    )
    disc_same_guides: list[tuple[int, int]] | None = None
    disc_same_targets: list[dict] | None = None
    for _ in range(120):
        anchor = (random.choice(tuple(same_rows)), random.choice(tuple(same_cols)))
        guides = _apply_transform(LOCAL_DISC_GUIDES, disc_same_rot, anchor)
        targets = _apply_transform_targets(LOCAL_DISC_TARGETS[:same_count], disc_same_rot, anchor)
        if _chebyshev_far(_network_coords(guides, targets, None), all_main_coords, 3):
            disc_same_guides = guides
            disc_same_targets = targets
            break
    if disc_same_guides is None or disc_same_targets is None:
        raise RuntimeError("failed to place same-color distractor")

    extra_coords = all_main_coords + _network_coords(disc_same_guides, disc_same_targets, None)
    disc_other_guides: list[tuple[int, int]] = []
    disc_other_targets: list[dict] = []
    if other_count:
        disc_other_rot = random.randint(0, 3)
        other_local_coords = LOCAL_DISC_GUIDES + [item["center"] for item in LOCAL_DISC_TARGETS[:other_count]]
        other_rows, other_cols = _anchor_range(
            [_rot_offset(coord, disc_other_rot) for coord in other_local_coords], lat_h, lat_w
        )
        placed = False
        for _ in range(120):
            anchor = (random.choice(tuple(other_rows)), random.choice(tuple(other_cols)))
            guides = _apply_transform(LOCAL_DISC_GUIDES, disc_other_rot, anchor)
            targets = _apply_transform_targets(LOCAL_DISC_TARGETS[:other_count], disc_other_rot, anchor)
            if _chebyshev_far(_network_coords(guides, targets, None), extra_coords, 3):
                disc_other_guides = guides
                disc_other_targets = targets
                placed = True
                break
        if not placed:
            raise RuntimeError("failed to place other-color distractor")

    ring_colors = [random.choice(RING_COLORS) for _ in range(total_rings)]
    while ring_colors[0] == ring_colors[1]:
        ring_colors[1] = random.choice(RING_COLORS)

    for target, color in zip(
        reachable_targets + disc_same_targets + disc_other_targets,
        ring_colors,
    ):
        target["color"] = int(color)

    main_guide_color = random.choice(GUIDE_COLORS)
    other_guide_color_choices = [color for color in GUIDE_COLORS if color != main_guide_color]
    other_guide_color = random.choice(other_guide_color_choices)

    grid = np.zeros((h, w), dtype=int)
    _draw_template(grid, template_mask)
    _draw_checker(grid, checker)
    for target in reachable_targets:
        _draw_ring(grid, target["center"], target["color"])
    for target in disc_same_targets:
        _draw_ring(grid, target["center"], target["color"])
    for target in disc_other_targets:
        _draw_ring(grid, target["center"], target["color"])
    _draw_guides(grid, main_guides + disc_same_guides, main_guide_color)
    if disc_other_guides:
        _draw_guides(grid, disc_other_guides, other_guide_color)

    ordered_targets = _main_bfs_target_order(checker, main_guides, reachable_targets)
    expected = _render_output(grid.shape, template_mask, template_bbox, ordered_targets)

    all_rings_render = _render_output(
        grid.shape,
        template_mask,
        template_bbox,
        ordered_targets
        + [
            {
                "center": target["center"],
                "rotation": _target_rotation(target["center"], target["attach"]),
                "color": target["color"],
            }
            for target in disc_same_targets + disc_other_targets
        ],
    )
    no_rotation_render = _render_output(
        grid.shape,
        template_mask,
        template_bbox,
        [
            {
                "center": item["center"],
                "rotation": 0,
                "color": item["color"],
            }
            for item in ordered_targets
        ],
    )
    reversed_render = _render_output(
        grid.shape,
        template_mask,
        template_bbox,
        list(reversed(ordered_targets)),
    )

    assert len(reachable_targets) >= 2
    assert len(disc_same_targets) + len(disc_other_targets) >= 1
    assert len({item["rotation"] for item in ordered_targets}) >= 2
    assert not np.array_equal(expected, all_rings_render)
    assert not np.array_equal(expected, no_rotation_render)
    assert not np.array_equal(expected, reversed_render)
    assert np.array_equal(main(grid), expected)
    assert 4 <= len(reachable_targets) + len(disc_same_targets) + len(disc_other_targets) <= 6

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, _expected = _build_scene()
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate puzzle input")
