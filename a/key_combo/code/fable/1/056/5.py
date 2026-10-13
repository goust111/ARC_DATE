from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BOUNDARY = 3  # ARC green
ONION_COLOR_POOL = (1, 2, 4, 5, 6, 7, 8, 9)
GEN_MAX_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _mask_from_coords(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _bbox_strict_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        int(outer[0]) < int(inner[0])
        and int(outer[1]) < int(inner[1])
        and int(outer[2]) > int(inner[2])
        and int(outer[3]) > int(inner[3])
    )


def _local_interior_mask(component_mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(component_mask)
    out = np.zeros_like(component_mask, dtype=bool)
    if coords.size == 0:
        return out

    h, w = component_mask.shape
    r0, c0, r1, c1 = bbox_coords(coords)
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)

    crop = crop_bbox(component_mask, (rr0, cc0, rr1, cc1))
    open_mask = ~crop
    seen = np.zeros_like(crop, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    ch, cw = crop.shape
    for rr in range(ch):
        for cc in range(cw):
            if rr not in (0, ch - 1) and cc not in (0, cw - 1):
                continue
            if open_mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((int(rr), int(cc)))

    while q:
        rr, cc = q.popleft()
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < ch and 0 <= nc < cw and open_mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((int(nr), int(nc)))

    interior = open_mask & (~seen)
    out[rr0 : rr1 + 1, cc0 : cc1 + 1] = interior
    return out


def _legend_components(grid: np.ndarray, bg: int, wall: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg), int(wall)}):
        for coords in _components_from_mask(grid == int(color)):
            bbox = bbox_coords(coords)
            comps.append({"color": int(color), "coords": coords, "bbox": bbox})

    assert comps, "expected a non-green legend onion"
    comps.sort(key=lambda item: (-_bbox_area(item["bbox"]), item["bbox"][0], item["bbox"][1]))
    for outer, inner in zip(comps, comps[1:]):
        assert _bbox_strict_contains(outer["bbox"], inner["bbox"]), "legend layers must be nested"
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each background region by its true closed-boundary nesting depth, using the corner onion legend as the depth-to-color lookup.

    Input structure:
    - Black background with green 1-pixel boundary curves; some are genuinely closed and some have a 1-2 pixel gap.
    - A separate corner legend consists of concentric monochrome rectangular frames in non-green colors.
    - The legend and the green subject do not touch.

    Rule:
    - Read the legend from outermost layer to innermost layer; these colors represent depths 1, 2, 3, ...
    - Erase the legend for the output, and among green components count only those that topologically enclose a real interior.
    - For every black cell, count how many such closed green boundaries strictly surround it; recolor depth-d cells with legend layer d, keep depth-0 black, and keep all green boundary pixels unchanged.

    Key insight:
    - Apparent layer count is misleading: open green frames must be rejected by a topology check before the remaining closed layers can be counted and translated through the separate onion legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    wall = int(BOUNDARY)
    legend = _legend_components(g, bg, wall)
    legend_colors = [int(item["color"]) for item in legend]
    legend_mask = np.zeros_like(g, dtype=bool)
    for item in legend:
        coords = item["coords"]
        legend_mask[coords[:, 0], coords[:, 1]] = True

    work = g.copy()
    work[legend_mask] = bg

    depth_map = np.zeros_like(g, dtype=int)
    for coords in _components_from_mask(work == wall):
        comp_mask = _mask_from_coords(g.shape, coords)
        interior = _local_interior_mask(comp_mask)
        if interior.any():
            depth_map += interior.astype(int)

    max_depth = int(depth_map.max())
    assert max_depth <= len(legend_colors), "legend onion is too shallow for the subject depth"

    out = work.copy()
    fill_mask = (work == bg) & (depth_map > 0)
    if fill_mask.any():
        lookup = np.array(legend_colors, dtype=int)
        out[fill_mask] = lookup[depth_map[fill_mask] - 1]
    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _gap_cells(boundary: dict) -> list[tuple[int, int]]:
    side, start, length = boundary["gap"]
    r0, c0, r1, c1 = map(int, boundary["bbox"])
    cells: list[tuple[int, int]] = []
    if side == "top":
        cells = [(r0, int(start) + k) for k in range(int(length))]
    elif side == "bottom":
        cells = [(r1, int(start) + k) for k in range(int(length))]
    elif side == "left":
        cells = [(int(start) + k, c0) for k in range(int(length))]
    else:
        cells = [(int(start) + k, c1) for k in range(int(length))]
    return cells


def _draw_boundary(grid: np.ndarray, boundary: dict) -> None:
    _draw_frame(grid, boundary["bbox"], BOUNDARY)
    if not boundary["closed"]:
        for r, c in _gap_cells(boundary):
            grid[int(r), int(c)] = BG


def _sample_gap(bbox: tuple[int, int, int, int]) -> tuple[str, int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    options: list[tuple[str, int, int]] = []
    for length in (1, 2):
        if width >= length + 4:
            for side in ("top", "bottom"):
                start = random.randint(c0 + 1, c1 - length)
                options.append((side, int(start), int(length)))
        if height >= length + 4:
            for side in ("left", "right"):
                start = random.randint(r0 + 1, r1 - length)
                options.append((side, int(start), int(length)))
    return random.choice(options)


def _nested_bboxes(outer_bbox: tuple[int, int, int, int], n_layers: int) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = map(int, outer_bbox)
    out = []
    for k in range(int(n_layers)):
        inset = 2 * int(k)
        out.append((r0 + inset, c0 + inset, r1 - inset, c1 - inset))
    return out


def _make_statuses(closed_count: int, gap_count: int) -> list[bool]:
    total = int(closed_count) + int(gap_count)
    statuses = [True] * total
    if gap_count <= 0:
        return statuses
    gap_positions: set[int] = set()
    if total >= 3:
        gap_positions.add(random.randint(1, total - 2))
    while len(gap_positions) < gap_count:
        gap_positions.add(random.randint(0, total - 1))
    for idx in gap_positions:
        statuses[int(idx)] = False
    return statuses


def _build_tree_boundaries(
    outer_bbox: tuple[int, int, int, int],
    statuses: list[bool],
    *,
    tree_id: int,
) -> list[dict]:
    boundaries: list[dict] = []
    for bbox, closed in zip(_nested_bboxes(outer_bbox, len(statuses)), statuses):
        entry = {"bbox": bbox, "closed": bool(closed), "tree_id": int(tree_id)}
        if not closed:
            entry["gap"] = _sample_gap(bbox)
        boundaries.append(entry)
    return boundaries


def _bbox_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    pad = int(pad)
    return (
        int(a[2]) + pad < int(b[0])
        or int(b[2]) + pad < int(a[0])
        or int(a[3]) + pad < int(b[1])
        or int(b[3]) + pad < int(a[1])
    )


def _candidate_positions(
    shape: tuple[int, int],
    obj_h: int,
    obj_w: int,
    occupied: list[tuple[int, int, int, int]],
    *,
    border: int = 1,
    pad: int = 2,
) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    candidates: list[tuple[int, int]] = []
    for r0 in range(int(border), h - int(obj_h) - int(border) + 1):
        for c0 in range(int(border), w - int(obj_w) - int(border) + 1):
            bbox = (r0, c0, r0 + int(obj_h) - 1, c0 + int(obj_w) - 1)
            if all(_bbox_separated(bbox, other, pad) for other in occupied):
                candidates.append((int(r0), int(c0)))
    return candidates


def _place_onion_bbox(
    shape: tuple[int, int],
    outer_h: int,
    outer_w: int,
    corner: str,
    row_margin: int,
    col_margin: int,
) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    if corner == "tl":
        r0 = int(row_margin)
        c0 = int(col_margin)
    elif corner == "tr":
        r0 = int(row_margin)
        c0 = w - int(col_margin) - int(outer_w)
    elif corner == "bl":
        r0 = h - int(row_margin) - int(outer_h)
        c0 = int(col_margin)
    else:
        r0 = h - int(row_margin) - int(outer_h)
        c0 = w - int(col_margin) - int(outer_w)
    return int(r0), int(c0), int(r0 + outer_h - 1), int(c0 + outer_w - 1)


def _sample_onion_colors(depth_count: int) -> list[int]:
    while True:
        colors = random.sample(list(ONION_COLOR_POOL), int(depth_count))
        if int(depth_count) < 3:
            return [int(c) for c in colors]
        if colors != sorted(colors) and colors != sorted(colors, reverse=True):
            return [int(c) for c in colors]


def _render_scene(
    shape: tuple[int, int],
    boundaries: list[dict],
    onion_bboxes: list[tuple[int, int, int, int]],
    onion_colors: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inp = np.full(shape, BG, dtype=int)
    subject = np.full(shape, BG, dtype=int)
    for boundary in boundaries:
        _draw_boundary(inp, boundary)
        _draw_boundary(subject, boundary)

    depth_map = np.zeros(shape, dtype=int)
    visual_depth = np.zeros(shape, dtype=int)
    for boundary in boundaries:
        r0, c0, r1, c1 = map(int, boundary["bbox"])
        visual_depth[r0 + 1 : r1, c0 + 1 : c1] += 1
        if boundary["closed"]:
            depth_map[r0 + 1 : r1, c0 + 1 : c1] += 1

    expected = subject.copy()
    fill_mask = (subject == BG) & (depth_map > 0)
    lookup = np.array(onion_colors, dtype=int)
    expected[fill_mask] = lookup[depth_map[fill_mask] - 1]

    for bbox, color in zip(onion_bboxes, onion_colors):
        _draw_frame(inp, bbox, int(color))

    return inp, expected, depth_map, visual_depth


def _validate_scene(
    inp: np.ndarray,
    expected: np.ndarray,
    boundaries: list[dict],
    onion_bboxes: list[tuple[int, int, int, int]],
    onion_colors: list[int],
    depth_map: np.ndarray,
    visual_depth: np.ndarray,
) -> bool:
    if inp.shape != expected.shape or inp.ndim != 2:
        return False
    if max(inp.shape) > 30:
        return False
    if np.array_equal(inp, expected):
        return False
    if int(depth_map.max()) != len(onion_colors):
        return False
    if not all((depth_map == d).any() for d in range(1, len(onion_colors) + 1)):
        return False
    if not ((expected == BG).any() and (expected == BOUNDARY).any()):
        return False
    if not any(not boundary["closed"] for boundary in boundaries):
        return False
    if not np.any((depth_map != visual_depth) & (visual_depth > 0)):
        return False

    legend = _legend_components(inp, BG, BOUNDARY)
    if [item["color"] for item in legend] != [int(c) for c in onion_colors]:
        return False
    if [item["bbox"] for item in legend] != [tuple(map(int, b)) for b in onion_bboxes]:
        return False

    for boundary in boundaries:
        test_grid = np.full(inp.shape, BG, dtype=int)
        _draw_boundary(test_grid, boundary)
        comp_mask = test_grid == BOUNDARY
        is_closed = bool(_local_interior_mask(comp_mask).any())
        if is_closed != bool(boundary["closed"]):
            return False
        if not boundary["closed"] and len(_gap_cells(boundary)) not in (1, 2):
            return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _sample_scene() -> tuple[np.ndarray, np.ndarray, list[dict], list[tuple[int, int, int, int]], list[int], np.ndarray, np.ndarray] | None:
    depth_count = random.choice([2, 3, 3])
    if depth_count == 3:
        height = random.randint(27, 30)
        width = random.randint(27, 30)
        main_gap_count = 1
        main_inner_h = random.randint(5, 6)
        main_inner_w = random.randint(5, 6)
        second_tree_prob = 0.35
    else:
        height = random.randint(25, 30)
        width = random.randint(25, 30)
        main_gap_count = random.choice([1, 2])
        main_inner_h = random.randint(5, 7)
        main_inner_w = random.randint(5, 8)
        second_tree_prob = 0.6
    shape = (height, width)

    main_statuses = _make_statuses(depth_count, main_gap_count)
    main_outer_h = main_inner_h + 4 * (len(main_statuses) - 1)
    main_outer_w = main_inner_w + 4 * (len(main_statuses) - 1)

    onion_inner_h = random.randint(1, 2)
    onion_inner_w = random.randint(1, 2)
    onion_outer_h = onion_inner_h + 4 * (depth_count - 1)
    onion_outer_w = onion_inner_w + 4 * (depth_count - 1)
    corner = random.choice(["tl", "tr", "bl", "br"])
    onion_bbox = _place_onion_bbox(
        shape,
        onion_outer_h,
        onion_outer_w,
        corner,
        random.randint(0, 1),
        random.randint(0, 1),
    )
    onion_bboxes = _nested_bboxes(onion_bbox, depth_count)
    onion_colors = _sample_onion_colors(depth_count)

    occupied = [onion_bbox]
    main_positions = _candidate_positions(shape, main_outer_h, main_outer_w, occupied, border=1, pad=2)
    if not main_positions:
        return None
    main_r0, main_c0 = random.choice(main_positions)
    main_outer_bbox = (main_r0, main_c0, main_r0 + main_outer_h - 1, main_c0 + main_outer_w - 1)
    boundaries = _build_tree_boundaries(main_outer_bbox, main_statuses, tree_id=0)
    occupied.append(main_outer_bbox)

    if random.random() < second_tree_prob:
        closed2 = random.randint(1, min(2, depth_count))
        gap2 = random.randint(0, 1)
        second_statuses = _make_statuses(closed2, gap2)
        second_inner_h = random.randint(5, 7)
        second_inner_w = random.randint(5, 7)
        second_outer_h = second_inner_h + 4 * (len(second_statuses) - 1)
        second_outer_w = second_inner_w + 4 * (len(second_statuses) - 1)
        second_positions = _candidate_positions(shape, second_outer_h, second_outer_w, occupied, border=1, pad=2)
        if second_positions:
            r0, c0 = random.choice(second_positions)
            bbox = (r0, c0, r0 + second_outer_h - 1, c0 + second_outer_w - 1)
            boundaries.extend(_build_tree_boundaries(bbox, second_statuses, tree_id=1))
            occupied.append(bbox)

    inp, expected, depth_map, visual_depth = _render_scene(shape, boundaries, onion_bboxes, onion_colors)
    return inp, expected, boundaries, onion_bboxes, onion_colors, depth_map, visual_depth


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        sample = _sample_scene()
        if sample is None:
            continue
        inp, expected, boundaries, onion_bboxes, onion_colors, depth_map, visual_depth = sample
        if _validate_scene(inp, expected, boundaries, onion_bboxes, onion_colors, depth_map, visual_depth):
            return inp
    raise RuntimeError(f"failed to generate valid input after {GEN_MAX_ATTEMPTS} attempts")
