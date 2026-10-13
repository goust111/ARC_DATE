from __future__ import annotations

import random

import numpy as np


BG = 0
PATH = 5
RECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400
MIN_RECT_SIDE = 2
MAX_RECT_SIDE = 4


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(cnts.argmax())])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _distance_to_bbox(pt: tuple[int, int], bbox: tuple[int, int, int, int]) -> int:
    r, c = pt
    r0, c0, r1, c1 = bbox
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return dr + dc


def _trace_path(grid: np.ndarray, path_color: int) -> list[tuple[int, int]]:
    path_mask = grid == path_color
    comps = _components(path_mask)
    assert len(comps) == 1, "expected a single grey path component"
    pts = comps[0]
    pset = set(pts)
    h, w = grid.shape

    endpoints = []
    for r, c in pts:
        deg = sum((nr, nc) in pset for nr, nc in _neighbors4(r, c))
        if deg == 1:
            endpoints.append((r, c))
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    border_endpoints = [
        (r, c)
        for r, c in endpoints
        if r == 0 or c == 0 or r == h - 1 or c == w - 1
    ]
    start = border_endpoints[0] if border_endpoints else min(endpoints)

    ordered = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [
            (nr, nc)
            for nr, nc in _neighbors4(*cur)
            if (nr, nc) in pset and (nr, nc) != prev
        ]
        if not nxts:
            break
        assert len(nxts) == 1, "path must not branch"
        prev, cur = cur, nxts[0]
        ordered.append(cur)

    assert len(ordered) == len(pts), "path traversal must visit every path pixel"
    return ordered


def _find_rectangles(
    grid: np.ndarray, bg: int, path_color: int
) -> tuple[list[dict[str, object]], np.ndarray]:
    h, w = grid.shape
    owner = np.full((h, w), -1, dtype=int)
    rects: list[dict[str, object]] = []

    for color in sorted(set(map(int, np.unique(grid)))):
        if color in (bg, path_color):
            continue
        comps = _components(grid == color)
        assert len(comps) == 1, "each rectangle color must appear in one component"
        coords = comps[0]
        r0, c0, r1, c1 = _bbox(coords)
        assert len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1), "component must be a solid rectangle"
        rid = len(rects)
        for r, c in coords:
            owner[r, c] = rid
        rects.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "height": r1 - r0 + 1,
                "width": c1 - c0 + 1,
            }
        )

    assert rects, "expected at least one colored rectangle"
    return rects, owner


def _corner_bbox(
    shape: tuple[int, int], corner: str, height: int, width: int
) -> tuple[int, int, int, int]:
    h, w = shape
    if corner == "tl":
        return 0, 0, height - 1, width - 1
    if corner == "tr":
        return 0, w - width, height - 1, w - 1
    if corner == "bl":
        return h - height, 0, h - 1, width - 1
    if corner == "br":
        return h - height, w - width, h - 1, w - 1
    raise ValueError(corner)


def _choose_target_corner(
    grid: np.ndarray, endpoint: tuple[int, int], outer_h: int, outer_w: int, bg: int
) -> tuple[int, int, int, int]:
    best: tuple[int, int, tuple[int, int, int, int]] | None = None
    for i, corner in enumerate(("tl", "tr", "bl", "br")):
        r0, c0, r1, c1 = _corner_bbox(grid.shape, corner, outer_h, outer_w)
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == bg):
            continue
        dist = _distance_to_bbox(endpoint, (r0, c0, r1, c1))
        score = (dist, i)
        if best is None or score < (best[0], best[1]):
            best = (dist, i, (r0, c0, r1, c1))
    assert best is not None, "could not locate a corner blank region large enough for the nest"
    return best[2]


def _draw_ring(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _draw_nested(
    out: np.ndarray,
    target_bbox: tuple[int, int, int, int],
    colors_outer_to_inner: list[int],
    inner_h: int,
    inner_w: int,
) -> None:
    r0, c0, r1, c1 = target_bbox
    layers = len(colors_outer_to_inner)
    for i, color in enumerate(colors_outer_to_inner[:-1]):
        _draw_ring(out, r0 + i, c0 + i, r1 - i, c1 - i, color)
    inner_r0 = r0 + layers - 1
    inner_c0 = c0 + layers - 1
    out[inner_r0 : inner_r0 + inner_h, inner_c0 : inner_c0 + inner_w] = colors_outer_to_inner[-1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the grey path to order the rectangles, then rebuild them as nested corner rings.

    Input structure:
    - The background is black.
    - Several solid rectangles of distinct colors are scattered in the grid.
    - A single-pixel grey path starts from one border endpoint, meanders through the grid, and becomes 4-neighbor adjacent to the rectangles without overlapping them.
    - One corner contains a large reserved black area.

    Rule:
    - Traverse the grey path from its border endpoint to its other endpoint.
    - Record the rectangles in the order in which the path first becomes 4-neighbor adjacent to them.
    - Remove the path and all original rectangles.
    - In the reserved corner area, draw the last-touched rectangle as the inner solid rectangle with its original size, then wrap one-pixel-thick rectangular rings around it in reverse encounter order so the first-touched color becomes the outermost ring.

    Key insight:
    - The grey path is the ordering device: rectangle size and position do not determine the nesting order, only first-contact order along the traced path does.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h > 0 and w > 0

    bg = _most_common_color(g)
    path = _trace_path(g, PATH)
    rects, owner = _find_rectangles(g, bg, PATH)

    seen: set[int] = set()
    ordered_ids: list[int] = []
    for r, c in path:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and owner[nr, nc] != -1:
                rid = int(owner[nr, nc])
                if rid not in seen:
                    seen.add(rid)
                    ordered_ids.append(rid)

    assert ordered_ids, "path must touch at least one rectangle"
    last = rects[ordered_ids[-1]]
    inner_h = int(last["height"])
    inner_w = int(last["width"])
    layers = len(ordered_ids)
    outer_h = inner_h + 2 * (layers - 1)
    outer_w = inner_w + 2 * (layers - 1)

    target_bbox = _choose_target_corner(g, path[-1], outer_h, outer_w, bg)
    out = np.full_like(g, bg)
    colors = [int(rects[rid]["color"]) for rid in ordered_ids]
    _draw_nested(out, target_bbox, colors, inner_h, inner_w)
    return out


def _bbox_intersects(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0
) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _random_partition(total: int, k: int) -> list[int] | None:
    if k <= 0 or total < k:
        return None
    parts = [1] * k
    for _ in range(total - k):
        parts[random.randrange(k)] += 1
    random.shuffle(parts)
    return parts


def _build_path(
    start: tuple[int, int],
    end: tuple[int, int],
    start_orientation: str,
    seg_count: int,
) -> tuple[list[tuple[int, int]], list[dict[str, object]]] | None:
    dr = end[0] - start[0]
    dc = end[1] - start[1]
    if dr == 0 or dc == 0:
        return None

    orientations = [
        start_orientation if i % 2 == 0 else ("H" if start_orientation == "V" else "V")
        for i in range(seg_count)
    ]
    nv = sum(o == "V" for o in orientations)
    nh = seg_count - nv
    v_parts = _random_partition(abs(dr), nv)
    h_parts = _random_partition(abs(dc), nh)
    if v_parts is None or h_parts is None:
        return None

    step_r = 1 if dr > 0 else -1
    step_c = 1 if dc > 0 else -1
    path = [start]
    runs: list[dict[str, object]] = []
    cur = start
    iv = 0
    ih = 0
    for orientation in orientations:
        start_idx = len(path) - 1
        if orientation == "V":
            length = v_parts[iv]
            iv += 1
            step = (step_r, 0)
        else:
            length = h_parts[ih]
            ih += 1
            step = (0, step_c)
        for _ in range(length):
            cur = (cur[0] + step[0], cur[1] + step[1])
            path.append(cur)
        runs.append(
            {
                "orientation": orientation,
                "direction": step,
                "start_idx": start_idx,
                "length": length,
            }
        )

    if path[-1] != end or len(set(path)) != len(path):
        return None
    return path, runs


def _candidate_bbox(
    path: list[tuple[int, int]],
    run: dict[str, object],
    offset: int,
    rect_h: int,
    rect_w: int,
    side: int,
) -> tuple[tuple[int, int, int, int], int, int]:
    start_idx = int(run["start_idx"])
    orientation = str(run["orientation"])
    dr, dc = run["direction"]  # type: ignore[misc]
    contact_idx = start_idx + 1 + offset
    pr, pc = path[contact_idx]

    if orientation == "H":
        if dc > 0:
            c0 = pc
        else:
            c0 = pc - rect_w + 1
        r0 = pr - rect_h if side < 0 else pr + 1
        along = rect_w
    else:
        if dr > 0:
            r0 = pr
        else:
            r0 = pr - rect_h + 1
        c0 = pc - rect_w if side < 0 else pc + 1
        along = rect_h

    bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
    return bbox, contact_idx, along


def _adjacent_path_indices(
    bbox: tuple[int, int, int, int], path: list[tuple[int, int]]
) -> list[int]:
    r0, c0, r1, c1 = bbox
    idxs = []
    for i, (r, c) in enumerate(path):
        if r0 <= r <= r1 and (c == c0 - 1 or c == c1 + 1):
            idxs.append(i)
        elif c0 <= c <= c1 and (r == r0 - 1 or r == r1 + 1):
            idxs.append(i)
    return idxs


def _valid_rectangle_bbox(
    bbox: tuple[int, int, int, int],
    path: list[tuple[int, int]],
    reserve_bbox: tuple[int, int, int, int],
    existing: list[tuple[int, int, int, int]],
    expected_contact: int,
    expected_along: int,
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
        return False
    if _bbox_intersects(bbox, reserve_bbox):
        return False
    for other in existing:
        if _bbox_intersects(bbox, other, pad=1):
            return False

    box_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    if any(pt in box_cells for pt in path):
        return False

    idxs = _adjacent_path_indices(bbox, path)
    return idxs == list(range(expected_contact, expected_contact + expected_along))


def _place_rectangles(
    shape: tuple[int, int],
    path: list[tuple[int, int]],
    runs: list[dict[str, object]],
    reserve_bbox: tuple[int, int, int, int],
    n_rects: int,
    inner_h: int,
    inner_w: int,
) -> list[dict[str, object]] | None:
    candidate_runs = [i for i, run in enumerate(runs) if int(run["length"]) >= 2]
    if len(candidate_runs) < n_rects:
        return None

    valid_last_runs = []
    for i in candidate_runs:
        run = runs[i]
        if str(run["orientation"]) == "H" and int(run["length"]) >= inner_w:
            valid_last_runs.append(i)
        if str(run["orientation"]) == "V" and int(run["length"]) >= inner_h:
            valid_last_runs.append(i)
    if not valid_last_runs:
        return None

    last_run_idx = random.choice(valid_last_runs)
    earlier = [i for i in candidate_runs if i < last_run_idx]
    if len(earlier) < n_rects - 1:
        return None
    selected = sorted(random.sample(earlier, n_rects - 1) + [last_run_idx])

    colors = random.sample(RECT_COLORS, n_rects)
    placements: list[dict[str, object]] = []
    bboxes: list[tuple[int, int, int, int]] = []

    for order_idx, run_idx in enumerate(selected):
        run = runs[run_idx]
        orientation = str(run["orientation"])
        run_length = int(run["length"])

        if order_idx == n_rects - 1:
            rect_h, rect_w = inner_h, inner_w
        else:
            if orientation == "H":
                rect_w = random.randint(MIN_RECT_SIDE, min(MAX_RECT_SIDE, run_length))
                rect_h = random.randint(MIN_RECT_SIDE, MAX_RECT_SIDE)
            else:
                rect_h = random.randint(MIN_RECT_SIDE, min(MAX_RECT_SIDE, run_length))
                rect_w = random.randint(MIN_RECT_SIDE, MAX_RECT_SIDE)

        along = rect_w if orientation == "H" else rect_h
        if along > run_length:
            return None

        options = []
        for offset in range(run_length - along + 1):
            for side in (-1, 1):
                bbox, contact_idx, expected_along = _candidate_bbox(
                    path, run, offset, rect_h, rect_w, side
                )
                if _valid_rectangle_bbox(
                    bbox,
                    path,
                    reserve_bbox,
                    bboxes,
                    contact_idx,
                    expected_along,
                    shape,
                ):
                    options.append((bbox, contact_idx))
        if not options:
            return None

        bbox, contact_idx = random.choice(options)
        placements.append(
            {
                "bbox": bbox,
                "color": colors[order_idx],
                "height": rect_h,
                "width": rect_w,
                "contact_idx": contact_idx,
            }
        )
        bboxes.append(bbox)

    placements.sort(key=lambda x: int(x["contact_idx"]))
    return placements


def _expected_output(
    shape: tuple[int, int],
    reserve_corner: str,
    ordered_rectangles: list[dict[str, object]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    inner_h = int(ordered_rectangles[-1]["height"])
    inner_w = int(ordered_rectangles[-1]["width"])
    layers = len(ordered_rectangles)
    outer_h = inner_h + 2 * (layers - 1)
    outer_w = inner_w + 2 * (layers - 1)
    bbox = _corner_bbox(shape, reserve_corner, outer_h, outer_w)
    colors = [int(rect["color"]) for rect in ordered_rectangles]
    _draw_nested(out, bbox, colors, inner_h, inner_w)
    return out


def _sample_layout() -> tuple[int, int, str, int, int, int]:
    n_rects = random.randint(2, 5)
    inner_h = random.randint(2, 3)
    inner_w = random.randint(2, 3)
    outer_h = inner_h + 2 * (n_rects - 1)
    outer_w = inner_w + 2 * (n_rects - 1)
    h = random.randint(max(18, outer_h + 8), min(30, outer_h + 14))
    w = random.randint(max(18, outer_w + 8), min(30, outer_w + 14))
    corner = random.choice(["tl", "tr", "bl", "br"])
    return h, w, corner, n_rects, inner_h, inner_w


def _sample_end(shape: tuple[int, int], reserve_bbox: tuple[int, int, int, int], corner: str) -> tuple[int, int]:
    h, w = shape
    r0, c0, r1, c1 = reserve_bbox
    if corner == "tl":
        choices = []
        if r1 + 1 < h:
            choices.append((r1 + 1, random.randint(c0, c1)))
        if c1 + 1 < w:
            choices.append((random.randint(r0, r1), c1 + 1))
        return random.choice(choices)
    if corner == "tr":
        choices = []
        if r1 + 1 < h:
            choices.append((r1 + 1, random.randint(c0, c1)))
        if c0 - 1 >= 0:
            choices.append((random.randint(r0, r1), c0 - 1))
        return random.choice(choices)
    if corner == "bl":
        choices = []
        if r0 - 1 >= 0:
            choices.append((r0 - 1, random.randint(c0, c1)))
        if c1 + 1 < w:
            choices.append((random.randint(r0, r1), c1 + 1))
        return random.choice(choices)
    choices = []
    if r0 - 1 >= 0:
        choices.append((r0 - 1, random.randint(c0, c1)))
    if c0 - 1 >= 0:
        choices.append((random.randint(r0, r1), c0 - 1))
    return random.choice(choices)


def _sample_start(shape: tuple[int, int], corner: str, end: tuple[int, int]) -> tuple[tuple[int, int], str, int]:
    h, w = shape
    if corner == "tl":
        start_side = random.choice(["bottom", "right"])
    elif corner == "tr":
        start_side = random.choice(["bottom", "left"])
    elif corner == "bl":
        start_side = random.choice(["top", "right"])
    else:
        start_side = random.choice(["top", "left"])

    if start_side == "top":
        return (0, random.randint(1, w - 2)), "V", random.randint(4, 6)
    if start_side == "bottom":
        return (h - 1, random.randint(1, w - 2)), "V", random.randint(4, 6)
    if start_side == "left":
        return (random.randint(1, h - 2), 0), "H", random.randint(4, 6)
    return (random.randint(1, h - 2), w - 1), "H", random.randint(4, 6)


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    h, w, corner, n_rects, inner_h, inner_w = _sample_layout()
    outer_h = inner_h + 2 * (n_rects - 1)
    outer_w = inner_w + 2 * (n_rects - 1)
    reserve_bbox = _corner_bbox((h, w), corner, outer_h, outer_w)
    end = _sample_end((h, w), reserve_bbox, corner)
    start, start_orientation, seg_count = _sample_start((h, w), corner, end)

    built = _build_path(start, end, start_orientation, seg_count)
    if built is None:
        return None
    path, runs = built

    rectangles = _place_rectangles((h, w), path, runs, reserve_bbox, n_rects, inner_h, inner_w)
    if rectangles is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for r, c in path:
        grid[r, c] = PATH
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]  # type: ignore[misc]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])

    expected = _expected_output(grid.shape, corner, rectangles)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _generate_case()
        if case is None:
            continue
        grid, expected = case
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
