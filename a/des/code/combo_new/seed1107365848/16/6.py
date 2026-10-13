import numpy as np


_RNG = np.random.default_rng()
_BG = 0
_GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            rows = [pt[0] for pt in coords]
            cols = [pt[1] for pt in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _border_coords(r0: int, c0: int, side: int) -> set[tuple[int, int]]:
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    pts = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _side_centers(bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    mr = (r0 + r1) // 2
    mc = (c0 + c1) // 2
    return {
        "top": (r0, mc),
        "right": (mr, c1),
        "bottom": (r1, mc),
        "left": (mr, c0),
    }


def _frame_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    candidates: list[dict] = []
    for comp in _components(grid, bg):
        r0, c0, r1, c1 = comp["bbox"]
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        if side_h != side_w or side_h < 3:
            continue
        border = _border_coords(r0, c0, side_h)
        comp_set = set(comp["coords"])
        if not comp_set.issubset(border):
            continue
        missing = border - comp_set
        if len(missing) > 1:
            continue
        marker_pos = None
        if missing:
            marker_pos = next(iter(missing))
            if marker_pos not in set(_side_centers(comp["bbox"]).values()):
                continue
        candidates.append(
            {
                "color": int(comp["color"]),
                "bbox": comp["bbox"],
                "side": side_h,
                "marker_pos": marker_pos,
            }
        )
    return candidates


def _find_stamp_and_canvas(grid: np.ndarray, bg: int) -> tuple[dict, dict, int, int]:
    by_color: dict[int, list[dict]] = {}
    for cand in _frame_candidates(grid, bg):
        by_color.setdefault(int(cand["color"]), []).append(cand)

    matches: list[tuple[dict, dict, int, int]] = []
    for frame_color, cands in by_color.items():
        stamps = [cand for cand in cands if cand["marker_pos"] is not None]
        canvases = [cand for cand in cands if cand["marker_pos"] is None]
        for stamp in stamps:
            marker_r, marker_c = stamp["marker_pos"]
            marker_color = int(grid[marker_r, marker_c])
            if marker_color in (bg, frame_color):
                continue
            for canvas in canvases:
                if int(stamp["side"]) >= int(canvas["side"]):
                    continue
                r0, c0, r1, c1 = canvas["bbox"]
                inner = grid[r0 + 1 : r1, c0 + 1 : c1]
                if np.count_nonzero(inner == marker_color) == 0:
                    continue
                matches.append((stamp, canvas, frame_color, marker_color))

    assert len(matches) == 1, "expected exactly one valid stamp/canvas pair"
    return matches[0]


def _rotation_turns(stamp_bbox: tuple[int, int, int, int], marker_pos: tuple[int, int]) -> int:
    centers = _side_centers(stamp_bbox)
    if marker_pos == centers["top"]:
        return 0
    if marker_pos == centers["right"]:
        return 1
    if marker_pos == centers["bottom"]:
        return 2
    if marker_pos == centers["left"]:
        return 3
    raise AssertionError("invalid stamp marker position")


def _apply_stamp(
    out: np.ndarray,
    pattern: np.ndarray,
    canvas_bbox: tuple[int, int, int, int],
    marker_points: list[tuple[int, int]],
    bg: int,
) -> None:
    r0, c0, r1, c1 = canvas_bbox
    ph, pw = pattern.shape
    cr = ph // 2
    cc = pw // 2
    ink = np.argwhere(pattern != bg)

    for mr, mc in marker_points:
        for pr, pc in ink:
            tr = int(mr + int(pr) - cr)
            tc = int(mc + int(pc) - cc)
            if r0 < tr < r1 and c0 < tc < c1:
                out[tr, tc] = int(pattern[int(pr), int(pc)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the framed stamp by its border marker and stamp it onto each matching marker inside the larger framed canvas.

    Input structure:
    - The background is black, with exactly two disjoint square frame objects sharing one border color.
    - The smaller frame is the stamp: its border is one pixel thick, it has one non-background interior pattern, and one border-middle marker pixel of another color.
    - The larger frame is the canvas: its border uses the same frame color and its black interior contains several single-pixel markers in the same color as the stamp marker.

    Rule:
    - Identify the smaller framed square as the stamp and the larger framed square as the canvas.
    - Read the stamp marker position on the frame border to determine a clockwise rotation: top=0, right=90, bottom=180, left=270 degrees.
    - Extract the stamp interior pattern, rotate it by that amount, and stamp its non-black pixels onto the canvas interior centered on each canvas marker from top to bottom, then left to right; pixels that would land outside the canvas interior are skipped, and later stamps overwrite earlier ones.
    - Remove the canvas marker pixels after stamping, while leaving the original stamp object unchanged.

    Key insight:
    - The same marker color plays two roles: on the small frame it encodes the rotation angle, and inside the large frame it encodes the placement centers for repeated stamping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    stamp, canvas, _frame_color, marker_color = _find_stamp_and_canvas(g, bg)
    turns = _rotation_turns(stamp["bbox"], stamp["marker_pos"])

    sr0, sc0, sr1, sc1 = stamp["bbox"]
    pattern = g[sr0 + 1 : sr1, sc0 + 1 : sc1].copy()
    rotated = np.rot90(pattern, -turns)

    cr0, cc0, cr1, cc1 = canvas["bbox"]
    canvas_inner = g[cr0 + 1 : cr1, cc0 + 1 : cc1]
    local_markers = np.argwhere(canvas_inner == marker_color)
    assert local_markers.size > 0, "canvas must contain at least one marker"
    marker_points = sorted((int(cr0 + 1 + r), int(cc0 + 1 + c)) for r, c in local_markers)

    out = g.copy()
    for r, c in marker_points:
        out[r, c] = bg
    _apply_stamp(out, rotated, canvas["bbox"], marker_points, bg)
    return out


def _random_colors(count: int, exclude: set[int]) -> list[int]:
    pool = [color for color in range(1, 10) if color not in exclude]
    choice = _RNG.choice(pool, size=count, replace=False)
    return [int(x) for x in np.atleast_1d(choice)]


def _random_connected_mask(side: int) -> np.ndarray:
    cells = [(r, c) for r in range(side) for c in range(side)]
    start = cells[int(_RNG.integers(len(cells)))]
    target = int(_RNG.integers(3, side * side))
    chosen = {start}

    while len(chosen) < target:
        frontier = []
        for r, c in chosen:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in chosen:
                    frontier.append((nr, nc))
        if not frontier:
            break
        chosen.add(frontier[int(_RNG.integers(len(frontier)))])

    mask = np.zeros((side, side), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _build_pattern(inner_side: int, colors: list[int]) -> np.ndarray:
    for _ in range(200):
        mask = _random_connected_mask(inner_side)
        if np.count_nonzero(mask) < 3:
            continue
        pattern = np.zeros((inner_side, inner_side), dtype=int)
        coords = list(zip(*np.where(mask)))
        _RNG.shuffle(coords)
        for idx, (r, c) in enumerate(coords):
            pattern[r, c] = colors[idx % len(colors)]
        if all(not np.array_equal(pattern, np.rot90(pattern, k)) for k in (1, 2, 3)):
            return pattern
    raise RuntimeError("failed to build an asymmetric stamp pattern")


def _stamp_marker_pos(side: int, turns: int) -> tuple[int, int]:
    mid = side // 2
    options = {
        0: (0, mid),
        1: (mid, side - 1),
        2: (side - 1, mid),
        3: (mid, 0),
    }
    return options[int(turns)]


def _distinct_positions(pool: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    idxs = _RNG.choice(len(pool), size=count, replace=False)
    return [pool[int(i)] for i in np.atleast_1d(idxs)]


def _overlap_neighbor(
    base: tuple[int, int],
    inner_side: int,
    pattern_side: int,
    used: set[tuple[int, int]],
) -> tuple[int, int] | None:
    br, bc = base
    candidates = []
    for dr in range(-(pattern_side - 2), pattern_side - 1):
        for dc in range(-(pattern_side - 2), pattern_side - 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = br + dr, bc + dc
            if not (0 <= nr < inner_side and 0 <= nc < inner_side):
                continue
            if (nr, nc) in used:
                continue
            if max(abs(dr), abs(dc)) < pattern_side - 1:
                candidates.append((nr, nc))
    if not candidates:
        return None
    return candidates[int(_RNG.integers(len(candidates)))]


def _sample_canvas_markers(inner_side: int, pattern_side: int, scenario: str) -> list[tuple[int, int]]:
    all_cells = [(r, c) for r in range(inner_side) for c in range(inner_side)]
    center = pattern_side // 2
    edge_cells = [
        (r, c)
        for r, c in all_cells
        if r < center or c < center or r >= inner_side - center or c >= inner_side - center
    ]
    interior_cells = [
        (r, c)
        for r, c in all_cells
        if center <= r < inner_side - center and center <= c < inner_side - center
    ]

    used: set[tuple[int, int]] = set()
    markers: list[tuple[int, int]] = []

    def add(pt: tuple[int, int]) -> None:
        if pt not in used:
            used.add(pt)
            markers.append(pt)

    if scenario == "normal":
        count = int(_RNG.integers(2, 6))
        for pt in _distinct_positions(all_cells, count):
            add(pt)
    elif scenario == "overlap":
        base = interior_cells[int(_RNG.integers(len(interior_cells)))] if interior_cells else all_cells[int(_RNG.integers(len(all_cells)))]
        add(base)
        neighbor = _overlap_neighbor(base, inner_side, pattern_side, used)
        if neighbor is not None:
            add(neighbor)
        extra = int(_RNG.integers(0, 3))
        leftovers = [pt for pt in all_cells if pt not in used]
        if leftovers and extra > 0:
            for pt in _distinct_positions(leftovers, min(extra, len(leftovers))):
                add(pt)
    elif scenario == "edge":
        add(edge_cells[int(_RNG.integers(len(edge_cells)))])
        extra = int(_RNG.integers(1, 4))
        leftovers = [pt for pt in all_cells if pt not in used]
        for pt in _distinct_positions(leftovers, min(extra, len(leftovers))):
            add(pt)
    else:
        add(edge_cells[int(_RNG.integers(len(edge_cells)))])
        base = interior_cells[int(_RNG.integers(len(interior_cells)))] if interior_cells else all_cells[int(_RNG.integers(len(all_cells)))]
        add(base)
        neighbor = _overlap_neighbor(base, inner_side, pattern_side, used)
        if neighbor is not None:
            add(neighbor)
        leftovers = [pt for pt in all_cells if pt not in used]
        extra = int(_RNG.integers(0, 3))
        if leftovers and extra > 0:
            for pt in _distinct_positions(leftovers, min(extra, len(leftovers))):
                add(pt)

    return sorted(markers)


def _place_squares(stamp_side: int, canvas_side: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    max_gap = int(30 - stamp_side - canvas_side)
    assert max_gap >= 1, "objects must fit with at least one-cell separation"
    gap = int(_RNG.integers(1, max_gap + 1))
    horizontal = bool(int(_RNG.integers(2)))
    stamp_first = bool(int(_RNG.integers(2)))

    if horizontal:
        min_h = max(stamp_side, canvas_side)
        min_w = stamp_side + canvas_side + gap
        extra_h = int(_RNG.integers(0, 31 - min_h))
        extra_w = int(_RNG.integers(0, 31 - min_w))
        h = min_h + extra_h
        w = min_w + extra_w
        lead = int(_RNG.integers(extra_w + 1))
        if stamp_first:
            stamp_c = lead
            canvas_c = lead + stamp_side + gap
        else:
            canvas_c = lead
            stamp_c = lead + canvas_side + gap
        stamp_r = int(_RNG.integers(0, h - stamp_side + 1))
        canvas_r = int(_RNG.integers(0, h - canvas_side + 1))
    else:
        min_h = stamp_side + canvas_side + gap
        min_w = max(stamp_side, canvas_side)
        extra_h = int(_RNG.integers(0, 31 - min_h))
        extra_w = int(_RNG.integers(0, 31 - min_w))
        h = min_h + extra_h
        w = min_w + extra_w
        lead = int(_RNG.integers(extra_h + 1))
        if stamp_first:
            stamp_r = lead
            canvas_r = lead + stamp_side + gap
        else:
            canvas_r = lead
            stamp_r = lead + canvas_side + gap
        stamp_c = int(_RNG.integers(0, w - stamp_side + 1))
        canvas_c = int(_RNG.integers(0, w - canvas_side + 1))

    return (h, w), (stamp_r, stamp_c), (canvas_r, canvas_c)


def _draw_frame(grid: np.ndarray, top: int, left: int, side: int, color: int) -> None:
    bottom = top + side - 1
    right = left + side - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _gen_attempt() -> np.ndarray | None:
    stamp_side = int(_RNG.choice([5, 7]))
    canvas_choices = [side for side in range(3 * stamp_side, 30 - stamp_side, 2)]
    canvas_side = int(canvas_choices[int(_RNG.integers(len(canvas_choices)))])
    turns = int(_RNG.integers(4))
    scenario = ("normal", "overlap", "edge", "mixed")[int(_RNG.integers(4))]

    frame_color, marker_color = _random_colors(2, {_BG})
    pattern_color_count = int(_RNG.integers(1, 4))
    pattern_colors = _random_colors(pattern_color_count, {_BG, frame_color, marker_color})
    pattern = _build_pattern(stamp_side - 2, pattern_colors)

    (h, w), (stamp_r, stamp_c), (canvas_r, canvas_c) = _place_squares(stamp_side, canvas_side)
    grid = np.full((h, w), _BG, dtype=int)

    _draw_frame(grid, stamp_r, stamp_c, stamp_side, frame_color)
    _draw_frame(grid, canvas_r, canvas_c, canvas_side, frame_color)

    marker_sr, marker_sc = _stamp_marker_pos(stamp_side, turns)
    grid[stamp_r + marker_sr, stamp_c + marker_sc] = marker_color
    grid[stamp_r + 1 : stamp_r + stamp_side - 1, stamp_c + 1 : stamp_c + stamp_side - 1] = pattern

    canvas_inner_side = canvas_side - 2
    markers = _sample_canvas_markers(canvas_inner_side, stamp_side - 2, scenario)
    for r, c in markers:
        grid[canvas_r + 1 + r, canvas_c + 1 + c] = marker_color

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
