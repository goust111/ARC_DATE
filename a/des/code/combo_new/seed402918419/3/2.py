import random
from collections import deque

import numpy as np


BG = 0
TRACK = 5
FRAGMENT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400
PLACEMENT_RETRIES = 250
GAP_SELECTION_RETRIES = 200


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _touches_border(points: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in points)


def _normalize(points: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    r0, c0 = min(rs), min(cs)
    rel = sorted((r - r0, c - c0) for r, c in points)
    return rel, max(r for r, _ in rel) + 1, max(c for _, c in rel) + 1


def _connected(points: set[tuple[int, int]]) -> bool:
    if not points:
        return False
    start = next(iter(points))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in points and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(points)


def _line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r1, c1 = a
    r2, c2 = b
    if r1 == r2:
        step = 1 if c2 >= c1 else -1
        return [(r1, c) for c in range(c1, c2 + step, step)]
    if c1 == c2:
        step = 1 if r2 >= r1 else -1
        return [(r, c1) for r in range(r1, r2 + step, step)]
    raise ValueError("segments must be axis-aligned")


def _cycle_from_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for i in range(len(vertices)):
        seg = _line(vertices[i], vertices[(i + 1) % len(vertices)])
        if path:
            seg = seg[1:]
        path.extend(seg)
    if path[-1] == path[0]:
        path.pop()
    assert len(path) == len(set(path))
    return path


def _rotate_path(path: list[tuple[int, int]], turns: int) -> list[tuple[int, int]]:
    pts = path[:]
    h = max(r for r, _ in pts) + 1
    w = max(c for _, c in pts) + 1
    for _ in range(turns % 4):
        pts = [(c, h - 1 - r) for r, c in pts]
        h, w = w, h
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return [(r - min_r, c - min_c) for r, c in pts]


def _shift(points: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in points]


def _build_rectangle_path() -> list[tuple[int, int]]:
    height = random.randint(7, 12)
    width = random.randint(8, 13)
    verts = [(0, 0), (0, width - 1), (height - 1, width - 1), (height - 1, 0)]
    return _cycle_from_vertices(verts)


def _build_l_path() -> list[tuple[int, int]]:
    height = random.randint(8, 13)
    width = random.randint(8, 13)
    bend_r = random.randint(2, height - 3)
    bend_c = random.randint(2, width - 3)
    verts = [(0, 0), (0, width - 1), (bend_r, width - 1), (bend_r, bend_c), (height - 1, bend_c), (height - 1, 0)]
    path = _cycle_from_vertices(verts)
    return _rotate_path(path, random.randint(0, 3))


def _straight_segments(path: list[tuple[int, int]]) -> list[list[int]]:
    n = len(path)
    dirs = [(path[(i + 1) % n][0] - path[i][0], path[(i + 1) % n][1] - path[i][1]) for i in range(n)]
    starts = [0]
    for i in range(1, n):
        if dirs[i] != dirs[i - 1]:
            starts.append(i)

    segments = []
    for idx, start in enumerate(starts):
        if idx + 1 < len(starts):
            end_edge = starts[idx + 1] - 1
            segments.append(list(range(start, end_edge + 2)))
        else:
            tail = list(range(start, n))
            segments.append(tail + [0])
    return segments


def _place_fragment(
    grid: np.ndarray,
    occupied: np.ndarray,
    fragment_rel: list[tuple[int, int]],
    frag_h: int,
    frag_w: int,
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    side_order = random.sample(["top", "bottom", "left", "right"], 4)
    for side in side_order:
        anchors = []
        if side == "top":
            anchors = [(0, c) for c in range(0, w - frag_w + 1)]
        elif side == "bottom":
            anchors = [(h - frag_h, c) for c in range(0, w - frag_w + 1)]
        elif side == "left":
            anchors = [(r, 0) for r in range(0, h - frag_h + 1)]
        else:
            anchors = [(r, w - frag_w) for r in range(0, h - frag_h + 1)]
        random.shuffle(anchors)
        for ar, ac in anchors:
            placed = [(ar + rr, ac + cc) for rr, cc in fragment_rel]
            if any(grid[r, c] != BG for r, c in placed):
                continue
            ok = True
            for r, c in placed:
                for nr in range(r - 1, r + 2):
                    for nc in range(c - 1, c + 2):
                        if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc] and (nr, nc) not in placed:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                return placed
    return None


def _find_fragment_target(
    grid: np.ndarray,
    scaffold: set[tuple[int, int]],
    markers: set[tuple[int, int]],
    fragment_rel: list[tuple[int, int]],
    frag_h: int,
    frag_w: int,
    used: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    h, w = grid.shape
    valid = []
    for ar in range(h - frag_h + 1):
        for ac in range(w - frag_w + 1):
            placed = {(ar + rr, ac + cc) for rr, cc in fragment_rel}
            if placed & used:
                continue
            if any(grid[r, c] != BG for r, c in placed):
                continue
            bad_neighbor = False
            for r, c in placed:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) in scaffold and (nr, nc) not in markers:
                        bad_neighbor = True
                        break
                if bad_neighbor:
                    break
            if bad_neighbor:
                continue
            union = placed | markers
            good = True
            for cell in union:
                deg = sum((nr, nc) in union for nr, nc in _neighbors4(*cell))
                if cell in markers:
                    if deg != 1:
                        good = False
                        break
                elif deg != 2:
                    good = False
                    break
            if not good or not _connected(union):
                continue
            valid.append(placed)
    assert len(valid) == 1, f"expected unique placement for color-marked fragment, found {len(valid)}"
    return valid[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each broken loop segment by inserting the matching colored fragment back into the grey track.

    Input structure:
    - The background is black.
    - A central one-pixel-wide grey loop (rectangular or L-shaped) contains several black gaps.
    - The two track pixels immediately beside each gap are recolored to the same non-grey color.
    - A same-colored 4-connected fragment touching the grid border matches each gap.

    Rule:
    - For each non-grey color, treat the border-touching component of that color as the movable fragment.
    - Treat the two interior single pixels of that color as the marked ends of the corresponding gap.
    - Translate the fragment, without rotating or flipping it, to the unique black location where it forms a simple path between the two markers.
    - Recolor both markers and the inserted fragment to grey, and erase the fragment from its original border position to black.

    Key insight:
    The colored pieces are only pairing signals: color identifies which fragment belongs to which break, while the final restored loop is entirely grey.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    colors = [int(c) for c in np.unique(g) if c not in (BG, TRACK)]
    assert colors, "input must contain colored markers and fragments"

    infos = []
    scaffold = set(map(tuple, np.argwhere(g == TRACK)))
    for color in colors:
        comps = _components(g == color)
        border_comps = [pts for pts in comps if _touches_border(pts, h, w)]
        marker_comps = [pts for pts in comps if not _touches_border(pts, h, w)]
        assert len(border_comps) == 1, "each color must have exactly one border fragment"
        assert len(marker_comps) == 2 and all(len(pts) == 1 for pts in marker_comps), "each gap must have exactly two interior marker pixels"
        fragment = border_comps[0]
        markers = {pts[0] for pts in marker_comps}
        scaffold |= markers
        rel, frag_h, frag_w = _normalize(fragment)
        infos.append({"color": color, "fragment": fragment, "markers": markers, "rel": rel, "h": frag_h, "w": frag_w})

    out = g.copy()
    used: set[tuple[int, int]] = set()
    for info in sorted(infos, key=lambda x: -len(x["fragment"])):
        target = _find_fragment_target(out, scaffold, info["markers"], info["rel"], info["h"], info["w"], used)
        used |= target
        for r, c in info["fragment"]:
            out[r, c] = BG
        for r, c in info["markers"]:
            out[r, c] = TRACK
        for r, c in target:
            out[r, c] = TRACK

    final_track = set(map(tuple, np.argwhere(out == TRACK)))
    assert all(out[r, c] in (BG, TRACK) for r in range(h) for c in range(w))
    assert _connected(final_track)
    for cell in final_track:
        deg = sum((nr, nc) in final_track for nr, nc in _neighbors4(*cell))
        assert deg == 2, "restored track must be a single-pixel loop"
    return out


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    shape_kind = random.choice(["rectangle", "lshape"])
    local_path = _build_rectangle_path() if shape_kind == "rectangle" else _build_l_path()
    local_h = max(r for r, _ in local_path) + 1
    local_w = max(c for _, c in local_path) + 1

    h_low = max(18, local_h + 12)
    h_high = min(30, local_h + 16)
    w_low = max(18, local_w + 12)
    w_high = min(30, local_w + 16)
    if h_low > h_high or w_low > w_high:
        return None

    h = random.randint(h_low, h_high)
    w = random.randint(w_low, w_high)
    top = random.randint(6, h - local_h - 6)
    left = random.randint(6, w - local_w - 6)
    path = _shift(local_path, top, left)
    segments = [seg for seg in _straight_segments(path) if len(seg) >= 4]
    if not segments:
        return None
    max_gaps = min(4, len(FRAGMENT_COLORS), len(segments))
    gap_count = random.randint(1, max_gaps)
    chosen_segments = random.sample(segments, gap_count)

    input_grid = np.full((h, w), BG, dtype=int)
    output_grid = np.full((h, w), BG, dtype=int)
    for r, c in path:
        input_grid[r, c] = TRACK
        output_grid[r, c] = TRACK

    colors = random.sample(FRAGMENT_COLORS, gap_count)
    occupied = input_grid != BG

    for color, seg in zip(colors, chosen_segments):
        max_len = min(5, len(seg) - 3)
        if max_len < 1:
            return None
        length = random.randint(1, max_len)
        start_offset = random.randint(1, len(seg) - length - 2)
        gap_indices = seg[start_offset : start_offset + length]
        gap_cells = [path[i] for i in gap_indices]
        left_marker = path[seg[start_offset - 1]]
        right_marker = path[seg[start_offset + length]]

        for r, c in gap_cells:
            input_grid[r, c] = BG
            occupied[r, c] = False
        for r, c in (left_marker, right_marker):
            input_grid[r, c] = color

        fragment_rel, frag_h, frag_w = _normalize(gap_cells)
        placed = _place_fragment(input_grid, occupied, fragment_rel, frag_h, frag_w)
        if placed is None:
            return None
        for r, c in placed:
            input_grid[r, c] = color
            occupied[r, c] = True

    try:
        solved = main(input_grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, output_grid):
        return None
    return input_grid, output_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            built = _gen_attempt()
        except AssertionError:
            built = None
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid broken-loop input")
