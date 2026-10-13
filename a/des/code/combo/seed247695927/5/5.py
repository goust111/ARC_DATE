import random

import numpy as np


MAX_ATTEMPTS = 400
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR8 if connectivity == 8 else DIR4
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(points: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _, _ = _bbox(points)
    return {(r - r0, c - c0) for r, c in points}


def _shift(points: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in points}


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _infer_colors(grid: np.ndarray) -> tuple[int, int, int]:
    bg = _background_color(grid)
    fg_colors = [int(v) for v in np.unique(grid) if int(v) != bg]
    assert len(fg_colors) == 2, "expected exactly two foreground colors"

    scored = []
    for color in fg_colors:
        comps = _components(grid == color, connectivity=8)
        scored.append((len(comps), int(np.count_nonzero(grid == color)), color))

    connector_candidates = [item for item in scored if item[0] == 2]
    assert connector_candidates, "connector color must form exactly two components"
    connector_candidates.sort(key=lambda item: (item[1], item[2]))
    ctrl = connector_candidates[0][2]
    obj = next(color for color in fg_colors if color != ctrl)
    return bg, obj, ctrl


def _sprite_groups(grid: np.ndarray, obj: int, ctrl: int) -> list[dict[str, set[tuple[int, int]]]]:
    conn_comps = _components(grid == ctrl, connectivity=8)
    assert len(conn_comps) == 2, "expected exactly two connector components"
    conn_sets = [set(comp) for comp in conn_comps]
    obj_comps = _components(grid == obj, connectivity=8)

    groups = [{"conn": conn, "body": set()} for conn in conn_sets]
    for comp in obj_comps:
        pts = set(comp)
        best_idx = None
        best_dist = None
        second = None
        for idx, conn in enumerate(conn_sets):
            dist = min(abs(r1 - r2) + abs(c1 - c2) for r1, c1 in pts for r2, c2 in conn)
            if best_dist is None or dist < best_dist:
                second = best_dist
                best_dist = dist
                best_idx = idx
            elif second is None or dist < second:
                second = dist
        assert best_idx is not None
        assert second is None or best_dist < second, "fragment assignment must be unambiguous"
        groups[best_idx]["body"].update(pts)

    for group in groups:
        group["all"] = group["conn"] | group["body"]
        assert group["body"], "each sprite needs a visible body"
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect two sprites using matching connector pixels, then tight-crop.

    The sprite whose bounding box is closest to the top-left is the base.
    The other sprite is translated so that its connector sits one row above
    the base connector, then both connector colors are recolored to the body
    color and the foreground is tightly cropped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, obj, ctrl = _infer_colors(g)
    groups = _sprite_groups(g, obj, ctrl)
    groups.sort(key=lambda group: (_bbox(group["all"])[0] + _bbox(group["all"])[1], _bbox(group["all"])[0], _bbox(group["all"])[1]))
    base, plug = groups

    base_conn = base["conn"]
    plug_conn = plug["conn"]
    assert _normalize(base_conn) == _normalize(plug_conn), "connector shapes must match"

    br0, bc0, _, _ = _bbox(base_conn)
    pr0, pc0, _, _ = _bbox(plug_conn)
    dr = br0 - 1 - pr0
    dc = bc0 - pc0

    base_body = set(base["body"])
    base_conn_shifted = set(base_conn)
    plug_body_shifted = _shift(plug["body"], dr, dc)
    plug_conn_shifted = _shift(plug_conn, dr, dc)

    all_points = base["all"] | plug_body_shifted | plug_conn_shifted
    r0, c0, r1, c1 = _bbox(all_points)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)

    def stamp(points: set[tuple[int, int]], color: int) -> None:
        for r, c in points:
            rr, cc = r - r0, c - c0
            current = int(out[rr, cc])
            if current != bg and current != color:
                assert current == ctrl or color == ctrl, "only connector overlaps are allowed"
            out[rr, cc] = color

    stamp(base_body, obj)
    stamp(base_conn_shifted, ctrl)
    stamp(plug_body_shifted, obj)
    stamp(plug_conn_shifted, ctrl)
    out[out == ctrl] = obj

    pts = np.argwhere(out != bg)
    rr0, cc0 = pts.min(axis=0)
    rr1, cc1 = pts.max(axis=0)
    return out[int(rr0) : int(rr1) + 1, int(cc0) : int(cc1) + 1]


def _grow_connected(mask: np.ndarray, target: int, rng: random.Random) -> None:
    h, w = mask.shape
    frontier = list(zip(*np.where(mask)))
    while int(mask.sum()) < target and frontier:
        r, c = frontier[rng.randrange(len(frontier))]
        neighbors = []
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                neighbors.append((nr, nc))
        if not neighbors:
            frontier.remove((r, c))
            continue
        nr, nc = neighbors[rng.randrange(len(neighbors))]
        mask[nr, nc] = True
        frontier.append((nr, nc))


def _sample_run(mask: np.ndarray, length: int, top_exposed: bool) -> list[tuple[int, int]] | None:
    h, w = mask.shape
    candidates: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c0 in range(w - length + 1):
            cells = [(r, c) for c in range(c0, c0 + length)]
            if not all(mask[r, c] for _, c in cells):
                continue
            if top_exposed:
                if r + 1 >= h or not all(mask[r + 1, c] for _, c in cells):
                    continue
                if any(r - 1 >= 0 and mask[r - 1, c] for _, c in cells):
                    continue
            else:
                if r - 1 < 0 or not all(mask[r - 1, c] for _, c in cells):
                    continue
                if any(r + 1 < h and mask[r + 1, c] for _, c in cells):
                    continue
            candidates.append(cells)
    if not candidates:
        return None
    return random.choice(candidates)


def _random_body_with_connector(rng: random.Random, top_exposed: bool, conn_len: int) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    for _ in range(80):
        h = rng.randint(4, 8)
        w = rng.randint(max(4, conn_len + 1), 9)
        mask = np.zeros((h, w), dtype=bool)
        seed_row = h - 1 if top_exposed else 0
        seed_col = rng.randrange(w)
        mask[seed_row, seed_col] = True
        target = rng.randint(max(conn_len + 3, 6), min(h * w - 1, 18))
        _grow_connected(mask, target, rng)
        run = _sample_run(mask, conn_len, top_exposed=top_exposed)
        if run is None:
            continue
        return mask, run
    return None


def _make_plug_absolute(
    base_obj: set[tuple[int, int]],
    base_conn: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    base_conn_r = base_conn[0][0]
    base_cols = sorted(c for _, c in base_conn)
    conn_len = len(base_cols)
    for _ in range(120):
        plug_h = rng.randint(4, 8)
        plug_w = rng.randint(max(4, conn_len + 1), 9)
        conn_start = rng.randint(0, plug_w - conn_len)
        abs_left = base_cols[0] - conn_start
        mask = np.zeros((plug_h, plug_w), dtype=bool)
        for c in range(conn_start, conn_start + conn_len):
            mask[1, c] = True

        frontier = list(zip(*np.where(mask)))
        target = rng.randint(conn_len + 3, min(plug_h * plug_w - conn_len, 18))
        while int(mask.sum()) < target and frontier:
            r, c = frontier[rng.randrange(len(frontier))]
            moves = []
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if not (1 <= nr < plug_h and 0 <= nc < plug_w):
                    continue
                if mask[nr, nc]:
                    continue
                ar = base_conn_r - 1 + nr
                ac = abs_left + nc
                if (ar, ac) in base_obj:
                    continue
                moves.append((nr, nc))
            if not moves:
                frontier.remove((r, c))
                continue
            nr, nc = moves[rng.randrange(len(moves))]
            mask[nr, nc] = True
            frontier.append((nr, nc))

        conn = {(base_conn_r - 1, c) for c in base_cols}
        body = set()
        valid = True
        for r, c in zip(*np.where(mask)):
            ar = base_conn_r - 1 + int(r)
            ac = abs_left + int(c)
            if (ar, ac) in base_obj:
                valid = False
                break
            body.add((ar, ac))
        if not valid or not body:
            continue
        return body, conn
    return None


def _normalize_sprite(body: set[tuple[int, int]], conn: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    all_pts = body | conn
    r0, c0, _, _ = _bbox(all_pts)
    return {(r - r0, c - c0) for r, c in body}, {(r - r0, c - c0) for r, c in conn}


def _add_fragment(
    body: set[tuple[int, int]],
    conn: set[tuple[int, int]],
    rng: random.Random,
    side_hint: str,
) -> set[tuple[int, int]]:
    all_pts = body | conn
    r0, c0, r1, c1 = _bbox(all_pts)
    candidates: list[tuple[int, int]] = []
    for r in range(r0 - 3, r1 + 4):
        for c in range(c0 - 3, c1 + 4):
            if (r, c) in all_pts:
                continue
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in all_pts):
                continue
            if side_hint == "base":
                if r > r0 + 1:
                    continue
                if c > c1:
                    continue
            if side_hint == "plug":
                if r < r1 - 1:
                    continue
                if c < c0:
                    continue
            candidates.append((r, c))
    if not candidates:
        return body
    body = set(body)
    body.add(candidates[rng.randrange(len(candidates))])
    return body


def _place_sprite(
    grid: np.ndarray,
    body: set[tuple[int, int]],
    conn: set[tuple[int, int]],
    top: int,
    left: int,
    obj: int,
    ctrl: int,
) -> bool:
    h, w = grid.shape
    for r, c in body | conn:
        rr, cc = top + r, left + c
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if grid[rr, cc] != -1:
            return False
    for r, c in body:
        grid[top + r, left + c] = obj
    for r, c in conn:
        grid[top + r, left + c] = ctrl
    return True


def _expected_output(
    base_body: set[tuple[int, int]],
    base_conn: set[tuple[int, int]],
    plug_body: set[tuple[int, int]],
    plug_conn: set[tuple[int, int]],
    obj: int,
) -> np.ndarray:
    points = base_body | base_conn | plug_body | plug_conn
    r0, c0, r1, c1 = _bbox(points)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in points:
        out[r - r0, c - c0] = obj
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    obj, ctrl = rng.sample(colors, 2)

    conn_len = rng.randint(1, 3)
    base_data = _random_body_with_connector(rng, top_exposed=False, conn_len=conn_len)
    if base_data is None:
        return None
    base_mask, base_conn_list = base_data
    base_body = {(int(r), int(c)) for r, c in zip(*np.where(base_mask))}
    base_conn = set(base_conn_list)
    base_body -= base_conn
    if not base_body:
        return None

    plug_data = _make_plug_absolute(base_body, base_conn_list, rng)
    if plug_data is None:
        return None
    plug_body_abs, plug_conn_abs = plug_data

    if rng.random() < 0.45:
        base_body = _add_fragment(base_body, base_conn, rng, side_hint="base")
    if rng.random() < 0.45:
        plug_body_abs = _add_fragment(plug_body_abs, plug_conn_abs, rng, side_hint="plug")

    expected = _expected_output(base_body, base_conn, plug_body_abs, plug_conn_abs, obj)
    if expected.shape[0] < 3 or expected.shape[1] < 3:
        return None

    plug_body, plug_conn = _normalize_sprite(plug_body_abs, plug_conn_abs)
    base_body, base_conn = _normalize_sprite(base_body, base_conn)

    base_box = _bbox(base_body | base_conn)
    plug_box = _bbox(plug_body | plug_conn)
    base_h = base_box[2] - base_box[0] + 1
    base_w = base_box[3] - base_box[1] + 1
    plug_h = plug_box[2] - plug_box[0] + 1
    plug_w = plug_box[3] - plug_box[1] + 1

    min_grid_h = max(base_h + plug_h + 5, 13)
    min_grid_w = max(base_w + plug_w + 6, 14)
    grid_h = rng.randint(min_grid_h, max(22, min_grid_h))
    grid_w = rng.randint(min_grid_w, max(24, min_grid_w))
    grid = np.full((grid_h, grid_w), -1, dtype=int)

    base_top = rng.randint(0, max(0, grid_h // 3))
    base_left = rng.randint(0, max(0, grid_w // 3))
    plug_top_min = max(base_top + 3, grid_h // 2 - 1)
    plug_left_min = max(base_left + 3, grid_w // 2 - 1)
    if plug_top_min + plug_h > grid_h or plug_left_min + plug_w > grid_w:
        return None
    plug_top = rng.randint(plug_top_min, grid_h - plug_h)
    plug_left = rng.randint(plug_left_min, grid_w - plug_w)

    if not _place_sprite(grid, base_body, base_conn, base_top, base_left, obj, ctrl):
        return None
    if not _place_sprite(grid, plug_body, plug_conn, plug_top, plug_left, obj, ctrl):
        return None

    grid[grid == -1] = bg
    if _background_color(grid) != bg:
        return None
    if np.count_nonzero(grid == ctrl) >= np.count_nonzero(grid == obj):
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
