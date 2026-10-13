import numpy as np


BG = 0
YELLOW = 4
ORANGE = 7
MARKER_COLORS = {YELLOW, ORANGE}
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


GEN_RETRIES = 300
GEN_OBJECT_COUNT_RANGE = (2, 5)
GEN_SIDE_BASE = 12
GEN_SIDE_PER_OBJECT = 3
GEN_SIDE_MAX = 29
GEN_MARGIN = 1
GEN_PLACEMENT_TRIES = 120
GEN_SHAPE_TRIES = 80
GEN_OBJECT_COLORS = [1, 2, 3, 5, 6, 8, 9]
GEN_LINE_LENGTHS = [2, 3, 4, 5]
GEN_BASE_SHAPES = [
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 1), (0, 2), (1, 0), (1, 1), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
]


def _neighbors4(pos: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pos
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _normalize(coords) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    min_r = min(rows)
    min_c = min(cols)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _orient_shape(coords, flip_lr: bool, rot_k: int) -> tuple[tuple[int, int], ...]:
    pts = list(coords)
    for _ in range(rot_k % 4):
        pts = [(-c, r) for r, c in pts]
    if flip_lr:
        pts = [(r, -c) for r, c in pts]
    return _normalize(pts)


def _shape_dims(coords) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return max(rows) + 1, max(cols) + 1


def _translate(coords, dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in coords}


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in _neighbors4(cur):
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(coords)


def _mirror_within_bbox(
    coords, bbox_shape: tuple[int, int], direction: tuple[int, int]
) -> set[tuple[int, int]]:
    h, w = bbox_shape
    if direction[0] == 0:
        return {(h - 1 - r, c) for r, c in coords}
    return {(r, w - 1 - c) for r, c in coords}


def _attachment_cells(
    shape: tuple[tuple[int, int], ...], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    shape_set = set(shape)
    h, w = _shape_dims(shape)
    dr, dc = direction
    result = []
    for r, c in shape:
        if (r + dr, c + dc) in shape_set:
            continue
        rr, cc = r + dr, c + dc
        ray_clear = True
        while 0 <= rr < h and 0 <= cc < w:
            if (rr, cc) in shape_set:
                ray_clear = False
                break
            rr += dr
            cc += dc
        if not ray_clear:
            continue
        off_axis = False
        for nr, nc in _neighbors4((r, c)):
            if (nr, nc) not in shape_set:
                continue
            if dr != 0 and nc != c:
                off_axis = True
            if dc != 0 and nr != r:
                off_axis = True
        if off_axis:
            result.append((r, c))
    return result


def _split_body_and_line(
    component: set[tuple[int, int]],
    marker: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[set[tuple[int, int]], list[tuple[int, int]], int]:
    dr, dc = direction
    run = []
    cur = (marker[0] - dr, marker[1] - dc)
    while cur in component:
        run.append(cur)
        cur = (cur[0] - dr, cur[1] - dc)
    assert len(run) >= 2, "projection line must include at least one body-adjacent segment"

    candidates = []
    removed = set()
    for idx, cell in enumerate(run[:-1]):
        removed.add(cell)
        attach = run[idx + 1]
        body = component - removed
        if not (3 <= len(body) <= 5):
            continue
        if not _is_connected(body):
            continue

        removed_list = run[: idx + 1]
        ok = True
        for j, pt in enumerate(removed_list):
            allowed = {
                marker if j == 0 else removed_list[j - 1],
                attach if j == idx else removed_list[j + 1],
            }
            for nb in _neighbors4(pt):
                if nb in component and nb not in allowed:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        off_axis = False
        for nb in _neighbors4(attach):
            if nb not in body or nb == removed_list[-1]:
                continue
            d_nb = (nb[0] - attach[0], nb[1] - attach[1])
            if d_nb not in {direction, (-dr, -dc)}:
                off_axis = True
                break
        if not off_axis:
            continue

        candidates.append((idx + 1, body))

    assert len(candidates) == 1, "object-line split must be unique"
    line_color_len, body = candidates[0]
    return body, run[:line_color_len], line_color_len + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each colored projection line to place a copied or mirrored shadow of its object, then erase the line.

    Input structure:
    - The background is black.
    - Each non-marker color forms one small 4-connected object together with a same-colored straight projection line.
    - Every projection line ends in a yellow or orange terminal marker, and each object color appears exactly once.
    - The underlying object body occupies 3-5 cells, while the projection line is horizontal or vertical.

    Rule:
    - For each terminal marker, infer the line direction from the adjacent same-colored cell and recover the object body by removing the straight line segment.
    - Let L be the full projection-line length including the terminal marker.
    - Place a copy of the object body translated by L cells in the line direction.
    - If the terminal marker is yellow, keep the copy unchanged; if it is orange, mirror the copy within the object's bbox.
    - For horizontal projection lines, the orange case flips the body up-down; for vertical projection lines, it flips the body left-right.
    - Keep the original objects and erase every projection line and terminal marker to black.

    Key insight:
    - The endpoint marker color determines whether the shifted replica is preserved or mirrored, while the line geometry supplies both the translation direction and its distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    object_colors = set(int(v) for v in np.unique(g)) - {BG} - MARKER_COLORS
    markers = [tuple(map(int, pos)) for pos in np.argwhere(np.isin(g, list(MARKER_COLORS)))]
    assert markers, "expected at least one projection marker"
    assert len(markers) == len(object_colors), "each object needs one unique marker"

    infos = []
    seen_colors = set()
    for marker in markers:
        r, c = marker
        marker_color = int(g[r, c])
        assert marker_color in MARKER_COLORS

        colored_neighbors = []
        for nr, nc in _neighbors4(marker):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                value = int(g[nr, nc])
                if value not in MARKER_COLORS and value != BG:
                    colored_neighbors.append((nr, nc, value))
        assert len(colored_neighbors) == 1, "marker must touch exactly one line color"

        nr, nc, color = colored_neighbors[0]
        assert color not in seen_colors, "each object color should have one marker"
        seen_colors.add(color)

        direction = (r - nr, c - nc)
        assert direction in DIRS4

        component = {
            tuple(map(int, pos))
            for pos in np.argwhere(g == color)
        }
        body, _line_cells, line_length = _split_body_and_line(component, marker, direction)
        rows = [rr for rr, _ in body]
        cols = [cc for _, cc in body]
        r0, c0 = min(rows), min(cols)
        r1, c1 = max(rows), max(cols)
        rel_body = {(rr - r0, cc - c0) for rr, cc in body}
        bbox_shape = (r1 - r0 + 1, c1 - c0 + 1)

        infos.append(
            {
                "color": color,
                "body": body,
                "rel_body": rel_body,
                "bbox_origin": (r0, c0),
                "bbox_shape": bbox_shape,
                "direction": direction,
                "line_length": line_length,
                "marker_color": marker_color,
            }
        )

    assert seen_colors == object_colors

    out = np.full_like(g, BG)
    for info in infos:
        for r, c in info["body"]:
            out[r, c] = info["color"]

    for info in infos:
        copy_rel = set(info["rel_body"])
        if info["marker_color"] == ORANGE:
            copy_rel = _mirror_within_bbox(copy_rel, info["bbox_shape"], info["direction"])
        dr, dc = info["direction"]
        shift_r = dr * info["line_length"]
        shift_c = dc * info["line_length"]
        base_r, base_c = info["bbox_origin"]
        for rr, cc in copy_rel:
            ar = base_r + shift_r + rr
            ac = base_c + shift_c + cc
            assert 0 <= ar < g.shape[0] and 0 <= ac < g.shape[1], "shadow must stay in bounds"
            assert out[ar, ac] == BG, "shadows must not overlap other objects"
            out[ar, ac] = info["color"]

    return out


def _expand8(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _build_expected(
    placements: list[dict[str, object]], h: int, w: int
) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    for item in placements:
        color = int(item["color"])
        for r, c in item["body_abs"]:
            assert out[r, c] == BG
            out[r, c] = color
    for item in placements:
        color = int(item["color"])
        for r, c in item["copy_abs"]:
            assert out[r, c] == BG
            out[r, c] = color
    return out


def _sample_object_spec(
    rng: np.random.Generator,
    direction: tuple[int, int],
    marker_color: int,
):
    for _ in range(GEN_SHAPE_TRIES):
        base = GEN_BASE_SHAPES[int(rng.integers(len(GEN_BASE_SHAPES)))]
        shape = _orient_shape(
            base,
            bool(rng.integers(2)),
            int(rng.integers(4)),
        )
        attachments = _attachment_cells(shape, direction)
        if not attachments:
            continue

        bbox_shape = _shape_dims(shape)
        copy_rel = set(shape)
        if marker_color == ORANGE:
            mirrored = _mirror_within_bbox(copy_rel, bbox_shape, direction)
            if mirrored == copy_rel:
                continue
            copy_rel = mirrored

        dr, dc = direction
        valid_lengths = []
        for length in GEN_LINE_LENGTHS:
            shifted_copy = _translate(copy_rel, dr * length, dc * length)
            if set(shape).isdisjoint(shifted_copy):
                valid_lengths.append(length)
        if not valid_lengths:
            continue

        attach = attachments[int(rng.integers(len(attachments)))]
        length = int(valid_lengths[int(rng.integers(len(valid_lengths)))])
        return shape, attach, length, copy_rel
    return None


def _try_place_object(
    rng: np.random.Generator,
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    color: int,
    direction: tuple[int, int],
    marker_color: int,
):
    spec = _sample_object_spec(rng, direction, marker_color)
    if spec is None:
        return None
    shape, attach, length, copy_rel = spec
    dr, dc = direction

    body_rel = set(shape)
    line_rel = {(attach[0] + dr * k, attach[1] + dc * k) for k in range(1, length)}
    marker_rel = {(attach[0] + dr * length, attach[1] + dc * length)}
    copy_shifted_rel = _translate(copy_rel, dr * length, dc * length)
    full_rel = body_rel | line_rel | marker_rel | copy_shifted_rel

    rows = [r for r, _ in full_rel]
    cols = [c for _, c in full_rel]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    r_lo = GEN_MARGIN - min_r
    r_hi = grid.shape[0] - GEN_MARGIN - 1 - max_r
    c_lo = GEN_MARGIN - min_c
    c_hi = grid.shape[1] - GEN_MARGIN - 1 - max_c
    if r_lo > r_hi or c_lo > c_hi:
        return None

    for _ in range(GEN_PLACEMENT_TRIES):
        base_r = int(rng.integers(r_lo, r_hi + 1))
        base_c = int(rng.integers(c_lo, c_hi + 1))

        body_abs = _translate(body_rel, base_r, base_c)
        line_abs = _translate(line_rel, base_r, base_c)
        marker_abs = _translate(marker_rel, base_r, base_c)
        copy_abs = _translate(copy_shifted_rel, base_r, base_c)
        footprint = body_abs | line_abs | marker_abs | copy_abs

        if any(cell in blocked for cell in footprint):
            continue
        if body_abs & copy_abs:
            continue

        for r, c in body_abs:
            grid[r, c] = color
        for r, c in line_abs:
            grid[r, c] = color
        for r, c in marker_abs:
            grid[r, c] = marker_color

        return {
            "color": color,
            "body_abs": body_abs,
            "copy_abs": copy_abs,
            "footprint": footprint,
        }
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    object_count = int(rng.integers(*GEN_OBJECT_COUNT_RANGE))
    side_low = GEN_SIDE_BASE + GEN_SIDE_PER_OBJECT * object_count
    h = int(rng.integers(side_low, GEN_SIDE_MAX + 1))
    w = int(rng.integers(side_low, GEN_SIDE_MAX + 1))
    grid = np.full((h, w), BG, dtype=int)

    colors = rng.choice(np.array(GEN_OBJECT_COLORS, dtype=int), size=object_count, replace=False).tolist()

    directions = list(rng.permutation(np.array(DIRS4, dtype=object))[:object_count])
    directions = [tuple(map(int, direction)) for direction in directions]

    marker_colors = [YELLOW] * (object_count // 2) + [ORANGE] * (object_count // 2)
    if object_count % 2 == 1:
        marker_colors.append(YELLOW if int(rng.integers(2)) == 0 else ORANGE)
    marker_colors = list(rng.permutation(np.array(marker_colors, dtype=int)))

    blocked: set[tuple[int, int]] = set()
    placements = []
    for color, direction, marker_color in zip(colors, directions, marker_colors):
        placed = _try_place_object(rng, grid, blocked, int(color), direction, int(marker_color))
        if placed is None:
            return None
        placements.append(placed)
        blocked |= _expand8(set(placed["footprint"]), h, w)

    expected = _build_expected(placements, h, w)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
