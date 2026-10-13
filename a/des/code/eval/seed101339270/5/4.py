import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
GREY = 5
PALETTE = (2, 4, 6, 7, 8, 9)

_RNG = np.random.default_rng()
_GEN_RETRIES = 500


def _neighbors4(r: int, c: int):
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _border_coords(top: int, left: int, height: int, width: int):
    coords = []
    bottom = top + height - 1
    right = left + width - 1
    for c in range(left, right + 1):
        coords.append((top, c))
        coords.append((bottom, c))
    for r in range(top + 1, bottom):
        coords.append((r, left))
        coords.append((r, right))
    return coords


def _find_frames(grid: np.ndarray, border_color: int, interior_must_be_black: bool):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frames = []
    for outer_h in range(5, 8):
        for outer_w in range(5, 8):
            for top in range(h - outer_h + 1):
                for left in range(w - outer_w + 1):
                    bottom = top + outer_h - 1
                    right = left + outer_w - 1
                    sub = g[top : bottom + 1, left : right + 1]
                    if not (
                        np.all(sub[0, :] == border_color)
                        and np.all(sub[-1, :] == border_color)
                        and np.all(sub[:, 0] == border_color)
                        and np.all(sub[:, -1] == border_color)
                    ):
                        continue
                    inner = sub[1:-1, 1:-1]
                    if interior_must_be_black:
                        if not np.all(inner == BLACK):
                            continue
                    else:
                        if not np.any(inner != BLACK):
                            continue
                    frames.append(
                        {
                            "top": top,
                            "left": left,
                            "bottom": bottom,
                            "right": right,
                            "inner": inner.copy(),
                            "shape": inner.shape,
                            "border": _border_coords(top, left, outer_h, outer_w),
                        }
                    )

    frames.sort(key=lambda item: (item["top"], item["left"], item["bottom"], item["right"]))

    occupied = np.zeros_like(g, dtype=bool)
    for idx, frame in enumerate(frames):
        top = frame["top"]
        left = frame["left"]
        bottom = frame["bottom"]
        right = frame["right"]
        region = occupied[top : bottom + 1, left : right + 1]
        assert not region.any(), "framed objects must not overlap"
        occupied[top : bottom + 1, left : right + 1] = True
        frame["id"] = idx
    return frames


def _collect_component(mask: np.ndarray, start: tuple[int, int], seen: np.ndarray):
    h, w = mask.shape
    stack = [start]
    seen[start] = True
    comp = []
    while stack:
        r, c = stack.pop()
        comp.append((r, c))
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                stack.append((rr, cc))
    return comp


def _trace_component(
    component,
    grid: np.ndarray,
    source_border_owner: dict[tuple[int, int], int],
    receiver_border_owner: dict[tuple[int, int], int],
):
    comp_set = set(component)
    neighbors = {}
    for cell in component:
        r, c = cell
        adj = [nb for nb in _neighbors4(r, c) if nb in comp_set]
        assert len(adj) <= 2, "paths must be simple non-branching chains"
        neighbors[cell] = adj

    source_ids = set()
    receiver_ids = set()
    source_touch_cells = []
    receiver_touch_cells = []

    for cell in component:
        sids = {source_border_owner[nb] for nb in _neighbors4(*cell) if nb in source_border_owner}
        rids = {receiver_border_owner[nb] for nb in _neighbors4(*cell) if nb in receiver_border_owner}
        if sids:
            assert len(sids) == 1, "a path cell cannot touch multiple sources"
            source_ids.update(sids)
            source_touch_cells.append(cell)
        if rids:
            assert len(rids) == 1, "a path cell cannot touch multiple receivers"
            receiver_ids.update(rids)
            receiver_touch_cells.append(cell)

    assert len(source_ids) == 1 and len(receiver_ids) == 1, "each path must connect one source and one receiver"
    assert len(source_touch_cells) == 1, "path must touch its source at exactly one cell"
    assert len(receiver_touch_cells) == 1, "path must touch its receiver at exactly one cell"

    start = source_touch_cells[0]
    end = receiver_touch_cells[0]

    if len(component) == 1:
        assert start == end
        ordered = [start]
    else:
        assert len(neighbors[start]) == 1, "path start must be an endpoint"
        ordered = []
        prev = None
        cur = start
        while True:
            ordered.append(cur)
            nxt = [nb for nb in neighbors[cur] if nb != prev]
            if not nxt:
                break
            assert len(nxt) == 1, "path traversal must stay unambiguous"
            prev, cur = cur, nxt[0]
        assert set(ordered) == comp_set, "path traversal must visit the full component"
        assert ordered[-1] == end, "path order must run from source to receiver"

    filters = [int(grid[r, c]) for r, c in ordered if int(grid[r, c]) != GREY]
    return {
        "source_id": next(iter(source_ids)),
        "receiver_id": next(iter(receiver_ids)),
        "filters": filters,
    }


def _apply_filters(pattern: np.ndarray, filters):
    out = np.array(pattern, copy=True)
    if not filters:
        return out
    if len(filters) == 1:
        out[out == filters[0]] = BLACK
        return out
    for idx, color in enumerate(filters):
        replacement = filters[(idx + 1) % len(filters)]
        out[out == color] = replacement
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each green-framed source to its blue-framed receiver through a grey path,
    read the ordered filter colors along that path, and apply the resulting sequential
    recolor operations to the source pattern before writing it into the receiver.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sources = _find_frames(g, GREEN, interior_must_be_black=False)
    receivers = _find_frames(g, BLUE, interior_must_be_black=True)
    assert sources and receivers
    assert len(sources) == len(receivers)

    frame_mask = np.zeros_like(g, dtype=bool)
    source_border_owner = {}
    receiver_border_owner = {}

    for frame in sources + receivers:
        top = frame["top"]
        left = frame["left"]
        bottom = frame["bottom"]
        right = frame["right"]
        region = frame_mask[top : bottom + 1, left : right + 1]
        assert not region.any(), "framed objects must be disjoint"
        frame_mask[top : bottom + 1, left : right + 1] = True

    for frame in sources:
        for cell in frame["border"]:
            source_border_owner[cell] = frame["id"]
    for frame in receivers:
        for cell in frame["border"]:
            receiver_border_owner[cell] = frame["id"]

    path_mask = (~frame_mask) & (g != BLACK)
    seen = np.zeros_like(path_mask, dtype=bool)
    links = []
    for r, c in np.argwhere(path_mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        component = _collect_component(path_mask, (rr, cc), seen)
        links.append(_trace_component(component, g, source_border_owner, receiver_border_owner))

    assert len(links) == len(sources), "each source must have exactly one path"
    assert {item["source_id"] for item in links} == {frame["id"] for frame in sources}
    assert {item["receiver_id"] for item in links} == {frame["id"] for frame in receivers}

    out = g.copy()
    source_by_id = {frame["id"]: frame for frame in sources}
    receiver_by_id = {frame["id"]: frame for frame in receivers}
    for link in links:
        source = source_by_id[link["source_id"]]
        receiver = receiver_by_id[link["receiver_id"]]
        assert source["shape"] == receiver["shape"], "matched source and receiver must have the same size"
        transformed = _apply_filters(source["inner"], link["filters"])
        out[
            receiver["top"] + 1 : receiver["bottom"],
            receiver["left"] + 1 : receiver["right"],
        ] = transformed

    return out


def _line(a: tuple[int, int], b: tuple[int, int]):
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    assert c0 == c1
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _make_pattern(rng: np.random.Generator, h: int, w: int, required_colors):
    required = [int(c) for c in required_colors]
    extra_pool = [c for c in PALETTE if c not in required]
    min_colors = max(2, len(required))
    max_colors = min(len(PALETTE), len(required) + 2)
    total_colors = int(rng.integers(min_colors, max_colors + 1))
    extras_needed = total_colors - len(required)
    extras = []
    if extras_needed > 0:
        extras = [int(c) for c in rng.choice(extra_pool, size=extras_needed, replace=False)]
    colors = [int(c) for c in rng.permutation(required + extras)]

    style = int(rng.integers(0, 5))
    offset = int(rng.integers(0, len(colors)))
    row_step = int(rng.integers(1, 3))
    col_step = int(rng.integers(1, 3))
    stripe_h = int(rng.integers(1, 3))
    stripe_w = int(rng.integers(1, 3))

    pattern = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            if style == 0:
                idx = (r + c + offset) % len(colors)
            elif style == 1:
                idx = (row_step * r + col_step * c + offset) % len(colors)
            elif style == 2:
                block = (2 * r // max(1, h)) + 2 * (2 * c // max(1, w))
                idx = (block + offset) % len(colors)
            elif style == 3:
                idx = ((r // stripe_h) + (c // stripe_w) + offset) % len(colors)
            else:
                idx = ((r % 2) * 2 + (c % 2) + offset) % len(colors)
            pattern[r, c] = colors[idx]

    if len(set(map(int, pattern.ravel()))) < 2:
        pattern[0, 0] = colors[0]
        pattern[-1, -1] = colors[1 % len(colors)]

    all_positions = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(all_positions)
    for color, pos in zip(required, all_positions):
        pattern[pos] = color

    return pattern


def _apply_symmetry(grid: np.ndarray, code: int):
    if code == 0:
        return np.array(grid, copy=True)
    if code == 1:
        return np.rot90(grid, 1).copy()
    if code == 2:
        return np.rot90(grid, 2).copy()
    if code == 3:
        return np.rot90(grid, 3).copy()
    if code == 4:
        return np.fliplr(grid).copy()
    if code == 5:
        return np.flipud(grid).copy()
    if code == 6:
        return np.transpose(grid).copy()
    return np.fliplr(np.rot90(grid, 1)).copy()


def _gen_attempt():
    rng = _RNG
    pair_count = int(rng.integers(2, 4))
    filter_counts = [int(x) for x in rng.permutation(np.array([0, 1, 2, 3]))[:pair_count]]

    specs = []
    max_outer_w = 0
    band_heights = []
    for count in filter_counts:
        inner_h = int(rng.integers(3, 6))
        inner_w = int(rng.integers(3, 6))
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        filters = []
        if count > 0:
            filters = [int(c) for c in rng.choice(PALETTE, size=count, replace=False)]
        pattern = _make_pattern(rng, inner_h, inner_w, filters)
        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "filters": filters,
                "pattern": pattern,
            }
        )
        band_heights.append(outer_h + int(rng.integers(3, 5)))
        max_outer_w = max(max_outer_w, outer_w)

    width = max(28, 2 * max_outer_w + int(rng.integers(14, 22)))
    height = 2 + sum(band_heights) + (pair_count - 1)

    grid = np.full((height, width), BLACK, dtype=int)
    expected = np.full_like(grid, BLACK)

    band_top = 1
    for idx, spec in enumerate(specs):
        band_h = band_heights[idx]
        outer_h = spec["outer_h"]
        outer_w = spec["outer_w"]
        inner_h = spec["inner_h"]
        inner_w = spec["inner_w"]

        offset_limit = band_h - outer_h - 1
        assert offset_limit >= 2
        source_top = band_top + int(rng.integers(1, offset_limit + 1))

        straight = bool(rng.random() < 0.4)
        if straight:
            receiver_top = source_top
        else:
            receiver_choices = [band_top + off for off in range(1, offset_limit + 1) if band_top + off != source_top]
            receiver_top = int(rng.choice(receiver_choices))

        source_left = int(rng.integers(1, 4))
        receiver_left = width - outer_w - int(rng.integers(1, 4))

        source_right = source_left + outer_w - 1
        receiver_right = receiver_left + outer_w - 1
        assert receiver_left - source_right >= 6

        source_slice = (slice(source_top, source_top + outer_h), slice(source_left, source_left + outer_w))
        receiver_slice = (slice(receiver_top, receiver_top + outer_h), slice(receiver_left, receiver_left + outer_w))
        assert np.all(grid[source_slice] == BLACK)
        assert np.all(grid[receiver_slice] == BLACK)

        grid[source_slice] = BLACK
        grid[receiver_slice] = BLACK
        expected[source_slice] = BLACK
        expected[receiver_slice] = BLACK

        grid[source_top : source_top + outer_h, source_left] = GREEN
        grid[source_top : source_top + outer_h, source_right] = GREEN
        grid[source_top, source_left : source_right + 1] = GREEN
        grid[source_top + outer_h - 1, source_left : source_right + 1] = GREEN
        grid[
            source_top + 1 : source_top + outer_h - 1,
            source_left + 1 : source_right,
        ] = spec["pattern"]

        expected[source_top : source_top + outer_h, source_left] = GREEN
        expected[source_top : source_top + outer_h, source_right] = GREEN
        expected[source_top, source_left : source_right + 1] = GREEN
        expected[source_top + outer_h - 1, source_left : source_right + 1] = GREEN
        expected[
            source_top + 1 : source_top + outer_h - 1,
            source_left + 1 : source_right,
        ] = spec["pattern"]

        grid[receiver_top : receiver_top + outer_h, receiver_left] = BLUE
        grid[receiver_top : receiver_top + outer_h, receiver_right] = BLUE
        grid[receiver_top, receiver_left : receiver_right + 1] = BLUE
        grid[receiver_top + outer_h - 1, receiver_left : receiver_right + 1] = BLUE

        expected[receiver_top : receiver_top + outer_h, receiver_left] = BLUE
        expected[receiver_top : receiver_top + outer_h, receiver_right] = BLUE
        expected[receiver_top, receiver_left : receiver_right + 1] = BLUE
        expected[receiver_top + outer_h - 1, receiver_left : receiver_right + 1] = BLUE
        expected[
            receiver_top + 1 : receiver_top + outer_h - 1,
            receiver_left + 1 : receiver_right,
        ] = _apply_filters(spec["pattern"], spec["filters"])

        if straight:
            port_offset = int(rng.integers(0, inner_h))
            source_port_row = source_top + 1 + port_offset
            receiver_port_row = receiver_top + 1 + port_offset
        else:
            source_port_row = source_top + 1 + int(rng.integers(0, inner_h))
            receiver_port_row = receiver_top + 1 + int(rng.integers(0, inner_h))
            if source_port_row == receiver_port_row:
                receiver_port_row += 1
                if receiver_port_row >= receiver_top + outer_h - 1:
                    receiver_port_row = receiver_top + 1
                assert source_port_row != receiver_port_row

        start = (source_port_row, source_right + 1)
        end = (receiver_port_row, receiver_left - 1)
        assert 0 <= start[1] < width and 0 <= end[1] < width

        if straight:
            path_cells = _line(start, end)
        else:
            mid_col = int(rng.integers(start[1] + 1, end[1]))
            path_cells = _line(start, (start[0], mid_col))
            path_cells += _line((start[0], mid_col), (end[0], mid_col))[1:]
            path_cells += _line((end[0], mid_col), end)[1:]

        assert len(path_cells) >= 3
        for r, c in path_cells:
            assert grid[r, c] == BLACK

        path_colors = [GREY] * len(path_cells)
        if spec["filters"]:
            slots = np.arange(1, len(path_cells) - 1)
            chosen = np.sort(rng.choice(slots, size=len(spec["filters"]), replace=False))
            for path_idx, color in zip(chosen, spec["filters"]):
                path_colors[int(path_idx)] = int(color)

        for (r, c), color in zip(path_cells, path_colors):
            grid[r, c] = color
            expected[r, c] = color

        band_top += band_h + 1

    symmetry = int(rng.integers(0, 8))
    transformed_input = _apply_symmetry(grid, symmetry)
    transformed_expected = _apply_symmetry(expected, symmetry)
    if not np.array_equal(main(transformed_input), transformed_expected):
        return None
    return transformed_input


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
