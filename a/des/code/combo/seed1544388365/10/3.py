import numpy as np
from collections import deque


RED = 2
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NON_RED_COLORS = [0, 1, 3, 4, 5, 6, 7, 8, 9]

GEN_RETRIES = 300
GEN_HEIGHT_RANGE = (16, 30)
GEN_WIDTH_RANGE = (16, 30)
GEN_LANE_COUNT_CHOICES = (2, 3)
GEN_TOTAL_SHAPES_RANGE = (4, 8)  # high is exclusive
GEN_MAX_LANE_LEN = 4
GEN_BAND_SIZE_RANGE = (3, 6)  # high is exclusive
GEN_MINOR_SIZE_RANGE = (1, 5)  # high is exclusive
GEN_MAJOR_SIZE_RANGE = (2, 6)  # high is exclusive
GEN_GAP_RANGE = (2, 5)  # gap between consecutive shape bodies


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _label_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _extract_shapes(grid: np.ndarray, bg: int) -> tuple[dict[int, dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    id_grid = np.full(g.shape, -1, dtype=int)
    shapes: dict[int, dict] = {}
    next_id = 0

    for color in np.unique(g):
        color = int(color)
        if color == bg or color == RED:
            continue
        for coords in _label_mask(g == color):
            shapes[next_id] = {"color": color, "coords": coords}
            id_grid[coords[:, 0], coords[:, 1]] = next_id
            next_id += 1

    assert shapes, "expected at least one non-background shape"
    return shapes, id_grid


def _parse_scene(grid: np.ndarray) -> tuple[int, dict[int, dict], dict[int, int | None], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = get_bg(g)
    shapes, id_grid = _extract_shapes(g, bg)

    shape_to_arrow: dict[int, dict] = {}
    for r, c in np.argwhere(g == RED):
        contacts: list[tuple[int, tuple[int, int]]] = []
        for dr, dc in DIR4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and id_grid[nr, nc] >= 0:
                contacts.append((int(id_grid[nr, nc]), (-dr, -dc)))

        assert contacts, "every red pixel must protrude from a shape"
        shape_ids = {sid for sid, _ in contacts}
        dirs = {direction for _, direction in contacts}
        assert len(shape_ids) == 1, "a red arrow may touch only one shape"
        assert len(dirs) == 1, "arrow direction must be unambiguous"

        sid = next(iter(shape_ids))
        assert sid not in shape_to_arrow, "each shape must have exactly one arrow"
        shape_to_arrow[sid] = {"pos": (int(r), int(c)), "dir": next(iter(dirs))}

    assert len(shape_to_arrow) == len(shapes), "each shape must carry one red arrow"

    outgoing: dict[int, int | None] = {sid: None for sid in shapes}
    incoming: dict[int, int] = {}
    for sid, arrow in shape_to_arrow.items():
        r, c = arrow["pos"]
        dr, dc = arrow["dir"]
        rr = r + dr
        cc = c + dc
        target = None
        while 0 <= rr < h and 0 <= cc < w:
            other = int(id_grid[rr, cc])
            if other >= 0 and other != sid:
                target = other
                break
            rr += dr
            cc += dc
        outgoing[sid] = target
        if target is not None:
            assert target not in incoming, "each shape may receive at most one arrow"
            incoming[target] = sid

    return bg, shapes, outgoing, incoming


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer each sender shape's color along its red arrow to the first hit shape.

    The red pixel determines a cardinal ray. The first other shape hit by that ray
    receives the sender's original color. Shapes with incoming arrows keep their
    outlines and adopt their predecessor's color, even if they also send onward.
    Shapes that send to another shape but receive no incoming color disappear.
    Red arrow pixels always disappear.
    """
    g = np.asarray(grid, dtype=int)
    bg, shapes, outgoing, incoming = _parse_scene(g)

    out = np.full_like(g, bg)
    for sid, shape in shapes.items():
        if sid in incoming:
            color = int(shapes[incoming[sid]]["color"])
        elif outgoing[sid] is not None:
            continue
        else:
            color = int(shape["color"])

        coords = shape["coords"]
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _scatter_slack(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    result = [0] * parts
    for _ in range(total):
        result[int(rng.integers(parts))] += 1
    return result


def _sample_bands(
    axis_len: int,
    band_sizes: list[int],
    rng: np.random.Generator,
    outer_margin: int = 1,
    min_gap: int = 2,
) -> list[tuple[int, int]] | None:
    required = 2 * outer_margin + sum(band_sizes) + min_gap * (len(band_sizes) - 1)
    if required > axis_len:
        return None

    slack = axis_len - required
    extras = _scatter_slack(slack, len(band_sizes) + 1, rng)
    pos = outer_margin + extras[0]
    bands: list[tuple[int, int]] = []
    for idx, size in enumerate(band_sizes):
        start = pos
        end = start + size - 1
        bands.append((start, end))
        pos = end + 1
        if idx < len(band_sizes) - 1:
            pos += min_gap + extras[idx + 1]
    assert pos + outer_margin + extras[-1] <= axis_len
    return bands


def _sample_lane_lengths(total_shapes: int, n_lanes: int, rng: np.random.Generator) -> list[int] | None:
    if total_shapes < n_lanes:
        return None
    lengths = [1] * n_lanes
    remaining = total_shapes - n_lanes
    while remaining > 0:
        choices = [idx for idx, length in enumerate(lengths) if length < GEN_MAX_LANE_LEN]
        if not choices:
            return None
        pick = int(choices[int(rng.integers(len(choices)))])
        lengths[pick] += 1
        remaining -= 1
    return lengths


def _place_horizontal_lane(
    grid: np.ndarray,
    band: tuple[int, int],
    colors: list[int],
    rng: np.random.Generator,
) -> list[dict] | None:
    top, bottom = band
    band_h = bottom - top + 1
    width = grid.shape[1]
    ray_r = int(rng.integers(top, bottom + 1))

    major = [int(rng.integers(*GEN_MAJOR_SIZE_RANGE)) for _ in colors]
    minor = [int(rng.integers(1, min(band_h, GEN_MINOR_SIZE_RANGE[1] - 1) + 1)) for _ in colors]
    gaps = [int(rng.integers(*GEN_GAP_RANGE)) for _ in range(max(0, len(colors) - 1))]
    span = sum(major) + sum(gaps)
    max_start = width - span - 1
    if max_start < 1:
        return None

    start_c = int(rng.integers(1, max_start + 1))
    direction = 1 if rng.random() < 0.5 else -1

    cols: list[int] = []
    cursor = start_c
    for idx, shape_w in enumerate(major):
        cols.append(cursor)
        cursor += shape_w
        if idx < len(gaps):
            cursor += gaps[idx]

    nodes: list[dict] = []
    for idx, color in enumerate(colors):
        shape_h = minor[idx]
        shape_w = major[idx]
        c0 = cols[idx]
        r0_lo = max(top, ray_r - shape_h + 1)
        r0_hi = min(ray_r, bottom - shape_h + 1)
        if r0_lo > r0_hi:
            return None
        r0 = int(rng.integers(r0_lo, r0_hi + 1))
        arrow_c = c0 + shape_w if direction == 1 else c0 - 1
        if not (0 <= arrow_c < width):
            return None
        grid[r0 : r0 + shape_h, c0 : c0 + shape_w] = int(color)
        grid[ray_r, arrow_c] = RED
        nodes.append({"color": int(color), "target_color": None})

    ordered = list(range(len(colors))) if direction == 1 else list(range(len(colors) - 1, -1, -1))
    for idx, src in enumerate(ordered):
        if idx + 1 < len(ordered):
            nodes[src]["target_color"] = int(colors[ordered[idx + 1]])
    return nodes


def _place_vertical_lane(
    grid: np.ndarray,
    band: tuple[int, int],
    colors: list[int],
    rng: np.random.Generator,
) -> list[dict] | None:
    left, right = band
    band_w = right - left + 1
    height = grid.shape[0]
    ray_c = int(rng.integers(left, right + 1))

    major = [int(rng.integers(*GEN_MAJOR_SIZE_RANGE)) for _ in colors]
    minor = [int(rng.integers(1, min(band_w, GEN_MINOR_SIZE_RANGE[1] - 1) + 1)) for _ in colors]
    gaps = [int(rng.integers(*GEN_GAP_RANGE)) for _ in range(max(0, len(colors) - 1))]
    span = sum(major) + sum(gaps)
    max_start = height - span - 1
    if max_start < 1:
        return None

    start_r = int(rng.integers(1, max_start + 1))
    direction = 1 if rng.random() < 0.5 else -1

    rows: list[int] = []
    cursor = start_r
    for idx, shape_h in enumerate(major):
        rows.append(cursor)
        cursor += shape_h
        if idx < len(gaps):
            cursor += gaps[idx]

    nodes: list[dict] = []
    for idx, color in enumerate(colors):
        shape_h = major[idx]
        shape_w = minor[idx]
        r0 = rows[idx]
        c0_lo = max(left, ray_c - shape_w + 1)
        c0_hi = min(ray_c, right - shape_w + 1)
        if c0_lo > c0_hi:
            return None
        c0 = int(rng.integers(c0_lo, c0_hi + 1))
        arrow_r = r0 + shape_h if direction == 1 else r0 - 1
        if not (0 <= arrow_r < height):
            return None
        grid[r0 : r0 + shape_h, c0 : c0 + shape_w] = int(color)
        grid[arrow_r, ray_c] = RED
        nodes.append({"color": int(color), "target_color": None})

    ordered = list(range(len(colors))) if direction == 1 else list(range(len(colors) - 1, -1, -1))
    for idx, src in enumerate(ordered):
        if idx + 1 < len(ordered):
            nodes[src]["target_color"] = int(colors[ordered[idx + 1]])
    return nodes


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    axis = "H" if rng.random() < 0.5 else "V"
    h = int(rng.integers(*GEN_HEIGHT_RANGE))
    w = int(rng.integers(*GEN_WIDTH_RANGE))
    bg = int(NON_RED_COLORS[int(rng.integers(len(NON_RED_COLORS)))])

    n_lanes = int(GEN_LANE_COUNT_CHOICES[int(rng.integers(len(GEN_LANE_COUNT_CHOICES)))])
    total_shapes = int(rng.integers(*GEN_TOTAL_SHAPES_RANGE))
    lengths = _sample_lane_lengths(total_shapes, n_lanes, rng)
    if lengths is None:
        return None

    band_sizes = [int(rng.integers(*GEN_BAND_SIZE_RANGE)) for _ in range(n_lanes)]
    bands = _sample_bands(h if axis == "H" else w, band_sizes, rng)
    if bands is None:
        return None

    palette = [color for color in NON_RED_COLORS if color != bg]
    if total_shapes > len(palette):
        return None
    picked = rng.choice(np.array(palette, dtype=int), size=total_shapes, replace=False)
    colors = [int(x) for x in picked]

    grid = np.full((h, w), bg, dtype=int)
    planned_nodes: list[dict] = []
    offset = 0
    for lane_idx in range(n_lanes):
        lane_colors = colors[offset : offset + lengths[lane_idx]]
        offset += lengths[lane_idx]
        if axis == "H":
            nodes = _place_horizontal_lane(grid, bands[lane_idx], lane_colors, rng)
        else:
            nodes = _place_vertical_lane(grid, bands[lane_idx], lane_colors, rng)
        if nodes is None:
            return None
        planned_nodes.extend(nodes)

    if not any(node["target_color"] is not None for node in planned_nodes):
        return None
    if int(np.count_nonzero(grid == bg)) <= grid.size // 2:
        return None

    try:
        parsed_bg, shapes, outgoing, _incoming = _parse_scene(grid)
    except AssertionError:
        return None

    if parsed_bg != bg or len(shapes) != len(planned_nodes):
        return None

    color_to_sid = {int(shape["color"]): sid for sid, shape in shapes.items()}
    if set(color_to_sid) != {node["color"] for node in planned_nodes}:
        return None

    for node in planned_nodes:
        sid = color_to_sid[int(node["color"])]
        target_sid = outgoing[sid]
        parsed_target_color = None if target_sid is None else int(shapes[target_sid]["color"])
        if parsed_target_color != node["target_color"]:
            return None

    out = main(grid)
    if out.shape != grid.shape or np.any(out == RED) or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
