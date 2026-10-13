import random
from collections import Counter

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9

    NOT_BLACK = [BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude=None) -> list[int]:
    colors = set(map(int, grid.flatten()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


QUADS = ("TL", "TR", "BL", "BR")
H_MATE = {"TL": "BL", "TR": "BR", "BL": "TL", "BR": "TR"}
V_MATE = {"TL": "TR", "TR": "TL", "BL": "BR", "BR": "BL"}


def _find_mirror_lines(grid: np.ndarray) -> tuple[int, int]:
    grey = Color.GREY
    rows = np.where(np.all(grid == grey, axis=1))[0]
    cols = np.where(np.all(grid == grey, axis=0))[0]
    assert len(rows) == 1 and len(cols) == 1, "expected one full grey row and one full grey column"
    return int(rows[0]), int(cols[0])


def _quadrant_slices(hr: int, vc: int) -> dict[str, tuple[slice, slice]]:
    return {
        "TL": (slice(0, hr), slice(0, vc)),
        "TR": (slice(0, hr), slice(vc + 1, None)),
        "BL": (slice(hr + 1, None), slice(0, vc)),
        "BR": (slice(hr + 1, None), slice(vc + 1, None)),
    }


def _quadrant_pixel_counts(grid: np.ndarray, color: int, hr: int, vc: int) -> dict[str, int]:
    counts = {}
    for quad, (rs, cs) in _quadrant_slices(hr, vc).items():
        counts[quad] = int(np.count_nonzero(grid[rs, cs] == color))
    return counts


def _coords_in_quadrant(grid: np.ndarray, color: int, quad: str, hr: int, vc: int) -> np.ndarray:
    rs, cs = _quadrant_slices(hr, vc)[quad]
    sub = np.argwhere(grid[rs, cs] == color)
    if sub.size == 0:
        return sub.reshape(0, 2)
    row_offset = 0 if rs.start is None else rs.start
    col_offset = 0 if cs.start is None else cs.start
    sub[:, 0] += row_offset
    sub[:, 1] += col_offset
    return sub


def _reflect_horizontal(coords: np.ndarray, hr: int) -> np.ndarray:
    out = coords.copy()
    out[:, 0] = 2 * hr - out[:, 0]
    return out


def _reflect_vertical(coords: np.ndarray, vc: int) -> np.ndarray:
    out = coords.copy()
    out[:, 1] = 2 * vc - out[:, 1]
    return out


def _infer_source_role(grid: np.ndarray) -> tuple[int, str, int, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    hr, vc = _find_mirror_lines(g)

    candidates: list[tuple[int, str]] = []
    for color in colors_present(g, exclude={bg, Color.GREY}):
        counts = _quadrant_pixel_counts(g, color, hr, vc)
        occupied = [quad for quad, count in counts.items() if count > 0]
        singleton = [quad for quad, count in counts.items() if count == 1]
        if len(occupied) == 3 and len(singleton) == 1 and all(counts[q] > 1 for q in occupied if q != singleton[0]):
            candidates.append((int(color), singleton[0]))

    assert len(candidates) == 1, "expected exactly one color with a singleton source anchor and two partial mirrors"
    color, source_quad = candidates[0]
    return color, source_quad, hr, vc


def _solve_from_parts(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    source_color, source_quad, hr, vc = _infer_source_role(g)
    h_quad = H_MATE[source_quad]
    v_quad = V_MATE[source_quad]

    h_partial = _coords_in_quadrant(g, source_color, h_quad, hr, vc)
    v_partial = _coords_in_quadrant(g, source_color, v_quad, hr, vc)
    assert h_partial.shape[0] > 1 and v_partial.shape[0] > 1

    from_h = _reflect_horizontal(h_partial, hr)
    from_v = _reflect_vertical(v_partial, vc)
    merged = np.vstack([from_h, from_v])
    merged = np.unique(merged, axis=0)
    assert merged.shape[0] >= 3

    source_pixels = _coords_in_quadrant(g, source_color, source_quad, hr, vc)
    assert source_pixels.shape[0] == 1, "source quadrant must contain only the singleton anchor"
    anchor = tuple(map(int, source_pixels[0]))
    assert anchor in {tuple(map(int, rc)) for rc in merged}, "anchor must lie on the restored source sprite"

    out = g.copy()
    out[merged[:, 0], merged[:, 1]] = source_color
    return out, {
        "source_color": source_color,
        "source_quad": source_quad,
        "hr": hr,
        "vc": vc,
        "restored": merged,
        "anchor": np.array(anchor, dtype=int),
    }


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden source sprite by unioning two complementary one-axis mirror remnants.

    Input structure:
    - A full grey horizontal mirror line and a full grey vertical mirror line
      split the grid into four quadrants.
    - One quadrant contains only a singleton anchor pixel whose color matches a
      hidden source sprite that is otherwise absent there.
    - The quadrants across the horizontal and vertical mirror lines each contain
      a same-color partial mirror image of that source sprite, with different
      missing pixels.
    - The diagonal fourth quadrant may contain unrelated colored noise.

    Rule:
    - Identify the source color as the unique non-grey color that appears in
      exactly three quadrants, with exactly one of those quadrants containing a
      single pixel.
    - Treat that singleton quadrant as the source quadrant.
    - Reflect the partial remnant across the horizontal mirror line back into
      the source quadrant, and reflect the other remnant across the vertical
      mirror line back into the source quadrant.
    - Paint the union of those two reflected coordinate sets in the source
      quadrant, leaving all other cells unchanged.

    Key insight:
    - No complete sprite is ever shown directly; only the two incomplete
      one-axis reflections share enough complementary evidence to reconstruct
      the hidden original at the anchor.
    """
    out, _ = _solve_from_parts(np.asarray(input_grid, dtype=int))
    return out


def _random_connected_shape(height: int, width: int, area: int) -> np.ndarray:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        cells.add(random.choice(frontier))
    coords = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    if r1 == r0 or c1 == c0:
        raise ValueError("degenerate shape bbox")
    return coords


def _local_to_global(local_coords: np.ndarray, quad: str, hr: int, vc: int) -> np.ndarray:
    out = local_coords.copy()
    if quad in ("BL", "BR"):
        out[:, 0] += hr + 1
    if quad in ("TR", "BR"):
        out[:, 1] += vc + 1
    return out


def _place_noise(grid: np.ndarray, quad: str, hr: int, vc: int, forbidden_colors: set[int]) -> None:
    if random.random() < 0.25:
        return

    rs, cs = _quadrant_slices(hr, vc)[quad]
    r_base = 0 if rs.start is None else rs.start
    c_base = 0 if cs.start is None else cs.start
    qh = rs.stop - r_base if rs.stop is not None else grid.shape[0] - r_base
    qw = cs.stop - c_base if cs.stop is not None else grid.shape[1] - c_base

    occupied: set[tuple[int, int]] = set()
    all_noise_colors = [c for c in Color.NOT_BLACK if c not in forbidden_colors]
    random.shuffle(all_noise_colors)
    count = random.randint(1, min(3, len(all_noise_colors)))
    for color in all_noise_colors[:count]:
        for _ in range(30):
            kind = random.choice(("dot", "domino_h", "domino_v", "l3"))
            if kind == "dot":
                shape = np.array([[0, 0]], dtype=int)
            elif kind == "domino_h":
                shape = np.array([[0, 0], [0, 1]], dtype=int)
            elif kind == "domino_v":
                shape = np.array([[0, 0], [1, 0]], dtype=int)
            else:
                shape = np.array([[0, 0], [1, 0], [1, 1]], dtype=int)

            sh = int(shape[:, 0].max()) + 1
            sw = int(shape[:, 1].max()) + 1
            if sh > qh or sw > qw:
                continue
            r0 = random.randint(0, qh - sh)
            c0 = random.randint(0, qw - sw)
            placed = {(r_base + r0 + int(r), c_base + c0 + int(c)) for r, c in shape}
            if placed & occupied:
                continue
            if any(grid[r, c] != Color.BLACK for r, c in placed):
                continue
            for r, c in placed:
                grid[r, c] = color
            occupied |= placed
            break


def _build_case() -> tuple[np.ndarray, np.ndarray, dict]:
    qh = random.randint(5, 12)
    qw = random.randint(5, 12)
    hr = qh
    vc = qw
    grid = np.full((2 * qh + 1, 2 * qw + 1), Color.BLACK, dtype=int)
    grid[hr, :] = Color.GREY
    grid[:, vc] = Color.GREY

    source_quad = random.choice(QUADS)
    source_color = random.choice([c for c in Color.NOT_BLACK if c != Color.GREY])

    while True:
        shape_h = random.randint(2, min(6, qh))
        shape_w = random.randint(2, min(6, qw))
        if shape_h * shape_w >= 5:
            break
    area_min = max(5, (shape_h * shape_w) // 2)
    area_max = min(shape_h * shape_w, max(area_min, 9))
    shape_area = random.randint(area_min, area_max)
    local_shape = _random_connected_shape(shape_h, shape_w, shape_area)

    off_r = random.randint(0, qh - shape_h)
    off_c = random.randint(0, qw - shape_w)
    local_shape = local_shape + np.array([off_r, off_c], dtype=int)

    source_global = _local_to_global(local_shape, source_quad, hr, vc)
    anchor_idx = random.randrange(source_global.shape[0])
    anchor = source_global[anchor_idx]

    all_idx = list(range(source_global.shape[0]))
    missing_h_count = random.randint(1, max(1, source_global.shape[0] // 3))
    missing_h = set(random.sample(all_idx, missing_h_count))
    remaining = [idx for idx in all_idx if idx not in missing_h]
    missing_v_count = random.randint(1, max(1, len(remaining) // 3))
    missing_v = set(random.sample(remaining, missing_v_count))

    keep_h = np.array([idx not in missing_h for idx in all_idx], dtype=bool)
    keep_v = np.array([idx not in missing_v for idx in all_idx], dtype=bool)
    assert keep_h.sum() < source_global.shape[0] and keep_v.sum() < source_global.shape[0]
    assert keep_h.sum() >= 2 and keep_v.sum() >= 2

    h_partial = _reflect_horizontal(source_global[keep_h], hr)
    v_partial = _reflect_vertical(source_global[keep_v], vc)

    grid[int(anchor[0]), int(anchor[1])] = source_color
    grid[h_partial[:, 0], h_partial[:, 1]] = source_color
    grid[v_partial[:, 0], v_partial[:, 1]] = source_color

    noise_quad = H_MATE[V_MATE[source_quad]]
    _place_noise(grid, noise_quad, hr, vc, {Color.BLACK, Color.GREY, source_color})

    expected = grid.copy()
    expected[source_global[:, 0], source_global[:, 1]] = source_color
    return grid, expected, {
        "source_color": source_color,
        "source_quad": source_quad,
        "hr": hr,
        "vc": vc,
        "source_global": source_global,
        "anchor": anchor.copy(),
        "missing_h": missing_h,
        "missing_v": missing_v,
    }


def _validate_case(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    out, solved_meta = _solve_from_parts(inp)
    if not np.array_equal(out, expected):
        return False

    source_color = meta["source_color"]
    source_quad = meta["source_quad"]
    hr = meta["hr"]
    vc = meta["vc"]
    source_global = meta["source_global"]

    if tuple(map(int, solved_meta["anchor"])) != tuple(map(int, meta["anchor"])):
        return False
    if solved_meta["source_color"] != source_color or solved_meta["source_quad"] != source_quad:
        return False

    counts = _quadrant_pixel_counts(inp, source_color, hr, vc)
    if counts[source_quad] != 1:
        return False
    if counts[H_MATE[source_quad]] >= source_global.shape[0]:
        return False
    if counts[V_MATE[source_quad]] >= source_global.shape[0]:
        return False
    if not meta["missing_h"] or not meta["missing_v"] or (meta["missing_h"] & meta["missing_v"]):
        return False

    restored = {tuple(map(int, rc)) for rc in solved_meta["restored"]}
    target = {tuple(map(int, rc)) for rc in source_global}
    if restored != target:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        inp, expected, meta = _build_case()
        if _validate_case(inp, expected, meta):
            return inp
    raise RuntimeError("failed to generate a valid double-mirror reconstruction task")
