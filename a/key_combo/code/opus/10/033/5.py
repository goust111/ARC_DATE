import math
import random

import numpy as np


BG = 0
VERTEX = 1
MOTIF = 2
GEN_RETRIES = 400

_GEN_CALL_INDEX = 0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("expected at least one foreground pixel")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0 and dc == 0:
        return [(int(r0), int(c0))]
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    steps = max(abs(dr), abs(dc))
    return [(int(r0 + sr * k), int(c0 + sc * k)) for k in range(steps + 1)]


def _aligned(a: tuple[int, int], b: tuple[int, int]) -> bool:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _polar_cycle(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points) / len(points)
    cc = sum(c for _, c in points) / len(points)

    def key(p: tuple[int, int]) -> tuple[float, float, int, int]:
        r, c = p
        ang = math.atan2(-(r - cr), c - cc)
        if ang < 0:
            ang += 2 * math.pi
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, int(r), int(c))

    return sorted(points, key=key)


def _draw_outline(base: np.ndarray, vertices: list[tuple[int, int]]) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    cycle = _polar_cycle(vertices)
    n = len(cycle)
    for i in range(n):
        a = cycle[i]
        b = cycle[(i + 1) % n]
        assert _aligned(a, b), f"edge is not H/V/45: {a}->{b}"
        cells = _segment_cells(a, b)
        for rr, cc in cells[1:-1]:
            if int(base[rr, cc]) != BG:
                raise AssertionError("edge interior is blocked in the original input")
        for rr, cc in cells:
            out[rr, cc] = VERTEX
    return out


def _mirror_target(r: int, c: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    if w > h:
        return int(r), int(w - 1 - c)
    return int(h - 1 - r), int(c)


def _mirror_motif(crop: np.ndarray) -> np.ndarray:
    out = np.asarray(crop, dtype=int).copy()
    h, w = out.shape
    assert h != w, "crop bbox must have a non-square aspect ratio"
    motif_coords = [tuple(map(int, rc)) for rc in np.argwhere(crop == MOTIF)]
    for r, c in motif_coords:
        rr, cc = _mirror_target(r, c, crop.shape)
        if 0 <= rr < h and 0 <= cc < w and int(out[rr, cc]) == BG:
            out[rr, cc] = MOTIF
    return out


def _normalize_coords(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.asarray(points, dtype=int)
    arr -= arr.min(axis=0)
    return [tuple(map(int, p)) for p in arr]


def _transform_points(
    points: list[tuple[int, int]],
    *,
    transpose: bool,
    flip_r: bool,
    flip_c: bool,
) -> list[tuple[int, int]]:
    arr = np.asarray(points, dtype=int)
    if transpose:
        arr = arr[:, ::-1]
    arr -= arr.min(axis=0)
    if flip_r:
        arr[:, 0] = arr[:, 0].max() - arr[:, 0]
    if flip_c:
        arr[:, 1] = arr[:, 1].max() - arr[:, 1]
    arr -= arr.min(axis=0)
    return [tuple(map(int, p)) for p in arr]


def _inside_convex_polygon(cell: tuple[int, int], cycle: list[tuple[int, int]]) -> bool:
    x = float(cell[1])
    y = float(-cell[0])
    sign = 0
    n = len(cycle)
    for i in range(n):
        r0, c0 = cycle[i]
        r1, c1 = cycle[(i + 1) % n]
        x0, y0 = float(c0), float(-r0)
        x1, y1 = float(c1), float(-r1)
        cross = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
        if abs(cross) < 1e-9:
            return False
        cur = 1 if cross > 0 else -1
        if sign == 0:
            sign = cur
        elif cur != sign:
            return False
    return True


def _strict_interior_mask(shape: tuple[int, int], cycle: list[tuple[int, int]]) -> np.ndarray:
    h, w = map(int, shape)
    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            mask[r, c] = _inside_convex_polygon((r, c), cycle)
    return mask


def _make_quad(wide: bool) -> list[tuple[int, int]]:
    slant = random.randint(2, 4)
    span = random.randint(max(3, slant + 1), slant + 5)
    points = [(0, 0), (0, span), (slant, slant + span), (slant, slant)]
    if not wide:
        points = [(c, r) for r, c in points]
    return _normalize_coords(points)


def _make_pentagon(wide: bool) -> list[tuple[int, int]]:
    bevel = random.randint(2, 3)
    stem = random.randint(2, 5)
    run = random.randint(max(3, stem + 1), stem + 5)
    points = [(0, 0), (0, run), (bevel, bevel + run), (bevel + stem, bevel + run), (bevel + stem, 0)]
    if not wide:
        points = [(c, r) for r, c in points]
    return _normalize_coords(points)


def _make_hexagon(wide: bool) -> list[tuple[int, int]]:
    bevel = random.randint(2, 3)
    height_mid = random.randint(2, 4)
    run = random.randint(bevel + height_mid + 1, bevel + height_mid + 4)
    points = [
        (0, bevel),
        (bevel, 0),
        (bevel + height_mid, 0),
        (2 * bevel + height_mid, bevel),
        (2 * bevel + height_mid, bevel + run),
        (0, bevel + run),
    ]
    if not wide:
        points = [(c, r) for r, c in points]
    return _normalize_coords(points)


def _sample_vertices(prefer_wide: bool) -> list[tuple[int, int]]:
    builder = random.choice([_make_quad, _make_pentagon, _make_hexagon])
    points = builder(prefer_wide)
    points = _transform_points(
        points,
        transpose=False,
        flip_r=bool(random.getrandbits(1)),
        flip_c=bool(random.getrandbits(1)),
    )
    cycle = _polar_cycle(points)
    for i in range(len(cycle)):
        if not _aligned(cycle[i], cycle[(i + 1) % len(cycle)]):
            raise AssertionError("sampled polygon lost 8-direction alignment")
    h = 1 + max(r for r, _ in points)
    w = 1 + max(c for _, c in points)
    if prefer_wide and not (w > h):
        raise AssertionError("expected a wide polygon")
    if not prefer_wide and not (h > w):
        raise AssertionError("expected a tall polygon")
    return points


def _build_patch(vertices: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], np.ndarray]:
    cycle = _polar_cycle(vertices)
    h = 1 + max(r for r, _ in cycle)
    w = 1 + max(c for _, c in cycle)
    outline = np.zeros((h, w), dtype=int)
    for i in range(len(cycle)):
        for rr, cc in _segment_cells(cycle[i], cycle[(i + 1) % len(cycle)]):
            outline[rr, cc] = VERTEX
    interior = _strict_interior_mask((h, w), cycle) & (outline == BG)
    return outline, interior, cycle, np.asarray(vertices, dtype=int)


def _source_candidates(
    outline: np.ndarray,
    interior: np.ndarray,
    choose_low_side: bool,
) -> list[tuple[int, int]]:
    h, w = outline.shape
    cells = []
    for r, c in np.argwhere(interior):
        rr, cc = _mirror_target(int(r), int(c), outline.shape)
        if (rr, cc) == (int(r), int(c)):
            continue
        if not interior[rr, cc]:
            continue
        if int(outline[rr, cc]) != BG:
            continue
        if w > h:
            if choose_low_side and not (int(c) < cc):
                continue
            if (not choose_low_side) and not (int(c) > cc):
                continue
        else:
            if choose_low_side and not (int(r) < rr):
                continue
            if (not choose_low_side) and not (int(r) > rr):
                continue
        cells.append((int(r), int(c)))
    return cells


def _render_expected(outline: np.ndarray, reds: list[tuple[int, int]]) -> np.ndarray:
    patch = np.asarray(outline, dtype=int).copy()
    for r, c in reds:
        patch[int(r), int(c)] = MOTIF
    return _mirror_motif(patch)


def _embed_patch(patch: np.ndarray, canvas_shape: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = patch.shape
    H, W = canvas_shape
    grid = np.zeros((H, W), dtype=int)
    margin_r = random.randint(1, max(1, H - h - 1))
    margin_c = random.randint(1, max(1, W - w - 1))
    grid[margin_r : margin_r + h, margin_c : margin_c + w] = patch
    return grid, (margin_r, margin_c)


def _localize(points: list[tuple[int, int]], offset: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = offset
    return [(int(r - dr), int(c - dc)) for r, c in points]


def _validate_sample(inp: np.ndarray, expected: np.ndarray) -> None:
    assert inp.ndim == 2 and expected.ndim == 2
    assert max(inp.shape) <= 30 and max(expected.shape) <= 30
    assert set(np.unique(inp)).issubset({BG, VERTEX, MOTIF})
    verts = [tuple(map(int, rc)) for rc in np.argwhere(inp == VERTEX)]
    reds = [tuple(map(int, rc)) for rc in np.argwhere(inp == MOTIF)]
    assert 4 <= len(verts) <= 6
    assert 2 <= len(reds) <= 4
    cycle = _polar_cycle(verts)
    for i in range(len(cycle)):
        a = cycle[i]
        b = cycle[(i + 1) % len(cycle)]
        assert _aligned(a, b)
        for rr, cc in _segment_cells(a, b)[1:-1]:
            assert int(inp[rr, cc]) == BG

    drawn = _draw_outline(inp, verts)
    box = _bbox((drawn == VERTEX) | (drawn == MOTIF))
    crop_before = _crop(drawn, box)
    assert crop_before.shape != (crop_before.shape[0], crop_before.shape[0])
    local_cycle = _localize(cycle, (box[0], box[1]))
    interior = _strict_interior_mask(crop_before.shape, local_cycle)
    motif_local = _localize(reds, (box[0], box[1]))
    branch_wide = crop_before.shape[1] > crop_before.shape[0]
    side_values = []
    for r, c in motif_local:
        assert interior[r, c]
        rr, cc = _mirror_target(r, c, crop_before.shape)
        assert interior[rr, cc]
        assert int(crop_before[rr, cc]) == BG
        side_values.append(c if branch_wide else r)
    mirror_values = [
        _mirror_target(r, c, crop_before.shape)[1 if branch_wide else 0]
        for r, c in motif_local
    ]
    if branch_wide:
        assert all(v < mv for v, mv in zip(side_values, mirror_values)) or all(v > mv for v, mv in zip(side_values, mirror_values))
    else:
        assert all(v < mv for v, mv in zip(side_values, mirror_values)) or all(v > mv for v, mv in zip(side_values, mirror_values))

    actual = main(inp)
    assert np.array_equal(actual, expected)
    assert np.any((expected == MOTIF) & (crop_before == BG))


def _generate_attempt(prefer_wide: bool) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        vertices = _sample_vertices(prefer_wide)
        outline, interior, _cycle, _vertex_arr = _build_patch(vertices)
        choose_low_side = bool(random.getrandbits(1))
        candidates = _source_candidates(outline, interior, choose_low_side)
        if len(candidates) < 2:
            return None
        count = random.randint(2, min(4, len(candidates)))
        reds = random.sample(candidates, count)

        input_patch = np.zeros_like(outline)
        for r, c in vertices:
            input_patch[int(r), int(c)] = VERTEX
        for r, c in reds:
            input_patch[int(r), int(c)] = MOTIF

        expected = _render_expected(outline, reds)
        ph, pw = outline.shape
        min_h = max(13, ph + 3)
        min_w = max(13, pw + 3)
        max_h = min(19, 30)
        max_w = min(19, 30)
        if min_h > max_h or min_w > max_w:
            return None
        canvas_shape = (
            random.randint(min_h, max_h),
            random.randint(min_w, max_w),
        )
        inp, _offset = _embed_patch(input_patch, canvas_shape)
        _validate_sample(inp, expected)
        return inp, expected
    except AssertionError:
        return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the scattered blue vertices into a polygon, crop it, and mirror the red interior motif according to the crop aspect ratio.

    Input structure:
    - Black background contains exactly one set of 4-6 blue vertex pixels and 2-4 red motif pixels.
    - The blue pixels are the polygon vertices; every edge implied by polar-angle order is horizontal, vertical, or 45-degree diagonal.
    - The red pixels lie strictly inside the polygon on only one side of the later mirror axis, and edge interiors are clear background in the input.

    Rule:
    - Sort the blue vertices by polar angle around their centroid using the standard mathematical angle from the positive rightward direction counterclockwise.
    - Draw blue edges between consecutive sorted vertices and from the last back to the first, but only when the edge interior is background in the original input.
    - Crop to the tight bounding box of all blue and red pixels, then mirror each red pixel left-right if the crop is wider than tall, otherwise top-bottom; only fill mirrored targets that are still black.

    Key insight:
    - The polygon must be reconstructed before its own cropped aspect ratio can be read, and that recovered geometry is what determines the final mirror direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    vertices = [tuple(map(int, rc)) for rc in np.argwhere(g == VERTEX)]
    assert 4 <= len(vertices) <= 6, "expected 4-6 blue vertices"

    outlined = _draw_outline(g, vertices)
    crop_box = _bbox((outlined == VERTEX) | (outlined == MOTIF))
    cropped = _crop(outlined, crop_box)
    return _mirror_motif(cropped)


def generate_input() -> np.ndarray:
    global _GEN_CALL_INDEX
    prefer_wide = (_GEN_CALL_INDEX % 2) == 0
    for _ in range(GEN_RETRIES):
        sample = _generate_attempt(prefer_wide)
        if sample is not None:
            _GEN_CALL_INDEX += 1
            return sample[0]
    raise RuntimeError("failed to generate a valid puzzle input")
