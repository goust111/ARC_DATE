from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

_GEN_MIN_SIZE = 17
_GEN_MAX_SIZE = 29
_GEN_MIN_D = 2
_GEN_MAX_D = 5
_GEN_ATTEMPTS = 400
_GEN_PAIR_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _seed_coords(center: tuple[int, int], d: int, slope: int) -> tuple[tuple[int, int], ...]:
    r, c = map(int, center)
    d = int(d)
    slope = 1 if int(slope) >= 0 else -1
    return (
        (r - d, c - slope * d),
        (r, c),
        (r + d, c + slope * d),
    )


def _extract_seed_spec(grid: np.ndarray, color: int) -> dict[str, object]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == int(color))]
    assert len(coords) == 3, f"color {color} must have exactly three seeds"

    farthest = None
    max_dist2 = -1
    for i in range(3):
        for j in range(i + 1, 3):
            dr = coords[i][0] - coords[j][0]
            dc = coords[i][1] - coords[j][1]
            dist2 = dr * dr + dc * dc
            if dist2 > max_dist2:
                max_dist2 = dist2
                farthest = (i, j)
    assert farthest is not None

    i, j = farthest
    outer_a = coords[i]
    outer_b = coords[j]
    center = coords[3 - i - j]

    assert outer_a[0] + outer_b[0] == 2 * center[0]
    assert outer_a[1] + outer_b[1] == 2 * center[1]

    dr = outer_a[0] - center[0]
    dc = outer_a[1] - center[1]
    d = abs(int(dr))
    assert d > 0 and abs(int(dc)) == d, "seeds must lie on a 45-degree diagonal"
    assert outer_b[0] - center[0] == -dr and outer_b[1] - center[1] == -dc

    slope = 1 if dr * dc > 0 else -1
    seeds = tuple(sorted(coords))
    assert seeds == tuple(sorted(_seed_coords(center, d, slope)))
    return {"color": int(color), "center": center, "d": d, "slope": slope, "seeds": seeds}


def _border_intersects(shape: tuple[int, int], center: tuple[int, int], radius: int) -> bool:
    height, width = map(int, shape)
    r, c = map(int, center)
    radius = int(radius)
    top = r - radius
    bottom = r + radius
    left = c - radius
    right = c + radius

    return (
        (0 <= top < height and _intervals_overlap(left, right, 0, width - 1))
        or (0 <= bottom < height and _intervals_overlap(left, right, 0, width - 1))
        or (0 <= left < width and _intervals_overlap(top, bottom, 0, height - 1))
        or (0 <= right < width and _intervals_overlap(top, bottom, 0, height - 1))
    )


def _draw_square_border(mask: np.ndarray, center: tuple[int, int], radius: int) -> None:
    height, width = mask.shape
    r, c = map(int, center)
    radius = int(radius)
    top = r - radius
    bottom = r + radius
    left = c - radius
    right = c + radius

    c0 = max(0, left)
    c1 = min(width - 1, right)
    r0 = max(0, top)
    r1 = min(height - 1, bottom)

    if 0 <= top < height and c0 <= c1:
        mask[top, c0 : c1 + 1] = True
    if 0 <= bottom < height and c0 <= c1:
        mask[bottom, c0 : c1 + 1] = True
    if 0 <= left < width and r0 <= r1:
        mask[r0 : r1 + 1, left] = True
    if 0 <= right < width and r0 <= r1:
        mask[r0 : r1 + 1, right] = True


def _group_mask(shape: tuple[int, int], spec: dict[str, object]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    center = tuple(spec["center"])
    d = int(spec["d"])
    for r, c in spec["seeds"]:
        assert _in_bounds(shape, int(r), int(c))
        mask[int(r), int(c)] = True

    layer = 1
    while _border_intersects(shape, center, layer * d):
        _draw_square_border(mask, center, layer * d)
        layer += 1
    return mask


def _xor_output(shape: tuple[int, int], red_spec: dict[str, object], blue_spec: dict[str, object]) -> np.ndarray:
    red_mask = _group_mask(shape, red_spec)
    blue_mask = _group_mask(shape, blue_spec)
    out = np.zeros(shape, dtype=int)
    out[red_mask & ~blue_mask] = RED
    out[blue_mask & ~red_mask] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow two concentric square patterns from diagonal seed triplets and XOR their overlap.

    Input structure:
    - The grid has a black background with exactly two foreground colors: red and blue.
    - Each color appears as exactly three singleton seed pixels on a 45-degree diagonal with equal spacing.
    - Within each color, the middle seed is the growth center and the outer seeds mark the first square-corner radius.

    Rule:
    - For each color separately, recover the center seed and the diagonal spacing d.
    - From that center, keep the three seeds and add axis-aligned square borders at Chebyshev radii d, 2d, 3d, ... until the next border no longer intersects the grid.
    - Combine the red-grown mask and blue-grown mask by pixelwise XOR: cells occupied by only one mask keep that mask's color, while cells occupied by both become black.

    Key insight:
    - The visible black holes in the output are not empty background from one source pattern; they are cancellation sites where two independently grown concentric square systems overlap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, BLUE, RED}
    assert BLUE in colors and RED in colors

    red_spec = _extract_seed_spec(g, RED)
    blue_spec = _extract_seed_spec(g, BLUE)
    return _xor_output(g.shape, red_spec, blue_spec)


def _spec(center: tuple[int, int], d: int, slope: int, color: int) -> dict[str, object]:
    return {
        "center": tuple(map(int, center)),
        "d": int(d),
        "slope": 1 if int(slope) >= 0 else -1,
        "color": int(color),
        "seeds": _seed_coords(center, d, slope),
    }


def _sample_d(rng: np.random.Generator, max_d: int, equal_to: int | None = None) -> int:
    if equal_to is not None:
        return int(equal_to)
    return int(rng.integers(_GEN_MIN_D, max_d + 1))


def _candidate_centers(shape: tuple[int, int], d: int) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    pad = 2 * int(d)
    return [
        (r, c)
        for r in range(pad, height - pad)
        for c in range(pad, width - pad)
    ]


def _make_input(shape: tuple[int, int], red_spec: dict[str, object], blue_spec: dict[str, object]) -> np.ndarray:
    inp = np.zeros(shape, dtype=int)
    for r, c in red_spec["seeds"]:
        inp[int(r), int(c)] = RED
    for r, c in blue_spec["seeds"]:
        assert inp[int(r), int(c)] == BLACK
        inp[int(r), int(c)] = BLUE
    return inp


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _seeds_cross_other_mask(shape: tuple[int, int], a_spec: dict[str, object], b_spec: dict[str, object]) -> bool:
    b_mask = _group_mask(shape, b_spec)
    return any(b_mask[int(r), int(c)] for r, c in a_spec["seeds"])


def _full_ring_in_bounds(shape: tuple[int, int], center: tuple[int, int], radius: int) -> bool:
    height, width = map(int, shape)
    r, c = map(int, center)
    radius = int(radius)
    return (
        0 <= r - radius
        and r + radius < height
        and 0 <= c - radius
        and c + radius < width
    )


def _validate_case(inp: np.ndarray, red_spec: dict[str, object], blue_spec: dict[str, object]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if int((inp == RED).sum()) != 3 or int((inp == BLUE).sum()) != 3:
        return False

    try:
        parsed_red = _extract_seed_spec(inp, RED)
        parsed_blue = _extract_seed_spec(inp, BLUE)
        out = main(inp)
    except Exception:
        return False

    if parsed_red["seeds"] != red_spec["seeds"] or parsed_blue["seeds"] != blue_spec["seeds"]:
        return False

    expected = _xor_output(inp.shape, red_spec, blue_spec)
    if not np.array_equal(out, expected):
        return False

    red_mask = _group_mask(inp.shape, red_spec)
    blue_mask = _group_mask(inp.shape, blue_spec)
    overlap = int((red_mask & blue_mask).sum())
    red_only = int((red_mask & ~blue_mask).sum())
    blue_only = int((blue_mask & ~red_mask).sum())
    if overlap < 4 or red_only < 6 or blue_only < 6:
        return False

    if not np.any(out == RED) or not np.any(out == BLUE):
        return False

    overlay_rb = np.zeros_like(out)
    overlay_rb[red_mask] = RED
    overlay_rb[blue_mask] = BLUE
    overlay_br = np.zeros_like(out)
    overlay_br[blue_mask] = BLUE
    overlay_br[red_mask] = RED
    if np.array_equal(out, overlay_rb) or np.array_equal(out, overlay_br):
        return False

    if int(np.count_nonzero(out)) >= int(0.8 * inp.size):
        return False

    if not _full_ring_in_bounds(inp.shape, tuple(red_spec["center"]), int(red_spec["d"])):
        return False
    if not _full_ring_in_bounds(inp.shape, tuple(red_spec["center"]), 2 * int(red_spec["d"])):
        return False
    if not _full_ring_in_bounds(inp.shape, tuple(blue_spec["center"]), int(blue_spec["d"])):
        return False
    if not _full_ring_in_bounds(inp.shape, tuple(blue_spec["center"]), 2 * int(blue_spec["d"])):
        return False

    return True


def _attempt_generate_input(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(_GEN_MIN_SIZE, _GEN_MAX_SIZE + 1))
    width = int(rng.integers(_GEN_MIN_SIZE, _GEN_MAX_SIZE + 1))
    max_d = min(_GEN_MAX_D, (height - 1) // 4, (width - 1) // 4)
    if max_d < _GEN_MIN_D:
        return None

    equal_d = bool(rng.random() < 0.35)
    same_slope = bool(rng.random() < 0.5)
    require_seed_cross = bool(rng.random() < 0.45)

    d1 = _sample_d(rng, max_d)
    d2 = _sample_d(rng, max_d, equal_to=d1 if equal_d else None)
    if not equal_d and max_d > _GEN_MIN_D:
        for _ in range(20):
            d2 = _sample_d(rng, max_d)
            if d2 != d1:
                break
        if d2 == d1:
            return None

    slope1 = 1 if rng.random() < 0.5 else -1
    slope2 = slope1 if same_slope else -slope1

    centers1 = _candidate_centers((height, width), d1)
    centers2 = _candidate_centers((height, width), d2)
    if not centers1 or not centers2:
        return None

    for _ in range(_GEN_PAIR_ATTEMPTS):
        center1 = centers1[int(rng.integers(len(centers1)))]
        center2 = centers2[int(rng.integers(len(centers2)))]
        if center1 == center2:
            continue
        if _chebyshev(center1, center2) <= min(d1, d2):
            continue

        red_spec = _spec(center1, d1, slope1, RED)
        blue_spec = _spec(center2, d2, slope2, BLUE)
        all_seeds = set(red_spec["seeds"]) | set(blue_spec["seeds"])
        if len(all_seeds) != 6:
            continue

        inp = _make_input((height, width), red_spec, blue_spec)
        red_mask = _group_mask(inp.shape, red_spec)
        blue_mask = _group_mask(inp.shape, blue_spec)
        overlap = int((red_mask & blue_mask).sum())
        if overlap < 4:
            continue

        if require_seed_cross and not (
            _seeds_cross_other_mask(inp.shape, red_spec, blue_spec)
            or _seeds_cross_other_mask(inp.shape, blue_spec, red_spec)
        ):
            continue

        out = _xor_output(inp.shape, red_spec, blue_spec)
        if int((out == RED).sum()) < 6 or int((out == BLUE).sum()) < 6:
            continue
        if int(np.count_nonzero(out)) <= overlap:
            continue

        if _validate_case(inp, red_spec, blue_spec):
            return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_ATTEMPTS):
        inp = _attempt_generate_input(rng)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid dual-source XOR-growth ARC input")
