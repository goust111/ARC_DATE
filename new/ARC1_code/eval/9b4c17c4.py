from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, in_bounds, neighbors4


def _carrier_color(g: np.ndarray, coords: np.ndarray) -> int:
    adjacent = set()
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        for nr, nc in neighbors4(r, c):
            if in_bounds(g.shape, nr, nc) and int(g[nr, nc]) != Color.RED:
                adjacent.add(int(g[nr, nc]))

    assert len(adjacent) == 1, "each red object must touch exactly one carrier color"
    carrier = next(iter(adjacent))
    assert carrier in {Color.BLUE, Color.TEAL}, "carrier color must be blue or teal"
    return carrier


def _containing_region(base: np.ndarray, coords: np.ndarray, color: int) -> np.ndarray:
    coord_set = {(int(r), int(c)) for r, c in coords}
    hits = []
    for region in components_mask(base == int(color), connectivity=4):
        if any((int(r), int(c)) in coord_set for r, c in region):
            hits.append(region)

    assert len(hits) == 1, "red object must belong to exactly one carrier region"
    return hits[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move every red object horizontally until it is flush with the side of its enclosing blue or teal region.

    Input structure
    - The grid contains only blue, red, and teal pixels.
    - The red pixels form one or more 4-connected objects.
    - Each red object is embedded in exactly one adjacent non-red carrier color.

    Rule
    - Clear each red object back to its carrier color to recover the blue and teal regions.
    - A red object carried by blue moves right until its bounding box touches the right edge of that blue region.
    - A red object carried by teal moves left until its bounding box touches the left edge of that teal region.
    - Preserve the object's shape and rows, and apply all object moves simultaneously.

    Key insight
    - The red objects are sliders inside colored horizontal lanes; the carrier color encodes the slide direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    assert set(map(int, np.unique(g))) == {Color.BLUE, Color.RED, Color.TEAL}, "expected blue, red, and teal only"

    red_objects = components_mask(g == Color.RED, connectivity=4)
    assert red_objects, "expected at least one red object"

    base = g.copy()
    objects: list[tuple[np.ndarray, int]] = []
    for coords in red_objects:
        carrier = _carrier_color(g, coords)
        base[coords[:, 0], coords[:, 1]] = carrier
        objects.append((coords, carrier))

    out = base.copy()
    occupied = np.zeros(g.shape, dtype=bool)
    H, W = g.shape

    for coords, carrier in objects:
        obj_box = bbox_coords(coords)
        region = _containing_region(base, coords, carrier)
        region_box = bbox_coords(region)
        obj_c0, obj_c1 = obj_box[1], obj_box[3]
        reg_c0, reg_c1 = region_box[1], region_box[3]

        obj_w = int(obj_c1 - obj_c0 + 1)
        if carrier == Color.BLUE:
            dst_c0 = int(reg_c1 - obj_w + 1)
        else:
            assert carrier == Color.TEAL
            dst_c0 = int(reg_c0)

        moved = coords.copy()
        moved[:, 1] += dst_c0 - int(obj_c0)

        assert np.all((0 <= moved[:, 0]) & (moved[:, 0] < H) & (0 <= moved[:, 1]) & (moved[:, 1] < W))
        assert np.all(base[moved[:, 0], moved[:, 1]] == carrier), "destination must stay inside the carrier region"
        assert not np.any(occupied[moved[:, 0], moved[:, 1]]), "moved red objects must not overlap"

        occupied[moved[:, 0], moved[:, 1]] = True
        out[moved[:, 0], moved[:, 1]] = Color.RED

    return out

# --------------- generator (o5x) ---------------
# Palette.
_GEN_BLUE = Color.BLUE
_GEN_RED = Color.RED
_GEN_TEAL = Color.TEAL

# Canvas and split geometry.
_GEN_CANVAS_MIN = 9
_GEN_CANVAS_MAX_EXCLUSIVE = 16
_GEN_SPLIT_MIN_RATIO = 0.30
_GEN_SPLIT_MAX_RATIO = 0.70
_GEN_SPLIT_TYPES = ("horizontal", "vertical")

# Red rectangle layout.
_GEN_RECT_SIZES = (
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 2),
    (4, 2),
)
_GEN_COUNT_OPTIONS = ((2, 2), (3, 2), (2, 3))
_GEN_MIN_ROW_GAP = 1
_GEN_MIN_RECT_HEIGHT = min(h for h, _ in _GEN_RECT_SIZES)

# Slide distances and validity guards.
_GEN_MAX_SLIDE_GAP = 10
_GEN_VERTICAL_SPLIT_CLEARANCE = 1
_GEN_RETRY_LIMIT = 300


def _gen_split(length: int, rng: np.random.Generator) -> int:
    lo = max(1, int(np.ceil(length * _GEN_SPLIT_MIN_RATIO)))
    hi = min(length - 1, int(np.floor(length * _GEN_SPLIT_MAX_RATIO)))
    choices = np.arange(lo, hi + 1)
    return int(rng.choice(choices))


def _gen_regions(height: int, width: int, split_type: str, cut: int) -> list[dict[str, int | str]]:
    if split_type == "horizontal":
        return [
            {"side": "top", "r0": 0, "r1": cut - 1, "c0": 0, "c1": width - 1},
            {"side": "bottom", "r0": cut, "r1": height - 1, "c0": 0, "c1": width - 1},
        ]
    return [
        {"side": "left", "r0": 0, "r1": height - 1, "c0": 0, "c1": cut - 1},
        {"side": "right", "r0": 0, "r1": height - 1, "c0": cut, "c1": width - 1},
    ]


def _gen_safe_rows(region: dict[str, int | str], split_type: str) -> tuple[int, int] | None:
    r0 = int(region["r0"])
    r1 = int(region["r1"])
    if split_type == "horizontal":
        if region["side"] == "top":
            r1 -= 1
        else:
            r0 += 1
    if r0 > r1:
        return None
    return r0, r1


def _gen_max_gap(region: dict[str, int | str], color: int, width: int, split_type: str) -> int:
    reg_w = int(region["c1"]) - int(region["c0"]) + 1
    max_gap = reg_w - width
    if split_type == "vertical":
        if color == _GEN_BLUE and region["side"] == "right":
            max_gap -= _GEN_VERTICAL_SPLIT_CLEARANCE
        elif color == _GEN_TEAL and region["side"] == "left":
            max_gap -= _GEN_VERTICAL_SPLIT_CLEARANCE
    return min(_GEN_MAX_SLIDE_GAP, max_gap)


def _gen_region_capacity(region: dict[str, int | str], split_type: str) -> int:
    safe = _gen_safe_rows(region, split_type)
    if safe is None:
        return 0
    return safe[1] - safe[0] + 1


def _gen_choose_counts(
    regions: list[dict[str, int | str]], split_type: str, rng: np.random.Generator
) -> list[int] | None:
    options = []
    for counts in _GEN_COUNT_OPTIONS:
        ok = True
        for region, count in zip(regions, counts):
            min_rows = count * _GEN_MIN_RECT_HEIGHT + (count - 1) * _GEN_MIN_ROW_GAP
            if _gen_region_capacity(region, split_type) < min_rows:
                ok = False
                break
        if ok:
            options.append(counts)
    if not options:
        return None
    return list(options[int(rng.integers(0, len(options)))])


def _gen_choose_rects(
    region: dict[str, int | str], count: int, split_type: str, rng: np.random.Generator
) -> list[tuple[int, int]] | None:
    safe = _gen_safe_rows(region, split_type)
    if safe is None:
        return None
    row_budget = safe[1] - safe[0] + 1 - _GEN_MIN_ROW_GAP * (count - 1)
    if row_budget < count * _GEN_MIN_RECT_HEIGHT:
        return None

    rects = []
    for i in range(count):
        remaining = count - i - 1
        choices = []
        remaining_min_height = remaining * _GEN_MIN_RECT_HEIGHT
        for h, w in _GEN_RECT_SIZES:
            fits_rows = h <= row_budget - remaining_min_height
            fits_gap = _gen_max_gap(region, int(region["color"]), w, split_type) >= 1
            if fits_rows and fits_gap:
                choices.append((h, w))
        if not choices:
            return None
        h, w = choices[int(rng.integers(0, len(choices)))]
        rects.append((h, w))
        row_budget -= h
    rng.shuffle(rects)
    return rects


def _gen_rows_for_rects(
    safe_r0: int, safe_r1: int, rects: list[tuple[int, int]], rng: np.random.Generator
) -> list[int]:
    used = sum(h for h, _ in rects)
    min_between = _GEN_MIN_ROW_GAP * (len(rects) - 1)
    extra = safe_r1 - safe_r0 + 1 - used - min_between
    slots = rng.multinomial(extra, np.full(len(rects) + 1, 1 / (len(rects) + 1)))

    rows = []
    r = safe_r0 + int(slots[0])
    for i, (h, _) in enumerate(rects):
        rows.append(r)
        r += h
        if i < len(rects) - 1:
            r += _GEN_MIN_ROW_GAP + int(slots[i + 1])
    return rows


def _gen_place_region_objects(
    out: np.ndarray,
    inp: np.ndarray,
    region: dict[str, int | str],
    count: int,
    split_type: str,
    rng: np.random.Generator,
) -> bool:
    """Mutate output/input grids with one region's flush and displaced red rectangles."""
    color = int(region["color"])
    rects = _gen_choose_rects(region, count, split_type, rng)
    if rects is None:
        return False
    safe = _gen_safe_rows(region, split_type)
    if safe is None:
        return False
    rows = _gen_rows_for_rects(safe[0], safe[1], rects, rng)

    for (h, w), r0 in zip(rects, rows):
        out_c0 = int(region["c1"]) - w + 1 if color == _GEN_BLUE else int(region["c0"])

        max_gap = _gen_max_gap(region, color, w, split_type)
        if max_gap < 1:
            return False
        gap = int(rng.integers(1, max_gap + 1))
        in_c0 = out_c0 - gap if color == _GEN_BLUE else out_c0 + gap

        out[r0 : r0 + h, out_c0 : out_c0 + w] = _GEN_RED
        inp[r0 : r0 + h, in_c0 : in_c0 + w] = _GEN_RED
    return True


def _gen_matches_oracle(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        solved = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, out))


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(_GEN_CANVAS_MIN, _GEN_CANVAS_MAX_EXCLUSIVE))
    width = int(rng.integers(_GEN_CANVAS_MIN, _GEN_CANVAS_MAX_EXCLUSIVE))
    split_type = _GEN_SPLIT_TYPES[int(rng.integers(0, len(_GEN_SPLIT_TYPES)))]
    cut = _gen_split(height if split_type == "horizontal" else width, rng)
    regions = _gen_regions(height, width, split_type, cut)

    if int(rng.integers(0, 2)) == 0:
        colors = [_GEN_BLUE, _GEN_TEAL]
    else:
        colors = [_GEN_TEAL, _GEN_BLUE]
    for region, color in zip(regions, colors):
        region["color"] = color

    counts = _gen_choose_counts(regions, split_type, rng)
    if counts is None:
        return None

    base = np.empty((height, width), dtype=int)
    for region in regions:
        base[int(region["r0"]) : int(region["r1"]) + 1, int(region["c0"]) : int(region["c1"]) + 1] = int(
            region["color"]
        )

    out = base.copy()
    inp = base.copy()
    for region, count in zip(regions, counts):
        if not _gen_place_region_objects(out, inp, region, count, split_type, rng):
            return None

    if not _gen_matches_oracle(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 9b4c17c4")
