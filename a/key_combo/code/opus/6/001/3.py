from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
BarSpec = dict[str, object]
PairSpec = dict[str, object]
Box = tuple[int, int, int, int]


BG = 0
PAIR_COLORS = (1, 2, 3, 4)
ODD_LENGTHS = (3, 5, 7)
ORIENTATIONS = ("H", "V")
PATTERN_CATALOG = (("H", "V"), ("V", "H"), ("H", "H"), ("V", "V"))
CELL_SIZE_RANGE = (8, 11)
OUTER_MARGIN_RANGE = (1, 2)
CELL_GAP_RANGE = (2, 3)
PAIR_COUNT_RANGE = (2, 4)
PAIR_ATTEMPTS = 500
GRID_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _bar_pixels(center: Coord, length: int, orient: str) -> tuple[Coord, ...]:
    r, c = center
    half = length // 2
    if orient == "H":
        return tuple((r, c + d) for d in range(-half, half + 1))
    return tuple((r + d, c) for d in range(-half, half + 1))


def _bar_endpoints(center: Coord, length: int, orient: str) -> tuple[Coord, Coord]:
    pixels = _bar_pixels(center, length, orient)
    return pixels[0], pixels[-1]


def _swap_orient(orient: str) -> str:
    return "V" if orient == "H" else "H"


def _axis_for(orient: str) -> Coord:
    return (0, 1) if orient == "H" else (1, 0)


def _diag_open_path(a: Coord, b: Coord) -> tuple[Coord, ...]:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    assert abs(dr) == abs(dc) and dr != 0
    sr = 1 if dr > 0 else -1
    sc = 1 if dc > 0 else -1
    steps = abs(dr)
    return tuple((a[0] + sr * k, a[1] + sc * k) for k in range(1, steps))


def _touching_or_overlapping(cells1: set[Coord], cells2: set[Coord]) -> bool:
    for r, c in cells1:
        if (r, c) in cells2:
            return True
        if (r - 1, c) in cells2 or (r + 1, c) in cells2:
            return True
        if (r, c - 1) in cells2 or (r, c + 1) in cells2:
            return True
    return False


def _same_line(bar1: BarSpec, bar2: BarSpec) -> bool:
    if str(bar1["orient"]) != str(bar2["orient"]):
        return False
    if str(bar1["orient"]) == "H":
        return int(bar1["center"][0]) == int(bar2["center"][0])
    return int(bar1["center"][1]) == int(bar2["center"][1])


def _describe_input_bar(comp: dict[str, object]) -> BarSpec:
    coords = np.asarray(comp["coords"], dtype=int)
    r0, c0, r1, c1 = map(int, comp["bbox"])
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    area = int(comp["area"])
    assert height == 1 or width == 1
    length = max(height, width)
    assert area == length
    assert 3 <= length <= 7 and length % 2 == 1
    orient = "H" if height == 1 else "V"
    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    expected = set(_bar_pixels(center, length, orient))
    actual = {tuple(map(int, xy)) for xy in coords}
    assert actual == expected
    rot_orient = _swap_orient(orient)
    return {
        "color": int(comp["color"]),
        "center": center,
        "length": length,
        "orient": orient,
        "pixels": tuple(sorted(expected)),
        "endpoints": _bar_endpoints(center, length, orient),
        "rot_orient": rot_orient,
        "rot_pixels": _bar_pixels(center, length, rot_orient),
        "rot_endpoints": _bar_endpoints(center, length, rot_orient),
    }


def _diagonal_endpoint_candidates(
    endpoints1: tuple[Coord, Coord], endpoints2: tuple[Coord, Coord], *, require_open: bool
) -> list[dict[str, object]]:
    out = []
    for a in endpoints1:
        for b in endpoints2:
            dr = int(b[0] - a[0])
            dc = int(b[1] - a[1])
            if abs(dr) != abs(dc) or dr == 0:
                continue
            if require_open and abs(dr) < 2:
                continue
            out.append(
                {
                    "a": a,
                    "b": b,
                    "distance": abs(dr),
                    "slope_sign": 1 if dr * dc > 0 else -1,
                    "path": _diag_open_path(a, b),
                }
            )
    return out


def _pair_solution(bar1: BarSpec, bar2: BarSpec, *, require_open: bool) -> dict[str, object]:
    candidates = _diagonal_endpoint_candidates(
        bar1["rot_endpoints"], bar2["rot_endpoints"], require_open=require_open
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    rot_pixels = set(bar1["rot_pixels"]) | set(bar2["rot_pixels"])
    assert not (rot_pixels & set(candidate["path"]))
    return candidate


def _analyze_scene(grid: np.ndarray, *, require_open: bool) -> tuple[list[BarSpec], dict[int, dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    comps = _find_components(g, bg)
    assert 4 <= len(comps) <= 8 and len(comps) % 2 == 0

    bars = [_describe_input_bar(comp) for comp in comps]
    by_color: dict[int, list[BarSpec]] = {}
    for bar in bars:
        by_color.setdefault(int(bar["color"]), []).append(bar)
    assert 2 <= len(by_color) <= 4
    assert all(len(group) == 2 for group in by_color.values())

    pair_info: dict[int, dict[str, object]] = {}
    for color, pair in by_color.items():
        bar1, bar2 = pair
        assert not _same_line(bar1, bar2)
        assert not _touching_or_overlapping(set(bar1["pixels"]), set(bar2["pixels"]))
        assert not _diagonal_endpoint_candidates(
            bar1["endpoints"], bar2["endpoints"], require_open=True
        )
        pair_info[color] = _pair_solution(bar1, bar2, require_open=require_open)

    return bars, pair_info


def _render_rotated_output(
    shape: tuple[int, int], bars: list[BarSpec], pair_info: dict[int, dict[str, object]], bg: int
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for bar in bars:
        color = int(bar["color"])
        for r, c in bar["rot_pixels"]:
            assert out[r, c] == bg
            out[r, c] = color
    for color, info in pair_info.items():
        for r, c in info["path"]:
            assert out[r, c] == bg
            out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each thin bar about its own center, then diagonally connect the matching post-rotation endpoints of each color pair.

    Input structure:
    - The background is the dominant color.
    - Every foreground component is a monochromatic 4-connected one-cell-thick horizontal or vertical bar of odd length.
    - Each foreground color appears on exactly two separated bars, and the connection-worthy diagonal endpoint relation exists only after rotation.

    Rule:
    - For each bar, erase its original pixels and redraw the same-length bar rotated 90 degrees around the same center.
    - For each same-color pair after rotation, find the unique pair of endpoints whose row and column offsets have equal nonzero magnitude.
    - Draw the open 45-degree diagonal segment between those two endpoints in the same color, leaving the endpoint pixels themselves as bar pixels.

    Key insight:
    - The connector cannot be chosen from the input geometry directly; the correct endpoints are revealed only after mentally rotating the bars in place.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    bars, pair_info = _analyze_scene(g, require_open=False)
    return _render_rotated_output(g.shape, bars, pair_info, bg)


def _center_choices(box: Box, half: int) -> list[Coord]:
    r0, c0, r1, c1 = box
    rows = range(r0 + half, r1 - half + 1)
    cols = range(c0 + half, c1 - half + 1)
    return [(r, c) for r in rows for c in cols]


def _inside_box(cells: tuple[Coord, ...], box: Box) -> bool:
    r0, c0, r1, c1 = box
    return all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in cells)


def _make_bar(center: Coord, length: int, orient: str, color: int) -> BarSpec:
    rot_orient = _swap_orient(orient)
    return {
        "color": int(color),
        "center": center,
        "length": int(length),
        "orient": orient,
        "pixels": _bar_pixels(center, length, orient),
        "endpoints": _bar_endpoints(center, length, orient),
        "rot_orient": rot_orient,
        "rot_pixels": _bar_pixels(center, length, rot_orient),
        "rot_endpoints": _bar_endpoints(center, length, rot_orient),
    }


def _construct_pair(
    box: Box, color: int, input_orients: tuple[str, str], slope_sign: int
) -> PairSpec | None:
    h = box[2] - box[0] + 1
    w = box[3] - box[1] + 1
    lengths = [length for length in ODD_LENGTHS if length <= min(h, w)]
    if not lengths:
        return None

    out_orients = (_swap_orient(input_orients[0]), _swap_orient(input_orients[1]))
    axis1 = _axis_for(out_orients[0])
    axis2 = _axis_for(out_orients[1])

    for _ in range(PAIR_ATTEMPTS):
        length1 = random.choice(lengths)
        length2 = random.choice(lengths)
        half1 = length1 // 2
        half2 = length2 // 2
        center1 = random.choice(_center_choices(box, half1))
        endpoint_side1 = random.choice((-1, 1))
        endpoint_side2 = random.choice((-1, 1))
        row_sign = random.choice((-1, 1))
        distance = random.randint(2, 5)
        delta = (row_sign * distance, row_sign * slope_sign * distance)

        center2 = (
            int(center1[0] + endpoint_side1 * half1 * axis1[0] + delta[0] - endpoint_side2 * half2 * axis2[0]),
            int(center1[1] + endpoint_side1 * half1 * axis1[1] + delta[1] - endpoint_side2 * half2 * axis2[1]),
        )
        if center2 not in _center_choices(box, half2):
            continue

        bar1 = _make_bar(center1, length1, input_orients[0], color)
        bar2 = _make_bar(center2, length2, input_orients[1], color)
        if not _inside_box(bar1["pixels"], box) or not _inside_box(bar2["pixels"], box):
            continue
        if not _inside_box(bar1["rot_pixels"], box) or not _inside_box(bar2["rot_pixels"], box):
            continue
        if _same_line(bar1, bar2):
            continue
        if _touching_or_overlapping(set(bar1["pixels"]), set(bar2["pixels"])):
            continue
        if set(bar1["rot_pixels"]) & set(bar2["rot_pixels"]):
            continue
        if _diagonal_endpoint_candidates(bar1["endpoints"], bar2["endpoints"], require_open=True):
            continue

        candidates = _diagonal_endpoint_candidates(
            bar1["rot_endpoints"], bar2["rot_endpoints"], require_open=True
        )
        if len(candidates) != 1:
            continue
        connector = candidates[0]
        if int(connector["slope_sign"]) != int(slope_sign):
            continue
        footprint = set(bar1["rot_pixels"]) | set(bar2["rot_pixels"])
        if footprint & set(connector["path"]):
            continue
        if not _inside_box(connector["path"], box):
            continue

        return {
            "color": int(color),
            "bars": (bar1, bar2),
            "connector": connector,
            "box": box,
        }
    return None


def _sample_layout() -> tuple[tuple[int, int], list[Box]]:
    top = random.randint(*OUTER_MARGIN_RANGE)
    bottom = random.randint(*OUTER_MARGIN_RANGE)
    left = random.randint(*OUTER_MARGIN_RANGE)
    right = random.randint(*OUTER_MARGIN_RANGE)
    gap_r = random.randint(*CELL_GAP_RANGE)
    gap_c = random.randint(*CELL_GAP_RANGE)
    h0 = random.randint(*CELL_SIZE_RANGE)
    h1 = random.randint(*CELL_SIZE_RANGE)
    w0 = random.randint(*CELL_SIZE_RANGE)
    w1 = random.randint(*CELL_SIZE_RANGE)
    rows = top + h0 + gap_r + h1 + bottom
    cols = left + w0 + gap_c + w1 + right
    boxes = [
        (top, left, top + h0 - 1, left + w0 - 1),
        (top, left + w0 + gap_c, top + h0 - 1, left + w0 + gap_c + w1 - 1),
        (top + h0 + gap_r, left, top + h0 + gap_r + h1 - 1, left + w0 - 1),
        (
            top + h0 + gap_r,
            left + w0 + gap_c,
            top + h0 + gap_r + h1 - 1,
            left + w0 + gap_c + w1 - 1,
        ),
    ]
    return (rows, cols), boxes


def _target_patterns(count: int) -> list[tuple[str, str]]:
    pool = list(PATTERN_CATALOG)
    random.shuffle(pool)
    if count <= len(pool):
        return pool[:count]
    out = []
    while len(out) < count:
        random.shuffle(pool)
        out.extend(pool)
    return out[:count]


def _target_slopes(count: int) -> list[int]:
    slopes = [1, -1]
    while len(slopes) < count:
        slopes.append(random.choice((-1, 1)))
    random.shuffle(slopes)
    return slopes[:count]


def _render_input(shape: tuple[int, int], pairs: list[PairSpec]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for pair in pairs:
        for bar in pair["bars"]:
            color = int(bar["color"])
            for r, c in bar["pixels"]:
                assert grid[r, c] == BG
                grid[r, c] = color
    return grid


def _expected_from_pairs(shape: tuple[int, int], pairs: list[PairSpec]) -> np.ndarray:
    bars = [bar for pair in pairs for bar in pair["bars"]]
    pair_info = {int(pair["color"]): dict(pair["connector"]) for pair in pairs}
    return _render_rotated_output(shape, bars, pair_info, BG)


def _semantic_validator(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        bars, pair_info = _analyze_scene(inp, require_open=True)
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    rebuilt = _render_rotated_output(inp.shape, bars, pair_info, _get_bg(inp))
    if not np.array_equal(rebuilt, expected):
        return False
    if len({tuple(sorted(info["path"])) for info in pair_info.values()}) != len(pair_info):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shape, boxes = _sample_layout()
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    chosen_cells = random.sample(range(4), pair_count)
    colors = random.sample(PAIR_COLORS, pair_count)
    patterns = _target_patterns(pair_count)
    slopes = _target_slopes(pair_count)

    pairs: list[PairSpec] = []
    for cell_idx, color, pattern, slope in zip(chosen_cells, colors, patterns, slopes):
        pair = _construct_pair(boxes[cell_idx], color, pattern, slope)
        if pair is None:
            return None
        pairs.append(pair)

    inp = _render_input(shape, pairs)
    expected = _expected_from_pairs(shape, pairs)
    if not _semantic_validator(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GRID_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotate-then-connect sample")
