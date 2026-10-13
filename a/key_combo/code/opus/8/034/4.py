from __future__ import annotations

import hashlib
import itertools
import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

GRID_HEIGHT_RANGE = (13, 18)
GRID_WIDTH_RANGE = (15, 21)
SOURCE_BBOX_RANGE = (3, 5)
SOURCE_PIXELS_RANGE = (5, 10)
MARKER_GAPS = (4, 5, 6, 7, 8)
GENERATION_RETRIES = 500

_SEEN_INPUTS: set[str] = set()
_SEEN_OUTPUTS: set[str] = set()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _connected(coords: set[tuple[int, int]]) -> bool:
    start = next(iter(coords))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _shape_sig(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _parse_input(grid: np.ndarray) -> tuple[set[tuple[int, int]], list[dict]]:
    g = np.asarray(grid, dtype=int)
    colors = {int(v) for v in np.unique(g)}
    assert colors.issubset({BLACK, BLUE, RED, GREY})

    source_coords = {tuple(map(int, rc)) for rc in np.argwhere(g == GREY)}
    assert 5 <= len(source_coords) <= 10
    assert _connected(source_coords)

    bbox = _bbox_from_coords(source_coords)
    r0, c0, r1, c1 = bbox
    assert 1 <= r1 - r0 + 1 <= 5 and 1 <= c1 - c0 + 1 <= 5

    markers = []
    for color in (BLUE, RED):
        pts = np.argwhere(g == color)
        if pts.size == 0:
            continue
        assert len(pts) == 1
        r, c = map(int, pts[0])
        markers.append({"color": color, "pos": (r, c)})
    assert 1 <= len(markers) <= 2
    return source_coords, markers


def _complement_mask_coords(source_coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox_from_coords(source_coords)
    comp = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) not in source_coords:
                comp.append((r, c))
    assert len(comp) >= 2
    return comp


def _find_anchor(source_coords: set[tuple[int, int]], marker: tuple[int, int]) -> tuple[str, tuple[int, int]]:
    mr, mc = marker
    row_hits = [pt for pt in source_coords if pt[0] == mr]
    col_hits = [pt for pt in source_coords if pt[1] == mc]
    row_ok = len(row_hits) == 1
    col_ok = len(col_hits) == 1
    assert row_ok != col_ok
    if row_ok:
        return "row", row_hits[0]
    return "col", col_hits[0]


def _reflect_coords(
    mask_coords: list[tuple[int, int]],
    marker: tuple[int, int],
    anchor: tuple[int, int],
    axis_kind: str,
) -> list[tuple[int, int]]:
    mr, mc = marker
    ar, ac = anchor
    out = []
    if axis_kind == "row":
        for r, c in mask_coords:
            out.append((r, ac + mc - c))
    else:
        for r, c in mask_coords:
            out.append((ar + mr - r, c))
    return out


def _apply_rule(grid: np.ndarray, *, use_complement: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    source_coords, markers = _parse_input(g)
    stamp_coords = _complement_mask_coords(source_coords) if use_complement else sorted(source_coords)

    out = g.copy()
    H, W = out.shape
    markers = sorted(markers, key=lambda item: (item["pos"][1], item["pos"][0]))
    for marker in markers:
        axis_kind, anchor = _find_anchor(source_coords, marker["pos"])
        targets = _reflect_coords(stamp_coords, marker["pos"], anchor, axis_kind)
        for r, c in targets:
            if 0 <= r < H and 0 <= c < W:
                out[r, c] = int(marker["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror the bounding-box complement of the grey source and stamp it in each marker color.

    Input structure:
    - A black background contains one irregular grey 4-connected source object.
    - The grey object's bounding box contains at least two black cells.
    - There are one or two singleton markers, colored blue and/or red, and each marker shares exactly one row or exactly one column with exactly one grey source pixel.

    Rule:
    - Keep the grey source unchanged and compute the complement mask inside its bounding box: every bbox cell that is not grey belongs to the stamp.
    - For each marker, find the unique aligned grey anchor pixel. A shared row defines a vertical mirror axis midway between the marker and anchor; a shared column defines a horizontal mirror axis midway between them.
    - Reflect every complement-mask pixel across that axis, color the reflected copy with the marker color, discard out-of-bounds reflected pixels, and process markers from left to right, then top to bottom for ties.

    Key insight:
    - The marker does not mirror the grey shape itself; it mirrors the grey shape's internal bbox background pattern, and the anchor-marker alignment alone fixes the mirror axis.
    """
    return _apply_rule(grid, use_complement=True)


def _row_col_counts(coords: set[tuple[int, int]]) -> tuple[Counter, Counter]:
    row_counts = Counter(r for r, _ in coords)
    col_counts = Counter(c for _, c in coords)
    return row_counts, col_counts


def _generate_source_shape() -> set[tuple[int, int]] | None:
    h = random.randint(*SOURCE_BBOX_RANGE)
    w = random.randint(*SOURCE_BBOX_RANGE)
    area = h * w
    max_pixels = min(SOURCE_PIXELS_RANGE[1], area - 2)
    if max_pixels < SOURCE_PIXELS_RANGE[0]:
        return None
    n = random.randint(SOURCE_PIXELS_RANGE[0], max_pixels)

    for _ in range(120):
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < n:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != n:
            continue
        if not _connected(cells):
            continue
        rs = {r for r, _ in cells}
        cs = {c for _, c in cells}
        if 0 not in rs or h - 1 not in rs or 0 not in cs or w - 1 not in cs:
            continue
        complement = area - len(cells)
        if complement < 2:
            continue

        row_counts, col_counts = _row_col_counts(cells)
        if not any(v == 1 for v in row_counts.values()) and not any(v == 1 for v in col_counts.values()):
            continue
        return cells
    return None


def _marker_options(
    source_coords: set[tuple[int, int]],
    complement_coords: list[tuple[int, int]],
) -> list[dict]:
    row_counts, col_counts = _row_col_counts(source_coords)
    source_cols = {c for _, c in source_coords}
    source_rows = {r for r, _ in source_coords}
    options: list[dict] = []

    for ar, ac in source_coords:
        if row_counts[ar] == 1:
            for side in (-1, 1):
                for dist in MARKER_GAPS:
                    marker = (ar, ac + side * dist)
                    if marker[1] in source_cols:
                        continue
                    targets = _reflect_coords(complement_coords, marker, (ar, ac), "row")
                    if set(targets) & source_coords:
                        continue
                    options.append(
                        {
                            "axis": "row",
                            "anchor": (ar, ac),
                            "marker": marker,
                            "targets": tuple(targets),
                        }
                    )
        if col_counts[ac] == 1:
            for side in (-1, 1):
                for dist in MARKER_GAPS:
                    marker = (ar + side * dist, ac)
                    if marker[0] in source_rows:
                        continue
                    targets = _reflect_coords(complement_coords, marker, (ar, ac), "col")
                    if set(targets) & source_coords:
                        continue
                    options.append(
                        {
                            "axis": "col",
                            "anchor": (ar, ac),
                            "marker": marker,
                            "targets": tuple(targets),
                        }
                    )
    return options


def _pair_is_clear(opt_a: dict, opt_b: dict) -> bool:
    if opt_a["marker"] == opt_b["marker"]:
        return False
    marker_positions = {opt_a["marker"], opt_b["marker"]}
    if marker_positions & set(opt_a["targets"]):
        return False
    if marker_positions & set(opt_b["targets"]):
        return False
    if set(opt_a["targets"]) & set(opt_b["targets"]):
        return False
    return True


def _choose_marker_specs(options: list[dict]) -> list[dict] | None:
    row_opts = [opt for opt in options if opt["axis"] == "row"]
    col_opts = [opt for opt in options if opt["axis"] == "col"]

    categories: list[tuple[int, str, list[tuple[dict, ...]]]] = []
    if row_opts:
        categories.append((3, "single_row", [(opt,) for opt in row_opts]))
    if col_opts:
        categories.append((3, "single_col", [(opt,) for opt in col_opts]))

    hv_pairs = [(a, b) for a in row_opts for b in col_opts if _pair_is_clear(a, b)]
    hh_pairs = [pair for pair in itertools.combinations(row_opts, 2) if _pair_is_clear(*pair)]
    vv_pairs = [pair for pair in itertools.combinations(col_opts, 2) if _pair_is_clear(*pair)]
    if hv_pairs:
        categories.append((4, "pair_hv", hv_pairs))
    if hh_pairs:
        categories.append((1, "pair_hh", hh_pairs))
    if vv_pairs:
        categories.append((1, "pair_vv", vv_pairs))
    if not categories:
        return None

    weights = [weight for weight, _, combos in categories if combos]
    chosen_index = random.choices(range(len(categories)), weights=weights, k=1)[0]
    combos = categories[chosen_index][2]
    return [dict(item) for item in random.choice(combos)]


def _shift_bounds(points: set[tuple[int, int]], H: int, W: int) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox_from_coords(points)
    min_dr = -r0
    max_dr = H - 1 - r1
    min_dc = -c0
    max_dc = W - 1 - c1
    if min_dr > max_dr or min_dc > max_dc:
        return None
    return min_dr, max_dr, min_dc, max_dc


def _render_case(
    H: int,
    W: int,
    source_coords: set[tuple[int, int]],
    marker_specs: list[dict],
    dr: int,
    dc: int,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)

    colors = [BLUE, RED]
    if len(marker_specs) == 1:
        colors = [random.choice(colors)]
    else:
        random.shuffle(colors)

    global_markers = []
    for (r, c) in source_coords:
        inp[dr + r, dc + c] = GREY
        out[dr + r, dc + c] = GREY

    for spec, color in zip(marker_specs, colors):
        mr, mc = spec["marker"]
        global_markers.append((dc + mc, dr + mr, color, spec))
        inp[dr + mr, dc + mc] = color
        out[dr + mr, dc + mc] = color

    global_markers.sort(key=lambda item: (item[0], item[1]))
    for _gc, _gr, color, spec in global_markers:
        for r, c in spec["targets"]:
            rr = dr + r
            cc = dc + c
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = color
    return inp, out


def _visible_target_count(spec: dict, H: int, W: int, dr: int, dc: int) -> int:
    count = 0
    for r, c in spec["targets"]:
        rr = dr + r
        cc = dc + c
        if 0 <= rr < H and 0 <= cc < W:
            count += 1
    return count


def _validate_generated_case(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        out = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(out, expected):
        return False
    if np.array_equal(out, inp):
        return False

    source_coords, markers = _parse_input(inp)
    comp = _complement_mask_coords(source_coords)
    if len(comp) < 2:
        return False

    wrong_shape = _apply_rule(inp, use_complement=False)
    if np.array_equal(wrong_shape, out):
        return False

    for marker in markers:
        if int(np.count_nonzero(out == marker["color"])) <= 1:
            return False

    if len(markers) == 2:
        for marker in markers:
            reduced = inp.copy()
            rr, cc = marker["pos"]
            reduced[rr, cc] = BLACK
            try:
                reduced_out = main(reduced)
            except AssertionError:
                return False
            if np.array_equal(reduced_out, out):
                return False
    return True


def _generate_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_HEIGHT_RANGE)
    W = random.randint(*GRID_WIDTH_RANGE)

    source_coords = _generate_source_shape()
    if source_coords is None:
        return None
    complement_coords = _complement_mask_coords(source_coords)
    options = _marker_options(source_coords, complement_coords)
    marker_specs = _choose_marker_specs(options)
    if marker_specs is None:
        return None

    occupied = set(source_coords)
    for spec in marker_specs:
        occupied.add(spec["marker"])
        occupied.update(spec["targets"])
    bounds = _shift_bounds(occupied, H, W)
    if bounds is None:
        return None

    min_dr, max_dr, min_dc, max_dc = bounds
    for _ in range(40):
        dr = random.randint(min_dr, max_dr)
        dc = random.randint(min_dc, max_dc)
        if any(_visible_target_count(spec, H, W, dr, dc) < 2 for spec in marker_specs):
            continue
        inp, expected = _render_case(H, W, source_coords, marker_specs, dr, dc)
        if not _validate_generated_case(inp, expected):
            continue

        inp_sig = _shape_sig(inp)
        out_sig = _shape_sig(expected)
        if inp_sig in _SEEN_INPUTS or out_sig in _SEEN_OUTPUTS:
            continue
        _SEEN_INPUTS.add(inp_sig)
        _SEEN_OUTPUTS.add(out_sig)
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
