from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_fable_2_036_4", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load arc_utils from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
Color = _ARC.Color
bbox_coords = _ARC.bbox_coords
crop_bbox = _ARC.crop_bbox
find_components = _ARC.find_components
get_bg = _ARC.get_bg
paste = _ARC.paste


BLACK = Color.BLACK
BASE_CANVAS_SHAPE = (30, 30)
COUNT_VALUES = (1, 2, 3)
VALID_LENGTHS = {
    1: (4, 5, 6, 7, 8),
    2: (4, 6, 7, 8),
    3: (6, 8),
}
COUNTS_BY_LENGTH = {
    4: (1, 2),
    5: (1,),
    6: (1, 2, 3),
    7: (1, 2),
    8: (1, 2, 3),
}
MODES = ("mirror_pair", "quota_pair", "mixed")
GEN_ATTEMPTS = 500
LEGEND_KEY_COL = 1
LEGEND_ROW_START = 1
LEGEND_ROW_STEP = 2
SIDE_GAP = 2
OBJECT_PAD = 1
MARGIN_RANGE = (1, 3)
GEN_CALL_COUNT = 0


def _point_to_bbox_distance(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> int:
    r, c = point
    r0, c0, r1, c1 = bbox
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return int(dr + dc)


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _bboxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = OBJECT_PAD) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _legend_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    candidates = []
    for comp in find_components(
        grid,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        colors=[BLACK],
        return_format="dict",
    ):
        box = tuple(map(int, comp["bbox"]))
        patch = crop_bbox(grid, box)
        if np.any((patch != int(bg)) & (patch != BLACK)):
            candidates.append(comp)
    return candidates


def _find_legend_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    candidates = _legend_candidates(grid, bg)
    assert candidates, "expected a black legend box"
    candidates.sort(key=lambda comp: (int(comp["area"]), tuple(map(int, comp["bbox"]))), reverse=True)
    return tuple(map(int, candidates[0]["bbox"]))


def _parse_legend(
    grid: np.ndarray,
    legend_bbox: tuple[int, int, int, int],
    bg: int,
) -> tuple[dict[int, int], dict[int, int]]:
    box = crop_bbox(grid, legend_bbox)
    counts: dict[int, int] = {}
    key_rows: dict[int, int] = {}
    occupied = np.argwhere((box != int(bg)) & (box != BLACK))
    assert occupied.size > 0, "legend must contain colored entries"

    for rr in range(box.shape[0]):
        cols = np.flatnonzero((box[rr] != int(bg)) & (box[rr] != BLACK))
        if cols.size == 0:
            continue
        key_col = int(cols[0])
        color = int(box[rr, key_col])
        assert color not in counts, "legend colors must be unique"
        assert np.all(box[rr, cols] == color), "each legend row must use one color"
        expected_cols = np.array([key_col] + [key_col + 2 * i for i in range(1, cols.size)], dtype=int)
        assert np.array_equal(cols, expected_cols), "legend dots must be spaced every other cell"
        counts[color] = int(cols.size - 1)
        key_rows[color] = int(rr)

    assert counts, "expected at least one legend row"
    return counts, key_rows


def _bar_info(
    grid: np.ndarray,
    comp: dict,
    legend_bbox: tuple[int, int, int, int],
) -> dict:
    color = int(comp["color"])
    bbox = tuple(map(int, comp["bbox"]))
    r0, c0, r1, c1 = bbox
    patch = crop_bbox(grid, bbox)
    h, w = map(int, patch.shape)
    assert np.all(patch == color), "bar must be a solid monochromatic rectangle"

    if h == 3 and w >= 4:
        orientation = "h"
        length = w
        ends = {
            "left": (r0 + 1, c0),
            "right": (r0 + 1, c1),
        }
    elif w == 3 and h >= 4:
        orientation = "v"
        length = h
        ends = {
            "top": (r0, c0 + 1),
            "bottom": (r1, c0 + 1),
        }
    else:
        raise AssertionError("outside-legend components must be 3-wide bars")

    distances = {side: _point_to_bbox_distance(point, legend_bbox) for side, point in ends.items()}
    start_side = min(distances, key=distances.get)
    sorted_distances = sorted(distances.values())
    assert len(sorted_distances) == 2 and sorted_distances[0] < sorted_distances[1], "near end must be unique"
    return {
        "color": color,
        "bbox": bbox,
        "orientation": orientation,
        "length": int(length),
        "start_side": start_side,
    }


def _hole_coords(bar: dict, count: int) -> tuple[tuple[int, int], ...]:
    assert count > 0
    r0, c0, r1, c1 = map(int, bar["bbox"])
    length = int(bar["length"])
    start_side = str(bar["start_side"])
    coords: list[tuple[int, int]] = []

    if bar["orientation"] == "h":
        row = r0 + 1
        if start_side == "left":
            cols = [c0 + 1 + 2 * i for i in range(count)]
        else:
            cols = [c1 - 1 - 2 * i for i in range(count)]
        coords = [(row, int(col)) for col in cols]
    else:
        col = c0 + 1
        if start_side == "top":
            rows = [r0 + 1 + 2 * i for i in range(count)]
        else:
            rows = [r1 - 1 - 2 * i for i in range(count)]
        coords = [(int(row), col) for row in rows]

    assert len(coords) == count
    assert len(set(coords)) == count
    assert count <= length // 2, "bar must be long enough for requested holes"
    if length % 2 == 1:
        assert count < (length - 1) // 2, "odd bars may not fill the whole alternating parity"
    return tuple(coords)


def _opposite_start_side(bar: dict) -> str:
    if bar["orientation"] == "h":
        return "right" if bar["start_side"] == "left" else "left"
    return "bottom" if bar["start_side"] == "top" else "top"


def _bar_with_start(bar: dict, start_side: str) -> dict:
    updated = dict(bar)
    updated["start_side"] = start_side
    return updated


def _find_bars(
    grid: np.ndarray,
    bg: int,
    legend_bbox: tuple[int, int, int, int],
    key_colors: set[int],
) -> dict[int, dict]:
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=sorted(key_colors),
        return_format="dict",
    )
    bars: dict[int, dict] = {}
    for comp in comps:
        bbox = tuple(map(int, comp["bbox"]))
        area = int(comp["area"])
        if _bbox_contains(legend_bbox, bbox):
            continue
        if area < 12:
            continue
        info = _bar_info(grid, comp, legend_bbox)
        color = int(info["color"])
        assert color not in bars, "expected one outside bar per legend color"
        bars[color] = info

    assert set(bars) == set(key_colors), "legend colors and bar colors must match one-to-one"
    return bars


def _solve_with_trace(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg != BLACK, "generator uses a non-black background so the legend box is visible"

    legend_bbox = _find_legend_bbox(g, bg)
    counts, key_rows = _parse_legend(g, legend_bbox, bg)
    bars = _find_bars(g, bg, legend_bbox, set(counts))

    out = g.copy()
    trace_bars: dict[int, dict] = {}
    for color, count in counts.items():
        bar = bars[int(color)]
        holes = _hole_coords(bar, int(count))
        for rr, cc in holes:
            out[int(rr), int(cc)] = bg
        trace_bars[int(color)] = {
            **bar,
            "count": int(count),
            "holes": holes,
            "legend_row": int(key_rows[int(color)]),
        }

    return out, {
        "bg": bg,
        "legend_bbox": legend_bbox,
        "counts": counts,
        "bars": trace_bars,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Punch each color's quota of holes into the matching 3-wide bar, using the legend count and the end nearest the legend.

    Input structure:
    - The background is a single non-black color, and one black legend box contains one keyed row per active color.
    - In each used legend row, the leftmost colored pixel is the key and same-color dots continue to the right with one black gap between dots.
    - Outside the legend, each key color appears exactly once as a solid monochromatic rectangle whose short side is 3 and whose long side may be horizontal or vertical.

    Rule:
    - Count the dots to the right of each legend key pixel; that count is the number of holes for the same-colored bar.
    - For each bar, look only at its long-axis center line and choose the physical end whose center cell is nearer to the legend box.
    - Starting from the second center-line cell in from that nearer end, erase every other center-line cell to the background color until the required number of holes has been punched.
    - Leave the legend box, all other pixels of each bar, and the rest of the grid unchanged.

    Key insight:
    - The bar itself does not reveal how many holes to make, and the legend alone does not reveal which parity to erase; the output is determined only by combining legend counts with the bar end that is spatially closest to the legend.
    """
    return _solve_with_trace(grid)[0]


def _legend_patch(colors: list[int], counts: list[int]) -> np.ndarray:
    height = LEGEND_ROW_START + LEGEND_ROW_STEP * len(colors)
    width = 2 * max(counts) + 3 + random.randint(0, 1)
    patch = np.full((height, width), BLACK, dtype=int)
    for i, (color, count) in enumerate(zip(colors, counts)):
        row = LEGEND_ROW_START + LEGEND_ROW_STEP * i
        patch[row, LEGEND_KEY_COL] = int(color)
        for j in range(count):
            patch[row, LEGEND_KEY_COL + 2 * (j + 1)] = int(color)
    return patch


def _valid_lengths_for_count(count: int) -> tuple[int, ...]:
    return VALID_LENGTHS[int(count)]


def _region_start_side(orientation: str, region_side: str) -> str:
    if orientation == "h":
        return "right" if region_side == "west" else "left"
    return "bottom" if region_side == "north" else "top"


def _bar_shape(spec: dict) -> tuple[int, int]:
    if spec["orientation"] == "h":
        return 3, int(spec["length"])
    return int(spec["length"]), 3


def _bar_bbox_from_top_left(spec: dict, r0: int, c0: int) -> tuple[int, int, int, int]:
    h, w = _bar_shape(spec)
    return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)


def _region_ok(
    bbox: tuple[int, int, int, int],
    legend_bbox: tuple[int, int, int, int],
    orientation: str,
    region_side: str,
) -> bool:
    if orientation == "h":
        if region_side == "west":
            return bbox[3] <= legend_bbox[1] - SIDE_GAP - 1
        return bbox[1] >= legend_bbox[3] + SIDE_GAP + 1
    if region_side == "north":
        return bbox[2] <= legend_bbox[0] - SIDE_GAP - 1
    return bbox[0] >= legend_bbox[2] + SIDE_GAP + 1


def _bbox_center_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar = a[0] + a[2]
    ac = a[1] + a[3]
    br = b[0] + b[2]
    bc = b[1] + b[3]
    return int(abs(ar - br) + abs(ac - bc))


def _place_bars(
    legend_bbox: tuple[int, int, int, int],
    specs: list[dict],
) -> list[dict] | None:
    occupied = [legend_bbox]
    placed: list[dict] = []
    H, W = BASE_CANVAS_SHAPE

    for spec in sorted(specs, key=lambda item: (int(item["length"]), item["orientation"]), reverse=True):
        h, w = _bar_shape(spec)
        candidates = []
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                bbox = _bar_bbox_from_top_left(spec, r0, c0)
                if not _region_ok(bbox, legend_bbox, spec["orientation"], spec["region_side"]):
                    continue
                if not all(_bboxes_separated(bbox, other) for other in occupied):
                    continue
                candidates.append((bbox, r0, c0))

        if not candidates:
            return None

        candidates.sort(key=lambda item: _bbox_center_distance(item[0], legend_bbox))
        pool = candidates[: min(len(candidates), 40)]
        bbox, r0, c0 = random.choice(pool)
        placed_spec = dict(spec)
        placed_spec["bbox"] = bbox
        placed_spec["top_left"] = (int(r0), int(c0))
        placed_spec["start_side"] = _region_start_side(spec["orientation"], spec["region_side"])
        placed.append(placed_spec)
        occupied.append(bbox)

    return placed


def _trim_canvas(inp: np.ndarray, out: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    coords = np.argwhere(inp != int(bg))
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(coords)
    for _ in range(30):
        mt = random.randint(*MARGIN_RANGE)
        mb = random.randint(*MARGIN_RANGE)
        ml = random.randint(*MARGIN_RANGE)
        mr = random.randint(*MARGIN_RANGE)
        tr0 = max(0, int(r0) - mt)
        tc0 = max(0, int(c0) - ml)
        tr1 = min(inp.shape[0] - 1, int(r1) + mb)
        tc1 = min(inp.shape[1] - 1, int(c1) + mr)
        cropped_inp = inp[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()
        cropped_out = out[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()
        if max(cropped_inp.shape) > 30:
            continue
        if int(get_bg(cropped_inp)) != int(bg):
            continue
        return cropped_inp, cropped_out
    return None


def _sample_palette(n_colors: int) -> tuple[int, list[int]]:
    palette = list(Color.NOT_BLACK)
    bg = random.choice(palette)
    remaining = [color for color in palette if color != bg]
    colors = random.sample(remaining, k=n_colors)
    return int(bg), [int(color) for color in colors]


def _same_length_diff_count_pair() -> tuple[int, int, int]:
    length = random.choice([4, 6, 7, 8])
    a, b = random.sample(COUNTS_BY_LENGTH[int(length)], 2)
    return int(length), int(a), int(b)


def _same_count_diff_length_pair() -> tuple[int, int, int]:
    count = random.choice([1, 2, 3])
    length_a, length_b = random.sample(_valid_lengths_for_count(count), 2)
    return int(count), int(length_a), int(length_b)


def _build_specs(mode: str, colors: list[int]) -> list[dict]:
    specs: list[dict] = []
    n_colors = len(colors)

    if mode == "mirror_pair":
        pair_orientation = random.choice(["h", "v"])
        pair_count = random.choice(COUNT_VALUES)
        pair_length = random.choice(_valid_lengths_for_count(pair_count))
        pair_sides = ("west", "east") if pair_orientation == "h" else ("north", "south")
        specs.extend(
            [
                {
                    "color": colors[0],
                    "count": int(pair_count),
                    "length": int(pair_length),
                    "orientation": pair_orientation,
                    "region_side": pair_sides[0],
                },
                {
                    "color": colors[1],
                    "count": int(pair_count),
                    "length": int(pair_length),
                    "orientation": pair_orientation,
                    "region_side": pair_sides[1],
                },
            ]
        )
        remaining_orientation = "v" if pair_orientation == "h" else "h"
        for idx in range(2, n_colors):
            orientation = remaining_orientation if idx == 2 else random.choice(["h", "v"])
            region_side = random.choice(["west", "east"]) if orientation == "h" else random.choice(["north", "south"])
            count = random.choice(COUNT_VALUES)
            length = random.choice(_valid_lengths_for_count(count))
            specs.append(
                {
                    "color": colors[idx],
                    "count": int(count),
                    "length": int(length),
                    "orientation": orientation,
                    "region_side": region_side,
                }
            )
    elif mode == "quota_pair":
        pair_orientation = random.choice(["h", "v"])
        pair_side = random.choice(["west", "east"]) if pair_orientation == "h" else random.choice(["north", "south"])
        pair_length, count_a, count_b = _same_length_diff_count_pair()
        specs.extend(
            [
                {
                    "color": colors[0],
                    "count": int(count_a),
                    "length": int(pair_length),
                    "orientation": pair_orientation,
                    "region_side": pair_side,
                },
                {
                    "color": colors[1],
                    "count": int(count_b),
                    "length": int(pair_length),
                    "orientation": pair_orientation,
                    "region_side": pair_side,
                },
            ]
        )
        remaining_orientation = "v" if pair_orientation == "h" else "h"
        for idx in range(2, n_colors):
            orientation = remaining_orientation if idx == 2 else random.choice(["h", "v"])
            region_side = random.choice(["west", "east"]) if orientation == "h" else random.choice(["north", "south"])
            count = random.choice(COUNT_VALUES)
            length = random.choice(_valid_lengths_for_count(count))
            specs.append(
                {
                    "color": colors[idx],
                    "count": int(count),
                    "length": int(length),
                    "orientation": orientation,
                    "region_side": region_side,
                }
            )
    elif mode == "mixed":
        pair_type = random.choice(["same_length_diff_count", "same_count_diff_length"])
        if pair_type == "same_length_diff_count":
            pair_length, count_a, count_b = _same_length_diff_count_pair()
            pair_data = ((count_a, pair_length), (count_b, pair_length))
        else:
            pair_count, length_a, length_b = _same_count_diff_length_pair()
            pair_data = ((pair_count, length_a), (pair_count, length_b))

        first_orientation = random.choice(["h", "v"])
        second_orientation = "v" if first_orientation == "h" else "h"
        orientations = [first_orientation, second_orientation]
        if n_colors > 2:
            orientations.extend(random.choice(["h", "v"]) for _ in range(n_colors - 2))
        if "h" not in orientations:
            orientations[0] = "h"
        if "v" not in orientations:
            orientations[1] = "v"

        for idx, color in enumerate(colors):
            orientation = orientations[idx]
            region_side = random.choice(["west", "east"]) if orientation == "h" else random.choice(["north", "south"])
            if idx < 2:
                count, length = pair_data[idx]
            else:
                count = random.choice(COUNT_VALUES)
                length = random.choice(_valid_lengths_for_count(count))
            specs.append(
                {
                    "color": color,
                    "count": int(count),
                    "length": int(length),
                    "orientation": orientation,
                    "region_side": region_side,
                }
            )
    else:
        raise ValueError(f"unknown mode: {mode}")

    orientations = {spec["orientation"] for spec in specs}
    assert orientations == {"h", "v"}, "generator must cover both bar orientations in each sample"
    return specs


def _legend_position_range(legend_shape: tuple[int, int], specs: list[dict]) -> tuple[range, range]:
    legend_h, legend_w = map(int, legend_shape)
    west_need = max(
        [int(spec["length"]) for spec in specs if spec["orientation"] == "h" and spec["region_side"] == "west"],
        default=0,
    )
    east_need = max(
        [int(spec["length"]) for spec in specs if spec["orientation"] == "h" and spec["region_side"] == "east"],
        default=0,
    )
    north_need = max(
        [int(spec["length"]) for spec in specs if spec["orientation"] == "v" and spec["region_side"] == "north"],
        default=0,
    )
    south_need = max(
        [int(spec["length"]) for spec in specs if spec["orientation"] == "v" and spec["region_side"] == "south"],
        default=0,
    )

    row_min = north_need + SIDE_GAP + 1
    row_max = BASE_CANVAS_SHAPE[0] - legend_h - south_need - SIDE_GAP - 1
    col_min = west_need + SIDE_GAP + 1
    col_max = BASE_CANVAS_SHAPE[1] - legend_w - east_need - SIDE_GAP - 1
    if row_min > row_max or col_min > col_max:
        raise ValueError("legend cannot fit with current side assignments")
    return range(row_min, row_max + 1), range(col_min, col_max + 1)


def _render_attempt(mode: str) -> tuple[np.ndarray, np.ndarray, dict] | None:
    n_colors = 3 if mode == "mirror_pair" else random.choice([3, 4])
    if mode == "mixed":
        n_colors = random.choice([2, 3, 4])

    bg, colors = _sample_palette(n_colors)
    specs = _build_specs(mode, colors)
    counts = [int(spec["count"]) for spec in specs]
    legend = _legend_patch(colors, counts)

    try:
        row_range, col_range = _legend_position_range(legend.shape, specs)
    except ValueError:
        return None

    full_inp = np.full(BASE_CANVAS_SHAPE, int(bg), dtype=int)
    legend_r0 = random.choice(list(row_range))
    legend_c0 = random.choice(list(col_range))
    legend_bbox = (legend_r0, legend_c0, legend_r0 + legend.shape[0] - 1, legend_c0 + legend.shape[1] - 1)
    paste(full_inp, legend, legend_r0, legend_c0)

    placed = _place_bars(legend_bbox, specs)
    if placed is None:
        return None

    for spec in placed:
        r0, c0 = spec["top_left"]
        patch = np.full(_bar_shape(spec), int(spec["color"]), dtype=int)
        paste(full_inp, patch, int(r0), int(c0))

    full_out = full_inp.copy()
    for spec in placed:
        bar = {
            "bbox": tuple(map(int, spec["bbox"])),
            "orientation": spec["orientation"],
            "length": int(spec["length"]),
            "start_side": spec["start_side"],
        }
        holes = _hole_coords(bar, int(spec["count"]))
        spec["holes"] = holes
        for rr, cc in holes:
            full_out[int(rr), int(cc)] = int(bg)

    trimmed = _trim_canvas(full_inp, full_out, bg)
    if trimmed is None:
        return None
    inp, out = trimmed

    return inp, out, {"mode": mode, "bg": int(bg), "specs": placed}


def _has_same_length_diff_count(bars: list[dict]) -> bool:
    for i, a in enumerate(bars):
        for b in bars[i + 1 :]:
            if int(a["length"]) == int(b["length"]) and int(a["count"]) != int(b["count"]):
                return True
    return False


def _has_same_count_diff_length(bars: list[dict]) -> bool:
    for i, a in enumerate(bars):
        for b in bars[i + 1 :]:
            if int(a["count"]) == int(b["count"]) and int(a["length"]) != int(b["length"]):
                return True
    return False


def _has_mirror_pair(bars: list[dict]) -> bool:
    for i, a in enumerate(bars):
        for b in bars[i + 1 :]:
            if (
                int(a["length"]) == int(b["length"])
                and int(a["count"]) == int(b["count"])
                and a["orientation"] == b["orientation"]
                and a["start_side"] != b["start_side"]
            ):
                return True
    return False


def _validate_attempt(inp: np.ndarray, expected: np.ndarray, mode: str) -> bool:
    try:
        actual, trace = _solve_with_trace(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    bg = int(trace["bg"])
    if int(get_bg(inp)) != bg:
        return False

    bars = list(trace["bars"].values())
    if {bar["orientation"] for bar in bars} != {"h", "v"}:
        return False

    for bar in bars:
        opposite = _hole_coords(_bar_with_start(bar, _opposite_start_side(bar)), int(bar["count"]))
        if set(opposite) == set(bar["holes"]):
            return False

    if mode == "mirror_pair" and not _has_mirror_pair(bars):
        return False
    if mode == "quota_pair" and not _has_same_length_diff_count(bars):
        return False
    if mode == "mixed" and not (_has_same_length_diff_count(bars) or _has_same_count_diff_length(bars)):
        return False

    return True


def _gen_mode() -> str:
    global GEN_CALL_COUNT
    mode = MODES[GEN_CALL_COUNT % len(MODES)]
    GEN_CALL_COUNT += 1
    return mode


def generate_input() -> np.ndarray:
    mode = _gen_mode()
    for _ in range(GEN_ATTEMPTS):
        attempt = _render_attempt(mode)
        if attempt is None:
            continue
        inp, expected, meta = attempt
        if not _validate_attempt(inp, expected, meta["mode"]):
            continue
        return inp
    raise RuntimeError("failed to generate a valid legend-controlled hole-punch input")
