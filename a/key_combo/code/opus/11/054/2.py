from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np

_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_shared_arc_utils", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg


def _legend_entries(grid: np.ndarray, bg: int) -> list[dict]:
    entries = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 != 0:
            continue
        assert c0 == c1, "legend entries must be single columns"
        color = int(comp["color"])
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)
        entries.append({"color": color, "height": int(r1 - r0 + 1), "col": c0, "bbox": (r0, c0, r1, c1)})

    assert entries, "expected top-row legend entries"
    heights = sorted(entry["height"] for entry in entries)
    assert heights == list(range(1, len(entries) + 1)), "legend heights must be exactly the rank set 1..N"
    assert len({entry["color"] for entry in entries}) == len(entries), "legend colors must be unique"
    return sorted(entries, key=lambda entry: entry["col"])


def _frame_objects(grid: np.ndarray, bg: int, legend_colors: set[int]) -> list[dict]:
    frame_colors = sorted(set(map(int, np.unique(grid))) - {bg, Color.GREY} - set(legend_colors))
    frames = find_components(
        grid,
        bg=bg,
        colors=frame_colors,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert 3 <= len(frames) <= 4, "expected 3-4 colored square frames"

    grey_cover = np.zeros(grid.shape, dtype=int)
    parsed = []
    for frame in frames:
        color = int(frame["color"])
        r0, c0, r1, c1 = map(int, frame["bbox"])
        side_h = int(r1 - r0 + 1)
        side_w = int(c1 - c0 + 1)
        assert side_h == side_w and 3 <= side_h <= 5, "frames must be 3x3 to 5x5 squares"

        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((side_h, side_h), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.all(region[border] == color), "frame border must be monochrome"

        inner = region[1:-1, 1:-1]
        assert np.all((inner == bg) | (inner == Color.GREY)), "frame interior must contain only bg and grey markers"
        marker_mask = inner == Color.GREY
        marker_count = int(marker_mask.sum())
        assert 1 <= marker_count <= 4, "marker counts must stay in 1..4"

        grey_cover[r0 + 1 : r1, c0 + 1 : c1] += marker_mask.astype(int)
        parsed.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "side": side_h,
                "count": marker_count,
                "final_side": int(side_h + 2 * marker_count),
            }
        )

    grey_mask = grid == Color.GREY
    assert np.all(grey_cover[grey_mask] == 1), "every grey marker must belong to exactly one frame interior"
    counts = [frame["count"] for frame in parsed]
    assert len(counts) == len(set(counts)), "marker counts must be unique"
    final_sizes = [frame["final_side"] for frame in parsed]
    assert len(final_sizes) == len(set(final_sizes)), "grown outer sizes must be unique"
    return parsed


def _ranked_frames(frames: list[dict]) -> list[dict]:
    ranked = [dict(frame) for frame in frames]
    ranked.sort(key=lambda frame: frame["final_side"])
    for rank, frame in enumerate(ranked, start=1):
        frame["rank"] = rank
    return ranked


def _render_output(grid: np.ndarray, frames: list[dict], rank_to_color: dict[int, int], *, clear_markers: bool = True) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    bg = int(get_bg(out))
    if clear_markers:
        out[out == Color.GREY] = bg

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        k = int(frame["count"])
        color = int(rank_to_color[int(frame["rank"])])
        fr0, fc0, fr1, fc1 = r0 - k, c0 - k, r1 + k, c1 + k
        out[fr0 : fr1 + 1, fc0 : fc1 + 1] = color
        out[r0 + 1 : r1, c0 + 1 : c1] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count grey markers inside each square frame, grow concentric square layers, then recolor by grown-size rank using the top legend.

    Input structure:
    - A black background contains 3-4 separated colored square frames with side lengths 3-5.
    - Each frame has a one-pixel border, a black interior, and 1-4 grey marker pixels inside that interior.
    - The top row contains single-column legend blocks whose colors are distinct from frame colors; each legend block's height encodes a rank.

    Rule:
    - For each frame, count the grey markers inside its interior; call this count k.
    - Grow the frame outward by k concentric one-pixel square shells, so its final outer side length becomes original_side + 2k.
    - Rank frames by final outer side length from smallest to largest, starting at rank 1.
    - Recolor every pixel of each grown frame structure, including the original border and all grown shells, with the legend color whose block height equals that frame's rank.
    - Clear all grey marker pixels to black and leave the legend unchanged.

    Key insight:
    - The legend heights are not geometric sizes; they are rank indices that must be matched after converting marker counts into actual grown outer sizes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    legend = _legend_entries(g, bg)
    rank_to_color = {entry["height"]: entry["color"] for entry in legend}
    frames = _ranked_frames(_frame_objects(g, bg, {entry["color"] for entry in legend}))
    assert len(rank_to_color) == len(frames), "legend entry count must match frame count"
    return _render_output(g, frames, rank_to_color)


_BG = Color.BLACK
_FRAME_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW]
_LEGEND_COLORS = [Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
_GRID_SIZE_RANGE = (22, 30)
_FRAME_COUNT_OPTIONS = (3, 4)
_TOP_CLEARANCE = 6
_EDGE_MARGIN = 4
_MIN_FINAL_GAP = 1
_MIN_VISUAL_GAP = 4
_GEN_ATTEMPTS = 600


def _interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return int(b0 - a1 - 1)
    if b1 < a0:
        return int(a0 - b1 - 1)
    return 0


def _bboxes_visually_clear(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = _MIN_VISUAL_GAP) -> bool:
    row_gap = _interval_gap(a[0], a[2], b[0], b[2])
    col_gap = _interval_gap(a[1], a[3], b[1], b[3])
    row_overlap = row_gap == 0
    col_overlap = col_gap == 0
    if row_overlap:
        return col_gap >= gap
    if col_overlap:
        return row_gap >= gap
    return row_gap >= gap or col_gap >= gap


def _final_boxes_clear(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = _MIN_FINAL_GAP) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _feasible_sides(marker_count: int) -> list[int]:
    return [side for side in range(3, 6) if (side - 2) ** 2 >= marker_count]


def _choose_frame_specs() -> list[dict]:
    frame_count = random.choice(_FRAME_COUNT_OPTIONS)
    counts = random.sample([1, 2, 3, 4], frame_count)
    colors = random.sample(_FRAME_COLORS, frame_count)

    specs = []
    for color, count in zip(colors, counts):
        specs.append({"color": int(color), "count": int(count), "side": random.choice(_feasible_sides(int(count)))})

    if len({spec["side"] for spec in specs}) < 2:
        return []

    for spec in specs:
        spec["final_side"] = int(spec["side"] + 2 * spec["count"])
    if len({spec["final_side"] for spec in specs}) != frame_count:
        return []

    return specs


def _choose_legend(frame_count: int, width: int) -> list[dict] | None:
    colors = random.sample(_LEGEND_COLORS, frame_count)
    heights = list(range(1, frame_count + 1))
    entries = [{"color": int(color), "height": height} for color, height in zip(colors, heights)]
    random.shuffle(entries)
    if [entry["height"] for entry in entries] == heights:
        return None

    gaps = [random.randint(1, 2) for _ in range(frame_count - 1)]
    total_width = frame_count + sum(gaps)
    if total_width > width:
        return None

    start_col = random.randint(0, min(4, width - total_width))
    col = start_col
    for i, entry in enumerate(entries):
        entry["col"] = col
        if i < len(gaps):
            col += 1 + gaps[i]
    return entries


def _choose_grid_shape(frame_count: int) -> tuple[int, int]:
    low, high = _GRID_SIZE_RANGE
    if frame_count == 4:
        return random.randint(26, high), random.randint(26, high)
    return random.randint(low, high), random.randint(low, high)


def _frame_marker_cells(spec: dict) -> list[tuple[int, int]]:
    side = int(spec["side"])
    cells = [(r, c) for r in range(1, side - 1) for c in range(1, side - 1)]
    return random.sample(cells, int(spec["count"]))


def _try_place_frames(height: int, width: int, specs: list[dict]) -> list[dict] | None:
    placed: list[dict] = []
    for spec in sorted(specs, key=lambda item: item["final_side"], reverse=True):
        k = int(spec["count"])
        side = int(spec["side"])
        final_side = int(spec["final_side"])
        candidates = []
        for r in range(_TOP_CLEARANCE, height - final_side + 1):
            for c in range(0, width - final_side + 1):
                orig_r0 = r + k
                orig_c0 = c + k
                orig_r1 = orig_r0 + side - 1
                orig_c1 = orig_c0 + side - 1
                if min(orig_r0, orig_c0, height - 1 - orig_r1, width - 1 - orig_c1) < _EDGE_MARGIN:
                    continue

                final_bbox = (r, c, r + final_side - 1, c + final_side - 1)
                orig_bbox = (orig_r0, orig_c0, orig_r1, orig_c1)
                ok = True
                for other in placed:
                    if not _final_boxes_clear(final_bbox, other["final_bbox"]):
                        ok = False
                        break
                    if not _bboxes_visually_clear(orig_bbox, other["bbox"]):
                        ok = False
                        break
                if ok:
                    candidates.append((final_bbox, orig_bbox))
        if not candidates:
            return None

        final_bbox, orig_bbox = random.choice(candidates)
        placed.append(
            {
                "color": int(spec["color"]),
                "count": k,
                "side": side,
                "final_side": final_side,
                "bbox": orig_bbox,
                "final_bbox": final_bbox,
                "markers": _frame_marker_cells(spec),
            }
        )
    return placed


def _render_input(height: int, width: int, legend: list[dict], frames: list[dict]) -> np.ndarray:
    grid = np.full((height, width), _BG, dtype=int)
    for entry in legend:
        grid[: entry["height"], entry["col"]] = int(entry["color"])

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        color = int(frame["color"])
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        for rr, cc in frame["markers"]:
            grid[r0 + rr, c0 + cc] = Color.GREY
    return grid


def _swap_rank_colors(rank_to_color: dict[int, int]) -> dict[int, int]:
    swapped = dict(rank_to_color)
    keys = sorted(swapped)
    swapped[keys[0]], swapped[keys[1]] = swapped[keys[1]], swapped[keys[0]]
    return swapped


def _validate_instance(grid: np.ndarray, legend: list[dict], frames: list[dict]) -> bool:
    rank_to_color = {entry["height"]: entry["color"] for entry in legend}
    ranked = _ranked_frames(frames)
    expected = _render_output(grid, ranked, rank_to_color)

    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    parsed_legend = _legend_entries(grid, _BG)
    parsed_frames = _ranked_frames(_frame_objects(grid, _BG, {entry["color"] for entry in parsed_legend}))
    if len(parsed_legend) != len(legend) or len(parsed_frames) != len(frames):
        return False
    if sorted(entry["height"] for entry in parsed_legend) != list(range(1, len(frames) + 1)):
        return False

    markers_kept = expected.copy()
    markers_kept[grid == Color.GREY] = Color.GREY
    if np.array_equal(expected, markers_kept):
        return False

    for i, frame in enumerate(ranked):
        altered = [dict(item) for item in ranked]
        step = -1 if frame["count"] > 1 else 1
        altered[i]["count"] = int(frame["count"] + step)
        altered[i]["final_side"] = int(altered[i]["side"] + 2 * altered[i]["count"])
        if len({item["final_side"] for item in altered}) != len(altered):
            continue
        if np.array_equal(expected, _render_output(grid, _ranked_frames(altered), rank_to_color)):
            return False

    if len(rank_to_color) >= 2:
        if np.array_equal(expected, _render_output(grid, ranked, _swap_rank_colors(rank_to_color))):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    specs = _choose_frame_specs()
    if not specs:
        return None

    height, width = _choose_grid_shape(len(specs))
    legend = _choose_legend(len(specs), width)
    if legend is None:
        return None

    frames = _try_place_frames(height, width, specs)
    if frames is None:
        return None

    grid = _render_input(height, width, legend, frames)
    if not _validate_instance(grid, legend, frames):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for opus/11/054/2")
