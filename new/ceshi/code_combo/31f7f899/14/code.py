# Combo variant of 31f7f899 (index 14)
# combo_spec: D1/0+D4/1+D5/2
# Name: 颜色匹配+框架填洞+旋转放入

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste


# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior and trace
# - Input keeps the source side from the seed task: a full-width baseline near the bottom and
#   several monochrome vertical bars attached to it. Their colors are unique.
# - The destination is replaced by a separate large frame on the right. Inside the frame are
#   several empty horizontal holes of equal thickness. Each hole has a same-colored marker strip
#   just outside the frame wall, so the solver must match source bar color to hole marker color.
# - Output deletes the source bars, preserves the baseline/frame/markers, then fills each hole
#   with the bar of the same color after a 90-degree rotation. A source rectangle of size h x t
#   becomes a horizontal rectangle of size t x h, placed flush against the left edge of its hole.
# - Small trace: suppose source bars are blue/red/green with heights 7/4/6, while the frame's
#   hole markers top-to-bottom are green/blue/red. The seed task would sort heights and map them
#   to slots by order. The fused task instead routes blue to the blue-marked hole, red to the
#   red-marked hole, green to the green-marked hole, and each placed object is horizontal rather
#   than vertical. The three changed stages are therefore: slot discovery by colored markers,
#   destination geometry as enclosed holes, and shape placement after rotation.
#
# b) Complexity calibration
# - Hole lengths are deliberately larger than every rotated bar length, and usually equal across
#   holes. This prevents a shortcut where a solver could match bars to holes by exact geometric
#   fit instead of color.
# - There are 3-5 bars/holes and the color order on the source side is permuted independently from
#   the hole-marker order. The generator rejects weak cases with too few order mismatches.
# - Bar thickness is only 1 or 2, while heights are strictly larger than thickness, so the
#   rotation is visually obvious without introducing too many shape categories.
#
# c) Hidden-assumption audit against the seed solver
# - Seed base-row detection used the globally most common non-bg row/color. A large frame breaks
#   that, so `main()` now looks for the unique non-bg color spanning both left and right edges.
# - Seed slot identity came from source x-positions. That assumption is invalid: slots are now
#   bg holes inside a separate frame, and their identity is read from external marker strips.
# - Seed rendering started from an empty canvas plus baseline. That would erase the frame and the
#   markers, so the new solver starts from the input and erases only components that touch the
#   baseline and are not the baseline/frame colors.
# - Seed placement reused source relative coordinates directly. Here that must be replaced by an
#   explicit rotation: output patch shape is (source_width, source_height), not (source_height,
#   source_width).
# - New combined assumptions: each source color appears exactly twice outside the frame interior
#   (one source bar, one marker strip), every hole has one marker strip immediately left of the
#   frame, and source bar width equals hole thickness. The generator enforces all of these.
#
# d) Placement legality guaranteed by the generator
# - The frame never touches the grid edges or the baseline, and holes are fully enclosed bg
#   components separated by frame-colored rows, so hole extraction is unambiguous.
# - Marker strips sit one column left of the frame, aligned with exactly one hole each, and do
#   not touch the baseline.
# - Source bars live in the left region, are spaced by at least one bg column, and fit above the
#   baseline. Rotated bars fit inside their holes because hole thickness equals source width and
#   hole length is strictly greater than source height.
#
# e) Visual observability
# - Across 2-4 examples, a human should see three cues at once: bars disappear from the baseline
#   area, same-colored holes inside the frame are filled, and the filled shapes are horizontal.
# - The generator rejects cases where the old "sort by height into hole order" rule would produce
#   the same target, so color matching is not incidental.
#
# New implicit assumptions added by `main()`, and how `generate_input()` guarantees them
# - `_find_base_row_and_color()` assumes exactly one edge-spanning non-bg row. Generator keeps the
#   baseline full-width and leaves all other rows with bg at at least one edge.
# - `_find_frame_component()` assumes the largest non-baseline monochrome component is the frame.
#   Generator makes the frame far larger than any source bar or marker strip.
# - `_extract_source_bars()` assumes source bars are precisely the non-bg, non-frame components
#   that touch the baseline row. Generator draws markers away from the baseline.
# - `_extract_holes()` assumes every enclosed bg cavity has a unique non-bg marker strip on the
#   immediate column to the left of the frame. Generator creates exactly one such strip per hole.
# - `_solve(..., match_by_color=True)` assumes hole colors and source bar colors form a bijection.
#   Generator samples unique colors without replacement.


def _find_base_row_and_color(g: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = g.shape
    best = None  # (count, row, color)
    for r in range(h):
        edge_col = int(g[r, 0])
        if edge_col == int(bg) or edge_col != int(g[r, w - 1]):
            continue
        cnt = int(np.count_nonzero(g[r] == edge_col))
        if best is None or cnt > best[0]:
            best = (cnt, int(r), int(edge_col))
    assert best is not None
    return int(best[1]), int(best[2])


def _find_frame_component(g: np.ndarray, bg: int, base_col: int) -> dict[str, object]:
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    frame = max((comp for comp in comps if int(comp["color"]) != int(base_col)), key=lambda comp: int(comp["area"]))
    return frame


def _extract_source_bars(
    g: np.ndarray,
    bg: int,
    base_r: int,
    base_col: int,
    frame_col: int,
) -> list[dict[str, object]]:
    bars: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {int(bg), int(base_col), int(frame_col)}:
            continue
        pts = comp["coords"]
        if not np.any(pts[:, 0] == int(base_r)):
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        bars.append(
            {
                "color": int(col),
                "coords": pts,
                "height": int(r1 - r0 + 1),
                "width": int(c1 - c0 + 1),
                "left": int(c0),
            }
        )
    return sorted(bars, key=lambda bar: (int(bar["left"]), int(bar["color"])))


def _extract_holes(g: np.ndarray, bg: int, base_col: int, frame: dict[str, object]) -> list[dict[str, int]]:
    frame_col = int(frame["color"])
    fr0, fc0, fr1, _fc1 = map(int, frame["bbox"])
    sub = crop_bbox(g, frame["bbox"])
    hole_mask = sub == int(bg)

    holes: list[dict[str, int]] = []
    marker_c = int(fc0 - 1)
    for coords in components_mask(hole_mask, connectivity=4):
        if coords.size == 0:
            continue
        hr0, hc0, hr1, hc1 = map(int, bbox_coords(coords))
        if hr0 == 0 or hc0 == 0 or hr1 == sub.shape[0] - 1 or hc1 == sub.shape[1] - 1:
            continue
        gr0 = int(fr0 + hr0)
        gc0 = int(fc0 + hc0)
        gr1 = int(fr0 + hr1)
        gc1 = int(fc0 + hc1)
        marker_vals = [
            int(v)
            for v in g[gr0 : gr1 + 1, marker_c]
            if int(v) not in {int(bg), int(base_col), int(frame_col)}
        ]
        assert marker_vals and len(set(marker_vals)) == 1
        holes.append(
            {
                "color": int(marker_vals[0]),
                "r0": int(gr0),
                "c0": int(gc0),
                "r1": int(gr1),
                "c1": int(gc1),
                "height": int(gr1 - gr0 + 1),
                "width": int(gc1 - gc0 + 1),
            }
        )
    return sorted(holes, key=lambda hole: (int(hole["r0"]), int(hole["color"])))


def _solve(grid: np.ndarray, *, match_by_color: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_r, base_col = _find_base_row_and_color(g, bg)
    frame = _find_frame_component(g, bg, base_col)
    frame_col = int(frame["color"])

    bars = _extract_source_bars(g, bg, base_r, base_col, frame_col)
    holes = _extract_holes(g, bg, base_col, frame)
    assert bars and holes and len(bars) == len(holes)

    out = g.copy()
    for bar in bars:
        for r, c in bar["coords"]:
            rr = int(r)
            cc = int(c)
            out[rr, cc] = int(base_col) if rr == int(base_r) else int(bg)

    if match_by_color:
        bar_by_color = {int(bar["color"]): bar for bar in bars}
        assignments = [(hole, bar_by_color[int(hole["color"])]) for hole in holes]
    else:
        shapes = sorted(bars, key=lambda bar: (int(bar["height"]), int(bar["width"]), int(bar["left"]), int(bar["color"])))
        assignments = list(zip(holes, shapes))

    for hole, bar in assignments:
        patch_h = int(bar["width"])
        patch_w = int(bar["height"])
        assert patch_h == int(hole["height"])
        assert patch_w <= int(hole["width"])
        patch = np.full((patch_h, patch_w), int(hole["color"]), dtype=int)
        paste(out, patch, int(hole["r0"]), int(hole["c0"]))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read vertical source bars from the baseline region, find same-colored horizontal holes inside
    the separate frame, rotate each bar by 90 degrees, and insert it into the matching hole.
    """
    return _solve(grid, match_by_color=True)


def _sorted_hole_output(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, match_by_color=False)


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_r, base_col = _find_base_row_and_color(g, bg)
    frame = _find_frame_component(g, bg, base_col)
    frame_col = int(frame["color"])
    bars = _extract_source_bars(g, bg, base_r, base_col, frame_col)
    holes = _extract_holes(g, bg, base_col, frame)
    bar_by_color = {int(bar["color"]): bar for bar in bars}

    source_order = [int(bar["color"]) for bar in bars]
    hole_order = [int(hole["color"]) for hole in holes]
    heights_in_hole_order = [int(bar_by_color[color]["height"]) for color in hole_order]
    sorted_heights = sorted(heights_in_hole_order)
    return {
        "k": int(len(bars)),
        "color_displacement": int(sum(a != b for a, b in zip(source_order, hole_order))),
        "sorted_displacement": int(sum(a != b for a, b in zip(heights_in_hole_order, sorted_heights))),
        "rotation_margin": int(min(int(bar["height"]) - int(bar["width"]) for bar in bars)),
    }


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_starts(span: int, k: int, width: int) -> list[int]:
    used = int(k * width + (k - 1))
    extra = int(span - used)
    if extra < 0:
        raise ValueError("items do not fit")
    gaps = _rand_composition(extra, k + 1)
    starts: list[int] = []
    cur = int(gaps[0])
    for i in range(k):
        starts.append(int(cur))
        cur += int(width)
        if i < k - 1:
            cur += 1 + int(gaps[i + 1])
    return starts


def _perm_with_mismatch(k: int, ref: list[int], *, min_mismatch: int) -> list[int]:
    for _ in range(80):
        perm = [int(x) for x in np.random.permutation(ref)]
        mismatch = sum(int(a != b) for a, b in zip(perm, ref))
        if mismatch >= int(min_mismatch):
            return perm
    return [int(x) for x in np.random.permutation(ref)]


def _draw_base_scene(
    n: int,
    bg: int,
    base_r: int,
    base_col: int,
    frame_col: int,
    frame_top: int,
    frame_left: int,
    hole_len: int,
    hole_height: int,
    hole_colors: list[int],
    gap_y: int,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    grid = np.full((int(n), int(n)), int(bg), dtype=int)
    grid[int(base_r), :] = int(base_col)

    inner_h = int(len(hole_colors) * hole_height + (len(hole_colors) - 1) * gap_y)
    frame_h = int(inner_h + 2)
    frame_w = int(hole_len + 2)
    frame_bottom = int(frame_top + frame_h - 1)
    frame_right = int(frame_left + frame_w - 1)
    grid[int(frame_top) : int(frame_bottom) + 1, int(frame_left) : int(frame_right) + 1] = int(frame_col)

    holes: list[dict[str, int]] = []
    cur_r = int(frame_top + 1)
    marker_c = int(frame_left - 1)
    for color in hole_colors:
        r0 = int(cur_r)
        r1 = int(r0 + hole_height - 1)
        c0 = int(frame_left + 1)
        c1 = int(c0 + hole_len - 1)
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        grid[r0 : r1 + 1, int(marker_c)] = int(color)
        holes.append(
            {
                "color": int(color),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "height": int(hole_height),
                "width": int(hole_len),
            }
        )
        cur_r = int(r1 + 1 + gap_y)

    return grid, holes


def _fill_holes(base: np.ndarray, holes: list[dict[str, int]], height_by_color: dict[int, int]) -> np.ndarray:
    out = np.array(base, copy=True)
    for hole in holes:
        color = int(hole["color"])
        height = int(height_by_color[color])
        patch = np.full((int(hole["height"]), height), int(color), dtype=int)
        paste(out, patch, int(hole["r0"]), int(hole["c0"]))
    return out


def _add_source_bars(
    base: np.ndarray,
    base_r: int,
    lefts: list[int],
    bar_colors: list[int],
    height_by_color: dict[int, int],
    bar_width: int,
) -> np.ndarray:
    out = np.array(base, copy=True)
    for left, color in zip(lefts, bar_colors):
        height = int(height_by_color[int(color)])
        patch = np.full((height, int(bar_width)), int(color), dtype=int)
        paste(out, patch, int(base_r - height + 1), int(left))
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = 8
    base_col = 6
    frame_col = 4
    palette = [1, 2, 3, 5, 7, 9]

    bar_width = int(np.random.choice([1, 2]))
    k_choices = [3, 4, 5] if bar_width == 1 else [3, 4]
    k = int(np.random.choice(k_choices))

    height_choices = list(range(bar_width + 2, (9 if bar_width == 1 else 8) + 1))
    if len(height_choices) < k:
        return None
    colors = [int(c) for c in np.random.choice(palette, size=k, replace=False)]
    chosen_heights = [int(h) for h in np.random.choice(height_choices, size=k, replace=False)]
    height_by_color = {int(color): int(height) for color, height in zip(colors, chosen_heights)}

    max_h = max(int(v) for v in height_by_color.values())
    hole_len = int(max_h + np.random.choice([1, 2, 3]))
    gap_y = 1
    frame_h = int(k * bar_width + (k - 1) * gap_y + 2)
    frame_w = int(hole_len + 2)
    source_min_span = int(k * bar_width + (k - 1))
    min_n = max(int(frame_h + 5), int(frame_w + source_min_span + 5), int(max_h + 3))
    candidate_ns = [n for n in range(max(22, min_n), 31)]
    if not candidate_ns:
        return None
    n = int(np.random.choice(candidate_ns))
    base_r = int(n - 2)

    max_frame_top = int(base_r - frame_h - 2)
    if max_frame_top < 1:
        return None
    frame_top = int(np.random.randint(1, max_frame_top + 1))
    frame_left = int(n - frame_w - 1)
    if frame_left < 3:
        return None

    # Keep one full bg column between the rightmost source bar and the marker column at
    # `frame_left - 1`, otherwise a same-colored bar can merge with its marker under 4-connectivity.
    source_span = int(frame_left - 3)
    if source_span < source_min_span:
        return None
    source_lefts = [int(x + 1) for x in _slot_starts(source_span, k, bar_width)]

    hole_order = _perm_with_mismatch(k, colors, min_mismatch=min(2, k))
    source_order = _perm_with_mismatch(k, hole_order, min_mismatch=min(2, k))
    if source_order == hole_order:
        return None

    base_scene, holes = _draw_base_scene(
        n=n,
        bg=bg,
        base_r=base_r,
        base_col=base_col,
        frame_col=frame_col,
        frame_top=frame_top,
        frame_left=frame_left,
        hole_len=hole_len,
        hole_height=bar_width,
        hole_colors=hole_order,
        gap_y=gap_y,
    )
    out = _fill_holes(base_scene, holes, height_by_color)
    inp = _add_source_bars(base_scene, base_r, source_lefts, source_order, height_by_color, bar_width)

    metrics = _behavior_metrics(inp)
    if metrics["color_displacement"] < min(2, k):
        return None
    if metrics["sorted_displacement"] < min(2, k):
        return None
    if metrics["rotation_margin"] <= 0:
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_sorted_hole_output(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 600 attempts")
