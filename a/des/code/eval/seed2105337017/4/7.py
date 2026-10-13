from __future__ import annotations

from functools import lru_cache
import random

import numpy as np


BG = 0
GREY = 5
PIPE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _other_side(orient: str, side: str) -> str:
    if orient == "h":
        return "right" if side == "left" else "left"
    return "bottom" if side == "top" else "top"


def _segment_cells(seg: dict) -> list[tuple[int, int]]:
    if seg["orient"] == "h":
        r = int(seg["fixed"])
        return [(r, c) for c in range(int(seg["lo"]), int(seg["hi"]) + 1)]
    c = int(seg["fixed"])
    return [(r, c) for r in range(int(seg["lo"]), int(seg["hi"]) + 1)]


def _endpoint_cell(seg: dict, side: str) -> tuple[int, int]:
    if seg["orient"] == "h":
        r = int(seg["fixed"])
        if side == "left":
            return r, int(seg["lo"])
        return r, int(seg["hi"])
    c = int(seg["fixed"])
    if side == "top":
        return int(seg["lo"]), c
    return int(seg["hi"]), c


def _marker_cell(seg: dict, side: str) -> tuple[int, int]:
    if seg["orient"] == "h":
        r = int(seg["fixed"])
        if side == "left":
            return r, int(seg["lo"]) - 1
        return r, int(seg["hi"]) + 1
    c = int(seg["fixed"])
    if side == "top":
        return int(seg["lo"]) - 1, c
    return int(seg["hi"]) + 1, c


def _pipe_mask(grid: np.ndarray) -> np.ndarray:
    return (grid != BG) & (grid != GREY)


def _scan_segments(grid: np.ndarray) -> tuple[list[dict], np.ndarray, np.ndarray]:
    h, w = grid.shape
    pipe = _pipe_mask(grid)
    segments: list[dict] = []
    h_owner = np.full((h, w), -1, dtype=int)
    v_owner = np.full((h, w), -1, dtype=int)

    for r in range(h):
        c = 0
        while c < w:
            if not pipe[r, c]:
                c += 1
                continue
            c0 = c
            while c < w and pipe[r, c]:
                c += 1
            c1 = c - 1
            if c1 - c0 + 1 < 3:
                continue
            seg_id = len(segments)
            h_owner[r, c0 : c1 + 1] = seg_id
            segments.append(
                {
                    "id": seg_id,
                    "orient": "h",
                    "fixed": int(r),
                    "lo": int(c0),
                    "hi": int(c1),
                    "crossings": [],
                }
            )

    for c in range(w):
        r = 0
        while r < h:
            if not pipe[r, c]:
                r += 1
                continue
            r0 = r
            while r < h and pipe[r, c]:
                r += 1
            r1 = r - 1
            if r1 - r0 + 1 < 3:
                continue
            seg_id = len(segments)
            v_owner[r0 : r1 + 1, c] = seg_id
            segments.append(
                {
                    "id": seg_id,
                    "orient": "v",
                    "fixed": int(c),
                    "lo": int(r0),
                    "hi": int(r1),
                    "crossings": [],
                }
            )

    assert segments, "expected at least one segment"
    assert np.all((~pipe) | (h_owner != -1) | (v_owner != -1)), "every pipe cell must belong to a segment"
    return segments, h_owner, v_owner


def _segment_markers(grid: np.ndarray, seg: dict) -> dict[str, tuple[int, int]]:
    h, w = grid.shape
    markers: dict[str, tuple[int, int]] = {}
    sides = ("left", "right") if seg["orient"] == "h" else ("top", "bottom")
    for side in sides:
        r, c = _marker_cell(seg, side)
        if 0 <= r < h and 0 <= c < w and int(grid[r, c]) == GREY:
            markers[side] = (int(r), int(c))
    return markers


def _analyze_grid(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    segments, h_owner, v_owner = _scan_segments(g)
    cross_mask = (h_owner != -1) & (v_owner != -1)

    grey_cells = set(map(tuple, np.argwhere(g == GREY)))
    used_markers: set[tuple[int, int]] = set()

    for seg in segments:
        visible = [g[r, c] for r, c in _segment_cells(seg) if not cross_mask[r, c]]
        assert visible, "each segment needs at least one non-crossing cell"
        original = int(visible[0])
        assert original not in (BG, GREY)
        assert all(int(v) == original for v in visible), "non-crossing cells must expose one original color"
        seg["original_color"] = original

        markers = _segment_markers(g, seg)
        assert markers, "each segment needs at least one grey exit marker"
        seg["markers"] = markers
        used_markers.update(markers.values())

        if len(markers) == 1:
            downstream_side = next(iter(markers))
        else:
            downstream_side = "right" if seg["orient"] == "h" else "bottom"
        seg["downstream_side"] = downstream_side
        seg["upstream_side"] = _other_side(seg["orient"], downstream_side)
        seg["direction"] = 1 if downstream_side in {"right", "bottom"} else -1

    assert grey_cells == used_markers, "grey cells must be exactly the exit markers"

    crossings: list[dict] = []
    for r, c in np.argwhere(cross_mask):
        h_id = int(h_owner[r, c])
        v_id = int(v_owner[r, c])
        h_seg = segments[h_id]
        v_seg = segments[v_id]
        assert h_seg["orient"] == "h" and v_seg["orient"] == "v"

        cell = (int(r), int(c))
        assert cell not in {
            _endpoint_cell(h_seg, "left"),
            _endpoint_cell(h_seg, "right"),
            _endpoint_cell(v_seg, "top"),
            _endpoint_cell(v_seg, "bottom"),
        }, "crossings cannot sit on endpoints"

        h_color = int(h_seg["original_color"])
        v_color = int(v_seg["original_color"])
        pix = int(g[r, c])
        if h_color == v_color:
            upper = None
        elif pix == h_color:
            upper = h_id
        else:
            assert pix == v_color, "crossing pixel must show one of the two pipes"
            upper = v_id

        cross_id = len(crossings)
        crossings.append(
            {
                "id": cross_id,
                "row": int(r),
                "col": int(c),
                "h": h_id,
                "v": v_id,
                "upper": upper,
                "index_by_seg": {},
            }
        )
        h_seg["crossings"].append(cross_id)
        v_seg["crossings"].append(cross_id)

    for seg in segments:
        key = "col" if seg["orient"] == "h" else "row"
        seg["crossings"] = tuple(
            sorted(seg["crossings"], key=lambda cross_id: int(crossings[cross_id][key]), reverse=seg["direction"] < 0)
        )
        for idx, cross_id in enumerate(seg["crossings"]):
            crossings[cross_id]["index_by_seg"][seg["id"]] = int(idx)

    return segments, crossings


def _downstream_colors(segments: list[dict], crossings: list[dict]) -> dict[int, int]:
    @lru_cache(maxsize=None)
    def color_after(seg_id: int, cross_idx: int) -> int:
        if cross_idx < 0:
            return int(segments[seg_id]["original_color"])

        cross_id = int(segments[seg_id]["crossings"][cross_idx])
        cross = crossings[cross_id]
        upper = cross["upper"]
        if upper is None or upper == seg_id:
            return color_after(seg_id, cross_idx - 1)

        upper_idx = int(cross["index_by_seg"][upper])
        return color_after(int(upper), upper_idx - 1)

    return {
        int(seg["id"]): color_after(int(seg["id"]), len(seg["crossings"]) - 1)
        for seg in segments
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Track color contamination through line-segment crossings and repaint the grey exit markers.

    Input structure:
    - Black background.
    - Horizontal and vertical 1-pixel-wide colored segments.
    - A crossing pixel shows the color of the pipe on top.
    - Each segment has at least one grey pixel directly outside an endpoint.

    Rule:
    - A segment initially carries its own color.
    - Each segment is traced toward its grey-marked exit.
    - At a crossing, the lower pipe adopts the upper pipe's current carried color for the rest of its downstream part.
    - Pipe pixels stay visually unchanged; only grey exit markers are repainted with the final downstream color.
    """
    g = np.asarray(grid, dtype=int)
    segments, crossings = _analyze_grid(g)
    final_colors = _downstream_colors(segments, crossings)

    out = g.copy()
    for seg in segments:
        seg_id = int(seg["id"])
        downstream_side = str(seg["downstream_side"])
        downstream_color = int(final_colors[seg_id])
        for side, marker in seg["markers"].items():
            rr, cc = marker
            out[rr, cc] = downstream_color if side == downstream_side else int(seg["original_color"])
    return out


# --------------- generator ---------------

GRID_MIN = 14
GRID_MAX = 22
SEGMENTS_PER_AXIS = (2, 4)
PIPE_LEN_MIN = 4
PIPE_LEN_MAX = 10
POSITION_GAP = 2
MAX_TOTAL_CROSSINGS = 6
GENERATION_ATTEMPTS = 500
SPACING_ATTEMPTS = 80
SIDE_MIX_PROB = 0.5


def _sample_spaced_positions(lo: int, hi: int, count: int, gap: int) -> list[int] | None:
    values = list(range(lo, hi + 1))
    if count <= 0:
        return []
    for _ in range(SPACING_ATTEMPTS):
        random.shuffle(values)
        chosen: list[int] = []
        for value in values:
            if all(abs(value - other) >= gap for other in chosen):
                chosen.append(int(value))
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _random_interval(limit: int) -> tuple[int, int]:
    length = random.randint(PIPE_LEN_MIN, min(PIPE_LEN_MAX, limit - 2))
    start = random.randint(1, limit - length - 1)
    return int(start), int(start + length - 1)


def _geometric_crossings(segments: list[dict]) -> list[dict]:
    horizontals = [seg for seg in segments if seg["orient"] == "h"]
    verticals = [seg for seg in segments if seg["orient"] == "v"]
    out: list[dict] = []
    for h_seg in horizontals:
        row = int(h_seg["fixed"])
        for v_seg in verticals:
            col = int(v_seg["fixed"])
            if int(h_seg["lo"]) <= col <= int(h_seg["hi"]) and int(v_seg["lo"]) <= row <= int(v_seg["hi"]):
                out.append(
                    {
                        "row": row,
                        "col": col,
                        "h": int(h_seg["id"]),
                        "v": int(v_seg["id"]),
                    }
                )
    return out


def _layout_is_clear(height: int, width: int, segments: list[dict], crossings: list[dict]) -> bool:
    owner_map: dict[tuple[int, int], set[int]] = {}
    marker_map: dict[tuple[int, int], int] = {}
    crossing_cells = {(int(cross["row"]), int(cross["col"])) for cross in crossings}

    for seg in segments:
        for cell in _segment_cells(seg):
            ids = owner_map.setdefault(cell, set())
            ids.add(int(seg["id"]))
            if len(ids) > 2:
                return False
            if len(ids) == 2:
                a, b = sorted(ids)
                if segments[a]["orient"] == segments[b]["orient"]:
                    return False
            if cell in {
                _endpoint_cell(seg, "left") if seg["orient"] == "h" else _endpoint_cell(seg, "top"),
                _endpoint_cell(seg, "right") if seg["orient"] == "h" else _endpoint_cell(seg, "bottom"),
            } and cell in crossing_cells:
                return False

        marker = _marker_cell(seg, str(seg["marker_side"]))
        mr, mc = marker
        if not (0 <= mr < height and 0 <= mc < width):
            return False
        if marker in owner_map or marker in marker_map:
            return False
        marker_map[marker] = int(seg["id"])

    occupied = set(owner_map) | set(marker_map)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r, c in occupied:
        for dr, dc in deltas:
            rr, cc = r + dr, c + dc
            if (rr, cc) not in occupied:
                continue

            a_seg = owner_map.get((r, c))
            b_seg = owner_map.get((rr, cc))
            a_marker = marker_map.get((r, c))
            b_marker = marker_map.get((rr, cc))

            if a_seg is not None and b_seg is not None:
                if not (a_seg & b_seg):
                    return False
                continue

            if a_marker is not None and b_marker is not None:
                return False

            if a_marker is not None and b_seg is not None:
                side = str(segments[a_marker]["marker_side"])
                if _endpoint_cell(segments[a_marker], side) != (rr, cc) or a_marker not in b_seg:
                    return False
                continue

            if b_marker is not None and a_seg is not None:
                side = str(segments[b_marker]["marker_side"])
                if _endpoint_cell(segments[b_marker], side) != (r, c) or b_marker not in a_seg:
                    return False
                continue

    return True


def _assign_colors(segments: list[dict], crossings: list[dict]) -> dict[int, int] | None:
    neighbors = {int(seg["id"]): set() for seg in segments}
    for cross in crossings:
        h_id = int(cross["h"])
        v_id = int(cross["v"])
        neighbors[h_id].add(v_id)
        neighbors[v_id].add(h_id)

    order = sorted(neighbors, key=lambda seg_id: (len(neighbors[seg_id]), random.random()), reverse=True)
    colors: dict[int, int] = {}
    for seg_id in order:
        blocked = {colors[nbr] for nbr in neighbors[seg_id] if nbr in colors}
        choices = [color for color in PIPE_COLORS if color not in blocked]
        if not choices:
            return None
        colors[seg_id] = int(random.choice(choices))

    if len(set(colors.values())) < 3:
        return None
    return colors


def _paint_candidate(height: int, width: int, segments: list[dict], crossings: list[dict], colors: dict[int, int]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for seg in segments:
        color = int(colors[int(seg["id"])])
        for r, c in _segment_cells(seg):
            grid[r, c] = color

    for cross in crossings:
        upper = int(cross["upper"])
        grid[int(cross["row"]), int(cross["col"])] = int(colors[upper])

    for seg in segments:
        mr, mc = _marker_cell(seg, str(seg["marker_side"]))
        grid[mr, mc] = GREY
    return grid


def _has_bridge_segment(segments: list[dict], crossings: list[dict]) -> bool:
    for seg in segments:
        seen_lower = False
        for cross_id in seg["crossings"]:
            cross = crossings[int(cross_id)]
            upper = cross["upper"]
            if upper is None:
                continue
            if int(upper) == int(seg["id"]):
                if seen_lower:
                    return True
            else:
                seen_lower = True
    return False


def _input_mode_ok(mode: str, segments: list[dict], crossings: list[dict], final_colors: dict[int, int]) -> bool:
    if not crossings:
        return False

    max_crossings = max(len(seg["crossings"]) for seg in segments)
    contaminated = any(int(final_colors[int(seg["id"])]) != int(seg["original_color"]) for seg in segments)
    if not contaminated:
        return False

    if mode == "basic":
        return True
    if mode == "multi":
        return max_crossings >= 2
    return max_crossings >= 2 and _has_bridge_segment(segments, crossings)


def _generate_attempt() -> np.ndarray | None:
    mode = random.choice(("basic", "multi", "chain"))
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)

    n_h = random.randint(*SEGMENTS_PER_AXIS)
    n_v = random.randint(*SEGMENTS_PER_AXIS)
    rows = _sample_spaced_positions(2, height - 3, n_h, POSITION_GAP)
    cols = _sample_spaced_positions(2, width - 3, n_v, POSITION_GAP)
    if rows is None or cols is None:
        return None

    segments: list[dict] = []
    for row in rows:
        lo, hi = _random_interval(width)
        segments.append(
            {
                "id": len(segments),
                "orient": "h",
                "fixed": int(row),
                "lo": int(lo),
                "hi": int(hi),
                "marker_side": random.choice(("left", "right")),
            }
        )
    for col in cols:
        lo, hi = _random_interval(height)
        segments.append(
            {
                "id": len(segments),
                "orient": "v",
                "fixed": int(col),
                "lo": int(lo),
                "hi": int(hi),
                "marker_side": random.choice(("top", "bottom")),
            }
        )

    crossings = _geometric_crossings(segments)
    if not (2 <= len(crossings) <= MAX_TOTAL_CROSSINGS):
        return None

    cross_counts = {int(seg["id"]): 0 for seg in segments}
    for cross in crossings:
        cross_counts[int(cross["h"])] += 1
        cross_counts[int(cross["v"])] += 1
    if any((int(seg["hi"]) - int(seg["lo"]) + 1) - cross_counts[int(seg["id"])] < 2 for seg in segments):
        return None

    if not _layout_is_clear(height, width, segments, crossings):
        return None

    require_side_mix = len(crossings) >= 3 and random.random() < SIDE_MIX_PROB
    upper_sides: set[str] = set()
    for cross in crossings:
        if random.random() < 0.5:
            cross["upper"] = int(cross["h"])
            upper_sides.add("h")
        else:
            cross["upper"] = int(cross["v"])
            upper_sides.add("v")
    if require_side_mix and len(upper_sides) < 2:
        return None

    colors = _assign_colors(segments, crossings)
    if colors is None:
        return None

    grid = _paint_candidate(height, width, segments, crossings, colors)
    try:
        parsed_segments, parsed_crossings = _analyze_grid(grid)
        final_colors = _downstream_colors(parsed_segments, parsed_crossings)
        if not _input_mode_ok(mode, parsed_segments, parsed_crossings, final_colors):
            return None
        if np.array_equal(main(grid), grid):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
