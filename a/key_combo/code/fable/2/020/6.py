from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.exists() and str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

try:
    from arc_utils import find_components, get_bg, in_bounds
except Exception:  # pragma: no cover - fallback keeps SourceFileLoader usage robust.
    from collections import Counter

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        min_area: int = 1,
        return_format: str = "dict",
        colors=None,
    ):
        assert return_format == "dict"
        assert monochromatic
        if bg is None:
            bg = get_bg(grid)
        h, w = grid.shape
        seen = np.zeros((h, w), dtype=bool)
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        out = []
        color_set = None if colors is None else {int(c) for c in colors}
        for r in range(h):
            for c in range(w):
                if seen[r, c] or int(grid[r, c]) == int(bg):
                    continue
                color = int(grid[r, c])
                if color_set is not None and color not in color_set:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                coords = []
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for dr, dc in dirs:
                        nr, nc = rr + dr, cc + dc
                        if (
                            0 <= nr < h
                            and 0 <= nc < w
                            and not seen[nr, nc]
                            and int(grid[nr, nc]) == color
                        ):
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                if len(coords) < int(min_area):
                    continue
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


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]
Direction = tuple[int, int]

_BG = 0
_PALETTE = tuple(range(1, 10))
_MAX_GEN_ATTEMPTS = 400
_CANVAS = 30


@dataclass(frozen=True)
class Frame:
    idx: int
    color: int
    bbox: BBox
    gap: Coord
    direction: Direction
    area: int


def _border_cells(bbox: BBox) -> set[Coord]:
    r0, c0, r1, c1 = bbox
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _gap_direction(gap: Coord, bbox: BBox) -> Direction:
    r0, c0, r1, c1 = bbox
    gr, gc = gap
    if gr == r0:
        return (-1, 0)
    if gr == r1:
        return (1, 0)
    if gc == c0:
        return (0, -1)
    if gc == c1:
        return (0, 1)
    raise ValueError("gap must lie on the frame boundary")


def _parse_frames(grid: np.ndarray) -> tuple[list[Frame], np.ndarray]:
    comps = find_components(grid, bg=_BG, connectivity=4, monochromatic=True, min_area=5)
    raw_frames: list[Frame] = []
    for comp in comps:
        bbox = tuple(map(int, comp["bbox"]))
        r0, c0, r1, c1 = bbox
        if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
            continue
        coords = {tuple(map(int, cell)) for cell in comp["coords"]}
        border = _border_cells(bbox)
        if coords - border:
            continue
        missing = border - coords
        if len(missing) != 1:
            continue
        gap = next(iter(missing))
        if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            continue
        raw_frames.append(
            Frame(
                idx=-1,
                color=int(comp["color"]),
                bbox=bbox,
                gap=gap,
                direction=_gap_direction(gap, bbox),
                area=(r1 - r0 + 1) * (c1 - c0 + 1),
            )
        )

    raw_frames.sort(key=lambda fr: (fr.area, fr.bbox))
    frames: list[Frame] = []
    wall_mask = np.zeros(grid.shape, dtype=bool)
    for idx, frame in enumerate(raw_frames):
        fixed = Frame(
            idx=idx,
            color=frame.color,
            bbox=frame.bbox,
            gap=frame.gap,
            direction=frame.direction,
            area=frame.area,
        )
        frames.append(fixed)
        for r, c in _border_cells(frame.bbox) - {frame.gap}:
            wall_mask[r, c] = True

    return frames, wall_mask


def _contains_interior(frame: Frame, pos: Coord) -> bool:
    r, c = pos
    r0, c0, r1, c1 = frame.bbox
    return r0 < r < r1 and c0 < c < c1


def _containing_frame_ids(frames: list[Frame], pos: Coord) -> set[int]:
    return {frame.idx for frame in frames if _contains_interior(frame, pos)}


def _origin_frame(frames: list[Frame], pos: Coord) -> Frame | None:
    matches = [frame for frame in frames if _contains_interior(frame, pos)]
    if not matches:
        return None
    return min(matches, key=lambda frame: (frame.area, frame.bbox))


def _sort_key(marker: dict) -> tuple[int, int, int, int, int]:
    dr, dc = marker["direction"]
    r, c = marker["start"]
    if (dr, dc) == (0, 1):
        return (0, -c, r, marker["origin"], marker["color"])
    if (dr, dc) == (0, -1):
        return (1, c, r, marker["origin"], marker["color"])
    if (dr, dc) == (1, 0):
        return (2, -r, c, marker["origin"], marker["color"])
    return (3, r, c, marker["origin"], marker["color"])


def _trace_marker(
    marker: dict,
    frames: list[Frame],
    wall_mask: np.ndarray,
    occupied: set[Coord],
) -> tuple[Coord, dict]:
    r, c = marker["start"]
    dr, dc = marker["direction"]
    origin = int(marker["origin"])
    prev_containing = _containing_frame_ids(frames, (r, c))
    entered_frames: list[int] = []
    left_origin = origin not in prev_containing
    steps = 0

    while True:
        nr, nc = r + dr, c + dc
        if not in_bounds(wall_mask.shape, nr, nc):
            stop_reason = "boundary"
            break
        if wall_mask[nr, nc]:
            stop_reason = "wall"
            break
        if (nr, nc) in occupied:
            stop_reason = "marker"
            break

        r, c = nr, nc
        steps += 1
        now = _containing_frame_ids(frames, (r, c))
        if origin not in now:
            left_origin = True
        for frame_id in sorted(now - prev_containing):
            if frame_id != origin and frame_id not in entered_frames:
                entered_frames.append(frame_id)
        prev_containing = now

    event = {
        "start": tuple(marker["start"]),
        "final": (int(r), int(c)),
        "origin": origin,
        "direction": (int(dr), int(dc)),
        "color": int(marker["color"]),
        "left_origin": bool(left_origin),
        "entered_frames": entered_frames,
        "stop_reason": stop_reason,
        "steps": int(steps),
    }
    return (int(r), int(c)), event


def _simulate(grid: np.ndarray, *, return_events: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames, wall_mask = _parse_frames(g)
    assert frames, "expected at least one gapped frame"

    out = np.full_like(g, _BG)
    out[wall_mask] = g[wall_mask]

    markers = []
    for r, c in np.argwhere((g != _BG) & (~wall_mask)):
        rr, cc = int(r), int(c)
        frame = _origin_frame(frames, (rr, cc))
        assert frame is not None, "every non-frame pixel must belong to an innermost frame"
        markers.append(
            {
                "start": (rr, cc),
                "origin": frame.idx,
                "direction": frame.direction,
                "color": int(g[rr, cc]),
            }
        )

    markers.sort(key=_sort_key)
    occupied: set[Coord] = set()
    events = []
    for marker in markers:
        final_pos, event = _trace_marker(marker, frames, wall_mask, occupied)
        occupied.add(final_pos)
        out[final_pos] = int(marker["color"])
        events.append(event)

    if return_events:
        return out, frames, events
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each frame's interior markers toward its one-cell gap, letting aligned markers leak into later frames or stop at the next obstacle.

    Input structure:
    - The background is black.
    - Several one-pixel-thick hollow rectangles appear; each rectangle is missing exactly one boundary cell on one side.
    - Colored single-pixel markers lie inside the frame interiors, and some gaps face other gaps or outer walls.

    Rule:
    - Detect each frame's gap side and use that side as the slide direction for markers that start in that frame.
    - Markers not aligned with the gap row or column slide until they hit the corresponding inner wall, stacking against earlier settled markers.
    - Markers aligned with the gap pass through it and keep moving in the same direction; they can enter another frame through its matching gap, or else stop when the next step would hit a wall, the grid edge, or a settled marker.

    Key insight:
    - The gap is simultaneously a direction cue and a doorway, so solving the task requires tracing each marker's full path across frame interiors and open background, not just compacting markers locally.
    """
    return _simulate(np.asarray(grid, dtype=int), return_events=False)


def _gap_from_side(bbox: BBox, side: str, offset: int) -> Coord:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0, c0 + offset
    if side == "bottom":
        return r1, c0 + offset
    if side == "left":
        return r0 + offset, c0
    if side == "right":
        return r0 + offset, c1
    raise ValueError(side)


def _frame_spec(bbox: BBox, side: str, offset: int, color: int, markers: list[tuple[int, int, int]]) -> dict:
    return {"bbox": bbox, "gap": _gap_from_side(bbox, side, offset), "color": int(color), "markers": markers}


def _draw_spec(grid: np.ndarray, spec: dict) -> None:
    bbox = tuple(spec["bbox"])
    gap = tuple(spec["gap"])
    color = int(spec["color"])
    for r, c in _border_cells(bbox) - {gap}:
        grid[r, c] = color
    for r, c, marker_color in spec["markers"]:
        grid[int(r), int(c)] = int(marker_color)


def _marker_color(pool: list[int]) -> int:
    return int(random.choice(pool))


def _sample_distinct(values: list[int], count: int) -> list[int]:
    values = values[:]
    random.shuffle(values)
    return sorted(values[:count])


def _build_chain_specs(frame_colors: list[int], marker_pool: list[int]) -> tuple[list[dict], bool]:
    for _ in range(80):
        nested = len(frame_colors) >= 3
        h_a, w_a = random.randint(5, 6), random.randint(5, 6)
        h_b, w_b = random.randint(9, 11), random.randint(9, 11)
        r_a0 = random.randint(2, 5)
        c_a0 = random.randint(1, 3)
        gap_a_offset = random.randint(1, h_a - 2)
        gap_row = r_a0 + gap_a_offset
        c_b0 = c_a0 + w_a + random.randint(3, 5)
        r_b0 = gap_row - random.randint(2, h_b - 3)
        if r_b0 < 1 or r_b0 + h_b > 15 or c_b0 + w_b > 22:
            continue

        specs = []
        bbox_a = (r_a0, c_a0, r_a0 + h_a - 1, c_a0 + w_a - 1)
        bbox_b = (r_b0, c_b0, r_b0 + h_b - 1, c_b0 + w_b - 1)

        gap_cols = list(range(c_a0 + 1, c_a0 + w_a - 1))
        escape_cols = _sample_distinct(gap_cols, min(len(gap_cols), random.randint(2, 3)))
        a_markers = [(gap_row, c, _marker_color(marker_pool)) for c in escape_cols]
        off_rows_a = [r for r in range(r_a0 + 1, r_a0 + h_a - 1) if r != gap_row]
        a_markers.append((random.choice(off_rows_a), random.choice(gap_cols), _marker_color(marker_pool)))
        specs.append(_frame_spec(bbox_a, "right", gap_a_offset, frame_colors[0], a_markers))

        c_rows_for_b = [r for r in range(r_b0 + 1, r_b0 + h_b - 1) if r != gap_row]
        b_marker_rows = _sample_distinct(c_rows_for_b, random.randint(1, min(2, len(c_rows_for_b))))
        b_markers = [(r, c_b0 + 2, _marker_color(marker_pool)) for r in b_marker_rows]
        specs.append(_frame_spec(bbox_b, "left", gap_row - r_b0, frame_colors[1], b_markers))

        if not nested:
            return specs, False

        h_c, w_c = random.randint(5, 6), random.randint(5, 6)
        possible_c0 = [
            c
            for c in range(c_b0 + 3, c_b0 + w_b - w_c - 1)
            if c + w_c - 1 <= c_b0 + w_b - 3
        ]
        possible_r0 = [
            gap_row - offset
            for offset in range(1, h_c - 1)
            if r_b0 + 2 <= gap_row - offset and gap_row - offset + h_c - 1 <= r_b0 + h_b - 3
        ]
        if not possible_c0 or not possible_r0:
            return specs, False

        c_c0 = random.choice(possible_c0)
        r_c0 = random.choice(possible_r0)
        bbox_c = (r_c0, c_c0, r_c0 + h_c - 1, c_c0 + w_c - 1)
        c_rows = [r for r in range(r_c0 + 1, r_c0 + h_c - 1) if r != gap_row]
        c_markers = [(random.choice(c_rows), c_c0 + w_c - 2, _marker_color(marker_pool))]
        specs.append(_frame_spec(bbox_c, "left", gap_row - r_c0, frame_colors[2], c_markers))
        return specs, True

    raise RuntimeError("failed to build chain specs")


def _build_stop_specs(frame_colors: list[int], marker_pool: list[int]) -> tuple[list[dict], str]:
    if len(frame_colors) >= 2:
        for _ in range(80):
            h_d, w_d = random.randint(5, 6), random.randint(5, 6)
            h_e, w_e = random.randint(5, 7), random.randint(5, 7)
            r_d0 = random.randint(18, 21)
            c_d0 = random.randint(3, 6)
            gap_d_offset = random.randint(1, h_d - 2)
            gap_row = r_d0 + gap_d_offset
            c_e0 = c_d0 + w_d + random.randint(3, 5)
            r_e0 = gap_row - random.randint(1, h_e - 2)
            if r_e0 < 16 or r_e0 + h_e > 29 or c_e0 + w_e > 29:
                continue

            bbox_d = (r_d0, c_d0, r_d0 + h_d - 1, c_d0 + w_d - 1)
            bbox_e = (r_e0, c_e0, r_e0 + h_e - 1, c_e0 + w_e - 1)

            gap_cols = list(range(c_d0 + 1, c_d0 + w_d - 1))
            escape_cols = _sample_distinct(gap_cols, min(len(gap_cols), random.randint(2, 3)))
            d_markers = [(gap_row, c, _marker_color(marker_pool)) for c in escape_cols]
            other_rows = [r for r in range(r_d0 + 1, r_d0 + h_d - 1) if r != gap_row]
            d_markers.append((random.choice(other_rows), random.choice(gap_cols), _marker_color(marker_pool)))

            blocker_side = random.choice(["top", "bottom", "right"])
            if blocker_side in {"top", "bottom"}:
                blocker_offset = random.randint(1, w_e - 2)
            else:
                blocker_offset = random.randint(1, h_e - 2)

            impact_rows = [r for r in range(r_e0 + 1, r_e0 + h_e - 1) if r != gap_row]
            e_markers = [(random.choice(impact_rows or [r_e0 + 1]), c_e0 + 2, _marker_color(marker_pool))]
            return [
                _frame_spec(bbox_d, "right", gap_d_offset, frame_colors[0], d_markers),
                _frame_spec(bbox_e, blocker_side, blocker_offset, frame_colors[1], e_markers),
            ], "wall"

    h_d, w_d = random.randint(5, 6), random.randint(5, 6)
    r_d0 = random.randint(18, 22)
    c_d0 = random.randint(18, 21)
    gap_d_offset = random.randint(1, h_d - 2)
    gap_row = r_d0 + gap_d_offset
    bbox_d = (r_d0, c_d0, r_d0 + h_d - 1, c_d0 + w_d - 1)
    gap_cols = list(range(c_d0 + 1, c_d0 + w_d - 1))
    escape_cols = _sample_distinct(gap_cols, min(len(gap_cols), random.randint(2, 3)))
    d_markers = [(gap_row, c, _marker_color(marker_pool)) for c in escape_cols]
    other_rows = [r for r in range(r_d0 + 1, r_d0 + h_d - 1) if r != gap_row]
    d_markers.append((random.choice(other_rows), random.choice(gap_cols), _marker_color(marker_pool)))
    return [_frame_spec(bbox_d, "right", gap_d_offset, frame_colors[0], d_markers)], "boundary"


def _build_stack_spec(frame_color: int, marker_pool: list[int]) -> dict:
    for _ in range(80):
        h_f, w_f = random.randint(5, 7), random.randint(5, 7)
        r_f0 = random.randint(2, 9)
        c_f0 = random.randint(22, 29 - w_f)
        side = random.choice(["top", "bottom", "left", "right"])
        if side in {"top", "bottom"}:
            gap_offset = random.randint(1, w_f - 2)
            lane = c_f0 + random.choice([i for i in range(1, w_f - 1) if i != gap_offset])
            row_choices = list(range(r_f0 + 1, r_f0 + h_f - 1))
            marker_rows = _sample_distinct(row_choices, min(2, len(row_choices)))
            markers = [(r, lane, _marker_color(marker_pool)) for r in marker_rows]
            other_lane_choices = [c for c in range(c_f0 + 1, c_f0 + w_f - 1) if c not in {lane, c_f0 + gap_offset}]
            if other_lane_choices:
                markers.append((random.choice(row_choices), random.choice(other_lane_choices), _marker_color(marker_pool)))
        else:
            gap_offset = random.randint(1, h_f - 2)
            lane = r_f0 + random.choice([i for i in range(1, h_f - 1) if i != gap_offset])
            col_choices = list(range(c_f0 + 1, c_f0 + w_f - 1))
            marker_cols = _sample_distinct(col_choices, min(2, len(col_choices)))
            markers = [(lane, c, _marker_color(marker_pool)) for c in marker_cols]
            other_row_choices = [r for r in range(r_f0 + 1, r_f0 + h_f - 1) if r not in {lane, r_f0 + gap_offset}]
            if other_row_choices:
                markers.append((random.choice(other_row_choices), random.choice(col_choices), _marker_color(marker_pool)))
        return _frame_spec((r_f0, c_f0, r_f0 + h_f - 1, c_f0 + w_f - 1), side, gap_offset, frame_color, markers)
    raise RuntimeError("failed to build stack spec")


def _apply_transform(grid: np.ndarray, op: int) -> np.ndarray:
    out = np.rot90(grid, op % 4)
    if op >= 4:
        out = np.fliplr(out)
    return out.copy()


def _crop_with_margin(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != _BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = random.randint(0, min(3, int(r0)))
    left = random.randint(0, min(3, int(c0)))
    bottom = random.randint(0, min(3, int(grid.shape[0] - 1 - r1)))
    right = random.randint(0, min(3, int(grid.shape[1] - 1 - c1)))
    return grid[int(r0) - top : int(r1) + bottom + 1, int(c0) - left : int(c1) + right + 1].copy()


def _build_candidate() -> tuple[np.ndarray, dict]:
    include_stack = random.random() < 0.75
    chain_nested_planned = random.random() < 0.65
    stop_needs = "wall" if random.random() < 0.55 else "boundary"

    n_chain = 3 if chain_nested_planned else 2
    n_stop = 2 if stop_needs == "wall" else 1
    n_stack = 1 if include_stack else 0
    total_frames = n_chain + n_stop + n_stack
    frame_colors = random.sample(list(_PALETTE), total_frames)
    marker_pool = [color for color in _PALETTE if color not in frame_colors]

    grid = np.zeros((_CANVAS, _CANVAS), dtype=int)
    cursor = 0
    chain_specs, chain_nested_actual = _build_chain_specs(frame_colors[cursor : cursor + n_chain], marker_pool)
    cursor += n_chain
    stop_specs, stop_actual = _build_stop_specs(frame_colors[cursor : cursor + n_stop], marker_pool)
    cursor += n_stop
    stack_specs = []
    if include_stack:
        stack_specs.append(_build_stack_spec(frame_colors[cursor], marker_pool))

    for spec in chain_specs + stop_specs + stack_specs:
        _draw_spec(grid, spec)

    grid = _apply_transform(grid, random.randrange(8))
    grid = _crop_with_margin(grid)
    meta = {
        "want_chain": chain_nested_actual,
        "want_stop": stop_actual,
        "expect_frames": len(chain_specs) + len(stop_specs) + len(stack_specs),
    }
    return grid, meta


def _semantic_checks(grid: np.ndarray, meta: dict) -> bool:
    try:
        out, frames, events = _simulate(grid, return_events=True)
    except AssertionError:
        return False

    if len(frames) != int(meta["expect_frames"]):
        return False
    if np.array_equal(grid, out):
        return False

    moved_inside = any(event["steps"] > 0 and not event["left_origin"] for event in events)
    if not moved_inside:
        return False

    transferred = any(event["left_origin"] and event["entered_frames"] for event in events)
    if not transferred:
        return False

    if meta["want_chain"] and not any(len(event["entered_frames"]) >= 2 for event in events):
        return False

    if meta["want_stop"] == "wall":
        if not any(
            event["left_origin"] and not event["entered_frames"] and event["stop_reason"] == "wall"
            for event in events
        ):
            return False
    else:
        if not any(
            event["left_origin"] and not event["entered_frames"] and event["stop_reason"] == "boundary"
            for event in events
        ):
            return False

    # Preserve the frame geometry and ensure markers remain single settled pixels.
    frame_only = np.zeros_like(grid)
    out_frame_only = np.zeros_like(out)
    for frame in frames:
        for r, c in _border_cells(frame.bbox) - {frame.gap}:
            frame_only[r, c] = frame.color
            out_frame_only[r, c] = out[r, c]
    if not np.array_equal(frame_only, out_frame_only):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid, meta = _build_candidate()
        if max(grid.shape) > 30:
            continue
        if _semantic_checks(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid gap-leak sample")
