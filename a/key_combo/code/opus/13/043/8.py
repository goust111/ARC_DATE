from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BACKGROUND = 0
FRAME = 5
TARGET = 7
CW_COLORS = {1, 2, 3}
CCW_COLORS = {4, 6}
MARKER_COLORS = CW_COLORS | CCW_COLORS

DIR_STEP = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
TURN_DIR = {
    "cw": {
        "down": "left",
        "right": "down",
        "up": "right",
        "left": "up",
    },
    "ccw": {
        "down": "right",
        "right": "up",
        "up": "left",
        "left": "down",
    },
}


@dataclass(frozen=True)
class FrameInfo:
    r0: int
    c0: int
    r1: int
    c1: int
    target_r: int
    target_c: int


@dataclass(frozen=True)
class MarkerTrace:
    marker_r: int
    marker_c: int
    color: int
    side: str
    initial_dir: str
    turned_dir: str
    entry_r: int
    entry_c: int
    path: tuple[tuple[int, int], ...]
    stop_reason: str
    frame_index: int


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_frames(grid: np.ndarray) -> list[FrameInfo]:
    frames: list[FrameInfo] = []
    for coords in _components(grid == FRAME):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 10 or w < 10:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.array_equal(patch == FRAME, _border_mask(h, w)):
            continue
        interior = patch[1:-1, 1:-1]
        target_coords = np.argwhere(interior == TARGET)
        assert target_coords.size > 0, "each frame interior must contain a target"
        tr0, tc0 = target_coords.min(axis=0)
        tr1, tc1 = target_coords.max(axis=0)
        target_h = int(tr1 - tr0 + 1)
        target_w = int(tc1 - tc0 + 1)
        assert target_h == 2 and target_w == 2, "target must be a 2x2 block"
        assert np.all(interior[tr0 : tr1 + 1, tc0 : tc1 + 1] == TARGET), "target must be solid"
        assert int(np.count_nonzero(interior == TARGET)) == 4, "exactly one 2x2 target per frame"
        frames.append(
            FrameInfo(
                int(r0),
                int(c0),
                int(r1),
                int(c1),
                int(r0 + 1 + tr0),
                int(c0 + 1 + tc0),
            )
        )
    assert 1 <= len(frames) <= 2, "expected one or two grey frames"
    frames.sort(key=lambda fr: (fr.r0, fr.c0))
    return frames


def _marker_group(color: int) -> str:
    color = int(color)
    if color in CW_COLORS:
        return "cw"
    if color in CCW_COLORS:
        return "ccw"
    raise AssertionError("unexpected marker color")


def _marker_side(frame: FrameInfo, r: int, c: int) -> str | None:
    if r == frame.r0 - 1 and frame.c0 + 1 <= c <= frame.c1 - 1:
        return "top"
    if r == frame.r1 + 1 and frame.c0 + 1 <= c <= frame.c1 - 1:
        return "bottom"
    if c == frame.c0 - 1 and frame.r0 + 1 <= r <= frame.r1 - 1:
        return "left"
    if c == frame.c1 + 1 and frame.r0 + 1 <= r <= frame.r1 - 1:
        return "right"
    return None


def _entry_from_side(frame: FrameInfo, side: str, r: int, c: int) -> tuple[str, tuple[int, int]]:
    if side == "top":
        return "down", (frame.r0 + 1, c)
    if side == "bottom":
        return "up", (frame.r1 - 1, c)
    if side == "left":
        return "right", (r, frame.c0 + 1)
    assert side == "right"
    return "left", (r, frame.c1 - 1)


def _trace_marker(
    grid: np.ndarray,
    frame: FrameInfo,
    frame_index: int,
    r: int,
    c: int,
    color_override: int | None = None,
) -> MarkerTrace:
    side = _marker_side(frame, r, c)
    assert side is not None, "marker must touch exactly one non-corner frame wall"
    color = int(grid[r, c]) if color_override is None else int(color_override)
    group = _marker_group(color)
    initial_dir, (er, ec) = _entry_from_side(frame, side, r, c)
    turned_dir = TURN_DIR[group][initial_dir]
    assert grid[er, ec] == BACKGROUND, "entry cell must be empty background"

    dr, dc = DIR_STEP[turned_dir]
    path: list[tuple[int, int]] = []
    cr, cc = er, ec
    while True:
        assert frame.r0 < cr < frame.r1 and frame.c0 < cc < frame.c1
        assert grid[cr, cc] == BACKGROUND, "projection path must start on clean background"
        path.append((int(cr), int(cc)))
        nr, nc = cr + dr, cc + dc
        if not (frame.r0 < nr < frame.r1 and frame.c0 < nc < frame.c1):
            stop = "wall"
            break
        if grid[nr, nc] == TARGET:
            stop = "target"
            break
        assert grid[nr, nc] == BACKGROUND, "paths may not cross other objects in the input"
        cr, cc = nr, nc
    assert path, "projection path must contain at least one cell"
    return MarkerTrace(
        int(r),
        int(c),
        color,
        side,
        initial_dir,
        turned_dir,
        int(er),
        int(ec),
        tuple(path),
        stop,
        int(frame_index),
    )


def _collect_traces(grid: np.ndarray, *, enforce_marker_counts: bool = True) -> tuple[list[FrameInfo], list[MarkerTrace]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    allowed = {BACKGROUND, FRAME, TARGET, *MARKER_COLORS}
    assert set(map(int, np.unique(g))) <= allowed, "unexpected colors in input"

    frames = _find_frames(g)
    traces: list[MarkerTrace] = []
    for r, c in sorted(map(tuple, np.argwhere(np.isin(g, list(MARKER_COLORS))))):
        owners: list[tuple[int, FrameInfo]] = []
        for idx, frame in enumerate(frames):
            if _marker_side(frame, int(r), int(c)) is not None:
                owners.append((idx, frame))
        assert len(owners) == 1, "each marker must belong to exactly one frame"
        idx, frame = owners[0]
        traces.append(_trace_marker(g, frame, idx, int(r), int(c)))

    assert traces, "at least one marker is required"
    if enforce_marker_counts:
        for frame_index in range(len(frames)):
            count = sum(trace.frame_index == frame_index for trace in traces)
            assert 2 <= count <= 4, "each frame must have 2-4 markers"

    occupied: set[tuple[int, int]] = set()
    for trace in traces:
        path_set = set(trace.path)
        assert occupied.isdisjoint(path_set), "projection paths must not intersect"
        occupied |= path_set
    return frames, traces


def main(grid: np.ndarray) -> np.ndarray:
    """Project exterior markers through a frame wall, turn inside the frame, and stop before the target block or opposite wall.

    Input structure:
    - Black background contains one or two thin grey rectangular frames with interiors at least 8x8.
    - Each frame contains exactly one orange 2x2 target block and has 2-4 colored marker pixels outside its walls.
    - Markers touch one non-corner frame wall cell orthogonally; marker colors come from {1,2,3,4,6}.

    Rule:
    - For each marker in row-major order, find the touched wall and enter at the first interior cell just inside that wall.
    - Immediately turn 90 degrees: colors {1,2,3} use the clockwise mapping and colors {4,6} use the counterclockwise mapping.
    - Draw the marker-colored line along the turned direction until the next step would enter the orange target or leave the interior at the opposite wall; keep the frame and target unchanged, and erase the source marker pixel.

    Key insight:
    - The frame wall determines both the entry point and the initial heading, while the marker color alone determines which way the path turns inside the frame.
    """
    g = np.asarray(grid, dtype=int)
    _, traces = _collect_traces(g, enforce_marker_counts=False)

    out = g.copy()
    for trace in traces:
        out[trace.marker_r, trace.marker_c] = BACKGROUND
        for pr, pc in trace.path:
            assert out[pr, pc] == BACKGROUND, "paths may not overlap in output"
            out[pr, pc] = trace.color
    return out


def _local_target_top_left(length: int, anchor: str) -> int:
    if anchor == "near_start":
        return 1
    if anchor == "near_end":
        return length - 3
    return random.randint(2, length - 4)


def _empty_local_frame(h: int, w: int, tr: int, tc: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[0, :] = FRAME
    grid[-1, :] = FRAME
    grid[:, 0] = FRAME
    grid[:, -1] = FRAME
    grid[tr : tr + 2, tc : tc + 2] = TARGET
    return grid


def _candidate_trace(h: int, w: int, tr: int, tc: int, side: str, pos: int, color: int) -> MarkerTrace | None:
    grid = _empty_local_frame(h, w, tr, tc)
    if side == "top":
        mr, mc = -1, pos
    elif side == "bottom":
        mr, mc = h, pos
    elif side == "left":
        mr, mc = pos, -1
    else:
        mr, mc = pos, w
    frame = FrameInfo(0, 0, h - 1, w - 1, tr, tc)
    try:
        trace = _trace_marker(grid, frame, 0, mr, mc, color_override=color)
    except AssertionError:
        return None
    if len(trace.path) < 2:
        return None
    return trace


def _frame_candidates(h: int, w: int, tr: int, tc: int) -> list[MarkerTrace]:
    candidates: list[MarkerTrace] = []
    for color in sorted(MARKER_COLORS):
        for c in range(1, w - 1):
            top = _candidate_trace(h, w, tr, tc, "top", c, color)
            bottom = _candidate_trace(h, w, tr, tc, "bottom", c, color)
            if top is not None:
                candidates.append(top)
            if bottom is not None:
                candidates.append(bottom)
        for r in range(1, h - 1):
            left = _candidate_trace(h, w, tr, tc, "left", r, color)
            right = _candidate_trace(h, w, tr, tc, "right", r, color)
            if left is not None:
                candidates.append(left)
            if right is not None:
                candidates.append(right)
    return candidates


def _search_marker_subset(
    candidates: list[MarkerTrace],
    desired_count: int,
    require_target: bool,
    require_wall: bool,
    require_cw: bool,
    require_ccw: bool,
    min_distinct_sides: int,
) -> list[MarkerTrace] | None:
    random.shuffle(candidates)
    best: list[MarkerTrace] | None = None

    def backtrack(
        index: int,
        chosen: list[MarkerTrace],
        used_cells: set[tuple[int, int]],
        hit_target: bool,
        hit_wall: bool,
        has_cw: bool,
        has_ccw: bool,
        sides: set[str],
    ) -> None:
        nonlocal best
        if best is not None:
            return
        if len(chosen) == desired_count:
            if require_target and not hit_target:
                return
            if require_wall and not hit_wall:
                return
            if require_cw and not has_cw:
                return
            if require_ccw and not has_ccw:
                return
            if len(sides) < min_distinct_sides:
                return
            best = chosen.copy()
            return
        if index >= len(candidates):
            return
        remaining_slots = desired_count - len(chosen)
        if len(candidates) - index < remaining_slots:
            return

        cand = candidates[index]
        path_set = set(cand.path)
        if used_cells.isdisjoint(path_set):
            chosen.append(cand)
            backtrack(
                index + 1,
                chosen,
                used_cells | path_set,
                hit_target or cand.stop_reason == "target",
                hit_wall or cand.stop_reason == "wall",
                has_cw or cand.color in CW_COLORS,
                has_ccw or cand.color in CCW_COLORS,
                sides | {cand.side},
            )
            chosen.pop()
        backtrack(index + 1, chosen, used_cells, hit_target, hit_wall, has_cw, has_ccw, sides)

    backtrack(0, [], set(), False, False, False, False, set())
    return best


def _make_frame_plan(require_target: bool, require_wall: bool, require_cw: bool, require_ccw: bool) -> tuple[int, int, int, int, list[MarkerTrace]]:
    for _ in range(400):
        h = random.randint(10, 14)
        w = random.randint(10, 14)
        row_anchor = random.choices(["near_start", "middle", "near_end"], weights=[3, 2, 3])[0]
        col_anchor = random.choices(["near_start", "middle", "near_end"], weights=[3, 2, 3])[0]
        if row_anchor == "middle" and col_anchor == "middle":
            continue
        tr = _local_target_top_left(h, row_anchor)
        tc = _local_target_top_left(w, col_anchor)
        candidates = _frame_candidates(h, w, tr, tc)
        if not candidates:
            continue

        desired_options = [random.randint(2, 4)]
        for extra in (2, 3, 4):
            if extra not in desired_options:
                desired_options.append(extra)
        for desired in desired_options:
            subset = _search_marker_subset(
                candidates,
                desired_count=desired,
                require_target=require_target,
                require_wall=require_wall,
                require_cw=require_cw,
                require_ccw=require_ccw,
                min_distinct_sides=2,
            )
            if subset is not None:
                return h, w, tr, tc, subset
    raise RuntimeError("failed to build a valid frame plan")


def _expanded_box(r0: int, c0: int, h: int, w: int) -> tuple[int, int, int, int]:
    return r0 - 1, c0 - 1, r0 + h, c0 + w


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_frames(plans: list[tuple[int, int, int, int, list[MarkerTrace]]]) -> tuple[int, int, list[tuple[int, int]]]:
    if len(plans) == 1:
        h_min = max(15, plans[0][0] + 4)
        w_min = max(15, plans[0][1] + 4)
        H = random.randint(h_min, min(22, max(h_min, 22)))
        W = random.randint(w_min, min(22, max(w_min, 22)))
    else:
        h_need = max(plan[0] for plan in plans) + 4
        w_need = max(plan[1] for plan in plans) + 4
        H = random.randint(max(22, h_need), 30)
        W = random.randint(max(22, w_need), 30)

    placed: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []
    for h, w, _tr, _tc, _markers in plans:
        success = False
        for _ in range(300):
            r0 = random.randint(1, H - h - 1)
            c0 = random.randint(1, W - w - 1)
            box = _expanded_box(r0, c0, h, w)
            if any(not _boxes_separated(box, prev, gap=1) for prev in boxes):
                continue
            placed.append((r0, c0))
            boxes.append(box)
            success = True
            break
        if not success:
            raise RuntimeError("failed to place frames")
    return H, W, placed


def _render_input(plans: list[tuple[int, int, int, int, list[MarkerTrace]]], H: int, W: int, origins: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    for (h, w, tr, tc, traces), (r0, c0) in zip(plans, origins, strict=True):
        grid[r0, c0 : c0 + w] = FRAME
        grid[r0 + h - 1, c0 : c0 + w] = FRAME
        grid[r0 : r0 + h, c0] = FRAME
        grid[r0 : r0 + h, c0 + w - 1] = FRAME
        grid[r0 + tr : r0 + tr + 2, c0 + tc : c0 + tc + 2] = TARGET
        for trace in traces:
            grid[r0 + trace.marker_r, c0 + trace.marker_c] = trace.color
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    frames, traces = _collect_traces(grid)
    assert 1 <= len(frames) <= 2
    assert any(trace.stop_reason == "target" for trace in traces), "need at least one target collision"
    assert any(trace.stop_reason == "wall" for trace in traces), "need at least one wall stop"
    assert any(trace.color in CW_COLORS for trace in traces), "need a clockwise marker"
    assert any(trace.color in CCW_COLORS for trace in traces), "need a counterclockwise marker"
    assert len({trace.color for trace in traces}) >= 2, "need multiple visible marker colors"

    out = main(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid), "output must differ from input"

    # Every marker must matter, and flipping its turn group must change the result.
    for trace in traces:
        removed = grid.copy()
        removed[trace.marker_r, trace.marker_c] = BACKGROUND
        removed_out = removed.copy()
        for other in traces:
            if (other.marker_r, other.marker_c) == (trace.marker_r, trace.marker_c):
                continue
            removed_out[other.marker_r, other.marker_c] = BACKGROUND
            for pr, pc in other.path:
                removed_out[pr, pc] = other.color
        assert not np.array_equal(removed_out, out), "each marker must affect the output"

        recolored = grid.copy()
        if trace.color in CW_COLORS:
            recolored[trace.marker_r, trace.marker_c] = 4
        else:
            recolored[trace.marker_r, trace.marker_c] = 1
        assert not np.array_equal(main(recolored), out), "marker color must determine the turn"


def generate_input() -> np.ndarray:
    for _ in range(500):
        frame_count = 1 if random.random() < 0.55 else 2
        try:
            if frame_count == 1:
                plans = [
                    _make_frame_plan(
                        require_target=True,
                        require_wall=True,
                        require_cw=True,
                        require_ccw=True,
                    )
                ]
            else:
                plans = [
                    _make_frame_plan(
                        require_target=True,
                        require_wall=False,
                        require_cw=True,
                        require_ccw=False,
                    ),
                    _make_frame_plan(
                        require_target=False,
                        require_wall=True,
                        require_cw=False,
                        require_ccw=True,
                    ),
                ]
            H, W, origins = _place_frames(plans)
            grid = _render_input(plans, H, W, origins)
            _validate_generated(grid)
            return grid
        except (AssertionError, RuntimeError, ValueError):
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
