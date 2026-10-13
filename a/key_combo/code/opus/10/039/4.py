from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass(frozen=True)
class Frame:
    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def center(self) -> tuple[float, float]:
        return ((self.r0 + self.r1) / 2.0, (self.c0 + self.c1) / 2.0)

    def contains_interior(self, r: int, c: int) -> bool:
        return self.r0 < r < self.r1 and self.c0 < c < self.c1

    def perimeter_cells(self) -> set[tuple[int, int]]:
        cells = set()
        for cc in range(self.c0, self.c1 + 1):
            cells.add((self.r0, cc))
            cells.add((self.r1, cc))
        for rr in range(self.r0, self.r1 + 1):
            cells.add((rr, self.c0))
            cells.add((rr, self.c1))
        return cells


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in _DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> Frame:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return Frame(min(rs), min(cs), max(rs), max(cs))


def _extract_frames(grid: np.ndarray) -> list[Frame]:
    frames: list[Frame] = []
    for comp in _components_of_color(grid, GREEN):
        frame = _bbox(comp)
        if set(comp) == frame.perimeter_cells():
            frames.append(frame)
    return sorted(frames, key=lambda f: (f.r0, f.c0, f.r1, f.c1))


def _classify_red_pixels(grid: np.ndarray, frames: list[Frame]) -> tuple[list[tuple[int, int, Frame]], list[tuple[int, int]]]:
    active: list[tuple[int, int, Frame]] = []
    distractors: list[tuple[int, int]] = []
    for r, c in np.argwhere(grid == RED):
        rr = int(r)
        cc = int(c)
        containing = [frame for frame in frames if frame.contains_interior(rr, cc)]
        if len(containing) == 1:
            active.append((rr, cc, containing[0]))
        else:
            distractors.append((rr, cc))
    active.sort(key=lambda item: (item[0], item[1]))
    distractors.sort()
    return active, distractors


def _trace_marker(grid_shape: tuple[int, int], frame: Frame, start: tuple[int, int]) -> tuple[int, int, int]:
    r, c = start
    cr, cc = frame.center
    dr = _sign(r - cr)
    dc = _sign(c - cc)
    if dr == 0 and dc == 0:
        raise ValueError("marker cannot sit exactly at the frame center")

    h, w = grid_shape
    cur_r, cur_c = r, c
    dist = 0
    while True:
        nr, nc = cur_r + dr, cur_c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if (nr, nc) in frame.perimeter_cells():
            break
        cur_r, cur_c = nr, nc
        dist += 1
    return cur_r, cur_c, dist


def _draw_hollow_square(out: np.ndarray, center_r: int, center_c: int, radius: int) -> None:
    h, w = out.shape
    top = center_r - radius
    bottom = center_r + radius
    left = center_c - radius
    right = center_c + radius

    for cc in range(left, right + 1):
        for rr in (top, bottom):
            if 0 <= rr < h and 0 <= cc < w and int(out[rr, cc]) not in (GREEN, RED):
                out[rr, cc] = BLUE
    for rr in range(top, bottom + 1):
        for cc in (left, right):
            if 0 <= rr < h and 0 <= cc < w and int(out[rr, cc]) not in (GREEN, RED):
                out[rr, cc] = BLUE


def _render_from_metadata(
    grid: np.ndarray,
    frames: list[Frame],
    markers: list[tuple[int, int, Frame]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == RED] = BLACK
    traces = [((r, c), frame, _trace_marker(grid.shape, frame, (r, c))) for r, c, frame in markers]
    for _start, _frame, (stop_r, stop_c, _dist) in traces:
        out[stop_r, stop_c] = RED
    for _start, _frame, (stop_r, stop_c, dist) in traces:
        _draw_hollow_square(out, stop_r, stop_c, dist)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop valid red markers away from each green frame center and stamp blue hollow squares sized by the travel distance.

    Input structure:
    - The grid has a black background, 2-3 green 1-pixel-thick hollow rectangular frames, and red singleton pixels.
    - Some red pixels lie strictly inside a green frame's bounding box interior and are the only markers that matter.
    - Additional red pixels may appear outside every frame interior as distractors and must be ignored.

    Rule:
    - Detect every green hollow frame and keep only the red pixels strictly inside exactly one frame interior; erase all other red pixels.
    - For each valid marker, compare its position to that frame's center and move it one step at a time in the sign(row-center), sign(col-center) direction until the next step would hit the green frame or leave the grid.
    - Let d be the number of moved steps; keep a red pixel at the stop cell and stamp a blue hollow square of side 2d+1 centered there, clipped at the grid edge and skipping any green or red pixel.

    Key insight:
    - Bounding-box containment filters the true markers, the marker's position relative to the frame center encodes the movement direction, and the travel distance determines the stamped square size.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    frames = _extract_frames(g)
    active_markers, _distractors = _classify_red_pixels(g, frames)
    return _render_from_metadata(g, frames, active_markers)


def _ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _frames_too_close(a: Frame, b: Frame, clearance: int) -> bool:
    return _ranges_overlap(a.r0 - clearance, a.r1 + clearance, b.r0, b.r1) and _ranges_overlap(
        a.c0 - clearance, a.c1 + clearance, b.c0, b.c1
    )


def _sample_frames(rng: random.Random, h: int, w: int, count: int) -> list[Frame] | None:
    sizes = [7, 9, 11]
    clearance = 3
    touch_edge = rng.random() < 0.35
    frames: list[Frame] = []

    for idx in range(count):
        for _ in range(250):
            side = rng.choice(sizes if count == 2 else [7, 7, 9, 9, 11])
            if side > h or side > w:
                continue

            if idx == 0 and touch_edge:
                mode = rng.choice(["top", "bottom", "left", "right", "tl", "tr", "bl", "br"])
                if mode in {"top", "tl", "tr"}:
                    r0 = 0
                elif mode in {"bottom", "bl", "br"}:
                    r0 = h - side
                else:
                    r0 = rng.randint(0, h - side)

                if mode in {"left", "tl", "bl"}:
                    c0 = 0
                elif mode in {"right", "tr", "br"}:
                    c0 = w - side
                else:
                    c0 = rng.randint(0, w - side)
            else:
                r0 = rng.randint(0, h - side)
                c0 = rng.randint(0, w - side)

            cand = Frame(r0, c0, r0 + side - 1, c0 + side - 1)
            if any(_frames_too_close(cand, other, clearance) for other in frames):
                continue
            frames.append(cand)
            break
        else:
            return None

    return sorted(frames, key=lambda f: (f.r0, f.c0))


def _quadrant_positions(frame: Frame) -> dict[tuple[int, int], list[tuple[int, int]]]:
    center_r = (frame.r0 + frame.r1) // 2
    center_c = (frame.c0 + frame.c1) // 2
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(frame.r0 + 2, frame.r1 - 1):
        for c in range(frame.c0 + 2, frame.c1 - 1):
            if r == center_r or c == center_c:
                continue
            key = (_sign(r - center_r), _sign(c - center_c))
            buckets.setdefault(key, []).append((r, c))
    return buckets


def _pick_markers_for_frame(
    rng: random.Random,
    frame: Frame,
    used_stops: set[tuple[int, int]],
) -> list[tuple[int, int, Frame]] | None:
    buckets = _quadrant_positions(frame)
    if not buckets:
        return None

    quadrants = list(buckets)
    marker_count = rng.choice([1, 2])
    if marker_count == 2 and len(quadrants) == 1:
        marker_count = 1

    for _ in range(120):
        picks: list[tuple[int, int, Frame]] = []
        chosen_quadrants = rng.sample(quadrants, k=marker_count)
        local_stops: set[tuple[int, int]] = set()
        for quad in chosen_quadrants:
            candidates = buckets[quad][:]
            rng.shuffle(candidates)
            for r, c in candidates:
                stop_r, stop_c, dist = _trace_marker((frame.r1 + 1, frame.c1 + 1), frame, (r, c))
                stop = (stop_r, stop_c)
                if dist < 1 or stop in used_stops or stop in local_stops:
                    continue
                local_stops.add(stop)
                picks.append((r, c, frame))
                break
        if len(picks) == marker_count:
            return picks
    return None


def _make_frame_canvas(grid: np.ndarray, frame: Frame) -> None:
    for r, c in frame.perimeter_cells():
        grid[r, c] = GREEN


def _scene_output_without_distractors(
    h: int,
    w: int,
    frames: list[Frame],
    markers: list[tuple[int, int, Frame]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for frame in frames:
        _make_frame_canvas(grid, frame)
    for r, c, _frame in markers:
        grid[r, c] = RED
    return _render_from_metadata(grid, frames, sorted(markers, key=lambda x: (x[0], x[1])))


def _distance_to_bbox(frame: Frame, r: int, c: int) -> int:
    dr = 0 if frame.r0 <= r <= frame.r1 else min(abs(r - frame.r0), abs(r - frame.r1))
    dc = 0 if frame.c0 <= c <= frame.c1 else min(abs(c - frame.c0), abs(c - frame.c1))
    if dr == 0:
        return dc
    if dc == 0:
        return dr
    return dr + dc


def _choose_distractors(
    rng: random.Random,
    grid: np.ndarray,
    frames: list[Frame],
    output_wo_distractors: np.ndarray,
    count: int,
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    base_candidates = []
    visible_erase = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BLACK:
                continue
            if any(frame.contains_interior(r, c) or (r, c) in frame.perimeter_cells() for frame in frames):
                continue
            dmin = min(_distance_to_bbox(frame, r, c) for frame in frames)
            if not (1 <= dmin <= 5):
                continue
            base_candidates.append((r, c))
            if int(output_wo_distractors[r, c]) == BLACK:
                visible_erase.append((r, c))

    if len(base_candidates) < count or not visible_erase:
        return None

    chosen = [rng.choice(visible_erase)]
    pool = [cell for cell in base_candidates if cell != chosen[0]]
    rng.shuffle(pool)
    for cell in pool:
        if len(chosen) == count:
            break
        if any(max(abs(cell[0] - rr), abs(cell[1] - cc)) <= 1 for rr, cc in chosen):
            continue
        chosen.append(cell)
    if len(chosen) != count:
        return None
    return sorted(chosen)


def _build_input(
    h: int,
    w: int,
    frames: list[Frame],
    markers: list[tuple[int, int, Frame]],
    distractors: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for frame in frames:
        _make_frame_canvas(grid, frame)
    for r, c, _frame in markers:
        grid[r, c] = RED
    for r, c in distractors:
        grid[r, c] = RED
    return grid


def _validate_scene(
    grid: np.ndarray,
    frames: list[Frame],
    markers: list[tuple[int, int, Frame]],
    distractors: list[tuple[int, int]],
) -> None:
    parsed_frames = _extract_frames(grid)
    assert {(f.r0, f.c0, f.r1, f.c1) for f in parsed_frames} == {(f.r0, f.c0, f.r1, f.c1) for f in frames}

    active, parsed_distractors = _classify_red_pixels(grid, parsed_frames)
    assert {(r, c) for r, c, _ in active} == {(r, c) for r, c, _ in markers}
    assert set(parsed_distractors) == set(distractors)

    dirs = []
    dists = []
    for r, c, frame in active:
        cr, cc = frame.center
        dirs.append((_sign(r - cr), _sign(c - cc)))
        stop_r, stop_c, dist = _trace_marker(grid.shape, frame, (r, c))
        assert dist >= 1
        dists.append(dist)
        assert frame.contains_interior(stop_r, stop_c)
    assert len(set(dirs)) >= 2
    assert len(set(dists)) >= 2

    expected = _scene_output_without_distractors(grid.shape[0], grid.shape[1], frames, markers)
    solved = main(grid)
    assert np.array_equal(solved, expected)
    assert np.count_nonzero((grid == RED) & (expected == BLACK)) >= 1
    assert np.count_nonzero(expected == BLUE) >= len(markers)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        h = rng.randint(20, 28)
        w = rng.randint(20, 28)
        frame_count = rng.choice([2, 3])
        frames = _sample_frames(rng, h, w, frame_count)
        if frames is None:
            continue

        markers: list[tuple[int, int, Frame]] = []
        used_stops: set[tuple[int, int]] = set()
        per_frame_counts = []
        for frame in frames:
            picked = _pick_markers_for_frame(rng, frame, used_stops)
            if picked is None:
                break
            for r, c, picked_frame in picked:
                stop_r, stop_c, _dist = _trace_marker((h, w), picked_frame, (r, c))
                used_stops.add((stop_r, stop_c))
            markers.extend(picked)
            per_frame_counts.append(len(picked))
        else:
            if not all(1 <= n <= 2 for n in per_frame_counts):
                continue
            output_wo_distractors = _scene_output_without_distractors(h, w, frames, markers)
            distractor_count = rng.choice([2, 3])
            temp_grid = _build_input(h, w, frames, markers, [])
            distractors = _choose_distractors(rng, temp_grid, frames, output_wo_distractors, distractor_count)
            if distractors is None:
                continue
            grid = _build_input(h, w, frames, markers, distractors)
            try:
                _validate_scene(grid, frames, markers, distractors)
            except AssertionError:
                continue
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
