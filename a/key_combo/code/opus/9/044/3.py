from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
SEED_COLORS = (BLUE, GREEN, YELLOW)


@dataclass(frozen=True)
class Frame:
    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def center_r(self) -> float:
        return (self.r0 + self.r1) / 2.0

    @property
    def center_c(self) -> float:
        return (self.c0 + self.c1) / 2.0

    @property
    def border(self) -> list[tuple[int, int]]:
        pts: list[tuple[int, int]] = []
        for c in range(self.c0, self.c1 + 1):
            pts.append((self.r0, c))
            pts.append((self.r1, c))
        for r in range(self.r0 + 1, self.r1):
            pts.append((r, self.c0))
            pts.append((r, self.c1))
        return pts

    def is_corner(self, cell: tuple[int, int]) -> bool:
        r, c = cell
        return r in {self.r0, self.r1} and c in {self.c0, self.c1}

    def contains(self, r: int, c: int) -> bool:
        return self.r0 <= r <= self.r1 and self.c0 <= c <= self.c1


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_from_component(comp: list[tuple[int, int]]) -> Frame:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    frame = Frame(min(rows), min(cols), max(rows), max(cols))
    assert frame.r1 - frame.r0 + 1 >= 4
    assert frame.c1 - frame.c0 + 1 >= 4

    expected = set(frame.border)
    actual = set(comp)
    assert actual == expected, "red component must be a 1-pixel hollow rectangle"
    return frame


def _find_frames(grid: np.ndarray) -> list[Frame]:
    frames = [_frame_from_component(comp) for comp in _components_of_color(grid, RED)]
    frames.sort(key=lambda fr: (fr.r0, fr.c0, fr.r1, fr.c1))
    assert 2 <= len(frames) <= 3
    return frames


def _find_seeds(grid: np.ndarray) -> list[tuple[int, int, int]]:
    seeds: list[tuple[int, int, int]] = []
    for r, c in np.argwhere((grid != BG) & (grid != RED)):
        color = int(grid[r, c])
        assert color in SEED_COLORS
        seeds.append((int(r), int(c), color))
    assert 3 <= len(seeds) <= 5
    seeds.sort()
    return seeds


def _frame_distance(frame: Frame, seed: tuple[int, int]) -> int:
    sr, sc = seed
    return min(abs(sr - r) + abs(sc - c) for r, c in frame.border)


def _assign_frame(seed: tuple[int, int], frames: list[Frame]) -> int:
    ranked = sorted(
        ((int(_frame_distance(frame, seed)), frame.r0, frame.c0, idx) for idx, frame in enumerate(frames))
    )
    return ranked[0][3]


def _clockwise(dr: int, dc: int) -> tuple[int, int]:
    return dc, -dr


def _toward_center(value: int, center: float) -> int:
    return 1 if value < center else -1


def _trace_seed(frame: Frame, seed: tuple[int, int]) -> dict:
    sr, sc = seed
    candidates: list[tuple[int, int, str, tuple[int, int]]] = []
    if frame.r0 <= sr <= frame.r1:
        if sc < frame.c0:
            candidates.append((0, 1, "left", (sr, frame.c0)))
        if sc > frame.c1:
            candidates.append((0, -1, "right", (sr, frame.c1)))
    if frame.c0 <= sc <= frame.c1:
        if sr < frame.r0:
            candidates.append((1, 0, "top", (frame.r0, sc)))
        if sr > frame.r1:
            candidates.append((-1, 0, "bottom", (frame.r1, sc)))
    assert len(candidates) == 1, "seed must align with exactly one incoming ray toward its frame"

    dr, dc, wall_side, hit = candidates[0]
    hr, hc = hit
    outer: list[tuple[int, int]] = []
    cr, cc = sr + dr, sc + dc
    while (cr, cc) != hit:
        outer.append((cr, cc))
        cr += dr
        cc += dc

    corner = frame.is_corner(hit)
    if corner:
        if hit == (frame.r0, frame.c0):
            start = (frame.r0 + 1, frame.c0 + 1)
        elif hit == (frame.r0, frame.c1):
            start = (frame.r0 + 1, frame.c1 - 1)
        elif hit == (frame.r1, frame.c1):
            start = (frame.r1 - 1, frame.c1 - 1)
        else:
            start = (frame.r1 - 1, frame.c0 + 1)
        ndr, ndc = _clockwise(dr, dc)
        if not (frame.r0 < start[0] < frame.r1 and frame.c0 < start[1] < frame.c1):
            raise AssertionError("reflection must enter the frame interior")
        trial = (start[0] + ndr, start[1] + ndc)
        if not (frame.r0 <= trial[0] <= frame.r1 and frame.c0 <= trial[1] <= frame.c1):
            if dr == 0:
                ndr, ndc = _toward_center(hr, frame.center_r), 0
            else:
                ndr, ndc = 0, _toward_center(hc, frame.center_c)
    else:
        if wall_side == "left":
            start = (hr, frame.c0 + 1)
            ndr, ndc = _toward_center(hr, frame.center_r), 0
        elif wall_side == "right":
            start = (hr, frame.c1 - 1)
            ndr, ndc = _toward_center(hr, frame.center_r), 0
        elif wall_side == "top":
            start = (frame.r0 + 1, hc)
            ndr, ndc = 0, _toward_center(hc, frame.center_c)
        else:
            start = (frame.r1 - 1, hc)
            ndr, ndc = 0, _toward_center(hc, frame.center_c)

    inner: list[tuple[int, int]] = []
    cr, cc = start
    while frame.r0 < cr < frame.r1 and frame.c0 < cc < frame.c1:
        inner.append((cr, cc))
        cr += ndr
        cc += ndc
    assert inner, "reflection segment must occupy at least one interior cell"

    return {
        "seed": seed,
        "frame": frame,
        "incoming": (dr, dc),
        "wall_side": wall_side,
        "hit": hit,
        "outer": outer,
        "inner": inner,
        "corner": corner,
    }


def _solve_with_traces(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[Frame]]:
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    seeds = _find_seeds(g)

    out = g.copy()
    traces: list[dict] = []
    for r, c, color in seeds:
        frame = frames[_assign_frame((r, c), frames)]
        trace = _trace_seed(frame, (r, c))
        traces.append(trace)
        for rr, cc in trace["outer"]:
            out[rr, cc] = GREY
        for rr, cc in trace["inner"]:
            out[rr, cc] = color

    for frame in frames:
        for rr, cc in frame.border:
            out[rr, cc] = RED
    return out, traces, frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Send each colored seed toward its nearest red frame, then reflect the ray into the frame interior.

    Input structure:
    - A black background contains 2 or 3 red hollow rectangular frames.
    - Outside the frames sit 3 to 5 single-pixel seeds colored blue, green, or yellow.
    - Every seed is horizontally or vertically aligned with one wall of its nearest frame.

    Rule:
    - Assign each seed to the red frame whose border has the smallest Manhattan distance to that seed; break ties by the frame's topmost row, then leftmost column.
    - Extend a ray from the seed straight toward the aligned wall, coloring only the exterior cells gray and leaving the seed itself unchanged.
    - When the ray reaches the frame wall, keep the wall red and continue from the wall's inner side after a 90-degree turn: horizontal arrivals at vertical walls turn vertically toward the frame center, vertical arrivals at horizontal walls turn horizontally toward the frame center, and corner hits use the clockwise turn.
    - Color the interior reflection segment with the seed's original color and stop before covering the opposite wall.

    Key insight:
    - The same red frame border plays two roles at once: it terminates the outside ray and also acts as the mirror that redirects the ray inside the box.
    """
    out, _traces, _frames = _solve_with_traces(grid)
    return out


def _bbox_gap(a: Frame, b: Frame) -> tuple[int, int]:
    row_gap = max(0, max(a.r0 - b.r1 - 1, b.r0 - a.r1 - 1))
    col_gap = max(0, max(a.c0 - b.c1 - 1, b.c0 - a.c1 - 1))
    return int(row_gap), int(col_gap)


def _frames_far_enough(a: Frame, b: Frame) -> bool:
    row_gap, col_gap = _bbox_gap(a, b)
    return row_gap >= 3 or col_gap >= 3


def _place_frames(h: int, w: int, rng: random.Random) -> list[Frame] | None:
    for _ in range(300):
        count = rng.randint(2, 3)
        frames: list[Frame] = []
        ok = True
        for _idx in range(count):
            placed = False
            for _attempt in range(120):
                fh = rng.randint(4, 8)
                fw = rng.randint(4, 8)
                if min(fh, fw) > 7:
                    continue
                r0 = rng.randint(1, h - fh - 2)
                c0 = rng.randint(1, w - fw - 2)
                frame = Frame(r0, c0, r0 + fh - 1, c0 + fw - 1)
                if all(_frames_far_enough(frame, other) for other in frames):
                    frames.append(frame)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            frames.sort(key=lambda fr: (fr.r0, fr.c0))
            return frames
    return None


def _render_frames(h: int, w: int, frames: list[Frame]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for frame in frames:
        for r, c in frame.border:
            grid[r, c] = RED
    return grid


def _axis_values(frame: Frame, axis: str) -> list[int]:
    if axis in {"left", "right"}:
        values = list(range(frame.r0 + 1, frame.r1))
        if frame.center_r.is_integer():
            center = int(frame.center_r)
            values = [v for v in values if v != center]
        return values
    values = list(range(frame.c0 + 1, frame.c1))
    if frame.center_c.is_integer():
        center = int(frame.center_c)
        values = [v for v in values if v != center]
    return values


def _candidate_proposals(
    frames: list[Frame],
    base_grid: np.ndarray,
    target_idx: int,
    rng: random.Random,
) -> list[dict]:
    h, w = base_grid.shape
    frame = frames[target_idx]
    proposals: list[dict] = []

    def add_seed(pos: tuple[int, int], tag: str) -> None:
        sr, sc = pos
        if not _in_bounds(base_grid.shape, sr, sc):
            return
        if int(base_grid[sr, sc]) != BG:
            return
        if _assign_frame((sr, sc), frames) != target_idx:
            return
        trace = _trace_seed(frame, (sr, sc))
        if len(trace["outer"]) < 1 or len(trace["inner"]) < 1:
            return
        if any(int(base_grid[r, c]) != BG for r, c in trace["outer"]):
            return
        proposals.append(
            {
                "seed": (sr, sc),
                "trace": trace,
                "frame_idx": target_idx,
                "axis": "horizontal" if trace["incoming"][0] == 0 else "vertical",
                "tag": tag,
            }
        )

    for r in _axis_values(frame, "left"):
        for gap in range(2, frame.c0 + 1):
            add_seed((r, frame.c0 - gap), "left")
    for r in _axis_values(frame, "right"):
        for gap in range(2, w - frame.c1):
            add_seed((r, frame.c1 + gap), "right")
    for c in _axis_values(frame, "top"):
        for gap in range(2, frame.r0 + 1):
            add_seed((frame.r0 - gap, c), "top")
    for c in _axis_values(frame, "bottom"):
        for gap in range(2, h - frame.r1):
            add_seed((frame.r1 + gap, c), "bottom")

    for gap in range(2, frame.c0 + 1):
        add_seed((frame.r0, frame.c0 - gap), "corner_tl_from_left")
    for gap in range(2, frame.r0 + 1):
        add_seed((frame.r0 - gap, frame.c1), "corner_tr_from_top")
    for gap in range(2, w - frame.c1):
        add_seed((frame.r1, frame.c1 + gap), "corner_br_from_right")
    for gap in range(2, h - frame.r1):
        add_seed((frame.r1 + gap, frame.c0), "corner_bl_from_bottom")

    rng.shuffle(proposals)
    unique: dict[tuple[int, int], dict] = {}
    for proposal in proposals:
        unique.setdefault(proposal["seed"], proposal)
    return list(unique.values())


def _tie_proposals(frames: list[Frame], base_grid: np.ndarray, rng: random.Random) -> list[dict]:
    proposals: list[dict] = []
    for i, frame_a in enumerate(frames):
        for j, frame_b in enumerate(frames):
            if i >= j:
                continue

            if frame_a.c1 < frame_b.c0 and (frame_a.c1 + frame_b.c0) % 2 == 0:
                col = (frame_a.c1 + frame_b.c0) // 2
                if frame_b.c0 - frame_a.c1 >= 4:
                    rows = range(max(frame_a.r0 + 1, frame_b.r0 + 1), min(frame_a.r1 - 1, frame_b.r1 - 1) + 1)
                    for row in rows:
                        if frame_a.center_r.is_integer() and row == int(frame_a.center_r):
                            continue
                        if frame_b.center_r.is_integer() and row == int(frame_b.center_r):
                            continue
                        if int(base_grid[row, col]) != BG:
                            continue
                        winner = _assign_frame((row, col), frames)
                        expected = min((i, frame_a.r0, frame_a.c0), (j, frame_b.r0, frame_b.c0))[0]
                        if winner != expected:
                            continue
                        trace = _trace_seed(frames[winner], (row, col))
                        if len(trace["outer"]) >= 1 and len(trace["inner"]) >= 1:
                            proposals.append(
                                {
                                    "seed": (row, col),
                                    "trace": trace,
                                    "frame_idx": winner,
                                    "axis": "horizontal",
                                    "tag": "tie_horizontal",
                                }
                            )

            if frame_a.r1 < frame_b.r0 and (frame_a.r1 + frame_b.r0) % 2 == 0:
                row = (frame_a.r1 + frame_b.r0) // 2
                if frame_b.r0 - frame_a.r1 >= 4:
                    cols = range(max(frame_a.c0 + 1, frame_b.c0 + 1), min(frame_a.c1 - 1, frame_b.c1 - 1) + 1)
                    for col in cols:
                        if frame_a.center_c.is_integer() and col == int(frame_a.center_c):
                            continue
                        if frame_b.center_c.is_integer() and col == int(frame_b.center_c):
                            continue
                        if int(base_grid[row, col]) != BG:
                            continue
                        winner = _assign_frame((row, col), frames)
                        expected = min((i, frame_a.r0, frame_a.c0), (j, frame_b.r0, frame_b.c0))[0]
                        if winner != expected:
                            continue
                        trace = _trace_seed(frames[winner], (row, col))
                        if len(trace["outer"]) >= 1 and len(trace["inner"]) >= 1:
                            proposals.append(
                                {
                                    "seed": (row, col),
                                    "trace": trace,
                                    "frame_idx": winner,
                                    "axis": "vertical",
                                    "tag": "tie_vertical",
                                }
                            )
    rng.shuffle(proposals)
    unique: dict[tuple[int, int], dict] = {}
    for proposal in proposals:
        unique.setdefault(proposal["seed"], proposal)
    return list(unique.values())


def _compatible_with_selected(proposal: dict, selected: list[dict]) -> bool:
    seed = proposal["seed"]
    outer = set(proposal["trace"]["outer"])
    inner = set(proposal["trace"]["inner"])
    occupied = {item["seed"] for item in selected}
    used_outer = set()
    used_inner = set()
    for item in selected:
        used_outer.update(item["trace"]["outer"])
        used_inner.update(item["trace"]["inner"])

    if seed in occupied or seed in used_outer or seed in used_inner:
        return False
    if outer & occupied or inner & occupied:
        return False
    if outer & used_outer:
        return False
    if outer & used_inner:
        return False
    if inner & used_outer:
        return False
    if inner & used_inner:
        return False
    return True


def _pick_seed_set(frames: list[Frame], base_grid: np.ndarray, rng: random.Random) -> list[dict] | None:
    per_frame = [_candidate_proposals(frames, base_grid, idx, rng) for idx in range(len(frames))]
    if any(not proposals for proposals in per_frame):
        return None

    tie_pool = _tie_proposals(frames, base_grid, rng)
    target_count = rng.randint(max(3, len(frames)), 5)
    want_corner = rng.random() < 0.55
    want_tie = tie_pool and target_count > len(frames) and rng.random() < 0.35

    for _ in range(160):
        selected: list[dict] = []
        corner_used = False

        if want_tie:
            for proposal in tie_pool:
                if _compatible_with_selected(proposal, selected):
                    selected.append(proposal)
                    corner_used = corner_used or proposal["trace"]["corner"]
                    break

        for idx, proposals in enumerate(per_frame):
            ordered = proposals[:]
            rng.shuffle(ordered)
            if want_corner and not corner_used:
                ordered.sort(key=lambda item: 0 if item["trace"]["corner"] else 1)
            picked = None
            for proposal in ordered:
                if any(item["frame_idx"] == idx for item in selected):
                    break
                if _compatible_with_selected(proposal, selected):
                    picked = proposal
                    break
            if picked is None and not any(item["frame_idx"] == idx for item in selected):
                for proposal in ordered:
                    if _compatible_with_selected(proposal, selected):
                        picked = proposal
                        break
            if picked is None:
                selected = []
                break
            if not any(item["frame_idx"] == idx for item in selected):
                selected.append(picked)
                corner_used = corner_used or picked["trace"]["corner"]

        if not selected:
            continue

        pool = [item for proposals in per_frame for item in proposals] + tie_pool
        rng.shuffle(pool)
        for proposal in pool:
            if len(selected) >= target_count:
                break
            if _compatible_with_selected(proposal, selected):
                selected.append(proposal)
                corner_used = corner_used or proposal["trace"]["corner"]

        if len(selected) != target_count:
            continue
        if any(not any(item["frame_idx"] == idx for item in selected) for idx in range(len(frames))):
            continue

        axes = {item["axis"] for item in selected}
        if axes != {"horizontal", "vertical"}:
            continue
        if want_corner and not corner_used:
            continue
        return selected
    return None


def _assign_colors(selected: list[dict], rng: random.Random) -> list[int]:
    colors = [rng.choice(SEED_COLORS) for _ in selected]
    if len(set(colors)) == 1 and len(colors) > 1:
        colors[0] = SEED_COLORS[(SEED_COLORS.index(colors[0]) + 1) % len(SEED_COLORS)]
    return colors


def _wrong_outer_only(grid: np.ndarray, traces: list[dict]) -> np.ndarray:
    out = grid.copy()
    for trace in traces:
        for r, c in trace["outer"]:
            out[r, c] = GREY
    return out


def _wrong_no_switch(grid: np.ndarray, traces: list[dict]) -> np.ndarray:
    out = grid.copy()
    for trace in traces:
        sr, sc = trace["seed"]
        color = int(grid[sr, sc])
        for r, c in trace["outer"]:
            out[r, c] = color
        for r, c in trace["inner"]:
            out[r, c] = color
    return out


def _wrong_straight_through(grid: np.ndarray, traces: list[dict]) -> np.ndarray:
    out = grid.copy()
    for trace in traces:
        sr, sc = trace["seed"]
        color = int(grid[sr, sc])
        for r, c in trace["outer"]:
            out[r, c] = GREY
        dr, dc = trace["incoming"]
        hr, hc = trace["hit"]
        frame: Frame = trace["frame"]
        if dr == 0:
            cc = hc + dc
            while frame.c0 < cc < frame.c1:
                out[hr, cc] = color
                cc += dc
        else:
            rr = hr + dr
            while frame.r0 < rr < frame.r1:
                out[rr, hc] = color
                rr += dr
    return out


def _validate_generated(grid: np.ndarray) -> None:
    out, traces, frames = _solve_with_traces(grid)
    seeds = _find_seeds(grid)
    assert 2 <= len(frames) <= 3
    assert 3 <= len(seeds) <= 5
    assert len(set(color for _, _, color in seeds)) >= 2

    per_frame = [0] * len(frames)
    axes = set()
    for trace in traces:
        frame_idx = frames.index(trace["frame"])
        per_frame[frame_idx] += 1
        axes.add("horizontal" if trace["incoming"][0] == 0 else "vertical")
        sr, sc = trace["seed"]
        seed_color = int(grid[sr, sc])
        assert trace["outer"]
        assert trace["inner"]
        for r, c in trace["outer"]:
            assert int(out[r, c]) == GREY
        for r, c in trace["inner"]:
            assert int(out[r, c]) == seed_color
        hr, hc = trace["hit"]
        assert int(out[hr, hc]) == RED

    assert all(count >= 1 for count in per_frame)
    assert axes == {"horizontal", "vertical"}
    assert not np.array_equal(out, grid)
    assert not np.array_equal(out, _wrong_outer_only(grid, traces))
    assert not np.array_equal(out, _wrong_no_switch(grid, traces))
    assert not np.array_equal(out, _wrong_straight_through(grid, traces))


def generate_input() -> np.ndarray:
    for attempt in range(500):
        rng = random.Random(random.randrange(1 << 30) + attempt)
        h = rng.randint(18, 24)
        w = rng.randint(18, 24)
        frames = _place_frames(h, w, rng)
        if frames is None:
            continue
        grid = _render_frames(h, w, frames)
        selected = _pick_seed_set(frames, grid, rng)
        if selected is None:
            continue

        colors = _assign_colors(selected, rng)
        for item, color in zip(selected, colors):
            r, c = item["seed"]
            grid[r, c] = color

        try:
            _validate_generated(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid ray-reflection puzzle instance")
