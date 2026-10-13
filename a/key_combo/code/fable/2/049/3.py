from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RED = 2
RING_COLORS = tuple(color for color in range(1, 10) if color != RED)
SIDE_NAMES = ("top", "bottom", "left", "right")

MAX_GENERATION_ATTEMPTS = 800
GRID_HEIGHT_RANGE = (10, 24)
GRID_WIDTH_RANGE = (10, 24)
CENTER_MARGIN = 3


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _segment_key(seg: dict) -> tuple[int, int, int, str]:
    return int(seg["fixed"]), int(seg["lo"]), int(seg["hi"]), str(seg["orientation"])


def _extract_segments(grid: np.ndarray, color: int) -> list[dict]:
    segments: list[dict] = []
    for coords in _components_mask(grid == int(color)):
        r0, c0, r1, c1 = _bbox_coords(coords)
        if r0 == r1:
            length = c1 - c0 + 1
            assert length >= 2, "each fragment must be a non-corner line segment"
            assert int(coords.shape[0]) == length, "horizontal fragment must be contiguous"
            assert bool((grid[r0, c0 : c1 + 1] == int(color)).all())
            segments.append({"orientation": "h", "fixed": int(r0), "lo": int(c0), "hi": int(c1)})
        elif c0 == c1:
            length = r1 - r0 + 1
            assert length >= 2, "each fragment must be a non-corner line segment"
            assert int(coords.shape[0]) == length, "vertical fragment must be contiguous"
            assert bool((grid[r0 : r1 + 1, c0] == int(color)).all())
            segments.append({"orientation": "v", "fixed": int(c0), "lo": int(r0), "hi": int(r1)})
        else:
            raise AssertionError("each foreground component must be a straight segment")
    assert segments, "expected at least one fragment"
    segments.sort(key=_segment_key)
    return segments


def _segment_radius_for_center(seg: dict, center: tuple[int, int]) -> int | None:
    cr, cc = map(int, center)
    if seg["orientation"] == "h":
        d = abs(int(seg["fixed"]) - cr)
        lateral = max(abs(int(seg["lo"]) - cc), abs(int(seg["hi"]) - cc))
    else:
        d = abs(int(seg["fixed"]) - cc)
        lateral = max(abs(int(seg["lo"]) - cr), abs(int(seg["hi"]) - cr))
    if d <= lateral:
        return None
    return int(d)


def _find_solutions(shape: tuple[int, int], segments: list[dict], blocked: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, ...]]]:
    h, w = map(int, shape)
    out: list[tuple[tuple[int, int], tuple[int, ...]]] = []
    for cr in range(h):
        for cc in range(w):
            if bool(blocked[cr, cc]):
                continue
            radii: set[int] = set()
            ok = True
            for seg in segments:
                d = _segment_radius_for_center(seg, (cr, cc))
                if d is None:
                    ok = False
                    break
                radii.add(int(d))
            if ok:
                out.append(((int(cr), int(cc)), tuple(sorted(radii))))
    return out


def _render_output(shape: tuple[int, int], center: tuple[int, int], radii: tuple[int, ...], ring_color: int) -> np.ndarray:
    h, w = map(int, shape)
    cr, cc = map(int, center)
    out = np.full((h, w), BG, dtype=int)
    for d in radii:
        top = cr - d
        bottom = cr + d
        left = cc - d
        right = cc + d
        c0 = max(0, left)
        c1 = min(w - 1, right)
        r0 = max(0, top)
        r1 = min(h - 1, bottom)
        if 0 <= top < h:
            out[top, c0 : c1 + 1] = int(ring_color)
        if 0 <= bottom < h:
            out[bottom, c0 : c1 + 1] = int(ring_color)
        if 0 <= left < w:
            out[r0 : r1 + 1, left] = int(ring_color)
        if 0 <= right < w:
            out[r0 : r1 + 1, right] = int(ring_color)
    out[cr, cc] = RED
    return out


def _solve_puzzle(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...], int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert max(g.shape) <= 30
    colors = sorted(int(color) for color in np.unique(g) if int(color) != BG)
    assert len(colors) == 1, "input must contain exactly one non-background fragment color"
    ring_color = int(colors[0])
    assert ring_color != RED, "fragment color must differ from the red center marker"
    segments = _extract_segments(g, ring_color)
    solutions = _find_solutions(g.shape, segments, g != BG)
    assert len(solutions) == 1, "fragments must determine a unique hidden center"
    center, radii = solutions[0]
    assert radii, "at least one radius must be present"
    return center, radii, ring_color, segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden common center from cornerless square-ring fragments, then complete every indicated ring and mark the center.

    Input structure:
    - The background is black.
    - All foreground pixels share one non-red color and form separate horizontal or vertical straight fragments.
    - Every fragment is a cornerless subset of one edge from a concentric Chebyshev square ring around one hidden center cell.

    Rule:
    - Find the unique black cell that can serve as the common center for all fragments.
    - For each radius that has at least one fragment, draw the entire square ring of that radius, clipped by the grid boundary.
    - Paint the recovered center cell red.

    Key insight:
    - A lone straight residue never fixes the center; only the joint constraints from multiple fragment orientations and radii reveal the missing concentric generator.
    """
    center, radii, ring_color, _segments = _solve_puzzle(np.asarray(grid, dtype=int))
    return _render_output(grid.shape, center, radii, ring_color)


def _ring_is_clipped(shape: tuple[int, int], center: tuple[int, int], radius: int) -> bool:
    h, w = map(int, shape)
    cr, cc = map(int, center)
    return cr - radius < 0 or cr + radius >= h or cc - radius < 0 or cc + radius >= w


def _visible_run(shape: tuple[int, int], center: tuple[int, int], radius: int, side: str) -> dict | None:
    h, w = map(int, shape)
    cr, cc = map(int, center)
    if side == "top":
        row = cr - radius
        if not (0 <= row < h):
            return None
        lo = max(0, cc - radius + 1)
        hi = min(w - 1, cc + radius - 1)
        orientation, fixed = "h", row
    elif side == "bottom":
        row = cr + radius
        if not (0 <= row < h):
            return None
        lo = max(0, cc - radius + 1)
        hi = min(w - 1, cc + radius - 1)
        orientation, fixed = "h", row
    elif side == "left":
        col = cc - radius
        if not (0 <= col < w):
            return None
        lo = max(0, cr - radius + 1)
        hi = min(h - 1, cr + radius - 1)
        orientation, fixed = "v", col
    elif side == "right":
        col = cc + radius
        if not (0 <= col < w):
            return None
        lo = max(0, cr - radius + 1)
        hi = min(h - 1, cr + radius - 1)
        orientation, fixed = "v", col
    else:
        raise ValueError(f"unknown side: {side}")
    if hi - lo + 1 < 2:
        return None
    return {
        "side": side,
        "orientation": orientation,
        "fixed": int(fixed),
        "lo": int(lo),
        "hi": int(hi),
        "radius": int(radius),
    }


def _visible_runs(shape: tuple[int, int], center: tuple[int, int], radius: int) -> list[dict]:
    runs = []
    for side in SIDE_NAMES:
        run = _visible_run(shape, center, radius, side)
        if run is not None:
            runs.append(run)
    return runs


def _choose_fragment(run: dict, center: tuple[int, int]) -> dict:
    axis_center = int(center[1] if run["orientation"] == "h" else center[0])
    lo = int(run["lo"])
    hi = int(run["hi"])
    run_len = hi - lo + 1
    candidates: list[tuple[int, int]] = []
    for length in range(2, run_len + 1):
        for start in range(lo, hi - length + 2):
            candidates.append((int(start), int(start + length - 1)))
    if run_len > 2 and random.random() < 0.8:
        partial = [cand for cand in candidates if cand[1] - cand[0] + 1 < run_len]
        if partial:
            candidates = partial
    off_center = [cand for cand in candidates if cand[0] + cand[1] != 2 * axis_center]
    if off_center and random.random() < 0.8:
        candidates = off_center
    seg_lo, seg_hi = random.choice(candidates)
    seg = dict(run)
    seg["lo"] = int(seg_lo)
    seg["hi"] = int(seg_hi)
    seg["off_center"] = bool(seg_lo + seg_hi != 2 * axis_center)
    return seg


def _render_input(shape: tuple[int, int], segments: list[dict], color: int) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for seg in segments:
        if seg["orientation"] == "h":
            grid[int(seg["fixed"]), int(seg["lo"]) : int(seg["hi"]) + 1] = int(color)
        else:
            grid[int(seg["lo"]) : int(seg["hi"]) + 1, int(seg["fixed"])] = int(color)
    return grid


def _subset_solutions(shape: tuple[int, int], segments: list[dict], color: int) -> list[tuple[tuple[int, int], tuple[int, ...]]]:
    subset_grid = _render_input(shape, segments, color)
    return _find_solutions(shape, segments, subset_grid != BG)


def _matches_unique_solution(
    shape: tuple[int, int],
    segments: list[dict],
    color: int,
    center: tuple[int, int],
    radii: tuple[int, ...],
) -> bool:
    sols = _subset_solutions(shape, segments, color)
    return len(sols) == 1 and sols[0][0] == center and sols[0][1] == radii


def _center_is_uniquely_known(
    shape: tuple[int, int],
    segments: list[dict],
    color: int,
    center: tuple[int, int],
) -> bool:
    sols = _subset_solutions(shape, segments, color)
    return len(sols) == 1 and sols[0][0] == center


def _semantic_checks(
    shape: tuple[int, int],
    segments: list[dict],
    color: int,
    center: tuple[int, int],
    radii: tuple[int, ...],
) -> bool:
    if len(segments) < 3 or len(radii) < 2:
        return False
    if {seg["orientation"] for seg in segments} != {"h", "v"}:
        return False
    if not any(bool(seg.get("off_center", False)) for seg in segments):
        return False
    if not _matches_unique_solution(shape, segments, color, center, radii):
        return False

    horizontal = [seg for seg in segments if seg["orientation"] == "h"]
    vertical = [seg for seg in segments if seg["orientation"] == "v"]
    if _center_is_uniquely_known(shape, horizontal, color, center):
        return False
    if _center_is_uniquely_known(shape, vertical, color, center):
        return False

    for radius in radii:
        same_radius = [seg for seg in segments if int(seg["radius"]) == int(radius)]
        if _center_is_uniquely_known(shape, same_radius, color, center):
            return False

    for idx in range(len(segments)):
        subset = segments[:idx] + segments[idx + 1 :]
        if _matches_unique_solution(shape, subset, color, center, radii):
            return False
    return True


def _sample_center(h: int, w: int, clipped_mode: bool) -> tuple[int, int] | None:
    candidates = []
    for r in range(h):
        for c in range(w):
            margin = min(r, h - 1 - r, c, w - 1 - c)
            if clipped_mode and margin <= 2:
                candidates.append((r, c))
            if not clipped_mode and margin >= CENTER_MARGIN:
                candidates.append((r, c))
    return random.choice(candidates) if candidates else None


def _feasible_radii(shape: tuple[int, int], center: tuple[int, int]) -> tuple[list[int], list[int], list[int]]:
    h, w = map(int, shape)
    cr, cc = map(int, center)
    limit = max(cr, h - 1 - cr, cc, w - 1 - cc)
    feasible: list[int] = []
    full: list[int] = []
    clipped: list[int] = []
    for radius in range(1, limit + 1):
        runs = _visible_runs(shape, center, radius)
        if not runs:
            continue
        feasible.append(int(radius))
        if _ring_is_clipped(shape, center, radius):
            clipped.append(int(radius))
        else:
            full.append(int(radius))
    return feasible, full, clipped


def _pick_radii(full: list[int], clipped: list[int], clipped_mode: bool) -> list[int] | None:
    if clipped_mode:
        feasible = sorted(set(full) | set(clipped))
        if len(feasible) < 2 or not clipped:
            return None
        count = min(len(feasible), random.choice((2, 2, 3, 3, 4)))
        first = random.choice(clipped)
        rest = [radius for radius in feasible if radius != first]
        if len(rest) < count - 1:
            return None
        picked = [int(first)] + random.sample(rest, count - 1)
        return sorted(picked)
    if len(full) < 2:
        return None
    count = min(len(full), random.choice((2, 2, 3, 3, 4)))
    return sorted(random.sample(full, count))


def _pick_segments(shape: tuple[int, int], center: tuple[int, int], radii: list[int]) -> list[dict] | None:
    runs_by_radius = {int(radius): _visible_runs(shape, center, int(radius)) for radius in radii}
    all_h = [(radius, run) for radius, runs in runs_by_radius.items() for run in runs if run["orientation"] == "h"]
    all_v = [(radius, run) for radius, runs in runs_by_radius.items() for run in runs if run["orientation"] == "v"]
    if not all_h or not all_v:
        return None

    selected_runs: list[dict] = []
    selected_keys: set[tuple[int, str]] = set()

    def add_run(run: dict) -> None:
        key = (int(run["radius"]), str(run["side"]))
        if key not in selected_keys:
            selected_keys.add(key)
            selected_runs.append(run)

    add_run(random.choice(all_h)[1])
    add_run(random.choice(all_v)[1])

    for radius in radii:
        if any(int(run["radius"]) == int(radius) for run in selected_runs):
            continue
        add_run(random.choice(runs_by_radius[int(radius)]))

    extra_target = min(
        sum(len(runs) for runs in runs_by_radius.values()),
        max(len(selected_runs), len(radii) + random.choice((1, 1, 2))),
    )
    candidates = [run for runs in runs_by_radius.values() for run in runs if (int(run["radius"]), str(run["side"])) not in selected_keys]
    random.shuffle(candidates)
    for run in candidates:
        if len(selected_runs) >= extra_target:
            break
        add_run(run)

    segments = [_choose_fragment(run, center) for run in selected_runs]
    segments.sort(key=_segment_key)
    return segments


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    clipped_mode = bool(random.random() < 0.55)
    center = _sample_center(h, w, clipped_mode)
    if center is None:
        return None

    _feasible, full, clipped = _feasible_radii((h, w), center)
    radii = _pick_radii(full, clipped, clipped_mode)
    if radii is None:
        return None

    segments = _pick_segments((h, w), center, radii)
    if segments is None:
        return None

    ring_color = random.choice(RING_COLORS)
    inp = _render_input((h, w), segments, ring_color)

    try:
        extracted = _extract_segments(inp, ring_color)
    except AssertionError:
        return None
    if len(extracted) != len(segments):
        return None

    radii_tuple = tuple(sorted(int(radius) for radius in radii))
    if not _semantic_checks((h, w), segments, ring_color, center, radii_tuple):
        return None

    expected = _render_output((h, w), center, radii_tuple, ring_color)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-ring fragment puzzle")
