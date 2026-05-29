# Variant of 28a6681f
# Dimension: D9 — 轨迹涂色 (trajectory)
# Change: 无轨迹——水像素从源移动到最终位置，中间路径不留痕迹，源位置恢复为背景色... → 水流路径轨迹
#
# Design analysis:
# variant:
#   core_change:
#     original_output: only the final settled water cells remain blue; any vacated source/path cells return to bg.
#     new_output: every moved water particle leaves a persistent trail in a dedicated color (fixed to 5 here).
#     invariant: the physical water motion is unchanged; only the visualization layer differs.
#   concrete_visualization:
#     grid_size: keep the original 10x10 regime.
#     colors: bg=0, water=1, trail=5, solids sampled from {2,3,4,6,7,8,9}.
#     source_to_sink_view:
#       - edge-reservoir cases show a vertical or ragged source sheet on one side and a gray path descending into bottom basins.
#       - guide cases show a small supported top-edge run that follows a staircase/ramp into the basin area, with the traversed cells preserved as trail.
#   behavior_trace_example:
#     original:
#       - remove one water pixel from its source cell.
#       - let it fall / slide until it settles.
#       - write blue only at the final resting cell; the source cell becomes 0 again.
#     variant:
#       - run the same motion step-by-step.
#       - record every visited cell in that particle path.
#       - after the whole simulation converges, paint all visited non-final cells with trail color 5 and keep only final resting cells blue.
#     difference: the solver now needs a hidden "physics grid" plus a separate trail mask; writing trail directly into the grid mid-simulation would incorrectly block later water.
#   complexity_calibration:
#     - Trail color is global and fixed rather than per-source, so humans can infer one simple rule from 2-4 examples.
#     - Generator still uses the original basin vocabulary: adjacent lower basins, occasional top guide, supported source water, and frequent overflow-like spreading.
#     - Trails are forced to be visible (minimum trail length check) so the variant does not collapse back to the original task.
#   implicit_assumption_review:
#     - Input checking from the original solver still applies to blue source water; every input water cell must be supported and cannot have bg on both sides.
#     - The old code freely wrote bg into the source cell before settling; under D9 that cell often belongs in the output trail, so source erasure must become a trail-side effect, not a final visual decision.
#     - The original settle() only returned the endpoint. That is insufficient now: it must return the full path.
#     - Final stability checks must only inspect blue water cells, not trail cells, because trail is descriptive rather than physical.
#     - A new assumption is introduced: the trail color must not already appear in the input as a solid color, otherwise output semantics become ambiguous. The generator enforces this and main() asserts it.
#   placement_legality:
#     - All generated solids are drawn inside the 10x10 frame and source water occupies only bg cells.
#     - Edge-source stacks are contiguous and terminate on solid support, so input validity assertions hold.
#     - Guide samples use a 4-connected staircase entering from an edge, matching the original guide constraints.
#   visual_observability:
#     - The output visibly differs from the original even when final blue water occupies the same basin cells, because a gray canal remains from source to sink.
#     - Mirror choices, multiple basin profiles, and both guide / no-guide families provide repeated evidence that gray means traversed path rather than "another object".
#   cleanliness_and_color_inference:
#     - Only one new semantic color is introduced.
#     - Trail never overlaps final blue water in the rendered output; blue wins at endpoints, keeping the scene readable.

from __future__ import annotations

import random

import numpy as np

BG = 0
WATER = 1
TRAIL = 5
STRUCTURE_PALETTE = (2, 3, 4, 6, 7, 8, 9)
H = 10
W = 10

EDGE_PROFILES: tuple[tuple[int, ...], ...] = (
    (0, 1, 1, 1, 2, 3, 2, 4, 4, 4),
    (0, 0, 1, 2, 3, 1, 2, 3, 4, 5),
    (0, 1, 2, 2, 3, 1, 2, 4, 4, 5),
    (0, 2, 2, 1, 3, 1, 2, 4, 5, 5),
)


def _component_count(points: set[tuple[int, int]], use_diagonal: bool) -> int:
    remaining = set(points)
    if use_diagonal:
        neighbors = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                nxt = (int(r + dr), int(c + dc))
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
    return count


def _simulate_water(
    grid: np.ndarray,
    *,
    track_trail: bool,
    check_input: bool,
    check_output: bool,
) -> tuple[np.ndarray, np.ndarray]:
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape
    trail_mask = np.zeros((h, w), dtype=bool)

    if check_input:
        assert not np.any(g == int(TRAIL)), "input must not already contain the trail color"
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == h - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
            right_bg = cc + 1 < w and int(g[rr, cc + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    bottom_colors = set(map(int, g[h - 1]))
    for color in sorted(set(map(int, g[:7].flatten())) - {BG, WATER} - bottom_colors):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, w - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"

    def settle(r: int, c: int) -> tuple[int, int, list[tuple[int, int]]]:
        rr = int(r)
        cc = int(c)
        path = [(int(rr), int(cc))]
        for _ in range(h * w):
            if rr + 1 < h and int(g[rr + 1, cc]) == int(BG):
                rr += 1
                path.append((int(rr), int(cc)))
                continue

            opts = []
            if (
                rr + 1 < h
                and cc - 1 >= 0
                and int(g[rr, cc - 1]) == int(BG)
                and int(g[rr + 1, cc - 1]) == int(BG)
            ):
                opts.append((int(rr + 1), int(cc - 1)))
            if (
                rr + 1 < h
                and cc + 1 < w
                and int(g[rr, cc + 1]) == int(BG)
                and int(g[rr + 1, cc + 1]) == int(BG)
            ):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
                rr, cc = opts[0]
                path.append((int(rr), int(cc)))
                continue

            if rr + 1 < h and int(g[rr + 1, cc]) != int(BG):
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == int(BG):
                    left -= 1
                right = int(cc)
                while right + 1 < w and int(g[rr, right + 1]) == int(BG):
                    right += 1

                drops = [int(x) for x in range(int(left), int(right) + 1) if int(g[rr + 1, int(x)]) == int(BG)]
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
                best = nearest[0]
                assert best != int(cc)
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                assert int(g[rr, nx]) == int(BG)
                cc = nx
                path.append((int(rr), int(cc)))
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")
        return int(rr), int(cc), path

    particles = [(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))]

    for _pass in range(h * w):
        moved = False
        order = sorted(range(len(particles)), key=lambda idx: particles[idx], reverse=True)
        for idx in order:
            r, c = particles[idx]
            if int(g[r, c]) != int(WATER):
                continue
            g[r, c] = int(BG)
            rr, cc, path = settle(int(r), int(c))
            particles[idx] = (int(rr), int(cc))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[rr, cc] = int(WATER)
            if track_trail and (int(rr), int(cc)) != (int(r), int(c)):
                for pr, pc in path[:-1]:
                    trail_mask[int(pr), int(pc)] = True
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert cc == 0 or int(g[rr, cc - 1]) != int(BG)
            assert rr == h - 1 or int(g[rr + 1, cc]) != int(BG)
            assert cc == w - 1 or int(g[rr, cc + 1]) != int(BG)

    trail_mask &= g != int(WATER)
    return g, trail_mask


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate basin-seeking water while preserving the traveled canal as a trail.

    Input structure:
    - bg = 0, source water = 1, trail color is reserved and absent from the input.
    - Other colors form fixed basin walls, steps, and optional edge-entering guides.
    - Water begins in supported source reservoirs or short guide-adjacent runs above the basin system.

    Rule:
    - Water motion follows the original local physics: fall straight down through bg, use a unique open diagonal when blocked, and slide horizontally along a supported shelf toward a unique drop.
    - The physical simulation ignores trail entirely; trail is only a visualization of visited cells.
    - After convergence, every moved particle paints all non-final cells on its path with the trail color, while its resting cell stays blue.

    Key insight: The destination logic is unchanged; the new information is the complete source-to-basin route, so the solver must track trajectories separately from occupancy.
    """
    final_water, trail_mask = _simulate_water(
        grid,
        track_trail=True,
        check_input=check_input,
        check_output=check_output,
    )
    out = np.array(final_water, dtype=int).copy()
    out[trail_mask] = int(TRAIL)
    return out


def _mirror_grid(grid: np.ndarray) -> np.ndarray:
    return np.flip(grid, axis=1).copy()


def _paint_profile(profile: tuple[int, ...], colors: tuple[int, int, int]) -> np.ndarray:
    g = np.zeros((H, W), dtype=int)
    cuts = (4, 7)
    for c, height in enumerate(profile):
        if height <= 0:
            continue
        color = colors[0] if c < cuts[0] else colors[1] if c < cuts[1] else colors[2]
        g[H - height :, c] = int(color)
    return g


def _add_right_source(grid: np.ndarray, profile: tuple[int, ...], a: int, b: int) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    top0 = H - int(profile[9])
    top1 = H - int(profile[8])
    assert 1 <= a <= top0
    assert 0 <= b <= min(a, top1)
    g[top0 - a : top0, 9] = int(WATER)
    if b:
        g[top1 - b : top1, 8] = int(WATER)
    return g


def _edge_case(rng: random.Random) -> np.ndarray:
    profile = rng.choice(EDGE_PROFILES)
    colors = tuple(rng.sample(STRUCTURE_PALETTE, 3))
    grid = _paint_profile(profile, colors)

    top0 = H - int(profile[9])
    top1 = H - int(profile[8])
    candidates = []
    for a in range(2, top0 + 1):
        for b in range(0, min(a, top1) + 1):
            if a + b < 4:
                continue
            if a + b > 12:
                continue
            candidates.append((a, b))
    a, b = rng.choice(candidates)
    grid = _add_right_source(grid, profile, a, b)

    if rng.random() < 0.5:
        grid = _mirror_grid(grid)
    return grid


GUIDE_BASE_COORDS = (
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 2),
    (2, 3),
    (3, 3),
    (3, 4),
    (4, 4),
    (4, 5),
    (5, 5),
)

BASIN_A_COORDS = (
    (9, 0),
    (9, 1),
    (9, 2),
    (8, 0),
    (7, 0),
)

BASIN_B_COORDS = (
    (9, 3),
    (9, 4),
    (9, 5),
    (9, 6),
    (8, 3),
    (7, 3),
    (6, 4),
    (8, 5),
    (7, 5),
    (8, 6),
    (7, 6),
)


def _guide_case(rng: random.Random) -> np.ndarray:
    guide_color, basin_color_a, basin_color_b = rng.sample(STRUCTURE_PALETTE, 3)
    g = np.zeros((H, W), dtype=int)
    for r, c in GUIDE_BASE_COORDS:
        g[r, c] = int(guide_color)
    for r, c in BASIN_A_COORDS:
        g[r, c] = int(basin_color_a)
    for r, c in BASIN_B_COORDS:
        g[r, c] = int(basin_color_b)

    run_len = rng.choice((3, 4))
    g[0, :run_len] = int(WATER)
    if rng.random() < 0.5:
        g = _mirror_grid(g)
    return g


def _trail_count(grid: np.ndarray) -> int:
    return int(np.count_nonzero(grid == int(TRAIL)))


def _blue_count(grid: np.ndarray) -> int:
    return int(np.count_nonzero(grid == int(WATER)))


def _spill_span(grid: np.ndarray) -> int:
    cols = np.where(np.any(grid == int(WATER), axis=0))[0]
    return 0 if len(cols) == 0 else int(cols[-1] - cols[0] + 1)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    builder = rng.choices((_edge_case, _guide_case), weights=(3, 2))[0]
    grid = builder(rng)

    try:
        out = main(grid)
    except Exception:
        return None

    if _trail_count(out) < 3:
        return None
    if _blue_count(out) < 2:
        return None

    # Keep a mix of compact settle and wider overflow-like spread.
    if builder is _edge_case and _spill_span(out) < 4:
        return None

    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the D9 trajectory-trail variant."""
    rng = random.Random()
    for _ in range(400):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
