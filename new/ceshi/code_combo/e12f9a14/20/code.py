# Combo variant of e12f9a14 (index 20)
# combo_spec: D1/1+D6/1+D15/0
# Name: 反射弹跳+额外障碍物+碰前景停

import numpy as np

from arc_utils import get_bg, paste


OBSTACLE_COLOR = 5


# Design analysis for the fused variant:
# a) Concrete behavior and trace:
#    - Keep the original visible emitter object: a monochrome 2x2 block inside a shared-color 4x4 frame.
#    - Restrict generation to corner gaps so every launched beam is diagonal from the start, matching the
#      intended "reflecting diagonal ray" reading of the combo.
#    - The old task kept moving until boundary/collision and then applied special 45/90-degree beam-beam
#      redirection rules. Here that entire downstream collision system is removed. A beam now behaves like a
#      reflected laser on an empty checkerboard: when the next diagonal step would leave the grid, reflect
#      the corresponding sign(s) and continue from the same head cell.
#    - After that reflection decision, the beam checks occupancy. If the would-be next cell is any
#      foreground pixel in the current output, it stops immediately instead of colliding, turning, or merging.
#      This includes singleton obstacle pixels, frame walls, block interiors, and previously painted beam
#      cells. In the intended samples the decisive stop is an obstacle pixel.
#    - Example trace: suppose a NW corner gap launches a beam toward the top-left. In the base task it would
#      paint a short diagonal and die at the boundary. In this combo it bounces off the top and/or left edge,
#      continues on the mirrored diagonal, and later halts one cell before a gray singleton obstacle. The
#      visible difference is therefore "same diagonal color, but longer mirrored path that ends at a static
#      obstacle" rather than any beam-beam turning pattern.
# b) Complexity calibration:
#    - The combo is intentionally reduced to one gap per frame and corner gaps only. This keeps the reflection
#      rule legible and avoids reintroducing the original collision zoo through dense multi-beam interaction.
#    - Extra obstacles use one fixed reserved color (gray 5), distinct from background, frames, and blocks.
#      Human solvers can therefore spot the stopping targets directly.
#    - The generator requires every beam to reflect at least once and then stop on an obstacle. This prevents
#      degeneration back to the original "straight to edge" task and prevents samples where the obstacle color
#      is present but irrelevant.
# c) Implicit assumption audit:
#    - Original proposal logic assumed "out of bounds => beam dies". That is invalid now; proposal generation
#      must mirror direction first, then test the reflected next cell.
#    - Original `is_free()` only reasoned about background plus a small `blocked` set used by collision gaps.
#      That is invalid now; any non-background pixel in the evolving output is a hard stop.
#    - Original cell/cross/segment collision handlers depended on multi-beam meetings. They are removed
#      entirely because the new termination rule preempts those mechanics.
#    - Reflection makes self-reentry possible. The solver therefore intentionally treats a beam's own painted
#      trail as foreground, which stops the beam instead of allowing a loop.
# d) Placement legality:
#    - Frames still obey fixed-size 4x4 geometry, remain inside the grid, and keep a generous bbox gap.
#    - Obstacles are only placed on background cells that lie on a precomputed reflected path prefix for some
#      beam, are not adjacent to static foreground in the input, and are unique to that beam's chosen prefix.
#    - Candidate prefixes are chosen so different beams never share painted cells or obstacle targets. This
#      removes order-sensitive dynamic interference and keeps the synchronous solver deterministic.
# e) Visual observability:
#    - Each sample guarantees that every launched beam reflects at least once before stopping on a gray pixel.
#      Across examples the only stable explanation is "diagonal rays bounce on borders and stop before
#      entering any foreground, especially the gray obstacles."
#    - No hidden ordering by color id is used. Colors only bind a ray to its source block.
# f) Visual cleanliness:
#    - Sparse block counts, single-gap frames, larger spacing, and disjoint prefixes keep the output readable.
#    - The gray obstacles stay unchanged in the output, so there is no ambiguity about whether they are part
#      of the colored beam or part of the input scene.
# g) New solver assumptions enforced by the generator:
#    - Obstacles are isolated singletons, so `_find_blocks_2x2` never mistakes them for emitters.
#    - All structural gaps are corner gaps, so every emitted beam is diagonal.
#    - Selected beam prefixes are pairwise disjoint and avoid other obstacles, so the solver should observe
#      obstacle stops rather than contentions or trail/frame stops in generated samples.


_CORNER_GAPS = ((0, 0), (0, 3), (3, 3), (3, 0))


def _find_blocks_2x2(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    h, w = g.shape
    blocks: list[tuple[int, int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            v = int(g[r, c])
            if v == bg:
                continue
            if int(g[r, c + 1]) == v and int(g[r + 1, c]) == v and int(g[r + 1, c + 1]) == v:
                blocks.append((v, r, c))
    return blocks


def _infer_frame_color(g: np.ndarray, bg: int, br: int, bc: int, block_col: int) -> int:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    vals: list[int] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            v = int(g[r, c])
            if v != bg and v != block_col:
                vals.append(v)
    frame_colors = sorted(set(vals))
    assert len(frame_colors) == 1, "expected a unique frame color around each 2x2 block"
    return frame_colors[0]


def _frame_gaps_for_block(g: np.ndarray, bg: int, wall: int, br: int, bc: int) -> list[tuple[int, int]]:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    gaps: list[tuple[int, int]] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            if int(g[r, c]) == bg:
                gaps.append((r, c))

    out: list[tuple[int, int]] = []
    for r, c in gaps:
        dr = -1 if r == fr0 else (1 if r == fr1 else 0)
        dc = -1 if c == fc0 else (1 if c == fc1 else 0)
        assert dr != 0 or dc != 0, "expected candidate gap on frame border"
        if dr == 0:
            ok = (
                0 <= r - 1 < h
                and 0 <= r + 1 < h
                and int(g[r - 1, c]) == wall
                and int(g[r + 1, c]) == wall
            )
        elif dc == 0:
            ok = (
                0 <= c - 1 < w
                and 0 <= c + 1 < w
                and int(g[r, c - 1]) == wall
                and int(g[r, c + 1]) == wall
            )
        else:
            ok = (
                0 <= r - dr < h
                and 0 <= c - dc < w
                and int(g[r - dr, c]) == wall
                and int(g[r, c - dc]) == wall
            )
        if ok:
            out.append((r, c))
    return out


def _normalize_dir(dr: int, dc: int) -> tuple[int, int]:
    if dr == 0 and dc == 0:
        return (0, 0)
    if dr != 0 and dc != 0:
        return (int(np.sign(dr)), int(np.sign(dc)))
    if dr != 0:
        return (int(np.sign(dr)), 0)
    return (0, int(np.sign(dc)))


def _reflect_dir(d: tuple[int, int], flip_r: bool, flip_c: bool) -> tuple[int, int]:
    dr, dc = d
    if flip_r:
        dr = -dr
    if flip_c:
        dc = -dc
    return int(dr), int(dc)


def _gap_dir(gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = gap
    return _normalize_dir(-1 if gr == 0 else 1, -1 if gc == 0 else 1)


class _Beam:
    __slots__ = ("color", "pos", "direction", "active", "reflections")

    def __init__(self, color: int, pos: tuple[int, int], direction: tuple[int, int]):
        self.color = int(color)
        self.pos = pos
        self.direction = direction
        self.active = True
        self.reflections = 0


def _simulate(grid: np.ndarray, collect_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

    frame_colors = {_infer_frame_color(g, bg, br, bc, col) for col, br, bc in blocks}
    assert len(frame_colors) == 1, "expected a single shared frame color"
    frame_color = next(iter(frame_colors))
    block_colors = {col for col, _br, _bc in blocks}
    assert bg != OBSTACLE_COLOR and frame_color != OBSTACLE_COLOR and OBSTACLE_COLOR not in block_colors, (
        "gray(5) is reserved for singleton obstacles in this variant"
    )

    beams: dict[int, _Beam] = {}
    next_bid = 0
    obstacle_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == OBSTACLE_COLOR)}

    stop_reasons: dict[int, str] = {}

    for col, br, bc in blocks:
        gaps = _frame_gaps_for_block(g, bg, frame_color, br, bc)
        assert gaps, "expected each framed block to expose at least one structural gap"
        for gr, gc in gaps:
            d = _normalize_dir(
                -1 if gr == br - 1 else (1 if gr == br + 2 else 0),
                -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0),
            )
            assert d != (0, 0), "expected each gap to face outward"
            assert abs(d[0]) == 1 and abs(d[1]) == 1, "this combo expects diagonal launches only"
            assert int(out[gr, gc]) == bg, "expected a background gap cell"
            out[gr, gc] = int(col)
            beams[next_bid] = _Beam(int(col), (gr, gc), d)
            next_bid += 1

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    stats = {
        "beam_count": len(beams),
        "total_reflections": 0,
        "max_reflections_on_beam": 0,
        "obstacle_stop_beams": 0,
        "static_foreground_stop_beams": 0,
        "trail_stop_beams": 0,
        "contention_stop_beams": 0,
        "beams_with_reflection_before_stop": 0,
        "stop_reasons": {},
        "reflection_counts": {},
    }

    max_iters = h * w * 8
    for _ in range(max_iters):
        proposals: dict[tuple[int, int], list[tuple[int, tuple[int, int], bool]]] = {}
        any_active = False

        for bid, beam in beams.items():
            if not beam.active:
                continue
            any_active = True

            d = beam.direction
            nr, nc = beam.pos[0] + d[0], beam.pos[1] + d[1]
            reflected = False
            if not (0 <= nr < h and 0 <= nc < w):
                d = _reflect_dir(d, not (0 <= nr < h), not (0 <= nc < w))
                nr, nc = beam.pos[0] + d[0], beam.pos[1] + d[1]
                reflected = True
            nxt = (nr, nc)
            assert in_bounds(nxt), "reflected step must stay in bounds"

            if int(out[nr, nc]) != bg:
                if reflected:
                    beam.reflections += 1
                    beam.direction = d
                beam.active = False
                if nxt in obstacle_cells:
                    stop_reasons[bid] = "obstacle"
                    stats["obstacle_stop_beams"] += 1
                elif int(g[nr, nc]) != bg:
                    stop_reasons[bid] = "static_foreground"
                    stats["static_foreground_stop_beams"] += 1
                else:
                    stop_reasons[bid] = "trail"
                    stats["trail_stop_beams"] += 1
                if beam.reflections > 0:
                    stats["beams_with_reflection_before_stop"] += 1
                continue

            proposals.setdefault(nxt, []).append((bid, d, reflected))

        if not any_active or not proposals:
            break

        moved = False
        for cell, bidders in proposals.items():
            if len(bidders) > 1:
                for bid, d, reflected in bidders:
                    beam = beams[bid]
                    if reflected:
                        beam.reflections += 1
                        beam.direction = d
                    beam.active = False
                    stop_reasons[bid] = "contention"
                    stats["contention_stop_beams"] += 1
                    if beam.reflections > 0:
                        stats["beams_with_reflection_before_stop"] += 1
                continue

            bid, d, reflected = bidders[0]
            beam = beams[bid]
            if reflected:
                beam.reflections += 1
            beam.direction = d
            beam.pos = cell
            out[cell[0], cell[1]] = beam.color
            moved = True

        if not moved:
            break
    else:
        raise AssertionError("beam simulation did not converge")

    for bid, beam in beams.items():
        stats["reflection_counts"][bid] = beam.reflections
        stats["total_reflections"] += beam.reflections
        stats["max_reflections_on_beam"] = max(stats["max_reflections_on_beam"], beam.reflections)
        if bid not in stop_reasons:
            stop_reasons[bid] = "exhausted"
            if beam.reflections > 0:
                stats["beams_with_reflection_before_stop"] += 1
    stats["stop_reasons"] = dict(stop_reasons)

    if not collect_stats:
        return out
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit diagonal beams from framed 2x2 blocks, let them reflect off the grid boundary, and stop them as soon as any foreground pixel blocks the reflected path.

    Input structure:
    - The most common color is the background, and each source is a monochrome 2x2 block inside a 4x4 frame of one shared wall color.
    - Each active frame exposes one or more corner gaps, so every emitted beam starts as a diagonal ray.
    - Additional gray singleton pixels may appear in the background; these are inert obstacles, not emitters.

    Rule:
    - Every gap is recolored with its block color and launches a diagonal beam in the outward corner direction.
    - If the next step would leave the grid, the beam reflects on the touched border and continues with the mirrored diagonal direction from the same head cell.
    - After that reflection check, the beam stops immediately if its next cell is any non-background pixel in the current grid state, including gray obstacles, frame/block pixels, or previously painted ray cells. If several beams try to enter the same empty cell on the same step, none of them paints it.

    Key insight:
    - The informative structure is a bounced diagonal prefix ending right before visible foreground, especially the gray singleton obstacles: you must follow the mirrored path first, then apply a simple "foreground means stop" rule instead of the original beam-collision redirections.
    """
    return _simulate(grid, collect_stats=False)


# --------------- generator (combo) ---------------

BLOCK_COUNT_RANGE = (1, 4)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {1: 12, 2: 18, 3: 24}
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
FRAME_BBOX_GAP = 4
PLACEMENT_MARGIN = 2
MIN_PREFIX_LEN = 6
MIN_REFLECTIONS_BEFORE_STOP = 1
MIN_BEAM_PIXELS_PER_GAP = 5
GEN_RETRIES = 600


def _frames_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _all_positions(n: int) -> list[tuple[int, int]]:
    hi = n - 4 - PLACEMENT_MARGIN
    return [(r, c) for r in range(PLACEMENT_MARGIN, hi + 1) for c in range(PLACEMENT_MARGIN, hi + 1)]


def _choose_positions(n: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    positions = _all_positions(n)
    selected: list[tuple[int, int]] = []
    for _ in range(count):
        viable = [p for p in positions if all(_frames_separated(p, q) for q in selected)]
        if not viable:
            return None
        selected.append(viable[int(rng.integers(len(viable)))])
    return selected


def _draw_framed_block(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    block_color: int,
    gaps: tuple[tuple[int, int], ...],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    for gr, gc in gaps:
        patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c not in (frame_color, OBSTACLE_COLOR)]
    block_colors = list(map(int, rng.choice(palette, size=block_count, replace=False)))
    return bg, frame_color, block_colors


def _grid_size_min(block_count: int) -> int:
    result = 12
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _single_reflect_trace(
    grid: np.ndarray,
    bg: int,
    seed: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[int], str]:
    h, w = grid.shape
    pos = seed
    cur_d = direction
    path = [seed]
    entry_reflections = [0]
    reflections = 0

    for _ in range(h * w * 8):
        nr, nc = pos[0] + cur_d[0], pos[1] + cur_d[1]
        if not (0 <= nr < h and 0 <= nc < w):
            cur_d = _reflect_dir(cur_d, not (0 <= nr < h), not (0 <= nc < w))
            nr, nc = pos[0] + cur_d[0], pos[1] + cur_d[1]
            reflections += 1
        nxt = (nr, nc)
        if int(grid[nr, nc]) != bg:
            return path, entry_reflections, "static_foreground"
        if nxt in path:
            return path, entry_reflections, "trail"
        pos = nxt
        path.append(nxt)
        entry_reflections.append(reflections)
    raise AssertionError("single-beam reflection trace did not converge")


def _isolated_obstacle_site(grid: np.ndarray, bg: int, cell: tuple[int, int]) -> bool:
    h, w = grid.shape
    r, c = cell
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if int(grid[r, c]) != bg:
        return False
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if (rr, cc) == cell:
                continue
            if int(grid[rr, cc]) != bg:
                return False
    return True


def _candidate_options_for_block(
    base_grid: np.ndarray,
    bg: int,
    r0: int,
    c0: int,
) -> list[dict[str, object]]:
    options: list[dict[str, object]] = []
    for gap in _CORNER_GAPS:
        seed = (r0 + gap[0], c0 + gap[1])
        direction = _gap_dir(gap)
        path, entry_reflections, stop_reason = _single_reflect_trace(base_grid, bg, seed, direction)
        if stop_reason == "static_foreground" and len(path) < MIN_PREFIX_LEN + 1:
            continue
        for idx in range(1, len(path)):
            if len(path[:idx]) < MIN_PREFIX_LEN:
                continue
            if entry_reflections[idx] < MIN_REFLECTIONS_BEFORE_STOP:
                continue
            obstacle = path[idx]
            if not _isolated_obstacle_site(base_grid, bg, obstacle):
                continue
            options.append(
                {
                    "gap": gap,
                    "prefix": tuple(path[:idx]),
                    "obstacle": obstacle,
                    "reflections_before_stop": entry_reflections[idx],
                    "prefix_len": idx,
                }
            )
    options.sort(
        key=lambda opt: (
            int(opt["reflections_before_stop"]),
            int(opt["prefix_len"]),
        ),
        reverse=True,
    )
    return options


def _obstacles_far_enough(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1


def _choose_block_plans(
    options_per_block: list[list[dict[str, object]]],
    rng: np.random.Generator,
) -> list[dict[str, object]] | None:
    order = list(range(len(options_per_block)))
    rng.shuffle(order)
    shuffled = [[opts[i] for i in rng.permutation(len(opts))] for opts in options_per_block]

    chosen: list[dict[str, object] | None] = [None] * len(options_per_block)
    used_prefix: set[tuple[int, int]] = set()
    used_obstacles: list[tuple[int, int]] = []

    def dfs(k: int) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        for opt in shuffled[idx]:
            prefix = set(opt["prefix"])
            obstacle = tuple(opt["obstacle"])
            if prefix & used_prefix:
                continue
            if obstacle in used_prefix or obstacle in prefix:
                continue
            if obstacle in used_obstacles:
                continue
            if any(not _obstacles_far_enough(obstacle, other) for other in used_obstacles):
                continue

            chosen[idx] = opt
            used_prefix.update(prefix)
            used_obstacles.append(obstacle)
            if dfs(k + 1):
                return True
            used_obstacles.pop()
            used_prefix.difference_update(prefix)
            chosen[idx] = None
        return False

    if not dfs(0):
        return None
    return [opt for opt in chosen if opt is not None]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))
    bg, frame_color, block_colors = _pick_colors(rng, block_count)

    positions = _choose_positions(n, block_count, rng)
    if positions is None:
        return None

    base_grid = np.full((n, n), bg, dtype=int)
    for (r0, c0), color in zip(positions, block_colors):
        _draw_framed_block(base_grid, r0, c0, bg, frame_color, int(color), ())

    options_per_block = [_candidate_options_for_block(base_grid, bg, r0, c0) for r0, c0 in positions]
    if any(not opts for opts in options_per_block):
        return None

    plans = _choose_block_plans(options_per_block, rng)
    if plans is None:
        return None

    grid = np.full((n, n), bg, dtype=int)
    for (r0, c0), color, plan in zip(positions, block_colors, plans):
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), (tuple(plan["gap"]),))
        orow, ocol = map(int, plan["obstacle"])
        grid[orow, ocol] = OBSTACLE_COLOR

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out, stats = _simulate(grid, collect_stats=True)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < block_count * MIN_BEAM_PIXELS_PER_GAP:
        return None
    if stats["beam_count"] != block_count:
        return None
    if stats["obstacle_stop_beams"] != block_count:
        return None
    if stats["static_foreground_stop_beams"] != 0:
        return None
    if stats["trail_stop_beams"] != 0:
        return None
    if stats["contention_stop_beams"] != 0:
        return None
    if stats["beams_with_reflection_before_stop"] != block_count:
        return None
    if stats["max_reflections_on_beam"] < MIN_REFLECTIONS_BEFORE_STOP:
        return None
    if stats["total_reflections"] < block_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
