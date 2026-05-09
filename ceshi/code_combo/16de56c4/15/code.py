# Combo variant of 16de56c4 (index 15)
# combo_spec: D1/2+D12/2
# Name: 递增步长+反射传播

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
import random
import sys

import numpy as np

try:
    from arc_utils import get_bg
except ImportError:
    sys.path.append(str(Path("/home/lixinyu/snap/new/solvers")))
    from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace
#    The source task used one constant beat per active line and filled it in one
#    shot.  D1/2 removes that constant beat: the two marker dots now give only
#    the first hop length `gap`, and the ray then lands at distances
#    `gap, gap+1, gap+2, ...` from its previous hit.  D12/2 removes the one-shot
#    fill: the ray becomes a stateful traveler that can bounce and keep going.
#    The fused rule is therefore a single simulation, not two independent
#    post-processes.
#    Example on a left-firing row of length 12 with marker color 1 at positions
#    0 and 1 (`gap = 1`) and companions at positions 3 (color 2) and 6 (color 3):
#      original task: 0,1,2,3,4,... on a fixed beat; the first aligned companion
#      would simply cap the fill at 3.
#      D1-only: outward accelerating prefix 0 -> 1 -> 3 -> 6 -> 10, then stop.
#      fused combo:
#        step1: 0 -> 1  (paint 1)
#        step2: 1 -> 3  (hit companion 2, paint 2, reverse)
#        step3: 3 -> 0  (paint 2 on the source marker)
#        step4: 0 -> 4  (boundary-reflected continuation, still growing step)
#        step5: 4 -> 9
#        step6: 9 -> 7
#        ...
#        later: 5 -> 6  (hit companion 3, reverse again, paint later hits with 3)
#    The visible difference is that cells such as 4, 7, 9 are impossible under a
#    simple outward accelerating prefix but appear once reflection and continued
#    step growth are both active.
# b) Complexity calibration
#    To keep the puzzle human-readable, each active line still has exactly one
#    repeated marker color touching the shooting edge.  Extra dots remain sparse:
#    at most two reflective companions plus at most one bystander.  That is
#    enough to show both boundary bounces and companion-triggered reversals
#    without turning a row/column into noise.
# c) Hidden assumptions that break
#    - The original modular test `p % step == offset` is invalid because the path
#      is no longer periodic.
#    - The original span fill `[lo, hi]` is invalid because a reflected growing
#      walk can revisit old cells and reach new cells outside the outward prefix.
#    - The original "first aligned singleton ends the line" assumption is invalid
#      because several companions may exist and only their first hit matters.
#    - A new solver assumption appears: non-marker dots are singleton colors, and
#      any singleton that lies on the simulated walk is a reflective companion.
#      The generator must therefore place bystanders strictly off the actual walk.
# d) Placement legality
#    The generator always builds each line from the shooting edge inward, so the
#    repeated marker color occupies the source cell and the first accelerated hit.
#    Companion candidates are searched by simulating the same state machine used
#    by the solver and keeping only positions that are actually reachable.
#    Bystanders are chosen from interior cells outside the simulated orbit, so
#    they cannot be accidentally promoted into companions.
# e) Visual observability
#    Human-solvable examples need evidence that reflection matters.  The generator
#    therefore enforces at least one companion-free line (showing wall bounce
#    alone) and at least one multi-companion line.  It also requires lines whose
#    visited cells extend beyond the simple outward accelerating prefix, which
#    makes the combo visibly different from D1-only samples.
# f) Visual cleanliness
#    All activity on a line is still point-like: the output paints discrete hit
#    cells only, never thick segments.  Distinct active lines are separated by at
#    least one empty line, preserving the sparse visual grammar of the source
#    task.
# g) Color inference
#    Colors keep structural roles instead of numeric roles.  The repeated color is
#    the source marker; each companion, when first hit, donates its own color to
#    subsequent newly paintable hits; bystanders are untouched.  No ordering by
#    color id is used anywhere.
# h) Fusion metric used for generator validation
#    For each active line, `extension_positions` is the set of cells first reached
#    only after the ray has already touched a boundary at least once, minus the
#    simple outward accelerating prefix.  Non-empty `extension_positions` prove
#    that "continued growth after reflection" is doing real work instead of
#    collapsing back to a one-way accelerating fill.
# i) New solver assumptions mirrored in generation
#    The solver expects exactly one repeated marker color per active line, that
#    marker to include the shooting-edge cell, and every extra color on that line
#    to be a singleton.  The generator enforces all three conditions and rejects
#    any candidate line whose companions are not actually consumed by the fused
#    simulation.


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_GAP_RANGE = (1, 4)
_MAX_COMPANIONS = 2
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_P_BYSTANDER = 0.35
_MAX_RETRIES = 500


def _edge_params(edge: str, line_len: int) -> tuple[int, int]:
    """Return the source position and the initial inward direction."""
    if edge in ("left", "top"):
        return 0, 1
    return line_len - 1, -1


def _detect_edge(g: np.ndarray, bg: int) -> str:
    """Find the unique boundary with the most non-background dots."""
    H, W = g.shape
    edge_counts = {
        "top": sum(1 for c in range(W) if g[0, c] != bg),
        "bottom": sum(1 for c in range(W) if g[H - 1, c] != bg),
        "left": sum(1 for r in range(H) if g[r, 0] != bg),
        "right": sum(1 for r in range(H) if g[r, W - 1] != bg),
    }
    max_count = max(edge_counts.values())
    assert max_count > 0, "No non-bg dots on any edge"
    winners = [edge for edge, count in edge_counts.items() if count == max_count]
    assert len(winners) == 1, f"Ambiguous shooting edge: {winners} (counts={edge_counts})"
    return winners[0]


def _outward_prefix_positions(edge: str, line_len: int, gap: int) -> list[int]:
    """Enumerate the one-way accelerating path before any reflection is allowed."""
    assert gap >= 1
    pos, direction = _edge_params(edge, line_len)
    path = [pos]
    step = gap
    while True:
        nxt = pos + direction * step
        if not (0 <= nxt < line_len):
            break
        path.append(int(nxt))
        pos = int(nxt)
        step += 1
    return path


def _advance_reflect(pos: int, direction: int, distance: int, line_len: int) -> tuple[int, int, int]:
    """Move one accelerating step on a 1D segment, reflecting off both ends."""
    assert direction in (-1, 1)
    assert distance >= 1
    if line_len == 1:
        return 0, 1, 0

    span = line_len - 1
    period = 2 * span

    # Unfold the segment into a back-and-forth track where each step advances
    # forward by `distance`; folding it back yields the reflected landing cell.
    unfolded = pos if direction == 1 else period - pos
    total = unfolded + distance
    boundary_hits = (total // span) - (unfolded // span)
    folded = total % period

    if folded == 0:
        return 0, 1, int(boundary_hits)
    if folded == span:
        return span, -1, int(boundary_hits)
    if folded < span:
        return int(folded), 1, int(boundary_hits)
    return int(period - folded), -1, int(boundary_hits)


def _line_profile(
    g: np.ndarray,
    line: int,
    *,
    by_row: bool,
    edge: str,
    bg: int,
) -> dict | None:
    """Parse one shooting-direction line into marker and singleton dots."""
    line_len = g.shape[1] if by_row else g.shape[0]

    def cell(pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    colors: dict[int, list[int]] = defaultdict(list)
    for p in range(line_len):
        v = int(g[cell(p)])
        if v != bg:
            colors[v].append(p)

    if not colors:
        return None

    repeating = {c: ps for c, ps in colors.items() if len(ps) >= 2}
    assert len(repeating) == 1, (
        f"Line {line} ({'row' if by_row else 'col'}) needs exactly one repeating "
        f"marker color: {dict(colors)}"
    )
    marker, mpos = next(iter(repeating.items()))
    assert len(mpos) == 2, (
        f"Line {line} ({'row' if by_row else 'col'}) marker {marker} has "
        f"{len(mpos)} dots (expected 2): {mpos}"
    )

    edge_pos, _direction = _edge_params(edge, line_len)
    assert edge_pos in mpos, (
        f"Line {line} ({'row' if by_row else 'col'}) marker {marker} must touch "
        f"the shooting edge at {edge_pos}: {mpos}"
    )

    gap = abs(mpos[0] - mpos[1])
    prefix = _outward_prefix_positions(edge, line_len, gap)
    assert len(prefix) >= 2, f"Line {line} gap {gap} yields no interior marker"
    assert prefix[1] in mpos, (
        f"Line {line} ({'row' if by_row else 'col'}) marker {marker} must place "
        f"its second dot on the first accelerated landing: {mpos} vs {prefix[:2]}"
    )

    singletons: dict[int, int] = {}
    for col, ps in colors.items():
        if col == marker:
            continue
        assert len(ps) == 1, (
            f"Line {line} ({'row' if by_row else 'col'}) non-marker color {col} "
            f"must be a singleton: {ps}"
        )
        pos = int(ps[0])
        assert pos not in singletons, f"Duplicate singleton position on line {line}: {pos}"
        singletons[pos] = int(col)

    return {
        "marker": int(marker),
        "gap": int(gap),
        "prefix": prefix,
        "singletons": singletons,
    }


def _simulate_line_trace(
    marker: int,
    gap: int,
    line_len: int,
    edge: str,
    singletons: dict[int, int],
) -> dict:
    """Run the fused accelerating-reflection process for one line."""
    edge_pos, direction = _edge_params(edge, line_len)
    prefix = _outward_prefix_positions(edge, line_len, gap)
    prefix_set = set(prefix)

    pos = edge_pos
    step_len = gap
    cur_color = int(marker)
    unused = {int(p): int(c) for p, c in singletons.items()}
    seen: set[tuple] = set()

    visited_positions = [int(edge_pos)]
    first_visit_order = [int(edge_pos)]
    distinct_positions = {int(edge_pos)}
    extension_positions: set[int] = set()
    companion_hit_positions: list[int] = []
    paint_events: list[tuple[int, int, bool]] = []
    boundary_bounce_steps = 0
    total_boundary_hits = 0
    had_boundary = False

    while True:
        period = max(1, 2 * (line_len - 1))
        state = (
            int(pos),
            int(direction),
            int(step_len % period),
            int(cur_color),
            tuple(sorted(unused.items())),
        )
        if state in seen:
            break
        seen.add(state)

        next_pos, next_direction, boundary_hits = _advance_reflect(pos, direction, step_len, line_len)
        pos = int(next_pos)
        direction = int(next_direction)

        visited_positions.append(pos)
        if pos not in distinct_positions:
            distinct_positions.add(pos)
            first_visit_order.append(pos)

        if boundary_hits > 0:
            boundary_bounce_steps += 1
            total_boundary_hits += int(boundary_hits)

        after_boundary = had_boundary or boundary_hits > 0
        if after_boundary and pos not in prefix_set:
            extension_positions.add(pos)
        had_boundary = after_boundary

        hit_companion = pos in unused
        if hit_companion:
            cur_color = int(unused.pop(pos))
            direction *= -1
            companion_hit_positions.append(pos)
            paint_events.append((pos, cur_color, True))
        else:
            paint_events.append((pos, cur_color, False))

        step_len += 1

    return {
        "prefix": prefix,
        "visited_positions": visited_positions,
        "first_visit_order": first_visit_order,
        "distinct_positions": distinct_positions,
        "extension_positions": extension_positions,
        "companion_hit_positions": companion_hit_positions,
        "used_companions": set(companion_hit_positions),
        "unused_companions": set(unused),
        "boundary_bounce_steps": int(boundary_bounce_steps),
        "total_boundary_hits": int(total_boundary_hits),
        "paint_events": paint_events,
    }


def _simulate_line_into_output(
    out: np.ndarray,
    line: int,
    profile: dict,
    *,
    by_row: bool,
    edge: str,
    bg: int,
) -> dict:
    """Apply one parsed line to the output grid and return its trace."""
    line_len = out.shape[1] if by_row else out.shape[0]
    marker = int(profile["marker"])
    trace = _simulate_line_trace(marker, int(profile["gap"]), line_len, edge, profile["singletons"])

    def cell(pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    for pos, color, hit_companion in trace["paint_events"]:
        rc = cell(int(pos))
        if hit_companion:
            out[rc] = int(color)
            continue
        cur = int(out[rc])
        if cur == bg or cur == marker:
            out[rc] = int(color)

    return trace


def _choose_dims(edge: str, n_active: int) -> tuple[int, int, int, int] | None:
    """Pick dimensions compatible with the chosen edge and active-line count."""
    min_num_lines = 2 * n_active - 1
    by_row = edge in ("left", "right")
    if by_row:
        lo = max(_ROWS_RANGE[0], min_num_lines)
        if lo > _ROWS_RANGE[1]:
            return None
        H = random.randint(lo, _ROWS_RANGE[1])
        W = random.randint(*_COLS_RANGE)
        return H, W, H, W
    lo = max(_COLS_RANGE[0], min_num_lines)
    if lo > _COLS_RANGE[1]:
        return None
    W = random.randint(lo, _COLS_RANGE[1])
    H = random.randint(*_ROWS_RANGE)
    return H, W, W, H


def _spaced_indices(n: int, total: int) -> list[int]:
    """Choose n line indices with at least one blank line between neighbors."""
    available = total - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [picks[i] + i for i in range(n)]


def _count_other_edge_dots(
    dots_rc: list[tuple[int, int]],
    edge: str,
    H: int,
    W: int,
) -> dict[str, int]:
    """Count how many placed dots land on the non-shooting boundaries."""
    boundary = {
        "top": lambda r, c: r == 0,
        "bottom": lambda r, c: r == H - 1,
        "left": lambda r, c: c == 0,
        "right": lambda r, c: c == W - 1,
    }
    counts = {}
    for other in _EDGES:
        if other == edge:
            continue
        check = boundary[other]
        counts[other] = sum(1 for r, c in dots_rc if check(r, c))
    return counts


def _candidate_gaps(line_len: int) -> list[int]:
    """Prefer gaps whose outward accelerating prefix is visibly long."""
    gaps: list[int] = []
    max_gap = min(_GAP_RANGE[1], line_len - 1)
    for min_len in (5, 4, 3, 2):
        cur = []
        for gap in range(_GAP_RANGE[0], max_gap + 1):
            prefix = _outward_prefix_positions("left", line_len, gap)
            if len(prefix) >= min_len:
                cur.append(gap)
        if cur:
            gaps = cur
            break
    random.shuffle(gaps)
    return gaps


def _plan_line(
    edge: str,
    line_len: int,
    palette: list[int],
    n_companions: int,
) -> dict | None:
    """Construct one active line that really exercises the fused dynamics."""
    assert 0 <= n_companions <= _MAX_COMPANIONS

    marker_candidates = palette[:]
    random.shuffle(marker_candidates)

    for marker_color in marker_candidates:
        gaps = _candidate_gaps(line_len)
        for gap in gaps:
            prefix = _outward_prefix_positions(edge, line_len, gap)
            if len(prefix) < 2:
                continue

            base_trace = _simulate_line_trace(marker_color, gap, line_len, edge, {})
            candidate_positions = [
                p for p in base_trace["first_visit_order"]
                if p not in prefix[:2] and p not in {0, line_len - 1}
            ]
            candidate_positions = list(dict.fromkeys(candidate_positions))[:8]
            if len(candidate_positions) < n_companions:
                continue

            free_colors = [c for c in palette if c != marker_color]
            if len(free_colors) < n_companions:
                continue

            combos = list(combinations(candidate_positions, n_companions))
            random.shuffle(combos)
            if n_companions == 0:
                combos = [()]

            for combo in combos[:40]:
                companion_colors = random.sample(free_colors, n_companions)
                companions = {
                    int(pos): int(color) for pos, color in zip(combo, companion_colors)
                }
                trace = _simulate_line_trace(marker_color, gap, line_len, edge, companions)
                if trace["unused_companions"]:
                    continue
                if len(trace["used_companions"]) != n_companions:
                    continue
                if trace["boundary_bounce_steps"] == 0:
                    continue
                if not trace["extension_positions"]:
                    continue

                used_colors = {marker_color, *companions.values()}
                bystander = None
                if random.random() < _P_BYSTANDER:
                    bystander_colors = [c for c in palette if c not in used_colors]
                    bystander_positions = [
                        p
                        for p in range(1, line_len - 1)
                        if p not in trace["distinct_positions"] and p not in companions
                    ]
                    if bystander_colors and bystander_positions:
                        bystander = (
                            random.choice(bystander_positions),
                            random.choice(bystander_colors),
                        )

                edge_pos, _direction = _edge_params(edge, line_len)
                dots = [(edge_pos, marker_color), (prefix[1], marker_color)]
                dots.extend((int(pos), int(color)) for pos, color in companions.items())
                if bystander is not None:
                    dots.append((int(bystander[0]), int(bystander[1])))

                return {
                    "dots": dots,
                    "marker": int(marker_color),
                    "gap": int(gap),
                    "n_companions": int(n_companions),
                    "trace": trace,
                }

    return None


def _fusion_signature(grid: np.ndarray) -> dict[str, int]:
    """Measure whether a generated sample really exhibits the combo behavior."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    edge = _detect_edge(g, bg)
    by_row = edge in ("left", "right")
    num_lines = g.shape[0] if by_row else g.shape[1]

    active_lines = 0
    companion_reflection_lines = 0
    multi_companion_lines = 0
    boundary_extension_lines = 0
    total_extension_positions = 0
    total_boundary_bounces = 0

    for li in range(num_lines):
        profile = _line_profile(g, li, by_row=by_row, edge=edge, bg=bg)
        if profile is None:
            continue
        active_lines += 1
        line_len = g.shape[1] if by_row else g.shape[0]
        trace = _simulate_line_trace(
            int(profile["marker"]),
            int(profile["gap"]),
            line_len,
            edge,
            profile["singletons"],
        )
        n_used = len(trace["used_companions"])
        if n_used > 0:
            companion_reflection_lines += 1
        if n_used >= 2:
            multi_companion_lines += 1
        if trace["extension_positions"]:
            boundary_extension_lines += 1
            total_extension_positions += len(trace["extension_positions"])
        total_boundary_bounces += int(trace["boundary_bounce_steps"])

    return {
        "active_lines": active_lines,
        "companion_reflection_lines": companion_reflection_lines,
        "multi_companion_lines": multi_companion_lines,
        "boundary_extension_lines": boundary_extension_lines,
        "total_extension_positions": total_extension_positions,
        "total_boundary_bounces": total_boundary_bounces,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate accelerating rays from one dominant edge with reflective rebounds.

    Input structure:
    - The background is the most frequent color and all non-background cells are
      sparse dots.
    - Exactly one boundary has strictly more non-background dots than the other
      three; that boundary identifies the shooting edge.
    - Every active row/column on that shooting axis has exactly one repeated
      marker color with exactly two dots: one on the shooting edge and one on
      the first inward landing of the ray.  Other colors on that line are
      singletons.
    - A singleton can be either a reflective companion if the simulated ray
      lands on it, or a bystander if the ray never reaches it.

    Rule:
    - Determine the unique shooting edge.  Left/right edges launch row-wise
      rays; top/bottom edges launch column-wise rays.
    - For each active line, the repeated marker pair gives the initial hop
      length `gap`.  Starting from the shooting-edge marker, move by lengths
      `gap, gap+1, gap+2, ...`.
    - When a hop would leave the line, reflect off the boundary and finish that
      same hop on the mirrored side; later hops keep increasing in length.
    - The first time the ray lands on a singleton dot, that dot is a companion:
      it keeps its own color, flips the direction for the next hop, and its
      color becomes the active paint color.  Used companions are no longer
      reflective.  Singleton dots never reached by the walk are bystanders and
      remain unchanged.
    - Only background cells and original marker cells are paintable during
      travel.  The process stops once the full simulation state repeats, because
      the future motion would then cycle forever without new information.

    Key insight:
    The repeated pair no longer defines a modular beat or a one-way prefix.  It
    defines only the first jump of a reflected accelerating walk, so solving the
    puzzle requires simulating a finite state machine over each line rather than
    applying residue classes or bounded spans.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    edge = _detect_edge(g, bg)
    by_row = edge in ("left", "right")
    num_lines = g.shape[0] if by_row else g.shape[1]

    for li in range(num_lines):
        profile = _line_profile(g, li, by_row=by_row, edge=edge, bg=bg)
        if profile is None:
            continue
        _simulate_line_into_output(out, li, profile, by_row=by_row, edge=edge, bg=bg)

    return out


def _gen_attempt() -> np.ndarray | None:
    """Build one candidate input and reject it unless the fusion is visible."""
    n_active = random.randint(*_ACTIVE_LINES_RANGE)
    edge = random.choice(_EDGES)
    by_row = edge in ("left", "right")

    dims = _choose_dims(edge, n_active)
    if dims is None:
        return None
    H, W, num_lines, line_len = dims

    active_lines = _spaced_indices(n_active, num_lines)
    if not active_lines:
        return None

    palette_size = random.randint(5, min(7, len(_COLORS)))
    palette = random.sample(_COLORS, palette_size)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    shooting_edge_dots = 0

    planned_companions = [0, 2]
    while len(planned_companions) < n_active:
        planned_companions.append(random.choices([1, 2, 0], weights=[4, 2, 2])[0])
    random.shuffle(planned_companions)

    for li, n_companions in zip(active_lines, planned_companions):
        line_plan = _plan_line(edge, line_len, palette, n_companions)
        if line_plan is None:
            return None

        for pos, color in line_plan["dots"]:
            r, c = cell(li, int(pos))
            if inp[r, c] != 0:
                return None
            inp[r, c] = int(color)
            all_rc.append((r, c))

        shooting_edge_dots += 1

    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
        return None

    try:
        out = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None

    sig = _fusion_signature(inp)
    if sig["active_lines"] != n_active:
        return None
    if sig["companion_reflection_lines"] == 0:
        return None
    if sig["multi_companion_lines"] == 0:
        return None
    if sig["boundary_extension_lines"] < max(2, n_active // 2):
        return None
    if sig["total_extension_positions"] < max(4, n_active):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
