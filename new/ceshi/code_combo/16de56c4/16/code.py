# Combo variant of 16de56c4 (index 16)
# combo_spec: D5/0+D12/2+D17/0
# Name: 碰撞停止+反射+旁观者障碍

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg

# Design analysis for the combo variant
# ------------------------------------
# a) Concrete behavior and trace
#    - The original task had one repeated marker pair per active line. An aligned
#      extra dot was a companion that recolored one bounded span, while an
#      off-beat dot was inert.
#    - This combo keeps the same marker-pair cue but changes the dynamics:
#      aligned singleton dots are reflective checkpoints, and an off-beat
#      singleton dot is an absorbing obstacle.
#    - The ray travels continuously along the row/column, but it only paints the
#      beat cells. Therefore an off-beat obstacle can terminate the motion while
#      sitting between two successive beat cells.
#    - Small left-edge row trace, step=2, beats [0,2,4,6,8,10]:
#        input dots: marker 1 at 0 and 2, companion 6 at 4, companion 7 at 8,
#                    obstacle 9 at 9.
#        original: first aligned foreign dot would dominate one span and the
#                  off-beat 9 would be ignored.
#        combo: 0 -> 2 -> 4(hit 6, reverse) -> 2 -> 0(source bounce)
#               -> 2 -> 4(pass used checkpoint) -> 6 -> 8(hit 7, reverse)
#               -> 6 -> 4 -> 2 -> 0(source bounce)
#               -> 2 -> 4 -> 6 -> 8 -> stop at obstacle 9 before 10.
#      The visible difference is that the line grows in stages through multiple
#      reflections, but the final extension is truncated by the off-beat absorber.
#
# b) Complexity calibration
#    - The generator uses only four line archetypes:
#      clear / absorb_only / reflect_only / reflect_absorb.
#      This keeps the object vocabulary small enough for humans while still
#      exposing each sub-rule side by side inside one grid.
#    - Reflective lines use 1-3 companions, but the generator guarantees at
#      least one 2+-companion reflective line and one 2+-companion
#      reflect-then-absorb line so the multi-pass behavior is visible.
#
# c) Hidden assumptions from the original solver that break here
#    - "At most one relevant extra dot per line" is false: reflective lines can
#      contain several aligned singleton dots.
#    - "An aligned extra dot ends the computation in one shot" is false: a
#      companion now causes a reversal and later becomes pass-through.
#    - "Off-beat dots are irrelevant" is false: the absorber can stop the ray
#      between two beat cells, so motion cannot be modeled as a pure span fill.
#    - "Returning to the source with no remaining companions always ends the
#      line" is only true when no absorber exists. With an absorber present, the
#      ray must keep going until that obstacle is actually reached.
#
# d) Placement legality
#    - Every active line still has exactly one repeated marker color with exactly
#      two dots, one on the shooting edge and one one step inward.
#    - Every non-marker color is a singleton on that line.
#    - Companions lie on later beat cells, never on the opposite boundary; the
#      absorber lies off-beat, beyond the inner marker, and never on a boundary.
#      This avoids overlaps, out-of-bounds placements, and most edge-count ties.
#
# e) Visual observability
#    - A human can compare several rows/columns:
#      some fill fully, some stop early at an absorber, some extend in colored
#      stages through reflections, and at least one line does both.
#    - The rule uses only visible geometry: repeated spacing identifies the beat,
#      aligned dots are on that beat, and the absorber is visibly off the beat.
#      No color-id ordering or hidden numeric trick is involved.
#
# f) Visual cleanliness
#    - Interaction stays on one sparse arithmetic progression per active line.
#      The absorber is a single off-beat dot, so the output remains readable
#      instead of turning into a dense simulation with overlapping traces.
#
# g) Color inference
#    - Marker color paints the initial outward segments.
#    - Each companion donates its own color to newly reached beat cells after the
#      bounce that it triggers.
#    - The absorber never recolors anything; it only terminates motion.
#
# h) New solver assumptions checked against the generator
#    - The solver assumes each active line has exactly one repeating marker color,
#      all other colors are singleton, and all extra dots are beyond the inner
#      marker in shooting order. The generator enforces that.
#    - The solver assumes at most one absorber per line. The generator enforces
#      that to keep the first absorbing collision unambiguous.
#    - The solver assumes reflective companions are only "active" once; after a
#      bounce they become pass-through so deeper checkpoints and the absorber can
#      eventually be reached. The generator is designed around that semantics.


def _detect_edge(g: np.ndarray, bg: int) -> str:
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


def _make_beats(edge: str, line_len: int, step: int) -> list[int]:
    if edge in ("left", "top"):
        return list(range(0, line_len, step))
    return list(range(line_len - 1, -1, -step))


def _line_profile(
    g: np.ndarray,
    line: int,
    *,
    by_row: bool,
    edge: str,
    bg: int,
) -> dict | None:
    line_len = g.shape[1] if by_row else g.shape[0]

    def cell(pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    colors: dict[int, list[int]] = defaultdict(list)
    for pos in range(line_len):
        value = int(g[cell(pos)])
        if value != bg:
            colors[value].append(pos)

    if not colors:
        return None

    repeating = {color: ps for color, ps in colors.items() if len(ps) >= 2}
    assert len(repeating) == 1, (
        f"Line {line} ({'row' if by_row else 'col'}) needs exactly one repeating "
        f"marker color: {dict(colors)}"
    )
    marker, mpos = next(iter(repeating.items()))
    assert len(mpos) == 2, (
        f"Line {line} ({'row' if by_row else 'col'}) marker {marker} has "
        f"{len(mpos)} dots (expected 2): {mpos}"
    )

    beats = _make_beats(edge, line_len, abs(mpos[1] - mpos[0]))
    edge_pos = beats[0]
    assert edge_pos in mpos, (
        f"Line {line} ({'row' if by_row else 'col'}) marker {marker} must touch "
        f"the shooting edge at {edge_pos}: {mpos}"
    )
    step = abs(mpos[1] - mpos[0])
    assert step >= 1
    assert len(beats) >= 2
    assert sorted(mpos) == sorted(beats[:2]), (
        f"Line {line} marker {marker} inconsistent with beat order: "
        f"marker={mpos}, beats={beats[:3]}"
    )

    forward = 1 if edge in ("left", "top") else -1
    inner_pos = beats[1]
    pos_to_idx = {pos: idx for idx, pos in enumerate(beats)}

    companions: list[tuple[int, int]] = []
    absorber: tuple[int, int] | None = None
    for color, ps in colors.items():
        if color == marker:
            continue
        assert len(ps) == 1, (
            f"Line {line} ({'row' if by_row else 'col'}) non-marker color {color} "
            f"must be a singleton: {ps}"
        )
        pos = ps[0]
        assert (pos - inner_pos) * forward > 0, (
            f"Line {line} extra dot must be beyond the inner marker: "
            f"marker={mpos}, extra={(color, pos)}"
        )
        if pos in pos_to_idx:
            companions.append((pos_to_idx[pos], int(color)))
        else:
            assert absorber is None, (
                f"Line {line} has multiple off-beat absorbers: {absorber}, {(pos, color)}"
            )
            absorber = (int(pos), int(color))

    companions.sort(key=lambda item: item[0])
    return {
        "marker": int(marker),
        "beats": beats,
        "line_len": int(line_len),
        "companions": companions,
        "absorber": absorber,
    }


def _obstacle_between(start: int, end: int, obstacle_pos: int | None) -> bool:
    if obstacle_pos is None:
        return False
    lo, hi = sorted((start, end))
    return lo < obstacle_pos < hi


def _run_line(profile: dict, visit=None) -> dict[str, int | bool]:
    marker = int(profile["marker"])
    beats: list[int] = list(profile["beats"])
    line_len = int(profile["line_len"])
    companions = list(profile["companions"])
    absorber = profile["absorber"]

    unused = {int(idx): int(color) for idx, color in companions}
    absorber_pos = None if absorber is None else int(absorber[0])

    cur_idx = 0
    direction = 1
    cur_color = marker
    max_steps = max(8, 6 * len(beats) * (len(unused) + 2))

    summary: dict[str, int | bool] = {
        "companion_hits": 0,
        "wall_bounces": 0,
        "stopped_by_absorber": False,
        "returned_to_source": False,
    }

    for _ in range(max_steps):
        current_pos = beats[cur_idx]
        next_idx = cur_idx + direction

        if 0 <= next_idx < len(beats):
            target_pos = beats[next_idx]
            if _obstacle_between(current_pos, target_pos, absorber_pos):
                summary["stopped_by_absorber"] = True
                return summary

            cur_idx = next_idx
            pos = beats[cur_idx]
            if cur_idx in unused:
                comp_color = unused.pop(cur_idx)
                summary["companion_hits"] += 1
                if visit is not None:
                    visit(pos, comp_color, True, marker)
                cur_color = comp_color
                direction *= -1
            else:
                if visit is not None:
                    visit(pos, cur_color, False, marker)

            if absorber_pos is None and cur_idx == 0 and direction == -1 and not unused:
                summary["returned_to_source"] = True
                return summary
            continue

        wall_pos = line_len - 1 if direction == 1 else 0
        if _obstacle_between(current_pos, wall_pos, absorber_pos):
            summary["stopped_by_absorber"] = True
            return summary

        direction *= -1
        summary["wall_bounces"] += 1

    raise AssertionError(f"Line did not terminate under combo dynamics: {profile}")


def _combo_signature(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    edge = _detect_edge(g, bg)
    by_row = edge in ("left", "right")
    num_lines = g.shape[0] if by_row else g.shape[1]

    active_lines = 0
    clear_lines = 0
    absorb_only_lines = 0
    reflect_only_lines = 0
    reflect_absorb_lines = 0
    multi_reflect_absorb_lines = 0
    wall_reflect_lines = 0
    total_companions = 0

    for line in range(num_lines):
        profile = _line_profile(g, line, by_row=by_row, edge=edge, bg=bg)
        if profile is None:
            continue
        active_lines += 1
        total_companions += len(profile["companions"])
        summary = _run_line(profile)
        n_comp = len(profile["companions"])
        has_absorber = profile["absorber"] is not None

        if summary["wall_bounces"]:
            wall_reflect_lines += 1
        if not has_absorber and n_comp == 0:
            clear_lines += 1
        elif has_absorber and n_comp == 0:
            absorb_only_lines += 1
        elif not has_absorber and n_comp > 0:
            reflect_only_lines += 1
        else:
            reflect_absorb_lines += 1
            if int(summary["companion_hits"]) >= 2:
                multi_reflect_absorb_lines += 1

    return {
        "active_lines": active_lines,
        "clear_lines": clear_lines,
        "absorb_only_lines": absorb_only_lines,
        "reflect_only_lines": reflect_only_lines,
        "reflect_absorb_lines": reflect_absorb_lines,
        "multi_reflect_absorb_lines": multi_reflect_absorb_lines,
        "wall_reflect_lines": wall_reflect_lines,
        "total_companions": total_companions,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch periodic rays from one dominant edge, with one-time reflections and absorbing obstacles.

    Input structure:
    - Background is the majority color and all other cells are sparse dots.
    - One boundary has strictly more non-bg dots than the other three; that is
      the shooting edge, so left/right means row-wise motion and top/bottom
      means column-wise motion.
    - Each active shooting-direction line contains exactly one repeated marker
      color with exactly two dots: one on the shooting edge and one step cells
      inward, which reveals the beat spacing.
    - Additional aligned singleton dots on that beat are companions; at most one
      off-beat singleton dot may also appear farther along the line as an absorber.

    Rule:
    - Detect the unique shooting edge and parse each active row/column
      independently.
    - Launch a ray from the edge marker and move it continuously away from the
      edge; only beat cells are paintable, but off-beat absorbers can still be
      hit between two successive beat cells.
    - When the ray reaches an unused companion, that companion cell takes its own
      color, the direction reverses, and later passes can move through it.
    - When the ray reaches a line end, it also reflects. When it reaches the
      off-beat absorber, motion stops before the next beat. Only background cells
      and original marker cells may be repainted by the traveling color.

    Key insight: the marker pair still gives the period, but the solver now has
    to simulate a path with state. Aligned singleton dots are not final span
    endpoints anymore; they are one-time reflectors, while the off-beat dot is
    the true absorbing obstacle that eventually terminates the motion.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    edge = _detect_edge(g, bg)

    by_row = edge in ("left", "right")
    num_lines = g.shape[0] if by_row else g.shape[1]

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    for line in range(num_lines):
        profile = _line_profile(g, line, by_row=by_row, edge=edge, bg=bg)
        if profile is None:
            continue

        def visit(pos: int, color: int, is_companion: bool, base_marker: int) -> None:
            rc = cell(line, pos)
            if is_companion:
                out[rc] = color
                return
            cur = int(out[rc])
            if cur == bg or cur == base_marker:
                out[rc] = color

        _run_line(profile, visit=visit)

    return out


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")
_MAX_RETRIES = 500

_EXTRA_LINE_SPECS = (
    ("clear", 0, 0, 1),
    ("absorb_only", 0, 0, 2),
    ("reflect_only", 1, 3, 3),
    ("reflect_absorb", 1, 3, 4),
)


def _choose_dims(edge: str, n_active: int) -> tuple[int, int, int, int] | None:
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
    boundary = {
        "top": lambda r, c: r == 0,
        "bottom": lambda r, c: r == H - 1,
        "left": lambda r, c: c == 0,
        "right": lambda r, c: c == W - 1,
    }
    counts: dict[str, int] = {}
    for other_edge in _EDGES:
        if other_edge == edge:
            continue
        check = boundary[other_edge]
        counts[other_edge] = sum(1 for r, c in dots_rc if check(r, c))
    return counts


def _companion_index_pool(beats: list[int], line_len: int) -> list[int]:
    return [idx for idx in range(2, len(beats)) if 0 < beats[idx] < line_len - 1]


def _obstacle_candidates(beats: list[int], line_len: int, start_idx: int) -> list[int]:
    if not (0 <= start_idx < len(beats)):
        return []
    beat_set = set(beats)
    direction = 1 if len(beats) == 1 or beats[1] > beats[0] else -1
    start_pos = beats[start_idx]
    if direction == 1:
        return [pos for pos in range(start_pos + 1, line_len - 1) if pos not in beat_set]
    return [pos for pos in range(1, start_pos) if pos not in beat_set]


def _between_beat_candidates(beats: list[int], left_idx: int, right_idx: int) -> list[int]:
    if not (0 <= left_idx < len(beats) and 0 <= right_idx < len(beats)):
        return []
    lo, hi = sorted((beats[left_idx], beats[right_idx]))
    return list(range(lo + 1, hi))


def _sample_line_dots(
    edge: str,
    line_len: int,
    palette: list[int],
    kind: str,
    min_comp: int,
    max_comp: int,
) -> list[tuple[int, int]] | None:
    is_core = kind.endswith("_core")
    base_kind = kind[:-5] if is_core else kind
    need_absorber = base_kind in ("absorb_only", "reflect_absorb")
    need_reflection = base_kind in ("reflect_only", "reflect_absorb")
    max_step = min(_STEP_RANGE[1], line_len - 1)
    steps = list(range(_STEP_RANGE[0], max_step + 1))
    random.shuffle(steps)

    for step in steps:
        beats = _make_beats(edge, line_len, step)
        if len(beats) < 2:
            continue
        if need_absorber and step == 1:
            continue

        pool = _companion_index_pool(beats, line_len)
        if need_reflection:
            local_max_comp = min(max_comp, len(pool))
            if local_max_comp < min_comp:
                continue
            comp_options = list(range(local_max_comp, min_comp - 1, -1))
        else:
            comp_options = [0]

        for n_comp in comp_options:
            for _ in range(12):
                companion_indices: list[int] = []
                absorber_pos: int | None = None

                if need_reflection:
                    preferred_pool = [idx for idx in pool if idx < len(beats) - 1]
                    source_pool = pool
                    if need_absorber and len(preferred_pool) >= n_comp and random.random() < 0.85:
                        source_pool = preferred_pool
                    companion_indices = sorted(random.sample(source_pool, n_comp))
                    if is_core and n_comp >= 2:
                        span = companion_indices[-1] - companion_indices[0] + 1
                        if span <= n_comp:
                            continue

                if need_absorber:
                    if base_kind == "absorb_only":
                        if len(beats) < 4:
                            continue
                        candidates = _between_beat_candidates(beats, 2, 3)
                    else:
                        assert companion_indices
                        if is_core:
                            if companion_indices[-1] + 1 >= len(beats):
                                continue
                            candidates = _between_beat_candidates(
                                beats, companion_indices[-1], companion_indices[-1] + 1
                            )
                        else:
                            preferred_start = min(companion_indices[-1] + 1, len(beats) - 1)
                            candidates = _obstacle_candidates(beats, line_len, preferred_start)
                            if not candidates:
                                candidates = _obstacle_candidates(beats, line_len, companion_indices[-1])
                    if not candidates:
                        continue
                    absorber_pos = random.choice(candidates)

                needed_colors = 1 + n_comp + (1 if absorber_pos is not None else 0)
                if len(palette) < needed_colors:
                    return None

                marker_color = random.choice(palette)
                extras = random.sample([c for c in palette if c != marker_color], needed_colors - 1)
                dots = [(beats[0], marker_color), (beats[1], marker_color)]

                for idx, color in zip(companion_indices, extras[:n_comp]):
                    dots.append((beats[idx], color))

                if absorber_pos is not None:
                    dots.append((absorber_pos, extras[-1]))

                return dots

    return None


def _planned_line_specs(n_active: int) -> list[tuple[str, int, int]]:
    specs: list[tuple[str, int, int]] = [
        ("clear", 0, 0),
        ("absorb_only", 0, 0),
        ("reflect_only_core", 2, 3),
        ("reflect_absorb_core", 2, 3),
    ]
    labels = [item[0] for item in _EXTRA_LINE_SPECS]
    weights = [item[3] for item in _EXTRA_LINE_SPECS]
    bounds = {item[0]: (item[1], item[2]) for item in _EXTRA_LINE_SPECS}
    while len(specs) < n_active:
        label = random.choices(labels, weights=weights, k=1)[0]
        lo, hi = bounds[label]
        specs.append((label, lo, hi))
    random.shuffle(specs)
    return specs


def _gen_attempt() -> np.ndarray | None:
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
    specs = _planned_line_specs(n_active)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    shooting_edge_dots = 0

    for line, (kind, min_comp, max_comp) in zip(active_lines, specs):
        dots = _sample_line_dots(edge, line_len, palette, kind, min_comp, max_comp)
        if dots is None:
            return None

        for pos, color in dots:
            r, c = cell(line, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
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

    sig = _combo_signature(inp)
    if sig["clear_lines"] == 0:
        return None
    if sig["absorb_only_lines"] == 0:
        return None
    if sig["reflect_only_lines"] == 0:
        return None
    if sig["reflect_absorb_lines"] == 0:
        return None
    if sig["multi_reflect_absorb_lines"] == 0:
        return None
    if sig["wall_reflect_lines"] == 0:
        return None
    if sig["total_companions"] < max(4, n_active):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
