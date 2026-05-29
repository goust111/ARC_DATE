# Variant of 16de56c4
# Dimension: D12 — 多步处理 (multi_step)
# Change: 单步处理——所有活跃行独立并行计算，无依赖关系... → 反射传播
#
# Design analysis
# a) Concrete behavior and trace
#    - The original task treats each active row/column as a single fill decision:
#      either fill the full beat with the marker color, or fill one bounded span
#      with the companion color.
#    - This variant keeps the same edge detection and the same "two marker dots
#      define the beat" cue, but aligned non-marker dots now form a sequence of
#      reflective checkpoints on that beat.
#    - The ray starts on the shooting-edge marker and advances one beat at a time
#      away from the edge. Hitting an unused companion reverses direction and
#      changes the current paint color to that companion's color. Returning to the
#      shooting edge while some companions are still unused causes another outward
#      pass. Once every companion on that line has been triggered, the process
#      stops the first time the ray comes back to the shooting edge.
#    - Small trace, left-edge row, step=2, beats [0,2,4,6,8]:
#      input dots: marker 1 at 0 and 2, companion 6 at 4, companion 7 at 8.
#      Original logic would only see aligned non-marker dots and collapse to one
#      bounded fill. Variant logic does:
#        0 -> 2 -> 4(hit 6, reverse) -> 2 -> 0(reflect at source, one companion
#        still unused) -> 2 -> 4(pass used checkpoint) -> 6 -> 8(hit 7, reverse)
#        -> 6 -> 4 -> 2 -> 0(stop).
#      Because only bg / marker cells are paintable, early reflected color 6 stays
#      visible on the prefix, while the later checkpoint 7 appears farther out.
#
# b) Complexity calibration
#    - A single companion often still resembles the original bounded fill, so the
#      generator biases toward 2-3 companions on many lines and keeps a few
#      companion-free lines so end reflection is also observable.
#    - Companions stay on the same beat and remain single-pixel dots, so the task
#      is still sparse and readable rather than turning into a dense simulation.
#
# c) Hidden assumptions from the original solver that must change
#    - "First aligned non-marker dot ends the fill" is no longer valid; there can
#      now be several aligned singleton colors on the same line.
#    - "Processing is one-shot and span-based" is no longer valid; we need a
#      directional state, a set of unused companions, and repeated passes.
#    - "Painting can overwrite all marker cells inside the span" is unsafe here:
#      if we freely repaint prior trail cells, later reflections erase the staged
#      structure and the output degenerates visually. The new solver only paints
#      cells that are still bg or the original marker color.
#
# d) Placement legality
#    - Every active line still has exactly one repeating color with exactly two
#      dots, one on the shooting edge and one one-step inward.
#    - Every companion is a singleton color on the same beat and strictly beyond
#      the inner marker, so checkpoints are encountered in a well-defined order.
#    - Optional bystanders are singleton colors off the beat and are placed away
#      from boundaries to avoid edge-count ambiguity.
#
# e) Visual observability
#    - Human-solvable examples need repeated outward / inward passes to matter, so
#      the generator enforces at least one line with 0 companions and at least one
#      line with 2+ companions.
#    - The key visible cue is that farther checkpoints do not replace the whole
#      previously painted prefix; instead they extend the colored progression in
#      stages.
#
# f) Visual cleanliness
#    - All interaction remains on one sparse arithmetic progression per active line.
#      Bystanders are off-beat and untouched, so there is no ambiguous overlap.
#
# g) Color inference
#    - Colors are never ordered or transformed by id. A checkpoint simply lends
#      its own visible color to the next newly reached segment, analogous to the
#      color inheritance used by the reflection reference task.

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


def _make_beats(edge: str, line_len: int, step: int) -> list[int]:
    edge_pos = 0 if edge in ("left", "top") else line_len - 1
    if edge_pos == 0:
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

    edge_pos = 0 if edge in ("left", "top") else line_len - 1
    assert edge_pos in mpos, (
        f"Line {line} ({'row' if by_row else 'col'}) marker {marker} must touch "
        f"the shooting edge at {edge_pos}: {mpos}"
    )

    a, b = sorted(mpos)
    step = b - a
    assert step >= 1
    offset = edge_pos % step
    beats = _make_beats(edge, line_len, step)
    pos_to_idx = {p: i for i, p in enumerate(beats)}

    companions: list[tuple[int, int]] = []
    bystanders: list[tuple[int, int]] = []
    for col, ps in colors.items():
        if col == marker:
            continue
        assert len(ps) == 1, (
            f"Line {line} ({'row' if by_row else 'col'}) non-marker color {col} "
            f"must be a singleton: {ps}"
        )
        pos = ps[0]
        if pos % step == offset:
            assert pos in pos_to_idx, f"Aligned pos {pos} missing from beat set"
            companions.append((pos_to_idx[pos], int(col)))
        else:
            bystanders.append((pos, int(col)))

    companions.sort(key=lambda item: item[0])
    return {
        "marker": int(marker),
        "step": int(step),
        "beats": beats,
        "companions": companions,
        "bystanders": bystanders,
    }


def _simulate_line(
    out: np.ndarray,
    g: np.ndarray,
    line: int,
    profile: dict,
    *,
    by_row: bool,
    bg: int,
) -> None:
    marker = int(profile["marker"])
    beats: list[int] = list(profile["beats"])
    unused = {int(idx): int(col) for idx, col in profile["companions"]}

    def cell(pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    cur_idx = 0
    direction = 1
    cur_color = marker
    max_steps = max(1, 4 * len(beats) * (len(unused) + 1))

    for _ in range(max_steps):
        next_idx = cur_idx + direction
        if not (0 <= next_idx < len(beats)):
            direction *= -1
            next_idx = cur_idx + direction
        cur_idx = next_idx
        rc = cell(beats[cur_idx])

        if cur_idx in unused:
            comp_color = unused.pop(cur_idx)
            out[rc] = comp_color
            cur_color = comp_color
            direction *= -1
        else:
            cur = int(out[rc])
            if cur == bg or cur == marker:
                out[rc] = cur_color

        if cur_idx == 0 and direction == -1 and not unused:
            return

    raise AssertionError(
        f"Line {line} did not terminate under reflective propagation: {profile}"
    )


def _reflection_signature(grid: np.ndarray) -> dict[str, int]:
    """Quantify whether a generated input really exercises reflective sequencing."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    edge_counts = {
        "top": sum(1 for c in range(W) if g[0, c] != bg),
        "bottom": sum(1 for c in range(W) if g[H - 1, c] != bg),
        "left": sum(1 for r in range(H) if g[r, 0] != bg),
        "right": sum(1 for r in range(H) if g[r, W - 1] != bg),
    }
    edge = max(edge_counts, key=edge_counts.get)
    by_row = edge in ("left", "right")
    num_lines = H if by_row else W

    active = 0
    end_reflect_lines = 0
    multi_checkpoint_lines = 0
    total_companions = 0
    for li in range(num_lines):
        profile = _line_profile(g, li, by_row=by_row, edge=edge, bg=bg)
        if profile is None:
            continue
        active += 1
        n_comp = len(profile["companions"])
        total_companions += n_comp
        if n_comp == 0:
            end_reflect_lines += 1
        if n_comp >= 2:
            multi_checkpoint_lines += 1

    return {
        "active_lines": active,
        "end_reflect_lines": end_reflect_lines,
        "multi_checkpoint_lines": multi_checkpoint_lines,
        "total_companions": total_companions,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate periodic rays from one dominant edge, reflecting across checkpoints.

    Input structure:
    - bg is the majority color; the non-bg cells are sparse dots on a subset of
      rows or columns determined by one unique shooting edge.
    - Each active shooting-direction line contains exactly one repeating color with
      exactly two dots: one on the shooting edge and one one period inward.
    - Extra singleton dots on the same beat are reflective companions; singleton
      dots off that beat are bystanders and stay unchanged.

    Rule:
    - Find the unique shooting edge by boundary-dot count. Left/right edges imply
      row-wise propagation; top/bottom edges imply column-wise propagation.
    - On each active line, the two marker dots define the beat spacing. Launch a
      ray from the shooting-edge marker, moving away from the edge one beat at a
      time.
    - Reaching an unused companion flips the direction and changes the active paint
      color to that companion's color. Reaching a line end also flips direction.
    - The ray keeps bouncing until every companion on that line has been triggered
      once and the path returns to the shooting-edge marker. Only bg cells and the
      original marker cells are paintable; bystanders are never touched.

    Key insight: the puzzle is no longer a single bounded fill. The marker pair
    still reveals the beat, but aligned singleton dots must be read as an ordered
    sequence of reflective color checkpoints, so the solver has to follow several
    passes of one ray instead of resolving each line in one shot.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
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
    edge = winners[0]

    by_row = edge in ("left", "right")
    num_lines = H if by_row else W

    for li in range(num_lines):
        profile = _line_profile(g, li, by_row=by_row, edge=edge, bg=bg)
        if profile is None:
            continue
        _simulate_line(out, g, li, profile, by_row=by_row, bg=bg)

    return out


# --------------- generator ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_MAX_COMPANIONS = 3
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_P_BYSTANDER = 0.35
_MAX_RETRIES = 400


def _choose_dims(edge: str, n_active: int):
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
    counts = {}
    for other in _EDGES:
        if other == edge:
            continue
        check = boundary[other]
        counts[other] = sum(1 for r, c in dots_rc if check(r, c))
    return counts


def _build_line_dots(
    edge: str,
    line_len: int,
    marker_color: int,
    n_companions: int,
) -> list[tuple[int, int]]:
    max_step = min(_STEP_RANGE[1], (line_len - 1) // (n_companions + 1))
    assert max_step >= 1, (line_len, n_companions)
    step = random.randint(_STEP_RANGE[0], max_step)
    beats = _make_beats(edge, line_len, step)
    assert len(beats) >= n_companions + 2

    dots = [(beats[0], marker_color), (beats[1], marker_color)]
    free_colors = [c for c in _COLORS if c != marker_color]
    companion_indices = sorted(random.sample(range(2, len(beats)), n_companions))
    companion_colors = random.sample(free_colors, n_companions)
    for idx, color in zip(companion_indices, companion_colors):
        dots.append((beats[idx], color))

    used_colors = {marker_color, *companion_colors}
    if random.random() < _P_BYSTANDER:
        other_colors = [c for c in _COLORS if c not in used_colors]
        non_beat = [p for p in range(1, line_len - 1) if p not in beats]
        if other_colors and non_beat:
            dots.append((random.choice(non_beat), random.choice(other_colors)))

    return dots


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

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    shooting_edge_dots = 0

    planned_companions = [0, 2]
    while len(planned_companions) < n_active:
        planned_companions.append(random.choices([1, 2, 3, 0], weights=[2, 4, 2, 1])[0])
    random.shuffle(planned_companions)

    for li, n_comp in zip(active_lines, planned_companions):
        marker_color = random.choice(_COLORS)
        dots = _build_line_dots(edge, line_len, marker_color, n_comp)
        for pos, color in dots:
            r, c = cell(li, pos)
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

    sig = _reflection_signature(inp)
    if sig["multi_checkpoint_lines"] == 0 or sig["end_reflect_lines"] == 0:
        return None
    if sig["total_companions"] < max(3, n_active):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
