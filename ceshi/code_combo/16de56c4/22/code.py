# Combo variant of 16de56c4 (index 22)
# combo_spec: D1/2+D3/0+D5/0
# Name: 递增步长+交替填充+碰撞停止

from __future__ import annotations

from collections import defaultdict
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
#    Keep the original "one repeated marker color per active line" scaffold, but
#    fuse the three changes into a single object role:
#    - the two marker dots still define the firing edge and the initial gap;
#    - a single farther dot now always exists on the same line;
#    - that extra dot simultaneously donates the second stripe color and acts as
#      the collision obstacle.
#    Example on a left-firing row of length 12:
#      marker A at positions 0 and 2  -> initial gap = 2
#      accelerated path               -> 0, 2, 5, 9
#      blocker/color-donor B at 7     -> output paints 0,2,5 as A,B,A and stops
#                                         before reaching 9 because 7 blocks it.
#      blocker/color-donor B at 9     -> output shows A,B,A,B on 0,2,5,9, but
#                                         the last B is the untouched blocker.
#    Compared with the base task, the ray is no longer modular, no longer
#    monochrome, and no longer span-bounded by companion alignment.
#
# b) Complexity calibration
#    To make the fusion visible in 2-4 examples, every active line uses exactly
#    one extra dot, and each grid mixes aligned blockers with off-path blockers.
#    This keeps the vocabulary small while exposing all three dimensions:
#    irregular gaps, ABAB coloring, and collision before the next hit.
#
# c) Implicit assumptions that break
#    - Any `p % step == offset` test is invalid because D1/2 removes constant
#      spacing; the solver must explicitly enumerate the trajectory.
#    - "Companion only matters if aligned" is invalid because D5/0 makes any
#      farther non-bg pixel a blocker.
#    - "Companion color floods a span" is invalid because D3/0 now alternates
#      color by hit index and D5/0 forbids overwriting the blocker cell itself.
#    - Sorting marker positions is not enough to orient parity; the edge marker
#      must be treated as beat index 0 for both forward and reverse firing.
#
# d) Placement legality
#    The generator puts the first marker on the shooting edge, the second marker
#    on the first accelerated hop, and the blocker strictly farther inward.
#    Off-path blockers are sampled between two future hits so they visibly cancel
#    at least one later hit.  Aligned blockers are restricted to odd-index hits
#    so the blocker's own color matches the visible ABAB stripe phase.
#
# e) Visual observability
#    Humans can read the rule from several lines in one grid:
#    - aligned blocker lines display visible A/B/A/B stripes on accelerated hits;
#    - offset blocker lines show that even a non-hit obstacle truncates the ray;
#    - multiple gaps repeat the 1,2,3... growth pattern rather than equal beats.
#    No ordering depends on color ids or hidden metadata.
#
# f) Visual cleanliness
#    Active lines stay sparse: exactly three dots in the input and one trajectory
#    in the output.  Lines remain separated by blank rows/columns, and the palette
#    is kept small per grid to avoid color noise.
#
# g) Color-change inferability
#    The second color is always anchored by the visible blocker on that line, so
#    the alternation does not require guessing a global color mapping.  The same
#    palette is reused across lines to provide repeated evidence without forcing
#    any invisible correspondence.
#
# h) Degeneracy guard
#    The generator rejects weak cases by requiring at least one off-path blocker,
#    at least two aligned blockers, and enough filled hits to expose acceleration.
#    This prevents samples that could be mistaken for the original periodic-span
#    task or for a plain monochrome collision task.
#
# i) New solver assumptions mirrored by generation
#    The solver assumes each active line has exactly one repeated marker color
#    and at least one farther non-marker dot.  The nearest farther non-bg dot is
#    treated as the collision obstacle and as the donor of the alternating second
#    color.  The generator still uses exactly one such dot per line so the rule
#    stays maximally legible, and it validates samples through a full
#    `main()` round-trip before returning any sample.


def _edge_params(edge: str, line_len: int) -> tuple[int, int]:
    if edge in ("left", "top"):
        return 0, 1
    return line_len - 1, -1


def _shooting_edge(g: np.ndarray, bg: int) -> str:
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


def _line_axes(edge: str, shape: tuple[int, int]):
    H, W = shape
    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    return by_row, num_lines, line_len, cell


def _trajectory_positions(start: int, gap: int, direction: int, line_len: int) -> list[int]:
    """Enumerate positions reached by jumps of gap, gap+1, gap+2, ..."""
    assert gap >= 1
    assert direction in (-1, 1)
    pos = start
    step = gap
    path = [start]
    while True:
        pos += direction * step
        if not (0 <= pos < line_len):
            break
        path.append(pos)
        step += 1
    return path


def _line_colors(g: np.ndarray, bg: int, line: int, line_len: int, cell):
    colors: dict[int, list[int]] = defaultdict(list)
    for p in range(line_len):
        v = int(g[cell(line, p)])
        if v != bg:
            colors[v].append(p)
    return colors


def _blocker_before(pos: int, stop_pos: int, direction: int) -> bool:
    return pos < stop_pos if direction == 1 else pos > stop_pos


def _farther_inward(ref_pos: int, pos: int, direction: int) -> bool:
    return pos > ref_pos if direction == 1 else pos < ref_pos


def _nearest_blocker(extras: list[tuple[int, int]], ref_pos: int, direction: int) -> tuple[int, int]:
    farther = [(color, pos) for color, pos in extras if _farther_inward(ref_pos, pos, direction)]
    assert farther, "No blocker farther inward than the inner marker"
    if direction == 1:
        return min(farther, key=lambda item: item[1])
    return max(farther, key=lambda item: item[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire accelerating rays from a unique edge, alternate their colors, and stop on collision.

    Input structure:
    - Background is the most common color and all informative cells are isolated dots.
    - Exactly one boundary has strictly more non-background dots than the other three;
      that boundary is the shooting edge and fixes whether the active lines are rows or columns.
    - Each active line has exactly one repeated marker color with two dots: one on the shooting
      edge and one on the first inward hit of the ray, which defines the initial gap.
    - Each active line also has one or more farther dots of different colors; the nearest such
      dot supplies the alternating second color and serves as the obstacle that terminates the ray.

    Rule:
    - Identify the unique shooting edge and process each row/column in that firing direction.
    - From the edge marker, enumerate the ray path with jump lengths gap, gap+1, gap+2, ...
      where gap is the distance between the two marker dots.
    - Paint the reached trajectory cells in alternating colors anchored at the shooting edge:
      hit 0 uses the marker color, hit 1 uses the extra dot's color, hit 2 uses the marker
      color again, and so on.
    - The extra dot is a solid collision obstacle regardless of whether it lies on the
      trajectory itself; only hits strictly before that obstacle are newly painted, and the
      obstacle cell remains as it was in the input.

    Key insight:
    The repeated pair no longer defines a modular residue class.  It defines only the first hop
    of an accelerating ray, while the third dot has a fused role: its color drives the A/B stripe
    pattern and its position physically truncates the ray before any later accelerated hit.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    edge = _shooting_edge(g, bg)
    by_row, num_lines, line_len, cell = _line_axes(edge, g.shape)
    edge_pos, direction = _edge_params(edge, line_len)

    for li in range(num_lines):
        colors = _line_colors(g, bg, li, line_len, cell)
        if not colors:
            continue

        repeating = {color: ps for color, ps in colors.items() if len(ps) >= 2}
        assert len(repeating) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) must have exactly one "
            f"repeating marker color: {dict(colors)}"
        )
        marker, mpos = next(iter(repeating.items()))
        assert len(mpos) == 2, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} has "
            f"{len(mpos)} dots (expected 2): {mpos}"
        )
        assert edge_pos in mpos, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} must include "
            f"shooting-edge position {edge_pos}: {mpos}"
        )

        gap = abs(mpos[0] - mpos[1])
        path = _trajectory_positions(edge_pos, gap, direction, line_len)
        assert len(path) >= 2, f"Line {li} gap {gap} yields no interior marker"
        assert path[1] in mpos, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} does not match "
            f"the first accelerated hop: {mpos}"
        )

        extras = [(color, p) for color, ps in colors.items() if color != marker for p in ps]
        assert extras, (
            f"Line {li} ({'row' if by_row else 'col'}) must have at least one non-marker "
            f"obstacle/color donor: {dict(colors)}"
        )
        alt_color, stop_pos = _nearest_blocker(extras, path[1], direction)
        assert alt_color != marker
        assert _farther_inward(path[1], stop_pos, direction), (
            f"Line {li} ({'row' if by_row else 'col'}) blocker must be farther inward than "
            f"the inner marker: marker={mpos}, blocker={stop_pos}, extras={extras}"
        )

        for hit_idx, pos in enumerate(path):
            if not _blocker_before(pos, stop_pos, direction):
                break
            rc = cell(li, pos)
            cur = int(out[rc])
            assert cur in (bg, marker), (
                f"Line {li} ({'row' if by_row else 'col'}) has unexpected obstacle on the "
                f"painted trajectory prefix at {pos}: {cur}"
            )
            out[rc] = marker if hit_idx % 2 == 0 else alt_color

    return out


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_GAP_RANGE = (1, 4)
_ALL_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")
_MAX_RETRIES = 500


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


def _choose_line_types(n_active: int) -> list[str]:
    kinds = ["aligned", "aligned", "offset"]
    while len(kinds) < n_active:
        kinds.append(random.choice(("aligned", "offset")))
    random.shuffle(kinds)
    return kinds


def _other_edge_pos(edge: str, line_len: int) -> int:
    return line_len - 1 if edge in ("left", "top") else 0


def _line_options(edge: str, line_len: int, gap: int):
    edge_pos, direction = _edge_params(edge, line_len)
    other_edge = _other_edge_pos(edge, line_len)
    path = _trajectory_positions(edge_pos, gap, direction, line_len)
    if len(path) < 4:
        return None

    aligned_hits = [idx for idx in range(3, len(path)) if idx % 2 == 1 and path[idx] != other_edge]
    offset_choices: list[tuple[int, list[int]]] = []
    for idx in range(2, len(path) - 1):
        lo, hi = sorted((path[idx], path[idx + 1]))
        blockers = [p for p in range(lo + 1, hi) if p not in (0, line_len - 1)]
        if blockers:
            offset_choices.append((idx, blockers))

    return {
        "path": path,
        "aligned_hits": aligned_hits,
        "offset_choices": offset_choices,
    }


def _build_line_dots(edge: str, line_len: int, marker_color: int, palette: list[int], kind: str):
    candidates = []
    for gap in range(_GAP_RANGE[0], min(_GAP_RANGE[1], line_len - 1) + 1):
        options = _line_options(edge, line_len, gap)
        if options is None:
            continue
        if kind == "aligned" and options["aligned_hits"]:
            candidates.append((gap, options))
        if kind == "offset" and options["offset_choices"]:
            candidates.append((gap, options))
    if not candidates:
        return None

    gap, options = random.choice(candidates)
    path = options["path"]
    other_colors = [c for c in palette if c != marker_color]
    if not other_colors:
        other_colors = [c for c in _ALL_COLORS if c != marker_color]
    if not other_colors:
        return None
    blocker_color = random.choice(other_colors)

    if kind == "aligned":
        aligned_hits = options["aligned_hits"]
        if len(aligned_hits) >= 2 and random.random() < 0.7:
            hit_idx = random.choice(aligned_hits[1:])
        else:
            hit_idx = random.choice(aligned_hits)
        blocker_pos = path[hit_idx]
        visible_hits = hit_idx
    else:
        offset_choices = options["offset_choices"]
        if len(offset_choices) >= 2 and random.random() < 0.7:
            prefix_idx, blockers = random.choice(offset_choices[1:])
        else:
            prefix_idx, blockers = random.choice(offset_choices)
        blocker_pos = random.choice(blockers)
        visible_hits = prefix_idx + 1

    dots = [(path[0], marker_color), (path[1], marker_color), (blocker_pos, blocker_color)]
    meta = {
        "kind": kind,
        "gap": gap,
        "path_len": len(path),
        "visible_hits": visible_hits,
        "blocker_on_path": blocker_pos in set(path),
    }
    return dots, meta


def _count_other_edge_dots(
    dots_rc: list[tuple[int, int]], edge: str, H: int, W: int
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


def _visibility_score(meta: list[dict]) -> int:
    score = 0
    for item in meta:
        score += item["visible_hits"]
        score += max(0, item["path_len"] - 3)
        if item["kind"] == "offset":
            score += 2
    return score


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

    palette_size = random.randint(3, 5)
    palette = random.sample(_ALL_COLORS, palette_size)
    line_types = _choose_line_types(n_active)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    meta: list[dict] = []
    shooting_edge_dots = 0

    for li, kind in zip(active_lines, line_types):
        marker_color = random.choice(palette)
        built = _build_line_dots(edge, line_len, marker_color, palette, kind)
        if built is None:
            return None
        dots, line_meta = built
        meta.append(line_meta)

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

    if sum(item["kind"] == "offset" for item in meta) < 1:
        return None
    if sum(item["kind"] == "aligned" for item in meta) < 2:
        return None
    if _visibility_score(meta) < 3 * n_active + 6:
        return None

    try:
        out = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
