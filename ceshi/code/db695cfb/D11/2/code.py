# Variant of db695cfb
# Dimension: D11 — 蓝色像素对的配对规则 (meta)
# Change: 网格中恰好两个蓝色像素，天然配对... → 蓝色像素通过标记颜色配对

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D11 variant
# ===================================
# a) Concrete behavior:
#    - The old puzzle could treat every blue pixel as belonging to one global pool and
#      connect blue endpoints by diagonal geometry alone.
#    - In this variant, each active blue endpoint has exactly one orthogonally adjacent
#      marker pixel. Two blue endpoints with the same marker color form one pair.
#      Example: a red marker next to one blue endpoint and another red marker next to a
#      distant blue endpoint means those two blues must be connected, even if other blue
#      endpoints are closer in raw geometry. Magenta pixels on the interior of that
#      diagonal still emit perpendicular magenta rays exactly as before.
#    - Flow difference on a small example:
#      old logic: blue A, B, C, D are all color 1, so grouping by blue alone can connect
#      the extremes on shared diagonals and accidentally mix intended pairs.
#      new logic: identify A and C as red-marked, B and D as green-marked, then connect
#      only A↔C and B↔D; triggers are read only along those chosen diagonals.
#
# b) Complexity calibration:
#    - To make the new rule visually inferable, each generated grid has at least two
#      marker colors, and every marker color appears exactly twice.
#    - Marker colors are drawn from a small visible palette and are never reused for
#      distractors. This keeps the grouping signal strong without introducing too many
#      categories.
#
# c) Implicit-assumption review against the old solver:
#    - Old `by_color[line_color]` assumed all meaningful blue pixels were one group.
#      That is false now, so grouping must move to "marker color attached to each blue".
#    - Old trigger collection treated any non-blue pixel on the diagonal as a trigger.
#      With added markers this is unsafe if markers ever land on the path. The generator
#      therefore forbids marker placement on diagonal paths, and the solver only treats
#      magenta (6) cells on the chosen path as ray sources.
#    - Old generator allowed unpaired blue distractors. That would now require extra
#      semantics for unmatched marker colors, so this variant removes blue distractors
#      and keeps every blue endpoint paired by construction.
#
# d) Placement legality:
#    - Each blue endpoint must have exactly one 4-neighbor marker.
#    - That marker must not touch any other blue endpoint, sit on any diagonal fill cell,
#      or occupy a future ray cell.
#    - Diagonal segments and their ray footprints remain non-overlapping in the same
#      output-first style as the source task.
#
# e) Visual observability:
#    - Samples always contain at least two pairs, so the solver cannot rely on "the only
#      two blue pixels are the pair". Repeated marker colors across distant endpoints make
#      the pairing rule visible in 2-3 examples.
#    - The generator additionally rejects any sample for which the legacy "treat all blue
#      pixels as one pool" logic would produce the same output. This prevents regression
#      to visually old-style cases where markers are present but irrelevant.
#
# f) Visual cleanliness:
#    - Markers are single orthogonal neighbors of endpoints, never stacked onto diagonals
#      or ray interiors. Segment paths are kept disjoint, so the grid remains readable.
#
# g) Color-change inferability:
#    - Marker colors do not transform; they only encode pairing. The actual drawing colors
#      stay stable: blue for diagonals, magenta for rays. This keeps the new color role
#      easy to infer from few examples.


_LINE_COLOR = 1
_TRIGGER_COLOR = 6


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair blue endpoints by nearby marker colors, then extend magenta rays from magenta cells on the chosen diagonals.

    Input structure:
    - bg = most common color. Blue (1) pixels are diagonal endpoints, and each such blue has one orthogonally adjacent marker pixel in a non-blue, non-magenta color.
    - Two blue endpoints sharing the same marker color belong to the same pair. Magenta (6) trigger pixels may lie on the interior of those diagonals.

    Rule:
    - For each marker color, connect its two associated blue endpoints with a blue diagonal segment.
    - Whenever that segment passes through a magenta pixel, launch magenta rays along the perpendicular diagonal direction until the grid edge, filling only background cells.

    Key insight:
    - Blue alone no longer determines the pairing; the adjacent marker color is the pairing label, and only the diagonals implied by those labels are active.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    bg = get_bg(grid)

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    blue_points = [(r, c) for r in range(h) for c in range(w) if int(grid[r, c]) == _LINE_COLOR]
    assert blue_points, "grid must contain blue endpoints"

    marker_groups: dict[int, list[tuple[int, int]]] = {}
    for r, c in blue_points:
        markers = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if not in_bounds(rr, cc):
                continue
            v = int(grid[rr, cc])
            if v in (bg, _LINE_COLOR, _TRIGGER_COLOR):
                continue
            markers.append((rr, cc, v))
        assert len(markers) == 1, "each blue endpoint must have exactly one adjacent marker"
        _, _, mcol = markers[0]
        marker_groups.setdefault(mcol, []).append((r, c))

    triggers: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for pts in marker_groups.values():
        assert len(pts) == 2, "each marker color must label exactly one blue pair"
        p, q = pts
        dr = q[0] - p[0]
        dc = q[1] - p[1]
        assert dr != 0 and abs(dr) == abs(dc), "paired blue endpoints must share a diagonal"

        path = _diag_path(p, q)
        step_r = 1 if dr > 0 else -1
        step_c = 1 if dc > 0 else -1
        perp_dirs = [(step_r, -step_c), (-step_r, step_c)]

        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = _LINE_COLOR
            elif v == _TRIGGER_COLOR:
                triggers.setdefault((r, c), set()).update(perp_dirs)

    for (r0, c0), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = _TRIGGER_COLOR
                r += ddr
                c += ddc

    return out


# --------------- generator ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_SEGMENT_COUNT = (2, 4)
_SEG_LEN_RANGE = (3, 16)
_TRIGGERS_PER_SEG = (0, 2)

# Marker palette: small set, visible repetition, never blue/magenta/background.
_MARKER_COLORS = [2, 3, 4, 5, 7, 8, 9]

# === CFG: distractor counts ===
_OFF_DIAG_TRIGGERS = (0, 4)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 120
_MARKER_PLACE_RETRIES = 20
_DISTRACTOR_RETRIES = 60
_GEN_RETRIES = 400


def _compute_ray_cells(
    trigger_positions: list[tuple[int, int]],
    perp_dirs: list[tuple[int, int]],
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for tr, tc in trigger_positions:
        for ddr, ddc in perp_dirs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < h and 0 <= c < w:
                cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _build_expected(inp: np.ndarray, bg: int, segments: list[dict]) -> np.ndarray:
    expected = inp.copy()
    h, w = inp.shape
    for seg in segments:
        for r, c in seg["path"]:
            if int(expected[r, c]) == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc in seg["trigger_positions"]:
            for ddr, ddc in seg["perp_dirs"]:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < h and 0 <= c < w:
                    if int(expected[r, c]) == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
    return expected


def _legacy_output(grid: np.ndarray) -> np.ndarray:
    """Original-style behavior used only to reject degenerate variant samples."""
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    bg = get_bg(grid)

    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    best = None
    for col, pts in by_color.items():
        if len(pts) < 2:
            continue
        for i in range(len(pts)):
            r0, c0 = pts[i]
            for j in range(i + 1, len(pts)):
                r1, c1 = pts[j]
                dr = r1 - r0
                dc = c1 - c0
                if dr == 0 or abs(dr) != abs(dc):
                    continue
                cand = (abs(dr), col)
                if best is None or cand > best:
                    best = cand

    if best is None:
        return out

    _, line_color = best
    pts = by_color[line_color]
    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = [(dr, -dc), (-dr, dc)]
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                triggers.setdefault((r, c, v), set()).update(perp_dirs)

    for group in groups_a.values():
        if len(group) >= 2:
            group = sorted(group)
            draw_segment(group[0], group[-1])

    for group in groups_b.values():
        if len(group) >= 2:
            group = sorted(group)
            draw_segment(group[0], group[-1])

    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc
    return out


def _find_marker_cell(
    endpoint: tuple[int, int],
    path: list[tuple[int, int]],
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
    future_rays: set[tuple[int, int]],
    reserved_blue: set[tuple[int, int]],
) -> tuple[int, int] | None:
    path_set = set(path)
    er, ec = endpoint
    for _ in range(_MARKER_PLACE_RETRIES):
        dr, dc = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        mr, mc = er + dr, ec + dc
        if not (0 <= mr < h and 0 <= mc < w):
            continue
        cell = (mr, mc)
        if cell in occupied or cell in drawn or cell in path_set or cell in future_rays:
            continue
        # Exactly one blue adjacency: the marker should not touch any other endpoint.
        bad = False
        for nr, nc in ((mr - 1, mc), (mr + 1, mc), (mr, mc - 1), (mr, mc + 1)):
            if (nr, nc) != endpoint and (nr, nc) in reserved_blue:
                bad = True
                break
        if bad:
            continue
        return cell
    return None


def _try_place_segment(
    slope_type: str,
    seg_len: int,
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int] | None:
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if h - 1 - seg_len < 0 or w - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, h - 1 - seg_len)
            c0 = random.randint(0, w - 1 - seg_len)
            dc = 1
        else:
            if h - 1 - seg_len < 0 or seg_len > w - 1:
                return None
            r0 = random.randint(0, h - 1 - seg_len)
            c0 = random.randint(seg_len, w - 1)
            dc = -1

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        if any(rc in occupied or rc in drawn for rc in path):
            continue
        return path, dc
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_H_RANGE)
    w = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    max_seg_len = min(h - 1, w - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    max_segments = min(_SEGMENT_COUNT[1], len([c for c in _MARKER_COLORS if c != bg]))
    if max_segments < _SEGMENT_COUNT[0]:
        return None
    n_segments = random.randint(_SEGMENT_COUNT[0], max_segments)
    marker_colors = random.sample([c for c in _MARKER_COLORS if c != bg], n_segments)

    occupied: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    segments: list[dict] = []

    for marker_color in marker_colors:
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))
        result = _try_place_segment(slope_type, seg_len, h, w, occupied, drawn)
        if result is None:
            return None
        path, dc = result

        path_interior = path[1:-1]
        n_triggers = random.randint(_TRIGGERS_PER_SEG[0], min(_TRIGGERS_PER_SEG[1], len(path_interior)))
        trigger_positions = random.sample(path_interior, n_triggers) if n_triggers > 0 else []
        perp_dirs = [(1, -dc), (-1, dc)]
        ray_cells = _compute_ray_cells(trigger_positions, perp_dirs, h, w)
        if any(rc in occupied for rc in ray_cells):
            return None

        endpoints = [path[0], path[-1]]
        reserved_blue = occupied | set(endpoints)
        marker_cells = []
        for endpoint in endpoints:
            marker_cell = _find_marker_cell(endpoint, path, h, w, occupied, drawn, ray_cells, reserved_blue)
            if marker_cell is None:
                return None
            marker_cells.append(marker_cell)
            reserved_blue.add(endpoint)
            occupied.add(marker_cell)

        occupied.update(endpoints)
        occupied.update(trigger_positions)
        drawn.update(set(path_interior) - set(trigger_positions))
        drawn.update(ray_cells)
        segments.append(
            {
                "marker_color": marker_color,
                "endpoints": endpoints,
                "marker_cells": marker_cells,
                "trigger_positions": trigger_positions,
                "path": path,
                "perp_dirs": perp_dirs,
            }
        )

    inp = np.full((h, w), bg, dtype=int)
    for seg in segments:
        (ep0, ep1) = seg["endpoints"]
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for mr, mc in seg["marker_cells"]:
            inp[mr, mc] = seg["marker_color"]
        for tr, tc in seg["trigger_positions"]:
            inp[tr, tc] = _TRIGGER_COLOR

    # Off-diagonal magenta distractors remain allowed, but must not mimic markers or sit
    # on active output cells.
    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            if any((r, c) in seg["path"] for seg in segments):
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            break

    try:
        result = main(inp)
    except Exception:
        return None

    expected = _build_expected(inp, bg, segments)
    if not np.array_equal(result, expected):
        return None

    if np.array_equal(_legacy_output(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
