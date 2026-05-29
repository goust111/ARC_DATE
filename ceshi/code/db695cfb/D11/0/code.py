# Variant of db695cfb
# Dimension: D11 — 蓝色像素对的配对规则 (meta)
# Change: 网格中恰好两个蓝色像素，天然配对... → 多蓝色像素按颜色深浅配对

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D11 variant:
#
# spec_variant:
#   visible_objects:
#     - two intended diagonal endpoint pairs, one dark blue (1) and one light blue (8)
#     - 0-2 magenta trigger pixels (6) on each intended diagonal interior
#     - optional unpaired dark/light blue distractor pixels and off-diagonal magenta distractors
#   behavioral_change:
#     - original: solve after finding one line color with a diagonal pair
#     - variant: first split blue-family pixels by shade, then connect each shade's own pair
#   concrete_trace:
#     - original local step: if 1 and 8 both appear, the old solver picks only one color and ignores
#       the other diagonal entirely.
#     - new local step: collect every non-trigger color that actually has a same-color diagonal pair,
#       then run the original diagonal-fill and trigger-ray logic independently per color.
#     - difference: the output now contains both the dark-blue diagonal and the light-blue diagonal.
#   complexity_calibration:
#     - keep exactly one intended pair per blue shade so the new idea is visually obvious and there is
#       no hidden ambiguity about which same-shade points should pair.
#     - optional same-shade distractors remain unpaired, so grouping by shade is required but pairing
#       inside a shade stays readable from 2-4 examples.
#   old_assumptions_audited:
#     - "choose one best line color" is invalid now and replaced.
#     - "any non-line color hit by a segment is a trigger" is narrowed: other blue-shade line cells are
#       not treated as triggers, only non-line colors such as magenta can emit rays.
#     - grouping by diagonal orientation still works, but now it must run once per shade.
#   placement_legality:
#     - the generator places the two intended diagonals and their rays on disjoint cells.
#     - optional same-shade distractors are rejected if they diagonally align with any existing pixel of
#       the same shade, preventing accidental extra segments.
#   visual_observability:
#     - every generated puzzle contains both dark blue and light blue endpoint pairs, so the solver must
#       use visible shade grouping rather than an invisible ordering rule.
#   visual_cleanliness:
#     - diagonals and rays are kept non-overlapping, so line color, trigger color, and background remain
#       separable by eye.
#   color_inference:
#     - no color-id-based ordering drives the transform; the only color logic is the visible partition
#       into two blue shades, each following the same geometric rule.
#   solver_assumptions_checked:
#     - for each candidate line color, there is at most one intended same-shade diagonal pair in the
#       generator, so connecting the diagonal extremes reproduces the intended segment.
#     - generator guarantees rays never need to overwrite another input pixel or another drawn feature.


# --------------- solver ---------------

_DARK_BLUE = 1
_LIGHT_BLUE = 8
_LINE_COLORS = (_DARK_BLUE, _LIGHT_BLUE)
_TRIGGER_COLOR = 6


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect dark-blue and light-blue diagonal endpoint pairs separately, then emit magenta rays from triggers on those diagonals.

    Input structure:
    - bg is the most common color.
    - Blue-family endpoints appear in two visible shades: dark blue and light blue.
    - Each shade contributes its own diagonal pair, and magenta trigger pixels may sit on either diagonal.

    Rule:
    - Partition the blue-family pixels by shade, and for each shade connect the matching diagonal pair by filling the full 45-degree segment.
    - When a filled diagonal passes through a magenta trigger pixel, launch magenta rays along the perpendicular diagonal in both directions until the grid edge.

    Key insight:
    - The geometry is unchanged from the original task; the extra reasoning step is to classify the blue pixels into two shades first, because each shade only connects to itself.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    bg = get_bg(grid)

    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    def has_diagonal_pair(points: list[tuple[int, int]]) -> bool:
        for i, (r0, c0) in enumerate(points):
            for r1, c1 in points[i + 1:]:
                if abs(r1 - r0) == abs(c1 - c0) and r1 != r0:
                    return True
        return False

    def diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
        r0, c0 = p
        r1, c1 = q
        dr = 1 if r1 > r0 else -1
        dc = 1 if c1 > c0 else -1
        steps = abs(r1 - r0)
        return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]

    line_colors = [col for col, pts in by_color.items() if col != _TRIGGER_COLOR and len(pts) >= 2 and has_diagonal_pair(pts)]
    assert line_colors, "at least one diagonal-aligned same-shade pair must exist"
    line_color_set = set(line_colors)

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    def draw_segment(p: tuple[int, int], q: tuple[int, int], line_color: int) -> None:
        path = diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = [(dr, -dc), (-dr, dc)]
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v not in line_color_set:
                key = (r, c, v)
                triggers.setdefault(key, set()).update(perp_dirs)

    for line_color in line_colors:
        pts = by_color[line_color]
        groups_a: dict[int, list[tuple[int, int]]] = {}
        groups_b: dict[int, list[tuple[int, int]]] = {}
        for r, c in pts:
            groups_a.setdefault(r - c, []).append((r, c))
            groups_b.setdefault(r + c, []).append((r, c))

        for group in groups_a.values():
            if len(group) >= 2:
                group = sorted(group)
                draw_segment(group[0], group[-1], line_color)

        for group in groups_b.values():
            if len(group) >= 2:
                group = sorted(group)
                draw_segment(group[0], group[-1], line_color)

    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc

    return out


# --------------- generator ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_SEG_LEN_RANGE = (3, 16)
_TRIGGERS_PER_SEG = (0, 2)

# === CFG: distractor counts ===
_UNPAIRED_SHADE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 4)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 120
_DISTRACTOR_RETRIES = 60
_GEN_RETRIES = 400


def _compute_ray_cells(triggers, perp_dirs, H, W):
    """Compute all cells touched by perpendicular rays from trigger positions."""
    cells = set()
    for tr, tc in triggers:
        for ddr, ddc in perp_dirs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < H and 0 <= c < W:
                cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _build_expected(inp, bg, segments):
    """Build the expected output grid by filling each shade's diagonal and its rays."""
    H, W = inp.shape
    expected = inp.copy()
    for line_color, _, _, trigger_positions, path, perp_dirs in segments:
        for r, c in path:
            if expected[r, c] == bg:
                expected[r, c] = line_color
        for tr, tc in trigger_positions:
            for ddr, ddc in perp_dirs:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < H and 0 <= c < W:
                    if expected[r, c] == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
    return expected


def _try_place_segment(seg_len, H, W, occupied, drawn):
    """Try to place one diagonal segment, returning (path, dc) or None."""
    for _ in range(_SEGMENT_PLACE_RETRIES):
        slope_type = random.choice(["backslash", "slash"])
        if slope_type == "backslash":
            if H - 1 - seg_len < 0 or W - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(0, W - 1 - seg_len)
            dc = 1
        else:
            if H - 1 - seg_len < 0 or seg_len > W - 1:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(seg_len, W - 1)
            dc = -1

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        if any(rc in occupied or rc in drawn for rc in path):
            continue
        return path, dc
    return None


def _same_shade_diagonal(r, c, points):
    return any(abs(r - pr) == abs(c - pc) and r != pr for pr, pc in points)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_DARK_BLUE, _LIGHT_BLUE, _TRIGGER_COLOR)])

    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied = set()
    drawn = set()
    segments = []
    shade_points = {_DARK_BLUE: [], _LIGHT_BLUE: []}

    trigger_counts = []
    for line_color in (_DARK_BLUE, _LIGHT_BLUE):
        placed = None
        for _ in range(_SEGMENT_PLACE_RETRIES):
            seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))
            result = _try_place_segment(seg_len, H, W, occupied, drawn)
            if result is None:
                continue
            path, dc = result

            path_interior = path[1:-1]
            max_triggers = min(_TRIGGERS_PER_SEG[1], len(path_interior))
            n_triggers = random.randint(_TRIGGERS_PER_SEG[0], max_triggers)
            trigger_positions = random.sample(path_interior, n_triggers) if n_triggers > 0 else []
            perp_dirs = [(1, -dc), (-1, dc)]

            ray_cells = _compute_ray_cells(trigger_positions, perp_dirs, H, W)
            if any(rc in occupied or rc in drawn for rc in ray_cells):
                continue

            ep0, ep1 = path[0], path[-1]
            placed = (line_color, ep0, ep1, trigger_positions, path, perp_dirs, ray_cells)
            break

        if placed is None:
            return None

        line_color, ep0, ep1, trigger_positions, path, perp_dirs, ray_cells = placed
        segments.append((line_color, ep0, ep1, trigger_positions, path, perp_dirs))
        shade_points[line_color].extend([ep0, ep1])
        trigger_counts.append(len(trigger_positions))
        occupied.update([ep0, ep1])
        occupied.update(trigger_positions)
        drawn.update(set(path[1:-1]) - set(trigger_positions))
        drawn.update(ray_cells)

    if sum(trigger_counts) == 0:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for line_color, ep0, ep1, trigger_positions, _, _ in segments:
        inp[ep0] = line_color
        inp[ep1] = line_color
        for tp in trigger_positions:
            inp[tp] = _TRIGGER_COLOR

    for _ in range(random.randint(*_UNPAIRED_SHADE_PIXELS)):
        shade = random.choice(_LINE_COLORS)
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            if _same_shade_diagonal(r, c, shade_points[shade]):
                continue
            inp[r, c] = shade
            occupied.add((r, c))
            shade_points[shade].append((r, c))
            break

    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            break

    try:
        result = main(inp)
    except Exception:
        return None

    expected = _build_expected(inp, bg, segments)
    if np.array_equal(result, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
