# Variant of db695cfb
# Dimension: D5 — 延伸轨迹涂色 (trajectory)
# Change: 品红色连续涂色——垂直对角延伸线全部涂品红色... → 继承触发点颜色但加以区分

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import get_bg
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "solvers"))
    from arc_utils import get_bg


# Design analysis for this D5 variant
# -----------------------------------
# concrete_behavior:
#   - The original task has one stable skeleton: blue endpoints define a full diagonal,
#     magenta pixels lying on that diagonal trigger two perpendicular diagonal rays.
#   - The variant keeps that geometry unchanged. The only semantic change is the color of
#     the emitted rays: trigger pixels stay magenta, but each trigger's pair of rays gets
#     a color chosen by trigger order along its own blue diagonal.
#   - Ordering is visual, not hidden: sort the trigger cells from the upper endpoint to the
#     lower endpoint of the blue segment. The 1st trigger emits red rays, the 2nd green,
#     the 3rd yellow, then the cycle repeats if ever needed.
#   - Example on one backslash segment:
#       input: blue endpoints at (1,1),(6,6), magenta triggers at (2,2),(4,4)
#       original output: path (1,1)..(6,6) blue, rays from both triggers all magenta
#       variant output: same blue path, trigger at (2,2) emits red rays, trigger at (4,4)
#       emits green rays. The trigger cells themselves remain magenta.
#   - The changed step is therefore the "emit perpendicular rays" stage, not the detection
#     of the main blue diagonal structure.
# complexity_calibration:
#   - A fixed small palette [red, green, yellow] keeps the rule inferable from 2-4 samples.
#   - The generator enforces at least one segment with >=2 triggers, otherwise the new color
#     ordering would collapse to a single-ray-color case and become hard to observe.
#   - The color order resets per segment. This avoids a global counting rule across multiple
#     unrelated diagonals, which would be much harder to infer from a few examples.
# implicit_assumptions_review:
#   - Original `draw_segment()` only needed trigger positions plus their source color because
#     every emitted ray used the same color. That assumption is no longer valid.
#   - In the variant, the color depends on trigger index along the segment, so the solver must
#     preserve path order and assign a ray color before drawing rays.
#   - Original generator allowed ray/ray overlap because identical magenta fills made overlap
#     harmless. That would now create order-dependent blocking between different colors, so the
#     variant generator rejects any segment whose rays would intersect existing drawn cells.
# placement_legality:
#   - Endpoints, triggers, distractors, blue fills, and colored rays must all stay pairwise
#     non-overlapping in the final construction, except that a trigger sits on its own segment.
#   - Rays are rejected if they would cross any previously reserved output cell. This keeps the
#     solver deterministic and the image readable.
# visual_observability:
#   - Every generated puzzle has at least two on-segment triggers overall and at least one
#     segment with two or more triggers, so the red→green progression is visible directly.
#   - Background color is forbidden from matching line, trigger, or ray-cycle colors.
# visual_cleanliness:
#   - Segment count is slightly reduced relative to the original seed, and ray intersections are
#     disallowed. This keeps the added color signal legible instead of turning outputs into dense
#     multi-color crossings.
# color_inference:
#   - The ordering cue is purely spatial: "first trigger from top to bottom on a blue diagonal".
#     No sorting by color id, no hidden metadata, and no dependence on off-diagonal distractors.


_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_RAY_COLOR_CYCLE = (2, 3, 4)  # red, green, yellow


def _ray_color_for_trigger(index: int) -> int:
    return int(_RAY_COLOR_CYCLE[index % len(_RAY_COLOR_CYCLE)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color diagonal endpoint pairs, then emit perpendicular rays whose colors depend on trigger order.

    Input structure:
    - The background is the dominant color. Blue endpoint pairs lie on 45-degree diagonals, and magenta trigger pixels may appear between the endpoints on those diagonals.

    Rule:
    - Fill each blue diagonal segment between its extreme blue endpoints.
    - Scan the filled segment from its upper endpoint to its lower endpoint; the 1st trigger emits red perpendicular rays, the 2nd green, the 3rd yellow, then the cycle repeats.

    Key insight:
    - The trigger pixels still mark where beams split, but the ray color is no longer copied from the trigger itself; it is determined by the trigger's position in the sequence along that blue diagonal.
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

    def diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
        r0, c0 = p
        r1, c1 = q
        dr = 1 if r1 > r0 else -1
        dc = 1 if c1 > c0 else -1
        steps = abs(r1 - r0)
        return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]

    # Keep the original line-color inference: select the color with the longest diagonal pair.
    best = None  # (dist, color)
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
                dist = abs(dr)
                cand = (dist, col)
                if best is None or cand > best:
                    best = cand

    assert best is not None, "at least one diagonal-aligned same-color pair must exist"

    _, line_color = best
    pts = by_color[line_color]

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    trigger_specs: list[tuple[int, int, int, tuple[tuple[int, int], tuple[int, int]]]] = []
    seen_triggers: set[tuple[int, int]] = set()

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = ((dr, -dc), (-dr, dc))

        ordered_triggers: list[tuple[int, int]] = []
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                ordered_triggers.append((r, c))

        for idx, (r, c) in enumerate(ordered_triggers):
            if (r, c) in seen_triggers:
                continue
            seen_triggers.add((r, c))
            trigger_specs.append((r, c, _ray_color_for_trigger(idx), perp_dirs))

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for r0, c0, ray_color, dirs in trigger_specs:
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = ray_color
                r += ddr
                c += ddc

    return out


# --------------- generator (c) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_SEGMENT_COUNT = (1, 3)
_SEG_LEN_RANGE = (3, 16)
_TRIGGERS_PER_SEG = (1, 3)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 3)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 400


def _compute_ray_cells(
    trigger_positions: list[tuple[int, int]],
    perp_dirs: tuple[tuple[int, int], tuple[int, int]],
    height: int,
    width: int,
) -> set[tuple[int, int]]:
    """Return all cells touched by the rays, excluding the trigger cells."""
    cells: set[tuple[int, int]] = set()
    for tr, tc in trigger_positions:
        for ddr, ddc in perp_dirs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < height and 0 <= c < width:
                cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _build_expected(
    inp: np.ndarray,
    bg: int,
    segments: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], list[tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]]],
) -> np.ndarray:
    """Build the target output by drawing blue diagonals and ordered colored rays."""
    height, width = inp.shape
    expected = inp.copy()
    for _, _, trigger_positions, path, perp_dirs in segments:
        for r, c in path:
            if int(expected[r, c]) == bg:
                expected[r, c] = _LINE_COLOR
        for idx, (tr, tc) in enumerate(trigger_positions):
            ray_color = _ray_color_for_trigger(idx)
            for ddr, ddc in perp_dirs:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < height and 0 <= c < width:
                    if int(expected[r, c]) == bg:
                        expected[r, c] = ray_color
                    r += ddr
                    c += ddc
    return expected


def _try_place_segment(
    slope_type: str,
    seg_len: int,
    height: int,
    width: int,
    occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int] | None:
    """Try to place one diagonal segment, returning (path, dc) or None."""
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if height - 1 - seg_len < 0 or width - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, height - 1 - seg_len)
            c0 = random.randint(0, width - 1 - seg_len)
            dc = 1
        else:
            if height - 1 - seg_len < 0 or seg_len > width - 1:
                return None
            r0 = random.randint(0, height - 1 - seg_len)
            c0 = random.randint(seg_len, width - 1)
            dc = -1

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        if any(rc in occupied or rc in drawn for rc in path):
            continue
        return path, dc
    return None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*_H_RANGE)
    width = random.randint(*_W_RANGE)
    forbidden_bg = {_LINE_COLOR, _TRIGGER_COLOR, *_RAY_COLOR_CYCLE}
    bg = random.choice([c for c in range(10) if c not in forbidden_bg])

    n_segments = random.randint(*_SEGMENT_COUNT)
    must_multi_idx = random.randrange(n_segments)
    max_seg_len = min(height - 1, width - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    segments: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], list[tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]]] = []

    for seg_idx in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))

        result = _try_place_segment(slope_type, seg_len, height, width, occupied, drawn)
        if result is None:
            continue
        path, dc = result
        path_interior = path[1:-1]
        if not path_interior:
            continue

        min_triggers = 2 if seg_idx == must_multi_idx else _TRIGGERS_PER_SEG[0]
        max_triggers = min(_TRIGGERS_PER_SEG[1], len(path_interior))
        if min_triggers > max_triggers:
            continue
        n_triggers = random.randint(min_triggers, max_triggers)
        trigger_positions = random.sample(path_interior, n_triggers)
        trigger_positions.sort(key=path.index)

        perp_dirs = ((1, -dc), (-1, dc))
        ray_cells = _compute_ray_cells(trigger_positions, perp_dirs, height, width)
        if any(rc in occupied or rc in drawn for rc in ray_cells):
            continue

        ep0, ep1 = path[0], path[-1]
        segments.append((ep0, ep1, trigger_positions, path, perp_dirs))
        occupied.update([ep0, ep1])
        occupied.update(trigger_positions)
        drawn.update(set(path_interior) - set(trigger_positions))
        drawn.update(ray_cells)

    if not segments:
        return None
    if sum(len(trigger_positions) for _, _, trigger_positions, _, _ in segments) < 2:
        return None
    if not any(len(trigger_positions) >= 2 for _, _, trigger_positions, _, _ in segments):
        return None

    inp = np.full((height, width), bg, dtype=int)
    for ep0, ep1, trigger_positions, _, _ in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tp in trigger_positions:
            inp[tp] = _TRIGGER_COLOR

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, height - 1), random.randint(0, width - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            if any(abs(r - lr) == abs(c - lc) and r != lr for lr, lc in all_line_positions):
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add((r, c))
            all_line_positions.append((r, c))
            break

    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, height - 1), random.randint(0, width - 1)
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
