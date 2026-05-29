# Variant of db695cfb
# Dimension: D17 — 符号编码参数 (symbol_instructions)
# Change: 无符号编码——所有行为由对象颜色和位置直接决定... → 图例区域定义延伸规则

import random

import numpy as np

from arc_utils import get_bg


_LINE_COLOR = 1
_LEGEND_BG = 0
_LEGEND_SYMBOL = 5
_LEGEND_W = 4
_RULE_DIAG = "diag_perp"
_RULE_HORIZONTAL = "horizontal"
_RULE_VERTICAL = "vertical"
_ALL_RULES = (_RULE_DIAG, _RULE_HORIZONTAL, _RULE_VERTICAL)

# ---------------------------------------------------------------------------
# Design analysis for the D17 variant
#
# concrete_design:
#   legend_area:
#     placement: a full-height strip on the left edge, width 4
#     palette_role:
#       0: legend background / inert wall
#       5: legend symbol strokes
#       other key colors: trigger colors reused in the main field
#     per-rule glyph: 3x3 block inside the strip, stacked from the top
#       center colored pixel + surrounding gray symbol
#       X corners -> emit perpendicular diagonal rays (original behavior)
#       left/right gray -> emit horizontal rays
#       up/down gray -> emit vertical rays
#   main_field:
#     placement: columns >= 4
#     unchanged_structure: blue endpoints still define diagonal segments exactly
#     changed_structure: trigger pixels on those diagonals may now use multiple
#       colors; their extension mode is no longer implicit in color id alone and
#       must be read from the legend strip first
#
# behavior_trace:
#   original:
#     a magenta trigger on a slash/backslash segment always emitted the opposite
#     diagonal pair of rays
#   variant:
#     if the legend shows red with a horizontal symbol, then a red trigger on the
#     same segment emits left/right rays instead; if the legend shows magenta with
#     an X symbol, that trigger still emits the original perpendicular diagonal
#   visible_difference:
#     the same kind of "pixel lying on the blue diagonal" can now branch into
#     different geometric outputs, and the clue is isolated in the edge legend
#
# complexity_calibration:
#   keep only three ray families: perpendicular diagonal, horizontal, vertical
#   sample 2-3 legend entries per grid and force at least one non-diagonal rule
#   to actually appear on a live diagonal trigger, so the legend lookup is both
#   necessary and human-visible in a few examples
#
# implicit_assumption_audit:
#   main() in the seed inferred trigger behavior directly from segment slope; that
#   is no longer valid, so each trigger must carry a legend-decoded rule
#   the seed also scanned the whole grid for candidate colors; with a legend strip,
#   doing so would let legend pixels interfere with line-color inference, so the
#   solver must isolate the main field before grouping blue endpoints
#   rays must not repaint the legend strip, therefore emission ignores columns < 4
#
# placement_legality:
#   generator reserves the left strip exclusively for legend pixels
#   all blue diagonals stay fully inside the main field
#   candidate segments are rejected if their filled interiors or rays would collide
#   with existing main-field pixels or previously reserved output cells
#
# visual_readability:
#   the left strip is a clean black band and never overlaps the constructed output
#   the three symbols are shape-distinct even in 3x3 form
#   distractor triggers use legend colors but are kept off all blue diagonals, so
#   the examples still demonstrate "legend color alone is not enough; position on
#   a connected diagonal is also required"
# ---------------------------------------------------------------------------


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _dirs_for_mode(mode: str, dr: int, dc: int) -> list[tuple[int, int]]:
    if mode == _RULE_DIAG:
        return [(dr, -dc), (-dr, dc)]
    if mode == _RULE_HORIZONTAL:
        return [(0, -1), (0, 1)]
    if mode == _RULE_VERTICAL:
        return [(-1, 0), (1, 0)]
    raise AssertionError(f"unknown legend mode: {mode}")


def _read_legend(grid: np.ndarray, bg: int) -> dict[int, str]:
    h, w = grid.shape
    assert w > _LEGEND_W, "grid must leave room for a main field"

    rules: dict[int, str] = {}
    for r0 in range(0, h - 2, 3):
        center = int(grid[r0 + 1, 1])
        if center in (bg, _LEGEND_BG, _LEGEND_SYMBOL, _LINE_COLOR):
            continue

        left = int(grid[r0 + 1, 0]) == _LEGEND_SYMBOL
        right = int(grid[r0 + 1, 2]) == _LEGEND_SYMBOL
        up = int(grid[r0, 1]) == _LEGEND_SYMBOL
        down = int(grid[r0 + 2, 1]) == _LEGEND_SYMBOL
        corners = (
            int(grid[r0, 0]) == _LEGEND_SYMBOL
            and int(grid[r0, 2]) == _LEGEND_SYMBOL
            and int(grid[r0 + 2, 0]) == _LEGEND_SYMBOL
            and int(grid[r0 + 2, 2]) == _LEGEND_SYMBOL
        )

        if corners and not (left or right or up or down):
            rules[center] = _RULE_DIAG
        elif left and right and not (up or down or corners):
            rules[center] = _RULE_HORIZONTAL
        elif up and down and not (left or right or corners):
            rules[center] = _RULE_VERTICAL
        else:
            raise AssertionError("legend block is malformed")

    assert rules, "legend must define at least one trigger rule"
    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect blue diagonal endpoint pairs, then extend each on-segment trigger by the rule shown for its color in the edge legend.

    Input structure:
    - A black legend strip sits on the left edge. Each 3x3 legend glyph contains one trigger color and a gray direction symbol.
    - In the main field, blue pixels still appear as endpoint pairs on 45-degree diagonals, and colored trigger pixels may lie on those diagonals.

    Rule:
    - Decode the legend strip to map trigger colors to extension modes.
    - Fill each blue diagonal segment between its extreme endpoints, and whenever a trigger color lies on that segment, emit rays in the legend-prescribed directions until the grid boundary.

    Key insight: Trigger colors no longer encode their own behavior; the left-edge legend is the instruction table that tells you whether that color shoots diagonal, horizontal, or vertical rays.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    bg = get_bg(grid)

    legend_rules = _read_legend(grid, bg)

    pts = [(r, c) for r in range(h) for c in range(_LEGEND_W, w) if int(grid[r, c]) == _LINE_COLOR]
    assert len(pts) >= 2, "at least one blue diagonal pair must exist"

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
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = _LINE_COLOR
            elif c >= _LEGEND_W and v != _LINE_COLOR and v in legend_rules:
                mode = legend_rules[v]
                triggers.setdefault((r, c, v), set()).update(_dirs_for_mode(mode, dr, dc))

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
                if c >= _LEGEND_W and int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc

    return out


# --------------- generator (c) ---------------

_H_RANGE = (10, 25)
_W_RANGE = (12, 21)
_SEGMENT_COUNT = (1, 4)
_SEG_LEN_RANGE = (3, 16)
_TRIGGERS_PER_SEG = (1, 3)
_RULE_COUNT = (2, 3)

_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 4)

_SEGMENT_PLACE_RETRIES = 80
_DISTRACTOR_RETRIES = 60
_GEN_RETRIES = 400


def _sample_rule_modes(k: int) -> list[str]:
    if k == 2:
        return [_RULE_DIAG, random.choice([_RULE_HORIZONTAL, _RULE_VERTICAL])]
    return list(_ALL_RULES)


def _paint_legend(grid: np.ndarray, rule_colors: list[int], rule_modes: list[str]) -> None:
    grid[:, :_LEGEND_W] = _LEGEND_BG
    for i, (color, mode) in enumerate(zip(rule_colors, rule_modes)):
        r0 = 3 * i
        grid[r0 + 1, 1] = int(color)
        if mode == _RULE_DIAG:
            grid[r0, 0] = _LEGEND_SYMBOL
            grid[r0, 2] = _LEGEND_SYMBOL
            grid[r0 + 2, 0] = _LEGEND_SYMBOL
            grid[r0 + 2, 2] = _LEGEND_SYMBOL
        elif mode == _RULE_HORIZONTAL:
            grid[r0 + 1, 0] = _LEGEND_SYMBOL
            grid[r0 + 1, 2] = _LEGEND_SYMBOL
        elif mode == _RULE_VERTICAL:
            grid[r0, 1] = _LEGEND_SYMBOL
            grid[r0 + 2, 1] = _LEGEND_SYMBOL
        else:
            raise AssertionError(f"unknown legend mode: {mode}")


def _compute_ray_cells(
    trigger_specs: list[tuple[int, int, int, str]],
    dr: int,
    dc: int,
    H: int,
    W: int,
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for tr, tc, _, mode in trigger_specs:
        for ddr, ddc in _dirs_for_mode(mode, dr, dc):
            r, c = tr + ddr, tc + ddc
            while 0 <= r < H and 0 <= c < W:
                if c >= _LEGEND_W:
                    cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _build_expected(inp: np.ndarray, bg: int, segments) -> np.ndarray:
    H, W = inp.shape
    expected = inp.copy()
    for _, _, trigger_specs, path, dr, dc in segments:
        for r, c in path:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc, tcol, mode in trigger_specs:
            for ddr, ddc in _dirs_for_mode(mode, dr, dc):
                r, c = tr + ddr, tc + ddc
                while 0 <= r < H and 0 <= c < W:
                    if c >= _LEGEND_W and expected[r, c] == bg:
                        expected[r, c] = tcol
                    r += ddr
                    c += ddc
    return expected


def _try_place_segment(
    slope_type: str,
    seg_len: int,
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int] | None:
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if H - 1 - seg_len < 0 or W - 1 - seg_len < _LEGEND_W:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(_LEGEND_W, W - 1 - seg_len)
            dc = 1
        else:
            if H - 1 - seg_len < 0 or _LEGEND_W + seg_len > W - 1:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(_LEGEND_W + seg_len, W - 1)
            dc = -1

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        if any(c < _LEGEND_W for _, c in path):
            continue
        if any(rc in occupied or rc in drawn for rc in path):
            continue
        return path, dc
    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)

    k_rules = random.randint(*_RULE_COUNT)
    if H < 3 * k_rules or W <= _LEGEND_W + _SEG_LEN_RANGE[0]:
        return None

    bg_choices = [c for c in range(10) if c not in (_LINE_COLOR, _LEGEND_BG, _LEGEND_SYMBOL)]
    bg = random.choice(bg_choices)

    rule_modes = _sample_rule_modes(k_rules)
    random.shuffle(rule_modes)
    rule_colors = random.sample([c for c in range(10) if c not in (bg, _LINE_COLOR, _LEGEND_BG, _LEGEND_SYMBOL)], k_rules)
    legend_rules = dict(zip(rule_colors, rule_modes))

    inp = np.full((H, W), bg, dtype=int)
    _paint_legend(inp, rule_colors, rule_modes)

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(H - 1, W - _LEGEND_W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    segments = []
    pending_colors = random.sample(rule_colors, len(rule_colors))

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))
        result = _try_place_segment(slope_type, seg_len, H, W, occupied, drawn)
        if result is None:
            continue

        path, dc = result
        path_interior = path[1:-1]
        max_triggers_here = min(_TRIGGERS_PER_SEG[1], len(path_interior))
        if max_triggers_here <= 0:
            continue
        min_triggers_here = min(_TRIGGERS_PER_SEG[0], max_triggers_here)
        n_triggers = random.randint(min_triggers_here, max_triggers_here)
        trigger_positions = random.sample(path_interior, n_triggers)

        must_take = pending_colors[: min(len(pending_colors), n_triggers)]
        color_list = list(must_take)
        while len(color_list) < n_triggers:
            color_list.append(random.choice(rule_colors))
        random.shuffle(color_list)

        trigger_specs = []
        for (tr, tc), color in zip(trigger_positions, color_list):
            trigger_specs.append((tr, tc, color, legend_rules[color]))

        ray_cells = _compute_ray_cells(trigger_specs, 1, dc, H, W)
        if any(rc in occupied or rc in drawn for rc in ray_cells):
            continue

        ep0, ep1 = path[0], path[-1]
        segments.append((ep0, ep1, trigger_specs, path, 1, dc))
        occupied.update([ep0, ep1])
        occupied.update((tr, tc) for tr, tc, _, _ in trigger_specs)
        drawn.update(set(path_interior) - {(tr, tc) for tr, tc, _, _ in trigger_specs})
        drawn.update(ray_cells)
        pending_colors = pending_colors[len(must_take) :]

    if not segments or pending_colors:
        return None

    used_non_diag = any(mode != _RULE_DIAG for _, _, trigger_specs, _, _, _ in segments for _, _, _, mode in trigger_specs)
    if not used_non_diag:
        return None

    for ep0, ep1, trigger_specs, _, _, _ in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tr, tc, color, _ in trigger_specs:
            inp[tr, tc] = int(color)

    all_line_positions = list(zip(*np.where(inp[:, _LEGEND_W:] == _LINE_COLOR)))
    all_line_positions = [(r, c + _LEGEND_W) for r, c in all_line_positions]
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r = random.randint(0, H - 1)
            c = random.randint(_LEGEND_W, W - 1)
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
            r = random.randint(0, H - 1)
            c = random.randint(_LEGEND_W, W - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            inp[r, c] = random.choice(rule_colors)
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
