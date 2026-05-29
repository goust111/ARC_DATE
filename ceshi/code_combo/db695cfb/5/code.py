# Combo variant of db695cfb (index 5)
# combo_spec: D4/1+D7/3
# Name: 虚线主路径+邻近触发

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# spec_variant:
#   preserved_core:
#     - The dominant color is still background.
#     - One non-background color forms endpoint pairs on 45-degree diagonals.
#     - Another non-background color emits perpendicular diagonal rays when activated.
#     - The line color is still identified as the color owning the longest diagonal-aligned pair.
#   fused_change:
#     - D4/1 changes the visible blue connector from a solid diagonal to an every-other-cell dash.
#     - D7/3 changes the trigger test from "pixel lies on the path" to "pixel is one perpendicular
#       diagonal step beside an interior path cell".
#     - The important fusion point is that adjacency is checked against the full hidden path, not just
#       against the cells that remain blue in the dashed rendering.
#   behavior_trace:
#     - Example "\" segment: endpoints at (1,1) and (7,7), so the hidden path is
#       [(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7)].
#     - Visible blue output only paints even path indices: (1,1),(3,3),(5,5),(7,7). Cells (2,2),(4,4),
#       (6,6) are still part of the inferred trajectory but stay blank unless hit by some ray.
#     - If a magenta pixel sits at (3,1), it is adjacent to hidden path cell (2,2), so it is active
#       even though no blue pixel exists at (2,2). Its slash ray begins from (3,1), immediately enters
#       (2,2), and can therefore reveal the missing dash position with magenta.
#     - Original task would have required a magenta exactly on the diagonal. The combo instead asks the
#       solver to reconstruct the invisible full diagonal first, then test side-neighbors of that path.
#   complexity_calibration:
#     - Dashed rendering uses a single repeated parity rule, so the pattern is visible rather than noisy.
#     - Generator restricts segment lengths to even values. That keeps both endpoints on the visible dash
#       parity, making the dashed connector visually clean across all examples.
#     - At least one active trigger per generated sample is anchored to a hidden dash-gap, so the fused
#       rule is not optional decoration.
#   implicit_assumption_audit:
#     - Original draw loop assumed every path cell should be colored; now only visible parity cells are.
#     - Original trigger detection assumed "non-line pixel encountered while drawing the line"; this is
#       invalid after the trigger moves off the path, so detection must scan perpendicular neighbors of
#       every interior hidden-path cell.
#     - Because hidden path cells are still semantically occupied, the generator must reserve the full
#       interior path, not just the visible blue subset, otherwise distractors could land on an unseen
#       trajectory and create ambiguity.
#     - Off-path magenta distractors must avoid every segment's adjacency zone; otherwise they would
#       accidentally become valid triggers.
#   placement_legality:
#     - A trigger zone is the set of in-bounds perpendicular-diagonal neighbors of every interior path
#       cell on the hidden full segment.
#     - Active triggers are chosen from that zone; distractor magenta pixels are forced outside it.
#     - Segment interiors, ray cells, and trigger zones are reserved to stop overlaps and accidental
#       multi-segment ownership.
#   visual_observability:
#     - The solver-facing novelty is human-visible because some rays originate beside empty cells that
#       later become magenta, revealing that the path continues through blank gaps.
#     - The dashed connector is globally consistent, so a few train pairs are enough to infer the parity.
#   cleanliness:
#     - Full hidden paths are kept clear of unrelated pixels.
#     - Trigger zones of different segments are kept disjoint.
#     - The generator rejects samples with no hidden-gap trigger, preventing regression to a simpler
#       "continuous path + adjacent trigger" reading.


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _in_bounds(H: int, W: int, r: int, c: int) -> bool:
    return 0 <= r < H and 0 <= c < W


def _diag_aligned(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a != b and abs(a[0] - b[0]) == abs(a[1] - b[1])


def _is_visible_path_index(idx: int) -> bool:
    return idx % 2 == 0


def _visible_path_cells(path):
    return [cell for idx, cell in enumerate(path) if _is_visible_path_index(idx)]


def _segment_zone(path, perp_dirs, H, W):
    zone = set()
    for r, c in path[1:-1]:
        for ddr, ddc in perp_dirs:
            nr, nc = r + ddr, c + ddc
            if _in_bounds(H, W, nr, nc):
                zone.add((nr, nc))
    return zone


def _compute_ray_cells(triggers, perp_dirs, H, W):
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
    H, W = inp.shape
    expected = inp.copy()
    for segment in segments:
        for r, c in _visible_path_cells(segment["path"]):
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc in segment["trigger_positions"]:
            for ddr, ddc in segment["perp_dirs"]:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < H and 0 <= c < W:
                    if expected[r, c] == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
    return expected


def _try_place_segment(slope_type, seg_len, H, W, occupied, reserved):
    for _ in range(_SEGMENT_PLACE_RETRIES):
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
        if any(rc in occupied or rc in reserved for rc in path):
            continue
        return path, dc
    return None


def _choose_adjacent_triggers(
    path,
    perp_dirs,
    H,
    W,
    occupied,
    reserved,
    all_trigger_positions,
    existing_trigger_zone,
):
    anchor_options = []
    for idx in range(1, len(path) - 1):
        r, c = path[idx]
        options = []
        for ddr, ddc in perp_dirs:
            cell = (r + ddr, c + ddc)
            if not _in_bounds(H, W, cell[0], cell[1]):
                continue
            if cell in occupied or cell in reserved or cell in existing_trigger_zone:
                continue
            if any(_diag_aligned(cell, other) for other in all_trigger_positions):
                continue
            options.append(cell)
        if options:
            anchor_options.append(
                {
                    "anchor_idx": idx,
                    "hidden": not _is_visible_path_index(idx),
                    "options": options,
                }
            )

    if not anchor_options:
        return []

    max_triggers = min(_TRIGGERS_PER_SEG[1], len(anchor_options))
    min_triggers = min(_TRIGGERS_PER_SEG[0], max_triggers)
    n_triggers = random.randint(min_triggers, max_triggers)
    if n_triggers == 0:
        return []

    chosen = []
    for item in random.sample(anchor_options, len(anchor_options)):
        if len(chosen) >= n_triggers:
            break
        valid = [
            cell
            for cell in random.sample(item["options"], len(item["options"]))
            if not any(_diag_aligned(cell, prev["cell"]) for prev in chosen)
        ]
        if not valid:
            continue
        chosen.append(
            {
                "cell": valid[0],
                "anchor_idx": item["anchor_idx"],
                "hidden": item["hidden"],
            }
        )

    if len(chosen) != n_triggers:
        return None
    return chosen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a hidden full diagonal between same-color endpoints, render it as a dashed line, then emit perpendicular rays from offset trigger pixels adjacent to that hidden path.

    Input structure:
    - The background is the dominant color. One non-background color appears as endpoint pairs on shared 45-degree diagonals, while trigger pixels of another color sit one perpendicular-diagonal step beside interior cells of those diagonals rather than directly on them.

    Rule:
    - For each maximal diagonal endpoint pair, infer the full endpoint-to-endpoint path but only paint every other path cell in the line color, producing a dashed connector.
    - For every non-line pixel found one perpendicular-diagonal step away from any interior hidden-path cell, extend that pixel's color along the path's perpendicular diagonal in both directions, starting from the trigger pixel itself and filling only background cells.

    Key insight:
    - The visible blue cells are only a sampled trace of the real path. Triggers are validated against the invisible complete diagonal, so a magenta pixel can activate even when it sits beside a blank gap in the dash pattern.
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

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = [(dr, -dc), (-dr, dc)]

        for idx, (r, c) in enumerate(path):
            if not _is_visible_path_index(idx):
                continue
            if int(out[r, c]) == bg:
                out[r, c] = line_color

        for r, c in path[1:-1]:
            for ddr, ddc in perp_dirs:
                nr, nc = r + ddr, c + ddc
                if not _in_bounds(h, w, nr, nc):
                    continue
                v = int(grid[nr, nc])
                if v != bg and v != line_color:
                    key = (nr, nc, v)
                    triggers.setdefault(key, set()).update(perp_dirs)

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc

    return out


# --------------- generator (combo) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEGMENT_COUNT = (1, 4)
_SEG_LEN_RANGE = (3, 16)
_TRIGGERS_PER_SEG = (0, 2)
_SEG_LEN_OPTIONS = tuple(n for n in range(_SEG_LEN_RANGE[0], _SEG_LEN_RANGE[1] + 1) if n % 2 == 0)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 4)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 500


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    seg_len_options = [n for n in _SEG_LEN_OPTIONS if n <= max_seg_len]
    if not seg_len_options:
        return None

    occupied = set()
    reserved = set()
    non_line_positions = set()
    all_trigger_positions = set()
    trigger_zone = set()
    segments = []
    has_hidden_anchor_trigger = False

    for _ in range(n_segments):
        placed = False
        for _ in range(_SEGMENT_PLACE_RETRIES):
            slope_type = random.choice(["backslash", "slash"])
            seg_len = random.choice(seg_len_options)
            result = _try_place_segment(slope_type, seg_len, H, W, occupied, reserved)
            if result is None:
                continue

            path, dc = result
            perp_dirs = [(1, -dc), (-1, dc)]
            zone = _segment_zone(path, perp_dirs, H, W)
            if zone & non_line_positions:
                continue

            trigger_specs = _choose_adjacent_triggers(
                path,
                perp_dirs,
                H,
                W,
                occupied,
                reserved,
                all_trigger_positions,
                trigger_zone,
            )
            if trigger_specs is None:
                continue

            trigger_positions = [item["cell"] for item in trigger_specs]
            ray_cells = _compute_ray_cells(trigger_positions, perp_dirs, H, W)
            current_occupied = occupied | {path[0], path[-1]} | set(trigger_positions)
            if ray_cells & current_occupied:
                continue
            if ray_cells & reserved:
                continue

            segments.append(
                {
                    "ep0": path[0],
                    "ep1": path[-1],
                    "trigger_positions": trigger_positions,
                    "path": path,
                    "perp_dirs": perp_dirs,
                    "trigger_specs": trigger_specs,
                }
            )
            occupied.update([path[0], path[-1]])
            occupied.update(trigger_positions)
            non_line_positions.update(trigger_positions)
            all_trigger_positions.update(trigger_positions)
            reserved.update(path[1:-1])
            reserved.update(ray_cells)
            trigger_zone.update(zone)
            has_hidden_anchor_trigger = has_hidden_anchor_trigger or any(item["hidden"] for item in trigger_specs)
            placed = True
            break

        if not placed:
            continue

    if not segments or not all_trigger_positions or not has_hidden_anchor_trigger:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for segment in segments:
        inp[segment["ep0"]] = _LINE_COLOR
        inp[segment["ep1"]] = _LINE_COLOR
        for tp in segment["trigger_positions"]:
            inp[tp] = _TRIGGER_COLOR

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            cell = (r, c)
            if cell in occupied or cell in reserved or cell in trigger_zone:
                continue
            if any(_diag_aligned(cell, other) for other in all_line_positions):
                continue
            inp[cell] = _LINE_COLOR
            occupied.add(cell)
            all_line_positions.append(cell)
            break

    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            cell = (r, c)
            if cell in occupied or cell in reserved or cell in trigger_zone:
                continue
            if any(_diag_aligned(cell, other) for other in all_trigger_positions):
                continue
            inp[cell] = _TRIGGER_COLOR
            occupied.add(cell)
            non_line_positions.add(cell)
            all_trigger_positions.add(cell)
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
