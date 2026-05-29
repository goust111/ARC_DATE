# Combo variant of b5ca7ac4 (index 33)
# combo_spec: D9/1+D11/2
# Name: 两轮处理+跨组重叠检测

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste

# Design analysis for the fused variant:
# - Concrete trace: if azure rows are [3, 4, 6] and red rows are [10, 11, 12],
#   the original one-pass/same-group-only logic gives azure cols [0, 5, 10] and
#   red cols [17, 12, 7]. In the fused task, round 1 still starts at [0, 0, 0]
#   and [17, 17, 17], but round 2 runs iterative sweeps: azure settles to
#   [0, 5, 10], then the third red square first moves to 7 because of same-group
#   overlap and then to 2 because 7 still collides with the deep azure square at
#   col 10. The visible difference is a red frame pushed much farther inward by a
#   cross-group collision.
# - Complexity calibration: the visual vocabulary stays simple (same 5x5 frames,
#   same colors, same 5-cell column step) while the final columns now span a few
#   discrete depths. This makes the new behavior observable without introducing
#   too many degrees of freedom.
# - Assumption audit: the original code assumed two independent groups, at most
#   one inward step per side, and no need to revisit placements. All three are
#   false here. The solver therefore performs bounded iterative sweeps and the
#   generator must build layouts that remain solvable under that stronger rule.
# - Placement legality: every shift is by exactly one square width, always inward,
#   and is rejected if it would leave the grid. The generator uses a solvable row
#   template plus validation against the actual solver before returning an input.
# - Visual observability: accepted samples require both a deep placement
#   (depth >= 2) and at least one square whose final column changes only because
#   cross-group overlap is enabled. This prevents degeneration back to the source
#   task.
# - Visual cleanliness: extra squares are added only in benign rows so the core
#   collision band remains readable; outputs keep the same sparse framed-object
#   look as the source task.
# - New solver assumptions guaranteed by the generator: all squares remain
#   isolated 5x5 objects, row ordering is deterministic via row-first sorting,
#   and the sampled layouts always converge within the finite inward-depth limit.

# --------------- shared configuration ---------------

GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
INPUT_COL_RETRIES = 4
ROW_TEMPLATE_RETRIES = 120
GEN_RETRIES = 400
MAX_SETTLE_PASSES = 20


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    p = np.full((SQ, SQ), frame_color, dtype=int)
    p[1:-1, 1:-1] = inner_color
    return p


def _rows_overlap(r_a: int, r_b: int) -> bool:
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    return _rows_overlap(int(r1), int(r2)) and abs(int(c1) - int(c2)) < SQ


def _scan_squares(grid: np.ndarray) -> list[dict]:
    H, W = grid.shape
    squares: list[dict] = []
    for r0 in range(H - SQ + 1):
        for c0 in range(W - SQ + 1):
            patch = crop_bbox(grid, (int(r0), int(c0), int(r0 + SQ - 1), int(c0 + SQ - 1)))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_col = int(border[0])
            if frame_col not in {AZURE, RED}:
                continue
            inner = patch[1:-1, 1:-1]
            if not bool((inner != frame_col).any()):
                continue
            squares.append(
                {
                    "frame": frame_col,
                    "r0": int(r0),
                    "c0": int(c0),
                    "patch": patch,
                }
            )
    return squares


def _sort_groups(squares: list[dict]) -> tuple[list[dict], list[dict]]:
    azure = sorted(
        [dict(s) for s in squares if int(s["frame"]) == AZURE],
        key=lambda s: (int(s["r0"]), int(s["c0"])),
    )
    red = sorted(
        [dict(s) for s in squares if int(s["frame"]) == RED],
        key=lambda s: (-int(s["r0"]), -int(s["c0"])),
    )
    return azure, red


def _overlaps_any(row: int, col: int, positions: list[tuple[int, int]]) -> bool:
    return any(_rects_overlap(int(row), int(col), int(r), int(c)) for r, c in positions)


def _resolve_columns(
    azure: list[dict],
    red: list[dict],
    width: int,
    *,
    include_cross_group: bool,
) -> tuple[list[dict], list[dict]] | None:
    azure_state = sorted(
        [dict(s) for s in azure],
        key=lambda s: (int(s["r0"]), int(s.get("c0", 0))),
    )
    red_state = sorted(
        [dict(s) for s in red],
        key=lambda s: (-int(s["r0"]), -int(s.get("c0", 0))),
    )
    for s in azure_state:
        s["dest_c"] = 0
    for s in red_state:
        s["dest_c"] = int(width - SQ)

    for _ in range(MAX_SETTLE_PASSES):
        changed = False

        fixed_azure: list[tuple[int, int]] = []
        red_positions = [(int(s["r0"]), int(s["dest_c"])) for s in red_state] if include_cross_group else []
        for s in azure_state:
            dest_c = int(s["dest_c"])
            while _overlaps_any(int(s["r0"]), dest_c, fixed_azure) or _overlaps_any(int(s["r0"]), dest_c, red_positions):
                dest_c += SQ
                if dest_c > int(width - SQ):
                    return None
            if dest_c != int(s["dest_c"]):
                changed = True
                s["dest_c"] = dest_c
            fixed_azure.append((int(s["r0"]), dest_c))

        fixed_red: list[tuple[int, int]] = []
        azure_positions = [(int(s["r0"]), int(s["dest_c"])) for s in azure_state] if include_cross_group else []
        for s in red_state:
            dest_c = int(s["dest_c"])
            while _overlaps_any(int(s["r0"]), dest_c, fixed_red) or _overlaps_any(int(s["r0"]), dest_c, azure_positions):
                dest_c -= SQ
                if dest_c < 0:
                    return None
            if dest_c != int(s["dest_c"]):
                changed = True
                s["dest_c"] = dest_c
            fixed_red.append((int(s["r0"]), dest_c))

        if not changed:
            return azure_state, red_state

    return None


def _render_output(shape: tuple[int, int], bg: int, azure: list[dict], red: list[dict]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for s in azure + red:
        paste(out, s["patch"], int(s["r0"]), int(s["dest_c"]), transparent=int(bg))
    return out


def _depth(frame: int, dest_c: int, width: int) -> int:
    if int(frame) == AZURE:
        return int(dest_c // SQ)
    return int((width - SQ - dest_c) // SQ)


def _behavior_metrics(azure: list[dict], red: list[dict], width: int) -> dict[str, int]:
    full = _resolve_columns(azure, red, width, include_cross_group=True)
    same_only = _resolve_columns(azure, red, width, include_cross_group=False)
    if full is None or same_only is None:
        return {"cross_group_adjusted": -1, "max_depth": -1}

    azure_full, red_full = full
    azure_same, red_same = same_only
    cross_group_adjusted = 0
    for s_full, s_same in zip(azure_full + red_full, azure_same + red_same):
        if int(s_full["dest_c"]) != int(s_same["dest_c"]):
            cross_group_adjusted += 1

    max_depth = max(
        [_depth(int(s["frame"]), int(s["dest_c"]), width) for s in azure_full + red_full],
        default=0,
    )
    return {
        "cross_group_adjusted": int(cross_group_adjusted),
        "max_depth": int(max_depth),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather each colored 5x5 frame onto its side, but solve the column stacking in two rounds.

    Input structure:
    - The background is the majority color.
    - The foreground consists of isolated 5x5 framed squares with frame color teal (8) or red (2).
    - Every square keeps its original top row; only the destination column changes.

    Rule:
    - Round 1 ignores collisions: teal frames are assigned to the left edge and red frames to the right edge.
    - Round 2 repeatedly adjusts columns until stable: teal is swept from top to bottom and shifts right by 5 when blocked; red is swept from bottom to top and shifts left by 5 when blocked.
    - A block can come from either the same color group already fixed earlier in that sweep or from the other color group at its current column, so cross-group overlaps also force extra inward motion.

    Key insight:
    The output is not decided by a single pass. First everything snaps to the edge, then iterative sweeps resolve both same-group and cross-group rectangle collisions, which can push a frame multiple square-widths inward.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    squares = _scan_squares(g)
    azure, red = _sort_groups(squares)
    resolved = _resolve_columns(azure, red, W, include_cross_group=True)
    if resolved is None:
        raise ValueError("No valid stable placement for fused stacking rule")
    azure_final, red_final = resolved
    return _render_output((H, W), bg, azure_final, red_final)


# --------------- generator ---------------


def _pick_rows(
    count: int,
    *,
    blocked_rows: list[int],
    extra_blocked_rows: list[int] | None = None,
) -> list[int] | None:
    if count == 0:
        return []
    chosen: list[int] = []
    blocked = list(blocked_rows)
    if extra_blocked_rows is not None:
        blocked.extend(extra_blocked_rows)
    candidates = list(range(GRID_H - SQ + 1))
    random.shuffle(candidates)
    for r in candidates:
        if any(_rows_overlap(r, br) for br in blocked):
            continue
        if any(_rows_overlap(r, cr) for cr in chosen):
            continue
        chosen.append(int(r))
        if len(chosen) == count:
            return chosen
    return None


def _sample_rows(n_azure: int, n_red: int) -> tuple[list[int], list[int]] | None:
    for _ in range(ROW_TEMPLATE_RETRIES):
        anchor = random.randint(0, 7)
        azure_rows = [anchor + d for d in random.choice([(0, 1, 3), (0, 2, 3), (0, 2, 4)])]
        red_rows = [anchor + d for d in random.choice([(7, 8, 9), (7, 8, 10), (6, 8, 9)])]
        azure_rows = sorted(azure_rows)
        red_rows = sorted(red_rows)

        extra_azure = _pick_rows(
            n_azure - 3,
            blocked_rows=azure_rows,
            extra_blocked_rows=[min(red_rows)],
        )
        if extra_azure is None:
            continue
        extra_red = _pick_rows(
            n_red - 3,
            blocked_rows=red_rows,
            extra_blocked_rows=[],
        )
        if extra_red is None:
            continue

        all_azure = sorted(azure_rows + extra_azure)
        all_red = sorted(red_rows + extra_red)

        azure_specs = [{"frame": AZURE, "r0": int(r), "c0": 0, "patch": np.zeros((SQ, SQ), dtype=int)} for r in all_azure]
        red_specs = [{"frame": RED, "r0": int(r), "c0": GRID_W - SQ, "patch": np.zeros((SQ, SQ), dtype=int)} for r in all_red]
        metrics = _behavior_metrics(azure_specs, red_specs, GRID_W)
        if metrics["cross_group_adjusted"] >= 1 and metrics["max_depth"] >= 2:
            return all_azure, all_red

    return None


def _assign_input_columns(squares: list[dict]) -> bool:
    # A three-lane source layout keeps every input frame isolated while staying far
    # from the output lanes, so the motion remains visible.
    source_cols = [1, 8, 16]
    random.shuffle(source_cols)
    order = list(range(len(squares)))
    order.sort(
        key=lambda i: (
            -sum(abs(int(squares[i]["r0"]) - int(squares[j]["r0"])) <= SQ for j in range(len(squares)) if j != i),
            int(squares[i]["r0"]),
        )
    )
    assigned: dict[int, int] = {}

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        square = squares[idx]
        for c in source_cols:
            if int(c) == int(square["dest_c"]):
                continue
            if any(
                abs(int(square["r0"]) - int(squares[j]["r0"])) <= SQ and int(assigned[j]) == int(c)
                for j in assigned
            ):
                continue
            assigned[idx] = int(c)
            if backtrack(pos + 1):
                return True
            del assigned[idx]
        return False

    ok = backtrack(0)
    if not ok:
        return False
    for idx, c in assigned.items():
        squares[idx]["c0"] = int(c)
    return True


def _build_squares_from_rows(
    bg: int,
    azure_rows: list[int],
    red_rows: list[int],
) -> tuple[list[dict], list[dict]] | None:
    avail_azure = [c for c in range(10) if c not in {bg, AZURE}]
    avail_red = [c for c in range(10) if c not in {bg, RED}]
    if len(avail_azure) < len(azure_rows) or len(avail_red) < len(red_rows):
        return None

    azure_colors = random.sample(avail_azure, len(azure_rows))
    red_colors = random.sample(avail_red, len(red_rows))
    azure = [
        {
            "frame": AZURE,
            "r0": int(r),
            "c0": 0,
            "patch": _make_patch(AZURE, int(col)),
        }
        for r, col in zip(azure_rows, azure_colors)
    ]
    red = [
        {
            "frame": RED,
            "r0": int(r),
            "c0": GRID_W - SQ,
            "patch": _make_patch(RED, int(col)),
        }
        for r, col in zip(red_rows, red_colors)
    ]
    resolved = _resolve_columns(azure, red, GRID_W, include_cross_group=True)
    if resolved is None:
        return None
    azure_final, red_final = resolved
    metrics = _behavior_metrics(azure_final, red_final, GRID_W)
    if metrics["cross_group_adjusted"] < 1 or metrics["max_depth"] < 2:
        return None
    return azure_final, red_final


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    row_plan = _sample_rows(n_azure, n_red)
    if row_plan is None:
        return None
    azure_rows, red_rows = row_plan

    built = _build_squares_from_rows(bg, azure_rows, red_rows)
    if built is None:
        return None
    azure_final, red_final = built
    all_squares = azure_final + red_final

    for _ in range(INPUT_COL_RETRIES):
        input_squares = [dict(s) for s in all_squares]
        if not _assign_input_columns(input_squares):
            continue

        inp = np.full((GRID_H, GRID_W), bg, dtype=int)
        for s in input_squares:
            paste(inp, s["patch"], int(s["r0"]), int(s["c0"]))

        expected = _render_output((GRID_H, GRID_W), bg, azure_final, red_final)
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid fused sample after {GEN_RETRIES} attempts")
