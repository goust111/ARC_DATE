# Combo variant of b5ca7ac4 (index 40)
# combo_spec: D9/0+D11/2
# Name: 交替处理+跨组重叠检测
#
# Design analysis for the fused rule
# - Concrete trace: with teal rows [b, b+1, b+2, b+3] and red rows [b+9, b+8, b+6],
#   the alternating queue is A(b), R(b+9), A(b+1), R(b+8), A(b+2), R(b+6), A(b+3).
#   Teal occupies cols 0, 5, 10; red occupies 17, 12; then the last red wants 7 under
#   same-group-only logic but is forced to 2 because teal at col 10 already occupies the
#   overlapping band. This makes the fusion behavior measurable instead of cosmetic.
# - Complexity calibration: keeping a fixed 4-teal/3-red core avoids ambiguity while still
#   showing repeated stacking on both sides. Colors stay human-visible and rows remain the
#   only ordering key; nothing depends on hidden numeric color ordering.
# - Implicit assumptions audit: the original code assumed two independent passes, overlap
#   checks only inside each group, and only two landing columns per side. Under D11/2 that
#   third assumption breaks: on a 22x22 grid, teal {0,5} and red {12,17} never intersect,
#   so cross-group checks would be inert. The fused solver therefore searches every inward
#   slot spaced by 5 cells until the full 5x5 footprint avoids all earlier placements.
# - Placement legality: generator rows are chosen from a translated template already known
#   to admit a valid global layout; input columns are then solved by backtracking so no
#   two 5x5 patches overlap in the source grid.
# - Visual observability / cleanliness: the output still consists of intact 5x5 frames,
#   but some late frames reach deeper center lanes. The effect is easy to see because the
#   displaced frame keeps its row while moving farther inward than same-group stacking alone.
# - Generator anti-degeneration guard: every generated sample is compared against a baseline
#   "separate groups, no cross-group collision" placement, and samples are rejected unless
#   at least one frame lands in a different column.

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
GEN_RETRIES = 300

RED_ROW_PATTERNS = [
    (9, 8, 6),
    (10, 8, 6),
    (10, 9, 6),
]


def _rows_overlap(r_a: int, r_b: int) -> bool:
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    return _rows_overlap(r1, r2) and abs(int(c1) - int(c2)) < SQ


def _candidate_cols(frame: int, width: int) -> list[int]:
    if int(frame) == AZURE:
        return list(range(0, int(width) - SQ + 1, SQ))
    return list(range(int(width) - SQ, -1, -SQ))


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((SQ, SQ), int(frame_color), dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
    return patch


def _detect_squares(grid: np.ndarray) -> list[dict]:
    squares: list[dict] = []
    H, W = grid.shape
    for r0 in range(H - SQ + 1):
        for c0 in range(W - SQ + 1):
            patch = crop_bbox(grid, (r0, c0, r0 + SQ - 1, c0 + SQ - 1))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_col = int(border[0])
            if frame_col not in {AZURE, RED}:
                continue
            inner = patch[1:-1, 1:-1]
            if not bool((inner != frame_col).any()):
                continue
            squares.append({"frame": frame_col, "r0": int(r0), "c0": int(c0), "patch": patch})
    return squares


def _combo_order(squares: list[dict]) -> list[dict]:
    azure = sorted(
        (s for s in squares if int(s["frame"]) == AZURE),
        key=lambda s: (int(s["r0"]), int(s.get("c0", 0))),
    )
    red = sorted(
        (s for s in squares if int(s["frame"]) == RED),
        key=lambda s: (-int(s["r0"]), int(s.get("c0", 0))),
    )

    order: list[dict] = []
    for idx in range(max(len(azure), len(red))):
        if idx < len(azure):
            order.append(azure[idx])
        if idx < len(red):
            order.append(red[idx])
    return order


def _place_sequence(order: list[dict], width: int) -> list[dict]:
    placed: list[dict] = []
    for sq in order:
        r0 = int(sq["r0"])
        chosen = None
        for dest_c in _candidate_cols(int(sq["frame"]), int(width)):
            if all(not _rects_overlap(r0, dest_c, int(prev["r0"]), int(prev["dest_c"])) for prev in placed):
                chosen = int(dest_c)
                break
        if chosen is None:
            raise ValueError("No legal inward slot for fused placement")
        placed.append({**sq, "dest_c": chosen})
    return placed


def _place_group_only(squares: list[dict], width: int) -> list[dict]:
    placed: list[dict] = []
    for sq in squares:
        r0 = int(sq["r0"])
        chosen = None
        for dest_c in _candidate_cols(int(sq["frame"]), int(width)):
            if all(not _rects_overlap(r0, dest_c, int(prev["r0"]), int(prev["dest_c"])) for prev in placed):
                chosen = int(dest_c)
                break
        if chosen is None:
            raise ValueError("No legal same-group slot")
        placed.append({**sq, "dest_c": chosen})
    return placed


def _baseline_layout(squares: list[dict], width: int) -> dict[tuple[int, int], int]:
    azure = sorted(
        (s for s in squares if int(s["frame"]) == AZURE),
        key=lambda s: (int(s["r0"]), int(s.get("c0", 0))),
    )
    red = sorted(
        (s for s in squares if int(s["frame"]) == RED),
        key=lambda s: (-int(s["r0"]), int(s.get("c0", 0))),
    )
    base = _place_group_only(list(azure), width) + _place_group_only(list(red), width)
    return {(int(s["frame"]), int(s["r0"])): int(s["dest_c"]) for s in base}


def _fusion_metric(squares: list[dict], width: int) -> int:
    combo = _place_sequence(_combo_order(squares), width)
    baseline = _baseline_layout(squares, width)
    return sum(1 for s in combo if baseline[(int(s["frame"]), int(s["r0"]))] != int(s["dest_c"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repack the teal and red 5x5 frames with an alternating global queue.

    Input structure:
    - The background is the most frequent color.
    - Foreground objects are scattered 5x5 bordered squares with teal (8) or red (2) frames and non-frame 3x3 interiors.

    Rule:
    - Keep every square on its original top row.
    - Sort teal squares from top to bottom, sort red squares from bottom to top, then place them by alternating teal, red, teal, red...
    - A teal square tests columns 0, 5, 10, ... from the left; a red square tests columns W-5, W-10, W-15, ... from the right, taking the first slot whose full 5x5 area avoids every square already placed, including the opposite color.

    Key insight:
    - The alternating queue lets one color occupy centerward lanes before the other color is finished, so global rectangle-overlap checks across both groups can push later squares farther inward than separate per-color stacking would.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    out = np.full((H, W), bg, dtype=int)
    placed = _place_sequence(_combo_order(_detect_squares(g)), W)
    for sq in placed:
        paste(out, sq["patch"], int(sq["r0"]), int(sq["dest_c"]), transparent=bg)
    return out


def _build_core_rows() -> tuple[list[int], list[int]]:
    pattern = random.choice(RED_ROW_PATTERNS)
    max_offset = max(3, *pattern)
    base = random.randint(0, (GRID_H - SQ) - max_offset)
    azure_rows = [base + offset for offset in (0, 1, 2, 3)]
    red_rows = sorted([base + offset for offset in pattern], reverse=True)
    return azure_rows, red_rows


def _backtracking_columns(rows: list[int], forbidden: list[int]) -> list[int] | None:
    n = len(rows)
    neighbors = [
        [j for j in range(n) if i != j and _rows_overlap(rows[i], rows[j])]
        for i in range(n)
    ]

    def search(domains: list[list[int]]) -> list[int] | None:
        assignment = [None] * n

        def dfs() -> bool:
            pending = [i for i in range(n) if assignment[i] is None]
            if not pending:
                return True

            idx = min(pending, key=lambda i: (len(domains[i]), -len(neighbors[i]), i))
            options = domains[idx][:]
            random.shuffle(options)
            for col in options:
                if any(
                    assignment[j] is not None and abs(int(col) - int(assignment[j])) < SQ
                    for j in neighbors[idx]
                ):
                    continue
                assignment[idx] = int(col)
                if dfs():
                    return True
                assignment[idx] = None
            return False

        if dfs():
            return [int(v) for v in assignment]
        return None

    full = list(range(GRID_W - SQ + 1))
    strict_domains = [[c for c in full if c != int(forbidden[i])] for i in range(n)]
    result = search(strict_domains)
    if result is not None:
        return result
    relaxed_domains = [full[:] for _ in range(n)]
    return search(relaxed_domains)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    azure_rows, red_rows = _build_core_rows()

    azure_colors = random.sample([c for c in range(10) if c not in {bg, AZURE}], len(azure_rows))
    red_colors = random.sample([c for c in range(10) if c not in {bg, RED}], len(red_rows))

    squares: list[dict] = []
    for row, color in zip(azure_rows, azure_colors):
        squares.append({"frame": AZURE, "r0": int(row), "patch": _make_patch(AZURE, int(color))})
    for row, color in zip(red_rows, red_colors):
        squares.append({"frame": RED, "r0": int(row), "patch": _make_patch(RED, int(color))})

    metric = _fusion_metric(squares, GRID_W)
    if metric <= 0:
        return None

    placed = _place_sequence(_combo_order(squares), GRID_W)
    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    for sq in placed:
        paste(output, sq["patch"], int(sq["r0"]), int(sq["dest_c"]))

    rows = [int(sq["r0"]) for sq in squares]
    forbidden = []
    combo_by_key = {(int(s["frame"]), int(s["r0"])): int(s["dest_c"]) for s in placed}
    for sq in squares:
        forbidden.append(combo_by_key[(int(sq["frame"]), int(sq["r0"]))])

    input_cols = _backtracking_columns(rows, forbidden)
    if input_cols is None:
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for sq, col in zip(squares, input_cols):
        paste(inp, sq["patch"], int(sq["r0"]), int(col))

    if np.array_equal(main(inp), output):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
