from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
SEPARATOR = 5
PANEL_COLORS = (1, 3, 4, 6, 7, 8, 9)
MODES = ("down_right", "down_left")
MAX_GENERATION_ATTEMPTS = 600


def _residue(r: int, c: int, period: int, mode: str) -> int:
    if mode == "down_right":
        return (int(r) + int(c)) % int(period)
    if mode == "down_left":
        return (int(c) - int(r)) % int(period)
    raise AssertionError(f"unknown mode: {mode}")


def _render_panel(shape: tuple[int, int], table: list[int], mode: str) -> np.ndarray:
    h, w = map(int, shape)
    period = len(table)
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = int(table[_residue(r, c, period, mode)])
    return out


def _separator_index(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert w % 2 == 1, "expected odd grid width"
    mid = w // 2
    assert mid > 0 and mid < w - 1, "expected non-empty panels"
    sep_color = int(g[0, mid])
    assert np.all(g[:, mid] == sep_color), "expected a full separator column"
    return mid


def _infer_left_cycle(panel: np.ndarray) -> tuple[str, list[int]]:
    visible = np.argwhere(panel != BLACK)
    assert len(visible) > 0, "expected visible evidence"

    colors = sorted(set(map(int, panel[panel != BLACK].ravel())))
    period = len(colors)
    assert period >= 3, "expected at least three diagonal residues"

    matches: list[tuple[str, list[int]]] = []
    for mode in MODES:
        table = [-1] * period
        ok = True
        for r, c in visible:
            key = _residue(int(r), int(c), period, mode)
            color = int(panel[r, c])
            if table[key] == -1:
                table[key] = color
            elif table[key] != color:
                ok = False
                break
        if ok and all(color != -1 for color in table) and len(set(table)) == period and sorted(table) == colors:
            matches.append((mode, table))

    assert len(matches) == 1, f"expected one uniquely implied cycle, got {len(matches)}"
    return matches[0]


def _infer_table(panel: np.ndarray, period: int, mode: str) -> list[int]:
    visible = np.argwhere(panel != BLACK)
    assert len(visible) > 0, "expected visible evidence"

    table = [-1] * int(period)
    for r, c in visible:
        key = _residue(int(r), int(c), int(period), mode)
        color = int(panel[r, c])
        if table[key] == -1:
            table[key] = color
        else:
            assert table[key] == color, "residue class must have one color"
    assert all(color != -1 for color in table), "every residue class must remain visible"
    return table


def _best_shift(left_table: list[int], right_table: list[int]) -> tuple[int, list[bool]]:
    period = len(left_table)
    scores: list[tuple[int, list[bool]]] = []
    for shift in range(period):
        matches = [int(left_table[(k - shift) % period]) == int(right_table[k]) for k in range(period)]
        scores.append((int(sum(matches)), matches))

    best = max(score for score, _matches in scores)
    winners = [shift for shift, (score, _matches) in enumerate(scores) if score == best]
    assert len(winners) == 1, f"expected one unique best shift, got {winners}"
    shift = winners[0]
    return shift, scores[shift][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the two holed diagonal cycles by phase, then mark only the unalignable residue classes red.

    Input structure:
    - The grid has odd width, with one full-height separator column in the center.
    - The left and right sides are same-size panels with black holes punched out of diagonal periodic color patterns.
    - Both panels use the same diagonal direction and period, but the right panel is phase-shifted and has a few residue classes recolored.

    Rule:
    - Split at the center separator and use the visible cells to reconstruct the left panel's diagonal direction and residue-to-color cycle.
    - Using that same direction and period, reconstruct the right panel's residue-to-color table from its visible cells.
    - Find the unique cyclic shift that maximizes matches between the shifted left table and the right table.
    - Output a grid the size of the right panel: residue classes that match under the best shift use the restored right-panel color, and residue classes that still disagree become full red diagonals.

    Key insight:
    - The apparent left-vs-right pixel disagreements are mostly fake, caused by a global phase offset; only after aligning the residue tables does the true set of modified diagonals stand out.
    """
    g = np.asarray(grid, dtype=int)
    mid = _separator_index(g)
    left = g[:, :mid]
    right = g[:, mid + 1 :]

    mode, left_table = _infer_left_cycle(left)
    right_table = _infer_table(right, len(left_table), mode)
    shift, residue_matches = _best_shift(left_table, right_table)

    out = np.zeros_like(right)
    period = len(left_table)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            key = _residue(r, c, period, mode)
            if residue_matches[key]:
                out[r, c] = int(right_table[key])
            else:
                out[r, c] = RED
    return out


def _random_right_table(left_table: list[int], shift: int) -> tuple[list[int], list[int]]:
    period = len(left_table)
    right_table = [int(left_table[(k - shift) % period]) for k in range(period)]
    mismatch_count = random.randint(1, max(1, period // 3))
    mismatch_residues = random.sample(range(period), mismatch_count)

    for key in mismatch_residues:
        options = [color for color in PANEL_COLORS if color != right_table[key] and color != RED and color != SEPARATOR]
        right_table[key] = int(random.choice(options))
    return right_table, mismatch_residues


def _punch_holes(panel: np.ndarray, period: int, mode: str) -> np.ndarray:
    h, w = panel.shape
    coords_by_residue = [[] for _ in range(period)]
    for r in range(h):
        for c in range(w):
            coords_by_residue[_residue(r, c, period, mode)].append((r, c))

    keep = np.zeros((h, w), dtype=bool)
    mandatory: set[tuple[int, int]] = set()
    for coords in coords_by_residue:
        cell = random.choice(coords)
        mandatory.add(cell)
        keep[cell] = True

    keep_prob = random.uniform(0.45, 0.72)
    for r in range(h):
        for c in range(w):
            if (r, c) not in mandatory and random.random() < keep_prob:
                keep[r, c] = True

    if int(np.count_nonzero(~keep)) < max(2, period):
        extras = [(r, c) for r in range(h) for c in range(w) if (r, c) not in mandatory]
        random.shuffle(extras)
        for r, c in extras[: max(2, period) - int(np.count_nonzero(~keep))]:
            keep[r, c] = False

    out = panel.copy()
    out[~keep] = BLACK
    return out


def _expected_output(shape: tuple[int, int], right_table: list[int], mode: str, mismatch_residues: list[int]) -> np.ndarray:
    h, w = map(int, shape)
    period = len(right_table)
    mismatches = set(map(int, mismatch_residues))
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            key = _residue(r, c, period, mode)
            out[r, c] = RED if key in mismatches else int(right_table[key])
    return out


def _validate_semantics(
    inp: np.ndarray,
    mode: str,
    left_table: list[int],
    right_table: list[int],
    shift: int,
    mismatch_residues: list[int],
) -> bool:
    mid = _separator_index(inp)
    left = inp[:, :mid]
    right = inp[:, mid + 1 :]

    try:
        got_mode, got_left = _infer_left_cycle(left)
        got_right = _infer_table(right, len(left_table), got_mode)
        got_shift, residue_matches = _best_shift(got_left, got_right)
    except Exception:
        return False

    if got_mode != mode or got_left != left_table or got_right != right_table or got_shift != shift:
        return False

    expected_match_mask = [key not in set(mismatch_residues) for key in range(len(left_table))]
    if residue_matches != expected_match_mask:
        return False

    if shift == 0:
        return False
    zero_shift_score = sum(int(left_table[k] == right_table[k]) for k in range(len(left_table)))
    best_score = sum(int(flag) for flag in residue_matches)
    if zero_shift_score >= best_score:
        return False
    if best_score <= 0 or best_score >= len(left_table):
        return False

    out = main(inp)
    if not np.array_equal(out, _expected_output(right.shape, right_table, mode, mismatch_residues)):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    period = random.randint(4, min(7, len(PANEL_COLORS)))
    panel_h = random.randint(max(6, period + 1), 14)
    panel_w = random.randint(max(6, period + 1), 14)
    mode = random.choice(MODES)

    left_table = random.sample(PANEL_COLORS, period)
    shift = random.randint(1, period - 1)
    right_table, mismatch_residues = _random_right_table(left_table, shift)
    if len(set(mismatch_residues)) != len(mismatch_residues):
        return None

    try:
        got_shift, match_mask = _best_shift(left_table, right_table)
    except Exception:
        return None
    if got_shift != shift:
        return None
    mismatch_residues = [k for k, ok in enumerate(match_mask) if not ok]
    if not mismatch_residues:
        return None

    left_full = _render_panel((panel_h, panel_w), left_table, mode)
    right_full = _render_panel((panel_h, panel_w), right_table, mode)
    left = _punch_holes(left_full, period, mode)
    right = _punch_holes(right_full, period, mode)

    grid = np.full((panel_h, panel_w * 2 + 1), SEPARATOR, dtype=int)
    grid[:, :panel_w] = left
    grid[:, panel_w + 1 :] = right

    if max(grid.shape) > 30:
        return None
    if not _validate_semantics(grid, mode, left_table, right_table, shift, mismatch_residues):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
