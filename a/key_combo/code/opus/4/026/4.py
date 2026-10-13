import random
from collections import Counter

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
TEAL = 8
MARKERS = {RED, TEAL}
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        if _in_bounds(mask.shape, rr, cc) and bool(mask[rr, cc]):
            out.append((dr, dc))
    return out


def _line_runs(line: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(line):
        if bool(value) and start is None:
            start = i
        if start is not None and (not bool(value) or i == len(line) - 1):
            end = i if bool(value) else i - 1
            runs.append((start, end))
            start = None
    return runs


def _grid_with_mask(base: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    out[out == GREY] = BLACK
    out[mask] = GREY
    return out


def _corridor_line_ok(grid: np.ndarray, kind: str, idx: int) -> bool:
    g = np.asarray(grid, dtype=int)
    mask = g == GREY
    h, w = g.shape

    if kind == "row":
        r = int(idx)
        if int(g[r, 0]) not in MARKERS or int(g[r, w - 1]) not in MARKERS:
            return False
        line = g[r, 1 : w - 1]
        if np.any((line != BLACK) & (line != GREY)):
            return False
        grey_cols = np.flatnonzero(line == GREY) + 1
        if grey_cols.size == 0:
            return True
        if int(grey_cols[-1] - grey_cols[0] + 1) != int(grey_cols.size):
            return False
        if r == 0 or r == h - 1:
            return False
        for c in grey_cols:
            if not bool(mask[r - 1, c] and mask[r + 1, c]):
                return False
        return True

    c = int(idx)
    if int(g[0, c]) not in MARKERS or int(g[h - 1, c]) not in MARKERS:
        return False
    line = g[1 : h - 1, c]
    if np.any((line != BLACK) & (line != GREY)):
        return False
    grey_rows = np.flatnonzero(line == GREY) + 1
    if grey_rows.size == 0:
        return True
    if int(grey_rows[-1] - grey_rows[0] + 1) != int(grey_rows.size):
        return False
    if c == 0 or c == w - 1:
        return False
    for r in grey_rows:
        if not bool(mask[r, c - 1] and mask[r, c + 1]):
            return False
    return True


def _detect_corridors(grid: np.ndarray) -> list[tuple[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out: list[tuple[str, int]] = []
    for r in range(1, h - 1):
        if _corridor_line_ok(g, "row", r):
            out.append(("row", r))
    for c in range(1, w - 1):
        if _corridor_line_ok(g, "col", c):
            out.append(("col", c))
    return out


def _opposite_edge(mask: np.ndarray, r: int, c: int, dr: int, dc: int) -> tuple[int, int]:
    if dr:
        rows = np.flatnonzero(mask[:, c])
        return (int(rows.max() if dr > 0 else rows.min()), int(c))
    cols = np.flatnonzero(mask[r])
    return (int(r), int(cols.max() if dc > 0 else cols.min()))


def _outside_opposite_edge(
    mask: np.ndarray, r: int, c: int, dr: int, dc: int
) -> tuple[int, int] | None:
    if dr:
        rows = np.flatnonzero(mask[:, c])
        rr = int(rows.max() + 1 if dr > 0 else rows.min() - 1)
        return (rr, int(c)) if _in_bounds(mask.shape, rr, c) else None
    cols = np.flatnonzero(mask[r])
    cc = int(cols.max() + 1 if dc > 0 else cols.min() - 1)
    return (int(r), cc) if _in_bounds(mask.shape, r, cc) else None


def _list_candidates(mask: np.ndarray) -> list[dict]:
    mask = np.asarray(mask, dtype=bool)
    candidates: list[dict] = []

    tab_sources: list[tuple[int, int]] = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not bool(mask[r, c]):
                continue
            neighbors = _neighbors4(mask, r, c)
            if len(neighbors) != 1:
                continue
            dr, dc = neighbors[0]
            target = _opposite_edge(mask, r, c, dr, dc)
            if target == (r, c):
                continue
            tab_sources.append((r, c))
            candidates.append(
                {
                    "kind": "tab",
                    "source": (r, c),
                    "target": target,
                    "value": False,
                }
            )

    tab_cells = set(tab_sources)
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if bool(mask[r, c]):
                continue
            fg_dirs: list[tuple[int, int]] = []
            bg_dirs: list[tuple[int, int]] = []
            for dr, dc in DIRS4:
                rr, cc = r + dr, c + dc
                if _in_bounds(mask.shape, rr, cc) and bool(mask[rr, cc]):
                    fg_dirs.append((dr, dc))
                else:
                    bg_dirs.append((dr, dc))
            if len(fg_dirs) != 3 or len(bg_dirs) != 1:
                continue
            outside_dr, outside_dc = bg_dirs[0]
            dr, dc = -outside_dr, -outside_dc
            side_a = (r - dc, c + dr)
            side_b = (r + dc, c - dr)
            if side_a in tab_cells or side_b in tab_cells:
                continue
            target = _outside_opposite_edge(mask, r, c, dr, dc)
            if target is None or bool(mask[target]):
                continue
            candidates.append(
                {
                    "kind": "notch",
                    "source": (r, c),
                    "target": target,
                    "value": True,
                }
            )

    unique: dict[tuple[str, tuple[int, int], tuple[int, int]], dict] = {}
    for candidate in candidates:
        key = (candidate["kind"], candidate["source"], candidate["target"])
        unique[key] = candidate
    return sorted(unique.values(), key=lambda item: (item["kind"], item["source"], item["target"]))


def _candidate_forbidden(
    base: np.ndarray, current_corridors: list[tuple[str, int]], candidate: dict
) -> bool:
    temp = np.asarray(base, dtype=int).copy()
    tr, tc = candidate["target"]
    temp[tr, tc] = GREY if candidate["value"] else BLACK
    return any(not _corridor_line_ok(temp, kind, idx) for kind, idx in current_corridors)


def _solve_structure(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[tuple[str, int]]]:
    g = np.asarray(grid, dtype=int)
    mask = g == GREY
    current_corridors = _detect_corridors(g)
    candidates = _list_candidates(mask)
    status: list[dict] = []
    out = g.copy()

    for candidate in candidates:
        blocked = _candidate_forbidden(g, current_corridors, candidate)
        enriched = dict(candidate)
        enriched["blocked"] = blocked
        status.append(enriched)
        if not blocked:
            tr, tc = candidate["target"]
            out[tr, tc] = GREY if candidate["value"] else BLACK

    return out, status, current_corridors


def main(grid: np.ndarray) -> np.ndarray:
    """Selectively mirror tabs and notches across a thick grey shape while preserving existing edge-to-edge corridors, then fill surviving corridors green.

    Input structure:
    - The grid has a black background, one thick grey rectilinear object, and red/teal marker cells on selected outer border positions.
    - The grey object contains one-pixel tabs on its boundary and one-pixel notches cut into its boundary.
    - Some marked rows or columns form current corridors: their opposite border endpoints are colored, and their interior contains only black cells or a single orthogonally supported grey body span.

    Rule:
    - Detect every tab as a grey boundary pixel with exactly one grey 4-neighbor, and detect every notch as a black cell with exactly three grey 4-neighbors.
    - For each tab, project through the same row or column and remove the opposite edge pixel; for each notch, project to the opposite side and add one grey pixel just outside that edge.
    - Reject any candidate change that would destroy any currently valid corridor, then execute all remaining changes and fill every black interior cell of every surviving corridor with green.

    Key insight:
    - The opposite-side tab/notch completion is not unconditional: corridors are identified on the input first, and those protected spans veto any symmetry completion that would place a new boundary irregularity on them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    bg = _get_bg(g)
    assert bg == BLACK, "expected black background"

    out, _status, _current = _solve_structure(g)
    final_corridors = _detect_corridors(out)
    for kind, idx in final_corridors:
        if kind == "row":
            r = idx
            for c in range(1, out.shape[1] - 1):
                if int(out[r, c]) == BLACK:
                    out[r, c] = GREEN
        else:
            c = idx
            for r in range(1, out.shape[0] - 1):
                if int(out[r, c]) == BLACK:
                    out[r, c] = GREEN
    return out


def _choose_non_adjacent(values: list[int], count: int) -> list[int] | None:
    pool = values[:]
    random.shuffle(pool)
    chosen: list[int] = []
    for value in pool:
        if all(abs(value - other) >= 2 for other in chosen):
            chosen.append(value)
        if len(chosen) == count:
            return sorted(chosen)
    return None


def _sample_rect(h: int, w: int) -> tuple[int, int, int, int] | None:
    target_lo = int(0.40 * h * w)
    target_hi = int(0.60 * h * w)
    choices: list[tuple[int, int, int, int]] = []
    for rh in range(8, h - 2):
        for rw in range(8, w - 2):
            area = rh * rw
            if not (target_lo <= area <= target_hi):
                continue
            for r0 in range(2, h - rh):
                for c0 in range(2, w - rw):
                    r1 = r0 + rh - 1
                    c1 = c0 + rw - 1
                    if r1 + 1 >= h - 1 or c1 + 1 >= w - 1:
                        continue
                    choices.append((r0, c0, r1, c1))
    return random.choice(choices) if choices else None


def _place_markers(grid: np.ndarray, rows: list[int], cols: list[int]) -> None:
    h, w = grid.shape
    for r in rows:
        grid[r, 0] = random.choice((RED, TEAL))
        grid[r, w - 1] = random.choice((RED, TEAL))
    for c in cols:
        grid[0, c] = random.choice((RED, TEAL))
        grid[h - 1, c] = random.choice((RED, TEAL))


def _apply_feature(grid: np.ndarray, rect: tuple[int, int, int, int], side: str, pos: int, kind: str) -> bool:
    r0, c0, r1, c1 = rect
    if side == "left":
        rr, cc = pos, c0
        tr, tc = pos, c0 - 1
    elif side == "right":
        rr, cc = pos, c1
        tr, tc = pos, c1 + 1
    elif side == "top":
        rr, cc = r0, pos
        tr, tc = r0 - 1, pos
    else:
        rr, cc = r1, pos
        tr, tc = r1 + 1, pos

    if kind == "notch":
        if int(grid[rr, cc]) != GREY:
            return False
        grid[rr, cc] = BLACK
        return True

    if int(grid[tr, tc]) != BLACK or int(grid[rr, cc]) != GREY:
        return False
    grid[tr, tc] = GREY
    return True


def _sample_corridors(rect: tuple[int, int, int, int]) -> tuple[list[int], list[int]] | None:
    r0, c0, r1, c1 = rect
    interior_rows = list(range(r0 + 1, r1))
    interior_cols = list(range(c0 + 1, c1))
    modes = [
        (2, 0),
        (0, 2),
        (1, 1),
        (2, 1),
        (1, 2),
    ]
    random.shuffle(modes)
    for row_count, col_count in modes:
        if len(interior_rows) < row_count or len(interior_cols) < col_count:
            continue
        rows = _choose_non_adjacent(interior_rows, row_count)
        cols = _choose_non_adjacent(interior_cols, col_count)
        if rows is not None and cols is not None:
            return rows, cols
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 19)
    w = random.randint(15, 19)
    grid = np.zeros((h, w), dtype=int)

    rect = _sample_rect(h, w)
    if rect is None:
        return None
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = GREY

    corridor_choice = _sample_corridors(rect)
    if corridor_choice is None:
        return None
    row_corridors, col_corridors = corridor_choice
    _place_markers(grid, row_corridors, col_corridors)

    desired_tabs = random.randint(3, 5)
    desired_notches = random.randint(3, 5)

    protected_specs: list[tuple[str, int]] = []
    corridor_slots = [("row", r) for r in row_corridors] + [("col", c) for c in col_corridors]
    random.shuffle(corridor_slots)
    protected_count = random.randint(2, min(3, len(corridor_slots)))
    for kind, idx in corridor_slots[:protected_count]:
        protected_specs.append((kind, idx))

    used_side_rows: dict[str, set[int]] = {"left": set(), "right": set()}
    used_side_cols: dict[str, set[int]] = {"top": set(), "bottom": set()}
    used_any_rows: set[int] = set()
    used_any_cols: set[int] = set()

    notches: list[tuple[str, int]] = []
    for kind, idx in protected_specs:
        if kind == "row":
            side = random.choice(("left", "right"))
            if idx in used_any_rows:
                return None
            notches.append((side, idx))
            used_side_rows[side].add(idx)
            used_any_rows.add(idx)
        else:
            side = random.choice(("top", "bottom"))
            if idx in used_any_cols:
                return None
            notches.append((side, idx))
            used_side_cols[side].add(idx)
            used_any_cols.add(idx)

    side_row_values = list(range(r0 + 1, r1))
    side_col_values = list(range(c0 + 1, c1))
    random.shuffle(side_row_values)
    random.shuffle(side_col_values)

    while len(notches) < desired_notches:
        horizontal = random.random() < 0.5
        if horizontal:
            candidates = [v for v in side_row_values if v not in used_any_rows and v not in row_corridors]
            if not candidates:
                horizontal = False
            else:
                idx = random.choice(candidates)
                side = random.choice(("left", "right"))
                if any(abs(idx - other) < 2 for other in used_side_rows[side]):
                    side_row_values.remove(idx)
                    continue
                notches.append((side, idx))
                used_side_rows[side].add(idx)
                used_any_rows.add(idx)
                continue
        candidates = [v for v in side_col_values if v not in used_any_cols and v not in col_corridors]
        if not candidates:
            return None
        idx = random.choice(candidates)
        side = random.choice(("top", "bottom"))
        if any(abs(idx - other) < 2 for other in used_side_cols[side]):
            side_col_values.remove(idx)
            continue
        notches.append((side, idx))
        used_side_cols[side].add(idx)
        used_any_cols.add(idx)

    tabs: list[tuple[str, int]] = []
    attempts = 0
    while len(tabs) < desired_tabs and attempts < 80:
        attempts += 1
        horizontal = random.random() < 0.5
        if horizontal:
            candidates = [v for v in side_row_values if v not in used_any_rows and v not in row_corridors]
            if not candidates:
                continue
            idx = random.choice(candidates)
            side = random.choice(("left", "right"))
            if any(abs(idx - other) < 2 for other in used_side_rows[side]):
                continue
            tabs.append((side, idx))
            used_side_rows[side].add(idx)
            used_any_rows.add(idx)
        else:
            candidates = [v for v in side_col_values if v not in used_any_cols and v not in col_corridors]
            if not candidates:
                continue
            idx = random.choice(candidates)
            side = random.choice(("top", "bottom"))
            if any(abs(idx - other) < 2 for other in used_side_cols[side]):
                continue
            tabs.append((side, idx))
            used_side_cols[side].add(idx)
            used_any_cols.add(idx)
    if len(tabs) != desired_tabs:
        return None

    for side, idx in notches:
        if not _apply_feature(grid, rect, side, idx, "notch"):
            return None
    for side, idx in tabs:
        if not _apply_feature(grid, rect, side, idx, "tab"):
            return None

    if not _validate_instance(grid):
        return None
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if _get_bg(g) != BLACK or np.any(g == GREEN):
        return False

    candidates = _list_candidates(g == GREY)
    tab_count = sum(candidate["kind"] == "tab" for candidate in candidates)
    notch_count = sum(candidate["kind"] == "notch" for candidate in candidates)
    if not (3 <= tab_count <= 5 and 3 <= notch_count <= 5):
        return False

    corridors = _detect_corridors(g)
    if not (2 <= len(corridors) <= 3):
        return False

    protected = 0
    allowed = 0
    allowed_tabs = 0
    allowed_notches = 0
    for candidate in candidates:
        blocked = _candidate_forbidden(g, corridors, candidate)
        if blocked:
            protected += 1
        else:
            allowed += 1
            if candidate["kind"] == "tab":
                allowed_tabs += 1
            else:
                allowed_notches += 1
    if protected < 2 or allowed == 0 or allowed_tabs == 0 or allowed_notches == 0:
        return False

    solved_mask_grid, status, current = _solve_structure(g)
    if any(not _corridor_line_ok(solved_mask_grid, kind, idx) for kind, idx in current):
        return False
    if sum(item["blocked"] for item in status) < 2:
        return False
    filled = main(g)
    if not np.any(filled == GREEN):
        return False
    if np.array_equal(g, filled):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid selective corridor symmetry task")
