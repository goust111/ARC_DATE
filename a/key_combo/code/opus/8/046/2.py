from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

ARM_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _marker_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr, dc in ARM_OFFSETS}


def _draw_marker(grid: np.ndarray, center: tuple[int, int]) -> None:
    for r, c in _marker_cells(center):
        grid[r, c] = RED


def _find_marker_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    centers: list[tuple[int, int]] = []
    claimed_red: set[tuple[int, int]] = set()

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != BLACK:
                continue

            arms = {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
            corners = {(r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)}
            if not all(int(g[rr, cc]) == RED for rr, cc in arms):
                continue
            if any(int(g[rr, cc]) == RED for rr, cc in corners):
                continue

            centers.append((r, c))
            claimed_red |= arms

    all_red = {tuple(map(int, pos)) for pos in np.argwhere(g == RED)}
    assert claimed_red == all_red, "red pixels must form hollow-plus markers only"
    assert 3 <= len(centers) <= 5, "expected 3 to 5 hollow-plus markers"
    return sorted(centers)


def _deflector_rows(grid: np.ndarray) -> set[int]:
    g = np.asarray(grid, dtype=int)
    rows = {int(r) for r in range(g.shape[0]) if int(g[r, 0]) == GREEN}
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if int(g[r, c]) == GREEN:
                assert c == 0, "deflector pixels must lie on the left edge"
    assert 2 <= len(rows) <= 4, "expected 2 to 4 left-edge deflector pixels"
    return rows


def _candidate_edges(centers: list[tuple[int, int]]) -> list[dict]:
    rows: dict[int, list[int]] = defaultdict(list)
    cols: dict[int, list[int]] = defaultdict(list)
    for r, c in centers:
        rows[r].append(c)
        cols[c].append(r)

    edges: list[dict] = []
    for r, cols_in_row in rows.items():
        ordered = sorted(cols_in_row)
        for c0, c1 in zip(ordered, ordered[1:]):
            edges.append(
                {
                    "kind": "H",
                    "a": (int(r), int(c0)),
                    "b": (int(r), int(c1)),
                    "steps": int(c1 - c0 - 3),
                }
            )
    for c, rows_in_col in cols.items():
        ordered = sorted(rows_in_col)
        for r0, r1 in zip(ordered, ordered[1:]):
            edges.append(
                {
                    "kind": "V",
                    "a": (int(r0), int(c)),
                    "b": (int(r1), int(c)),
                    "steps": int(r1 - r0 - 3),
                }
            )
    return edges


def _trace_edge(
    shape: tuple[int, int],
    edge: dict,
    deflector_rows: set[int],
    red_mask: np.ndarray,
    green_mask: np.ndarray,
) -> dict:
    h, w = shape
    kind = edge["kind"]
    (r0, c0) = edge["a"]
    (r1, c1) = edge["b"]
    steps = int(edge["steps"])
    cells: list[tuple[int, int]] = []
    deflections = 0

    if steps <= 0:
        return {
            "edge": edge,
            "cells": cells,
            "legal": False,
            "reason": "degenerate",
            "deflections": 0,
        }

    if kind == "H":
        r = int(r0)
        c = int(c0 + 1)
        for _ in range(steps):
            c += 1
            if r in deflector_rows:
                r += 1
                deflections += 1
            cells.append((r, c))
            if not (0 <= r < h and 0 <= c < w):
                return {
                    "edge": edge,
                    "cells": cells,
                    "legal": False,
                    "reason": "oob",
                    "deflections": deflections,
                }
            if bool(red_mask[r, c]):
                return {
                    "edge": edge,
                    "cells": cells,
                    "legal": False,
                    "reason": "red",
                    "deflections": deflections,
                }
            if bool(green_mask[r, c]):
                return {
                    "edge": edge,
                    "cells": cells,
                    "legal": False,
                    "reason": "green",
                    "deflections": deflections,
                }
    else:
        r = int(r0 + 1)
        c = int(c0)
        for _ in range(steps):
            r += 1
            if r in deflector_rows:
                c += 1
                deflections += 1
            cells.append((r, c))
            if not (0 <= r < h and 0 <= c < w):
                return {
                    "edge": edge,
                    "cells": cells,
                    "legal": False,
                    "reason": "oob",
                    "deflections": deflections,
                }
            if bool(red_mask[r, c]):
                return {
                    "edge": edge,
                    "cells": cells,
                    "legal": False,
                    "reason": "red",
                    "deflections": deflections,
                }
            if bool(green_mask[r, c]):
                return {
                    "edge": edge,
                    "cells": cells,
                    "legal": False,
                    "reason": "green",
                    "deflections": deflections,
                }

    return {
        "edge": edge,
        "cells": cells,
        "legal": True,
        "reason": None,
        "deflections": deflections,
    }


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    centers = _find_marker_centers(g)
    deflector_rows = _deflector_rows(g)
    edges = _candidate_edges(centers)
    red_mask = g == RED
    green_mask = g == GREEN
    traces = [
        _trace_edge(g.shape, edge, deflector_rows, red_mask, green_mask) for edge in edges
    ]
    return {
        "centers": centers,
        "deflector_rows": deflector_rows,
        "edges": edges,
        "traces": traces,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect adjacent aligned hollow-plus anchors with blue paths whose stepwise deflections can invalidate the connection.

    Input structure:
    - The grid uses black background, 3 to 5 red hollow-plus markers, and 2 to 4 green single-pixel deflectors on the left edge.
    - Each marker is a 3x3 hollow plus: only the north, south, east, and west arm pixels are red, while the center remains black.
    - Some marker centers share a row or a column, so they define horizontal or vertical candidate connections.

    Rule:
    - Find every hollow-plus center, then connect only adjacent centers within each shared row and each shared column.
    - For a horizontal candidate, walk from the left marker's outer arm toward the right one one column at a time; whenever the current row has a green deflector, that step also shifts one row downward. For a vertical candidate, walk one row at a time and shift one column rightward whenever the newly entered row has a green deflector.
    - Cancel a candidate if any visited path cell goes out of bounds, lands on a red marker pixel, or lands on a green deflector; otherwise paint the visited path cells blue while preserving all red and green pixels.

    Key insight:
    - The green edge pixels do not directly draw anything; they alter the geometry of each candidate connector, and only by simulating that bent path step by step can you know which adjacent-anchor connections survive.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    meta = _analyze(g)
    for trace in meta["traces"]:
        if trace["legal"]:
            for r, c in trace["cells"]:
                out[r, c] = BLUE
    return out


def _centers_clear(existing: list[tuple[int, int]], candidate: tuple[int, int]) -> bool:
    r, c = candidate
    if r < 1 or c < 1:
        return False
    for rr, cc in existing:
        if max(abs(r - rr), abs(c - cc)) < 3:
            return False
    return True


def _render_input(
    height: int,
    width: int,
    centers: list[tuple[int, int]],
    deflector_rows: set[int],
) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for r in deflector_rows:
        grid[r, 0] = GREEN
    for center in centers:
        _draw_marker(grid, center)
    return grid


def _pick_extra_center(
    height: int,
    width: int,
    centers: list[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> tuple[int, int] | None:
    used_rows = {r for r, _ in centers}
    used_cols = {c for _, c in centers}
    choices = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if r not in used_rows and c not in used_cols and _centers_clear(centers, (r, c))
    ]
    random.shuffle(choices)
    for center in choices:
        if _marker_cells(center) & forbidden:
            continue
        return center
    return None


def _augment_with_isolated_markers(
    height: int,
    width: int,
    centers: list[tuple[int, int]],
    deflector_rows: set[int],
    target_count: int,
) -> list[tuple[int, int]] | None:
    if len(centers) >= target_count:
        return centers

    base_grid = _render_input(height, width, centers, deflector_rows)
    meta = _analyze(base_grid)
    forbidden = {(r, c) for r, c in np.argwhere(base_grid != BLACK)}
    for trace in meta["traces"]:
        forbidden |= set(trace["cells"])

    out = list(centers)
    while len(out) < target_count:
        extra = _pick_extra_center(height, width, out, forbidden)
        if extra is None:
            return None
        out.append(extra)
        forbidden |= _marker_cells(extra)
    return out


def _template_shared_anchor_oob() -> tuple[int, int, list[tuple[int, int]], set[int]] | None:
    height = random.randint(14, 19)
    width = random.randint(18, 24)
    anchor_col = width - 2
    left_gap = random.randint(4, min(8, anchor_col - 1))
    left_col = anchor_col - left_gap
    row = random.randint(2, height - 10)
    bottom_row = random.randint(row + 6, min(height - 2, row + 10))
    vertical_rows = list(range(row + 2, bottom_row - 1))
    if len(vertical_rows) < 2:
        return None

    centers = [(row, left_col), (row, anchor_col), (bottom_row, anchor_col)]
    deflector_rows = {row, *random.sample(vertical_rows, 2)}
    if random.random() < 0.4:
        spare = [rr for rr in range(height) if rr not in deflector_rows and rr != row + 1]
        if spare:
            deflector_rows.add(random.choice(spare))
    return height, width, centers, deflector_rows


def _template_pair_pair_oob() -> tuple[int, int, list[tuple[int, int]], set[int]] | None:
    height = random.randint(16, 20)
    width = random.randint(19, 25)

    row_h = random.randint(2, height - 9)
    left_h = random.randint(3, width - 12)
    right_h = left_h + random.randint(4, 7)

    col_v = width - 2
    top_candidates = [rr for rr in range(2, height - 7) if abs(rr - row_h) >= 3]
    if not top_candidates:
        return None
    top_v = random.choice(top_candidates)
    bottom_v = random.randint(top_v + 6, min(height - 2, top_v + 10))
    vertical_rows = list(range(top_v + 2, bottom_v - 1))
    if len(vertical_rows) < 2:
        return None

    centers = [(row_h, left_h), (row_h, right_h), (top_v, col_v), (bottom_v, col_v)]
    if not all(_centers_clear(centers[:i], centers[i]) for i in range(len(centers))):
        return None

    deflector_rows = {row_h, *random.sample(vertical_rows, 2)}
    if random.random() < 0.5:
        spare = [rr for rr in range(height) if rr not in deflector_rows and rr != row_h + 1]
        if spare:
            deflector_rows.add(random.choice(spare))
    return height, width, centers, deflector_rows


def _template_h_collision_v_legal() -> tuple[int, int, list[tuple[int, int]], set[int]] | None:
    height = random.randint(15, 20)
    width = random.randint(18, 24)

    row_h = random.randint(2, height - 9)
    left_h = random.randint(3, width - 12)
    right_h = left_h + random.randint(6, 8)
    blocker_col = random.randint(left_h + 3, right_h - 3)

    blocker_row = row_h + 2
    bottom_v = random.randint(blocker_row + 5, min(height - 2, blocker_row + 9))
    vertical_rows = list(range(blocker_row + 2, bottom_v - 1))
    if not vertical_rows:
        return None

    centers = [
        (row_h, left_h),
        (row_h, right_h),
        (blocker_row, blocker_col),
        (bottom_v, blocker_col),
    ]
    if not all(_centers_clear(centers[:i], centers[i]) for i in range(len(centers))):
        return None

    deflector_rows = {row_h, random.choice(vertical_rows)}
    if random.random() < 0.6:
        spare = [rr for rr in range(height) if rr not in deflector_rows and rr != row_h + 1]
        if spare:
            deflector_rows.add(random.choice(spare))
    return height, width, centers, deflector_rows


def _construct_case() -> np.ndarray | None:
    template = random.choices(
        [_template_shared_anchor_oob, _template_pair_pair_oob, _template_h_collision_v_legal],
        weights=[3, 3, 4],
        k=1,
    )[0]
    built = template()
    if built is None:
        return None

    height, width, centers, deflector_rows = built
    target_count = min(5, len(centers) + random.choice([0, 0, 1]))
    centers = _augment_with_isolated_markers(height, width, centers, deflector_rows, target_count)
    if centers is None:
        return None
    return _render_input(height, width, centers, deflector_rows)


def _valid_generated_case(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False

    try:
        meta = _analyze(grid)
    except AssertionError:
        return False

    kinds = {trace["edge"]["kind"] for trace in meta["traces"]}
    if kinds != {"H", "V"}:
        return False

    legal = [trace for trace in meta["traces"] if trace["legal"]]
    illegal = [trace for trace in meta["traces"] if not trace["legal"]]
    if not legal or not illegal:
        return False
    if not any(trace["deflections"] > 0 for trace in legal):
        return False
    if not any(trace["reason"] in {"oob", "red"} for trace in illegal):
        return False
    if not any(trace["edge"]["kind"] == "H" for trace in meta["traces"]):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if int(np.count_nonzero(out == BLUE)) == 0:
        return False

    legal_paths = [set(trace["cells"]) for trace in legal]
    if len(legal_paths) >= 2 and len(legal_paths[0] & legal_paths[1]) > 1:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _construct_case()
        if grid is not None and _valid_generated_case(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
