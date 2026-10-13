from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS) not in sys.path:
    sys.path.append(str(_ARC_UTILS))

from arc_utils import Color, colors_present


SCAFFOLD_COLOR = Color.GREY
MARKER_COLORS = (Color.BLUE, Color.RED, Color.GREEN)
GRID_BLOCK_RANGE = (3, 7)  # overall grid size = 4*n - 1, so max 27
GENERATE_ATTEMPTS = 400


def _true_runs(flags: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def _infer_frame_starts(g: np.ndarray, scaffold_color: int) -> tuple[list[int], list[int]]:
    row_runs = _true_runs([bool(np.any(g[r] == scaffold_color)) for r in range(g.shape[0])])
    col_runs = _true_runs([bool(np.any(g[:, c] == scaffold_color)) for c in range(g.shape[1])])
    assert row_runs and col_runs, "expected a visible scaffold lattice"

    row_starts = [r0 for r0, r1 in row_runs]
    col_starts = [c0 for c0, c1 in col_runs]
    assert all(r1 - r0 + 1 == 3 for r0, r1 in row_runs), "each scaffold row run must have height 3"
    assert all(c1 - c0 + 1 == 3 for c0, c1 in col_runs), "each scaffold column run must have width 3"
    assert len(row_starts) >= 2 and len(col_starts) >= 2, "expected multiple scaffold frames"
    assert all(b - a == 4 for a, b in zip(row_starts, row_starts[1:])), "row frames must be regularly spaced"
    assert all(b - a == 4 for a, b in zip(col_starts, col_starts[1:])), "column frames must be regularly spaced"
    return row_starts, col_starts


def _infer_scaffold_color(g: np.ndarray, bg: int) -> int:
    foreground = colors_present(g, exclude={bg})
    assert foreground, "expected scaffold and markers"
    counts = {color: int(np.count_nonzero(g == color)) for color in foreground}
    max_count = max(counts.values())
    candidates = [color for color, count in counts.items() if count == max_count]
    assert len(candidates) == 1, "expected a unique scaffold color"
    return int(candidates[0])


def _extract_pairs(
    g: np.ndarray,
    bg: int,
    scaffold_color: int,
    row_starts: list[int],
    col_starts: list[int],
) -> list[dict]:
    pairs: list[dict] = []
    frame_edge_rows = {r for r0 in row_starts for r in (r0, r0 + 2)}
    frame_edge_cols = {c for c0 in col_starts for c in (c0, c0 + 2)}

    for color in colors_present(g, exclude={bg, scaffold_color}):
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each marker color must appear exactly twice"
        (r1, c1), (r2, c2) = [tuple(map(int, rc)) for rc in coords]

        if r1 == r2:
            lo, hi = sorted((c1, c2))
            row_matches = [ri for ri, r0 in enumerate(row_starts) if r1 in (r0, r0 + 2)]
            assert len(row_matches) == 1, "horizontal markers must lie on one scaffold edge row"
            assert r1 in frame_edge_rows and c1 in frame_edge_cols and c2 in frame_edge_cols
            pairs.append(
                {
                    "color": int(color),
                    "orientation": "h",
                    "fixed": int(r1),
                    "lo": int(lo),
                    "hi": int(hi),
                    "row_index": int(row_matches[0]),
                    "endpoints": ((r1, c1), (r2, c2)),
                }
            )
        elif c1 == c2:
            lo, hi = sorted((r1, r2))
            col_matches = [ci for ci, c0 in enumerate(col_starts) if c1 in (c0, c0 + 2)]
            assert len(col_matches) == 1, "vertical markers must lie on one scaffold edge column"
            assert c1 in frame_edge_cols and r1 in frame_edge_rows and r2 in frame_edge_rows
            pairs.append(
                {
                    "color": int(color),
                    "orientation": "v",
                    "fixed": int(c1),
                    "lo": int(lo),
                    "hi": int(hi),
                    "col_index": int(col_matches[0]),
                    "endpoints": ((r1, c1), (r2, c2)),
                }
            )
        else:
            raise AssertionError("same-color markers must share a row or a column")

    assert pairs, "expected at least one marker pair"
    return pairs


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(Color.BLACK)
    assert np.any(g == bg), "expected black background"
    scaffold_color = _infer_scaffold_color(g, bg)
    row_starts, col_starts = _infer_frame_starts(g, scaffold_color)
    pairs = _extract_pairs(g, bg, scaffold_color, row_starts, col_starts)

    out = g.copy()
    center_roles: dict[tuple[int, int], dict[str, int]] = {}
    frame_centers = {(r0 + 1, c0 + 1) for r0 in row_starts for c0 in col_starts}
    pair_paint_counts: dict[int, int] = {}

    for pair in pairs:
        color = int(pair["color"])
        paint_count = 0
        if pair["orientation"] == "h":
            r = int(pair["fixed"])
            for c in range(int(pair["lo"]) + 1, int(pair["hi"])):
                if out[r, c] == bg:
                    out[r, c] = color
                    paint_count += 1

            ri = int(pair["row_index"])
            for cj, c0 in enumerate(col_starts):
                if pair["hi"] < c0 or pair["lo"] > c0 + 2:
                    continue
                center = (row_starts[ri] + 1, c0 + 1)
                prev = center_roles.setdefault(center, {}).get("h")
                assert prev in (None, color), "a frame cannot receive two different horizontal colors"
                center_roles[center]["h"] = color
        else:
            c = int(pair["fixed"])
            for r in range(int(pair["lo"]) + 1, int(pair["hi"])):
                if out[r, c] == bg:
                    out[r, c] = color
                    paint_count += 1

            cj = int(pair["col_index"])
            for ri, r0 in enumerate(row_starts):
                if pair["hi"] < r0 or pair["lo"] > r0 + 2:
                    continue
                center = (r0 + 1, col_starts[cj] + 1)
                prev = center_roles.setdefault(center, {}).get("v")
                assert prev in (None, color), "a frame cannot receive two different vertical colors"
                center_roles[center]["v"] = color

        pair_paint_counts[color] = paint_count

    for center, roles in center_roles.items():
        out[center] = int(roles["h"] if "h" in roles else roles["v"])

    return out, {
        "bg": bg,
        "scaffold_color": scaffold_color,
        "row_starts": row_starts,
        "col_starts": col_starts,
        "pairs": pairs,
        "frame_centers": frame_centers,
        "center_roles": center_roles,
        "pair_paint_counts": pair_paint_counts,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each aligned same-color marker pair along the scaffold and fill the centers of the traversed hollow frames.

    Input structure:
    - A black background contains a regular lattice of grey hollow 3x3 frames, separated by one-cell gaps.
    - Non-grey, non-black marker pixels appear in same-color pairs on scaffold corner coordinates.
    - Each color appears exactly twice, and each pair is aligned on one row or one column.

    Rule:
    - For every same-color pair, draw the straight row or column segment between the two markers, painting only background cells while leaving grey scaffold pixels unchanged.
    - A horizontal pair activates every frame whose top or bottom edge lies on that segment; a vertical pair activates every frame whose left or right edge lies on that segment.
    - Fill each activated frame center with the activating line color.
    - If one frame is activated by both a horizontal and a vertical line, the horizontal line color wins.
    - Keep the marker pixels and the grey frame outlines unchanged.

    Key insight:
    - The markers do not directly choose frame centers; they first induce scaffold-aligned line paths, and those paths determine which hollow frames become filled.
    """
    return _solve_details(grid)[0]


def _build_scaffold(n_rows: int, n_cols: int) -> tuple[np.ndarray, list[int], list[int]]:
    height = 4 * int(n_rows) - 1
    width = 4 * int(n_cols) - 1
    grid = np.zeros((height, width), dtype=int)
    row_starts = [4 * i for i in range(int(n_rows))]
    col_starts = [4 * i for i in range(int(n_cols))]
    for r0 in row_starts:
        for c0 in col_starts:
            grid[r0 : r0 + 3, c0 : c0 + 3] = SCAFFOLD_COLOR
            grid[r0 + 1, c0 + 1] = Color.BLACK
    return grid, row_starts, col_starts


def _horizontal_specs(row_starts: list[int], col_starts: list[int]) -> list[dict]:
    specs: list[dict] = []
    for ri, r0 in enumerate(row_starts):
        for edge in (0, 2):
            r = r0 + edge
            for c_lo in range(len(col_starts) - 1):
                for c_hi in range(c_lo + 1, len(col_starts)):
                    specs.append(
                        {
                            "orientation": "h",
                            "row_index": ri,
                            "fixed": r,
                            "lo": col_starts[c_lo],
                            "hi": col_starts[c_hi] + 2,
                            "endpoints": ((r, col_starts[c_lo]), (r, col_starts[c_hi] + 2)),
                            "frames": {(ri, cj) for cj in range(c_lo, c_hi + 1)},
                        }
                    )
    return specs


def _vertical_specs(row_starts: list[int], col_starts: list[int]) -> list[dict]:
    specs: list[dict] = []
    for cj, c0 in enumerate(col_starts):
        for edge in (0, 2):
            c = c0 + edge
            for r_lo in range(len(row_starts) - 1):
                for r_hi in range(r_lo + 1, len(row_starts)):
                    specs.append(
                        {
                            "orientation": "v",
                            "col_index": cj,
                            "fixed": c,
                            "lo": row_starts[r_lo],
                            "hi": row_starts[r_hi] + 2,
                            "endpoints": ((row_starts[r_lo], c), (row_starts[r_hi] + 2, c)),
                            "frames": {(ri, cj) for ri in range(r_lo, r_hi + 1)},
                        }
                    )
    return specs


def _point_on_spec(point: tuple[int, int], spec: dict) -> bool:
    r, c = point
    if spec["orientation"] == "h":
        return r == spec["fixed"] and spec["lo"] <= c <= spec["hi"]
    return c == spec["fixed"] and spec["lo"] <= r <= spec["hi"]


def _specs_are_clean(specs: list[dict]) -> bool:
    endpoints = [pt for spec in specs for pt in spec["endpoints"]]
    if len(set(endpoints)) != len(endpoints):
        return False

    for i, spec in enumerate(specs):
        for pt in spec["endpoints"]:
            for j, other in enumerate(specs):
                if i == j:
                    continue
                if _point_on_spec(pt, other):
                    return False
    return True


def _choose_specs(row_starts: list[int], col_starts: list[int]) -> list[dict] | None:
    h_specs = _horizontal_specs(row_starts, col_starts)
    v_specs = _vertical_specs(row_starts, col_starts)

    for _ in range(200):
        h_count = random.choice((1, 2))
        v_count = 3 - h_count

        h0 = random.choice(h_specs)
        conflict_vs = [v for v in v_specs if h0["frames"] & v["frames"]]
        if not conflict_vs:
            continue
        v0 = random.choice(conflict_vs)

        selected_h = [h0]
        selected_v = [v0]
        used_rows = {h0["row_index"]}
        used_cols = {v0["col_index"]}

        while len(selected_h) < h_count:
            candidates = [h for h in h_specs if h["row_index"] not in used_rows]
            random.shuffle(candidates)
            for h in candidates:
                if _specs_are_clean(selected_h + selected_v + [h]):
                    selected_h.append(h)
                    used_rows.add(h["row_index"])
                    break
            else:
                break
        if len(selected_h) != h_count:
            continue

        while len(selected_v) < v_count:
            candidates = [v for v in v_specs if v["col_index"] not in used_cols]
            random.shuffle(candidates)
            for v in candidates:
                if _specs_are_clean(selected_h + selected_v + [v]):
                    selected_v.append(v)
                    used_cols.add(v["col_index"])
                    break
            else:
                break
        if len(selected_v) != v_count:
            continue

        specs = selected_h + selected_v
        if _specs_are_clean(specs):
            random.shuffle(specs)
            return specs
    return None


def _render_input(base: np.ndarray, specs: list[dict], colors: list[int]) -> np.ndarray:
    grid = base.copy()
    for spec, color in zip(specs, colors):
        for r, c in spec["endpoints"]:
            assert grid[r, c] == SCAFFOLD_COLOR
            grid[r, c] = int(color)
    return grid


def _validate_generated(inp: np.ndarray) -> bool:
    try:
        out, details = _solve_details(inp)
    except AssertionError:
        return False

    centers = details["frame_centers"]
    roles = details["center_roles"]
    pairs = details["pairs"]
    paint_counts = details["pair_paint_counts"]

    if set(int(np.count_nonzero(inp == color)) for color in MARKER_COLORS) != {2}:
        return False

    if any(paint_counts.get(pair["color"], 0) <= 0 for pair in pairs):
        return False

    conflicts = [center for center, role in roles.items() if "h" in role and "v" in role and role["h"] != role["v"]]
    if not conflicts:
        return False

    only_h = [center for center, role in roles.items() if "h" in role and "v" not in role]
    only_v = [center for center, role in roles.items() if "v" in role and "h" not in role]
    if not only_h or not only_v:
        return False

    if not any(tuple(center) not in roles for center in centers):
        return False

    if len({int(out[center]) for center in roles}) < 2:
        return False

    for center in conflicts:
        if int(out[center]) != int(roles[center]["h"]):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(*GRID_BLOCK_RANGE)
    n_cols = random.randint(*GRID_BLOCK_RANGE)
    base, row_starts, col_starts = _build_scaffold(n_rows, n_cols)
    specs = _choose_specs(row_starts, col_starts)
    if specs is None:
        return None

    colors = random.sample(list(MARKER_COLORS), len(specs))
    inp = _render_input(base, specs, colors)
    if not _validate_generated(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid connect-activate scaffold instance")
