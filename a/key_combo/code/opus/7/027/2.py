from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

TOP_COLORS = (BLUE, GREEN)
BOTTOM_COLORS = (RED, YELLOW)
LEFT_COLOR = MAGENTA
RIGHT_COLOR = ORANGE

SIDE_TO_DIR = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}

GEN_ATTEMPTS = 400
GEN_SIZE_RANGE = (13, 17)
GEN_BLOCKER_COUNT_RANGE = (3, 6)


def _collect_markers(grid: np.ndarray) -> dict[str, list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 5
    h, w = g.shape

    assert int(g[0, 0]) == BLACK
    assert int(g[0, w - 1]) == BLACK
    assert int(g[h - 1, 0]) == BLACK
    assert int(g[h - 1, w - 1]) == BLACK
    assert not np.any(g[1:-1, 1:-1] < 0)

    markers: dict[str, list[tuple[int, int, int]]] = {
        "top": [],
        "bottom": [],
        "left": [],
        "right": [],
    }

    for c in range(1, w - 1):
        top = int(g[0, c])
        bottom = int(g[h - 1, c])
        if top != BLACK:
            assert top in TOP_COLORS
            markers["top"].append((0, c, top))
        if bottom != BLACK:
            assert bottom in BOTTOM_COLORS
            markers["bottom"].append((h - 1, c, bottom))

    for r in range(1, h - 1):
        left = int(g[r, 0])
        right = int(g[r, w - 1])
        if left != BLACK:
            assert left == LEFT_COLOR
            markers["left"].append((r, 0, left))
        if right != BLACK:
            assert right == RIGHT_COLOR
            markers["right"].append((r, w - 1, right))

    interior_colors = set(map(int, np.unique(g[1:-1, 1:-1])))
    assert interior_colors.issubset({BLACK, GREY})
    assert not np.any(g[0, :] == GREY)
    assert not np.any(g[h - 1, :] == GREY)
    assert not np.any(g[:, 0] == GREY)
    assert not np.any(g[:, w - 1] == GREY)

    vertical_cols = [c for _, c, _ in markers["top"] + markers["bottom"]]
    horizontal_rows = [r for r, _, _ in markers["left"] + markers["right"]]
    assert len(set(vertical_cols)) == len(vertical_cols)
    assert len(set(horizontal_rows)) == len(horizontal_rows)
    return markers


def _trace_ray(
    grid: np.ndarray,
    start: tuple[int, int, int],
    side: str,
    *,
    ignore_blockers: bool = False,
) -> tuple[list[tuple[int, int, int]], bool]:
    h, w = grid.shape
    r, c, color = start
    dr, dc = SIDE_TO_DIR[side]
    rr, cc = r + dr, c + dc
    dist = 1
    hits: list[tuple[int, int, int]] = []
    blocked = False

    while 0 < rr < h - 1 and 0 < cc < w - 1:
        cell = int(grid[rr, cc])
        assert cell in (BLACK, GREY), "interior must contain only background or blockers"
        if cell == GREY:
            if not ignore_blockers:
                blocked = True
                break
        else:
            hits.append((rr, cc, dist))
        rr += dr
        cc += dc
        dist += 1

    return hits, blocked


def _trace_rays(
    grid: np.ndarray,
    *,
    ignore_blockers: bool = False,
) -> tuple[
    dict[tuple[int, int], dict[str, int | str | tuple[int, int]]],
    dict[tuple[int, int], dict[str, int | str | tuple[int, int]]],
    list[dict[str, object]],
    dict[str, list[tuple[int, int, int]]],
]:
    markers = _collect_markers(grid)
    vertical_hits: dict[tuple[int, int], dict[str, int | str | tuple[int, int]]] = {}
    horizontal_hits: dict[tuple[int, int], dict[str, int | str | tuple[int, int]]] = {}
    ray_stats: list[dict[str, object]] = []

    for side in ("top", "bottom", "left", "right"):
        orientation = "vertical" if side in ("top", "bottom") else "horizontal"
        hit_map = vertical_hits if orientation == "vertical" else horizontal_hits
        for marker in markers[side]:
            hits, blocked = _trace_ray(grid, marker, side, ignore_blockers=ignore_blockers)
            ray_stats.append(
                {
                    "side": side,
                    "orientation": orientation,
                    "start": (int(marker[0]), int(marker[1])),
                    "color": int(marker[2]),
                    "blocked": blocked,
                    "length": len(hits),
                }
            )
            for rr, cc, dist in hits:
                key = (rr, cc)
                assert key not in hit_map
                hit_map[key] = {
                    "color": int(marker[2]),
                    "dist": int(dist),
                    "side": side,
                    "start": (int(marker[0]), int(marker[1])),
                }

    return vertical_hits, horizontal_hits, ray_stats, markers


def _compose_output(
    grid: np.ndarray,
    vertical_hits: dict[tuple[int, int], dict[str, int | str | tuple[int, int]]],
    horizontal_hits: dict[tuple[int, int], dict[str, int | str | tuple[int, int]]],
    *,
    mode: str = "distance",
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for cell in vertical_hits.keys() | horizontal_hits.keys():
        v = vertical_hits.get(cell)
        h = horizontal_hits.get(cell)
        winner: dict[str, int | str | tuple[int, int]]
        if v is None:
            winner = h  # type: ignore[assignment]
        elif h is None:
            winner = v
        elif mode == "fixed_horizontal":
            winner = h
        elif mode == "fixed_vertical":
            winner = v
        elif mode == "tie_vertical":
            winner = v if int(v["dist"]) <= int(h["dist"]) else h
        else:
            if int(v["dist"]) < int(h["dist"]):
                winner = v
            elif int(h["dist"]) < int(v["dist"]):
                winner = h
            else:
                winner = h
        out[cell] = int(winner["color"])
    return out


def _solve_with_mode(grid: np.ndarray, mode: str = "distance") -> np.ndarray:
    ignore_blockers = mode == "ignore_blockers"
    vertical_hits, horizontal_hits, _, _ = _trace_rays(grid, ignore_blockers=ignore_blockers)
    compose_mode = "distance" if ignore_blockers else mode
    return _compose_output(grid, vertical_hits, horizontal_hits, mode=compose_mode)


def _analyze(grid: np.ndarray) -> dict[str, object]:
    vertical_hits, horizontal_hits, ray_stats, markers = _trace_rays(grid)
    crossings = []
    for cell in sorted(vertical_hits.keys() & horizontal_hits.keys()):
        v = vertical_hits[cell]
        h = horizontal_hits[cell]
        if int(v["dist"]) < int(h["dist"]):
            kind = "vertical"
            winner_side = str(v["side"])
            winner_color = int(v["color"])
        elif int(h["dist"]) < int(v["dist"]):
            kind = "horizontal"
            winner_side = str(h["side"])
            winner_color = int(h["color"])
        else:
            kind = "tie"
            winner_side = str(h["side"])
            winner_color = int(h["color"])
        crossings.append(
            {
                "cell": cell,
                "kind": kind,
                "winner_side": winner_side,
                "winner_color": winner_color,
                "vertical_dist": int(v["dist"]),
                "horizontal_dist": int(h["dist"]),
            }
        )

    blockers = {tuple(map(int, cell)) for cell in np.argwhere(np.asarray(grid, dtype=int) == GREY)}
    output = _compose_output(grid, vertical_hits, horizontal_hits)
    return {
        "markers": markers,
        "blockers": blockers,
        "vertical_hits": vertical_hits,
        "horizontal_hits": horizontal_hits,
        "crossings": crossings,
        "blocked_vertical": sum(1 for s in ray_stats if s["orientation"] == "vertical" and s["blocked"]),
        "blocked_horizontal": sum(1 for s in ray_stats if s["orientation"] == "horizontal" and s["blocked"]),
        "output": output,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Extend all side markers inward as rays and resolve crossing colors by shorter travel distance.

    Input structure:
    - The grid has a black background, colored marker pixels on all four edges, and interior gray blocker pixels.
    - Top and bottom markers define vertical rays; left and right markers define horizontal rays.
    - Each used marker row or marker column is unique within its orientation, so crossings happen only between one horizontal and one vertical ray.

    Rule:
    - Every edge marker emits a ray perpendicular to its side and paints only background cells.
    - A ray stops immediately before a gray blocker and also stops when it reaches the opposite boundary.
    - If one horizontal and one vertical ray reach the same cell, the ray with the shorter distance from its marker keeps the cell; ties go to the horizontal ray.

    Key insight:
    - The final color at each crossing is not a fixed layer priority: it depends on per-cell travel distance, while blockers change which distant crossings are even reachable.
    """
    return _solve_with_mode(np.asarray(grid, dtype=int), mode="distance")


def _opposite(side: str) -> str:
    return {
        "top": "bottom",
        "bottom": "top",
        "left": "right",
        "right": "left",
    }[side]


def _choose_anchor_config(h: int, w: int, rng: np.random.Generator) -> dict[str, object]:
    configs: list[dict[str, object]] = []
    for vertical_side in ("top", "bottom"):
        for horizontal_side in ("left", "right"):
            for d_h in range(1, 4):
                for d_v in range(1, 4):
                    for tie_dist in range(2, max(h, w)):
                        row = tie_dist if vertical_side == "top" else h - 1 - tie_dist
                        tie_c = tie_dist if horizontal_side == "left" else w - 1 - tie_dist
                        if not (1 <= row <= h - 2 and 1 <= tie_c <= w - 2):
                            continue

                        if horizontal_side == "left":
                            horiz_win_c = tie_c - d_h
                            vert_win_c = tie_c + d_v
                            blocker_cols = [c for c in range(vert_win_c + 1, w - 1)]
                        else:
                            horiz_win_c = tie_c + d_h
                            vert_win_c = tie_c - d_v
                            blocker_cols = [c for c in range(1, vert_win_c)]

                        cols = (horiz_win_c, tie_c, vert_win_c)
                        if any(not (1 <= c <= w - 2) for c in cols):
                            continue
                        if len(set(cols)) != 3 or not blocker_cols:
                            continue

                        configs.append(
                            {
                                "vertical_side": vertical_side,
                                "horizontal_side": horizontal_side,
                                "row": int(row),
                                "tie_col": int(tie_c),
                                "horiz_win_col": int(horiz_win_c),
                                "vert_win_col": int(vert_win_c),
                                "blocker_cols": blocker_cols,
                            }
                        )
    if not configs:
        raise RuntimeError("failed to build anchor configuration")
    return configs[int(rng.integers(0, len(configs)))]


def _sample_without_replacement(values: list[int], count: int, rng: np.random.Generator) -> list[int]:
    assert 0 <= count <= len(values)
    if count == 0:
        return []
    picks = rng.choice(values, size=count, replace=False)
    return [int(v) for v in np.atleast_1d(picks)]


def _random_side_colors(
    count: int,
    palette: tuple[int, int],
    rng: np.random.Generator,
    *,
    ensure_both: bool = False,
) -> list[int]:
    if count <= 0:
        return []
    if ensure_both and count >= 2:
        colors = [palette[0], palette[1]]
        while len(colors) < count:
            colors.append(int(palette[int(rng.integers(0, 2))]))
        rng.shuffle(colors)
        return [int(c) for c in colors]
    return [int(palette[int(rng.integers(0, 2))]) for _ in range(count)]


def _choose_vertical_blocker_row(
    side: str,
    h: int,
    horizontal_rows: list[int],
    forbidden_rows: set[int],
    rng: np.random.Generator,
) -> int | None:
    if side == "top":
        lo, hi = 2, max(horizontal_rows) - 1
    else:
        lo, hi = min(horizontal_rows) + 1, h - 3
    candidates = [r for r in range(lo, hi + 1) if r not in forbidden_rows]
    if not candidates:
        return None
    return int(candidates[int(rng.integers(0, len(candidates)))])


def _choose_horizontal_blocker_col(
    side: str,
    w: int,
    vertical_cols: list[int],
    forbidden_cols: set[int],
    rng: np.random.Generator,
) -> int | None:
    if side == "left":
        lo, hi = 2, max(vertical_cols) - 1
    else:
        lo, hi = min(vertical_cols) + 1, w - 3
    candidates = [c for c in range(lo, hi + 1) if c not in forbidden_cols]
    if not candidates:
        return None
    return int(candidates[int(rng.integers(0, len(candidates)))])


def _build_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GEN_SIZE_RANGE))
    w = int(rng.integers(*GEN_SIZE_RANGE))
    config = _choose_anchor_config(h, w, rng)

    vertical_side = str(config["vertical_side"])
    horizontal_side = str(config["horizontal_side"])
    opposite_vertical = _opposite(vertical_side)
    opposite_horizontal = _opposite(horizontal_side)

    anchor_cols = [
        int(config["horiz_win_col"]),
        int(config["tie_col"]),
        int(config["vert_win_col"]),
    ]
    anchor_row = int(config["row"])
    anchor_blocker_col = int(config["blocker_cols"][int(rng.integers(0, len(config["blocker_cols"])))])

    vertical_counts = {"top": 2 + int(rng.integers(0, 2)), "bottom": 2 + int(rng.integers(0, 2))}
    vertical_counts[vertical_side] = 3
    horizontal_counts = {"left": 1 + int(rng.integers(0, 2)), "right": 1 + int(rng.integers(0, 2))}

    remaining_cols = [c for c in range(1, w - 1) if c not in anchor_cols]
    opposite_cols = _sample_without_replacement(remaining_cols, vertical_counts[opposite_vertical], rng)
    side_cols = {
        vertical_side: list(anchor_cols),
        opposite_vertical: opposite_cols,
    }

    extra_rows_needed = horizontal_counts["left"] + horizontal_counts["right"] - 1
    remaining_rows = [r for r in range(1, h - 1) if r != anchor_row]
    if extra_rows_needed > len(remaining_rows):
        return None
    extra_rows = _sample_without_replacement(remaining_rows, extra_rows_needed, rng)
    row_cursor = 0
    side_rows = {"left": [], "right": []}
    side_rows[horizontal_side].append(anchor_row)
    for side in ("left", "right"):
        needed = horizontal_counts[side] - (1 if side == horizontal_side else 0)
        side_rows[side].extend(extra_rows[row_cursor : row_cursor + needed])
        row_cursor += needed

    top_colors = _random_side_colors(len(side_cols["top"]), TOP_COLORS, rng, ensure_both=True)
    bottom_colors = _random_side_colors(len(side_cols["bottom"]), BOTTOM_COLORS, rng, ensure_both=True)

    grid = np.zeros((h, w), dtype=int)
    for c, color in zip(side_cols["top"], top_colors):
        grid[0, c] = color
    for c, color in zip(side_cols["bottom"], bottom_colors):
        grid[h - 1, c] = color
    for r in side_rows["left"]:
        grid[r, 0] = LEFT_COLOR
    for r in side_rows["right"]:
        grid[r, w - 1] = RIGHT_COLOR

    all_horizontal_rows = sorted(side_rows["left"] + side_rows["right"])
    all_vertical_cols = sorted(side_cols["top"] + side_cols["bottom"])
    blockers: set[tuple[int, int]] = {(anchor_row, anchor_blocker_col)}

    vertical_blocker_col = int(opposite_cols[int(rng.integers(0, len(opposite_cols)))])
    vertical_blocker_row = _choose_vertical_blocker_row(
        opposite_vertical,
        h,
        all_horizontal_rows,
        forbidden_rows=set(all_horizontal_rows),
        rng=rng,
    )
    if vertical_blocker_row is None:
        return None
    blockers.add((vertical_blocker_row, vertical_blocker_col))

    opposite_rows = side_rows[opposite_horizontal]
    horizontal_blocker_row = int(opposite_rows[int(rng.integers(0, len(opposite_rows)))])
    horizontal_blocker_col = _choose_horizontal_blocker_col(
        opposite_horizontal,
        w,
        all_vertical_cols,
        forbidden_cols=set(all_vertical_cols),
        rng=rng,
    )
    if horizontal_blocker_col is None:
        return None
    blockers.add((horizontal_blocker_row, horizontal_blocker_col))

    target_blockers = int(rng.integers(*GEN_BLOCKER_COUNT_RANGE))
    protected = {
        (anchor_row, int(config["horiz_win_col"])),
        (anchor_row, int(config["tie_col"])),
        (anchor_row, int(config["vert_win_col"])),
    }
    free_cells = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if (r, c) not in blockers and (r, c) not in protected
    ]
    rng.shuffle(free_cells)
    for cell in free_cells:
        if len(blockers) >= target_blockers:
            break
        blockers.add(cell)

    for r, c in blockers:
        grid[r, c] = GREY

    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2 and max(g.shape) <= 30
    assert 13 <= h <= 16 and 13 <= w <= 16

    analysis = _analyze(g)
    markers = analysis["markers"]
    blockers = analysis["blockers"]
    crossings = analysis["crossings"]

    assert 2 <= len(markers["top"]) <= 3
    assert 2 <= len(markers["bottom"]) <= 3
    assert 1 <= len(markers["left"]) <= 2
    assert 1 <= len(markers["right"]) <= 2
    assert 3 <= len(blockers) <= 5
    assert len(crossings) >= 3

    kinds = {crossing["kind"] for crossing in crossings}
    assert {"horizontal", "vertical", "tie"}.issubset(kinds)
    assert analysis["blocked_vertical"] >= 1
    assert analysis["blocked_horizontal"] >= 1

    correct = np.asarray(analysis["output"], dtype=int)
    fixed_horizontal = _solve_with_mode(g, "fixed_horizontal")
    fixed_vertical = _solve_with_mode(g, "fixed_vertical")
    tie_vertical = _solve_with_mode(g, "tie_vertical")
    ignore_blockers = _solve_with_mode(g, "ignore_blockers")

    assert np.any(correct != g)
    assert not np.array_equal(correct, fixed_horizontal)
    assert not np.array_equal(correct, fixed_vertical)
    assert not np.array_equal(correct, tie_vertical)
    assert not np.array_equal(correct, ignore_blockers)

    painted = np.count_nonzero((correct[1:-1, 1:-1] != BLACK) & (correct[1:-1, 1:-1] != GREY))
    interior = (h - 2) * (w - 2)
    assert interior * 0.10 <= painted <= interior * 0.60
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is None:
            continue
        try:
            if _validate_generated(grid):
                return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid ray-interference ARC input")
