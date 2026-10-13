from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
SIGNAL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_RETRIES = 400
_GENERATOR_CALLS = 0


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _orientation(dr: int, dc: int) -> str:
    return "h" if dc != 0 else "v"


def _pipe_crosses(pipe: np.ndarray) -> np.ndarray:
    h, w = pipe.shape
    crosses = np.zeros((h, w), dtype=bool)
    if h < 3 or w < 3:
        return crosses
    crosses[1:-1, 1:-1] = (
        pipe[1:-1, 1:-1]
        & pipe[1:-1, :-2]
        & pipe[1:-1, 2:]
        & pipe[:-2, 1:-1]
        & pipe[2:, 1:-1]
    )
    return crosses


def _extract_sources(grid: np.ndarray, pipe: np.ndarray) -> list[dict[str, int]]:
    h, w = grid.shape
    sources: list[dict[str, int]] = []
    source_mask = (grid != BLACK) & (grid != GREY)
    for r, c in sorted((int(r), int(c)) for r, c in np.argwhere(source_mask)):
        assert r in (0, h - 1) or c in (0, w - 1), "source must lie on the border"
        neighbors = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if _in_bounds(h, w, nr, nc) and bool(pipe[nr, nc]):
                neighbors.append((dr, dc))
        assert len(neighbors) == 1, "each source must touch exactly one pipe cell"
        dr, dc = neighbors[0]
        sources.append(
            {
                "r": r,
                "c": c,
                "dr": dr,
                "dc": dc,
                "color": int(grid[r, c]),
            }
        )
    return sources


def _simulate(grid: np.ndarray, *, collect_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    pipe = g == GREY
    crosses = _pipe_crosses(pipe)
    sources = _extract_sources(g, pipe)
    assert sources, "expected at least one signal source"

    out = g.copy()
    active = [src.copy() for src in sources]
    cross_owner: dict[tuple[int, int], dict[str, int | str]] = {}
    meta = {"claims": [], "blocked": []}

    for step in range(1, h * w + 1):
        if not active:
            break

        next_active: list[dict[str, int]] = []
        non_cross_moves: list[tuple[dict[str, int], int, int]] = []
        cross_attempts: dict[tuple[int, int], list[dict[str, int]]] = {}

        for state in active:
            nr = int(state["r"] + state["dr"])
            nc = int(state["c"] + state["dc"])
            if not _in_bounds(h, w, nr, nc) or not bool(pipe[nr, nc]):
                continue
            if bool(crosses[nr, nc]):
                cross_attempts.setdefault((nr, nc), []).append(state)
            else:
                non_cross_moves.append((state, nr, nc))

        for state, nr, nc in non_cross_moves:
            out[nr, nc] = int(state["color"])
            moved = state.copy()
            moved["r"] = nr
            moved["c"] = nc
            next_active.append(moved)

        for pos in sorted(cross_attempts):
            attempts = cross_attempts[pos]
            by_orientation: dict[str, list[dict[str, int]]] = {}
            for state in attempts:
                orient = _orientation(int(state["dr"]), int(state["dc"]))
                by_orientation.setdefault(orient, []).append(state)

            owner = cross_owner.get(pos)
            if owner is not None:
                owner_orientation = str(owner["orientation"])
                for orient, states in by_orientation.items():
                    for state in states:
                        meta["blocked"].append(
                            {
                                "pos": pos,
                                "time": step,
                                "orientation": orient,
                                "color": int(state["color"]),
                                "reason": "claimed" if orient != owner_orientation else "same_orientation",
                            }
                        )
                continue

            winner: dict[str, int] | None = None
            losers: list[dict[str, int]] = []
            mode = "single"

            if "h" in by_orientation and "v" in by_orientation:
                winner = by_orientation["h"][0]
                losers.extend(by_orientation["h"][1:])
                losers.extend(by_orientation["v"])
                mode = "tie"
            elif "h" in by_orientation:
                winner = by_orientation["h"][0]
                losers.extend(by_orientation["h"][1:])
            else:
                winner = by_orientation["v"][0]
                losers.extend(by_orientation["v"][1:])

            assert winner is not None
            winner_orientation = _orientation(int(winner["dr"]), int(winner["dc"]))
            cross_owner[pos] = {
                "orientation": winner_orientation,
                "time": step,
                "color": int(winner["color"]),
            }
            out[pos] = int(winner["color"])
            meta["claims"].append(
                {
                    "pos": pos,
                    "time": step,
                    "orientation": winner_orientation,
                    "color": int(winner["color"]),
                    "mode": mode,
                }
            )

            moved = winner.copy()
            moved["r"], moved["c"] = pos
            next_active.append(moved)

            loser_reason = "tie" if mode == "tie" else "same_orientation"
            for state in losers:
                meta["blocked"].append(
                    {
                        "pos": pos,
                        "time": step,
                        "orientation": _orientation(int(state["dr"]), int(state["dc"])),
                        "color": int(state["color"]),
                        "reason": loser_reason,
                    }
                )

        active = next_active

    if collect_meta:
        return out, meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each border source along a grey pipe. At each cross, the first signal to
    arrive claims the intersection; if both orientations arrive together, horizontal wins.
    Losers stop on the cell immediately before the cross.
    """
    return _simulate(grid)


def _distance_to_index(size: int, side: str, distance: int) -> int:
    if side in ("left", "top"):
        return distance
    return size - 1 - distance


def _draw_horizontal(grid: np.ndarray, row: int, source_side: str | None, color: int | None) -> None:
    if source_side is None:
        grid[row, :] = GREY
    elif source_side == "left":
        grid[row, 1:] = GREY
        grid[row, 0] = int(color)
    else:
        assert source_side == "right"
        grid[row, :-1] = GREY
        grid[row, -1] = int(color)


def _draw_vertical(grid: np.ndarray, col: int, source_side: str | None, color: int | None) -> None:
    if source_side is None:
        grid[:, col] = GREY
    elif source_side == "top":
        grid[1:, col] = GREY
        grid[0, col] = int(color)
    else:
        assert source_side == "bottom"
        grid[:-1, col] = GREY
        grid[-1, col] = int(color)


def _pick_spaced_positions(size: int, main_index: int, count: int) -> list[int] | None:
    if count == 0:
        return []
    candidates = [idx for idx in range(2, size - 2) if idx != main_index]
    random.shuffle(candidates)
    chosen: list[int] = []
    for idx in candidates:
        if abs(idx - main_index) < 2:
            continue
        if all(abs(idx - prev) >= 2 for prev in chosen):
            chosen.append(idx)
            if len(chosen) == count:
                return sorted(chosen)
    if len(candidates) < count:
        return None
    return sorted(random.sample(candidates, count))


def _scenario_sequence() -> str:
    global _GENERATOR_CALLS
    scenario = ("single", "shorter", "tie")[_GENERATOR_CALLS % 3]
    _GENERATOR_CALLS += 1
    return scenario


def _branch_matches(
    meta: dict[str, list[dict[str, object]]],
    main_cross: tuple[int, int],
    scenario: str,
    expected_winner: str | None,
) -> bool:
    claims = [event for event in meta["claims"] if tuple(event["pos"]) == main_cross]
    if len(claims) != 1:
        return False
    claim = claims[0]

    if scenario == "single":
        return claim["mode"] == "single" and not any(
            tuple(event["pos"]) == main_cross and event["orientation"] != claim["orientation"] for event in meta["blocked"]
        )

    if scenario == "tie":
        return claim["mode"] == "tie" and claim["orientation"] == "h"

    if scenario == "shorter":
        if claim["orientation"] != expected_winner:
            return False
        blocked = [
            event
            for event in meta["blocked"]
            if tuple(event["pos"]) == main_cross and event["reason"] == "claimed" and event["orientation"] != expected_winner
        ]
        return len(blocked) >= 1 and claim["mode"] == "single"

    return False


def _gen_attempt(scenario: str) -> np.ndarray | None:
    h = random.randint(10, 18)
    w = random.randint(10, 18)
    grid = np.full((h, w), BLACK, dtype=int)

    if scenario == "single":
        source_orientation = random.choice(["h", "v"])
        h_side = random.choice(["left", "right"]) if source_orientation == "h" else None
        v_side = random.choice(["top", "bottom"]) if source_orientation == "v" else None
        h_dist = random.randint(2, w - 3) if h_side is not None else None
        v_dist = random.randint(2, h - 3) if v_side is not None else None
        main_col = _distance_to_index(w, h_side, h_dist) if h_side is not None else random.randint(2, w - 3)
        main_row = _distance_to_index(h, v_side, v_dist) if v_side is not None else random.randint(2, h - 3)
        winner = source_orientation
    elif scenario == "tie":
        h_side = random.choice(["left", "right"])
        v_side = random.choice(["top", "bottom"])
        max_dist = min(w - 3, h - 3)
        distance = random.randint(2, max_dist)
        h_dist = distance
        v_dist = distance
        main_col = _distance_to_index(w, h_side, h_dist)
        main_row = _distance_to_index(h, v_side, v_dist)
        winner = "h"
    else:
        assert scenario == "shorter"
        h_side = random.choice(["left", "right"])
        v_side = random.choice(["top", "bottom"])
        winner = random.choice(["h", "v"])
        if winner == "h":
            max_h_dist = min(w - 3, h - 4)
            if max_h_dist < 2:
                return None
            h_dist = random.randint(2, max_h_dist)
            v_dist = random.randint(h_dist + 1, h - 3)
        else:
            max_v_dist = min(h - 3, w - 4)
            if max_v_dist < 2:
                return None
            v_dist = random.randint(2, max_v_dist)
            h_dist = random.randint(v_dist + 1, w - 3)
        main_col = _distance_to_index(w, h_side, h_dist)
        main_row = _distance_to_index(h, v_side, v_dist)

    source_count = int(h_side is not None) + int(v_side is not None)
    colors = random.sample(SIGNAL_COLORS, source_count)
    h_color = colors[0] if h_side is not None else None
    v_color = colors[-1] if v_side is not None else None

    max_h_extra = min(2, max(0, h - 6))
    max_v_extra = min(2, max(0, w - 6))
    extra_h_count = random.randint(0, max_h_extra)
    extra_v_count = random.randint(0, max_v_extra)
    extra_rows = _pick_spaced_positions(h, main_row, extra_h_count)
    extra_cols = _pick_spaced_positions(w, main_col, extra_v_count)
    if extra_rows is None or extra_cols is None:
        return None

    _draw_horizontal(grid, main_row, h_side, h_color)
    _draw_vertical(grid, main_col, v_side, v_color)
    for row in extra_rows:
        _draw_horizontal(grid, row, None, None)
    for col in extra_cols:
        _draw_vertical(grid, col, None, None)

    pipe = grid == GREY
    if not np.any(_pipe_crosses(pipe)):
        return None

    try:
        _ = _extract_sources(grid, pipe)
        _, meta = _simulate(grid, collect_meta=True)
    except AssertionError:
        return None

    if not _branch_matches(meta, (main_row, main_col), scenario, winner):
        return None
    return grid


def generate_input() -> np.ndarray:
    scenario = _scenario_sequence()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(scenario)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input for scenario {scenario}")
