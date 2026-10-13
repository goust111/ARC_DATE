from __future__ import annotations

import random
from typing import Iterable

import numpy as np


class Color:
    BLACK = 0
    RED = 2
    GREEN = 3
    GREY = 5
    ORANGE = 7


PANEL_PATTERNS: dict[int, tuple[np.ndarray, ...]] = {
    2: (
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=int),
        np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=int),
    ),
    3: (
        np.array([[1, 0, 1], [0, 0, 0], [0, 0, 1]], dtype=int),
        np.array([[1, 0, 0], [0, 0, 0], [1, 0, 1]], dtype=int),
        np.array([[0, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=int),
    ),
    4: (
        np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=int),
        np.array([[1, 1, 1], [0, 0, 0], [1, 0, 0]], dtype=int),
    ),
    5: (
        np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
        np.array([[1, 0, 1], [1, 0, 0], [1, 0, 1]], dtype=int),
    ),
    6: (
        np.array([[1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 1], [1, 0, 1]], dtype=int),
    ),
    7: (
        np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=int),
    ),
}

GEN_RETRIES = 400
CONTROL_SEARCH_RETRIES = 120


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _bbox_distance(cell: tuple[int, int], box: tuple[int, int, int, int]) -> int:
    r, c = cell
    r0, c0, r1, c1 = box
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return dr + dc


def _neighbors8(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _find_controls(grid: np.ndarray) -> list[tuple[int, int]]:
    controls = [tuple(map(int, rc)) for rc in np.argwhere(grid == Color.ORANGE)]
    assert 3 <= len(controls) <= 5, "expected 3-5 orange control pixels"
    for r, c in controls:
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and (nr, nc) != (r, c):
                assert int(grid[nr, nc]) != Color.ORANGE, "control pixels must be isolated"
    return controls


def _panel_candidates(grid: np.ndarray) -> tuple[set[tuple[int, int]], list[dict]]:
    height, width = map(int, grid.shape)
    grey_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == Color.GREY)}
    assert grey_cells, "expected at least one grey symbol panel"

    candidates: list[dict] = []
    for r0 in range(1, height - 3):
        for c0 in range(0, width - 2):
            window = grid[r0 : r0 + 3, c0 : c0 + 3]
            if not set(map(int, np.unique(window))) <= {Color.BLACK, Color.GREY}:
                continue
            count = int(np.sum(window == Color.GREY))
            if not (1 <= count <= 7):
                continue
            grey_in_window = {
                (r0 + int(rr), c0 + int(cc))
                for rr, cc in np.argwhere(window == Color.GREY)
            }
            if not grey_in_window:
                continue
            rows = [cell[0] for cell in grey_in_window]
            cols = [cell[1] for cell in grey_in_window]
            if max(rows) - min(rows) != 2 or max(cols) - min(cols) != 2:
                continue
            cells = frozenset((r0 + dr, c0 + dc) for dr in range(3) for dc in range(3))
            candidates.append(
                {
                    "top": r0,
                    "left": c0,
                    "box": (r0, c0, r0 + 2, c0 + 2),
                    "center": (r0 + 1, c0 + 1),
                    "count": count,
                    "grey_cells": grey_in_window,
                    "window_cells": cells,
                }
            )
    return grey_cells, candidates


def _solve_panels(grid: np.ndarray) -> list[dict]:
    grey_cells, candidates = _panel_candidates(grid)
    by_grey: dict[tuple[int, int], list[int]] = {cell: [] for cell in grey_cells}
    for idx, cand in enumerate(candidates):
        for cell in cand["grey_cells"]:
            if cell in by_grey:
                by_grey[cell].append(idx)
    assert all(by_grey[cell] for cell in grey_cells), "every grey cell must belong to some 3x3 panel"

    solutions: list[list[int]] = []

    def backtrack(uncovered: set[tuple[int, int]], used_windows: set[tuple[int, int]], chosen: list[int]) -> None:
        if len(solutions) > 1:
            return
        if not uncovered:
            solutions.append(chosen.copy())
            return
        pivot = min(uncovered, key=lambda cell: len(by_grey[cell]))
        for idx in by_grey[pivot]:
            cand = candidates[idx]
            if not cand["grey_cells"] <= uncovered:
                continue
            if cand["window_cells"] & used_windows:
                continue
            chosen.append(idx)
            backtrack(uncovered - cand["grey_cells"], used_windows | cand["window_cells"], chosen)
            chosen.pop()

    backtrack(set(grey_cells), set(), [])
    assert len(solutions) == 1, "symbol panels must have a unique non-overlapping 3x3 cover"
    panels = [candidates[idx] for idx in solutions[0]]
    panels.sort(key=lambda panel: (panel["top"], panel["left"]))
    assert 3 <= len(panels) <= 5, "expected 3-5 symbol panels"
    return panels


def _ray_spec(grid: np.ndarray, panels: list[dict], control: tuple[int, int]) -> dict:
    r, c = control
    dists = [_manhattan(control, panel["center"]) for panel in panels]
    best = min(dists)
    nearest = [idx for idx, dist in enumerate(dists) if dist == best]
    assert len(nearest) == 1, "each control must have a unique nearest panel"
    panel_idx = nearest[0]
    panel = panels[panel_idx]
    distance = int(panel["count"])

    d_top = r
    d_bot = (grid.shape[0] - 1) - r
    going_up = d_top <= d_bot
    dr = -1 if going_up else 1
    color = Color.RED if going_up else Color.GREEN
    panel_cells = set().union(*(panel["window_cells"] for panel in panels))

    painted: list[tuple[int, int]] = []
    blocker = None
    for step in range(1, distance + 1):
        nr = r + dr * step
        nc = c
        if not (0 <= nr < grid.shape[0]):
            break
        if (nr, nc) in panel_cells:
            blocker = Color.GREY
            break
        value = int(grid[nr, nc])
        if value != Color.BLACK:
            blocker = value
            break
        painted.append((nr, nc))

    return {
        "control": control,
        "panel_index": panel_idx,
        "panel": panel,
        "distance": distance,
        "direction": "up" if going_up else "down",
        "tie": d_top == d_bot,
        "color": color,
        "paint": painted,
        "blocked": len(painted) < distance,
        "blocked_by_panel": blocker == Color.GREY,
        "blocker": blocker,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each control pixel's nearest grey symbol panel into a distance, then cast a colored vertical ray toward the nearer edge bar.

    Input structure:
    - A black grid has a full red top edge bar and a full green bottom edge bar.
    - Between those bars sit 3-5 isolated orange control pixels and 3-5 embedded 3x3 black/grey symbol panels.
    - Each symbol panel is a unique non-overlapping 3x3 window, and every control has one unique nearest panel by Manhattan distance to the panel center.

    Rule:
    - Recover each 3x3 panel and decode its distance as the number of grey cells inside it.
    - For every orange control, read the distance from its nearest panel center.
    - If the control is at least as close to the top bar as to the bottom bar, paint up to that many black cells upward in red; otherwise paint downward in green.
    - Start from the adjacent cell, stop before the first non-black cell, and leave the bars, control pixels, and symbol panels unchanged.

    Key insight:
    - The grey symbol does not map to a color or shape output; it supplies the ray length parameter, while the nearer edge bar independently determines the ray direction and paint color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 7 and width >= 7, "grid is too small for bars, panels, and controls"
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.RED, Color.GREEN, Color.GREY, Color.ORANGE}
    assert np.all(g[0, :] == Color.RED), "expected a full red top bar"
    assert np.all(g[-1, :] == Color.GREEN), "expected a full green bottom bar"

    panels = _solve_panels(g)
    controls = _find_controls(g)

    out = g.copy()
    planned: dict[tuple[int, int], int] = {}
    for control in controls:
        spec = _ray_spec(g, panels, control)
        for cell in spec["paint"]:
            assert int(g[cell]) == Color.BLACK, "rays may only overwrite black input cells"
            prev = planned.get(cell)
            assert prev in (None, spec["color"]), "independent rays must not conflict"
            planned[cell] = spec["color"]

    for (r, c), color in planned.items():
        out[r, c] = color
    return out


def _render_panel(grid: np.ndarray, top: int, left: int, pattern: np.ndarray) -> None:
    window = np.where(pattern == 1, Color.GREY, Color.BLACK)
    grid[top : top + 3, left : left + 3] = window


def _sample_panel_layout(height: int, width: int, n_panels: int) -> list[tuple[int, int]] | None:
    spots = [(r, c) for r in range(1, height - 3) for c in range(0, width - 2)]
    random.shuffle(spots)
    chosen: list[tuple[int, int]] = []
    for r, c in spots:
        box = (r, c, r + 2, c + 2)
        ok = True
        for rr, cc in chosen:
            other = (rr, cc, rr + 2, cc + 2)
            if not (box[2] + 1 < other[0] or other[2] + 1 < box[0] or box[3] + 1 < other[1] or other[3] + 1 < box[1]):
                ok = False
                break
        if ok:
            chosen.append((r, c))
            if len(chosen) == n_panels:
                return chosen
    return None


def _build_control_candidates(grid: np.ndarray, panels: list[dict]) -> list[dict]:
    candidates: list[dict] = []
    height, width = map(int, grid.shape)
    panel_boxes = [panel["box"] for panel in panels]
    for r in range(1, height - 1):
        for c in range(width):
            if int(grid[r, c]) != Color.BLACK:
                continue
            if min(_bbox_distance((r, c), box) for box in panel_boxes) < 2:
                continue
            try:
                spec = _ray_spec(grid, panels, (r, c))
            except AssertionError:
                continue
            if not spec["paint"]:
                continue
            candidates.append(spec)
    random.shuffle(candidates)

    reduced: list[dict] = []
    buckets: dict[tuple[int, str, bool, bool], list[dict]] = {}
    for spec in candidates:
        key = (spec["panel_index"], spec["direction"], spec["blocked_by_panel"], spec["tie"])
        bucket = buckets.setdefault(key, [])
        if any(existing["control"][1] == spec["control"][1] for existing in bucket):
            continue
        if len(bucket) >= 4:
            continue
        bucket.append(spec)
        reduced.append(spec)
    return reduced


def _compatible(candidate: dict, chosen: list[dict]) -> bool:
    r, c = candidate["control"]
    for other in chosen:
        rr, cc = other["control"]
        if c == cc:
            return False
        if abs(r - rr) <= 1 and abs(c - cc) <= 1:
            return False
    return True


def _meets_selection_constraints(chosen: list[dict], require_tie: bool) -> bool:
    if not (3 <= len(chosen) <= 5):
        return False
    if not any(spec["direction"] == "up" for spec in chosen):
        return False
    if not any(spec["direction"] == "down" for spec in chosen):
        return False
    if require_tie and not any(spec["tie"] for spec in chosen):
        return False
    if not any(spec["blocked_by_panel"] for spec in chosen):
        return False
    if not any(not spec["blocked"] for spec in chosen):
        return False
    if len({spec["panel_index"] for spec in chosen}) < 2:
        return False
    if len({spec["distance"] for spec in chosen}) < 2:
        return False
    return True


def _select_controls(candidates: list[dict], n_controls: int, require_tie: bool) -> list[dict] | None:
    candidates = list(candidates)
    for _ in range(240):
        random.shuffle(candidates)
        chosen: list[dict] = []
        required_preds = [
            lambda spec: spec["blocked_by_panel"],
            lambda spec: not spec["blocked"],
            lambda spec: spec["direction"] == "up",
            lambda spec: spec["direction"] == "down",
        ]
        if require_tie:
            required_preds.append(lambda spec: spec["tie"])
        random.shuffle(required_preds)

        feasible = True
        for pred in required_preds:
            if len(chosen) >= n_controls or any(pred(spec) for spec in chosen):
                continue
            options = [spec for spec in candidates if pred(spec) and _compatible(spec, chosen)]
            if not options:
                feasible = False
                break
            chosen.append(random.choice(options))
        if not feasible:
            continue

        while len(chosen) < n_controls:
            panels_used = {spec["panel_index"] for spec in chosen}
            distances_used = {spec["distance"] for spec in chosen}
            options = [spec for spec in candidates if _compatible(spec, chosen)]
            if not options:
                break

            def score(spec: dict) -> tuple[int, float]:
                value = 0
                if spec["panel_index"] not in panels_used:
                    value += 4
                if spec["distance"] not in distances_used:
                    value += 3
                if spec["blocked_by_panel"] and not any(x["blocked_by_panel"] for x in chosen):
                    value += 5
                if (not spec["blocked"]) and not any(not x["blocked"] for x in chosen):
                    value += 5
                if spec["direction"] == "up" and not any(x["direction"] == "up" for x in chosen):
                    value += 4
                if spec["direction"] == "down" and not any(x["direction"] == "down" for x in chosen):
                    value += 4
                if require_tie and spec["tie"] and not any(x["tie"] for x in chosen):
                    value += 5
                return value, random.random()

            options.sort(key=score, reverse=True)
            chosen.append(options[0])

        if _meets_selection_constraints(chosen, require_tie):
            return chosen
    return None


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    panels = _solve_panels(grid)
    controls = _find_controls(grid)
    out = grid.copy()
    for control in controls:
        spec = _ray_spec(grid, panels, control)
        for r, c in spec["paint"]:
            out[r, c] = spec["color"]
    return out


def _fast_semantic_checks(grid: np.ndarray, panels: list[dict], chosen: list[dict], require_tie: bool) -> bool:
    if not _meets_selection_constraints(chosen, require_tie):
        return False
    if len({panel["count"] for panel in panels}) != len(panels):
        return False
    painted = {cell for spec in chosen for cell in spec["paint"]}
    if not painted:
        return False
    for spec in chosen:
        r, c = spec["control"]
        if int(grid[r, c]) != Color.BLACK:
            return False
        for cell in spec["paint"]:
            if int(grid[cell]) != Color.BLACK:
                return False
    return True


def _validate_semantics(grid: np.ndarray) -> None:
    panels = _solve_panels(grid)
    controls = _find_controls(grid)
    assert 3 <= len(panels) <= 5
    assert 3 <= len(controls) <= 5
    assert len({panel["count"] for panel in panels}) == len(panels), "panel counts should be distinct"

    specs = [_ray_spec(grid, panels, control) for control in controls]
    assert any(spec["direction"] == "up" for spec in specs)
    assert any(spec["direction"] == "down" for spec in specs)
    assert any(spec["blocked_by_panel"] for spec in specs), "need a panel-blocked ray to exclude bar-only shortcuts"
    assert any(not spec["blocked"] for spec in specs), "need a fully extended ray example"
    assert len({spec["panel_index"] for spec in specs}) >= 2
    assert len({spec["distance"] for spec in specs}) >= 2

    out = main(grid)
    expected = _oracle_output(grid)
    assert np.array_equal(out, expected)
    changed = {(int(r), int(c)) for r, c in np.argwhere(out != grid)}
    painted = {cell for spec in specs for cell in spec["paint"]}
    assert changed == painted
    for panel in panels:
        assert np.array_equal(out[panel["top"] : panel["top"] + 3, panel["left"] : panel["left"] + 3], grid[panel["top"] : panel["top"] + 3, panel["left"] : panel["left"] + 3])
    for r, c in controls:
        assert int(out[r, c]) == Color.ORANGE


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(19, 25)
    width = random.randint(18, 24)
    n_panels = random.randint(3, 5)
    n_controls = random.randint(3, 5)

    grid = np.zeros((height, width), dtype=int)
    grid[0, :] = Color.RED
    grid[-1, :] = Color.GREEN

    panel_positions = _sample_panel_layout(height, width, n_panels)
    if panel_positions is None:
        return None

    counts = random.sample(list(PANEL_PATTERNS.keys()), n_panels)
    random.shuffle(counts)
    panels_for_generation: list[dict] = []
    for (top, left), count in zip(panel_positions, counts):
        pattern = random.choice(PANEL_PATTERNS[count])
        _render_panel(grid, top, left, pattern)
        panels_for_generation.append(
            {
                "top": top,
                "left": left,
                "box": (top, left, top + 2, left + 2),
                "center": (top + 1, left + 1),
                "count": count,
            }
        )

    try:
        panels = _solve_panels(grid)
    except AssertionError:
        return None
    if len(panels) != n_panels:
        return None

    candidates = _build_control_candidates(grid, panels)
    if not candidates:
        return None

    require_tie = bool(height % 2 == 1 and any(spec["tie"] for spec in candidates) and random.random() < 0.6)
    for _ in range(CONTROL_SEARCH_RETRIES):
        random.shuffle(candidates)
        chosen = _select_controls(candidates, n_controls, require_tie)
        if chosen is None:
            continue
        out = grid.copy()
        expected = out.copy()
        for spec in chosen:
            r, c = spec["control"]
            out[r, c] = Color.ORANGE
            expected[r, c] = Color.ORANGE
            for rr, cc in spec["paint"]:
                expected[rr, cc] = spec["color"]
        if not _fast_semantic_checks(grid, panels, chosen, require_tie):
            continue
        try:
            actual = main(out)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue
        return out
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symbol-decoding ray-cast input")
