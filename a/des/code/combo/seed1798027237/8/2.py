from collections import deque

import numpy as np


DIR_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIRS = tuple(DIR_TO_VEC)

GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)
ARROW_COUNT_RANGE = (2, 6)
SHAFT_LEN_RANGE = (1, 5)
GEN_RETRIES = 400

_RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors8(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _arrow_rel_cells(direction: str, shaft_len: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    if direction == "right":
        cells.update({(-1, -shaft_len), (1, -shaft_len)})
        cells.update((0, -step) for step in range(1, shaft_len + 1))
    elif direction == "left":
        cells.update({(-1, shaft_len), (1, shaft_len)})
        cells.update((0, step) for step in range(1, shaft_len + 1))
    elif direction == "up":
        cells.update({(shaft_len, -1), (shaft_len, 1)})
        cells.update((step, 0) for step in range(1, shaft_len + 1))
    elif direction == "down":
        cells.update({(-shaft_len, -1), (-shaft_len, 1)})
        cells.update((-step, 0) for step in range(1, shaft_len + 1))
    else:
        raise AssertionError(f"unknown direction: {direction}")
    return cells


def _arrow_cells(tip: tuple[int, int], direction: str, shaft_len: int) -> set[tuple[int, int]]:
    tr, tc = map(int, tip)
    return {(tr + dr, tc + dc) for dr, dc in _arrow_rel_cells(direction, shaft_len)}


def _forward_space(tip: tuple[int, int], direction: str, h: int, w: int) -> int:
    r, c = map(int, tip)
    if direction == "up":
        return int(r)
    if direction == "down":
        return int(h - 1 - r)
    if direction == "left":
        return int(c)
    if direction == "right":
        return int(w - 1 - c)
    raise AssertionError(f"unknown direction: {direction}")


def _ray_cells(tip: tuple[int, int], direction: str, length: int, h: int, w: int) -> list[tuple[int, int]]:
    dr, dc = DIR_TO_VEC[direction]
    tr, tc = map(int, tip)
    cells: list[tuple[int, int]] = []
    for step in range(1, int(length) + 1):
        rr = int(tr + dr * step)
        cc = int(tc + dc * step)
        if not (0 <= rr < h and 0 <= cc < w):
            break
        cells.append((rr, cc))
    return cells


def _fit_arrow(coords: list[tuple[int, int]]) -> dict:
    pts = {(int(r), int(c)) for r, c in coords}
    matches: list[dict] = []
    for tip in pts:
        for direction in DIRS:
            for shaft_len in range(SHAFT_LEN_RANGE[0], SHAFT_LEN_RANGE[1] + 1):
                if len(pts) != shaft_len + 3:
                    continue
                if _arrow_cells(tip, direction, shaft_len) == pts:
                    matches.append(
                        {
                            "tip": (int(tip[0]), int(tip[1])),
                            "direction": direction,
                            "shaft_len": int(shaft_len),
                        }
                    )
    assert len(matches) == 1, f"expected exactly one arrow match, found {len(matches)} for {sorted(pts)}"
    return matches[0]


def _detect_arrows(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    arrows: list[dict] = []
    for comp in _components(g, bg):
        fit = _fit_arrow(comp["coords"])
        arrows.append(
            {
                "color": int(comp["color"]),
                "tip": fit["tip"],
                "direction": fit["direction"],
                "shaft_len": int(fit["shaft_len"]),
                "cells": {(int(r), int(c)) for r, c in comp["coords"]},
            }
        )
    arrows.sort(key=lambda item: (item["tip"][0], item["tip"][1]))
    return arrows, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect monochrome arrow sprites, then extend a same-color ray beyond each tip.

    Each arrow is preserved as drawn in the input. The extension length is twice the
    shaft length; the newly grown strip starts just beyond the tip so the tip itself
    remains the original end of the arrowhead.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    arrows, _bg = _detect_arrows(g)

    out = g.copy()
    for arrow in arrows:
        ray = _ray_cells(arrow["tip"], arrow["direction"], 2 * int(arrow["shaft_len"]), h, w)
        for r, c in ray:
            out[int(r), int(c)] = int(arrow["color"])
    return out


def _mark_forbidden(forbidden: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = forbidden.shape
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    forbidden[rr, cc] = True


def _candidate_specs(
    forbidden: np.ndarray,
    directions: tuple[str, ...] = DIRS,
    shaft_lens: tuple[int, ...] | None = None,
    predicate=None,
) -> list[dict]:
    h, w = forbidden.shape
    if shaft_lens is None:
        shaft_lens = tuple(range(SHAFT_LEN_RANGE[0], SHAFT_LEN_RANGE[1] + 1))

    candidates: list[dict] = []
    for direction in directions:
        for shaft_len in shaft_lens:
            for r in range(h):
                for c in range(w):
                    tip = (int(r), int(c))
                    cells = _arrow_cells(tip, direction, shaft_len)
                    if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in cells):
                        continue
                    if any(forbidden[rr, cc] for rr, cc in cells):
                        continue
                    if predicate is not None and not predicate(tip, direction, shaft_len):
                        continue
                    candidates.append(
                        {
                            "tip": tip,
                            "direction": direction,
                            "shaft_len": int(shaft_len),
                            "cells": cells,
                        }
                    )
    return candidates


def _simulate_events(arrows: list[dict], h: int, w: int) -> tuple[bool, bool]:
    original = set()
    for arrow in arrows:
        original.update(arrow["cells"])

    painted: set[tuple[int, int]] = set()
    saw_overlap = False
    saw_truncation = False

    for arrow in sorted(arrows, key=lambda item: (item["tip"][0], item["tip"][1])):
        ray_len = 2 * int(arrow["shaft_len"])
        ray = _ray_cells(arrow["tip"], arrow["direction"], ray_len, h, w)
        if len(ray) < ray_len:
            saw_truncation = True
        for cell in ray:
            if cell in painted or cell in (original - arrow["cells"]):
                saw_overlap = True
            painted.add(cell)

    return saw_overlap, saw_truncation


def _place_overlap_pair(forbidden: np.ndarray) -> list[dict] | None:
    h, w = forbidden.shape
    for _ in range(200):
        shaft_len_a = int(_RNG.integers(SHAFT_LEN_RANGE[0], SHAFT_LEN_RANGE[1] + 1))
        shaft_len_b = int(_RNG.integers(SHAFT_LEN_RANGE[0], SHAFT_LEN_RANGE[1] + 1))
        row = int(_RNG.integers(1, h - 1))

        left_tip_min = shaft_len_a
        left_tip_max = w - shaft_len_b - 5
        if left_tip_min > left_tip_max:
            continue
        left_tip_c = int(_RNG.integers(left_tip_min, left_tip_max + 1))

        max_delta = min(2 * shaft_len_a + 2 * shaft_len_b, (w - 1 - shaft_len_b) - left_tip_c)
        if max_delta < 4:
            continue
        delta = int(_RNG.integers(4, max_delta + 1))
        right_tip_c = int(left_tip_c + delta)

        spec_a = {
            "tip": (row, left_tip_c),
            "direction": "right",
            "shaft_len": shaft_len_a,
        }
        spec_b = {
            "tip": (row, right_tip_c),
            "direction": "left",
            "shaft_len": shaft_len_b,
        }
        spec_a["cells"] = _arrow_cells(spec_a["tip"], spec_a["direction"], spec_a["shaft_len"])
        spec_b["cells"] = _arrow_cells(spec_b["tip"], spec_b["direction"], spec_b["shaft_len"])

        if any(forbidden[r, c] for r, c in spec_a["cells"] | spec_b["cells"]):
            continue
        if spec_a["cells"] & spec_b["cells"]:
            continue
        return [spec_a, spec_b]
    return None


def _grid_from_arrows(h: int, w: int, bg: int, arrows: list[dict]) -> np.ndarray:
    grid = np.full((h, w), int(bg), dtype=int)
    for arrow in arrows:
        for r, c in arrow["cells"]:
            grid[int(r), int(c)] = int(arrow["color"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(_RNG.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    bg = int(_RNG.integers(0, 10))

    want_overlap = bool(_RNG.random() < 0.45)
    want_truncation = bool(_RNG.random() < 0.45)

    min_count, max_count = ARROW_COUNT_RANGE
    arrow_count = int(_RNG.integers(min_count, max_count + 1))
    if want_overlap:
        arrow_count = max(arrow_count, 2)

    forbidden = np.zeros((h, w), dtype=bool)
    arrows: list[dict] = []

    if want_overlap:
        pair = _place_overlap_pair(forbidden)
        if pair is None:
            return None
        arrows.extend(pair)
        for spec in pair:
            _mark_forbidden(forbidden, spec["cells"])

    if want_truncation:
        trunc_candidates = _candidate_specs(
            forbidden,
            predicate=lambda tip, direction, shaft_len: _forward_space(tip, direction, h, w) < 2 * int(shaft_len),
        )
        if not trunc_candidates:
            return None
        trunc_spec = trunc_candidates[int(_RNG.integers(len(trunc_candidates)))]
        arrows.append(trunc_spec)
        _mark_forbidden(forbidden, trunc_spec["cells"])

    while len(arrows) < arrow_count:
        require_full = bool(_RNG.random() < 0.6)
        pred = None
        if require_full:
            pred = lambda tip, direction, shaft_len: _forward_space(tip, direction, h, w) >= 2 * int(shaft_len)
        candidates = _candidate_specs(forbidden, predicate=pred)
        if not candidates:
            return None
        spec = candidates[int(_RNG.integers(len(candidates)))]
        arrows.append(spec)
        _mark_forbidden(forbidden, spec["cells"])

    if len({arrow["direction"] for arrow in arrows}) == 1:
        return None

    colors = [color for color in range(10) if color != bg]
    if len(arrows) > len(colors):
        return None
    _RNG.shuffle(colors)
    for arrow, color in zip(arrows, colors):
        arrow["color"] = int(color)

    grid = _grid_from_arrows(h, w, bg, arrows)

    detected, detected_bg = _detect_arrows(grid)
    if detected_bg != bg or len(detected) != len(arrows):
        return None
    if any(d["color"] != a["color"] for d, a in zip(sorted(detected, key=lambda x: x["color"]), sorted(arrows, key=lambda x: x["color"]))):
        return None

    saw_overlap, saw_truncation = _simulate_events(arrows, h, w)
    if want_overlap and not saw_overlap:
        return None
    if want_truncation and not saw_truncation:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow grid")
