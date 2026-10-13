from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r: int, c: int, shape: tuple[int, int]) -> Iterable[tuple[int, int]]:
    h, w = shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _detect_rectangles(grid: np.ndarray) -> list[dict]:
    rectangles: list[dict] = []
    colors = [int(color) for color in np.unique(grid) if int(color) not in (BG, GREY)]
    for color in colors:
        pts = np.argwhere(grid == color)
        assert pts.size > 0, "foreground color must be present"
        (r0, c0), (r1, c1) = pts.min(axis=0), pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = sub.shape
        assert 3 <= h <= 6 and 3 <= w <= 6, "rectangles must be between 3x3 and 6x6"

        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        assert int(sub[0, 0]) == color
        assert int(sub[0, -1]) == color
        assert int(sub[-1, 0]) == color
        assert int(sub[-1, -1]) == color
        assert not np.any((sub != color) & (sub != BG)), "rectangle bbox may only contain its color and border gaps"
        if h > 2 and w > 2:
            assert np.all(sub[1:-1, 1:-1] == color), "rectangle interior must be solid"

        gaps: list[dict] = []
        for rr in range(h):
            for cc in range(w):
                if not border[rr, cc] or int(sub[rr, cc]) != BG:
                    continue
                assert (rr, cc) not in {
                    (0, 0),
                    (0, w - 1),
                    (h - 1, 0),
                    (h - 1, w - 1),
                }, "gaps cannot be corners"
                if rr == 0:
                    side = "top"
                elif rr == h - 1:
                    side = "bottom"
                elif cc == 0:
                    side = "left"
                else:
                    side = "right"
                gaps.append(
                    {
                        "pos": (r0 + rr, c0 + cc),
                        "side": side,
                        "dir": SIDE_TO_DIR[side],
                    }
                )

        assert 1 <= len(gaps) <= 4, "expected 1-4 border gaps per rectangle"
        assert int(pts.shape[0]) == h * w - len(gaps), "all missing cells must be border gaps"

        rectangles.append(
            {
                "color": color,
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "gaps": gaps,
            }
        )

    rectangles.sort(key=lambda rect: (rect["r0"], rect["c0"], rect["color"]))
    assert rectangles, "expected at least one rectangle"
    return rectangles


def _simulate_tentacles(grid: np.ndarray, rectangles: list[dict]) -> tuple[list[dict], list[tuple[int, int, int, tuple[int, int]]]]:
    h, w = grid.shape
    occupied = {tuple(map(int, pt)) for pt in np.argwhere(grid != BG)}

    rays: list[dict] = []
    for rect_index, rect in enumerate(rectangles):
        for gap_index, gap in enumerate(rect["gaps"]):
            rays.append(
                {
                    "rect_index": rect_index,
                    "gap_index": gap_index,
                    "gap": tuple(gap["pos"]),
                    "dir": tuple(gap["dir"]),
                    "color": int(rect["color"]),
                    "path": [],
                    "stop_reason": None,
                }
            )
    rays.sort(key=lambda ray: (ray["gap"][0], ray["gap"][1], ray["dir"][0], ray["dir"][1], ray["color"]))

    events: list[tuple[int, int, int, tuple[int, int]]] = []
    active = list(range(len(rays)))
    distance = 1
    while active:
        proposals: dict[tuple[int, int], list[int]] = {}
        next_active: list[int] = []
        for ray_index in active:
            gap_r, gap_c = rays[ray_index]["gap"]
            dr, dc = rays[ray_index]["dir"]
            nr = gap_r + dr * distance
            nc = gap_c + dc * distance
            if not (0 <= nr < h and 0 <= nc < w):
                rays[ray_index]["stop_reason"] = "boundary"
                continue
            if (nr, nc) in occupied:
                rays[ray_index]["stop_reason"] = "occupied"
                continue
            proposals.setdefault((nr, nc), []).append(ray_index)

        accepted: list[tuple[int, tuple[int, int]]] = []
        for pos, ids in proposals.items():
            if len(ids) == 1:
                accepted.append((ids[0], pos))
            else:
                for ray_index in ids:
                    rays[ray_index]["stop_reason"] = "collision"

        accepted.sort(key=lambda item: item[0])
        for ray_index, pos in accepted:
            rays[ray_index]["path"].append(pos)
            occupied.add(pos)
            events.append((distance, ray_index, rays[ray_index]["color"], pos))
            next_active.append(ray_index)

        active = next_active
        distance += 1

    return rays, events


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect near-solid rectangles with border gaps, emit outward tentacles from the gaps,
    recolor adjacent grey targets by earliest contact, then repair the gaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rectangles = _detect_rectangles(g)
    rays, events = _simulate_tentacles(g, rectangles)
    assert rays, "expected at least one gap ray"

    out = g.copy()
    for _distance, _ray_index, color, (r, c) in events:
        out[r, c] = color

    grey_positions = {tuple(map(int, pt)) for pt in np.argwhere(g == GREY)}
    claims: dict[tuple[int, int], tuple[int, int, int]] = {}
    for distance, ray_index, color, (r, c) in events:
        for nr, nc in _neighbors4(r, c, g.shape):
            if (nr, nc) not in grey_positions:
                continue
            candidate = (distance, ray_index, color)
            current = claims.get((nr, nc))
            if current is None or candidate[:2] < current[:2]:
                claims[(nr, nc)] = candidate

    for (r, c), (_distance, _ray_index, color) in claims.items():
        out[r, c] = color

    for rect in rectangles:
        color = int(rect["color"])
        for gap in rect["gaps"]:
            gr, gc = gap["pos"]
            out[gr, gc] = color

    return out


GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 26)
RECT_COUNT_CHOICES = (2, 3, 4)
RECT_COUNT_WEIGHTS = (2, 3, 3)
RECT_SIZE_RANGE = (3, 6)
RECT_MARGIN = 1
RECT_PLACE_RETRIES = 120
GEN_RETRIES = 500
MAX_GREY_DISTRACTORS = 2


def _boxes_apart(a: dict, b: dict, pad: int = RECT_MARGIN) -> bool:
    return (
        a["r1"] + pad < b["r0"]
        or b["r1"] + pad < a["r0"]
        or a["c1"] + pad < b["c0"]
        or b["c1"] + pad < a["c0"]
    )


def _build_full_rect_grid(shape: tuple[int, int], rectangles: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for rect in rectangles:
        grid[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = rect["color"]
    return grid


def _trace_static(grid: np.ndarray, gap: tuple[int, int], direction: tuple[int, int]) -> tuple[list[tuple[int, int]], str]:
    h, w = grid.shape
    dr, dc = direction
    r0, c0 = gap
    path: list[tuple[int, int]] = []
    distance = 1
    while True:
        nr = r0 + dr * distance
        nc = c0 + dc * distance
        if not (0 <= nr < h and 0 <= nc < w):
            return path, "boundary"
        if int(grid[nr, nc]) != BG:
            return path, "occupied"
        path.append((nr, nc))
        distance += 1


def _gap_candidates_for_rect(rect: dict, grid: np.ndarray, rect_index: int) -> list[dict]:
    candidates: list[dict] = []
    r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
    for c in range(c0 + 1, c1):
        for pos, side in (((r0, c), "top"), ((r1, c), "bottom")):
            path, stop = _trace_static(grid, pos, SIDE_TO_DIR[side])
            candidates.append(
                {
                    "rect_index": rect_index,
                    "pos": pos,
                    "side": side,
                    "dir": SIDE_TO_DIR[side],
                    "path_len": len(path),
                    "stop": stop,
                }
            )
    for r in range(r0 + 1, r1):
        for pos, side in (((r, c0), "left"), ((r, c1), "right")):
            path, stop = _trace_static(grid, pos, SIDE_TO_DIR[side])
            candidates.append(
                {
                    "rect_index": rect_index,
                    "pos": pos,
                    "side": side,
                    "dir": SIDE_TO_DIR[side],
                    "path_len": len(path),
                    "stop": stop,
                }
            )
    return candidates


def _bbox_cells(rectangles: list[dict]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for rect in rectangles:
        for r in range(rect["r0"], rect["r1"] + 1):
            for c in range(rect["c0"], rect["c1"] + 1):
                cells.add((r, c))
    return cells


def _claim_summary(records: list[tuple[int, int, int]]) -> tuple[tuple[int, int, int] | None, bool]:
    by_ray: dict[int, tuple[int, int]] = {}
    for distance, ray_index, color in records:
        prev = by_ray.get(ray_index)
        if prev is None or distance < prev[0]:
            by_ray[ray_index] = (distance, color)
    ranked = sorted((distance, ray_index, color) for ray_index, (distance, color) in by_ray.items())
    if not ranked:
        return None, False
    if len(ranked) >= 2 and ranked[1][0] == ranked[0][0]:
        return None, False
    return ranked[0], len(ranked) >= 2


def _can_place_grey(pos: tuple[int, int], chosen: set[tuple[int, int]]) -> bool:
    r, c = pos
    for rr, cc in chosen:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    return True


def _far_from_rectangles(pos: tuple[int, int], rectangles: list[dict]) -> bool:
    r, c = pos
    for rect in rectangles:
        if rect["r0"] - 1 <= r <= rect["r1"] + 1 and rect["c0"] - 1 <= c <= rect["c1"] + 1:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    rect_count = rng.choices(RECT_COUNT_CHOICES, weights=RECT_COUNT_WEIGHTS, k=1)[0]
    want_blocked = rng.random() < 0.55
    want_contest = rng.random() < 0.35
    want_distractor = rng.random() < 0.5

    colors = rng.sample(PALETTE, rect_count)
    rectangles: list[dict] = []
    for color in colors:
        placed = False
        for _ in range(RECT_PLACE_RETRIES):
            rh = rng.randint(*RECT_SIZE_RANGE)
            rw = rng.randint(*RECT_SIZE_RANGE)
            if h - rh - 1 < 1 or w - rw - 1 < 1:
                return None
            r0 = rng.randint(1, h - rh - 1)
            c0 = rng.randint(1, w - rw - 1)
            rect = {
                "color": color,
                "r0": r0,
                "c0": c0,
                "r1": r0 + rh - 1,
                "c1": c0 + rw - 1,
            }
            if all(_boxes_apart(rect, other) for other in rectangles):
                rectangles.append(rect)
                placed = True
                break
        if not placed:
            return None

    rectangles.sort(key=lambda rect: (rect["r0"], rect["c0"], rect["color"]))
    full_grid = _build_full_rect_grid((h, w), rectangles)
    candidate_map = {
        index: _gap_candidates_for_rect(rect, full_grid, index)
        for index, rect in enumerate(rectangles)
    }

    chosen_by_rect: dict[int, list[dict]] = {}
    for index, candidates in candidate_map.items():
        if not candidates:
            return None
        gap_count = rng.randint(1, min(4, len(candidates)))
        chosen_by_rect[index] = rng.sample(candidates, gap_count)

    selected = [cand for picked in chosen_by_rect.values() for cand in picked]
    if len(selected) > 1 and len({cand["side"] for cand in selected}) < 2:
        return None
    if not any(cand["stop"] == "boundary" for cand in selected):
        return None

    blocked_pool = [cand for cands in candidate_map.values() for cand in cands if cand["stop"] == "occupied"]
    if want_blocked:
        if not blocked_pool:
            return None
        if not any(cand["stop"] == "occupied" for cand in selected):
            forced = rng.choice(blocked_pool)
            picked = chosen_by_rect[forced["rect_index"]]
            if all(cand["pos"] != forced["pos"] for cand in picked):
                if len(picked) < 4:
                    picked.append(forced)
                else:
                    picked[rng.randrange(len(picked))] = forced

    for index, rect in enumerate(rectangles):
        rect["gaps"] = [
            {"pos": tuple(cand["pos"]), "side": cand["side"], "dir": tuple(cand["dir"])}
            for cand in chosen_by_rect[index]
        ]

    grid = full_grid.copy()
    for rect in rectangles:
        for gap in rect["gaps"]:
            r, c = gap["pos"]
            grid[r, c] = BG

    rays, events = _simulate_tentacles(grid, rectangles)
    if not events:
        return None

    static_check = {tuple(ray["gap"]) + tuple(ray["dir"]): _trace_static(grid, tuple(ray["gap"]), tuple(ray["dir"])) for ray in rays}
    for ray in rays:
        key = tuple(ray["gap"]) + tuple(ray["dir"])
        static_path, static_stop = static_check[key]
        if ray["stop_reason"] != static_stop or len(ray["path"]) != len(static_path):
            return None
        if static_stop == "collision":
            return None

    if want_blocked and not any(ray["stop_reason"] == "occupied" for ray in rays):
        return None
    if not any(ray["stop_reason"] == "boundary" for ray in rays):
        return None

    ray_cells = {pos for _distance, _ray_index, _color, pos in events}
    rect_cells = _bbox_cells(rectangles)
    adjacency: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for distance, ray_index, color, (r, c) in events:
        for nr, nc in _neighbors4(r, c, grid.shape):
            if int(grid[nr, nc]) != BG:
                continue
            if (nr, nc) in ray_cells or (nr, nc) in rect_cells:
                continue
            adjacency.setdefault((nr, nc), []).append((distance, ray_index, color))

    useful_cells: list[tuple[int, int]] = []
    contest_cells: list[tuple[int, int]] = []
    for pos, records in adjacency.items():
        winner, contested = _claim_summary(records)
        if winner is None:
            continue
        useful_cells.append(pos)
        if contested:
            contest_cells.append(pos)

    if want_contest and not contest_cells:
        return None
    if len(useful_cells) < 2:
        return None

    grey_positions: set[tuple[int, int]] = set()
    if want_contest:
        rng.shuffle(contest_cells)
        for pos in contest_cells:
            if _can_place_grey(pos, grey_positions):
                grey_positions.add(pos)
                break
        if not grey_positions:
            return None

    rng.shuffle(useful_cells)
    desired_adjacent = min(len(useful_cells), rng.randint(2, 5))
    for pos in useful_cells:
        if len(grey_positions) >= desired_adjacent:
            break
        if _can_place_grey(pos, grey_positions):
            grey_positions.add(pos)
    if not grey_positions:
        return None

    distractor_candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if int(grid[r, c]) == BG
        and (r, c) not in rect_cells
        and (r, c) not in ray_cells
        and (r, c) not in adjacency
        and _far_from_rectangles((r, c), rectangles)
    ]
    rng.shuffle(distractor_candidates)
    distractor_goal = rng.randint(1, MAX_GREY_DISTRACTORS) if want_distractor else 0
    added_distractors = 0
    for pos in distractor_candidates:
        if added_distractors >= distractor_goal:
            break
        if _can_place_grey(pos, grey_positions):
            grey_positions.add(pos)
            added_distractors += 1
    if want_distractor and distractor_goal > 0 and added_distractors == 0:
        return None

    for r, c in grey_positions:
        grid[r, c] = GREY

    try:
        out = main(grid)
    except AssertionError:
        return None

    grey_in = {tuple(map(int, pt)) for pt in np.argwhere(grid == GREY)}
    recolored = sum(int(out[r, c] != GREY) for r, c in grey_in)
    if recolored == 0:
        return None
    if want_distractor and added_distractors > 0 and not any(int(out[r, c]) == GREY for r, c in grey_in):
        return None
    if np.array_equal(grid, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
