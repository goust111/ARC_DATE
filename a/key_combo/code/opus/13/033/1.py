from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5

ORTH_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_FROM_VECTOR = {
    (0, 1): [(-1, 1), (1, 1)],   # right -> up-right, down-right
    (0, -1): [(-1, -1), (1, -1)],  # left -> up-left, down-left
    (1, 0): [(1, -1), (1, 1)],   # down -> down-left, down-right
    (-1, 0): [(-1, -1), (-1, 1)],  # up -> up-left, up-right
}
ALL_DIAGS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

GEN_ATTEMPTS = 500
_GEN_CALL_COUNT = 0


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _adjacent_of_color(grid: np.ndarray, cell: tuple[int, int], color: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for rr, cc in _neighbors4(r, c):
        if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == int(color):
            out.append((int(rr), int(cc)))
    return out


def _find_pairs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    pairs: list[dict] = []
    paired_reds: set[tuple[int, int]] = set()
    paired_greens: set[tuple[int, int]] = set()
    for rr, cc in zip(*np.where(g == GREEN), strict=False):
        green = (int(rr), int(cc))
        red_neighbors = _adjacent_of_color(g, green, RED)
        if len(red_neighbors) != 1:
            continue
        red = red_neighbors[0]
        if len(_adjacent_of_color(g, red, GREEN)) != 1:
            continue
        if red in paired_reds or green in paired_greens:
            continue
        vec = (green[0] - red[0], green[1] - red[1])
        if vec not in DIAG_FROM_VECTOR:
            continue
        paired_reds.add(red)
        paired_greens.add(green)
        pairs.append({"red": red, "green": green, "vec": vec})
    pairs.sort(key=lambda item: (item["green"][0], item["green"][1]))
    return pairs


def _simulate(
    grid: np.ndarray,
    *,
    all_four: bool = False,
    ignore_grey: bool = False,
    ignore_color: bool = False,
    erase_all_paired_reds_first: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    pairs = _find_pairs(g)
    original_rg = {
        (int(r), int(c))
        for r, c in zip(*np.where((g == RED) | (g == GREEN)), strict=False)
    }
    paired_reds = {pair["red"] for pair in pairs}
    blockers = set(original_rg)

    if erase_all_paired_reds_first:
        for red in paired_reds:
            blockers.discard(red)
            out[red] = BLACK

    traces: list[dict] = []
    for pair in pairs:
        red = pair["red"]
        green = pair["green"]
        vec = pair["vec"]

        if not erase_all_paired_reds_first:
            blockers.discard(red)
            out[red] = BLACK

        dirs = ALL_DIAGS if all_four else DIAG_FROM_VECTOR[vec]
        before_green = out == GREEN
        visible_added: set[tuple[int, int]] = set()
        ray_stops: list[str] = []

        for dr, dc in dirs:
            rr = green[0] + dr
            cc = green[1] + dc
            stop_kind = "edge"
            while _in_bounds(g.shape, rr, cc):
                if not ignore_grey and int(g[rr, cc]) == GREY:
                    stop_kind = "grey"
                    break
                if not ignore_color and (int(rr), int(cc)) in blockers:
                    stop_kind = "color"
                    break
                if not before_green[rr, cc]:
                    visible_added.add((int(rr), int(cc)))
                out[rr, cc] = GREEN
                rr += dr
                cc += dc
            ray_stops.append(stop_kind)

        traces.append(
            {
                "red": red,
                "green": green,
                "vec": vec,
                "ray_stops": ray_stops,
                "visible_added": visible_added,
            }
        )

    return out, traces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each paired red half and fire two green diagonal rays from the paired green half.

    Input structure:
    - A black background contains red (2), green (3), and gray (5) single pixels.
    - Some red-green cells form unique 4-neighbor dominoes; other red/green cells are unpaired distractors.
    - Gray cells never participate in pairing and instead act as ray blockers.

    Rule:
    - Detect every red-green 4-neighbor domino where the red touches exactly one green and that green touches exactly one red.
    - Process those dominoes in ascending order of the green cell's `(row, col)`.
    - For each domino, erase the red cell to black, then from the green cell shoot two green rays determined by the red-to-green direction:
      right -> up-right and down-right, left -> up-left and down-left, down -> down-left and down-right, up -> up-left and up-right.
    - Each ray starts from the next diagonal cell and stops before the first gray cell, before the grid edge, or before any original red/green cell that has not been erased.
      Rays may overwrite black cells and previously drawn green ray cells.

    Key insight:
    The paired red cell is both a membership cue and a direction code: its position relative to the green cell decides which two diagonals are allowed, while gray and surviving colored pixels terminate those rays.
    """
    out, _ = _simulate(grid)
    return out


def _touches_any_rg(grid: np.ndarray, cell: tuple[int, int], *, allow: set[tuple[int, int]] | None = None) -> bool:
    allowed = set() if allow is None else set(allow)
    r, c = cell
    for rr, cc in _neighbors4(r, c):
        if not _in_bounds(grid.shape, rr, cc):
            continue
        if (rr, cc) in allowed:
            continue
        if int(grid[rr, cc]) in (RED, GREEN):
            return True
    return False


def _can_place_domino(grid: np.ndarray, green: tuple[int, int], vec: tuple[int, int]) -> bool:
    red = (green[0] - vec[0], green[1] - vec[1])
    if not _in_bounds(grid.shape, *green) or not _in_bounds(grid.shape, *red):
        return False
    if int(grid[green]) != BLACK or int(grid[red]) != BLACK:
        return False
    if _touches_any_rg(grid, green, allow={red}):
        return False
    if _touches_any_rg(grid, red, allow={green}):
        return False
    return True


def _place_domino(grid: np.ndarray, green: tuple[int, int], vec: tuple[int, int]) -> dict:
    red = (green[0] - vec[0], green[1] - vec[1])
    grid[red] = RED
    grid[green] = GREEN
    return {"red": red, "green": green, "vec": vec}


def _raw_ray_cells(shape: tuple[int, int], green: tuple[int, int], vec: tuple[int, int]) -> list[list[tuple[int, int]]]:
    paths: list[list[tuple[int, int]]] = []
    for dr, dc in DIAG_FROM_VECTOR[vec]:
        rr = green[0] + dr
        cc = green[1] + dc
        path: list[tuple[int, int]] = []
        while _in_bounds(shape, rr, cc):
            path.append((int(rr), int(cc)))
            rr += dr
            cc += dc
        paths.append(path)
    return paths


def _candidate_on_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    *,
    require_black: bool = True,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for cell in path[1:]:
        if require_black and int(grid[cell]) != BLACK:
            continue
        if _touches_any_rg(grid, cell):
            continue
        candidates.append(cell)
    return candidates


def _place_unpaired(grid: np.ndarray, color: int, preferred: list[tuple[int, int]] | None = None) -> bool:
    candidates = []
    if preferred:
        candidates.extend(preferred)
    for rr in range(grid.shape[0]):
        for cc in range(grid.shape[1]):
            cell = (int(rr), int(cc))
            if int(grid[cell]) != BLACK:
                continue
            if _touches_any_rg(grid, cell):
                continue
            candidates.append(cell)
    seen: set[tuple[int, int]] = set()
    unique = []
    for cell in candidates:
        if cell not in seen:
            seen.add(cell)
            unique.append(cell)
    if not unique:
        return False
    cell = random.choice(unique)
    grid[cell] = int(color)
    return True


def _place_grey(grid: np.ndarray, preferred: list[tuple[int, int]] | None = None) -> bool:
    candidates = []
    if preferred:
        candidates.extend(preferred)
    for rr in range(grid.shape[0]):
        for cc in range(grid.shape[1]):
            cell = (int(rr), int(cc))
            if int(grid[cell]) != BLACK:
                continue
            if _touches_any_rg(grid, cell):
                continue
            candidates.append(cell)
    seen: set[tuple[int, int]] = set()
    unique = []
    for cell in candidates:
        if cell not in seen:
            seen.add(cell)
            unique.append(cell)
    if not unique:
        return False
    cell = random.choice(unique)
    grid[cell] = GREY
    return True


def _mode_requirements(call_index: int) -> dict:
    mode = call_index % 4
    if mode == 0:
        return {
            "required_vecs": [(0, 1), (1, 0)],
            "require_grey": True,
            "require_color": False,
            "require_order": False,
            "min_dominoes": 2,
        }
    if mode == 1:
        return {
            "required_vecs": [(0, -1), (-1, 0)],
            "require_grey": False,
            "require_color": True,
            "require_order": False,
            "min_dominoes": 2,
        }
    if mode == 2:
        return {
            "required_vecs": [(0, 1), (0, -1), (1, 0), (-1, 0)],
            "require_grey": True,
            "require_color": True,
            "require_order": False,
            "min_dominoes": 4,
        }
    return {
        "required_vecs": [(0, 1), (1, 0)],
        "require_grey": True,
        "require_color": True,
        "require_order": True,
        "min_dominoes": 3,
    }


def _place_order_sensitive_seed(grid: np.ndarray) -> list[dict] | None:
    h, w = grid.shape
    options = []
    for ar in range(1, h - 5):
        for ac in range(2, w - 4):
            a_green = (ar, ac)
            a_vec = (0, 1)
            a_red = (ar, ac - 1)
            if not _can_place_domino(grid, a_green, a_vec):
                continue
            max_k = min(h - ar - 2, w - ac - 3)
            for k in range(3, max_k + 1):
                b_red = (ar + k, ac + k)
                for b_vec in [(0, 1), (1, 0)]:
                    b_green = (b_red[0] + b_vec[0], b_red[1] + b_vec[1])
                    if not _can_place_domino(grid, b_green, b_vec):
                        continue
                    if b_green == a_green or b_green == a_red:
                        continue
                    options.append((a_green, a_vec, b_green, b_vec))
    if not options:
        return None
    a_green, a_vec, b_green, b_vec = random.choice(options)
    return [_place_domino(grid, a_green, a_vec), _place_domino(grid, b_green, b_vec)]


def _place_random_dominoes(grid: np.ndarray, count: int, required_vecs: list[tuple[int, int]]) -> list[dict] | None:
    placed: list[dict] = []
    needed = list(required_vecs)
    orientation_pool = needed + [random.choice(list(DIAG_FROM_VECTOR)) for _ in range(max(0, count - len(needed)))]
    random.shuffle(orientation_pool)

    for vec in orientation_pool:
        candidates = []
        for rr in range(grid.shape[0]):
            for cc in range(grid.shape[1]):
                green = (int(rr), int(cc))
                if _can_place_domino(grid, green, vec):
                    candidates.append(green)
        if not candidates:
            return None
        placed.append(_place_domino(grid, random.choice(candidates), vec))
    return placed


def _stop_kind_set(traces: list[dict]) -> set[str]:
    found: set[str] = set()
    for trace in traces:
        found.update(trace["ray_stops"])
    return found


def _build_attempt(call_index: int) -> np.ndarray | None:
    req = _mode_requirements(call_index)
    h = random.randint(14, 20)
    w = random.randint(14, 20)
    grid = np.zeros((h, w), dtype=int)

    domino_count = random.randint(max(2, req["min_dominoes"]), 5)
    dominoes: list[dict] = []
    required_vecs = list(req["required_vecs"])

    if req["require_order"]:
        seed = _place_order_sensitive_seed(grid)
        if seed is None:
            return None
        dominoes.extend(seed)
        remaining_required = required_vecs.copy()
        for pair in seed:
            if pair["vec"] in remaining_required:
                remaining_required.remove(pair["vec"])
        extra = domino_count - len(seed)
        more = _place_random_dominoes(grid, extra, remaining_required)
        if more is None:
            return None
        dominoes.extend(more)
    else:
        dominoes = _place_random_dominoes(grid, domino_count, required_vecs)
        if dominoes is None:
            return None

    pairs = _find_pairs(grid)
    if len(pairs) != domino_count:
        return None

    raw_paths = []
    for pair in pairs:
        for path in _raw_ray_cells(grid.shape, pair["green"], pair["vec"]):
            raw_paths.append({"pair": pair, "path": path})
    random.shuffle(raw_paths)

    if req["require_grey"]:
        placed = False
        for item in raw_paths:
            candidates = _candidate_on_path(grid, item["path"])
            if not candidates:
                continue
            grid[random.choice(candidates)] = GREY
            placed = True
            break
        if not placed:
            return None

    if req["require_color"] and not req["require_order"]:
        placed = False
        for item in raw_paths:
            candidates = _candidate_on_path(grid, item["path"])
            if not candidates:
                continue
            grid[random.choice(candidates)] = random.choice([RED, GREEN])
            placed = True
            break
        if not placed:
            return None

    extra_greys = random.randint(1, 4)
    for _ in range(extra_greys):
        preferred: list[tuple[int, int]] = []
        if raw_paths and random.random() < 0.6:
            chosen = random.choice(raw_paths)
            preferred = _candidate_on_path(grid, chosen["path"])
        if not _place_grey(grid, preferred):
            return None

    extra_reds = random.randint(1, 3)
    extra_greens = random.randint(1, 3)
    for color, amount in [(RED, extra_reds), (GREEN, extra_greens)]:
        for _ in range(amount):
            preferred = []
            if raw_paths and random.random() < 0.45:
                chosen = random.choice(raw_paths)
                preferred = _candidate_on_path(grid, chosen["path"])
            if not _place_unpaired(grid, color, preferred):
                return None

    pairs = _find_pairs(grid)
    if len(pairs) != domino_count:
        return None

    oriented = {pair["vec"] for pair in pairs}
    if not set(req["required_vecs"]).issubset(oriented):
        return None

    for rr, cc in zip(*np.where(grid == GREY), strict=False):
        for nr, nc in _neighbors4(int(rr), int(cc)):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) in (RED, GREEN):
                return None

    for rr, cc in zip(*np.where(grid == RED), strict=False):
        red = (int(rr), int(cc))
        if red not in {pair["red"] for pair in pairs} and _adjacent_of_color(grid, red, GREEN):
            return None
    for rr, cc in zip(*np.where(grid == GREEN), strict=False):
        green = (int(rr), int(cc))
        if green not in {pair["green"] for pair in pairs} and _adjacent_of_color(grid, green, RED):
            return None

    correct, traces = _simulate(grid)
    if np.array_equal(correct, grid):
        return None
    if sum(len(trace["visible_added"]) for trace in traces) < len(traces) + 2:
        return None
    if any(len(trace["visible_added"]) == 0 for trace in traces):
        return None

    stop_kinds = _stop_kind_set(traces)
    if req["require_grey"] and "grey" not in stop_kinds:
        return None
    if req["require_color"] and "color" not in stop_kinds:
        return None
    if "edge" not in stop_kinds:
        return None

    all_four_out, _ = _simulate(grid, all_four=True)
    if np.array_equal(correct, all_four_out):
        return None

    if req["require_grey"]:
        ignore_grey_out, _ = _simulate(grid, ignore_grey=True)
        if np.array_equal(correct, ignore_grey_out):
            return None

    if req["require_color"]:
        ignore_color_out, _ = _simulate(grid, ignore_color=True)
        if np.array_equal(correct, ignore_color_out):
            return None

    if req["require_order"]:
        erase_first_out, _ = _simulate(grid, erase_all_paired_reds_first=True)
        if np.array_equal(correct, erase_first_out):
            return None

    if len(np.unique(grid)) < 4:
        return None

    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt(_GEN_CALL_COUNT)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError("failed to generate a valid input for directional domino rays")
