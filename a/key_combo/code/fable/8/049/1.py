import random
from typing import Dict, List, Optional, Tuple

import numpy as np


BLACK = 0
RED = 2
TEAL = 8
TRUE_COLORS = (1, 3, 4, 5, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 400


def _proper_divisors(n: int) -> List[int]:
    return [d for d in range(1, n) if n % d == 0]


def _has_axis_period(tile: np.ndarray, pr: int, pc: int) -> bool:
    h, w = tile.shape
    row_idx = np.arange(h) % int(pr)
    col_idx = np.arange(w) % int(pc)
    return bool(np.array_equal(tile, tile[row_idx][:, col_idx]))


def _fundamental_period(tile: np.ndarray) -> Tuple[int, int]:
    h, w = tile.shape

    pr = h
    for cand in _proper_divisors(h):
        if _has_axis_period(tile, cand, w):
            pr = cand
            break

    pc = w
    for cand in _proper_divisors(w):
        if _has_axis_period(tile, pr, cand):
            pc = cand
            break

    return pr, pc


def _candidate_info(grid: np.ndarray, pr: int, pc: int) -> Optional[Dict[str, object]]:
    h, w = grid.shape
    pattern = np.zeros((pr, pc), dtype=int)
    errors = 0

    for rr in range(pr):
        for cc in range(pc):
            counts = [0] * 10
            for r in range(rr, h, pr):
                for c in range(cc, w, pc):
                    color = int(grid[r, c])
                    if color != BLACK:
                        counts[color] += 1

            total = sum(counts[1:])
            if total == 0:
                return None

            best = max(counts[1:])
            winners = [color for color in range(1, 10) if counts[color] == best]
            if len(winners) != 1:
                return None

            color = winners[0]
            pattern[rr, cc] = color
            errors += total - best

    return {
        "pr": int(pr),
        "pc": int(pc),
        "area": int(pr * pc),
        "errors": int(errors),
        "pattern": pattern,
    }


def _best_period_info(grid: np.ndarray) -> Dict[str, object]:
    h, w = grid.shape
    candidates: List[Dict[str, object]] = []
    for pr in _proper_divisors(h):
        for pc in _proper_divisors(w):
            info = _candidate_info(grid, pr, pc)
            if info is not None:
                candidates.append(info)

    if not candidates:
        raise ValueError("no valid nontrivial translational period found")

    candidates.sort(key=lambda info: (int(info["errors"]), int(info["area"])))
    return candidates[0]


def _reconstruct_true_grid(grid: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], Dict[str, object]]:
    info = _best_period_info(grid)
    pr = int(info["pr"])
    pc = int(info["pc"])
    pattern = np.asarray(info["pattern"], dtype=int)
    h, w = grid.shape
    row_idx = np.arange(h) % pr
    col_idx = np.arange(w) % pc
    truth = pattern[row_idx][:, col_idx]
    return truth, (pr, pc), info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden periodic tiling, then emit a damage-class map instead of the repaired image.

    Input structure:
    - The grid is a rectangular translational tiling whose true colors are all non-black.
    - Some cells are damaged: missing cells are turned black, while other cells are replaced by a wrong non-black color.
    - The undamaged visible copies uniquely determine the underlying row/column period and each orbit's true color.

    Rule:
    - Find the nontrivial rectangular period that best explains the visible cells.
    - For each period orbit, recover the true color from the unique non-black majority among its visible copies.
    - Output a same-size map: clean cells become black, blacked-out damage becomes teal, and wrong non-black damage becomes red.

    Key insight:
    - The reference image never appears explicitly; each cell must be judged by comparing it with the majority color of its translational orbit.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    truth, _period, _info = _reconstruct_true_grid(g)
    out = np.zeros_like(g)
    out[(g == BLACK) & (truth != BLACK)] = TEAL
    out[(g != BLACK) & (g != truth)] = RED
    return out


def _make_tile(pr: int, pc: int) -> np.ndarray:
    palette_size = random.randint(3, min(5, len(TRUE_COLORS)))
    palette = random.sample(list(TRUE_COLORS), k=palette_size)
    tile = np.array([[random.choice(palette) for _ in range(pc)] for _ in range(pr)], dtype=int)

    for color in palette:
        if color not in tile:
            r = random.randrange(pr)
            c = random.randrange(pc)
            tile[r, c] = color

    return tile


def _orbit_coords(pr: int, pc: int, vr: int, vc: int, rr: int, cc: int) -> List[Tuple[int, int]]:
    return [(rr + k * pr, cc + j * pc) for k in range(vr) for j in range(vc)]


def _pick_wrong_color(true_color: int, tile_palette: List[int]) -> int:
    if len(tile_palette) >= 2 and random.random() < 0.7:
        choices = [color for color in tile_palette if color != true_color]
    else:
        choices = [color for color in TRUE_COLORS if color != true_color]
    return random.choice(choices)


def _damage_grid(truth: np.ndarray, tile: np.ndarray, pr: int, pc: int, vr: int, vc: int) -> np.ndarray:
    grid = truth.copy()
    states = ["clean", "black", "red", "both"]
    weights = [6, 3, 3, 2]
    orbit_states: Dict[Tuple[int, int], str] = {}

    for rr in range(pr):
        for cc in range(pc):
            orbit_states[(rr, cc)] = random.choices(states, weights=weights, k=1)[0]

    if not any(state in {"black", "both"} for state in orbit_states.values()):
        orbit_states[random.choice(list(orbit_states.keys()))] = "black"
    if not any(state in {"red", "both"} for state in orbit_states.values()):
        orbit_states[random.choice(list(orbit_states.keys()))] = "red"
    if all(state != "clean" for state in orbit_states.values()):
        orbit_states[random.choice(list(orbit_states.keys()))] = "clean"

    tile_palette = sorted(set(int(color) for color in tile.flatten()))
    total_damaged = 0

    for (rr, cc), state in orbit_states.items():
        coords = _orbit_coords(pr, pc, vr, vc, rr, cc)
        random.shuffle(coords)
        true_color = int(tile[rr, cc])

        if state == "black":
            r, c = coords[0]
            grid[r, c] = BLACK
            total_damaged += 1
        elif state == "red":
            r, c = coords[0]
            grid[r, c] = _pick_wrong_color(true_color, tile_palette)
            total_damaged += 1
        elif state == "both":
            (r0, c0), (r1, c1) = coords[:2]
            grid[r0, c0] = BLACK
            grid[r1, c1] = _pick_wrong_color(true_color, tile_palette)
            total_damaged += 2

    min_damage = max(2, truth.size // 14)
    max_damage = max(min_damage + 1, truth.size // 4)
    if total_damaged < min_damage or total_damaged > max_damage:
        raise ValueError("damage density out of range")

    return grid


def _expected_damage_map(inp: np.ndarray, truth: np.ndarray) -> np.ndarray:
    out = np.zeros_like(inp)
    out[(inp == BLACK) & (truth != BLACK)] = TEAL
    out[(inp != BLACK) & (inp != truth)] = RED
    return out


def _validate_case(inp: np.ndarray, truth: np.ndarray, actual_period: Tuple[int, int]) -> bool:
    h, w = inp.shape
    pr, pc = actual_period
    infos: List[Dict[str, object]] = []
    for cand_pr in _proper_divisors(h):
        for cand_pc in _proper_divisors(w):
            info = _candidate_info(inp, cand_pr, cand_pc)
            if info is not None:
                infos.append(info)

    if not infos:
        return False

    infos.sort(key=lambda info: (int(info["errors"]), int(info["area"])))
    best = infos[0]
    if (int(best["pr"]), int(best["pc"])) != (pr, pc):
        return False
    if len(infos) > 1:
        second = infos[1]
        if (
            int(second["errors"]) == int(best["errors"])
            and int(second["area"]) == int(best["area"])
        ):
            return False

    recon, detected_period, _info = _reconstruct_true_grid(inp)
    if detected_period != actual_period:
        return False
    if not np.array_equal(recon, truth):
        return False

    expected = _expected_damage_map(inp, truth)
    actual = main(inp)
    if not np.array_equal(expected, actual):
        return False

    if not np.any(expected == TEAL):
        return False
    if not np.any(expected == RED):
        return False
    if not np.any(expected == BLACK):
        return False

    return True


def _generate_case() -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    pr = random.randint(2, 5)
    pc = random.randint(2, 5)
    vr = random.randint(2, min(5, 30 // pr))
    vc = random.randint(2, min(5, 30 // pc))

    tile = _make_tile(pr, pc)
    if _fundamental_period(tile) != (pr, pc):
        raise ValueError("tile not fundamental")

    truth = np.tile(tile, (vr, vc))
    inp = _damage_grid(truth, tile, pr, pc, vr, vc)
    return inp, truth, (pr, pc)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            inp, truth, period = _generate_case()
        except ValueError:
            continue
        if _validate_case(inp, truth, period):
            return inp
    raise RuntimeError("failed to generate a valid periodic damage map")
