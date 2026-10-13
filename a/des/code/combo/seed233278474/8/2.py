from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

# --------------- solver ---------------


def _pair_endpoints(pair: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    if pair["orientation"] == "h":
        r = int(pair["fixed"])
        return (r, int(pair["a0"])), (r, int(pair["a1"]))
    c = int(pair["fixed"])
    return (int(pair["a0"]), c), (int(pair["a1"]), c)


def _core_cells(pair: dict) -> list[tuple[int, int]]:
    if pair["orientation"] == "h":
        r = int(pair["fixed"])
        return [(r, c) for c in range(int(pair["a0"]), int(pair["a1"]) + 1)]
    c = int(pair["fixed"])
    return [(r, c) for r in range(int(pair["a0"]), int(pair["a1"]) + 1)]


def _core_interior_cells(pair: dict) -> list[tuple[int, int]]:
    if int(pair["d"]) == 0:
        return []
    if pair["orientation"] == "h":
        r = int(pair["fixed"])
        return [(r, c) for c in range(int(pair["a0"]) + 1, int(pair["a1"]))]
    c = int(pair["fixed"])
    return [(r, c) for r in range(int(pair["a0"]) + 1, int(pair["a1"]))]


def _extension_specs(pair: dict) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    lo, hi = _pair_endpoints(pair)
    if pair["orientation"] == "h":
        return [(lo, (0, -1)), (hi, (0, 1))]
    return [(lo, (-1, 0)), (hi, (1, 0))]


def _pair_order_key(pair: dict) -> tuple[int, int, int, int, int]:
    lo, hi = _pair_endpoints(pair)
    return (int(pair["d"]), int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1]))


def _extract_pairs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    assert colors, "expected at least one non-background color"

    pairs: list[dict] = []
    for idx, color in enumerate(colors):
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each foreground color must appear exactly twice"
        (r0, c0), (r1, c1) = [tuple(map(int, rc)) for rc in coords]
        same_row = int(r0) == int(r1)
        same_col = int(c0) == int(c1)
        assert same_row ^ same_col, "paired pixels must lie on one row or one column"

        if same_row:
            a0, a1 = sorted((int(c0), int(c1)))
            fixed = int(r0)
            orientation = "h"
        else:
            a0, a1 = sorted((int(r0), int(r1)))
            fixed = int(c0)
            orientation = "v"

        pairs.append(
            {
                "id": idx,
                "color": int(color),
                "orientation": orientation,
                "fixed": int(fixed),
                "a0": int(a0),
                "a1": int(a1),
                "d": int(a1 - a0 - 1),
            }
        )
    return pairs


def _apply_pairs_to_grid(
    grid: np.ndarray, pairs: list[dict], *, validate_core: bool
) -> tuple[np.ndarray, dict[int, dict], bool]:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    details: dict[int, dict] = {}

    for pair in sorted(pairs, key=_pair_order_key):
        color = int(pair["color"])
        endpoints = set(_pair_endpoints(pair))

        for r, c in endpoints:
            if int(out[r, c]) != color:
                return out, details, False

        for r, c in _core_interior_cells(pair):
            if validate_core and int(out[r, c]) != BG:
                return out, details, False

        for r, c in _core_cells(pair):
            if validate_core and (r, c) not in endpoints and int(out[r, c]) not in (BG, color):
                return out, details, False
            out[r, c] = color

        side_infos = []
        for (r, c), (dr, dc) in _extension_specs(pair):
            filled = 0
            stop = "complete"
            cur_r, cur_c = int(r), int(c)
            for _step in range(int(pair["d"])):
                cur_r += dr
                cur_c += dc
                if not (0 <= cur_r < h and 0 <= cur_c < w):
                    stop = "boundary"
                    break
                if int(out[cur_r, cur_c]) != BG:
                    stop = "collision"
                    break
                out[cur_r, cur_c] = color
                filled += 1
            side_infos.append({"filled": int(filled), "stop": stop})
        details[int(pair["id"])] = {"sides": side_infos}

    return out, details, True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each same-color axis-aligned pixel pair into a symmetric stripe.

    Rule:
    - Let D be the number of cells strictly between the two same-color pixels.
    - Fill the whole segment between them with that color.
    - From each endpoint, extend outward away from the other endpoint by D cells.
    - Extensions stop before the first occupied cell or at the grid edge.
    - Smaller D pairs are processed first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    pairs = _extract_pairs(g)
    out, _details, valid = _apply_pairs_to_grid(g, pairs, validate_core=True)
    assert valid, "input violates the pair-stripe assumptions"
    return out


# --------------- generator ---------------

GRID_H_RANGE = (10, 18)
GRID_W_RANGE = (10, 18)
PAIR_COUNT_CHOICES = (2, 3, 4)
PAIR_COUNT_WEIGHTS = (2, 3, 2)
MAX_D_CAP = 4
GEN_RETRIES = 400

FORCE_COLLISION_PROB = 0.6
FORCE_BOUNDARY_PROB = 0.6
FORCE_SHARED_AXIS_PROB = 0.4
FORCE_BOTH_ORIENTATIONS_PROB = 0.7


def _build_input_grid(shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    g = np.zeros(shape, dtype=int)
    for pair in pairs:
        for r, c in _pair_endpoints(pair):
            if int(g[r, c]) != BG:
                raise ValueError("endpoint overlap")
            g[r, c] = int(pair["color"])
    return g


def _pair_mode(detail: dict) -> str:
    stops = [str(side["stop"]) for side in detail["sides"]]
    if "collision" in stops:
        return "collision"
    if "boundary" in stops:
        return "boundary"
    return "open"


def _reuses_axis(pair: dict, placed_pairs: list[dict]) -> bool:
    if pair["orientation"] == "h":
        return any(
            other["orientation"] == "h" and int(other["fixed"]) == int(pair["fixed"])
            for other in placed_pairs
        )
    return any(
        other["orientation"] == "v" and int(other["fixed"]) == int(pair["fixed"])
        for other in placed_pairs
    )


def _enumerate_candidates(
    shape: tuple[int, int], color: int, d: int, placed_pairs: list[dict]
) -> list[dict]:
    h, w = shape
    base = _build_input_grid(shape, placed_pairs)
    candidates: list[dict] = []

    for orientation in ("h", "v"):
        if orientation == "h":
            for fixed in range(h):
                for a0 in range(w - d - 1):
                    a1 = a0 + d + 1
                    if np.any(base[fixed, a0 : a1 + 1] != BG):
                        continue
                    pair = {
                        "id": len(placed_pairs),
                        "color": int(color),
                        "orientation": "h",
                        "fixed": int(fixed),
                        "a0": int(a0),
                        "a1": int(a1),
                        "d": int(d),
                    }
                    temp = base.copy()
                    temp[fixed, a0] = color
                    temp[fixed, a1] = color
                    _out, details, valid = _apply_pairs_to_grid(temp, placed_pairs + [pair], validate_core=True)
                    if not valid:
                        continue
                    candidates.append({"pair": pair, "mode": _pair_mode(details[pair["id"]])})
        else:
            for fixed in range(w):
                for a0 in range(h - d - 1):
                    a1 = a0 + d + 1
                    if np.any(base[a0 : a1 + 1, fixed] != BG):
                        continue
                    pair = {
                        "id": len(placed_pairs),
                        "color": int(color),
                        "orientation": "v",
                        "fixed": int(fixed),
                        "a0": int(a0),
                        "a1": int(a1),
                        "d": int(d),
                    }
                    temp = base.copy()
                    temp[a0, fixed] = color
                    temp[a1, fixed] = color
                    _out, details, valid = _apply_pairs_to_grid(temp, placed_pairs + [pair], validate_core=True)
                    if not valid:
                        continue
                    candidates.append({"pair": pair, "mode": _pair_mode(details[pair["id"]])})
    return candidates


def _choose_candidate(
    shape: tuple[int, int],
    color: int,
    d: int,
    placed_pairs: list[dict],
    *,
    orientation_target: str | None,
    mode_target: str | None,
    strict_mode: bool,
    reuse_axis: bool,
) -> dict | None:
    candidates = _enumerate_candidates(shape, color, d, placed_pairs)
    if not candidates:
        return None

    if orientation_target is not None:
        oriented = [cand for cand in candidates if cand["pair"]["orientation"] == orientation_target]
        if oriented:
            candidates = oriented

    if reuse_axis:
        reused = [cand for cand in candidates if _reuses_axis(cand["pair"], placed_pairs)]
        if reused:
            candidates = reused

    if mode_target is not None:
        matched = [cand for cand in candidates if cand["mode"] == mode_target]
        if matched:
            candidates = matched
        elif strict_mode:
            return None

    return random.choice(candidates)["pair"]


def _shared_axis_present(pairs: list[dict]) -> bool:
    rows = [int(pair["fixed"]) for pair in pairs if pair["orientation"] == "h"]
    cols = [int(pair["fixed"]) for pair in pairs if pair["orientation"] == "v"]
    return len(rows) != len(set(rows)) or len(cols) != len(set(cols))


def _actual_feature_flags(details: dict[int, dict]) -> tuple[bool, bool]:
    has_collision = False
    has_boundary = False
    for detail in details.values():
        for side in detail["sides"]:
            stop = str(side["stop"])
            if stop == "collision":
                has_collision = True
            elif stop == "boundary":
                has_boundary = True
    return has_collision, has_boundary


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_H_RANGE[0], GRID_H_RANGE[1])
    w = random.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])
    shape = (h, w)

    max_d = min(MAX_D_CAP, max(h, w) // 3)
    if max_d < 2:
        return None

    max_pairs = min(len(PAIR_COUNT_CHOICES), max_d)
    allowed_pair_counts = [n for n in PAIR_COUNT_CHOICES if n <= max_d]
    allowed_weights = [PAIR_COUNT_WEIGHTS[PAIR_COUNT_CHOICES.index(n)] for n in allowed_pair_counts]
    n_pairs = random.choices(allowed_pair_counts, weights=allowed_weights, k=1)[0]

    want_collision = n_pairs >= 2 and random.random() < FORCE_COLLISION_PROB
    want_boundary = random.random() < FORCE_BOUNDARY_PROB
    want_shared_axis = n_pairs >= 3 and random.random() < FORCE_SHARED_AXIS_PROB
    want_both_orientations = n_pairs >= 2 and random.random() < FORCE_BOTH_ORIENTATIONS_PROB

    if want_collision and want_boundary and n_pairs == 2 and max_d >= 3:
        n_pairs = 3

    colors = random.sample(PALETTE, n_pairs)
    d_values = sorted(random.sample(range(1, max_d + 1), n_pairs))

    orientation_targets = [random.choice(("h", "v")) for _ in range(n_pairs)]
    if want_both_orientations and len(set(orientation_targets)) == 1:
        orientation_targets[-1] = "v" if orientation_targets[0] == "h" else "h"

    mode_targets: list[str | None] = [None] * n_pairs
    mode_targets[0] = random.choice(("open", "boundary"))
    if want_boundary:
        mode_targets[-1] = "boundary"
    if want_collision:
        collision_idx = n_pairs - 2 if want_boundary and n_pairs >= 3 else n_pairs - 1
        mode_targets[collision_idx] = "collision"
    for i in range(1, n_pairs):
        if mode_targets[i] is None:
            mode_targets[i] = random.choices(
                ("open", "boundary", "collision"), weights=(4, 2, 2), k=1
            )[0]

    reuse_idx = random.randrange(1, n_pairs) if want_shared_axis else -1

    placed: list[dict] = []
    for i, (color, d) in enumerate(zip(colors, d_values)):
        pair = _choose_candidate(
            shape,
            int(color),
            int(d),
            placed,
            orientation_target=orientation_targets[i],
            mode_target=mode_targets[i],
            strict_mode=(i == n_pairs - 1 and mode_targets[i] is not None),
            reuse_axis=(i == reuse_idx),
        )
        if pair is None:
            return None
        placed.append(pair)

    grid = _build_input_grid(shape, placed)
    out, details, valid = _apply_pairs_to_grid(grid, _extract_pairs(grid), validate_core=True)
    if not valid:
        return None

    if not np.any(out != grid):
        return None

    has_collision, has_boundary = _actual_feature_flags(details)
    if want_collision and not has_collision:
        return None
    if want_boundary and not has_boundary:
        return None
    if want_shared_axis and not _shared_axis_present(placed):
        return None
    if want_both_orientations and len({pair["orientation"] for pair in placed}) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
