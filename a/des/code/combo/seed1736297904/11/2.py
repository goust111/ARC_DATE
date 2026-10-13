from __future__ import annotations

import numpy as np


BG = 0
DIR_BY_NAME = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
ARROW_PATTERNS = {
    "U": frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}),
    "D": frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
    "L": frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}),
    "R": frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)}),
}
TIP_RELATIVE = {
    "U": (0, 1),
    "D": (2, 1),
    "L": (1, 0),
    "R": (1, 2),
}
CELLS_FROM_TIP = {
    "U": ((0, 0), (1, 0), (2, -1), (2, 0), (2, 1)),
    "D": ((0, 0), (-1, 0), (-2, -1), (-2, 0), (-2, 1)),
    "L": ((0, 0), (0, 1), (-1, 2), (0, 2), (1, 2)),
    "R": ((0, 0), (0, -1), (-1, -2), (0, -2), (1, -2)),
}

GEN_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
GEN_H_RANGE = (8, 18)
GEN_W_RANGE = (8, 18)
GEN_SLIDE_MAX = 6
GEN_POSITION_GAP = 4
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    H, W = grid.shape
    components: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _classify_arrow(coords: list[tuple[int, int]], color: int) -> dict[str, object]:
    assert len(coords) == 5, "arrow sprite must contain exactly 5 pixels"
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 - r0 == 2 and c1 - c0 == 2, "arrow sprite must occupy a 3x3 box"
    normalized = frozenset((r - r0, c - c0) for r, c in coords)
    for direction, pattern in ARROW_PATTERNS.items():
        if normalized == pattern:
            tr, tc = TIP_RELATIVE[direction]
            return {
                "color": int(color),
                "direction": direction,
                "tip": (int(r0 + tr), int(c0 + tc)),
                "coords": [(int(r), int(c)) for r, c in coords],
            }
    raise AssertionError("unrecognized arrow shape")


def _extract_scene(grid: np.ndarray) -> tuple[list[dict[str, object]], dict[int, list[tuple[int, int]]]]:
    colors = [int(color) for color in np.unique(grid) if int(color) != BG]
    arrows: list[dict[str, object]] = []
    targets_by_color: dict[int, list[tuple[int, int]]] = {}
    for color in colors:
        for comp in _find_components_of_color(grid, color):
            if len(comp) == 1:
                targets_by_color.setdefault(int(color), []).append(comp[0])
            else:
                arrows.append(_classify_arrow(comp, color))
    assert arrows, "expected at least one arrow"
    assert targets_by_color, "expected at least one target"
    return arrows, targets_by_color


def _forward_distance(
    tip: tuple[int, int], target: tuple[int, int], step: tuple[int, int]
) -> int | None:
    dr, dc = step
    if dr != 0:
        if tip[1] != target[1]:
            return None
        delta = int(target[0] - tip[0])
    else:
        if tip[0] != target[0]:
            return None
        delta = int(target[1] - tip[1])
    if delta * (dr + dc) <= 0:
        return None
    return abs(delta)


def _match_color_group(
    arrows: list[dict[str, object]], targets: list[tuple[int, int]]
) -> list[tuple[dict[str, object], tuple[int, int], int]]:
    assert len(arrows) == len(targets), "each arrow color must have one matching target"
    candidate_ids: dict[int, list[int]] = {}
    distances: dict[tuple[int, int], int] = {}
    for i, arrow in enumerate(arrows):
        step = DIR_BY_NAME[str(arrow["direction"])]
        tip = tuple(arrow["tip"])
        options: list[tuple[int, int]] = []
        for j, target in enumerate(targets):
            distance = _forward_distance(tip, target, step)
            if distance is not None:
                options.append((int(distance), int(j)))
                distances[(i, j)] = int(distance)
        options.sort()
        candidate_ids[i] = [j for _distance, j in options]
        assert candidate_ids[i], "every arrow must see a same-color target ahead of its tip"

    order = sorted(candidate_ids, key=lambda idx: len(candidate_ids[idx]))
    assignment: dict[int, int] = {}
    used_targets: set[int] = set()

    def search(k: int) -> bool:
        if k == len(order):
            return True
        arrow_idx = order[k]
        for target_idx in candidate_ids[arrow_idx]:
            if target_idx in used_targets:
                continue
            used_targets.add(target_idx)
            assignment[arrow_idx] = target_idx
            if search(k + 1):
                return True
            used_targets.remove(target_idx)
            del assignment[arrow_idx]
        return False

    assert search(0), "color group does not admit an unambiguous pairing"
    return [
        (arrows[i], targets[assignment[i]], distances[(i, assignment[i])])
        for i in range(len(arrows))
    ]


def _scene_matches(
    arrows: list[dict[str, object]], targets_by_color: dict[int, list[tuple[int, int]]]
) -> list[tuple[dict[str, object], tuple[int, int], int]]:
    grouped_arrows: dict[int, list[dict[str, object]]] = {}
    for arrow in arrows:
        grouped_arrows.setdefault(int(arrow["color"]), []).append(arrow)

    matches: list[tuple[dict[str, object], tuple[int, int], int]] = []
    assert set(grouped_arrows) == set(targets_by_color), "arrow colors and target colors must match"
    for color, color_arrows in grouped_arrows.items():
        matches.extend(_match_color_group(color_arrows, targets_by_color[color]))
    return matches


def _arrow_cells_from_tip(tip: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    base = CELLS_FROM_TIP[direction]
    return [(int(tip[0] + dr), int(tip[1] + dc)) for dr, dc in base]


def _collect_ray_claims(
    base: np.ndarray, targets_by_color: dict[int, list[tuple[int, int]]]
) -> dict[tuple[int, int], int]:
    claims: dict[tuple[int, int], int] = {}
    for color, targets in targets_by_color.items():
        for target in targets:
            for dr, dc in DIR_BY_NAME.values():
                rr = int(target[0] + dr)
                cc = int(target[1] + dc)
                while _in_bounds(base.shape, rr, cc) and int(base[rr, cc]) == BG:
                    prev = claims.get((rr, cc))
                    assert prev in (None, int(color)), "simultaneous rays must not conflict"
                    claims[(rr, cc)] = int(color)
                    rr += int(dr)
                    cc += int(dc)
    return claims


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    arrows, targets_by_color = _extract_scene(g)
    matches = _scene_matches(arrows, targets_by_color)

    out = g.copy()
    for arrow in arrows:
        for r, c in arrow["coords"]:
            out[r, c] = BG

    occupied = {
        (int(r), int(c))
        for targets in targets_by_color.values()
        for r, c in targets
    }
    moved_cells: set[tuple[int, int]] = set()
    for arrow, _target, distance in matches:
        dr, dc = DIR_BY_NAME[str(arrow["direction"])]
        shift = int(distance - 1)
        assert shift >= 0
        coords = []
        for r, c in arrow["coords"]:
            nr = int(r + dr * shift)
            nc = int(c + dc * shift)
            assert _in_bounds(g.shape, nr, nc)
            assert (nr, nc) not in occupied, "moved arrow must stop next to, not on, a target"
            assert (nr, nc) not in moved_cells, "moved arrows must not overlap"
            coords.append((nr, nc))
            moved_cells.add((nr, nc))
        for nr, nc in coords:
            out[nr, nc] = int(arrow["color"])

    ray_claims = _collect_ray_claims(out, targets_by_color)
    for (r, c), color in ray_claims.items():
        out[r, c] = int(color)
    return out


def _sample_spaced_positions(length: int, count: int, rng: np.random.Generator) -> list[int] | None:
    candidates = np.arange(1, length - 1, dtype=int)
    if len(candidates) < count:
        return None
    for _ in range(80):
        chosen = sorted(int(v) for v in rng.choice(candidates, size=count, replace=False))
        if count == 1 or min(np.diff(chosen)) >= GEN_POSITION_GAP:
            return chosen
    return None


def _build_pair(
    shape: tuple[int, int],
    direction: str,
    target: tuple[int, int],
    slide: int,
    color: int,
) -> dict[str, object] | None:
    dr, dc = DIR_BY_NAME[direction]
    final_tip = (int(target[0] - dr), int(target[1] - dc))
    initial_tip = (int(final_tip[0] - dr * slide), int(final_tip[1] - dc * slide))
    final_cells = _arrow_cells_from_tip(final_tip, direction)
    initial_cells = _arrow_cells_from_tip(initial_tip, direction)
    if not all(_in_bounds(shape, r, c) for r, c in final_cells + initial_cells + [target]):
        return None
    return {
        "color": int(color),
        "direction": direction,
        "target": (int(target[0]), int(target[1])),
        "initial_cells": initial_cells,
        "final_cells": final_cells,
    }


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(GEN_H_RANGE[0], GEN_H_RANGE[1] + 1))
    W = int(rng.integers(GEN_W_RANGE[0], GEN_W_RANGE[1] + 1))
    family = "vertical" if float(rng.random()) < 0.5 else "horizontal"
    pair_count = 2 if float(rng.random()) < 0.6 else 1

    if family == "vertical":
        directions = ["U", "D"] if pair_count == 2 else [("U" if float(rng.random()) < 0.5 else "D")]
        lines = _sample_spaced_positions(W, pair_count, rng)
        max_slide = H - 4
    else:
        directions = ["L", "R"] if pair_count == 2 else [("L" if float(rng.random()) < 0.5 else "R")]
        lines = _sample_spaced_positions(H, pair_count, rng)
        max_slide = W - 4
    if lines is None or max_slide < 1:
        return None

    colors = [int(v) for v in rng.choice(GEN_PALETTE, size=pair_count, replace=False)]
    rng.shuffle(lines)
    pairs: list[dict[str, object]] = []
    for color, direction, line in zip(colors, directions, lines):
        slide = int(rng.integers(1, min(GEN_SLIDE_MAX, max_slide) + 1))
        if direction == "U":
            target = (0, int(line))
        elif direction == "D":
            target = (H - 1, int(line))
        elif direction == "L":
            target = (int(line), 0)
        else:
            target = (int(line), W - 1)
        pair = _build_pair((H, W), direction, target, slide, color)
        if pair is None:
            return None
        pairs.append(pair)

    input_grid = np.zeros((H, W), dtype=int)
    final_grid = np.zeros((H, W), dtype=int)
    input_occupied: set[tuple[int, int]] = set()
    final_occupied: set[tuple[int, int]] = set()
    for pair in pairs:
        cells_in = set(pair["initial_cells"]) | {pair["target"]}
        cells_out = set(pair["final_cells"]) | {pair["target"]}
        if input_occupied & cells_in or final_occupied & cells_out:
            return None
        input_occupied |= cells_in
        final_occupied |= cells_out
        for r, c in pair["initial_cells"]:
            input_grid[r, c] = int(pair["color"])
        tr, tc = pair["target"]
        input_grid[tr, tc] = int(pair["color"])
        for r, c in pair["final_cells"]:
            final_grid[r, c] = int(pair["color"])
        final_grid[tr, tc] = int(pair["color"])

    targets_by_color = {int(pair["color"]): [pair["target"]] for pair in pairs}
    claims = _collect_ray_claims(final_grid, targets_by_color)
    if len(claims) < max(H, W) // 2:
        return None

    out = main(input_grid)
    if out.shape != input_grid.shape or np.array_equal(out, input_grid):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
