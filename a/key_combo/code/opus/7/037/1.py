from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
TEAL = 8

# Generator bounds tuned to keep the grids compact while leaving enough room
# for 2-3 carrier rectangles and visible ray lengths.
GRID_H_MIN = 14
GRID_H_MAX = 18
GRID_W_PAD_MAX = 5
EDGE_MARGIN = 2
RECT_H_OPTIONS = (4, 5)
RECT_W_OPTIONS = (6, 7, 8)
RECT_COUNT_OPTIONS = (2, 3)
PLACEMENT_ATTEMPTS = 120
GENERATION_ATTEMPTS = 400
PAIRED_LAYOUT_PROB = 0.6
BLOCKED_VARIANT_PROB = 0.65


def _carrier_from_neighbors(grid: np.ndarray, r: int, c: int) -> int:
    colors = set()
    h, w = grid.shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != RED:
            colors.add(int(grid[nr, nc]))
    assert len(colors) == 1, "each red marker must sit inside exactly one carrier"
    carrier = next(iter(colors))
    assert carrier in (BLUE, TEAL), "carrier color must be blue or teal"
    return int(carrier)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each embedded red marker horizontally using its carrier color for direction and its row for ray length.

    Input structure:
    - The grid has a black background plus 2-3 solid blue or teal rectangles acting as carrier regions.
    - Each carrier contains 1-2 isolated red marker pixels strictly inside the rectangle, never on the carrier border.
    - Blue carriers encode rightward motion and teal carriers encode leftward motion.

    Rule:
    - For each red marker, identify its carrier color from the surrounding rectangle.
    - Let L be the marker's distance to the nearer top or bottom grid edge: L = min(r, H-1-r).
    - Starting from the horizontally adjacent cell in the carrier-indicated direction, paint up to L red cells, but stop before any cell that is neither black nor the marker's carrier color.
    - Restore the original marker cell back to its carrier color after drawing the ray.

    Key insight:
    - The carrier color determines the horizontal direction, while the marker's vertical position determines the horizontal ray length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    h, w = g.shape
    markers = [tuple(map(int, coord)) for coord in np.argwhere(g == RED)]
    assert markers, "expected at least one red marker"

    carriers = {(r, c): _carrier_from_neighbors(g, r, c) for r, c in markers}

    base = g.copy()
    for (r, c), carrier in carriers.items():
        base[r, c] = int(carrier)

    out = base.copy()
    for r, c in markers:
        carrier = int(carriers[(r, c)])
        direction = 1 if carrier == BLUE else -1
        ray_limit = min(int(r), int(h - 1 - r))
        for step in range(1, ray_limit + 1):
            nc = int(c + direction * step)
            if not (0 <= nc < w):
                break
            if int(out[r, nc]) not in (BACKGROUND, carrier):
                break
            out[r, nc] = RED

    return out


def _make_rect(r0: int, c0: int, h: int, w: int, color: int) -> dict[str, int]:
    return {
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r0 + h - 1),
        "c1": int(c0 + w - 1),
        "h": int(h),
        "w": int(w),
        "color": int(color),
    }


def _rects_overlap(a: dict[str, int], b: dict[str, int], pad: int = 1) -> bool:
    return not (
        int(a["r1"]) + pad < int(b["r0"])
        or int(b["r1"]) + pad < int(a["r0"])
        or int(a["c1"]) + pad < int(b["c0"])
        or int(b["c1"]) + pad < int(a["c0"])
    )


def _choose_dims(rng: random.Random) -> tuple[int, int]:
    h = int(rng.randint(GRID_H_MIN, GRID_H_MAX))
    w = int(rng.randint(max(15, h - 1), h + GRID_W_PAD_MAX))
    return h, w


def _place_general_layout(rng: random.Random) -> tuple[int, int, list[dict[str, int]]] | None:
    h, w = _choose_dims(rng)
    n_rects = int(rng.choice(RECT_COUNT_OPTIONS))

    colors = [BLUE, TEAL]
    if n_rects == 3:
        colors.append(int(rng.choice((BLUE, TEAL))))
    rng.shuffle(colors)

    rects: list[dict[str, int]] = []
    for color in colors:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            rh = int(rng.choice(RECT_H_OPTIONS))
            rw = int(rng.choice(RECT_W_OPTIONS))
            if h - rh - EDGE_MARGIN < EDGE_MARGIN or w - rw - EDGE_MARGIN < EDGE_MARGIN:
                continue
            r0 = int(rng.randint(EDGE_MARGIN, h - rh - EDGE_MARGIN))
            c0 = int(rng.randint(EDGE_MARGIN, w - rw - EDGE_MARGIN))
            rect = _make_rect(r0, c0, rh, rw, int(color))
            if any(_rects_overlap(rect, other, pad=1) for other in rects):
                continue
            rects.append(rect)
            placed = True
            break
        if not placed:
            return None

    return h, w, rects


def _place_paired_layout(rng: random.Random) -> tuple[int, int, list[dict[str, int]]] | None:
    h, w = _choose_dims(rng)
    n_rects = int(rng.choice(RECT_COUNT_OPTIONS))

    blue_h = int(rng.choice(RECT_H_OPTIONS))
    blue_w = int(rng.choice(RECT_W_OPTIONS))
    teal_h = int(rng.choice(RECT_H_OPTIONS))
    teal_w = int(rng.choice(RECT_W_OPTIONS))
    gap = int(rng.randint(1, 3))
    total_w = blue_w + gap + teal_w
    if total_w > w - 2 * EDGE_MARGIN:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        blue_r0 = int(rng.randint(EDGE_MARGIN, h - blue_h - EDGE_MARGIN))
        teal_r0 = int(rng.randint(EDGE_MARGIN, h - teal_h - EDGE_MARGIN))

        blue_rows = set(range(blue_r0 + 1, blue_r0 + blue_h - 1))
        teal_rows = set(range(teal_r0 + 1, teal_r0 + teal_h - 1))
        if len(blue_rows & teal_rows) < 2:
            continue

        slack = int(w - 2 * EDGE_MARGIN - total_w)
        blue_c0 = int(EDGE_MARGIN + rng.randint(0, slack))
        teal_c0 = int(blue_c0 + blue_w + gap)

        rects = [
            _make_rect(blue_r0, blue_c0, blue_h, blue_w, BLUE),
            _make_rect(teal_r0, teal_c0, teal_h, teal_w, TEAL),
        ]

        if n_rects == 3:
            extra_color = int(rng.choice((BLUE, TEAL)))
            placed_extra = False
            for __ in range(PLACEMENT_ATTEMPTS):
                extra_h = int(rng.choice(RECT_H_OPTIONS))
                extra_w = int(rng.choice(RECT_W_OPTIONS))
                if h - extra_h - EDGE_MARGIN < EDGE_MARGIN or w - extra_w - EDGE_MARGIN < EDGE_MARGIN:
                    continue
                extra_r0 = int(rng.randint(EDGE_MARGIN, h - extra_h - EDGE_MARGIN))
                extra_c0 = int(rng.randint(EDGE_MARGIN, w - extra_w - EDGE_MARGIN))
                extra = _make_rect(extra_r0, extra_c0, extra_h, extra_w, extra_color)
                if any(_rects_overlap(extra, other, pad=1) for other in rects):
                    continue
                rects.append(extra)
                placed_extra = True
                break
            if not placed_extra:
                continue

        rng.shuffle(rects)
        return h, w, rects

    return None


def _place_rectangles(rng: random.Random) -> tuple[int, int, list[dict[str, int]]] | None:
    if rng.random() < PAIRED_LAYOUT_PROB:
        paired = _place_paired_layout(rng)
        if paired is not None:
            return paired
    return _place_general_layout(rng)


def _build_base_grid(h: int, w: int, rects: list[dict[str, int]]) -> tuple[np.ndarray, np.ndarray]:
    base = np.zeros((h, w), dtype=int)
    owner = np.full((h, w), -1, dtype=int)
    for idx, rect in enumerate(rects):
        base[int(rect["r0"]) : int(rect["r1"]) + 1, int(rect["c0"]) : int(rect["c1"]) + 1] = int(rect["color"])
        owner[int(rect["r0"]) : int(rect["r1"]) + 1, int(rect["c0"]) : int(rect["c1"]) + 1] = int(idx)
    return base, owner


def _trace_candidate(
    h: int,
    w: int,
    base: np.ndarray,
    owner: np.ndarray,
    rects: list[dict[str, int]],
    rect_idx: int,
    r: int,
    c: int,
) -> dict[str, object] | None:
    rect = rects[int(rect_idx)]
    carrier = int(rect["color"])
    direction = 1 if carrier == BLUE else -1
    ray_limit = min(int(r), int(h - 1 - r))

    if ray_limit < 2:
        return None
    if direction == 1 and c + ray_limit >= w:
        return None
    if direction == -1 and c - ray_limit < 0:
        return None

    path: list[tuple[int, int]] = []
    blocked = False

    for step in range(1, ray_limit + 1):
        nc = int(c + direction * step)
        cell_owner = int(owner[r, nc])
        cell_color = int(base[r, nc])

        if cell_owner != -1 and cell_owner != rect_idx:
            if cell_color == carrier:
                return None
            blocked = True
            break
        if cell_color not in (BACKGROUND, carrier):
            blocked = True
            break
        path.append((int(r), int(nc)))

    if not path:
        return None

    return {
        "rect": int(rect_idx),
        "r": int(r),
        "c": int(c),
        "L": int(ray_limit),
        "blocked": bool(blocked),
        "path": tuple(path),
    }


def _enumerate_candidates(
    h: int,
    w: int,
    base: np.ndarray,
    owner: np.ndarray,
    rects: list[dict[str, int]],
    rng: random.Random,
) -> list[list[dict[str, object]]]:
    rect_candidates: list[list[dict[str, object]]] = []
    for idx, rect in enumerate(rects):
        candidates: list[dict[str, object]] = []
        for r in range(int(rect["r0"]) + 1, int(rect["r1"])):
            for c in range(int(rect["c0"]) + 1, int(rect["c1"])):
                candidate = _trace_candidate(h, w, base, owner, rects, idx, r, c)
                if candidate is not None:
                    candidates.append(candidate)
        rng.shuffle(candidates)
        rect_candidates.append(candidates)
    return rect_candidates


def _choose_marker_counts(n_rects: int, rng: random.Random) -> list[int]:
    counts = [1 + int(rng.random() < 0.45) for _ in range(n_rects)]
    if sum(counts) == n_rects:
        counts[int(rng.randrange(n_rects))] = 2
    return counts


def _select_markers(
    rects: list[dict[str, int]],
    rect_candidates: list[list[dict[str, object]]],
    counts: list[int],
    rng: random.Random,
) -> list[dict[str, object]] | None:
    total_markers = int(sum(counts))
    require_blocked = bool(
        rng.random() < BLOCKED_VARIANT_PROB
        and any(bool(candidate["blocked"]) for candidates in rect_candidates for candidate in candidates)
    )

    rect_order = sorted(range(len(rects)), key=lambda idx: (len(rect_candidates[idx]), counts[idx]))
    selected: list[dict[str, object]] = []
    used_path: set[tuple[int, int]] = set()
    marker_positions: set[tuple[int, int]] = set()
    solutions: list[list[dict[str, object]]] = []

    def compatible(candidate: dict[str, object]) -> bool:
        r = int(candidate["r"])
        c = int(candidate["c"])
        pos = (r, c)
        if pos in used_path:
            return False
        for pr, pc in marker_positions:
            if max(abs(pr - r), abs(pc - c)) <= 1:
                return False

        path = set(candidate["path"])
        if path & used_path:
            return False
        if any(p in marker_positions for p in path):
            return False
        for picked in selected:
            if pos in picked["path"]:
                return False
        return True

    def final_ok() -> bool:
        if total_markers > 1 and len({int(candidate["L"]) for candidate in selected}) < 2:
            return False
        if not any(not bool(candidate["blocked"]) for candidate in selected):
            return False
        if require_blocked and not any(bool(candidate["blocked"]) for candidate in selected):
            return False
        return True

    def backtrack(rect_pos: int) -> None:
        if len(solutions) >= 24:
            return
        if rect_pos == len(rect_order):
            if final_ok():
                solutions.append([dict(candidate) for candidate in selected])
            return

        rect_idx = int(rect_order[rect_pos])
        candidates = list(rect_candidates[rect_idx])
        rng.shuffle(candidates)
        chosen_here: list[dict[str, object]] = []

        def pick_from(start: int) -> None:
            if len(chosen_here) == counts[rect_idx]:
                backtrack(rect_pos + 1)
                return

            need = counts[rect_idx] - len(chosen_here)
            for cand_idx in range(start, len(candidates)):
                if len(candidates) - cand_idx < need:
                    break
                candidate = candidates[cand_idx]
                if not compatible(candidate):
                    continue

                chosen_here.append(candidate)
                selected.append(candidate)
                pos = (int(candidate["r"]), int(candidate["c"]))
                marker_positions.add(pos)
                for cell in candidate["path"]:
                    used_path.add(cell)

                pick_from(cand_idx + 1)

                for cell in candidate["path"]:
                    used_path.remove(cell)
                marker_positions.remove(pos)
                selected.pop()
                chosen_here.pop()

        pick_from(0)

    backtrack(0)
    if not solutions:
        return None
    return list(rng.choice(solutions))


def _attempt_input(rng: random.Random) -> np.ndarray | None:
    placed = _place_rectangles(rng)
    if placed is None:
        return None

    h, w, rects = placed
    base, owner = _build_base_grid(h, w, rects)
    rect_candidates = _enumerate_candidates(h, w, base, owner, rects, rng)

    counts = _choose_marker_counts(len(rects), rng)
    if any(len(rect_candidates[idx]) < counts[idx] for idx in range(len(rects))):
        return None

    selected = _select_markers(rects, rect_candidates, counts, rng)
    if selected is None:
        return None

    grid = base.copy()
    for candidate in selected:
        grid[int(candidate["r"]), int(candidate["c"])] = RED

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_input(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
