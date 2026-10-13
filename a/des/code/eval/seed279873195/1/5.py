import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
MIN_RECTS = 3
MAX_RECTS = 5
SCENE_H_RANGE = (9, 16)
SCENE_W_RANGE = (10, 18)
CORE_SIZE_RANGE = (1, 2)
RECT_MIN_H = 4
RECT_MIN_W = 4
GEN_RETRIES = 500


def _marker_column(grid: np.ndarray) -> int:
    h, w = grid.shape
    for c in range(w - 1, -1, -1):
        rows = np.flatnonzero(grid[:, c] != BG)
        if rows.size < 2:
            continue
        if c > 0 and not np.all(grid[:, c - 1] == BG):
            continue
        colors = [int(grid[r, c]) for r in rows]
        if len(colors) != len(set(colors)):
            continue
        return int(c)
    raise AssertionError("expected a right-side marker column")


def _scene_and_hint(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    marker_c = _marker_column(grid)
    assert marker_c >= 1, "marker column must be separated from the scene"
    scene = np.asarray(grid[:, : marker_c - 1], dtype=int)
    rows = np.flatnonzero(grid[:, marker_c] != BG)
    hint_order = [int(grid[r, marker_c]) for r in rows]
    assert hint_order, "marker column must contain at least one color"
    return scene, hint_order


def _visible_bboxes(scene: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    for color in sorted(set(map(int, scene.flatten())) - {BG}):
        pts = np.argwhere(scene == color)
        assert pts.size > 0, "every foreground color must stay visible"
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        bboxes[int(color)] = (int(r0), int(c0), int(r1), int(c1))
    return bboxes


def _candidate_orders(hint_order: list[int]) -> list[list[int]]:
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    def add(order: list[int]) -> None:
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            out.append(order)

    add(list(hint_order))
    for i in range(len(hint_order) - 1):
        cand = list(hint_order)
        cand[i], cand[i + 1] = cand[i + 1], cand[i]
        add(cand)
    return out


def _order_violation_score(
    scene: np.ndarray, bboxes: dict[int, tuple[int, int, int, int]], order: list[int]
) -> int:
    pos = {color: i for i, color in enumerate(order)}
    if len(pos) != len(order):
        return 10**9

    violations = 0
    for color in order:
        if color not in bboxes:
            violations += 10**6
            continue
        r0, c0, r1, c1 = bboxes[color]
        window = scene[r0 : r1 + 1, c0 : c1 + 1]
        allowed = set(order[: pos[color] + 1])
        for value in map(int, window.flatten()):
            if value == BG or value not in allowed:
                violations += 1
    return violations


def _infer_actual_order(scene: np.ndarray, hint_order: list[int]) -> tuple[list[int], dict[int, tuple[int, int, int, int]]]:
    bboxes = _visible_bboxes(scene)
    scene_colors = set(bboxes)
    assert scene_colors == set(hint_order), "marker colors must match visible rectangle colors"

    scored = []
    for cand in _candidate_orders(hint_order):
        score = _order_violation_score(scene, bboxes, cand)
        scored.append((score, cand))
    scored.sort(key=lambda item: item[0])

    best_score = scored[0][0]
    best = [cand for score, cand in scored if score == best_score]
    assert best_score == 0, "one candidate order should fit the occlusion evidence exactly"
    assert len(best) == 1, "expected a unique corrected layer order"
    return best[0], bboxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the complete bottom rectangle after correcting a nearly-correct layer hint.

    Input structure:
    - The left area is a black-background scene made of several distinct solid rectangles.
    - Rectangles partially overlap, so higher layers hide parts of lower layers.
    - The rightmost marker column lists the rectangle colors from top to bottom, except
      that exactly one adjacent pair is swapped.

    Rule:
    - Use the visible occlusion evidence to correct the marker order.
    - Identify the bottom-most rectangle in the corrected order.
    - Output that rectangle alone as a fully filled crop at its full size.

    Key insight:
    - For the true order, every visible bounding box contains only its own color or colors
      above it. The swapped hint order breaks that consistency for at least one rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    scene, hint_order = _scene_and_hint(g)
    actual_order, bboxes = _infer_actual_order(scene, hint_order)

    bottom_color = int(actual_order[-1])
    r0, c0, r1, c1 = bboxes[bottom_color]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    return np.full((h, w), bottom_color, dtype=int)


def _render_scene(
    scene_h: int,
    scene_w: int,
    actual_order: list[int],
    rects: dict[int, tuple[int, int, int, int]],
) -> np.ndarray:
    scene = np.full((scene_h, scene_w), BG, dtype=int)
    for color in reversed(actual_order):
        r0, c0, r1, c1 = rects[color]
        scene[r0 : r1 + 1, c0 : c1 + 1] = color
    return scene


def _sample_rectangles(
    rng: random.Random, scene_h: int, scene_w: int, colors: list[int]
) -> dict[int, tuple[int, int, int, int]] | None:
    core_h = rng.randint(*CORE_SIZE_RANGE)
    core_w = rng.randint(*CORE_SIZE_RANGE)
    core_r0 = rng.randint(2, scene_h - core_h - 3)
    core_c0 = rng.randint(2, scene_w - core_w - 3)
    core_r1 = core_r0 + core_h - 1
    core_c1 = core_c0 + core_w - 1

    rects: dict[int, tuple[int, int, int, int]] = {}
    used: set[tuple[int, int, int, int]] = set()
    for color in colors:
        placed = False
        for _ in range(80):
            r0 = rng.randint(0, core_r0)
            r1 = rng.randint(core_r1, scene_h - 1)
            c0 = rng.randint(0, core_c0)
            c1 = rng.randint(core_c1, scene_w - 1)
            if r1 - r0 + 1 < RECT_MIN_H or c1 - c0 + 1 < RECT_MIN_W:
                continue
            box = (int(r0), int(c0), int(r1), int(c1))
            if box in used:
                continue
            rects[int(color)] = box
            used.add(box)
            placed = True
            break
        if not placed:
            return None
    return rects


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    n_rects = rng.randint(MIN_RECTS, MAX_RECTS)
    scene_h = rng.randint(max(SCENE_H_RANGE[0], n_rects + 3), SCENE_H_RANGE[1])
    scene_w = rng.randint(SCENE_W_RANGE[0], SCENE_W_RANGE[1])
    actual_order = rng.sample(COLOR_POOL, n_rects)

    rects = _sample_rectangles(rng, scene_h, scene_w, actual_order)
    if rects is None:
        return None

    scene = _render_scene(scene_h, scene_w, actual_order, rects)
    visible = _visible_bboxes(scene)
    if set(visible) != set(actual_order):
        return None
    if any(visible[color] != rects[color] for color in actual_order):
        return None

    for i in range(n_rects - 1):
        above = actual_order[i]
        below = actual_order[i + 1]
        r0, c0, r1, c1 = rects[below]
        if not np.any(scene[r0 : r1 + 1, c0 : c1 + 1] == above):
            return None

    swap_idx = rng.randrange(n_rects - 1)
    hint_order = list(actual_order)
    hint_order[swap_idx], hint_order[swap_idx + 1] = hint_order[swap_idx + 1], hint_order[swap_idx]

    full = np.full((scene_h, scene_w + 2), BG, dtype=int)
    full[:, :scene_w] = scene
    marker_rows = sorted(rng.sample(range(scene_h), n_rects))
    for row, color in zip(marker_rows, hint_order):
        full[row, scene_w + 1] = color

    solved = main(full)
    bottom = actual_order[-1]
    r0, c0, r1, c1 = rects[bottom]
    expected = np.full((r1 - r0 + 1, c1 - c0 + 1), bottom, dtype=int)
    if not np.array_equal(solved, expected):
        return None

    return full, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        case = _build_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid occlusion-order sample")
