from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
MAROON = 9

SPRITE_COLORS = (BLUE, RED, YELLOW, PINK, ORANGE)
GEN_RETRIES = 500


def _clear_neighborhood_mask(grid: np.ndarray, bg: int = BLACK) -> np.ndarray:
    h, w = map(int, grid.shape)
    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            r0 = max(0, r - 1)
            r1 = min(h, r + 2)
            c0 = max(0, c - 1)
            c1 = min(w, c + 2)
            mask[r, c] = np.all(grid[r0:r1, c0:c1] == int(bg))
    return mask


def _maximal_true_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int, int, int, int]]:
    h, w = map(int, mask.shape)
    rects: set[tuple[int, int, int, int, int, int, int]] = set()
    for r0 in range(h):
        valid_cols = np.ones(w, dtype=bool)
        for r1 in range(r0, h):
            valid_cols &= mask[r1]
            c = 0
            while c < w:
                if not valid_cols[c]:
                    c += 1
                    continue
                c0 = c
                while c < w and valid_cols[c]:
                    c += 1
                c1 = c - 1

                can_extend_up = r0 > 0 and np.all(mask[r0 - 1, c0 : c1 + 1])
                can_extend_down = r1 + 1 < h and np.all(mask[r1 + 1, c0 : c1 + 1])
                can_extend_left = c0 > 0 and np.all(mask[r0 : r1 + 1, c0 - 1])
                can_extend_right = c1 + 1 < w and np.all(mask[r0 : r1 + 1, c1 + 1])
                if can_extend_up or can_extend_down or can_extend_left or can_extend_right:
                    continue

                hh = r1 - r0 + 1
                ww = c1 - c0 + 1
                rects.add((hh * ww, hh, ww, r0, c0, r1, c1))

    return sorted(
        rects,
        key=lambda rect: (-rect[0], -max(rect[1], rect[2]), rect[3], rect[4], rect[5], rect[6]),
    )


def _find_true_runs(arr: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    n = int(arr.shape[0])
    i = 0
    while i < n:
        if not arr[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and arr[j + 1]:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def _window_has_no_grey(grid: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> bool:
    h, w = map(int, grid.shape)
    rr0 = max(0, int(r0))
    rr1 = min(h - 1, int(r1))
    cc0 = max(0, int(c0))
    cc1 = min(w - 1, int(c1))
    return bool(np.all(grid[rr0 : rr1 + 1, cc0 : cc1 + 1] != GREY))


def _settled_positions(
    orientation: str,
    sprite_a: tuple[int, int],
    sprite_b: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    (r1, c1), (r2, c2) = sprite_a, sprite_b
    if orientation == "horizontal":
        if c1 > c2:
            r1, c1, r2, c2 = r2, c2, r1, c1
        assert r1 == r2
        assert c2 - c1 >= 5 and (c2 - c1) % 2 == 1
        steps = (c2 - c1 - 1) // 2
        return (r1, c1 + steps), (r2, c2 - steps)
    if r1 > r2:
        r1, c1, r2, c2 = r2, c2, r1, c1
    assert c1 == c2
    assert r2 - r1 >= 5 and (r2 - r1) % 2 == 1
    steps = (r2 - r1 - 1) // 2
    return (r1 + steps, c1), (r2 - steps, c2)


def _detect_corridors(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    clear = _clear_neighborhood_mask(g, bg=BLACK)
    rects = _maximal_true_rectangles(clear)
    min_major = min(h, w) // 2

    row_spans: set[tuple[int, int]] = set()
    col_spans: set[tuple[int, int]] = set()
    for _area, rh, rw, r0, c0, r1, c1 in rects:
        if rh <= 2 and rw >= 3:
            row_spans.add((int(r0), int(r1)))
        if rw <= 2 and rh >= 3:
            col_spans.add((int(c0), int(c1)))

    corridors: list[dict] = []
    seen: set[tuple[int, int, int, int]] = set()

    for r0, r1 in sorted(row_spans):
        valid = np.zeros(w, dtype=bool)
        for c in range(w):
            valid[c] = _window_has_no_grey(g, r0 - 1, r1 + 1, c - 1, c + 1)
        for c0, c1 in _find_true_runs(valid):
            if c1 - c0 + 1 < min_major:
                continue
            sprite_cells = [
                (r, c)
                for r in range(r0, r1 + 1)
                for c in range(c0, c1 + 1)
                if int(g[r, c]) in SPRITE_COLORS
            ]
            if len(sprite_cells) != 2:
                continue
            rows = {r for r, _c in sprite_cells}
            if len(rows) != 1:
                continue
            sprite_cells = sorted(sprite_cells, key=lambda rc: rc[1])
            if sprite_cells[1][1] - sprite_cells[0][1] < 5:
                continue
            if (sprite_cells[1][1] - sprite_cells[0][1]) % 2 == 0:
                continue
            bbox = (r0, c0, r1, c1)
            if bbox in seen:
                continue
            final_a, final_b = _settled_positions("horizontal", sprite_cells[0], sprite_cells[1])
            corridors.append(
                {
                    "bbox": bbox,
                    "orientation": "horizontal",
                    "sprites": tuple(sprite_cells),
                    "final_positions": (final_a, final_b),
                }
            )
            seen.add(bbox)

    for c0, c1 in sorted(col_spans):
        valid = np.zeros(h, dtype=bool)
        for r in range(h):
            valid[r] = _window_has_no_grey(g, r - 1, r + 1, c0 - 1, c1 + 1)
        for r0, r1 in _find_true_runs(valid):
            if r1 - r0 + 1 < min_major:
                continue
            sprite_cells = [
                (r, c)
                for r in range(r0, r1 + 1)
                for c in range(c0, c1 + 1)
                if int(g[r, c]) in SPRITE_COLORS
            ]
            if len(sprite_cells) != 2:
                continue
            cols = {c for _r, c in sprite_cells}
            if len(cols) != 1:
                continue
            sprite_cells = sorted(sprite_cells, key=lambda rc: rc[0])
            if sprite_cells[1][0] - sprite_cells[0][0] < 5:
                continue
            if (sprite_cells[1][0] - sprite_cells[0][0]) % 2 == 0:
                continue
            bbox = (r0, c0, r1, c1)
            if bbox in seen:
                continue
            final_a, final_b = _settled_positions("vertical", sprite_cells[0], sprite_cells[1])
            corridors.append(
                {
                    "bbox": bbox,
                    "orientation": "vertical",
                    "sprites": tuple(sprite_cells),
                    "final_positions": (final_a, final_b),
                }
            )
            seen.add(bbox)

    corridors.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["orientation"]))
    return corridors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal hidden corridor tracks, slide the two sprites in each track toward each other, and repaint collision pairs green.

    Input structure:
    - The grid uses black (0) as background and grey (5) as clutter.
    - Hidden corridors are long horizontal or vertical black rectangular tracks embedded in the clutter.
    - Each corridor contains exactly two single-pixel sprites of distinct non-grey colors, placed on one lane near opposite ends.

    Rule:
    - Keep only black pixels whose full in-grid 3x3 neighborhood is black, then use the long thin maximal rectangles in that clear mask to recover each corridor track.
    - For every corridor, move its two sprites simultaneously by one pixel per step along the corridor axis toward each other.
    - Stop when the two sprites become 4-neighbor adjacent; repaint both final sprite pixels green (3).
    - Repaint every other corridor pixel maroon (9), while preserving all grey clutter and all non-corridor black background.

    Key insight:
    - The 3x3-clear negative space isolates the real corridor lanes from random black gaps, and once the corridor axis is known the sprite motion is a pure synchronous midpoint collision on that lane.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    corridors = _detect_corridors(g)
    assert corridors, "expected at least one corridor"

    out = g.copy()
    for corridor in corridors:
        r0, c0, r1, c1 = corridor["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = MAROON
        for r, c in corridor["final_positions"]:
            out[r, c] = GREEN
    return out


def _expand_bbox(bbox: tuple[int, int, int, int], amount: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 - amount, c0 - amount, r1 + amount, c1 + amount


def _bbox_in_bounds(bbox: tuple[int, int, int, int], h: int, w: int) -> bool:
    r0, c0, r1, c1 = bbox
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _mark_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _sample_distinct_colors(rng: np.random.Generator) -> tuple[int, int]:
    cols = rng.choice(np.array(SPRITE_COLORS, dtype=int), size=2, replace=False)
    return int(cols[0]), int(cols[1])


def _place_barriers(grid: np.ndarray, corridor: dict) -> None:
    h, w = map(int, grid.shape)
    r0, c0, r1, c1 = corridor["bbox"]
    if corridor["orientation"] == "horizontal":
        rr0 = r0 - 1
        rr1 = r1 + 1
        for cc in (c0 - 2, c1 + 2):
            if 0 <= cc < w:
                grid[rr0 : rr1 + 1, cc] = GREY
    else:
        cc0 = c0 - 1
        cc1 = c1 + 1
        for rr in (r0 - 2, r1 + 2):
            if 0 <= rr < h:
                grid[rr, cc0 : cc1 + 1] = GREY


def _make_horizontal_plan(
    h: int,
    w: int,
    occupied: np.ndarray,
    rng: np.random.Generator,
) -> dict | None:
    min_major = min(h, w) // 2
    length_hi = w - 6
    if length_hi < min_major + 2:
        return None
    for _ in range(200):
        height = int(rng.integers(1, 3))
        length = int(rng.integers(min_major + 2, length_hi + 1))
        r0 = int(rng.integers(2, h - height - 2))
        c0 = int(rng.integers(2, w - length - 2))
        bbox = (r0, c0, r0 + height - 1, c0 + length - 1)
        protected = _expand_bbox(bbox, 2)
        if not _bbox_in_bounds(protected, h, w):
            continue
        pr0, pc0, pr1, pc1 = protected
        if np.any(occupied[pr0 : pr1 + 1, pc0 : pc1 + 1]):
            continue

        lane_r = r0 + int(rng.integers(0, height))
        max_offset = min(3, length - 4)
        if max_offset < 1:
            continue
        for _inner in range(60):
            left = c0 + int(rng.integers(1, max_offset + 1))
            right = c0 + length - 1 - int(rng.integers(1, max_offset + 1))
            if right - left >= 5 and (right - left) % 2 == 1:
                color_a, color_b = _sample_distinct_colors(rng)
                sprites = ((lane_r, left, color_a), (lane_r, right, color_b))
                plan = {"bbox": bbox, "orientation": "horizontal", "sprites": sprites}
                _mark_bbox(occupied, protected)
                return plan
    return None


def _make_vertical_plan(
    h: int,
    w: int,
    occupied: np.ndarray,
    rng: np.random.Generator,
) -> dict | None:
    min_major = min(h, w) // 2
    length_hi = h - 6
    if length_hi < min_major + 2:
        return None
    for _ in range(200):
        width = int(rng.integers(1, 3))
        length = int(rng.integers(min_major + 2, length_hi + 1))
        r0 = int(rng.integers(2, h - length - 2))
        c0 = int(rng.integers(2, w - width - 2))
        bbox = (r0, c0, r0 + length - 1, c0 + width - 1)
        protected = _expand_bbox(bbox, 2)
        if not _bbox_in_bounds(protected, h, w):
            continue
        pr0, pc0, pr1, pc1 = protected
        if np.any(occupied[pr0 : pr1 + 1, pc0 : pc1 + 1]):
            continue

        lane_c = c0 + int(rng.integers(0, width))
        max_offset = min(3, length - 4)
        if max_offset < 1:
            continue
        for _inner in range(60):
            top = r0 + int(rng.integers(1, max_offset + 1))
            bottom = r0 + length - 1 - int(rng.integers(1, max_offset + 1))
            if bottom - top >= 5 and (bottom - top) % 2 == 1:
                color_a, color_b = _sample_distinct_colors(rng)
                sprites = ((top, lane_c, color_a), (bottom, lane_c, color_b))
                plan = {"bbox": bbox, "orientation": "vertical", "sprites": sprites}
                _mark_bbox(occupied, protected)
                return plan
    return None


def _make_short_distractor(
    h: int,
    w: int,
    occupied: np.ndarray,
    rng: np.random.Generator,
) -> dict | None:
    orientation = str(rng.choice(np.array(["horizontal", "vertical"], dtype=object)))
    for _ in range(120):
        if orientation == "horizontal":
            height = int(rng.integers(1, 3))
            length = int(rng.integers(3, 7))
            if w - length - 2 <= 2:
                continue
            r0 = int(rng.integers(2, h - height - 2))
            c0 = int(rng.integers(2, w - length - 2))
            bbox = (r0, c0, r0 + height - 1, c0 + length - 1)
        else:
            width = int(rng.integers(1, 3))
            length = int(rng.integers(3, 7))
            if h - length - 2 <= 2:
                continue
            r0 = int(rng.integers(2, h - length - 2))
            c0 = int(rng.integers(2, w - width - 2))
            bbox = (r0, c0, r0 + length - 1, c0 + width - 1)

        protected = _expand_bbox(bbox, 2)
        if not _bbox_in_bounds(protected, h, w):
            continue
        pr0, pc0, pr1, pc1 = protected
        if np.any(occupied[pr0 : pr1 + 1, pc0 : pc1 + 1]):
            continue
        _mark_bbox(occupied, protected)
        return {"bbox": bbox, "orientation": orientation}
    return None


def _build_expected_output(inp: np.ndarray, plans: list[dict]) -> np.ndarray:
    out = inp.copy()
    for plan in plans:
        r0, c0, r1, c1 = plan["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = MAROON
        sprite_a = (int(plan["sprites"][0][0]), int(plan["sprites"][0][1]))
        sprite_b = (int(plan["sprites"][1][0]), int(plan["sprites"][1][1]))
        final_a, final_b = _settled_positions(plan["orientation"], sprite_a, sprite_b)
        out[final_a] = GREEN
        out[final_b] = GREEN
    return out


def _validate_plan(inp: np.ndarray, plans: list[dict], distractors: list[dict]) -> bool:
    expected = _build_expected_output(inp, plans)
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    detected = _detect_corridors(inp)
    if len(detected) != len(plans):
        return False
    expected_bboxes = {tuple(plan["bbox"]) for plan in plans}
    actual_bboxes = {tuple(item["bbox"]) for item in detected}
    if expected_bboxes != actual_bboxes:
        return False

    for plan in plans:
        bbox = tuple(plan["bbox"])
        matched = [item for item in detected if tuple(item["bbox"]) == bbox]
        if len(matched) != 1:
            return False
        det = matched[0]
        if det["orientation"] != plan["orientation"]:
            return False
        expected_sprites = {
            (int(plan["sprites"][0][0]), int(plan["sprites"][0][1])),
            (int(plan["sprites"][1][0]), int(plan["sprites"][1][1])),
        }
        if set(det["sprites"]) != expected_sprites:
            return False

    min_major = min(inp.shape) // 2
    for item in distractors:
        r0, c0, r1, c1 = item["bbox"]
        major = max(r1 - r0 + 1, c1 - c0 + 1)
        if major >= min_major and np.any(inp[r0 : r1 + 1, c0 : c1 + 1] == GREY):
            return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(17, 23))
    w = int(rng.integers(17, 23))
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    n_corridors = int(rng.integers(2, 4))
    if n_corridors == 2:
        patterns = (("horizontal", "vertical"), ("horizontal", "horizontal"), ("vertical", "vertical"))
        weights = np.array([0.6, 0.2, 0.2], dtype=float)
        orientations = list(patterns[int(rng.choice(np.arange(len(patterns)), p=weights))])
    else:
        third = str(rng.choice(np.array(["horizontal", "vertical"], dtype=object)))
        orientations = ["horizontal", "vertical", third]
        rng.shuffle(orientations)

    plans: list[dict] = []
    for orientation in orientations:
        if orientation == "horizontal":
            plan = _make_horizontal_plan(h, w, occupied, rng)
        else:
            plan = _make_vertical_plan(h, w, occupied, rng)
        if plan is None:
            return None
        plans.append(plan)

    distractors: list[dict] = []
    for _ in range(int(rng.integers(0, 3))):
        item = _make_short_distractor(h, w, occupied, rng)
        if item is not None:
            distractors.append(item)

    keep_black = np.zeros((h, w), dtype=bool)
    for item in [*plans, *distractors]:
        track = _expand_bbox(item["bbox"], 1)
        if not _bbox_in_bounds(track, h, w):
            return None
        _mark_bbox(keep_black, track)

    clutter_prob = float(rng.uniform(0.38, 0.48))
    clutter_mask = (~keep_black) & (rng.random((h, w)) < clutter_prob)
    grid[clutter_mask] = GREY

    for plan in plans:
        _place_barriers(grid, plan)

    for item in distractors:
        _place_barriers(grid, item)

    for plan in plans:
        for r, c, color in plan["sprites"]:
            grid[int(r), int(c)] = int(color)

    if np.count_nonzero(grid == GREY) < (h * w) // 5:
        return None

    if not _validate_plan(grid, plans, distractors):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid corridor-collision input")
