import random
from collections import Counter

import numpy as np


DIVIDER = 5
LEFT_PANEL_COLOR = 1
RIGHT_PANEL_COLOR = 3
CONTROL_COLOR = 2
RECT_COLOR = 7

SIDES = ("up", "down", "left", "right")
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MIN_HEIGHT = 16
MAX_HEIGHT = 22
MIN_WIDTH = 23
MAX_WIDTH = 29
MIN_PANEL_WIDTH = 11
MAX_GENERATION_ATTEMPTS = 400
MIN_MARGIN = 4
RECT_SIDE_RANGE = (3, 4)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _find_separator(grid: np.ndarray) -> int:
    cols = np.flatnonzero(np.all(grid == DIVIDER, axis=0))
    assert len(cols) == 1, "expected one full-height divider column"
    return int(cols[0])


def _dominant_color(values: np.ndarray) -> int:
    return int(Counter(map(int, values.ravel())).most_common(1)[0][0])


def _panel_backgrounds(grid: np.ndarray, split_c: int) -> tuple[int, int]:
    left_bg = _dominant_color(grid[:, :split_c])
    right_bg = _dominant_color(grid[:, split_c + 1 :])
    assert left_bg != DIVIDER and right_bg != DIVIDER
    assert left_bg != right_bg
    return int(left_bg), int(right_bg)


def _find_rectangles(grid: np.ndarray, split_c: int) -> list[dict]:
    rects: list[dict] = []
    for coords in _components_of_color(grid, RECT_COLOR):
        r0, c0, r1, c1 = _bbox(coords)
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == RECT_COLOR), "orange objects must be solid rectangles"
        panel = "left" if c1 < split_c else "right"
        assert c0 > split_c or c1 < split_c, "rectangle may not cross divider"
        rects.append({"bbox": (r0, c0, r1, c1), "panel": panel})
    rects.sort(key=lambda item: (item["panel"], item["bbox"][0], item["bbox"][1]))
    assert rects, "expected at least one rectangle"
    return rects


def _control_matches(cell: tuple[int, int], rects: list[dict], split_c: int) -> list[tuple[int, str]]:
    r, c = cell
    panel = "left" if c < split_c else "right" if c > split_c else None
    assert panel is not None, "control pixel may not sit on the divider"
    matches: list[tuple[int, str]] = []
    for idx, rect in enumerate(rects):
        if rect["panel"] != panel:
            continue
        r0, c0, r1, c1 = rect["bbox"]
        if c0 <= c <= c1:
            if r < r0:
                matches.append((idx, "up"))
            if r > r1:
                matches.append((idx, "down"))
        if r0 <= r <= r1:
            if c < c0:
                matches.append((idx, "left"))
            if c > c1:
                matches.append((idx, "right"))
    return matches


def _count_controls(grid: np.ndarray, rects: list[dict], split_c: int) -> tuple[list[dict], list[tuple[int, int]]]:
    counts = [{side: 0 for side in SIDES} for _ in rects]
    controls = [tuple(map(int, rc)) for rc in np.argwhere(grid == CONTROL_COLOR)]
    for cell in controls:
        matches = _control_matches(cell, rects, split_c)
        assert len(matches) == 1, "each control pixel must align with exactly one rectangle side"
        rect_idx, side = matches[0]
        counts[rect_idx][side] += 1
    return counts, controls


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each orange rectangle by the count of its side controls, using the opposite panel's background color.

    Input structure:
    - A full-height vertical grey divider splits the grid into a left panel and a right panel.
    - The left panel has one background color, the right panel has another, and each panel contains one or two solid orange rectangles.
    - Red singleton control pixels lie only in the side strips aligned with a rectangle edge, never in diagonal quadrants.

    Rule:
    - Find every solid orange rectangle and determine which panel contains it.
    - Count how many red control pixels align above, below, left, and right of that rectangle within its panel.
    - Extend the rectangle outward by those counts on the four sides, painting only the new extension stripes with the opposite panel's background color, then erase all red control pixels back to their panel background.

    Key insight:
    - The control pixels encode lengths, but the fill color comes from the other panel, so panel membership and side counts must both be inferred to get the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    split_c = _find_separator(g)
    left_bg, right_bg = _panel_backgrounds(g, split_c)
    rects = _find_rectangles(g, split_c)
    counts, controls = _count_controls(g, rects, split_c)

    out = g.copy()
    h, w = g.shape

    for rect, side_counts in zip(rects, counts):
        r0, c0, r1, c1 = rect["bbox"]
        target = right_bg if rect["panel"] == "left" else left_bg

        n_up = side_counts["up"]
        n_down = side_counts["down"]
        n_left = side_counts["left"]
        n_right = side_counts["right"]

        if n_up:
            rr0 = max(0, r0 - n_up)
            out[rr0:r0, c0 : c1 + 1] = target
        if n_down:
            rr1 = min(h, r1 + 1 + n_down)
            out[r1 + 1 : rr1, c0 : c1 + 1] = target
        if n_left:
            cc0 = max(0, c0 - n_left)
            out[r0 : r1 + 1, cc0:c0] = target
        if n_right:
            cc1 = min(w, c1 + 1 + n_right)
            out[r0 : r1 + 1, c1 + 1 : cc1] = target

    for r, c in controls:
        out[r, c] = left_bg if c < split_c else right_bg

    return out


def _rand_rect_size() -> tuple[int, int]:
    return random.randint(*RECT_SIDE_RANGE), random.randint(*RECT_SIDE_RANGE)


def _panel_span(split_c: int, panel: str, width: int) -> tuple[int, int]:
    if panel == "left":
        return 0, split_c - 1
    return split_c + 1, width - 1


def _choose_panel_counts(height: int) -> tuple[int, int]:
    counts = [1, 1]
    if height >= 17 and random.random() < 0.6:
        counts[random.randrange(2)] = 2
    if height >= 20 and counts.count(2) == 0 and random.random() < 0.25:
        counts[random.randrange(2)] = 2
    return counts[0], counts[1]


def _random_col(panel_c0: int, panel_c1: int, rect_w: int) -> int | None:
    lo = panel_c0 + MIN_MARGIN
    hi = panel_c1 - MIN_MARGIN - rect_w + 1
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _build_panel_rects(height: int, width: int, split_c: int, panel: str, count: int) -> list[dict] | None:
    panel_c0, panel_c1 = _panel_span(split_c, panel, width)
    for _ in range(120):
        if count == 1:
            rect_h, rect_w = _rand_rect_size()
            if height - rect_h - 2 * MIN_MARGIN < 0:
                continue
            r0 = random.randint(MIN_MARGIN, height - MIN_MARGIN - rect_h)
            c0 = _random_col(panel_c0, panel_c1, rect_w)
            if c0 is None:
                continue
            return [
                {
                    "panel": panel,
                    "bbox": (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1),
                    "counts": {side: 0 for side in SIDES},
                    "controls": [],
                }
            ]

        top_h, top_w = _rand_rect_size()
        bot_h, bot_w = _rand_rect_size()
        max_top = height - MIN_MARGIN - top_h - 2 - bot_h - MIN_MARGIN
        if max_top < MIN_MARGIN:
            continue
        top_r0 = random.randint(MIN_MARGIN, max_top)
        gap = random.randint(2, min(5, height - top_r0 - top_h - bot_h - MIN_MARGIN))
        bot_r0 = top_r0 + top_h + gap
        if bot_r0 + bot_h - 1 > height - MIN_MARGIN - 1:
            continue

        top_c0 = _random_col(panel_c0, panel_c1, top_w)
        bot_c0 = _random_col(panel_c0, panel_c1, bot_w)
        if top_c0 is None or bot_c0 is None:
            continue

        return [
            {
                "panel": panel,
                "bbox": (top_r0, top_c0, top_r0 + top_h - 1, top_c0 + top_w - 1),
                "counts": {side: 0 for side in SIDES},
                "controls": [],
                "blocked": {"down"},
            },
            {
                "panel": panel,
                "bbox": (bot_r0, bot_c0, bot_r0 + bot_h - 1, bot_c0 + bot_w - 1),
                "counts": {side: 0 for side in SIDES},
                "controls": [],
                "blocked": {"up"},
            },
        ]
    return None


def _assign_side_counts(rects: list[dict]) -> None:
    for rect in rects:
        blocked = set(rect.get("blocked", set()))
        allowed = [side for side in SIDES if side not in blocked]
        counts = {side: 0 for side in SIDES}
        for side in allowed:
            counts[side] = random.randint(0, 3)
        if sum(counts.values()) == 0:
            chosen = random.choice(allowed)
            counts[chosen] = random.randint(1, 3)
        rect["counts"] = counts

    vertical_sides = [(rect, side) for rect in rects for side in ("up", "down") if side not in rect.get("blocked", set())]
    horizontal_sides = [(rect, side) for rect in rects for side in ("left", "right") if side not in rect.get("blocked", set())]
    if not any(rect["counts"]["up"] or rect["counts"]["down"] for rect in rects):
        rect, side = random.choice(vertical_sides)
        rect["counts"][side] = random.randint(1, 3)
    if not any(rect["counts"]["left"] or rect["counts"]["right"] for rect in rects):
        rect, side = random.choice(horizontal_sides)
        rect["counts"][side] = random.randint(1, 3)


def _cell_distance_from_side(cell: tuple[int, int], rect: dict, side: str) -> int:
    r, c = cell
    r0, c0, r1, c1 = rect["bbox"]
    if side == "up":
        return r0 - r
    if side == "down":
        return r - r1
    if side == "left":
        return c0 - c
    return c - c1


def _side_candidates(shape: tuple[int, int], split_c: int, rects: list[dict], rect_idx: int, side: str) -> list[tuple[int, int]]:
    h, w = shape
    rect = rects[rect_idx]
    r0, c0, r1, c1 = rect["bbox"]
    panel_c0, panel_c1 = _panel_span(split_c, rect["panel"], w)

    cells: list[tuple[int, int]] = []
    if side == "up":
        for r in range(0, r0):
            for c in range(c0, c1 + 1):
                cells.append((r, c))
    elif side == "down":
        for r in range(r1 + 1, h):
            for c in range(c0, c1 + 1):
                cells.append((r, c))
    elif side == "left":
        for r in range(r0, r1 + 1):
            for c in range(panel_c0, c0):
                cells.append((r, c))
    else:
        for r in range(r0, r1 + 1):
            for c in range(c1 + 1, panel_c1 + 1):
                cells.append((r, c))

    unique = []
    for cell in cells:
        if _control_matches(cell, rects, split_c) == [(rect_idx, side)]:
            unique.append(cell)
    return unique


def _place_controls(grid: np.ndarray, split_c: int, rects: list[dict]) -> bool:
    occupied = {
        tuple(map(int, rc))
        for rc in np.argwhere((grid == RECT_COLOR) | (grid == DIVIDER) | (grid == CONTROL_COLOR))
    }
    for rect_idx, rect in enumerate(rects):
        rect["controls"] = []
        for side, need in rect["counts"].items():
            if need == 0:
                continue
            candidates = _side_candidates(grid.shape, split_c, rects, rect_idx, side)
            candidates = [cell for cell in candidates if cell not in occupied]
            if len(candidates) < need:
                return False

            preferred = [cell for cell in candidates if _cell_distance_from_side(cell, rect, side) > need]
            source = preferred if len(preferred) >= need else candidates
            chosen = random.sample(source, need)
            for cell in chosen:
                r, c = cell
                grid[r, c] = CONTROL_COLOR
                occupied.add(cell)
                rect["controls"].append(cell)
    return True


def _expansion_cells(shape: tuple[int, int], rect: dict) -> set[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = rect["bbox"]
    counts = rect["counts"]
    cells: set[tuple[int, int]] = set()
    for r in range(max(0, r0 - counts["up"]), r0):
        for c in range(c0, c1 + 1):
            cells.add((r, c))
    for r in range(r1 + 1, min(h, r1 + 1 + counts["down"])):
        for c in range(c0, c1 + 1):
            cells.add((r, c))
    for r in range(r0, r1 + 1):
        for c in range(max(0, c0 - counts["left"]), c0):
            cells.add((r, c))
    for r in range(r0, r1 + 1):
        for c in range(c1 + 1, min(w, c1 + 1 + counts["right"])):
            cells.add((r, c))
    return cells


def _render_expected(inp: np.ndarray, rects: list[dict], split_c: int) -> np.ndarray:
    out = inp.copy()
    for rect in rects:
        target = RIGHT_PANEL_COLOR if rect["panel"] == "left" else LEFT_PANEL_COLOR
        r0, c0, r1, c1 = rect["bbox"]
        counts = rect["counts"]
        if counts["up"]:
            out[r0 - counts["up"] : r0, c0 : c1 + 1] = target
        if counts["down"]:
            out[r1 + 1 : r1 + 1 + counts["down"], c0 : c1 + 1] = target
        if counts["left"]:
            out[r0 : r1 + 1, c0 - counts["left"] : c0] = target
        if counts["right"]:
            out[r0 : r1 + 1, c1 + 1 : c1 + 1 + counts["right"]] = target
    for r, c in np.argwhere(inp == CONTROL_COLOR):
        out[r, c] = LEFT_PANEL_COLOR if c < split_c else RIGHT_PANEL_COLOR
    return out


def _validate_generated(inp: np.ndarray, rects: list[dict], split_c: int) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if _find_separator(inp) != split_c:
        return False
    found_rects = _find_rectangles(inp, split_c)
    if len(found_rects) != len(rects):
        return False

    expected_counts = [rect["counts"] for rect in rects]
    got_counts, controls = _count_controls(inp, found_rects, split_c)
    if got_counts != expected_counts:
        return False
    if len(controls) != sum(sum(rect["counts"].values()) for rect in rects):
        return False

    all_expansions: set[tuple[int, int]] = set()
    for rect in rects:
        cells = _expansion_cells(inp.shape, rect)
        if all_expansions & cells:
            return False
        all_expansions |= cells

    expected = _render_expected(inp, rects, split_c)
    actual = main(inp)
    if not np.array_equal(expected, actual):
        return False
    if np.array_equal(inp, actual):
        return False
    if not any(rect["panel"] == "left" for rect in rects):
        return False
    if not any(rect["panel"] == "right" for rect in rects):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_HEIGHT, MAX_HEIGHT)
    width = random.randint(MIN_WIDTH, MAX_WIDTH)
    split_c = random.randint(MIN_PANEL_WIDTH, width - MIN_PANEL_WIDTH - 1)

    left_count, right_count = _choose_panel_counts(height)
    left_rects = _build_panel_rects(height, width, split_c, "left", left_count)
    right_rects = _build_panel_rects(height, width, split_c, "right", right_count)
    if left_rects is None or right_rects is None:
        return None

    rects = left_rects + right_rects
    _assign_side_counts(rects)

    grid = np.full((height, width), RIGHT_PANEL_COLOR, dtype=int)
    grid[:, :split_c] = LEFT_PANEL_COLOR
    grid[:, split_c] = DIVIDER

    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = RECT_COLOR

    if not _place_controls(grid, split_c, rects):
        return None
    if not _validate_generated(grid, rects, split_c):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
