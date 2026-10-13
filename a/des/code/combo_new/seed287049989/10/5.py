from __future__ import annotations

import random

import numpy as np


BG = 0
GRAY = 5
ARROW_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
RECT_SIDE_RANGE = (2, 4)
EXTRA_RECT_RANGE = (0, 2)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)

    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _arrow_direction(
    bbox: tuple[int, int, int, int], h: int, w: int
) -> tuple[str, tuple[int, int], int]:
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1

    if height == 1:
        left_margin = c0
        right_margin = w - 1 - c1
        assert left_margin != right_margin, "horizontal arrow must be closer to one side edge"
        if left_margin < right_margin:
            return "right", (r0, c1), width - 1
        return "left", (r0, c0), width - 1

    assert width == 1, "arrow must be horizontal or vertical"
    top_margin = r0
    bottom_margin = h - 1 - r1
    assert top_margin != bottom_margin, "vertical arrow must be closer to one side edge"
    if top_margin < bottom_margin:
        return "down", (r1, c0), height - 1
    return "up", (r0, c0), height - 1


def _delta(direction: str, amount: int) -> tuple[int, int]:
    if direction == "right":
        return 0, amount
    if direction == "left":
        return 0, -amount
    if direction == "down":
        return amount, 0
    assert direction == "up"
    return -amount, 0


def main(grid: np.ndarray) -> np.ndarray:
    """Move each aligned gray rectangle by the nearby arrow's encoded distance, then recolor it with that arrow.

    Input structure
    - The background is black.
    - Several solid gray rectangles of different sizes are scattered in the grid.
    - Several colored arrows sit near the grid edges. Each arrow is a single-color
      horizontal or vertical 1-cell-thick line: a straight body plus one extra tip
      cell continuing in the pointing direction.
    - For each arrow, the tip's row (for horizontal arrows) or tip's column (for
      vertical arrows) intersects exactly one gray rectangle.

    Rule
    - Remove every arrow from the output.
    - For each arrow, infer its pointing direction from which border it is closest
      to, and let L be its body length, i.e. the full arrow length minus the tip.
    - Find the unique gray rectangle selected by the tip-aligned row or column.
    - Translate that rectangle by L cells in the arrow direction, erase its
      original position, and repaint the moved rectangle with the arrow's color.
    - Gray rectangles not selected by any arrow remain unchanged.

    Key insight
    - Each arrow simultaneously specifies the target rectangle, the translation
      distance, and the output color of the moved rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = BG if np.any(g == BG) else int(np.bincount(g.ravel()).argmax())

    objects: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for coords in _components(g == color):
            r0, c0, r1, c1 = _bbox(coords)
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                }
            )

    rectangles: list[dict[str, object]] = []
    arrows: list[dict[str, object]] = []
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]  # type: ignore[index]
        area = int(obj["area"])
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if area == height * width and height >= 2 and width >= 2:
            rectangles.append(obj)
        elif min(height, width) == 1 and area == max(height, width) and max(height, width) >= 3:
            arrows.append(obj)
        else:
            raise AssertionError("unexpected foreground component shape")

    assert rectangles and arrows, "expected both gray rectangles and colored arrows"
    rectangle_colors = {int(obj["color"]) for obj in rectangles}
    assert len(rectangle_colors) == 1, "expected a single shared rectangle color"
    rectangle_color = next(iter(rectangle_colors))
    assert all(int(obj["color"]) != rectangle_color for obj in arrows), "arrows must not be gray"

    arrow_specs: list[dict[str, object]] = []
    targeted_rects: set[int] = set()
    for arrow in arrows:
        direction, tip, amount = _arrow_direction(arrow["bbox"], h, w)  # type: ignore[arg-type]
        if direction in ("left", "right"):
            matches = [
                i
                for i, rect in enumerate(rectangles)
                if int(rect["r0"]) <= tip[0] <= int(rect["r1"])
            ]
        else:
            matches = [
                i
                for i, rect in enumerate(rectangles)
                if int(rect["c0"]) <= tip[1] <= int(rect["c1"])
            ]
        assert len(matches) == 1, "arrow must select exactly one rectangle"
        rect_idx = int(matches[0])
        assert rect_idx not in targeted_rects, "a rectangle may not be targeted twice"
        targeted_rects.add(rect_idx)
        arrow_specs.append(
            {
                "color": int(arrow["color"]),
                "coords": arrow["coords"],
                "direction": direction,
                "amount": int(amount),
                "rect_idx": rect_idx,
            }
        )

    out = g.copy()
    for spec in arrow_specs:
        for r, c in spec["coords"]:  # type: ignore[index]
            out[int(r), int(c)] = bg

    for rect_idx in targeted_rects:
        rect = rectangles[rect_idx]
        out[int(rect["r0"]) : int(rect["r1"]) + 1, int(rect["c0"]) : int(rect["c1"]) + 1] = bg

    for spec in arrow_specs:
        rect = rectangles[int(spec["rect_idx"])]
        dr, dc = _delta(str(spec["direction"]), int(spec["amount"]))
        nr0 = int(rect["r0"]) + dr
        nc0 = int(rect["c0"]) + dc
        nr1 = int(rect["r1"]) + dr
        nc1 = int(rect["c1"]) + dc
        assert 0 <= nr0 <= nr1 < h and 0 <= nc0 <= nc1 < w, "translated rectangle leaves the grid"
        assert np.all(out[nr0 : nr1 + 1, nc0 : nc1 + 1] == bg), "translated rectangles may not collide"
        out[nr0 : nr1 + 1, nc0 : nc1 + 1] = int(spec["color"])

    return out


def _interval_clear(start: int, end: int, spans: list[tuple[int, int]], gap: int = 1) -> bool:
    for a0, a1 in spans:
        if not (end + gap < a0 or start - gap > a1):
            return False
    return True


def _max_arrow_length(direction: str, r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> int:
    if direction == "right":
        return min(c0 - 2, w - 1 - c1)
    if direction == "left":
        return min(w - c1 - 3, c0)
    if direction == "down":
        return min(r0 - 2, h - 1 - r1)
    assert direction == "up"
    return min(h - r1 - 3, r0)


def _arrow_cells(
    direction: str, align: int, amount: int, h: int, w: int
) -> list[tuple[int, int]]:
    if direction == "right":
        return [(align, c) for c in range(amount + 1)]
    if direction == "left":
        return [(align, c) for c in range(w - amount - 1, w)]
    if direction == "down":
        return [(r, align) for r in range(amount + 1)]
    assert direction == "up"
    return [(r, align) for r in range(h - amount - 1, h)]


def _destination_bbox(
    direction: str, r0: int, c0: int, r1: int, c1: int, amount: int
) -> tuple[int, int, int, int]:
    dr, dc = _delta(direction, amount)
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _sample_rect_size(rng: random.Random) -> tuple[int, int]:
    return rng.randint(*RECT_SIDE_RANGE), rng.randint(*RECT_SIDE_RANGE)


def _choose_grid_size(rng: random.Random, rect_count: int) -> tuple[int, int]:
    min_side = max(18, rect_count * 4 + 4)
    return rng.randint(min_side, 30), rng.randint(min_side, 30)


def _place_targeted_rectangles(
    rng: random.Random, directions: list[str], h: int, w: int
) -> list[dict[str, int | str]]:
    row_spans: list[tuple[int, int]] = []
    col_spans: list[tuple[int, int]] = []
    arrow_occupied: set[tuple[int, int]] = set()
    rects: list[dict[str, int | str]] = []

    for direction in directions:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            rh, rw = _sample_rect_size(rng)
            if rh >= h or rw >= w:
                continue
            r0 = rng.randint(0, h - rh)
            c0 = rng.randint(0, w - rw)
            r1 = r0 + rh - 1
            c1 = c0 + rw - 1
            if not _interval_clear(r0, r1, row_spans, gap=1):
                continue
            if not _interval_clear(c0, c1, col_spans, gap=1):
                continue
            max_len = _max_arrow_length(direction, r0, c0, r1, c1, h, w)
            if max_len < 2:
                continue
            amount = rng.randint(2, max_len)
            if direction in ("left", "right"):
                aligns = list(range(r0, r1 + 1))
            else:
                aligns = list(range(c0, c1 + 1))
            rng.shuffle(aligns)
            for align in aligns:
                cells = _arrow_cells(direction, align, amount, h, w)
                if any(cell in arrow_occupied for cell in cells):
                    continue
                arrow_occupied.update(cells)
                row_spans.append((r0, r1))
                col_spans.append((c0, c1))
                rects.append(
                    {
                        "r0": r0,
                        "c0": c0,
                        "r1": r1,
                        "c1": c1,
                        "direction": direction,
                        "amount": amount,
                        "align": int(align),
                    }
                )
                placed = True
                break
            if placed:
                break
        if not placed:
            raise RuntimeError("failed to place targeted rectangle")

    return rects


def _place_extra_rectangles(
    rng: random.Random,
    rects: list[dict[str, int | str]],
    h: int,
    w: int,
    extra_count: int,
) -> None:
    row_spans = [(int(rect["r0"]), int(rect["r1"])) for rect in rects]
    col_spans = [(int(rect["c0"]), int(rect["c1"])) for rect in rects]

    for _ in range(extra_count):
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            rh, rw = _sample_rect_size(rng)
            if rh >= h or rw >= w:
                continue
            r0 = rng.randint(0, h - rh)
            c0 = rng.randint(0, w - rw)
            r1 = r0 + rh - 1
            c1 = c0 + rw - 1
            if not _interval_clear(r0, r1, row_spans, gap=1):
                continue
            if not _interval_clear(c0, c1, col_spans, gap=1):
                continue
            row_spans.append((r0, r1))
            col_spans.append((c0, c1))
            rects.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1})
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place extra rectangle")


def _render_input(
    h: int,
    w: int,
    rects: list[dict[str, int | str]],
    arrow_colors: list[int],
) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    grid = np.full((h, w), BG, dtype=int)
    arrows: list[dict[str, int | str]] = []

    for rect in rects:
        grid[int(rect["r0"]) : int(rect["r1"]) + 1, int(rect["c0"]) : int(rect["c1"]) + 1] = GRAY

    targeted = [rect for rect in rects if "direction" in rect]
    for rect, color in zip(targeted, arrow_colors):
        cells = _arrow_cells(
            str(rect["direction"]),
            int(rect["align"]),
            int(rect["amount"]),
            h,
            w,
        )
        for r, c in cells:
            assert int(grid[r, c]) == BG, "arrows must not overlap rectangles"
            grid[r, c] = int(color)
        arrows.append(
            {
                "r0": int(rect["r0"]),
                "c0": int(rect["c0"]),
                "r1": int(rect["r1"]),
                "c1": int(rect["c1"]),
                "direction": str(rect["direction"]),
                "amount": int(rect["amount"]),
                "color": int(color),
            }
        )

    return grid, arrows


def _expected_output(
    h: int,
    w: int,
    rects: list[dict[str, int | str]],
    arrows: list[dict[str, int | str]],
) -> np.ndarray | None:
    out = np.full((h, w), BG, dtype=int)

    targeted_boxes = {
        (int(arrow["r0"]), int(arrow["c0"]), int(arrow["r1"]), int(arrow["c1"])) for arrow in arrows
    }
    for rect in rects:
        box = (int(rect["r0"]), int(rect["c0"]), int(rect["r1"]), int(rect["c1"]))
        if box not in targeted_boxes:
            out[box[0] : box[2] + 1, box[1] : box[3] + 1] = GRAY

    for arrow in arrows:
        nr0, nc0, nr1, nc1 = _destination_bbox(
            str(arrow["direction"]),
            int(arrow["r0"]),
            int(arrow["c0"]),
            int(arrow["r1"]),
            int(arrow["c1"]),
            int(arrow["amount"]),
        )
        if not (0 <= nr0 <= nr1 < h and 0 <= nc0 <= nc1 < w):
            return None
        if not np.all(out[nr0 : nr1 + 1, nc0 : nc1 + 1] == BG):
            return None
        out[nr0 : nr1 + 1, nc0 : nc1 + 1] = int(arrow["color"])

    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    arrow_count = rng.randint(1, 4)
    directions = rng.sample(["right", "left", "down", "up"], arrow_count)
    extra_count = rng.randint(*EXTRA_RECT_RANGE)
    rect_count = arrow_count + extra_count
    h, w = _choose_grid_size(rng, rect_count)

    try:
        rects = _place_targeted_rectangles(rng, directions, h, w)
        _place_extra_rectangles(rng, rects, h, w, extra_count)
    except RuntimeError:
        return None

    arrow_colors = rng.sample(ARROW_COLORS, arrow_count)
    grid, arrows = _render_input(h, w, rects, arrow_colors)
    expected = _expected_output(h, w, rects, arrows)
    if expected is None:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
