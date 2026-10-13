import numpy as np


BG = 0
GREY = 5
ORANGE = 7
PALETTE = (1, 2, 3, 4, 6, 8, 9)

GRID_HEIGHTS = (8, 10, 12, 14)
GRID_WIDTHS = (10, 12, 14, 16)
PAIR_COUNT_RANGE = (3, 6)
GEN_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values = np.bincount(grid.ravel())
    return int(np.argmax(values))


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    divider_cols = []
    for c in range(w):
        column = grid[:, c]
        if np.all(column == column[0]) and int(column[0]) != bg:
            divider_cols.append((c, int(column[0])))
    assert len(divider_cols) == 1, "expected one full-height divider column"
    axis, color = divider_cols[0]
    assert 0 < axis < w - 1, "divider must split the board"
    return axis, color


def _collect_rectangles(grid: np.ndarray, bg: int, divider: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    rects = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (bg, divider) or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            assert 1 <= r1 - r0 + 1 <= 3
            assert 1 <= c1 - c0 + 1 <= 3
            expected_area = (r1 - r0 + 1) * (c1 - c0 + 1)
            assert len(coords) == expected_area, "components must be solid rectangles"
            assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)

            rects.append(
                {
                    "color": color,
                    "bbox": (r0, r1, c0, c1),
                }
            )

    return rects


def _mirror_bbox(bbox: tuple[int, int, int, int], axis: int) -> tuple[int, int, int, int]:
    r0, r1, c0, c1 = bbox
    return (r0, r1, 2 * axis - c1, 2 * axis - c0)


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, r1, c0, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Symmetrize colored rectangles across a grey divider.

    Each monochrome rectangle is compared with the rectangle at its mirror position:
    - missing mirror: copy the rectangle to the empty side
    - same-color pair: recolor both rectangles orange
    - different-color pair: swap the two colors across the divider
    The grey divider is removed in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h % 2 == 0 and w >= 5

    bg = _dominant_color(g)
    axis, divider = _find_divider(g, bg)
    pair_radius = min(axis, w - axis - 1)
    assert pair_radius >= 1, "divider must have mirrorable space on both sides"

    if axis - pair_radius > 0:
        assert np.all(g[:, : axis - pair_radius] == bg), "extra left margin must be background"
    if axis + pair_radius + 1 < w:
        assert np.all(g[:, axis + pair_radius + 1 :] == bg), "extra right margin must be background"

    rects = _collect_rectangles(g, bg, divider)
    bbox_to_idx = {rect["bbox"]: idx for idx, rect in enumerate(rects)}

    for rect in rects:
        r0, r1, c0, c1 = rect["bbox"]
        on_left = c1 < axis
        on_right = c0 > axis
        assert on_left ^ on_right, "rectangles must stay on one side of the divider"
        assert axis - pair_radius <= c0 and c1 <= axis + pair_radius

    out = g.copy()
    seen = set()
    for idx, rect in enumerate(rects):
        if idx in seen:
            continue

        bbox = rect["bbox"]
        mirror_bbox = _mirror_bbox(bbox, axis)
        mr0, mr1, mc0, mc1 = mirror_bbox
        assert 0 <= mr0 <= mr1 < h
        assert 0 <= mc0 <= mc1 < w
        assert mc1 != axis and mc0 != axis

        partner_idx = bbox_to_idx.get(mirror_bbox)
        if partner_idx is None:
            _paint_rect(out, mirror_bbox, rect["color"])
            seen.add(idx)
            continue

        partner = rects[partner_idx]
        if rect["color"] == partner["color"]:
            _paint_rect(out, bbox, ORANGE)
            _paint_rect(out, mirror_bbox, ORANGE)
        else:
            _paint_rect(out, bbox, partner["color"])
            _paint_rect(out, mirror_bbox, rect["color"])

        seen.add(idx)
        seen.add(partner_idx)

    out[:, axis] = bg
    return out


def _case_sequence(rng: np.random.Generator, count: int) -> list[str]:
    cases = ["copy", "merge", "swap"]
    while len(cases) < count:
        cases.append(str(rng.choice(("copy", "merge", "swap"))))
    rng.shuffle(cases)
    return cases


def _place_slots(
    rng: np.random.Generator, height: int, width: int, count: int
) -> list[tuple[int, int, int, int]] | None:
    blocked = np.zeros((height, width), dtype=bool)
    slots = []

    for _ in range(count):
        placed = False
        for _ in range(200):
            rect_h = int(rng.integers(1, 4))
            rect_w = int(rng.integers(1, min(4, width + 1)))
            if rect_w > width or rect_h > height:
                continue
            r0 = int(rng.integers(0, height - rect_h + 1))
            c0 = int(rng.integers(0, width - rect_w + 1))
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1

            rr0 = max(0, r0 - 1)
            rr1 = min(height, r1 + 2)
            cc0 = max(0, c0 - 1)
            cc1 = min(width, c1 + 2)
            if np.any(blocked[rr0:rr1, cc0:cc1]):
                continue

            blocked[rr0:rr1, cc0:cc1] = True
            slots.append((r0, r1, c0, c1))
            placed = True
            break

        if not placed:
            return None

    return slots


def _fill_case(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    axis: int,
    case: str,
    rng: np.random.Generator,
) -> None:
    mirror_bbox = _mirror_bbox(bbox, axis)

    if case == "copy":
        color = int(rng.choice(PALETTE))
        if rng.random() < 0.5:
            _paint_rect(grid, bbox, color)
        else:
            _paint_rect(grid, mirror_bbox, color)
        return

    if case == "merge":
        color = int(rng.choice(PALETTE))
        _paint_rect(grid, bbox, color)
        _paint_rect(grid, mirror_bbox, color)
        return

    left_color, right_color = rng.choice(PALETTE, size=2, replace=False)
    _paint_rect(grid, bbox, int(left_color))
    _paint_rect(grid, mirror_bbox, int(right_color))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.choice(GRID_HEIGHTS))
    total_width = int(rng.choice(GRID_WIDTHS))
    axis = total_width // 2 - 1 if rng.random() < 0.5 else total_width // 2
    pair_radius = min(axis, total_width - axis - 1)
    if pair_radius < 3:
        return None

    left_start = axis - pair_radius
    slot_count = int(rng.integers(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1] + 1))
    slots = _place_slots(rng, height, pair_radius, slot_count)
    if slots is None:
        return None

    grid = np.full((height, total_width), BG, dtype=int)
    grid[:, axis] = GREY
    cases = _case_sequence(rng, slot_count)

    for local_bbox, case in zip(slots, cases):
        r0, r1, c0, c1 = local_bbox
        bbox = (r0, r1, left_start + c0, left_start + c1)
        _fill_case(grid, bbox, axis, case, rng)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if ORANGE not in out:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-divider puzzle")
