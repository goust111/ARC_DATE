import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = tuple(color for color in range(10) if color not in (BLACK, GREY))
SIDES = ("top", "bottom", "left", "right")

# Generator configuration.
GRID_H_RANGE = (14, 28)
GRID_W_RANGE = (14, 28)
OBJECT_COUNT_CHOICES = (1, 2, 2, 3)
STRIPE_COUNT_RANGE = (1, 4)
LEAD_BLANK_MAX = 2
TRAIL_BLANK_MAX = 2
SPAN_RANGE = (3, 8)
GEN_ATTEMPTS = 500
PLACE_ATTEMPTS = 250


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _rectangle_boundary(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _detect_boxes(grid: np.ndarray) -> list[dict[str, object]]:
    boxes: list[dict[str, object]] = []
    for comp in _components_of_color(grid, GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        comp_set = set(comp)
        boundary = _rectangle_boundary(r0, c0, r1, c1)
        if comp_set - boundary:
            continue

        missing = boundary - comp_set
        if len(missing) != 1:
            continue

        gap = next(iter(missing))
        gap_r, gap_c = gap
        if (gap_r in (r0, r1)) and (gap_c in (c0, c1)):
            continue

        if gap_r == r0:
            side = "top"
        elif gap_r == r1:
            side = "bottom"
        elif gap_c == c0:
            side = "left"
        else:
            side = "right"

        boxes.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "side": side,
            }
        )

    boxes.sort(key=lambda box: (box["bbox"][0], box["bbox"][1]))  # type: ignore[index]
    return boxes


def _read_stripes(grid: np.ndarray, box: dict[str, object]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box["bbox"]  # type: ignore[misc]
    side = box["side"]  # type: ignore[assignment]
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert inner.size > 0, "frame must have a non-empty interior"

    stripes: list[tuple[int, int]] = []
    if side in ("left", "right"):
        last_row = None
        for local_r in range(inner.shape[0]):
            row = inner[local_r]
            nz = np.flatnonzero(row != BLACK)
            if nz.size == 0:
                continue
            colors = {int(v) for v in row[nz]}
            assert GREY not in colors, "interior stripes may not use the frame color"
            assert len(colors) == 1, "each stripe row must be monochrome"
            assert int(nz[-1]) - int(nz[0]) + 1 == int(nz.size), "each stripe row must be contiguous"
            if last_row is not None:
                assert local_r - last_row >= 2, "horizontal stripes must be separated by black rows"
            last_row = local_r
            color = next(iter(colors))
            stripes.append((color, int(nz.size)))
    else:
        last_col = None
        for local_c in range(inner.shape[1]):
            col = inner[:, local_c]
            nz = np.flatnonzero(col != BLACK)
            if nz.size == 0:
                continue
            colors = {int(v) for v in col[nz]}
            assert GREY not in colors, "interior stripes may not use the frame color"
            assert len(colors) == 1, "each stripe column must be monochrome"
            assert int(nz[-1]) - int(nz[0]) + 1 == int(nz.size), "each stripe column must be contiguous"
            if last_col is not None:
                assert local_c - last_col >= 2, "vertical stripes must be separated by black columns"
            last_col = local_c
            color = next(iter(colors))
            stripes.append((color, int(nz.size)))

    assert stripes, "each frame must contain at least one stripe"
    assert len({color for color, _ in stripes}) == len(stripes), "stripe colors must be distinct"
    return stripes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored tentacles outside each grey frame.

    The gap side determines the growth direction. The inner stripes are read
    perpendicular to that direction, and each stripe becomes one same-colored
    tentacle whose length matches the stripe's pixel length.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    boxes = _detect_boxes(g)
    assert boxes, "expected at least one grey frame with a single-pixel gap"

    for box in boxes:
        stripes = _read_stripes(g, box)
        gap_r, gap_c = box["gap"]  # type: ignore[misc]
        side = box["side"]  # type: ignore[assignment]

        for idx, (color, length) in enumerate(stripes):
            if side == "right":
                rr = gap_r + 2 * idx
                for step in range(1, length + 1):
                    cc = gap_c + step
                    if cc >= w:
                        break
                    if g[rr, cc] == BLACK:
                        out[rr, cc] = color
            elif side == "left":
                rr = gap_r + 2 * idx
                for step in range(1, length + 1):
                    cc = gap_c - step
                    if cc < 0:
                        break
                    if g[rr, cc] == BLACK:
                        out[rr, cc] = color
            elif side == "bottom":
                cc = gap_c + 2 * idx
                for step in range(1, length + 1):
                    rr = gap_r + step
                    if rr >= h:
                        break
                    if g[rr, cc] == BLACK:
                        out[rr, cc] = color
            else:
                cc = gap_c + 2 * idx
                for step in range(1, length + 1):
                    rr = gap_r - step
                    if rr < 0:
                        break
                    if g[rr, cc] == BLACK:
                        out[rr, cc] = color

    return out


def _sample_spec() -> dict[str, object]:
    side = random.choice(SIDES)
    stripe_count = random.randint(*STRIPE_COUNT_RANGE)
    lead = random.randint(0, LEAD_BLANK_MAX)
    trail = random.randint(0, TRAIL_BLANK_MAX)
    colors = random.sample(PALETTE, stripe_count)

    if side in ("left", "right"):
        inner_h = lead + 2 * stripe_count - 1 + trail
        inner_w = random.randint(*SPAN_RANGE)
        lengths = [random.randint(1, inner_w - 1) for _ in range(stripe_count)]
    else:
        inner_w = lead + 2 * stripe_count - 1 + trail
        inner_h = random.randint(*SPAN_RANGE)
        lengths = [random.randint(1, inner_h - 1) for _ in range(stripe_count)]

    return {
        "side": side,
        "lead": lead,
        "lengths": lengths,
        "colors": colors,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": inner_h + 2,
        "outer_w": inner_w + 2,
    }


def _input_paints(spec: dict[str, object], top: int, left: int) -> list[tuple[int, int, int]]:
    side = spec["side"]
    lead = spec["lead"]
    lengths = spec["lengths"]
    colors = spec["colors"]
    inner_h = spec["inner_h"]
    inner_w = spec["inner_w"]
    outer_h = spec["outer_h"]
    outer_w = spec["outer_w"]

    r0, c0 = top, left
    r1, c1 = top + outer_h - 1, left + outer_w - 1
    paints: list[tuple[int, int, int]] = []

    for c in range(c0, c1 + 1):
        paints.append((r0, c, GREY))
        paints.append((r1, c, GREY))
    for r in range(r0 + 1, r1):
        paints.append((r, c0, GREY))
        paints.append((r, c1, GREY))

    if side == "top":
        gap = (r0, c0 + 1 + lead)
    elif side == "bottom":
        gap = (r1, c0 + 1 + lead)
    elif side == "left":
        gap = (r0 + 1 + lead, c0)
    else:
        gap = (r0 + 1 + lead, c1)
    paints = [paint for paint in paints if (paint[0], paint[1]) != gap]

    if side == "right":
        for idx, (color, length) in enumerate(zip(colors, lengths, strict=False)):
            rr = r0 + 1 + lead + 2 * idx
            for dc in range(length):
                paints.append((rr, c0 + 1 + dc, color))
    elif side == "left":
        for idx, (color, length) in enumerate(zip(colors, lengths, strict=False)):
            rr = r0 + 1 + lead + 2 * idx
            start = c0 + inner_w - length
            for dc in range(length):
                paints.append((rr, start + dc, color))
    elif side == "bottom":
        for idx, (color, length) in enumerate(zip(colors, lengths, strict=False)):
            cc = c0 + 1 + lead + 2 * idx
            for dr in range(length):
                paints.append((r0 + 1 + dr, cc, color))
    else:
        for idx, (color, length) in enumerate(zip(colors, lengths, strict=False)):
            cc = c0 + 1 + lead + 2 * idx
            start = r0 + inner_h - length
            for dr in range(length):
                paints.append((start + dr, cc, color))

    return paints


def _tentacle_paints(
    spec: dict[str, object],
    top: int,
    left: int,
    height: int,
    width: int,
) -> list[tuple[int, int, int]]:
    side = spec["side"]
    lead = spec["lead"]
    lengths = spec["lengths"]
    colors = spec["colors"]
    outer_h = spec["outer_h"]
    outer_w = spec["outer_w"]

    r0, c0 = top, left
    r1, c1 = top + outer_h - 1, left + outer_w - 1
    paints: list[tuple[int, int, int]] = []

    if side == "right":
        gap_r = r0 + 1 + lead
        for idx, (color, length) in enumerate(zip(colors, lengths, strict=False)):
            rr = gap_r + 2 * idx
            for step in range(1, length + 1):
                cc = c1 + step
                if cc >= width:
                    break
                paints.append((rr, cc, color))
    elif side == "left":
        gap_r = r0 + 1 + lead
        for idx, (color, length) in enumerate(zip(colors, lengths, strict=False)):
            rr = gap_r + 2 * idx
            for step in range(1, length + 1):
                cc = c0 - step
                if cc < 0:
                    break
                paints.append((rr, cc, color))
    elif side == "bottom":
        gap_c = c0 + 1 + lead
        for idx, (color, length) in enumerate(zip(colors, lengths, strict=False)):
            cc = gap_c + 2 * idx
            for step in range(1, length + 1):
                rr = r1 + step
                if rr >= height:
                    break
                paints.append((rr, cc, color))
    else:
        gap_c = c0 + 1 + lead
        for idx, (color, length) in enumerate(zip(colors, lengths, strict=False)):
            cc = gap_c + 2 * idx
            for step in range(1, length + 1):
                rr = r0 - step
                if rr < 0:
                    break
                paints.append((rr, cc, color))

    return paints


def _expanded(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    halo = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    halo.add((rr, cc))
    return halo


def _paint(grid: np.ndarray, paints: list[tuple[int, int, int]]) -> None:
    for r, c, color in paints:
        grid[r, c] = color


def _build_expected(inp: np.ndarray, tentacles: list[tuple[int, int, int]]) -> np.ndarray:
    out = inp.copy()
    for r, c, color in tentacles:
        if out[r, c] == BLACK:
            out[r, c] = color
    return out


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    grid = np.full((height, width), BLACK, dtype=int)
    reserved: set[tuple[int, int]] = set()
    all_tentacles: list[tuple[int, int, int]] = []

    for _ in range(random.choice(OBJECT_COUNT_CHOICES)):
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            spec = _sample_spec()
            outer_h = spec["outer_h"]
            outer_w = spec["outer_w"]
            if outer_h > height - 2 or outer_w > width - 2:
                continue

            top = random.randint(1, height - outer_h - 1)
            left = random.randint(1, width - outer_w - 1)
            input_paints = _input_paints(spec, top, left)
            tentacle_paints = _tentacle_paints(spec, top, left, height, width)
            if not tentacle_paints:
                continue

            occupied = {(r, c) for r, c, _ in input_paints}
            occupied.update((r, c) for r, c, _ in tentacle_paints)
            if occupied & reserved:
                continue

            _paint(grid, input_paints)
            all_tentacles.extend(tentacle_paints)
            reserved |= _expanded(occupied, height, width)
            placed = True
            break

        if not placed:
            return None

    expected = _build_expected(grid, all_tentacles)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _attempt_generate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
