from __future__ import annotations

import numpy as np


BACKGROUND = 0
GREY = 5
INPUT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _find_rectangles(grid: np.ndarray) -> list[dict[str, int]]:
    """Extract monochrome 4-connected solid rectangles from a black background."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    rects: list[dict[str, int]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == BACKGROUND:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            rect_h = r1 - r0 + 1
            rect_w = c1 - c0 + 1
            assert len(coords) == rect_h * rect_w
            assert bool(np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color))

            rects.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "h": rect_h,
                    "w": rect_w,
                }
            )

    assert rects
    rects.sort(key=lambda item: (item["r0"], item["c0"], item["color"]))
    return rects


def _draw_rect(canvas: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    """Draw a solid rectangle, rejecting overlap and out-of-bounds placement."""
    H, W = canvas.shape
    assert 0 <= r0 <= H - h
    assert 0 <= c0 <= W - w
    target = canvas[r0 : r0 + h, c0 : c0 + w]
    assert not np.any(target != BACKGROUND)
    target[:] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each rectangular object by 90 degrees around its bbox center.

    Squares keep their original color because width/height swapping leaves their
    bbox unchanged. Non-squares are redrawn with swapped dimensions, centered on
    the original bbox center, and recolored grey.
    """
    g = np.asarray(grid, dtype=int)
    rects = _find_rectangles(g)
    out = np.full_like(g, BACKGROUND)

    for rect in rects:
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        h = int(rect["h"])
        w = int(rect["w"])
        color = int(rect["color"])

        if h == w:
            _draw_rect(out, r0, c0, h, w, color)
            continue

        new_h, new_w = w, h
        new_r0 = r0 + (h - new_h) // 2
        new_c0 = c0 + (w - new_w) // 2
        _draw_rect(out, new_r0, new_c0, new_h, new_w, GREY)

    return out


# --------------- generator (safe-square placement) ---------------

_RNG = np.random.default_rng()

GRID_H_RANGE = (16, 31)
GRID_W_RANGE = (16, 31)
OBJECT_COUNT_RANGE = (2, 7)
SAFE_SIDE_RANGE = (2, 9)
MAX_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200


def _centered_top_left(square_r0: int, square_c0: int, side: int, h: int, w: int) -> tuple[int, int]:
    """Top-left of an h×w rectangle centered inside a side×side safety square."""
    return square_r0 + (side - h) // 2, square_c0 + (side - w) // 2


def _sample_square_spec() -> dict[str, int]:
    side = int(_RNG.integers(SAFE_SIDE_RANGE[0], SAFE_SIDE_RANGE[1]))
    return {"h": side, "w": side, "side": side, "kind": 0}


def _sample_rectangle_spec() -> dict[str, int]:
    side = int(_RNG.integers(4, SAFE_SIDE_RANGE[1]))
    smaller_choices = [d for d in range(2, side) if d % 2 == side % 2]
    assert smaller_choices
    smaller = int(_RNG.choice(smaller_choices))
    if bool(_RNG.integers(0, 2)):
        h, w = side, smaller
    else:
        h, w = smaller, side
    return {"h": h, "w": w, "side": side, "kind": 1}


def _sample_specs() -> list[dict[str, int]]:
    count = int(_RNG.integers(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1]))
    kinds = [0, 1] + [int(_RNG.integers(0, 2)) for _ in range(count - 2)]
    _RNG.shuffle(kinds)
    specs = [_sample_square_spec() if kind == 0 else _sample_rectangle_spec() for kind in kinds]
    colors = _RNG.choice(INPUT_COLORS, size=count, replace=False)
    for spec, color in zip(specs, colors):
        spec["color"] = int(color)
    return specs


def _choose_grid_shape(specs: list[dict[str, int]]) -> tuple[int, int]:
    max_side = max(int(spec["side"]) for spec in specs)
    padded_area = sum((int(spec["side"]) + 2) ** 2 for spec in specs)

    for _ in range(PLACEMENT_ATTEMPTS):
        H = int(_RNG.integers(max(max_side + 4, GRID_H_RANGE[0]), GRID_H_RANGE[1]))
        W = int(_RNG.integers(max(max_side + 4, GRID_W_RANGE[0]), GRID_W_RANGE[1]))
        if H * W >= padded_area:
            return H, W

    span = int(np.ceil(np.sqrt(float(padded_area)))) + max_side
    H = max(GRID_H_RANGE[0], min(GRID_H_RANGE[1] - 1, span))
    W = max(GRID_W_RANGE[0], min(GRID_W_RANGE[1] - 1, span))
    return H, W


def _fits(occupied: np.ndarray, r0: int, c0: int, side: int) -> bool:
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + side > H or c0 + side > W:
        return False
    r_lo = max(0, r0 - 1)
    c_lo = max(0, c0 - 1)
    r_hi = min(H, r0 + side + 1)
    c_hi = min(W, c0 + side + 1)
    return not np.any(occupied[r_lo:r_hi, c_lo:c_hi])


def _place_specs(specs: list[dict[str, int]], H: int, W: int) -> bool:
    occupied = np.zeros((H, W), dtype=bool)
    order = sorted(range(len(specs)), key=lambda i: int(specs[i]["side"]), reverse=True)

    for idx in order:
        side = int(specs[idx]["side"])
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = int(_RNG.integers(0, H - side + 1))
            c0 = int(_RNG.integers(0, W - side + 1))
            if not _fits(occupied, r0, c0, side):
                continue
            occupied[r0 : r0 + side, c0 : c0 + side] = True
            specs[idx]["square_r0"] = r0
            specs[idx]["square_c0"] = c0
            placed = True
            break
        if not placed:
            return False
    return True


def _render_input(specs: list[dict[str, int]], H: int, W: int) -> np.ndarray:
    grid = np.full((H, W), BACKGROUND, dtype=int)
    for spec in specs:
        side = int(spec["side"])
        h = int(spec["h"])
        w = int(spec["w"])
        r0, c0 = _centered_top_left(int(spec["square_r0"]), int(spec["square_c0"]), side, h, w)
        _draw_rect(grid, r0, c0, h, w, int(spec["color"]))
    return grid


def _expected_output(specs: list[dict[str, int]], H: int, W: int) -> np.ndarray:
    out = np.full((H, W), BACKGROUND, dtype=int)
    for spec in specs:
        side = int(spec["side"])
        h = int(spec["h"])
        w = int(spec["w"])
        if h == w:
            r0, c0 = _centered_top_left(int(spec["square_r0"]), int(spec["square_c0"]), side, h, w)
            _draw_rect(out, r0, c0, h, w, int(spec["color"]))
            continue
        r0, c0 = _centered_top_left(int(spec["square_r0"]), int(spec["square_c0"]), side, w, h)
        _draw_rect(out, r0, c0, w, h, GREY)
    return out


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    H, W = _choose_grid_shape(specs)
    if not _place_specs(specs, H, W):
        return None

    grid = _render_input(specs, H, W)
    expected = _expected_output(specs, H, W)

    square_count = sum(1 for spec in specs if int(spec["h"]) == int(spec["w"]))
    rectangle_count = len(specs) - square_count
    assert square_count >= 1 and rectangle_count >= 1

    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
