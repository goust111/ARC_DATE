from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
MARKER_COLORS = (1, 2, 3)
YELLOW = 4
FRAME = 5
RECT_SIZES = (7, 9, 11)
MAX_GEN_RETRIES = 400


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _x_mask(height: int, width: int) -> np.ndarray:
    assert height > 0 and width > 0
    if height > width:
        return _x_mask(width, height).T

    mask = np.zeros((height, width), dtype=bool)
    center_rows = {(height - 1) // 2, height // 2}
    for r in range(height):
        d = min(r, height - 1 - r)
        left = d
        right = width - 1 - d
        if r in center_rows:
            mask[r, left : right + 1] = True
        else:
            mask[r, left] = True
            mask[r, right] = True
    return mask


def _ring_fill(height: int, width: int, xmask: np.ndarray) -> np.ndarray:
    rows, cols = np.indices((height, width))
    center_r = height // 2
    center_c = width // 2
    dist = np.maximum(np.abs(rows - center_r), np.abs(cols - center_c))
    return (~xmask) & (dist % 2 == 0)


def _extract_rectangles(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert not np.any((g != BG) & (g != FRAME) & ~np.isin(g, MARKER_COLORS)), "unexpected colors in input"

    rects = []
    for comp in _components4(g != BG):
        r0, c0, r1, c1 = _bbox(comp)
        block = g[r0 : r1 + 1, c0 : c1 + 1]
        height, width = block.shape
        assert height in RECT_SIZES and width in RECT_SIZES, "rectangle size out of range"
        assert height % 2 == 1 and width % 2 == 1, "rectangle dimensions must be odd"
        assert np.all(block != BG), "rectangle component must fill its full bbox"

        border = np.concatenate((block[0, :], block[-1, :], block[1:-1, 0], block[1:-1, -1]))
        assert np.all(border == FRAME), "rectangle border must be uniformly gray"

        inner = block[1:-1, 1:-1]
        marker_cells = np.argwhere(np.isin(inner, MARKER_COLORS))
        assert marker_cells.shape[0] == 1, "each rectangle must contain exactly one interior marker"
        marker_color = int(inner[tuple(marker_cells[0])])
        rects.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "marker": marker_color,
            }
        )

    assert 2 <= len(rects) <= 3, "expected 2 or 3 rectangles"
    markers = [int(rect["marker"]) for rect in rects]
    assert len(set(markers)) == len(markers), "marker colors must be distinct across rectangles"
    return sorted(rects, key=lambda rect: (int(rect["r0"]), int(rect["c0"])))


def _render_output(shape: tuple[int, int], rects: list[dict[str, int | tuple[int, int, int, int]]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for rect in rects:
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        r1 = int(rect["r1"])
        c1 = int(rect["c1"])
        marker = int(rect["marker"])

        out[r0, c0 : c1 + 1] = FRAME
        out[r1, c0 : c1 + 1] = FRAME
        out[r0 : r1 + 1, c0] = FRAME
        out[r0 : r1 + 1, c1] = FRAME

        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        xmask = _x_mask(inner_h, inner_w)
        fill_mask = _ring_fill(inner_h, inner_w, xmask)
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[fill_mask] = YELLOW
        interior[xmask] = marker
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each marked solid gray rectangle with a gray frame whose interior combines a marker-colored X and alternating Chebyshev rings.

    Input structure:
    - The background is black.
    - The foreground consists of 2 or 3 non-overlapping solid gray rectangles, each with odd height and odd width.
    - Every rectangle contains exactly one interior marker pixel, and the marker colors are distinct choices from blue, red, and green.

    Rule:
    - Turn every solid rectangle into a hollow gray frame.
    - Inside each frame, draw the canonical X/cross pattern in that rectangle's marker color using the rectangle interior as the reference frame.
    - Then, for the remaining interior black cells only, fill the even Chebyshev-distance bands from the interior center with yellow; leave odd-distance cells black.

    Key insight:
    - The marker position is only a color cue: it selects the X color, while the ring fill is centered on the rectangle interior and must skip cells already occupied by the X.
    """
    g = np.asarray(grid, dtype=int)
    rects = _extract_rectangles(g)
    return _render_output(g.shape, rects)


def _shape_kind(size: tuple[int, int]) -> str:
    h, w = size
    if h < w:
        return "wide"
    if h > w:
        return "tall"
    return "square"


def _random_size(kind: str) -> tuple[int, int]:
    if kind == "wide":
        return random.choice([(7, 9), (7, 11), (9, 11)])
    if kind == "tall":
        return random.choice([(9, 7), (11, 7), (11, 9)])
    assert kind == "square"
    return random.choice([(9, 9), (11, 11)])


def _choose_sizes(n_rects: int) -> list[tuple[int, int]]:
    kinds = ["wide", "tall"]
    if n_rects == 3:
        kinds.append(random.choices(["square", "wide", "tall"], weights=[3, 2, 2], k=1)[0])
    sizes = [_random_size(kind) for kind in kinds]
    random.shuffle(sizes)
    return sizes


def _well_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 2) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_rectangles(shape: tuple[int, int], sizes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    boxes: list[tuple[int, int, int, int]] = []
    for rect_h, rect_w in sorted(sizes, key=lambda size: size[0] * size[1], reverse=True):
        placed = False
        for _ in range(200):
            r0 = random.randint(1, h - rect_h - 1)
            c0 = random.randint(1, w - rect_w - 1)
            box = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            if all(_well_separated(box, other) for other in boxes):
                boxes.append(box)
                placed = True
                break
        if not placed:
            return None
    random.shuffle(boxes)
    return boxes


def _marker_positions(rect_h: int, rect_w: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    xmask = _x_mask(rect_h - 2, rect_w - 2)
    on_x = [tuple(cell) for cell in np.argwhere(xmask)]
    off_x = [tuple(cell) for cell in np.argwhere(~xmask)]
    return on_x, off_x


def _render_input(
    shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    colors: tuple[int, ...],
) -> tuple[np.ndarray, list[dict[str, int]]]:
    grid = np.zeros(shape, dtype=int)
    rects: list[dict[str, int]] = []
    required_positions = ["off", "on"] + (["off"] if len(boxes) == 3 and random.random() < 0.5 else ["on"])
    required_positions = required_positions[: len(boxes)]

    for idx, (box, marker) in enumerate(zip(boxes, colors)):
        r0, c0, r1, c1 = box
        rect_h = r1 - r0 + 1
        rect_w = c1 - c0 + 1
        grid[r0 : r1 + 1, c0 : c1 + 1] = FRAME

        on_x, off_x = _marker_positions(rect_h, rect_w)
        desired = required_positions[idx]
        candidates = off_x if desired == "off" and off_x else on_x
        if desired == "on" and on_x:
            candidates = on_x
        elif desired == "off" and not off_x:
            candidates = on_x
        marker_r, marker_c = random.choice(candidates)
        grid[r0 + 1 + marker_r, c0 + 1 + marker_c] = marker

        rects.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1, "marker": marker})

    return grid, rects


def _validate_scene(inp: np.ndarray, rects: list[dict[str, int]], expected: np.ndarray) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    if not (2 <= len(rects) <= 3):
        return False
    kinds = {_shape_kind((int(rect["r1"]) - int(rect["r0"]) + 1, int(rect["c1"]) - int(rect["c0"]) + 1)) for rect in rects}
    if "wide" not in kinds or "tall" not in kinds:
        return False

    off_x_markers = 0
    for rect in rects:
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        r1 = int(rect["r1"])
        c1 = int(rect["c1"])
        marker = int(rect["marker"])
        rect_h = r1 - r0 + 1
        rect_w = c1 - c0 + 1
        xmask = _x_mask(rect_h - 2, rect_w - 2)
        fill_mask = _ring_fill(rect_h - 2, rect_w - 2, xmask)
        interior_in = inp[r0 + 1 : r1, c0 + 1 : c1]
        interior_out = expected[r0 + 1 : r1, c0 + 1 : c1]

        marker_pos = tuple(np.argwhere(interior_in == marker)[0])
        if not xmask[marker_pos]:
            off_x_markers += 1

        if not np.any(interior_out == marker):
            return False
        if not np.any(interior_out == YELLOW):
            return False
        if not np.any(interior_out == BG):
            return False
        if not np.array_equal(interior_out[xmask], np.full(np.count_nonzero(xmask), marker, dtype=int)):
            return False
        if not np.all(interior_out[fill_mask] == YELLOW):
            return False
        if not np.all(interior_out[(~xmask) & (~fill_mask)] == BG):
            return False
        if not np.any(xmask & ~fill_mask):
            return False
        if not np.any(fill_mask):
            return False

    return off_x_markers >= 1


def _attempt() -> np.ndarray | None:
    n_rects = random.choices([2, 3], weights=[2, 3], k=1)[0]
    shape = (random.randint(22, 28), random.randint(22, 28))
    sizes = _choose_sizes(n_rects)
    boxes = _place_rectangles(shape, sizes)
    if boxes is None:
        return None

    colors = tuple(random.sample(MARKER_COLORS, n_rects))
    inp, rects = _render_input(shape, boxes, colors)
    expected = _render_output(shape, rects)
    if not _validate_scene(inp, rects, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
