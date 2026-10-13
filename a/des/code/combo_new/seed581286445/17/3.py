from __future__ import annotations

from collections import deque
import random

import numpy as np


PATTERN_COLOR = 5
FRAME_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return components


def _is_rectangular_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected_area = 2 * (h + w) - 4
    if int(comp["area"]) != expected_area:
        return False

    mask = np.zeros((h, w), dtype=bool)
    for r, c in comp["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def _frame_chain(grid: np.ndarray, bg: int) -> list[dict]:
    frames = [comp for comp in _find_components(grid, bg) if _is_rectangular_frame(comp)]
    by_bbox = {tuple(map(int, comp["bbox"])): comp for comp in frames}

    best: list[dict] = []
    for comp in frames:
        chain = [comp]
        cur = tuple(map(int, comp["bbox"]))
        while True:
            inner = (cur[0] + 2, cur[1] + 2, cur[2] - 2, cur[3] - 2)
            if inner not in by_bbox:
                break
            chain.append(by_bbox[inner])
            cur = inner
        if len(chain) > len(best):
            best = chain

    assert len(best) >= 3, "expected at least three nested frames"
    return best


def _outside_sequence(grid: np.ndarray, outer_bbox: tuple[int, int, int, int], bg: int) -> list[int]:
    r0, c0, r1, c1 = map(int, outer_bbox)
    mask = grid != int(bg)
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    coords = np.argwhere(mask)
    assert 2 <= len(coords) <= 4, "expected a short outside color sequence"

    rows = sorted({int(r) for r, _ in coords})
    assert len(rows) == 1, "outside sequence must be horizontal"
    row = rows[0]
    cols = sorted(int(c) for _r, c in coords)
    assert cols == list(range(cols[0], cols[-1] + 1)), "outside sequence must be contiguous"
    return [int(grid[row, c]) for c in cols]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the innermost motif by reading a span on the nested-frame color scale.

    Input structure:
    - The background is black.
    - Near the center sits one nested stack of rectangular one-pixel frames with
      one black cell of spacing between consecutive layers.
    - Each frame layer has a distinct color, ordered from the outermost frame to
      the innermost frame.
    - Inside the innermost frame is a small grey motif on black, and elsewhere in
      one corner is a short horizontal sequence made from frame colors.

    Rule:
    - Identify the nested frame chain and read its colors from outer to inner as
      an ordered scale of layer numbers.
    - Read the corner sequence from left to right, look up the first and last
      colors on that scale, and compute the absolute difference of their layer
      indices.
    - Crop the interior of the innermost frame, rotate that cropped motif area
      clockwise by that many quarter-turns, and return only the rotated interior.

    Key insight:
    - The frame colors are not decorative: their nesting order defines the ruler
      that turns the corner sequence's endpoint span into a rotation count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    chain = _frame_chain(g, bg)
    color_order = [int(comp["color"]) for comp in chain]
    color_to_index = {color: idx for idx, color in enumerate(color_order)}

    seq = _outside_sequence(g, tuple(chain[0]["bbox"]), bg)
    assert seq[0] in color_to_index and seq[-1] in color_to_index
    turns = abs(int(color_to_index[seq[0]]) - int(color_to_index[seq[-1]]))

    inner_r0, inner_c0, inner_r1, inner_c1 = map(int, chain[-1]["bbox"])
    interior = g[inner_r0 + 1 : inner_r1, inner_c0 + 1 : inner_c1].copy()
    return np.rot90(interior, -turns)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _rotation_signatures(mask: np.ndarray) -> set[tuple[tuple[int, int], bytes]]:
    trimmed = _trim_mask(mask)
    sigs = set()
    for k in range(4):
        rotated = np.rot90(trimmed, k)
        sigs.add((tuple(map(int, rotated.shape)), rotated.tobytes()))
    return sigs


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start_r = random.randrange(h)
    start_c = random.randrange(w)
    mask[start_r, start_c] = True

    while int(mask.sum()) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True
    return mask


def _make_pattern(inner_size: int) -> np.ndarray:
    for _ in range(200):
        max_dim = min(4, inner_size)
        h = random.randint(2, max_dim)
        w = random.randint(2, max_dim)
        max_area = min(h * w - 1, 6)
        if max_area < 3:
            continue
        area = random.randint(3, max_area)
        mask = _grow_connected_mask(h, w, area)
        if len(_rotation_signatures(mask)) != 4:
            continue

        top_min = 1 if inner_size - h >= 2 else 0
        left_min = 1 if inner_size - w >= 2 else 0
        top_max = inner_size - h - 1 if inner_size - h >= 2 else inner_size - h
        left_max = inner_size - w - 1 if inner_size - w >= 2 else inner_size - w
        top = random.randint(top_min, top_max)
        left = random.randint(left_min, left_max)

        canvas = np.zeros((inner_size, inner_size), dtype=int)
        canvas[top : top + h, left : left + w][mask] = PATTERN_COLOR
        if len(_rotation_signatures(canvas == PATTERN_COLOR)) != 4:
            continue
        return canvas
    raise RuntimeError("failed to build a non-symmetric pattern")


def _choose_layers(rotation_turns: int) -> tuple[int, int]:
    min_layers = max(3, rotation_turns + 1)
    layer_count = random.randint(min_layers, 5)
    diff_choices = [d for d in range(layer_count) if d % 4 == rotation_turns]
    diff = random.choice(diff_choices)
    return layer_count, diff


def _sequence_endpoints(layer_count: int, diff: int) -> tuple[int, int]:
    pairs = [
        (a, b)
        for a in range(layer_count)
        for b in range(layer_count)
        if abs(a - b) == diff
    ]
    return random.choice(pairs)


def _build_sequence(frame_colors: list[int], diff: int, length: int) -> list[int]:
    start_idx, end_idx = _sequence_endpoints(len(frame_colors), diff)
    seq = [int(frame_colors[start_idx])]
    for _ in range(max(0, length - 2)):
        seq.append(int(random.choice(frame_colors)))
    seq.append(int(frame_colors[end_idx]))
    return seq


def _place_sequence(
    grid: np.ndarray,
    seq: list[int],
    outer_bbox: tuple[int, int, int, int],
    corner: str,
) -> None:
    r0, c0, r1, c1 = map(int, outer_bbox)
    h, w = map(int, grid.shape)
    length = len(seq)

    if corner == "tl":
        row = random.randint(0, r0 - 1)
        col = random.randint(0, c0 - length)
    elif corner == "tr":
        row = random.randint(0, r0 - 1)
        col = random.randint(c1 + 1, w - length)
    elif corner == "bl":
        row = random.randint(r1 + 1, h - 1)
        col = random.randint(0, c0 - length)
    else:
        row = random.randint(r1 + 1, h - 1)
        col = random.randint(c1 + 1, w - length)

    for offset, color in enumerate(seq):
        grid[row, col + offset] = int(color)


def _build_canvas(
    outer_size: int,
    seq_length: int,
    corner: str,
) -> tuple[int, int, int, int]:
    for _ in range(200):
        top = random.randint(1, 3)
        bottom = random.randint(1, 3)
        left = random.randint(1, 4)
        right = random.randint(1, 4)

        if "t" in corner:
            top = max(top, 1)
        else:
            bottom = max(bottom, 1)
        if "l" in corner:
            left = max(left, seq_length)
        else:
            right = max(right, seq_length)

        if outer_size + top + bottom > 30 or outer_size + left + right > 30:
            continue
        return top, bottom, left, right
    raise RuntimeError("failed to choose canvas margins")


def _gen_attempt() -> np.ndarray | None:
    rotation_turns = random.randint(0, 3)
    layer_count, diff = _choose_layers(rotation_turns)
    seq_length = random.randint(2, 4)
    max_inner = min(7, 31 - seq_length - 4 * layer_count)
    if max_inner < 5:
        return None
    inner_size = random.randint(5, max_inner)
    outer_size = inner_size + 4 * layer_count - 2

    corner = random.choice(["tl", "tr", "bl", "br"])
    top, bottom, left, right = _build_canvas(outer_size, seq_length, corner)
    height = outer_size + top + bottom
    width = outer_size + left + right

    grid = np.zeros((height, width), dtype=int)
    frame_colors = random.sample(FRAME_COLOR_POOL, layer_count)
    pattern = _make_pattern(inner_size)

    for layer_idx, color in enumerate(frame_colors):
        inset = 2 * layer_idx
        r0 = top + inset
        c0 = left + inset
        r1 = top + outer_size - 1 - inset
        c1 = left + outer_size - 1 - inset
        grid[r0, c0 : c1 + 1] = int(color)
        grid[r1, c0 : c1 + 1] = int(color)
        grid[r0 : r1 + 1, c0] = int(color)
        grid[r0 : r1 + 1, c1] = int(color)

    inner_r0 = top + 2 * (layer_count - 1)
    inner_c0 = left + 2 * (layer_count - 1)
    grid[inner_r0 + 1 : inner_r0 + 1 + inner_size, inner_c0 + 1 : inner_c0 + 1 + inner_size] = pattern

    seq = _build_sequence(frame_colors, diff, seq_length)
    _place_sequence(
        grid,
        seq,
        (top, left, top + outer_size - 1, left + outer_size - 1),
        corner,
    )

    expected = np.rot90(pattern, -diff)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
