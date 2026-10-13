import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

SOURCE_COLORS = (BLUE, RED, GREEN)
ROTATION_BY_COLOR = {
    RED: 0,
    GREEN: 1,   # 90 clockwise
    BLUE: 2,    # 180
}

SIDE_PRIORITY = ("top", "left", "bottom", "right")
SIDE_TO_MARKER = {
    "top": YELLOW,
    "left": MAROON,
    "bottom": MAGENTA,
    "right": ORANGE,
}
MARKER_TO_SIDE = {value: key for key, value in SIDE_TO_MARKER.items()}

_GEN_CALL_COUNT = 0
_POSITION_CACHE: dict[int, list[tuple[tuple[int, int], ...]]] = {}


def _mid_index(length: int) -> int:
    return (int(length) - 1) // 2


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_connected4(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not coords:
        return False
    seen = {coords[0]}
    stack = [coords[0]]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _rotations_all_distinct(mask: np.ndarray) -> bool:
    variants = [np.rot90(mask, k) for k in range(4)]
    return all(not np.array_equal(variants[i], variants[j]) for i in range(4) for j in range(i + 1, 4))


def _rotate_clockwise(arr: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(arr, -int(quarter_turns))


def _nearest_side_name(block_r: int, block_c: int) -> str:
    center_r = block_r * 4 + 1.5
    center_c = block_c * 4 + 1.5
    distances = {
        "top": center_r,
        "left": center_c,
        "bottom": 15 - center_r,
        "right": 15 - center_c,
    }
    best_side = SIDE_PRIORITY[0]
    best_distance = distances[best_side]
    for side in SIDE_PRIORITY[1:]:
        if distances[side] < best_distance:
            best_side = side
            best_distance = distances[side]
    return best_side


def _non_adjacent(coords: tuple[tuple[int, int], ...]) -> bool:
    for (r1, c1), (r2, c2) in itertools.combinations(coords, 2):
        if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
            return False
    return True


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    h, w = grid.shape
    assert h >= 10 and w >= 6

    mid_r = _mid_index(h)
    mid_c = _mid_index(w)
    side_colors = {
        "top": int(grid[0, mid_c]),
        "right": int(grid[mid_r, w - 1]),
        "bottom": int(grid[h - 1, mid_c]),
        "left": int(grid[mid_r, 0]),
    }
    assert side_colors == SIDE_TO_MARKER, "expected fixed border midpoint markers"

    source_coords = np.argwhere(np.isin(grid, SOURCE_COLORS))
    assert 2 <= len(source_coords) <= 4, "source panel must have 2-4 foreground pixels"
    sr0, sc0, sr1, sc1 = _bbox(source_coords)
    assert (sr1 - sr0 + 1, sc1 - sc0 + 1) == (4, 4), "source foreground must span a 4x4 panel"
    source = grid[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
    assert source.shape == (4, 4)
    assert set(map(int, source.flatten())).issubset({BLACK, *SOURCE_COLORS})
    source_pts = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(source != BLACK)))
    assert _non_adjacent(source_pts), "source foreground pixels must be separated"

    template_r0 = sr1 + 2
    template_r1 = template_r0 + 4
    assert template_r1 <= h - 1, "template panel must fit below the source panel"
    assert np.all(grid[sr1 + 1, sc0 : sc0 + 4] == BLACK), "expected one blank separator row"
    template = grid[template_r0:template_r1, sc0 : sc0 + 4].copy()
    assert template.shape == (4, 4)
    assert set(map(int, template.flatten())).issubset({BLACK, GREY})

    template_mask = template == GREY
    template_cells = int(template_mask.sum())
    assert 3 <= template_cells <= 7, "template size must be 3-7"
    assert _is_connected4(template_mask), "template must be 4-connected"
    assert _rotations_all_distinct(template_mask), "template must distinguish all four rotations"

    return source, template_mask, side_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp a rotated template into each output block, with source color choosing rotation and block position choosing stamp color.

    Input structure:
    - The grid has black background plus four fixed border-midpoint markers: top yellow, right orange, bottom magenta, left maroon.
    - Inside the grid, an upper 4x4 source panel contains 2-4 separated pixels colored red/green/blue on black.
    - Exactly one blank row below it, a lower 4x4 template panel contains a 4-connected asymmetric grey shape on black.

    Rule:
    - Create a 16x16 black output.
    - For each non-black source pixel at source-panel position (r, c), rotate the grey template by a color-coded angle: red->0, green->90 clockwise, blue->180.
    - Place that rotated template into output block (r, c), but recolor all template cells by the border-marker color of the nearest output side to that block center, breaking ties by top, then left, then bottom, then right.

    Key insight:
    - Each source pixel splits into two independent signals: its color fixes orientation, while its position fixes the stamp color through nearest-side assignment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    source, template_mask, side_colors = _extract_panels(g)
    out = np.zeros((16, 16), dtype=int)

    for r, c in np.argwhere(source != BLACK):
        src_color = int(source[r, c])
        rotation = ROTATION_BY_COLOR[src_color]
        rotated_mask = _rotate_clockwise(template_mask, rotation)
        side = _nearest_side_name(int(r), int(c))
        stamp_color = side_colors[side]
        block = out[r * 4 : (r + 1) * 4, c * 4 : (c + 1) * 4]
        block[rotated_mask] = stamp_color

    return out


def _position_sets(count: int) -> list[tuple[tuple[int, int], ...]]:
    cached = _POSITION_CACHE.get(count)
    if cached is not None:
        return cached

    cells = [(r, c) for r in range(4) for c in range(4)]
    out = []
    for combo in itertools.combinations(cells, count):
        if not _non_adjacent(combo):
            continue
        rows = [r for r, _ in combo]
        cols = [c for _, c in combo]
        if max(rows) - min(rows) != 3 or max(cols) - min(cols) != 3:
            continue
        out.append(combo)
    _POSITION_CACHE[count] = out
    return out


def _pick_source_layout(mode: int) -> tuple[np.ndarray, list[dict[str, int]]]:
    rng = random

    if mode % 3 == 2:
        count = 2
    elif mode % 3 == 1:
        count = 3
    else:
        count = 4

    candidates = _position_sets(count)
    rng.shuffle(candidates)

    repeated_color = (RED, GREEN, BLUE)[mode % 3]
    other_color = {RED, GREEN, BLUE}.difference({repeated_color}).pop()
    third_color = ({RED, GREEN, BLUE} - {repeated_color, other_color}).pop()

    for coords in candidates:
        sides = [_nearest_side_name(r, c) for r, c in coords]
        counts_by_side = {side: sides.count(side) for side in set(sides)}
        if count == 2:
            if len(set(sides)) < 2:
                continue
            colors = [repeated_color, repeated_color]
        else:
            repeated_side = next((side for side, n in counts_by_side.items() if n >= 2), None)
            if repeated_side is None or len(set(sides)) < 2:
                continue

            colors = [None] * count
            repeated_side_indices = [i for i, side in enumerate(sides) if side == repeated_side]
            other_indices = [i for i, side in enumerate(sides) if side != repeated_side]
            colors[repeated_side_indices[0]] = repeated_color
            colors[repeated_side_indices[1]] = other_color
            colors[other_indices[0]] = repeated_color
            remaining_palette = [third_color, other_color, repeated_color]
            for idx in range(count):
                if colors[idx] is None:
                    colors[idx] = remaining_palette[(idx + mode) % len(remaining_palette)]

            same_color_diff_side = any(
                colors[i] == colors[j] and sides[i] != sides[j]
                for i in range(count)
                for j in range(i + 1, count)
            )
            same_side_diff_color = any(
                sides[i] == sides[j] and colors[i] != colors[j]
                for i in range(count)
                for j in range(i + 1, count)
            )
            if not (same_color_diff_side and same_side_diff_color):
                continue

        source = np.zeros((4, 4), dtype=int)
        info = []
        for (r, c), color in zip(coords, colors):
            source[r, c] = color
            info.append({"r": r, "c": c, "color": color, "side": _nearest_side_name(r, c)})
        return source, info

    raise RuntimeError("failed to choose a source layout")


def _random_connected_template(cell_count: int) -> np.ndarray:
    rng = random
    cells = [(r, c) for r in range(4) for c in range(4)]
    start = rng.choice(cells)
    chosen = {start}
    while len(chosen) < cell_count:
        frontier = []
        for r, c in chosen:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < 4 and 0 <= nc < 4 and (nr, nc) not in chosen:
                    frontier.append((nr, nc))
        if not frontier:
            return np.zeros((4, 4), dtype=bool)
        chosen.add(rng.choice(frontier))
    mask = np.zeros((4, 4), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _generate_template_mask(mode: int) -> np.ndarray:
    counts = [3, 4, 5, 6, 7]
    random.shuffle(counts)
    counts = counts[mode % len(counts) :] + counts[: mode % len(counts)]
    for cell_count in counts:
        for _ in range(80):
            mask = _random_connected_template(cell_count)
            if int(mask.sum()) != cell_count:
                continue
            if not _is_connected4(mask):
                continue
            if not _rotations_all_distinct(mask):
                continue
            return mask
    raise RuntimeError("failed to generate a template mask")


def _render_input(
    h: int,
    w: int,
    source_top: int,
    left: int,
    source: np.ndarray,
    template_mask: np.ndarray,
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[0, _mid_index(w)] = YELLOW
    grid[_mid_index(h), w - 1] = ORANGE
    grid[h - 1, _mid_index(w)] = MAGENTA
    grid[_mid_index(h), 0] = MAROON

    grid[source_top : source_top + 4, left : left + 4] = source
    template = np.where(template_mask, GREY, BLACK).astype(int)
    grid[source_top + 5 : source_top + 9, left : left + 4] = template
    return grid


def _oracle_from_parts(source: np.ndarray, template_mask: np.ndarray) -> np.ndarray:
    out = np.zeros((16, 16), dtype=int)
    for r, c in np.argwhere(source != BLACK):
        src_color = int(source[r, c])
        rotated_mask = _rotate_clockwise(template_mask, ROTATION_BY_COLOR[src_color])
        stamp_color = SIDE_TO_MARKER[_nearest_side_name(int(r), int(c))]
        out[r * 4 : (r + 1) * 4, c * 4 : (c + 1) * 4][rotated_mask] = stamp_color
    return out


def _validate_semantics(grid: np.ndarray, source: np.ndarray, template_mask: np.ndarray, info: list[dict[str, int]]) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    expected = _oracle_from_parts(source, template_mask)
    if not np.array_equal(out, expected):
        return False

    if template_mask.shape != (4, 4) or not _rotations_all_distinct(template_mask):
        return False

    nonempty_blocks = []
    for item in info:
        r = item["r"]
        c = item["c"]
        block = out[r * 4 : (r + 1) * 4, c * 4 : (c + 1) * 4]
        colors = set(map(int, block.flatten())) - {BLACK}
        if colors != {SIDE_TO_MARKER[item["side"]]}:
            return False
        nonempty_blocks.append(block.copy())

    if len({block.tobytes() for block in nonempty_blocks}) < 2:
        return False

    same_color_diff_side = any(
        a["color"] == b["color"] and a["side"] != b["side"]
        for a, b in itertools.combinations(info, 2)
    )
    same_side_diff_color = any(
        a["side"] == b["side"] and a["color"] != b["color"]
        for a, b in itertools.combinations(info, 2)
    )

    if len(info) == 2:
        return same_color_diff_side
    return same_color_diff_side and same_side_diff_color


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    for _ in range(200):
        mode = _GEN_CALL_COUNT
        _GEN_CALL_COUNT += 1

        h = random.randint(12, 15)
        w = random.randint(6, 8)
        source_top = random.randint(1, h - 10)
        left = random.randint(1, w - 5)

        source, info = _pick_source_layout(mode)
        template_mask = _generate_template_mask(mode)
        grid = _render_input(h, w, source_top, left, source, template_mask)

        if _validate_semantics(grid, source, template_mask, info):
            return grid

    raise RuntimeError("failed to generate a valid input")
