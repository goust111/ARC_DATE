from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

LEFT_GLYPH_COLOR = BLUE
RIGHT_GLYPH_COLOR = RED

MIN_GRID_ROWS = 17
MAX_GRID_ROWS = 24
MIN_GRID_COLS = 11
MAX_GRID_COLS = 15
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _glyph_catalog() -> tuple[np.ndarray, ...]:
    glyphs: list[np.ndarray] = []
    for bits in range(1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        area = int(mask.sum())
        if not (3 <= area <= 6):
            continue
        if _is_connected(mask):
            glyphs.append(mask)
    return tuple(glyphs)


GLYPH_CATALOG = _glyph_catalog()
LEFT_BLOCKING_GLYPHS = tuple(mask for mask in GLYPH_CATALOG if bool(mask[0, 2]))
RIGHT_SAFE_GLYPHS = tuple(mask for mask in GLYPH_CATALOG if not bool(mask[0, 0]))


def _stamp(grid: np.ndarray, center_r: int, center_c: int, color: int, mask: np.ndarray) -> None:
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if not bool(mask[dr + 1, dc + 1]):
                continue
            rr = center_r + dr
            cc = center_c + dc
            if _in_bounds(grid.shape, rr, cc):
                grid[rr, cc] = int(color)


def _extract_dictionary(grid: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
    grey_coords = np.argwhere(grid == GREY)
    assert grey_coords.shape[0] == 3, "expected one 3-cell grey separator"
    sep_cols = np.unique(grey_coords[:, 1])
    assert sep_cols.size == 1, "grey separator must be vertical"
    sep_col = int(sep_cols[0])
    rows = np.sort(grey_coords[:, 0])
    assert np.array_equal(rows, np.arange(int(rows[0]), int(rows[0]) + 3)), "grey separator must be contiguous"
    dict_r0 = int(rows[0])
    assert dict_r0 in (0, 1), "dictionary must start on row 0 or 1"
    assert sep_col - 3 >= 0 and sep_col + 3 < grid.shape[1], "dictionary must fit horizontally"

    left_patch = grid[dict_r0 : dict_r0 + 3, sep_col - 3 : sep_col]
    right_patch = grid[dict_r0 : dict_r0 + 3, sep_col + 1 : sep_col + 4]
    left_mask = left_patch == LEFT_GLYPH_COLOR
    right_mask = right_patch == RIGHT_GLYPH_COLOR

    assert 3 <= int(left_mask.sum()) <= 6 and 3 <= int(right_mask.sum()) <= 6, "glyph areas out of range"
    assert _is_connected(left_mask) and _is_connected(right_mask), "glyphs must be 4-connected"
    assert not np.array_equal(left_mask, right_mask), "glyphs must differ"
    assert np.all(np.isin(left_patch, [BLACK, LEFT_GLYPH_COLOR])), "left glyph patch has invalid colors"
    assert np.all(np.isin(right_patch, [BLACK, RIGHT_GLYPH_COLOR])), "right glyph patch has invalid colors"

    return dict_r0, sep_col, left_mask, right_mask


def _simulate(
    grid: np.ndarray,
    *,
    swap_glyphs: bool = False,
    enable_stamping: bool = True,
    enable_collision: bool = True,
) -> tuple[np.ndarray, dict]:
    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape
    dict_r0, _sep_col, left_mask, right_mask = _extract_dictionary(g)
    sep_row = dict_r0 + 3

    starts = np.argwhere(g == GREEN)
    assert starts.shape[0] == 1, "expected one green start"
    r, c = map(int, starts[0])
    assert r >= sep_row + 1, "start must be below the separator row"

    if swap_glyphs:
        left_mask, right_mask = right_mask, left_mask
        left_color, right_color = RIGHT_GLYPH_COLOR, LEFT_GLYPH_COLOR
    else:
        left_color, right_color = LEFT_GLYPH_COLOR, RIGHT_GLYPH_COLOR

    dc = 1
    bounce_events: list[dict] = []
    path = [(r, c)]
    collision_pos: tuple[int, int] | None = None
    stop_reason = "top"

    while True:
        nr = r - 1
        nc = c + dc
        bounced_side: str | None = None

        if nc < 0:
            dc = 1
            nc = c + dc
            bounced_side = "left"
        elif nc >= w:
            dc = -1
            nc = c + dc
            bounced_side = "right"

        if bounced_side is not None:
            if enable_stamping:
                if bounced_side == "left":
                    _stamp(g, r, c, left_color, left_mask)
                    stamp_mask = left_mask
                else:
                    _stamp(g, r, c, right_color, right_mask)
                    stamp_mask = right_mask
            else:
                stamp_mask = left_mask if bounced_side == "left" else right_mask
            bounce_events.append(
                {
                    "side": bounced_side,
                    "center": (r, c),
                    "outgoing": (nr, nc),
                    "mask": stamp_mask.copy(),
                }
            )

        if nr < sep_row:
            stop_reason = "top"
            break

        if enable_collision and int(g[nr, nc]) not in (BLACK, GREEN):
            collision_pos = (nr, nc)
            stop_reason = "collision"
            break

        g[nr, nc] = GREEN
        r, c = nr, nc
        path.append((r, c))

    scene = g[sep_row:, :]
    scene[scene == BLACK] = GREY

    meta = {
        "sep_row": sep_row,
        "stop_reason": stop_reason,
        "collision_pos": collision_pos,
        "path": tuple(path),
        "bounce_events": tuple(bounce_events),
        "bounce_count": len(bounce_events),
    }
    return g, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a bouncing diagonal ray, stamp wall-coded glyphs at each reflection, and stop when the ray hits a stamped glyph.

    Input structure:
    - Black background with a top dictionary strip: two 3x3 single-color glyphs separated by a 3-cell vertical grey bar.
    - The left glyph is blue and encodes the stamp used after a left-wall reflection; the right glyph is red and encodes the stamp used after a right-wall reflection.
    - Below the dictionary strip is a one-row black separator, and the scene contains exactly one green start pixel in the lower area.

    Rule:
    - Starting from the green pixel, move one cell up and one cell horizontally each step, initially toward the upper-right.
    - If the next column would leave the grid, reflect horizontally at the current pixel, stamp the wall-specific 3x3 glyph centered on that bounce pixel, then continue upward with the reversed horizontal direction.
    - Before drawing each next step, stop if that destination already contains a non-black, non-green pixel from an earlier stamp; also stop if the next row would enter the dictionary strip.
    - After stopping, recolor every remaining black pixel outside the dictionary strip to grey.

    Key insight:
    - The ray is not governed by geometry alone: each bounce both chooses a glyph from the dictionary and writes new obstacles, so the stamped feedback determines when the otherwise periodic reflection path terminates.
    """
    out, _meta = _simulate(grid)
    return out


def _build_base_grid(
    h: int,
    w: int,
    dict_r0: int,
    dict_left_c0: int,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for rr, cc in np.argwhere(left_mask):
        grid[dict_r0 + int(rr), dict_left_c0 + int(cc)] = LEFT_GLYPH_COLOR
    sep_col = dict_left_c0 + 3
    grid[dict_r0 : dict_r0 + 3, sep_col] = GREY
    for rr, cc in np.argwhere(right_mask):
        grid[dict_r0 + int(rr), sep_col + 1 + int(cc)] = RIGHT_GLYPH_COLOR
    return grid


def _is_valid_example(grid: np.ndarray) -> bool:
    out, meta = _simulate(grid)
    sep_row = int(meta["sep_row"])
    bounce_events = list(meta["bounce_events"])
    if meta["stop_reason"] != "collision":
        return False
    if len(bounce_events) < 2:
        return False
    if {event["side"] for event in bounce_events} != {"left", "right"}:
        return False
    if any(event["center"][0] - 1 < sep_row for event in bounce_events):
        return False
    if len(meta["path"]) < 8:
        return False
    if meta["collision_pos"] is None:
        return False

    collision_r, collision_c = meta["collision_pos"]
    if int(out[collision_r, collision_c]) not in (LEFT_GLYPH_COLOR, RIGHT_GLYPH_COLOR):
        return False

    no_collision_out, no_collision_meta = _simulate(grid, enable_collision=False)
    if no_collision_meta["stop_reason"] != "top":
        return False
    if len(no_collision_meta["path"]) <= len(meta["path"]):
        return False
    if np.array_equal(out, no_collision_out):
        return False

    no_stamp_out, no_stamp_meta = _simulate(grid, enable_stamping=False, enable_collision=False)
    if no_stamp_meta["stop_reason"] != "top":
        return False
    if np.array_equal(out, no_stamp_out):
        return False

    swapped_out, _swapped_meta = _simulate(grid, swap_glyphs=True)
    if np.array_equal(out, swapped_out):
        return False

    scene = out[sep_row:, :]
    if not np.any(scene == LEFT_GLYPH_COLOR):
        return False
    if not np.any(scene == RIGHT_GLYPH_COLOR):
        return False

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(MAX_GENERATION_ATTEMPTS):
        h = int(rng.integers(MIN_GRID_ROWS, MAX_GRID_ROWS + 1))
        w = int(rng.integers(MIN_GRID_COLS, MAX_GRID_COLS + 1))
        dict_r0 = int(rng.integers(0, 2))
        sep_row = dict_r0 + 3
        if h - sep_row < 8:
            continue

        left_mask = LEFT_BLOCKING_GLYPHS[int(rng.integers(len(LEFT_BLOCKING_GLYPHS)))]
        right_mask = RIGHT_SAFE_GLYPHS[int(rng.integers(len(RIGHT_SAFE_GLYPHS)))]
        if np.array_equal(left_mask, right_mask):
            continue

        dict_left_c0 = int(rng.integers(0, w - 6))
        base = _build_base_grid(h, w, dict_r0, dict_left_c0, left_mask, right_mask)

        min_start_col = max(1, sep_row + 2 * w - h)
        max_start_col = w - 2
        if min_start_col > max_start_col:
            continue

        candidate_cols = list(range(min_start_col, max_start_col + 1))
        rng.shuffle(candidate_cols)
        for start_c in candidate_cols:
            min_start_row = max(h // 2, sep_row + 2 * w - start_c - 1)
            if min_start_row > h - 1:
                continue
            candidate_rows = list(range(min_start_row, h))
            rng.shuffle(candidate_rows)
            for start_r in candidate_rows:
                candidate = base.copy()
                candidate[start_r, start_c] = GREEN
                if _is_valid_example(candidate):
                    return candidate

    raise RuntimeError("failed to generate a valid ARC input for puzzle 1")
