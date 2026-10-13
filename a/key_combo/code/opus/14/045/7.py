from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
ORANGE = 7

GEN_ATTEMPTS = 400
PATCH_ATTEMPTS = 200


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_separator(g: np.ndarray, orange_mask: np.ndarray) -> tuple[str, int, tuple[int, int, int, int]]:
    box = _bbox(orange_mask)
    assert box is not None, "expected at least one orange pixel"
    r0, c0, r1, c1 = box

    empty_rows = [r for r in range(r0 + 1, r1) if np.all(g[r, c0 : c1 + 1] == BLACK)]
    empty_cols = [c for c in range(c0 + 1, c1) if np.all(g[r0 : r1 + 1, c] == BLACK)]
    separators = [("row", r) for r in empty_rows] + [("col", c) for c in empty_cols]
    assert len(separators) == 1, "expected exactly one interior empty separator"
    axis, index = separators[0]
    return axis, int(index), box


def _analyze(g: np.ndarray) -> dict[str, object]:
    orange_mask = g == ORANGE
    marker_coords = np.argwhere(g == RED)
    assert marker_coords.shape == (1, 2), "expected exactly one red marker"
    marker = tuple(map(int, marker_coords[0]))

    axis, sep, box = _find_separator(g, orange_mask)
    r0, c0, r1, c1 = box
    mr, mc = marker

    if axis == "row":
        if mr == r0 - 1 and c0 <= mc <= c1 and g[r0, mc] == ORANGE:
            side = "top"
        elif mr == r1 + 1 and c0 <= mc <= c1 and g[r1, mc] == ORANGE:
            side = "bottom"
        else:
            raise AssertionError("row-separator marker must sit just above or below the object")
        source_mask = orange_mask.copy()
        source_mask[sep:, :] = False
        dest_mask = orange_mask.copy()
        dest_mask[: sep + 1, :] = False
        if side == "bottom":
            source_mask, dest_mask = dest_mask, source_mask
    else:
        if mc == c0 - 1 and r0 <= mr <= r1 and g[mr, c0] == ORANGE:
            side = "left"
        elif mc == c1 + 1 and r0 <= mr <= r1 and g[mr, c1] == ORANGE:
            side = "right"
        else:
            raise AssertionError("col-separator marker must sit just left or right of the object")
        source_mask = orange_mask.copy()
        source_mask[:, sep:] = False
        dest_mask = orange_mask.copy()
        dest_mask[:, : sep + 1] = False
        if side == "right":
            source_mask, dest_mask = dest_mask, source_mask

    return {
        "axis": axis,
        "sep": sep,
        "box": box,
        "marker": marker,
        "side": side,
        "orange_mask": orange_mask,
        "source_mask": source_mask,
        "dest_mask": dest_mask,
    }


def _mirror_coords(coords: np.ndarray, axis: str, sep: int) -> np.ndarray:
    mirrored = coords.copy()
    if axis == "row":
        mirrored[:, 0] = 2 * sep - mirrored[:, 0]
    else:
        mirrored[:, 1] = 2 * sep - mirrored[:, 1]
    return mirrored


def _transform(g: np.ndarray, override_side: str | None = None) -> np.ndarray:
    info = _analyze(g)
    axis = info["axis"]
    sep = int(info["sep"])
    marker_r, marker_c = info["marker"]
    orange_mask = np.asarray(info["orange_mask"], dtype=bool)
    source_side = str(info["side"]) if override_side is None else override_side

    if axis == "row":
        valid_sides = {"top", "bottom"}
        source_mask = orange_mask & (np.arange(g.shape[0])[:, None] < sep)
        if source_side == "bottom":
            source_mask = orange_mask & (np.arange(g.shape[0])[:, None] > sep)
    else:
        valid_sides = {"left", "right"}
        source_mask = orange_mask & (np.arange(g.shape[1])[None, :] < sep)
        if source_side == "right":
            source_mask = orange_mask & (np.arange(g.shape[1])[None, :] > sep)
    assert source_side in valid_sides

    out = g.copy()
    out[marker_r, marker_c] = BLACK
    src_coords = np.argwhere(source_mask)
    dst_coords = _mirror_coords(src_coords, axis, sep)
    for r, c in dst_coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
            out[rr, cc] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the unique blank interior separator as a mirror axis, mirror the marker-side half, and recolor the copy green.

    Input structure:
    - A black background contains orange foreground pixels inside one tight bounding box.
    - Strictly inside that orange bbox there is exactly one all-black row or one all-black column.
    - A single red marker pixel sits outside the orange bbox and touches its top/bottom side for a row separator or left/right side for a column separator.

    Rule:
    - Find the orange bbox and detect its unique interior blank row or column.
    - Treat that blank line as the mirror axis.
    - Determine which side of the axis is selected by the red marker.
    - Reflect the orange pixels on that selected side across the axis, paint only the reflected copy green, keep the original orange pixels, keep the separator black, and erase the red marker.

    Key insight:
    - The empty line has two roles at once: it is both the structural split inside the orange shape and the geometric axis of reflection, while the red marker chooses which half is the source rather than where the axis is.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _transform(g)


def _build_interval_patch(rows: int, cols: int, mode: str, forbid_flip: str | None = None) -> np.ndarray:
    assert mode in {"colwise", "rowwise"}
    for _ in range(PATCH_ATTEMPTS):
        patch = np.zeros((rows, cols), dtype=bool)
        prev: tuple[int, int] | None = None

        if mode == "colwise":
            for c in range(cols):
                candidates = []
                for a in range(rows):
                    for b in range(a, rows):
                        if prev is None or max(a, prev[0]) <= min(b, prev[1]):
                            candidates.append((a, b))
                a, b = random.choice(candidates)
                patch[a : b + 1, c] = True
                prev = (a, b)
        else:
            for r in range(rows):
                candidates = []
                for a in range(cols):
                    for b in range(a, cols):
                        if prev is None or max(a, prev[0]) <= min(b, prev[1]):
                            candidates.append((a, b))
                a, b = random.choice(candidates)
                patch[r, a : b + 1] = True
                prev = (a, b)

        row_counts = patch.sum(axis=1)
        col_counts = patch.sum(axis=0)
        area = int(patch.sum())
        max_area = max(4, int(np.floor(rows * cols * 0.8)))
        if np.any(row_counts == 0) or np.any(col_counts == 0):
            continue
        if area < 4 or area > max_area or area == rows * cols:
            continue
        if len(set(map(int, row_counts))) == 1 and len(set(map(int, col_counts))) == 1:
            continue
        if forbid_flip == "ud" and np.array_equal(patch, np.flipud(patch)):
            continue
        if forbid_flip == "lr" and np.array_equal(patch, np.fliplr(patch)):
            continue
        return patch
    raise RuntimeError("failed to build irregular patch")


def _compose_row_case(source_side: str) -> np.ndarray:
    assert source_side in {"top", "bottom"}
    width = random.randint(6, 10)
    source_h = random.randint(2, 4)
    target_h = random.randint(source_h, 5)
    if source_side == "top":
        top_h, bottom_h = source_h, target_h
    else:
        top_h, bottom_h = target_h, source_h

    top = _build_interval_patch(top_h, width, "colwise", forbid_flip="ud" if source_side == "top" else None)
    bottom = _build_interval_patch(bottom_h, width, "colwise", forbid_flip="ud" if source_side == "bottom" else None)

    local = np.zeros((top_h + 1 + bottom_h, width), dtype=int)
    local[:top_h, :] = top.astype(int) * ORANGE
    local[top_h + 1 :, :] = bottom.astype(int) * ORANGE
    return local


def _compose_col_case(source_side: str) -> np.ndarray:
    assert source_side in {"left", "right"}
    height = random.randint(6, 10)
    source_w = random.randint(2, 4)
    target_w = random.randint(source_w, 5)
    if source_side == "left":
        left_w, right_w = source_w, target_w
    else:
        left_w, right_w = target_w, source_w

    left = _build_interval_patch(height, left_w, "rowwise", forbid_flip="lr" if source_side == "left" else None)
    right = _build_interval_patch(height, right_w, "rowwise", forbid_flip="lr" if source_side == "right" else None)

    local = np.zeros((height, left_w + 1 + right_w), dtype=int)
    local[:, :left_w] = left.astype(int) * ORANGE
    local[:, left_w + 1 :] = right.astype(int) * ORANGE
    return local


def _place_marker(grid: np.ndarray, local: np.ndarray, axis: str, source_side: str, top: int, left: int) -> None:
    if axis == "row":
        sep = next(r for r in range(1, local.shape[0] - 1) if np.all(local[r, :] == BLACK))
        if source_side == "top":
            cols = [c for c in range(local.shape[1]) if local[0, c] == ORANGE]
            grid[top - 1, left + random.choice(cols)] = RED
        else:
            cols = [c for c in range(local.shape[1]) if local[-1, c] == ORANGE]
            grid[top + local.shape[0], left + random.choice(cols)] = RED
    else:
        sep = next(c for c in range(1, local.shape[1] - 1) if np.all(local[:, c] == BLACK))
        if source_side == "left":
            rows = [r for r in range(local.shape[0]) if local[r, 0] == ORANGE]
            grid[top + random.choice(rows), left - 1] = RED
        else:
            rows = [r for r in range(local.shape[0]) if local[r, -1] == ORANGE]
            grid[top + random.choice(rows), left + local.shape[1]] = RED


def _opposite_side(side: str) -> str:
    return {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[side]


def _reflected_dest_mask(info: dict[str, object], source_side: str) -> np.ndarray:
    orange_mask = np.asarray(info["orange_mask"], dtype=bool)
    axis = str(info["axis"])
    sep = int(info["sep"])
    if axis == "row":
        source_mask = orange_mask & (np.arange(orange_mask.shape[0])[:, None] < sep)
        if source_side == "bottom":
            source_mask = orange_mask & (np.arange(orange_mask.shape[0])[:, None] > sep)
    else:
        source_mask = orange_mask & (np.arange(orange_mask.shape[1])[None, :] < sep)
        if source_side == "right":
            source_mask = orange_mask & (np.arange(orange_mask.shape[1])[None, :] > sep)

    coords = np.argwhere(source_mask)
    mirrored = _mirror_coords(coords, axis, sep)
    out = np.zeros_like(orange_mask, dtype=bool)
    for r, c in mirrored:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
            out[rr, cc] = True
    return out


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        info = _analyze(inp)
        out = main(inp)
        alt = _transform(inp, override_side=_opposite_side(str(info["side"])))
    except Exception:
        return False

    axis = str(info["axis"])
    side = str(info["side"])
    source_mask = np.asarray(info["source_mask"], dtype=bool)
    dest_mask = np.asarray(info["dest_mask"], dtype=bool)

    if int(source_mask.sum()) < 4 or int(dest_mask.sum()) < 4:
        return False
    if np.any(out == RED):
        return False
    if np.array_equal(out, inp) or np.array_equal(out, alt):
        return False

    reflected = _reflected_dest_mask(info, side)
    if not np.any(reflected & ~dest_mask):
        return False

    r0, c0, r1, c1 = info["box"]
    sep = int(info["sep"])
    if axis == "row":
        if not np.all(inp[sep, c0 : c1 + 1] == BLACK):
            return False
    else:
        if not np.all(inp[r0 : r1 + 1, sep] == BLACK):
            return False

    green_count = int(np.count_nonzero(out == GREEN))
    orange_count = int(np.count_nonzero(out == ORANGE))
    if green_count < 4 or orange_count < 4:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    axis = random.choice(["row", "col"])
    if axis == "row":
        side = random.choice(["top", "bottom"])
        local = _compose_row_case(side)
    else:
        side = random.choice(["left", "right"])
        local = _compose_col_case(side)

    min_h = max(12, local.shape[0] + 2)
    min_w = max(12, local.shape[1] + 2)
    if min_h > 18 or min_w > 18:
        return None
    height = random.randint(min_h, 18)
    width = random.randint(min_w, 18)
    grid = np.zeros((height, width), dtype=int)

    row_lo = 1
    row_hi = height - local.shape[0] - 1
    col_lo = 1
    col_hi = width - local.shape[1] - 1
    if row_lo > row_hi or col_lo > col_hi:
        return None
    top = random.randint(row_lo, row_hi)
    left = random.randint(col_lo, col_hi)

    grid[top : top + local.shape[0], left : left + local.shape[1]] = local
    _place_marker(grid, local, axis, side, top, left)

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
