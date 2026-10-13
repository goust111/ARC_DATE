from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
MARKER = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 300
_RNG = random.Random()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            queue = deque([(int(r), int(c))])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(pts)
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    return int(mask.sum()) > 0 and len(_components(mask)) == 1


def _has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    empty = ~mask
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not bool(empty[r, c]) or bool(seen[r, c]):
                continue
            queue = deque([(int(r), int(c))])
            seen[r, c] = True
            touches_edge = False
            while queue:
                rr, cc = queue.popleft()
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(empty[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            if not touches_edge:
                return True
    return False


def _valid_shape(mask: np.ndarray, marker: tuple[int, int]) -> bool:
    mr, mc = marker
    if not bool(mask[mr, mc]):
        return False
    if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
        return False
    if not _is_connected(mask):
        return False
    if _has_hole(mask):
        return False
    fill_only = mask.copy()
    fill_only[mr, mc] = False
    if int(fill_only.sum()) <= 0:
        return False
    return _is_connected(fill_only)


def _make_shape(marker_side: str, rng: random.Random) -> tuple[np.ndarray, tuple[int, int]]:
    for _ in range(200):
        if marker_side in ("left", "right"):
            height = rng.randint(3, 5)
            width = rng.randint(2, 3)
            marker = (rng.randint(1, height - 2), 0 if marker_side == "left" else width - 1)
        else:
            height = rng.randint(2, 3)
            width = rng.randint(3, 5)
            marker = (0 if marker_side == "up" else height - 1, rng.randint(1, width - 2))

        total_cells = rng.randint(4, min(7, height * width))
        mask = np.ones((height, width), dtype=bool)
        while int(mask.sum()) > total_cells:
            candidates = [(r, c) for r, c in np.argwhere(mask) if (int(r), int(c)) != marker]
            rng.shuffle(candidates)
            removed = False
            for r, c in candidates:
                trial = mask.copy()
                trial[r, c] = False
                if _valid_shape(trial, marker):
                    mask = trial
                    removed = True
                    break
            if not removed:
                break

        if int(mask.sum()) == total_cells and _valid_shape(mask, marker):
            return mask, marker
    raise RuntimeError(f"failed to build shape for side {marker_side}")


def _find_wall(grid: np.ndarray) -> tuple[str, int]:
    h, w = grid.shape
    full_rows = [r for r in range(h) if np.all(grid[r, :] == WALL)]
    full_cols = [c for c in range(w) if np.all(grid[:, c] == WALL)]
    assert (len(full_rows), len(full_cols)) in ((1, 0), (0, 1)), "expected exactly one full grey wall"
    if full_rows:
        return "horizontal", int(full_rows[0])
    return "vertical", int(full_cols[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each marked shape onto the grey wall and draw a same-color shadow strip on the opposite side.

    Input structure:
    - The background is Black.
    - One full Grey line spans the grid horizontally or vertically and acts as a mirror wall.
    - Several non-Black, non-Grey solid shapes appear on both sides of the wall; each shape is 4-connected and contains exactly one single-cell Maroon marker inside it.

    Rule:
    - Keep every original shape, the Maroon markers, and the Grey wall unchanged.
    - For each shape, use the Maroon marker's extreme position within the whole shape bbox to determine the projection direction: leftmost means left, rightmost means right, topmost means up, bottommost means down.
    - Count the shape's area using only its own color pixels, excluding the Maroon marker. Project the shape orthogonally onto the wall, keep the covered wall pixels Grey, then extend a solid same-color rectangular strip from the wall onto the opposite side of the wall with width equal to the projection span and length equal to that area.

    Key insight:
    - The Maroon pixel is a geometric direction code, while the number of non-Maroon shape pixels becomes the shadow length drawn beyond the wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    orientation, wall_pos = _find_wall(g)
    object_mask = (g != BG) & (g != WALL)
    comps = _components(object_mask)
    assert comps, "expected at least one marked shape"

    out = g.copy()
    h, w = g.shape

    for comp in comps:
        pts = np.array(comp, dtype=int)
        values = g[pts[:, 0], pts[:, 1]]
        marker_pts = pts[values == MARKER]
        fill_colors = {int(v) for v in values if int(v) != MARKER}
        assert marker_pts.shape == (1, 2), "each shape must contain exactly one Maroon marker"
        assert len(fill_colors) == 1, "each component must have exactly one non-marker color"
        color = fill_colors.pop()
        area = int(np.count_nonzero(values == color))
        assert area > 0, "shape area must exclude but not equal the marker"

        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        mr, mc = map(int, marker_pts[0])

        if orientation == "vertical":
            if mc == c0:
                col_start = wall_pos - area
                col_stop = wall_pos
            elif mc == c1:
                col_start = wall_pos + 1
                col_stop = wall_pos + 1 + area
            else:
                raise AssertionError("vertical-wall shapes must point left or right")

            assert (mc == c0 and c0 > wall_pos) or (mc == c1 and c1 < wall_pos), "direction must point toward the wall"
            assert 0 <= col_start < col_stop <= w, "shadow must fit horizontally"
            target = (slice(r0, r1 + 1), slice(col_start, col_stop))
        else:
            if mr == r0:
                row_start = wall_pos - area
                row_stop = wall_pos
            elif mr == r1:
                row_start = wall_pos + 1
                row_stop = wall_pos + 1 + area
            else:
                raise AssertionError("horizontal-wall shapes must point up or down")

            assert (mr == r0 and r0 > wall_pos) or (mr == r1 and r1 < wall_pos), "direction must point toward the wall"
            assert 0 <= row_start < row_stop <= h, "shadow must fit vertically"
            target = (slice(row_start, row_stop), slice(c0, c1 + 1))

        assert np.all(g[target] == BG), "shadow area must be empty in the input"
        assert np.all((out[target] == BG) | (out[target] == color)), "shadow regions must not conflict"
        out[target] = color

    return out


def _pack_spans(spans: list[int], rng: random.Random) -> tuple[dict[int, int], int]:
    order = list(range(len(spans)))
    rng.shuffle(order)
    starts: dict[int, int] = {}
    pos = 0
    for i, idx in enumerate(order):
        if i > 0:
            pos += rng.randint(1, 2)
        starts[idx] = pos
        pos += spans[idx]
    return starts, pos


def _paint_object(
    grid: np.ndarray,
    top: int,
    left: int,
    mask: np.ndarray,
    marker: tuple[int, int],
    color: int,
) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    assert np.all(patch[mask] == BG), "objects must not overlap"
    patch[mask] = color
    mr, mc = marker
    grid[top + mr, left + mc] = MARKER


def _build_vertical(grid_rng: random.Random, left_count: int, right_count: int) -> np.ndarray:
    left_objs = []
    right_objs = []
    for _ in range(left_count):
        mask, marker = _make_shape("right", grid_rng)
        left_objs.append({"mask": mask, "marker": marker, "area": int(mask.sum()) - 1})
    for _ in range(right_count):
        mask, marker = _make_shape("left", grid_rng)
        right_objs.append({"mask": mask, "marker": marker, "area": int(mask.sum()) - 1})

    colors = grid_rng.sample(PALETTE, left_count + right_count)
    for obj, color in zip(left_objs + right_objs, colors):
        obj["color"] = int(color)

    left_spans = [int(obj["mask"].shape[0]) for obj in left_objs]
    right_spans = [int(obj["mask"].shape[0]) for obj in right_objs]
    left_starts, left_used = _pack_spans(left_spans, grid_rng)
    right_starts, right_used = _pack_spans(right_spans, grid_rng)

    height = max(left_used, right_used) + grid_rng.randint(3, 6)
    left_offset = grid_rng.randint(1, height - left_used - 1)
    right_offset = grid_rng.randint(1, height - right_used - 1)

    left_shadow_depth = max(int(obj["area"]) for obj in right_objs)
    right_shadow_depth = max(int(obj["area"]) for obj in left_objs)
    left_zone_width = max(int(obj["mask"].shape[1]) for obj in left_objs) + grid_rng.randint(1, 3)
    right_zone_width = max(int(obj["mask"].shape[1]) for obj in right_objs) + grid_rng.randint(1, 3)
    left_margin = grid_rng.randint(1, 2)
    right_margin = grid_rng.randint(1, 2)

    wall_col = left_margin + left_zone_width + left_shadow_depth
    width = left_margin + left_zone_width + left_shadow_depth + 1 + right_shadow_depth + right_zone_width + right_margin
    assert height <= 30 and width <= 30

    left_zone_start = left_margin
    left_zone_end = left_margin + left_zone_width - 1
    right_zone_start = wall_col + 1 + right_shadow_depth
    right_zone_end = right_zone_start + right_zone_width - 1

    grid = np.full((height, width), BG, dtype=int)
    grid[:, wall_col] = WALL

    for idx, obj in enumerate(left_objs):
        mask = obj["mask"]
        top = left_offset + left_starts[idx]
        left = grid_rng.randint(left_zone_start, left_zone_end - mask.shape[1] + 1)
        _paint_object(grid, top, left, mask, obj["marker"], int(obj["color"]))
    for idx, obj in enumerate(right_objs):
        mask = obj["mask"]
        top = right_offset + right_starts[idx]
        left = grid_rng.randint(right_zone_start, right_zone_end - mask.shape[1] + 1)
        _paint_object(grid, top, left, mask, obj["marker"], int(obj["color"]))

    return grid


def _build_horizontal(grid_rng: random.Random, top_count: int, bottom_count: int) -> np.ndarray:
    top_objs = []
    bottom_objs = []
    for _ in range(top_count):
        mask, marker = _make_shape("down", grid_rng)
        top_objs.append({"mask": mask, "marker": marker, "area": int(mask.sum()) - 1})
    for _ in range(bottom_count):
        mask, marker = _make_shape("up", grid_rng)
        bottom_objs.append({"mask": mask, "marker": marker, "area": int(mask.sum()) - 1})

    colors = grid_rng.sample(PALETTE, top_count + bottom_count)
    for obj, color in zip(top_objs + bottom_objs, colors):
        obj["color"] = int(color)

    top_spans = [int(obj["mask"].shape[1]) for obj in top_objs]
    bottom_spans = [int(obj["mask"].shape[1]) for obj in bottom_objs]
    top_starts, top_used = _pack_spans(top_spans, grid_rng)
    bottom_starts, bottom_used = _pack_spans(bottom_spans, grid_rng)

    width = max(top_used, bottom_used) + grid_rng.randint(3, 6)
    top_offset = grid_rng.randint(1, width - top_used - 1)
    bottom_offset = grid_rng.randint(1, width - bottom_used - 1)

    top_shadow_depth = max(int(obj["area"]) for obj in bottom_objs)
    bottom_shadow_depth = max(int(obj["area"]) for obj in top_objs)
    top_zone_height = max(int(obj["mask"].shape[0]) for obj in top_objs) + grid_rng.randint(1, 3)
    bottom_zone_height = max(int(obj["mask"].shape[0]) for obj in bottom_objs) + grid_rng.randint(1, 3)
    top_margin = grid_rng.randint(1, 2)
    bottom_margin = grid_rng.randint(1, 2)

    wall_row = top_margin + top_zone_height + top_shadow_depth
    height = top_margin + top_zone_height + top_shadow_depth + 1 + bottom_shadow_depth + bottom_zone_height + bottom_margin
    assert height <= 30 and width <= 30

    top_zone_start = top_margin
    top_zone_end = top_margin + top_zone_height - 1
    bottom_zone_start = wall_row + 1 + bottom_shadow_depth
    bottom_zone_end = bottom_zone_start + bottom_zone_height - 1

    grid = np.full((height, width), BG, dtype=int)
    grid[wall_row, :] = WALL

    for idx, obj in enumerate(top_objs):
        mask = obj["mask"]
        left = top_offset + top_starts[idx]
        top = grid_rng.randint(top_zone_start, top_zone_end - mask.shape[0] + 1)
        _paint_object(grid, top, left, mask, obj["marker"], int(obj["color"]))
    for idx, obj in enumerate(bottom_objs):
        mask = obj["mask"]
        left = bottom_offset + bottom_starts[idx]
        top = grid_rng.randint(bottom_zone_start, bottom_zone_end - mask.shape[0] + 1)
        _paint_object(grid, top, left, mask, obj["marker"], int(obj["color"]))

    return grid


def _gen_attempt() -> np.ndarray | None:
    orientation = _RNG.choice(("vertical", "horizontal"))
    try:
        if orientation == "vertical":
            grid = _build_vertical(_RNG, left_count=_RNG.randint(1, 2), right_count=_RNG.randint(1, 2))
        else:
            grid = _build_horizontal(_RNG, top_count=_RNG.randint(1, 2), bottom_count=_RNG.randint(1, 2))
        out = main(grid)
        assert not np.array_equal(grid, out), "output must add at least one shadow"
        return grid
    except (AssertionError, RuntimeError):
        return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
