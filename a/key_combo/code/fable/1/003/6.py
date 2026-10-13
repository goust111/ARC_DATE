from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg


GRID_H_RANGE = (17, 29)
GRID_W_RANGE = (17, 29)
RECT_COUNT_CHOICES = (3, 4, 4, 5)
RECT_H_MAX = 9
RECT_W_MAX = 11
PLACEMENT_TRIES = 220
GEN_ATTEMPTS = 800
PALETTE = tuple(range(10))


def _extract_rectangles(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    rects: list[dict] = []
    seen_colors: set[int] = set()
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        assert color not in seen_colors, "each rectangle must have a unique color"

        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        assert h >= 3 and w >= 3 and h % 2 == 1 and w % 2 == 1, "rectangles must have odd height and width >= 3"

        patch = crop_bbox(g, (r0, c0, r1, c1))
        expected = np.full((h, w), int(bg), dtype=int)
        expected[0, :] = int(color)
        expected[-1, :] = int(color)
        expected[:, 0] = int(color)
        expected[:, -1] = int(color)
        assert np.array_equal(patch, expected), "every foreground component must be a one-cell-thick hollow rectangle"

        rects.append(
            {
                "color": int(color),
                "bbox": (r0, c0, r1, c1),
                "center_r": int(r0 + h // 2),
                "center_c": int(c0 + w // 2),
            }
        )
        seen_colors.add(int(color))

    assert len(rects) >= 2, "expected multiple rectangles"
    assert len({rect["center_r"] for rect in rects}) == len(rects), "center rows must be unique"
    assert len({rect["center_c"] for rect in rects}) == len(rects), "center columns must be unique"
    rects.sort(key=lambda rect: (int(rect["center_r"]), int(rect["center_c"])))
    return bg, rects


def _bbox_cover_mask(shape: tuple[int, int], rects: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = map(int, rect["bbox"])
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _intersection_records(rects: list[dict]) -> list[dict]:
    records: list[dict] = []
    for row_idx, row_rect in enumerate(rects):
        for col_idx, col_rect in enumerate(rects):
            if row_idx == col_idx:
                continue
            r = int(row_rect["center_r"])
            c = int(col_rect["center_c"])

            blockers: list[int] = []
            blocked_interior = False
            for block_idx, block_rect in enumerate(rects):
                r0, c0, r1, c1 = map(int, block_rect["bbox"])
                if r0 <= r <= r1 and c0 <= c <= c1:
                    blockers.append(int(block_idx))
                    if r0 < r < r1 and c0 < c < c1:
                        blocked_interior = True

            records.append(
                {
                    "row_idx": int(row_idx),
                    "col_idx": int(col_idx),
                    "coord": (r, c),
                    "row_color": int(row_rect["color"]),
                    "col_color": int(col_rect["color"]),
                    "visible": len(blockers) == 0,
                    "blockers": tuple(blockers),
                    "blocked_interior": bool(blocked_interior),
                }
            )
    return records


def _render_intersections(
    grid: np.ndarray,
    rects: list[dict],
    *,
    color_source: str,
    block_mode: str,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    if block_mode == "bbox":
        blocked_mask = _bbox_cover_mask(g.shape, rects)
    elif block_mode == "foreground":
        blocked_mask = g != int(bg)
    else:
        raise ValueError("unsupported block mode")

    for row_rect in rects:
        for col_rect in rects:
            if row_rect is col_rect:
                continue
            r = int(row_rect["center_r"])
            c = int(col_rect["center_c"])
            if blocked_mask[r, c]:
                continue
            if color_source == "row":
                out[r, c] = int(row_rect["color"])
            elif color_source == "col":
                out[r, c] = int(col_rect["color"])
            else:
                raise ValueError("unsupported color source")
    return out


def _draw_hollow_rectangle(grid: np.ndarray, rect: dict) -> None:
    r0, c0, r1, c1 = map(int, rect["bbox"])
    color = int(rect["color"])
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _sample_rectangle(
    rng: np.random.Generator,
    shape: tuple[int, int],
    color: int,
    occupied: np.ndarray,
    used_center_rows: set[int],
    used_center_cols: set[int],
) -> dict | None:
    h_lim, w_lim = map(int, shape)
    h_choices = [size for size in range(3, min(RECT_H_MAX, h_lim) + 1, 2)]
    w_choices = [size for size in range(3, min(RECT_W_MAX, w_lim) + 1, 2)]
    if not h_choices or not w_choices:
        return None

    for _ in range(PLACEMENT_TRIES):
        h = int(rng.choice(h_choices))
        w = int(rng.choice(w_choices))
        r0 = int(rng.integers(0, h_lim - h + 1))
        c0 = int(rng.integers(0, w_lim - w + 1))
        r1 = int(r0 + h - 1)
        c1 = int(c0 + w - 1)
        center_r = int(r0 + h // 2)
        center_c = int(c0 + w // 2)
        if center_r in used_center_rows or center_c in used_center_cols:
            continue

        rr0 = max(0, r0 - 1)
        rr1 = min(h_lim, r1 + 2)
        cc0 = max(0, c0 - 1)
        cc1 = min(w_lim, c1 + 2)
        if bool(occupied[rr0:rr1, cc0:cc1].any()):
            continue

        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        used_center_rows.add(center_r)
        used_center_cols.add(center_c)
        return {
            "color": int(color),
            "bbox": (r0, c0, r1, c1),
            "center_r": center_r,
            "center_c": center_c,
        }
    return None


def _random_rectangles(
    rng: np.random.Generator,
    shape: tuple[int, int],
    bg: int,
    count: int,
) -> list[dict] | None:
    colors = [int(color) for color in PALETTE if int(color) != int(bg)]
    if len(colors) < count:
        return None
    rng.shuffle(colors)

    occupied = np.zeros(shape, dtype=bool)
    used_center_rows: set[int] = set()
    used_center_cols: set[int] = set()
    rects: list[dict] = []
    for color in colors[:count]:
        rect = _sample_rectangle(rng, shape, int(color), occupied, used_center_rows, used_center_cols)
        if rect is None:
            return None
        rects.append(rect)
    rects.sort(key=lambda rect: (int(rect["center_r"]), int(rect["center_c"])))
    return rects


def _semantic_case_ok(grid: np.ndarray, rects: list[dict]) -> bool:
    correct = _render_intersections(grid, rects, color_source="row", block_mode="bbox")
    swapped = _render_intersections(grid, rects, color_source="col", block_mode="bbox")
    foreground_only = _render_intersections(grid, rects, color_source="row", block_mode="foreground")
    records = _intersection_records(rects)

    visible = [record for record in records if bool(record["visible"])]
    blocked = [record for record in records if not bool(record["visible"])]
    if len(visible) < 3:
        return False
    if not blocked:
        return False
    if not any(bool(record["blocked_interior"]) for record in blocked):
        return False

    row_counts = Counter(int(record["row_idx"]) for record in visible)
    if len(row_counts) < 2:
        return False
    if max(row_counts.values()) < 2:
        return False
    if len({int(record["row_color"]) for record in visible}) < 2:
        return False

    if np.array_equal(correct, grid):
        return False
    if np.array_equal(correct, swapped):
        return False
    if np.array_equal(correct, foreground_only):
        return False

    try:
        produced = main(grid)
    except AssertionError:
        return False
    return np.array_equal(produced, correct)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_rect_hint = int(rng.choice(RECT_COUNT_CHOICES))
    h_low = 21 if n_rect_hint == 5 else GRID_H_RANGE[0]
    w_low = 21 if n_rect_hint == 5 else GRID_W_RANGE[0]
    h = int(rng.integers(h_low, GRID_H_RANGE[1] + 1))
    w = int(rng.integers(w_low, GRID_W_RANGE[1] + 1))
    bg = int(rng.choice(PALETTE))

    rects = _random_rectangles(rng, (h, w), bg, n_rect_hint)
    if rects is None:
        return None

    grid = np.full((h, w), int(bg), dtype=int)
    for rect in rects:
        _draw_hollow_rectangle(grid, rect)

    if not _semantic_case_ok(grid, rects):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark the background intersections of different hollow rectangles' invisible center row and center column, using the row rectangle's color.

    Input structure:
    - The grid has a uniform background and several one-cell-thick hollow rectangles in distinct colors.
    - Every rectangle has odd height and odd width, so each one has a unique center row and center column.
    - Different rectangles do not touch, and their center rows are all different from one another, as are their center columns.

    Rule:
    - Treat each rectangle's center row and center column as full invisible lines spanning the whole grid.
    - For every ordered pair of different rectangles A and B, look at the cell where A's center row meets B's center column.
    - If that intersection lies outside every rectangle's full bbox, paint that background cell with A's color; otherwise leave it unchanged.
    - Keep every original rectangle exactly as it was.

    Key insight:
    - No lines are ever drawn directly: the only visible output evidence is isolated intersection pixels, and their color is determined by the rectangle that supplies the row, not the column.
    """
    g = np.asarray(grid, dtype=int)
    bg, rects = _extract_rectangles(g)
    _ = bg
    return _render_intersections(g, rects, color_source="row", block_mode="bbox")


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
