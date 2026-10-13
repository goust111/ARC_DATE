from __future__ import annotations

import bisect
import random
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label


BG = 0
BLUE = 1
RED = 2
GREY = 5
ORANGE = 7

STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


@dataclass(frozen=True)
class Obj:
    color: int
    bbox: tuple[int, int, int, int]
    rel_rows: np.ndarray
    cols: np.ndarray
    height: int
    top_by_col: dict[int, int]
    bottom_by_col: dict[int, int]


def _swap_color(color: int) -> int:
    return BLUE if int(color) == RED else RED


def _detect_separator_rows(grid: np.ndarray) -> tuple[list[int], dict[int, int]]:
    rows: list[int] = []
    colors: dict[int, int] = {}
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(row == GREY) or np.all(row == ORANGE):
            rows.append(int(r))
            colors[int(r)] = int(row[0])
    return rows, colors


def _detect_objects(grid: np.ndarray) -> list[Obj]:
    comps: list[Obj] = []
    for color in (BLUE, RED):
        lab, n = label(grid == color, structure=STRUCT4)
        for cid in range(1, int(n) + 1):
            coords = np.argwhere(lab == cid)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            rel_rows = coords[:, 0] - int(r0)
            cols = coords[:, 1].astype(int)
            top_by_col: dict[int, int] = {}
            bottom_by_col: dict[int, int] = {}
            for rr, cc in zip(rel_rows.tolist(), cols.tolist()):
                top_by_col[cc] = rr if cc not in top_by_col else min(top_by_col[cc], rr)
                bottom_by_col[cc] = rr if cc not in bottom_by_col else max(bottom_by_col[cc], rr)
            comps.append(
                Obj(
                    color=int(color),
                    bbox=(int(r0), int(c0), int(r1), int(c1)),
                    rel_rows=rel_rows.astype(int),
                    cols=cols.astype(int),
                    height=int(r1 - r0 + 1),
                    top_by_col=top_by_col,
                    bottom_by_col=bottom_by_col,
                )
            )
    comps.sort(key=lambda obj: (obj.bbox[0], obj.bbox[1], obj.color))
    return comps


def _paint_object(out: np.ndarray, obj: Obj, top: int, color: int) -> None:
    rows = int(top) + obj.rel_rows
    out[rows, obj.cols] = int(color)


def _solve_object_fast(
    obj: Obj,
    separator_rows: list[int],
    separator_colors: dict[int, int],
    obstacle: np.ndarray,
    height: int,
) -> tuple[int, int]:
    top = int(obj.bbox[0])
    color = int(obj.color)
    seen: set[tuple[int, int]] = set()

    while True:
        state = (top, color)
        if state in seen:
            raise ValueError("orange separators create a cycle for this object")
        seen.add(state)

        if color == RED:
            sep_idx = bisect.bisect_left(separator_rows, top) - 1
            barrier_row = -1 if sep_idx < 0 else int(separator_rows[sep_idx])
            barrier_color = GREY if barrier_row == -1 else int(separator_colors[barrier_row])
            best_shift = int(top - (barrier_row + 1))
            hit_orange = barrier_color == ORANGE

            for col, rel_top in obj.top_by_col.items():
                front = int(top + rel_top)
                rows_above = np.flatnonzero(obstacle[:front, int(col)])
                if rows_above.size == 0:
                    continue
                shift = int(front - int(rows_above[-1]) - 1)
                if shift <= best_shift:
                    best_shift = shift
                    hit_orange = False

            top -= best_shift
            if hit_orange:
                color = BLUE
                continue
            return int(top), int(color)

        bottom = int(top + obj.height - 1)
        sep_idx = bisect.bisect_right(separator_rows, bottom)
        barrier_row = height if sep_idx >= len(separator_rows) else int(separator_rows[sep_idx])
        barrier_color = GREY if barrier_row == height else int(separator_colors[barrier_row])
        best_shift = int((barrier_row - 1) - bottom)
        hit_orange = barrier_color == ORANGE

        for col, rel_bottom in obj.bottom_by_col.items():
            front = int(top + rel_bottom)
            rows_below = np.flatnonzero(obstacle[int(front + 1) :, int(col)])
            if rows_below.size == 0:
                continue
            shift = int(rows_below[0])
            if shift <= best_shift:
                best_shift = shift
                hit_orange = False

        top += best_shift
        if hit_orange:
            color = RED
            continue
        return int(top), int(color)


def _semantic_main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    objects = _detect_objects(g)
    line_rows, line_colors = _detect_separator_rows(g)
    line_by_row = np.zeros(h, dtype=int)
    for row in line_rows:
        line_by_row[int(row)] = int(line_colors[int(row)])

    out[np.isin(out, (BLUE, RED))] = BG
    obstacle = np.zeros((h, w), dtype=bool)

    for obj in objects:
        rows = obj.rel_rows + int(obj.bbox[0])
        cols = obj.cols.copy()
        color = int(obj.color)
        seen: set[tuple[int, int]] = set()

        while True:
            top = int(rows.min())
            state = (top, color)
            if state in seen:
                raise ValueError("cycle detected in semantic simulator")
            seen.add(state)

            dr = -1 if color == RED else 1
            next_rows = rows + int(dr)
            if int(next_rows.min()) < 0 or int(next_rows.max()) >= h:
                break
            if obstacle[next_rows, cols].any():
                break

            touched = {int(r) for r in np.unique(next_rows) if int(line_by_row[int(r)]) != 0}
            if touched:
                hit_row = min(touched) if dr < 0 else max(touched)
                hit_color = int(line_by_row[hit_row])
                if hit_color == ORANGE:
                    color = _swap_color(color)
                    continue
                break

            rows = next_rows

        out[rows, cols] = int(color)
        obstacle[rows, cols] = True

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each red/blue object vertically between full-width separators, bouncing only on orange lines.

    Input structure:
    - The background is black, with 4-6 full-width horizontal separator rows colored grey or orange.
    - The top and bottom grid edges act as implicit grey boundaries even though they are not drawn.
    - Interior red and blue 4-connected objects lie strictly between separators and keep their shape and columns.

    Rule:
    - Red objects move upward; blue objects move downward.
    - An object slides as a rigid body until its leading side is adjacent to a separator row, the grid boundary, or a previously stopped object.
    - Grey separators, grid boundaries, and previously stopped objects stop the motion in place.
    - Orange separators instead flip the object's color between red and blue and immediately reverse its vertical direction; separator rows themselves remain fixed.
    - Objects are processed from top to bottom by their initial position, and their original cells become black.

    Key insight:
    - Separator color controls the interaction: grey is an absorbing wall, while orange is a color-flipping mirror whose bounce changes the object's next direction.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    separator_rows, separator_colors = _detect_separator_rows(g)
    objects = _detect_objects(g)

    out[np.isin(out, (BLUE, RED))] = BG
    obstacle = np.zeros((h, w), dtype=bool)

    for obj in objects:
        final_top, final_color = _solve_object_fast(obj, separator_rows, separator_colors, obstacle, h)
        _paint_object(out, obj, final_top, final_color)
        final_rows = final_top + obj.rel_rows
        obstacle[final_rows, obj.cols] = True

    return out


_BASE_SHAPES = [
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1, 1, 1, 1]], dtype=int),
    np.array([[1, 1, 1, 1, 1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=int),
]


def _shape_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for arr in (rot, np.fliplr(rot)):
            key = (arr.tobytes(), tuple(arr.shape))
            if key not in seen:
                seen.add(key)
                variants.append(arr.copy())
    return variants


def _random_shape(max_height: int, max_width: int) -> np.ndarray:
    variants: list[np.ndarray] = []
    for base in _BASE_SHAPES:
        for arr in _shape_variants(base):
            h, w = arr.shape
            area = int(arr.sum())
            if h <= max_height and w <= max_width and 3 <= area <= 5:
                variants.append(arr)
    if not variants:
        raise ValueError("no shape fits the chosen band")
    return random.choice(variants).copy()


def _band_data(line_rows: list[int], line_colors: dict[int, int], h: int) -> list[dict]:
    bounds = [-1, *line_rows, h]
    bands = []
    for idx in range(len(bounds) - 1):
        upper = int(bounds[idx])
        lower = int(bounds[idx + 1])
        top_color = GREY if upper == -1 else int(line_colors[upper])
        bottom_color = GREY if lower == h else int(line_colors[lower])
        bands.append(
            {
                "index": idx,
                "upper": upper,
                "lower": lower,
                "height": int(lower - upper - 1),
                "top_color": top_color,
                "bottom_color": bottom_color,
                "type": ("G" if top_color == GREY else "O") + ("G" if bottom_color == GREY else "O"),
            }
        )
    return bands


def _choose_layout() -> tuple[int, int, list[int], dict[int, int], list[dict]]:
    while True:
        num_lines = random.randint(4, 6)
        band_heights = [random.randint(3, 5) for _ in range(num_lines + 1)]
        h = int(sum(band_heights) + num_lines)
        if not (15 <= h <= 30):
            continue
        w = random.randint(18, 24)

        separator_rows: list[int] = []
        separator_colors: dict[int, int] = {}
        row = band_heights[0]
        start = ORANGE if random.random() < 0.5 else GREY
        for i in range(num_lines):
            separator_rows.append(int(row))
            separator_colors[int(row)] = int(start if i % 2 == 0 else (GREY if start == ORANGE else ORANGE))
            row += 1 + band_heights[i + 1]

        if sum(int(c == GREY) for c in separator_colors.values()) < 2:
            continue
        if sum(int(c == ORANGE) for c in separator_colors.values()) < 2:
            continue
        bands = _band_data(separator_rows, separator_colors, h)
        by_type = {t: [b for b in bands if b["type"] == t] for t in ("GG", "GO", "OG", "OO")}
        if by_type["GG"] and by_type["GO"] and by_type["OG"]:
            return h, w, separator_rows, separator_colors, bands


def _allowed_row_range(band: dict, shape_h: int) -> tuple[int, int] | None:
    lo = int(band["upper"] + 2)
    hi = int(band["lower"] - shape_h - 1)
    if lo > hi:
        return None
    return lo, hi


def _place_object(
    grid: np.ndarray,
    band: dict,
    color: int,
    used_cols: set[int],
) -> dict | None:
    max_height = max(1, int(band["height"] - 2))
    for _ in range(80):
        shape = _random_shape(max_height=max_height, max_width=5)
        shape_h, shape_w = shape.shape
        row_range = _allowed_row_range(band, shape_h)
        if row_range is None:
            continue

        starts = []
        for c0 in range(0, grid.shape[1] - shape_w + 1):
            cols = range(c0, c0 + shape_w)
            if any(c in used_cols for c in cols):
                continue
            starts.append(int(c0))
        if not starts:
            continue

        r0 = random.randint(row_range[0], row_range[1])
        c0 = random.choice(starts)
        rr, cc = np.nonzero(shape)
        abs_rows = rr + int(r0)
        abs_cols = cc + int(c0)
        if np.any(grid[abs_rows, abs_cols] != BG):
            continue
        grid[abs_rows, abs_cols] = int(color)
        for col in range(c0, c0 + shape_w):
            used_cols.add(int(col))
        return {
            "band": int(band["index"]),
            "color": int(color),
            "bbox": (int(r0), int(c0), int(r0 + shape_h - 1), int(c0 + shape_w - 1)),
            "type": band["type"],
        }
    return None


def _behavior_kind(band: dict, color: int) -> str:
    if color == RED:
        return "bounce" if int(band["top_color"]) == ORANGE else "direct"
    return "bounce" if int(band["bottom_color"]) == ORANGE else "direct"


def _build_attempt() -> np.ndarray | None:
    h, w, separator_rows, separator_colors, bands = _choose_layout()
    grid = np.zeros((h, w), dtype=int)
    for row in separator_rows:
        grid[int(row), :] = int(separator_colors[int(row)])

    by_type: dict[str, list[dict]] = {}
    for band in bands:
        by_type.setdefault(str(band["type"]), []).append(band)

    gg_band = random.choice(by_type["GG"])
    go_band = random.choice(by_type["GO"])
    og_band = random.choice(by_type["OG"])

    plans = [
        (gg_band, RED),
        (gg_band, BLUE),
        (go_band, BLUE),
        (og_band, RED),
    ]

    extra_count = random.randint(0, 2)
    candidate_bands = [band for band in bands if band["type"] != "OO"]
    per_band_count = {int(band["index"]): 0 for band in bands}
    for band, _color in plans:
        per_band_count[int(band["index"])] += 1

    for _ in range(extra_count):
        options = [band for band in candidate_bands if per_band_count[int(band["index"])] < 2]
        if not options:
            break
        band = random.choice(options)
        color = random.choice([BLUE, RED])
        plans.append((band, color))
        per_band_count[int(band["index"])] += 1

    random.shuffle(plans)
    used_cols: set[int] = set()
    placements = []
    for band, color in plans:
        placed = _place_object(grid, band, color, used_cols)
        if placed is None:
            return None
        placements.append(placed)

    if len(placements) < 4:
        return None
    return grid if _validate_generated(grid) else None


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    h, w = grid.shape
    if max(h, w) > 30:
        return False

    separator_rows, separator_colors = _detect_separator_rows(grid)
    if not (4 <= len(separator_rows) <= 6):
        return False
    if sum(int(c == GREY) for c in separator_colors.values()) < 2:
        return False
    if sum(int(c == ORANGE) for c in separator_colors.values()) < 2:
        return False

    bands = _band_data(separator_rows, separator_colors, h)
    objects = _detect_objects(grid)
    if len(objects) < 4:
        return False

    if not ({BLUE, RED} <= set(int(obj.color) for obj in objects)):
        return False

    seen_cols: set[int] = set()
    per_band = {int(band["index"]): 0 for band in bands}
    direct = 0
    bounce = 0
    for obj in objects:
        r0, c0, r1, c1 = obj.bbox
        band_idx = None
        for band in bands:
            if int(band["upper"]) < r0 and r1 < int(band["lower"]):
                band_idx = int(band["index"])
                break
        if band_idx is None:
            return False
        per_band[band_idx] += 1
        if per_band[band_idx] > 2:
            return False
        for col in range(c0, c1 + 1):
            if col in seen_cols:
                return False
            seen_cols.add(int(col))

        band = bands[band_idx]
        if _behavior_kind(band, obj.color) == "bounce":
            bounce += 1
        else:
            direct += 1

    if bounce < 2 or direct < 2:
        return False

    fast = main(grid)
    slow = _semantic_main(grid)
    if not np.array_equal(fast, slow):
        return False
    if np.array_equal(fast, grid):
        return False

    out_objects = _detect_objects(fast)
    if len(out_objects) != len(objects):
        return False
    if len({(obj.bbox[0], obj.bbox[1], obj.color) for obj in out_objects}) < 3:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
