from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
GUIDE = 2
OBJECT_COLORS = (1, 3, 4, 6)
AREA_CHOICES = (3, 4, 5)

HEIGHT_RANGE = (14, 20)
WIDTH_FLOOR = 10
WIDTH_CEIL = 15
GUIDE_FROM_BOTTOM = (3, 4)
OBJECT_COUNT_CHOICES = (2, 3, 4)
MAX_SHAPE_HEIGHT = 3
MAX_SPAN_WIDTH = 4
GENERATION_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_guide_row(grid: np.ndarray, bg: int) -> tuple[int, set[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, set[int]]] = []
    for r in range(h):
        row = grid[r]
        if not np.all((row == bg) | (row == GUIDE)):
            continue
        gap_cols = {int(c) for c in np.where(row == bg)[0]}
        if not 2 <= len(gap_cols) <= 4:
            continue
        if any((c + 1) in gap_cols for c in gap_cols):
            continue
        if np.count_nonzero(row == GUIDE) != w - len(gap_cols):
            continue
        candidates.append((r, gap_cols))
    assert len(candidates) == 1, "expected exactly one horizontal guide row"
    return candidates[0]


def _collect_objects(grid: np.ndarray, guide_row: int, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    objects: list[dict] = []
    for r in range(guide_row):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color in (bg, GUIDE):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < guide_row
                        and 0 <= nc < w
                        and not visited[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        visited[nr, nc] = True
                        q.append((nr, nc))
            coords_arr = np.asarray(coords, dtype=int)
            objects.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "area": int(coords_arr.shape[0]),
                    "bbox": (
                        int(coords_arr[:, 0].min()),
                        int(coords_arr[:, 1].min()),
                        int(coords_arr[:, 0].max()),
                        int(coords_arr[:, 1].max()),
                    ),
                }
            )
    return sorted(objects, key=lambda obj: obj["bbox"])


def main(grid: np.ndarray) -> np.ndarray:
    """Translate each object downward by its area, then keep only landed pixels aligned to solid guide-bar columns.

    Input structure:
    - Black background dominates the grid.
    - A single red horizontal guide bar sits in the lower half, spans the full width, and contains 2-4 isolated 1-cell black gaps.
    - Two to four small 4-connected monochrome objects of distinct non-red colors appear above the guide row.

    Rule:
    - Identify each non-guide object above the bar and count its pixels.
    - Erase every object from its original position, then translate the whole object straight downward by exactly its area in rows.
    - For every translated pixel, inspect the guide-row cell in the same column: keep the pixel only if that guide cell is red, recoloring the kept pixel red; otherwise erase it to black.
    - Leave the red guide bar itself unchanged.

    Key insight:
    - Object area controls the vertical landing position, while the guide bar's gap pattern filters the landed shape column by column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    guide_row, _gap_cols = _find_guide_row(g, bg)
    objects = _collect_objects(g, guide_row, bg)
    assert 2 <= len(objects) <= 4

    out = g.copy()
    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for obj in objects:
        shift = int(obj["area"])
        for r, c in obj["coords"]:
            rr = int(r) + shift
            cc = int(c)
            assert 0 <= rr < g.shape[0]
            if int(g[guide_row, cc]) == GUIDE:
                out[rr, cc] = GUIDE
            else:
                out[rr, cc] = bg

    return out


def _sample_size() -> tuple[int, int]:
    h = random.randint(*HEIGHT_RANGE)
    w = random.randint(WIDTH_FLOOR, min(WIDTH_CEIL, h - 1))
    return h, w


def _sample_gap_cols(width: int, count: int) -> set[int]:
    choices = list(range(1, width - 1))
    for _ in range(200):
        cols = set(random.sample(choices, count))
        if all(abs(a - b) > 1 for a in cols for b in cols if a != b):
            return cols
    raise RuntimeError("failed to place isolated guide gaps")


def _sample_widths(count: int, width: int) -> list[int] | None:
    available = width - (count - 1)
    for _ in range(200):
        widths = [random.randint(1, MAX_SPAN_WIDTH) for _ in range(count)]
        if sum(widths) > available:
            continue
        if sum(w >= 2 for w in widths) < 2:
            continue
        return widths
    return None


def _spans_from_widths(widths: list[int], width: int) -> list[tuple[int, int]]:
    extra = width - sum(widths) - (len(widths) - 1)
    buckets = [0] * (len(widths) + 1)
    for _ in range(extra):
        buckets[random.randrange(len(buckets))] += 1

    spans: list[tuple[int, int]] = []
    c = buckets[0]
    for i, span_width in enumerate(widths):
        spans.append((c, c + span_width - 1))
        c += span_width
        if i < len(widths) - 1:
            c += 1 + buckets[i + 1]
    return spans


def _choose_special_indices(
    spans: list[tuple[int, int]], gap_cols: set[int]
) -> tuple[int, int] | None:
    partial_candidates = []
    for i, (c0, c1) in enumerate(spans):
        cols = set(range(c0, c1 + 1))
        if len(cols) >= 2 and cols & gap_cols and cols - gap_cols:
            partial_candidates.append(i)
    if not partial_candidates:
        return None

    partial_idx = random.choice(partial_candidates)
    touch_candidates = [i for i, (c0, c1) in enumerate(spans) if i != partial_idx and (c1 - c0 + 1) >= 2]
    if not touch_candidates:
        return None
    touch_idx = random.choice(touch_candidates)
    return partial_idx, touch_idx


def _sample_roles(
    spans: list[tuple[int, int]],
    guide_row: int,
    gap_cols: set[int],
) -> list[dict] | None:
    picked = _choose_special_indices(spans, gap_cols)
    if picked is None:
        return None
    partial_idx, touch_idx = picked

    roles: list[dict] = []
    for i, (c0, c1) in enumerate(spans):
        span_width = c1 - c0 + 1
        max_area = min(5, 3 * span_width)
        feasible = [a for a in AREA_CHOICES if a <= max_area]
        if i == partial_idx:
            areas = feasible
            if not areas:
                return None
            area = random.choice(areas)
            bottom = guide_row - 1
            cols = set(range(c0, c1 + 1))
            gap_col = min(cols & gap_cols, key=lambda c: min(abs(c - s) for s in cols - gap_cols))
            solid_col = min(cols - gap_cols, key=lambda c: abs(c - gap_col))
            required_cols = tuple(sorted({gap_col - c0, solid_col - c0}))
        elif i == touch_idx:
            areas = [a for a in feasible if a >= 4]
            if not areas:
                return None
            area = random.choice(areas)
            bottom = guide_row
            required_cols = ()
        else:
            if not feasible:
                return None
            area = random.choice(feasible)
            bottom = guide_row - 1 if area == 3 else random.choice((guide_row - 1, guide_row))
            required_cols = ()
        roles.append(
            {
                "left": c0,
                "width": span_width,
                "area": area,
                "bottom": bottom,
                "required_cols": required_cols,
            }
        )

    areas = [role["area"] for role in roles]
    bottoms = [role["bottom"] for role in roles]
    if len(set(areas)) < 2:
        return None
    if guide_row not in bottoms or (guide_row - 1) not in bottoms:
        return None
    return roles


def _random_connected_mask(area: int, width: int, required_cols: tuple[int, ...]) -> np.ndarray | None:
    heights = [h for h in range(1, MAX_SHAPE_HEIGHT + 1) if area <= h * width]
    random.shuffle(heights)
    for height in heights:
        for _ in range(120):
            if required_cols:
                start = min(required_cols)
                end = max(required_cols)
                seed = {(height - 1, c) for c in range(start, end + 1)}
                if len(seed) > area:
                    continue
            else:
                seed = {(height - 1, random.randrange(width))}
            cells = set(seed)
            while len(cells) < area:
                frontier = set()
                for r, c in cells:
                    for nr, nc in _neighbors4(r, c):
                        if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                            frontier.add((nr, nc))
                if not frontier:
                    break
                cells.add(random.choice(tuple(frontier)))
            if len(cells) != area:
                continue
            used_cols = {c for _, c in cells}
            if any(c not in used_cols for c in required_cols):
                continue
            if not any(r == height - 1 for r, _ in cells):
                continue
            mask = np.zeros((height, width), dtype=bool)
            for r, c in cells:
                mask[r, c] = True
            return mask
    return None


def _build_case() -> tuple[np.ndarray, list[dict], int, set[int]] | None:
    h, w = _sample_size()
    guide_row = h - random.randint(*GUIDE_FROM_BOTTOM)
    gap_cols = _sample_gap_cols(w, random.randint(2, 4))

    count = random.choice(OBJECT_COUNT_CHOICES)
    widths = _sample_widths(count, w)
    if widths is None:
        return None
    random.shuffle(widths)
    spans = _spans_from_widths(widths, w)

    roles = _sample_roles(spans, guide_row, gap_cols)
    if roles is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[guide_row, :] = GUIDE
    for c in gap_cols:
        grid[guide_row, c] = BG

    colors = random.sample(list(OBJECT_COLORS), count)
    placements: list[dict] = []
    for color, role in zip(colors, roles):
        mask = _random_connected_mask(role["area"], role["width"], role["required_cols"])
        if mask is None:
            return None
        height = mask.shape[0]
        final_top = int(role["bottom"]) - height + 1
        input_top = final_top - int(role["area"])
        if input_top < 0:
            return None
        input_bottom = input_top + height - 1
        if input_bottom > guide_row - 4:
            return None

        rows, cols = np.where(mask)
        abs_rows = rows + input_top
        abs_cols = cols + int(role["left"])
        grid[abs_rows, abs_cols] = int(color)
        placements.append(
            {
                "color": int(color),
                "area": int(role["area"]),
                "left": int(role["left"]),
                "input_top": int(input_top),
                "final_top": int(final_top),
                "bottom": int(role["bottom"]),
                "mask": mask,
            }
        )

    return grid, placements, guide_row, gap_cols


def _expected_output(grid: np.ndarray, placements: list[dict], guide_row: int) -> np.ndarray:
    out = grid.copy()
    for place in placements:
        rows, cols = np.where(place["mask"])
        out[rows + place["input_top"], cols + place["left"]] = BG
    for place in placements:
        rows, cols = np.where(place["mask"])
        target_rows = rows + place["final_top"]
        target_cols = cols + place["left"]
        solid = grid[guide_row, target_cols] == GUIDE
        out[target_rows[solid], target_cols[solid]] = GUIDE
        out[target_rows[~solid], target_cols[~solid]] = BG
    return out


def _alt_ignore_gaps(grid: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = grid.copy()
    for place in placements:
        rows, cols = np.where(place["mask"])
        out[rows + place["input_top"], cols + place["left"]] = BG
        out[rows + place["final_top"], cols + place["left"]] = GUIDE
    return out


def _alt_no_shift(grid: np.ndarray, placements: list[dict], guide_row: int) -> np.ndarray:
    out = grid.copy()
    for place in placements:
        rows, cols = np.where(place["mask"])
        src_rows = rows + place["input_top"]
        src_cols = cols + place["left"]
        out[src_rows, src_cols] = BG
        solid = grid[guide_row, src_cols] == GUIDE
        out[src_rows[solid], src_cols[solid]] = GUIDE
    return out


def _alt_shift_by_height(grid: np.ndarray, placements: list[dict], guide_row: int) -> np.ndarray:
    out = grid.copy()
    for place in placements:
        rows, cols = np.where(place["mask"])
        src_rows = rows + place["input_top"]
        src_cols = cols + place["left"]
        out[src_rows, src_cols] = BG
        shifted_rows = src_rows + place["mask"].shape[0]
        solid = grid[guide_row, src_cols] == GUIDE
        out[shifted_rows[solid], src_cols[solid]] = GUIDE
        out[shifted_rows[~solid], src_cols[~solid]] = BG
    return out


def _validate_case(grid: np.ndarray, placements: list[dict], guide_row: int) -> bool:
    expected = _expected_output(grid, placements, guide_row)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    kept_pixels = []
    erased_pixels = []
    partial_objects = 0
    for place in placements:
        rows, cols = np.where(place["mask"])
        target_rows = rows + place["final_top"]
        target_cols = cols + place["left"]
        solid = grid[guide_row, target_cols] == GUIDE
        kept = [(int(r), int(c)) for r, c, ok in zip(target_rows, target_cols, solid) if ok]
        erased = [(int(r), int(c)) for r, c, ok in zip(target_rows, target_cols, solid) if not ok]
        kept_pixels.extend(kept)
        erased_pixels.extend(erased)
        if kept and erased:
            partial_objects += 1

    if not kept_pixels or not erased_pixels:
        return False
    if partial_objects < 1:
        return False
    if not any(r < guide_row for r, _ in kept_pixels):
        return False
    if not any(place["bottom"] == guide_row for place in placements):
        return False
    if not any(place["bottom"] == guide_row - 1 for place in placements):
        return False
    if not any(place["area"] != place["mask"].shape[0] for place in placements):
        return False

    guide_only = np.full_like(grid, BG)
    guide_only[guide_row, :] = grid[guide_row, :]
    if np.array_equal(actual, guide_only):
        return False
    if np.array_equal(actual, _alt_ignore_gaps(grid, placements)):
        return False
    if np.array_equal(actual, _alt_no_shift(grid, placements, guide_row)):
        return False
    if np.array_equal(actual, _alt_shift_by_height(grid, placements, guide_row)):
        return False

    objects = _collect_objects(grid, guide_row, BG)
    if len(objects) != len(placements):
        return False
    if {obj["color"] for obj in objects} != {place["color"] for place in placements}:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        grid, placements, guide_row, _gap_cols = built
        if _validate_case(grid, placements, guide_row):
            return grid
    raise RuntimeError("failed to generate a valid area-shift guide-gap input")
