from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

NOISE_PALETTE = (GREEN, PINK, ORANGE, TEAL, MAROON)
MIN_SIDE = 4

Rect = tuple[int, int, int, int, int]


def _right_runs(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    runs = np.ones((h, w), dtype=int)
    for r in range(h):
        for c in range(w - 2, -1, -1):
            if int(grid[r, c]) == int(grid[r, c + 1]):
                runs[r, c] = runs[r, c + 1] + 1
    return runs


def _maximal_uniform_rectangles(grid: np.ndarray, min_side: int = MIN_SIDE) -> list[Rect]:
    h, w = grid.shape
    runs = _right_runs(grid)
    rects: set[Rect] = set()

    for r0 in range(h):
        for c0 in range(w):
            color = int(grid[r0, c0])
            max_w = w - c0
            for r1 in range(r0, h):
                if int(grid[r1, c0]) != color:
                    break
                max_w = min(max_w, int(runs[r1, c0]))
                if r1 - r0 + 1 >= min_side and max_w >= min_side:
                    rects.add((r0, c0, r1, c0 + max_w - 1, color))

    maximal: list[Rect] = []
    for rect in rects:
        r0, c0, r1, c1, color = rect
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        contained = False
        for other in rects:
            if other == rect or other[4] != color:
                continue
            or0, oc0, or1, oc1, _ = other
            other_area = (or1 - or0 + 1) * (oc1 - oc0 + 1)
            if other_area > area and or0 <= r0 and oc0 <= c0 and r1 <= or1 and c1 <= oc1:
                contained = True
                break
        if not contained:
            maximal.append(rect)

    return sorted(maximal)


def _apply_marker_cycle(grid: np.ndarray) -> tuple[np.ndarray, set[int], set[int]]:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    active_cols: set[int] = set()
    marker_cols: set[int] = set()

    for c in range(w):
        top = int(grid[1, c])
        below = int(grid[2, c])
        if top == RED and below == RED:
            out[1, c] = BLACK
            out[2, c] = BLACK
            out[h - 2, c] = BLUE
            out[h - 1, c] = BLUE
            active_cols.add(c)
            marker_cols.add(c)
        elif top == RED and below == BLACK:
            out[2, c] = RED
            marker_cols.add(c)
        else:
            assert top == BLACK and below == BLACK, "markers must be single or double vertical pairs"

    return out, active_cols, marker_cols


def _reveal_with_columns(grid: np.ndarray, selected_cols: set[int], min_side: int = MIN_SIDE) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    band = out[3:-2, :]
    for r0, c0, r1, c1, _color in _maximal_uniform_rectangles(band, min_side=min_side):
        if any(c in selected_cols for c in range(c0, c1 + 1)):
            out[r0 + 3 : r1 + 4, c0 : c1 + 1] = YELLOW
    return out


def _solve_with_columns(grid: np.ndarray, selected_cols: set[int], min_side: int = MIN_SIDE) -> np.ndarray:
    cycled, _active_cols, _marker_cols = _apply_marker_cycle(grid)
    return _reveal_with_columns(cycled, selected_cols, min_side=min_side)


def main(grid: np.ndarray) -> np.ndarray:
    """Reveal only those hidden large solid rectangles whose columns are activated by completed marker migration.

    Input structure
    - The top row is a full blue bar.
    - Rows 1-2 contain red column markers: a single marker occupies only row 1, while a double marker occupies rows 1 and 2 in the same column.
    - Rows 3 through the third-from-last row are multicolor noise containing several large solid rectangles; the bottom two rows start as black.

    Rule
    - Convert each single red marker into a double by growing one cell downward.
    - Convert each double red marker into a migrated blue two-pixel mark on the bottom two rows of the same column, clearing its original red cells.
    - Treat the migrated blue columns as the active columns, detect all maximal solid monochrome rectangles in the noise band whose height and width are both at least 4, and recolor exactly the rectangles that overlap at least one active column to yellow.

    Key insight
    - The marker cycle decides which columns become active, and only those columns are allowed to expose hidden large rectangles, so both the migration state and the rectangle detection are necessary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 10
    assert np.all(g[0, :] == BLUE)

    cycled, active_cols, _marker_cols = _apply_marker_cycle(g)
    return _reveal_with_columns(cycled, active_cols, min_side=MIN_SIDE)


# --------------- generator (arc-like) ---------------

GEN_HEIGHT_RANGE = (19, 24)
GEN_WIDTH_RANGE = (18, 24)
GEN_LARGE_RECT_COUNT_RANGE = (3, 5)
GEN_LARGE_SIDE_RANGE = (4, 7)
GEN_SMALL_LONG_SIDE_RANGE = (4, 6)
GEN_MAX_ATTEMPTS = 400
GEN_COLUMN_GAP = 2


def _choose_spaced_columns(width: int, count: int) -> list[int] | None:
    for _ in range(300):
        cols = sorted(random.sample(range(width), count))
        if all(cols[i + 1] - cols[i] >= GEN_COLUMN_GAP for i in range(len(cols) - 1)):
            return cols
    return None


def _interval_kind(
    c0: int,
    c1: int,
    active_cols: set[int],
    single_cols: set[int],
    marker_cols: set[int],
) -> tuple[bool, bool, bool]:
    cols = set(range(c0, c1 + 1))
    return bool(cols & active_cols), bool(cols & single_cols), bool(cols & marker_cols)


def _fits_category(
    category: str,
    c0: int,
    width: int,
    active_cols: set[int],
    single_cols: set[int],
    marker_cols: set[int],
) -> bool:
    has_active, has_single, has_marker = _interval_kind(
        c0, c0 + width - 1, active_cols, single_cols, marker_cols
    )
    if category == "active":
        return has_active
    if category == "single_only":
        return has_single and not has_active
    if category == "inactive_clear":
        return not has_marker
    if category == "mixed":
        return has_active and has_single
    raise AssertionError(f"unknown category {category!r}")


def _placement_candidates(
    blocked: np.ndarray,
    height: int,
    width: int,
    category: str,
    active_cols: set[int],
    single_cols: set[int],
    marker_cols: set[int],
) -> list[tuple[int, int]]:
    noise_h, noise_w = blocked.shape
    out = []
    for r0 in range(noise_h - height + 1):
        for c0 in range(noise_w - width + 1):
            if not _fits_category(category, c0, width, active_cols, single_cols, marker_cols):
                continue
            r1 = r0 + height - 1
            c1 = c0 + width - 1
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(noise_h - 1, r1 + 1)
            cc1 = min(noise_w - 1, c1 + 1)
            if not np.any(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1]):
                out.append((r0, c0))
    return out


def _paint_patch(grid: np.ndarray, patch: dict) -> None:
    r0, c0, r1, c1 = patch["bbox"]
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(patch["color"])


def _forbidden_colors(noise_shape: tuple[int, int], patches: list[dict]) -> list[list[set[int]]]:
    noise_h, noise_w = noise_shape
    forbidden = [[set() for _ in range(noise_w)] for _ in range(noise_h)]
    for patch in patches:
        r0, c0, r1, c1 = patch["bbox"]
        color = int(patch["color"])
        for r in range(max(0, r0 - 1), min(noise_h, r1 + 2)):
            for c in range(max(0, c0 - 1), min(noise_w, c1 + 2)):
                if not (r0 <= r <= r1 and c0 <= c <= c1):
                    forbidden[r][c].add(color)
    return forbidden


def _valid_noise_color(
    noise: np.ndarray,
    patch_mask: np.ndarray,
    r: int,
    c: int,
    color: int,
) -> bool:
    noise_h, noise_w = noise.shape
    for rr0 in range(max(0, r - 1), min(noise_h - 2, r) + 1):
        for cc0 in range(max(0, c - 1), min(noise_w - 2, c) + 1):
            coords = [(rr0 + dr, cc0 + dc) for dr in (0, 1) for dc in (0, 1)]
            if any(patch_mask[rr, cc] for rr, cc in coords):
                continue
            values = []
            for rr, cc in coords:
                if rr == r and cc == c:
                    values.append(int(color))
                else:
                    other = int(noise[rr, cc])
                    if other == -1:
                        values = []
                        break
                    values.append(other)
            if values and len(set(values)) == 1:
                return False
    return True


def _fill_noise(noise: np.ndarray, patch_mask: np.ndarray, patches: list[dict]) -> bool:
    forbidden = _forbidden_colors(noise.shape, patches)
    for r in range(noise.shape[0]):
        for c in range(noise.shape[1]):
            if patch_mask[r, c]:
                continue
            choices = [
                color
                for color in NOISE_PALETTE
                if color not in forbidden[r][c] and _valid_noise_color(noise, patch_mask, r, c, color)
            ]
            if not choices:
                return False
            noise[r, c] = random.choice(choices)
    return True


def _large_rectangles_from_input(grid: np.ndarray, min_side: int = MIN_SIDE) -> list[Rect]:
    rects = _maximal_uniform_rectangles(grid[3:-2, :], min_side=min_side)
    return [(r0 + 3, c0, r1 + 3, c1, color) for r0, c0, r1, c1, color in rects]


def _patch_cells_abs(patch: dict) -> tuple[slice, slice]:
    r0, c0, r1, c1 = patch["bbox"]
    return slice(r0 + 3, r1 + 4), slice(c0, c1 + 1)


def _validate_generated(grid: np.ndarray, patches: list[dict]) -> bool:
    out = main(grid)
    cycled, active_cols, marker_cols = _apply_marker_cycle(grid)
    single_cols = marker_cols - active_cols

    if len(active_cols) < 2 or len(single_cols) < 2:
        return False

    large_patches = [patch for patch in patches if patch["kind"] == "large"]
    small_patches = [patch for patch in patches if patch["kind"] == "small"]
    if len(large_patches) < 3 or not small_patches:
        return False

    actual_large = _large_rectangles_from_input(grid, min_side=MIN_SIDE)
    if len(actual_large) != len(large_patches):
        return False

    active_large = 0
    single_only_large = 0
    for patch in large_patches:
        rows, cols = _patch_cells_abs(patch)
        patch_out = out[rows, cols]
        overlaps_active, overlaps_single, _overlaps_marker = _interval_kind(
            patch["bbox"][1], patch["bbox"][3], active_cols, single_cols, marker_cols
        )
        if overlaps_active:
            active_large += 1
            if not np.all(patch_out == YELLOW):
                return False
        else:
            if not np.array_equal(patch_out, cycled[rows, cols]):
                return False
            if overlaps_single:
                single_only_large += 1

    if active_large < 1 or single_only_large < 1:
        return False

    for patch in small_patches:
        rows, cols = _patch_cells_abs(patch)
        if not np.array_equal(out[rows, cols], cycled[rows, cols]):
            return False

    if np.array_equal(_solve_with_columns(grid, marker_cols, min_side=MIN_SIDE), out):
        return False
    if np.array_equal(_solve_with_columns(grid, set(range(grid.shape[1])), min_side=MIN_SIDE), out):
        return False
    if np.array_equal(_solve_with_columns(grid, active_cols, min_side=3), out):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    noise_h = h - 5

    single_count = random.randint(2, 3)
    double_count = random.randint(2, 4)
    marker_cols_list = _choose_spaced_columns(w, single_count + double_count)
    if marker_cols_list is None:
        return None

    marker_cols = set(marker_cols_list)
    active_cols = set(random.sample(marker_cols_list, double_count))
    single_cols = marker_cols - active_cols

    blocked = np.zeros((noise_h, w), dtype=bool)
    patch_mask = np.zeros((noise_h, w), dtype=bool)
    patches: list[dict] = []

    large_count = random.randint(*GEN_LARGE_RECT_COUNT_RANGE)
    categories = ["active", "single_only"]
    extra_pool = ["active", "inactive_clear", "single_only", "mixed"]
    while len(categories) < large_count:
        categories.append(random.choice(extra_pool))
    random.shuffle(categories)

    large_colors = random.sample(NOISE_PALETTE, min(large_count, len(NOISE_PALETTE)))
    while len(large_colors) < large_count:
        large_colors.append(random.choice(NOISE_PALETTE))

    specs: list[dict] = []
    for i, category in enumerate(categories):
        height = random.randint(*GEN_LARGE_SIDE_RANGE)
        width = random.randint(*GEN_LARGE_SIDE_RANGE)
        specs.append(
            {
                "kind": "large",
                "category": category,
                "height": height,
                "width": width,
                "color": large_colors[i],
            }
        )

    if random.random() < 0.5:
        small_h = 3
        small_w = random.randint(*GEN_SMALL_LONG_SIDE_RANGE)
    else:
        small_h = random.randint(*GEN_SMALL_LONG_SIDE_RANGE)
        small_w = 3
    specs.append(
        {
            "kind": "small",
            "category": "active",
            "height": small_h,
            "width": small_w,
            "color": random.choice(NOISE_PALETTE),
        }
    )

    specs.sort(key=lambda item: (item["category"] in {"single_only", "mixed"}, item["height"] * item["width"]), reverse=True)

    for spec in specs:
        candidates = _placement_candidates(
            blocked,
            spec["height"],
            spec["width"],
            spec["category"],
            active_cols,
            single_cols,
            marker_cols,
        )
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        r1 = r0 + spec["height"] - 1
        c1 = c0 + spec["width"] - 1
        patch = {
            "kind": spec["kind"],
            "category": spec["category"],
            "color": int(spec["color"]),
            "bbox": (r0, c0, r1, c1),
        }
        patches.append(patch)
        patch_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        blocked[max(0, r0 - 1) : min(noise_h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True

    noise = np.full((noise_h, w), -1, dtype=int)
    for patch in patches:
        _paint_patch(noise, patch)
    if not _fill_noise(noise, patch_mask, patches):
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[0, :] = BLUE
    grid[3:-2, :] = noise
    for c in sorted(single_cols):
        grid[1, c] = RED
    for c in sorted(active_cols):
        grid[1, c] = RED
        grid[2, c] = RED

    if not _validate_generated(grid, patches):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
