from __future__ import annotations

import random

import numpy as np


BG = 0
RECT_COLOR = 3
RAY_COLOR = 5
PIXEL_COLORS = (1, 2, 4, 6, 7, 9)
SETTLED_CODE = 3

GRID_H_RANGE = (15, 22)
GRID_W_RANGE = (17, 24)
RECT_H_RANGE = (4, 6)
RECT_W_RANGE = (5, 7)
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            out.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "size": int(coords.shape[0]),
                }
            )
    return out


def _locate_rectangle(grid: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    comps = _components(grid != BG)
    assert comps, "expected foreground objects"
    sizes = [int(comp["size"]) for comp in comps]
    largest = max(sizes)
    rect_candidates = [comp for comp in comps if int(comp["size"]) == largest]
    assert len(rect_candidates) == 1, "largest foreground component must be unique"
    rect = rect_candidates[0]
    r0, c0, r1, c1 = rect["bbox"]
    expected_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    assert int(rect["size"]) == expected_area, "largest component must be a filled rectangle"
    color = int(grid[r0, c0])
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle must be monochrome"
    return (r0, c0, r1, c1), color


def _parse_scene(grid: np.ndarray) -> tuple[tuple[int, int, int, int], int, list[tuple[int, int, int]], list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    rect, rect_color = _locate_rectangle(grid)
    r0, c0, r1, c1 = rect
    row_aligned: list[tuple[int, int, int]] = []
    col_aligned: list[tuple[int, int, int]] = []
    free_pixels: list[tuple[int, int, int]] = []

    for comp in _components(grid != BG):
        if comp["bbox"] == rect and int(comp["size"]) == (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        assert int(comp["size"]) == 1, "all non-rectangle objects must be isolated pixels"
        rr, cc = map(int, comp["coords"][0])
        color = int(grid[rr, cc])
        if r0 <= rr <= r1:
            row_aligned.append((rr, cc, color))
        elif c0 <= cc <= c1:
            col_aligned.append((rr, cc, color))
        else:
            free_pixels.append((rr, cc, color))
    return rect, rect_color, row_aligned, col_aligned, free_pixels


def _draw_fixed_obstacles(
    shape: tuple[int, int],
    rect: tuple[int, int, int, int],
    rect_color: int,
    row_aligned: list[tuple[int, int, int]],
    col_aligned: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    r0, c0, r1, c1 = rect
    out = np.full((h, w), BG, dtype=int)
    support = np.zeros((h, w), dtype=np.int8)

    out[r0 : r1 + 1, c0 : c1 + 1] = rect_color
    support[r0 : r1 + 1, c0 : c1 + 1] = 1

    for rr, cc, _color in row_aligned:
        if cc < c0:
            out[rr, cc:c0] = RAY_COLOR
            support[rr, cc:c0] = 2
        else:
            out[rr, c1 + 1 : cc + 1] = RAY_COLOR
            support[rr, c1 + 1 : cc + 1] = 2

    for rr, cc, _color in col_aligned:
        if rr < r0:
            out[rr:r0, cc] = RAY_COLOR
            support[rr:r0, cc] = 2
        else:
            out[r1 + 1 : rr + 1, cc] = RAY_COLOR
            support[r1 + 1 : rr + 1, cc] = 2

    return out, support


def _settle_free_pixels(
    base: np.ndarray,
    support: np.ndarray,
    free_pixels: list[tuple[int, int, int]],
) -> tuple[np.ndarray, list[dict]]:
    out = base.copy()
    occ = support.copy()
    per_col: dict[int, list[tuple[int, int]]] = {}
    for rr, cc, color in free_pixels:
        per_col.setdefault(int(cc), []).append((int(rr), int(color)))

    landings: list[dict] = []
    for cc, items in per_col.items():
        items.sort(key=lambda item: int(item[0]), reverse=True)
        for rr, color in items:
            below = np.flatnonzero(occ[rr + 1 :, cc])
            if below.size == 0:
                dest_r = out.shape[0] - 1
                support_kind = "bottom"
            else:
                blocker_r = int(rr + 1 + below[0])
                dest_r = blocker_r - 1
                blocker_code = int(occ[blocker_r, cc])
                support_kind = {1: "rectangle", 2: "ray", SETTLED_CODE: "stack"}[blocker_code]
            out[dest_r, cc] = color
            occ[dest_r, cc] = SETTLED_CODE
            landings.append(
                {
                    "src": (rr, cc),
                    "dst": (int(dest_r), int(cc)),
                    "color": int(color),
                    "support": support_kind,
                }
            )
    return out, landings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn rectangle-aligned pixels into gray support rays, then let the remaining pixels fall onto those new platforms.

    Input structure:
    - Black is the background.
    - One large monochrome filled rectangle is the unique largest foreground component.
    - All other foreground objects are isolated single pixels, some sharing a row or column with the rectangle and others sharing neither.

    Rule:
    - Identify the largest filled rectangle and treat every other foreground cell as a single-pixel object.
    - Any pixel sharing a row with the rectangle becomes a gray horizontal ray from that pixel toward the rectangle, stopping just before the rectangle.
    - Any pixel sharing a column with the rectangle becomes a gray vertical ray toward the rectangle, also stopping just before the rectangle.
    - Treat the rectangle and all gray rays as fixed obstacles.
    - For the remaining free pixels, process each column from lower source pixels to higher source pixels and drop each one straight down until the next occupied cell below or the grid bottom stops it.

    Key insight:
    - The aligned pixels sacrifice themselves to build new obstacles, and those newly created gray segments determine where the untouched pixels can settle under gravity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rect, rect_color, row_aligned, col_aligned, free_pixels = _parse_scene(g)
    base, support = _draw_fixed_obstacles(g.shape, rect, rect_color, row_aligned, col_aligned)
    out, _landings = _settle_free_pixels(base, support, free_pixels)
    return out


def _ray_span_columns(rect: tuple[int, int, int, int], emitter: tuple[int, int]) -> list[int]:
    _r, c = emitter
    _r0, c0, _r1, c1 = rect
    if c < c0:
        return list(range(int(c), int(c0)))
    return list(range(int(c1 + 1), int(c + 1)))


def _sample_colors(n: int) -> list[int]:
    colors = list(PIXEL_COLORS)
    random.shuffle(colors)
    if n <= len(colors):
        return colors[:n]
    extra = [random.choice(PIXEL_COLORS) for _ in range(n - len(colors))]
    out = colors + extra
    random.shuffle(out)
    return out


def _touches_rectangle(r: int, c: int, rect: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect
    return (r0 - 1) <= r <= (r1 + 1) and (c0 - 1) <= c <= (c1 + 1)


def _can_place_singleton(
    r: int,
    c: int,
    rect: tuple[int, int, int, int],
    used: list[tuple[int, int]],
) -> bool:
    r0, c0, r1, c1 = rect
    if not (0 <= r and 0 <= c):
        return False
    if r0 <= r <= r1 and c0 <= c <= c1:
        return False
    if _touches_rectangle(r, c, rect):
        return False
    for rr, cc in used:
        if max(abs(rr - r), abs(cc - c)) <= 1:
            return False
    return True


def _pick_row(
    c: int,
    lo: int,
    hi: int,
    rect: tuple[int, int, int, int],
    used: list[tuple[int, int]],
    forbidden_rows: set[int],
) -> int | None:
    if lo > hi:
        return None
    rows = list(range(int(lo), int(hi) + 1))
    random.shuffle(rows)
    for r in rows:
        if r in forbidden_rows:
            continue
        if _can_place_singleton(r, c, rect, used):
            return int(r)
    return None


def _render_input(scene: dict) -> np.ndarray:
    h, w = scene["shape"]
    r0, c0, r1, c1 = scene["rect"]
    grid = np.full((h, w), BG, dtype=int)
    grid[r0 : r1 + 1, c0 : c1 + 1] = RECT_COLOR
    for rr, cc, color in scene["row_aligned"]:
        grid[rr, cc] = color
    for rr, cc, color in scene["col_aligned"]:
        grid[rr, cc] = color
    for rr, cc, color in scene["free_pixels"]:
        grid[rr, cc] = color
    return grid


def _render_expected_from_scene(scene: dict) -> tuple[np.ndarray, list[dict]]:
    base, support = _draw_fixed_obstacles(
        scene["shape"],
        scene["rect"],
        RECT_COLOR,
        scene["row_aligned"],
        scene["col_aligned"],
    )
    return _settle_free_pixels(base, support, scene["free_pixels"])


def _scene_branch_summary(scene: dict) -> dict:
    expected, landings = _render_expected_from_scene(scene)
    counts: dict[str, int] = {"bottom": 0, "ray": 0, "rectangle": 0, "stack": 0}
    for landing in landings:
        counts[str(landing["support"])] += 1

    ray_support_rows = [
        int(landing["dst"][0] + 1)
        for landing in landings
        if str(landing["support"]) == "ray"
    ]
    return {
        "expected": expected,
        "landings": landings,
        "counts": counts,
        "shared_ray_multicatch": any(ray_support_rows.count(row) >= 2 for row in set(ray_support_rows)),
    }


def _validate_scene(scene: dict) -> bool:
    h, w = scene["shape"]
    if max(h, w) > 30:
        return False

    rect = scene["rect"]
    r0, c0, r1, c1 = rect
    if not (RECT_H_RANGE[0] <= r1 - r0 + 1 <= RECT_H_RANGE[1]):
        return False
    if not (RECT_W_RANGE[0] <= c1 - c0 + 1 <= RECT_W_RANGE[1]):
        return False

    row_aligned = scene["row_aligned"]
    col_aligned = scene["col_aligned"]
    free_pixels = scene["free_pixels"]
    total_singletons = len(row_aligned) + len(col_aligned) + len(free_pixels)
    if not (6 <= total_singletons <= 10):
        return False
    if not row_aligned or not col_aligned:
        return False

    used: list[tuple[int, int]] = []
    for rr, cc, _color in row_aligned + col_aligned + free_pixels:
        if not _can_place_singleton(int(rr), int(cc), rect, [p for p in used if p != (rr, cc)]):
            return False
        used.append((int(rr), int(cc)))

    row_keys = [int(rr) for rr, _cc, _color in row_aligned]
    col_keys = [int(cc) for _rr, cc, _color in col_aligned]
    if len(row_keys) != len(set(row_keys)):
        return False
    if len(col_keys) != len(set(col_keys)):
        return False

    for rr, cc, _color in row_aligned:
        if not (r0 <= rr <= r1):
            return False
        if c0 <= cc <= c1:
            return False
        if cc < c0 and c0 - cc < 2:
            return False
        if cc > c1 and cc - c1 < 2:
            return False

    for rr, cc, _color in col_aligned:
        if not (c0 <= cc <= c1):
            return False
        if r0 <= rr <= r1:
            return False
        if rr < r0 and r0 - rr < 2:
            return False
        if rr > r1 and rr - r1 < 2:
            return False

    rect_rows = set(range(r0, r1 + 1))
    rect_cols = set(range(c0, c1 + 1))
    for rr, cc, _color in free_pixels:
        if rr in rect_rows or cc in rect_cols:
            return False

    summary = _scene_branch_summary(scene)
    counts = summary["counts"]
    if counts["bottom"] < 1 or counts["ray"] < 2 or counts["stack"] < 1:
        return False
    if counts["rectangle"] != 0:
        return False
    if not summary["shared_ray_multicatch"]:
        return False
    if not all(int(landing["dst"][0]) > int(landing["src"][0]) for landing in summary["landings"]):
        return False

    grid = _render_input(scene)
    parsed_rect, _rect_color, parsed_rows, parsed_cols, parsed_free = _parse_scene(grid)
    if parsed_rect != rect:
        return False
    if sorted((r, c) for r, c, _ in parsed_rows) != sorted((r, c) for r, c, _ in row_aligned):
        return False
    if sorted((r, c) for r, c, _ in parsed_cols) != sorted((r, c) for r, c, _ in col_aligned):
        return False
    if sorted((r, c) for r, c, _ in parsed_free) != sorted((r, c) for r, c, _ in free_pixels):
        return False

    if not np.array_equal(main(grid), summary["expected"]):
        return False
    return True


def _sample_scene_once() -> dict | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    rect_h = random.randint(*RECT_H_RANGE)
    rect_w = random.randint(*RECT_W_RANGE)

    rect_r0_min = max(6, h // 2 - 1)
    rect_r0_max = h - rect_h - 3
    rect_c0_min = 4
    rect_c0_max = w - rect_w - 5
    if rect_r0_min > rect_r0_max or rect_c0_min > rect_c0_max:
        return None

    rect_r0 = random.randint(rect_r0_min, rect_r0_max)
    rect_c0 = random.randint(rect_c0_min, rect_c0_max)
    rect = (rect_r0, rect_c0, rect_r0 + rect_h - 1, rect_c0 + rect_w - 1)
    r0, c0, r1, c1 = rect

    platform_side = random.choice(["left", "right"])
    platform_row_choices = list(range(r0, r0 + (rect_h + 1) // 2))
    platform_row = random.choice(platform_row_choices)

    if platform_side == "left":
        platform_col = random.randint(1, c0 - 3)
        platform_usable_cols = list(range(platform_col + 1, c0))
        opposite_side = "right"
    else:
        platform_col = random.randint(c1 + 3, w - 2)
        platform_usable_cols = list(range(c1 + 1, platform_col))
        opposite_side = "left"
    if len(platform_usable_cols) < 2:
        return None

    row_emitters: list[tuple[int, int]] = [(platform_row, platform_col)]
    used = [(platform_row, platform_col)]

    extra_row_count = random.randint(0, min(2, rect_h - 1))
    extra_rows = [row for row in range(r0, r1 + 1) if row != platform_row]
    random.shuffle(extra_rows)
    for row in extra_rows[:extra_row_count]:
        if opposite_side == "left":
            col = random.randint(1, c0 - 2)
        else:
            col = random.randint(c1 + 2, w - 2)
        if not _can_place_singleton(row, col, rect, used):
            return None
        row_emitters.append((row, col))
        used.append((row, col))

    col_count = random.randint(1, min(2, rect_w))
    col_choices = list(range(c0, c1 + 1))
    random.shuffle(col_choices)
    col_emitters: list[tuple[int, int]] = []
    for col in col_choices[:col_count]:
        side = random.choice(["top", "bottom"])
        if side == "top":
            row = random.randint(1, r0 - 2)
        else:
            row = random.randint(r1 + 2, h - 2)
        if not _can_place_singleton(row, col, rect, used):
            return None
        col_emitters.append((row, col))
        used.append((row, col))

    all_row_columns = set()
    for emitter in row_emitters:
        all_row_columns.update(_ray_span_columns(rect, emitter))
    open_cols = [col for col in range(w) if col not in all_row_columns and col not in set(range(c0, c1 + 1))]
    if not open_cols:
        return None

    forbidden_rows = set(range(r0, r1 + 1))
    free_pixels: list[tuple[int, int]] = []

    stack_col = random.choice(platform_usable_cols)
    other_platform_cols = [col for col in platform_usable_cols if col != stack_col]
    if not other_platform_cols:
        return None
    second_platform_col = random.choice(other_platform_cols)

    low_row = _pick_row(stack_col, 0, platform_row - 2, rect, used, forbidden_rows)
    if low_row is None:
        return None
    used.append((low_row, stack_col))
    free_pixels.append((low_row, stack_col))

    high_row = _pick_row(stack_col, 0, low_row - 2, rect, used, forbidden_rows)
    if high_row is None:
        return None
    used.append((high_row, stack_col))
    free_pixels.append((high_row, stack_col))

    ray_row = _pick_row(second_platform_col, 0, platform_row - 2, rect, used, forbidden_rows)
    if ray_row is None:
        return None
    used.append((ray_row, second_platform_col))
    free_pixels.append((ray_row, second_platform_col))

    bottom_col = random.choice(open_cols)
    bottom_row = _pick_row(bottom_col, 0, h - 3, rect, used, forbidden_rows)
    if bottom_row is None:
        return None
    used.append((bottom_row, bottom_col))
    free_pixels.append((bottom_row, bottom_col))

    aligned_count = len(row_emitters) + len(col_emitters)
    max_extra = min(3, 10 - aligned_count - len(free_pixels))
    extra_free_count = random.randint(0, max_extra)

    platform_extra_cols = [col for col in platform_usable_cols if col not in {stack_col, second_platform_col}]
    random.shuffle(platform_extra_cols)
    shuffled_open_cols = open_cols[:]
    random.shuffle(shuffled_open_cols)

    for _ in range(extra_free_count):
        choose_platform = bool(platform_extra_cols) and (not shuffled_open_cols or random.random() < 0.5)
        if choose_platform:
            col = platform_extra_cols.pop()
            row = _pick_row(col, 0, platform_row - 2, rect, used, forbidden_rows)
        else:
            if not shuffled_open_cols:
                break
            col = shuffled_open_cols.pop()
            row = _pick_row(col, 0, h - 3, rect, used, forbidden_rows)
        if row is None:
            return None
        used.append((row, col))
        free_pixels.append((row, col))

    total_singletons = len(row_emitters) + len(col_emitters) + len(free_pixels)
    colors = _sample_colors(total_singletons)
    idx = 0
    row_aligned = []
    for rr, cc in row_emitters:
        row_aligned.append((rr, cc, colors[idx]))
        idx += 1
    col_aligned = []
    for rr, cc in col_emitters:
        col_aligned.append((rr, cc, colors[idx]))
        idx += 1
    colored_free = []
    for rr, cc in free_pixels:
        colored_free.append((rr, cc, colors[idx]))
        idx += 1

    return {
        "shape": (h, w),
        "rect": rect,
        "row_aligned": row_aligned,
        "col_aligned": col_aligned,
        "free_pixels": colored_free,
    }


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        scene = _sample_scene_once()
        if scene is None:
            continue
        if _validate_scene(scene):
            return _render_input(scene)
    raise RuntimeError("failed to generate a valid input for puzzle 6")
