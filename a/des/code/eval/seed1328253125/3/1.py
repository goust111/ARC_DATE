import random
from collections import Counter, deque

import numpy as np


GREY = 5
SIDES = ("top", "bottom", "left", "right")

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
INNER_SPAN_RANGE = (4, 8)
BRUSH_SIZES = (2, 3)
PLACEMENT_RETRIES = 200
GEN_RETRIES = 400

SIDE_INFO = {
    "top": {"outside": (-1, 0), "grow": (1, 0)},
    "bottom": {"outside": (1, 0), "grow": (-1, 0)},
    "left": {"outside": (0, -1), "grow": (0, 1)},
    "right": {"outside": (0, 1), "grow": (0, -1)},
}


def _most_common_color(grid: np.ndarray) -> int:
    return Counter(map(int, grid.reshape(-1))).most_common(1)[0][0]


def _neighbors4(r: int, c: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yield r + dr, c + dc


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _same_color_component(grid: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = grid.shape
    color = int(grid[start])
    comp = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in comp and int(grid[nr, nc]) == color:
                comp.add((nr, nc))
                q.append((nr, nc))
    return comp


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _find_fences(grid: np.ndarray, bg: int) -> list[dict]:
    fences = []
    for comp in _components_mask(grid == GREY):
        pts = set(comp)
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue

        border = _border_cells(r0, c0, r1, c1)
        if not pts <= border:
            continue
        missing = list(border - pts)
        if len(missing) != 1:
            continue
        gap = missing[0]
        if int(grid[gap]) != bg:
            continue

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or np.any(interior == GREY):
            continue
        seeds = np.argwhere(interior != bg)
        if len(seeds) != 1:
            continue
        sr_off, sc_off = map(int, seeds[0])
        sr, sc = r0 + 1 + sr_off, c0 + 1 + sc_off

        gr, gc = gap
        if gr == r0 and c0 < gc < c1:
            side = "top"
        elif gr == r1 and c0 < gc < c1:
            side = "bottom"
        elif gc == c0 and r0 < gr < r1:
            side = "left"
        elif gc == c1 and r0 < gr < r1:
            side = "right"
        else:
            continue

        fences.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "gap": gap,
                "side": side,
                "seed": (sr, sc),
                "seed_color": int(grid[sr, sc]),
            }
        )

    assert fences, "expected at least one valid grey fence"
    return fences


def _find_template(grid: np.ndarray, bg: int, fence: dict) -> dict:
    h, w = grid.shape
    gap_r, gap_c = fence["gap"]
    dr, dc = SIDE_INFO[fence["side"]]["outside"]
    tr, tc = gap_r + dr, gap_c + dc
    assert 0 <= tr < h and 0 <= tc < w, "gap must have an exterior neighbor"

    color = int(grid[tr, tc])
    assert color not in (bg, GREY), "template block must touch the gap exterior"

    comp = _same_color_component(grid, (tr, tc))
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    size_h = r1 - r0 + 1
    size_w = c1 - c0 + 1
    assert size_h == size_w and size_h in BRUSH_SIZES, "template block must be 2x2 or 3x3"
    assert len(comp) == size_h * size_w, "template block must be a solid square"

    if fence["side"] in ("top", "bottom"):
        assert fence["c0"] < c0 <= c1 < fence["c1"]
        assert c0 <= gap_c <= c1
        if fence["side"] == "top":
            assert r1 == fence["r0"] - 1
        else:
            assert r0 == fence["r1"] + 1
    else:
        assert fence["r0"] < r0 <= r1 < fence["r1"]
        assert r0 <= gap_r <= r1
        if fence["side"] == "left":
            assert c1 == fence["c0"] - 1
        else:
            assert c0 == fence["c1"] + 1

    return {"color": color, "r0": r0, "c0": c0, "r1": r1, "c1": c1, "size": size_h}


def _paint_growth(out: np.ndarray, fence: dict, template: dict) -> None:
    seed_r, seed_c = fence["seed"]
    side = fence["side"]

    if side in ("top", "bottom"):
        assert template["c0"] <= seed_c <= template["c1"]
        if side == "top":
            rows = range(seed_r, fence["r1"])
        else:
            rows = range(seed_r, fence["r0"], -1)
        cols = range(template["c0"], template["c1"] + 1)
        for r in rows:
            for c in cols:
                assert fence["r0"] < r < fence["r1"] and fence["c0"] < c < fence["c1"]
                if (r, c) != (seed_r, seed_c):
                    out[r, c] = template["color"]
    else:
        assert template["r0"] <= seed_r <= template["r1"]
        if side == "left":
            cols = range(seed_c, fence["c1"])
        else:
            cols = range(seed_c, fence["c0"], -1)
        rows = range(template["r0"], template["r1"] + 1)
        for c in cols:
            for r in rows:
                assert fence["r0"] < r < fence["r1"] and fence["c0"] < c < fence["c1"]
                if (r, c) != (seed_r, seed_c):
                    out[r, c] = template["color"]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    out = g.copy()
    for fence in _find_fences(g, bg):
        template = _find_template(g, bg, fence)
        _paint_growth(out, fence, template)
    return out


def _choose_object_count(area: int, rng: random.Random) -> int:
    if area < 420:
        return rng.choice([1, 2])
    if area < 600:
        return rng.choices([1, 2, 3], weights=[2, 3, 1], k=1)[0]
    return rng.choices([1, 2, 3], weights=[1, 2, 2], k=1)[0]


def _sample_object(rng: random.Random, seed_color: int, paint_color: int) -> dict:
    side = rng.choice(SIDES)
    brush = rng.choice(BRUSH_SIZES)

    if side in ("top", "bottom"):
        inner_h = rng.randint(INNER_SPAN_RANGE[0], INNER_SPAN_RANGE[1])
        inner_w = rng.randint(max(brush, INNER_SPAN_RANGE[0]), INNER_SPAN_RANGE[1])
        span_start = rng.randint(0, inner_w - brush)
        gap_offset = 1 if brush == 3 else rng.randint(0, 1)
        gap_inner = span_start + gap_offset
        seed_row_inner = rng.randint(0, inner_h - 2) if side == "top" else rng.randint(1, inner_h - 1)
        seed_col_inner = gap_inner
    else:
        inner_h = rng.randint(max(brush, INNER_SPAN_RANGE[0]), INNER_SPAN_RANGE[1])
        inner_w = rng.randint(INNER_SPAN_RANGE[0], INNER_SPAN_RANGE[1])
        span_start = rng.randint(0, inner_h - brush)
        gap_offset = 1 if brush == 3 else rng.randint(0, 1)
        gap_inner = span_start + gap_offset
        seed_row_inner = gap_inner
        seed_col_inner = rng.randint(0, inner_w - 2) if side == "left" else rng.randint(1, inner_w - 1)

    return {
        "side": side,
        "brush": brush,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "span_start": span_start,
        "gap_offset": gap_offset,
        "seed_row_inner": seed_row_inner,
        "seed_col_inner": seed_col_inner,
        "seed_color": seed_color,
        "paint_color": paint_color,
    }


def _union_bbox(top: int, left: int, cfg: dict) -> tuple[int, int, int, int]:
    outer_h = cfg["inner_h"] + 2
    outer_w = cfg["inner_w"] + 2
    r0, c0 = top, left
    r1, c1 = top + outer_h - 1, left + outer_w - 1
    s = cfg["brush"]
    if cfg["side"] == "top":
        return r0 - s, c0, r1, c1
    if cfg["side"] == "bottom":
        return r0, c0, r1 + s, c1
    if cfg["side"] == "left":
        return r0, c0 - s, r1, c1
    return r0, c0, r1, c1 + s


def _draw_object(grid: np.ndarray, bg: int, top: int, left: int, cfg: dict) -> None:
    inner_h = cfg["inner_h"]
    inner_w = cfg["inner_w"]
    r0, c0 = top, left
    r1, c1 = r0 + inner_h + 1, c0 + inner_w + 1

    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    side = cfg["side"]
    span_start = cfg["span_start"]
    brush = cfg["brush"]
    gap_offset = cfg["gap_offset"]

    if side == "top":
        gap_r = r0
        gap_c = c0 + 1 + span_start + gap_offset
        tr0, tr1 = r0 - brush, r0 - 1
        tc0, tc1 = c0 + 1 + span_start, c0 + span_start + brush
    elif side == "bottom":
        gap_r = r1
        gap_c = c0 + 1 + span_start + gap_offset
        tr0, tr1 = r1 + 1, r1 + brush
        tc0, tc1 = c0 + 1 + span_start, c0 + span_start + brush
    elif side == "left":
        gap_r = r0 + 1 + span_start + gap_offset
        gap_c = c0
        tr0, tr1 = r0 + 1 + span_start, r0 + span_start + brush
        tc0, tc1 = c0 - brush, c0 - 1
    else:
        gap_r = r0 + 1 + span_start + gap_offset
        gap_c = c1
        tr0, tr1 = r0 + 1 + span_start, r0 + span_start + brush
        tc0, tc1 = c1 + 1, c1 + brush

    grid[gap_r, gap_c] = bg
    grid[tr0 : tr1 + 1, tc0 : tc1 + 1] = cfg["paint_color"]

    seed_r = r0 + 1 + cfg["seed_row_inner"]
    seed_c = c0 + 1 + cfg["seed_col_inner"]
    grid[seed_r, seed_c] = cfg["seed_color"]


def _place_object(grid: np.ndarray, reserved: np.ndarray, bg: int, cfg: dict, rng: random.Random) -> bool:
    h, w = grid.shape
    outer_h = cfg["inner_h"] + 2
    outer_w = cfg["inner_w"] + 2
    brush = cfg["brush"]
    side = cfg["side"]

    extra_top = brush if side == "top" else 0
    extra_bottom = brush if side == "bottom" else 0
    extra_left = brush if side == "left" else 0
    extra_right = brush if side == "right" else 0

    min_top = extra_top + 1
    max_top = h - outer_h - extra_bottom - 1
    min_left = extra_left + 1
    max_left = w - outer_w - extra_right - 1
    if min_top > max_top or min_left > max_left:
        return False

    for _ in range(PLACEMENT_RETRIES):
        top = rng.randint(min_top, max_top)
        left = rng.randint(min_left, max_left)
        ur0, uc0, ur1, uc1 = _union_bbox(top, left, cfg)
        rr0, rc0 = ur0 - 1, uc0 - 1
        rr1, rc1 = ur1 + 1, uc1 + 1
        if reserved[rr0 : rr1 + 1, rc0 : rc1 + 1].any():
            continue

        _draw_object(grid, bg, top, left, cfg)
        reserved[rr0 : rr1 + 1, rc0 : rc1 + 1] = True
        return True
    return False


def _bg_is_unique_mode(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.reshape(-1)))
    bg_count = counts[bg]
    return sum(1 for count in counts.values() if count == bg_count) == 1


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    area = h * w
    bg = rng.choice([c for c in range(10) if c != GREY])
    n_objects = _choose_object_count(area, rng)

    colors = [c for c in range(10) if c not in (bg, GREY)]
    rng.shuffle(colors)
    if 2 * n_objects > len(colors):
        return None

    grid = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    for i in range(n_objects):
        cfg = _sample_object(rng, colors[2 * i], colors[2 * i + 1])
        if not _place_object(grid, reserved, bg, cfg, rng):
            return None

    if not _bg_is_unique_mode(grid, bg):
        return None

    try:
        fences = _find_fences(grid, bg)
        if len(fences) != n_objects:
            return None
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out != grid) < n_objects:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
