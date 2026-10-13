from __future__ import annotations

import heapq
import random

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
MAROON = 9
EMITTER_COLORS = [1, 2, 3, 4, 6, 8]
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_emitters(grid: np.ndarray) -> list[dict[str, object]]:
    mask = (grid != BG) & (grid != GREY) & (grid != MAROON)
    emitters: list[dict[str, object]] = []
    for coords in _components(mask):
        assert len(coords) == 2, "every emitter must contain exactly two pixels"
        r0, c0, r1, c1 = _bbox(coords)
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "emitter pixels must share one color"
        color = next(iter(colors))
        if r0 == r1 and c1 == c0 + 1:
            emitters.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "color": color,
                    "head": (r0, c1),
                    "dir": (0, 1),
                }
            )
        elif c0 == c1 and r1 == r0 + 1:
            emitters.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "color": color,
                    "head": (r1, c0),
                    "dir": (1, 0),
                }
            )
        else:
            raise AssertionError("emitters must be horizontal 1x2 or vertical 2x1 bars")
    emitters.sort(key=lambda item: tuple(item["bbox"]))
    return emitters


def _find_absorbers(grid: np.ndarray) -> list[dict[str, object]]:
    absorbers: list[dict[str, object]] = []
    for coords in _components(grid == MAROON):
        assert len(coords) == 4, "absorbers must be solid 2x2 squares"
        r0, c0, r1, c1 = _bbox(coords)
        assert r1 == r0 + 1 and c1 == c0 + 1, "absorbers must have a 2x2 bbox"
        expected = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        assert set(coords) == expected, "absorbers must fill their 2x2 bbox"
        absorbers.append({"bbox": (r0, c0, r1, c1), "coords": sorted(expected)})
    absorbers.sort(key=lambda item: tuple(item["bbox"]))
    return absorbers


def _split_dirs(dr: int, dc: int) -> list[tuple[int, int]]:
    if dr == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace emitter rays through prisms and absorbers to recolor their hit paths and targets.

    Input structure:
    - The background is black.
    - Emitters are two-pixel monochrome bars whose orientation encodes direction:
      horizontal bars shoot right and vertical bars shoot downward.
    - Grey singleton pixels are prisms, and maroon 2x2 squares are absorbers.

    Rule:
    - Launch one ray from each emitter and advance all rays in time order one cell at a time.
    - A ray paints black cells with its current color, stops at a grey prism, and recolors
      that prism; only original emitter rays then split into two perpendicular orange rays.
    - Any ray that reaches an absorber recolors the entire 2x2 absorber and stops before
      entering it; later hits on the same absorber do not change its first-hit color.
    - Rays also stop at grid boundaries and at any non-black cell, while split rays hitting
      another prism simply recolor it and stop without splitting again.

    Key insight:
    - Object roles are encoded purely by visible geometry and fixed marker colors: two-pixel
      bars define launch direction, grey singletons convert one ray into two orange rays,
      and maroon 2x2 blocks remember only the earliest incoming ray color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    emitters = _find_emitters(g)
    absorbers = _find_absorbers(g)
    prisms = {tuple(pos) for pos in np.argwhere(g == GREY)}
    assert emitters, "expected at least one emitter"

    covered = np.zeros_like(g, dtype=bool)
    for emitter in emitters:
        r0, c0, r1, c1 = emitter["bbox"]
        covered[r0 : r1 + 1, c0 : c1 + 1] = True
    for r, c in prisms:
        covered[r, c] = True
    for absorber in absorbers:
        for r, c in absorber["coords"]:
            covered[r, c] = True
    assert np.all((g == BG) | covered), "all foreground pixels must be emitters, prisms, or absorbers"

    out = g.copy()
    absorber_of: dict[tuple[int, int], int] = {}
    for idx, absorber in enumerate(absorbers):
        for cell in absorber["coords"]:
            absorber_of[cell] = idx

    activated_absorbers: set[int] = set()
    heap: list[tuple[int, int, int, int, int, int, int, bool]] = []
    seq = 0
    for emitter in emitters:
        head_r, head_c = emitter["head"]
        dr, dc = emitter["dir"]
        heapq.heappush(
            heap,
            (0, seq, int(head_r), int(head_c), int(dr), int(dc), int(emitter["color"]), True),
        )
        seq += 1

    while heap:
        time, _, r, c, dr, dc, color, can_split = heapq.heappop(heap)
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            continue

        if (nr, nc) in absorber_of:
            idx = absorber_of[(nr, nc)]
            if idx not in activated_absorbers:
                for ar, ac in absorbers[idx]["coords"]:
                    out[ar, ac] = color
                activated_absorbers.add(idx)
            continue

        target = int(out[nr, nc])
        if target == BG:
            out[nr, nc] = color
            heapq.heappush(heap, (time + 1, seq, nr, nc, dr, dc, color, can_split))
            seq += 1
            continue

        if target == GREY and (nr, nc) in prisms:
            out[nr, nc] = color
            if can_split:
                for sdr, sdc in _split_dirs(dr, dc):
                    heapq.heappush(heap, (time + 1, seq, nr, nc, sdr, sdc, ORANGE, False))
                    seq += 1
            continue

    return out


def _draw_colors(pool: list[int], count: int) -> list[int]:
    if len(pool) < count:
        refill = EMITTER_COLORS[:]
        random.shuffle(refill)
        for color in refill:
            if color not in pool:
                pool.append(color)
    picked = pool[:count]
    del pool[:count]
    return picked


def _build_direct_box(orientation: str, color: int) -> np.ndarray:
    if orientation == "horizontal":
        h = random.randint(4, 6)
        w = random.randint(7, 9)
        box = np.zeros((h, w), dtype=int)
        row = random.randint(1, h - 2)
        absorber_col = random.randint(4, w - 2)
        absorber_top_options = [top for top in (row - 1, row) if 0 <= top <= h - 2]
        absorber_top = random.choice(absorber_top_options)
        box[row, 1:3] = color
        box[absorber_top : absorber_top + 2, absorber_col : absorber_col + 2] = MAROON
        return box

    h = random.randint(7, 9)
    w = random.randint(4, 6)
    box = np.zeros((h, w), dtype=int)
    col = random.randint(1, w - 2)
    absorber_top = random.randint(4, h - 2)
    absorber_left_options = [left for left in (col - 1, col) if 0 <= left <= w - 2]
    absorber_left = random.choice(absorber_left_options)
    box[1:3, col] = color
    box[absorber_top : absorber_top + 2, absorber_left : absorber_left + 2] = MAROON
    return box


def _build_prism_box(orientation: str, color: int) -> np.ndarray:
    h = random.randint(8, 9)
    w = random.randint(8, 9)
    box = np.zeros((h, w), dtype=int)

    if orientation == "horizontal":
        row = random.randint(2, h - 3)
        prism_col = random.randint(4, w - 4)
        box[row, 1:3] = color
        box[row, prism_col] = GREY

        up_kind = random.choice(["absorber", "prism"])
        if up_kind == "absorber":
            bottom = random.randint(1, row - 1)
            left = random.choice([prism_col - 1, prism_col])
            box[bottom - 1 : bottom + 1, left : left + 2] = MAROON
        else:
            box[random.randint(0, row - 1), prism_col] = GREY

        down_kind = random.choice(["absorber", "prism"])
        if down_kind == "absorber":
            top = random.randint(row + 1, h - 2)
            left = random.choice([prism_col - 1, prism_col])
            box[top : top + 2, left : left + 2] = MAROON
        else:
            box[random.randint(row + 1, h - 1), prism_col] = GREY
        return box

    col = random.randint(2, w - 3)
    prism_row = random.randint(4, h - 4)
    box[1:3, col] = color
    box[prism_row, col] = GREY

    left_kind = random.choice(["absorber", "prism"])
    if left_kind == "absorber":
        right = random.randint(1, col - 1)
        top_options = [top for top in (prism_row - 1, prism_row) if 0 <= top <= h - 2]
        top = random.choice(top_options)
        box[top : top + 2, right - 1 : right + 1] = MAROON
    else:
        box[prism_row, random.randint(0, col - 1)] = GREY

    right_kind = random.choice(["absorber", "prism"])
    if right_kind == "absorber":
        left = random.randint(col + 1, w - 2)
        top_options = [top for top in (prism_row - 1, prism_row) if 0 <= top <= h - 2]
        top = random.choice(top_options)
        box[top : top + 2, left : left + 2] = MAROON
    else:
        box[prism_row, random.randint(col + 1, w - 1)] = GREY
    return box


def _build_shared_box(horizontal_color: int, vertical_color: int) -> np.ndarray:
    h = random.randint(8, 9)
    w = random.randint(8, 9)
    box = np.zeros((h, w), dtype=int)

    absorber_top = random.randint(4, h - 3)
    absorber_left = random.randint(4, w - 3)
    hit_row = random.choice([absorber_top, absorber_top + 1])
    hit_col = random.choice([absorber_left, absorber_left + 1])

    d_h_choices = [d for d in (2, 3, 4) if absorber_left - d - 1 >= 0]
    d_v_choices = [d for d in (2, 3, 4) if absorber_top - d - 1 >= 0]
    pairs = [(dh, dv) for dh in d_h_choices for dv in d_v_choices if dh != dv]
    assert pairs, "shared absorber layout requires distinct arrival times"
    d_h, d_v = random.choice(pairs)

    emitter_col = absorber_left - d_h - 1
    emitter_row = absorber_top - d_v - 1

    box[hit_row, emitter_col : emitter_col + 2] = horizontal_color
    box[emitter_row : emitter_row + 2, hit_col] = vertical_color
    box[absorber_top : absorber_top + 2, absorber_left : absorber_left + 2] = MAROON
    return box


def _build_boundary_box(orientation: str, color: int) -> np.ndarray:
    if orientation == "horizontal":
        h = random.randint(3, 5)
        w = random.randint(6, 9)
        box = np.zeros((h, w), dtype=int)
        row = random.randint(1, h - 2)
        box[row, 1:3] = color
        return box

    h = random.randint(6, 9)
    w = random.randint(3, 5)
    box = np.zeros((h, w), dtype=int)
    col = random.randint(1, w - 2)
    box[1:3, col] = color
    return box


def _quadrant_bounds(h: int, w: int) -> dict[str, tuple[int, int, int, int]]:
    mid_r = h // 2
    mid_c = w // 2
    return {
        "TL": (0, mid_r, 0, mid_c),
        "TR": (0, mid_r, mid_c, w),
        "BL": (mid_r, h, 0, mid_c),
        "BR": (mid_r, h, mid_c, w),
    }


def _candidate_quadrants(spec: dict[str, object]) -> list[str]:
    if spec["kind"] != "boundary":
        return ["TL", "TR", "BL", "BR"]
    if spec["orientation"] == "horizontal":
        return ["TR", "BR"]
    return ["BL", "BR"]


def _assign_quadrants(specs: list[dict[str, object]]) -> dict[int, str] | None:
    order = sorted(range(len(specs)), key=lambda idx: len(_candidate_quadrants(specs[idx])))

    def dfs(pos: int, used: set[str], assignment: dict[int, str]) -> dict[int, str] | None:
        if pos == len(order):
            return assignment.copy()
        spec_idx = order[pos]
        choices = [q for q in _candidate_quadrants(specs[spec_idx]) if q not in used]
        random.shuffle(choices)
        for quad in choices:
            assignment[spec_idx] = quad
            used.add(quad)
            found = dfs(pos + 1, used, assignment)
            if found is not None:
                return found
            used.remove(quad)
            del assignment[spec_idx]
        return None

    return dfs(0, set(), {})


def _place_box_in_quadrant(
    grid: np.ndarray,
    box: np.ndarray,
    quad: str,
    bounds: tuple[int, int, int, int],
    *,
    flush_right: bool = False,
    flush_bottom: bool = False,
) -> bool:
    h, w = box.shape
    r_lo, r_hi, c_lo, c_hi = bounds

    if flush_bottom:
        if quad not in {"BL", "BR"}:
            return False
        row_choices = [grid.shape[0] - h]
    else:
        start = r_lo + 1
        end = r_hi - h - 1
        if start > end:
            return False
        row_choices = list(range(start, end + 1))
        random.shuffle(row_choices)

    if flush_right:
        if quad not in {"TR", "BR"}:
            return False
        col_choices = [grid.shape[1] - w]
    else:
        start = c_lo + 1
        end = c_hi - w - 1
        if start > end:
            return False
        col_choices = list(range(start, end + 1))
        random.shuffle(col_choices)

    for r0 in row_choices:
        for c0 in col_choices:
            if not (r_lo <= r0 and r0 + h <= r_hi and c_lo <= c0 and c0 + w <= c_hi):
                continue
            if np.any((box != BG) & (grid[r0 : r0 + h, c0 : c0 + w] != BG)):
                continue
            grid[r0 : r0 + h, c0 : c0 + w] = box
            return True
    return False


def _spec_box(spec: dict[str, object], color_pool: list[int]) -> np.ndarray:
    kind = spec["kind"]
    orientation = str(spec["orientation"])
    if kind == "prism":
        return _build_prism_box(orientation, _draw_colors(color_pool, 1)[0])
    if kind == "direct":
        return _build_direct_box(orientation, _draw_colors(color_pool, 1)[0])
    if kind == "shared":
        c1, c2 = _draw_colors(color_pool, 2)
        return _build_shared_box(c1, c2)
    if kind == "boundary":
        return _build_boundary_box(orientation, _draw_colors(color_pool, 1)[0])
    raise AssertionError(f"unknown scenario kind: {kind}")


def _build_specs() -> list[dict[str, object]]:
    prism_orientation = random.choice(["horizontal", "vertical"])
    specs: list[dict[str, object]] = [
        {"kind": "prism", "orientation": prism_orientation},
    ]

    if random.random() < 0.5:
        specs.append({"kind": "shared", "orientation": "mixed"})
    else:
        opposite = "vertical" if prism_orientation == "horizontal" else "horizontal"
        specs.append({"kind": "direct", "orientation": opposite})

    if random.random() < 0.65:
        specs.append({"kind": "boundary", "orientation": random.choice(["horizontal", "vertical"])})

    if random.random() < 0.35 and len(specs) < 4:
        specs.append({"kind": "direct", "orientation": random.choice(["horizontal", "vertical"])})

    return specs


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(22, 30)
    w = random.randint(22, 30)
    specs = _build_specs()
    assignment = _assign_quadrants(specs)
    if assignment is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    bounds = _quadrant_bounds(h, w)
    color_pool = EMITTER_COLORS[:]
    random.shuffle(color_pool)

    for idx, spec in enumerate(specs):
        quad = assignment[idx]
        box = _spec_box(spec, color_pool)
        ok = _place_box_in_quadrant(
            grid,
            box,
            quad,
            bounds[quad],
            flush_right=spec["kind"] == "boundary" and spec["orientation"] == "horizontal",
            flush_bottom=spec["kind"] == "boundary" and spec["orientation"] == "vertical",
        )
        if not ok:
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.any(grid == GREY):
        return None
    if not np.any(out == ORANGE):
        return None
    if len(_find_emitters(grid)) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-and-prism ARC input")
