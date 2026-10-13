from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
BAR_COLORS = (1, 2, 3, 4)
DIVIDER = 5
FRAME = 7
SOLID = 8

PANEL_W = 4
WORK_C0 = PANEL_W + 1

DIRS = {
    "tl": (-1, -1),
    "tr": (-1, 1),
    "bl": (1, -1),
    "br": (1, 1),
}


@dataclass(frozen=True)
class Ray:
    name: str
    start: tuple[int, int]
    corner: tuple[int, int]
    direction: tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _find_divider_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER)]
    if len(cols) != 1:
        raise ValueError("expected exactly one full-height divider")
    return int(cols[0])


def _extract_bars(grid: np.ndarray, divider_col: int) -> list[dict[str, int]]:
    panel = grid[:, :divider_col]
    comps = _components4(np.isin(panel, BAR_COLORS))
    bars = []
    for comp in comps:
        vals = {int(panel[r, c]) for r, c in comp}
        if len(vals) != 1:
            raise ValueError("bar component must be monochrome")
        color = vals.pop()
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        if min(cols) != max(cols):
            raise ValueError("bar must be width 1")
        height = max(rows) - min(rows) + 1
        if len(comp) != height:
            raise ValueError("bar must be a solid vertical run")
        bars.append(
            {
                "color": int(color),
                "top": int(min(rows)),
                "left": int(min(cols)),
                "height": int(height),
            }
        )
    bars.sort(key=lambda bar: (bar["top"], bar["left"]))
    if not (1 <= len(bars) <= 4):
        raise ValueError("expected 1-4 bars")
    return bars


def _parse_geometry(grid: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], set[tuple[int, int]]]:
    solid_box = _bbox(grid == SOLID)
    sr0, sc0, sr1, sc1 = solid_box
    if not np.all(grid[sr0 : sr1 + 1, sc0 : sc1 + 1] == SOLID):
        raise ValueError("solid object must be a filled rectangle")

    frame_box = _bbox(grid == FRAME)
    fr0, fc0, fr1, fc1 = frame_box
    expected = np.zeros_like(grid, dtype=bool)
    expected[fr0, fc0 : fc1 + 1] = True
    expected[fr1, fc0 : fc1 + 1] = True
    expected[fr0 : fr1 + 1, fc0] = True
    expected[fr0 : fr1 + 1, fc1] = True
    if not np.array_equal(grid == FRAME, expected):
        raise ValueError("frame must be a single hollow rectangle")

    corners = {(fr0, fc0), (fr0, fc1), (fr1, fc0), (fr1, fc1)}
    return solid_box, frame_box, corners


def _corner_rays(solid_box: tuple[int, int, int, int], frame_box: tuple[int, int, int, int]) -> list[Ray]:
    sr0, sc0, sr1, sc1 = solid_box
    fr0, fc0, fr1, fc1 = frame_box
    return [
        Ray("tl", (sr0, sc0), (fr0, fc0), DIRS["tl"]),
        Ray("tr", (sr0, sc1), (fr0, fc1), DIRS["tr"]),
        Ray("bl", (sr1, sc0), (fr1, fc0), DIRS["bl"]),
        Ray("br", (sr1, sc1), (fr1, fc1), DIRS["br"]),
    ]


def _first_non_bg_non_solid(grid: np.ndarray, start: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int] | None:
    r, c = start
    dr, dc = direction
    r += dr
    c += dc
    while _in_bounds(grid.shape, r, c):
        value = int(grid[r, c])
        if value not in (BG, SOLID):
            return (r, c)
        r += dr
        c += dc
    return None


def _active_rays(grid: np.ndarray) -> tuple[list[Ray], list[dict[str, int]]]:
    divider_col = _find_divider_col(grid)
    bars = _extract_bars(grid, divider_col)
    solid_box, frame_box, corners = _parse_geometry(grid)

    rays = []
    for ray in _corner_rays(solid_box, frame_box):
        hit = _first_non_bg_non_solid(grid, ray.start, ray.direction)
        if hit == ray.corner and hit in corners:
            rays.append(ray)
    rays.sort(key=lambda ray: (ray.corner[0], ray.corner[1]))
    if len(rays) != len(bars):
        raise ValueError("bar count must match active ray count")
    return rays, bars


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend gated corner diagonals by depths encoded in a left reference panel.

    Input structure:
    - The background is black.
    - A full-height grey divider splits a 4-column left reference panel from the
      right work area.
    - The left panel contains 1-4 single-column bars in blue/red/green/yellow,
      while the work area contains one cyan solid rectangle and one orange hollow
      rectangle frame around it.

    Rule:
    - From the four outer corners of the cyan rectangle, trace the outward 45
      degree diagonal.
    - A ray is active only when its first non-black, non-cyan hit is the matching
      frame corner; otherwise that ray does nothing.
    - Sort active rays by the crossed frame-corner position in row-major order,
      and sort the reference bars by their top row.
    - For each matched pair, start one step beyond the frame corner and paint
      exactly as many cyan pixels along the same diagonal as the bar's height.

    Key insight:
    - Three separate structures interact: the solid rectangle chooses ray starts,
      the frame corners gate which diagonals activate, and the left panel encodes
      the continuation depth only through sorted bar order.
    """
    g = np.asarray(grid, dtype=int)
    rays, bars = _active_rays(g)
    out = g.copy()
    for ray, bar in zip(rays, bars):
        r, c = ray.corner
        dr, dc = ray.direction
        for step in range(1, bar["height"] + 1):
            rr = r + dr * step
            cc = c + dc * step
            if not _in_bounds(g.shape, rr, cc):
                raise ValueError("encoded ray length exceeds grid")
            if out[rr, cc] != BG:
                raise ValueError("ray extension must stay on background")
            out[rr, cc] = SOLID
    return out


def _signature_from_gaps(gaps: tuple[int, int, int, int]) -> tuple[bool, bool, bool, bool]:
    gt, gb, gl, gr = gaps
    return (gt == gl, gt == gr, gb == gl, gb == gr)


def _build_signature_table() -> dict[tuple[bool, bool, bool, bool], list[tuple[int, int, int, int]]]:
    table: dict[tuple[bool, bool, bool, bool], list[tuple[int, int, int, int]]] = {}
    for gt in range(1, 6):
        for gb in range(1, 6):
            for gl in range(1, 6):
                for gr in range(1, 6):
                    sig = _signature_from_gaps((gt, gb, gl, gr))
                    count = sum(sig)
                    if count in (1, 2, 4):
                        table.setdefault(sig, []).append((gt, gb, gl, gr))
    return table


SIGNATURE_TABLE = _build_signature_table()
SIGNATURES_BY_COUNT = {
    count: [sig for sig in SIGNATURE_TABLE if sum(sig) == count]
    for count in (1, 2, 4)
}


def _choose_partition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    result = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        result.append(cut - prev - 1)
        prev = cut
    return result


def _place_bars(grid: np.ndarray, heights: list[int]) -> list[dict[str, int]]:
    h = grid.shape[0]
    count = len(heights)
    min_used = sum(heights) + max(0, count - 1)
    slack = h - min_used
    if slack < 0:
        raise ValueError("panel too short for bars")
    extras = _choose_partition(slack, count + 1)
    row = extras[0]
    bars = []
    colors = list(BAR_COLORS)
    random.shuffle(colors)
    chosen_colors = colors[:count]
    for idx, height in enumerate(heights):
        col = random.randrange(PANEL_W)
        color = chosen_colors[idx]
        grid[row : row + height, col] = color
        bars.append({"color": color, "top": row, "left": col, "height": height})
        row += height
        if idx < count - 1:
            row += 1 + extras[idx + 1]
    return bars


def _paint_expected_from_params(
    grid: np.ndarray,
    active_order: list[str],
    heights_by_ray: dict[str, int],
    frame_corners: dict[str, tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    for name in active_order:
        r, c = frame_corners[name]
        dr, dc = DIRS[name]
        for step in range(1, heights_by_ray[name] + 1):
            rr = r + dr * step
            cc = c + dc * step
            if out[rr, cc] != BG:
                raise ValueError("expected trail overlaps foreground")
            out[rr, cc] = SOLID
    return out


def _semantic_check(
    grid: np.ndarray,
    expected: np.ndarray,
    active_signature: tuple[bool, bool, bool, bool],
    heights_by_ray: dict[str, int],
) -> bool:
    try:
        divider_col = _find_divider_col(grid)
        if divider_col != PANEL_W:
            return False
        bars = _extract_bars(grid, divider_col)
        solid_box, frame_box, _corners = _parse_geometry(grid)
        rays = _corner_rays(solid_box, frame_box)
        active_names = []
        for flag, ray in zip(active_signature, rays):
            hit = _first_non_bg_non_solid(grid, ray.start, ray.direction)
            if flag:
                if hit != ray.corner:
                    return False
                active_names.append(ray.name)
            else:
                if hit == ray.corner:
                    return False
        active_names.sort(key=lambda name: (
            frame_box[0] if name in ("tl", "tr") else frame_box[2],
            frame_box[1] if name in ("tl", "bl") else frame_box[3],
        ))
        if len(bars) != len(active_names):
            return False
        if [bar["height"] for bar in bars] != [heights_by_ray[name] for name in active_names]:
            return False
        if not np.array_equal(main(grid), expected):
            return False
        return True
    except Exception:
        return False


def _generate_once() -> np.ndarray:
    target_count = random.choices([1, 2, 4], weights=[4, 4, 2], k=1)[0]
    signature = random.choice(SIGNATURES_BY_COUNT[target_count])

    for _ in range(400):
        solid_h = random.randint(3, 5)
        solid_w = random.randint(3, 5)
        gap_options = [
            gaps
            for gaps in SIGNATURE_TABLE[signature]
            if 7 <= solid_h + gaps[0] + gaps[1] <= 11 and 7 <= solid_w + gaps[2] + gaps[3] <= 11
        ]
        if not gap_options:
            continue
        gt, gb, gl, gr = random.choice(gap_options)

        inner_h = solid_h + gt + gb
        inner_w = solid_w + gl + gr
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        h = random.randint(max(16, outer_h + 2), min(28, 30))
        w = random.randint(max(20, WORK_C0 + outer_w + 2), 30)

        vertical_slack = h - outer_h
        horizontal_slack = w - WORK_C0 - outer_w
        if vertical_slack < 2 or horizontal_slack < 2:
            continue

        mt = random.randint(1, vertical_slack - 1)
        mb = vertical_slack - mt
        ml = random.randint(1, horizontal_slack - 1)
        mr = horizontal_slack - ml

        frame_r0 = mt
        frame_c0 = WORK_C0 + ml
        frame_r1 = frame_r0 + outer_h - 1
        frame_c1 = frame_c0 + outer_w - 1

        solid_r0 = frame_r0 + 1 + gt
        solid_c0 = frame_c0 + 1 + gl
        solid_r1 = solid_r0 + solid_h - 1
        solid_c1 = solid_c0 + solid_w - 1

        frame_corners = {
            "tl": (frame_r0, frame_c0),
            "tr": (frame_r0, frame_c1),
            "bl": (frame_r1, frame_c0),
            "br": (frame_r1, frame_c1),
        }
        available = {
            "tl": min(mt, ml),
            "tr": min(mt, mr),
            "bl": min(mb, ml),
            "br": min(mb, mr),
        }
        active_names = [name for flag, name in zip(signature, ("tl", "tr", "bl", "br")) if flag]
        active_names.sort(key=lambda name: (frame_corners[name][0], frame_corners[name][1]))
        if any(available[name] < 1 for name in active_names):
            continue

        heights = [random.randint(1, min(5, available[name])) for name in active_names]
        if len(active_names) > 1 and random.random() < 0.7 and len(set(heights)) == 1:
            continue
        if sum(heights) + max(0, len(heights) - 1) > h:
            continue
        heights_by_ray = {name: height for name, height in zip(active_names, heights)}

        grid = np.zeros((h, w), dtype=int)
        grid[:, PANEL_W] = DIVIDER
        grid[frame_r0, frame_c0 : frame_c1 + 1] = FRAME
        grid[frame_r1, frame_c0 : frame_c1 + 1] = FRAME
        grid[frame_r0 : frame_r1 + 1, frame_c0] = FRAME
        grid[frame_r0 : frame_r1 + 1, frame_c1] = FRAME
        grid[solid_r0 : solid_r1 + 1, solid_c0 : solid_c1 + 1] = SOLID

        bars = _place_bars(grid, heights)
        bars.sort(key=lambda bar: (bar["top"], bar["left"]))
        if [bar["height"] for bar in bars] != heights:
            continue

        expected = _paint_expected_from_params(grid, active_names, heights_by_ray, frame_corners)
        if np.array_equal(expected, grid):
            continue
        if _semantic_check(grid, expected, signature, heights_by_ray):
            return grid
    raise RuntimeError("failed to build valid instance")


def generate_input() -> np.ndarray:
    for _ in range(300):
        try:
            return _generate_once()
        except Exception:
            pass
    raise RuntimeError("failed to generate puzzle input")
