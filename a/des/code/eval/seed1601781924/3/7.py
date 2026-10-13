import itertools
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
AZURE = 8
MAROON = 9

STAMP_COLORS = [1, 2, 3, 6, 7]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTION_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    shifted = sorted((r - r0, c - c0) for r, c in coords)
    return shifted, int(r1 - r0 + 1), int(c1 - c0 + 1)


def _min_manhattan(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    best = 10**9
    for ar, ac in a:
        for br, bc in b:
            best = min(best, abs(int(ar - br)) + abs(int(ac - bc)))
    return int(best)


def _best_assignment(cost: list[list[int]]) -> tuple[int, ...]:
    n = len(cost)
    best_perm: tuple[int, ...] | None = None
    best_total: int | None = None
    for perm in itertools.permutations(range(n)):
        total = sum(int(cost[i][perm[i]]) for i in range(n))
        if best_total is None or total < best_total:
            best_total = int(total)
            best_perm = tuple(int(x) for x in perm)
    if best_perm is None:
        raise ValueError("empty assignment matrix")
    return best_perm


def _decode_arrow(
    coords: list[tuple[int, int]],
    stamp_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[str, int]:
    pts = {(int(r), int(c)) for r, c in coords}
    row_counts: dict[int, int] = {}
    col_counts: dict[int, int] = {}
    for r, c in pts:
        row_counts[r] = row_counts.get(r, 0) + 1
        col_counts[c] = col_counts.get(c, 0) + 1

    bend_candidates = [(r, c) for r, c in pts if row_counts[r] > 1 and col_counts[c] > 1]
    if not bend_candidates:
        raise ValueError("arrow must have a bend")
    bend = min(bend_candidates)

    horizontal = sorted((r, c) for r, c in pts if r == bend[0])
    vertical = sorted((r, c) for r, c in pts if c == bend[1])
    h_len = len(horizontal)
    v_len = len(vertical)

    if h_len < v_len:
        tip = next(cell for cell in horizontal if cell != bend)
        direction = "right" if tip[1] > bend[1] else "left"
        steps = max(1, int(v_len - 2))
        return direction, steps
    if v_len < h_len:
        tip = next(cell for cell in vertical if cell != bend)
        direction = "down" if tip[0] > bend[0] else "up"
        steps = max(1, int(h_len - 2))
        return direction, steps

    deg1 = []
    for r, c in pts:
        degree = sum((nr, nc) in pts for nr, nc in _neighbors4(int(r), int(c)))
        if degree == 1:
            deg1.append((int(r), int(c)))
    if len(deg1) != 2:
        raise ValueError("ambiguous arrow endpoints")

    if stamp_bbox is not None:
        sr0, sc0, sr1, sc1 = stamp_bbox
        cr = (sr0 + sr1) / 2.0
        cc = (sc0 + sc1) / 2.0
        tip = max(deg1, key=lambda cell: (abs(cell[0] - cr) + abs(cell[1] - cc), cell))
    else:
        tip = max(deg1)

    if tip[0] == bend[0]:
        direction = "right" if tip[1] > bend[1] else "left"
    else:
        direction = "down" if tip[0] > bend[0] else "up"
    return direction, 1


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    divider_rows = np.where(np.all(g == GREY, axis=1))[0]
    if divider_rows.size == 0:
        raise ValueError("missing grey divider row")
    divider = int(divider_rows[0])

    upper = g[:divider, :]
    lower = g[divider + 1 :, :]

    frame_points = np.argwhere(lower == MAROON)
    if frame_points.size == 0:
        raise ValueError("missing maroon frame")
    fr0, fc0 = np.min(frame_points, axis=0)
    fr1, fc1 = np.max(frame_points, axis=0)
    fr0, fc0, fr1, fc1 = int(fr0), int(fc0), int(fr1), int(fc1)

    out = np.where(lower[fr0 : fr1 + 1, fc0 : fc1 + 1] == MAROON, MAROON, BLACK)
    out_h, out_w = out.shape

    yellow_points = np.argwhere(lower == YELLOW)
    if yellow_points.size == 0:
        raise ValueError("missing yellow origin")
    origin = None
    for rr, cc in yellow_points:
        rr = int(rr)
        cc = int(cc)
        if fr0 <= rr <= fr1 and fc0 <= cc <= fc1:
            origin = (rr - fr0, cc - fc0)
            break
    if origin is None:
        raise ValueError("yellow origin not inside frame")
    origin_r, origin_c = origin

    stamp_mask = (~np.isin(upper, [BLACK, GREY, AZURE]))
    stamp_comps = _components(stamp_mask)
    arrow_comps = _components(upper == AZURE)
    if not stamp_comps or len(stamp_comps) != len(arrow_comps):
        raise ValueError("stamps/arrows must match one-to-one")

    stamps = []
    for comp in stamp_comps:
        colors = {int(upper[r, c]) for r, c in comp}
        if len(colors) != 1:
            raise ValueError("each stamp must be monochrome")
        r0, c0, r1, c1 = _bbox(comp)
        rel, h, w = _normalize(comp)
        stamps.append(
            {
                "coords": comp,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "rel": rel,
                "height": int(h),
                "width": int(w),
                "color": int(next(iter(colors))),
            }
        )

    stamps.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    arrows = [sorted(comp) for comp in arrow_comps]

    costs = [[_min_manhattan(stamp["coords"], arrow) for arrow in arrows] for stamp in stamps]
    assignment = _best_assignment(costs)

    for i, stamp in enumerate(stamps):
        direction, steps = _decode_arrow(arrows[assignment[i]], stamp["bbox"])
        dr = 0
        dc = 0
        if direction == "up":
            dr = -steps * stamp["height"]
        elif direction == "down":
            dr = steps * stamp["height"]
        elif direction == "left":
            dc = -steps * stamp["width"]
        elif direction == "right":
            dc = steps * stamp["width"]
        else:
            raise ValueError(f"unknown direction {direction}")

        top = origin_r + dr
        left = origin_c + dc
        for rr, cc in stamp["rel"]:
            r = top + rr
            c = left + cc
            if 0 < r < out_h - 1 and 0 < c < out_w - 1:
                out[r, c] = stamp["color"]

    return out


def _transform_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    transformed = list(coords)
    for _ in range(random.randint(0, 3)):
        transformed = [(c, -r) for r, c in transformed]
    if random.choice([False, True]):
        transformed = [(r, -c) for r, c in transformed]
    if random.choice([False, True]):
        transformed = [(-r, c) for r, c in transformed]
    return _normalize(transformed)[0]


def _random_shape() -> tuple[list[tuple[int, int]], int, int]:
    for _ in range(100):
        target = random.randint(3, 7)
        pts = {(0, 0)}
        while len(pts) < target:
            br, bc = random.choice(tuple(pts))
            dr, dc = random.choice(DIR4)
            pts.add((br + dr, bc + dc))
        coords = _transform_coords(sorted(pts))
        coords, h, w = _normalize(coords)
        if h <= 4 and w <= 4 and not (h == 1 and w == 1):
            return coords, h, w
    raise RuntimeError("failed to build random stamp shape")


def _make_arrow(direction: str, steps: int) -> tuple[list[tuple[int, int]], int, int]:
    shaft_len = steps + 2
    tail_sign = random.choice([-1, 1])
    if direction in ("left", "right"):
        pts = [(tail_sign * k, 0) for k in range(shaft_len)]
        pts.append((0, 1 if direction == "right" else -1))
    else:
        pts = [(0, tail_sign * k) for k in range(shaft_len)]
        pts.append((1 if direction == "down" else -1, 0))
    return _normalize(pts)


def _build_group(
    stamp_coords: list[tuple[int, int]],
    stamp_color: int,
    direction: str,
    steps: int,
) -> np.ndarray:
    stamp_coords, stamp_h, stamp_w = _normalize(stamp_coords)
    arrow_coords, arrow_h, arrow_w = _make_arrow(direction, steps)
    side = random.choice(["top", "bottom", "left", "right"])
    gap = random.choice([0, 1])

    if side in ("top", "bottom"):
        total_h = stamp_h + arrow_h + gap
        total_w = max(stamp_w, arrow_w) + random.randint(0, 2)
        stamp_c = random.randint(0, total_w - stamp_w)
        arrow_c = random.randint(0, total_w - arrow_w)
        if side == "top":
            arrow_r, stamp_r = 0, arrow_h + gap
        else:
            stamp_r, arrow_r = 0, stamp_h + gap
    else:
        total_h = max(stamp_h, arrow_h) + random.randint(0, 2)
        total_w = stamp_w + arrow_w + gap
        stamp_r = random.randint(0, total_h - stamp_h)
        arrow_r = random.randint(0, total_h - arrow_h)
        if side == "left":
            arrow_c, stamp_c = 0, arrow_w + gap
        else:
            stamp_c, arrow_c = 0, stamp_w + gap

    group = np.full((total_h, total_w), BLACK, dtype=int)
    for r, c in stamp_coords:
        group[stamp_r + r, stamp_c + c] = stamp_color
    for r, c in arrow_coords:
        group[arrow_r + r, arrow_c + c] = AZURE

    points = np.argwhere(group != BLACK)
    r0, c0 = np.min(points, axis=0)
    r1, c1 = np.max(points, axis=0)
    return group[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _stamp_delta(direction: str, steps: int, h: int, w: int) -> tuple[int, int]:
    if direction == "up":
        return -steps * h, 0
    if direction == "down":
        return steps * h, 0
    if direction == "left":
        return 0, -steps * w
    if direction == "right":
        return 0, steps * w
    raise ValueError(f"unknown direction {direction}")


def _expected_output(stamps: list[dict], interior_h: int, interior_w: int, origin_r: int, origin_c: int) -> np.ndarray:
    out = np.full((interior_h + 2, interior_w + 2), BLACK, dtype=int)
    out[0, :] = MAROON
    out[-1, :] = MAROON
    out[:, 0] = MAROON
    out[:, -1] = MAROON

    base_r = origin_r + 1
    base_c = origin_c + 1
    for stamp in stamps:
        dr, dc = _stamp_delta(stamp["direction"], stamp["steps"], stamp["height"], stamp["width"])
        top = base_r + dr
        left = base_c + dc
        for rr, cc in stamp["coords"]:
            out[top + rr, left + cc] = stamp["color"]
    return out


def _build_lower_input(interior_h: int, interior_w: int, origin_r: int, origin_c: int) -> np.ndarray:
    lower = np.full((interior_h + 2, interior_w + 2), BLACK, dtype=int)
    lower[0, :] = MAROON
    lower[-1, :] = MAROON
    lower[:, 0] = MAROON
    lower[:, -1] = MAROON
    lower[origin_r + 1, origin_c + 1] = YELLOW
    return lower


def _gen_attempt() -> np.ndarray | None:
    stamp_count = random.randint(2, 5)
    colors = random.sample(STAMP_COLORS, stamp_count)
    directions = [random.choice(list(DIRECTION_VECTORS)) for _ in range(stamp_count)]
    steps = [random.randint(1, 3) for _ in range(stamp_count)]

    stamps: list[dict] = []
    for i in range(stamp_count):
        coords, h, w = _random_shape()
        stamps.append(
            {
                "coords": coords,
                "height": int(h),
                "width": int(w),
                "color": int(colors[i]),
                "direction": directions[i],
                "steps": int(steps[i]),
            }
        )

    row_tops = []
    row_bottoms = []
    col_lefts = []
    col_rights = []
    for stamp in stamps:
        dr, dc = _stamp_delta(stamp["direction"], stamp["steps"], stamp["height"], stamp["width"])
        row_tops.append(int(dr))
        row_bottoms.append(int(dr + stamp["height"]))
        col_lefts.append(int(dc))
        col_rights.append(int(dc + stamp["width"]))

    pad_top = random.randint(1, 3)
    pad_bottom = random.randint(1, 3)
    pad_left = random.randint(1, 3)
    pad_right = random.randint(1, 3)
    interior_h = int(max(row_bottoms) - min(row_tops) + pad_top + pad_bottom)
    interior_w = int(max(col_rights) - min(col_lefts) + pad_left + pad_right)
    origin_r = int(pad_top - min(row_tops))
    origin_c = int(pad_left - min(col_lefts))

    expected = _expected_output(stamps, interior_h, interior_w, origin_r, origin_c)

    groups = [_build_group(stamp["coords"], stamp["color"], stamp["direction"], stamp["steps"]) for stamp in stamps]
    gx = random.randint(1, 2)
    gy = random.randint(1, 2)
    group_positions = []
    max_upper_r = 0
    max_upper_c = 0
    for group in groups:
        gh, gw = group.shape
        group_positions.append((int(gy), int(gx), group))
        max_upper_r = max(max_upper_r, int(gy + gh))
        max_upper_c = max(max_upper_c, int(gx + gw))
        gx += gw + random.randint(2, 4)
        gy += random.randint(0, 2)
    upper_h = max_upper_r + random.randint(1, 2)
    upper_w = max_upper_c + random.randint(1, 2)

    lower = _build_lower_input(interior_h, interior_w, origin_r, origin_c)
    lower_h, lower_w = lower.shape
    grid_w = max(upper_w, lower_w + 4)
    lower_top = random.randint(1, 3)
    lower_bottom = random.randint(1, 2)
    grid_h = upper_h + 1 + lower_top + lower_h + lower_bottom
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    for r, c, group in group_positions:
        gh, gw = group.shape
        grid[r : r + gh, c : c + gw] = group

    divider = upper_h
    grid[divider, :] = GREY
    frame_left = random.randint(1, grid_w - lower_w - 1)
    frame_top = divider + 1 + lower_top
    grid[frame_top : frame_top + lower_h, frame_left : frame_left + lower_w] = lower

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
