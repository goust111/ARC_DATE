import random
from collections import deque

import numpy as np


BACKGROUND = 0
FRAME = 1
COUNT_TO_FILL = {1: 2, 2: 3, 3: 4, 4: 5}
SIDES = ("top", "bottom", "left", "right")
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
MARKER_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
FRAME_ATTEMPTS = 120


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    comps.sort(key=lambda pts: _bbox(pts))
    return comps


def _frame_border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }


def _ray_cells(shape: tuple[int, int], gap: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    rr, cc = gap[0] + dr, gap[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= rr < h and 0 <= cc < w:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = _find_color_components(g, FRAME)
    frames: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            raise AssertionError("frame bbox too small")
        coords = set(comp)
        border = _frame_border_cells(r0, c0, r1, c1)
        if not coords.issubset(border):
            raise AssertionError("blue component must lie on a rectangular border")
        gaps = list(border - coords)
        if len(gaps) != 1:
            raise AssertionError("expected exactly one missing border pixel")
        gap = gaps[0]
        gap_r, gap_c = gap
        if gap_r == r0:
            if not (c0 < gap_c < c1):
                raise AssertionError("gap cannot be on a corner")
            side = "top"
        elif gap_r == r1:
            if not (c0 < gap_c < c1):
                raise AssertionError("gap cannot be on a corner")
            side = "bottom"
        elif gap_c == c0:
            if not (r0 < gap_r < r1):
                raise AssertionError("gap cannot be on a corner")
            side = "left"
        elif gap_c == c1:
            if not (r0 < gap_r < r1):
                raise AssertionError("gap cannot be on a corner")
            side = "right"
        else:
            raise AssertionError("gap must lie on the frame border")

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        count = int(np.count_nonzero(interior != BACKGROUND))
        if count not in COUNT_TO_FILL:
            raise AssertionError("interior non-background count must be between 1 and 4")

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "side": side,
                "direction": SIDE_TO_DIR[side],
                "count": count,
                "fill": COUNT_TO_FILL[count],
                "ray": _ray_cells(g.shape, gap, SIDE_TO_DIR[side]),
            }
        )
    if not frames:
        raise AssertionError("expected at least one blue frame")
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the markers inside each one-gap blue rectangle, fill its interior by a fixed count-to-color map, and emit a same-color ray through the gap.

    Input structure:
    - The grid has a black background and 3-4 separated blue rectangular frame borders, each missing exactly one non-corner border pixel.
    - Every frame encloses a rectangular interior containing 1-4 scattered non-black marker pixels whose colors are irrelevant except for being countable.
    - Different frames may have different sizes and different gap sides.

    Rule:
    - For each blue one-gap rectangular frame, count the interior non-black pixels N.
    - Map N to the fill color by 1->red(2), 2->green(3), 3->yellow(4), 4->grey(5).
    - Fill the whole interior with that color, restore the missing border pixel to blue, and draw a straight ray of the same fill color from the cell just outside the gap to the grid edge.

    Key insight:
    - Two independent clues must be combined: the interior pixel count determines the color, while the gap side determines the ray direction.
    """
    g = np.asarray(grid, dtype=int)
    frames = _extract_frames(g)
    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(frame["fill"])
        gap_r, gap_c = frame["gap"]
        out[gap_r, gap_c] = FRAME
        for rr, cc in frame["ray"]:
            out[rr, cc] = int(frame["fill"])
    return out


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _gap_for_bbox(r0: int, c0: int, h: int, w: int, side: str, offset: int) -> tuple[int, int]:
    if side == "top":
        return r0, c0 + offset
    if side == "bottom":
        return r0 + h - 1, c0 + offset
    if side == "left":
        return r0 + offset, c0
    return r0 + offset, c0 + w - 1


def _interior_cells(meta: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = meta["bbox"]
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _choose_cluster_positions(cells: list[tuple[int, int]], count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    cell_set = set(cells)
    for _ in range(80):
        start = rng.choice(cells)
        chosen = [start]
        frontier = [start]
        chosen_set = {start}
        while frontier and len(chosen) < count:
            cur = frontier.pop(rng.randrange(len(frontier)))
            nbrs = [p for p in neighbors4(*cur) if p in cell_set and p not in chosen_set]
            rng.shuffle(nbrs)
            for nxt in nbrs:
                chosen.append(nxt)
                frontier.append(nxt)
                chosen_set.add(nxt)
                if len(chosen) == count:
                    break
        if len(chosen) == count:
            return chosen
    return None


def _choose_separated_positions(cells: list[tuple[int, int]], count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    for _ in range(120):
        chosen = rng.sample(cells, count)
        ok = True
        for i, (r0, c0) in enumerate(chosen):
            for r1, c1 in chosen[i + 1 :]:
                if abs(r0 - r1) + abs(c0 - c1) == 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return chosen
    return None


def _choose_marker_positions(cells: list[tuple[int, int]], count: int, mode: str, rng: random.Random) -> list[tuple[int, int]] | None:
    if count == 1:
        return [rng.choice(cells)]
    if mode == "cluster":
        return _choose_cluster_positions(cells, count, rng)
    if mode == "separated":
        return _choose_separated_positions(cells, count, rng)
    for _ in range(60):
        chosen = rng.sample(cells, count)
        if len(set(chosen)) == count:
            return chosen
    return None


def _marker_colors(count: int, mode: str, shared: int, rng: random.Random) -> list[int]:
    if count == 1:
        return [shared]
    pool = [c for c in MARKER_COLORS if c != shared]
    if mode == "cluster":
        return [shared] * count
    colors = [shared]
    if mode == "separated":
        rng.shuffle(pool)
        while len(colors) < count:
            colors.append(pool[(len(colors) - 1) % len(pool)])
        rng.shuffle(colors)
        return colors
    while len(colors) < count:
        colors.append(rng.choice(pool + [shared]))
    rng.shuffle(colors)
    return colors


def _has_size_count_inversion(metas: list[dict]) -> bool:
    for i, a in enumerate(metas):
        area_a = (a["bbox"][2] - a["bbox"][0] - 1) * (a["bbox"][3] - a["bbox"][1] - 1)
        for b in metas[i + 1 :]:
            area_b = (b["bbox"][2] - b["bbox"][0] - 1) * (b["bbox"][3] - b["bbox"][1] - 1)
            if area_a > area_b and a["count"] < b["count"]:
                return True
            if area_b > area_a and b["count"] < a["count"]:
                return True
    return False


def _expected_output(grid: np.ndarray, metas: list[dict]) -> np.ndarray:
    out = grid.copy()
    for meta in metas:
        r0, c0, r1, c1 = meta["bbox"]
        fill = COUNT_TO_FILL[int(meta["count"])]
        out[r0 + 1 : r1, c0 + 1 : c1] = fill
        gap_r, gap_c = meta["gap"]
        out[gap_r, gap_c] = FRAME
        for rr, cc in meta["ray"]:
            out[rr, cc] = fill
    return out


def _validate_generated(grid: np.ndarray, metas: list[dict]) -> bool:
    parsed = _extract_frames(grid)
    if len(parsed) != len(metas):
        return False

    by_bbox = {meta["bbox"]: meta for meta in metas}
    parsed_by_bbox = {meta["bbox"]: meta for meta in parsed}
    if set(by_bbox) != set(parsed_by_bbox):
        return False

    counts = sorted(int(meta["count"]) for meta in parsed)
    expected_counts = [1, 2, 3] if len(parsed) == 3 else [1, 2, 3, 4]
    if counts != expected_counts:
        return False

    if len({meta["side"] for meta in parsed}) < 2:
        return False
    if not _has_size_count_inversion(parsed):
        return False

    ray_cells: set[tuple[int, int]] = set()
    for meta in parsed:
        for rr, cc in meta["ray"]:
            if (rr, cc) in ray_cells:
                return False
            ray_cells.add((rr, cc))
            if int(grid[rr, cc]) != BACKGROUND:
                return False

    marker_sets = {meta["bbox"]: _marker_points(grid, meta) for meta in parsed}
    has_cluster = False
    has_separated = False
    for meta in parsed:
        pts = marker_sets[meta["bbox"]]
        if meta["count"] < 2:
            continue
        adjacent = any(
            abs(r0 - r1) + abs(c0 - c1) == 1
            for i, (r0, c0) in enumerate(pts)
            for r1, c1 in pts[i + 1 :]
        )
        if adjacent:
            has_cluster = True
        if meta["count"] >= 3 and not adjacent:
            has_separated = True
    if not has_cluster:
        return False
    if not has_separated:
        return False

    out = main(grid)
    if not np.array_equal(out, _expected_output(grid, metas)):
        return False
    return True


def _marker_points(grid: np.ndarray, meta: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = meta["bbox"]
    pts = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(grid[r, c]) != BACKGROUND:
                pts.append((r, c))
    return pts


def _place_frames(rng: random.Random) -> list[dict] | None:
    h = rng.randint(18, 24)
    w = rng.randint(18, 24)
    frame_count = rng.choice([3, 4])

    sides = [rng.choice(SIDES) for _ in range(frame_count)]
    if len(set(sides)) < 2:
        return None

    metas: list[dict] = []
    expanded_boxes: list[tuple[int, int, int, int]] = []
    occupied: set[tuple[int, int]] = set()

    for side in sides:
        placed = False
        for _ in range(FRAME_ATTEMPTS):
            fh = rng.randint(5, 7)
            fw = rng.randint(5, 7)
            r0 = rng.randint(1, h - fh - 1)
            c0 = rng.randint(1, w - fw - 1)
            r1 = r0 + fh - 1
            c1 = c0 + fw - 1
            offset = rng.randint(1, (fw - 2) if side in ("top", "bottom") else (fh - 2))
            gap = _gap_for_bbox(r0, c0, fh, fw, side, offset)
            ray = _ray_cells((h, w), gap, SIDE_TO_DIR[side])
            if not ray:
                continue

            bbox = (r0, c0, r1, c1)
            expanded = (r0 - 1, c0 - 1, r1 + 1, c1 + 1)
            if any(_rects_overlap(expanded, other) for other in expanded_boxes):
                continue

            expanded_cells = {
                (rr, cc)
                for rr in range(expanded[0], expanded[2] + 1)
                for cc in range(expanded[1], expanded[3] + 1)
                if in_bounds((h, w), rr, cc)
            }
            if any(cell in occupied for cell in expanded_cells):
                continue

            border = _frame_border_cells(r0, c0, r1, c1) - {gap}
            if border & occupied:
                continue
            if any(cell in occupied for cell in ray):
                continue

            metas.append(
                {
                    "shape": (h, w),
                    "bbox": bbox,
                    "gap": gap,
                    "side": side,
                    "direction": SIDE_TO_DIR[side],
                    "ray": ray,
                }
            )
            expanded_boxes.append(expanded)
            occupied |= expanded_cells
            occupied |= set(ray)
            placed = True
            break
        if not placed:
            return None

    counts = list(range(1, frame_count + 1))
    for _ in range(40):
        rng.shuffle(counts)
        for meta, count in zip(metas, counts):
            meta["count"] = count
        if _has_size_count_inversion(metas):
            return metas
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        metas = _place_frames(rng)
        if metas is None:
            continue

        h, w = metas[0]["shape"]
        grid = np.zeros((h, w), dtype=int)
        shared = rng.choice(MARKER_COLORS)

        for meta in metas:
            r0, c0, r1, c1 = meta["bbox"]
            gap = meta["gap"]
            for rr, cc in _frame_border_cells(r0, c0, r1, c1) - {gap}:
                grid[rr, cc] = FRAME

        for meta in metas:
            count = int(meta["count"])
            if count == 2:
                mode = "cluster"
            elif count == 3:
                mode = "separated"
            else:
                mode = rng.choice(["cluster", "separated", "random"])
            cells = _interior_cells(meta)
            points = _choose_marker_positions(cells, count, mode, rng)
            if points is None:
                break
            colors = _marker_colors(count, mode, shared, rng)
            for (rr, cc), color in zip(points, colors):
                grid[rr, cc] = int(color)
        else:
            if _validate_generated(grid, metas):
                return grid

    raise RuntimeError("failed to generate a valid input")
