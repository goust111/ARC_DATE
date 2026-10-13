from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2

SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400
PLACEMENT_RETRIES = 300
OBJECT_RETRIES = 120


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(pts: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(pts[:, 0].min()),
        int(pts[:, 1].min()),
        int(pts[:, 0].max()),
        int(pts[:, 1].max()),
    )


def _is_rectangular_ring(pts: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(pts)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    cells = {(int(r), int(c)) for r, c in pts}
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    return cells == expected


def _parse_rod(grid: np.ndarray, pts: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[str, int, int] | None:
    r0, c0, r1, c1 = rect
    rows = np.unique(pts[:, 0])
    cols = np.unique(pts[:, 1])

    if len(rows) == 1:
        row = int(rows[0])
        cmin = int(cols.min())
        cmax = int(cols.max())
        if len(pts) != cmax - cmin + 1:
            return None
        length = len(pts)
        if cmin == c0 + 1 and grid[row, c0] == BLUE:
            return "left", row, length
        if cmax == c1 - 1 and grid[row, c1] == BLUE:
            return "right", row, length
        return None

    if len(cols) == 1:
        col = int(cols[0])
        rmin = int(rows.min())
        rmax = int(rows.max())
        if len(pts) != rmax - rmin + 1:
            return None
        length = len(pts)
        if rmin == r0 + 1 and grid[r0, col] == BLUE:
            return "top", col, length
        if rmax == r1 - 1 and grid[r1, col] == BLUE:
            return "bottom", col, length
        return None

    return None


def _draw_protrusion(
    out: np.ndarray,
    rect: tuple[int, int, int, int],
    side: str,
    pos: int,
    length: int,
) -> None:
    h, w = out.shape
    r0, c0, r1, c1 = rect

    if side == "top":
        if 0 <= r0 < h and 0 <= pos < w:
            out[r0, pos] = BG
        tip = r0 - length
        for rr in range(tip, r0 + 1):
            if 0 <= rr < h:
                if 0 <= pos - 1 < w:
                    out[rr, pos - 1] = BLUE
                if 0 <= pos + 1 < w:
                    out[rr, pos + 1] = BLUE
        if 0 <= tip < h:
            for cc in range(pos - 1, pos + 2):
                if 0 <= cc < w:
                    out[tip, cc] = BLUE
        return

    if side == "bottom":
        if 0 <= r1 < h and 0 <= pos < w:
            out[r1, pos] = BG
        tip = r1 + length
        for rr in range(r1, tip + 1):
            if 0 <= rr < h:
                if 0 <= pos - 1 < w:
                    out[rr, pos - 1] = BLUE
                if 0 <= pos + 1 < w:
                    out[rr, pos + 1] = BLUE
        if 0 <= tip < h:
            for cc in range(pos - 1, pos + 2):
                if 0 <= cc < w:
                    out[tip, cc] = BLUE
        return

    if side == "left":
        if 0 <= pos < h and 0 <= c0 < w:
            out[pos, c0] = BG
        tip = c0 - length
        for cc in range(tip, c0 + 1):
            if 0 <= cc < w:
                if 0 <= pos - 1 < h:
                    out[pos - 1, cc] = BLUE
                if 0 <= pos + 1 < h:
                    out[pos + 1, cc] = BLUE
        if 0 <= tip < w:
            for rr in range(pos - 1, pos + 2):
                if 0 <= rr < h:
                    out[rr, tip] = BLUE
        return

    if 0 <= pos < h and 0 <= c1 < w:
        out[pos, c1] = BG
    tip = c1 + length
    for cc in range(c1, tip + 1):
        if 0 <= cc < w:
            if 0 <= pos - 1 < h:
                out[pos - 1, cc] = BLUE
            if 0 <= pos + 1 < h:
                out[pos + 1, cc] = BLUE
    if 0 <= tip < w:
        for rr in range(pos - 1, pos + 2):
            if 0 <= rr < h:
                out[rr, tip] = BLUE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert red indicator rods inside blue rectangular frames into outward blue protrusions.

    The rod's touched side selects the growth direction, while rod length selects the outward
    distance. Each protrusion replaces a single border cell with a 1-cell mouth and grows as
    a 3-cell-wide U shape: two side walls plus a capped end.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[out == RED] = BG

    blue_comps = _components(g == BLUE)
    red_comps = _components(g == RED)

    rects: list[tuple[int, int, int, int]] = []
    for comp in blue_comps:
        if _is_rectangular_ring(comp):
            rects.append(_bbox(comp))

    for rect in rects:
        r0, c0, r1, c1 = rect
        for comp in red_comps:
            if not np.all((comp[:, 0] > r0) & (comp[:, 0] < r1) & (comp[:, 1] > c0) & (comp[:, 1] < c1)):
                continue
            rod = _parse_rod(g, comp, rect)
            if rod is None:
                continue
            side, pos, length = rod
            _draw_protrusion(out, rect, side, pos, length)

    return out


def _draw_rectangle(canvas: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    canvas[r0, c0 : c1 + 1] = BLUE
    canvas[r1, c0 : c1 + 1] = BLUE
    canvas[r0 : r1 + 1, c0] = BLUE
    canvas[r0 : r1 + 1, c1] = BLUE


def _rod_cells_local(side: str, pos: int, length: int, h: int, w: int) -> set[tuple[int, int]]:
    if side == "top":
        return {(1 + k, pos) for k in range(length)}
    if side == "bottom":
        return {(h - 2 - k, pos) for k in range(length)}
    if side == "left":
        return {(pos, 1 + k) for k in range(length)}
    return {(pos, w - 2 - k) for k in range(length)}


def _touches_occupied(candidate: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in candidate:
        if (r, c) in occupied:
            return True
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in occupied:
                return True
    return False


def _sample_object(rng: np.random.Generator) -> dict[str, object] | None:
    for _ in range(OBJECT_RETRIES):
        h = int(rng.integers(6, 12))
        w = int(rng.integers(6, 12))

        available = []
        if h - 2 >= 2:
            available.extend(["top", "bottom"])
        if w - 2 >= 2:
            available.extend(["left", "right"])
        if not available:
            continue

        rod_count = int(rng.integers(1, min(3, len(available)) + 1))
        perm = rng.permutation(len(available))
        chosen_sides = [available[int(i)] for i in perm[:rod_count]]

        rods: list[dict[str, int | str]] = []
        occupied: set[tuple[int, int]] = set()
        ok = True

        for side in chosen_sides:
            if side in ("top", "bottom"):
                positions = list(range(2, w - 2))
                max_len = min(4, h - 2)
            else:
                positions = list(range(2, h - 2))
                max_len = min(4, w - 2)

            candidates: list[tuple[int, int, set[tuple[int, int]]]] = []
            for pos in positions:
                for length in range(2, max_len + 1):
                    cells = _rod_cells_local(side, pos, length, h, w)
                    if not _touches_occupied(cells, occupied):
                        candidates.append((pos, length, cells))

            if not candidates:
                ok = False
                break

            pick = candidates[int(rng.integers(len(candidates)))]
            pos, length, cells = pick
            occupied |= cells
            rods.append({"side": side, "pos": pos, "length": length})

        if not ok:
            continue

        extents = {side: 0 for side in SIDES}
        for rod in rods:
            extents[str(rod["side"])] = max(extents[str(rod["side"])], int(rod["length"]))

        return {"h": h, "w": w, "rods": rods, "extents": extents}

    return None


def _render_input(h: int, w: int, objects: list[dict[str, object]]) -> np.ndarray:
    canvas = np.full((h, w), BG, dtype=int)
    for obj in objects:
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        oh = int(obj["h"])
        ow = int(obj["w"])
        _draw_rectangle(canvas, r0, c0, oh, ow)
        r1 = r0 + oh - 1
        c1 = c0 + ow - 1
        for rod in obj["rods"]:
            side = str(rod["side"])
            pos = int(rod["pos"])
            length = int(rod["length"])
            if side == "top":
                canvas[r0 + 1 : r0 + 1 + length, c0 + pos] = RED
            elif side == "bottom":
                canvas[r1 - length : r1, c0 + pos] = RED
            elif side == "left":
                canvas[r0 + pos, c0 + 1 : c0 + 1 + length] = RED
            else:
                canvas[r0 + pos, c1 - length : c1] = RED
    return canvas


def _render_output(h: int, w: int, objects: list[dict[str, object]]) -> np.ndarray:
    canvas = np.full((h, w), BG, dtype=int)
    for obj in objects:
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        oh = int(obj["h"])
        ow = int(obj["w"])
        rect = (r0, c0, r0 + oh - 1, c0 + ow - 1)
        _draw_rectangle(canvas, r0, c0, oh, ow)
        for rod in obj["rods"]:
            _draw_protrusion(
                canvas,
                rect,
                str(rod["side"]),
                c0 + int(rod["pos"]) if str(rod["side"]) in ("top", "bottom") else r0 + int(rod["pos"]),
                int(rod["length"]),
            )
    return canvas


def _place_objects(rng: np.random.Generator, h: int, w: int, count: int) -> list[dict[str, object]] | None:
    reserved = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for _ in range(count):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            obj = _sample_object(rng)
            if obj is None:
                return None

            ext = obj["extents"]
            top = int(ext["top"])
            bottom = int(ext["bottom"])
            left = int(ext["left"])
            right = int(ext["right"])
            oh = int(obj["h"])
            ow = int(obj["w"])

            min_r0 = top + 1
            max_r0 = h - oh - bottom - 1
            min_c0 = left + 1
            max_c0 = w - ow - right - 1
            if min_r0 > max_r0 or min_c0 > max_c0:
                continue

            r0 = int(rng.integers(min_r0, max_r0 + 1))
            c0 = int(rng.integers(min_c0, max_c0 + 1))
            er0 = r0 - top
            ec0 = c0 - left
            er1 = r0 + oh - 1 + bottom
            ec1 = c0 + ow - 1 + right
            pr0 = er0 - 1
            pc0 = ec0 - 1
            pr1 = er1 + 1
            pc1 = ec1 + 1

            if reserved[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
                continue

            reserved[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
            obj = dict(obj)
            obj["r0"] = r0
            obj["c0"] = c0
            objects.append(obj)
            placed = True
            break

        if not placed:
            return None

    return objects


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 31))
    w = int(rng.integers(18, 31))
    if min(h, w) < 22:
        count_choices = np.array([1, 2, 3], dtype=int)
        probs = np.array([0.3, 0.45, 0.25])
    else:
        count_choices = np.array([1, 2, 3, 4], dtype=int)
        probs = np.array([0.2, 0.35, 0.3, 0.15])
    count = int(rng.choice(count_choices, p=probs))

    objects = _place_objects(rng, h, w, count)
    if not objects:
        return None

    grid = _render_input(h, w, objects)
    expected = _render_output(h, w, objects)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
