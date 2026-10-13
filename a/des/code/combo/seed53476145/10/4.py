from __future__ import annotations

import random

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _uniform_nonzero(arr: np.ndarray) -> int | None:
    vals = {int(v) for v in np.asarray(arr).ravel()}
    if len(vals) != 1 or BG in vals:
        return None
    return vals.pop()


def _find_regions(g: np.ndarray) -> list[dict[str, object]]:
    h, w = g.shape
    regions: list[dict[str, object]] = []

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if int(g[r0, c0]) == BG:
                continue
            if r0 > 0 and int(g[r0 - 1, c0]) != BG:
                continue
            if c0 > 0 and int(g[r0, c0 - 1]) != BG:
                continue
            if int(g[r0, c0 + 1]) == BG or int(g[r0 + 1, c0]) == BG:
                continue

            c1 = c0
            while c1 + 1 < w and int(g[r0, c1 + 1]) != BG:
                c1 += 1
            r1 = r0
            while r1 + 1 < h and int(g[r1 + 1, c0]) != BG:
                r1 += 1

            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            if not np.all(g[r0, c0 : c1 + 1] != BG):
                continue
            if not np.all(g[r1, c0 : c1 + 1] != BG):
                continue
            if not np.all(g[r0 : r1 + 1, c0] != BG):
                continue
            if not np.all(g[r0 : r1 + 1, c1] != BG):
                continue

            side_colors = {
                "top": _uniform_nonzero(g[r0, c0 + 1 : c1]),
                "bottom": _uniform_nonzero(g[r1, c0 + 1 : c1]),
                "left": _uniform_nonzero(g[r0 + 1 : r1, c0]),
                "right": _uniform_nonzero(g[r0 + 1 : r1, c1]),
            }
            if any(color is None for color in side_colors.values()):
                continue
            if len(set(side_colors.values())) != 4:
                continue

            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            allowed = set(side_colors.values())
            markers: list[tuple[int, int, int]] = []
            valid = True
            for rr, cc in np.argwhere(interior != BG):
                rr = int(rr) + r0 + 1
                cc = int(cc) + c0 + 1
                color = int(g[rr, cc])
                if color not in allowed:
                    valid = False
                    break
                for nr, nc in _neighbors4(rr, cc):
                    if r0 < nr < r1 and c0 < nc < c1 and int(g[nr, nc]) != BG:
                        valid = False
                        break
                if not valid:
                    break
                markers.append((rr, cc, color))
            if not valid:
                continue

            regions.append(
                {
                    "outer": (r0, c0, r1, c1),
                    "inner": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                    "side_colors": side_colors,
                    "markers": markers,
                }
            )

    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each interior marker toward the wall side that has the same color,
    leaving a straight same-color trail from the original position to the new one.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for region in _find_regions(g):
        ir0, ic0, ir1, ic1 = region["inner"]
        side_colors = region["side_colors"]
        markers = region["markers"]

        for r, c, color in markers:
            matches = [side for side, side_color in side_colors.items() if side_color == color]
            assert len(matches) == 1, "marker color must match exactly one wall side"
            side = matches[0]

            if side == "top":
                out[ir0 : r + 1, c] = color
            elif side == "bottom":
                out[r : ir1 + 1, c] = color
            elif side == "left":
                out[r, ic0 : c + 1] = color
            else:
                out[r, c : ic1 + 1] = color

    return out


# --------------- generator ---------------

PALETTE = tuple(range(1, 10))
GRID_SIDE_RANGE = (14, 24)
REGION_COUNT_RANGE = (1, 4)
INNER_HEIGHT_RANGE = (3, 7)
INNER_WIDTH_RANGE = (3, 7)
MARKER_COUNT_MAX = 4
GEN_RETRIES = 400


def _draw_region(
    grid: np.ndarray,
    outer: tuple[int, int, int, int],
    side_colors: dict[str, int],
) -> None:
    r0, c0, r1, c1 = outer
    grid[r0, c0 : c1 + 1] = side_colors["top"]
    grid[r1, c0 : c1 + 1] = side_colors["bottom"]
    grid[r0 : r1 + 1, c0] = side_colors["left"]
    grid[r0 : r1 + 1, c1] = side_colors["right"]


def _all_marker_candidates(
    inner: tuple[int, int, int, int]
) -> dict[str, list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]]]:
    ir0, ic0, ir1, ic1 = inner
    candidates = {side: [] for side in SIDES}

    for c in range(ic0, ic1 + 1):
        for r in range(ir0 + 1, ir1 + 1):
            path = tuple((rr, c) for rr in range(ir0, r + 1))
            candidates["top"].append(((r, c), path))
        for r in range(ir0, ir1):
            path = tuple((rr, c) for rr in range(r, ir1 + 1))
            candidates["bottom"].append(((r, c), path))

    for r in range(ir0, ir1 + 1):
        for c in range(ic0 + 1, ic1 + 1):
            path = tuple((r, cc) for cc in range(ic0, c + 1))
            candidates["left"].append(((r, c), path))
        for c in range(ic0, ic1):
            path = tuple((r, cc) for cc in range(c, ic1 + 1))
            candidates["right"].append(((r, c), path))

    for side in SIDES:
        random.shuffle(candidates[side])
    return candidates


def _place_markers(
    inp: np.ndarray,
    expected: np.ndarray,
    inner: tuple[int, int, int, int],
    side_colors: dict[str, int],
) -> bool:
    ir0, ic0, ir1, ic1 = inner
    area = (ir1 - ir0 + 1) * (ic1 - ic0 + 1)
    max_markers = min(MARKER_COUNT_MAX, max(1, area // 4))
    candidates = _all_marker_candidates(inner)

    for _ in range(60):
        count = random.randint(1, max_markers)
        requested_sides = [random.choice(SIDES) for _ in range(count)]
        occupied: set[tuple[int, int]] = set()
        placed: list[tuple[str, tuple[int, int], tuple[tuple[int, int], ...]]] = []
        ok = True

        for side in requested_sides:
            valid = [
                candidate
                for candidate in candidates[side]
                if not any(cell in occupied for cell in candidate[1])
            ]
            if not valid:
                ok = False
                break
            source, path = random.choice(valid)
            occupied.update(path)
            placed.append((side, source, path))

        if not ok:
            continue

        for side, source, path in placed:
            sr, sc = source
            color = side_colors[side]
            inp[sr, sc] = color
            for pr, pc in path:
                expected[pr, pc] = color
        return True

    return False


def _sample_regions(
    h: int, w: int
) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int], dict[str, int]]] | None:
    n_regions = random.randint(*REGION_COUNT_RANGE)
    reserved = np.zeros((h, w), dtype=bool)
    specs: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int], dict[str, int]]] = []

    for _ in range(n_regions):
        placed = False
        for _ in range(80):
            inner_h = random.randint(*INNER_HEIGHT_RANGE)
            inner_w = random.randint(*INNER_WIDTH_RANGE)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if outer_h + 2 > h or outer_w + 2 > w:
                return None

            r0 = random.randint(1, h - outer_h - 1)
            c0 = random.randint(1, w - outer_w - 1)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            rr0 = r0 - 1
            cc0 = c0 - 1
            rr1 = r1 + 1
            cc1 = c1 + 1
            if np.any(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1]):
                continue

            colors = random.sample(PALETTE, 4)
            side_colors = dict(zip(SIDES, colors))
            inner = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
            specs.append(((r0, c0, r1, c1), inner, side_colors))
            reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            placed = True
            break
        if not placed:
            return None

    return specs


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    inp = np.zeros((h, w), dtype=int)
    expected = inp.copy()

    specs = _sample_regions(h, w)
    if specs is None:
        return None

    for outer, inner, side_colors in specs:
        _draw_region(inp, outer, side_colors)
        _draw_region(expected, outer, side_colors)
        if not _place_markers(inp, expected, inner, side_colors):
            return None

    regions = _find_regions(inp)
    if len(regions) != len(specs):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
