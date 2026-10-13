from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BLACK = 0

DIRS = {
    "ul": {"missing": (0, 0), "corner": lambda h, w: (0, 0), "step": (1, 1)},
    "ur": {"missing": (0, 1), "corner": lambda h, w: (0, w - 1), "step": (1, -1)},
    "dl": {"missing": (1, 0), "corner": lambda h, w: (h - 1, 0), "step": (-1, 1)},
    "dr": {"missing": (1, 1), "corner": lambda h, w: (h - 1, w - 1), "step": (-1, -1)},
}


def _components_by_color(g: np.ndarray, bg: int = BLACK) -> list[dict[str, object]]:
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(g[sr, sc]) == bg:
                continue
            color = int(g[sr, sc])
            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [r for r, _c in coords]
            cols = [c for _r, c in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _corner_distance(direction: str, missing_r: int, missing_c: int, h: int, w: int) -> int:
    if direction == "ul":
        return int(missing_r + missing_c)
    if direction == "ur":
        return int(missing_r + (w - 1 - missing_c))
    if direction == "dl":
        return int((h - 1 - missing_r) + missing_c)
    if direction == "dr":
        return int((h - 1 - missing_r) + (w - 1 - missing_c))
    raise AssertionError(f"unknown direction {direction!r}")


def _extract_arrows(g: np.ndarray) -> list[dict[str, int | str]]:
    h, w = map(int, g.shape)
    arrows: list[dict[str, int | str]] = []
    for comp in _components_by_color(g, BLACK):
        coords = comp["coords"]
        assert isinstance(coords, list)
        assert len(coords) == 3, "each arrow must contain exactly three pixels"
        r0, c0, r1, c1 = comp["bbox"]  # type: ignore[misc]
        assert int(r1) - int(r0) == 1 and int(c1) - int(c0) == 1, "each arrow must occupy a 2x2 box"

        rel = {(int(r) - int(r0), int(c) - int(c0)) for r, c in coords}
        missing = [(r, c) for r in range(2) for c in range(2) if (r, c) not in rel]
        assert len(missing) == 1, "a 2x2 arrow must have exactly one missing corner"
        missing_rel = missing[0]

        direction = None
        for name, spec in DIRS.items():
            if spec["missing"] == missing_rel:
                direction = name
                break
        assert direction is not None

        mr = int(r0) + int(missing_rel[0])
        mc = int(c0) + int(missing_rel[1])
        arrows.append(
            {
                "direction": direction,
                "color": int(comp["color"]),
                "distance": _corner_distance(direction, mr, mc, h, w),
                "row": int(r0),
                "col": int(c0),
            }
        )
    assert arrows, "expected at least one arrow"
    return arrows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge same-facing triangular arrows into diagonal color bands from the matching corners.

    Input structure:
    - The grid has a Black background.
    - Every foreground object is a monochrome three-pixel right-triangle arrow
      inside a 2x2 box, with one corner missing.
    - The missing corner encodes one of four directions: upper-left,
      upper-right, lower-left, or lower-right.

    Rule:
    - Identify each arrow's direction from the missing corner of its 2x2 box.
    - Group arrows by direction.
    - Within each group, sort arrows by distance from that direction's matching
      grid corner, nearest first.
    - Erase the original arrows, then draw a one-cell-wide diagonal band from
      that corner into the grid, using the sorted arrow colors in order.

    Key insight:
    - The small triangle is a direction code; the ordered positions of triangles
      with the same direction become the palette for a new corner diagonal.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h > 0 and w > 0

    grouped: dict[str, list[dict[str, int | str]]] = defaultdict(list)
    for arrow in _extract_arrows(g):
        grouped[str(arrow["direction"])].append(arrow)

    out = np.full((h, w), BLACK, dtype=int)
    for direction in ("ul", "ur", "dl", "dr"):
        arrows = grouped.get(direction, [])
        if not arrows:
            continue
        arrows = sorted(arrows, key=lambda a: (int(a["distance"]), int(a["row"]), int(a["col"])))
        assert len(arrows) <= min(h, w), "a diagonal band must fit inside the grid"
        r, c = DIRS[direction]["corner"](h, w)  # type: ignore[operator]
        dr, dc = DIRS[direction]["step"]  # type: ignore[misc]
        for i, arrow in enumerate(arrows):
            rr = int(r) + int(dr) * i
            cc = int(c) + int(dc) * i
            assert 0 <= rr < h and 0 <= cc < w
            out[rr, cc] = int(arrow["color"])
    return out


def _shape_cells(direction: str, r0: int, c0: int) -> set[tuple[int, int]]:
    missing = DIRS[direction]["missing"]
    return {
        (r0 + rr, c0 + cc)
        for rr in range(2)
        for cc in range(2)
        if (rr, cc) != missing
    }


def _distance_for_box(direction: str, r0: int, c0: int, h: int, w: int) -> int:
    mr = r0 + int(DIRS[direction]["missing"][0])
    mc = c0 + int(DIRS[direction]["missing"][1])
    return _corner_distance(direction, mr, mc, h, w)


def _bands_are_disjoint(h: int, w: int, counts: dict[str, int]) -> bool:
    seen: set[tuple[int, int]] = set()
    for direction, count in counts.items():
        r, c = DIRS[direction]["corner"](h, w)  # type: ignore[operator]
        dr, dc = DIRS[direction]["step"]  # type: ignore[misc]
        for i in range(count):
            cell = (int(r) + int(dr) * i, int(c) + int(dc) * i)
            if cell in seen:
                return False
            seen.add(cell)
    return True


def _pick_direction_counts() -> dict[str, int]:
    directions = random.sample(list(DIRS), random.randint(1, 4))
    total = random.randint(len(directions), min(9, len(directions) * 3))
    counts = {direction: 1 for direction in directions}
    for _ in range(total - len(directions)):
        choices = [direction for direction in directions if counts[direction] < 3]
        counts[random.choice(choices)] += 1
    return counts


def _place_arrow(
    grid: np.ndarray,
    direction: str,
    color: int,
    blocked: set[tuple[int, int]],
    used_distances: set[int],
) -> bool:
    h, w = map(int, grid.shape)
    candidates = [(r, c) for r in range(h - 1) for c in range(w - 1)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        cells = _shape_cells(direction, r0, c0)
        padded = {
            (r, c)
            for rr in range(r0 - 1, r0 + 3)
            for cc in range(c0 - 1, c0 + 3)
            for r, c in [(rr, cc)]
            if 0 <= r < h and 0 <= c < w
        }
        if cells & blocked or padded & blocked:
            continue
        dist = _distance_for_box(direction, r0, c0, h, w)
        if dist in used_distances:
            continue
        for r, c in cells:
            grid[r, c] = color
        blocked.update(padded)
        used_distances.add(dist)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 30)
    w = random.randint(12, 30)
    counts = _pick_direction_counts()
    if not _bands_are_disjoint(h, w, counts):
        return None

    total = sum(counts.values())
    colors = random.sample(range(1, 10), total)
    random.shuffle(colors)

    grid = np.full((h, w), BLACK, dtype=int)
    blocked: set[tuple[int, int]] = set()
    used_distances = {direction: set() for direction in counts}
    color_i = 0
    for direction in random.sample(list(counts), len(counts)):
        for _ in range(counts[direction]):
            color = colors[color_i]
            color_i += 1
            if not _place_arrow(grid, direction, color, blocked, used_distances[direction]):
                return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if max(out.shape) > 30 or not np.any(out != BLACK):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid triangular-arrow input")
