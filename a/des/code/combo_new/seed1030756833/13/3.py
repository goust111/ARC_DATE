from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
AXIS = 9
COLORS = tuple(range(1, 9))
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _connected(mask: np.ndarray) -> bool:
    total = int(mask.sum())
    if total == 0:
        return False
    return len(_components(mask)) == 1


def _hole_count(mask: np.ndarray) -> int:
    """Count 4-connected background components enclosed inside a foreground mask."""
    h, w = mask.shape
    empty = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bool(empty[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_edge = True
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(empty[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _object_hole_count(g: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> int:
    """Count Black components inside a color object's bbox that are bounded only by that color."""
    r0, c0, r1, c1 = bbox
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    h, w = sub.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if int(sub[r, c]) != BG or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            touches_edge = False
            bounded_by_color = True
            while q:
                rr, cc = q.popleft()
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_edge = True
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if int(sub[nr, nc]) == BG:
                        if not bool(seen[nr, nc]):
                            seen[nr, nc] = True
                            q.append((nr, nc))
                    elif int(sub[nr, nc]) != int(color):
                        bounded_by_color = False
            if not touches_edge and bounded_by_color:
                holes += 1
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add width markers on the reference axis row indexed by each object's hole count.

    Input structure:
    - Black background with a Maroon vertical reference axis filling column 0.
    - Several distinct-color 4-connected foreground objects sit away from the axis and grid border.
    - Each object is a closed shape whose Black 4-connected interior cavities are its holes.

    Rule:
    - Preserve the input grid.
    - For every non-Black, non-Maroon object, count its enclosed Black holes N and measure its bounding-box width.
    - On row N, draw that object's color from column 1 for exactly that width.
    - Objects with the same N share the row, ordered by their bounding-box left edge and placed flush end to end.

    Key insight: The hidden topological count chooses the reference row, while the visible bounding-box width chooses the marker length.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    groups: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    object_colors = [int(c) for c in np.unique(g) if int(c) not in (BG, AXIS)]
    for color in object_colors:
        for pts in _components(g == color):
            r0, c0, r1, c1 = _bbox(pts)
            holes = _object_hole_count(g, int(color), (r0, c0, r1, c1))
            width = c1 - c0 + 1
            groups[int(holes)].append((int(c0), int(r0), int(width), int(color)))

    for row, items in groups.items():
        assert 0 <= row < h, "hole-count row must fit in the grid"
        col = 1
        for _left, _top, width, color in sorted(items, key=lambda item: (item[0], item[1])):
            assert col + width <= w, "marker bars must fit to the right of the axis"
            out[int(row), col : col + int(width)] = int(color)
            col += int(width)

    return out


def _maybe_round_corners(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    mask = mask.copy()
    h, w = mask.shape
    for r, c in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        if rng.random() < 0.45:
            mask[r, c] = False
    return mask


def _make_mask(holes: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(80):
        if holes == 0:
            h = rng.randint(3, 6)
            w = rng.randint(3, 8)
        else:
            h = rng.randint(5, 9)
            w = rng.randint(5, 9)
            if holes >= 5 and (h < 7 or w < 7):
                continue

        mask = np.ones((h, w), dtype=bool)
        if holes:
            centers = [(r, c) for r in range(1, h - 1, 2) for c in range(1, w - 1, 2)]
            if len(centers) < holes:
                continue
            rng.shuffle(centers)
            for r, c in centers[:holes]:
                mask[r, c] = False

        mask = _maybe_round_corners(mask, rng)
        if _connected(mask) and _hole_count(mask) == holes:
            return mask
    return None


def _planned_hole_counts(n: int, rng: random.Random) -> list[int]:
    counts = [rng.randrange(6) for _ in range(n)]
    if n >= 4 and rng.random() < 0.55:
        value = rng.randrange(6)
        i, j = rng.sample(range(n), 2)
        counts[i] = value
        counts[j] = value
    if n >= 5 and rng.random() < 0.35:
        counts[rng.randrange(n)] = rng.choice([0, 5])
    rng.shuffle(counts)
    return counts


def _marker_lengths_fit(width: int, infos: list[dict]) -> bool:
    by_holes: dict[int, int] = defaultdict(int)
    for info in infos:
        by_holes[int(info["holes"])] += int(info["mask"].shape[1])
    return all(total <= width - 1 for total in by_holes.values())


def _place_objects(
    grid: np.ndarray,
    infos: list[dict],
    rng: random.Random,
) -> list[dict] | None:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    occupied[:, 0] = True
    placed: list[dict] = []
    for info in sorted(infos, key=lambda item: int(item["mask"].sum()), reverse=True):
        mask = info["mask"]
        mh, mw = mask.shape
        positions = []
        for r0 in range(7, h - mh):
            for c0 in range(2, w - mw):
                rlo, rhi = max(0, r0 - 1), min(h, r0 + mh + 1)
                clo, chi = max(0, c0 - 1), min(w, c0 + mw + 1)
                if not bool(occupied[rlo:rhi, clo:chi].any()):
                    positions.append((r0, c0))
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        patch = grid[r0 : r0 + mh, c0 : c0 + mw]
        patch[mask] = int(info["color"])
        occupied[r0 : r0 + mh, c0 : c0 + mw] = True
        new_info = dict(info)
        new_info["top"] = int(r0)
        new_info["left"] = int(c0)
        placed.append(new_info)

    for holes in range(6):
        lefts = [int(info["left"]) for info in placed if int(info["holes"]) == holes]
        if len(lefts) != len(set(lefts)):
            return None
    return placed


def _expected_output(grid: np.ndarray, infos: list[dict]) -> np.ndarray:
    out = grid.copy()
    by_holes: dict[int, list[dict]] = defaultdict(list)
    for info in infos:
        by_holes[int(info["holes"])].append(info)
    for row, row_infos in by_holes.items():
        c = 1
        for info in sorted(row_infos, key=lambda item: (int(item["left"]), int(item["top"]))):
            width = int(info["mask"].shape[1])
            out[int(row), c : c + width] = int(info["color"])
            c += width
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(18, 30)
    width = rng.randint(18, 30)
    grid = np.zeros((height, width), dtype=int)
    grid[:, 0] = AXIS

    object_count = rng.randint(3, min(7, len(COLORS)))
    colors = list(COLORS)
    rng.shuffle(colors)
    hole_counts = _planned_hole_counts(object_count, rng)

    infos: list[dict] = []
    for i, holes in enumerate(hole_counts):
        mask = _make_mask(int(holes), rng)
        if mask is None:
            return None
        infos.append({"holes": int(holes), "color": int(colors[i]), "mask": mask})

    if not _marker_lengths_fit(width, infos):
        return None

    placed = _place_objects(grid, infos, rng)
    if placed is None:
        return None

    expected = _expected_output(grid, placed)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 500 attempts")
