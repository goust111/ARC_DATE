from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

MOVE_COLORS = [BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE]
DIRECTIONS = ("up", "down", "left", "right")
GEN_RETRIES = 400


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 8:
        steps = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    else:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _frame_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts = set()
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    return pts


def _inner_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 + 1, c0 + 1, r1 - 1, c1 - 1)


def _core_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 + 2, c0 + 2, r1 - 2, c1 - 2)


def _distance_to_floor(
    pt: tuple[int, int],
    direction: str,
    inner_bbox: tuple[int, int, int, int],
) -> int:
    r, c = pt
    ir0, ic0, ir1, ic1 = inner_bbox
    if direction == "down":
        return ir1 - r
    if direction == "up":
        return r - ir0
    if direction == "left":
        return c - ic0
    return ic1 - c


def _projection_value(pt: tuple[int, int], direction: str) -> int:
    return pt[1] if direction in ("up", "down") else pt[0]


def _group_order(
    groups: dict[int, list[tuple[int, int]]],
    direction: str,
    inner_bbox: tuple[int, int, int, int],
) -> list[tuple[int, list[tuple[int, int]]]]:
    def key(item: tuple[int, list[tuple[int, int]]]) -> tuple[float, int, tuple[tuple[int, int], ...]]:
        _color, pts = item
        dists = [_distance_to_floor(pt, direction, inner_bbox) for pt in pts]
        return (-sum(dists) / len(dists), -max(dists), tuple(sorted(pts)))

    return sorted(groups.items(), key=key)


def _settle_group(
    points: Iterable[tuple[int, int]],
    direction: str,
    inner_bbox: tuple[int, int, int, int],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    ir0, ic0, ir1, ic1 = inner_bbox
    pts = list(points)
    out: list[tuple[int, int]] = []

    if direction in ("up", "down"):
        by_col: dict[int, int] = {}
        for _r, c in pts:
            by_col[c] = by_col.get(c, 0) + 1
        for c, count in sorted(by_col.items()):
            free_rows = [r for r in range(ir0, ir1 + 1) if (r, c) not in occupied]
            if len(free_rows) < count:
                raise ValueError("not enough free cells in column")
            chosen = free_rows[:count] if direction == "up" else free_rows[-count:]
            out.extend((r, c) for r in chosen)
    else:
        by_row: dict[int, int] = {}
        for r, _c in pts:
            by_row[r] = by_row.get(r, 0) + 1
        for r, count in sorted(by_row.items()):
            free_cols = [c for c in range(ic0, ic1 + 1) if (r, c) not in occupied]
            if len(free_cols) < count:
                raise ValueError("not enough free cells in row")
            chosen = free_cols[:count] if direction == "left" else free_cols[-count:]
            out.extend((r, c) for c in chosen)

    out = sorted(out)
    occupied.update(out)
    return out


def _settle_groups(
    groups: dict[int, list[tuple[int, int]]],
    direction: str,
    inner_bbox: tuple[int, int, int, int],
    reverse: bool = False,
) -> dict[int, list[tuple[int, int]]]:
    order = _group_order(groups, direction, inner_bbox)
    if reverse:
        order = list(reversed(order))
    occupied: set[tuple[int, int]] = set()
    placed: dict[int, list[tuple[int, int]]] = {}
    for color, pts in order:
        placed[color] = _settle_group(pts, direction, inner_bbox, occupied)
    return placed


def _layout_signature(groups: dict[int, list[tuple[int, int]]]) -> tuple[tuple[int, int, int], ...]:
    return tuple(sorted((color, r, c) for color, pts in groups.items() for r, c in pts))


def _same_mean_distance(
    groups: dict[int, list[tuple[int, int]]],
    direction: str,
    inner_bbox: tuple[int, int, int, int],
) -> bool:
    vals: list[tuple[int, int]] = []
    for pts in groups.values():
        total = sum(_distance_to_floor(pt, direction, inner_bbox) for pt in pts)
        size = len(pts)
        vals.append((total, size))
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if vals[i][0] * vals[j][1] == vals[j][0] * vals[i][1]:
                return True
    return False


def _has_projection_overlap(groups: dict[int, list[tuple[int, int]]], direction: str) -> bool:
    projections = [{_projection_value(pt, direction) for pt in pts} for pts in groups.values()]
    for i in range(len(projections)):
        for j in range(i + 1, len(projections)):
            if projections[i] & projections[j]:
                return True
    return False


def _match_arrow(bbox: tuple[int, int, int, int], pts: set[tuple[int, int]]) -> str | None:
    r0, c0, r1, c1 = bbox
    for c in range(c0 + 1, c1):
        if pts == {(r0 - 1, c), (r0 - 2, c - 1), (r0 - 2, c + 1)}:
            return "up"
        if pts == {(r1 + 1, c), (r1 + 2, c - 1), (r1 + 2, c + 1)}:
            return "down"
    for r in range(r0 + 1, r1):
        if pts == {(r, c0 - 1), (r - 1, c0 - 2), (r + 1, c0 - 2)}:
            return "left"
        if pts == {(r, c1 + 1), (r - 1, c1 + 2), (r + 1, c1 + 2)}:
            return "right"
    return None


def _find_containers(grid: np.ndarray) -> list[dict[str, tuple[int, int, int, int] | str]]:
    grey_comps = _components(grid == GREY, connectivity=4)
    arrow_comps = [frozenset(comp) for comp in _components(grid == MAROON, connectivity=8) if len(comp) == 3]

    rects: list[tuple[int, int, int, int]] = []
    for comp in grey_comps:
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if set(comp) != _frame_cells(r0, c0, r1, c1):
            continue
        rects.append((r0, c0, r1, c1))

    used: set[frozenset[tuple[int, int]]] = set()
    containers: list[dict[str, tuple[int, int, int, int] | str]] = []
    for bbox in sorted(rects):
        matched_arrow: frozenset[tuple[int, int]] | None = None
        matched_dir: str | None = None
        for arrow in arrow_comps:
            if arrow in used:
                continue
            direction = _match_arrow(bbox, set(arrow))
            if direction is not None:
                matched_arrow = arrow
                matched_dir = direction
                break
        if matched_arrow is None or matched_dir is None:
            continue
        used.add(matched_arrow)
        containers.append({"bbox": bbox, "direction": matched_dir})
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each container's color groups toward the wall indicated by its outside arrow, stacking groups in farthest-first order.

    Input structure:
    - The background is black.
    - Several one-pixel-thick grey rectangular containers appear in the grid, each with scattered interior colored pixels that do not touch the walls or each other.
    - Each container has exactly one three-pixel maroon arrow outside one wall; the arrowhead points toward that wall.

    Rule:
    - The wall pointed to by the maroon arrow acts as the floor, so gravity points toward that wall.
    - Inside each container, merge all pixels of the same color into one group and sort the groups by their original average distance from the floor, farthest first.
    - Settle the groups one after another: keep each group's row or column coordinates perpendicular to gravity, but compress its pixels along gravity so they occupy the nearest available cells above the floor or earlier groups.
    - Preserve the grey container borders and the maroon arrows.

    Key insight: The outside arrow defines a per-container gravity direction, and the colored pixels do not fall independently; entire same-color groups settle in the reverse depth order.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for container in _find_containers(g):
        bbox = container["bbox"]
        direction = str(container["direction"])
        assert isinstance(bbox, tuple)
        inner_bbox = _inner_bbox(bbox)
        ir0, ic0, ir1, ic1 = inner_bbox

        groups: dict[int, list[tuple[int, int]]] = {}
        for r in range(ir0, ir1 + 1):
            for c in range(ic0, ic1 + 1):
                color = int(g[r, c])
                if color in (BLACK, GREY, MAROON):
                    continue
                groups.setdefault(color, []).append((r, c))

        if not groups:
            continue

        out[ir0:ir1 + 1, ic0:ic1 + 1] = BLACK
        settled = _settle_groups(groups, direction, inner_bbox, reverse=False)
        for color, pts in settled.items():
            for r, c in pts:
                out[r, c] = color

    return out


def _random_group_sizes(rng: random.Random, total: int, n_groups: int) -> list[int]:
    sizes = [1] * n_groups
    for _ in range(total - n_groups):
        choices = [idx for idx, size in enumerate(sizes) if size < 3]
        if not choices:
            break
        sizes[rng.choice(choices)] += 1
    rng.shuffle(sizes)
    return sizes


def _sample_non_touching_positions(
    rng: random.Random,
    bbox: tuple[int, int, int, int],
    count: int,
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = bbox
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    for _ in range(40):
        rng.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        blocked: set[tuple[int, int]] = set()
        for r, c in cells:
            if (r, c) in blocked:
                continue
            chosen.append((r, c))
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    blocked.add((rr, cc))
            if len(chosen) == count:
                return chosen
    return None


def _random_arrow_cells(
    rng: random.Random,
    bbox: tuple[int, int, int, int],
    direction: str,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if direction in ("up", "down"):
        lo = c0 + 2 if c1 - c0 >= 4 else c0 + 1
        hi = c1 - 2 if c1 - c0 >= 4 else c1 - 1
        c = rng.randint(lo, hi)
        if direction == "up":
            return [(r0 - 1, c), (r0 - 2, c - 1), (r0 - 2, c + 1)]
        return [(r1 + 1, c), (r1 + 2, c - 1), (r1 + 2, c + 1)]

    lo = r0 + 2 if r1 - r0 >= 4 else r0 + 1
    hi = r1 - 2 if r1 - r0 >= 4 else r1 - 1
    r = rng.randint(lo, hi)
    if direction == "left":
        return [(r, c0 - 1), (r - 1, c0 - 2), (r + 1, c0 - 2)]
    return [(r, c1 + 1), (r - 1, c1 + 2), (r + 1, c1 + 2)]


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0:r1 + 1, c0] = color
    grid[r0:r1 + 1, c1] = color
    grid[r0, c0:c1 + 1] = color
    grid[r1, c0:c1 + 1] = color


def _reserve_is_free(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return not np.any(mask[r0:r1 + 1, c0:c1 + 1])


def _mark_reserved(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    mask[r0:r1 + 1, c0:c1 + 1] = True


def _make_groups(
    rng: random.Random,
    bbox: tuple[int, int, int, int],
    direction: str,
) -> dict[int, list[tuple[int, int]]] | None:
    inner_bbox = _inner_bbox(bbox)
    core_bbox = _core_bbox(bbox)
    cr0, cc0, cr1, cc1 = core_bbox
    if cr0 > cr1 or cc0 > cc1:
        return None

    core_h = cr1 - cr0 + 1
    core_w = cc1 - cc0 + 1
    capacity = ((core_h + 1) // 2) * ((core_w + 1) // 2)
    if capacity < 3:
        return None

    for _ in range(120):
        max_groups = min(4, len(MOVE_COLORS), capacity)
        n_groups = rng.randint(2, max_groups)
        max_total = min(capacity, n_groups * 3)
        min_total = n_groups + 1 if max_total > n_groups and rng.random() < 0.75 else n_groups
        total = rng.randint(min_total, max_total)

        positions = _sample_non_touching_positions(rng, core_bbox, total)
        if positions is None:
            continue

        sizes = _random_group_sizes(rng, total, n_groups)
        colors = rng.sample(MOVE_COLORS, n_groups)
        rng.shuffle(positions)

        groups: dict[int, list[tuple[int, int]]] = {}
        idx = 0
        for color, size in zip(colors, sizes):
            groups[color] = sorted(positions[idx:idx + size])
            idx += size

        if _same_mean_distance(groups, direction, inner_bbox):
            continue
        if not _has_projection_overlap(groups, direction):
            continue

        forward = _settle_groups(groups, direction, inner_bbox, reverse=False)
        backward = _settle_groups(groups, direction, inner_bbox, reverse=True)
        if _layout_signature(forward) == _layout_signature(backward):
            continue
        if _layout_signature(forward) == _layout_signature(groups):
            continue
        return groups

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(20, 30)
    w = rng.randint(20, 30)
    area = h * w
    max_containers = 1
    if area >= 340:
        max_containers = 2
    if area >= 520:
        max_containers = 3
    n_containers = rng.randint(1, max_containers)

    grid = np.full((h, w), BLACK, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    directions = rng.sample(list(DIRECTIONS), n_containers)

    for direction in directions:
        placed = False
        for _ in range(180):
            inner_h = rng.randint(5, 8)
            inner_w = rng.randint(5, 8)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if outer_h + 6 > h or outer_w + 6 > w:
                continue

            r0 = rng.randint(3, h - outer_h - 3)
            c0 = rng.randint(3, w - outer_w - 3)
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            reserve_bbox = (
                max(0, r0 - 3),
                max(0, c0 - 3),
                min(h - 1, bbox[2] + 3),
                min(w - 1, bbox[3] + 3),
            )
            if not _reserve_is_free(reserved, reserve_bbox):
                continue

            groups = _make_groups(rng, bbox, direction)
            if groups is None:
                continue

            _paint_frame(grid, bbox, GREY)
            for r, c in _random_arrow_cells(rng, bbox, direction):
                grid[r, c] = MAROON
            for color, pts in groups.items():
                for r, c in pts:
                    grid[r, c] = color
            _mark_reserved(reserved, reserve_bbox)
            placed = True
            break

        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
