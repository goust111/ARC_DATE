from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


PALETTE = (1, 2, 3, 4, 6)
OBSTACLE_COLOR = 9
BACKGROUND = 0

RECT_HEIGHTS = (2, 3)
RECT_WIDTHS = (2, 3, 4)
ZONE_HEIGHT = 8
GEN_ATTEMPTS = 400


Rect = tuple[int, int, int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _rect_cells(rect: Rect) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _paint_rect(grid: np.ndarray, rect: Rect, color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _rect_gap_ok(a: Rect, b: Rect, gap: int = 2) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _find_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    rects: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if _in_bounds(g.shape, nr, nc) and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            rect = (min(rows), min(cols), max(rows), max(cols))
            r0, c0, r1, c1 = rect
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            assert area == len(coords), "every component must be a solid rectangle"
            assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle must be monochromatic"
            rects.append({"color": color, "bbox": rect})

    return rects


def _horizontal_first_path(src: Rect, dst: Rect) -> list[tuple[int, int]]:
    sr0, sc0, sr1, sc1 = src
    tr0, tc0, tr1, tc1 = dst
    assert tc0 > sc1 and tr0 > sr1

    cells: list[tuple[int, int]] = []
    for c in range(sc1 + 1, tc0 + 1):
        cells.append((sr1, c))
    for r in range(sr1 + 1, tr0):
        cells.append((r, tc0))
    return cells


def _vertical_first_path(src: Rect, dst: Rect) -> list[tuple[int, int]]:
    sr0, sc0, sr1, sc1 = src
    tr0, tc0, tr1, tc1 = dst
    assert tc0 > sc1 and tr0 > sr1

    cells: list[tuple[int, int]] = []
    for r in range(sr1 + 1, tr0 + 1):
        cells.append((r, sc1))
    for c in range(sc1 + 1, tc0):
        cells.append((tr0, c))
    return cells


def _path_is_background(path: list[tuple[int, int]], occupied: np.ndarray) -> bool:
    return all(_in_bounds(occupied.shape, r, c) and not occupied[r, c] for r, c in path)


def _remaining_groups(rects: list[dict[str, object]], obstacle_color: int) -> list[list[Rect]]:
    by_color: dict[int, list[Rect]] = defaultdict(list)
    for item in rects:
        color = int(item["color"])
        if color == obstacle_color:
            continue
        by_color[color].append(item["bbox"])  # type: ignore[arg-type]

    ordered_groups: list[list[Rect]] = []
    for group in by_color.values():
        group.sort(key=lambda rect: (rect[0], rect[1], rect[2], rect[3]))
        assert len(group) % 2 == 0, "non-obstacle colors must have an even number of rectangles"
        ordered_groups.append(group)
    ordered_groups.sort(key=lambda group: (group[0][0], group[0][1]))
    return ordered_groups


def _solve(grid: np.ndarray, *, return_branches: bool = False):
    g = np.asarray(grid, dtype=int)
    rects = _find_rectangles(g)
    counts = Counter(int(item["color"]) for item in rects)
    assert counts, "expected at least one rectangle"

    most_common = counts.most_common()
    assert len(most_common) >= 2, "need obstacle color and at least one connectable color"
    assert most_common[0][1] > most_common[1][1], "most frequent rectangle color must be unique"
    obstacle_color = int(most_common[0][0])

    out = g.copy()
    obstacle_rects = [item["bbox"] for item in rects if int(item["color"]) == obstacle_color]
    for rect in obstacle_rects:
        _paint_rect(out, rect, OBSTACLE_COLOR)

    branches: list[str] = []
    for group in _remaining_groups(rects, obstacle_color):
        color = int(g[group[0][0], group[0][1]])
        for i in range(0, len(group), 2):
            src = group[i]
            dst = group[i + 1]
            h_path = _horizontal_first_path(src, dst)
            v_path = _vertical_first_path(src, dst)

            occupied = out != BACKGROUND
            h_ok = _path_is_background(h_path, occupied)
            v_ok = _path_is_background(v_path, occupied)
            assert h_ok or v_ok, "at least one L path must stay on background"

            if h_ok:
                chosen = h_path
                branches.append("H")
            else:
                chosen = v_path
                branches.append("V")

            for r, c in chosen:
                out[r, c] = color

    if return_branches:
        return out, branches
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the most frequent rectangle color into obstacles, then connect the remaining same-color rectangle pairs with obstacle-avoiding L paths.

    Input structure:
    - A black background contains several separated solid monochromatic rectangles.
    - The non-black rectangle colors come from a small palette, and one color appears on more rectangle objects than any other color.
    - Every remaining non-black color appears on an even number of rectangles.

    Rule:
    - Count rectangle objects per non-black color.
    - Recolor every rectangle of the unique most frequent color to maroon (9); these rectangles become obstacles.
    - For each remaining color, sort its rectangles by top-left coordinate and pair them consecutively.
    - For each pair, use the first rectangle's lower-right corner and the second rectangle's upper-left corner as the L-path reference points: draw on background cells only, trying horizontal-then-vertical first and switching to vertical-then-horizontal if the first choice would hit a non-background cell.

    Key insight:
    - The same rectangles drive both stages: frequency analysis decides which color stops being a connection target and instead becomes the obstacle field that later paths must avoid.
    """
    return _solve(np.asarray(grid, dtype=int))


def _sample_rect(zone_top: int, zone_bottom: int, left_band: tuple[int, int], right_band: tuple[int, int], shape: tuple[int, int]) -> tuple[Rect, Rect] | None:
    h, w = shape
    ah = random.choice(RECT_HEIGHTS)
    aw = random.choice(RECT_WIDTHS)
    bh = random.choice(RECT_HEIGHTS)
    bw = random.choice(RECT_WIDTHS)

    src_r0_hi = min(zone_top + 1, zone_bottom - bh - ah - 1)
    if src_r0_hi < zone_top:
        return None
    src_r0 = random.randint(zone_top, src_r0_hi)

    dst_r0_lo = src_r0 + ah + 2
    dst_r0_hi = zone_bottom - bh + 1
    if dst_r0_lo > dst_r0_hi:
        return None
    dst_r0 = random.randint(dst_r0_lo, dst_r0_hi)

    src_c0_lo, src_c0_hi = left_band
    src_c0_hi = min(src_c0_hi, w - aw - 8)
    if src_c0_lo > src_c0_hi:
        return None
    src_c0 = random.randint(src_c0_lo, src_c0_hi)

    dst_c0_lo = max(right_band[0], src_c0 + aw + 3)
    dst_c0_hi = min(right_band[1], w - bw - 2)
    if dst_c0_lo > dst_c0_hi:
        return None
    dst_c0 = random.randint(dst_c0_lo, dst_c0_hi)

    src = (src_r0, src_c0, src_r0 + ah - 1, src_c0 + aw - 1)
    dst = (dst_r0, dst_c0, dst_r0 + bh - 1, dst_c0 + bw - 1)
    if not _rect_gap_ok(src, dst):
        return None
    return src, dst


def _choose_blocker(src: Rect, dst: Rect, used_rects: list[Rect], reserved: set[tuple[int, int]], shape: tuple[int, int]) -> Rect | None:
    h_path = _horizontal_first_path(src, dst)
    v_path = set(_vertical_first_path(src, dst))
    candidates = [(r, c) for r, c in h_path if (r, c) not in v_path]
    random.shuffle(candidates)

    for r, c in candidates:
        for bh in RECT_HEIGHTS:
            for bw in RECT_WIDTHS:
                for r0 in (r - bh + 1, r):
                    for c0 in (c - bw + 1, c):
                        rect = (r0, c0, r0 + bh - 1, c0 + bw - 1)
                        if not _in_bounds(shape, rect[0], rect[1]) or not _in_bounds(shape, rect[2], rect[3]):
                            continue
                        cells = set(_rect_cells(rect))
                        if (r, c) not in cells:
                            continue
                        if cells & reserved:
                            continue
                        if any(not _rect_gap_ok(rect, other) for other in used_rects):
                            continue
                        if cells & v_path:
                            continue
                        return rect
    return None


def _can_place_extra(rect: Rect, used_rects: list[Rect], reserved: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    if not _in_bounds(shape, rect[0], rect[1]) or not _in_bounds(shape, rect[2], rect[3]):
        return False
    if set(_rect_cells(rect)) & reserved:
        return False
    return all(_rect_gap_ok(rect, other) for other in used_rects)


def _place_extra_obstacles(used_rects: list[Rect], reserved: set[tuple[int, int]], shape: tuple[int, int], needed: int) -> list[Rect] | None:
    extras: list[Rect] = []
    h, w = shape
    for _ in range(needed):
        placed = False
        for _ in range(300):
            rh = random.choice(RECT_HEIGHTS)
            rw = random.choice(RECT_WIDTHS)
            rect = (
                random.randint(1, h - rh - 2),
                random.randint(1, w - rw - 2),
                0,
                0,
            )
            rect = (rect[0], rect[1], rect[0] + rh - 1, rect[1] + rw - 1)
            if _can_place_extra(rect, used_rects + extras, reserved, shape):
                extras.append(rect)
                placed = True
                break
        if not placed:
            return None
    return extras


def _generate_attempt() -> np.ndarray | None:
    pair_count = random.choice((2, 3))
    obstacle_count = random.choice((4, 6))
    colors = random.sample(PALETTE, pair_count + 1)
    obstacle_color = colors[0]
    pair_colors = colors[1:]

    min_h = max(20, 3 + pair_count * ZONE_HEIGHT)
    max_h = min(29, min_h + 2)
    h = random.randint(min_h, max_h)
    w = random.randint(20, 28)
    grid = np.zeros((h, w), dtype=int)

    zone_tops = [1 + i * ZONE_HEIGHT for i in range(pair_count)]
    zone_bottoms = [top + ZONE_HEIGHT - 1 for top in zone_tops]
    if zone_bottoms[-1] >= h - 2:
        return None

    used_rects: list[Rect] = []
    placements: list[tuple[Rect, Rect, int, str]] = []
    reserved_actual: set[tuple[int, int]] = set()
    blocker_rects: list[Rect] = []

    for i, color in enumerate(pair_colors):
        desired_branch = "V" if i == 0 else ("H" if i == 1 else random.choice(("H", "V")))
        success = False
        for _ in range(150):
            sample = _sample_rect(
                zone_tops[i],
                zone_bottoms[i],
                (1, 4),
                (max(11, w // 2), w - 6),
                grid.shape,
            )
            if sample is None:
                continue
            src, dst = sample
            if any(not _rect_gap_ok(src, other) or not _rect_gap_ok(dst, other) for other in used_rects):
                continue

            h_path = _horizontal_first_path(src, dst)
            v_path = _vertical_first_path(src, dst)
            h_set = set(h_path)
            v_set = set(v_path)

            if not h_path or not v_path:
                continue
            if any(cell in reserved_actual for cell in h_set | v_set):
                continue
            if desired_branch == "H":
                chosen_path = h_set
            else:
                chosen_path = v_set
            if any(cell in reserved_actual for cell in chosen_path):
                continue

            blocker = None
            if desired_branch == "V":
                blocker = _choose_blocker(src, dst, used_rects + [src, dst], reserved_actual | v_set, grid.shape)
                if blocker is None:
                    continue

            placements.append((src, dst, color, desired_branch))
            used_rects.extend([src, dst])
            reserved_actual |= chosen_path
            if blocker is not None:
                blocker_rects.append(blocker)
                used_rects.append(blocker)
            success = True
            break
        if not success:
            return None

    needed_extra = obstacle_count - len(blocker_rects)
    if needed_extra < 0:
        return None
    extras = _place_extra_obstacles(used_rects, reserved_actual, grid.shape, needed_extra)
    if extras is None:
        return None
    obstacle_rects = blocker_rects + extras

    for src, dst, color, _ in placements:
        _paint_rect(grid, src, color)
        _paint_rect(grid, dst, color)
    for rect in obstacle_rects:
        _paint_rect(grid, rect, obstacle_color)

    out, branches = _solve(grid, return_branches=True)
    if not ({"H", "V"} <= set(branches)):
        return None
    if len(_find_rectangles(grid)) != obstacle_count + 2 * pair_count:
        return None
    if not np.any(out == OBSTACLE_COLOR):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
