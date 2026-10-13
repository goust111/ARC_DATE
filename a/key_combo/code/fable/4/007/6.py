from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREY = 5
PINK = 6
RECT_COLORS = (1, 2, 3, 4, 7, 8, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 700


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _prefix(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    return int(ps[r1 + 1, c1 + 1] - ps[r0, c1 + 1] - ps[r1 + 1, c0] + ps[r0, c0])


def _border_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    full = _rect_sum(ps, r0, c0, r1, c1)
    inner = _rect_sum(ps, r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    return full - inner


def _gap_info(
    gap: tuple[int, int], bbox: tuple[int, int, int, int]
) -> tuple[str, tuple[int, int], int] | None:
    r, c = gap
    r0, c0, r1, c1 = bbox
    if r == r0 and c0 < c < c1:
        return "top", (1, 0), c
    if r == r1 and c0 < c < c1:
        return "bottom", (-1, 0), c
    if c == c0 and r0 < r < r1:
        return "left", (0, 1), r
    if c == c1 and r0 < r < r1:
        return "right", (0, -1), r
    return None


def _opposite_side(side: str) -> str:
    return {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[side]


def _opposite_gap(
    rect: dict, side: str, entry: tuple[int, int]
) -> tuple[int, int]:
    r, c = entry
    r0, c0, r1, c1 = rect["bbox"]
    if side == "top":
        return r1, c
    if side == "bottom":
        return r0, c
    if side == "left":
        return r, c1
    return r, c0


def _valid_gap_set(
    gaps: list[tuple[int, int]], bbox: tuple[int, int, int, int]
) -> bool:
    infos = [_gap_info(gap, bbox) for gap in gaps]
    if any(info is None for info in infos):
        return False
    if len(gaps) == 1:
        return True
    side0, _inward0, key0 = infos[0]
    side1, _inward1, key1 = infos[1]
    return _opposite_side(side0) == side1 and key0 == key1


def _find_rectangles(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    non_bg_ps = _prefix(grid != bg)
    bg_ps = _prefix(grid == bg)
    rects: list[dict] = []
    used_border: set[tuple[int, int]] = set()

    for color in sorted(set(map(int, np.unique(grid))) - {bg, GREY, PINK}):
        color_ps = _prefix(grid == color)
        coords = np.argwhere(grid == color)
        rows = sorted({int(r) for r, _c in coords})
        cols = sorted({int(c) for _r, c in coords})
        for i, r0 in enumerate(rows):
            for r1 in rows[i + 1 :]:
                if r1 - r0 < 2:
                    continue
                for j, c0 in enumerate(cols):
                    for c1 in cols[j + 1 :]:
                        if c1 - c0 < 2:
                            continue
                        perim = 2 * (r1 - r0 + 1) + 2 * (c1 - c0 + 1) - 4
                        color_border = _border_sum(color_ps, r0, c0, r1, c1)
                        missing = perim - color_border
                        if missing not in (1, 2):
                            continue
                        if _border_sum(bg_ps, r0, c0, r1, c1) != missing:
                            continue
                        if _rect_sum(non_bg_ps, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != 0:
                            continue

                        gaps: list[tuple[int, int]] = []
                        for c in range(c0, c1 + 1):
                            if int(grid[r0, c]) == bg:
                                gaps.append((r0, c))
                            if int(grid[r1, c]) == bg:
                                gaps.append((r1, c))
                        for r in range(r0 + 1, r1):
                            if int(grid[r, c0]) == bg:
                                gaps.append((r, c0))
                            if int(grid[r, c1]) == bg:
                                gaps.append((r, c1))

                        if len(gaps) != missing or not _valid_gap_set(gaps, (r0, c0, r1, c1)):
                            continue
                        border_cells = {
                            (r, c)
                            for r in range(r0, r1 + 1)
                            for c in range(c0, c1 + 1)
                            if r in (r0, r1) or c in (c0, c1)
                        } - set(gaps)
                        if border_cells & used_border:
                            continue
                        used_border |= border_cells
                        rects.append(
                            {
                                "id": len(rects),
                                "color": color,
                                "bbox": (r0, c0, r1, c1),
                                "gaps": tuple(sorted(gaps)),
                            }
                        )

    return sorted(rects, key=lambda rec: rec["bbox"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only the hollow rectangles reached by light entering through an aligned gap.

    Input structure:
    - The dominant background contains one gray source pixel and several detached
      monochrome hollow rectangles with one-cell wall gaps.
    - Some rectangles have a second aligned gap on the opposite wall, and sparse
      colored single pixels may appear as blockers or distractors.
    - Rectangle interiors and gap cells are background-colored.

    Rule:
    - Cast one straight ray from the gray source in each of the four cardinal directions.
    - A ray moves through background cells and stops at the first colored pixel unless
      that background cell is a rectangle gap perpendicular to the ray.
    - Entering through such a gap lights that rectangle, filling its whole interior pink.
      If the opposite wall has the aligned exit gap, the ray leaves and continues;
      otherwise it stops at the far inner wall.
    - Unlit rectangles, blockers, distractors, gaps, walls, and the source remain unchanged.

    Key insight:
    - The wall gap is a doorway, not decoration: visibility is determined by the
      source row or column, clear line of sight, and possible aligned exit gaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit inside 30x30"

    bg = _background(g)
    source_pixels = np.argwhere(g == GREY)
    assert source_pixels.shape == (1, 2), "expected one gray light source"
    sr, sc = map(int, source_pixels[0])

    rects = _find_rectangles(g, bg)
    assert rects, "expected at least one hollow rectangle"

    gap_lookup: dict[tuple[int, int], tuple[dict, str, tuple[int, int]]] = {}
    for rect in rects:
        for gap in rect["gaps"]:
            info = _gap_info(gap, rect["bbox"])
            assert info is not None
            side, inward, _key = info
            assert gap not in gap_lookup, "rectangle gaps must be distinct"
            gap_lookup[gap] = (rect, side, inward)

    lit: set[int] = set()
    for dr, dc in DIRS:
        r, c = sr, sc
        while True:
            nr, nc = r + dr, c + dc
            if not _in_bounds(g.shape, nr, nc):
                break
            gap_hit = gap_lookup.get((nr, nc))
            if gap_hit is not None:
                rect, side, inward = gap_hit
                if inward == (dr, dc):
                    lit.add(int(rect["id"]))
                    exit_gap = _opposite_gap(rect, side, (nr, nc))
                    if exit_gap in rect["gaps"]:
                        r, c = exit_gap
                        continue
                    break
                r, c = nr, nc
                continue
            if int(g[nr, nc]) == bg:
                r, c = nr, nc
                continue
            break

    out = g.copy()
    for rect in rects:
        if int(rect["id"]) not in lit:
            continue
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = PINK
    return out


# ---------------- generator ----------------


def _wall_cells(
    r0: int, c0: int, h: int, w: int, gaps: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    r1, c1 = r0 + h - 1, c0 + w - 1
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r in (r0, r1) or c in (c0, c1)) and (r, c) not in gaps:
                cells.add((r, c))
    return cells


def _padded_clear(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    gh, gw = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > gh or c0 + w > gw:
        return False
    rr0, cc0 = max(0, r0 - 1), max(0, c0 - 1)
    rr1, cc1 = min(gh, r0 + h + 1), min(gw, c0 + w + 1)
    return bool(np.all(grid[rr0:rr1, cc0:cc1] == BG))


def _paint_rect(grid: np.ndarray, record: dict) -> None:
    for r, c in record["wall"]:
        grid[r, c] = int(record["color"])


def _place_record(
    grid: np.ndarray,
    records: list[dict],
    r0: int,
    c0: int,
    h: int,
    w: int,
    color: int,
    gaps: set[tuple[int, int]],
    label: str,
) -> dict | None:
    if not _padded_clear(grid, r0, c0, h, w):
        return None
    bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
    if not _valid_gap_set(sorted(gaps), bbox):
        return None
    record = {
        "bbox": bbox,
        "color": int(color),
        "gaps": tuple(sorted(gaps)),
        "wall": _wall_cells(r0, c0, h, w, gaps),
        "label": label,
    }
    _paint_rect(grid, record)
    records.append(record)
    return record


def _aligned_geometry(
    src: tuple[int, int],
    direction: tuple[int, int],
    dist: int,
    h: int,
    w: int,
    offset: int,
    through: bool,
) -> tuple[int, int, set[tuple[int, int]], int]:
    sr, sc = src
    dr, dc = direction
    if (dr, dc) == (1, 0):
        r0, c0 = sr + dist, sc - offset
        gaps = {(r0, sc)}
        if through:
            gaps.add((r0 + h - 1, sc))
        far_dist = dist + h - 1
    elif (dr, dc) == (-1, 0):
        r0, c0 = sr - dist - h + 1, sc - offset
        gaps = {(r0 + h - 1, sc)}
        if through:
            gaps.add((r0, sc))
        far_dist = dist + h - 1
    elif (dr, dc) == (0, 1):
        r0, c0 = sr - offset, sc + dist
        gaps = {(sr, c0)}
        if through:
            gaps.add((sr, c0 + w - 1))
        far_dist = dist + w - 1
    else:
        r0, c0 = sr - offset, sc - dist - w + 1
        gaps = {(sr, c0 + w - 1)}
        if through:
            gaps.add((sr, c0))
        far_dist = dist + w - 1
    return r0, c0, gaps, far_dist


def _path_clear(
    grid: np.ndarray, src: tuple[int, int], direction: tuple[int, int], gap: tuple[int, int]
) -> bool:
    sr, sc = src
    dr, dc = direction
    r, c = sr + dr, sc + dc
    while (r, c) != gap:
        if not _in_bounds(grid.shape, r, c) or int(grid[r, c]) != BG:
            return False
        r += dr
        c += dc
    return True


def _place_aligned_rect(
    grid: np.ndarray,
    records: list[dict],
    src: tuple[int, int],
    direction: tuple[int, int],
    color: int,
    label: str,
    *,
    through: bool,
    min_dist: int,
    require_clear: bool = True,
) -> tuple[dict, int] | None:
    for _ in range(120):
        h = random.randint(4, 7)
        w = random.randint(4, 7)
        offset = random.randint(1, (w - 2) if direction[0] else (h - 2))
        dist = random.randint(min_dist, min_dist + 8)
        r0, c0, gaps, far_dist = _aligned_geometry(src, direction, dist, h, w, offset, through)
        entry_gap = min(gaps, key=lambda p: abs(p[0] - src[0]) + abs(p[1] - src[1]))
        if require_clear and not _path_clear(grid, src, direction, entry_gap):
            continue
        snapshot = grid.copy()
        record = _place_record(grid, records, r0, c0, h, w, color, gaps, label)
        if record is not None:
            return record, far_dist
        grid[:, :] = snapshot
    return None


def _random_gap_set(
    r0: int, c0: int, h: int, w: int, src: tuple[int, int]
) -> set[tuple[int, int]] | None:
    sr, sc = src
    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    for side in sides:
        if side in ("top", "bottom"):
            choices = [c for c in range(c0 + 1, c0 + w - 1) if c != sc]
            if not choices:
                continue
            c = random.choice(choices)
            if random.random() < 0.35:
                return {(r0, c), (r0 + h - 1, c)}
            return {((r0 if side == "top" else r0 + h - 1), c)}
        choices = [r for r in range(r0 + 1, r0 + h - 1) if r != sr]
        if not choices:
            continue
        r = random.choice(choices)
        if random.random() < 0.35:
            return {(r, c0), (r, c0 + w - 1)}
        return {(r, (c0 if side == "left" else c0 + w - 1))}
    return None


def _place_off_axis_rect(
    grid: np.ndarray,
    records: list[dict],
    src: tuple[int, int],
    color: int,
) -> bool:
    gh, gw = grid.shape
    for _ in range(180):
        h = random.randint(4, 7)
        w = random.randint(4, 7)
        r0 = random.randint(1, gh - h - 1)
        c0 = random.randint(1, gw - w - 1)
        gaps = _random_gap_set(r0, c0, h, w, src)
        if gaps is None:
            continue
        if _place_record(grid, records, r0, c0, h, w, color, gaps, "off_axis_unlit"):
            return True
    return False


def _place_blocker(
    grid: np.ndarray,
    src: tuple[int, int],
    direction: tuple[int, int],
    color: int,
) -> int | None:
    sr, sc = src
    dr, dc = direction
    for dist in random.sample(range(2, 8), 6):
        r, c = sr + dr * dist, sc + dc * dist
        if _in_bounds(grid.shape, r, c) and int(grid[r, c]) == BG:
            grid[r, c] = int(color)
            return dist
    return None


def _scatter_distractors(grid: np.ndarray, count: int) -> None:
    h, w = grid.shape
    for _ in range(count):
        color = random.choice(RECT_COLORS)
        for _try in range(80):
            r = random.randint(1, h - 2)
            c = random.randint(1, w - 2)
            if int(grid[r, c]) != BG:
                continue
            if any(int(grid[rr, cc]) != BG for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)):
                continue
            grid[r, c] = color
            break


def _record_lit(record: dict, out: np.ndarray) -> bool:
    r0, c0, r1, c1 = record["bbox"]
    return bool(np.all(out[r0 + 1 : r1, c0 + 1 : c1] == PINK))


def _detected_bboxes(grid: np.ndarray) -> set[tuple[int, int, int, int]]:
    return {tuple(rec["bbox"]) for rec in _find_rectangles(grid, _background(grid))}


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(24, 30)
    w = random.randint(24, 30)
    grid = np.full((h, w), BG, dtype=int)
    src = (random.randint(9, h - 10), random.randint(9, w - 10))
    grid[src] = GREY

    directions = list(DIRS)
    random.shuffle(directions)
    pass_dir, sink_dir, block_dir = directions[:3]
    colors = random.sample(RECT_COLORS, 5)
    records: list[dict] = []

    first = _place_aligned_rect(
        grid, records, src, pass_dir, colors[0], "pass_through_lit", through=True, min_dist=3
    )
    if first is None:
        return None
    _first_record, first_far = first

    second = _place_aligned_rect(
        grid,
        records,
        src,
        pass_dir,
        colors[1],
        "after_pass_lit",
        through=False,
        min_dist=first_far + random.randint(3, 5),
    )
    if second is None:
        return None

    direct = _place_aligned_rect(
        grid, records, src, sink_dir, colors[2], "direct_lit", through=False, min_dist=3
    )
    if direct is None:
        return None

    blocker_dist = _place_blocker(grid, src, block_dir, random.choice(RECT_COLORS))
    if blocker_dist is None:
        return None

    blocked = _place_aligned_rect(
        grid,
        records,
        src,
        block_dir,
        colors[3],
        "blocked_unlit",
        through=False,
        min_dist=blocker_dist + random.randint(3, 5),
        require_clear=False,
    )
    if blocked is None:
        return None

    if not _place_off_axis_rect(grid, records, src, colors[4]):
        return None

    _scatter_distractors(grid, random.randint(1, 4))

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None

    generated_bboxes = {tuple(rec["bbox"]) for rec in records}
    if not generated_bboxes <= _detected_bboxes(grid):
        return None

    lit_labels = {rec["label"] for rec in records if _record_lit(rec, out)}
    required_lit = {"pass_through_lit", "after_pass_lit", "direct_lit"}
    required_unlit = {"blocked_unlit", "off_axis_unlit"}
    if lit_labels != required_lit:
        return None
    if any(_record_lit(rec, out) for rec in records if rec["label"] in required_unlit):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gap-light input")
