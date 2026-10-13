from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_of_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _full_perimeter(bbox: BBox) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    cells: list[Coord] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _side_of_gap(bbox: BBox, rc: Coord) -> str:
    r0, c0, r1, c1 = bbox
    r, c = rc
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    raise AssertionError("gap must lie on the frame perimeter")


def _ray_start_and_step(bbox: BBox, gap: Coord) -> tuple[Coord, Coord]:
    r0, c0, r1, c1 = bbox
    r, c = gap
    side = _side_of_gap(bbox, gap)
    if side == "left":
        return (r, c0 - 1), (0, -1)
    if side == "right":
        return (r, c1 + 1), (0, 1)
    if side == "top":
        return (r0 - 1, c), (-1, 0)
    return (r1 + 1, c), (1, 0)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, BBox]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[int, BBox, int]] = []
    for color in sorted(set(map(int, g.ravel())) - {bg}):
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        bbox = _bbox_of_coords(coords)
        r0, c0, r1, c1 = bbox
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        perim = set(_full_perimeter(bbox))
        coord_set = {tuple(map(int, rc)) for rc in coords}
        if not coord_set <= perim:
            continue
        if any((corner not in coord_set) for corner in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))):
            continue
        missing = len(perim) - len(coord_set)
        if missing <= 0:
            continue
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        candidates.append((color, bbox, area))

    assert candidates, "expected a hollow rectangle frame with at least one gap"
    max_area = max(area for _color, _bbox, area in candidates)
    best = [(color, bbox) for color, bbox, area in candidates if area == max_area]
    assert len(best) == 1, "expected a unique largest gapped rectangular frame"
    return best[0]


def _interior_pixels(grid: np.ndarray, bbox: BBox, bg: int, frame_color: int) -> list[tuple[Coord, int]]:
    r0, c0, r1, c1 = bbox
    pixels: list[tuple[Coord, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(grid[r, c])
            if color not in (bg, frame_color):
                pixels.append(((r, c), color))
    return pixels


def _gap_alignment(
    bbox: BBox, gap: Coord, interior_pixels: list[tuple[Coord, int]]
) -> list[tuple[Coord, int]]:
    side = _side_of_gap(bbox, gap)
    r, c = gap
    if side in ("left", "right"):
        return [(rc, color) for rc, color in interior_pixels if rc[0] == r]
    return [(rc, color) for rc, color in interior_pixels if rc[1] == c]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decide each frame gap's fate: seal unmatched gaps and let uniquely aligned interior pixels escape as outward rays.

    Input structure:
    - The background contains one one-pixel-thick hollow rectangular frame with one-pixel gaps on its border.
    - The strict interior of the frame contains scattered colored single pixels.
    - Some outside cells may contain solid obstacle blocks that sit on potential ray paths.

    Rule:
    - Find every missing border cell of the rectangular frame.
    - For a left/right gap, check interior pixels on the same row; for a top/bottom gap, check interior pixels on the same column.
    - If exactly one interior pixel aligns to that gap, erase that pixel and draw its color outward from the first exterior cell until the grid edge or the cell before the first outside non-background blocker.
    - If no interior pixel aligns to the gap, fill that gap with the frame color.
    - Interior pixels that align to no gap stay where they are.

    Key insight:
    - The same visible object type, a border gap, has two different roles, and only the interior alignment relation decides whether it is an exit or merely a defect to repair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    frame_color, bbox = _find_frame(g, bg)
    r0, c0, r1, c1 = bbox
    perimeter = _full_perimeter(bbox)
    gaps = [rc for rc in perimeter if int(g[rc]) != frame_color]
    assert gaps, "expected at least one gap"

    interior = _interior_pixels(g, bbox, bg, frame_color)
    out = g.copy()

    outside_non_bg = np.zeros(g.shape, dtype=bool)
    outside_non_bg[:r0, :] = g[:r0, :] != bg
    outside_non_bg[r1 + 1 :, :] = g[r1 + 1 :, :] != bg
    outside_non_bg[r0 : r1 + 1, :c0] = g[r0 : r1 + 1, :c0] != bg
    outside_non_bg[r0 : r1 + 1, c1 + 1 :] = g[r0 : r1 + 1, c1 + 1 :] != bg

    used_pixels: set[Coord] = set()
    exit_count = 0
    fill_count = 0
    for gap in gaps:
        aligned = _gap_alignment(bbox, gap, interior)
        assert len(aligned) in (0, 1), "each gap must align to zero or one interior pixel"
        if not aligned:
            out[gap] = frame_color
            fill_count += 1
            continue

        (pixel_rc, ray_color) = aligned[0]
        assert pixel_rc not in used_pixels, "an interior pixel may feed only one gap"
        used_pixels.add(pixel_rc)
        out[pixel_rc] = bg
        exit_count += 1

        (rr, cc), (dr, dc) = _ray_start_and_step(bbox, gap)
        while _in_bounds(g.shape, rr, cc) and not outside_non_bg[rr, cc]:
            out[rr, cc] = ray_color
            rr += dr
            cc += dc

    assert exit_count >= 1 and fill_count >= 1, "expected both gap behaviors"
    return out


MIN_GRID = 16
MAX_GRID = 30
MIN_FRAME_H = 7
MAX_FRAME_H = 15
MIN_FRAME_W = 7
MAX_FRAME_W = 15
MAX_ATTEMPTS = 500


def _render_frame(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _available_side_positions(start: int, stop: int, used: set[int]) -> list[int]:
    vals = [v for v in range(start, stop + 1) if v not in used]
    random.shuffle(vals)
    return vals


def _pick_gap_set(bbox: BBox) -> tuple[list[tuple[str, Coord]], list[tuple[str, Coord]]]:
    r0, c0, r1, c1 = bbox
    interior_rows = list(range(r0 + 1, r1))
    interior_cols = list(range(c0 + 1, c1))
    max_row_gaps = len(interior_rows) - 1
    max_col_gaps = len(interior_cols) - 1
    max_total = min(7, max_row_gaps + max_col_gaps)
    assert max_total >= 3, "frame interior must support aligned, filler, and stray roles"

    total = random.randint(3, max_total)
    aligned_low = max(2, total - 3)
    aligned_high = min(4, total - 1)
    aligned_count = random.randint(aligned_low, aligned_high)
    filler_count = total - aligned_count

    row_gap_min = max(0, total - max_col_gaps)
    row_gap_max = min(max_row_gaps, total)
    row_gap_count = random.randint(row_gap_min, row_gap_max)
    col_gap_count = total - row_gap_count

    choices: list[tuple[str, Coord]] = []
    for row in random.sample(interior_rows, k=row_gap_count):
        side = random.choice(["left", "right"])
        choices.append((side, (row, c0 if side == "left" else c1)))
    for col in random.sample(interior_cols, k=col_gap_count):
        side = random.choice(["top", "bottom"])
        choices.append((side, (r0 if side == "top" else r1, col)))

    random.shuffle(choices)
    aligned_gaps = choices[:aligned_count]
    filler_gaps = choices[aligned_count:]
    assert len(filler_gaps) == filler_count
    return aligned_gaps, filler_gaps


def _candidate_cells_for_gap(
    bbox: BBox,
    gap: Coord,
    forbidden_rows: set[int],
    forbidden_cols: set[int],
    occupied: set[Coord],
) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    side = _side_of_gap(bbox, gap)
    cells: list[Coord] = []
    if side in ("left", "right"):
        row = gap[0]
        for c in range(c0 + 1, c1):
            rc = (row, c)
            if rc in occupied or c in forbidden_cols:
                continue
            cells.append(rc)
    else:
        col = gap[1]
        for r in range(r0 + 1, r1):
            rc = (r, col)
            if rc in occupied or r in forbidden_rows:
                continue
            cells.append(rc)
    random.shuffle(cells)
    return cells


def _candidate_stray_cells(
    bbox: BBox, forbidden_rows: set[int], forbidden_cols: set[int], occupied: set[Coord]
) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    cells = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if (r, c) not in occupied and r not in forbidden_rows and c not in forbidden_cols
    ]
    random.shuffle(cells)
    return cells


def _place_blocker(
    grid: np.ndarray,
    bbox: BBox,
    gap: Coord,
    color: int,
    reserved: set[Coord],
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    (sr, sc), (dr, dc) = _ray_start_and_step(bbox, gap)
    path: list[Coord] = []
    rr, cc = sr, sc
    while _in_bounds(grid.shape, rr, cc):
        path.append((rr, cc))
        rr += dr
        cc += dc
    if len(path) < 2:
        return None

    anchors = path[1:].copy()
    random.shuffle(anchors)
    for blocker_anchor in anchors:
        for _ in range(12):
            bh = random.randint(1, 3)
            bw = random.randint(1, 3)
            if dr == 0:
                row_lo = max(0, blocker_anchor[0] - bh + 1)
                row_hi = min(blocker_anchor[0], h - bh)
                if row_lo > row_hi:
                    continue
                br0 = random.randint(row_lo, row_hi)
                br1 = br0 + bh - 1
                if dc < 0:
                    bc1 = blocker_anchor[1]
                    bc0 = bc1 - bw + 1
                else:
                    bc0 = blocker_anchor[1]
                    bc1 = bc0 + bw - 1
            else:
                col_lo = max(0, blocker_anchor[1] - bw + 1)
                col_hi = min(blocker_anchor[1], w - bw)
                if col_lo > col_hi:
                    continue
                bc0 = random.randint(col_lo, col_hi)
                bc1 = bc0 + bw - 1
                if dr < 0:
                    br1 = blocker_anchor[0]
                    br0 = br1 - bh + 1
                else:
                    br0 = blocker_anchor[0]
                    br1 = br0 + bh - 1

            if br0 < 0 or bc0 < 0 or br1 >= h or bc1 >= w:
                continue
            cells = {(r, c) for r in range(br0, br1 + 1) for c in range(bc0, bc1 + 1)}
            if any(cell in reserved for cell in cells):
                continue
            r0, c0, r1, c1 = bbox
            if any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in cells):
                continue
            for r, c in cells:
                grid[r, c] = color
            reserved |= cells
            return br0, bc0, br1, bc1
    return None


def _ray_cells_until_block(grid: np.ndarray, bbox: BBox, gap: Coord, bg: int) -> list[Coord]:
    (rr, cc), (dr, dc) = _ray_start_and_step(bbox, gap)
    cells: list[Coord] = []
    while _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == bg:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _build_case() -> dict | None:
    h = random.randint(MIN_GRID, MAX_GRID)
    w = random.randint(MIN_GRID, MAX_GRID)
    bg = 0
    frame_color = random.choice(list(range(1, 10)))

    fh = random.randint(MIN_FRAME_H, min(MAX_FRAME_H, h - 4))
    fw = random.randint(MIN_FRAME_W, min(MAX_FRAME_W, w - 4))
    r0 = random.randint(2, h - fh - 2)
    c0 = random.randint(2, w - fw - 2)
    bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)

    grid = np.full((h, w), bg, dtype=int)
    _render_frame(grid, bbox, frame_color)

    aligned_gaps, filler_gaps = _pick_gap_set(bbox)
    all_gaps = aligned_gaps + filler_gaps
    for _side, gap in all_gaps:
        grid[gap] = bg

    gap_rows = {gap[0] for _side, gap in all_gaps if _side in ("left", "right")}
    gap_cols = {gap[1] for _side, gap in all_gaps if _side in ("top", "bottom")}

    occupied: set[Coord] = set()
    exit_pixels: dict[Coord, tuple[Coord, int]] = {}
    usable_colors = [c for c in range(1, 10) if c != frame_color]
    random.shuffle(usable_colors)
    needed_colors = len(aligned_gaps) + random.randint(1, 3)
    while len(usable_colors) < needed_colors:
        usable_colors.extend([c for c in range(1, 10) if c != frame_color])
    color_iter = iter(usable_colors)

    for _side, gap in aligned_gaps:
        cells = _candidate_cells_for_gap(bbox, gap, gap_rows, gap_cols, occupied)
        if not cells:
            return None
        pixel = cells[0]
        color = next(color_iter)
        grid[pixel] = color
        occupied.add(pixel)
        exit_pixels[gap] = (pixel, color)

    stray_candidates = _candidate_stray_cells(bbox, gap_rows, gap_cols, occupied)
    if not stray_candidates:
        return None
    stray_count = random.randint(1, min(3, len(stray_candidates)))
    stray_pixels: list[tuple[Coord, int]] = []
    for pixel in stray_candidates[:stray_count]:
        color = next(color_iter)
        grid[pixel] = color
        occupied.add(pixel)
        stray_pixels.append((pixel, color))

    blocker_colors = [c for c in range(1, 10) if c not in {bg, frame_color} and c not in {color for _pixel, color in exit_pixels.values()} and c not in {color for _pixel, color in stray_pixels}]
    if not blocker_colors:
        blocker_colors = [frame_color]
    random.shuffle(blocker_colors)

    reserved_outside: set[Coord] = set()
    blocked_subset = random.sample(aligned_gaps, k=random.randint(1, len(aligned_gaps)))
    if len(aligned_gaps) >= 2 and len(blocked_subset) == len(aligned_gaps):
        blocked_subset = blocked_subset[:-1]
    if len(aligned_gaps) >= 2 and not blocked_subset:
        blocked_subset = [aligned_gaps[0]]
    blockers: dict[Coord, BBox] = {}
    for idx, (_side, gap) in enumerate(blocked_subset):
        blocker = _place_blocker(grid, bbox, gap, blocker_colors[idx % len(blocker_colors)], reserved_outside)
        if blocker is None:
            return None
        blockers[gap] = blocker

    return {
        "input": grid,
        "bg": bg,
        "frame_color": frame_color,
        "bbox": bbox,
        "aligned_gaps": [gap for _side, gap in aligned_gaps],
        "filler_gaps": [gap for _side, gap in filler_gaps],
        "exit_pixels": exit_pixels,
        "stray_pixels": stray_pixels,
        "blocked_gaps": set(blockers),
    }


def _expected_from_spec(spec: dict) -> np.ndarray:
    grid = spec["input"]
    out = grid.copy()
    bg = spec["bg"]
    bbox = spec["bbox"]
    frame_color = spec["frame_color"]

    for gap in spec["filler_gaps"]:
        out[gap] = frame_color
    for gap, (pixel, color) in spec["exit_pixels"].items():
        out[pixel] = bg
        for rr, cc in _ray_cells_until_block(grid, bbox, gap, bg):
            out[rr, cc] = color
    return out


def _validate_spec(spec: dict) -> None:
    grid = spec["input"]
    bg = spec["bg"]
    bbox = spec["bbox"]
    frame_color = spec["frame_color"]
    aligned_gaps = spec["aligned_gaps"]
    filler_gaps = spec["filler_gaps"]
    exit_pixels = spec["exit_pixels"]
    stray_pixels = spec["stray_pixels"]
    blocked_gaps = spec["blocked_gaps"]

    assert aligned_gaps and filler_gaps and stray_pixels
    assert blocked_gaps and len(blocked_gaps) < len(aligned_gaps)

    frame_from_solver = _find_frame(grid, bg)
    assert frame_from_solver == (frame_color, bbox)

    interior = _interior_pixels(grid, bbox, bg, frame_color)
    interior_map = {rc: color for rc, color in interior}
    assert len(interior_map) == len(interior)
    for gap in aligned_gaps:
        aligned = _gap_alignment(bbox, gap, interior)
        assert len(aligned) == 1
        assert aligned[0] == exit_pixels[gap]
    for gap in filler_gaps:
        assert not _gap_alignment(bbox, gap, interior)
    for pixel, _color in stray_pixels:
        assert all(pixel != exit_pixel for exit_pixel, _color2 in exit_pixels.values())
        assert pixel[0] not in {gap[0] for gap in aligned_gaps if _side_of_gap(bbox, gap) in ("left", "right")}
        assert pixel[1] not in {gap[1] for gap in aligned_gaps if _side_of_gap(bbox, gap) in ("top", "bottom")}
        assert pixel[0] not in {gap[0] for gap in filler_gaps if _side_of_gap(bbox, gap) in ("left", "right")}
        assert pixel[1] not in {gap[1] for gap in filler_gaps if _side_of_gap(bbox, gap) in ("top", "bottom")}

    expected = _expected_from_spec(spec)
    solved = main(grid)
    assert np.array_equal(solved, expected)

    fill_only = grid.copy()
    for gap in aligned_gaps + filler_gaps:
        fill_only[gap] = frame_color
    assert not np.array_equal(fill_only, expected), "aligned-gap branch must matter"

    no_fill = expected.copy()
    for gap in filler_gaps:
        no_fill[gap] = bg
    assert not np.array_equal(no_fill, expected), "gap-repair branch must matter"

    ignore_blockers = expected.copy()
    for gap in blocked_gaps:
        pixel, color = exit_pixels[gap]
        (rr, cc), (dr, dc) = _ray_start_and_step(bbox, gap)
        while _in_bounds(grid.shape, rr, cc):
            if int(grid[rr, cc]) == bg:
                ignore_blockers[rr, cc] = color
            rr += dr
            cc += dc
    assert not np.array_equal(ignore_blockers, expected), "blocker logic must matter"


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        spec = _build_case()
        if spec is None:
            continue
        try:
            _validate_spec(spec)
        except AssertionError:
            continue
        return spec["input"]
    raise RuntimeError("failed to generate a valid gap-fate ARC input")
