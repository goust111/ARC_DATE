from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

PAIR_COLORS = (BLUE, RED, GREEN)
MARKER_COLORS = (YELLOW, ORANGE)
RECT_SIZES = ((1, 1), (1, 2), (2, 1), (2, 2))

MAX_GENERATION_ATTEMPTS = 400
FRAME_ATTEMPTS = 200
MIN_REGION_AREA = 5


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _outline_from_region(region: np.ndarray) -> np.ndarray:
    h, w = region.shape
    outline = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not region[r, c]:
                continue
            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                outline[r, c] = True
                continue
            for nr, nc in neighbors4(r, c):
                if not region[nr, nc]:
                    outline[r, c] = True
                    break
    return outline


def _outside_reachable(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if in_bounds(passable.shape, nr, nc) and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _enclosed_regions_from_wall(wall: np.ndarray) -> list[np.ndarray]:
    padded_wall = np.pad(wall.astype(bool), 1, constant_values=False)
    passable = ~padded_wall
    outside = _outside_reachable(passable)
    enclosed = passable & ~outside
    inner = enclosed[1:-1, 1:-1]
    return _components(inner)


def _marker_fill_color(grid: np.ndarray, region: np.ndarray) -> int:
    markers = [(int(r), int(c), int(grid[r, c])) for r, c in region if int(grid[r, c]) in MARKER_COLORS]
    assert markers, "each enclosed region must contain markers"
    counts = Counter(color for _r, _c, color in markers)
    if counts[YELLOW] > counts[ORANGE]:
        return YELLOW
    if counts[ORANGE] > counts[YELLOW]:
        return ORANGE
    return min(markers)[2]


def _boxes_and_line(
    start: tuple[int, int],
    end: tuple[int, int],
    upper_size: tuple[int, int],
    lower_size: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], int]:
    sr, sc = start
    er, ec = end
    if sr >= er:
        raise ValueError("upper corner must be above lower corner")
    if abs(er - sr) != abs(ec - sc):
        raise ValueError("corners must lie on a 45-degree line")

    sign = 1 if ec > sc else -1
    uh, uw = upper_size
    lh, lw = lower_size

    if sign == 1:
        ur0, uc0 = sr - uh + 1, sc - uw + 1
        lr0, lc0 = er, ec
    else:
        ur0, uc0 = sr - uh + 1, sc
        lr0, lc0 = er, ec - lw + 1

    upper = [(r, c) for r in range(ur0, ur0 + uh) for c in range(uc0, uc0 + uw)]
    lower = [(r, c) for r in range(lr0, lr0 + lh) for c in range(lc0, lc0 + lw)]
    line = [(sr + k, sc + sign * k) for k in range(1, er - sr)]
    return upper, lower, line, sign


def _find_components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    return _components(grid == int(color))


def _facing_corners(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    if box_a[0] <= box_b[0]:
        upper, lower = box_a, box_b
    else:
        upper, lower = box_b, box_a

    ur0, uc0, ur1, uc1 = upper
    lr0, lc0, _lr1, lc1 = lower
    assert ur1 < lr0, "same-color rectangles must be diagonally separated in rows"
    if uc1 < lc0:
        return (ur1, uc1), (lr0, lc0)
    if lc1 < uc0:
        return (ur1, uc0), (lr0, lc1)
    raise AssertionError("same-color rectangles must be diagonally separated in columns")


def _grid_with_reconstructed_connectors(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    connectors: list[dict] = []

    for color in sorted(c for c in np.unique(g) if int(c) not in (BG, GREY, YELLOW, ORANGE)):
        components = _find_components_of_color(g, int(color))
        assert len(components) == 2, "each pair color must appear as exactly two components"
        boxes = sorted((_bbox(comp) for comp in components), key=lambda b: (b[0], b[1]))
        start, end = _facing_corners(boxes[0], boxes[1])

        dr = 1
        dc = 1 if end[1] > start[1] else -1
        line: list[tuple[int, int]] = []
        r, c = start[0] + dr, start[1] + dc
        while (r, c) != end:
            if int(out[r, c]) != BG:
                break
            out[r, c] = int(color)
            line.append((r, c))
            r += dr
            c += dc

        connectors.append({"color": int(color), "boxes": boxes, "line": line})
    return out, connectors


def _fill_from_markers(original: np.ndarray, with_lines: np.ndarray) -> np.ndarray:
    out = with_lines.copy()
    wall = (with_lines != BG) & ~np.isin(with_lines, MARKER_COLORS)
    enclosed = ~wall & ~_outside_reachable(~wall)

    for region in _components(enclosed):
        fill_color = _marker_fill_color(original, region)
        for r, c in region:
            if int(original[r, c]) == BG:
                out[r, c] = fill_color

    out[np.isin(original, MARKER_COLORS)] = BG
    return out


def _solve_with_explicit_connectors(grid: np.ndarray, connectors: list[dict]) -> np.ndarray:
    with_lines = np.asarray(grid, dtype=int).copy()
    for connector in connectors:
        for r, c in connector["line"]:
            assert int(with_lines[r, c]) == BG
            with_lines[r, c] = int(connector["color"])
    return _fill_from_markers(np.asarray(grid, dtype=int), with_lines)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct diagonal partitions inside grey outlines, then flood-fill each newly enclosed subregion by its local marker majority.

    Input structure:
    - The grid has a black background, one or two grey 1-pixel-thick closed outline frames, and 2-3 same-color rectangle pairs inside those outlines.
    - Each pair uses one of the pair colors and appears as exactly two solid 1x1 to 2x2 rectangles placed on a common 45-degree diagonal offset.
    - Enclosed open space also contains isolated yellow/orange marker pixels that act as fill votes but are not walls.

    Rule:
    - Keep all grey outline pixels and rectangle pixels unchanged.
    - For each pair color, connect the facing nearest corners of the two rectangles along the 45-degree diagonal, excluding the endpoint corner cells already occupied by the rectangles; if the path hits a non-background cell, stop immediately before that obstacle.
    - Treat grey outline pixels, rectangle pixels, and the reconstructed diagonal pixels as walls; flood from the outer border through the remaining non-wall cells, and every unreachable connected open region is a closed subregion.
    - Fill only the black cells of each closed subregion with the majority marker color inside it, break yellow/orange ties by the uppermost marker, and finally erase all marker pixels back to black.

    Key insight:
    - The diagonals are not decorative output details: they change the topology of the open interior, which changes which markers belong to each enclosed subregion and therefore changes the fill color assignment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    with_lines, _connectors = _grid_with_reconstructed_connectors(g)
    return _fill_from_markers(g, with_lines)


def _rectangle_region(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _l_region(h: int, w: int) -> np.ndarray:
    region = np.ones((h, w), dtype=bool)
    corner = random.choice(("tl", "tr", "bl", "br"))
    cut_h = random.randint(3, h - 5)
    cut_w = random.randint(3, w - 5)
    if corner == "tl":
        region[:cut_h, :cut_w] = False
    elif corner == "tr":
        region[:cut_h, w - cut_w :] = False
    elif corner == "bl":
        region[h - cut_h :, :cut_w] = False
    else:
        region[h - cut_h :, w - cut_w :] = False
    return region


def _shape_region(frame_kind: str, h: int, w: int) -> np.ndarray:
    if frame_kind == "rect":
        return _rectangle_region(h, w)
    if frame_kind == "l":
        return _l_region(h, w)
    raise ValueError(frame_kind)


def _adjacent_to_outline(interior: np.ndarray, outline: np.ndarray) -> list[tuple[int, int]]:
    anchors: list[tuple[int, int]] = []
    h, w = interior.shape
    for r in range(h):
        for c in range(w):
            if not interior[r, c]:
                continue
            if any(in_bounds(interior.shape, nr, nc) and outline[nr, nc] for nr, nc in neighbors4(r, c)):
                anchors.append((r, c))
    return anchors


def _cells_fit(cells: list[tuple[int, int]], interior: np.ndarray, occupied: np.ndarray) -> bool:
    for r, c in cells:
        if not in_bounds(interior.shape, r, c):
            return False
        if not interior[r, c]:
            return False
        if occupied[r, c]:
            return False
    return True


def _choose_connector_candidate(region: np.ndarray, outline: np.ndarray, occupied: np.ndarray, current_regions: int) -> dict | None:
    interior = region & ~outline
    anchors = _adjacent_to_outline(interior, outline)
    anchor_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for start in anchors:
        for end in anchors:
            if end[0] <= start[0]:
                continue
            dr = end[0] - start[0]
            dc = end[1] - start[1]
            if abs(dr) != abs(dc):
                continue
            if dr < 2:
                continue
            anchor_pairs.append((start, end))
    random.shuffle(anchor_pairs)

    size_combos = [(a, b) for a in RECT_SIZES for b in RECT_SIZES]
    for start, end in anchor_pairs:
        combos = size_combos[:]
        random.shuffle(combos)
        for upper_size, lower_size in combos:
            try:
                upper, lower, line, sign = _boxes_and_line(start, end, upper_size, lower_size)
            except ValueError:
                continue
            if not line:
                continue
            if not _cells_fit(upper, interior, occupied):
                continue
            if not _cells_fit(lower, interior, occupied):
                continue
            if set(upper) & set(lower):
                continue
            if not _cells_fit(line, interior, occupied):
                continue

            trial = occupied.copy()
            for cells in (upper, lower, line):
                for r, c in cells:
                    trial[r, c] = True
            regions = _enclosed_regions_from_wall(trial)
            if len(regions) != current_regions + 1:
                continue
            if min(len(comp) for comp in regions) < MIN_REGION_AREA:
                continue
            return {
                "start": start,
                "end": end,
                "upper_size": upper_size,
                "lower_size": lower_size,
                "upper": upper,
                "lower": lower,
                "line": line,
                "sign": sign,
            }
    return None


def _sample_spread_cells(region_cells: list[tuple[int, int]], k: int) -> list[tuple[int, int]]:
    cells = region_cells[:]
    random.shuffle(cells)
    if not cells:
        return []
    chosen = [cells.pop()]
    while cells and len(chosen) < k:
        best = max(
            cells,
            key=lambda cell: min(abs(cell[0] - rr) + abs(cell[1] - cc) for rr, cc in chosen),
        )
        chosen.append(best)
        cells.remove(best)
    return chosen


def _place_markers(regions: list[np.ndarray]) -> list[tuple[int, int, int]] | None:
    n_regions = len(regions)
    desired_fills = [random.choice(MARKER_COLORS) for _ in range(n_regions)]
    if n_regions > 1 and len(set(desired_fills)) == 1:
        idx = random.randrange(n_regions)
        desired_fills[idx] = ORANGE if desired_fills[idx] == YELLOW else YELLOW

    tie_candidates = [
        idx
        for idx, region in enumerate(regions)
        if len(region) >= 3 and len({int(r) for r, _c in region}) >= 2
    ]
    tie_idx = random.choice(tie_candidates) if tie_candidates and random.random() < 0.45 else None

    markers: list[tuple[int, int, int]] = []
    for idx, region in enumerate(regions):
        cells = [(int(r), int(c)) for r, c in region]
        desired = desired_fills[idx]
        other = ORANGE if desired == YELLOW else YELLOW
        max_markers = min(3, len(cells) - 1)
        if max_markers <= 0:
            return None

        if idx == tie_idx:
            rows = sorted({r for r, _c in cells})
            top_rows = [row for row in rows[:-1]]
            if not top_rows:
                return None
            top_row = random.choice(top_rows)
            lower_rows = [row for row in rows if row > top_row]
            if not lower_rows:
                return None
            low_row = random.choice(lower_rows)
            top_cell = random.choice([cell for cell in cells if cell[0] == top_row])
            low_cell = random.choice([cell for cell in cells if cell[0] == low_row and cell != top_cell])
            markers.append((top_cell[0], top_cell[1], desired))
            markers.append((low_cell[0], low_cell[1], other))
            continue

        total = 1 if max_markers == 1 else random.randint(1, max_markers)
        picked = _sample_spread_cells(cells, total)
        if len(picked) != total:
            return None
        if total == 1:
            markers.append((picked[0][0], picked[0][1], desired))
            continue

        major_count = 2 if total >= 2 and random.random() < 0.7 else total
        major_count = min(major_count, total)
        if total - major_count >= major_count:
            major_count = total - 1
        for pos, cell in enumerate(picked):
            color = desired if pos < major_count else other
            markers.append((cell[0], cell[1], color))
    return markers


def _generate_frame_patch(pair_colors: list[int], frame_kind: str, small: bool) -> dict | None:
    for _ in range(FRAME_ATTEMPTS):
        if small:
            h = random.randint(10, 12 if frame_kind == "rect" else 13)
            w = random.randint(10, 12 if frame_kind == "rect" else 13)
        else:
            h = random.randint(11, 14 if frame_kind == "rect" else 15)
            w = random.randint(11, 14 if frame_kind == "rect" else 15)

        region = _shape_region(frame_kind, h, w)
        outline = _outline_from_region(region)
        interior = region & ~outline
        if interior.sum() < 30:
            continue

        occupied = outline.copy()
        connectors: list[dict] = []
        current_regions = len(_enclosed_regions_from_wall(occupied))
        if current_regions != 1:
            continue

        ok = True
        for color in pair_colors:
            candidate = _choose_connector_candidate(region, outline, occupied, current_regions)
            if candidate is None:
                ok = False
                break
            candidate["color"] = int(color)
            connectors.append(candidate)
            for cells in (candidate["upper"], candidate["lower"], candidate["line"]):
                for r, c in cells:
                    occupied[r, c] = True
            current_regions += 1
        if not ok:
            continue

        regions = _enclosed_regions_from_wall(occupied)
        if len(regions) != len(pair_colors) + 1:
            continue
        if min(len(comp) for comp in regions) < MIN_REGION_AREA:
            continue

        markers = _place_markers(regions)
        if markers is None:
            continue

        grid = np.zeros((h, w), dtype=int)
        grid[outline] = GREY
        for connector in connectors:
            for r, c in connector["upper"] + connector["lower"]:
                grid[r, c] = connector["color"]
        for r, c, color in markers:
            if grid[r, c] != BG:
                ok = False
                break
            grid[r, c] = color
        if not ok:
            continue

        return {
            "grid": grid,
            "connectors": connectors,
            "has_large_rect": any(
                connector["upper_size"] == (2, 2)
                or connector["lower_size"] == (2, 2)
                for connector in connectors
            ),
            "kind": frame_kind,
        }
    return None


def _compose_scene(patches: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    if len(patches) == 1:
        patch = patches[0]
        ph, pw = patch["grid"].shape
        top = random.randint(1, 4)
        left = random.randint(1, 4)
        bottom = random.randint(1, 4)
        right = random.randint(1, 4)
        h = ph + top + bottom
        w = pw + left + right
        if max(h, w) > 30:
            return None
        grid = np.zeros((h, w), dtype=int)
        grid[top : top + ph, left : left + pw] = patch["grid"]
        connectors = []
        for connector in patch["connectors"]:
            connectors.append(
                {
                    "color": connector["color"],
                    "line": [(r + top, c + left) for r, c in connector["line"]],
                }
            )
        return grid, connectors

    first, second = patches
    gap = random.randint(2, 4)
    orientation = random.choice(("h", "v"))
    ah, aw = first["grid"].shape
    bh, bw = second["grid"].shape

    if orientation == "h":
        top_a = random.randint(1, 4)
        top_b = random.randint(1, 4)
        left = random.randint(1, 3)
        right = random.randint(1, 3)
        h = max(top_a + ah, top_b + bh) + random.randint(1, 4)
        w = left + aw + gap + bw + right
        if max(h, w) > 30:
            return None
        apos = (top_a, left)
        bpos = (top_b, left + aw + gap)
    else:
        left_a = random.randint(1, 4)
        left_b = random.randint(1, 4)
        top = random.randint(1, 3)
        bottom = random.randint(1, 3)
        h = top + ah + gap + bh + bottom
        w = max(left_a + aw, left_b + bw) + random.randint(1, 4)
        if max(h, w) > 30:
            return None
        apos = (top, left_a)
        bpos = (top + ah + gap, left_b)

    grid = np.zeros((h, w), dtype=int)
    for patch, (r0, c0) in ((first, apos), (second, bpos)):
        ph, pw = patch["grid"].shape
        if np.any(grid[r0 : r0 + ph, c0 : c0 + pw] != BG):
            return None
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch["grid"]

    connectors = []
    for patch, (r0, c0) in ((first, apos), (second, bpos)):
        for connector in patch["connectors"]:
            connectors.append(
                {
                    "color": connector["color"],
                    "line": [(r + r0, c + c0) for r, c in connector["line"]],
                }
            )
    return grid, connectors


def _global_region_count(grid: np.ndarray, connectors: list[dict]) -> int:
    with_lines = np.asarray(grid, dtype=int).copy()
    for connector in connectors:
        for r, c in connector["line"]:
            with_lines[r, c] = connector["color"]
    wall = (with_lines != BG) & ~np.isin(with_lines, MARKER_COLORS)
    enclosed = ~wall & ~_outside_reachable(~wall)
    return len(_components(enclosed))


def _valid_scene(grid: np.ndarray, connectors: list[dict]) -> bool:
    expected = _solve_with_explicit_connectors(grid, connectors)
    if not np.array_equal(main(grid), expected):
        return False

    full_count = _global_region_count(grid, connectors)
    if full_count < 2:
        return False

    if _global_region_count(grid, []) >= full_count:
        return False

    changed = np.count_nonzero(expected != grid)
    if changed < 10:
        return False

    for idx in range(len(connectors)):
        reduced = connectors[:idx] + connectors[idx + 1 :]
        if _global_region_count(grid, reduced) >= full_count:
            return False
        if np.array_equal(_solve_with_explicit_connectors(grid, reduced), expected):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    n_frames = 2 if random.random() < 0.35 else 1
    total_pairs = 3 if random.random() < 0.45 else 2
    if n_frames == 1:
        frame_pair_counts = [total_pairs]
    elif total_pairs == 2:
        frame_pair_counts = [1, 1]
    else:
        frame_pair_counts = random.choice(([2, 1], [1, 2]))

    chosen_colors = random.sample(list(PAIR_COLORS), total_pairs)
    patches: list[dict] = []
    color_idx = 0
    for frame_pairs in frame_pair_counts:
        pair_colors = chosen_colors[color_idx : color_idx + frame_pairs]
        color_idx += frame_pairs
        frame_kind = "l" if random.random() < 0.4 else "rect"
        patch = _generate_frame_patch(pair_colors, frame_kind, small=(n_frames == 2))
        if patch is None:
            return None
        patches.append(patch)

    if not any(patch["kind"] == "l" for patch in patches) and random.random() < 0.25:
        return None
    if not any(patch["has_large_rect"] for patch in patches) and random.random() < 0.35:
        return None

    composed = _compose_scene(patches)
    if composed is None:
        return None
    grid, connectors = composed

    if not _valid_scene(grid, connectors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-partition flood-fill puzzle")
