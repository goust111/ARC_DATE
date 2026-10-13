from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
BLOCK = 2
WALL = 5
MARKER_COLORS = (1, 3, 4, 6, 7)

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in DIRS.values():
                    nr, nc = cr + dr, cc + dc
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_wall_side(grid: np.ndarray) -> str:
    sides = []
    if np.all(grid[0, :] == WALL):
        sides.append("top")
    if np.all(grid[-1, :] == WALL):
        sides.append("bottom")
    if np.all(grid[:, 0] == WALL):
        sides.append("left")
    if np.all(grid[:, -1] == WALL):
        sides.append("right")
    assert len(sides) == 1, "expected exactly one full grey edge bar"
    return sides[0]


def _find_blocks(grid: np.ndarray) -> list[tuple[int, int]]:
    comps = _components(grid == BLOCK)
    blocks: list[tuple[int, int]] = []
    for comp in comps:
        assert len(comp) == 4, "expected red blocks to be 2x2"
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert r1 == r0 + 1 and c1 == c0 + 1, "expected red blocks to be solid 2x2"
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                assert int(grid[rr, cc]) == BLOCK
        blocks.append((r0, c0))
    assert 2 <= len(blocks) <= 3, "expected 2-3 red 2x2 blocks"
    return sorted(blocks)


def _find_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    markers = []
    for r, c in np.argwhere((grid != BG) & (grid != BLOCK) & (grid != WALL)):
        markers.append((int(r), int(c), int(grid[r, c])))
    colors = [color for _, _, color in markers]
    assert 3 <= len(markers) <= 5, "expected 3-5 singleton markers"
    assert len(colors) == len(set(colors)), "expected distinct marker colors"
    return sorted(markers)


def _path_cells(src: tuple[int, int], dst: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = src
    dr = 0 if dst[0] == sr else (1 if dst[0] > sr else -1)
    dc = 0 if dst[1] == sc else (1 if dst[1] > sc else -1)
    cells = []
    cr, cc = sr + dr, sc + dc
    while (cr, cc) != dst:
        cells.append((cr, cc))
        cr += dr
        cc += dc
    cells.append(dst)
    return cells


def _candidate_to_block(
    grid: np.ndarray, marker: tuple[int, int], block: tuple[int, int]
) -> tuple[int, tuple[int, int], str] | None:
    r, c = marker
    br, bc = block

    options: list[tuple[int, tuple[int, int], str]] = []
    if r in (br, br + 1):
        if c < bc:
            anchor = (r, bc - 1)
            if anchor[1] >= 0 and all(int(grid[rr, cc]) == BG for rr, cc in _path_cells((r, c), anchor)):
                options.append((anchor[1] - c, anchor, "right"))
        elif c > bc + 1:
            anchor = (r, bc + 2)
            if anchor[1] < grid.shape[1] and all(
                int(grid[rr, cc]) == BG for rr, cc in _path_cells((r, c), anchor)
            ):
                options.append((c - anchor[1], anchor, "left"))

    if c in (bc, bc + 1):
        if r < br:
            anchor = (br - 1, c)
            if anchor[0] >= 0 and all(int(grid[rr, cc]) == BG for rr, cc in _path_cells((r, c), anchor)):
                options.append((anchor[0] - r, anchor, "down"))
        elif r > br + 1:
            anchor = (br + 2, c)
            if anchor[0] < grid.shape[0] and all(
                int(grid[rr, cc]) == BG for rr, cc in _path_cells((r, c), anchor)
            ):
                options.append((r - anchor[0], anchor, "up"))

    if not options:
        return None
    assert len(options) == 1
    return options[0]


def _best_candidate(
    grid: np.ndarray, marker: tuple[int, int], blocks: list[tuple[int, int]]
) -> dict[str, object]:
    candidates = []
    for idx, block in enumerate(blocks):
        cand = _candidate_to_block(grid, marker, block)
        if cand is None:
            continue
        distance, anchor, slide_dir = cand
        candidates.append(
            {
                "distance": int(distance),
                "anchor": anchor,
                "slide_dir": slide_dir,
                "block_index": idx,
            }
        )
    assert candidates, "each marker must see a block"
    candidates.sort(key=lambda item: int(item["distance"]))
    if len(candidates) >= 2:
        assert candidates[0]["distance"] != candidates[1]["distance"], "nearest block must be unique"
    return candidates[0]


def _analyze_grid(grid: np.ndarray) -> list[dict[str, object]]:
    _find_wall_side(grid)
    blocks = _find_blocks(grid)
    markers = _find_markers(grid)
    infos = []
    for r, c, color in markers:
        best = _best_candidate(grid, (r, c), blocks)
        infos.append(
            {
                "src": (r, c),
                "color": color,
                "distance": int(best["distance"]),
                "anchor": tuple(best["anchor"]),
                "slide_dir": str(best["slide_dir"]),
                "grow_dir": OPPOSITE[str(best["slide_dir"])],
                "block_index": int(best["block_index"]),
            }
        )
    return infos


def _simulate_growth(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    infos = _analyze_grid(g)
    infos.sort(key=lambda item: (-int(item["distance"]), item["src"][0], item["src"][1]))

    out = g.copy()
    for info in infos:
        sr, sc = info["src"]
        out[sr, sc] = BG

    grown = np.zeros_like(out, dtype=bool)
    events: list[dict[str, object]] = []

    for info in infos:
        ar, ac = info["anchor"]
        color = int(info["color"])
        dr, dc = DIRS[str(info["grow_dir"])]

        assert int(out[ar, ac]) == BG and not grown[ar, ac], "growth anchor must be free"
        out[ar, ac] = color
        grown[ar, ac] = True

        cr, cc = ar, ac
        stop = "boundary"
        while True:
            nr, nc = cr + dr, cc + dc
            if not in_bounds(out.shape, nr, nc):
                stop = "boundary"
                break
            if grown[nr, nc]:
                stop = "line"
                break
            if int(out[nr, nc]) == WALL:
                stop = "wall"
                break
            if int(out[nr, nc]) == BLOCK:
                stop = "block"
                break
            if int(out[nr, nc]) != BG:
                stop = "occupied"
                break
            out[nr, nc] = color
            grown[nr, nc] = True
            cr, cc = nr, nc

        events.append({**info, "stop": stop})

    return out, events


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide singleton markers to the nearest red block, then grow each marker backward from its docking point until blocked.

    Input structure
    - The background is black.
    - One full outer edge is a grey bar.
    - Two or three red 2x2 solid blocks sit in the interior.
    - Three to five distinct-colored singleton markers each have exactly one clear horizontal or vertical sightline to a nearest red block.

    Rule
    - For each marker, find the unique nearest red 2x2 block reachable along an unobstructed orthogonal sightline.
    - Move the marker to the background cell directly adjacent to that block on the same row or column, clearing the original source cell.
    - Reverse the slide direction and grow a line of the marker's color from the docking cell.
    - Grow markers in descending slide-distance order, breaking ties by row then column of the original marker.
    - Stop before the next step would leave the grid, hit the grey edge bar, or hit a line grown earlier; keep red blocks and the grey bar unchanged.

    Key insight
    - The slide identifies both the anchor cell and the line direction: every line grows exactly opposite to its marker's incoming motion, and the growth order controls which later lines get cut off by earlier ones.
    """
    out, _events = _simulate_growth(np.asarray(grid, dtype=int))
    return out


def _can_place_block(
    blocks: Iterable[tuple[int, int]],
    candidate: tuple[int, int],
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
) -> bool:
    h, w = shape
    r, c = candidate
    if r < 1 or c < 1 or r + 1 >= h - 1 or c + 1 >= w - 1:
        return False
    cells = {(r + dr, c + dc) for dr in range(2) for dc in range(2)}
    if cells & forbidden:
        return False
    for br, bc in blocks:
        for rr in range(br - 1, br + 3):
            for cc in range(bc - 1, bc + 3):
                if (rr, cc) in cells:
                    return False
    return True


def _markerless_grid(grid: np.ndarray) -> np.ndarray:
    base = grid.copy()
    base[(base != BG) & (base != BLOCK) & (base != WALL)] = BG
    return base


def _static_growth_room(grid: np.ndarray, anchor: tuple[int, int], grow_dir: str) -> int:
    base = _markerless_grid(grid)
    dr, dc = DIRS[grow_dir]
    r, c = anchor
    length = 0
    while True:
        r += dr
        c += dc
        if not in_bounds(base.shape, r, c):
            return length
        if int(base[r, c]) in (WALL, BLOCK):
            return length
        length += 1


def _enumerate_extra_markers(
    grid: np.ndarray, blocks: list[tuple[int, int]]
) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    options: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(grid == BG):
        rr, cc = int(r), int(c)
        try:
            best = _best_candidate(grid, (rr, cc), blocks)
        except AssertionError:
            continue
        if int(best["distance"]) < 2:
            continue
        if _static_growth_room(grid, tuple(best["anchor"]), OPPOSITE[str(best["slide_dir"])]) < 1:
            continue
        options.append((rr, cc, int(best["block_index"])))
    random.shuffle(options)
    return options


def _build_canonical_attempt() -> np.ndarray | None:
    h = random.randint(11, 14)
    w = random.randint(11, 14)
    grid = np.zeros((h, w), dtype=int)
    grid[-1, :] = WALL

    r_a = random.randint(4, h - 5)
    lane_r = r_a + random.randint(0, 1)
    c_a = random.randint(1, w - 7)
    anchor_a = (lane_r, c_a + 2)

    min_cross_c = anchor_a[1] + 2
    max_cross_c = w - 3
    if min_cross_c > max_cross_c:
        return None
    cross_c = random.randint(min_cross_c, max_cross_c)

    min_src_a = max(cross_c + 1, anchor_a[1] + 3)
    max_src_a = w - 2
    if min_src_a > max_src_a:
        return None
    src_a = (lane_r, random.randint(min_src_a, max_src_a))
    dist_a = src_a[1] - anchor_a[1]

    max_r_b = lane_r - 4
    if max_r_b < 1:
        return None
    c_b = random.choice([cross_c - 1, cross_c])
    if not (1 <= c_b <= w - 3):
        return None
    r_b = random.randint(1, max_r_b)
    anchor_b = (r_b + 2, cross_c)
    max_dist_b = min(dist_a - 1, lane_r - anchor_b[0] - 1)
    if max_dist_b < 1:
        return None
    dist_b = random.randint(1, max_dist_b)
    src_b = (anchor_b[0] + dist_b, cross_c)

    blocks = [(r_a, c_a), (r_b, c_b)]
    for br, bc in blocks:
        grid[br : br + 2, bc : bc + 2] = BLOCK

    planned = [src_a, src_b]
    forbidden_block_cells = {
        (lane_r, cc) for cc in range(anchor_a[1] + 1, w)
    } | {
        (rr, cross_c) for rr in range(anchor_b[0] + 1, h - 1)
    }

    if random.random() < 0.45:
        candidates = [
            (r, c)
            for r in range(1, h - 2)
            for c in range(1, w - 2)
            if _can_place_block(blocks, (r, c), grid.shape, forbidden_block_cells)
        ]
        random.shuffle(candidates)
        for extra_block in candidates:
            trial = grid.copy()
            er, ec = extra_block
            trial[er : er + 2, ec : ec + 2] = BLOCK
            trial_blocks = sorted(blocks + [extra_block])
            ok = True
            expected = {
                src_a: anchor_a,
                src_b: anchor_b,
            }
            for src in planned:
                try:
                    best = _best_candidate(trial, src, trial_blocks)
                except AssertionError:
                    ok = False
                    break
                if tuple(best["anchor"]) != expected[src]:
                    ok = False
                    break
            if ok:
                blocks = trial_blocks
                grid = trial
                break

    marker_total = random.choice([3, 4, 4, 5])
    colors = random.sample(MARKER_COLORS, marker_total)
    grid[src_a] = colors[0]
    grid[src_b] = colors[1]

    chosen_sources = {src_a, src_b}
    for color in colors[2:]:
        options = _enumerate_extra_markers(grid, blocks)
        pick = None
        for r, c, _block_index in options:
            if (r, c) in chosen_sources:
                continue
            pick = (r, c)
            break
        if pick is None:
            return None
        chosen_sources.add(pick)
        grid[pick] = color

    k = random.randint(0, 3)
    rotated = np.rot90(grid, k).copy()

    try:
        out, events = _simulate_growth(rotated)
    except AssertionError:
        return None

    if not any(event["stop"] == "line" for event in events):
        return None
    if any(event["stop"] in {"block", "occupied"} for event in events):
        return None
    if np.array_equal(out, rotated):
        return None
    return rotated


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_canonical_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for task 6")
