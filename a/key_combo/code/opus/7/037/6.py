import random
from itertools import combinations

import numpy as np


BLACK = 0
GREY = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
STEP_VECTORS = tuple(
    [(0, s) for s in range(1, 4)]
    + [(s, 0) for s in range(1, 4)]
    + [(s, s) for s in range(1, 4)]
    + [(s, -s) for s in range(1, 4)]
)
GEN_RETRY_LIMIT = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _sort_coords(coords) -> list[tuple[int, int]]:
    return sorted((int(r), int(c)) for r, c in coords)


def _allowed_step(dr: int, dc: int) -> bool:
    if dr == 0 and 1 <= dc <= 3:
        return True
    if dc == 0 and 1 <= dr <= 3:
        return True
    if 1 <= dr <= 3 and abs(dc) == dr:
        return True
    return False


def _triplet_step(coords) -> tuple[int, int] | None:
    pts = _sort_coords(coords)
    (r0, c0), (r1, c1), (r2, c2) = pts
    d1 = (r1 - r0, c1 - c0)
    d2 = (r2 - r1, c2 - c1)
    if d1 != d2:
        return None
    dr, dc = d1
    if dr == 0 and dc == 0:
        return None
    if not _allowed_step(dr, dc):
        return None
    return dr, dc


def _partition_marker_triplets(coords) -> list[list[tuple[int, int]]]:
    pts = _sort_coords(coords)
    index_of = {pt: i for i, pt in enumerate(pts)}
    valid_triplets = []
    for combo in combinations(pts, 3):
        if _triplet_step(combo) is None:
            continue
        valid_triplets.append(tuple(sorted(index_of[pt] for pt in combo)))

    solutions = []

    def backtrack(remaining: set[int], acc: list[tuple[int, int, int]]) -> None:
        if len(solutions) > 1:
            return
        if not remaining:
            solutions.append(acc.copy())
            return
        pivot = min(remaining)
        for triplet in valid_triplets:
            if pivot not in triplet:
                continue
            if all(idx in remaining for idx in triplet):
                next_remaining = remaining.difference(triplet)
                acc.append(triplet)
                backtrack(next_remaining, acc)
                acc.pop()

    backtrack(set(range(len(pts))), [])
    assert len(solutions) == 1, "grey markers must split into a unique set of arithmetic triplets"

    groups = []
    for triplet in solutions[0]:
        groups.append([pts[idx] for idx in triplet])
    return groups


def _ray_result(grid: np.ndarray, group) -> dict:
    pts = _sort_coords(group)
    step = _triplet_step(pts)
    assert step is not None
    dr, dc = step
    path = []
    r, c = pts[-1][0] + dr, pts[-1][1] + dc
    while _in_bounds(grid.shape, r, c):
        cell = int(grid[r, c])
        if cell == BLACK:
            path.append((r, c))
            r += dr
            c += dc
            continue
        assert cell != GREY, "rays should not hit other grey markers"
        return {"kind": "frame", "path": path, "color": cell, "hit": (r, c), "step": step}
    return {"kind": "boundary", "path": path, "color": GREY, "hit": None, "step": step}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate each grey arithmetic-marker line until it hits a colored frame border or the grid boundary.

    Input structure
    - The grid uses black as background, grey pixels as marker triples, and several other colors as hollow rectangular frames.
    - Each marker group contains exactly three grey pixels on one horizontal, vertical, or 45-degree arithmetic line.
    - The colored objects are rectangular perimeters whose borders act as collision targets for the extrapolated rays.

    Rule
    - Partition all grey pixels into row-major arithmetic triplets and read each triplet's repeated step vector.
    - Starting one step beyond the third marker, keep advancing by that vector across black cells and draw those visited cells grey.
    - If the next stepped position leaves the grid, stop and keep the whole extrapolated segment grey.
    - If the next stepped position is a colored frame-border pixel, stop before drawing that border cell and recolor the already drawn extrapolated segment to that frame's color.

    Key insight
    - The marker triples only define direction and spacing; the eventual color of the new segment is decided by what the ray collides with, not by the markers themselves.
    """
    g = np.asarray(grid, dtype=int)
    grey_coords = np.argwhere(g == GREY)
    assert len(grey_coords) % 3 == 0 and len(grey_coords) > 0

    groups = _partition_marker_triplets(grey_coords)
    out = g.copy()
    for group in groups:
        result = _ray_result(g, group)
        for r, c in result["path"]:
            out[r, c] = result["color"]
    return out


def _bbox_overlaps(a, b, gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    color = frame["color"]
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _place_frames(height: int, width: int, count: int) -> list[dict] | None:
    colors = random.sample(FRAME_COLORS, count)
    frames = []
    for color in colors:
        placed = False
        for _ in range(120):
            h = random.randint(4, 6)
            w = random.randint(4, 6)
            if h >= height - 1 or w >= width - 1:
                continue
            r0 = random.randint(1, height - h - 1)
            c0 = random.randint(1, width - w - 1)
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_bbox_overlaps(bbox, other["bbox"], gap=1) for other in frames):
                continue
            frames.append({"bbox": bbox, "color": color})
            placed = True
            break
        if not placed:
            return None
    return frames


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if _in_bounds(shape, nr, nc):
                    out.add((nr, nc))
    return out


def _enumerate_candidates(base: np.ndarray) -> list[dict]:
    height, width = base.shape
    candidates = []
    for dr, dc in STEP_VECTORS:
        for r0 in range(height):
            for c0 in range(width):
                markers = [(r0 + k * dr, c0 + k * dc) for k in range(3)]
                if not all(_in_bounds(base.shape, r, c) for r, c in markers):
                    continue
                if any(int(base[r, c]) != BLACK for r, c in markers):
                    continue
                result = _ray_result(base, markers)
                if not (1 <= len(result["path"]) <= 7):
                    continue
                cells = set(markers) | set(result["path"])
                candidates.append(
                    {
                        "markers": tuple(markers),
                        "path": tuple(result["path"]),
                        "kind": result["kind"],
                        "color": result["color"],
                        "step": (dr, dc),
                        "footprint": cells,
                        "halo": _halo(cells, base.shape),
                    }
                )
    random.shuffle(candidates)
    return candidates


def _choose_candidates(base: np.ndarray, group_count: int) -> list[dict] | None:
    candidates = _enumerate_candidates(base)
    needed = {"frame", "boundary"}
    chosen = []
    used_halo: set[tuple[int, int]] = set()
    used_markers: list[tuple[int, int]] = []

    def backtrack(start: int, kinds: set[str]) -> bool:
        if len(chosen) == group_count:
            if not needed.issubset(kinds):
                return False
            try:
                _partition_marker_triplets(np.array(used_markers, dtype=int))
            except AssertionError:
                return False
            return True

        remaining_slots = group_count - len(chosen)
        missing = len(needed - kinds)
        if missing > remaining_slots:
            return False

        for i in range(start, len(candidates)):
            cand = candidates[i]
            if not cand["footprint"].isdisjoint(used_halo):
                continue
            chosen.append(cand)
            old_halo = used_halo.copy()
            used_halo.update(cand["halo"])
            used_markers.extend(cand["markers"])
            if backtrack(i + 1, kinds | {cand["kind"]}):
                return True
            del used_markers[-3:]
            used_halo.clear()
            used_halo.update(old_halo)
            chosen.pop()
        return False

    if backtrack(0, set()):
        return chosen.copy()
    return None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(17, 22)
    width = random.randint(17, 22)
    frame_count = random.randint(3, 4)
    group_count = random.randint(2, 3)

    frames = _place_frames(height, width, frame_count)
    if frames is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for frame in frames:
        _draw_frame(grid, frame)

    groups = _choose_candidates(grid, group_count)
    if groups is None:
        return None

    for group in groups:
        for r, c in group["markers"]:
            grid[r, c] = GREY

    out = main(grid)
    grey_values = [group["kind"] for group in groups]
    if "frame" not in grey_values or "boundary" not in grey_values:
        return None
    if not np.any((out != grid) & (out != GREY)):
        return None
    if not np.any(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arithmetic-line frame-collision input")
