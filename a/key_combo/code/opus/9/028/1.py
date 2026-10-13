from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
FRAME = 5
WALL = 9

GEN_ATTEMPTS = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    label = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] >= 0:
                continue
            q = deque([(r, c)])
            labels[r, c] = label
            while q:
                rr, cc = q.popleft()
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] < 0:
                        labels[nr, nc] = label
                        q.append((nr, nc))
            label += 1
    return labels, label


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    labels, count = _label_mask(grid == color)
    return [np.argwhere(labels == idx) for idx in range(count)]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _frame_exterior_cells(grid: np.ndarray, frame: dict) -> list[tuple[int, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = frame["bbox"]
    cells = set()
    if r0 > 0:
        for c in range(c0, c1 + 1):
            if grid[r0 - 1, c] == BG:
                cells.add((r0 - 1, c))
    if r1 + 1 < h:
        for c in range(c0, c1 + 1):
            if grid[r1 + 1, c] == BG:
                cells.add((r1 + 1, c))
    if c0 > 0:
        for r in range(r0, r1 + 1):
            if grid[r, c0 - 1] == BG:
                cells.add((r, c0 - 1))
    if c1 + 1 < w:
        for r in range(r0, r1 + 1):
            if grid[r, c1 + 1] == BG:
                cells.add((r, c1 + 1))
    return sorted(cells)


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for coords in _components_of_color(grid, FRAME):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3, "frame must be at least 3x3"
        border = _border_coords(r0, c0, r1, c1)
        border_set = set(border)
        coord_set = {tuple(x) for x in coords.tolist()}
        assert coord_set == border_set, "grey component must be a 1-cell-thick hollow rectangle"
        assert np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BG), "frame interior must be background"
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "coords": np.array(border, dtype=int),
                "inner_area": (h - 2) * (w - 2),
            }
        )
    return sorted(frames, key=lambda item: item["bbox"])


def _analyze_layout(grid: np.ndarray) -> tuple[list[dict], dict[int, list[dict]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all(np.isin(g, [BG, FRAME, WALL])), "expected only background, frame, and wall colors"

    frames = _extract_frames(g)
    assert frames, "expected at least one hollow frame"

    labels, _ = _label_mask(g == BG)
    groups: dict[int, list[dict]] = {}
    for frame in frames:
        exterior = _frame_exterior_cells(g, frame)
        assert exterior, "every frame must have visible exterior background pixels"
        region_ids = {int(labels[r, c]) for r, c in exterior if labels[r, c] >= 0}
        assert len(region_ids) == 1, "each frame must belong to exactly one exterior background region"
        region_id = next(iter(region_ids))
        frame["region_id"] = region_id
        groups.setdefault(region_id, []).append(frame)
    return frames, groups


def _paint_frame(out: np.ndarray, frame: dict, color: int) -> None:
    coords = frame["coords"]
    out[coords[:, 0], coords[:, 1]] = color


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict], dict[int, list[dict]]]:
    frames, groups = _analyze_layout(grid)
    out = np.asarray(grid, dtype=int).copy()
    for region_frames in groups.values():
        if len(region_frames) == 1:
            _paint_frame(out, region_frames[0], YELLOW)
            continue
        winner_area = max(frame["inner_area"] for frame in region_frames)
        winners = [frame for frame in region_frames if frame["inner_area"] == winner_area]
        assert len(winners) == 1, "largest interior area must be unique within each multi-frame region"
        winner = winners[0]
        for frame in region_frames:
            _paint_frame(out, frame, BLUE if frame is winner else RED)
    return out, frames, groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor hollow rectangular frames by competing only within the wall-separated exterior background region they occupy.

    Input structure:
    - The background is black (0), with gray (5) 1-cell-thick hollow rectangular frames and maroon (9) wall pixels.
    - Every frame encloses only background pixels, and wall segments do not overlap any frame.
    - The gray frames are grouped by 4-connected reachability through exterior background cells, with frames and walls acting as blockers.

    Rule:
    - Flood-fill only through background (0) cells to identify exterior connected regions created by the walls.
    - Assign each frame to the unique exterior background region touching its outside border pixels.
    - In a region with one frame, recolor that frame yellow (4); in a region with multiple frames, recolor the unique largest interior-area frame blue (1) and every other frame red (2).
    - Leave all wall and background pixels unchanged.

    Key insight:
    - The walls matter only because they change which frames compete with each other; the area comparison is local to each exterior reachability partition, not global.
    """
    out, _, _ = _solve(np.asarray(grid, dtype=int))
    return out


def _compose_advances(total: int) -> list[int]:
    parts = []
    remaining = total
    while remaining > 0:
        step = random.randint(1, min(4, remaining))
        parts.append(step)
        remaining -= step
    return parts


def _build_barrier(shape: tuple[int, int], orientation: str, base: int, band: int) -> list[tuple[int, int]]:
    h, w = shape
    if orientation == "vertical":
        row = 0
        col = random.randint(max(1, base - band), min(w - 2, base + band))
        cells = [(row, col)]
        advances = _compose_advances(h - 1)
        lo = max(1, base - band)
        hi = min(w - 2, base + band)
        for idx, advance in enumerate(advances):
            for _ in range(advance):
                row += 1
                cells.append((row, col))
            if idx == len(advances) - 1:
                continue
            targets = [x for x in range(lo, hi + 1) if x != col and 1 <= abs(x - col) <= 4]
            if not targets:
                continue
            target = random.choice(targets)
            step = 1 if target > col else -1
            while col != target:
                col += step
                cells.append((row, col))
        return cells

    col = 0
    row = random.randint(max(1, base - band), min(h - 2, base + band))
    cells = [(row, col)]
    advances = _compose_advances(w - 1)
    lo = max(1, base - band)
    hi = min(h - 2, base + band)
    for idx, advance in enumerate(advances):
        for _ in range(advance):
            col += 1
            cells.append((row, col))
        if idx == len(advances) - 1:
            continue
        targets = [x for x in range(lo, hi + 1) if x != row and 1 <= abs(x - row) <= 4]
        if not targets:
            continue
        target = random.choice(targets)
        step = 1 if target > row else -1
        while row != target:
            row += step
            cells.append((row, col))
    return cells


def _choose_shape_and_bases(orientation: str, region_count: int) -> tuple[tuple[int, int], list[int]]:
    if orientation == "vertical":
        h = random.randint(18, 24)
        w = random.randint(17, 22) if region_count == 2 else random.randint(21, 26)
        if region_count == 2:
            bases = [random.randint(6, w - 7)]
        else:
            c1 = random.randint(6, w - 14)
            c2 = random.randint(c1 + 7, w - 7)
            bases = [c1, c2]
    else:
        h = random.randint(17, 22) if region_count == 2 else random.randint(21, 26)
        w = random.randint(18, 24)
        if region_count == 2:
            bases = [random.randint(6, h - 7)]
        else:
            r1 = random.randint(6, h - 14)
            r2 = random.randint(r1 + 7, h - 7)
            bases = [r1, r2]
    return (h, w), bases


def _candidate_frames(grid: np.ndarray, labels: np.ndarray, region_id: int) -> list[dict]:
    h, w = grid.shape
    candidates = []
    for fh in range(3, 8):
        for fw in range(3, 8):
            for r0 in range(1, h - fh):
                r1 = r0 + fh - 1
                for c0 in range(1, w - fw):
                    c1 = c0 + fw - 1
                    halo = grid[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2]
                    if np.any(halo != BG):
                        continue
                    region_patch = labels[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2]
                    if np.any(region_patch != region_id):
                        continue
                    candidates.append(
                        {
                            "bbox": (r0, c0, r1, c1),
                            "h": fh,
                            "w": fw,
                            "inner_area": (fh - 2) * (fw - 2),
                        }
                    )
    return candidates


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _weighted_choice(candidates: list[dict]) -> dict:
    weights = [cand["inner_area"] + cand["h"] + cand["w"] for cand in candidates]
    return random.choices(candidates, weights=weights, k=1)[0]


def _place_frames_for_region(grid: np.ndarray, labels: np.ndarray, region_id: int, count: int) -> list[dict] | None:
    for _ in range(150):
        working = grid.copy()
        chosen: list[dict] = []
        used_areas: set[int] = set()
        ok = True
        for _slot in range(count):
            candidates = _candidate_frames(working, labels, region_id)
            if count > 1:
                candidates = [cand for cand in candidates if cand["inner_area"] not in used_areas]
            if not candidates:
                ok = False
                break
            frame = dict(_weighted_choice(candidates))
            chosen.append(frame)
            used_areas.add(frame["inner_area"])
            _draw_frame(working, frame)
        if ok and len(chosen) == count:
            return chosen
    return None


def _frame_color_signature(out: np.ndarray, frames: list[dict]) -> tuple[int, ...]:
    return tuple(int(out[frame["bbox"][0], frame["bbox"][1]]) for frame in frames)


def _counts_options(region_count: int) -> list[tuple[int, ...]]:
    if region_count == 2:
        return [(1, 3), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3)]
    return [(1, 1, 2), (1, 2, 2), (2, 1, 2), (2, 2, 2), (1, 2, 3), (2, 1, 3), (3, 1, 2)]


def _gen_attempt() -> np.ndarray | None:
    region_count = random.choice([2, 3])
    orientation = random.choice(["vertical", "horizontal"])
    shape, bases = _choose_shape_and_bases(orientation, region_count)
    grid = np.full(shape, BG, dtype=int)

    for idx, base in enumerate(bases):
        band = 2 if region_count == 2 else 1
        barrier = _build_barrier(shape, orientation, base, band)
        if idx > 0:
            for r, c in barrier:
                if grid[r, c] == WALL:
                    return None
        for r, c in barrier:
            grid[r, c] = WALL

    labels, found_regions = _label_mask(grid == BG)
    if found_regions != region_count:
        return None

    region_ids = list(range(found_regions))
    random.shuffle(region_ids)
    counts = list(random.choice(_counts_options(region_count)))
    random.shuffle(counts)
    count_by_region = {region_id: count for region_id, count in zip(region_ids, counts)}
    region_areas = {region_id: int(np.sum(labels == region_id)) for region_id in region_ids}
    ordered_regions = sorted(region_ids, key=lambda region_id: (-count_by_region[region_id], region_areas[region_id]))

    for region_id in ordered_regions:
        placed = _place_frames_for_region(grid, labels, region_id, count_by_region[region_id])
        if placed is None:
            return None
        for frame in placed:
            _draw_frame(grid, frame)

    try:
        out, frames, groups = _solve(grid)
    except AssertionError:
        return None

    if not (4 <= len(frames) <= 6):
        return None
    if len(groups) != region_count:
        return None
    if not all(len(group) >= 1 for group in groups.values()):
        return None
    if not any(len(group) > 1 for group in groups.values()):
        return None
    if not all(len({frame["inner_area"] for frame in group}) == len(group) for group in groups.values() if len(group) > 1):
        return None

    signature = _frame_color_signature(out, frames)
    if len(set(signature)) < 2:
        return None
    if any(len(group) == 1 for group in groups.values()) and YELLOW not in signature:
        return None
    if sum(len(group) > 1 for group in groups.values()) >= 2 and sum(color == BLUE for color in signature) < 2:
        return None

    no_walls = grid.copy()
    no_walls[no_walls == WALL] = BG
    try:
        flat_out, flat_frames, _ = _solve(no_walls)
    except AssertionError:
        return None
    if _frame_color_signature(flat_out, flat_frames) == signature:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
