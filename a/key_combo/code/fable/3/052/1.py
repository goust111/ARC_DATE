from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_of_coords(coords: list[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _expand_bbox(bbox: BBox, step: int) -> BBox:
    r0, c0, r1, c1 = bbox
    return r0 - step, c0 - step, r1 + step, c1 + step


def _ring_cells(bbox: BBox) -> set[Coord]:
    r0, c0, r1, c1 = bbox
    cells: set[Coord] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _draw_ring(grid: np.ndarray, bbox: BBox, color: int) -> None:
    for r, c in _ring_cells(bbox):
        grid[r, c] = int(color)


def _bbox_contains(outer: BBox, inner: BBox) -> bool:
    return outer[0] <= inner[0] <= inner[2] <= outer[2] and outer[1] <= inner[1] <= inner[3] <= outer[3]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            q: deque[Coord] = deque([(r, c)])
            visited[r, c] = True
            coords: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_of_coords(coords),
                    "area": len(coords),
                }
            )
    return components


def _is_hollow_rectangle(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    border = _ring_cells(component["bbox"])
    return len(component["coords"]) == len(border) and set(component["coords"]) == border


def _extract_frames(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    frame_mask = np.zeros(grid.shape, dtype=bool)
    frames: list[dict] = []
    for component in _find_monochrome_components(grid, bg):
        if not _is_hollow_rectangle(component):
            continue
        frame = {
            "color": int(component["color"]),
            "bbox": tuple(map(int, component["bbox"])),
            "coords": [tuple(map(int, p)) for p in component["coords"]],
        }
        frames.append(frame)
        for r, c in frame["coords"]:
            frame_mask[r, c] = True
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["bbox"][2], item["bbox"][3], item["color"]))
    return frames, frame_mask


def _seed_coords_in_frame(cleared: np.ndarray, bbox: BBox, bg: int) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    seeds: list[Coord] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(cleared[r, c]) != int(bg):
                seeds.append((r, c))
    return seeds


def _ring_is_clear(bbox: BBox, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    r0, c0, r1, c1 = bbox
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    return all(not occupied[r, c] for r, c in _ring_cells(bbox))


def _max_expand_steps(bbox: BBox, occupied: np.ndarray) -> int:
    step = 0
    while _ring_is_clear(_expand_bbox(bbox, step + 1), occupied):
        step += 1
    return step


def _solve_scene(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    frames, frame_mask = _extract_frames(g, bg)
    assert frames, "expected at least one hollow rectangle"

    cleared = g.copy()
    cleared[frame_mask] = bg

    seeded: list[dict] = []
    empty: list[dict] = []
    all_frames: list[dict] = []
    for frame in frames:
        seeds = _seed_coords_in_frame(cleared, frame["bbox"], bg)
        enriched = dict(frame)
        enriched["seeds"] = seeds
        if seeds:
            seeded.append(enriched)
        else:
            empty.append(enriched)
        all_frames.append(enriched)

    stage1 = cleared.copy()
    for frame in seeded:
        seed_bbox = _bbox_of_coords(frame["seeds"])
        shrink_bbox = (seed_bbox[0] - 1, seed_bbox[1] - 1, seed_bbox[2] + 1, seed_bbox[3] + 1)
        assert _bbox_contains(frame["bbox"], shrink_bbox), "seed bbox must stay strictly inside its frame"
        frame["shrink_bbox"] = shrink_bbox
        _draw_ring(stage1, shrink_bbox, frame["color"])

    occupied = stage1 != bg
    final = stage1.copy()
    expanded_cells: set[Coord] = set()
    for frame in empty:
        step = _max_expand_steps(frame["bbox"], occupied)
        frame["expand_steps"] = int(step)
        frame["final_bbox"] = _expand_bbox(frame["bbox"], step)
        cells = _ring_cells(frame["final_bbox"])
        assert not (expanded_cells & cells), "expanded empty frames must not overlap each other"
        expanded_cells |= cells
        _draw_ring(final, frame["final_bbox"], frame["color"])

    return final, {
        "bg": bg,
        "frames": all_frames,
        "seeded": seeded,
        "empty": empty,
        "cleared": cleared,
        "stage1": stage1,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink seeded hollow rectangles to the seed bbox and then expand empty rectangles against that updated scene.

    Input structure:
    - The grid has one dominant background color plus several one-cell-thick hollow rectangular frames of different colors.
    - Some frames contain 1-3 interior seed pixels; other frames have empty interiors.
    - Seed colors are irrelevant cues: the deciding property is whether a frame encloses any non-background pixel.

    Rule:
    - First clear every frame border; for each seeded frame, redraw a same-color rectangular ring equal to the seeds' bounding box expanded by one cell, keeping the seed pixels unchanged.
    - Then treat the stage-one result as the collision reference and independently expand each originally empty frame outward around its original center.
    - Each empty frame grows to the largest same-center ring that stays inside the grid and whose border does not overlap any non-background pixel in the stage-one scene.

    Key insight:
    - The two object classes form a chain reaction: empty frames must test collisions after seeded frames have already shrunk, so they can grow into space that the shrink step frees.
    """
    return _solve_scene(np.asarray(grid, dtype=int))[0]


def _seed_bbox_margin_ok(frame_bbox: BBox, seed_coords: list[Coord]) -> bool:
    seed_bbox = _bbox_of_coords(seed_coords)
    return _bbox_contains(frame_bbox, (seed_bbox[0] - 1, seed_bbox[1] - 1, seed_bbox[2] + 1, seed_bbox[3] + 1))


def _sample_seed_points(frame_bbox: BBox, count: int, forced_free_side: str | None = None, min_free: int = 1) -> list[Coord] | None:
    r0, c0, r1, c1 = frame_bbox
    row_lo = r0 + 2
    row_hi = r1 - 2
    col_lo = c0 + 2
    col_hi = c1 - 2
    if forced_free_side == "up":
        row_lo = max(row_lo, r0 + min_free + 1)
    elif forced_free_side == "down":
        row_hi = min(row_hi, r1 - min_free - 1)
    elif forced_free_side == "left":
        col_lo = max(col_lo, c0 + min_free + 1)
    elif forced_free_side == "right":
        col_hi = min(col_hi, c1 - min_free - 1)
    if row_lo > row_hi or col_lo > col_hi:
        return None

    candidates = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    if len(candidates) < count:
        return None

    for _ in range(60):
        pts = random.sample(candidates, count)
        if _seed_bbox_margin_ok(frame_bbox, pts):
            return pts
    return None


def _bbox_with_pad(bbox: BBox, pad: int, shape: tuple[int, int]) -> BBox:
    h, w = shape
    return (
        max(0, bbox[0] - pad),
        max(0, bbox[1] - pad),
        min(h - 1, bbox[2] + pad),
        min(w - 1, bbox[3] + pad),
    )


def _area_clear(mask: np.ndarray, bbox: BBox) -> bool:
    r0, c0, r1, c1 = bbox
    return not bool(mask[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_area(mask: np.ndarray, bbox: BBox) -> None:
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _place_chain_pair(shape: tuple[int, int], reserved: np.ndarray, frame_colors: list[int], seed_palette: list[int]) -> list[dict] | None:
    h, w = shape
    direction = random.choice(["left", "right", "up", "down"])
    seeded_h = random.randint(7, 10)
    seeded_w = random.randint(7, 10)
    empty_h = random.randint(4, 7)
    empty_w = random.randint(4, 7)

    if direction in ("left", "right"):
        cluster_h = max(seeded_h, empty_h)
        cluster_w = seeded_w + empty_w + 1
    else:
        cluster_h = seeded_h + empty_h + 1
        cluster_w = max(seeded_w, empty_w)
    if cluster_h + 6 > h or cluster_w + 6 > w:
        return None

    for _ in range(80):
        top = random.randint(3, h - cluster_h - 3)
        left = random.randint(3, w - cluster_w - 3)
        if direction == "right":
            s_r0 = top + (cluster_h - seeded_h) // 2
            s_c0 = left
            e_r0 = top + (cluster_h - empty_h) // 2
            e_c0 = s_c0 + seeded_w + 1
            free_side = "right"
        elif direction == "left":
            e_r0 = top + (cluster_h - empty_h) // 2
            e_c0 = left
            s_r0 = top + (cluster_h - seeded_h) // 2
            s_c0 = e_c0 + empty_w + 1
            free_side = "left"
        elif direction == "down":
            s_r0 = top
            s_c0 = left + (cluster_w - seeded_w) // 2
            e_r0 = s_r0 + seeded_h + 1
            e_c0 = left + (cluster_w - empty_w) // 2
            free_side = "down"
        else:
            e_r0 = top
            e_c0 = left + (cluster_w - empty_w) // 2
            s_r0 = e_r0 + empty_h + 1
            s_c0 = left + (cluster_w - seeded_w) // 2
            free_side = "up"

        seeded_bbox = (s_r0, s_c0, s_r0 + seeded_h - 1, s_c0 + seeded_w - 1)
        empty_bbox = (e_r0, e_c0, e_r0 + empty_h - 1, e_c0 + empty_w - 1)
        if not _area_clear(reserved, _bbox_with_pad(seeded_bbox, 1, shape)):
            continue
        if not _area_clear(reserved, _bbox_with_pad(empty_bbox, 1, shape)):
            continue

        seed_count = random.randint(1, 3)
        min_free = random.randint(2, 3)
        seeds = _sample_seed_points(seeded_bbox, seed_count, forced_free_side=free_side, min_free=min_free)
        if seeds is None:
            continue

        seeded_color, empty_color = frame_colors[:2]
        seed_colors = [random.choice(seed_palette) for _ in seeds]
        pair = [
            {"color": seeded_color, "bbox": seeded_bbox, "seeds": list(zip(seeds, seed_colors))},
            {"color": empty_color, "bbox": empty_bbox, "seeds": []},
        ]
        for frame in pair:
            _mark_area(reserved, _bbox_with_pad(frame["bbox"], 2, shape))
        return pair
    return None


def _place_extra_frame(shape: tuple[int, int], reserved: np.ndarray, color: int, seed_palette: list[int], seeded: bool) -> dict | None:
    if seeded:
        frame_h = random.randint(6, 9)
        frame_w = random.randint(6, 9)
    else:
        frame_h = random.randint(4, 7)
        frame_w = random.randint(4, 7)

    h, w = shape
    if frame_h + 4 > h or frame_w + 4 > w:
        return None

    for _ in range(100):
        r0 = random.randint(2, h - frame_h - 2)
        c0 = random.randint(2, w - frame_w - 2)
        bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
        if not _area_clear(reserved, _bbox_with_pad(bbox, 1, shape)):
            continue
        seeds: list[tuple[Coord, int]] = []
        if seeded:
            coords = _sample_seed_points(bbox, random.randint(1, 3))
            if coords is None:
                continue
            seeds = [(coord, random.choice(seed_palette)) for coord in coords]
        _mark_area(reserved, _bbox_with_pad(bbox, 2, shape))
        return {"color": int(color), "bbox": bbox, "seeds": seeds}
    return None


def _render_input(shape: tuple[int, int], bg: int, frames: list[dict]) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for frame in frames:
        _draw_ring(grid, frame["bbox"], frame["color"])
    for frame in frames:
        for (r, c), color in frame["seeds"]:
            grid[r, c] = int(color)
    return grid


def _has_chain_effect(meta: dict) -> bool:
    naive = meta["cleared"].copy()
    for frame in meta["seeded"]:
        _draw_ring(naive, frame["bbox"], frame["color"])
    naive_occ = naive != meta["bg"]
    for frame in meta["empty"]:
        naive_steps = _max_expand_steps(frame["bbox"], naive_occ)
        if int(frame["expand_steps"]) > int(naive_steps):
            return True
    return False


def _is_valid_generated(grid: np.ndarray, expected_frames: int) -> bool:
    try:
        out, meta = _solve_scene(grid)
    except AssertionError:
        return False
    return (
        len(meta["frames"]) == expected_frames
        and len(meta["seeded"]) >= 1
        and len(meta["empty"]) >= 1
        and any(int(frame["expand_steps"]) > 0 for frame in meta["empty"])
        and _has_chain_effect(meta)
        and not np.array_equal(out, grid)
    )


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(20, 28), random.randint(20, 28))
    bg = random.choice(PALETTE)

    total_frames = random.randint(2, 5)
    seeded_count = random.randint(1, total_frames - 1)
    empty_count = total_frames - seeded_count

    frame_colors = random.sample([c for c in PALETTE if c != bg], total_frames)
    remaining = [c for c in PALETTE if c not in frame_colors and c != bg]
    seed_palette = remaining if remaining else [frame_colors[0]]

    reserved = np.zeros(shape, dtype=bool)
    frames = _place_chain_pair(shape, reserved, frame_colors, seed_palette)
    if frames is None:
        return None

    next_color = 2
    remaining_seeded = seeded_count - 1
    remaining_empty = empty_count - 1
    order = [True] * remaining_seeded + [False] * remaining_empty
    random.shuffle(order)
    for seeded in order:
        frame = _place_extra_frame(shape, reserved, frame_colors[next_color], seed_palette, seeded)
        if frame is None:
            return None
        frames.append(frame)
        next_color += 1

    random.shuffle(frames)
    grid = _render_input(shape, bg, frames)
    if not _is_valid_generated(grid, total_frames):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
