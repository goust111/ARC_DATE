from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
PALETTE = (1, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    mask = (g != BLACK) & (g != RED)
    objects: list[dict] = []
    for coords in _component_coords(mask):
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each translated object must be monochrome"
        color = next(iter(colors))
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        objects.append(
            {
                "coords": coords,
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "width": int(c1 - c0 + 1),
            }
        )
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"]))
    return objects


def _translated_state(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    objects = _find_objects(g)
    barrier = np.where(g == RED, RED, BLACK).astype(int)
    owner = np.full((h, w), -1, dtype=int)
    occupied = barrier == RED
    translated: list[dict] = []

    for idx, obj in enumerate(objects):
        shift = int(obj["width"])
        dst = obj["coords"] + np.array([0, shift], dtype=int)
        assert bool((0 <= dst[:, 0]).all() and (dst[:, 0] < h).all())
        assert bool((0 <= dst[:, 1]).all() and (dst[:, 1] < w).all())
        assert not bool(occupied[dst[:, 0], dst[:, 1]].any()), "translated objects must land on empty background"
        barrier[dst[:, 0], dst[:, 1]] = int(obj["color"])
        owner[dst[:, 0], dst[:, 1]] = idx
        occupied[dst[:, 0], dst[:, 1]] = True
        translated.append(
            {
                "color": int(obj["color"]),
                "width": shift,
                "coords": dst,
                "bbox": (
                    int(dst[:, 0].min()),
                    int(dst[:, 1].min()),
                    int(dst[:, 0].max()),
                    int(dst[:, 1].max()),
                ),
            }
        )

    return barrier, owner, translated


def _exterior_background(barrier: np.ndarray) -> np.ndarray:
    h, w = barrier.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def add(r: int, c: int) -> None:
        if int(barrier[r, c]) == BLACK and not exterior[r, c]:
            exterior[r, c] = True
            q.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(0, c)
        add(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                add(nr, nc)

    return exterior


def _enclosed_regions(barrier: np.ndarray) -> list[np.ndarray]:
    enclosed = (barrier == BLACK) & ~_exterior_background(barrier)
    return _component_coords(enclosed)


def _region_votes(region: np.ndarray, owner: np.ndarray) -> dict[int, set[tuple[int, int]]]:
    h, w = owner.shape
    votes: dict[int, set[tuple[int, int]]] = {}
    for r, c in region:
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w:
                obj_id = int(owner[nr, nc])
                if obj_id >= 0:
                    votes.setdefault(obj_id, set()).add((nr, nc))
    return votes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift each colored object right by its own width, then fill each newly enclosed chamber with that closing object's color.

    Input structure:
    - The grid has a black background, red one-cell-thick wall segments, and 3-5 small monochrome objects in distinct colors.
    - Each non-red object is 4-connected and sits to the left of a wall gap by exactly its own bounding-box width.
    - After translation, the shifted objects and the red walls together define which black chambers are closed.

    Rule:
    - Find every non-black, non-red 4-connected object and compute its bounding-box width w.
    - Move each object rigidly w columns to the right, clearing its original cells to black.
    - Treat the red walls plus all translated object pixels as barriers, flood-fill border-reachable black cells, and identify the remaining enclosed black regions.
    - For each enclosed region, count which translated object contributes the most orthogonally adjacent boundary pixels and recolor the whole region to that object's color.

    Key insight:
    - The same width value plays two roles at once: it is both the translation distance and the exact key size that lets the shifted object seal a wall gap and thereby determine the chamber's fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    barrier, owner, translated = _translated_state(g)
    out = barrier.copy()

    for region in _enclosed_regions(barrier):
        votes = _region_votes(region, owner)
        assert votes, "every enclosed region must touch at least one translated object"
        counts = {obj_id: len(pixels) for obj_id, pixels in votes.items()}
        best = max(counts.values())
        winners = [obj_id for obj_id, count in counts.items() if count == best]
        assert len(winners) == 1, "region color must be uniquely determined"
        fill_color = int(translated[winners[0]]["color"])
        out[region[:, 0], region[:, 1]] = fill_color

    return out


def _sample_sprite(width: int, gap_h: int) -> np.ndarray:
    if width == 2:
        sprite = np.zeros((gap_h, 2), dtype=int)
        sprite[:, 1] = 1
        extra_len = random.randint(1, min(6 - gap_h, gap_h))
        start = random.randint(0, gap_h - extra_len)
        sprite[start : start + extra_len, 0] = 1
        return sprite

    sprite = np.zeros((gap_h, 3), dtype=int)
    sprite[:, 2] = 1
    arm_row = random.randrange(gap_h)
    sprite[arm_row, 1] = 1
    sprite[arm_row, 0] = 1
    if gap_h == 3 and random.random() < 0.5:
        candidates = [rr for rr in (arm_row - 1, arm_row + 1) if 0 <= rr < gap_h]
        if candidates:
            sprite[random.choice(candidates), 1] = 1
    return sprite


def _draw_left_gap_box(
    grid: np.ndarray,
    frame_top: int,
    frame_left: int,
    inner_h: int,
    inner_w: int,
    gap_top: int,
    gap_h: int,
) -> tuple[int, int, int, int]:
    r0 = int(frame_top)
    c0 = int(frame_left)
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0, c0 : c1 + 1] = RED
    grid[r1, c0 : c1 + 1] = RED
    grid[r0 : r1 + 1, c0] = RED
    grid[r0 : r1 + 1, c1] = RED
    grid[gap_top : gap_top + gap_h, c0] = BLACK
    return r0, c0, r1, c1


def _place_sprite(grid: np.ndarray, sprite: np.ndarray, top: int, left: int, color: int) -> np.ndarray:
    coords = np.argwhere(sprite != 0)
    h, w = grid.shape
    placed = coords + np.array([top, left], dtype=int)
    assert bool((0 <= placed[:, 0]).all() and (placed[:, 0] < h).all())
    assert bool((0 <= placed[:, 1]).all() and (placed[:, 1] < w).all())
    assert not bool((grid[placed[:, 0], placed[:, 1]] != BLACK).any())
    grid[placed[:, 0], placed[:, 1]] = int(color)
    return placed


def _movement_path_cells(sprite: np.ndarray, top: int, start_left: int, shift: int) -> set[tuple[int, int]]:
    path: set[tuple[int, int]] = set()
    for rr, cc in np.argwhere(sprite != 0):
        row = int(top + rr)
        start = int(start_left + cc)
        for col in range(start, start + shift + 1):
            path.add((row, col))
    return path


def _build_candidate() -> np.ndarray | None:
    chamber_count = 3
    colors = random.sample(PALETTE, chamber_count)
    gap = random.randint(1, 2)
    cursor = random.randint(1, 2)
    modules: list[dict] = []

    for idx in range(chamber_count):
        width = random.choice((2, 3))
        gap_h = random.choice((3, 3, 4))
        sprite = _sample_sprite(width, gap_h)
        inner_h = random.randint(gap_h, gap_h + 1)
        inner_w = random.randint(4, 7)
        frame_left = random.randint(2 * width + 2, 2 * width + 6)
        frame_top = cursor
        gap_off = random.randint(1, inner_h - gap_h + 1)
        gap_top = frame_top + gap_off
        final_left = frame_left - (width - 1)
        input_left = final_left - width
        modules.append(
            {
                "kind": "chamber",
                "color": int(colors[idx]),
                "width": width,
                "gap_h": gap_h,
                "inner_w": inner_w,
                "inner_h": inner_h,
                "sprite": sprite,
                "frame_left": frame_left,
                "frame_top": frame_top,
                "gap_top": gap_top,
                "final_top": gap_top,
                "final_left": final_left,
                "input_top": gap_top,
                "input_left": input_left,
                "box_top": cursor,
                "box_bottom": cursor + inner_h + 1,
            }
        )
        cursor = modules[-1]["box_bottom"] + 1 + gap

    gap_h = random.choice((3, 3, 4))
    inner_h = random.randint(gap_h, gap_h + 1)
    inner_w = random.randint(4, 7)
    frame_left = random.randint(7, 12)
    frame_top = cursor
    gap_top = frame_top + random.randint(1, inner_h - gap_h + 1)
    modules.append(
        {
            "kind": "distractor",
            "gap_h": gap_h,
            "inner_w": inner_w,
            "inner_h": inner_h,
            "frame_left": frame_left,
            "frame_top": frame_top,
            "gap_top": gap_top,
            "box_top": cursor,
            "box_bottom": cursor + inner_h + 1,
        }
    )
    cursor = modules[-1]["box_bottom"] + 1 + gap
    distractor_probe = (frame_top + 1, frame_left + 1)

    h = cursor + random.randint(1, 2)
    w = max(module["frame_left"] + module["inner_w"] + 5 for module in modules)
    if h > 30 or w > 30:
        return None

    structure = np.zeros((h, w), dtype=int)
    barrier = np.zeros((h, w), dtype=int)
    chamber_modules = [module for module in modules if module["kind"] == "chamber"]

    for module in modules:
        _draw_left_gap_box(
            structure,
            module["frame_top"],
            module["frame_left"],
            module["inner_h"],
            module["inner_w"],
            module["gap_top"],
            module["gap_h"],
        )
    barrier[:] = structure

    input_path_sets: list[set[tuple[int, int]]] = []
    for module in chamber_modules:
        sprite = module["sprite"]
        if module["input_left"] < 0:
            return None
        placed = _place_sprite(barrier, sprite, module["final_top"], module["final_left"], module["color"])
        module["final_coords"] = placed
        path_cells = _movement_path_cells(sprite, module["input_top"], module["input_left"], module["width"])
        if any(structure[r, c] == RED for r, c in path_cells):
            return None
        if any(path_cells & other for other in input_path_sets):
            return None
        input_path_sets.append(path_cells)

    inp = barrier.copy()
    for module in chamber_modules:
        final_coords = module["final_coords"]
        inp[final_coords[:, 0], final_coords[:, 1]] = BLACK
    for module in chamber_modules:
        input_coords = _place_sprite(inp, module["sprite"], module["input_top"], module["input_left"], module["color"])
        if int(input_coords[:, 1].max()) >= w // 2:
            return None
        module["input_coords"] = input_coords

    objects = _find_objects(inp)
    if len(objects) != chamber_count:
        return None

    barrier_after_shift, owner, translated = _translated_state(inp)
    if len(translated) != chamber_count:
        return None

    regions = _enclosed_regions(barrier_after_shift)
    if len(regions) != chamber_count:
        return None

    seen_colors: set[int] = set()
    for region in regions:
        votes = _region_votes(region, owner)
        if len(votes) != 1:
            return None
        obj_id = next(iter(votes))
        seen_colors.add(int(translated[obj_id]["color"]))
        if len(votes[obj_id]) < 2:
            return None
    if seen_colors != {module["color"] for module in chamber_modules}:
        return None

    if distractor_probe is not None:
        exterior = _exterior_background(barrier_after_shift)
        if not exterior[distractor_probe]:
            return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
