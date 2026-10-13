from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

SOURCE_COLORS = (RED, GREY)
COPY_COLOR = {RED: GREEN, GREY: YELLOW}
MISSING_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))

GRID_ROWS = (13, 18)
GRID_COLS = (13, 18)
OBJECT_COUNT = (4, 6)
GEN_ATTEMPTS = 500
FEATURE_MODES = ("basic", "source_block", "color_priority")
FEATURE_WEIGHTS = (3, 2, 1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _dominant_color(grid: np.ndarray) -> int:
    return Counter(map(int, grid.flat)).most_common(1)[0][0]


def _source_cells(r0: int, c0: int, missing: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (r0 + rr, c0 + cc)
        for rr in (0, 1)
        for cc in (0, 1)
        if (rr, cc) != missing
    )


def _missing_vector(missing: tuple[int, int]) -> tuple[int, int]:
    mr, mc = missing
    return (-1 if mr == 0 else 1, -1 if mc == 0 else 1)


def _parse_missing(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    assert r1 == r0 + 1 and c1 == c0 + 1, "source must occupy a 2x2 bbox"
    rel = {(r - r0, c - c0) for r, c in coords}
    missing = tuple((_ for _ in (MISSING_CORNERS) if _ not in rel).__next__())
    return r0, c0, int(missing[0]), int(missing[1])


def _copy_spec(
    obj_id: int,
    color: int,
    r0: int,
    c0: int,
    missing: tuple[int, int],
) -> dict:
    source = _source_cells(r0, c0, missing)
    v_r, v_c = _missing_vector(missing)
    if color == RED:
        v_r, v_c = -v_r, -v_c
    copy_full = tuple((r + v_r, c + v_c) for r, c in source)
    cr0, cc0 = r0 + v_r, c0 + v_c
    elbow = (cr0 + (1 - missing[0]), cc0 + (1 - missing[1]))
    endpoints = []
    for rr in (0, 1):
        for cc in (0, 1):
            corner = (cr0 + rr, cc0 + cc)
            if corner in copy_full and corner != elbow:
                endpoints.append(corner)
    assert len(endpoints) == 2
    rays = []
    for er, ec in endpoints:
        rel_r = er - cr0
        rel_c = ec - cc0
        rays.append(((er, ec), (-1 if rel_r == 0 else 1, -1 if rel_c == 0 else 1)))
    return {
        "id": obj_id,
        "source_color": color,
        "copy_color": COPY_COLOR[color],
        "missing": missing,
        "source": source,
        "source_bbox": (r0, c0, r0 + 1, c0 + 1),
        "copy_full": copy_full,
        "copy_bbox": (cr0, cc0, cr0 + 1, cc0 + 1),
        "rays": tuple(rays),
    }


def _find_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    assert bg == BLACK, "background must be black"
    colors = set(map(int, np.unique(g)))
    assert colors.issubset({BLACK, RED, GREY}), "input must contain only black, red, and grey"
    assert RED in colors and GREY in colors, "input must contain both red and grey sources"

    objects = []
    seen = np.zeros(g.shape, dtype=bool)
    next_id = 0
    for color in SOURCE_COLORS:
        for r in range(g.shape[0]):
            for c in range(g.shape[1]):
                if seen[r, c] or int(g[r, c]) != color:
                    continue
                q = deque([(r, c)])
                seen[r, c] = True
                coords = []
                while q:
                    rr, cc = q.popleft()
                    coords.append((rr, cc))
                    for nr, nc in _neighbors4(rr, cc):
                        if _in_bounds(g.shape, nr, nc) and not seen[nr, nc] and int(g[nr, nc]) == color:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                assert len(coords) == 3, "each source object must be a triomino"
                r0, c0, mr, mc = _parse_missing(coords)
                objects.append(_copy_spec(next_id, color, r0, c0, (mr, mc)))
                next_id += 1
    assert 4 <= len(objects) <= 6, "expected 4-6 source triominoes"
    return objects


def _render(
    grid: np.ndarray,
    objects: list[dict],
    *,
    include_rays: bool,
    start_with_sources: bool,
    restore_sources: bool,
    color_order: tuple[int, int] = (YELLOW, GREEN),
    reverse_same_color: bool = False,
) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    out = g.copy() if start_with_sources else np.full_like(g, BLACK)
    owners = [[("bg", -1) for _ in range(g.shape[1])] for _ in range(g.shape[0])]
    if start_with_sources:
        for obj in objects:
            for r, c in obj["source"]:
                owners[r][c] = ("source", obj["id"])

    meta = {
        "ray_cells": 0,
        "copy_overwrites": 0,
        "ray_blocked": 0,
        "ray_hit_source": 0,
        "ray_hit_copy_or_ray": 0,
    }

    for copy_color in color_order:
        group = [obj for obj in objects if obj["copy_color"] == copy_color]
        group.sort(key=lambda obj: (obj["copy_bbox"][0], obj["copy_bbox"][1]), reverse=reverse_same_color)
        for obj in group:
            for r, c in obj["copy_full"]:
                if not _in_bounds(out.shape, r, c):
                    continue
                if int(out[r, c]) != BLACK:
                    meta["copy_overwrites"] += 1
                out[r, c] = obj["copy_color"]
                owners[r][c] = ("copy", obj["id"])

            if not include_rays:
                continue
            for (sr, sc), (dr, dc) in obj["rays"]:
                r, c = sr + dr, sc + dc
                while _in_bounds(out.shape, r, c):
                    if int(out[r, c]) != BLACK:
                        meta["ray_blocked"] += 1
                        if owners[r][c][0] == "source":
                            meta["ray_hit_source"] += 1
                        else:
                            meta["ray_hit_copy_or_ray"] += 1
                        break
                    out[r, c] = obj["copy_color"]
                    owners[r][c] = ("ray", obj["id"])
                    meta["ray_cells"] += 1
                    r += dr
                    c += dc

    if restore_sources:
        for obj in objects:
            for r, c in obj["source"]:
                out[r, c] = obj["source_color"]
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each red or grey L triomino to a recolored diagonal copy, then emit diagonal rays from the copy's two exposed corners.

    Input structure:
    - The background is black.
    - The foreground consists of 4 to 6 non-touching L triominoes, each made of three cells inside a 2x2 box.
    - Each source triomino is either red or grey, and each orientation is identified by its missing 2x2 corner.

    Rule:
    - For each triomino, derive the diagonal vector pointing from the 2x2 center toward the missing corner.
    - Grey sources translate one step along that vector and become yellow copies; red sources translate one step opposite that vector and become green copies.
    - Draw all yellow copies and their rays first, then all green copies and their rays; within one copy color, use the translated copy bbox top-left row/col order.
    - Each translated L copy emits one diagonal ray from each endpoint corner, pointing outward from the copy's 2x2 box, and each ray stops before the first non-background cell or grid boundary.
    - After all copies and rays are drawn, redraw every original source cell so the original triominoes keep their input colors.

    Key insight:
    - The missing corner fixes both the copy displacement and the translated L orientation, and that orientation determines which two copy corners are endpoints and which outward diagonals their rays follow.
    """
    objects = _find_objects(np.asarray(grid, dtype=int))
    out, _meta = _render(
        np.asarray(grid, dtype=int),
        objects,
        include_rays=True,
        start_with_sources=True,
        restore_sources=True,
    )
    return out


def _forbidden_with_gap(cells: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    blocked = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                blocked.add((r + dr, c + dc))
    return blocked


def _copy_visibility(obj: dict, shape: tuple[int, int]) -> tuple[int, int]:
    visible_copy = sum(_in_bounds(shape, r, c) for r, c in obj["copy_full"])
    visible_ray_starts = sum(_in_bounds(shape, sr + dr, sc + dc) for (sr, sc), (dr, dc) in obj["rays"])
    return visible_copy, visible_ray_starts


def _sample_specs(count: int) -> list[tuple[int, tuple[int, int]]]:
    n_red = random.randint(1, count - 1)
    colors = [RED] * n_red + [GREY] * (count - n_red)
    random.shuffle(colors)

    orientations = list(random.sample(MISSING_CORNERS, len(MISSING_CORNERS)))
    while len(orientations) < count:
        orientations.append(random.choice(MISSING_CORNERS))
    random.shuffle(orientations)
    return list(zip(colors, orientations[:count]))


def _place_scene(shape: tuple[int, int], specs: list[tuple[int, tuple[int, int]]]) -> list[dict] | None:
    H, W = shape
    objects = []
    forbidden: set[tuple[int, int]] = set()

    for obj_id, (color, missing) in enumerate(specs):
        candidates = []
        positions = [(r0, c0) for r0 in range(H - 1) for c0 in range(W - 1)]
        random.shuffle(positions)
        for r0, c0 in positions:
            source = _source_cells(r0, c0, missing)
            if any(cell in forbidden for cell in source):
                continue
            obj = _copy_spec(obj_id, color, r0, c0, missing)
            visible_copy, visible_ray_starts = _copy_visibility(obj, shape)
            if visible_copy < 2 or visible_ray_starts == 0:
                continue
            margin_penalty = sum(
                int(r in (0, H - 1) or c in (0, W - 1))
                for r, c in obj["copy_full"]
                if _in_bounds(shape, r, c)
            )
            score = (visible_copy, visible_ray_starts, -margin_penalty, random.random())
            candidates.append((score, obj))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        chosen = random.choice(candidates[: min(10, len(candidates))])[1]
        objects.append(chosen)
        forbidden.update(_forbidden_with_gap(chosen["source"]))
    return objects


def _semantic_ok(grid: np.ndarray, objects: list[dict], mode: str) -> bool:
    correct, meta = _render(grid, objects, include_rays=True, start_with_sources=True, restore_sources=True)
    copies_only, _ = _render(grid, objects, include_rays=False, start_with_sources=True, restore_sources=True)
    no_source_obstacles, _ = _render(grid, objects, include_rays=True, start_with_sources=False, restore_sources=True)
    reversed_colors, _ = _render(
        grid,
        objects,
        include_rays=True,
        start_with_sources=True,
        restore_sources=True,
        color_order=(GREEN, YELLOW),
    )

    if not np.array_equal(main(grid), correct):
        return False
    if np.array_equal(correct, grid) or np.array_equal(correct, copies_only):
        return False
    if meta["ray_cells"] < len(objects):
        return False
    if len({obj["missing"] for obj in objects}) < 2:
        return False
    if sum(obj["source_color"] == RED for obj in objects) == 0 or sum(obj["source_color"] == GREY for obj in objects) == 0:
        return False
    if np.count_nonzero(correct != grid) < len(objects) + 4:
        return False

    if mode == "basic":
        return True
    if mode == "source_block":
        return meta["ray_hit_source"] > 0 and not np.array_equal(correct, no_source_obstacles)
    if mode == "color_priority":
        return not np.array_equal(correct, reversed_colors)
    return False


def _gen_attempt(mode: str) -> np.ndarray | None:
    H = random.randint(*GRID_ROWS)
    W = random.randint(*GRID_COLS)
    count = random.randint(*OBJECT_COUNT)
    specs = _sample_specs(count)
    objects = _place_scene((H, W), specs)
    if objects is None:
        return None

    grid = np.full((H, W), BLACK, dtype=int)
    for obj in objects:
        for r, c in obj["source"]:
            grid[r, c] = obj["source_color"]

    if _semantic_ok(grid, objects, mode):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        mode = random.choices(FEATURE_MODES, weights=FEATURE_WEIGHTS, k=1)[0]
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt("basic")
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
