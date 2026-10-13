from __future__ import annotations

import importlib.util
import random
from collections import deque
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_arc_utils_shared", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"could not load arc_utils from {_ARC_UTILS_PATH}")
_arc_utils = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_arc_utils)

Color = _arc_utils.Color
get_bg = _arc_utils.get_bg


MAX_GENERATION_ATTEMPTS = 800
MOVABLE_COLORS = [c for c in range(10) if c not in (Color.BLACK, Color.GREY)]
BOTTOM_TEMPLATES = (
    np.asarray([0, 2, 1, 3, 0], dtype=int),
    np.asarray([1, 0, 2, 0, 1], dtype=int),
    np.asarray([0, 3, 1, 2, 0], dtype=int),
    np.asarray([2, 0, 1, 0, 2], dtype=int),
)
TOP_TEMPLATES = (
    np.asarray([0, 2, 1, 2, 0], dtype=int),
    np.asarray([1, 0, 2, 0, 1], dtype=int),
    np.asarray([0, 1, 2, 1, 0], dtype=int),
    np.asarray([0, 2, 0, 1, 0], dtype=int),
)


def _make_object(color: int, top: list[int] | np.ndarray, bottom: list[int] | np.ndarray) -> dict:
    top_arr = np.asarray(top, dtype=int)
    bottom_arr = np.asarray(bottom, dtype=int)
    height = int(bottom_arr.max()) + 1
    width = int(bottom_arr.size)
    mask = np.zeros((height, width), dtype=bool)
    for c in range(width):
        mask[int(top_arr[c]) : int(bottom_arr[c]) + 1, c] = True
    return {
        "color": int(color),
        "top": top_arr,
        "bottom": bottom_arr,
        "mask": mask,
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
    }


def _surface_from_solid(solid: np.ndarray) -> np.ndarray:
    h, w = solid.shape
    surface = np.full(w, h, dtype=int)
    for c in range(w):
        rows = np.flatnonzero(solid[:, c])
        if rows.size:
            surface[c] = int(rows[0])
    return surface


def _fit_positions(surface: np.ndarray, obj: dict) -> list[tuple[int, int]]:
    bottom = obj["bottom"]
    width = int(obj["width"])
    fits: list[tuple[int, int]] = []
    for x in range(surface.size - width + 1):
        segment = surface[x : x + width]
        offsets = segment - bottom
        if np.all(offsets == offsets[0]):
            y = int(offsets[0]) - 1
            if y >= 0:
                fits.append((int(x), y))
    return fits


def _surface_after_place(surface: np.ndarray, obj: dict, x: int, y: int) -> np.ndarray:
    updated = surface.copy()
    width = int(obj["width"])
    updated[x : x + width] = np.minimum(updated[x : x + width], y + obj["top"])
    return updated


def _place_object(grid: np.ndarray, solid: np.ndarray, obj: dict, x: int, y: int) -> None:
    color = int(obj["color"])
    mask = obj["mask"]
    h, w = mask.shape
    view = grid[y : y + h, x : x + w]
    view[mask] = color
    solid[y : y + h, x : x + w] |= mask


def _extract_components(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == int(color)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in cells:
                local[rr - r0, cc - c0] = True
            top = []
            bottom = []
            for lc in range(local.shape[1]):
                rows_here = np.flatnonzero(local[:, lc])
                if rows_here.size == 0:
                    raise ValueError("object columns must be contiguous without internal gaps")
                top.append(int(rows_here[0]))
                bottom.append(int(rows_here[-1]))
            components.append(_make_object(color, top, bottom))
    return components


def _terrain_grid(surface: np.ndarray, height: int, bg: int = Color.BLACK) -> np.ndarray:
    grid = np.full((height, surface.size), int(bg), dtype=int)
    for c, top in enumerate(surface):
        grid[int(top) :, c] = int(Color.GREY)
    return grid


def _apply_hidden_sequence(surface: np.ndarray, placements: list[tuple[dict, int]]) -> tuple[np.ndarray, list[np.ndarray]]:
    states = [surface.copy()]
    current = surface.copy()
    for obj, x in placements:
        fit = _fit_positions(current, obj)
        if fit != [(int(x), int(fit[0][1]))] and (len(fit) != 1 or fit[0][0] != int(x)):
            raise ValueError("hidden construction produced a non-unique placement")
        px, py = fit[0]
        current = _surface_after_place(current, obj, px, py)
        states.append(current.copy())
    return current, states


def _build_expected_grid(height: int, width: int, surface0: np.ndarray, placements: list[tuple[dict, int]]) -> np.ndarray:
    grid = _terrain_grid(surface0, height)
    solid = grid != Color.BLACK
    current_surface = _surface_from_solid(solid)
    for obj, _x in placements:
        fits = _fit_positions(current_surface, obj)
        if len(fits) != 1:
            raise ValueError("expected exactly one fit while building output")
        x, y = fits[0]
        _place_object(grid, solid, obj, x, y)
        current_surface = _surface_from_solid(solid)
    return grid


def _build_surface_with_target(
    rng: random.Random,
    height: int,
    width: int,
    target_x: int,
    target_pattern: np.ndarray,
    anchor_height: int,
) -> np.ndarray:
    low = max(height - 7, anchor_height - 1)
    high = height - 3
    surface = np.asarray([rng.randint(low, high) for _ in range(width)], dtype=int)
    target_w = int(target_pattern.size)
    surface[target_x : target_x + target_w] = int(anchor_height) + target_pattern
    return surface


def _pick_distractor(rng: random.Random, surfaces: list[np.ndarray], color: int) -> dict | None:
    width_options = [2, 3, 4, 5]
    rng.shuffle(width_options)
    for width in width_options:
        for _ in range(120):
            if rng.random() < 0.5:
                base = rng.choice([1, 2, 3])
                top = np.asarray([rng.randint(0, base) for _ in range(width)], dtype=int)
                bottom = np.full(width, base, dtype=int)
            else:
                top = np.zeros(width, dtype=int)
                bottom = np.asarray([rng.randint(0, 3) for _ in range(width)], dtype=int)
            if np.all(bottom == bottom[0]) and np.all(top == top[0]):
                continue
            obj = _make_object(color, top, bottom)
            if all(len(_fit_positions(surface, obj)) == 0 for surface in surfaces):
                return obj
    return None


def _can_place_floating(occupied: np.ndarray, obj: dict, x: int, y: int, terrain_surface: np.ndarray) -> bool:
    mask = obj["mask"]
    h, w = mask.shape
    if y < 0 or x < 0 or y + h > occupied.shape[0] or x + w > occupied.shape[1]:
        return False
    for c in range(w):
        rows = np.flatnonzero(mask[:, c])
        if rows.size and y + int(rows[-1]) >= int(terrain_surface[x + c]) - 1:
            return False
    if np.any(occupied[y : y + h, x : x + w] & mask):
        return False
    for rr in range(h):
        for cc in range(w):
            if not mask[rr, cc]:
                continue
            gr = y + rr
            gc = x + cc
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = gr + dr
                nc = gc + dc
                if 0 <= nr < occupied.shape[0] and 0 <= nc < occupied.shape[1] and occupied[nr, nc]:
                    return False
    return True


def _place_floating_objects(
    rng: random.Random,
    height: int,
    width: int,
    terrain_surface: np.ndarray,
    objects: list[dict],
) -> np.ndarray | None:
    grid = _terrain_grid(terrain_surface, height)
    occupied = grid != Color.BLACK
    order = objects[:]
    rng.shuffle(order)
    for obj in order:
        placed = False
        trials = list(range(160))
        for _ in trials:
            x = rng.randint(0, width - int(obj["width"]))
            max_y = int(max(0, terrain_surface.min() - obj["height"] - 2))
            y = rng.randint(0, max_y) if max_y > 0 else 0
            if not _can_place_floating(occupied, obj, x, y, terrain_surface):
                continue
            _place_object(grid, occupied, obj, x, y)
            placed = True
            break
        if not placed:
            return None
    return grid


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = rng.randint(18, 24)
        width = rng.randint(14, 20)
        colors = rng.sample(MOVABLE_COLORS, 5)
        p1 = rng.choice(BOTTOM_TEMPLATES).copy()
        t2 = rng.choice(TOP_TEMPLATES).copy()
        width1 = int(p1.size)
        x1 = rng.randint(2, width - width1 - 2)
        anchor_height = rng.randint(height - 10, height - 7)

        terrain_surface = _build_surface_with_target(rng, height, width, x1, p1, anchor_height)
        obj1 = _make_object(colors[0], np.zeros(width1, dtype=int), p1)
        fit1 = _fit_positions(terrain_surface, obj1)
        if fit1 != [(x1, anchor_height - 1)]:
            continue
        surface1 = _surface_after_place(terrain_surface, obj1, x1, anchor_height - 1)

        base2 = int(t2.max())
        obj2 = _make_object(colors[1], t2, np.full(width1, base2, dtype=int))
        fit2 = _fit_positions(surface1, obj2)
        if len(_fit_positions(terrain_surface, obj2)) != 0:
            continue
        if fit2 != [(x1, int(surface1[x1]) - base2 - 1)]:
            continue
        surface2 = _surface_after_place(surface1, obj2, fit2[0][0], fit2[0][1])

        obj3 = _make_object(colors[2], np.zeros(width1, dtype=int), t2)
        if len(_fit_positions(terrain_surface, obj3)) != 0:
            continue
        if len(_fit_positions(surface1, obj3)) != 0:
            continue
        fit3 = _fit_positions(surface2, obj3)
        if len(fit3) != 1 or fit3[0][0] != x1:
            continue
        surface3 = _surface_after_place(surface2, obj3, fit3[0][0], fit3[0][1])

        distractors: list[dict] = []
        surfaces = [terrain_surface, surface1, surface2, surface3]
        for color in colors[3:]:
            distractor = _pick_distractor(rng, surfaces, color)
            if distractor is not None:
                distractors.append(distractor)
        if not distractors:
            continue

        placements = [(obj1, x1), (obj2, x1), (obj3, x1)]
        expected = _build_expected_grid(height, width, terrain_surface, placements)
        input_grid = _place_floating_objects(rng, height, width, terrain_surface, [obj1, obj2, obj3, *distractors])
        if input_grid is None:
            continue
        if input_grid.shape[0] > 30 or input_grid.shape[1] > 30 or expected.shape[0] > 30 or expected.shape[1] > 30:
            continue
        if int(np.count_nonzero(input_grid == Color.BLACK)) <= int(np.count_nonzero(input_grid == Color.GREY)):
            continue
        if np.array_equal(input_grid, expected):
            continue
        return input_grid, expected
    raise RuntimeError("failed to generate a valid ARC case")


def generate_input() -> np.ndarray:
    rng = random.Random()
    grid, _expected = _generate_case(rng)
    return grid


def main(input_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid, dtype=int)
    if grid.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg = int(get_bg(grid))
    terrain_color = int(Color.GREY)
    output = grid.copy()
    movable_objects: list[dict] = []

    for color in sorted(int(v) for v in np.unique(grid)):
        if color in (bg, terrain_color):
            continue
        movable_objects.extend(_extract_components(grid, color))
        output[grid == color] = bg

    solid = output != bg
    movable_objects.sort(key=lambda obj: int(obj["color"]))

    while True:
        surface = _surface_from_solid(solid)
        chosen: tuple[dict, int, int] | None = None
        for obj in movable_objects:
            fits = _fit_positions(surface, obj)
            if len(fits) == 1:
                x, y = fits[0]
                chosen = (obj, x, y)
                break
        if chosen is None:
            break
        obj, x, y = chosen
        _place_object(output, solid, obj, x, y)
        movable_objects.remove(obj)

    return output
