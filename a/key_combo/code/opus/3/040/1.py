from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREY = 5
ORANGE = 7
INPUT_COLORS = (1, 3, 4, 6, 8, 9)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _extract_objects(grid: np.ndarray) -> list[dict]:
    mask = (grid != BLACK) & (grid != GREY)
    objects: list[dict] = []
    for coords in _components(mask):
        values = grid[coords[:, 0], coords[:, 1]].astype(int)
        r0, c0, r1, c1 = _bbox(coords)
        rel = coords - np.array([r0, c0], dtype=int)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
        patch[rel[:, 0], rel[:, 1]] = values
        objects.append(
            {
                "coords": coords.copy(),
                "values": values.copy(),
                "patch": patch,
                "bbox": (r0, c0, r1, c1),
                "bottom": int(r1),
                "left": int(c0),
            }
        )
    return objects


def _find_floor_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    assert rows.size == 1, "expected exactly one full-width grey floor row"
    return int(rows[0])


def _settle_grid(grid: np.ndarray) -> tuple[np.ndarray, list[dict], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    floor_r = _find_floor_row(g)

    objects = _extract_objects(g)
    objects.sort(key=lambda obj: (-obj["bottom"], obj["left"]))

    occupied = np.zeros((h, w), dtype=bool)
    occupied[floor_r, :] = True
    settled: list[dict] = []

    out = np.full_like(g, BLACK)
    out[floor_r, :] = GREY

    for obj in objects:
        coords = obj["coords"].copy()
        while True:
            moved = coords.copy()
            moved[:, 0] += 1
            if np.any(moved[:, 0] >= h):
                break
            if np.any(occupied[moved[:, 0], moved[:, 1]]):
                break
            coords = moved

        occupied[coords[:, 0], coords[:, 1]] = True
        out[coords[:, 0], coords[:, 1]] = obj["values"]
        settled.append(
            {
                "coords": coords,
                "values": obj["values"].copy(),
                "patch": obj["patch"].copy(),
                "bbox": _bbox(coords),
            }
        )

    return out, settled, floor_r


def _enclosed_black_regions(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    regions: list[np.ndarray] = []
    for coords in _components(grid == BLACK):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == h - 1):
            continue
        if np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == w - 1):
            continue
        regions.append(coords)
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the colored objects onto the grey floor, then fill any newly enclosed black chambers by area parity.

    Input structure:
    - Black background with exactly one full-width grey floor row near the lower half.
    - Several small 4-connected non-black, non-grey rigid objects appear only above the floor.
    - The objects may start at different heights but their shapes and colors are preserved during motion.

    Rule:
    - Treat every non-grey foreground component as one rigid object.
    - Settle the objects downward in order of decreasing lowest row, stopping at the grey floor, the bottom edge, or already settled objects.
    - After settling, find every 4-connected black region that does not touch the outer grid border.
    - Fill each enclosed black region red if its area is even, otherwise orange.

    Key insight:
    - The fill targets are not marked in the input; they only emerge after gravity changes the topology and creates closed chambers above the floor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    settled, _objects, _floor_r = _settle_grid(g)
    out = settled.copy()
    for region in _enclosed_black_regions(settled):
        color = RED if int(region.shape[0]) % 2 == 0 else ORANGE
        out[region[:, 0], region[:, 1]] = color
    return out


def _left_piece(width: int, color: int) -> np.ndarray:
    patch = np.zeros((2, width), dtype=int)
    patch[0, :] = color
    patch[1, 0] = color
    return patch


def _right_piece(width: int, color: int) -> np.ndarray:
    patch = np.zeros((2, width), dtype=int)
    patch[0, :] = color
    patch[1, -1] = color
    return patch


def _extra_patch(color: int) -> np.ndarray:
    choices = [
        np.array([[color, color]], dtype=int),
        np.array([[color, color, color]], dtype=int),
        np.array([[color], [color]], dtype=int),
        np.array([[color], [color], [color]], dtype=int),
        np.array([[color, color], [color, color]], dtype=int),
    ]
    return random.choice(choices).copy()


def _random_color() -> int:
    return int(random.choice(INPUT_COLORS))


def _random_gaps(total_extra: int, slots: int) -> list[int]:
    gaps = [0] * slots
    for _ in range(total_extra):
        gaps[random.randrange(slots)] += 1
    return gaps


def _render_objects(
    h: int,
    w: int,
    floor_r: int,
    placed: list[tuple[np.ndarray, int, int]],
    *,
    lifted: bool,
) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    grid[floor_r, :] = GREY
    occupied = np.zeros((h, w), dtype=bool)
    occupied[floor_r, :] = True

    for patch, final_c0, lift in placed:
        ph, pw = patch.shape
        final_r0 = floor_r - ph
        r0 = final_r0 - lift if lifted else final_r0
        if r0 < 0:
            return None
        cells = np.argwhere(patch != BLACK)
        rows = cells[:, 0] + r0
        cols = cells[:, 1] + final_c0
        if np.any(rows >= h) or np.any(cols >= w):
            return None
        if np.any(occupied[rows, cols]):
            return None
        grid[rows, cols] = patch[cells[:, 0], cells[:, 1]]
        occupied[rows, cols] = True

    return grid


def _build_layout() -> np.ndarray | None:
    even_widths = random.choice([(2, 2), (3, 3)])
    odd_widths = random.choice([(2, 3), (3, 2)])
    module_specs = [
        ("even", even_widths, _left_piece(even_widths[0], _random_color()), _right_piece(even_widths[1], _random_color())),
        ("odd", odd_widths, _left_piece(odd_widths[0], _random_color()), _right_piece(odd_widths[1], _random_color())),
    ]
    random.shuffle(module_specs)

    extra_count = random.randint(0, 2)
    groups: list[dict] = []
    for parity, widths, left_patch, right_patch in module_specs:
        groups.append(
            {
                "kind": "module",
                "parity": parity,
                "width": widths[0] + widths[1],
                "patches": [left_patch, right_patch],
            }
        )
    for _ in range(extra_count):
        patch = _extra_patch(_random_color())
        groups.append({"kind": "extra", "width": patch.shape[1], "patches": [patch]})
    random.shuffle(groups)

    min_w = sum(group["width"] for group in groups) + 1 + 1 + 2 * (len(groups) - 1)
    if min_w > 30:
        return None
    w = random.randint(max(13, min_w), 30)

    h = random.randint(11, 16)
    floor_r = random.randint(max(6, h // 2), h - 3)

    base_gaps = [1] + [2] * (len(groups) - 1) + [1]
    extra = w - sum(group["width"] for group in groups) - sum(base_gaps)
    gaps = base_gaps[:]
    bonus = _random_gaps(extra, len(gaps))
    gaps = [a + b for a, b in zip(gaps, bonus)]

    placed: list[tuple[np.ndarray, int, int]] = []
    cursor = gaps[0]
    for idx, group in enumerate(groups):
        if group["kind"] == "module":
            left_patch, right_patch = group["patches"]
            final_top = floor_r - left_patch.shape[0]
            if final_top < 2:
                return None
            lift_a, lift_b = random.sample(range(1, final_top + 1), 2)
            placed.append((left_patch, cursor, lift_a))
            placed.append((right_patch, cursor + left_patch.shape[1], lift_b))
        else:
            patch = group["patches"][0]
            final_top = floor_r - patch.shape[0]
            if final_top < 1:
                return None
            placed.append((patch, cursor, random.randint(1, final_top)))
        cursor += group["width"] + gaps[idx + 1]

    inp = _render_objects(h, w, floor_r, placed, lifted=True)
    if inp is None:
        return None
    out = main(inp)
    settled, _objects, _ = _settle_grid(inp)

    if len(_extract_objects(inp)) != len(placed):
        return None
    if _enclosed_black_regions(inp):
        return None

    enclosed = _enclosed_black_regions(settled)
    if not enclosed:
        return None
    parities = {int(region.shape[0]) % 2 for region in enclosed}
    if parities != {0, 1}:
        return None
    if not np.any(out == RED) or not np.any(out == ORANGE):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_layout()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
