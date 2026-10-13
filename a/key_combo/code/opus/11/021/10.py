from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

PALETTE_COLORS = (BLUE, RED, GREEN, YELLOW, PINK, ORANGE, TEAL)
WORK_LEFT_BOUNDARY = 1


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == color, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, cell)) for cell in coords}


def _sprite_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        sprite[int(r - r0), int(c - c0)] = True
    return sprite


def _paint_cells(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    for r, c in coords:
        grid[int(r), int(c)] = int(color)


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return coords + np.array([int(dr), int(dc)], dtype=int)


def _push_direction(coords: np.ndarray, marker_cells: set[tuple[int, int]]) -> str:
    sides: set[str] = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if (rr - 1, cc) in marker_cells:
            sides.add("up")
        if (rr + 1, cc) in marker_cells:
            sides.add("down")
        if (rr, cc - 1) in marker_cells:
            sides.add("left")
        if (rr, cc + 1) in marker_cells:
            sides.add("right")
    assert len(sides) == 1, "each grey object must touch the marker on exactly one side"
    contact_side = next(iter(sides))
    return {"up": "down", "down": "up", "left": "right", "right": "left"}[contact_side]


def _push_to_boundary(coords: np.ndarray, direction: str, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    r0, c0, r1, c1 = _bbox(coords)
    if direction == "left":
        dc = WORK_LEFT_BOUNDARY - c0
        return _translate(coords, 0, dc)
    if direction == "right":
        dc = (w - 1) - c1
        return _translate(coords, 0, dc)
    if direction == "up":
        dr = -r0
        return _translate(coords, dr, 0)
    if direction == "down":
        dr = (h - 1) - r1
        return _translate(coords, dr, 0)
    raise AssertionError(f"unknown direction {direction}")


def _sort_mode(directions: list[str]) -> str:
    kinds = set(directions)
    if kinds <= {"left", "right"}:
        return "horizontal"
    if kinds <= {"up", "down"}:
        return "vertical"
    return "mixed"


def _sort_key(coords: np.ndarray, mode: str) -> tuple[int, int]:
    r0, c0, _, _ = _bbox(coords)
    if mode == "horizontal":
        return int(c0), int(r0)
    return int(r0), int(c0)


def _read_palette(grid: np.ndarray) -> list[int]:
    rows = [r for r in range(grid.shape[0]) if int(grid[r, 0]) != BLACK]
    palette = [int(grid[r, 0]) for r in rows]
    assert 3 <= len(palette) <= 5, "expected 3 to 5 palette colors"
    assert len(set(palette)) == len(palette), "palette colors must be distinct"
    assert all(color not in {BLACK, GREY, MAROON} for color in palette), "unexpected palette color"
    return palette


def _analyze_scene(grid: np.ndarray) -> tuple[list[int], np.ndarray, list[dict], str]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette = _read_palette(g)

    markers = _components_of_color(g, MAROON)
    assert len(markers) == 1, "expected exactly one maroon marker"
    marker = markers[0]
    marker_cells = _coords_to_set(marker)

    objects = _components_of_color(g, GREY)
    assert len(objects) == len(palette), "palette length must equal grey-object count"

    records: list[dict] = []
    for index, coords in enumerate(objects):
        direction = _push_direction(coords, marker_cells)
        final_coords = _push_to_boundary(coords, direction, g.shape)
        records.append(
            {
                "id": index,
                "coords": coords,
                "final": final_coords,
                "direction": direction,
                "area": int(coords.shape[0]),
            }
        )

    mode = _sort_mode([record["direction"] for record in records])
    return palette, marker, records, mode


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each grey object away from the maroon marker, then recolor the landed objects by palette order with a singleton exception.

    Input structure:
    - The background is black, and column 0 is a palette column with 3 to 5 distinct colored pixels from top to bottom.
    - A single compact maroon marker sits in the main play area to the right of the palette.
    - The remaining foreground consists of grey 4-connected objects touching the marker on exactly one side, with exactly one singleton grey object.

    Rule:
    - Read the palette colors from top to bottom in column 0.
    - For each grey object, detect which side touches the maroon marker and push the object straight away from the marker until its front edge reaches the outer boundary of the play area.
    - Ignore singleton grey objects when forming the normal ordering; sort the non-singleton landed objects by final position, using column order for purely horizontal pushes, row order for purely vertical pushes, and row-major order when directions are mixed.
    - Recolor the sorted non-singleton objects with the palette colors from the start, recolor every singleton landed object with the last palette color, keep the maroon marker fixed, and erase the palette column to black.

    Key insight:
    - The marker side determines motion, motion determines the comparison order, and only after that order is known can the palette be matched; the singleton object breaks the apparent one-to-one queue and always jumps to the last palette color.
    """
    palette, marker, records, mode = _analyze_scene(grid)
    singletons = [record for record in records if record["area"] == 1]
    nonsingletons = [record for record in records if record["area"] > 1]
    assert len(singletons) == 1, "expected exactly one singleton grey object"
    assert len(nonsingletons) == len(palette) - 1, "palette must reserve its last color for the singleton"

    out = np.zeros_like(grid, dtype=int)
    _paint_cells(out, marker, MAROON)

    ordered = sorted(nonsingletons, key=lambda record: _sort_key(record["final"], mode))
    for record, color in zip(ordered, palette[:-1]):
        _paint_cells(out, record["final"], color)
    for record in singletons:
        _paint_cells(out, record["final"], palette[-1])
    return out


MARKER_TEMPLATES = (
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
)

GENERIC_TEMPLATES = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1], [1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1], [1], [1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
)

WIDTH_TEMPLATES: dict[int, tuple[np.ndarray, ...]] = {
    2: (
        np.array([[1, 1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1], [1, 1]], dtype=bool),
    ),
    3: (
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    ),
    4: (
        np.array([[1, 1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1, 0], [0, 0, 1, 1]], dtype=bool),
    ),
    5: (
        np.array([[1, 1, 1, 1, 1]], dtype=bool),
    ),
}

HEIGHT_TEMPLATES: dict[int, tuple[np.ndarray, ...]] = {
    2: tuple(template.T for template in WIDTH_TEMPLATES[2]),
    3: tuple(template.T for template in WIDTH_TEMPLATES[3]),
    4: tuple(template.T for template in WIDTH_TEMPLATES[4]),
    5: tuple(template.T for template in WIDTH_TEMPLATES[5]),
}


def _trim_sprite(sprite: np.ndarray) -> np.ndarray:
    rows = np.where(sprite.any(axis=1))[0]
    cols = np.where(sprite.any(axis=0))[0]
    return sprite[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].copy()


def _random_marker() -> np.ndarray:
    sprite = random.choice(MARKER_TEMPLATES).copy()
    if random.random() < 0.5:
        sprite = np.fliplr(sprite)
    turns = random.randint(0, 3)
    sprite = np.rot90(sprite, turns)
    return _trim_sprite(sprite)


def _make_object_sprite(branch: str, role: str, width_or_height: int | None = None) -> np.ndarray:
    if role == "singleton":
        return np.array([[1]], dtype=bool)
    if branch == "horizontal":
        assert width_or_height is not None
        return random.choice(WIDTH_TEMPLATES[int(width_or_height)]).copy()
    if branch == "vertical":
        assert width_or_height is not None
        return random.choice(HEIGHT_TEMPLATES[int(width_or_height)]).copy()
    sprite = random.choice(GENERIC_TEMPLATES).copy()
    if random.random() < 0.5:
        sprite = np.fliplr(sprite)
    turns = random.randint(0, 3)
    return _trim_sprite(np.rot90(sprite, turns))


def _sprite_cells_at(sprite: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(sprite)
    return coords + np.array([int(top), int(left)], dtype=int)


def _expanded_cells(coords: np.ndarray) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        out.add((rr, cc))
        out.add((rr - 1, cc))
        out.add((rr + 1, cc))
        out.add((rr, cc - 1))
        out.add((rr, cc + 1))
    return out


def _candidate_positions(
    sprite: np.ndarray,
    side: str,
    marker_top: int,
    marker_left: int,
    marker_sprite: np.ndarray,
    grid_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = map(int, sprite.shape)
    mh, mw = map(int, marker_sprite.shape)
    marker_cells = np.argwhere(marker_sprite)
    edge_rows = np.where(sprite[:, -1])[0]
    edge_cols = np.where(sprite[-1, :])[0]
    top_rows = np.where(sprite[0, :])[0]
    left_cols = np.where(sprite[:, 0])[0]

    candidates: set[tuple[int, int]] = set()
    if side == "left":
        left = marker_left - w
        if left < WORK_LEFT_BOUNDARY:
            return []
        marker_rows = [marker_top + int(r) for r, c in marker_cells if int(c) == 0]
        for edge_r in edge_rows:
            for row in marker_rows:
                top = int(row - edge_r)
                candidates.add((top, left))
    elif side == "right":
        left = marker_left + mw
        if left + w > grid_shape[1]:
            return []
        marker_rows = [marker_top + int(r) for r, c in marker_cells if int(c) == mw - 1]
        sprite_rows = np.where(sprite[:, 0])[0]
        for edge_r in sprite_rows:
            for row in marker_rows:
                top = int(row - edge_r)
                candidates.add((top, left))
    elif side == "up":
        top = marker_top - h
        if top < 0:
            return []
        marker_cols = [marker_left + int(c) for r, c in marker_cells if int(r) == 0]
        for edge_c in edge_cols:
            for col in marker_cols:
                left = int(col - edge_c)
                candidates.add((top, left))
    else:
        top = marker_top + mh
        if top + h > grid_shape[0]:
            return []
        marker_cols = [marker_left + int(c) for r, c in marker_cells if int(r) == mh - 1]
        sprite_cols = np.where(sprite[0, :])[0]
        for edge_c in sprite_cols:
            for col in marker_cols:
                left = int(col - edge_c)
                candidates.add((top, left))

    valid = []
    for top, left in candidates:
        if top < 0 or left < WORK_LEFT_BOUNDARY or top + h > grid_shape[0] or left + w > grid_shape[1]:
            continue
        valid.append((int(top), int(left)))
    random.shuffle(valid)
    return valid


def _push_cells_from_sprite(sprite: np.ndarray, top: int, left: int, direction: str, grid_shape: tuple[int, int]) -> np.ndarray:
    coords = _sprite_cells_at(sprite, top, left)
    return _push_to_boundary(coords, direction, grid_shape)


def _build_specs(k: int, branch: str) -> list[dict]:
    specs: list[dict] = []
    if branch == "horizontal":
        specs.append({"side": "left", "role": "singleton", "sprite": _make_object_sprite(branch, "singleton")})
        widths = random.sample([2, 3, 4, 5], k - 1)
        random.shuffle(widths)
        for width in widths:
            specs.append({"side": "right", "role": "normal", "sprite": _make_object_sprite(branch, "normal", width)})
    elif branch == "vertical":
        specs.append({"side": "up", "role": "singleton", "sprite": _make_object_sprite(branch, "singleton")})
        heights = random.sample([2, 3, 4, 5], k - 1)
        random.shuffle(heights)
        for height in heights:
            specs.append({"side": "down", "role": "normal", "sprite": _make_object_sprite(branch, "normal", height)})
    else:
        specs.append({"side": "right", "role": "singleton", "sprite": _make_object_sprite(branch, "singleton")})
        side_pool = ["left", "down"]
        extra_choices = ["up", "right", "left", "down"]
        while len(side_pool) < k - 1:
            side_pool.append(random.choice(extra_choices))
        for side in side_pool[: k - 1]:
            specs.append({"side": side, "role": "normal", "sprite": _make_object_sprite(branch, "normal")})
    if len(specs) > 2:
        tail = specs[1:]
        random.shuffle(tail)
        specs[1:] = tail
    return specs


def _sort_records(records: list[dict], mode: str, key_name: str) -> list[dict]:
    return sorted(records, key=lambda record: _sort_key(record[key_name], mode))


def _validate_non_degenerate(grid: np.ndarray, built_records: list[dict], branch: str) -> bool:
    palette, marker, records, mode = _analyze_scene(grid)
    if branch == "horizontal" and mode != "horizontal":
        return False
    if branch == "vertical" and mode != "vertical":
        return False
    if branch == "mixed" and mode != "mixed":
        return False

    directions = {record["direction"] for record in records}
    if len(directions) < 2:
        return False

    singletons = [record for record in records if record["area"] == 1]
    nonsingletons = [record for record in records if record["area"] > 1]
    if len(singletons) != 1 or len(nonsingletons) != len(palette) - 1:
        return False

    final_keys = [_sort_key(record["final"], mode) for record in nonsingletons]
    if len(set(final_keys)) != len(final_keys):
        return False

    all_objects_sorted = sorted(records, key=lambda record: _sort_key(record["final"], mode))
    singleton_rank = [record["area"] for record in all_objects_sorted].index(1)
    if palette[singleton_rank] == palette[-1]:
        return False

    before = [record["id"] for record in _sort_records(records=nonsingletons, mode=mode, key_name="coords")]
    after = [record["id"] for record in _sort_records(records=nonsingletons, mode=mode, key_name="final")]
    if before == after:
        return False

    out = main(grid)
    if not np.all(out[:, 0] == BLACK):
        return False
    if np.count_nonzero(out == MAROON) != np.count_nonzero(grid == MAROON):
        return False
    if np.array_equal(out, grid):
        return False

    marker_set = _coords_to_set(marker)
    for record in records:
        if record["area"] == 1 and not np.array_equal(np.argwhere(out == palette[-1]), record["final"]):
            break
    else:
        pass

    built_singletons = [record for record in built_records if record["role"] == "singleton"]
    if len(built_singletons) != 1:
        return False
    return marker_set is not None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 17)
    w = random.randint(16, 19)
    branch = random.choices(["mixed", "horizontal", "vertical"], weights=[5, 2, 2])[0]
    k = random.randint(3, 5) if branch == "mixed" else random.randint(3, 4)
    palette = random.sample(PALETTE_COLORS, k)
    marker_sprite = _random_marker()
    mh, mw = map(int, marker_sprite.shape)
    marker_top = random.randint(4, h - mh - 5)
    marker_left = random.randint(6, w - mw - 6)
    specs = _build_specs(k, branch)
    specs.sort(key=lambda spec: -int(np.count_nonzero(spec["sprite"])))

    records: list[dict] = []
    marker_coords = np.argwhere(marker_sprite) + np.array([marker_top, marker_left], dtype=int)
    marker_cells = _coords_to_set(marker_coords)
    used_input_buffer: set[tuple[int, int]] = set()
    used_final: set[tuple[int, int]] = set()

    for idx, spec in enumerate(specs):
        sprite = spec["sprite"]
        side = spec["side"]
        direction = {"left": "right", "right": "left", "up": "down", "down": "up"}[side]
        candidates = _candidate_positions(sprite, side, marker_top, marker_left, marker_sprite, (h, w))
        placed = False
        for top, left in candidates:
            coords = _sprite_cells_at(sprite, top, left)
            coord_set = _coords_to_set(coords)
            if coord_set & marker_cells:
                continue
            if coord_set & used_input_buffer:
                continue
            final_coords = _push_cells_from_sprite(sprite, top, left, direction, (h, w))
            final_set = _coords_to_set(final_coords)
            if any(c == 0 for _, c in final_set):
                continue
            if final_set & used_final:
                continue
            records.append(
                {
                    "id": idx,
                    "role": spec["role"],
                    "side": side,
                    "direction": direction,
                    "sprite": sprite,
                    "coords": coords,
                    "final": final_coords,
                }
            )
            used_input_buffer |= _expanded_cells(coords)
            used_final |= final_set
            placed = True
            break
        if not placed:
            return None

    grid = np.zeros((h, w), dtype=int)
    palette_rows = []
    row = random.randint(0, 1)
    for _ in palette:
        palette_rows.append(row)
        row += random.randint(2, 3)
    if palette_rows[-1] >= h:
        return None
    for r, color in zip(palette_rows, palette):
        grid[r, 0] = color
    _paint_cells(grid, marker_coords, MAROON)
    for record in records:
        _paint_cells(grid, record["coords"], GREY)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if not _validate_non_degenerate(grid, records, branch):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
