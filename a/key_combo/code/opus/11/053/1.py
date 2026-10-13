from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
PALETTE = [1, 2, 3, 4, 6]
MAX_ATTEMPTS = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not mask[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            cells: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _is_symmetric(mask: np.ndarray) -> bool:
    return (
        np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.rot90(mask, 2))
    )


def _room_components(grid: np.ndarray) -> list[np.ndarray]:
    return _components(grid != WALL)


def _analyze_room(grid: np.ndarray, room: np.ndarray) -> tuple[int, np.ndarray, bool]:
    room_colors = sorted({int(grid[r, c]) for r, c in room} - {BG})
    assert room_colors and WALL not in room_colors
    assert len(room_colors) == 1
    color = room_colors[0]
    sprite_coords = room[np.array([grid[r, c] == color for r, c in room], dtype=bool)]
    sprite_components = _components(_coords_to_mask(sprite_coords, grid.shape))
    assert len(sprite_components) == 1
    sprite = sprite_components[0]
    assert 3 <= len(sprite) <= 6
    for r, c in sprite:
        for nr, nc in _neighbors4(int(r), int(c)):
            assert 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]
            assert grid[nr, nc] != WALL
    return color, sprite, _is_symmetric(_crop_mask(sprite))


def _coords_to_mask(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify the lone sprite in each wall-bounded chamber by symmetry, then flood-fill or erase it.

    Input structure:
    - The grid has black background and grey one-pixel walls that partition the canvas into closed 4-connected chambers.
    - Each chamber contains exactly one 4-connected monochrome sprite in a non-wall color, with empty background around it.

    Rule:
    - Treat grey cells as barriers and find each non-wall chamber.
    - In every chamber, isolate the unique sprite, crop it to its bounding box, and test horizontal, vertical, and 180-degree rotational symmetry.
    - If the cropped sprite satisfies any of those symmetries, recolor the entire chamber to the sprite color.
    - Otherwise erase only the sprite back to black, leaving the chamber empty.

    Key insight:
    - Symmetry is evaluated independently inside each chamber and acts as the gate that chooses between two different in-place operations on that chamber.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for room in _room_components(g):
        color, sprite, symmetric = _analyze_room(g, room)
        if symmetric:
            out[room[:, 0], room[:, 1]] = color
        else:
            out[sprite[:, 0], sprite[:, 1]] = BG
    return out


SYMMETRIC_TEMPLATES = [
    np.array([(0, 0), (0, 1), (1, 0), (1, 1)], dtype=int),                      # 2x2 square
    np.array([(0, 0), (0, 1), (0, 2)], dtype=int),                              # 1x3 line
    np.array([(0, 0), (1, 0), (2, 0)], dtype=int),                              # 3x1 line
    np.array([(0, 1), (1, 0), (1, 1), (1, 2)], dtype=int),                      # T tetromino
    np.array([(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)], dtype=int),              # plus
    np.array([(0, 0), (0, 1), (0, 2), (1, 1)], dtype=int),                      # top-heavy T
]

ASYMMETRIC_TEMPLATES = [
    np.array([(0, 0), (0, 1), (1, 0)], dtype=int),                              # L triomino
    np.array([(0, 0), (1, 0), (2, 0), (2, 1)], dtype=int),                      # L tetromino
    np.array([(0, 0), (0, 1), (1, 1), (2, 1)], dtype=int),                      # hooked column
    np.array([(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)], dtype=int),              # staircase
    np.array([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)], dtype=int),              # P pentomino
    np.array([(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (2, 0)], dtype=int),      # asymmetric hexomino
]


def _template_bbox(template: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(template)
    return r1 - r0 + 1, c1 - c0 + 1


def _room_specs() -> list[dict]:
    kind = random.choice(["four", "five_top_heavy", "five_bottom_heavy"])
    if kind == "four":
        row_heights = [random.choice([4, 5]), random.choice([4, 5])]
        col_widths = [random.choice([4, 5]), random.choice([4, 5])]
        specs = []
        r0 = 1
        for rh in row_heights:
            c0 = 1
            for cw in col_widths:
                specs.append({"r0": r0, "c0": c0, "h": rh, "w": cw})
                c0 += cw + 1
            r0 += rh + 1
        return specs

    top_two = kind == "five_top_heavy"
    top_h = 4 if top_two else random.choice([4, 5])
    bottom_h = random.choice([4, 5]) if top_two else 4
    top_widths = [6, 6] if top_two else [4, 4, 4]
    bottom_widths = [4, 4, 4] if top_two else [6, 6]
    specs = []
    r0 = 1
    for band_h, widths in [(top_h, top_widths), (bottom_h, bottom_widths)]:
        c0 = 1
        for width in widths:
            specs.append({"r0": r0, "c0": c0, "h": band_h, "w": width})
            c0 += width + 1
        r0 += band_h + 1
    return specs


def _build_walls(room_specs: list[dict]) -> np.ndarray:
    h = max(spec["r0"] + spec["h"] for spec in room_specs) + 1
    w = max(spec["c0"] + spec["w"] for spec in room_specs) + 1
    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL
    for spec in room_specs:
        r0, c0, rh, cw = spec["r0"], spec["c0"], spec["h"], spec["w"]
        grid[r0 - 1 : r0 + rh + 1, c0 - 1] = WALL
        grid[r0 - 1 : r0 + rh + 1, c0 + cw] = WALL
        grid[r0 - 1, c0 - 1 : c0 + cw + 1] = WALL
        grid[r0 + rh, c0 - 1 : c0 + cw + 1] = WALL
    return grid


def _pick_template(symmetric: bool, usable_h: int, usable_w: int) -> np.ndarray:
    pool = SYMMETRIC_TEMPLATES if symmetric else ASYMMETRIC_TEMPLATES
    candidates = []
    for template in pool:
        th, tw = _template_bbox(template)
        if th <= usable_h and tw <= usable_w:
            candidates.append(template)
    assert candidates
    return random.choice(candidates).copy()


def _place_sprite(grid: np.ndarray, spec: dict, color: int, symmetric: bool) -> None:
    usable_h = spec["h"] - 2
    usable_w = spec["w"] - 2
    template = _pick_template(symmetric, usable_h, usable_w)
    th, tw = _template_bbox(template)
    max_r = usable_h - th
    max_c = usable_w - tw
    off_r = random.randint(0, max_r)
    off_c = random.randint(0, max_c)
    base_r = spec["r0"] + 1 + off_r
    base_c = spec["c0"] + 1 + off_c
    for dr, dc in template:
        grid[base_r + int(dr), base_c + int(dc)] = color


def _rand_transform(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.array(out, dtype=int, copy=True)


def _validate_templates() -> None:
    for template in SYMMETRIC_TEMPLATES:
        assert _is_symmetric(_crop_mask(template))
        assert len(template) >= 3
    for template in ASYMMETRIC_TEMPLATES:
        assert not _is_symmetric(_crop_mask(template))
        assert len(template) >= 3


def _validate_input(grid: np.ndarray) -> list[bool]:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert set(np.unique(grid)).issubset(set(PALETTE) | {BG, WALL})
    rooms = _room_components(grid)
    assert 4 <= len(rooms) <= 5
    colors = []
    symmetry_flags: list[bool] = []
    for room in rooms:
        color, sprite, symmetric = _analyze_room(grid, room)
        colors.append(color)
        symmetry_flags.append(symmetric)
        assert len(set(map(tuple, sprite))) == len(sprite)
        assert len(sprite) < len(room), "flood-fill branch must have background to consume"
        assert room[:, 0].min() > 0 and room[:, 1].min() > 0
        assert room[:, 0].max() < grid.shape[0] - 1 and room[:, 1].max() < grid.shape[1] - 1
    assert len(set(colors)) == len(colors)
    assert any(symmetry_flags) and not all(symmetry_flags)
    return symmetry_flags


def _validate_behavior(inp: np.ndarray, out: np.ndarray, symmetry_flags: list[bool]) -> None:
    rooms = _room_components(inp)
    assert np.array_equal(inp == WALL, out == WALL)
    assert not np.array_equal(inp, out)
    seen_fill = False
    seen_erase = False
    for room, symmetric in zip(rooms, symmetry_flags):
        color, sprite, _ = _analyze_room(inp, room)
        room_vals = out[room[:, 0], room[:, 1]]
        if symmetric:
            assert np.all(room_vals == color)
            seen_fill = True
        else:
            assert np.all(room_vals == BG)
            assert np.all(out[sprite[:, 0], sprite[:, 1]] == BG)
            seen_erase = True
    assert seen_fill and seen_erase


def generate_input() -> np.ndarray:
    _validate_templates()
    for _ in range(MAX_ATTEMPTS):
        room_specs = _room_specs()
        grid = _build_walls(room_specs)
        n = len(room_specs)
        colors = random.sample(PALETTE, n)

        sym_count = random.randint(1, n - 1)
        statuses = [True] * sym_count + [False] * (n - sym_count)
        random.shuffle(statuses)

        try:
            for spec, color, symmetric in zip(room_specs, colors, statuses):
                _place_sprite(grid, spec, color, symmetric)
            grid = _rand_transform(grid)
            symmetry_flags = _validate_input(grid)
            out = main(grid)
            _validate_behavior(grid, out, symmetry_flags)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
