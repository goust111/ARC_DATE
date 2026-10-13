from __future__ import annotations

from collections import Counter
import random

import numpy as np


SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _render_bars(shape: tuple[int, int], side_colors: dict[str, int], bg: int) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), bg, dtype=int)
    grid[0, 1:-1] = side_colors["top"]
    grid[h - 1, 1:-1] = side_colors["bottom"]
    grid[1:-1, 0] = side_colors["left"]
    grid[1:-1, w - 1] = side_colors["right"]
    return grid


def _detect_edge_bars(grid: np.ndarray, bg: int) -> dict[int, str]:
    h, w = grid.shape
    assert h >= 7 and w >= 7

    strips = {
        "top": grid[0, 1:-1],
        "bottom": grid[h - 1, 1:-1],
        "left": grid[1:-1, 0],
        "right": grid[1:-1, w - 1],
    }

    color_to_side: dict[int, str] = {}
    for side, strip in strips.items():
        vals = {int(v) for v in np.unique(strip)}
        assert len(vals) == 1
        color = vals.pop()
        assert color != bg
        assert color not in color_to_side
        color_to_side[color] = side

    assert np.all(grid[[0, 0, h - 1, h - 1], [0, w - 1, 0, w - 1]] == bg)
    return color_to_side


def _sprite_body(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _expanded_body(center: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    r, c = center
    cells = set()
    for rr in range(r - 2, r + 3):
        for cc in range(c - 2, c + 3):
            if 0 <= rr < h and 0 <= cc < w:
                cells.add((rr, cc))
    return cells


def _ray_geometry(
    shape: tuple[int, int], center: tuple[int, int], side: str
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[tuple[int, int], ...]]:
    h, w = shape
    r, c = center
    if side == "top":
        path = [(rr, c) for rr in range(r - 2, 0, -1)]
        hit = (0, c)
        stamp = ((0, c - 1), (0, c), (0, c + 1))
    elif side == "bottom":
        path = [(rr, c) for rr in range(r + 2, h - 1)]
        hit = (h - 1, c)
        stamp = ((h - 1, c - 1), (h - 1, c), (h - 1, c + 1))
    elif side == "left":
        path = [(r, cc) for cc in range(c - 2, 0, -1)]
        hit = (r, 0)
        stamp = ((r - 1, 0), (r, 0), (r + 1, 0))
    elif side == "right":
        path = [(r, cc) for cc in range(c + 2, w - 1)]
        hit = (r, w - 1)
        stamp = ((r - 1, w - 1), (r, w - 1), (r + 1, w - 1))
    else:
        raise ValueError(f"unknown side: {side}")

    assert path
    for rr, cc in stamp:
        assert 0 <= rr < h and 0 <= cc < w
    return path, hit, stamp


def _find_sprites(
    grid: np.ndarray, bg: int, color_to_side: dict[int, str]
) -> list[dict[str, object]]:
    h, w = grid.shape
    sprites: list[dict[str, object]] = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            patch = grid[r - 1 : r + 2, c - 1 : c + 2]
            center_color = int(patch[1, 1])
            if center_color == bg:
                continue

            ring = [int(patch[rr, cc]) for rr in range(3) for cc in range(3) if (rr, cc) != (1, 1)]
            if len(set(ring)) != 1:
                continue

            outer_color = ring[0]
            if outer_color == bg or outer_color == center_color:
                continue
            if outer_color not in color_to_side:
                continue

            body = _sprite_body((r, c))
            path, hit, stamp = _ray_geometry(grid.shape, (r, c), color_to_side[outer_color])
            sprites.append(
                {
                    "center": (r, c),
                    "outer": outer_color,
                    "inner": center_color,
                    "side": color_to_side[outer_color],
                    "body": body,
                    "path": path,
                    "hit": hit,
                    "stamp": stamp,
                }
            )

    assert sprites
    centers = [sprite["center"] for sprite in sprites]
    assert len(centers) == len(set(centers))
    return sprites


def _validate_scene_from_grid(
    grid: np.ndarray, bg: int, color_to_side: dict[int, str], sprites: list[dict[str, object]]
) -> None:
    interior = {(int(r), int(c)) for r, c in np.argwhere(grid != bg)}
    edge_cells = {
        (0, c) for c in range(1, grid.shape[1] - 1)
    } | {
        (grid.shape[0] - 1, c) for c in range(1, grid.shape[1] - 1)
    } | {
        (r, 0) for r in range(1, grid.shape[0] - 1)
    } | {
        (r, grid.shape[1] - 1) for r in range(1, grid.shape[0] - 1)
    }
    sprite_cells = set().union(*(sprite["body"] for sprite in sprites))
    assert interior == edge_cells | sprite_cells

    occupied_paths: set[tuple[int, int]] = set()
    occupied_stamps: set[tuple[int, int]] = set()
    seen_body_expanded: list[set[tuple[int, int]]] = []
    seen_body_cells: set[tuple[int, int]] = set()

    for sprite in sprites:
        center = sprite["center"]
        body = sprite["body"]
        path = sprite["path"]
        stamp = sprite["stamp"]
        hit = sprite["hit"]
        outer = int(sprite["outer"])
        inner = int(sprite["inner"])
        side = sprite["side"]

        assert outer in color_to_side and color_to_side[outer] == side
        assert inner != bg and inner != outer

        expanded = _expanded_body(center, grid.shape)
        assert not (body & seen_body_cells)
        assert all(not (expanded & other) for other in seen_body_expanded)
        assert not (body & occupied_paths)
        assert not (set(path) & occupied_paths)
        assert not (set(path) & seen_body_cells)
        assert not (set(stamp) & occupied_stamps)

        for rr, cc in path:
            assert grid[rr, cc] == bg

        if side in ("top", "bottom"):
            assert 2 <= hit[1] <= grid.shape[1] - 3
        else:
            assert 2 <= hit[0] <= grid.shape[0] - 3

        seen_body_expanded.append(expanded)
        seen_body_cells |= body
        occupied_paths |= set(path)
        occupied_stamps |= set(stamp)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire each two-tone sprite toward the edge bar named by its outer color, then stamp the hit on that bar.

    Input structure:
    - The background is the dominant color.
    - The non-corner cells on the four border sides form four full edge bars of distinct colors.
    - Inside the border sit one or more 3x3 two-tone sprites whose outer ring color matches exactly one edge bar color.

    Rule:
    - For each sprite, use the outer ring color as an address: it selects the corresponding top, bottom, left, or right edge bar.
    - Starting just outside the sprite on that side, draw a straight ray of the sprite center color until the cell immediately before the target bar.
    - On the hit bar cell, paint the center cell with the sprite inner color and paint its two neighbors along the bar with the sprite outer color.
    - Keep every sprite body unchanged.

    Key insight:
    - The ring color is not payload but direction: it names which edge receives the delivery, while the center color is what travels along the ray and lands on the bar.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    color_to_side = _detect_edge_bars(g, bg)
    sprites = _find_sprites(g, bg, color_to_side)
    _validate_scene_from_grid(g, bg, color_to_side, sprites)

    out = g.copy()
    for sprite in sprites:
        inner = int(sprite["inner"])
        for rr, cc in sprite["path"]:
            out[rr, cc] = inner

    for sprite in sprites:
        outer = int(sprite["outer"])
        inner = int(sprite["inner"])
        stamp = tuple(sprite["stamp"])
        side = str(sprite["side"])

        if side in ("top", "bottom"):
            out[stamp[0]] = outer
            out[stamp[1]] = inner
            out[stamp[2]] = outer
        else:
            out[stamp[0]] = outer
            out[stamp[1]] = inner
            out[stamp[2]] = outer

    return out


GRID_HEIGHT_RANGE = (14, 23)
GRID_WIDTH_RANGE = (14, 23)
SPRITE_COUNT_RANGE = (2, 5)
GENERATION_RETRIES = 400
MAX_CANDIDATES_PER_SIDE = 96
PLACEMENT_ATTEMPTS = 80
SIDES = ("top", "bottom", "left", "right")
BG = 0


def _render_scene(scene: dict[str, object]) -> np.ndarray:
    grid = _render_bars(scene["shape"], scene["side_colors"], BG)
    for sprite in scene["sprites"]:
        r, c = sprite["center"]
        outer = sprite["outer"]
        inner = sprite["inner"]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                grid[r + dr, c + dc] = outer
        grid[r, c] = inner
    return grid


def _oracle_output(scene: dict[str, object]) -> np.ndarray:
    out = _render_scene(scene)
    for sprite in scene["sprites"]:
        for rr, cc in sprite["path"]:
            out[rr, cc] = sprite["inner"]
    for sprite in scene["sprites"]:
        stamp = sprite["stamp"]
        outer = sprite["outer"]
        inner = sprite["inner"]
        out[stamp[0]] = outer
        out[stamp[1]] = inner
        out[stamp[2]] = outer
    return out


def _candidate_centers(shape: tuple[int, int], side: str) -> list[tuple[int, int]]:
    h, w = shape
    coords = []
    if side == "top":
        for r in range(3, h - 2):
            for c in range(2, w - 2):
                coords.append((r, c))
    elif side == "bottom":
        for r in range(2, h - 3):
            for c in range(2, w - 2):
                coords.append((r, c))
    elif side == "left":
        for r in range(2, h - 2):
            for c in range(3, w - 2):
                coords.append((r, c))
    elif side == "right":
        for r in range(2, h - 2):
            for c in range(2, w - 3):
                coords.append((r, c))
    else:
        raise ValueError(f"unknown side: {side}")
    random.shuffle(coords)
    return coords[:MAX_CANDIDATES_PER_SIDE]


def _place_sprites(
    shape: tuple[int, int], side_colors: dict[str, int], side_order: list[str], inner_colors: list[int]
) -> list[dict[str, object]] | None:
    candidates = {side: _candidate_centers(shape, side) for side in SIDES}
    for _ in range(PLACEMENT_ATTEMPTS):
        placed: list[dict[str, object]] = []
        used_body_cells: set[tuple[int, int]] = set()
        used_path_cells: set[tuple[int, int]] = set()
        used_stamp_cells: set[tuple[int, int]] = set()
        expanded_bodies: list[set[tuple[int, int]]] = []
        success = True

        for i, side in enumerate(side_order):
            outer = side_colors[side]
            inner = inner_colors[i]
            options = candidates[side][:]
            random.shuffle(options)
            picked = None

            for center in options:
                body = _sprite_body(center)
                expanded = _expanded_body(center, shape)
                path, hit, stamp = _ray_geometry(shape, center, side)
                path_set = set(path)
                stamp_set = set(stamp)

                if body & used_body_cells:
                    continue
                if any(expanded & other for other in expanded_bodies):
                    continue
                if body & used_path_cells:
                    continue
                if path_set & used_body_cells:
                    continue
                if path_set & used_path_cells:
                    continue
                if stamp_set & used_stamp_cells:
                    continue

                picked = {
                    "center": center,
                    "outer": outer,
                    "inner": inner,
                    "side": side,
                    "body": body,
                    "path": path,
                    "hit": hit,
                    "stamp": stamp,
                    "expanded": expanded,
                }
                break

            if picked is None:
                success = False
                break

            placed.append(picked)
            used_body_cells.update(picked["body"])
            used_path_cells.update(picked["path"])
            used_stamp_cells.update(picked["stamp"])
            expanded_bodies.append(picked["expanded"])

        if success:
            for sprite in placed:
                sprite.pop("expanded")
            return placed

    return None


def _choose_scene() -> dict[str, object] | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    shape = (h, w)

    bar_colors = random.sample(range(1, 10), 4)
    side_colors = dict(zip(SIDES, bar_colors))

    n_sides = random.randint(2, 4)
    used_sides = random.sample(list(SIDES), n_sides)
    min_sprites = n_sides + 1 if n_sides == 2 else n_sides
    n_sprites = random.randint(min_sprites, SPRITE_COUNT_RANGE[1])

    side_order = used_sides.copy()
    while len(side_order) < n_sprites:
        side_order.append(random.choice(used_sides))
    random.shuffle(side_order)

    payload_pool = [c for c in range(1, 10) if c not in bar_colors]
    palette_size = random.randint(2, min(4, len(payload_pool)))
    payload_palette = random.sample(payload_pool, palette_size)
    inner_colors = [random.choice(payload_palette) for _ in range(n_sprites)]

    placed = _place_sprites(shape, side_colors, side_order, inner_colors)
    if placed is None:
        return None

    scene = {
        "shape": shape,
        "side_colors": side_colors,
        "sprites": placed,
    }
    inp = _render_scene(scene)
    color_to_side = _detect_edge_bars(inp, BG)
    sprites = _find_sprites(inp, BG, color_to_side)
    _validate_scene_from_grid(inp, BG, color_to_side, sprites)

    expected = sorted(
        (sprite["side"], sprite["center"], sprite["outer"], sprite["inner"]) for sprite in placed
    )
    actual = sorted(
        (sprite["side"], sprite["center"], sprite["outer"], sprite["inner"]) for sprite in sprites
    )
    assert expected == actual

    side_counts = Counter(sprite["side"] for sprite in placed)
    assert len(side_counts) >= 2
    assert any(count >= 2 for count in side_counts.values()) or len(side_counts) >= 3
    return scene


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        scene = _choose_scene()
        if scene is None:
            continue

        inp = _render_scene(scene)
        oracle = _oracle_output(scene)
        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, oracle):
            return inp
    raise RuntimeError("failed to generate a valid color-address ray scene")
