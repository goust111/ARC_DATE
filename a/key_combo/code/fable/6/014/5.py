from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_HELPER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_HELPER_DIR) not in sys.path:
    sys.path.append(str(_HELPER_DIR))

from arc_utils import Color, find_components, get_bg


RELATIONS = ("nearest", "row", "symmetric")
GEN_HEIGHTS = (23, 25, 27, 29)
GEN_WIDTHS = (21, 23, 25, 27, 29)
GEN_ROWS = (2, 6, 10)
GEN_NEAREST_OFFSETS = ((-1, -2), (-1, 2), (1, -2), (1, 2))
GEN_ATTEMPTS = 400
GEN_PALETTE = [int(Color.BLUE), int(Color.RED), int(Color.GREEN), int(Color.YELLOW), int(Color.GREY), int(Color.PINK), int(Color.ORANGE), int(Color.TEAL), int(Color.MAROON)]


def _coord_tuple(coord: np.ndarray) -> tuple[int, int]:
    return int(coord[0]), int(coord[1])


def _ring_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, center)
    cells: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            cells.append((int(r + dr), int(c + dc)))
    return cells


def _is_ring_component(grid: np.ndarray, comp: dict) -> bool:
    if int(comp["area"]) != 8:
        return False
    r0, c0, r1, c1 = map(int, comp["bbox"])
    if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
        return False
    color = int(comp["color"])
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.ones((3, 3), dtype=bool)
    expected[1, 1] = False
    return np.array_equal(sub == color, expected)


def _find_sprites(grid: np.ndarray, bg: int) -> list[dict]:
    sprites: list[dict] = []
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    for comp in comps:
        if not _is_ring_component(grid, comp):
            continue
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        center = (r0 + 1, c0 + 1)
        sprites.append(
            {
                "border_color": int(comp["color"]),
                "bbox": tuple(map(int, comp["bbox"])),
                "center": center,
            }
        )
    sprites.sort(key=lambda sprite: sprite["center"])
    return sprites


def _find_dots(grid: np.ndarray, bg: int, sprite_centers: set[tuple[int, int]]) -> list[dict]:
    dots: list[dict] = []
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    for comp in comps:
        if int(comp["area"]) != 1:
            continue
        pos = _coord_tuple(comp["coords"][0])
        if pos in sprite_centers:
            continue
        dots.append({"pos": pos, "color": int(comp["color"])})
    dots.sort(key=lambda dot: dot["pos"])
    return dots


def _nearest_dot(center: tuple[int, int], dots: list[dict]) -> dict:
    r, c = map(int, center)
    best_dist = None
    best: list[dict] = []
    for dot in dots:
        rr, cc = map(int, dot["pos"])
        dist = abs(rr - r) + abs(cc - c)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = [dot]
        elif dist == best_dist:
            best.append(dot)
    assert best_dist is not None and len(best) == 1, "nearest dot must be unique"
    return best[0]


def _relation_dot(sprite: dict, dots: list[dict], shape: tuple[int, int], relation: str) -> dict:
    center = tuple(map(int, sprite["center"]))
    if relation == "nearest":
        return _nearest_dot(center, dots)
    if relation == "row":
        matches = [dot for dot in dots if int(dot["pos"][0]) == center[0]]
        assert len(matches) == 1, "same-row dot must be unique"
        return matches[0]
    if relation == "symmetric":
        target = (int(shape[0] - 1 - center[0]), int(shape[1] - 1 - center[1]))
        matches = [dot for dot in dots if tuple(map(int, dot["pos"])) == target]
        assert len(matches) == 1, "symmetric dot must exist exactly once"
        return matches[0]
    raise ValueError(f"unknown relation: {relation}")


def _candidate_colors(sprite: dict, dots: list[dict], shape: tuple[int, int]) -> dict[str, int]:
    colors = {relation: int(_relation_dot(sprite, dots, shape, relation)["color"]) for relation in RELATIONS}
    assert len(set(colors.values())) == len(RELATIONS), "candidate colors must be distinct"
    return colors


def _analyze_task(grid: np.ndarray) -> tuple[int, list[dict], list[dict], int, str]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    sprites = _find_sprites(g, bg)
    assert sprites, "expected hollow sprites"
    centers = {tuple(map(int, sprite["center"])) for sprite in sprites}
    dots = _find_dots(g, bg, centers)
    assert dots, "expected sparse dots"

    filled = [idx for idx, sprite in enumerate(sprites) if int(g[sprite["center"]]) != bg]
    assert len(filled) == 1, "expected exactly one exemplar sprite"
    exemplar_idx = int(filled[0])
    exemplar = sprites[exemplar_idx]
    exemplar_fill = int(g[exemplar["center"]])
    matches = [relation for relation, color in _candidate_colors(exemplar, dots, g.shape).items() if color == exemplar_fill]
    assert len(matches) == 1, "exemplar must identify exactly one relation"
    return bg, sprites, dots, exemplar_idx, matches[0]


def _solve(grid: np.ndarray) -> tuple[np.ndarray, str]:
    g = np.asarray(grid, dtype=int)
    _bg, sprites, dots, _exemplar_idx, rule = _analyze_task(g)
    out = g.copy()
    for sprite in sprites:
        out[sprite["center"]] = int(_relation_dot(sprite, dots, g.shape, rule)["color"])
    return out, rule


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Infer the active pairing rule from the single pre-filled hollow square, then
    fill every hollow square's center with the color of its paired sparse dot.
    Candidate relations are: unique nearest dot, unique same-row dot, and the
    dot at the point-reflected position across the whole-grid center.
    """
    out, _rule = _solve(np.asarray(input_grid, dtype=int))
    return out


def _sample_columns(rng: random.Random, width: int) -> list[int]:
    mid = (width - 1) // 2
    options = [col for col in range(4, width - 4) if abs(col - mid) >= 3]
    for _ in range(200):
        cols = rng.sample(options, k=len(GEN_ROWS))
        if len(set(cols)) == len(GEN_ROWS):
            return cols
    raise RuntimeError("failed to sample sprite columns")


def _draw_generated_case(
    shape: tuple[int, int],
    centers: list[tuple[int, int]],
    row_dots: list[tuple[int, int]],
    sym_dots: list[tuple[int, int]],
    near_dots: list[tuple[int, int]],
    border_colors: list[int],
    dot_colors: list[dict[str, int]],
    rule: str,
    exemplar_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, shape)
    grid = np.full((h, w), int(Color.BLACK), dtype=int)

    for idx, center in enumerate(centers):
        for rr, cc in _ring_cells(center):
            grid[rr, cc] = int(border_colors[idx])

    for idx in range(len(centers)):
        grid[row_dots[idx]] = int(dot_colors[idx]["row"])
        grid[sym_dots[idx]] = int(dot_colors[idx]["symmetric"])
        grid[near_dots[idx]] = int(dot_colors[idx]["nearest"])

    grid[centers[exemplar_idx]] = int(dot_colors[exemplar_idx][rule])

    expected = grid.copy()
    for idx, center in enumerate(centers):
        expected[center] = int(dot_colors[idx][rule])
    return grid, expected


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    height = int(rng.choice(GEN_HEIGHTS))
    width = int(rng.choice(GEN_WIDTHS))
    cols = _sample_columns(rng, width)
    centers = [(int(row), int(col)) for row, col in zip(GEN_ROWS, cols, strict=False)]

    row_dots: list[tuple[int, int]] = []
    sym_dots: list[tuple[int, int]] = []
    near_dots: list[tuple[int, int]] = []
    border_colors: list[int] = []
    dot_colors: list[dict[str, int]] = []

    for center in centers:
        row_side = rng.choice((0, width - 1))
        row_dots.append((int(center[0]), int(row_side)))
        sym_dots.append((int(height - 1 - center[0]), int(width - 1 - center[1])))
        dr, dc = rng.choice(GEN_NEAREST_OFFSETS)
        near_dots.append((int(center[0] + dr), int(center[1] + dc)))

        colors = rng.sample(GEN_PALETTE, 4)
        border_colors.append(int(colors[0]))
        dot_colors.append(
            {
                "nearest": int(colors[1]),
                "row": int(colors[2]),
                "symmetric": int(colors[3]),
            }
        )

    all_object_cells: set[tuple[int, int]] = set()
    for center in centers:
        all_object_cells.update(_ring_cells(center))
        all_object_cells.add(center)
    all_dot_cells = row_dots + sym_dots + near_dots
    assert len(set(all_dot_cells)) == len(all_dot_cells), "dots must be distinct"
    assert set(all_dot_cells).isdisjoint(all_object_cells), "dots must not overlap sprites"

    rule = rng.choice(RELATIONS)
    exemplar_idx = int(rng.randrange(len(centers)))
    return _draw_generated_case(
        shape=(height, width),
        centers=centers,
        row_dots=row_dots,
        sym_dots=sym_dots,
        near_dots=near_dots,
        border_colors=border_colors,
        dot_colors=dot_colors,
        rule=rule,
        exemplar_idx=exemplar_idx,
    )


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, expected = _build_case(rng)
            out, inferred_rule = _solve(grid)
            if inferred_rule not in RELATIONS:
                continue
            if np.array_equal(out, expected):
                return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid exemplar-pairing ARC input")
