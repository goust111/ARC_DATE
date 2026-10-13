from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
BODY = 5
RAY_COLORS = (1, 2, 3)
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Each 3x3 sprite has a fixed bright center and one missing side cell.
# The grey centroid shifts away from the gap, so the ray points toward the gap.
GAP_BY_DIRECTION = {
    (-1, 0): (0, 1),
    (1, 0): (2, 1),
    (0, -1): (1, 0),
    (0, 1): (1, 2),
}

_SMOKE_TESTED = False


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _decode_direction(grey_points: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    coords = np.array(grey_points, dtype=float)
    centroid_r, centroid_c = coords.mean(axis=0)
    row_bias = float(centroid_r - center_r)
    col_bias = float(centroid_c - center_c)
    if abs(row_bias) == abs(col_bias):
        raise AssertionError("grey centroid must choose a unique axis")
    if abs(row_bias) > abs(col_bias):
        return (-1, 0) if row_bias > 0 else (1, 0)
    return (0, -1) if col_bias > 0 else (0, 1)


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    sprites: list[dict] = []
    for pts in _components(g != BACKGROUND):
        r0, c0, r1, c1 = _bbox(pts)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        bright_local = np.argwhere((sub != BACKGROUND) & (sub != BODY))
        grey_local = np.argwhere(sub == BODY)
        assert bright_local.shape == (1, 2), "each sprite must have one bright code pixel"
        assert grey_local.shape == (7, 2), "each sprite must have seven grey body pixels"
        assert sub.shape == (3, 3), "generated sprites use a 3x3 bbox"
        center = (r0 + 1, c0 + 1)
        bright_pos = tuple(map(int, bright_local[0]))
        assert bright_pos == (1, 1), "code pixel must use the fixed center slot"
        grey_points = [(r0 + int(rr), c0 + int(cc)) for rr, cc in grey_local]
        direction = _decode_direction(grey_points, (r0, c0, r1, c1))
        color = int(sub[1, 1])
        sprites.append(
            {
                "bbox": (r0, c0, r1, c1),
                "origin": center,
                "direction": direction,
                "color": color,
            }
        )
    return sprites


def _ray_cells(shape: tuple[int, int], origin: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = origin
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _paint_rays(base: np.ndarray, sprites: list[dict]) -> np.ndarray:
    g = np.asarray(base, dtype=int)
    out = g.copy()
    for sprite in sprites:
        ray = _ray_cells(g.shape, sprite["origin"], sprite["direction"])
        for step, (r, c) in enumerate(ray, start=1):
            if step % 2 == 1 and g[r, c] == BACKGROUND:
                out[r, c] = sprite["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each dual-signal sprite and cast a colored dotted ray from its center.

    Input structure:
    - The grid has black background with 3x3 composite sprites scattered apart.
    - Each sprite contains seven grey body pixels plus one bright center pixel.
    - The grey body is asymmetric, and its centroid is biased toward one cardinal side of the bbox.

    Rule:
    - For each sprite, compare the grey-pixel centroid with the bbox center and take the opposite of the dominant row/column bias as the ray direction.
    - Read the unique non-black, non-grey center pixel as the ray color.
    - Starting from the bbox center, paint that color on every other cell along the decoded direction until the grid edge, skipping occupied cells but continuing onward.
    - Keep the original grey body and bright center pixels unchanged.

    Key insight:
    - Direction and color are encoded independently inside the same object: geometry supplies the ray direction, while the embedded bright marker supplies the ray color.
    """
    g = np.asarray(grid, dtype=int)
    sprites = _extract_sprites(g)
    assert sprites, "expected at least one composite sprite"
    return _paint_rays(g, sprites)


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _stamp_sprite(grid: np.ndarray, top: int, left: int, direction: tuple[int, int], color: int) -> dict:
    gap_r, gap_c = GAP_BY_DIRECTION[direction]
    sprite = {
        "bbox": (top, left, top + 2, left + 2),
        "origin": (top + 1, left + 1),
        "direction": direction,
        "color": color,
    }
    for rr in range(3):
        for cc in range(3):
            r = top + rr
            c = left + cc
            if (rr, cc) == (1, 1):
                grid[r, c] = color
            elif (rr, cc) == (gap_r, gap_c):
                grid[r, c] = BACKGROUND
            else:
                grid[r, c] = BODY
    return sprite


def _spec_bbox_cells(spec: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = spec["bbox"]
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _choose_directions(count: int) -> list[tuple[int, int]]:
    need_duplicate = random.random() < 0.6
    for _ in range(200):
        directions = [random.choice(CARDINALS) for _ in range(count)]
        if len(set(directions)) < 2:
            continue
        if need_duplicate and len(set(directions)) == count:
            continue
        return directions
    raise RuntimeError("failed to choose directions")


def _assign_colors(directions: list[tuple[int, int]]) -> list[int]:
    need_same_dir_diff_color = len(set(directions)) < len(directions)
    for _ in range(200):
        colors = [random.choice(RAY_COLORS) for _ in directions]
        if len(set(colors)) < 2:
            continue
        same_color_diff_dir = any(
            colors[i] == colors[j] and directions[i] != directions[j]
            for i in range(len(directions))
            for j in range(i + 1, len(directions))
        )
        if not same_color_diff_dir:
            continue
        if need_same_dir_diff_color:
            same_dir_diff_color = any(
                directions[i] == directions[j] and colors[i] != colors[j]
                for i in range(len(directions))
                for j in range(i + 1, len(directions))
            )
            if not same_dir_diff_color:
                continue
        return colors
    raise RuntimeError("failed to assign colors")


def _try_place_sprites(height: int, width: int, directions: list[tuple[int, int]]) -> list[dict] | None:
    specs: list[dict] = []
    for direction in directions:
        candidates = [(r, c) for r in range(height - 2) for c in range(width - 2)]
        random.shuffle(candidates)
        placed = None
        for top, left in candidates:
            bbox = (top, left, top + 2, left + 2)
            origin = (top + 1, left + 1)
            path_list = _ray_cells((height, width), origin, direction)
            path = set(path_list)
            paint_cells = set(path_list[::2])
            if len(path_list) < 3:
                continue
            candidate = {
                "bbox": bbox,
                "origin": origin,
                "direction": direction,
                "color": RAY_COLORS[0],
            }
            bbox_cells = _spec_bbox_cells(candidate)
            blocked = False
            for other in specs:
                other_bbox_cells = _spec_bbox_cells(other)
                other_path = set(_ray_cells((height, width), other["origin"], other["direction"]))
                other_paints = set(_ray_cells((height, width), other["origin"], other["direction"])[::2])
                if _rects_overlap(bbox, other["bbox"], pad=1):
                    blocked = True
                    break
                if path & other_path:
                    blocked = True
                    break
                if path & other_bbox_cells:
                    blocked = True
                    break
                if bbox_cells & other_path:
                    blocked = True
                    break
                if paint_cells & other_paints:
                    blocked = True
                    break
            if not blocked:
                placed = candidate
                break
        if placed is None:
            return None
        specs.append(placed)
    return specs


def _oracle_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    return _paint_rays(grid, specs)


def _validate_case(grid: np.ndarray, specs: list[dict]) -> bool:
    if not (grid.ndim == 2 and max(grid.shape) <= 30):
        return False
    if len(specs) not in (3, 4):
        return False
    extracted = _extract_sprites(grid)
    if len(extracted) != len(specs):
        return False
    extracted_sorted = sorted(
        ((s["origin"], s["direction"], s["color"]) for s in extracted),
        key=lambda item: (item[0], item[1], item[2]),
    )
    spec_sorted = sorted(
        ((s["origin"], s["direction"], s["color"]) for s in specs),
        key=lambda item: (item[0], item[1], item[2]),
    )
    if extracted_sorted != spec_sorted:
        return False
    directions = [s["direction"] for s in specs]
    colors = [s["color"] for s in specs]
    if len(set(directions)) < 2 or len(set(colors)) < 2:
        return False
    if not any(
        colors[i] == colors[j] and directions[i] != directions[j]
        for i in range(len(specs))
        for j in range(i + 1, len(specs))
    ):
        return False
    if len(set(directions)) < len(directions):
        if not any(
            directions[i] == directions[j] and colors[i] != colors[j]
            for i in range(len(specs))
            for j in range(i + 1, len(specs))
        ):
            return False
    expected = _oracle_from_specs(grid, specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False
    return True


def _build_attempt() -> tuple[np.ndarray, list[dict]] | None:
    height = random.randint(13, 18)
    width = random.randint(13, 18)
    count = 4 if random.random() < 0.65 else 3
    directions = _choose_directions(count)
    specs = _try_place_sprites(height, width, directions)
    if specs is None:
        return None
    colors = _assign_colors(directions)
    grid = np.zeros((height, width), dtype=int)
    final_specs: list[dict] = []
    for base_spec, color in zip(specs, colors):
        final_specs.append(_stamp_sprite(grid, base_spec["bbox"][0], base_spec["bbox"][1], base_spec["direction"], color))
    if not _validate_case(grid, final_specs):
        return None
    return grid, final_specs


def _run_semantic_smoke_tests() -> None:
    global _SMOKE_TESTED
    if _SMOKE_TESTED:
        return

    # Direction decoding for all four templates.
    for direction, color in zip(CARDINALS, (1, 2, 3, 1)):
        grid = np.zeros((7, 7), dtype=int)
        spec = _stamp_sprite(grid, 2, 2, direction, color)
        extracted = _extract_sprites(grid)
        assert len(extracted) == 1
        assert extracted[0]["direction"] == direction
        assert extracted[0]["color"] == color
        assert np.array_equal(main(grid), _oracle_from_specs(grid, [spec]))

    # Skip-over-body behavior: the left sprite's ray passes through the right sprite's body.
    grid = np.zeros((9, 10), dtype=int)
    left = _stamp_sprite(grid, 3, 1, (0, 1), 2)
    right = _stamp_sprite(grid, 3, 5, (-1, 0), 3)
    expected = _oracle_from_specs(grid, [left, right])
    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert actual[4, 5] == BODY and actual[4, 7] == BODY
    assert actual[4, 3] == 2 and actual[4, 9] == 2

    _SMOKE_TESTED = True


def generate_input() -> np.ndarray:
    _run_semantic_smoke_tests()
    for _ in range(400):
        attempt = _build_attempt()
        if attempt is None:
            continue
        grid, _ = attempt
        return grid
    raise RuntimeError("failed to generate a valid input grid")
