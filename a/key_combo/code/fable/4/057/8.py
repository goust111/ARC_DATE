from __future__ import annotations

import random
from collections import Counter

import numpy as np


PALETTE = list(range(10))
MAX_GENERATION_ATTEMPTS = 400
GRID_SIDE_RANGE = (18, 27)
SPRITE_COUNT_RANGE = (2, 4)
SPRITE_BBOX_SIDE_RANGE = (2, 5)
SPRITE_AREA_RANGE = (4, 8)
SPRITE_GAP = 1
DISTRACTOR_COUNT_RANGE = (1, 4)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _sprite_from_component(grid: np.ndarray, bg: int, coords: list[tuple[int, int]]) -> dict[str, object]:
    colors = Counter(int(grid[r, c]) for r, c in coords)
    assert len(coords) > 1, "sprite must contain more than one cell"
    assert len(colors) == 2, "each sprite must contain exactly one body color and one minority color"

    minority_items = [(color, count) for color, count in colors.items() if count == 1]
    assert len(minority_items) == 1, "sprite must contain exactly one minority pixel"
    minority_color = int(minority_items[0][0])
    body_items = [(color, count) for color, count in colors.items() if color != minority_color]
    assert len(body_items) == 1, "sprite must contain exactly one body color"
    body_color, body_count = int(body_items[0][0]), int(body_items[0][1])
    assert body_count == len(coords) - 1, "all non-minority sprite cells must have the body color"

    minority_cells = [(r, c) for r, c in coords if int(grid[r, c]) == minority_color]
    assert len(minority_cells) == 1, "minority color must occupy one cell"
    minority_r, minority_c = minority_cells[0]

    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])

    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "patch": patch,
        "body_color": body_color,
        "minority_color": minority_color,
        "minority_abs": (minority_r, minority_c),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each sprite so its unique minority pixel lands on the same-colored body target.

    Input structure:
    - The grid has one dominant background color.
    - Foreground components are either multicolor 4-connected sprites or isolated single pixels.
    - Each sprite has a unique body color, exactly one minority-colored cell, and exactly one isolated target pixel whose color matches the sprite body color.

    Rule:
    - Keep every isolated distractor point in place.
    - Erase every sprite from its original location.
    - For each sprite, translate the full sprite without rotation or reflection so that its minority cell lands on the isolated single pixel whose color matches that sprite's body color.

    Key insight:
    - Color identifies which target belongs to each sprite, but the unique minority pixel gives the alignment offset needed for the translation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert components, "expected foreground content"

    sprites: list[dict[str, object]] = []
    singletons: list[tuple[int, int, int]] = []
    for coords in components:
        if len(coords) == 1:
            r, c = coords[0]
            singletons.append((r, c, int(g[r, c])))
        else:
            sprites.append(_sprite_from_component(g, bg, coords))

    assert sprites, "expected at least one sprite"
    body_colors = [int(sprite["body_color"]) for sprite in sprites]
    assert len(body_colors) == len(set(body_colors)), "sprite body colors must be unique"

    targets_by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, color in singletons:
        targets_by_color.setdefault(int(color), []).append((int(r), int(c)))

    out = g.copy()
    for sprite in sprites:
        for r, c in sprite["coords"]:
            out[r, c] = bg

    for sprite in sorted(sprites, key=lambda item: tuple(item["bbox"])):
        body_color = int(sprite["body_color"])
        assert body_color in targets_by_color, "each sprite body color must have a matching target point"
        assert len(targets_by_color[body_color]) == 1, "each body color must map to exactly one isolated target"
        target_r, target_c = targets_by_color[body_color][0]

        minority_r, minority_c = sprite["minority_abs"]
        shift_r = int(target_r - minority_r)
        shift_c = int(target_c - minority_c)
        patch = np.asarray(sprite["patch"], dtype=int)
        patch_rows, patch_cols = np.where(patch != bg)

        for pr, pc in zip(patch_rows.tolist(), patch_cols.tolist()):
            rr = int(sprite["bbox"][0]) + pr + shift_r
            cc = int(sprite["bbox"][1]) + pc + shift_c
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], "translated sprite must stay in bounds"

            current = int(out[rr, cc])
            allowed_target = (rr, cc) == (target_r, target_c) and current == body_color
            assert current == bg or allowed_target, "translated sprites must not overlap each other or distractors"
            out[rr, cc] = int(patch[pr, pc])

    return out


def _rect_separated(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _point_clear_of_rect(point: tuple[int, int], rect: tuple[int, int, int, int], gap: int) -> bool:
    r, c = point
    r0, c0, r1, c1 = rect
    return not (r0 - gap <= r <= r1 + gap and c0 - gap <= c <= c1 + gap)


def _pick_shape() -> dict[str, object]:
    for _ in range(100):
        h = random.randint(*SPRITE_BBOX_SIDE_RANGE)
        w = random.randint(*SPRITE_BBOX_SIDE_RANGE)
        area = random.randint(SPRITE_AREA_RANGE[0], min(SPRITE_AREA_RANGE[1], h * w))

        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = {
                (nr, nc)
                for r, c in cells
                for nr, nc in _neighbors4(r, c)
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells
            }
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, c0 = min(rows), min(cols)
        cropped = sorted((r - r0, c - c0) for r, c in cells)
        ch = max(r for r, _ in cropped) + 1
        cw = max(c for _, c in cropped) + 1
        if ch == 1 or cw == 1:
            continue

        corners = {(0, 0), (0, cw - 1), (ch - 1, 0), (ch - 1, cw - 1)}
        center = {(ch // 2, cw // 2)} if ch % 2 == 1 and cw % 2 == 1 else set()
        preferred = [cell for cell in cropped if cell not in corners and cell not in center]
        minority = random.choice(preferred or cropped)

        return {"cells": cropped, "h": ch, "w": cw, "area": len(cropped), "minority": minority}
    raise RuntimeError("failed to sample a sprite shape")


def _place_rects(
    specs: list[dict[str, object]],
    height: int,
    width: int,
    *,
    forbidden_points: list[tuple[int, int]] | None = None,
    disallow_positions: dict[int, tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    order = list(range(len(specs)))
    random.shuffle(order)
    order.sort(key=lambda i: (int(specs[i]["h"]) * int(specs[i]["w"]), int(specs[i]["area"])), reverse=True)

    placements: list[tuple[int, int] | None] = [None] * len(specs)
    used_rects: list[tuple[int, int, int, int]] = []
    forbidden_points = forbidden_points or []
    disallow_positions = disallow_positions or {}

    for idx in order:
        spec = specs[idx]
        sh = int(spec["h"])
        sw = int(spec["w"])
        candidates = [
            (r0, c0)
            for r0 in range(height - sh + 1)
            for c0 in range(width - sw + 1)
            if all(_rect_separated((r0, c0, r0 + sh - 1, c0 + sw - 1), rect, SPRITE_GAP) for rect in used_rects)
            and all(_point_clear_of_rect(point, (r0, c0, r0 + sh - 1, c0 + sw - 1), 1) for point in forbidden_points)
            and disallow_positions.get(idx) != (r0, c0)
        ]
        if not candidates:
            return None
        choice = random.choice(candidates)
        placements[idx] = choice
        used_rects.append((choice[0], choice[1], choice[0] + sh - 1, choice[1] + sw - 1))

    return [(r0, c0) for r0, c0 in placements if (r0, c0) is not None]


def _render_sprite_patch(spec: dict[str, object], bg: int, body_color: int, minority_color: int) -> np.ndarray:
    patch = np.full((int(spec["h"]), int(spec["w"])), bg, dtype=int)
    minority = tuple(spec["minority"])
    for r, c in spec["cells"]:
        patch[r, c] = minority_color if (r, c) == minority else body_color
    return patch


def _choose_distractor_points(
    height: int,
    width: int,
    rects: list[tuple[int, int, int, int]],
    targets: list[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]]:
    candidates = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(candidates)
    points: list[tuple[int, int]] = []
    for point in candidates:
        if any(not _point_clear_of_rect(point, rect, 1) for rect in rects):
            continue
        if any(max(abs(point[0] - r), abs(point[1] - c)) <= 1 for r, c in targets):
            continue
        if any(max(abs(point[0] - r), abs(point[1] - c)) <= 1 for r, c in points):
            continue
        points.append(point)
        if len(points) == count:
            return points
    return []


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.ravel()))
    bg_count = counts.pop(int(bg), 0)
    return bg_count > max(counts.values(), default=0)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    bg = random.choice(PALETTE)

    n_sprites = random.randint(*SPRITE_COUNT_RANGE)
    body_colors = random.sample([color for color in PALETTE if color != bg], n_sprites)
    minority_pool = [color for color in PALETTE if color != bg and color not in body_colors]
    if not minority_pool:
        return None

    specs = [_pick_shape() for _ in range(n_sprites)]
    for spec, body_color in zip(specs, body_colors):
        spec["body_color"] = int(body_color)
        spec["minority_color"] = int(random.choice(minority_pool))

    final_positions = _place_rects(specs, height, width)
    if final_positions is None:
        return None

    targets = [
        (r0 + int(spec["minority"][0]), c0 + int(spec["minority"][1]))
        for spec, (r0, c0) in zip(specs, final_positions)
    ]
    input_positions = _place_rects(
        specs,
        height,
        width,
        forbidden_points=targets,
        disallow_positions={i: pos for i, pos in enumerate(final_positions)},
    )
    if input_positions is None:
        return None

    rects = []
    for spec, (r0, c0) in zip(specs, input_positions):
        rects.append((r0, c0, r0 + int(spec["h"]) - 1, c0 + int(spec["w"]) - 1))
    for spec, (r0, c0) in zip(specs, final_positions):
        rects.append((r0, c0, r0 + int(spec["h"]) - 1, c0 + int(spec["w"]) - 1))

    distractor_count = random.randint(*DISTRACTOR_COUNT_RANGE)
    distractor_points = _choose_distractor_points(height, width, rects, targets, distractor_count)
    if len(distractor_points) != distractor_count:
        return None

    distractor_colors = [int(random.choice(minority_pool)) for _ in distractor_points]

    inp = np.full((height, width), bg, dtype=int)
    out = np.full_like(inp, bg)

    for spec, (r0, c0) in zip(specs, input_positions):
        patch = _render_sprite_patch(spec, bg, int(spec["body_color"]), int(spec["minority_color"]))
        patch_rows, patch_cols = np.where(patch != bg)
        inp[patch_rows + r0, patch_cols + c0] = patch[patch_rows, patch_cols]

    for spec, (r0, c0) in zip(specs, final_positions):
        patch = _render_sprite_patch(spec, bg, int(spec["body_color"]), int(spec["minority_color"]))
        patch_rows, patch_cols = np.where(patch != bg)
        out[patch_rows + r0, patch_cols + c0] = patch[patch_rows, patch_cols]

    for spec, target in zip(specs, targets):
        inp[target] = int(spec["body_color"])

    for point, color in zip(distractor_points, distractor_colors):
        inp[point] = color
        out[point] = color

    if not _bg_is_dominant(inp, bg):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
