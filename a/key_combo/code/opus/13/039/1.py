from __future__ import annotations

import itertools
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MAROON = 9

BODY_PALETTE = (BLUE, RED, GREEN)
MARKER_PALETTE = (YELLOW, ORANGE, MAROON)
DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_cells(grid: np.ndarray, coords: np.ndarray) -> list[tuple[int, int, int]]:
    return [(int(r), int(c), int(grid[r, c])) for r, c in coords]


def _parse_rectangles(grid: np.ndarray) -> list[dict]:
    rects = []
    for coords in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 4 and w >= 4, "gray rectangles must be at least 4x4"
        assert int(coords.shape[0]) == h * w, "gray component must be a solid rectangle"
        rects.append({"bbox": (r0, c0, r1, c1), "coords": coords})
    assert 2 <= len(rects) <= 3, "expected 2-3 gray rectangles"
    return rects


def _parse_sprites(grid: np.ndarray) -> list[dict]:
    mask = (grid != BLACK) & (grid != GREY)
    sprites = []
    for coords in _components(mask):
        cells = _component_cells(grid, coords)
        colors = Counter(color for _, _, color in cells)
        assert len(colors) == 2, "sprite must contain exactly two colors"
        body_color, body_count = colors.most_common(1)[0]
        marker_candidates = [color for color, count in colors.items() if count == 1 and color != body_color]
        assert len(marker_candidates) == 1, "sprite must contain exactly one marker pixel"
        marker_color = marker_candidates[0]
        assert body_count >= 2, "sprite body must contain at least two pixels"
        marker_coord = next((r, c) for r, c, color in cells if color == marker_color)
        box = _bbox(coords)
        sprites.append(
            {
                "coords": coords,
                "bbox": box,
                "body_color": int(body_color),
                "marker_color": int(marker_color),
                "marker_coord": (int(marker_coord[0]), int(marker_coord[1])),
            }
        )
    assert 2 <= len(sprites) <= 3, "expected 2-3 sprites"
    sprites.sort(key=lambda spr: (spr["bbox"][1], spr["bbox"][0]))
    return sprites


def _direction_from_sprite(sprite: dict) -> str:
    r0, c0, r1, c1 = sprite["bbox"]
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    mr, mc = sprite["marker_coord"]
    dr = mr - center_r
    dc = mc - center_c
    assert dr != 0 or dc != 0, "marker cannot be at bbox center"
    if abs(dr) > abs(dc):
        return "up" if dr < 0 else "down"
    return "left" if dc < 0 else "right"


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    delta = np.array([dr, dc], dtype=int)
    return coords + delta


def _recolor_rectangle(out: np.ndarray, rect: tuple[int, int, int, int], body_color: int, marker_color: int) -> None:
    r0, c0, r1, c1 = rect
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r in (r0, r1)) and (c in (c0, c1)):
                out[r, c] = marker_color
            elif r in (r0, r1) or c in (c0, c1):
                out[r, c] = body_color
            else:
                out[r, c] = GREY


def _slide_to_target(
    coords: np.ndarray, direction: str, rect_id_grid: np.ndarray
) -> tuple[np.ndarray, int]:
    h, w = rect_id_grid.shape
    dr, dc = DIRECTIONS[direction]
    cur = coords.copy()
    for _ in range(h * w):
        nxt = _translate(cur, dr, dc)
        if np.any(nxt[:, 0] < 0) or np.any(nxt[:, 0] >= h) or np.any(nxt[:, 1] < 0) or np.any(nxt[:, 1] >= w):
            raise AssertionError("sprite must collide with a rectangle before leaving the grid")
        hit_ids = {int(rect_id_grid[r, c]) for r, c in nxt if int(rect_id_grid[r, c]) > 0}
        if hit_ids:
            assert len(hit_ids) == 1, "a sprite cannot hit multiple rectangles at once"
            return cur, next(iter(hit_ids)) - 1
        cur = nxt
    raise AssertionError("sprite did not reach a rectangle")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each marked sprite toward its encoded direction and recolor the struck gray rectangle with the sprite's two colors.

    Input structure:
    - The grid has black background, 2-3 solid gray rectangles, and 2-3 small 4-connected two-color sprites.
    - Each sprite has one repeated body color and exactly one differently colored marker pixel.
    - The marker lies on one side of the sprite bbox center and therefore encodes up/down/left/right.
    - Different sprites are separated and each sliding path reaches exactly one gray rectangle.

    Rule:
    - For each sprite, read the marker position relative to the sprite bbox center to get the sliding direction.
    - Translate the whole sprite step by step in that direction until the next step would overlap a gray rectangle; stop one step earlier.
    - The touched rectangle becomes that sprite's target rectangle.
    - Recolor the target rectangle so its four corners use the marker color, its non-corner border uses the body color, and its strict interior stays gray.
    - Erase every sprite from the output; untouched gray rectangles remain unchanged.

    Key insight:
    - The same marker pixel serves two roles at once: its position gives the motion direction, and its color supplies the corner color for the struck rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    rects = _parse_rectangles(g)
    sprites = _parse_sprites(g)

    rect_id_grid = np.zeros_like(g, dtype=int)
    for idx, rect in enumerate(rects, start=1):
        coords = rect["coords"]
        rect_id_grid[coords[:, 0], coords[:, 1]] = idx

    out = g.copy()
    for sprite in sprites:
        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = BLACK

    hit_rects = set()
    for sprite in sprites:
        direction = _direction_from_sprite(sprite)
        _final_coords, rect_idx = _slide_to_target(sprite["coords"], direction, rect_id_grid)
        assert rect_idx not in hit_rects, "different sprites must hit different rectangles"
        hit_rects.add(rect_idx)
        _recolor_rectangle(out, rects[rect_idx]["bbox"], sprite["body_color"], sprite["marker_color"])

    return out


SPRITE_TEMPLATES = {
    "up": (
        ((0, 1, "marker"), (1, 1, "body"), (2, 1, "body"), (1, 0, "body"), (1, 2, "body")),
        ((0, 1, "marker"), (1, 1, "body"), (2, 1, "body"), (1, 0, "body"), (1, 2, "body"), (2, 0, "body")),
        ((0, 1, "marker"), (1, 1, "body"), (2, 1, "body"), (1, 0, "body"), (1, 2, "body"), (2, 2, "body")),
    ),
    "down": (
        ((2, 1, "marker"), (1, 1, "body"), (0, 1, "body"), (1, 0, "body"), (1, 2, "body")),
        ((2, 1, "marker"), (1, 1, "body"), (0, 1, "body"), (1, 0, "body"), (1, 2, "body"), (0, 0, "body")),
        ((2, 1, "marker"), (1, 1, "body"), (0, 1, "body"), (1, 0, "body"), (1, 2, "body"), (0, 2, "body")),
    ),
    "left": (
        ((1, 0, "marker"), (1, 1, "body"), (1, 2, "body"), (0, 1, "body"), (2, 1, "body")),
        ((1, 0, "marker"), (1, 1, "body"), (1, 2, "body"), (0, 1, "body"), (2, 1, "body"), (0, 2, "body")),
        ((1, 0, "marker"), (1, 1, "body"), (1, 2, "body"), (0, 1, "body"), (2, 1, "body"), (2, 2, "body")),
    ),
    "right": (
        ((1, 2, "marker"), (1, 1, "body"), (1, 0, "body"), (0, 1, "body"), (2, 1, "body")),
        ((1, 2, "marker"), (1, 1, "body"), (1, 0, "body"), (0, 1, "body"), (2, 1, "body"), (0, 0, "body")),
        ((1, 2, "marker"), (1, 1, "body"), (1, 0, "body"), (0, 1, "body"), (2, 1, "body"), (2, 0, "body")),
    ),
}


def _rectangles_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 2) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_rectangles(rng: np.random.Generator, h: int, w: int, count: int) -> list[tuple[int, int, int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []
    attempts = 0
    while len(rects) < count and attempts < 300:
        attempts += 1
        rh = int(rng.integers(4, 7))
        rw = int(rng.integers(4, 8))
        r0 = int(rng.integers(2, h - rh - 1))
        c0 = int(rng.integers(2, w - rw - 1))
        rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
        if any(not _rectangles_separated(rect, other) for other in rects):
            continue
        rects.append(rect)
    if len(rects) != count:
        return None
    return rects


def _rect_id_grid(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for idx, (r0, c0, r1, c1) in enumerate(rects, start=1):
        grid[r0 : r1 + 1, c0 : c1 + 1] = idx
    return grid


def _render_input(
    shape: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
    candidates: list[dict],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    for cand in candidates:
        for r, c, color in cand["input_cells"]:
            grid[r, c] = color
    return grid


def _render_expected(
    shape: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
    assignments: list[dict],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for rect in rects:
        r0, c0, r1, c1 = rect
        out[r0 : r1 + 1, c0 : c1 + 1] = GREY
    for info in assignments:
        _recolor_rectangle(out, rects[info["rect_idx"]], info["body_color"], info["marker_color"])
    return out


def _candidate_positions(
    rect: tuple[int, int, int, int],
    direction: str,
    h: int,
    w: int,
) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = rect
    positions = []
    if direction == "up":
        final_r0 = r1 + 1
        if final_r0 + 2 >= h:
            return []
        for final_c0 in range(max(0, c0 - 1), min(w - 3, c1 - 1) + 1):
            max_gap = h - 3 - final_r0
            for gap in range(1, max_gap + 1):
                positions.append((final_r0 + gap, final_c0, gap))
    elif direction == "down":
        final_r0 = r0 - 3
        if final_r0 < 0:
            return []
        for final_c0 in range(max(0, c0 - 1), min(w - 3, c1 - 1) + 1):
            max_gap = final_r0
            for gap in range(1, max_gap + 1):
                positions.append((final_r0 - gap, final_c0, gap))
    elif direction == "left":
        final_c0 = c1 + 1
        if final_c0 + 2 >= w:
            return []
        for final_r0 in range(max(0, r0 - 1), min(h - 3, r1 - 1) + 1):
            max_gap = w - 3 - final_c0
            for gap in range(1, max_gap + 1):
                positions.append((final_r0, final_c0 + gap, gap))
    else:
        final_c0 = c0 - 3
        if final_c0 < 0:
            return []
        for final_r0 in range(max(0, r0 - 1), min(h - 3, r1 - 1) + 1):
            max_gap = final_c0
            for gap in range(1, max_gap + 1):
                positions.append((final_r0, final_c0 - gap, gap))
    return positions


def _instantiate_template(
    template: tuple[tuple[int, int, str], ...],
    top: int,
    left: int,
    body_color: int,
    marker_color: int,
) -> list[tuple[int, int, int]]:
    cells = []
    for dr, dc, role in template:
        color = marker_color if role == "marker" else body_color
        cells.append((top + dr, left + dc, color))
    return cells


def _candidate_from_template(
    rects: list[tuple[int, int, int, int]],
    rect_id_grid: np.ndarray,
    rect_idx: int,
    direction: str,
    template: tuple[tuple[int, int, str], ...],
    start_r0: int,
    start_c0: int,
    body_color: int,
    marker_color: int,
) -> dict | None:
    h, w = rect_id_grid.shape
    input_cells = _instantiate_template(template, start_r0, start_c0, body_color, marker_color)
    input_coords = np.array([(r, c) for r, c, _ in input_cells], dtype=int)
    if np.any(input_coords[:, 0] < 0) or np.any(input_coords[:, 0] >= h) or np.any(input_coords[:, 1] < 0) or np.any(input_coords[:, 1] >= w):
        return None
    if any(rect_id_grid[r, c] > 0 for r, c in input_coords):
        return None

    final_coords, hit_idx = _slide_to_target(input_coords, direction, rect_id_grid)
    if hit_idx != rect_idx:
        return None

    dr, dc = DIRECTIONS[direction]
    next_coords = _translate(final_coords, dr, dc)
    hit_ids = {int(rect_id_grid[r, c]) for r, c in next_coords if int(rect_id_grid[r, c]) > 0}
    if hit_ids != {rect_idx + 1}:
        return None

    sweep_cells = set()
    cur = input_coords.copy()
    for _ in range(h * w):
        for r, c in cur:
            sweep_cells.add((int(r), int(c)))
        if np.array_equal(cur, final_coords):
            break
        cur = _translate(cur, dr, dc)
    else:
        return None

    return {
        "rect_idx": rect_idx,
        "direction": direction,
        "body_color": body_color,
        "marker_color": marker_color,
        "input_cells": input_cells,
        "input_coords": input_coords,
        "sweep_cells": sweep_cells,
        "bbox": (start_r0, start_c0, start_r0 + 2, start_c0 + 2),
    }


def _enumerate_candidates(
    rects: list[tuple[int, int, int, int]],
    rect_idx: int,
    body_color: int,
    marker_color: int,
    rect_id_grid: np.ndarray,
    rng: np.random.Generator,
) -> list[dict]:
    h, w = rect_id_grid.shape
    rect = rects[rect_idx]
    candidates = []
    directions = list(DIRECTIONS)
    rng.shuffle(directions)
    for direction in directions:
        starts = _candidate_positions(rect, direction, h, w)
        rng.shuffle(starts)
        templates = list(SPRITE_TEMPLATES[direction])
        rng.shuffle(templates)
        for start_r0, start_c0, _gap in starts:
            for template in templates:
                cand = _candidate_from_template(
                    rects, rect_id_grid, rect_idx, direction, template, start_r0, start_c0, body_color, marker_color
                )
                if cand is not None:
                    candidates.append(cand)
    return candidates


def _search_assignments(
    targets: list[int],
    rects: list[tuple[int, int, int, int]],
    colors: list[tuple[int, int]],
    rect_id_grid: np.ndarray,
    rng: np.random.Generator,
) -> list[dict] | None:
    target_info = {rect_idx: colors[i] for i, rect_idx in enumerate(targets)}

    def buffered(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
        out = set(cells)
        for r, c in cells:
            out.update(_neighbors4(r, c))
        return out

    def backtrack(
        remaining: list[int],
        chosen: list[dict],
        blocked_initial: set[tuple[int, int]],
        blocked_initial_buffer: set[tuple[int, int]],
        blocked_path: set[tuple[int, int]],
    ) -> list[dict] | None:
        if not remaining:
            return list(chosen)

        best_rect = None
        best_candidates = None
        for rect_idx in remaining:
            body_color, marker_color = target_info[rect_idx]
            candidates = _enumerate_candidates(rects, rect_idx, body_color, marker_color, rect_id_grid, rng)
            filtered = []
            for cand in candidates:
                initial_cells = {(r, c) for r, c, _ in cand["input_cells"]}
                if initial_cells & blocked_initial_buffer:
                    continue
                if cand["sweep_cells"] & blocked_initial:
                    continue
                if cand["sweep_cells"] & blocked_path:
                    continue
                filtered.append(cand)
            if not filtered:
                return None
            if best_candidates is None or len(filtered) < len(best_candidates):
                best_rect = rect_idx
                best_candidates = filtered

        assert best_rect is not None and best_candidates is not None
        rng.shuffle(best_candidates)
        next_remaining = [idx for idx in remaining if idx != best_rect]
        for cand in best_candidates:
            initial_cells = {(r, c) for r, c, _ in cand["input_cells"]}
            result = backtrack(
                next_remaining,
                chosen + [cand],
                blocked_initial | initial_cells,
                blocked_initial_buffer | buffered(initial_cells),
                blocked_path | cand["sweep_cells"],
            )
            if result is not None:
                return result
        return None

    return backtrack(list(targets), [], set(), set(), set())


def _semantic_check(
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    assignments: list[dict],
    expected: np.ndarray,
) -> bool:
    rect_set = {assignment["rect_idx"] for assignment in assignments}
    if len(rect_set) != len(assignments):
        return False

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    untouched = set(range(len(rects))) - rect_set
    for idx, rect in enumerate(rects):
        r0, c0, r1, c1 = rect
        patch = actual[r0 : r1 + 1, c0 : c1 + 1]
        if idx in untouched:
            if not np.all(patch == GREY):
                return False
        else:
            assignment = next(item for item in assignments if item["rect_idx"] == idx)
            if int(patch[0, 0]) != assignment["marker_color"]:
                return False
            if int(patch[0, 1]) != assignment["body_color"]:
                return False
            if int(patch[1, 1]) != GREY:
                return False

    if np.any((actual != BLACK) & (actual != GREY) & (actual != BLUE) & (actual != RED) & (actual != GREEN) & (actual != YELLOW) & (actual != ORANGE) & (actual != MAROON)):
        return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rect_count = int(rng.choice([2, 3], p=[0.45, 0.55]))
    sprite_count = rect_count if rect_count == 2 else int(rng.choice([2, 3], p=[0.45, 0.55]))

    h = int(rng.integers(18 if rect_count == 3 else 16, 23))
    w = int(rng.integers(18 if rect_count == 3 else 16, 23))

    rects = _place_rectangles(rng, h, w, rect_count)
    if rects is None:
        return None

    rect_id_grid = _rect_id_grid((h, w), rects)
    targets = list(rng.choice(rect_count, size=sprite_count, replace=False))
    body_colors = list(rng.permutation(BODY_PALETTE)[:sprite_count])
    marker_colors = list(rng.permutation(MARKER_PALETTE)[:sprite_count])
    colors = list(zip(body_colors, marker_colors, strict=False))

    assignments = _search_assignments(targets, rects, [(b, m) for b, m in colors], rect_id_grid, rng)
    if assignments is None:
        return None

    grid = _render_input((h, w), rects, assignments)
    expected = _render_expected(
        (h, w),
        rects,
        [
            {
                "rect_idx": cand["rect_idx"],
                "body_color": cand["body_color"],
                "marker_color": cand["marker_color"],
            }
            for cand in assignments
        ],
    )

    if not _semantic_check(grid, rects, assignments, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
