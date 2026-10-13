from __future__ import annotations

from collections import deque
import itertools
from typing import Iterable

import numpy as np


BG = 0
PLACEHOLDER = 1
TEMPLATE_COLORS = (2, 3, 4, 7)
GUIDE = 6

LINE_CHOICES = ((1, 1), (2, 1), (1, 2))
GRID_H_RANGE = (17, 24)
GRID_W_RANGE = (17, 24)
PAIR_COUNT_CHOICES = (2, 3, 3, 4)
GENERATION_RETRIES = 500


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _find_components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == int(color):
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _normalized_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    mins = coords.min(axis=0)
    shifted = coords - mins
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _guide_rows_cols(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    rows = [r for r in range(1, h - 1) if int(grid[r, 0]) == GUIDE and int(grid[r, w - 1]) == GUIDE]
    cols = [c for c in range(1, w - 1) if int(grid[0, c]) == GUIDE and int(grid[h - 1, c]) == GUIDE]
    return rows, cols


def _guide_masks(shape: tuple[int, int], rows: list[int], cols: list[int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    guide = np.zeros((h, w), dtype=bool)
    for r in rows:
        guide[r, :] = True
    for c in cols:
        guide[:, c] = True

    touched = guide.copy()
    touched[1:, :] |= guide[:-1, :]
    touched[:-1, :] |= guide[1:, :]
    touched[:, 1:] |= guide[:, :-1]
    touched[:, :-1] |= guide[:, 1:]
    return guide, touched


def _parse_scene(grid: np.ndarray) -> dict:
    rows, cols = _guide_rows_cols(grid)
    assert 2 <= len(rows) + len(cols) <= 3
    assert rows and cols
    guide_mask, touch_mask = _guide_masks(grid.shape, rows, cols)

    templates: dict[tuple[tuple[int, int], ...], int] = {}
    template_components = []
    for color in TEMPLATE_COLORS:
        comps = _find_components_of_color(grid, color)
        assert len(comps) <= 1
        if comps:
            shape = _normalized_shape(comps[0])
            assert shape not in templates
            templates[shape] = color
            template_components.append({"color": color, "coords": comps[0], "shape": shape})

    placeholders = []
    for coords in _find_components_of_color(grid, PLACEHOLDER):
        shape = _normalized_shape(coords)
        assert shape in templates
        placeholders.append(
            {
                "coords": coords,
                "shape": shape,
                "match_color": templates[shape],
                "touched": bool(touch_mask[coords[:, 0], coords[:, 1]].any()),
                "overlap": bool(guide_mask[coords[:, 0], coords[:, 1]].any()),
            }
        )

    assert template_components
    assert placeholders
    return {
        "rows": rows,
        "cols": cols,
        "guide_mask": guide_mask,
        "touch_mask": touch_mask,
        "templates": template_components,
        "placeholders": placeholders,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Draw border-defined guide lines, then recolor only the touched blue placeholders by shape-matched template color.

    Input structure:
    - A black background grid contains blue 4-connected placeholder objects and non-blue template objects.
    - Template colors are red, green, yellow, or orange, and each template shape is unique.
    - Magenta singleton markers appear on matching opposite borders to define horizontal and vertical guide lines.

    Rule:
    - Recover every guide row from a left/right magenta border pair and every guide column from a top/bottom pair.
    - Paint each full guide row or column magenta across the canvas.
    - Using the original pre-line blue components, mark a placeholder as touched when any of its cells overlaps a guide line or is 4-neighbor adjacent to one.
    - Recolor only the touched placeholders to the color of the uniquely shape-matching template; untouched placeholders stay blue, while guide pixels remain visible except where a touched placeholder is repainted on top.

    Key insight:
    - Guide-line contact is the gate, while normalized-shape matching only supplies the replacement color for the gated placeholders.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    scene = _parse_scene(g)
    out = g.copy()
    out[scene["guide_mask"]] = GUIDE

    for placeholder in scene["placeholders"]:
        if not placeholder["touched"]:
            continue
        coords = placeholder["coords"]
        out[coords[:, 0], coords[:, 1]] = int(placeholder["match_color"])
    return out


def _shape_library() -> list[tuple[tuple[int, int], ...]]:
    cells = [(r, c) for r in range(3) for c in range(3)]
    seen = set()
    shapes = []
    for size in range(3, 6):
        for combo in itertools.combinations(cells, size):
            coords = np.array(combo, dtype=int)
            mask = np.zeros((3, 3), dtype=int)
            mask[coords[:, 0], coords[:, 1]] = 1
            if len(_find_components_of_color(mask, 1)) != 1:
                continue
            norm = _normalized_shape(coords)
            rows, cols = _shape_size(norm)
            if rows < 2 or cols < 2:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            shapes.append(norm)
    return shapes


SHAPES = _shape_library()


def _sample_spaced_positions(rng: np.random.Generator, limit: int, count: int) -> list[int] | None:
    positions = list(range(2, limit - 2))
    if count == 1:
        return [int(rng.choice(positions))]
    pairs = [(a, b) for i, a in enumerate(positions) for b in positions[i + 1 :] if b - a >= 4]
    if not pairs:
        return None
    pair = pairs[int(rng.integers(len(pairs)))]
    return [int(pair[0]), int(pair[1])]


def _coords_for(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> np.ndarray:
    return np.array([(r0 + r, c0 + c) for r, c in shape], dtype=int)


def _mark_blocked(blocked: np.ndarray, coords: np.ndarray) -> None:
    h, w = blocked.shape
    for r, c in coords:
        blocked[r, c] = True
        for nr, nc in _neighbors8(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _candidate_positions(
    shape: tuple[tuple[int, int], ...],
    blocked: np.ndarray,
    guide_mask: np.ndarray,
    touch_mask: np.ndarray,
    mode: str,
) -> list[np.ndarray]:
    h, w = blocked.shape
    sh, sw = _shape_size(shape)
    candidates = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            coords = _coords_for(shape, r0, c0)
            rr, cc = coords[:, 0], coords[:, 1]
            if blocked[rr, cc].any():
                continue
            on_guide = guide_mask[rr, cc]
            on_touch = touch_mask[rr, cc]
            if mode == "safe" and on_touch.any():
                continue
            if mode == "overlap":
                if not on_guide.any():
                    continue
                if int(on_guide.sum()) > max(2, len(shape) // 2):
                    continue
            if mode == "adjacent":
                if on_guide.any() or not on_touch.any():
                    continue
            candidates.append(coords)
    return candidates


def _place_objects(
    rng: np.random.Generator,
    h: int,
    w: int,
    guide_mask: np.ndarray,
    touch_mask: np.ndarray,
    shapes: list[tuple[tuple[int, int], ...]],
    colors: list[int],
    touched_modes: list[str | None],
) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    template_order = sorted(range(len(shapes)), key=lambda i: len(shapes[i]), reverse=True)
    for idx in template_order:
        candidates = _candidate_positions(shapes[idx], blocked, guide_mask, touch_mask, "safe")
        if not candidates:
            return None
        coords = candidates[int(rng.integers(len(candidates)))]
        grid[coords[:, 0], coords[:, 1]] = colors[idx]
        _mark_blocked(blocked, coords)

    placeholder_order = sorted(range(len(shapes)), key=lambda i: len(shapes[i]), reverse=True)
    for idx in placeholder_order:
        mode = "safe" if touched_modes[idx] is None else touched_modes[idx]
        candidates = _candidate_positions(shapes[idx], blocked, guide_mask, touch_mask, mode)
        if not candidates:
            return None
        coords = candidates[int(rng.integers(len(candidates)))]
        grid[coords[:, 0], coords[:, 1]] = PLACEHOLDER
        _mark_blocked(blocked, coords)
    return grid


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        scene = _parse_scene(inp)
        touched = [ph for ph in scene["placeholders"] if ph["touched"]]
        untouched = [ph for ph in scene["placeholders"] if not ph["touched"]]
        if not touched or not untouched:
            return False
        for template in scene["templates"]:
            coords = template["coords"]
            if scene["touch_mask"][coords[:, 0], coords[:, 1]].any():
                return False
        if len(scene["templates"]) != len(scene["placeholders"]):
            return False
        if len({tpl["shape"] for tpl in scene["templates"]}) != len(scene["templates"]):
            return False
        if len(touched) >= 2 and (not any(ph["overlap"] for ph in touched) or not any(not ph["overlap"] for ph in touched)):
            return False
        out = main(inp)
        if out.shape != inp.shape:
            return False
        for ph in touched:
            coords = ph["coords"]
            if not np.all(out[coords[:, 0], coords[:, 1]] == ph["match_color"]):
                return False
        for ph in untouched:
            coords = ph["coords"]
            visible = ~scene["guide_mask"][coords[:, 0], coords[:, 1]]
            if visible.any() and not np.all(out[coords[visible, 0], coords[visible, 1]] == PLACEHOLDER):
                return False
        for template in scene["templates"]:
            coords = template["coords"]
            visible = ~scene["guide_mask"][coords[:, 0], coords[:, 1]]
            if visible.any() and not np.all(out[coords[visible, 0], coords[visible, 1]] == template["color"]):
                return False
        return True
    except AssertionError:
        return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))

    row_count, col_count = LINE_CHOICES[int(rng.integers(len(LINE_CHOICES)))]
    rows = _sample_spaced_positions(rng, h, row_count)
    cols = _sample_spaced_positions(rng, w, col_count)
    if rows is None or cols is None:
        return None

    guide_mask, touch_mask = _guide_masks((h, w), rows, cols)
    pair_count = int(rng.choice(PAIR_COUNT_CHOICES))
    shape_ids = rng.choice(len(SHAPES), size=pair_count, replace=False)
    shapes = [SHAPES[int(i)] for i in shape_ids]
    colors = [int(c) for c in rng.choice(TEMPLATE_COLORS, size=pair_count, replace=False)]

    if pair_count == 2:
        touched_count = 1
    elif pair_count == 3:
        touched_count = 2
    else:
        touched_count = int(rng.choice([2, 3]))
    touched_modes: list[str | None] = [None] * pair_count
    touched_indices = list(range(pair_count))
    rng.shuffle(touched_indices)
    touched_indices = touched_indices[:touched_count]
    if touched_count == 1:
        touched_modes[touched_indices[0]] = str(rng.choice(["overlap", "adjacent"]))
    else:
        base_modes = ["overlap", "adjacent"] + [str(rng.choice(["overlap", "adjacent"])) for _ in range(touched_count - 2)]
        rng.shuffle(base_modes)
        for idx, mode in zip(touched_indices, base_modes):
            touched_modes[idx] = mode

    object_grid = _place_objects(rng, h, w, guide_mask, touch_mask, shapes, colors, touched_modes)
    if object_grid is None:
        return None

    inp = object_grid
    for r in rows:
        inp[r, 0] = GUIDE
        inp[r, w - 1] = GUIDE
    for c in cols:
        inp[0, c] = GUIDE
        inp[h - 1, c] = GUIDE

    return inp if _semantic_checks(inp) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle input")
