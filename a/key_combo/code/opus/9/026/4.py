from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
BACKGROUND = 0
GRID_MIN = 18
GRID_MAX = 28
RECT_COUNT_RANGE = (3, 5)
LAYER_COUNT_RANGE = (2, 5)
MAX_GENERATION_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
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
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _peel_layer_colors(rect: np.ndarray) -> list[int]:
    cur = np.asarray(rect, dtype=int)
    colors: list[int] = []
    while True:
        h, w = cur.shape
        assert h >= 1 and w >= 1, "empty rectangle during peeling"
        vals = np.unique(cur)
        if vals.size == 1:
            colors.append(int(vals[0]))
            return colors

        outer = int(cur[0, 0])
        max_t = min((h - 1) // 2, (w - 1) // 2)
        best_t = 0
        for t in range(1, max_t + 1):
            if not np.all(cur[:t, :] == outer):
                break
            if not np.all(cur[h - t :, :] == outer):
                break
            if not np.all(cur[:, :t] == outer):
                break
            if not np.all(cur[:, w - t :] == outer):
                break
            best_t = t

        assert best_t >= 1, "expected a peelable uniform outer layer"
        colors.append(outer)
        cur = cur[best_t : h - best_t, best_t : w - best_t]


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    rects = []
    for coords in _components4(g != bg):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch != bg), "foreground component must fill its bounding box"
        layer_colors = _peel_layer_colors(patch)
        rects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "left": c0,
                "layers": layer_colors,
            }
        )
    assert rects, "expected nested rectangles"
    lefts = [rect["left"] for rect in rects]
    assert len(lefts) == len(set(lefts)), "rectangle left edges must be distinct"
    return rects


def _render_histogram(rects: list[dict], bg: int) -> np.ndarray:
    ordered = sorted(rects, key=lambda rect: rect["left"])
    max_layers = max(len(rect["layers"]) for rect in ordered)
    out = np.full((max_layers, len(ordered)), bg, dtype=int)
    for col, rect in enumerate(ordered):
        for offset, color in enumerate(reversed(rect["layers"])):
            out[max_layers - 1 - offset, col] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Convert nested rectangles into left-to-right reversed-layer histogram columns.

    Input structure:
    - The grid has black background and 3-5 separate filled rectangular foreground objects.
    - Each foreground rectangle consists of 2-5 inward nested monochrome rectangular layers.
    - Every non-innermost layer is a uniform rectangular border; the innermost layer may be a solid rectangle.
    - Different rectangles have distinct bbox left edges, so their horizontal order is unambiguous.

    Rule:
    - For each rectangle independently, peel layers from outermost to innermost by repeatedly removing the current
      uniform outer border and recording its color.
    - Let the number of recorded layers be the rectangle's height in the output.
    - Sort rectangles by bbox left edge from left to right.
    - Build one output column per rectangle: place the recorded colors in reverse order from bottom to top
      (innermost at the bottom, outermost at the top), padding the top with black up to the global maximum layer count.

    Key insight:
    - The rectangles themselves are not redrawn; only their nested color sequence survives, and it is both counted
      for height and reversed to form a compact column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rects = _extract_rectangles(g)
    return _render_histogram(rects, get_bg(g))


def _build_rectangle(colors: tuple[int, ...], thicknesses: tuple[int, ...], inner_shape: tuple[int, int]) -> np.ndarray:
    inner_h, inner_w = inner_shape
    total_pad = 2 * sum(thicknesses)
    h = inner_h + total_pad
    w = inner_w + total_pad
    rect = np.empty((h, w), dtype=int)
    r0 = c0 = 0
    r1, c1 = h - 1, w - 1
    for color, thickness in zip(colors[:-1], thicknesses):
        rect[r0 : r1 + 1, c0 : c1 + 1] = int(color)
        r0 += thickness
        c0 += thickness
        r1 -= thickness
        c1 -= thickness
    rect[r0 : r1 + 1, c0 : c1 + 1] = int(colors[-1])
    return rect


def _sample_rectangle_spec(layer_count: int, force_thick_layer: bool, max_side: int, max_inner: int) -> dict | None:
    for _ in range(80):
        colors = tuple(random.sample(PALETTE, layer_count))
        thicknesses = [random.randint(1, 2) for _ in range(layer_count - 1)]
        if force_thick_layer and layer_count >= 3 and all(t == 1 for t in thicknesses):
            thicknesses[random.randrange(len(thicknesses))] = 2

        total_pad = 2 * sum(thicknesses)
        inner_limit = min(max_inner, max_side - total_pad)
        if inner_limit < 1:
            continue
        inner_h = random.randint(1, inner_limit)
        inner_w = random.randint(1, inner_limit)
        rect = _build_rectangle(colors, tuple(thicknesses), (inner_h, inner_w))
        h, w = rect.shape
        if h > max_side or w > max_side:
            continue
        return {
            "colors": colors,
            "layers": list(colors),
            "thicknesses": tuple(thicknesses),
            "inner_shape": (inner_h, inner_w),
            "patch": rect,
            "shape": rect.shape,
        }
    return None


def _choose_layer_counts(n_rects: int) -> list[int]:
    while True:
        counts = [random.randint(*LAYER_COUNT_RANGE) for _ in range(n_rects)]
        if len(set(counts)) >= 2:
            return counts


def _place_rectangles(rects: list[dict], h: int, w: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(rects)), key=lambda i: rects[i]["shape"][0] * rects[i]["shape"][1], reverse=True)
    occupied = np.zeros((h, w), dtype=bool)
    used_lefts: set[int] = set()
    positions: list[tuple[int, int] | None] = [None] * len(rects)

    for idx in order:
        rh, rw = rects[idx]["shape"]
        candidates = [(r, c) for r in range(h - rh + 1) for c in range(w - rw + 1) if c not in used_lefts]
        random.shuffle(candidates)
        placed = None
        for r0, c0 in candidates:
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + rh + 1)
            cc1 = min(w, c0 + rw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            placed = (r0, c0)
            occupied[rr0:rr1, cc0:cc1] = True
            used_lefts.add(c0)
            break
        if placed is None:
            return None
        positions[idx] = placed

    return [pos for pos in positions if pos is not None]


def _expected_output(rects: list[dict], placements: list[tuple[int, int]]) -> np.ndarray:
    ordered = sorted(
        [
            {
                "left": c0,
                "layers": rect["layers"],
            }
            for rect, (_r0, c0) in zip(rects, placements)
        ],
        key=lambda item: item["left"],
    )
    max_layers = max(len(item["layers"]) for item in ordered)
    out = np.full((max_layers, len(ordered)), BACKGROUND, dtype=int)
    for col, item in enumerate(ordered):
        for offset, color in enumerate(reversed(item["layers"])):
            out[max_layers - 1 - offset, col] = int(color)
    return out


def _generate_attempt() -> np.ndarray | None:
    n_rects = random.randint(*RECT_COUNT_RANGE)
    target_counts = _choose_layer_counts(n_rects)
    force_thick_rect = random.randrange(n_rects) if random.random() < 0.65 else None
    max_side = 11 if n_rects == 3 else 9 if n_rects == 4 else 8
    max_inner = 4 if n_rects == 3 else 3

    rects: list[dict] = []
    used_signatures: set[tuple[int, ...]] = set()
    for i, count in enumerate(target_counts):
        spec = None
        for _ in range(100):
            candidate = _sample_rectangle_spec(count, force_thick_rect == i, max_side, max_inner)
            if candidate is None:
                continue
            signature = tuple(candidate["layers"])
            if signature in used_signatures:
                continue
            spec = candidate
            used_signatures.add(signature)
            break
        if spec is None:
            return None
        rects.append(spec)

    total_area = sum(rect["shape"][0] * rect["shape"][1] for rect in rects)
    max_h = max(rect["shape"][0] for rect in rects)
    max_w = max(rect["shape"][1] for rect in rects)
    min_side = max(max_h + 1, max_w + 1, int(np.ceil(np.sqrt(total_area / 0.28))))
    if min_side > GRID_MAX:
        return None

    for _ in range(80):
        h = random.randint(max(GRID_MIN, min_side), GRID_MAX)
        w = random.randint(max(GRID_MIN, min_side), GRID_MAX)
        positions = _place_rectangles(rects, h, w)
        if positions is None:
            continue

        grid = np.full((h, w), BACKGROUND, dtype=int)
        for rect, (r0, c0) in zip(rects, positions):
            rh, rw = rect["shape"]
            grid[r0 : r0 + rh, c0 : c0 + rw] = rect["patch"]

        expected = _expected_output(rects, positions)
        columns = {tuple(expected[:, col]) for col in range(expected.shape[1])}
        if len(columns) != expected.shape[1]:
            continue
        if not any(2 in rect["thicknesses"] for rect in rects) and random.random() < 0.4:
            continue
        if len({len(rect["layers"]) for rect in rects}) < 2:
            continue

        analyzed = _extract_rectangles(grid)
        if len(analyzed) != n_rects:
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-rectangle histogram input")
