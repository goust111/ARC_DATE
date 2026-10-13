from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
_RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components_any(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _is_globally_isolated(grid: np.ndarray, r: int, c: int, bg: int) -> bool:
    h, w = grid.shape
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if (rr != r or cc != c) and int(grid[rr, cc]) != bg:
                return False
    return True


def _find_marker_rectangles(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int, int, int, int]], np.ndarray]:
    marker_mask = np.zeros_like(grid, dtype=bool)
    rectangles: list[tuple[int, int, int, int, int]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        coords = np.argwhere(grid == color)
        if len(coords) != 4:
            continue
        rows = sorted(set(int(x) for x in coords[:, 0]))
        cols = sorted(set(int(x) for x in coords[:, 1]))
        if len(rows) != 2 or len(cols) != 2:
            continue
        corners = {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}
        if corners != {tuple(map(int, p)) for p in coords}:
            continue
        if not all(_is_globally_isolated(grid, r, c, bg) for r, c in corners):
            continue
        rectangles.append((color, rows[0], cols[0], rows[1], cols[1]))
        for r, c in corners:
            marker_mask[r, c] = True
    rectangles.sort(key=lambda x: ((x[3] - x[1] + 1) * (x[4] - x[2] + 1), x[1], x[2]))
    return rectangles, marker_mask


def _capture_count(coords: np.ndarray, rectangles: list[tuple[int, int, int, int, int]]) -> int:
    total = 0
    for _color, r0, c0, r1, c1 in rectangles:
        inside = np.all((coords[:, 0] >= r0) & (coords[:, 0] <= r1) & (coords[:, 1] >= c0) & (coords[:, 1] <= c1))
        total += int(inside)
    return total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase objects that are covered by an even number of virtual corner-made rectangles.

    Input structure:
    - A black background contains several separated foreground objects.
    - Some colors appear only as four isolated singleton pixels; each such quartet marks the four corners of one axis-aligned rectangle.
    - The rectangles themselves are not drawn, and every foreground object lies either entirely inside or entirely outside each marked rectangle.

    Rule:
    - Interpret every valid four-corner marker quartet as one inclusive virtual rectangle.
    - Remove all corner-marker pixels from the output.
    - For each remaining foreground object, count how many virtual rectangles fully contain it; keep the object unchanged if that count is odd, otherwise erase the whole object.

    Key insight:
    - The corners define invisible capture nets, and the true decision is the parity of complete containment across all nets, not merely whether an object is inside at least one rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    rectangles, marker_mask = _find_marker_rectangles(g, bg)

    work = g.copy()
    work[marker_mask] = bg
    out = work.copy()
    for coords in _components_any(work, bg):
        if _capture_count(coords, rectangles) % 2 == 0:
            out[coords[:, 0], coords[:, 1]] = bg
    return out


_BASE_SHAPES = [
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
]


def _transforms(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen = set()
    cur = mask
    for _ in range(4):
        for cand in (cur, np.fliplr(cur)):
            key = cand.tobytes() + str(cand.shape).encode()
            if key not in seen:
                seen.add(key)
                out.append(cand.copy())
        cur = np.rot90(cur)
    return out


def _sample_nested_rectangles() -> tuple[int, int, list[tuple[int, int, int, int]]]:
    for _ in range(300):
        n_rect = int(_RNG.choice([2, 2, 3, 3, 4]))
        inner_h = int(_RNG.integers(4, 7))
        inner_w = int(_RNG.integers(4, 7))
        r0 = 0
        c0 = 0
        r1 = inner_h - 1
        c1 = inner_w - 1
        rects = [(r0, c0, r1, c1)]
        for _ in range(n_rect - 1):
            top = int(_RNG.integers(2, 4))
            bottom = int(_RNG.integers(2, 4))
            left = int(_RNG.integers(2, 4))
            right = int(_RNG.integers(2, 4))
            r0 -= top
            c0 -= left
            r1 += bottom
            c1 += right
            rects.append((r0, c0, r1, c1))
        rects = list(reversed(rects))
        outer_h = rects[0][2] - rects[0][0] + 1
        outer_w = rects[0][3] - rects[0][1] + 1
        margin_top = int(_RNG.integers(2, 5))
        margin_bottom = int(_RNG.integers(2, 5))
        margin_left = int(_RNG.integers(2, 5))
        margin_right = int(_RNG.integers(2, 5))
        h = outer_h + margin_top + margin_bottom
        w = outer_w + margin_left + margin_right
        if max(h, w) > 30:
            continue
        dr = margin_top - rects[0][0]
        dc = margin_left - rects[0][1]
        shifted = [(a + dr, b + dc, c + dr, d + dc) for a, b, c, d in rects]
        return h, w, shifted
    raise RuntimeError("failed to sample rectangles")


def _capture_grid(shape: tuple[int, int], rectangles: list[tuple[int, int, int, int]]) -> np.ndarray:
    capture = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in rectangles:
        capture[r0 : r1 + 1, c0 : c1 + 1] += 1
    return capture


def _build_safe_mask(shape: tuple[int, int], rectangles: list[tuple[int, int, int, int]]) -> np.ndarray:
    h, w = shape
    safe = np.ones((h, w), dtype=bool)
    for r0, c0, r1, c1 in rectangles:
        safe[r0, c0 : c1 + 1] = False
        safe[r1, c0 : c1 + 1] = False
        safe[r0 : r1 + 1, c0] = False
        safe[r0 : r1 + 1, c1] = False
        corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        for r, c in corners:
            safe[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = False
    return safe


def _initial_blocked(shape: tuple[int, int], rectangles: list[tuple[int, int, int, int]]) -> np.ndarray:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in rectangles:
        for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
            blocked[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True
    return blocked


def _find_candidates(region: np.ndarray, blocked: np.ndarray, shape_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = region.shape
    sh, sw = shape_mask.shape
    candidates = []
    for r in range(h - sh + 1):
        for c in range(w - sw + 1):
            sub_region = region[r : r + sh, c : c + sw]
            sub_blocked = blocked[r : r + sh, c : c + sw]
            if np.all(sub_region[shape_mask]) and not np.any(sub_blocked[shape_mask]):
                candidates.append((r, c))
    return candidates


def _mark_blocked(blocked: np.ndarray, coords: np.ndarray) -> None:
    h, w = blocked.shape
    for r, c in coords:
        blocked[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True


def _expected_output(grid: np.ndarray, rectangles: list[tuple[int, int, int, int]], objects: list[dict]) -> np.ndarray:
    out = grid.copy()
    for r0, c0, r1, c1 in rectangles:
        for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
            out[r, c] = BG
    for obj in objects:
        if obj["capture"] % 2 == 0:
            coords = obj["coords"]
            out[coords[:, 0], coords[:, 1]] = BG
    return out


def _place_objects(
    h: int,
    w: int,
    rectangles: list[tuple[int, int, int, int]],
    marker_colors: list[int],
) -> tuple[np.ndarray, list[dict]]:
    capture = _capture_grid((h, w), rectangles)
    safe = _build_safe_mask((h, w), rectangles)
    blocked = _initial_blocked((h, w), rectangles)

    required_layers = [0, 1, 2]
    if len(rectangles) >= 3:
        required_layers.append(3)
    extras_pool = list(range(len(rectangles) + 1))
    extra_count = int(_RNG.integers(1, 3))
    layer_targets = list(required_layers)
    for _ in range(extra_count):
        layer_targets.append(int(_RNG.choice(extras_pool)))

    layer_area = {k: int(np.count_nonzero((capture == k) & safe)) for k in range(len(rectangles) + 1)}
    layer_targets.sort(key=lambda k: (layer_area[k], k))

    all_colors = [c for c in range(1, 10) if c not in marker_colors]
    if len(layer_targets) > len(all_colors):
        raise RuntimeError("not enough colors for objects")
    object_colors = list(_RNG.choice(all_colors, size=len(layer_targets), replace=False))

    grid = np.zeros((h, w), dtype=int)
    objects: list[dict] = []
    for idx, target in enumerate(layer_targets):
        region = (capture == target) & safe
        if layer_area[target] < 3:
            raise RuntimeError("degenerate layer")
        success = False
        for _ in range(60):
            base = _BASE_SHAPES[int(_RNG.integers(0, len(_BASE_SHAPES)))]
            shape_mask = _transforms(base)[int(_RNG.integers(0, len(_transforms(base))))]
            if int(shape_mask.sum()) < 3:
                continue
            candidates = _find_candidates(region, blocked, shape_mask)
            if not candidates:
                continue
            r, c = candidates[int(_RNG.integers(0, len(candidates)))]
            rel = np.argwhere(shape_mask)
            coords = rel + np.array([r, c])
            color = int(object_colors[idx])
            grid[coords[:, 0], coords[:, 1]] = color
            _mark_blocked(blocked, coords)
            objects.append(
                {
                    "color": color,
                    "coords": coords.copy(),
                    "capture": int(target),
                    "bbox": _bbox(coords),
                }
            )
            success = True
            break
        if not success:
            raise RuntimeError("failed to place object")
    return grid, objects


def _validate_sample(grid: np.ndarray, rectangles: list[tuple[int, int, int, int]], objects: list[dict], expected: np.ndarray) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert len(rectangles) >= 2
    captures = [int(obj["capture"]) for obj in objects]
    assert 0 in captures
    assert 1 in captures
    assert 2 in captures
    assert any(v % 2 == 1 for v in captures)
    assert any(v == 0 for v in captures)
    assert any(v > 0 and v % 2 == 0 for v in captures)

    detected, marker_mask = _find_marker_rectangles(grid, BG)
    detected_boxes = sorted((r0, c0, r1, c1) for _color, r0, c0, r1, c1 in detected)
    assert detected_boxes == sorted(rectangles)

    work = grid.copy()
    work[marker_mask] = BG
    comps = _components_any(work, BG)
    by_color = {int(work[coords[0, 0], coords[0, 1]]): coords for coords in comps}
    assert len(by_color) == len(objects)

    for obj in objects:
        coords = by_color[obj["color"]]
        assert len(coords) == len(obj["coords"])
        assert _bbox(coords) == obj["bbox"]
        inferred = _capture_count(coords, [(9, *rect) for rect in rectangles])
        assert inferred == obj["capture"]
        for r0, c0, r1, c1 in rectangles:
            inside = np.all((coords[:, 0] >= r0) & (coords[:, 0] <= r1) & (coords[:, 1] >= c0) & (coords[:, 1] <= c1))
            outside = np.all((coords[:, 0] < r0) | (coords[:, 0] > r1) | (coords[:, 1] < c0) | (coords[:, 1] > c1))
            assert inside or outside

    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert np.any(actual != BG)
    assert np.any(expected != grid)


def _gen_attempt() -> np.ndarray:
    h, w, rectangles = _sample_nested_rectangles()
    marker_colors = list(_RNG.choice(range(1, 10), size=len(rectangles), replace=False))
    object_grid, objects = _place_objects(h, w, rectangles, marker_colors)

    grid = np.zeros((h, w), dtype=int)
    grid[:] = BG
    grid += object_grid
    for color, (r0, c0, r1, c1) in zip(marker_colors, rectangles):
        for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
            grid[r, c] = int(color)

    expected = _expected_output(grid, rectangles, objects)
    _validate_sample(grid, rectangles, objects, expected)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        try:
            return _gen_attempt()
        except RuntimeError:
            continue
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid virtual-rectangle parity sample")
