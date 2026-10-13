from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
MAX_FRAGMENT_AREA = 4
MIN_RING_SIZE = 5
RING_STEP_MIN = 3
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_by_color(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords_arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "area": int(coords_arr.shape[0]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return components


def _boundary_reachable(open_mask: np.ndarray) -> np.ndarray:
    h, w = open_mask.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if open_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if open_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and open_mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _interior_from_ring(mask: np.ndarray) -> np.ndarray:
    open_mask = ~mask
    outside = _boundary_reachable(open_mask)
    return open_mask & (~outside)


def _mask_from_coords(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each small fragment by the number of closed outlines that contain it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    palette = [int(v) for v in g[-1, :].tolist() if int(v) != BG]
    assert palette, "expected a bottom-row palette"

    scene = g[:-1, :]
    components = _components_by_color(scene)

    ring_interiors: list[np.ndarray] = []
    fragments: list[dict] = []
    for component in components:
        if int(component["area"]) > MAX_FRAGMENT_AREA:
            mask = _mask_from_coords(scene.shape, component["coords"])
            interior = _interior_from_ring(mask)
            if interior.any():
                ring_interiors.append(interior)
                continue
        fragments.append(component)

    out = g.copy()
    for fragment in fragments:
        r, c = map(int, fragment["coords"][0])
        depth = sum(int(interior[r, c]) for interior in ring_interiors)
        assert 0 <= depth < len(palette), "fragment depth exceeds palette"
        out[fragment["coords"][:, 0], fragment["coords"][:, 1]] = int(palette[depth])

    return out


def _normalize_shape(cells: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in cells))


def _rotate_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((int(c), int(-r)) for r, c in cells)


def _reflect_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((int(r), int(-c)) for r, c in cells)


def _build_shapes() -> tuple[tuple[tuple[int, int], ...], ...]:
    bases = (
        ((0, 0), (0, 1)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
    )
    seen: set[tuple[tuple[int, int], ...]] = set()
    shapes: list[tuple[tuple[int, int], ...]] = []
    for base in bases:
        for flipped in (False, True):
            current = _reflect_shape(base) if flipped else base
            for _ in range(4):
                norm = _normalize_shape(current)
                if norm not in seen:
                    seen.add(norm)
                    shapes.append(norm)
                current = _rotate_shape(current)
    return tuple(shapes)


SHAPES = _build_shapes()


def _rect_dims(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    return int(rect[2] - rect[0] + 1), int(rect[3] - rect[1] + 1)


def _rect_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        int(a[2]) + pad < int(b[0])
        or int(b[2]) + pad < int(a[0])
        or int(a[3]) + pad < int(b[1])
        or int(b[3]) + pad < int(a[1])
    )


def _ring_mask(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = map(int, rect)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _draw_ring(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, rect)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _sample_chain(
    scene_h: int,
    scene_w: int,
    depth: int,
    avoid: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]] | None:
    min_outer_h = MIN_RING_SIZE + 2 * RING_STEP_MIN * (depth - 1)
    min_outer_w = MIN_RING_SIZE + 2 * RING_STEP_MIN * (depth - 1)
    if scene_h < min_outer_h + 2 or scene_w < min_outer_w + 2:
        return None

    for _ in range(160):
        max_outer_h = min(scene_h - 2, min_outer_h + 6)
        max_outer_w = min(scene_w - 2, min_outer_w + 8)
        outer_h = random.randint(min_outer_h, max_outer_h)
        outer_w = random.randint(min_outer_w, max_outer_w)
        r0 = random.randint(1, scene_h - outer_h - 1)
        c0 = random.randint(1, scene_w - outer_w - 1)
        outer = (int(r0), int(c0), int(r0 + outer_h - 1), int(c0 + outer_w - 1))
        if any(_rect_overlaps(outer, other, pad=2) for other in avoid):
            continue

        rects = [outer]
        current = outer
        ok = True
        for level in range(1, depth):
            remaining = depth - level - 1
            min_h = MIN_RING_SIZE + 2 * RING_STEP_MIN * remaining
            min_w = MIN_RING_SIZE + 2 * RING_STEP_MIN * remaining
            cur_h, cur_w = _rect_dims(current)
            if cur_h < min_h + 2 * RING_STEP_MIN or cur_w < min_w + 2 * RING_STEP_MIN:
                ok = False
                break

            inner_h = random.randint(min_h, cur_h - 2 * RING_STEP_MIN)
            inner_w = random.randint(min_w, cur_w - 2 * RING_STEP_MIN)
            slack_h = cur_h - inner_h
            slack_w = cur_w - inner_w
            top_inset = random.randint(RING_STEP_MIN, slack_h - RING_STEP_MIN)
            left_inset = random.randint(RING_STEP_MIN, slack_w - RING_STEP_MIN)
            bottom_inset = slack_h - top_inset
            right_inset = slack_w - left_inset
            current = (
                int(current[0] + top_inset),
                int(current[1] + left_inset),
                int(current[2] - bottom_inset),
                int(current[3] - right_inset),
            )
            rects.append(current)

        if ok:
            return rects
    return None


def _shape_bounds(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    height = max(r for r, _ in shape) + 1
    width = max(c for _, c in shape) + 1
    return int(height), int(width)


def _can_place_shape(
    scene: np.ndarray,
    depth_map: np.ndarray,
    fragment_mask: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    anchor_r: int,
    anchor_c: int,
    depth: int,
) -> bool:
    h, w = scene.shape
    cells = [(anchor_r + dr, anchor_c + dc) for dr, dc in shape]
    for rr, cc in cells:
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if int(scene[rr, cc]) != BG or int(depth_map[rr, cc]) != depth:
            return False
    cell_set = set(cells)
    for rr, cc in cells:
        for nr, nc in _neighbors4(rr, cc):
            if (nr, nc) in cell_set:
                continue
            if 0 <= nr < h and 0 <= nc < w and fragment_mask[nr, nc]:
                return False
    return True


def _place_fragment(
    scene: np.ndarray,
    depth_map: np.ndarray,
    fragment_mask: np.ndarray,
    depth: int,
    color: int,
) -> dict | None:
    shapes = list(SHAPES)
    random.shuffle(shapes)
    h, w = scene.shape
    for shape in shapes:
        sh, sw = _shape_bounds(shape)
        anchors: list[tuple[int, int]] = []
        for r in range(h - sh + 1):
            for c in range(w - sw + 1):
                if _can_place_shape(scene, depth_map, fragment_mask, shape, r, c, depth):
                    anchors.append((r, c))
        if not anchors:
            continue
        r0, c0 = random.choice(anchors)
        coords = np.array([(r0 + dr, c0 + dc) for dr, dc in shape], dtype=int)
        scene[coords[:, 0], coords[:, 1]] = int(color)
        fragment_mask[coords[:, 0], coords[:, 1]] = True
        return {"coords": coords, "depth": int(depth), "color": int(color)}
    return None


def _depth_map_from_rings(shape: tuple[int, int], ring_rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    depth_map = np.zeros(shape, dtype=int)
    for rect in ring_rects:
        depth_map[_interior_from_ring(_ring_mask(shape, rect))] += 1
    return depth_map


def _expected_output(grid: np.ndarray, fragments: list[dict], palette: list[int]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for fragment in fragments:
        coords = fragment["coords"]
        out[coords[:, 0], coords[:, 1]] = int(palette[int(fragment["depth"])])
    return out


def _build_palette_row(width: int, palette: list[int]) -> np.ndarray:
    row = np.zeros(width, dtype=int)
    span = 2 * len(palette) - 1
    start = random.randint(1, width - span - 1)
    for i, color in enumerate(palette):
        row[start + 2 * i] = int(color)
    return row


def _generate_case() -> np.ndarray | None:
    max_depth = random.randint(1, 3)
    palette_len = max_depth + 1

    min_scene_h = max(15, MIN_RING_SIZE + 2 * RING_STEP_MIN * (max_depth - 1) + 2)
    min_scene_w = max(17, 2 * palette_len + 3, MIN_RING_SIZE + 2 * RING_STEP_MIN * (max_depth - 1) + 2)
    scene_h = random.randint(min_scene_h, min_scene_h + 7)
    scene_w = random.randint(min_scene_w, min_scene_w + 9)

    primary = _sample_chain(scene_h, scene_w, max_depth, [])
    if primary is None:
        return None

    extra: list[tuple[int, int, int, int]] = []
    if random.random() < 0.5:
        extra_chain = _sample_chain(scene_h, scene_w, 1, [primary[0]])
        if extra_chain is not None:
            extra = extra_chain

    ring_rects = primary + extra
    ring_count = len(ring_rects)
    if palette_len + ring_count >= 10:
        return None

    colors = list(range(1, 10))
    random.shuffle(colors)
    palette = colors[:palette_len]
    ring_colors = colors[palette_len : palette_len + ring_count]
    fragment_colors = colors[palette_len + ring_count :]
    if not fragment_colors:
        return None

    scene = np.zeros((scene_h, scene_w), dtype=int)
    for rect, color in zip(ring_rects, ring_colors, strict=False):
        _draw_ring(scene, rect, int(color))

    depth_map = _depth_map_from_rings(scene.shape, ring_rects)
    fragment_mask = np.zeros(scene.shape, dtype=bool)
    fragments: list[dict] = []

    for depth in range(max_depth + 1):
        fragment = _place_fragment(scene, depth_map, fragment_mask, depth, random.choice(fragment_colors))
        if fragment is None:
            return None
        fragments.append(fragment)

    extra_target = random.randint(0, max_depth + 2)
    for _ in range(extra_target):
        depth = random.randint(0, max_depth)
        fragment = _place_fragment(scene, depth_map, fragment_mask, depth, random.choice(fragment_colors))
        if fragment is None:
            break
        fragments.append(fragment)

    grid = np.vstack([scene, _build_palette_row(scene_w, palette)[None, :]])
    expected = _expected_output(grid, fragments, palette)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-depth recoloring task")
