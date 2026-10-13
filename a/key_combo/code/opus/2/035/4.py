import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
TOP_ROWS = 2
INPUT_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)


BASE_SHAPES = (
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 2), (1, 2), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (2, 0), (2, 1)),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _normalize_shape(shape) -> tuple[tuple[int, int], ...]:
    shape = list(shape)
    r0 = min(r for r, _ in shape)
    c0 = min(c for _, c in shape)
    return tuple(sorted((r - r0, c - c0) for r, c in shape))


def _rotate_shape(shape) -> tuple[tuple[int, int], ...]:
    h = max(r for r, _ in shape) + 1
    return _normalize_shape((c, h - 1 - r) for r, c in shape)


def _random_shape(*, max_size: int | None = None) -> tuple[tuple[int, int], ...]:
    pool = [shape for shape in BASE_SHAPES if max_size is None or len(shape) <= max_size]
    shape = tuple(pool[random.randrange(len(pool))])
    for _ in range(random.randrange(4)):
        shape = _rotate_shape(shape)
    return _normalize_shape(shape)


def _shape_dims(shape) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _translate(shape, r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r0 + r, c0 + c) for r, c in shape}


def _mirror_coords(coords, width: int) -> frozenset[tuple[int, int]]:
    return frozenset((r, width - 1 - c) for r, c in coords)


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            key = frozenset(coords)
            comps.append({"color": color, "coords": tuple(sorted(coords)), "key": key})
    return comps


def _analyze_scene(scene: np.ndarray) -> tuple[list[dict], set[int], dict[int, int]]:
    comps = _find_components(scene)
    by_color: dict[int, dict[frozenset[tuple[int, int]], int]] = {}
    for idx, comp in enumerate(comps):
        by_color.setdefault(int(comp["color"]), {})[comp["key"]] = idx

    paired_ids: set[int] = set()
    paired_counts: dict[int, int] = {}
    width = int(scene.shape[1])
    for idx, comp in enumerate(comps):
        color = int(comp["color"])
        mirror_key = _mirror_coords(comp["key"], width)
        partner = by_color.get(color, {}).get(mirror_key)
        if partner is None or partner == idx:
            continue
        paired_ids.add(idx)
        paired_counts[color] = paired_counts.get(color, 0) + 1
    return comps, paired_ids, paired_counts


def _marker_rows(top_strip: np.ndarray) -> list[tuple[int, int]]:
    markers = []
    for r in range(top_strip.shape[0]):
        cols = [int(v) for v in top_strip[r] if int(v) != BLACK]
        if cols:
            markers.append((r, cols[0]))
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only scene objects that belong to exact left-right mirror pairs, recolor those survivors blue, and turn the top markers into count bars.

    Input structure:
    - The top two rows form a legend strip with one marker pixel on each used row.
    - The remaining rows form a black scene panel of even width.
    - The scene contains separated 4-connected monochrome objects in the marker colors.

    Rule:
    - Split the grid into the top legend strip and the lower scene panel.
    - In the scene, extract 4-connected same-color objects and test each one against the vertical mirror axis of the panel.
    - If another object of the same color occupies exactly the reflected coordinate set, both objects survive and are recolored blue; otherwise the object is erased.
    - For each legend row, replace its marker pixel by a horizontal bar of the same color whose length equals the number of surviving scene objects of that original color.

    Key insight:
    - Mirror matching is checked at the whole-object level, not pixel by pixel: partial same-row overlaps across the midline do not count as a valid pair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] > TOP_ROWS and g.shape[1] % 2 == 0

    scene = g[TOP_ROWS:, :]
    comps, paired_ids, paired_counts = _analyze_scene(scene)

    out = np.zeros_like(g)
    for idx, comp in enumerate(comps):
        if idx not in paired_ids:
            continue
        for r, c in comp["coords"]:
            out[TOP_ROWS + r, c] = BLUE

    for row, color in _marker_rows(g[:TOP_ROWS, :]):
        length = int(paired_counts.get(int(color), 0))
        if length > 0:
            out[row, :length] = int(color)
    return out


def _touching(group_a, group_b) -> bool:
    for r1, c1 in group_a:
        for r2, c2 in group_b:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return True
    return False


def _can_place_groups(scene: np.ndarray, groups: list[set[tuple[int, int]]]) -> bool:
    h, w = scene.shape
    current = set(zip(*np.nonzero(scene != BLACK)))
    union = set()
    for group in groups:
        if not group:
            return False
        for r, c in group:
            if not (0 <= r < h and 0 <= c < w):
                return False
            if (r, c) in current or (r, c) in union:
                return False
            union.add((r, c))

    for group in groups:
        for cell in current:
            if _touching(group, {cell}):
                return False

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if _touching(groups[i], groups[j]):
                return False
    return True


def _paint_group(scene: np.ndarray, coords, color: int) -> None:
    for r, c in coords:
        scene[r, c] = int(color)


def _try_place_true_pair(scene: np.ndarray, color: int) -> bool:
    h, w = scene.shape
    half = w // 2
    for _ in range(160):
        shape = _random_shape()
        sh, sw = _shape_dims(shape)
        left_max = half - sw - 1
        if left_max < 0 or sh > h:
            continue
        r0 = random.randint(0, h - sh)
        c0 = random.randint(0, left_max)
        left = _translate(shape, r0, c0)
        right = set(_mirror_coords(left, w))
        if not _can_place_groups(scene, [left, right]):
            continue
        _paint_group(scene, left, color)
        _paint_group(scene, right, color)
        return True
    return False


def _try_place_near_miss(scene: np.ndarray, color: int) -> bool:
    h, w = scene.shape
    half = w // 2
    for _ in range(200):
        shape = _random_shape(max_size=5)
        sh, sw = _shape_dims(shape)
        left_max = half - sw - 1
        if left_max < 0 or sh > h:
            continue
        r0 = random.randint(0, h - sh)
        c0 = random.randint(0, left_max)
        left = _translate(shape, r0, c0)
        right = set(_mirror_coords(left, w))

        extras = set()
        for rr, cc in right:
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and nc >= half + 1 and (nr, nc) not in right:
                    extras.add((nr, nc))
        extras = list(extras)
        random.shuffle(extras)
        for extra in extras:
            noisy_right = set(right)
            noisy_right.add(extra)
            if len(noisy_right) > 6:
                continue
            if not _can_place_groups(scene, [left, noisy_right]):
                continue
            _paint_group(scene, left, color)
            _paint_group(scene, noisy_right, color)
            return True
    return False


def _valid_case(grid: np.ndarray, expected_counts: dict[int, int]) -> bool:
    scene = grid[TOP_ROWS:, :]
    comps, paired_ids, paired_counts = _analyze_scene(scene)
    if paired_counts != expected_counts:
        return False

    unpaired_by_color = {color: 0 for color in expected_counts}
    for idx, comp in enumerate(comps):
        color = int(comp["color"])
        if color not in unpaired_by_color:
            return False
        if idx not in paired_ids:
            unpaired_by_color[color] += 1
    if any(count < 2 for count in unpaired_by_color.values()):
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.any((out[TOP_ROWS:, :] != BLACK) & (out[TOP_ROWS:, :] != BLUE)):
        return False

    markers = _marker_rows(grid[:TOP_ROWS, :])
    if len(markers) != len(expected_counts):
        return False
    for row, color in markers:
        length = expected_counts[color]
        if not np.all(out[row, :length] == color):
            return False
        if np.any(out[row, length:] != BLACK):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        scene_h = random.randint(9, 14)
        scene_w = random.choice((12, 14, 16, 18))
        grid = np.zeros((scene_h + TOP_ROWS, scene_w), dtype=int)
        scene = grid[TOP_ROWS:, :]

        colors = random.sample(INPUT_COLORS, 2)
        if scene_h >= 11 and scene_w >= 16 and random.random() < 0.4:
            pair_counts = [1, 3]
        else:
            pair_counts = [1, 2]
        if random.random() < 0.5:
            pair_counts.reverse()

        work_items = []
        for color, n_pairs in zip(colors, pair_counts):
            for _ in range(n_pairs):
                work_items.append(("pair", color))
            work_items.append(("noise", color))
        random.shuffle(work_items)

        ok = True
        for kind, color in work_items:
            placed = _try_place_true_pair(scene, color) if kind == "pair" else _try_place_near_miss(scene, color)
            if not placed:
                ok = False
                break
        if not ok:
            continue

        marker_rows = [0, 1]
        random.shuffle(marker_rows)
        expected_counts = {}
        for row, color, n_pairs in zip(marker_rows, colors, pair_counts):
            grid[row, 0] = color
            expected_counts[color] = 2 * n_pairs

        if _valid_case(grid, expected_counts):
            return grid

    raise RuntimeError("failed to generate a valid mirror-count indicator puzzle")
