from __future__ import annotations

import random

import numpy as np


DIRECTIONS = ("up", "right", "down", "left")
ROTATION_K = {"up": 0, "right": 3, "down": 2, "left": 1}
COLOR_POOL = tuple(range(10))
GRID_SIDE_RANGE = (21, 30)
SEED_COUNT_RANGE = (4, 6)
TEMPLATE_BBOX_OPTIONS = ((3, 3), (3, 5), (5, 3), (5, 5))
TEMPLATE_ATTEMPTS = 200
PLACEMENT_PADDING = 1
MIN_DIRECTION_DISTANCE = 5
GENERATION_ATTEMPTS = 400


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c, h, w):
            if mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _rotation_keys(mask: np.ndarray) -> set[tuple[tuple[int, int], bytes]]:
    return {(tuple(np.rot90(mask, k).shape), np.rot90(mask, k).tobytes()) for k in range(4)}


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            r0, c0, r1, c1 = _bbox(arr)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "mask": mask,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(len(arr)),
                }
            )
    return components


def _direction_for_seed(seed_r: int, seed_c: int, center_r: int, center_c: int) -> str:
    dr = int(seed_r) - int(center_r)
    dc = int(seed_c) - int(center_c)
    assert dr != 0 or dc != 0, "seed cannot coincide with template center"
    assert abs(dr) != abs(dc), "seed must have an unambiguous primary direction"
    if abs(dr) > abs(dc):
        return "up" if dr < 0 else "down"
    return "left" if dc < 0 else "right"


def _rotated_mask(mask: np.ndarray, direction: str) -> np.ndarray:
    return np.rot90(mask, ROTATION_K[direction]).astype(bool)


def _stamp_cells(mask: np.ndarray, center: tuple[int, int]) -> list[tuple[int, int]]:
    cr, cc = map(int, center)
    anchor_r = mask.shape[0] // 2
    anchor_c = mask.shape[1] // 2
    out = []
    for r, c in np.argwhere(mask):
        out.append((cr + int(r) - anchor_r, cc + int(c) - anchor_c))
    return out


def _apply_stamp(out: np.ndarray, mask: np.ndarray, center: tuple[int, int], color: int, bg: int) -> None:
    h, w = out.shape
    for r, c in _stamp_cells(mask, center):
        assert 0 <= r < h and 0 <= c < w, "stamp leaves the grid"
        if int(out[r, c]) == bg:
            out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the central asymmetric template as a stamp, rotating it by each seed point's direction and recoloring it with that seed's color.

    Input structure:
    - A dominant background fills most of the grid.
    - Near the middle there is one multi-pixel monochrome template component whose tight bbox has odd side lengths and whose four rotations are all distinct.
    - Away from the template there are several single-pixel colored seed points.

    Rule:
    - Keep the central template unchanged.
    - For each seed, compare its position to the template bbox center and take the primary direction: up, right, down, or left according to the larger absolute row/column offset.
    - Rotate the template by 0/90/180/270 degrees for up/right/down/left respectively, align the rotated template's bbox center to the seed point, and paint its foreground cells in the seed color only onto background cells.

    Key insight:
    - The middle object is not another ordinary shape; it is a shared asymmetric stamp whose orientation is controlled by each seed's relative direction, while the seed's own color becomes the stamp color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _bg_color(g)
    components = _find_components(g, bg)
    template_candidates = [comp for comp in components if int(comp["area"]) > 1]
    assert len(template_candidates) == 1, "expected exactly one multi-pixel template"
    template = template_candidates[0]
    mask = np.asarray(template["mask"], dtype=bool)
    assert mask.shape[0] % 2 == 1 and mask.shape[1] % 2 == 1, "template bbox must have odd side lengths"
    assert len(_rotation_keys(mask)) == 4, "template must be asymmetric under 90-degree rotation"

    r0, c0, r1, c1 = template["bbox"]
    center_r = (int(r0) + int(r1)) // 2
    center_c = (int(c0) + int(c1)) // 2

    seed_components = [comp for comp in components if int(comp["area"]) == 1]
    assert seed_components, "expected at least one seed point"

    out = g.copy()
    for comp in seed_components:
        seed_r, seed_c = map(int, comp["coords"][0])
        seed_color = int(comp["color"])
        direction = _direction_for_seed(seed_r, seed_c, center_r, center_c)
        _apply_stamp(out, _rotated_mask(mask, direction), (seed_r, seed_c), seed_color, bg)

    return out


def _random_template_mask() -> np.ndarray | None:
    for _ in range(TEMPLATE_ATTEMPTS):
        h, w = random.choice(TEMPLATE_BBOX_OPTIONS)
        density = random.uniform(0.32, 0.62)
        mask = np.array([[random.random() < density for _ in range(w)] for _ in range(h)], dtype=bool)
        mask[h // 2, w // 2] = True
        area = int(mask.sum())
        if area < 4 or area > max(5, int(round(0.6 * h * w))):
            continue
        if not _touches_all_sides(mask):
            continue
        if not _connected(mask):
            continue
        if len(_rotation_keys(mask)) != 4:
            continue
        return mask
    return None


def _direction_match(r: int, c: int, center_r: int, center_c: int, direction: str) -> bool:
    dr = r - center_r
    dc = c - center_c
    if max(abs(dr), abs(dc)) < MIN_DIRECTION_DISTANCE:
        return False
    if direction == "up":
        return dr < 0 and -dr > abs(dc)
    if direction == "down":
        return dr > 0 and dr > abs(dc)
    if direction == "left":
        return dc < 0 and -dc > abs(dr)
    return dc > 0 and dc > abs(dr)


def _dilate_cells(cells: set[tuple[int, int]], h: int, w: int, radius: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _candidate_centers(
    shape: tuple[int, int],
    direction: str,
    template_center: tuple[int, int],
    blocked: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[tuple[int, int], set[tuple[int, int]]]]:
    center_r, center_c = template_center
    mask = np.ones(shape, dtype=bool)
    anchor_r = shape[0] // 2
    anchor_c = shape[1] // 2
    offsets = [(int(r) - anchor_r, int(c) - anchor_c) for r, c in np.argwhere(mask)]
    candidates: list[tuple[tuple[int, int], set[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            if not _direction_match(r, c, center_r, center_c, direction):
                continue
            cells: set[tuple[int, int]] = set()
            ok = True
            for dr, dc in offsets:
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    ok = False
                    break
                cells.add((rr, cc))
            if not ok or cells & blocked:
                continue
            candidates.append(((r, c), cells))
    return candidates


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    seed_count = random.randint(*SEED_COUNT_RANGE)

    template_mask = _random_template_mask()
    if template_mask is None:
        return None
    template_mask = np.rot90(template_mask, random.randrange(4)).astype(bool)

    palette = random.sample(COLOR_POOL, seed_count + 2)
    bg = int(palette[0])
    template_color = int(palette[1])
    seed_colors = [int(color) for color in palette[2:]]

    grid = np.full((h, w), bg, dtype=int)
    expected = np.full((h, w), bg, dtype=int)
    center = (h // 2, w // 2)

    template_cells = set(_stamp_cells(template_mask, center))
    for r, c in template_cells:
        grid[r, c] = template_color
        expected[r, c] = template_color

    blocked = _dilate_cells(template_cells, h, w, PLACEMENT_PADDING)
    directions = list(DIRECTIONS)
    extra = seed_count - len(directions)
    directions.extend(random.choice(DIRECTIONS) for _ in range(extra))
    random.shuffle(directions)

    for seed_color, direction in zip(seed_colors, directions):
        rotated = _rotated_mask(template_mask, direction)
        candidates = _candidate_centers(rotated.shape, direction, center, blocked, h, w)
        if not candidates:
            return None
        seed_center, stamp_cells = random.choice(candidates)
        sr, sc = seed_center
        grid[sr, sc] = seed_color
        expected[sr, sc] = seed_color
        _apply_stamp(expected, rotated, seed_center, seed_color, bg)
        blocked |= _dilate_cells(stamp_cells, h, w, PLACEMENT_PADDING)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-rotation ARC input")
