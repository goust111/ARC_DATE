from collections import deque

import numpy as np


BG = 0
BLUE = 1
SPRITE_COLORS = (2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _crop_non_bg(grid: np.ndarray) -> np.ndarray:
    pts = np.argwhere(grid != BG)
    assert pts.size > 0, "output must contain foreground pixels"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return _crop_bbox(grid, (int(r0), int(c0), int(r1), int(c1)))


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coord_tuple = tuple(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coord_tuple,
                    "bbox": _bbox_from_coords(coord_tuple),
                    "area": len(coord_tuple),
                }
            )
    return comps


def _shape_signature(comp: dict) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = comp["bbox"]
    return tuple(sorted((r - r0, c - c0) for r, c in comp["coords"]))


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _center_manhattan_twice(
    bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]
) -> int:
    return abs((bbox_a[0] + bbox_a[2]) - (bbox_b[0] + bbox_b[2])) + abs(
        (bbox_a[1] + bbox_a[3]) - (bbox_b[1] + bbox_b[3])
    )


def _parse_scene(grid: np.ndarray) -> tuple[dict, list[dict], tuple[tuple[int, int], ...]]:
    comps = _find_components(grid)
    blue_components = [comp for comp in comps if comp["color"] == BLUE]
    assert len(blue_components) == 1, "expected exactly one blue template"
    template = blue_components[0]
    shape = _shape_signature(template)

    sprites = [comp for comp in comps if comp["color"] not in (BG, BLUE)]
    assert 2 <= len(sprites) <= 3, "expected two or three non-blue sprites"
    assert len(comps) == 1 + len(sprites), "scene should contain only the template and matching sprites"
    for sprite in sprites:
        assert _shape_signature(sprite) == shape, "all sprites must match the template shape"

    distances = [_center_manhattan_twice(sprite["bbox"], template["bbox"]) for sprite in sprites]
    assert len(set(distances)) == len(distances), "sprite distances to the template must be unique"
    return template, sprites, shape


def _aligned_coords(
    template: dict, sprite: dict, shape: tuple[tuple[int, int], ...], *, align: bool
) -> set[tuple[int, int]]:
    tr0, _, _, _ = template["bbox"]
    sr0, sc0, _, _ = sprite["bbox"]
    base_r = tr0 if align else sr0
    return {(base_r + dr, sc0 + dc) for dr, dc in shape}


def _compose(grid: np.ndarray, *, align: bool, ascending: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    template, sprites, shape = _parse_scene(g)
    ordered = sorted(
        sprites,
        key=lambda sprite: _center_manhattan_twice(sprite["bbox"], template["bbox"]),
        reverse=not ascending,
    )

    canvas = np.zeros_like(g)
    for r, c in template["coords"]:
        canvas[r, c] = BLUE

    for sprite in ordered:
        for r, c in _aligned_coords(template, sprite, shape, align=align):
            canvas[r, c] = sprite["color"]

    return _crop_non_bg(canvas)


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _horizontal_overlap_count(shape: tuple[tuple[int, int], ...], shift: int) -> int:
    base = set(shape)
    shifted = {(r, c + shift) for r, c in shape}
    return len(base & shifted)


def _make_shape(rng: np.random.Generator) -> tuple[tuple[int, int], ...]:
    for _ in range(200):
        target_area = int(rng.integers(5, 9))
        seed_h = int(rng.integers(3, 5))
        seed_w = int(rng.integers(3, 6))
        cells = {(int(rng.integers(seed_h)), int(rng.integers(seed_w)))}

        while len(cells) < target_area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < seed_h and 0 <= nc < seed_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])

        if len(cells) != target_area:
            continue

        shape = _normalize_shape(cells)
        sh, sw = _shape_size(shape)
        row_counts = [sum(1 for r, _ in shape if r == rr) for rr in range(sh)]
        if sh < 2 or sw < 2:
            continue
        if all(count < 2 for count in row_counts):
            continue
        if len(shape) == sh * sw:
            continue
        if _horizontal_overlap_count(shape, 1) == 0:
            continue
        return shape

    raise RuntimeError("failed to build a valid irregular shape")


def _paint_shape(
    grid: np.ndarray, top: int, left: int, shape: tuple[tuple[int, int], ...], color: int
) -> None:
    for dr, dc in shape:
        grid[top + dr, left + dc] = color


def _build_candidate(rng: np.random.Generator) -> np.ndarray:
    shape = _make_shape(rng)
    sh, sw = _shape_size(shape)
    n_sprites = 2 if rng.random() < 0.5 else 3

    pitch = sh + 2
    slot_count = n_sprites + 1
    base_h = slot_count * pitch + 2
    extra_h = int(rng.integers(0, min(4, 30 - base_h) + 1))
    h = base_h + extra_h
    top_margin = 1 + int(rng.integers(0, extra_h + 1))
    row_tops = [top_margin + i * pitch for i in range(slot_count)]
    template_slot = 1

    min_w = max(18, sw + 8)
    max_w = min(30, max(min_w, sw + 14))
    w = int(rng.integers(min_w, max_w + 1))
    template_left = int(rng.integers(3, w - sw - 2))

    if n_sprites == 2:
        signed_shift = int(rng.choice(np.array([-1, 1])))
        if rng.random() < 0.5:
            shift_by_slot = {0: 0, 2: signed_shift}
        else:
            shift_by_slot = {0: signed_shift, 2: 0}
    else:
        near_shift = int(rng.choice(np.array([-1, 1])))
        far_shift = int(rng.choice(np.array([-1, 1])))
        if rng.random() < 0.5:
            shift_by_slot = {0: 0, 2: near_shift, 3: far_shift}
        else:
            shift_by_slot = {0: near_shift, 2: 0, 3: far_shift}

    grid = np.zeros((h, w), dtype=int)
    _paint_shape(grid, row_tops[template_slot], template_left, shape, BLUE)

    sprite_slots = [slot for slot in range(slot_count) if slot != template_slot]
    colors = list(rng.choice(np.array(SPRITE_COLORS), size=n_sprites, replace=False))
    rng.shuffle(colors)
    for color, slot in zip(colors, sprite_slots):
        left = template_left + shift_by_slot[slot]
        _paint_shape(grid, row_tops[slot], left, shape, int(color))

    return grid


def _validate_generated(grid: np.ndarray) -> None:
    template, sprites, shape = _parse_scene(grid)
    aligned_sets = [
        {
            "color": sprite["color"],
            "distance": _center_manhattan_twice(sprite["bbox"], template["bbox"]),
            "coords": _aligned_coords(template, sprite, shape, align=True),
        }
        for sprite in sprites
    ]
    template_coords = set(template["coords"])

    overlaps_with_template = sum(1 for item in aligned_sets if item["coords"] & template_coords)
    assert overlaps_with_template >= 2, "at least two aligned sprites must overlap the template"

    pair_overlap = False
    for i in range(len(aligned_sets)):
        for j in range(i + 1, len(aligned_sets)):
            if aligned_sets[i]["coords"] & aligned_sets[j]["coords"]:
                pair_overlap = True
    assert pair_overlap, "aligned non-blue sprites must overlap one another"

    out = _compose(grid, align=True, ascending=True)
    reversed_order = _compose(grid, align=True, ascending=False)
    no_align = _compose(grid, align=False, ascending=True)

    assert not np.array_equal(out, reversed_order), "distance order must matter"
    assert not np.array_equal(out, no_align), "vertical alignment must matter"

    visible_colors = set(map(int, np.unique(out))) - {BG, BLUE}
    assert len(visible_colors) >= 2, "output must retain at least two non-blue colors"
    assert out.shape[0] < grid.shape[0] or out.shape[1] < grid.shape[1], "cropping must reduce the canvas"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Vertically align same-shape sprites to a blue template, then resolve overlaps by original distance before cropping.

    Input structure:
    - Black background with one blue 4-connected irregular template sprite near the upper middle.
    - Two or three other monochrome sprites of different colors share exactly the same shape as the template.
    - The non-blue sprites sit at different original heights and their column spans partially overlap the template span.

    Rule:
    - Measure each non-blue sprite's Manhattan distance to the blue template using bbox centers in the original input.
    - Keep each non-blue sprite's columns fixed, but replace every pixel row by the corresponding template row so all sprites become vertically aligned to the template.
    - Composite the blue template first, then draw aligned non-blue sprites from smaller original distance to larger original distance so the farthest sprite overwrites overlaps.
    - Crop the final composite to the minimal bbox containing all non-black pixels.

    Key insight:
    - The aligned coordinates decide where collisions happen, but the original pre-alignment distances decide which color wins each collision.
    """
    return _compose(grid, align=True, ascending=True)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate(rng)
        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
