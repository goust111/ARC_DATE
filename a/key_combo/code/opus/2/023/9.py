from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


COLOR_TO_EDGE = {
    1: "left",
    2: "right",
    3: "top",
    4: "bottom",
}
EDGE_ORDER = ("left", "right", "top", "bottom")
SPRITE_SIZE = 3
PADDED_SIZE = 5
BG = 0


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _boxes_touch_or_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    *,
    padding: int = 1,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + padding < br0
        or br1 + padding < ar0
        or ac1 + padding < bc0
        or bc1 + padding < ac0
    )


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(coords)


def _find_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    seen = np.zeros(g.shape, dtype=bool)
    comps: list[dict] = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color not in colors or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(len(cells)),
                }
            )
    return comps


def _build_sprite_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1, 1 << 9):
        if bits == (1 << 9) - 1:
            continue
        flat = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool)
        mask = flat.reshape(3, 3)
        coords = np.argwhere(mask)
        if len(coords) < 3:
            continue
        if _bbox(coords) != (0, 0, 2, 2):
            continue
        if not _mask_connected(mask):
            continue
        masks.append(mask.astype(int))
    return masks


SPRITE_MASKS = _build_sprite_masks()


def _sprite_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=int)
    patch[mask.astype(bool)] = int(color)
    return patch


def _edge_pad_sprite(sprite: np.ndarray) -> np.ndarray:
    return np.pad(np.asarray(sprite, dtype=int), ((1, 1), (1, 1)), mode="edge")


def _distance_to_target_edge(sprite: dict, shape: tuple[int, int]) -> int:
    h, w = shape
    r0, c0, r1, c1 = sprite["bbox"]
    edge = COLOR_TO_EDGE[int(sprite["color"])]
    if edge == "left":
        return int(c0)
    if edge == "right":
        return int(w - 1 - c1)
    if edge == "top":
        return int(r0)
    return int(h - 1 - r1)


def _base_destination(sprite: dict, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    r0, c0, _r1, _c1 = sprite["bbox"]
    edge = COLOR_TO_EDGE[int(sprite["color"])]
    if edge == "left":
        return int(r0 - 1), 0
    if edge == "right":
        return int(r0 - 1), int(w - PADDED_SIZE)
    if edge == "top":
        return 0, int(c0 - 1)
    return int(h - PADDED_SIZE), int(c0 - 1)


def _parallel_span(edge: str, dest_r: int, dest_c: int) -> tuple[int, int]:
    if edge in {"left", "right"}:
        return int(dest_r), int(dest_r + PADDED_SIZE - 1)
    return int(dest_c), int(dest_c + PADDED_SIZE - 1)


def _sort_key(sprite: dict, shape: tuple[int, int]) -> tuple[int, int, int]:
    dist = _distance_to_target_edge(sprite, shape)
    r0, c0, _r1, _c1 = sprite["bbox"]
    return dist, int(r0), int(c0)


def _compute_placements(sprites: list[dict], shape: tuple[int, int]) -> dict[str, list[dict]]:
    h, w = shape
    by_edge: dict[str, list[dict]] = {edge: [] for edge in EDGE_ORDER}
    for sprite in sprites:
        by_edge[COLOR_TO_EDGE[int(sprite["color"])]].append(sprite)

    placed_by_edge: dict[str, list[dict]] = {edge: [] for edge in EDGE_ORDER}
    for edge in EDGE_ORDER:
        occupied_spans: list[tuple[int, int]] = []
        for sprite in sorted(by_edge[edge], key=lambda item: _sort_key(item, shape)):
            dest_r, dest_c = _base_destination(sprite, shape)
            span0, span1 = _parallel_span(edge, dest_r, dest_c)
            if any(_intervals_overlap(span0, span1, prev0, prev1) for prev0, prev1 in occupied_spans):
                if edge == "left":
                    dest_c += PADDED_SIZE
                elif edge == "right":
                    dest_c -= PADDED_SIZE
                elif edge == "top":
                    dest_r += PADDED_SIZE
                else:
                    dest_r -= PADDED_SIZE
            assert 0 <= dest_r <= h - PADDED_SIZE
            assert 0 <= dest_c <= w - PADDED_SIZE
            occupied_spans.append(_parallel_span(edge, dest_r, dest_c))
            placed_by_edge[edge].append(
                {
                    "color": int(sprite["color"]),
                    "bbox": sprite["bbox"],
                    "patch5": np.asarray(sprite["patch5"], dtype=int),
                    "dest": (int(dest_r), int(dest_c)),
                    "distance": _distance_to_target_edge(sprite, shape),
                }
            )
    return placed_by_edge


def _render_placed(placed_by_edge: dict[str, list[dict]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for edge in EDGE_ORDER:
        for sprite in placed_by_edge[edge]:
            r0, c0 = sprite["dest"]
            out[r0 : r0 + PADDED_SIZE, c0 : c0 + PADDED_SIZE] = sprite["patch5"]
    return out


def _sprites_from_grid(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = _find_components(g, {1, 2, 3, 4})
    assert 4 <= len(comps) <= 6, "expected 4 to 6 isolated sprites"

    sprites: list[dict] = []
    seen_boxes: list[tuple[int, int, int, int]] = []
    counts = Counter()
    for comp in comps:
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "each sprite bbox must be exactly 3x3"
        patch3 = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert set(map(int, np.unique(patch3))) <= {BG, color}, "sprite must be monochrome within its 3x3 box"
        assert np.any(patch3 == BG), "sprite must contain at least one black cell"
        assert np.any(patch3 == color), "sprite cannot be empty"
        box = (int(r0), int(c0), int(r1), int(c1))
        for prev in seen_boxes:
            assert not _boxes_touch_or_overlap(box, prev, padding=1), "sprite 3x3 boxes must be separated"
        seen_boxes.append(box)
        counts[color] += 1
        sprites.append(
            {
                "color": color,
                "bbox": box,
                "patch3": patch3,
                "patch5": _edge_pad_sprite(patch3),
            }
        )

    assert all(counts[color] >= 1 for color in COLOR_TO_EDGE), "each edge color must appear at least once"
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each 3x3 monochrome sprite by edge padding, then move it to the color-chosen grid edge with near-edge-first inward collision handling.

    Input structure:
    - Black background with 4 to 6 isolated monochrome sprites.
    - Each sprite occupies an exactly 3x3 bounding box, uses one color from blue/red/green/yellow, and contains at least one black cell inside that box.
    - Sprite 3x3 boxes are separated from each other by at least one blank cell.

    Rule:
    - For every sprite, take its full 3x3 box and perform one-cell edge padding, producing a 5x5 patch by copying the top/bottom rows and left/right columns outward.
    - Move blue sprites to the left edge, red to the right edge, green to the top edge, and yellow to the bottom edge, while keeping the original center coordinate on the non-moving axis.
    - Within each color group, process sprites by increasing original distance from their target edge; if a sprite's 5x5 row/column span overlaps any already placed sprite on that same edge, shift it one sprite-width inward.
    - Different edges do not block one another; render edge groups in left, then right, then top, then bottom order, with later groups overwriting earlier pixels.

    Key insight:
    - The initial edge-padding step is not cosmetic: it expands each sprite from 3x3 to 5x5, which changes the row/column overlap test and therefore changes which farther sprites are pushed inward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sprites = _sprites_from_grid(g)
    placed = _compute_placements(sprites, g.shape)
    return _render_placed(placed, g.shape)


GRID_SIDE_RANGE = (18, 26)
EXTRA_DUPLICATES_CHOICES = (1, 2)
PAIR_SAMPLE_ATTEMPTS = 200
SINGLE_SAMPLE_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400


def _choose_color_counts() -> tuple[dict[int, int], dict[int, bool]]:
    extra = random.choice(EXTRA_DUPLICATES_CHOICES)
    duplicated = random.sample(list(COLOR_TO_EDGE), extra)
    counts = {color: 1 for color in COLOR_TO_EDGE}
    for color in duplicated:
        counts[color] += 1

    collision_plan = {color: False for color in COLOR_TO_EDGE}
    if extra == 1:
        collision_plan[duplicated[0]] = random.random() < 0.7
    else:
        guaranteed = random.choice(duplicated)
        collision_plan[guaranteed] = True
        for color in duplicated:
            if color != guaranteed:
                collision_plan[color] = random.random() < 0.5
    return counts, collision_plan


def _random_mask() -> np.ndarray:
    return random.choice(SPRITE_MASKS)


def _box_from_top_left(r0: int, c0: int) -> tuple[int, int, int, int]:
    return int(r0), int(c0), int(r0 + 2), int(c0 + 2)


def _can_place(box: tuple[int, int, int, int], boxes: list[tuple[int, int, int, int]]) -> bool:
    return all(not _boxes_touch_or_overlap(box, other, padding=1) for other in boxes)


def _parallel_collision_flag(
    color: int,
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    if color in {1, 2}:
        return _intervals_overlap(ar0 - 1, ar1 + 1, br0 - 1, br1 + 1)
    return _intervals_overlap(ac0 - 1, ac1 + 1, bc0 - 1, bc1 + 1)


def _place_pair(
    color: int,
    want_collision: bool,
    h: int,
    w: int,
    boxes: list[tuple[int, int, int, int]],
) -> list[dict] | None:
    horizontal_edge = color in {1, 2}
    row_values = list(range(1, h - 3))
    col_values = list(range(1, w - 3))

    for _ in range(PAIR_SAMPLE_ATTEMPTS):
        if horizontal_edge:
            gap = 4 if want_collision else random.randint(5, max(5, h - 5))
            if gap > (h - 5):
                continue
            low = random.randint(1, h - 4 - gap)
            rows = [low, low + gap]
            random.shuffle(rows)
            c0, c1 = random.sample(col_values, 2)
            top_lefts = [(rows[0], c0), (rows[1], c1)]
        else:
            gap = 4 if want_collision else random.randint(5, max(5, w - 5))
            if gap > (w - 5):
                continue
            low = random.randint(1, w - 4 - gap)
            cols = [low, low + gap]
            random.shuffle(cols)
            r0, r1 = random.sample(row_values, 2)
            top_lefts = [(r0, cols[0]), (r1, cols[1])]

        pair_boxes = [_box_from_top_left(r, c) for r, c in top_lefts]
        if not _can_place(pair_boxes[0], boxes):
            continue
        if not _can_place(pair_boxes[1], boxes + [pair_boxes[0]]):
            continue

        if horizontal_edge:
            assert pair_boxes[0][1] != pair_boxes[1][1]
        else:
            assert pair_boxes[0][0] != pair_boxes[1][0]

        actual_collision = _parallel_collision_flag(color, pair_boxes[0], pair_boxes[1])
        if actual_collision != want_collision:
            continue

        return [
            {
                "color": color,
                "bbox": pair_boxes[0],
                "patch3": _sprite_patch(_random_mask(), color),
            },
            {
                "color": color,
                "bbox": pair_boxes[1],
                "patch3": _sprite_patch(_random_mask(), color),
            },
        ]
    return None


def _place_single(
    color: int,
    h: int,
    w: int,
    boxes: list[tuple[int, int, int, int]],
) -> dict | None:
    positions = [(r, c) for r in range(1, h - 3) for c in range(1, w - 3)]
    random.shuffle(positions)
    for r0, c0 in positions[:SINGLE_SAMPLE_ATTEMPTS]:
        box = _box_from_top_left(r0, c0)
        if _can_place(box, boxes):
            return {
                "color": color,
                "bbox": box,
                "patch3": _sprite_patch(_random_mask(), color),
            }
    return None


def _spec_to_sprite(spec: dict) -> dict:
    return {
        "color": int(spec["color"]),
        "bbox": tuple(map(int, spec["bbox"])),
        "patch3": np.asarray(spec["patch3"], dtype=int),
        "patch5": _edge_pad_sprite(np.asarray(spec["patch3"], dtype=int)),
    }


def _render_input(specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    shuffled = specs[:]
    random.shuffle(shuffled)
    for spec in shuffled:
        r0, c0, r1, c1 = spec["bbox"]
        patch3 = np.asarray(spec["patch3"], dtype=int)
        grid[r0 : r1 + 1, c0 : c1 + 1] = patch3
    return grid


def _validate_instance(
    grid: np.ndarray,
    specs: list[dict],
    collision_plan: dict[int, bool],
) -> bool:
    sprites = [_spec_to_sprite(spec) for spec in specs]
    boxes = [tuple(spec["bbox"]) for spec in specs]

    if any(_boxes_touch_or_overlap(a, b, padding=1) for i, a in enumerate(boxes) for b in boxes[i + 1 :]):
        return False

    counts = Counter(int(spec["color"]) for spec in specs)
    if set(counts) != set(COLOR_TO_EDGE):
        return False
    if any(count not in {1, 2} for count in counts.values()):
        return False

    by_color: dict[int, list[dict]] = defaultdict(list)
    for spec in specs:
        by_color[int(spec["color"])].append(spec)

    for color, items in by_color.items():
        if len(items) != 2:
            continue
        box_a = tuple(items[0]["bbox"])
        box_b = tuple(items[1]["bbox"])
        if color in {1, 2}:
            if _intervals_overlap(box_a[0], box_a[2], box_b[0], box_b[2]):
                return False
            if box_a[1] == box_b[1]:
                return False
        else:
            if _intervals_overlap(box_a[1], box_a[3], box_b[1], box_b[3]):
                return False
            if box_a[0] == box_b[0]:
                return False
        if _parallel_collision_flag(color, box_a, box_b) != collision_plan[color]:
            return False

    detected = _sprites_from_grid(grid)
    normalized_detected = sorted(
        (int(sprite["color"]), tuple(map(int, sprite["bbox"])), tuple(map(int, sprite["patch3"].ravel())))
        for sprite in detected
    )
    normalized_specs = sorted(
        (int(sprite["color"]), tuple(map(int, sprite["bbox"])), tuple(map(int, sprite["patch3"].ravel())))
        for sprite in sprites
    )
    if normalized_detected != normalized_specs:
        return False

    placed = _compute_placements(sprites, grid.shape)
    for color, items in by_color.items():
        if len(items) != 2:
            continue
        edge = COLOR_TO_EDGE[color]
        placed_group = placed[edge]
        if len(placed_group) != 2:
            return False
        ordered = sorted(placed_group, key=lambda item: (item["distance"], item["bbox"][0], item["bbox"][1]))
        first_dest = ordered[0]["dest"]
        second_dest = ordered[1]["dest"]
        if edge == "left":
            if first_dest[1] != 0:
                return False
            if second_dest[1] != (PADDED_SIZE if collision_plan[color] else 0):
                return False
        elif edge == "right":
            expected = grid.shape[1] - PADDED_SIZE
            if first_dest[1] != expected:
                return False
            if second_dest[1] != (expected - PADDED_SIZE if collision_plan[color] else expected):
                return False
        elif edge == "top":
            if first_dest[0] != 0:
                return False
            if second_dest[0] != (PADDED_SIZE if collision_plan[color] else 0):
                return False
        else:
            expected = grid.shape[0] - PADDED_SIZE
            if first_dest[0] != expected:
                return False
            if second_dest[0] != (expected - PADDED_SIZE if collision_plan[color] else expected):
                return False

    expected = _render_placed(placed, grid.shape)
    actual = main(grid)
    return np.array_equal(actual, expected)


def _generate_once() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    counts, collision_plan = _choose_color_counts()

    specs: list[dict] = []
    boxes: list[tuple[int, int, int, int]] = []
    duplicated = [color for color, count in counts.items() if count == 2]
    singles = [color for color, count in counts.items() if count == 1]
    random.shuffle(duplicated)
    random.shuffle(singles)

    for color in duplicated:
        pair = _place_pair(color, collision_plan[color], h, w, boxes)
        if pair is None:
            return None
        for spec in pair:
            boxes.append(tuple(spec["bbox"]))
            specs.append(spec)

    for color in singles:
        spec = _place_single(color, h, w, boxes)
        if spec is None:
            return None
        boxes.append(tuple(spec["bbox"]))
        specs.append(spec)

    grid = _render_input(specs, (h, w))
    if not _validate_instance(grid, specs, collision_plan):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
