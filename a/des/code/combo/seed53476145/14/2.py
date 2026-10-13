from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
FRAME_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_MIN = 18
GRID_MAX = 30
MAX_CHAINS = 3
MAX_FRAME_DEPTH = 4
MIN_DEEPEST_FRAME_SIZE = 5
CHAIN_GAP = 2
SPRITE_MIN_CELLS = 2
SPRITE_MAX_CELLS = 9
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _box_dims(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _frame_pixel_count(box: tuple[int, int, int, int]) -> int:
    h, w = _box_dims(box)
    return 2 * (h + w) - 4


def _box_area(box: tuple[int, int, int, int]) -> int:
    h, w = _box_dims(box)
    return h * w


def _is_rectangular_frame(mask: np.ndarray, coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h, w = _box_dims((r0, c0, r1, c1))
    if h < 3 or w < 3:
        return None
    if len(coords) != _frame_pixel_count((r0, c0, r1, c1)):
        return None

    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(sub[0, :]) or not np.all(sub[-1, :]):
        return None
    if not np.all(sub[:, 0]) or not np.all(sub[:, -1]):
        return None
    if h > 2 and w > 2 and np.any(sub[1:-1, 1:-1]):
        return None
    return r0, c0, r1, c1


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {BG, GREY}):
        for coords in _components(grid == color):
            box = _is_rectangular_frame(grid == color, coords)
            if box is None:
                continue
            frames.append({"color": int(color), "bbox": box})
    return frames


def _enclosing_frames(
    sprite_bbox: tuple[int, int, int, int], frames: list[dict[str, object]]
) -> list[dict[str, object]]:
    sr0, sc0, sr1, sc1 = sprite_bbox
    enclosing: list[dict[str, object]] = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        if r0 < sr0 and c0 < sc0 and sr1 < r1 and sc1 < c1:
            enclosing.append(frame)
    return enclosing


def _target_color(enclosing: list[dict[str, object]], color_counts: dict[int, int]) -> int:
    best = min(
        enclosing,
        key=lambda frame: (
            color_counts[int(frame["color"])],
            _frame_pixel_count(frame["bbox"]),
            _box_area(frame["bbox"]),
        ),
    )
    return int(best["color"])


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each grey sprite with the rarest color among all enclosing rectangular frames."""
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _find_frames(g)
    if not frames:
        return out

    color_counts = {int(color): int(np.count_nonzero(g == color)) for color in np.unique(g)}
    for sprite in _components(g == GREY):
        enclosing = _enclosing_frames(_bbox(sprite), frames)
        if not enclosing:
            continue
        color = _target_color(enclosing, color_counts)
        out[sprite[:, 0], sprite[:, 1]] = color
    return out


def _min_outer_size(depth: int) -> int:
    return MIN_DEEPEST_FRAME_SIZE + 4 * (depth - 1)


def _sample_depths(mode: str) -> list[int] | None:
    for _ in range(80):
        if mode == "single":
            n_chains = random.randint(1, MAX_CHAINS)
            return [1] * n_chains

        if mode == "mixed":
            n_chains = random.randint(2, MAX_CHAINS)
            depths = [1, random.randint(2, MAX_FRAME_DEPTH)]
            while len(depths) < n_chains:
                depths.append(random.randint(1, MAX_FRAME_DEPTH - 1))
            random.shuffle(depths)
            if sum(depths) <= len(FRAME_COLORS):
                return depths
            continue

        if mode == "nested":
            n_chains = random.randint(1, 2)
            depths = [random.randint(2, MAX_FRAME_DEPTH) for _ in range(n_chains)]
            if sum(depths) <= len(FRAME_COLORS):
                return depths
            continue

        raise ValueError(f"unknown mode: {mode}")
    return None


def _random_outer_dims(height: int, width: int, depth: int, n_chains: int) -> tuple[int, int] | None:
    min_dim = _min_outer_size(depth)
    slack = 5 if n_chains > 1 else 9
    max_h = min(height - 2, min_dim + slack)
    max_w = min(width - 2, min_dim + slack)
    if min_dim > max_h or min_dim > max_w:
        return None
    return random.randint(min_dim, max_h), random.randint(min_dim, max_w)


def _find_outer_location(occupied: np.ndarray, h: int, w: int, gap: int) -> tuple[int, int] | None:
    grid_h, grid_w = occupied.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, grid_h - h):
        for c0 in range(1, grid_w - w):
            rs = max(0, r0 - gap)
            re = min(grid_h, r0 + h + gap)
            cs = max(0, c0 - gap)
            ce = min(grid_w, c0 + w + gap)
            if not occupied[rs:re, cs:ce].any():
                positions.append((r0, c0))
    if not positions:
        return None
    return random.choice(positions)


def _reserve_outer_box(occupied: np.ndarray, box: tuple[int, int, int, int], gap: int) -> None:
    grid_h, grid_w = occupied.shape
    r0, c0, r1, c1 = box
    rs = max(0, r0 - gap)
    re = min(grid_h, r1 + gap + 1)
    cs = max(0, c0 - gap)
    ce = min(grid_w, c1 + gap + 1)
    occupied[rs:re, cs:ce] = True


def _build_nested_boxes(outer: tuple[int, int, int, int], depth: int) -> list[tuple[int, int, int, int]] | None:
    boxes = [outer]
    for remaining_depth in range(depth - 1, 0, -1):
        pr0, pc0, pr1, pc1 = boxes[-1]
        parent_h, parent_w = _box_dims(boxes[-1])
        min_child_dim = _min_outer_size(remaining_depth)
        max_child_h = parent_h - 4
        max_child_w = parent_w - 4
        if min_child_dim > max_child_h or min_child_dim > max_child_w:
            return None

        child_h = random.randint(min_child_dim, max_child_h)
        child_w = random.randint(min_child_dim, max_child_w)

        extra_h = parent_h - child_h
        extra_w = parent_w - child_w
        top_margin = random.randint(2, extra_h - 2)
        left_margin = random.randint(2, extra_w - 2)
        bottom_margin = extra_h - top_margin
        right_margin = extra_w - left_margin
        if bottom_margin < 2 or right_margin < 2:
            return None

        child = (
            pr0 + top_margin,
            pc0 + left_margin,
            pr1 - bottom_margin,
            pc1 - right_margin,
        )
        boxes.append(child)
    return boxes


def _paint_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _zone_masks(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for index, box in enumerate(boxes):
        r0, c0, r1, c1 = box
        zone = np.zeros(shape, dtype=bool)
        zone[r0 + 1 : r1, c0 + 1 : c1] = True
        if index + 1 < len(boxes):
            cr0, cc0, cr1, cc1 = boxes[index + 1]
            zone[cr0 : cr1 + 1, cc0 : cc1 + 1] = False
        masks.append(zone)
    return masks


def _grow_sprite(zone: np.ndarray) -> np.ndarray | None:
    available = [tuple(map(int, cell)) for cell in np.argwhere(zone)]
    if not available:
        return None

    max_cells = min(SPRITE_MAX_CELLS, len(available))
    min_cells = 1 if max_cells == 1 else min(SPRITE_MIN_CELLS, max_cells)
    for _ in range(40):
        target = random.randint(min_cells, max_cells)
        cells = {random.choice(available)}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < zone.shape[0] and 0 <= nc < zone.shape[1] and zone[nr, nc]:
                        if (nr, nc) not in cells:
                            frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) >= min_cells:
            return np.array(sorted(cells), dtype=int)
    return None


def _generate_case() -> np.ndarray | None:
    grid_h = random.randint(GRID_MIN, GRID_MAX)
    grid_w = random.randint(GRID_MIN, GRID_MAX)
    mode = random.choice(["single", "mixed", "nested"])
    depths = _sample_depths(mode)
    if depths is None:
        return None

    colors = random.sample(FRAME_COLORS, sum(depths))
    chain_specs = []
    color_index = 0
    for depth in depths:
        dims = _random_outer_dims(grid_h, grid_w, depth, len(depths))
        if dims is None:
            return None
        frame_colors = colors[color_index : color_index + depth]
        color_index += depth
        chain_specs.append(
            {
                "depth": depth,
                "outer_h": dims[0],
                "outer_w": dims[1],
                "colors": frame_colors,
            }
        )

    chain_specs.sort(key=lambda spec: spec["outer_h"] * spec["outer_w"], reverse=True)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    chains: list[dict[str, object]] = []
    for spec in chain_specs:
        pos = _find_outer_location(occupied, spec["outer_h"], spec["outer_w"], CHAIN_GAP)
        if pos is None:
            return None
        r0, c0 = pos
        outer = (r0, c0, r0 + spec["outer_h"] - 1, c0 + spec["outer_w"] - 1)
        boxes = _build_nested_boxes(outer, spec["depth"])
        if boxes is None:
            return None
        _reserve_outer_box(occupied, outer, CHAIN_GAP)
        chains.append({"boxes": boxes, "colors": spec["colors"]})

    zone_data = []
    for chain in chains:
        masks = _zone_masks((grid_h, grid_w), chain["boxes"])
        any_zones = [i for i, mask in enumerate(masks) if int(mask.sum()) >= SPRITE_MIN_CELLS]
        deep_zones = [i for i in any_zones if i > 0]
        if not any_zones:
            return None
        zone_data.append((masks, any_zones, deep_zones))

    chosen_zones = [random.choice(any_zones) for _masks, any_zones, _deep_zones in zone_data]
    if mode != "single" and not any(zone > 0 for zone in chosen_zones):
        candidates = [i for i, (_masks, _any, deep) in enumerate(zone_data) if deep]
        if not candidates:
            return None
        idx = random.choice(candidates)
        chosen_zones[idx] = random.choice(zone_data[idx][2])

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    color_counts: dict[int, int] = {}
    for chain in chains:
        for box, color in zip(chain["boxes"], chain["colors"]):
            _paint_frame(grid, box, color)
            color_counts[int(color)] = color_counts.get(int(color), 0) + _frame_pixel_count(box)

    sprite_targets: list[tuple[np.ndarray, int]] = []
    for chain, zone_index, (masks, _any_zones, _deep_zones) in zip(chains, chosen_zones, zone_data):
        sprite = _grow_sprite(masks[zone_index])
        if sprite is None:
            return None
        grid[sprite[:, 0], sprite[:, 1]] = GREY

        enclosing = [
            {"color": int(color), "bbox": box}
            for box, color in zip(chain["boxes"][: zone_index + 1], chain["colors"][: zone_index + 1])
        ]
        sprite_targets.append((sprite, _target_color(enclosing, color_counts)))

    expected = grid.copy()
    for sprite, color in sprite_targets:
        expected[sprite[:, 0], sprite[:, 1]] = color

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
