from __future__ import annotations

import random

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ORIENTATION_ORDER = ("up", "left", "right", "down")
PALETTE = tuple(range(10))
ARC_MAX_SIDE = 30

# Generator ranges.
OBJECT_COUNT_RANGE = (4, 6)
GROUP_COUNT_CHOICES = (2, 3, 4)
BBOX_SIDE_RANGE = (3, 5)
BAR_THICKNESS_RANGE = (1, 2)
PLACEMENT_MARGIN = 1
GRID_SIDE_RANGE = (14, 28)
GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 100
PLACEMENT_ATTEMPTS = 200


def _bg_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _edge_counts(mask: np.ndarray) -> dict[str, int]:
    return {
        "up": int(mask[0, :].sum()),
        "down": int(mask[-1, :].sum()),
        "left": int(mask[:, 0].sum()),
        "right": int(mask[:, -1].sum()),
    }


def _opening_direction(mask: np.ndarray) -> str | None:
    counts = _edge_counts(mask)
    best = min(counts.values())
    winners = [name for name, value in counts.items() if value == best]
    if len(winners) != 1:
        return None
    return winners[0]


def _crop_patch(grid: np.ndarray, color: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    mask = grid == int(color)
    r0, c0, r1, c1 = _bbox(mask)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), _bg_color(grid), dtype=int)
    patch[mask[r0 : r1 + 1, c0 : c1 + 1]] = int(color)
    return patch, (r0, c0, r1, c1)


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    objects: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        patch, box = _crop_patch(g, color)
        direction = _opening_direction(patch == color)
        assert direction is not None, "each object must have a unique opening direction"
        objects.append(
            {
                "color": int(color),
                "bbox": box,
                "patch": patch,
                "direction": direction,
            }
        )

    assert OBJECT_COUNT_RANGE[0] <= len(objects) <= OBJECT_COUNT_RANGE[1]
    objects.sort(key=lambda obj: (int(obj["bbox"][1]), int(obj["bbox"][0])))
    return objects


def _paste(canvas: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    window = canvas[top : top + h, left : left + w]
    mask = patch != bg
    window[mask] = patch[mask]


def _pack_group(objects: list[dict], direction: str, bg: int) -> np.ndarray | None:
    if not objects:
        return None

    patches = [obj["patch"] for obj in objects]
    if direction in {"up", "down"}:
        height = max(int(p.shape[0]) for p in patches)
        width = sum(int(p.shape[1]) for p in patches)
        out = np.full((height, width), bg, dtype=int)
        col = 0
        for patch in patches:
            _paste(out, patch, 0, col, bg)
            col += int(patch.shape[1])
        return out

    height = sum(int(p.shape[0]) for p in patches)
    width = max(int(p.shape[1]) for p in patches)
    out = np.full((height, width), bg, dtype=int)
    row = 0
    for patch in patches:
        _paste(out, patch, row, 0, bg)
        row += int(patch.shape[0])
    return out


def _group_composites(objects: list[dict], bg: int) -> dict[str, np.ndarray | None]:
    by_direction = {name: [] for name in DIRS}
    for obj in objects:
        by_direction[obj["direction"]].append(obj)
    return {name: _pack_group(by_direction[name], name, bg) for name in DIRS}


def _assemble(composites: dict[str, np.ndarray | None], bg: int) -> np.ndarray:
    up = composites["up"]
    down = composites["down"]
    left = composites["left"]
    right = composites["right"]

    up_h = 0 if up is None else int(up.shape[0])
    up_w = 0 if up is None else int(up.shape[1])
    down_h = 0 if down is None else int(down.shape[0])
    down_w = 0 if down is None else int(down.shape[1])
    left_h = 0 if left is None else int(left.shape[0])
    left_w = 0 if left is None else int(left.shape[1])
    right_h = 0 if right is None else int(right.shape[0])
    right_w = 0 if right is None else int(right.shape[1])

    middle_present = left is not None or right is not None
    middle_h = max(left_h, right_h) if middle_present else 0
    if left is not None and right is not None:
        middle_w = left_w + 1 + right_w
    else:
        middle_w = max(left_w, right_w)

    band_heights = [h for h in (up_h, middle_h, down_h) if h > 0]
    total_h = sum(band_heights) + max(0, len(band_heights) - 1)
    total_w = max(up_w, middle_w, down_w)
    out = np.full((total_h, total_w), bg, dtype=int)

    row = 0
    if up is not None:
        _paste(out, up, row, (total_w - up_w) // 2, bg)
        row += up_h
        if middle_present or down is not None:
            row += 1

    if middle_present:
        if left is not None:
            left_top = row + (middle_h - left_h) // 2
            _paste(out, left, left_top, 0, bg)
        if right is not None:
            right_top = row + (middle_h - right_h) // 2
            _paste(out, right, right_top, total_w - right_w, bg)
        row += middle_h
        if down is not None:
            row += 1

    if down is not None:
        _paste(out, down, row, (total_w - down_w) // 2, bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group open-sided objects by opening direction and rebuild them into four directional composites.

    Input structure:
    - The most common color is the background.
    - The foreground contains 4-6 separated monochrome objects, each shaped like a thick rectangular frame missing one side.
    - Different objects use different colors, and each opening faces up, down, left, or right.

    Rule:
    - Crop every object to its tight bounding box and identify the opening direction from the sparsest box edge.
    - Sort the objects by their input positions from left to right, then top to bottom, and group them by opening direction.
    - Pack up/down groups horizontally and left/right groups vertically, keeping each cropped object unchanged and leaving no gap inside a group.
    - Place the resulting group composites on the smallest output canvas with one background gap between occupied bands: up at the top, down at the bottom, left at the left, and right at the right.

    Key insight:
    The visible cue is geometric opening direction rather than color or size, and that direction determines both the grouping and the packing axis of each composite.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    objects = _extract_objects(g)
    composites = _group_composites(objects, bg)
    return _assemble(composites, bg)


def _random_shape(direction: str, color: int, bg: int) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(*BBOX_SIDE_RANGE)
        w = random.randint(*BBOX_SIDE_RANGE)
        t1 = random.randint(*BAR_THICKNESS_RANGE)
        t2 = random.randint(*BAR_THICKNESS_RANGE)
        t3 = random.randint(*BAR_THICKNESS_RANGE)

        mask = np.zeros((h, w), dtype=bool)
        if direction == "up":
            mask[:, : min(t1, w)] = True
            mask[:, max(0, w - t2) :] = True
            mask[max(0, h - t3) :, :] = True
        elif direction == "down":
            mask[:, : min(t1, w)] = True
            mask[:, max(0, w - t2) :] = True
            mask[: min(t3, h), :] = True
        elif direction == "left":
            mask[: min(t1, h), :] = True
            mask[max(0, h - t2) :, :] = True
            mask[:, max(0, w - t3) :] = True
        else:
            mask[: min(t1, h), :] = True
            mask[max(0, h - t2) :, :] = True
            mask[:, : min(t3, w)] = True

        if not np.any(mask) or int(mask.sum()) == h * w:
            continue
        if _opening_direction(mask) != direction:
            continue

        patch = np.full((h, w), bg, dtype=int)
        patch[mask] = int(color)
        return patch

    raise RuntimeError(f"failed to generate a {direction} object")


def _sample_directions(count: int) -> list[str]:
    group_count = random.choice([k for k in GROUP_COUNT_CHOICES if k <= count])
    groups = random.sample(list(DIRS), group_count)
    directions = groups[:]
    while len(directions) < count:
        directions.append(random.choice(groups))
    random.shuffle(directions)
    return directions


def _placement_rects(placed: list[tuple[int, int, int, int]], pad: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r0, c0, r1, c1 in placed:
        for rr in range(r0 - pad, r1 + pad + 1):
            for cc in range(c0 - pad, c1 + pad + 1):
                cells.add((rr, cc))
    return cells


def _place_patches(patches: list[np.ndarray], bg: int) -> np.ndarray | None:
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    total_h = sum(int(p.shape[0]) for p in patches)
    total_w = sum(int(p.shape[1]) for p in patches)
    min_h = max(GRID_SIDE_RANGE[0], max_h + 2, total_h // 2 + 2)
    min_w = max(GRID_SIDE_RANGE[0], max_w + 2, total_w // 2 + 2)
    if min_h > ARC_MAX_SIDE or min_w > ARC_MAX_SIDE:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        height = random.randint(min_h, min(GRID_SIDE_RANGE[1], ARC_MAX_SIDE))
        width = random.randint(min_w, min(GRID_SIDE_RANGE[1], ARC_MAX_SIDE))
        grid = np.full((height, width), bg, dtype=int)
        occupied = set()
        boxes: list[tuple[int, int, int, int]] = []
        order = list(range(len(patches)))
        random.shuffle(order)

        success = True
        for idx in order:
            patch = patches[idx]
            ph, pw = map(int, patch.shape)
            candidates = [
                (r, c)
                for r in range(0, height - ph + 1)
                for c in range(0, width - pw + 1)
            ]
            random.shuffle(candidates)
            chosen = None
            for r0, c0 in candidates:
                r1 = r0 + ph - 1
                c1 = c0 + pw - 1
                blocked = False
                for rr in range(r0 - PLACEMENT_MARGIN, r1 + PLACEMENT_MARGIN + 1):
                    for cc in range(c0 - PLACEMENT_MARGIN, c1 + PLACEMENT_MARGIN + 1):
                        if (rr, cc) in occupied:
                            blocked = True
                            break
                    if blocked:
                        break
                if not blocked:
                    chosen = (r0, c0)
                    break

            if chosen is None:
                success = False
                break

            r0, c0 = chosen
            _paste(grid, patch, r0, c0, bg)
            boxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
            occupied = _placement_rects(boxes, PLACEMENT_MARGIN)

        if not success:
            continue
        return grid

    return None


def _foreground_is_sparse(grid: np.ndarray) -> bool:
    bg = _bg_color(grid)
    return int(np.sum(grid != bg)) * 2 < int(grid.size)


def _valid_sample(grid: np.ndarray) -> bool:
    try:
        objects = _extract_objects(grid)
        out = main(grid)
    except Exception:
        return False

    if max(grid.shape) > ARC_MAX_SIDE or max(out.shape) > ARC_MAX_SIDE:
        return False
    if not _foreground_is_sparse(grid):
        return False
    if len({obj["direction"] for obj in objects}) < 2:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        bg = random.choice(PALETTE)
        count = random.randint(*OBJECT_COUNT_RANGE)
        colors = random.sample([c for c in PALETTE if c != bg], count)
        directions = _sample_directions(count)
        patches = [_random_shape(direction, color, bg) for direction, color in zip(directions, colors)]
        grid = _place_patches(patches, bg)
        if grid is None:
            continue
        if _valid_sample(grid):
            return grid
    raise RuntimeError("failed to generate a valid input grid")
