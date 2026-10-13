import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

GRID_LIMIT = 30
GEN_RETRIES = 400
SHAPE_RETRIES = 200
PLACE_RETRIES = 250


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _color_components(grid: np.ndarray, color: int, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    H, W = grid.shape
    dirs = DIR8 if connectivity == 8 else DIR4
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _sprite_patch(coords: list[tuple[int, int]], color: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    assert side_h == side_w and side_h % 2 == 1 and side_h >= 3
    patch = np.zeros((side_h, side_w), dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = color
    return (r0, c0, r1, c1), patch


def _match_marker(grid: np.ndarray, marker: tuple[int, int], sprite: dict) -> tuple[int, bool] | None:
    r, c = marker
    r0, c0, r1, c1 = sprite["bbox"]
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2

    if r == center_r:
        if c < c0:
            gap = c0 - c - 1
            if 1 <= gap <= 3 and np.all(grid[r, c + 1 : c0] == 0):
                return gap, True
        if c > c1:
            gap = c - c1 - 1
            if 1 <= gap <= 3 and np.all(grid[r, c1 + 1 : c] == 0):
                return gap, True

    if c == center_c:
        if r < r0:
            gap = r0 - r - 1
            if 1 <= gap <= 3 and np.all(grid[r + 1 : r0, c] == 0):
                return gap, False
        if r > r1:
            gap = r - r1 - 1
            if 1 <= gap <= 3 and np.all(grid[r1 + 1 : r, c] == 0):
                return gap, False

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each monochrome sprite in place using the gap to its same-color singleton marker, then erase the markers.

    Input structure:
    - The background is black.
    - Each foreground color forms one or more odd-sized square-bbox sprites plus equally colored isolated single-pixel markers.
    - Every marker lies on the center row or center column of its matched sprite bbox, separated by 1 to 3 black cells.

    Rule:
    - For each isolated marker, measure the number of black cells between the marker and the matched sprite bbox edge.
    - If the marker is left or right of the sprite, rotate that sprite clockwise by that many quarter turns.
    - If the marker is above or below the sprite, rotate that sprite counterclockwise by that many quarter turns.
    - Keep every sprite in its original bbox and remove all marker pixels.

    Key insight:
    - The singleton pixel is not moved; its axis gives rotation direction and its gap to the sprite gives rotation count.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    marker_cells: list[tuple[int, int]] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) != 0):
        comps = _color_components(g, color, connectivity=8)
        sprites = []
        markers = []
        for comp in comps:
            if len(comp) == 1:
                markers.append(comp[0])
            else:
                bbox, patch = _sprite_patch(comp, color)
                sprites.append({"bbox": bbox, "patch": patch, "color": color})

        assert sprites, "expected at least one sprite per foreground color"
        assert len(sprites) == len(markers), "expected one marker per sprite"

        used_sprites = set()
        for marker in markers:
            matches = []
            for idx, sprite in enumerate(sprites):
                info = _match_marker(g, marker, sprite)
                if info is not None:
                    matches.append((idx, info))
            assert len(matches) == 1, "marker-to-sprite relation must be unique"

            idx, (turns, clockwise) = matches[0]
            assert idx not in used_sprites, "each sprite must use exactly one marker"
            used_sprites.add(idx)

            sprite = sprites[idx]
            r0, c0, r1, c1 = sprite["bbox"]
            k = (-turns) % 4 if clockwise else turns % 4
            rotated = np.rot90(sprite["patch"], k)
            out[r0 : r1 + 1, c0 : c1 + 1] = rotated
            marker_cells.append(marker)

        assert len(used_sprites) == len(sprites)

    for r, c in marker_cells:
        out[r, c] = 0
    return out


def _random_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    rr, cc = end
    cells = [(r, c)]
    while (r, c) != (rr, cc):
        steps = []
        if r < rr:
            steps.append((1, 0))
        elif r > rr:
            steps.append((-1, 0))
        if c < cc:
            steps.append((0, 1))
        elif c > cc:
            steps.append((0, -1))
        dr, dc = random.choice(steps)
        r += dr
        c += dc
        cells.append((r, c))
    return cells


def _generate_mask(side: int) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        center = (random.randrange(side), random.randrange(side))
        anchors = {
            (0, random.randrange(side)),
            (side - 1, random.randrange(side)),
            (random.randrange(side), 0),
            (random.randrange(side), side - 1),
        }

        cells = {center}
        for anchor in anchors:
            cells.update(_random_path(center, anchor))

        min_area = max(len(cells) + 1, side + 2, int(np.ceil(side * side * 0.35)))
        max_area = min(side * side - 1, int(np.ceil(side * side * 0.7)))
        if min_area > max_area:
            continue
        target_area = random.randint(min_area, max_area)

        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        mask = np.zeros((side, side), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        return mask

    raise RuntimeError(f"failed to build asymmetric {side}x{side} mask")


def _marker_relative(side: int, direction: str, gap: int) -> tuple[int, int]:
    center = side // 2
    if direction == "left":
        return center, -gap - 1
    if direction == "right":
        return center, side + gap
    if direction == "up":
        return -gap - 1, center
    if direction == "down":
        return side + gap, center
    raise ValueError(direction)


def _reserved_rect(side: int, direction: str, gap: int) -> tuple[int, int, int, int]:
    mr, mc = _marker_relative(side, direction, gap)
    return min(0, mr) - 1, min(0, mc) - 1, max(side - 1, mr) + 1, max(side - 1, mc) + 1


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _paint_object(
    canvas: np.ndarray,
    top: int,
    left: int,
    mask: np.ndarray,
    color: int,
    direction: str,
    gap: int,
) -> None:
    side = mask.shape[0]
    patch = np.zeros((side, side), dtype=int)
    patch[mask] = color
    canvas[top : top + side, left : left + side] = np.maximum(canvas[top : top + side, left : left + side], patch)
    mr, mc = _marker_relative(side, direction, gap)
    canvas[top + mr, left + mc] = color


def _crop_canvas(canvas: np.ndarray) -> np.ndarray:
    coords = np.argwhere(canvas != 0)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_margin = random.randint(0, min(3, int(r0)))
    left_margin = random.randint(0, min(3, int(c0)))
    bottom_margin = random.randint(0, min(3, int(GRID_LIMIT - 1 - r1)))
    right_margin = random.randint(0, min(3, int(GRID_LIMIT - 1 - c1)))
    return canvas[r0 - top_margin : r1 + bottom_margin + 1, c0 - left_margin : c1 + right_margin + 1].copy()


def _specs() -> list[dict]:
    count = random.choice([3, 4])
    colors = random.sample(list(range(1, 10)), count)

    directions = ["left", "right", "up", "down"]
    random.shuffle(directions)
    directions = directions[:count]

    gaps = [1, 2, 3]
    if count == 4:
        gaps.append(random.choice([1, 2, 3]))
    random.shuffle(gaps)
    gaps = gaps[:count]

    sizes = [random.choice([3, 5]) for _ in range(count)]
    if count == 3 and random.random() < 0.4:
        sizes[random.randrange(count)] = 7

    specs = []
    for color, direction, gap, side in zip(colors, directions, gaps, sizes):
        mask = _generate_mask(side)
        mask = np.rot90(mask, random.randrange(4))
        specs.append(
            {
                "color": color,
                "direction": direction,
                "gap": gap,
                "side": side,
                "mask": mask,
            }
        )
    return specs


def _gen_attempt() -> np.ndarray | None:
    canvas = np.zeros((GRID_LIMIT, GRID_LIMIT), dtype=int)
    specs = _specs()
    specs.sort(key=lambda spec: spec["side"], reverse=True)
    placed_rects = []

    for spec in specs:
        side = spec["side"]
        direction = spec["direction"]
        gap = spec["gap"]
        rr0, rc0, rr1, rc1 = _reserved_rect(side, direction, gap)
        top_min = -rr0
        top_max = GRID_LIMIT - 1 - rr1
        left_min = -rc0
        left_max = GRID_LIMIT - 1 - rc1
        if top_min > top_max or left_min > left_max:
            return None

        placed = False
        for _ in range(PLACE_RETRIES):
            top = random.randint(top_min, top_max)
            left = random.randint(left_min, left_max)
            rect = (top + rr0, left + rc0, top + rr1, left + rc1)
            if any(_rects_overlap(rect, other) for other in placed_rects):
                continue
            _paint_object(canvas, top, left, spec["mask"], spec["color"], direction, gap)
            placed_rects.append(rect)
            placed = True
            break
        if not placed:
            return None

    grid = _crop_canvas(canvas)
    try:
        out = main(grid)
    except AssertionError:
        return None

    if grid.shape[0] > GRID_LIMIT or grid.shape[1] > GRID_LIMIT:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
