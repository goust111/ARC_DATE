from __future__ import annotations

import random
from collections import Counter, deque
from itertools import combinations

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


BBox = tuple[int, int, int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    colors: list[int] | tuple[int, ...] | None = None,
    connectivity: int = 4,
    monochromatic: bool = True,
) -> list[dict]:
    if connectivity != 4:
        raise ValueError("This task only needs 4-connectivity.")
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = get_bg(g)
    bg = int(bg)
    allowed = None if colors is None else {int(color) for color in colors}
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color == bg:
                continue
            if allowed is not None and color not in allowed:
                continue

            seen[r, c] = True
            queue = deque([(r, c)])
            coords = []
            root_color = color

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w) or seen[nr, nc]:
                        continue
                    neighbor_color = int(g[nr, nc])
                    if neighbor_color == bg:
                        continue
                    if allowed is not None and neighbor_color not in allowed:
                        continue
                    if monochromatic and neighbor_color != root_color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            coords_array = np.array(coords, dtype=int)
            components.append(
                {
                    "color": root_color if monochromatic else None,
                    "coords": coords_array,
                    "bbox": bbox_coords(coords_array),
                    "area": int(coords_array.shape[0]),
                }
            )

    return components


def _u_container_from_component(component: dict) -> dict | None:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = component["coords"] - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    h, w = mask.shape
    if h < 4 or w < 5:
        return None
    if not np.all(mask[:, 0]) or not np.all(mask[:, -1]) or not np.all(mask[-1, :]):
        return None
    if np.any(mask[:-1, 1:-1]):
        return None
    return {"top": r0, "bottom": r1, "left": c0, "right": c1}


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)

    sprite_candidates = [component for component in components if component["bbox"][0] <= 4]
    assert len(sprite_candidates) == 1, "expected exactly one top sprite"
    sprite = sprite_candidates[0]
    sr0, sc0, sr1, sc1 = sprite["bbox"]
    sprite_w = sc1 - sc0 + 1
    step = sprite_w + 1

    slots = []
    slot_left = sc0
    idx = 0
    while slot_left + sprite_w - 1 < g.shape[1]:
        slots.append((idx, slot_left, slot_left + sprite_w - 1))
        idx += 1
        slot_left += step
    assert len(slots) >= 5, "expected at least five periodic slots"

    containers = []
    for component in components:
        if int(component["color"]) != Color.RED:
            continue
        info = _u_container_from_component(component)
        if info is not None and info["top"] >= 5:
            containers.append(info)
    containers.sort(key=lambda item: (item["left"], item["top"]))
    assert 2 <= len(containers) <= 4, "expected 2-4 red containers"

    return {
        "bg": bg,
        "sprite": sprite,
        "sprite_color": int(sprite["color"]),
        "sprite_bbox": sprite["bbox"],
        "sprite_w": sprite_w,
        "slots": slots,
        "containers": containers,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Expand the top sprite into periodic drop slots that fill red containers or paint bottom-row puddles.

    Input structure:
    - The background is black, with one small 4-connected monochrome sprite near the top.
    - Lower in the grid there are 2-4 red U-shaped containers with empty rectangular interiors.
    - The sprite's bounding-box width defines a horizontal period of candidate drop slots.

    Rule:
    - Erase the original top sprite, but keep its bounding-box left edge and width as the slot seed.
    - Starting from that left edge, make slots every (sprite width + 1) columns; slots with index 2, 5, 8, ... use pink, and all others use the sprite's original color.
    - If a slot's full column span lies inside a red container's horizontal footprint, fill that container interior with the slot color; otherwise color only those slot columns on the bottom row.
    - Keep all red container wall pixels unchanged, and let later slots overwrite earlier fills in the same container.

    Key insight:
    - The visible top sprite is only a seed for periodic column placement and color cycling; the final pattern is determined by where those periodic column bands land relative to the red containers.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_input(g)

    out = g.copy()
    sprite_coords = info["sprite"]["coords"]
    out[sprite_coords[:, 0], sprite_coords[:, 1]] = info["bg"]

    for idx, left, right in info["slots"]:
        fill_color = Color.PINK if idx % 3 == 2 else info["sprite_color"]
        target = None
        for container in info["containers"]:
            if container["left"] <= left and right <= container["right"]:
                target = container
                break
        if target is None:
            out[-1, left : right + 1] = fill_color
        else:
            out[target["top"] : target["bottom"], target["left"] + 1 : target["right"]] = fill_color

    return out


_SPRITE_DIMS = [(3, 3), (3, 4), (4, 3)]
_SPRITE_COLORS = [Color.BLUE, Color.GREEN, Color.YELLOW, Color.GREY, Color.ORANGE, Color.TEAL, Color.MAROON]
_GEN_RETRIES = 400


def _random_connected_mask(height: int, width: int, min_area: int, max_area: int) -> np.ndarray | None:
    for _ in range(200):
        target_area = random.randint(min_area, max_area)
        start = (random.randrange(height), random.randrange(width))
        cells = {start}
        frontier = {
            (nr, nc)
            for nr, nc in _neighbors4(*start)
            if 0 <= nr < height and 0 <= nc < width
        }

        while len(cells) < target_area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nr, nc in _neighbors4(*cell):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) != target_area:
            continue
        coords = np.array(sorted(cells), dtype=int)
        if bbox_coords(coords) != (0, 0, height - 1, width - 1):
            continue

        mask = np.zeros((height, width), dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        return mask
    return None


def _slot_ranges(slot_left0: int, sprite_w: int, slot_count: int) -> list[tuple[int, int]]:
    step = sprite_w + 1
    return [
        (slot_left0 + idx * step, slot_left0 + idx * step + sprite_w - 1)
        for idx in range(slot_count)
    ]


def _valid_group_subsets(
    slot_count: int,
    sprite_w: int,
    slot_ranges: list[tuple[int, int]],
) -> list[list[tuple[int, ...]]]:
    candidates = [(idx,) for idx in range(slot_count)]
    if sprite_w == 3:
        candidates += [(idx, idx + 1) for idx in range(slot_count - 1)]

    valid = []
    for size in range(2, 5):
        for combo in combinations(candidates, size):
            captured = set()
            singles = []
            ok = True
            for group in combo:
                group_cells = set(group)
                if captured & group_cells:
                    ok = False
                    break
                captured |= group_cells
                if len(group) == 1:
                    singles.append(group[0])
            if not ok or len(captured) >= slot_count:
                continue
            singles.sort()
            if any(b == a + 1 for a, b in zip(singles, singles[1:])):
                continue
            ordered = sorted(combo, key=lambda group: group[0])
            footprints = [_group_footprint(group, slot_ranges) for group in ordered]
            if any(next_left <= right + 1 for (_, right), (next_left, _) in zip(footprints, footprints[1:])):
                continue
            valid.append(ordered)
    return valid


def _group_footprint(group: tuple[int, ...], ranges: list[tuple[int, int]]) -> tuple[int, int]:
    if len(group) == 1:
        left, right = ranges[group[0]]
        return left - 1, right + 1
    first_left = ranges[group[0]][0]
    last_right = ranges[group[-1]][1]
    return first_left, last_right


def _build_expected(
    grid: np.ndarray,
    sprite_coords: np.ndarray,
    sprite_color: int,
    slot_ranges: list[tuple[int, int]],
    containers: list[dict],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[sprite_coords[:, 0], sprite_coords[:, 1]] = Color.BLACK

    for idx, (left, right) in enumerate(slot_ranges):
        color = Color.PINK if idx % 3 == 2 else sprite_color
        target = None
        for container in containers:
            if container["left"] <= left and right <= container["right"]:
                target = container
                break
        if target is None:
            out[-1, left : right + 1] = color
        else:
            out[target["top"] : target["bottom"], target["left"] + 1 : target["right"]] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    sprite_h, sprite_w = random.choice(_SPRITE_DIMS)
    min_area = 4 if (sprite_h, sprite_w) == (3, 3) else 5
    sprite_mask = _random_connected_mask(sprite_h, sprite_w, min_area, 6)
    if sprite_mask is None:
        return None

    c0 = random.randint(1, 2)
    width = random.randint(24, 30)
    step = sprite_w + 1
    max_slots = 1 + (width - c0 - sprite_w - 1) // step
    if max_slots < 5:
        return None
    slot_count = random.randint(5, min(7, max_slots))
    slot_ranges = _slot_ranges(c0, sprite_w, slot_count)

    group_options = _valid_group_subsets(slot_count, sprite_w, slot_ranges)
    if not group_options:
        return None
    double_options = [groups for groups in group_options if any(len(group) == 2 for group in groups)]
    if double_options and random.random() < 0.45:
        groups = random.choice(double_options)
    else:
        groups = random.choice(group_options)

    height = random.randint(18, 24)
    grid = np.full((height, width), Color.BLACK, dtype=int)
    sprite_color = random.choice(_SPRITE_COLORS)
    sprite_top = random.randint(0, max(0, 4 - sprite_h))
    sprite_patch = grid[sprite_top : sprite_top + sprite_h, c0 : c0 + sprite_w]
    sprite_patch[sprite_mask] = sprite_color
    sprite_coords = np.argwhere(sprite_mask) + np.array([sprite_top, c0], dtype=int)

    containers = []
    for group in groups:
        left, right = _group_footprint(group, slot_ranges)
        interior_h = random.randint(3, 5)
        max_top = height - interior_h - 2
        if max_top < 8:
            return None
        top = random.randint(8, max_top)
        bottom = top + interior_h
        grid[top : bottom + 1, left] = Color.RED
        grid[top : bottom + 1, right] = Color.RED
        grid[bottom, left : right + 1] = Color.RED
        containers.append({"top": top, "bottom": bottom, "left": left, "right": right})

    expected = _build_expected(
        grid=grid,
        sprite_coords=sprite_coords,
        sprite_color=sprite_color,
        slot_ranges=slot_ranges,
        containers=sorted(containers, key=lambda item: (item["left"], item["top"])),
    )
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    bottom_colors = set(map(int, expected[-1, expected[-1] != Color.BLACK]))
    interior_colors = set()
    for container in containers:
        fill = expected[container["top"] : container["bottom"], container["left"] + 1 : container["right"]]
        interior_colors.update(map(int, np.unique(fill[fill != Color.BLACK])))
    if not bottom_colors or not interior_colors:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-container ARC input")
