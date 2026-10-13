from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_MASKS = {
    "up": np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    "down": np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=bool),
    "left": np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
    "right": np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=bool),
}
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_to_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _arrow_direction(mask: np.ndarray) -> str | None:
    for direction, patt in ARROW_MASKS.items():
        if mask.shape == patt.shape and np.array_equal(mask, patt):
            return direction
    return None


def _block_cells(top: int, left: int) -> list[tuple[int, int]]:
    return [(top, left), (top, left + 1), (top + 1, left), (top + 1, left + 1)]


def _distance_between(block_top: int, block_left: int, arrow_top: int, arrow_left: int, direction: str) -> int:
    br0 = int(block_top)
    bc0 = int(block_left)
    br1 = br0 + 1
    bc1 = bc0 + 1
    ar0 = int(arrow_top)
    ac0 = int(arrow_left)
    ar1 = ar0 + 2
    ac1 = ac0 + 2

    if direction in ("left", "right"):
        assert max(br0, ar0) <= min(br1, ar1), "horizontal arrow must share a row with its block"
        if ac1 < bc0:
            return int(bc0 - ac1 - 1)
        if bc1 < ac0:
            return int(ac0 - bc1 - 1)
        raise AssertionError("horizontal arrow and block must be disjoint along columns")

    assert max(bc0, ac0) <= min(bc1, ac1), "vertical arrow must share a column with its block"
    if ar1 < br0:
        return int(br0 - ar1 - 1)
    if br1 < ar0:
        return int(ar0 - br1 - 1)
    raise AssertionError("vertical arrow and block must be disjoint along rows")


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    wall = int(g[0, 0])
    assert wall != BG
    assert np.all(g[0, :] == wall)
    assert np.all(g[-1, :] == wall)
    assert np.all(g[:, 0] == wall)
    assert np.all(g[:, -1] == wall)

    colors = sorted(int(v) for v in np.unique(g) if int(v) not in (BG, wall))
    objects: list[dict[str, int]] = []
    for color in colors:
        comps = _components(g == color)
        block: dict[str, int] | None = None
        arrow: dict[str, int] | None = None
        for coords in comps:
            box = _bbox(coords)
            r0, c0, r1, c1 = box
            mask = _coords_to_mask(coords, box)
            if coords.shape[0] == 4 and mask.shape == (2, 2) and bool(mask.all()):
                assert block is None, "each color has at most one 2x2 block"
                block = {"block_top": r0, "block_left": c0}
                continue
            if coords.shape[0] == 5 and mask.shape == (3, 3):
                direction = _arrow_direction(mask)
                if direction is not None:
                    assert arrow is None, "each color has at most one arrow"
                    arrow = {
                        "arrow_top": r0,
                        "arrow_left": c0,
                        "direction_idx": 0,
                    }
                    arrow["direction"] = direction
                    continue
            raise AssertionError("unexpected same-color component shape")

        assert block is not None and arrow is not None, "each active color must have one block and one arrow"
        direction = str(arrow["direction"])
        distance = _distance_between(
            int(block["block_top"]),
            int(block["block_left"]),
            int(arrow["arrow_top"]),
            int(arrow["arrow_left"]),
            direction,
        )
        objects.append(
            {
                "color": int(color),
                "block_top": int(block["block_top"]),
                "block_left": int(block["block_left"]),
                "arrow_top": int(arrow["arrow_top"]),
                "arrow_left": int(arrow["arrow_left"]),
                "distance": int(distance),
                "direction": direction,
            }
        )

    objects.sort(key=lambda obj: (int(obj["distance"]), int(obj["color"])))
    return wall, objects


def _simulate_positions(shape: tuple[int, int], objects: list[dict[str, int]], wall_mask: np.ndarray) -> dict[int, tuple[int, int]]:
    positions = {
        int(obj["color"]): (int(obj["block_top"]), int(obj["block_left"]))
        for obj in objects
    }
    occupied = {
        (r, c)
        for top, left in positions.values()
        for r, c in _block_cells(int(top), int(left))
    }

    for obj in objects:
        color = int(obj["color"])
        dr, dc = DIR_VECS[str(obj["direction"])]
        top, left = positions[color]
        current_cells = _block_cells(int(top), int(left))
        for cell in current_cells:
            occupied.remove(cell)

        while True:
            cand_top = int(top + dr)
            cand_left = int(left + dc)
            cand_cells = _block_cells(cand_top, cand_left)
            blocked = False
            for rr, cc in cand_cells:
                if not (0 <= rr < shape[0] and 0 <= cc < shape[1]):
                    blocked = True
                    break
                if wall_mask[rr, cc] or (rr, cc) in occupied:
                    blocked = True
                    break
            if blocked:
                break
            top = cand_top
            left = cand_left

        positions[color] = (int(top), int(left))
        for cell in _block_cells(int(top), int(left)):
            occupied.add(cell)

    return positions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored 2x2 block according to its matching arrow, executing nearer arrows first.

    Input structure:
    - Background is black, and the outer border is a complete grey wall.
    - Inside the wall are several solid 2x2 colored blocks, with at most one block per color.
    - Inside the wall are several 5-pixel arrows; each arrow is a monochrome 3x3 up/down/left/right shape whose color matches one block.

    Rule:
    - For every arrow, find the same-colored 2x2 block.
    - Measure the black-cell gap between the arrow and its block along the arrow's row or column, then execute arrows from the smallest gap to the largest.
    - When executed, the matching block slides in the arrow's pointing direction until its front edge would hit the grey wall or another block, and arrows never act as obstacles.
    - After all moves, remove every arrow and keep only the wall plus the moved blocks.

    Key insight:
    - The arrow color chooses which block moves, the arrow shape chooses the movement direction, and the subtle second rule is that execution order depends on arrow-to-block distance rather than on color or reading order.
    """
    g = np.asarray(grid, dtype=int)
    wall, objects = _parse_scene(g)
    wall_mask = g == wall
    positions = _simulate_positions(g.shape, objects, wall_mask)

    out = g.copy()
    out[(g != BG) & (g != wall)] = BG
    for obj in objects:
        color = int(obj["color"])
        top, left = positions[color]
        for r, c in _block_cells(int(top), int(left)):
            out[r, c] = color
    return out


def _rand_comp(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    return [int(x) for x in np.random.multinomial(int(total), np.full(parts, 1.0 / parts))]


def _lane_counts(n_items: int) -> list[int]:
    min_lanes = (int(n_items) + 1) // 2
    max_lanes = min(3, int(n_items))
    lane_count = random.randint(min_lanes, max_lanes)
    counts = [1] * lane_count
    extra = int(n_items - lane_count)
    order = list(range(lane_count))
    random.shuffle(order)
    for idx in order[:extra]:
        counts[idx] += 1
    random.shuffle(counts)
    return counts


def _stamp_block(grid: np.ndarray, top: int, left: int, color: int) -> None:
    grid[top : top + 2, left : left + 2] = color


def _stamp_arrow(grid: np.ndarray, top: int, left: int, direction: str, color: int) -> None:
    mask = ARROW_MASKS[direction]
    patch = grid[top : top + 3, left : left + 3]
    patch[mask] = color


def _build_horizontal_grid(lanes: list[dict[str, object]]) -> np.ndarray | None:
    lane_count = len(lanes)
    inner_h = 5 * lane_count + random.randint(0, min(4, 28 - 5 * lane_count))
    h = inner_h + 2
    min_inner_w = max(int(lane["min_span"]) for lane in lanes)
    extra_w = random.randint(0, min(6, 28 - min_inner_w))
    inner_w = min_inner_w + extra_w
    w = inner_w + 2
    if max(h, w) > 30:
        return None

    slot_extras = _rand_comp(inner_h - 5 * lane_count, lane_count + 1)
    slot_starts: list[int] = []
    cursor = 1 + int(slot_extras[0])
    for idx in range(lane_count):
        slot_starts.append(int(cursor))
        cursor += 5
        if idx + 1 < lane_count:
            cursor += int(slot_extras[idx + 1])

    g = np.full((h, w), BG, dtype=int)
    g[0, :] = WALL
    g[-1, :] = WALL
    g[:, 0] = WALL
    g[:, -1] = WALL

    for lane, slot_start in zip(lanes, slot_starts):
        items = list(lane["items"])
        random.shuffle(items)
        widths = [5 + int(item["distance"]) for item in items]
        base_span = sum(widths) + (len(items) + 1)
        gaps = [1 + x for x in _rand_comp(inner_w - base_span, len(items) + 1)]
        row = int(slot_start + random.choice((1, 2)))
        pos = int(gaps[0])
        for item, width, gap_after in zip(items, widths, gaps[1:]):
            start_col = 1 + int(pos)
            arrow_top = row - 1 if bool(item["align_first"]) else row
            if item["side"] == "left":
                _stamp_arrow(g, arrow_top, start_col, str(lane["direction"]), int(item["color"]))
                _stamp_block(g, row, start_col + 3 + int(item["distance"]), int(item["color"]))
            else:
                _stamp_block(g, row, start_col, int(item["color"]))
                _stamp_arrow(g, arrow_top, start_col + 2 + int(item["distance"]), str(lane["direction"]), int(item["color"]))
            pos += int(width + gap_after)

    return g


def _build_vertical_grid(lanes: list[dict[str, object]]) -> np.ndarray | None:
    lane_count = len(lanes)
    inner_w = 5 * lane_count + random.randint(0, min(4, 28 - 5 * lane_count))
    w = inner_w + 2
    min_inner_h = max(int(lane["min_span"]) for lane in lanes)
    extra_h = random.randint(0, min(6, 28 - min_inner_h))
    inner_h = min_inner_h + extra_h
    h = inner_h + 2
    if max(h, w) > 30:
        return None

    slot_extras = _rand_comp(inner_w - 5 * lane_count, lane_count + 1)
    slot_starts: list[int] = []
    cursor = 1 + int(slot_extras[0])
    for idx in range(lane_count):
        slot_starts.append(int(cursor))
        cursor += 5
        if idx + 1 < lane_count:
            cursor += int(slot_extras[idx + 1])

    g = np.full((h, w), BG, dtype=int)
    g[0, :] = WALL
    g[-1, :] = WALL
    g[:, 0] = WALL
    g[:, -1] = WALL

    for lane, slot_start in zip(lanes, slot_starts):
        items = list(lane["items"])
        random.shuffle(items)
        heights = [5 + int(item["distance"]) for item in items]
        base_span = sum(heights) + (len(items) + 1)
        gaps = [1 + x for x in _rand_comp(inner_h - base_span, len(items) + 1)]
        col = int(slot_start + random.choice((1, 2)))
        pos = int(gaps[0])
        for item, height, gap_after in zip(items, heights, gaps[1:]):
            start_row = 1 + int(pos)
            arrow_left = col - 1 if bool(item["align_first"]) else col
            if item["side"] == "above":
                _stamp_arrow(g, start_row, arrow_left, str(lane["direction"]), int(item["color"]))
                _stamp_block(g, start_row + 3 + int(item["distance"]), col, int(item["color"]))
            else:
                _stamp_block(g, start_row, col, int(item["color"]))
                _stamp_arrow(g, start_row + 2 + int(item["distance"]), arrow_left, str(lane["direction"]), int(item["color"]))
            pos += int(height + gap_after)

    return g


def _generate_attempt() -> np.ndarray | None:
    n_items = random.randint(2, 4)
    colors = random.sample(COLORS, n_items)
    distances = random.sample(range(1, 7), n_items)
    counts = _lane_counts(n_items)
    axis = random.choice(("horizontal", "vertical"))
    lane_dirs = ("left", "right") if axis == "horizontal" else ("up", "down")

    lanes: list[dict[str, object]] = []
    color_idx = 0
    dist_idx = 0
    for count in counts:
        items = []
        for _ in range(count):
            if axis == "horizontal":
                side = random.choice(("left", "right"))
            else:
                side = random.choice(("above", "below"))
            items.append(
                {
                    "color": int(colors[color_idx]),
                    "distance": int(distances[dist_idx]),
                    "side": side,
                    "align_first": bool(random.getrandbits(1)),
                }
            )
            color_idx += 1
            dist_idx += 1
        lanes.append(
            {
                "direction": random.choice(lane_dirs),
                "items": items,
                "min_span": int(sum(5 + int(item["distance"]) for item in items) + len(items) + 1),
            }
        )

    if axis == "horizontal":
        grid = _build_horizontal_grid(lanes)
    else:
        grid = _build_vertical_grid(lanes)
    if grid is None:
        return None

    wall, objects = _parse_scene(grid)
    if len(objects) != n_items:
        return None
    distances_seen = [int(obj["distance"]) for obj in objects]
    if len(set(distances_seen)) != len(distances_seen):
        return None

    positions = _simulate_positions(grid.shape, objects, grid == wall)
    if not any(
        positions[int(obj["color"])] != (int(obj["block_top"]), int(obj["block_left"]))
        for obj in objects
    ):
        return None

    if n_items >= 3 and not any(len(list(lane["items"])) == 2 for lane in lanes):
        return None

    main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated retries")
