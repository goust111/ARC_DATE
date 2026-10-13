from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
CARDINALS = ("up", "down", "left", "right")
DIR_STEP = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PALETTE = tuple(range(1, 10))
MARGIN = 1
ZONE_RANGE = (12, 13)
GAP_RANGE = (1, 2)
BLOCK_SIDE_CHOICES = (5, 7)
MAX_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    out.sort(key=lambda comp: comp["bbox"])
    return out


def _hole_direction(bbox: BBox, local_hole: Coord) -> str:
    r0, c0, r1, c1 = bbox
    hr = int(r0 + local_hole[0])
    hc = int(c0 + local_hole[1])
    row_center2 = int(r0 + r1)
    col_center2 = int(c0 + c1)

    if hr * 2 == row_center2 and hc * 2 < col_center2:
        return "left"
    if hr * 2 == row_center2 and hc * 2 > col_center2:
        return "right"
    if hc * 2 == col_center2 and hr * 2 < row_center2:
        return "up"
    if hc * 2 == col_center2 and hr * 2 > row_center2:
        return "down"
    raise AssertionError("hole must lie on the block center row or center column")


def _classify_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    blocks: list[dict] = []
    walls: list[dict] = []

    for comp in find_components(g, bg):
        color = int(comp["color"])
        bbox = comp["bbox"]
        r0, c0, r1, c1 = bbox
        crop = crop_bbox(g, bbox)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)

        if (h == 1 or w == 1) and np.all(crop == color):
            walls.append(
                {
                    "color": color,
                    "coords": comp["coords"],
                    "bbox": bbox,
                    "orientation": "horizontal" if h == 1 else "vertical",
                }
            )
            continue

        holes = np.argwhere(crop == bg)
        if holes.shape[0] == 1 and int(comp["area"]) == int(crop.size) - 1 and np.all((crop == color) | (crop == bg)):
            hr, hc = map(int, holes[0])
            assert 0 < hr < h - 1 and 0 < hc < w - 1, "hole must be internal"
            direction = _hole_direction(bbox, (hr, hc))
            blocks.append(
                {
                    "color": color,
                    "coords": comp["coords"],
                    "bbox": bbox,
                    "hole": (r0 + hr, c0 + hc),
                    "direction": direction,
                    "size": (h, w),
                }
            )
            continue

        raise AssertionError("every foreground component must be either a straight wall or a one-hole block")

    assert blocks, "expected at least one block"
    assert walls, "expected at least one wall"
    return bg, blocks, walls


def _wall_distance(block: dict, wall: dict, direction: str) -> int | None:
    br0, bc0, br1, bc1 = block["bbox"]
    wr0, wc0, wr1, wc1 = wall["bbox"]

    if direction == "right":
        if wall["orientation"] != "vertical" or not (wr0 <= br0 and wr1 >= br1) or wc0 <= bc1:
            return None
        return int(wc0 - bc1 - 1)
    if direction == "left":
        if wall["orientation"] != "vertical" or not (wr0 <= br0 and wr1 >= br1) or wc0 >= bc0:
            return None
        return int(bc0 - wc0 - 1)
    if direction == "down":
        if wall["orientation"] != "horizontal" or not (wc0 <= bc0 and wc1 >= bc1) or wr0 <= br1:
            return None
        return int(wr0 - br1 - 1)
    if direction == "up":
        if wall["orientation"] != "horizontal" or not (wc0 <= bc0 and wc1 >= bc1) or wr0 >= br0:
            return None
        return int(br0 - wr0 - 1)
    raise ValueError(direction)


def _choose_wall(block: dict, walls: list[dict], direction: str) -> tuple[int, dict]:
    candidates = []
    for wall in walls:
        dist = _wall_distance(block, wall, direction)
        if dist is not None and dist >= 0:
            candidates.append((dist, wall["bbox"], wall))
    assert candidates, f"no blocking wall found for {direction}"
    dist, _bbox, wall = min(candidates, key=lambda item: (item[0], item[1]))
    return int(dist), wall


def _distance_to_edge(shape: tuple[int, int], bbox: BBox, direction: str) -> int:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if direction == "left":
        return int(c0)
    if direction == "right":
        return int(w - 1 - c1)
    if direction == "up":
        return int(r0)
    if direction == "down":
        return int(h - 1 - r1)
    raise ValueError(direction)


def _apply_moves(
    grid: np.ndarray,
    blocks: list[dict],
    plans: list[dict],
    bg: int,
    *,
    strict: bool,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for block in blocks:
        coords = block["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    occupied = np.zeros(out.shape, dtype=bool)
    for block, plan in zip(blocks, plans, strict=True):
        r0, c0, r1, c1 = block["bbox"]
        dr, dc = map(int, plan["shift"])
        nr0, nc0, nr1, nc1 = r0 + dr, c0 + dc, r1 + dr, c1 + dc
        assert 0 <= nr0 <= nr1 < out.shape[0]
        assert 0 <= nc0 <= nc1 < out.shape[1]

        if strict:
            assert np.all(out[nr0 : nr1 + 1, nc0 : nc1 + 1] == bg), "moved block must land on cleared background"
            assert not occupied[nr0 : nr1 + 1, nc0 : nc1 + 1].any(), "moved blocks must not overlap"

        out[nr0 : nr1 + 1, nc0 : nc1 + 1] = int(block["color"])
        hr, hc = block["hole"]
        fill = plan["fill_color"]
        out[hr + dr, hc + dc] = bg if fill is None else int(fill)
        occupied[nr0 : nr1 + 1, nc0 : nc1 + 1] = True

    return out


def _correct_plans(blocks: list[dict], walls: list[dict]) -> list[dict]:
    plans = []
    for block in blocks:
        direction = block["direction"]
        dist, wall = _choose_wall(block, walls, direction)
        step_r, step_c = DIR_STEP[direction]
        plans.append({"shift": (step_r * dist, step_c * dist), "fill_color": int(wall["color"])})
    return plans


def _nearest_wall_plans(blocks: list[dict], walls: list[dict]) -> list[dict]:
    plans = []
    for block in blocks:
        candidates = []
        for direction in CARDINALS:
            try:
                dist, wall = _choose_wall(block, walls, direction)
            except AssertionError:
                continue
            candidates.append((dist, direction, wall["bbox"], int(wall["color"])))
        assert candidates, "every block must see at least one wall"
        dist, direction, _bbox, fill_color = min(candidates, key=lambda item: (item[0], item[2], item[1]))
        step_r, step_c = DIR_STEP[direction]
        plans.append({"shift": (step_r * dist, step_c * dist), "fill_color": fill_color})
    return plans


def _edge_plans(blocks: list[dict], shape: tuple[int, int]) -> list[dict]:
    plans = []
    for block in blocks:
        direction = block["direction"]
        dist = _distance_to_edge(shape, block["bbox"], direction)
        step_r, step_c = DIR_STEP[direction]
        plans.append({"shift": (step_r * dist, step_c * dist), "fill_color": None})
    return plans


def _no_fill_plans(blocks: list[dict], walls: list[dict]) -> list[dict]:
    plans = []
    for block in blocks:
        direction = block["direction"]
        dist, _wall = _choose_wall(block, walls, direction)
        step_r, step_c = DIR_STEP[direction]
        plans.append({"shift": (step_r * dist, step_c * dist), "fill_color": None})
    return plans


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each one-hole block in the hole-offset direction until it meets the first wall, then plug the hole with that wall's color.

    Input structure:
    - The grid has a dominant background, several straight one-cell-thick wall segments, and several monochrome rectangular blocks.
    - Each block is a filled rectangle except for exactly one internal background hole.
    - The hole lies on the block's center row or center column, clearly offset toward up, down, left, or right.

    Rule:
    - For every block, read the hole's offset from the block center to determine the slide direction.
    - Translate the whole block along that direction until its leading side becomes adjacent to the first wall segment whose span fully covers the block on the perpendicular axis.
    - Erase the old block position, keep the wall unchanged, and fill the moved hole with the color of the wall that stopped the block.

    Key insight:
    - The internal hole is not damage to repair first; it is the steering cue that chooses which wall matters, and only the struck wall determines the final fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, blocks, walls = _classify_scene(g)
    plans = _correct_plans(blocks, walls)
    return _apply_moves(g, blocks, plans, bg, strict=True)


def _horizontal_specs(zone_w: int) -> list[tuple[int, int, int, int]]:
    specs = []
    for block_w in BLOCK_SIDE_CHOICES:
        for near_gap in (1, 2):
            for far_gap in (near_gap + 1, near_gap + 2):
                slack = int(zone_w - (block_w + near_gap + far_gap + 2))
                if slack >= 0:
                    specs.append((block_w, near_gap, far_gap, slack))
    return specs


def _vertical_specs(zone_h: int) -> list[tuple[int, int, int, int]]:
    specs = []
    for block_h in BLOCK_SIDE_CHOICES:
        for near_gap in (1, 2):
            for far_gap in (near_gap + 1, near_gap + 2):
                slack = int(zone_h - (block_h + near_gap + far_gap + 2))
                if slack >= 0:
                    specs.append((block_h, near_gap, far_gap, slack))
    return specs


def _zone_patch(
    zone_h: int,
    zone_w: int,
    direction: str,
    block_color: int,
    target_color: int,
    other_color: int,
) -> tuple[np.ndarray, dict]:
    patch = np.full((zone_h, zone_w), BG, dtype=int)

    if direction in {"left", "right"}:
        block_h_choices = [h for h in BLOCK_SIDE_CHOICES if h <= zone_h - 2]
        assert block_h_choices
        block_h = random.choice(block_h_choices)
        block_w, near_gap, far_gap, slack_w = random.choice(_horizontal_specs(zone_w))
        extra_left = random.randint(0, slack_w)
        extra_right = slack_w - extra_left

        if direction == "right":
            left_gap, right_gap = near_gap, far_gap
            left_color, right_color = other_color, target_color
        else:
            left_gap, right_gap = far_gap, near_gap
            left_color, right_color = target_color, other_color

        left_wall_c = extra_left
        block_c0 = left_wall_c + 1 + left_gap
        block_c1 = block_c0 + block_w - 1
        right_wall_c = block_c1 + 1 + right_gap
        assert right_wall_c + extra_right == zone_w - 1

        block_r0 = random.randint(0, zone_h - block_h)
        block_r1 = block_r0 + block_h - 1
        left_wall_r0 = random.randint(0, block_r0)
        left_wall_r1 = random.randint(block_r1, zone_h - 1)
        right_wall_r0 = random.randint(0, block_r0)
        right_wall_r1 = random.randint(block_r1, zone_h - 1)

        patch[left_wall_r0 : left_wall_r1 + 1, left_wall_c] = left_color
        patch[right_wall_r0 : right_wall_r1 + 1, right_wall_c] = right_color
        patch[block_r0 : block_r1 + 1, block_c0 : block_c1 + 1] = block_color

        center_r = block_r0 + block_h // 2
        center_c = block_c0 + block_w // 2
        hole_offset = random.randint(1, block_w // 2 - 1)
        hole_r = center_r
        hole_c = center_c + hole_offset if direction == "right" else center_c - hole_offset
        patch[hole_r, hole_c] = BG

        target_gap = right_gap if direction == "right" else left_gap
        opposite_gap = left_gap if direction == "right" else right_gap
        target_wall_bbox = (right_wall_r0, right_wall_c, right_wall_r1, right_wall_c) if direction == "right" else (
            left_wall_r0,
            left_wall_c,
            left_wall_r1,
            left_wall_c,
        )
    else:
        block_w_choices = [w for w in BLOCK_SIDE_CHOICES if w <= zone_w - 2]
        assert block_w_choices
        block_w = random.choice(block_w_choices)
        block_h, near_gap, far_gap, slack_h = random.choice(_vertical_specs(zone_h))
        extra_top = random.randint(0, slack_h)
        extra_bottom = slack_h - extra_top

        if direction == "down":
            top_gap, bottom_gap = near_gap, far_gap
            top_color, bottom_color = other_color, target_color
        else:
            top_gap, bottom_gap = far_gap, near_gap
            top_color, bottom_color = target_color, other_color

        top_wall_r = extra_top
        block_r0 = top_wall_r + 1 + top_gap
        block_r1 = block_r0 + block_h - 1
        bottom_wall_r = block_r1 + 1 + bottom_gap
        assert bottom_wall_r + extra_bottom == zone_h - 1

        block_c0 = random.randint(0, zone_w - block_w)
        block_c1 = block_c0 + block_w - 1
        top_wall_c0 = random.randint(0, block_c0)
        top_wall_c1 = random.randint(block_c1, zone_w - 1)
        bottom_wall_c0 = random.randint(0, block_c0)
        bottom_wall_c1 = random.randint(block_c1, zone_w - 1)

        patch[top_wall_r, top_wall_c0 : top_wall_c1 + 1] = top_color
        patch[bottom_wall_r, bottom_wall_c0 : bottom_wall_c1 + 1] = bottom_color
        patch[block_r0 : block_r1 + 1, block_c0 : block_c1 + 1] = block_color

        center_r = block_r0 + block_h // 2
        center_c = block_c0 + block_w // 2
        hole_offset = random.randint(1, block_h // 2 - 1)
        hole_r = center_r + hole_offset if direction == "down" else center_r - hole_offset
        hole_c = center_c
        patch[hole_r, hole_c] = BG

        target_gap = bottom_gap if direction == "down" else top_gap
        opposite_gap = top_gap if direction == "down" else bottom_gap
        target_wall_bbox = (bottom_wall_r, bottom_wall_c0, bottom_wall_r, bottom_wall_c1) if direction == "down" else (
            top_wall_r,
            top_wall_c0,
            top_wall_r,
            top_wall_c1,
        )

    return patch, {
        "direction": direction,
        "block_color": block_color,
        "target_wall_color": target_color,
        "target_gap": int(target_gap),
        "opposite_gap": int(opposite_gap),
        "block_bbox": (block_r0, block_c0, block_r1, block_c1),
        "hole": (hole_r, hole_c),
        "size": (block_r1 - block_r0 + 1, block_c1 - block_c0 + 1),
        "target_wall_bbox": target_wall_bbox,
    }


def _manual_output(grid: np.ndarray, records: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for rec in records:
        r0, c0, r1, c1 = rec["block_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

    for rec in records:
        step_r, step_c = DIR_STEP[rec["direction"]]
        shift_r = step_r * int(rec["target_gap"])
        shift_c = step_c * int(rec["target_gap"])
        r0, c0, r1, c1 = rec["block_bbox"]
        out[r0 + shift_r : r1 + shift_r + 1, c0 + shift_c : c1 + shift_c + 1] = int(rec["block_color"])
        hr, hc = rec["hole"]
        out[hr + shift_r, hc + shift_c] = int(rec["target_wall_color"])
    return out


def _build_attempt() -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    top_h = random.randint(*ZONE_RANGE)
    bottom_h = random.randint(*ZONE_RANGE)
    left_w = random.randint(*ZONE_RANGE)
    right_w = random.randint(*ZONE_RANGE)
    gap_r = random.randint(*GAP_RANGE)
    gap_c = random.randint(*GAP_RANGE)

    height = MARGIN + top_h + gap_r + bottom_h + MARGIN
    width = MARGIN + left_w + gap_c + right_w + MARGIN
    if max(height, width) > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    zones = [
        (MARGIN, MARGIN, top_h, left_w),
        (MARGIN, MARGIN + left_w + gap_c, top_h, right_w),
        (MARGIN + top_h + gap_r, MARGIN, bottom_h, left_w),
        (MARGIN + top_h + gap_r, MARGIN + left_w + gap_c, bottom_h, right_w),
    ]

    directions = list(CARDINALS)
    random.shuffle(directions)
    colors = random.sample(PALETTE, 8)
    block_colors = colors[:4]
    wall_palette = colors[4:]

    records: list[dict] = []
    for zone, direction, block_color in zip(zones, directions, block_colors, strict=True):
        zr0, zc0, zh, zw = zone
        target_color, other_color = random.sample(wall_palette, 2)
        patch, local = _zone_patch(zh, zw, direction, block_color, target_color, other_color)
        grid[zr0 : zr0 + zh, zc0 : zc0 + zw] = patch

        br0, bc0, br1, bc1 = local["block_bbox"]
        hr, hc = local["hole"]
        wr0, wc0, wr1, wc1 = local["target_wall_bbox"]
        local["block_bbox"] = (zr0 + br0, zc0 + bc0, zr0 + br1, zc0 + bc1)
        local["hole"] = (zr0 + hr, zc0 + hc)
        local["target_wall_bbox"] = (zr0 + wr0, zc0 + wc0, zr0 + wr1, zc0 + wc1)
        records.append(local)

    if len({rec["size"] for rec in records}) == len(records):
        return None
    if len({rec["target_gap"] for rec in records}) < 2:
        return None
    if len({rec["target_wall_color"] for rec in records}) < 2:
        return None
    if not all(rec["target_gap"] > rec["opposite_gap"] for rec in records):
        return None

    expected = _manual_output(grid, records)
    return grid, expected, records


def _semantic_ok(inp: np.ndarray, expected: np.ndarray, records: list[dict]) -> bool:
    try:
        bg, blocks, walls = _classify_scene(inp)
        if bg != BG or len(blocks) != 4:
            return False
        if set(block["direction"] for block in blocks) != set(CARDINALS):
            return False

        blocks_sorted = sorted(blocks, key=lambda block: block["bbox"])
        records_sorted = sorted(records, key=lambda rec: rec["block_bbox"])
        for block, rec in zip(blocks_sorted, records_sorted, strict=True):
            if block["color"] != rec["block_color"]:
                return False
            if block["bbox"] != rec["block_bbox"]:
                return False
            if block["hole"] != rec["hole"]:
                return False
            if block["direction"] != rec["direction"]:
                return False

        if not np.array_equal(main(inp), expected):
            return False
        if np.array_equal(_apply_moves(inp, blocks, _nearest_wall_plans(blocks, walls), bg, strict=False), expected):
            return False
        if np.array_equal(_apply_moves(inp, blocks, _edge_plans(blocks, inp.shape), bg, strict=False), expected):
            return False
        if np.array_equal(_apply_moves(inp, blocks, _no_fill_plans(blocks, walls), bg, strict=False), expected):
            return False
    except Exception:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_attempt()
        if built is None:
            continue
        inp, expected, records = built
        if _semantic_ok(inp, expected, records):
            return inp
    raise RuntimeError("failed to generate a valid hole-steering sample")
