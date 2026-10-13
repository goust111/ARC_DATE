from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


DIRS = {
    "R": (0, 1),
    "L": (0, -1),
    "D": (1, 0),
    "U": (-1, 0),
}

PERPENDICULAR = {
    "R": ("U", "D"),
    "L": ("U", "D"),
    "U": ("L", "R"),
    "D": ("L", "R"),
}

COLORS = list(range(1, 10))
BG = 0


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touch_side(box: tuple[int, int, int, int], shape: tuple[int, int]) -> str | None:
    r0, c0, r1, c1 = box
    h, w = shape
    sides = []
    if r0 == 0:
        sides.append("top")
    if r1 == h - 1:
        sides.append("bottom")
    if c0 == 0:
        sides.append("left")
    if c1 == w - 1:
        sides.append("right")
    if len(sides) != 1:
        return None
    return sides[0]


def _side_to_dir(side: str) -> str:
    return {"left": "R", "right": "L", "top": "D", "bottom": "U"}[side]


def _solid_rect(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return int(comp["coords"].shape[0]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _find_components(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comp_map = np.full((h, w), -1, dtype=int)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(pts, dtype=int)
            cid = len(comps)
            comp_map[coords[:, 0], coords[:, 1]] = cid
            comps.append(
                {
                    "id": cid,
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )
    return comps, comp_map


def _lane_band(comp: dict, direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    if direction in ("R", "L"):
        return r0, r1
    return c0, c1


def _lane_thickness(comp: dict, direction: str) -> int:
    r0, c0, r1, c1 = comp["bbox"]
    return (r1 - r0 + 1) if direction in ("R", "L") else (c1 - c0 + 1)


def _lane_shaped(comp: dict, band: tuple[int, int], direction: str) -> bool:
    if not _solid_rect(comp):
        return False
    r0, c0, r1, c1 = comp["bbox"]
    if direction in ("R", "L"):
        return (r0, r1) == band
    return (c0, c1) == band


def _scan_lane(
    comp_map: np.ndarray,
    comps: list[dict],
    band: tuple[int, int],
    direction: str,
    start: int,
) -> list[dict]:
    h, w = comp_map.shape
    out = []
    last_id = None
    if direction == "R":
        axis = range(start, w)
        for c in axis:
            ids = set(int(x) for x in comp_map[band[0] : band[1] + 1, c] if int(x) >= 0)
            assert len(ids) <= 1
            if not ids:
                last_id = None
                continue
            cid = next(iter(ids))
            if cid != last_id:
                out.append(comps[cid])
            last_id = cid
    elif direction == "L":
        axis = range(start, -1, -1)
        for c in axis:
            ids = set(int(x) for x in comp_map[band[0] : band[1] + 1, c] if int(x) >= 0)
            assert len(ids) <= 1
            if not ids:
                last_id = None
                continue
            cid = next(iter(ids))
            if cid != last_id:
                out.append(comps[cid])
            last_id = cid
    elif direction == "D":
        axis = range(start, h)
        for r in axis:
            ids = set(int(x) for x in comp_map[r, band[0] : band[1] + 1] if int(x) >= 0)
            assert len(ids) <= 1
            if not ids:
                last_id = None
                continue
            cid = next(iter(ids))
            if cid != last_id:
                out.append(comps[cid])
            last_id = cid
    else:
        axis = range(start, -1, -1)
        for r in axis:
            ids = set(int(x) for x in comp_map[r, band[0] : band[1] + 1] if int(x) >= 0)
            assert len(ids) <= 1
            if not ids:
                last_id = None
                continue
            cid = next(iter(ids))
            if cid != last_id:
                out.append(comps[cid])
            last_id = cid
    return out


def _front_start(comp: dict, direction: str) -> int:
    r0, c0, r1, c1 = comp["bbox"]
    if direction == "R":
        return c1 + 1
    if direction == "L":
        return c0 - 1
    if direction == "D":
        return r1 + 1
    return r0 - 1


def _gap_to_obstacle(comp: dict, obstacle: dict | None, direction: str, shape: tuple[int, int]) -> int:
    h, w = shape
    r0, c0, r1, c1 = comp["bbox"]
    if obstacle is None:
        if direction == "R":
            return w - 1 - c1
        if direction == "L":
            return c0
        if direction == "D":
            return h - 1 - r1
        return r0
    or0, oc0, or1, oc1 = obstacle["bbox"]
    if direction == "R":
        return oc0 - c1 - 1
    if direction == "L":
        return c0 - oc1 - 1
    if direction == "D":
        return or0 - r1 - 1
    return r0 - or1 - 1


def _shift_coords(coords: np.ndarray, direction: str, dist: int) -> np.ndarray:
    dr, dc = DIRS[direction]
    delta = np.array([dr * dist, dc * dist], dtype=int)
    return coords + delta


def _detect_corner(
    comp: dict,
    incoming: str,
    comp_map: np.ndarray,
    comps: list[dict],
) -> str | None:
    r0, c0, r1, c1 = comp["bbox"]
    if not _solid_rect(comp):
        return None
    if (r1 - r0) != (c1 - c0):
        return None
    band_size = _lane_thickness(comp, incoming)
    if (r1 - r0 + 1) != band_size or (c1 - c0 + 1) != band_size:
        return None

    exits = []
    for direction in PERPENDICULAR[incoming]:
        if direction in ("U", "D"):
            band = (c0, c1)
        else:
            band = (r0, r1)
        start = _front_start(comp, direction)
        seen = _scan_lane(comp_map, comps, band, direction, start)
        if seen:
            exits.append(direction)
    if len(exits) == 1:
        return exits[0]
    return None


def _parse_lane(
    comp_map: np.ndarray,
    comps: list[dict],
    start_comp: dict,
    direction: str,
    shape: tuple[int, int],
) -> dict:
    band = _lane_band(start_comp, direction)
    seen = _scan_lane(comp_map, comps, band, direction, _front_start(start_comp, direction))
    assert seen, "lane must contain at least one component ahead"

    for comp in seen[:-1]:
        assert _lane_shaped(comp, band, direction), "intermediate lane blocks must be lane-shaped"

    last = seen[-1]
    exit_dir = _detect_corner(last, direction, comp_map, comps)
    if exit_dir is not None:
        free = seen[:-1]
        assert free, "corner lane must contain free blocks before the corner"
        return {
            "band": band,
            "movables": [start_comp] + free if start_comp["kind"] == "launcher" else free,
            "terminal": last,
            "terminal_kind": "corner",
            "exit_dir": exit_dir,
        }

    if _lane_shaped(last, band, direction):
        free = seen
        terminal = None
        terminal_kind = "boundary"
    else:
        free = seen[:-1]
        terminal = last
        terminal_kind = "wall"
    if start_comp["kind"] == "launcher":
        free = [start_comp] + free
    assert free, "lane must have at least one movable block"
    return {
        "band": band,
        "movables": free,
        "terminal": terminal,
        "terminal_kind": terminal_kind,
        "exit_dir": None,
    }


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps, comp_map = _find_components(g, bg)
    launchers = []
    for comp in comps:
        side = _touch_side(comp["bbox"], g.shape)
        if side is not None and _solid_rect(comp):
            launchers.append((comp, side))
    assert len(launchers) == 1, "expected exactly one edge launcher rectangle"

    launcher, side = launchers[0]
    launcher["kind"] = "launcher"
    direction = _side_to_dir(side)

    primary = _parse_lane(comp_map, comps, launcher, direction, g.shape)
    secondary = None
    if primary["terminal_kind"] == "corner":
        corner = primary["terminal"]
        assert corner is not None
        corner["kind"] = "corner"
        secondary = _parse_lane(comp_map, comps, corner, primary["exit_dir"], g.shape)
        secondary["movables"] = secondary["movables"][1:] if secondary["movables"] and secondary["movables"][0]["id"] == corner["id"] else secondary["movables"]
        assert secondary["movables"], "corner exit must contain movable blocks"

    out = g.copy()
    to_move = list(primary["movables"])
    if secondary is not None:
        to_move.extend(secondary["movables"])
    for comp in to_move:
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    def paint_chain(movables: list[dict], terminal: dict | None, move_dir: str) -> None:
        for i, comp in enumerate(movables):
            obstacle = movables[i + 1] if i + 1 < len(movables) else terminal
            gap = _gap_to_obstacle(comp, obstacle, move_dir, g.shape)
            assert gap > 0
            shifted = _shift_coords(comp["coords"], move_dir, gap)
            out[shifted[:, 0], shifted[:, 1]] = comp["color"]

    paint_chain(primary["movables"], primary["terminal"], direction)
    if secondary is not None:
        paint_chain(secondary["movables"], secondary["terminal"], primary["exit_dir"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push the edge launcher through a lane of blocks, deflecting the chain once at a fixed corner wall.

    Input structure:
    - A background grid contains exactly one solid launcher rectangle touching one outer edge; its motion direction points away from that edge.
    - In the same-width lane ahead sit several monochrome rectangular free blocks, and there may be one fixed wall-color corner square that turns the force into a perpendicular exit lane.
    - A lane ends either at the grid boundary or at a fixed wall block; wall blocks never move.

    Rule:
    - The launcher slides forward only until it touches the nearest block ahead, then stops.
    - The touched free block then slides in the same direction by its own original forward gap to the nearest obstacle; this gap-consumption propagates block by block.
    - If the propagated force reaches the corner wall, the wall stays fixed and the force turns 90 degrees into the unique exit lane, where the nearest free block starts a new chain with the same rule.

    Key insight:
    - This is gap transfer, not whole-lane translation: every block advances exactly by its own original forward gap, and after a turn the outgoing lane moves away from the corner rather than packing against it.
    """
    return _solve(grid)


def _paint_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> tuple[int, int, int, int]:
    grid[r0 : r0 + h, c0 : c0 + w] = color
    return r0, c0, r0 + h - 1, c0 + w - 1


def _rect_coords(box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    pts = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    return np.array(pts, dtype=int)


def _sample_lengths(count: int) -> list[int]:
    return [random.randint(1, 3) for _ in range(count)]


def _sample_gaps(count: int) -> list[int]:
    return [random.randint(1, 3) for _ in range(count)]


def _build_primary_chain(row0: int, t: int, launcher_len: int, free_lens: list[int], gaps: list[int], colors: list[int]) -> tuple[list[dict], int]:
    blocks = []
    c = 0
    blocks.append({"kind": "launcher", "color": colors[0], "bbox": (row0, c, row0 + t - 1, c + launcher_len - 1)})
    prev_end = c + launcher_len - 1
    for i, length in enumerate(free_lens):
        start = prev_end + 1 + gaps[i]
        box = (row0, start, row0 + t - 1, start + length - 1)
        blocks.append({"kind": "free", "color": colors[i + 1], "bbox": box})
        prev_end = start + length - 1
    return blocks, prev_end


def _render_from_boxes(shape: tuple[int, int], blocks: list[dict], walls: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for item in walls + blocks:
        r0, c0, r1, c1 = item["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = item["color"]
    return grid


def _canonical_output(shape: tuple[int, int], blocks: list[dict], walls: list[dict], chains: list[tuple[list[dict], str, dict | None]]) -> np.ndarray:
    out = _render_from_boxes(shape, [], walls)
    for chain, direction, terminal in chains:
        for i, block in enumerate(chain):
            obstacle = chain[i + 1] if i + 1 < len(chain) else terminal
            gap = _gap_to_obstacle(
                {"bbox": block["bbox"], "coords": _rect_coords(block["bbox"]), "color": block["color"]},
                None if obstacle is None else {"bbox": obstacle["bbox"]},
                direction,
                shape,
            )
            r0, c0, r1, c1 = block["bbox"]
            dr, dc = DIRS[direction]
            shifted = (r0 + dr * gap, c0 + dc * gap, r1 + dr * gap, c1 + dc * gap)
            rr0, cc0, rr1, cc1 = shifted
            out[rr0 : rr1 + 1, cc0 : cc1 + 1] = block["color"]
    return out


def _build_corner_case() -> tuple[np.ndarray, np.ndarray]:
    t = random.randint(1, 3)
    launcher_len = random.randint(1, 3)
    n_primary = random.randint(1, 3)
    n_exit = random.randint(1, 3)
    primary_lens = _sample_lengths(n_primary)
    primary_gaps = _sample_gaps(n_primary)
    primary_front_gap = random.randint(1, 3)
    corner_gap = random.randint(1, 3)
    exit_lens = _sample_lengths(n_exit)
    exit_gaps = _sample_gaps(max(0, n_exit - 1))
    exit_front_gap = random.randint(1, 3)
    exit_dir = random.choice(("U", "D"))
    exit_wall = bool(random.getrandbits(1))
    right_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)

    palette = random.sample(COLORS, 2 + n_primary + n_exit)
    launcher_color = palette[0]
    wall_color = palette[1]
    primary_colors = [launcher_color] + palette[2 : 2 + n_primary]
    exit_colors = palette[2 + n_primary :]

    primary_blocks, prev_end = _build_primary_chain(0, t, launcher_len, primary_lens, primary_gaps, primary_colors)
    corner_c0 = prev_end + 1 + primary_front_gap

    if exit_dir == "U":
        top_pad = random.randint(1, 3) if exit_wall else 0
        wall_h = random.randint(1, 2) if exit_wall else 0
        above = corner_gap + sum(exit_lens) + sum(exit_gaps) + exit_front_gap + top_pad + wall_h
        row0 = above
        h = row0 + t + bottom_margin
        corner_box = (row0, corner_c0, row0 + t - 1, corner_c0 + t - 1)

        exit_blocks = []
        bottom = row0 - corner_gap - 1
        for i, length in enumerate(exit_lens):
            top = bottom - length + 1
            exit_blocks.append({"kind": "free", "color": exit_colors[i], "bbox": (top, corner_c0, bottom, corner_c0 + t - 1)})
            if i < len(exit_gaps):
                bottom = top - exit_gaps[i] - 1
        walls = [{"kind": "corner", "color": wall_color, "bbox": corner_box}]
        if exit_wall:
            walls.append(
                {
                    "kind": "wall",
                    "color": wall_color,
                    "bbox": (top_pad, corner_c0 - 1, top_pad + wall_h - 1, corner_c0 + t),
                }
            )
            terminal = walls[-1]
        else:
            terminal = None
    else:
        top_margin = random.randint(1, 3)
        wall_h = random.randint(1, 2) if exit_wall else 0
        bottom_pad = random.randint(1, 3) if exit_wall else 0
        row0 = top_margin
        below = corner_gap + sum(exit_lens) + sum(exit_gaps) + exit_front_gap + wall_h + bottom_pad
        h = row0 + t + below
        corner_box = (row0, corner_c0, row0 + t - 1, corner_c0 + t - 1)

        exit_blocks = []
        top = row0 + t + corner_gap
        for i, length in enumerate(exit_lens):
            bottom = top + length - 1
            exit_blocks.append({"kind": "free", "color": exit_colors[i], "bbox": (top, corner_c0, bottom, corner_c0 + t - 1)})
            if i < len(exit_gaps):
                top = bottom + exit_gaps[i] + 1
        walls = [{"kind": "corner", "color": wall_color, "bbox": corner_box}]
        if exit_wall:
            wall_top = h - bottom_pad - wall_h
            walls.append(
                {
                    "kind": "wall",
                    "color": wall_color,
                    "bbox": (wall_top, corner_c0 - 1, wall_top + wall_h - 1, corner_c0 + t),
                }
            )
            terminal = walls[-1]
        else:
            terminal = None

    w = corner_c0 + t + right_margin
    assert h <= 30 and w <= 30
    assert corner_c0 >= 1 and corner_c0 + t <= w - 1

    for block in primary_blocks:
        r0, c0, r1, c1 = block["bbox"]
        block["bbox"] = (r0 + row0, c0, r1 + row0, c1)

    primary_terminal = {"kind": "corner", "color": wall_color, "bbox": corner_box}
    inp = _render_from_boxes((h, w), primary_blocks + exit_blocks, walls)
    expected = _canonical_output(
        (h, w),
        primary_blocks + exit_blocks,
        walls,
        [
            (primary_blocks, "R", primary_terminal),
            (exit_blocks, exit_dir, terminal),
        ],
    )

    assert np.any(inp != expected)
    assert len(set(block["color"] for block in primary_blocks + exit_blocks)) == len(primary_blocks) + len(exit_blocks)
    assert primary_front_gap >= 1 and corner_gap >= 1
    assert any(g > 1 for g in primary_gaps + [primary_front_gap, corner_gap] + exit_gaps + [exit_front_gap]) or any(
        length > 1 for length in primary_lens + exit_lens + [launcher_len]
    )

    first_exit_shift = exit_gaps[0] if exit_gaps else exit_front_gap
    if exit_dir == "U":
        original_gap = corner_box[0] - exit_blocks[0]["bbox"][2] - 1
        moved_bottom = exit_blocks[0]["bbox"][2] - first_exit_shift
        final_gap = corner_box[0] - moved_bottom - 1
    else:
        original_gap = exit_blocks[0]["bbox"][0] - corner_box[2] - 1
        moved_top = exit_blocks[0]["bbox"][0] + first_exit_shift
        final_gap = moved_top - corner_box[2] - 1
    assert original_gap == corner_gap
    assert final_gap == corner_gap + first_exit_shift
    return inp, expected


def _build_straight_case() -> tuple[np.ndarray, np.ndarray]:
    t = random.randint(1, 3)
    launcher_len = random.randint(1, 3)
    n_primary = random.randint(2, 3)
    primary_lens = _sample_lengths(n_primary)
    primary_gaps = _sample_gaps(n_primary)
    front_gap = random.randint(1, 3)
    wall_end = bool(random.getrandbits(1))
    right_margin = random.randint(1, 3) if wall_end else 0
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    wall_w = random.randint(1, 2) if wall_end else 0

    palette = random.sample(COLORS, 2 + n_primary)
    launcher_color = palette[0]
    wall_color = palette[1]
    primary_colors = [launcher_color] + palette[2:]

    primary_blocks, prev_end = _build_primary_chain(top_margin, t, launcher_len, primary_lens, primary_gaps, primary_colors)
    wall_c0 = prev_end + 1 + front_gap
    w = wall_c0 + wall_w + right_margin if wall_end else prev_end + 1 + front_gap
    h = top_margin + t + bottom_margin
    assert h <= 30 and w <= 30

    walls = []
    terminal = None
    if wall_end:
        wall_box = (top_margin - 1, wall_c0 + front_gap, top_margin + t, wall_c0 + front_gap + wall_w - 1)
        walls.append({"kind": "wall", "color": wall_color, "bbox": wall_box})
        terminal = walls[0]

    inp = _render_from_boxes((h, w), primary_blocks, walls)
    expected = _canonical_output((h, w), primary_blocks, walls, [(primary_blocks, "R", terminal)])
    assert np.any(inp != expected)
    assert any(g > 1 for g in primary_gaps + [front_gap]) or any(length > 1 for length in primary_lens + [launcher_len])
    return inp, expected


def _validate_scene(inp: np.ndarray, expected: np.ndarray) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    bg = _get_bg(inp)
    comps, _ = _find_components(inp, bg)
    launchers = [comp for comp in comps if _touch_side(comp["bbox"], inp.shape) is not None and _solid_rect(comp)]
    if len(launchers) != 1:
        return False
    if np.array_equal(inp, expected):
        return False
    return True


def _rotate_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = random.randint(0, 3)
    return np.rot90(inp, k=k).copy(), np.rot90(out, k=k).copy()


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            if random.random() < 0.65:
                inp, expected = _build_corner_case()
            else:
                inp, expected = _build_straight_case()
            inp, expected = _rotate_pair(inp, expected)
            if not _validate_scene(inp, expected):
                continue
            if not np.array_equal(main(inp), expected):
                continue
            return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid Newton-chain example")
