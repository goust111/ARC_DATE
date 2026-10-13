from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME = 3
SPRITE = 5
INNER_COLORS = (1, 2, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GAP_SPECS = (
    {"name": "top_left", "offset": (0, 1), "dir": (-1, -1)},
    {"name": "top_right", "offset": (0, 2), "dir": (-1, 1)},
    {"name": "bottom_left", "offset": (3, 1), "dir": (1, -1)},
    {"name": "bottom_right", "offset": (3, 2), "dir": (1, 1)},
    {"name": "left_top", "offset": (1, 0), "dir": (-1, -1)},
    {"name": "left_bottom", "offset": (2, 0), "dir": (1, -1)},
    {"name": "right_top", "offset": (1, 3), "dir": (-1, 1)},
    {"name": "right_bottom", "offset": (2, 3), "dir": (1, 1)},
)
GAP_BY_NAME = {spec["name"]: spec for spec in GAP_SPECS}
GAP_OFFSETS = {spec["offset"] for spec in GAP_SPECS}

CONTACT_BASE_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (0, 2)),
)
GENERIC_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
)
TRANSFORMS = ("identity", "flip_h", "flip_v", "rot180")


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_blocks(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    blocks: list[dict] = []
    for fr in range(h - 3):
        for fc in range(w - 3):
            inner = grid[fr + 1 : fr + 3, fc + 1 : fc + 3]
            inner_color = int(inner[0, 0])
            if inner_color in (BG, FRAME, SPRITE):
                continue
            if not np.all(inner == inner_color):
                continue

            ok = True
            gaps: list[dict] = []
            for dr in range(4):
                for dc in range(4):
                    if 1 <= dr <= 2 and 1 <= dc <= 2:
                        continue
                    value = int(grid[fr + dr, fc + dc])
                    if (dr, dc) in GAP_OFFSETS:
                        if value == BG:
                            gaps.append(next(spec for spec in GAP_SPECS if spec["offset"] == (dr, dc)))
                        elif value != FRAME:
                            ok = False
                            break
                    elif value != FRAME:
                        ok = False
                        break
                if not ok:
                    break
            if not ok or not (1 <= len(gaps) <= 2):
                continue
            blocks.append(
                {
                    "frame": (fr, fc),
                    "inner": (fr + 1, fc + 1),
                    "color": inner_color,
                    "gaps": sorted(gaps, key=lambda spec: spec["offset"]),
                }
            )

    unique: list[dict] = []
    seen_frames: set[tuple[int, int]] = set()
    for block in blocks:
        if block["frame"] not in seen_frames:
            seen_frames.add(block["frame"])
            unique.append(block)
    unique.sort(key=lambda block: (block["inner"][0], block["inner"][1]))
    assert 1 <= len(unique) <= 2, "expected 1-2 framed blocks"
    return unique


def _make_sprite_index_map(shape: tuple[int, int], sprites: list[np.ndarray]) -> np.ndarray:
    sprite_map = np.full(shape, -1, dtype=int)
    for idx, coords in enumerate(sprites):
        sprite_map[coords[:, 0], coords[:, 1]] = idx
    return sprite_map


def _simulate(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    sprites = _components_of_color(g, SPRITE)
    sprite_map = _make_sprite_index_map(g.shape, sprites)
    sprite_hit = [False] * len(sprites)
    blocks = _find_blocks(g)

    rays: list[dict] = []
    for block_order, block in enumerate(blocks):
        fr, fc = block["frame"]
        for gap_order, gap in enumerate(block["gaps"]):
            gr = fr + gap["offset"][0]
            gc = fc + gap["offset"][1]
            rays.append(
                {
                    "order": (block_order, gap_order),
                    "color": int(block["color"]),
                    "dir": gap["dir"],
                    "pos": (gr, gc),
                    "active": True,
                }
            )

    for ray in rays:
        r, c = ray["pos"]
        assert out[r, c] == BG, "gap cells must be background in the input"
        out[r, c] = ray["color"]

    max_steps = g.shape[0] * g.shape[1] * max(1, len(rays))
    for _ in range(max_steps):
        any_active = False
        progressed = False
        for ray in rays:
            if not ray["active"]:
                continue
            any_active = True
            r, c = ray["pos"]
            dr, dc = ray["dir"]
            nr, nc = r + dr, c + dc
            if not _in_bounds(g.shape, nr, nc):
                ray["active"] = False
                continue
            if out[nr, nc] == BG:
                out[nr, nc] = ray["color"]
                ray["pos"] = (nr, nc)
                progressed = True
                continue
            sprite_idx = int(sprite_map[nr, nc])
            if sprite_idx >= 0 and not sprite_hit[sprite_idx]:
                out[sprites[sprite_idx][:, 0], sprites[sprite_idx][:, 1]] = ray["color"]
                sprite_hit[sprite_idx] = True
                ray["active"] = False
                progressed = True
                continue
            ray["active"] = False
        if not any_active:
            break
        if not progressed and not any(ray["active"] for ray in rays):
            break
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor gray sprites when the first diagonal ray from a framed block reaches them, and stop the ray there.

    Input structure:
    - The background is black.
    - One or two green 4x4 frames each contain a monochrome 2x2 inner block.
    - Each frame has one or two missing side pixels that act as ray gaps, and separate gray 4-connected sprites are scattered elsewhere.

    Rule:
    - Detect every frame gap and convert it to a diagonal ray whose direction is determined by the gap side/offset and whose color is the inner 2x2 block color.
    - All rays start from their gap cells; at each distance step, shorter paths act first, and ties use the block order (top-to-bottom, then left-to-right) and then the gap order.
    - A ray paints only background cells. If its next cell is a gray sprite, the ray stops before entering that cell and recolors the entire sprite to the ray color. If the next cell is out of bounds or any already non-background cell, the ray simply stops.

    Key insight:
    - A gray sprite is both a receiver and a shield: the first ray that reaches it recolors the whole sprite, and that same hit prevents the ray from ever reaching sprites farther along the same diagonal.
    """
    return _simulate(np.asarray(grid, dtype=int))


def _transform_point(point: tuple[int, int], n: int, name: str) -> tuple[int, int]:
    r, c = point
    if name == "identity":
        return r, c
    if name == "flip_h":
        return r, n - 1 - c
    if name == "flip_v":
        return n - 1 - r, c
    if name == "rot180":
        return n - 1 - r, n - 1 - c
    raise ValueError(name)


def _transform_grid(grid: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return grid.copy()
    if name == "flip_h":
        return np.fliplr(grid).copy()
    if name == "flip_v":
        return np.flipud(grid).copy()
    if name == "rot180":
        return np.rot90(grid, 2).copy()
    raise ValueError(name)


def _transform_sprite_record(record: dict, n: int, name: str) -> dict:
    return {
        "expected": int(record["expected"]),
        "coords": [_transform_point((int(r), int(c)), n, name) for r, c in record["coords"]],
    }


def _all_ray_lines(blocks: list[dict], n: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for block in blocks:
        fr, fc = block["frame"]
        for gap_name in block["gap_names"]:
            spec = GAP_BY_NAME[gap_name]
            r = fr + spec["offset"][0]
            c = fc + spec["offset"][1]
            dr, dc = spec["dir"]
            while 0 <= r < n and 0 <= c < n:
                cells.add((r, c))
                r += dr
                c += dc
    return cells


def _blocked_with_margin(coords: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        if (r, c) in occupied:
            return True
        for dr, dc in NEIGHBORS8:
            if (r + dr, c + dc) in occupied:
                return True
    return False


def _paint_block(grid: np.ndarray, fr: int, fc: int, inner_color: int, gap_names: list[str]) -> dict:
    grid[fr : fr + 4, fc : fc + 4] = FRAME
    grid[fr + 1 : fr + 3, fc + 1 : fc + 3] = inner_color
    for gap_name in gap_names:
        spec = GAP_BY_NAME[gap_name]
        grid[fr + spec["offset"][0], fc + spec["offset"][1]] = BG
    return {"frame": (fr, fc), "color": int(inner_color), "gap_names": list(gap_names)}


def _shape_for_direction(direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    base = random.choice(CONTACT_BASE_SHAPES)
    dr, dc = direction
    shaped: list[tuple[int, int]] = []
    for rr, cc in base:
        nr = rr if dr > 0 else -rr
        nc = cc if dc > 0 else -cc
        shaped.append((nr, nc))
    return tuple(shaped)


def _sprite_coords_from_contact(contact: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    cr, cc = contact
    return [(cr + dr, cc + dc) for dr, dc in _shape_for_direction(direction)]


def _generic_shape_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    variants = []
    arr = np.array(shape, dtype=int)
    for k in range(4):
        pts = arr.copy()
        for _ in range(k):
            pts = np.stack([pts[:, 1], -pts[:, 0]], axis=1)
        pts = pts - pts.min(axis=0, keepdims=True)
        variants.append(tuple(sorted((int(r), int(c)) for r, c in pts)))
    return list(dict.fromkeys(variants))


def _place_sprite_coords(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    if any(not _in_bounds((h, w), r, c) for r, c in coords):
        return False
    if _blocked_with_margin(coords, occupied):
        return False
    for r, c in coords:
        grid[r, c] = SPRITE
        occupied.add((r, c))
    return True


def _place_contact_sprite(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    contact: tuple[int, int],
    direction: tuple[int, int],
    expected_color: int,
) -> dict | None:
    coords = _sprite_coords_from_contact(contact, direction)
    if not _place_sprite_coords(grid, coords, occupied):
        return None
    return {"expected": int(expected_color), "coords": coords}


def _place_free_sprite(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    expected_color: int,
    prefer_adjacent_to: set[tuple[int, int]] | None = None,
) -> dict | None:
    n = grid.shape[0]
    variants = []
    for shape in GENERIC_SHAPES:
        variants.extend(_generic_shape_variants(shape))
    random.shuffle(variants)
    anchors = [(r, c) for r in range(n) for c in range(n)]
    random.shuffle(anchors)
    for shape in variants:
        h = max(r for r, _ in shape) + 1
        w = max(c for _, c in shape) + 1
        for ar, ac in anchors:
            if ar + h > n or ac + w > n:
                continue
            coords = [(ar + dr, ac + dc) for dr, dc in shape]
            if any(cell in forbidden for cell in coords):
                continue
            if prefer_adjacent_to is not None:
                adjacent = False
                for r, c in coords:
                    for dr, dc in NEIGHBORS8:
                        if (r + dr, c + dc) in prefer_adjacent_to:
                            adjacent = True
                            break
                    if adjacent:
                        break
                if not adjacent:
                    continue
            if _place_sprite_coords(grid, coords, occupied):
                return {"expected": int(expected_color), "coords": coords}
    return None


def _ray_contact(origin: tuple[int, int], direction: tuple[int, int], distance: int) -> tuple[int, int]:
    return origin[0] + direction[0] * distance, origin[1] + direction[1] * distance


def _sprite_origin_from_block(block: dict, gap_name: str) -> tuple[int, int]:
    spec = GAP_BY_NAME[gap_name]
    fr, fc = block["frame"]
    return fr + spec["offset"][0], fc + spec["offset"][1]


def _expected_winner(blocks: list[dict], block_a_idx: int, gap_a: str, dist_a: int, block_b_idx: int, gap_b: str, dist_b: int) -> int:
    if dist_a != dist_b:
        return int(blocks[block_a_idx]["color"] if dist_a < dist_b else blocks[block_b_idx]["color"])
    order_a = (blocks[block_a_idx]["frame"][0] + 1, blocks[block_a_idx]["frame"][1] + 1, GAP_BY_NAME[gap_a]["offset"])
    order_b = (blocks[block_b_idx]["frame"][0] + 1, blocks[block_b_idx]["frame"][1] + 1, GAP_BY_NAME[gap_b]["offset"])
    return int(blocks[block_a_idx]["color"] if order_a < order_b else blocks[block_b_idx]["color"])


def _build_single_occlusion(n: int) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full((n, n), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()
    color = random.choice(INNER_COLORS)
    fr = random.randint(max(2, n // 2 - 3), min(n - 7, n // 2))
    fc = random.randint(2, 3)
    gap_names = ["right_bottom"]
    if random.random() < 0.6:
        gap_names.append("right_top")
    block = _paint_block(grid, fr, fc, color, gap_names)
    for r in range(fr, fr + 4):
        for c in range(fc, fc + 4):
            occupied.add((r, c))
    block["frame"] = (fr, fc)
    blocks = [block]
    expected: list[dict] = []

    origin = _sprite_origin_from_block(block, "right_bottom")
    front = _place_contact_sprite(
        grid,
        occupied,
        _ray_contact(origin, (1, 1), random.randint(4, 6)),
        (1, 1),
        color,
    )
    if front is None:
        return None
    expected.append(front)

    back = _place_contact_sprite(
        grid,
        occupied,
        _ray_contact(origin, (1, 1), random.randint(8, 10)),
        (1, 1),
        SPRITE,
    )
    if back is None:
        return None
    expected.append(back)

    all_lines = _all_ray_lines(blocks, n)
    near_miss = _place_free_sprite(grid, occupied, all_lines, SPRITE, prefer_adjacent_to=all_lines)
    if near_miss is None:
        return None
    expected.append(near_miss)

    if "right_top" in gap_names:
        origin2 = _sprite_origin_from_block(block, "right_top")
        extra = _place_contact_sprite(
            grid,
            occupied,
            _ray_contact(origin2, (-1, 1), random.randint(4, 6)),
            (-1, 1),
            color,
        )
        if extra is None:
            return None
        expected.append(extra)
    else:
        extra = _place_free_sprite(grid, occupied, all_lines, SPRITE)
        if extra is None:
            return None
        expected.append(extra)

    return grid, expected


def _pick_competition_blocks(n: int, tie: bool) -> tuple[list[dict], tuple[int, int, int, int, str, str]] | None:
    contest_r = random.randint(n // 2 - 1, n // 2 + 1)
    contest_c = random.randint(n // 2 - 1, n // 2 + 1)
    for _ in range(60):
        dist_a = random.randint(4, 6)
        dist_b = dist_a if tie else dist_a + random.choice((1, 2))
        gap_a = random.choice(("bottom_right", "right_bottom"))
        gap_b = random.choice(("bottom_left", "left_bottom"))

        spec_a = GAP_BY_NAME[gap_a]
        spec_b = GAP_BY_NAME[gap_b]
        origin_a = (contest_r - dist_a, contest_c - dist_a)
        origin_b = (contest_r - dist_b, contest_c + dist_b)
        fr_a = origin_a[0] - spec_a["offset"][0]
        fc_a = origin_a[1] - spec_a["offset"][1]
        fr_b = origin_b[0] - spec_b["offset"][0]
        fc_b = origin_b[1] - spec_b["offset"][1]
        if not (1 <= fr_a <= n - 6 and 1 <= fc_a <= n - 6 and 1 <= fr_b <= n - 6 and 1 <= fc_b <= n - 6):
            continue
        if max(fr_a, fr_b) < min(fr_a + 4, fr_b + 4) and max(fc_a, fc_b) < min(fc_a + 4, fc_b + 4):
            continue

        extra_a = ["right_top"] if random.random() < 0.5 else []
        extra_b = ["left_top"] if random.random() < 0.5 else []
        block_a = {"frame": (fr_a, fc_a), "color": 0, "gap_names": [gap_a] + extra_a}
        block_b = {"frame": (fr_b, fc_b), "color": 0, "gap_names": [gap_b] + extra_b}
        return [block_a, block_b], (contest_r, contest_c, dist_a, dist_b, gap_a, gap_b)
    return None


def _build_competition(n: int, tie: bool) -> tuple[np.ndarray, list[dict]] | None:
    picked = _pick_competition_blocks(n, tie)
    if picked is None:
        return None
    block_specs, detail = picked
    contest_r, contest_c, dist_a, dist_b, gap_a, gap_b = detail

    colors = random.sample(INNER_COLORS, 2)
    for idx, block in enumerate(block_specs):
        block["color"] = int(colors[idx])

    grid = np.full((n, n), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()
    blocks: list[dict] = []
    for block in block_specs:
        painted = _paint_block(grid, block["frame"][0], block["frame"][1], block["color"], block["gap_names"])
        painted["gap_names"] = list(block["gap_names"])
        blocks.append(painted)
        fr, fc = block["frame"]
        for r in range(fr, fr + 4):
            for c in range(fc, fc + 4):
                occupied.add((r, c))

    expected: list[dict] = []
    winner = _expected_winner(blocks, 0, gap_a, dist_a, 1, gap_b, dist_b)
    contest = _place_contact_sprite(grid, occupied, (contest_r, contest_c), (1, 1), winner)
    if contest is None:
        return None
    expected.append(contest)

    winner_idx = 0 if winner == blocks[0]["color"] else 1
    winner_gap = gap_a if winner_idx == 0 else gap_b
    winner_dist = dist_a if winner_idx == 0 else dist_b
    winner_origin = _sprite_origin_from_block(blocks[winner_idx], winner_gap)
    winner_dir = GAP_BY_NAME[winner_gap]["dir"]
    blocked = _place_contact_sprite(
        grid,
        occupied,
        _ray_contact(winner_origin, winner_dir, winner_dist + random.randint(3, 4)),
        winner_dir,
        SPRITE,
    )
    if blocked is None:
        return None
    expected.append(blocked)

    all_lines = _all_ray_lines(blocks, n)
    near_miss = _place_free_sprite(grid, occupied, all_lines, SPRITE, prefer_adjacent_to=all_lines)
    if near_miss is None:
        return None
    expected.append(near_miss)

    spare = _place_free_sprite(grid, occupied, all_lines, SPRITE)
    if spare is None:
        return None
    expected.append(spare)

    if len(expected) < 5 and random.random() < 0.55:
        optional_gap = None
        optional_block = None
        for candidate_idx, block in enumerate(blocks):
            for gap_name in block_specs[candidate_idx]["gap_names"][1:]:
                optional_gap = gap_name
                optional_block = candidate_idx
                break
            if optional_gap is not None:
                break
        if optional_gap is not None and optional_block is not None:
            origin = _sprite_origin_from_block(blocks[optional_block], optional_gap)
            direction = GAP_BY_NAME[optional_gap]["dir"]
            extra = _place_contact_sprite(
                grid,
                occupied,
                _ray_contact(origin, direction, random.randint(4, 6)),
                direction,
                blocks[optional_block]["color"],
            )
            if extra is not None:
                expected.append(extra)

    return grid, expected


def _semantic_audit(grid: np.ndarray, out: np.ndarray, expected: list[dict]) -> bool:
    if np.array_equal(grid, out):
        return False
    if not np.array_equal((grid == FRAME), (out == FRAME)):
        return False
    if not np.array_equal(grid[(grid != BG) & (grid != FRAME) & (grid != SPRITE)], out[(grid != BG) & (grid != FRAME) & (grid != SPRITE)]):
        return False

    hit_count = 0
    gray_count = 0
    for record in expected:
        vals = {int(out[r, c]) for r, c in record["coords"]}
        if vals != {int(record["expected"])}:
            return False
        if int(record["expected"]) == SPRITE:
            gray_count += 1
        else:
            hit_count += 1
    return hit_count >= 1 and gray_count >= 1


def _build_case() -> tuple[np.ndarray, list[dict]] | None:
    n = random.randint(18, 24)
    scenario = random.choices(
        ("single", "shorter", "tie"),
        weights=(0.35, 0.4, 0.25),
        k=1,
    )[0]
    if scenario == "single":
        built = _build_single_occlusion(n)
    elif scenario == "shorter":
        built = _build_competition(n, tie=False)
    else:
        built = _build_competition(n, tie=True)
    if built is None:
        return None

    grid, expected = built
    transform = random.choice(TRANSFORMS)
    transformed_grid = _transform_grid(grid, transform)
    transformed_expected = [_transform_sprite_record(record, n, transform) for record in expected]
    return transformed_grid, transformed_expected


def generate_input() -> np.ndarray:
    for _ in range(500):
        built = _build_case()
        if built is None:
            continue
        grid, expected = built
        out = main(grid)
        if _semantic_audit(grid, out, expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
