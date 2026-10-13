import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4]
GEN_ATTEMPTS = 400


BASE_SHAPES = [
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)),
    ((0, 1), (1, 1), (2, 1), (2, 0), (2, 2), (0, 0)),
]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox(coords: set[Coord] | list[Coord] | tuple[Coord, ...]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize(coords: set[Coord] | list[Coord] | tuple[Coord, ...]) -> tuple[Coord, ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shift(coords: set[Coord] | list[Coord] | tuple[Coord, ...], dr: int, dc: int) -> set[Coord]:
    return {(r + dr, c + dc) for r, c in coords}


def _in_bounds(shape: tuple[int, int], coords: set[Coord]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in coords)


def _component_min_manhattan(a: set[Coord], b: set[Coord]) -> int:
    return min(abs(ar - br) + abs(ac - bc) for ar, ac in a for br, bc in b)


def _cc4(mask: np.ndarray) -> list[set[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: set[Coord] = set()
            while q:
                cr, cc = q.popleft()
                comp.add((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_solid_2x2(comp: set[Coord]) -> bool:
    if len(comp) != 4:
        return False
    r0, c0, r1, c1 = _bbox(comp)
    return r1 == r0 + 1 and c1 == c0 + 1 and comp == {
        (r0, c0),
        (r0, c0 + 1),
        (r0 + 1, c0),
        (r0 + 1, c0 + 1),
    }


def _rotate_origin_cw(coords: tuple[Coord, ...]) -> tuple[Coord, ...]:
    return tuple((c, -r) for r, c in coords)


def _variants(shape: tuple[Coord, ...]) -> list[tuple[Coord, ...]]:
    out: set[tuple[Coord, ...]] = set()
    cur = tuple(shape)
    for _ in range(4):
        out.add(_normalize(cur))
        flipped = tuple((r, -c) for r, c in cur)
        out.add(_normalize(flipped))
        cur = _rotate_origin_cw(cur)
    return sorted(out)


def _is_rot_asymmetric(shape: tuple[Coord, ...]) -> bool:
    base = _normalize(shape)
    cur = base
    for _ in range(3):
        cur = _normalize(_rotate_origin_cw(cur))
        if cur == base:
            return False
    return True


SHAPES_BY_SIZE: dict[int, list[tuple[Coord, ...]]] = {}
for base in BASE_SHAPES:
    for variant in _variants(base):
        if _is_rot_asymmetric(variant):
            SHAPES_BY_SIZE.setdefault(len(variant), [])
            if variant not in SHAPES_BY_SIZE[len(variant)]:
                SHAPES_BY_SIZE[len(variant)].append(variant)


def _rotate_coords_cw(coords: set[Coord], k: int) -> set[Coord]:
    k %= 4
    if k == 0:
        return set(coords)
    r0, c0, r1, c1 = _bbox(coords)
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    out = {(r, c) for r, c in coords}
    for _ in range(k):
        nxt: set[Coord] = set()
        for r, c in out:
            dr = r - cr
            dc = c - cc
            nr = cr + dc
            nc = cc - dr
            nxt.add((int(round(nr)), int(round(nc))))
        out = nxt
    return out


def _unique_sightline(sprite: set[Coord], block: set[Coord], occupied: set[Coord]) -> str | None:
    directions: set[str] = set()
    for sr, sc in sprite:
        for br, bc in block:
            if sr == br:
                step = 1 if bc > sc else -1
                clear = True
                for cc in range(sc + step, bc, step):
                    if (sr, cc) in occupied:
                        clear = False
                        break
                if clear:
                    directions.add("right" if bc > sc else "left")
            if sc == bc:
                step = 1 if br > sr else -1
                clear = True
                for rr in range(sr + step, br, step):
                    if (rr, sc) in occupied:
                        clear = False
                        break
                if clear:
                    directions.add("down" if br > sr else "up")
    return next(iter(directions)) if len(directions) == 1 else None


def _slide_delta(coords: set[Coord], direction: str, block_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = block_bbox
    sr0, sc0, sr1, sc1 = _bbox(coords)
    if direction == "right":
        return 0, c0 - 1 - sc1
    if direction == "left":
        return 0, c1 + 1 - sc0
    if direction == "down":
        return r0 - 1 - sr1, 0
    if direction == "up":
        return r1 + 1 - sr0, 0
    raise ValueError(direction)


def _apply_delta(coords: set[Coord], delta: tuple[int, int]) -> set[Coord]:
    dr, dc = delta
    return {(r + dr, c + dc) for r, c in coords}


def _final_coords(
    sprite: set[Coord],
    block_bbox: tuple[int, int, int, int],
    direction: str,
    mode: str,
) -> set[Coord]:
    k = len(sprite) % 4
    rotated = _rotate_coords_cw(sprite, k) if mode != "none" else set(sprite)
    if mode == "post_slide":
        delta = _slide_delta(sprite, direction, block_bbox)
        return _apply_delta(rotated, delta)
    delta = _slide_delta(rotated, direction, block_bbox)
    return _apply_delta(rotated, delta)


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    scene: list[dict] = []
    for color in sorted(set(map(int, g.flat)) - {bg}):
        comps = _cc4(g == color)
        block_candidates = [comp for comp in comps if _is_solid_2x2(comp)]
        sprite_candidates = [comp for comp in comps if not _is_solid_2x2(comp)]
        assert len(block_candidates) == 1, f"expected one block for color {color}"
        assert len(sprite_candidates) == 1, f"expected one sprite for color {color}"
        sprite = sprite_candidates[0]
        block = block_candidates[0]
        scene.append(
            {
                "color": color,
                "sprite": sprite,
                "block": block,
                "block_bbox": _bbox(block),
                "size": len(sprite),
                "distance": _component_min_manhattan(sprite, block),
                "top_row": _bbox(sprite)[0],
                "left_col": _bbox(sprite)[1],
            }
        )
    scene.sort(key=lambda item: (item["distance"], item["top_row"], item["left_col"], item["color"]))
    return bg, scene


def _solve_variant(
    grid: np.ndarray,
    *,
    visibility: str = "dynamic",
    rotate_mode: str = "normal",
    return_meta: bool = False,
):
    g = np.asarray(grid, dtype=int)
    bg, scene = _parse_scene(g)
    out = g.copy()
    original_occupied = {tuple(map(int, rc)) for rc in np.argwhere(g != bg)}
    current_occupied = set(original_occupied)
    meta: list[dict] = []
    for item in scene:
        sprite = set(item["sprite"])
        block = set(item["block"])
        visible_occupied = current_occupied if visibility == "dynamic" else original_occupied
        direction = _unique_sightline(sprite, block, visible_occupied)
        status = "blocked"
        final = set(sprite)
        if direction is not None:
            candidate = _final_coords(sprite, item["block_bbox"], direction, rotate_mode)
            if (
                candidate
                and _in_bounds(g.shape, candidate)
                and not (candidate & (current_occupied - sprite))
            ):
                status = "moved"
                final = candidate
                for r, c in sprite:
                    out[r, c] = bg
                for r, c in final:
                    out[r, c] = item["color"]
                current_occupied -= sprite
                current_occupied |= final
        meta.append(
            {
                "color": item["color"],
                "size": item["size"],
                "distance": item["distance"],
                "direction": direction,
                "status": status,
                "sprite": sprite,
                "final": final,
            }
        )
    return (out, meta) if return_meta else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each small sprite by its own pixel-count code, then slide it along its sightline until it docks beside the same-color 2x2 block.

    Input structure:
    - Black is the dominant background.
    - Each used color has exactly one filled 2x2 block and one separate monochrome 4-connected sprite of 3-6 pixels.
    - Every sprite matches exactly one block of the same color, and the input layout gives a unique horizontal or vertical clear sightline between them.

    Rule:
    - For every sprite, count its pixels P and rotate the sprite clockwise by (P mod 4) quarter-turns around its bounding-box center.
    - Process sprites in ascending sprite-to-block Manhattan distance; ties use the smaller sprite row.
    - At the moment a sprite is processed, if its original position still has a unique unobstructed sightline to its same-color block, slide the rotated sprite along that direction until its nearest pixel is exactly one cell away from the block edge.
    - If previously docked sprites now block that sightline, leave the current sprite unchanged in its original position.

    Key insight:
    - The sprite's own area encodes the rotation, and that rotation must happen before sliding because the rotated silhouette changes which translated position is the first valid docking pose beside the block.
    """
    return _solve_variant(grid)


def _paint_scene(
    shape: tuple[int, int],
    block_positions: dict[int, tuple[int, int]],
    sprite_positions: dict[int, set[Coord]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color, (r0, c0) in block_positions.items():
        grid[r0 : r0 + 2, c0 : c0 + 2] = color
    for color, coords in sprite_positions.items():
        for r, c in coords:
            grid[r, c] = color
    return grid


def _random_blocks(
    h: int,
    w: int,
    colors: list[int],
    preplaced: dict[int, tuple[int, int]] | None = None,
    min_sep: int = 5,
) -> dict[int, tuple[int, int]] | None:
    placed = {} if preplaced is None else dict(preplaced)
    remaining = [color for color in colors if color not in placed]
    for color in remaining:
        options = [(r, c) for r in range(1, h - 2) for c in range(1, w - 2)]
        random.shuffle(options)
        picked = None
        for r, c in options:
            ok = True
            for or0, oc0 in placed.values():
                if max(abs(r - or0), abs(c - oc0)) < min_sep:
                    ok = False
                    break
            if ok:
                picked = (r, c)
                break
        if picked is None:
            return None
        placed[color] = picked
    return placed


def _blocked_layout(
    h: int,
    w: int,
    colors: list[int],
    order: list[int],
) -> tuple[dict[int, tuple[int, int]], dict[int, str], Coord] | None:
    blocker, blocked = order[0], order[1]
    for _ in range(80):
        forced_dirs: dict[int, str] = {}
        if random.choice([False, True]):
            dir_b = random.choice(["right", "left"])
            if dir_b == "right":
                c_b = random.randint(max(7, w // 2), w - 4)
                target_cols = list(range(3, c_b - 1))
            else:
                c_b = random.randint(1, min(w // 2 - 2, w - 4))
                target_cols = list(range(c_b + 3, w - 3))
            if not target_cols:
                continue
            r_b = random.randint(2, h - 4)
            lane_r = r_b + random.choice([0, 1])
            above = random.choice([False, True])
            dir_a = "down" if above else "up"
            a_r = lane_r - 2 if above else lane_r + 1
            if not (0 <= a_r <= h - 2):
                continue
            target = (lane_r, random.choice(target_cols))
            a_c_choices = [target[1] - 1, target[1], target[1] - 2]
            random.shuffle(a_c_choices)
            a_c = next((c for c in a_c_choices if 0 <= c <= w - 2), None)
            if a_c is None:
                continue
        else:
            dir_b = random.choice(["down", "up"])
            if dir_b == "down":
                r_b = random.randint(max(7, h // 2), h - 4)
                target_rows = list(range(3, r_b - 1))
            else:
                r_b = random.randint(1, min(h // 2 - 2, h - 4))
                target_rows = list(range(r_b + 3, h - 3))
            if not target_rows:
                continue
            c_b = random.randint(2, w - 4)
            lane_c = c_b + random.choice([0, 1])
            left = random.choice([False, True])
            dir_a = "right" if left else "left"
            a_c = lane_c - 2 if left else lane_c + 1
            if not (0 <= a_c <= w - 2):
                continue
            target = (random.choice(target_rows), lane_c)
            a_r_choices = [target[0] - 1, target[0], target[0] - 2]
            random.shuffle(a_r_choices)
            a_r = next((r for r in a_r_choices if 0 <= r <= h - 2), None)
            if a_r is None:
                continue
        block_positions = {
            blocker: (a_r, a_c),
            blocked: (r_b, c_b),
        }
        if max(abs(a_r - r_b), abs(a_c - c_b)) < 2:
            continue
        forced_dirs[blocker] = dir_a
        forced_dirs[blocked] = dir_b
        other_colors = [color for color in colors if color not in block_positions]
        filled = _random_blocks(h, w, other_colors, block_positions, min_sep=5)
        if filled is not None:
            return filled, forced_dirs, target
    return None


def _find_candidates(
    h: int,
    w: int,
    block: set[Coord],
    block_bbox: tuple[int, int, int, int],
    size: int,
    direction: str,
    allowed_distances: set[int],
    input_occupied: set[Coord],
    output_occupied: set[Coord],
    previous_finals: set[Coord],
    *,
    want_blocked: bool,
    must_cover: Coord | None = None,
) -> list[dict]:
    candidates: list[dict] = []
    variants = list(SHAPES_BY_SIZE[size])
    random.shuffle(variants)
    for variant in variants:
        vr1 = max(r for r, _ in variant)
        vc1 = max(c for _, c in variant)
        rows = list(range(h - vr1))
        cols = list(range(w - vc1))
        random.shuffle(rows)
        random.shuffle(cols)
        for r0 in rows:
            for c0 in cols:
                sprite = {(r + r0, c + c0) for r, c in variant}
                if sprite & input_occupied:
                    continue
                if sprite & output_occupied:
                    continue
                if _component_min_manhattan(sprite, block) not in allowed_distances:
                    continue
                initial_occ = input_occupied | sprite
                if _unique_sightline(sprite, block, initial_occ) != direction:
                    continue
                prior_occ = previous_finals | sprite
                dynamic_dir = _unique_sightline(sprite, block, prior_occ)
                if want_blocked:
                    if dynamic_dir is not None:
                        continue
                    final = set(sprite)
                    has_order_effect = False
                else:
                    if dynamic_dir != direction:
                        continue
                    final = _final_coords(sprite, block_bbox, direction, "normal")
                    post_final = _final_coords(sprite, block_bbox, direction, "post_slide")
                    has_order_effect = final != post_final
                    if (
                        not _in_bounds((h, w), final)
                        or (final & output_occupied)
                        or (final & sprite)
                    ):
                        continue
                    if must_cover is not None and must_cover not in final:
                        continue
                candidates.append({"sprite": sprite, "final": final, "has_order_effect": has_order_effect})
                if len(candidates) >= 80:
                    return candidates
    return candidates


def _validate_semantics(grid: np.ndarray, expect_blocked: bool) -> bool:
    try:
        bg, scene = _parse_scene(grid)
    except Exception:
        return False
    occupied = {tuple(map(int, rc)) for rc in np.argwhere(grid != bg)}
    sizes = [item["size"] for item in scene]
    if len(set(sizes)) < 2 or all(size % 4 == 0 for size in sizes):
        return False
    if sum(size % 4 != 0 for size in sizes) < 1:
        return False
    if np.count_nonzero(grid == 0) < grid.size * 0.65:
        return False
    for item in scene:
        if item["size"] < 3 or item["size"] > 6:
            return False
        if not _is_rot_asymmetric(_normalize(item["sprite"])):
            return False
        if _unique_sightline(item["sprite"], item["block"], occupied) is None:
            return False
    out, meta = _solve_variant(grid, return_meta=True)
    moved = [item for item in meta if item["status"] == "moved"]
    if not moved:
        return False
    if expect_blocked and not any(item["status"] == "blocked" for item in meta):
        return False
    alt_no_rot = _solve_variant(grid, rotate_mode="none")
    alt_post = _solve_variant(grid, rotate_mode="post_slide")
    if np.array_equal(out, alt_no_rot):
        return False
    if np.array_equal(out, alt_post):
        return False
    if expect_blocked:
        alt_static = _solve_variant(grid, visibility="static")
        if np.array_equal(out, alt_static):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 18)
    w = random.randint(14, 18)
    count = random.randint(3, 4)
    colors = random.sample(PALETTE, count)
    sizes = [random.choice([3, 4, 5, 6]) for _ in range(count)]
    if len(set(sizes)) < 2 or all(size % 4 == 0 for size in sizes):
        return None
    order = random.sample(colors, len(colors))
    expect_blocked = count >= 3 and random.random() < 0.6

    forced_dirs: dict[int, str] = {}
    blocker_target: Coord | None = None
    if expect_blocked:
        layout = _blocked_layout(h, w, colors, order)
        if layout is None:
            return None
        block_positions, forced_dirs, blocker_target = layout
    else:
        block_positions = _random_blocks(h, w, colors, min_sep=5)
        if block_positions is None:
            return None

    block_sets = {
        color: {
            (r0, c0),
            (r0, c0 + 1),
            (r0 + 1, c0),
            (r0 + 1, c0 + 1),
        }
        for color, (r0, c0) in block_positions.items()
    }
    block_boxes = {color: _bbox(coords) for color, coords in block_sets.items()}

    input_occ = set().union(*block_sets.values())
    output_occ = set(input_occ)
    previous_finals = set(input_occ)
    sprite_positions: dict[int, set[Coord]] = {}

    size_by_color = dict(zip(colors, sizes, strict=True))
    placed_distances: dict[int, int] = {}
    has_order_effect = False

    for idx, color in enumerate(order):
        want_blocked = expect_blocked and idx == 1
        if expect_blocked and idx == 0:
            allowed_distances = {2, 3, 4}
        elif expect_blocked and idx == 1:
            prev_dist = placed_distances[order[0]]
            allowed_distances = {dist for dist in range(prev_dist + 1, 7)}
            if not allowed_distances:
                return None
        else:
            allowed_distances = {2, 3, 4, 5, 6}
        direction_choices = [forced_dirs[color]] if color in forced_dirs else random.sample(
            ["right", "left", "down", "up"], 4
        )
        all_candidates: list[dict] = []
        for direction in direction_choices:
            candidates = _find_candidates(
                h,
                w,
                block_sets[color],
                block_boxes[color],
                size_by_color[color],
                direction,
                allowed_distances,
                input_occ,
                output_occ,
                previous_finals,
                want_blocked=want_blocked,
                must_cover=blocker_target if expect_blocked and idx == 0 else None,
            )
            for cand in candidates:
                cand = dict(cand)
                cand["direction"] = direction
                all_candidates.append(cand)
        if not all_candidates:
            return None
        chosen_pool = all_candidates
        if (
            not has_order_effect
            and not want_blocked
            and size_by_color[color] % 4 != 0
        ):
            effect_pool = [cand for cand in all_candidates if cand["has_order_effect"]]
            if effect_pool:
                chosen_pool = effect_pool
        chosen = random.choice(chosen_pool)
        sprite_positions[color] = chosen["sprite"]
        placed_distances[color] = _component_min_manhattan(chosen["sprite"], block_sets[color])
        input_occ |= chosen["sprite"]
        output_occ |= chosen["final"]
        previous_finals |= chosen["final"]
        has_order_effect = has_order_effect or chosen["has_order_effect"]

    if not has_order_effect:
        return None

    grid = _paint_scene((h, w), block_positions, sprite_positions)
    if not _validate_semantics(grid, expect_blocked):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
