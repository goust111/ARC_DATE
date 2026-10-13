import numpy as np


BLACK = 0
BLUE = 1
YELLOW = 4
ORANGE = 7
TEAL = 8

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

HEIGHT_RANGE = (9, 13)
PANEL_WIDTH_RANGE = (9, 13)
YELLOW_COUNT_RANGE = (8, 15)
GEN_ATTEMPTS = 600


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _divider_col(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cols = [c for c in range(w) if np.all(g[:, c] == BLUE)]
    assert len(cols) == 1, "expected exactly one full-height blue divider column"
    div = cols[0]
    assert div == w // 2 and div == w - div - 1, "divider must split two equal-width panels"
    return div


def _find_dominoes(panel: np.ndarray) -> list[dict]:
    seen = np.zeros(panel.shape, dtype=bool)
    dominoes = []
    h, w = panel.shape
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(panel[r, c]) != ORANGE:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(panel.shape, nr, nc) and not seen[nr, nc] and int(panel[nr, nc]) == ORANGE:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            assert len(coords) == 2, "every orange object must be a domino"
            coords = sorted(coords)
            (r0, c0), (r1, c1) = coords
            if r0 == r1 and c1 == c0 + 1:
                orientation = "h"
                ends = ((r0, c0), (r0, c1))
            elif c0 == c1 and r1 == r0 + 1:
                orientation = "v"
                ends = ((r0, c0), (r1, c0))
            else:
                raise AssertionError("orange domino must be horizontal or vertical")
            dominoes.append({"cells": tuple(coords), "orientation": orientation, "ends": ends})
    assert 1 <= len(dominoes) <= 2, "each panel must contain 1 or 2 orange dominoes"
    return dominoes


def _ray_path(start: tuple[int, int], direction: tuple[int, int], obstacle_mask: np.ndarray, orange_mask: np.ndarray) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    path = []
    while in_bounds(obstacle_mask.shape, r, c) and not obstacle_mask[r, c] and not orange_mask[r, c]:
        path.append((r, c))
        r += dr
        c += dc
    return path


def _trace_ray(start: tuple[int, int], direction: tuple[int, int], obstacle_mask: np.ndarray, orange_mask: np.ndarray) -> dict:
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    path = []
    while True:
        if not in_bounds(obstacle_mask.shape, r, c):
            return {"painted": tuple(path), "stop_reason": "boundary", "stop_cell": None}
        if orange_mask[r, c]:
            return {"painted": tuple(path), "stop_reason": "orange", "stop_cell": (r, c)}
        if obstacle_mask[r, c]:
            return {"painted": tuple(path), "stop_reason": "obstacle", "stop_cell": (r, c)}
        path.append((r, c))
        r += dr
        c += dc


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    div = _divider_col(g)
    left = g[:, :div]
    right = g[:, div + 1 :]
    assert left.shape == right.shape

    for panel in (left, right):
        allowed = {BLACK, YELLOW, ORANGE}
        assert set(map(int, np.unique(panel))) <= allowed, "panels may only contain black, yellow, and orange"

    left_y = left == YELLOW
    right_y = right == YELLOW
    assert not np.any(left_y & right_y), "yellow masks must be complementary after alignment"

    dominoes = []
    for owner, panel in enumerate((left, right)):
        panel_dominoes = _find_dominoes(panel)
        orange_cells = {cell for d in panel_dominoes for cell in d["cells"]}
        for r, c in np.argwhere(panel == YELLOW):
            for nr, nc in neighbors4(int(r), int(c)):
                if (nr, nc) in orange_cells:
                    raise AssertionError("yellow may not be orthogonally adjacent to a same-panel orange domino")
        for domino in panel_dominoes:
            dominoes.append(
                {
                    "owner": owner,
                    "cells": tuple(domino["cells"]),
                    "orientation": domino["orientation"],
                    "ends": domino["ends"],
                }
            )

    orange_mask = np.zeros(left.shape, dtype=bool)
    for domino in dominoes:
        for cell in domino["cells"]:
            assert not orange_mask[cell], "merged dominoes may not overlap"
            orange_mask[cell] = True

    return {
        "shape": left.shape,
        "left_y": left_y,
        "right_y": right_y,
        "dominoes": dominoes,
        "orange_mask": orange_mask,
    }


def _solve_scene(scene: dict, obstacle_mode: str = "union") -> np.ndarray:
    h, w = scene["shape"]
    out = np.full((h, w), BLACK, dtype=int)
    union_obs = scene["left_y"] | scene["right_y"]
    out[union_obs] = TEAL
    out[scene["orange_mask"]] = ORANGE

    ray_cells: set[tuple[int, int]] = set()
    for domino in scene["dominoes"]:
        if obstacle_mode == "union":
            obstacle_mask = union_obs
        elif obstacle_mode == "owner":
            obstacle_mask = scene["left_y"] if domino["owner"] == 0 else scene["right_y"]
        else:
            raise ValueError(f"unknown obstacle mode {obstacle_mode}")

        if domino["orientation"] == "h":
            starts_dirs = ((domino["ends"][0], (0, -1)), (domino["ends"][1], (0, 1)))
        else:
            starts_dirs = ((domino["ends"][0], (-1, 0)), (domino["ends"][1], (1, 0)))
        for start, direction in starts_dirs:
            ray_cells.update(_ray_path(start, direction, obstacle_mask, scene["orange_mask"]))

    for r, c in ray_cells:
        if out[r, c] == BLACK:
            out[r, c] = ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the two yellow side panels into one teal obstacle map, then fire each orange domino along its axis until the merged obstacles stop it.

    Input structure:
    - A blue full-height divider column splits the grid into two equal black-based panels.
    - Each panel contains scattered yellow mask pixels and 1-2 orange two-cell domino markers.
    - Yellow pixels from the two panels never occupy the same aligned output position, and orange dominoes stay distinct after alignment.

    Rule:
    - Remove the blue divider and align the left and right panels into one output canvas.
    - Recolor every aligned yellow pixel from either panel to teal, keep shared black cells black, and keep all orange domino markers in place.
    - For each horizontal domino, cast one orange ray leftward from its left end and one rightward from its right end; for each vertical domino, cast one upward and one downward from its two ends.
    - A ray colors only black cells and stops just before the first teal obstacle, another original orange domino cell, or the grid boundary.

    Key insight:
    - The obstacle field is only complete after the two yellow masks are overlaid, so some ray lengths are determined by blockers contributed by the opposite panel.
    """
    scene = _extract_scene(grid)
    return _solve_scene(scene, obstacle_mode="union")


def _empty_cells_with_allowed_sides(h: int, w: int, dominoes: list[dict]) -> dict[tuple[int, int], tuple[int, ...]]:
    all_orange = {cell for domino in dominoes for cell in domino["cells"]}
    side_forbidden = []
    for side in (0, 1):
        forbidden = set()
        for domino in dominoes:
            if domino["owner"] != side:
                continue
            for r, c in domino["cells"]:
                forbidden.add((r, c))
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w:
                        forbidden.add((nr, nc))
        side_forbidden.append(forbidden)

    allowed = {}
    for r in range(h):
        for c in range(w):
            if (r, c) in all_orange:
                continue
            sides = tuple(side for side in (0, 1) if (r, c) not in side_forbidden[side])
            if sides:
                allowed[(r, c)] = sides
    return allowed


def _place_dominoes(rng: np.random.Generator, h: int, w: int, owners: list[int], orientations: list[str]) -> list[dict] | None:
    dominoes: list[dict] = []
    occupied: set[tuple[int, int]] = set()

    for owner, orientation in zip(owners, orientations):
        placed = False
        for _ in range(200):
            if orientation == "h":
                r = int(rng.integers(1, h - 1))
                c = int(rng.integers(1, w - 2))
                cells = ((r, c), (r, c + 1))
                ends = cells
            else:
                r = int(rng.integers(1, h - 2))
                c = int(rng.integers(1, w - 1))
                cells = ((r, c), (r + 1, c))
                ends = cells

            if any(cell in occupied for cell in cells):
                continue
            if any(max(abs(r0 - r1), abs(c0 - c1)) <= 1 for r0, c0 in cells for r1, c1 in occupied):
                continue

            dominoes.append({"owner": owner, "orientation": orientation, "cells": cells, "ends": ends})
            occupied.update(cells)
            placed = True
            break
        if not placed:
            return None
    return dominoes


def _ray_infos(h: int, w: int, dominoes: list[dict]) -> list[dict] | None:
    orange_mask = np.zeros((h, w), dtype=bool)
    for domino in dominoes:
        for cell in domino["cells"]:
            orange_mask[cell] = True

    infos = []
    for dom_idx, domino in enumerate(dominoes):
        if domino["orientation"] == "h":
            starts_dirs = ((domino["ends"][0], (0, -1)), (domino["ends"][1], (0, 1)))
        else:
            starts_dirs = ((domino["ends"][0], (-1, 0)), (domino["ends"][1], (1, 0)))
        for ray_idx, (start, direction) in enumerate(starts_dirs):
            cells = _ray_path(start, direction, np.zeros((h, w), dtype=bool), orange_mask)
            if not cells:
                return None
            infos.append(
                {
                    "dom_idx": dom_idx,
                    "owner": domino["owner"],
                    "ray_idx": ray_idx,
                    "start": start,
                    "direction": direction,
                    "cells": tuple(cells),
                }
            )
    return infos


def _choose_plans(rng: np.random.Generator, ray_infos: list[dict], allowed_sides: dict[tuple[int, int], tuple[int, ...]]) -> dict | None:
    special_candidates = []
    for idx, info in enumerate(ray_infos):
        opp = 1 - info["owner"]
        for cell_index in range(1, min(len(info["cells"]), 5)):
            if opp in allowed_sides.get(info["cells"][cell_index], ()):
                special_candidates.append((idx, cell_index))
    if not special_candidates:
        return None

    special_ray_idx, special_cell_index = special_candidates[int(rng.integers(len(special_candidates)))]
    plans = {}
    by_dom = {}
    for idx, info in enumerate(ray_infos):
        by_dom.setdefault(info["dom_idx"], []).append(idx)

    for dom_idx, ray_ids in by_dom.items():
        candidates = []
        for ray_id in ray_ids:
            info = ray_infos[ray_id]
            options = []
            for cell_index, cell in enumerate(info["cells"]):
                if cell_index == 0 and len(info["cells"]) > 1:
                    continue
                for side in allowed_sides.get(cell, ()):
                    options.append((cell_index, side))
            if options:
                candidates.append(ray_id)
        if not candidates:
            return None
        primary = special_ray_idx if special_ray_idx in ray_ids else candidates[int(rng.integers(len(candidates)))]
        for ray_id in ray_ids:
            info = ray_infos[ray_id]
            options = []
            for cell_index, cell in enumerate(info["cells"]):
                if cell_index == 0 and len(info["cells"]) > 1:
                    continue
                for side in allowed_sides.get(cell, ()):
                    options.append((cell_index, side))
            if ray_id == primary:
                mode = "obstacle"
            else:
                mode = "obstacle" if options and rng.random() < 0.6 else "boundary"
            plans[ray_id] = {"mode": mode, "options": options}

    plans[special_ray_idx]["mode"] = "obstacle"
    plans[special_ray_idx]["forced_choice"] = (special_cell_index, 1 - ray_infos[special_ray_idx]["owner"])

    if all(plan["mode"] == "obstacle" for plan in plans.values()):
        non_special = [idx for idx in plans if idx != special_ray_idx]
        flip = non_special[int(rng.integers(len(non_special)))]
        plans[flip]["mode"] = "boundary"

    blockers = {}
    protected_prefixes = {}
    for idx, info in enumerate(ray_infos):
        plan = plans[idx]
        if plan["mode"] == "boundary":
            protected_prefixes[idx] = set(info["cells"])
            continue
        if "forced_choice" in plan:
            choice = plan["forced_choice"]
            if choice not in plan["options"]:
                return None
        else:
            options = plan["options"]
            if not options:
                return None
            own_side = info["owner"]
            own_options = [opt for opt in options if opt[1] == own_side]
            opp_options = [opt for opt in options if opt[1] != own_side]
            pool = own_options if own_options and rng.random() < 0.6 else (opp_options if opp_options else options)
            choice = pool[int(rng.integers(len(pool)))]
        cell_index, side = choice
        block_cell = info["cells"][cell_index]
        if block_cell in blockers:
            return None
        blockers[block_cell] = side
        protected_prefixes[idx] = set(info["cells"][:cell_index])
        plans[idx]["choice"] = choice

    for idx, info in enumerate(ray_infos):
        block_cell = None
        if plans[idx]["mode"] == "obstacle":
            cell_index, _side = plans[idx]["choice"]
            block_cell = info["cells"][cell_index]
        for other_idx, other_prefix in protected_prefixes.items():
            if other_idx == idx:
                continue
            if block_cell is not None and block_cell in other_prefix:
                return None

    return {"plans": plans, "special_ray_idx": special_ray_idx, "blockers": blockers, "protected": protected_prefixes}


def _fill_obstacles(
    rng: np.random.Generator,
    h: int,
    w: int,
    dominoes: list[dict],
    blockers: dict[tuple[int, int], int],
    protected: dict[int, set[tuple[int, int]]],
    allowed_sides: dict[tuple[int, int], tuple[int, ...]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    targets = [int(rng.integers(YELLOW_COUNT_RANGE[0], YELLOW_COUNT_RANGE[1] + 1)) for _ in range(2)]
    left_cells = {cell for cell, side in blockers.items() if side == 0}
    right_cells = {cell for cell, side in blockers.items() if side == 1}
    if len(left_cells) > targets[0] or len(right_cells) > targets[1]:
        return None

    forbidden = set(blockers)
    for prefix in protected.values():
        forbidden.update(prefix)

    candidates = [(r, c) for r in range(h) for c in range(w) if (r, c) not in forbidden and (r, c) in allowed_sides]
    rng.shuffle(candidates)

    remaining = [targets[0] - len(left_cells), targets[1] - len(right_cells)]
    for cell in candidates:
        allowed = [side for side in allowed_sides[cell] if remaining[side] > 0]
        if not allowed:
            continue
        if len(allowed) == 2:
            side = 0 if remaining[0] > remaining[1] else 1
        else:
            side = allowed[0]
        if side == 0:
            left_cells.add(cell)
        else:
            right_cells.add(cell)
        remaining[side] -= 1
        if remaining == [0, 0]:
            break

    if remaining != [0, 0]:
        return None
    return left_cells, right_cells


def _render_input(h: int, w: int, dominoes: list[dict], left_y: set[tuple[int, int]], right_y: set[tuple[int, int]]) -> np.ndarray:
    grid = np.full((h, 2 * w + 1), BLACK, dtype=int)
    grid[:, w] = BLUE
    for r, c in left_y:
        grid[r, c] = YELLOW
    for r, c in right_y:
        grid[r, w + 1 + c] = YELLOW
    for domino in dominoes:
        for r, c in domino["cells"]:
            col = c if domino["owner"] == 0 else w + 1 + c
            grid[r, col] = ORANGE
    return grid


def _validate_generated_input(grid: np.ndarray) -> bool:
    try:
        scene = _extract_scene(grid)
    except AssertionError:
        return False

    left_count = int(np.sum(scene["left_y"]))
    right_count = int(np.sum(scene["right_y"]))
    if not (YELLOW_COUNT_RANGE[0] <= left_count <= YELLOW_COUNT_RANGE[1]):
        return False
    if not (YELLOW_COUNT_RANGE[0] <= right_count <= YELLOW_COUNT_RANGE[1]):
        return False

    overlay_only = np.full(scene["shape"], BLACK, dtype=int)
    overlay_only[scene["left_y"] | scene["right_y"]] = TEAL
    overlay_only[scene["orange_mask"]] = ORANGE

    full_out = _solve_scene(scene, obstacle_mode="union")
    owner_only_out = _solve_scene(scene, obstacle_mode="owner")
    if np.array_equal(full_out, overlay_only):
        return False
    if np.array_equal(full_out, owner_only_out):
        return False

    ray_pixels = (full_out == ORANGE) & ~scene["orange_mask"]
    if int(np.sum(ray_pixels)) < 3:
        return False

    dependent = False
    for domino in scene["dominoes"]:
        if domino["orientation"] == "h":
            rays = ((domino["ends"][0], (0, -1)), (domino["ends"][1], (0, 1)))
        else:
            rays = ((domino["ends"][0], (-1, 0)), (domino["ends"][1], (1, 0)))
        own_obs = scene["left_y"] if domino["owner"] == 0 else scene["right_y"]
        other_obs = scene["right_y"] if domino["owner"] == 0 else scene["left_y"]
        for start, direction in rays:
            union_trace = _trace_ray(start, direction, scene["left_y"] | scene["right_y"], scene["orange_mask"])
            own_trace = _trace_ray(start, direction, own_obs, scene["orange_mask"])
            if union_trace["stop_reason"] != "obstacle":
                continue
            stop = union_trace["stop_cell"]
            if stop is None or not other_obs[stop] or own_obs[stop]:
                continue
            if len(own_trace["painted"]) > len(union_trace["painted"]):
                dependent = True
                break
        if dependent:
            break
    if not dependent:
        return False

    try:
        solved = main(grid)
    except AssertionError:
        return False
    return np.array_equal(solved, full_out)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(HEIGHT_RANGE[0], HEIGHT_RANGE[1] + 1))
    w = int(rng.integers(PANEL_WIDTH_RANGE[0], PANEL_WIDTH_RANGE[1] + 1))

    left_count = int(rng.integers(1, 3))
    right_count = int(rng.integers(1, 3))
    owners = [0] * left_count + [1] * right_count
    rng.shuffle(owners)

    total = len(owners)
    orientations = ["h", "v"]
    while len(orientations) < total:
        orientations.append("h" if rng.random() < 0.5 else "v")
    rng.shuffle(orientations)

    dominoes = _place_dominoes(rng, h, w, owners, orientations)
    if dominoes is None:
        return None

    ray_infos = _ray_infos(h, w, dominoes)
    if ray_infos is None:
        return None

    allowed_sides = _empty_cells_with_allowed_sides(h, w, dominoes)
    plan_bundle = _choose_plans(rng, ray_infos, allowed_sides)
    if plan_bundle is None:
        return None

    obstacle_sets = _fill_obstacles(
        rng,
        h,
        w,
        dominoes,
        plan_bundle["blockers"],
        plan_bundle["protected"],
        allowed_sides,
    )
    if obstacle_sets is None:
        return None
    left_y, right_y = obstacle_sets

    grid = _render_input(h, w, dominoes, left_y, right_y)
    return grid if _validate_generated_input(grid) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-panel ray-overlay input")
