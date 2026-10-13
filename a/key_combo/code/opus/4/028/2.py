import random
from typing import Iterable

import numpy as np


BG = 0
RED = 2
GREEN = 3

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
SIDE_CENTER_TO_DIR = {
    (0, 1): UP,
    (2, 1): DOWN,
    (1, 0): LEFT,
    (1, 2): RIGHT,
}
DIR_TO_SIDE_CENTER = {v: k for k, v in SIDE_CENTER_TO_DIR.items()}

GEN_RETRIES = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _extra_corner(direction: tuple[int, int], style: int) -> tuple[int, int] | None:
    if style == 0:
        return None
    if direction == UP:
        return (2, 0) if style == 1 else (2, 2)
    if direction == DOWN:
        return (0, 2) if style == 1 else (0, 0)
    if direction == LEFT:
        return (0, 2) if style == 1 else (2, 2)
    return (2, 0) if style == 1 else (0, 0)


def _arrow_mask(direction: tuple[int, int], style: int) -> np.ndarray:
    mask = np.ones((3, 3), dtype=bool)
    miss_r, miss_c = DIR_TO_SIDE_CENTER[direction]
    mask[miss_r, miss_c] = False
    extra = _extra_corner(direction, style)
    if extra is not None:
        mask[extra] = False
    assert mask[1, 1]
    assert sum(not mask[pos] for pos in SIDE_CENTER_TO_DIR) == 1
    assert len(_components_mask(mask)) == 1
    return mask


def _bbox_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)}


def _arrow_cells(center: tuple[int, int], direction: tuple[int, int], style: int) -> set[tuple[int, int]]:
    r, c = center
    mask = _arrow_mask(direction, style)
    return {
        (r - 1 + rr, c - 1 + cc)
        for rr in range(3)
        for cc in range(3)
        if mask[rr, cc]
    }


def _draw_arrow(grid: np.ndarray, center: tuple[int, int], direction: tuple[int, int], style: int, color: int) -> None:
    patch = np.zeros((3, 3), dtype=int)
    patch[_arrow_mask(direction, style)] = int(color)
    r, c = center
    grid[r - 1 : r + 2, c - 1 : c + 2] = patch


def _extract_arrows(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    arrows: list[dict] = []
    for color in (RED, GREEN):
        for coords in _components_mask(g == color):
            r0, c0, r1, c1 = _bbox_coords(coords)
            if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
                raise AssertionError("arrow bbox must be 3x3")
            center = (r0 + 1, c0 + 1)
            local = np.zeros((3, 3), dtype=bool)
            rel = coords - np.array([r0, c0])
            local[rel[:, 0], rel[:, 1]] = True
            if not local[1, 1]:
                raise AssertionError("arrow center must be filled")
            missing = [pos for pos in SIDE_CENTER_TO_DIR if not local[pos]]
            if len(missing) != 1:
                raise AssertionError("arrow must have exactly one missing side-center")
            direction = SIDE_CENTER_TO_DIR[missing[0]]
            arrows.append(
                {
                    "color": int(color),
                    "coords": {(int(r), int(c)) for r, c in coords},
                    "bbox": (r0, c0, r1, c1),
                    "center": center,
                    "mask": local,
                    "direction": direction,
                }
            )
    return arrows


def _ray_cells_to_target(
    source_center: tuple[int, int],
    direction: tuple[int, int],
    target_center: tuple[int, int],
) -> list[tuple[int, int]]:
    target_bbox = _bbox_cells(target_center)
    dr, dc = direction
    rr, cc = source_center
    path: list[tuple[int, int]] = []
    while True:
        rr += dr
        cc += dc
        path.append((rr, cc))
        if (rr, cc) in target_bbox:
            return path


def _ray_cells_to_boundary(
    shape: tuple[int, int],
    source_center: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    rr, cc = source_center
    path: list[tuple[int, int]] = []
    while True:
        rr += dr
        cc += dc
        if not (0 <= rr < h and 0 <= cc < w):
            return path
        path.append((rr, cc))


def _render_expected(shape: tuple[int, int], arrow_specs: list[dict], chain_order: list[int]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)
    for spec in arrow_specs:
        color = RED if spec["index"] == chain_order[0] else GREEN
        _draw_arrow(out, spec["center"], spec["direction"], spec["style"], color)

    seed = arrow_specs[chain_order[0]]
    seed_mask = _arrow_mask(seed["direction"], seed["style"])
    seed_patch = np.zeros((3, 3), dtype=int)
    seed_patch[seed_mask] = RED

    for src_idx, dst_idx in zip(chain_order, chain_order[1:]):
        src = arrow_specs[src_idx]
        dst = arrow_specs[dst_idx]
        for r, c in _ray_cells_to_target(src["center"], src["direction"], dst["center"]):
            out[r, c] = RED
        for r, c in _arrow_cells(dst["center"], dst["direction"], dst["style"]):
            if out[r, c] != RED:
                out[r, c] = BG
        rr, cc = dst["center"]
        view = out[rr - 1 : rr + 2, cc - 1 : cc + 2]
        view[seed_mask] = RED

    last = arrow_specs[chain_order[-1]]
    for r, c in _ray_cells_to_boundary(shape, last["center"], last["direction"]):
        out[r, c] = RED

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Activate a chain of arrows by straight rays, then stamp the seed arrow shape onto each activated target.

    Input structure:
    - The background is black.
    - Foreground consists of 3x3-bbox, 4-connected arrow sprites in green, plus one red seed arrow.
    - Each arrow keeps its center cell and omits exactly one side-center cell; that missing side-center encodes up/down/left/right.

    Rule:
    - Find the unique red seed arrow and repeatedly cast a 1-cell-wide red ray from the current arrow center in that arrow's own direction.
    - A ray stops when it leaves the grid or first enters another arrow's bbox; a newly hit green arrow becomes activated and immediately emits its own ray.
    - Every activated non-seed arrow is overlaid at its own center with the seed arrow's original 3x3 mask, while all ray pixels remain red and untouched green arrows stay unchanged.

    Key insight:
    - Two independent signals must be tracked at once: each local arrow mask determines where propagation goes next, but the global seed mask determines what every activated target finally looks like.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    arrows = _extract_arrows(g)
    red_arrows = [i for i, arrow in enumerate(arrows) if arrow["color"] == RED]
    assert len(red_arrows) == 1, "expected exactly one red seed arrow"

    bbox_owner: dict[tuple[int, int], int] = {}
    for idx, arrow in enumerate(arrows):
        r0, c0, r1, c1 = arrow["bbox"]
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                bbox_owner[(rr, cc)] = idx

    seed_idx = red_arrows[0]
    seed_mask = arrows[seed_idx]["mask"]
    out = g.copy()
    activated = {seed_idx}
    current = seed_idx

    while True:
        dr, dc = arrows[current]["direction"]
        rr, cc = arrows[current]["center"]
        hit: int | None = None
        while True:
            rr += dr
            cc += dc
            if not (0 <= rr < h and 0 <= cc < w):
                hit = None
                break
            out[rr, cc] = RED
            owner = bbox_owner.get((rr, cc))
            if owner is not None and owner != current:
                hit = owner
                break
        if hit is None:
            break
        if hit in activated:
            break
        target = arrows[hit]
        for r, c in target["coords"]:
            if out[r, c] != RED:
                out[r, c] = BG
        tr, tc = target["center"]
        view = out[tr - 1 : tr + 2, tc - 1 : tc + 2]
        view[seed_mask] = RED
        activated.add(hit)
        current = hit

    return out


def _lane_values(limit: int) -> list[int]:
    return [x for x in range(2, limit - 1, 4)]


def _candidate_chain_step(
    shape: tuple[int, int],
    current: tuple[int, int],
    used_centers: set[tuple[int, int]],
    occupied_bboxes: set[tuple[int, int]],
    used_ray_cells: set[tuple[int, int]],
    row_lanes: list[int],
    col_lanes: list[int],
) -> list[dict]:
    h, w = shape
    del h, w
    cur_bbox = _bbox_cells(current)
    candidates: list[dict] = []
    for direction in DIRS:
        dr, dc = direction
        if dr != 0:
            for nr in row_lanes:
                if (nr - current[0]) * dr <= 0 or abs(nr - current[0]) < 4:
                    continue
                center = (nr, current[1])
                if center in used_centers:
                    continue
                bbox_cells = _bbox_cells(center)
                if bbox_cells & occupied_bboxes:
                    continue
                if bbox_cells & used_ray_cells:
                    continue
                path = _ray_cells_to_target(current, direction, center)
                if set(path) & used_ray_cells:
                    continue
                if set(path) & (occupied_bboxes - cur_bbox):
                    continue
                candidates.append({"center": center, "direction": direction, "path": path})
        else:
            for nc in col_lanes:
                if (nc - current[1]) * dc <= 0 or abs(nc - current[1]) < 4:
                    continue
                center = (current[0], nc)
                if center in used_centers:
                    continue
                bbox_cells = _bbox_cells(center)
                if bbox_cells & occupied_bboxes:
                    continue
                if bbox_cells & used_ray_cells:
                    continue
                path = _ray_cells_to_target(current, direction, center)
                if set(path) & used_ray_cells:
                    continue
                if set(path) & (occupied_bboxes - cur_bbox):
                    continue
                candidates.append({"center": center, "direction": direction, "path": path})
    return candidates


def _candidate_final_rays(
    shape: tuple[int, int],
    current: tuple[int, int],
    occupied_bboxes: set[tuple[int, int]],
    used_ray_cells: set[tuple[int, int]],
) -> list[dict]:
    cur_bbox = _bbox_cells(current)
    candidates: list[dict] = []
    for direction in DIRS:
        path = _ray_cells_to_boundary(shape, current, direction)
        if len(path) < 3:
            continue
        if set(path) & used_ray_cells:
            continue
        if set(path) & (occupied_bboxes - cur_bbox):
            continue
        candidates.append({"direction": direction, "path": path})
    return candidates


def _build_attempt() -> tuple[np.ndarray, np.ndarray, dict] | None:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    row_lanes = _lane_values(h)
    col_lanes = _lane_values(w)
    if len(row_lanes) < 4 or len(col_lanes) < 4:
        return None

    chain_count = random.choice([4, 5])
    total_arrows = random.randint(max(5, chain_count + 1), min(7, chain_count + 2))

    start = (random.choice(row_lanes), random.choice(col_lanes))
    centers = [start]
    used_centers = {start}
    occupied_bboxes = set(_bbox_cells(start))
    used_ray_cells: set[tuple[int, int]] = set()
    chain_dirs: list[tuple[int, int]] = []
    chain_paths: list[list[tuple[int, int]]] = []

    for _ in range(chain_count - 1):
        candidates = _candidate_chain_step((h, w), centers[-1], used_centers, occupied_bboxes, used_ray_cells, row_lanes, col_lanes)
        if chain_dirs:
            turning = [cand for cand in candidates if cand["direction"] != chain_dirs[-1]]
            if turning:
                candidates = turning
        if not candidates:
            return None
        choice = random.choice(candidates)
        centers.append(choice["center"])
        chain_dirs.append(choice["direction"])
        chain_paths.append(choice["path"])
        used_centers.add(choice["center"])
        occupied_bboxes |= _bbox_cells(choice["center"])
        used_ray_cells |= set(choice["path"])

    final_options = _candidate_final_rays((h, w), centers[-1], occupied_bboxes, used_ray_cells)
    if not final_options:
        return None
    final = random.choice(final_options)
    chain_dirs.append(final["direction"])
    used_ray_cells |= set(final["path"])

    seed_style = random.randint(0, 2)
    chain_styles = [seed_style] + [random.randint(0, 2) for _ in range(chain_count - 1)]
    if all(style == seed_style for style in chain_styles[1:]):
        chain_styles[random.randrange(1, chain_count)] = (seed_style + 1 + random.randrange(2)) % 3

    arrow_specs: list[dict] = []
    for idx, center in enumerate(centers):
        arrow_specs.append(
            {
                "index": idx,
                "center": center,
                "direction": chain_dirs[idx],
                "style": chain_styles[idx],
                "chain": True,
            }
        )

    distractor_needed = total_arrows - chain_count
    available_centers = [
        (r, c)
        for r in row_lanes
        for c in col_lanes
        if (r, c) not in used_centers
        and not (_bbox_cells((r, c)) & occupied_bboxes)
        and not (_bbox_cells((r, c)) & used_ray_cells)
    ]
    random.shuffle(available_centers)
    if len(available_centers) < distractor_needed:
        return None

    for _ in range(distractor_needed):
        center = available_centers.pop()
        used_centers.add(center)
        occupied_bboxes |= _bbox_cells(center)
        arrow_specs.append(
            {
                "index": len(arrow_specs),
                "center": center,
                "direction": random.choice(DIRS),
                "style": random.randint(0, 2),
                "chain": False,
            }
        )

    inp = np.zeros((h, w), dtype=int)
    for spec in arrow_specs:
        color = RED if spec["chain"] and spec["index"] == 0 else GREEN
        _draw_arrow(inp, spec["center"], spec["direction"], spec["style"], color)

    expected = _render_expected((h, w), arrow_specs, list(range(chain_count)))
    meta = {
        "chain_count": chain_count,
        "total_arrows": total_arrows,
        "seed_style": seed_style,
        "chain_styles": chain_styles,
        "arrow_specs": arrow_specs,
        "final_path_len": len(final["path"]),
    }
    return inp, expected, meta


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    arrows = _extract_arrows(inp)
    if len(arrows) != meta["total_arrows"]:
        return False
    if sum(arrow["color"] == RED for arrow in arrows) != 1:
        return False
    if meta["chain_count"] < 4:
        return False
    if meta["final_path_len"] < 3:
        return False
    if not any(style != meta["seed_style"] for style in meta["chain_styles"][1:]):
        return False
    if not np.any((inp == GREEN) & (expected == RED)):
        return False
    if not np.any((inp == GREEN) & (expected == GREEN)):
        return False
    activated_green_count = sum(
        spec["chain"] and spec["index"] > 0
        for spec in meta["arrow_specs"]
    )
    if activated_green_count < 3:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_attempt()
        if built is None:
            continue
        inp, expected, meta = built
        if not np.array_equal(main(inp), expected):
            continue
        if not _semantic_checks(inp, expected, meta):
            continue
        return inp
    raise RuntimeError("failed to generate a valid arrow-chain input")
