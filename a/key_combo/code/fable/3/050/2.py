from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
DIRS: dict[str, Coord] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_ORDER = ("up", "down", "left", "right")


@dataclass(frozen=True)
class MovingObject:
    coords: tuple[Coord, ...]
    color: int
    direction: str
    depth: int
    container_bbox: BBox


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[dict] = []
    for r, c in np.argwhere(grid != BG):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        color = int(grid[r, c])
        q: deque[Coord] = deque([(r, c)])
        seen[r, c] = True
        coords: list[Coord] = []
        while q:
            rr, cc = q.popleft()
            coords.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if (
                    0 <= nr < h
                    and 0 <= nc < w
                    and not seen[nr, nc]
                    and int(grid[nr, nc]) == color
                ):
                    seen[nr, nc] = True
                    q.append((nr, nc))
        arr = np.array(coords, dtype=int)
        comps.append({"color": color, "coords": arr, "bbox": _bbox(arr), "area": len(coords)})
    return comps


def _required_u_mask(h: int, w: int, direction: str) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if direction == "up":
        mask[:, 0] = True
        mask[:, w - 1] = True
        mask[h - 1, :] = True
    elif direction == "down":
        mask[:, 0] = True
        mask[:, w - 1] = True
        mask[0, :] = True
    elif direction == "left":
        mask[0, :] = True
        mask[h - 1, :] = True
        mask[:, w - 1] = True
    elif direction == "right":
        mask[0, :] = True
        mask[h - 1, :] = True
        mask[:, 0] = True
    else:
        raise ValueError(direction)
    return mask


def _u_direction(component: dict) -> str | None:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    local = np.zeros((h, w), dtype=bool)
    coords = component["coords"]
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    for direction in DIR_ORDER:
        if np.array_equal(local, _required_u_mask(h, w, direction)):
            return direction
    return None


def _strictly_inside(coords: np.ndarray, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(np.all((r0 < rows) & (rows < r1) & (c0 < cols) & (cols < c1)))


def _front_depth(coords: np.ndarray, box: BBox, direction: str) -> int:
    r0, c0, r1, c1 = box
    rows = coords[:, 0]
    cols = coords[:, 1]
    if direction == "up":
        depth = int(rows.min() - r0)
    elif direction == "down":
        depth = int(r1 - rows.max())
    elif direction == "left":
        depth = int(cols.min() - c0)
    elif direction == "right":
        depth = int(c1 - cols.max())
    else:
        raise ValueError(direction)
    assert depth > 0, "contained object must sit behind the open rim"
    return depth


def _find_moving_objects(grid: np.ndarray) -> list[MovingObject]:
    comps = _components(grid)
    containers: list[tuple[int, dict, str]] = []
    for i, comp in enumerate(comps):
        direction = _u_direction(comp)
        if direction is not None:
            containers.append((i, comp, direction))

    records: list[MovingObject] = []
    claimed: set[int] = set()
    for ci, container, direction in containers:
        candidates = []
        for oi, obj in enumerate(comps):
            if oi == ci or int(obj["color"]) == int(container["color"]):
                continue
            if _strictly_inside(obj["coords"], container["bbox"]):
                candidates.append(oi)
        assert len(candidates) <= 1, "each U container may contain only one object"
        if not candidates:
            continue
        oi = candidates[0]
        assert oi not in claimed, "a contained object must belong to one container"
        claimed.add(oi)
        obj = comps[oi]
        depth = _front_depth(obj["coords"], container["bbox"], direction)
        coords = tuple((int(r), int(c)) for r, c in obj["coords"])
        records.append(
            MovingObject(
                coords=coords,
                color=int(obj["color"]),
                direction=direction,
                depth=depth,
                container_bbox=container["bbox"],
            )
        )

    assert records, "expected at least one filled U container"
    depths = [rec.depth for rec in records]
    assert len(set(depths)) == len(depths), "contained object depths must be unique"
    return records


def _shifted(coords: np.ndarray, direction: str, amount: int) -> np.ndarray:
    dr, dc = DIRS[direction]
    return coords + np.array([dr * amount, dc * amount], dtype=int)


def _legal_position(coords: np.ndarray, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    if np.any(coords[:, 0] < 0) or np.any(coords[:, 0] >= h):
        return False
    if np.any(coords[:, 1] < 0) or np.any(coords[:, 1] >= w):
        return False
    return not bool(np.any(occupied[coords[:, 0], coords[:, 1]]))


def _solve(grid: np.ndarray, *, trace: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"
    assert np.any(g == BG), "expected black background"

    records = _find_moving_objects(g)
    out = g.copy()
    for rec in records:
        rr, cc = np.array(rec.coords, dtype=int).T
        out[rr, cc] = BG

    occupied = out != BG
    owner = np.zeros(g.shape, dtype=int)
    owner[occupied] = -1
    details = []

    for settle_id, rec in enumerate(sorted(records, key=lambda r: -r.depth), start=1):
        coords = np.array(rec.coords, dtype=int)
        planned = 2 * int(rec.depth)
        actual = 0
        reason = "range"
        blocker_owner = 0
        for step in range(1, planned + 1):
            nxt = _shifted(coords, rec.direction, step)
            h, w = g.shape
            if (
                np.any(nxt[:, 0] < 0)
                or np.any(nxt[:, 0] >= h)
                or np.any(nxt[:, 1] < 0)
                or np.any(nxt[:, 1] >= w)
            ):
                reason = "edge"
                break
            hit = occupied[nxt[:, 0], nxt[:, 1]]
            if bool(np.any(hit)):
                owners = owner[nxt[:, 0], nxt[:, 1]][hit]
                settled = owners[owners > 0]
                if settled.size:
                    reason = "settled"
                    blocker_owner = int(settled[0])
                else:
                    reason = "static"
                    blocker_owner = -1
                break
            actual = step

        final = _shifted(coords, rec.direction, actual)
        assert _legal_position(final, occupied), "settled object must occupy a legal final position"
        out[final[:, 0], final[:, 1]] = int(rec.color)
        occupied[final[:, 0], final[:, 1]] = True
        owner[final[:, 0], final[:, 1]] = int(settle_id)
        if trace:
            details.append(
                {
                    "record": rec,
                    "planned": planned,
                    "actual": actual,
                    "reason": reason,
                    "blocker_owner": blocker_owner,
                    "final": tuple((int(r), int(c)) for r, c in final),
                }
            )

    return (out, details) if trace else out


def main(grid: np.ndarray) -> np.ndarray:
    """Eject each object from its U-shaped container by the mirror distance set by its depth.

    Input structure:
    - A black-background grid contains several monochrome U/C containers, each a
      one-cell-thick hollow rectangle missing exactly one side.
    - Each active container encloses exactly one differently colored monochrome
      object strictly inside its rectangular cavity; outside the containers are
      static colored blockers or distractors.

    Rule:
    - Keep all containers and outside objects fixed, and remove the contained
      objects from their original cells before settling them.
    - For each contained object, let s be the distance from its front edge to
      the container's open rim. Move the whole object 2s cells in the opening
      direction, unless an intermediate footprint, landing footprint, or the
      grid edge blocks it; in that case stop at the farthest legal earlier
      footprint.
    - Settle objects from larger planned distance to smaller planned distance,
      so earlier landed objects block later ones.

    Key insight:
    - The open side gives the direction, and the object's depth gives a finite
      mirror-range spring throw; collision is only a cap, not the reason every
      object stops.
    """
    return _solve(grid)


# -------------------------- generator --------------------------

GEN_ATTEMPTS = 500
PLACE_ATTEMPTS = 250
COLOR_POOL = tuple(range(1, 10))


def _container_cells(direction: str, r0: int, c0: int, h: int, w: int) -> list[Coord]:
    mask = _required_u_mask(h, w, direction)
    return [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)]


def _paint(grid: np.ndarray, cells: list[Coord], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _can_place_bundle(grid: np.ndarray, colored: list[tuple[int, int, int]], pad: int = 1) -> bool:
    h, w = grid.shape
    cells = {(int(r), int(c)) for r, c, _color in colored}
    if len(cells) != len(colored):
        return False
    for r, c, _color in colored:
        r = int(r)
        c = int(c)
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != BG:
            return False
    if pad <= 0:
        return True
    for r, c in cells:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells and int(grid[rr, cc]) != BG:
                    return False
    return True


def _same_color_touch(grid: np.ndarray, cell: Coord, color: int) -> bool:
    h, w = grid.shape
    r, c = cell
    for nr, nc in _neighbors4(int(r), int(c)):
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == int(color):
            return True
    return False


def _object_cells(
    box: BBox,
    direction: str,
    depth: int,
    shape: str,
    rng: random.Random,
) -> list[Coord]:
    r0, c0, r1, c1 = box
    if direction in ("left", "right"):
        front_col = c0 + depth if direction == "left" else c1 - depth
        if shape == "bar" and r1 - r0 >= 4:
            row = rng.randint(r0 + 1, r1 - 2)
            return [(row, front_col), (row + 1, front_col)]
        row = rng.randint(r0 + 1, r1 - 1)
        return [(row, front_col)]

    front_row = r0 + depth if direction == "up" else r1 - depth
    if shape == "bar" and c1 - c0 >= 4:
        col = rng.randint(c0 + 1, c1 - 2)
        return [(front_row, col), (front_row, col + 1)]
    col = rng.randint(c0 + 1, c1 - 1)
    return [(front_row, col)]


def _try_place_extra_container(
    grid: np.ndarray,
    direction: str,
    depth: int,
    container_color: int,
    object_color: int,
    rng: random.Random,
    *,
    require_free_extra: bool,
) -> tuple[list[Coord], str] | None:
    h_grid, w_grid = grid.shape
    for _ in range(PLACE_ATTEMPTS):
        shape = rng.choice(("dot", "bar"))
        if direction in ("left", "right"):
            h = rng.randint(5, 7)
            w = rng.randint(depth + 3, depth + 5)
        else:
            h = rng.randint(depth + 3, depth + 5)
            w = rng.randint(5, 7)
        if h + 2 > h_grid or w + 2 > w_grid:
            continue
        r0 = rng.randint(1, h_grid - h - 1)
        c0 = rng.randint(1, w_grid - w - 1)
        box = (r0, c0, r0 + h - 1, c0 + w - 1)
        obj = _object_cells(box, direction, depth, shape, rng)
        cont = _container_cells(direction, r0, c0, h, w)
        colored = [(r, c, container_color) for r, c in cont] + [(r, c, object_color) for r, c in obj]
        if not _can_place_bundle(grid, colored, pad=1):
            continue
        temp = grid.copy()
        _paint(temp, cont, container_color)
        _paint(temp, obj, object_color)

        coords = np.array(obj, dtype=int)
        occupied = temp != BG
        occupied[coords[:, 0], coords[:, 1]] = False
        limit = 2 * depth + (1 if require_free_extra else 0)
        if all(_legal_position(_shifted(coords, direction, step), occupied) for step in range(1, limit + 1)):
            grid[:, :] = temp
            return obj, shape
    return None


def _add_facing_pair(
    grid: np.ndarray,
    orientation: str,
    deep: int,
    shallow: int,
    colors: list[int],
    rng: random.Random,
) -> bool:
    h_grid, w_grid = grid.shape
    deep_first = rng.choice((True, False))
    shape = rng.choice(("dot", "bar"))

    if orientation == "horizontal":
        height = rng.randint(5, 6)
        left_depth = deep if deep_first else shallow
        right_depth = shallow if deep_first else deep
        left_w = left_depth + 3
        right_w = right_depth + 3
        gap = deep + shallow - 1
        total_w = left_w + gap + right_w
        if total_w + 2 > w_grid or height + 2 > h_grid:
            return False
        r0 = rng.randint(1, h_grid - height - 1)
        c0 = rng.randint(1, w_grid - total_w - 1)
        left_box = (r0, c0, r0 + height - 1, c0 + left_w - 1)
        right_c0 = c0 + left_w + gap
        right_box = (r0, right_c0, r0 + height - 1, right_c0 + right_w - 1)

        if shape == "bar":
            row_cells = [r0 + height // 2, r0 + height // 2 + 1]
            if row_cells[-1] >= r0 + height - 1:
                row_cells = [r0 + height // 2]
        else:
            row_cells = [r0 + height // 2]
        left_front = left_box[3] - left_depth
        right_front = right_box[1] + right_depth
        left_obj = [(r, left_front) for r in row_cells]
        right_obj = [(r, right_front) for r in row_cells]
        left_cont = _container_cells("right", r0, c0, height, left_w)
        right_cont = _container_cells("left", r0, right_c0, height, right_w)
        colored = (
            [(r, c, colors[0]) for r, c in left_cont]
            + [(r, c, colors[1]) for r, c in left_obj]
            + [(r, c, colors[2]) for r, c in right_cont]
            + [(r, c, colors[3]) for r, c in right_obj]
        )
    else:
        width = rng.randint(5, 6)
        top_depth = deep if deep_first else shallow
        bottom_depth = shallow if deep_first else deep
        top_h = top_depth + 3
        bottom_h = bottom_depth + 3
        gap = deep + shallow - 1
        total_h = top_h + gap + bottom_h
        if total_h + 2 > h_grid or width + 2 > w_grid:
            return False
        r0 = rng.randint(1, h_grid - total_h - 1)
        c0 = rng.randint(1, w_grid - width - 1)
        top_box = (r0, c0, r0 + top_h - 1, c0 + width - 1)
        bottom_r0 = r0 + top_h + gap
        bottom_box = (bottom_r0, c0, bottom_r0 + bottom_h - 1, c0 + width - 1)

        if shape == "bar":
            col_cells = [c0 + width // 2, c0 + width // 2 + 1]
            if col_cells[-1] >= c0 + width - 1:
                col_cells = [c0 + width // 2]
        else:
            col_cells = [c0 + width // 2]
        top_front = top_box[2] - top_depth
        bottom_front = bottom_box[0] + bottom_depth
        top_obj = [(top_front, c) for c in col_cells]
        bottom_obj = [(bottom_front, c) for c in col_cells]
        top_cont = _container_cells("down", r0, c0, top_h, width)
        bottom_cont = _container_cells("up", bottom_r0, c0, bottom_h, width)
        colored = (
            [(r, c, colors[0]) for r, c in top_cont]
            + [(r, c, colors[1]) for r, c in top_obj]
            + [(r, c, colors[2]) for r, c in bottom_cont]
            + [(r, c, colors[3]) for r, c in bottom_obj]
        )

    if not _can_place_bundle(grid, colored, pad=1):
        return False
    for r, c, color in colored:
        grid[r, c] = color
    return True


def _try_place_blocked_container(
    grid: np.ndarray,
    direction: str,
    depth: int,
    container_color: int,
    object_color: int,
    blocker_color: int,
    rng: random.Random,
) -> bool:
    for _ in range(PLACE_ATTEMPTS):
        candidate = grid.copy()
        placed = _try_place_extra_container(
            candidate,
            direction,
            depth,
            container_color,
            object_color,
            rng,
            require_free_extra=False,
        )
        if placed is None:
            continue
        obj, _shape = placed
        coords = np.array(obj, dtype=int)
        possible_steps = list(range(depth + 1, 2 * depth + 1))
        rng.shuffle(possible_steps)
        for step in possible_steps:
            target = _shifted(coords, direction, step)
            h, w = candidate.shape
            if (
                np.any(target[:, 0] < 0)
                or np.any(target[:, 0] >= h)
                or np.any(target[:, 1] < 0)
                or np.any(target[:, 1] >= w)
            ):
                continue
            cells = [(int(r), int(c)) for r, c in target]
            rng.shuffle(cells)
            for cell in cells:
                r, c = cell
                if int(candidate[r, c]) != BG or _same_color_touch(candidate, cell, blocker_color):
                    continue
                test = candidate.copy()
                test[r, c] = int(blocker_color)
                occupied = test != BG
                occupied[coords[:, 0], coords[:, 1]] = False
                if not all(_legal_position(_shifted(coords, direction, s), occupied) for s in range(1, step)):
                    continue
                if _legal_position(_shifted(coords, direction, step), occupied):
                    continue
                grid[:, :] = test
                return True
    return False


def _range_has_extra_space(out: np.ndarray, detail: dict) -> bool:
    rec: MovingObject = detail["record"]
    final = np.array(detail["final"], dtype=int)
    occupied = out != BG
    occupied[final[:, 0], final[:, 1]] = False
    return _legal_position(_shifted(final, rec.direction, 1), occupied)


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        out, details = _solve(grid, trace=True)
    except Exception:
        return False
    if len(details) < 4:
        return False
    depths = [int(d["record"].depth) for d in details]
    if len(set(depths)) != len(depths):
        return False
    dirs = {d["record"].direction for d in details}
    if len(dirs) < 2:
        return False
    if np.array_equal(out, grid):
        return False
    range_free = any(d["reason"] == "range" and d["actual"] == d["planned"] and _range_has_extra_space(out, d) for d in details)
    static_stop = any(d["reason"] == "static" and d["actual"] < d["planned"] for d in details)
    settled_stop = any(d["reason"] == "settled" and d["actual"] < d["planned"] for d in details)
    foreground_preserved = int(np.sum(out != BG)) == int(np.sum(grid != BG))
    return bool(range_free and static_stop and settled_stop and foreground_preserved)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(24, 30)
    w = rng.randint(24, 30)
    grid = np.zeros((h, w), dtype=int)
    colors = list(rng.sample(COLOR_POOL, 9))

    depth_pair = rng.sample((1, 2, 3, 4), 2)
    deep, shallow = sorted(depth_pair, reverse=True)
    remaining = [d for d in (1, 2, 3, 4) if d not in depth_pair]
    rng.shuffle(remaining)
    free_depth, blocked_depth = remaining

    pair_orientation = rng.choice(("horizontal", "vertical"))
    if not _add_facing_pair(grid, pair_orientation, deep, shallow, colors[:4], rng):
        return None

    directions = list(DIR_ORDER)
    rng.shuffle(directions)
    free_direction = directions[0]
    blocked_direction = directions[1]
    if not _try_place_extra_container(
        grid,
        free_direction,
        free_depth,
        colors[4],
        colors[5],
        rng,
        require_free_extra=True,
    ):
        return None

    if not _try_place_blocked_container(
        grid,
        blocked_direction,
        blocked_depth,
        colors[6],
        colors[7],
        colors[8],
        rng,
    ):
        return None

    return grid if _semantic_checks(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spring-container input")
