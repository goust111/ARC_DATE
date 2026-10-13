from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
COLORS = tuple(range(10))
SIDE_CHOICES = (1, 1, 3)
ORIENTATIONS = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(g: np.ndarray, color: int) -> list[np.ndarray]:
    mask = g == int(color)
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    H, W = mask.shape
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _anchor_and_away(g: np.ndarray, bg: int) -> tuple[tuple[int, int], tuple[int, int]]:
    H, W = g.shape
    border: list[tuple[int, int]] = []
    for c in range(W):
        if int(g[0, c]) not in (int(bg), GREY):
            border.append((0, c))
        if int(g[H - 1, c]) not in (int(bg), GREY):
            border.append((H - 1, c))
    for r in range(1, H - 1):
        if int(g[r, 0]) not in (int(bg), GREY):
            border.append((r, 0))
        if int(g[r, W - 1]) not in (int(bg), GREY):
            border.append((r, W - 1))

    assert len(border) == 1, "expected exactly one non-grey border anchor"
    ar, ac = border[0]
    assert not (ar in (0, H - 1) and ac in (0, W - 1)), "anchor must be on one edge, not a corner"
    if ar == 0:
        return (ar, ac), (1, 0)
    if ar == H - 1:
        return (ar, ac), (-1, 0)
    if ac == 0:
        return (ar, ac), (0, 1)
    if ac == W - 1:
        return (ar, ac), (0, -1)
    assert False, "anchor must lie on the border"


def _to_s(anchor: tuple[int, int], away: tuple[int, int], r: int, c: int) -> int:
    ar, ac = anchor
    dr, dc = away
    return int((int(r) - ar) * dr + (int(c) - ac) * dc)


def _from_s(anchor: tuple[int, int], away: tuple[int, int], s: int) -> tuple[int, int]:
    ar, ac = anchor
    dr, dc = away
    return int(ar + dr * int(s)), int(ac + dc * int(s))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    H, W = shape
    return 0 <= int(r) < H and 0 <= int(c) < W


def _find_blocks(g: np.ndarray, bg: int, anchor: tuple[int, int], away: tuple[int, int]) -> list[dict]:
    ar, ac = anchor
    anchor_color = int(g[ar, ac])
    objects: list[dict] = []
    trail_cells: set[tuple[int, int]] = set()

    for raw in np.unique(g):
        color = int(raw)
        if color in (int(bg), GREY):
            continue
        for coords in _components_of_color(g, color):
            cells = {(int(r), int(c)) for r, c in coords}
            if (ar, ac) in cells:
                assert color == anchor_color and len(cells) == 1, "anchor must be an isolated border pixel"
                continue

            r0, c0, r1, c1 = _bbox(coords)
            side_h = r1 - r0 + 1
            side_w = c1 - c0 + 1
            assert side_h == side_w and side_h % 2 == 1, "blocks must be odd-sided solid squares"
            assert len(cells) == side_h * side_w, "block component must fill its square bbox"

            if away[0] == 0:
                assert r0 <= ar <= r1, "horizontal block must sit on the anchor row"
            else:
                assert c0 <= ac <= c1, "vertical block must sit on the anchor column"

            s_values = [_to_s(anchor, away, r, c) for r, c in cells]
            assert min(s_values) > 0, "all blocks must be on the inward side of the anchor"
            front = int(min(s_values))
            rear = int(max(s_values))
            assert rear - front + 1 == side_h, "block axis span must match its square side"

            fuel = 0
            own_trail: list[tuple[int, int]] = []
            while True:
                rr, cc = _from_s(anchor, away, rear + fuel + 1)
                if not _in_bounds(g.shape, rr, cc) or int(g[rr, cc]) != GREY:
                    break
                own_trail.append((rr, cc))
                trail_cells.add((rr, cc))
                fuel += 1
            assert fuel > 0, "each block must have a grey trail segment as fuel"

            objects.append(
                {
                    "color": color,
                    "coords": coords.copy(),
                    "front": front,
                    "rear": rear,
                    "fuel": fuel,
                    "trail": own_trail,
                }
            )

    all_grey = {(int(r), int(c)) for r, c in np.argwhere(g == GREY)}
    assert all_grey == trail_cells, "every grey pixel must be part of exactly one trail"
    fronts = [obj["front"] for obj in objects]
    assert len(objects) >= 1 and len(fronts) == len(set(fronts)), "blocks must form a unique queue order"
    objects.sort(key=lambda obj: obj["front"])
    return objects


def _solve(grid: np.ndarray, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    anchor, away = _anchor_and_away(g, bg)
    objects = _find_blocks(g, bg, anchor, away)

    out = g.copy()
    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)
        for r, c in obj["trail"]:
            out[r, c] = int(bg)

    obstacle_end = 0
    meta = []
    for obj in objects:
        target_front = obstacle_end + 1
        needed = int(obj["front"] - target_front)
        assert needed >= 0, "queued blocks must appear behind the current obstacle"
        fuel = int(obj["fuel"])
        move = min(needed, fuel)
        residual = fuel - move

        for r, c in obj["coords"]:
            rr = int(r) - int(away[0]) * move
            cc = int(c) - int(away[1]) * move
            assert _in_bounds(out.shape, rr, cc) and int(out[rr, cc]) == int(bg), "moved blocks must not collide"
            out[rr, cc] = int(obj["color"])

        new_rear = int(obj["rear"] - move)
        for j in range(1, residual + 1):
            rr, cc = _from_s(anchor, away, new_rear + j)
            assert _in_bounds(out.shape, rr, cc) and int(out[rr, cc]) == int(bg), "residual trail must fit"
            out[rr, cc] = GREY

        obstacle_end = new_rear + residual
        meta.append(
            {
                "front": int(obj["front"]),
                "needed": needed,
                "fuel": fuel,
                "move": move,
                "residual": residual,
                "obstacle_end": obstacle_end,
            }
        )

    if return_meta:
        return out, meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide queued fuel-tailed blocks inward to the border anchor, stopping each block when its trail fuel runs out.

    Input structure:
    - A dominant background contains one non-grey anchor pixel on a non-corner border cell.
    - On the anchor row or column, several odd-sided solid colored square blocks lie on the inward side of that anchor.
    - Each block has a one-cell-wide grey trail attached to its rear face along the anchor projection line; the trail length is that block's fuel.

    Rule:
    - Process blocks from nearest to farthest from the anchor.
    - A block needs d steps to put its front face immediately behind the anchor or behind the farthest occupied cell of the already stopped block and residual trail ahead.
    - It moves min(d, trail length) steps toward the anchor, burning that many grey trail cells from the far end while the remaining trail stays attached behind the moved block.
    - Original block and trail cells vacated by the move become background; later blocks treat every stopped block plus any residual grey trail as the next obstacle.

    Key insight:
    - The grey trail is not decoration: it is both the fuel cap for the current slide and, when unburned, part of the physical queue that changes every later berth.
    """
    return _solve(grid)


def _orientation_geometry(orientation: str, axis_len: int, perp_len: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    line = random.randint(2, perp_len - 3)
    if orientation == "top":
        return (axis_len + 1, perp_len), (0, line), (1, 0)
    if orientation == "bottom":
        return (axis_len + 1, perp_len), (axis_len, line), (-1, 0)
    if orientation == "left":
        return (perp_len, axis_len + 1), (line, 0), (0, 1)
    if orientation == "right":
        return (perp_len, axis_len + 1), (line, axis_len), (0, -1)
    raise ValueError(f"unknown orientation: {orientation}")


def _block_cells(anchor: tuple[int, int], away: tuple[int, int], front: int, side: int) -> list[tuple[int, int]]:
    ar, ac = anchor
    half = side // 2
    cells: list[tuple[int, int]] = []
    if away[0] == 0:
        for off in range(-half, half + 1):
            for s in range(front, front + side):
                r = ar + off
                c = ac + away[1] * s
                cells.append((int(r), int(c)))
    else:
        for s in range(front, front + side):
            for off in range(-half, half + 1):
                r = ar + away[0] * s
                c = ac + off
                cells.append((int(r), int(c)))
    return cells


def _paint_spec_grid(
    shape: tuple[int, int],
    bg: int,
    anchor: tuple[int, int],
    anchor_color: int,
    away: tuple[int, int],
    specs: list[dict],
) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    grid[anchor] = int(anchor_color)
    for spec in specs:
        front = int(spec["front"])
        side = int(spec["side"])
        rear = front + side - 1
        for r, c in _block_cells(anchor, away, front, side):
            grid[r, c] = int(spec["color"])
        for j in range(1, int(spec["fuel"]) + 1):
            grid[_from_s(anchor, away, rear + j)] = GREY
    return grid


def _simulate_specs(
    shape: tuple[int, int],
    bg: int,
    anchor: tuple[int, int],
    anchor_color: int,
    away: tuple[int, int],
    specs: list[dict],
    *,
    ignore_residual_obstacle: bool = False,
    unlimited_fuel: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    out = np.full(shape, int(bg), dtype=int)
    out[anchor] = int(anchor_color)
    obstacle_end = 0
    meta: list[dict] = []
    for spec in specs:
        front = int(spec["front"])
        side = int(spec["side"])
        rear = front + side - 1
        fuel = int(spec["fuel"])
        needed = front - (obstacle_end + 1)
        if needed < 0:
            raise ValueError("invalid spec order")
        move = needed if unlimited_fuel else min(needed, fuel)
        residual = max(0, fuel - move)
        new_front = front - move
        new_rear = rear - move
        for r, c in _block_cells(anchor, away, new_front, side):
            out[r, c] = int(spec["color"])
        for j in range(1, residual + 1):
            out[_from_s(anchor, away, new_rear + j)] = GREY
        obstacle_end = new_rear if ignore_residual_obstacle else new_rear + residual
        meta.append({"needed": needed, "fuel": fuel, "move": move, "residual": residual})
    return out, meta


def _build_specs(axis_len: int, block_colors: list[int]) -> list[dict] | None:
    for _ in range(80):
        n = random.randint(4, min(5, len(block_colors)))
        branches = ["surplus", "tail_sensitive", "deficit", "after_deficit"]
        branches.extend(random.choice(("surplus", "exact", "deficit")) for _ in range(n - 4))

        specs: list[dict] = []
        cursor = random.randint(2, 3)
        obstacle_end = 0
        previous_residual = 0
        ok = True
        for i, branch in enumerate(branches):
            side = random.choice(SIDE_CHOICES)
            target = obstacle_end + 1
            min_front = max(cursor, target + 1)
            if branch == "deficit":
                min_front = max(min_front, target + 2)
            front = min_front + random.randint(0, 1)
            needed = front - target
            if needed <= 0:
                ok = False
                break

            if branch == "surplus":
                fuel = needed + random.randint(1, 2)
            elif branch == "tail_sensitive":
                fuel = needed + max(1, previous_residual) + random.randint(0, 1)
            elif branch == "after_deficit":
                fuel = needed + random.randint(0, 2)
            elif branch == "exact":
                fuel = needed
            else:
                if needed < 2:
                    ok = False
                    break
                fuel = random.randint(1, needed - 1)

            rear = front + side - 1
            if rear + fuel > axis_len:
                ok = False
                break

            move = min(needed, fuel)
            residual = fuel - move
            specs.append({"front": front, "side": side, "fuel": fuel, "color": int(block_colors[i])})
            obstacle_end = rear - move + residual
            previous_residual = residual
            cursor = rear + fuel + random.randint(1, 2)

        if ok:
            return specs
    return None


def _semantically_valid(inp: np.ndarray, expected: np.ndarray, meta: list[dict], alt_no_tail: np.ndarray, alt_no_fuel: np.ndarray) -> bool:
    if np.array_equal(inp, expected):
        return False
    if not any(m["fuel"] > m["needed"] for m in meta):
        return False
    if not any(m["fuel"] < m["needed"] for m in meta[:-1]):
        return False
    if not any(m["residual"] > 0 for m in meta[:-1]):
        return False
    if np.array_equal(expected, alt_no_tail):
        return False
    if np.array_equal(expected, alt_no_fuel):
        return False
    try:
        actual, solver_meta = _solve(inp, return_meta=True)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    return len(solver_meta) == len(meta)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in COLORS if c != GREY])
    anchor_color = random.choice([c for c in COLORS if c not in (bg, GREY)])
    block_colors = [c for c in COLORS if c not in (bg, GREY, anchor_color)]
    random.shuffle(block_colors)

    orientation = random.choice(ORIENTATIONS)
    axis_len = random.randint(22, 29)
    perp_len = random.randint(7, 18)
    shape, anchor, away = _orientation_geometry(orientation, axis_len, perp_len)

    specs = _build_specs(axis_len, block_colors)
    if specs is None:
        return None

    inp = _paint_spec_grid(shape, bg, anchor, anchor_color, away, specs)
    expected, meta = _simulate_specs(shape, bg, anchor, anchor_color, away, specs)
    alt_no_tail, _ = _simulate_specs(
        shape, bg, anchor, anchor_color, away, specs, ignore_residual_obstacle=True
    )
    alt_no_fuel, _ = _simulate_specs(
        shape, bg, anchor, anchor_color, away, specs, unlimited_fuel=True
    )
    if not _semantically_valid(inp, expected, meta, alt_no_tail, alt_no_fuel):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid fuel-tail queue input after {GEN_ATTEMPTS} attempts")
