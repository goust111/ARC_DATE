import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
MARKER_COLORS = (2, 3, 4, 7)

GEN_ATTEMPTS = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _make_cup(r0: int, c0: int, inner_h: int, inner_w: int, color: int, ray_len: int) -> dict:
    return {
        "r0": int(r0),
        "c0": int(c0),
        "inner_h": int(inner_h),
        "inner_w": int(inner_w),
        "color": int(color),
        "ray_len": int(ray_len),
    }


def _cup_bounds(cup: dict) -> tuple[int, int, int, int]:
    r0 = cup["r0"]
    c0 = cup["c0"]
    r1 = r0 + cup["inner_h"]
    c1 = c0 + cup["inner_w"] + 1
    return r0, c0, r1, c1


def _cup_marker(cup: dict) -> tuple[int, int]:
    r0, c0, _r1, _c1 = _cup_bounds(cup)
    return r0 + cup["ray_len"], c0 + 1 + cup["inner_w"] // 2


def _cup_conservative_box(cup: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _cup_bounds(cup)
    e = cup["ray_len"]
    return r0 - e, c0 - e, r1, c1 + e


def _stamp_cup(grid: np.ndarray, cup: dict) -> None:
    r0, c0, r1, c1 = _cup_bounds(cup)
    grid[r0:r1 + 1, c0] = BLUE
    grid[r0:r1 + 1, c1] = BLUE
    grid[r1, c0:c1 + 1] = BLUE
    mr, mc = _cup_marker(cup)
    grid[mr, mc] = cup["color"]


def _analyze_cups(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    cups: list[dict] = []
    for comp in _components(g == BLUE):
        r0, c0, r1, c1 = _bbox(comp)
        inner_h = r1 - r0
        inner_w = c1 - c0 - 1
        if inner_h < 4 or inner_h > 7 or inner_w < 3 or inner_w > 5:
            raise AssertionError("blue component has invalid cup dimensions")

        expected = set()
        for rr in range(r0, r1 + 1):
            expected.add((rr, c0))
            expected.add((rr, c1))
        for cc in range(c0, c1 + 1):
            expected.add((r1, cc))
        if set(comp) != expected:
            raise AssertionError("blue component is not a valid U-shaped cup")

        marker = None
        for rr in range(r0, r1):
            for cc in range(c0 + 1, c1):
                value = int(g[rr, cc])
                if value == BLUE:
                    raise AssertionError("cup interior contains blue pixels")
                if value != BLACK:
                    if marker is not None:
                        raise AssertionError("cup interior must contain exactly one marker")
                    marker = (rr, cc, value)
        if marker is None:
            raise AssertionError("cup interior must contain exactly one marker")

        mr, mc, color = marker
        ray_len = mr - r0
        if color not in MARKER_COLORS:
            raise AssertionError("marker color must come from the visible marker palette")
        if ray_len <= 0 or ray_len >= inner_h - 1:
            raise AssertionError("marker row must leave both fill and ray non-zero")

        cups.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "marker_r": mr,
                "marker_c": mc,
                "color": color,
                "ray_len": ray_len,
            }
        )

    cups.sort(key=lambda cup: (cup["r0"], cup["c0"]))
    return cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each blue cup up to its marker height and convert the remaining height into matching overflow rays.

    Input structure:
    - Black (0) background with 2-3 blue (1) U-shaped cup outlines.
    - Each cup has a rectangular interior, open top, and exactly one non-blue marker pixel inside.
    - Marker colors come from a visible palette and the marker row is neither the top nor bottom interior row.

    Rule:
    - Detect every blue cup and locate its unique interior marker.
    - Fill the cup interior from the bottom up through the marker row with the marker color.
    - Let the unfilled height above the marker determine the ray length; draw left/up and right/up diagonal overflow rays from the two cup-rim corners, stopping before any non-black input pixel or the grid boundary.
    - Add one horizontal cap row of the same color directly above the cup opening, while keeping the blue outline unchanged.

    Key insight:
    - The marker's vertical position encodes a conserved split: filled height plus overflow-ray length equals the cup's interior height.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for cup in _analyze_cups(g):
        r0, c0, r1, c1 = cup["r0"], cup["c0"], cup["r1"], cup["c1"]
        color = cup["color"]
        marker_r = cup["marker_r"]
        ray_len = cup["ray_len"]

        out[marker_r:r1, c0 + 1:c1] = color
        out[r0 - 1, c0:c1 + 1] = color

        for step in range(ray_len):
            rr = r0 - 1 - step
            cc = c0 - 1 - step
            if rr < 0 or cc < 0 or g[rr, cc] != BLACK:
                break
            out[rr, cc] = color

        for step in range(ray_len):
            rr = r0 - 1 - step
            cc = c1 + 1 + step
            if rr < 0 or cc >= g.shape[1] or g[rr, cc] != BLACK:
                break
            out[rr, cc] = color

    return out


def _inflate_mask(mask: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return mask.copy()
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    coords = np.argwhere(mask)
    for r, c in coords:
        r0 = max(0, int(r) - pad)
        r1 = min(h, int(r) + pad + 1)
        c0 = max(0, int(c) - pad)
        c1 = min(w, int(c) + pad + 1)
        out[r0:r1, c0:c1] = True
    return out


def _box_clear(mask: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    if r0 < 0 or c0 < 0 or r1 >= mask.shape[0] or c1 >= mask.shape[1]:
        return False
    return not bool(mask[r0:r1 + 1, c0:c1 + 1].any())


def _ray_cells_from_input(grid: np.ndarray, cup: dict, side: str) -> list[tuple[int, int]]:
    if side not in ("left", "right"):
        raise ValueError(side)
    r0, c0, _r1, c1 = cup["r0"], cup["c0"], cup["r1"], cup["c1"]
    dc = -1 if side == "left" else 1
    start_c = c0 - 1 if side == "left" else c1 + 1
    cells: list[tuple[int, int]] = []
    for step in range(cup["ray_len"]):
        rr = r0 - 1 - step
        cc = start_c + dc * step
        if rr < 0 or cc < 0 or cc >= grid.shape[1] or grid[rr, cc] != BLACK:
            break
        cells.append((rr, cc))
    return cells


def _cup_additions(grid: np.ndarray, cup: dict) -> tuple[set[tuple[int, int]], bool]:
    added: set[tuple[int, int]] = set()
    r0, c0, r1, c1 = cup["r0"], cup["c0"], cup["r1"], cup["c1"]
    for rr in range(cup["marker_r"], r1):
        for cc in range(c0 + 1, c1):
            added.add((rr, cc))
    for cc in range(c0, c1 + 1):
        added.add((r0 - 1, cc))

    blocked = False
    for side in ("left", "right"):
        cells = _ray_cells_from_input(grid, cup, side)
        added.update(cells)
        if len(cells) < cup["ray_len"]:
            step = len(cells)
            if side == "left":
                rr = r0 - 1 - step
                cc = c0 - 1 - step
            else:
                rr = r0 - 1 - step
                cc = c1 + 1 + step
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and grid[rr, cc] != BLACK:
                blocked = True
            else:
                raise AssertionError("ray shortened by boundary, which the generator forbids")
    return added, blocked


def _semantic_ok(grid: np.ndarray, require_blocked: bool | None) -> bool:
    try:
        cups = _analyze_cups(grid)
        if len(cups) < 2 or len(cups) > 3:
            return False
        if len({cup["color"] for cup in cups}) != len(cups):
            return False
        if len({cup["ray_len"] for cup in cups}) != len(cups):
            return False
        if any(cup["r0"] < 5 for cup in cups):
            return False

        out = main(grid)
        if np.array_equal(out, grid):
            return False

        blocked_found = False
        addition_sets: list[set[tuple[int, int]]] = []
        for cup in cups:
            added, blocked = _cup_additions(grid, cup)
            if blocked:
                blocked_found = True
            if not added:
                return False
            for cell in added:
                if out[cell] != cup["color"]:
                    return False
            addition_sets.append(added)

        for i in range(len(addition_sets)):
            for j in range(i + 1, len(addition_sets)):
                if addition_sets[i] & addition_sets[j]:
                    return False

        if require_blocked is True and not blocked_found:
            return False
        if require_blocked is False and blocked_found:
            return False
        return True
    except Exception:
        return False


def _place_open_cups(shape: tuple[int, int], cups: list[dict]) -> np.ndarray | None:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    for cup in sorted(cups, key=lambda item: (item["inner_w"] + item["ray_len"], item["inner_h"]), reverse=True):
        placed = False
        for _ in range(250):
            candidate = dict(cup)
            max_c0 = w - candidate["inner_w"] - candidate["ray_len"] - 2
            if max_c0 < candidate["ray_len"]:
                return None
            max_r0 = h - candidate["inner_h"] - 1
            if max_r0 < 5:
                return None
            candidate["c0"] = random.randint(candidate["ray_len"], max_c0)
            candidate["r0"] = random.randint(5, max_r0)
            if not _box_clear(occupied, _cup_conservative_box(candidate)):
                continue
            _stamp_cup(grid, candidate)
            occupied = _inflate_mask(main(grid) != BLACK, 1)
            placed = True
            break
        if not placed:
            return None
    return grid


def _random_open_scene() -> np.ndarray | None:
    n_cups = random.choice((2, 2, 3))
    ray_lens = random.sample((1, 2, 3, 4, 5), n_cups)
    colors = random.sample(MARKER_COLORS, n_cups)

    cups = []
    for color, ray_len in zip(colors, ray_lens):
        inner_h = random.randint(max(4, ray_len + 2), 7)
        inner_w = random.randint(3, 5)
        cups.append(_make_cup(0, 0, inner_h, inner_w, color, ray_len))

    if n_cups == 2:
        shape = (random.randint(16, 22), random.randint(18, 25))
    else:
        shape = (random.randint(18, 24), random.randint(22, 28))
    grid = _place_open_cups(shape, cups)
    if grid is None:
        return None
    return grid if _semantic_ok(grid, require_blocked=False) else None


def _blocked_pair_scene() -> np.ndarray | None:
    n_cups = random.choice((2, 2, 3))
    source_ray = random.randint(2, 5)
    other_rays = random.sample([v for v in (1, 2, 3, 4, 5) if v != source_ray], n_cups - 1)
    ray_lens = [source_ray] + other_rays
    colors = random.sample(MARKER_COLORS, n_cups)

    source = _make_cup(0, 0, random.randint(source_ray + 2, 7), random.randint(3, 5), colors[0], source_ray)
    target = _make_cup(0, 0, random.randint(max(4, other_rays[0] + 2), 7), random.randint(3, 5), colors[1], other_rays[0])
    side = random.choice(("left", "right"))
    block_step = random.randint(1, source_ray - 1)

    if n_cups == 2:
        shape = (random.randint(18, 24), random.randint(20, 28))
    else:
        shape = (random.randint(20, 25), random.randint(24, 29))
    h, w = shape

    for _ in range(250):
        s = dict(source)
        t = dict(target)
        max_r0 = h - s["inner_h"] - 1
        if max_r0 < 5:
            return None

        if side == "left":
            min_c0 = max(
                s["ray_len"],
                t["ray_len"] + block_step + t["inner_w"] + 2,
            )
            max_c0 = w - s["inner_w"] - s["ray_len"] - 2
            min_r0 = max(5, 6 + block_step + t["inner_h"])
            if min_c0 > max_c0 or min_r0 > max_r0:
                return None
            s["c0"] = random.randint(min_c0, max_c0)
            s["r0"] = random.randint(min_r0, max_r0)
            blocker_r = s["r0"] - 1 - block_step
            blocker_c = s["c0"] - 1 - block_step
            t["c0"] = blocker_c - t["inner_w"] - 1
            t["r0"] = blocker_r - t["inner_h"]
        else:
            min_c0 = s["ray_len"]
            max_c0 = min(
                w - s["inner_w"] - s["ray_len"] - 2,
                w - s["inner_w"] - t["inner_w"] - t["ray_len"] - block_step - 4,
            )
            min_r0 = max(5, 6 + block_step + t["inner_h"])
            if min_c0 > max_c0 or min_r0 > max_r0:
                return None
            s["c0"] = random.randint(min_c0, max_c0)
            s["r0"] = random.randint(min_r0, max_r0)
            blocker_r = s["r0"] - 1 - block_step
            blocker_c = s["c0"] + s["inner_w"] + 2 + block_step
            t["c0"] = blocker_c
            t["r0"] = blocker_r - t["inner_h"]

        if t["r0"] < 5:
            continue
        if t["c0"] < t["ray_len"]:
            continue
        if t["c0"] + t["inner_w"] + 1 + t["ray_len"] >= w:
            continue

        grid = np.zeros((h, w), dtype=int)
        _stamp_cup(grid, s)
        _stamp_cup(grid, t)
        if not _semantic_ok(grid, require_blocked=True):
            continue

        if n_cups == 2:
            return grid

        extra = _make_cup(
            0,
            0,
            random.randint(max(4, other_rays[1] + 2), 7),
            random.randint(3, 5),
            colors[2],
            other_rays[1],
        )
        occupied = _inflate_mask(main(grid) != BLACK, 1)
        placed = False
        for _ in range(250):
            candidate = dict(extra)
            max_c0 = w - candidate["inner_w"] - candidate["ray_len"] - 2
            max_r0 = h - candidate["inner_h"] - 1
            if max_c0 < candidate["ray_len"] or max_r0 < 5:
                break
            candidate["c0"] = random.randint(candidate["ray_len"], max_c0)
            candidate["r0"] = random.randint(5, max_r0)
            if not _box_clear(occupied, _cup_conservative_box(candidate)):
                continue
            grid2 = grid.copy()
            _stamp_cup(grid2, candidate)
            if _semantic_ok(grid2, require_blocked=True):
                return grid2
            placed = True
        if placed:
            return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        if random.random() < 0.45:
            grid = _blocked_pair_scene()
        else:
            grid = _random_open_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cup-ray ARC input")
