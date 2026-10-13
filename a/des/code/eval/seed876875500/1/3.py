import random
from collections import Counter

import numpy as np


SIDES = ("top", "bottom", "left", "right")
PALETTE = tuple(range(10))
GEN_RETRIES = 400
PLACE_RETRIES = 200


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _fill_single_pixel_gaps(grid: np.ndarray, bg: int, color: int) -> np.ndarray:
    mask = grid == color
    filled = mask.copy()
    h, w = grid.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != bg:
                continue
            if mask[r, c - 1] and mask[r, c + 1]:
                filled[r, c] = True
            if mask[r - 1, c] and mask[r + 1, c]:
                filled[r, c] = True
    return filled


def _gap_indices(side_values: np.ndarray, bg: int, color: int) -> list[int] | None:
    vals = [int(v) for v in side_values.tolist()]
    if any(v not in (bg, color) for v in vals):
        return None
    gaps = [idx for idx, v in enumerate(vals) if v == bg]
    for a, b in zip(gaps, gaps[1:]):
        if b == a + 1:
            return None
    return gaps


def _detect_rooms(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    h, w = grid.shape
    groups: dict[int, list[dict]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        closed = _fill_single_pixel_gaps(grid, bg, color)
        rooms: list[dict] = []
        for comp in _components(closed):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
                continue
            frame = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            frame[0, :] = True
            frame[-1, :] = True
            frame[:, 0] = True
            frame[:, -1] = True
            if not np.array_equal(closed[r0 : r1 + 1, c0 : c1 + 1], frame):
                continue
            if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg):
                continue
            if not (
                int(grid[r0, c0]) == color
                and int(grid[r0, c1]) == color
                and int(grid[r1, c0]) == color
                and int(grid[r1, c1]) == color
            ):
                continue

            top = _gap_indices(grid[r0, c0 + 1 : c1], bg, color)
            bottom = _gap_indices(grid[r1, c0 + 1 : c1], bg, color)
            left = _gap_indices(grid[r0 + 1 : r1, c0], bg, color)
            right = _gap_indices(grid[r0 + 1 : r1, c1], bg, color)
            if top is None or bottom is None or left is None or right is None:
                continue
            gap_count = len(top) + len(bottom) + len(left) + len(right)
            if gap_count == 0:
                continue
            rooms.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "color": color,
                    "gaps": {
                        "top": [c0 + 1 + idx for idx in top],
                        "bottom": [c0 + 1 + idx for idx in bottom],
                        "left": [r0 + 1 + idx for idx in left],
                        "right": [r0 + 1 + idx for idx in right],
                    },
                    "gap_count": gap_count,
                }
            )
        if rooms:
            groups[color] = rooms

    assert groups, "expected at least one room"
    ranked = sorted(
        groups.items(),
        key=lambda item: (
            len(item[1]),
            sum((room["bbox"][2] - room["bbox"][0] + 1) * (room["bbox"][3] - room["bbox"][1] + 1) for room in item[1]),
        ),
        reverse=True,
    )
    wall_color, rooms = ranked[0]
    return int(wall_color), rooms


def _legend_map(grid: np.ndarray, bg: int, wall_color: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, wall_color):
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            assert height == 1, "legend rows must be horizontal"
            assert width == comp.shape[0], "legend rows must be solid"
            assert width not in mapping, "legend lengths must be unique"
            mapping[width] = color
    assert mapping, "expected a color legend"
    return mapping


def _paint_rays_from_rooms(grid: np.ndarray, rooms: list[dict], legend: dict[int, int]) -> np.ndarray:
    out = grid.copy()
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        gap_count = int(room["gap_count"])
        color = int(legend[gap_count])
        for col in room["gaps"]["top"]:
            for step in range(1, gap_count + 1):
                rr = r0 + step
                if rr >= r1:
                    break
                out[rr, col] = color
        for col in room["gaps"]["bottom"]:
            for step in range(1, gap_count + 1):
                rr = r1 - step
                if rr <= r0:
                    break
                out[rr, col] = color
        for row in room["gaps"]["left"]:
            for step in range(1, gap_count + 1):
                cc = c0 + step
                if cc >= c1:
                    break
                out[row, cc] = color
        for row in room["gaps"]["right"]:
            for step in range(1, gap_count + 1):
                cc = c1 - step
                if cc <= c0:
                    break
                out[row, cc] = color
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    wall_color, rooms = _detect_rooms(g, bg)
    legend = _legend_map(g, bg, wall_color)
    assert all(room["gap_count"] in legend for room in rooms), "legend missing a required length"
    return _paint_rays_from_rooms(g, rooms, legend)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each room's wall gaps, read the matching color from the legend row of
    the same length, and draw inward rays of that color from every gap.
    """
    return _solve(grid)


def _choose_palette(rng: random.Random, max_n: int) -> tuple[int, int, dict[int, int]]:
    bg = rng.choice(PALETTE)
    remaining = [c for c in PALETTE if c != bg]
    wall = rng.choice(remaining)
    remaining = [c for c in remaining if c != wall]
    legend_colors = rng.sample(remaining, max_n)
    legend = {idx + 1: legend_colors[idx] for idx in range(max_n)}
    return bg, wall, legend


def _legend_bbox(h: int, w: int, max_n: int, corner: str) -> tuple[int, int, int, int]:
    legend_h = 2 * max_n - 1
    legend_w = max_n
    if corner == "tl":
        return 0, 0, legend_h - 1, legend_w - 1
    if corner == "tr":
        return 0, w - legend_w, legend_h - 1, w - 1
    if corner == "bl":
        return h - legend_h, 0, h - 1, legend_w - 1
    return h - legend_h, w - legend_w, h - 1, w - 1


def _draw_legend(grid: np.ndarray, legend: dict[int, int], corner: str) -> None:
    h, w = grid.shape
    max_n = max(legend)
    for n, color in legend.items():
        row_offset = 2 * (n - 1)
        if corner in ("tl", "tr"):
            r = row_offset
        else:
            r = h - 1 - row_offset
        if corner in ("tl", "bl"):
            c0 = 0
            c1 = n
        else:
            c0 = w - n
            c1 = w
        grid[r, c0:c1] = color
    assert _legend_bbox(h, w, max_n, corner)


def _sample_non_adjacent_positions(length: int, count: int, rng: random.Random) -> list[int] | None:
    if count == 0:
        return []
    positions = list(range(1, length + 1))
    for _ in range(80):
        chosen = sorted(rng.sample(positions, count))
        if all(b - a > 1 for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _sample_room_shape(n: int, rng: random.Random) -> tuple[int, int, dict[str, list[int]]] | None:
    for _ in range(120):
        inner_h = rng.randint(max(3, n), max(6, n + 3))
        inner_w = rng.randint(max(3, n), max(6, n + 3))
        side_caps = {
            "top": (inner_w + 1) // 2,
            "bottom": (inner_w + 1) // 2,
            "left": (inner_h + 1) // 2,
            "right": (inner_h + 1) // 2,
        }
        if sum(side_caps.values()) < n:
            continue

        counts = {side: 0 for side in SIDES}
        remaining = dict(side_caps)
        for _ in range(n):
            candidates = [side for side in SIDES if remaining[side] > 0]
            side = rng.choice(candidates)
            counts[side] += 1
            remaining[side] -= 1

        gaps: dict[str, list[int]] = {}
        ok = True
        for side in SIDES:
            length = inner_w if side in ("top", "bottom") else inner_h
            positions = _sample_non_adjacent_positions(length, counts[side], rng)
            if positions is None:
                ok = False
                break
            gaps[side] = positions
        if ok:
            return inner_h, inner_w, gaps
    return None


def _too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 2) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _draw_room(grid: np.ndarray, wall: int, spec: dict) -> None:
    r0, c0, r1, c1 = spec["bbox"]
    grid[r0, c0 : c1 + 1] = wall
    grid[r1, c0 : c1 + 1] = wall
    grid[r0 : r1 + 1, c0] = wall
    grid[r0 : r1 + 1, c1] = wall
    for col in spec["gaps"]["top"]:
        grid[r0, c0 + col] = spec["bg"]
    for col in spec["gaps"]["bottom"]:
        grid[r1, c0 + col] = spec["bg"]
    for row in spec["gaps"]["left"]:
        grid[r0 + row, c0] = spec["bg"]
    for row in spec["gaps"]["right"]:
        grid[r0 + row, c1] = spec["bg"]


def _expected_from_specs(grid: np.ndarray, legend: dict[int, int], specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        n = int(spec["n"])
        color = int(legend[n])
        for col in spec["gaps"]["top"]:
            for step in range(1, n + 1):
                rr = r0 + step
                if rr >= r1:
                    break
                out[rr, c0 + col] = color
        for col in spec["gaps"]["bottom"]:
            for step in range(1, n + 1):
                rr = r1 - step
                if rr <= r0:
                    break
                out[rr, c0 + col] = color
        for row in spec["gaps"]["left"]:
            for step in range(1, n + 1):
                cc = c0 + step
                if cc >= c1:
                    break
                out[r0 + row, cc] = color
        for row in spec["gaps"]["right"]:
            for step in range(1, n + 1):
                cc = c1 - step
                if cc <= c0:
                    break
                out[r0 + row, cc] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    max_n = rng.randint(3, 5)
    room_count = rng.randint(1, 4)
    h = rng.randint(max(18, 2 * max_n + 7), 30)
    w = rng.randint(max(18, max_n + 12), 30)
    bg, wall, legend = _choose_palette(rng, max_n)
    corner = rng.choice(("tl", "tr", "bl", "br"))

    grid = np.full((h, w), bg, dtype=int)
    _draw_legend(grid, legend, corner)
    forbidden = [_legend_bbox(h, w, max_n, corner)]

    specs: list[dict] = []
    for _ in range(room_count):
        n = rng.randint(1, max_n)
        shape = _sample_room_shape(n, rng)
        if shape is None:
            return None
        inner_h, inner_w, gaps = shape
        room_h = inner_h + 2
        room_w = inner_w + 2

        placed = False
        for _ in range(PLACE_RETRIES):
            r0 = rng.randint(1, h - room_h - 1)
            c0 = rng.randint(1, w - room_w - 1)
            bbox = (r0, c0, r0 + room_h - 1, c0 + room_w - 1)
            if any(_too_close(bbox, other) for other in forbidden):
                continue
            spec = {"bbox": bbox, "gaps": gaps, "n": n, "bg": bg}
            _draw_room(grid, wall, spec)
            specs.append(spec)
            forbidden.append(bbox)
            placed = True
            break
        if not placed:
            return None

    expected = _expected_from_specs(grid, legend, specs)
    solved = _solve(grid)
    if not np.array_equal(expected, solved):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
