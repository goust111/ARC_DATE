from __future__ import annotations

from collections import Counter

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
SIGNAL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
RNG = np.random.default_rng()

SIDE_ORDER = ("top", "right", "bottom", "left")


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _find_color_components(grid: np.ndarray, bg: int) -> list[dict]:
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    comps: list[dict] = []

    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if visited[r, c] or color == int(bg):
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )

    return comps


def _border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _is_frame_component(comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 5 or width < 5:
        return False
    coords = {tuple(map(int, rc)) for rc in comp["coords"]}
    return coords == _border_cells((r0, c0, r1, c1))


def _extract_frames(grid: np.ndarray, bg: int | None = None) -> list[dict]:
    if bg is None:
        bg = _most_common_color(grid)
    frames = [
        comp
        for comp in _find_color_components(grid, int(bg))
        if _is_frame_component(comp)
    ]
    frames.sort(key=lambda comp: tuple(map(int, comp["bbox"])))
    return frames


def _side_signal_colors(grid: np.ndarray, frame: dict, bg: int) -> dict[str, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    frame_color = int(frame["color"])
    height, width = grid.shape

    side_cells = {
        "top": [(r0 - 1, c) for c in range(c0, c1 + 1)] if r0 > 0 else [],
        "right": [(r, c1 + 1) for r in range(r0, r1 + 1)] if c1 + 1 < width else [],
        "bottom": [(r1 + 1, c) for c in range(c0, c1 + 1)] if r1 + 1 < height else [],
        "left": [(r, c0 - 1) for r in range(r0, r1 + 1)] if c0 > 0 else [],
    }

    result: dict[str, int] = {}
    for side in SIDE_ORDER:
        colors = [
            int(grid[r, c])
            for r, c in side_cells[side]
            if int(grid[r, c]) not in (int(bg), frame_color)
        ]
        if not colors:
            continue
        result[side] = int(Counter(colors).most_common(1)[0][0])
    return result


def _paint_arm(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    center: tuple[int, int],
    side: str,
    color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    cr, cc = center

    if side == "top":
        if r0 + 1 <= cr - 1:
            out[r0 + 1 : cr, cc] = int(color)
    elif side == "right":
        if cc + 1 <= c1 - 1:
            out[cr, cc + 1 : c1] = int(color)
    elif side == "bottom":
        if cr + 1 <= r1 - 1:
            out[cr + 1 : r1, cc] = int(color)
    elif side == "left":
        if c0 + 1 <= cc - 1:
            out[cr, c0 + 1 : cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inside each rectangular frame, reflect external side markers as inward arms.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _most_common_color(g)

    for frame in _extract_frames(g, bg):
        side_colors = _side_signal_colors(g, frame, bg)
        if not side_colors:
            continue

        r0, c0, r1, c1 = map(int, frame["bbox"])
        center = ((r0 + r1) // 2, (c0 + c1) // 2)

        for side in SIDE_ORDER:
            if side in side_colors:
                _paint_arm(out, (r0, c0, r1, c1), center, side, side_colors[side])

        unique_colors = set(side_colors.values())
        # Keep the shared center deterministic without arbitrarily preferring
        # one side when mixed colors meet.
        if len(unique_colors) == 1:
            out[center[0], center[1]] = int(next(iter(unique_colors)))

    return out


def _randint(lo: int, hi: int) -> int:
    return int(RNG.integers(lo, hi + 1))


def _pick_spaced_positions(candidates: list[int], count: int) -> list[int] | None:
    if count <= 0:
        return []

    order = list(candidates)
    RNG.shuffle(order)
    chosen: list[int] = []
    for pos in order:
        if all(abs(pos - prev) >= 2 for prev in chosen):
            chosen.append(int(pos))
            if len(chosen) == count:
                break

    if len(chosen) != count:
        return None
    return sorted(chosen)


def _frame_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    return sorted(_border_cells(bbox))


def _tentacle_cells_for_side(
    bbox: tuple[int, int, int, int], side: str, positions: list[int]
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0 - 1, c) for c in positions]
    if side == "right":
        return [(r, c1 + 1) for r in positions]
    if side == "bottom":
        return [(r1 + 1, c) for c in positions]
    return [(r, c0 - 1) for r in positions]


def _gen_attempt() -> np.ndarray | None:
    height = _randint(17, 28)
    width = _randint(17, 28)
    grid = np.full((height, width), BG_COLOR, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    area = height * width
    if area < 360:
        max_frames = 2
    elif area < 520:
        max_frames = 3
    else:
        max_frames = 4
    n_frames = _randint(1, max_frames)

    if n_frames >= 3:
        dim_choices = (5, 7, 9)
    elif n_frames == 2:
        dim_choices = (5, 7, 9, 11)
    else:
        dim_choices = (5, 7, 9, 11, 13)

    specs: list[dict] = []
    for _ in range(n_frames):
        placed = False
        for _trial in range(120):
            outer_h = int(RNG.choice(dim_choices))
            outer_w = int(RNG.choice(dim_choices))
            if outer_h + 2 > height or outer_w + 2 > width:
                continue

            r0 = _randint(1, height - outer_h - 1)
            c0 = _randint(1, width - outer_w - 1)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            rr0 = r0 - 1
            rr1 = r1 + 1
            cc0 = c0 - 1
            cc1 = c1 + 1
            if reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue

            side_mask = _randint(1, 15)
            active_sides = [
                side for bit, side in enumerate(SIDE_ORDER) if side_mask & (1 << bit)
            ]
            color = int(RNG.choice(SIGNAL_COLORS))
            tentacles: dict[str, list[tuple[int, int]]] = {}
            ok = True

            for side in active_sides:
                if side in ("top", "bottom"):
                    candidates = list(range(c0 + 1, c1))
                else:
                    candidates = list(range(r0 + 1, r1))
                if not candidates:
                    ok = False
                    break

                max_count = min(3, (len(candidates) + 1) // 2)
                count = _randint(1, max_count)
                picked = _pick_spaced_positions(candidates, count)
                if picked is None:
                    ok = False
                    break
                tentacles[side] = _tentacle_cells_for_side((r0, c0, r1, c1), side, picked)

            if not ok:
                continue

            for r, c in _frame_cells((r0, c0, r1, c1)):
                grid[r, c] = FRAME_COLOR
            for side in active_sides:
                for r, c in tentacles[side]:
                    grid[r, c] = color

            reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            specs.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "color": color,
                    "active_sides": tuple(active_sides),
                }
            )
            placed = True
            break

        if not placed:
            return None

    vals, counts = np.unique(grid, return_counts=True)
    color_counts = {int(v): int(c) for v, c in zip(vals, counts)}
    if color_counts.get(BG_COLOR, 0) <= max(
        color_counts.get(FRAME_COLOR, 0),
        max((color_counts.get(color, 0) for color in SIGNAL_COLORS), default=0),
    ):
        return None

    frames = _extract_frames(grid, BG_COLOR)
    if len(frames) != len(specs):
        return None

    spec_by_bbox = {tuple(spec["bbox"]): spec for spec in specs}
    multi_direction = 0
    for frame in frames:
        bbox = tuple(map(int, frame["bbox"]))
        spec = spec_by_bbox.get(bbox)
        if spec is None:
            return None
        side_colors = _side_signal_colors(grid, frame, BG_COLOR)
        expected = {side: int(spec["color"]) for side in spec["active_sides"]}
        if side_colors != expected:
            return None
        if len(side_colors) >= 2:
            multi_direction += 1

    if n_frames >= 2 and multi_direction == 0:
        return None

    if np.array_equal(main(grid), grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
