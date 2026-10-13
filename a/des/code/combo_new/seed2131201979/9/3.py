from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    assert counts, "grid must be nonempty"
    return int(counts.most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _rank_palette(bg: int, n: int) -> list[int]:
    preferred = [2, 1, 3, 4, 6, 8, 9, 7, 0, 5]
    usable = [color for color in preferred if int(color) != int(bg)]
    assert usable
    return [int(usable[i % len(usable)]) for i in range(int(n))]


def _pair_and_connect(grid: np.ndarray, bg: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    counts = Counter(int(v) for v in g.ravel() if int(v) != int(bg))
    port_colors = [color for color, count in counts.items() if int(count) == 2]
    pipe_colors = [color for color, count in counts.items() if int(count) != 2]

    assert port_colors, "expected at least one port color"
    assert len(pipe_colors) == 1, f"expected exactly one pipe color, got {pipe_colors}"

    out = g.copy()
    for color in sorted(port_colors):
        coords = np.argwhere(g == int(color))
        assert coords.shape == (2, 2), f"port color {color} must appear exactly twice"
        (r0, c0), (r1, c1) = [tuple(map(int, rc)) for rc in coords]

        if r0 == r1:
            c_lo, c_hi = sorted((c0, c1))
            if c_hi - c_lo > 1:
                original_mid = g[r0, c_lo + 1 : c_hi]
                current_mid = out[r0, c_lo + 1 : c_hi]
                assert np.all(original_mid == bg), "horizontal port path is blocked by a pipe"
                assert np.all(current_mid == bg), "horizontal port path collides with another connection"
            out[r0, c_lo : c_hi + 1] = int(color)
        elif c0 == c1:
            r_lo, r_hi = sorted((r0, r1))
            if r_hi - r_lo > 1:
                original_mid = g[r_lo + 1 : r_hi, c0]
                current_mid = out[r_lo + 1 : r_hi, c0]
                assert np.all(original_mid == bg), "vertical port path is blocked by a pipe"
                assert np.all(current_mid == bg), "vertical port path collides with another connection"
            out[r_lo : r_hi + 1, c0] = int(color)
        else:
            raise AssertionError(f"port pair {color} is neither row-aligned nor col-aligned")

    return out


def _background_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != int(bg):
                continue

            queue: deque[tuple[int, int]] = deque([(int(r), int(c))])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_edge = False

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True

                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == int(bg):
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            components.append({"coords": coords, "touches_edge": touches_edge})

    return components


def _non_bg_components(grid: np.ndarray, bg: int) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    h, w = map(int, grid.shape)
    labels = np.zeros((h, w), dtype=int)
    components: dict[int, list[tuple[int, int]]] = {}
    next_id = 1

    for r in range(h):
        for c in range(w):
            if labels[r, c] != 0 or int(grid[r, c]) == int(bg):
                continue

            queue: deque[tuple[int, int]] = deque([(int(r), int(c))])
            labels[r, c] = next_id
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == 0 and int(grid[nr, nc]) != int(bg):
                        labels[nr, nc] = next_id
                        queue.append((nr, nc))

            components[next_id] = coords
            next_id += 1

    return labels, components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect color-matched pipe ports, then recolor each completed loop by enclosed-area rank.

    Input structure:
    - The grid has a dominant background color plus one grey pipe color and several port colors.
    - Grey straight segments are horizontal or vertical and have their two endpoint pixels replaced by colored ports.
    - Each port color appears exactly twice, and the two matching ports lie on the same row or the same column.
    - The straight background path between a matching port pair is clear of other pipe material.

    Rule:
    - Connect each matching port pair with a straight line of that port color along their shared row or column.
    - Treat the completed drawing as barriers and find every 4-connected background region that does not touch the grid edge.
    - Each such enclosed background region corresponds to one closed loop; its area is the number of background pixels in that region.
    - Sort loops by enclosed area from small to large and recolor every pixel on each loop to red, then blue, then green, and so on.

    Key insight:
    - The colored endpoints are wiring instructions rather than final colors: only after drawing those missing connections do the true closed loops and their area-based recoloring order become visible.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    connected = _pair_and_connect(g, bg)
    h, w = map(int, connected.shape)
    comp_labels, comp_coords = _non_bg_components(connected, bg)

    regions: list[dict[str, object]] = []
    for component in _background_components(connected, bg):
        if bool(component["touches_edge"]):
            continue

        loop_ids: set[int] = set()
        coords = component["coords"]
        for r, c in coords:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and int(connected[nr, nc]) != int(bg):
                    loop_ids.add(int(comp_labels[nr, nc]))

        assert loop_ids, "enclosed region must touch a non-background loop"
        assert len(loop_ids) == 1, f"expected one loop per enclosed region, got {loop_ids}"
        loop_id = next(iter(loop_ids))
        anchor = min(coords)
        regions.append(
            {
                "area": int(len(coords)),
                "anchor": tuple(map(int, anchor)),
                "loop_coords": comp_coords[loop_id],
            }
        )

    assert regions, "expected at least one closed loop after connecting ports"

    out = connected.copy()
    ordered = sorted(regions, key=lambda item: (int(item["area"]), item["anchor"]))
    for color, region in zip(_rank_palette(bg, len(ordered)), ordered):
        for r, c in region["loop_coords"]:
            out[int(r), int(c)] = int(color)

    return out


# --------------- generator ---------------

BG = 0
PIPE = 5
PORT_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_SIDE = 30
LOOP_COUNT_OPTIONS = (1, 2, 3)
ORIENTATION_OPTIONS = ("vertical", "horizontal")
LOOP_HEIGHT_RANGE = (4, 10)
LOOP_WIDTH_RANGE = (4, 10)
PLACEMENT_PADDING = 2
CROP_MARGIN_MAX = 4
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200


def _loop_area(spec: dict[str, int | str]) -> int:
    return int((int(spec["h"]) - 2) * (int(spec["w"]) - 2))


def _sample_specs(loop_count: int) -> list[dict[str, int | str]] | None:
    port_colors = random.sample(PORT_POOL, 2 * int(loop_count))
    specs: list[dict[str, int | str]] = []
    seen_areas: set[int] = set()

    for i in range(int(loop_count)):
        placed = False
        for _ in range(80):
            orientation = random.choice(ORIENTATION_OPTIONS)
            h = random.randint(*LOOP_HEIGHT_RANGE)
            w = random.randint(*LOOP_WIDTH_RANGE)
            area = (h - 2) * (w - 2)
            if area in seen_areas:
                continue

            spec = {
                "orientation": orientation,
                "h": int(h),
                "w": int(w),
                "color_a": int(port_colors[2 * i]),
                "color_b": int(port_colors[2 * i + 1]),
            }
            specs.append(spec)
            seen_areas.add(int(area))
            placed = True
            break
        if not placed:
            return None

    return specs


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, int(r0) - PLACEMENT_PADDING)
    cc0 = max(0, int(c0) - PLACEMENT_PADDING)
    rr1 = min(MAX_SIDE, int(r0) + int(h) + PLACEMENT_PADDING)
    cc1 = min(MAX_SIDE, int(c0) + int(w) + PLACEMENT_PADDING)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, int(r0) - PLACEMENT_PADDING)
    cc0 = max(0, int(c0) - PLACEMENT_PADDING)
    rr1 = min(MAX_SIDE, int(r0) + int(h) + PLACEMENT_PADDING)
    cc1 = min(MAX_SIDE, int(c0) + int(w) + PLACEMENT_PADDING)
    occupied[rr0:rr1, cc0:cc1] = True


def _place_specs(specs: list[dict[str, int | str]]) -> list[dict[str, int | str]] | None:
    occupied = np.zeros((MAX_SIDE, MAX_SIDE), dtype=bool)
    order = sorted(range(len(specs)), key=lambda i: int(specs[i]["h"]) * int(specs[i]["w"]), reverse=True)

    for idx in order:
        spec = specs[idx]
        h = int(spec["h"])
        w = int(spec["w"])
        candidates = [(r0, c0) for r0 in range(MAX_SIDE - h + 1) for c0 in range(MAX_SIDE - w + 1)]
        random.shuffle(candidates)

        placed = False
        for r0, c0 in candidates[:PLACEMENT_ATTEMPTS]:
            if not _can_place(occupied, r0, c0, h, w):
                continue
            spec["r0"] = int(r0)
            spec["c0"] = int(c0)
            _mark_occupied(occupied, r0, c0, h, w)
            placed = True
            break

        if not placed:
            return None

    return specs


def _render_input(specs: list[dict[str, int | str]]) -> np.ndarray:
    grid = np.full((MAX_SIDE, MAX_SIDE), BG, dtype=int)

    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        h = int(spec["h"])
        w = int(spec["w"])
        color_a = int(spec["color_a"])
        color_b = int(spec["color_b"])

        if spec["orientation"] == "vertical":
            grid[r0 + 1 : r0 + h - 1, c0] = PIPE
            grid[r0 + 1 : r0 + h - 1, c0 + w - 1] = PIPE
            grid[r0, c0] = color_a
            grid[r0, c0 + w - 1] = color_a
            grid[r0 + h - 1, c0] = color_b
            grid[r0 + h - 1, c0 + w - 1] = color_b
        else:
            grid[r0, c0 + 1 : c0 + w - 1] = PIPE
            grid[r0 + h - 1, c0 + 1 : c0 + w - 1] = PIPE
            grid[r0, c0] = color_a
            grid[r0 + h - 1, c0] = color_a
            grid[r0, c0 + w - 1] = color_b
            grid[r0 + h - 1, c0 + w - 1] = color_b

    return grid


def _render_expected(specs: list[dict[str, int | str]]) -> np.ndarray:
    connected = _render_input(specs)
    out = connected.copy()

    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        h = int(spec["h"])
        w = int(spec["w"])
        color_a = int(spec["color_a"])
        color_b = int(spec["color_b"])

        if spec["orientation"] == "vertical":
            connected[r0, c0 : c0 + w] = color_a
            connected[r0 + h - 1, c0 : c0 + w] = color_b
        else:
            connected[r0 : r0 + h, c0] = color_a
            connected[r0 : r0 + h, c0 + w - 1] = color_b

    out[:, :] = connected
    ordered = sorted(specs, key=lambda spec: (_loop_area(spec), int(spec["r0"]), int(spec["c0"])))
    for color, spec in zip(_rank_palette(BG, len(ordered)), ordered):
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        h = int(spec["h"])
        w = int(spec["w"])
        out[r0, c0 : c0 + w] = int(color)
        out[r0 + h - 1, c0 : c0 + w] = int(color)
        out[r0 : r0 + h, c0] = int(color)
        out[r0 : r0 + h, c0 + w - 1] = int(color)

    return out


def _non_bg_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != BG)
    assert coords.size > 0
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = _non_bg_bbox(inp)
    top = random.randint(0, min(CROP_MARGIN_MAX, r0))
    left = random.randint(0, min(CROP_MARGIN_MAX, c0))
    bottom = random.randint(0, min(CROP_MARGIN_MAX, MAX_SIDE - 1 - r1))
    right = random.randint(0, min(CROP_MARGIN_MAX, MAX_SIDE - 1 - c1))

    rr0 = r0 - top
    cc0 = c0 - left
    rr1 = r1 + bottom
    cc1 = c1 + right
    return inp[rr0 : rr1 + 1, cc0 : cc1 + 1], out[rr0 : rr1 + 1, cc0 : cc1 + 1]


def _gen_attempt() -> np.ndarray | None:
    loop_count = random.choice(LOOP_COUNT_OPTIONS)
    specs = _sample_specs(loop_count)
    if specs is None:
        return None

    specs = _place_specs(specs)
    if specs is None:
        return None

    inp = _render_input(specs)
    expected = _render_expected(specs)
    inp, expected = _crop_pair(inp, expected)

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
