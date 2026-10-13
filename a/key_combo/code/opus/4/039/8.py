from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS) not in sys.path:
    sys.path.append(str(ARC_UTILS))

from arc_utils import Color, crop_bbox, find_components


BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
MAROON = Color.MAROON
DIRS = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}


def _marker_neighbors(marker: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = marker
    return {(r + dr, c + dc) for dr, dc in DIRS.values()}


def _between_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        return [(r0, c) for c in range(c0 + step, c1, step)]
    if c0 == c1:
        step = 1 if r1 > r0 else -1
        return [(r, c0) for r in range(r0 + step, r1, step)]
    return []


def _clear_straight_path(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    if a[0] != b[0] and a[1] != b[1]:
        return False
    return all(int(grid[r, c]) == BG for r, c in _between_cells(a, b))


def _find_markers(grid: np.ndarray) -> list[tuple[int, int]]:
    comps = find_components(
        grid,
        bg=BG,
        colors=[MAROON],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    markers = []
    for comp in comps:
        if int(comp["area"]) == 1:
            r, c = map(int, comp["coords"][0])
            markers.append((r, c))
    return sorted(markers)


def _unique_marker_for_pixel(
    grid: np.ndarray, pos: tuple[int, int], markers: list[tuple[int, int]]
) -> tuple[int, int]:
    matches = [marker for marker in markers if _clear_straight_path(grid, pos, marker)]
    assert len(matches) == 1, f"pixel {pos} must align with exactly one marker"
    return matches[0]


def _compress_pixels(grid: np.ndarray, markers: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    moves: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    dests: set[tuple[int, int]] = set()

    for r, c in map(tuple, np.argwhere(np.isin(out, [BLUE, RED]))):
        color = int(out[r, c])
        marker = _unique_marker_for_pixel(out, (int(r), int(c)), markers)
        dr = 0 if marker[0] == r else (1 if r > marker[0] else -1)
        dc = 0 if marker[1] == c else (1 if c > marker[1] else -1)
        dest = (marker[0] + dr, marker[1] + dc)
        assert dest not in dests, "compressed pixels may not collide"
        if dest != (r, c):
            assert int(out[dest]) == BG, "target adjacent cell must be empty"
        dests.add(dest)
        moves.append(((int(r), int(c)), dest, color))

    for src, _dest, _color in moves:
        out[src] = BG
    for _src, dest, color in moves:
        out[dest] = color
    return out


def _adjacent_counts(grid: np.ndarray, marker: tuple[int, int]) -> dict[int, int]:
    counts = {BLUE: 0, RED: 0}
    r, c = marker
    for dr, dc in DIRS.values():
        rr = r + dr
        cc = c + dc
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
            value = int(grid[rr, cc])
            if value in counts:
                counts[value] += 1
    return counts


def _container_from_component(grid: np.ndarray, comp: dict) -> dict:
    bbox = tuple(map(int, comp["bbox"]))
    r0, c0, r1, c1 = bbox
    sprite = crop_bbox(grid, bbox)
    h, w = sprite.shape
    assert h >= 3 and w >= 3, "containers must be at least 3x3"
    assert np.all(sprite[:, 0] == MAROON)
    assert np.all(sprite[:, -1] == MAROON)
    assert np.all(sprite[-1, :] == MAROON)
    assert np.all(sprite[:-1, 1:-1] == BG)
    width = w - 2
    cavity = (r0, c0 + 1, r1 - 1, c1 - 1)
    return {"bbox": bbox, "width": int(width), "cavity": cavity}


def _find_containers(grid: np.ndarray) -> list[dict]:
    comps = find_components(
        grid,
        bg=BG,
        colors=[MAROON],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    containers = []
    for comp in comps:
        if int(comp["area"]) > 1:
            containers.append(_container_from_component(grid, comp))
    containers.sort(key=lambda item: item["bbox"])
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress aligned colored pixels beside each marker, count them by color, then fill the width-matched U containers.

    Input structure:
    - Black background with two isolated maroon single-pixel markers in the upper half.
    - Blue and red pixels are scattered so each one shares exactly one clear row or column with one marker.
    - The lower half contains 3 or 4 maroon U-shaped containers with distinct inner widths.

    Rule:
    - Identify the two isolated maroon markers.
    - Slide every blue or red pixel straight toward its unique aligned marker until it reaches the adjacent cell beside that marker.
    - Count the blue and red pixels in the four orthogonal neighbor cells around each marker.
    - Find the U containers below, measure each inner width, and fill the container whose width equals a marker's blue count with blue and the container whose width equals its red count with red.
    - Keep the markers, the compressed counting pixels, the maroon container walls, and any unmatched container unchanged.

    Key insight:
    - The scattered pixels do not move into the containers; they only encode counts, and those counts alone select which container cavities get filled.
    """
    g = np.asarray(grid, dtype=int)
    markers = _find_markers(g)
    assert len(markers) == 2, "expected exactly two isolated markers"

    compressed = _compress_pixels(g, markers)
    counts_by_marker = {marker: _adjacent_counts(compressed, marker) for marker in markers}

    containers = _find_containers(g)
    assert 3 <= len(containers) <= 4, "expected 3 or 4 U containers"
    width_to_container = {container["width"]: container for container in containers}
    assert len(width_to_container) == len(containers), "container widths must be distinct"

    out = compressed.copy()
    for counts in counts_by_marker.values():
        blue_width = counts[BLUE]
        red_width = counts[RED]
        assert blue_width > 0 and red_width > 0
        blue_box = width_to_container[blue_width]["cavity"]
        red_box = width_to_container[red_width]["cavity"]
        br0, bc0, br1, bc1 = blue_box
        rr0, rc0, rr1, rc1 = red_box
        out[br0 : br1 + 1, bc0 : bc1 + 1] = BLUE
        out[rr0 : rr1 + 1, rc0 : rc1 + 1] = RED
    return out


def _make_container(inner_width: int, height: int) -> np.ndarray:
    patch = np.zeros((height, inner_width + 2), dtype=int)
    patch[:, 0] = MAROON
    patch[:, -1] = MAROON
    patch[-1, :] = MAROON
    return patch


def _pattern_counts() -> list[tuple[int, int]]:
    if random.choice([True, False]):
        reds = random.sample([2, 3], 2)
        return [(1, reds[0]), (1, reds[1])]
    blues = random.sample([2, 3], 2)
    return [(blues[0], 1), (blues[1], 1)]


def _max_distance(top_h: int, shape: tuple[int, int], marker: tuple[int, int], direction: str) -> int:
    r, c = marker
    h, w = shape
    if direction == "N":
        return r
    if direction == "S":
        return top_h - 1 - r
    if direction == "W":
        return c
    if direction == "E":
        return w - 1 - c
    raise ValueError(direction)


def _direction_assignment(blue_count: int, red_count: int) -> dict[str, int]:
    total = blue_count + red_count
    used_dirs = list(DIRS)
    if total == 3:
        used_dirs.remove(random.choice(used_dirs))
    random.shuffle(used_dirs)
    blue_dirs = set(random.sample(used_dirs, blue_count))
    return {direction: (BLUE if direction in blue_dirs else RED) for direction in used_dirs}


def _place_sources(
    grid: np.ndarray,
    top_h: int,
    markers: list[tuple[int, int]],
    counts: list[tuple[int, int]],
) -> list[dict] | None:
    sources: list[dict] = []
    occupied = {marker for marker in markers}

    for idx, marker in enumerate(markers):
        assignment = _direction_assignment(*counts[idx])
        other = markers[1 - idx]
        for direction, color in assignment.items():
            max_d = _max_distance(top_h, grid.shape, marker, direction)
            if max_d < 2:
                return None
            dr, dc = DIRS[direction]
            options = []
            for dist in range(2, max_d + 1):
                rr = marker[0] + dr * dist
                cc = marker[1] + dc * dist
                if direction in ("E", "W") and cc == other[1]:
                    continue
                if direction in ("N", "S") and rr == other[0]:
                    continue
                if (rr, cc) in occupied:
                    continue
                options.append((rr, cc))
            if not options:
                return None
            rr, cc = random.choice(options)
            occupied.add((rr, cc))
            grid[rr, cc] = color
            sources.append(
                {
                    "src": (rr, cc),
                    "marker": marker,
                    "direction": direction,
                    "color": color,
                }
            )
    return sources


def _expected_from_meta(grid: np.ndarray, meta: dict) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for item in meta["sources"]:
        out[item["src"]] = BG
        dr, dc = DIRS[item["direction"]]
        mr, mc = item["marker"]
        out[mr + dr, mc + dc] = item["color"]
    for width, color in meta["fills"].items():
        r0, c0, r1, c1 = meta["cavities"][width]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _marker_counts_to_fill_map(counts: list[tuple[int, int]]) -> dict[int, int]:
    fills: dict[int, int] = {}
    for blue_count, red_count in counts:
        if blue_count in fills:
            assert fills[blue_count] == BLUE, "a width may not map to two colors"
        fills[blue_count] = BLUE
        if red_count in fills:
            assert fills[red_count] == RED, "a width may not map to two colors"
        fills[red_count] = RED
    return fills


def _validate_scene(grid: np.ndarray, meta: dict) -> bool:
    markers = _find_markers(grid)
    if markers != sorted(meta["markers"]):
        return False

    containers = _find_containers(grid)
    widths = sorted(container["width"] for container in containers)
    if widths != meta["widths"]:
        return False

    spans = []
    cavities = {}
    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        spans.append((c0, c1))
        cavities[container["width"]] = container["cavity"]
    if cavities != meta["cavities"]:
        return False

    for _r, c in markers:
        if any(c0 <= c <= c1 for c0, c1 in spans):
            return False

    compressed = _compress_pixels(grid, markers)
    observed = [_adjacent_counts(compressed, marker) for marker in markers]
    expected_counts = [
        {BLUE: blue_count, RED: red_count}
        for blue_count, red_count in meta["counts_by_marker"]
    ]
    if observed != expected_counts:
        return False

    for counts in observed:
        total = counts[BLUE] + counts[RED]
        if counts[BLUE] == 0 or counts[RED] == 0 or total not in (3, 4):
            return False

    expected = _expected_from_meta(grid, meta)
    if not np.array_equal(main(grid), expected):
        return False

    if len(meta["widths"]) == 4:
        decoy = 4
        r0, c0, r1, c1 = meta["cavities"][decoy]
        if np.any(expected[r0 : r1 + 1, c0 : c1 + 1] != BG):
            return False

    for item in meta["sources"]:
        marker = _unique_marker_for_pixel(grid, item["src"], markers)
        if marker != item["marker"]:
            return False

    return True


def _generate_attempt() -> tuple[np.ndarray, dict] | None:
    top_h = random.randint(8, 10)
    use_decoy = random.choice([True, False])
    widths = [1, 2, 3] + ([4] if use_decoy else [])
    random.shuffle(widths)

    min_w = 26 if use_decoy else 18
    width = random.randint(min_w, 29)
    height = random.randint(top_h + 7, 26)
    max_container_h = min(6, height - top_h - 2)
    if max_container_h < 3:
        return None

    outer_widths = [inner + 2 for inner in widths]
    gaps = [random.randint(1, 3) for _ in range(len(widths) - 1)]
    total_span = sum(outer_widths) + sum(gaps)
    if total_span > width - 2:
        return None

    grid = np.zeros((height, width), dtype=int)
    left_margin = random.randint(1, width - total_span - 1)
    col = left_margin
    cavities: dict[int, tuple[int, int, int, int]] = {}

    for idx, inner_width in enumerate(widths):
        cont_h = random.randint(3, max_container_h)
        patch = _make_container(inner_width, cont_h)
        r0 = height - cont_h
        c0 = col
        grid[r0 : r0 + cont_h, c0 : c0 + inner_width + 2] = patch
        cavities[inner_width] = (r0, c0 + 1, height - 2, c0 + inner_width)
        col += inner_width + 2
        if idx < len(gaps):
            col += gaps[idx]

    blocked_cols = set()
    for inner_width in widths:
        _r0, c0, _r1, c1 = (
            cavities[inner_width][0],
            cavities[inner_width][1] - 1,
            height - 1,
            cavities[inner_width][3] + 1,
        )
        blocked_cols.update(range(c0, c1 + 1))

    safe_cols = [c for c in range(2, width - 2) if c not in blocked_cols]
    if len(safe_cols) < 2:
        return None

    row_options = list(range(2, top_h - 2))
    if len(row_options) < 2:
        return None
    random.shuffle(row_options)

    marker_rows = None
    for r0 in row_options:
        choices = [r1 for r1 in row_options if r1 != r0]
        if choices:
            marker_rows = [r0, random.choice(choices)]
            break
    if marker_rows is None:
        return None

    markers = None
    for _ in range(30):
        marker_cols = random.sample(safe_cols, 2)
        if marker_rows[0] == marker_rows[1] or marker_cols[0] == marker_cols[1]:
            continue
        candidate = sorted([(marker_rows[0], marker_cols[0]), (marker_rows[1], marker_cols[1])])
        if _marker_neighbors(candidate[0]).isdisjoint(_marker_neighbors(candidate[1])):
            markers = candidate
            break
    if markers is None:
        return None

    counts = _pattern_counts()

    for marker in markers:
        grid[marker] = MAROON

    sources = _place_sources(grid, top_h, markers, counts)
    if sources is None:
        return None

    fills = _marker_counts_to_fill_map(counts)
    meta = {
        "markers": markers,
        "counts_by_marker": counts,
        "widths": sorted(widths),
        "cavities": cavities,
        "fills": fills,
        "sources": sources,
    }
    if not _validate_scene(grid, meta):
        return None
    return grid, meta


def generate_input() -> np.ndarray:
    for _ in range(400):
        attempt = _generate_attempt()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("failed to generate a valid ARC scene")
