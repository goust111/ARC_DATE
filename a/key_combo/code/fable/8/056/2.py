from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
Step = tuple[int, int]

DIRS: tuple[Step, ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def chebyshev(a: Coord, b: Coord) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def anchor_cells(center: Coord) -> list[Coord]:
    cr, cc = center
    return [(cr + dr, cc + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def normalize_delta(dr: int, dc: int) -> Step | None:
    if dr == 0 and dc == 0:
        return None
    if dr == 0:
        return (0, 1 if dc > 0 else -1)
    if dc == 0:
        return (1 if dr > 0 else -1, 0)
    if abs(dr) == abs(dc):
        return (1 if dr > 0 else -1, 1 if dc > 0 else -1)
    return None


def step_family(step: Step) -> str:
    dr, dc = step
    if dr == 0:
        return "h"
    if dc == 0:
        return "v"
    if dr == dc:
        return "d1"
    return "d2"


def max_distance_to_edge(center: Coord, step: Step, shape: tuple[int, int]) -> int:
    cr, cc = center
    dr, dc = step
    h, w = shape
    t = 0
    while in_bounds(shape, cr + (t + 1) * dr, cc + (t + 1) * dc):
        t += 1
    return t


def packed_cells(center: Coord, step: Step, count: int) -> list[Coord]:
    cr, cc = center
    dr, dc = step
    return [(cr + (i + 2) * dr, cc + (i + 2) * dc) for i in range(count)]


def full_anchor_centers(mask: np.ndarray) -> list[Coord]:
    h, w = mask.shape
    centers: list[Coord] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if bool(mask[r - 1 : r + 2, c - 1 : c + 2].all()):
                centers.append((r, c))
    return centers


def hidden_center_candidates(coords: np.ndarray, shape: tuple[int, int]) -> list[Coord]:
    h, w = shape
    candidates: list[Coord] = []
    for cr in range(1, h - 1):
        for cc in range(1, w - 1):
            families: set[str] = set()
            ok = True
            for raw_r, raw_c in coords:
                r = int(raw_r)
                c = int(raw_c)
                dr = r - cr
                dc = c - cc
                if max(abs(dr), abs(dc)) < 2:
                    ok = False
                    break
                step = normalize_delta(dr, dc)
                if step is None:
                    ok = False
                    break
                families.add(step_family(step))
            if ok and len(families) >= 2:
                candidates.append((cr, cc))
    return candidates


def parse_color(grid: np.ndarray, color: int) -> tuple[Coord, bool, dict[Step, list[int]]]:
    mask = grid == color
    coords = np.argwhere(mask)
    assert len(coords) > 0

    anchors = full_anchor_centers(mask)
    if anchors:
        assert len(anchors) == 1
        center = anchors[0]
        visible = True
        loose = [
            (int(r), int(c))
            for r, c in coords
            if max(abs(int(r) - center[0]), abs(int(c) - center[1])) >= 2
        ]
    else:
        candidates = hidden_center_candidates(coords, grid.shape)
        assert len(candidates) == 1
        center = candidates[0]
        visible = False
        loose = [(int(r), int(c)) for r, c in coords]

    rays: dict[Step, list[int]] = {}
    for r, c in loose:
        dr = r - center[0]
        dc = c - center[1]
        step = normalize_delta(dr, dc)
        assert step is not None
        dist = max(abs(dr), abs(dc))
        assert dist >= 2
        rays.setdefault(step, []).append(dist)

    families = {step_family(step) for step in rays}
    assert len(families) >= 2
    for step, dists in rays.items():
        assert len(dists) == len(set(dists))
        rays[step] = sorted(dists)

    return center, visible, rays


def main(grid: np.ndarray) -> np.ndarray:
    """Rebuild each color's 3x3 anchor from concurrent sightlines, then compress the ray pixels inward.

    Input structure:
    - A uniform background contains several foreground colors.
    - For each color, pixels lie on horizontal, vertical, or 45-degree diagonal rays around one color-specific center.
    - Some colors already include a solid 3x3 block centered there, while others show only loose ray pixels whose supporting lines uniquely concur at that missing center.

    Rule:
    - For a color with a visible 3x3 block, keep that block fixed.
    - For a color without one, infer the unique common center of its rays and draw a solid 3x3 block there.
    - In both cases, erase the old loose pixels and repack each directional ray into a consecutive segment that starts immediately outside the block along the same ray.

    Key insight:
    - Visible and invisible-anchor colors obey the same law: the sparse pixels encode ray directions and counts, and the missing block is just the unique concurrency point those rays reveal.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = np.full_like(g, bg)
    occupied: dict[Coord, int] = {}

    for color in colors_present(g, exclude={bg}):
        center, visible, rays = parse_color(g, color)
        cr, cc = center

        for cell in anchor_cells(center):
            assert in_bounds(g.shape, *cell)
            assert cell not in occupied
            occupied[cell] = color
            out[cell] = color

        loose_count = 0
        for step, dists in rays.items():
            loose_count += len(dists)
            for idx, _dist in enumerate(dists):
                cell = (cr + (idx + 2) * step[0], cc + (idx + 2) * step[1])
                assert in_bounds(g.shape, *cell)
                assert cell not in occupied
                occupied[cell] = color
                out[cell] = color

        in_count = int(np.sum(g == color))
        expected = loose_count + (9 if visible else 0)
        assert in_count == expected
        assert int(np.sum(out == color)) == loose_count + 9

    return out


PALETTE = list(range(10))
GRID_MIN = 15
GRID_MAX = 24
COLOR_COUNT_MIN = 2
COLOR_COUNT_MAX = 4
GEN_RETRIES = 400


def close_conflict(cells: set[Coord], occupied: set[Coord], gap: int = 1) -> bool:
    for a in cells:
        for b in occupied:
            if chebyshev(a, b) <= gap:
                return True
    return False


def build_grids(
    shape: tuple[int, int],
    bg: int,
    specs: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, bg, dtype=int)
    out = np.full(shape, bg, dtype=int)
    for spec in specs:
        color = spec["color"]
        center = spec["center"]
        if spec["visible"]:
            for r, c in anchor_cells(center):
                inp[r, c] = color
        for r, c in anchor_cells(center):
            out[r, c] = color
        for step, distances in spec["input_rays"].items():
            for dist in distances:
                r = center[0] + dist * step[0]
                c = center[1] + dist * step[1]
                inp[r, c] = color
        for step, count in spec["ray_counts"].items():
            for r, c in packed_cells(center, step, count):
                out[r, c] = color
    return inp, out


def choose_directions(center: Coord, shape: tuple[int, int]) -> dict[Step, int] | None:
    feasible = [step for step in DIRS if max_distance_to_edge(center, step, shape) >= 3]
    by_family: dict[str, list[Step]] = {}
    for step in feasible:
        by_family.setdefault(step_family(step), []).append(step)
    if len(by_family) < 2:
        return None

    family_count = random.randint(2, min(4, len(by_family)))
    chosen_families = random.sample(sorted(by_family), family_count)
    chosen_steps = [random.choice(by_family[family]) for family in chosen_families]

    remaining = [step for step in feasible if step not in chosen_steps]
    target_count = random.randint(family_count, min(6, len(feasible)))
    if target_count > family_count:
        chosen_steps.extend(random.sample(remaining, target_count - family_count))
    random.shuffle(chosen_steps)

    ray_counts: dict[Step, int] = {}
    for step in chosen_steps:
        max_count = min(3, max_distance_to_edge(center, step, shape) - 2)
        if max_count < 1:
            return None
        ray_counts[step] = random.randint(1, max_count)

    growable = [step for step, count in ray_counts.items() if count >= 2]
    if growable and random.random() < 0.6:
        step = random.choice(growable)
        ray_counts[step] = max(ray_counts[step], 2)
    return ray_counts


def choose_input_distances(
    center: Coord,
    ray_counts: dict[Step, int],
    shape: tuple[int, int],
) -> dict[Step, list[int]] | None:
    input_rays: dict[Step, list[int]] = {}
    moved_any = False
    for step, count in ray_counts.items():
        max_dist = max_distance_to_edge(center, step, shape)
        options = list(range(2, max_dist + 1))
        packed = list(range(2, count + 2))
        for _ in range(40):
            chosen = sorted(random.sample(options, count))
            if chosen != packed:
                moved_any = True
                input_rays[step] = chosen
                break
        else:
            return None
    if not moved_any:
        return None
    return input_rays


def sample_color_spec(
    shape: tuple[int, int],
    color: int,
    visible: bool,
    occupied_in_reserved: set[Coord],
    occupied_out: set[Coord],
) -> dict | None:
    h, w = shape
    for _ in range(120):
        center = (random.randint(1, h - 2), random.randint(1, w - 2))
        ray_counts = choose_directions(center, shape)
        if ray_counts is None:
            continue
        input_rays = choose_input_distances(center, ray_counts, shape)
        if input_rays is None:
            continue

        anchor = set(anchor_cells(center))
        out_cells = set(anchor)
        for step, count in ray_counts.items():
            out_cells.update(packed_cells(center, step, count))

        in_cells = set(anchor) if visible else set()
        for step, distances in input_rays.items():
            for dist in distances:
                in_cells.add((center[0] + dist * step[0], center[1] + dist * step[1]))

        if len(in_cells) != sum(ray_counts.values()) + (9 if visible else 0):
            continue
        if len(out_cells) != sum(ray_counts.values()) + 9:
            continue
        if close_conflict(anchor, occupied_in_reserved, gap=1):
            continue
        if close_conflict(in_cells, occupied_in_reserved, gap=1):
            continue
        if close_conflict(out_cells, occupied_out, gap=1):
            continue

        return {
            "color": color,
            "center": center,
            "visible": visible,
            "ray_counts": ray_counts,
            "input_rays": input_rays,
        }
    return None


def validate_instance(inp: np.ndarray, out: np.ndarray, specs: list[dict], bg: int) -> bool:
    if np.array_equal(inp, out):
        return False
    if get_bg(inp) != bg or get_bg(out) != bg:
        return False
    if not any(spec["visible"] for spec in specs):
        return False
    if not any(not spec["visible"] for spec in specs):
        return False

    for spec in specs:
        color = spec["color"]
        center = spec["center"]
        ray_counts = spec["ray_counts"]
        if len(ray_counts) < 2:
            return False
        if len({step_family(step) for step in ray_counts}) < 2:
            return False
        packed_signature = {step: list(range(2, count + 2)) for step, count in ray_counts.items()}
        if spec["input_rays"] == packed_signature:
            return False

        inp_mask = inp == color
        anchors = full_anchor_centers(inp_mask)
        if spec["visible"]:
            if anchors != [center]:
                return False
        else:
            if anchors:
                return False
            candidates = hidden_center_candidates(np.argwhere(inp_mask), inp.shape)
            if candidates != [center]:
                return False

        out_mask = out == color
        if full_anchor_centers(out_mask) != [center]:
            return False

    try:
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, out)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    bg = random.choice(PALETTE)
    fg_pool = [color for color in PALETTE if color != bg]
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(fg_pool, color_count)

    hidden_count = random.randint(1, color_count - 1)
    visibilities = [False] * hidden_count + [True] * (color_count - hidden_count)
    random.shuffle(visibilities)

    specs: list[dict] = []
    occupied_in_reserved: set[Coord] = set()
    occupied_out: set[Coord] = set()

    for color, visible in zip(colors, visibilities):
        spec = sample_color_spec((h, w), color, visible, occupied_in_reserved, occupied_out)
        if spec is None:
            return None
        specs.append(spec)
        occupied_in_reserved.update(anchor_cells(spec["center"]))
        for step, distances in spec["input_rays"].items():
            for dist in distances:
                occupied_in_reserved.add((spec["center"][0] + dist * step[0], spec["center"][1] + dist * step[1]))
        occupied_out.update(anchor_cells(spec["center"]))
        for step, count in spec["ray_counts"].items():
            occupied_out.update(packed_cells(spec["center"], step, count))

    inp, out = build_grids((h, w), bg, specs)
    if not validate_instance(inp, out, specs, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
