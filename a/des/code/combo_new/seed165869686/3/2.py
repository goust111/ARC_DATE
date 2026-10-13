from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append({"color": color, "coords": coords, "area": len(coords)})
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Teleport each monochrome object from its touched portal pixel to the matching portal pixel.

    Input structure:
    - The grid has one dominant background color.
    - Non-background content consists of small single-color 4-connected objects and several portal pairs.
    - Each portal pair is exactly two isolated same-color pixels, and portal colors never appear on the objects.
    - Every object is 4-neighbor-adjacent to exactly one portal pixel.

    Rule:
    - Identify portal colors as the colors that form exactly two singleton components.
    - For each object, locate the touched portal pixel P1 and the same-color partner portal P2.
    - Copy every object cell to the position with the same offset from P2 that it had from P1.
    - Clear the original objects and remove all portal pixels; any teleported cells outside the grid are clipped away.

    Key insight:
    - The paired singleton pixels are functional markers rather than payload objects, and they transfer the whole touched object by preserving its portal-relative offsets.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)

    comps = _find_components(g, bg)
    by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(comp)

    portal_pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    portal_lookup: dict[tuple[int, int], int] = {}
    objects: list[dict[str, object]] = []
    for color, parts in by_color.items():
        if len(parts) == 2 and all(int(part["area"]) == 1 for part in parts):
            a = tuple(parts[0]["coords"][0])
            b = tuple(parts[1]["coords"][0])
            portal_pairs[int(color)] = (a, b)
            portal_lookup[a] = int(color)
            portal_lookup[b] = int(color)
        else:
            objects.extend(parts)

    out = np.full_like(g, bg)
    used_portal_colors: set[int] = set()
    portal_coords = set(portal_lookup)
    for comp in objects:
        touched = set()
        for r, c in comp["coords"]:
            for nr, nc in _neighbors4(int(r), int(c), h, w):
                if (nr, nc) in portal_coords:
                    touched.add((nr, nc))
        assert len(touched) == 1
        p1 = next(iter(touched))
        portal_color = portal_lookup[p1]
        used_portal_colors.add(portal_color)
        pa, pb = portal_pairs[portal_color]
        p2 = pb if p1 == pa else pa
        obj_color = int(comp["color"])
        for r, c in comp["coords"]:
            rr = int(p2[0] + (int(r) - p1[0]))
            cc = int(p2[1] + (int(c) - p1[1]))
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = obj_color

    assert used_portal_colors == set(portal_pairs)
    return out


# --------------- generator ---------------

GRID_MIN = {1: 10, 2: 12, 3: 15, 4: 18}
GRID_MAX = {1: 18, 2: 22, 3: 26, 4: 30}
MAX_ATTEMPTS = 500


def _shape_bbox(cells: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _generate_shape(side: str, size: int) -> tuple[tuple[int, int], ...] | None:
    seed = SIDES[side]
    for _ in range(200):
        cells = {seed}
        while len(cells) < size:
            candidates = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) == (0, 0) or (nr, nc) in cells:
                        continue
                    if abs(nr) + abs(nc) == 1 and (nr, nc) != seed:
                        continue
                    trial = set(cells)
                    trial.add((nr, nc))
                    bh, bw = _shape_bbox(trial)
                    if bh > 5 or bw > 5:
                        continue
                    candidates.add((nr, nc))
            if not candidates:
                break
            cells.add(random.choice(tuple(candidates)))
        if len(cells) != size:
            continue
        if sum(abs(r) + abs(c) == 1 for r, c in cells) != 1:
            continue
        return tuple(sorted(cells))
    return None


def _all_inside(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _can_place_with_padding(
    cells: list[tuple[int, int]], occupied: np.ndarray, h: int, w: int, pad: int = 1
) -> bool:
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        if bool(occupied[r0:r1, c0:c1].any()):
            return False
    return True


def _mark_cells(cells: list[tuple[int, int]], occupied: np.ndarray) -> None:
    for r, c in cells:
        occupied[r, c] = True


def _choose_exit_portal(
    offsets: tuple[tuple[int, int], ...],
    h: int,
    w: int,
    output_occupied: np.ndarray,
    portal_occupied: np.ndarray,
    must_clip: bool,
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    candidates = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(candidates)
    total = len(offsets)
    for p2 in candidates:
        if not _can_place_with_padding([p2], portal_occupied, h, w, pad=1):
            continue
        translated = [(p2[0] + dr, p2[1] + dc) for dr, dc in offsets]
        visible = [(r, c) for r, c in translated if 0 <= r < h and 0 <= c < w]
        if must_clip:
            if not (2 <= len(visible) < total):
                continue
        else:
            if len(visible) != total:
                continue
        if not _can_place_with_padding(visible, output_occupied, h, w, pad=1):
            continue
        return p2, visible
    return None


def _choose_entry_portal(
    offsets: tuple[tuple[int, int], ...], h: int, w: int, occupied: np.ndarray
) -> tuple[int, int] | None:
    candidates = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(candidates)
    for p1 in candidates:
        cluster = [p1] + [(p1[0] + dr, p1[1] + dc) for dr, dc in offsets]
        if not _all_inside(cluster, h, w):
            continue
        if not _can_place_with_padding(cluster, occupied, h, w, pad=1):
            continue
        return p1
    return None


def _expected_output(
    shape_specs: list[dict[str, object]], h: int, w: int, bg: int
) -> np.ndarray:
    out = np.full((h, w), bg, dtype=int)
    for spec in shape_specs:
        color = int(spec["object_color"])
        p2 = spec["p2"]
        for dr, dc in spec["offsets"]:
            rr = int(p2[0] + dr)
            cc = int(p2[1] + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = color
    return out


def _structure_ok(grid: np.ndarray) -> bool:
    bg = _get_bg(grid)
    comps = _find_components(grid, bg)
    by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(comp)

    portal_lookup: dict[tuple[int, int], int] = {}
    portal_colors = set()
    objects = []
    for color, parts in by_color.items():
        if len(parts) == 2 and all(int(part["area"]) == 1 for part in parts):
            portal_colors.add(int(color))
            for part in parts:
                portal_lookup[tuple(part["coords"][0])] = int(color)
        else:
            for part in parts:
                if not (3 <= int(part["area"]) <= 6):
                    return False
                objects.append(part)

    if not portal_colors or len(objects) != len(portal_colors):
        return False

    portal_coords = set(portal_lookup)
    used = set()
    for comp in objects:
        touched = set()
        for r, c in comp["coords"]:
            for nr, nc in _neighbors4(int(r), int(c), grid.shape[0], grid.shape[1]):
                if (nr, nc) in portal_coords:
                    touched.add((nr, nc))
        if len(touched) != 1:
            return False
        used.add(portal_lookup[next(iter(touched))])

    counts = {int(v): int((grid == v).sum()) for v in np.unique(grid)}
    return counts[bg] == max(counts.values()) and used == portal_colors


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(1, 4)
    h = random.randint(GRID_MIN[object_count], GRID_MAX[object_count])
    w = random.randint(GRID_MIN[object_count], GRID_MAX[object_count])

    bg = random.randrange(10)
    usable = [c for c in range(10) if c != bg]
    colors = random.sample(usable, object_count * 2)
    object_colors = colors[:object_count]
    portal_colors = colors[object_count:]

    force_clip = random.random() < 0.5
    clip_indices = set(random.sample(range(object_count), random.randint(1, object_count))) if force_clip else set()

    output_occupied = np.zeros((h, w), dtype=bool)
    portal_occupied = np.zeros((h, w), dtype=bool)
    specs: list[dict[str, object]] = []

    for idx in range(object_count):
        side_choices = list(SIDES)
        random.shuffle(side_choices)
        placed = False
        for side in side_choices:
            size = random.randint(3, 6)
            offsets = _generate_shape(side, size)
            if offsets is None:
                continue
            result = _choose_exit_portal(
                offsets,
                h,
                w,
                output_occupied,
                portal_occupied,
                must_clip=idx in clip_indices,
            )
            if result is None:
                continue
            p2, visible = result
            _mark_cells(visible, output_occupied)
            _mark_cells([p2], portal_occupied)
            specs.append(
                {
                    "offsets": offsets,
                    "p2": p2,
                    "visible": visible,
                    "object_color": int(object_colors[idx]),
                    "portal_color": int(portal_colors[idx]),
                }
            )
            placed = True
            break
        if not placed:
            return None

    input_occupied = portal_occupied.copy()
    for spec in sorted(specs, key=lambda item: len(item["offsets"]), reverse=True):
        p1 = _choose_entry_portal(spec["offsets"], h, w, input_occupied)
        if p1 is None:
            return None
        spec["p1"] = p1
        cluster = [p1] + [(p1[0] + dr, p1[1] + dc) for dr, dc in spec["offsets"]]
        _mark_cells(cluster, input_occupied)

    inp = np.full((h, w), bg, dtype=int)
    for spec in specs:
        p1 = spec["p1"]
        p2 = spec["p2"]
        portal_color = int(spec["portal_color"])
        object_color = int(spec["object_color"])
        inp[int(p1[0]), int(p1[1])] = portal_color
        inp[int(p2[0]), int(p2[1])] = portal_color
        for dr, dc in spec["offsets"]:
            rr = int(p1[0] + dr)
            cc = int(p1[1] + dc)
            inp[rr, cc] = object_color

    if not _structure_ok(inp):
        return None

    expected = _expected_output(specs, h, w, bg)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal-teleport input")
