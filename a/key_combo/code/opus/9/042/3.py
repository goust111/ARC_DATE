from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
ANCHOR = 4
PALETTE = [1, 2, 3, 5, 6, 7, 8]
MAX_ATTEMPTS = 600
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _label_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                if connectivity == 4:
                    nbrs = neighbors4(rr, cc)
                else:
                    nbrs = (
                        (rr + dr, cc + dc)
                        for dr in (-1, 0, 1)
                        for dc in (-1, 0, 1)
                        if dr or dc
                    )
                for nr, nc in nbrs:
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _find_components(grid: np.ndarray, bg: int, *, skip: np.ndarray | None = None) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    if skip is not None:
        seen |= skip
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.asarray(pts, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": int(len(pts)),
                    "bbox": bbox_coords(coords),
                }
            )
    return comps


def _anchor_component(grid: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]], tuple[int, int, int, int]]:
    comps = _label_mask(grid == ANCHOR, connectivity=4)
    matches = []
    for coords in comps:
        if coords.shape[0] != 4:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        if r1 - r0 == 1 and c1 - c0 == 1:
            box = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
            pts = {tuple(map(int, p)) for p in coords}
            if pts == box:
                matches.append((coords, pts, (r0, c0, r1, c1)))
    assert len(matches) == 1, "expected exactly one yellow 2x2 anchor"
    return matches[0]


def _touches_set(coords: np.ndarray, target: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for nr, nc in neighbors4(rr, cc):
            if (nr, nc) in target:
                return True
    return False


def _slide_right_until_contact(
    coords: np.ndarray,
    occupied: set[tuple[int, int]],
    width: int,
) -> tuple[np.ndarray, int]:
    dx = 0
    while True:
        can_advance = True
        for r, c in coords:
            nr = int(r)
            nc = int(c) + dx + 1
            if nc >= width or (nr, nc) in occupied:
                can_advance = False
                break
        if not can_advance:
            break
        dx += 1
    shifted = coords.copy()
    shifted[:, 1] += dx
    assert _touches_set(shifted, occupied), "object must stop on first direct rightward contact"
    return shifted, dx


def _sorted_objects(components: list[dict], *, mode: str, anchor_left: int) -> list[dict]:
    if mode == "area":
        return sorted(components, key=lambda obj: (obj["area"], obj["bbox"][0], obj["bbox"][1]))
    if mode == "distance":
        def gap(obj: dict) -> int:
            return min(anchor_left - int(c) - 1 for _r, c in obj["coords"])
        return sorted(components, key=lambda obj: (gap(obj), obj["bbox"][0], obj["bbox"][1]))
    raise ValueError(mode)


def _solve(grid: np.ndarray, *, mode: str = "area") -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    anchor_coords, anchor_pts, anchor_bbox = _anchor_component(g)
    anchor_left = int(anchor_bbox[1])
    skip = np.zeros_like(g, dtype=bool)
    skip[anchor_coords[:, 0], anchor_coords[:, 1]] = True
    objects = _find_components(g, bg, skip=skip)
    assert 4 <= len(objects) <= 5, "expected 4-5 objects besides the anchor"
    assert len({obj["area"] for obj in objects}) == len(objects), "areas must be unique"

    ordered = _sorted_objects(objects, mode=mode, anchor_left=anchor_left)
    out = np.full_like(g, bg)
    out[anchor_coords[:, 0], anchor_coords[:, 1]] = ANCHOR
    occupied = set(anchor_pts)
    placed: list[dict] = []

    for obj in ordered:
        coords = np.asarray(obj["coords"], dtype=int)
        shifted, _ = _slide_right_until_contact(coords, occupied, g.shape[1])
        color = int(obj["color"])
        out[shifted[:, 0], shifted[:, 1]] = color
        placed.append({"color": color, "area": obj["area"], "coords": shifted})
        occupied |= {tuple(map(int, p)) for p in shifted}

    touching_anchor = [obj for obj in placed if _touches_set(obj["coords"], anchor_pts)]
    assert len(touching_anchor) == 1, "exactly one object should directly touch the anchor"
    recolor = touching_anchor[0]["coords"]
    out[recolor[:, 0], recolor[:, 1]] = ANCHOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the left-side objects rightward in ascending area order so they pack against a fixed yellow anchor, then recolor the unique anchor-touching object yellow.

    Input structure:
    - A black background contains one fixed yellow 2x2 anchor square in the right half.
    - Four or five separated 4-connected monochromatic objects sit entirely to the left of the anchor.
    - Each object has a unique area and shares at least one row with the anchor's two-row band, so a horizontal slide can eventually make contact.

    Rule:
    - Identify the yellow 2x2 anchor and all other monochromatic 4-connected objects.
    - Sort the non-anchor objects by ascending pixel area, breaking ties by bbox top row then left column.
    - Process them in that order: slide each whole object straight right until one more step would overlap the anchor or an already placed object, so it stops at first 4-connected contact.
    - After all objects settle, recolor the unique object that is directly 4-neighbor-adjacent to the yellow anchor so all of its pixels become yellow; keep the anchor fixed.

    Key insight:
    - Area does not set how far an object moves; it sets which object claims the nearest slot first, and that cascade of earlier placements blocks every later slide.
    """
    return _solve(grid, mode="area")


def _normalize(coords: set[tuple[int, int]] | list[tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(list(coords), dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    arr = arr[np.lexsort((arr[:, 1], arr[:, 0]))]
    return arr


def _transform(coords: np.ndarray) -> np.ndarray:
    arr = coords.copy()
    for _ in range(random.randrange(4)):
        arr = np.column_stack((arr[:, 1], -arr[:, 0]))
    if random.random() < 0.5:
        arr[:, 1] *= -1
    return _normalize({tuple(map(int, p)) for p in arr})


def _random_polyomino(area: int, *, smallest: bool = False) -> np.ndarray:
    if area == 2:
        return np.asarray([(0, 0), (1, 0)], dtype=int)

    templates = []
    if smallest:
        templates = [
            np.asarray([(0, 1), (1, 0), (1, 1)], dtype=int),
            np.asarray([(0, 0), (0, 1), (1, 1)], dtype=int),
            np.asarray([(0, 1), (1, 1), (2, 0), (2, 1)], dtype=int),
        ]
    if templates and area in {3, 4} and random.random() < 0.6:
        shape = random.choice([tpl for tpl in templates if tpl.shape[0] == area])
        return _transform(shape)

    for _ in range(300):
        cells = {(0, 0)}
        frontier = {(0, 1), (1, 0), (0, -1), (-1, 0)}
        while len(cells) < area:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            r, c = cell
            for dr, dc in DIR4:
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.add(nxt)
        arr = _transform(_normalize(cells))
        r0, c0, r1, c1 = bbox_coords(arr)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h > 4 or w > 5:
            continue
        if area > 2 and h * w == area:
            continue
        rows = sorted(set(map(int, arr[:, 0])))
        if smallest:
            if not any((r + 1) in rows for r in rows):
                continue
            max_c = int(arr[:, 1].max())
            if not all(np.any((arr[:, 0] == rr) & (arr[:, 1] == max_c)) for rr in rows[:2]):
                consecutive = [(r, r + 1) for r in rows if (r + 1) in rows]
                if not consecutive:
                    continue
                rr0, rr1 = random.choice(consecutive)
                keep = np.any((arr[:, 0] == rr0) & (arr[:, 1] == max_c)) and np.any(
                    (arr[:, 0] == rr1) & (arr[:, 1] == max_c)
                )
                if not keep:
                    continue
        return arr
    raise RuntimeError(f"failed to generate polyomino of area {area}")


def _top_choices(shape: np.ndarray, anchor_top: int, height: int, *, smallest: bool) -> list[int]:
    rows = sorted(set(map(int, shape[:, 0])))
    choices = []
    if smallest:
        for rr in rows:
            if (rr + 1) in rows:
                top = anchor_top - rr
                if 0 <= top and top + int(shape[:, 0].max()) < height:
                    choices.append(top)
    else:
        for rr in rows:
            for target in (anchor_top, anchor_top + 1):
                top = target - rr
                if 0 <= top and top + int(shape[:, 0].max()) < height:
                    choices.append(top)
    return sorted(set(choices))


def _coords_to_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, p)) for p in coords}


def _build_final_layout(
    height: int,
    width: int,
    anchor_top: int,
    anchor_left: int,
    areas: list[int],
    colors: list[int],
) -> list[dict] | None:
    anchor_pts = {
        (anchor_top, anchor_left),
        (anchor_top, anchor_left + 1),
        (anchor_top + 1, anchor_left),
        (anchor_top + 1, anchor_left + 1),
    }
    occupied = set(anchor_pts)
    finals = []
    for i, (area, color) in enumerate(zip(areas, colors)):
        smallest = i == 0
        for _ in range(120):
            shape = _random_polyomino(area, smallest=smallest)
            tops = _top_choices(shape, anchor_top, height, smallest=smallest)
            if not tops:
                continue
            top = random.choice(tops)
            coords = shape.copy()
            coords[:, 0] += top
            shifted, _ = _slide_right_until_contact(coords, occupied, width)
            if shifted[:, 1].min() < 1:
                continue
            if _touches_set(shifted, anchor_pts) != smallest:
                continue
            shifted_set = _coords_to_set(shifted)
            if shifted_set & occupied:
                continue
            finals.append(
                {
                    "area": area,
                    "color": color,
                    "shape": shape,
                    "final": shifted,
                }
            )
            occupied |= shifted_set
            break
        else:
            return None

    touching = [obj for obj in finals if _touches_set(obj["final"], anchor_pts)]
    if len(touching) != 1 or touching[0]["area"] != min(areas):
        return None
    return finals


def _compose_input(
    height: int,
    width: int,
    anchor_top: int,
    anchor_left: int,
    finals: list[dict],
    travels: list[int],
) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    grid[anchor_top : anchor_top + 2, anchor_left : anchor_left + 2] = ANCHOR
    occ = np.zeros((height, width), dtype=int)
    occ[anchor_top : anchor_top + 2, anchor_left : anchor_left + 2] = -1

    for obj, travel in zip(finals, travels):
        coords = obj["final"].copy()
        coords[:, 1] -= int(travel)
        if coords[:, 1].min() < 0:
            return None
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if occ[rr, cc] != 0:
                return None
            occ[rr, cc] = int(obj["color"])
            grid[rr, cc] = int(obj["color"])

    return grid


def _objects_not_adjacent(grid: np.ndarray) -> bool:
    bg = get_bg(grid)
    _, anchor_pts, _ = _anchor_component(grid)
    skip = np.zeros_like(grid, dtype=bool)
    for r, c in anchor_pts:
        skip[r, c] = True
    comps = _find_components(grid, bg, skip=skip)
    owner = {}
    for idx, obj in enumerate(comps):
        for r, c in obj["coords"]:
            owner[(int(r), int(c))] = idx
    for (r, c), idx in owner.items():
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in owner and owner[(nr, nc)] != idx:
                return False
    return True


def _distance_sort_output(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, mode="distance")


def _travel_candidates(finals: list[dict]) -> list[list[int]]:
    candidates = []
    for obj in finals:
        max_shift = min(10, int(obj["final"][:, 1].min()))
        if max_shift < 1:
            return []
        candidates.append(list(range(1, max_shift + 1)))
    return candidates


def _semantic_checks(inp: np.ndarray, finals: list[dict], travels: list[int]) -> bool:
    out = main(inp)
    try:
        distance_out = _distance_sort_output(inp)
    except AssertionError:
        distance_out = None
    if distance_out is not None and np.array_equal(out, distance_out):
        return False

    bg = get_bg(inp)
    anchor_coords, anchor_pts, anchor_bbox = _anchor_component(inp)
    skip = np.zeros_like(inp, dtype=bool)
    skip[anchor_coords[:, 0], anchor_coords[:, 1]] = True
    comps = _find_components(inp, bg, skip=skip)
    if len(comps) != len(finals):
        return False
    if len({obj["area"] for obj in comps}) != len(comps):
        return False

    anchor_rows = {anchor_bbox[0], anchor_bbox[0] + 1}
    if not all(anchor_rows & set(map(int, obj["coords"][:, 0])) for obj in comps):
        return False
    if travels[0] >= max(travels):
        return False
    if not _objects_not_adjacent(inp):
        return False

    reconstructed = []
    occupied = set(anchor_pts)
    for obj in sorted(comps, key=lambda it: (it["area"], it["bbox"][0], it["bbox"][1])):
        shifted, _ = _slide_right_until_contact(obj["coords"], occupied, inp.shape[1])
        reconstructed.append({"area": obj["area"], "coords": shifted})
        occupied |= _coords_to_set(shifted)
    direct = [obj for obj in reconstructed if _touches_set(obj["coords"], anchor_pts)]
    if len(direct) != 1 or direct[0]["area"] != min(obj["area"] for obj in reconstructed):
        return False
    if sum(not _touches_set(obj["coords"], anchor_pts) for obj in reconstructed[1:]) < 2:
        return False

    # At least one later object must settle because a smaller object blocks its shared row.
    anchor_left = anchor_bbox[1]
    for obj in comps:
        if obj["area"] == min(o["area"] for o in comps):
            continue
        nearest_anchor_gap = min(anchor_left - int(c) - 1 for _r, c in obj["coords"])
        if nearest_anchor_gap != travels[0]:
            break
    else:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(15, 20)
    width = random.randint(24, 29)
    n_objects = random.choice([4, 5])
    areas = [3, 4, 5, 6] if n_objects == 4 else [2, 3, 4, 5, 6]
    colors = random.sample(PALETTE, n_objects)

    anchor_top = random.randint(4, height - 6)
    anchor_left = random.randint(max(width // 2 + 2, 15), width - 4)
    finals = _build_final_layout(height, width, anchor_top, anchor_left, areas, colors)
    if finals is None:
        return None

    candidate_ranges = _travel_candidates(finals)
    if not candidate_ranges:
        return None

    for _ in range(320):
        travels = [random.choice(opts) for opts in candidate_ranges]
        if travels[0] >= max(travels):
            continue
        inp = _compose_input(height, width, anchor_top, anchor_left, finals, travels)
        if inp is None:
            continue
        if not _semantic_checks(inp, finals, travels):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
