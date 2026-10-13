import random
from collections import defaultdict, deque

import numpy as np


BG = 0
TEMPLATE_COLOR = 5
TRIANGLE_COLORS = (2, 3, 4, 6, 7)
DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _triangle_cells(start: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = start
    dr, dc = direction
    return ((r, c), (r + dr, c), (r, c + dc))


def _halo(cells: tuple[tuple[int, int], ...], h: int, w: int) -> frozenset[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                out.add((rr, cc))
    return frozenset(out)


def _triangle_from_component(coords: np.ndarray) -> dict | None:
    if len(coords) != 3:
        return None
    cells = {(int(r), int(c)) for r, c in coords}
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    missing = list(corners - cells)
    if len(missing) != 1:
        return None
    miss = missing[0]
    start = max(cells, key=lambda rc: abs(rc[0] - miss[0]) + abs(rc[1] - miss[1]))
    direction = (miss[0] - start[0], miss[1] - start[1])
    if direction not in DIRS:
        return None
    expect = set(_triangle_cells(start, direction))
    if expect != cells:
        return None
    return {
        "start": start,
        "dir": direction,
        "cells": tuple(sorted(cells)),
    }


def _template_offsets(mask_coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = _bbox(mask_coords)
    if (r0 + r1) % 2 != 0 or (c0 + c1) % 2 != 0:
        raise ValueError("template bbox center must lie on a grid cell")
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    return tuple(sorted((int(r - cr), int(c - cc)) for r, c in mask_coords))


def _stamp_fits(shape: tuple[int, int], center: tuple[int, int], offsets: tuple[tuple[int, int], ...]) -> bool:
    h, w = shape
    r, c = center
    return all(0 <= r + dr < h and 0 <= c + dc < w for dr, dc in offsets)


def _trace_landpoint(
    start: tuple[int, int],
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[tuple[int, int], str, tuple[int, int] | None, list[tuple[int, int]]]:
    h, w = shape
    r, c = start
    dr, dc = direction
    rr, cc = r + dr, c + dc
    path: list[tuple[int, int]] = []
    last: tuple[int, int] | None = None
    while 0 <= rr < h and 0 <= cc < w:
        cur = (rr, cc)
        if cur in occupied:
            if last is None:
                raise ValueError("projectile has no free landing cell")
            return last, "obstacle", cur, path
        path.append(cur)
        last = cur
        rr += dr
        cc += dc
    if last is None:
        raise ValueError("projectile leaves the grid immediately")
    return last, "edge", None, path


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    template_comps = _components(g == TEMPLATE_COLOR, connectivity=4)
    if len(template_comps) != 1:
        raise ValueError("expected exactly one template component")
    template = template_comps[0]
    template_offsets = _template_offsets(template)
    template_cells = {tuple(map(int, rc)) for rc in template.tolist()}

    triangles: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    colors = sorted(int(v) for v in np.unique(g) if int(v) not in (BG, TEMPLATE_COLOR))
    for color in colors:
        comps = _components(g == color, connectivity=8)
        if len(comps) != 1:
            raise ValueError("each triangle color must form exactly one component")
        tri = _triangle_from_component(comps[0])
        if tri is None:
            raise ValueError("invalid triangle sprite")
        tri["color"] = color
        occupied.update(tri["cells"])
        triangles.append(tri)

    contributions: dict[tuple[int, int], list[dict]] = defaultdict(list)
    output = g.copy()
    triangle_cells: set[tuple[int, int]] = set()
    stop_types: list[str] = []

    for tri in triangles:
        triangle_cells.update(tri["cells"])

    for tri in triangles:
        own = set(tri["cells"])
        land, stop_type, stopper, path = _trace_landpoint(tri["start"], tri["dir"], occupied - own, (h, w))
        if not _stamp_fits((h, w), land, template_offsets):
            raise ValueError("stamp center would place mask out of bounds")
        tri["land"] = land
        tri["stop_type"] = stop_type
        tri["stopper"] = stopper
        tri["path"] = path
        tri["distance"] = abs(land[0] - tri["start"][0]) + abs(land[1] - tri["start"][1])
        stop_types.append(stop_type)
        prio = (tri["distance"], tri["start"][0], tri["start"][1])
        tri["priority"] = prio
        for dr, dc in template_offsets:
            cell = (land[0] + dr, land[1] + dc)
            if cell in template_cells:
                continue
            contributions[cell].append({"priority": prio, "color": tri["color"], "triangle": tri})

    for cell, items in contributions.items():
        winner = min(items, key=lambda item: item["priority"])
        output[cell] = winner["color"]

    for r, c in triangle_cells:
        output[r, c] = BG

    overlap_cells = {cell: items for cell, items in contributions.items() if len(items) >= 2}
    return {
        "template": template,
        "template_offsets": template_offsets,
        "template_cells": template_cells,
        "triangles": triangles,
        "contributions": contributions,
        "overlap_cells": overlap_cells,
        "output": output,
        "stop_types": stop_types,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire each 3-pixel corner triangle through its missing corner and stamp the shared template at the stopping cell.

    Input structure:
    - Background is black (0).
    - One grey (5) connected component is a template mask placed in a corner region.
    - Three to five other monochrome components are 3-pixel L-shapes occupying three corners of a 2x2 box.

    Rule:
    - Read the grey template as a set of offsets from its bbox center.
    - For each triangle, find the unique occupied corner opposite the missing 2x2 corner; that pixel is the firing start, and the missing corner gives the diagonal firing direction.
    - Trace the projectile cell by cell, treating grey as passable, and stop on the last free cell before another triangle or before leaving the grid.
    - Stamp the template there in the triangle's color; on stamp overlaps, prefer shorter firing distance, then smaller start row, then smaller start column.
    - Keep the grey template unchanged and erase every original triangle to black.

    Key insight:
    - The triangle does not mark the answer directly: its geometry selects a ray, the ray's stopping point selects a stamp center, and only then does the shared template appear in color.
    """
    return _analyze_scene(np.asarray(grid, dtype=int))["output"]


def _rot90_mask(mask: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(mask, k)


def _random_template_mask(rng: random.Random, family: str) -> np.ndarray:
    if family == "line":
        bases = [np.array([[1, 1, 1]], dtype=int)]
    else:
        bases = [
            np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=int),
            np.array([[0, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=int),
            np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
        ]
    mask = bases[rng.randrange(len(bases))].copy()
    mask = _rot90_mask(mask, rng.randrange(4))
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    return mask.astype(int)


def _place_template(
    h: int,
    w: int,
    mask: np.ndarray,
    corner: str,
    rng: random.Random,
) -> tuple[np.ndarray, set[tuple[int, int]], tuple[tuple[int, int], ...]]:
    mh, mw = mask.shape
    if corner == "tl":
        r0 = rng.randint(0, 4 - mh)
        c0 = rng.randint(0, 4 - mw)
    elif corner == "tr":
        r0 = rng.randint(0, 4 - mh)
        c0 = w - mw - rng.randint(0, 4 - mw)
    elif corner == "bl":
        r0 = h - mh - rng.randint(0, 4 - mh)
        c0 = rng.randint(0, 4 - mw)
    else:
        r0 = h - mh - rng.randint(0, 4 - mh)
        c0 = w - mw - rng.randint(0, 4 - mw)

    grid = np.zeros((h, w), dtype=int)
    cells: set[tuple[int, int]] = set()
    for rr, cc in np.argwhere(mask == 1):
        cell = (r0 + int(rr), c0 + int(cc))
        grid[cell] = TEMPLATE_COLOR
        cells.add(cell)
    offsets = _template_offsets(np.argwhere(grid == TEMPLATE_COLOR))
    return grid, cells, offsets


def _generate_candidates(
    h: int,
    w: int,
    template_cells: set[tuple[int, int]],
    offsets: tuple[tuple[int, int], ...],
) -> tuple[list[dict], dict[tuple[int, int, int, int], dict], dict[tuple[int, int], list[dict]]]:
    candidates: list[dict] = []
    by_start_dir: dict[tuple[int, int, int, int], dict] = {}
    by_cell: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in range(h):
        for c in range(w):
            for dr, dc in DIRS:
                cells = _triangle_cells((r, c), (dr, dc))
                if not all(0 <= rr < h and 0 <= cc < w for rr, cc in cells):
                    continue
                if set(cells) & template_cells:
                    continue
                halo = _halo(cells, h, w)
                if halo & template_cells:
                    continue
                path: list[tuple[int, int]] = []
                rr, cc = r + dr, c + dc
                bad = False
                while 0 <= rr < h and 0 <= cc < w:
                    cur = (rr, cc)
                    if cur in template_cells:
                        bad = True
                        break
                    path.append(cur)
                    rr += dr
                    cc += dc
                if bad or not path:
                    continue
                cand = {
                    "start": (r, c),
                    "dir": (dr, dc),
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": halo,
                    "path": tuple(path),
                    "edge_land": path[-1],
                    "edge_ok": _stamp_fits((h, w), path[-1], offsets),
                }
                candidates.append(cand)
                by_start_dir[(r, c, dr, dc)] = cand
                for cell in cells:
                    by_cell[cell].append(cand)
    return candidates, by_start_dir, by_cell


def _fits_candidate(cand: dict, placements: list[dict]) -> bool:
    used = set()
    for placed in placements:
        used.update(placed["cell_set"])
    return not bool(cand["halo"] & used)


def _make_grid(h: int, w: int, template_grid: np.ndarray, placements: list[dict], colors: list[int]) -> np.ndarray:
    grid = template_grid.copy()
    for cand, color in zip(placements, colors):
        for r, c in cand["cells"]:
            grid[r, c] = color
    return grid


def _scene_is_non_degenerate(grid: np.ndarray, mode: str) -> bool:
    try:
        scene = _analyze_scene(grid)
    except Exception:
        return False

    triangles = scene["triangles"]
    if not (3 <= len(triangles) <= 5):
        return False
    if np.array_equal(scene["output"], grid):
        return False

    added_pixels = int(np.sum((scene["output"] != grid) & (scene["output"] != BG)))
    if added_pixels < len(triangles):
        return False

    if mode == "mixed":
        if "edge" not in scene["stop_types"] or "obstacle" not in scene["stop_types"]:
            return False
    elif mode == "generic":
        if "obstacle" not in scene["stop_types"]:
            return False
    else:
        if not scene["overlap_cells"]:
            return False

    if mode == "overlap_shorter":
        ok = False
        for items in scene["overlap_cells"].values():
            dists = {item["triangle"]["distance"] for item in items}
            if len(dists) >= 2:
                ok = True
                break
        if not ok:
            return False

    if mode == "overlap_tie_row":
        ok = False
        for items in scene["overlap_cells"].values():
            tris = [item["triangle"] for item in items]
            for i in range(len(tris)):
                for j in range(i + 1, len(tris)):
                    a, b = tris[i], tris[j]
                    if a["distance"] == b["distance"] and a["start"][0] != b["start"][0]:
                        ok = True
                        break
                if ok:
                    break
            if ok:
                break
        if not ok:
            return False

    if mode == "overlap_tie_col":
        ok = False
        for items in scene["overlap_cells"].values():
            tris = [item["triangle"] for item in items]
            for i in range(len(tris)):
                for j in range(i + 1, len(tris)):
                    a, b = tris[i], tris[j]
                    if a["distance"] == b["distance"] and a["start"][0] == b["start"][0] and a["start"][1] != b["start"][1]:
                        ok = True
                        break
                if ok:
                    break
            if ok:
                break
        if not ok:
            return False

    # Each triangle must matter.
    for tri in triangles:
        reduced = grid.copy()
        for r, c in tri["cells"]:
            reduced[r, c] = BG
        try:
            if np.array_equal(main(reduced), scene["output"]):
                return False
        except Exception:
            pass

    return True


def _pick_candidate_at_start(
    by_start_dir: dict[tuple[int, int, int, int], dict],
    start: tuple[int, int],
    direction: tuple[int, int],
) -> dict | None:
    return by_start_dir.get((start[0], start[1], direction[0], direction[1]))


def _choose_overlap_mode(rng: random.Random) -> str:
    x = rng.random()
    if x < 0.36:
        return "mixed"
    if x < 0.62:
        return "overlap_shorter"
    if x < 0.81:
        return "overlap_tie_row"
    return "overlap_tie_col"


def _build_mixed(
    h: int,
    w: int,
    offsets: tuple[tuple[int, int], ...],
    candidates: list[dict],
    by_start_dir: dict[tuple[int, int, int, int], dict],
    by_cell: dict[tuple[int, int], list[dict]],
    rng: random.Random,
) -> list[dict] | None:
    for _ in range(240):
        placements: list[dict] = []
        target_count = rng.choice((3, 4))
        land = (rng.randint(4, h - 5), rng.randint(4, w - 5))
        direction = rng.choice(DIRS)
        if not _stamp_fits((h, w), land, offsets):
            continue
        steps = rng.randint(3, 6)
        start = (land[0] - steps * direction[0], land[1] - steps * direction[1])
        shooter = _pick_candidate_at_start(by_start_dir, start, direction)
        if shooter is None:
            continue
        placements.append(shooter)

        stop_cell = (land[0] + direction[0], land[1] + direction[1])
        blockers = by_cell.get(stop_cell, [])
        rng.shuffle(blockers)
        blocker = None
        for cand in blockers:
            if cand is shooter:
                continue
            if _fits_candidate(cand, placements):
                blocker = cand
                break
        if blocker is None:
            continue
        placements.append(blocker)

        occupied = set()
        for placed in placements:
            occupied.update(placed["cell_set"])
        if _trace_landpoint(shooter["start"], shooter["dir"], occupied - shooter["cell_set"], (h, w))[:2] != (land, "obstacle"):
            continue

        extras = [cand for cand in candidates if _fits_candidate(cand, placements)]
        rng.shuffle(extras)
        edge_added = False
        for cand in extras:
            if not cand["edge_ok"]:
                continue
            if any(cell in occupied for cell in cand["path"]):
                continue
            placements.append(cand)
            occupied.update(cand["cell_set"])
            edge_added = True
            break
        if not edge_added:
            continue

        extras = [cand for cand in candidates if _fits_candidate(cand, placements)]
        rng.shuffle(extras)
        while len(placements) < target_count:
            found = False
            for cand in extras:
                if _fits_candidate(cand, placements):
                    placements.append(cand)
                    found = True
                    break
            if not found:
                break
            extras = [cand for cand in candidates if _fits_candidate(cand, placements)]
            rng.shuffle(extras)
        if len(placements) < 3:
            continue
        return placements
    return None


def _build_overlap(
    h: int,
    w: int,
    offsets: tuple[tuple[int, int], ...],
    candidates: list[dict],
    by_start_dir: dict[tuple[int, int, int, int], dict],
    by_cell: dict[tuple[int, int], list[dict]],
    rng: random.Random,
    mode: str,
) -> list[dict] | None:
    if mode == "overlap_tie_row":
        dir_pairs = [((1, 1), (-1, 1)), ((1, -1), (-1, -1))]
    elif mode == "overlap_tie_col":
        dir_pairs = [((1, 1), (1, -1)), ((-1, 1), (-1, -1))]
    else:
        dir_pairs = [
            ((1, 1), (-1, 1)),
            ((1, -1), (-1, -1)),
            ((1, 1), (1, -1)),
            ((-1, 1), (-1, -1)),
        ]

    for _ in range(320):
        placements: list[dict] = []
        land = (rng.randint(4, h - 5), rng.randint(4, w - 5))
        if not _stamp_fits((h, w), land, offsets):
            continue
        d1, d2 = rng.choice(dir_pairs)
        if mode == "overlap_shorter":
            k1 = rng.randint(2, 4)
            k2_choices = [k for k in range(2, 6) if k != k1]
            k2 = rng.choice(k2_choices)
        else:
            k1 = rng.randint(2, 5)
            k2 = k1

        starts = [
            (land[0] - k1 * d1[0], land[1] - k1 * d1[1]),
            (land[0] - k2 * d2[0], land[1] - k2 * d2[1]),
        ]
        shooters = [
            _pick_candidate_at_start(by_start_dir, starts[0], d1),
            _pick_candidate_at_start(by_start_dir, starts[1], d2),
        ]
        if shooters[0] is None or shooters[1] is None:
            continue
        if not _fits_candidate(shooters[0], placements):
            continue
        placements.append(shooters[0])
        if not _fits_candidate(shooters[1], placements):
            continue
        placements.append(shooters[1])

        blockers: list[dict] = []
        ok = True
        for direction in (d1, d2):
            stop_cell = (land[0] + direction[0], land[1] + direction[1])
            options = by_cell.get(stop_cell, [])
            rng.shuffle(options)
            picked = None
            for cand in options:
                if cand in placements or cand in blockers:
                    continue
                if _fits_candidate(cand, placements + blockers):
                    picked = cand
                    break
            if picked is None:
                ok = False
                break
            blockers.append(picked)
        if not ok:
            continue
        placements.extend(blockers)

        occupied = set()
        for placed in placements:
            occupied.update(placed["cell_set"])
        t1 = _trace_landpoint(shooters[0]["start"], shooters[0]["dir"], occupied - shooters[0]["cell_set"], (h, w))
        t2 = _trace_landpoint(shooters[1]["start"], shooters[1]["dir"], occupied - shooters[1]["cell_set"], (h, w))
        if t1[0] != land or t1[1] != "obstacle" or t2[0] != land or t2[1] != "obstacle":
            continue

        need_edge = True
        for cand in placements:
            stop = _trace_landpoint(cand["start"], cand["dir"], occupied - cand["cell_set"], (h, w))
            if stop[1] == "edge":
                need_edge = False
                break

        target_count = 4
        if need_edge:
            extras = [cand for cand in candidates if _fits_candidate(cand, placements)]
            rng.shuffle(extras)
            edge_added = False
            for cand in extras:
                if not cand["edge_ok"]:
                    continue
                if any(cell in occupied for cell in cand["path"]):
                    continue
                placements.append(cand)
                occupied.update(cand["cell_set"])
                edge_added = True
                break
            if not edge_added:
                continue
            target_count = 5
        elif rng.random() < 0.5:
            target_count = 5

        extras = [cand for cand in candidates if _fits_candidate(cand, placements)]
        rng.shuffle(extras)
        while len(placements) < target_count:
            added = False
            for cand in extras:
                if _fits_candidate(cand, placements):
                    placements.append(cand)
                    added = True
                    break
            if not added:
                break
            extras = [cand for cand in candidates if _fits_candidate(cand, placements)]
            rng.shuffle(extras)
        if len(placements) < 4:
            continue
        return placements
    return None


def _build_generic_random(
    h: int,
    w: int,
    template_grid: np.ndarray,
    candidates: list[dict],
    rng: random.Random,
) -> list[dict] | None:
    for _ in range(320):
        target_count = rng.choice((4, 5))
        order = candidates[:]
        rng.shuffle(order)
        placements: list[dict] = []
        for cand in order:
            if _fits_candidate(cand, placements):
                placements.append(cand)
                if len(placements) == target_count:
                    break
        if len(placements) < target_count:
            continue
        probe_colors = list(TRIANGLE_COLORS[:target_count])
        grid = _make_grid(h, w, template_grid, placements, probe_colors)
        if _scene_is_non_degenerate(grid, "generic"):
            return placements
    return None


def _assign_colors(rng: random.Random, n: int) -> list[int]:
    cols = list(TRIANGLE_COLORS)
    rng.shuffle(cols)
    return cols[:n]


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        h = rng.randint(18, 24)
        w = rng.randint(18, 24)

        family = "irregular" if rng.random() < 0.28 else "line"
        template_mask = _random_template_mask(rng, family)
        corner = rng.choice(("tl", "tr", "bl", "br"))
        template_grid, template_cells, offsets = _place_template(h, w, template_mask, corner, rng)

        candidates, by_start_dir, by_cell = _generate_candidates(h, w, template_cells, offsets)
        if len(candidates) < 20:
            continue

        if family == "irregular":
            mode = "generic"
            placements = _build_generic_random(h, w, template_grid, candidates, rng)
        else:
            mode = _choose_overlap_mode(rng)
            if mode == "mixed":
                placements = _build_mixed(h, w, offsets, candidates, by_start_dir, by_cell, rng)
            else:
                placements = _build_overlap(h, w, offsets, candidates, by_start_dir, by_cell, rng, mode)
        if placements is None:
            continue

        colors = _assign_colors(rng, len(placements))
        grid = _make_grid(h, w, template_grid, placements, colors)
        if _scene_is_non_degenerate(grid, mode):
            return grid

    raise RuntimeError("failed to generate a valid puzzle input")
