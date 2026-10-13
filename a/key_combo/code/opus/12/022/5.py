import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
ORANGE = 7

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def neighbors8(r: int, c: int):
    for dr, dc in DIRS8:
        yield r + dr, c + dc


def connected_components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIRS4 if connectivity == 4 else DIRS8
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def bbox_of(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(connected_components(mask, connectivity=4)) == 1


def defect_is_valid(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    if len(cells) > 2:
        return False
    for r, c in cells:
        for nr, nc in neighbors8(r, c):
            if not in_bounds(grid.shape, nr, nc):
                return False
            if (nr, nc) in cell_set:
                continue
            if int(grid[nr, nc]) != BLUE:
                return False
    return True


def build_blue_component_ids(grid: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    comps = connected_components(grid == BLUE, connectivity=4)
    ids = np.full(grid.shape, -1, dtype=int)
    for idx, comp in enumerate(comps):
        for r, c in comp:
            ids[r, c] = idx
    return ids, comps


def find_caps(grid: np.ndarray) -> list[dict]:
    caps: list[dict] = []
    for comp in connected_components(grid == GREY, connectivity=4):
        if len(comp) != 2:
            continue
        (r0, c0), (r1, c1) = sorted(comp)
        if not (r0 == r1 or c0 == c1):
            continue
        caps.append({"cells": comp, "bbox": bbox_of(comp)})
    return caps


def cap_direction(cap_bbox: tuple[int, int, int, int], comp_bbox: tuple[int, int, int, int]) -> tuple[int, int] | None:
    cr0, cc0, cr1, cc1 = cap_bbox
    br0, bc0, br1, bc1 = comp_bbox
    if cr0 == cr1:
        if cr1 < br0:
            return (-1, 0)
        if cr0 > br1:
            return (1, 0)
    if cc0 == cc1:
        if cc1 < bc0:
            return (0, -1)
        if cc0 > bc1:
            return (0, 1)
    return None


def analyze_scene(grid: np.ndarray) -> list[dict]:
    blue_ids, blue_components = build_blue_component_ids(grid)
    blue_bboxes = [bbox_of(comp) for comp in blue_components]
    caps = find_caps(grid)

    cap_to_blue: list[int | None] = []
    for cap in caps:
        adjacent_ids: set[int] = set()
        for r, c in cap["cells"]:
            for nr, nc in neighbors4(r, c):
                if in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BLUE:
                    adjacent_ids.add(int(blue_ids[nr, nc]))
        cap_to_blue.append(next(iter(adjacent_ids)) if len(adjacent_ids) == 1 else None)

    defect_infos: list[dict] = []
    for comp in connected_components(grid == BLACK, connectivity=4):
        if not defect_is_valid(grid, comp):
            continue
        touched_blue: set[int] = set()
        for r, c in comp:
            for nr, nc in neighbors8(r, c):
                if in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BLUE:
                    touched_blue.add(int(blue_ids[nr, nc]))
        if len(touched_blue) != 1:
            continue
        blue_id = next(iter(touched_blue))
        matched_caps = [caps[i] for i, bid in enumerate(cap_to_blue) if bid == blue_id]
        direction = None
        if matched_caps:
            direction = cap_direction(matched_caps[0]["bbox"], blue_bboxes[blue_id])
        defect_infos.append(
            {
                "cells": comp,
                "blue_id": blue_id,
                "blue_bbox": blue_bboxes[blue_id],
                "cap": matched_caps[0] if matched_caps else None,
                "direction": direction,
            }
        )
    return defect_infos


def halo_cells(grid: np.ndarray, defect_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    halos: set[tuple[int, int]] = set()
    defect_set = set(defect_cells)
    for r, c in defect_cells:
        for nr, nc in neighbors8(r, c):
            if in_bounds(grid.shape, nr, nc) and (nr, nc) not in defect_set and grid[nr, nc] == BLUE:
                halos.add((nr, nc))
    return sorted(halos)


def make_tile(
    defect_cells: list[tuple[int, int]],
    halo: list[tuple[int, int]],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    all_cells = defect_cells + halo
    r0, c0, r1, c1 = bbox_of(all_cells)
    tile = np.full((r1 - r0 + 1, c1 - c0 + 1), -1, dtype=int)
    for r, c in halo:
        tile[r - r0, c - c0] = ORANGE
    for r, c in defect_cells:
        tile[r - r0, c - c0] = BLACK
    return tile, (r0, c0, r1, c1)


def apply_tiling(
    out: np.ndarray,
    tile: np.ndarray,
    bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> None:
    dr, dc = direction
    tile_h, tile_w = tile.shape
    step = tile_h if dr != 0 else tile_w
    r0, c0, _r1, _c1 = bbox
    k = 1
    while True:
        placed_any = False
        anchor_r = r0 + k * step * dr
        anchor_c = c0 + k * step * dc
        for tr in range(tile_h):
            for tc in range(tile_w):
                val = int(tile[tr, tc])
                if val < 0:
                    continue
                rr = anchor_r + tr
                cc = anchor_c + tc
                if not in_bounds(out.shape, rr, cc):
                    continue
                placed_any = True
                if int(out[rr, cc]) in (BLUE, GREY, ORANGE):
                    continue
                out[rr, cc] = val
        if not placed_any:
            break
        k += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect black defects inside blue regions, halo them in orange, then repeat each halo tile outward in the grey-cap direction.

    Input structure:
    - The background is black.
    - The main foreground consists of 4-connected blue regions, some containing a tiny black internal defect.
    - Each blue region that contains a defect has exactly one grey 2-pixel cap just outside its boundary.
    - The cap is horizontal for vertical extension and vertical for horizontal extension.

    Rule:
    - Find every 4-connected black component of size at most 2 whose full 8-neighborhood is blue except for the component itself.
    - Recolor the blue 8-neighbors of each such defect orange while keeping the defect cells black.
    - Take the minimal bounding box of the defect plus its orange halo as a tile, treating all non-black and non-orange cells inside that box as transparent.
    - Use the associated grey cap position to determine whether that tile repeats upward, downward, leftward, or rightward, and periodically copy it until the grid edge without overwriting blue, grey, or orange cells.

    Key insight:
    - The repeated pattern is not a fixed stamp: it is the defect's own halo shape, and the grey cap only supplies the extension direction.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    infos = analyze_scene(g)

    prepared: list[tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int]]] = []
    for info in infos:
        halo = halo_cells(g, info["cells"])
        for r, c in halo:
            if out[r, c] == BLUE:
                out[r, c] = ORANGE
        if info["direction"] is None:
            continue
        tile, box = make_tile(info["cells"], halo)
        prepared.append((tile, box, info["direction"]))

    for tile, box, direction in prepared:
        apply_tiling(out, tile, box, direction)

    return out


def full_blue_neighbors(shape: tuple[int, int], cells: list[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    for r, c in cells:
        for nr, nc in neighbors8(r, c):
            if not in_bounds(shape, nr, nc):
                return False
            if (nr, nc) in cell_set:
                continue
    return True


def choose_defect(shape: np.ndarray, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    h, w = shape.shape
    options: list[list[tuple[int, int]]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            cells = [(r, c)]
            if all(shape[nr, nc] for nr, nc in neighbors8(r, c)):
                options.append(cells)
    for r in range(1, h - 1):
        for c in range(1, w - 2):
            cells = [(r, c), (r, c + 1)]
            ok = True
            cell_set = set(cells)
            for rr, cc in cells:
                for nr, nc in neighbors8(rr, cc):
                    if not in_bounds(shape.shape, nr, nc):
                        ok = False
                        break
                    if (nr, nc) in cell_set:
                        continue
                    if not shape[nr, nc]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                options.append(cells)
    for r in range(1, h - 2):
        for c in range(1, w - 1):
            cells = [(r, c), (r + 1, c)]
            ok = True
            cell_set = set(cells)
            for rr, cc in cells:
                for nr, nc in neighbors8(rr, cc):
                    if not in_bounds(shape.shape, nr, nc):
                        ok = False
                        break
                    if (nr, nc) in cell_set:
                        continue
                    if not shape[nr, nc]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                options.append(cells)
    if not options:
        return None
    return options[int(rng.integers(len(options)))]


def side_runs(shape: np.ndarray, direction: str) -> list[int]:
    h, w = shape.shape
    runs: list[int] = []
    if direction == "up":
        for c in range(w - 1):
            if shape[0, c] and shape[0, c + 1]:
                runs.append(c)
    elif direction == "down":
        for c in range(w - 1):
            if shape[h - 1, c] and shape[h - 1, c + 1]:
                runs.append(c)
    elif direction == "left":
        for r in range(h - 1):
            if shape[r, 0] and shape[r + 1, 0]:
                runs.append(r)
    elif direction == "right":
        for r in range(h - 1):
            if shape[r, w - 1] and shape[r + 1, w - 1]:
                runs.append(r)
    return runs


def build_component_shape(
    rng: np.random.Generator,
    with_defect: bool,
    direction: str | None,
) -> tuple[np.ndarray, list[tuple[int, int]], int | None]:
    dims = [(5, 5), (5, 6), (6, 5)]
    for _ in range(300):
        h, w = dims[int(rng.integers(len(dims)))]
        base = np.ones((h, w), dtype=bool)
        defect = choose_defect(base, rng) if with_defect else []
        if with_defect and not defect:
            continue

        protected: set[tuple[int, int]] = set()
        for r, c in defect:
            protected.add((r, c))
            for nr, nc in neighbors8(r, c):
                if in_bounds(base.shape, nr, nc):
                    protected.add((nr, nc))

        boundary = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if (r in (0, h - 1) or c in (0, w - 1)) and (r, c) not in protected
        ]
        rng.shuffle(boundary)
        remove_target = int(rng.integers(1, 5))
        shape = base.copy()
        removed = 0
        for r, c in boundary:
            if removed >= remove_target:
                break
            candidate = shape.copy()
            candidate[r, c] = False
            blue_candidate = candidate.copy()
            for rr, cc in defect:
                blue_candidate[rr, cc] = False
            if blue_candidate.sum() < 15 or blue_candidate.sum() > 30:
                continue
            if not is_connected(blue_candidate):
                continue
            if with_defect:
                still_valid = True
                defect_set = set(defect)
                for rr, cc in defect:
                    for nr, nc in neighbors8(rr, cc):
                        if not in_bounds(candidate.shape, nr, nc):
                            still_valid = False
                            break
                        if (nr, nc) in defect_set:
                            continue
                        if not candidate[nr, nc]:
                            still_valid = False
                            break
                    if not still_valid:
                        break
                if not still_valid:
                    continue
            if direction is not None and not side_runs(candidate, direction):
                continue
            shape = candidate
            removed += 1

        blue_shape = shape.copy()
        for rr, cc in defect:
            blue_shape[rr, cc] = False
        if blue_shape.sum() < 15 or blue_shape.sum() > 30:
            continue
        if not is_connected(blue_shape):
            continue
        if direction is not None:
            runs = side_runs(shape, direction)
            if not runs:
                continue
            run = runs[int(rng.integers(len(runs)))]
        else:
            run = None
        return shape, defect, run
    raise RuntimeError("failed to build component shape")


def slot_ranges(h: int, w: int) -> dict[str, tuple[int, int, int, int]]:
    return {
        "up": (h - 8, 1, h - 2, 8),
        "down": (1, w - 9, 7, w - 2),
        "right": (1, 1, 7, 8),
        "left": (h - 8, w - 9, h - 2, w - 2),
        "plain": (max(1, h // 2 - 3), max(1, w // 2 - 4), min(h - 2, h // 2 + 3), min(w - 2, w // 2 + 4)),
    }


def place_component(
    grid: np.ndarray,
    rng: np.random.Generator,
    slot: tuple[int, int, int, int],
    shape: np.ndarray,
    defect: list[tuple[int, int]],
    direction: str | None,
    cap_run: int | None,
) -> dict | None:
    sr0, sc0, sr1, sc1 = slot
    h, w = shape.shape

    r_min = sr0
    r_max = sr1 - h + 1
    c_min = sc0
    c_max = sc1 - w + 1
    if direction == "up":
        r_min = max(r_min, sr0 + 1)
    elif direction == "down":
        r_max = min(r_max, sr1 - h)
    elif direction == "left":
        c_min = max(c_min, sc0 + 1)
    elif direction == "right":
        c_max = min(c_max, sc1 - w)

    if r_min > r_max or c_min > c_max:
        return None

    candidates = [(r, c) for r in range(r_min, r_max + 1) for c in range(c_min, c_max + 1)]
    rng.shuffle(candidates)
    for top, left in candidates:
        coords = [(top + r, left + c) for r in range(h) for c in range(w) if shape[r, c]]
        if any(grid[r, c] != BLACK for r, c in coords):
            continue

        cap_cells: list[tuple[int, int]] = []
        if direction == "up":
            cap_cells = [(top - 1, left + cap_run), (top - 1, left + cap_run + 1)]
        elif direction == "down":
            cap_cells = [(top + h, left + cap_run), (top + h, left + cap_run + 1)]
        elif direction == "left":
            cap_cells = [(top + cap_run, left - 1), (top + cap_run + 1, left - 1)]
        elif direction == "right":
            cap_cells = [(top + cap_run, left + w), (top + cap_run + 1, left + w)]

        if cap_cells:
            if any(not in_bounds(grid.shape, r, c) or grid[r, c] != BLACK for r, c in cap_cells):
                continue
            # Cap must touch only this component in 4-neighborhood.
            touched = 0
            for r, c in cap_cells:
                for nr, nc in neighbors4(r, c):
                    if (nr, nc) in coords:
                        touched += 1
            if touched == 0:
                continue

        for r, c in coords:
            if (r - top, c - left) in defect:
                continue
            grid[r, c] = BLUE
        for dr, dc in defect:
            grid[top + dr, left + dc] = BLACK
        for r, c in cap_cells:
            grid[r, c] = GREY

        return {
            "top": top,
            "left": left,
            "shape": shape,
            "defect": [(top + r, left + c) for r, c in defect],
            "direction": direction,
        }
    return None


def intended_layout(rng: np.random.Generator) -> tuple[str, list[str], bool]:
    if int(rng.integers(2)) == 0:
        mode = "vertical"
        defect_dirs = ["up", "down"]
    else:
        mode = "horizontal"
        defect_dirs = ["left", "right"]

    pattern = int(rng.integers(4))
    if pattern == 0:
        chosen = [defect_dirs[0], defect_dirs[1]]
        include_plain = True
    elif pattern == 1:
        chosen = [defect_dirs[int(rng.integers(2))]]
        include_plain = True
    elif pattern == 2:
        chosen = [defect_dirs[0], defect_dirs[1]]
        include_plain = False
    else:
        chosen = [defect_dirs[int(rng.integers(2))]]
        include_plain = bool(int(rng.integers(2)))
    return mode, chosen, include_plain


def copy_targets(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    targets: list[set[tuple[int, int]]] = []
    for info in analyze_scene(grid):
        if info["direction"] is None:
            continue
        halo = halo_cells(grid, info["cells"])
        tile, box = make_tile(info["cells"], halo)
        dr, dc = info["direction"]
        step = tile.shape[0] if dr != 0 else tile.shape[1]
        r0, c0, _r1, _c1 = box
        cells: set[tuple[int, int]] = set()
        k = 1
        while True:
            placed_any = False
            ar = r0 + k * step * dr
            ac = c0 + k * step * dc
            for tr in range(tile.shape[0]):
                for tc in range(tile.shape[1]):
                    if tile[tr, tc] < 0:
                        continue
                    rr = ar + tr
                    cc = ac + tc
                    if not in_bounds(grid.shape, rr, cc):
                        continue
                    placed_any = True
                    if int(grid[rr, cc]) not in (BLUE, GREY, ORANGE):
                        cells.add((rr, cc))
            if not placed_any:
                break
            k += 1
        targets.append(cells)
    return targets


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        h = int(rng.integers(16, 21))
        w = int(rng.integers(16, 21))
        grid = np.zeros((h, w), dtype=int)
        slots = slot_ranges(h, w)
        mode, defect_dirs, include_plain = intended_layout(rng)

        placed: list[dict] = []
        ok = True
        for direction in defect_dirs:
            shape, defect, run = build_component_shape(rng, with_defect=True, direction=direction)
            placed_info = place_component(grid, rng, slots[direction], shape, defect, direction, run)
            if placed_info is None:
                ok = False
                break
            placed.append(placed_info)
        if not ok:
            continue

        if include_plain:
            shape, defect, run = build_component_shape(rng, with_defect=False, direction=None)
            placed_info = place_component(grid, rng, slots["plain"], shape, defect, None, run)
            if placed_info is None:
                continue
            placed.append(placed_info)

        infos = analyze_scene(grid)
        if len(infos) != len(defect_dirs):
            continue
        if any(info["direction"] is None for info in infos):
            continue
        if len(connected_components(grid == BLUE, connectivity=4)) != len(placed):
            continue

        target_sets = copy_targets(grid)
        if not target_sets or any(len(s) == 0 for s in target_sets):
            continue
        union: set[tuple[int, int]] = set()
        disjoint = True
        for cell_set in target_sets:
            if union & cell_set:
                disjoint = False
                break
            union |= cell_set
        if not disjoint:
            continue

        out = main(grid)
        if np.array_equal(out, grid):
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
