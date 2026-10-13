from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GRAY = 5
ORANGE = 7

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RECT_SHAPES = [(2, 2), (2, 3), (3, 2), (3, 3)]
MAX_GENERATION_ATTEMPTS = 500


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    comps = []
    for coords in component_coords(grid == color):
        comps.append(
            {
                "coords": coords,
                "bbox": bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return comps


def is_solid_rectangle(grid: np.ndarray, color: int, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color))


def marker_box(grid: np.ndarray) -> tuple[int, int, int, int]:
    markers = []
    for comp in find_color_components(grid, RED):
        r0, c0, r1, c1 = comp["bbox"]
        if comp["area"] == 4 and (r1 - r0 + 1, c1 - c0 + 1) == (2, 2) and is_solid_rectangle(grid, RED, comp["bbox"]):
            markers.append(comp["bbox"])
    assert len(markers) == 1, "expected exactly one solid 2x2 red marker"
    return markers[0]


def gray_orientation(gray_coords: np.ndarray, contacts: list[tuple[int, int, int, int]]) -> str:
    r0, c0, r1, c1 = bbox_coords(gray_coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h == 1 and w >= 2:
        return "horizontal"
    if w == 1 and h >= 2:
        return "vertical"
    assert h == 1 and w == 1, "gray bar must be a straight segment"
    kinds = set()
    for tr, tc, sr, sc in contacts:
        if tr != sr and tc == sc:
            kinds.add("horizontal")
        elif tc != sc and tr == sr:
            kinds.add("vertical")
        else:
            raise AssertionError("gray bar must 4-touch its container by a side")
    assert len(kinds) == 1, "singleton gray bar has ambiguous orientation"
    return next(iter(kinds))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the marker-selected lattice bands and only let band-covered containers convert adjacent gray bars into red interior stripes.

    Input structure:
    - The background is black, blue pixels form a repeated lattice, and one red 2x2 block replaces a lattice intersection.
    - Orange objects are solid rectangular containers placed on black lattice gaps, and some containers 4-touch one-cell-thick gray bars.
    - Each gray bar is a straight horizontal or vertical segment touching exactly one container.

    Rule:
    - Find the red 2x2 marker and read its two rows and two columns.
    - Recolor every blue lattice pixel on those rows or columns to red.
    - A container is activated iff any of its pixels lies on one of the marker rows or one of the marker columns.
    - Erase every gray bar to black. For each activated container with adjacent gray bars, count the total gray pixels; horizontal bars fill that many full rows from the touched top or bottom edge inward, and vertical bars fill that many full columns from the touched left or right edge inward, using red. Unactivated containers never receive stripes.

    Key insight:
    - The same marker band has two linked effects: it recolors the lattice itself and, separately, gates whether a nearby gray-to-stripe transfer is allowed inside each orange container.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mr0, mc0, mr1, mc1 = marker_box(g)
    marker_rows = {mr0, mr1}
    marker_cols = {mc0, mc1}

    containers = find_color_components(g, ORANGE)
    assert 3 <= len(containers) <= 5, "expected 3 to 5 orange containers"

    orange_lookup: dict[tuple[int, int], int] = {}
    container_boxes: list[tuple[int, int, int, int]] = []
    for idx, comp in enumerate(containers):
        box = comp["bbox"]
        assert is_solid_rectangle(g, ORANGE, box), "orange objects must be solid rectangles"
        container_boxes.append(box)
        for rc in comp["coords"]:
            orange_lookup[(int(rc[0]), int(rc[1]))] = idx

    gray_groups = find_color_components(g, GRAY)
    sources_by_container: dict[int, list[tuple[dict, list[tuple[int, int, int, int]]]]] = {}
    for comp in gray_groups:
        touched: dict[int, list[tuple[int, int, int, int]]] = {}
        for sr_raw, sc_raw in comp["coords"]:
            sr = int(sr_raw)
            sc = int(sc_raw)
            for tr, tc in neighbors4(sr, sc):
                idx = orange_lookup.get((tr, tc))
                if idx is not None:
                    touched.setdefault(idx, []).append((tr, tc, sr, sc))
        assert len(touched) == 1, "each gray bar must touch exactly one container"
        idx = next(iter(touched))
        sources_by_container.setdefault(idx, []).append((comp, touched[idx]))

    out = g.copy()
    band_mask = np.zeros(g.shape, dtype=bool)
    band_mask[list(marker_rows), :] = True
    band_mask[:, list(marker_cols)] = True
    out[(g == BLUE) & band_mask] = RED
    out[g == GRAY] = BLACK

    for idx, box in enumerate(container_boxes):
        r0, c0, r1, c1 = box
        active = any(r in marker_rows for r in range(r0, r1 + 1)) or any(c in marker_cols for c in range(c0, c1 + 1))
        if idx not in sources_by_container or not active:
            continue

        orientations = {gray_orientation(src["coords"], contacts) for src, contacts in sources_by_container[idx]}
        assert len(orientations) == 1, "gray bars around one container must share one orientation"
        orientation = next(iter(orientations))
        count = sum(int(src["area"]) for src, _ in sources_by_container[idx])

        if orientation == "horizontal":
            assert count <= (r1 - r0 + 1), "horizontal bar count must fit container height"
            contact_rows = {tr for _src, contacts in sources_by_container[idx] for tr, _tc, _sr, _sc in contacts}
            assert contact_rows in ({r0}, {r1}), "horizontal bars must touch exactly one extreme row"
            if contact_rows == {r0}:
                out[r0 : r0 + count, c0 : c1 + 1] = RED
            else:
                out[r1 - count + 1 : r1 + 1, c0 : c1 + 1] = RED
        else:
            assert count <= (c1 - c0 + 1), "vertical bar count must fit container width"
            contact_cols = {tc for _src, contacts in sources_by_container[idx] for _tr, tc, _sr, _sc in contacts}
            assert contact_cols in ({c0}, {c1}), "vertical bars must touch exactly one extreme column"
            if contact_cols == {c0}:
                out[r0 : r1 + 1, c0 : c0 + count] = RED
            else:
                out[r0 : r1 + 1, c1 - count + 1 : c1 + 1] = RED

    return out


def build_lattice() -> tuple[np.ndarray, list[list[int]], list[list[int]]]:
    n_row_bands = random.choice([4, 5])
    n_col_bands = random.choice([4, 5])
    row_gaps = [random.choice([2, 3]) for _ in range(n_row_bands + 1)]
    col_gaps = [random.choice([2, 3]) for _ in range(n_col_bands + 1)]

    h = 2 * n_row_bands + sum(row_gaps)
    w = 2 * n_col_bands + sum(col_gaps)
    grid = np.full((h, w), BLACK, dtype=int)

    row_bands: list[list[int]] = []
    cur = row_gaps[0]
    for i in range(n_row_bands):
        row_bands.append([cur, cur + 1])
        cur += 2 + row_gaps[i + 1]

    col_bands: list[list[int]] = []
    cur = col_gaps[0]
    for i in range(n_col_bands):
        col_bands.append([cur, cur + 1])
        cur += 2 + col_gaps[i + 1]

    row_mask = np.zeros(h, dtype=bool)
    col_mask = np.zeros(w, dtype=bool)
    for band in row_bands:
        row_mask[band] = True
    for band in col_bands:
        col_mask[band] = True
    grid[np.ix_(row_mask, col_mask)] = BLUE
    return grid, row_bands, col_bands


def enumerate_rect_candidates(
    grid: np.ndarray,
    marker_rows: set[int],
    marker_cols: set[int],
) -> list[dict]:
    h, w = grid.shape
    candidates = []
    for rh, cw in RECT_SHAPES:
        for r0 in range(h - rh + 1):
            for c0 in range(w - cw + 1):
                r1 = r0 + rh - 1
                c1 = c0 + cw - 1
                if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BLACK):
                    continue
                hit_row = any(r in marker_rows for r in range(r0, r1 + 1))
                hit_col = any(c in marker_cols for c in range(c0, c1 + 1))
                if hit_row and hit_col:
                    continue
                mode = "inactive"
                if hit_row:
                    mode = "row"
                elif hit_col:
                    mode = "col"
                coords = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
                candidates.append(
                    {
                        "bbox": (r0, c0, r1, c1),
                        "shape": (rh, cw),
                        "mode": mode,
                        "cells": coords,
                    }
                )
    return candidates


def boxes_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def choose_candidates(pool: list[dict], count: int, chosen: list[dict]) -> list[dict] | None:
    order = pool[:]
    random.shuffle(order)
    picked: list[dict] = []
    for cand in order:
        if any(boxes_conflict(cand["bbox"], other["bbox"]) for other in chosen + picked):
            continue
        picked.append(cand)
        if len(picked) == count:
            return picked
    return None


def pick_containers(
    grid: np.ndarray,
    marker_rows: set[int],
    marker_cols: set[int],
) -> list[dict] | None:
    candidates = enumerate_rect_candidates(grid, marker_rows, marker_cols)
    row_pool = [cand for cand in candidates if cand["mode"] == "row"]
    col_pool = [cand for cand in candidates if cand["mode"] == "col"]
    inactive_pool = [cand for cand in candidates if cand["mode"] == "inactive"]
    if not inactive_pool:
        return None

    total = random.randint(3, 5)
    active_count = random.randint(2, min(4, total - 1))
    inactive_count = total - active_count
    if len(inactive_pool) < inactive_count:
        return None

    mode_plans = []
    for row_need in range(active_count + 1):
        col_need = active_count - row_need
        if row_need <= len(row_pool) and col_need <= len(col_pool):
            if row_need > 0 and col_need > 0:
                mode_plans.extend([(row_need, col_need)] * 3)
            else:
                mode_plans.append((row_need, col_need))
    if not mode_plans:
        return None
    random.shuffle(mode_plans)

    for row_need, col_need in mode_plans:
        chosen: list[dict] = []
        row_pick = choose_candidates(row_pool, row_need, chosen)
        if row_pick is None:
            continue
        chosen.extend(row_pick)
        col_pick = choose_candidates(col_pool, col_need, chosen)
        if col_pick is None:
            continue
        chosen.extend(col_pick)
        inactive_pick = choose_candidates(inactive_pool, inactive_count, chosen)
        if inactive_pick is None:
            continue
        chosen.extend(inactive_pick)
        random.shuffle(chosen)
        return chosen
    return None


def bar_touches_only_owner(
    cells: list[tuple[int, int]],
    owner_cells: set[tuple[int, int]],
    occupied_gray: set[tuple[int, int]],
    container_lookup: dict[tuple[int, int], int],
    owner_id: int,
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    saw_touch = False
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied_gray:
            return False
        for nr, nc in neighbors4(r, c):
            idx = container_lookup.get((nr, nc))
            if idx is None:
                continue
            if idx != owner_id:
                return False
            if (nr, nc) in owner_cells:
                saw_touch = True
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in occupied_gray:
                return False
    return saw_touch


def enumerate_bar_options(
    base_grid: np.ndarray,
    container: dict,
    container_id: int,
    occupied_gray: set[tuple[int, int]],
    container_lookup: dict[tuple[int, int], int],
) -> list[dict]:
    h, w = base_grid.shape
    r0, c0, r1, c1 = container["bbox"]
    owner_cells = container["cells"]
    options: list[dict] = []

    for length in range(1, min(3, r1 - r0 + 1) + 1):
        for start in range(c0, c1 - length + 2):
            cells = [(r0 - 1, c) for c in range(start, start + length)]
            if r0 - 1 >= 0 and all(base_grid[r, c] == BLACK for r, c in cells):
                if bar_touches_only_owner(cells, owner_cells, occupied_gray, container_lookup, container_id, (h, w)):
                    options.append({"orientation": "horizontal", "side": "top", "cells": cells, "length": length})
            cells = [(r1 + 1, c) for c in range(start, start + length)]
            if r1 + 1 < h and all(base_grid[r, c] == BLACK for r, c in cells):
                if bar_touches_only_owner(cells, owner_cells, occupied_gray, container_lookup, container_id, (h, w)):
                    options.append({"orientation": "horizontal", "side": "bottom", "cells": cells, "length": length})

    for length in range(1, min(3, c1 - c0 + 1) + 1):
        for start in range(r0, r1 - length + 2):
            cells = [(r, c0 - 1) for r in range(start, start + length)]
            if c0 - 1 >= 0 and all(base_grid[r, c] == BLACK for r, c in cells):
                if bar_touches_only_owner(cells, owner_cells, occupied_gray, container_lookup, container_id, (h, w)):
                    options.append({"orientation": "vertical", "side": "left", "cells": cells, "length": length})
            cells = [(r, c1 + 1) for r in range(start, start + length)]
            if c1 + 1 < w and all(base_grid[r, c] == BLACK for r, c in cells):
                if bar_touches_only_owner(cells, owner_cells, occupied_gray, container_lookup, container_id, (h, w)):
                    options.append({"orientation": "vertical", "side": "right", "cells": cells, "length": length})
    return options


def pick_bar_plan(base_grid: np.ndarray, containers: list[dict]) -> list[dict] | None:
    container_lookup: dict[tuple[int, int], int] = {}
    for idx, container in enumerate(containers):
        for cell in container["cells"]:
            container_lookup[cell] = idx

    active_ids = [i for i, c in enumerate(containers) if c["mode"] in {"row", "col"}]
    inactive_ids = [i for i, c in enumerate(containers) if c["mode"] == "inactive"]
    if not active_ids or not inactive_ids:
        return None

    bar_ids = set(active_ids + inactive_ids)
    if len(containers) >= 4 and random.random() < 0.5:
        removable = [i for i in range(len(containers)) if i not in {active_ids[0], inactive_ids[0]}]
        if removable:
            bar_ids.discard(random.choice(removable))

    order = list(bar_ids)
    random.shuffle(order)
    occupied_gray: set[tuple[int, int]] = set()
    plan: list[dict] = []
    forced_orients: list[str] = []
    if len(order) >= 2:
        forced_orients = ["horizontal", "vertical"]

    for step, idx in enumerate(order):
        container = containers[idx]
        options = enumerate_bar_options(base_grid, container, idx, occupied_gray, container_lookup)
        if not options:
            return None
        if step < len(forced_orients):
            subset = [opt for opt in options if opt["orientation"] == forced_orients[step]]
            if subset:
                options = subset
        random.shuffle(options)
        chosen = options[0]
        occupied_gray.update(chosen["cells"])
        plan.append(
            {
                "container_id": idx,
                "orientation": chosen["orientation"],
                "side": chosen["side"],
                "cells": chosen["cells"],
                "length": chosen["length"],
            }
        )

    active_with_bars = {item["container_id"] for item in plan if containers[item["container_id"]]["mode"] in {"row", "col"}}
    inactive_with_bars = {item["container_id"] for item in plan if containers[item["container_id"]]["mode"] == "inactive"}
    if not active_with_bars or not inactive_with_bars:
        return None
    return plan


def render_input(base_grid: np.ndarray, marker_box_: tuple[int, int, int, int], containers: list[dict], bar_plan: list[dict]) -> np.ndarray:
    grid = base_grid.copy()
    mr0, mc0, mr1, mc1 = marker_box_
    grid[mr0 : mr1 + 1, mc0 : mc1 + 1] = RED
    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = ORANGE
    for item in bar_plan:
        for r, c in item["cells"]:
            grid[r, c] = GRAY
    return grid


def render_expected(
    inp: np.ndarray,
    marker_rows: set[int],
    marker_cols: set[int],
    containers: list[dict],
    bar_plan: list[dict],
    activate_all: bool = False,
    fill_bars: bool = True,
    recolor_bands: bool = True,
) -> np.ndarray:
    out = inp.copy()
    out[inp == GRAY] = BLACK
    if recolor_bands:
        band_mask = np.zeros(inp.shape, dtype=bool)
        band_mask[list(marker_rows), :] = True
        band_mask[:, list(marker_cols)] = True
        out[(inp == BLUE) & band_mask] = RED

    by_container: dict[int, dict] = {item["container_id"]: item for item in bar_plan}
    for idx, container in enumerate(containers):
        item = by_container.get(idx)
        if item is None or not fill_bars:
            continue
        active = container["mode"] in {"row", "col"}
        if not (active or activate_all):
            continue
        r0, c0, r1, c1 = container["bbox"]
        n = item["length"]
        if item["orientation"] == "horizontal":
            if item["side"] == "top":
                out[r0 : r0 + n, c0 : c1 + 1] = RED
            else:
                out[r1 - n + 1 : r1 + 1, c0 : c1 + 1] = RED
        else:
            if item["side"] == "left":
                out[r0 : r1 + 1, c0 : c0 + n] = RED
            else:
                out[r0 : r1 + 1, c1 - n + 1 : c1 + 1] = RED
    return out


def validate_scene(
    inp: np.ndarray,
    expected: np.ndarray,
    containers: list[dict],
    bar_plan: list[dict],
    marker_rows: set[int],
    marker_cols: set[int],
) -> bool:
    active_count = sum(container["mode"] in {"row", "col"} for container in containers)
    if not (2 <= active_count <= 4):
        return False

    active_bar = any(containers[item["container_id"]]["mode"] in {"row", "col"} for item in bar_plan)
    inactive_bar = any(containers[item["container_id"]]["mode"] == "inactive" for item in bar_plan)
    if not (active_bar and inactive_bar):
        return False

    if np.array_equal(expected, render_expected(inp, marker_rows, marker_cols, containers, bar_plan, activate_all=True)):
        return False
    if np.array_equal(expected, render_expected(inp, marker_rows, marker_cols, containers, bar_plan, fill_bars=False)):
        return False
    if np.array_equal(expected, render_expected(inp, marker_rows, marker_cols, containers, bar_plan, recolor_bands=False)):
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        base_grid, row_bands, col_bands = build_lattice()
        marker_rows = set(random.choice(row_bands))
        marker_cols = set(random.choice(col_bands))
        marker = (min(marker_rows), min(marker_cols), max(marker_rows), max(marker_cols))

        containers = pick_containers(base_grid, marker_rows, marker_cols)
        if containers is None:
            continue

        bar_plan = pick_bar_plan(base_grid, containers)
        if bar_plan is None:
            continue

        inp = render_input(base_grid, marker, containers, bar_plan)
        expected = render_expected(inp, marker_rows, marker_cols, containers, bar_plan)
        if validate_scene(inp, expected, containers, bar_plan, marker_rows, marker_cols):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 6")
