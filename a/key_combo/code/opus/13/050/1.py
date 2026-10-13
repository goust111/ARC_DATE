from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GRID_SIZES = (13, 15, 17)
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_cells(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _mirror_positions(shape: tuple[int, int], r: int, c: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    h, w = shape
    return (
        (h - 1 - r, c),
        (r, w - 1 - c),
        (h - 1 - r, w - 1 - c),
    )


def _find_marker_rect(g: np.ndarray) -> tuple[int, int, int, int]:
    mask = g == GREY
    box = _bbox(mask)
    assert box is not None, "expected a grey marker rectangle"
    r0, c0, r1, c1 = box
    assert int(mask.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1), "grey cells must form one solid rectangle"
    return r0, c0, r1, c1


def _restore_barrier(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    r0, c0, r1, c1 = _find_marker_rect(g)

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            visible = []
            for rr, cc in _mirror_positions(g.shape, r, c):
                value = int(g[rr, cc])
                if value != GREY:
                    visible.append(value)
            assert visible, "each grey cell must have a visible mirror counterpart"
            assert set(visible).issubset({BLACK, RED}), "mirror references must reveal only the hidden barrier/background"
            assert len(set(visible)) == 1, "visible mirror references must agree"
            out[r, c] = visible[0]

    return out


def _flood_from_blue(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    seeds = [tuple(map(int, rc)) for rc in np.argwhere(g == BLUE)]
    assert seeds, "expected at least one blue seed"

    q: deque[tuple[int, int]] = deque(seeds)
    seen = np.zeros(g.shape, dtype=bool)
    for r, c in seeds:
        seen[r, c] = True

    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if not _in_bounds(g.shape, rr, cc) or seen[rr, cc]:
                continue
            if int(g[rr, cc]) != BLACK:
                continue
            seen[rr, cc] = True
            out[rr, cc] = BLUE
            q.append((rr, cc))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the grey-occluded symmetric red barrier, then flood-fill black background from the blue seeds.

    Input structure:
    - A square black grid contains a connected red barrier pattern that is symmetric across the horizontal and vertical center lines.
    - One solid grey rectangle hides a mix of red barrier cells and black background cells.
    - Two to four isolated blue seed pixels lie on visible black background outside the grey marker.

    Rule:
    - Find the grey rectangle and, for each covered cell, inspect its horizontal mirror, vertical mirror, and 180-degree counterpart.
    - Replace the grey cell with the unique visible non-grey value shared by those counterparts, restoring the red/black symmetric barrier layout.
    - Starting from every blue seed, flood through 4-connected black cells only and recolor every reached black cell blue.
    - Keep all original blue seeds and all red barrier cells unchanged.

    Key insight:
    - The grey patch is not a blocker itself; it hides part of a mirror-symmetric red wall, and only after reconstructing that wall does the blue flood have the correct boundaries.
    """
    repaired = _restore_barrier(np.asarray(grid, dtype=int))
    return _flood_from_blue(repaired)


def _black_components(grid: np.ndarray) -> tuple[list[set[tuple[int, int]]], np.ndarray]:
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=int)
    comps: list[set[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BLACK or labels[r, c] != -1:
                continue
            comp_id = len(comps)
            comp: set[tuple[int, int]] = set()
            q: deque[tuple[int, int]] = deque([(r, c)])
            labels[r, c] = comp_id
            while q:
                rr, cc = q.popleft()
                comp.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if int(grid[nr, nc]) != BLACK or labels[nr, nc] != -1:
                        continue
                    labels[nr, nc] = comp_id
                    q.append((nr, nc))
            comps.append(comp)

    return comps, labels


def _build_barrier_grid(size: int, half_h: int, half_w: int) -> tuple[np.ndarray, dict[str, int]]:
    g = np.zeros((size, size), dtype=int)
    cy = cx = size // 2
    top, bottom = cy - half_h, cy + half_h
    left, right = cx - half_w, cx + half_w

    g[top, left : right + 1] = RED
    g[bottom, left : right + 1] = RED
    g[top : bottom + 1, left] = RED
    g[top : bottom + 1, right] = RED
    g[top : bottom + 1, cx] = RED
    g[cy, left : right + 1] = RED

    return g, {"cy": cy, "cx": cx, "top": top, "bottom": bottom, "left": left, "right": right}


def _choose_mask(meta: dict[str, int]) -> tuple[np.ndarray, dict[str, object]]:
    cy = meta["cy"]
    cx = meta["cx"]
    top = meta["top"]
    bottom = meta["bottom"]
    left = meta["left"]
    right = meta["right"]
    half_h = cy - top
    half_w = cx - left

    mode = random.choice(("outer", "spoke_v", "spoke_h"))
    info: dict[str, object] = {"mode": mode}

    if mode == "outer":
        side = random.choice(("top", "bottom", "left", "right"))
        half = random.choice(("first", "second"))
        info["side"] = side

        if side in ("top", "bottom"):
            width = random.randint(3, max(3, half_w - 1))
            row0 = top - 1 if side == "top" else bottom - 1
            if half == "first":
                c_min, c_max = left + 1, cx - width
            else:
                c_min, c_max = cx + 1, right - width
            col0 = random.randint(c_min, c_max)
            if side == "top":
                leaked = "NW" if half == "first" else "NE"
            else:
                leaked = "SW" if half == "first" else "SE"
            rect = (row0, col0, row0 + 2, col0 + width - 1)
        else:
            height = random.randint(3, max(3, half_h - 1))
            col0 = left - 1 if side == "left" else right - 1
            if half == "first":
                r_min, r_max = top + 1, cy - height
            else:
                r_min, r_max = cy + 1, bottom - height
            row0 = random.randint(r_min, r_max)
            if side == "left":
                leaked = "NW" if half == "first" else "SW"
            else:
                leaked = "NE" if half == "first" else "SE"
            rect = (row0, col0, row0 + height - 1, col0 + 2)

        info["leaked_region"] = leaked

    elif mode == "spoke_v":
        width = 3
        half = random.choice(("top", "bottom"))
        info["half"] = half
        if half == "top":
            height = random.randint(3, max(3, half_h - 1))
            row0 = random.randint(top + 1, cy - height)
            pair = ("NW", "NE")
        else:
            height = random.randint(3, max(3, half_h - 1))
            row0 = random.randint(cy + 1, bottom - height)
            pair = ("SW", "SE")
        rect = (row0, cx - 1, row0 + height - 1, cx + 1)
        info["pair"] = pair

    else:
        height = 3
        half = random.choice(("left", "right"))
        info["half"] = half
        if half == "left":
            width = random.randint(3, max(3, half_w - 1))
            col0 = random.randint(left + 1, cx - width)
            pair = ("NW", "SW")
        else:
            width = random.randint(3, max(3, half_w - 1))
            col0 = random.randint(cx + 1, right - width)
            pair = ("NE", "SE")
        rect = (cy - 1, col0, cy + 1, col0 + width - 1)
        info["pair"] = pair

    r0, c0, r1, c1 = rect
    mask = np.zeros((2 * cy + 1, 2 * cx + 1), dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask, info


def _reference_cells_for_mask(mask: np.ndarray) -> set[tuple[int, int]]:
    refs: set[tuple[int, int]] = set()
    hidden = _mask_cells(mask)
    for r, c in hidden:
        for cell in _mirror_positions(mask.shape, r, c):
            if cell not in hidden:
                refs.add(cell)
    return refs


def _pick_region_cells(base: np.ndarray, meta: dict[str, int]) -> dict[str, set[tuple[int, int]]]:
    comps, labels = _black_components(base)
    regions = {
        "OUT": (0, 0),
        "NW": (meta["top"] + 1, meta["left"] + 1),
        "NE": (meta["top"] + 1, meta["cx"] + 1),
        "SW": (meta["cy"] + 1, meta["left"] + 1),
        "SE": (meta["cy"] + 1, meta["cx"] + 1),
    }
    out: dict[str, set[tuple[int, int]]] = {}
    for name, (r, c) in regions.items():
        assert int(base[r, c]) == BLACK
        out[name] = comps[int(labels[r, c])]
    return out


def _choose_seed_regions(info: dict[str, object]) -> list[str]:
    chambers = ["NW", "NE", "SW", "SE"]
    if info["mode"] == "outer":
        seeded = ["OUT"]
        leaked = str(info["leaked_region"])
        options = [name for name in chambers if name != leaked]
    else:
        pair = list(info["pair"])  # type: ignore[index]
        seeded = [random.choice(pair)]
        options = ["OUT"] + [name for name in chambers if name not in pair]

    random.shuffle(options)
    extra_count = random.randint(0, min(2, len(options)))
    seeded.extend(options[:extra_count])
    return list(dict.fromkeys(seeded))


def _select_cells(cells: set[tuple[int, int]], blocked: set[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    usable = [cell for cell in cells if cell not in blocked]
    if len(usable) < count:
        return None

    for _ in range(80):
        chosen: list[tuple[int, int]] = []
        local_blocked = set(blocked)
        order = usable[:]
        random.shuffle(order)
        for cell in order:
            if cell in local_blocked:
                continue
            chosen.append(cell)
            local_blocked.add(cell)
            for nb in _neighbors4(*cell):
                local_blocked.add(nb)
            if len(chosen) == count:
                return chosen
    return None


def _wrong_black_gap_output(inp: np.ndarray) -> np.ndarray:
    wrong = np.asarray(inp, dtype=int).copy()
    wrong[wrong == GREY] = BLACK
    return _flood_from_blue(wrong)


def _validate_generated(inp: np.ndarray, base_with_seeds: np.ndarray, mask: np.ndarray, info: dict[str, object], meta: dict[str, int]) -> bool:
    try:
        repaired = _restore_barrier(inp)
    except Exception:
        return False

    if not np.array_equal(repaired, base_with_seeds):
        return False

    out = main(inp)
    if not np.array_equal(out, _flood_from_blue(base_with_seeds)):
        return False

    if np.array_equal(out, _wrong_black_gap_output(inp)):
        return False

    if np.array_equal(base_with_seeds, out):
        return False

    marker_box = _find_marker_rect(inp)
    if int(mask.sum()) != (marker_box[2] - marker_box[0] + 1) * (marker_box[3] - marker_box[1] + 1):
        return False

    hidden_values = set(map(int, base_with_seeds[mask].tolist()))
    if hidden_values != {BLACK, RED}:
        return False

    base_barrier = base_with_seeds.copy()
    base_barrier[base_barrier == BLUE] = BLACK
    if not np.array_equal(base_barrier, np.flipud(base_barrier)):
        return False
    if not np.array_equal(base_barrier, np.fliplr(base_barrier)):
        return False

    red_cells = {tuple(map(int, rc)) for rc in np.argwhere(base_barrier == RED)}
    if not red_cells:
        return False

    seed_cells = [tuple(map(int, rc)) for rc in np.argwhere(base_with_seeds == BLUE)]
    if not (2 <= len(seed_cells) <= 4):
        return False
    for i, (r, c) in enumerate(seed_cells):
        if int(inp[r, c]) == GREY:
            return False
        for rr, cc in seed_cells[i + 1 :]:
            if abs(r - rr) + abs(c - cc) == 1:
                return False

    refs = _reference_cells_for_mask(mask)
    if any(cell in refs for cell in seed_cells):
        return False

    visible_count_two = False
    for r, c in _mask_cells(mask):
        vals = []
        count = 0
        for rr, cc in _mirror_positions(inp.shape, r, c):
            if not mask[rr, cc]:
                vals.append(int(base_with_seeds[rr, cc]))
                count += 1
        if count == 2:
            visible_count_two = True
        if count == 0 or len(set(vals)) != 1:
            return False

    if info["mode"] != "outer" and not visible_count_two:
        return False

    out_black = int(np.count_nonzero(out == BLACK))
    out_blue = int(np.count_nonzero(out == BLUE))
    if out_black == 0 or out_blue <= len(seed_cells):
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    size = random.choice(GRID_SIZES)
    max_half = size // 2 - 2
    half_h = random.randint(4, max_half)
    half_w = random.randint(4, max_half)

    base, meta = _build_barrier_grid(size, half_h, half_w)
    mask, info = _choose_mask(meta)
    ref_cells = _reference_cells_for_mask(mask)
    region_cells = _pick_region_cells(base, meta)
    seed_regions = _choose_seed_regions(info)

    total_seeds = random.randint(max(2, len(seed_regions)), 4)
    counts = {name: 1 for name in seed_regions}
    while sum(counts.values()) < total_seeds:
        name = random.choice(seed_regions)
        counts[name] += 1

    blocked = set(_mask_cells(mask)) | ref_cells
    seeds: list[tuple[int, int]] = []
    region_order = seed_regions[:]
    random.shuffle(region_order)
    for name in region_order:
        chosen = _select_cells(region_cells[name], blocked, counts[name])
        if chosen is None:
            return None
        for cell in chosen:
            seeds.append(cell)
            blocked.add(cell)
            for nb in _neighbors4(*cell):
                blocked.add(nb)

    base_with_seeds = base.copy()
    for r, c in seeds:
        base_with_seeds[r, c] = BLUE

    inp = base_with_seeds.copy()
    inp[mask] = GREY

    if not _validate_generated(inp, base_with_seeds, mask, info, meta):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-repair flood-fill input")
