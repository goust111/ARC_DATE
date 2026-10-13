import random
from collections import Counter

import numpy as np


BLACK = 0
SKELETON = 8
PALETTE = (2, 3, 4, 6)
ARM_DISTANCES = (2, 3)
GENERATION_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    return _components_from_mask(grid == int(color))


def _classify_bar(coords: list[tuple[int, int]]) -> tuple[str, frozenset[tuple[int, int]]] | None:
    if len(coords) != 2:
        return None
    (r1, c1), (r2, c2) = sorted(coords)
    cells = frozenset(coords)
    if r1 == r2 and abs(c1 - c2) == 1:
        return "h", cells
    if c1 == c2 and abs(r1 - r2) == 1:
        return "v", cells
    return None


def _collect_bars(grid: np.ndarray) -> dict[int, dict[str, set[frozenset[tuple[int, int]]]]]:
    colors = sorted(set(map(int, grid.flatten())) - {BLACK, SKELETON})
    bars: dict[int, dict[str, set[frozenset[tuple[int, int]]]]] = {}
    for color in colors:
        entries = {"h": set(), "v": set()}
        for comp in _components_of_color(grid, color):
            classified = _classify_bar(comp)
            assert classified is not None, "every non-skeleton foreground object must be a 2-pixel bar"
            orient, cells = classified
            entries[orient].add(cells)
        bars[int(color)] = entries
    return bars


def _cross_bar_sets(r: int, c: int, d: int) -> dict[str, frozenset[tuple[int, int]]]:
    return {
        "up": frozenset({(r - d - 1, c), (r - d, c)}),
        "down": frozenset({(r + d, c), (r + d + 1, c)}),
        "left": frozenset({(r, c - d - 1), (r, c - d)}),
        "right": frozenset({(r, c + d), (r, c + d + 1)}),
    }


def _cross_bar_cells(r: int, c: int, d: int) -> set[tuple[int, int]]:
    bars = _cross_bar_sets(r, c, d)
    return set().union(*bars.values())


def _cross_footprint(r: int, c: int, d: int) -> set[tuple[int, int]]:
    cells = {(r, cc) for cc in range(c - d - 1, c + d + 2)}
    cells |= {(rr, c) for rr in range(r - d - 1, r + d + 2)}
    return cells


def _is_hidden_center(
    grid: np.ndarray,
    bars: dict[int, dict[str, set[frozenset[tuple[int, int]]]]],
    r: int,
    c: int,
    color: int,
    d: int,
) -> bool:
    if int(grid[r, c]) != BLACK:
        return False
    for k in range(1, d):
        if int(grid[r - k, c]) != BLACK or int(grid[r + k, c]) != BLACK:
            return False
        if int(grid[r, c - k]) != BLACK or int(grid[r, c + k]) != BLACK:
            return False
    need = _cross_bar_sets(r, c, d)
    return (
        need["up"] in bars[color]["v"]
        and need["down"] in bars[color]["v"]
        and need["left"] in bars[color]["h"]
        and need["right"] in bars[color]["h"]
    )


def _find_hidden_centers(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bars = _collect_bars(g)
    centers: list[tuple[int, int, int, int]] = []
    claimed: dict[tuple[int, int], tuple[int, int]] = {}
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != BLACK:
                continue
            for color in sorted(bars):
                for d in ARM_DISTANCES:
                    footprint = _cross_footprint(r, c, d)
                    if not all(_in_bounds(g.shape, rr, cc) for rr, cc in footprint):
                        continue
                    if _is_hidden_center(g, bars, r, c, color, d):
                        key = (r, c)
                        found = (color, d)
                        assert key not in claimed or claimed[key] == found, "hidden center is ambiguous"
                        claimed[key] = found
                        centers.append((r, c, color, d))
    centers.sort()
    assert centers, "expected at least one hidden cross center"
    return centers


def _find_chambers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    chambers: list[dict] = []
    for comp in _components_from_mask(g == BLACK):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if len(comp) != 9 or r1 - r0 != 2 or c1 - c0 != 2:
            continue
        chambers.append(
            {
                "coords": sorted(comp),
                "center": ((r0 + r1) // 2, (c0 + c1) // 2),
            }
        )
    chambers.sort(key=lambda item: item["center"])
    assert chambers, "expected enclosed 3x3 chambers"
    return chambers


def _nearest_chamber(r: int, c: int, chambers: list[dict]) -> tuple[int, int] | None:
    dists = sorted(
        (abs(r - cr) + abs(c - cc), idx)
        for idx, (cr, cc) in enumerate(ch["center"] for ch in chambers)
    )
    if len(dists) > 1 and dists[0][0] == dists[1][0]:
        return None
    return dists[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed chamber with the color of the hidden cross center that points to it most closely.

    Input structure:
    - Black background contains a cyan lattice skeleton that encloses several 3x3 black chambers.
    - Outside the lattice, colored foreground objects are isolated 2-pixel horizontal or vertical bars.
    - Some same-colored bars form an implied cross around an unchanged black center pixel, with every arm using the same gap length 2 or 3.

    Rule:
    - Detect every hidden cross center by finding a black pixel whose up/down bars are vertical and whose left/right bars are horizontal, all in one color and all starting the same distance from the center.
    - Find each enclosed 3x3 black chamber inside the cyan lattice and measure Manhattan distance from the hidden center to every chamber center pixel.
    - Fill the uniquely nearest chamber with the hidden center's color, leaving the center pixel, the lattice, and all distractor bars unchanged.

    Key insight:
    - The colored bars are not the direct targets: they implicitly encode a missing center, and that invisible center selects the chamber by nearest-distance matching.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert _get_bg(g) == BLACK, "black must be the dominant background"
    centers = _find_hidden_centers(g)
    chambers = _find_chambers(g)

    out = g.copy()
    used_chambers: set[int] = set()
    for r, c, color, _d in centers:
        nearest = _nearest_chamber(r, c, chambers)
        assert nearest is not None, "nearest chamber must be unique"
        _dist, idx = nearest
        assert idx not in used_chambers, "multiple hidden centers selected the same chamber"
        used_chambers.add(idx)
        for rr, cc in chambers[idx]["coords"]:
            out[rr, cc] = int(color)
    return out


def _build_lattice(nr: int, nc: int, top: int, left: int, bottom_margin: int, right_margin: int) -> tuple[np.ndarray, list[dict], dict]:
    lattice_h = 4 * nr + 1
    lattice_w = 4 * nc + 1
    h = top + lattice_h + bottom_margin
    w = left + lattice_w + right_margin
    assert h <= 30 and w <= 30

    grid = np.zeros((h, w), dtype=int)
    bottom = top + lattice_h - 1
    right = left + lattice_w - 1

    for r in range(top, bottom + 1, 4):
        grid[r, left : right + 1] = SKELETON
    for c in range(left, right + 1, 4):
        grid[top : bottom + 1, c] = SKELETON

    chambers: list[dict] = []
    for ir in range(nr):
        for ic in range(nc):
            r0 = top + 1 + 4 * ir
            c0 = left + 1 + 4 * ic
            coords = [(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3)]
            chambers.append({"coords": coords, "center": (r0 + 1, c0 + 1)})

    meta = {"top": top, "left": left, "bottom": bottom, "right": right}
    return grid, chambers, meta


def _cells_are_free(grid: np.ndarray, cells: set[tuple[int, int]]) -> bool:
    return all(_in_bounds(grid.shape, r, c) and int(grid[r, c]) == BLACK for r, c in cells)


def _place_cross(grid: np.ndarray, r: int, c: int, d: int, color: int) -> None:
    for rr, cc in _cross_bar_cells(r, c, d):
        grid[rr, cc] = int(color)


def _outer_side(meta: dict, shape: tuple[int, int], side: str, r: int, c: int) -> bool:
    if side == "top":
        return r < meta["top"]
    if side == "bottom":
        return r > meta["bottom"]
    if side == "left":
        return c < meta["left"]
    if side == "right":
        return c > meta["right"]
    raise ValueError(f"unknown side: {side}")


def _candidate_centers(
    grid: np.ndarray,
    chambers: list[dict],
    meta: dict,
    side: str,
    d: int,
    chamber_idx: int,
    reserved: set[tuple[int, int]],
    used_distances: set[int],
) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if not _outer_side(meta, grid.shape, side, r, c):
                continue
            footprint = _cross_footprint(r, c, d)
            bars = _cross_bar_cells(r, c, d)
            if footprint & reserved:
                continue
            if not _cells_are_free(grid, footprint):
                continue
            nearest_info = _nearest_chamber(r, c, chambers)
            if nearest_info is None:
                continue
            dist, nearest = nearest_info
            if nearest != chamber_idx or dist in used_distances:
                continue
            candidates.append((r, c, dist))
    return candidates


def _touches_other_color(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    h, w = grid.shape
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cell_set:
                continue
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) not in (BLACK, SKELETON):
                return True
    return False


def _candidate_distractor_bars(
    grid: np.ndarray,
    meta: dict,
    reserved: set[tuple[int, int]],
) -> list[tuple[str, list[tuple[int, int]]]]:
    h, w = grid.shape
    out: list[tuple[str, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            if not (_outer_side(meta, grid.shape, "top", r, c)
                    or _outer_side(meta, grid.shape, "bottom", r, c)
                    or _outer_side(meta, grid.shape, "left", r, c)
                    or _outer_side(meta, grid.shape, "right", r, c)):
                continue
            h_cells = [(r, c), (r, c + 1)]
            v_cells = [(r, c), (r + 1, c)]
            for orient, cells in (("h", h_cells), ("v", v_cells)):
                if not all(_in_bounds(grid.shape, rr, cc) for rr, cc in cells):
                    continue
                if set(cells) & reserved:
                    continue
                if any(int(grid[rr, cc]) != BLACK for rr, cc in cells):
                    continue
                if _touches_other_color(grid, cells):
                    continue
                out.append((orient, cells))
    random.shuffle(out)
    return out


def _expected_output(inp: np.ndarray, placements: list[dict], chambers: list[dict]) -> np.ndarray:
    out = inp.copy()
    for item in placements:
        for rr, cc in chambers[item["chamber_idx"]]["coords"]:
            out[rr, cc] = int(item["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    num_crosses = random.randint(1, len(PALETTE))
    colors = random.sample(PALETTE, num_crosses)
    specs = []
    for color in colors:
        specs.append(
            {
                "color": color,
                "d": random.choice(ARM_DISTANCES),
                "side": random.choice(("top", "bottom", "left", "right")),
            }
        )

    top = random.randint(2, 4)
    left = random.randint(2, 4)
    bottom_margin = random.randint(2, 4)
    right_margin = random.randint(2, 4)
    for spec in specs:
        need = 2 * spec["d"] + 3 + random.randint(0, 1)
        if spec["side"] == "top":
            top = max(top, need)
        elif spec["side"] == "bottom":
            bottom_margin = max(bottom_margin, need)
        elif spec["side"] == "left":
            left = max(left, need)
        else:
            right_margin = max(right_margin, need)

    nr = random.randint(2, 3)
    nc = random.randint(2, 3)
    try:
        grid, chambers, meta = _build_lattice(nr, nc, top, left, bottom_margin, right_margin)
    except AssertionError:
        return None

    random.shuffle(specs)
    chamber_order = list(range(len(chambers)))
    random.shuffle(chamber_order)
    reserved: set[tuple[int, int]] = set()
    used_distances: set[int] = set()
    used_chambers: set[int] = set()
    placements: list[dict] = []

    for spec in specs:
        options: list[tuple[int, int, int, int]] = []
        for chamber_idx in chamber_order:
            if chamber_idx in used_chambers:
                continue
            for r, c, dist in _candidate_centers(
                grid, chambers, meta, spec["side"], spec["d"], chamber_idx, reserved, used_distances
            ):
                options.append((chamber_idx, r, c, dist))
        if not options:
            return None
        chamber_idx, r, c, dist = random.choice(options)
        _place_cross(grid, r, c, spec["d"], spec["color"])
        reserved |= _cross_footprint(r, c, spec["d"])
        used_distances.add(dist)
        used_chambers.add(chamber_idx)
        placements.append(
            {
                "color": spec["color"],
                "d": spec["d"],
                "side": spec["side"],
                "center": (r, c),
                "chamber_idx": chamber_idx,
                "dist": dist,
            }
        )

    expected = _expected_output(grid, placements, chambers)
    detected = _find_hidden_centers(grid)
    expected_centers = sorted((item["center"][0], item["center"][1], item["color"], item["d"]) for item in placements)
    if detected != expected_centers:
        return None

    add_distractors = random.random() < 0.8
    distractor_targets = {item["color"]: 0 for item in placements}
    if add_distractors:
        noisy_colors = random.sample(
            [item["color"] for item in placements],
            k=random.randint(1, len(placements)),
        )
        for color in noisy_colors:
            distractor_targets[color] = random.randint(1, 3)

    for color, count in distractor_targets.items():
        placed = 0
        attempts = 0
        while placed < count and attempts < 80:
            attempts += 1
            candidates = _candidate_distractor_bars(grid, meta, reserved)
            if not candidates:
                break
            _orient, cells = random.choice(candidates)
            if any(int(grid[r, c]) != BLACK for r, c in cells):
                continue
            snapshot = grid.copy()
            for r, c in cells:
                grid[r, c] = int(color)
            try:
                if _find_hidden_centers(grid) != expected_centers:
                    grid[:, :] = snapshot
                    continue
                _ = main(grid)
            except Exception:
                grid[:, :] = snapshot
                continue
            reserved |= set(cells)
            placed += 1

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-cross chamber grid")
