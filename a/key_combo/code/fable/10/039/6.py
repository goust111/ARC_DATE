from __future__ import annotations

import numpy as np


RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_GEN_HEIGHT_RANGE = (12, 23)
_GEN_PANEL_WIDTH_RANGE = (8, 15)
_GEN_MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _most_common_color(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, p)) for p in coords}


def _mirror_point(r: int, c: int, axis_c: int) -> tuple[int, int]:
    return int(r), int(2 * axis_c - c)


def _mirror_set(coords: set[tuple[int, int]], axis_c: int) -> set[tuple[int, int]]:
    return {_mirror_point(r, c, axis_c) for r, c in coords}


def _divider_axis(grid: np.ndarray) -> int:
    grey_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(grey_cols) == 1, "expected exactly one full-height grey divider"
    axis = int(grey_cols[0])
    assert axis > 0 and axis < grid.shape[1] - 1
    assert grid.shape[1] == 2 * axis + 1, "panels must have equal width around the divider"
    return axis


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    axis = _divider_axis(g)
    left_bg = _most_common_color(g[:, :axis])
    right_bg = _most_common_color(g[:, axis + 1 :])
    assert left_bg == YELLOW and right_bg == GREEN, "unexpected panel colors"
    assert np.all(np.isin(g, [RED, GREEN, YELLOW, GREY]))

    comps = _components(g == RED)
    left, right = [], []
    for idx, coords in enumerate(comps):
        cols = coords[:, 1]
        entry = {"id": idx, "coords": coords, "set": _coord_set(coords)}
        if np.all(cols < axis):
            left.append(entry)
        elif np.all(cols > axis):
            right.append(entry)
        else:
            raise AssertionError("red component crosses the divider")

    left_to_right: dict[int, list[int]] = {i: [] for i in range(len(left))}
    right_to_left: dict[int, list[int]] = {j: [] for j in range(len(right))}
    overlap_cells: dict[tuple[int, int], set[tuple[int, int]]] = {}

    for i, lcomp in enumerate(left):
        for j, rcomp in enumerate(right):
            mirrored_right = _mirror_set(rcomp["set"], axis)
            overlap = lcomp["set"] & mirrored_right
            if overlap:
                left_to_right[i].append(j)
                right_to_left[j].append(i)
                overlap_cells[(i, j)] = overlap

    assert all(len(v) <= 1 for v in left_to_right.values()), "ambiguous left-side pairing"
    assert all(len(v) <= 1 for v in right_to_left.values()), "ambiguous right-side pairing"

    pairs = []
    for i, js in left_to_right.items():
        if not js:
            continue
        j = js[0]
        assert right_to_left[j] == [i], "pairing must be mutual"
        pairs.append(
            {
                "left": left[i],
                "right": right[j],
                "overlap": overlap_cells[(i, j)],
            }
        )

    paired_left = {pair["left"]["id"] for pair in pairs}
    paired_right = {pair["right"]["id"] for pair in pairs}
    unpaired_left = [comp for comp in left if comp["id"] not in paired_left]
    unpaired_right = [comp for comp in right if comp["id"] not in paired_right]

    return {
        "axis": axis,
        "left_bg": left_bg,
        "right_bg": right_bg,
        "pairs": pairs,
        "unpaired_left": unpaired_left,
        "unpaired_right": unpaired_right,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Repair mirrored broken objects across the divider, borrowing missing pixels from the opposite panel and erasing unsupported phantoms.

    Input structure:
    - A single full-height grey column splits the grid into a yellow left panel and a green right panel.
    - Red 4-connected objects appear on both panels; some left objects and right objects occupy mirrored positions across the divider.
    - In each mirrored pair, both sides are incomplete, and the pixels missing on one side are present on the other side.

    Rule:
    - Pair red objects whose shapes overlap after reflecting one side across the grey divider.
    - For each pair, take the mirrored union as the full shape: restore left-side missing pixels in green and right-side missing pixels in yellow, while keeping existing red pixels red.
    - Any red object with no mirrored overlap partner is a phantom and is erased back to its own panel background.

    Key insight:
    - The divider is the only valid mirror axis, so opposite-panel evidence both identifies true counterparts and determines the patch color used to repair each side.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)
    axis = info["axis"]

    out = g.copy()
    for comp in info["unpaired_left"]:
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = info["left_bg"]
    for comp in info["unpaired_right"]:
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = info["right_bg"]

    for pair in info["pairs"]:
        left_set = pair["left"]["set"]
        right_set = pair["right"]["set"]
        full_left = left_set | _mirror_set(right_set, axis)
        full_right = right_set | _mirror_set(left_set, axis)

        for r, c in full_left - left_set:
            assert g[r, c] == info["left_bg"]
            out[r, c] = info["right_bg"]
        for r, c in full_right - right_set:
            assert g[r, c] == info["right_bg"]
            out[r, c] = info["left_bg"]

    return out


def _normalize(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return {(r - min_r, c - min_c) for r, c in coords}


def _shift(coords: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in coords}


def _orient(coords: set[tuple[int, int]], rot: int, flip: bool) -> set[tuple[int, int]]:
    out = set(coords)
    if flip:
        out = {(r, -c) for r, c in out}
    for _ in range(rot % 4):
        out = {(c, -r) for r, c in out}
    return _normalize(out)


def _attachment_candidates(core: set[tuple[int, int]]) -> list[tuple[int, int]]:
    cands = set()
    for r, c in core:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) not in core:
                cands.add((nr, nc))
    return sorted(cands)


def _random_core(rng: np.random.Generator) -> set[tuple[int, int]]:
    for _ in range(200):
        target = int(rng.integers(4, 8))
        cells = {(0, 0)}
        frontier = set(_neighbors4(0, 0))
        while len(cells) < target and frontier:
            frontier_list = list(frontier)
            choice = frontier_list[int(rng.integers(len(frontier_list)))]
            frontier.discard(choice)
            cells.add(choice)
            for nr, nc in _neighbors4(*choice):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells = _normalize(cells)
        h = max(r for r, _ in cells) + 1
        w = max(c for _, c in cells) + 1
        if h < 2 or w < 2 or h > 5 or w > 5:
            continue
        if len(_attachment_candidates(cells)) < 3:
            continue
        return cells
    raise RuntimeError("failed to generate core polyomino")


def _random_pair_shape(rng: np.random.Generator) -> dict:
    for _ in range(200):
        core = _orient(_random_core(rng), int(rng.integers(4)), bool(rng.integers(2)))
        cands = _attachment_candidates(core)
        rng.shuffle(cands)
        max_left = min(3, len(cands) - 1)
        left_n = int(rng.integers(1, max_left + 1))
        max_right = min(3, len(cands) - left_n)
        if max_right < 1:
            continue
        right_n = int(rng.integers(1, max_right + 1))
        left_only = set(cands[:left_n])
        right_only = set(cands[left_n : left_n + right_n])
        full = core | left_only | right_only
        min_r = min(r for r, _ in full)
        min_c = min(c for _, c in full)
        full = _shift(full, -min_r, -min_c)
        core = _shift(core, -min_r, -min_c)
        left_only = _shift(left_only, -min_r, -min_c)
        right_only = _shift(right_only, -min_r, -min_c)

        full_h = max(r for r, _ in full) + 1
        full_w = max(c for _, c in full) + 1
        if full_h > 6 or full_w > 6:
            continue
        return {
            "core": core,
            "left_only": left_only,
            "right_only": right_only,
            "full": full,
        }
    raise RuntimeError("failed to generate paired shape")


def _translate(coords: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in coords}


def _valid_translations(
    shape: set[tuple[int, int]],
    blocked: np.ndarray,
    panel_h: int,
    panel_w: int,
) -> list[tuple[int, int]]:
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    translations = []
    for dr in range(1, panel_h - max_r):
        for dc in range(1, panel_w - max_c):
            placed = _translate(shape, dr, dc)
            if all(not blocked[r, c] for r, c in placed):
                translations.append((dr, dc))
    return translations


def _mark_blocked(blocked: np.ndarray, coords: set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _choose_phantom_subset(
    rng: np.random.Generator,
    core: set[tuple[int, int]],
    extras: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    extras_list = sorted(extras)
    take = int(rng.integers(0, len(extras_list) + 1))
    if take == 0:
        return set(core)
    rng.shuffle(extras_list)
    return set(core) | set(extras_list[:take])


def _build_expected(
    grid: np.ndarray,
    pairs: list[dict],
    left_phantoms: list[set[tuple[int, int]]],
    right_phantoms: list[set[tuple[int, int]]],
    axis: int,
) -> np.ndarray:
    out = grid.copy()
    for coords in left_phantoms:
        for r, c in coords:
            out[r, c] = YELLOW
    for coords in right_phantoms:
        for r, c in coords:
            out[r, c] = GREEN
    for pair in pairs:
        for r, c in pair["right_only_left"]:
            out[r, c] = GREEN
        for r, c in _mirror_set(pair["left_only_left"], axis):
            out[r, c] = YELLOW
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*_GEN_HEIGHT_RANGE))
    panel_w = int(rng.integers(*_GEN_PANEL_WIDTH_RANGE))
    w = 2 * panel_w + 1
    axis = panel_w

    panel_area = h * panel_w
    capacity = 3 if panel_area < 120 else 4 if panel_area < 180 else 5
    pair_count = int(rng.integers(1, min(3, capacity - 1) + 1))
    phantom_total = int(rng.integers(1, capacity - pair_count + 1))
    left_phantoms_n = int(rng.integers(0, phantom_total + 1))
    right_phantoms_n = phantom_total - left_phantoms_n
    if left_phantoms_n == 0 and right_phantoms_n == 0:
        return None

    states = ["pair"] * pair_count + ["left"] * left_phantoms_n + ["right"] * right_phantoms_n
    rng.shuffle(states)

    blocked = np.zeros((h, panel_w), dtype=bool)
    pair_meta = []
    left_phantoms: list[set[tuple[int, int]]] = []
    right_phantoms: list[set[tuple[int, int]]] = []

    grid = np.full((h, w), YELLOW, dtype=int)
    grid[:, axis] = GREY
    grid[:, axis + 1 :] = GREEN

    for state in states:
        placed = False
        for _ in range(120):
            shape = _random_pair_shape(rng)
            translations = _valid_translations(shape["full"], blocked, h, panel_w)
            if not translations:
                continue
            dr, dc = translations[int(rng.integers(len(translations)))]

            core = _translate(shape["core"], dr, dc)
            left_only = _translate(shape["left_only"], dr, dc)
            right_only = _translate(shape["right_only"], dr, dc)
            full = _translate(shape["full"], dr, dc)

            if state == "pair":
                left_partial = core | left_only
                right_partial = _mirror_set(core | right_only, axis)
                for r, c in left_partial:
                    grid[r, c] = RED
                for r, c in right_partial:
                    grid[r, c] = RED
                pair_meta.append(
                    {
                        "left_only_left": left_only,
                        "right_only_left": right_only,
                    }
                )
            elif state == "left":
                phantom = _choose_phantom_subset(rng, core, left_only | right_only)
                for r, c in phantom:
                    grid[r, c] = RED
                left_phantoms.append(phantom)
            else:
                phantom = _choose_phantom_subset(rng, core, left_only | right_only)
                mirrored = _mirror_set(phantom, axis)
                for r, c in mirrored:
                    grid[r, c] = RED
                right_phantoms.append(mirrored)

            _mark_blocked(blocked, full)
            placed = True
            break
        if not placed:
            return None

    expected = _build_expected(grid, pair_meta, left_phantoms, right_phantoms, axis)
    try:
        info = _analyze(grid)
    except AssertionError:
        return None

    if len(info["pairs"]) != pair_count:
        return None
    if len(info["unpaired_left"]) != left_phantoms_n or len(info["unpaired_right"]) != right_phantoms_n:
        return None

    for pair in info["pairs"]:
        left_set = pair["left"]["set"]
        right_left = _mirror_set(pair["right"]["set"], axis)
        if not (left_set - right_left):
            return None
        if not (right_left - left_set):
            return None
        if len(pair["overlap"]) < 3:
            return None

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(grid, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-panel repair input")
