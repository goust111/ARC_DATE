import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

GEN_RETRIES = 600
GRID_H_RANGE = (15, 23)
GRID_W_RANGE = (18, 28)
CLUSTER_COUNT_RANGE = (3, 5)
PLACEMENT_PADDING = 2
FORCE_TIE_PROB = 0.4
BRANCH_PROB = 0.75


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """ARC-style inclusive bbox crop helper, kept self-contained for SourceFileLoader use."""
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _detect_candidates(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    for r in range(h - 2):
        for c in range(w - 2):
            block = g[r : r + 3, c : c + 3]
            if np.any(block == BLACK):
                continue
            if int(block[1, 1]) != GREY:
                continue
            reds = np.argwhere(block == RED)
            if reds.size == 0:
                continue
            red_set = frozenset((r + int(rr), c + int(cc)) for rr, cc in reds)
            candidates.append(
                {
                    "top": (r, c),
                    "center": (r + 1, c + 1),
                    "reds": red_set,
                    "window": block.copy(),
                }
            )
    return candidates


def _maximal_candidates(candidates: list[dict]) -> list[dict]:
    selected = []
    for idx, cand in enumerate(candidates):
        if any(cand["reds"] < other["reds"] for j, other in enumerate(candidates) if j != idx):
            continue
        selected.append(cand)
    return selected


def _sort_candidates(candidates: list[dict], blue: tuple[int, int]) -> list[dict]:
    return sorted(
        candidates,
        key=lambda cand: (_manhattan(cand["center"], blue), cand["center"][0], cand["center"][1]),
    )


def _assemble(candidates: Iterable[dict]) -> np.ndarray:
    windows = [cand["window"] for cand in candidates]
    assert windows
    return np.hstack(windows)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract maximal marker-defined 3x3 windows and arrange them by distance to the blue target.

    Input structure:
    - The grid has black background, grey foreground material, red marker pixels embedded in that material, and exactly one blue target pixel on the background.
    - A candidate window is any 3x3 region whose nine cells are non-black, whose center cell is grey, and which contains at least one red marker.
    - Neighboring candidates may overlap because the same red markers can support more than one 3x3 detection.

    Rule:
    - Enumerate every candidate 3x3 window.
    - Keep only those candidates whose red-pixel coordinate set is not a strict subset of another candidate's red-pixel set.
    - Compute each kept window center's Manhattan distance to the unique blue pixel, breaking ties by center row then center column.
    - Return a 3-row grid formed by concatenating the selected 3x3 windows from nearest to farthest, preserving each window's original grey and red cells.

    Key insight:
    - The red markers first decide which overlapping detections are genuine via maximal-set disambiguation, and only those surviving windows are then distance-ranked and cropped into the output strip.
    """
    g = np.asarray(grid, dtype=int)
    blue_locs = np.argwhere(g == BLUE)
    assert blue_locs.shape == (1, 2), "expected exactly one blue target pixel"
    blue = tuple(map(int, blue_locs[0]))

    candidates = _detect_candidates(g)
    selected = _maximal_candidates(candidates)
    assert selected, "expected at least one selected window"
    ordered = _sort_candidates(selected, blue)
    return _assemble(ordered)


KIND_TO_LAYOUT = {
    "h_lr": {"shape": (3, 4), "real_offset": (0, 0), "fake_offset": (0, 1)},
    "h_rl": {"shape": (3, 4), "real_offset": (0, 1), "fake_offset": (0, 0)},
    "v_tb": {"shape": (4, 3), "real_offset": (0, 0), "fake_offset": (1, 0)},
    "v_bt": {"shape": (4, 3), "real_offset": (1, 0), "fake_offset": (0, 0)},
}

KIND_TO_POSITIONS = {
    "h_lr": {
        "overlap": [(0, 1), (0, 2), (2, 1), (2, 2)],
        "unique": [(0, 0), (1, 0), (2, 0)],
        "forbidden": {(1, 1), (1, 2)},
    },
    "h_rl": {
        "overlap": [(0, 0), (0, 1), (2, 0), (2, 1)],
        "unique": [(0, 2), (1, 2), (2, 2)],
        "forbidden": {(1, 0), (1, 1)},
    },
    "v_tb": {
        "overlap": [(1, 0), (1, 2), (2, 0), (2, 2)],
        "unique": [(0, 0), (0, 1), (0, 2)],
        "forbidden": {(1, 1), (2, 1)},
    },
    "v_bt": {
        "overlap": [(0, 0), (0, 2), (1, 0), (1, 2)],
        "unique": [(2, 0), (2, 1), (2, 2)],
        "forbidden": {(0, 1), (1, 1)},
    },
}


def _random_subset(items: list[tuple[int, int]], min_count: int, max_count: int) -> list[tuple[int, int]]:
    count = random.randint(min_count, min(max_count, len(items)))
    return random.sample(items, count)


def _make_selected_window(kind: str) -> np.ndarray:
    spec = KIND_TO_POSITIONS[kind]
    window = np.full((3, 3), GREY, dtype=int)
    overlap_reds = _random_subset(list(spec["overlap"]), 1, 2)
    unique_reds = _random_subset(list(spec["unique"]), 1, 2)
    for r, c in overlap_reds + unique_reds:
        if (r, c) not in spec["forbidden"]:
            window[r, c] = RED
    for r, c in spec["forbidden"]:
        window[r, c] = GREY
    assert np.any(window == RED)
    assert int(window[1, 1]) == GREY
    return window


def _build_cluster(kind: str, selected_window: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    layout = KIND_TO_LAYOUT[kind]
    h, w = layout["shape"]
    patch = np.full((h, w), GREY, dtype=int)
    rr, rc = layout["real_offset"]
    patch[rr : rr + 3, rc : rc + 3] = selected_window
    real_top = layout["real_offset"]
    fake_top = layout["fake_offset"]
    return patch, real_top, fake_top


def _rect_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _place_clusters(
    h: int, w: int, clusters: list[dict]
) -> list[dict] | None:
    placed = []
    used_rects = []
    for cluster in clusters:
        ph, pw = cluster["patch"].shape
        positions = [
            (r, c)
            for r in range(1, h - ph - 1)
            for c in range(1, w - pw - 1)
        ]
        random.shuffle(positions)
        found = None
        for r, c in positions:
            rect = (r, c, r + ph - 1, c + pw - 1)
            if any(_rect_overlaps(rect, other, PLACEMENT_PADDING) for other in used_rects):
                continue
            found = (r, c, rect)
            break
        if found is None:
            return None
        r, c, rect = found
        placed_cluster = dict(cluster)
        placed_cluster["origin"] = (r, c)
        placed_cluster["rect"] = rect
        placed.append(placed_cluster)
        used_rects.append(rect)
    return placed


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: tuple[int, int]) -> None:
    r, c = top
    h, w = patch.shape
    grid[r : r + h, c : c + w] = patch


def _best_corner(rect: tuple[int, int, int, int], target: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    return min(corners, key=lambda p: _manhattan(p, target))


def _draw_grey_path(grid: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    er, ec = end
    first_horizontal = random.random() < 0.5
    if first_horizontal:
        while c != ec:
            if int(grid[r, c]) == BLACK:
                grid[r, c] = GREY
            c += 1 if ec > c else -1
        while r != er:
            if int(grid[r, c]) == BLACK:
                grid[r, c] = GREY
            r += 1 if er > r else -1
    else:
        while r != er:
            if int(grid[r, c]) == BLACK:
                grid[r, c] = GREY
            r += 1 if er > r else -1
        while c != ec:
            if int(grid[r, c]) == BLACK:
                grid[r, c] = GREY
            c += 1 if ec > c else -1
    if int(grid[er, ec]) == BLACK:
        grid[er, ec] = GREY


def _add_stub(grid: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    choices = [
        ((r0, c0), (-1, 0)),
        ((r0, c1), (0, 1)),
        ((r1, c0), (0, -1)),
        ((r1, c1), (1, 0)),
    ]
    random.shuffle(choices)
    for (r, c), (dr, dc) in choices:
        length = random.randint(1, 2)
        cells = []
        ok = True
        rr, cc = r, c
        for _ in range(length):
            rr += dr
            cc += dc
            if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
                ok = False
                break
            if int(grid[rr, cc]) != BLACK:
                ok = False
                break
            cells.append((rr, cc))
        if not ok:
            continue
        for rr, cc in cells:
            grid[rr, cc] = GREY
        return


def _window_signature(window: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in window.reshape(-1))


def _candidate_indices_order(candidates: list[dict], ordered: list[dict]) -> list[int]:
    by_top = {cand["top"]: idx for idx, cand in enumerate(candidates)}
    return [by_top[cand["top"]] for cand in ordered]


def _blue_options(
    grid: np.ndarray,
    selected: list[dict],
    force_tie: bool,
) -> list[tuple[int, int]]:
    material = np.argwhere((grid == GREY) | (grid == RED))
    raster = sorted(selected, key=lambda cand: (cand["top"][0], cand["top"][1]))
    raster_tops = [cand["top"] for cand in raster]
    options = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) != BLACK:
                continue
            if min(abs(r - rr) + abs(c - cc) for rr, cc in material) < 3:
                continue
            dists = [_manhattan(cand["center"], (r, c)) for cand in selected]
            if force_tie and len(set(dists)) == len(dists):
                continue
            ordered = _sort_candidates(selected, (r, c))
            if [cand["top"] for cand in ordered] == raster_tops:
                continue
            options.append((r, c))
    return options


def _windows_overlap(a: dict, b: dict) -> bool:
    ar, ac = a["top"]
    br, bc = b["top"]
    return not (ar + 2 < br or br + 2 < ar or ac + 2 < bc or bc + 2 < ac)


def _validate_grid(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    blue_locs = np.argwhere(g == BLUE)
    if blue_locs.shape != (1, 2):
        return False
    candidates = _detect_candidates(g)
    selected = _maximal_candidates(candidates)
    if not (3 <= len(selected) <= 5):
        return False
    if len(candidates) != 2 * len(selected):
        return False
    if len({_window_signature(cand["window"]) for cand in selected}) != len(selected):
        return False
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            if _windows_overlap(selected[i], selected[j]):
                return False
    blue = tuple(map(int, blue_locs[0]))
    ordered = _sort_candidates(selected, blue)
    raster = sorted(selected, key=lambda cand: (cand["top"][0], cand["top"][1]))
    if [cand["top"] for cand in ordered] == [cand["top"] for cand in raster]:
        return False
    if np.array_equal(_assemble(ordered), _assemble(raster)):
        return False
    all_sorted = _sort_candidates(candidates, blue)
    if np.array_equal(_assemble(ordered), _assemble(all_sorted)):
        return False
    if not np.array_equal(main(g), _assemble(ordered)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*CLUSTER_COUNT_RANGE)
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(max(GRID_W_RANGE[0], 5 * count + 2), GRID_W_RANGE[1])
    kinds = random.choices(list(KIND_TO_LAYOUT), k=count)

    windows_seen = set()
    clusters = []
    for kind in kinds:
        for _ in range(50):
            selected_window = _make_selected_window(kind)
            sig = _window_signature(selected_window)
            if sig in windows_seen:
                continue
            patch, real_offset, fake_offset = _build_cluster(kind, selected_window)
            clusters.append(
                {
                    "kind": kind,
                    "patch": patch,
                    "selected_window": selected_window,
                    "real_offset": real_offset,
                    "fake_offset": fake_offset,
                }
            )
            windows_seen.add(sig)
            break
        else:
            return None

    placed = _place_clusters(h, w, clusters)
    if placed is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    for cluster in placed:
        _paint_patch(grid, cluster["patch"], cluster["origin"])

    if random.random() < BRANCH_PROB:
        chain = sorted(
            placed,
            key=lambda cluster: (
                cluster["origin"][0] + cluster["patch"].shape[0] // 2,
                cluster["origin"][1] + cluster["patch"].shape[1] // 2,
            ),
        )
        for left, right in zip(chain, chain[1:]):
            left_center = (
                left["origin"][0] + left["patch"].shape[0] // 2,
                left["origin"][1] + left["patch"].shape[1] // 2,
            )
            right_center = (
                right["origin"][0] + right["patch"].shape[0] // 2,
                right["origin"][1] + right["patch"].shape[1] // 2,
            )
            start = _best_corner(left["rect"], right_center)
            end = _best_corner(right["rect"], left_center)
            _draw_grey_path(grid, start, end)

    for cluster in placed:
        if random.random() < 0.55:
            _add_stub(grid, cluster["rect"])

    candidates = _detect_candidates(grid)
    selected = _maximal_candidates(candidates)
    if len(candidates) != 2 * count or len(selected) != count:
        return None

    force_tie = random.random() < FORCE_TIE_PROB
    blue_choices = _blue_options(grid, selected, force_tie)
    if not blue_choices:
        return None
    blue_r, blue_c = random.choice(blue_choices)
    grid[blue_r, blue_c] = BLUE

    if not _validate_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
