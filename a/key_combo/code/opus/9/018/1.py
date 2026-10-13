import random
from collections import Counter

import numpy as np


BG = 0
OCCLUDED = 9
MARKER_COLORS = (1, 3, 7)
PATTERN_COLORS = (2, 4, 5, 6, 8)
GEN_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _edge_cells(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    cells = []
    for c in range(w):
        cells.append((0, c))
        if h > 1:
            cells.append((h - 1, c))
    for r in range(1, h - 1):
        cells.append((r, 0))
        if w > 1:
            cells.append((r, w - 1))
    return cells


def _marker_side(shape: tuple[int, int], r: int, c: int) -> str:
    h, w = shape
    if c == 0:
        return "L"
    if c == w - 1:
        return "R"
    if r == 0:
        return "T"
    if r == h - 1:
        return "B"
    raise AssertionError("marker must lie on the border")


def _marker_step(side: str) -> tuple[int, int]:
    return {
        "L": (0, 1),
        "R": (0, -1),
        "T": (1, 0),
        "B": (-1, 0),
    }[side]


def _edge_markers(grid: np.ndarray) -> list[dict]:
    markers = []
    seen = set()
    for r, c in _edge_cells(grid.shape):
        if (r, c) in seen:
            continue
        seen.add((r, c))
        value = int(grid[r, c])
        if value == BG:
            continue
        side = _marker_side(grid.shape, r, c)
        markers.append({"pos": (r, c), "side": side, "color": value})
    assert 2 <= len(markers) <= 3, "expected 2-3 edge markers"
    horizontal_rows = [m["pos"][0] for m in markers if m["side"] in ("L", "R")]
    vertical_cols = [m["pos"][1] for m in markers if m["side"] in ("T", "B")]
    assert len(horizontal_rows) == len(set(horizontal_rows)), "horizontal markers must use distinct rows"
    assert len(vertical_cols) == len(set(vertical_cols)), "vertical markers must use distinct cols"
    return markers


def _pattern_bbox(grid: np.ndarray, markers: list[dict]) -> tuple[int, int, int, int]:
    core = grid.copy()
    for marker in markers:
        r, c = marker["pos"]
        core[r, c] = BG
    coords = np.argwhere(core != BG)
    assert coords.size > 0, "expected a central pattern"
    return _bbox_coords(coords)


def _orbit_cells(r: int, c: int, bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    mr = r0 + r1 - r
    mc = c0 + c1 - c
    cells = {(r, c), (mr, c), (r, mc), (mr, mc)}
    return sorted(cells)


def _complete_pattern(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    r0, c0, r1, c1 = bbox
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            orbit = _orbit_cells(r, c, bbox)
            visible = {int(grid[rr, cc]) for rr, cc in orbit if int(grid[rr, cc]) != OCCLUDED}
            assert len(visible) == 1, "every orbit must reveal one unique original color"
            value = visible.pop()
            for rr, cc in orbit:
                if out[rr, cc] == OCCLUDED:
                    out[rr, cc] = value
    return out


def _simulate_from_completed(
    completed: np.ndarray,
    markers: list[dict],
    bbox: tuple[int, int, int, int],
    *,
    reverse_order: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    r0, c0, r1, c1 = bbox
    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    out = completed.copy()
    for marker in markers:
        r, c = marker["pos"]
        out[r, c] = BG

    order = sorted(
        markers,
        key=lambda m: abs(m["pos"][0] - center[0]) + abs(m["pos"][1] - center[1]),
        reverse=not reverse_order,
    )

    placed = {}
    details = []
    h, w = out.shape
    for marker in order:
        r, c = marker["pos"]
        dr, dc = _marker_step(marker["side"])
        prev = (r, c)
        rr, cc = r + dr, c + dc
        while 0 <= rr < h and 0 <= cc < w:
            if int(out[rr, cc]) != BG:
                absorb = int(out[rr, cc])
                final = prev
                out[final] = absorb
                placed[final] = absorb
                details.append(
                    {
                        "start": (r, c),
                        "side": marker["side"],
                        "final": final,
                        "blocked": (rr, cc),
                        "blocked_type": "marker" if (rr, cc) in placed else "pattern",
                        "absorbed": absorb,
                    }
                )
                break
            prev = (rr, cc)
            rr += dr
            cc += dc
        else:
            raise AssertionError("every marker must hit an obstacle before leaving the grid")
    return out, details


def _simulate(grid: np.ndarray, *, reverse_order: bool = False) -> tuple[np.ndarray, list[dict], tuple[int, int, int, int], np.ndarray]:
    markers = _edge_markers(grid)
    bbox = _pattern_bbox(grid, markers)
    completed = _complete_pattern(grid, bbox)
    out, details = _simulate_from_completed(completed, markers, bbox, reverse_order=reverse_order)
    return out, details, bbox, completed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the occluded symmetric pattern, then slide border markers inward until they hit the completed obstacle map.

    Input structure:
    - A black square grid contains one central odd-sized rectangular pattern region.
    - Inside that bbox, the true pattern is horizontally and vertically symmetric around one shared center, but some cells are replaced by maroon 9.
    - Two or three single-pixel markers lie on the outer border; left/right markers move along their row, top/bottom markers move along their column.

    Rule:
    - Remove the border markers from consideration, find the central bbox, and restore every 9 by copying the unique color revealed by its reflection orbit under the bbox's horizontal and vertical symmetry axes.
    - Treat the completed pattern as an obstacle map. Process markers from larger to smaller Manhattan distance to the pattern center.
    - Each marker slides one cell at a time toward the center and stops on the last background cell before the first non-background obstacle, which may be either a completed pattern cell or an already-settled marker.
    - The marker takes the blocking pixel's color, and its original border cell becomes black.

    Key insight:
    - The same completed symmetric pattern is used twice: first to decide which 9 cells are real obstacles or passable background, and then to determine both marker stopping points and the colors they absorb.
    """
    out, _details, _bbox, _completed = _simulate(np.asarray(grid, dtype=int))
    return out


def _orbit_key(cell: tuple[int, int], size: int) -> tuple[tuple[int, int], ...]:
    bbox = (0, 0, size - 1, size - 1)
    return tuple(_orbit_cells(cell[0], cell[1], bbox))


def _paint_orbit(patch: np.ndarray, cell: tuple[int, int], value: int) -> None:
    size = patch.shape[0]
    for r, c in _orbit_cells(cell[0], cell[1], (0, 0, size - 1, size - 1)):
        patch[r, c] = int(value)


def _make_patch(size: int, rng: random.Random) -> np.ndarray | None:
    mid = size // 2
    colors = list(PATTERN_COLORS)
    rng.shuffle(colors)
    used = colors[: rng.randint(3, len(colors))]

    patch = np.zeros((size, size), dtype=int)
    mandatory = [(0, mid), (mid, 0), (mid, mid)]
    if mid >= 2:
        mandatory.extend([(1, mid - 1), (mid - 1, 1)])
    for cell in mandatory:
        _paint_orbit(patch, cell, rng.choice(used))

    for r in range(mid + 1):
        for c in range(mid + 1):
            if patch[r, c] != BG:
                continue
            p = 0.52 if (r == mid or c == mid) else 0.38
            if rng.random() < p:
                _paint_orbit(patch, (r, c), rng.choice(used))

    fg = patch != BG
    if not (size * size * 0.25 <= int(fg.sum()) <= size * size * 0.72):
        return None
    if len(set(int(v) for v in patch[fg])) < 3:
        return None
    if not (fg[0].any() and fg[-1].any() and fg[:, 0].any() and fg[:, -1].any()):
        return None
    return patch


def _row_cols(patch: np.ndarray) -> list[list[int]]:
    return [list(np.where(patch[r] != BG)[0]) for r in range(patch.shape[0])]


def _col_rows(patch: np.ndarray) -> list[list[int]]:
    return [list(np.where(patch[:, c] != BG)[0]) for c in range(patch.shape[1])]


def _distance_for_side(
    grid_size: int,
    bbox_origin: tuple[int, int],
    patch_size: int,
    side: str,
    line: int,
) -> int:
    r0, c0 = bbox_origin
    center_r = r0 + patch_size // 2
    center_c = c0 + patch_size // 2
    if side == "L":
        return abs((r0 + line) - center_r) + center_c
    if side == "R":
        return abs((r0 + line) - center_r) + (grid_size - 1 - center_c)
    if side == "T":
        return center_r + abs((c0 + line) - center_c)
    return (grid_size - 1 - center_r) + abs((c0 + line) - center_c)


def _valid_occlusion_cell(cell: tuple[int, int], patch: np.ndarray) -> bool:
    return patch[cell] != BG and len(_orbit_key(cell, patch.shape[0])) > 1


def _choose_chain_pair(
    patch: np.ndarray,
    grid_size: int,
    bbox_origin: tuple[int, int],
    rng: random.Random,
) -> list[dict] | None:
    size = patch.shape[0]
    rows = _row_cols(patch)
    cols = _col_rows(patch)
    candidates = []

    for r, fg_cols in enumerate(rows):
        if not fg_cols:
            continue

        left_block = fg_cols[0]
        if left_block >= 1 and _valid_occlusion_cell((r, left_block), patch):
            stop_col = left_block - 1
            top_rows = cols[stop_col]
            if top_rows and top_rows[0] > r and _valid_occlusion_cell((top_rows[0], stop_col), patch):
                candidates.append(
                    [
                        {"side": "L", "line": r, "path_occ": (r, left_block), "kind": "pattern"},
                        {"side": "T", "line": stop_col, "path_occ": (top_rows[0], stop_col), "kind": "chain"},
                    ]
                )
            if top_rows and top_rows[-1] < r and _valid_occlusion_cell((top_rows[-1], stop_col), patch):
                candidates.append(
                    [
                        {"side": "L", "line": r, "path_occ": (r, left_block), "kind": "pattern"},
                        {"side": "B", "line": stop_col, "path_occ": (top_rows[-1], stop_col), "kind": "chain"},
                    ]
                )

        right_block = fg_cols[-1]
        if right_block <= size - 2 and _valid_occlusion_cell((r, right_block), patch):
            stop_col = right_block + 1
            col_rows = cols[stop_col]
            if col_rows and col_rows[0] > r and _valid_occlusion_cell((col_rows[0], stop_col), patch):
                candidates.append(
                    [
                        {"side": "R", "line": r, "path_occ": (r, right_block), "kind": "pattern"},
                        {"side": "T", "line": stop_col, "path_occ": (col_rows[0], stop_col), "kind": "chain"},
                    ]
                )
            if col_rows and col_rows[-1] < r and _valid_occlusion_cell((col_rows[-1], stop_col), patch):
                candidates.append(
                    [
                        {"side": "R", "line": r, "path_occ": (r, right_block), "kind": "pattern"},
                        {"side": "B", "line": stop_col, "path_occ": (col_rows[-1], stop_col), "kind": "chain"},
                    ]
                )

    rng.shuffle(candidates)
    for pair in candidates:
        keys = {_orbit_key(spec["path_occ"], size) for spec in pair}
        if len(keys) != 2:
            continue
        d0 = _distance_for_side(grid_size, bbox_origin, size, pair[0]["side"], pair[0]["line"])
        d1 = _distance_for_side(grid_size, bbox_origin, size, pair[1]["side"], pair[1]["line"])
        if d0 <= d1:
            continue
        if d0 == d1:
            continue
        return pair
    return None


def _simple_candidates(patch: np.ndarray) -> list[dict]:
    size = patch.shape[0]
    rows = _row_cols(patch)
    cols = _col_rows(patch)
    candidates = []
    for r, fg_cols in enumerate(rows):
        if fg_cols:
            if _valid_occlusion_cell((r, fg_cols[0]), patch):
                candidates.append({"side": "L", "line": r, "path_occ": (r, fg_cols[0]), "kind": "pattern"})
            if _valid_occlusion_cell((r, fg_cols[-1]), patch):
                candidates.append({"side": "R", "line": r, "path_occ": (r, fg_cols[-1]), "kind": "pattern"})
    for c, fg_rows in enumerate(cols):
        if fg_rows:
            if _valid_occlusion_cell((fg_rows[0], c), patch):
                candidates.append({"side": "T", "line": c, "path_occ": (fg_rows[0], c), "kind": "pattern"})
            if _valid_occlusion_cell((fg_rows[-1], c), patch):
                candidates.append({"side": "B", "line": c, "path_occ": (fg_rows[-1], c), "kind": "pattern"})
    return candidates


def _compatible_marker_specs(existing: list[dict], candidate: dict, patch: np.ndarray) -> bool:
    key = _orbit_key(candidate["path_occ"], patch.shape[0])
    for spec in existing:
        if spec["side"] in ("L", "R") and candidate["side"] in ("L", "R") and spec["line"] == candidate["line"]:
            return False
        if spec["side"] in ("T", "B") and candidate["side"] in ("T", "B") and spec["line"] == candidate["line"]:
            return False
        if _orbit_key(spec["path_occ"], patch.shape[0]) == key:
            return False
    return True


def _choose_specs(
    patch: np.ndarray,
    grid_size: int,
    bbox_origin: tuple[int, int],
    rng: random.Random,
) -> tuple[list[dict], bool] | None:
    scenario = rng.choices(
        ["chain3", "chain2", "simple3", "simple2"],
        weights=[4, 3, 2, 1],
        k=1,
    )[0]

    specs: list[dict] = []
    needs_chain = scenario.startswith("chain")
    if needs_chain:
        pair = _choose_chain_pair(patch, grid_size, bbox_origin, rng)
        if pair is None:
            return None
        specs.extend(pair)

    target = 3 if scenario.endswith("3") else 2
    candidates = _simple_candidates(patch)
    rng.shuffle(candidates)
    for cand in candidates:
        if len(specs) >= target:
            break
        if not _compatible_marker_specs(specs, cand, patch):
            continue
        specs.append(cand)

    if len(specs) != target:
        return None

    dists = [
        _distance_for_side(grid_size, bbox_origin, patch.shape[0], spec["side"], spec["line"])
        for spec in specs
    ]
    if len(dists) != len(set(dists)):
        return None
    return specs, needs_chain


def _extra_occlusions(
    patch: np.ndarray,
    selected_specs: list[dict],
    rng: random.Random,
) -> list[tuple[int, int]]:
    size = patch.shape[0]
    used_keys = {_orbit_key(spec["path_occ"], size) for spec in selected_specs}
    occlusions = [spec["path_occ"] for spec in selected_specs]

    bg_choices = []
    fg_choices = []
    for r in range(size):
        for c in range(size):
            key = _orbit_key((r, c), size)
            if len(key) == 1 or key in used_keys:
                continue
            if patch[r, c] == BG:
                bg_choices.append((r, c))
            else:
                fg_choices.append((r, c))

    rng.shuffle(bg_choices)
    rng.shuffle(fg_choices)

    if bg_choices:
        cell = bg_choices.pop()
        occlusions.append(cell)
        used_keys.add(_orbit_key(cell, size))

    target = rng.randint(max(4, len(occlusions)), 8)
    pool = bg_choices + fg_choices
    rng.shuffle(pool)
    for cell in pool:
        if len(occlusions) >= target:
            break
        key = _orbit_key(cell, size)
        if key in used_keys:
            continue
        occlusions.append(cell)
        used_keys.add(key)
    return occlusions


def _place_specs_on_grid(
    patch: np.ndarray,
    grid_size: int,
    bbox_origin: tuple[int, int],
    specs: list[dict],
    occlusions: list[tuple[int, int]],
    rng: random.Random,
) -> np.ndarray:
    grid = np.zeros((grid_size, grid_size), dtype=int)
    r0, c0 = bbox_origin
    size = patch.shape[0]
    grid[r0 : r0 + size, c0 : c0 + size] = patch

    for rr, cc in occlusions:
        grid[r0 + rr, c0 + cc] = OCCLUDED

    marker_colors = list(MARKER_COLORS)
    rng.shuffle(marker_colors)
    for spec, color in zip(specs, marker_colors):
        if spec["side"] == "L":
            pos = (r0 + spec["line"], 0)
        elif spec["side"] == "R":
            pos = (r0 + spec["line"], grid_size - 1)
        elif spec["side"] == "T":
            pos = (0, c0 + spec["line"])
        else:
            pos = (grid_size - 1, c0 + spec["line"])
        spec["pos"] = pos
        spec["color"] = int(color)
        grid[pos] = int(color)
    return grid


def _raw_obstacle_simulation(grid: np.ndarray) -> np.ndarray:
    markers = _edge_markers(grid)
    bbox = _pattern_bbox(grid, markers)
    core = np.array(grid, dtype=int, copy=True)
    for marker in markers:
        r, c = marker["pos"]
        core[r, c] = BG
    out, _details = _simulate_from_completed(core, markers, bbox)
    return out


def _validate_instance(grid: np.ndarray, specs: list[dict], needs_chain: bool) -> bool:
    try:
        out, details, bbox, completed = _simulate(grid)
    except Exception:
        return False

    if np.any(grid == OCCLUDED) and np.array_equal(out, _raw_obstacle_simulation(grid)):
        return False

    if needs_chain and not any(d["blocked_type"] == "marker" for d in details):
        return False
    if needs_chain:
        reverse_out, _reverse_details, _bbox2, _completed2 = _simulate(grid, reverse_order=True)
        if np.array_equal(out, reverse_out):
            return False

    r0, c0, r1, c1 = bbox
    for spec in specs:
        if spec["side"] in ("L", "R"):
            line = r0 + spec["line"]
            segment = grid[line, :]
        else:
            line = c0 + spec["line"]
            segment = grid[:, line]
        completed_segment = completed[line, :] if spec["side"] in ("L", "R") else completed[:, line]
        if not np.any((segment == OCCLUDED) & (completed_segment != BG)):
            return False

    markers = _edge_markers(grid)
    if len(markers) != len(specs):
        return False
    if not (4 <= int(np.sum(grid == OCCLUDED)) <= 8):
        return False
    if any(int(d["absorbed"]) in MARKER_COLORS for d in details):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_size = rng.choice([13, 14, 15, 16, 17, 18, 19])
    patch_size = rng.choice([5, 7, 9])
    if patch_size + 4 > grid_size:
        patch_size = 5

    patch = _make_patch(patch_size, rng)
    if patch is None:
        return None

    center_guess = (grid_size - patch_size) // 2
    r0 = max(2, min(grid_size - patch_size - 2, center_guess + rng.randint(-1, 1)))
    c0 = max(2, min(grid_size - patch_size - 2, center_guess + rng.randint(-1, 1)))
    bbox_origin = (r0, c0)

    choice = _choose_specs(patch, grid_size, bbox_origin, rng)
    if choice is None:
        return None
    specs, needs_chain = choice

    occlusions = _extra_occlusions(patch, specs, rng)
    if not (4 <= len(occlusions) <= 8):
        return None

    grid = _place_specs_on_grid(patch, grid_size, bbox_origin, specs, occlusions, rng)
    if not _validate_instance(grid, specs, needs_chain):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

