from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MARKER_COLORS = (BLUE, RED, GREEN, YELLOW, ORANGE)

DIRS_URDL = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIRS4 = DIRS_URDL


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _exterior_black(mask_black: np.ndarray) -> np.ndarray:
    h, w = mask_black.shape
    exterior = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def add(r: int, c: int) -> None:
        if mask_black[r, c] and not exterior[r, c]:
            exterior[r, c] = True
            queue.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(0, c)
        add(h - 1, c)

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                add(nr, nc)
    return exterior


def _marker_list(grid: np.ndarray) -> list[dict]:
    markers: list[dict] = []
    for r, c in np.argwhere((grid != BLACK) & (grid != GREY)):
        color = int(grid[r, c])
        markers.append({"row": int(r), "col": int(c), "color": color})
    markers.sort(key=lambda m: (m["row"], m["col"]))
    for idx, marker in enumerate(markers):
        marker["id"] = idx
    return markers


def _emit_rays(
    shape: tuple[int, int], grey_mask: np.ndarray, markers: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    owner = np.full((h, w), -1, dtype=int)
    color = np.full((h, w), BLACK, dtype=int)
    for marker in markers:
        idx = int(marker["id"])
        r0 = int(marker["row"])
        c0 = int(marker["col"])
        marker_color = int(marker["color"])
        for dr, dc in DIRS_URDL:
            r, c = r0, c0
            while True:
                owner[r, c] = idx
                color[r, c] = marker_color
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    break
                if grey_mask[nr, nc]:
                    break
                r, c = nr, nc
    return owner, color


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    grey_mask = g == GREY
    markers = _marker_list(g)
    colors = [int(m["color"]) for m in markers]
    assert len(markers) >= 1, "expected at least one marker"
    assert len(set(colors)) == len(colors), "marker colors must be unique"

    owner, ray_colors = _emit_rays(g.shape, grey_mask, markers)
    wall_mask = grey_mask | (owner >= 0)
    black_mask = ~wall_mask
    exterior = _exterior_black(black_mask)
    enclosed_mask = black_mask & ~exterior
    regions = _components(enclosed_mask)

    active_owner_ids: set[int] = set()
    region_infos: list[dict] = []
    out = g.copy()
    out[owner >= 0] = ray_colors[owner >= 0]

    for coords in regions:
        boundary_cells_by_owner: dict[int, set[tuple[int, int]]] = {}
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and owner[nr, nc] >= 0:
                    marker_id = int(owner[nr, nc])
                    boundary_cells_by_owner.setdefault(marker_id, set()).add((nr, nc))

        assert boundary_cells_by_owner, "each enclosed region must touch at least one ray"
        counts = {idx: len(cells) for idx, cells in boundary_cells_by_owner.items()}
        tied = [
            idx
            for idx, count in counts.items()
            if count == max(counts.values())
        ]
        winner = min(
            tied,
            key=lambda idx: (markers[idx]["row"], markers[idx]["col"]),
        )
        fill_color = int(markers[winner]["color"])
        out[tuple(coords.T)] = fill_color
        active_owner_ids.update(counts)
        region_infos.append(
            {
                "coords": coords,
                "counts": counts,
                "winner": winner,
                "fill_color": fill_color,
                "tie": len(tied) > 1,
                "contributors": sorted(counts),
            }
        )

    inactive_ids = {int(m["id"]) for m in markers} - active_owner_ids
    for idx in inactive_ids:
        out[owner == idx] = BLACK

    return {
        "markers": markers,
        "grey_mask": grey_mask,
        "owner": owner,
        "ray_colors": ray_colors,
        "regions": region_infos,
        "active_ids": active_owner_ids,
        "inactive_ids": inactive_ids,
        "output": out,
        "pre_enclosed_mask": enclosed_mask,
    }


def _min_manhattan_to_grey(grey_cells: list[tuple[int, int]], r: int, c: int) -> int:
    if not grey_cells:
        return 10**9
    return min(abs(r - rr) + abs(c - cc) for rr, cc in grey_cells)


def _hits_grey(shape: tuple[int, int], grey_mask: np.ndarray, marker: dict) -> bool:
    h, w = shape
    r0, c0 = int(marker["row"]), int(marker["col"])
    for dr, dc in DIRS_URDL:
        r, c = r0, c0
        while True:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if grey_mask[nr, nc]:
                return True
            r, c = nr, nc
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit cardinal rays from colored markers, fill the black regions newly enclosed by those rays and the grey walls, then erase markers whose rays never bound any enclosed region.

    Input structure:
    - A black background contains one-cell-thick grey wall segments with deliberate gaps.
    - Three to five non-grey, non-grey-wall marker pixels of distinct colors sit away from the walls.
    - No enclosed black region is already present before the rays are emitted.

    Rule:
    - Process markers in ascending `(row, col)` order, and for each marker emit four same-color rays in the order up, right, down, left.
    - A ray advances cell by cell until the next step would enter a grey wall or leave the grid; rays ignore previously drawn rays, and later rays overwrite earlier ray colors where they cross.
    - After all rays are drawn, treat every grey cell and every visible ray cell as a wall; each black region that cannot reach the grid boundary through 4-neighbor black steps is enclosed.
    - For each enclosed region, count how many boundary ray pixels come from each marker source; fill the region with the color of the source with the largest count, breaking ties by the smaller marker row.
    - Finally erase any marker whose visible rays never appear on the boundary of any enclosed region, while keeping grey walls and all contributing markers unchanged.

    Key insight:
    - The rays are structural, not decorative: they dynamically complete missing walls and split open space into new sealed chambers, and the dominant boundary ray around each chamber determines its fill color.
    """
    return _analyze(np.asarray(grid, dtype=int))["output"]


def _draw_segment(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        c_lo, c_hi = sorted((c0, c1))
        grid[r0, c_lo : c_hi + 1] = GREY
        return
    if c0 == c1:
        r_lo, r_hi = sorted((r0, r1))
        grid[r_lo : r_hi + 1, c0] = GREY
        return
    raise ValueError("segments must be axis-aligned")


def _place_split_chamber(
    grid: np.ndarray,
    side: str,
    top: int,
    left: int,
    height: int,
    width: int,
    split_row: int,
    color_vertical: int,
    color_horizontal: int,
) -> list[dict]:
    bottom = top + height - 1
    right = left + width - 1
    if side == "right":
        _draw_segment(grid, top, left, top, right - 2)
        _draw_segment(grid, bottom, left, bottom, right)
        _draw_segment(grid, top, left, bottom, left)
        markers = [
            {"row": 0, "col": right, "color": color_vertical},
            {"row": top, "col": grid.shape[1] - 1, "color": color_horizontal},
        ]
    else:
        _draw_segment(grid, top, left + 2, top, right)
        _draw_segment(grid, bottom, left, bottom, right)
        _draw_segment(grid, top, right, bottom, right)
        markers = [
            {"row": 0, "col": left, "color": color_vertical},
            {"row": top, "col": 0, "color": color_horizontal},
        ]
    return markers


def _place_simple_chamber(
    grid: np.ndarray,
    side: str,
    top: int,
    left: int,
    height: int,
    width: int,
    color_vertical: int,
) -> list[dict]:
    bottom = top + height - 1
    right = left + width - 1
    if side == "right":
        _draw_segment(grid, top, left, top, right - 1)
        _draw_segment(grid, bottom, left, bottom, right)
        _draw_segment(grid, top, left, bottom, left)
        marker = {"row": 0, "col": right, "color": color_vertical}
    else:
        _draw_segment(grid, top, left + 1, top, right)
        _draw_segment(grid, bottom, left, bottom, right)
        _draw_segment(grid, top, right, bottom, right)
        marker = {"row": 0, "col": left, "color": color_vertical}
    return [marker]


def _segment_cells_for_useless(
    r: int, c: int, orientation: str
) -> list[tuple[int, int]]:
    if orientation == "left":
        return [(r - 1, c - 2), (r, c - 2), (r + 1, c - 2)]
    if orientation == "right":
        return [(r - 1, c + 2), (r, c + 2), (r + 1, c + 2)]
    if orientation == "up":
        return [(r - 2, c - 1), (r - 2, c), (r - 2, c + 1)]
    if orientation == "down":
        return [(r + 2, c - 1), (r + 2, c), (r + 2, c + 1)]
    raise ValueError("invalid useless-marker orientation")


def _safe_lines(
    h: int,
    w: int,
    boxes: list[tuple[int, int, int, int]],
    markers: list[dict],
) -> tuple[list[int], list[int]]:
    busy_rows: set[int] = set()
    busy_cols: set[int] = set()
    for top, left, bottom, right in boxes:
        for r in range(max(0, top - 1), min(h - 1, bottom + 1) + 1):
            busy_rows.add(r)
        for c in range(max(0, left - 1), min(w - 1, right + 1) + 1):
            busy_cols.add(c)
    for marker in markers:
        busy_rows.add(int(marker["row"]))
        busy_cols.add(int(marker["col"]))
    return (
        [r for r in range(1, h - 1) if r not in busy_rows],
        [c for c in range(1, w - 1) if c not in busy_cols],
    )


def _place_useless_markers(
    grid: np.ndarray,
    markers: list[dict],
    boxes: list[tuple[int, int, int, int]],
    colors: list[int],
    count: int,
) -> bool:
    h, w = grid.shape
    candidate_rows, candidate_cols = _safe_lines(h, w, boxes, markers)
    used_rows = {int(m["row"]) for m in markers}
    used_cols = {int(m["col"]) for m in markers}

    options = [
        (r, c)
        for r in candidate_rows
        for c in candidate_cols
        if r not in used_rows and c not in used_cols and grid[r, c] == BLACK
    ]
    random.shuffle(options)
    orientations = ["left", "right", "up", "down"]

    for r, c in options:
        if count == 0:
            break
        local_orients = orientations[:]
        random.shuffle(local_orients)
        for orientation in local_orients:
            seg = _segment_cells_for_useless(r, c, orientation)
            if not all(0 <= rr < h and 0 <= cc < w for rr, cc in seg):
                continue
            if any(grid[rr, cc] != BLACK for rr, cc in seg):
                continue
            for rr, cc in seg:
                grid[rr, cc] = GREY
            markers.append({"row": r, "col": c, "color": colors.pop()})
            used_rows.add(r)
            used_cols.add(c)
            count -= 1
            break
        if count == 0:
            break
    return count == 0


def _grey_only_enclosed_count(grid: np.ndarray) -> int:
    grey_mask = np.asarray(grid == GREY, dtype=bool)
    black_mask = ~grey_mask
    enclosed = black_mask & ~_exterior_black(black_mask)
    return len(_components(enclosed))


def _validate_generated_input(grid: np.ndarray, want_tie: bool) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    markers = _marker_list(grid)
    if not (3 <= len(markers) <= 5):
        return False
    if len({m["color"] for m in markers}) != len(markers):
        return False
    grey_cells = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)]
    if any(_min_manhattan_to_grey(grey_cells, m["row"], m["col"]) < 2 for m in markers):
        return False
    if _grey_only_enclosed_count(grid) != 0:
        return False

    try:
        info = _analyze(grid)
    except AssertionError:
        return False

    if len(info["regions"]) not in {2, 3}:
        return False
    if not any(len(region["contributors"]) >= 2 for region in info["regions"]):
        return False
    if len(info["inactive_ids"]) < 1:
        return False
    if not any(region["tie"] for region in info["regions"]) and want_tie:
        return False
    if any(not _hits_grey(grid.shape, info["grey_mask"], marker) for marker in markers):
        return False

    out = info["output"]
    if np.array_equal(out, grid):
        return False
    if len({tuple(region["coords"][0]) for region in info["regions"]}) != len(info["regions"]):
        return False
    return True


def _build_candidate() -> tuple[np.ndarray, bool] | None:
    h = random.randint(16, 17)
    w = random.randint(16, 17)
    want_third_region = random.random() < 0.5
    side = random.choice(["right", "left"])
    colors = random.sample(list(MARKER_COLORS), 5)

    base = np.zeros((h, w), dtype=int)
    markers: list[dict] = []

    main_h = random.randint(5, 6)
    main_w = random.randint(5, 6)
    top1 = random.randint(3, 5)
    if side == "right":
        left1 = random.randint(2, 3)
    else:
        left1 = w - main_w - random.randint(2, 3)
    right1 = left1 + main_w - 1
    bottom1 = top1 + main_h - 1
    if bottom1 >= h - 2:
        return None
    if side == "right" and right1 > w - 3:
        return None
    if side == "left" and left1 < 2:
        return None
    markers.extend(
        _place_split_chamber(
            base,
            side,
            top1,
            left1,
            main_h,
            main_w,
            top1,
            colors.pop(),
            colors.pop(),
        )
    )

    expected_regions = 2
    if want_third_region:
        simple_h = random.randint(4, 5)
        simple_w = random.randint(4, 5)
        top2 = bottom1 + random.randint(2, 3)
        if top2 + simple_h - 1 >= h:
            return None
        if side == "right":
            left2 = min(w - simple_w - 3, left1 + random.randint(0, 1))
        else:
            left2 = max(2, left1 - random.randint(0, 1))
        right2 = left2 + simple_w - 1
        if right2 >= w:
            return None
        if side == "right" and right2 > w - 3:
            return None
        if side == "left" and left2 < 2:
            return None
        markers.extend(
            _place_simple_chamber(
                base,
                side,
                top2,
                left2,
                simple_h,
                simple_w,
                colors.pop(),
            )
        )
        expected_regions = 3

    for marker in markers:
        r, c = int(marker["row"]), int(marker["col"])
        if not (0 <= r < h and 0 <= c < w):
            return None
        if base[r, c] != BLACK:
            return None
        base[r, c] = int(marker["color"])

    if len(_analyze(base)["regions"]) != expected_regions:
        return None

    candidates = [(int(r), int(c)) for r, c in np.argwhere(base == BLACK)]
    random.shuffle(candidates)
    orientations = ["left", "right", "up", "down"]
    for r, c in candidates:
        for orientation in random.sample(orientations, len(orientations)):
            seg = _segment_cells_for_useless(r, c, orientation)
            if not all(0 <= rr < h and 0 <= cc < w for rr, cc in seg):
                continue
            if any(base[rr, cc] != BLACK for rr, cc in seg):
                continue
            trial = base.copy()
            for rr, cc in seg:
                trial[rr, cc] = GREY
            if trial[r, c] != BLACK:
                continue
            trial[r, c] = colors.pop()
            ok = _validate_generated_input(trial, True)
            if not ok:
                colors.append(int(trial[r, c]))
                continue
            info = _analyze(trial)
            colors.append(int(trial[r, c]))
            need_active = 3 if expected_regions == 3 else 2
            if (
                len(info["regions"]) == expected_regions
                and len(info["inactive_ids"]) >= 1
                and len(info["active_ids"]) >= need_active
            ):
                return trial, True
    return None


def generate_input() -> np.ndarray:
    for _ in range(600):
        built = _build_candidate()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid ray-enclosure input")
