from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    arc_utils_path = str(_ARC_UTILS_DIR)
    if arc_utils_path not in sys.path:
        sys.path.insert(0, arc_utils_path)

from arc_utils import Color, crop_bbox, find_components


BG_COLOR = int(Color.BLACK)
CUP_COLOR = int(Color.BLUE)
YELLOW_COLOR = int(Color.YELLOW)
COUNT_TO_FILL = {
    0: int(Color.RED),
    1: int(Color.GREEN),
    2: int(Color.ORANGE),
}

GEN_ATTEMPTS = 400
MASK_ATTEMPTS = 600


def _boundary_reachable(open_mask: np.ndarray) -> np.ndarray:
    h, w = open_mask.shape
    seen = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if open_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if open_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and open_mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return seen


def _cup_interior_and_opening(subgrid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    blue = subgrid == CUP_COLOR
    assert blue.any(), "cup bbox must contain blue pixels"

    top_cols = np.flatnonzero(blue[0])
    assert top_cols.size >= 2, "cup top row must expose an opening between two walls"
    left_wall = int(top_cols.min())
    right_wall = int(top_cols.max())
    assert right_wall - left_wall >= 2, "cup opening must be at least one cell wide"

    closed = blue.copy()
    closed[0, left_wall : right_wall + 1] = True
    outside = _boundary_reachable(~closed)
    interior = (~closed) & (~outside)
    interior[0, left_wall + 1 : right_wall] = ~blue[0, left_wall + 1 : right_wall]

    assert interior.any(), "cup must contain an internal fillable region"
    return interior, (left_wall + 1, right_wall - 1)


def _cross_footprint(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _cross_centers(yellow_mask: np.ndarray, interior_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = yellow_mask.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not (interior_mask[r, c] and yellow_mask[r, c]):
                continue
            if all(
                interior_mask[nr, nc] and yellow_mask[nr, nc]
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            ):
                centers.append((r, c))
    return centers


def _analyze_cups(grid: np.ndarray) -> list[dict]:
    cups = []
    components = find_components(
        np.asarray(grid, dtype=int),
        bg=BG_COLOR,
        colors=[CUP_COLOR],
        connectivity=4,
        monochromatic=True,
    )
    for comp in components:
        bbox = tuple(map(int, comp["bbox"]))
        r0, c0, _r1, _c1 = bbox
        subgrid = crop_bbox(grid, bbox)
        interior_mask, opening_rel = _cup_interior_and_opening(subgrid)
        yellow_mask = subgrid == YELLOW_COLOR
        assert not np.any(yellow_mask & (subgrid == CUP_COLOR))
        assert np.all(~yellow_mask | interior_mask), "yellow pixels must stay inside the cup interior"

        centers_rel = _cross_centers(yellow_mask, interior_mask)
        cross_cells_rel: set[tuple[int, int]] = set()
        for center in centers_rel:
            cross_cells_rel |= _cross_footprint(center)

        yellow_coords_rel = [tuple(map(int, rc)) for rc in np.argwhere(yellow_mask)]
        extra_coords_rel = [coord for coord in yellow_coords_rel if coord not in cross_cells_rel]
        isolated_extra_rel = [
            (r, c)
            for r, c in extra_coords_rel
            if all(
                not (
                    0 <= nr < yellow_mask.shape[0]
                    and 0 <= nc < yellow_mask.shape[1]
                    and yellow_mask[nr, nc]
                )
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            )
        ]

        cups.append(
            {
                "bbox": bbox,
                "subgrid": subgrid,
                "interior_mask": interior_mask,
                "opening_abs": (c0 + opening_rel[0], c0 + opening_rel[1]),
                "cross_centers_rel": centers_rel,
                "cross_count": len(centers_rel),
                "yellow_total": len(yellow_coords_rel),
                "extra_coords_rel": extra_coords_rel,
                "isolated_extra_rel": isolated_extra_rel,
                "shape": tuple(map(int, subgrid.shape)),
                "top_row": r0,
            }
        )
    cups.sort(key=lambda cup: (cup["bbox"][1], cup["bbox"][0]))
    return cups


def _paint_output(grid: np.ndarray, cups: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for cup in cups:
        fill_color = COUNT_TO_FILL.get(int(cup["cross_count"]))
        assert fill_color is not None, "each cup must contain 0, 1, or 2 crosses"

        r0, c0, _r1, _c1 = cup["bbox"]
        cup_view = out[r0 : cup["bbox"][2] + 1, c0 : cup["bbox"][3] + 1]
        cup_view[cup["interior_mask"]] = fill_color

        cap_row = int(cup["top_row"] - 1)
        assert cap_row >= 0, "cups need one row of headroom for the cap layer"
        cap_c0, cap_c1 = map(int, cup["opening_abs"])
        assert np.all(out[cap_row, cap_c0 : cap_c1 + 1] == BG_COLOR)
        out[cap_row, cap_c0 : cap_c1 + 1] = fill_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Count plus-sign markers inside each blue cup, fill the cup by that count, and add a cap above the opening.

    Input structure:
    - A black background contains 3 or 4 blue cup outlines, each an open-top U-shaped container.
    - Inside each cup, yellow pixels occupy some interior cells; some groups form `+` crosses and others are distractors.
    - The cup walls are blue, the opening has no top border, and each cup has one empty row above it for a later cap.

    Rule:
    - For each blue cup, determine its interior region inside the walls.
    - Within that interior only, count yellow cross centers: a yellow cell whose up/down/left/right neighbors are also yellow and all lie inside the same cup interior.
    - Map the cross count to the fill color: 0 -> red, 1 -> green, 2 -> orange.
    - Recolor every non-blue interior cell of that cup to the mapped fill color, then paint the opening span on the row immediately above the cup with the same color.

    Key insight:
    - The final fill color is not read from any single marker pixel; it is derived indirectly by detecting and counting local `+` patterns inside each cup.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    cups = _analyze_cups(g)
    assert 3 <= len(cups) <= 4, "expected 3 or 4 cups"
    return _paint_output(g, cups)


def _all_center_sets(h: int, w: int, count: int) -> list[tuple[tuple[int, int], ...]]:
    centers = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    if count == 0:
        return [tuple()]

    valid: list[tuple[tuple[int, int], ...]] = []
    for combo in combinations(centers, count):
        footprints = [_cross_footprint(center) for center in combo]
        union = set().union(*footprints)
        if len(union) == 5 * count:
            valid.append(tuple(combo))
    random.shuffle(valid)
    return valid


def _has_isolated_extra(mask: np.ndarray, centers: tuple[tuple[int, int], ...]) -> bool:
    cross_cells = set().union(*(_cross_footprint(center) for center in centers)) if centers else set()
    for r, c in np.argwhere(mask):
        coord = (int(r), int(c))
        if coord in cross_cells:
            continue
        if all(
            not (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc])
            for nr, nc in ((coord[0] - 1, coord[1]), (coord[0] + 1, coord[1]), (coord[0], coord[1] - 1), (coord[0], coord[1] + 1))
        ):
            return True
    return False


def _generate_yellow_mask(h: int, w: int, cross_count: int, total_yellow: int) -> np.ndarray | None:
    assert total_yellow > 5 * cross_count

    if cross_count == 0:
        parity = random.randrange(2)
        cells = [(r, c) for r in range(h) for c in range(w) if (r + c) % 2 == parity]
        if len(cells) >= total_yellow:
            random.shuffle(cells)
            mask = np.zeros((h, w), dtype=bool)
            for r, c in cells[:total_yellow]:
                mask[r, c] = True
            return mask

    full_interior = np.ones((h, w), dtype=bool)
    for centers in _all_center_sets(h, w, cross_count):
        mask = np.zeros((h, w), dtype=bool)
        for center in centers:
            for r, c in _cross_footprint(center):
                mask[r, c] = True

        need = total_yellow - int(mask.sum())
        if need <= 0:
            continue
        free_cells = [(r, c) for r in range(h) for c in range(w) if not mask[r, c]]

        for _ in range(MASK_ATTEMPTS):
            trial = mask.copy()
            order = random.sample(free_cells, len(free_cells))
            picked = 0
            for r, c in order:
                if picked == need:
                    break
                trial[r, c] = True
                if set(_cross_centers(trial, full_interior)) != set(centers):
                    trial[r, c] = False
                    continue
                picked += 1
            if picked == need and _has_isolated_extra(trial, centers):
                return trial
    return None


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _attach_mask(spec: dict) -> bool:
    inner_h = int(spec["h"] - 1)
    inner_w = int(spec["w"] - 2)
    for total_yellow in spec["totals"]:
        mask = _generate_yellow_mask(inner_h, inner_w, int(spec["cross_count"]), int(total_yellow))
        if mask is not None:
            spec["yellow_mask"] = mask
            spec["total_yellow"] = int(total_yellow)
            return True
    return False


def _render_cup(grid: np.ndarray, top: int, left: int, spec: dict) -> None:
    h = int(spec["h"])
    w = int(spec["w"])
    right = left + w - 1
    bottom = top + h - 1

    grid[top : bottom + 1, left] = CUP_COLOR
    grid[top : bottom + 1, right] = CUP_COLOR
    grid[bottom, left : right + 1] = CUP_COLOR

    mask = spec["yellow_mask"]
    for r, c in np.argwhere(mask):
        grid[top + int(r), left + 1 + int(c)] = YELLOW_COLOR


def _specs_for_attempt() -> list[dict]:
    shared_total = random.choice([11, 12])
    if random.random() < 0.55:
        specs = [
            {"h": 6, "w": 7, "cross_count": 0, "totals": [shared_total]},
            {"h": 6, "w": 7, "cross_count": 1, "totals": [shared_total]},
            {"h": 6, "w": 7, "cross_count": 2, "totals": [shared_total]},
        ]
    else:
        extra_cross = random.choice([0, 1])
        small_h0 = random.choice([4, 5])
        small_h1 = random.choice([4, 5])
        specs = [
            {"h": 6, "w": 7, "cross_count": 1, "totals": [shared_total]},
            {"h": 6, "w": 7, "cross_count": 2, "totals": [shared_total]},
            {"h": small_h0, "w": 5, "cross_count": 0, "totals": [6, 5, 4, 3]},
            {"h": small_h1, "w": 5, "cross_count": extra_cross, "totals": [8, 7, 6] if extra_cross == 1 else [6, 5, 4, 3]},
        ]

    for spec in specs:
        random.shuffle(spec["totals"])
        if not _attach_mask(spec):
            return []
    random.shuffle(specs)
    return specs


def _validate_semantics(grid: np.ndarray) -> bool:
    cups = _analyze_cups(grid)
    if len(cups) not in (3, 4):
        return False

    if sorted({cup["cross_count"] for cup in cups}) != [0, 1, 2]:
        return False

    for cup in cups:
        if cup["cross_count"] not in COUNT_TO_FILL:
            return False
        if not cup["extra_coords_rel"]:
            return False
        if not cup["isolated_extra_rel"]:
            return False

    for left_cup, right_cup in zip(cups, cups[1:]):
        gap = int(right_cup["bbox"][1] - left_cup["bbox"][3] - 1)
        if gap < 2:
            return False

    if not any(
        a["shape"] == b["shape"] and a["cross_count"] != b["cross_count"]
        for i, a in enumerate(cups)
        for b in cups[i + 1 :]
    ):
        return False

    if not any(
        a["yellow_total"] == b["yellow_total"] and a["cross_count"] != b["cross_count"]
        for i, a in enumerate(cups)
        for b in cups[i + 1 :]
    ):
        return False

    out = main(grid)
    for cup in cups:
        fill = COUNT_TO_FILL[cup["cross_count"]]
        r0, c0, r1, c1 = cup["bbox"]
        interior_view = out[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(interior_view[cup["interior_mask"]] == fill):
            return False
        cap_row = cup["top_row"] - 1
        cap_c0, cap_c1 = cup["opening_abs"]
        if not np.all(out[cap_row, cap_c0 : cap_c1 + 1] == fill):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    specs = _specs_for_attempt()
    if not specs:
        return None

    widths = [int(spec["w"]) for spec in specs]
    heights = [int(spec["h"]) for spec in specs]
    n = len(specs)
    min_width = sum(widths) + 2 * (n - 1)
    max_width = 30
    if min_width > max_width:
        return None
    grid_w = random.randint(min_width, max_width)
    extra_w = grid_w - min_width
    slots = _random_composition(extra_w, n + 1)

    lefts: list[int] = []
    cursor = slots[0]
    for i, width in enumerate(widths):
        lefts.append(cursor)
        cursor += width
        if i < n - 1:
            cursor += 2 + slots[i + 1]

    max_h = max(heights)
    grid_h = random.randint(max_h + 2, min(18, max_h + 8))
    tops = [random.randint(1, grid_h - h) for h in heights]

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for top, left, spec in zip(tops, lefts, specs):
        _render_cup(grid, top, left, spec)

    if not _validate_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cup-fill ARC input")
