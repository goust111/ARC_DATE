from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, get_bg, paste


PALETTE = [Color.BLUE, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
QUADRANTS = ("TL", "TR", "BL", "BR")


def _blank_rows_cols(grid: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == bg)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == bg)]
    return rows, cols


def _find_fold_cross(grid: np.ndarray, bg: int) -> tuple[int, int]:
    rows, cols = _blank_rows_cols(grid, bg)
    assert len(rows) == 1 and len(cols) == 1, "expected exactly one fully blank row and one fully blank column"
    return rows[0], cols[0]


def _layout_from_cross(shape: tuple[int, int], fold_r: int, fold_c: int) -> dict:
    height, width = map(int, shape)
    top_h = int(fold_r)
    bottom_h = int(height - fold_r - 1)
    left_w = int(fold_c)
    right_w = int(width - fold_c - 1)
    assert top_h > 0 and bottom_h > 0 and left_w > 0 and right_w > 0, "cross must split four nonempty quadrants"
    assert top_h != bottom_h and left_w != right_w, "target quadrant must be uniquely larger in each axis"

    target_v = "top" if top_h > bottom_h else "bottom"
    target_h = "left" if left_w > right_w else "right"
    target_shape = (max(top_h, bottom_h), max(left_w, right_w))

    specs = {
        "TL": {"vside": "top", "hside": "left", "r0": 0, "c0": 0, "h": top_h, "w": left_w},
        "TR": {"vside": "top", "hside": "right", "r0": 0, "c0": fold_c + 1, "h": top_h, "w": right_w},
        "BL": {"vside": "bottom", "hside": "left", "r0": fold_r + 1, "c0": 0, "h": bottom_h, "w": left_w},
        "BR": {"vside": "bottom", "hside": "right", "r0": fold_r + 1, "c0": fold_c + 1, "h": bottom_h, "w": right_w},
    }
    target_id = next(qid for qid, spec in specs.items() if spec["vside"] == target_v and spec["hside"] == target_h)

    return {
        "top_h": top_h,
        "bottom_h": bottom_h,
        "left_w": left_w,
        "right_w": right_w,
        "target_v": target_v,
        "target_h": target_h,
        "target_shape": target_shape,
        "target_id": target_id,
        "specs": specs,
    }


def _coverage_origin(qh: int, qw: int, target_shape: tuple[int, int], target_v: str, target_h: str) -> tuple[int, int]:
    target_hh, target_ww = target_shape
    row0 = 0 if target_v == "bottom" else target_hh - qh
    col0 = 0 if target_h == "right" else target_ww - qw
    return int(row0), int(col0)


def _coverage_mask(spec: dict, target_shape: tuple[int, int], target_v: str, target_h: str) -> np.ndarray:
    mask = np.zeros(target_shape, dtype=bool)
    row0, col0 = _coverage_origin(spec["h"], spec["w"], target_shape, target_v, target_h)
    mask[row0 : row0 + spec["h"], col0 : col0 + spec["w"]] = True
    return mask


def _map_quadrant_to_target(
    quad: np.ndarray,
    spec: dict,
    target_shape: tuple[int, int],
    target_v: str,
    target_h: str,
) -> tuple[np.ndarray, int, int]:
    mapped = quad.copy()
    if spec["vside"] != target_v:
        mapped = np.flipud(mapped)
    if spec["hside"] != target_h:
        mapped = np.fliplr(mapped)
    row0, col0 = _coverage_origin(spec["h"], spec["w"], target_shape, target_v, target_h)
    return mapped, row0, col0


def _neighbors4(r: int, c: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    out = []
    height, width = shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            out.append((nr, nc))
    return out


def _random_coord(mask: np.ndarray) -> tuple[int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r, c = coords[random.randrange(len(coords))]
    return int(r), int(c)


def _grow_mask(allowed: np.ndarray, seeds: list[tuple[int, int]], desired: int) -> np.ndarray:
    mask = np.zeros_like(allowed, dtype=bool)
    for r, c in seeds:
        if allowed[r, c]:
            mask[r, c] = True

    while int(mask.sum()) < desired:
        frontier = set()
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c), mask.shape):
                if allowed[nr, nc] and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if frontier:
            nr, nc = random.choice(list(frontier))
            mask[nr, nc] = True
            continue

        remaining = np.argwhere(allowed & ~mask)
        if remaining.size == 0:
            break
        nr, nc = remaining[random.randrange(len(remaining))]
        mask[int(nr), int(nc)] = True

    return mask


def _paint_pattern(mask: np.ndarray) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert len(coords) >= 6, "pattern must have enough cells"
    color_count = random.randint(2, min(4, len(coords)))
    colors = random.sample(PALETTE, color_count)
    seed_cells = random.sample(coords, color_count)

    canvas = np.zeros(mask.shape, dtype=int)
    for r, c in coords:
        best_idx = min(range(color_count), key=lambda idx: abs(r - seed_cells[idx][0]) + abs(c - seed_cells[idx][1]))
        canvas[r, c] = colors[best_idx]

    used = {int(v) for v in canvas[mask]}
    assert len(used) >= 2, "pattern should be visibly multicolor"
    return canvas


def _reserve_cells(
    cells: list[tuple[int, int]],
    candidates: dict[tuple[int, int], list[str]],
    pattern_quads: list[str],
    per_quad: int,
) -> dict[str, list[tuple[int, int]]] | None:
    needs = [qid for qid in pattern_quads for _ in range(per_quad)]
    ordered_cells = sorted(cells, key=lambda cell: len(candidates[cell]))
    chosen = {qid: [] for qid in pattern_quads}

    def backtrack(i: int, used: set[tuple[int, int]]) -> bool:
        if i == len(needs):
            return True

        qid = needs[i]
        options = [cell for cell in ordered_cells if cell not in used and qid in candidates[cell]]
        random.shuffle(options)
        options.sort(key=lambda cell: len(candidates[cell]))
        for cell in options:
            used.add(cell)
            chosen[qid].append(cell)
            if backtrack(i + 1, used):
                return True
            chosen[qid].pop()
            used.remove(cell)
        return False

    if not backtrack(0, set()):
        return None
    return chosen


def _assign_cells(
    pattern_canvas: np.ndarray,
    coverage_by_quad: dict[str, np.ndarray],
    pattern_quads: list[str],
) -> dict[str, np.ndarray] | None:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(pattern_canvas != 0)]
    candidates = {
        cell: [qid for qid in pattern_quads if coverage_by_quad[qid][cell]]
        for cell in cells
    }
    if any(not options for options in candidates.values()):
        return None

    per_quad = 2 if len(cells) >= 2 * len(pattern_quads) + 4 else 1
    reserved = _reserve_cells(cells, candidates, pattern_quads, per_quad)
    if reserved is None:
        return None

    assigned_to: dict[tuple[int, int], str] = {}
    counts = {qid: 0 for qid in pattern_quads}
    for qid, picked in reserved.items():
        for cell in picked:
            assigned_to[cell] = qid
            counts[qid] += 1

    remaining = [cell for cell in cells if cell not in assigned_to]
    random.shuffle(remaining)
    for cell in remaining:
        options = candidates[cell][:]
        random.shuffle(options)
        options.sort(key=lambda qid: counts[qid])
        chosen = options[0]
        assigned_to[cell] = chosen
        counts[chosen] += 1

    fragments = {qid: np.zeros_like(pattern_canvas, dtype=int) for qid in pattern_quads}
    for (r, c), qid in assigned_to.items():
        fragments[qid][r, c] = int(pattern_canvas[r, c])

    if any(np.count_nonzero(fragments[qid]) == 0 for qid in pattern_quads):
        return None
    return fragments


def _choose_eraser(overlap_mask: np.ndarray, max_keep: int) -> np.ndarray | None:
    overlap_cells = [tuple(map(int, rc)) for rc in np.argwhere(overlap_mask)]
    if len(overlap_cells) < 2:
        return None

    max_erase = min(len(overlap_cells), max_keep)
    if max_erase < 2:
        return None
    erase_count = random.randint(2, max_erase)

    seed = random.choice(overlap_cells)
    erase = np.zeros_like(overlap_mask, dtype=bool)
    erase[seed] = True
    while int(erase.sum()) < erase_count:
        frontier = set()
        for r, c in np.argwhere(erase):
            for nr, nc in _neighbors4(int(r), int(c), erase.shape):
                if overlap_mask[nr, nc] and not erase[nr, nc]:
                    frontier.add((nr, nc))
        if frontier:
            nr, nc = random.choice(list(frontier))
            erase[nr, nc] = True
            continue

        remaining = [cell for cell in overlap_cells if not erase[cell]]
        if not remaining:
            break
        nr, nc = random.choice(remaining)
        erase[nr, nc] = True

    return erase


def _target_to_source(
    full_canvas: np.ndarray,
    spec: dict,
    target_shape: tuple[int, int],
    target_v: str,
    target_h: str,
) -> np.ndarray:
    row0, col0 = _coverage_origin(spec["h"], spec["w"], target_shape, target_v, target_h)
    patch = full_canvas[row0 : row0 + spec["h"], col0 : col0 + spec["w"]].copy()
    if spec["vside"] != target_v:
        patch = np.flipud(patch)
    if spec["hside"] != target_h:
        patch = np.fliplr(patch)
    return patch


def _local_to_target(
    spec: dict,
    local_r: int,
    local_c: int,
    target_shape: tuple[int, int],
    target_v: str,
    target_h: str,
) -> tuple[int, int]:
    row0, col0 = _coverage_origin(spec["h"], spec["w"], target_shape, target_v, target_h)
    mapped_r = spec["h"] - 1 - local_r if spec["vside"] != target_v else local_r
    mapped_c = spec["w"] - 1 - local_c if spec["hside"] != target_h else local_c
    return int(row0 + mapped_r), int(col0 + mapped_c)


def _nontrivial_span(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.max() - rows.min()) >= 2 and int(cols.max() - cols.min()) >= 2


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Fold the four unequal blank-cross quadrants onto the largest quadrant and erase with the red fold-in.

    Input structure:
    - The grid has exactly one fully background row and one fully background column, forming a blank cross.
    - The cross splits the grid into four unequal quadrants; three contain complementary non-red pattern fragments.
    - The remaining quadrant contains only red foreground cells, and red does not appear elsewhere.

    Rule:
    - Use the blank row and column as fold lines and determine the target as the quadrant larger on both axes.
    - Mirror each nonblank quadrant onto that target canvas, aligning its fold-adjacent edge to the target fold edge.
    - Union the three pattern quadrants, then use the mirrored red quadrant as an eraser that turns overlapping cells back to background.

    Key insight:
    - The blank cross is a pair of fold creases, not a separator to crop away; unequal side lengths force mirror-fold alignment, and the red quadrant contributes subtraction instead of addition.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    bg = get_bg(grid)
    fold_r, fold_c = _find_fold_cross(grid, bg)
    layout = _layout_from_cross(grid.shape, fold_r, fold_c)
    target_shape = layout["target_shape"]
    target_v = layout["target_v"]
    target_h = layout["target_h"]

    quadrants = {}
    for qid, spec in layout["specs"].items():
        quadrants[qid] = grid[spec["r0"] : spec["r0"] + spec["h"], spec["c0"] : spec["c0"] + spec["w"]].copy()

    red_quads = []
    for qid, quad in quadrants.items():
        fg = quad[quad != bg]
        if fg.size > 0 and np.all(fg == Color.RED):
            red_quads.append(qid)
    assert len(red_quads) == 1, "expected exactly one all-red eraser quadrant"
    red_qid = red_quads[0]

    out = np.full(target_shape, bg, dtype=int)
    eraser = np.zeros(target_shape, dtype=bool)

    for qid, quad in quadrants.items():
        mapped, row0, col0 = _map_quadrant_to_target(quad, layout["specs"][qid], target_shape, target_v, target_h)
        region = out[row0 : row0 + mapped.shape[0], col0 : col0 + mapped.shape[1]]
        if qid == red_qid:
            red_mask = mapped == Color.RED
            eraser[row0 : row0 + mapped.shape[0], col0 : col0 + mapped.shape[1]] |= red_mask
            continue

        paint_mask = mapped != bg
        assert not np.any(mapped == Color.RED), "red must be reserved for the eraser quadrant"
        assert np.all(region[paint_mask] == bg), "pattern quadrants must be complementary after folding"
        region[paint_mask] = mapped[paint_mask]

    out[eraser] = bg
    return out


def _compose_grid(layout: dict, target_shape: tuple[int, int], fragments: dict[str, np.ndarray], red_qid: str, erase_mask: np.ndarray) -> np.ndarray:
    grid = np.zeros((layout["top_h"] + layout["bottom_h"] + 1, layout["left_w"] + layout["right_w"] + 1), dtype=int)
    for qid in QUADRANTS:
        if qid == red_qid:
            red_canvas = np.zeros(target_shape, dtype=int)
            red_canvas[erase_mask] = Color.RED
            quad = _target_to_source(red_canvas, layout["specs"][qid], target_shape, layout["target_v"], layout["target_h"])
        else:
            quad = _target_to_source(fragments[qid], layout["specs"][qid], target_shape, layout["target_v"], layout["target_h"])
        paste(grid, quad, layout["specs"][qid]["r0"], layout["specs"][qid]["c0"], transparent=Color.BLACK)
    return grid


def _add_support_cell(
    kind: str,
    index: int,
    layout: dict,
    target_shape: tuple[int, int],
    fragments: dict[str, np.ndarray],
    red_qid: str,
    erase_mask: np.ndarray,
    full_pattern: np.ndarray,
) -> bool:
    if kind == "row":
        if index < layout["top_h"]:
            candidate_quads = ["TL", "TR"]
        elif index > layout["top_h"]:
            candidate_quads = ["BL", "BR"]
        else:
            return False
    else:
        if index < layout["left_w"]:
            candidate_quads = ["TL", "BL"]
        elif index > layout["left_w"]:
            candidate_quads = ["TR", "BR"]
        else:
            return False

    candidate_quads = [qid for qid in candidate_quads if qid != red_qid]
    random.shuffle(candidate_quads)

    for prefer_clean in (True, False):
        for qid in candidate_quads:
            spec = layout["specs"][qid]
            if kind == "row":
                local_r = index - spec["r0"]
                if not (0 <= local_r < spec["h"]):
                    continue
                positions = [(local_r, lc) for lc in range(spec["w"])]
            else:
                local_c = index - spec["c0"]
                if not (0 <= local_c < spec["w"]):
                    continue
                positions = [(lr, local_c) for lr in range(spec["h"])]

            random.shuffle(positions)
            for local_r, local_c in positions:
                tr, tc = _local_to_target(spec, local_r, local_c, target_shape, layout["target_v"], layout["target_h"])
                if full_pattern[tr, tc] != 0:
                    continue
                if prefer_clean and erase_mask[tr, tc]:
                    continue

                neighbor_colors = [
                    int(full_pattern[nr, nc])
                    for nr, nc in _neighbors4(tr, tc, target_shape)
                    if full_pattern[nr, nc] not in (0, Color.RED)
                ]
                color = random.choice(neighbor_colors) if neighbor_colors else random.choice(PALETTE)
                fragments[qid][tr, tc] = color
                full_pattern[tr, tc] = color
                return True
    return False


def _repair_blank_cross(
    layout: dict,
    target_shape: tuple[int, int],
    fragments: dict[str, np.ndarray],
    red_qid: str,
    erase_mask: np.ndarray,
    full_pattern: np.ndarray,
) -> np.ndarray | None:
    for _ in range(80):
        grid = _compose_grid(layout, target_shape, fragments, red_qid, erase_mask)
        rows, cols = _blank_rows_cols(grid, 0)
        extra_rows = [r for r in rows if r != layout["top_h"]]
        extra_cols = [c for c in cols if c != layout["left_w"]]
        if not extra_rows and not extra_cols:
            return grid

        progressed = False
        for row in extra_rows:
            progressed |= _add_support_cell("row", row, layout, target_shape, fragments, red_qid, erase_mask, full_pattern)
        for col in extra_cols:
            progressed |= _add_support_cell("col", col, layout, target_shape, fragments, red_qid, erase_mask, full_pattern)
        if not progressed:
            return None
    return None


def _build_instance() -> tuple[np.ndarray, np.ndarray, dict] | None:
    target_v = random.choice(["top", "bottom"])
    target_h = random.choice(["left", "right"])

    for _ in range(80):
        target_hh = random.randint(6, 14)
        other_hh = random.randint(3, target_hh - 1)
        target_ww = random.randint(6, 14)
        other_ww = random.randint(3, target_ww - 1)
        if target_hh + other_hh + 1 > 30 or target_ww + other_ww + 1 > 30:
            continue

        top_h = target_hh if target_v == "top" else other_hh
        bottom_h = target_hh if target_v == "bottom" else other_hh
        left_w = target_ww if target_h == "left" else other_ww
        right_w = target_ww if target_h == "right" else other_ww
        areas = [top_h * left_w, top_h * right_w, bottom_h * left_w, bottom_h * right_w]
        if len(set(areas)) != 4:
            continue
        break
    else:
        return None

    height = top_h + bottom_h + 1
    width = left_w + right_w + 1
    fold_r = top_h
    fold_c = left_w
    layout = _layout_from_cross((height, width), fold_r, fold_c)
    target_shape = layout["target_shape"]

    red_qid = random.choice(QUADRANTS)
    pattern_quads = [qid for qid in QUADRANTS if qid != red_qid]
    coverage_by_quad = {
        qid: _coverage_mask(spec, target_shape, layout["target_v"], layout["target_h"])
        for qid, spec in layout["specs"].items()
    }
    allowed = np.zeros(target_shape, dtype=bool)
    for qid in pattern_quads:
        allowed |= coverage_by_quad[qid]

    red_coverage = coverage_by_quad[red_qid]
    red_overlap = allowed & red_coverage
    if int(allowed.sum()) < 12 or int(red_overlap.sum()) < 4:
        return None

    seeds = []
    for qid in pattern_quads:
        seed = _random_coord(coverage_by_quad[qid] & allowed)
        if seed is not None:
            seeds.append(seed)
    overlap_seed = _random_coord(red_overlap)
    if overlap_seed is not None:
        seeds.append(overlap_seed)
    if not seeds:
        return None

    min_cells = max(12, len(pattern_quads) * 4)
    max_cells = min(int(allowed.sum()), max(min_cells + 3, int(target_shape[0] * target_shape[1] * 0.42)))
    if max_cells < min_cells:
        return None
    desired_cells = random.randint(min_cells, max_cells)
    pattern_mask = _grow_mask(allowed, seeds, desired_cells)
    if int(pattern_mask.sum()) < min_cells or not _nontrivial_span(pattern_mask):
        return None
    if int(np.count_nonzero(pattern_mask & red_overlap)) < 2:
        return None

    full_pattern = _paint_pattern(pattern_mask)
    fragments = _assign_cells(full_pattern, coverage_by_quad, pattern_quads)
    if fragments is None:
        return None

    erase_mask = _choose_eraser((full_pattern != 0) & red_coverage, max_keep=max(2, int(np.count_nonzero(full_pattern) * 0.45)))
    if erase_mask is None:
        return None

    expected = full_pattern.copy()
    expected[erase_mask] = 0
    if np.count_nonzero(expected) < 6:
        return None
    if np.array_equal(expected, full_pattern):
        return None
    if len({int(v) for v in expected[expected != 0]}) < 2:
        return None

    grid = _repair_blank_cross(layout, target_shape, fragments, red_qid, erase_mask, full_pattern)
    if grid is None:
        return None
    expected = full_pattern.copy()
    expected[erase_mask] = 0
    if np.count_nonzero(expected) < 6 or len({int(v) for v in expected[expected != 0]}) < 2:
        return None

    metadata = {
        "fold_r": fold_r,
        "fold_c": fold_c,
        "layout": layout,
        "red_qid": red_qid,
        "pattern_quads": pattern_quads,
        "full_pattern": full_pattern,
        "erase_mask": erase_mask,
        "expected": expected,
    }
    return grid, expected, metadata


def _validate_generated(inp: np.ndarray, expected: np.ndarray, meta: dict) -> None:
    assert isinstance(inp, np.ndarray) and isinstance(expected, np.ndarray)
    assert inp.ndim == 2 and expected.ndim == 2
    assert max(inp.shape) <= 30 and max(expected.shape) <= 30
    assert np.array_equal(main(inp), expected), "solver must match latent construction"

    bg = 0
    rows, cols = _blank_rows_cols(inp, bg)
    assert rows == [meta["fold_r"]] and cols == [meta["fold_c"]], "input must contain exactly one blank fold row and one blank fold column"

    layout = meta["layout"]
    quadrant_areas = [layout["specs"][qid]["h"] * layout["specs"][qid]["w"] for qid in QUADRANTS]
    assert len(set(quadrant_areas)) == 4, "quadrant areas must all differ"

    red_seen_elsewhere = False
    for qid, spec in layout["specs"].items():
        quad = inp[spec["r0"] : spec["r0"] + spec["h"], spec["c0"] : spec["c0"] + spec["w"]]
        fg = quad[quad != bg]
        if qid == meta["red_qid"]:
            assert fg.size > 0 and np.all(fg == Color.RED), "eraser quadrant must be nonempty and all red"
        else:
            assert np.count_nonzero(quad) > 0, "each pattern quadrant must contribute a fragment"
            red_seen_elsewhere |= bool(np.any(quad == Color.RED))
    assert not red_seen_elsewhere, "red must be unique to the eraser quadrant"

    assert np.count_nonzero(meta["erase_mask"]) >= 2, "eraser must affect the union"
    assert np.count_nonzero(meta["full_pattern"]) > np.count_nonzero(expected), "erasing must remove at least one cell"
    assert len({int(v) for v in meta["full_pattern"][meta["full_pattern"] != 0]}) >= 2
    assert len({int(v) for v in expected[expected != 0]}) >= 2


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _build_instance()
        if built is None:
            continue
        grid, expected, meta = built
        try:
            _validate_generated(grid, expected, meta)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid blank-cross fold-and-erase ARC instance")
