# Combo variant of b99e7126 (index 22)
# combo_spec: D4/1+D16/1
# Name: 反向分形 + 仅覆盖背景区域

# Design analysis for combo D4/1 + D16/1
# a) Concrete behavior trace:
#    - Source task: a tiny anomalous 3x3 cell is expanded into a 3x3 block of cells.
#    - Fused task here: a visible 3x3 block of repeated marker cells is the large-scale source,
#      and it is shrunk back into one unique defect cell. The shrink is not a full replacement:
#      only defect-background pixels are repainted, while defect-foreground pixels stay.
#    - Example trace: if the source block has marker cells at (0,1),(1,0),(1,2),(2,1), then the
#      defect cell receives that same cross-shaped 3x3 mask in miniature, but only on its shared
#      background pixels. Existing defect spikes remain, so the result is a hybrid.
# b) Complexity calibration:
#    - A direct "copy the normal cell into the defect" would make D4 inert because cell interiors
#      are already 3x3. To keep the inversion observable, the shrunken source is the macro 3x3
#      marker block itself, not a trivial copy of one ordinary cell.
#    - The input keeps exactly three cell patterns with clearly separated counts:
#      majority >> marker-block actives (4) >> defect (1).
# c) Broken assumptions from the original solver:
#    - Original assumption "the rare cell pattern is also the miniature blueprint" is false now.
#      The unique rare cell is the repair target, not the source.
#    - Original assumption "output changes many cells" is false; this combo changes exactly one cell.
#    - Original assumption "overlay fully overwrites target content" is false because D16/1 keeps
#      defect foreground intact.
# d) Placement legality:
#    - The generator places one 3x3 marker block fully inside the 7x7 cell lattice.
#    - The defect cell is forced outside that 3x3 block, so source and target never overlap.
# e) Visual observability:
#    - The macro source is a compact 3x3 block of solid marker cells, easy to read by eye.
#    - The output changes only one cell, and the inserted miniature uses the same marker color as
#      the block, making the shrink direction legible across examples.
# f) Visual cleanliness:
#    - Only one cell is partially overpainted, so there is no multi-layer clutter elsewhere.
#    - Marker cells are solid and repeated, which keeps the source silhouette easy to parse.
# g) Non-degeneracy check:
#    - The critical control variable is the 3x3 source-block mask. The repaired defect cell depends
#      on that mask; it is not recoverable from the ordinary repeated majority cell alone.
#    - The generator enforces both "repaired background pixels exist" and "preserved defect
#      foreground outside the source mask exists", so the output is neither a no-op nor a full copy.
# 9) New solver assumptions guaranteed by the generator:
#    - Exactly one defect pattern occurs once.
#    - Exactly one repeated non-majority marker pattern occurs four times.
#    - Those four marker cells span a full 3x3 bounding box, so their bbox is the shrink source.
#    - The defect background is the majority color inside the defect cell, so selective repainting
#      is well-defined.

from functools import reduce
from math import gcd
import random

import numpy as np

from arc_utils import paste


def _infer_cell_grid_params(g: np.ndarray) -> tuple[int, list[int], list[int]]:
    """
    Infer a regular cell grid separated by full rows/cols of the same line color.
    Returns (step, row_divs, col_divs).
    """
    h, w = g.shape
    line = int(g[0, 0])
    row_divs = [r for r in range(h) if np.all(g[r, :] == line)]
    col_divs = [c for c in range(w) if np.all(g[:, c] == line)]
    assert len(row_divs) >= 2 and len(col_divs) >= 2
    row_divs.sort()
    col_divs.sort()
    drs = [row_divs[i + 1] - row_divs[i] for i in range(len(row_divs) - 1) if row_divs[i + 1] > row_divs[i]]
    dcs = [col_divs[i + 1] - col_divs[i] for i in range(len(col_divs) - 1) if col_divs[i + 1] > col_divs[i]]
    step = reduce(gcd, drs + dcs)
    assert step > 1
    assert row_divs[0] == 0 and col_divs[0] == 0
    assert row_divs[-1] == h - 1 and col_divs[-1] == w - 1
    assert (h - 1) % step == 0 and (w - 1) % step == 0
    return int(step), row_divs, col_divs


def _cell_origin(i: int, j: int, step: int = 4) -> tuple[int, int]:
    return i * step + 1, j * step + 1


def _extract_cell_patterns(g: np.ndarray, step: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    inner = step - 1
    n = (g.shape[0] - 1) // step
    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_pat: list[list[tuple[int, ...]]] = []
    for i in range(n):
        row: list[tuple[int, ...]] = []
        for j in range(n):
            r0, c0 = _cell_origin(i, j, step)
            arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            key = tuple(arr.ravel().tolist())
            if key not in patterns:
                patterns[key] = arr
            row.append(key)
        cell_pat.append(row)
    return patterns, cell_pat


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _most_common_color(arr: np.ndarray) -> int:
    vals, cnts = np.unique(arr, return_counts=True)
    return int(vals[cnts.argmax()])


def _repair_defect(defect_arr: np.ndarray, source_mask: np.ndarray, stamp_color: int) -> np.ndarray:
    repaired = defect_arr.copy()
    defect_bg = _most_common_color(defect_arr)
    repaint = (defect_arr == defect_bg) & source_mask
    repaired[repaint] = stamp_color
    return repaired


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink a macro 3x3 marker pattern back into a single damaged cell, repairing only its background.

    Input structure:
    - The board is split into 3x3 interior cells by uniform separator lines.
    - Most cells share one ordinary pattern.
    - Four identical solid marker cells form a single 3x3 block-shaped macro pattern elsewhere.
    - One unique defect cell uses a different foreground on a shared background.

    Rule:
    - Read the 3x3 macro pattern traced by the marker-cell positions.
    - Compress that block into the unique defect cell as a 3x3 miniature.
    - Apply the miniature selectively: paint only where the defect currently shows background.
    - Keep all pre-existing defect foreground pixels unchanged.

    Key insight:
    - The direction of self-similarity is reversed: the visible large-scale cell arrangement is the
      source, and the destination is a single cell. The final cell is a hybrid because only its
      background is repaired.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    inner = step - 1
    assert inner == 3

    patterns, cell_pat = _extract_cell_patterns(g, step)
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1

    assert len(counts) == 3, "expected majority, marker, and defect cell patterns"
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    defect_candidates = [pat for pat, cnt in counts.items() if cnt == 1]
    assert len(defect_candidates) == 1, "need exactly one defect cell"
    defect_pat = defect_candidates[0]
    marker_candidates = [pat for pat in counts if pat not in (majority_pat, defect_pat)]
    assert len(marker_candidates) == 1, "need exactly one repeated marker pattern"
    marker_pat = marker_candidates[0]

    marker_cells = []
    defect_pos = None
    for i, row in enumerate(cell_pat):
        for j, pat in enumerate(row):
            if pat == marker_pat:
                marker_cells.append((i, j))
            elif pat == defect_pat:
                defect_pos = (i, j)
    marker_coords = np.array(marker_cells, dtype=int)
    assert marker_coords.shape == (4, 2), "generator guarantees exactly four marker cells"
    assert defect_pos is not None

    r0, c0, r1, c1 = _bbox(marker_coords)
    assert r1 - r0 == 2 and c1 - c0 == 2, "marker cells must span one 3x3 macro block"
    source_mask = np.zeros((3, 3), dtype=bool)
    for r, c in marker_cells:
        source_mask[r - r0, c - c0] = True

    marker_arr = patterns[marker_pat]
    marker_colors = np.unique(marker_arr)
    assert len(marker_colors) == 1, "marker cells are solid-color stamps"
    stamp_color = int(marker_colors[0])

    defect_arr = patterns[defect_pat]
    repaired = _repair_defect(defect_arr, source_mask, stamp_color)

    out = g.copy()
    rr, cc = _cell_origin(*defect_pos, step)
    out[rr : rr + inner, cc : cc + inner] = repaired
    return out


CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = CELL_INNER + 1
GRID_SIZE = CELL_GRID_N * CELL_STEP + 1
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
SOURCE_ACTIVE_COUNT = 4
COLOR_POOL = 10
MAX_RETRIES = 400


def _precompute_source_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1 << (MASK_SIZE * MASK_SIZE)):
        mask = np.array([(bits >> k) & 1 for k in range(MASK_SIZE * MASK_SIZE)], dtype=bool).reshape(MASK_SIZE, MASK_SIZE)
        if int(mask.sum()) != SOURCE_ACTIVE_COUNT:
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) < 3 or len(cols) < 3:
            continue
        masks.append(mask)
    return masks


def _precompute_defect_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1 << (MASK_SIZE * MASK_SIZE)):
        mask = np.array([(bits >> k) & 1 for k in range(MASK_SIZE * MASK_SIZE)], dtype=bool).reshape(MASK_SIZE, MASK_SIZE)
        fg = int(mask.sum())
        if fg < 3 or fg > 4:
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) < 2 or len(cols) < 2:
            continue
        masks.append(mask)
    return masks


SOURCE_MASKS = _precompute_source_masks()
DEFECT_MASKS = _precompute_defect_masks()


def _random_majority_pattern(bg: int, fg: int) -> np.ndarray:
    arr = np.full((CELL_INNER, CELL_INNER), bg, dtype=int)
    cells = [(r, c) for r in range(CELL_INNER) for c in range(CELL_INNER)]
    random.shuffle(cells)
    for r, c in cells[: random.choice((3, 4))]:
        arr[r, c] = fg
    return arr


def _choose_defect_mask(source_mask: np.ndarray) -> np.ndarray | None:
    candidates = []
    for mask in DEFECT_MASKS:
        repair_count = int((source_mask & ~mask).sum())
        preserve_count = int((~source_mask & mask).sum())
        if repair_count >= 2 and preserve_count >= 1:
            candidates.append(mask)
    if not candidates:
        return None
    return random.choice(candidates)


def _build_input(separator: int, majority: np.ndarray, marker_arr: np.ndarray, defect_arr: np.ndarray,
                 source_mask: np.ndarray, top: int, left: int, defect_pos: tuple[int, int]) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j, CELL_STEP))
    for r, c in np.argwhere(source_mask):
        paste(grid, marker_arr, *_cell_origin(top + int(r), left + int(c), CELL_STEP))
    paste(grid, defect_arr, *_cell_origin(*defect_pos, CELL_STEP))
    return grid


def _build_output(inp: np.ndarray, defect_pos: tuple[int, int], source_mask: np.ndarray, stamp_color: int) -> np.ndarray:
    out = inp.copy()
    r0, c0 = _cell_origin(*defect_pos, CELL_STEP)
    defect_arr = out[r0 : r0 + CELL_INNER, c0 : c0 + CELL_INNER].copy()
    out[r0 : r0 + CELL_INNER, c0 : c0 + CELL_INNER] = _repair_defect(defect_arr, source_mask, stamp_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    separator, bg, majority_fg, marker_fg, defect_fg = random.sample(range(COLOR_POOL), 5)
    majority = _random_majority_pattern(bg, majority_fg)
    marker_arr = np.full((CELL_INNER, CELL_INNER), marker_fg, dtype=int)

    source_mask = random.choice(SOURCE_MASKS)
    defect_mask = _choose_defect_mask(source_mask)
    if defect_mask is None:
        return None
    defect_arr = np.where(defect_mask, defect_fg, bg).astype(int)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    source_block = {(top + r, left + c) for r in range(MASK_SIZE) for c in range(MASK_SIZE)}

    defect_choices = [
        (i, j)
        for i in range(CELL_GRID_N)
        for j in range(CELL_GRID_N)
        if (i, j) not in source_block
    ]
    defect_pos = random.choice(defect_choices)

    inp = _build_input(separator, majority, marker_arr, defect_arr, source_mask, top, left, defect_pos)
    out = _build_output(inp, defect_pos, source_mask, marker_fg)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    changed = solved != inp
    if int(changed.sum()) < 2:
        return None
    dr, dc = _cell_origin(*defect_pos, CELL_STEP)
    cell_before = inp[dr : dr + CELL_INNER, dc : dc + CELL_INNER]
    cell_after = solved[dr : dr + CELL_INNER, dc : dc + CELL_INNER]
    repaired_count = int(((cell_before == bg) & (cell_after == marker_fg)).sum())
    preserved_count = int(((cell_before == defect_fg) & (cell_after == defect_fg)).sum())
    if repaired_count < 2 or preserved_count < 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 D4/1+D16/1 input")
