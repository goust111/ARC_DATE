from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


BG = 0
BAR = 5
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def connected_mask(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    seen = np.zeros_like(mask, dtype=bool)
    stack = [tuple(map(int, pts[0]))]
    seen[stack[0]] = True
    count = 0
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        count += 1
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return count == int(mask.sum())


def find_components(grid: np.ndarray, *, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append({"coords": arr, "bbox": bbox_coords(arr), "area": int(arr.shape[0])})
    return comps


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> np.ndarray:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch
    return dst


def _contiguous_span(indices: np.ndarray) -> list[int]:
    vals = [int(x) for x in indices]
    assert vals, "expected at least one projected row/column"
    assert vals == list(range(vals[0], vals[-1] + 1)), "projection rows/cols must form one contiguous span"
    return vals


def _run_down(grid: np.ndarray, row0: int, col: int, color: int) -> int:
    h = grid.shape[0]
    length = 0
    while row0 + length < h and int(grid[row0 + length, col]) == color:
        length += 1
    return length


def _run_right(grid: np.ndarray, row: int, col0: int, color: int) -> int:
    w = grid.shape[1]
    length = 0
    while col0 + length < w and int(grid[row, col0 + length]) == color:
        length += 1
    return length


def _projection_spec(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int], tuple[int, ...], tuple[int, ...], np.ndarray]:
    top_colors = {int(v) for v in grid[0] if int(v) != bg}
    left_colors = {int(v) for v in grid[:, 0] if int(v) != bg}
    candidates = sorted(top_colors & left_colors)
    assert len(candidates) == 1, f"expected one edge-bar color, got {candidates}"
    bar_color = candidates[0]

    target_cols = _contiguous_span(np.where(grid[0] == bar_color)[0])
    target_rows = _contiguous_span(np.where(grid[:, 0] == bar_color)[0])
    col_hist = tuple(_run_down(grid, 0, c, bar_color) for c in target_cols)
    row_hist = tuple(_run_right(grid, r, 0, bar_color) for r in target_rows)

    bar_mask = np.zeros_like(grid, dtype=bool)
    for c, count in zip(target_cols, col_hist):
        bar_mask[:count, c] = True
    for r, count in zip(target_rows, row_hist):
        bar_mask[r, :count] = True

    assert all(count < target_rows[0] for count in col_hist), "top bars must stay above the hidden bbox"
    assert all(count < target_cols[0] for count in row_hist), "left bars must stay left of the hidden bbox"
    return bar_color, target_rows, target_cols, row_hist, col_hist, bar_mask


def _hist_signature(patch: np.ndarray, bg: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    mask = patch != bg
    return (
        tuple(int(x) for x in mask.sum(axis=1)),
        tuple(int(x) for x in mask.sum(axis=0)),
    )


def _templates(grid: np.ndarray, *, bg: int, bar_mask: np.ndarray) -> list[dict]:
    body = grid.copy()
    body[bar_mask] = bg
    out: list[dict] = []
    for comp in find_components(body, bg=bg):
        patch = crop_bbox(body, comp["bbox"])
        row_hist, col_hist = _hist_signature(patch, bg)
        out.append(
            {
                "bbox": comp["bbox"],
                "patch": patch,
                "shape": tuple(map(int, patch.shape)),
                "row_hist": row_hist,
                "col_hist": col_hist,
            }
        )
    return out


def _matching_templates(templates: list[dict], row_hist: tuple[int, ...], col_hist: tuple[int, ...]) -> list[dict]:
    wanted_shape = (len(row_hist), len(col_hist))
    return [
        tpl
        for tpl in templates
        if tpl["shape"] == wanted_shape and tpl["row_hist"] == row_hist and tpl["col_hist"] == col_hist
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the missing object from its top and left projection bars by stamping back the uniquely matching template.

    Input structure:
    - A black background contains several separated multicolor template objects.
    - One hidden object's bbox is indicated only indirectly: gray vertical bars hang from the top edge over its columns, and gray horizontal bars extend from the left edge across its rows.
    - At least two visible templates can share the same bbox size, so size alone does not identify the hidden one.

    Rule:
    - Read the top bars as column pixel counts and the left bars as row pixel counts; the rows and columns carrying bars define the hidden bbox.
    - Remove the gray bar cells from consideration, then crop every visible template object to its own bbox.
    - Find the unique template whose row histogram and column histogram exactly equal the bar lengths, and stamp that full colored template into the hidden bbox.
    - Clear all gray projection bars in the output.

    Key insight:
    - The edge bars are the hidden object's two-axis projection shadow, so the solution must match both row and column counts rather than relying on bbox size.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    _bar_color, target_rows, target_cols, row_hist, col_hist, bar_mask = _projection_spec(g, bg)
    templates = _templates(g, bg=bg, bar_mask=bar_mask)
    matches = _matching_templates(templates, row_hist, col_hist)
    assert len(matches) == 1, f"expected one template matching the projection histograms, got {len(matches)}"

    out = g.copy()
    out[bar_mask] = bg
    paste(out, matches[0]["patch"], target_rows[0], target_cols[0])
    return out


# --------------- generator ---------------

GRID_H_RANGE = (20, 28)
GRID_W_RANGE = (20, 28)
TARGET_H_RANGE = (4, 7)
TARGET_W_RANGE = (4, 7)
TEMPLATE_COUNT_RANGE = (3, 5)
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200
MASK_ATTEMPTS = 200


def _random_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    area = h * w
    min_area = max(h + w - 1, int(np.ceil(area * 0.45)))
    max_area = area - 1
    for _ in range(MASK_ATTEMPTS):
        mask = np.ones((h, w), dtype=bool)
        desired = rng.randint(min_area, max_area)
        cells = [(r, c) for r in range(h) for c in range(w)]
        rng.shuffle(cells)
        progressed = True
        while int(mask.sum()) > desired and progressed:
            progressed = False
            for r, c in cells:
                if int(mask.sum()) <= desired or not mask[r, c]:
                    continue
                if int(mask[r].sum()) == 1 or int(mask[:, c].sum()) == 1:
                    continue
                mask[r, c] = False
                if connected_mask(mask):
                    progressed = True
                else:
                    mask[r, c] = True
        row_hist, col_hist = _hist_signature(mask.astype(int), 0)
        if (
            connected_mask(mask)
            and min(row_hist) > 0
            and min(col_hist) > 0
            and int(mask.sum()) >= min_area
            and not np.all(mask)
            and not (len(set(row_hist)) == 1 and len(set(col_hist)) == 1)
        ):
            return mask
    raise RuntimeError("failed to generate a connected template mask")


def _colorize(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    n_colors = rng.choice([2, 2, 3])
    palette = rng.sample(OBJECT_COLORS, n_colors)
    mode = rng.choice(["row", "col", "diag", "checker"])
    offset = rng.randrange(n_colors)
    patch = np.full(mask.shape, BG, dtype=int)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if mode == "row":
            idx = (rr + offset) % n_colors
        elif mode == "col":
            idx = (cc + offset) % n_colors
        elif mode == "diag":
            idx = (rr + cc + offset) % n_colors
        else:
            idx = ((2 * rr) + cc + offset) % n_colors
        patch[rr, cc] = palette[idx]

    used = sorted(set(int(v) for v in patch.ravel()) - {BG})
    if len(used) < min(2, int(mask.sum())):
        cells = [tuple(map(int, x)) for x in np.argwhere(mask)]
        rng.shuffle(cells)
        for color, (r, c) in zip(palette[1:], cells):
            patch[r, c] = color
    return patch


def _random_patch(h: int, w: int, rng: random.Random) -> np.ndarray:
    return _colorize(_random_mask(h, w, rng), rng)


def _same_hist(a: np.ndarray, b: np.ndarray) -> bool:
    return _hist_signature(a, BG) == _hist_signature(b, BG)


def _blocked_with_padding(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        r0 = max(0, rr - 1)
        r1 = min(h, rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(w, cc + 2)
        out[r0:r1, c0:c1] = True
    return out


def _can_place(blocked: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    if r0 < 1 or c0 < 1 or r0 + h >= blocked.shape[0] or c0 + w >= blocked.shape[1]:
        return False
    return not blocked[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1].any()


def _place_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, rng: random.Random) -> bool:
    h, w = patch.shape
    candidates = [(r, c) for r in range(1, grid.shape[0] - h) for c in range(1, grid.shape[1] - w)]
    rng.shuffle(candidates)
    for r0, c0 in candidates[:PLACEMENT_ATTEMPTS]:
        if not _can_place(blocked, patch, r0, c0):
            continue
        paste(grid, patch, r0, c0)
        blocked[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] = True
        return True
    return False


def _make_bar_mask(shape: tuple[int, int], target_r0: int, target_c0: int, row_hist: tuple[int, ...], col_hist: tuple[int, ...]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for dc, count in enumerate(col_hist):
        mask[:count, target_c0 + dc] = True
    for dr, count in enumerate(row_hist):
        mask[target_r0 + dr, :count] = True
    return mask


def _expected_output(inp: np.ndarray, target_patch: np.ndarray) -> np.ndarray:
    bg = get_bg(inp)
    _bar_color, target_rows, target_cols, _row_hist, _col_hist, bar_mask = _projection_spec(inp, bg)
    out = inp.copy()
    out[bar_mask] = bg
    paste(out, target_patch, target_rows[0], target_cols[0])
    return out


def _semantic_ok(inp: np.ndarray, target_patch: np.ndarray) -> bool:
    bg = get_bg(inp)
    try:
        _bar_color, target_rows, target_cols, row_hist, col_hist, bar_mask = _projection_spec(inp, bg)
    except AssertionError:
        return False

    templates = _templates(inp, bg=bg, bar_mask=bar_mask)
    same_size = [tpl for tpl in templates if tpl["shape"] == tuple(map(int, target_patch.shape))]
    matches = _matching_templates(templates, row_hist, col_hist)

    if len(matches) != 1:
        return False
    if not np.array_equal(matches[0]["patch"], target_patch):
        return False
    if len(same_size) < 2:
        return False
    if not any(
        tpl["row_hist"] != row_hist or tpl["col_hist"] != col_hist
        for tpl in same_size
        if not np.array_equal(tpl["patch"], target_patch)
    ):
        return False
    if np.any(inp[target_rows[0] : target_rows[-1] + 1, target_cols[0] : target_cols[-1] + 1] != bg):
        return False
    return np.array_equal(main(inp), _expected_output(inp, target_patch))


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    H = rng.randint(*GRID_H_RANGE)
    W = rng.randint(*GRID_W_RANGE)
    target_h = rng.randint(*TARGET_H_RANGE)
    target_w = rng.randint(*TARGET_W_RANGE)

    target_patch = _random_patch(target_h, target_w, rng)
    target_row_hist, target_col_hist = _hist_signature(target_patch, BG)

    max_top = max(target_col_hist)
    max_left = max(target_row_hist)
    if H - target_h - 2 < max_top + 1 or W - target_w - 2 < max_left + 1:
        return None

    target_r0 = rng.randint(max_top + 1, H - target_h - 2)
    target_c0 = rng.randint(max_left + 1, W - target_w - 2)
    bar_mask = _make_bar_mask((H, W), target_r0, target_c0, target_row_hist, target_col_hist)
    target_box_mask = np.zeros((H, W), dtype=bool)
    target_box_mask[target_r0 : target_r0 + target_h, target_c0 : target_c0 + target_w] = True

    grid = np.full((H, W), BG, dtype=int)
    grid[bar_mask] = BAR
    blocked = _blocked_with_padding(bar_mask | target_box_mask)

    patches = [target_patch]
    for _ in range(MASK_ATTEMPTS):
        decoy = _random_patch(target_h, target_w, rng)
        if not _same_hist(target_patch, decoy):
            patches.append(decoy)
            break
    if len(patches) != 2:
        return None

    total_templates = rng.randint(*TEMPLATE_COUNT_RANGE)
    while len(patches) < total_templates:
        h = rng.randint(3, 7)
        w = rng.randint(3, 7)
        patch = _random_patch(h, w, rng)
        if _hist_signature(patch, BG) == (target_row_hist, target_col_hist):
            continue
        patches.append(patch)

    rng.shuffle(patches)
    for patch in patches:
        if not _place_patch(grid, blocked, patch, rng):
            return None

    if not _semantic_ok(grid, target_patch):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
