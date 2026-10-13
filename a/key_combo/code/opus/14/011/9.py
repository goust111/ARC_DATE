from __future__ import annotations

import random
from collections import Counter

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors(connectivity: int) -> list[tuple[int, int]]:
    if int(connectivity) == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    nbrs = _neighbors(connectivity)
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in nbrs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_non_bg(patch: np.ndarray, bg: int = 0) -> np.ndarray:
    coords = np.argwhere(patch != int(bg))
    if coords.size == 0:
        return np.zeros((1, 1), dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _largest_enclosed_component_bbox(mask: np.ndarray, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    h, w = mask.shape
    best_bbox: tuple[int, int, int, int] | None = None
    best_area = -1
    for comp in _components(mask.astype(bool), connectivity):
        rs = comp[:, 0]
        cs = comp[:, 1]
        if bool((rs == 0).any() or (rs == h - 1).any() or (cs == 0).any() or (cs == w - 1).any()):
            continue
        if int(len(comp)) > best_area:
            best_area = int(len(comp))
            best_bbox = _bbox(comp)
    return best_bbox


def _frame_base_color(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> int:
    fr0, fc0, fr1, fc1 = frame_bbox
    border = []
    border.extend(map(int, grid[fr0, fc0 : fc1 + 1]))
    border.extend(map(int, grid[fr1, fc0 : fc1 + 1]))
    border.extend(map(int, grid[fr0 + 1 : fr1, fc0]))
    border.extend(map(int, grid[fr0 + 1 : fr1, fc1]))
    return int(Counter(border).most_common(1)[0][0])


def _find_divider_column(grid: np.ndarray, bg: int, frame_left: int) -> int:
    h, _ = grid.shape
    candidates: list[int] = []
    for c in range(frame_left):
        col = grid[:, c]
        if int(np.count_nonzero(col != int(bg))) != h:
            continue
        vals = set(map(int, col))
        if len(vals) == 1 and int(next(iter(vals))) != int(bg):
            candidates.append(c)
    assert candidates, "expected a full-height divider column left of the frame"
    return int(max(candidates))


def _extract_variants(grid: np.ndarray, divider_c: int, bg: int) -> dict[int, np.ndarray]:
    left_mask = (grid[:, :divider_c] != int(bg))
    variants: dict[int, np.ndarray] = {}
    for comp in _components(left_mask, connectivity=8):
        r0, c0, r1, c1 = _bbox(comp)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        label = int(patch[0, 0])
        body = patch.copy()
        body[0, 0] = int(bg)
        body = _crop_non_bg(body, bg)
        variants[label] = body
    return variants


def _stamp_centered(
    canvas: np.ndarray,
    patch: np.ndarray,
    center_r: int,
    center_c: int,
    bounds: tuple[int, int, int, int],
    bg: int,
) -> None:
    ir0, ic0, ir1, ic1 = bounds
    h, w = patch.shape
    top = int(center_r) - h // 2
    left = int(center_c) - w // 2
    for rr in range(h):
        for cc in range(w):
            val = int(patch[rr, cc])
            if val == int(bg):
                continue
            gr = top + rr
            gc = left + cc
            if ir0 <= gr <= ir1 and ic0 <= gc <= ic1:
                canvas[gr, gc] = val


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    inner = _largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner is not None, "expected a framed interior"
    ir0, ic0, ir1, ic1 = inner
    frame_bbox = (ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1)
    fr0, fc0, fr1, fc1 = frame_bbox
    divider_c = _find_divider_column(g, bg, fc0)
    frame_base = _frame_base_color(g, frame_bbox)

    row_targets: list[tuple[int, int]] = []
    for r in range(ir0, ir1 + 1):
        left = int(g[r, fc0])
        right = int(g[r, fc1])
        if left == right and left not in (int(bg), int(frame_base)):
            row_targets.append((r, left))

    col_targets: list[int] = []
    for c in range(ic0, ic1 + 1):
        top = int(g[fr0, c])
        bottom = int(g[fr1, c])
        if top == bottom and top not in (int(bg), int(frame_base)):
            col_targets.append(c)

    variants = _extract_variants(g, divider_c, bg)
    return {
        "grid": g,
        "bg": int(bg),
        "inner_bbox": (ir0, ic0, ir1, ic1),
        "frame_bbox": frame_bbox,
        "divider_c": int(divider_c),
        "frame_base": int(frame_base),
        "row_targets": sorted((int(r), int(color)) for r, color in row_targets),
        "col_targets": sorted(map(int, col_targets)),
        "variants": variants,
    }


def _apply_rule(
    scene: dict,
    *,
    row_order: list[tuple[int, int]] | None = None,
    fixed_variant: np.ndarray | None = None,
    erase_left: bool = True,
) -> np.ndarray:
    out = scene["grid"].copy()
    bg = int(scene["bg"])
    divider_c = int(scene["divider_c"])
    if erase_left:
        out[:, :divider_c] = int(bg)

    rows = scene["row_targets"] if row_order is None else row_order
    cols = scene["col_targets"]
    inner = scene["inner_bbox"]
    variants = scene["variants"]

    for r, color in rows:
        patch = fixed_variant if fixed_variant is not None else variants[int(color)]
        for c in cols:
            _stamp_centered(out, patch, int(r), int(c), inner, bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy rotated sprite variants into a marked frame using row-color selection.

    Input structure:
    - Black background with a grey separator column and, on the right, a grey rectangular frame whose edge cells may be replaced by colored markers.
    - On the left of the separator are 2-4 multicolor sprite variants stacked vertically; each variant has a one-pixel color label at its top-left corner.
    - Matching left/right frame-edge markers identify target rows, and matching top/bottom frame-edge markers identify target columns.

    Rule:
    - Find the frame interior from the enclosed background rectangle, then read target rows from equal-color left/right edge pairs and target columns from equal-color top/bottom edge pairs.
    - For each target row, use that row's marker color to choose the left-side sprite variant whose label has the same color.
    - Remove the label pixel, recrop the chosen variant, and stamp it centered at every target-column intersection for that row, processing targets in row-major order so later stamps overwrite earlier ones.
    - Erase the entire area left of the separator back to background, while keeping the separator and the frame.

    Key insight:
    - The row marker color is not the output color itself; it is a key that selects which rotated copy of the same base sprite must be placed at that row's frame intersections.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    return _apply_rule(scene)


def _rotation_key(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, arr.shape)), arr.tobytes()


def _make_body(colors: list[int]) -> np.ndarray:
    for _ in range(300):
        h, w = random.choice([(3, 3), (3, 4), (4, 4)])
        palette = random.sample(colors, random.randint(2, min(3, len(colors))))
        body = np.array([[random.choice(palette) for _ in range(w)] for _ in range(h)], dtype=int)
        if len(set(map(int, body.ravel()))) < 2:
            continue
        keys = {_rotation_key(np.rot90(body, k)) for k in range(4)}
        if len(keys) == 4:
            return body
    raise RuntimeError("failed to generate an asymmetric base sprite")


def _safe_center_bounds(size: int, start: int, end: int) -> tuple[int, int]:
    half = size // 2
    tail = size - 1 - half
    return int(start + half), int(end - tail)


def _pick_rows(
    count: int,
    lo: int,
    hi: int,
    overlap_gap: int,
) -> list[int] | None:
    if hi - lo + 1 < count:
        return None
    candidates = list(range(lo, hi + 1))
    for _ in range(400):
        rows = sorted(random.sample(candidates, count))
        if any(rows[i + 1] - rows[i] < overlap_gap for i in range(len(rows) - 1)):
            return rows
    return None


def _pick_cols(count: int, lo: int, hi: int, min_gap: int) -> list[int] | None:
    if hi - lo + 1 < count:
        return None
    candidates = list(range(lo, hi + 1))
    for _ in range(400):
        cols = sorted(random.sample(candidates, count))
        if all(cols[i + 1] - cols[i] >= min_gap for i in range(len(cols) - 1)):
            return cols
    return None


def _place_left_variants(
    grid: np.ndarray,
    patches: list[tuple[int, np.ndarray]],
    left_width: int,
) -> bool:
    h = grid.shape[0]
    gaps = [random.randint(1, 2) for _ in range(max(0, len(patches) - 1))]
    total_h = sum(p.shape[0] for _, p in patches) + sum(gaps)
    if total_h > h:
        return False
    top = random.randint(0, h - total_h)
    order = patches[:]
    random.shuffle(order)
    cur = top
    for idx, (_, patch) in enumerate(order):
        ph, pw = patch.shape
        left = random.randint(0, left_width - pw)
        grid[cur : cur + ph, left : left + pw] = patch
        cur += ph
        if idx < len(order) - 1:
            cur += gaps[idx]
    return True


def _metadata_expected(
    inp: np.ndarray,
    divider_c: int,
    row_pairs: list[tuple[int, int]],
    col_positions: list[int],
    variant_map: dict[int, np.ndarray],
) -> np.ndarray:
    scene = {
        "grid": np.asarray(inp, dtype=int),
        "bg": 0,
        "divider_c": int(divider_c),
        "row_targets": sorted((int(r), int(color)) for r, color in row_pairs),
        "col_targets": sorted(map(int, col_positions)),
        "variants": {int(k): np.asarray(v, dtype=int) for k, v in variant_map.items()},
        "inner_bbox": None,
    }
    parsed = _parse_scene(inp)
    scene["inner_bbox"] = parsed["inner_bbox"]
    return _apply_rule(scene)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    frame_color = 5
    n_variants = random.choice([2, 2, 3, 3, 4])

    usable_colors = [1, 2, 3, 4, 6, 7, 8, 9]
    random.shuffle(usable_colors)
    row_colors = usable_colors[:n_variants]
    col_color = usable_colors[n_variants]
    body_colors = usable_colors[n_variants + 1 :]
    if len(body_colors) < 2:
        return None

    body = _make_body(body_colors)
    rotations = random.sample([0, 1, 2, 3], n_variants)
    variant_map = {int(color): np.rot90(body, int(k)).copy() for color, k in zip(row_colors, rotations)}
    max_h = max(p.shape[0] for p in variant_map.values())
    max_w = max(p.shape[1] for p in variant_map.values())

    interior_h = random.randint(max(9, n_variants + max_h + 1), 13)
    interior_w = random.randint(max(9, 2 * max_w + 1), 13)
    frame_h = interior_h + 2
    frame_w = interior_w + 2

    patches = []
    for color in row_colors:
        patch = np.zeros((variant_map[color].shape[0] + 1, variant_map[color].shape[1] + 1), dtype=int)
        patch[0, 0] = int(color)
        patch[1:, 1:] = variant_map[color]
        patches.append((int(color), patch))

    left_width = max(p.shape[1] for _, p in patches) + random.randint(1, 3)
    right_margin = random.randint(0, 1)
    total_w = left_width + 1 + frame_w + right_margin
    if total_w > 30:
        return None

    gaps = [2 for _ in range(max(0, len(patches) - 1))]
    stack_min_h = sum(p.shape[0] for _, p in patches) + sum(gaps)
    total_h = max(frame_h + random.randint(2, 5), stack_min_h + random.randint(0, 3))
    if total_h > 30:
        return None

    frame_top = random.randint(1, total_h - frame_h - 1)
    divider_c = left_width
    frame_left = divider_c + 1
    frame_right = frame_left + frame_w - 1
    frame_bottom = frame_top + frame_h - 1
    ir0, ic0 = frame_top + 1, frame_left + 1
    ir1, ic1 = frame_bottom - 1, frame_right - 1

    row_lo, row_hi = _safe_center_bounds(max_h, ir0, ir1)
    col_lo, col_hi = _safe_center_bounds(max_w, ic0, ic1)
    rows = _pick_rows(n_variants, row_lo, row_hi, max_h)
    if rows is None:
        return None
    n_cols = random.choice([2, 2, 3])
    cols = _pick_cols(n_cols, col_lo, col_hi, max_w)
    if cols is None:
        return None

    grid = np.zeros((total_h, total_w), dtype=int)
    grid[:, divider_c] = frame_color
    grid[frame_top, frame_left : frame_right + 1] = frame_color
    grid[frame_bottom, frame_left : frame_right + 1] = frame_color
    grid[frame_top : frame_bottom + 1, frame_left] = frame_color
    grid[frame_top : frame_bottom + 1, frame_right] = frame_color

    for r, color in zip(rows, row_colors):
        grid[r, frame_left] = int(color)
        grid[r, frame_right] = int(color)
    for c in cols:
        grid[frame_top, c] = int(col_color)
        grid[frame_bottom, c] = int(col_color)

    if not _place_left_variants(grid, patches, left_width):
        return None

    row_pairs = list(zip(rows, row_colors))
    expected = _metadata_expected(grid, divider_c, row_pairs, cols, variant_map)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None

    scene = _parse_scene(grid)
    if set(scene["variants"].keys()) != set(map(int, row_colors)):
        return None
    if len(scene["row_targets"]) != n_variants or len(scene["col_targets"]) != n_cols:
        return None

    fixed_variant = variant_map[row_colors[0]]
    wrong_same_variant = _apply_rule(scene, fixed_variant=fixed_variant)
    if np.array_equal(wrong_same_variant, actual):
        return None

    reverse_rows = list(reversed(scene["row_targets"]))
    wrong_reverse = _apply_rule(scene, row_order=reverse_rows)
    if np.array_equal(wrong_reverse, actual):
        return None

    wrong_no_erase = _apply_rule(scene, erase_left=False)
    if np.array_equal(wrong_no_erase, actual):
        return None

    values, counts = np.unique(grid, return_counts=True)
    if int(values[np.argmax(counts)]) != bg:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
