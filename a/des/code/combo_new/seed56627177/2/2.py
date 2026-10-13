from __future__ import annotations

from collections import deque

import numpy as np


BG = 8
FRAME = 5
PALETTE = (0, 1, 2, 3, 4, 6, 7, 9)
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            mins = arr.min(axis=0)
            maxs = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])),
                    "area": int(len(coords)),
                }
            )
    return comps


def _is_rect_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return False
    expected_area = 2 * (h + w) - 4
    if int(comp["area"]) != expected_area:
        return False
    cells = {tuple(map(int, rc)) for rc in comp["coords"]}
    for r, c in cells:
        if r not in (r0, r1) and c not in (c0, c1):
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in cells or (r, c1) not in cells:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in cells or (r1, c) not in cells:
            return False
    return True


def _extract_runs(values: np.ndarray, start_index: int) -> dict[int, tuple[int, int]]:
    runs: dict[int, tuple[int, int]] = {}
    i = 0
    while i < len(values):
        color = int(values[i])
        j = i + 1
        while j < len(values) and int(values[j]) == color:
            j += 1
        if color in runs:
            raise AssertionError("band color must appear in one contiguous segment")
        runs[color] = (start_index + i, start_index + j)
        i = j
    return runs


def _is_band_component(
    comp: dict,
    top_row: int,
    top_c0: int,
    top_c1: int,
    left_col: int,
    left_r0: int,
    left_r1: int,
) -> bool:
    coords = comp["coords"]
    if bool(np.all(coords[:, 0] == top_row)) and int(coords[:, 1].min()) >= top_c0 and int(coords[:, 1].max()) <= top_c1:
        return True
    if bool(np.all(coords[:, 1] == left_col)) and int(coords[:, 0].min()) >= left_r0 and int(coords[:, 0].max()) <= left_r1:
        return True
    return False


def _random_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(rng.choice(np.arange(1, total), size=parts - 1, replace=False).tolist())
    points = [0] + cuts + [total]
    return [int(points[i + 1] - points[i]) for i in range(parts)]


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _grown_mask(h: int, w: int, target: int, rng: np.random.Generator) -> np.ndarray:
    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {start}
    while len(cells) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(frontier[int(rng.integers(0, len(frontier)))])
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _normalize_mask(mask)


def _fragment_mask(max_h: int, max_w: int, rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(1, max_h + 1))
    w = int(rng.integers(1, max_w + 1))
    modes = ["rect", "grow", "ell", "ring", "plus"]
    mode = str(rng.choice(modes))

    if mode == "rect" or h == 1 or w == 1:
        mask = np.ones((h, w), dtype=bool)
    elif mode == "ell":
        arm_h = int(rng.integers(1, h + 1))
        arm_w = int(rng.integers(1, w + 1))
        mask = np.zeros((h, w), dtype=bool)
        mask[:, 0] = True
        mask[0, :arm_w] = True
        mask[:arm_h, :] |= (np.arange(w)[None, :] == 0)
        if rng.random() < 0.5:
            mask[-1, :arm_w] = True
        mask = _normalize_mask(mask)
    elif mode == "ring" and h >= 3 and w >= 3:
        mask = np.ones((h, w), dtype=bool)
        mask[1:-1, 1:-1] = False
    elif mode == "plus" and h >= 3 and w >= 3:
        mask = np.zeros((h, w), dtype=bool)
        mask[:, w // 2] = True
        mask[h // 2, :] = True
        if rng.random() < 0.4:
            mask[:, 0] = True
        mask = _normalize_mask(mask)
    else:
        area_lo = max(1, (h * w + 2) // 3)
        area = int(rng.integers(area_lo, h * w + 1))
        mask = _grown_mask(h, w, area, rng)

    if rng.random() < 0.5:
        mask = np.flipud(mask)
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    return _normalize_mask(mask)


def _place_patch(
    grid: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    color: int,
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph > h or pw > w:
        return False
    for _ in range(600):
        r0 = int(rng.integers(0, h - ph + 1))
        c0 = int(rng.integers(0, w - pw + 1))
        rr0 = max(0, r0 - 1)
        rr1 = min(h, r0 + ph + 1)
        cc0 = max(0, c0 - 1)
        cc1 = min(w, c0 + pw + 1)
        if bool(blocked[rr0:rr1, cc0:cc1].any()):
            continue
        grid[r0 : r0 + ph, c0 : c0 + pw][patch] = color
        blocked[rr0:rr1, cc0:cc1] = True
        return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """Move each scattered monochrome fragment into the frame slot indexed by its band color.

    Input structure:
    - The background is azure.
    - One large grey 1-cell-thick hollow rectangle forms the unique frame, and its interior is empty.
    - A colored band runs immediately above the frame interior and another runs immediately left of the frame interior; each fragment color appears exactly once in each band.
    - The remaining non-background components are monochrome connected fragments scattered outside the frame.

    Rule:
    - Identify the unique grey rectangular frame and read contiguous same-color segments from the top band and the left band.
    - For each fragment, use its color to find the matching top-band column span and left-band row span.
    - Their intersection defines the fragment's destination rectangle inside the frame; translate the fragment so its bbox top-left aligns with that rectangle's top-left.
    - Erase the fragment from its old location, paste it into the frame, and keep the frame, bands, and background unchanged.

    Key insight:
    - Fragment colors are lookup keys: the top band provides the destination columns, the left band provides the destination rows, and their Cartesian intersection gives a unique placement slot.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _bg_color(g)

    components = _color_components(g, bg)
    frame_candidates = [comp for comp in components if _is_rect_frame(comp)]
    assert frame_candidates
    frame = max(frame_candidates, key=lambda comp: ((comp["bbox"][2] - comp["bbox"][0] + 1) * (comp["bbox"][3] - comp["bbox"][1] + 1), comp["area"]))
    fr0, fc0, fr1, fc1 = frame["bbox"]
    assert fr0 >= 1 and fc0 >= 1

    top_row = fr0 - 1
    left_col = fc0 - 1
    top_vals = g[top_row, fc0 + 1 : fc1]
    left_vals = g[fr0 + 1 : fr1, left_col]
    assert len(top_vals) == fc1 - fc0 - 1
    assert len(left_vals) == fr1 - fr0 - 1
    assert bool(np.all(top_vals != bg))
    assert bool(np.all(left_vals != bg))

    top_runs = _extract_runs(top_vals, fc0 + 1)
    left_runs = _extract_runs(left_vals, fr0 + 1)
    assert set(top_runs) == set(left_runs)

    fragments = [
        comp
        for comp in components
        if comp is not frame
        and not _is_band_component(comp, top_row, fc0 + 1, fc1 - 1, left_col, fr0 + 1, fr1 - 1)
    ]
    assert len(fragments) == len(top_runs)

    out = g.copy()
    planned: list[tuple[dict, int, int]] = []
    for frag in fragments:
        color = int(frag["color"])
        assert color in top_runs and color in left_runs
        rr0, rr1 = left_runs[color]
        cc0, cc1 = top_runs[color]
        r0, c0, r1, c1 = frag["bbox"]
        frag_h = r1 - r0 + 1
        frag_w = c1 - c0 + 1
        assert frag_h <= rr1 - rr0
        assert frag_w <= cc1 - cc0
        planned.append((frag, rr0, cc0))
        out[frag["coords"][:, 0], frag["coords"][:, 1]] = bg

    for frag, target_r0, target_c0 in planned:
        r0, c0, r1, c1 = frag["bbox"]
        dr = target_r0 - r0
        dc = target_c0 - c0
        shifted = frag["coords"] + np.array([dr, dc], dtype=int)
        assert int(shifted[:, 0].min()) >= fr0 + 1 and int(shifted[:, 0].max()) <= fr1 - 1
        assert int(shifted[:, 1].min()) >= fc0 + 1 and int(shifted[:, 1].max()) <= fc1 - 1
        assert bool(np.all(out[shifted[:, 0], shifted[:, 1]] == bg))
        out[shifted[:, 0], shifted[:, 1]] = int(frag["color"])

    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_fragments = int(rng.integers(2, 7))
    inner_h = int(rng.integers(max(4, n_fragments), 13))
    inner_w = int(rng.integers(max(4, n_fragments), 13))
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    max_grid_h = 30
    max_grid_w = 30
    extra_h = int(rng.integers(6, max_grid_h - frame_h + 1))
    extra_w = int(rng.integers(6, max_grid_w - frame_w + 1))
    grid_h = frame_h + extra_h
    grid_w = frame_w + extra_w
    if grid_h > 30 or grid_w > 30:
        return None

    frame_r0 = int(rng.integers(1, grid_h - frame_h))
    frame_c0 = int(rng.integers(1, grid_w - frame_w))
    frame_r1 = frame_r0 + frame_h - 1
    frame_c1 = frame_c0 + frame_w - 1

    colors = list(rng.choice(np.array(PALETTE, dtype=int), size=n_fragments, replace=False))
    top_order = list(rng.permutation(colors))
    left_order = list(rng.permutation(colors))
    top_sizes = _random_composition(inner_w, n_fragments, rng)
    left_sizes = _random_composition(inner_h, n_fragments, rng)

    top_ranges: dict[int, tuple[int, int]] = {}
    cursor = frame_c0 + 1
    for color, size in zip(top_order, top_sizes):
        top_ranges[int(color)] = (cursor, cursor + int(size))
        cursor += int(size)

    left_ranges: dict[int, tuple[int, int]] = {}
    cursor = frame_r0 + 1
    for color, size in zip(left_order, left_sizes):
        left_ranges[int(color)] = (cursor, cursor + int(size))
        cursor += int(size)

    fragments = []
    for color in colors:
        row_span = left_ranges[int(color)][1] - left_ranges[int(color)][0]
        col_span = top_ranges[int(color)][1] - top_ranges[int(color)][0]
        patch = _fragment_mask(row_span, col_span, rng)
        if patch.shape[0] > row_span or patch.shape[1] > col_span:
            return None
        fragments.append((int(color), patch))

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[frame_r0, frame_c0 : frame_c1 + 1] = FRAME
    grid[frame_r1, frame_c0 : frame_c1 + 1] = FRAME
    grid[frame_r0 : frame_r1 + 1, frame_c0] = FRAME
    grid[frame_r0 : frame_r1 + 1, frame_c1] = FRAME

    for color, (c0, c1) in top_ranges.items():
        grid[frame_r0 - 1, c0:c1] = int(color)
    for color, (r0, r1) in left_ranges.items():
        grid[r0:r1, frame_c0 - 1] = int(color)

    blocked = np.zeros_like(grid, dtype=bool)
    blocked[frame_r0 : frame_r1 + 1, frame_c0 : frame_c1 + 1] = True
    blocked[frame_r0 - 1, frame_c0 + 1 : frame_c1] = True
    blocked[frame_r0 + 1 : frame_r1, frame_c0 - 1] = True

    for color, patch in sorted(fragments, key=lambda item: int(item[1].sum()), reverse=True):
        if not _place_patch(grid, blocked, patch, int(color), rng):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
