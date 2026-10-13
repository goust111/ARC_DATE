import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
MARKER_COLORS = (3, 6, 7, 8, 9)

MAX_GENERATION_ATTEMPTS = 500

GRID_HEIGHT_RANGE = (16, 23)
GRID_WIDTH_RANGE = (16, 23)

TOP_HEIGHT_RANGE = (4, 8)
TOP_WIDTH_RANGE = (5, 10)
TILE_HEIGHT_RANGE = (2, 5)
TILE_WIDTH_RANGE = (2, 5)
TARGET_COUNT_RANGE = (5, 15)

FRAME_COUNT_CHOICES = (2, 3)
MARKER_COLOR_COUNT_CHOICES = (2, 3)
FRAME_INTERIOR_SIDE_RANGE = (1, 8)


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    coord_set = set(coords)
    expected_area = 2 * h + 2 * w - 4
    if len(coord_set) != expected_area:
        return None
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in coord_set):
                return None
    return r0, c0, r1, c1


def _checkerboard_capacity(frame_bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = frame_bbox
    ih = r1 - r0 - 1
    iw = c1 - c0 - 1
    return (ih * iw + 1) // 2


def _minimal_period(block: np.ndarray) -> tuple[int, int]:
    h, w = block.shape
    best = (h, w)
    best_area = h * w
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            area = ph * pw
            if area > best_area:
                continue
            ok = True
            for r in range(h):
                for c in range(w):
                    if int(block[r, c]) != int(block[r % ph, c % pw]):
                        ok = False
                        break
                if not ok:
                    break
            if ok and (area < best_area or (area == best_area and (ph, pw) < best)):
                best = (ph, pw)
                best_area = area
    return best


def _tile_full(tile: np.ndarray, h: int, w: int) -> np.ndarray:
    rr = np.arange(h)[:, None] % tile.shape[0]
    cc = np.arange(w)[None, :] % tile.shape[1]
    return tile[rr, cc]


def _complete_top_patch(top_patch: np.ndarray) -> np.ndarray:
    full_grey_rows = np.where(np.all(top_patch == GREY, axis=1))[0]
    full_grey_cols = np.where(np.all(top_patch == GREY, axis=0))[0]
    assert len(full_grey_rows) >= 1 and len(full_grey_cols) >= 1, "expected bottom/right grey suffixes"

    r_cut = int(full_grey_rows[0])
    c_cut = int(full_grey_cols[0])
    visible = top_patch[:r_cut, :c_cut]
    assert visible.size > 0 and not np.any(visible == GREY), "visible block must be non-grey"

    ph, pw = _minimal_period(visible)
    tile = visible[:ph, :pw]
    return _tile_full(tile, *top_patch.shape)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the hidden periodic tile, count its filled area, then use that count to choose one frame and one marker color.

    Input structure:
    - A black background contains one top-left rectangle made from blue/red periodic cells plus a grey L-shaped placeholder suffix.
    - Lower in the grid are 2-3 yellow one-cell-thick hollow rectangular frames whose interiors are black.
    - Outside those frames are scattered single-pixel markers in 2-3 other colors, with each color appearing a distinct number of times.

    Rule:
    - Infer the minimal periodic tile from the visible top-left block, extend it through the whole grey-masked rectangle, and replace the grey cells with the continued pattern.
    - Count the completed top rectangle's non-black cells to get N.
    - For each yellow frame, count the interior checkerboard cells starting from the interior upper-left cell; pick the unique frame whose count equals N.
    - Find the unique scattered marker color that appears exactly N times, erase all scattered markers, and fill the chosen frame's checkerboard cells with that color while leaving all other frames unchanged.

    Key insight:
    - The top pattern does not directly tell you where to paint; it only produces a numeric key, and the same key must independently identify both a frame capacity and a marker-color population.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    top_bbox = _bbox_from_mask(np.isin(g, [BLUE, RED, GREY]))
    assert top_bbox is not None, "expected one periodic top patch"
    tr0, tc0, tr1, tc1 = top_bbox
    top_patch = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
    completed_top = _complete_top_patch(top_patch)
    n = int(np.count_nonzero(completed_top))

    frames: list[tuple[int, tuple[int, int, int, int]]] = []
    for comp in _components_of_color(g, YELLOW):
        box = _frame_bbox(comp)
        assert box is not None, "all yellow components must be hollow rectangular frames"
        frames.append((_checkerboard_capacity(box), box))
    assert 2 <= len(frames) <= 3, "expected 2-3 frames"

    matching_frames = [box for capacity, box in frames if capacity == n]
    assert len(matching_frames) == 1, "expected a unique frame with checkerboard count N"
    target_frame = matching_frames[0]

    marker_colors = sorted(set(int(v) for v in np.unique(g)) - {BG, BLUE, RED, YELLOW, GREY})
    counts = {color: int(np.sum(g == color)) for color in marker_colors}
    matching_colors = [color for color, count in counts.items() if count == n]
    assert len(matching_colors) == 1, "expected a unique marker color with population N"
    fill_color = matching_colors[0]

    out = g.copy()
    out[tr0 : tr1 + 1, tc0 : tc1 + 1] = completed_top
    for color in marker_colors:
        out[out == color] = BG

    r0, c0, r1, c1 = target_frame
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if (r - (r0 + 1) + c - (c0 + 1)) % 2 == 0:
                out[r, c] = fill_color
            else:
                out[r, c] = BG
    return out


def _capacity_to_sizes() -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {}
    for ih in range(FRAME_INTERIOR_SIDE_RANGE[0], FRAME_INTERIOR_SIDE_RANGE[1] + 1):
        for iw in range(FRAME_INTERIOR_SIDE_RANGE[0], FRAME_INTERIOR_SIDE_RANGE[1] + 1):
            cap = (ih * iw + 1) // 2
            out.setdefault(cap, []).append((ih, iw))
    return out


CAPACITY_TO_SIZES = _capacity_to_sizes()


def _design_top_patch() -> tuple[np.ndarray, np.ndarray, int]:
    for _ in range(800):
        top_h = random.randint(*TOP_HEIGHT_RANGE)
        top_w = random.randint(*TOP_WIDTH_RANGE)
        tile_h = random.randint(TILE_HEIGHT_RANGE[0], min(TILE_HEIGHT_RANGE[1] - 1, top_h - 1))
        tile_w = random.randint(TILE_WIDTH_RANGE[0], min(TILE_WIDTH_RANGE[1] - 1, top_w - 1))

        tile = np.random.choice([BG, BLUE, RED], size=(tile_h, tile_w), p=[0.45, 0.275, 0.275]).astype(int)
        if not np.any(tile == BLUE) or not np.any(tile == RED) or not np.any(tile == BG):
            continue

        full = _tile_full(tile, top_h, top_w)
        n = int(np.count_nonzero(full))
        if n not in CAPACITY_TO_SIZES or not (TARGET_COUNT_RANGE[0] <= n <= TARGET_COUNT_RANGE[1]):
            continue
        if _minimal_period(full) != (tile_h, tile_w):
            continue
        if not np.any(full[0] != BG) or not np.any(full[:, 0] != BG):
            continue

        r_cut_lo = tile_h + 1
        c_cut_lo = tile_w + 1
        if r_cut_lo >= top_h or c_cut_lo >= top_w:
            continue
        r_cut = random.randint(r_cut_lo, top_h - 1)
        c_cut = random.randint(c_cut_lo, top_w - 1)
        visible = full[:r_cut, :c_cut]
        if _minimal_period(visible) != (tile_h, tile_w):
            continue
        if not np.any(visible == BLUE) or not np.any(visible == RED):
            continue

        inp = full.copy()
        inp[r_cut:, :] = GREY
        inp[:, c_cut:] = GREY
        return inp, full, n
    raise RuntimeError("failed to design top patch")


def _frame_patch(ih: int, iw: int) -> np.ndarray:
    patch = np.zeros((ih + 2, iw + 2), dtype=int)
    patch[0, :] = YELLOW
    patch[-1, :] = YELLOW
    patch[:, 0] = YELLOW
    patch[:, -1] = YELLOW
    return patch


def _choose_frames(target_n: int, frame_count: int) -> list[dict[str, int | np.ndarray | bool]]:
    capacities = [target_n]
    other_caps = [cap for cap in CAPACITY_TO_SIZES if cap != target_n and 2 <= cap <= 18]
    random.shuffle(other_caps)
    capacities.extend(other_caps[: frame_count - 1])
    random.shuffle(capacities)

    frames: list[dict[str, int | np.ndarray | bool]] = []
    target_assigned = False
    for cap in capacities:
        ih, iw = random.choice(CAPACITY_TO_SIZES[cap])
        is_target = (cap == target_n) and not target_assigned
        target_assigned = target_assigned or is_target
        frames.append(
            {
                "ih": ih,
                "iw": iw,
                "cap": cap,
                "is_target": is_target,
                "patch": _frame_patch(ih, iw),
            }
        )
    return frames


def _mark_forbidden(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0], r0 + h + pad)
    cc1 = min(mask.shape[1], c0 + w + pad)
    mask[rr0:rr1, cc0:cc1] = True


def _place_frames(
    grid: np.ndarray,
    frames: list[dict[str, int | np.ndarray | bool]],
    lower_start: int,
    top_bbox: tuple[int, int, int, int],
) -> bool:
    h, w = grid.shape
    forbidden = np.zeros((h, w), dtype=bool)
    tr0, tc0, tr1, tc1 = top_bbox
    _mark_forbidden(forbidden, tr0, tc0, tr1 - tr0 + 1, tc1 - tc0 + 1, pad=1)

    ordered = sorted(frames, key=lambda frame: int(frame["patch"].size), reverse=True)
    for frame in ordered:
        patch = frame["patch"]
        oh, ow = patch.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(lower_start, h - oh):
            for c0 in range(1, w - ow):
                if forbidden[r0 : r0 + oh, c0 : c0 + ow].any():
                    continue
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(h, r0 + oh + 1)
                cc1 = min(w, c0 + ow + 1)
                if forbidden[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        grid[r0 : r0 + oh, c0 : c0 + ow] = patch
        frame["r0"] = r0
        frame["c0"] = c0
        _mark_forbidden(forbidden, r0, c0, oh, ow, pad=1)
    return True


def _choose_marker_specs(target_n: int, frame_count: int) -> list[tuple[int, int]]:
    color_count = random.choice(MARKER_COLOR_COUNT_CHOICES)
    colors = random.sample(MARKER_COLORS, color_count)
    target_index = random.randrange(color_count)
    forbidden_counts = {target_n}
    specs: list[tuple[int, int]] = []
    for i, color in enumerate(colors):
        if i == target_index:
            specs.append((color, target_n))
            continue
        choices = [count for count in range(1, 13) if count not in forbidden_counts]
        count = random.choice(choices)
        forbidden_counts.add(count)
        specs.append((color, count))
    random.shuffle(specs)
    return specs


def _scatter_markers(
    grid: np.ndarray,
    marker_specs: list[tuple[int, int]],
    lower_start: int,
    top_bbox: tuple[int, int, int, int],
    frames: list[dict[str, int | np.ndarray | bool]],
) -> bool:
    h, w = grid.shape
    allowed = np.zeros((h, w), dtype=bool)
    allowed[lower_start:, :] = True

    tr0, tc0, tr1, tc1 = top_bbox
    allowed[tr0 : tr1 + 1, tc0 : tc1 + 1] = False
    for frame in frames:
        patch = frame["patch"]
        r0 = int(frame["r0"])
        c0 = int(frame["c0"])
        oh, ow = patch.shape
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h, r0 + oh + 1)
        cc1 = min(w, c0 + ow + 1)
        allowed[rr0:rr1, cc0:cc1] = False

    blocked = np.zeros((h, w), dtype=bool)
    jobs: list[int] = []
    for color, count in marker_specs:
        jobs.extend([color] * count)
    random.shuffle(jobs)

    placements: list[tuple[int, int]] = []
    for color in jobs:
        candidates = [(r, c) for r, c in zip(*np.where(allowed & ~blocked))]
        if not candidates:
            return False
        r, c = random.choice(candidates)
        grid[r, c] = color
        placements.append((r, c))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True
        blocked[r, c] = True
        allowed[r, c] = False
    return True


def _build_expected_output(inp: np.ndarray) -> np.ndarray:
    return main(inp)


def _gen_attempt() -> np.ndarray | None:
    top_input, top_full, target_n = _design_top_patch()
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    frames = _choose_frames(target_n, frame_count)
    marker_specs = _choose_marker_specs(target_n, frame_count)

    grid_h = random.randint(*GRID_HEIGHT_RANGE)
    grid_w = random.randint(*GRID_WIDTH_RANGE)
    if top_input.shape[0] + 6 >= grid_h or max(top_input.shape[1], max(int(frame["patch"].shape[1]) for frame in frames)) + 2 >= grid_w:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    top_r0 = random.choice([0, 1])
    top_c0 = random.randint(0, min(2, grid_w - top_input.shape[1]))
    top_h, top_w = top_input.shape
    grid[top_r0 : top_r0 + top_h, top_c0 : top_c0 + top_w] = top_input
    top_bbox = (top_r0, top_c0, top_r0 + top_h - 1, top_c0 + top_w - 1)

    lower_start = top_r0 + top_h + random.randint(1, 3)
    if lower_start >= grid_h - 4:
        return None
    if not _place_frames(grid, frames, lower_start, top_bbox):
        return None
    if not _scatter_markers(grid, marker_specs, lower_start, top_bbox, frames):
        return None

    try:
        out = _build_expected_output(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.array_equal(out[top_r0 : top_r0 + top_h, top_c0 : top_c0 + top_w], top_full):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
