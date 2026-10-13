import math
import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
STAMP_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_RETRIES = 400
SHAPE_RETRIES = 300
PLACEMENT_RETRIES = 80


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected True-cell components as integer coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.asarray(cells, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = np.min(coords, axis=0)
    r1, c1 = np.max(coords, axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    cropped = _crop_bool(mask)
    pts = np.argwhere(cropped)
    return tuple(sorted((int(r), int(c)) for r, c in pts))


def _dihedral_signatures(mask: np.ndarray) -> set[tuple[tuple[int, int], ...]]:
    base = _crop_bool(mask)
    sigs: set[tuple[tuple[int, int], ...]] = set()
    for k in range(4):
        rot = np.rot90(base, k)
        sigs.add(_shape_signature(rot))
        sigs.add(_shape_signature(np.fliplr(rot)))
    return sigs


def _has_internal_hole(mask: np.ndarray) -> bool:
    empty = ~_crop_bool(mask)
    if not np.any(empty):
        return False
    h, w = empty.shape
    for comp in _components(empty):
        if not np.any(comp[:, 0] == 0) and not np.any(comp[:, 0] == h - 1) and not np.any(comp[:, 1] == 0) and not np.any(comp[:, 1] == w - 1):
            return True
    return False


def _apply_transform(mask: np.ndarray, variant: int) -> np.ndarray:
    out = np.rot90(mask, variant % 4)
    if variant >= 4:
        out = np.fliplr(out)
    return out.copy()


def _window_candidates(g: np.ndarray, bg: int) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        color_windows: list[dict] = []
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            hole_mask = sub == bg
            if not np.any(hole_mask):
                continue
            if np.any(hole_mask[0, :]) or np.any(hole_mask[-1, :]) or np.any(hole_mask[:, 0]) or np.any(hole_mask[:, -1]):
                continue
            if not np.all((sub == bg) | (sub == color)):
                continue
            hole_parts = _components(hole_mask)
            if len(hole_parts) != 1:
                continue
            if _has_internal_hole(hole_mask):
                continue
            color_windows.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "hole_sig": _shape_signature(hole_mask),
                    "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                }
            )
        if color_windows:
            out[color] = color_windows
    return out


def _match_stamps_to_windows(stamps: list[dict], windows: list[dict]) -> dict[int, int]:
    options = {
        s_idx: [w_idx for w_idx, window in enumerate(windows) if window["hole_sig"] in stamps[s_idx]["variants"]]
        for s_idx in range(len(stamps))
    }
    assert all(options.values()), "every stamp must match at least one window hole"

    order = sorted(range(len(stamps)), key=lambda idx: len(options[idx]))
    used: set[int] = set()
    assignment: dict[int, int] = {}

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        s_idx = order[pos]
        for w_idx in options[s_idx]:
            if w_idx in used:
                continue
            used.add(w_idx)
            assignment[s_idx] = w_idx
            if dfs(pos + 1):
                return True
            used.remove(w_idx)
            del assignment[s_idx]
        return False

    assert dfs(0), "stamp/window matching must admit a bijection"
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match scattered stamps to window holes, then recolor each matched window into a solid rectangle.

    Input structure:
    - The background is black.
    - Several template windows are monochrome rectangles whose border color surrounds one irregular black hole.
    - Several scattered monochrome stamp components sit outside the windows; each stamp shape matches one hole up to rotation and reflection.

    Rule:
    - Detect the window color by finding components whose bbox is a rectangle with exactly one enclosed black hole.
    - For every external stamp, compare its shape against each hole under the 8 dihedral transforms (4 rotations, with or without a horizontal flip).
    - Each matched window is recolored with the stamp color on both the border and the hole, so its whole bounding rectangle becomes a solid block.
    - Stamp pixels disappear from their original locations; the rest of the background stays black.

    Key insight:
    - The essential signal is shape equivalence under rotation and reflection, and the matched stamp color propagates to the entire window rather than only to the hole.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG

    candidates = _window_candidates(g, bg)
    assert candidates, "expected at least one window candidate"
    window_color = max(
        candidates,
        key=lambda color: (len(candidates[color]), sum(item["area"] for item in candidates[color])),
    )
    windows = candidates[window_color]

    stamps: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (bg, window_color)):
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[comp[:, 0] - r0, comp[:, 1] - c0] = True
            stamps.append({"color": color, "variants": _dihedral_signatures(mask)})

    assert len(stamps) == len(windows), "stamp count must equal window count"

    assignment = _match_stamps_to_windows(stamps, windows)
    out = np.full_like(g, bg)
    for s_idx, w_idx in assignment.items():
        color = stamps[s_idx]["color"]
        r0, c0, r1, c1 = windows[w_idx]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _random_connected_mask(height: int, width: int, target_area: int, rng: random.Random) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    start = (rng.randrange(height), rng.randrange(width))
    mask[start] = True
    frontier = set()
    for dr, dc in DIRS4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((nr, nc))

    while int(mask.sum()) < target_area:
        if not frontier:
            return None
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        for dr, dc in DIRS4:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))
    return _crop_bool(mask)


def _make_hole_shape(
    rng: random.Random,
    used_canonicals: set[tuple[tuple[int, int], ...]],
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]] | None:
    for _ in range(SHAPE_RETRIES):
        box_h = rng.randint(2, 5)
        box_w = rng.randint(2, 5)
        max_area = box_h * box_w - 1
        if max_area < 3:
            continue
        min_area = max(3, math.ceil(box_h * box_w * 0.35))
        area = rng.randint(min_area, max_area)
        mask = _random_connected_mask(box_h, box_w, area, rng)
        if mask is None:
            continue
        mask = _crop_bool(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if np.all(mask):
            continue
        if _has_internal_hole(mask):
            continue
        sigs = _dihedral_signatures(mask)
        if len(sigs) < 2:
            continue
        canonical = min(sigs)
        if canonical in used_canonicals:
            continue
        return mask, canonical
    return None


def _build_window(hole_mask: np.ndarray, rng: random.Random) -> np.ndarray:
    hole_h, hole_w = hole_mask.shape
    for _ in range(50):
        top = rng.randint(1, 3)
        bottom = rng.randint(1, 3)
        left = rng.randint(1, 3)
        right = rng.randint(1, 3)
        height = hole_h + top + bottom
        width = hole_w + left + right
        if height <= 10 and width <= 10:
            window = np.full((height, width), GREY, dtype=int)
            sub = window[top : top + hole_h, left : left + hole_w]
            sub[hole_mask] = BG
            return window
    raise RuntimeError("failed to size a window")


def _rect_clear(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not np.any(occupied[r0:r1, c0:c1])


def _mark_rect(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    occupied[r0:r1, c0:c1] = True


def _place_arrays(arrays: list[np.ndarray], rng: random.Random) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    heights = [int(arr.shape[0]) for arr in arrays]
    widths = [int(arr.shape[1]) for arr in arrays]
    padded_area = sum((h + 2) * (w + 2) for h, w in zip(heights, widths))
    base = max(max(heights) + 2, max(widths) + 2, int(math.ceil(math.sqrt(padded_area))))

    for _ in range(PLACEMENT_RETRIES):
        grid_h = rng.randint(max(18, base), 30)
        grid_w = rng.randint(max(18, base), 30)
        occupied = np.zeros((grid_h, grid_w), dtype=bool)
        positions: list[tuple[int, int] | None] = [None] * len(arrays)
        order = sorted(range(len(arrays)), key=lambda idx: arrays[idx].shape[0] * arrays[idx].shape[1], reverse=True)
        ok = True

        for idx in order:
            height, width = arrays[idx].shape
            options: list[tuple[int, int]] = []
            for top in range(grid_h - height + 1):
                for left in range(grid_w - width + 1):
                    if _rect_clear(occupied, top, left, height, width, pad=1):
                        options.append((top, left))
            if not options:
                ok = False
                break
            top, left = rng.choice(options)
            positions[idx] = (top, left)
            _mark_rect(occupied, top, left, height, width, pad=1)

        if ok:
            return (grid_h, grid_w), [(int(r), int(c)) for r, c in positions]
    return None


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    count = rng.randint(2, 4)
    colors = rng.sample(STAMP_COLORS, count)
    used_canonicals: set[tuple[tuple[int, int], ...]] = set()

    window_arrays: list[np.ndarray] = []
    stamp_arrays: list[np.ndarray] = []
    expected_colors: list[int] = []

    for color in colors:
        shape_data = _make_hole_shape(rng, used_canonicals)
        if shape_data is None:
            return None
        hole_mask, canonical = shape_data
        used_canonicals.add(canonical)

        window_arrays.append(_build_window(hole_mask, rng))

        variant = rng.randrange(8)
        stamp_mask = _apply_transform(hole_mask, variant)
        stamp = np.zeros(stamp_mask.shape, dtype=int)
        stamp[stamp_mask] = color
        stamp_arrays.append(stamp)
        expected_colors.append(color)

    arrays = window_arrays + stamp_arrays
    placement = _place_arrays(arrays, rng)
    if placement is None:
        return None

    (grid_h, grid_w), positions = placement
    inp = np.full((grid_h, grid_w), BG, dtype=int)
    out = np.full((grid_h, grid_w), BG, dtype=int)

    for idx, window in enumerate(window_arrays):
        top, left = positions[idx]
        height, width = window.shape
        region = inp[top : top + height, left : left + width]
        mask = window != BG
        region[mask] = window[mask]
        out[top : top + height, left : left + width] = expected_colors[idx]

    for idx, stamp in enumerate(stamp_arrays, start=count):
        top, left = positions[idx]
        height, width = stamp.shape
        region = inp[top : top + height, left : left + width]
        mask = stamp != BG
        region[mask] = stamp[mask]

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        case = _generate_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
