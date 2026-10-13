from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
REF_PALETTE = (1, 4, 5, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _is_rect_frame(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h not in (3, 4) or w not in (3, 4) or h != w:
        return False
    expected = np.zeros_like(mask, dtype=bool)
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    return np.array_equal(mask, expected)


def _find_reference_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    mask = ~np.isin(g, (BG, RED, GREEN))
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a multicolor reference pattern"
    return _bbox(coords)


def _find_windows(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    windows = []
    for comp in _components(g == GREEN):
        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros_like(g, dtype=bool)
        mask[comp[:, 0], comp[:, 1]] = True
        if _is_rect_frame(mask):
            windows.append((r0, c0, r1, c1))
    assert windows, "expected at least one green window frame"
    windows.sort()
    return windows


def _marker_for_window(g: np.ndarray, window: tuple[int, int, int, int], remaining: set[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = window
    best = None
    best_key = None
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    for mr, mc in remaining:
        if c0 <= mc <= c1 and (mr < r0 or mr > r1):
            vertical_gap = min(abs(mr - r0), abs(mr - r1))
            key = (vertical_gap, abs(mc - center_c), abs(mr - center_r))
            if best_key is None or key < best_key:
                best_key = key
                best = (mr, mc)
    assert best is not None, "each window must have a nearby red marker in the same column band"
    return best


def _crop_bounds(anchor_r: int, anchor_c: int, inner_h: int, inner_w: int) -> tuple[int, int]:
    top = anchor_r - (inner_h - 1) // 2
    left = anchor_c - (inner_w - 1) // 2
    return top, left


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each green window with a crop from the upper reference pattern.

    - The reference pattern is the unique non-black/non-red/non-green area.
    - Each green 3x3 or 4x4 rectangular frame has one red marker aligned to one
      of the frame's columns.
    - The marker column gives the reference anchor column directly.
    - The marker row offset from the frame center gives the anchor row relative
      to the reference center.
    - The window interior receives the crop and the red markers are erased.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    ref_r0, ref_c0, ref_r1, ref_c1 = _find_reference_bbox(g)
    ref_center_r = (ref_r0 + ref_r1) // 2
    windows = _find_windows(g)

    red_pixels = {tuple(map(int, p)) for p in np.argwhere(g == RED)}
    assert len(red_pixels) >= len(windows), "expected at least one red marker per window"

    for window in windows:
        r0, c0, r1, c1 = window
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        center_r = (r0 + r1) // 2

        mr, mc = _marker_for_window(g, window, red_pixels)
        red_pixels.remove((mr, mc))

        anchor_r = ref_center_r + (mr - center_r)
        anchor_c = mc
        top, left = _crop_bounds(anchor_r, anchor_c, inner_h, inner_w)

        assert ref_r0 <= top and top + inner_h - 1 <= ref_r1
        assert ref_c0 <= left and left + inner_w - 1 <= ref_c1

        patch = g[top : top + inner_h, left : left + inner_w]
        out[r0 + 1 : r1, c0 + 1 : c1] = patch
        out[mr, mc] = BG

    return out


def _make_reference(rng: random.Random, h: int, w: int) -> np.ndarray:
    colors = rng.sample(REF_PALETTE, rng.randint(3, min(5, len(REF_PALETTE))))
    ref = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            ref[r, c] = colors[(r + 2 * c + rng.randrange(len(colors))) % len(colors)]
    # Force visible local structure so crops are readable and varied.
    for _ in range(rng.randint(3, 6)):
        rr = rng.randrange(h)
        cc = rng.randrange(w)
        rh = rng.randint(1, min(3, h - rr))
        cw = rng.randint(1, min(3, w - cc))
        color = rng.choice(colors)
        ref[rr : rr + rh, cc : cc + cw] = color
    return ref


def _draw_frame(grid: np.ndarray, r0: int, c0: int, size: int) -> None:
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    grid[r0 : r1 + 1, c0] = GREEN
    grid[r0 : r1 + 1, c1] = GREEN
    grid[r0, c0 : c1 + 1] = GREEN
    grid[r1, c0 : c1 + 1] = GREEN


def _sample_window_positions(
    rng: random.Random,
    H: int,
    W: int,
    ref_bbox: tuple[int, int, int, int],
    plans: list[dict],
) -> list[dict] | None:
    ref_r0, ref_c0, ref_r1, ref_c1 = ref_bbox
    ref_center_r = (ref_r0 + ref_r1) // 2
    occupied = np.zeros((H, W), dtype=bool)
    occupied[ref_r0 : ref_r1 + 1, ref_c0 : ref_c1 + 1] = True

    order = list(range(len(plans)))
    rng.shuffle(order)
    placed: list[dict | None] = [None] * len(plans)

    for idx in order:
        plan = plans[idx]
        size = plan["size"]
        anchor_r = plan["anchor_r"]
        anchor_c = plan["anchor_c"]
        delta = anchor_r - ref_center_r

        candidates = []
        for slot in range(size):
            c0 = anchor_c - slot
            if c0 < 1 or c0 + size >= W - 1:
                continue
            for r0 in range(ref_r1 + 3, H - size - 1):
                center_r = (r0 + r0 + size - 1) // 2
                mr = center_r + delta
                if not (0 <= mr < H):
                    continue
                if r0 <= mr <= r0 + size - 1:
                    continue
                if abs(mr - center_r) > 6:
                    continue
                marker_gap = min(abs(mr - r0), abs(mr - (r0 + size - 1)))
                if marker_gap < 1 or marker_gap > 4:
                    continue
                rr0 = min(r0, mr)
                rr1 = max(r0 + size - 1, mr)
                cc0 = c0
                cc1 = c0 + size - 1
                if occupied[rr0 : rr1 + 1, max(0, cc0 - 1) : min(W, cc1 + 2)].any():
                    continue
                candidates.append((r0, c0, mr, anchor_c))
        if not candidates:
            return None
        r0, c0, mr, mc = rng.choice(candidates)
        r1 = r0 + size - 1
        c1 = c0 + size - 1
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        occupied[mr, mc] = True
        placed[idx] = {
            **plan,
            "r0": r0,
            "c0": c0,
            "r1": r1,
            "c1": c1,
            "mr": mr,
            "mc": mc,
        }

    return [item for item in placed if item is not None]


def _build_case(rng: random.Random) -> np.ndarray | None:
    H = rng.randint(21, 28)
    W = rng.randint(18, 28)
    ref_h = rng.choice((9, 11))
    ref_w = rng.choice((9, 10, 11))
    ref_r0 = rng.randint(1, 3)
    ref_c0 = rng.randint(1, W - ref_w - 2)
    ref_r1 = ref_r0 + ref_h - 1
    ref_c1 = ref_c0 + ref_w - 1

    grid = np.full((H, W), BG, dtype=int)
    reference = _make_reference(rng, ref_h, ref_w)
    grid[ref_r0 : ref_r1 + 1, ref_c0 : ref_c1 + 1] = reference

    window_count = rng.randint(2, 4)
    sizes = [rng.choice((3, 4)) for _ in range(window_count)]
    if 3 not in sizes:
        sizes[rng.randrange(window_count)] = 3
    if 4 not in sizes:
        sizes[rng.randrange(window_count)] = 4

    ref_center_r = (ref_r0 + ref_r1) // 2
    plans = []
    for size in sizes:
        inner = size - 2
        allowed_delta = [-4, -3, -2, 2, 3, 4] if size == 3 else [-5, -4, -3, 3, 4, 5]
        valid_delta = [d for d in allowed_delta if ref_r0 <= ref_center_r + d <= ref_r1 - (inner - 1)]
        if not valid_delta:
            return None
        delta = rng.choice(valid_delta)
        anchor_r = ref_center_r + delta
        anchor_c_min = ref_c0
        anchor_c_max = ref_c1 - (inner - 1)
        if anchor_c_min > anchor_c_max:
            return None
        anchor_c = rng.randint(anchor_c_min, anchor_c_max)
        plans.append({"size": size, "anchor_r": anchor_r, "anchor_c": anchor_c})

    placed = _sample_window_positions(rng, H, W, (ref_r0, ref_c0, ref_r1, ref_c1), plans)
    if placed is None:
        return None

    for item in placed:
        _draw_frame(grid, item["r0"], item["c0"], item["size"])
        grid[item["mr"], item["mc"]] = RED

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _build_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
