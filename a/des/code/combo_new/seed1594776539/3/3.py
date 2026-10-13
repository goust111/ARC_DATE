from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 400


def _neighbors(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            pts = np.array(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            comps.append(
                {
                    "coords": pts,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return comps


def _mask_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    h, w = mask.shape
    start = tuple(int(x) for x in cells[0])
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _valid_half_mask(mask: np.ndarray, span_kind: str) -> bool:
    if not _mask_connected(mask):
        return False
    if not mask[0].any() or not mask[-1].any():
        return False
    if not mask[:, 0].any() or not mask[:, -1].any():
        return False
    if span_kind == "cols":
        return bool(mask.any(axis=0).all())
    if span_kind == "rows":
        return bool(mask.any(axis=1).all())
    raise AssertionError(f"unknown span kind: {span_kind}")


def _component_colors(grid: np.ndarray, comp: dict) -> np.ndarray:
    coords = comp["coords"]
    return np.unique(grid[coords[:, 0], coords[:, 1]])


def _is_horizontal_strip(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    if r0 != r1 or c1 == c0:
        return False
    values = grid[r0, c0 : c1 + 1]
    return len(set(int(v) for v in values)) == values.size


def _is_vertical_strip(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    if c0 != c1 or r1 == r0:
        return False
    values = grid[r0 : r1 + 1, c0]
    return len(set(int(v) for v in values)) == values.size


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each half-missing symmetric figure using the adjacent strip as recoloring instructions.

    Input structure:
    - The background is black.
    - Each non-black monochrome component is the complete left half or complete top half of a
      larger mirror-symmetric figure.
    - A horizontal multicolor strip sits one step down-right from a left half and encodes the
      colors for the missing right-half columns; a vertical multicolor strip sits one step
      down-right from a top half and encodes the colors for the missing bottom-half rows.

    Rule:
    - For every horizontal strip, find the unique monochrome component whose lower-right corner
      is diagonally above-left of the strip and whose width matches the strip length.
    - Mirror that component across the vertical axis just to its right, producing the missing
      right half, and color each reflected column by the strip colors from nearest the axis
      outward.
    - For every vertical strip, find the unique monochrome component whose lower-right corner is
      diagonally above-left of the strip and whose height matches the strip length.
    - Mirror that component across the horizontal axis just below it, producing the missing
      bottom half, and color each reflected row by the strip colors from nearest the axis
      outward. Keep the original halves and the strips unchanged.

    Key insight:
    - The monochrome component gives only the geometry of one symmetric half; the nearby strip is
      not decoration but an ordered code telling how to recolor the reflected layers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    comps = _components(g != BG)
    strips: list[tuple[str, dict]] = []
    patterns: list[dict] = []

    for comp in comps:
        if _is_horizontal_strip(g, comp):
            strips.append(("vertical", comp))
            continue
        if _is_vertical_strip(g, comp):
            strips.append(("horizontal", comp))
            continue

        colors = _component_colors(g, comp)
        assert colors.size == 1, "pattern halves must be monochrome"
        patterns.append(comp)

    used: set[int] = set()

    for kind, strip in strips:
        sr0, sc0, sr1, sc1 = strip["bbox"]
        if kind == "vertical":
            strip_colors = g[sr0, sc0 : sc1 + 1]
            span = int(strip_colors.size)
            candidates = []
            for idx, comp in enumerate(patterns):
                if idx in used:
                    continue
                pr0, pc0, pr1, pc1 = comp["bbox"]
                if pr1 + 1 == sr0 and pc1 + 1 == sc0 and (pc1 - pc0 + 1) == span:
                    candidates.append((idx, comp))
            assert len(candidates) == 1, "expected one pattern half for each horizontal strip"

            idx, comp = candidates[0]
            used.add(idx)
            pr0, pc0, pr1, pc1 = comp["bbox"]
            for r, c in comp["coords"]:
                rr = int(r)
                cc = 2 * pc1 + 1 - int(c)
                dist = cc - (pc1 + 1)
                out[rr, cc] = int(strip_colors[dist])
        else:
            strip_colors = g[sr0 : sr1 + 1, sc0]
            span = int(strip_colors.size)
            candidates = []
            for idx, comp in enumerate(patterns):
                if idx in used:
                    continue
                pr0, pc0, pr1, pc1 = comp["bbox"]
                if pr1 + 1 == sr0 and pc1 + 1 == sc0 and (pr1 - pr0 + 1) == span:
                    candidates.append((idx, comp))
            assert len(candidates) == 1, "expected one pattern half for each vertical strip"

            idx, comp = candidates[0]
            used.add(idx)
            pr0, pc0, pr1, pc1 = comp["bbox"]
            for r, c in comp["coords"]:
                rr = 2 * pr1 + 1 - int(r)
                cc = int(c)
                dist = rr - (pr1 + 1)
                out[rr, cc] = int(strip_colors[dist])

    assert len(used) == len(patterns) == len(strips), "every pattern must pair with exactly one strip"
    return out


def _generate_half_mask(h: int, w: int, span_kind: str) -> np.ndarray | None:
    mask = np.ones((h, w), dtype=bool)
    lower = max(4, int(np.ceil(0.45 * h * w)))
    upper = max(lower, int(np.floor(0.8 * h * w)))
    target = random.randint(lower, upper)

    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    stalled = 0
    limit = max(50, 6 * len(cells))

    while int(mask.sum()) > target and stalled < limit:
        r, c = random.choice(cells)
        if not mask[r, c]:
            stalled += 1
            continue
        mask[r, c] = False
        if _valid_half_mask(mask, span_kind):
            stalled = 0
        else:
            mask[r, c] = True
            stalled += 1

    return mask if _valid_half_mask(mask, span_kind) else None


def _sample_colors(count: int) -> list[int]:
    return random.sample(PALETTE, count)


def _build_vertical_patch() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(3, 6)
    w = random.randint(2, 6)
    mask = _generate_half_mask(h, w, "cols")
    if mask is None:
        return None

    base, *strip = _sample_colors(w + 1)
    strip_colors = np.array(strip, dtype=int)
    inp = np.zeros((h + 1, 2 * w), dtype=int)
    out = np.zeros_like(inp)

    inp[:h, :w][mask] = base
    out[:h, :w][mask] = base
    inp[h, w:] = strip_colors
    out[h, w:] = strip_colors

    rows, cols = np.where(mask)
    for r, c in zip(rows.tolist(), cols.tolist()):
        mc = 2 * w - 1 - c
        out[r, mc] = int(strip_colors[mc - w])
    return inp, out


def _build_horizontal_patch() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(2, 6)
    w = random.randint(3, 6)
    mask = _generate_half_mask(h, w, "rows")
    if mask is None:
        return None

    base, *strip = _sample_colors(h + 1)
    strip_colors = np.array(strip, dtype=int)
    inp = np.zeros((2 * h, w + 1), dtype=int)
    out = np.zeros_like(inp)

    inp[:h, :w][mask] = base
    out[:h, :w][mask] = base
    inp[h:, w] = strip_colors
    out[h:, w] = strip_colors

    rows, cols = np.where(mask)
    for r, c in zip(rows.tolist(), cols.tolist()):
        mr = 2 * h - 1 - r
        out[mr, c] = int(strip_colors[mr - h])
    return inp, out


def _reserve_slice(lo: int, hi: int, limit: int) -> slice:
    return slice(max(0, lo), min(limit, hi))


def _place_patches(patches: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    if not patches:
        return None

    min_h = max(max(inp.shape[0] for inp, _ in patches) + 2, 24)
    min_w = max(max(inp.shape[1] for inp, _ in patches) + 2, 24)
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(120):
        h = random.randint(min_h, 30)
        w = random.randint(min_w, 30)
        inp_grid = np.zeros((h, w), dtype=int)
        out_grid = np.zeros((h, w), dtype=int)
        reserved = np.zeros((h, w), dtype=bool)
        ok = True

        order = patches[:]
        random.shuffle(order)
        for inp_patch, out_patch in order:
            ph, pw = inp_patch.shape
            positions: list[tuple[int, int]] = []
            for r0 in range(h - ph + 1):
                for c0 in range(w - pw + 1):
                    if reserved[r0 : r0 + ph, c0 : c0 + pw].any():
                        continue
                    positions.append((r0, c0))
            if not positions:
                ok = False
                break

            r0, c0 = random.choice(positions)
            in_mask = inp_patch != BG
            out_mask = out_patch != BG
            inp_grid[r0 : r0 + ph, c0 : c0 + pw][in_mask] = inp_patch[in_mask]
            out_grid[r0 : r0 + ph, c0 : c0 + pw][out_mask] = out_patch[out_mask]

            rr = _reserve_slice(r0 - 1, r0 + ph + 1, h)
            cc = _reserve_slice(c0 - 1, c0 + pw + 1, w)
            reserved[rr, cc] = True

        if ok:
            return inp_grid, out_grid
    return None


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(2, 4)
    n_vertical = random.randint(1, n_objects - 1)
    orientations = ["vertical"] * n_vertical + ["horizontal"] * (n_objects - n_vertical)
    random.shuffle(orientations)

    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for orientation in orientations:
        patch = _build_vertical_patch() if orientation == "vertical" else _build_horizontal_patch()
        if patch is None:
            return None
        patches.append(patch)

    placed = _place_patches(patches)
    if placed is None:
        return None
    inp, expected = placed

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
