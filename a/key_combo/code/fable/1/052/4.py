from __future__ import annotations

from collections import Counter
import hashlib
import random

import numpy as np


_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _signature(grid: np.ndarray) -> str:
    arr = np.asarray(grid, dtype=int)
    return hashlib.md5(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def _center_place(patch: np.ndarray, out_shape: tuple[int, int], bg: int) -> np.ndarray:
    patch = np.asarray(patch, dtype=int)
    out_h, out_w = map(int, out_shape)
    ph, pw = map(int, patch.shape)
    dr = out_h - ph
    dc = out_w - pw
    assert dr >= 0 and dc >= 0, "patch must fit inside centered canvas"
    assert dr % 2 == 0 and dc % 2 == 0, "bbox centers must align on integer cells"
    r0 = dr // 2
    c0 = dc // 2
    out = np.full((out_h, out_w), int(bg), dtype=int)
    view = out[r0 : r0 + ph, c0 : c0 + pw]
    mask = patch != int(bg)
    view[mask] = patch[mask]
    return out


def _aligned_overlay(grid: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one non-background color"

    patches: list[tuple[int, np.ndarray]] = []
    out_h = 0
    out_w = 0
    for color in colors:
        mask = g == color
        box = bbox(mask)
        assert box is not None
        patch = crop_bbox(np.where(mask, color, bg), box)
        patches.append((color, patch))
        out_h = max(out_h, int(patch.shape[0]))
        out_w = max(out_w, int(patch.shape[1]))

    out = np.full((out_h, out_w), bg, dtype=int)
    for _color, patch in patches:
        placed = _center_place(patch, (out_h, out_w), bg)
        overlap = (out != bg) & (placed != bg)
        assert not np.any(overlap), "centered fragments must not overlap"
        out[placed != bg] = placed[placed != bg]
    return out, bg


def _enclosed_regions(grid: np.ndarray, bg: int) -> list[tuple[np.ndarray, Counter[int], int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    enclosed: list[tuple[np.ndarray, Counter[int], int]] = []
    for comp in components_mask(g == int(bg)):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == h - 1) or np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == w - 1):
            continue
        boundary = set()
        for r, c in comp:
            for nr, nc in neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != int(bg):
                    boundary.add((int(nr), int(nc)))
        counts = Counter(int(g[r, c]) for r, c in boundary)
        assert counts, "every enclosed region must have a foreground boundary"
        ordered = counts.most_common()
        assert len(ordered) == 1 or ordered[0][1] > ordered[1][1], "boundary majority must be unique"
        enclosed.append((comp, counts, int(ordered[0][0])))
    return enclosed


def _fill_lakes(grid: np.ndarray, bg: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for comp, _counts, fill_color in _enclosed_regions(out, bg):
        out[comp[:, 0], comp[:, 1]] = int(fill_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Center-align color fragments by their own bounding boxes, overlay them, then fill each enclosed lake with its boundary-majority color.

    Input structure:
    - One background color dominates the grid.
    - Each non-background color forms one scattered fragment pattern, possibly split into multiple 4-connected pieces.
    - Every color pattern is open on its own, but its tight bbox is the reference used for later alignment.

    Rule:
    - For each non-background color, crop its tight bbox and place that crop onto a new background canvas whose size is the per-axis maximum fragment bbox size.
    - Center-align every cropped fragment by bbox center and overlay them transparently; the centered fragments do not overlap.
    - For every background component that is enclosed after overlay, count the colors of its unique 4-neighbor boundary pixels and fill the whole component with the unique majority color.

    Key insight:
    - The lake does not exist in any input fragment alone; it appears only when all color crops are center-aligned, and the fill color comes from the enclosing boundary composition rather than from area or position.
    """
    overlay, bg = _aligned_overlay(np.asarray(grid, dtype=int))
    return _fill_lakes(overlay, bg)


def _build_single_lake(bg: int, colors: list[int]) -> tuple[np.ndarray, dict]:
    horiz, vert = map(int, colors[:2])
    valid = [(hh, ww) for hh in (3, 5, 7, 9, 11) for ww in (3, 5, 7, 9, 11) if hh != ww]
    hole_h, hole_w = random.choice(valid)
    out_h = hole_h + 2
    out_w = hole_w + 2
    corner_owner = random.choice(["h", "v"])

    grid = np.full((out_h, out_w), bg, dtype=int)
    if corner_owner == "h":
        grid[0, :] = horiz
        grid[-1, :] = horiz
        grid[1:-1, 0] = vert
        grid[1:-1, -1] = vert
    else:
        grid[0, 1:-1] = horiz
        grid[-1, 1:-1] = horiz
        grid[:, 0] = vert
        grid[:, -1] = vert

    dominant = horiz if hole_w > hole_h else vert
    return grid, {"holes": 1, "dominant": [dominant], "mode": "single"}


def _build_double_lr(bg: int, colors: list[int]) -> tuple[np.ndarray, dict]:
    tb, outer, divider = map(int, colors[:3])
    valid = [
        (hh, ww)
        for hh in (3, 5, 7)
        for ww in (3, 5)
        if 2 * ww > hh
    ]
    hole_h, hole_w = random.choice(valid)
    out_h = hole_h + 2
    out_w = 2 * hole_w + 3
    corner_owner = random.choice(["tb", "outer"])

    grid = np.full((out_h, out_w), bg, dtype=int)
    if corner_owner == "tb":
        grid[0, :] = tb
        grid[-1, :] = tb
        grid[1:-1, 0] = outer
        grid[1:-1, -1] = outer
    else:
        grid[0, 1:-1] = tb
        grid[-1, 1:-1] = tb
        grid[:, 0] = outer
        grid[:, -1] = outer
    grid[1:-1, hole_w + 1] = divider

    return grid, {"holes": 2, "dominant": [tb, tb], "mode": "double_lr"}


def _build_double_tb(bg: int, colors: list[int]) -> tuple[np.ndarray, dict]:
    lr, outer, divider = map(int, colors[:3])
    valid = [
        (hh, ww)
        for hh in (3, 5)
        for ww in (3, 5, 7)
        if 2 * hh > ww
    ]
    hole_h, hole_w = random.choice(valid)
    out_h = 2 * hole_h + 3
    out_w = hole_w + 2
    corner_owner = random.choice(["lr", "outer"])

    grid = np.full((out_h, out_w), bg, dtype=int)
    if corner_owner == "lr":
        grid[:, 0] = lr
        grid[:, -1] = lr
        grid[0, 1:-1] = outer
        grid[-1, 1:-1] = outer
    else:
        grid[1:-1, 0] = lr
        grid[1:-1, -1] = lr
        grid[0, :] = outer
        grid[-1, :] = outer
    grid[hole_h + 1, 1:-1] = divider

    return grid, {"holes": 2, "dominant": [lr, lr], "mode": "double_tb"}


def _extract_color_patches(overlay: np.ndarray, bg: int) -> list[tuple[int, np.ndarray]]:
    patches: list[tuple[int, np.ndarray]] = []
    for color in [int(v) for v in np.unique(overlay) if int(v) != bg]:
        mask = overlay == color
        box = bbox(mask)
        assert box is not None
        patches.append((color, crop_bbox(np.where(mask, color, bg), box)))
    random.shuffle(patches)
    return patches


def _rect_clear(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for ar0, ac0, ar1, ac1 in rects:
        if not (r1 + pad < ar0 or ar1 + pad < r0 or c1 + pad < ac0 or ac1 + pad < c0):
            return False
    return True


def _scatter_patches(patches: list[tuple[int, np.ndarray]], bg: int) -> np.ndarray | None:
    patch_dims = [patch.shape for _color, patch in patches]
    min_h = max(int(h) for h, _w in patch_dims) + 2
    min_w = max(int(w) for _h, w in patch_dims) + 2
    total_h = sum(int(h) for h, _w in patch_dims) + len(patches) + 1
    total_w = sum(int(w) for _h, w in patch_dims) + len(patches) + 1

    for _ in range(120):
        grid_h = random.randint(min(min_h, 30), min(30, max(min_h, total_h + 5)))
        grid_w = random.randint(min(min_w, 30), min(30, max(min_w, total_w + 5)))
        if grid_h < min_h or grid_w < min_w:
            continue
        canvas = np.full((grid_h, grid_w), bg, dtype=int)
        rects: list[tuple[int, int, int, int]] = []
        order = list(range(len(patches)))
        random.shuffle(order)
        order.sort(key=lambda idx: int(np.prod(patches[idx][1].shape)), reverse=True)

        ok = True
        for idx in order:
            color, patch = patches[idx]
            ph, pw = map(int, patch.shape)
            if grid_h - ph - 1 < 1 or grid_w - pw - 1 < 1:
                ok = False
                break
            placed = False
            for _ in range(200):
                r0 = random.randint(1, grid_h - ph - 1)
                c0 = random.randint(1, grid_w - pw - 1)
                if not _rect_clear(rects, r0, c0, ph, pw, pad=1):
                    continue
                view = canvas[r0 : r0 + ph, c0 : c0 + pw]
                mask = patch != bg
                if np.any(view[mask] != bg):
                    continue
                view[mask] = int(color)
                rects.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue
        if get_bg(canvas) != bg:
            continue
        return canvas
    return None


def _has_enclosed_bg(grid: np.ndarray, bg: int) -> bool:
    return len(_enclosed_regions(grid, bg)) > 0


def _validate_instance(inp: np.ndarray, structure: np.ndarray, meta: dict) -> bool:
    inp = np.asarray(inp, dtype=int)
    structure = np.asarray(structure, dtype=int)
    bg = get_bg(inp)

    if inp.ndim != 2 or structure.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(structure.shape) > 30:
        return False
    if get_bg(structure) != bg:
        return False
    if _has_enclosed_bg(inp, bg):
        return False

    reconstructed, recon_bg = _aligned_overlay(inp)
    if recon_bg != bg or not np.array_equal(reconstructed, structure):
        return False

    colors = [int(v) for v in np.unique(inp) if int(v) != bg]
    if len(colors) not in (2, 3):
        return False

    for color in colors:
        single = np.full_like(reconstructed, bg)
        mask = reconstructed == color
        single[mask] = color
        if _has_enclosed_bg(single, bg):
            return False

    holes = _enclosed_regions(reconstructed, bg)
    if len(holes) != int(meta["holes"]):
        return False
    if np.array_equal(_fill_lakes(reconstructed, bg), reconstructed):
        return False

    dominant = [fill_color for _comp, _counts, fill_color in holes]
    if sorted(dominant) != sorted(meta["dominant"]):
        return False

    out = main(inp)
    if not np.array_equal(out, _fill_lakes(structure, bg)):
        return False
    return True


def _sample_structure() -> tuple[np.ndarray, dict]:
    bg = random.randint(0, 9)
    branch = random.choices(
        population=["single", "double_lr", "double_tb"],
        weights=[5, 2, 2],
        k=1,
    )[0]
    color_count = 2 if branch == "single" else 3
    colors = random.sample([c for c in range(10) if c != bg], color_count)
    if branch == "single":
        return _build_single_lake(bg, colors)
    if branch == "double_lr":
        return _build_double_lr(bg, colors)
    return _build_double_tb(bg, colors)


def generate_input() -> np.ndarray:
    for _ in range(400):
        structure, meta = _sample_structure()
        bg = get_bg(structure)
        patches = _extract_color_patches(structure, bg)
        inp = _scatter_patches(patches, bg)
        if inp is None:
            continue
        out = main(inp)
        out_sig = _signature(out)
        if out_sig in _SEEN_OUTPUT_SIGNATURES:
            continue
        if not _validate_instance(inp, structure, meta):
            continue
        _SEEN_OUTPUT_SIGNATURES.add(out_sig)
        return inp
    raise AssertionError("failed to generate a valid lake-overlay instance")
