# Variant of 1818057f
# Dimension: D12 — 对称性检测与利用 (meta)
# Change: '+'字图案本身是关于中心点四重旋转对称的，但规则中不显式利用对称性... → 仅保留对称的前景连通分量

from __future__ import annotations

import numpy as np
from arc_utils import get_bg

# Design analysis for the D12 variant
# ===================================
# a) concretization_and_trace:
#    - Input still uses one dominant red background (2) and one yellow foreground (4).
#    - The foreground is no longer a dense soup whose local 5-cell '+' motifs matter.
#      Instead it is split into several 4-connected components. Some components are
#      globally mirror-symmetric, some are almost symmetric but have one extra spur.
#    - Output is component-wise: a symmetric component is kept and recolored teal (8);
#      a non-symmetric component is erased to background.
#    - Concrete trace:
#      original: a lopsided blob containing one local '+' would partially turn teal.
#      variant: the same lopsided blob is treated as one whole component; because the
#      complete footprint is not mirror-symmetric, the entire blob disappears.
#      Conversely, a globally symmetric component with no local '+' still survives.
# b) complexity_calibration:
#    - To keep the change visible, each grid contains both kept and erased components.
#    - To avoid an underdetermined "any line is symmetric" dataset, components are
#      generated as medium-size connected blobs (typically 5-25 cells), not singles.
#    - Each grid contains at least one vertically symmetric component and one
#      horizontally symmetric component, so the rule is "has some axis" rather than
#      "always vertical" or "always horizontal".
# c) implicit_assumption_review_of_original_main:
#    - The original `colors[0]` foreground pick was harmless because there was exactly
#      one yellow foreground color. In this variant the true decision unit is the full
#      foreground mask, so the new solver works on all non-background cells together.
#    - The original interior scan assumed the rule is local per pixel. That breaks here:
#      a component can be symmetric even when no pixel has four orthogonal neighbors.
#    - The original `out = g.copy()` preserved non-matching foreground. That would be
#      wrong now because asymmetric components must be erased, so the new solver starts
#      from a background canvas and redraws only kept components.
# d) placement_legality:
#    - Generator places components by tight bounding boxes with a one-cell moat between
#      boxes, so components never merge accidentally and remain visually separable.
#    - All generated components are 4-connected by construction, stay in-bounds, and
#      are revalidated with the same symmetry test used by the solver.
# e) visual_observability:
#    - The visual cue is coarse and readable: whole yellow blobs either vanish or turn
#      teal. This is inferable from 2-3 examples in a way that local plus-detection is
#      not, because some kept components have no '+' and some erased components do.
# f) visual_cleanliness:
#    - No overlapping layers are used. Each component occupies its own region with
#      padding, so the user can inspect one blob at a time.
# g) color_inference:
#    - Colors stay minimal: red background, yellow input foreground, teal output keep.
#      Color change is tied to a structural property only, never to color id ordering.
# h) quantitative_non_degeneracy_target:
#    - Each sample is forced to contain both symmetry branches (keep + erase), and the
#      shapes are validated against the old rule later to confirm the new behavior is
#      not collapsing back into local plus-highlighting.
# i) new_main_assumptions_and_generator_guarantees:
#    - main() assumes foreground components are 4-disconnected from each other.
#      generate_input() enforces a one-cell gap between component bounding boxes.
#    - main() assumes symmetry is checked on a component's exact footprint inside its
#      own bbox. generate_input() uses the same helper, so kept/erased labels align.


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _extract_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(H):
        for c in range(W):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _component_to_local_points(coords: list[tuple[int, int]]):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    pts = {(r - r0, c - c0) for r, c in coords}
    return pts, r0, c0, r1 - r0 + 1, c1 - c0 + 1


def _has_vertical_axis(pts: set[tuple[int, int]], width: int) -> bool:
    for k in range(0, 2 * (width - 1) + 1):
        ok = True
        for r, c in pts:
            cc = k - c
            if not (0 <= cc < width and (r, cc) in pts):
                ok = False
                break
        if ok:
            return True
    return False


def _has_horizontal_axis(pts: set[tuple[int, int]], height: int) -> bool:
    for k in range(0, 2 * (height - 1) + 1):
        ok = True
        for r, c in pts:
            rr = k - r
            if not (0 <= rr < height and (rr, c) in pts):
                ok = False
                break
        if ok:
            return True
    return False


def _component_has_axis_symmetry(coords: list[tuple[int, int]]) -> bool:
    pts, _, _, h, w = _component_to_local_points(coords)
    return _has_vertical_axis(pts, w) or _has_horizontal_axis(pts, h)


def _mask_has_axis_symmetry(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not coords:
        return False
    return _component_has_axis_symmetry(coords)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_extract_components(mask)) == 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only mirror-symmetric foreground components and recolor them teal.

    Input structure:
    - bg = most common color.
    - One foreground color forms several disconnected 4-connected components on the bg.
    - Some components are exact mirror-symmetric as whole shapes; others are slightly lopsided.

    Rule:
    - Split all non-background pixels into 4-connected components.
    - If a component has at least one horizontal or vertical mirror axis, recolor the entire component to teal (8).
    - If a component has no such axis, erase the entire component to bg.

    Key insight: The decision is global per component, not local per pixel. A blob survives only when its complete footprint matches itself under some mirror reflection.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    mask = g != bg
    out = np.full_like(g, bg)

    for comp in _extract_components(mask):
        if _component_has_axis_symmetry(comp):
            for r, c in comp:
                out[r, c] = 8

    return out


def _random_profile(rng: np.random.Generator, length: int, lo: int, hi: int) -> list[int]:
    profile = [int(rng.integers(lo, hi + 1))]
    for _ in range(1, length):
        prev = profile[-1]
        delta = int(rng.integers(-1, 2))
        profile.append(min(hi, max(lo, prev + delta)))
    return profile


def _build_vertical_symmetric_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(80):
        odd = bool(rng.integers(0, 2))
        h = int(rng.integers(3, 8))
        side = int(rng.integers(1, 4))
        if odd:
            w = 2 * side + 1
            radii = _random_profile(rng, h, 0, side)
            if max(radii) == 0:
                continue
            mask = np.zeros((h, w), dtype=bool)
            center = side
            for r, radius in enumerate(radii):
                mask[r, center] = True
                for d in range(1, radius + 1):
                    mask[r, center - d] = True
                    mask[r, center + d] = True
        else:
            w = 2 * side
            extents = _random_profile(rng, h, 1, side)
            mask = np.zeros((h, w), dtype=bool)
            for r, extent in enumerate(extents):
                left = side - extent
                right = side + extent
                mask[r, left:right] = True

        area = int(mask.sum())
        if not (5 <= area <= 25):
            continue
        if not _is_connected(mask):
            continue
        if not _mask_has_axis_symmetry(mask):
            continue
        return mask
    raise RuntimeError("failed to build symmetric component")


def _build_symmetric_component(rng: np.random.Generator, orientation: str) -> np.ndarray:
    mask = _build_vertical_symmetric_mask(rng)
    if orientation == "horizontal":
        mask = mask.T
    elif orientation != "vertical":
        raise ValueError("orientation must be vertical or horizontal")
    return mask.astype(int)


def _adjacent_background_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    H, W = mask.shape
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < H and 0 <= nc < W and not bool(mask[nr, nc]):
                cells.add((nr, nc))
    return list(cells)


def _build_asymmetric_component(rng: np.random.Generator) -> np.ndarray:
    for _ in range(120):
        orientation = "vertical" if bool(rng.integers(0, 2)) else "horizontal"
        base = _build_symmetric_component(rng, orientation).astype(bool)
        candidates = _adjacent_background_cells(base)
        if not candidates:
            continue
        rng.shuffle(candidates)

        for r, c in candidates:
            test = base.copy()
            test[r, c] = True
            if 5 <= int(test.sum()) <= 26 and _is_connected(test) and not _mask_has_axis_symmetry(test):
                return test.astype(int)

            second_hop = []
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < test.shape[0] and 0 <= nc < test.shape[1] and not bool(test[nr, nc]):
                    second_hop.append((nr, nc))
            rng.shuffle(second_hop)
            for nr, nc in second_hop[:4]:
                test2 = test.copy()
                test2[nr, nc] = True
                if 6 <= int(test2.sum()) <= 28 and _is_connected(test2) and not _mask_has_axis_symmetry(test2):
                    return test2.astype(int)
    raise RuntimeError("failed to build asymmetric component")


def _rects_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _place_component(
    rng: np.random.Generator,
    canvas: np.ndarray,
    shape: np.ndarray,
    color: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    H, W = canvas.shape
    h, w = shape.shape
    for _ in range(200):
        r0 = int(rng.integers(0, H - h + 1))
        c0 = int(rng.integers(0, W - w + 1))
        rect = (r0, c0, r0 + h - 1, c0 + w - 1)
        if any(_rects_too_close(rect, other) for other in placed_rects):
            continue
        for rr in range(h):
            for cc in range(w):
                if int(shape[rr, cc]):
                    canvas[r0 + rr, c0 + cc] = color
        placed_rects.append(rect)
        return r0, c0
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(12, 23))
    bg, fg, keep = 2, 4, 8

    sym_components = [
        _build_symmetric_component(rng, "vertical"),
        _build_symmetric_component(rng, "horizontal"),
    ]
    asym_components = [_build_asymmetric_component(rng)]

    extra_count = int(rng.integers(0, 3))
    for _ in range(extra_count):
        if bool(rng.integers(0, 2)):
            orientation = "vertical" if bool(rng.integers(0, 2)) else "horizontal"
            sym_components.append(_build_symmetric_component(rng, orientation))
        else:
            asym_components.append(_build_asymmetric_component(rng))

    items: list[tuple[np.ndarray, bool]] = [(comp, True) for comp in sym_components] + [(comp, False) for comp in asym_components]
    order = rng.permutation(len(items))
    items = [items[int(i)] for i in order]

    inp = np.full((H, W), bg, dtype=int)
    out = np.full((H, W), bg, dtype=int)
    placed_rects: list[tuple[int, int, int, int]] = []

    for shape, keep_component in items:
        placed = _place_component(rng, inp, shape, fg, placed_rects)
        if placed is None:
            return None
        r0, c0 = placed
        if keep_component:
            h, w = shape.shape
            for rr in range(h):
                for cc in range(w):
                    if int(shape[rr, cc]):
                        out[r0 + rr, c0 + cc] = keep

    fg_cells = int(np.sum(inp != bg))
    total = H * W
    density = fg_cells / total
    if fg_cells >= total // 2:
        return None
    if not (0.18 <= density <= 0.42):
        return None
    if not np.any(out == keep):
        return None
    if not np.any((inp != bg) & (out == bg)):
        return None
    if np.array_equal(main(inp), inp):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
