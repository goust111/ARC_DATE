# Combo variant of 1818057f (index 20)
# combo_spec: D6/1+D12/0
# Name: 对称孤立十字检测

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg


# Design analysis
# a) Concrete behavior and trace:
#    - The original task recolors any local '+' center, even when that center sits inside
#      a larger blob. For example, the 6-cell component
#      {(1,1),(0,1),(2,1),(1,0),(1,2),(1,3)} contains a valid local '+' at (1,1), so the
#      original solver recolors five cells and leaves the tail cell yellow.
#    - In the fused variant, the unit of reasoning is the full 4-connected component:
#      first require component size exactly 5, then require whole-component mirror symmetry,
#      then verify the component footprint is exactly a '+'. The same 6-cell component now
#      fails at the size gate and stays unchanged.
#    - A 5-cell line such as {(0,0),(0,1),(0,2),(0,3),(0,4)} is mirror-symmetric but not
#      a '+', so it survives unchanged as a distractor. A lopsided 5-cell L passes the
#      size gate but fails symmetry. Only the exact isolated cross passes all filters.
# b) Complexity calibration:
#    - If the generator emitted only isolated crosses plus random noise, the task would
#      visually collapse to the D6 single-variant. To keep both dimensions legible, each
#      sample includes at least one isolated '+', at least one larger embedded-plus
#      component, and at least one symmetric non-plus component.
#    - The component palette is small and repeated: exact pluses, embedded pluses, lines,
#      T-shapes, and lopsided pentominoes. That keeps the rule inferable from 2-4 examples
#      without creating too many unrelated degrees of freedom.
# c) Implicit assumption review of the original solver:
#    - Original `for r,c` interior scan assumes the rule is local per pixel. That breaks:
#      whether a local '+' matters now depends on the size and symmetry of its full
#      component, so we must extract connected components before recoloring.
#    - Original `out = g.copy()` is still valid because non-qualifying foreground should
#      remain visible, unlike the pure D12 variant that erased asymmetric blobs.
#    - Original `colors[0]` works only because there is exactly one foreground color.
#      This fused solver keeps that assumption, and the generator enforces it.
# d) Placement legality:
#    - Every component is placed only on background cells and must keep a 4-neighbor moat
#      from earlier components, so no accidental merges occur.
#    - Shape transforms are normalized before placement, guaranteeing in-bounds bbox math.
# e) Visual observability:
#    - The user can compare three nearby structural cases: an exact isolated cross turns
#      teal, a bigger cross-with-tail does not, and a symmetric non-cross line also does
#      not. That exposes both the isolation test and the shape/symmetry test visually.
# f) Visual cleanliness:
#    - Components are small, disconnected, and separated by a moat, so each object is
#      readable as a whole rather than blending into one dense foreground mass.
# g) Color-change inference:
#    - Only one recolor exists: foreground -> teal. The deciding cue is purely structural:
#      exact 5-cell symmetric cross component.
# h) Quantitative non-degeneracy target:
#    - Generated samples are validated to contain all three observable branches:
#      recolored isolated pluses, ignored embedded-plus components, and ignored symmetric
#      non-plus components. They are also checked against the legacy solver so the fused
#      dataset does not collapse back to the original "recolor every local plus" behavior.
# i) New solver assumptions and generator guarantees:
#    - `main()` assumes target components are isolated 4-connected components of size 5;
#      `_can_place_component()` enforces a moat so component membership is unambiguous.
#    - `main()` assumes symmetry is tested on the exact component footprint in its own bbox;
#      the generator uses the same symmetry helper when validating mandatory distractors.


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PLUS_OFFSETS = [(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)]

SYMMETRIC_NON_PLUS_SHAPES = [
    [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
    [(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
]

ASYMMETRIC_SIZE5_SHAPES = [
    [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
    [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
]

FILLER_SHAPES = [
    [(0, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(0, 0), (1, 0), (2, 0), (3, 0)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
]

ASYMMETRIC_FILLER_SHAPES = [
    [(0, 0), (1, 0), (1, 1)],
    [(0, 1), (1, 0), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 1), (1, 1), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0)],
    [(0, 0), (0, 1), (0, 2), (1, 2)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
    [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
    [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
]

SYMMETRIC_FILLER_SHAPES = [
    [(0, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(0, 0), (1, 0), (2, 0), (3, 0)],
]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
    return pts, r1 - r0 + 1, c1 - c0 + 1


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
    pts, height, width = _component_to_local_points(coords)
    return _has_vertical_axis(pts, width) or _has_horizontal_axis(pts, height)


def _contains_local_plus(comp: list[tuple[int, int]]) -> bool:
    cells = set(comp)
    for r, c in comp:
        if all((r + dr, c + dc) in cells for dr, dc in DIRS4):
            return True
    return False


def _is_exact_plus(comp: list[tuple[int, int]]) -> bool:
    if len(comp) != 5:
        return False
    return _contains_local_plus(comp)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor only perfectly isolated symmetric '+' components to teal.

    Input structure:
    - The most common color is the background.
    - Exactly one foreground color forms several disconnected 4-connected components.
    - Some components are exact 5-cell crosses, some are larger blobs containing an embedded '+',
      and some are symmetric shapes that are not crosses at all.

    Rule:
    - Split the foreground into 4-connected components.
    - Recolor a component to teal (8) only if all three tests pass:
      it has size exactly 5, it has at least one horizontal or vertical mirror axis,
      and its footprint is exactly a thin '+' with one center and four orthogonal arms.
    - Leave every larger embedded-plus component and every non-plus symmetric component unchanged.

    Key insight: The correct target is not a local '+' neighborhood but a whole component that is
    simultaneously isolated, mirror-symmetric, and topologically the exact 5-cell cross.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    for comp in _connected_components(g == fg):
        if len(comp) != 5:
            continue
        if not _component_has_axis_symmetry(comp):
            continue
        if not _is_exact_plus(comp):
            continue
        for r, c in comp:
            out[r, c] = 8
    return out


def _normalize_shape(shape: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return sorted((r - min_r, c - min_c) for r, c in shape)


def _transform_shape(
    shape: list[tuple[int, int]],
    rot: int,
    flip_lr: bool,
) -> list[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in shape]
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    if flip_lr:
        pts = [(r, -c) for r, c in pts]
    return _normalize_shape(pts)


def _random_shape_variant(rng: np.random.Generator, shape: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return _transform_shape(shape, int(rng.integers(0, 4)), bool(rng.integers(0, 2)))


def _translated(shape: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + dr, c0 + dc) for dr, dc in shape]


def _shape_bounds(shape: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _can_place_component(grid: np.ndarray, cells: list[tuple[int, int]], bg: int) -> bool:
    H, W = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != bg:
            return False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in cell_set and int(grid[nr, nc]) != bg:
                return False
    return True


def _place_component(grid: np.ndarray, cells: list[tuple[int, int]], fg: int) -> None:
    for r, c in cells:
        grid[r, c] = fg


def _attempt_place_shape(
    grid: np.ndarray,
    rng: np.random.Generator,
    shape: list[tuple[int, int]],
    bg: int,
    fg: int,
    tries: int = 200,
) -> bool:
    H, W = grid.shape
    h, w = _shape_bounds(shape)
    for _ in range(tries):
        r0 = int(rng.integers(0, H - h + 1))
        c0 = int(rng.integers(0, W - w + 1))
        cells = _translated(shape, r0, c0)
        if _can_place_component(grid, cells, bg):
            _place_component(grid, cells, fg)
            return True
    return False


def _isolated_plus_shape() -> list[tuple[int, int]]:
    return list(PLUS_OFFSETS)


def _embedded_plus_shape(asymmetric: bool) -> list[tuple[int, int]]:
    if asymmetric:
        return [(1, 1), (0, 1), (2, 1), (1, 0), (1, 2), (1, 3), (0, 3)]
    return [(1, 2), (0, 2), (2, 2), (1, 1), (1, 3), (1, 0), (1, 4)]


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])
    H, W = g.shape
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(g[r, c]) != fg:
                continue
            if all(int(g[r + dr, c + dc]) == fg for dr, dc in DIRS4):
                for dr, dc in [(0, 0), *DIRS4]:
                    out[r + dr, c + dc] = 8
    return out


def _component_metrics(grid: np.ndarray, fg: int) -> dict[str, int]:
    comps = _connected_components(grid == fg)
    metrics = {
        "isolated_plus": 0,
        "embedded_plus": 0,
        "embedded_plus_asym": 0,
        "embedded_plus_sym": 0,
        "symmetric_non_plus": 0,
        "asymmetric_size5": 0,
    }
    for comp in comps:
        symmetric = _component_has_axis_symmetry(comp)
        exact_plus = _is_exact_plus(comp)
        local_plus = _contains_local_plus(comp)
        if exact_plus and symmetric:
            metrics["isolated_plus"] += 1
        elif local_plus:
            metrics["embedded_plus"] += 1
            if symmetric:
                metrics["embedded_plus_sym"] += 1
            else:
                metrics["embedded_plus_asym"] += 1
        elif symmetric:
            metrics["symmetric_non_plus"] += 1
        elif len(comp) == 5:
            metrics["asymmetric_size5"] += 1
    return metrics


def _fill_with_distractors(
    grid: np.ndarray,
    rng: np.random.Generator,
    bg: int,
    fg: int,
    target_fg: int,
) -> None:
    current_fg = int(np.sum(grid == fg))
    stall = 0
    while current_fg < target_fg and stall < 500:
        metrics = _component_metrics(grid, fg)
        if metrics["symmetric_non_plus"] < 3 and bool(rng.integers(0, 5) == 0):
            shape_bank = SYMMETRIC_NON_PLUS_SHAPES + SYMMETRIC_FILLER_SHAPES
        else:
            shape_bank = ASYMMETRIC_FILLER_SHAPES
        shape = _random_shape_variant(rng, shape_bank[int(rng.integers(0, len(shape_bank)))])
        if _attempt_place_shape(grid, rng, shape, bg, fg, tries=60):
            current_fg += len(shape)
            stall = 0
        else:
            stall += 1


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(12, 23))
    bg, fg = 2, 4
    grid = np.full((H, W), bg, dtype=int)

    mandatory_shapes: list[list[tuple[int, int]]] = []

    keep_count = int(rng.integers(1, 4))
    for _ in range(keep_count):
        mandatory_shapes.append(_random_shape_variant(rng, _isolated_plus_shape()))

    mandatory_shapes.append(_random_shape_variant(rng, _embedded_plus_shape(True)))
    if bool(rng.integers(0, 2)):
        mandatory_shapes.append(_random_shape_variant(rng, _embedded_plus_shape(True)))
    mandatory_shapes.append(_random_shape_variant(rng, _embedded_plus_shape(False)))
    mandatory_shapes.append(
        _random_shape_variant(
            rng,
            SYMMETRIC_NON_PLUS_SHAPES[int(rng.integers(0, len(SYMMETRIC_NON_PLUS_SHAPES)))],
        )
    )
    mandatory_shapes.append(
        _random_shape_variant(
            rng,
            ASYMMETRIC_SIZE5_SHAPES[int(rng.integers(0, len(ASYMMETRIC_SIZE5_SHAPES)))],
        )
    )

    mandatory_shapes.sort(key=len, reverse=True)
    for shape in mandatory_shapes:
        if not _attempt_place_shape(grid, rng, shape, bg, fg):
            return None

    target_density = float(rng.uniform(0.24, 0.36))
    target_fg = int(H * W * target_density)
    _fill_with_distractors(grid, rng, bg, fg, target_fg)

    fg_cells = int(np.sum(grid == fg))
    density = fg_cells / float(H * W)
    if fg_cells >= (H * W) // 2:
        return None
    if not (0.20 <= density <= 0.42):
        return None

    metrics = _component_metrics(grid, fg)
    if metrics["isolated_plus"] < 1:
        return None
    if metrics["embedded_plus_asym"] < 1:
        return None
    if metrics["embedded_plus_sym"] < 1:
        return None
    if metrics["symmetric_non_plus"] < 1:
        return None
    if metrics["symmetric_non_plus"] > 3:
        return None
    if metrics["asymmetric_size5"] < 1:
        return None

    new_out = main(grid)
    old_out = _legacy_main(grid)
    if not np.any(new_out == 8):
        return None
    if np.array_equal(new_out, old_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
