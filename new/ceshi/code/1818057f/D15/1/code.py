# Variant of 1818057f
# Dimension: D15 — 图例/符号指令 (symbol_instructions)
# Change: 无图例或符号指令，变换规则完全隐含在输入输出的差异中... → 网格角落的迷你'+'样板

from __future__ import annotations

from collections import Counter
import numpy as np


# Design analysis for the D15 variant
# -----------------------------------
# a) Concretization and behavior trace
#    - The top-left 3x3 corner now contains a teal '+' example:
#        .#.
#        ###
#        .#.
#      on the red background. Rows/cols 0..4 are otherwise kept clear so the
#      example reads as a legend instead of another object in the field.
#    - The main field contains yellow 4-connected components. Some components
#      are exact 5-cell '+' copies; some are larger distractors, including
#      "plus with a tail" shapes that contain a local cross center.
#    - Original task at a decoy component:
#      yellow plus + one extra yellow tail cell -> local center still has four
#      orthogonal yellow neighbors -> original solver recolors the five cross
#      cells.
#    - Variant task at the same component:
#      the corner legend describes an entire object, so matching is done at the
#      component level; the 6-cell decoy no longer matches the 5-cell legend and
#      stays yellow. Only isolated 5-cell plus components are recolored teal.
#    - This avoids a degenerate "decorative legend only" variant where the new
#      inputs would still produce the old outputs under the original solver.
#
# b) Complexity calibration
#    - The legend uses exactly one symbol and one target color; there is only
#      one searchable foreground color in the field. This keeps inference local.
#    - The generator always includes at least one exact plus target and at least
#      one non-matching plus-like decoy, so the legend changes behavior in a
#      visible way across examples.
#    - Distractors are restricted to small clean components to keep the puzzle
#      legible while still providing repeated evidence that "same local center"
#      is not enough; the whole component shape must match.
#
# c) Hidden assumption audit for main()
#    - The old `colors[0]` assumption is no longer valid because the input now
#      intentionally contains two non-bg colors: yellow field objects and teal
#      legend cells. The new solver must separate "instruction color" from
#      "search color" using the corner region.
#    - The old per-pixel neighbor test assumed target membership can be decided
#      locally. That is false in this variant; larger components with a local
#      plus center are distractors and must survive unchanged.
#    - The new solver therefore assumes the corner sample is isolated inside the
#      top-left 3x3 window and that the searchable components use colors other
#      than the legend color. The generator enforces both conditions.
#
# d) Placement legality
#    - The generator reserves a 5x5 instruction zone in the top-left corner.
#    - All main-field components are placed with a 1-cell outer margin and are
#      forbidden from being 4-neighbor adjacent to any existing non-bg cells,
#      so components never merge and component matching stays well-defined.
#    - Exact plus components and plus-tail decoys are built from explicit shape
#      templates, so they never go out of bounds and always remain valid
#      4-connected objects.
#
# e) Visual observability
#    - From 2-4 examples, a human can see the teal corner symbol persists and
#      every other teal change copies that same symbol as a whole object.
#    - Because at least one yellow "plus with a tail" remains unchanged in each
#      sample, the rule cannot be misread as "paint any place with four arms."
#
# f) Visual cleanliness
#    - The corner legend has its own moat.
#    - Components are separated instead of piled into dense blobs, so the legend,
#      targets, and decoys remain distinguishable.
#
# g) Color-change inferability
#    - Only three colors appear: red background, yellow field objects, teal
#      legend/output. Teal is anchored by the corner legend, so the recolor cue
#      is explicit rather than hidden in color ids.
#
# i) New assumptions after modification
#    - main() assumes the legend occupies the top-left 3x3 window and forms one
#      component shape. generate_input() constructs exactly that.
#    - main() assumes target components are isolated 4-connected components of
#      a non-legend color. generate_input() enforces component separation.


PLUS_SHAPE: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 1),
)

RESERVED = 5


def _normalize_shape(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rows = [int(r) for r, _ in cells]
    cols = [int(c) for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _component_signature(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return _normalize_shape(tuple(cells))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _extract_corner_template(g: np.ndarray, bg: int) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    if g.shape[0] < 3 or g.shape[1] < 3:
        return None
    zone = g[:3, :3]
    colors = sorted(set(map(int, zone.flatten())) - {int(bg)})
    if len(colors) != 1:
        return None
    color = int(colors[0])
    cells = [tuple(map(int, coord)) for coord in np.argwhere(zone == color)]
    if not cells:
        return None
    return color, _component_signature(cells)


def _find_components(g: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(g[r, c]) != int(color):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the corner legend as a template and recolor only exact '+' copies.

    Input structure:
    - The background is the most common color.
    - The top-left 3x3 corner contains a small '+' example already drawn in the target color.
    - Elsewhere, one other foreground color forms isolated components: some are exact 5-cell '+' objects, some are larger plus-like distractors.

    Rule:
    - Read the corner symbol to obtain both the target shape and the recolor color.
    - Find 4-connected components in the other foreground color whose full component footprint exactly matches that legend shape.
    - Recolor every cell of each matching component to the legend color.
    - Leave the corner legend and every non-matching component unchanged.

    Key insight:
    - The corner '+' is an instruction, not another candidate; matching happens at the component level, so larger yellow objects that merely contain a local cross center are distractors.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    template = _extract_corner_template(g, bg)
    if template is None:
        return out
    target_color, template_shape = template

    colors = sorted(set(map(int, g.flatten())) - {bg, target_color})
    if not colors:
        return out

    for fg in colors:
        for comp in _find_components(g, int(fg)):
            if len(comp) != len(template_shape):
                continue
            if _component_signature(comp) != template_shape:
                continue
            for r, c in comp:
                out[r, c] = int(target_color)
    return out


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])
    H, W = g.shape
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(g[r, c]) != fg:
                continue
            if all(int(g[r + dr, c + dc]) == fg for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                    out[r + dr, c + dc] = 8
    return out


def _rotate_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rotated = [(-c, r) for r, c in shape]
    return _normalize_shape(tuple(rotated))


def _shape_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    variants: list[tuple[tuple[int, int], ...]] = []
    cur = _normalize_shape(shape)
    for _ in range(4):
        if cur not in variants:
            variants.append(cur)
        cur = _rotate_shape(cur)
    return variants


DECOY_VARIANTS = _shape_variants(PLUS_SHAPE + ((1, 3),))
DISTRACTOR_LIBRARY: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
)


def _make_reserved_mask(H: int, W: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mask[: min(RESERVED, H), : min(RESERVED, W)] = True
    return mask


def _can_place(grid: np.ndarray, shape: tuple[tuple[int, int], ...], top: int, left: int, reserved: np.ndarray, bg: int) -> bool:
    H, W = grid.shape
    cells = {(int(top + r), int(left + c)) for r, c in shape}
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if reserved[r, c] or int(grid[r, c]) != int(bg):
            return False
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in cells and int(grid[nr, nc]) != int(bg):
                return False
    return True


def _paint_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], top: int, left: int, color: int) -> list[tuple[int, int]]:
    placed: list[tuple[int, int]] = []
    for r, c in shape:
        rr = int(top + r)
        cc = int(left + c)
        grid[rr, cc] = int(color)
        placed.append((rr, cc))
    return placed


def _random_origin(rng: np.random.Generator, H: int, W: int, shape: tuple[tuple[int, int], ...]) -> tuple[int, int] | None:
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    if H - h < 2 or W - w < 2:
        return None
    top = int(rng.integers(1, H - h))
    left = int(rng.integers(1, W - w))
    return top, left


def _place_random_shape(
    rng: np.random.Generator,
    grid: np.ndarray,
    shapes: list[tuple[tuple[int, int], ...]] | tuple[tuple[tuple[int, int], ...], ...],
    color: int,
    reserved: np.ndarray,
    bg: int,
    attempts: int = 200,
) -> list[tuple[int, int]] | None:
    H, W = grid.shape
    for _ in range(attempts):
        shape = shapes[int(rng.integers(0, len(shapes)))]
        origin = _random_origin(rng, H, W, shape)
        if origin is None:
            return None
        top, left = origin
        if not _can_place(grid, shape, top, left, reserved, bg):
            continue
        return _paint_shape(grid, shape, top, left, color)
    return None


def _paint_corner_legend(grid: np.ndarray, bg: int, target: int) -> None:
    grid[: min(RESERVED, grid.shape[0]), : min(RESERVED, grid.shape[1])] = int(bg)
    for r, c in PLUS_SHAPE:
        grid[r, c] = int(target)


def _count_exact_plus_components(grid: np.ndarray, fg: int) -> int:
    count = 0
    for comp in _find_components(grid, fg):
        if len(comp) == len(PLUS_SHAPE) and _component_signature(comp) == PLUS_SHAPE:
            count += 1
    return count


def _count_local_plus_centers(grid: np.ndarray, fg: int) -> int:
    H, W = grid.shape
    total = 0
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != int(fg):
                continue
            if all(int(grid[r + dr, c + dc]) == int(fg) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                total += 1
    return total


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    bg, fg, target = 2, 4, 8

    grid_out = np.full((H, W), bg, dtype=int)
    reserved = _make_reserved_mask(H, W)
    _paint_corner_legend(grid_out, bg, target)

    area = H * W
    target_density = float(rng.uniform(0.18, 0.32))
    target_fg_cells = int(area * target_density)

    target_components: list[list[tuple[int, int]]] = []
    n_targets = int(rng.integers(1, min(7, max(2, area // 45)) + 1))
    for _ in range(n_targets):
        placed = _place_random_shape(rng, grid_out, [PLUS_SHAPE], target, reserved, bg, attempts=300)
        if placed is None:
            break
        target_components.append(placed)

    if not target_components:
        return None

    decoy_cells = 0
    n_decoys = int(rng.integers(1, min(5, max(2, area // 70)) + 1))
    for _ in range(n_decoys):
        placed = _place_random_shape(rng, grid_out, DECOY_VARIANTS, fg, reserved, bg, attempts=300)
        if placed is None:
            continue
        decoy_cells += len(placed)

    if decoy_cells == 0:
        return None

    current_fg = int(np.count_nonzero(grid_out != bg))
    distractor_placed = 0
    for _ in range(area * 8):
        if current_fg >= target_fg_cells:
            break
        placed = _place_random_shape(rng, grid_out, DISTRACTOR_LIBRARY, fg, reserved, bg, attempts=20)
        if placed is None:
            continue
        current_fg += len(placed)
        distractor_placed += 1

    if distractor_placed == 0:
        return None

    grid_in = grid_out.copy()
    for comp in target_components:
        for r, c in comp:
            grid_in[r, c] = int(fg)

    if int(np.count_nonzero(grid_in == bg)) <= area // 2:
        return None

    if _count_exact_plus_components(grid_in, fg) != len(target_components):
        return None
    if _count_local_plus_centers(grid_in, fg) <= len(target_components):
        return None

    expected = grid_out
    actual = main(grid_in)
    if not np.array_equal(actual, expected):
        return None

    legacy = _legacy_main(grid_in)
    if np.array_equal(legacy, expected):
        return None

    return grid_in


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
