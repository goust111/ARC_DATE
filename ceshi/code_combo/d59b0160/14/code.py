# Combo variant of d59b0160 (index 14)
# combo_spec: D4/0+D10/0
# Name: 迭代删除+8-连通

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


# Design analysis for combo D4/0 + D10/0
# ---------------------------------------
# a) Concrete behavior trace:
#    - Original task: treat every non-background object as one 4-connected component and delete
#      it iff that component's visible color-set contains all legend colors.
#    - Fused task: use 8-connectivity instead, then repeat "find components -> delete matches"
#      until no more matches remain.
#    - Small trace:
#         Legend colors = {1,2,4}.
#         Outside the legend, place three 2x2 zero-filled blocks at offsets
#           A at (0,0), B at (2,2), C at (4,4) inside one local patch.
#         Put color 1 only inside A, color 2 only inside B, color 4 only inside C.
#         Under the original 4-connected rule, A/B/C are three separate components, each missing
#         two legend colors, so all survive.
#         Under the fused 8-connected rule, the corner touches make A+B+C one component, whose
#         union of colors is {0,1,2,4}, so the whole diagonal chain is erased.
#         After that deletion we recompute 8-components and stop when no match remains.
#    - Important structural note: because this task deletes whole components rather than partial
#      cells, the extra D10 fixed-point rounds are algebraically idempotent once the legend panel
#      is excluded from the candidate mask. The code still implements the requested fixed-point
#      loop exactly, but the generator makes D4/0 visibly active; D10/0 mainly changes solver
#      structure and the required reasoning procedure.
# b) Complexity calibration:
#    - To keep the difference visible but readable, each sample contains two diagonal "delete
#      clusters" built from 2-3 small rectangles and three well-separated kept rectangles.
#    - The delete clusters always rely on diagonal-only contact, so the 8-connected merge is a
#      repeated cue rather than a one-off accident.
#    - The number of patterns is deliberately small: 2 or 3 blocks per delete cluster, and
#      exactly three kept objects. This keeps the task inferable from a few examples.
# c) Hidden assumptions review:
#    - Original code labels `g != bg` directly. Under 4-connectivity that was safe because the
#      L-shaped legend border and its swatches stayed split into harmless fragments.
#    - Under 8-connectivity, legend swatches can merge with the border, so the legend must be
#      masked out explicitly; otherwise the specification panel itself becomes a candidate object.
#    - Original single-pass deletion assumes the connected-component partition is final after one
#      scan. The new solver must recompute components after each delete round even though, on this
#      object model, later rounds usually find nothing new.
# d) Placement legality:
#    - Every external object or multi-block cluster is first built as one local patch, then placed
#      with `random_free_location(..., padding=1, padding_connectivity=8, border=1)`.
#    - That guarantees no overlap, no out-of-bounds placement, and no accidental 8-connections
#      between unrelated objects or between objects and the legend.
# e) Visual observability:
#    - The diagonal chains are easy to see by eye: several small zero-base rectangles only touch
#      at corners. Their disappearance is exactly the cue that 8-connectivity matters.
#    - Using two delete clusters gives repetition, so the human solver can infer that corner-touch
#      alone is enough to join blocks into one deletable object.
# f) Visual cleanliness:
#    - The board stays uncluttered: one fixed legend, five external objects total, and at least a
#      one-cell moat between unrelated objects.
#    - Color changes remain local sprinkles inside rectangular zero regions, matching the visual
#      style of the source task.
# g) Change inferability:
#    - No behavior depends on raw color ids or hidden ordering. Colors only matter through visible
#      membership in the top-left legend palette.
#    - The generator rejects samples where the fused 8-connected solver matches an adapted 4-
#      connected baseline, so the produced data does not collapse back to the original rule.


GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3
MAX_RETRIES = 500

SAFE_LEGEND_PATTERNS = [
    [(0, 0), (0, 2), (2, 2)],
    [(0, 1), (2, 0), (2, 2)],
    [(1, 0), (0, 2), (2, 2)],
]

KEEPER_SIZES = [(3, 4), (4, 3), (2, 5), (5, 2), (3, 3)]
BLOCK_SIZES = [(2, 2), (2, 3), (3, 2)]


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Find the top-left legend whose bottom row and right column share one border color.

    Returns (size, border_color, legend_colors).
    """
    h, w = g.shape
    for sz in range(2, min(h, w) + 1):
        border_col = int(g[sz - 1, sz - 1])
        if border_col == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == border_col) and np.all(g[0:sz, sz - 1] == border_col):
            inside = g[0 : sz - 1, 0 : sz - 1]
            colors = set(map(int, np.unique(inside)))
            colors.discard(bg)
            colors.discard(border_col)
            return int(sz), int(border_col), colors
    raise AssertionError("no legend border found in grid")


def _legend_free_mask(shape: tuple[int, int], legend_size: int) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    mask[:legend_size, :legend_size] = False
    return mask


def _erase_matches(
    grid: np.ndarray,
    *,
    bg: int,
    legend_size: int,
    legend_colors: set[int],
    connectivity: int,
    iterative: bool,
) -> tuple[np.ndarray, int, int]:
    """Delete legend-matching components and return (output, rounds_with_deletions, deleted_cells)."""
    out = np.array(grid, dtype=int, copy=True)
    candidate_mask = _legend_free_mask(out.shape, legend_size)
    rounds = 0
    deleted_cells = 0

    while True:
        deleted_this_round = False
        comps = components_mask((out != bg) & candidate_mask, connectivity=connectivity)
        for pts in comps:
            cols = {int(out[r, c]) for r, c in pts}
            if legend_colors.issubset(cols):
                out[pts[:, 0], pts[:, 1]] = bg
                deleted_cells += int(len(pts))
                deleted_this_round = True
        if not deleted_this_round:
            break
        rounds += 1
        if not iterative:
            break

    return out, rounds, deleted_cells


def _baseline_main_4(grid: np.ndarray) -> np.ndarray:
    """Adapted original behavior: same legend masking, but 4-connectivity and one pass."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    legend_size, _border, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"
    out, _rounds, _deleted_cells = _erase_matches(
        g,
        bg=bg,
        legend_size=legend_size,
        legend_colors=legend_colors,
        connectivity=4,
        iterative=False,
    )
    return out


def _analyze_grid(grid: np.ndarray) -> dict[str, int]:
    """Generator-side observability metrics."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    legend_size, _border, legend_colors = _find_legend(g, bg)
    candidate_mask = _legend_free_mask(g.shape, legend_size)

    comps4 = components_mask((g != bg) & candidate_mask, connectivity=4)
    comps8 = components_mask((g != bg) & candidate_mask, connectivity=8)
    fused_out, rounds, deleted_cells = _erase_matches(
        g,
        bg=bg,
        legend_size=legend_size,
        legend_colors=legend_colors,
        connectivity=8,
        iterative=True,
    )
    base_out = _baseline_main_4(g)
    return {
        "candidate_components_4": int(len(comps4)),
        "candidate_components_8": int(len(comps8)),
        "merge_gap": int(len(comps4) - len(comps8)),
        "rounds": int(rounds),
        "deleted_cells": int(deleted_cells),
        "delta_vs_4": int(np.count_nonzero(fused_out != base_out)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase external 8-connected objects whose palette covers the legend, repeating until stable.

    Input structure:
    - The outer frame defines the background color and fills most of the board.
    - A 4x4 legend sits in the top-left corner: its bottom row and right column share one border
      color, and its 3x3 interior shows the target palette.
    - Elsewhere are rectangular non-background objects made mostly of color 0 with a few colored
      pixels sprinkled inside; some rectangles touch only diagonally.

    Rule:
    - Read the target color set from the legend interior.
    - Treat the legend square itself as a specification panel, not as a deletable object.
    - On the remaining grid, find 8-connected non-background components.
    - Delete every component whose color-set contains all legend colors, then recompute 8-
      connected components and repeat until a full pass finds no new match.

    Key insight:
    - The decisive change is that diagonal contact now fuses nearby rectangles into one object,
      so the legend colors can be distributed across several corner-touching blocks that vanish
      together as a single 8-connected component.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    legend_size, _border, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"
    out, _rounds, _deleted_cells = _erase_matches(
        g,
        bg=bg,
        legend_size=legend_size,
        legend_colors=legend_colors,
        connectivity=8,
        iterative=True,
    )
    return out


def _place_legend(grid: np.ndarray, targets: list[int]) -> None:
    """Draw a top-left legend whose 8-connected fragments still do not carry all three targets."""
    s = LEGEND_SIZE
    grid[:s, :s] = BG
    grid[s - 1, :s] = BORDER
    grid[:s, s - 1] = BORDER

    positions = list(random.choice(SAFE_LEGEND_PATTERNS))
    random.shuffle(positions)
    for color, (r, c) in zip(targets, positions):
        grid[r, c] = color


def _paint_rectangles(
    shape: tuple[int, int],
    rect_specs: list[tuple[int, int, int, int, list[int], list[int]]],
) -> np.ndarray:
    """
    Build a local patch on background `BG`.

    Each rectangle spec is:
      (r0, c0, h, w, forced_colors, allowed_extra_colors)
    The rectangle body is filled with color 0, then a few cells are recolored.
    """
    patch = np.full(shape, BG, dtype=int)
    for r0, c0, h, w, forced_colors, allowed_extra in rect_specs:
        patch[r0 : r0 + h, c0 : c0 + w] = 0
        coords = [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]
        random.shuffle(coords)

        forced = list(forced_colors)
        for color, (rr, cc) in zip(forced, coords):
            patch[rr, cc] = int(color)

        remaining = coords[len(forced) :]
        max_extra = min(len(remaining), max(0, (h * w) // 2 - len(forced)))
        extra_count = random.randint(0, max_extra) if max_extra > 0 else 0
        for rr, cc in remaining[:extra_count]:
            patch[rr, cc] = int(random.choice(allowed_extra))
    return patch


def _delete_cluster_patch(targets: list[int], block_count: int) -> np.ndarray:
    """Build a diagonal chain of 2 or 3 rectangles that is deletable only under 8-connectivity."""
    dims = [random.choice(BLOCK_SIZES) for _ in range(block_count)]
    rects: list[tuple[int, int, int, int]] = []
    r = 0
    c = 0
    for h, w in dims:
        rects.append((r, c, h, w))
        r += h
        c += w

    shape = (r, c)
    non_targets = [color for color in PALETTE if color not in targets]
    perm = random.sample(targets, len(targets))
    if block_count == 3:
        assignments = [[perm[0]], [perm[1]], [perm[2]]]
    else:
        assignments = [[perm[0], perm[1]], [perm[2]]]

    specs: list[tuple[int, int, int, int, list[int], list[int]]] = []
    for (r0, c0, h, w), forced in zip(rects, assignments):
        extras = list(forced)
        if non_targets:
            extras.append(random.choice(non_targets))
        specs.append((r0, c0, h, w, forced, extras))
    return _paint_rectangles(shape, specs)


def _keeper_patch(targets: list[int]) -> np.ndarray:
    """Build one separated rectangle that never contains all legend colors."""
    h, w = random.choice(KEEPER_SIZES)
    non_targets = [color for color in PALETTE if color not in targets]
    used_targets = random.sample(targets, random.randint(0, NUM_TARGETS - 1))
    allowed = list(non_targets) + list(used_targets)
    if not allowed:
        allowed = list(non_targets) if non_targets else [targets[0]]
    forced = random.sample(allowed, min(len(allowed), random.randint(1, min(3, len(allowed)))))
    return _paint_rectangles(
        (h, w),
        [(0, 0, h, w, forced, allowed)],
    )


def _place_patch(grid: np.ndarray, patch: np.ndarray) -> bool:
    """Place a local patch with a one-cell 8-neighborhood moat."""
    try:
        r, c = random_free_location(
            grid,
            patch,
            transparent=BG,
            border=1,
            padding=1,
            padding_connectivity=8,
        )
    except ValueError:
        return False
    paste(grid, patch, r, c, transparent=BG)
    return True


def _gen_attempt() -> np.ndarray | None:
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(grid, targets)

    patches = [
        _delete_cluster_patch(targets, 3),
        _delete_cluster_patch(targets, 2),
        _keeper_patch(targets),
        _keeper_patch(targets),
        _keeper_patch(targets),
    ]
    random.shuffle(patches)

    for patch in patches:
        if not _place_patch(grid, patch):
            return None

    metrics = _analyze_grid(grid)
    if metrics["delta_vs_4"] <= 0:
        return None
    if metrics["merge_gap"] <= 0:
        return None
    if metrics["deleted_cells"] < 10:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
