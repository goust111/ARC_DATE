# Combo variant of d59b0160 (index 31)
# combo_spec: D1/0+D6/1+D10/0
# Name: 迭代删除+ANY匹配+部分删除

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import components_mask, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior / trace:
#    Keep the original scene grammar: a 4x4 legend in the top-left corner, orange
#    background, and several separated non-background objects elsewhere. The fusion
#    changes three linked decisions at once:
#      - matching uses ANY legend overlap, not ALL;
#      - a matched component loses only its legend-colored pixels;
#      - processing is phrased as a fixed-point loop that re-labels components after
#        each deletion round.
#    Small trace with legend colors {1,4,8}:
#      - Input object A is a barbell made from black 0-cells plus grey accents, with
#        a 1-colored bridge connecting the two lobes. Under the original task A is
#        kept because it does not contain all three legend colors. Under this combo,
#        A matches immediately because it contains color 1, but only the bridge cells
#        colored 1 are erased. The surviving non-target lobes remain and split into
#        two components. A later pass recomputes components on that changed grid and
#        confirms the process has stabilized.
#      - Input object B uses only non-target colors, so it never matches and stays.
#    The visible difference from the base task is therefore "partial bite marks plus
#    fragmentation", not whole-object disappearance.
# b) Complexity calibration:
#    The generator keeps a single legend, the same grid size, and 4-6 objects so the
#    task still reads like d59b0160. To make the ANY rule inferable, every sample has
#    at least one deleted object carrying only 1-2 legend colors; to make the partial
#    deletion obvious, at least one deleted object uses a legend-colored bridge whose
#    removal splits the object into multiple survivors.
# c) Hidden-assumption audit:
#    - The original solver labeled all non-background components, including the
#      legend. With ANY matching that would wrongly erase the legend swatches, so the
#      legend square must be removed from the candidate mask explicitly.
#    - Component membership can change after partial deletion, so connected components
#      must be recomputed from the current grid state, not cached from the input.
#    - Matching must also be read from the current grid state; otherwise the solver
#      would keep consulting colors that have already been erased.
#    - Under the literal D1/0 + D6/1 semantics, a matched component loses all of its
#      legend-colored pixels the first time it is hit, so later passes usually do not
#      delete extra pixels. D10 still matters as a fixed-point wrapper around the
#      changing component structure, and the generator forces post-deletion splitting
#      so this re-segmentation step is genuinely exercised rather than dead code.
# d) Placement legality:
#    Each patch is placed with random_free_location(..., padding=1) while reserving
#    its whole bounding box. Objects therefore stay in bounds, do not overlap, and
#    keep a one-cell gap from the legend and from each other.
# e) Visual observability:
#    A solver can see repeated evidence that "sharing even one legend color is enough"
#    because deleted objects often show only one or two legend colors. The output also
#    preserves non-target structure, so the change reads as selective erosion instead
#    of blanket color filtering or full deletion.
# f) Visual cleanliness:
#    Objects are still sparse axis-aligned shapes with limited palettes. Using one
#    bridge-split object per sample creates a strong cue without filling the board
#    with tangled fragments.
# g) Inference / new solver assumptions:
#    The new main() assumes the legend remains confined to the top-left 4x4 square and
#    that non-legend objects are separated initially. The generator enforces both, and
#    it also guarantees at least one ANY-only deleted object and at least one object
#    whose survivors split after deletion, so the intended fusion behavior is visible.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Detect the smallest top-left square whose bottom edge and right edge share one
    non-background color, then read the interior legend colors from that square.
    """
    h, w = g.shape
    for sz in range(2, min(h, w) + 1):
        x = int(g[sz - 1, sz - 1])
        if x == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == x) and np.all(g[0:sz, sz - 1] == x):
            inside = g[0 : sz - 1, 0 : sz - 1]
            colors = set(map(int, np.unique(inside)))
            colors.discard(bg)
            colors.discard(x)
            return int(sz), int(x), colors
    raise AssertionError("no legend border found in grid")


def _nonlegend_mask(g: np.ndarray, bg: int, legend_size: int) -> np.ndarray:
    mask = (g != bg).copy()
    mask[:legend_size, :legend_size] = False
    return mask


def _apply_partial_any_round(
    g: np.ndarray,
    bg: int,
    legend_size: int,
    legend_colors: set[int],
) -> bool:
    """
    Apply one current-grid round:
    - label non-legend components,
    - match any component sharing at least one legend color,
    - erase only the legend-colored pixels inside matched components.
    """
    changed = False
    comps = components_mask(_nonlegend_mask(g, bg, legend_size), connectivity=4)
    for pts in comps:
        cols = {int(g[r, c]) for r, c in pts}
        if not (cols & legend_colors):
            continue
        for r, c in pts:
            if int(g[r, c]) in legend_colors:
                g[r, c] = bg
                changed = True
    return changed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Iteratively punch out legend-colored pixels from any component that overlaps the legend palette.

    Input structure:
    - A dominant background fills the board.
    - The top-left corner contains one small legend square whose bottom and right
      edges share a border color; the interior swatches define the target palette.
    - Away from the legend, the remaining non-background cells form separate
      4-connected multicolor objects.

    Rule:
    - Read the legend color set from inside the corner legend and ignore the legend
      square itself when searching for candidate objects.
    - On the current grid, find every non-legend 4-connected component that contains
      **any** legend color.
    - For each matched component, erase only the pixels whose own color belongs to
      the legend set; keep all non-legend-colored pixels of that component.
    - Recompute connected components on the updated grid and repeat until a full pass
      makes no further deletions.

    Key insight:
    - The legend is an overlap-triggered erosion rule rather than a full-palette test:
      even one shared legend color is enough to mark a component, but the deletion is
      selective, so objects are bitten apart instead of disappearing wholesale.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    legend_size, _border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    while _apply_partial_any_round(out, bg, legend_size, legend_colors):
        pass
    return out


# --------------- generator (o5x) ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 3, 6]
NUM_KEPT = 3
MAX_RETRIES = 500

KEEP_SHAPE_WEIGHTS = [4, 2, 2]  # medium, tall, wide
TALL_H = (7, 11)
TALL_W = (2, 3)
WIDE_H = (2, 4)
WIDE_W = (6, 10)
MED_H = (3, 6)
MED_W = (3, 7)


def _place_legend(inp: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend box at top-left and place the 3 target colors."""
    s = LEGEND_SIZE
    inp[:s, :s] = BG
    inp[s - 1, :s] = BORDER
    inp[:s, s - 1] = BORDER

    isolated = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    border_side = [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    side_choices = [
        p for p in border_side
        if abs(p[0] - isolated[0]) + abs(p[1] - isolated[1]) != 1
    ]
    positions = [isolated] + random.sample(side_choices, 2)
    random.shuffle(positions)
    for color, (r, c) in zip(targets, positions):
        inp[r, c] = color


def _rect_dims() -> tuple[int, int]:
    mode = random.choices(["medium", "tall", "wide"], weights=KEEP_SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _occupied_coords(patch: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(patch != BG)]


def _add_accents(
    patch: np.ndarray,
    allowed_colors: list[int],
    *,
    min_accents: int,
    max_accents: int,
) -> None:
    coords = _occupied_coords(patch)
    if not coords or not allowed_colors:
        return
    n_accents = random.randint(min_accents, min(max_accents, len(coords)))
    for r, c in random.sample(coords, n_accents):
        patch[r, c] = random.choice(allowed_colors)


def _build_keep_patch(non_targets: list[int]) -> np.ndarray:
    h, w = _rect_dims()
    patch = np.zeros((h, w), dtype=int)
    _add_accents(
        patch,
        non_targets,
        min_accents=1,
        max_accents=max(1, min(len(_occupied_coords(patch)) // 3, 6)),
    )
    return patch


def _horizontal_split_patch(targets: list[int], non_targets: list[int]) -> np.ndarray:
    h = random.randint(3, 5)
    left_w = random.randint(2, 3)
    bridge_len = random.randint(2, 4)
    right_w = random.randint(2, 3)
    patch = np.full((h, left_w + bridge_len + right_w), BG, dtype=int)

    patch[:, :left_w] = 0
    patch[:, left_w + bridge_len :] = 0

    bridge_row = random.randint(1, h - 2) if h >= 3 else 0
    chosen_targets = random.sample(targets, random.randint(1, min(2, len(targets))))
    for idx, c in enumerate(range(left_w, left_w + bridge_len)):
        patch[bridge_row, c] = chosen_targets[idx % len(chosen_targets)]

    _add_accents(patch[:, :left_w], non_targets, min_accents=1, max_accents=min(3, h))
    _add_accents(
        patch[:, left_w + bridge_len :],
        non_targets,
        min_accents=1,
        max_accents=min(3, h),
    )
    return patch


def _vertical_split_patch(targets: list[int], non_targets: list[int]) -> np.ndarray:
    top_h = random.randint(2, 3)
    bridge_len = random.randint(2, 4)
    bottom_h = random.randint(2, 3)
    w = random.randint(3, 5)
    patch = np.full((top_h + bridge_len + bottom_h, w), BG, dtype=int)

    patch[:top_h, :] = 0
    patch[top_h + bridge_len :, :] = 0

    bridge_col = random.randint(1, w - 2) if w >= 3 else 0
    chosen_targets = random.sample(targets, random.randint(1, min(2, len(targets))))
    for idx, r in enumerate(range(top_h, top_h + bridge_len)):
        patch[r, bridge_col] = chosen_targets[idx % len(chosen_targets)]

    _add_accents(patch[:top_h, :], non_targets, min_accents=1, max_accents=min(3, w))
    _add_accents(
        patch[top_h + bridge_len :, :],
        non_targets,
        min_accents=1,
        max_accents=min(3, w),
    )
    return patch


def _build_split_delete_patch(targets: list[int], non_targets: list[int]) -> np.ndarray:
    if random.random() < 0.6:
        return _horizontal_split_patch(targets, non_targets)
    return _vertical_split_patch(targets, non_targets)


def _build_perforated_delete_patch(targets: list[int], non_targets: list[int]) -> np.ndarray:
    h, w = _rect_dims()
    patch = np.zeros((h, w), dtype=int)
    coords = _occupied_coords(patch)

    target_count = random.randint(1, min(2, len(targets)))
    chosen_targets = random.sample(targets, target_count)
    forced_colors = list(chosen_targets) + [random.choice(non_targets)]

    n_extra = random.randint(0, max(2, len(coords) // 4))
    color_pool = chosen_targets * 3 + non_targets
    colors = forced_colors + [random.choice(color_pool) for _ in range(n_extra)]
    random.shuffle(colors)

    chosen_coords = random.sample(coords, len(colors))
    for (r, c), color in zip(chosen_coords, colors):
        patch[r, c] = color
    return patch


def _component_roles(component_count: int) -> list[str]:
    deleted_count = component_count - NUM_KEPT
    deleted_roles = ["split_delete"]
    for _ in range(deleted_count - 1):
        deleted_roles.append(random.choices(["split_delete", "perforated_delete"], weights=[2, 1])[0])
    roles = deleted_roles + ["keep"] * NUM_KEPT
    random.shuffle(roles)
    return roles


def _component_patch(role: str, targets: list[int], non_targets: list[int]) -> np.ndarray:
    if role == "keep":
        return _build_keep_patch(non_targets)
    if role == "split_delete":
        return _build_split_delete_patch(targets, non_targets)
    if role == "perforated_delete":
        return _build_perforated_delete_patch(targets, non_targets)
    raise ValueError(f"unknown role: {role}")


def _survivor_piece_count(inp: np.ndarray, out: np.ndarray, pts: np.ndarray) -> int:
    mask = np.zeros_like(inp, dtype=bool)
    for r, c in pts:
        if int(out[r, c]) != BG:
            mask[r, c] = True
    return len(components_mask(mask, connectivity=4))


def _behavior_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    bg = _bg_boundary(inp)
    legend_size, _border, legend_colors = _find_legend(inp, bg)
    comps = components_mask(_nonlegend_mask(inp, bg, legend_size), connectivity=4)

    matched_components = 0
    partial_hit_deleted = 0
    split_deleted = 0
    changed_components = 0
    for pts in comps:
        cols = {int(inp[r, c]) for r, c in pts}
        if not (cols & legend_colors):
            continue
        matched_components += 1
        if not legend_colors.issubset(cols):
            partial_hit_deleted += 1
        if any(int(out[r, c]) != int(inp[r, c]) for r, c in pts):
            changed_components += 1
        if _survivor_piece_count(inp, out, pts) >= 2:
            split_deleted += 1

    return {
        "matched_components": matched_components,
        "partial_hit_deleted": partial_hit_deleted,
        "split_deleted": split_deleted,
        "changed_components": changed_components,
    }


def _gen_attempt() -> np.ndarray | None:
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    targets = random.sample(PALETTE, NUM_TARGETS)
    non_targets = [v for v in PALETTE if v not in targets]
    _place_legend(inp, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    for role in _component_roles(component_count):
        patch = _component_patch(role, targets, non_targets)
        try:
            r, c = random_free_location(
                inp,
                np.zeros_like(patch),
                transparent=BG,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, patch, r, c, transparent=BG)

    out = main(inp)
    metrics = _behavior_metrics(inp, out)
    if metrics["matched_components"] < 1:
        return None
    if metrics["partial_hit_deleted"] < 1:
        return None
    if metrics["split_deleted"] < 1:
        return None
    if metrics["changed_components"] != metrics["matched_components"]:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
