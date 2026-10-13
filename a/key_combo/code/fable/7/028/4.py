from __future__ import annotations

import importlib.util
from itertools import permutations
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_fable_7_028_4", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC_UTILS = _load_arc_utils()
Color = _ARC_UTILS.Color
get_bg = _ARC_UTILS.get_bg


PANEL_POSITIONS = ("TL", "TR", "BL", "BR")
LAYOUTS = ((3, 4), (4, 3), (3, 5), (5, 3))
PAIR_FAMILIES = (
    frozenset({0, 1}),
    frozenset({1, 2}),
    frozenset({2, 3}),
    frozenset({0, 3}),
    frozenset({0, 2}),
    frozenset({1, 3}),
)
INPUT_PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

MOTIFS = {
    1: (
        np.array([[1, 0], [0, 0]], dtype=bool),
        np.array([[0, 1], [0, 0]], dtype=bool),
        np.array([[0, 0], [1, 0]], dtype=bool),
        np.array([[0, 0], [0, 1]], dtype=bool),
    ),
    2: (
        np.array([[1, 1], [0, 0]], dtype=bool),
        np.array([[0, 0], [1, 1]], dtype=bool),
        np.array([[1, 0], [1, 0]], dtype=bool),
        np.array([[0, 1], [0, 1]], dtype=bool),
    ),
    3: (
        np.array([[0, 1], [1, 1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1], [0, 1]], dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
    ),
    4: (
        np.array([[1, 1], [1, 1]], dtype=bool),
    ),
}


def _find_full_cross(grid: np.ndarray) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        rows = np.where(np.all(grid == color, axis=1))[0]
        cols = np.where(np.all(grid == color, axis=0))[0]
        for r in rows:
            for c in cols:
                if grid.shape[0] == 2 * int(r) + 1 and grid.shape[1] == 2 * int(c) + 1:
                    candidates.append((color, int(r), int(c)))
    assert len(candidates) == 1, "expected exactly one full cross"
    return candidates[0]


def _extract_panels(grid: np.ndarray) -> tuple[int, list[np.ndarray]]:
    cross_color, cross_r, cross_c = _find_full_cross(grid)
    tl = grid[:cross_r, :cross_c]
    tr = grid[:cross_r, cross_c + 1 :]
    bl = grid[cross_r + 1 :, :cross_c]
    br = grid[cross_r + 1 :, cross_c + 1 :]
    panels = [tl, tr, bl, br]
    assert len({panel.shape for panel in panels}) == 1, "panels must have equal size"
    return cross_color, panels


def _panel_bg_fg(panel: np.ndarray) -> tuple[int, int, int]:
    colors, counts = np.unique(panel, return_counts=True)
    assert len(colors) == 2, "each panel must contain exactly one bg and one fg color"
    order = np.argsort(counts)
    fg_idx = int(order[0])
    bg_idx = int(order[1])
    assert int(counts[fg_idx]) != int(counts[bg_idx]), "fg and bg counts must differ"
    return int(colors[bg_idx]), int(colors[fg_idx]), int(counts[fg_idx])


def main(grid: np.ndarray) -> np.ndarray:
    """Collapse the four cross-separated panels into a depth-colored overlap map.

    Input structure:
    - A full gray row and a full gray column split the grid into four equal panels.
    - Each panel contains exactly one sparse foreground color over one transparent background color.
    - The four foreground pixel counts are pairwise distinct.

    Rule:
    - Remove the cross and align the four panels by their local coordinates.
    - For each local cell, count the overlap depth j: how many panels show foreground there.
    - Paint depth 0 black; for depth j>=1, paint the foreground color of the panel whose total foreground area is the j-th smallest among the four panels.

    Key insight:
    - The output color at a cell depends only on overlap depth, but the depth-to-color palette is indexed by a separate global count ordering over the panels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    cross_color, panels = _extract_panels(g)
    panel_masks: list[np.ndarray] = []
    ranked_inputs: list[tuple[int, int]] = []

    for panel in panels:
        panel_colors = set(map(int, np.unique(panel)))
        assert cross_color not in panel_colors, "cross color may not appear inside a panel"
        bg_color, fg_color, fg_count = _panel_bg_fg(panel)
        assert bg_color != fg_color
        panel_masks.append(panel == fg_color)
        ranked_inputs.append((fg_count, fg_color))

    fg_counts = [count for count, _color in ranked_inputs]
    assert len(set(fg_counts)) == 4, "foreground areas must be pairwise distinct"
    rank_colors = [color for _count, color in sorted(ranked_inputs)]

    depth = np.zeros(panels[0].shape, dtype=int)
    for mask in panel_masks:
        depth += mask.astype(int)

    out = np.full(panels[0].shape, Color.BLACK, dtype=int)
    for j, color in enumerate(rank_colors, start=1):
        out[depth == j] = int(color)
    return out


def _family_list(rng: np.random.Generator) -> list[frozenset[int]]:
    pair_choices = rng.choice(len(PAIR_FAMILIES), size=2, replace=False)
    pairs = [PAIR_FAMILIES[int(i)] for i in pair_choices]
    return [
        frozenset({0, 1, 2, 3}),
        frozenset({1, 2, 3}),
        frozenset({0, 2, 3}),
        frozenset({0, 1, 3}),
        frozenset({0, 1, 2}),
        *pairs,
        frozenset({0}),
        frozenset({1}),
        frozenset({2}),
        frozenset({3}),
    ]


def _sample_area(rng: np.random.Generator, depth: int) -> int:
    if depth == 4:
        return int(rng.integers(2, 5))
    if depth == 3:
        return int(rng.integers(2, 5))
    if depth == 2:
        return int(rng.integers(1, 5))
    return int(rng.integers(1, 4))


def _choose_motif(rng: np.random.Generator, area: int) -> np.ndarray:
    motif_bank = MOTIFS[int(area)]
    return motif_bank[int(rng.integers(len(motif_bank)))]


def _slot_top_left(slot: tuple[int, int]) -> tuple[int, int]:
    slot_r, slot_c = slot
    return 3 * int(slot_r), 3 * int(slot_c)


def _actual_members(abs_members: frozenset[int], pos_to_abs: np.ndarray) -> frozenset[int]:
    members = {pos for pos, abs_panel in enumerate(pos_to_abs) if int(abs_panel) in abs_members}
    return frozenset(members)


def _oracle_from_spec(spec: dict) -> np.ndarray:
    depth = np.zeros(spec["panel_shape"], dtype=int)
    for mask in spec["panel_masks"]:
        depth += mask.astype(int)

    order = np.argsort(spec["fg_counts"])
    rank_colors = [int(spec["fg_colors"][int(idx)]) for idx in order]
    out = np.full(spec["panel_shape"], Color.BLACK, dtype=int)
    for j, color in enumerate(rank_colors, start=1):
        out[depth == j] = color
    return out


def _overlay_from_order(spec: dict, order: tuple[int, ...]) -> np.ndarray:
    out = np.full(spec["panel_shape"], Color.BLACK, dtype=int)
    for pos in order:
        mask = spec["panel_masks"][int(pos)]
        out[mask] = int(spec["fg_colors"][int(pos)])
    return out


def _build_case(rng: np.random.Generator) -> tuple[np.ndarray, dict] | None:
    layout_rows, layout_cols = LAYOUTS[int(rng.integers(len(LAYOUTS)))]
    panel_h = 3 * int(layout_rows) - 1
    panel_w = 3 * int(layout_cols) - 1
    total_slots = [(r, c) for r in range(layout_rows) for c in range(layout_cols)]

    families = _family_list(rng)
    if len(total_slots) < len(families):
        return None
    chosen_slot_ids = rng.choice(len(total_slots), size=len(families), replace=False)
    chosen_slots = [total_slots[int(idx)] for idx in chosen_slot_ids]

    abstract_masks = [np.zeros((panel_h, panel_w), dtype=bool) for _ in range(4)]
    atoms_abs: list[dict] = []
    for family, slot in zip(families, chosen_slots):
        area = _sample_area(rng, len(family))
        motif = _choose_motif(rng, area)
        r0, c0 = _slot_top_left(slot)
        atom_mask = np.zeros((panel_h, panel_w), dtype=bool)
        atom_mask[r0 : r0 + 2, c0 : c0 + 2] = motif
        atoms_abs.append({"members": family, "mask": atom_mask, "depth": len(family)})
        for abs_panel in family:
            abstract_masks[int(abs_panel)] |= atom_mask

    abstract_counts = [int(mask.sum()) for mask in abstract_masks]
    if len(set(abstract_counts)) != 4:
        return None
    if max(abstract_counts) >= (panel_h * panel_w) // 2:
        return None

    pos_to_abs = rng.permutation(4)
    panel_masks = [abstract_masks[int(abs_panel)].copy() for abs_panel in pos_to_abs]
    fg_counts = [int(mask.sum()) for mask in panel_masks]
    if len(set(fg_counts)) != 4:
        return None

    color_perm = rng.permutation(INPUT_PALETTE)
    bg_colors = [int(v) for v in color_perm[:4]]
    fg_colors = [int(v) for v in color_perm[4:]]

    depth = np.zeros((panel_h, panel_w), dtype=int)
    subset_by_depth: dict[int, set[frozenset[int]]] = {1: set(), 2: set(), 3: set(), 4: set()}
    for atom in atoms_abs:
        actual_family = _actual_members(atom["members"], pos_to_abs)
        subset_by_depth[len(actual_family)].add(actual_family)
        depth[atom["mask"]] = len(actual_family)

    if set(np.unique(depth)) != {0, 1, 2, 3, 4}:
        return None
    if len(subset_by_depth[1]) < 4 or len(subset_by_depth[3]) < 4 or len(subset_by_depth[2]) < 2:
        return None

    cross_r, cross_c = panel_h, panel_w
    grid = np.full((2 * panel_h + 1, 2 * panel_w + 1), Color.GREY, dtype=int)
    quadrant_slices = (
        (slice(0, panel_h), slice(0, panel_w)),
        (slice(0, panel_h), slice(panel_w + 1, 2 * panel_w + 1)),
        (slice(panel_h + 1, 2 * panel_h + 1), slice(0, panel_w)),
        (slice(panel_h + 1, 2 * panel_h + 1), slice(panel_w + 1, 2 * panel_w + 1)),
    )
    for pos, (rs, cs) in enumerate(quadrant_slices):
        panel = np.full((panel_h, panel_w), bg_colors[pos], dtype=int)
        panel[panel_masks[pos]] = fg_colors[pos]
        grid[rs, cs] = panel

    spec = {
        "panel_shape": (panel_h, panel_w),
        "panel_masks": panel_masks,
        "fg_colors": fg_colors,
        "bg_colors": bg_colors,
        "fg_counts": fg_counts,
        "depth": depth,
        "subset_by_depth": subset_by_depth,
    }
    oracle = _oracle_from_spec(spec)
    if not np.array_equal(main(grid), oracle):
        return None
    if any(np.array_equal(_overlay_from_order(spec, order), oracle) for order in permutations(range(4))):
        return None
    return grid, spec


def _generate_case() -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng()
    for _ in range(500):
        case = _build_case(rng)
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid depth-overlay case")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
