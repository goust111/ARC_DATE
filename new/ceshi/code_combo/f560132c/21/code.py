# Combo variant of f560132c (index 21)
# combo_spec: D1/4+D5/3+D14/1
# Name: 洞数匹配+洞数编码图例+颜色映射表

import itertools
import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location

# Design analysis for this combo variant
# --------------------------------------
# a) Concrete behavior trace
#    - Original family: four scattered pieces were reassembled, and a legend piece gave colors by spatial quadrant.
#      Shape geometry drove the placement-color association.
#    - Here the output is still a 2x2 rectangular composite, but the placement key is no longer shape.
#      Each monochrome piece has a unique hole count 0..3.
#      The separate legend is a 2x2 table of four entries:
#        1. each entry has its own hole count 0..3
#        2. each entry also shows a repeated vertical pair [source_color, target_color]
#      A piece is placed by matching its hole count to the legend entry with the same hole count, then using that
#      entry's physical quadrant in the legend table as the output slot. The piece is recolored by looking up its
#      source color anywhere in the table and reading the paired target color.
#    - Small trace:
#      suppose a red piece has 2 holes. If the only legend entry with 2 holes sits at top-right, the red piece must
#      be placed in the output top-right slot, even if the red->target pair is stored in a different legend entry.
#      If the red->yellow pair lives in the bottom-left legend entry, the red piece still becomes yellow after being
#      placed top-right. Placement and recoloring now come from two different reads of the same legend.
# b) Complexity calibration
#    - The hole code uses only four values (0,1,2,3), matching the four output quadrants exactly.
#    - All four pieces share the same outer square bbox size, so shape is intentionally unhelpful and the change from
#      "shape matching" to "hole counting" is visually obvious.
#    - The generator enforces two full derangements:
#        1. hole counts are not stored in their own physical legend quadrants
#        2. a piece's source->target entry is not stored in the same legend slot as its hole-count entry
#      This prevents collapse into either the original spatial-color rule or a pure color-table-only rule.
# c) Hidden assumptions from the original solver that break here
#    - Original code assumed one connected component doubled as both a piece and the legend. That no longer holds:
#      there are now four monochrome pieces plus one separate multi-color legend component.
#    - Original `_extract_legend_map(...)` assumed each legend cell was itself the target color. Here each legend
#      entry must be decoded into two attributes: hole count and source->target pair.
#    - Original assembly logic assumed geometry or centroid position determined which legend color applied.
#      In this combo, placement is driven by hole-count matching to a legend slot, while recoloring is driven by
#      source-color lookup in the mapping table.
#    - Original exact-cover search is unnecessary because the generated pieces deliberately have equal bbox size and
#      are composed into a fixed 2x2 rectangle; the nontrivial inference is symbolic, not geometric.
# d) Placement legality
#    - The generator builds the output first, then scatters the four piece patches and the legend patch on a larger
#      canvas with bbox padding via `random_free_location`, so components never overlap or leave the canvas.
#    - Legend holes are internal black cells inside fixed 5x5 entry windows. Piece holes are also internal black
#      cells. Both are counted as genuine enclosed background regions by the solver.
# e) Visual observability
#    - Human-solvers can see three stable cues across examples:
#        1. pieces differ by number of interior holes
#        2. legend entries also differ by hole count and occupy visible 2x2 positions
#        3. legend entries contain repeated [src, tgt] color bars
#      From a few examples one can infer: hole count chooses slot; source color chooses final color.
# f) Visual cleanliness
#    - Each legend entry lives in its own 5x5 window inside an 11x11 block with a one-cell separator cross.
#      That keeps hole-coding and color-pair coding readable without stacking too many marks into the same cells.
#    - Pieces are simple monochrome square-like sprites with black holes, so their counting cue remains salient.
# g) Color-change inferability
#    - There are exactly four source colors and four target colors, all distinct and all visible in the legend.
#    - The source/target pair is repeated vertically, which makes the table-like intent visible instead of looking
#      like arbitrary colored noise.
# h) New solver assumptions and generator guarantees
#    - There is exactly one legend component, uniquely identified by having the most distinct non-background colors.
#    - Every non-legend piece is monochrome except for black holes, and the four piece hole counts are exactly 0..3.
#    - The legend bbox is exactly 11x11 and contains four 5x5 entry windows in a 2x2 arrangement.
#    - Each entry window has one repeated source stripe and one repeated target stripe, and the four hole counts are
#      all distinct. The generator enforces these conditions directly.

K = 2
PIECE_SIZE = 5
ENTRY_SIZE = 5
LEGEND_GAP = 1
LEGEND_SIZE = ENTRY_SIZE * K + LEGEND_GAP
INPUT_SIZE_LO = 27
INPUT_SIZE_HI = 35
DENSITY_LO = 0.16
DENSITY_HI = 0.24
SCATTER_BORDER = 1
SCATTER_PADDING = 1
GEN_RETRIES = 300

ENTRY_OFFSETS = {
    (0, 0): (0, 0),
    (0, 1): (0, ENTRY_SIZE + LEGEND_GAP),
    (1, 0): (ENTRY_SIZE + LEGEND_GAP, 0),
    (1, 1): (ENTRY_SIZE + LEGEND_GAP, ENTRY_SIZE + LEGEND_GAP),
}
SLOT_ORDER = [(0, 0), (0, 1), (1, 0), (1, 1)]
HOLE_POSITIONS = [(1, 1), (1, 3), (3, 1)]


def _count_holes(mask: np.ndarray) -> int:
    pts = np.argwhere(mask)
    assert len(pts) > 0, "mask must be non-empty"
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)
    empty = ~sub
    if not empty.any():
        return 0
    holes = 0
    for comp in components_mask(empty, connectivity=4):
        rs = comp[:, 0]
        cs = comp[:, 1]
        touches = (
            np.any(rs == 0)
            or np.any(rs == sub.shape[0] - 1)
            or np.any(cs == 0)
            or np.any(cs == sub.shape[1] - 1)
        )
        if not touches:
            holes += 1
    return int(holes)


def _piece_body_color(g: np.ndarray, pts: np.ndarray, bg: int) -> int:
    counts = Counter(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
    assert counts, "piece must contain at least one non-background color"
    return int(counts.most_common(1)[0][0])


def _crop_component(g: np.ndarray, pts: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(pts)
    return np.asarray(g[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _decode_legend(legend_patch: np.ndarray, bg: int) -> tuple[dict[int, tuple[int, int]], dict[int, int]]:
    assert legend_patch.shape == (LEGEND_SIZE, LEGEND_SIZE), "legend bbox must match the fixed table size"
    hole_to_slot: dict[int, tuple[int, int]] = {}
    source_to_target: dict[int, int] = {}

    for slot in SLOT_ORDER:
        r0, c0 = ENTRY_OFFSETS[slot]
        entry = np.asarray(legend_patch[r0 : r0 + ENTRY_SIZE, c0 : c0 + ENTRY_SIZE], dtype=int)
        hole_count = _count_holes(entry != bg)

        src_col = set(map(int, entry[3:5, 2]))
        tgt_col = set(map(int, entry[3:5, 3]))
        assert len(src_col) == 1 and len(tgt_col) == 1, "legend source/target stripes must be vertically repeated"
        src = int(next(iter(src_col)))
        tgt = int(next(iter(tgt_col)))
        assert src != bg and tgt != bg and src != tgt, "legend mapping colors must be distinct non-background colors"

        hole_to_slot[hole_count] = slot
        source_to_target[src] = tgt

    assert set(hole_to_slot.keys()) == {0, 1, 2, 3}, "legend must encode each hole count exactly once"
    assert len(source_to_target) == 4 and len(set(source_to_target.values())) == 4, "legend mapping must be bijective"
    return hole_to_slot, source_to_target


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rearrange scattered hole-coded sprites into a 2x2 rectangle, using the legend for both slot selection and recoloring.

    Input structure:
    - Background is the most common color.
    - Four monochrome piece sprites are scattered separately; each piece may contain 0-3 enclosed black holes.
    - A separate multi-color legend component is an 11x11 block containing four 5x5 entries arranged as a 2x2 table.
      Inside each legend entry, the hole count is a position code and a repeated source/target color pair forms one
      row of the recoloring table.

    Rule:
    - Count holes in each piece.
    - In the legend, find which physical table slot contains each hole count 0..3; that slot is where the matching
      piece must go in the output rectangle.
    - Independently decode the source->target color pairs from the legend.
    - Place each cropped piece into its matched output slot and recolor its body cells to the target color associated
      with that piece's original source color. Preserve black holes.

    Key insight: The legend entries carry two orthogonal signals at once: hole count tells you where a piece belongs,
    while the source/target pair tells you what color that piece becomes. Neither cue alone is sufficient.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    assert len(comps) == 5, "input must contain four pieces plus one legend"

    color_sets = []
    for pts in comps:
        color_sets.append(set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg))

    max_distinct = max(len(cols) for cols in color_sets)
    legend_candidates = [i for i, cols in enumerate(color_sets) if len(cols) == max_distinct]
    assert len(legend_candidates) == 1, "legend must be uniquely identified by color diversity"
    legend_idx = legend_candidates[0]

    legend_patch = _crop_component(g, comps[legend_idx])
    hole_to_slot, source_to_target = _decode_legend(legend_patch, bg)

    piece_info = []
    for i, pts in enumerate(comps):
        if i == legend_idx:
            continue
        patch = _crop_component(g, pts)
        body = _piece_body_color(g, pts, bg)
        holes = _count_holes(patch != bg)
        piece_info.append({"patch": patch, "body": body, "holes": holes, "shape": patch.shape})

    assert len(piece_info) == 4, "must have exactly four non-legend pieces"
    assert {p["holes"] for p in piece_info} == {0, 1, 2, 3}, "piece hole counts must be 0..3 exactly once"
    shapes = {tuple(map(int, p["shape"])) for p in piece_info}
    assert len(shapes) == 1, "all pieces must share the same cropped size"
    piece_h, piece_w = next(iter(shapes))

    out = np.zeros((piece_h * K, piece_w * K), dtype=int)
    for p in piece_info:
        slot_r, slot_c = hole_to_slot[int(p["holes"])]
        tgt = int(source_to_target[int(p["body"])])
        recolored = np.asarray(p["patch"], dtype=int).copy()
        recolored[recolored == int(p["body"])] = tgt
        paste(out, recolored, slot_r * piece_h, slot_c * piece_w, transparent=bg)

    return out


def _piece_mask(hole_count: int) -> np.ndarray:
    mask = np.ones((PIECE_SIZE, PIECE_SIZE), dtype=bool)
    for r, c in HOLE_POSITIONS[:hole_count]:
        mask[r, c] = False
    assert _count_holes(mask) == hole_count, "piece template must realize the requested hole count"
    return mask


def _random_d4(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=int)
    if random.random() < 0.5:
        out = np.fliplr(out)
    k = random.randint(0, 3)
    out = np.rot90(out, k)
    return np.asarray(out, dtype=int).copy()


def _entry_patch(hole_count: int, src: int, tgt: int, base: int) -> np.ndarray:
    patch = np.full((ENTRY_SIZE, ENTRY_SIZE), int(base), dtype=int)
    for r, c in HOLE_POSITIONS[:hole_count]:
        patch[r, c] = 0
    patch[3:5, 2] = int(src)
    patch[3:5, 3] = int(tgt)
    assert _count_holes(patch != 0) == hole_count, "legend entry must preserve its hole code"
    return patch


def _build_legend(
    hole_at_slot: dict[int, int],
    pair_at_slot: dict[int, tuple[int, int]],
    base: int,
) -> np.ndarray:
    legend = np.full((LEGEND_SIZE, LEGEND_SIZE), int(base), dtype=int)
    for slot_idx, slot in enumerate(SLOT_ORDER):
        r0, c0 = ENTRY_OFFSETS[slot]
        hole_count = int(hole_at_slot[slot_idx])
        src, tgt = pair_at_slot[slot_idx]
        legend[r0 : r0 + ENTRY_SIZE, c0 : c0 + ENTRY_SIZE] = _entry_patch(hole_count, src, tgt, base)
    return legend


def _scatter_components(patches: list[np.ndarray]) -> np.ndarray | None:
    total_nz = sum(int(np.count_nonzero(p)) for p in patches)
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total_nz / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total_nz / DENSITY_LO)))
    if nlo > nhi:
        return None

    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    order = list(range(len(patches)))
    random.shuffle(order)
    for idx in order:
        patch = np.asarray(patches[idx], dtype=int)
        box = np.ones(patch.shape, dtype=int)
        try:
            r0, c0 = random_free_location(
                occupied,
                box,
                transparent=0,
                border=SCATTER_BORDER,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = 1
        paste(grid, patch, r0, c0, transparent=0)
    return grid


def _derangements(items: list[int]) -> list[tuple[int, ...]]:
    out = []
    for perm in itertools.permutations(items):
        if all(int(a) != int(b) for a, b in zip(items, perm)):
            out.append(tuple(map(int, perm)))
    return out


def _variant_metrics(slot_for_hole: dict[int, int], slot_for_source: dict[int, int]) -> dict[str, int]:
    position_deranged = sum(int(slot_for_hole[h] != h) for h in range(4))
    cross_split = sum(int(slot_for_source[h] != slot_for_hole[h]) for h in range(4))
    return {
        "position_deranged": int(position_deranged),
        "cross_split": int(cross_split),
    }


def _gen_attempt() -> np.ndarray | None:
    colors = random.sample(range(1, 10), 9)
    piece_sources = colors[:4]
    piece_targets = colors[4:8]
    legend_base = colors[8]

    piece_patches: list[np.ndarray] = []
    expected = np.zeros((PIECE_SIZE * K, PIECE_SIZE * K), dtype=int)

    slot_indices = [0, 1, 2, 3]
    hole_perm = random.choice(_derangements(slot_indices))
    slot_for_hole = {hole: int(hole_perm[hole]) for hole in range(4)}

    valid_source_perms = []
    for perm in itertools.permutations(slot_indices):
        ok = True
        for hole in range(4):
            if int(perm[hole]) == int(slot_for_hole[hole]):
                ok = False
                break
        if ok:
            valid_source_perms.append(tuple(map(int, perm)))
    if not valid_source_perms:
        return None
    src_perm = random.choice(valid_source_perms)
    slot_for_source = {hole: int(src_perm[hole]) for hole in range(4)}

    hole_at_slot: dict[int, int] = {}
    for hole, slot_idx in slot_for_hole.items():
        hole_at_slot[int(slot_idx)] = int(hole)
    pair_at_slot: dict[int, tuple[int, int]] = {}
    for hole, slot_idx in slot_for_source.items():
        pair_at_slot[int(slot_idx)] = (int(piece_sources[hole]), int(piece_targets[hole]))

    legend_patch = _build_legend(hole_at_slot, pair_at_slot, legend_base)

    for hole in range(4):
        mask = _piece_mask(hole)
        oriented_mask = _random_d4(mask.astype(int)).astype(bool)
        patch = np.zeros((PIECE_SIZE, PIECE_SIZE), dtype=int)
        patch[oriented_mask] = int(piece_sources[hole])
        piece_patches.append(patch)

        slot_idx = int(slot_for_hole[hole])
        slot_r, slot_c = SLOT_ORDER[slot_idx]
        recolored = patch.copy()
        recolored[recolored == int(piece_sources[hole])] = int(piece_targets[hole])
        expected[
            slot_r * PIECE_SIZE : (slot_r + 1) * PIECE_SIZE,
            slot_c * PIECE_SIZE : (slot_c + 1) * PIECE_SIZE,
        ] = recolored

    metrics = _variant_metrics(slot_for_hole, slot_for_source)
    if metrics["position_deranged"] != 4 or metrics["cross_split"] != 4:
        return None

    grid = _scatter_components(piece_patches + [legend_patch])
    if grid is None:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate combo variant input")
