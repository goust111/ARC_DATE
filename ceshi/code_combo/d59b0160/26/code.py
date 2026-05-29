# Combo variant of d59b0160 (index 26)
# combo_spec: D1/0+D4/0+D17/0
# Name: ANY匹配+8-连通+面积阈值

import random
import sys
from pathlib import Path

import numpy as np

for _parent in Path(__file__).resolve().parents:
    if (_parent / "arc_utils.py").exists():
        sys.path.insert(0, str(_parent))
        break

from arc_utils import components_mask, paste, random_free_location


# Design analysis for combo D1/0 + D4/0 + D17/0
# - Concrete behavior and trace:
#   Keep the original visual grammar: a 4x4 legend in the top-left corner lists
#   three target colors, and several multicolor objects sit elsewhere on orange bg.
#   The fused rule is:
#     1) ignore the legend square itself,
#     2) find non-bg components with 8-connectivity,
#     3) erase a component iff its area is at least AREA_THRESHOLD and its color set
#        intersects the legend colors.
#   A concrete trace with legend {1,4,8}:
#     * A 4x4 rectangle using colors {0,1,9} is deleted here. In the original task it
#       survived because it lacked 4 and 8; this is the D1/0 ANY change.
#     * Three separate 2x2 blocks touching only by corners form one 8-connected area-12
#       object. Under 4-connectivity plus the same threshold, each block would still be
#       area 4 and survive; under 8-connectivity they merge and the whole object is
#       deleted. This is the D4/0 + D17/0 interaction that the generator must show.
#     * A tiny 2x2 object that contains color 4 stays, because area 4 < 10. Without
#       D17/0 it would also be deleted under ANY.
# - Complexity calibration:
#   The generator does not leave these cases to chance. Every sample contains:
#     * one large partial-hit deleted component (shows ANY vs ALL),
#     * one diagonal bridge deleted component (shows 8-connectivity can push area above
#       threshold),
#     * one small target-hit kept component (shows the area exemption),
#     * two large no-hit kept components (negative controls).
#   This keeps the rule readable from a few examples without exploding the number of
#   object types or thresholds: the only numeric cutoff is a fixed area 10.
# - Hidden assumption audit:
#   The original solver labeled all non-bg pixels, which becomes unsafe under ANY
#   because the legend itself contains target colors; the solver must mask out the
#   detected legend square before component analysis.
#   The original 4-connected assumption no longer holds, so both solver and generator
#   must reason in 8-connected components. The area check must run after that merge,
#   not on bboxes, not on colored-pixel count, and not on the old 4-connected pieces.
# - Placement legality:
#   Each object is pasted inside a reserved bbox chosen by random_free_location with
#   padding_connectivity=8, so separate objects never accidentally merge diagonally.
#   Only the intentionally crafted bridge object has diagonal internal contacts.
# - Visual observability and cleanliness:
#   The diagonal bridge is sparse but still composed of chunky 2x2 blocks, so humans
#   can see that corner-touching matters. The small exempt object is isolated and uses
#   a legend color, making the threshold effect visible. Objects do not overlap, and
#   the legend remains the only marker area in the corner.
# - New solver assumptions checked against generation:
#   main() assumes there is a valid top-left legend, all scene objects live outside it,
#   and any 8-connected target-hit component below threshold should survive. The
#   generator enforces each of those assumptions and rejects attempts that do not
#   exhibit the three fusion-specific behaviors simultaneously.


AREA_THRESHOLD = 10


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
    assert False, "no legend border found in grid"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove only the large target-hitting scene components.

    Input structure:
    - One background color dominates the board.
    - The top-left corner contains a small legend square whose bottom edge and right
      edge share one border color; the interior shows the target colors.
    - Outside that legend, the remaining non-background cells form separated scene
      objects, but some objects may connect only at diagonal corners.

    Rule:
    - Read the target color set from the legend interior.
    - Exclude the legend square itself from further component analysis.
    - Partition the other non-background pixels into 8-connected components.
    - Erase a component iff it contains **any** legend color and its area is at
      least `AREA_THRESHOLD`; keep target-free components and small target-hitting
      components.

    Key insight: The legend is now a loose color trigger, but size matters and
    diagonal contact matters too: corner-touching fragments can merge into one large
    removable object, while small matching fragments are exempt.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    sz, _border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    mask = (g != bg).copy()
    mask[:sz, :sz] = False
    for pts in components_mask(mask, connectivity=8):
        area = int(pts.shape[0])
        cols = {int(g[r, c]) for r, c in pts}
        if area >= AREA_THRESHOLD and (cols & legend_colors):
            for r, c in pts:
                out[r, c] = bg
    return out


# --------------- generator ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3
MAX_RETRIES = 400

KEEP_DIMS = [(3, 4), (3, 5), (4, 4), (5, 3)]
DELETE_DIMS = [(3, 4), (4, 4), (3, 5), (4, 5)]
SMALL_HIT_DIMS = [(2, 2), (2, 3), (3, 2)]
BRIDGE_BLOCK_SIZE = 2
BRIDGE_BLOCK_OFFSETS = [(0, 0), (2, 2), (4, 4)]
BRIDGE_PATCH_SHAPE = (6, 6)


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend box at top-left and place the three target colors."""
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER

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
        out[r, c] = color


def _sample_cells(coords: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    return random.sample(coords, count)


def _sprinkle_colors(
    patch: np.ndarray,
    coords: list[tuple[int, int]],
    forced_colors: list[int],
    allowed_colors: list[int],
    extra_count: int,
) -> None:
    total = len(forced_colors) + int(extra_count)
    chosen = _sample_cells(coords, total)
    colors = list(forced_colors) + [random.choice(allowed_colors) for _ in range(total - len(forced_colors))]
    random.shuffle(colors)
    for (r, c), color in zip(chosen, colors):
        patch[r, c] = int(color)


def _solid_patch(h: int, w: int, forced_colors: list[int], allowed_colors: list[int]) -> np.ndarray:
    """Full rectangle of 0s with a few colored pixels sprinkled in."""
    patch = np.zeros((h, w), dtype=int)
    area = h * w
    extra_count = random.randint(1, max(1, min(5, area - len(forced_colors))))
    coords = [(r, c) for r in range(h) for c in range(w)]
    _sprinkle_colors(patch, coords, forced_colors, allowed_colors, extra_count)
    return patch


def _bridge_patch(targets: list[int]) -> np.ndarray:
    """
    Three 2x2 blocks touching only diagonally.

    Under 8-connectivity this is one area-12 component.
    Under 4-connectivity it splits into three separate area-4 pieces.
    """
    patch = np.full(BRIDGE_PATCH_SHAPE, BG, dtype=int)
    block_coords: list[tuple[int, int]] = []
    for r0, c0 in BRIDGE_BLOCK_OFFSETS:
        patch[r0 : r0 + BRIDGE_BLOCK_SIZE, c0 : c0 + BRIDGE_BLOCK_SIZE] = 0
        for r in range(r0, r0 + BRIDGE_BLOCK_SIZE):
            for c in range(c0, c0 + BRIDGE_BLOCK_SIZE):
                block_coords.append((r, c))

    hit_targets = random.sample(targets, random.randint(1, NUM_TARGETS - 1))
    allowed = [v for v in PALETTE if v not in targets] + hit_targets
    extra_count = random.randint(2, 4)
    _sprinkle_colors(patch, block_coords, list(hit_targets), allowed, extra_count)
    return patch


def _large_hit_delete_patch(targets: list[int]) -> np.ndarray:
    """Large rectangle that hits a strict subset of legend colors, showing ANY vs ALL."""
    h, w = random.choice(DELETE_DIMS)
    hit_targets = random.sample(targets, random.randint(1, NUM_TARGETS - 1))
    allowed = [v for v in PALETTE if v not in targets] + hit_targets
    return _solid_patch(h, w, list(hit_targets), allowed)


def _small_hit_keep_patch(targets: list[int]) -> np.ndarray:
    """Small target-hitting component that stays because it is below threshold."""
    h, w = random.choice(SMALL_HIT_DIMS)
    hit_target = [random.choice(targets)]
    allowed = [v for v in PALETTE if v not in targets] + hit_target
    return _solid_patch(h, w, hit_target, allowed)


def _clean_keep_patch(targets: list[int]) -> np.ndarray:
    """Large component with no legend colors at all."""
    h, w = random.choice(KEEP_DIMS)
    non_targets = [v for v in PALETTE if v not in targets]
    forced = random.sample(non_targets, random.randint(1, min(2, len(non_targets))))
    return _solid_patch(h, w, forced, non_targets)


def _reserve_and_paste(canvas: np.ndarray, patch: np.ndarray) -> tuple[int, int]:
    r, c = random_free_location(
        canvas,
        np.zeros_like(patch),
        transparent=BG,
        padding=1,
        padding_connectivity=8,
    )
    paste(canvas, patch, r, c)
    return r, c


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether the sample really exercises the fused behavior.

    bridge_delete:
      deleted 8-component that splits into >=2 four-connected pieces, each below the
      threshold individually.
    small_hit_keep:
      kept target-hitting component whose 8-area is below the threshold.
    partial_large_delete:
      deleted component that hits only a strict subset of legend colors, proving ANY.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    sz, _border_color, legend_colors = _find_legend(g, bg)
    mask = (g != bg).copy()
    mask[:sz, :sz] = False

    stats = {
        "bridge_delete": 0,
        "small_hit_keep": 0,
        "partial_large_delete": 0,
    }
    for pts8 in components_mask(mask, connectivity=8):
        area8 = int(pts8.shape[0])
        cols = {int(g[r, c]) for r, c in pts8}
        hits = bool(cols & legend_colors)
        delete = area8 >= AREA_THRESHOLD and hits

        if delete and not legend_colors.issubset(cols):
            stats["partial_large_delete"] += 1
        if (not delete) and hits and area8 < AREA_THRESHOLD:
            stats["small_hit_keep"] += 1

        comp_mask = np.zeros_like(mask, dtype=bool)
        comp_mask[pts8[:, 0], pts8[:, 1]] = True
        comps4 = components_mask(comp_mask, connectivity=4)
        if delete and len(comps4) >= 2 and all(int(pts4.shape[0]) < AREA_THRESHOLD for pts4 in comps4):
            stats["bridge_delete"] += 1

    return stats


def _gen_attempt() -> np.ndarray | None:
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    roles = [
        ("bridge_delete", _bridge_patch(targets), False),
        ("large_hit_delete", _large_hit_delete_patch(targets), False),
        ("small_hit_keep", _small_hit_keep_patch(targets), True),
        ("clean_keep_a", _clean_keep_patch(targets), True),
        ("clean_keep_b", _clean_keep_patch(targets), True),
    ]
    random.shuffle(roles)

    for _name, patch, keep in roles:
        try:
            r, c = _reserve_and_paste(inp, patch)
        except ValueError:
            return None
        if keep:
            paste(out, patch, r, c)

    if not np.array_equal(main(inp), out):
        return None

    metrics = _fusion_metrics(inp)
    if min(metrics.values()) < 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
