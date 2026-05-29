# Variant of c7f57c3e
# Dimension: D6 — 颜色模式交换方式 (transform)
# Change: 两组之间完全交换颜色模式（A组获得B组的颜色，B组获得A组的颜色），同时保持尺度不变... → 颜色通道轮换

# Design analysis for this D6 variant
# -----------------------------------
# a) Concrete behavior and trace:
#    The original task changed geometry labels at fixed scale: every sprite of structural family A
#    became family B, and vice versa. Here geometry no longer changes. The grid still contains 3-4
#    scaled copies of two vertically related multicolor prototypes, but both families now share the
#    same three variable colors. The output rotates those three visible channels in-place for every
#    sprite: top-role color -> middle-role color -> bottom-role color -> top-role color in the
#    canonical orientation. Example on one 4-row prototype:
#      input rows (canonical A): role2=teal, role1=blue-anchor, role3=red, role4=green
#      output rows           : role2=red,  role1=blue-anchor, role3=green, role4=teal
#    Under the old logic, that sprite would have been replaced by the flipped prototype. Under the
#    new logic, the occupied cells stay put and only the three non-anchor colors rotate.
# b) Complexity calibration:
#    A two-color recolor would be too weak and too close to "swap". To make the new behavior visible,
#    each prototype now has three variable roles plus the fixed anchor color. The same 3-cycle is
#    repeated across all sprites and across both structural families, so 2-4 examples are enough for
#    a human to see a single global recolor rule rather than unrelated per-object edits.
# c) Old assumptions that break:
#    - The old solver assumed the smallest sprites expose exactly two fully colored prototypes and then
#      used overlap scoring to decide which opposite prototype to paste.
#    - That relies on "changing geometry family" as the transformation target; it is no longer valid
#      because the output keeps the same occupancy mask and same placement for every sprite.
#    - The old replacement code also relied on a color-1 anchor bbox to align pasted geometry. Since
#      geometry is preserved here, no anchor-based re-placement is needed in `main()`. The anchor stays
#      in the generator only as a stable visible structural cue and to keep the task close to the seed.
# d) Placement legality:
#    Sprites are still placed by bbox with a one-cell border and two-cell padding. Because input and
#    output shapes are identical in this variant, generator placement is simpler than the seed: the
#    same bbox is reserved for both input and output patches, so there is no alignment drift risk.
# e) Visual observability:
#    The cycle order is derived from visible vertical role order in a canonical orientation, not from
#    hidden color-id sorting. Canonicalization uses only the occupancy mask versus its vertical flip.
#    Both scale-1 families are always present, so the viewer sees the same three colors rotating on
#    both the upright and flipped structures.
# f) Visual cleanliness:
#    No overlapping markers are introduced. Objects remain separated exactly as in the seed, and the
#    output only changes colors inside already isolated components, so the scene stays easy to parse.
# g) Color-change inferability:
#    Only three variable colors participate in the cycle. They appear repeatedly across all sprites and
#    in fixed semantic rows of the canonical prototype, which provides the structure needed to infer the
#    mapping from a few examples.
# h) Generator anti-degeneracy:
#    The occupancy mask is forced to be vertically non-symmetric, otherwise the canonical orientation
#    and therefore the cycle direction would be ambiguous. The generator also verifies `main(input)`.
# i) New solver assumptions to guarantee:
#    `main()` assumes: exactly two distinct scale-1 families exist at the minimum bbox area; their
#    canonicalized occupancy masks collapse to one prototype; that prototype contains exactly three
#    non-bg, non-anchor colors with distinct top-to-bottom ordering. `generate_input()` enforces all of
#    these properties directly.

from pathlib import Path
import sys
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "solvers"))

from arc_utils import components_mask, bbox_coords, get_bg, paste, random_free_location


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Return the bbox of color-1 pixels inside a sprite."""
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "sprite array has no color-1 pixels and should always have them"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _occupancy_key(arr: np.ndarray, bg: int) -> tuple[int, ...]:
    return tuple((arr != bg).astype(int).ravel().tolist())


def _canonicalize_sprite(arr: np.ndarray, bg: int) -> np.ndarray:
    """Choose a stable vertical orientation using only the visible occupancy mask."""
    flipped = np.flipud(arr)
    key = _occupancy_key(arr, bg)
    fkey = _occupancy_key(flipped, bg)
    if fkey < key:
        return flipped.copy()
    return arr.copy()


def _ordered_cycle_colors(arr: np.ndarray, bg: int) -> list[int]:
    """List the three rotating colors in canonical top-to-bottom order."""
    colors = [int(c) for c in np.unique(arr) if c not in (bg, 1)]
    assert len(colors) == 3, "prototype must expose exactly three rotating colors"
    stats = []
    for color in colors:
        rs, _ = np.where(arr == color)
        stats.append((int(rs.min()), float(rs.mean()), color))
    stats.sort()
    return [color for _min_r, _mean_r, color in stats]


def _apply_color_cycle(arr: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = arr.copy()
    for src, dst in mapping.items():
        out[arr == src] = dst
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the three variable color channels inside every sprite while keeping geometry fixed.

    Input structure:
    - bg is the dominant color. The grid contains 3-4 disconnected sprite instances built from two
      small prototype families related by vertical flip, appearing at integer scales.
    - Every sprite uses the same fixed anchor color 1 plus the same three rotating colors; the two
      families differ by structure, not by having separate palettes.

    Rule:
    - Recover a canonical smallest prototype by vertical-flip normalization using the occupancy mask.
    - Read the three non-bg, non-anchor colors in canonical top-to-bottom role order.
    - Apply one forward 3-cycle of those colors everywhere in the grid, leaving background and the
      anchor color unchanged.

    Key insight:
    - The output does not switch a sprite into the other structural family anymore. The only change is
      a shared color-cycle, and the cycle direction is encoded by the visible row order of the roles in
      the canonical prototype, not by arbitrary color ids.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least 2 sprite components"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})

    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]

    raw_keys = {tuple(s["arr"].ravel().tolist()) for s in base}
    assert len(raw_keys) == 2, "smallest-area sprites must yield exactly 2 structural families"

    canonical = {}
    for s in base:
        carr = _canonicalize_sprite(s["arr"], bg)
        canonical[tuple(carr.ravel().tolist())] = carr
    assert len(canonical) == 1, "the two smallest families must collapse to one canonical prototype"

    proto = next(iter(canonical.values()))
    cycle = _ordered_cycle_colors(proto, bg)
    mapping = {
        cycle[0]: cycle[1],
        cycle[1]: cycle[2],
        cycle[2]: cycle[0],
    }
    return _apply_color_cycle(g, mapping)


# --------------- generator (o5x) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 6

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 3), (4, 4), (5, 4)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]

# --- CFG: row-mask weights ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7
ROLE_ORDER = [2, 1, 3, 4]

# --- CFG: sprite instances ---
SPRITE_COUNTS = [3, 4]
BIG_SCALES = [2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 300
MAX_LABEL_RETRIES = 200

_ROW_MASKS = {
    "center": {3: [0, 1, 0], 4: [0, 1, 1, 0]},
    "outer": {3: [1, 0, 1], 4: [1, 0, 0, 1]},
    "full": {3: [1, 1, 1], 4: [1, 1, 1, 1]},
}


def _row_mask(kind: str, w: int) -> list[int]:
    return list(_ROW_MASKS[kind][w])


def _anchor_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    rs, cs = np.where(arr == 1)
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _scale(arr: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, k, axis=0), k, axis=1)


def _make_labels() -> np.ndarray:
    """Build a vertically ordered label grid with roles 1(anchor), 2, 3, 4."""
    for _ in range(MAX_LABEL_RETRIES):
        h, w = random.choice(PROTO_SHAPES)
        counts = {1: 1, 2: 1, 3: 1, 4: 1}
        if h > 4:
            counts[random.choice(ROLE_ORDER)] += h - 4

        weights_by_role = {
            1: ANCHOR_ROW_WEIGHTS,
            2: VARIABLE_ROW_WEIGHTS,
            3: VARIABLE_ROW_WEIGHTS,
            4: VARIABLE_ROW_WEIGHTS,
        }
        roles: list[int] = []
        masks: list[list[int]] = []
        for role in ROLE_ORDER:
            entries = weights_by_role[role]
            kinds, wts = zip(*entries)
            for _row_idx in range(counts[role]):
                roles.append(role)
                masks.append(_row_mask(random.choices(kinds, weights=wts)[0], w))

        if not any(mask[0] or mask[-1] for mask in masks):
            i = random.randrange(h)
            if roles[i] == 1 and random.random() < OUTER_FIXUP_PROB:
                masks[i] = _row_mask("outer", w)
            else:
                masks[i] = _row_mask("full", w)

        if all(all(mask) for mask in masks):
            candidates = [i for i, role in enumerate(roles) if role != 1]
            masks[random.choice(candidates)] = _row_mask("center", w)

        labels = np.zeros((h, w), dtype=int)
        for r, (role, mask) in enumerate(zip(roles, masks, strict=False)):
            for c, bit in enumerate(mask):
                if bit:
                    labels[r, c] = role

        occ = labels != 0
        if np.array_equal(occ, np.flipud(occ)):
            continue
        if len(components_mask(occ, connectivity=8)) != 1:
            continue
        return labels

    raise RuntimeError("failed to build a non-symmetric connected label grid")


def _make_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Create two vertically related prototypes that share one 3-color cycle."""
    labels = _make_labels()
    pool = [c for c in range(10) if c not in (bg, 1)]
    cycle_colors = random.sample(pool, 3)

    def _colorize(lab: np.ndarray) -> np.ndarray:
        proto = np.full(lab.shape, bg, dtype=int)
        proto[lab == 1] = 1
        proto[lab == 2] = cycle_colors[0]
        proto[lab == 3] = cycle_colors[1]
        proto[lab == 4] = cycle_colors[2]
        return proto

    proto_a = _colorize(labels)
    proto_b = _colorize(np.flipud(labels))
    assert _occupancy_key(proto_a, bg) != _occupancy_key(np.flipud(proto_a), bg)
    return proto_a, proto_b, cycle_colors


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid; return None on placement failure or oracle mismatch."""
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b, cycle_colors = _make_prototypes(bg)
    protos = [proto_a, proto_b]

    forward_cycle = {
        cycle_colors[0]: cycle_colors[1],
        cycle_colors[1]: cycle_colors[2],
        cycle_colors[2]: cycle_colors[0],
    }
    inverse_cycle = {dst: src for src, dst in forward_cycle.items()}

    count = random.choice(SPRITE_COUNTS)
    big_scale = random.choice(BIG_SCALES)
    instances = [(0, 1), (1, 1), (random.randrange(2), big_scale)]
    if count == 4:
        instances.append((random.randrange(2), 1))
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)

    max_dim = max(max(protos[typ].shape) * scale for typ, scale in instances)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    out = np.full((n, n), bg, dtype=int)
    inp = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for typ, scale in instances:
        output_patch = _scale(protos[typ], scale)
        input_patch = _apply_color_cycle(output_patch, inverse_cycle)
        marker = np.full(output_patch.shape, 1, dtype=int)
        try:
            r, c = random_free_location(
                reserve,
                marker,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None

        paste(reserve, marker, r, c)
        paste(out, output_patch, r, c, transparent=bg)
        paste(inp, input_patch, r, c, transparent=bg)

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c7f57c3e input")
