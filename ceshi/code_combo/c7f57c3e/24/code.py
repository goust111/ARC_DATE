# Combo variant of c7f57c3e (index 24)
# combo_spec: D6/1+D9/2+D10/0
# Name: 颜色通道轮换 + 链式传递 + 三组结构

# Design analysis for this combo variant
# --------------------------------------
# a) Concrete behavior and trace:
#    The seed task had two geometric families, and every sprite switched to the opposite family while
#    keeping scale and anchor placement. This combo replaces that binary toggle with three phase groups
#    that all share one geometry but differ by a cyclic assignment of three visible variable colors.
#    Example with canonical role order [top, lower-mid, bottom] and colors [red, green, teal]:
#      group A = (red, green, teal)
#      group B = (green, teal, red)
#      group C = (teal, red, green)
#    The output applies one forward channel rotation to every sprite, so A->B, B->C, C->A. The group
#    chain and the color-channel cycle are therefore the same transformation seen at two levels.
# b) Complexity calibration:
#    Using only two colors would collapse back toward "swap". Using three role colors makes the cycle
#    visually explicit while keeping the hypothesis space small: three groups, one direction, one
#    global mapping. Every puzzle contains one scale-1 example of each group plus a larger scaled copy,
#    so the cyclic relation repeats enough times for humans to see it.
# c) Implicit assumption review:
#    - The seed solver assumed exactly two smallest prototypes and used anchor-aligned replacement.
#      That breaks here because geometry is no longer the changing signal.
#    - The new solver instead assumes the smallest sprites expose exactly three distinct phase patterns
#      with identical occupancy masks and the same three rotating colors.
#    - Order cannot come from hidden color ids. It is recovered from visible role positions inside the
#      shared prototype: first variable zone, second variable zone, third variable zone.
# d) Placement legality:
#    Input and output use identical occupied cells for each sprite, so generator placement only needs
#    one reserved bbox per sprite. `random_free_location` with border/padding still guarantees no
#    overlap and no out-of-bounds placement.
# e) Visual observability:
#    The viewer sees the same three color triples recurring on the same sprite shape. Across examples,
#    each triple advances to the next cyclic triple rather than reflecting or changing shape. Because
#    the three groups all appear in one grid, the A->B->C->A ring is directly observable.
# f) Visual cleanliness:
#    The scene keeps the seed's separated sprite layout and modest color count: bg, anchor color 1, and
#    exactly three rotating colors. No extra markers or overlaid guides are introduced.
# g) Color-change inferability:
#    The color mapping is tied to stable spatial roles, not numeric ids. The generator fixes a visible
#    top-to-bottom role order and ensures all three phases are present among the smallest sprites, so
#    the global rotation can be inferred from repeated structure.
# h) Anti-degeneracy checks:
#    The generator always includes three distinct scale-1 phases, and the output is produced by the
#    inverse of one global 3-cycle. This prevents degeneration to the seed's binary geometry swap or to
#    a trivial per-object recolor with unrelated mappings.
# i) Solver assumptions guaranteed by the generator:
#    - Smallest-area sprites contain exactly the three phase patterns.
#    - All phases share one occupancy mask and one anchor region.
#    - Each prototype exposes exactly three variable colors with distinct row-major first appearances.
#    - Every sprite uses only bg, anchor color 1, and those three cycle colors.

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "solvers"))

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _occupancy_key(arr: np.ndarray, bg: int) -> tuple[int, ...]:
    return tuple((arr != bg).astype(int).ravel().tolist())


def _ordered_role_colors(arr: np.ndarray, bg: int) -> tuple[int, int, int]:
    """Return the three rotating colors in visible role order."""
    colors = [int(c) for c in np.unique(arr) if int(c) not in (bg, 1)]
    assert len(colors) == 3, "each prototype must expose exactly three rotating colors"
    first_hits = []
    for color in colors:
        flat = np.flatnonzero(arr == color)
        assert flat.size > 0
        first_hits.append((int(flat[0]), color))
    first_hits.sort()
    assert len({hit for hit, _color in first_hits}) == 3, "variable roles must have distinct visible order"
    return tuple(color for _hit, color in first_hits)


def _apply_color_cycle(arr: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = arr.copy()
    for src, dst in mapping.items():
        out[arr == src] = dst
    return out


def _recover_cycle_mapping(base_protos: list[np.ndarray], bg: int) -> dict[int, int]:
    """Recover the shared forward 3-cycle from the three smallest phase prototypes."""
    occ_keys = {_occupancy_key(proto, bg) for proto in base_protos}
    assert len(occ_keys) == 1, "all phase groups must share one occupancy mask"

    triples = [_ordered_role_colors(proto, bg) for proto in base_protos]
    triple_set = set(triples)
    assert len(triple_set) == 3, "expected exactly three distinct phase triples"

    seed = triples[0]
    rot1 = (seed[1], seed[2], seed[0])
    rot2 = (seed[2], seed[0], seed[1])
    assert triple_set == {seed, rot1, rot2}, "phase groups must form one 3-cycle"

    mapping = {
        seed[0]: seed[1],
        seed[1]: seed[2],
        seed[2]: seed[0],
    }
    for triple in triple_set:
        for idx in range(3):
            assert mapping[triple[idx]] == triple[(idx + 1) % 3], "cycle must be globally consistent"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance every sprite one step around a 3-group color cycle while leaving its geometry untouched.

    Input structure:
    - The background is the most common color. The grid contains 4-5 disconnected sprites that are
      integer-scaled copies of one shared multicolor prototype with bg holes and a fixed color-1 anchor.
    - The smallest sprites reveal exactly three phase groups. They have identical occupied cells but
      differ by which three variable colors occupy the three visible variable roles.

    Rule:
    - Recover the three smallest phase prototypes and read their variable colors in visible role order.
    - Verify that those three role-color triples are cyclic rotations of one another, giving a directed
      chain A->B->C->A.
    - Apply that same forward 3-cycle of colors everywhere in the grid, keeping background, anchor
      pixels, scale, and placement unchanged.

    Key insight:
    - The group transition and the color-channel rotation are the same ring transform. Once the three
      smallest sprites expose the cycle colors in stable spatial roles, every other sprite can be moved
      to its successor phase by one global recoloring rather than by swapping geometry.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 4, "grid must contain at least 4 sprite components"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"bbox": (r0, c0, r1, c1), "arr": arr})

    areas = [int((r1 - r0 + 1) * (c1 - c0 + 1)) for r0, c0, r1, c1 in (s["bbox"] for s in sprites)]
    min_area = min(areas)
    base = [s for s, area in zip(sprites, areas, strict=False) if area == min_area]

    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(s["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = s["arr"]
    assert len(uniq) == 3, "smallest-area sprites must yield exactly 3 distinct phase prototypes"

    mapping = _recover_cycle_mapping(list(uniq.values()), bg)
    return _apply_color_cycle(g, mapping)


# --------------- generator (o5x) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 6

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 3), (4, 4), (5, 4)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
ROLE_ORDER = [2, 1, 3, 4]

# --- CFG: row-mask weights ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.70), ("full", 0.30)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: sprite instances ---
SPRITE_COUNTS = [4, 5]
BIG_SCALES = [2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 400
MAX_LABEL_RETRIES = 200

_ROW_MASKS = {
    "center": {3: [0, 1, 0], 4: [0, 1, 1, 0]},
    "outer": {3: [1, 0, 1], 4: [1, 0, 0, 1]},
    "full": {3: [1, 1, 1], 4: [1, 1, 1, 1]},
}


def _row_mask(kind: str, w: int) -> list[int]:
    return list(_ROW_MASKS[kind][w])


def _make_labels() -> np.ndarray:
    """Build one shared prototype geometry with 3 ordered variable roles plus anchor."""
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
            for _ in range(counts[role]):
                roles.append(role)
                masks.append(_row_mask(random.choices(kinds, weights=wts)[0], w))

        if not any(mask[0] or mask[-1] for mask in masks):
            idx = random.randrange(h)
            if roles[idx] == 1 and random.random() < OUTER_FIXUP_PROB:
                masks[idx] = _row_mask("outer", w)
            else:
                masks[idx] = _row_mask("full", w)

        if all(all(mask) for mask in masks):
            candidates = [i for i, role in enumerate(roles) if role != 1]
            masks[random.choice(candidates)] = _row_mask("center", w)

        labels = np.zeros((h, w), dtype=int)
        for r, (role, mask) in enumerate(zip(roles, masks, strict=False)):
            for c, bit in enumerate(mask):
                if bit:
                    labels[r, c] = role

        occ = labels != 0
        if len(components_mask(occ, connectivity=8)) != 1:
            continue
        if any(not np.any(labels == role) for role in (1, 2, 3, 4)):
            continue

        role_hits = [int(np.flatnonzero(labels == role)[0]) for role in (2, 3, 4)]
        if len(set(role_hits)) != 3:
            continue
        if role_hits != sorted(role_hits):
            continue
        return labels

    raise RuntimeError("failed to build an ordered connected prototype")


def _make_prototypes(bg: int) -> tuple[list[np.ndarray], dict[int, int]]:
    """Create the three phase groups as cyclic permutations of one 3-color assignment."""
    labels = _make_labels()
    pool = [c for c in range(10) if c not in (bg, 1)]
    cycle_colors = random.sample(pool, 3)

    def _colorize(phase: int) -> np.ndarray:
        proto = np.full(labels.shape, bg, dtype=int)
        proto[labels == 1] = 1
        proto[labels == 2] = cycle_colors[phase % 3]
        proto[labels == 3] = cycle_colors[(phase + 1) % 3]
        proto[labels == 4] = cycle_colors[(phase + 2) % 3]
        return proto

    protos = [_colorize(0), _colorize(1), _colorize(2)]
    mapping = {
        cycle_colors[0]: cycle_colors[1],
        cycle_colors[1]: cycle_colors[2],
        cycle_colors[2]: cycle_colors[0],
    }
    assert set(_ordered_role_colors(proto, bg) for proto in protos) == {
        tuple(cycle_colors),
        (cycle_colors[1], cycle_colors[2], cycle_colors[0]),
        (cycle_colors[2], cycle_colors[0], cycle_colors[1]),
    }
    return protos, mapping


def _build_instances() -> list[tuple[int, int]]:
    count = random.choice(SPRITE_COUNTS)
    big_scale = random.choice(BIG_SCALES)
    instances = [(0, 1), (1, 1), (2, 1), (random.randrange(3), big_scale)]
    if count == 5:
        instances.append((random.randrange(3), 1))
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)
    return instances


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    protos, forward_cycle = _make_prototypes(bg)
    inverse_cycle = {dst: src for src, dst in forward_cycle.items()}
    instances = _build_instances()

    max_dim = max(max(protos[phase].shape) * scale for phase, scale in instances)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for out_phase, scale in instances:
        output_patch = _scale_nn(protos[out_phase], scale)
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
        paste(inp, input_patch, r, c, transparent=bg)
        paste(out, output_patch, r, c, transparent=bg)

    try:
        base_comps = components_mask(inp != bg, connectivity=8)
        if len(base_comps) < 4:
            return None
        areas = []
        base_patterns = set()
        for pts in base_comps:
            r0, c0, r1, c1 = bbox_coords(pts)
            arr = inp[r0 : r1 + 1, c0 : c1 + 1]
            areas.append(int(arr.shape[0] * arr.shape[1]))
        min_area = min(areas)
        for pts, area in zip(base_comps, areas, strict=False):
            if area != min_area:
                continue
            r0, c0, r1, c1 = bbox_coords(pts)
            arr = inp[r0 : r1 + 1, c0 : c1 + 1]
            base_patterns.add(tuple(arr.ravel().tolist()))
        if len(base_patterns) != 3:
            return None
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
    raise RuntimeError("failed to generate a valid c7f57c3e combo input")
