# Combo variant of e3721c99 (index 19)
# combo_spec: D2/1+D8/3
# Name: 中空框架精灵+洞数取反匹配

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste


# Design analysis
# spec:
#   fused_rule:
#     old:
#       Grey sprites were irregular solid blobs with actual hole counts.
#       Matching used the same visible hole count on the legend and on the grey sprite.
#     new:
#       Grey sprites become thin hollow frames: wire-like rectangles partitioned into chambers.
#       Matching uses the complement within 0..3, so legend_actual_holes == 3 - sprite_actual_holes.
#   concrete_visualization:
#     legend_panel:
#       We keep the original colored legend bars because D2/1 only changes the source sprites, not the references.
#       Legend entries still visibly show actual hole counts 0..3 through solid 3-cell-thick bars.
#     workspace_sprites:
#       3 holes -> one outer frame split into three enclosed rooms by full divider walls.
#       2 holes -> one outer frame split into two rooms.
#       1 hole -> a single closed rectangular frame.
#       0 holes -> the same frame family but with one doorway opened in the outer wall, so the interior leaks out.
#   behavior_trace:
#     example:
#       legends:
#         0_holes -> red
#         1_hole  -> green
#         3_holes -> yellow
#       sprite:
#         actual shape = grey frame split into three chambers
#       original_flow:
#         actual holes = 3
#         direct lookup key = 3
#         output color = yellow
#       fused_flow:
#         actual holes = 3
#         complement key = 3 - 3 = 0
#         output color = red
#       visible_difference:
#         the most partitioned frame now maps to the most solid legend patch, reversing the old topology pairing.
#   complexity_calibration:
#     We restrict all actual hole counts to 0..3.
#     This keeps four readable topological classes, gives two clean opposite pairs (0<->3, 1<->2),
#     and lets the generator show the inversion clearly with only three legend entries per puzzle.
#     Using thin chambered frames keeps the D2 visual change obvious without adding noisy micro-structure.
#   implicit_assumptions_review:
#     _hole_count:
#       Still valid. It counts enclosed background components inside the sprite bbox, which matches frame chambers.
#       The new 0-hole class relies on a deliberate doorway so the inner region touches the bbox border and is not counted.
#     _legend_mapping:
#       Still valid because legend components remain closed colored shapes with unique actual hole counts.
#     grey component extraction:
#       Still valid because every generated frame is a single 4-connected component even after opening a doorway.
#     direct legend lookup:
#       No longer valid. main() must convert actual sprite holes with the complement map before recoloring.
#     out = g.copy():
#       Still safe because components are detected from the original grid, so recoloring one frame never changes later masks.
#   placement_legality:
#     Divider and legend panel keep the original protected-region layout.
#     Frame sprites are placed only in workspace cells, never overlap, and still keep a 1-cell padded gap via dilation.
#     Chamber dimensions are capped so 3-hole frames fit easily inside the 30x30 workspace alongside distractors.
#   visual_observability:
#     Every generated puzzle has exactly three legend classes out of four possible 0..3 classes.
#     Therefore:
#       at least one class is absent, producing distractors that only erase under the complement rule;
#       at least one matched sprite has actual holes different from its legend partner's visible holes.
#     The human can see both halves of the fusion: grey objects are unmistakably frames, and the color transfer is reversed.
#   visual_cleanliness:
#     Legends stay isolated in a dedicated panel behind the divider.
#     Workspace objects are thin, non-touching frames with consistent geometry, so the holes remain countable by eye.
#   color_inference:
#     Colors are attached only to explicit legend shapes.
#     No ordering by color id, size, or position is used; only visible hole topology and the fixed complement rule matter.
#   generator_guarantees_for_new_main:
#     Legend actual hole counts are unique and lie in 0..3.
#     Each matching sprite is generated with actual holes equal to 3 - legend_actual_holes.
#     Distractors use the one actual-hole class whose complementary legend key is absent.
#     At least one matched sprite per legend color is placed, so every legend participates in the output.
#
#   new_main_assumptions_checked:
#     Each non-divider legend color contributes exactly one interior legend component; layout construction enforces this.
#     Every grey component's actual hole count stays in 0..3; the frame template generator enforces this structurally.
#     Legend keys are unique after complementing because legend visible hole counts are unique.


HOLE_VALUE_MAX = 3


def _hole_count(mask: np.ndarray) -> int:
    """
    Count holes in a binary mask using 4-connectivity.
    A hole is a background component that does not touch the mask's bounding box border.
    """
    h, w = mask.shape
    bg = ~mask
    vis = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or vis[r0, c0]:
                continue
            stack = [(r0, c0)]
            vis[r0, c0] = True
            touches = False
            while stack:
                r, c = stack.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        stack.append((rr, cc))
            if not touches:
                holes += 1
    return holes


def _inverted_hole_value(hole_count: int) -> int:
    """Map an actual hole count in 0..3 to its complementary lookup key."""
    return HOLE_VALUE_MAX - int(hole_count)


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """
    Pick one legend component per non-(bg,grey) color:
      - component is at least 2x2,
      - component does not touch the outer grid border.
    Map the legend's actual hole count -> legend color.
    """
    h, w = g.shape
    mapping: dict[int, int] = {}
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey)]
    for c in colors:
        comps = components_mask(g == c, connectivity=4)
        candidates = []
        for pts in comps:
            r0, c0, r1, c1 = bbox_coords(pts)
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            if (r1 - r0 + 1) < 2 or (c1 - c0 + 1) < 2:
                continue
            candidates.append((r0, c0, r1, c1))
        if not candidates:
            continue
        assert len(candidates) == 1, f"color {c} must have exactly one legend component"
        r0, c0, r1, c1 = candidates[0]
        sub = g[r0 : r1 + 1, c0 : c1 + 1] == c
        hc = _hole_count(sub)
        assert hc not in mapping, f"legend hole count {hc} must map to exactly one color"
        mapping[hc] = c
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey frame sprites by matching the complement of their hole count to a color legend.

    Input structure:
    - The background is the dominant color, with grey (5) sprites scattered through the workspace.
    - The grey sprites are hollow frame-like components: thin outlines with zero to three enclosed chambers.
    - A separated legend panel contains colored reference shapes, each showing a distinct actual hole count.

    Rule:
    - Read each colored legend shape by its visible number of enclosed holes.
    - For every grey frame sprite, count its actual enclosed chambers, convert that count with `3 - holes`,
      and use the result as the legend lookup key.
    - Recolor the grey sprite to the matching legend color; erase sprites whose complementary key is absent.

    Key insight:
    - The task fuses a structural change with a logical inversion: the workspace objects are no longer solid blobs
      but thin chambered frames, and they match the legend by the opposite hole count within the fixed 0..3 range.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5

    legend = _legend_mapping(g, bg, grey)
    assert legend, "legend mapping must be non-empty: grid must contain colored legend shapes"

    out = g.copy()
    for pts in components_mask(g == grey, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            sub[r - r0, c - c0] = True
        actual_holes = _hole_count(sub)
        key = _inverted_hole_value(actual_holes)
        newc = int(legend[key]) if key in legend else bg
        for r, c in pts:
            out[r, c] = newc

    return out


# --------------- generator ---------------

BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# -- legend panel --
LEGEND_COUNT_RANGE = (3, 3)
MAX_HOLE_COUNT = HOLE_VALUE_MAX + 1
PANEL_SIZE_RANGE = (5, 8)
FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3

# -- sprites --
TOTAL_SPRITES_RANGE = (8, 11)
MAX_DISTRACTORS = 7
MAX_EXTRA_PER_LEGEND = 2
SPRITE_RETRIES = 20
CHAMBER_SPAN_RANGE = (2, 3)
CHAMBER_DEPTH_RANGE = (2, 4)
OPEN_GAP_SIZE_RANGE = (1, 2)
SPRITE_MAX_H = 10
SPRITE_MAX_W = 10

# -- generation --
GEN_RETRIES = 300

_STRUCT8 = generate_binary_structure(2, 2)


def _legend_span(hole_count: int) -> int:
    """Width (or height) of a single legend patch along the arrangement axis."""
    return 3 if hole_count == 0 else 2 * hole_count + 1


def _legend_patch(hole_count: int, color: int, arrangement: str) -> np.ndarray:
    """Build a 3xSpan (horizontal) or Spanx3 (vertical) legend patch."""
    span = _legend_span(hole_count)
    if arrangement == "horizontal":
        patch = np.full((3, span), int(color), dtype=int)
        if hole_count:
            patch[1, 1::2] = BG
    else:
        patch = np.full((span, 3), int(color), dtype=int)
        if hole_count:
            patch[1::2, 1] = BG
    return patch


def _line_span(legends: list[tuple[int, int]]) -> int:
    """Total span of legend patches plus 1-cell gaps between them."""
    return sum(_legend_span(h) for h, _ in legends) + len(legends) - 1


def _choose_legends(arrangement: str, max_span: int) -> list[tuple[int, int]]:
    """Pick 3 legends with distinct actual hole counts that fit within max_span."""
    count = random.randint(*LEGEND_COUNT_RANGE)
    combos = [
        holes
        for holes in itertools.combinations(range(MAX_HOLE_COUNT), count)
        if sum(_legend_span(h) for h in holes) + count - 1 <= max_span
    ]
    if not combos:
        return _choose_legends(arrangement, max_span + 2)
    holes = list(random.choice(combos))
    random.shuffle(holes)
    colors = random.sample(LEGEND_COLORS, count)
    return list(zip(holes, colors))


def _place_legend_line(
    grid: np.ndarray, legends: list[tuple[int, int]], arrangement: str, r: int, c: int
) -> None:
    """Paste legend patches sequentially along the arrangement axis."""
    rr, cc = int(r), int(c)
    for hole_count, color in legends:
        patch = _legend_patch(hole_count, color, arrangement)
        paste(grid, patch, rr, cc)
        if arrangement == "horizontal":
            cc += patch.shape[1] + 1
        else:
            rr += patch.shape[0] + 1


def _layout_full_horizontal(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place a full-row horizontal divider with legend panel above or below."""
    legends = _choose_legends("horizontal", FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    panel_h = random.randint(*PANEL_SIZE_RANGE)
    top = random.choice([True, False])
    c0 = random.randint(1, GRID_SIZE - 1 - total)
    if top:
        div_r = panel_h
        grid[div_r, :] = DIV
        blocked[: div_r + 1, :] = 9
        r0 = random.randint(1, panel_h - 4)
    else:
        div_r = GRID_SIZE - 1 - panel_h
        grid[div_r, :] = DIV
        blocked[div_r:, :] = 9
        r0 = random.randint(div_r + 2, GRID_SIZE - 4)
    _place_legend_line(grid, legends, "horizontal", r0, c0)
    return legends


def _layout_full_vertical(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place a full-column vertical divider with legend panel left or right."""
    legends = _choose_legends("vertical", FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    panel_w = random.randint(*PANEL_SIZE_RANGE)
    left = random.choice([True, False])
    r0 = random.randint(1, GRID_SIZE - 1 - total)
    if left:
        div_c = panel_w
        grid[:, div_c] = DIV
        blocked[:, : div_c + 1] = 9
        c0 = random.randint(1, panel_w - 4)
    else:
        div_c = GRID_SIZE - 1 - panel_w
        grid[:, div_c] = DIV
        blocked[:, div_c:] = 9
        c0 = random.randint(div_c + 2, GRID_SIZE - 4)
    _place_legend_line(grid, legends, "vertical", r0, c0)
    return legends


def _l_corner_geometry(
    corner: str, panel_h: int, panel_w: int, arrangement: str, total: int
) -> tuple[int, int, int, int, int, int]:
    """Compute divider pivot and legend placement bounds for an L-corner."""
    n = GRID_SIZE
    cross_dim = 3 if arrangement == "horizontal" else total
    along_dim = total if arrangement == "horizontal" else 3

    if corner == "tl":
        dr, dc = panel_h, panel_w
        r_min, r_max = 1, panel_h - cross_dim - 1
        c_min, c_max = 1, panel_w - along_dim - 1
    elif corner == "tr":
        dr, dc = panel_h, n - 1 - panel_w
        r_min, r_max = 1, panel_h - cross_dim - 1
        c_min, c_max = dc + 2, n - 1 - along_dim
    elif corner == "bl":
        dr, dc = n - 1 - panel_h, panel_w
        r_min, r_max = dr + 2, n - 1 - cross_dim
        c_min, c_max = 1, panel_w - along_dim - 1
    else:
        dr, dc = n - 1 - panel_h, n - 1 - panel_w
        r_min, r_max = dr + 2, n - 1 - cross_dim
        c_min, c_max = dc + 2, n - 1 - along_dim

    return dr, dc, r_min, r_max, c_min, c_max


def _layout_l(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place an L-shaped divider with legend panel in one corner."""
    arrangement = random.choice(["horizontal", "vertical"])
    legends = _choose_legends(arrangement, L_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    along = total + 2 + random.randint(0, min(L_LAYOUT_EXTRA_PADDING, 27 - (total + 2)))
    cross = random.randint(*PANEL_SIZE_RANGE)
    if arrangement == "horizontal":
        panel_h, panel_w = cross, along
    else:
        panel_h, panel_w = along, cross
    corner = random.choice(["tl", "tr", "bl", "br"])

    dr, dc, r_min, r_max, c_min, c_max = _l_corner_geometry(
        corner, panel_h, panel_w, arrangement, total
    )

    if corner == "tl":
        grid[dr, : dc + 1] = DIV
        grid[: dr + 1, dc] = DIV
        blocked[:dr, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[: dr + 1, dc] = 9
    elif corner == "tr":
        grid[dr, dc:] = DIV
        grid[: dr + 1, dc] = DIV
        blocked[:dr, dc + 1:] = 9
        blocked[dr, dc:] = 9
        blocked[: dr + 1, dc] = 9
    elif corner == "bl":
        grid[dr, : dc + 1] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1:, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[dr:, dc] = 9
    else:
        grid[dr, dc:] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1:, dc + 1:] = 9
        blocked[dr, dc:] = 9
        blocked[dr:, dc] = 9

    r0 = random.randint(r_min, r_max)
    c0 = random.randint(c_min, c_max)
    _place_legend_line(grid, legends, arrangement, r0, c0)
    return legends


def _add_layout(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Randomly choose and apply a divider layout."""
    kind = random.choice(["horizontal", "vertical", "l"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    if kind == "vertical":
        return _layout_full_vertical(grid, blocked)
    return _layout_l(grid, blocked)


def _closed_frame_mask(chambers: int, chamber_span: int, chamber_depth: int) -> np.ndarray:
    """Build a thin rectangular frame split into `chambers` sealed rooms."""
    h = chamber_depth + 2
    w = chambers * chamber_span + (chambers + 1)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    for k in range(1, chambers):
        wall_c = k * (chamber_span + 1)
        mask[:, wall_c] = True
    return mask


def _door_cells(mask: np.ndarray, side: str, gap: int) -> list[tuple[int, int]]:
    """Cells to clear for a doorway while preserving component connectivity."""
    h, w = mask.shape
    if side in ("top", "bottom"):
        span = w - 2
        gap = min(gap, span)
        start = 1 + (span - gap) // 2
        row = 0 if side == "top" else h - 1
        return [(row, c) for c in range(start, start + gap)]
    span = h - 2
    gap = min(gap, span)
    start = 1 + (span - gap) // 2
    col = 0 if side == "left" else w - 1
    return [(r, col) for r in range(start, start + gap)]


def _open_frame_mask(chamber_span: int, chamber_depth: int) -> np.ndarray:
    """Build a frame-like shape whose interior is open to the outside, so it has 0 holes."""
    mask = _closed_frame_mask(1, chamber_span, chamber_depth)
    side = random.choice(["top", "bottom", "left", "right"])
    gap = random.randint(*OPEN_GAP_SIZE_RANGE)
    for r, c in _door_cells(mask, side, gap):
        mask[r, c] = False
    return mask


def _frame_mask(hole_count: int) -> np.ndarray:
    """Create a 4-connected hollow-frame sprite with the requested actual hole count."""
    chamber_span = random.randint(*CHAMBER_SPAN_RANGE)
    chamber_depth = random.randint(*CHAMBER_DEPTH_RANGE)
    if hole_count == 0:
        mask = _open_frame_mask(chamber_span, chamber_depth)
    else:
        mask = _closed_frame_mask(hole_count, chamber_span, chamber_depth)
    if random.choice([False, True]):
        mask = mask.T
    assert _hole_count(mask) == hole_count
    assert mask.shape[0] <= SPRITE_MAX_H and mask.shape[1] <= SPRITE_MAX_W
    return mask


def _make_sprite(hole_count: int) -> np.ndarray | None:
    """Generate a thin grey frame sprite with the requested actual hole count."""
    for _ in range(SPRITE_RETRIES):
        mask = _frame_mask(hole_count)
        sprite = np.where(mask, GREY, BG).astype(int)
        if _hole_count(mask) == hole_count:
            return sprite
    return None


def _sprite_plan(legends: list[tuple[int, int]]) -> list[tuple[int, int | None]]:
    """Build a shuffled list of (sprite_actual_holes, color_or_None) to place."""
    total = random.randint(*TOTAL_SPRITES_RANGE)
    remaining = total - len(legends)
    max_d = min(MAX_DISTRACTORS, remaining)

    legend_holes = {h for h, _ in legends}
    distractor_holes = [h for h in range(MAX_HOLE_COUNT) if _inverted_hole_value(h) not in legend_holes]
    min_d = 1 if distractor_holes and max_d >= 1 else 0
    valid_d = [d for d in range(min_d, max_d + 1) if remaining - d <= MAX_EXTRA_PER_LEGEND * len(legends)]
    n_distractors = random.choice(valid_d) if valid_d else 0

    extra_matches = remaining - n_distractors
    extras = [0] * len(legends)
    for _ in range(extra_matches):
        choices = [i for i, extra in enumerate(extras) if extra < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    plan = []
    for i, (legend_actual_holes, color) in enumerate(legends):
        sprite_actual_holes = _inverted_hole_value(legend_actual_holes)
        for _ in range(1 + extras[i]):
            plan.append((sprite_actual_holes, color))
    for _ in range(n_distractors):
        plan.append((random.choice(distractor_holes), None))
    random.shuffle(plan)
    return plan


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    """8-connectivity dilation of a boolean mask."""
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location(
    static_blocked: np.ndarray, occupied: np.ndarray, sprite: np.ndarray
) -> tuple[int, int] | None:
    """Find a random valid placement for sprite, avoiding blocked and occupied areas."""
    h, w = sprite.shape
    sprite_mask = sprite == GREY
    occupied_pad = _dilate_mask(occupied != BG)
    n = GRID_SIZE
    candidates = [(r, c) for r in range(n - h + 1) for c in range(n - w + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        rr0, cc0 = max(0, r - 1), max(0, c - 1)
        rr1, cc1 = min(n, r + h + 1), min(n, c + w + 1)
        if np.any(static_blocked[rr0:rr1, cc0:cc1] != BG):
            continue
        if np.any(occupied_pad[r : r + h, c : c + w] & sprite_mask):
            continue
        return int(r), int(c)
    return None


def _place_sprites(
    grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[tuple[int, int]]
) -> bool:
    """Generate and place all frame sprites (matches + distractors) onto the grid."""
    items = []
    for hole_count, color in _sprite_plan(legends):
        sprite = _make_sprite(hole_count)
        if sprite is None:
            return False
        items.append((sprite.size, sprite, color))
    items.sort(key=lambda x: x[0], reverse=True)

    occupied = np.zeros_like(grid)
    for _, sprite, color in items:
        loc = _random_sprite_location(blocked, occupied, sprite)
        if loc is None:
            return False
        r, c = loc
        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)
        if color is not None:
            out_patch = np.where(sprite == GREY, int(color), BG)
            paste(expected, out_patch, r, c, transparent=BG)
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build layout, place sprites, validate via oracle."""
    n = GRID_SIZE
    grid = np.zeros((n, n), dtype=int)
    expected = np.zeros((n, n), dtype=int)
    blocked = np.zeros((n, n), dtype=int)

    legends = _add_layout(grid, blocked)
    expected[:, :] = grid
    if not _place_sprites(grid, expected, blocked, legends):
        return None
    if int(np.sum(grid == BG)) <= int(np.sum(grid == GREY)):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e3721c99 combo input")
