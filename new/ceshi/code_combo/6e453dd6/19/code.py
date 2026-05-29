# Combo variant of 6e453dd6 (index 19)
# combo_spec: D4/1+D5/1
# Name: 多色精灵+宽松凹面触发
#
# Design analysis
# a) concretization + trace
#   - The original task had black-only sprites. After sliding flush to the grey separator, only fully enclosed
#     background holes that touched the inner-face column fired red rays.
#   - This combo makes every sprite two-colored: a dominant shell color plus a sparse marker color.
#   - The trigger geometry is also relaxed: a right-facing cavity may be a closed hole or an open C/L recess,
#     as long as some cavity cell that reaches the inner-face column belongs to a bg region adjacent to the
#     marker color.
#   - Concrete trace:
#       original flush shell: rectangular ring -> enclosed bg hole at sep-2 -> ray rows fire
#       combo flush shell: same ring still fires, but only because one shell cell bordering the cavity is recolored
#       to the marker color; an otherwise similar open C/L recess can also fire if it has that marker adjacency
#       even though the bg region is not enclosed.
#   - Difference from original: enclosure is no longer mandatory, while marker adjacency is now mandatory.
#
# b) complexity calibration
#   - The generator forces one open emitter and one closed-hole emitter in every sample, so the D5 relaxation is
#     visible without erasing the original enclosed-hole branch.
#   - It also forces at least one right-facing open recess whose marker is present elsewhere on the same sprite but
#     not adjacent to the recess. This makes the D4 color role visible instead of reducing the puzzle to geometry.
#   - Marker color is unique and sparse (one cell per sprite), which is enough to be readable without clutter.
#
# c) implicit-assumption audit for main()
#   - `left == black` no longer works, because sprites now use two colors. The solver must slide all non-bg cells
#     as one connected object.
#   - Clearing old sprite pixels must erase both shell and marker colors, not just black.
#   - Global enclosed-hole detection is invalid now: open recesses connect to outside bg, so the solver must inspect
#     each shifted component inside its own bbox and look for right-facing bg regions there.
#   - The old `hole touches sep-2 => emit` assumption is replaced by `candidate bg region touches the inner-face
#     column and is 4-neighbor adjacent to the marker color`.
#   - Trigger rows are deduplicated with a set because multiple candidate rows can arise from one recess.
#
# d) placement legality
#   - Row bands are pairwise disjoint with at least one full bg row between them, so sliding cannot merge sprites.
#   - Output-first construction stamps every sprite already flush to the separator; unsliding then shifts each sprite
#     left independently without overlap.
#   - Sprite widths are capped so every flush sprite still fits strictly inside the left pane.
#
# e) visual observability
#   - Every generated sample contains:
#       1. an open marked emitter,
#       2. a closed marked emitter,
#       3. an open unmarked-right-face distractor.
#   - That gives repeated support for both axes of the combo: open recesses can emit, enclosed holes still emit,
#     and color placement still matters.
#
# f) visual cleanliness
#   - One marker pixel per sprite keeps the two-color encoding readable.
#   - Right-side outputs remain simple full-width red rows, and sprites stay separated by blank bands.
#
# g) color/structure inference
#   - No hidden numeric ordering is used. The human-visible cue is "the rare marker color touching a right-facing
#     cavity near the separator", not the color id itself.
#   - Because the generator repeats both triggering and non-triggering marked sprites, the role of the marker is
#     inferable from a few examples.
#
# h) fusion metric used after implementation
#   - `open_trigger_rows`: rows emitted by open C/L recesses that would be rejected by the original enclosure-only rule.
#   - `suppressed_unmarked_rows`: right-facing recess rows that would emit under a color-agnostic relaxed rule but stay
#     silent here because the marker is not adjacent.
#   - The generator rejects samples unless both metrics are positive, so the combo cannot silently degenerate.
#
# i) new solver assumptions checked against generator
#   - Each sprite has exactly one marker-colored cell.
#   - The marker-colored cell remains 4-connected to the shell because it replaces a shell pixel.
#   - Candidate trigger rows never clash across sprites because row bands do not overlap.

import random

import numpy as np

from arc_utils import components_mask, paste


def _components4(mask):
    return components_mask(mask, connectivity=4)


def _find_separator_col(g: np.ndarray, sep_color: int = 5) -> int:
    _, w = g.shape
    cols = [c for c in range(w) if np.all(g[:, c] == sep_color)]
    assert len(cols) == 1, f"expected a unique full separator column, got {cols}"
    return int(cols[0])


def _background_color(g: np.ndarray, sep_col: int, sep_color: int = 5) -> int:
    h, w = g.shape
    border = []
    for c in range(w):
        if c != sep_col and g[0, c] != sep_color:
            border.append(int(g[0, c]))
        if c != sep_col and g[h - 1, c] != sep_color:
            border.append(int(g[h - 1, c]))
    for r in range(h):
        if g[r, 0] != sep_color:
            border.append(int(g[r, 0]))
        if g[r, w - 1] != sep_color:
            border.append(int(g[r, w - 1]))
    assert border, "expected non-separator border pixels to infer the background"
    vals, counts = np.unique(np.array(border, dtype=int), return_counts=True)
    winners = vals[counts == counts.max()]
    assert len(winners) == 1, f"background color must be unique on the border, got {winners.tolist()}"
    return int(winners[0])


def _infer_sprite_colors(left: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(left) if int(c) != bg]
    assert len(colors) == 2, f"expected exactly two sprite colors on the left, got {colors}"
    counts = {c: int(np.sum(left == c)) for c in colors}
    ordered = sorted(colors, key=lambda c: (-counts[c], c))
    shell_color, marker_color = ordered
    assert counts[shell_color] > counts[marker_color], "shell color must dominate marker color"
    return shell_color, marker_color


def _slide_left_components(left: np.ndarray, bg: int) -> tuple[np.ndarray, list[list[tuple[int, int, int]]]]:
    comps = _components4(left != bg)
    shifted = left.copy()
    shifted[shifted != bg] = bg
    target_right = left.shape[1] - 1
    shifted_components: list[list[tuple[int, int, int]]] = []

    for comp in comps:
        cells = [(int(r), int(c), int(left[r, c])) for r, c in comp]
        max_c = max(c for _, c, _ in cells)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        moved = []
        for r, c, val in cells:
            nc = c + delta
            assert 0 <= nc < left.shape[1], f"shifted pixel {(r, c)} left the pane"
            shifted[r, nc] = val
            moved.append((r, nc, val))
        shifted_components.append(moved)
    return shifted, shifted_components


def _component_patch(cells: list[tuple[int, int, int]], bg: int) -> tuple[np.ndarray, int, int]:
    rows = [r for r, _, _ in cells]
    cols = [c for _, c, _ in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c, val in cells:
        patch[r - r0, c - c0] = val
    return patch, r0, c0


def _port_specs_from_patch(
    patch: np.ndarray,
    bg: int,
    shell_color: int,
    marker_color: int,
) -> list[dict]:
    h, w = patch.shape
    if w < 2:
        return []

    ports = []
    for pts in _components4(patch == bg):
        trigger_rows = sorted(
            {
                int(r)
                for r, c in pts
                if int(c) == w - 2 and int(patch[int(r), w - 1]) == shell_color
            }
        )
        if not trigger_rows:
            continue

        marked = False
        for r, c in pts:
            rr, cc = int(r), int(c)
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and int(patch[nr, nc]) == marker_color:
                    marked = True
                    break
            if marked:
                break

        open_recess = any(int(r) in (0, h - 1) or int(c) == 0 for r, c in pts)
        ports.append({"rows": trigger_rows, "marked": marked, "open": open_recess})
    return ports


def _analyze_shifted_left(
    left: np.ndarray,
    bg: int,
    shell_color: int,
    marker_color: int,
) -> dict:
    metrics = {
        "trigger_rows": set(),
        "open_trigger_rows": 0,
        "closed_trigger_rows": 0,
        "suppressed_unmarked_rows": 0,
        "candidate_ports": 0,
    }

    for comp in _components4(left != bg):
        cells = [(int(r), int(c), int(left[r, c])) for r, c in comp]
        patch, r0, _ = _component_patch(cells, bg)
        for port in _port_specs_from_patch(patch, bg, shell_color, marker_color):
            metrics["candidate_ports"] += 1
            global_rows = {r0 + lr for lr in port["rows"]}
            if port["marked"]:
                metrics["trigger_rows"].update(global_rows)
                if port["open"]:
                    metrics["open_trigger_rows"] += len(global_rows)
                else:
                    metrics["closed_trigger_rows"] += len(global_rows)
            else:
                metrics["suppressed_unmarked_rows"] += len(global_rows)
    return metrics


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each two-color sprite to the grey separator, then fire red rays from marker-tagged right-facing recesses.

    Input structure:
    - One full-height grey (5) separator splits the board into a left sprite pane and a right empty pane.
    - Every left-side sprite is a single 4-connected object made from two colors: a dominant shell color and a sparse marker color.

    Rule:
    - Shift every two-color connected component right until its right edge is flush with the separator.
    - Inspect each shifted sprite inside its own bbox. A candidate source is any background region that reaches the column just inside the sprite's right wall.
    - If that right-facing bg region is 4-neighbor adjacent to the marker color, paint a red ray across the right pane for each row where the region reaches that inner-face column. Closed holes and open C/L recesses both count.

    Key insight:
    - The active source is no longer "an enclosed hole near the separator". It is "a marked nozzle near the separator", so color role and recess structure have to be read together.
    """
    g = np.asarray(grid, dtype=int)
    sep_color = 5
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()
    left = out[:, :sep].copy()
    shell_color, marker_color = _infer_sprite_colors(left, bg)
    shifted_left, _ = _slide_left_components(left, bg)
    out[:, :sep] = shifted_left

    metrics = _analyze_shifted_left(shifted_left, bg, shell_color, marker_color)
    for r in sorted(metrics["trigger_rows"]):
        assert np.all(out[r, sep + 1 :] == bg), f"right half of row {r} must be background before ray emission"
        out[r, sep + 1 :] = red
    return out


# --------------- generator ---------------

BG = 6
SHELL = 0
RED = 2
MARK = 3
SEP_COLOR = 5

H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 2, 4

N_SPRITES_MIN, N_SPRITES_MAX = 3, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

FLUSH_PROB = 0.35
MAX_RETRIES = 400

K_EMIT_C = "emit_c"
K_EMIT_L = "emit_l"
K_EMIT_HOLE = "emit_hole"
K_DECOY_OPEN = "decoy_open"
K_DECOY_SOLID = "decoy_solid"
K_DECOY_WRONG = "decoy_wrong"

KIND_MIN_H = {
    K_EMIT_C: 3,
    K_EMIT_L: 3,
    K_EMIT_HOLE: 3,
    K_DECOY_OPEN: 3,
    K_DECOY_SOLID: 2,
    K_DECOY_WRONG: 3,
}


def _scatter(total: int, slots: int) -> list[int]:
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _choose_grid_dims() -> tuple[int, int, int]:
    h = random.randint(H_MIN, H_MAX)
    right_w = random.randint(RIGHT_W_MIN, RIGHT_W_MAX)
    valid_seps = [s for s in range(SEP_MIN, SEP_MAX + 1) if W_MIN <= s + 1 + right_w <= W_MAX]
    sep = random.choice(valid_seps)
    w = sep + 1 + right_w
    return h, w, sep


def _choose_kinds() -> list[str]:
    n = random.randint(N_SPRITES_MIN, N_SPRITES_MAX)
    # Keep all fusion branches visible in each sample: one relaxed open trigger,
    # one classic closed-hole trigger, and one color-only distractor.
    kinds = [random.choice([K_EMIT_C, K_EMIT_L]), K_EMIT_HOLE, K_DECOY_OPEN]

    while len(kinds) < n:
        if len(kinds) == 3:
            # The first extra sprite alternates between structural and color distractors
            # so the sample set does not collapse to "all marked cavities fire".
            kinds.append(random.choice([K_DECOY_WRONG, K_DECOY_SOLID, K_EMIT_C, K_EMIT_L]))
            continue
        kinds.append(random.choice([K_EMIT_HOLE, K_EMIT_C, K_EMIT_L, K_DECOY_SOLID, K_DECOY_WRONG, K_DECOY_OPEN]))

    random.shuffle(kinds)
    return kinds


def _sprite_bands(h: int, kinds: list[str]) -> list[tuple[int, int]] | None:
    n = len(kinds)
    heights = [KIND_MIN_H[k] for k in kinds]
    base_gaps = [MIN_BAND_GAP] * (n + 1)
    if sum(heights) + sum(base_gaps) > h:
        return None

    extra = h - sum(heights) - sum(base_gaps)
    for _ in range(extra):
        choices = [i for i, cur_h in enumerate(heights) if cur_h < SPRITE_H_MAX]
        if not choices:
            break
        heights[random.choice(choices)] += 1

    gaps = [a + b for a, b in zip(base_gaps, _scatter(h - sum(heights) - sum(base_gaps), n + 1))]
    bands = []
    r = gaps[0]
    for i, sh in enumerate(heights):
        bands.append((r, sh))
        r += sh + gaps[i + 1]
    return bands


def _paint_marker(patch: np.ndarray, pos: tuple[int, int]) -> np.ndarray:
    r, c = pos
    assert int(patch[r, c]) == SHELL, f"marker must replace shell, got {patch[r, c]}"
    patch[r, c] = MARK
    return patch


def _hole_patch(sh: int, max_sw: int) -> np.ndarray:
    sw = random.randint(3, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    patch[0, :] = SHELL
    patch[-1, :] = SHELL
    patch[:, 0] = SHELL
    patch[:, -1] = SHELL
    return _paint_marker(patch, (0, sw - 2))


def _c_left_patch(sh: int, max_sw: int, marker_adjacent: bool) -> np.ndarray:
    sw = random.randint(3, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    patch[0, :] = SHELL
    patch[-1, :] = SHELL
    patch[:, -1] = SHELL
    marker_pos = (0, sw - 2) if marker_adjacent else (0, sw - 1)
    return _paint_marker(patch, marker_pos)


def _l_patch(sh: int, max_sw: int, marker_adjacent: bool) -> np.ndarray:
    sw = random.randint(3, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    patch[:, -1] = SHELL
    if random.random() < 0.5:
        patch[0, :] = SHELL
        marker_pos = (0, sw - 2) if marker_adjacent else (0, sw - 1)
    else:
        patch[-1, :] = SHELL
        marker_pos = (sh - 1, sw - 2) if marker_adjacent else (sh - 1, sw - 1)
    return _paint_marker(patch, marker_pos)


def _wrong_orientation_patch(sh: int, max_sw: int) -> np.ndarray:
    sw = random.randint(3, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    patch[0, :] = SHELL
    patch[-1, :] = SHELL
    patch[:, 0] = SHELL
    return _paint_marker(patch, (0, 1))


def _solid_patch(sh: int, max_sw: int) -> np.ndarray:
    sw = random.randint(2, min(3, max_sw))
    patch = np.full((sh, sw), SHELL, dtype=int)
    return _paint_marker(patch, (0, 0))


def _build_patch(kind: str, sh: int, max_sw: int) -> np.ndarray:
    if kind == K_EMIT_HOLE:
        return _hole_patch(sh, max_sw)
    if kind == K_EMIT_C:
        return _c_left_patch(sh, max_sw, marker_adjacent=True)
    if kind == K_EMIT_L:
        return _l_patch(sh, max_sw, marker_adjacent=True)
    if kind == K_DECOY_OPEN:
        if random.random() < 0.5:
            return _c_left_patch(sh, max_sw, marker_adjacent=False)
        return _l_patch(sh, max_sw, marker_adjacent=False)
    if kind == K_DECOY_WRONG:
        return _wrong_orientation_patch(sh, max_sw)
    if kind == K_DECOY_SOLID:
        return _solid_patch(sh, max_sw)
    raise ValueError(f"unknown kind: {kind}")


def _render_output(
    h: int,
    w: int,
    sep: int,
    bands: list[tuple[int, int]],
    kinds: list[str],
) -> tuple[np.ndarray, list[tuple[np.ndarray, int, int]], dict]:
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR
    sprites: list[tuple[np.ndarray, int, int]] = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    assert max_sw >= 3, "left pane must be wide enough for cavity sprites"

    metrics = {
        "open_trigger_rows": 0,
        "suppressed_unmarked_rows": 0,
        "closed_trigger_rows": 0,
    }

    for (r0, sh), kind in zip(bands, kinds):
        patch = _build_patch(kind, sh, max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)

        for port in _port_specs_from_patch(patch, BG, SHELL, MARK):
            rows = [r0 + lr for lr in port["rows"]]
            if port["marked"]:
                for rr in rows:
                    out[rr, sep + 1 :] = RED
                if port["open"]:
                    metrics["open_trigger_rows"] += len(rows)
                else:
                    metrics["closed_trigger_rows"] += len(rows)
            else:
                metrics["suppressed_unmarked_rows"] += len(rows)
        sprites.append((patch, r0, c0))
    return out, sprites, metrics


def _unslide_sprites(h: int, w: int, sep: int, sprites: list[tuple[np.ndarray, int, int]]) -> np.ndarray | None:
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR
    flush_i = random.randrange(len(sprites)) if random.random() < FLUSH_PROB else -1

    for i, (patch, r0, c0) in enumerate(sprites):
        if c0 < 0:
            return None
        if i == flush_i:
            delta = 0
        elif c0 == 0:
            delta = 0
        else:
            delta = random.randint(1, c0)
        paste(inp, patch, r0, c0 - delta, transparent=BG)
    return inp


def _gen_attempt() -> np.ndarray | None:
    h, w, sep = _choose_grid_dims()
    kinds = _choose_kinds()
    bands = _sprite_bands(h, kinds)
    if bands is None:
        return None

    out, sprites, render_metrics = _render_output(h, w, sep, bands, kinds)
    if render_metrics["open_trigger_rows"] <= 0:
        return None
    if render_metrics["suppressed_unmarked_rows"] <= 0:
        return None

    inp = _unslide_sprites(h, w, sep, sprites)
    if inp is None:
        return None

    try:
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(got, out):
        return None

    left = out[:, :sep]
    metrics = _analyze_shifted_left(left, BG, SHELL, MARK)
    if metrics["open_trigger_rows"] <= 0 or metrics["suppressed_unmarked_rows"] <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo input for 6e453dd6")
