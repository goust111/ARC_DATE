# Combo variant of 6e453dd6 (index 25)
# combo_spec: D12/1+D18/0
# Name: 非4连通干扰+任意封闭洞触发

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the fused D12/1 + D18/0 variant
# ----------------------------------------------------
# a) Concrete behavior and step trace
# - Seed task:
#   1. Treat every black sprite on the left as one 4-connected component.
#   2. Slide each such component right until its right edge touches the grey separator.
#   3. Only enclosed bg holes whose pixels touch column `sep-2` emit red rays.
# - D18/0 change:
#   - Remove the `sep-2` filter. After sliding, every enclosed bg hole emits on all rows it spans.
#   - Concrete trace: the shell
#       ####
#       #.##
#       #.##
#       ####
#     has a true hole in its second column. In the seed task that hole is ignored because it does not
#     reach the separator-facing inner column; in the fused task its two rows both fire.
# - D12/1 change:
#   - Some left-side "sprites" are only 8-connected, e.g.
#       #...
#       .#..
#       ..#.
#     A human sees one diagonal chain, but under 4-connectivity it is really three independent pixels.
#   - If one incorrectly slides it as a single 8-connected object, it stays diagonal after moving.
#     The correct 4-connected solver slides the three pixels independently, collapsing them into one
#     vertical line against the separator. Because the split pieces are tiny, they create no valid holes.
# - Fused difference:
#   - Real emitters are still 4-connected shells, but at least one shell has an off-wall hole that now
#     matters only because D18/0 is active.
#   - Fake diagonal clusters make connectivity itself behaviorally relevant on the left half.
#
# b) Complexity calibration
# - Keep only two semantic families:
#   - true 4-connected emitters with one enclosed hole
#   - fake 8-connected-only distractors with no hole after 4-splitting
# - Off-wall holes use only a small set of widths (1-2 cells) and a single fixed ray color, so the new
#   freedom stays visible without introducing extra attribute mappings.
# - Each accepted sample contains both:
#   - at least one off-wall hole row, so the output differs from the seed task
#   - at least one black 8-component whose 4-vs-8 slide results differ, so the connectivity twist is real
#
# c) Implicit-assumption review of the original main()
# - `components_mask(..., connectivity=4)` on black pixels remains correct and must not be relaxed to 8:
#   otherwise fake diagonal clusters would slide rigidly and the intended decoy behavior would disappear.
# - The original hole loop assumed "trigger iff touches `sep-2`". That assumption is removed here.
# - The original uniqueness assert on trigger rows is too strong once more holes are allowed; multiple holes
#   could share a row, so the fused solver collects rows in a set.
# - Global hole detection on the slid left pane is still safe because the generator keeps sprite row bands
#   disjoint, preventing cross-band accidental enclosures.
#
# d) Placement legality
# - The generator is input-first rather than output-first. That is deliberate: the D12/1 fake objects are
#   defined by their pre-slide diagonal connectivity, so constructing inputs directly is the cleanest way to
#   preserve the fused phenomenon.
# - Every semantic sprite gets its own vertical band with at least one full bg row around neighbors.
# - Widths are capped so every patch fits strictly left of the separator; the separator and right pane stay
#   free of black pixels in the input.
#
# e) Visual observability
# - A solver can see that some red rays come from holes that are visibly not adjacent to the separator after
#   sliding, which is enough evidence for "any enclosed hole" instead of "only separator-adjacent holes".
# - The fake diagonal chains are also visible: they look like single objects in the input, but the output
#   left half reveals that their diagonal links do not count for motion.
#
# f) Visual cleanliness
# - The board remains partitioned into left black geometry, one grey separator column, and right red rays.
# - No extra colors or metadata are introduced, and bands prevent left-side overlaps after sliding.
#
# g) Post-test degeneration metrics
# - Metric 1: `new_only_rows = rows(any enclosed hole) - rows(hole touching sep-2)`. We require this to be
#   non-empty, otherwise D18/0 has visually collapsed back to the seed task.
# - Metric 2: `slide_diff_pixels = diff(slide_by_4, slide_by_8)` on the left pane. We require this to be
#   positive, otherwise the fake 8-connected distractor had no behavioral effect.
#
# h) New solver assumptions the generator now guarantees
# - At least one true 4-connected emitter exists, and at least one of its holes is off-wall.
# - At least one fake 8-connected cluster exists, and its subcomponents have different right edges so
#   4-connected sliding and 8-connected sliding are observably different.
# - The right pane starts as pure background, so every emitted row can paint a full red segment safely.


BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 2, 4

N_SPRITES_MIN, N_SPRITES_MAX = 3, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
FAKE_H_MIN, FAKE_H_MAX = 3, 4
EMITTER_H_MIN = 4
MIN_BAND_GAP = 1

FAKE_EXTRA_PROB = 0.4
FLUSH_PROB = 0.35
MAX_RETRIES = 600


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    return components_mask(mask, connectivity=connectivity)


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    return _components(mask, connectivity=4)


def _components8(mask: np.ndarray) -> list[np.ndarray]:
    return _components(mask, connectivity=8)


def _find_separator_col(g: np.ndarray, sep_color: int = SEP_COLOR) -> int:
    _, w = g.shape
    cols = [c for c in range(w) if np.all(g[:, c] == sep_color)]
    assert len(cols) == 1, f"expected a unique full separator column, got {cols}"
    return int(cols[0])


def _background_color(g: np.ndarray, sep_col: int, sep_color: int = SEP_COLOR) -> int:
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


def _slide_left_components(left: np.ndarray, bg: int, connectivity: int) -> np.ndarray:
    slid = left.copy()
    comps = _components(slid == BLACK, connectivity=connectivity)
    slid[slid == BLACK] = bg
    target_right = slid.shape[1] - 1
    for comp in comps:
        max_c = int(comp[:, 1].max())
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        for r, c in comp:
            slid[int(r), int(c) + delta] = BLACK
    return slid


def _hole_records(left: np.ndarray, bg: int) -> list[dict]:
    h, w = left.shape
    records = []
    for pts in _components4(left == bg):
        pts = [(int(r), int(c)) for r, c in pts]
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in pts):
            continue
        rows = sorted({r for r, _ in pts})
        records.append(
            {
                "rows": rows,
                "touches_inner_adjacent": any(c == w - 2 for _, c in pts),
                "rightmost_col": max(c for _, c in pts),
            }
        )
    return records


def _trigger_rows_from_left(left: np.ndarray, bg: int, require_inner_adjacent: bool) -> list[int]:
    rows = set()
    for rec in _hole_records(left, bg):
        if require_inner_adjacent and not rec["touches_inner_adjacent"]:
            continue
        rows.update(rec["rows"])
    return sorted(rows)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black pieces to the separator, then fire from every enclosed hole rather than only the separator-facing ones.

    Input structure:
    - One full grey separator column splits the board into a left sprite field and an empty right field.
    - Black marks on the left include true 4-connected shells and diagonal 8-connected decoys over one bg color.

    Rule:
    - Segment black pixels with 4-connectivity and slide each resulting component right until its right edge is flush with the separator.
    - In the slid left pane, locate every enclosed bg component; each such hole contributes all of its rows, no matter where inside the shell it sits.
    - Paint a red ray across the whole right pane on every triggered row.

    Key insight: The two changes cooperate: diagonal links between black pixels do not make one movable object, and once the true 4-connected objects are moved, any sealed hole inside them emits even if it is set back from the separator wall.
    """
    g = np.asarray(grid, dtype=int)
    sep = _find_separator_col(g, sep_color=SEP_COLOR)
    bg = _background_color(g, sep_col=sep, sep_color=SEP_COLOR)

    out = g.copy()
    left = _slide_left_components(out[:, :sep].copy(), bg=bg, connectivity=4)
    out[:, :sep] = left

    trigger_rows = _trigger_rows_from_left(left, bg=bg, require_inner_adjacent=False)
    for r in trigger_rows:
        assert np.all(out[r, sep + 1 :] == bg), f"right half of row {r} must be background before ray emission"
        out[r, sep + 1 :] = RED

    return out


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


def _role_heights(role: str, max_fake_h: int) -> tuple[int, int]:
    if role == "fake":
        return FAKE_H_MIN, max_fake_h
    return EMITTER_H_MIN, SPRITE_H_MAX


def _sprite_bands(h: int, roles: list[str], max_fake_h: int) -> list[tuple[int, int]] | None:
    n = len(roles)
    min_heights = []
    max_heights = []
    for role in roles:
        lo, hi = _role_heights(role, max_fake_h)
        min_heights.append(lo)
        max_heights.append(hi)

    base_gaps = [MIN_BAND_GAP] * (n + 1)
    height_cap = h - sum(base_gaps)
    if sum(min_heights) > height_cap:
        return None

    heights = min_heights[:]
    max_extra = sum(mx - cur for mx, cur in zip(max_heights, heights))
    extra = random.randint(0, min(max_extra, height_cap - sum(heights)))
    for _ in range(extra):
        choices = [i for i in range(n) if heights[i] < max_heights[i]]
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


def _choose_roles(h: int, max_fake_h: int) -> tuple[list[str], list[tuple[int, int]]] | tuple[None, None]:
    candidates = []
    for n_total in range(N_SPRITES_MIN, N_SPRITES_MAX + 1):
        for n_fake in range(1, min(2, n_total - 2) + 1):
            if n_fake == 1 or random.random() < FAKE_EXTRA_PROB:
                n_emit = n_total - n_fake
                roles = ["offwall", "edge"] + [random.choice(["offwall", "edge"]) for _ in range(n_emit - 2)] + ["fake"] * n_fake
                random.shuffle(roles)
                bands = _sprite_bands(h, roles, max_fake_h=max_fake_h)
                if bands is not None:
                    candidates.append((roles, bands))
    if not candidates:
        return None, None
    return random.choice(candidates)


def _edge_emitter_patch(sh: int, max_sw: int) -> np.ndarray:
    sw = random.randint(3, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[1:-1, 0] = BLACK
    patch[1:-1, -1] = BLACK
    return patch


def _offwall_emitter_patch(sh: int, max_sw: int) -> np.ndarray:
    sw = random.randint(4, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[1:-1, 0] = BLACK
    patch[1:-1, -1] = BLACK
    # Keep exactly one enclosed hole and place it away from the separator-facing wall.
    inner_wall = sw - 2
    patch[1:-1, inner_wall] = BLACK
    return patch


def _fake_patch(sh: int, max_sw: int) -> np.ndarray:
    assert sh <= max_sw, "fake diagonal patch must fit the left pane width budget"
    patch = np.full((sh, sh), BG, dtype=int)
    reverse = random.random() < 0.5
    for r in range(sh):
        c = sh - 1 - r if reverse else r
        patch[r, c] = BLACK
    return patch


def _place_sprite(inp: np.ndarray, patch: np.ndarray, r0: int, sep: int, force_flush: bool) -> None:
    sw = patch.shape[1]
    max_c0 = sep - sw
    assert max_c0 >= 0, "patch must fit strictly left of the separator"
    min_c0 = 1 if max_c0 >= 1 else 0
    if force_flush:
        c0 = max_c0
    elif max_c0 > min_c0:
        c0 = random.randint(min_c0, max_c0 - 1)
    else:
        c0 = max_c0
    paste(inp, patch, r0, c0, transparent=BG)


def _connectivity_decoy_count(left: np.ndarray) -> int:
    count = 0
    for comp8 in _components8(left == BLACK):
        mask = np.zeros_like(left, dtype=bool)
        mask[comp8[:, 0], comp8[:, 1]] = True
        sub4 = _components4(mask)
        right_edges = {int(comp[:, 1].max()) for comp in sub4}
        if len(sub4) > 1 and len(right_edges) > 1:
            count += 1
    return count


def _sample_metrics(inp: np.ndarray) -> dict:
    sep = _find_separator_col(inp, sep_color=SEP_COLOR)
    bg = _background_color(inp, sep_col=sep, sep_color=SEP_COLOR)
    left = inp[:, :sep]
    slid4 = _slide_left_components(left.copy(), bg=bg, connectivity=4)
    slid8 = _slide_left_components(left.copy(), bg=bg, connectivity=8)
    rows_any = set(_trigger_rows_from_left(slid4, bg=bg, require_inner_adjacent=False))
    rows_old = set(_trigger_rows_from_left(slid4, bg=bg, require_inner_adjacent=True))
    offwall_holes = sum(1 for rec in _hole_records(slid4, bg=bg) if not rec["touches_inner_adjacent"])
    edge_holes = sum(1 for rec in _hole_records(slid4, bg=bg) if rec["touches_inner_adjacent"])
    return {
        "ray_rows": len(rows_any),
        "new_only_rows": len(rows_any - rows_old),
        "offwall_holes": offwall_holes,
        "edge_holes": edge_holes,
        "slide_diff_pixels": int(np.count_nonzero(slid4 != slid8)),
        "decoy_clusters": _connectivity_decoy_count(left),
    }


def _build_input(h: int, w: int, sep: int, roles: list[str], bands: list[tuple[int, int]]) -> np.ndarray | None:
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR
    max_sw = min(SPRITE_W_MAX, sep - 1)
    if max_sw < 4:
        return None

    flush_i = random.randrange(len(roles)) if random.random() < FLUSH_PROB else -1
    for i, ((r0, sh), role) in enumerate(zip(bands, roles)):
        if role == "fake":
            patch = _fake_patch(sh, max_sw=max_sw)
        elif role == "offwall":
            patch = _offwall_emitter_patch(sh, max_sw=max_sw)
        else:
            patch = _edge_emitter_patch(sh, max_sw=max_sw)
        _place_sprite(inp, patch, r0, sep, force_flush=(i == flush_i))
    return inp


def _gen_attempt() -> np.ndarray | None:
    h, w, sep = _choose_grid_dims()
    max_sw = min(SPRITE_W_MAX, sep - 1)
    max_fake_h = min(FAKE_H_MAX, max_sw)
    if max_sw < 4 or max_fake_h < FAKE_H_MIN:
        return None

    roles, bands = _choose_roles(h, max_fake_h=max_fake_h)
    if roles is None or bands is None:
        return None

    inp = _build_input(h, w, sep, roles, bands)
    if inp is None:
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    metrics = _sample_metrics(inp)
    if metrics["ray_rows"] == 0:
        return None
    if metrics["new_only_rows"] == 0:
        return None
    if metrics["offwall_holes"] == 0:
        return None
    if metrics["edge_holes"] == 0:
        return None
    if metrics["slide_diff_pixels"] == 0:
        return None
    if metrics["decoy_clusters"] == 0:
        return None
    if not np.any(out[:, sep + 1 :] == RED):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6e453dd6 combo D12/1+D18/0 input")
