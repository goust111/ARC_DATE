# Combo variant of 4c416de3 (index 20)
# combo_spec: D3/0+D4/1+D9/0
# Name: 部分线索+色交换+一对一
#
# Design analysis:
# a) Concretization and behavior trace
#    - Keep the original object vocabulary: 2-3 hollow 0-frames on a solid bg, plus one shared
#      diagonally symmetric corner glyph.
#    - Each active source corner now shows the full glyph mostly in a source-main color `S`, but
#      its corner cell `(0,0)` is recolored to a pair color `T`. Each active target corner shows
#      only 2-3 same-colored clue cells in color `T`, all drawn from the glyph's non-outward part.
#    - Output performs a pairwise swap: that target is completed in `S`, and that source becomes a
#      fully uniform `T` glyph.
#    - Small trace: source A is mostly blue with a red corner-cell accent; target A has two red
#      clue pixels. Source B is mostly green with a yellow accent; target B has yellow clues.
#      Original task would fill targets with their own clue colors and leave sources unchanged.
#      This combo instead matches red clues to the red-accent source, fills that target in blue,
#      and repaints that source fully red; likewise for the green/yellow pair.
# b) Complexity calibration
#    - Pairing is one-to-one, not 1:many: target clue colors are unique, and each appears as the
#      unique accent color of exactly one source.
#    - Only two pairs are used per grid, so the color logic is visible without exploding to too
#      many unrelated colors. Within a pair, the same pair color repeats on both source and target,
#      which gives the solver a human-visible matching cue.
#    - Target clues are restricted to 2-3 non-outward cells, and at least one target omits the
#      deepest diagonal cell. This keeps the D3 change visually obvious and prevents regression to
#      the original single-marker task.
# c) Implicit assumptions to replace
#    - Original `main()` assumed a source glyph is monochrome. That is false here: sources are
#      mostly one color but contain one fixed accent cell carrying the partner target color.
#    - Original `main()` assumed every target exposes the deepest diagonal marker. That is false:
#      targets are recognized from any 2-3 same-colored visible clue cells inside the non-outward
#      clue zone.
#    - Original `main()` assumed target output color comes from the target itself. Here it comes
#      from the matched source's main color, while the source is repainted with the target clue
#      color.
#    - Pairing introduces a new assumption: clue colors must uniquely match source accent colors.
#      The generator guarantees this by using distinct pair colors and exactly one target per pair.
# d) Placement legality
#    - The canvas and frame placement stay within the original 18-24 range and keep >=4 bg cells
#      between frames. Pattern templates stay within radius 2, so the corner scans remain local.
#    - The generator activates only four corners (two sources, two targets) and rejects layouts
#      where those full glyph footprints would overlap in the final output.
# e) Visual observability
#    - A human sees two dense sources and two sparse targets. Each sparse target color reappears as
#      the corner-cell accent of exactly one source, making the 1:1 color-based pairing inferable.
#    - The before/after contrast is clear: sources densify in their accent color, while targets
#      densify in the source-main color.
# f) Visual cleanliness
#    - Inactive corners stay plain 0 corners, so the grid is not cluttered by extra colored noise.
#    - All clue pixels remain attached to their own frame corners and never overlap other active
#      glyphs.
# g) Change inferability
#    - Color changes are structurally anchored: source-main color identifies what a target will
#      receive, and source accent / target clue color identifies what the source will become.
#    - No hidden numeric ordering is used; the pairing depends only on visible color equality plus
#      the stable source/target shape distinction.

import random
from itertools import combinations

import numpy as np

from arc_utils import draw_line
from arc_utils import get_bg


CORNER_KINDS = ("TL", "TR", "BL", "BR")

EDGE_DIRS = {
    "TL": ((+1, 0), (0, +1)),
    "TR": ((+1, 0), (0, -1)),
    "BL": ((-1, 0), (0, +1)),
    "BR": ((-1, 0), (0, -1)),
}
OUT_DIRS = {
    "TL": ((-1, 0), (0, -1)),
    "TR": ((-1, 0), (0, +1)),
    "BL": ((+1, 0), (0, -1)),
    "BR": ((+1, 0), (0, +1)),
}

_NEIGH_R = 3
ACCENT_CANON = (0, 0)
N_PAIRS = 2


def _rot(dr, dc, kind):
    """Map actual offset <-> canonical (TL-oriented) offset; self-inverse."""
    if kind == "TL":
        return (dr, dc)
    if kind == "TR":
        return (dr, -dc)
    if kind == "BL":
        return (-dr, dc)
    if kind == "BR":
        return (-dr, -dc)
    raise ValueError(f"unknown corner kind {kind}")


def _edge_metric(g, r, c, dr, dc, bg, H, W, max_step=10):
    """Walk from (r,c) along (dr,dc) until bg/boundary. Return (non_bg_count, saw_zero)."""
    cnt = 0
    saw_zero = False
    for k in range(1, max_step + 1):
        rr, cc = r + k * dr, c + k * dc
        if not (0 <= rr < H and 0 <= cc < W):
            break
        v = int(g[rr, cc])
        if v == bg:
            break
        cnt += 1
        if v == 0:
            saw_zero = True
    return cnt, saw_zero


def _find_corners(g, bg):
    H, W = g.shape
    corners = []
    for r in range(H):
        for c in range(W):
            if int(g[r, c]) == bg:
                continue
            for kind in CORNER_KINDS:
                (d1r, d1c), (d2r, d2c) = EDGE_DIRS[kind]
                r1, c1 = r + d1r, c + d1c
                r2, c2 = r + d2r, c + d2c
                if not (0 <= r1 < H and 0 <= c1 < W and 0 <= r2 < H and 0 <= c2 < W):
                    continue
                if int(g[r1, c1]) == bg or int(g[r2, c2]) == bg:
                    continue
                cnt1, sz1 = _edge_metric(g, r, c, d1r, d1c, bg, H, W)
                cnt2, sz2 = _edge_metric(g, r, c, d2r, d2c, bg, H, W)
                if not (cnt1 >= 3 and sz1 and cnt2 >= 3 and sz2):
                    continue
                ok = True
                for odr, odc in OUT_DIRS[kind]:
                    rr, cc = r + odr, c + odc
                    if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == 0:
                        ok = False
                        break
                if not ok:
                    continue
                corners.append((r, c, kind))
                break
    return corners


def _colored_around(g, r, c, kind, bg):
    H, W = g.shape
    items = []
    for dr in range(-_NEIGH_R, _NEIGH_R + 1):
        for dc in range(-_NEIGH_R, _NEIGH_R + 1):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            v = int(g[rr, cc])
            if v == bg or v == 0:
                continue
            items.append((_rot(dr, dc, kind), v))
    return items


def _is_outward(off):
    return off[0] < 0 or off[1] < 0


def _clue_offsets(pattern):
    return tuple(sorted(off for off in pattern if not _is_outward(off)))


def _extract_source_info(c_key, items):
    outward = [(off, color) for off, color in items if _is_outward(off)]
    colors = {color for _, color in outward}
    assert len(outward) >= 2 and len(colors) == 1, f"invalid source at {c_key}"
    main_color = next(iter(colors))

    offset_to_color = {}
    for off, color in items:
        assert off not in offset_to_color, f"duplicate source offset {off} at {c_key}"
        offset_to_color[off] = color

    accent_cells = [(off, color) for off, color in items if color != main_color]
    assert len(accent_cells) == 1, f"source {c_key} should have exactly one accent cell"
    accent_off, accent_color = accent_cells[0]
    assert accent_off == ACCENT_CANON, f"source {c_key} accent must be at {ACCENT_CANON}"
    assert not _is_outward(accent_off), f"source {c_key} accent cannot be outward"

    canon = frozenset(offset_to_color)
    return canon, main_color, accent_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sparse target clues to accented source glyphs, then swap colors pairwise.

    Input structure:
    - bg is the most frequent color, and 0-valued hollow frames provide detectable
      TL/TR/BL/BR corners.
    - Multiple source corners already show the full rotated glyph. Each source is
      mostly one color, but its corner cell carries a different accent color.
    - The same number of target corners keep only 2-3 same-colored clue cells from
      the glyph's non-outward region; at least some targets may omit the deepest
      diagonal cell. Remaining corners are plain frame corners.

    Rule:
    - Recover the shared canonical glyph from the sources.
    - Pair each target with the unique source whose accent color matches that target's
      clue color.
    - Repaint the target with that source's main color, and repaint the source into
      the target/source shared pair color.

    Key insight:
    - Shape recovery still comes from the full sources, but color is now bidirectional
      and pair-specific. The source main color tells you what the matched target gains,
      while the source accent color and target clue color identify which source and
      target belong together.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corners = _find_corners(g, bg)
    assert corners, "no frame corner detected"

    per_corner = {c: _colored_around(g, *c, bg) for c in corners}

    sources = []
    source_meta = {}
    for c_key, items in per_corner.items():
        outward = [(off, color) for off, color in items if _is_outward(off)]
        if len(outward) < 2:
            continue
        colors = {color for _, color in outward}
        assert len(colors) == 1, f"source {c_key} has mixed outward colors: {colors}"
        sources.append(c_key)
        source_meta[c_key] = _extract_source_info(c_key, items)

    assert len(sources) >= 2, "fusion variant requires multiple sources"

    canon, _, _ = source_meta[sources[0]]
    for s in sources[1:]:
        other, _, _ = source_meta[s]
        assert other == canon, f"source patterns disagree: {sorted(canon)} vs {sorted(other)}"

    for dr, dc in canon:
        assert (dc, dr) in canon, f"pattern not diagonally symmetric: ({dr},{dc}) has no mirror ({dc},{dr})"

    diag_ks = [dr for dr, dc in canon if dr == dc and dr > 0]
    assert diag_ks, "pattern has no inward-diagonal support"

    clue_zone = frozenset(_clue_offsets(canon))
    assert ACCENT_CANON in clue_zone, "accent cell must be clue-visible"
    assert len(clue_zone) >= 3, "pattern needs at least three non-outward clue cells"

    source_by_pair_color = {}
    source_main_colors = set()
    for c_key in sources:
        _canon, main_color, pair_color = source_meta[c_key]
        assert main_color not in source_main_colors, "source main colors must be unique"
        source_main_colors.add(main_color)
        assert pair_color not in source_by_pair_color, f"duplicate source pair color {pair_color}"
        source_by_pair_color[pair_color] = (c_key, main_color)

    src_set = set(sources)
    targets = []
    target_by_pair_color = {}
    for c_key in corners:
        if c_key in src_set:
            continue
        r, c, kind = c_key
        items = per_corner[c_key]
        if not items:
            continue
        visible = [(off, color) for off, color in items if off in clue_zone]
        if len(visible) != len(items):
            continue
        colors = {color for _, color in visible}
        if len(colors) != 1 or not (2 <= len(visible) <= 3):
            continue
        pair_color = next(iter(colors))
        assert pair_color not in target_by_pair_color, f"duplicate target pair color {pair_color}"
        target_by_pair_color[pair_color] = c_key
        targets.append((r, c, kind, pair_color))

    assert len(targets) >= 2, "fusion variant requires multiple targets"
    assert len(targets) == len(sources), "one-to-one pairing requires equal source/target counts"
    assert set(target_by_pair_color) == set(source_by_pair_color), "target colors must match source accents"

    out = g.copy()
    for pair_color, (src_key, main_color) in source_by_pair_color.items():
        src_r, src_c, src_kind = src_key
        tgt_r, tgt_c, tgt_kind = target_by_pair_color[pair_color]
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, src_kind)
            rr, cc = src_r + adr, src_c + adc
            assert 0 <= rr < H and 0 <= cc < W
            v = int(out[rr, cc])
            assert v in (bg, 0, main_color, pair_color), (
                f"paint conflict at ({rr},{cc}) for source {src_key}: existing {v}"
            )
            out[rr, cc] = pair_color
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, tgt_kind)
            rr, cc = tgt_r + adr, tgt_c + adc
            assert 0 <= rr < H and 0 <= cc < W
            v = int(out[rr, cc])
            assert v in (bg, 0, main_color, pair_color), (
                f"paint conflict at ({rr},{cc}) for target {target_by_pair_color[pair_color]}: existing {v}"
            )
            out[rr, cc] = main_color

    return out


# --------------- generator (output-first pairwise color-swap replication) ---------------

H_RANGE = (18, 24)
W_RANGE = (18, 24)
BORDER = 1
FRAME_GAP = 4
FRAME_SIZE_RANGE = (6, 8)
MAX_ATTEMPTS = 300
N_FRAMES_2_PROB = 0.5
TWO_FRAME_HORIZONTAL_PROB = 0.5

PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
)

_CORNER_OFFSETS = {"TL": (0, 0), "TR": (0, 1), "BL": (1, 0), "BR": (1, 1)}


def _split_total(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out, prev = [], -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _choose_pattern():
    pattern = list(random.choice(PATTERN_TEMPLATES))
    k_max = max(dr for dr, dc in pattern if dr == dc and dr > 0)
    return pattern, k_max


def _draw_frame(grid: np.ndarray, r: int, c: int, size: int) -> None:
    r2, c2 = r + size - 1, c + size - 1
    draw_line(grid, r, c, end_r=r, end_c=c2, color=0)
    draw_line(grid, r, c, end_r=r2, end_c=c, color=0)
    draw_line(grid, r2, c, end_r=r2, end_c=c2, color=0)
    draw_line(grid, r, c2, end_r=r2, end_c=c2, color=0)


def _corner_coord(frame: dict, kind: str):
    dr, dc = _CORNER_OFFSETS[kind]
    s1 = frame["size"] - 1
    return frame["r"] + dr * s1, frame["c"] + dc * s1


def _sample_frame_pair(span: int):
    lo, hi = FRAME_SIZE_RANGE
    pair_limit = span - (2 * BORDER + FRAME_GAP)
    max_first = min(hi, pair_limit - lo)
    if max_first < lo:
        return None
    s1 = random.randint(lo, max_first)
    s2 = random.randint(lo, min(hi, pair_limit - s1))
    slack = span - (2 * BORDER + s1 + FRAME_GAP + s2)
    before, between, _after = _split_total(slack, 3)
    o1 = BORDER + before
    o2 = o1 + s1 + FRAME_GAP + between
    return s1, s2, o1, o2


def _sample_two_frames(H: int, W: int):
    horizontal = random.random() < TWO_FRAME_HORIZONTAL_PROB
    axis = _sample_frame_pair(W if horizontal else H)
    if axis is None:
        return None
    s1, s2, o1, o2 = axis
    if horizontal:
        row = random.randint(BORDER, H - max(s1, s2) - BORDER)
        return [{"r": row, "c": o1, "size": s1}, {"r": row, "c": o2, "size": s2}]
    col = random.randint(BORDER, W - max(s1, s2) - BORDER)
    return [{"r": o1, "c": col, "size": s1}, {"r": o2, "c": col, "size": s2}]


def _sample_three_frames(H: int, W: int):
    lo, hi = FRAME_SIZE_RANGE
    missing = random.choice(CORNER_KINDS)
    present = tuple(k for k in CORNER_KINDS if k != missing)

    def _bands(sizes: dict[str, int]):
        top = max((sizes[s] for s in present if s[0] == "T"), default=0)
        bot = max((sizes[s] for s in present if s[0] == "B"), default=0)
        left = max((sizes[s] for s in present if s[1] == "L"), default=0)
        right = max((sizes[s] for s in present if s[1] == "R"), default=0)
        return top, bot, left, right

    feasible = []
    n_sizes = hi - lo + 1
    for combo in np.ndindex(*(n_sizes for _ in present)):
        sizes = {slot: lo + i for slot, i in zip(present, combo)}
        top, bot, left, right = _bands(sizes)
        if top + FRAME_GAP + bot > H - 2 * BORDER:
            continue
        if left + FRAME_GAP + right > W - 2 * BORDER:
            continue
        feasible.append(sizes)
    if not feasible:
        return None

    sizes = random.choice(feasible)
    top, bot, left, right = _bands(sizes)
    top_extra, mid_y, _ = _split_total(H - (2 * BORDER + top + FRAME_GAP + bot), 3)
    left_extra, mid_x, _ = _split_total(W - (2 * BORDER + left + FRAME_GAP + right), 3)
    top_row = BORDER + top_extra
    bot_row = top_row + top + FRAME_GAP + mid_y
    left_col = BORDER + left_extra
    right_col = left_col + left + FRAME_GAP + mid_x
    coords = {
        "TL": (top_row, left_col),
        "TR": (top_row, right_col),
        "BL": (bot_row, left_col),
        "BR": (bot_row, right_col),
    }
    return [{"r": coords[s][0], "c": coords[s][1], "size": sizes[s]} for s in present]


def _paint_pattern(grid: np.ndarray, r: int, c: int, kind: str, pattern, color: int) -> None:
    for dr, dc in pattern:
        rr, cc = _rot(dr, dc, kind)
        grid[r + rr, c + cc] = color


def _corner_cells(frame: dict, kind: str, pattern) -> set[tuple[int, int]]:
    r, c = _corner_coord(frame, kind)
    cells = set()
    for dr, dc in pattern:
        rr, cc = _rot(dr, dc, kind)
        cells.add((r + rr, c + cc))
    return cells


def _keys_are_clean(frames: list[dict], keys: list[tuple[int, str]], pattern) -> bool:
    per_key_cells = {
        key: _corner_cells(frames[key[0]], key[1], pattern)
        for key in keys
    }
    seen = set()
    for key in keys:
        cells = per_key_cells[key]
        if seen & cells:
            return False
        seen |= cells

    for fi, kind in keys:
        r, c = _corner_coord(frames[fi], kind)
        for other_key, other_cells in per_key_cells.items():
            if other_key == (fi, kind):
                continue
            for rr, cc in other_cells:
                if abs(rr - r) <= _NEIGH_R and abs(cc - c) <= _NEIGH_R:
                    return False
    return True


def _pick_active_corners(frames: list[dict], pattern):
    all_keys = [(fi, kind) for fi in range(len(frames)) for kind in CORNER_KINDS]
    for _ in range(200):
        active = random.sample(all_keys, 2 * N_PAIRS)
        if not _keys_are_clean(frames, active, pattern):
            continue
        sources = set(random.sample(active, N_PAIRS))
        targets = [key for key in active if key not in sources]
        return list(sources), targets
    return None


def _target_clue_candidates(pattern, k_max: int):
    clueable = _clue_offsets(pattern)
    deepest = (k_max, k_max)
    assert ACCENT_CANON in clueable
    all_sets = []
    for size in range(2, min(3, len(clueable)) + 1):
        for subset in combinations(clueable, size):
            all_sets.append(tuple(subset))
    with_deepest = [subset for subset in all_sets if deepest in subset]
    without_deepest = [subset for subset in all_sets if deepest not in subset]
    assert with_deepest and without_deepest
    return all_sets, with_deepest, without_deepest


def _assign_target_clues(target_keys, pattern, k_max: int):
    all_sets, with_deepest, without_deepest = _target_clue_candidates(pattern, k_max)
    triple_sets = [subset for subset in all_sets if len(subset) == 3]
    chosen = {}
    omit_key = random.choice(target_keys)
    other_keys = [key for key in target_keys if key != omit_key]
    deep_key = random.choice(other_keys) if other_keys else omit_key
    for key in target_keys:
        if key == omit_key:
            chosen[key] = random.choice(without_deepest)
        elif key == deep_key and triple_sets:
            chosen[key] = random.choice([subset for subset in triple_sets if (k_max, k_max) in subset] or triple_sets)
        else:
            chosen[key] = random.choice(with_deepest if random.random() < 0.5 else all_sets)
    return chosen


def _pick_pair_colors(bg: int):
    palette = [c for c in range(1, 10) if c != bg]
    colors = random.sample(palette, 2 * N_PAIRS)
    return colors[:N_PAIRS], colors[N_PAIRS:]


def _render_output(base: np.ndarray, frames: list[dict], pattern, pair_specs: list[dict]) -> np.ndarray:
    out = base.copy()
    for spec in pair_specs:
        src_r, src_c = _corner_coord(frames[spec["source"][0]], spec["source"][1])
        tgt_r, tgt_c = _corner_coord(frames[spec["target"][0]], spec["target"][1])
        _paint_pattern(out, src_r, src_c, spec["source"][1], pattern, spec["pair_color"])
        _paint_pattern(out, tgt_r, tgt_c, spec["target"][1], pattern, spec["main_color"])
    return out


def _derive_input(base: np.ndarray, frames: list[dict], pattern, pair_specs: list[dict], target_clues) -> np.ndarray:
    inp = base.copy()
    for spec in pair_specs:
        src_r, src_c = _corner_coord(frames[spec["source"][0]], spec["source"][1])
        _paint_pattern(inp, src_r, src_c, spec["source"][1], pattern, spec["main_color"])
        adr, adc = _rot(*ACCENT_CANON, spec["source"][1])
        inp[src_r + adr, src_c + adc] = spec["pair_color"]

        tgt_r, tgt_c = _corner_coord(frames[spec["target"][0]], spec["target"][1])
        for dr, dc in target_clues[spec["target"]]:
            rr, cc = _rot(dr, dc, spec["target"][1])
            inp[tgt_r + rr, tgt_c + cc] = spec["pair_color"]
    return inp


def _gen_attempt():
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    pattern, k_max = _choose_pattern()

    frames = _sample_two_frames(H, W) if random.random() < N_FRAMES_2_PROB else _sample_three_frames(H, W)
    if frames is None:
        return None

    active = _pick_active_corners(frames, pattern)
    if active is None:
        return None
    sources, targets = active
    target_clues = _assign_target_clues(targets, pattern, k_max)

    source_colors, pair_colors = _pick_pair_colors(bg)
    random.shuffle(sources)
    random.shuffle(targets)
    pair_specs = []
    for source_key, target_key, main_color, pair_color in zip(sources, targets, source_colors, pair_colors):
        pair_specs.append(
            {
                "source": source_key,
                "target": target_key,
                "main_color": main_color,
                "pair_color": pair_color,
            }
        )

    base = np.full((H, W), bg, dtype=int)
    for frame in frames:
        _draw_frame(base, frame["r"], frame["c"], frame["size"])

    output = _render_output(base, frames, pattern, pair_specs)
    inp = _derive_input(base, frames, pattern, pair_specs, target_clues)

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
