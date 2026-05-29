# Combo variant of a251c730 (index 24)
# combo_spec: D7/0+D10/1
# Name: 一对多+迭代印刻

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for this combo variant
# --------------------------------------
# a) Concrete behavior trace:
#    - Original a251c730 is one-shot and effectively one-to-one: each singleton
#      marker in the destination matches one source sprite anchor color, and the
#      matched sprite is stamped once at that marker.
#    - This combo keeps the same two-rectangle wallpaper layout, but changes the
#      destination dynamics. Example: source has three sprite types with anchor
#      colors A0, A1, A2. Type A0 also contains two relay pixels colored A1,
#      and type A1 contains two relay pixels colored A2.
#      Input destination may show two A0 markers and nothing else.
#      Round 1: stamp the A0 sprite at both A0 markers.
#      Round 2: each fresh A1 relay pixel now behaves as a marker, so the A1
#      sprite is stamped at every such location.
#      Round 3: newly created A2 relay pixels trigger every A2 stamp.
#      Stop when no unseen anchor-colored pixels remain.
#    - Difference from the original: the same root sprite is copied to several
#      starting locations, and later markers are not present in the input at all;
#      they are created by earlier stamping rounds.
#
# b) Complexity calibration:
#    - Keep only 2-3 sprite types so the iterative rule is visible without
#      becoming combinatorially noisy.
#    - Force 2-3 repeated root markers of a single anchor color. This makes the
#      one-to-many relation obvious in every sample.
#    - Use sprite templates with exactly one unique anchor pixel and two relay
#      pixels. That gives visible multi-round growth while keeping the cascade
#      compact enough to fit inside the destination rectangle.
#
# c) Hidden-assumption audit:
#    - Original code assumed a single pass over destination markers was enough.
#      Broken here because later markers are created during stamping.
#    - Original code implicitly tolerated repeated marker colors but did not make
#      that behavior essential. Here repeated root markers are mandatory.
#    - Original anchor extraction used a color-id tiebreak among least-frequent
#      colors. That is not visually justified, so the combo requires a uniquely
#      rare anchor color per source sprite.
#    - The iterative solver now assumes any anchor-colored pixel stamped in the
#      destination can become a next-round marker unless it was already used at
#      that exact coordinate and color. The generator guarantees relay colors are
#      always other sprite anchors, not arbitrary body colors.
#
# d) Placement legality:
#    - Generation stays output-first. It first simulates the whole cascade from a
#      single root marker, crops that footprint, and only then places several
#      non-overlapping copies of that full footprint inside the destination.
#    - Because the whole future footprint is reserved up front, every later round
#      remains in bounds and different repeated roots do not interfere.
#    - Source sprites are still placed with padding, so connected components stay
#      separable and template recovery remains stable.
#
# e) Visual observability:
#    - A human sees repeated identical root markers in the destination input, so
#      the one-to-many part is explicit.
#    - The output contains additional sprite types centered on colors that only
#      appear after the first round, so one-pass stamping is visibly insufficient.
#    - The same chain structure appears in multiple places in one sample because
#      several identical roots spawn parallel cascades.
#
# f) Visual cleanliness:
#    - Cascades from different roots are disjoint at the full-footprint level,
#      preventing unrelated overlap clutter.
#    - Overlap is limited to the intended parent/child interaction within a
#      single cascade, which is the visual evidence for the iterative rule.
#    - Borders, wallpaper, anchors, and body colors remain distinct where needed.
#
# g) Inferability of the change:
#    - Colors are only used through visible equality. A relay pixel has exactly
#      the same color as the next sprite's anchor in the source rectangle and as
#      the generated next-round marker in the destination.
#    - The generator rejects degenerate cases where full iterative stamping would
#      equal the first round only, so D10 behavior is never accidental or absent.
#
# Main() assumptions guaranteed by generate_input():
#    - Exactly two bordered rectangles exist: a source rectangle with multi-cell
#      sprites and a destination rectangle with singleton seed markers.
#    - Each source sprite type has one uniquely least-frequent anchor color.
#    - Body colors never equal any anchor color; relay colors are always anchor
#      colors of later sprite types in the chain.
#    - The destination input contains 2-3 repeated root markers, all valid and
#      fully in bounds for their eventual cascades.


GRID_SIZE = 30
GAP_RANGE = (1, 4)

SRC_H_RANGE = (10, 20)
SRC_W_RANGE = (9, 18)
DST_H_RANGE = (10, 19)
DST_W_RANGE = (9, 20)
DST_W_SIDE_MAX = 20

SPRITE_TYPE_CHOICES = (2, 3)
INITIAL_MARKER_COUNT_RANGE = (2, 3)
SOURCE_SPRITE_RANGE = (3, 5)
SOURCE_SPRITE_PADDING = 1
MAX_GEN_ATTEMPTS = 400
# Three-stage cascades already consume noticeably more area than two-stage ones.
# Keeping them at two repeated roots avoids clutter and makes the multi-round
# rule easier to read, while still covering both D7 multiplicity values overall.
BRANCH_TYPE_ROOT_PAIRS = ((2, 2), (2, 3), (3, 2))

CHAIN_SPRITE_SHAPES = [
    (("AR.", "BBR"), (0, 0)),
    ((".A.", "RBR", ".B."), (0, 1)),
    (("BRA", "RBB"), (0, 2)),
    (("B.R", "RAB"), (1, 1)),
    (("RAB", "BBR"), (0, 1)),
    (("B.B", "RBR", ".A."), (2, 1)),
]


def _build_frame(h: int, w: int, border: int, bg: int) -> np.ndarray:
    rect = np.full((h, w), int(bg), dtype=int)
    rect[0, :] = rect[-1, :] = int(border)
    rect[:, 0] = rect[:, -1] = int(border)
    return rect


def _render_sprite(shape_rows: tuple[str, ...], colors: dict[str, int], bg: int) -> np.ndarray:
    char_map = {"A": int(colors["A"]), "R": int(colors["R"]), "B": int(colors["B"])}
    rows = [[char_map.get(ch, int(bg)) for ch in row] for row in shape_rows]
    return np.array(rows, dtype=int)


def _make_wallpaper(direction: str, palette: list[int]) -> np.ndarray:
    period = len(palette)
    stripe = np.array(palette * ((GRID_SIZE + period - 1) // period), dtype=int)[:GRID_SIZE]
    if direction == "vertical":
        return np.tile(stripe, (GRID_SIZE, 1))
    return np.tile(stripe.reshape(-1, 1), (1, GRID_SIZE))


def _choose_counts(n_total: int, n_bins: int) -> list[int]:
    counts = [1] * int(n_bins)
    for _ in range(int(n_total) - int(n_bins)):
        counts[random.randrange(int(n_bins))] += 1
    return counts


def _choose_dimensions(layout: str) -> tuple[int, int, int, int]:
    gap = random.randint(*GAP_RANGE)
    if layout == "side_by_side":
        src_w = random.randint(*SRC_W_RANGE)
        dst_w_max = min(DST_W_SIDE_MAX, GRID_SIZE - gap - src_w)
        if dst_w_max < DST_W_RANGE[0]:
            raise ValueError("no side-by-side width fit")
        dst_w = random.randint(DST_W_RANGE[0], dst_w_max)
        src_h = random.randint(*SRC_H_RANGE)
        dst_h = random.randint(*DST_H_RANGE)
    else:
        src_h = random.randint(*SRC_H_RANGE)
        dst_h_max = min(DST_H_RANGE[1], GRID_SIZE - gap - src_h)
        if dst_h_max < DST_H_RANGE[0]:
            raise ValueError("no stacked height fit")
        dst_h = random.randint(DST_H_RANGE[0], dst_h_max)
        src_w = random.randint(*SRC_W_RANGE)
        dst_w = random.randint(*DST_W_RANGE)
    return int(src_h), int(src_w), int(dst_h), int(dst_w)


def _place_rectangles(
    layout: str, src_h: int, src_w: int, dst_h: int, dst_w: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    if layout == "side_by_side":
        gap = random.randint(1, GRID_SIZE - src_w - dst_w)
        slack = GRID_SIZE - src_w - dst_w - gap
        left_margin = random.randint(0, slack)
        src_r = random.randint(0, GRID_SIZE - src_h)
        dst_r = random.randint(0, GRID_SIZE - dst_h)
        if random.random() < 0.5:
            src_c, dst_c = left_margin, left_margin + src_w + gap
        else:
            dst_c, src_c = left_margin, left_margin + dst_w + gap
    else:
        gap = random.randint(1, GRID_SIZE - src_h - dst_h)
        slack = GRID_SIZE - src_h - dst_h - gap
        top_margin = random.randint(0, slack)
        src_c = random.randint(0, GRID_SIZE - src_w)
        dst_c = random.randint(0, GRID_SIZE - dst_w)
        if random.random() < 0.5:
            src_r, dst_r = top_margin, top_margin + src_h + gap
        else:
            dst_r, src_r = top_margin, top_margin + dst_h + gap
    return (int(src_r), int(src_c)), (int(dst_r), int(dst_c))


def _find_problem_rectangles(g: np.ndarray) -> tuple[tuple, tuple]:
    H, W = g.shape

    def interior_bg(rect: tuple[int, int, int, int]) -> int:
        r0, c0, r1, c1 = map(int, rect)
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        return int(Counter(map(int, inner.flatten())).most_common(1)[0][0])

    def rect_components(rect: tuple[int, int, int, int]) -> tuple[int, list[dict]]:
        r0, c0, r1, c1 = map(int, rect)
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        ibg = int(Counter(map(int, inner.flatten())).most_common(1)[0][0])
        comps = []
        for coords in components_mask(inner != int(ibg), connectivity=4):
            if coords.size == 0:
                continue
            rr0, cc0, rr1, cc1 = bbox_coords(coords)
            mask = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
            mask[coords[:, 0] - int(rr0), coords[:, 1] - int(cc0)] = True
            patch = inner[int(rr0) : int(rr1) + 1, int(cc0) : int(cc1) + 1].copy()
            patch[~mask] = 0
            comps.append(
                {
                    "size": int(coords.shape[0]),
                    "colors": dict(Counter(int(inner[int(r), int(c)]) for r, c in coords)),
                    "mask": mask,
                    "patch": patch,
                }
            )
        comps.sort(key=lambda d: -int(d["size"]))
        return int(ibg), comps

    rects: list[tuple[int, tuple[int, int, int, int], int, int]] = []
    for r0 in range(int(H)):
        for c0 in range(int(W)):
            bc = int(g[int(r0), int(c0)])
            for r1 in range(int(r0 + 5), int(H)):
                for c1 in range(int(c0 + 5), int(W)):
                    if int(g[int(r0), int(c1)]) != int(bc) or int(g[int(r1), int(c0)]) != int(bc):
                        continue
                    if int(g[int(r1), int(c1)]) != int(bc):
                        continue
                    if not bool((g[int(r0), int(c0) : int(c1) + 1] == int(bc)).all()):
                        continue
                    if not bool((g[int(r1), int(c0) : int(c1) + 1] == int(bc)).all()):
                        continue
                    if not bool((g[int(r0) : int(r1) + 1, int(c0)] == int(bc)).all()):
                        continue
                    if not bool((g[int(r0) : int(r1) + 1, int(c1)] == int(bc)).all()):
                        continue
                    ibg = interior_bg((int(r0), int(c0), int(r1), int(c1)))
                    if int(ibg) == int(bc):
                        continue
                    area = int((int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1))
                    rects.append((area, (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))

    assert rects
    rects.sort(reverse=True, key=lambda x: int(x[0]))
    rects = rects[:6]

    src = None
    dst = None
    for _area, rect, bc, _ibg in rects:
        ibg2, comps = rect_components(rect)
        max_sz = max((int(c["size"]) for c in comps), default=0)
        if max_sz > 1 and src is None:
            src = (rect, int(bc), int(ibg2), comps)
        if max_sz == 1 and comps and dst is None:
            dst = (rect, int(bc), int(ibg2), comps)

    assert src is not None and dst is not None
    return src, dst


def _extract_templates_from_components(src_comps: list[dict]) -> dict[int, dict]:
    templates: dict[int, dict] = {}
    for comp in src_comps:
        cols = {int(k): int(v) for k, v in comp["colors"].items() if int(k) != 0}
        if not cols:
            continue
        min_count = min(int(v) for v in cols.values())
        anchor_candidates = [int(col) for col, cnt in cols.items() if int(cnt) == int(min_count)]
        if len(anchor_candidates) != 1:
            continue
        anchor_col = int(anchor_candidates[0])
        patch = comp["patch"].copy()
        mask = comp["mask"].copy()
        anchor_coords = np.argwhere((patch == int(anchor_col)) & mask)
        if anchor_coords.shape[0] != 1:
            continue
        ar, ac = map(int, anchor_coords[0])
        if anchor_col not in templates or int(mask.sum()) > int(templates[anchor_col]["mask"].sum()):
            templates[anchor_col] = {
                "patch": patch,
                "mask": mask,
                "anchor_rc": (int(ar), int(ac)),
            }

    anchor_colors = set(int(k) for k in templates)
    for anchor_col, tpl in list(templates.items()):
        trigger_cells = []
        for rr, cc in np.argwhere(tpl["mask"]):
            color = int(tpl["patch"][int(rr), int(cc)])
            if color in anchor_colors:
                trigger_cells.append((int(rr), int(cc)))
        tpl["trigger_cells"] = trigger_cells
        templates[int(anchor_col)] = tpl
    return templates


def _simulate_iterative_stamp(
    templates: dict[int, dict],
    initial_markers: list[tuple[int, int, int]],
    canvas_h: int,
    canvas_w: int,
    bg: int,
    *,
    max_rounds: int | None = None,
) -> tuple[np.ndarray, dict] | None:
    out = np.full((int(canvas_h), int(canvas_w)), int(bg), dtype=int)
    processed: set[tuple[int, int, int]] = set()
    frontier = sorted({(int(r), int(c), int(col)) for r, c, col in initial_markers})
    anchor_colors = set(int(k) for k in templates)
    round_sizes: list[int] = []
    total_stamps = 0
    initial_count = len(frontier)

    while frontier and (max_rounds is None or len(round_sizes) < int(max_rounds)):
        frontier = [m for m in frontier if m not in processed and int(m[2]) in templates]
        if not frontier:
            break
        round_sizes.append(len(frontier))
        next_candidates: set[tuple[int, int, int]] = set()
        for mr, mc, mcol in frontier:
            tpl = templates[int(mcol)]
            patch = tpl["patch"]
            mask = tpl["mask"]
            ar, ac = tpl["anchor_rc"]
            top = int(mr - ar)
            left = int(mc - ac)
            h, w = map(int, patch.shape)
            if top < 0 or left < 0 or top + h > int(canvas_h) or left + w > int(canvas_w):
                return None

            before = out[top : top + h, left : left + w].copy()
            paste(out, patch, int(top), int(left), mask=mask, overwrite=True, clip=False)
            after = out[top : top + h, left : left + w]

            for rr, cc in tpl["trigger_cells"]:
                new_col = int(after[int(rr), int(cc)])
                if new_col not in anchor_colors:
                    continue
                if int(before[int(rr), int(cc)]) == int(new_col):
                    continue
                next_candidates.add((int(top + rr), int(left + cc), int(new_col)))

            processed.add((int(mr), int(mc), int(mcol)))
            total_stamps += 1

        frontier = sorted(
            cand
            for cand in next_candidates
            if cand not in processed and int(out[int(cand[0]), int(cand[1])]) == int(cand[2])
        )

    stats = {
        "rounds": int(len(round_sizes)),
        "round_sizes": [int(x) for x in round_sizes],
        "initial_markers": int(initial_count),
        "total_stamps": int(total_stamps),
        "spawned_stamps": int(max(0, total_stamps - initial_count)),
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp source sprites into every matching destination marker, then keep propagating whenever stamped anchor-colored pixels create new markers.

    Input structure:
    - Two bordered rectangles sit on a striped wallpaper: the source rectangle contains multi-cell sprites, while the destination rectangle contains only singleton seed markers.
    - Each source sprite has one uniquely rare anchor color, and some non-anchor pixels may reuse other sprite anchors as relay colors. A root anchor color may appear at several destination markers.

    Rule:
    - Recover one template per source sprite type using its unique anchor color and anchor location.
    - In the destination, stamp the matching template at every current marker of the same color, not just one of them.
    - After each round, treat every newly stamped anchor-colored relay pixel as a fresh marker for the next round, and repeat until no unseen markers remain. Return the cropped destination rectangle after the full cascade.

    Key insight: Anchor colors now do double duty across time as well as space: repeated visible markers trigger repeated root copies, and anchor-colored pixels created by those copies recursively trigger later sprite templates.
    """
    g = np.array(grid, dtype=int)
    src, dst = _find_problem_rectangles(g)

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

    templates = _extract_templates_from_components(src_comps)
    assert templates

    inner = g[int(dr) + 1 : int(dr1), int(dc) + 1 : int(dc1)]
    initial_markers = [(int(rr), int(cc), int(inner[int(rr), int(cc)])) for rr, cc in np.argwhere(inner != int(dst_bg))]

    sim = _simulate_iterative_stamp(templates, initial_markers, inner.shape[0], inner.shape[1], int(dst_bg))
    assert sim is not None
    out_inner, _stats = sim

    out = _build_frame(int(dr1 - dr + 1), int(dc1 - dc + 1), int(dst_bc), int(dst_bg))
    paste(out[1:-1, 1:-1], out_inner, 0, 0)
    return out


def _assign_palette(n_types: int) -> tuple[list[int], int, int, list[int], list[int]] | None:
    wallpaper_n = 4 if int(n_types) == 3 else random.choice((4, 5))
    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, wallpaper_n)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)
    if len(remaining) < int(n_types) + 1:
        return None
    src_border = int(remaining.pop())
    anchors = [int(remaining.pop()) for _ in range(int(n_types))]
    if remaining and random.random() < 0.5:
        dst_border = int(remaining.pop())
    else:
        border_choices = [c for c in wallpaper if int(c) != int(src_border)]
        if not border_choices:
            return None
        dst_border = int(random.choice(border_choices))
    extras = [int(c) for c in remaining]
    return [int(c) for c in wallpaper], int(src_border), int(dst_border), anchors, extras


def _sprite_templates_from_types(sprite_types: list[dict], bg: int) -> dict[int, dict]:
    templates: dict[int, dict] = {}
    anchor_colors = {int(s["anchor"]) for s in sprite_types}
    for sprite in sprite_types:
        patch = _render_sprite(
            sprite["shape_rows"],
            {"A": sprite["anchor"], "R": sprite["relay"], "B": sprite["body"]},
            int(bg),
        )
        mask = patch != int(bg)
        trigger_cells = []
        for rr, cc in np.argwhere(mask):
            color = int(patch[int(rr), int(cc)])
            if color in anchor_colors:
                trigger_cells.append((int(rr), int(cc)))
        templates[int(sprite["anchor"])] = {
            "patch": patch,
            "mask": mask,
            "anchor_rc": tuple(map(int, sprite["anchor_rc"])),
            "trigger_cells": trigger_cells,
        }
    return templates


def _place_sources(sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int) -> np.ndarray | None:
    inner = np.full((int(src_h) - 2, int(src_w) - 2), int(src_bg), dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[int(type_id)]
        patch = _render_sprite(
            sprite["shape_rows"],
            {"A": sprite["anchor"], "R": sprite["relay"], "B": sprite["body"]},
            int(src_bg),
        )
        try:
            r0, c0 = random_free_location(
                inner,
                patch,
                transparent=int(src_bg),
                border=SOURCE_SPRITE_PADDING,
                padding=SOURCE_SPRITE_PADDING,
            )
        except ValueError:
            return None
        paste(inner, patch, int(r0), int(c0), transparent=int(src_bg))
    return inner


def _single_root_footprint(sprite_types: list[dict], dst_bg: int) -> tuple[np.ndarray, tuple[int, int], dict] | None:
    templates = _sprite_templates_from_types(sprite_types, int(dst_bg))
    center = 12
    sim = _simulate_iterative_stamp(
        templates,
        [(int(center), int(center), int(sprite_types[0]["anchor"]))],
        25,
        25,
        int(dst_bg),
    )
    if sim is None:
        return None
    out, stats = sim
    if int(stats["rounds"]) < 2 or int(stats["spawned_stamps"]) < 1:
        return None
    coords = np.argwhere(out != int(dst_bg))
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()
    return patch, (int(center - r0), int(center - c0)), stats


def _place_root_markers(
    sprite_types: list[dict],
    dst_h: int,
    dst_w: int,
    dst_bg: int,
    n_roots: int,
) -> tuple[list[tuple[int, int, int]], np.ndarray, dict] | None:
    footprint_info = _single_root_footprint(sprite_types, int(dst_bg))
    if footprint_info is None:
        return None
    footprint_patch, root_rel, base_stats = footprint_info
    inner_h, inner_w = int(dst_h - 2), int(dst_w - 2)
    placement = np.full((inner_h, inner_w), int(dst_bg), dtype=int)
    markers: list[tuple[int, int, int]] = []

    for _ in range(int(n_roots)):
        try:
            top, left = random_free_location(
                placement,
                footprint_patch,
                transparent=int(dst_bg),
                padding=1,
            )
        except ValueError:
            return None
        paste(placement, footprint_patch, int(top), int(left), transparent=int(dst_bg))
        markers.append((int(top + root_rel[0]), int(left + root_rel[1]), int(sprite_types[0]["anchor"])))

    templates = _sprite_templates_from_types(sprite_types, int(dst_bg))
    sim = _simulate_iterative_stamp(templates, markers, inner_h, inner_w, int(dst_bg))
    if sim is None:
        return None
    out_inner, stats = sim
    one_round = _simulate_iterative_stamp(templates, markers, inner_h, inner_w, int(dst_bg), max_rounds=1)
    if one_round is None:
        return None
    first_only, _ = one_round
    if np.array_equal(first_only, out_inner):
        return None

    stats["base_rounds"] = int(base_stats["rounds"])
    stats["root_repetitions"] = int(n_roots)
    stats["first_round_diff"] = int(np.count_nonzero(first_only != out_inner))
    return markers, out_inner, stats


def _gen_attempt(
    return_meta: bool = False,
    *,
    forced_n_types: int | None = None,
    forced_n_roots: int | None = None,
    forced_layout: str | None = None,
):
    n_types = int(forced_n_types) if forced_n_types is not None else random.choice(list(SPRITE_TYPE_CHOICES))
    palette = _assign_palette(int(n_types))
    if palette is None:
        return None
    wallpaper, src_border, dst_border, anchors, extras = palette

    src_bg = int(random.choice(wallpaper))
    dst_bg_choices = [c for c in wallpaper if int(c) != int(src_bg) and int(c) != int(dst_border)]
    if not dst_bg_choices:
        return None
    dst_bg = int(random.choice(dst_bg_choices))

    body_choices = [
        int(c)
        for c in wallpaper + extras
        if int(c) not in anchors and int(c) not in {int(src_bg), int(dst_bg), int(src_border), int(dst_border)}
    ]
    if not body_choices:
        return None

    sprite_specs = random.sample(CHAIN_SPRITE_SHAPES, int(n_types))
    body_colors = [int(random.choice(body_choices)) for _ in range(int(n_types))]
    sprite_types = []
    for i, ((shape_rows, anchor_rc), body) in enumerate(zip(sprite_specs, body_colors)):
        relay = int(anchors[i + 1]) if i + 1 < int(n_types) else int(body)
        sprite_types.append(
            {
                "shape_rows": shape_rows,
                "anchor_rc": tuple(map(int, anchor_rc)),
                "anchor": int(anchors[i]),
                "relay": int(relay),
                "body": int(body),
                "h": len(shape_rows),
                "w": len(shape_rows[0]),
            }
        )

    n_source = random.randint(max(int(n_types), SOURCE_SPRITE_RANGE[0]), SOURCE_SPRITE_RANGE[1])
    source_type_ids = [i for i, ct in enumerate(_choose_counts(int(n_source), int(n_types))) for _ in range(int(ct))]
    random.shuffle(source_type_ids)

    layout = str(forced_layout) if forced_layout is not None else random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if int(src_h) * int(src_w) <= int(dst_h) * int(dst_w):
        return None

    n_roots = int(forced_n_roots) if forced_n_roots is not None else random.randint(*INITIAL_MARKER_COUNT_RANGE)
    marker_result = _place_root_markers(sprite_types, int(dst_h), int(dst_w), int(dst_bg), int(n_roots))
    if marker_result is None:
        return None
    marker_points, out_inner, stats = marker_result

    output = _build_frame(int(dst_h), int(dst_w), int(dst_border), int(dst_bg))
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_inner = _place_sources(sprite_types, source_type_ids, int(src_h), int(src_w), int(src_bg))
    if source_inner is None:
        return None

    source_rect = _build_frame(int(src_h), int(src_w), int(src_border), int(src_bg))
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(int(dst_h), int(dst_w), int(dst_border), int(dst_bg))
    for mr, mc, _col in marker_points:
        dest_rect[1 + int(mr), 1 + int(mc)] = int(sprite_types[0]["anchor"])

    try:
        src_pos, dst_pos = _place_rectangles(layout, int(src_h), int(src_w), int(dst_h), int(dst_w))
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, int(src_pos[0]), int(src_pos[1]))
    paste(grid, dest_rect, int(dst_pos[0]), int(dst_pos[1]))

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, output):
        return None

    if return_meta:
        meta = {
            "n_types": int(n_types),
            "n_roots": int(n_roots),
            "rounds": int(stats["rounds"]),
            "round_sizes": [int(x) for x in stats["round_sizes"]],
            "spawned_stamps": int(stats["spawned_stamps"]),
            "total_stamps": int(stats["total_stamps"]),
            "first_round_diff": int(stats["first_round_diff"]),
            "root_repetitions": int(stats["root_repetitions"]),
            "layout": layout,
        }
        return grid, output, meta
    return grid


def generate_input() -> np.ndarray:
    primary_branch = random.choice(BRANCH_TYPE_ROOT_PAIRS)
    other_branches = [spec for spec in BRANCH_TYPE_ROOT_PAIRS if spec != primary_branch]
    random.shuffle(other_branches)
    branch_specs = [primary_branch] + other_branches

    primary_budget = max(1, MAX_GEN_ATTEMPTS // 2)
    remaining_budget = MAX_GEN_ATTEMPTS - primary_budget
    fallback_budget = max(1, remaining_budget // max(1, len(other_branches)))

    for i, (n_types, n_roots) in enumerate(branch_specs):
        branch_budget = primary_budget if i == 0 else fallback_budget
        if i == len(branch_specs) - 1:
            used_budget = primary_budget + fallback_budget * max(0, len(branch_specs) - 2)
            branch_budget = max(1, MAX_GEN_ATTEMPTS - used_budget)
        for _ in range(branch_budget):
            result = _gen_attempt(return_meta=False, forced_n_types=int(n_types), forced_n_roots=int(n_roots))
            if result is not None:
                return result
    raise RuntimeError(f"failed to generate a251c730 combo input after {MAX_GEN_ATTEMPTS} attempts")
