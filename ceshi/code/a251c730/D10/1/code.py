# Variant of a251c730
# Dimension: D10 — 多步处理：印刻顺序与覆盖 (multi_step)
# Change: 所有精灵一次性印刻，无特定顺序要求（精灵间不重叠）... → 迭代处理：印刻后生成新标记

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVERS = _ROOT / "solvers"
if str(_SOLVERS) not in sys.path:
    sys.path.append(str(_SOLVERS))

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis for the D10 variant
# ----------------------------------
# a) Concrete behavior
#    - The global layout stays the same: striped wallpaper, one source rectangle with several multi-cell sprites,
#      and one destination rectangle that initially contains only singleton markers.
#    - The change is inside the stamping rule. Each sprite now has:
#        A = one anchor pixel (least frequent color, used to identify the template),
#        R = one or two relay pixels whose color equals the anchor color of the next sprite type,
#        B = filler body pixels that do not trigger anything.
#    - The destination input only places markers for the first sprite type. After round 1, newly stamped relay
#      pixels act as round-2 markers; round 2 creates more relay pixels, and so on until the chain tail.
#    - Small example:
#        type0 uses colors A0/R=A1/B=x, type1 uses A1/R=A2/B=y, type2 uses A2/B=z.
#        Original task: marker A0 stamps only type0 once.
#        New task: marker A0 stamps type0; its A1 relay pixels become markers; these stamp type1; their A2 relay
#        pixels become markers; these stamp type2; no more anchor-colored relay pixels remain, so the process stops.
#
# b) Complexity calibration
#    - The chain length is limited to 2 or 3 sprite types, so the rule is deeper than the original but still
#      inferable from 2-4 examples.
#    - Relay branching is limited by using only 2 relay pixels per sprite; this keeps the output readable and avoids
#      explosive growth.
#    - Initial markers are only for the chain root, so one-pass stamping is visibly insufficient.
#
# c) Original assumptions that break
#    - Original code scans the destination markers once and stamps immediately. That assumes all usable markers are
#      present in the input before any stamping. This is false here because later markers are produced by earlier
#      rounds.
#    - Original code can ignore overwrite order because sprites do not interact. Here later rounds intentionally
#      overwrite earlier pixels, so we need explicit round-by-round processing.
#    - Original anchor selection broke ties by color id. This variant avoids color-id-driven behavior by generating
#      sprites whose anchor is the unique least-frequent color.
#
# d) Placement legality
#    - The generator precomputes the full cascade footprint of one root marker, then places translated copies of that
#      whole footprint inside the destination interior. This guarantees that every later-round stamp also stays in
#      bounds and keeps different cascades separated.
#    - Source sprites are still placed with padding inside the source rectangle so the solver can recover each
#      template component cleanly.
#
# e) Visual observability
#    - The relay mechanism is visible because the intermediate anchor colors appear inside already stamped sprites and
#      then cause new sprites to appear centered on those pixels in the next round.
#    - Since the input contains only root markers, the difference from the original one-pass rule is directly visible
#      in the output.
#
# f) Visual cleanliness
#    - Cascades from different roots are placed disjointly at the footprint level, so clutter is limited to the
#      intended parent/child overlap inside one cascade.
#    - Borders, backgrounds, anchors, and relay/body colors stay distinct where needed.
#
# g) Color inference
#    - No hidden ordering uses color ids. Colors matter only through visible equality: a relay pixel reuses the same
#      color as the next sprite's anchor in the source rectangle and as the generated marker in the destination.

# --------------- shared helpers ---------------

GRID_SIZE = 30
GAP_RANGE = (1, 4)

SRC_H_RANGE = (10, 20)
SRC_W_RANGE = (9, 18)
DST_H_RANGE = (10, 19)
DST_W_RANGE = (9, 20)
DST_W_SIDE_MAX = 20

SPRITE_TYPE_CHOICES = (2, 3)
INITIAL_MARKER_COUNT_RANGE = (1, 2)
SOURCE_SPRITE_RANGE = (3, 5)
MARKER_PLACE_RETRIES = 30
SOURCE_SPRITE_PADDING = 1
MAX_GEN_ATTEMPTS = 400

CHAIN_SPRITE_SHAPES = [
    (("AR.", "BBR"), (0, 0)),
    ((".A.", "RBR", ".B."), (0, 1)),
    (("BRA", "RBB"), (0, 2)),
    (("B.R", "RAB"), (1, 1)),
    (("RAB", "BBR"), (0, 1)),
    (("B.B", "RBR", ".A."), (2, 1)),
]


def _render_sprite(shape_rows: tuple[str, ...], colors: dict[str, int], bg: int) -> np.ndarray:
    char_map = {"A": colors["A"], "R": colors["R"], "B": colors["B"]}
    rows = [[char_map.get(ch, bg) for ch in row] for row in shape_rows]
    return np.array(rows, dtype=int)


def _make_wallpaper(direction: str, palette: list[int]) -> np.ndarray:
    period = len(palette)
    stripe = np.array(palette * ((GRID_SIZE + period - 1) // period), dtype=int)[:GRID_SIZE]
    if direction == "vertical":
        return np.tile(stripe, (GRID_SIZE, 1))
    return np.tile(stripe.reshape(-1, 1), (1, GRID_SIZE))


def _build_frame(h: int, w: int, border: int, bg: int) -> np.ndarray:
    rect = np.full((h, w), bg, dtype=int)
    rect[0, :] = rect[-1, :] = border
    rect[:, 0] = rect[:, -1] = border
    return rect


def _choose_counts(n_total: int, n_bins: int) -> list[int]:
    counts = [1] * n_bins
    for _ in range(n_total - n_bins):
        counts[random.randrange(n_bins)] += 1
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
    return src_h, src_w, dst_h, dst_w


def _place_rectangles(layout: str, src_h: int, src_w: int, dst_h: int, dst_w: int) -> tuple[tuple[int, int], tuple[int, int]]:
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
    return (src_r, src_c), (dst_r, dst_c)


def _sprite_templates_from_types(sprite_types: list[dict], bg: int) -> dict[int, dict]:
    templates: dict[int, dict] = {}
    anchor_colors = {int(s["anchor"]) for s in sprite_types}
    for sprite in sprite_types:
        patch = _render_sprite(sprite["shape_rows"], {"A": sprite["anchor"], "R": sprite["relay"], "B": sprite["body"]}, bg)
        mask = patch != int(bg)
        trigger_cells = []
        for rr, cc in np.argwhere(mask):
            col = int(patch[int(rr), int(cc)])
            if col in anchor_colors:
                trigger_cells.append((int(rr), int(cc)))
        templates[int(sprite["anchor"])] = {
            "patch": patch,
            "mask": mask,
            "anchor_rc": tuple(map(int, sprite["anchor_rc"])),
            "trigger_cells": trigger_cells,
        }
    return templates


def _simulate_chain_from_templates(
    templates: dict[int, dict],
    initial_markers: list[tuple[int, int, int]],
    canvas_h: int,
    canvas_w: int,
    bg: int,
    *,
    max_rounds: int | None = None,
) -> tuple[np.ndarray, dict] | None:
    out = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    processed: set[tuple[int, int, int]] = set()
    frontier = sorted({(int(r), int(c), int(col)) for r, c, col in initial_markers})
    round_sizes: list[int] = []
    total_stamps = 0
    anchor_colors = set(int(k) for k in templates)
    initial_count = len(frontier)

    while frontier and (max_rounds is None or len(round_sizes) < max_rounds):
        frontier = [m for m in frontier if m not in processed and int(m[2]) in templates]
        if not frontier:
            break
        round_sizes.append(len(frontier))
        next_candidates: set[tuple[int, int, int]] = set()
        for mr, mc, mcol in frontier:
            tpl = templates[int(mcol)]
            patch = tpl["patch"]
            ar, ac = tpl["anchor_rc"]
            top = int(mr - ar)
            left = int(mc - ac)
            h, w = map(int, patch.shape)
            if top < 0 or left < 0 or top + h > canvas_h or left + w > canvas_w:
                return None
            before = out[top : top + h, left : left + w].copy()
            paste(out, patch, top, left, transparent=int(bg))
            after = out[top : top + h, left : left + w]
            for rr, cc in tpl["trigger_cells"]:
                new_col = int(after[int(rr), int(cc)])
                if new_col not in anchor_colors:
                    continue
                if int(before[int(rr), int(cc)]) == new_col:
                    continue
                next_candidates.add((top + int(rr), left + int(cc), new_col))
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
            mask = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
            mask[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
            patch = inner[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
            patch[~mask] = 0
            vals = [int(inner[int(r), int(c)]) for r, c in coords]
            comps.append(
                {
                    "size": int(coords.shape[0]),
                    "colors": dict(Counter(vals)),
                    "bbox": (int(rr0), int(cc0), int(rr1), int(cc1)),
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
                    if int(g[int(r0), int(c1)]) != bc or int(g[int(r1), int(c0)]) != bc or int(g[int(r1), int(c1)]) != bc:
                        continue
                    if not bool((g[int(r0), int(c0) : int(c1) + 1] == bc).all()):
                        continue
                    if not bool((g[int(r1), int(c0) : int(c1) + 1] == bc).all()):
                        continue
                    if not bool((g[int(r0) : int(r1) + 1, int(c0)] == bc).all()):
                        continue
                    if not bool((g[int(r0) : int(r1) + 1, int(c1)] == bc).all()):
                        continue
                    ibg = interior_bg((int(r0), int(c0), int(r1), int(c1)))
                    if ibg == bc:
                        continue
                    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
                    rects.append((area, (int(r0), int(c0), int(r1), int(c1)), bc, ibg))

    assert rects
    rects.sort(reverse=True, key=lambda x: int(x[0]))
    rects = rects[:6]

    src = None
    dst = None
    for _area, rect, bc, _ibg in rects:
        ibg2, comps = rect_components(rect)
        max_sz = max([int(c["size"]) for c in comps], default=0)
        if max_sz > 1 and src is None:
            src = (rect, bc, ibg2, comps)
        if max_sz == 1 and len(comps) > 0 and dst is None:
            dst = (rect, bc, ibg2, comps)
    assert src is not None and dst is not None
    return src, dst


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate source sprites through a destination frame by repeatedly turning stamped relay pixels into new markers.

    Input structure:
    - Two bordered rectangles sit on the wallpaper: a source rectangle with multi-color sprites and a destination rectangle with singleton seed markers.
    - In each source sprite, the unique least-frequent color is the anchor; some other pixels may reuse another sprite's anchor color as relay pixels.

    Rule:
    - Read every source sprite as a template keyed by its anchor color.
    - Start from the singleton markers in the destination interior, stamp the matching templates, then treat newly stamped anchor-colored relay pixels as the next round of markers.
    - Repeat round by round until no unseen marker colors remain, and output the cropped destination rectangle after the full cascade.

    Key insight: Matching is no longer one-shot; anchor-colored pixels created by earlier stamps become visible triggers for later stamps, so the answer depends on iterative propagation.
    """
    g = np.array(grid, dtype=int)
    src, dst = _find_problem_rectangles(g)

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

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
        if anchor_coords.size != 2:
            continue
        ar, ac = map(int, anchor_coords[0])
        templates[int(anchor_col)] = {
            "patch": patch,
            "mask": mask,
            "anchor_rc": (int(ar), int(ac)),
        }

    anchor_colors = set(int(k) for k in templates)
    for anchor_col, tpl in templates.items():
        trigger_cells = []
        for rr, cc in np.argwhere(tpl["mask"]):
            col = int(tpl["patch"][int(rr), int(cc)])
            if col in anchor_colors:
                trigger_cells.append((int(rr), int(cc)))
        tpl["trigger_cells"] = trigger_cells
        templates[int(anchor_col)] = tpl

    inner = g[int(dr) + 1 : int(dr1), int(dc) + 1 : int(dc1)]
    initial_markers = [(int(rr), int(cc), int(inner[int(rr), int(cc)])) for rr, cc in np.argwhere(inner != int(dst_bg))]

    sim = _simulate_chain_from_templates(templates, initial_markers, inner.shape[0], inner.shape[1], int(dst_bg))
    assert sim is not None
    out_inner, _stats = sim

    out = _build_frame(int(dr1 - dr + 1), int(dc1 - dc + 1), int(dst_bc), int(dst_bg))
    paste(out[1:-1, 1:-1], out_inner, 0, 0)
    return out


# --------------- generator ---------------

def _assign_palette(n_types: int) -> tuple[list[int], int, int, list[int], list[int]] | None:
    wallpaper_n = 4 if n_types == 3 else random.choice((4, 5))
    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, wallpaper_n)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)
    if len(remaining) < n_types + 1:
        return None
    src_border = remaining.pop()
    anchors = [remaining.pop() for _ in range(n_types)]
    if remaining and random.random() < 0.5:
        dst_border = remaining.pop()
    else:
        border_choices = [c for c in wallpaper if c != src_border]
        if not border_choices:
            return None
        dst_border = random.choice(border_choices)
    extras = remaining[:]
    return wallpaper, src_border, dst_border, anchors, extras


def _place_sources(sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), int(src_bg), dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], {"A": sprite["anchor"], "R": sprite["relay"], "B": sprite["body"]}, src_bg)
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
    sim = _simulate_chain_from_templates(
        templates,
        [(center, center, int(sprite_types[0]["anchor"]))],
        25,
        25,
        int(dst_bg),
    )
    if sim is None:
        return None
    out, stats = sim
    if stats["rounds"] < 2 or stats["spawned_stamps"] < 1:
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
    for _ in range(n_roots):
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
    sim = _simulate_chain_from_templates(templates, markers, inner_h, inner_w, int(dst_bg))
    if sim is None:
        return None
    out_inner, stats = sim
    one_round = _simulate_chain_from_templates(templates, markers, inner_h, inner_w, int(dst_bg), max_rounds=1)
    if one_round is None:
        return None
    first_only, _ = one_round
    if np.array_equal(first_only, out_inner):
        return None
    stats["base_rounds"] = int(base_stats["rounds"])
    return markers, out_inner, stats


def _gen_attempt(return_meta: bool = False):
    n_types = random.choice(list(SPRITE_TYPE_CHOICES))
    palette = _assign_palette(n_types)
    if palette is None:
        return None
    wallpaper, src_border, dst_border, anchors, extras = palette

    src_bg = random.choice(wallpaper)
    dst_bg_choices = [c for c in wallpaper if c != src_bg and c != dst_border]
    if not dst_bg_choices:
        return None
    dst_bg = random.choice(dst_bg_choices)

    body_choices = [c for c in wallpaper + extras if c not in anchors and c not in {src_bg, dst_bg, src_border, dst_border}]
    if not body_choices:
        return None

    sprite_specs = random.sample(CHAIN_SPRITE_SHAPES, n_types)
    body_colors = [random.choice(body_choices) for _ in range(n_types)]
    sprite_types = []
    for i, ((shape_rows, anchor_rc), body) in enumerate(zip(sprite_specs, body_colors)):
        relay = int(anchors[i + 1]) if i + 1 < n_types else int(body)
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

    n_source = random.randint(max(n_types, SOURCE_SPRITE_RANGE[0]), SOURCE_SPRITE_RANGE[1])
    source_type_ids = [i for i, ct in enumerate(_choose_counts(n_source, n_types)) for _ in range(ct)]
    random.shuffle(source_type_ids)

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if src_h * src_w <= dst_h * dst_w:
        return None

    n_roots = random.randint(*INITIAL_MARKER_COUNT_RANGE)
    marker_result = _place_root_markers(sprite_types, dst_h, dst_w, int(dst_bg), int(n_roots))
    if marker_result is None:
        return None
    marker_points, out_inner, stats = marker_result

    output = _build_frame(dst_h, dst_w, int(dst_border), int(dst_bg))
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_inner = _place_sources(sprite_types, source_type_ids, src_h, src_w, int(src_bg))
    if source_inner is None:
        return None

    source_rect = _build_frame(src_h, src_w, int(src_border), int(src_bg))
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, int(dst_border), int(dst_bg))
    for mr, mc, _type_anchor in marker_points:
        dest_rect[1 + int(mr), 1 + int(mc)] = int(sprite_types[0]["anchor"])

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, int(src_pos[0]), int(src_pos[1]))
    paste(grid, dest_rect, int(dst_pos[0]), int(dst_pos[1]))

    solved = main(grid)
    if not np.array_equal(solved, output):
        return None

    if return_meta:
        meta = {
            "rounds": int(stats["rounds"]),
            "round_sizes": list(stats["round_sizes"]),
            "spawned_stamps": int(stats["spawned_stamps"]),
            "n_types": int(n_types),
            "n_roots": int(n_roots),
        }
        return grid, meta
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(return_meta=False)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a251c730 D10 input after {MAX_GEN_ATTEMPTS} attempts")
