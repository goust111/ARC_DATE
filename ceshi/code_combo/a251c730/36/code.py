# Combo variant of a251c730 (index 36)
# combo_spec: D6/2+D11/1
# Name: 对角条纹+移动轨迹

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the fused variant
# a) Behavior trace:
#    The original task only crops the destination frame after stamping sprites onto markers.
#    Here the source frame still provides sprite templates, but the output keeps the full 30x30
#    wallpaper. Each source sprite is treated as one mover: it disappears from the source frame,
#    reappears in the destination frame with anchor-to-marker alignment, and paints a body-colored
#    L-shaped trail between the two rectangles. The wallpaper itself changes from axial stripes to
#    diagonal stripes defined by (r+c) mod period, so the trail visually interferes with the slant.
# b) Complexity calibration:
#    To keep the movement rule inferable, the generator uses 2-3 unique sprite types and exactly
#    one source instance / one destination marker per type. That removes ambiguous clone-vs-move
#    cases while still giving repeated support across multiple colored trajectories in each sample.
# c) Main() assumptions reviewed:
#    - Rectangle detection still relies on two uniform borders; generator preserves that.
#    - Source components must be separated 4-connected blobs; generator keeps padding >= 1.
#    - The least frequent sprite color is still the anchor; each sprite remains a 2-color shape
#      with a singleton anchor pixel.
#    - Matching now also assumes one marker per anchor color and one source instance per anchor
#      color. Generator constructs that exactly, so trail origins are well-defined.
#    - Path routing assumes the rectangles are placed either side-by-side or stacked, never in a
#      diagonal overlap layout. Generator uses only those two layouts.
# d) Placement legality:
#    Source and destination frames are placed non-overlapping inside 30x30. Marker placement still
#    guarantees the stamped sprite fits fully inside the destination interior. Trails are routed
#    through the corridor implied by the chosen layout, so they stay in-bounds.
# e) Visual observability:
#    Humans can see three coupled cues at once: diagonal wallpaper, emptied source frame, and
#    colored trails leading into stamped target sprites. The output is therefore visibly different
#    from the original cropped-destination behavior even in a single example.
# f) Visual cleanliness:
#    Using unique anchor colors and one mover per color keeps correspondences crisp. The trail is
#    a single-cell Manhattan polyline rather than a thick flood, so it reads cleanly over the
#    wallpaper and does not obliterate the destination stamp.
# g) Non-degeneration guard:
#    Because the output keeps the whole grid and alters cells outside both rectangles, it cannot
#    collapse back to the original task behavior on the same input. Post-test metrics below verify
#    nonzero trail cells outside the frames and diagonal wallpaper in generated samples.


def _line_coords(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Inclusive coords for an axis-aligned segment."""
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(int(r0), int(c)) for c in range(int(c0), int(c1) + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(int(r), int(c0)) for r in range(int(r0), int(r1) + step, step)]
    raise ValueError("_line_coords requires horizontal or vertical endpoints")


def _trace_path(
    start: tuple[int, int],
    end: tuple[int, int],
    src_rect: tuple[int, int, int, int],
    dst_rect: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    """
    Build a consistent trail that stays on the wallpaper corridor outside both frames.

    The trail starts immediately outside the source frame, bends once in the free corridor,
    and ends immediately outside the destination frame. This keeps the source interior visibly
    cleared while still showing a readable movement route between the rectangles.
    """
    sr0, sc0, sr1, sc1 = map(int, src_rect)
    dr0, dc0, dr1, dc1 = map(int, dst_rect)
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)

    if int(sc1) < int(dc0) or int(dc1) < int(sc0):
        if int(sc1) < int(dc0):
            start_out = (int(r0), int(sc1 + 1))
            end_out = (int(r1), int(dc0 - 1))
        else:
            start_out = (int(r0), int(sc0 - 1))
            end_out = (int(r1), int(dc1 + 1))
        elbow = (int(r1), int(start_out[1]))
    else:
        if int(sr1) < int(dr0):
            start_out = (int(sr1 + 1), int(c0))
            end_out = (int(dr0 - 1), int(c1))
        else:
            start_out = (int(sr0 - 1), int(c0))
            end_out = (int(dr1 + 1), int(c1))
        elbow = (int(start_out[0]), int(c1))

    coords = _line_coords(start_out, elbow)
    coords.extend(_line_coords(elbow, end_out)[1:])
    return coords


def _render_sprite(shape_rows: tuple[str, ...], body: int, anchor: int, bg: int) -> np.ndarray:
    """Render a sprite template into a colored patch. 'A'=anchor, 'B'=body, '.'=bg."""
    char_map = {"B": int(body), "A": int(anchor)}
    rows = [[char_map.get(ch, int(bg)) for ch in row] for row in shape_rows]
    return np.array(rows, dtype=int)


def _make_wallpaper(palette: list[int]) -> np.ndarray:
    """Create a 30x30 diagonal striped wallpaper with period len(palette)."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    period = int(len(palette))
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            grid[r, c] = int(palette[(r + c) % period])
    return grid


def _build_frame(h: int, w: int, border: int, bg: int) -> np.ndarray:
    """Create an h x w rectangle with a 1-pixel border and interior filled with bg."""
    rect = np.full((h, w), int(bg), dtype=int)
    rect[0, :] = rect[-1, :] = int(border)
    rect[:, 0] = rect[:, -1] = int(border)
    return rect


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move source sprites onto color-matched markers and keep the whole diagonal-wallpaper scene.

    Input structure:
    - A 30x30 grid with diagonal striped wallpaper plus two bordered rectangles.
    - The source rectangle contains separated 2-color sprites; the destination rectangle contains
      singleton marker pixels whose colors match sprite anchors.

    Rule:
    - Pair each marker with the source sprite that has the same anchor color.
    - Clear both rectangle interiors, stamp the sprite into the destination so the anchor lands on
      the marker, and draw a body-colored trail from the source anchor to that destination anchor.
    - Return the full grid, not a cropped destination frame.

    Key insight:
    - Anchor colors still provide the correspondence key, but the output now combines three effects:
      source disappearance, destination stamping, and a routed trail over the diagonal wallpaper.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    def interior_bg(rect: tuple[int, int, int, int]) -> int:
        r0, c0, r1, c1 = map(int, rect)
        inner = g[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
        return int(Counter(map(int, inner.flatten())).most_common(1)[0][0])

    def rect_components(rect: tuple[int, int, int, int]) -> tuple[int, list[dict]]:
        r0, c0, r1, c1 = map(int, rect)
        inner = g[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
        ibg = int(Counter(map(int, inner.flatten())).most_common(1)[0][0])
        comps: list[dict] = []

        for coords in components_mask(inner != int(ibg), connectivity=4):
            if coords.size == 0:
                continue
            rr0, cc0, rr1, cc1 = bbox_coords(coords)
            mask = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
            mask[coords[:, 0] - int(rr0), coords[:, 1] - int(cc0)] = True
            patch = inner[int(rr0) : int(rr1) + 1, int(cc0) : int(cc1) + 1].copy()
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

        comps.sort(key=lambda d: (-int(d["size"]), int(d["bbox"][0]), int(d["bbox"][1])))
        return int(ibg), comps

    rects: list[tuple[int, tuple[int, int, int, int], int, int]] = []
    min_h = 6
    min_w = 6
    for r0 in range(int(H)):
        for c0 in range(int(W)):
            bc = int(g[int(r0), int(c0)])
            for r1 in range(int(r0 + min_h - 1), int(H)):
                for c1 in range(int(c0 + min_w - 1), int(W)):
                    if (
                        int(g[int(r0), int(c1)]) != int(bc)
                        or int(g[int(r1), int(c0)]) != int(bc)
                        or int(g[int(r1), int(c1)]) != int(bc)
                    ):
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
                    rects.append((int(area), (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))

    assert rects, "expected at least two bordered rectangles"

    rects.sort(reverse=True, key=lambda x: int(x[0]))
    rects = rects[:6]

    src = None
    dst = None
    for _area, rect, bc, _ibg in rects:
        ibg2, comps = rect_components(rect)
        sizes = [int(c["size"]) for c in comps]
        max_sz = max(sizes, default=0)
        if max_sz > 1 and src is None:
            src = (rect, bc, ibg2, comps)
        if max_sz == 1 and len(comps) > 0 and dst is None:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find source sprites and destination markers"

    (sr, sc, sr1, sc1), _src_bc, src_bg, src_comps = src
    (dr, dc, dr1, dc1), _dst_bc, dst_bg, dst_comps = dst

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int], int]] = {}
    source_positions: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for comp in src_comps:
        cols = {int(k): int(v) for k, v in comp["colors"].items() if int(k) != 0}
        assert cols, "source component must contain visible colors"
        anchor_col = int(min(cols.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
        body_col = int(max(((k, v) for k, v in cols.items() if int(k) != int(anchor_col)), key=lambda kv: (int(kv[1]), -int(kv[0])))[0])
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere((patch == int(anchor_col)) & mask)
        assert anchors.size > 0, "source component must expose its anchor pixel"
        ar, ac = map(int, anchors[0])
        rr0, cc0, _rr1, _cc1 = map(int, comp["bbox"])
        abs_anchor = (int(sr + 1 + rr0 + ar), int(sc + 1 + cc0 + ac))
        source_positions[int(anchor_col)].append(abs_anchor)
        if int(anchor_col) not in templates or int(mask.sum()) > int(templates[int(anchor_col)][1].sum()):
            templates[int(anchor_col)] = (patch.copy(), mask.copy(), (int(ar), int(ac)), int(body_col))

    marker_positions: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for comp in dst_comps:
        rr0, cc0, _rr1, _cc1 = map(int, comp["bbox"])
        color = int(comp["patch"][0, 0])
        marker_positions[int(color)].append((int(dr + 1 + rr0), int(dc + 1 + cc0)))

    out = g.copy()
    out[int(sr) + 1 : int(sr1), int(sc) + 1 : int(sc1)] = int(src_bg)
    out[int(dr) + 1 : int(dr1), int(dc) + 1 : int(dc1)] = int(dst_bg)

    for key in list(source_positions.keys()):
        source_positions[key].sort()
    for key in list(marker_positions.keys()):
        marker_positions[key].sort()

    for anchor_col, markers in marker_positions.items():
        assert int(anchor_col) in templates, "every destination marker must match a source sprite"
        assert len(source_positions[int(anchor_col)]) == len(markers), "source/marker counts must agree per color"
        patch, mask, (ar, ac), body_col = templates[int(anchor_col)]
        for src_anchor, dst_anchor in zip(source_positions[int(anchor_col)], markers):
            for rr, cc in _trace_path(src_anchor, dst_anchor, (sr, sc, sr1, sc1), (dr, dc, dr1, dc1)):
                out[int(rr), int(cc)] = int(body_col)
            top = int(dst_anchor[0] - dr - ar)
            left = int(dst_anchor[1] - dc - ac)
            paste(out[int(dr) : int(dr1) + 1, int(dc) : int(dc1) + 1], patch, int(top), int(left), mask=mask, overwrite=True, clip=True)

    return out


# --------------- generator ---------------

GRID_SIZE = 30
WALLPAPER_COLORS = (4, 5)
GAP_RANGE = (1, 4)

SRC_H_RANGE = (9, 19)
SRC_W_RANGE = (8, 18)
DST_H_RANGE = (9, 19)
DST_W_RANGE = (8, 20)
DST_W_SIDE_MAX = 20

SPRITE_TYPE_CHOICES = (2, 3)
MARKER_PLACE_RETRIES = 30
SOURCE_SPRITE_PADDING = 1
MAX_GEN_ATTEMPTS = 400

SPRITE_SHAPES = [
    (("AB.", "BBB"), (0, 0)),
    (("ABB", "BBB"), (0, 0)),
    ((".A.", "BBB", ".B."), (0, 1)),
    (("AB.", "BBB", ".BB"), (0, 0)),
    (("ABB", "BBB", "..B"), (0, 0)),
    ((".AB", "BBB", "BB."), (0, 1)),
    (("B.B", "BBB", ".A."), (2, 1)),
    ((".B.", "BAB", ".B."), (1, 1)),
    (("BBB", "BAB"), (1, 1)),
    (("AB.", "BBB", ".B."), (0, 0)),
    (("BAB", "BBB"), (0, 1)),
]


def _choose_dimensions(layout: str) -> tuple[int, int, int, int]:
    """Pick (src_h, src_w, dst_h, dst_w) that fit the 30x30 canvas for the given layout."""
    gap = random.randint(*GAP_RANGE)
    if layout == "side_by_side":
        src_w = random.randint(*SRC_W_RANGE)
        dst_w_max = min(int(DST_W_SIDE_MAX), int(GRID_SIZE - gap - src_w))
        if int(dst_w_max) < int(DST_W_RANGE[0]):
            raise ValueError("no side-by-side width fit")
        dst_w = random.randint(int(DST_W_RANGE[0]), int(dst_w_max))
        src_h = random.randint(*SRC_H_RANGE)
        dst_h = random.randint(*DST_H_RANGE)
    else:
        src_h = random.randint(*SRC_H_RANGE)
        dst_h_max = min(int(DST_H_RANGE[1]), int(GRID_SIZE - gap - src_h))
        if int(dst_h_max) < int(DST_H_RANGE[0]):
            raise ValueError("no stacked height fit")
        dst_h = random.randint(int(DST_H_RANGE[0]), int(dst_h_max))
        src_w = random.randint(*SRC_W_RANGE)
        dst_w = random.randint(*DST_W_RANGE)
    return int(src_h), int(src_w), int(dst_h), int(dst_w)


def _place_rectangles(
    layout: str, src_h: int, src_w: int, dst_h: int, dst_w: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Compute non-overlapping (row, col) origins for src and dst rectangles."""
    if layout == "side_by_side":
        gap = random.randint(1, int(GRID_SIZE - src_w - dst_w))
        slack = int(GRID_SIZE - src_w - dst_w - gap)
        left_margin = random.randint(0, int(slack))
        src_r = random.randint(0, int(GRID_SIZE - src_h))
        dst_r = random.randint(0, int(GRID_SIZE - dst_h))
        if random.random() < 0.5:
            src_c, dst_c = int(left_margin), int(left_margin + src_w + gap)
        else:
            dst_c, src_c = int(left_margin), int(left_margin + dst_w + gap)
    else:
        gap = random.randint(1, int(GRID_SIZE - src_h - dst_h))
        slack = int(GRID_SIZE - src_h - dst_h - gap)
        top_margin = random.randint(0, int(slack))
        src_c = random.randint(0, int(GRID_SIZE - src_w))
        dst_c = random.randint(0, int(GRID_SIZE - dst_w))
        if random.random() < 0.5:
            src_r, dst_r = int(top_margin), int(top_margin + src_h + gap)
        else:
            dst_r, src_r = int(top_margin), int(top_margin + dst_h + gap)
    return (int(src_r), int(src_c)), (int(dst_r), int(dst_c))


def _assign_palette(n_types: int) -> tuple[list[int], int, int, int, int, list[int], list[int]] | None:
    """
    Allocate colors for wallpaper, borders, anchors and bodies.

    We keep source border off-wallpaper for easy rectangle detection; destination border may share
    a wallpaper color to keep the total distinct color count bounded when n_types == 3.
    """
    wallpaper_n = 5 if int(n_types) == 3 else random.choice(list(WALLPAPER_COLORS))
    all_colors = list(range(1, 10))
    wallpaper = random.sample(all_colors, int(wallpaper_n))
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)

    if len(remaining) < int(n_types + 1):
        return None

    src_border = int(remaining.pop())
    anchors = [int(remaining.pop()) for _ in range(int(n_types))]

    bg_choices = wallpaper[:]
    random.shuffle(bg_choices)
    src_bg = int(bg_choices.pop())
    dst_bg_candidates = [c for c in bg_choices if int(c) != int(src_bg)]
    if not dst_bg_candidates:
        return None
    dst_bg = int(random.choice(dst_bg_candidates))

    dst_border_choices = [c for c in wallpaper if int(c) != int(dst_bg)]
    if not dst_border_choices:
        return None
    dst_border = int(random.choice(dst_border_choices))

    body_choices = [c for c in wallpaper if int(c) not in {int(src_bg), int(dst_bg)}]
    if len(body_choices) < int(n_types):
        return None
    bodies = random.sample(body_choices, int(n_types))
    return wallpaper, src_border, dst_border, src_bg, dst_bg, anchors, bodies


def _place_markers(
    sprite_types: list[dict], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    """Place one marker per type in the destination interior."""
    inner = np.full((dst_h - 2, dst_w - 2), int(dst_bg), dtype=int)
    marker_points: list[tuple[int, int, int]] = []
    for type_id, sprite in enumerate(sprite_types):
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], dst_bg)
        placed = False
        for _ in range(MARKER_PLACE_RETRIES):
            try:
                r0, c0 = random_free_location(inner, patch, transparent=dst_bg)
            except ValueError:
                return None
            mr = int(r0 + sprite["anchor_rc"][0])
            mc = int(c0 + sprite["anchor_rc"][1])
            if any(abs(int(mr) - int(pr)) + abs(int(mc) - int(pc)) <= 1 for pr, pc, _ in marker_points):
                continue
            paste(inner, patch, int(r0), int(c0), transparent=dst_bg)
            marker_points.append((int(mr), int(mc), int(type_id)))
            placed = True
            break
        if not placed:
            return None
    return inner, marker_points


def _place_sources(
    sprite_types: list[dict], src_h: int, src_w: int, src_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    """Place one source sprite per type and record anchor positions in source-local coords."""
    inner = np.full((src_h - 2, src_w - 2), int(src_bg), dtype=int)
    anchors: list[tuple[int, int, int]] = []
    for type_id, sprite in enumerate(sprite_types):
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], src_bg)
        try:
            r0, c0 = random_free_location(
                inner,
                patch,
                transparent=src_bg,
                border=SOURCE_SPRITE_PADDING,
                padding=SOURCE_SPRITE_PADDING,
            )
        except ValueError:
            return None
        paste(inner, patch, int(r0), int(c0), transparent=src_bg)
        anchors.append((int(r0 + sprite["anchor_rc"][0]), int(c0 + sprite["anchor_rc"][1]), int(type_id)))
    return inner, anchors


def _trail_metrics(
    src_rect: tuple[int, int, int, int],
    dst_rect: tuple[int, int, int, int],
    source_anchors: list[tuple[int, int, int]],
    marker_points: list[tuple[int, int, int]],
) -> tuple[int, int]:
    """Return (outside_rect_cells, overlapping_cells) for the planned trails."""
    src_abs = {int(t): (int(src_rect[0] + 1 + r), int(src_rect[1] + 1 + c)) for r, c, t in source_anchors}
    dst_abs = {int(t): (int(dst_rect[0] + 1 + r), int(dst_rect[1] + 1 + c)) for r, c, t in marker_points}

    cover: Counter = Counter()
    outside = 0
    sr0, sc0, sr1, sc1 = map(int, src_rect)
    dr0, dc0, dr1, dc1 = map(int, dst_rect)

    for type_id in sorted(src_abs):
        for cell in _trace_path(src_abs[int(type_id)], dst_abs[int(type_id)], src_rect, dst_rect):
            cover[cell] += 1
            r, c = map(int, cell)
            in_src = int(sr0) <= int(r) <= int(sr1) and int(sc0) <= int(c) <= int(sc1)
            in_dst = int(dr0) <= int(r) <= int(dr1) and int(dc0) <= int(c) <= int(dc1)
            if not in_src and not in_dst:
                outside += 1

    overlap = sum(1 for v in cover.values() if int(v) > 1)
    return int(outside), int(overlap)


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid fused-variant input grid."""
    n_types = int(random.choice(list(SPRITE_TYPE_CHOICES)))

    palette_result = _assign_palette(n_types)
    if palette_result is None:
        return None
    wallpaper, src_border, dst_border, src_bg, dst_bg, anchors, bodies = palette_result

    sprite_specs = random.sample(SPRITE_SHAPES, int(n_types))
    sprite_types = [
        {
            "shape_rows": shape_rows,
            "anchor_rc": anchor_rc,
            "body": int(body),
            "anchor": int(anchor),
            "h": len(shape_rows),
            "w": len(shape_rows[0]),
        }
        for (shape_rows, anchor_rc), body, anchor in zip(sprite_specs, bodies, anchors)
    ]

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if int(src_h * src_w) <= int(dst_h * dst_w):
        return None

    marker_result = _place_markers(sprite_types, dst_h, dst_w, dst_bg)
    if marker_result is None:
        return None
    _out_inner, marker_points = marker_result

    source_result = _place_sources(sprite_types, src_h, src_w, src_bg)
    if source_result is None:
        return None
    source_inner, source_anchors = source_result

    source_rect = _build_frame(src_h, src_w, src_border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    for mr, mc, type_id in marker_points:
        dest_rect[int(1 + mr), int(1 + mc)] = int(sprite_types[int(type_id)]["anchor"])

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    src_rect = (int(src_pos[0]), int(src_pos[1]), int(src_pos[0] + src_h - 1), int(src_pos[1] + src_w - 1))
    dst_rect = (int(dst_pos[0]), int(dst_pos[1]), int(dst_pos[0] + dst_h - 1), int(dst_pos[1] + dst_w - 1))
    outside_cells, overlap_cells = _trail_metrics(src_rect, dst_rect, source_anchors, marker_points)
    if int(outside_cells) < int(4 * n_types):
        return None
    if int(overlap_cells) > 2:
        return None

    grid = _make_wallpaper(wallpaper)
    paste(grid, source_rect, int(src_pos[0]), int(src_pos[1]))
    paste(grid, dest_rect, int(dst_pos[0]), int(dst_pos[1]))

    out = main(grid)
    if not bool((out[int(src_pos[0]) + 1 : int(src_pos[0]) + src_h - 1, int(src_pos[1]) + 1 : int(src_pos[1]) + src_w - 1] == int(src_bg)).all()):
        return None
    changed = int(np.count_nonzero(out != grid))
    if int(changed) < int(10 * n_types):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate fused a251c730 input after {MAX_GEN_ATTEMPTS} attempts")
