# Combo variant of 581f7754 (index 14)
# combo_spec: D3/2+D9/3+D11/1
# Name: 空心精灵+拓扑标记+不同色小精灵干扰

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste

# Design analysis for the combo variant
# a) Concrete behavior + trace:
#    - A movable object is no longer one 8-connected multicolor blob. It is now a
#      hollow monochrome frame plus several detached colored pixels inside the hole.
#    - The true control pixel is the enclosed singleton whose 4-neighborhood is most
#      surrounded by non-sprite space: in practice the generator puts one deep interior
#      singleton (exposure 4), one same-color wall-adjacent singleton (exposure 3),
#      and one different-color wall-adjacent singleton (exposure 3 or 2).
#    - Example trace: the original solver would look for a least-frequent color inside
#      one connected component. Here that assumption fails twice: the frame and the
#      interior dots are disconnected, and the anchor color is repeated inside the hole.
#      The fused solver must first recover "which dots belong to which hollow body",
#      then score enclosed singleton pixels by local topology, then check whether the
#      winning pixel's color has a border anchor.
#    - Extra small hollow sprites are also present. They use the same interior-topology
#      pattern, but their winning singleton color is absent from the anchor set, so they
#      stay fixed and act as plausible decoys.
# b) Complexity calibration:
#    - Hollow frames are enlarged to 5-7 cells per side so the hole can visibly contain
#      several detached dots. This prevents the task from collapsing into the original
#      minority-color heuristic or a trivial "single enclosed dot" rule.
#    - True sprites are reduced to 2-4 per grid and distractors to 1-2. That keeps the
#      object family repeated enough for inference without making the canvas crowded.
#    - The same anchor color is intentionally reused on the deep true marker and on a
#      near-wall false marker inside each real sprite. Picking by color alone is not enough.
# c) Implicit-assumption audit of the original main():
#    - `find_components(... monochromatic=False, connectivity=8)` assumed body and marker
#      form one foreground component. That is false once markers are detached inside holes.
#    - The original least-frequency marker logic assumed exactly one minority pixel per
#      sprite. Here the anchor color is repeated and a non-anchor singleton may be rarer.
#    - The original singleton branch would incorrectly move detached dots on their own.
#      In the combo variant, enclosed dots only move together with the hollow frame that
#      encloses them, while unmatched hollow distractors must not move at all.
# d) Placement legality:
#    - Generator placement reserves the full sprite bounding box, including the hollow
#      cavity, so no other object can be placed inside a hole by accident.
#    - All anchors remain isolated border singletons with a one-cell background moat.
#    - Hollow sprites stay fully inside the grid before motion; only the final translated
#      paste can clip, matching the original task rule.
# e) Visual observability:
#    - Across a few examples a human can see: border singleton color picks a candidate
#      family; the relevant interior dot is the deepest/loneliest one inside each hollow
#      frame; same-looking small hollow sprites with unmatched marker colors do not move.
#    - Because every real sprite contains both a deep dot and near-wall dots inside the
#      same cavity, the topology cue is directly visible rather than hidden in metadata.
# f) Visual cleanliness:
#    - The cavity contains only three small interior marks for true sprites, which keeps
#      the hole readable. Distractors are smaller and fewer than the main sprites.
#    - Objects are separated by an 8-neighborhood gap, so no enclosure relation or anchor
#      isolation becomes visually ambiguous.
# g) Change inferability / safety:
#    - No step depends on the numeric ordering of colors. Colors are used only by equality:
#      winning interior singleton color equals anchor color -> move; otherwise ignore.
#    - The generator guarantees each processed hollow body has exactly one highest-exposure
#      enclosed singleton. The solver relies on that and skips malformed bodies otherwise.
# h) Variant metric used in validation:
#    - For every real sprite, the generator records `topological_gap = marker_exposure -
#      best_decoy_exposure`; it is enforced to be >= 1, so topology genuinely controls the
#      selected pixel.
#    - Generated grids also record how many stationary unmatched hollow distractors were
#      inserted; a healthy sample should show this frequently, otherwise D11 would be absent.
# i) New solver assumptions and generator guarantees:
#    - main() assumes each movable or stationary hollow sprite is represented by one frame
#      component plus enclosed detached monochrome components inside the cavity.
#    - main() assumes the true marker is an enclosed singleton, unique by exposure score.
#      The generator explicitly builds one deep singleton and lower-exposure decoys.
#    - main() assumes stationary distractors are structurally valid hollow sprites whose
#      winning marker color has no border anchor. The generator constructs exactly that.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide only the hollow sprites whose deepest enclosed singleton matches an isolated border anchor.

    Input structure:
    - The background is the most common color.
    - Each candidate sprite is a hollow monochrome frame in the interior with several detached colored pixels trapped inside its cavity.
    - Extra small hollow sprites use the same pattern, but their topological marker color does not appear among the border anchors.

    Rule:
    - For each hollow frame, collect the enclosed interior pieces and score every enclosed singleton by how many of its 4-neighbors lie outside the full sprite (frame + enclosed pieces).
    - The unique highest-exposure singleton is the marker. If its color matches a border anchor, slide the whole hollow sprite: top/bottom anchors align marker columns, left/right anchors align marker rows.
    - Hollow sprites whose winning singleton color has no anchor are ignored. Moved pixels clip at the border and never overwrite settled non-background cells.

    Key insight:
    - The task is three-stage: recover enclosure, choose the deepest singleton by local topology rather than by color frequency, then check whether that singleton has a valid color match on the border.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    def enclosed_open_cells(body_coords: np.ndarray) -> set[tuple[int, int]]:
        body = {(int(r), int(c)) for r, c in body_coords}
        r0 = min(r for r, _c in body)
        c0 = min(c for _r, c in body)
        r1 = max(r for r, _c in body)
        c1 = max(c for _r, c in body)
        hh = int(r1 - r0 + 3)
        ww = int(c1 - c0 + 3)
        blocked = np.zeros((hh, ww), dtype=bool)
        for r, c in body:
            blocked[int(r - r0 + 1), int(c - c0 + 1)] = True
        seen = np.zeros_like(blocked, dtype=bool)
        stack = [(0, 0)]
        seen[0, 0] = True
        while stack:
            rr, cc = stack.pop()
            for dr, dc in _NEIGHBORS4:
                nr = int(rr + dr)
                nc = int(cc + dc)
                if not (0 <= nr < hh and 0 <= nc < ww):
                    continue
                if seen[nr, nc] or blocked[nr, nc]:
                    continue
                seen[nr, nc] = True
                stack.append((nr, nc))
        enclosed = set()
        for rr in range(1, hh - 1):
            for cc in range(1, ww - 1):
                if blocked[rr, cc] or seen[rr, cc]:
                    continue
                enclosed.add((int(r0 + rr - 1), int(c0 + cc - 1)))
        return enclosed

    anchors_by_color: dict[int, tuple[int, int]] = {}
    for r in range(H):
        for c in range(W):
            if not is_border(r, c):
                continue
            color = int(g[r, c])
            if color == int(bg):
                continue
            if any(0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) != int(bg) for rr, cc in neighbors4(int(r), int(c))):
                continue
            if color not in anchors_by_color:
                anchors_by_color[int(color)] = (int(r), int(c))

    assert anchors_by_color

    mono_comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    moves: list[tuple[np.ndarray, int, int]] = []

    for comp in mono_comps:
        body_coords = comp["coords"]
        if body_coords.size == 0 or int(body_coords.shape[0]) <= 1:
            continue
        if any(is_border(int(r), int(c)) for r, c in body_coords):
            continue

        enclosed = enclosed_open_cells(body_coords)
        if not enclosed:
            continue

        enclosed_parts = []
        for other in mono_comps:
            other_coords = other["coords"]
            if other_coords is body_coords or other_coords.size == 0:
                continue
            pts = {(int(r), int(c)) for r, c in other_coords}
            if pts and pts <= enclosed:
                enclosed_parts.append(other)
        if not enclosed_parts:
            continue

        sprite_chunks = [body_coords] + [part["coords"] for part in enclosed_parts]
        sprite_coords = np.vstack(sprite_chunks)
        sprite_set = {(int(r), int(c)) for r, c in sprite_coords}

        singleton_scores: list[tuple[int, int, int, int]] = []
        for part in enclosed_parts:
            part_coords = part["coords"]
            if int(part_coords.shape[0]) != 1:
                continue
            r, c = map(int, part_coords[0])
            exposure = 0
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) not in sprite_set:
                    exposure += 1
            singleton_scores.append((int(exposure), int(r), int(c), int(part["color"])))
        if not singleton_scores:
            continue

        best = max(score for score, _r, _c, _color in singleton_scores)
        winners = [(r, c, color) for score, r, c, color in singleton_scores if int(score) == int(best)]
        if len(winners) != 1:
            continue

        marker_r, marker_c, marker_color = winners[0]
        if int(marker_color) not in anchors_by_color:
            continue

        ar, ac = anchors_by_color[int(marker_color)]
        horizontal = int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1))
        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - marker_c)
        else:
            dr = int(ar - marker_r)
        moves.append((sprite_coords, int(dr), int(dc)))

    out = g.copy()
    for coords, _dr, _dc in moves:
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for coords, dr, dc in moves:
        for r, c in coords:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(g[int(r), int(c)])

    return out


# --------------- generator (o5x) ---------------

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (12, 22)
_GRID_WIDTH_RANGE = (12, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_ANCHOR_COUNT_RANGE = (1, 3)
_SPRITE_COUNT_RANGE = (2, 4)
_GEN_RETRY_LIMIT = 700

_MAIN_FRAME_HEIGHT_RANGE = (5, 7)
_MAIN_FRAME_WIDTH_RANGE = (5, 7)
_DISTRACTOR_FRAME_HEIGHT_RANGE = (5, 6)
_DISTRACTOR_FRAME_WIDTH_RANGE = (5, 6)
_MAX_TEETH = 2
_FLIP_PROB = 0.5

_DISTRACTOR_COUNT_RANGE = (1, 2)
_DISTRACTOR_PLACE_TRIES = 40


def _norm(cells):
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _bbox(cells):
    return (
        min(r for r, _c in cells),
        min(c for _r, c in cells),
        max(r for r, _c in cells),
        max(c for _r, c in cells),
    )


def _flip(cells):
    cells = set(cells)
    if random.random() < _FLIP_PROB:
        mc = max(c for _r, c in cells)
        cells = {(r, mc - c) for r, c in cells}
    if random.random() < _FLIP_PROB:
        mr = max(r for r, _c in cells)
        cells = {(mr - r, c) for r, c in cells}
    return _norm(cells)


def _inclusive_range(bounds):
    return range(int(bounds[0]), int(bounds[1]) + 1)


def _enclosed_open_cells_from_set(body):
    body = {(int(r), int(c)) for r, c in body}
    r0, c0, r1, c1 = _bbox(body)
    hh = int(r1 - r0 + 3)
    ww = int(c1 - c0 + 3)
    blocked = np.zeros((hh, ww), dtype=bool)
    for r, c in body:
        blocked[int(r - r0 + 1), int(c - c0 + 1)] = True
    seen = np.zeros_like(blocked, dtype=bool)
    stack = [(0, 0)]
    seen[0, 0] = True
    while stack:
        rr, cc = stack.pop()
        for dr, dc in _NEIGHBORS4:
            nr = int(rr + dr)
            nc = int(cc + dc)
            if not (0 <= nr < hh and 0 <= nc < ww):
                continue
            if seen[nr, nc] or blocked[nr, nc]:
                continue
            seen[nr, nc] = True
            stack.append((nr, nc))
    enclosed = set()
    for rr in range(1, hh - 1):
        for cc in range(1, ww - 1):
            if blocked[rr, cc] or seen[rr, cc]:
                continue
            enclosed.add((int(r0 + rr - 1), int(c0 + cc - 1)))
    return enclosed


def _make_hollow_body(height_range, width_range):
    h = random.randint(*height_range)
    w = random.randint(*width_range)
    cells = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if r in (0, h - 1) or c in (0, w - 1)
    }
    side_specs = [
        (lambda pos: (1, pos), range(2, w - 2)),
        (lambda pos: (h - 2, pos), range(2, w - 2)),
        (lambda pos: (pos, 1), range(2, h - 2)),
        (lambda pos: (pos, w - 2), range(2, h - 2)),
    ]
    for _ in range(random.randint(0, _MAX_TEETH)):
        valid = [(build, list(positions)) for build, positions in side_specs if list(positions)]
        if not valid:
            break
        build, positions = random.choice(valid)
        cells.add(build(int(random.choice(positions))))
    body = _flip(cells)
    return body if _enclosed_open_cells_from_set(body) else _make_hollow_body(height_range, width_range)


def _cell_exposure(sprite_cells, pos):
    r, c = pos
    return sum((int(r + dr), int(c + dc)) not in sprite_cells for dr, dc in _NEIGHBORS4)


def _not_adjacent4(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) != 1


def _build_inner_layout(body, marker_color, foreign_color):
    body = {(int(r), int(c)) for r, c in body}
    enclosed = _enclosed_open_cells_from_set(body)
    deep = []
    wall = []
    for cell in enclosed:
        touches_body = any((int(cell[0] + dr), int(cell[1] + dc)) in body for dr, dc in _NEIGHBORS4)
        if touches_body:
            wall.append((int(cell[0]), int(cell[1])))
        else:
            deep.append((int(cell[0]), int(cell[1])))
    random.shuffle(deep)
    random.shuffle(wall)
    if not deep or len(wall) < 2:
        return None

    for marker in deep:
        for same_color_decoy in wall:
            if not _not_adjacent4(marker, same_color_decoy):
                continue
            for foreign_decoy in wall:
                if foreign_decoy == same_color_decoy:
                    continue
                if not _not_adjacent4(marker, foreign_decoy):
                    continue
                if not _not_adjacent4(same_color_decoy, foreign_decoy):
                    continue
                inner = {
                    (int(marker[0]), int(marker[1])): int(marker_color),
                    (int(same_color_decoy[0]), int(same_color_decoy[1])): int(marker_color),
                    (int(foreign_decoy[0]), int(foreign_decoy[1])): int(foreign_color),
                }
                sprite_cells = set(body) | set(inner)
                scores = {pos: _cell_exposure(sprite_cells, pos) for pos in inner}
                other_best = max(score for pos, score in scores.items() if pos != marker)
                if int(scores[marker]) > int(other_best):
                    return {
                        "cells": inner,
                        "marker": marker,
                        "scores": scores,
                        "hole_area": len(enclosed),
                    }
    return None


def _make_sprite(body_color, marker_color, foreign_color, bg, *, distractor=False):
    height_range = _DISTRACTOR_FRAME_HEIGHT_RANGE if distractor else _MAIN_FRAME_HEIGHT_RANGE
    width_range = _DISTRACTOR_FRAME_WIDTH_RANGE if distractor else _MAIN_FRAME_WIDTH_RANGE
    for _ in range(80):
        body = _make_hollow_body(height_range, width_range)
        layout = _build_inner_layout(body, marker_color, foreign_color)
        if layout is None:
            continue
        all_cells = set(body) | set(layout["cells"])
        r0, c0, r1, c1 = _bbox(all_cells)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
        for r, c in body:
            patch[int(r - r0), int(c - c0)] = int(body_color)
        for (r, c), color in layout["cells"].items():
            patch[int(r - r0), int(c - c0)] = int(color)
        marker = layout["marker"]
        topological_gap = int(layout["scores"][marker] - max(score for pos, score in layout["scores"].items() if pos != marker))
        return {
            "patch": patch,
            "offset": (int(marker[0] - r0), int(marker[1] - c0)),
            "marker_color": int(marker_color),
            "body_color": int(body_color),
            "foreign_color": int(foreign_color),
            "hole_area": int(layout["hole_area"]),
            "topological_gap": int(topological_gap),
        }
    return None


def _top_left(sprite, marker_r, marker_c):
    off_r, off_c = sprite["offset"]
    return int(marker_r) - int(off_r), int(marker_c) - int(off_c)


def _touches_foreground(canvas, mask, top, left, bg):
    h, w = mask.shape
    radius = _ISOLATION_RADIUS
    occupied = canvas[top - radius : top + h + radius, left - radius : left + w + radius] != int(bg)
    touch_zone = np.zeros((h + 2 * radius, w + 2 * radius), dtype=bool)
    span = 2 * radius + 1
    for dr in range(span):
        for dc in range(span):
            touch_zone[dr : dr + h, dc : dc + w] |= mask
    return bool((occupied & touch_zone).any())


def _can_place(canvas, sprite, marker_r, marker_c, bg):
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    h, w = patch.shape
    H, W = canvas.shape
    if top < _INNER_BORDER_GAP or left < _INNER_BORDER_GAP:
        return False
    if top + h > H - _INNER_BORDER_GAP or left + w > W - _INNER_BORDER_GAP:
        return False
    view = canvas[top : top + h, left : left + w]
    if bool((view != int(bg)).any()):
        return False
    mask = patch != int(bg)
    return not _touches_foreground(canvas, mask, top, left, bg)


def _place(canvas, sprite, marker_r, marker_c, bg):
    top, left = _top_left(sprite, marker_r, marker_c)
    paste(canvas, sprite["patch"], top, left, transparent=int(bg))


def _isolated_bg_cell(canvas, r, c, bg):
    H, W = canvas.shape
    radius = _ISOLATION_RADIUS
    r0, r1 = max(0, r - radius), min(H, r + radius + 1)
    c0, c1 = max(0, c - radius), min(W, c + radius + 1)
    return not bool((canvas[r0:r1, c0:c1] != int(bg)).any())


def _anchor_ok(canvas, r, c, bg):
    H, W = canvas.shape
    if not (r in (0, H - 1) or c in (0, W - 1)):
        return False
    if r in (0, H - 1) and c in (0, W - 1):
        return False
    return _isolated_bg_cell(canvas, r, c, bg)


def _anchor_candidates(canvas, bg):
    H, W = canvas.shape
    rows = list(range(_ANCHOR_EDGE_MARGIN, H - _ANCHOR_EDGE_MARGIN))
    rows = rows or list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP))
    cols = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN))
    cols = cols or list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP))
    out = []
    for c in cols:
        out.append(("h", 0, c))
        out.append(("h", H - 1, c))
    for r in rows:
        out.append(("v", r, 0))
        out.append(("v", r, W - 1))
    random.shuffle(out)
    return [(axis, r, c) for axis, r, c in out if _anchor_ok(canvas, r, c, bg)]


def _put_anchor(canvas, r, c, color):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _aligned_positions(canvas, sprite, anchor, bg):
    H, W = canvas.shape
    if anchor["axis"] == "h":
        return [
            (r, anchor["c"])
            for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
            if _can_place(canvas, sprite, r, anchor["c"], bg)
        ]
    return [
        (anchor["r"], c)
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
        if _can_place(canvas, sprite, anchor["r"], c, bg)
    ]


def _scatter_positions(canvas, sprite, anchor, marker, bg):
    H, W = canvas.shape
    mr, mc = marker
    if anchor["axis"] == "h":
        return [
            (mr, c)
            for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
            if c != mc and _can_place(canvas, sprite, mr, c, bg)
        ]
    return [
        (r, mc)
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
        if r != mr and _can_place(canvas, sprite, r, mc, bg)
    ]


def _stationary_positions(out, inp, sprite, bg):
    H, W = out.shape
    positions = []
    for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP):
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP):
            if _can_place(out, sprite, r, c, bg) and _can_place(inp, sprite, r, c, bg):
                positions.append((r, c))
    random.shuffle(positions)
    return positions


def _pick_color(bg, forbidden=(), avoid=()):
    forbidden = {int(bg)} | {int(c) for c in forbidden}
    avoid = {int(c) for c in avoid}
    pool = [c for c in _COLORS if c not in forbidden and c not in avoid]
    if not pool:
        pool = [c for c in _COLORS if c not in forbidden]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _new_canvas_pair():
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.randint(*_BG_COLOR_RANGE)
    out = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    return bg, out, inp


def _place_anchors(out, inp, bg):
    anchor_colors = random.sample([c for c in _COLORS if c != int(bg)], random.randint(*_ANCHOR_COUNT_RANGE))
    anchors = []
    for color in anchor_colors:
        candidates = _anchor_candidates(out, bg)
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        _put_anchor(out, r, c, color)
        _put_anchor(inp, r, c, color)
    return anchor_colors, anchors


def _marker_color_sequence(anchor_colors):
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    marker_colors = anchor_colors[:]
    while len(marker_colors) < n_sprites:
        marker_colors.append(random.choice(anchor_colors))
    random.shuffle(marker_colors)
    return marker_colors


def _place_aligned_sprites(out, anchors, marker_colors, bg):
    by_color = {a["color"]: a for a in anchors}
    anchor_color_set = set(by_color)
    used_body_colors = set()
    sprites = []
    for marker in marker_colors:
        anchor = by_color[int(marker)]
        body_color = _pick_color(bg, forbidden={marker}, avoid=used_body_colors)
        foreign_color = _pick_color(bg, forbidden={marker, body_color} | anchor_color_set)
        sprite = _make_sprite(body_color, marker, foreign_color, bg, distractor=False)
        if sprite is None:
            return None
        positions = _aligned_positions(out, sprite, anchor, bg)
        if not positions:
            return None
        marker_pos = random.choice(positions)
        _place(out, sprite, marker_pos[0], marker_pos[1], bg)
        used_body_colors.add(int(body_color))
        sprites.append(
            {
                "sprite": sprite,
                "anchor": anchor,
                "output_marker_pos": (int(marker_pos[0]), int(marker_pos[1])),
            }
        )
    return sprites


def _scatter_sprites(inp, sprites, bg):
    random.shuffle(sprites)
    for item in sprites:
        sprite = item["sprite"]
        anchor = item["anchor"]
        marker_pos = item["output_marker_pos"]
        positions = _scatter_positions(inp, sprite, anchor, marker_pos, bg)
        if not positions:
            return False
        input_pos = random.choice(positions)
        _place(inp, sprite, input_pos[0], input_pos[1], bg)
        item["input_marker_pos"] = (int(input_pos[0]), int(input_pos[1]))
    return True


def _unmatched_marker_color(bg, anchors):
    anchor_colors = {int(a["color"]) for a in anchors}
    pool = [c for c in _COLORS if c != int(bg) and c not in anchor_colors]
    return None if not pool else random.choice(pool)


def _place_stationary_distractors(out, inp, anchors, bg):
    marker = _unmatched_marker_color(bg, anchors)
    if marker is None:
        return []

    used_body_colors = set()
    placed = []
    target = random.randint(*_DISTRACTOR_COUNT_RANGE)
    for _ in range(target):
        success = False
        for _attempt in range(_DISTRACTOR_PLACE_TRIES):
            body_color = _pick_color(bg, forbidden={marker}, avoid=used_body_colors)
            foreign_color = _pick_color(bg, forbidden={marker, body_color, marker})
            sprite = _make_sprite(body_color, marker, foreign_color, bg, distractor=True)
            if sprite is None:
                continue
            positions = _stationary_positions(out, inp, sprite, bg)
            if not positions:
                continue
            marker_pos = random.choice(positions)
            _place(out, sprite, marker_pos[0], marker_pos[1], bg)
            _place(inp, sprite, marker_pos[0], marker_pos[1], bg)
            used_body_colors.add(int(body_color))
            placed.append(
                {
                    "sprite": sprite,
                    "marker_pos": (int(marker_pos[0]), int(marker_pos[1])),
                }
            )
            success = True
            break
        if not success:
            break
    return placed


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    return bool(np.array_equal(main(inp), out))


def _gen_attempt(return_meta=False):
    bg, out, inp = _new_canvas_pair()
    placed = _place_anchors(out, inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors = placed

    marker_colors = _marker_color_sequence(anchor_colors)
    sprites = _place_aligned_sprites(out, anchors, marker_colors, bg)
    if sprites is None:
        return None
    if not _scatter_sprites(inp, sprites, bg):
        return None

    distractors = _place_stationary_distractors(out, inp, anchors, bg)
    if not distractors:
        return None

    if not _generated_grid_is_valid(inp, out, bg):
        return None

    if return_meta:
        return inp, out, {"bg": bg, "anchors": anchors, "sprites": sprites, "distractors": distractors}
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 combo input")
