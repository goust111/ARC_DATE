# Combo variant of 64efde09 (index 19)
# combo_spec: D5/2+D13/2+D15/2
# Name: 面对面碰撞截断射线

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior:
#    The original task sends each labeled port ray from a sprite to an outer boundary.
#    Here the sprites are paired face-to-face, so the "outward" direction now points
#    into a corridor between two sprites. A small example is N=7 with labeled rows
#    [1, 3, 5]. In the top pair, the left sprite and right sprite expose the same
#    template ids on those rows, so the three rays are same-color head-on collisions.
#    In the bottom pair, one sprite is vertically reversed, so row 1 on the left faces
#    row 1 on the right but those rows correspond to different template ids; the result
#    is a different-color head-on truncation on two rows and a same-color collision on
#    the center row. This is the intended fused contrast.
# b) Complexity calibration:
#    Four sprites in exactly two horizontal pairs is enough to make the collision rule
#    visible without introducing ambiguous multi-way intersections. Three labeled rows
#    are repeated across both pairs, so the user sees the same template-to-color mapping
#    support multiple times.
# c) Broken original assumptions:
#    - "Active side faces a boundary" is no longer true, so boundary-marker placement
#      and draw-until-boundary logic must be removed.
#    - "Any non-bg blocks a ray" is no longer true. Markers now sit inside the corridor
#      and act as transparent metadata for labeling, not as blockers.
#    - "Different colors never share a line" is no longer guaranteed. A reversed facing
#      pair intentionally creates rows where the opposing rays carry different colors.
# d) Placement legality:
#    The generator keeps both pairs separated vertically, keeps each corridor internal
#    to the grid, and leaves at least one bg cell between a sprite and its nearest
#    marker so markers remain isolated single-pixel components.
# e) Visual observability:
#    The top pair shows solid monochrome bridges (same-color collisions). The bottom
#    pair shows split two-color bridges with a center gap on odd-width corridors
#    (different-color truncation). Those two outcomes are easy to compare visually.
# f) Visual cleanliness:
#    Only one collision corridor exists per row, and there are no perpendicular
#    crossings. That avoids unreadable overlaps and makes the collision site obvious.
# g) New solver assumptions enforced by the generator:
#    - exactly four dense rectangular sprites,
#    - arranged as two horizontal face-to-face pairs,
#    - markers lie on three well-spaced rows and appear before the opposing sprite
#      when scanning along the active direction,
#    - the labeled rows are symmetric under row reversal so the lower reversed pair
#      always contains both same-color and different-color collisions.


def _dihedral_variants(template: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    ids = np.arange(template.size, dtype=int).reshape(template.shape)
    variants: list[tuple[str, np.ndarray, np.ndarray]] = []
    for k in range(4):
        rotated = np.rot90(template, k)
        rotated_ids = np.rot90(ids, k)
        variants.append((f"rot{k}", rotated, rotated_ids))
        variants.append((f"rot{k}_fliplr", np.fliplr(rotated), np.fliplr(rotated_ids)))
    return variants


def _active_side(sub: np.ndarray) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    h, w = sub.shape
    assert h != w, "sprites must be non-square rectangles"

    if h > w:
        sides = [
            ((0, -1), [(r, 0) for r in range(h)]),
            ((0, 1), [(r, w - 1) for r in range(h)]),
        ]
    else:
        sides = [
            ((-1, 0), [(0, c) for c in range(w)]),
            ((1, 0), [(h - 1, c) for c in range(w)]),
        ]

    scored = []
    for direction, coords in sides:
        vals = [int(sub[r, c]) for r, c in coords]
        scored.append((max(Counter(vals).values()), direction, set(coords)))

    scored.sort(key=lambda item: item[0], reverse=True)
    assert scored[0][0] > scored[1][0], "active side must be structurally unique"
    return scored[0][1], scored[0][2]


def _first_marker_on_ray(
    markers: dict[tuple[int, int], int],
    sprite_cells: set[tuple[int, int]],
    r: int,
    c: int,
    dr: int,
    dc: int,
    H: int,
    W: int,
) -> int | None:
    r += dr
    c += dc
    while 0 <= r < H and 0 <= c < W:
        if (r, c) in markers:
            return int(markers[(r, c)])
        if (r, c) in sprite_cells:
            return None
        r += dr
        c += dc
    return None


def _corridor_cells(
    sprite_cells: set[tuple[int, int]],
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
    H: int,
    W: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r, c = start_r, start_c
    while 0 <= r < H and 0 <= c < W:
        if (r, c) in sprite_cells:
            assert cells, "corridor must contain at least one cell before the facing sprite"
            return cells, (r, c)
        cells.append((r, c))
        r += dr
        c += dc
    raise AssertionError("every active side must face another sprite")


def _full_sprite_cells(sprites: list[dict]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for sprite in sprites:
        r0, c0, r1, c1 = sprite["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                cells.add((int(r), int(c)))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer the marker-defined template ports onto paired face-to-face sprites and resolve the rays by head-on collision.

    Input structure:
    - bg is the most common color.
    - There are four dense rectangular sprites that are dihedral transforms of one thin template.
    - Three isolated single-pixel markers sit inside the gaps between facing active sides, with one bg cell separating every marker from the sprite that labels it.

    Rule:
    - The active side of a sprite is still the long edge with the stronger same-color majority, and a marker seen first along that outward line labels the corresponding template port.
    - Every sprite emits those labeled colors from its matching active-edge ports toward the sprite that faces it.
    - When two rays on the same corridor row/column carry the same color, they merge into a solid bridge; when they carry different colors, each side paints only its half and the contested middle cell stays bg on odd-width gaps.

    Key insight:
    - The markers do not define absolute target cells anymore; they identify template ports, and the visible change comes from how the two transformed copies expose those ports in the shared face-to-face corridor.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False)

    sprites_raw: list[dict] = []
    markers: dict[tuple[int, int], int] = {}
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = int(comp["area"])
        if area == 1:
            r, c = map(int, comp["coords"][0])
            markers[(r, c)] = int(g[r, c])
        else:
            assert area == h * w, "sprite components must be dense rectangles"
            sprites_raw.append({"bbox": (r0, c0, r1, c1), "sub": crop_bbox(g, (r0, c0, r1, c1))})

    assert len(sprites_raw) == 4, "expected exactly two face-to-face pairs"
    assert len(markers) == 3, "expected exactly three control markers"
    assert len(set(markers.values())) == 3, "control marker colors must be unique"

    variants = _dihedral_variants(sprites_raw[0]["sub"])
    sprites: list[dict] = []
    for idx, sprite in enumerate(sprites_raw):
        matches = [
            (ids, name)
            for name, sub, ids in variants
            if sub.shape == sprite["sub"].shape and np.array_equal(sub, sprite["sub"])
        ]
        assert len(matches) == 1, "each sprite must match the template in exactly one orientation"
        ids, name = matches[0]
        direction, active = _active_side(sprite["sub"])
        sprites.append({**sprite, "ids": ids, "direction": direction, "active": active, "index": idx, "name": name})

    sprite_cells = _full_sprite_cells(sprites)
    coord_to_sprite: dict[tuple[int, int], int] = {}
    for sprite in sprites:
        r0, c0, r1, c1 = sprite["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                coord_to_sprite[(int(r), int(c))] = int(sprite["index"])

    H, W = g.shape
    color_to_template_id: dict[int, int] = {}
    template_id_to_color: dict[int, int] = {}
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for lr, lc in sprite["active"]:
            marker_color = _first_marker_on_ray(
                markers,
                sprite_cells,
                r0 + lr,
                c0 + lc,
                dr,
                dc,
                H,
                W,
            )
            if marker_color is None:
                continue
            template_id = int(sprite["ids"][lr, lc])
            if marker_color in color_to_template_id:
                assert color_to_template_id[marker_color] == template_id
            if template_id in template_id_to_color:
                assert template_id_to_color[template_id] == marker_color
            color_to_template_id[marker_color] = template_id
            template_id_to_color[template_id] = marker_color

    assert set(color_to_template_id) == set(markers.values()), "every marker must label one template port"

    out = g.copy()
    corridor_groups: dict[tuple[int, int, str, int], list[dict]] = {}
    for color, template_id in color_to_template_id.items():
        for sprite in sprites:
            locs = np.argwhere(sprite["ids"] == template_id)
            assert locs.shape[0] == 1
            lr, lc = map(int, locs[0])
            assert (lr, lc) in sprite["active"], "labeled port must land on the active edge"

            r0, c0, _r1, _c1 = sprite["bbox"]
            dr, dc = sprite["direction"]
            start_r = r0 + lr + dr
            start_c = c0 + lc + dc
            corridor, hit_cell = _corridor_cells(sprite_cells, start_r, start_c, dr, dc, H, W)
            dst_sprite = coord_to_sprite[hit_cell]
            axis = "h" if dr == 0 else "v"
            fixed = start_r if axis == "h" else start_c
            key = (min(int(sprite["index"]), int(dst_sprite)), max(int(sprite["index"]), int(dst_sprite)), axis, int(fixed))
            corridor_groups.setdefault(key, []).append(
                {
                    "cells": corridor,
                    "color": int(color),
                    "dr": int(dr),
                    "dc": int(dc),
                    "start": (int(start_r), int(start_c)),
                }
            )

    for records in corridor_groups.values():
        assert 1 <= len(records) <= 2, "each corridor line must have one or two emitters"
        all_cells = sorted(
            set().union(*(set(rec["cells"]) for rec in records)),
            key=lambda rc: (rc[1], rc[0]) if records[0]["dr"] == 0 else (rc[0], rc[1]),
        )

        if len(records) == 1:
            rec = records[0]
            for r, c in rec["cells"]:
                out[r, c] = int(rec["color"])
            continue

        rec_a, rec_b = sorted(
            records,
            key=lambda rec: (rec["start"][1], rec["start"][0]) if rec["dr"] == 0 else (rec["start"][0], rec["start"][1]),
        )
        color_a = int(rec_a["color"])
        color_b = int(rec_b["color"])

        if color_a == color_b:
            for r, c in all_cells:
                out[r, c] = color_a
            continue

        split = len(all_cells) // 2
        for r, c in all_cells[:split]:
            out[r, c] = color_a
        for r, c in all_cells[len(all_cells) - split :]:
            out[r, c] = color_b
        if len(all_cells) % 2 == 1:
            mr, mc = all_cells[split]
            out[mr, mc] = bg

    return out


# --------------- generator ---------------

_BG = 8
_NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
_GRID_H_RANGE = (21, 29)
_GRID_W_RANGE = (27, 29)
_TEMPLATE_N_CHOICES = [7, 9]
_TEMPLATE_COLOR_COUNT = (3, 4)
_ACTIVE_MAJORITY_RANGE = (5, 9)
_GEN_RETRIES = 400


def _build_column(n: int, colors: list[int], majority: int, dominant: int) -> list[int]:
    col = [dominant] * majority
    others = [c for c in colors if c != dominant]
    fill = others if others else colors
    for _ in range(n - majority):
        col.append(random.choice(fill))
    random.shuffle(col)
    return col


def _named_variants(
    template: np.ndarray,
    ids_base: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]]:
    raw = _dihedral_variants(template)
    raw_ids = _dihedral_variants(ids_base)
    named: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]] = {}
    seen_shapes: list[np.ndarray] = []
    for (name, sub, _), (_, _, ids_arr) in zip(raw, raw_ids):
        direction, active_set = _active_side(sub)
        named[name] = (sub, ids_arr, direction, sorted(active_set))
        if any(arr.shape == sub.shape and np.array_equal(arr, sub) for arr in seen_shapes):
            raise AssertionError("dihedral variants must be unique for the chosen template")
        seen_shapes.append(sub)
    return named


def _design_template(N: int) -> tuple[np.ndarray, np.ndarray, list[int], dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]]] | None:
    num_colors = random.randint(*_TEMPLATE_COLOR_COUNT)
    template_colors = random.sample(_NON_BG_COLORS, num_colors)

    active_majority = random.randint(_ACTIVE_MAJORITY_RANGE[0], min(N, _ACTIVE_MAJORITY_RANGE[1]))
    left_col = _build_column(N, template_colors, active_majority, random.choice(template_colors))

    max_right_majority = active_majority - 1
    if max_right_majority < 2:
        return None
    right_majority = random.randint(max(2, N // num_colors), max_right_majority)
    right_col = _build_column(N, template_colors, right_majority, random.choice(template_colors))

    if max(Counter(left_col).values()) <= max(Counter(right_col).values()):
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    if len(set(map(int, template.flat))) < num_colors:
        return None

    ids_base = np.arange(N * 2, dtype=int).reshape(N, 2)
    try:
        variants = _named_variants(template, ids_base)
    except AssertionError:
        return None

    needed = {"rot0", "rot0_fliplr", "rot2"}
    if not needed.issubset(variants):
        return None

    if variants["rot0"][2] != (0, -1):
        return None
    if variants["rot0_fliplr"][2] != (0, 1):
        return None
    if variants["rot2"][2] != (0, 1):
        return None

    return template, ids_base, template_colors, variants


def _choose_rows(N: int) -> list[int]:
    mid = N // 2
    side = random.randint(0, max(0, mid - 2))
    return [side, mid, N - 1 - side]


def _place_pair(
    grid: np.ndarray,
    top: int,
    left_x: int,
    corridor_len: int,
    left_variant: tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]],
    right_variant: tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]],
) -> tuple[dict, dict]:
    left_sub, left_ids, left_dir, left_active = left_variant
    right_sub, right_ids, right_dir, right_active = right_variant
    h, w = left_sub.shape
    assert right_sub.shape == (h, w)
    right_x = left_x + w + corridor_len

    grid[top : top + h, left_x : left_x + w] = left_sub
    grid[top : top + h, right_x : right_x + w] = right_sub

    left_sprite = {
        "bbox": (top, left_x, top + h - 1, left_x + w - 1),
        "sub": left_sub,
        "ids": left_ids,
        "direction": left_dir,
        "active": set(left_active),
    }
    right_sprite = {
        "bbox": (top, right_x, top + h - 1, right_x + w - 1),
        "sub": right_sub,
        "ids": right_ids,
        "direction": right_dir,
        "active": set(right_active),
    }
    return left_sprite, right_sprite


def _marker_position(sprite: dict, template_id: int, offset: int) -> tuple[int, int]:
    r0, c0, _r1, _c1 = sprite["bbox"]
    dr, dc = sprite["direction"]
    locs = np.argwhere(sprite["ids"] == template_id)
    assert locs.shape[0] == 1
    lr, lc = map(int, locs[0])
    assert (lr, lc) in sprite["active"]
    return int(r0 + lr + dr * offset), int(c0 + lc + dc * offset)


def _gen_attempt() -> np.ndarray | None:
    N = random.choice(_TEMPLATE_N_CHOICES)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)

    design = _design_template(N)
    if design is None:
        return None
    _template, ids_base, template_colors, variants = design

    if H < 2 * N + 5:
        return None

    max_corridor = W - 8
    odd_corridors = [v for v in range(7, max_corridor + 1) if v % 2 == 1]
    if not odd_corridors:
        return None

    top_corridor = random.choice(odd_corridors)
    bottom_corridor = random.choice(odd_corridors)

    top_rows = _choose_rows(N)
    template_ids = [int(ids_base[r, 0]) for r in top_rows]
    available_marker_colors = [c for c in _NON_BG_COLORS if c not in template_colors]
    if len(available_marker_colors) < 3:
        return None
    marker_colors = random.sample(available_marker_colors, 3)

    left_margin_top = random.randint(1, W - (top_corridor + 4))
    left_margin_bottom = random.randint(1, W - (bottom_corridor + 4))

    top_pad = random.randint(1, 3)
    bottom_pad = random.randint(1, 3)
    top_y = top_pad
    bottom_y = H - N - bottom_pad
    if bottom_y - (top_y + N) < 3:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    top_left, top_right = _place_pair(
        grid,
        top_y,
        left_margin_top,
        top_corridor,
        variants["rot0_fliplr"],
        variants["rot0"],
    )
    bottom_left, bottom_right = _place_pair(
        grid,
        bottom_y,
        left_margin_bottom,
        bottom_corridor,
        variants["rot2"],
        variants["rot0"],
    )

    marker_positions: list[tuple[int, int]] = []
    for color, template_id in zip(marker_colors, template_ids):
        mr, mc = _marker_position(top_left, template_id, offset=2)
        if not (0 <= mr < H and 0 <= mc < W):
            return None
        if grid[mr, mc] != _BG:
            return None
        marker_positions.append((mr, mc))
        grid[mr, mc] = int(color)

    if len(set(marker_positions)) != 3:
        return None

    # Keep markers isolated under 8-connectivity.
    for i, (r1, c1) in enumerate(marker_positions):
        for r2, c2 in marker_positions[i + 1 :]:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return None

    try:
        out = main(grid)
    except Exception:
        return None

    # Reject degenerate samples where the lower reversed pair behaves like same-order.
    diff_rows = 0
    for row in top_rows:
        r = bottom_y + row
        c_left = left_margin_bottom + 2
        c_right = left_margin_bottom + 1 + bottom_corridor
        span = out[r, c_left : c_right + 1]
        non_bg = [int(v) for v in span if int(v) != _BG]
        if len(set(non_bg)) >= 2:
            diff_rows += 1
    if diff_rows < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
