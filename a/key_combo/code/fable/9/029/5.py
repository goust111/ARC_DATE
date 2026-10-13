from __future__ import annotations

import random
from collections import Counter

import numpy as np


ANCHORS = ("tl", "tr", "bl", "br")
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
BACKGROUND = 0
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIR4 if int(connectivity) == 4 else DIR8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    k = int(factor)
    return np.kron(sprite, np.ones((k, k), dtype=int))


def _sprite_top_left(anchor: str, h: int, w: int) -> tuple[int, int]:
    if anchor == "tl":
        return 0, 0
    if anchor == "tr":
        return 0, -(w - 1)
    if anchor == "bl":
        return -(h - 1), 0
    if anchor == "br":
        return -(h - 1), -(w - 1)
    raise ValueError(anchor)


def _anchor_corner_of_bbox(bbox: tuple[int, int, int, int], anchor: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if anchor == "tl":
        return r0, c0
    if anchor == "tr":
        return r0, c1
    if anchor == "bl":
        return r1, c0
    if anchor == "br":
        return r1, c1
    raise ValueError(anchor)


def _expand_bbox_for_frame(
    bbox: tuple[int, int, int, int], anchor: str
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    far_r = random.randint(1, 3)
    far_c = random.randint(1, 3)
    if anchor == "tl":
        return r0 - 1, c0 - 1, r1 + far_r, c1 + far_c
    if anchor == "tr":
        return r0 - 1, c0 - far_c, r1 + far_r, c1 + 1
    if anchor == "bl":
        return r0 - far_r, c0 - 1, r1 + 1, c1 + far_c
    if anchor == "br":
        return r0 - far_r, c0 - far_c, r1 + 1, c1 + 1
    raise ValueError(anchor)


def _is_rect_frame(coords: np.ndarray) -> bool:
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = _bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    rel = coords - np.array([r0, c0])
    on_border = (
        (rel[:, 0] == 0)
        | (rel[:, 0] == h - 1)
        | (rel[:, 1] == 0)
        | (rel[:, 1] == w - 1)
    )
    return bool(np.all(on_border)) and int(coords.shape[0]) == 2 * (h + w) - 4


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _pair_matches(
    parent: tuple[int, int, int, int], child: tuple[int, int, int, int], anchor: str
) -> bool:
    pr0, pc0, pr1, pc1 = parent
    cr0, cc0, cr1, cc1 = child
    if not (pr0 <= cr0 <= cr1 <= pr1 and pc0 <= cc0 <= cc1 <= pc1):
        return False
    if anchor == "tl":
        return cr0 == pr0 + 1 and cc0 == pc0 + 1 and pr1 - cr1 >= 1 and pc1 - cc1 >= 1
    if anchor == "tr":
        return cr0 == pr0 + 1 and cc1 == pc1 - 1 and pr1 - cr1 >= 1 and cc0 - pc0 >= 1
    if anchor == "bl":
        return cr1 == pr1 - 1 and cc0 == pc0 + 1 and cr0 - pr0 >= 1 and pc1 - cc1 >= 1
    if anchor == "br":
        return cr1 == pr1 - 1 and cc1 == pc1 - 1 and cr0 - pr0 >= 1 and cc0 - pc0 >= 1
    raise ValueError(anchor)


def _sprite_has_anchor_pixel(sprite: np.ndarray, anchor: str, bg: int) -> bool:
    h, w = sprite.shape
    if anchor == "tl":
        return int(sprite[0, 0]) != bg
    if anchor == "tr":
        return int(sprite[0, w - 1]) != bg
    if anchor == "bl":
        return int(sprite[h - 1, 0]) != bg
    if anchor == "br":
        return int(sprite[h - 1, w - 1]) != bg
    raise ValueError(anchor)


def _local_sprite_coords(sprite: np.ndarray, anchor: str, bg: int) -> list[tuple[int, int, int]]:
    h, w = sprite.shape
    tr, tc = _sprite_top_left(anchor, h, w)
    cells: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            color = int(sprite[r, c])
            if color == bg:
                continue
            cells.append((tr + r, tc + c, color))
    return cells


def _frame_border_coords(
    bbox: tuple[int, int, int, int], color: int
) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c, color))
        cells.append((r1, c, color))
    for r in range(r0 + 1, r1):
        cells.append((r, c0, color))
        cells.append((r, c1, color))
    return cells


def _render_group_output_coords(
    sprite: np.ndarray, anchor: str, k: int, bg: int
) -> list[tuple[int, int, int]]:
    scaled = _scale_sprite(sprite, k)
    return _local_sprite_coords(scaled, anchor, bg)


def _group_extents(group: dict) -> tuple[int, int, int, int]:
    all_cells = group["input_cells"] + group["output_cells"]
    rows = [r for r, _c, _color in all_cells]
    cols = [c for _r, c, _color in all_cells]
    return min(rows), min(cols), max(rows), max(cols)


def _shift_cells(
    cells: list[tuple[int, int, int]], dr: int, dc: int
) -> list[tuple[int, int, int]]:
    return [(r + dr, c + dc, color) for r, c, color in cells]


def _blocked_with_padding(
    cells: list[tuple[int, int, int]], h: int, w: int
) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c, _color in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked.add((nr, nc))
    return blocked


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int, int]]) -> None:
    for r, c, color in cells:
        assert 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
        grid[r, c] = int(color)


def _monotone_path(
    start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    path = [(r, c)]
    row_steps = [1 if tr > r else -1] * abs(tr - r)
    col_steps = [1 if tc > c else -1] * abs(tc - c)
    moves = [("r", step) for step in row_steps] + [("c", step) for step in col_steps]
    random.shuffle(moves)
    for axis, step in moves:
        if axis == "r":
            r += step
        else:
            c += step
        path.append((r, c))
    return path


def _grow_connected_region(
    h: int, w: int, anchor: str, target_area: int
) -> np.ndarray | None:
    anchor_cell = {
        "tl": (0, 0),
        "tr": (0, w - 1),
        "bl": (h - 1, 0),
        "br": (h - 1, w - 1),
    }[anchor]
    far_cell = {
        "tl": (h - 1, w - 1),
        "tr": (h - 1, 0),
        "bl": (0, w - 1),
        "br": (0, 0),
    }[anchor]

    cells = set(_monotone_path(anchor_cell, far_cell))
    frontier = set()
    while len(cells) < target_area:
        frontier.clear()
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        choice = random.choice(tuple(frontier))
        cells.add(choice)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _sprite_is_asymmetric(sprite: np.ndarray) -> bool:
    return not (
        np.array_equal(sprite, sprite[::-1, :])
        or np.array_equal(sprite, sprite[:, ::-1])
        or np.array_equal(sprite, sprite[::-1, ::-1])
    )


def _make_sprite(anchor: str, disallow_colors: set[int]) -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        target_area = random.randint(max(3, (h * w + 1) // 2), min(h * w, h * w - 1 + (h * w <= 4)))
        mask = _grow_connected_region(h, w, anchor, target_area)
        if mask is None:
            continue

        coords = np.argwhere(mask)
        if _bbox_coords(coords) != (0, 0, h - 1, w - 1):
            continue

        palette = [c for c in range(1, 10) if c not in disallow_colors]
        n_colors = 2 if target_area >= 4 else 1
        if target_area >= 6 and random.random() < 0.5:
            n_colors = 3
        colors = random.sample(palette, n_colors)

        sprite = np.zeros((h, w), dtype=int)
        region = list(map(tuple, coords))
        random.shuffle(region)
        if len(colors) == 1:
            for r, c in region:
                sprite[r, c] = colors[0]
        else:
            ordered = sorted(region, key=lambda rc: (rc[0] + rc[1], rc[0], rc[1]))
            cuts = []
            if len(colors) == 2:
                split = random.randint(1, len(ordered) - 1)
                cuts = [split]
            else:
                split1 = random.randint(1, len(ordered) - 2)
                split2 = random.randint(split1 + 1, len(ordered) - 1)
                cuts = [split1, split2]
            parts = np.split(np.array(ordered, dtype=int), cuts)
            for color, part in zip(colors, parts):
                for r, c in part:
                    sprite[int(r), int(c)] = int(color)

        if not _sprite_has_anchor_pixel(sprite, anchor, BACKGROUND):
            continue
        if target_area >= 4 and not _sprite_is_asymmetric(sprite):
            continue

        bad = False
        for color in np.unique(sprite):
            if int(color) == BACKGROUND:
                continue
            for comp in _components_mask(sprite == int(color), connectivity=4):
                if _is_rect_frame(comp):
                    bad = True
                    break
            if bad:
                break
        if bad:
            continue
        return sprite
    return None


def _make_group(anchor: str, k: int) -> dict | None:
    for _ in range(120):
        sprite = _make_sprite(anchor, set())
        if sprite is None:
            continue
        sprite_colors = set(map(int, np.unique(sprite))) - {BACKGROUND}
        frame_palette = [c for c in range(1, 10) if c not in sprite_colors]
        if len(frame_palette) < k:
            continue
        frame_colors = random.sample(frame_palette, k)

        h, w = sprite.shape
        top, left = _sprite_top_left(anchor, h, w)
        cur_bbox = (top, left, top + h - 1, left + w - 1)
        inner_to_outer = []
        for _ in range(k):
            cur_bbox = _expand_bbox_for_frame(cur_bbox, anchor)
            inner_to_outer.append(cur_bbox)
        frames = list(reversed(inner_to_outer))

        input_cells = _local_sprite_coords(sprite, anchor, BACKGROUND)
        for color, bbox in zip(frame_colors, reversed(frames)):
            input_cells.extend(_frame_border_coords(bbox, color))

        output_cells = _render_group_output_coords(sprite, anchor, k, BACKGROUND)

        in_rows = [r for r, _c, _color in input_cells]
        in_cols = [c for _r, c, _color in input_cells]
        out_rows = [r for r, _c, _color in output_cells]
        out_cols = [c for _r, c, _color in output_cells]
        if max(in_rows) - min(in_rows) + 1 > 14 or max(in_cols) - min(in_cols) + 1 > 14:
            continue
        if max(out_rows) - min(out_rows) + 1 > 12 or max(out_cols) - min(out_cols) + 1 > 12:
            continue

        return {
            "anchor": anchor,
            "k": int(k),
            "sprite": sprite,
            "frames_outer_to_inner": frames,
            "frame_colors_outer_to_inner": list(reversed(frame_colors)),
            "input_cells": input_cells,
            "output_cells": output_cells,
        }
    return None


def _choose_group_parameters() -> list[tuple[str, int]]:
    n_groups = random.randint(3, 4)
    anchors = random.sample(list(ANCHORS), random.randint(2, min(4, n_groups)))
    while len(anchors) < n_groups:
        anchors.append(random.choice(anchors))
    random.shuffle(anchors)

    repeated_k = random.randint(2, 4)
    other_ks = [k for k in (2, 3, 4) if k != repeated_k]
    k_values = [repeated_k, repeated_k]
    while len(k_values) < n_groups:
        k_values.append(random.choice(other_ks))
    random.shuffle(k_values)

    if len(set(k_values)) < 2:
        k_values[0] = 2
        k_values[1] = 3
    if len(set(anchors)) < 2:
        anchors[0] = "tl"
        anchors[1] = "br"

    return list(zip(anchors, k_values))


def _build_groups() -> list[dict] | None:
    params = _choose_group_parameters()
    groups = []
    seen_output_signatures: set[tuple] = set()
    for anchor, k in params:
        group = _make_group(anchor, k)
        if group is None:
            return None
        sig = tuple(sorted(group["output_cells"]))
        if sig in seen_output_signatures:
            return None
        seen_output_signatures.add(sig)
        groups.append(group)
    return groups


def _place_groups(groups: list[dict], h: int, w: int) -> list[dict] | None:
    placed: list[dict] = []
    input_blocked: set[tuple[int, int]] = set()
    output_blocked: set[tuple[int, int]] = set()

    ordered = sorted(groups, key=lambda g: len(g["output_cells"]) + len(g["input_cells"]), reverse=True)
    for group in ordered:
        mn_r, mn_c, mx_r, mx_c = _group_extents(group)
        candidates = [
            (ar, ac)
            for ar in range(-mn_r, h - mx_r)
            for ac in range(-mn_c, w - mx_c)
        ]
        random.shuffle(candidates)

        chosen = None
        for ar, ac in candidates:
            shifted_in = _shift_cells(group["input_cells"], ar, ac)
            shifted_out = _shift_cells(group["output_cells"], ar, ac)
            if any((r, c) in input_blocked for r, c, _color in shifted_in):
                continue
            if any((r, c) in output_blocked for r, c, _color in shifted_out):
                continue
            chosen = (ar, ac, shifted_in, shifted_out)
            break

        if chosen is None:
            return None

        ar, ac, shifted_in, shifted_out = chosen
        input_blocked |= _blocked_with_padding(shifted_in, h, w)
        output_blocked |= _blocked_with_padding(shifted_out, h, w)
        spec = dict(group)
        spec["anchor_rc"] = (int(ar), int(ac))
        spec["input_cells_abs"] = shifted_in
        spec["output_cells_abs"] = shifted_out
        placed.append(spec)

    return placed


def _render_case(placed: list[dict], h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    for spec in placed:
        _paint_cells(inp, spec["input_cells_abs"])
        _paint_cells(out, spec["output_cells_abs"])
    return inp, out


def _analyze_group(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict:
    r0, c0, r1, c1 = _bbox_coords(coords)
    crop = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    frame_entries = []
    frame_mask = np.zeros_like(crop, dtype=bool)

    for color in sorted(set(map(int, crop.ravel())) - {bg}):
        for comp in _components_mask(crop == int(color), connectivity=4):
            if _is_rect_frame(comp):
                b = _bbox_coords(comp)
                frame_entries.append({"color": int(color), "bbox": b, "coords": comp})
                frame_mask[comp[:, 0], comp[:, 1]] = True

    assert frame_entries, "expected at least one frame"
    frame_entries.sort(key=lambda item: _bbox_area(item["bbox"]), reverse=True)
    frame_areas = [_bbox_area(item["bbox"]) for item in frame_entries]
    assert len(frame_areas) == len(set(frame_areas)), "frame areas must define a unique nesting order"

    sprite_mask = (crop != bg) & (~frame_mask)
    assert bool(sprite_mask.any()), "expected a residual sprite inside the frames"
    sprite_bbox = _bbox_coords(np.argwhere(sprite_mask))
    sr0, sc0, sr1, sc1 = sprite_bbox
    sprite = np.full((sr1 - sr0 + 1, sc1 - sc0 + 1), bg, dtype=int)
    sprite_view = crop[sr0 : sr1 + 1, sc0 : sc1 + 1]
    sprite_local_mask = sprite_mask[sr0 : sr1 + 1, sc0 : sc1 + 1]
    sprite[sprite_local_mask] = sprite_view[sprite_local_mask]

    anchors = []
    chain = [entry["bbox"] for entry in frame_entries] + [sprite_bbox]
    for anchor in ANCHORS:
        if not _sprite_has_anchor_pixel(sprite, anchor, bg):
            continue
        if all(_pair_matches(parent, child, anchor) for parent, child in zip(chain, chain[1:])):
            anchors.append(anchor)
    assert len(anchors) == 1, "group anchor must be uniquely determined"
    anchor = anchors[0]

    sprite_bbox_global = (r0 + sr0, c0 + sc0, r0 + sr1, c0 + sc1)
    anchor_rc = _anchor_corner_of_bbox(sprite_bbox_global, anchor)
    return {
        "anchor": anchor,
        "k": len(frame_entries),
        "sprite": sprite,
        "anchor_rc": anchor_rc,
        "frame_bboxes": [entry["bbox"] for entry in frame_entries],
    }


def _render_scaled_sprite_abs(
    canvas: np.ndarray, sprite: np.ndarray, anchor: str, anchor_rc: tuple[int, int], k: int, bg: int
) -> set[tuple[int, int, int]]:
    scaled = _scale_sprite(sprite, k)
    sh, sw = scaled.shape
    top, left = _sprite_top_left(anchor, sh, sw)
    base_r = int(anchor_rc[0]) + top
    base_c = int(anchor_rc[1]) + left
    cells: set[tuple[int, int, int]] = set()
    for r in range(sh):
        for c in range(sw):
            color = int(scaled[r, c])
            if color == bg:
                continue
            rr = base_r + r
            cc = base_c + c
            assert 0 <= rr < canvas.shape[0] and 0 <= cc < canvas.shape[1], "scaled sprite left the grid"
            assert int(canvas[rr, cc]) in (bg, color), "scaled sprites must not overlap"
            canvas[rr, cc] = color
            cells.add((rr, cc, color))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each nested-frame group and enlarge its inner sprite by the frame count, anchored at the shared corner.

    Input structure:
    - The grid has a black background and several disconnected multicolor groups.
    - Each group contains one small sprite plus one or more hollow rectangular frames nested around it.
    - Within a group, every inner box is tucked into the same corner of the next outer box, and the sprite uses that same corner as its anchor side.

    Rule:
    - For each group, count how many hollow frames it has and infer which corner is shared by the whole nesting chain.
    - Delete all frame pixels.
    - Enlarge the inner sprite by that frame count: every sprite pixel becomes a same-color k x k block.
    - Keep the sprite corner nearest the shared nesting corner fixed, and expand only away from that corner.

    Key insight:
    - The frames are only control signals: their depth gives the enlargement factor, and their common aligned corner gives the enlargement anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    groups = _components_mask(g != bg, connectivity=4)
    assert groups, "expected at least one foreground group"

    out = np.full_like(g, bg)
    for coords in groups:
        info = _analyze_group(g, coords, bg)
        _render_scaled_sprite_abs(out, info["sprite"], info["anchor"], info["anchor_rc"], info["k"], bg)

    return out


def _validate_case(inp: np.ndarray, expected: np.ndarray, specs: list[dict]) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.count_nonzero(inp == BACKGROUND) <= inp.size // 2:
        return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    groups = _components_mask(inp != BACKGROUND, connectivity=4)
    if len(groups) != len(specs):
        return False

    analyzed = [_analyze_group(inp, coords, BACKGROUND) for coords in groups]
    by_anchor = {tuple(info["anchor_rc"]): info for info in analyzed}

    if len({spec["anchor"] for spec in specs}) < 2:
        return False
    if len({spec["k"] for spec in specs}) < 2:
        return False

    repeated = Counter(spec["k"] for spec in specs)
    if max(repeated.values()) < 2:
        return False
    repeated_k = max(repeated, key=lambda k: repeated[k])
    repeated_specs = [spec for spec in specs if spec["k"] == repeated_k]
    if len({spec["anchor"] for spec in repeated_specs}) < 2:
        return False

    output_signatures = set()
    for spec in specs:
        key = tuple(spec["anchor_rc"])
        if key not in by_anchor:
            return False
        recovered = by_anchor[key]
        if recovered["anchor"] != spec["anchor"] or recovered["k"] != spec["k"]:
            return False
        if not np.array_equal(recovered["sprite"], spec["sprite"]):
            return False

        correct = set(map(tuple, spec["output_cells_abs"]))
        output_signatures.add(tuple(sorted(correct)))

        for other_anchor in ANCHORS:
            if other_anchor == spec["anchor"]:
                continue
            trial = np.zeros_like(expected)
            try:
                alt = _render_scaled_sprite_abs(
                    trial, spec["sprite"], other_anchor, spec["anchor_rc"], spec["k"], BACKGROUND
                )
            except AssertionError:
                alt = set()
            if alt == correct:
                return False

        for other_k in (2, 3, 4):
            if other_k == spec["k"]:
                continue
            trial = np.zeros_like(expected)
            try:
                alt = _render_scaled_sprite_abs(
                    trial, spec["sprite"], spec["anchor"], spec["anchor_rc"], other_k, BACKGROUND
                )
            except AssertionError:
                alt = set()
            if alt == correct:
                return False

    if len(output_signatures) != len(specs):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        groups = _build_groups()
        if groups is None:
            continue

        h = random.randint(22, 30)
        w = random.randint(22, 30)
        placed = _place_groups(groups, h, w)
        if placed is None:
            continue

        inp, expected = _render_case(placed, h, w)
        if _validate_case(inp, expected, placed):
            return inp
    raise RuntimeError("failed to generate a valid nested-frame upscale puzzle")
