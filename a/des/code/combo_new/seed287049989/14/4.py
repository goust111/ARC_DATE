from __future__ import annotations

import numpy as np


BG = 0
TEMPLATE = 5
FRAGMENT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GEN_ATTEMPTS = 400

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _border_mode(grid: np.ndarray) -> int:
    border = np.concatenate(
        [grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]]
    )
    vals, counts = np.unique(border, return_counts=True)
    return int(vals[counts.argmax()])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        neighbors = NEIGHBORS4
    else:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        pts: list[tuple[int, int]] = []
        while stack:
            cr, cc2 = stack.pop()
            pts.append((cr, cc2))
            for dr, dc in neighbors:
                nr = cr + dr
                nc = cc2 + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, (r0, c0)


def _exact_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(bool).tobytes()


def _all_dihedral(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            rot = np.rot90(base, k)
            key = _exact_key(rot)
            if key not in seen:
                seen.add(key)
                variants.append(rot.copy())
    return variants


def _free_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_exact_key(v) for v in _all_dihedral(mask))


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = _dilate4(mask)
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    arr = np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)
    coords = np.argwhere(arr)
    mask, _ = _tight_mask(coords)
    return mask


def _build_shape_library() -> list[np.ndarray]:
    raw = [
        ["###"],
        ["#.", "##"],
        ["####"],
        ["##", "##"],
        ["###", ".#."],
        ["##.", ".##"],
        ["#..", "###"],
        ["#####"],
        ["###", ".#.", ".#."],
        ["#..", "#..", "###"],
        ["##", "##", "#."],
        [".#.", "###", ".#."],
        ["###", "#.#"],
        [".##", "##.", "#.."],
        ["######"],
        ["###", "###"],
        ["####", "..##"],
        [".#.", "###", ".##"],
    ]
    library: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for rows in raw:
        mask = _mask_from_rows(rows)
        area = int(mask.sum())
        if not (3 <= area <= 6):
            continue
        key = _free_key(mask)
        if key in seen:
            continue
        seen.add(key)
        library.append(mask)
    return library


SHAPE_LIBRARY = _build_shape_library()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble each colored fragment into the matching slot and recolor that slot's grey border.

    Input structure:
    - The background is Black.
    - One large monochrome template component sits in the upper part of the grid; inside its bounding box are several enclosed Black slot components.
    - Several small monochrome colored fragments are scattered elsewhere on the Black background, and each fragment has the exact same shape as one slot under translation only.

    Rule:
    - Identify the large template component and extract each enclosed Black slot inside it as a 4-connected shape.
    - Match every outside colored fragment to the unique slot with the same tight-cropped shape, without rotation or reflection.
    - Replace the slot cells with the fragment color and also recolor every template pixel that is 4-neighbor adjacent to that slot.
    - Clear the fragment's original location to Black and leave all other template pixels unchanged.

    Key insight:
    The fragment-to-slot correspondence is determined purely by exact translated shape matching, and the color must spread not only into the slot but also onto its local grey border.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _border_mode(g)

    comps: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for coords in _components(g == color, connectivity=4):
            comps.append({"color": color, "coords": coords, "area": int(len(coords))})
    assert comps, "expected at least one foreground component"

    template = max(comps, key=lambda item: int(item["area"]))
    template_coords = np.asarray(template["coords"], dtype=int)
    template_mask = np.zeros((h, w), dtype=bool)
    template_mask[template_coords[:, 0], template_coords[:, 1]] = True

    tr0, tc0, tr1, tc1 = _bbox(template_coords)
    sub = g[tr0 : tr1 + 1, tc0 : tc1 + 1]

    slots: list[dict[str, object]] = []
    for sub_coords in _components(sub == bg, connectivity=4):
        if np.any(sub_coords[:, 0] == 0) or np.any(sub_coords[:, 0] == sub.shape[0] - 1):
            continue
        if np.any(sub_coords[:, 1] == 0) or np.any(sub_coords[:, 1] == sub.shape[1] - 1):
            continue
        slot_mask, _ = _tight_mask(sub_coords)
        global_coords = sub_coords + np.array([tr0, tc0], dtype=int)
        border_mask = np.zeros((h, w), dtype=bool)
        for rr, cc in global_coords:
            for dr, dc in NEIGHBORS4:
                nr = int(rr) + dr
                nc = int(cc) + dc
                if 0 <= nr < h and 0 <= nc < w and template_mask[nr, nc]:
                    border_mask[nr, nc] = True
        slots.append(
            {
                "key": _exact_key(slot_mask),
                "cells": global_coords,
                "border": border_mask,
            }
        )
    assert slots, "expected at least one enclosed slot"

    fragments: list[dict[str, object]] = []
    for comp in comps:
        if comp is template:
            continue
        coords = np.asarray(comp["coords"], dtype=int)
        frag_mask, _ = _tight_mask(coords)
        fragments.append(
            {
                "key": _exact_key(frag_mask),
                "coords": coords,
                "color": int(comp["color"]),
            }
        )
    assert len(fragments) == len(slots), "fragment count must match slot count"

    fragment_by_key: dict[tuple[tuple[int, int], bytes], dict[str, object]] = {}
    for frag in fragments:
        key = frag["key"]
        assert key not in fragment_by_key, "fragment shapes must be unique"
        fragment_by_key[key] = frag

    out = g.copy()
    for frag in fragments:
        coords = np.asarray(frag["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = bg

    for slot in slots:
        key = slot["key"]
        assert key in fragment_by_key, "every slot must have a matching fragment"
        color = int(fragment_by_key[key]["color"])
        cells = np.asarray(slot["cells"], dtype=int)
        out[cells[:, 0], cells[:, 1]] = color
        out[np.asarray(slot["border"], dtype=bool)] = color

    return out


def _random_shapes(rng: np.random.Generator, count: int) -> list[np.ndarray]:
    picks = rng.choice(len(SHAPE_LIBRARY), size=count, replace=False)
    shapes: list[np.ndarray] = []
    used: set[tuple[tuple[int, int], bytes]] = set()
    for idx in picks:
        variants = _all_dihedral(SHAPE_LIBRARY[int(idx)])
        rng.shuffle(variants)
        chosen = None
        for variant in variants:
            key = _exact_key(variant)
            if key not in used:
                chosen = variant
                used.add(key)
                break
        assert chosen is not None
        shapes.append(chosen)
    return shapes


def _place_slots(
    rng: np.random.Generator, shapes: list[np.ndarray]
) -> tuple[int, int, list[dict[str, object]]] | None:
    order = sorted(
        [{"shape": s.copy(), "area": int(s.sum())} for s in shapes],
        key=lambda item: (-int(item["area"]), -int(np.prod(item["shape"].shape))),
    )
    max_h = max(int(item["shape"].shape[0]) for item in order)
    max_w = max(int(item["shape"].shape[1]) for item in order)
    count = len(order)

    for _ in range(120):
        th_lo = max(10, max_h + 5, 8 + count)
        tw_lo = max(12, max_w + 5, 10 + 2 * count)
        th_hi = min(17, th_lo + 4)
        tw_hi = min(22, tw_lo + 5)
        if th_lo > th_hi or tw_lo > tw_hi:
            continue
        template_h = int(rng.integers(th_lo, th_hi + 1))
        template_w = int(rng.integers(tw_lo, tw_hi + 1))

        forbidden = np.zeros((template_h, template_w), dtype=bool)
        placed: list[dict[str, object]] = []
        ok = True
        for item in order:
            shape = np.asarray(item["shape"], dtype=bool)
            sh, sw = shape.shape
            if template_h - sh - 4 < 0 or template_w - sw - 4 < 0:
                ok = False
                break
            success = False
            for _ in range(240):
                r0 = int(rng.integers(2, template_h - sh - 1))
                c0 = int(rng.integers(2, template_w - sw - 1))
                hole = np.zeros((template_h, template_w), dtype=bool)
                hole[r0 : r0 + sh, c0 : c0 + sw] = shape
                recolor = _dilate4(hole)
                if np.any(recolor[0, :]) or np.any(recolor[-1, :]) or np.any(recolor[:, 0]) or np.any(recolor[:, -1]):
                    continue
                if (forbidden & recolor).any():
                    continue
                forbidden |= recolor
                placed.append({"shape": shape, "hole": hole, "recolor": recolor})
                success = True
                break
            if not success:
                ok = False
                break
        if ok:
            return template_h, template_w, placed
    return None


def _place_fragments(
    rng: np.random.Generator,
    zone_h: int,
    zone_w: int,
    items: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    ordered = sorted(
        items,
        key=lambda item: (-int(np.asarray(item["shape"]).sum()), -int(np.prod(np.asarray(item["shape"]).shape))),
    )
    forbidden = np.zeros((zone_h, zone_w), dtype=bool)
    placed: list[dict[str, object]] = []
    for item in ordered:
        shape = np.asarray(item["shape"], dtype=bool)
        sh, sw = shape.shape
        if sh > zone_h or sw > zone_w:
            return None
        success = False
        for _ in range(240):
            r0 = int(rng.integers(0, zone_h - sh + 1))
            c0 = int(rng.integers(0, zone_w - sw + 1))
            cells = np.zeros((zone_h, zone_w), dtype=bool)
            cells[r0 : r0 + sh, c0 : c0 + sw] = shape
            if (forbidden & cells).any():
                continue
            forbidden |= _dilate8(cells)
            placed.append(
                {
                    "shape": shape,
                    "color": int(item["color"]),
                    "r0": r0,
                    "c0": c0,
                }
            )
            success = True
            break
        if not success:
            return None
    return placed


def _build_expected(
    inp: np.ndarray,
    template_top: int,
    template_left: int,
    slots: list[dict[str, object]],
    fragments: list[dict[str, object]],
) -> np.ndarray:
    out = inp.copy()
    for frag in fragments:
        shape = np.asarray(frag["shape"], dtype=bool)
        r0 = int(frag["r0"])
        c0 = int(frag["c0"])
        gr0 = int(frag["grid_r0"])
        gc0 = int(frag["grid_c0"])
        patch = out[gr0 : gr0 + shape.shape[0], gc0 : gc0 + shape.shape[1]]
        patch[shape] = BG
        out[gr0 : gr0 + shape.shape[0], gc0 : gc0 + shape.shape[1]] = patch

    template_patch = out[
        template_top : template_top + slots[0]["hole"].shape[0],
        template_left : template_left + slots[0]["hole"].shape[1],
    ]
    for slot, frag in zip(slots, fragments, strict=True):
        template_patch[np.asarray(slot["recolor"], dtype=bool)] = int(frag["color"])
    out[
        template_top : template_top + slots[0]["hole"].shape[0],
        template_left : template_left + slots[0]["hole"].shape[1],
    ] = template_patch
    return out


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    count = int(rng.integers(2, 5))
    shapes = _random_shapes(rng, count)
    colors = [int(v) for v in rng.choice(np.array(FRAGMENT_COLORS), size=count, replace=False)]

    placed_slots = _place_slots(rng, shapes)
    if placed_slots is None:
        return None
    template_h, template_w, slots = placed_slots

    top_margin = int(rng.integers(1, 4))
    gap = int(rng.integers(2, 5))
    bottom_margin = int(rng.integers(1, 3))
    zone_h = int(rng.integers(7, 11))
    grid_h = top_margin + template_h + gap + zone_h + bottom_margin
    if grid_h > 30:
        return None

    grid_w_lo = max(template_w + 4, 18)
    grid_w_hi = min(30, max(grid_w_lo, template_w + 10))
    if grid_w_lo > grid_w_hi:
        return None
    grid_w = int(rng.integers(grid_w_lo, grid_w_hi + 1))

    template_top = top_margin
    centered_left = (grid_w - template_w) // 2
    left_lo = max(1, centered_left - 1)
    left_hi = min(grid_w - template_w - 1, centered_left + 1)
    if left_lo > left_hi:
        return None
    template_left = int(rng.integers(left_lo, left_hi + 1))

    items = []
    for slot, color in zip(slots, colors, strict=True):
        items.append({"shape": np.asarray(slot["shape"], dtype=bool), "color": int(color), "slot": slot})

    zone_w = grid_w - 2
    fragment_layout = _place_fragments(rng, zone_h, zone_w, items)
    if fragment_layout is None:
        return None

    by_key = {
        _exact_key(np.asarray(slot["shape"], dtype=bool)): {"slot": slot, "color": color}
        for slot, color in zip(slots, colors, strict=True)
    }

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    inp[
        template_top : template_top + template_h,
        template_left : template_left + template_w,
    ] = TEMPLATE
    for slot in slots:
        hole = np.asarray(slot["hole"], dtype=bool)
        patch = inp[
            template_top : template_top + template_h,
            template_left : template_left + template_w,
        ]
        patch[hole] = BG
        inp[
            template_top : template_top + template_h,
            template_left : template_left + template_w,
        ] = patch

    fragments: list[dict[str, object]] = []
    for frag in fragment_layout:
        shape = np.asarray(frag["shape"], dtype=bool)
        key = _exact_key(shape)
        color = int(by_key[key]["color"])
        grid_r0 = template_top + template_h + gap + int(frag["r0"])
        grid_c0 = 1 + int(frag["c0"])
        patch = inp[grid_r0 : grid_r0 + shape.shape[0], grid_c0 : grid_c0 + shape.shape[1]]
        patch[shape] = color
        inp[grid_r0 : grid_r0 + shape.shape[0], grid_c0 : grid_c0 + shape.shape[1]] = patch
        fragments.append(
            {
                "shape": shape,
                "color": color,
                "r0": int(frag["r0"]),
                "c0": int(frag["c0"]),
                "grid_r0": grid_r0,
                "grid_c0": grid_c0,
            }
        )

    fragments.sort(key=lambda item: _exact_key(np.asarray(item["shape"], dtype=bool)))
    slots_sorted = sorted(slots, key=lambda item: _exact_key(np.asarray(item["shape"], dtype=bool)))
    out = _build_expected(inp, template_top, template_left, slots_sorted, fragments)

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        case = _generate_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid ARC input")
