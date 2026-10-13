from __future__ import annotations

import random
from collections import Counter, deque
from typing import Iterable

import numpy as np


BG = 0
BAR = 1
SEPARATOR = 5
SPRITE_COLORS = (2, 3, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    cells = list(cells)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_shape(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    cells = list(cells)
    r0, c0, _, _ = _bbox(cells)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _shape_mask(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _shape_dims(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _shape_key(shape: tuple[tuple[int, int], ...]) -> tuple[bytes, tuple[int, int]]:
    mask = _shape_mask(shape)
    return mask.tobytes(), mask.shape


def _shape_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    mask = _shape_mask(shape)
    out: list[tuple[tuple[int, int], ...]] = []
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for arr in (rot, np.fliplr(rot)):
            cells = _normalize_shape(map(tuple, np.argwhere(arr)))
            key = _shape_key(cells)
            if key not in seen:
                seen.add(key)
                out.append(cells)
    out.sort(key=_shape_key)
    return out


def _translate(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r0 + r, c0 + c) for r, c in shape}


def _adjacent(a: Iterable[tuple[int, int]], b: Iterable[tuple[int, int]]) -> bool:
    aset = set(a)
    bset = set(b)
    for r, c in aset:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in bset:
                return True
    return False


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _monochrome_components(grid: np.ndarray, ignored: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in ignored or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": _bbox(cells),
                    "first": min(cells),
                    "shape": _normalize_shape(cells),
                }
            )
    return comps


def _find_separator_row(grid: np.ndarray) -> tuple[int, int]:
    bg = _most_common_color(grid)
    h, w = grid.shape
    candidates = []
    for r in range(h):
        row = grid[r]
        if int(row[0]) == bg:
            continue
        if np.all(row == row[0]) and r > 0 and r < h - 1 and r == h - 1 - r:
            candidates.append(r)
    assert len(candidates) == 1, "expected one equal-height solid separator row"
    return candidates[0], bg


def _overlay_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    sep, bg = _find_separator_row(grid)
    top = np.array(grid[:sep], dtype=int)
    bottom = np.array(grid[sep + 1 :], dtype=int)
    assert top.shape == bottom.shape and top.size > 0
    assert not np.any((top != bg) & (bottom != bg)), "overlay collision violates puzzle assumption"
    merged = top.copy()
    take_bottom = (merged == bg) & (bottom != bg)
    merged[take_bottom] = bottom[take_bottom]
    return merged, top, bottom, bg


def _decode_instance(grid: np.ndarray) -> tuple[np.ndarray, dict, list[dict], list[dict], int]:
    merged, _top, _bottom, bg = _overlay_panels(grid)
    comps = _monochrome_components(merged, {bg})
    bar_candidates = [comp for comp in comps if comp["color"] == BAR]
    assert bar_candidates, "expected a blue central bar"
    bar = max(bar_candidates, key=lambda comp: len(comp["cells"]))
    sprites = [comp for comp in comps if comp is not bar]

    r0, c0, r1, c1 = bar["bbox"]
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in bar["cells"]:
        local[r - r0, c - c0] = True

    notches: list[dict] = []
    for comp in _components_from_mask(~local):
        edges = set()
        if any(rr == 0 for rr, _ in comp):
            edges.add("top")
        if any(rr == local.shape[0] - 1 for rr, _ in comp):
            edges.add("bottom")
        if any(cc == 0 for _, cc in comp):
            edges.add("left")
        if any(cc == local.shape[1] - 1 for _, cc in comp):
            edges.add("right")
        if len(edges) != 1:
            continue
        side = next(iter(edges))
        gcells = {(r0 + rr, c0 + cc) for rr, cc in comp}
        nr0, nc0, nr1, nc1 = _bbox(gcells)
        notches.append(
            {
                "side": side,
                "cells": gcells,
                "bbox": (nr0, nc0, nr1, nc1),
                "shape": _normalize_shape(gcells),
                "sort_key": nc0 if side in ("top", "bottom") else nr0,
            }
        )
    notches.sort(key=lambda item: item["sort_key"])
    sprites.sort(key=lambda item: item["first"])
    return merged, bar, notches, sprites, bg


def _root_assignment(notches: list[dict], sprites: list[dict]) -> dict[int, int]:
    candidates: list[list[int]] = []
    for notch in notches:
        matches = []
        for i, sprite in enumerate(sprites):
            if any(variant == notch["shape"] for variant in _shape_variants(sprite["shape"])):
                matches.append(i)
        candidates.append(matches)

    order = sorted(range(len(notches)), key=lambda idx: (len(candidates[idx]), idx))
    solution: dict[int, int] = {}
    used: set[int] = set()
    all_solutions: list[dict[int, int]] = []

    def dfs(pos: int) -> None:
        if len(all_solutions) > 1:
            return
        if pos == len(order):
            all_solutions.append(solution.copy())
            return
        notch_idx = order[pos]
        for sprite_idx in candidates[notch_idx]:
            if sprite_idx in used:
                continue
            used.add(sprite_idx)
            solution[notch_idx] = sprite_idx
            dfs(pos + 1)
            used.remove(sprite_idx)
            solution.pop(notch_idx, None)

    dfs(0)
    assert len(all_solutions) == 1, "root matching must be unique"
    return all_solutions[0]


def _chain_direction(side: str) -> tuple[int, int]:
    return {
        "top": (-1, 0),
        "bottom": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[side]


def _place_chain_variant(
    canvas: np.ndarray,
    head_cells: set[tuple[int, int]],
    head_bbox: tuple[int, int, int, int],
    side: str,
    variant: tuple[tuple[int, int], ...],
    bg: int,
) -> tuple[set[tuple[int, int]], tuple[int, int, int, int]] | None:
    h, w = _shape_dims(variant)
    hr0, hc0, hr1, hc1 = head_bbox
    if side in ("top", "bottom"):
        strip_width = hc1 - hc0 + 1
        if w != strip_width:
            return None
        top = hr0 - h if side == "top" else hr1 + 1
        left = hc0
    else:
        strip_width = hr1 - hr0 + 1
        if h != strip_width:
            return None
        top = hr0
        left = hc0 - w if side == "left" else hc1 + 1

    if top < 0 or left < 0 or top + h > canvas.shape[0] or left + w > canvas.shape[1]:
        return None

    cells = _translate(variant, top, left)
    if any(int(canvas[r, c]) != bg for r, c in cells):
        return None
    if not _adjacent(cells, head_cells):
        return None
    return cells, (top, left, top + h - 1, left + w - 1)


def _solve(grid: np.ndarray) -> np.ndarray:
    merged, bar, notches, sprites, bg = _decode_instance(grid)
    out = np.full(merged.shape, bg, dtype=int)
    for r, c in bar["cells"]:
        out[r, c] = BAR

    roots = _root_assignment(notches, sprites)
    used = set()
    chains: list[dict] = []
    for notch_idx, notch in enumerate(notches):
        sprite_idx = roots[notch_idx]
        used.add(sprite_idx)
        color = int(sprites[sprite_idx]["color"])
        for r, c in notch["cells"]:
            out[r, c] = color
        chains.append(
            {
                "side": notch["side"],
                "head_cells": set(notch["cells"]),
                "head_bbox": notch["bbox"],
            }
        )

    remaining = [i for i in range(len(sprites)) if i not in used]
    remaining.sort(key=lambda idx: sprites[idx]["first"])

    for chain in chains:
        while True:
            chosen = None
            for sprite_idx in remaining:
                sprite = sprites[sprite_idx]
                for variant in _shape_variants(sprite["shape"]):
                    placed = _place_chain_variant(
                        out,
                        chain["head_cells"],
                        chain["head_bbox"],
                        chain["side"],
                        variant,
                        bg,
                    )
                    if placed is not None:
                        cells, bbox = placed
                        chosen = (sprite_idx, cells, bbox, int(sprite["color"]))
                        break
                if chosen is not None:
                    break
            if chosen is None:
                break
            sprite_idx, cells, bbox, color = chosen
            for r, c in cells:
                out[r, c] = color
            chain["head_cells"] = cells
            chain["head_bbox"] = bbox
            remaining.remove(sprite_idx)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge the separated panels, fit matching sprites into the bar notches, and extend chains along each notch strip.

    Input structure:
    - A black background grid is split by one solid horizontal separator row into two equal-height panels.
    - The top panel contains a blue horizontal bar with 2-3 black notches cut from one boundary edge of the bar.
    - The bottom panel contains several small colored 4-connected sprites whose shapes may need rotation or reflection.

    Rule:
    - Delete the separator and overlay the bottom panel onto the top panel by local coordinates.
    - Treat each boundary-opening hole in the blue bar as a notch, and find the unique sprite whose shape matches that notch up to rotation/reflection.
    - Fill each notch with its matched root sprite color, then for that root repeatedly scan the remaining sprites in row-major input order and place the first sprite whose transformed bbox fits immediately outward along the notch direction while preserving the root strip width.
    - Stop a chain when no remaining sprite fits; unused sprites stay absent from the output.

    Key insight:
    - The divider hides the bar and the sprite pool from each other; only after overlay can the notch shapes identify the roots, and those roots then define the strip widths that control all later chain placements.
    """
    return _solve(np.asarray(grid, dtype=int))


def _polyomino_library() -> list[tuple[tuple[int, int], ...]]:
    by_area: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for area in range(2, 6):
        cur: set[tuple[tuple[int, int], ...]] = set()
        for shape in by_area[area - 1]:
            cells = set(shape)
            for r, c in list(cells):
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt in cells:
                        continue
                    grown = set(cells)
                    grown.add(nxt)
                    variants = _shape_variants(_normalize_shape(grown))
                    cur.add(min(variants, key=_shape_key))
        by_area[area] = cur
    out: list[tuple[tuple[int, int], ...]] = []
    for area in range(2, 6):
        out.extend(sorted(by_area[area], key=_shape_key))
    return out


ALL_SHAPES = _polyomino_library()
COMPACT_SHAPES = [
    shape
    for shape in ALL_SHAPES
    if any(_shape_dims(variant)[0] <= 3 and _shape_dims(variant)[1] <= 3 for variant in _shape_variants(shape))
]
ROOT_VARIANTS = {
    shape: [
        variant
        for variant in _shape_variants(shape)
        if _shape_dims(variant)[0] <= 2 and _shape_dims(variant)[1] <= 3
    ]
    for shape in COMPACT_SHAPES
}
ROOT_SHAPES = [shape for shape, variants in ROOT_VARIANTS.items() if variants]
INPUT_VARIANTS = {
    shape: [
        variant
        for variant in _shape_variants(shape)
        if _shape_dims(variant)[0] <= 3 and _shape_dims(variant)[1] <= 3
    ]
    for shape in COMPACT_SHAPES
}


def _intervals_for_widths(rng: random.Random, start: int, bar_w: int, widths: list[int]) -> list[int] | None:
    needed = sum(widths) + (len(widths) - 1)
    usable = bar_w - 2
    if needed > usable:
        return None
    slack = usable - needed
    extras = [0] * (len(widths) + 1)
    for _ in range(slack):
        extras[rng.randrange(len(extras))] += 1
    cols = []
    cur = start + 1 + extras[0]
    for i, width in enumerate(widths):
        cols.append(cur)
        cur += width
        if i < len(widths) - 1:
            cur += 1 + extras[i + 1]
    return cols


def _root_cells(side: str, bar_bbox: tuple[int, int, int, int], variant: tuple[tuple[int, int], ...], c0: int) -> set[tuple[int, int]]:
    br0, _bc0, br1, _bc1 = bar_bbox
    if side == "top":
        return {(br0 + r, c0 + c) for r, c in variant}
    h, _w = _shape_dims(variant)
    return {(br1 - r, c0 + c) for r, c in variant if 0 <= h}


def _step_options(
    canvas_shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    head_cells: set[tuple[int, int]],
    head_bbox: tuple[int, int, int, int],
    side: str,
    forbidden_keys: set[tuple[bytes, tuple[int, int]]],
) -> list[dict]:
    dummy = np.full(canvas_shape, BG, dtype=int)
    for r, c in occupied:
        dummy[r, c] = 9
    opts: list[dict] = []
    for shape in COMPACT_SHAPES:
        key = _shape_key(shape)
        if key in forbidden_keys:
            continue
        for variant in _shape_variants(shape):
            placed = _place_chain_variant(dummy, head_cells, head_bbox, side, variant, BG)
            if placed is None:
                continue
            cells, bbox = placed
            opts.append(
                {
                    "shape": shape,
                    "key": key,
                    "variant": variant,
                    "cells": cells,
                    "bbox": bbox,
                }
            )
    return opts


def _assign_extra_counts(rng: random.Random, nroots: int, total: int, force_double: bool) -> list[int] | None:
    counts = [0] * nroots
    if force_double:
        if total < 2:
            return None
        idx = rng.randrange(nroots)
        counts[idx] = 2
        total -= 2
    while total > 0:
        choices = [i for i, val in enumerate(counts) if val < 2]
        if not choices:
            return None
        idx = rng.choice(choices)
        counts[idx] += 1
        total -= 1
    return counts


def _scatter_sprites(
    rng: random.Random,
    panel_h: int,
    width: int,
    bar_rows: tuple[int, int],
    sprites: list[dict],
) -> list[dict] | None:
    blocked = np.zeros((panel_h, width), dtype=bool)
    blocked[bar_rows[0] : bar_rows[1] + 1, :] = True
    occupied = np.zeros((panel_h, width), dtype=bool)
    prev_first = (-1, -1)
    placed: list[dict] = []
    for sprite in sprites:
        shape = rng.choice(INPUT_VARIANTS[sprite["shape"]])
        h, w = _shape_dims(shape)
        candidates = []
        for r0 in range(panel_h - h + 1):
            for c0 in range(width - w + 1):
                cells = _translate(shape, r0, c0)
                first = min(cells)
                if first <= prev_first:
                    continue
                if any(blocked[r, c] for r, c in cells):
                    continue
                conflict = False
                for r, c in cells:
                    rr0 = max(0, r - 1)
                    rr1 = min(panel_h, r + 2)
                    cc0 = max(0, c - 1)
                    cc1 = min(width, c + 2)
                    if np.any(occupied[rr0:rr1, cc0:cc1]):
                        conflict = True
                        break
                if not conflict:
                    candidates.append((r0, c0, cells, first, shape))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[3])
        take = candidates[: max(5, min(20, len(candidates)))]
        r0, c0, cells, first, shape = rng.choice(take)
        for r, c in cells:
            occupied[r, c] = True
        prev_first = first
        placed.append(
            {
                **sprite,
                "input_shape": shape,
                "input_cells": cells,
                "input_first": first,
            }
        )
    return placed


def _validate_generated(inp: np.ndarray, expected: np.ndarray, root_count: int) -> bool:
    try:
        got = main(inp)
        if not np.array_equal(got, expected):
            return False
        merged, bar, notches, sprites, _bg = _decode_instance(inp)
        if len(notches) != root_count or len(notches) < 2 or len(notches) > 3:
            return False
        roots = _root_assignment(notches, sprites)
        if len(roots) != len(notches):
            return False
        counts = Counter(int(v) for v in got.ravel())
        if counts[BAR] <= 0 or len([c for c in counts if counts[c] > 0 and c != BG]) < 3:
            return False
        if merged.shape != expected.shape:
            return False
        return True
    except Exception:
        return False


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    panel_h = rng.randint(8, 11)
    width = rng.randint(14, 18)
    bar_h = rng.randint(3, 4)
    if panel_h - bar_h < 4:
        return None
    top_space = rng.randint(2, panel_h - bar_h - 2)
    bar_r0 = top_space
    bar_r1 = bar_r0 + bar_h - 1
    bottom_space = panel_h - bar_r1 - 1

    nroots = rng.choice([2, 2, 3])
    bar_w_max = min(13, width - 2)
    if bar_w_max < 10:
        return None
    bar_w = rng.randint(10, bar_w_max)
    bar_c0 = rng.randint(1, width - bar_w - 1)
    bar_bbox = (bar_r0, bar_c0, bar_r1, bar_c0 + bar_w - 1)

    root_specs = []
    chosen_root_keys = set()
    for _ in range(80):
        root_specs.clear()
        chosen_root_keys.clear()
        widths = []
        shapes = rng.sample(ROOT_SHAPES, nroots)
        ok = True
        for shape in shapes:
            variant = rng.choice(ROOT_VARIANTS[shape])
            key = _shape_key(shape)
            if key in chosen_root_keys:
                ok = False
                break
            chosen_root_keys.add(key)
            h, w = _shape_dims(variant)
            widths.append(w)
            root_specs.append(
                {
                    "shape": shape,
                    "variant": variant,
                    "height": h,
                    "width": w,
                    "side": rng.choice(["top", "bottom"]),
                    "key": key,
                }
            )
        if not ok:
            continue
        starts = _intervals_for_widths(rng, bar_c0, bar_w, widths)
        if starts is None:
            continue
        for spec, c0 in zip(root_specs, starts):
            spec["c0"] = c0
        root_specs.sort(key=lambda item: item["c0"])
        break
    else:
        return None

    target_total = rng.randint(4, 7)
    extra_total = rng.randint(1, min(3, target_total - nroots))
    decoy_low = max(0, 4 - nroots - extra_total)
    decoy_high = min(2, 7 - nroots - extra_total)
    if decoy_low > decoy_high:
        return None
    decoy_count = rng.randint(decoy_low, decoy_high)
    total_sprites = nroots + extra_total + decoy_count
    if total_sprites < 4 or total_sprites > 7:
        return None
    want_order_branch = extra_total >= 2 and rng.random() < 0.45
    extra_counts = _assign_extra_counts(rng, nroots, extra_total, want_order_branch)
    if extra_counts is None:
        return None

    output_sprites: list[dict] = []
    occupied = {
        (r, c)
        for r in range(bar_r0, bar_r1 + 1)
        for c in range(bar_c0, bar_c0 + bar_w)
    }
    used_keys = set(chosen_root_keys)

    for spec in root_specs:
        cells = _root_cells(spec["side"], bar_bbox, spec["variant"], spec["c0"])
        spec["output_cells"] = cells
        spec["output_bbox"] = _bbox(cells)
        output_sprites.append(
            {
                "shape": spec["shape"],
                "role": "root",
                "output_cells": cells,
                "side": spec["side"],
            }
        )

    order_branch_done = False
    nonroot_used: list[dict] = []
    for idx, spec in enumerate(root_specs):
        head_cells = set(spec["output_cells"])
        head_bbox = spec["output_bbox"]
        side = spec["side"]
        count = extra_counts[idx]
        if count == 0:
            continue

        if want_order_branch and count >= 2 and not order_branch_done:
            opts0 = _step_options((panel_h, width), occupied, head_cells, head_bbox, side, used_keys)
            rng.shuffle(opts0)
            pair = None
            for opt_a in opts0:
                occ1 = occupied | opt_a["cells"]
                used1 = used_keys | {opt_a["key"]}
                opts1 = _step_options((panel_h, width), occ1, opt_a["cells"], opt_a["bbox"], side, used1)
                other_keys0 = {opt["key"] for opt in opts0 if opt["key"] != opt_a["key"]}
                opts1 = [opt for opt in opts1 if opt["key"] in other_keys0]
                if opts1:
                    opt_b = rng.choice(opts1)
                    pair = (opt_a, opt_b)
                    break
            if pair is not None:
                for opt in pair:
                    nonroot_used.append(
                        {
                            "shape": opt["shape"],
                            "role": "chain",
                            "output_cells": opt["cells"],
                            "side": side,
                            "forced_early": opt is pair[0],
                        }
                    )
                    occupied |= opt["cells"]
                    used_keys.add(opt["key"])
                    head_cells = set(opt["cells"])
                    head_bbox = opt["bbox"]
                count -= 2
                order_branch_done = True

        for _ in range(count):
            opts = _step_options((panel_h, width), occupied, head_cells, head_bbox, side, used_keys)
            if not opts:
                return None
            opt = rng.choice(opts)
            nonroot_used.append(
                {
                    "shape": opt["shape"],
                    "role": "chain",
                    "output_cells": opt["cells"],
                    "side": side,
                    "forced_early": False,
                }
            )
            occupied |= opt["cells"]
            used_keys.add(opt["key"])
            head_cells = set(opt["cells"])
            head_bbox = opt["bbox"]

    decoys: list[dict] = []
    available_decoys = [shape for shape in COMPACT_SHAPES if _shape_key(shape) not in used_keys]
    rng.shuffle(available_decoys)
    if len(available_decoys) < decoy_count:
        return None
    for shape in available_decoys[:decoy_count]:
        used_keys.add(_shape_key(shape))
        decoys.append({"shape": shape, "role": "decoy", "forced_early": False})

    all_sprites = output_sprites + nonroot_used + decoys
    if len(all_sprites) != total_sprites:
        return None

    colors = list(rng.sample(SPRITE_COLORS, len(all_sprites)))
    for sprite, color in zip(all_sprites, colors):
        sprite["color"] = color

    roots = [sprite for sprite in all_sprites if sprite["role"] == "root"]
    nonroots = [sprite for sprite in all_sprites if sprite["role"] != "root"]
    forced = [sprite for sprite in nonroots if sprite.get("forced_early")]
    others = [sprite for sprite in nonroots if not sprite.get("forced_early")]
    rng.shuffle(roots)
    rng.shuffle(others)
    scatter_order = roots + forced + others
    scattered = _scatter_sprites(rng, panel_h, width, (bar_r0, bar_r1), scatter_order)
    if scattered is None:
        return None

    top = np.zeros((panel_h, width), dtype=int)
    top[bar_r0 : bar_r1 + 1, bar_c0 : bar_c0 + bar_w] = BAR
    for sprite in scattered:
        if sprite["role"] == "root":
            for r, c in sprite["output_cells"]:
                top[r, c] = BG

    bottom = np.zeros((panel_h, width), dtype=int)
    for sprite in scattered:
        for r, c in sprite["input_cells"]:
            if top[r, c] != BG:
                return None
            bottom[r, c] = sprite["color"]

    expected = np.full((panel_h, width), BG, dtype=int)
    expected[bar_r0 : bar_r1 + 1, bar_c0 : bar_c0 + bar_w] = BAR
    for sprite in scattered:
        if sprite["role"] != "decoy":
            for r, c in sprite["output_cells"]:
                expected[r, c] = sprite["color"]

    inp = np.full((2 * panel_h + 1, width), BG, dtype=int)
    inp[:panel_h] = top
    inp[panel_h, :] = SEPARATOR
    inp[panel_h + 1 :] = bottom

    if not _validate_generated(inp, expected, nroots):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        built = _gen_attempt(rng)
        if built is not None:
            return built[0]
    raise AssertionError("failed to generate a valid puzzle instance")
