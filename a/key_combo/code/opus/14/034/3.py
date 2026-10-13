from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PALETTE = [1, 2, 3, 4, 6, 7, 8]
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _paint_mask(out: np.ndarray, mask: np.ndarray, top: tuple[int, int], color: int) -> None:
    r0, c0 = top
    rows, cols = np.nonzero(mask)
    out[rows + r0, cols + c0] = int(color)


def _translated_cells(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _cells_with_padding(cells: set[tuple[int, int]], pad: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                out.add((r + dr, c + dc))
    return out


def _min_manhattan_to_set(point: tuple[int, int], cells: set[tuple[int, int]]) -> int:
    r, c = point
    return min(abs(r - rr) + abs(c - cc) for rr, cc in cells)


def _mirror_col(c: int, axis: int) -> int:
    return 2 * axis - c


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    rects = []
    for comp in _mask_components(grid == MAROON):
        r0, c0, r1, c1 = _bbox(comp)
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(patch == MAROON):
            continue
        rects.append({"bbox": (r0, c0, r1, c1), "cells": set(comp)})
    return rects


def _adjacent_sprite_color(grid: np.ndarray, rect: dict) -> int:
    counts: Counter[int] = Counter()
    for r, c in rect["cells"]:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                color = int(grid[nr, nc])
                if color not in (BLACK, GREY, MAROON):
                    counts[color] += 1
    assert counts, "each occluder must touch one visible sprite color"
    color, _ = counts.most_common(1)[0]
    assert len(counts) == 1, "each occluder should be adjacent to exactly one sprite color"
    return int(color)


def _reconstruct_sprite(grid: np.ndarray, rect: dict, color: int) -> dict:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
    assert coords, "expected visible pixels for each sprite color"
    cr0, cc0, cr1, cc1 = _bbox(coords)
    rr0, rc0, rr1, rc1 = rect["bbox"]
    r0, r1 = min(cr0, rr0), max(cr1, rr1)
    c0, c1 = min(cc0, rc0), max(cc1, rc1)
    assert (c1 - c0) % 2 == 0, "sprite width must be odd to expose an integer symmetry axis"
    axis = (c0 + c1) // 2
    h, w = r1 - r0 + 1, c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)

    mask[:, axis - c0] = True
    for r, c in coords:
        if not (r0 <= r <= r1 and c0 <= c <= c1):
            continue
        mask[r - r0, c - c0] = True
        mc = _mirror_col(c, axis)
        assert c0 <= mc <= c1
        mask[r - r0, mc - c0] = True

    side = None
    if rc0 == c0 and rc1 >= axis:
        side = "left"
    if rc1 == c1 and rc0 <= axis:
        side = "right" if side is None else side
    assert side in {"left", "right"}

    candidates = []
    for r in range(rr0, rr1 + 1):
        for c in range(rc0, rc1 + 1):
            if c == axis:
                continue
            if side == "left" and c > axis:
                continue
            if side == "right" and c < axis:
                continue
            if mask[r - r0, c - c0]:
                continue
            mc = _mirror_col(c, axis)
            if not (c0 <= mc <= c1):
                continue
            if mask[r - r0, mc - c0]:
                continue
            ok = True
            for nr, nc in _neighbors4(r, c):
                if not (r0 <= nr <= r1 and c0 <= nc <= c1 and mask[nr - r0, nc - c0]):
                    ok = False
                    break
            if ok:
                candidates.append((r, c))
    assert len(candidates) == 1, "hidden source anchor must be uniquely recoverable"

    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "axis": int(axis),
        "mask": mask,
        "source_anchor": candidates[0],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each maroon occluder, mirror-complete the symmetric sprite it hid, infer the hidden source anchor, and copy the sprite to the matching gray target anchor.

    Input structure:
    - A black background contains 2-3 differently colored monochrome sprites that are left-right symmetric once completed.
    - Each sprite is partially covered by one solid maroon rectangle that hides one whole side of the sprite and some axis pixels.
    - The only visible gray pixels are isolated target anchors placed far from every sprite and occluder.

    Rule:
    - Detect every solid maroon rectangle and treat it as removable occlusion.
    - For the adjacent sprite color, use the rectangle span plus the visible same-color fragment to recover the full bbox and vertical symmetry axis; restore the sprite by filling the full axis column and mirroring visible pixels across that axis.
    - Inside the former occluder, identify the unique mirrored background hole fully enclosed by sprite pixels as the hidden source anchor.
    - Pair each hidden source anchor with its uniquely nearest visible gray target anchor by Manhattan distance, translate the completed sprite by that offset, keep the restored original sprite, and erase all gray anchors.

    Key insight:
    - The copy offset is controlled by a source anchor that is not visible in the input grid; it becomes uniquely inferable only after the occluder is removed and the sprite is mirror-completed, because exactly one enclosed hole remains in the covered half.
    """
    g = np.asarray(grid, dtype=int)
    rects = _find_rectangles(g)
    targets = [tuple(map(int, rc)) for rc in np.argwhere(g == GREY)]
    assert len(rects) == len(targets) and 2 <= len(rects) <= 3

    sprites = []
    used_colors = set()
    for rect in rects:
        color = _adjacent_sprite_color(g, rect)
        assert color not in used_colors, "sprite colors should be distinct"
        used_colors.add(color)
        sprites.append(_reconstruct_sprite(g, rect, color))

    assignment = []
    for sprite in sprites:
        distances = [abs(sprite["source_anchor"][0] - tr) + abs(sprite["source_anchor"][1] - tc) for tr, tc in targets]
        best = min(distances)
        assert distances.count(best) == 1, "nearest target must be unique for each source anchor"
        assignment.append(distances.index(best))
    assert len(set(assignment)) == len(assignment), "nearest-target pairing must be one-to-one"

    out = np.zeros_like(g)
    for sprite in sprites:
        _paint_mask(out, sprite["mask"], sprite["bbox"][:2], sprite["color"])
    for sprite, idx in zip(sprites, assignment):
        tr, tc = targets[idx]
        sr, sc = sprite["source_anchor"]
        dr, dc = tr - sr, tc - sc
        r0, c0, _r1, _c1 = sprite["bbox"]
        rows, cols = np.nonzero(sprite["mask"])
        rr = rows + r0 + dr
        cc = cols + c0 + dc
        assert rr.min() >= 0 and rr.max() < g.shape[0]
        assert cc.min() >= 0 and cc.max() < g.shape[1]
        out[rr, cc] = sprite["color"]
    return out


def _generate_template() -> dict | None:
    for _ in range(200):
        h = random.randint(5, 7)
        half_span = random.choice([2, 3])
        w = 2 * half_span + 1
        band_r0 = random.randint(1, h - 4)
        band_r1 = random.randint(band_r0 + 2, h - 2)
        extents = [0] * h
        for r in range(band_r0, band_r1 + 1):
            extents[r] = random.randint(1, half_span)
        if max(extents) < half_span:
            extents[random.randint(band_r0, band_r1)] = half_span
        hole_rows = [r for r in range(band_r0 + 1, band_r1) if extents[r] >= 2]
        if not hole_rows:
            continue
        hole_row = random.choice(hole_rows)
        axis = half_span
        mask = np.zeros((h, w), dtype=bool)
        mask[:, axis] = True
        for r, extent in enumerate(extents):
            for d in range(1, extent + 1):
                mask[r, axis - d] = True
                mask[r, axis + d] = True
        mask[hole_row, axis - 1] = False
        mask[hole_row, axis + 1] = False
        area = int(mask.sum())
        if not 12 <= area <= 20:
            continue
        if len(_mask_components(mask)) != 1:
            continue
        hole_candidates = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                ok = True
                for nr, nc in _neighbors4(r, c):
                    if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
                        ok = False
                        break
                if ok:
                    hole_candidates.append((r, c))
        if set(hole_candidates) != {(hole_row, axis - 1), (hole_row, axis + 1)}:
            continue
        side = random.choice(["left", "right"])
        rect = (band_r0, 0, band_r1, axis) if side == "left" else (band_r0, axis, band_r1, w - 1)
        anchor = (hole_row, axis - 1 if side == "left" else axis + 1)
        return {
            "mask": mask,
            "rect": rect,
            "anchor": anchor,
            "side": side,
            "shape": (h, w),
        }
    return None


def _absolute_cells(mask: np.ndarray, top: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top
    rows, cols = np.nonzero(mask)
    return {(int(r0 + r), int(c0 + c)) for r, c in zip(rows, cols)}


def _rect_cells(rect: tuple[int, int, int, int], top: tuple[int, int]) -> set[tuple[int, int]]:
    rr0, rc0, rr1, rc1 = rect
    tr, tc = top
    return {(tr + r, tc + c) for r in range(rr0, rr1 + 1) for c in range(rc0, rc1 + 1)}


def _can_place(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> bool:
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
    if cells & occupied:
        return False
    if pad > 0 and _cells_with_padding(cells, pad) & occupied:
        return False
    return True


def _place_originals(h: int, w: int, templates: list[dict], mode: str) -> list[dict] | None:
    source_limit = w // 2 - 2 if mode == "horizontal" else h // 2 - 2
    occupied: set[tuple[int, int]] = set()
    placed = []
    for tpl in sorted(templates, key=lambda item: item["shape"][0] * item["shape"][1], reverse=True):
        th, tw = tpl["shape"]
        ok = False
        for _ in range(300):
            if mode == "horizontal":
                r0 = random.randint(1, h - th - 1)
                c0 = random.randint(1, source_limit - tw)
            else:
                r0 = random.randint(1, source_limit - th)
                c0 = random.randint(1, w - tw - 1)
            sprite_cells = _absolute_cells(tpl["mask"], (r0, c0))
            rect_cells = _rect_cells(tpl["rect"], (r0, c0))
            all_cells = sprite_cells | rect_cells
            if _can_place(all_cells, occupied, h, w, pad=2):
                tpl = dict(tpl)
                tpl["top"] = (r0, c0)
                tpl["sprite_cells"] = sprite_cells
                tpl["rect_cells"] = rect_cells
                tpl["source_anchor_abs"] = (r0 + tpl["anchor"][0], c0 + tpl["anchor"][1])
                placed.append(tpl)
                occupied |= all_cells
                ok = True
                break
        if not ok:
            return None
    return placed


def _place_copies(h: int, w: int, sprites: list[dict], mode: str) -> list[dict] | None:
    originals = set().union(*(sprite["sprite_cells"] | sprite["rect_cells"] for sprite in sprites))
    output_occupied = set().union(*(sprite["sprite_cells"] for sprite in sprites))
    targets: set[tuple[int, int]] = set()
    ordering = sorted(sprites, key=lambda item: item["source_anchor_abs"][0 if mode == "horizontal" else 1])
    for sprite in ordering:
        th, tw = sprite["shape"]
        ar, ac = sprite["anchor"]
        sr, sc = sprite["source_anchor_abs"]
        row_offsets = [-2, -1, 0, 1, 2]
        placed = False
        for _ in range(400):
            if mode == "horizontal":
                r0 = sr - ar + random.choice(row_offsets)
                c0 = random.randint(w // 2 + 1, w - tw - 1)
            else:
                r0 = random.randint(h // 2 + 1, h - th - 1)
                c0 = sc - ac + random.choice(row_offsets)
            copy_cells = _absolute_cells(sprite["mask"], (r0, c0))
            target = (r0 + ar, c0 + ac)
            if not _can_place(copy_cells, output_occupied, h, w, pad=1):
                continue
            if not (1 <= target[0] < h - 1 and 1 <= target[1] < w - 1):
                continue
            if target in copy_cells or target in targets:
                continue
            if _min_manhattan_to_set(target, originals) < 5:
                continue
            if any(abs(target[0] - tr) + abs(target[1] - tc) < 4 for tr, tc in targets):
                continue
            sprite["copy_top"] = (r0, c0)
            sprite["copy_cells"] = copy_cells
            sprite["target_anchor_abs"] = target
            targets.add(target)
            output_occupied |= copy_cells
            placed = True
            break
        if not placed:
            return None
    return sprites


def _build_input(h: int, w: int, sprites: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for sprite in sprites:
        _paint_mask(grid, sprite["mask"], sprite["top"], sprite["color"])
        rr0, rc0, rr1, rc1 = sprite["rect"]
        tr, tc = sprite["top"]
        grid[tr + rr0 : tr + rr1 + 1, tc + rc0 : tc + rc1 + 1] = MAROON
        trr, tcc = sprite["target_anchor_abs"]
        grid[trr, tcc] = GREY
    return grid


def _build_output(h: int, w: int, sprites: list[dict]) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for sprite in sprites:
        _paint_mask(out, sprite["mask"], sprite["top"], sprite["color"])
        _paint_mask(out, sprite["mask"], sprite["copy_top"], sprite["color"])
    return out


def _validate_semantics(inp: np.ndarray, out: np.ndarray, sprites: list[dict]) -> bool:
    if inp.ndim != 2 or out.ndim != 2 or max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if np.count_nonzero(inp == GREY) != len(sprites):
        return False
    if np.any(out == GREY) or np.any(out == MAROON):
        return False
    originals = set().union(*(sprite["sprite_cells"] for sprite in sprites))
    rects = set().union(*(sprite["rect_cells"] for sprite in sprites))
    copies = set().union(*(sprite["copy_cells"] for sprite in sprites))
    if originals & copies or rects & copies:
        return False
    for sprite in sprites:
        sr = sprite["source_anchor_abs"]
        tr = sprite["target_anchor_abs"]
        if sr == tr:
            return False
        if sr == sprite["top"]:
            return False
        if _min_manhattan_to_set(tr, originals | rects) < 5:
            return False
        rr0, rc0, rr1, rc1 = sprite["rect"]
        tr0, tc0 = sprite["top"]
        if not (tr0 + rr0 <= sr[0] <= tr0 + rr1 and tc0 + rc0 <= sr[1] <= tc0 + rc1):
            return False
        rect_center = ((tr0 + rr0 + tr0 + rr1) // 2, (tc0 + rc0 + tc0 + rc1) // 2)
        bbox_center = (
            tr0 + sprite["shape"][0] // 2,
            tc0 + sprite["shape"][1] // 2,
        )
        if sr == rect_center or sr == bbox_center:
            return False
    sources = [sprite["source_anchor_abs"] for sprite in sprites]
    targets = [sprite["target_anchor_abs"] for sprite in sprites]
    mapping = []
    for src, sprite in zip(sources, sprites):
        distances = [abs(src[0] - t[0]) + abs(src[1] - t[1]) for t in targets]
        best = min(distances)
        if distances.count(best) != 1:
            return False
        idx = distances.index(best)
        mapping.append(idx)
        if targets[idx] != sprite["target_anchor_abs"]:
            return False
    if len(set(mapping)) != len(mapping):
        return False
    if np.array_equal(inp, out):
        return False
    return np.array_equal(main(inp), out)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(24, 30)
        w = random.randint(24, 30)
        count = random.choice([2, 3])
        mode = random.choice(["horizontal", "vertical"])
        templates = []
        made = True
        for color in random.sample(PALETTE, count):
            tpl = _generate_template()
            if tpl is None:
                made = False
                break
            tpl["color"] = int(color)
            templates.append(tpl)
        if not made:
            continue
        originals = _place_originals(h, w, templates, mode)
        if originals is None:
            continue
        sprites = _place_copies(h, w, originals, mode)
        if sprites is None:
            continue
        inp = _build_input(h, w, sprites)
        out = _build_output(h, w, sprites)
        if _validate_semantics(inp, out, sprites):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
