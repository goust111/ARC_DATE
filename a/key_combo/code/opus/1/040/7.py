from __future__ import annotations

from collections import deque
import random

import numpy as np


BBox = tuple[int, int, int, int]
Direction = tuple[int, int]

BG = 0
TEMPLATE = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS: tuple[Direction, ...] = ((0, 1), (0, -1), (1, 0), (-1, 0))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _largest_component(mask: np.ndarray) -> np.ndarray:
    comps = _components_mask(mask)
    assert comps, "expected at least one component"
    sizes = [int(len(comp)) for comp in comps]
    best = int(np.argmax(sizes))
    largest = np.zeros_like(mask, dtype=bool)
    largest[comps[best][:, 0], comps[best][:, 1]] = True
    return largest


def _mid_positions(lo: int, hi: int) -> set[int]:
    size = hi - lo + 1
    if size % 2 == 1:
        return {lo + size // 2}
    mid = lo + size // 2
    return {mid - 1, mid}


def _is_full(mask: np.ndarray, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    return bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))


def _shift_box(box: BBox, direction: Direction, length: int) -> BBox:
    r0, c0, r1, c1 = box
    dr, dc = direction
    return r0 + dr * length, c0 + dc * length, r1 + dr * length, c1 + dc * length


def _box_in_bounds(box: BBox, shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = box
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _slider_cells(body: BBox, direction: Direction, length: int) -> np.ndarray:
    r0, c0, r1, c1 = body
    dr, dc = direction
    rr, cc = np.mgrid[r0 : r1 + 1, c0 : c1 + 1]
    body_cells = np.column_stack((rr.ravel(), cc.ravel()))
    if dc > 0:
        tail_row = (r0 + r1) // 2
        tail = np.column_stack((np.full(length, tail_row), np.arange(c1 + 1, c1 + length + 1)))
    elif dc < 0:
        tail_row = (r0 + r1) // 2
        tail = np.column_stack((np.full(length, tail_row), np.arange(c0 - length, c0)))
    elif dr > 0:
        tail_col = (c0 + c1) // 2
        tail = np.column_stack((np.arange(r1 + 1, r1 + length + 1), np.full(length, tail_col)))
    else:
        tail_col = (c0 + c1) // 2
        tail = np.column_stack((np.arange(r0 - length, r0), np.full(length, tail_col)))
    return np.vstack((body_cells, tail)).astype(int, copy=False)


def _detect_slider(mask: np.ndarray) -> tuple[BBox, Direction, int] | None:
    coords = np.argwhere(mask)
    assert coords.size, "expected a non-empty mask"
    r0, c0, r1, c1 = _bbox_coords(coords)
    candidates: list[tuple[BBox, Direction, int]] = []

    def add_if_slider(body: BBox, direction: Direction, length: int, outside: np.ndarray) -> None:
        br0, bc0, br1, bc1 = body
        if length < 2:
            return
        if br1 - br0 + 1 < 3 or bc1 - bc0 + 1 < 3:
            return
        if outside.shape[0] != length or not _is_full(mask, body):
            return
        rows = set(map(int, outside[:, 0]))
        cols = set(map(int, outside[:, 1]))
        dr, dc = direction
        if dc > 0:
            if len(rows) != 1 or cols != set(range(bc1 + 1, c1 + 1)):
                return
            if next(iter(rows)) not in _mid_positions(br0, br1):
                return
        elif dc < 0:
            if len(rows) != 1 or cols != set(range(c0, bc0)):
                return
            if next(iter(rows)) not in _mid_positions(br0, br1):
                return
        elif dr > 0:
            if len(cols) != 1 or rows != set(range(br1 + 1, r1 + 1)):
                return
            if next(iter(cols)) not in _mid_positions(bc0, bc1):
                return
        else:
            if len(cols) != 1 or rows != set(range(r0, br0)):
                return
            if next(iter(cols)) not in _mid_positions(bc0, bc1):
                return
        candidates.append((body, direction, length))

    for body_c1 in range(c0 + 2, c1):
        body = (r0, c0, r1, body_c1)
        add_if_slider(body, (0, 1), c1 - body_c1, coords[coords[:, 1] > body_c1])
    for body_c0 in range(c0 + 1, c1 - 1):
        body = (r0, body_c0, r1, c1)
        add_if_slider(body, (0, -1), body_c0 - c0, coords[coords[:, 1] < body_c0])
    for body_r1 in range(r0 + 2, r1):
        body = (r0, c0, body_r1, c1)
        add_if_slider(body, (1, 0), r1 - body_r1, coords[coords[:, 0] > body_r1])
    for body_r0 in range(r0 + 1, r1 - 1):
        body = (body_r0, c0, r1, c1)
        add_if_slider(body, (-1, 0), body_r0 - r0, coords[coords[:, 0] < body_r0])

    assert len(candidates) <= 1, "slider decomposition must be unique"
    return candidates[0] if candidates else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each tailed palette rectangle into the gray template and color only the covered mask cells.

    Input structure:
    - The background is black.
    - The largest gray 4-connected component is an irregular template mask with holes and concavities.
    - Every other foreground component is monochrome and consists of a solid rectangle plus a straight one-cell-wide tail attached at the midpoint of one side.

    Rule:
    - Keep only the gray template and erase every colored input object from its original position.
    - For each colored component, recover its rectangular body, tail direction, and tail length from the body-plus-tail geometry.
    - Translate the rectangle body by exactly the tail length in the tail direction, then recolor only the translated cells that land on gray template pixels.
    - If multiple translated rectangles cover the same template cell, the object with the longer tail is drawn last and stays visible on top.

    Key insight:
    - The tail simultaneously encodes a displacement vector and an overlap priority, while the gray template clips every moved rectangle down to the intended stencil pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    template_mask = _largest_component(g == TEMPLATE)

    sliders: list[dict[str, object]] = []
    for color in sorted(set(map(int, np.unique(g))) - {bg, TEMPLATE}):
        for comp in _components_mask(g == color):
            comp_mask = np.zeros_like(g, dtype=bool)
            comp_mask[comp[:, 0], comp[:, 1]] = True
            slider = _detect_slider(comp_mask)
            assert slider is not None, "every non-gray object must be a tail slider"
            body, direction, length = slider
            sliders.append(
                {
                    "color": int(color),
                    "body": body,
                    "direction": direction,
                    "length": int(length),
                }
            )

    sliders.sort(key=lambda spec: int(spec["length"]))

    out = np.full_like(g, bg)
    out[template_mask] = TEMPLATE
    for spec in sliders:
        dest = _shift_box(spec["body"], spec["direction"], int(spec["length"]))
        r0, c0, r1, c1 = dest
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        cover = template_mask[r0 : r1 + 1, c0 : c1 + 1]
        patch[cover] = int(spec["color"])
    return out


def _count_holes(mask: np.ndarray) -> int:
    holes = 0
    for comp in _components_mask(~mask):
        rows = comp[:, 0]
        cols = comp[:, 1]
        if rows.min() == 0 or rows.max() == mask.shape[0] - 1 or cols.min() == 0 or cols.max() == mask.shape[1] - 1:
            continue
        holes += 1
    return holes


def _template_ok(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    if (r0, c0, r1, c1) != (0, 0, mask.shape[0] - 1, mask.shape[1] - 1):
        return False
    if len(_components_mask(mask)) != 1:
        return False
    if not np.any(~mask[0, :]) and not np.any(~mask[-1, :]) and not np.any(~mask[:, 0]) and not np.any(~mask[:, -1]):
        return False
    if _count_holes(mask) < 1:
        return False
    area = int(mask.sum())
    if not 30 <= area <= 50:
        return False
    return True


def _boundary_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                cells.append((r, c))
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if not mask[r + dr, c + dc]:
                    cells.append((r, c))
                    break
    return cells


def _carve_notch(mask: np.ndarray, side: str) -> np.ndarray | None:
    h, w = mask.shape
    cand = mask.copy()
    if side in ("top", "bottom"):
        width = random.randint(1, min(3, w - 4))
        depth = random.randint(1, min(2, h - 4))
        c0 = random.randint(1, w - width - 2)
        if side == "top":
            cand[0:depth, c0 : c0 + width] = False
        else:
            cand[h - depth : h, c0 : c0 + width] = False
    else:
        height = random.randint(1, min(3, h - 4))
        depth = random.randint(1, min(2, w - 4))
        r0 = random.randint(1, h - height - 2)
        if side == "left":
            cand[r0 : r0 + height, 0:depth] = False
        else:
            cand[r0 : r0 + height, w - depth : w] = False
    return cand


def _make_template_local() -> np.ndarray:
    templates = (
        (
            "1110111",
            "1111111",
            "1011101",
            "1110111",
            "1101111",
            "1111101",
            "0111111",
        ),
        (
            "1111011",
            "1111111",
            "1101111",
            "1110111",
            "1011111",
            "1111101",
            "1111111",
            "0111111",
        ),
        (
            "11110111",
            "11111111",
            "11011101",
            "11101111",
            "10111111",
            "11111011",
            "11111110",
        ),
    )
    rows = random.choice(templates)
    mask = np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)
    assert _template_ok(mask), "template library must contain only valid masks"
    return mask


def _place_template(shape: tuple[int, int], local: np.ndarray) -> tuple[np.ndarray, BBox]:
    h, w = shape
    th, tw = local.shape
    center_r = (h - th) // 2
    center_c = (w - tw) // 2
    r0 = min(max(center_r + random.randint(-1, 1), 2), h - th - 2)
    c0 = min(max(center_c + random.randint(-1, 1), 2), w - tw - 2)
    mask = np.zeros((h, w), dtype=bool)
    mask[r0 : r0 + th, c0 : c0 + tw] = local
    return mask, (r0, c0, r0 + th - 1, c0 + tw - 1)


def _lin(r: int, c: int, width: int) -> int:
    return int(r * width + c)


def _neighbor_block(coords: np.ndarray, shape: tuple[int, int]) -> set[int]:
    h, w = shape
    blocked: set[int] = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked.add(_lin(nr, nc, w))
    return blocked


def _enumerate_candidates(shape: tuple[int, int], template_mask: np.ndarray, template_box: BBox, length: int) -> list[dict[str, object]]:
    h, w = shape
    tr0, tc0, tr1, tc1 = template_box
    candidates: list[dict[str, object]] = []
    for direction in DIRECTIONS:
        dr, dc = direction
        if dc:
            size_options = [(3, 3), (3, 4), (3, 5)]
            axis = "h"
        else:
            size_options = [(3, 3), (4, 3), (5, 3)]
            axis = "v"

        for box_h, box_w in size_options:
            r_lo = max(0, tr0 - box_h + 1)
            r_hi = min(h - box_h, tr1)
            c_lo = max(0, tc0 - box_w + 1)
            c_hi = min(w - box_w, tc1)
            for dest_r0 in range(r_lo, r_hi + 1):
                for dest_c0 in range(c_lo, c_hi + 1):
                    dest = (dest_r0, dest_c0, dest_r0 + box_h - 1, dest_c0 + box_w - 1)
                    cover = template_mask[dest[0] : dest[2] + 1, dest[1] : dest[3] + 1]
                    overlap = int(cover.sum())
                    area = box_h * box_w
                    if overlap < 3 or overlap >= area:
                        continue
                    body = _shift_box(dest, (-dr, -dc), length)
                    if not _box_in_bounds(body, shape):
                        continue
                    input_cells = _slider_cells(body, direction, length)
                    if np.any(template_mask[input_cells[:, 0], input_cells[:, 1]]):
                        continue
                    cover_set = {
                        _lin(r, c, w)
                        for r in range(dest[0], dest[2] + 1)
                        for c in range(dest[1], dest[3] + 1)
                        if template_mask[r, c]
                    }
                    if not cover_set:
                        continue
                    input_set = {_lin(int(r), int(c), w) for r, c in input_cells}
                    candidates.append(
                        {
                            "body": body,
                            "dest": dest,
                            "direction": direction,
                            "length": int(length),
                            "axis": axis,
                            "input_cells": input_cells,
                            "input_set": input_set,
                            "blocked_set": _neighbor_block(input_cells, shape),
                            "cover_set": cover_set,
                            "cover_count": overlap,
                            "area": area,
                        }
                    )
    random.shuffle(candidates)
    candidates.sort(
        key=lambda cand: (
            abs(int(cand["cover_count"]) - int(cand["area"]) // 2),
            int(cand["area"]) - int(cand["cover_count"]),
            random.random(),
        )
    )
    return candidates


def _multi_cover(selected: list[dict[str, object]]) -> set[int]:
    counts: dict[int, int] = {}
    for cand in selected:
        for idx in cand["cover_set"]:
            counts[idx] = counts.get(idx, 0) + 1
    return {idx for idx, count in counts.items() if count > 1}


def _choose_objects(
    shape: tuple[int, int],
    template_area: int,
    candidates_by_length: dict[int, list[dict[str, object]]],
    lengths: list[int],
    overlap_mode: bool,
) -> list[dict[str, object]] | None:
    def accept(selected: list[dict[str, object]], cover_union: set[int], overlap_required: bool) -> bool:
        directions = {cand["direction"] for cand in selected}
        axes = {cand["axis"] for cand in selected}
        if len(directions) < 2 or axes != {"h", "v"}:
            return False
        if len(cover_union) >= template_area:
            return False
        multi = _multi_cover(selected)
        if overlap_required:
            if not multi or len(multi) > max(6, template_area // 5):
                return False
        elif multi:
            return False
        return True

    def rec_non_overlap(
        order: list[int],
        pos: int,
        selected: list[dict[str, object]],
        blocked_input: set[int],
        cover_union: set[int],
        overlap_required: bool,
    ) -> list[dict[str, object]] | None:
        if pos == len(order):
            return selected if accept(selected, cover_union, overlap_required) else None

        length = order[pos]
        candidates = candidates_by_length[length][:]
        random.shuffle(candidates)
        for cand in candidates[:140]:
            if cand["input_set"] & blocked_input:
                continue
            if cand["cover_set"] & cover_union:
                continue
            new_cover_union = cover_union | cand["cover_set"]
            if len(new_cover_union) >= template_area:
                continue
            result = rec_non_overlap(
                order,
                pos + 1,
                selected + [cand],
                blocked_input | cand["blocked_set"],
                new_cover_union,
                overlap_required,
            )
            if result is not None:
                return result
        return None

    if not overlap_mode:
        order = sorted(lengths, key=lambda length: len(candidates_by_length[length]))
        return rec_non_overlap(order, 0, [], set(), set(), False)

    pair_options: list[tuple[dict[str, object], dict[str, object]]] = []
    for i, length_a in enumerate(lengths):
        for length_b in lengths[i + 1 :]:
            sample_a = candidates_by_length[length_a][:90]
            sample_b = candidates_by_length[length_b][:90]
            for cand_a in sample_a:
                for cand_b in sample_b:
                    inter = cand_a["cover_set"] & cand_b["cover_set"]
                    if not inter or len(inter) > 4:
                        continue
                    if cand_a["input_set"] & cand_b["blocked_set"]:
                        continue
                    if cand_b["input_set"] & cand_a["blocked_set"]:
                        continue
                    pair_options.append((cand_a, cand_b))
    random.shuffle(pair_options)

    for cand_a, cand_b in pair_options[:160]:
        selected = [cand_a, cand_b]
        blocked = cand_a["blocked_set"] | cand_b["blocked_set"]
        cover_union = cand_a["cover_set"] | cand_b["cover_set"]
        if len(cover_union) >= template_area:
            continue
        remaining = [length for length in lengths if length not in (cand_a["length"], cand_b["length"])]
        order = sorted(remaining, key=lambda length: len(candidates_by_length[length]))
        result = rec_non_overlap(order, 0, selected, blocked, cover_union, True)
        if result is not None:
            return result
    return None


def _sample_candidate(
    shape: tuple[int, int],
    template_mask: np.ndarray,
    template_box: BBox,
    length: int,
    direction: Direction,
) -> dict[str, object] | None:
    h, w = shape
    tr0, tc0, tr1, tc1 = template_box
    dr, dc = direction
    size_options = [(3, 3), (3, 4), (3, 5)] if dc else [(3, 3), (4, 3), (5, 3)]

    for _ in range(160):
        box_h, box_w = random.choice(size_options)
        r_lo = max(0, tr0 - box_h + 1)
        r_hi = min(h - box_h, tr1)
        c_lo = max(0, tc0 - box_w + 1)
        c_hi = min(w - box_w, tc1)
        if r_lo > r_hi or c_lo > c_hi:
            continue

        dest_r0 = random.randint(r_lo, r_hi)
        dest_c0 = random.randint(c_lo, c_hi)
        dest = (dest_r0, dest_c0, dest_r0 + box_h - 1, dest_c0 + box_w - 1)
        cover = template_mask[dest[0] : dest[2] + 1, dest[1] : dest[3] + 1]
        overlap = int(cover.sum())
        area = box_h * box_w
        if overlap < 3 or overlap >= area:
            continue

        body = _shift_box(dest, (-dr, -dc), length)
        if not _box_in_bounds(body, shape):
            continue
        input_cells = _slider_cells(body, direction, length)
        if np.any(template_mask[input_cells[:, 0], input_cells[:, 1]]):
            continue

        cover_set = {
            _lin(r, c, w)
            for r in range(dest[0], dest[2] + 1)
            for c in range(dest[1], dest[3] + 1)
            if template_mask[r, c]
        }
        input_set = {_lin(int(r), int(c), w) for r, c in input_cells}
        axis = "h" if dc else "v"
        return {
            "body": body,
            "dest": dest,
            "direction": direction,
            "length": int(length),
            "axis": axis,
            "input_cells": input_cells,
            "input_set": input_set,
            "blocked_set": _neighbor_block(input_cells, shape),
            "cover_set": cover_set,
            "cover_count": overlap,
            "area": area,
        }
    return None


def _choose_objects_sampled(
    shape: tuple[int, int],
    template_area: int,
    template_mask: np.ndarray,
    template_box: BBox,
    lengths: list[int],
    overlap_mode: bool,
) -> list[dict[str, object]] | None:
    for _ in range(260):
        dirs = list(random.choices(DIRECTIONS, k=len(lengths)))
        if len({direction for direction in dirs}) < 2:
            dirs[-1] = random.choice([direction for direction in DIRECTIONS if direction != dirs[0]])
        if {("h" if dc else "v") for _, dc in dirs} != {"h", "v"}:
            dirs[-1] = random.choice([(0, 1), (0, -1)] if dirs[0][1] == 0 else [(1, 0), (-1, 0)])

        selected: list[dict[str, object]] = []
        blocked_input: set[int] = set()
        cover_union: set[int] = set()
        ok = True
        for length, direction in zip(lengths, dirs):
            cand = None
            for _ in range(90):
                trial = _sample_candidate(shape, template_mask, template_box, length, direction)
                if trial is None:
                    continue
                if trial["input_set"] & blocked_input:
                    continue
                if not overlap_mode and trial["cover_set"] & cover_union:
                    continue
                if len(cover_union | trial["cover_set"]) >= template_area:
                    continue
                cand = trial
                break
            if cand is None:
                ok = False
                break
            selected.append(cand)
            blocked_input |= cand["blocked_set"]
            cover_union |= cand["cover_set"]
        if not ok:
            continue

        multi = _multi_cover(selected)
        if overlap_mode and not multi:
            continue
        if not overlap_mode and multi:
            continue
        if len(multi) > max(6, template_area // 5):
            continue
        return selected
    return None


def _render_input(shape: tuple[int, int], template_mask: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    inp = np.full(shape, BG, dtype=int)
    inp[template_mask] = TEMPLATE
    for spec in specs:
        cells = _slider_cells(spec["body"], spec["direction"], int(spec["length"]))
        inp[cells[:, 0], cells[:, 1]] = int(spec["color"])
    return inp


def _render_output(shape: tuple[int, int], template_mask: np.ndarray, specs: list[dict[str, object]], longest_on_top: bool = True) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[template_mask] = TEMPLATE
    draw_order = sorted(specs, key=lambda spec: int(spec["length"]), reverse=not longest_on_top)
    for spec in draw_order:
        r0, c0, r1, c1 = _shift_box(spec["body"], spec["direction"], int(spec["length"]))
        cover = template_mask[r0 : r1 + 1, c0 : c1 + 1]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[cover] = int(spec["color"])
    return out


def _render_no_slide(shape: tuple[int, int], template_mask: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[template_mask] = TEMPLATE
    for spec in specs:
        r0, c0, r1, c1 = spec["body"]
        cover = template_mask[r0 : r1 + 1, c0 : c1 + 1]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[cover] = int(spec["color"])
    return out


def _render_without_mask(shape: tuple[int, int], template_mask: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[template_mask] = TEMPLATE
    for spec in sorted(specs, key=lambda spec: int(spec["length"])):
        r0, c0, r1, c1 = _shift_box(spec["body"], spec["direction"], int(spec["length"]))
        out[r0 : r1 + 1, c0 : c1 + 1] = int(spec["color"])
    return out


def _validate_scene(inp: np.ndarray, expected: np.ndarray, specs: list[dict[str, object]], overlap_mode: bool) -> bool:
    if _get_bg(inp) != BG:
        return False
    gray_comps = _components_mask(inp == TEMPLATE)
    if len(gray_comps) != 1:
        return False
    template_mask = inp == TEMPLATE
    if int(template_mask.sum()) <= 0:
        return False

    if not np.array_equal(main(inp), expected):
        return False
    if np.array_equal(_render_no_slide(inp.shape, template_mask, specs), expected):
        return False
    if np.array_equal(_render_without_mask(inp.shape, template_mask, specs), expected):
        return False
    if overlap_mode and np.array_equal(_render_output(inp.shape, template_mask, specs, longest_on_top=False), expected):
        return False

    if int(np.sum(expected == TEMPLATE)) <= 0:
        return False

    by_color = {int(spec["color"]): spec for spec in specs}
    for color, spec in by_color.items():
        comps = _components_mask(inp == color)
        if len(comps) != 1:
            return False
        comp_mask = np.zeros_like(inp, dtype=bool)
        comp = comps[0]
        comp_mask[comp[:, 0], comp[:, 1]] = True
        detected = _detect_slider(comp_mask)
        if detected is None:
            return False
        if detected != (spec["body"], spec["direction"], int(spec["length"])):
            return False

    multi = _multi_cover(specs)
    if overlap_mode:
        if not multi:
            return False
    else:
        if multi:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    local_template = _make_template_local()
    h = random.randint(max(21, local_template.shape[0] + 12), min(24, local_template.shape[0] + 16))
    w = random.randint(max(22, local_template.shape[1] + 14), min(26, local_template.shape[1] + 18))
    shape = (h, w)
    template_mask, template_box = _place_template(shape, local_template)

    n_objects = random.randint(3, 4)
    lengths = random.sample([3, 4, 5, 6], n_objects)
    overlap_mode = random.random() < 0.7

    candidates_by_length: dict[int, list[dict[str, object]]] = {}
    for length in lengths:
        candidates = _enumerate_candidates(shape, template_mask, template_box, length)
        if not candidates:
            return None
        candidates_by_length[length] = candidates

    selected = _choose_objects(shape, int(template_mask.sum()), candidates_by_length, lengths, overlap_mode)
    if selected is None:
        return None

    colors = random.sample(COLOR_POOL, len(selected))
    random.shuffle(colors)
    specs: list[dict[str, object]] = []
    for cand, color in zip(selected, colors):
        specs.append(
            {
                "color": int(color),
                "body": cand["body"],
                "direction": cand["direction"],
                "length": int(cand["length"]),
                "cover_set": set(cand["cover_set"]),
            }
        )
    random.shuffle(specs)

    inp = _render_input(shape, template_mask, specs)
    expected = _render_output(shape, template_mask, specs, longest_on_top=True)
    if not _validate_scene(inp, expected, specs, overlap_mode):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle input")
