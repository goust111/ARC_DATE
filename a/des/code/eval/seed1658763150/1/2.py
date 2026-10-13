from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
FRAGMENT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert len(pts) > 0, "expected a nonempty mask"
    r0, c0, r1, c1 = _bbox(pts)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _mask_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    trimmed = _trim_mask(mask)
    return int(trimmed.shape[0]), int(trimmed.shape[1]), trimmed.astype(np.uint8).tobytes()


def _rotation_signature(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min(_mask_key(np.rot90(mask, k)) for k in range(4))


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _largest_non_bg_color(grid: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(grid[grid != int(bg)], return_counts=True)
    assert len(vals) > 0, "expected a non-background color"
    best = counts.max()
    cands = vals[counts == best]
    assert len(cands) == 1, "expected a unique dominant template color"
    return int(cands[0])


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return
    grid[top + pts[:, 0], left + pts[:, 1]] = int(color)


def _extract_template_mask(grid: np.ndarray, grey: int) -> np.ndarray:
    comps = _components(grid == int(grey))
    assert comps, "expected a grey template"
    largest = max(comps, key=len)
    mask = np.zeros_like(grid, dtype=bool)
    mask[largest[:, 0], largest[:, 1]] = True
    return mask


def _extract_slots(template_mask: np.ndarray) -> list[dict]:
    bbox = _bbox(np.argwhere(template_mask))
    r0, c0, r1, c1 = bbox
    local = template_mask[r0 : r1 + 1, c0 : c1 + 1]
    holes: list[dict] = []
    for pts in _components(~local):
        if any(
            rr == 0 or cc == 0 or rr == local.shape[0] - 1 or cc == local.shape[1] - 1
            for rr, cc in pts
        ):
            continue
        hr0, hc0, hr1, hc1 = _bbox(pts)
        mask = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
        mask[pts[:, 0] - hr0, pts[:, 1] - hc0] = True
        holes.append(
            {
                "bbox": (r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1),
                "mask": mask,
                "cells": [(int(r0 + rr), int(c0 + cc)) for rr, cc in pts],
                "area": int(len(pts)),
            }
        )
    holes.sort(key=lambda item: item["bbox"])
    assert holes, "expected at least one interior slot"
    return holes


def _extract_fragments(grid: np.ndarray, bg: int, grey: int) -> list[dict]:
    frags: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid)):
        if color in {int(bg), int(grey)}:
            continue
        for pts in _components(grid == int(color)):
            r0, c0, r1, c1 = _bbox(pts)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[pts[:, 0] - r0, pts[:, 1] - c0] = True
            frags.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "color": int(color),
                    "cells": [(int(rr), int(cc)) for rr, cc in pts],
                    "area": int(len(pts)),
                }
            )
    frags.sort(key=lambda item: item["bbox"])
    assert frags, "expected at least one fragment"
    return frags


def _match_slots_to_fragments(slots: list[dict], fragments: list[dict]) -> dict[int, int]:
    slot_by_sig: dict[tuple[int, int, bytes], int] = {}
    for idx, slot in enumerate(slots):
        sig = _rotation_signature(slot["mask"])
        assert sig not in slot_by_sig, "slot shapes must be unique up to rotation"
        slot_by_sig[sig] = idx

    mapping: dict[int, int] = {}
    for frag_idx, frag in enumerate(fragments):
        sig = _rotation_signature(frag["mask"])
        assert sig in slot_by_sig, "fragment must match a slot up to rotation"
        slot_idx = slot_by_sig[sig]
        assert slot_idx not in mapping, "matching must be one-to-one"
        mapping[slot_idx] = frag_idx

    assert len(mapping) == len(slots) == len(fragments), "every slot must have exactly one fragment"
    return mapping


def _fragment_adjacencies(grid: np.ndarray, bg: int, fragments: list[dict]) -> set[tuple[int, int]]:
    frag_id = np.full(grid.shape, -1, dtype=int)
    for idx, frag in enumerate(fragments):
        for r, c in frag["cells"]:
            frag_id[int(r), int(c)] = int(idx)

    pairs: set[tuple[int, int]] = set()
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != int(bg):
                continue
            if 0 < r < h - 1:
                a = int(frag_id[r - 1, c])
                b = int(frag_id[r + 1, c])
                if a >= 0 and b >= 0 and a != b:
                    pairs.add((min(a, b), max(a, b)))
            if 0 < c < w - 1:
                a = int(frag_id[r, c - 1])
                b = int(frag_id[r, c + 1])
                if a >= 0 and b >= 0 and a != b:
                    pairs.add((min(a, b), max(a, b)))
    return pairs


def _separator_pixels(template_mask: np.ndarray, slots: list[dict]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    slot_id = np.full(template_mask.shape, -1, dtype=int)
    for idx, slot in enumerate(slots):
        for r, c in slot["cells"]:
            slot_id[int(r), int(c)] = int(idx)

    sep: dict[tuple[int, int], set[tuple[int, int]]] = {}
    h, w = template_mask.shape
    for r in range(h):
        for c in range(w):
            if not bool(template_mask[r, c]):
                continue
            pairs: set[tuple[int, int]] = set()
            if 0 < r < h - 1:
                a = int(slot_id[r - 1, c])
                b = int(slot_id[r + 1, c])
                if a >= 0 and b >= 0 and a != b:
                    pairs.add((min(a, b), max(a, b)))
            if 0 < c < w - 1:
                a = int(slot_id[r, c - 1])
                b = int(slot_id[r, c + 1])
                if a >= 0 and b >= 0 and a != b:
                    pairs.add((min(a, b), max(a, b)))
            if len(pairs) != 1:
                continue
            pair = next(iter(pairs))
            sep.setdefault(pair, set()).add((int(r), int(c)))

    return {pair: sorted(list(pixels)) for pair, pixels in sep.items()}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored fragment to a same-shape slot inside the grey template, fill the slot,
    and recolor one-pixel grey separators only when the corresponding input fragments were also
    adjacent across a single black pixel.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    grey = _largest_non_bg_color(g, bg)
    template_mask = _extract_template_mask(g, grey)
    slots = _extract_slots(template_mask)
    fragments = _extract_fragments(g, bg, grey)
    slot_to_fragment = _match_slots_to_fragments(slots, fragments)
    fragment_adj = _fragment_adjacencies(g, bg, fragments)
    separators = _separator_pixels(template_mask, slots)

    out = np.full_like(g, int(bg))
    out[template_mask] = int(grey)

    for slot_idx, slot in enumerate(slots):
        frag = fragments[slot_to_fragment[slot_idx]]
        for r, c in slot["cells"]:
            out[int(r), int(c)] = int(frag["color"])

    for (slot_a, slot_b), pixels in separators.items():
        frag_a = slot_to_fragment[slot_a]
        frag_b = slot_to_fragment[slot_b]
        if (min(frag_a, frag_b), max(frag_a, frag_b)) not in fragment_adj:
            continue
        area_a = int(fragments[frag_a]["area"])
        area_b = int(fragments[frag_b]["area"])
        if area_a == area_b:
            continue
        winner = frag_a if area_a > area_b else frag_b
        color = int(fragments[winner]["color"])
        for r, c in pixels:
            out[int(r), int(c)] = int(color)

    return out


def _random_shape(area: int) -> np.ndarray | None:
    side = area + 2
    mask = np.zeros((side, side), dtype=bool)
    center = side // 2
    mask[center, center] = True

    while int(mask.sum()) < int(area):
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < side and 0 <= nc < side and not bool(mask[nr, nc]):
                    frontier.add((int(nr), int(nc)))
        if not frontier:
            return None
        cells = list(frontier)
        random.shuffle(cells)
        weights = []
        for r, c in cells:
            degree = sum(
                1
                for nr, nc in _neighbors4(r, c)
                if 0 <= nr < side and 0 <= nc < side and bool(mask[nr, nc])
            )
            weights.append(3 if degree == 1 else 1)
        choice = random.choices(cells, weights=weights, k=1)[0]
        mask[choice] = True

    trimmed = _trim_mask(mask)
    if min(trimmed.shape) < 2 or max(trimmed.shape) > 4:
        return None
    return trimmed


def _edge_overlap(mask_a: np.ndarray, mask_b: np.ndarray, axis: str, delta: int) -> bool:
    if axis == "h":
        rows_a = {int(r) for r in np.where(mask_a[:, -1])[0]}
        rows_b = {int(r + delta) for r in np.where(mask_b[:, 0])[0]}
        return bool(rows_a & rows_b)
    cols_a = {int(c) for c in np.where(mask_a[-1, :])[0]}
    cols_b = {int(c + delta) for c in np.where(mask_b[0, :])[0]}
    return bool(cols_a & cols_b)


def _place_chain(
    masks: list[np.ndarray],
    axis: str,
    gaps: list[int],
    require_edges: list[bool],
) -> list[tuple[int, int]] | None:
    assert len(masks) >= 2
    assert len(gaps) == len(masks) - 1
    assert len(require_edges) == len(masks) - 1

    positions: list[tuple[int, int]] = [(0, 0)]
    for i in range(1, len(masks)):
        prev = masks[i - 1]
        curr = masks[i]
        candidates: list[tuple[int, int, int]] = []
        if axis == "h":
            delta_range = list(range(-(curr.shape[0] - 1), prev.shape[0]))
            random.shuffle(delta_range)
            for delta in delta_range:
                if require_edges[i - 1] and not _edge_overlap(prev, curr, axis, delta):
                    continue
                r = positions[-1][0] + int(delta)
                c = positions[-1][1] + int(prev.shape[1]) + int(gaps[i - 1])
                candidates.append((abs(int(delta)), r, c))
        else:
            delta_range = list(range(-(curr.shape[1] - 1), prev.shape[1]))
            random.shuffle(delta_range)
            for delta in delta_range:
                if require_edges[i - 1] and not _edge_overlap(prev, curr, axis, delta):
                    continue
                r = positions[-1][0] + int(prev.shape[0]) + int(gaps[i - 1])
                c = positions[-1][1] + int(delta)
                candidates.append((abs(int(delta)), r, c))
        if not candidates:
            return None
        best_abs = min(item[0] for item in candidates)
        tight = [(r, c) for delta_abs, r, c in candidates if delta_abs <= best_abs + 1]
        positions.append(random.choice(tight))

    min_r = min(r for r, _c in positions)
    min_c = min(c for _r, c in positions)
    return [(int(r - min_r), int(c - min_c)) for r, c in positions]


def _irregular_template_rect(height: int, width: int) -> np.ndarray:
    for _ in range(20):
        mask = np.ones((height, width), dtype=bool)
        for r in range(height):
            left_trim = random.randint(0, 1)
            right_trim = random.randint(0, 1)
            if left_trim:
                mask[r, :left_trim] = False
            if right_trim:
                mask[r, width - right_trim :] = False
        for c in range(width):
            top_trim = random.randint(0, 1)
            bottom_trim = random.randint(0, 1)
            if top_trim:
                mask[:top_trim, c] = False
            if bottom_trim:
                mask[height - bottom_trim :, c] = False
        comps = _components(mask)
        if len(comps) == 1:
            return mask
    return np.ones((height, width), dtype=bool)


def _build_template_patch(
    slot_masks: list[np.ndarray], positions: list[tuple[int, int]], pad: int
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    inner_h = max(r + mask.shape[0] for mask, (r, _c) in zip(slot_masks, positions, strict=False))
    inner_w = max(c + mask.shape[1] for mask, (_r, c) in zip(slot_masks, positions, strict=False))
    patch_h = inner_h + 2 * pad
    patch_w = inner_w + 2 * pad
    base = _irregular_template_rect(patch_h, patch_w)
    hole_ids = np.full((patch_h, patch_w), -1, dtype=int)
    shifted: list[tuple[int, int]] = []
    for idx, (mask, (r, c)) in enumerate(zip(slot_masks, positions, strict=False)):
        rr = int(r + pad)
        cc = int(c + pad)
        base_view = base[rr : rr + mask.shape[0], cc : cc + mask.shape[1]]
        assert bool(np.all(base_view[mask]))
        id_view = hole_ids[rr : rr + mask.shape[0], cc : cc + mask.shape[1]].copy()
        id_view[mask] = int(idx)
        hole_ids[rr : rr + mask.shape[0], cc : cc + mask.shape[1]] = id_view
        shifted.append((rr, cc))
    template = base.copy()
    template[hole_ids >= 0] = False
    return template, hole_ids, shifted


def _separator_pixels_from_holes(template_mask: np.ndarray, hole_ids: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    sep: dict[tuple[int, int], set[tuple[int, int]]] = {}
    h, w = template_mask.shape
    for r in range(h):
        for c in range(w):
            if not bool(template_mask[r, c]):
                continue
            pairs: set[tuple[int, int]] = set()
            if 0 < r < h - 1:
                a = int(hole_ids[r - 1, c])
                b = int(hole_ids[r + 1, c])
                if a >= 0 and b >= 0 and a != b:
                    pairs.add((min(a, b), max(a, b)))
            if 0 < c < w - 1:
                a = int(hole_ids[r, c - 1])
                b = int(hole_ids[r, c + 1])
                if a >= 0 and b >= 0 and a != b:
                    pairs.add((min(a, b), max(a, b)))
            if len(pairs) != 1:
                continue
            pair = next(iter(pairs))
            sep.setdefault(pair, set()).add((int(r), int(c)))
    return {pair: sorted(list(pixels)) for pair, pixels in sep.items()}


def _build_fragment_patch(
    fragment_masks: list[np.ndarray], axis: str, adj_flags: list[bool]
) -> tuple[np.ndarray, list[tuple[int, int]], list[int]] | None:
    gaps = [1 if flag else random.randint(2, 4) for flag in adj_flags]
    positions = _place_chain(fragment_masks, axis=axis, gaps=gaps, require_edges=adj_flags)
    if positions is None:
        return None
    height = max(r + mask.shape[0] for mask, (r, _c) in zip(fragment_masks, positions, strict=False))
    width = max(c + mask.shape[1] for mask, (_r, c) in zip(fragment_masks, positions, strict=False))
    return np.zeros((height, width), dtype=int), positions, gaps


def _compose_grid(
    template_patch: np.ndarray,
    slot_masks: list[np.ndarray],
    slot_positions: list[tuple[int, int]],
    fragment_masks: list[np.ndarray],
    fragment_positions: list[tuple[int, int]],
    colors: list[int],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    th, tw = template_patch.shape
    fh = max(r + mask.shape[0] for mask, (r, _c) in zip(fragment_masks, fragment_positions, strict=False))
    fw = max(c + mask.shape[1] for mask, (_r, c) in zip(fragment_masks, fragment_positions, strict=False))

    if fw >= fh:
        side = random.choice(["top", "bottom"])
    else:
        side = random.choice(["left", "right"])
    gap = random.randint(2, 4)
    base_margin = random.randint(1, 3)

    if side == "top":
        mt = base_margin
        mb = max(1, base_margin + (fh + gap) // 2 + random.randint(-1, 1))
        ml = random.randint(1, 3)
        mr = random.randint(1, 3)
        H = mt + fh + gap + th + mb
        W = ml + max(tw, fw) + mr
        tr = mt + fh + gap
        fr = mt
        tc = ml + random.randint(0, max(tw, fw) - tw)
        fc = ml + random.randint(0, max(tw, fw) - fw)
    elif side == "bottom":
        mt = max(1, base_margin + (fh + gap) // 2 + random.randint(-1, 1))
        mb = base_margin
        ml = random.randint(1, 3)
        mr = random.randint(1, 3)
        H = mt + th + gap + fh + mb
        W = ml + max(tw, fw) + mr
        tr = mt
        fr = mt + th + gap
        tc = ml + random.randint(0, max(tw, fw) - tw)
        fc = ml + random.randint(0, max(tw, fw) - fw)
    elif side == "left":
        ml = base_margin
        mr = max(1, base_margin + (fw + gap) // 2 + random.randint(-1, 1))
        mt = random.randint(1, 3)
        mb = random.randint(1, 3)
        H = mt + max(th, fh) + mb
        W = ml + fw + gap + tw + mr
        tr = mt + random.randint(0, max(th, fh) - th)
        fr = mt + random.randint(0, max(th, fh) - fh)
        tc = ml + fw + gap
        fc = ml
    else:
        ml = max(1, base_margin + (fw + gap) // 2 + random.randint(-1, 1))
        mr = base_margin
        mt = random.randint(1, 3)
        mb = random.randint(1, 3)
        H = mt + max(th, fh) + mb
        W = ml + tw + gap + fw + mr
        tr = mt + random.randint(0, max(th, fh) - th)
        fr = mt + random.randint(0, max(th, fh) - fh)
        tc = ml
        fc = ml + tw + gap

    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    _paint_mask(inp, template_patch, tr, tc, GREY)
    _paint_mask(out, template_patch, tr, tc, GREY)

    for color, mask, (r, c) in zip(colors, fragment_masks, fragment_positions, strict=False):
        _paint_mask(inp, mask, fr + r, fc + c, color)

    for color, mask, (r, c) in zip(colors, slot_masks, slot_positions, strict=False):
        _paint_mask(out, mask, tr + r, tc + c, color)

    return inp, out, (tr, tc)


def _gen_attempt() -> np.ndarray | None:
    n_slots = random.randint(2, 4)
    areas = random.sample([3, 4, 5, 6], k=n_slots)

    base_shapes: list[np.ndarray] = []
    seen_sigs: set[tuple[int, int, bytes]] = set()
    for area in areas:
        shape = None
        for _ in range(60):
            cand = _random_shape(area)
            if cand is None:
                continue
            sig = _rotation_signature(cand)
            if sig in seen_sigs:
                continue
            shape = cand
            seen_sigs.add(sig)
            break
        if shape is None:
            return None
        base_shapes.append(shape)

    slot_masks = [np.rot90(shape, random.randint(0, 3)) for shape in base_shapes]
    template_axis = random.choice(["h", "v"])
    template_positions = _place_chain(
        slot_masks,
        axis=template_axis,
        gaps=[1] * (n_slots - 1),
        require_edges=[True] * (n_slots - 1),
    )
    if template_positions is None:
        return None

    pad = random.randint(3, 4)
    template_patch, hole_ids, shifted_slots = _build_template_patch(slot_masks, template_positions, pad)
    if len(_components(template_patch)) != 1:
        return None

    separators = _separator_pixels_from_holes(template_patch, hole_ids)
    path_edges = {(i, i + 1) for i in range(n_slots - 1)}
    if set(separators) != path_edges:
        return None

    fragment_adj_flags = [bool(random.getrandbits(1)) for _ in range(n_slots - 1)]
    fragment_masks = [np.rot90(shape, random.randint(0, 3)) for shape in base_shapes]
    fragment_axis = random.choice(["h", "v"])
    fragment_pack = _build_fragment_patch(fragment_masks, fragment_axis, fragment_adj_flags)
    if fragment_pack is None:
        return None
    _fragment_canvas, fragment_positions, _fragment_gaps = fragment_pack

    colors = random.sample(FRAGMENT_COLORS, k=n_slots)
    inp, expected, (tr, tc) = _compose_grid(
        template_patch,
        slot_masks,
        shifted_slots,
        fragment_masks,
        fragment_positions,
        colors,
    )

    for i, flag in enumerate(fragment_adj_flags):
        if not flag:
            continue
        area_a = int(slot_masks[i].sum())
        area_b = int(slot_masks[i + 1].sum())
        if area_a == area_b:
            continue
        winner = i if area_a > area_b else i + 1
        color = int(colors[winner])
        for r, c in separators[(i, i + 1)]:
            expected[tr + int(r), tc + int(c)] = int(color)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
