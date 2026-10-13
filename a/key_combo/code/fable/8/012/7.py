from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_true(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    cropped = _crop_true(mask)
    packed = np.ascontiguousarray(cropped.astype(np.uint8))
    return tuple(map(int, packed.shape)), packed.tobytes()


def _extract_hole(local_mask: np.ndarray) -> np.ndarray:
    h, w = local_mask.shape
    seen = np.zeros_like(local_mask, dtype=bool)
    holes = []
    for r in range(h):
        for c in range(w):
            if local_mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            touches_border = False
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not local_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                holes.append(coords)
    assert len(holes) == 1
    hole = np.zeros_like(local_mask, dtype=bool)
    for r, c in holes[0]:
        hole[r, c] = True
    return hole


def _component_info(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    color = int(grid[coords[0, 0], coords[0, 1]])
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    hole = _extract_hole(local)
    outline = local | hole
    return {
        "color": color,
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "local_mask": local,
        "hole_mask": hole,
        "outline_mask": outline,
        "hole_sig": _mask_signature(hole),
        "outline_sig": _mask_signature(outline),
    }


def _adjacent_grey_dirs(grid: np.ndarray, coords: np.ndarray) -> set[tuple[int, int]]:
    h, w = grid.shape
    dirs = set()
    for r, c in coords:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GREY:
                dirs.add((dr, dc))
    return dirs


def _extract_slabs(grid: np.ndarray) -> tuple[dict, list[dict]]:
    slab_infos = []
    for color in sorted(set(map(int, np.unique(grid))) - {BLACK, GREY}):
        for coords in _find_components(grid == color):
            slab_infos.append(_component_info(grid, coords))

    template_candidates = []
    for info in slab_infos:
        dirs = _adjacent_grey_dirs(grid, info["coords"])
        has_vertical = any(dr != 0 for dr, _ in dirs)
        has_horizontal = any(dc != 0 for _, dc in dirs)
        if has_vertical and has_horizontal:
            template_candidates.append(info)

    assert len(template_candidates) == 1
    template = template_candidates[0]
    others = [info for info in slab_infos if info is not template]
    return template, others


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the slabs hit by the hole-to-outline relay chain to the framed template slab's color.

    Input structure:
    - A black background contains one grey L-shaped frame and several colored slab objects.
    - The slab tucked into the frame has a single black hole; every free slab also has exactly one black hole.
    - For each free slab, its matchable outline is the solid shape obtained by filling its own hole.

    Rule:
    - Start from the framed template slab's hole shape.
    - Repeatedly find the unique unused free slab whose filled outline is translation-normalized identical to the current hole.
    - After claiming a slab, continue from that slab's own hole; stop when no unused slab outline matches.
    - Recolor every claimed slab to the template slab's color while leaving all holes, the frame, the template slab, and all decoys unchanged.

    Key insight:
    - Matching alternates between negative and positive space: a hole claims a solid outline, then that slab's hole becomes the next query.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g == GREY)

    template, slabs = _extract_slabs(g)
    target_color = template["color"]

    out = g.copy()
    current_sig = template["hole_sig"]
    remaining = list(slabs)
    while True:
        matches = [info for info in remaining if info["outline_sig"] == current_sig]
        if not matches:
            break
        assert len(matches) == 1
        match = matches[0]
        remaining.remove(match)
        rows = match["coords"][:, 0]
        cols = match["coords"][:, 1]
        out[rows, cols] = target_color
        current_sig = match["hole_sig"]

    return out


def _normalize_shape(mask: np.ndarray) -> np.ndarray:
    return _crop_true(mask)


def _random_polyomino(area: int, max_h: int = 4, max_w: int = 4) -> np.ndarray | None:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in cells:
                    trial = cells | {(nr, nc)}
                    rows = [x for x, _ in trial]
                    cols = [y for _, y in trial]
                    if max(rows) - min(rows) + 1 <= max_h and max(cols) - min(cols) + 1 <= max_w:
                        frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    h = max(rs) - min(rs) + 1
    w = max(cs) - min(cs) + 1
    out = np.zeros((h, w), dtype=bool)
    r0 = min(rs)
    c0 = min(cs)
    for r, c in cells:
        out[r - r0, c - c0] = True
    return out


def _rect_mask(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _random_rect_shape(min_h: int, max_h: int, min_w: int, max_w: int) -> np.ndarray:
    return _rect_mask(random.randint(min_h, max_h), random.randint(min_w, max_w))


def _build_rect_slab_from_inner(inner_shape: np.ndarray, max_extra: int = 2) -> tuple[np.ndarray, np.ndarray]:
    inner = _normalize_shape(inner_shape)
    ih, iw = inner.shape
    oh = ih + random.randint(2, max_extra)
    ow = iw + random.randint(2, max_extra)
    outline = np.ones((oh, ow), dtype=bool)
    hole = np.zeros((oh, ow), dtype=bool)
    hole_r = random.randint(1, oh - ih - 1)
    hole_c = random.randint(1, ow - iw - 1)
    hole[hole_r : hole_r + ih, hole_c : hole_c + iw] = inner
    return outline, hole


def _transform_mask(mask: np.ndarray, code: int) -> np.ndarray:
    out = mask
    if code >= 4:
        out = np.fliplr(out)
    return np.rot90(out, code % 4).copy()


def _render_slab(outline: np.ndarray, hole: np.ndarray, color: int) -> np.ndarray:
    arr = np.zeros_like(outline, dtype=int)
    arr[outline & ~hole] = color
    return arr


def _build_framed_template(template_slab: np.ndarray, corner: str) -> np.ndarray:
    h, w = template_slab.shape
    out = np.zeros((h + 2, w + 2), dtype=int)
    if corner == "tl":
        out[0, :] = GREY
        out[:, 0] = GREY
        out[1 : h + 1, 1 : w + 1] = template_slab
    elif corner == "tr":
        out[0, :] = GREY
        out[:, -1] = GREY
        out[1 : h + 1, 0:w] = template_slab
    elif corner == "bl":
        out[-1, :] = GREY
        out[:, 0] = GREY
        out[0:h, 1 : w + 1] = template_slab
    else:
        assert corner == "br"
        out[-1, :] = GREY
        out[:, -1] = GREY
        out[0:h, 0:w] = template_slab
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        for nr, nc in _neighbors8(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _forbidden_offsets(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        coords.add((rr, cc))
        for nr, nc in _neighbors8(rr, cc):
            coords.add((nr, nc))
    return sorted(coords)


def _place_objects(objects: list[np.ndarray]) -> np.ndarray | None:
    objects = sorted(objects, key=lambda arr: int(np.count_nonzero(arr)), reverse=True)
    total = sum(int(np.count_nonzero(arr)) for arr in objects)
    min_side = max(max(arr.shape[0] for arr in objects), max(arr.shape[1] for arr in objects)) + 2
    base = min(30, max(18, min_side, int(np.ceil(np.sqrt(total * 2.0)))))
    for _ in range(80):
        h = random.randint(base, 30)
        w = random.randint(base, 30)
        canvas = np.zeros((h, w), dtype=int)
        blocked = np.zeros((h, w), dtype=bool)
        placed = True
        for obj in objects:
            oh, ow = obj.shape
            mask = obj != BLACK
            forbid_offsets = _forbidden_offsets(mask)
            positions = []
            for r0 in range(h - oh + 1):
                for c0 in range(w - ow + 1):
                    ok = True
                    for dr, dc in forbid_offsets:
                        rr = r0 + dr
                        cc = c0 + dc
                        if 0 <= rr < h and 0 <= cc < w and blocked[rr, cc]:
                            ok = False
                            break
                    if not ok:
                        continue
                    positions.append((r0, c0))
            if not positions:
                placed = False
                break
            r0, c0 = random.choice(positions)
            patch = canvas[r0 : r0 + oh, c0 : c0 + ow]
            patch[mask] = obj[mask]
            for dr, dc in forbid_offsets:
                rr = r0 + dr
                cc = c0 + dc
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True
        if placed:
            return canvas
    return None


def _make_chain_specs(chain_len: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]] | None:
    next_shape = _random_rect_shape(1, 3, 1, 3)
    chain_rev = []
    for _ in range(chain_len):
        outline, hole = _build_rect_slab_from_inner(next_shape, max_extra=2)
        chain_rev.append((outline, hole))
        next_shape = outline

    template = _build_rect_slab_from_inner(next_shape, max_extra=2)
    chain_rev.reverse()
    return chain_rev, template


def _make_decoy(used_sigs: set[tuple[tuple[int, int], bytes]]) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(80):
        inner = _random_rect_shape(1, 4, 1, 4)
        outline, hole = _build_rect_slab_from_inner(inner, max_extra=2)
        outline_sig = _mask_signature(outline)
        hole_sig = _mask_signature(hole)
        if outline_sig in used_sigs or hole_sig in used_sigs:
            continue
        return outline, hole
    return None


def _oracle_chain_count(inp: np.ndarray, out: np.ndarray) -> int:
    template, slabs = _extract_slabs(inp)
    target = template["color"]
    changed = 0
    for info in slabs:
        rows = info["coords"][:, 0]
        cols = info["coords"][:, 1]
        if np.all(out[rows, cols] == target) and np.any(inp[rows, cols] != target):
            changed += 1
    return changed


def _gen_attempt() -> np.ndarray | None:
    chain_len = random.randint(1, 3)
    decoy_count = random.randint(1, min(3, 8 - (chain_len + 1)))
    chain_specs_template = _make_chain_specs(chain_len)
    if chain_specs_template is None:
        return None
    chain_specs, template_spec = chain_specs_template

    global_transform = random.randint(0, 7)
    chain_specs = [
        (_transform_mask(outline, global_transform), _transform_mask(hole, global_transform))
        for outline, hole in chain_specs
    ]
    template_outline, template_hole = template_spec
    template_outline = _transform_mask(template_outline, global_transform)
    template_hole = _transform_mask(template_hole, global_transform)

    total_slab_count = 1 + chain_len + decoy_count
    colors = random.sample(PALETTE, total_slab_count)
    template_color = colors[0]
    chain_colors = colors[1 : 1 + chain_len]
    decoy_colors = colors[1 + chain_len :]

    objects = []
    used_sigs = {
        _mask_signature(template_outline),
        _mask_signature(template_hole),
    }
    for outline, hole in chain_specs:
        used_sigs.add(_mask_signature(outline))
        used_sigs.add(_mask_signature(hole))

    template_arr = _render_slab(template_outline, template_hole, template_color)
    frame_corner = random.choice(("tl", "tr", "bl", "br"))
    objects.append(_build_framed_template(template_arr, frame_corner))

    for (outline, hole), color in zip(chain_specs, chain_colors):
        objects.append(_render_slab(outline, hole, color))

    for color in decoy_colors:
        decoy = _make_decoy(used_sigs)
        if decoy is None:
            return None
        outline, hole = decoy
        decoy_transform = random.randint(0, 7)
        outline = _transform_mask(outline, decoy_transform)
        hole = _transform_mask(hole, decoy_transform)
        outline_sig = _mask_signature(outline)
        hole_sig = _mask_signature(hole)
        if outline_sig in used_sigs or hole_sig in used_sigs:
            return None
        used_sigs.add(outline_sig)
        used_sigs.add(hole_sig)
        objects.append(_render_slab(outline, hole, color))

    inp = _place_objects(objects)
    if inp is None:
        return None

    try:
        out = main(inp)
    except AssertionError:
        return None

    changed = _oracle_chain_count(inp, out)
    if changed != chain_len:
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
