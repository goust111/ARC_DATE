import random
from collections import Counter

import numpy as np


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _bbox_from_coords(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros((0, 0), dtype=bool)
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _mask_components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _foreground_components(grid: np.ndarray, bg: int):
    h, w = grid.shape
    mask = grid != bg
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            comps.append({"coords": arr, "bbox": (r0, c0, r1, c1), "area": len(coords)})
    return comps


def _paste(dst: np.ndarray, src: np.ndarray, top: int, left: int, mask: np.ndarray | None = None):
    h, w = src.shape
    if mask is None:
        dst[top : top + h, left : left + w] = src
        return
    view = dst[top : top + h, left : left + w]
    view[mask] = src[mask]


def _dihedral_masks(mask: np.ndarray):
    variants = []
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for cand in (rot, np.fliplr(rot)):
            key = (cand.shape, cand.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(cand.copy())
    return variants


def _canonical_key(mask: np.ndarray):
    keys = []
    for cand in _dihedral_masks(_trim_mask(mask)):
        keys.append((cand.shape, cand.tobytes()))
    return min(keys)


def _downsample_blocks(mask: np.ndarray, k: int):
    h, w = mask.shape
    if h % k != 0 or w % k != 0:
        return None
    out = np.zeros((h // k, w // k), dtype=bool)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            block = mask[r * k : (r + 1) * k, c * k : (c + 1) * k]
            if block.all():
                out[r, c] = True
            elif block.any():
                return None
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])

    comps = _foreground_components(g, bg)
    assert comps, "expected a frame component"
    frame = max(comps, key=lambda comp: int(comp["area"]))
    fr0, fc0, fr1, fc1 = frame["bbox"]

    frame_colors = [int(g[r, c]) for r, c in frame["coords"]]
    frame_color = int(Counter(frame_colors).most_common(1)[0][0])

    frame_patch = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    hole_mask = frame_patch == bg
    hole_box = _bbox_from_coords(np.argwhere(hole_mask))
    hr0, hc0, hr1, hc1 = hole_box
    layout = hole_mask[hr0 : hr1 + 1, hc0 : hc1 + 1]

    slots = []
    for coords in _mask_components(layout):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        slots.append({"r0": r0, "c0": c0, "mask": mask, "area": int(mask.sum())})
    slots.sort(key=lambda s: (-s["area"], s["r0"], s["c0"]))
    assert slots, "frame must contain at least one hole"

    pieces = []
    for comp in comps:
        if comp is frame or comp["area"] <= 1:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[comp["coords"][:, 0] - r0, comp["coords"][:, 1] - c0] = True
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        pieces.append({"patch": patch, "mask": mask, "area": int(mask.sum())})
    assert pieces, "expected outside pieces"

    solid_square_sizes = []
    for piece in pieces:
        h, w = piece["mask"].shape
        if h == w and piece["mask"].all():
            solid_square_sizes.append(h)
    assert solid_square_sizes, "expected at least one solid square piece"
    k = min(solid_square_sizes)

    candidates = []
    for idx, piece in enumerate(pieces):
        for krot in range(4):
            patch_rot = np.rot90(piece["patch"], krot)
            mask_rot = np.rot90(piece["mask"], krot)
            for patch_var, mask_var in ((patch_rot, mask_rot), (np.fliplr(patch_rot), np.fliplr(mask_rot))):
                down = _downsample_blocks(mask_var, k)
                if down is None:
                    continue
                candidates.append(
                    {
                        "piece_idx": idx,
                        "area": int(piece["area"]),
                        "down": down,
                        "patch": patch_var,
                        "mask": mask_var,
                    }
                )

    used = set()
    placements = []
    for slot in slots:
        need_area = slot["area"] * k * k
        match = None
        for cand in candidates:
            if cand["piece_idx"] in used:
                continue
            if cand["area"] != need_area:
                continue
            if cand["down"].shape != slot["mask"].shape:
                continue
            if np.array_equal(cand["down"], slot["mask"]):
                match = cand
                break
        assert match is not None, "slot-piece matching failed"
        used.add(match["piece_idx"])
        placements.append((slot, match))

    out = np.full(((layout.shape[0] + 2) * k, (layout.shape[1] + 2) * k), frame_color, dtype=int)
    for slot, match in placements:
        top = (slot["r0"] + 1) * k
        left = (slot["c0"] + 1) * k
        _paste(out, match["patch"], top, left, match["mask"])
    return out


def _random_connected_mask(area: int, max_dim: int = 4) -> np.ndarray | None:
    board = np.zeros((max_dim, max_dim), dtype=bool)
    start = (random.randrange(max_dim), random.randrange(max_dim))
    cells = {start}
    board[start] = True
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c, max_dim, max_dim):
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(frontier)
        cells.add((nr, nc))
        board[nr, nc] = True
    trimmed = _trim_mask(board)
    if trimmed.size == 0:
        return None
    return trimmed


def _sample_slot_masks():
    count = random.randint(3, 5)
    masks = [np.array([[True]], dtype=bool)]
    seen = {_canonical_key(masks[0])}
    while len(masks) < count:
        area = random.randint(2, 5)
        mask = _random_connected_mask(area, max_dim=random.randint(3, 5))
        if mask is None:
            continue
        if max(mask.shape) > 4:
            continue
        key = _canonical_key(mask)
        if key in seen:
            continue
        seen.add(key)
        masks.append(mask)
    random.shuffle(masks)
    return masks


def _try_pack_layout(masks):
    layout_h = random.randint(7, 10)
    layout_w = random.randint(7, 10)
    layout = np.zeros((layout_h, layout_w), dtype=bool)
    placements = []
    ordered = sorted(masks, key=lambda m: -int(m.sum()))
    for mask in ordered:
        placed = False
        variants = _dihedral_masks(mask)
        random.shuffle(variants)
        for var in variants:
            h, w = var.shape
            positions = [(r, c) for r in range(layout_h - h + 1) for c in range(layout_w - w + 1)]
            random.shuffle(positions)
            for r0, c0 in positions:
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(layout_h, r0 + h + 1)
                cc1 = min(layout_w, c0 + w + 1)
                if layout[rr0:rr1, cc0:cc1].any():
                    continue
                if np.any(layout[r0 : r0 + h, c0 : c0 + w] & var):
                    continue
                layout[r0 : r0 + h, c0 : c0 + w] |= var
                placements.append({"mask": var.copy(), "r0": r0, "c0": c0})
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return layout, placements


def _build_expected(layout: np.ndarray, slot_infos, k: int, frame_color: int):
    out = np.full(((layout.shape[0] + 2) * k, (layout.shape[1] + 2) * k), frame_color, dtype=int)
    for info in slot_infos:
        h, w = info["mask"].shape
        patch = np.full((h * k, w * k), info["color"], dtype=int)
        mask = np.kron(info["mask"], np.ones((k, k), dtype=bool))
        _paste(out, patch, (info["r0"] + 1) * k, (info["c0"] + 1) * k, mask)
    return out


def _scatter_objects(canvas: np.ndarray, objects):
    h, w = canvas.shape
    occupied = np.zeros((h, w), dtype=bool)
    for obj in sorted(objects, key=lambda x: -(x["patch"].shape[0] * x["patch"].shape[1])):
        ph, pw = obj["patch"].shape
        positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc1 = min(w, c0 + pw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            view = canvas[r0 : r0 + ph, c0 : c0 + pw]
            if np.any((obj["mask"]) & (view != obj["bg"])):
                continue
            view[obj["mask"]] = obj["patch"][obj["mask"]]
            occupied[r0 : r0 + ph, c0 : c0 + pw] |= obj["mask"]
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt():
    bg = 0
    frame_color = random.choice([c for c in range(1, 10)])
    k = random.choice([2, 3])

    masks = _sample_slot_masks()
    packed = None
    for _ in range(40):
        packed = _try_pack_layout(masks)
        if packed is not None:
            break
    if packed is None:
        return None
    layout, placements = packed

    hole_area = int(layout.sum())
    frame_area = int((layout.shape[0] + 2) * (layout.shape[1] + 2) - hole_area)
    max_piece_area = max(int(info["mask"].sum()) * k * k for info in placements)
    if frame_area <= max_piece_area:
        return None

    piece_colors = random.sample([c for c in range(1, 10) if c != frame_color], len(placements))
    slot_infos = []
    piece_objects = []
    for info, color in zip(placements, piece_colors):
        base_mask = info["mask"]
        h, w = base_mask.shape
        patch = np.full((h * k, w * k), color, dtype=int)
        mask = np.kron(base_mask, np.ones((k, k), dtype=bool))
        variants = []
        for rot in range(4):
            p = np.rot90(patch, rot)
            m = np.rot90(mask, rot)
            variants.append((p, m))
            variants.append((np.fliplr(p), np.fliplr(m)))
        piece_patch, piece_mask = random.choice(variants)
        slot_infos.append({"mask": base_mask, "r0": info["r0"], "c0": info["c0"], "color": color})
        piece_objects.append({"patch": piece_patch, "mask": piece_mask, "bg": bg})

    frame_patch = np.full((layout.shape[0] + 2, layout.shape[1] + 2), frame_color, dtype=int)
    frame_patch[1:-1, 1:-1][layout] = bg

    min_h = max(16, frame_patch.shape[0] + 4)
    min_w = max(16, frame_patch.shape[1] + 4)
    area_budget = frame_patch.size + sum(obj["patch"].size for obj in piece_objects) + 80
    canvas_h = random.randint(min_h, max(min_h, min(28, int(np.sqrt(area_budget) * 2.2))))
    canvas_w = random.randint(min_w, max(min_w, min(28, int(np.sqrt(area_budget) * 2.4))))
    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)

    objects = [{"patch": frame_patch, "mask": np.ones(frame_patch.shape, dtype=bool), "bg": bg}] + piece_objects
    if not _scatter_objects(canvas, objects):
        return None

    expected = _build_expected(layout, slot_infos, k, frame_color)
    try:
        got = main(canvas)
    except AssertionError:
        return None
    if not np.array_equal(got, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
