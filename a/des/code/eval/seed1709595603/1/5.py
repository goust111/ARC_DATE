from __future__ import annotations

import random
from typing import Optional

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _bbox_of_mask(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = [(r, c)]
            coords = []
            while queue:
                rr, cc = queue.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    if not mask.any():
        return mask.copy()

    outside = ~mask
    seen = np.zeros((h, w), dtype=bool)
    queue = []
    for r in range(h):
        for c in (0, w - 1):
            if outside[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if outside[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        rr, cc = queue.pop()
        for dr, dc in DIR4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and outside[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))

    filled = mask.copy()
    filled[outside & ~seen] = True
    return filled


def _all_dihedral_masks(mask: np.ndarray) -> list[np.ndarray]:
    base = np.asarray(mask, dtype=bool)
    variants: list[np.ndarray] = []
    seen = set()
    for flip in (False, True):
        arr = np.fliplr(base) if flip else base
        for k in range(4):
            cur = np.rot90(arr, k)
            key = (cur.shape, cur.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(cur.copy())
    return variants


def _all_dihedral_pairs(mask_a: np.ndarray, mask_b: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    a0 = np.asarray(mask_a, dtype=bool)
    b0 = np.asarray(mask_b, dtype=bool)
    variants: list[tuple[np.ndarray, np.ndarray]] = []
    seen = set()
    for flip in (False, True):
        a1 = np.fliplr(a0) if flip else a0
        b1 = np.fliplr(b0) if flip else b0
        for k in range(4):
            aa = np.rot90(a1, k)
            bb = np.rot90(b1, k)
            key = (aa.shape, aa.tobytes(), bb.shape, bb.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append((aa.copy(), bb.copy()))
    return variants


def _canonical_mask(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    keys = [(variant.shape, variant.tobytes()) for variant in _all_dihedral_masks(mask)]
    return min(keys)


def _frame_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    candidates: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(sub[0, :] == color):
            continue
        if not np.all(sub[-1, :] == color):
            continue
        if not np.all(sub[:, 0] == color):
            continue
        if not np.all(sub[:, -1] == color):
            continue
        inside = sub[1:-1, 1:-1]
        if inside.size == 0 or not np.any(inside == bg):
            continue
        candidates.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "bbox_area": int(h * w),
                "color_area": int(coords.shape[0]),
            }
        )
    candidates.sort(key=lambda d: (-d["bbox_area"], -d["color_area"], d["color"]))
    return candidates


def _extract_slots(grid: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = frame_bbox
    inner = np.asarray(grid[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
    bg_mask = inner == bg
    slots = []
    for coords in _components_mask(bg_mask):
        rr0, cc0, rr1, cc1 = _bbox_from_coords(coords)
        mask = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        mask[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append(
            {
                "top": int(r0 + 1 + rr0),
                "left": int(c0 + 1 + cc0),
                "mask": mask,
                "area": int(mask.sum()),
            }
        )
    slots.sort(key=lambda d: (-d["area"], d["top"], d["left"]))
    return slots


def _extract_pieces(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    frame_bbox: tuple[int, int, int, int],
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    fr0, fc0, fr1, fc1 = frame_bbox
    pieces = []
    for color in np.unique(g):
        color = int(color)
        if color in {bg, frame_color}:
            continue
        for coords in _components_mask(g == color):
            if np.any(
                (coords[:, 0] >= fr0)
                & (coords[:, 0] <= fr1)
                & (coords[:, 1] >= fc0)
                & (coords[:, 1] <= fc1)
            ):
                continue
            pr0, pc0, pr1, pc1 = _bbox_from_coords(coords)
            mask = np.zeros((pr1 - pr0 + 1, pc1 - pc0 + 1), dtype=bool)
            mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
            solid = _fill_holes(mask)
            pieces.append(
                {
                    "color": color,
                    "top": int(pr0),
                    "left": int(pc0),
                    "mask": mask,
                    "solid": solid,
                    "solid_area": int(solid.sum()),
                }
            )
    pieces.sort(key=lambda d: (-d["solid_area"], d["color"], d["top"], d["left"]))
    return pieces


def _match_pieces_to_slots(pieces: list[dict], slots: list[dict]) -> list[dict]:
    assert len(pieces) == len(slots)
    options: list[list[dict]] = []
    for piece in pieces:
        cur_options = []
        for slot_idx, slot in enumerate(slots):
            if piece["solid_area"] != slot["area"]:
                continue
            for color_mask, solid_mask in _all_dihedral_pairs(piece["mask"], piece["solid"]):
                if solid_mask.shape != slot["mask"].shape:
                    continue
                if np.array_equal(solid_mask, slot["mask"]):
                    cur_options.append(
                        {
                            "slot_idx": int(slot_idx),
                            "color_mask": color_mask,
                            "solid_mask": solid_mask,
                        }
                    )
        assert cur_options, "every piece must match at least one slot"
        cur_options.sort(key=lambda d: (d["slot_idx"], d["color_mask"].shape, d["color_mask"].tobytes()))
        options.append(cur_options)

    order = sorted(
        range(len(pieces)),
        key=lambda i: (len({opt["slot_idx"] for opt in options[i]}), len(options[i]), -pieces[i]["solid_area"]),
    )
    used_slots: set[int] = set()
    assignment: list[Optional[dict]] = [None] * len(pieces)

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        piece_idx = order[pos]
        for opt in options[piece_idx]:
            slot_idx = int(opt["slot_idx"])
            if slot_idx in used_slots:
                continue
            used_slots.add(slot_idx)
            assignment[piece_idx] = opt
            if dfs(pos + 1):
                return True
            assignment[piece_idx] = None
            used_slots.remove(slot_idx)
        return False

    assert dfs(0), "piece-slot matching must be solvable"
    return [opt for opt in assignment if opt is not None]


def _apply_solution(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    pieces: list[dict],
    slots: list[dict],
    assignment: list[dict],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for piece in pieces:
        patch = out[piece["top"] : piece["top"] + piece["mask"].shape[0], piece["left"] : piece["left"] + piece["mask"].shape[1]]
        patch[piece["mask"]] = bg

    for piece, match in zip(pieces, assignment):
        slot = slots[int(match["slot_idx"])]
        top = int(slot["top"])
        left = int(slot["left"])
        h, w = slot["mask"].shape
        patch = out[top : top + h, left : left + w]
        color_mask = np.asarray(match["color_mask"], dtype=bool)
        solid_mask = np.asarray(match["solid_mask"], dtype=bool)
        patch[solid_mask & ~color_mask] = frame_color
        patch[color_mask] = int(piece["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move outside colored fragments into matching frame slots under rotation/flip.

    Fragments are matched to frame-interior background slots by their filled silhouette.
    Any holes inside a fragment become frame-color pixels after insertion.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    candidates = _frame_candidates(g, bg)
    assert candidates, "frame not found"
    frame = candidates[0]
    frame_color = int(frame["color"])
    frame_bbox = tuple(frame["bbox"])

    slots = _extract_slots(g, bg, frame_bbox)
    pieces = _extract_pieces(g, bg, frame_color, frame_bbox)
    assert slots and pieces and len(slots) == len(pieces), "slot/piece count mismatch"

    assignment = _match_pieces_to_slots(pieces, slots)
    return _apply_solution(g, bg, frame_color, pieces, slots, assignment)


def _mask(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=bool)


SHAPE_LIBRARY = [
    _mask([[1, 1], [1, 0]]),
    _mask([[1, 1, 1], [1, 0, 0]]),
    _mask([[1, 1, 1], [0, 1, 0]]),
    _mask([[1, 1, 0], [0, 1, 1]]),
    _mask([[1, 1, 1], [1, 1, 0]]),
    _mask([[1, 1, 1], [0, 1, 1]]),
    _mask([[1, 1, 1], [1, 0, 1]]),
    _mask([[1, 1, 1], [1, 0, 0], [1, 0, 0]]),
    _mask([[1, 1, 1], [0, 1, 0], [0, 1, 0]]),
    _mask([[1, 1, 0], [0, 1, 0], [0, 1, 1]]),
    _mask([[1, 1, 1], [1, 0, 0], [1, 1, 0]]),
    _mask([[1, 1, 1], [0, 1, 1], [0, 0, 1]]),
    _mask([[1, 1, 1, 1], [1, 0, 0, 0]]),
    _mask([[1, 1, 1, 1], [0, 1, 0, 0]]),
    _mask([[1, 1, 1, 0], [0, 1, 1, 1]]),
]

HOLE_SLOT_LIBRARY = [
    _mask([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
    _mask([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]),
    _mask([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]),
    _mask([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]),
]

MAX_GENERATION_ATTEMPTS = 400


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components_mask(mask)) == 1


def _choose_slot_masks(rng: random.Random, n_slots: int, use_hole_piece: bool) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    used = set()

    if use_hole_piece:
        hole_slot = rng.choice(HOLE_SLOT_LIBRARY).copy()
        masks.append(hole_slot)
        used.add(_canonical_mask(hole_slot))

    library = SHAPE_LIBRARY[:]
    rng.shuffle(library)
    for base in library:
        if len(masks) >= n_slots:
            break
        key = _canonical_mask(base)
        if key in used:
            continue
        variant = rng.choice(_all_dihedral_masks(base))
        masks.append(variant.copy())
        used.add(key)

    if len(masks) < n_slots:
        raise AssertionError("shape library exhausted")

    if all(len(_all_dihedral_masks(mask)) == 1 for mask in masks):
        raise AssertionError("need at least one orientation-sensitive slot")

    rng.shuffle(masks)
    return masks


def _can_place_slot(existing_bg: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = existing_bg.shape
    mh, mw = mask.shape
    if top < 0 or left < 0 or top + mh > h or left + mw > w:
        return False
    for r in range(mh):
        for c in range(mw):
            if not mask[r, c]:
                continue
            rr = top + r
            cc = left + c
            if existing_bg[rr, cc]:
                return False
            for dr, dc in DIR4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and existing_bg[nr, nc]:
                    return False
    return True


def _layout_slots(rng: random.Random, slot_masks: list[np.ndarray]) -> Optional[tuple[int, int, list[dict]]]:
    total_area = sum(int(mask.sum()) for mask in slot_masks)
    max_h = max(mask.shape[0] for mask in slot_masks)
    max_w = max(mask.shape[1] for mask in slot_masks)

    for _ in range(80):
        inner_h = rng.randint(max(8, max_h + 1), 15)
        inner_w = rng.randint(max(8, max_w + 1), 15)
        if inner_h * inner_w < total_area * 2:
            continue

        existing_bg = np.zeros((inner_h, inner_w), dtype=bool)
        records: list[Optional[dict]] = [None] * len(slot_masks)
        order = sorted(range(len(slot_masks)), key=lambda i: (-int(slot_masks[i].sum()), slot_masks[i].shape))
        ok = True

        for idx in order:
            mask = slot_masks[idx]
            mh, mw = mask.shape
            positions = [(r, c) for r in range(inner_h - mh + 1) for c in range(inner_w - mw + 1)]
            rng.shuffle(positions)
            placed = False
            for top, left in positions:
                if not _can_place_slot(existing_bg, mask, top, left):
                    continue
                sub = existing_bg[top : top + mh, left : left + mw]
                sub[mask] = True
                records[idx] = {"top": int(top), "left": int(left), "mask": mask.copy(), "area": int(mask.sum())}
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        walls = np.ones((inner_h + 2, inner_w + 2), dtype=bool)
        wall_inner = walls[1:-1, 1:-1]
        wall_inner[existing_bg] = False
        if len(_components_mask(walls)) != 1:
            continue

        return inner_h, inner_w, [rec for rec in records if rec is not None]
    return None


def _make_center_hole_piece(solid: np.ndarray) -> np.ndarray:
    piece = solid.copy()
    h, w = piece.shape
    if h % 2 == 1 and w % 2 == 1:
        piece[h // 2, w // 2] = False
    elif h % 2 == 0 and w % 2 == 0:
        piece[h // 2 - 1 : h // 2 + 1, w // 2 - 1 : w // 2 + 1] = False
    elif h % 2 == 0:
        piece[h // 2 - 1 : h // 2 + 1, w // 2] = False
    else:
        piece[h // 2, w // 2 - 1 : w // 2 + 1] = False
    assert _is_connected(piece)
    assert np.array_equal(_fill_holes(piece), solid)
    return piece


def _piece_variants_are_unambiguous(input_color: np.ndarray, input_solid: np.ndarray, slot_mask: np.ndarray) -> bool:
    matches = set()
    for color_mask, solid_mask in _all_dihedral_pairs(input_color, input_solid):
        if solid_mask.shape == slot_mask.shape and np.array_equal(solid_mask, slot_mask):
            matches.add((color_mask.shape, color_mask.tobytes()))
    return len(matches) == 1


def _make_piece_specs(rng: random.Random, slots: list[dict], colors: list[int], use_hole_piece: bool) -> list[dict]:
    hole_indices = []
    if use_hole_piece:
        eligible = [i for i, slot in enumerate(slots) if min(slot["mask"].shape) >= 3 and np.all(slot["mask"])]
        if not eligible:
            raise AssertionError("no hole-capable slot")
        hole_indices = [rng.choice(eligible)]

    specs = []
    for idx, (slot, color) in enumerate(zip(slots, colors)):
        solid = slot["mask"].copy()
        slot_color_mask = _make_center_hole_piece(solid) if idx in hole_indices else solid.copy()
        variants = _all_dihedral_pairs(slot_color_mask, solid)
        input_color, input_solid = rng.choice(variants)
        if not _piece_variants_are_unambiguous(input_color, input_solid, solid):
            raise AssertionError("ambiguous transformed piece")
        specs.append(
            {
                "color": int(color),
                "slot_top": int(slot["top"]),
                "slot_left": int(slot["left"]),
                "slot_mask": solid,
                "slot_color_mask": slot_color_mask,
                "input_color_mask": input_color,
                "input_solid_mask": input_solid,
            }
        )
    return specs


def _can_place_patch(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    H, W = occupied.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(H, top + h + 1)
    c1 = min(W, left + w + 1)
    return not bool(occupied[r0:r1, c0:c1].any())


def _build_expected_output(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    frame_top: int,
    frame_left: int,
    piece_specs: list[dict],
) -> np.ndarray:
    out = grid.copy()
    for spec in piece_specs:
        itop = int(spec["input_top"])
        ileft = int(spec["input_left"])
        mask = np.asarray(spec["input_color_mask"], dtype=bool)
        out[itop : itop + mask.shape[0], ileft : ileft + mask.shape[1]][mask] = bg

    for spec in piece_specs:
        top = int(frame_top + 1 + spec["slot_top"])
        left = int(frame_left + 1 + spec["slot_left"])
        solid = np.asarray(spec["slot_mask"], dtype=bool)
        color_mask = np.asarray(spec["slot_color_mask"], dtype=bool)
        patch = out[top : top + solid.shape[0], left : left + solid.shape[1]]
        patch[solid & ~color_mask] = frame_color
        patch[color_mask] = int(spec["color"])
    return out


def _gen_attempt() -> Optional[np.ndarray]:
    rng = random.Random()
    bg = rng.randrange(10)
    remaining = [c for c in range(10) if c != bg]
    frame_color = rng.choice(remaining)
    remaining = [c for c in remaining if c != frame_color]

    n_slots = rng.randint(3, 5)
    use_hole_piece = rng.random() < 0.5
    slot_masks = _choose_slot_masks(rng, n_slots, use_hole_piece)
    layout = _layout_slots(rng, slot_masks)
    if layout is None:
        return None
    inner_h, inner_w, slots = layout

    frame_h = inner_h + 2
    frame_w = inner_w + 2
    grid_h = rng.randint(max(20, frame_h + 5), 29)
    grid_w = rng.randint(max(20, frame_w + 5), 29)
    if grid_h * grid_w - frame_h * frame_w < sum(int(slot["mask"].size) for slot in slots) * 2:
        return None

    frame_top = rng.randint(1, grid_h - frame_h - 1)
    frame_left = rng.randint(1, grid_w - frame_w - 1)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    frame = grid[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w]
    frame[:, :] = frame_color
    inner = frame[1:-1, 1:-1]
    inner[:, :] = frame_color
    for slot in slots:
        sub = inner[slot["top"] : slot["top"] + slot["mask"].shape[0], slot["left"] : slot["left"] + slot["mask"].shape[1]]
        sub[slot["mask"]] = bg

    piece_colors = rng.sample(remaining, n_slots)
    piece_specs = _make_piece_specs(rng, slots, piece_colors, use_hole_piece)

    occupied = np.zeros_like(grid, dtype=bool)
    r0 = max(0, frame_top - 1)
    c0 = max(0, frame_left - 1)
    r1 = min(grid_h, frame_top + frame_h + 1)
    c1 = min(grid_w, frame_left + frame_w + 1)
    occupied[r0:r1, c0:c1] = True

    for spec in piece_specs:
        mask = np.asarray(spec["input_color_mask"], dtype=bool)
        ph, pw = mask.shape
        positions = [(r, c) for r in range(grid_h - ph + 1) for c in range(grid_w - pw + 1)]
        rng.shuffle(positions)
        placed = False
        for top, left in positions:
            if not _can_place_patch(occupied, top, left, ph, pw):
                continue
            grid[top : top + ph, left : left + pw][mask] = int(spec["color"])
            rr0 = max(0, top - 1)
            cc0 = max(0, left - 1)
            rr1 = min(grid_h, top + ph + 1)
            cc1 = min(grid_w, left + pw + 1)
            occupied[rr0:rr1, cc0:cc1] = True
            spec["input_top"] = int(top)
            spec["input_left"] = int(left)
            placed = True
            break
        if not placed:
            return None

    counts = np.unique(grid, return_counts=True)
    count_map = {int(v): int(c) for v, c in zip(*counts)}
    if count_map.get(bg, 0) <= max(count for color, count in count_map.items() if color != bg):
        return None

    if len(_frame_candidates(grid, bg)) != 1:
        return None

    expected = _build_expected_output(grid, bg, frame_color, frame_top, frame_left, piece_specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
