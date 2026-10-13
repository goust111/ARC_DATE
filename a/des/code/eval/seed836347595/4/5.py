from __future__ import annotations

import math
import random

import numpy as np


BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]

FRAME_COUNT_CHOICES = (2, 3)
PIECE_COUNT_CHOICES = (2, 3)
PIECE_AREA_VALUES = (2, 3, 4, 5, 6, 7)

GROUP_RETRIES = 160
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    assert rows.any() and cols.any()
    return mask[np.ix_(rows, cols)]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), np.asarray(mask, dtype=bool).tobytes()


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    rots: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = _crop_mask(np.rot90(mask, k))
        key = _mask_key(rot)
        if key not in seen:
            seen.add(key)
            rots.append(rot)
    return rots


def _has_internal_hole(mask: np.ndarray) -> bool:
    bg = ~mask
    h, w = bg.shape
    for comp in _components(bg):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        return True
    return False


def _enclosed_background(subgrid: np.ndarray) -> np.ndarray:
    bg_mask = subgrid == BG
    h, w = bg_mask.shape
    cavity = np.zeros((h, w), dtype=bool)
    for comp in _components(bg_mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        for r, c in comp:
            cavity[r, c] = True
    return cavity


def _extract_color_components(grid: np.ndarray, color: int) -> list[dict]:
    comps: list[dict] = []
    for idx, coords in enumerate(_components(grid == color)):
        r0, c0, r1, c1 = _bbox(coords)
        comps.append(
            {
                "id": idx,
                "bbox": (r0, c0, r1, c1),
                "mask": _mask_from_coords(coords),
                "area": len(coords),
            }
        )
    return comps


def _is_frame_component(grid: np.ndarray, color: int, comp: dict) -> tuple[bool, np.ndarray]:
    r0, c0, r1, c1 = comp["bbox"]
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border_ok = (
        np.all(sub[0, :] == color)
        and np.all(sub[-1, :] == color)
        and np.all(sub[:, 0] == color)
        and np.all(sub[:, -1] == color)
    )
    if not border_ok:
        return False, np.zeros_like(sub, dtype=bool)
    cavity = _enclosed_background(sub)
    return bool(cavity.any()), cavity


def _best_fit(remaining: np.ndarray, piece_mask: np.ndarray) -> dict | None:
    h, w = remaining.shape
    best: dict | None = None
    for rot_idx, rot in enumerate(_unique_rotations(piece_mask)):
        ph, pw = rot.shape
        if ph > h or pw > w:
            continue
        for r in range(h - ph + 1):
            for c in range(w - pw + 1):
                window = remaining[r : r + ph, c : c + pw]
                if np.all(window[rot]):
                    cand = {"r": r, "c": c, "rot_idx": rot_idx, "mask": rot}
                    if best is None or (r, c, rot_idx) < (best["r"], best["c"], best["rot_idx"]):
                        best = cand
    return best


def _pack_pieces_into_cavity(cavity: np.ndarray, piece_infos: list[dict]) -> list[dict] | None:
    remaining = cavity.copy()
    ordered = sorted(
        piece_infos,
        key=lambda info: (-int(info["area"]), int(info.get("sort_r", 0)), int(info.get("sort_c", 0)), int(info["id"])),
    )
    placements: list[dict] = []
    for info in ordered:
        best = _best_fit(remaining, info["mask"])
        if best is None:
            return None
        ph, pw = best["mask"].shape
        window = remaining[best["r"] : best["r"] + ph, best["c"] : best["c"] + pw].copy()
        window[best["mask"]] = False
        remaining[best["r"] : best["r"] + ph, best["c"] : best["c"] + pw] = window
        placements.append(
            {
                "id": info["id"],
                "r": best["r"],
                "c": best["c"],
                "mask": best["mask"],
            }
        )
    if remaining.any():
        return None
    return placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered monochrome fragments by color, rotate them, and place them into
    same-colored frame cavities from largest area to smallest.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        comps = _extract_color_components(g, color)
        frame_idx = None
        cavity = None
        for idx, comp in enumerate(comps):
            is_frame, cand_cavity = _is_frame_component(g, color, comp)
            if is_frame:
                assert frame_idx is None
                frame_idx = idx
                cavity = cand_cavity
        assert frame_idx is not None and cavity is not None

        frame = comps[frame_idx]
        pieces = []
        for idx, comp in enumerate(comps):
            if idx == frame_idx:
                continue
            pieces.append(
                {
                    "id": comp["id"],
                    "bbox": comp["bbox"],
                    "mask": comp["mask"],
                    "area": comp["area"],
                    "sort_r": comp["bbox"][0],
                    "sort_c": comp["bbox"][1],
                }
            )
        assert pieces
        assert _pack_pieces_into_cavity(cavity, pieces) is not None

        for piece in pieces:
            r0, c0, r1, c1 = piece["bbox"]
            patch = out[r0 : r1 + 1, c0 : c1 + 1].copy()
            patch[piece["mask"]] = BG
            out[r0 : r1 + 1, c0 : c1 + 1] = patch

        r0, c0, r1, c1 = frame["bbox"]
        frame_patch = out[r0 : r1 + 1, c0 : c1 + 1].copy()
        frame_patch[cavity] = color
        out[r0 : r1 + 1, c0 : c1 + 1] = frame_patch

    return out


def _random_polyomino(area: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(120):
        cells = {(0, 0)}
        frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        while len(cells) < area and frontier:
            nr, nc = rng.choice(tuple(frontier))
            frontier.remove((nr, nc))
            cells.add((nr, nc))
            nbrs = [(nr + 1, nc), (nr - 1, nc), (nr, nc + 1), (nr, nc - 1)]
            rng.shuffle(nbrs)
            for rr, cc in nbrs:
                if (rr, cc) not in cells:
                    frontier.add((rr, cc))
        if len(cells) != area:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        rmin, cmin = min(rs), min(cs)
        norm = [(r - rmin, c - cmin) for r, c in cells]
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if h > 4 or w > 4:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[r, c] = True
        if _has_internal_hole(mask):
            continue
        return mask
    return None


def _touches_union(occ: np.ndarray, piece: np.ndarray, top: int, left: int) -> bool:
    ph, pw = piece.shape
    for r in range(ph):
        for c in range(pw):
            if not piece[r, c]:
                continue
            gr = top + r
            gc = left + c
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = gr + dr
                nc = gc + dc
                if 0 <= nr < occ.shape[0] and 0 <= nc < occ.shape[1] and occ[nr, nc]:
                    return True
    return False


def _make_group(color: int, rng: random.Random) -> dict | None:
    for _ in range(GROUP_RETRIES):
        piece_count = rng.choice(PIECE_COUNT_CHOICES)
        areas = sorted(rng.sample(PIECE_AREA_VALUES, piece_count), reverse=True)
        piece_masks = []
        for area in areas:
            mask = _random_polyomino(area, rng)
            if mask is None:
                break
            piece_masks.append(mask)
        if len(piece_masks) != piece_count:
            continue

        total_area = sum(int(mask.sum()) for mask in piece_masks)
        max_h = max(mask.shape[0] for mask in piece_masks)
        max_w = max(mask.shape[1] for mask in piece_masks)
        cavity = None
        placements = None

        for _ in range(80):
            base_side = max(int(math.ceil(math.sqrt(total_area))), max_h, max_w)
            canvas_h = rng.randint(base_side, base_side + 2)
            canvas_w = rng.randint(base_side, base_side + 3)
            if canvas_h * canvas_w <= total_area:
                continue
            occ = np.zeros((canvas_h, canvas_w), dtype=bool)

            placed_ok = True
            for idx, mask in enumerate(piece_masks):
                candidates: list[tuple[int, int, np.ndarray]] = []
                for rot in _unique_rotations(mask):
                    ph, pw = rot.shape
                    if ph > canvas_h or pw > canvas_w:
                        continue
                    for r in range(canvas_h - ph + 1):
                        for c in range(canvas_w - pw + 1):
                            window = occ[r : r + ph, c : c + pw]
                            if np.any(window & rot):
                                continue
                            if idx > 0 and not _touches_union(occ, rot, r, c):
                                continue
                            candidates.append((r, c, rot))
                if not candidates:
                    placed_ok = False
                    break
                r, c, rot = rng.choice(candidates)
                ph, pw = rot.shape
                window = occ[r : r + ph, c : c + pw].copy()
                window[rot] = True
                occ[r : r + ph, c : c + pw] = window

            if not placed_ok:
                continue

            cavity = _crop_mask(occ)
            if cavity.shape[0] < 2 or cavity.shape[1] < 2:
                continue
            if int(cavity.sum()) != total_area:
                continue
            if int(cavity.sum()) == cavity.shape[0] * cavity.shape[1]:
                continue

            piece_infos = [{"id": idx, "mask": mask, "area": int(mask.sum())} for idx, mask in enumerate(piece_masks)]
            placements = _pack_pieces_into_cavity(cavity, piece_infos)
            if placements is None:
                continue
            break

        if cavity is None or placements is None:
            continue

        if not any(len(_unique_rotations(mask)) > 1 for mask in piece_masks):
            continue

        margin_top = rng.randint(0, 2)
        margin_bottom = rng.randint(0, 2)
        margin_left = rng.randint(0, 2)
        margin_right = rng.randint(0, 2)
        if margin_top + margin_bottom + margin_left + margin_right == 0:
            margin_bottom = 1

        inner_h = cavity.shape[0] + margin_top + margin_bottom
        inner_w = cavity.shape[1] + margin_left + margin_right
        frame_patch = np.full((inner_h + 2, inner_w + 2), color, dtype=int)
        cavity_in_frame = np.zeros_like(frame_patch, dtype=bool)
        r0 = 1 + margin_top
        c0 = 1 + margin_left
        cavity_in_frame[r0 : r0 + cavity.shape[0], c0 : c0 + cavity.shape[1]] = cavity
        frame_patch[cavity_in_frame] = BG

        placed_by_id = {entry["id"]: entry for entry in placements}
        pieces = []
        for idx, mask in enumerate(piece_masks):
            pieces.append(
                {
                    "id": idx,
                    "color": color,
                    "base_mask": mask,
                    "placed_mask": placed_by_id[idx]["mask"],
                    "area": int(mask.sum()),
                }
            )

        return {
            "color": color,
            "frame_patch": frame_patch,
            "cavity_mask": cavity_in_frame,
            "pieces": pieces,
        }
    return None


def _random_free_position(
    reserved: np.ndarray, obj_h: int, obj_w: int, rng: random.Random, pad: int = 1
) -> tuple[int, int] | None:
    h, w = reserved.shape
    positions: list[tuple[int, int]] = []
    for r in range(1, h - obj_h):
        for c in range(1, w - obj_w):
            rr0 = max(0, r - pad)
            rr1 = min(h, r + obj_h + pad)
            cc0 = max(0, c - pad)
            cc1 = min(w, c + obj_w + pad)
            if not reserved[rr0:rr1, cc0:cc1].any():
                positions.append((r, c))
    if not positions:
        return None
    return rng.choice(positions)


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    sub = grid[top : top + h, left : left + w].copy()
    mask = patch != BG
    sub[mask] = patch[mask]
    grid[top : top + h, left : left + w] = sub


def _mark_reserved(reserved: np.ndarray, top: int, left: int, obj_h: int, obj_w: int, pad: int = 1) -> None:
    h, w = reserved.shape
    reserved[max(0, top - pad) : min(h, top + obj_h + pad), max(0, left - pad) : min(w, left + obj_w + pad)] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.choice(FRAME_COUNT_CHOICES)
    colors = rng.sample(COLOR_POOL, frame_count)
    groups = []
    for color in colors:
        group = _make_group(color, rng)
        if group is None:
            return None
        groups.append(group)

    rotatable_pieces = []
    for group in groups:
        for piece in group["pieces"]:
            placed_key = _mask_key(piece["placed_mask"])
            alt_masks = [rot for rot in _unique_rotations(piece["base_mask"]) if _mask_key(rot) != placed_key]
            if alt_masks:
                rotatable_pieces.append((group, piece, alt_masks))
    if not rotatable_pieces:
        return None
    forced_group, forced_piece, forced_alts = rng.choice(rotatable_pieces)

    piece_patches = []
    for group in groups:
        for piece in group["pieces"]:
            if group is forced_group and piece["id"] == forced_piece["id"]:
                display_mask = rng.choice(forced_alts)
            else:
                display_mask = rng.choice(_unique_rotations(piece["base_mask"]))
            piece["display_mask"] = display_mask
            patch = np.zeros(display_mask.shape, dtype=int)
            patch[display_mask] = group["color"]
            piece_patches.append({"group": group, "piece": piece, "patch": patch})

    object_area = 0
    max_h = 0
    max_w = 0
    for group in groups:
        h, w = group["frame_patch"].shape
        object_area += (h + 2) * (w + 2)
        max_h = max(max_h, h)
        max_w = max(max_w, w)
    for entry in piece_patches:
        h, w = entry["patch"].shape
        object_area += (h + 2) * (w + 2)
        max_h = max(max_h, h)
        max_w = max(max_w, w)

    side = int(math.ceil(math.sqrt(object_area * 1.25)))
    h_lo = max(max_h + 2, side - 2)
    h_hi = min(30, h_lo + 4)
    w_lo = max(max_w + 2, side - 1)
    w_hi = min(30, w_lo + 5)
    if h_lo > 30 or w_lo > 30 or h_lo > h_hi or w_lo > w_hi:
        return None

    grid = np.zeros((rng.randint(h_lo, h_hi), rng.randint(w_lo, w_hi)), dtype=int)
    reserved = np.zeros_like(grid, dtype=bool)
    expected = np.zeros_like(grid)

    group_positions: dict[int, tuple[int, int]] = {}
    for group in sorted(groups, key=lambda grp: grp["frame_patch"].shape[0] * grp["frame_patch"].shape[1], reverse=True):
        patch = group["frame_patch"]
        pos = _random_free_position(reserved, patch.shape[0], patch.shape[1], rng)
        if pos is None:
            return None
        top, left = pos
        group_positions[group["color"]] = (top, left)
        _paint_patch(grid, patch, top, left)
        _paint_patch(expected, patch, top, left)
        _mark_reserved(reserved, top, left, patch.shape[0], patch.shape[1])

        cavity_abs = np.zeros_like(grid, dtype=bool)
        cavity_abs[top : top + patch.shape[0], left : left + patch.shape[1]] = group["cavity_mask"]
        expected[cavity_abs] = group["color"]

    for entry in sorted(piece_patches, key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True):
        patch = entry["patch"]
        pos = _random_free_position(reserved, patch.shape[0], patch.shape[1], rng)
        if pos is None:
            return None
        top, left = pos
        _paint_patch(grid, patch, top, left)
        _mark_reserved(reserved, top, left, patch.shape[0], patch.shape[1])

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
