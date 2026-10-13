from __future__ import annotations

import random
from collections import Counter

import numpy as np


SIDE_ORDER = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "right": "left", "bottom": "top", "left": "right"}
ROTATIONS = (0, 1, 2, 3)
BG = 0
LAYOUT_CHOICES = ("row2", "row3", "row4", "col2", "col3", "col4", "grid")
SIZE_MIN = 4
SIZE_MAX = 6
GEN_RETRIES = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _most_common(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    return int(counts.most_common(1)[0][0])


def _frame_touches_corner(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return (
        (r0 == 0 and c0 == 0)
        or (r0 == 0 and c1 == w - 1)
        or (r1 == h - 1 and c0 == 0)
        or (r1 == h - 1 and c1 == w - 1)
    )


def _is_corner_frame(patch: np.ndarray, bbox: tuple[int, int, int, int], shape: tuple[int, int], bg: int) -> bool:
    if not _frame_touches_corner(bbox, shape):
        return False
    colors = set(map(int, patch[patch != bg].reshape(-1)))
    if len(colors) != 1:
        return False
    h, w = patch.shape
    if h < 4 or w < 4:
        return False
    mask = patch != bg
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(mask, border)


def _side_signature(feature: tuple | None) -> tuple:
    if feature is None:
        return ("outer",)
    return tuple(feature)


def _parse_piece(patch: np.ndarray, bg: int) -> dict | None:
    vals = patch[patch != bg]
    if vals.size == 0:
        return None
    counts = Counter(map(int, vals))
    top_count = max(counts.values())
    winners = [color for color, cnt in counts.items() if cnt == top_count]
    if len(winners) != 1:
        return None
    body_color = int(winners[0])
    body_mask = patch == body_color
    box = _bbox(body_mask)
    if box is None:
        return None
    br0, bc0, br1, bc1 = box
    h = int(br1 - br0 + 1)
    w = int(bc1 - bc0 + 1)
    if h < SIZE_MIN or w < SIZE_MIN:
        return None

    sub = patch[br0 : br1 + 1, bc0 : bc1 + 1]
    inner = sub[1:-1, 1:-1]
    if inner.size and np.any(inner != body_color):
        return None
    if np.any((sub != bg) & (sub != body_color)):
        return None

    sides: dict[str, tuple | None] = {}
    accounted = np.zeros_like(patch, dtype=bool)
    accounted[br0 : br1 + 1, bc0 : bc1 + 1] = sub == body_color

    def tooth_feature(side: str) -> tuple | None:
        if side == "top":
            if br0 == 0:
                return None
            cols = np.flatnonzero(patch[br0 - 1, bc0 : bc1 + 1] != bg)
            if len(cols) > 1:
                return None
            if len(cols) == 1:
                off = int(cols[0])
                accounted[br0 - 1, bc0 + off] = True
                return ("tooth", off, int(patch[br0 - 1, bc0 + off]))
        elif side == "bottom":
            if br1 == patch.shape[0] - 1:
                return None
            cols = np.flatnonzero(patch[br1 + 1, bc0 : bc1 + 1] != bg)
            if len(cols) > 1:
                return None
            if len(cols) == 1:
                off = int(cols[0])
                accounted[br1 + 1, bc0 + off] = True
                return ("tooth", off, int(patch[br1 + 1, bc0 + off]))
        elif side == "left":
            if bc0 == 0:
                return None
            rows = np.flatnonzero(patch[br0 : br1 + 1, bc0 - 1] != bg)
            if len(rows) > 1:
                return None
            if len(rows) == 1:
                off = int(rows[0])
                accounted[br0 + off, bc0 - 1] = True
                return ("tooth", off, int(patch[br0 + off, bc0 - 1]))
        else:
            if bc1 == patch.shape[1] - 1:
                return None
            rows = np.flatnonzero(patch[br0 : br1 + 1, bc1 + 1] != bg)
            if len(rows) > 1:
                return None
            if len(rows) == 1:
                off = int(rows[0])
                accounted[br0 + off, bc1 + 1] = True
                return ("tooth", off, int(patch[br0 + off, bc1 + 1]))
        return None

    def slot_feature(side: str) -> tuple | None:
        if side == "top":
            offs = np.flatnonzero(sub[0, :] == bg)
        elif side == "bottom":
            offs = np.flatnonzero(sub[-1, :] == bg)
        elif side == "left":
            offs = np.flatnonzero(sub[:, 0] == bg)
        else:
            offs = np.flatnonzero(sub[:, -1] == bg)
        if len(offs) > 1:
            return None
        if len(offs) == 1:
            return ("slot", int(offs[0]))
        return None

    for side in SIDE_ORDER:
        tooth = tooth_feature(side)
        slot = slot_feature(side)
        if tooth is not None and slot is not None:
            return None
        sides[side] = tooth if tooth is not None else slot

    if np.any((patch != bg) & ~accounted):
        return None

    return {"color": body_color, "h": h, "w": w, "sides": sides}


def _piece_variants(patch: np.ndarray, bg: int) -> list[dict]:
    variants: list[dict] = []
    seen: set[tuple] = set()
    for k in ROTATIONS:
        rotated = np.rot90(patch, k)
        meta = _parse_piece(rotated, bg)
        if meta is None:
            continue
        sig = (
            int(meta["color"]),
            int(meta["h"]),
            int(meta["w"]),
            tuple(_side_signature(meta["sides"][side]) for side in SIDE_ORDER),
        )
        if sig in seen:
            continue
        seen.add(sig)
        variants.append(meta)
    return variants


def _match_features(a: dict, side_a: str, b: dict, side_b: str) -> bool:
    fa = a["sides"][side_a]
    fb = b["sides"][side_b]
    if fa is None or fb is None:
        return False
    if fa[0] == "tooth" and fb[0] == "slot":
        return int(fa[1]) == int(fb[1]) and int(fa[2]) == int(b["color"])
    if fa[0] == "slot" and fb[0] == "tooth":
        return int(fa[1]) == int(fb[1]) and int(fb[2]) == int(a["color"])
    return False


def _first_empty(board: np.ndarray) -> tuple[int, int] | None:
    coords = np.argwhere(board < 0)
    if coords.size == 0:
        return None
    return int(coords[0, 0]), int(coords[0, 1])


def _can_place(
    board: np.ndarray,
    placements: dict[int, dict],
    meta: dict,
    piece_id: int,
    r: int,
    c: int,
    target_h: int,
    target_w: int,
) -> bool:
    h = int(meta["h"])
    w = int(meta["w"])
    if r + h > target_h or c + w > target_w:
        return False
    if np.any(board[r : r + h, c : c + w] >= 0):
        return False

    at = {
        "top": r == 0,
        "bottom": r + h == target_h,
        "left": c == 0,
        "right": c + w == target_w,
    }
    for side in SIDE_ORDER:
        feat = meta["sides"][side]
        if feat is None and not at[side]:
            return False
        if feat is not None and at[side]:
            return False

    if r > 0:
        above = board[r - 1, c : c + w]
        if np.any(above < 0):
            return False
        if len(set(map(int, above))) != 1:
            return False
        neighbor_id = int(above[0])
        neighbor = placements[neighbor_id]
        nmeta = neighbor["meta"]
        if int(neighbor["r"]) + int(nmeta["h"]) != r or int(neighbor["c"]) != c or int(nmeta["w"]) != w:
            return False
        if not _match_features(nmeta, "bottom", meta, "top"):
            return False

    if c > 0:
        left = board[r : r + h, c - 1]
        if np.any(left < 0):
            return False
        if len(set(map(int, left))) != 1:
            return False
        neighbor_id = int(left[0])
        neighbor = placements[neighbor_id]
        nmeta = neighbor["meta"]
        if int(neighbor["c"]) + int(nmeta["w"]) != c or int(neighbor["r"]) != r or int(nmeta["h"]) != h:
            return False
        if not _match_features(nmeta, "right", meta, "left"):
            return False

    return True


def _solve_layout(target_h: int, target_w: int, variant_sets: list[list[dict]]) -> list[dict] | None:
    n = len(variant_sets)
    board = np.full((target_h, target_w), -1, dtype=int)
    used = [False] * n
    placements: dict[int, dict] = {}

    def backtrack() -> bool:
        pos = _first_empty(board)
        if pos is None:
            return all(used)
        r, c = pos
        for piece_id in range(n):
            if used[piece_id]:
                continue
            for meta in variant_sets[piece_id]:
                if not _can_place(board, placements, meta, piece_id, r, c, target_h, target_w):
                    continue
                h = int(meta["h"])
                w = int(meta["w"])
                board[r : r + h, c : c + w] = piece_id
                placements[piece_id] = {"piece_id": piece_id, "meta": meta, "r": r, "c": c}
                used[piece_id] = True
                if backtrack():
                    return True
                used[piece_id] = False
                del placements[piece_id]
                board[r : r + h, c : c + w] = -1
        return False

    if not backtrack():
        return None
    return [placements[i] for i in range(n)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered rectangular jigsaw pieces by matching tooth colors to target body colors
    and matching single-cell teeth to single-cell slots on shared edges.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(_most_common(g))

    objects = []
    for coords in _components(g != bg):
        rs = coords[:, 0]
        cs = coords[:, 1]
        r0, c0 = int(rs.min()), int(cs.min())
        r1, c1 = int(rs.max()), int(cs.max())
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        objects.append({"bbox": (r0, c0, r1, c1), "patch": patch})

    frame_candidates = [
        obj for obj in objects if _is_corner_frame(obj["patch"], obj["bbox"], g.shape, bg)
    ]
    assert len(frame_candidates) == 1, "expected exactly one corner size frame"
    frame = frame_candidates[0]
    target_h, target_w = frame["patch"].shape

    piece_patches = [obj["patch"] for obj in objects if obj is not frame]
    assert piece_patches, "expected at least one jigsaw piece"
    variant_sets = [_piece_variants(patch, bg) for patch in piece_patches]
    assert all(variant_sets), "every piece must admit at least one valid orientation"

    placements = _solve_layout(int(target_h), int(target_w), variant_sets)
    assert placements is not None, "pieces must tile the target rectangle"

    out = np.zeros((int(target_h), int(target_w)), dtype=int)
    for placed in placements:
        meta = placed["meta"]
        r = int(placed["r"])
        c = int(placed["c"])
        h = int(meta["h"])
        w = int(meta["w"])
        out[r : r + h, c : c + w] = int(meta["color"])
    return out


def _build_piece_patch(piece: dict) -> np.ndarray:
    h = int(piece["h"])
    w = int(piece["w"])
    sides = piece["sides"]
    top_extra = 1 if sides["top"] is not None and sides["top"][0] == "tooth" else 0
    bottom_extra = 1 if sides["bottom"] is not None and sides["bottom"][0] == "tooth" else 0
    left_extra = 1 if sides["left"] is not None and sides["left"][0] == "tooth" else 0
    right_extra = 1 if sides["right"] is not None and sides["right"][0] == "tooth" else 0

    patch = np.full((h + top_extra + bottom_extra, w + left_extra + right_extra), BG, dtype=int)
    br0 = top_extra
    bc0 = left_extra
    patch[br0 : br0 + h, bc0 : bc0 + w] = int(piece["color"])

    for side, feature in sides.items():
        if feature is None:
            continue
        kind = feature[0]
        off = int(feature[1])
        if kind == "slot":
            if side == "top":
                patch[br0, bc0 + off] = BG
            elif side == "bottom":
                patch[br0 + h - 1, bc0 + off] = BG
            elif side == "left":
                patch[br0 + off, bc0] = BG
            else:
                patch[br0 + off, bc0 + w - 1] = BG
        else:
            tooth_color = int(feature[2])
            if side == "top":
                patch[br0 - 1, bc0 + off] = tooth_color
            elif side == "bottom":
                patch[br0 + h, bc0 + off] = tooth_color
            elif side == "left":
                patch[br0 + off, bc0 - 1] = tooth_color
            else:
                patch[br0 + off, bc0 + w] = tooth_color
    return patch


def _draw_frame(canvas: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    canvas[r0, c0 : c1 + 1] = color
    canvas[r1, c0 : c1 + 1] = color
    canvas[r0 : r1 + 1, c0] = color
    canvas[r0 : r1 + 1, c1] = color


def _sample_layout() -> tuple[list[dict], np.ndarray]:
    kind = random.choice(LAYOUT_CHOICES)
    if kind.startswith("row"):
        rows = 1
        cols = int(kind[-1])
    elif kind.startswith("col"):
        rows = int(kind[-1])
        cols = 1
    else:
        rows = 2
        cols = 2

    row_heights = [random.randint(SIZE_MIN, SIZE_MAX) for _ in range(rows)]
    col_widths = [random.randint(SIZE_MIN, SIZE_MAX) for _ in range(cols)]

    total_h = sum(row_heights)
    total_w = sum(col_widths)
    output = np.zeros((total_h, total_w), dtype=int)

    piece_count = rows * cols
    palette = random.sample(range(1, 10), piece_count + 1)
    piece_colors = palette[:piece_count]

    row_starts = [sum(row_heights[:r]) for r in range(rows)]
    col_starts = [sum(col_widths[:c]) for c in range(cols)]
    pieces: list[dict] = []
    index_at: dict[tuple[int, int], int] = {}
    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            h = int(row_heights[rr])
            w = int(col_widths[cc])
            color = int(piece_colors[idx])
            r0 = int(row_starts[rr])
            c0 = int(col_starts[cc])
            output[r0 : r0 + h, c0 : c0 + w] = color
            pieces.append(
                {
                    "row": rr,
                    "col": cc,
                    "h": h,
                    "w": w,
                    "color": color,
                    "sides": {side: None for side in SIDE_ORDER},
                }
            )
            index_at[(rr, cc)] = idx
            idx += 1

    for rr in range(rows):
        for cc in range(cols - 1):
            left_idx = index_at[(rr, cc)]
            right_idx = index_at[(rr, cc + 1)]
            off = random.randint(1, int(row_heights[rr]) - 2)
            if random.random() < 0.5:
                pieces[left_idx]["sides"]["right"] = ("tooth", off, int(pieces[right_idx]["color"]))
                pieces[right_idx]["sides"]["left"] = ("slot", off)
            else:
                pieces[left_idx]["sides"]["right"] = ("slot", off)
                pieces[right_idx]["sides"]["left"] = ("tooth", off, int(pieces[left_idx]["color"]))

    for rr in range(rows - 1):
        for cc in range(cols):
            top_idx = index_at[(rr, cc)]
            bottom_idx = index_at[(rr + 1, cc)]
            off = random.randint(1, int(col_widths[cc]) - 2)
            if random.random() < 0.5:
                pieces[top_idx]["sides"]["bottom"] = ("tooth", off, int(pieces[bottom_idx]["color"]))
                pieces[bottom_idx]["sides"]["top"] = ("slot", off)
            else:
                pieces[top_idx]["sides"]["bottom"] = ("slot", off)
                pieces[bottom_idx]["sides"]["top"] = ("tooth", off, int(pieces[top_idx]["color"]))

    return pieces, output


def _expand_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int], pad: int = 1) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _mark_blocked(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _frame_bbox(corner: str, canvas_h: int, canvas_w: int, frame_h: int, frame_w: int) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, 0, frame_h - 1, frame_w - 1
    if corner == "tr":
        return 0, canvas_w - frame_w, frame_h - 1, canvas_w - 1
    if corner == "bl":
        return canvas_h - frame_h, 0, canvas_h - 1, frame_w - 1
    return canvas_h - frame_h, canvas_w - frame_w, canvas_h - 1, canvas_w - 1


def _place_patch(
    canvas: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
) -> bool:
    h, w = canvas.shape
    ph, pw = patch.shape
    candidates = [
        (r, c)
        for r in range(h - ph + 1)
        for c in range(w - pw + 1)
        if not blocked[r : r + ph, c : c + pw].any()
    ]
    random.shuffle(candidates)
    for r, c in candidates:
        mask = patch != BG
        canvas[r : r + ph, c : c + pw][mask] = patch[mask]
        bbox = _expand_bbox((r, c, r + ph - 1, c + pw - 1), canvas.shape, pad=1)
        _mark_blocked(blocked, bbox)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    pieces, expected = _sample_layout()
    frame_h, frame_w = expected.shape
    frame_color = int(random.choice([c for c in range(1, 10) if c not in {p["color"] for p in pieces}]))

    rotated_patches = []
    for piece in pieces:
        patch = _build_piece_patch(piece)
        patch = np.rot90(patch, random.choice(ROTATIONS))
        rotated_patches.append(patch)

    max_piece_h = max(int(p.shape[0]) for p in rotated_patches)
    max_piece_w = max(int(p.shape[1]) for p in rotated_patches)
    canvas_h_lo = max(22, int(frame_h) + 5, max_piece_h + 5)
    canvas_w_lo = max(22, int(frame_w) + 5, max_piece_w + 5)
    canvas_h = random.randint(canvas_h_lo, min(30, canvas_h_lo + 4))
    canvas_w = random.randint(canvas_w_lo, min(30, canvas_w_lo + 4))

    canvas = np.zeros((canvas_h, canvas_w), dtype=int)
    blocked = np.zeros_like(canvas, dtype=bool)

    corner = random.choice(("tl", "tr", "bl", "br"))
    bbox = _frame_bbox(corner, canvas_h, canvas_w, int(frame_h), int(frame_w))
    _draw_frame(canvas, bbox, frame_color)
    _mark_blocked(blocked, _expand_bbox(bbox, canvas.shape, pad=1))

    order = list(range(len(rotated_patches)))
    order.sort(key=lambda i: int(rotated_patches[i].shape[0] * rotated_patches[i].shape[1]), reverse=True)
    for idx in order:
        if not _place_patch(canvas, blocked, rotated_patches[idx]):
            return None

    try:
        if np.array_equal(main(canvas), expected):
            return canvas
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
