from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
PIECE = 9
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    best: tuple[int, int, bytes] | None = None
    for rot in range(4):
        turned = np.rot90(mask, rot)
        for flip in (False, True):
            variant = np.fliplr(turned) if flip else turned
            key = (int(variant.shape[0]), int(variant.shape[1]), variant.tobytes())
            if best is None or key < best:
                best = key
    assert best is not None
    return best


def _apply_transform(mask: np.ndarray, rot: int, flip: bool) -> np.ndarray:
    out = np.rot90(mask, rot)
    if flip:
        out = np.fliplr(out)
    return out


def _clear_component(out: np.ndarray, coords: np.ndarray) -> None:
    out[coords[:, 0], coords[:, 1]] = BG


def _fill_component(out: np.ndarray, coords: np.ndarray, color: int) -> None:
    out[coords[:, 0], coords[:, 1]] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each hole inside the grey frame with the outside maroon piece of the same shape
    (up to rotation/flip), then recolor the whole hole with the piece's unique marker color.
    """
    g = np.asarray(grid, dtype=int)

    frame_components = _components(g == FRAME)
    assert frame_components, "expected a grey frame"
    frame_coords = max(frame_components, key=len)
    fr0, fc0, fr1, fc1 = _bbox(frame_coords)

    sub = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    holes_by_key: dict[tuple[int, int, bytes], list[np.ndarray]] = {}
    for coords in _components(sub == BG):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 1] == 0):
            continue
        if np.any(coords[:, 0] == sub.shape[0] - 1) or np.any(coords[:, 1] == sub.shape[1] - 1):
            continue
        mask, _ = _tight_mask(coords)
        key = _shape_key(mask)
        absolute = coords + np.array([fr0, fc0], dtype=int)
        holes_by_key.setdefault(key, []).append(absolute)

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    piece_components = _components((g != BG) & outside)

    pieces_by_key: dict[tuple[int, int, bytes], list[tuple[np.ndarray, int]]] = {}
    for coords in piece_components:
        values = np.array([int(g[r, c]) for r, c in coords], dtype=int)
        marker_idx = np.flatnonzero(values != PIECE)
        assert marker_idx.size == 1, "each piece must have exactly one marker cell"
        marker_color = int(values[marker_idx[0]])
        mask, _ = _tight_mask(coords)
        key = _shape_key(mask)
        pieces_by_key.setdefault(key, []).append((coords, marker_color))

    assert holes_by_key, "expected at least one hole"
    assert sum(len(v) for v in holes_by_key.values()) == sum(len(v) for v in pieces_by_key.values())

    out = g.copy()
    for groups in pieces_by_key.values():
        for coords, _ in groups:
            _clear_component(out, coords)

    for key, holes in holes_by_key.items():
        pieces = pieces_by_key.get(key)
        assert pieces is not None, "missing matching piece"
        assert len(holes) == len(pieces), "shape multiplicity mismatch"
        holes_sorted = sorted(holes, key=_bbox)
        pieces_sorted = sorted(pieces, key=lambda item: _bbox(item[0]))
        for hole_coords, (_, marker_color) in zip(holes_sorted, pieces_sorted, strict=True):
            _fill_component(out, hole_coords, marker_color)

    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _mark_forbidden(forbidden: np.ndarray, coords: np.ndarray) -> None:
    stamp = np.zeros_like(forbidden)
    stamp[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(stamp)


def _grow_shape(allowed: np.ndarray, area: int, rng: random.Random) -> np.ndarray | None:
    free_cells = [tuple(map(int, cell)) for cell in np.argwhere(allowed)]
    if len(free_cells) < area:
        return None
    rng.shuffle(free_cells)
    seed_count = min(len(free_cells), 12)
    for start in free_cells[:seed_count]:
        chosen = {start}
        frontier = {start}
        while len(chosen) < area:
            candidates: set[tuple[int, int]] = set()
            for r, c in frontier:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < allowed.shape[0] and 0 <= nc < allowed.shape[1] and allowed[nr, nc]:
                        if (nr, nc) not in chosen:
                            candidates.add((nr, nc))
            if not candidates:
                break
            nxt = rng.choice(tuple(candidates))
            chosen.add(nxt)
            frontier.add(nxt)
        if len(chosen) != area:
            continue
        coords = np.array(sorted(chosen), dtype=int)
        mask, _ = _tight_mask(coords)
        if min(mask.shape) < 2:
            continue
        if mask.all():
            continue
        return coords
    return None


def _generate_holes(inner_h: int, inner_w: int, n_holes: int, rng: random.Random) -> list[dict] | None:
    forbidden = np.zeros((inner_h, inner_w), dtype=bool)
    holes: list[dict] = []
    used_keys: set[tuple[int, int, bytes]] = set()
    avg_budget = max(5, (inner_h * inner_w) // (n_holes + 4))
    max_area = min(8, avg_budget)

    for _ in range(n_holes):
        placed = None
        for _ in range(80):
            area = rng.randint(3, max_area)
            coords = _grow_shape(~forbidden, area, rng)
            if coords is None:
                continue
            mask, bbox = _tight_mask(coords)
            key = _shape_key(mask)
            if key in used_keys:
                continue
            placed = {"coords": coords, "mask": mask, "bbox": bbox, "key": key}
            break
        if placed is None:
            return None
        holes.append(placed)
        used_keys.add(placed["key"])
        _mark_forbidden(forbidden, placed["coords"])

    return holes


def _mark_bbox(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    occupied[r0:r1, c0:c1] = True


def _find_piece_location(
    occupied: np.ndarray, patch_h: int, patch_w: int, rng: random.Random
) -> tuple[int, int] | None:
    h, w = occupied.shape
    candidates = []
    for top in range(1, h - patch_h):
        for left in range(1, w - patch_w):
            r0 = top - 1
            c0 = left - 1
            r1 = top + patch_h + 1
            c1 = left + patch_w + 1
            if not occupied[r0:r1, c0:c1].any():
                candidates.append((top, left))
    if not candidates:
        return None
    return rng.choice(candidates)


def _build_piece(mask: np.ndarray, marker_color: int, rng: random.Random) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = PIECE
    cells = np.argwhere(mask)
    marker_r, marker_c = map(int, cells[rng.randrange(len(cells))])
    patch[marker_r, marker_c] = int(marker_color)
    return patch


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(22, 30)
    width = rng.randint(22, 30)

    max_frame_h = min(16, height - 7)
    max_frame_w = min(16, width - 7)
    if max_frame_h < 9 or max_frame_w < 10:
        return None

    frame_h = rng.randint(9, max_frame_h)
    frame_w = rng.randint(10, max_frame_w)
    frame_top = rng.randint(2, height - frame_h - 2)
    frame_left = rng.randint(2, width - frame_w - 2)
    frame_bottom = frame_top + frame_h - 1
    frame_right = frame_left + frame_w - 1

    inner_h = frame_h - 2
    inner_w = frame_w - 2
    max_holes = min(5, max(2, (inner_h * inner_w) // 10))
    n_holes = rng.randint(2, max_holes)

    holes = _generate_holes(inner_h, inner_w, n_holes, rng)
    if holes is None:
        return None

    marker_colors = rng.sample(MARKER_COLORS, n_holes)
    pieces = []
    for hole, marker_color in zip(holes, marker_colors, strict=True):
        rot = rng.randrange(4)
        flip = bool(rng.randrange(2))
        piece_mask = _apply_transform(hole["mask"], rot, flip)
        patch = _build_piece(piece_mask, marker_color, rng)
        pieces.append(
            {
                "patch": patch,
                "mask": patch != BG,
                "marker_color": int(marker_color),
                "bbox_area": int(patch.shape[0] * patch.shape[1]),
            }
        )

    occupied = np.zeros((height, width), dtype=bool)
    _mark_bbox(occupied, frame_top, frame_left, frame_h, frame_w, pad=1)

    for piece in sorted(pieces, key=lambda item: item["bbox_area"], reverse=True):
        patch = piece["patch"]
        loc = _find_piece_location(occupied, patch.shape[0], patch.shape[1], rng)
        if loc is None:
            return None
        top, left = loc
        piece["top"] = int(top)
        piece["left"] = int(left)
        _mark_bbox(occupied, top, left, patch.shape[0], patch.shape[1], pad=1)

    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)

    inp[frame_top : frame_bottom + 1, frame_left : frame_right + 1] = FRAME
    out[frame_top : frame_bottom + 1, frame_left : frame_right + 1] = FRAME

    for hole, marker_color in zip(holes, marker_colors, strict=True):
        abs_coords = hole["coords"] + np.array([frame_top + 1, frame_left + 1], dtype=int)
        inp[abs_coords[:, 0], abs_coords[:, 1]] = BG
        out[abs_coords[:, 0], abs_coords[:, 1]] = int(marker_color)

    for piece in pieces:
        top = piece["top"]
        left = piece["left"]
        patch = piece["patch"]
        mask = piece["mask"]
        region = inp[top : top + patch.shape[0], left : left + patch.shape[1]]
        region[mask] = patch[mask]

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
