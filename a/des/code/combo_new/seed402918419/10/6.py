from __future__ import annotations

import random

import numpy as np


BG = 0
SIDES = ("left", "right", "top", "bottom")
PALETTE = tuple(range(1, 10))
MAX_GEN_ATTEMPTS = 400
LOCK_COUNT_RANGE = (1, 3)
GRID_SIDE_RANGE = (18, 30)
LOCK_SIDE_RANGE = (6, 11)
NOTCH_SPAN_RANGE = (2, 4)
NOTCH_DEPTH_RANGE = (2, 4)
PLACEMENT_PADDING = 1
PLACEMENT_BORDER = 1


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _binary_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _coords_to_component(coords: list[tuple[int, int]]) -> dict:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "area": len(coords),
        "mask": mask,
    }


def _color_components(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == int(color)
    return [_coords_to_component(coords) for coords in _binary_components(mask)]


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _lock_info(grid: np.ndarray, color: int, comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any((patch != int(BG)) & (patch != int(color))):
        return None

    occ = patch == int(color)
    hole = ~occ
    if not bool(hole.any()):
        return None

    hole_comps = _binary_components(hole)
    if len(hole_comps) != 1:
        return None

    sides = []
    if bool(hole[:, 0].any()):
        sides.append("left")
    if bool(hole[:, -1].any()):
        sides.append("right")
    if bool(hole[0, :].any()):
        sides.append("top")
    if bool(hole[-1, :].any()):
        sides.append("bottom")
    if len(sides) != 1:
        return None

    hole_coords_rel = hole_comps[0]
    hole_rows = [r for r, _ in hole_coords_rel]
    hole_cols = [c for _, c in hole_coords_rel]
    hr0, hr1 = min(hole_rows), max(hole_rows)
    hc0, hc1 = min(hole_cols), max(hole_cols)
    hole_mask = hole[hr0 : hr1 + 1, hc0 : hc1 + 1]
    hole_coords_abs = [(r0 + rr, c0 + cc) for rr, cc in hole_coords_rel]

    return {
        "side": sides[0],
        "hole_mask": hole_mask,
        "hole_coords_abs": hole_coords_abs,
    }


def _matching_rotations(key_mask: np.ndarray, hole_mask: np.ndarray) -> list[int]:
    matches = []
    for k in range(4):
        rot = np.rot90(key_mask, k)
        if rot.shape == hole_mask.shape and np.array_equal(rot, hole_mask):
            matches.append(k)
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored key to its same-colored lock body, rotate the key to the fitting orientation, and fill the side notch.

    Input structure:
    - The background is black.
    - Each non-black color appears as one large monochrome lock body and one small monochrome key.
    - A lock body is a near-solid rectangle with one side-attached missing notch; a key is a separate L- or T-shaped component of the same color.

    Rule:
    - Group components by color and identify the lock as the component whose bbox contains a single background cavity touching exactly one bbox side.
    - Extract the lock-hole shape, then test the same-colored key under the four rotations.
    - The unique rotation whose occupied cells equal the hole mask is translated into the hole location.
    - Erase the key's original pixels and fill the hole so the lock becomes a complete rectangle.

    Key insight:
    - Color selects the pair, but the actual decision is geometric: the key must be mentally rotated until its silhouette exactly matches the notch shape.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(BG)):
        comps = _color_components(g, color)
        assert len(comps) >= 2, "expected one lock and one key per color"

        lock_candidates = []
        key_candidates = []
        for comp in comps:
            info = _lock_info(g, color, comp)
            if info is None:
                key_candidates.append(comp)
            else:
                lock_candidates.append((comp, info))

        assert len(lock_candidates) == 1, "expected a unique lock component per color"
        _lock_comp, lock = lock_candidates[0]

        matches = []
        for key in key_candidates:
            rots = _matching_rotations(key["mask"], lock["hole_mask"])
            if rots:
                matches.append((key, rots))

        assert len(matches) == 1, "expected a unique key matching the hole"
        key_comp, rots = matches[0]
        assert len(rots) == 1, "expected a unique fitting rotation"

        for r, c in key_comp["coords"]:
            out[int(r), int(c)] = int(BG)
        for r, c in lock["hole_coords_abs"]:
            out[int(r), int(c)] = int(color)

    return out


def _rotation_signature(mask: np.ndarray) -> tuple[tuple[bytes, tuple[int, int]], ...]:
    return tuple((np.rot90(mask, k).tobytes(), tuple(np.rot90(mask, k).shape)) for k in range(4))


def _build_hole_mask(side: str, kind: str, span: int, depth: int) -> np.ndarray:
    if side in ("left", "right"):
        mask = np.zeros((span, depth), dtype=bool)
        edge_col = 0 if side == "left" else depth - 1
        mask[:, edge_col] = True
        if kind == "L":
            row = 0 if random.random() < 0.5 else span - 1
            mask[row, :] = True
        else:
            row = random.choice(list(range(1, span - 1)))
            mask[row, :] = True
        return mask

    mask = np.zeros((depth, span), dtype=bool)
    edge_row = 0 if side == "top" else depth - 1
    mask[edge_row, :] = True
    if kind == "L":
        col = 0 if random.random() < 0.5 else span - 1
        mask[:, col] = True
    else:
        col = random.choice(list(range(1, span - 1)))
        mask[:, col] = True
    return mask


def _sample_pair(color: int) -> dict:
    side = random.choice(SIDES)
    kind = random.choice(("L", "T"))

    if side in ("left", "right"):
        span_lo = 3 if kind == "T" else NOTCH_SPAN_RANGE[0]
        span = random.randint(span_lo, NOTCH_SPAN_RANGE[1])
        depth = random.randint(*NOTCH_DEPTH_RANGE)
        lock_h = random.randint(max(LOCK_SIDE_RANGE[0], span + 2), LOCK_SIDE_RANGE[1])
        lock_w = random.randint(max(LOCK_SIDE_RANGE[0], depth + 2), LOCK_SIDE_RANGE[1])
        offset = random.randint(1, lock_h - span - 1)
        hole_r0 = offset
        hole_c0 = 0 if side == "left" else lock_w - depth
    else:
        span_lo = 3 if kind == "T" else NOTCH_SPAN_RANGE[0]
        span = random.randint(span_lo, NOTCH_SPAN_RANGE[1])
        depth = random.randint(*NOTCH_DEPTH_RANGE)
        lock_h = random.randint(max(LOCK_SIDE_RANGE[0], depth + 2), LOCK_SIDE_RANGE[1])
        lock_w = random.randint(max(LOCK_SIDE_RANGE[0], span + 2), LOCK_SIDE_RANGE[1])
        offset = random.randint(1, lock_w - span - 1)
        hole_r0 = 0 if side == "top" else lock_h - depth
        hole_c0 = offset

    hole_mask = _build_hole_mask(side, kind, span, depth)
    sig = _rotation_signature(hole_mask)
    assert len(set(sig)) == 4, "generated key shape must have four distinct rotations"

    lock_patch = np.full((lock_h, lock_w), int(color), dtype=int)
    sub = lock_patch[hole_r0 : hole_r0 + hole_mask.shape[0], hole_c0 : hole_c0 + hole_mask.shape[1]]
    sub[hole_mask] = int(BG)

    key_mask = np.rot90(hole_mask, random.randint(0, 3))
    key_patch = np.full(key_mask.shape, int(BG), dtype=int)
    key_patch[key_mask] = int(color)

    return {
        "color": int(color),
        "lock_patch": lock_patch,
        "output_patch": np.full((lock_h, lock_w), int(color), dtype=int),
        "key_patch": key_patch,
        "side": side,
        "kind": kind,
    }


def _can_place_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    gh, gw = occupied.shape
    if r0 < PLACEMENT_BORDER or c0 < PLACEMENT_BORDER:
        return False
    if r0 + h > gh - PLACEMENT_BORDER or c0 + w > gw - PLACEMENT_BORDER:
        return False

    rr0 = max(0, r0 - PLACEMENT_PADDING)
    cc0 = max(0, c0 - PLACEMENT_PADDING)
    rr1 = min(gh, r0 + h + PLACEMENT_PADDING)
    cc1 = min(gw, c0 + w + PLACEMENT_PADDING)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _candidate_positions(occupied: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    gh, gw = occupied.shape
    positions = []
    for r0 in range(gh - h + 1):
        for c0 in range(gw - w + 1):
            if _can_place_bbox(occupied, r0, c0, h, w):
                positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(BG)
    assert not bool(np.any((sub != int(BG)) & mask))
    sub[mask] = patch[mask]


def _mark_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _attempt_generate() -> np.ndarray | None:
    n_pairs = random.randint(*LOCK_COUNT_RANGE)
    min_side = max(GRID_SIDE_RANGE[0], 14 + 3 * (n_pairs - 1))
    gh = random.randint(min_side, GRID_SIDE_RANGE[1])
    gw = random.randint(min_side, GRID_SIDE_RANGE[1])

    pairs = [_sample_pair(color) for color in random.sample(PALETTE, n_pairs)]
    pairs.sort(key=lambda d: d["output_patch"].shape[0] * d["output_patch"].shape[1], reverse=True)

    inp = np.full((gh, gw), int(BG), dtype=int)
    out = np.full((gh, gw), int(BG), dtype=int)
    occupied = np.zeros((gh, gw), dtype=bool)

    for pair in pairs:
        lock_patch = pair["lock_patch"]
        h, w = lock_patch.shape
        positions = _candidate_positions(occupied, h, w)
        if not positions:
            return None
        r0, c0 = positions[0]
        _paint_patch(inp, lock_patch, r0, c0)
        _paint_patch(out, pair["output_patch"], r0, c0)
        _mark_bbox(occupied, r0, c0, h, w)

    key_items = list(pairs)
    random.shuffle(key_items)
    key_items.sort(key=lambda d: d["key_patch"].shape[0] * d["key_patch"].shape[1], reverse=True)
    for pair in key_items:
        key_patch = pair["key_patch"]
        h, w = key_patch.shape
        positions = _candidate_positions(occupied, h, w)
        if not positions:
            return None
        r0, c0 = positions[0]
        _paint_patch(inp, key_patch, r0, c0)
        _mark_bbox(occupied, r0, c0, h, w)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {MAX_GEN_ATTEMPTS} attempts")
