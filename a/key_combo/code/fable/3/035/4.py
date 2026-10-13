from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

GEN_RETRY_LIMIT = 400


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox_from_mask(mask)
    if box is None:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = _normalize_mask(np.rot90(mask, k))
        if not any(np.array_equal(rotated, old) for old in variants):
            variants.append(rotated)
    return variants


def _canonical_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(
        ((int(rot.shape[0]), int(rot.shape[1])), rot.astype(np.uint8).tobytes())
        for rot in _unique_rotations(mask)
    )


def _extract_pockets(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[dict]]:
    blue_mask = grid == BLUE
    blue_box = _bbox_from_mask(blue_mask)
    if blue_box is None:
        return (0, 0, -1, -1), []
    r0, c0, r1, c1 = blue_box
    band = grid[r0 : r1 + 1, c0 : c1 + 1] == BLACK
    pockets: list[dict] = []
    for coords in _find_components(band):
        if not np.any(coords[:, 0] == band.shape[0] - 1):
            continue
        absolute = coords + np.array([r0, c0], dtype=int)
        mask = _mask_from_coords(absolute)
        pockets.append(
            {
                "coords": absolute,
                "mask": mask,
                "bbox": _bbox_from_coords(absolute),
                "signature": _canonical_signature(mask),
            }
        )
    return blue_box, pockets


def _is_insertable(coords: np.ndarray) -> bool:
    bottom = int(coords[:, 0].max())
    for col in sorted({int(c) for c in coords[:, 1]}):
        rows = np.sort(coords[coords[:, 1] == col][:, 0]).astype(int)
        if rows[-1] != bottom:
            return False
        if not np.array_equal(rows, np.arange(rows[0], bottom + 1, dtype=int)):
            return False
    return True


def _place_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert only the gray piece that exactly matches a downward blue pocket and can slide straight up into it; recolor blocked matches red.

    Input structure
    - The grid has a black background, one upper blue 4-connected structure, and several gray 4-connected candidate pieces below it.
    - Inside the blue structure bbox, each black 4-connected component that touches the bbox bottom edge is a downward pocket.

    Rule
    - Compare every gray piece with every pocket up to rotation; a match requires exact shape equality.
    - A matched pocket is insertable only when every occupied column of the pocket is a continuous vertical run that reaches the mouth row at the pocket bottom.
    - Insertable matches are removed from their original position and painted into the pocket in gray; matched but undercut pockets leave the piece in place and recolor it red.
    - Gray pieces that match no pocket stay unchanged.

    Key insight
    - Shape equality is only the first test: the decisive condition is vertical motion, so any pocket column that does not stay open all the way to the mouth makes the part impossible to push in from below.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    _blue_box, pockets = _extract_pockets(g)
    pockets_by_sig: dict[tuple[tuple[int, int], bytes], list[dict]] = defaultdict(list)
    for pocket in pockets:
        pockets_by_sig[pocket["signature"]].append(pocket)

    matched_pockets: set[int] = set()
    gray_components = _find_components(g == GREY)

    for piece in gray_components:
        piece_mask = _mask_from_coords(piece)
        signature = _canonical_signature(piece_mask)
        candidates = pockets_by_sig.get(signature, [])
        if len(candidates) != 1:
            continue
        pocket = candidates[0]
        pocket_id = id(pocket)
        if pocket_id in matched_pockets:
            continue
        matched_pockets.add(pocket_id)

        if _is_insertable(pocket["coords"]):
            out[piece[:, 0], piece[:, 1]] = BLACK
            out[pocket["coords"][:, 0], pocket["coords"][:, 1]] = GREY
        else:
            out[piece[:, 0], piece[:, 1]] = RED

    return out


def _sample_insertable_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        h = int(rng.integers(2, 6))
        w = int(rng.integers(2, 6))
        tops = rng.integers(0, h, size=w)
        mask = np.zeros((h, w), dtype=bool)
        for c, top in enumerate(tops):
            mask[int(top) :, c] = True
        if 4 <= int(mask.sum()) <= 12:
            return _normalize_mask(mask)
    raise RuntimeError("failed to sample insertable mask")


def _sample_blocked_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        h = int(rng.integers(3, 7))
        w = int(rng.integers(3, 6))
        bottoms = rng.integers(0, h - 1, size=w)
        mouth_width = int(rng.integers(1, min(3, w) + 1))
        mouth_start = int(rng.integers(0, w - mouth_width + 1))
        bottoms[mouth_start : mouth_start + mouth_width] = h - 1
        if not np.any(bottoms < h - 1):
            continue
        mask = np.zeros((h, w), dtype=bool)
        for c, bottom in enumerate(bottoms):
            mask[: int(bottom) + 1, c] = True
        if 5 <= int(mask.sum()) <= 14:
            return _normalize_mask(mask)
    raise RuntimeError("failed to sample blocked mask")


def _sample_random_connected_mask(
    rng: np.random.Generator,
    *,
    min_cells: int,
    max_cells: int,
    max_span: int,
) -> np.ndarray:
    for _ in range(300):
        target = int(rng.integers(min_cells, max_cells + 1))
        cells = {(0, 0)}
        growth_steps = 0
        while len(cells) < target and growth_steps < 800:
            base = list(cells)[int(rng.integers(len(cells)))]
            dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(4))]
            nxt = (base[0] + dr, base[1] + dc)
            growth_steps += 1
            if nxt in cells:
                continue
            trial = cells | {nxt}
            rows = [r for r, _ in trial]
            cols = [c for _, c in trial]
            if max(rows) - min(rows) + 1 > max_span or max(cols) - min(cols) + 1 > max_span:
                continue
            cells = trial
        if len(cells) != target:
            continue
        coords = np.array(sorted(cells), dtype=int)
        coords -= coords.min(axis=0)
        mask = np.zeros((coords[:, 0].max() + 1, coords[:, 1].max() + 1), dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        mask = _normalize_mask(mask)
        if mask.shape[0] >= 2 and mask.shape[1] >= 2:
            return mask
    raise RuntimeError("failed to sample connected mask")


def _sample_unique_pocket_masks(rng: np.random.Generator) -> list[dict]:
    extra_kind = "insertable" if float(rng.random()) < 0.5 else "blocked"
    kinds = ["insertable", "blocked", extra_kind]
    pockets: list[dict] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for kind in kinds:
        for _ in range(300):
            mask = _sample_insertable_mask(rng) if kind == "insertable" else _sample_blocked_mask(rng)
            sig = _canonical_signature(mask)
            if sig in seen:
                continue
            seen.add(sig)
            pockets.append({"kind": kind, "mask": mask, "signature": sig})
            break
        else:
            raise RuntimeError("failed to sample unique pocket masks")
    return pockets


def _carve_structure(
    grid: np.ndarray,
    rng: np.random.Generator,
    pockets: list[dict],
) -> tuple[tuple[int, int, int, int], list[dict]]:
    h, w = grid.shape
    side_margin_left = int(rng.integers(1, 4))
    side_margin_right = int(rng.integers(1, 4))
    separators = [int(rng.integers(1, 4)) for _ in range(len(pockets) - 1)]
    structure_w = side_margin_left + side_margin_right + sum(p["mask"].shape[1] for p in pockets) + sum(separators)
    structure_h = max(p["mask"].shape[0] for p in pockets) + int(rng.integers(1, 4))
    if structure_w > w - 2 or structure_h > h - 12:
        raise RuntimeError("structure does not fit")

    top = int(rng.integers(1, 4))
    left = int(rng.integers(1, w - structure_w))
    grid[top : top + structure_h, left : left + structure_w] = BLUE

    placements: list[dict] = []
    cursor = left + side_margin_left
    bottom = top + structure_h - 1
    for index, pocket in enumerate(pockets):
        mask = pocket["mask"]
        ph, pw = mask.shape
        pocket_top = bottom - ph + 1
        pocket_left = cursor
        window = grid[pocket_top : pocket_top + ph, pocket_left : pocket_left + pw]
        window[mask] = BLACK
        coords = np.argwhere(mask) + np.array([pocket_top, pocket_left], dtype=int)
        placements.append(
            {
                "kind": pocket["kind"],
                "mask": mask,
                "signature": pocket["signature"],
                "coords": coords,
                "bbox": _bbox_from_coords(coords),
            }
        )
        cursor += pw
        if index < len(separators):
            cursor += separators[index]
    return (top, left, top + structure_h - 1, left + structure_w - 1), placements


def _can_place_piece(mask: np.ndarray, top: int, left: int, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    ph, pw = mask.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    coords = np.argwhere(mask)
    for r, c in coords:
        rr = top + int(r)
        cc = left + int(c)
        if occupied[rr, cc]:
            return False
        for nr in range(max(0, rr - 1), min(h, rr + 2)):
            for nc in range(max(0, cc - 1), min(w, cc + 2)):
                if occupied[nr, nc]:
                    return False
    return True


def _place_pieces(
    grid: np.ndarray,
    rng: np.random.Generator,
    structure_box: tuple[int, int, int, int],
    pocket_placements: list[dict],
) -> list[dict]:
    h, w = grid.shape
    occupied = grid != BLACK
    lower_start = structure_box[2] + 3
    if lower_start >= h - 2:
        raise RuntimeError("not enough room below the structure")

    forbidden = {pocket["signature"] for pocket in pocket_placements}
    pieces: list[dict] = []
    for pocket in pocket_placements:
        variants = _unique_rotations(pocket["mask"])
        piece_mask = variants[int(rng.integers(len(variants)))]
        pieces.append({"mask": piece_mask, "role": pocket["kind"], "signature": pocket["signature"]})

    unmatched_count = 1 + int(float(rng.random()) < 0.45)
    unmatched_seen: set[tuple[tuple[int, int], bytes]] = set()
    for _ in range(unmatched_count):
        for _ in range(300):
            piece_mask = _sample_random_connected_mask(
                rng,
                min_cells=4,
                max_cells=9,
                max_span=5,
            )
            sig = _canonical_signature(piece_mask)
            if sig in forbidden or sig in unmatched_seen:
                continue
            unmatched_seen.add(sig)
            pieces.append({"mask": piece_mask, "role": "unmatched", "signature": sig})
            break
        else:
            raise RuntimeError("failed to sample unmatched piece")

    order = sorted(range(len(pieces)), key=lambda idx: int(pieces[idx]["mask"].sum()), reverse=True)
    placed: list[dict | None] = [None] * len(pieces)
    for idx in order:
        piece = pieces[idx]
        mask = piece["mask"]
        ph, pw = mask.shape
        if ph > h - lower_start or pw > w - 2:
            raise RuntimeError("piece does not fit")
        success = False
        for _ in range(400):
            top = int(rng.integers(lower_start, h - ph + 1))
            left = int(rng.integers(1, w - pw))
            if not _can_place_piece(mask, top, left, occupied):
                continue
            _place_mask(grid, mask, top, left, GREY)
            coords = np.argwhere(mask) + np.array([top, left], dtype=int)
            occupied[coords[:, 0], coords[:, 1]] = True
            placed[idx] = {
                "mask": mask,
                "role": piece["role"],
                "signature": piece["signature"],
                "coords": coords,
            }
            success = True
            break
        if not success:
            raise RuntimeError("failed to place piece")
    return [item for item in placed if item is not None]


def _validate_generated_sample(
    grid: np.ndarray,
    output: np.ndarray,
    pocket_placements: list[dict],
    placed_pieces: list[dict],
) -> bool:
    if not np.array_equal(output, main(grid)):
        return False
    if np.array_equal(grid, output):
        return False

    pocket_by_sig = {pocket["signature"]: pocket for pocket in pocket_placements}
    saw_inserted = False
    saw_blocked = False
    saw_unmatched = False

    for piece in placed_pieces:
        coords = piece["coords"]
        role = piece["role"]
        if role == "insertable":
            pocket = pocket_by_sig[piece["signature"]]
            if not np.all(output[coords[:, 0], coords[:, 1]] == BLACK):
                return False
            if not np.all(output[pocket["coords"][:, 0], pocket["coords"][:, 1]] == GREY):
                return False
            saw_inserted = True
        elif role == "blocked":
            if not np.all(output[coords[:, 0], coords[:, 1]] == RED):
                return False
            saw_blocked = True
        else:
            if not np.all(output[coords[:, 0], coords[:, 1]] == GREY):
                return False
            saw_unmatched = True

    return saw_inserted and saw_blocked and saw_unmatched


def _generate_one(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(21, 29))
    w = int(rng.integers(21, 29))
    grid = np.full((h, w), BLACK, dtype=int)

    pocket_specs = _sample_unique_pocket_masks(rng)
    structure_box, pocket_placements = _carve_structure(grid, rng, pocket_specs)
    placed_pieces = _place_pieces(grid, rng, structure_box, pocket_placements)

    output = main(grid)
    if not _validate_generated_sample(grid, output, pocket_placements, placed_pieces):
        raise RuntimeError("generated sample failed validation")
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRY_LIMIT):
        try:
            return _generate_one(rng)
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid input")
