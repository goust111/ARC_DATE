from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
MOLD_COLOR = 5
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 500
CANVAS_SIDE = 30
PLACEMENT_GAP = 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


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
                coords.append((int(cr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    return _tight_mask_from_coords(coords)


def _mask_signature(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    tight = _normalize_mask(np.asarray(mask, dtype=bool))
    return (
        int(tight.shape[0]),
        int(tight.shape[1]),
        tuple(int(x) for x in tight.ravel()),
    )


def _apply_transform(mask: np.ndarray, op: int) -> np.ndarray:
    out = np.rot90(np.asarray(mask, dtype=bool), op % 4)
    if op >= 4:
        out = np.fliplr(out)
    return _normalize_mask(out)


def _d4_masks(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[int, int, tuple[int, ...]]] = set()
    for op in range(8):
        transformed = _apply_transform(mask, op)
        sig = _mask_signature(transformed)
        if sig not in seen:
            seen.add(sig)
            variants.append(transformed)
    return variants


def _canonical_signature(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    return min(_mask_signature(variant) for variant in _d4_masks(mask))


def _has_internal_hole(mask: np.ndarray) -> bool:
    mask = _normalize_mask(mask)
    empty = ~mask
    for comp in _components(empty):
        if (
            np.any(comp[:, 0] == 0)
            or np.any(comp[:, 0] == mask.shape[0] - 1)
            or np.any(comp[:, 1] == 0)
            or np.any(comp[:, 1] == mask.shape[1] - 1)
        ):
            continue
        return True
    return False


def _find_molds(grid: np.ndarray, bg: int) -> list[dict]:
    molds: list[dict] = []
    for color in (int(x) for x in np.unique(grid) if int(x) != int(bg)):
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            crop = grid[r0 : r1 + 1, c0 : c1 + 1]
            if np.count_nonzero(crop == color) != len(comp):
                continue
            if np.any((crop != bg) & (crop != color)):
                continue
            if (
                np.any(crop[0, :] != color)
                or np.any(crop[-1, :] != color)
                or np.any(crop[:, 0] != color)
                or np.any(crop[:, -1] != color)
            ):
                continue
            hole_mask = crop == bg
            if not np.any(hole_mask):
                continue
            hole_comps = _components(hole_mask)
            if len(hole_comps) != 1:
                continue
            hole = hole_comps[0]
            if (
                np.any(hole[:, 0] == 0)
                or np.any(hole[:, 0] == hole_mask.shape[0] - 1)
                or np.any(hole[:, 1] == 0)
                or np.any(hole[:, 1] == hole_mask.shape[1] - 1)
            ):
                continue
            hr0, hc0, hr1, hc1 = _bbox(hole)
            tight_hole = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
            tight_hole[hole[:, 0] - hr0, hole[:, 1] - hc0] = True
            molds.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "hole_bbox": (int(r0 + hr0), int(c0 + hc0), int(r0 + hr1), int(c0 + hc1)),
                    "hole_mask": tight_hole,
                }
            )
    molds.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return molds


def _find_pieces(grid: np.ndarray, bg: int, molds: Iterable[dict]) -> list[dict]:
    blocked = np.zeros_like(grid, dtype=bool)
    for mold in molds:
        r0, c0, r1, c1 = mold["bbox"]
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    pieces: list[dict] = []
    for color in (int(x) for x in np.unique(grid) if int(x) != int(bg)):
        comps = _components((grid == color) & ~blocked)
        for comp in comps:
            mask = _tight_mask_from_coords(comp)
            pieces.append(
                {
                    "color": int(color),
                    "coords": comp,
                    "mask": mask,
                    "canon": _canonical_signature(mask),
                }
            )
    pieces.sort(key=lambda item: (item["color"], len(item["coords"])))
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """Fit each scattered colored fragment into the unique grey mold cavity with the same shape up to rotation or reflection.

    Input structure:
    - The background is black.
    - Several grey rectangular mold components each contain one enclosed black cavity.
    - Outside the molds, several colored 4-connected fragments appear on the background.

    Rule:
    - Extract every mold cavity as a normalized shape.
    - Match each outside fragment to the unique cavity with the same shape under the 8 dihedral transforms.
    - Erase the fragment from its source location and paint its color into the matched cavity.
    - Keep the remaining grey mold pixels and all other background pixels unchanged.

    Key insight:
    - Fragment-to-cavity correspondence is determined purely by shape equivalence under rotation and reflection, not by color or position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _most_common_color(g)

    molds = _find_molds(g, bg)
    assert molds, "expected at least one mold"

    pieces = _find_pieces(g, bg, molds)
    assert pieces, "expected at least one piece"
    assert len(pieces) == len(molds), "pieces and cavities must pair one-to-one"

    hole_by_signature: dict[tuple[int, int, tuple[int, ...]], dict] = {}
    for mold in molds:
        sig = _canonical_signature(mold["hole_mask"])
        assert sig not in hole_by_signature, "cavity shapes must be uniquely matchable"
        hole_by_signature[sig] = mold

    out = g.copy()
    for piece in pieces:
        out[piece["coords"][:, 0], piece["coords"][:, 1]] = bg

    used_holes: set[tuple[int, int, int, int]] = set()
    for piece in pieces:
        sig = piece["canon"]
        assert sig in hole_by_signature, "piece shape must match a cavity"
        mold = hole_by_signature[sig]
        hole_bbox = mold["hole_bbox"]
        assert hole_bbox not in used_holes, "multiple pieces cannot use the same cavity"
        used_holes.add(hole_bbox)
        hr0, hc0, hr1, hc1 = hole_bbox
        patch = out[hr0 : hr1 + 1, hc0 : hc1 + 1]
        patch[mold["hole_mask"]] = int(piece["color"])

    return out


def _random_shape() -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        target = random.randint(3, min(h * w - 1, 8))
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier = {
            (start[0] + dr, start[1] + dc)
            for dr, dc in NEIGHBORS4
            if 0 <= start[0] + dr < h and 0 <= start[1] + dc < w
        }
        while int(mask.sum()) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            if mask[cell]:
                continue
            mask[cell] = True
            for dr, dc in NEIGHBORS4:
                nr, nc = cell[0] + dr, cell[1] + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((int(nr), int(nc)))
        if int(mask.sum()) != target:
            continue
        mask = _normalize_mask(mask)
        if _has_internal_hole(mask):
            continue
        if int(mask.sum()) == int(mask.shape[0] * mask.shape[1]):
            continue
        return mask
    return None


def _make_mold_patch(hole_mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    hole_h, hole_w = hole_mask.shape
    pad_top = random.randint(1, 2)
    pad_bottom = random.randint(1, 2)
    pad_left = random.randint(1, 2)
    pad_right = random.randint(1, 2)
    patch_h = hole_h + pad_top + pad_bottom
    patch_w = hole_w + pad_left + pad_right
    patch = np.full((patch_h, patch_w), MOLD_COLOR, dtype=int)
    patch[pad_top : pad_top + hole_h, pad_left : pad_left + hole_w][hole_mask] = BG
    return patch, (
        int(pad_top),
        int(pad_left),
        int(pad_top + hole_h - 1),
        int(pad_left + hole_w - 1),
    )


def _place_patch(
    canvas: np.ndarray,
    reserved: np.ndarray,
    patch: np.ndarray,
    gap: int = PLACEMENT_GAP,
) -> tuple[int, int] | None:
    h, w = patch.shape
    rows = list(range(canvas.shape[0] - h + 1))
    cols = list(range(canvas.shape[1] - w + 1))
    random.shuffle(rows)
    random.shuffle(cols)
    for r0 in rows:
        for c0 in cols:
            rr0 = max(0, r0 - gap)
            cc0 = max(0, c0 - gap)
            rr1 = min(canvas.shape[0], r0 + h + gap)
            cc1 = min(canvas.shape[1], c0 + w + gap)
            if np.any(reserved[rr0:rr1, cc0:cc1]):
                continue
            patch_mask = patch != BG
            canvas[r0 : r0 + h, c0 : c0 + w][patch_mask] = patch[patch_mask]
            reserved[rr0:rr1, cc0:cc1] = True
            return int(r0), int(c0)
    return None


def _crop_with_margin(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used = np.argwhere(inp != BG)
    r0, c0, r1, c1 = _bbox(used)
    top = random.randint(0, min(3, r0))
    left = random.randint(0, min(3, c0))
    bottom = random.randint(0, min(3, inp.shape[0] - 1 - r1))
    right = random.randint(0, min(3, inp.shape[1] - 1 - c1))
    slc = (slice(r0 - top, r1 + bottom + 1), slice(c0 - left, c1 + right + 1))
    return inp[slc].copy(), out[slc].copy()


def _gen_attempt() -> np.ndarray | None:
    piece_count = random.randint(2, 4)
    colors = random.sample(PIECE_COLORS, piece_count)

    holes: list[np.ndarray] = []
    seen_signatures: set[tuple[int, int, tuple[int, ...]]] = set()
    for _ in range(piece_count):
        hole = None
        for _ in range(100):
            candidate = _random_shape()
            if candidate is None:
                continue
            sig = _canonical_signature(candidate)
            if sig in seen_signatures:
                continue
            hole = candidate
            seen_signatures.add(sig)
            break
        if hole is None:
            return None
        holes.append(hole)

    piece_ops = [random.randrange(8) for _ in range(piece_count)]
    if all(
        _mask_signature(_apply_transform(holes[i], piece_ops[i])) == _mask_signature(holes[i])
        for i in range(piece_count)
    ):
        adjustable = [
            i
            for i, hole in enumerate(holes)
            if any(_mask_signature(_apply_transform(hole, op)) != _mask_signature(hole) for op in range(8))
        ]
        if not adjustable:
            return None
        idx = random.choice(adjustable)
        alternatives = [
            op for op in range(8) if _mask_signature(_apply_transform(holes[idx], op)) != _mask_signature(holes[idx])
        ]
        piece_ops[idx] = random.choice(alternatives)

    canvas = np.zeros((CANVAS_SIDE, CANVAS_SIDE), dtype=int)
    reserved = np.zeros_like(canvas, dtype=bool)
    expected = canvas.copy()

    mold_infos: list[dict] = []
    for hole in holes:
        patch, local_hole_bbox = _make_mold_patch(hole)
        pos = _place_patch(canvas, reserved, patch)
        if pos is None:
            return None
        mr, mc = pos
        hr0, hc0, hr1, hc1 = local_hole_bbox
        mold_infos.append(
            {
                "hole_bbox": (int(mr + hr0), int(mc + hc0), int(mr + hr1), int(mc + hc1)),
                "hole_mask": hole,
            }
        )

    piece_infos: list[dict] = []
    for color, hole, op in zip(colors, holes, piece_ops):
        mask = _apply_transform(hole, op)
        patch = np.where(mask, int(color), BG)
        pos = _place_patch(canvas, reserved, patch)
        if pos is None:
            return None
        pr, pc = pos
        coords = np.argwhere(mask)
        abs_coords = coords + np.array([[pr, pc]])
        piece_infos.append({"color": int(color), "coords": abs_coords})

    expected[:] = canvas
    for piece_info, mold_info in zip(piece_infos, mold_infos):
        expected[piece_info["coords"][:, 0], piece_info["coords"][:, 1]] = BG
        hr0, hc0, hr1, hc1 = mold_info["hole_bbox"]
        expected[hr0 : hr1 + 1, hc0 : hc1 + 1][mold_info["hole_mask"]] = int(piece_info["color"])

    inp, out = _crop_with_margin(canvas, expected)
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mold-and-piece puzzle")
