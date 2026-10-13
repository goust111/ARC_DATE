from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
PINK = 6

DIAGONAL_COLORS = (BLUE, RED, GREEN)
GEN_ATTEMPTS = 500


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int, copy=True)


def _find_components(mask: np.ndarray) -> list[dict[str, object]]:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    start_r, start_c = map(int, cells[0])
    seen = np.zeros_like(mask, dtype=bool)
    q = deque([(start_r, start_c)])
    seen[start_r, start_c] = True
    reached = 0
    while q:
        r, c = q.popleft()
        reached += 1
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return reached == int(mask.sum())


def _dihedral_arrays(arr: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(arr, k)
        out.append(np.array(rot, dtype=arr.dtype, copy=True))
        out.append(np.array(np.fliplr(rot), dtype=arr.dtype, copy=True))
    return out


_MAP_CACHE: dict[int, list[dict[str, np.ndarray]]] = {}


def _orientation_maps(side: int) -> list[dict[str, np.ndarray]]:
    if side in _MAP_CACHE:
        return _MAP_CACHE[side]

    base = np.arange(side * side, dtype=int).reshape(side, side)
    maps: list[dict[str, np.ndarray]] = []
    for oriented in _dihedral_arrays(base):
        oriented_to_base = np.empty((side, side, 2), dtype=int)
        base_to_oriented = np.empty((side, side, 2), dtype=int)
        for r in range(side):
            for c in range(side):
                idx = int(oriented[r, c])
                br, bc = divmod(idx, side)
                oriented_to_base[r, c] = (br, bc)
                base_to_oriented[br, bc] = (r, c)
        maps.append(
            {
                "oriented_to_base": oriented_to_base,
                "base_to_oriented": base_to_oriented,
            }
        )
    _MAP_CACHE[side] = maps
    return maps


def _transform_coords(coords: set[tuple[int, int]], side: int, orientation: int) -> set[tuple[int, int]]:
    mapper = _orientation_maps(side)[orientation]["base_to_oriented"]
    return {tuple(map(int, mapper[r, c])) for r, c in coords}


def _signature_coords(patch: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(patch == PINK)}


def _palette_from_standard(patch: np.ndarray) -> tuple[int, int, int] | None:
    if patch.shape[0] != patch.shape[1]:
        return None
    side = int(patch.shape[0])
    palette: list[int | None] = [None, None, None]
    for r in range(side):
        for c in range(side):
            color = int(patch[r, c])
            if color == PINK:
                continue
            if color == BLACK or color not in DIAGONAL_COLORS:
                return None
            residue = (r + c) % 3
            if palette[residue] is None:
                palette[residue] = color
            elif palette[residue] != color:
                return None
    if any(color is None for color in palette):
        return None
    if set(int(color) for color in palette) != set(DIAGONAL_COLORS):
        return None
    return tuple(int(color) for color in palette)


def _standard_candidates(full_patch: np.ndarray) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for oriented in _dihedral_arrays(full_patch):
        palette = _palette_from_standard(oriented)
        if palette is None:
            continue
        signature = _signature_coords(oriented)
        if len(signature) not in (2, 3):
            continue
        candidates.append({"patch": oriented, "palette": palette, "signature": signature})
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each transformed template copy by using its magenta signature to recover orientation and restore the missing diagonal-phase colors.

    Input structure:
    - The background is black.
    - The foreground consists of 3-5 separated square template copies of one 4x4 or 5x5 pattern.
    - Every template cell is normally colored by a 3-color diagonal cycle, except for 2-3 magenta signature cells that stay magenta in every copy.
    - Exactly one or more copies may be complete, while the incomplete copies contain black holes but keep their magenta signature visible.

    Rule:
    - Find a complete copy and reorient it into a standard local frame where non-magenta colors depend only on `(row + col) mod 3`.
    - Enumerate the 8 dihedral transforms of that standard template and match each incomplete copy by its magenta signature positions inside the copy bbox.
    - For every black hole, map its oriented local coordinate back into the standard local frame and recolor it with the diagonal-cycle palette for that standard coordinate.
    - Leave all existing non-black cells unchanged.

    Key insight:
    - The missing color is not determined by the copy's visible global diagonals alone; it depends on the copy's dihedral orientation, so signature matching and diagonal-phase recovery must be composed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _find_components(g != BLACK)
    assert 3 <= len(comps) <= 5, "expected 3-5 template copies"

    side_values = set()
    patches: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    complete_patches: list[np.ndarray] = []
    for comp in comps:
        bbox = comp["bbox"]
        patch = _crop(g, bbox)
        h, w = map(int, patch.shape)
        assert h == w and h in (4, 5), "each copy must be a 4x4 or 5x5 square"
        side_values.add(h)
        patches.append((bbox, patch))
        if np.all(patch != BLACK):
            complete_patches.append(patch)
    assert len(side_values) == 1, "all copies must share one template size"
    assert complete_patches, "expected at least one complete copy"

    standard: dict[str, object] | None = None
    for patch in complete_patches:
        candidates = _standard_candidates(patch)
        if candidates:
            standard = candidates[0]
            break
    assert standard is not None, "failed to infer a standard template"

    side = int(next(iter(side_values)))
    palette = tuple(int(color) for color in standard["palette"])
    signature = set(standard["signature"])
    transformed_signatures = [_transform_coords(signature, side, idx) for idx in range(8)]
    maps = _orientation_maps(side)

    out = g.copy()
    for bbox, patch in patches:
        r0, c0, _r1, _c1 = bbox
        magenta = _signature_coords(patch)
        matches = [idx for idx, coords in enumerate(transformed_signatures) if coords == magenta]
        assert len(matches) == 1, "signature must determine a unique orientation"
        orientation = matches[0]
        oriented_to_base = maps[orientation]["oriented_to_base"]
        for r, c in np.argwhere(patch == BLACK):
            br, bc = map(int, oriented_to_base[int(r), int(c)])
            out[r0 + int(r), c0 + int(c)] = palette[(br + bc) % 3]
    return out


def _make_base_patch(side: int, palette: tuple[int, int, int], signature: set[tuple[int, int]]) -> np.ndarray:
    patch = np.empty((side, side), dtype=int)
    for r in range(side):
        for c in range(side):
            patch[r, c] = palette[(r + c) % 3]
    for r, c in signature:
        patch[r, c] = PINK
    return patch


def _random_signature(side: int) -> set[tuple[int, int]] | None:
    all_cells = [(r, c) for r in range(side) for c in range(side)]
    count = random.choice((2, 3))
    for _ in range(200):
        sig = set(random.sample(all_cells, count))
        patch = _make_base_patch(side, (BLUE, RED, GREEN), sig)
        if _palette_from_standard(patch) is None:
            continue
        transformed = frozenset(frozenset(coords) for coords in (_transform_coords(sig, side, idx) for idx in range(8)))
        if len(transformed) != 8:
            continue
        return sig
    return None


def _easy_orientation(base_patch: np.ndarray, orientation: int) -> bool:
    oriented = _dihedral_arrays(base_patch)[orientation]
    return _palette_from_standard(oriented) is not None


def _choose_orientations(base_patch: np.ndarray, copy_count: int) -> tuple[list[int], int] | None:
    all_ids = list(range(8))
    easy = {idx for idx in all_ids if _easy_orientation(base_patch, idx)}
    hard = [idx for idx in all_ids if idx not in easy]
    if not hard:
        return None
    for _ in range(200):
        chosen = random.sample(all_ids, copy_count)
        complete_index = random.randrange(copy_count)
        incomplete = [chosen[i] for i in range(copy_count) if i != complete_index]
        if not any(idx in easy for idx in incomplete):
            continue
        if not any(idx not in easy for idx in incomplete):
            continue
        return chosen, complete_index
    return None


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_boxes(H: int, W: int, side: int, count: int) -> list[tuple[int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    positions: list[tuple[int, int]] = []
    for _ in range(300):
        boxes.clear()
        positions.clear()
        ok = True
        for _idx in range(count):
            placed = False
            for _try in range(200):
                r0 = random.randint(0, H - side)
                c0 = random.randint(0, W - side)
                box = (r0, c0, r0 + side - 1, c0 + side - 1)
                if all(_boxes_separated(box, other, gap=1) for other in boxes):
                    boxes.append(box)
                    positions.append((r0, c0))
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return positions
    return None


def _hole_patch(full_patch: np.ndarray, orientation: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    side = int(full_patch.shape[0])
    oriented_to_base = _orientation_maps(side)[orientation]["oriented_to_base"]
    candidates = [tuple(map(int, rc)) for rc in np.argwhere(full_patch != PINK)]
    for _ in range(200):
        hole_count = random.randint(2, 4)
        holes = random.sample(candidates, hole_count)
        patch = np.array(full_patch, copy=True)
        for r, c in holes:
            patch[r, c] = BLACK
        if not _is_connected(patch != BLACK):
            continue
        info = []
        residues = set()
        colors = set()
        for r, c in holes:
            br, bc = map(int, oriented_to_base[r, c])
            residue = (br + bc) % 3
            color = int(full_patch[r, c])
            info.append((r, c, residue, color))
            residues.add(residue)
            colors.add(color)
        if len(residues) < 2:
            continue
        if len(colors) < 2:
            continue
        return patch, info
    return None


def _build_instance() -> tuple[np.ndarray, np.ndarray, dict[str, object]] | None:
    side = random.choice((4, 5))
    palette = tuple(random.sample(list(DIAGONAL_COLORS), 3))
    signature = _random_signature(side)
    if signature is None:
        return None
    base_patch = _make_base_patch(side, palette, signature)

    copy_count = random.randint(3, 5)
    chosen = _choose_orientations(base_patch, copy_count)
    if chosen is None:
        return None
    orientations, complete_index = chosen

    H = random.randint(max(18, side * 3 + 2), 26)
    W = random.randint(max(18, side * 3 + 2), 26)
    positions = _place_boxes(H, W, side, copy_count)
    if positions is None:
        return None

    full_patches = [_dihedral_arrays(base_patch)[idx] for idx in orientations]
    input_patches: list[np.ndarray] = []
    global_residues: set[int] = set()
    global_colors: set[int] = set()
    hole_orientations: list[int] = []
    for i, (orientation, full_patch) in enumerate(zip(orientations, full_patches)):
        if i == complete_index:
            input_patches.append(np.array(full_patch, copy=True))
            continue
        hole_data = _hole_patch(full_patch, orientation)
        if hole_data is None:
            return None
        patch, info = hole_data
        input_patches.append(patch)
        hole_orientations.append(orientation)
        for _r, _c, residue, color in info:
            global_residues.add(int(residue))
            global_colors.add(int(color))

    if global_residues != {0, 1, 2}:
        return None
    if global_colors != set(DIAGONAL_COLORS):
        return None

    grid_in = np.zeros((H, W), dtype=int)
    grid_out = np.zeros((H, W), dtype=int)
    for (r0, c0), in_patch, out_patch in zip(positions, input_patches, full_patches):
        grid_in[r0 : r0 + side, c0 : c0 + side] = in_patch
        grid_out[r0 : r0 + side, c0 : c0 + side] = out_patch

    meta = {
        "side": side,
        "palette": palette,
        "signature": signature,
        "orientations": orientations,
        "complete_index": complete_index,
        "positions": positions,
        "hole_orientations": hole_orientations,
    }
    return grid_in, grid_out, meta


def _validate_instance(inp: np.ndarray, out: np.ndarray, meta: dict[str, object]) -> bool:
    try:
        if not np.array_equal(main(inp), out):
            return False
        comps = _find_components(inp != BLACK)
        if len(comps) != len(meta["orientations"]):
            return False
        full_count = 0
        side = int(meta["side"])
        for comp in comps:
            patch = _crop(inp, comp["bbox"])
            if patch.shape != (side, side):
                return False
            magenta = _signature_coords(patch)
            matches = 0
            for idx in range(8):
                if magenta == _transform_coords(set(meta["signature"]), side, idx):
                    matches += 1
            if matches != 1:
                return False
            if np.all(patch != BLACK):
                full_count += 1
        if full_count != 1:
            return False
        if len(set(int(x) for x in meta["hole_orientations"])) != len(meta["hole_orientations"]):
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_instance()
        if built is None:
            continue
        inp, out, meta = built
        if _validate_instance(inp, out, meta):
            return inp
    raise RuntimeError("failed to generate a valid puzzle-7 input")
