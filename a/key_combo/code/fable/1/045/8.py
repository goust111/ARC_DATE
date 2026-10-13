from __future__ import annotations

import random
import sys
from itertools import combinations, permutations
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, crop_bbox, paste, random_sprite


CORNER_ORDER = ("tl", "tr", "br", "bl")
GENERATION_RETRIES = 400


def _corner_bbox(shape: tuple[int, int], side: int, corner: int) -> tuple[int, int, int, int]:
    h, w = shape
    side = int(side)
    if corner == 0:
        return 0, 0, side - 1, side - 1
    if corner == 1:
        return 0, w - side, side - 1, w - 1
    if corner == 2:
        return h - side, w - side, h - 1, w - 1
    if corner == 3:
        return h - side, 0, h - 1, side - 1
    raise ValueError("invalid corner")


def _extract_corner(grid: np.ndarray, side: int, corner: int) -> np.ndarray:
    return crop_bbox(grid, _corner_bbox(grid.shape, side, corner))


def _orient_to_canonical(patch: np.ndarray, corner: int) -> np.ndarray:
    if corner == 0:
        return patch.copy()
    if corner == 1:
        return np.fliplr(patch)
    if corner == 2:
        return np.flipud(np.fliplr(patch))
    if corner == 3:
        return np.flipud(patch)
    raise ValueError("invalid corner")


def _from_canonical(patch: np.ndarray, corner: int) -> np.ndarray:
    return _orient_to_canonical(patch, corner)


def _infer_corner_side(grid: np.ndarray) -> int:
    coords = np.argwhere(grid != Color.BLACK)
    if coords.size == 0:
        raise ValueError("expected non-black corner patterns")

    h, w = grid.shape
    d_tl = coords[:, 0] + coords[:, 1]
    d_tr = coords[:, 0] + (w - 1 - coords[:, 1])
    d_br = (h - 1 - coords[:, 0]) + (w - 1 - coords[:, 1])
    d_bl = (h - 1 - coords[:, 0]) + coords[:, 1]
    ds = np.stack([d_tl, d_tr, d_br, d_bl], axis=1)
    corner_idx = ds.argmin(axis=1)

    side_candidates: list[int] = []
    for corner in range(4):
        pts = coords[corner_idx == corner]
        if pts.size == 0:
            continue
        rs = pts[:, 0]
        cs = pts[:, 1]
        if corner == 0:
            k = max(int(rs.max() + 1), int(cs.max() + 1))
        elif corner == 1:
            k = max(int(rs.max() + 1), int(w - cs.min()))
        elif corner == 2:
            k = max(int(h - rs.min()), int(w - cs.min()))
        else:
            k = max(int(h - rs.min()), int(cs.max() + 1))
        side_candidates.append(k)

    if not side_candidates:
        raise ValueError("failed to infer corner size")
    return int(max(side_candidates))


def _infer_color_mapping(src: np.ndarray, dst: np.ndarray) -> dict[str, object] | None:
    overlap = (src != Color.BLACK) & (dst != Color.BLACK)
    if not np.any(overlap):
        return None

    mapping: dict[int, int] = {}
    inverse: dict[int, int] = {}
    pairs = set()
    for s, d in zip(src[overlap], dst[overlap]):
        s = int(s)
        d = int(d)
        pairs.add((s, d))
        if s in mapping and mapping[s] != d:
            return None
        if d in inverse and inverse[d] != s:
            return None
        mapping[s] = d
        inverse[d] = s
    return {
        "mapping": mapping,
        "inverse": inverse,
        "overlap": int(overlap.sum()),
        "unique_pairs": len(pairs),
    }


def _candidate_info(canonical_patches: list[np.ndarray], trio: tuple[int, int, int]) -> dict[str, object] | None:
    trio_patches = [canonical_patches[idx] for idx in trio]
    masks = np.stack([patch != Color.BLACK for patch in trio_patches], axis=0)
    support = masks.sum(axis=0)
    if not np.any(support >= 2):
        return None
    if np.any(support == 1):
        return None

    base_mask = support >= 2
    missing_counts = [int(np.count_nonzero(base_mask & ~masks[pos])) for pos in range(3)]
    if any(count == 0 for count in missing_counts):
        return None

    pair_maps: dict[tuple[int, int], dict[int, int]] = {}
    overlap_total = 0
    unique_pair_total = 0
    for a, b in combinations(range(3), 2):
        info = _infer_color_mapping(trio_patches[a], trio_patches[b])
        if info is None:
            return None
        mapping = info["mapping"]
        inverse = info["inverse"]
        if len(mapping) < 2:
            return None
        pair_maps[(a, b)] = mapping
        pair_maps[(b, a)] = inverse
        overlap_total += int(info["overlap"])
        unique_pair_total += int(info["unique_pairs"])

    composition_support = 0
    for a, b, c in permutations(range(3), 3):
        comp = pair_maps[(b, c)]
        direct = pair_maps[(a, c)]
        comparable = 0
        for src_color, mid_color in pair_maps[(a, b)].items():
            if mid_color in comp and src_color in direct:
                comparable += 1
                if comp[mid_color] != direct[src_color]:
                    return None
        composition_support += comparable

    best_ref = None
    best_ref_patch = None
    best_ref_score = -1
    for ref in range(3):
        ref_patch = np.zeros_like(trio_patches[ref])
        ok = True
        for r, c in np.argwhere(base_mask):
            converted: list[int] = []
            for pos, patch in enumerate(trio_patches):
                color = int(patch[r, c])
                if color == Color.BLACK:
                    continue
                if pos == ref:
                    converted.append(color)
                    continue
                back = pair_maps[(pos, ref)]
                if color not in back:
                    ok = False
                    break
                converted.append(int(back[color]))
            if not ok or len(converted) < 2 or len(set(converted)) != 1:
                ok = False
                break
            ref_patch[r, c] = converted[0]
        if not ok:
            continue

        ref_colors = {int(v) for v in ref_patch[base_mask]}
        if any(ref != pos and not ref_colors.issubset(pair_maps[(ref, pos)].keys()) for pos in range(3)):
            continue

        ref_score = (
            int(base_mask.sum()) * 100
            + overlap_total * 10
            + unique_pair_total * 5
            + composition_support
            + sum(missing_counts)
        )
        if ref_score > best_ref_score:
            best_ref = ref
            best_ref_patch = ref_patch
            best_ref_score = ref_score

    if best_ref is None or best_ref_patch is None:
        return None

    return {
        "trio": trio,
        "base_mask": base_mask,
        "pair_maps": pair_maps,
        "ref_pos": best_ref,
        "ref_patch": best_ref_patch,
        "score": best_ref_score,
    }


def _solve_metadata(grid: np.ndarray) -> dict[str, object]:
    side = _infer_corner_side(grid)
    corner_patches = [_extract_corner(grid, side, corner) for corner in range(4)]
    canonical = [_orient_to_canonical(patch, corner) for corner, patch in enumerate(corner_patches)]

    candidates = []
    for trio in combinations(range(4), 3):
        info = _candidate_info(canonical, trio)
        if info is not None:
            candidates.append(info)
    if not candidates:
        raise ValueError("no self-consistent genuine trio found")

    candidates.sort(key=lambda item: int(item["score"]), reverse=True)
    best = candidates[0]
    best_trio = set(best["trio"])
    impostor = next(corner for corner in range(4) if corner not in best_trio)

    ref_pos = int(best["ref_pos"])
    trio = tuple(int(v) for v in best["trio"])
    ref_patch = np.array(best["ref_patch"], dtype=int)
    base_mask = np.array(best["base_mask"], dtype=bool)
    pair_maps = best["pair_maps"]

    completed_canonical: dict[int, np.ndarray] = {}
    for pos, corner in enumerate(trio):
        full_patch = np.zeros_like(ref_patch)
        if pos == ref_pos:
            full_patch[base_mask] = ref_patch[base_mask]
        else:
            color_map = pair_maps[(ref_pos, pos)]
            for r, c in np.argwhere(base_mask):
                full_patch[r, c] = int(color_map[int(ref_patch[r, c])])
        completed_canonical[corner] = full_patch

    return {
        "side": side,
        "corner_patches": corner_patches,
        "canonical_patches": canonical,
        "genuine_trio": trio,
        "impostor": impostor,
        "completed_canonical": completed_canonical,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the three genuine corner-family patterns while leaving the incompatible complete corner unchanged.

    Input structure:
    - A square grid contains one same-size square pattern region in each corner, separated by black background.
    - Three corner regions come from one mother pattern after corner-dependent mirroring and a one-step color-cycle shift around the corner order TL->TR->BR->BL.
    - Those three genuine corners have black missing cells; the remaining corner is a complete impostor that does not fit the same mirror-and-cycle system.

    Rule:
    - Infer the common corner square size and extract the four corner patches.
    - Mirror every corner back to one canonical orientation, then test which three patches share one majority-supported shape and pairwise one-to-one color remappings that compose consistently.
    - Reconstruct the complete canonical mother-family patch from the genuine trio, map its colors back to each genuine corner, and keep the impostor patch and all black separator cells unchanged.

    Key insight:
    - Shape agreement alone is insufficient because matching cells cycle through colors from corner to corner; only the trio that is jointly consistent under both canonical mirroring and one global color permutation can explain every missing cell.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    try:
        info = _solve_metadata(g)
    except ValueError:
        return g.copy()

    side = int(info["side"])
    out = g.copy()
    for corner, patch in info["completed_canonical"].items():
        placed = _from_canonical(np.asarray(patch, dtype=int), int(corner))
        r0, c0, r1, c1 = _corner_bbox(g.shape, side, int(corner))
        out[r0 : r1 + 1, c0 : c1 + 1] = placed
    return out


def _sample_mask(side: int, rng: random.Random) -> np.ndarray:
    for _ in range(80):
        sprite = random_sprite(
            side,
            side,
            density=rng.uniform(0.45, 0.72),
            color_palette=[Color.BLUE],
            connectivity=4,
            background=Color.BLACK,
        )
        mask = sprite != Color.BLACK
        area = int(mask.sum())
        if area < side + 6:
            continue
        if np.any(mask[0]) and np.any(mask[-1]) and np.any(mask[:, 0]) and np.any(mask[:, -1]):
            return mask
    raise RuntimeError("failed to sample mask")


def _colorize_mask(mask: np.ndarray, cycle: list[int], rng: random.Random) -> np.ndarray:
    side = mask.shape[0]
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]

    formulas = []
    for a in (1, 2, 3):
        for b in (1, 2):
            for phase in range(len(cycle)):
                formulas.append((a, b, phase))
    rng.shuffle(formulas)

    for a, b, phase in formulas:
        patch = np.zeros((side, side), dtype=int)
        for r, c in coords:
            idx = (a * r + b * c + phase) % len(cycle)
            patch[r, c] = int(cycle[idx])
        used = {int(v) for v in patch[mask]}
        counts = {color: int(np.count_nonzero(patch == color)) for color in cycle}
        if used == set(cycle) and min(counts.values()) >= 2:
            return patch

    # Fallback: order cells by a visible diagonal sweep, then cycle colors.
    patch = np.zeros((side, side), dtype=int)
    coords.sort(key=lambda rc: (rc[0] + rc[1], rc[0], rc[1]))
    offset = rng.randrange(len(cycle))
    for idx, (r, c) in enumerate(coords):
        patch[r, c] = int(cycle[(idx + offset) % len(cycle)])
    return patch


def _shift_colors(patch: np.ndarray, cycle: list[int], steps: int) -> np.ndarray:
    out = patch.copy()
    positions = {color: idx for idx, color in enumerate(cycle)}
    for color, idx in positions.items():
        out[patch == color] = int(cycle[(idx + steps) % len(cycle)])
    return out


def _make_impostor(side: int, cycle: list[int], mother_mask: np.ndarray, rng: random.Random) -> np.ndarray:
    remaining = [color for color in Color.NOT_BLACK if color not in cycle]
    if len(remaining) < 2:
        remaining = list(Color.NOT_BLACK)

    for _ in range(120):
        mask = _sample_mask(side, rng)
        if np.array_equal(mask, mother_mask):
            continue
        palette_size = min(len(remaining), rng.randint(2, min(4, len(remaining))))
        palette = rng.sample(remaining, palette_size)
        patch = np.zeros((side, side), dtype=int)
        bias = rng.randrange(3)
        for r, c in np.argwhere(mask):
            patch[r, c] = int(palette[(int(r) * bias + int(c) + rng.randrange(len(palette))) % len(palette)])
        if np.count_nonzero(patch) == 0:
            continue
        return patch
    raise RuntimeError("failed to sample impostor")


def _assign_missing(mask: np.ndarray, genuine_corners: list[int], rng: random.Random) -> dict[int, set[tuple[int, int]]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    rng.shuffle(coords)
    missing = {corner: set() for corner in genuine_corners}

    for corner, cell in zip(genuine_corners, coords[: len(genuine_corners)]):
        missing[corner].add(cell)

    for cell in coords[len(genuine_corners) :]:
        choice = rng.choices([None, *genuine_corners], weights=[7, 1, 1, 1], k=1)[0]
        if choice is not None:
            missing[int(choice)].add(cell)
    return missing


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    side = rng.randint(5, 8)
    gap = rng.randint(2, 30 - 2 * side)
    size = 2 * side + gap
    if size > 30:
        return None

    cycle_len = rng.randint(3, 5)
    cycle = rng.sample(list(Color.NOT_BLACK), cycle_len)
    mother_mask = _sample_mask(side, rng)
    mother_patch = _colorize_mask(mother_mask, cycle, rng)

    impostor_corner = rng.randrange(4)
    genuine_corners = [corner for corner in range(4) if corner != impostor_corner]
    missing = _assign_missing(mother_mask, genuine_corners, rng)
    impostor_patch = _make_impostor(side, cycle, mother_mask, rng)

    canonical_patches: dict[int, np.ndarray] = {}
    for corner in range(4):
        if corner == impostor_corner:
            canonical_patches[corner] = impostor_patch.copy()
            continue
        patch = _shift_colors(mother_patch, cycle, corner)
        for r, c in missing[corner]:
            patch[r, c] = Color.BLACK
        canonical_patches[corner] = patch

    expected = np.zeros((size, size), dtype=int)
    inp = np.zeros((size, size), dtype=int)
    for corner in range(4):
        full_patch = impostor_patch if corner == impostor_corner else _shift_colors(mother_patch, cycle, corner)
        oriented_full = _from_canonical(full_patch, corner)
        oriented_inp = _from_canonical(canonical_patches[corner], corner)
        r0, c0, _r1, _c1 = _corner_bbox(inp.shape, side, corner)
        paste(expected, oriented_full, r0, c0, transparent=Color.BLACK)
        paste(inp, oriented_inp, r0, c0, transparent=Color.BLACK)

    if not _semantic_check(
        inp=inp,
        expected=expected,
        side=side,
        mother_mask=mother_mask,
        mother_patch=mother_patch,
        cycle=cycle,
        genuine_corners=genuine_corners,
        impostor_corner=impostor_corner,
        missing=missing,
    ):
        return None
    return inp, expected


def _semantic_check(
    *,
    inp: np.ndarray,
    expected: np.ndarray,
    side: int,
    mother_mask: np.ndarray,
    mother_patch: np.ndarray,
    cycle: list[int],
    genuine_corners: list[int],
    impostor_corner: int,
    missing: dict[int, set[tuple[int, int]]],
) -> bool:
    if _infer_corner_side(inp) != side:
        return False

    canonical = [_orient_to_canonical(_extract_corner(inp, side, corner), corner) for corner in range(4)]
    support = np.zeros_like(mother_mask, dtype=int)
    for corner in genuine_corners:
        patch = canonical[corner]
        support += (patch != Color.BLACK).astype(int)
        if not np.any((patch == Color.BLACK) & mother_mask):
            return False
        visible = {int(v) for v in patch[patch != Color.BLACK]}
        if visible != set(cycle):
            return False
    if not np.array_equal(support >= 2, mother_mask):
        return False
    if np.any(support[mother_mask] < 2):
        return False
    if not np.array_equal((canonical[impostor_corner] != Color.BLACK), mother_mask):
        pass
    else:
        impostor_info = _infer_color_mapping(canonical[genuine_corners[0]], canonical[impostor_corner])
        if impostor_info is not None and len(impostor_info["mapping"]) == len(cycle):
            return False

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    trio = set(_solve_metadata(inp)["genuine_trio"])
    if trio != set(genuine_corners):
        return False
    ref_patch = mother_patch
    return np.array_equal(ref_patch != Color.BLACK, mother_mask)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        built = _build_case(rng)
        if built is not None:
            inp, _expected = built
            return inp
    raise RuntimeError("failed to generate a valid corner-cycle impostor puzzle")
