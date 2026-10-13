from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
CANVAS_SIDE = 30
MAX_GENERATION_ATTEMPTS = 400
TARGET_ATTEMPTS = 240
PLACEMENT_ATTEMPTS = 300


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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(np.any(mask, axis=1))
    cols = np.flatnonzero(np.any(mask, axis=0))
    if rows.size == 0 or cols.size == 0:
        return np.zeros((0, 0), dtype=bool)
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _component_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _is_rectangular_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * h + 2 * w - 4
    if len(coords) != expected:
        return False
    coord_set = set(coords)
    for c in range(c0, c1 + 1):
        if (r0, c) not in coord_set or (r1, c) not in coord_set:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in coord_set or (r, c1) not in coord_set:
            return False
    for r, c in coords:
        if r not in (r0, r1) and c not in (c0, c1):
            return False
    return True


def _matching_rotations(piece_mask: np.ndarray, target_mask: np.ndarray) -> list[int]:
    target = _crop_mask(target_mask)
    matches: list[int] = []
    for k in range(4):
        rotated = _crop_mask(np.rot90(piece_mask, k))
        if rotated.shape == target.shape and np.array_equal(rotated, target):
            matches.append(k)
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored fragment into the same-colored frame and fill the obstacle-shaped cavity.

    Input structure:
    - The background is black.
    - Several colors each appear twice: once as a 1-pixel-thick rectangular frame and once as a separate monochrome fragment.
    - Inside every frame, the interior is mostly black but contains one or two grey rectangular obstacle blocks.

    Rule:
    - Pair every fragment with the frame that has the same color.
    - Rotate the fragment by 0, 90, 180, or 270 degrees until its shape matches the frame's black interior cells exactly, excluding the grey obstacles.
    - Erase the fragment from its original position and repaint all matching interior black cells inside the frame with the fragment color.

    Key insight:
    - The frame color gives the pairing, while the grey blockers carve a unique cavity shape that determines the only valid fragment rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    found_any = False

    for color in [int(c) for c in np.unique(g) if int(c) not in (BLACK, GREY)]:
        comps = _components(g == color)
        if len(comps) < 2:
            continue

        frame_comp: list[tuple[int, int]] | None = None
        other_comps: list[list[tuple[int, int]]] = []
        for comp in comps:
            if _is_rectangular_frame(comp):
                if frame_comp is None or len(comp) > len(frame_comp):
                    frame_comp = comp
            else:
                other_comps.append(comp)

        if frame_comp is None or not other_comps:
            continue

        r0, c0, r1, c1 = _bbox(frame_comp)
        target_mask = g[r0 + 1 : r1, c0 + 1 : c1] == BLACK
        target_area = int(target_mask.sum())

        piece_candidates = [comp for comp in other_comps if len(comp) == target_area]
        if len(piece_candidates) == 1:
            piece_comp = piece_candidates[0]
        else:
            assert len(other_comps) == 1, "same-color inputs should contain one frame and one fragment"
            piece_comp = other_comps[0]

        piece_mask = _component_mask(piece_comp)
        matches = _matching_rotations(piece_mask, target_mask)
        assert matches, "fragment must match its frame cavity under some rotation"

        for r, c in piece_comp:
            out[r, c] = BLACK
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[target_mask] = color
        found_any = True

    assert found_any, "expected at least one frame-fragment pair"
    return out


def _rects_conflict(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return not (
        ar + ah - 1 + gap < br
        or br + bh - 1 + gap < ar
        or ac + aw - 1 + gap < bc
        or bc + bw - 1 + gap < ac
    )


def _self_match_count(mask: np.ndarray) -> int:
    base = _crop_mask(mask)
    count = 0
    for k in range(4):
        rotated = _crop_mask(np.rot90(mask, k))
        if rotated.shape == base.shape and np.array_equal(rotated, base):
            count += 1
    return count


def _make_target_mask(inner_h: int, inner_w: int, obstacle_count: int) -> np.ndarray | None:
    max_h = min(3, inner_h - 2)
    max_w = min(3, inner_w - 2)
    if max_h < 1 or max_w < 1:
        return None

    for _ in range(TARGET_ATTEMPTS):
        target = np.ones((inner_h, inner_w), dtype=bool)
        obstacles: list[tuple[int, int, int, int]] = []
        success = True
        for _ in range(obstacle_count):
            placed = False
            for _ in range(80):
                oh = random.randint(1, max_h)
                ow = random.randint(1, max_w)
                if inner_h - oh - 1 <= 0 or inner_w - ow - 1 <= 0:
                    continue
                top = random.randint(1, inner_h - oh - 1)
                left = random.randint(1, inner_w - ow - 1)
                rect = (top, left, oh, ow)
                if any(_rects_conflict(rect, other, gap=1) for other in obstacles):
                    continue
                obstacles.append(rect)
                target[top : top + oh, left : left + ow] = False
                placed = True
                break
            if not placed:
                success = False
                break
        if not success:
            continue
        if int(target.sum()) < 10:
            continue
        if len(_components(target)) != 1:
            continue
        if _self_match_count(target) != 1:
            continue
        return target
    return None


def _frame_patch(color: int, target_mask: np.ndarray, filled: bool) -> np.ndarray:
    inner_h, inner_w = target_mask.shape
    patch = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    interior = patch[1:-1, 1:-1]
    interior[~target_mask] = GREY
    if filled:
        interior[target_mask] = color
    return patch


def _piece_patch(color: int, target_mask: np.ndarray, rotation: int) -> np.ndarray:
    rotated = np.rot90(target_mask, rotation)
    return np.where(rotated, color, BLACK).astype(int)


def _can_place(reserved: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(reserved.shape[0], top + h + pad)
    c1 = min(reserved.shape[1], left + w + pad)
    return not reserved[r0:r1, c0:c1].any()


def _reserve(reserved: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(reserved.shape[0], top + h + pad)
    c1 = min(reserved.shape[1], left + w + pad)
    reserved[r0:r1, c0:c1] = True


def _blit_nonblack(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = canvas[top : top + h, left : left + w]
    mask = patch != BLACK
    region[mask] = patch[mask]


def _random_crop(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BLACK)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_pad = random.randint(0, min(3, int(r0)))
    left_pad = random.randint(0, min(3, int(c0)))
    bottom_pad = random.randint(0, min(3, int(inp.shape[0] - 1 - r1)))
    right_pad = random.randint(0, min(3, int(inp.shape[1] - 1 - c1)))
    rr0 = int(r0 - top_pad)
    cc0 = int(c0 - left_pad)
    rr1 = int(r1 + bottom_pad + 1)
    cc1 = int(c1 + right_pad + 1)
    return inp[rr0:rr1, cc0:cc1].copy(), out[rr0:rr1, cc0:cc1].copy()


def _build_pair(color: int) -> dict[str, np.ndarray | int] | None:
    for _ in range(TARGET_ATTEMPTS):
        inner_h = random.randint(4, 7)
        inner_w = random.randint(5, 8)
        if inner_h == inner_w:
            continue
        obstacle_count = random.choice([1, 2])
        target_mask = _make_target_mask(inner_h, inner_w, obstacle_count)
        if target_mask is None:
            continue
        rotation = random.randint(0, 3)
        return {
            "frame_input": _frame_patch(color, target_mask, filled=False),
            "frame_output": _frame_patch(color, target_mask, filled=True),
            "piece": _piece_patch(color, target_mask, rotation),
        }
    return None


def _place_patch_random(
    reserved: np.ndarray,
    patch: np.ndarray,
    min_gap: int,
) -> tuple[int, int] | None:
    h, w = patch.shape
    if h > reserved.shape[0] or w > reserved.shape[1]:
        return None
    for _ in range(PLACEMENT_ATTEMPTS):
        top = random.randint(0, reserved.shape[0] - h)
        left = random.randint(0, reserved.shape[1] - w)
        if _can_place(reserved, top, left, h, w, min_gap):
            _reserve(reserved, top, left, h, w, min_gap)
            return top, left
    return None


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = random.choice([2, 3])
    colors = random.sample(COLOR_POOL, pair_count)

    pairs: list[dict[str, np.ndarray | int]] = []
    for color in colors:
        pair = _build_pair(color)
        if pair is None:
            return None
        pairs.append(pair)

    pairs.sort(
        key=lambda item: -int(np.prod(np.asarray(item["frame_input"]).shape)),
    )

    inp = np.full((CANVAS_SIDE, CANVAS_SIDE), BLACK, dtype=int)
    out = np.full((CANVAS_SIDE, CANVAS_SIDE), BLACK, dtype=int)
    frame_reserved = np.zeros((CANVAS_SIDE, CANVAS_SIDE), dtype=bool)
    piece_reserved = np.zeros((CANVAS_SIDE, CANVAS_SIDE), dtype=bool)

    frame_reserved[:] = False
    piece_reserved[:] = False

    for pair in pairs:
        frame_input = np.asarray(pair["frame_input"])
        frame_output = np.asarray(pair["frame_output"])
        pos = _place_patch_random(frame_reserved, frame_input, min_gap=1)
        if pos is None:
            return None
        top, left = pos
        _blit_nonblack(inp, frame_input, top, left)
        _blit_nonblack(out, frame_output, top, left)
        _reserve(piece_reserved, top, left, frame_input.shape[0], frame_input.shape[1], pad=1)

    piece_order = pairs[:]
    random.shuffle(piece_order)
    for pair in piece_order:
        piece = np.asarray(pair["piece"])
        pos = _place_patch_random(piece_reserved, piece, min_gap=1)
        if pos is None:
            return None
        top, left = pos
        _blit_nonblack(inp, piece, top, left)

    cropped_inp, cropped_out = _random_crop(inp, out)
    if max(cropped_inp.shape) > 30 or max(cropped_out.shape) > 30:
        return None
    if not np.array_equal(main(cropped_inp), cropped_out):
        return None
    return cropped_inp, cropped_out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        generated = _gen_attempt()
        if generated is None:
            continue
        return generated[0]
    raise RuntimeError("failed to generate a valid input")
