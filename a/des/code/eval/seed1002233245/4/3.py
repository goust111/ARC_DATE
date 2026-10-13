from __future__ import annotations

import random
from typing import Any

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
PIECE_COLORS = [2, 4, 5, 6, 7, 8, 9]
GAP_TO_K = {"top": 0, "right": 3, "bottom": 2, "left": 1}
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
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
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    tight = _tight_mask(mask)
    return tight.shape, tight.astype(np.uint8).tobytes()


def _rotation_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants = [_mask_key(np.rot90(mask, k)) for k in range(4)]
    return min(variants)


def _self_rotation_count(mask: np.ndarray) -> int:
    tight = _tight_mask(mask)
    total = 0
    for k in range(4):
        rot = _tight_mask(np.rot90(tight, k))
        if rot.shape == tight.shape and np.array_equal(rot, tight):
            total += 1
    return total


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == BLUE)]
    assert len(rows) == 1
    return int(rows[0])


def _gap_side(patch_mask: np.ndarray) -> str:
    h, w = patch_mask.shape
    boundary_missing: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if not patch_mask[r, c]:
                boundary_missing.append((r, c))
    assert len(boundary_missing) == 1
    r, c = boundary_missing[0]
    if r == 0:
        return "top"
    if r == h - 1:
        return "bottom"
    if c == 0:
        return "left"
    return "right"


def _frame_info(top: np.ndarray) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    for coords in _components(top == GREEN):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        patch = top[r0 : r1 + 1, c0 : c1 + 1]
        mask = patch == GREEN
        h, w = mask.shape
        assert h >= 3 and w >= 3

        outer = np.zeros_like(mask, dtype=bool)
        outer[0, :] = True
        outer[-1, :] = True
        outer[:, 0] = True
        outer[:, -1] = True
        missing = outer & ~mask
        assert int(missing.sum()) == 1

        cavity_full = ~mask[1:-1, 1:-1]
        comps = _components(cavity_full)
        assert len(comps) == 1
        cavity_coords = comps[0]
        ir0, ic0, ir1, ic1 = _bbox_from_coords(cavity_coords)
        cavity = cavity_full[ir0 : ir1 + 1, ic0 : ic1 + 1].astype(bool)

        infos.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap_side": _gap_side(mask),
                "cavity": cavity,
                "inner_bbox": (ir0, ic0, ir1, ic1),
            }
        )
    assert infos
    return infos


def _piece_info(bottom: np.ndarray) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    comps = _components(bottom != BLACK)
    assert comps
    for coords in comps:
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        patch = bottom[r0 : r1 + 1, c0 : c1 + 1]
        colors = [int(v) for v in np.unique(patch) if int(v) != BLACK]
        assert len(colors) == 1
        color = colors[0]
        mask = patch == color
        assert len(_components(mask)) == 1
        infos.append({"bbox": (r0, c0, r1, c1), "color": color, "mask": mask})
    assert infos
    return infos


def _matching_pairs(frames: list[dict[str, Any]], pieces: list[dict[str, Any]]) -> dict[int, int]:
    assert len(frames) == len(pieces)
    n = len(frames)
    targets = [GAP_TO_K[frame["gap_side"]] for frame in frames]
    candidates: dict[int, list[int]] = {}
    for i, frame in enumerate(frames):
        need = frame["cavity"]
        rot_k = targets[i]
        matches: list[int] = []
        for j, piece in enumerate(pieces):
            got = np.rot90(piece["mask"], rot_k)
            if got.shape == need.shape and np.array_equal(got, need):
                matches.append(j)
        assert matches
        candidates[i] = matches

    order = sorted(range(n), key=lambda idx: (len(candidates[idx]), idx))
    used: set[int] = set()
    assignment: dict[int, int] = {}

    def backtrack(pos: int) -> bool:
        if pos == n:
            return True
        frame_idx = order[pos]
        for piece_idx in candidates[frame_idx]:
            if piece_idx in used:
                continue
            used.add(piece_idx)
            assignment[frame_idx] = piece_idx
            if backtrack(pos + 1):
                return True
            used.remove(piece_idx)
            del assignment[frame_idx]
        return False

    assert backtrack(0)
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble fragmented pieces into frame cavities, using the frame gap side as a rotation code.

    Input structure:
    - The background is black.
    - A blue horizontal line splits the grid into a top frame region and a bottom piece region.
    - Each top object is a green rectangular frame outline with exactly one missing border pixel.
    - The frame interior may contain extra green walls, leaving one connected black cavity.
    - Each bottom object is a solid colored piece whose shape matches one cavity up to rotation.

    Rule:
    - Pair pieces with frame cavities by shape.
    - Decode rotation from the frame gap: top=0, right=90 clockwise, bottom=180, left=270 clockwise.
    - Rotate the matched piece by that amount, place it into the cavity, keep its color,
      and clear its original bottom position to black.
    """
    g = np.asarray(grid, dtype=int)
    sep = _separator_row(g)
    top = g[:sep, :]
    bottom = g[sep + 1 :, :]

    frames = _frame_info(top)
    pieces = _piece_info(bottom)
    assignment = _matching_pairs(frames, pieces)

    out = g.copy()
    out[sep + 1 :, :][bottom != BLACK] = BLACK

    for frame_idx, piece_idx in assignment.items():
        frame = frames[frame_idx]
        piece = pieces[piece_idx]
        rot_k = GAP_TO_K[frame["gap_side"]]
        rotated = np.rot90(piece["mask"], rot_k)
        cavity = frame["cavity"]
        assert rotated.shape == cavity.shape and np.array_equal(rotated, cavity)

        fr0, fc0, _fr1, _fc1 = frame["bbox"]
        ir0, ic0, _ir1, _ic1 = frame["inner_bbox"]
        rr0 = fr0 + 1 + ir0
        cc0 = fc0 + 1 + ic0
        patch = out[rr0 : rr0 + cavity.shape[0], cc0 : cc0 + cavity.shape[1]]
        patch[rotated] = int(piece["color"])

    return out


def _rectangle_mask() -> np.ndarray:
    h = random.randint(1, 4)
    w = random.randint(1, 4)
    if h == 1 and w == 1:
        w = 2
    return np.ones((h, w), dtype=bool)


def _line_mask() -> np.ndarray:
    if random.random() < 0.5:
        return np.ones((1, random.randint(2, 5)), dtype=bool)
    return np.ones((random.randint(2, 5), 1), dtype=bool)


def _l_mask() -> np.ndarray:
    h = random.randint(2, 5)
    w = random.randint(2, 5)
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[0, :] = True
    return mask


def _t_mask() -> np.ndarray:
    h = random.randint(2, 5)
    w = random.choice([3, 5])
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[:, w // 2] = True
    return mask


def _plus_mask() -> np.ndarray:
    size = random.choice([3, 5])
    mask = np.zeros((size, size), dtype=bool)
    mid = size // 2
    mask[mid, :] = True
    mask[:, mid] = True
    return mask


def _u_mask() -> np.ndarray:
    h = random.randint(3, 5)
    w = random.randint(3, 5)
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[:, -1] = True
    mask[-1, :] = True
    return mask


def _c_mask() -> np.ndarray:
    h = random.randint(3, 5)
    w = random.randint(3, 5)
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[0, :] = True
    mask[-1, :] = True
    return mask


def _stair_mask() -> np.ndarray:
    steps = random.randint(2, 4)
    mask = np.zeros((steps, steps + 1), dtype=bool)
    for i in range(steps):
        mask[i, i] = True
        mask[i, i + 1] = True
    return mask


def _sample_shape(force_symmetric: bool) -> np.ndarray:
    if force_symmetric:
        maker = random.choice([_rectangle_mask, _line_mask, _plus_mask])
    else:
        maker = random.choice(
            [_rectangle_mask, _line_mask, _l_mask, _t_mask, _plus_mask, _u_mask, _c_mask, _stair_mask]
        )
    mask = maker()
    mask = _tight_mask(mask)
    if random.random() < 0.75:
        mask = np.rot90(mask, random.randint(0, 3))
    return _tight_mask(mask)


def _green_connected(frame_patch: np.ndarray) -> bool:
    return len(_components(frame_patch == GREEN)) == 1


def _build_frame_and_piece(
    cavity: np.ndarray, color: int, gap_side: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ch, cw = cavity.shape
    pad_t = 0 if gap_side == "top" else random.randint(0, 2)
    pad_b = 0 if gap_side == "bottom" else random.randint(0, 2)
    pad_l = 0 if gap_side == "left" else random.randint(0, 2)
    pad_r = 0 if gap_side == "right" else random.randint(0, 2)

    inner_h = ch + pad_t + pad_b
    inner_w = cw + pad_l + pad_r
    inner = np.zeros((inner_h, inner_w), dtype=bool)
    inner[pad_t : pad_t + ch, pad_l : pad_l + cw] = cavity

    frame = np.full((inner_h + 2, inner_w + 2), GREEN, dtype=int)
    frame[1:-1, 1:-1][inner] = BLACK

    if gap_side == "top":
        choices = np.where(cavity[0, :])[0]
        gap_pos = 1 + pad_l + int(random.choice(choices.tolist()))
        frame[0, gap_pos] = BLACK
    elif gap_side == "bottom":
        choices = np.where(cavity[-1, :])[0]
        gap_pos = 1 + pad_l + int(random.choice(choices.tolist()))
        frame[-1, gap_pos] = BLACK
    elif gap_side == "left":
        choices = np.where(cavity[:, 0])[0]
        gap_pos = 1 + pad_t + int(random.choice(choices.tolist()))
        frame[gap_pos, 0] = BLACK
    else:
        choices = np.where(cavity[:, -1])[0]
        gap_pos = 1 + pad_t + int(random.choice(choices.tolist()))
        frame[gap_pos, -1] = BLACK

    assert _green_connected(frame)

    piece_mask = np.rot90(cavity, (-GAP_TO_K[gap_side]) % 4)
    piece = np.where(piece_mask, color, BLACK).astype(int)

    spec = {
        "gap_side": gap_side,
        "cavity": cavity,
        "inner_offset": (pad_t, pad_l),
        "color": color,
    }
    return frame, piece, spec


def _place_rectangles(region_h: int, region_w: int, patches: list[np.ndarray], padding: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    occupied = np.zeros((region_h, region_w), dtype=bool)
    positions: dict[int, tuple[int, int]] = {}
    for idx in order:
        ph, pw = patches[idx].shape
        candidates: list[tuple[int, int]] = []
        for r in range(region_h - ph + 1):
            for c in range(region_w - pw + 1):
                rr0 = max(0, r - padding)
                cc0 = max(0, c - padding)
                rr1 = min(region_h, r + ph + padding)
                cc1 = min(region_w, c + pw + padding)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((r, c))
        if not candidates:
            return None
        pos = random.choice(candidates)
        positions[idx] = pos
        r, c = pos
        occupied[r : r + ph, c : c + pw] = True
    return [positions[i] for i in range(len(patches))]


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch != BLACK
    sub = dst[r0 : r0 + h, c0 : c0 + w]
    sub[mask] = patch[mask]


def _expected_output(
    inp: np.ndarray,
    sep: int,
    frame_positions: list[tuple[int, int]],
    piece_positions: list[tuple[int, int]],
    frame_specs: list[dict[str, Any]],
    piece_patches: list[np.ndarray],
) -> np.ndarray:
    out = inp.copy()
    out[sep + 1 :, :][inp[sep + 1 :, :] != BLACK] = BLACK
    for idx, (fr, fc) in enumerate(frame_positions):
        spec = frame_specs[idx]
        cavity = spec["cavity"]
        pad_t, pad_l = spec["inner_offset"]
        rr0 = fr + 1 + pad_t
        cc0 = fc + 1 + pad_l
        out[rr0 : rr0 + cavity.shape[0], cc0 : cc0 + cavity.shape[1]][cavity] = spec["color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(2, 4)
    colors = random.sample(PIECE_COLORS, count)

    shapes: list[np.ndarray] = []
    signatures: set[tuple[tuple[int, int], bytes]] = set()
    forced_sym = random.randrange(count)
    for idx in range(count):
        for _ in range(80):
            mask = _sample_shape(force_symmetric=idx == forced_sym)
            if int(mask.sum()) < 2:
                continue
            sig = _rotation_signature(mask)
            if sig in signatures:
                continue
            shapes.append(mask)
            signatures.add(sig)
            break
        else:
            return None

    if not any(_self_rotation_count(mask) > 1 for mask in shapes):
        return None

    gap_sides = [random.choice(["top", "right", "bottom", "left"]) for _ in range(count)]
    frame_patches: list[np.ndarray] = []
    piece_patches: list[np.ndarray] = []
    frame_specs: list[dict[str, Any]] = []
    for idx in range(count):
        frame_patch, piece_patch, spec = _build_frame_and_piece(shapes[idx], colors[idx], gap_sides[idx])
        frame_patches.append(frame_patch)
        piece_patches.append(piece_patch)
        frame_specs.append(spec)

    upper_h = random.randint(max(10, max(p.shape[0] for p in frame_patches) + 2), 18)
    lower_h = random.randint(max(8, max(p.shape[0] for p in piece_patches) + 2), 16)
    min_width = max(max(p.shape[1] for p in frame_patches), max(p.shape[1] for p in piece_patches)) + 4
    width = random.randint(max(14, min_width), 24)

    frame_positions = _place_rectangles(upper_h, width, frame_patches, padding=1)
    piece_positions = _place_rectangles(lower_h, width, piece_patches, padding=1)
    if frame_positions is None or piece_positions is None:
        return None

    upper = np.full((upper_h, width), BLACK, dtype=int)
    lower = np.full((lower_h, width), BLACK, dtype=int)
    for patch, (r, c) in zip(frame_patches, frame_positions, strict=False):
        _paste(upper, patch, r, c)
    for patch, (r, c) in zip(piece_patches, piece_positions, strict=False):
        _paste(lower, patch, r, c)

    sep = upper_h
    inp = np.vstack([upper, np.full((1, width), BLUE, dtype=int), lower])
    expected = _expected_output(inp, sep, frame_positions, piece_positions, frame_specs, piece_patches)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
