from __future__ import annotations

import numpy as np


BLACK = 0
FRAME_COLOR = 5
SPRITE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _rotation_masks(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen = set()
    for k in range(4):
        rot = _trim(np.rot90(mask, k))
        key = (tuple(map(int, rot.shape)), rot.tobytes())
        if key not in seen:
            seen.add(key)
            out.append(rot)
    return out


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    keys = []
    for k in range(4):
        rot = _trim(np.rot90(mask, k))
        keys.append((tuple(map(int, rot.shape)), rot.tobytes()))
    return min(keys)


def _bit_positions(bits: int):
    while bits:
        lsb = bits & -bits
        idx = lsb.bit_length() - 1
        yield idx
        bits ^= lsb


def _mask_to_bits(mask: np.ndarray, top: int, left: int, width: int) -> int:
    bits = 0
    for r, c in np.argwhere(mask):
        idx = (top + int(r)) * width + left + int(c)
        bits |= 1 << idx
    return bits


def _placed_coords(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    pts = np.argwhere(mask)
    pts = pts + np.array([top, left], dtype=int)
    return pts.astype(int)


def _infer_frame(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    candidates = []
    for color in map(int, np.unique(g)):
        if color == BLACK:
            continue
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all((patch == color) | (patch == BLACK)):
            continue
        if not np.all(patch[0, :] == color):
            continue
        if not np.all(patch[-1, :] == color):
            continue
        if not np.all(patch[:, 0] == color):
            continue
        if not np.all(patch[:, -1] == color):
            continue
        if not np.any(patch[1:-1, 1:-1] == BLACK):
            continue
        candidates.append((h * w, int(np.sum(patch == color)), color, (r0, c0, r1, c1)))
    assert candidates, "frame not found"
    candidates.sort(reverse=True)
    _area, _count, color, box = candidates[0]
    return color, box


def _extract_sprites(grid: np.ndarray, frame_color: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    sprites = []
    for color in sorted(set(map(int, np.unique(g))) - {BLACK, frame_color}):
        coords = np.argwhere(g == color)
        assert coords.size > 0, "sprite color must be present"
        assert len(_components(g == color)) == 1, "each sprite color must form one component"
        mask = _coords_to_mask(coords)
        sprites.append(
            {
                "color": color,
                "area": int(coords.shape[0]),
                "mask": mask,
                "bbox": _bbox(coords),
            }
        )
    assert len(sprites) >= 2, "need multiple sprites"
    return sprites


def _enumerate_candidates(union_mask: np.ndarray, sprite_mask: np.ndarray) -> list[int]:
    frame_h, frame_w = union_mask.shape
    candidates: list[int] = []
    seen = set()
    for rot in _rotation_masks(sprite_mask):
        h, w = rot.shape
        pts = np.argwhere(rot)
        for top in range(frame_h - h + 1):
            for left in range(frame_w - w + 1):
                ok = True
                bits = 0
                for r, c in pts:
                    rr = top + int(r)
                    cc = left + int(c)
                    if not union_mask[rr, cc]:
                        ok = False
                        break
                    bits |= 1 << (rr * frame_w + cc)
                if ok and bits not in seen:
                    seen.add(bits)
                    candidates.append(bits)
    return candidates


def _solve_placements(union_mask: np.ndarray, sprites: list[dict]) -> list[int]:
    frame_h, frame_w = union_mask.shape
    union_bits = _mask_to_bits(union_mask, 0, 0, frame_w)
    data = []
    for idx, sprite in enumerate(sprites):
        candidates = _enumerate_candidates(union_mask, sprite["mask"])
        assert candidates, "sprite has no matching hole placement"
        cover = 0
        for bits in candidates:
            cover |= int(bits)
        data.append(
            {
                "orig_idx": idx,
                "area": int(sprite["area"]),
                "candidates": candidates,
                "cover": cover,
            }
        )
    order = sorted(range(len(data)), key=lambda i: (len(data[i]["candidates"]), -data[i]["area"]))
    ordered = [data[i] for i in order]

    suffix_cover = [0] * (len(ordered) + 1)
    for i in range(len(ordered) - 1, -1, -1):
        suffix_cover[i] = suffix_cover[i + 1] | ordered[i]["cover"]

    solutions: list[list[int]] = []
    chosen = [0] * len(ordered)

    def dfs(pos: int, covered: int) -> None:
        if len(solutions) >= 2:
            return
        if (covered | suffix_cover[pos]) != union_bits:
            return
        if pos == len(ordered):
            if covered == union_bits:
                placed = [0] * len(sprites)
                for i, item in enumerate(ordered):
                    placed[item["orig_idx"]] = chosen[i]
                solutions.append(placed)
            return
        candidates = sorted(
            ordered[pos]["candidates"],
            key=lambda bits: (-int((bits & ~covered).bit_count()), bits),
        )
        for bits in candidates:
            new_covered = covered | bits
            if (new_covered | suffix_cover[pos + 1]) != union_bits:
                continue
            chosen[pos] = bits
            dfs(pos + 1, new_covered)

    dfs(0, 0)
    assert len(solutions) == 1, "hole decomposition must be unique"
    return solutions[0]


def _render_output(
    frame_patch: np.ndarray,
    frame_color: int,
    sprites: list[dict],
    placements: list[int],
) -> np.ndarray:
    out = frame_patch.copy()
    height, width = out.shape
    order = sorted(range(len(sprites)), key=lambda i: int(sprites[i]["area"]))
    for idx in order:
        color = int(sprites[idx]["color"])
        for flat in _bit_positions(int(placements[idx])):
            r = flat // width
            c = flat % width
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match outside sprites to overlapping holes inside a grey frame, then fill by sprite area.

    The black cells inside the frame are the union of several rotated hole shapes.
    Each outside monochrome sprite matches one hidden hole shape up to rotation.
    Smaller sprites are painted first and larger sprites later, so larger areas win
    on overlapping cells.
    """
    g = np.asarray(grid, dtype=int)
    frame_color, (r0, c0, r1, c1) = _infer_frame(g)
    frame_patch = g[r0 : r1 + 1, c0 : c1 + 1]
    union_mask = frame_patch == BLACK
    sprites = _extract_sprites(g, frame_color)
    placements = _solve_placements(union_mask, sprites)
    return _render_output(frame_patch, frame_color, sprites, placements)


def _random_shape(rng: np.random.Generator, area: int) -> np.ndarray | None:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            pick = frontier[int(rng.integers(len(frontier)))]
            cells.add(pick)
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        min_r = min(rs)
        min_c = min(cs)
        norm = sorted((r - min_r, c - min_c) for r, c in cells)
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if h < 2 or w < 2 or h > 5 or w > 5:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[r, c] = True
        if area == h * w:
            continue
        if len(_rotation_masks(mask)) == 1:
            continue
        return mask
    return None


def _candidate_hole_bits(mask: np.ndarray, frame_h: int, frame_w: int) -> list[int]:
    h, w = mask.shape
    pts = np.argwhere(mask)
    candidates = []
    for top in range(1, frame_h - h):
        for left in range(1, frame_w - w):
            bits = 0
            for r, c in pts:
                rr = top + int(r)
                cc = left + int(c)
                bits |= 1 << (rr * frame_w + cc)
            candidates.append(bits)
    return candidates


def _choose_hole_placements(
    rng: np.random.Generator,
    hole_masks: list[np.ndarray],
    frame_h: int,
    frame_w: int,
) -> list[int] | None:
    candidate_lists = [_candidate_hole_bits(mask, frame_h, frame_w) for mask in hole_masks]
    if any(not lst for lst in candidate_lists):
        return None
    for _ in range(500):
        chosen = [int(lst[int(rng.integers(len(lst)))]) for lst in candidate_lists]
        pair_overlaps = []
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                pair_overlaps.append((i, j, int((chosen[i] & chosen[j]).bit_count())))
        if not pair_overlaps or max(v for _, _, v in pair_overlaps) == 0:
            continue
        if any(v >= min(int(hole_masks[i].sum()), int(hole_masks[j].sum())) for i, j, v in pair_overlaps):
            continue
        ok = True
        for i, bits in enumerate(chosen):
            others = 0
            for j, other in enumerate(chosen):
                if i != j:
                    others |= other
            if int((bits & ~others).bit_count()) < 2:
                ok = False
                break
        if not ok:
            continue
        union_bits = 0
        for bits in chosen:
            union_bits |= bits
        if int(union_bits.bit_count()) > (frame_h * frame_w) // 2:
            continue
        return chosen
    return None


def _bits_to_mask(bits: int, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for idx in _bit_positions(bits):
        mask[idx // width, idx % width] = True
    return mask


def _place_sprites_outside(
    rng: np.random.Generator,
    grid: np.ndarray,
    frame_box: tuple[int, int, int, int],
    sprite_masks: list[np.ndarray],
    colors: list[int],
) -> bool:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = frame_box
    occupied[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True
    for mask, color in zip(sprite_masks, colors, strict=False):
        sh, sw = mask.shape
        candidates = []
        for top in range(h - sh + 1):
            for left in range(w - sw + 1):
                rr0 = max(0, top - 1)
                cc0 = max(0, left - 1)
                rr1 = min(h, top + sh + 1)
                cc1 = min(w, left + sw + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((top, left))
        if not candidates:
            return False
        top, left = candidates[int(rng.integers(len(candidates)))]
        pts = np.argwhere(mask)
        for r, c in pts:
            grid[top + int(r), left + int(c)] = int(color)
        occupied[max(0, top - 1) : min(h, top + sh + 1), max(0, left - 1) : min(w, left + sw + 1)] = True
    return True


def _expected_output(
    frame_h: int,
    frame_w: int,
    hole_bits: list[int],
    frame_color: int,
    colors: list[int],
    areas: list[int],
) -> np.ndarray:
    out = np.full((frame_h, frame_w), frame_color, dtype=int)
    order = sorted(range(len(hole_bits)), key=lambda i: int(areas[i]))
    for idx in order:
        for flat in _bit_positions(int(hole_bits[idx])):
            out[flat // frame_w, flat % frame_w] = int(colors[idx])
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.choice([2, 3]))
    areas = sorted(rng.choice(np.arange(4, 10), size=n, replace=False).tolist())
    shapes = []
    keys = set()
    for area in areas:
        shape = _random_shape(rng, int(area))
        if shape is None:
            return None
        key = _canonical_key(shape)
        if key in keys:
            return None
        keys.add(key)
        shapes.append(shape)

    max_h = max(int(mask.shape[0]) for mask in shapes)
    max_w = max(int(mask.shape[1]) for mask in shapes)
    frame_h = int(rng.integers(max(9, max_h + 5), max(13, max_h + 9)))
    frame_w = int(rng.integers(max(9, max_w + 5), max(13, max_w + 9)))

    hole_masks = []
    ext_masks = []
    for shape in shapes:
        hole_masks.append(_trim(np.rot90(shape, int(rng.integers(4)))))
        ext_masks.append(_trim(np.rot90(shape, int(rng.integers(4)))))

    hole_bits = _choose_hole_placements(rng, hole_masks, frame_h, frame_w)
    if hole_bits is None:
        return None

    union_bits = 0
    for bits in hole_bits:
        union_bits |= bits
    union_mask = _bits_to_mask(union_bits, frame_h, frame_w)
    if not np.any(union_mask[1:-1, 1:-1]):
        return None

    extra_h = int(rng.integers(8, 13))
    extra_w = int(rng.integers(8, 13))
    grid_h = frame_h + extra_h
    grid_w = frame_w + extra_w
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    frame_top = int(rng.integers(2, grid_h - frame_h - 1))
    frame_left = int(rng.integers(2, grid_w - frame_w - 1))
    frame_box = (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)
    grid[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = FRAME_COLOR
    frame_patch = grid[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w]
    frame_patch[union_mask] = BLACK

    colors = rng.choice(SPRITE_COLORS, size=n, replace=False).tolist()
    if not _place_sprites_outside(rng, grid, frame_box, ext_masks, [int(c) for c in colors]):
        return None

    expected = _expected_output(frame_h, frame_w, hole_bits, FRAME_COLOR, colors, areas)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
