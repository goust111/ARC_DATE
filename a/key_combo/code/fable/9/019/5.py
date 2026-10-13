from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_SIDE = 30
GEN_RETRIES = 500
PLACE_RETRIES = 120
MIN_GAP = 2


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: list[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_mono_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            bbox = _bbox_from_coords(coords)
            comps.append({"color": color, "coords": coords, "bbox": bbox, "area": len(coords)})
    return comps


def _decode_copies(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_mono_components(g, bg)
    assert comps, "expected foreground blocks"

    by_scale: dict[int, list[dict]] = {}
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert height == width, "each visible fragment must be a square block"
        assert comp["area"] == height * width, "each visible fragment must be a solid square"
        by_scale.setdefault(height, []).append(comp)

    copies: list[dict] = []
    prototype_shape: tuple[int, int] | None = None
    for scale, parts in sorted(by_scale.items()):
        min_r = min(comp["bbox"][0] for comp in parts)
        min_c = min(comp["bbox"][1] for comp in parts)
        max_r = max(comp["bbox"][2] for comp in parts)
        max_c = max(comp["bbox"][3] for comp in parts)

        assert (max_r - min_r + 1) % scale == 0
        assert (max_c - min_c + 1) % scale == 0
        proto_h = (max_r - min_r + 1) // scale
        proto_w = (max_c - min_c + 1) // scale
        partial = np.full((proto_h, proto_w), bg, dtype=int)

        for comp in parts:
            r0, c0, r1, c1 = comp["bbox"]
            assert r1 - r0 + 1 == scale and c1 - c0 + 1 == scale
            assert (r0 - min_r) % scale == 0 and (c0 - min_c) % scale == 0
            rr = (r0 - min_r) // scale
            cc = (c0 - min_c) // scale
            assert partial[rr, cc] == bg or partial[rr, cc] == int(comp["color"])
            partial[rr, cc] = int(comp["color"])

        assert partial[0, 0] != bg and partial[0, -1] != bg
        assert partial[-1, 0] != bg and partial[-1, -1] != bg
        assert np.any(partial == bg), "no copy may already be complete"

        if prototype_shape is None:
            prototype_shape = partial.shape
        else:
            assert partial.shape == prototype_shape, "all scales must come from one prototype"

        copies.append(
            {
                "scale": scale,
                "origin": (min_r, min_c),
                "partial": partial,
                "bbox": (min_r, min_c, max_r, max_c),
            }
        )

    assert len(copies) >= 2, "expected several complementary copies"
    return copies


def _reconstruct_prototype(copies: list[dict], bg: int) -> np.ndarray:
    proto = np.full(copies[0]["partial"].shape, bg, dtype=int)
    cover = np.zeros_like(proto, dtype=int)
    for copy in copies:
        part = copy["partial"]
        mask = part != bg
        conflict = mask & (proto != bg) & (proto != part)
        assert not np.any(conflict), "complementary fragments must agree where they overlap"
        proto[mask] = part[mask]
        cover += mask.astype(int)
    assert np.all(cover > 0), "fragments must cover the full prototype together"
    assert np.all(proto != bg), "prototype is expected to be fully colored"
    return proto


def _paint_scaled_block(grid: np.ndarray, top: int, left: int, scale: int, color: int) -> None:
    grid[top : top + scale, left : left + scale] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden prototype across scales and use it to complete every partial enlarged copy.

    Input structure:
    - A background canvas contains several scattered copies of one small multicolor prototype.
    - Each copy is enlarged by its own integer scale, so every visible prototype pixel becomes a same-color solid square block.
    - Every copy is incomplete: some whole blocks are missing as background, but each copy keeps all four corner blocks and the different copies use distinct scales.

    Rule:
    - Read each copy's scale from the side length of its visible solid square blocks.
    - Shrink every copy back to prototype resolution by mapping each visible block to one prototype cell inside that copy's bounding box.
    - Overlay the shrunken fragments, aligned by their bounding boxes, to reconstruct the full prototype.
    - For each copy, whenever the reconstructed prototype has a color but that copy is missing the corresponding block, fill in the entire scaled block with that color.

    Key insight:
    - No single copy is complete; the answer only becomes available after normalizing all fragmentary copies to a common scale and letting them complement one another.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    copies = _decode_copies(g)
    prototype = _reconstruct_prototype(copies, bg)

    out = g.copy()
    for copy in copies:
        scale = int(copy["scale"])
        r0, c0 = copy["origin"]
        part = copy["partial"]
        missing = part == bg
        for rr, cc in zip(*np.where(missing)):
            color = int(prototype[rr, cc])
            _paint_scaled_block(out, r0 + rr * scale, c0 + cc * scale, scale, color)
    return out


def _random_prototype(rng: random.Random, h: int, w: int) -> np.ndarray:
    while True:
        color_count = rng.randint(3, min(5, len(PALETTE)))
        colors = rng.sample(PALETTE, color_count)
        proto = np.zeros((h, w), dtype=int)
        for r in range(h):
            for c in range(w):
                forbidden = set()
                if r > 0:
                    forbidden.add(int(proto[r - 1, c]))
                if c > 0:
                    forbidden.add(int(proto[r, c - 1]))
                choices = [color for color in colors if color not in forbidden]
                if not choices:
                    choices = [color for color in PALETTE if color not in forbidden]
                proto[r, c] = rng.choice(choices)

        if len(set(map(int, proto.ravel()))) < 3:
            continue
        corner_colors = {int(proto[0, 0]), int(proto[0, -1]), int(proto[-1, 0]), int(proto[-1, -1])}
        if len(corner_colors) < 3:
            continue
        if np.array_equal(proto, np.flipud(proto)):
            continue
        if np.array_equal(proto, np.fliplr(proto)):
            continue
        if np.array_equal(proto, np.rot90(proto, 2)):
            continue
        return proto


def _choose_scales(rng: random.Random, copy_count: int) -> list[int]:
    candidates = {
        2: [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
        3: [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)],
        4: [(1, 2, 3, 4)],
    }
    return list(rng.choice(candidates[copy_count]))


def _non_corner_cells(h: int, w: int) -> list[Coord]:
    return [(r, c) for r in range(h) for c in range(w) if (r, c) not in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}]


def _presence_masks(rng: random.Random, proto: np.ndarray, copy_count: int) -> np.ndarray | None:
    h, w = proto.shape
    cells = _non_corner_cells(h, w)
    if len(cells) < copy_count:
        return None

    presence = np.zeros((copy_count, h, w), dtype=bool)
    presence[:, 0, 0] = True
    presence[:, 0, -1] = True
    presence[:, -1, 0] = True
    presence[:, -1, -1] = True

    rng.shuffle(cells)
    essential = cells[:copy_count]
    for idx, (r, c) in enumerate(essential):
        presence[idx, r, c] = True

    for r, c in cells[copy_count:]:
        if copy_count == 2:
            subset_size = rng.choices([1, 2], weights=[4, 3])[0]
        else:
            subset_size = rng.choices(
                list(range(1, copy_count + 1)),
                weights=[5] + [3] * (copy_count - 2) + [2],
            )[0]
        owners = rng.sample(range(copy_count), subset_size)
        for idx in owners:
            presence[idx, r, c] = True

    total = h * w
    coverage = presence.sum(axis=0)
    if np.any(coverage == 0):
        return None
    for idx in range(copy_count):
        shown = int(presence[idx].sum())
        missing = total - shown
        if missing < 1:
            return None
        if shown < max(4, total // 3):
            return None
        if np.any(presence[idx].sum(axis=0) == 0):
            return None
        if np.any(presence[idx].sum(axis=1) == 0):
            return None
        others = np.any(np.delete(presence, idx, axis=0), axis=0)
        er, ec = essential[idx]
        if others[er, ec]:
            return None
    return presence


def _render_partial(proto: np.ndarray, scale: int, mask: np.ndarray, bg: int) -> np.ndarray:
    h, w = proto.shape
    patch = np.full((h * scale, w * scale), bg, dtype=int)
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                _paint_scaled_block(patch, r * scale, c * scale, scale, int(proto[r, c]))
    return patch


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return not (
        ar + ah + gap <= br
        or br + bh + gap <= ar
        or ac + aw + gap <= bc
        or bc + bw + gap <= ac
    )


def _place_rectangles(rng: random.Random, rects: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    order = sorted(range(len(rects)), key=lambda idx: rects[idx][0] * rects[idx][1], reverse=True)
    for _ in range(PLACE_RETRIES):
        placed: dict[int, tuple[int, int]] = {}
        ok = True
        for idx in order:
            h, w = rects[idx]
            candidates: list[tuple[int, int]] = []
            for r in range(MAX_SIDE - h + 1):
                for c in range(MAX_SIDE - w + 1):
                    rect = (r, c, h, w)
                    if any(_overlaps(rect, (rr, cc, rects[j][0], rects[j][1]), MIN_GAP) for j, (rr, cc) in placed.items()):
                        continue
                    candidates.append((r, c))
            if not candidates:
                ok = False
                break
            placed[idx] = rng.choice(candidates)
        if not ok:
            continue

        min_r = min(placed[idx][0] for idx in range(len(rects)))
        min_c = min(placed[idx][1] for idx in range(len(rects)))
        max_r = max(placed[idx][0] + rects[idx][0] for idx in range(len(rects)))
        max_c = max(placed[idx][1] + rects[idx][1] for idx in range(len(rects)))

        pad_top = rng.randint(0, min(2, min_r))
        pad_left = rng.randint(0, min(2, min_c))
        pad_bottom = rng.randint(0, min(2, MAX_SIDE - max_r))
        pad_right = rng.randint(0, min(2, MAX_SIDE - max_c))

        height = max_r - min_r + pad_top + pad_bottom
        width = max_c - min_c + pad_left + pad_right
        if height > MAX_SIDE or width > MAX_SIDE:
            continue

        adjusted = [None] * len(rects)
        for idx in range(len(rects)):
            r, c = placed[idx]
            adjusted[idx] = (r - min_r + pad_top, c - min_c + pad_left)
        return adjusted, (height, width)
    return None


def _build_completed_scene(proto: np.ndarray, scales: list[int], positions: list[tuple[int, int]], shape: tuple[int, int], bg: int) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for scale, (r0, c0) in zip(scales, positions):
        patch = np.kron(proto, np.ones((scale, scale), dtype=int))
        out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
    return out


def _semantic_checks(
    inp: np.ndarray,
    expected_out: np.ndarray,
    proto: np.ndarray,
    scales: list[int],
    masks: np.ndarray,
    positions: list[tuple[int, int]],
) -> bool:
    if inp.ndim != 2 or max(inp.shape) > MAX_SIDE:
        return False
    if expected_out.ndim != 2 or max(expected_out.shape) > MAX_SIDE:
        return False

    solved = main(inp)
    if not np.array_equal(solved, expected_out):
        return False

    bg = _get_bg(inp)
    copies = _decode_copies(inp)
    if sorted(copy["scale"] for copy in copies) != sorted(scales):
        return False

    recon = _reconstruct_prototype(copies, bg)
    if not np.array_equal(recon, proto):
        return False

    by_scale = {int(scale): idx for idx, scale in enumerate(scales)}
    for copy in copies:
        idx = by_scale[int(copy["scale"])]
        if tuple(copy["origin"]) != tuple(positions[idx]):
            return False
        expected_partial = np.where(masks[idx], proto, bg)
        if not np.array_equal(copy["partial"], expected_partial):
            return False

    for idx in range(len(scales)):
        if not np.any(~masks[idx]):
            return False
        others = np.any(np.delete(masks, idx, axis=0), axis=0)
        if np.all(others):
            return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    copy_count = rng.choices([2, 3, 4], weights=[3, 4, 2])[0]
    if copy_count == 4:
        proto_h = rng.randint(3, 4)
        proto_w = rng.randint(3, 4)
    else:
        proto_h = rng.randint(3, 5)
        proto_w = rng.randint(3, 5)

    proto = _random_prototype(rng, proto_h, proto_w)
    scales = _choose_scales(rng, copy_count)
    masks = _presence_masks(rng, proto, copy_count)
    if masks is None:
        return None

    patches = [_render_partial(proto, scale, masks[idx], BG) for idx, scale in enumerate(scales)]
    rects = [patch.shape for patch in patches]
    placed = _place_rectangles(rng, rects)
    if placed is None:
        return None
    positions, shape = placed

    inp = np.full(shape, BG, dtype=int)
    for patch, (r0, c0) in zip(patches, positions):
        h, w = patch.shape
        inp[r0 : r0 + h, c0 : c0 + w] = patch

    expected_out = _build_completed_scene(proto, scales, positions, shape, BG)
    if not _semantic_checks(inp, expected_out, proto, scales, masks, positions):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid multi-scale complementary-fragment input")
