from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np


BG = 0
FRAME_COLOR = 8
SPRITE_COLORS = [1, 2, 3, 4]
LATTICE_CHOICES = [5, 6, 7, 9]
MAX_GEN_ATTEMPTS = 800

CHAIN_TEMPLATES = {
    3: [
        {
            "order": (1, 2, 0),
            "coords": [
                [(0, 1), (1, 0), (1, 2), (2, 1), (3, 2), (4, 1), (4, 3)],
                [(0, 0), (0, 2), (1, 1), (1, 3), (3, 3), (4, 4)],
                [(0, 1), (1, 0), (1, 2), (2, 3), (3, 0), (3, 2)],
            ],
        },
        {
            "order": (0, 2, 1),
            "coords": [
                [(0, 0), (1, 1), (3, 1), (3, 3), (4, 2), (4, 4)],
                [(0, 0), (2, 0), (2, 2), (2, 4), (3, 3), (4, 4)],
                [(0, 1), (0, 3), (2, 3), (3, 0), (3, 4), (4, 1), (4, 3)],
            ],
        },
        {
            "order": (1, 2, 0),
            "coords": [
                [(0, 0), (1, 1), (2, 0), (2, 2), (3, 1), (3, 3), (4, 0), (4, 2)],
                [(0, 1), (1, 0), (1, 2), (3, 2)],
                [(0, 0), (1, 1), (3, 1), (3, 3)],
            ],
        },
    ],
    4: [
        {
            "order": (1, 0, 3, 2),
            "coords": [
                [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2), (2, 4)],
                [(0, 0), (0, 2), (1, 1), (3, 1), (3, 3)],
                [(0, 0), (0, 2), (1, 1), (1, 3), (2, 2), (3, 1), (3, 3), (4, 4)],
                [(0, 1), (0, 3), (1, 0), (1, 4), (2, 1), (2, 3), (3, 2), (3, 4)],
            ],
        },
        {
            "order": (2, 1, 3, 0),
            "coords": [
                [(0, 2), (1, 1), (1, 3), (2, 0), (2, 4)],
                [(0, 0), (0, 2), (1, 1), (3, 1), (3, 3)],
                [(0, 0), (0, 2), (1, 3), (2, 2), (4, 2)],
                [(0, 0), (0, 2), (2, 2), (3, 3), (4, 0), (4, 2)],
            ],
        },
        {
            "order": (0, 3, 1, 2),
            "coords": [
                [(0, 0), (1, 1), (3, 1), (3, 3)],
                [(0, 1), (0, 3), (1, 0), (1, 2), (1, 4), (2, 1)],
                [(0, 0), (0, 2), (1, 1), (1, 3), (3, 3), (4, 4)],
                [(0, 0), (0, 2), (1, 3), (2, 0), (2, 4), (3, 3)],
            ],
        },
    ],
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _dense_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=int)
    r0, c0, r1, c1 = _bbox_coords(coords)
    rel = coords - np.array([r0, c0], dtype=int)
    mask = np.zeros((int(r1 - r0 + 2), int(c1 - c0 + 2)), dtype=bool)
    for r, c in rel:
        mask[int(r) : int(r) + 2, int(c) : int(c) + 2] = True
    return mask


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    seen = {tuple(map(int, pts[0]))}
    stack = [tuple(map(int, pts[0]))]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _is_sparse_frame(coords: np.ndarray) -> bool:
    coords = np.asarray(coords, dtype=int)
    if coords.shape[0] < 8:
        return False
    r0, c0, r1, c1 = _bbox_coords(coords)
    if r1 - r0 < 4 or c1 - c0 < 4:
        return False
    if (r1 - r0) % 2 != 0 or (c1 - c0) % 2 != 0:
        return False
    points = {tuple(map(int, p)) for p in coords}
    expected: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1, 2):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1, 2):
        expected.add((r0, c))
        expected.add((r1, c))
    return points == expected


def _edge_rows(mask: np.ndarray, side: str) -> set[int]:
    if side == "left":
        return set(np.where(mask[:, 0])[0].tolist())
    if side == "right":
        return set(np.where(mask[:, -1])[0].tolist())
    raise ValueError(side)


def _sprite_info_from_coords(coords: np.ndarray, color: int) -> dict:
    coords = np.asarray(coords, dtype=int)
    r0, c0, r1, c1 = _bbox_coords(coords)
    rel = coords - np.array([r0, c0], dtype=int)
    dense = _dense_mask_from_coords(coords)
    sparse_left = {int(r) for r, c in rel if int(c) == 0}
    sparse_right = {int(r) for r, c in rel if int(c) == int(c1 - c0)}
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "coords": coords.copy(),
        "rel": rel,
        "dense": dense,
        "sparse_left": sparse_left,
        "sparse_right": sparse_right,
        "dense_left": _edge_rows(dense, "left"),
        "dense_right": _edge_rows(dense, "right"),
    }


def _dense_match(a: dict, b: dict) -> bool:
    return bool(a["dense_right"] & b["dense_left"])


def _unique_chain_order(sprites: list[dict]) -> tuple[int, ...]:
    perms = []
    for perm in itertools.permutations(range(len(sprites))):
        if all(_dense_match(sprites[perm[i]], sprites[perm[i + 1]]) for i in range(len(perm) - 1)):
            perms.append(tuple(int(x) for x in perm))
    assert len(perms) == 1, "dense edge relation must admit a unique chain"
    return perms[0]


def _paint_mask(dst: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w][mask] = int(color)


def _place_ordered_sprites(
    out: np.ndarray,
    sprites: list[dict],
    order: tuple[int, ...],
    frame_bbox: tuple[int, int, int, int],
) -> None:
    r0, c0, r1, c1 = frame_bbox
    inner_h = int(r1 - r0 - 1)
    inner_w = int(c1 - c0 - 1)
    horizontal = (c1 - c0 + 1) > (r1 - r0 + 1)
    ordered = [sprites[i] for i in order]

    if horizontal:
        sizes = [sp["dense"].shape[1] for sp in ordered]
        total = sum(sizes)
        slack = int(inner_w - total)
        assert slack >= 0, "frame interior is too short"
        gap = slack // (len(ordered) + 1)
        extra = slack - gap * (len(ordered) + 1)
        lead = gap + (extra + 1) // 2
        cursor = int(c0 + 1 + lead)
        for idx, sp in enumerate(ordered):
            mask = sp["dense"]
            top = int(r0 + 1 + (inner_h - mask.shape[0]) // 2)
            _paint_mask(out, mask, top, cursor, sp["color"])
            cursor += int(mask.shape[1])
            if idx < len(ordered) - 1:
                cursor += gap
    else:
        sizes = [sp["dense"].shape[0] for sp in ordered]
        total = sum(sizes)
        slack = int(inner_h - total)
        assert slack >= 0, "frame interior is too short"
        gap = slack // (len(ordered) + 1)
        extra = slack - gap * (len(ordered) + 1)
        lead = gap + (extra + 1) // 2
        cursor = int(r0 + 1 + lead)
        for idx, sp in enumerate(ordered):
            mask = sp["dense"]
            left = int(c0 + 1 + (inner_w - mask.shape[1]) // 2)
            _paint_mask(out, mask, cursor, left, sp["color"])
            cursor += int(mask.shape[0])
            if idx < len(ordered) - 1:
                cursor += gap


def main(grid: np.ndarray) -> np.ndarray:
    """
    Densify the sparse frame and sparse sprites, then arrange the sprites inside the frame by dense-edge chain order.

    Input structure:
    - The grid has black background, a regular lattice color, one large sparse rectangular frame, and 3-4 sparse monochrome sprites outside that frame.
    - The frame is drawn as a dashed perimeter on a two-step grid, while each sprite is a small sparse point pattern of a single color.
    - Sprite colors are distinct from the frame and lattice colors.

    Rule:
    - Identify the unique largest sparse rectangular perimeter as the frame and redraw its bbox as a solid 1-pixel border, clearing everything else.
    - Densify each sprite by replacing every foreground pixel with a same-colored 2x2 block whose top-left corner is the original pixel.
    - Compare densified sprites using right-edge versus left-edge row-offset overlap; the unique permutation whose consecutive pairs overlap is the chain order.
    - If the frame is wider than tall, place the chained sprites left-to-right inside it; otherwise place them top-to-bottom. Use equal internal gaps from floor division and assign any remainder only to the first and last margins.

    Key insight:
    - The linking relation is not defined by the original sparse points; it appears only after densification changes each sprite's edge silhouette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    counts = Counter(map(int, g.ravel()))
    non_bg = [int(c) for c in sorted(counts) if int(c) != int(bg)]
    assert non_bg, "expected non-background colors"

    frame_candidates = []
    for color in non_bg:
        coords = np.argwhere(g == int(color))
        if _is_sparse_frame(coords):
            frame_candidates.append((int(color), _bbox_coords(coords), coords))
    assert frame_candidates, "expected a sparse frame"

    max_area = max((b[2] - b[0] + 1) * (b[3] - b[1] + 1) for _color, b, _coords in frame_candidates)
    biggest = [(color, bbox, coords) for color, bbox, coords in frame_candidates if (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1) == max_area]
    assert len(biggest) == 1, "frame must be unique"
    frame_color, frame_bbox, _frame_coords = biggest[0]

    remain = [c for c in non_bg if c != int(frame_color)]
    assert len(remain) >= 4, "expected lattice plus 3-4 sprites"
    lattice_color = max(remain, key=lambda c: counts[int(c)])
    sprite_colors = [int(c) for c in remain if int(c) != int(lattice_color)]
    assert 3 <= len(sprite_colors) <= 4

    r0, c0, r1, c1 = frame_bbox
    sprites = []
    for color in sprite_colors:
        coords = np.argwhere(g == int(color))
        sr0, sc0, sr1, sc1 = _bbox_coords(coords)
        assert sr1 < r0 or sr0 > r1 or sc1 < c0 or sc0 > c1, "sprites must stay outside the frame"
        sprites.append(_sprite_info_from_coords(coords, color))

    for i, sp_a in enumerate(sprites):
        for j, sp_b in enumerate(sprites):
            if i == j:
                continue
            assert not (sp_a["sparse_right"] & sp_b["sparse_left"]), "sparse edge matching must fail before densification"

    order = _unique_chain_order(sprites)

    out = np.full_like(g, int(bg))
    out[r0, c0 : c1 + 1] = int(frame_color)
    out[r1, c0 : c1 + 1] = int(frame_color)
    out[r0 : r1 + 1, c0] = int(frame_color)
    out[r0 : r1 + 1, c1] = int(frame_color)
    _place_ordered_sprites(out, sprites, order, frame_bbox)
    return out


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 - pad <= br1 and br0 - pad <= ar1 and ac0 - pad <= bc1 and bc0 - pad <= ac1


def _template_sprites(count: int) -> tuple[list[dict], tuple[int, ...]]:
    tpl = random.choice(CHAIN_TEMPLATES[count])
    coords_list = [np.array(coords, dtype=int) for coords in tpl["coords"]]
    sprites = []
    for arr in coords_list:
        dense = _dense_mask_from_coords(arr)
        assert _mask_connected(dense)
        sprites.append(_sprite_info_from_coords(arr, color=1))
    order = _unique_chain_order(sprites)
    return sprites, order


def _choose_frame_dims(ordered_sprites: list[dict], horizontal: bool) -> tuple[int, int] | None:
    along_sizes = [sp["dense"].shape[1] if horizontal else sp["dense"].shape[0] for sp in ordered_sprites]
    cross_sizes = [sp["dense"].shape[0] if horizontal else sp["dense"].shape[1] for sp in ordered_sprites]
    total_along = sum(int(v) for v in along_sizes)
    n = len(ordered_sprites)

    options = []
    for gap in (1, 2):
        for extra in (0, 1, 2):
            inner_along = int(total_along + gap * (n + 1) + extra)
            for cross_pad in (1, 2):
                for cross_extra in (0, 1):
                    inner_cross = int(max(cross_sizes) + 2 * cross_pad + cross_extra)
                    outer_along = inner_along + 2
                    outer_cross = inner_cross + 2
                    if outer_along % 2 == 0:
                        outer_along += 1
                    if outer_cross % 2 == 0:
                        outer_cross += 1
                    if outer_along > 25 or outer_cross > 17:
                        continue
                    if outer_along == outer_cross:
                        continue
                    if horizontal:
                        options.append((outer_cross, outer_along))
                    else:
                        options.append((outer_along, outer_cross))
    if not options:
        return None
    return random.choice(options)


def _stamp_sparse_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0 : r1 + 1 : 2, c0] = int(color)
    grid[r0 : r1 + 1 : 2, c1] = int(color)
    grid[r0, c0 : c1 + 1 : 2] = int(color)
    grid[r1, c0 : c1 + 1 : 2] = int(color)


def _try_place_sources(
    canvas_h: int,
    canvas_w: int,
    frame_bbox: tuple[int, int, int, int],
    sprites: list[dict],
) -> list[tuple[int, int]] | None:
    placed: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = [frame_bbox]
    order = list(range(len(sprites)))
    random.shuffle(order)
    pos = [None] * len(sprites)
    for idx in order:
        sp = sprites[idx]
        h, w = sp["dense"].shape[0] - 1, sp["dense"].shape[1] - 1
        for _ in range(400):
            r0 = random.randint(0, canvas_h - h)
            c0 = random.randint(0, canvas_w - w)
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_overlap(box, other, pad=1) for other in boxes):
                continue
            pos[idx] = (int(r0), int(c0))
            boxes.append(box)
            break
        if pos[idx] is None:
            return None
    return [tuple(map(int, p)) for p in pos]


def _source_visual_order(positions: list[tuple[int, int]], horizontal_frame: bool) -> tuple[int, ...]:
    if horizontal_frame:
        return tuple(sorted(range(len(positions)), key=lambda i: (positions[i][1], positions[i][0])))
    return tuple(sorted(range(len(positions)), key=lambda i: (positions[i][0], positions[i][1])))


def _gen_attempt() -> np.ndarray | None:
    count = random.choice([3, 4])
    sprites, chain_order = _template_sprites(count)

    horizontal_frame = bool(random.getrandbits(1))
    ordered = [sprites[i] for i in chain_order]
    frame_dims = _choose_frame_dims(ordered, horizontal_frame)
    if frame_dims is None:
        return None
    frame_h, frame_w = frame_dims

    min_h = max(20, frame_h + 5)
    min_w = max(20, frame_w + 5)
    if min_h > 29 or min_w > 29:
        return None
    max_h = max(min_h, min(29, frame_h + 10))
    max_w = max(min_w, min(29, frame_w + 10))
    canvas_h = random.randint(min_h, max_h)
    canvas_w = random.randint(min_w, max_w)

    frame_r0_choices = [r for r in range(2, canvas_h - frame_h - 1) if r % 2 == 0]
    frame_c0_choices = [c for c in range(2, canvas_w - frame_w - 1) if c % 2 == 0]
    if not frame_r0_choices or not frame_c0_choices:
        return None
    frame_r0 = random.choice(frame_r0_choices)
    frame_c0 = random.choice(frame_c0_choices)
    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)

    positions = _try_place_sources(canvas_h, canvas_w, frame_bbox, sprites)
    if positions is None:
        return None
    if _source_visual_order(positions, horizontal_frame) == chain_order:
        return None

    lattice_color = random.choice(LATTICE_CHOICES)
    sprite_colors = random.sample(SPRITE_COLORS, count)

    for color, sp in zip(sprite_colors, sprites):
        sp["color"] = int(color)

    grid = np.full((canvas_h, canvas_w), BG, dtype=int)
    grid[::2, ::2] = int(lattice_color)
    _stamp_sparse_frame(grid, frame_bbox, FRAME_COLOR)

    for sp, (r0, c0) in zip(sprites, positions):
        for rr, cc in sp["rel"]:
            grid[int(r0 + rr), int(c0 + cc)] = int(sp["color"])

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
