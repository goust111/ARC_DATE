from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
TEMPLATE = 5
SOURCE_COLORS = [1, 3, 4]
PALETTE_COLORS = [2, 6, 7, 8, 9]
MAX_ATTEMPTS = 800


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste_clip(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(dst.shape[0], r0 + h)
    cc1 = min(dst.shape[1], c0 + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    pr0 = rr0 - r0
    pc0 = cc0 - c0
    dst[rr0:rr1, cc0:cc1] = patch[pr0 : pr0 + (rr1 - rr0), pc0 : pc0 + (cc1 - cc0)]


def _normalize_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0 = coords.min(axis=0)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _mask_from_key(key: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in key) + 1
    w = max(c for _, c in key) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in key:
        mask[r, c] = True
    return mask


def _component_count(mask: np.ndarray) -> int:
    return len(_components_mask(mask))


def _find_palette(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    h, _w = grid.shape
    top = grid[h - 2]
    bottom = grid[h - 1]
    cols = np.flatnonzero((top != BG) | (bottom != BG))
    assert len(cols) == 3 and np.all(np.diff(cols) == 1), "palette must be a 2x3 block in the last two rows"
    assert np.all(top[cols] != BG) and np.all(bottom[cols] != BG), "palette cells must be non-background"
    return h - 2, int(cols[0]), grid[h - 2 : h, cols[0] : cols[0] + 3].copy()


def _extract_templates(area: np.ndarray) -> list[dict]:
    comps = _components_mask(area == TEMPLATE)
    templates: list[dict] = []
    for coords in comps:
        box = _bbox_from_coords(coords)
        patch = _crop(area, box)
        assert np.all((patch == BG) | (patch == TEMPLATE)), "template box may contain only background and grey"
        hole_coords = np.argwhere(patch == BG)
        assert hole_coords.size > 0, "every grey template must contain a hole"
        hole_box_local = _bbox_from_coords(hole_coords)
        hole_mask = _crop(patch == BG, hole_box_local)
        hr0, hc0, _hr1, _hc1 = hole_box_local
        r0, c0, _r1, _c1 = box
        templates.append(
            {
                "template_bbox": box,
                "hole_anchor": (r0 + hr0, c0 + hc0),
                "hole_key": _normalize_coords(np.argwhere(hole_mask)),
                "hole_mask": hole_mask,
            }
        )
    return templates


def _extract_sources(area: np.ndarray) -> list[dict]:
    sources: list[dict] = []
    for color in sorted(int(v) for v in np.unique(area) if int(v) not in (BG, TEMPLATE)):
        coords = np.argwhere(area == color)
        box = _bbox_from_coords(coords)
        patch_mask = _crop(area == color, box)
        key = _normalize_coords(np.argwhere(patch_mask))
        sources.append(
            {
                "color": int(color),
                "bbox": box,
                "coords": coords,
                "shape_key": key,
                "shape_mask": patch_mask,
            }
        )
    return sources


def _parse(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict], int]:
    palette_r0, _palette_c0, palette = _find_palette(grid)
    play = grid[:palette_r0]
    templates = _extract_templates(play)
    sources = _extract_sources(play)
    assert 2 <= len(templates) <= 3, "expected 2 or 3 templates"
    assert len(templates) == len(sources), "template/source counts must match"

    tkeys = [t["hole_key"] for t in templates]
    skeys = [s["shape_key"] for s in sources]
    assert len(set(tkeys)) == len(tkeys), "hole shapes must be unique"
    assert sorted(tkeys) == sorted(skeys), "source shapes must match hole shapes exactly"
    return palette, templates, sources, palette_r0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored source shape to the unique grey hole it completes, then stamp a palette-colored Kronecker expansion at that hole.

    Input structure:
    - The background is black. Above the bottom palette sit 2 or 3 grey rectangular templates, each containing one background hole shape.
    - The right side contains the same number of small monochrome source shapes in distinct colors.
    - The last two rows contain a 2x3 palette block; left-to-right columns encode component counts 1, 2, and 3, with foreground color on top and background color below.

    Rule:
    - Normalize every hole shape and every source shape to bbox-relative coordinates, and match identical shapes one-to-one.
    - Count the 4-connected components of each matched source shape; this selects one palette column.
    - Use the matched hole as the outer mask and the source shape as the inner tile of a Kronecker product; outer foreground cells become the inner tile in the selected foreground color, and outer background cells become solid blocks of the selected background color.
    - Clear the original right-side source shapes to black, then place each Kronecker patch with its top-left aligned to the matched hole bbox top-left; clip only if the patch reaches the canvas edge. Leave the palette unchanged.

    Key insight:
    - Shape matching identifies which source pairs with which hole, while the source shape's component count chooses the palette used to color the Kronecker stamp.
    """
    g = np.asarray(grid, dtype=int)
    palette, templates, sources, _palette_r0 = _parse(g)
    template_by_key = {t["hole_key"]: t for t in templates}

    out = g.copy()
    for src in sources:
        rr = src["coords"][:, 0]
        cc = src["coords"][:, 1]
        out[rr, cc] = BG

    for src in sources:
        key = src["shape_key"]
        template = template_by_key[key]
        k = _component_count(src["shape_mask"])
        assert 1 <= k <= 3, "component count must map to one palette column"
        fg = int(palette[0, k - 1])
        bg = int(palette[1, k - 1])
        painted = np.kron(template["hole_mask"].astype(int), src["shape_mask"].astype(int)).astype(bool)
        patch = np.where(painted, fg, bg).astype(int)
        r0, c0 = template["hole_anchor"]
        _paste_clip(out, patch, r0, c0)

    return out


BASE_SHAPES: dict[int, list[tuple[tuple[int, int], ...]]] = {
    1: [
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 1), (2, 1)),
    ],
    2: [
        ((0, 0), (0, 1), (2, 0), (2, 1)),
        ((0, 0), (1, 0), (0, 2), (1, 2)),
        ((0, 0), (1, 0), (1, 1), (0, 3)),
        ((0, 0), (0, 1), (2, 1), (2, 2)),
    ],
    3: [
        ((0, 0), (0, 2), (2, 1)),
        ((0, 0), (0, 2), (2, 0), (2, 1)),
        ((0, 1), (1, 0), (1, 2)),
        ((0, 0), (1, 2), (2, 0)),
    ],
}


def _transform_key(key: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    pts = np.array(key, dtype=int)
    h = int(pts[:, 0].max()) + 1
    w = int(pts[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[pts[:, 0], pts[:, 1]] = True
    for _ in range(random.randint(0, 3)):
        mask = np.rot90(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return _normalize_coords(np.argwhere(mask))


def _non_identity_permutation(n: int) -> list[int]:
    perm = list(range(n))
    random.shuffle(perm)
    if n > 1 and perm == list(range(n)):
        perm[0], perm[1] = perm[1], perm[0]
    return perm


def _ordered_palette() -> np.ndarray:
    pairs = [(a, b) for a, b in itertools.permutations(PALETTE_COLORS, 2)]
    chosen = random.sample(pairs, 3)
    return np.array([[p[0] for p in chosen], [p[1] for p in chosen]], dtype=int)


def _template_dims(hole_mask: np.ndarray) -> tuple[int, int, int, int]:
    _hole_h, _hole_w = hole_mask.shape
    return 1, 1, 1, 1


def _can_place_mask(canvas: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > canvas.shape[0] or c0 + w > canvas.shape[1]:
        return False
    r1 = r0 + h
    c1 = c0 + w
    return not bool((canvas[r0:r1, c0:c1] & mask).any())


def _reserve(canvas: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    h, w = mask.shape
    canvas[r0 : r0 + h, c0 : c0 + w] |= mask


def _dilate(mask: np.ndarray, pad: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(pad):
        nxt = out.copy()
        nxt[1:, :] |= out[:-1, :]
        nxt[:-1, :] |= out[1:, :]
        nxt[:, 1:] |= out[:, :-1]
        nxt[:, :-1] |= out[:, 1:]
        out = nxt
    return out


def _build_expected(
    inp: np.ndarray,
    holes: list[dict],
    sources: list[dict],
    palette: np.ndarray,
) -> np.ndarray:
    out = inp.copy()
    hole_by_key = {hole["key"]: hole for hole in holes}
    for src in sources:
        rr = src["coords"][:, 0]
        cc = src["coords"][:, 1]
        out[rr, cc] = BG
    for src in sources:
        hole = hole_by_key[src["key"]]
        k = _component_count(src["mask"])
        fg = int(palette[0, k - 1])
        bg = int(palette[1, k - 1])
        patch = np.where(np.kron(hole["mask"].astype(int), src["mask"].astype(int)).astype(bool), fg, bg)
        _paste_clip(out, patch.astype(int), hole["anchor"][0], hole["anchor"][1])
    return out


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, counts: list[int]) -> bool:
    palette, templates, sources, palette_r0 = _parse(inp)
    if not np.array_equal(main(inp), expected):
        return False
    if palette_r0 != inp.shape[0] - 2:
        return False
    if len(set(counts)) < 2:
        return False
    if len(sources) == 3 and sorted(counts) != [1, 2, 3]:
        return False
    if len(set(tuple(col) for col in palette.T.tolist())) != 3:
        return False
    stamp_sigs = set()
    for src in sources:
        tpl = {t["hole_key"]: t for t in templates}[src["shape_key"]]
        k = _component_count(src["shape_mask"])
        fg = int(palette[0, k - 1])
        bg = int(palette[1, k - 1])
        patch = np.where(np.kron(tpl["hole_mask"].astype(int), src["shape_mask"].astype(int)).astype(bool), fg, bg)
        stamp_sigs.add((patch.shape, patch.tobytes()))
    return len(stamp_sigs) == len(sources)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(24, 26)
    w = random.randint(29, 30)
    inp = np.zeros((h, w), dtype=int)
    palette = _ordered_palette()
    palette_c0 = random.randint(w // 2 - 1, w // 2 + 2)
    inp[h - 2 : h, palette_c0 : palette_c0 + 3] = palette

    num = random.choice([2, 3])
    counts = [1, 2, 3] if num == 3 else random.sample([1, 2, 3], 2)
    shape_keys = [_transform_key(random.choice(BASE_SHAPES[k])) for k in counts]
    if any(_component_count(_mask_from_key(key)) != k for key, k in zip(shape_keys, counts)):
        return None

    hole_specs = []
    patch_rects = []
    slots = [(1, 1), (1, 11), (11, 1), (11, 11)]
    random.shuffle(slots)
    if num > len(slots):
        return None
    for key, slot in zip(shape_keys, slots[:num]):
        hole_mask = _mask_from_key(key)
        top, left, bottom, right = _template_dims(hole_mask)
        anchor = (slot[0] + top, slot[1] + left)
        patch_h = hole_mask.shape[0] * hole_mask.shape[0]
        patch_w = hole_mask.shape[1] * hole_mask.shape[1]
        patch_rects.append((anchor[0], anchor[1], patch_h, patch_w))
        hole_specs.append(
            {
                "key": key,
                "mask": hole_mask,
                "anchor": anchor,
                "template_r0": slot[0],
                "template_c0": slot[1],
                "template_h": top + hole_mask.shape[0] + bottom,
                "template_w": left + hole_mask.shape[1] + right,
                "hole_offset": (top, left),
            }
        )

    play_h = h - 2
    for r0, c0, ph, pw in patch_rects:
        if r0 + ph > play_h or c0 + pw > w:
            return None
    for i, (r0, c0, ph, pw) in enumerate(patch_rects):
        for j in range(i + 1, len(patch_rects)):
            r1, c1, qh, qw = patch_rects[j]
            if not (r0 + ph + 1 <= r1 or r1 + qh + 1 <= r0 or c0 + pw + 1 <= c1 or c1 + qw + 1 <= c0):
                return None

    occupied = np.zeros((play_h, w), dtype=bool)
    for hole in hole_specs:
        tr0 = hole["template_r0"]
        tc0 = hole["template_c0"]
        th = hole["template_h"]
        tw = hole["template_w"]
        if tc0 >= w // 2:
            return None
        template = np.full((th, tw), TEMPLATE, dtype=int)
        hr, hc = hole["hole_offset"]
        mask = hole["mask"]
        template[hr : hr + mask.shape[0], hc : hc + mask.shape[1]][mask] = BG
        if _can_place_mask(occupied, np.ones((th, tw), dtype=bool), tr0, tc0):
            inp[tr0 : tr0 + th, tc0 : tc0 + tw] = template
            _reserve(occupied, _dilate(np.ones((th, tw), dtype=bool), pad=1), tr0, tc0)
        else:
            return None

    perm = _non_identity_permutation(num)
    source_specs = [hole_specs[i] for i in perm]
    source_occupied = occupied.copy()
    source_colors = SOURCE_COLORS[:num]
    random.shuffle(source_colors)
    sources_for_expected = []
    for color, hole in zip(source_colors, source_specs):
        mask = hole["mask"]
        sh, sw = mask.shape
        rect = np.ones((sh, sw), dtype=bool)
        placed = False
        for _ in range(80):
            r0 = random.randint(1, play_h - sh - 2)
            c0 = random.randint(w // 2 + 2, w - sw - 2)
            if not _can_place_mask(source_occupied, rect, r0, c0):
                continue
            view = inp[r0 : r0 + sh, c0 : c0 + sw]
            view[mask] = color
            _reserve(source_occupied, _dilate(rect, pad=1), r0, c0)
            coords = np.argwhere(mask) + np.array([r0, c0])
            sources_for_expected.append({"coords": coords, "mask": mask, "key": hole["key"]})
            placed = True
            break
        if not placed:
            return None

    holes_for_expected = [{"anchor": hspec["anchor"], "mask": hspec["mask"], "key": hspec["key"]} for hspec in hole_specs]
    expected = _build_expected(inp, holes_for_expected, sources_for_expected, palette)
    if not _validate_semantics(inp, expected, counts):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid complementary-Kronecker ARC input")
