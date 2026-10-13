from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
THUMB_COLORS = (1, 2, 3, 4)
TEMPLATE_COLOR = 5
MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected True components as coordinate arrays of shape (n, 2)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_mask(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return (r0, c0, r1, c1), mask


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = color


def _template_masks(size: int) -> list[np.ndarray]:
    area = size * size
    lo_hi = {3: (4, 7), 4: (6, 10)}[size]
    out: list[np.ndarray] = []
    for bits in range(1 << area):
        count = bits.bit_count()
        if not (lo_hi[0] <= count <= lo_hi[1]):
            continue
        mask = np.zeros((size, size), dtype=bool)
        for idx in range(area):
            if (bits >> idx) & 1:
                mask[idx // size, idx % size] = True
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if len(_components(mask)) != 1:
            continue
        out.append(mask)
    assert out
    return out


TEMPLATE_MASKS = {3: _template_masks(3), 4: _template_masks(4)}


def _thumbnail_mask(size: int) -> np.ndarray | None:
    lo_hi = {3: (4, 6), 4: (5, 8)}[size]
    for _ in range(120):
        count = random.randint(*lo_hi)
        cells = random.sample(range(size * size), count)
        mask = np.zeros((size, size), dtype=bool)
        for idx in cells:
            mask[idx // size, idx % size] = True
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        return mask
    return None


def _assign_thumbnail_colors(mask: np.ndarray) -> dict[tuple[int, int], int] | None:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if len(cells) < 4:
        return None
    for _ in range(120):
        colors = {cell: random.choice(THUMB_COLORS) for cell in cells}
        used = set(colors.values())
        if not ({1, 3} & used and {2, 4} & used):
            continue
        same_adjacent = False
        for (r, c), color in colors.items():
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in colors and colors[(nr, nc)] == color:
                    same_adjacent = True
                    break
            if same_adjacent:
                break
        if same_adjacent:
            continue
        return colors
    return None


def _find_template(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    comps = _components(grid == TEMPLATE_COLOR)
    assert len(comps) == 1, "expected exactly one grey template"
    bbox, mask = _component_mask(comps[0])
    assert mask.shape[0] == mask.shape[1] and mask.shape[0] in (3, 4), "template must be 3x3 or 4x4"
    assert not np.array_equal(mask, np.fliplr(mask)), "template must not be LR-symmetric"
    return bbox, mask


def _find_thumbnail(grid: np.ndarray) -> list[dict[str, int]]:
    singles: list[tuple[int, int, int]] = []
    for color in THUMB_COLORS:
        for comp in _components(grid == color):
            if len(comp) == 1:
                r, c = map(int, comp[0])
                singles.append((r, c, color))
    assert len(singles) >= 4, "expected at least four thumbnail pixels"
    coords = np.array([(r, c) for r, c, _ in singles], dtype=int)
    r0, c0, r1, c1 = _bbox_coords(coords)
    assert r1 - r0 == c1 - c0 and (r1 - r0 + 1) in (3, 4), "thumbnail must occupy a 3x3 or 4x4 corner box"
    thumb = []
    for r, c, color in singles:
        thumb.append({"r": r - r0, "c": c - c0, "color": color})
    return thumb


def _find_blocks(grid: np.ndarray, template_mask: np.ndarray) -> list[dict[str, int]]:
    n = template_mask.shape[0]
    mirror_mask = np.fliplr(template_mask)
    blocks: list[dict[str, int]] = []
    for color in THUMB_COLORS:
        for comp in _components(grid == color):
            if len(comp) == 1:
                continue
            bbox, mask = _component_mask(comp)
            if mask.shape != (n, n):
                continue
            mirrored = None
            if np.array_equal(mask, template_mask):
                mirrored = 0
            elif np.array_equal(mask, mirror_mask):
                mirrored = 1
            if mirrored is None:
                continue
            blocks.append(
                {
                    "r": bbox[0],
                    "c": bbox[1],
                    "color": color,
                    "mirrored": mirrored,
                }
            )
    assert len(blocks) >= 2, "expected at least two placed colored blocks"
    return blocks


def _solve_pair_mapping(
    thumb_a: dict[str, int],
    thumb_b: dict[str, int],
    block_a: dict[str, int],
    block_b: dict[str, int],
    min_step: int,
) -> tuple[int, int, int] | None:
    tr = thumb_b["r"] - thumb_a["r"]
    tc = thumb_b["c"] - thumb_a["c"]
    br = block_b["r"] - block_a["r"]
    bc = block_b["c"] - block_a["c"]

    kr = None
    kc = None
    if tr == 0:
        if br != 0:
            return None
    else:
        if br % tr != 0:
            return None
        kr = br // tr
    if tc == 0:
        if bc != 0:
            return None
    else:
        if bc % tc != 0:
            return None
        kc = bc // tc

    if kr is None and kc is None:
        return None
    if kr is not None and kc is not None and kr != kc:
        return None
    k = kr if kr is not None else kc
    assert k is not None
    if k < min_step:
        return None
    dr = block_a["r"] - thumb_a["r"] * k
    dc = block_a["c"] - thumb_a["c"] * k
    return int(k), int(dr), int(dc)


def _mapping_valid(
    grid: np.ndarray,
    thumb: list[dict[str, int]],
    blocks: list[dict[str, int]],
    template_size: int,
    k: int,
    dr: int,
    dc: int,
) -> bool:
    h, w = grid.shape
    targets: dict[tuple[int, int], dict[str, int]] = {}
    for px in thumb:
        rr = px["r"] * k + dr
        cc = px["c"] * k + dc
        if not (0 <= rr <= h - template_size and 0 <= cc <= w - template_size):
            return False
        key = (rr, cc)
        if key in targets:
            return False
        targets[key] = px
    for block in blocks:
        px = targets.get((block["r"], block["c"]))
        if px is None:
            return False
        if px["color"] != block["color"]:
            return False
        if (px["color"] % 2 == 1) != bool(block["mirrored"]):
            return False
    return True


def _infer_mapping(
    grid: np.ndarray,
    thumb: list[dict[str, int]],
    blocks: list[dict[str, int]],
    template_size: int,
) -> tuple[int, int, int]:
    candidates_by_block: list[list[dict[str, int]]] = []
    for block in blocks:
        options = [
            px
            for px in thumb
            if px["color"] == block["color"] and ((px["color"] % 2 == 1) == bool(block["mirrored"]))
        ]
        assert options, "every placed block must match some thumbnail pixel"
        candidates_by_block.append(options)

    valid: set[tuple[int, int, int]] = set()
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            for thumb_a in candidates_by_block[i]:
                for thumb_b in candidates_by_block[j]:
                    if thumb_a["r"] == thumb_b["r"] and thumb_a["c"] == thumb_b["c"]:
                        continue
                    cand = _solve_pair_mapping(thumb_a, thumb_b, blocks[i], blocks[j], template_size + 1)
                    if cand is None:
                        continue
                    if _mapping_valid(grid, thumb, blocks, template_size, *cand):
                        valid.add(cand)
    assert len(valid) == 1, "thumbnail-to-block mapping must be uniquely determined"
    return next(iter(valid))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top-left thumbnail to place colored copies of the grey template, mirroring the template when the thumbnail color is odd.

    Input structure:
    - The background is black, with a 3x3 or 4x4 thumbnail near the top-left made of isolated colored pixels.
    - A single grey 3x3 or 4x4 connected template sprite appears elsewhere, and some larger colored copies of that sprite are already placed in the main area.
    - The placed colored copies have the same bounding-box size as the grey template and align to a regular coordinate grid.

    Rule:
    - Infer the common step and offset from thumbnail-local coordinates to main-area sprite origins by matching the already placed colored copies.
    - For each non-background thumbnail pixel, project its local coordinate through that mapping to get a destination origin.
    - If the thumbnail color is odd, stamp a left-right mirror of the grey template there; if the color is even, stamp the template unchanged there, recoloring grey cells to the thumbnail color.
    - Keep any already placed copies and erase the grey template from the output.

    Key insight:
    - Each thumbnail pixel carries two pieces of information at once: its position selects where a sprite goes, and its color parity selects whether that sprite is mirrored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _, template_mask = _find_template(g)
    thumb = _find_thumbnail(g)
    blocks = _find_blocks(g, template_mask)
    k, dr, dc = _infer_mapping(g, thumb, blocks, template_mask.shape[0])

    out = g.copy()
    out[out == TEMPLATE_COLOR] = BG
    mirror_mask = np.fliplr(template_mask)
    for px in thumb:
        rr = px["r"] * k + dr
        cc = px["c"] * k + dc
        mask = mirror_mask if px["color"] % 2 == 1 else template_mask
        _paint_mask(out, mask, rr, cc, px["color"])
    return out


def _choose_anchor_cells(colors: dict[tuple[int, int], int]) -> list[tuple[int, int]] | None:
    cells = list(colors)
    max_count = min(4, len(cells) - 1)
    if max_count < 2:
        return None
    for _ in range(100):
        count = random.choice([2, 3, 3, 4][: max_count + 1])
        count = min(count, max_count)
        anchors = random.sample(cells, count)
        rows = {r for r, _ in anchors}
        cols = {c for _, c in anchors}
        parities = {colors[cell] % 2 for cell in anchors}
        if len(rows) < 2 or len(cols) < 2:
            continue
        if len(parities) < 2:
            continue
        return anchors
    return None


def _generation_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    template_size = random.choice([3, 4])
    thumb_size = random.choice([3, 4])
    template_mask = random.choice(TEMPLATE_MASKS[template_size]).copy()
    thumb_mask = _thumbnail_mask(thumb_size)
    if thumb_mask is None:
        return None
    thumb_colors = _assign_thumbnail_colors(thumb_mask)
    if thumb_colors is None:
        return None
    anchors = _choose_anchor_cells(thumb_colors)
    if anchors is None:
        return None

    gap = random.randint(1, 3)
    step = template_size + gap
    thumb_top, thumb_left = 1, 1

    if random.random() < 0.5:
        template_top = thumb_top + thumb_size + random.randint(1, 2)
        template_left = thumb_left + random.randint(0, 1)
        min_base_r = template_top + template_size + random.randint(2, 4)
        min_base_c = 8
    else:
        template_top = thumb_top + random.randint(0, 1)
        template_left = thumb_left + thumb_size + random.randint(1, 2)
        min_base_r = 8
        min_base_c = template_left + template_size + random.randint(2, 4)

    extent = (thumb_size - 1) * step + template_size
    max_base_r = 29 - extent
    max_base_c = 29 - extent
    if min_base_r > max_base_r or min_base_c > max_base_c:
        return None
    base_r = random.randint(min_base_r, max_base_r)
    base_c = random.randint(min_base_c, max_base_c)

    height_min = max(base_r + extent + 1, template_top + template_size + 1, thumb_top + thumb_size + 1)
    width_min = max(base_c + extent + 1, template_left + template_size + 1, thumb_left + thumb_size + 1)
    if height_min > 30 or width_min > 30:
        return None
    height = random.randint(height_min, min(30, height_min + 2))
    width = random.randint(width_min, min(30, width_min + 2))

    inp = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)

    for (rr, cc), color in thumb_colors.items():
        inp[thumb_top + rr, thumb_left + cc] = color
        out[thumb_top + rr, thumb_left + cc] = color

    _paint_mask(inp, template_mask, template_top, template_left, TEMPLATE_COLOR)

    mirror_mask = np.fliplr(template_mask)
    for (rr, cc), color in thumb_colors.items():
        dst_r = base_r + rr * step
        dst_c = base_c + cc * step
        mask = mirror_mask if color % 2 == 1 else template_mask
        _paint_mask(out, mask, dst_r, dst_c, color)
        if (rr, cc) in anchors:
            _paint_mask(inp, mask, dst_r, dst_c, color)

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        sample = _generation_attempt()
        if sample is None:
            continue
        inp, expected = sample
        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
