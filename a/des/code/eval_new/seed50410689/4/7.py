from __future__ import annotations

import numpy as np


BG = 0
MAROON = 9
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8)
CORNERS = ("tl", "tr", "bl", "br")
NEAR_SIDES = {
    "tl": ("up", "left"),
    "tr": ("up", "right"),
    "bl": ("down", "left"),
    "br": ("down", "right"),
}
GEN_RETRIES = 400
PLACE_RETRIES = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_anchor(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    anchors: list[tuple[int, int]] = []
    for r, c in np.argwhere(grid == MAROON):
        rr = int(r)
        cc = int(c)
        if all(
            not (0 <= nr < h and 0 <= nc < w) or int(grid[nr, nc]) == bg
            for nr, nc in _neighbors4(rr, cc)
        ):
            anchors.append((rr, cc))
    assert len(anchors) == 1, "expected exactly one isolated maroon anchor"
    return anchors[0]


def _extract_blocks(grid: np.ndarray, bg: int) -> tuple[tuple[int, int], list[dict[str, int]]]:
    anchor = _find_anchor(grid, bg)
    mask = grid != bg
    mask[anchor] = False

    blocks: list[dict[str, int]] = []
    for comp in _components(mask):
        coords = np.array(comp, dtype=int)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = sub.shape
        assert coords.shape[0] == h * w, "each object must be a solid rectangle"

        marker_pts = np.argwhere(sub == MAROON)
        assert marker_pts.shape[0] == 1, "each block must contain one maroon marker"
        mr = int(marker_pts[0, 0]) + int(r0)
        mc = int(marker_pts[0, 1]) + int(c0)
        assert int(r0) < mr < int(r1) and int(c0) < mc < int(c1), "marker must be interior"

        colors = {int(v) for v in np.unique(sub) if int(v) not in (bg, MAROON)}
        assert len(colors) == 1, "each block must have one non-maroon color"
        color = colors.pop()

        up = mr - int(r0)
        down = int(r1) - mr
        left = mc - int(c0)
        right = int(c1) - mc
        blocks.append(
            {
                "color": int(color),
                "area": int(h * w),
                "up": int(up),
                "down": int(down),
                "left": int(left),
                "right": int(right),
            }
        )

    areas = [block["area"] for block in blocks]
    assert len(areas) == len(set(areas)), "block areas must be distinct"
    assert blocks, "expected at least one colored rectangle"
    return anchor, blocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered marked rectangles into nested hollow frames aligned to one corner anchor.

    Input structure:
    - The background is black.
    - Several disconnected solid rectangles of distinct colors are scattered across the grid.
    - Each rectangle contains exactly one interior maroon pixel, while one additional isolated maroon pixel sits alone in a black corner region as the global anchor.

    Rule:
    - Treat the isolated maroon pixel as the anchor location.
    - For every colored rectangle, use its maroon interior pixel as a local alignment marker and record the rectangle's size.
    - Sort the rectangles by area from largest to smallest, translate each one so its marker lands on the anchor, and redraw it as a one-pixel-thick hollow rectangle in its original color.
    - Later, smaller frames overwrite earlier, larger frames where their borders coincide.

    Key insight:
    - The maroon pixels inside the rectangles are not decoration: they are alignment anchors, and the final nested-frame structure comes from area order plus common marker alignment rather than from the original scattered positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = BG
    anchor, blocks = _extract_blocks(g, bg)
    ar, ac = anchor

    out = np.full(g.shape, bg, dtype=int)
    for block in sorted(blocks, key=lambda item: item["area"], reverse=True):
        r0 = ar - block["up"]
        r1 = ar + block["down"]
        c0 = ac - block["left"]
        c1 = ac + block["right"]
        assert 0 <= r0 < g.shape[0] and 0 <= c0 < g.shape[1]
        assert 0 <= r1 < g.shape[0] and 0 <= c1 < g.shape[1]

        color = block["color"]
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color

    return out


def _sample_anchor(rng: np.random.Generator, h: int, w: int, layers: int, corner: str) -> tuple[int, int]:
    near_limit_r = min(6, h - 1 - layers)
    near_limit_c = min(6, w - 1 - layers)
    if near_limit_r < layers or near_limit_c < layers:
        raise ValueError("grid too small for requested layer count")

    top_dist = int(rng.integers(layers, near_limit_r + 1))
    bottom_dist = int(rng.integers(layers, near_limit_r + 1))
    left_dist = int(rng.integers(layers, near_limit_c + 1))
    right_dist = int(rng.integers(layers, near_limit_c + 1))

    if corner == "tl":
        return top_dist, left_dist
    if corner == "tr":
        return top_dist, w - 1 - right_dist
    if corner == "bl":
        return h - 1 - bottom_dist, left_dist
    return h - 1 - bottom_dist, w - 1 - right_dist


def _sample_side_sequence(
    rng: np.random.Generator,
    layers: int,
    cap: int,
    bias_far: bool,
) -> tuple[int, ...] | None:
    extra_lo, extra_hi = ((3, 8) if bias_far else (0, 3))
    high_lo = layers + extra_lo
    high_hi = min(cap, layers + extra_hi)
    if high_hi < layers:
        return None
    high = int(rng.integers(min(high_lo, high_hi), high_hi + 1))
    vals = rng.choice(np.arange(1, high + 1), size=layers, replace=False)
    vals.sort()
    return tuple(int(v) for v in vals[::-1])


def _sample_nested_extents(
    rng: np.random.Generator,
    h: int,
    w: int,
    layers: int,
    corner: str,
    anchor: tuple[int, int],
) -> list[dict[str, int]] | None:
    ar, ac = anchor
    caps = {
        "up": int(ar),
        "down": int(h - 1 - ar),
        "left": int(ac),
        "right": int(w - 1 - ac),
    }
    near = set(NEAR_SIDES[corner])

    seqs: dict[str, tuple[int, ...]] = {}
    for side, cap in caps.items():
        seq = _sample_side_sequence(rng, layers, cap, bias_far=side not in near)
        if seq is None:
            return None
        seqs[side] = seq

    if not all(seqs[far][0] >= seqs[near_side][0] + 1 for near_side in near for far in caps if far not in near):
        return None

    blocks: list[dict[str, int]] = []
    for i in range(layers):
        up = seqs["up"][i]
        down = seqs["down"][i]
        left = seqs["left"][i]
        right = seqs["right"][i]
        blocks.append(
            {
                "up": int(up),
                "down": int(down),
                "left": int(left),
                "right": int(right),
                "area": int((up + down + 1) * (left + right + 1)),
            }
        )

    areas = [block["area"] for block in blocks]
    if len(areas) != len(set(areas)):
        return None
    return blocks


def _reserve_corner_patch(
    shape: tuple[int, int],
    corner: str,
    anchor: tuple[int, int],
) -> np.ndarray:
    h, w = shape
    ar, ac = anchor
    mask = np.zeros((h, w), dtype=bool)
    if corner == "tl":
        mask[: ar + 2, : ac + 2] = True
    elif corner == "tr":
        mask[: ar + 2, max(0, ac - 1) :] = True
    elif corner == "bl":
        mask[max(0, ar - 1) :, : ac + 2] = True
    else:
        mask[max(0, ar - 1) :, max(0, ac - 1) :] = True
    return mask


def _place_tile(
    rng: np.random.Generator,
    grid: np.ndarray,
    occupied: np.ndarray,
    tile: np.ndarray,
) -> bool:
    h, w = grid.shape
    th, tw = tile.shape
    for _ in range(PLACE_RETRIES):
        r0 = int(rng.integers(0, h - th + 1))
        c0 = int(rng.integers(0, w - tw + 1))
        rs = max(0, r0 - 1)
        re = min(h, r0 + th + 1)
        cs = max(0, c0 - 1)
        ce = min(w, c0 + tw + 1)
        if occupied[rs:re, cs:ce].any():
            continue
        grid[r0 : r0 + th, c0 : c0 + tw] = tile
        occupied[rs:re, cs:ce] = True
        return True
    return False


def _block_tile(block: dict[str, int], color: int) -> np.ndarray:
    h = block["up"] + block["down"] + 1
    w = block["left"] + block["right"] + 1
    tile = np.full((h, w), int(color), dtype=int)
    tile[block["up"], block["left"]] = MAROON
    return tile


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 31))
    w = int(rng.integers(18, 31))
    layers = int(rng.integers(2, 6))
    corner = str(rng.choice(CORNERS))

    try:
        anchor = _sample_anchor(rng, h, w, layers, corner)
    except ValueError:
        return None

    blocks = _sample_nested_extents(rng, h, w, layers, corner, anchor)
    if blocks is None:
        return None

    colors = [int(c) for c in rng.choice(np.array(PALETTE), size=layers, replace=False)]
    for block, color in zip(blocks, colors):
        block["color"] = int(color)

    order = sorted(blocks, key=lambda item: item["area"], reverse=True)
    grid = np.full((h, w), BG, dtype=int)
    reserve = _reserve_corner_patch((h, w), corner, anchor)
    occupied = reserve.copy()
    ar, ac = anchor
    grid[ar, ac] = MAROON
    occupied[ar, ac] = True

    for block in order:
        tile = _block_tile(block, block["color"])
        if not _place_tile(rng, grid, occupied, tile):
            return None

    try:
        parsed_anchor, parsed_blocks = _extract_blocks(grid, BG)
        assert parsed_anchor == anchor
        assert len(parsed_blocks) == layers
        out = main(grid)
        assert out.shape == grid.shape
        assert out.dtype == int
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
