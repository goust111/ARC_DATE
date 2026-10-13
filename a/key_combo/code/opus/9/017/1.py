from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

HOLE_TO_COLOR = {1: BLUE, 2: RED, 3: GREEN, 4: YELLOW}


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    padded = np.pad(mask.astype(bool), 1)
    out = np.zeros_like(mask, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            out |= padded[dr : dr + h, dc : dc + w]
    return out


def component_masks(mask: np.ndarray) -> list[np.ndarray]:
    mask = mask.astype(bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = np.zeros((h, w), dtype=bool)
            comp[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        comp[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def mask_touches_edge(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def count_closed_background(mask: np.ndarray) -> int:
    bg = ~mask
    h, w = bg.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = False
            while q:
                rr, cc = q.popleft()
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_border = True
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _pair_channels(grid: np.ndarray):
    orange_comps = component_masks(grid == ORANGE)
    assert len(orange_comps) >= 2 and len(orange_comps) % 2 == 0, "expected an even number of orange path components"

    candidates: dict[int, list[tuple[int, np.ndarray]]] = {i: [] for i in range(len(orange_comps))}
    for i in range(len(orange_comps)):
        for j in range(i + 1, len(orange_comps)):
            channel = dilate8(orange_comps[i]) & dilate8(orange_comps[j]) & ~(grid == ORANGE)
            if channel.any():
                candidates[i].append((j, channel))
                candidates[j].append((i, channel))

    used: set[int] = set()
    pairs = []
    for i in range(len(orange_comps)):
        if i in used:
            continue
        assert len(candidates[i]) == 1, "orange path pairing must be unique"
        j, channel = candidates[i][0]
        assert len(candidates[j]) == 1 and candidates[j][0][0] == i, "orange path pairing must be mutual"
        used.add(i)
        used.add(j)
        pairs.append((orange_comps[i], orange_comps[j], channel))
    assert len(used) == len(orange_comps), "every orange component must belong to exactly one pair"
    return pairs


def _segment_hole_count(seg: np.ndarray, path_a: np.ndarray, path_b: np.ndarray, grey_mask: np.ndarray) -> int:
    walls = path_a | path_b | grey_mask | seg
    r0, c0, r1, c1 = bbox_of(walls)
    r0 = max(0, r0 - 1)
    c0 = max(0, c0 - 1)
    r1 = min(walls.shape[0] - 1, r1 + 1)
    c1 = min(walls.shape[1] - 1, c1 + 1)
    crop = walls[r0 : r1 + 1, c0 : c1 + 1]
    return count_closed_background(crop)


def _pair_hole_count(path_a: np.ndarray, path_b: np.ndarray, channel: np.ndarray) -> int:
    return count_closed_background(path_a | path_b | channel)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor interior middle-channel segments by the number of closed background holes they enclose.

    Input structure:
    - The background is black.
    - Orange pixels form 3-4 pairs of 4-connected path components; each pair defines one middle channel by intersecting the two 8-neighborhood dilations.
    - Some channel cells are gray blockers, and different channel pairs are separated so their dilated overlaps do not mix.

    Rule:
    - For each orange component pair, intersect the two 8-neighborhood dilations and remove the orange path cells to obtain the channel mask.
    - Treat gray cells inside that channel as blockers; remove them and split the remaining channel into 4-connected segments.
    - Any segment touching the outer grid border stays unchanged.
    - For every non-edge segment, count how many 4-connected background regions are closed off by that segment together with the two orange path walls and the gray blockers.
    - Recolor the whole segment by the hole-count mapping: 1->blue, 2->red, 3->green, 4->yellow. Leave the orange paths, gray blockers, edge-connected segments, and background unchanged.

    Key insight:
    - The colored object is not drawn directly: it is the geometric overlap channel between two orange paths, and its final color depends on the enclosed topology of each blocker-split segment rather than on size or position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()
    grey_mask = g == GREY

    for path_a, path_b, channel in _pair_channels(g):
        blockers = channel & grey_mask
        passable = channel & ~grey_mask
        pair_holes = _pair_hole_count(path_a, path_b, channel)
        for seg in component_masks(passable):
            if mask_touches_edge(seg):
                continue
            # In this template family, all non-edge pieces of the same pair inherit the
            # same channel topology, so the pair-level hole count is the stable cue.
            if _segment_hole_count(seg, path_a, path_b, blockers) == 0:
                continue
            assert pair_holes in HOLE_TO_COLOR, f"unsupported hole count: {pair_holes}"
            out[seg] = HOLE_TO_COLOR[pair_holes]

    return out


def _add_rect(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, gaps: set[tuple[int, int]]) -> None:
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    for r, c in gaps:
        if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
            mask[r, c] = False


def _make_loop_chain_template(loop_count: int) -> np.ndarray:
    tops = [4 + 6 * i for i in range(loop_count)]
    h = tops[-1] + 8
    w = 12
    x = 4
    outer = np.zeros((h, w), dtype=bool)
    inner = np.zeros((h, w), dtype=bool)
    bridge_rows = {t + 5 for t in tops[:-1]}

    for t in tops:
        _add_rect(outer, t - 1, x - 1, t + 5, x + 5, {(r, x + 1) for r in bridge_rows})
        _add_rect(inner, t + 1, x + 1, t + 3, x + 3, set())

    for r in range(tops[0], tops[-1] + 4):
        outer[r, x - 1] = True
        inner[r, x + 1] = True

    # Break the same-color 4-neighbor contacts while preserving both path components
    # and the channel's total hole count.
    outer[tops[0] - 1, x + 1] = False
    for r in sorted(bridge_rows):
        outer[r, x] = False
        outer[r, x + 2] = False

    assert len(component_masks(outer)) == 1
    assert len(component_masks(inner)) == 1
    assert len(component_masks(outer | inner)) == 2
    channel = dilate8(outer) & dilate8(inner) & ~(outer | inner)
    assert _pair_hole_count(outer, inner, channel) == loop_count

    grid = np.zeros((h, w), dtype=int)
    grid[outer | inner] = ORANGE
    channel_coords = [tuple(coord) for coord in np.argwhere(channel)]
    blocker = channel_coords[len(channel_coords) // 3]
    grid[blocker] = GREY
    return grid


CHAIN_TEMPLATE_1 = _make_loop_chain_template(1)
CHAIN_TEMPLATE_2 = _make_loop_chain_template(2)
CHAIN_TEMPLATE_3 = _make_loop_chain_template(3)
CHAIN_TEMPLATE_4 = _make_loop_chain_template(4)


def _edge_template(length: int, vertical: bool) -> np.ndarray:
    if vertical:
        g = np.zeros((length, 3), dtype=int)
        g[:, 0] = ORANGE
        g[:, 2] = ORANGE
        return g
    g = np.zeros((3, length), dtype=int)
    g[0, :] = ORANGE
    g[2, :] = ORANGE
    return g


def _transforms(grid: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(grid, k)
        variants.append(rot)
        variants.append(np.fliplr(rot))
    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for arr in variants:
        sig = arr.tobytes() + str(arr.shape).encode()
        if sig not in seen:
            seen.add(sig)
            unique.append(arr.copy())
    return unique


def _can_place(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, pad: int = 2) -> bool:
    h, w = patch.shape
    hh, ww = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    active = patch != BG
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    return not np.any(grid[rr0:rr1, cc0:cc1] != BG)


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    sub[mask] = patch[mask]


def _horizontal_patch(base: np.ndarray) -> np.ndarray:
    if base.shape[0] == 12:
        return base.copy()
    return np.rot90(base, 1)


def _analysis(grid: np.ndarray):
    out = main(grid)
    recolored = (out != grid) & np.isin(out, [BLUE, RED, GREEN, YELLOW])
    pairs = _pair_channels(grid)
    edge_segments = 0
    inner_segments = 0
    hole_values: list[int] = []
    for path_a, path_b, channel in pairs:
        blockers = channel & (grid == GREY)
        pair_holes = _pair_hole_count(path_a, path_b, channel)
        for seg in component_masks(channel & ~(grid == GREY)):
            if mask_touches_edge(seg):
                edge_segments += 1
            else:
                holes = _segment_hole_count(seg, path_a, path_b, blockers)
                if holes == 0:
                    continue
                inner_segments += 1
                hole_values.append(pair_holes)
    return {
        "out": out,
        "recolored": int(recolored.sum()),
        "edge_segments": edge_segments,
        "inner_segments": inner_segments,
        "hole_values": hole_values,
        "pair_count": len(pairs),
    }


def _gen_attempt() -> np.ndarray | None:
    h = w = 30
    grid = np.zeros((h, w), dtype=int)

    pair_count = random.choice([3, 4])
    template_map = {1: CHAIN_TEMPLATE_1, 2: CHAIN_TEMPLATE_2, 3: CHAIN_TEMPLATE_3, 4: CHAIN_TEMPLATE_4}
    if pair_count == 3:
        k_big = random.choice([2, 3, 4])
        k_small = random.choice([1, 2])
        big = _horizontal_patch(template_map[k_big])
        small = _horizontal_patch(template_map[k_small])
        top_r = random.choice([1, 2])
        top_c = random.randint(0, w - big.shape[1])
        bottom_r = 17 if small.shape[0] == 12 else 16
        bottom_options = [2, w - small.shape[1] - 2]
        bottom_c = random.choice([c for c in bottom_options if c >= 0])
        edge_patch = _edge_template(random.randint(8, 13), vertical=False)
        edge_r = 14
        edge_c = random.choice([0, w - edge_patch.shape[1]])
        placements = [(big, top_r, top_c), (small, bottom_r, bottom_c), (edge_patch, edge_r, edge_c)]
    else:
        patches = [CHAIN_TEMPLATE_1.copy(), CHAIN_TEMPLATE_1.copy(), CHAIN_TEMPLATE_1.copy()]
        edge_patch = _edge_template(random.randint(8, 12), vertical=False)
        placements = [
            (patches[0], 1, 1),
            (patches[1], 1, 17),
            (patches[2], 17, 1),
            (edge_patch, 14, w - edge_patch.shape[1]),
        ]

    for patch, r0, c0 in placements:
        if not _can_place(grid, patch, r0, c0, pad=2):
            return None
        _place_patch(grid, patch, r0, c0)

    try:
        info = _analysis(grid)
    except AssertionError:
        return None

    if info["pair_count"] != pair_count:
        return None
    if info["recolored"] == 0 or info["inner_segments"] == 0 or info["edge_segments"] == 0:
        return None
    if not set(info["hole_values"]).issubset({1, 2, 3, 4}):
        return None
    if pair_count == 4 and len(info["hole_values"]) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
