from __future__ import annotations

import random
from typing import Iterable

import numpy as np


EDGE_ORDER = ("top", "bottom", "left", "right")

VERTICAL_SPRITES = [
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
]
HORIZONTAL_SPRITES = [mask.T.copy() for mask in VERTICAL_SPRITES]

GEN_ATTEMPTS = 400
GEN_INNER_H = (9, 14)
GEN_INNER_W = (10, 16)
GEN_MARGIN = (3, 7)
GEN_BLOCK_COUNT = (2, 4)


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            queue = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            head = 0
            while head < len(queue):
                cr, cc = queue[head]
                head += 1
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = comp["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _coords_on_bbox_perimeter(
    coords: np.ndarray, bbox: tuple[int, int, int, int]
) -> bool:
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 4 or c1 - c0 < 4:
        return False
    for r, c in coords:
        if r not in (r0, r1) and c not in (c0, c1):
            return False
    return True


def _find_frame(grid: np.ndarray, bg: int) -> dict:
    best = None
    for color in (int(c) for c in np.unique(grid) if int(c) != bg):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        bbox = (int(r0), int(c0), int(r1), int(c1))
        if not _coords_on_bbox_perimeter(coords, bbox):
            continue
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        if np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == color):
            continue
        candidate = {"color": color, "bbox": bbox, "area": int(coords.shape[0])}
        if best is None or candidate["area"] > best["area"]:
            best = candidate
    assert best is not None, "frame not found"
    return best


def _is_inner_block(comp: dict, frame_bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    fr0, fc0, fr1, fc1 = frame_bbox
    return (
        comp["area"] == 4
        and r1 - r0 == 1
        and c1 - c0 == 1
        and fr0 < r0 <= r1 < fr1
        and fc0 < c0 <= c1 < fc1
    )


def _edge_holes(
    grid: np.ndarray, frame_bbox: tuple[int, int, int, int], bg: int
) -> dict[str, list[int]]:
    fr0, fc0, fr1, fc1 = frame_bbox
    return {
        "top": [c for c in range(fc0 + 1, fc1) if int(grid[fr0, c]) == bg],
        "bottom": [c for c in range(fc0 + 1, fc1) if int(grid[fr1, c]) == bg],
        "left": [r for r in range(fr0 + 1, fr1) if int(grid[r, fc0]) == bg],
        "right": [r for r in range(fr0 + 1, fr1) if int(grid[r, fc1]) == bg],
    }


def _nearest_edge(
    block_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]
) -> str:
    br0, bc0, br1, bc1 = block_bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    distances = {
        "top": br0 - fr0,
        "bottom": fr1 - br1,
        "left": bc0 - fc0,
        "right": fc1 - bc1,
    }
    return min(EDGE_ORDER, key=lambda side: (distances[side], EDGE_ORDER.index(side)))


def _mirrored_mask(mask: np.ndarray, side: str) -> np.ndarray:
    return np.flipud(mask) if side in ("top", "bottom") else np.fliplr(mask)


def _aligned_top_left(
    target_r2: int, target_c2: int, shape: tuple[int, int]
) -> tuple[int, int]:
    h, w = shape
    return (target_r2 - (h - 1)) // 2, (target_c2 - (w - 1)) // 2


def _projection_tasks(
    block_bbox: tuple[int, int, int, int],
    side: str,
    gaps: list[int],
    sprite_mask: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    mirrored = _mirrored_mask(sprite_mask, side)
    br0, bc0, br1, bc1 = block_bbox
    row2 = br0 + br1
    col2 = bc0 + bc1
    origins: list[tuple[int, int]] = []
    for gap in gaps:
        if side in ("top", "bottom"):
            target_r2, target_c2 = row2, 2 * gap
        else:
            target_r2, target_c2 = 2 * gap, col2
        origins.append(_aligned_top_left(target_r2, target_c2, mirrored.shape))
    return mirrored, origins


def _paint_mask_clipped(
    out: np.ndarray,
    mask: np.ndarray,
    color: int,
    top: int,
    left: int,
    clip_bbox: tuple[int, int, int, int],
) -> None:
    h, w = mask.shape
    cr0, cc0, cr1, cc1 = clip_bbox
    r0 = max(top, cr0)
    c0 = max(left, cc0)
    r1 = min(top + h - 1, cr1)
    c1 = min(left + w - 1, cc1)
    if r0 > r1 or c0 > c1:
        return
    mr0 = r0 - top
    mc0 = c0 - left
    mr1 = mr0 + (r1 - r0 + 1)
    mc1 = mc0 + (c1 - c0 + 1)
    patch = out[r0 : r1 + 1, c0 : c1 + 1]
    patch_mask = mask[mr0:mr1, mc0:mc1]
    patch[patch_mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_components(g, bg)
    assert comps, "expected non-background objects"

    frame = _find_frame(g, bg)
    frame_bbox = frame["bbox"]
    frame_color = int(frame["color"])
    holes = _edge_holes(g, frame_bbox, bg)

    blocks: dict[int, dict] = {}
    sprites: dict[int, dict] = {}
    for comp in comps:
        if int(comp["color"]) == frame_color:
            continue
        color = int(comp["color"])
        if _is_inner_block(comp, frame_bbox):
            blocks[color] = comp
        else:
            sprites[color] = comp

    fr0, fc0, fr1, fc1 = frame_bbox
    inner_clip = (fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1)
    out = g.copy()

    for comp in blocks.values():
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for color, block in blocks.items():
        assert color in sprites, "missing matching sprite"
        side = _nearest_edge(block["bbox"], frame_bbox)
        sprite_mask = _component_mask(sprites[color])
        mirrored, origins = _projection_tasks(
            block["bbox"], side, holes[side], sprite_mask
        )
        for top, left in origins:
            _paint_mask_clipped(out, mirrored, color, top, left, inner_clip)

    return out


def _sample_spaced_positions(lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    values = list(range(lo, hi + 1))
    random.shuffle(values)
    picked: list[int] = []
    for value in values:
        if all(abs(value - prev) >= min_gap for prev in picked):
            picked.append(value)
            if len(picked) == count:
                return sorted(picked)
    return None


def _edge_sprite_choices(side: str) -> list[np.ndarray]:
    base = VERTICAL_SPRITES if side in ("top", "bottom") else HORIZONTAL_SPRITES
    choices: list[np.ndarray] = []
    for mask in base:
        choice = mask.copy()
        if side in ("top", "bottom") and random.random() < 0.5:
            choice = np.fliplr(choice)
        if side in ("left", "right") and random.random() < 0.5:
            choice = np.flipud(choice)
        choices.append(choice)
    random.shuffle(choices)
    return choices


def _zone_positions(
    frame_bbox: tuple[int, int, int, int], side: str
) -> list[tuple[int, int]]:
    fr0, fc0, fr1, fc1 = frame_bbox
    if side == "top":
        rows = [r for r in (fr0 + 2, fr0 + 3) if r + 1 < fr1]
        cols = range(fc0 + 2, fc1 - 2)
    elif side == "bottom":
        rows = [r for r in (fr1 - 4, fr1 - 3) if fr0 < r and r + 1 < fr1]
        cols = range(fc0 + 2, fc1 - 2)
    elif side == "left":
        rows = range(fr0 + 2, fr1 - 2)
        cols = [c for c in (fc0 + 2, fc0 + 3) if c + 1 < fc1]
    else:
        rows = range(fr0 + 2, fr1 - 2)
        cols = [c for c in (fc1 - 4, fc1 - 3) if fc0 < c and c + 1 < fc1]
    positions = [(r, c) for r in rows for c in cols]
    random.shuffle(positions)
    return positions


def _bbox_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _projected_cells(
    block_bbox: tuple[int, int, int, int],
    side: str,
    gaps: list[int],
    sprite_mask: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, set[tuple[int, int]]] | None:
    fr0, fc0, fr1, fc1 = frame_bbox
    inner_clip = (fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1)
    mirrored, origins = _projection_tasks(block_bbox, side, gaps, sprite_mask)
    cells: set[tuple[int, int]] = set()
    for top, left in origins:
        for rr, cc in np.argwhere(mirrored):
            r = int(top + rr)
            c = int(left + cc)
            if not (inner_clip[0] <= r <= inner_clip[2] and inner_clip[1] <= c <= inner_clip[3]):
                return None
            if (r, c) in cells:
                return None
            cells.add((r, c))
    return mirrored, cells


def _touches_existing_block(
    block_occ: np.ndarray, bbox: tuple[int, int, int, int]
) -> bool:
    h, w = block_occ.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    return bool(block_occ[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _place_blocks(
    frame_bbox: tuple[int, int, int, int], holes: dict[str, list[int]], block_specs: list[dict]
) -> list[dict] | None:
    fr0, fc0, fr1, fc1 = frame_bbox
    h = fr1 + 1 + 1
    w = fc1 + 1 + 1
    block_occ = np.zeros((h, w), dtype=bool)
    proj_occ = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []

    for spec in sorted(block_specs, key=lambda item: -len(holes[item["side"]])):
        side = spec["side"]
        chosen = None
        for sprite_mask in _edge_sprite_choices(side):
            for r0, c0 in _zone_positions(frame_bbox, side):
                bbox = (r0, c0, r0 + 1, c0 + 1)
                if _nearest_edge(bbox, frame_bbox) != side:
                    continue
                if _touches_existing_block(block_occ, bbox):
                    continue
                project = _projected_cells(bbox, side, holes[side], sprite_mask, frame_bbox)
                if project is None:
                    continue
                mirrored, proj_cells = project
                block_cells = _bbox_cells(bbox)
                if block_cells & proj_cells:
                    continue
                if any(proj_occ[r, c] for r, c in proj_cells):
                    continue
                if any(proj_occ[r, c] for r, c in block_cells):
                    continue
                chosen = {
                    "color": spec["color"],
                    "side": side,
                    "bbox": bbox,
                    "sprite_mask": sprite_mask,
                    "mirrored_mask": mirrored,
                    "projected_cells": proj_cells,
                }
                break
            if chosen is not None:
                break
        if chosen is None:
            return None
        for r, c in _bbox_cells(chosen["bbox"]):
            block_occ[r, c] = True
        for r, c in chosen["projected_cells"]:
            proj_occ[r, c] = True
        placed.append(chosen)

    return placed


def _frame_grid(
    shape: tuple[int, int],
    bg: int,
    frame_color: int,
    frame_bbox: tuple[int, int, int, int],
    holes: dict[str, list[int]],
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    fr0, fc0, fr1, fc1 = frame_bbox
    grid[fr0, fc0 : fc1 + 1] = frame_color
    grid[fr1, fc0 : fc1 + 1] = frame_color
    grid[fr0 : fr1 + 1, fc0] = frame_color
    grid[fr0 : fr1 + 1, fc1] = frame_color
    for c in holes["top"]:
        grid[fr0, c] = bg
    for c in holes["bottom"]:
        grid[fr1, c] = bg
    for r in holes["left"]:
        grid[r, fc0] = bg
    for r in holes["right"]:
        grid[r, fc1] = bg
    return grid


def _free_positions(occ: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = occ.shape
    ph, pw = shape
    positions = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc1 = min(w, c0 + pw + 1)
            if occ[rr0:rr1, cc0:cc1].any():
                continue
            positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _expected_from_plan(
    inp: np.ndarray,
    bg: int,
    frame_bbox: tuple[int, int, int, int],
    holes: dict[str, list[int]],
    blocks: list[dict],
) -> np.ndarray:
    out = inp.copy()
    fr0, fc0, fr1, fc1 = frame_bbox
    inner_clip = (fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1)
    for block in blocks:
        r0, c0, r1, c1 = block["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg
    for block in blocks:
        mirrored, origins = _projection_tasks(
            block["bbox"], block["side"], holes[block["side"]], block["sprite_mask"]
        )
        for top, left in origins:
            _paint_mask_clipped(out, mirrored, block["color"], top, left, inner_clip)
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    frame_color = int(palette.pop())

    inner_h = random.randint(*GEN_INNER_H)
    inner_w = random.randint(*GEN_INNER_W)
    top_margin = random.randint(*GEN_MARGIN)
    bottom_margin = random.randint(*GEN_MARGIN)
    left_margin = random.randint(*GEN_MARGIN)
    right_margin = random.randint(*GEN_MARGIN)

    h = top_margin + inner_h + 2 + bottom_margin
    w = left_margin + inner_w + 2 + right_margin
    frame_bbox = (
        top_margin,
        left_margin,
        top_margin + inner_h + 1,
        left_margin + inner_w + 1,
    )
    fr0, fc0, fr1, fc1 = frame_bbox

    holes: dict[str, list[int]] = {}
    for side in EDGE_ORDER:
        count = random.randint(1, 2)
        if side in ("top", "bottom"):
            picked = _sample_spaced_positions(fc0 + 2, fc1 - 2, count, min_gap=4)
        else:
            picked = _sample_spaced_positions(fr0 + 2, fr1 - 2, count, min_gap=4)
        if picked is None:
            return None
        holes[side] = picked

    n_blocks = random.randint(*GEN_BLOCK_COUNT)
    if len(palette) < n_blocks:
        return None
    block_colors = [int(palette.pop()) for _ in range(n_blocks)]
    sides = random.sample(list(EDGE_ORDER), k=n_blocks)
    block_specs = [{"color": color, "side": side} for color, side in zip(block_colors, sides)]

    placed_blocks = _place_blocks(frame_bbox, holes, block_specs)
    if placed_blocks is None:
        return None

    inp = _frame_grid((h, w), bg, frame_color, frame_bbox, holes)
    for block in placed_blocks:
        r0, c0, _r1, _c1 = block["bbox"]
        inp[r0 : r0 + 2, c0 : c0 + 2] = block["color"]

    occ = np.zeros((h, w), dtype=bool)
    occ[max(0, fr0 - 1) : min(h, fr1 + 2), max(0, fc0 - 1) : min(w, fc1 + 2)] = True
    for block in sorted(placed_blocks, key=lambda item: -int(item["sprite_mask"].sum())):
        sprite_mask = block["sprite_mask"]
        shape = sprite_mask.shape
        positions = _free_positions(occ, shape)
        if not positions:
            return None
        r0, c0 = positions[0]
        patch = inp[r0 : r0 + shape[0], c0 : c0 + shape[1]]
        patch[sprite_mask] = block["color"]
        occ[max(0, r0 - 1) : min(h, r0 + shape[0] + 1), max(0, c0 - 1) : min(w, c0 + shape[1] + 1)] = True

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    expected = _expected_from_plan(inp, bg, frame_bbox, holes, placed_blocks)
    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid example")
