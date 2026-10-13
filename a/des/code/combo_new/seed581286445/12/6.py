from __future__ import annotations

import math
import random

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")
SIDE_STEPS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_RETRIES = 400
PLACEMENT_RETRIES = 200


def _dominant_color(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(flat)
    return int(np.argmax(counts))


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((int(r), int(c)))
                for dr, dc in SIDE_STEPS.values():
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(cells)
    return comps


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(g[r0, c0])
            if color == bg or seen[r0, c0]:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr, dc in SIDE_STEPS.values():
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            r0b, r1b = min(rows), max(rows)
            c0b, c1b = min(cols), max(cols)
            mask = np.zeros((r1b - r0b + 1, c1b - c0b + 1), dtype=bool)
            for r, c in coords:
                mask[r - r0b, c - c0b] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0b, c0b, r1b, c1b),
                    "mask": mask,
                }
            )

    return comps


def _component_frame_mask(h: int, w: int, side: str) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[-1, :] = True
    elif side == "bottom":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[0, :] = True
    elif side == "left":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, -1] = True
    elif side == "right":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
    else:
        raise ValueError(side)
    return mask


def _open_frame_mask(h: int, w: int, side: str) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        mask[-1, :] = True
        mask[1:, 0] = True
        mask[1:, -1] = True
    elif side == "bottom":
        mask[0, :] = True
        mask[:-1, 0] = True
        mask[:-1, -1] = True
    elif side == "left":
        mask[0, 1:] = True
        mask[-1, 1:] = True
        mask[:, -1] = True
    elif side == "right":
        mask[0, :-1] = True
        mask[-1, :-1] = True
        mask[:, 0] = True
    else:
        raise ValueError(side)
    return mask


def _parse_frames(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []

    for comp in _monochrome_components(g, bg):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue

        local = np.asarray(comp["mask"], dtype=bool)
        matched_side = None
        for side in SIDES:
            if np.array_equal(local, _component_frame_mask(h, w, side)):
                matched_side = side
                break
        if matched_side is None:
            continue

        full_r0, full_c0, full_r1, full_c1 = r0, c0, r1, c1
        if matched_side == "top":
            full_r0 -= 1
        elif matched_side == "bottom":
            full_r1 += 1
        elif matched_side == "left":
            full_c0 -= 1
        else:
            full_c1 += 1

        if full_r0 < 0 or full_c0 < 0 or full_r1 >= g.shape[0] or full_c1 >= g.shape[1]:
            continue

        full_patch = g[full_r0 : full_r1 + 1, full_c0 : full_c1 + 1].copy()
        if matched_side == "top" and np.any(full_patch[0, :] != bg):
            continue
        if matched_side == "bottom" and np.any(full_patch[-1, :] != bg):
            continue
        if matched_side == "left" and np.any(full_patch[:, 0] != bg):
            continue
        if matched_side == "right" and np.any(full_patch[:, -1] != bg):
            continue

        payload = g[full_r0 + 1 : full_r1, full_c0 + 1 : full_c1].copy()
        if payload.size == 0 or not np.any(payload != bg):
            continue
        frames.append(
            {
                "color": int(comp["color"]),
                "bbox": (full_r0, full_c0, full_r1, full_c1),
                "gap_side": matched_side,
                "payload": payload,
                "full_patch": full_patch,
            }
        )

    assert frames, "expected at least one open frame"
    colors = [frame["color"] for frame in frames]
    assert len(colors) == len(set(colors)), "frame colors must be unique"
    return frames


def _rect_cycle_positions(h: int, w: int) -> list[tuple[int, int]]:
    if h < 2 or w < 2:
        return []
    positions: list[tuple[int, int]] = []
    for c in range(w):
        positions.append((0, c))
    for r in range(1, h):
        positions.append((r, w - 1))
    for c in range(w - 2, -1, -1):
        positions.append((h - 1, c))
    for r in range(h - 2, 0, -1):
        positions.append((r, 0))
    return positions


def _ring_order(grid: np.ndarray, frames: list[dict], bg: int) -> list[int]:
    g = np.asarray(grid, dtype=int)
    frame_colors = {int(frame["color"]) for frame in frames}
    outside = g != bg
    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        outside[r0 : r1 + 1, c0 : c1 + 1] = False

    candidates: list[list[int]] = []
    for coords in _components_mask(outside):
        colors = [int(g[r, c]) for r, c in coords]
        if len(colors) != len(frame_colors):
            continue
        if len(set(colors)) != len(colors):
            continue
        if set(colors) != frame_colors:
            continue

        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        positions = _rect_cycle_positions(h, w)
        if len(positions) != len(coords):
            continue

        coord_set = set(coords)
        abs_positions = [(r0 + rr, c0 + cc) for rr, cc in positions]
        if coord_set != set(abs_positions):
            continue

        candidates.append([int(g[r, c]) for r, c in abs_positions])

    assert len(candidates) == 1, "expected one rectangular color ring"
    return candidates[0]


def _paint_non_bg(out: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = map(int, patch.shape)
    view = out[r0 : r0 + h, c0 : c0 + w]
    mask = patch != bg
    assert not np.any(mask & (view != bg)), "assembled pieces should not overlap on non-background pixels"
    view[mask] = patch[mask]


def _assemble(frames_in_order: list[dict], bg: int) -> np.ndarray:
    first = frames_in_order[0]
    first_full = np.asarray(first["full_patch"], dtype=int)
    first_payload = np.asarray(first["payload"], dtype=int)
    first_ph, first_pw = map(int, first_payload.shape)
    fh, fw = map(int, first_full.shape)

    pieces: list[tuple[np.ndarray, int, int, bool]] = [(first_full, 0, 0, True)]
    min_r = 0
    min_c = 0
    max_r = fh
    max_c = fw

    cur_r = 1
    cur_c = 1
    cur_h = first_ph
    cur_w = first_pw

    for idx in range(len(frames_in_order) - 1):
        side = str(frames_in_order[idx]["gap_side"])
        next_patch = np.asarray(frames_in_order[idx + 1]["payload"], dtype=int)
        next_h, next_w = map(int, next_patch.shape)

        if side == "top":
            nr = cur_r - next_h
            nc = cur_c
        elif side == "bottom":
            nr = cur_r + cur_h
            nc = cur_c
        elif side == "left":
            nr = cur_r
            nc = cur_c - next_w
        else:
            nr = cur_r
            nc = cur_c + cur_w

        pieces.append((next_patch, nr, nc, False))
        min_r = min(min_r, nr)
        min_c = min(min_c, nc)
        max_r = max(max_r, nr + next_h)
        max_c = max(max_c, nc + next_w)
        cur_r, cur_c, cur_h, cur_w = nr, nc, next_h, next_w

    out = np.full((max_r - min_r, max_c - min_c), bg, dtype=int)
    row_off = -min_r
    col_off = -min_c
    for patch, pr, pc, full_assign in pieces:
        h, w = map(int, patch.shape)
        rr = pr + row_off
        cc = pc + col_off
        if full_assign:
            out[rr : rr + h, cc : cc + w] = patch
        else:
            _paint_non_bg(out, patch, rr, cc, bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the inner patterns from open frames, order them by a clockwise color ring, and chain them by each frame's opening side.

    Input structure:
    - The background is black.
    - Several disjoint objects each contain a one-pixel-thick monochrome rectangular frame with exactly one whole side missing, plus a multicolor pattern in the frame's interior window.
    - Outside those objects there is one separate rectangular color ring made of unique colors, and those colors match the frame colors one-to-one.

    Rule:
    - Detect every open rectangular frame and record its frame color, missing-side direction, and full interior window.
    - Read the external rectangular color ring clockwise starting from its top-left corner to obtain the object order.
    - Keep the first ordered object as its original three-sided frame with its interior pattern, then place each next object's interior window flush against the previous interior window on the side indicated by the previous object's opening.
    - Drop every later frame and crop the result to the tight bounding box of the assembled figure.

    Key insight:
    - Two independent signals must be combined: the external ring gives the color order, while each frame's missing side gives the local attachment direction for the next interior pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    frames = _parse_frames(g, bg)
    ring_order = _ring_order(g, frames, bg)
    by_color = {int(frame["color"]): frame for frame in frames}
    assert set(ring_order) == set(by_color), "ring colors must match frame colors"

    ordered_frames = [by_color[color] for color in ring_order]
    out = _assemble(ordered_frames, bg)
    assert max(out.shape) <= 30, "output must stay within ARC limits"
    return out


def _build_ring(order: list[int], h: int, w: int) -> np.ndarray:
    ring = np.zeros((h, w), dtype=int)
    positions = _rect_cycle_positions(h, w)
    assert len(order) == len(positions)
    for color, (r, c) in zip(order, positions, strict=True):
        ring[r, c] = int(color)
    return ring


def _build_frame_patch(frame_color: int, payload: np.ndarray, side: str) -> np.ndarray:
    ph, pw = map(int, payload.shape)
    patch = np.zeros((ph + 2, pw + 2), dtype=int)
    patch[_open_frame_mask(ph + 2, pw + 2, side)] = int(frame_color)
    patch[1:-1, 1:-1] = payload
    return patch


def _ensure_multicolor(pattern: np.ndarray, palette: list[int], rng: random.Random) -> np.ndarray:
    out = np.asarray(pattern, dtype=int).copy()
    if len(np.unique(out)) >= 2:
        return out
    alt = [color for color in palette if color != int(out[0, 0])]
    if not alt:
        raise ValueError("pattern palette must contain at least two colors")
    out[-1, -1] = rng.choice(alt)
    return out


def _make_pattern(h: int, w: int, palette: list[int], rng: random.Random) -> np.ndarray:
    mode = rng.choice(("rows", "cols", "checker", "diag", "split", "noise"))
    pattern = np.zeros((h, w), dtype=int)
    k = len(palette)

    if mode == "rows":
        for r in range(h):
            pattern[r, :] = palette[(r + rng.randrange(k)) % k]
    elif mode == "cols":
        for c in range(w):
            pattern[:, c] = palette[(c + rng.randrange(k)) % k]
    elif mode == "checker":
        offset = rng.randrange(k)
        for r in range(h):
            for c in range(w):
                pattern[r, c] = palette[(r + c + offset) % k]
    elif mode == "diag":
        offset = rng.randrange(k)
        for r in range(h):
            for c in range(w):
                pattern[r, c] = palette[(2 * r + c + offset) % k]
    elif mode == "split":
        if rng.random() < 0.5:
            split = rng.randint(1, h - 1)
            pattern[:split, :] = palette[0]
            pattern[split:, :] = palette[1 % k]
            if k > 2 and w > 1:
                pattern[:, -1] = palette[2]
        else:
            split = rng.randint(1, w - 1)
            pattern[:, :split] = palette[0]
            pattern[:, split:] = palette[1 % k]
            if k > 2 and h > 1:
                pattern[-1, :] = palette[2]
    else:
        for r in range(h):
            for c in range(w):
                pattern[r, c] = rng.choice(palette)

    if rng.random() < 0.5:
        pattern = np.rot90(pattern, rng.randrange(4))
        if pattern.shape != (h, w):
            pattern = np.rot90(pattern)
    if rng.random() < 0.5:
        pattern = np.fliplr(pattern)
    if rng.random() < 0.5:
        pattern = np.flipud(pattern)
    return _ensure_multicolor(pattern, palette, rng)


def _self_avoiding_walk(length: int, rng: random.Random) -> list[str]:
    for _ in range(200):
        pos = (0, 0)
        visited = {pos}
        sides: list[str] = []
        ok = True
        for _step in range(length):
            options: list[tuple[str, tuple[int, int]]] = []
            for side in rng.sample(list(SIDES), len(SIDES)):
                dr, dc = SIDE_STEPS[side]
                nxt = (pos[0] + dr, pos[1] + dc)
                if nxt not in visited:
                    options.append((side, nxt))
            if not options:
                ok = False
                break
            side, nxt = rng.choice(options)
            sides.append(side)
            pos = nxt
            visited.add(pos)
        if ok:
            return sides
    raise RuntimeError("failed to build self-avoiding direction chain")


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    hh, ww = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(hh, r0 + h + 1)
    cc1 = min(ww, c0 + w + 1)
    return not np.any(occupied[rr0:rr1, cc0:cc1])


def _place_patches(shapes: list[tuple[int, int]], rng: random.Random) -> tuple[int, int, list[tuple[int, int]]]:
    max_h = max(h for h, _w in shapes)
    max_w = max(w for _h, w in shapes)
    total_area = sum(h * w for h, w in shapes)
    base = max(max_h + 2, max_w + 2, int(math.ceil(math.sqrt(total_area * 2.3))))
    base = min(base, 28)

    for _ in range(PLACEMENT_RETRIES):
        h = rng.randint(base, 30)
        w = rng.randint(base, 30)
        occupied = np.zeros((h, w), dtype=bool)
        placements: list[tuple[int, int] | None] = [None] * len(shapes)
        order = sorted(range(len(shapes)), key=lambda idx: shapes[idx][0] * shapes[idx][1], reverse=True)
        ok = True

        for idx in order:
            ph, pw = shapes[idx]
            placed = None
            for _try in range(250):
                r0 = rng.randint(0, h - ph)
                c0 = rng.randint(0, w - pw)
                if _can_place(occupied, r0, c0, ph, pw):
                    occupied[r0 : r0 + ph, c0 : c0 + pw] = True
                    placed = (r0, c0)
                    break
            if placed is None:
                ok = False
                break
            placements[idx] = placed

        if ok:
            return h, w, [(int(r), int(c)) for r, c in placements if r is not None and c is not None]

    raise RuntimeError("failed to place patches on the canvas")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.choice((4, 6))
    ring_h, ring_w = (2, 2) if n_objects == 4 else rng.choice(((2, 3), (3, 2)))
    payload_h = rng.randint(2, 4)
    payload_w = rng.randint(2, 4)

    frame_colors = rng.sample(list(range(1, 10)), n_objects)
    pattern_pool = [color for color in range(1, 10) if color not in frame_colors]
    if len(pattern_pool) < 2:
        return None

    walk_sides = _self_avoiding_walk(n_objects - 1, rng)
    gap_sides = walk_sides + [rng.choice(SIDES)]

    payloads: list[np.ndarray] = []
    for _ in range(n_objects):
        palette_size = rng.randint(2, min(3, len(pattern_pool)))
        palette = rng.sample(pattern_pool, palette_size)
        payloads.append(_make_pattern(payload_h, payload_w, palette, rng))

    object_patches = [
        _build_frame_patch(frame_color, payload, side)
        for frame_color, payload, side in zip(frame_colors, payloads, gap_sides, strict=True)
    ]
    ring_patch = _build_ring(frame_colors, ring_h, ring_w)

    all_patches = object_patches + [ring_patch]
    shapes = [tuple(map(int, patch.shape)) for patch in all_patches]

    try:
        grid_h, grid_w, placements = _place_patches(shapes, rng)
    except RuntimeError:
        return None

    if len(placements) != len(all_patches):
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for patch, (r0, c0) in zip(all_patches, placements, strict=True):
        h, w = map(int, patch.shape)
        mask = patch != BG
        view = grid[r0 : r0 + h, c0 : c0 + w]
        if np.any(mask & (view != BG)):
            return None
        view[mask] = patch[mask]

    try:
        out = main(grid)
    except Exception:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
