from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
TAIL_COLOR = 5
BORDER_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MOTIF_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

DIR_TO_STEP = {
    "up": (-4, 0),
    "right": (0, 4),
    "down": (4, 0),
    "left": (0, -4),
}

DIR_TO_ICON_POS = {
    "up": (0, 2),
    "right": (2, 4),
    "down": (4, 2),
    "left": (2, 0),
}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_to_bbox(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    b = _bbox(mask)
    assert b is not None
    r0, c0, r1, c1 = b
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in (
                    (cr - 1, cc - 1),
                    (cr - 1, cc),
                    (cr - 1, cc + 1),
                    (cr, cc - 1),
                    (cr, cc + 1),
                    (cr + 1, cc - 1),
                    (cr + 1, cc),
                    (cr + 1, cc + 1),
                ):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _rounded_border_positions(h: int, w: int) -> set[tuple[int, int]]:
    pts = set()
    for c in range(1, w - 1):
        pts.add((0, c))
        pts.add((h - 1, c))
    for r in range(1, h - 1):
        pts.add((r, 0))
        pts.add((r, w - 1))
    return pts


def _make_icon(frame_color: int, connector_color: int, direction: str, motif: np.ndarray) -> np.ndarray:
    icon = np.zeros((5, 5), dtype=int)
    icon[0, 1:4] = frame_color
    icon[4, 1:4] = frame_color
    icon[1:4, 0] = frame_color
    icon[1:4, 4] = frame_color
    mh, mw = motif.shape
    assert 1 <= mh <= 3 and 1 <= mw <= 3
    rr = 1 + (3 - mh) // 2
    cc = 1 + (3 - mw) // 2
    icon[rr : rr + mh, cc : cc + mw] = motif
    pr, pc = DIR_TO_ICON_POS[direction]
    icon[pr, pc] = connector_color
    return icon


def _parse_frame_component(grid: np.ndarray, color: int, comp: list[tuple[int, int]]) -> dict | None:
    coords = np.array(comp, dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 5 or w < 5 or h % 2 == 0 or w % 2 == 0:
        return None

    border_local = _rounded_border_positions(h, w)
    comp_local = {(int(r - r0), int(c - c0)) for r, c in comp}
    if not comp_local <= border_local:
        return None

    border_global = {(int(r0 + r), int(c0 + c)) for r, c in border_local}
    if comp_local != {(r, c) for (r, c) in border_local if int(grid[r0 + r, c0 + c]) == color}:
        return None

    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if any(int(grid[r, c]) != BG for r, c in corners):
        return None

    midpoints = {
        "up": (int(r0), int(c0 + w // 2)),
        "right": (int(r0 + h // 2), int(c1)),
        "down": (int(r1), int(c0 + w // 2)),
        "left": (int(r0 + h // 2), int(c0)),
    }

    connector_hits: list[tuple[str, int]] = []
    for name, (mr, mc) in midpoints.items():
        val = int(grid[mr, mc])
        if val == color:
            continue
        if val == BG:
            return None
        connector_hits.append((name, val))

    if len(connector_hits) != 1:
        return None
    direction, connector_color = connector_hits[0]

    for br, bc in border_global:
        if (br, bc) == midpoints[direction]:
            continue
        if int(grid[br, bc]) != color:
            return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if np.any(interior == color) or np.any(interior == connector_color):
        return None
    motif_mask = interior != BG
    if not motif_mask.any():
        return None
    motif = _crop_to_bbox(interior, motif_mask)
    mh, mw = motif.shape
    if mh > 3 or mw > 3:
        return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "border_color": int(color),
        "connector_color": int(connector_color),
        "direction": direction,
        "motif": motif.copy(),
        "icon": _make_icon(int(color), int(connector_color), direction, motif),
    }


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(x) for x in np.unique(g) if int(x) != BG)
    sprites: list[dict] = []
    for color in colors:
        for comp in _components_of_color(g, color):
            parsed = _parse_frame_component(g, color, comp)
            if parsed is not None:
                sprites.append(parsed)

    assert len(sprites) in (3, 4), "expected 3 or 4 framed sprites"
    bboxes = [s["bbox"] for s in sprites]
    for i, a in enumerate(bboxes):
        for b in bboxes[i + 1 :]:
            assert a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1], "sprites must not overlap"
    border_colors = [s["border_color"] for s in sprites]
    assert len(set(border_colors)) == len(border_colors), "frame colors must be unique"
    return sprites


def _chain_order(sprites: list[dict]) -> list[dict]:
    by_border = {s["border_color"]: s for s in sprites}
    referenced = {s["connector_color"] for s in sprites if s["connector_color"] in by_border}
    heads = [s for s in sprites if s["border_color"] not in referenced]
    assert len(heads) == 1, "expected exactly one chain head"

    ordered: list[dict] = []
    seen: set[int] = set()
    cur = heads[0]
    while True:
        ordered.append(cur)
        seen.add(cur["border_color"])
        nxt_color = cur["connector_color"]
        if nxt_color not in by_border:
            break
        cur = by_border[nxt_color]
        assert cur["border_color"] not in seen, "chain must not cycle"

    assert len(ordered) == len(sprites), "chain must visit every sprite exactly once"
    assert ordered[-1]["connector_color"] not in by_border, "chain tail must terminate outside border colors"
    return ordered


def _assemble_chain(ordered: list[dict]) -> np.ndarray:
    positions = [(0, 0)]
    for sprite in ordered[:-1]:
        pr, pc = positions[-1]
        dr, dc = DIR_TO_STEP[sprite["direction"]]
        positions.append((int(pr + dr), int(pc + dc)))

    min_r = min(r for r, _ in positions)
    min_c = min(c for _, c in positions)
    max_r = max(r for r, _ in positions)
    max_c = max(c for _, c in positions)
    out = np.zeros((max_r - min_r + 5, max_c - min_c + 5), dtype=int)

    for sprite, (r, c) in zip(ordered, positions):
        icon = sprite["icon"].copy()
        pr, pc = DIR_TO_ICON_POS[sprite["direction"]]
        icon[pr, pc] = sprite["border_color"]
        rr = r - min_r
        cc = c - min_c
        mask = icon != BG
        out[rr : rr + 5, cc : cc + 5][mask] = icon[mask]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Order framed sprites by connector-color chain, normalize each to a 5x5 icon, and dock them along the connector directions.

    Input structure:
    - Black background with 3 or 4 separated sprites, each enclosed by a rounded rectangular frame whose corners stay black.
    - Every sprite has a unique frame color, a sparse interior motif, and exactly one border midpoint replaced by a connector color.
    - Connector colors form one directed chain: one frame color is never referenced, and one terminal connector color does not match any frame color.

    Rule:
    - Find the sprites and recover the chain order by following `current connector color -> next frame color`, starting from the unique unreferenced frame color.
    - For each sprite, crop the interior motif to its tight bbox, center it inside a 3x3 area, and wrap it in a rounded 5x5 frame of the sprite's frame color while keeping the connector on the corresponding side midpoint.
    - Starting from the chain head, place each next 5x5 icon in the direction indicated by the previous sprite's connector so adjacent icons share one border row or column; then recolor connector pixels back to their own frame colors and tight-crop.

    Key insight:
    - The output needs two coupled clues at once: color matching gives the chain order, while the connector side on each normalized icon gives the docking direction.
    """
    sprites = _extract_sprites(np.asarray(grid, dtype=int))
    ordered = _chain_order(sprites)
    return _assemble_chain(ordered)


def _choose_motif_shape(rng: random.Random) -> tuple[int, int, int]:
    mh = rng.randint(1, 3)
    mw = rng.randint(1, 3)
    min_count = 1 if (mh, mw) == (1, 1) else 2
    count = rng.randint(min_count, mh * mw)
    return mh, mw, count


def _full_bbox_cells(rng: random.Random, mh: int, mw: int, count: int) -> list[tuple[int, int]]:
    if (mh, mw) == (1, 1):
        return [(0, 0)]
    if mh == 1:
        required = [(0, 0), (0, mw - 1)]
    elif mw == 1:
        required = [(0, 0), (mh - 1, 0)]
    else:
        required = [(0, 0), (mh - 1, mw - 1)] if rng.random() < 0.5 else [(0, mw - 1), (mh - 1, 0)]

    cells = list(required)
    pool = [(r, c) for r in range(mh) for c in range(mw) if (r, c) not in cells]
    rng.shuffle(pool)
    cells.extend(pool[: count - len(cells)])
    return cells


def _make_motif(rng: random.Random, forbidden: set[int]) -> np.ndarray:
    mh, mw, count = _choose_motif_shape(rng)
    palette = [c for c in MOTIF_PALETTE if c not in forbidden]
    n_colors = rng.randint(1, min(3, len(palette), count))
    colors = rng.sample(palette, n_colors)
    motif = np.zeros((mh, mw), dtype=int)
    cells = _full_bbox_cells(rng, mh, mw, count)
    rng.shuffle(cells)

    if n_colors == 1:
        for r, c in cells:
            motif[r, c] = colors[0]
    else:
        for idx, (r, c) in enumerate(cells):
            motif[r, c] = colors[idx] if idx < n_colors else rng.choice(colors)
    return motif


def _draw_rounded_frame(grid: np.ndarray, color: int) -> None:
    grid[0, 1:-1] = color
    grid[-1, 1:-1] = color
    grid[1:-1, 0] = color
    grid[1:-1, -1] = color


def _make_raw_sprite(
    rng: random.Random,
    frame_color: int,
    connector_color: int,
    direction: str,
    motif: np.ndarray,
) -> np.ndarray:
    mh, mw = motif.shape
    height = rng.choice([5, 7, 9])
    width = rng.choice([5, 7, 9])
    while height - 2 < mh or width - 2 < mw:
        height = rng.choice([5, 7, 9])
        width = rng.choice([5, 7, 9])

    sprite = np.zeros((height, width), dtype=int)
    _draw_rounded_frame(sprite, frame_color)
    pr, pc = {
        "up": (0, width // 2),
        "right": (height // 2, width - 1),
        "down": (height - 1, width // 2),
        "left": (height // 2, 0),
    }[direction]
    sprite[pr, pc] = connector_color

    r_off = rng.randint(0, height - 2 - mh)
    c_off = rng.randint(0, width - 2 - mw)
    sprite[1 + r_off : 1 + r_off + mh, 1 + c_off : 1 + c_off + mw] = motif
    return sprite


def _make_walk(rng: random.Random, n: int) -> list[str]:
    dirs = list(DIR_TO_STEP)
    for _ in range(200):
        path_dirs: list[str] = []
        pos = (0, 0)
        seen = {pos}
        prev = None
        ok = True
        for step_idx in range(n - 1):
            candidates = []
            for name in dirs:
                dr, dc = {
                    "up": (-1, 0),
                    "right": (0, 1),
                    "down": (1, 0),
                    "left": (0, -1),
                }[name]
                nxt = (pos[0] + dr, pos[1] + dc)
                if nxt in seen:
                    continue
                candidates.append((name, nxt))
            if not candidates:
                ok = False
                break
            if prev is not None:
                turners = [cand for cand in candidates if cand[0] != prev]
                if turners and (step_idx == n - 2 or rng.random() < 0.7):
                    candidates = turners
            name, nxt = rng.choice(candidates)
            path_dirs.append(name)
            seen.add(nxt)
            pos = nxt
            prev = name
        if ok and len(set(path_dirs)) > 1:
            return path_dirs
    raise RuntimeError("failed to build turning walk")


def _place_non_overlapping(rng: random.Random, sprites: list[np.ndarray], h: int, w: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    for _ in range(300):
        order = list(range(len(sprites)))
        rng.shuffle(order)
        placements: dict[int, tuple[int, int]] = {}
        blocked = np.zeros((h, w), dtype=bool)
        ok = True
        for idx in order:
            sh, sw = sprites[idx].shape
            cells: list[tuple[int, int]] = []
            for r in range(h - sh + 1):
                for c in range(w - sw + 1):
                    r0 = max(0, r - 1)
                    c0 = max(0, c - 1)
                    r1 = min(h, r + sh + 1)
                    c1 = min(w, c + sw + 1)
                    if blocked[r0:r1, c0:c1].any():
                        continue
                    cells.append((r, c))
            if not cells:
                ok = False
                break
            placements[idx] = rng.choice(cells)
            r, c = placements[idx]
            blocked[r : r + sh, c : c + sw] = True
        if not ok:
            continue

        grid = np.zeros((h, w), dtype=int)
        locs = [placements[i] for i in range(len(sprites))]
        for sprite, (r, c) in zip(sprites, locs):
            mask = sprite != BG
            grid[r : r + sprite.shape[0], c : c + sprite.shape[1]][mask] = sprite[mask]
        return grid, locs
    raise RuntimeError("failed to place sprites")


def _wrong_position_order_output(sprites: list[dict]) -> np.ndarray:
    ordered = sorted(sprites, key=lambda s: (s["bbox"][0], s["bbox"][1]))
    return _assemble_chain(ordered)


def _build_instance(rng: random.Random) -> np.ndarray:
    n = rng.choice([3, 4])
    border_colors = rng.sample(list(BORDER_PALETTE), n)
    walk_dirs = _make_walk(rng, n)
    tail_dir = rng.choice(list(DIR_TO_STEP))
    all_dirs = walk_dirs + [tail_dir]

    motifs: list[np.ndarray] = []
    motif_sigs: set[tuple] = set()
    forbidden_global = set(border_colors) | {TAIL_COLOR}
    while len(motifs) < n:
        motif = _make_motif(rng, forbidden_global)
        sig = (motif.shape, tuple(int(x) for x in motif.flatten()))
        if sig in motif_sigs:
            continue
        motif_sigs.add(sig)
        motifs.append(motif)

    sprites = []
    for i in range(n):
        connector_color = border_colors[i + 1] if i < n - 1 else TAIL_COLOR
        sprite = _make_raw_sprite(rng, border_colors[i], connector_color, all_dirs[i], motifs[i])
        sprites.append(sprite)

    canvas_h = rng.randint(20, 25)
    canvas_w = rng.randint(20, 25)
    grid, _locs = _place_non_overlapping(rng, sprites, canvas_h, canvas_w)

    parsed = _extract_sprites(grid)
    ordered = _chain_order(parsed)
    out = _assemble_chain(ordered)

    if max(out.shape) > 30:
        raise RuntimeError("output too large")
    if [s["border_color"] for s in ordered] != border_colors:
        raise RuntimeError("parser recovered wrong chain")
    if len({s["direction"] for s in ordered[:-1]}) < 2:
        raise RuntimeError("path failed to turn")

    pos_sorted = [s["border_color"] for s in sorted(parsed, key=lambda s: (s["bbox"][0], s["bbox"][1]))]
    if pos_sorted == border_colors or pos_sorted == list(reversed(border_colors)):
        raise RuntimeError("positional shortcut still available")
    if np.array_equal(_wrong_position_order_output(parsed), out):
        raise RuntimeError("position-sorted assembly matches true output")

    fresh_out = main(grid)
    if not np.array_equal(fresh_out, out):
        raise RuntimeError("oracle mismatch")
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        try:
            return _build_instance(rng)
        except Exception:
            continue
    raise RuntimeError("failed to generate valid input")
