from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
SIDE_TOP_BOTTOM = ("top", "bottom")
SIDE_LEFT_RIGHT = ("left", "right")
ALL_SIDES = ("top", "bottom", "left", "right")

BG = 0
MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _parse_sprite(g: np.ndarray, coords: list[tuple[int, int]], bg: int) -> dict | None:
    h, w = g.shape
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2) or len(coords) != 4:
        return None
    if not (r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
        return None

    sub = g[r0:r1 + 1, c0:c1 + 1]
    if np.any(sub == bg):
        return None
    a, b = int(sub[0, 0]), int(sub[0, 1])
    c, d = int(sub[1, 0]), int(sub[1, 1])
    if not (a == d and b == c and a != b):
        return None

    counts = Counter(int(v) for v in sub.flat)
    if sorted(counts.values()) != [2, 2]:
        return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "coords": [(int(r), int(c)) for r, c in coords],
        "colors": {int(a), int(b)},
        "main_diag_color": int(a),
        "anti_diag_color": int(b),
    }


def _parse_frame(g: np.ndarray, coords: list[tuple[int, int]], bg: int) -> dict | None:
    colors = {int(g[r, c]) for r, c in coords}
    if len(colors) != 1:
        return None

    color = next(iter(colors))
    r0, c0, r1, c1 = _bbox(coords)
    side = r1 - r0 + 1
    if side != c1 - c0 + 1 or side < 4:
        return None

    sub = g[r0:r1 + 1, c0:c1 + 1]
    rows = np.arange(side)[:, None]
    cols = np.arange(side)[None, :]
    perimeter = (rows == 0) | (rows == side - 1) | (cols == 0) | (cols == side - 1)

    if not np.all(sub[1:-1, 1:-1] == bg):
        return None

    gap_positions = np.argwhere(perimeter & (sub == bg))
    if len(gap_positions) != 1:
        return None
    gr, gc = map(int, gap_positions[0])
    if (gr in (0, side - 1)) and (gc in (0, side - 1)):
        return None

    perim_without_gap = perimeter.copy()
    perim_without_gap[gr, gc] = False
    if not np.all(sub[perim_without_gap] == color):
        return None
    if np.any(sub[~perimeter] != bg):
        return None

    if gr == 0:
        gap_side = "top"
    elif gr == side - 1:
        gap_side = "bottom"
    elif gc == 0:
        gap_side = "left"
    else:
        gap_side = "right"

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "color": int(color),
        "gap_side": gap_side,
        "gap_rc": (int(r0 + gr), int(c0 + gc)),
    }


def _half_assignment(frame: dict, frame_on_main: bool, fill_color: int) -> tuple[tuple[slice, slice], tuple[slice, slice], int, int]:
    r0, c0, r1, c1 = frame["bbox"]
    gap_r, gap_c = frame["gap_rc"]
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    near_color = int(fill_color) if frame_on_main else int(frame["color"])
    far_color = int(frame["color"]) if frame_on_main else int(fill_color)

    if frame["gap_side"] in SIDE_TOP_BOTTOM:
        mid = inner_w // 2
        near_is_left = (gap_c - (c0 + 1)) < mid
        if near_is_left:
            near_slice = (slice(0, inner_h), slice(0, mid))
            far_slice = (slice(0, inner_h), slice(mid, inner_w))
        else:
            near_slice = (slice(0, inner_h), slice(mid, inner_w))
            far_slice = (slice(0, inner_h), slice(0, mid))
    else:
        mid = inner_h // 2
        near_is_top = (gap_r - (r0 + 1)) < mid
        if near_is_top:
            near_slice = (slice(0, mid), slice(0, inner_w))
            far_slice = (slice(mid, inner_h), slice(0, inner_w))
        else:
            near_slice = (slice(mid, inner_h), slice(0, inner_w))
            far_slice = (slice(0, mid), slice(0, inner_w))

    return near_slice, far_slice, near_color, far_color


def _fill_frame_interior(out: np.ndarray, frame: dict, frame_on_main: bool, fill_color: int) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    inner = out[r0 + 1:r1, c0 + 1:c1]
    near_slice, far_slice, near_color, far_color = _half_assignment(frame, frame_on_main, fill_color)
    inner[far_slice] = int(far_color)
    inner[near_slice] = int(near_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow square frame using the edge sprite that shares its border color.

    Input structure:
    - The background is black and covers most of the grid.
    - Several single-color hollow square frames appear in the interior; each frame has exactly one 1-cell gap on one border side.
    - Several 2x2 two-color sprites sit on the grid edge; each sprite uses one frame color plus one unique partner color, arranged on opposite diagonals.

    Rule:
    - Detect the hollow frames and the 2x2 edge sprites, then pair each frame with the unique sprite that contains the frame's border color.
    - The gap side chooses the split axis inside the frame: a top/bottom gap means split the interior into left/right halves, while a left/right gap means split it into top/bottom halves.
    - The gap's position along that side tells which half is nearer to the gap; the other half is farther away.
    - If the frame color occupies the sprite's main diagonal, put the sprite-only color in the near half and the frame color in the far half; otherwise reverse those two half-colors. Keep the frame border and its gap unchanged, and erase all sprites.

    Key insight:
    - Solving requires combining two geometric cues: the frame gap determines how to partition the interior, and the sprite's diagonal color arrangement determines which half receives the new fill color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    comps = _components(g != bg)
    frames: list[dict] = []
    sprites: list[dict] = []
    for coords in comps:
        sprite = _parse_sprite(g, coords, bg)
        if sprite is not None:
            sprites.append(sprite)
            continue
        frame = _parse_frame(g, coords, bg)
        if frame is not None:
            frames.append(frame)

    assert frames and sprites
    assert len(frames) == len(sprites)

    frame_colors = {int(frame["color"]) for frame in frames}
    sprite_map: dict[int, dict] = {}
    for sprite in sprites:
        matching = [color for color in sprite["colors"] if color in frame_colors]
        assert len(matching) == 1
        frame_color = int(matching[0])
        fill_color = next(color for color in sprite["colors"] if color != frame_color)
        sprite_map[frame_color] = {
            "fill_color": int(fill_color),
            "frame_on_main": int(sprite["main_diag_color"]) == int(frame_color),
            "coords": sprite["coords"],
        }

    assert set(sprite_map) == frame_colors

    out = g.copy()
    for sprite in sprites:
        for r, c in sprite["coords"]:
            out[r, c] = int(bg)

    for frame in frames:
        match = sprite_map[int(frame["color"])]
        _fill_frame_interior(
            out,
            frame,
            frame_on_main=bool(match["frame_on_main"]),
            fill_color=int(match["fill_color"]),
        )

    return out


def _render_frame_patch(size: int, color: int, gap_side: str, gap_idx: int) -> np.ndarray:
    patch = np.full((size, size), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if gap_side == "top":
        patch[0, gap_idx] = BG
    elif gap_side == "bottom":
        patch[-1, gap_idx] = BG
    elif gap_side == "left":
        patch[gap_idx, 0] = BG
    else:
        patch[gap_idx, -1] = BG
    return patch


def _render_sprite_patch(frame_color: int, fill_color: int, frame_on_main: bool) -> np.ndarray:
    if frame_on_main:
        return np.array(
            [[frame_color, fill_color], [fill_color, frame_color]],
            dtype=int,
        )
    return np.array(
        [[fill_color, frame_color], [frame_color, fill_color]],
        dtype=int,
    )


def _mark_occ(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occ[r0:r0 + h, c0:c0 + w] = True


def _fits(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    oh, ow = occ.shape
    if r0 < 0 or c0 < 0 or r0 + h > oh or c0 + w > ow:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(oh, r0 + h + pad)
    cc1 = min(ow, c0 + w + pad)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _frame_positions(h: int, w: int, size: int, margin: int) -> list[tuple[int, int]]:
    pos = []
    for r0 in range(margin, h - size - margin + 1):
        for c0 in range(margin, w - size - margin + 1):
            pos.append((int(r0), int(c0)))
    random.shuffle(pos)
    return pos


def _sprite_positions(h: int, w: int) -> list[tuple[str, int, int]]:
    positions: list[tuple[str, int, int]] = []
    for c0 in range(1, w - 2):
        positions.append(("top", 0, c0))
        positions.append(("bottom", h - 2, c0))
    for r0 in range(1, h - 2):
        positions.append(("left", r0, 0))
        positions.append(("right", r0, w - 2))
    random.shuffle(positions)
    return positions


def _choose_gap(size: int, side: str, rng: random.Random) -> int:
    inner = size - 2
    half = rng.choice([0, 1])
    if half == 0:
        return rng.randint(1, inner // 2)
    return rng.randint(inner // 2 + 1, inner)


def _balanced_sides(count: int, rng: random.Random) -> list[str]:
    if count == 2:
        sides = [rng.choice(SIDE_TOP_BOTTOM), rng.choice(SIDE_LEFT_RIGHT)]
    elif count == 3:
        sides = [rng.choice(SIDE_TOP_BOTTOM), rng.choice(SIDE_LEFT_RIGHT), rng.choice(ALL_SIDES)]
    else:
        sides = list(ALL_SIDES)
    rng.shuffle(sides)
    return sides


def _balanced_diagonals(count: int, rng: random.Random) -> list[bool]:
    if count == 2:
        flags = [True, False]
    elif count == 3:
        flags = [True, False, rng.choice([True, False])]
    else:
        flags = [True, True, False, False]
    rng.shuffle(flags)
    return flags


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    target = grid[r0:r0 + h, c0:c0 + w]
    mask = patch != BG
    target[mask] = patch[mask]


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    frame_count = rng.randint(2, 4)
    frame_size = rng.choice([6, 8])

    if frame_count == 2:
        h = rng.randint(18, 30)
        w = rng.randint(18, 30)
    elif frame_count == 3:
        h = rng.randint(20, 30)
        w = rng.randint(20, 30)
    else:
        h = rng.randint(24, 30)
        w = rng.randint(24, 30)

    colors = list(range(1, 10))
    rng.shuffle(colors)
    needed = frame_count * 2
    if needed > len(colors):
        return None
    frame_colors = colors[:frame_count]
    fill_colors = colors[frame_count:needed]

    sides = _balanced_sides(frame_count, rng)
    diagonals = _balanced_diagonals(frame_count, rng)

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    frame_specs = []
    for idx in range(frame_count):
        gap_side = sides[idx]
        gap_idx = _choose_gap(frame_size, gap_side, rng)
        frame_specs.append(
            {
                "color": int(frame_colors[idx]),
                "fill_color": int(fill_colors[idx]),
                "gap_side": gap_side,
                "gap_idx": int(gap_idx),
                "frame_on_main": bool(diagonals[idx]),
            }
        )

    for spec in frame_specs:
        placed = False
        for r0, c0 in _frame_positions(h, w, frame_size, margin=3):
            if not _fits(occ, r0, c0, frame_size, frame_size, pad=1):
                continue
            patch = _render_frame_patch(frame_size, spec["color"], spec["gap_side"], spec["gap_idx"])
            _paint_patch(inp, patch, r0, c0)
            _paint_patch(out, patch, r0, c0)
            frame = _parse_frame(out, [(r, c) for r in range(r0, r0 + frame_size) for c in range(c0, c0 + frame_size) if patch[r - r0, c - c0] != BG], BG)
            assert frame is not None
            _fill_frame_interior(out, frame, spec["frame_on_main"], spec["fill_color"])
            spec["frame_bbox"] = (int(r0), int(c0), int(r0 + frame_size - 1), int(c0 + frame_size - 1))
            _mark_occ(occ, r0, c0, frame_size, frame_size)
            placed = True
            break
        if not placed:
            return None

    sprite_positions = _sprite_positions(h, w)
    for spec in frame_specs:
        sprite = _render_sprite_patch(spec["color"], spec["fill_color"], spec["frame_on_main"])
        placed = False
        for _side, r0, c0 in sprite_positions:
            if not _fits(occ, r0, c0, 2, 2, pad=1):
                continue
            _paint_patch(inp, sprite, r0, c0)
            _mark_occ(occ, r0, c0, 2, 2)
            placed = True
            sprite_positions = [pos for pos in sprite_positions if pos[1] != r0 or pos[2] != c0]
            break
        if not placed:
            return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
