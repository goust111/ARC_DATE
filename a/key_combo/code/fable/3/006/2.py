from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _mode(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors(r: int, c: int, connectivity: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if connectivity == 4 and abs(dr) + abs(dc) != 1:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    mask = patch != bg
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    view[mask] = patch[mask]


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _find_frames(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames = []
    for color in sorted(set(map(int, g.ravel())) - {int(bg)}):
        for coords in _components(g == color, connectivity=4):
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            expected_area = 2 * h + 2 * w - 4
            if len(coords) != expected_area:
                continue
            on_edge = (
                (coords[:, 0] == r0)
                | (coords[:, 0] == r1)
                | (coords[:, 1] == c0)
                | (coords[:, 1] == c1)
            )
            if not bool(np.all(on_edge)):
                continue
            if not (
                np.all(g[r0, c0 : c1 + 1] == color)
                and np.all(g[r1, c0 : c1 + 1] == color)
                and np.all(g[r0 : r1 + 1, c0] == color)
                and np.all(g[r0 : r1 + 1, c1] == color)
            ):
                continue
            if np.any(g[r0 + 1 : r1, c0 + 1 : c1] == color):
                continue
            frames.append({"bbox": (r0, c0, r1, c1), "color": int(color)})
    return frames


def _interior(g: np.ndarray, frame: dict[str, object]) -> np.ndarray:
    r0, c0, r1, c1 = frame["bbox"]
    return g[r0 + 1 : r1, c0 + 1 : c1]


def _is_lr_symmetric(coords: np.ndarray, colors: np.ndarray) -> bool:
    if len(coords) == 0:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    by_pos = {(int(r), int(c)): int(col) for (r, c), col in zip(coords, colors)}
    for (r, c), col in by_pos.items():
        mirrored = (r, c0 + c1 - c)
        if by_pos.get(mirrored) != col:
            return False
    return True


def _symmetry_anchor(sprite_grid: np.ndarray, coords: np.ndarray) -> Coord:
    colors = np.array([int(sprite_grid[r, c]) for r, c in coords], dtype=int)
    anchors: list[Coord] = []
    for i, (r, c) in enumerate(coords):
        remaining_coords = np.delete(coords, i, axis=0)
        remaining_colors = np.delete(colors, i, axis=0)
        if _is_lr_symmetric(remaining_coords, remaining_colors):
            anchors.append((int(r), int(c)))
    assert len(anchors) == 1, "each sprite must have exactly one symmetry-breaking pixel"
    return anchors[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each source sprite onto the target marker whose color matches its symmetry-breaking anchor.

    Input structure:
    - The grid contains exactly two monochrome bordered rectangles on a dominant background.
    - One rectangle is a source area holding separated multicolor sprites; the other is a target area holding isolated one-pixel markers.
    - In every source sprite, removing exactly one pixel makes the remaining colored sprite left-right mirror symmetric, and that anchor color matches one target marker.

    Rule:
    - Identify the target rectangle as the framed area whose interior objects are all isolated single pixels, then clear those markers.
    - For each source sprite, find the unique pixel whose deletion leaves a left-right symmetric colored sprite in the sprite's own remaining bbox.
    - Translate the whole sprite so that this anchor pixel lands on the same-colored target marker, and stamp all sprite pixels into the target rectangle.

    Key insight:
    - The target color selects the sprite, but the placement point is the pixel that alone breaks mirror symmetry, not the rarest color or an arbitrary same-colored cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _mode(g)

    frames = _find_frames(g, bg)
    assert len(frames) == 2, "expected exactly two bordered rectangles"

    frame_info = []
    for frame in frames:
        inner = _interior(g, frame)
        comps = _components(inner != bg, connectivity=8)
        frame_info.append((frame, inner, comps))

    target_infos = [info for info in frame_info if info[2] and all(len(comp) == 1 for comp in info[2])]
    assert len(target_infos) == 1, "exactly one rectangle must contain only isolated markers"
    target_frame, target_inner, target_comps = target_infos[0]
    source_frame, source_inner, source_comps = [info for info in frame_info if info[0] is not target_frame][0]

    markers: dict[int, Coord] = {}
    for comp in target_comps:
        r, c = map(int, comp[0])
        color = int(target_inner[r, c])
        assert color != bg and color not in markers
        markers[color] = (r + 1, c + 1)

    assert source_comps and len(source_comps) == len(markers)

    tr0, tc0, tr1, tc1 = target_frame["bbox"]
    out = g[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()
    out[1:-1, 1:-1] = bg

    used_marker_colors: set[int] = set()
    for comp in source_comps:
        assert len(comp) > 1
        anchor = _symmetry_anchor(source_inner, comp)
        ar, ac = anchor
        anchor_color = int(source_inner[ar, ac])
        assert anchor_color in markers and anchor_color not in used_marker_colors
        used_marker_colors.add(anchor_color)
        mr, mc = markers[anchor_color]

        for sr, sc in comp:
            sr = int(sr)
            sc = int(sc)
            rr = mr + sr - ar
            cc = mc + sc - ac
            assert 1 <= rr < out.shape[0] - 1 and 1 <= cc < out.shape[1] - 1
            assert int(out[rr, cc]) == bg
            out[rr, cc] = int(source_inner[sr, sc])

    assert used_marker_colors == set(markers)
    return out


# --------------- generator ---------------

MAX_ATTEMPTS = 500
GAP = 1


def _dilate(mask: np.ndarray, pad: int = 1) -> np.ndarray:
    if pad <= 0:
        return mask.copy()
    h, w = mask.shape
    out = mask.copy()
    pts = np.argwhere(mask)
    for r, c in pts:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                rr, cc = int(r) + dr, int(c) + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = True
    return out


def _place_patch(region: np.ndarray, patch: np.ndarray, bg: int, blocked: np.ndarray) -> tuple[int, int] | None:
    h, w = patch.shape
    rh, rw = region.shape
    mask = patch != bg
    candidates = [(r, c) for r in range(rh - h + 1) for c in range(rw - w + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        target = blocked[r : r + h, c : c + w]
        if np.any(target & mask):
            continue
        _paste(region, patch, r, c, bg)
        placed = np.zeros_like(region, dtype=bool)
        placed[r : r + h, c : c + w] |= mask
        blocked |= _dilate(placed, 1)
        return r, c
    return None


def _reserve_stamp(mask_shape: tuple[int, int], patch: np.ndarray, bg: int, blocked: np.ndarray) -> tuple[int, int] | None:
    h, w = patch.shape
    rh, rw = mask_shape
    mask = patch != bg
    candidates = [(r, c) for r in range(rh - h + 1) for c in range(rw - w + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        if np.any(blocked[r : r + h, c : c + w] & mask):
            continue
        placed = np.zeros(mask_shape, dtype=bool)
        placed[r : r + h, c : c + w] |= mask
        blocked |= _dilate(placed, 1)
        return r, c
    return None


def _make_sprite(bg: int, anchor_color: int, filler_color: int, side: str) -> tuple[np.ndarray, Coord]:
    for _ in range(100):
        h = random.randint(3, 5)
        core_w = random.choice([3, 5])
        axis = core_w // 2
        anchor_row = random.randrange(h)
        arms = [random.randint(0, axis) for _ in range(h)]
        arms[anchor_row] = axis
        if sum(1 + 2 * a for a in arms) < 7:
            continue

        patch = np.full((h, core_w + 1), bg, dtype=int)
        offset = 1 if side == "left" else 0
        for r, arm in enumerate(arms):
            for d in range(arm + 1):
                patch[r, offset + axis - d] = anchor_color
                patch[r, offset + axis + d] = anchor_color

        anchor = (anchor_row, 0) if side == "left" else (anchor_row, core_w)
        patch[anchor] = anchor_color

        axis_cells = [(r, offset + axis) for r in range(h) if r != anchor_row]
        rare = random.choice(axis_cells or [(anchor_row, offset + axis)])
        patch[rare] = filler_color

        if random.choice([False, True]):
            row = random.randrange(h)
            if arms[row] >= 1:
                d = random.randint(1, arms[row])
                patch[row, offset + axis - d] = filler_color
                patch[row, offset + axis + d] = filler_color

        coords = np.argwhere(patch != bg)
        try:
            found = _symmetry_anchor(patch, coords)
        except AssertionError:
            continue
        if found != anchor:
            continue
        counts = Counter(map(int, patch[patch != bg]))
        if counts[anchor_color] <= 1:
            continue
        if min(counts.values()) >= counts[anchor_color]:
            continue
        return patch, anchor
    raise RuntimeError("failed to make a valid sprite")


def _frame_sizes(sprites: list[np.ndarray], layout: str) -> tuple[int, int] | None:
    heights = [p.shape[0] for p in sprites]
    widths = [p.shape[1] for p in sprites]
    n = len(sprites)
    if layout == "side":
        min_h = sum(heights) + n + 1
        min_w = max(widths) + 2
        if min_h > 26 or min_w > 12:
            return None
        return random.randint(max(16, min_h), 26), random.randint(max(9, min_w), 12)
    min_h = max(heights) + 2
    min_w = sum(widths) + n + 1
    if min_h > 12 or min_w > 26:
        return None
    return random.randint(max(9, min_h), 12), random.randint(max(16, min_w), 26)


def _assemble_grid(
    bg: int,
    border: int,
    source_inner: np.ndarray,
    target_markers: dict[int, Coord],
    expected_inner: np.ndarray,
    layout: str,
) -> tuple[np.ndarray, np.ndarray]:
    sih, siw = source_inner.shape
    tih, tiw = expected_inner.shape
    sfh, sfw = sih + 2, siw + 2
    tfh, tfw = tih + 2, tiw + 2

    if layout == "side":
        h = max(sfh, tfh)
        w = sfw + GAP + tfw
        first_source = random.choice([False, True])
        a_r = random.randint(0, h - sfh)
        b_r = random.randint(0, h - tfh)
        if first_source:
            source_pos = (a_r, 0)
            target_pos = (b_r, sfw + GAP)
        else:
            target_pos = (b_r, 0)
            source_pos = (a_r, tfw + GAP)
    else:
        h = sfh + GAP + tfh
        w = max(sfw, tfw)
        first_source = random.choice([False, True])
        a_c = random.randint(0, w - sfw)
        b_c = random.randint(0, w - tfw)
        if first_source:
            source_pos = (0, a_c)
            target_pos = (sfh + GAP, b_c)
        else:
            target_pos = (0, b_c)
            source_pos = (tfh + GAP, a_c)

    grid = np.full((h, w), bg, dtype=int)
    sr, sc = source_pos
    tr, tc = target_pos
    _draw_frame(grid, sr, sc, sfh, sfw, border)
    _draw_frame(grid, tr, tc, tfh, tfw, border)
    grid[sr + 1 : sr + 1 + sih, sc + 1 : sc + 1 + siw] = source_inner
    for color, (mr, mc) in target_markers.items():
        grid[tr + 1 + mr, tc + 1 + mc] = int(color)

    expected = np.full((tfh, tfw), bg, dtype=int)
    _draw_frame(expected, 0, 0, tfh, tfw, border)
    expected[1:-1, 1:-1] = expected_inner
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    colors = list(range(10))
    bg = random.choice(colors)
    colors.remove(bg)
    border = random.choice(colors)
    colors.remove(border)

    n_sprites = random.randint(2, 3)
    picked = random.sample(colors, 2 * n_sprites)
    anchor_colors = picked[:n_sprites]
    filler_colors = picked[n_sprites:]

    sides = ["left", "right"] + [random.choice(["left", "right"]) for _ in range(n_sprites - 2)]
    random.shuffle(sides)
    sprites = []
    anchors = []
    for anchor_color, filler_color, side in zip(anchor_colors, filler_colors, sides):
        patch, anchor = _make_sprite(bg, anchor_color, filler_color, side)
        sprites.append(patch)
        anchors.append(anchor)

    layout = random.choice(["side", "stack"])
    dims = _frame_sizes(sprites, "side" if layout == "side" else "stack")
    if dims is None:
        return None
    inner_h, inner_w = dims

    source_inner = np.full((inner_h, inner_w), bg, dtype=int)
    source_blocked = np.zeros_like(source_inner, dtype=bool)
    for patch in sprites:
        if _place_patch(source_inner, patch, bg, source_blocked) is None:
            return None

    target_inner = np.full((inner_h, inner_w), bg, dtype=int)
    target_blocked = np.zeros_like(target_inner, dtype=bool)
    markers: dict[int, Coord] = {}
    order = list(range(n_sprites))
    random.shuffle(order)
    for idx in order:
        top_left = _reserve_stamp(target_inner.shape, sprites[idx], bg, target_blocked)
        if top_left is None:
            return None
        r0, c0 = top_left
        ar, ac = anchors[idx]
        marker = (r0 + ar, c0 + ac)
        if target_inner[marker] != bg:
            return None
        markers[anchor_colors[idx]] = marker
        _paste(target_inner, sprites[idx], r0, c0, bg)

    grid, expected = _assemble_grid(bg, border, source_inner, markers, target_inner, layout)
    counts = Counter(map(int, grid.ravel()))
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-anchor stamping input")
