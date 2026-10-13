from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME_COUNT = 4
MAX_GENERATION_ATTEMPTS = 400
_RNG = random.Random()


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _mask_from_points(
    pts: list[tuple[int, int]], bbox: tuple[int, int, int, int] | None = None
) -> np.ndarray:
    if bbox is None:
        bbox = _bbox(pts)
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        mask[r - r0, c - c0] = True
    return mask


def _rot_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, (-turns) % 4)


def _mask_sig(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(mask, dtype=np.uint8)
    return arr.shape, arr.tobytes()


def _canonical_rotation_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_mask_sig(np.rot90(mask, k)) for k in range(4))


def _all_rotations_distinct(mask: np.ndarray) -> bool:
    return len({_mask_sig(np.rot90(mask, k)) for k in range(4)}) == 4


def _perimeter_mask(h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=bool)
    out[0, :] = True
    out[-1, :] = True
    out[:, 0] = True
    out[:, -1] = True
    return out


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        color_mask = g == color
        color_mask[0, :] = False
        for pts in _components(color_mask):
            r0, c0, r1, c1 = _bbox(pts)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 5 or w < 5:
                continue
            local = _mask_from_points(pts, (r0, c0, r1, c1))
            if np.array_equal(local, _perimeter_mask(h, w)):
                frames.append(
                    {
                        "color": color,
                        "bbox": (r0, c0, r1, c1),
                    }
                )
    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _find_holes_in_frame(grid: np.ndarray, frame: dict) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = frame["bbox"]
    inner = g[r0 + 1 : r1, c0 + 1 : c1] == BG
    holes: list[dict] = []
    for pts in _components(inner):
        lr0, lc0, lr1, lc1 = _bbox(pts)
        mask = _mask_from_points(pts, (lr0, lc0, lr1, lc1))
        holes.append(
            {
                "bbox": (r0 + 1 + lr0, c0 + 1 + lc0, r0 + 1 + lr1, c0 + 1 + lc1),
                "mask": mask,
            }
        )
    holes.sort(key=lambda hole: hole["bbox"])
    return holes


def _find_sprites(grid: np.ndarray, frames: list[dict]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    mask = g != BG
    mask[0, :] = False
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        mask[r0 : r1 + 1, c0 : c1 + 1] = False

    sprites: list[dict] = []
    for pts in _components(mask):
        colors = {int(g[r, c]) for r, c in pts}
        assert len(colors) == 1
        color = next(iter(colors))
        bbox = _bbox(pts)
        sprites.append(
            {
                "color": color,
                "bbox": bbox,
                "mask": _mask_from_points(pts, bbox),
                "pts": pts,
            }
        )
    sprites.sort(key=lambda sprite: sprite["bbox"])
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend = [int(v) for v in g[0] if int(v) != BG]
    assert legend
    legend_pos = {color: idx for idx, color in enumerate(legend)}

    frames = _find_frames(g)
    assert frames
    sprites = _find_sprites(g, frames)

    out = g.copy()
    used: set[int] = set()
    for frame in frames:
        turns = legend_pos[frame["color"]] % 4
        holes = _find_holes_in_frame(g, frame)
        for hole in holes:
            matches = []
            for idx, sprite in enumerate(sprites):
                if idx in used:
                    continue
                rotated = _rot_cw(sprite["mask"], turns)
                if rotated.shape == hole["mask"].shape and np.array_equal(rotated, hole["mask"]):
                    matches.append(idx)
            assert len(matches) == 1
            sprite = sprites[matches[0]]
            used.add(matches[0])

            for r, c in sprite["pts"]:
                out[r, c] = BG

            hr0, hc0, _, _ = hole["bbox"]
            rotated = _rot_cw(sprite["mask"], turns)
            patch = out[hr0 : hr0 + rotated.shape[0], hc0 : hc0 + rotated.shape[1]]
            patch[rotated] = sprite["color"]

    return out


def _random_connected_mask() -> np.ndarray | None:
    for _ in range(200):
        h = _RNG.randint(2, 4)
        w = _RNG.randint(2, 4)
        max_cells = min(h * w - 1, 7)
        if max_cells < 3:
            continue
        target = _RNG.randint(3, max_cells)
        start = (_RNG.randrange(h), _RNG.randrange(w))
        cells = {start}
        frontier = set()
        for nr, nc in _neighbors4(*start):
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))

        while len(cells) < target and frontier:
            cell = _RNG.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nr, nc in _neighbors4(*cell):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) != target:
            continue

        mask = _mask_from_points(list(cells))
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) == mask.size:
            continue
        if not _all_rotations_distinct(mask):
            continue
        return mask
    return None


def _layout_holes(hole_masks: list[np.ndarray]) -> tuple[int, int, list[tuple[int, int, np.ndarray]]]:
    margin = 1
    if len(hole_masks) == 1:
        hole = hole_masks[0]
        hh, hw = hole.shape
        extra_h = _RNG.randint(0, 2)
        extra_w = _RNG.randint(0, 2)
        inner_h = hh + 2 * margin + extra_h
        inner_w = hw + 2 * margin + extra_w
        top = _RNG.randint(margin, inner_h - margin - hh)
        left = _RNG.randint(margin, inner_w - margin - hw)
        return inner_h, inner_w, [(top, left, hole)]

    first, second = hole_masks
    h1, w1 = first.shape
    h2, w2 = second.shape
    if _RNG.random() < 0.5:
        gap = _RNG.randint(2, 3)
        inner_h = max(h1, h2) + 2 * margin + _RNG.randint(0, 2)
        inner_w = w1 + w2 + gap + 2 * margin
        top1 = _RNG.randint(margin, inner_h - margin - h1)
        top2 = _RNG.randint(margin, inner_h - margin - h2)
        left1 = margin
        left2 = margin + w1 + gap
        placements = [(top1, left1, first), (top2, left2, second)]
    else:
        gap = _RNG.randint(2, 3)
        inner_h = h1 + h2 + gap + 2 * margin
        inner_w = max(w1, w2) + 2 * margin + _RNG.randint(0, 2)
        top1 = margin
        top2 = margin + h1 + gap
        left1 = _RNG.randint(margin, inner_w - margin - w1)
        left2 = _RNG.randint(margin, inner_w - margin - w2)
        placements = [(top1, left1, first), (top2, left2, second)]
    return inner_h, inner_w, placements


def _draw_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    border_color: int,
    filler_color: int,
    inner_h: int,
    inner_w: int,
    hole_layout: list[tuple[int, int, np.ndarray]],
) -> list[dict]:
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    grid[top, left : left + frame_w] = border_color
    grid[top + frame_h - 1, left : left + frame_w] = border_color
    grid[top : top + frame_h, left] = border_color
    grid[top : top + frame_h, left + frame_w - 1] = border_color
    grid[top + 1 : top + frame_h - 1, left + 1 : left + frame_w - 1] = filler_color

    holes: list[dict] = []
    for hole_top, hole_left, hole_mask in hole_layout:
        hh, hw = hole_mask.shape
        rr = top + 1 + hole_top
        cc = left + 1 + hole_left
        region = grid[rr : rr + hh, cc : cc + hw]
        region[hole_mask] = BG
        holes.append({"bbox": (rr, cc, rr + hh - 1, cc + hw - 1), "mask": hole_mask})
    return holes


def _place_legend(grid: np.ndarray, legend_colors: list[int], legend_gaps: list[int]) -> None:
    cols = [0]
    for gap in legend_gaps:
        cols.append(cols[-1] + 1 + gap)
    span = cols[-1] + 1
    start = _RNG.randint(0, grid.shape[1] - span)
    for idx, color in enumerate(legend_colors):
        grid[0, start + cols[idx]] = color


def _expanded_slice(start: int, end: int, limit: int) -> tuple[int, int]:
    return max(0, start - 1), min(limit, end + 1)


def _place_sprite(
    grid: np.ndarray,
    occupied: np.ndarray,
    sprite_mask: np.ndarray,
    color: int,
) -> tuple[int, int] | None:
    h, w = grid.shape
    sh, sw = sprite_mask.shape
    for _ in range(300):
        top = _RNG.randint(2, h - sh)
        left = _RNG.randint(0, w - sw)
        rr0, rr1 = _expanded_slice(top, top + sh, h)
        cc0, cc1 = _expanded_slice(left, left + sw, w)
        if occupied[rr0:rr1, cc0:cc1].any():
            continue
        patch = grid[top : top + sh, left : left + sw]
        patch[sprite_mask] = color
        occupied[rr0:rr1, cc0:cc1] = True
        return top, left
    return None


def _expected_output(inp: np.ndarray, assignments: list[dict]) -> np.ndarray:
    out = inp.copy()
    for item in assignments:
        sprite_top, sprite_left = item["sprite_origin"]
        sprite_mask = item["source_mask"]
        sh, sw = sprite_mask.shape
        patch = out[sprite_top : sprite_top + sh, sprite_left : sprite_left + sw]
        patch[sprite_mask] = BG

        hole_top, hole_left, _, _ = item["hole_bbox"]
        hole_mask = item["hole_mask"]
        target = out[hole_top : hole_top + hole_mask.shape[0], hole_left : hole_left + hole_mask.shape[1]]
        target[hole_mask] = item["color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    legend_len = _RNG.randint(4, 7)
    legend_colors = _RNG.sample(list(range(1, 10)), legend_len)
    legend_gaps = [_RNG.randint(0, 1) for _ in range(legend_len - 1)]
    legend_span = legend_len + sum(legend_gaps)

    residue_positions: dict[int, int] = {}
    for residue in range(4):
        choices = [idx for idx in range(legend_len) if idx % 4 == residue]
        residue_positions[residue] = _RNG.choice(choices)
    frame_colors = [legend_colors[residue_positions[idx]] for idx in range(4)]

    filler_choices = [color for color in range(1, 10) if color not in legend_colors]
    filler_color = _RNG.choice(filler_choices)
    sprite_pool = [color for color in range(1, 10) if color not in set(frame_colors) | {filler_color}]

    used_shapes: set[tuple[tuple[int, int], bytes]] = set()
    frames: list[dict] = []
    for residue in range(4):
        hole_count = 1 + (_RNG.random() < 0.5)
        hole_masks: list[np.ndarray] = []
        for _ in range(hole_count):
            chosen = None
            for _ in range(200):
                mask = _random_connected_mask()
                if mask is None:
                    continue
                canon = _canonical_rotation_signature(mask)
                if canon in used_shapes:
                    continue
                used_shapes.add(canon)
                chosen = mask
                break
            if chosen is None:
                return None
            hole_masks.append(chosen)

        if len(hole_masks) == 2 and _RNG.random() < 0.5:
            hole_masks = [hole_masks[1], hole_masks[0]]

        inner_h, inner_w, hole_layout = _layout_holes(hole_masks)
        frames.append(
            {
                "turns": residue_positions[residue] % 4,
                "border_color": frame_colors[residue],
                "inner_h": inner_h,
                "inner_w": inner_w,
                "hole_layout": hole_layout,
            }
        )

    _RNG.shuffle(frames)
    top_left, top_right, bottom_left, bottom_right = frames
    col_widths = [
        max(top_left["inner_w"] + 2, bottom_left["inner_w"] + 2),
        max(top_right["inner_w"] + 2, bottom_right["inner_w"] + 2),
    ]
    row_heights = [
        max(top_left["inner_h"] + 2, top_right["inner_h"] + 2),
        max(bottom_left["inner_h"] + 2, bottom_right["inner_h"] + 2),
    ]

    top_margin = 3 + _RNG.randint(0, 1)
    left_margin = 1 + _RNG.randint(0, 2)
    hgap = _RNG.randint(4, 6)
    vgap = _RNG.randint(3, 5)

    placements = {}
    bases = [
        (0, 0, top_left),
        (0, 1, top_right),
        (1, 0, bottom_left),
        (1, 1, bottom_right),
    ]
    for row_idx, col_idx, frame in bases:
        slot_top = top_margin + row_idx * (row_heights[0] + vgap)
        slot_left = left_margin + col_idx * (col_widths[0] + hgap)
        frame_h = frame["inner_h"] + 2
        frame_w = frame["inner_w"] + 2
        max_r_off = row_heights[row_idx] - frame_h
        max_c_off = col_widths[col_idx] - frame_w
        placements[id(frame)] = (
            slot_top + _RNG.randint(0, max_r_off),
            slot_left + _RNG.randint(0, max_c_off),
        )

    frame_bottom = max(placements[id(frame)][0] + frame["inner_h"] + 1 for frame in frames)
    frame_right = max(placements[id(frame)][1] + frame["inner_w"] + 1 for frame in frames)

    grid_h = max(frame_bottom + 9 + _RNG.randint(0, 4), 24 + _RNG.randint(0, 6))
    grid_w = max(frame_right + 5 + _RNG.randint(0, 5), legend_span + 4, 28 + _RNG.randint(0, 8))

    grid = np.zeros((grid_h, grid_w), dtype=int)
    _place_legend(grid, legend_colors, legend_gaps)

    assignments: list[dict] = []
    frame_boxes: list[tuple[int, int, int, int]] = []
    for frame in frames:
        top, left = placements[id(frame)]
        holes = _draw_frame(
            grid,
            top,
            left,
            frame["border_color"],
            filler_color,
            frame["inner_h"],
            frame["inner_w"],
            frame["hole_layout"],
        )
        frame_boxes.append((top, left, top + frame["inner_h"] + 1, left + frame["inner_w"] + 1))
        for hole in holes:
            source_mask = np.rot90(hole["mask"], frame["turns"])
            assignments.append(
                {
                    "hole_bbox": hole["bbox"],
                    "hole_mask": hole["mask"],
                    "source_mask": source_mask,
                    "turns": frame["turns"],
                    "color": _RNG.choice(sprite_pool),
                }
            )

    occupied = np.zeros_like(grid, dtype=bool)
    occupied[:2, :] = True
    for r0, c0, r1, c1 in frame_boxes:
        rr0, rr1 = _expanded_slice(r0, r1 + 1, grid_h)
        cc0, cc1 = _expanded_slice(c0, c1 + 1, grid_w)
        occupied[rr0:rr1, cc0:cc1] = True

    _RNG.shuffle(assignments)
    for item in assignments:
        origin = _place_sprite(grid, occupied, item["source_mask"], item["color"])
        if origin is None:
            return None
        item["sprite_origin"] = origin

    expected = _expected_output(grid, assignments)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
