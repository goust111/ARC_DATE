from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 500


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                }
            )
    return comps


def _is_frame_component(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 6 or w < 6:
        return False
    if int(comp["coords"].shape[0]) != 2 * (h + w) - 4:
        return False

    mask = np.zeros((h, w), dtype=bool)
    rel = comp["coords"] - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def _parse_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _monochrome_components(grid, bg=BG):
        if not _is_frame_component(comp):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        marker_coords = np.argwhere(inner != BG)
        if marker_coords.size == 0:
            continue
        frames.append(
            {
                "color": int(comp["color"]),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "markers": [(int(r0 + 1 + rr), int(c0 + 1 + cc)) for rr, cc in marker_coords],
            }
        )

    frames.sort(key=lambda item: item["bbox"])
    assert 2 <= len(frames) <= 4
    return frames


def _parse_sprites(grid: np.ndarray, frames: list[dict]) -> dict[int, dict]:
    covered = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        covered[r0 : r1 + 1, c0 : c1 + 1] = True

    outside_mask = (grid != BG) & ~covered
    comps = _components_from_mask(outside_mask)
    sprites: dict[int, dict] = {}
    for coords in comps:
        vals = [int(grid[r, c]) for r, c in coords]
        counts = Counter(vals)
        assert len(counts) == 2
        anchor_colors = [color for color, count in counts.items() if int(count) == 1]
        assert len(anchor_colors) == 1
        anchor_color = int(anchor_colors[0])
        body_color = next(int(color) for color in counts if int(color) != anchor_color)

        r0, c0, r1, c1 = _bbox_from_coords(coords)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        assert 3 <= h <= 5 and 3 <= w <= 5

        mask = np.zeros((h, w), dtype=bool)
        anchor = None
        for r, c in coords:
            rr = int(r - r0)
            cc = int(c - c0)
            mask[rr, cc] = True
            if int(grid[r, c]) == anchor_color:
                anchor = (rr, cc)
        assert anchor is not None
        assert body_color not in sprites
        sprites[body_color] = {"mask": mask, "anchor": anchor}

    assert sprites
    return sprites


def _stamp_into_frame(
    out: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    marker_pos: tuple[int, int],
    sprite_mask: np.ndarray,
    anchor: tuple[int, int],
    color: int,
) -> None:
    r0, c0, r1, c1 = frame_bbox
    mr, mc = marker_pos
    ar, ac = anchor
    for sr, sc in np.argwhere(sprite_mask):
        tr = int(mr + sr - ar)
        tc = int(mc + sc - ac)
        if r0 + 1 <= tr <= r1 - 1 and c0 + 1 <= tc <= c1 - 1:
            out[tr, tc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each interior marker to the same-colored warehouse sprite and stamp the
    sprite into the frame interior using the sprite's unique odd-color pixel as
    the anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _parse_frames(g)
    sprites = _parse_sprites(g, frames)

    out = g.copy()
    for frame in frames:
        bbox = frame["bbox"]
        for marker_pos in sorted(frame["markers"]):
            color = int(g[marker_pos])
            assert color in sprites
            sprite = sprites[color]
            _stamp_into_frame(out, bbox, marker_pos, sprite["mask"], sprite["anchor"], color)
    return out


def _is_ring_mask(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_shape() -> tuple[np.ndarray, tuple[int, int]]:
    for _ in range(200):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        target = random.randint(max(4, (h + w) // 2), h * w - 1)

        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break

            missing: list[tuple[int, int]] = []
            if not any(r == 0 for r, _c in cells):
                missing.extend([cell for cell in frontier if cell[0] == 0])
            if not any(r == h - 1 for r, _c in cells):
                missing.extend([cell for cell in frontier if cell[0] == h - 1])
            if not any(c == 0 for _r, c in cells):
                missing.extend([cell for cell in frontier if cell[1] == 0])
            if not any(c == w - 1 for _r, c in cells):
                missing.extend([cell for cell in frontier if cell[1] == w - 1])
            cells.add(random.choice(missing or list(frontier)))

        raw = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            raw[int(r), int(c)] = True
        mask = _trim_mask(raw)

        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        if int(mask.sum()) < 4 or int(mask.sum()) == int(mask.size):
            continue
        if _is_ring_mask(mask):
            continue

        anchor_cells = np.argwhere(mask)
        ar, ac = map(int, random.choice(anchor_cells.tolist()))
        return mask, (ar, ac)
    raise RuntimeError("failed to generate a sprite shape")


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return int(mask.shape[0]), int(mask.shape[1]), mask.astype(np.uint8).tobytes()


def _mask_to_patch(mask: np.ndarray, body_color: int, anchor_color: int, anchor: tuple[int, int]) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(body_color)
    patch[int(anchor[0]), int(anchor[1])] = int(anchor_color)
    return patch


def _assign_marker_colors(frame_count: int, sprite_colors: list[int]) -> list[list[int]]:
    while True:
        capacities = [random.randint(1, min(3, len(sprite_colors))) for _ in range(frame_count)]
        if sum(capacities) >= len(sprite_colors):
            break

    frame_markers = [[] for _ in range(frame_count)]
    remaining = capacities[:]
    for color in sprite_colors:
        choices = [idx for idx in range(frame_count) if remaining[idx] > 0]
        idx = random.choice(choices)
        frame_markers[idx].append(int(color))
        remaining[idx] -= 1

    for idx in range(frame_count):
        while remaining[idx] > 0:
            choices = [color for color in sprite_colors if color not in frame_markers[idx]]
            frame_markers[idx].append(int(random.choice(choices or sprite_colors)))
            remaining[idx] -= 1
        random.shuffle(frame_markers[idx])

    return frame_markers


def _place_warehouse(
    grid: np.ndarray,
    sprite_patches: list[np.ndarray],
    side: str,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    band_h = 7
    widths = [int(patch.shape[1]) for patch in sprite_patches]
    gaps = [random.randint(1, 2) for _ in range(len(sprite_patches) - 1)]
    total_w = sum(widths) + sum(gaps)
    if total_w > w - 2:
        raise ValueError("warehouse row does not fit")

    row0 = 0 if side == "top" else h - band_h
    start_col = random.randint(1, w - 1 - total_w)
    order = list(range(len(sprite_patches)))
    random.shuffle(order)

    positions: list[tuple[int, int] | None] = [None] * len(sprite_patches)
    col = start_col
    for gap_idx, sprite_idx in enumerate(order):
        patch = sprite_patches[sprite_idx]
        ph, pw = map(int, patch.shape)
        min_row = row0 + 1
        max_row = row0 + band_h - ph - 1
        if min_row > max_row:
            raise ValueError("warehouse sprite does not fit vertically")
        top = random.randint(min_row, max_row)
        positions[sprite_idx] = (int(top), int(col))
        grid[top : top + ph, col : col + pw] = np.where(patch != BG, patch, grid[top : top + ph, col : col + pw])
        if gap_idx < len(gaps):
            col += pw + gaps[gap_idx]

    return [pos for pos in positions if pos is not None]


def _rect_is_clear(blocked: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    hh, ww = blocked.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(hh, top + h + pad)
    c1 = min(ww, left + w + pad)
    return not bool(blocked[r0:r1, c0:c1].any())


def _mark_rect(blocked: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> None:
    hh, ww = blocked.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(hh, top + h + pad)
    c1 = min(ww, left + w + pad)
    blocked[r0:r1, c0:c1] = True


def _place_frames(
    grid_shape: tuple[int, int],
    frame_sizes: list[tuple[int, int]],
    side: str,
) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    band_h = 7
    blocked = np.zeros((h, w), dtype=bool)
    if side == "top":
        blocked[: band_h + 1, :] = True
        min_row = band_h + 1
        max_row_for = lambda fh: h - fh - 1
    else:
        blocked[h - band_h - 1 :, :] = True
        min_row = 1
        max_row_for = lambda fh: h - band_h - fh - 1

    positions: list[tuple[int, int] | None] = [None] * len(frame_sizes)
    order = sorted(range(len(frame_sizes)), key=lambda idx: frame_sizes[idx][0] * frame_sizes[idx][1], reverse=True)
    for idx in order:
        fh, fw = frame_sizes[idx]
        min_col = 1
        max_col = w - fw - 1
        max_row = max_row_for(fh)
        if min_row > max_row or min_col > max_col:
            return None

        for _ in range(300):
            top = random.randint(min_row, max_row)
            left = random.randint(min_col, max_col)
            if not _rect_is_clear(blocked, top, left, fh, fw, pad=1):
                continue
            positions[idx] = (int(top), int(left))
            _mark_rect(blocked, top, left, fh, fw, pad=1)
            break
        if positions[idx] is None:
            return None

    return [pos for pos in positions if pos is not None]


def _stamp_cells(
    interior_shape: tuple[int, int],
    sprite_mask: np.ndarray,
    anchor: tuple[int, int],
    marker_rel: tuple[int, int],
) -> tuple[list[tuple[int, int]], bool]:
    ih, iw = interior_shape
    ar, ac = anchor
    mr, mc = marker_rel
    cells: list[tuple[int, int]] = []
    clipped = False
    for sr, sc in np.argwhere(sprite_mask):
        rr = int(mr + sr - ar)
        cc = int(mc + sc - ac)
        if 0 <= rr < ih and 0 <= cc < iw:
            cells.append((rr, cc))
        else:
            clipped = True
    return cells, clipped


def _place_markers_for_frame(
    frame_size: tuple[int, int],
    colors: list[int],
    sprite_defs: dict[int, dict],
    must_clip_index: int | None,
) -> list[dict] | None:
    fh, fw = frame_size
    ih, iw = fh - 2, fw - 2
    occupied = np.zeros((ih, iw), dtype=bool)
    used_markers: set[tuple[int, int]] = set()
    placements: list[dict] = []

    for idx, color in enumerate(colors):
        sprite = sprite_defs[color]
        full_area = int(sprite["mask"].sum())
        clipped_options: list[dict] = []
        fitted_options: list[dict] = []

        for mr in range(ih):
            for mc in range(iw):
                if (mr, mc) in used_markers:
                    continue
                cells, clipped = _stamp_cells((ih, iw), sprite["mask"], sprite["anchor"], (mr, mc))
                if not cells:
                    continue
                if any(occupied[rr, cc] for rr, cc in cells):
                    continue
                item = {"marker_rel": (mr, mc), "cells": cells}
                if clipped:
                    if 2 <= len(cells) < full_area:
                        clipped_options.append(item)
                else:
                    fitted_options.append(item)

        want_clip = must_clip_index is not None and idx == must_clip_index
        options = clipped_options if want_clip else fitted_options
        if not options:
            return None

        choice = random.choice(options)
        used_markers.add(choice["marker_rel"])
        for rr, cc in choice["cells"]:
            occupied[rr, cc] = True
        placements.append(
            {
                "color": int(color),
                "marker_rel": tuple(int(v) for v in choice["marker_rel"]),
                "cells": [(int(rr), int(cc)) for rr, cc in choice["cells"]],
            }
        )

    return placements


def _expected_output(
    grid: np.ndarray,
    frame_positions: list[tuple[int, int]],
    frame_sizes: list[tuple[int, int]],
    marker_placements: list[list[dict]],
    sprite_defs: dict[int, dict],
) -> np.ndarray:
    out = grid.copy()
    for frame_idx, placements in enumerate(marker_placements):
        top, left = frame_positions[frame_idx]
        fh, fw = frame_sizes[frame_idx]
        bbox = (int(top), int(left), int(top + fh - 1), int(left + fw - 1))
        for placement in placements:
            mr, mc = placement["marker_rel"]
            marker_pos = (int(top + 1 + mr), int(left + 1 + mc))
            sprite = sprite_defs[placement["color"]]
            _stamp_into_frame(out, bbox, marker_pos, sprite["mask"], sprite["anchor"], placement["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(2, 4)
    max_sprites = min(4, 8 - frame_count)
    sprite_count = random.randint(2, max_sprites)

    all_colors = list(range(1, 10))
    chosen = random.sample(all_colors, frame_count + sprite_count + 1)
    random.shuffle(chosen)
    frame_colors = chosen[:frame_count]
    sprite_colors = chosen[frame_count : frame_count + sprite_count]
    anchor_color = chosen[-1]

    sprite_defs: dict[int, dict] = {}
    used_shapes: set[tuple[int, int, bytes]] = set()
    for color in sprite_colors:
        for _ in range(100):
            mask, anchor = _random_shape()
            key = _shape_key(mask)
            if key in used_shapes:
                continue
            used_shapes.add(key)
            sprite_defs[int(color)] = {"mask": mask, "anchor": anchor}
            break
        if int(color) not in sprite_defs:
            return None

    frame_markers = _assign_marker_colors(frame_count, sprite_colors)
    frame_sizes: list[tuple[int, int]] = []
    for colors in frame_markers:
        min_side = 8 if len(colors) <= 2 else 9
        max_side = 11 if frame_count <= 3 else 10
        fh = random.randint(min_side, max_side)
        fw = random.randint(min_side, max_side)
        frame_sizes.append((int(fh), int(fw)))

    h_min = 24 + max(0, frame_count - 2)
    w_min = 24 + max(0, frame_count - 2)
    grid_h = random.randint(h_min, 30)
    grid_w = random.randint(w_min, 30)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    warehouse_side = random.choice(["top", "bottom"])
    sprite_patches = [
        _mask_to_patch(sprite_defs[color]["mask"], color, anchor_color, sprite_defs[color]["anchor"])
        for color in sprite_colors
    ]
    try:
        _place_warehouse(grid, sprite_patches, warehouse_side)
    except ValueError:
        return None

    frame_positions = _place_frames((grid_h, grid_w), frame_sizes, warehouse_side)
    if frame_positions is None:
        return None

    need_clip = bool(random.getrandbits(1))
    clip_frame = random.randrange(frame_count) if need_clip else None
    clip_marker = random.randrange(len(frame_markers[clip_frame])) if need_clip else None

    marker_placements: list[list[dict]] = []
    for frame_idx in range(frame_count):
        placements = _place_markers_for_frame(
            frame_sizes[frame_idx],
            frame_markers[frame_idx],
            sprite_defs,
            clip_marker if frame_idx == clip_frame else None,
        )
        if placements is None:
            return None
        marker_placements.append(placements)

    for frame_idx, (top, left) in enumerate(frame_positions):
        fh, fw = frame_sizes[frame_idx]
        color = frame_colors[frame_idx]
        grid[top, left : left + fw] = color
        grid[top + fh - 1, left : left + fw] = color
        grid[top : top + fh, left] = color
        grid[top : top + fh, left + fw - 1] = color
        for placement in marker_placements[frame_idx]:
            mr, mc = placement["marker_rel"]
            grid[top + 1 + mr, left + 1 + mc] = placement["color"]

    expected = _expected_output(grid, frame_positions, frame_sizes, marker_placements, sprite_defs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
