from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 5
STAMP_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120
REGION_ATTEMPTS = 120


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _mask_from_coords(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = 1 + max(r for r, _ in coords)
    w = 1 + max(c for _, c in coords)
    mask = np.zeros((h, w), dtype=int)
    for r, c in coords:
        mask[r, c] = 1
    return mask


def _rotated_stamp(stamp: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return np.rot90(stamp, 2)
    if side == "right":
        return np.rot90(stamp, 3)
    if side == "left":
        return np.rot90(stamp, 1)
    return stamp.copy()


def _stamp_origin(
    marker_r: int,
    marker_c: int,
    side: str,
    stamp_shape: tuple[int, int],
) -> tuple[int, int]:
    h, w = stamp_shape
    if side == "top":
        return marker_r - h, marker_c - w // 2
    if side == "bottom":
        return marker_r + 1, marker_c - w // 2
    if side == "left":
        return marker_r - h // 2, marker_c - w
    return marker_r - h // 2, marker_c + 1


def _stamp_copy_coords(
    stamp: np.ndarray,
    marker: tuple[int, int],
    side: str,
) -> set[tuple[int, int]]:
    rotated = _rotated_stamp(stamp, side)
    top, left = _stamp_origin(marker[0], marker[1], side, rotated.shape)
    coords: set[tuple[int, int]] = set()
    for r, c in np.argwhere(rotated != 0):
        coords.add((top + int(r), left + int(c)))
    return coords


def _is_frame_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    if any(int(grid[r, c]) != FRAME for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))):
        return False
    perimeter = set()
    for c in range(c0, c1 + 1):
        perimeter.add((r0, c))
        perimeter.add((r1, c))
    for r in range(r0, r1 + 1):
        perimeter.add((r, c0))
        perimeter.add((r, c1))
    return perimeter == set(coords)


def _parse_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    for comp in _components(g != BG):
        if not _is_frame_component(g, comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        stamp_comps = _components(inner != BG)
        assert len(stamp_comps) == 1, "each frame must contain one interior stamp"
        stamp_rel = stamp_comps[0]
        stamp_abs = [(r0 + 1 + r, c0 + 1 + c) for r, c in stamp_rel]
        colors = {int(g[r, c]) for r, c in stamp_abs}
        assert len(colors) == 1, "stamp must be monochromatic"
        color = next(iter(colors))
        assert color not in (BG, FRAME), "stamp color must differ from bg/frame"

        sr0, sc0, sr1, sc1 = _bbox(stamp_abs)
        stamp = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=int)
        for r, c in stamp_abs:
            stamp[r - sr0, c - sc0] = color

        markers: list[dict] = []
        for c in range(c0 + 1, c1):
            v = int(g[r0, c])
            assert v in (FRAME, color)
            if v == color:
                markers.append({"pos": (r0, c), "side": "top"})
        for c in range(c0 + 1, c1):
            v = int(g[r1, c])
            assert v in (FRAME, color)
            if v == color:
                markers.append({"pos": (r1, c), "side": "bottom"})
        for r in range(r0 + 1, r1):
            v = int(g[r, c0])
            assert v in (FRAME, color)
            if v == color:
                markers.append({"pos": (r, c0), "side": "left"})
        for r in range(r0 + 1, r1):
            v = int(g[r, c1])
            assert v in (FRAME, color)
            if v == color:
                markers.append({"pos": (r, c1), "side": "right"})

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "color": color,
                "stamp": stamp,
                "markers": markers,
            }
        )
    assert frames, "expected at least one frame"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find each grey rectangular frame, extract its interior stamp, then stamp rotated copies
    just outside the frame at every border marker position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    paint_steps: list[tuple[int, int, np.ndarray, str]] = []
    for frame in _parse_frames(g):
        for marker in frame["markers"]:
            mr, mc = marker["pos"]
            paint_steps.append((mc, mr, frame["stamp"], marker["side"]))

    # The prompt specifies left-to-right, then top-to-bottom overwrite order.
    for marker_c, marker_r, stamp, side in sorted(paint_steps):
        rotated = _rotated_stamp(stamp, side)
        top, left = _stamp_origin(marker_r, marker_c, side, rotated.shape)

        for r, c in np.argwhere(rotated != 0):
            rr = top + int(r)
            cc = left + int(c)
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "stamp copy leaves the grid"
            out[rr, cc] = int(rotated[r, c])
    return out


def _random_stamp(rng: random.Random) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        target = rng.randint(3, 6)
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = {
                (r + dr, c + dc)
                for r, c in cells
                for dr, dc in _neighbors4(0, 0)
                if (r + dr, c + dc) not in cells
            }
            cells.add(rng.choice(sorted(frontier)))
        norm = _normalize(cells)
        mask = _mask_from_coords(norm)
        h, w = mask.shape
        if not (2 <= h <= 4 and 2 <= w <= 4):
            continue
        if int(mask.sum()) == h * w:
            continue
        variants = {_normalize(set(map(tuple, np.argwhere(np.rot90(mask, k) != 0)))) for k in range(4)}
        if len(variants) < 4:
            continue
        return mask
    raise RuntimeError("failed to generate an asymmetric stamp")


def _marker_count_split(rng: random.Random, n_objects: int, total_markers: int) -> list[int]:
    counts = [1] * n_objects
    for _ in range(total_markers - n_objects):
        choices = [idx for idx, count in enumerate(counts) if count < 4]
        counts[rng.choice(choices)] += 1
    rng.shuffle(counts)
    return counts


def _assign_marker_sides(rng: random.Random, n_objects: int) -> list[list[str]]:
    max_total = min(8, 3 * n_objects)
    total_markers = rng.randint(max(4, n_objects), max_total)
    counts = _marker_count_split(rng, n_objects, total_markers)
    sides = list(SIDES) + [rng.choice(SIDES) for _ in range(total_markers - 4)]
    for _ in range(200):
        rng.shuffle(sides)
        groups: list[list[str]] = []
        cursor = 0
        ok = True
        for count in counts:
            group = sides[cursor : cursor + count]
            cursor += count
            if max(Counter(group).values()) > 2:
                ok = False
                break
            groups.append(group)
        if ok:
            return groups
    raise RuntimeError("failed to distribute marker sides")


def _build_region(rng: random.Random, color: int, sides: list[str]) -> np.ndarray | None:
    counts = Counter(sides)
    for _ in range(REGION_ATTEMPTS):
        stamp = _random_stamp(rng)
        sh, sw = stamp.shape

        inner_h_min = max(
            sh + 2,
            counts["left"] * sw + max(0, counts["left"] - 1),
            counts["right"] * sw + max(0, counts["right"] - 1),
        )
        inner_w_min = max(
            sw + 2,
            counts["top"] * sw + max(0, counts["top"] - 1),
            counts["bottom"] * sw + max(0, counts["bottom"] - 1),
        )

        inner_h = inner_h_min + rng.randint(0, 3)
        inner_w = inner_w_min + rng.randint(0, 3)
        frame_h = inner_h + 2
        frame_w = inner_w + 2
        slack_h = inner_h - sh
        slack_w = inner_w - sw
        gap_top = rng.randint(1, slack_h - 1)
        gap_left = rng.randint(1, slack_w - 1)

        stamp_cells = {
            (1 + gap_top + r, 1 + gap_left + c)
            for r, c in np.argwhere(stamp != 0)
        }
        used_markers: set[tuple[int, int]] = set()
        copy_union: set[tuple[int, int]] = set()
        marker_specs: list[tuple[int, int, str]] = []

        side_order = sides[:]
        rng.shuffle(side_order)
        for side in side_order:
            if side == "top":
                candidates = [(0, c) for c in range(1, frame_w - 1)]
            elif side == "bottom":
                candidates = [(frame_h - 1, c) for c in range(1, frame_w - 1)]
            elif side == "left":
                candidates = [(r, 0) for r in range(1, frame_h - 1)]
            else:
                candidates = [(r, frame_w - 1) for r in range(1, frame_h - 1)]
            rng.shuffle(candidates)

            placed = False
            for marker in candidates:
                if marker in used_markers:
                    continue
                copy_cells = _stamp_copy_coords(stamp, marker, side)
                if copy_union & copy_cells:
                    continue
                used_markers.add(marker)
                copy_union |= copy_cells
                marker_specs.append((marker[0], marker[1], side))
                placed = True
                break
            if not placed:
                break
        else:
            frame_cells = set()
            for c in range(frame_w):
                frame_cells.add((0, c))
                frame_cells.add((frame_h - 1, c))
            for r in range(frame_h):
                frame_cells.add((r, 0))
                frame_cells.add((r, frame_w - 1))

            all_cells = frame_cells | stamp_cells | copy_union
            r0, c0, r1, c1 = _bbox(all_cells)
            shift_r = -r0
            shift_c = -c0
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)

            for r, c in frame_cells:
                patch[r + shift_r, c + shift_c] = FRAME
            for r, c in stamp_cells:
                patch[r + shift_r, c + shift_c] = color
            for r, c, _ in marker_specs:
                patch[r + shift_r, c + shift_c] = color
            return patch
    return None


def _row_partition(rng: random.Random, n_items: int, n_rows: int) -> list[int]:
    cuts = sorted(rng.sample(range(1, n_items), n_rows - 1))
    sizes = []
    prev = 0
    for cut in cuts + [n_items]:
        sizes.append(cut - prev)
        prev = cut
    return sizes


def _arrange_patches(rng: random.Random, patches: list[np.ndarray]) -> np.ndarray:
    order = patches[:]
    rng.shuffle(order)
    row_count = rng.randint(1, min(3, len(order)))
    row_sizes = _row_partition(rng, len(order), row_count)

    rows: list[list[np.ndarray]] = []
    cursor = 0
    for size in row_sizes:
        rows.append(order[cursor : cursor + size])
        cursor += size

    row_layouts = []
    total_width = 0
    total_height = 0
    for row in rows:
        gap = [rng.randint(1, 3) for _ in range(max(0, len(row) - 1))]
        row_height = max(int(p.shape[0]) for p in row)
        row_width = sum(int(p.shape[1]) for p in row) + sum(gap)
        row_layouts.append((row, gap, row_height, row_width))
        total_width = max(total_width, row_width)
        total_height += row_height
    row_gaps = [rng.randint(1, 3) for _ in range(max(0, len(rows) - 1))]
    total_height += sum(row_gaps)

    top_pad = rng.randint(0, 3)
    bottom_pad = rng.randint(0, 3)
    left_pad = rng.randint(0, 3)
    right_pad = rng.randint(0, 3)
    grid = np.full((top_pad + total_height + bottom_pad, left_pad + total_width + right_pad), BG, dtype=int)

    row_y = top_pad
    for row_index, (row, gaps, row_height, row_width) in enumerate(row_layouts):
        row_x = left_pad + rng.randint(0, total_width - row_width)
        cur_x = row_x
        for patch_index, patch in enumerate(row):
            ph, pw = map(int, patch.shape)
            y = row_y + rng.randint(0, row_height - ph)
            sub = grid[y : y + ph, cur_x : cur_x + pw]
            assert sub.shape == patch.shape
            assert not np.any((sub != BG) & (patch != BG)), "patches should not overlap"
            sub[patch != BG] = patch[patch != BG]
            cur_x += pw
            if patch_index < len(gaps):
                cur_x += gaps[patch_index]
        row_y += row_height
        if row_index < len(row_gaps):
            row_y += row_gaps[row_index]
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.randint(2, 4)
    colors = rng.sample(list(STAMP_COLORS), n_objects)
    side_groups = _assign_marker_sides(rng, n_objects)

    patches = []
    for color, sides in zip(colors, side_groups):
        patch = _build_region(rng, color, sides)
        if patch is None:
            return None
        patches.append(patch)

    grid = _arrange_patches(rng, patches)
    assert len(_parse_frames(grid)) == n_objects
    out = main(grid)
    assert out.shape == grid.shape
    assert len(set(side for group in side_groups for side in group)) == 4
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
