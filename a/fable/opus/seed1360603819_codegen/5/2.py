from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GRAY = 5
CONTENT_COLORS = (1, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


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
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    return _components_from_mask(grid == int(color))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_boundary(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components_from_mask(mask)
    return len(comps) == 1


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _hole_count_in_mask(mask: np.ndarray) -> int:
    h, w = mask.shape
    count = 0
    for comp in _components_from_mask(~mask):
        if not _touches_boundary(comp, h, w):
            count += 1
    return count


def _extract_legend_bars(grid: np.ndarray) -> list[dict]:
    bars = []
    for coords in _color_components(grid, GRAY):
        r0, c0, r1, c1 = _bbox(coords)
        if r0 != r1:
            continue
        if int(coords.shape[0]) != c1 - c0 + 1:
            continue
        bars.append(
            {
                "row": r0,
                "c0": c0,
                "c1": c1,
                "width": c1 - c0 + 1,
            }
        )

    assert len(bars) >= 2, "expected the right-side gray parity legend"
    bars.sort(key=lambda bar: (bar["c1"], bar["width"]), reverse=True)
    chosen = sorted(bars[:2], key=lambda bar: bar["row"])
    assert chosen[0]["row"] < chosen[1]["row"], "legend bars must be vertically stacked"
    assert chosen[0]["width"] < chosen[1]["width"], "top legend bar must be the shorter one"
    return chosen


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for coords in _color_components(grid, RED):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 5 and w >= 5, "frames must have non-trivial interiors"

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.all(patch[border] == RED), "red component must be a full rectangle border"
        assert not np.any(patch[~border] == RED), "frame interior may not contain red pixels"

        interior = patch[1:-1, 1:-1]
        inside_colors = set(map(int, np.unique(interior))) - {BLACK}
        assert len(inside_colors) == 1, "each frame must contain exactly one content color"
        color = inside_colors.pop()
        content_mask = interior == color
        assert _is_connected(content_mask), "content block must be connected"

        holes = []
        ih, iw = interior.shape
        for comp in _components_from_mask(interior == BLACK):
            if _touches_boundary(comp, ih, iw):
                continue
            abs_coords = comp.copy()
            abs_coords[:, 0] += r0 + 1
            abs_coords[:, 1] += c0 + 1
            holes.append(abs_coords)

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "color": color,
                "mask": content_mask,
                "holes": holes,
                "hole_count": len(holes),
            }
        )

    assert len(frames) >= 3, "expected at least three frames"
    counts = [frame["hole_count"] for frame in frames]
    assert len(set(counts)) == len(counts), "hole counts must uniquely order the frames"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Order red frames by their enclosed black-hole counts, rotate the interior colors along
    that order, then fill only the odd-hole frames' holes with the incoming color.

    The right-side gray legend is validated here as two horizontal bars stacked vertically,
    with the shorter bar on top and the longer one below. In this task family the shorter
    upper bar is the odd marker, so odd-hole frames are the ones whose holes get filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _extract_legend_bars(g)
    frames = _extract_frames(g)
    order = sorted(range(len(frames)), key=lambda i: frames[i]["hole_count"])

    out = g.copy()
    for k, frame_id in enumerate(order):
        donor_id = order[(k + 1) % len(order)]
        new_color = int(frames[donor_id]["color"])
        r0, c0, r1, c1 = frames[frame_id]["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[frames[frame_id]["mask"]] = new_color
        if frames[frame_id]["hole_count"] % 2 == 1:
            for hole in frames[frame_id]["holes"]:
                out[hole[:, 0], hole[:, 1]] = new_color

    return out


# --------------- generator (self-contained) ---------------

RING_SIZES = ((3, 3), (3, 4), (4, 3), (4, 4))
FRAME_COUNT_CHOICES = (3, 4)
GEN_ATTEMPTS = 300


def _draw_ring(mask: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    mask[r0, c0 : c0 + w] = True
    mask[r0 + h - 1, c0 : c0 + w] = True
    mask[r0 : r0 + h, c0] = True
    mask[r0 : r0 + h, c0 + w - 1] = True


def _build_blob(hole_count: int) -> np.ndarray:
    ring_sizes = [random.choice(RING_SIZES) for _ in range(hole_count)]
    branches = [random.randint(1, 3) for _ in range(hole_count)]
    gaps = [1 for _ in range(max(0, hole_count - 1))]

    height = sum(h for h, _ in ring_sizes) + sum(gaps)
    width = 2 + max(branch + w for branch, (_h, w) in zip(branches, ring_sizes, strict=False))
    mask = np.zeros((height, width), dtype=bool)
    mask[:, 0] = True

    ring_boxes = []
    row = 0
    for i, ((rh, rw), branch) in enumerate(zip(ring_sizes, branches, strict=False)):
        col = 1 + branch
        _draw_ring(mask, row, col, rh, rw)
        bridge_row = row + rh // 2
        mask[bridge_row, : col + 1] = True
        ring_boxes.append((row, col, rh, rw))
        row += rh
        if i + 1 < hole_count:
            row += gaps[i]

    for r0, c0, rh, rw in ring_boxes:
        if random.random() < 0.65:
            stub_len = random.randint(1, 2)
            rr = r0 if random.random() < 0.5 else r0 + rh - 1
            start = c0 + rw
            end = min(mask.shape[1], start + stub_len)
            mask[rr, start:end] = True

    if hole_count >= 4 and mask.shape[0] > mask.shape[1]:
        mask = mask.T
    elif hole_count == 3 and mask.shape[0] > mask.shape[1] and random.random() < 0.75:
        mask = mask.T
    elif random.random() < 0.45:
        mask = mask.T
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    mask = _crop_mask(mask)

    assert _is_connected(mask)
    assert _hole_count_in_mask(mask) == hole_count
    return mask


def _sample_hole_counts(frame_count: int) -> list[int]:
    counts = [1, 2, 3, 4]
    if frame_count == 3:
        counts.remove(random.choice(counts))
    random.shuffle(counts)
    return counts[:frame_count]


def _sample_colors_by_rank(frame_count: int) -> list[int]:
    if frame_count >= 3 and random.random() < 0.35:
        distinct = random.sample(list(CONTENT_COLORS), frame_count - 1)
        duplicate_color = distinct[0]
        pair_start = random.randint(0, frame_count - 2)
        colors = [None] * frame_count
        colors[pair_start] = duplicate_color
        colors[pair_start + 1] = duplicate_color
        others = distinct[1:]
        random.shuffle(others)
        fill_positions = [i for i, color in enumerate(colors) if color is None]
        for pos, color in zip(fill_positions, others, strict=False):
            colors[pos] = color
        return [int(color) for color in colors]
    return [int(color) for color in random.sample(list(CONTENT_COLORS), frame_count)]


def _make_frame_patch(hole_count: int, color: int) -> np.ndarray:
    blob = _build_blob(hole_count)
    bh, bw = blob.shape
    pad_top = random.randint(1, 2)
    pad_bottom = random.randint(1, 2)
    pad_left = random.randint(1, 2)
    pad_right = random.randint(1, 2)

    interior = np.full((bh + pad_top + pad_bottom, bw + pad_left + pad_right), BLACK, dtype=int)
    sub = interior[pad_top : pad_top + bh, pad_left : pad_left + bw]
    sub[blob] = int(color)

    patch = np.full((interior.shape[0] + 2, interior.shape[1] + 2), BLACK, dtype=int)
    patch[0, :] = RED
    patch[-1, :] = RED
    patch[:, 0] = RED
    patch[:, -1] = RED
    patch[1:-1, 1:-1] = interior
    return patch


def _row_groups(patches: list[np.ndarray]) -> list[list[int]]:
    order = sorted(range(len(patches)), key=lambda idx: int(patches[idx].shape[1]), reverse=True)
    if len(patches) == 4:
        rows = [order[::2], order[1::2]]
        for row in rows:
            random.shuffle(row)
        if random.random() < 0.5:
            rows.reverse()
        return rows

    if random.random() < 0.6:
        rows = [[order[0]], order[1:]]
    else:
        rows = [order[:2], order[2:]]
    for row in rows:
        random.shuffle(row)
    if random.random() < 0.5:
        rows.reverse()
    return rows


def _layout_patches(patches: list[np.ndarray]) -> tuple[list[tuple[int, int]], int, int]:
    groups = _row_groups(patches)
    placements = [(-1, -1)] * len(patches)
    y_cursor = 0
    cluster_w = 0

    for gi, group in enumerate(groups):
        row_h = max(int(patches[idx].shape[0]) for idx in group)
        x_cursor = 0
        order = group[:]
        random.shuffle(order)
        for pos, idx in enumerate(order):
            ph, pw = patches[idx].shape
            y = y_cursor + random.randint(0, row_h - ph)
            placements[idx] = (y, x_cursor)
            x_cursor += pw
            if pos + 1 < len(order):
                x_cursor += random.randint(1, 3)
        cluster_w = max(cluster_w, x_cursor)
        y_cursor += row_h
        if gi + 1 < len(groups):
            y_cursor += random.randint(2, 4)

    return placements, y_cursor, cluster_w


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    target = grid[r0 : r0 + h, c0 : c0 + w]
    assert target.shape == patch.shape
    assert np.all(target == BLACK)
    target[:] = patch


def _build_candidate() -> np.ndarray:
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    hole_counts = _sample_hole_counts(frame_count)
    sorted_counts = sorted(hole_counts)
    colors_by_rank = _sample_colors_by_rank(frame_count)
    color_for_count = {count: color for count, color in zip(sorted_counts, colors_by_rank, strict=False)}

    patches = [_make_frame_patch(count, color_for_count[count]) for count in hole_counts]
    placements, cluster_h, cluster_w = _layout_patches(patches)

    short_len = random.randint(2, 3)
    long_len = short_len + random.randint(1, 2)
    left_margin = random.randint(1, 3)
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    gap_to_legend = random.randint(3, 5)
    right_margin = random.randint(1, 3)
    legend_zone_w = long_len + random.randint(1, 2)

    height = max(top_margin + cluster_h + bottom_margin, 10)
    width = left_margin + cluster_w + gap_to_legend + legend_zone_w + right_margin
    grid = np.full((height, width), BLACK, dtype=int)

    for patch, (r, c) in zip(patches, placements, strict=False):
        _paste_patch(grid, patch, top_margin + r, left_margin + c)

    legend_c0 = left_margin + cluster_w + gap_to_legend
    max_top_row = height - 3
    top_row = random.randint(1, max(1, max_top_row - 2))
    bottom_row = min(height - 2, top_row + random.randint(2, 3))
    grid[top_row, legend_c0 : legend_c0 + short_len] = GRAY
    grid[bottom_row, legend_c0 : legend_c0 + long_len] = GRAY

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        try:
            out = main(grid)
        except Exception:
            continue
        assert out.shape == grid.shape
        return grid
    raise RuntimeError("failed to generate a valid ARC instance")
