from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask)


def _canonical_shape(mask: np.ndarray) -> tuple[str, ...]:
    base = _tight_mask(mask.astype(bool))
    variants = []
    for k in range(4):
        rot = np.rot90(base, k)
        variants.append(_mask_key(_tight_mask(rot)))
        variants.append(_mask_key(_tight_mask(np.fliplr(rot))))
    return min(variants)


def _segments(flags: np.ndarray) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(flags.tolist()):
        if not flag and start is None:
            start = i
        elif flag and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(flags)))
    return segs


def _split_panels(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], list[int], list[int]]:
    grey_rows = np.all(grid == GREY, axis=1)
    grey_cols = np.all(grid == GREY, axis=0)
    row_segs = _segments(grey_rows)
    col_segs = _segments(grey_cols)
    assert row_segs and col_segs
    heights = {r1 - r0 for r0, r1 in row_segs}
    widths = {c1 - c0 for c0, c1 in col_segs}
    assert len(heights) == 1 and len(widths) == 1
    assert len(row_segs) * len(col_segs) >= 2

    panels = []
    for r0, r1 in row_segs:
        for c0, c1 in col_segs:
            panels.append((r0, r1, c0, c1))
    return panels, np.flatnonzero(grey_rows).tolist(), np.flatnonzero(grey_cols).tolist()


def _panel_info(panel: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    grey_comps = _components(panel == GREY)
    assert len(grey_comps) == 1
    shape_cells = grey_comps[0]
    shape_mask = np.zeros(panel.shape, dtype=bool)
    shape_mask[shape_cells[:, 0], shape_cells[:, 1]] = True

    marker_coords = np.argwhere((panel != BG) & (panel != GREY))
    assert marker_coords.shape == (1, 2)
    mr, mc = map(int, marker_coords[0])
    return shape_mask, np.array([mr, mc], dtype=int), int(panel[mr, mc])


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a panel grid separated by grey divider rows and columns, recolor each set
    of congruent grey shapes by the majority marker color in that set.
    """
    g = np.asarray(grid, dtype=int)
    panels, _grey_rows, _grey_cols = _split_panels(g)

    data = []
    for r0, r1, c0, c1 in panels:
        panel = g[r0:r1, c0:c1]
        shape_mask, marker_pos, marker_color = _panel_info(panel)
        signature = _canonical_shape(shape_mask)
        data.append(
            {
                "bounds": (r0, r1, c0, c1),
                "shape_mask": shape_mask,
                "marker_pos": marker_pos,
                "marker_color": marker_color,
                "signature": signature,
            }
        )

    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for item in data:
        grouped[item["signature"]].append(item)

    out = g.copy()
    for items in grouped.values():
        target_color = GREY
        if len(items) > 1:
            counts = Counter(int(item["marker_color"]) for item in items)
            mode_color, mode_count = counts.most_common(1)[0]
            assert list(counts.values()).count(mode_count) == 1
            target_color = int(mode_color)
        for item in items:
            r0, r1, c0, c1 = item["bounds"]
            panel_out = out[r0:r1, c0:c1]
            panel_out[tuple(item["marker_pos"])] = BG
            panel_out[item["shape_mask"]] = target_color
    return out


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))

    add_frontier(*start)
    while int(mask.sum()) < area and frontier:
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        mask[r, c] = True
        add_frontier(r, c)
    if int(mask.sum()) != area:
        return np.zeros((0, 0), dtype=bool)
    return _tight_mask(mask)


def _random_shape_signature(
    used: set[tuple[str, ...]], max_h: int, max_w: int
) -> tuple[np.ndarray, tuple[str, ...]] | None:
    limit = min(max_h, max_w)
    options = [
        (h, w, area)
        for h in range(2, limit + 1)
        for w in range(2, limit + 1)
        for area in range(3, min(h * w, max(4, (h * w * 2) // 3)) + 1)
        if area < h * w
    ]
    random.shuffle(options)
    for h, w, area in options[:120]:
        for _ in range(20):
            mask = _random_connected_mask(h, w, area)
            if mask.size == 0:
                continue
            sig = _canonical_shape(mask)
            if sig in used:
                continue
            return mask, sig
    return None


def _oriented_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    k = random.randrange(4)
    return np.rot90(out, k).copy()


def _place_panel_object(panel_h: int, panel_w: int, shape: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    shape = _oriented_mask(shape)
    sh, sw = shape.shape
    assert sh <= panel_h - 1 and sw <= panel_w - 1

    positions = []
    for r0 in range(panel_h - sh + 1):
        for c0 in range(panel_w - sw + 1):
            marker_choices = []
            occupied = np.zeros((panel_h, panel_w), dtype=bool)
            occupied[r0 : r0 + sh, c0 : c0 + sw] = shape
            for mr in range(panel_h):
                for mc in range(panel_w):
                    if occupied[mr, mc]:
                        continue
                    marker_choices.append((mr, mc))
            if marker_choices:
                positions.append((r0, c0, marker_choices))
    r0, c0, marker_choices = random.choice(positions)
    marker = random.choice(marker_choices)
    panel = np.full((panel_h, panel_w), BG, dtype=int)
    panel[r0 : r0 + sh, c0 : c0 + sw][shape] = GREY
    return panel, marker


def _sample_group_sizes(total: int) -> list[int]:
    assert total >= 4
    if total == 4:
        return random.choice([[1, 1, 2], [1, 3], [2, 2]])

    for _ in range(100):
        sizes = [1]
        remaining = total - 1
        while remaining > 0:
            lo = 1 if len(sizes) >= 2 else 2
            size = random.randint(lo, min(4, remaining))
            sizes.append(size)
            remaining -= size
        if any(size > 1 for size in sizes):
            random.shuffle(sizes)
            return sizes
    raise AssertionError("unable to sample group sizes")


def _marker_colors_for_group(size: int) -> list[int]:
    colors = random.sample(PALETTE, k=min(len(PALETTE), 4))
    if size == 1:
        return [random.choice(colors)]

    mode = colors[0]
    others = [c for c in colors[1:] if c != mode]
    mode_count = random.randint(size // 2 + 1, size)
    remaining = size - mode_count
    out = [mode] * mode_count
    for _ in range(remaining):
        out.append(random.choice(others))
    random.shuffle(out)
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        panel_rows = random.choice([2, 2, 2, 3])
        panel_cols = random.choice([2, 3])
        panel_h = random.randint(5, 8)
        panel_w = random.randint(5, 8)
        sep = 1
        total_panels = panel_rows * panel_cols
        if total_panels < 4:
            continue

        group_sizes = _sample_group_sizes(total_panels)
        used_signatures: set[tuple[str, ...]] = set()
        base_shapes: list[np.ndarray] = []
        for _size in group_sizes:
            made = _random_shape_signature(used_signatures, panel_h - 1, panel_w - 1)
            if made is None:
                break
            mask, sig = made
            used_signatures.add(sig)
            base_shapes.append(mask)
        else:
            assignments: list[tuple[np.ndarray, int]] = []
            for shape, size in zip(base_shapes, group_sizes):
                colors = _marker_colors_for_group(size)
                assignments.extend((shape, color) for color in colors)
            random.shuffle(assignments)

            grid_h = panel_rows * panel_h + (panel_rows - 1) * sep
            grid_w = panel_cols * panel_w + (panel_cols - 1) * sep
            grid = np.full((grid_h, grid_w), BG, dtype=int)
            for sr in range(1, panel_rows):
                grid[sr * panel_h + (sr - 1) * sep, :] = GREY
            for sc in range(1, panel_cols):
                grid[:, sc * panel_w + (sc - 1) * sep] = GREY

            idx = 0
            for pr in range(panel_rows):
                for pc in range(panel_cols):
                    r0 = pr * (panel_h + sep)
                    c0 = pc * (panel_w + sep)
                    base_shape, marker_color = assignments[idx]
                    idx += 1
                    panel, marker = _place_panel_object(panel_h, panel_w, base_shape)
                    panel[marker] = marker_color
                    grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel

            out = main(grid)
            if np.array_equal(out, grid):
                continue
            return grid
    raise RuntimeError("failed to generate a valid input")
