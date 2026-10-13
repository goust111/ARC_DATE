from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
GREEN = 3
PATH_COLORS = [2, 4, 5, 6, 7, 8, 9]
GAP = 2
SEP_W = 2
LEGEND_W = 4
MAX_GEN_ATTEMPTS = 400


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


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
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return [(r0, c) for c in range(lo, hi + 1)]
    lo, hi = sorted((r0, r1))
    return [(r, c0) for r in range(lo, hi + 1)]


def _degree(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    deg = 0
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            deg += 1
    return deg


def _endpoints(mask: np.ndarray) -> set[tuple[int, int]]:
    pts = np.argwhere(mask)
    return {
        (int(r), int(c))
        for r, c in pts
        if _degree(mask, int(r), int(c)) == 1
    }


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            comp_id += 1
            q = deque([(r, c)])
            labels[r, c] = comp_id
            while q:
                rr, cc = q.popleft()
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = comp_id
                        q.append((nr, nc))
    return labels, comp_id


def _is_filled_rect(grid: np.ndarray, comp: list[tuple[int, int]], color: int) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return patch.shape[0] >= 2 and patch.shape[1] >= 2 and bool(np.all(patch == color))


def _is_hollow_rect(grid: np.ndarray, comp: list[tuple[int, int]], color: int) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros_like(patch, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(patch[border] == color) and np.all(patch[~border] == BG))


def _find_legend_cut(grid: np.ndarray) -> int:
    h, w = grid.shape
    for cut in range(w):
        suffix = grid[:, cut:]
        values = set(int(v) for v in np.unique(suffix))
        if BLUE not in values or GREEN not in values:
            continue
        if not values.issubset({BG, BLUE, GREEN}):
            continue
        green_ok = any(_is_filled_rect(suffix, comp, GREEN) for comp in _components(suffix == GREEN))
        blue_ok = any(_is_hollow_rect(suffix, comp, BLUE) for comp in _components(suffix == BLUE))
        if green_ok and blue_ok:
            return cut

    nonlegend = np.argwhere(~np.isin(grid, [BG, BLUE, GREEN]))
    assert nonlegend.size > 0, "failed to locate the right-side legend strip"
    return int(nonlegend[:, 1].max()) + 1


def _candidate_connections(
    base: np.ndarray,
    mask: np.ndarray,
    color: int,
    endpoints: set[tuple[int, int]],
) -> dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]]:
    h, w = base.shape
    candidates: dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]] = {}

    for r, c in endpoints:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            gap = 0
            while 0 <= nr < h and 0 <= nc < w:
                if mask[nr, nc]:
                    if gap >= 1 and (nr, nc) in endpoints:
                        a = (r, c)
                        b = (int(nr), int(nc))
                        key = tuple(sorted((a, b)))
                        candidates[key] = _line_cells(a, b)
                    break
                if int(base[nr, nc]) != BG:
                    break
                gap += 1
                nr += dr
                nc += dc

    return candidates


def _reconstruct_color_path(base: np.ndarray, color: int) -> np.ndarray:
    mask = base == color
    for _ in range(mask.size):
        endpoints = _endpoints(mask)
        if len(endpoints) < 2:
            break

        candidates = _candidate_connections(base, mask, color, endpoints)
        if not candidates:
            break

        options: dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int]]]] = {
            pt: [] for pt in endpoints
        }
        for key in candidates:
            a, b = key
            options[a].append(key)
            options[b].append(key)

        chosen = [
            key
            for key in candidates
            if len(options[key[0]]) == 1 and len(options[key[1]]) == 1
        ]

        if not chosen:
            labels, ncomp = _label_components(mask)
            if ncomp <= 1:
                break
            cross = [
                key
                for key in candidates
                if labels[key[0]] != labels[key[1]]
            ]
            if len(cross) == 1:
                chosen = cross
            else:
                break

        for key in chosen:
            for r, c in candidates[key]:
                mask[r, c] = True

    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconnect same-colored line fragments and then either fill or keep the recovered path.

    Input structure:
    - Black background with several colors of short horizontal or vertical line fragments.
    - Fragments of the same color belong to one orthogonal path but are separated by black gaps.
    - A right-side legend strip contains a small solid Green square and a small hollow Blue square.

    Rule:
    - Remove the right-side legend strip.
    - For each path color, repeatedly connect aligned endpoint pairs of that color by drawing straight
      horizontal or vertical links through black gaps, using other colors as blockers.
    - If the reconstructed path is closed, fill its whole rectangular bounding box with that color.
    - If the reconstructed path is open, keep only the path pixels and leave the rest of its box black.

    Key insight:
    The same-color pieces are completed by endpoint-to-endpoint extension, and the legend's two icons
    encode the branch between a filled rectangle for loops and an outline-only result for open paths.
    """
    g = np.asarray(grid, dtype=int)
    cut = _find_legend_cut(g)
    content = g[:, :cut].copy()
    out = np.zeros_like(content)

    for color in sorted(int(v) for v in np.unique(content) if int(v) != BG):
        mask = _reconstruct_color_path(content, color)
        pts = np.argwhere(mask)
        if pts.size == 0:
            continue
        r0 = int(pts[:, 0].min())
        r1 = int(pts[:, 0].max())
        c0 = int(pts[:, 1].min())
        c1 = int(pts[:, 1].max())
        labels, ncomp = _label_components(mask)
        closed = ncomp == 1 and len(_endpoints(mask)) == 0
        if closed:
            out[r0 : r1 + 1, c0 : c1 + 1] = color
        else:
            out[mask] = color

    return out


def _add_segment(
    cells: set[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    for cell in _line_cells(start, end):
        cells.add(cell)


def _build_patch_from_cells(
    input_cells: set[tuple[int, int]],
    output_cells: set[tuple[int, int]],
    color: int,
    fill_bbox: bool,
) -> tuple[np.ndarray, np.ndarray]:
    all_cells = input_cells | output_cells
    rows = [r for r, _ in all_cells]
    cols = [c for _, c in all_cells]
    max_r = max(rows) + 2
    max_c = max(cols) + 2
    inp = np.zeros((max_r + 1, max_c + 1), dtype=int)
    out = np.zeros_like(inp)

    for r, c in input_cells:
        inp[r + 1, c + 1] = color

    if fill_bbox:
        shifted = [(r + 1, c + 1) for r, c in output_cells]
        rs = [r for r, _ in shifted]
        cs = [c for _, c in shifted]
        out[min(rs) : max(rs) + 1, min(cs) : max(cs) + 1] = color
    else:
        for r, c in output_cells:
            out[r + 1, c + 1] = color

    return inp, out


def _make_open_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    seg_count = random.randint(2, 5)
    start_horizontal = bool(random.getrandbits(1))

    start_owned = [False] * seg_count
    end_owned = [False] * seg_count
    start_owned[0] = True
    end_owned[-1] = True
    for i in range(seg_count - 1):
        if random.random() < 0.5:
            end_owned[i] = True
            start_owned[i + 1] = False
        else:
            end_owned[i] = False
            start_owned[i + 1] = True

    frag_lens = [random.randint(2, 4) for _ in range(seg_count)]
    dists = []
    for i in range(seg_count):
        gap_start = 0 if start_owned[i] else GAP
        gap_end = 0 if end_owned[i] else GAP
        dists.append(frag_lens[i] + gap_start + gap_end - 1)

    vertices = [(0, 0)]
    for i, dist in enumerate(dists):
        r, c = vertices[-1]
        horizontal = (i % 2 == 0) == start_horizontal
        if horizontal:
            vertices.append((r, c + dist))
        else:
            vertices.append((r + dist, c))

    full_cells: set[tuple[int, int]] = set()
    input_cells: set[tuple[int, int]] = set()

    for i in range(seg_count):
        a = vertices[i]
        b = vertices[i + 1]
        _add_segment(full_cells, a, b)

        gap_start = 0 if start_owned[i] else GAP
        gap_end = 0 if end_owned[i] else GAP
        if a[0] == b[0]:
            sign = 1 if b[1] >= a[1] else -1
            frag_a = (a[0], a[1] + sign * gap_start)
            frag_b = (a[0], b[1] - sign * gap_end)
        else:
            sign = 1 if b[0] >= a[0] else -1
            frag_a = (a[0] + sign * gap_start, a[1])
            frag_b = (b[0] - sign * gap_end, a[1])
        _add_segment(input_cells, frag_a, frag_b)

    return _build_patch_from_cells(input_cells, full_cells, color, fill_bbox=False)


def _make_closed_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    width_len = random.randint(3, 4)
    width_dist = width_len - 1
    height_dist = random.randint(6, 8)

    if random.random() < 0.5:
        left_top, left_bottom = 2, 3
        right_top, right_bottom = 3, 2
    else:
        left_top, left_bottom = 3, 2
        right_top, right_bottom = 2, 3

    vertices = [(0, 0), (0, width_dist), (height_dist, width_dist), (height_dist, 0)]
    segments = [
        ((0, 0), (0, width_dist)),
        ((right_top, width_dist), (height_dist - right_bottom, width_dist)),
        ((height_dist, width_dist), (height_dist, 0)),
        ((height_dist - left_bottom, 0), (left_top, 0)),
    ]

    full_cells: set[tuple[int, int]] = set()
    input_cells: set[tuple[int, int]] = set()

    for i in range(4):
        _add_segment(full_cells, vertices[i], vertices[(i + 1) % 4])
    for a, b in segments:
        _add_segment(input_cells, a, b)

    return _build_patch_from_cells(input_cells, full_cells, color, fill_bbox=True)


def _random_transform(
    inp: np.ndarray,
    out: np.ndarray,
    *,
    allow_quarter_turns: bool,
) -> tuple[np.ndarray, np.ndarray]:
    k = random.randint(0, 3) if allow_quarter_turns else random.choice([0, 2])
    inp = np.rot90(inp, k)
    out = np.rot90(out, k)
    if random.random() < 0.5:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < 0.5:
        inp = np.fliplr(inp)
        out = np.fliplr(out)
    return inp.copy(), out.copy()


def _place_patches(
    patches: list[tuple[np.ndarray, np.ndarray]],
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    content_in = np.zeros((height, width), dtype=int)
    content_out = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    order = list(range(len(patches)))
    random.shuffle(order)
    for idx in order:
        pin, pout = patches[idx]
        ph, pw = pin.shape
        placed = False
        for _ in range(300):
            r0 = random.randint(0, height - ph)
            c0 = random.randint(0, width - pw)
            rr0 = max(0, r0 - 1)
            rr1 = min(height, r0 + ph + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(width, c0 + pw + 1)
            if reserved[rr0:rr1, cc0:cc1].any():
                continue
            patch_mask = pin != BG
            out_mask = pout != BG
            content_in[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = pin[patch_mask]
            content_out[r0 : r0 + ph, c0 : c0 + pw][out_mask] = pout[out_mask]
            reserved[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None

    nz = np.argwhere(content_in != BG)
    if nz.size == 0:
        return None
    right = int(nz[:, 1].max()) + 1
    return content_in[:, :right], content_out[:, :right]


def _make_legend(height: int) -> np.ndarray:
    legend = np.zeros((height, LEGEND_W), dtype=int)
    top_r = max(1, height // 6)
    bot_r = min(height - 4, height - 1 - height // 3)

    legend[top_r : top_r + 2, 0:2] = GREEN
    legend[bot_r, 0:3] = BLUE
    legend[bot_r + 2, 0:3] = BLUE
    legend[bot_r : bot_r + 3, 0] = BLUE
    legend[bot_r : bot_r + 3, 2] = BLUE
    return legend


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        obj_count = random.choice([2, 3])
        kinds = ["open", "closed"]
        if obj_count == 3:
            kinds.append(random.choice(["open", "closed"]))
        random.shuffle(kinds)

        colors = random.sample(PATH_COLORS, obj_count)
        patches: list[tuple[np.ndarray, np.ndarray]] = []
        for kind, color in zip(kinds, colors):
            if kind == "open":
                pin, pout = _make_open_patch(color)
                patches.append(_random_transform(pin, pout, allow_quarter_turns=True))
            else:
                pin, pout = _make_closed_patch(color)
                patches.append(_random_transform(pin, pout, allow_quarter_turns=False))

        max_h = max(pin.shape[0] for pin, _ in patches)
        max_w = max(pin.shape[1] for pin, _ in patches)
        height_lo = max(12, max_h + 1)
        height_hi = min(24, max_h + 10)
        width_lo = max(14, max_w + 2)
        width_hi = 23
        if height_lo > height_hi or width_lo > width_hi:
            continue

        height = random.randint(height_lo, height_hi)
        width = random.randint(width_lo, width_hi)
        placed = _place_patches(patches, height, width)
        if placed is None:
            continue

        content_in, content_out = placed
        h, w = content_in.shape
        if w + SEP_W + LEGEND_W > 30:
            continue

        full = np.zeros((h, w + SEP_W + LEGEND_W), dtype=int)
        full[:, :w] = content_in
        full[:, w + SEP_W :] = _make_legend(h)

        solved = main(full)
        if solved.shape != content_out.shape:
            continue
        if not np.array_equal(solved, content_out):
            continue
        return full

    raise RuntimeError("failed to generate a valid ARC sample")
