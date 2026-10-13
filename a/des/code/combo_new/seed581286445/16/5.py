from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _background_color(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _flood_from_border(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _interior_mask(ring: np.ndarray) -> np.ndarray:
    background = ~ring
    exterior = _flood_from_border(background)
    interior = background & (~exterior)
    assert bool(interior.any()), "ring must enclose a non-empty interior"
    return interior


def _walk_cycle(mask: np.ndarray, start: tuple[int, int], first: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    total = int(mask.sum())
    path = [start, first]
    seen = {start, first}
    prev = start
    curr = first

    while True:
        nbrs = []
        for nr, nc in _neighbors4(*curr):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) != prev:
                nbrs.append((nr, nc))
        assert len(nbrs) == 1, "ring pixels must form a simple cycle"
        nxt = nbrs[0]
        if nxt == start:
            break
        assert nxt not in seen, "cycle should not self-intersect"
        path.append(nxt)
        seen.add(nxt)
        prev, curr = curr, nxt

    assert len(path) == total, "cycle walk must cover every ring pixel exactly once"
    return path


def _signed_area(path: list[tuple[int, int]]) -> int:
    area2 = 0
    for (r1, c1), (r2, c2) in zip(path, path[1:] + path[:1], strict=False):
        area2 += c1 * r2 - c2 * r1
    return int(area2)


def _extract_cycle(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert coords, "expected a non-empty ring"

    for r, c in coords:
        deg = 0
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                deg += 1
        assert deg == 2, "ring must be single-pixel wide with degree-2 vertices"

    start = min(coords)
    firsts = sorted((nr, nc) for nr, nc in _neighbors4(*start) if 0 <= nr < h and 0 <= nc < w and mask[nr, nc])
    assert len(firsts) == 2, "cycle start must have exactly two neighbors"
    path = _walk_cycle(mask, start, firsts[0])
    if _signed_area(path) < 0:
        path = [path[0], *path[:0:-1]]
    assert _signed_area(path) > 0, "cycle orientation should be clockwise after normalization"
    return path


def _segments_from_cycle(g: np.ndarray, path: list[tuple[int, int]]) -> list[dict[str, object]]:
    colors = [int(g[r, c]) for r, c in path]
    segments: list[dict[str, object]] = []
    cur_color = colors[0]
    cur_pixels = [path[0]]
    cur_indices = [0]

    for idx in range(1, len(path)):
        if colors[idx] == cur_color:
            cur_pixels.append(path[idx])
            cur_indices.append(idx)
        else:
            segments.append({"color": cur_color, "pixels": cur_pixels, "indices": cur_indices})
            cur_color = colors[idx]
            cur_pixels = [path[idx]]
            cur_indices = [idx]
    segments.append({"color": cur_color, "pixels": cur_pixels, "indices": cur_indices})

    if len(segments) > 1 and int(segments[0]["color"]) == int(segments[-1]["color"]):
        merged_pixels = list(segments[-1]["pixels"]) + list(segments[0]["pixels"])
        merged_indices = list(segments[-1]["indices"]) + list(segments[0]["indices"])
        segments = [{"color": int(segments[0]["color"]), "pixels": merged_pixels, "indices": merged_indices}] + segments[1:-1]

    assert all(int(seg["color"]) != BG for seg in segments), "ring segments must be non-background"
    return segments


def _right_normal(step: tuple[int, int]) -> tuple[int, int]:
    dr, dc = step
    assert abs(dr) + abs(dc) == 1, "cycle steps must be 4-connected"
    return dc, -dr


def _segment_normals(
    segment: dict[str, object], pos: int, path: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    pixels = segment["pixels"]
    indices = segment["indices"]
    curr_r, curr_c = pixels[pos]
    dirs: list[tuple[int, int]] = []

    if len(pixels) == 1:
        path_idx = int(indices[0])
        prev_r, prev_c = path[(path_idx - 1) % len(path)]
        next_r, next_c = path[(path_idx + 1) % len(path)]
        dirs.append((curr_r - prev_r, curr_c - prev_c))
        dirs.append((next_r - curr_r, next_c - curr_c))
    else:
        if pos > 0:
            prev_r, prev_c = pixels[pos - 1]
            dirs.append((curr_r - prev_r, curr_c - prev_c))
        if pos + 1 < len(pixels):
            next_r, next_c = pixels[pos + 1]
            dirs.append((next_r - curr_r, next_c - curr_c))

    normals = []
    seen = set()
    for step in dirs:
        normal = _right_normal(step)
        if normal not in seen:
            seen.add(normal)
            normals.append(normal)
    assert normals, "every segment pixel must induce at least one inward normal"
    return normals


def main(grid: np.ndarray) -> np.ndarray:
    """Thicken each colored segment of a thin loop inward by its own segment length.

    Input structure:
    - Black or dominant background fills the grid except for one single-pixel-wide 4-connected closed loop.
    - The loop encloses one background interior region, and the exterior around it is also background.
    - Consecutive same-colored pixels along the loop form color segments, and adjacent segments have different colors.

    Rule:
    - Trace the loop in clockwise order and split it into maximal same-color segments.
    - Starting from the unique red segment, process the segments clockwise; for a segment of length L, keep its color and use L as its inward thickness.
    - For each pixel of that segment, extend a ray of length L along that pixel's inward normal direction(s), including the original pixel, and stop a ray when it would leave the loop's interior-or-boundary support.
    - When multiple expanded regions target the same cell, preserve the color from the earlier segment in the red-first clockwise drawing order.

    Key insight:
    - The loop must be read as an ordered cycle: segment length is reused as geometric thickness, and the overlap resolution depends on the clockwise order starting at red.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    bg = _background_color(g)
    ring = g != int(bg)
    path = _extract_cycle(ring)
    support = ring | _interior_mask(ring)

    segments = _segments_from_cycle(g, path)
    red_idxs = [i for i, seg in enumerate(segments) if int(seg["color"]) == RED]
    assert len(red_idxs) == 1, "expected exactly one red segment"
    start = red_idxs[0]
    ordered_segments = segments[start:] + segments[:start]

    out = np.full_like(g, int(bg))
    h, w = g.shape
    for segment in ordered_segments:
        color = int(segment["color"])
        pixels = segment["pixels"]
        length = len(pixels)
        for pos, (r0, c0) in enumerate(pixels):
            for dr, dc in _segment_normals(segment, pos, path):
                r, c = int(r0), int(c0)
                for _ in range(length):
                    if not (0 <= r < h and 0 <= c < w) or not support[r, c]:
                        break
                    if out[r, c] == int(bg):
                        out[r, c] = color
                    r += dr
                    c += dc

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    if not bool(mask.any()):
        return False
    seen = np.zeros_like(mask, dtype=bool)
    start = tuple(map(int, np.argwhere(mask)[0]))
    q: deque[tuple[int, int]] = deque([start])
    seen[start] = True
    count = 1

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
                count += 1
    return count == int(mask.sum())


def _region_boundary(region: np.ndarray) -> np.ndarray:
    h, w = region.shape
    boundary = np.zeros_like(region, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not region[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or not region[nr, nc]:
                    boundary[r, c] = True
                    break
    return boundary


def _simple_ring_from_region(region: np.ndarray) -> np.ndarray | None:
    if not _mask_connected(region):
        return None
    boundary = _region_boundary(region)
    if int(boundary.sum()) < 8:
        return None
    if not bool((region & (~boundary)).any()):
        return None
    try:
        _extract_cycle(boundary)
    except AssertionError:
        return None
    return boundary


def _try_carve_notch(region: np.ndarray) -> np.ndarray | None:
    h, w = region.shape
    side = random.choice(("top", "bottom", "left", "right"))
    cand = region.copy()

    if side in ("top", "bottom"):
        if w < 6 or h < 6:
            return None
        width = random.randint(2, w - 2)
        depth = random.randint(1, max(1, min(h - 4, h // 2)))
        c0 = random.randint(1, w - width - 1)
        rows = slice(0, depth) if side == "top" else slice(h - depth, h)
        cand[rows, c0 : c0 + width] = False
    else:
        if h < 6 or w < 6:
            return None
        height = random.randint(2, h - 2)
        depth = random.randint(1, max(1, min(w - 4, w // 2)))
        r0 = random.randint(1, h - height - 1)
        cols = slice(0, depth) if side == "left" else slice(w - depth, w)
        cand[r0 : r0 + height, cols] = False

    return cand if _simple_ring_from_region(cand) is not None else None


def _random_region() -> np.ndarray | None:
    h = random.randint(8, 18)
    w = random.randint(8, 18)
    region = np.ones((h, w), dtype=bool)

    notch_count = random.randint(0, 3)
    for _ in range(notch_count):
        carved = _try_carve_notch(region)
        if carved is not None:
            region = carved

    for _ in range(random.randint(0, 3)):
        region = np.rot90(region)
    if random.choice((False, True)):
        region = np.fliplr(region)
    if random.choice((False, True)):
        region = np.flipud(region)

    boundary = _simple_ring_from_region(region)
    if boundary is None:
        return None
    return boundary


def _random_segment_lengths(total: int, parts: int, min_len: int) -> list[int]:
    lengths = [min_len] * parts
    for _ in range(total - parts * min_len):
        lengths[random.randrange(parts)] += 1
    random.shuffle(lengths)
    return lengths


def _paint_segments_on_ring(boundary: np.ndarray) -> np.ndarray | None:
    path = _extract_cycle(boundary)
    n = len(path)
    max_parts = min(8, n // 2)
    if max_parts < 4:
        return None
    parts = random.randint(4, max_parts)
    lengths = _random_segment_lengths(n, parts, min_len=2)
    if len(set(lengths)) < 2:
        return None

    colors = random.sample([c for c in PALETTE if c != RED], parts - 1)
    colors.insert(random.randrange(parts), RED)

    local = np.full(boundary.shape, BG, dtype=int)
    idx = random.randrange(n)
    for color, length in zip(colors, lengths, strict=False):
        for _ in range(length):
            r, c = path[idx]
            local[r, c] = int(color)
            idx = (idx + 1) % n

    return local


def _embed(local: np.ndarray) -> np.ndarray | None:
    h0, w0 = local.shape
    max_h_pad = 30 - h0
    max_w_pad = 30 - w0
    if max_h_pad < 2 or max_w_pad < 2:
        return None

    top = random.randint(1, max_h_pad - 1)
    bottom = random.randint(1, max_h_pad - top)
    left = random.randint(1, max_w_pad - 1)
    right = random.randint(1, max_w_pad - left)

    out = np.full((h0 + top + bottom, w0 + left + right), BG, dtype=int)
    out[top : top + h0, left : left + w0] = local
    return out


def _gen_attempt() -> np.ndarray | None:
    boundary = _random_region()
    if boundary is None:
        return None

    local = _paint_segments_on_ring(boundary)
    if local is None:
        return None

    inp = _embed(local)
    if inp is None:
        return None

    out = main(inp)
    if np.array_equal(inp, out):
        return None
    if len(set(map(int, inp[inp != BG]))) < 4:
        return None
    if len(set(map(int, out[out != BG]))) < 3:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt()
        except AssertionError:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
