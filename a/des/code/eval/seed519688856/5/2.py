import random
from collections import Counter, deque

import numpy as np


GEN_BG = 0
GEN_ANCHOR = 5
GEN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400
SPLIT_ATTEMPTS = 60


def _components(mask: np.ndarray) -> list[dict]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in pts]
            cols = [p[1] for p in pts]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for pr, pc in pts:
                sub[pr - r0, pc - c0] = True
            comps.append({"bbox": (r0, c0, r1, c1), "mask": sub, "area": len(pts)})
    return comps


def _ring_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _ring_cycle(h: int, w: int) -> list[tuple[int, int]]:
    pts = []
    for c in range(w):
        pts.append((0, c))
    for r in range(1, h - 1):
        pts.append((r, w - 1))
    for c in range(w - 1, -1, -1):
        pts.append((h - 1, c))
    for r in range(h - 2, 0, -1):
        pts.append((r, 0))
    return pts


def _sample_lengths(total: int, parts: int, minimum: int, rng: random.Random) -> list[int] | None:
    if total < parts * minimum:
        return None
    lengths = []
    remaining = total
    for idx in range(parts - 1):
        min_remaining = minimum * (parts - idx - 1)
        value = rng.randint(minimum, remaining - min_remaining)
        lengths.append(value)
        remaining -= value
    lengths.append(remaining)
    rng.shuffle(lengths)
    return lengths


def _split_ring(h: int, w: int, parts: int, rng: random.Random) -> list[np.ndarray] | None:
    cycle = _ring_cycle(h, w)
    lengths = _sample_lengths(len(cycle), parts, minimum=2, rng=rng)
    if lengths is None:
        return None
    start = rng.randrange(len(cycle))
    pieces = []
    offset = start
    for length in lengths:
        mask = np.zeros((h, w), dtype=bool)
        for step in range(length):
            rr, cc = cycle[(offset + step) % len(cycle)]
            mask[rr, cc] = True
        comps = _components(mask)
        if len(comps) != 1:
            return None
        pieces.append(comps[0]["mask"])
        offset = (offset + length) % len(cycle)
    return pieces


def _candidate_ring_dims(area: int, max_h: int, max_w: int) -> list[tuple[int, int]]:
    if (area + 4) % 2 != 0:
        return []
    total = (area + 4) // 2
    dims = []
    for h in range(max(3, max_h), total - 2 + 1):
        w = total - h
        if w < max(3, max_w):
            continue
        if h % 2 == 0 or w % 2 == 0:
            continue
        dims.append((h, w))
    return dims


def _placement_bitmasks(ring: np.ndarray, frag: np.ndarray) -> list[int]:
    h, w = ring.shape
    fh, fw = frag.shape
    ring_pts = [(int(r), int(c)) for r, c in np.argwhere(ring)]
    index = {pt: idx for idx, pt in enumerate(ring_pts)}
    frag_pts = [(int(r), int(c)) for r, c in np.argwhere(frag)]
    masks = []
    for r0 in range(h - fh + 1):
        for c0 in range(w - fw + 1):
            bits = 0
            ok = True
            for fr, fc in frag_pts:
                rr, cc = r0 + fr, c0 + fc
                if not ring[rr, cc]:
                    ok = False
                    break
                bits |= 1 << index[(rr, cc)]
            if ok:
                masks.append(bits)
    return sorted(set(masks))


def _has_exact_cover(h: int, w: int, fragments: list[np.ndarray]) -> bool:
    ring = _ring_mask(h, w)
    ring_area = int(ring.sum())
    if sum(int(f.sum()) for f in fragments) != ring_area:
        return False
    options = []
    for frag in fragments:
        placements = _placement_bitmasks(ring, frag)
        if not placements:
            return False
        options.append(placements)
    order = sorted(range(len(fragments)), key=lambda idx: len(options[idx]))
    ordered = [options[idx] for idx in order]
    full = (1 << ring_area) - 1

    def dfs(depth: int, covered: int) -> bool:
        if depth == len(ordered):
            return covered == full
        for bits in ordered[depth]:
            if covered & bits:
                continue
            if dfs(depth + 1, covered | bits):
                return True
        return False

    return dfs(0, 0)


def _recover_ring_dims(fragments: list[np.ndarray]) -> tuple[int, int]:
    area = sum(int(f.sum()) for f in fragments)
    max_h = max(int(f.shape[0]) for f in fragments)
    max_w = max(int(f.shape[1]) for f in fragments)
    matches = []
    for h, w in _candidate_ring_dims(area, max_h, max_w):
        if _has_exact_cover(h, w, fragments):
            matches.append((h, w))
    assert len(matches) == 1, "ring fragments must reconstruct a unique rectangular ring"
    return matches[0]


def _assemble_output(rings: list[tuple[int, int, int]], bg: int) -> np.ndarray:
    outer_h = max(h for h, _, _ in rings)
    outer_w = max(w for _, w, _ in rings)
    out = np.full((outer_h, outer_w), int(bg), dtype=int)
    for h, w, color in sorted(rings, key=lambda item: (item[0] * item[1], item[0] + item[1])):
        r0 = (outer_h - h) // 2
        c0 = (outer_w - w) // 2
        mask = _ring_mask(h, w)
        out[r0 : r0 + h, c0 : c0 + w][mask] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    counts = Counter(int(v) for v in g.ravel())
    bg = max(counts, key=counts.get)
    anchor_colors = [color for color, count in counts.items() if count == 1 and color != bg]
    assert len(anchor_colors) == 1, "expected a unique single-pixel anchor color"
    anchor = anchor_colors[0]

    rings = []
    for color in sorted(counts):
        if color in (bg, anchor):
            continue
        comps = _components(g == int(color))
        assert 1 <= len(comps) <= 4, "each ring color should have between 1 and 4 fragments"
        dims = _recover_ring_dims([comp["mask"] for comp in comps])
        rings.append((dims[0], dims[1], int(color)))

    assert rings, "expected at least one colored ring"
    return _assemble_output(rings, bg)


def _paste_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    region = grid[top : top + h, left : left + w]
    region[mask] = int(color)


def _scatter_boxes(shapes: list[tuple[int, int]], h: int, w: int, rng: random.Random) -> tuple[list[tuple[int, int]], np.ndarray] | None:
    blocked = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(shapes)
    order = sorted(range(len(shapes)), key=lambda idx: shapes[idx][0] * shapes[idx][1], reverse=True)
    for idx in order:
        sh, sw = shapes[idx]
        candidates = []
        for top in range(1, h - sh):
            for left in range(1, w - sw):
                r0 = top - 1
                r1 = top + sh + 1
                c0 = left - 1
                c1 = left + sw + 1
                if not blocked[r0:r1, c0:c1].any():
                    candidates.append((top, left))
        if not candidates:
            return None
        top, left = rng.choice(candidates)
        placements[idx] = (top, left)
        blocked[top - 1 : top + sh + 1, left - 1 : left + sw + 1] = True
    return [p for p in placements if p is not None], blocked


def _choose_anchor(blocked: np.ndarray, rng: random.Random) -> tuple[int, int] | None:
    h, w = blocked.shape
    preferred = []
    fallback = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if blocked[r, c]:
                continue
            fallback.append((r, c))
            if h // 4 <= r < h - h // 4 and w // 4 <= c < w - w // 4:
                preferred.append((r, c))
    pool = preferred if preferred else fallback
    if not pool:
        return None
    return rng.choice(pool)


def _make_pieces_for_ring(h: int, w: int, parts: int, rng: random.Random) -> list[np.ndarray] | None:
    for _ in range(SPLIT_ATTEMPTS):
        pieces = _split_ring(h, w, parts, rng)
        if pieces is None:
            continue
        try:
            dims = _recover_ring_dims(pieces)
        except AssertionError:
            continue
        if dims == (h, w):
            return pieces
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    ring_count = rng.choice([2, 3, 4])
    base_h = rng.choice([3, 5, 7])
    base_w = rng.choice([3, 5, 7, 9])
    colors = rng.sample(GEN_COLORS, ring_count)

    rings = []
    fragments = []
    for idx in range(ring_count):
        h = base_h + 4 * idx
        w = base_w + 4 * idx
        color = colors[idx]
        parts = rng.choice([2, 3, 4])
        pieces = _make_pieces_for_ring(h, w, parts, rng)
        if pieces is None:
            return None
        rings.append((h, w, color))
        for piece in pieces:
            fragments.append((piece, color))

    max_h = max(h for h, _, _ in rings)
    max_w = max(w for _, w, _ in rings)
    grid_h = rng.randint(max(28, max_h + 10), max(34, max_h + 18))
    grid_w = rng.randint(max(28, max_w + 10), max(34, max_w + 18))

    placement = _scatter_boxes([(mask.shape[0], mask.shape[1]) for mask, _ in fragments], grid_h, grid_w, rng)
    if placement is None:
        return None
    locations, blocked = placement
    anchor = _choose_anchor(blocked, rng)
    if anchor is None:
        return None

    grid = np.full((grid_h, grid_w), GEN_BG, dtype=int)
    for (mask, color), (top, left) in zip(fragments, locations):
        _paste_mask(grid, mask, top, left, color)
    grid[anchor] = GEN_ANCHOR

    expected = _assemble_output(rings, GEN_BG)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring puzzle")
