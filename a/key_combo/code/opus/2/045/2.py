import random
from collections import deque

import numpy as np

BG = 0
GRAY = 5
FRAME_COLORS = (1, 2, 3, 4, 7)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _norm_key(cells):
    r0, r1, c0, c1 = _bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=np.int8)
    for r, c in cells:
        mask[r - r0, c - c0] = 1
    return tuple(mask.flatten())


def _find_gray_cross(grid: np.ndarray):
    rows = [int(r) for r in np.where(np.all(grid == GRAY, axis=1))[0]]
    cols = [int(c) for c in np.where(np.all(grid == GRAY, axis=0))[0]]
    if not rows or not cols:
        return None, None
    return rows[0], cols[0]


def _quadrants(h: int, w: int, sep_r: int, sep_c: int):
    return {
        0: (0, sep_r, 0, sep_c),
        1: (0, sep_r, sep_c + 1, w),
        2: (sep_r + 1, h, sep_c + 1, w),
        3: (sep_r + 1, h, 0, sep_c),
    }


def _enclosed_zero_components(sub: np.ndarray):
    h, w = sub.shape
    zero = sub == BG
    outside = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if zero[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if zero[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and zero[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    holes = []
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not zero[r, c] or outside[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and zero[nr, nc] and not outside[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            holes.append(cells)
    return holes


def _cycle_color(color: int, steps: int) -> int:
    if color not in FRAME_COLORS:
        return color
    ring = FRAME_COLORS
    idx = ring.index(color)
    return ring[(idx + (steps % len(ring))) % len(ring)]


def _to_global(cells, off_r: int, off_c: int):
    return [(r + off_r, c + off_c) for r, c in cells]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape

    sep_r, sep_c = _find_gray_cross(g)
    if sep_r is None or sep_c is None:
        return g.copy()

    quads = _quadrants(h, w, sep_r, sep_c)

    template_idx = None
    template_holes = []
    template_bbox = None
    for idx, (r0, r1, c0, c1) in quads.items():
        sub = g[r0:r1, c0:c1]
        holes = [h_ for h_ in _enclosed_zero_components(sub) if 3 <= len(h_) <= 5]
        if 2 <= len(holes) <= 3 and len(holes) > len(template_holes):
            template_idx = idx
            template_holes = holes
            template_bbox = (r0, r1, c0, c1)

    if template_idx is None or template_bbox is None:
        # Fallback: keep the first quadrant with any enclosed zero region
        for idx, (r0, r1, c0, c1) in quads.items():
            holes = [h_ for h_ in _enclosed_zero_components(g[r0:r1, c0:c1]) if len(h_) > 0]
            if holes:
                template_idx = idx
                template_holes = holes
                template_bbox = (r0, r1, c0, c1)
                break

    if template_idx is None or template_bbox is None:
        return g.copy()

    # Build reference shape->(color, quadrant) map from non-template quadrants
    ref_shapes = {}
    for qi, (r0, r1, c0, c1) in quads.items():
        if qi == template_idx:
            continue
        sub = g[r0:r1, c0:c1]
        mask = (sub != BG) & (sub != GRAY)
        for comp in _components(mask):
            if not comp:
                continue
            key = _norm_key(comp)
            color = int(sub[comp[0]])
            ref_shapes.setdefault(key, []).append((qi, color))

    out = g[template_bbox[0] : template_bbox[1], template_bbox[2] : template_bbox[3]].copy()
    for hole in template_holes:
        if not (3 <= len(hole) <= 5):
            continue
        key = _norm_key(hole)
        candidates = ref_shapes.get(key)
        if not candidates:
            continue
        q_src, color = candidates[0]
        steps = (q_src - template_idx) % 4
        fill_color = _cycle_color(color, steps)
        for r, c in hole:
            out[r, c] = fill_color

    return out


def _draw_shape(grid: np.ndarray, top: int, left: int, shape, color: int):
    h, w = grid.shape
    for dr, dc in shape:
        rr = top + dr
        cc = left + dc
        if 0 <= rr < h and 0 <= cc < w:
            grid[rr, cc] = color


def _random_shape_choice(rng_shapes, *, min_size=3, max_size=5):
    pool = [s for s in rng_shapes if min_size <= len(s) <= max_size]
    return random.choice(pool)


def generate_input() -> np.ndarray:
    h, w = 15, 17
    sep_r, sep_c = 7, 8
    g = np.zeros((h, w), dtype=int)

    g[sep_r, :] = GRAY
    g[:, sep_c] = GRAY

    quads = _quadrants(h, w, sep_r, sep_c)
    template_idx = random.randint(0, 3)
    template_color = random.choice([1, 2, 3, 4, 7])

    candidate_colors = [1, 2, 3, 4, 7]
    ref_colors = [c for c in candidate_colors if c != template_color]

    hole_shapes = [
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)],
    ]

    # Fill all quadrants with background; then carve template + references.
    for idx, (r0, r1, c0, c1) in quads.items():
        g[r0:r1, c0:c1] = BG

    # Build template and holes.
    tr0, tr1, tc0, tc1 = quads[template_idx]
    template = g[tr0:tr1, tc0:tc1]
    template[:, :] = template_color

    hole_count = 3
    used = set()
    available_shapes = list(hole_shapes)
    random.shuffle(available_shapes)
    selected_shapes = available_shapes[:hole_count]

    for shape in selected_shapes:
        h0 = max(r for r, _ in shape)
        w0 = max(c for _, c in shape)
        max_r = max(tr0, tr1 - h0 - 2)
        max_c = max(tc0, tc1 - w0 - 2)
        for _ in range(250):
            top = random.randint(tr0 + 1, max_r)
            left = random.randint(tc0 + 1, max_c)
            cells = [(top + dr, left + dc) for dr, dc in shape]
            if any((rr, cc) in used for rr, cc in cells):
                continue
            for rr, cc in cells:
                template[rr - tr0, cc - tc0] = BG
                used.add((rr, cc))
            break

    # Place one matching reference object in each other quadrant.
    ref_shapes = selected_shapes.copy()
    random.shuffle(ref_shapes)
    for qi in [q for q in [0, 1, 2, 3] if q != template_idx]:
        shape = random.choice(ref_shapes)
        r0, r1, c0, c1 = quads[qi]
        qh = max(r for r, _ in shape)
        qw = max(c for _, c in shape)
        color = ref_colors[len(ref_shapes) % len(ref_colors)]
        ref_colors.remove(color)

        max_r = max(r0, r1 - qh - 1)
        max_c = max(c0, c1 - qw - 1)
        if max_r <= r0 or max_c <= c0:
            continue
        placed = False
        for _ in range(250):
            top = random.randint(r0, max_r)
            left = random.randint(c0, max_c)
            cells = [(top + dr, left + dc) for dr, dc in shape]
            if any(g[rr, cc] != BG for rr, cc in cells):
                continue
            for rr, cc in cells:
                g[rr, cc] = color
            placed = True
            break

        if not placed:
            # deterministic fallback at a safe location.
            top = (r0 + r1 - qh - 1) // 2
            left = (c0 + c1 - qw - 1) // 2
            for rr, cc in ((top + dr, left + dc) for dr, dc in shape):
                if 0 <= rr < h and 0 <= cc < w:
                    g[rr, cc] = color

    # Keep template color in the template area even after fallback placements.
    g[tr0:tr1, tc0:tc1] = template
    return g
