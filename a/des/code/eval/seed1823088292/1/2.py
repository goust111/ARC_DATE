from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
RED = 2
FRAGMENT_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# --------------- solver ---------------


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    b = _bbox(mask)
    assert b is not None
    r0, c0, r1, c1 = b
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = np.asarray(mask, dtype=bool)
    return tuple(map(int, m.shape)), m.tobytes()


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False

    h, w = map(int, mask.shape)
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(pts)


def _find_split_row(g: np.ndarray) -> int:
    rows = [r for r in range(g.shape[0]) if bool(np.all(g[r, :] == RED))]
    assert len(rows) == 1
    return int(rows[0])


def _find_fragments(top: np.ndarray) -> list[dict]:
    h, w = map(int, top.shape)
    seen = np.zeros((h, w), dtype=bool)
    fragments: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(top[r, c])
            if color == BLACK or seen[r, c]:
                continue

            coords: list[tuple[int, int]] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(top[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rs = [rr for rr, _ in coords]
            cs = [cc for _, cc in coords]
            r0 = min(rs)
            r1 = max(rs)
            c0 = min(cs)
            c1 = max(cs)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True

            fragments.append(
                {
                    "color": color,
                    "area": int(mask.sum()),
                    "bbox_area": int(mask.shape[0] * mask.shape[1]),
                    "mask": mask,
                }
            )

    return fragments


def _pair_fragments(fragments: list[dict]) -> list[dict]:
    by_mask: dict[tuple[tuple[int, int], bytes], list[int]] = defaultdict(list)
    for i, frag in enumerate(fragments):
        by_mask[_mask_signature(frag["mask"])].append(i)

    used: set[int] = set()
    pairs: list[dict] = []
    order = sorted(
        range(len(fragments)),
        key=lambda i: (fragments[i]["bbox_area"], fragments[i]["area"]),
        reverse=True,
    )
    for i in order:
        if i in used:
            continue

        neg = ~fragments[i]["mask"]
        if not bool(neg.any()):
            continue
        candidate_mask = _crop_bool(neg)
        sig = _mask_signature(candidate_mask)
        candidates = [j for j in by_mask.get(sig, []) if j != i and j not in used]
        if len(candidates) != 1:
            continue

        j = int(candidates[0])
        used.add(i)
        used.add(j)
        h, w = map(int, fragments[i]["mask"].shape)
        fill = int(fragments[i]["color"])
        if fragments[j]["area"] > fragments[i]["area"]:
            fill = int(fragments[j]["color"])
        pairs.append({"h": h, "w": w, "area": h * w, "fill": fill})

    assert len(used) == len(fragments) and len(pairs) * 2 == len(fragments)
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble complementary fragment pairs into rectangles and sort them by rectangle area."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)

    split = _find_split_row(g)
    fragments = _find_fragments(g[:split, :])
    assert fragments and len(fragments) % 2 == 0

    rects = _pair_fragments(fragments)
    rects.sort(key=lambda item: (item["area"], item["w"], item["h"]))

    out = np.full((h, w), BLACK, dtype=int)
    out[split, :] = RED

    row0 = split + 2
    col0 = 0
    for rect in rects:
        rh = int(rect["h"])
        rw = int(rect["w"])
        assert row0 + rh <= h
        assert col0 + rw <= w
        out[row0 : row0 + rh, col0 : col0 + rw] = int(rect["fill"])
        col0 += rw + 1

    return out


# --------------- generator ---------------

PAIR_COUNT_MIN = 2
PAIR_COUNT_MAX = 4
RECT_SIDE_MIN = 4
RECT_SIDE_MAX = 7
GRID_SIDE_MAX = 30
PLACEMENT_GAP = 1
MAX_GENERATION_ATTEMPTS = 300
PAIR_ATTEMPTS = 200
PIECE_ATTEMPTS = 200
LAYOUT_ATTEMPTS = 120

ALL_DIMS = [(hh, ww) for hh in range(RECT_SIDE_MIN, RECT_SIDE_MAX + 1) for ww in range(RECT_SIDE_MIN, RECT_SIDE_MAX + 1)]
TIE_GROUPS = [
    ((hh, ww), (ww, hh))
    for hh in range(RECT_SIDE_MIN, RECT_SIDE_MAX + 1)
    for ww in range(hh + 1, RECT_SIDE_MAX + 1)
]


def _random_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    out = [(r, c)]
    while (r, c) != (tr, tc):
        moves: list[tuple[int, int]] = []
        if r != tr:
            moves.append((1 if tr > r else -1, 0))
        if c != tc:
            moves.append((0, 1 if tc > c else -1))
        dr, dc = random.choice(moves)
        r += dr
        c += dc
        out.append((r, c))
    return out


def _seed_touching_all_sides(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    anchors = [
        (0, random.randrange(w)),
        (h - 1, random.randrange(w)),
        (random.randrange(h), 0),
        (random.randrange(h), w - 1),
    ]
    hub = (random.randrange(h), random.randrange(w))
    for anchor in anchors:
        for r, c in _random_path(anchor, hub):
            mask[r, c] = True
    return mask


def _grow_connected(mask: np.ndarray, target_area: int) -> np.ndarray:
    h, w = map(int, mask.shape)
    out = mask.copy()
    while int(out.sum()) < target_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(out):
            rr = int(r)
            cc = int(c)
            for dr, dc in NEIGHBORS4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and not out[nr, nc]:
                    frontier.add((nr, nc))
        assert frontier
        nr, nc = random.choice(list(frontier))
        out[nr, nc] = True
    return out


def _build_pair_masks(h: int, w: int, piece_dominant: bool) -> tuple[np.ndarray, np.ndarray] | None:
    full_area = h * w
    for _ in range(PAIR_ATTEMPTS):
        bh = random.randint(3, h)
        bw = random.randint(3, w)
        if bh == h and bw == w:
            continue

        r0 = random.randint(0, h - bh)
        c0 = random.randint(0, w - bw)
        max_piece_area = bh * bw - 1
        if piece_dominant:
            low = full_area // 2 + 1
            high = max_piece_area
        else:
            low = 1
            high = min(max_piece_area, (full_area - 1) // 2)
        if low > high:
            continue

        for _ in range(PIECE_ATTEMPTS):
            seed = _seed_touching_all_sides(bh, bw)
            min_piece_area = int(seed.sum())
            lo = max(low, min_piece_area)
            if lo > high:
                continue

            target_area = random.randint(lo, high)
            piece_local = _grow_connected(seed, target_area)
            if not _connected(piece_local):
                continue
            if _bbox(piece_local) != (0, 0, bh - 1, bw - 1):
                continue
            if int(piece_local.sum()) >= bh * bw:
                continue

            piece = np.zeros((h, w), dtype=bool)
            piece[r0 : r0 + bh, c0 : c0 + bw] = piece_local
            host = ~piece
            if not _connected(host):
                continue
            if _bbox(host) != (0, 0, h - 1, w - 1):
                continue
            return host, piece_local
    return None


def _choose_dims(n_pairs: int) -> list[tuple[int, int]] | None:
    want_tie = n_pairs >= 2 and random.random() < 0.5
    for _ in range(200):
        dims: list[tuple[int, int]] = []
        used: set[tuple[int, int]] = set()
        if want_tie:
            pair = random.choice(TIE_GROUPS)
            dims.extend(pair)
            used |= set(pair)

        pool = [dim for dim in ALL_DIMS if dim not in used]
        random.shuffle(pool)
        dims.extend(pool[: n_pairs - len(dims)])
        if len(dims) != n_pairs:
            continue
        if sum(width for _height, width in dims) + (n_pairs - 1) <= GRID_SIDE_MAX:
            return dims
    return None


def _rect_clear(candidate: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]) -> bool:
    r, c, h, w = candidate
    for pr, pc, ph, pw in placed:
        separated = (
            r + h + PLACEMENT_GAP <= pr
            or pr + ph + PLACEMENT_GAP <= r
            or c + w + PLACEMENT_GAP <= pc
            or pc + pw + PLACEMENT_GAP <= c
        )
        if not separated:
            return False
    return True


def _place_fragments(fragments: list[dict], top_h: int, width: int) -> np.ndarray | None:
    top = np.full((top_h, width), BLACK, dtype=int)
    placed: list[tuple[int, int, int, int]] = []

    order = list(range(len(fragments)))
    random.shuffle(order)
    order.sort(
        key=lambda i: fragments[i]["patch"].shape[0] * fragments[i]["patch"].shape[1],
        reverse=True,
    )

    for idx in order:
        patch = fragments[idx]["patch"]
        color = int(fragments[idx]["color"])
        ph, pw = map(int, patch.shape)
        if ph > top_h or pw > width:
            return None

        ok = False
        for _ in range(200):
            r0 = random.randint(0, top_h - ph)
            c0 = random.randint(0, width - pw)
            if not _rect_clear((r0, c0, ph, pw), placed):
                continue
            top[r0 : r0 + ph, c0 : c0 + pw][patch] = color
            placed.append((r0, c0, ph, pw))
            ok = True
            break
        if not ok:
            return None

    return top


def _render_expected(shape: tuple[int, int], split: int, rects: list[dict]) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), BLACK, dtype=int)
    out[split, :] = RED

    row0 = split + 2
    col0 = 0
    for rect in sorted(rects, key=lambda item: (item["area"], item["w"], item["h"])):
        rh = int(rect["h"])
        rw = int(rect["w"])
        out[row0 : row0 + rh, col0 : col0 + rw] = int(rect["fill"])
        col0 += rw + 1
    return out


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.randint(PAIR_COUNT_MIN, PAIR_COUNT_MAX)
    dims = _choose_dims(n_pairs)
    if dims is None:
        return None

    colors = random.sample(FRAGMENT_COLORS, 2 * n_pairs)
    pairs: list[dict] = []
    fragments: list[dict] = []

    for i, (h, w) in enumerate(dims):
        host_color = colors[2 * i]
        piece_color = colors[2 * i + 1]
        piece_dominant = random.random() < 0.5
        built = _build_pair_masks(h, w, piece_dominant)
        if built is None:
            return None
        host_mask, piece_mask = built
        host_area = int(host_mask.sum())
        piece_area = int(piece_mask.sum())
        fill = piece_color if piece_area > host_area else host_color

        host_patch = np.full((h, w), BLACK, dtype=int)
        host_patch[host_mask] = host_color
        piece_patch = np.full(piece_mask.shape, BLACK, dtype=int)
        piece_patch[piece_mask] = piece_color

        pairs.append({"h": h, "w": w, "area": h * w, "fill": fill})
        fragments.append({"patch": host_mask, "color": host_color})
        fragments.append({"patch": piece_mask, "color": piece_color})

        if not _connected(host_mask) or not _connected(piece_mask):
            return None

    required_out_w = sum(pair["w"] for pair in pairs) + (len(pairs) - 1)
    if required_out_w > GRID_SIDE_MAX:
        return None

    fragment_area_budget = sum((patch["patch"].shape[0] + 1) * (patch["patch"].shape[1] + 1) for patch in fragments)
    max_frag_h = max(patch["patch"].shape[0] for patch in fragments)
    max_rect_h = max(pair["h"] for pair in pairs)

    for _ in range(LAYOUT_ATTEMPTS):
        width_low = max(required_out_w, max(patch["patch"].shape[1] for patch in fragments), 14)
        width = random.randint(width_low, GRID_SIDE_MAX)

        top_base = max(
            max_frag_h + 1,
            (fragment_area_budget + max(1, width - 1) - 1) // max(1, width - 1),
        )
        top_cap = GRID_SIDE_MAX - (max_rect_h + 2)
        if top_base > top_cap:
            continue
        top_h = random.randint(top_base, min(top_base + 4, top_cap))
        height = top_h + 2 + max_rect_h + random.randint(0, 2)
        if height > GRID_SIDE_MAX:
            continue

        top = _place_fragments(fragments, top_h, width)
        if top is None:
            continue

        grid = np.full((height, width), BLACK, dtype=int)
        grid[:top_h, :] = top
        grid[top_h, :] = RED

        expected = _render_expected(grid.shape, top_h, pairs)
        if not np.array_equal(main(grid), expected):
            continue
        if int(np.sum(grid == BLACK)) <= int(np.sum(grid != BLACK)):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
