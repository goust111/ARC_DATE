from __future__ import annotations

from collections import Counter, deque

import numpy as np


TYPE_ORDER = ("both", "h_only", "v_only", "none")
LEGEND_SIZE = 2
PALETTE = tuple(range(10))

_RNG = np.random.default_rng()


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _mask_from_component(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    mask = np.zeros((h, w), dtype=bool)
    rel = coords - np.array([[r0, c0]], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _classify_mask(mask: np.ndarray) -> str:
    mask = np.asarray(mask, dtype=bool)
    h_sym = bool(np.array_equal(mask, np.flipud(mask)))
    v_sym = bool(np.array_equal(mask, np.fliplr(mask)))
    if h_sym and v_sym:
        return "both"
    if h_sym:
        return "h_only"
    if v_sym:
        return "v_only"
    return "none"


def _legend_colors(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 2 * LEGEND_SIZE and w >= 2 * LEGEND_SIZE

    # Corner position is the legend:
    # top border => horizontal bit, left border => vertical bit.
    patches = {
        "both": g[:LEGEND_SIZE, :LEGEND_SIZE],
        "h_only": g[:LEGEND_SIZE, w - LEGEND_SIZE : w],
        "v_only": g[h - LEGEND_SIZE : h, :LEGEND_SIZE],
        "none": g[h - LEGEND_SIZE : h, w - LEGEND_SIZE : w],
    }

    legend: dict[str, int] = {}
    colors: list[int] = []
    for key, patch in patches.items():
        color = int(patch[0, 0])
        assert bool(np.all(patch == color)), "corner legend blocks must be solid 2x2"
        legend[key] = color
        colors.append(color)
    assert len(set(colors)) == 4, "legend colors must be distinct"
    return legend


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let same-color objects vote by symmetry type and output the winning legend color.

    Input structure:
    - The dominant color is background.
    - Several monochrome 4-connected objects share one foreground color.
    - Each object's tight bounding box is square.
    - The four corners contain solid 2x2 legend blocks with distinct colors.

    Rule:
    - Read the legend from corner position: top+left => both symmetries, top only => horizontal,
      left only => vertical, neither => none.
    - Ignore the four legend blocks.
    - Classify each remaining object by reflection symmetry of its bbox mask.
    - Count votes, require a unique winner, and output a 2x2 square in that legend color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 6

    bg = int(np.bincount(g.ravel()).argmax())
    legend = _legend_colors(g)

    work = g.copy()
    h, w = work.shape
    work[:LEGEND_SIZE, :LEGEND_SIZE] = bg
    work[:LEGEND_SIZE, w - LEGEND_SIZE : w] = bg
    work[h - LEGEND_SIZE : h, :LEGEND_SIZE] = bg
    work[h - LEGEND_SIZE : h, w - LEGEND_SIZE : w] = bg

    payload_colors = [int(c) for c in np.unique(work) if int(c) != bg]
    assert len(payload_colors) == 1, "expected exactly one object color after removing the legend"
    obj_color = payload_colors[0]

    comps = _components4(work == obj_color)
    assert comps, "expected at least one voting object"

    votes = Counter()
    for comp in comps:
        mask = _mask_from_component(comp)
        assert mask.shape[0] == mask.shape[1], "object bbox must be square"
        votes[_classify_mask(mask)] += 1

    best = max(votes.values())
    winners = [kind for kind, count in votes.items() if count == best]
    assert len(winners) == 1, "vote winner must be unique"
    return np.full((LEGEND_SIZE, LEGEND_SIZE), int(legend[winners[0]]), dtype=int)


# --------------- generator ---------------

GRID_H_RANGE = (22, 31)
GRID_W_RANGE = (22, 31)
OBJECT_SIZE_RANGE = (3, 7)
GEN_RETRIES = 500
PLACE_RETRIES = 400
PROFILES = (
    (3, 2, 1, 1),
    (3, 2, 2, 1),
    (4, 2, 1, 1),
    (4, 3, 1, 1),
)


def _center_slice(n: int) -> slice:
    if n % 2 == 1:
        return slice(n // 2, n // 2 + 1)
    return slice(n // 2 - 1, n // 2 + 1)


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_valid_shape(mask: np.ndarray, kind: str) -> bool:
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if mask.shape[0] != mask.shape[1]:
        return False
    if not bool(mask.any()):
        return False
    if not _touches_all_sides(mask):
        return False
    comps = _components4(mask)
    if len(comps) != 1:
        return False
    return _classify_mask(mask) == kind


def _make_both(n: int) -> np.ndarray:
    for _ in range(40):
        mode = int(_RNG.integers(0, 4))
        mask = np.zeros((n, n), dtype=bool)
        if mode == 0:
            mask[:, :] = True
        elif mode == 1:
            mid_r = _center_slice(n)
            mid_c = _center_slice(n)
            mask[mid_r, :] = True
            mask[:, mid_c] = True
        elif mode == 2:
            mask[[0, -1], :] = True
            mask[:, [0, -1]] = True
        else:
            rr, cc = np.indices((n, n))
            dist = np.abs(2 * rr - (n - 1)) + np.abs(2 * cc - (n - 1))
            threshold = n - 1 + int(_RNG.integers(0, 2))
            mask = dist <= threshold
        if _is_valid_shape(mask, "both"):
            return mask
    raise RuntimeError("failed to create both-symmetric shape")


def _make_h_only(n: int) -> np.ndarray:
    mid_r = _center_slice(n)
    mid_cols = range(mid_r.start, mid_r.stop)
    left_pool = [c for c in range(n) if c < mid_r.start]
    right_pool = [c for c in range(n) if c >= mid_r.stop]

    for _ in range(60):
        mask = np.zeros((n, n), dtype=bool)
        mask[mid_r, :] = True

        pool = left_pool if int(_RNG.integers(0, 2)) == 0 else right_pool
        assert pool
        stem_count = int(_RNG.integers(1, min(2, len(pool)) + 1))
        stems = _RNG.choice(pool, size=stem_count, replace=False)
        mask[:, stems] = True

        # Optional same-side mirrored bumps for shape variety while preserving horizontal symmetry.
        if n >= 5 and int(_RNG.integers(0, 2)) == 1:
            bump_col = int(_RNG.choice(pool))
            offset = int(_RNG.integers(1, mid_r.start + 1))
            top = mid_r.start - offset
            bottom = mid_r.stop - 1 + offset
            mask[top, max(0, bump_col - 1) : min(n, bump_col + 2)] = True
            mask[bottom, max(0, bump_col - 1) : min(n, bump_col + 2)] = True

        # Guard against degenerating into full symmetry.
        if not any(c not in mid_cols for c in stems):
            continue
        if _is_valid_shape(mask, "h_only"):
            return mask
    raise RuntimeError("failed to create h-only shape")


def _make_v_only(n: int) -> np.ndarray:
    return _make_h_only(n).T.copy()


def _make_none_from_h(n: int) -> np.ndarray:
    mid_r = _center_slice(n)
    for _ in range(80):
        mask = _make_h_only(n).copy()
        full_cols = [c for c in range(n) if bool(np.all(mask[:, c]))]
        if not full_cols:
            continue
        if max(full_cols) < mid_r.start:
            opposite_pool = [c for c in range(mid_r.stop, n)]
        else:
            opposite_pool = [c for c in range(0, mid_r.start)]
        if not opposite_pool:
            opposite_pool = [c for c in range(n) if c not in full_cols]
        if not opposite_pool:
            continue

        spur_col = int(_RNG.choice(opposite_pool))
        if int(_RNG.integers(0, 2)) == 0:
            mask[: mid_r.start, spur_col] = True
        else:
            mask[mid_r.stop :, spur_col] = True
        if _is_valid_shape(mask, "none"):
            return mask
    raise RuntimeError("failed to create non-symmetric shape")


def _make_none(n: int) -> np.ndarray:
    mask = _make_none_from_h(n)
    if int(_RNG.integers(0, 2)) == 1:
        mask = mask.T.copy()
    assert _is_valid_shape(mask, "none")
    return mask


def _random_shape(kind: str, n: int) -> np.ndarray:
    if kind == "both":
        return _make_both(n)
    if kind == "h_only":
        return _make_h_only(n)
    if kind == "v_only":
        return _make_v_only(n)
    if kind == "none":
        return _make_none(n)
    raise ValueError(kind)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(h, h - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(w, w - dc)
            dst_r0 = max(0, dr)
            dst_r1 = min(h, h + dr)
            dst_c0 = max(0, dc)
            dst_c1 = min(w, w + dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _reserve_legend_moat(h: int, w: int) -> np.ndarray:
    blocked = np.zeros((h, w), dtype=bool)
    zones = (
        (0, min(h, LEGEND_SIZE + 1), 0, min(w, LEGEND_SIZE + 1)),
        (0, min(h, LEGEND_SIZE + 1), max(0, w - LEGEND_SIZE - 1), w),
        (max(0, h - LEGEND_SIZE - 1), h, 0, min(w, LEGEND_SIZE + 1)),
        (max(0, h - LEGEND_SIZE - 1), h, max(0, w - LEGEND_SIZE - 1), w),
    )
    for r0, r1, c0, c1 in zones:
        blocked[r0:r1, c0:c1] = True
    return blocked


def _vote_profile() -> tuple[str, dict[str, int]]:
    winner = TYPE_ORDER[int(_RNG.integers(0, len(TYPE_ORDER)))]
    profile = list(PROFILES[int(_RNG.integers(0, len(PROFILES)))])
    losers = profile[1:]
    _RNG.shuffle(losers)
    counts: dict[str, int] = {}
    loser_idx = 0
    for kind in TYPE_ORDER:
        if kind == winner:
            counts[kind] = int(profile[0])
        else:
            counts[kind] = int(losers[loser_idx])
            loser_idx += 1
    return winner, counts


def _sample_specs() -> tuple[str, dict[str, int], list[tuple[str, np.ndarray]]]:
    winner, counts = _vote_profile()
    specs: list[tuple[str, np.ndarray]] = []
    for kind in TYPE_ORDER:
        for _ in range(counts[kind]):
            n = int(_RNG.integers(*OBJECT_SIZE_RANGE))
            specs.append((kind, _random_shape(kind, n)))
    _RNG.shuffle(specs)
    specs.sort(key=lambda item: int(item[1].sum()), reverse=True)
    return winner, counts, specs


def _place_objects(
    h: int,
    w: int,
    bg: int,
    obj_color: int,
    legend: dict[str, int],
    specs: list[tuple[str, np.ndarray]],
) -> np.ndarray | None:
    grid = np.full((h, w), int(bg), dtype=int)
    grid[:LEGEND_SIZE, :LEGEND_SIZE] = int(legend["both"])
    grid[:LEGEND_SIZE, w - LEGEND_SIZE : w] = int(legend["h_only"])
    grid[h - LEGEND_SIZE : h, :LEGEND_SIZE] = int(legend["v_only"])
    grid[h - LEGEND_SIZE : h, w - LEGEND_SIZE : w] = int(legend["none"])

    occupied = np.zeros((h, w), dtype=bool)
    base_blocked = _reserve_legend_moat(h, w)

    for kind, mask in specs:
        del kind
        n = int(mask.shape[0])
        placed = False
        for _ in range(PLACE_RETRIES):
            current_blocked = base_blocked | _dilate8(occupied)
            r0 = int(_RNG.integers(0, h - n + 1))
            c0 = int(_RNG.integers(0, w - n + 1))
            if bool((current_blocked[r0 : r0 + n, c0 : c0 + n] & mask).any()):
                continue
            sub = grid[r0 : r0 + n, c0 : c0 + n]
            if bool((sub != bg).any()):
                continue
            sub[mask] = int(obj_color)
            occupied[r0 : r0 + n, c0 : c0 + n] |= mask
            placed = True
            break
        if not placed:
            return None

    return grid


def _generate_case() -> np.ndarray | None:
    h = int(_RNG.integers(*GRID_H_RANGE))
    w = int(_RNG.integers(*GRID_W_RANGE))
    if h < 10 or w < 10:
        return None

    colors = list(_RNG.choice(PALETTE, size=6, replace=False))
    bg, obj_color, legend_both, legend_h, legend_v, legend_none = map(int, colors)
    legend = {
        "both": legend_both,
        "h_only": legend_h,
        "v_only": legend_v,
        "none": legend_none,
    }

    winner, counts, specs = _sample_specs()
    if counts[winner] <= max(v for k, v in counts.items() if k != winner):
        return None

    grid = _place_objects(h, w, bg, obj_color, legend, specs)
    if grid is None:
        return None

    bg_count = int(np.count_nonzero(grid == bg))
    if bg_count <= (grid.size // 2):
        return None

    expected = np.full((LEGEND_SIZE, LEGEND_SIZE), int(legend[winner]), dtype=int)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid voting-by-symmetry ARC input")
