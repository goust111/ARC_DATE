from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in N4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bool(bg[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_edge = True
                for dr, dc in N4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(bg[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _find_dividers(g: np.ndarray) -> tuple[int, int]:
    full_rows = [r for r in range(g.shape[0]) if bool(np.all(g[r, :] == GREY))]
    full_cols = [c for c in range(g.shape[1]) if bool(np.all(g[:, c] == GREY))]
    assert full_rows and full_cols, "expected full grey horizontal and vertical divider lines"
    return int(full_rows[0]), int(full_cols[0])


def _find_slots(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = g.shape
    slots: list[tuple[int, int, int, int]] = []
    for pts in _components(g == BG):
        r0, c0, r1, c1 = _bbox(pts)
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if int(len(pts)) != int(area):
            continue
        if not bool(np.all(g[r0 : r1 + 1, c0 : c1 + 1] == BG)):
            continue
        if not bool(np.all(g[r0 - 1, c0 - 1 : c1 + 2] == GREY)):
            continue
        if not bool(np.all(g[r1 + 1, c0 - 1 : c1 + 2] == GREY)):
            continue
        if not bool(np.all(g[r0 - 1 : r1 + 2, c0 - 1] == GREY)):
            continue
        if not bool(np.all(g[r0 - 1 : r1 + 2, c1 + 1] == GREY)):
            continue
        slots.append((r0 - 1, c0 - 1, r1 + 1, c1 + 1))
    assert slots, "expected grey-framed target slots"
    return slots


def _slot_rows(slots: list[tuple[int, int, int, int]]) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    rows: list[list[tuple[int, int, int, int]]] = []
    for box in sorted(slots, key=lambda b: ((b[0] + b[2]) / 2.0, b[1])):
        center = (box[0] + box[2]) / 2.0
        placed = False
        for row in rows:
            row_center = sum((b[0] + b[2]) / 2.0 for b in row) / len(row)
            if abs(center - row_center) <= 1.5:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])
    rows.sort(key=lambda row: sum((b[0] + b[2]) / 2.0 for b in row) / len(row))
    assert len(rows) == 2, "expected exactly two target-slot rows"
    return sorted(rows[0], key=lambda b: b[1]), sorted(rows[1], key=lambda b: b[1])


def _extract_fragments(g: np.ndarray, hline: int, vline: int) -> list[dict]:
    fragments = []
    fg = (g != BG) & (g != GREY)
    for pts in _components(fg):
        vals = {int(g[int(r), int(c)]) for r, c in pts}
        assert len(vals) == 1, "each fragment must be monochrome"
        color = vals.pop()
        r0, c0, r1, c1 = _bbox(pts)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        patch = np.zeros(mask.shape, dtype=int)
        for r, c in pts:
            rr, cc = int(r - r0), int(c - c0)
            mask[rr, cc] = True
            patch[rr, cc] = int(color)

        cr = float(pts[:, 0].mean())
        cc = float(pts[:, 1].mean())
        if cr < hline and cc < vline:
            quadrant = 0
            rot_k = 0
        elif cr < hline and cc > vline:
            quadrant = 1
            rot_k = -1
        elif cr > hline and cc < vline:
            quadrant = 2
            rot_k = 1
        elif cr > hline and cc > vline:
            quadrant = 3
            rot_k = 2
        else:
            raise AssertionError("fragment center must lie in a quadrant")

        fragments.append(
            {
                "coords": pts,
                "patch": patch,
                "holes": _hole_count(mask),
                "quadrant": quadrant,
                "center_col": cc,
                "center_row": cr,
                "rot_k": rot_k,
            }
        )
    assert fragments, "expected at least one colored fragment"
    return fragments


def _paste_centered(out: np.ndarray, patch: np.ndarray, slot: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = slot
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    ph, pw = patch.shape
    assert ph <= inner_h and pw <= inner_w, "rotated fragment must fit inside its slot"
    rr = r0 + 1 + (inner_h - ph) // 2
    cc = c0 + 1 + (inner_w - pw) // 2
    mask = patch != BG
    out[rr : rr + ph, cc : cc + pw][mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate quadrant-coded colored fragments and file them into hole-count target rows.

    Input structure:
    - Background is black, with one full grey horizontal divider and one full grey vertical divider forming four quadrants.
    - Each quadrant contains separated small monochrome colored fragments; some fragments enclose one or two black holes.
    - The lower-right area contains grey-framed target slots arranged in two rows, with black interiors large enough for any rotated fragment.

    Rule:
    - A fragment's quadrant determines its rotation: top-left unchanged, top-right clockwise 90 degrees, bottom-right 180 degrees, and bottom-left counterclockwise 90 degrees.
    - Count enclosed black holes in each fragment; zero-hole fragments go to the upper slot row, while fragments with at least one hole go to the lower slot row.
    - Within each target row, place fragments in reading order by quadrant (top-left, top-right, bottom-left, bottom-right), then by original center column inside the quadrant.
    - Clear each original fragment position to black, rotate the fragment, and center it in its assigned grey-framed slot without changing its color.

    Key insight:
    - Quadrant position encodes rotation, topology (hole count) encodes the destination row, and source reading order encodes each row's left-to-right placement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    hline, vline = _find_dividers(g)
    top_slots, bottom_slots = _slot_rows(_find_slots(g))
    fragments = _extract_fragments(g, hline, vline)
    ordered = sorted(fragments, key=lambda f: (int(f["quadrant"]), float(f["center_col"]), float(f["center_row"])))
    top_frags = [f for f in ordered if int(f["holes"]) == 0]
    bottom_frags = [f for f in ordered if int(f["holes"]) >= 1]
    assert len(top_frags) == len(top_slots), "zero-hole fragment count must match upper slot count"
    assert len(bottom_frags) == len(bottom_slots), "holed fragment count must match lower slot count"

    out = g.copy()
    for frag in fragments:
        pts = frag["coords"]
        out[pts[:, 0], pts[:, 1]] = BG

    for frag, slot in zip(top_frags, top_slots):
        rotated = np.rot90(frag["patch"], int(frag["rot_k"]))
        _paste_centered(out, rotated, slot)
    for frag, slot in zip(bottom_frags, bottom_slots):
        rotated = np.rot90(frag["patch"], int(frag["rot_k"]))
        _paste_centered(out, rotated, slot)

    return out


def _is_connected(mask: np.ndarray) -> bool:
    return int(mask.sum()) > 0 and len(_components(mask)) == 1


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_transform(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.rot90(mask, rng.randrange(4))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    return out.copy()


def _zero_hole_mask(rng: random.Random) -> np.ndarray:
    for _ in range(100):
        h = rng.choice((3, 4))
        w = rng.choice((3, 4))
        mask = np.ones((h, w), dtype=bool)
        border = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if r in (0, h - 1) or c in (0, w - 1)
        ]
        rng.shuffle(border)
        for r, c in border[: rng.randint(1, 4)]:
            trial = mask.copy()
            trial[r, c] = False
            cropped = _crop_mask(trial)
            if cropped.shape[0] >= 3 and cropped.shape[1] >= 3 and _is_connected(cropped) and _hole_count(cropped) == 0:
                mask = trial
        mask = _crop_mask(mask)
        if mask.shape[0] <= 4 and mask.shape[1] <= 4 and _is_connected(mask) and _hole_count(mask) == 0:
            return _random_transform(mask, rng)
    return np.ones((3, 3), dtype=bool)


def _holed_mask(hole_count: int, rng: random.Random) -> np.ndarray:
    if int(hole_count) == 1:
        h, w = rng.choice(((3, 3), (3, 4), (4, 3), (4, 4)))
        mask = np.ones((h, w), dtype=bool)
        mask[rng.randint(1, h - 2), rng.randint(1, w - 2)] = False
    else:
        mask = np.ones((4, 4), dtype=bool)
        if rng.random() < 0.5:
            holes = ((1, 1), (2, 2))
        else:
            holes = ((1, 2), (2, 1))
        for r, c in holes:
            mask[r, c] = False
    assert _is_connected(mask)
    assert _hole_count(mask) == int(hole_count)
    return _random_transform(mask, rng)


def _draw_slot_strip(grid: np.ndarray, r0: int, c0: int, count: int, inner: int) -> list[tuple[int, int, int, int]]:
    outer_h = inner + 2
    width = count * inner + count + 1
    grid[r0, c0 : c0 + width] = GREY
    grid[r0 + outer_h - 1, c0 : c0 + width] = GREY
    slots = []
    for i in range(count + 1):
        c = c0 + i * (inner + 1)
        grid[r0 : r0 + outer_h, c] = GREY
    for i in range(count):
        c = c0 + i * (inner + 1)
        slots.append((r0, c, r0 + outer_h - 1, c + inner + 1))
    return slots


def _mark_rect(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 0) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _sample_counts(rng: random.Random) -> tuple[int, int, int, int]:
    candidates = []
    for tl in range(1, 4):
        for tr in range(1, 4):
            for bl in range(1, 4):
                for br in range(1, 4):
                    total = tl + tr + bl + br
                    if 4 <= total <= 6:
                        candidates.append((tl, tr, bl, br))
    return rng.choice(candidates)


def _quadrant_bounds(q: int, hline: int, vline: int, h: int, w: int) -> tuple[int, int, int, int]:
    if q == 0:
        return 1, hline - 1, 1, vline - 1
    if q == 1:
        return 1, hline - 1, vline + 1, w - 2
    if q == 2:
        return hline + 1, h - 2, 1, vline - 1
    return hline + 1, h - 2, vline + 1, w - 2


def _place_fragment(
    grid: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    color: int,
    bounds: tuple[int, int, int, int],
    rng: random.Random,
) -> bool:
    h, w = mask.shape
    rlo, rhi, clo, chi = bounds
    if rhi - rlo + 1 < h or chi - clo + 1 < w:
        return False
    for _ in range(250):
        r0 = rng.randint(rlo, rhi - h + 1)
        c0 = rng.randint(clo, chi - w + 1)
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(grid.shape[0], r0 + h + 1)
        cc1 = min(grid.shape[1], c0 + w + 1)
        if bool(blocked[rr0:rr1, cc0:cc1].any()):
            continue
        grid[r0 : r0 + h, c0 : c0 + w][mask] = int(color)
        _mark_rect(blocked, r0, c0, r0 + h - 1, c0 + w - 1, pad=1)
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    counts = _sample_counts(rng)
    total = sum(counts)
    holed_n = rng.randint(max(1, total - 3), min(3, total - 1))
    zero_n = total - holed_n
    if not (1 <= zero_n <= 3 and 1 <= holed_n <= 3):
        return None

    inner = 4
    top_width = zero_n * inner + zero_n + 1
    bottom_width = holed_n * inner + holed_n + 1
    block_width = max(top_width, bottom_width)
    block_height = 2 * (inner + 2) + 1

    h = rng.randint(29, 30)
    w = rng.randint(29, 30)
    block_left = w - block_width
    block_top = h - block_height
    if block_left < 11 or block_top < 15:
        return None
    vline_max = block_left - 2
    if vline_max < 8:
        return None
    vline = rng.randint(8, min(12, vline_max))
    hline = rng.randint(10, max(10, block_top - 5))
    if hline >= block_top - 3:
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[hline, :] = GREY
    grid[:, vline] = GREY

    bottom_r0 = h - (inner + 2)
    top_r0 = bottom_r0 - (inner + 2) - 1
    top_c0 = w - top_width
    bottom_c0 = w - bottom_width
    top_slots = _draw_slot_strip(grid, top_r0, top_c0, zero_n, inner)
    bottom_slots = _draw_slot_strip(grid, bottom_r0, bottom_c0, holed_n, inner)

    blocked = grid != BG
    for r0, c0, r1, c1 in top_slots + bottom_slots:
        _mark_rect(blocked, r0, c0, r1, c1, pad=1)

    flags = [0] * zero_n + [rng.choice((1, 2)) for _ in range(holed_n)]
    rng.shuffle(flags)
    colors = list(COLORS)
    rng.shuffle(colors)
    assert total <= len(colors)

    idx = 0
    for quadrant, count in enumerate(counts):
        bounds = _quadrant_bounds(quadrant, hline, vline, h, w)
        for _ in range(count):
            holes = flags[idx]
            color = colors[idx]
            idx += 1
            piece = _zero_hole_mask(rng) if holes == 0 else _holed_mask(holes, rng)
            if not _place_fragment(grid, blocked, piece, color, bounds, rng):
                return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
