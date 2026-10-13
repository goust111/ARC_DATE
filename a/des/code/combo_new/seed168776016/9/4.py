import random
from collections import deque

import numpy as np


BG = 0
STAMP = 9
PATTERN_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]
GEN_RETRIES = 400


def _bbox_non_bg(arr: np.ndarray, bg: int = BG) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(arr != int(bg))
    if pts.size == 0:
        return None
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _stamp_mask(crop: np.ndarray) -> np.ndarray:
    raw = crop == STAMP
    if not bool(raw.any()):
        return raw

    detected = np.zeros_like(raw, dtype=bool)
    for comp in _components4(raw):
        lo = comp.min(axis=0)
        hi = comp.max(axis=0)
        r0, c0 = int(lo[0]), int(lo[1])
        r1, c1 = int(hi[0]), int(hi[1])
        hh = int(r1 - r0 + 1)
        ww = int(c1 - c0 + 1)
        if 3 <= hh <= 5 and 3 <= ww <= 5 and int(comp.shape[0]) == hh * ww:
            detected[r0 : r1 + 1, c0 : c1 + 1] = True

    return detected if bool(detected.any()) else raw


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a maroon-occluded central pattern by using its 180-degree rotational symmetry.

    Input structure:
    - Background is black.
    - A large multicolor pattern sits away from the canvas border and is symmetric under a 180-degree rotation about its center.
    - One or more solid maroon 3x3 to 5x5 rectangles lie within the pattern and hide parts of it.

    Rule:
    - Find the bounding box of the non-black pattern area and crop to that box.
    - Detect the maroon rectangular stamp cells inside the cropped pattern.
    - For each stamped cell, copy the color from its 180-degree opposite position inside the cropped box.
    - Return the restored cropped pattern only.

    Key insight:
    - Maroon cells are not part of the true pattern; they are removable occluders, and each hidden cell can be recovered from the point-symmetric cell across the pattern center.
    """
    g = np.asarray(grid, dtype=int)
    box = _bbox_non_bg(g, BG)
    if box is None:
        return g.copy()

    r0, c0, r1, c1 = box
    crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    h, w = crop.shape
    stamps = _stamp_mask(crop)
    out = crop.copy()

    rows, cols = np.where(stamps)
    for r, c in zip(rows.tolist(), cols.tolist()):
        rr = h - 1 - int(r)
        cc = w - 1 - int(c)
        src = int(crop[rr, cc])
        if src != STAMP:
            out[int(r), int(c)] = src

    unresolved = out == STAMP
    if bool(unresolved.any()):
        rows, cols = np.where(unresolved)
        for r, c in zip(rows.tolist(), cols.tolist()):
            rr = h - 1 - int(r)
            cc = w - 1 - int(c)
            src = int(out[rr, cc])
            if src != STAMP:
                out[int(r), int(c)] = src

    return out


def _random_dim(rng: random.Random, lo: int, hi: int) -> int:
    parity = rng.randrange(2)
    choices = [n for n in range(lo, hi + 1) if n % 2 == parity]
    return int(rng.choice(choices))


def _centered_span(rng: random.Random, total: int, lo: int, hi: int) -> tuple[int, int]:
    lo = max(1, int(lo))
    hi = min(int(total), int(hi))
    if lo > hi:
        lo = hi = int(total)
    length = rng.randint(lo, hi)
    if length % 2 != total % 2:
        if length < hi:
            length += 1
        else:
            length -= 1
    length = max(1, min(total, int(length)))
    if length % 2 != total % 2:
        if length < total:
            length += 1
        else:
            length -= 1
    start = (total - length) // 2
    return int(start), int(start + length)


def _paint_rot_rect(arr: np.ndarray, r0: int, c0: int, hh: int, ww: int, color: int) -> None:
    h, w = arr.shape
    rr0 = h - int(r0) - int(hh)
    cc0 = w - int(c0) - int(ww)
    arr[int(r0) : int(r0 + hh), int(c0) : int(c0 + ww)] = int(color)
    arr[int(rr0) : int(rr0 + hh), int(cc0) : int(cc0 + ww)] = int(color)


def _make_pattern(rng: random.Random) -> np.ndarray | None:
    h = _random_dim(rng, 9, 21)
    w = _random_dim(rng, 9, 21)
    palette = rng.sample(PATTERN_COLORS, rng.randint(3, 5))
    base = int(palette[0])
    accents = [int(c) for c in palette[1:]]
    pat = np.full((h, w), base, dtype=int)

    border_color = int(rng.choice(accents))
    thickness = rng.randint(1, max(1, min(2, (min(h, w) - 1) // 4)))
    pat[:thickness, :] = border_color
    pat[-thickness:, :] = border_color
    pat[:, :thickness] = border_color
    pat[:, -thickness:] = border_color

    r0, r1 = _centered_span(rng, h, max(1, h // 5), max(2, int(round(h * 0.7))))
    c0, c1 = _centered_span(rng, w, max(1, w // 5), max(2, int(round(w * 0.7))))
    pat[r0:r1, c0:c1] = int(rng.choice(accents))

    if rng.random() < 0.8:
        c0, c1 = _centered_span(rng, w, 1, max(1, w // 3))
        pat[:, c0:c1] = int(rng.choice(accents))
    if rng.random() < 0.8:
        r0, r1 = _centered_span(rng, h, 1, max(1, h // 3))
        pat[r0:r1, :] = int(rng.choice(accents))

    for _ in range(rng.randint(6, 11)):
        hh = rng.randint(1, min(4, h))
        ww = rng.randint(1, min(4, w))
        r0 = rng.randint(0, h - hh)
        c0 = rng.randint(0, w - ww)
        _paint_rot_rect(pat, r0, c0, hh, ww, int(rng.choice(accents)))

    used, counts = np.unique(pat, return_counts=True)
    if len(used) < 3:
        return None
    if int(counts.max()) > int(0.82 * h * w):
        return None
    if int(np.count_nonzero(pat != base)) < int(0.18 * h * w):
        return None
    if not bool(np.array_equal(pat, np.rot90(pat, 2))):
        return None
    return pat


def _boxes_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    ar1 = ar0 + ah - 1
    ac1 = ac0 + aw - 1
    br1 = br0 + bh - 1
    bc1 = bc0 + bw - 1
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _mirror_rect(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, hh, ww = rect
    return int(h - r0 - hh), int(w - c0 - ww), int(hh), int(ww)


def _apply_stamps(rng: random.Random, clean: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    h, w = clean.shape
    if min(h, w) < 5:
        return None

    area = h * w
    max_stamps = min(4, max(2, area // 70 + 1))
    want = rng.randint(1, max_stamps)
    rects: list[tuple[int, int, int, int]] = []
    stamp_mask = np.zeros((h, w), dtype=bool)

    for _ in range(want):
        placed = False
        for _ in range(250):
            hh = rng.randint(3, min(5, h))
            ww = rng.randint(3, min(5, w))
            rect = (
                rng.randint(0, h - hh),
                rng.randint(0, w - ww),
                int(hh),
                int(ww),
            )
            mirror = _mirror_rect((h, w), rect)
            if _boxes_touch(rect, mirror, pad=0):
                continue
            if any(_boxes_touch(rect, old, pad=1) for old in rects):
                continue
            r0, c0, _, _ = rect
            mr0, mc0, _, _ = mirror
            if bool(stamp_mask[r0 : r0 + hh, c0 : c0 + ww].any()):
                continue
            if bool(stamp_mask[mr0 : mr0 + hh, mc0 : mc0 + ww].any()):
                continue
            stamp_mask[r0 : r0 + hh, c0 : c0 + ww] = True
            rects.append(rect)
            placed = True
            break
        if not placed:
            return None

    stamped = clean.copy()
    stamped[stamp_mask] = STAMP
    return stamped, rects


def _embed_on_canvas(rng: random.Random, patch: np.ndarray) -> np.ndarray:
    h, w = patch.shape
    extra_h = rng.randint(2, min(9, 30 - h))
    extra_w = rng.randint(2, min(9, 30 - w))
    top = rng.randint(1, extra_h - 1)
    left = rng.randint(1, extra_w - 1)
    canvas = np.zeros((h + extra_h, w + extra_w), dtype=int)
    canvas[top : top + h, left : left + w] = patch
    return canvas


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    clean = _make_pattern(rng)
    if clean is None:
        return None
    stamped = _apply_stamps(rng, clean)
    if stamped is None:
        return None
    inp = _embed_on_canvas(rng, stamped[0])
    try:
        out = main(inp)
    except Exception:
        return None
    if not bool(np.array_equal(out, clean)):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
