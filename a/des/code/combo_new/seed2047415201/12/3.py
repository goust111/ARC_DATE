from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 8
GREY = 5
SIDES = ("up", "down", "left", "right")


def _most_common(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 8:
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        cells: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in nbrs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _crop_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0)


def _l_base(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[0, :] = True
    m[:, w - 1] = True
    return m


def _t_base(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[0, :] = True
    m[:, w // 2] = True
    return m


def _candidate_bases() -> list[tuple[str, np.ndarray]]:
    bases: list[tuple[str, np.ndarray]] = []
    for h in (3, 4):
        for w in (2, 3):
            bases.append(("L", _l_base(h, w)))
    for h in (3, 4):
        for w in (3, 5):
            bases.append(("T", _t_base(h, w)))
    return bases


def _classify_key(mask: np.ndarray) -> tuple[str, np.ndarray]:
    for kind, base in _candidate_bases():
        for k in range(4):
            if np.array_equal(mask, np.rot90(base, k)):
                return kind, base
    raise AssertionError("key must be a generated L or T shape")


def _target_mask(base: np.ndarray, opening: str) -> np.ndarray:
    # Base masks have a one-cell plug on the bottom edge. Rotate that plug
    # toward the frame side containing the lockhole.
    rotations = {"up": 0, "down": 2, "left": 1, "right": 3}
    return np.rot90(base, rotations[opening])


def _plug_cell(mask: np.ndarray, opening: str) -> tuple[int, int]:
    if opening == "up":
        row = mask.shape[0] - 1
        cols = np.where(mask[row, :])[0]
        assert len(cols) == 1
        return int(row), int(cols[0])
    if opening == "down":
        cols = np.where(mask[0, :])[0]
        assert len(cols) == 1
        return 0, int(cols[0])
    if opening == "left":
        col = mask.shape[1] - 1
        rows = np.where(mask[:, col])[0]
        assert len(rows) == 1
        return int(rows[0]), int(col)
    col = 0
    rows = np.where(mask[:, col])[0]
    assert len(rows) == 1
    return int(rows[0]), int(col)


def _paste_mask(out: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    h, w = mask.shape
    sub = out[r0 : r0 + h, c0 : c0 + w]
    sub[mask] = int(color)


def _find_locks(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    locks: list[dict] = []
    seen: set[tuple[int, int, int, int]] = set()
    for r0 in range(h):
        for r1 in range(r0 + 4, h):
            for c0 in range(w):
                for c1 in range(c0 + 4, w):
                    if (r0, c0, r1, c1) in seen:
                        continue
                    if not (
                        g[r0, c0] == GREY
                        and g[r0, c1] == GREY
                        and g[r1, c0] == GREY
                        and g[r1, c1] == GREY
                    ):
                        continue
                    if not bool((g[r0 + 1 : r1, c0 + 1 : c1] == bg).all()):
                        continue

                    odd: list[tuple[str, int, int, int]] = []
                    ok = True
                    for c in range(c0 + 1, c1):
                        for side, r in (("up", r0), ("down", r1)):
                            val = int(g[r, c])
                            if val == GREY:
                                continue
                            if val == bg:
                                ok = False
                                break
                            odd.append((side, int(r), int(c), val))
                        if not ok:
                            break
                    if not ok:
                        continue
                    for r in range(r0 + 1, r1):
                        for side, c in (("left", c0), ("right", c1)):
                            val = int(g[r, c])
                            if val == GREY:
                                continue
                            if val == bg:
                                ok = False
                                break
                            odd.append((side, int(r), int(c), val))
                        if not ok:
                            break
                    if not ok or len(odd) != 1:
                        continue

                    side, gr, gc, color = odd[0]
                    if color in (bg, GREY):
                        continue
                    locks.append(
                        {
                            "bbox": (int(r0), int(c0), int(r1), int(c1)),
                            "gap": (int(gr), int(gc)),
                            "side": side,
                            "color": int(color),
                        }
                    )
                    seen.add((r0, c0, r1, c1))
    locks.sort(key=lambda d: (d["bbox"][0], d["bbox"][1]))
    colors = [int(d["color"]) for d in locks]
    assert len(colors) == len(set(colors)), "lockhole colors must be unique"
    return locks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert each colored key into the matching colored notch of a grey lock frame.

    Input structure:
    - The background is Azure and the locks are hollow Grey rectangular frames.
    - Each frame has exactly one non-corner side cell recolored as a unique lockhole color.
    - Away from the frames there is one same-colored small L-shaped or T-shaped key for each lockhole.

    Rule:
    - Match each key to the lockhole with the same color.
    - Read the lockhole side as the opening direction: top, bottom, left, or right.
    - Clear every key from its original location, rotate its L/T shape so its one-cell plug faces the matching side, and translate that plug onto the colored lockhole cell.
    - Leave the Grey frame borders, the colored lockhole cells, and the Azure background otherwise unchanged.

    Key insight:
    - Color gives the key-lock pairing, while the side of the colored notch supplies the rotation needed before translation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _most_common(g)
    locks = _find_locks(g, bg)
    assert locks, "at least one lock is required"

    gap_cells = {(int(d["gap"][0]), int(d["gap"][1])) for d in locks}
    out = g.copy()
    placements: list[tuple[np.ndarray, int, int, int]] = []

    for lock in locks:
        color = int(lock["color"])
        color_mask = g == color
        for r, c in gap_cells:
            if g[r, c] == color:
                color_mask[r, c] = False
        comps = _components(color_mask, connectivity=4)
        comps = [c for c in comps if int(c.shape[0]) >= 4]
        assert len(comps) == 1, "each lockhole color must have one separate key"

        for r, c in comps[0]:
            out[int(r), int(c)] = bg

        key_mask, _origin = _crop_mask(comps[0])
        _kind, base = _classify_key(key_mask)
        target = _target_mask(base, str(lock["side"]))
        pr, pc = _plug_cell(target, str(lock["side"]))
        gr, gc = map(int, lock["gap"])
        tr, tc = int(gr - pr), int(gc - pc)
        assert 0 <= tr and tr + target.shape[0] <= h
        assert 0 <= tc and tc + target.shape[1] <= w
        placements.append((target, color, tr, tc))

    for target, color, tr, tc in placements:
        _paste_mask(out, target, color, tr, tc)
    return out


def _mask_cells(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(mask)]


def _touches_reserved(reserved: np.ndarray, cells: list[tuple[int, int]], pad: int) -> bool:
    h, w = reserved.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return True
        r0, r1 = max(0, r - pad), min(h, r + pad + 1)
        c0, c1 = max(0, c - pad), min(w, c + pad + 1)
        if bool(reserved[r0:r1, c0:c1].any()):
            return True
    return False


def _frame_cells(r0: int, c0: int, fh: int, fw: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    r1, c1 = r0 + fh - 1, c0 + fw - 1
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _gap_for(r0: int, c0: int, fh: int, fw: int, side: str, rng: random.Random) -> tuple[int, int]:
    r1, c1 = r0 + fh - 1, c0 + fw - 1
    if side == "up":
        return r0, rng.randint(c0 + 1, c1 - 1)
    if side == "down":
        return r1, rng.randint(c0 + 1, c1 - 1)
    if side == "left":
        return rng.randint(r0 + 1, r1 - 1), c0
    return rng.randint(r0 + 1, r1 - 1), c1


def _draw_frame(grid: np.ndarray, r0: int, c0: int, fh: int, fw: int, gap: tuple[int, int], color: int) -> None:
    r1, c1 = r0 + fh - 1, c0 + fw - 1
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[int(gap[0]), int(gap[1])] = int(color)


def _place_lock(
    grid: np.ndarray,
    reserved: np.ndarray,
    side: str,
    color: int,
    base: np.ndarray,
    rng: random.Random,
) -> dict | None:
    h, w = grid.shape
    target = _target_mask(base, side)
    for _ in range(300):
        fh = rng.randint(5, 8)
        fw = rng.randint(5, 8)
        r0 = rng.randint(0, h - fh)
        c0 = rng.randint(0, w - fw)
        gap = _gap_for(r0, c0, fh, fw, side, rng)
        pr, pc = _plug_cell(target, side)
        tr, tc = int(gap[0] - pr), int(gap[1] - pc)
        key_cells = _mask_cells(target, tr, tc)
        frame = _frame_cells(r0, c0, fh, fw)
        frame_list = sorted(frame)

        if _touches_reserved(reserved, frame_list + key_cells, pad=1):
            continue
        if any(cell in frame and cell != gap for cell in key_cells):
            continue

        _draw_frame(grid, r0, c0, fh, fw, gap, color)
        for r, c in set(frame_list).union(key_cells):
            reserved[r, c] = True
        return {"color": int(color), "base": base, "side": side}
    return None


def _place_key(
    grid: np.ndarray,
    reserved: np.ndarray,
    base: np.ndarray,
    color: int,
    rng: random.Random,
) -> bool:
    mask = np.rot90(base, rng.randint(0, 3))
    h, w = grid.shape
    mh, mw = mask.shape
    for _ in range(300):
        r0 = rng.randint(0, h - mh)
        c0 = rng.randint(0, w - mw)
        cells = _mask_cells(mask, r0, c0)
        if _touches_reserved(reserved, cells, pad=1):
            continue
        for r, c in cells:
            grid[r, c] = int(color)
            reserved[r, c] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(22, 30)
    w = rng.randint(22, 30)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    place_reserved = np.zeros((h, w), dtype=bool)

    n_locks = rng.randint(3, 4)
    dirs = list(SIDES)
    rng.shuffle(dirs)
    dirs = dirs[:n_locks]

    colors = [c for c in range(10) if c not in (BG, GREY)]
    rng.shuffle(colors)
    colors = colors[:n_locks]

    bases_by_kind = {"L": [], "T": []}
    for kind, base in _candidate_bases():
        bases_by_kind[kind].append(base)

    kinds = ["L", "T"] + [rng.choice(["L", "T"]) for _ in range(n_locks - 2)]
    rng.shuffle(kinds)

    locks: list[dict] = []
    for side, color, kind in zip(dirs, colors, kinds):
        base = rng.choice(bases_by_kind[kind])
        lock = _place_lock(grid, place_reserved, side, color, base, rng)
        if lock is None:
            return None
        locks.append(lock)

    reserved |= place_reserved
    rng.shuffle(locks)
    for lock in locks:
        if not _place_key(grid, reserved, lock["base"], int(lock["color"]), rng):
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-and-key sample")
