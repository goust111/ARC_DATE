from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FRAME_COLOR = 5
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("left", "right", "top", "bottom")
OPPOSITE_SIDE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
STAMP_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _count_inner_zero_components(patch: np.ndarray) -> int:
    zero = patch == BG
    count = 0
    for comp in _components(zero):
        if all(0 < r < patch.shape[0] - 1 and 0 < c < patch.shape[1] - 1 for r, c in comp):
            count += 1
    return count


def _extract_frame_info(grid: np.ndarray, color: int, pts: list[tuple[int, int]]) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(pts)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    if h < 3 or w < 3:
        return None
    if np.any((patch != BG) & (patch != color)):
        return None
    if np.any(patch[1:-1, 1:-1] != BG):
        return None
    if not (
        patch[0, 0] == color
        and patch[0, -1] == color
        and patch[-1, 0] == color
        and patch[-1, -1] == color
    ):
        return None

    top_missing = [c for c in range(1, w - 1) if patch[0, c] == BG]
    bottom_missing = [c for c in range(1, w - 1) if patch[-1, c] == BG]
    left_missing = [r for r in range(1, h - 1) if patch[r, 0] == BG]
    right_missing = [r for r in range(1, h - 1) if patch[r, -1] == BG]
    counts = {
        "top": len(top_missing),
        "bottom": len(bottom_missing),
        "left": len(left_missing),
        "right": len(right_missing),
    }
    sides = [side for side, count in counts.items() if count > 0]
    if len(sides) != 1:
        return None
    side = sides[0]

    if side != "top" and not np.all(patch[0, 1:-1] == color):
        return None
    if side != "bottom" and not np.all(patch[-1, 1:-1] == color):
        return None
    if side != "left" and not np.all(patch[1:-1, 0] == color):
        return None
    if side != "right" and not np.all(patch[1:-1, -1] == color):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "gap_count": counts[side],
    }


def _find_frames(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    colors = [int(c) for c in np.unique(grid) if int(c) != BG]
    matches: list[tuple[int, list[dict[str, object]]]] = []
    for color in colors:
        infos: list[dict[str, object]] = []
        for comp in _components(grid == color):
            info = _extract_frame_info(grid, color, comp)
            if info is None:
                infos = []
                break
            infos.append(info)
        if infos:
            matches.append((color, infos))
    assert len(matches) == 1, "expected exactly one frame color"
    return matches[0]


def _extract_stamp_info(grid: np.ndarray, color: int, pts: list[tuple[int, int]]) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(pts)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    if h < 3 or w < 3:
        return None
    if np.any((patch != BG) & (patch != color)):
        return None
    if not (
        np.all(patch[0, :] == color)
        and np.all(patch[-1, :] == color)
        and np.all(patch[:, 0] == color)
        and np.all(patch[:, -1] == color)
    ):
        return None

    holes = _count_inner_zero_components(patch)
    if holes <= 0:
        return None
    return {
        "color": color,
        "patch": patch.copy(),
        "hole_count": holes,
    }


def _find_stamps(grid: np.ndarray, frame_color: int) -> list[dict[str, object]]:
    stamps: list[dict[str, object]] = []
    for color in [int(c) for c in np.unique(grid) if int(c) not in (BG, frame_color)]:
        comps = _components(grid == color)
        if len(comps) != 1:
            continue
        info = _extract_stamp_info(grid, color, comps[0])
        if info is not None:
            stamps.append(info)
    assert stamps, "expected at least one stamp"
    return stamps


def _tile_patch(patch: np.ndarray, cavity_shape: tuple[int, int], side: str) -> np.ndarray:
    sh, sw = patch.shape
    ch, cw = cavity_shape
    out = np.zeros((ch, cw), dtype=int)

    if side in ("left", "right"):
        row_starts = list(range(0, ch, sh))
        if side == "left":
            col_starts = list(range(0, cw, sw))
        else:
            start = cw - ((cw + sw - 1) // sw) * sw
            col_starts = list(range(start, cw, sw))
        for c0 in col_starts:
            for r0 in row_starts:
                rr0 = max(0, r0)
                cc0 = max(0, c0)
                rr1 = min(ch, r0 + sh)
                cc1 = min(cw, c0 + sw)
                out[rr0:rr1, cc0:cc1] = patch[rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0]
    else:
        col_starts = list(range(0, cw, sw))
        if side == "top":
            row_starts = list(range(0, ch, sh))
        else:
            start = ch - ((ch + sh - 1) // sh) * sh
            row_starts = list(range(start, ch, sh))
        for r0 in row_starts:
            for c0 in col_starts:
                rr0 = max(0, r0)
                cc0 = max(0, c0)
                rr1 = min(ch, r0 + sh)
                cc1 = min(cw, c0 + sw)
                out[rr0:rr1, cc0:cc1] = patch[rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each notched frame with the unique stamp having the same number of holes,
    then tile that stamp into the frame cavity from the side indicated by the notch.
    """
    g = np.asarray(grid, dtype=int)
    frame_color, frames = _find_frames(g)
    stamps = _find_stamps(g, frame_color)

    by_holes: dict[int, dict[str, object]] = {}
    for stamp in stamps:
        hole_count = int(stamp["hole_count"])
        assert hole_count not in by_holes, "stamp hole counts must be unique"
        by_holes[hole_count] = stamp

    out = g.copy()
    for stamp in stamps:
        out[g == int(stamp["color"])] = BG

    for frame in frames:
        gap_count = int(frame["gap_count"])
        assert gap_count in by_holes, "every frame must have a matching stamp"
        stamp = by_holes[gap_count]
        r0, c0, r1, c1 = frame["bbox"]
        cavity = _tile_patch(
            np.asarray(stamp["patch"], dtype=int),
            (r1 - r0 - 1, c1 - c0 - 1),
            str(frame["side"]),
        )
        out[r0 + 1 : r1, c0 + 1 : c1] = cavity

    return out


def _choose_separated_cells(
    cells: list[tuple[int, int]], count: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    for _ in range(80):
        order = cells[:]
        rng.shuffle(order)
        chosen: list[tuple[int, int]] = []
        for r, c in order:
            if all(abs(r - rr) > 1 or abs(c - cc) > 1 for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == count:
                    return chosen
    return None


def _make_stamp_patch(hole_count: int, side: str, color: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(120):
        if side in ("left", "right"):
            h = rng.randint(max(4, hole_count + 1), 6)
            w = rng.randint(4, 6)
        else:
            h = rng.randint(4, 6)
            w = rng.randint(max(4, hole_count + 1), 6)
        if (h - 2) * (w - 2) < hole_count:
            continue
        holes = _choose_separated_cells(
            [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)],
            hole_count,
            rng,
        )
        if holes is None:
            continue
        patch = np.full((h, w), color, dtype=int)
        for r, c in holes:
            patch[r, c] = BG
        if np.array_equal(patch, np.fliplr(patch)):
            continue
        if np.array_equal(patch, np.flipud(patch)):
            continue
        return patch
    return None


def _make_frame_patch(
    gap_count: int, side: str, cavity_shape: tuple[int, int], rng: random.Random
) -> np.ndarray:
    ch, cw = cavity_shape
    patch = np.zeros((ch + 2, cw + 2), dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR

    if side == "left":
        gaps = rng.sample(list(range(1, ch + 1)), gap_count)
        patch[gaps, 0] = BG
    elif side == "right":
        gaps = rng.sample(list(range(1, ch + 1)), gap_count)
        patch[gaps, -1] = BG
    elif side == "top":
        gaps = rng.sample(list(range(1, cw + 1)), gap_count)
        patch[0, gaps] = BG
    else:
        gaps = rng.sample(list(range(1, cw + 1)), gap_count)
        patch[-1, gaps] = BG
    return patch


def _direction_effective(stamp: np.ndarray, cavity_shape: tuple[int, int], side: str) -> bool:
    return not np.array_equal(
        _tile_patch(stamp, cavity_shape, side),
        _tile_patch(stamp, cavity_shape, OPPOSITE_SIDE[side]),
    )


def _make_pair(
    hole_count: int, side: str, color: int, rng: random.Random
) -> dict[str, object] | None:
    for _ in range(160):
        stamp = _make_stamp_patch(hole_count, side, color, rng)
        if stamp is None:
            continue
        sh, sw = stamp.shape
        if side in ("left", "right"):
            cavity_h = sh
            width_choices = [w for w in range(sw + 1, min(11, sw * 3) + 1) if w % sw != 0]
            if not width_choices:
                continue
            cavity_w = rng.choice(width_choices)
        else:
            cavity_w = sw
            height_choices = [h for h in range(sh + 1, min(11, sh * 3) + 1) if h % sh != 0]
            if not height_choices:
                continue
            cavity_h = rng.choice(height_choices)

        cavity_shape = (cavity_h, cavity_w)
        if not _direction_effective(stamp, cavity_shape, side):
            continue

        return {
            "count": hole_count,
            "side": side,
            "color": color,
            "stamp": stamp,
            "frame": _make_frame_patch(hole_count, side, cavity_shape, rng),
        }
    return None


def _can_place(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    hh, ww = occ.shape
    if r0 < 1 or c0 < 1 or r0 + h >= hh or c0 + w >= ww:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    return not occ[rr0:rr1, cc0:cc1].any()


def _mark_occ(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occ.shape[0], r0 + h + pad)
    cc1 = min(occ.shape[1], c0 + w + pad)
    occ[rr0:rr1, cc0:cc1] = True


def _paste_nonzero(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    sub = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    mask = patch != BG
    sub[mask] = patch[mask]


def _place_patch(
    grid: np.ndarray, occ: np.ndarray, patch: np.ndarray, rng: random.Random, pad: int
) -> tuple[int, int] | None:
    h, w = patch.shape
    origins = [
        (r, c)
        for r in range(1, grid.shape[0] - h)
        for c in range(1, grid.shape[1] - w)
    ]
    rng.shuffle(origins)
    for r0, c0 in origins:
        if _can_place(occ, r0, c0, h, w, pad):
            _paste_nonzero(grid, patch, r0, c0)
            _mark_occ(occ, r0, c0, h, w, pad)
            return r0, c0
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    n_pairs = rng.randint(2, 4)
    counts = rng.sample([1, 2, 3, 4], n_pairs)
    sides = list(SIDES)
    rng.shuffle(sides)
    sides = sides[:n_pairs]
    colors = rng.sample(STAMP_COLORS, n_pairs)

    pairs: list[dict[str, object]] = []
    for count, side, color in zip(counts, sides, colors):
        pair = _make_pair(count, side, color, rng)
        if pair is None:
            return None
        pairs.append(pair)

    max_h = max(
        max(int(pair["frame"].shape[0]), int(pair["stamp"].shape[0])) for pair in pairs
    )
    max_w = max(
        max(int(pair["frame"].shape[1]), int(pair["stamp"].shape[1])) for pair in pairs
    )
    grid_h = rng.randint(max(22, max_h + 6), 30)
    grid_w = rng.randint(max(22, max_w + 6), 30)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    occ = np.zeros((grid_h, grid_w), dtype=bool)

    for pair in sorted(pairs, key=lambda item: int(np.prod(item["frame"].shape)), reverse=True):
        if _place_patch(grid, occ, np.asarray(pair["frame"], dtype=int), rng, pad=1) is None:
            return None
    for pair in sorted(pairs, key=lambda item: int(np.prod(item["stamp"].shape)), reverse=True):
        if _place_patch(grid, occ, np.asarray(pair["stamp"], dtype=int), rng, pad=1) is None:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
