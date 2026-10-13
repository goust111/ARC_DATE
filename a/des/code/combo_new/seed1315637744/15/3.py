from __future__ import annotations

from collections import Counter, deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return comps


def _gap_run(side: np.ndarray, color: int) -> tuple[int, int] | None:
    missing = side != int(color)
    if not bool(missing.any()):
        return None
    if bool(missing[0]) or bool(missing[-1]):
        return None

    idxs = np.flatnonzero(missing)
    if idxs.size == 0:
        return None
    if int(idxs[-1] - idxs[0] + 1) != int(idxs.size):
        return None

    start = int(idxs[0])
    length = int(idxs.size)
    if not (2 <= length <= 4):
        return None
    return start, length


def _frame_info(grid: np.ndarray, bg: int, comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 5 or w < 5:
        return None

    color = int(comp["color"])
    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)

    if (
        int(patch[0, 0]) != color
        or int(patch[0, -1]) != color
        or int(patch[-1, 0]) != color
        or int(patch[-1, -1]) != color
    ):
        return None

    perimeter = np.zeros((h, w), dtype=bool)
    perimeter[0, :] = True
    perimeter[-1, :] = True
    perimeter[:, 0] = True
    perimeter[:, -1] = True

    if bool(np.any(patch[~perimeter] != bg)):
        return None
    if set(map(int, np.unique(patch[perimeter]))) - {bg, color}:
        return None

    side_runs: list[tuple[str, int, int]] = []
    for name, side in (
        ("top", patch[0, :]),
        ("bottom", patch[-1, :]),
        ("left", patch[:, 0]),
        ("right", patch[:, -1]),
    ):
        run = _gap_run(side, color)
        if run is None:
            if bool(np.any(side != color)):
                return None
            continue
        side_runs.append((name, int(run[0]), int(run[1])))

    if len(side_runs) != 1:
        return None

    side, start, gap_len = side_runs[0]
    gap_coords: list[tuple[int, int]] = []
    if side == "top":
        gap_coords = [(int(r0), int(c0 + cc)) for cc in range(start, start + gap_len)]
        orient = "h"
    elif side == "bottom":
        gap_coords = [(int(r1), int(c0 + cc)) for cc in range(start, start + gap_len)]
        orient = "h"
    elif side == "left":
        gap_coords = [(int(r0 + rr), int(c0)) for rr in range(start, start + gap_len)]
        orient = "v"
    else:
        gap_coords = [(int(r0 + rr), int(c1)) for rr in range(start, start + gap_len)]
        orient = "v"

    return {
        "color": color,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "side": side,
        "orient": orient,
        "gap_len": int(gap_len),
        "gap_coords": gap_coords,
    }


def _key_info(comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    length = max(h, w)
    if min(h, w) != 1 or not (2 <= length <= 4):
        return None
    if int(comp["area"]) != h * w:
        return None
    return {
        "color": int(comp["color"]),
        "coords": list(comp["coords"]),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "orient": "h" if h == 1 else "v",
        "gap_len": int(length),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each frame gap with the uniquely sized outside key, dock the key into the gap, and fill that frame's interior with the key color.

    Input structure:
    - Black background dominates the grid.
    - Several monochrome one-pixel-thick rectangular frames appear, and each frame has exactly one continuous gap on a single side.
    - The inside of every frame is empty background.
    - Outside the frames are small monochrome bar-shaped key pieces whose length matches one frame gap; key colors are distinct from frame colors.

    Rule:
    - Detect each rectangular frame and measure the missing run on its broken side.
    - Find the outside key whose bar length and orientation match that gap.
    - Remove the key from its original location, place it into the frame gap, and then fill every background cell inside that frame with the key color.
    - Keep the original frame border color everywhere except the cells replaced by the inserted key.

    Key insight:
    The outside bars are not decorations: each bar is a movable key identified by gap geometry, and the same key color controls the final fill of the frame interior.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _components(g, bg)

    frames: list[dict] = []
    keys: list[dict] = []
    for comp in comps:
        frame = _frame_info(g, bg, comp)
        if frame is not None:
            frames.append(frame)
            continue
        key = _key_info(comp)
        if key is not None:
            keys.append(key)

    out = g.copy()
    unused_keys = list(keys)

    frames.sort(key=lambda item: (item["gap_len"], item["bbox"]))
    for frame in frames:
        candidates = [
            key
            for key in unused_keys
            if key["gap_len"] == frame["gap_len"] and key["orient"] == frame["orient"]
        ]
        if len(candidates) != 1:
            candidates = [key for key in unused_keys if key["gap_len"] == frame["gap_len"]]
        assert len(candidates) == 1, "each frame gap must have a unique matching key"
        key = candidates[0]
        unused_keys.remove(key)

        for r, c in key["coords"]:
            out[int(r), int(c)] = bg
        for r, c in frame["gap_coords"]:
            out[int(r), int(c)] = int(key["color"])

        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(key["color"])

    return out


CANVAS_SIZE = 30
BG = 0
GAP_LENGTHS = (2, 3, 4)
FRAME_COUNT_CHOICES = (2, 3)
FRAME_SIZE_MIN = 5
FRAME_SIZE_MAX = 10
PLACEMENT_PAD = 1
GEN_ATTEMPTS = 400


def _paint(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    dst[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]


def _frame_pair(
    frame_color: int,
    key_color: int,
    gap_len: int,
    side: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if side in ("top", "bottom"):
        h = int(rng.integers(FRAME_SIZE_MIN, FRAME_SIZE_MAX + 1))
        w_min = max(FRAME_SIZE_MIN, gap_len + 2)
        w = int(rng.integers(w_min, FRAME_SIZE_MAX + 1))
        key_patch = np.full((1, gap_len), key_color, dtype=int)
        gap_start = int(rng.integers(1, w - gap_len))
    else:
        w = int(rng.integers(FRAME_SIZE_MIN, FRAME_SIZE_MAX + 1))
        h_min = max(FRAME_SIZE_MIN, gap_len + 2)
        h = int(rng.integers(h_min, FRAME_SIZE_MAX + 1))
        key_patch = np.full((gap_len, 1), key_color, dtype=int)
        gap_start = int(rng.integers(1, h - gap_len))

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    inp[0, :] = frame_color
    inp[-1, :] = frame_color
    inp[:, 0] = frame_color
    inp[:, -1] = frame_color
    out[:, :] = inp

    if side == "top":
        inp[0, gap_start : gap_start + gap_len] = BG
        out[0, gap_start : gap_start + gap_len] = key_color
    elif side == "bottom":
        inp[-1, gap_start : gap_start + gap_len] = BG
        out[-1, gap_start : gap_start + gap_len] = key_color
    elif side == "left":
        inp[gap_start : gap_start + gap_len, 0] = BG
        out[gap_start : gap_start + gap_len, 0] = key_color
    else:
        inp[gap_start : gap_start + gap_len, -1] = BG
        out[gap_start : gap_start + gap_len, -1] = key_color

    out[1:-1, 1:-1] = key_color
    return inp, out, key_patch


def _free_positions(
    occ: np.ndarray,
    shape: tuple[int, int],
    pad: int,
) -> list[tuple[int, int]]:
    h, w = shape
    H, W = occ.shape
    spots: list[tuple[int, int]] = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            rr0 = max(0, r0 - pad)
            rr1 = min(H, r0 + h + pad)
            cc0 = max(0, c0 - pad)
            cc1 = min(W, c0 + w + pad)
            if not bool(occ[rr0:rr1, cc0:cc1].any()):
                spots.append((int(r0), int(c0)))
    return spots


def _mark_occ(occ: np.ndarray, r0: int, c0: int, shape: tuple[int, int], pad: int) -> None:
    h, w = shape
    H, W = occ.shape
    rr0 = max(0, r0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(W, c0 + w + pad)
    occ[rr0:rr1, cc0:cc1] = True


def _crop_with_margin(
    inp: np.ndarray, out: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    used = np.argwhere(inp != BG)
    r0, c0 = used.min(axis=0)
    r1, c1 = used.max(axis=0)
    top = int(rng.integers(0, min(4, int(r0)) + 1))
    left = int(rng.integers(0, min(4, int(c0)) + 1))
    bottom = int(rng.integers(0, min(4, int(inp.shape[0] - 1 - r1)) + 1))
    right = int(rng.integers(0, min(4, int(inp.shape[1] - 1 - c1)) + 1))
    rr0 = int(r0 - top)
    cc0 = int(c0 - left)
    rr1 = int(r1 + bottom)
    cc1 = int(c1 + right)
    return inp[rr0 : rr1 + 1, cc0 : cc1 + 1], out[rr0 : rr1 + 1, cc0 : cc1 + 1]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_frames = int(rng.choice(FRAME_COUNT_CHOICES))
    gap_lengths = [int(x) for x in rng.choice(np.array(GAP_LENGTHS), size=n_frames, replace=False)]
    rng.shuffle(gap_lengths)
    sides = [str(x) for x in rng.choice(np.array(["top", "bottom", "left", "right"]), size=n_frames, replace=True)]

    colors = [int(x) for x in rng.choice(np.arange(1, 10), size=2 * n_frames, replace=False)]
    frame_colors = colors[:n_frames]
    key_colors = colors[n_frames:]

    inp = np.full((CANVAS_SIZE, CANVAS_SIZE), BG, dtype=int)
    out = np.full((CANVAS_SIZE, CANVAS_SIZE), BG, dtype=int)
    occ = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)

    key_patches: list[np.ndarray] = []
    for frame_color, key_color, gap_len, side in zip(frame_colors, key_colors, gap_lengths, sides):
        frame_in, frame_out, key_patch = _frame_pair(frame_color, key_color, gap_len, side, rng)
        spots = _free_positions(occ, frame_in.shape, PLACEMENT_PAD)
        if not spots:
            return None
        r0, c0 = spots[int(rng.integers(0, len(spots)))]
        _paint(inp, frame_in, r0, c0)
        _paint(out, frame_out, r0, c0)
        _mark_occ(occ, r0, c0, frame_in.shape, PLACEMENT_PAD)
        key_patches.append(key_patch)

    for key_patch in key_patches:
        spots = _free_positions(occ, key_patch.shape, PLACEMENT_PAD)
        if not spots:
            return None
        r0, c0 = spots[int(rng.integers(0, len(spots)))]
        _paint(inp, key_patch, r0, c0)
        _mark_occ(occ, r0, c0, key_patch.shape, PLACEMENT_PAD)

    cropped_inp, cropped_out = _crop_with_margin(inp, out, rng)

    bg = _get_bg(cropped_inp)
    if bg != BG:
        return None
    if max(cropped_inp.shape) > 30 or max(cropped_out.shape) > 30:
        return None
    if np.array_equal(cropped_inp, cropped_out):
        return None
    try:
        produced = main(cropped_inp)
    except Exception:
        return None
    if not np.array_equal(produced, cropped_out):
        return None
    return cropped_inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
