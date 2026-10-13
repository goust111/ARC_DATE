from __future__ import annotations

from collections import deque
import random

import numpy as np


def _zoom(sprite: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbor integer pixel upscale, matching arc_utils.scale_sprite."""
    k = int(factor)
    assert k > 0
    return np.repeat(np.repeat(np.asarray(sprite, dtype=int), k, axis=0), k, axis=1)


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    mask = grid == int(color)
    seen = np.zeros((h, w), dtype=bool)
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
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _ring_mask(shape: tuple[int, int], thickness: int) -> np.ndarray:
    h, w = map(int, shape)
    t = int(thickness)
    rr, cc = np.indices((h, w))
    return (rr < t) | (rr >= h - t) | (cc < t) | (cc >= w - t)


def _frame_from_component(grid: np.ndarray, color: int, coords: np.ndarray) -> dict | None:
    if coords.shape[0] < 16:
        return None

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True

    matches = []
    for t in range(2, (min(h, w) - 1) // 2 + 1):
        inner_h = h - 2 * t
        inner_w = w - 2 * t
        if inner_h <= 0 or inner_w <= 0:
            continue
        if inner_h % t != 0 or inner_w % t != 0:
            continue

        sample_h = inner_h // t
        sample_w = inner_w // t
        if (inner_h - sample_h) % 2 != 0 or (inner_w - sample_w) % 2 != 0:
            continue

        expected = _ring_mask((h, w), t)
        if int(local.sum()) == int(expected.sum()) and np.array_equal(local, expected):
            matches.append(t)

    if len(matches) != 1:
        return None

    t = int(matches[0])
    return {"r0": r0, "c0": c0, "r1": r1, "c1": c1, "t": t, "color": int(color)}


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(map(int, np.unique(grid))):
        for coords in _color_components(grid, color):
            frame = _frame_from_component(grid, color, coords)
            if frame is not None:
                frames.append(frame)

    frames.sort(key=lambda f: (f["r0"], f["c0"], f["r1"], f["c1"], f["color"]))
    assert 1 <= len(frames) <= 2, "expected one or two rectangular thick frames"
    if len(frames) == 2:
        assert frames[0]["t"] != frames[1]["t"], "two frames must have different thicknesses"
    return frames


def _interior_and_sample(frame: dict) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    r0, c0, r1, c1, t = (int(frame[k]) for k in ("r0", "c0", "r1", "c1", "t"))
    ir0, ic0, ir1, ic1 = r0 + t, c0 + t, r1 - t, c1 - t
    inner_h = ir1 - ir0 + 1
    inner_w = ic1 - ic0 + 1
    assert inner_h > 0 and inner_w > 0
    assert inner_h % t == 0 and inner_w % t == 0

    sample_h = inner_h // t
    sample_w = inner_w // t
    assert (inner_h - sample_h) % 2 == 0 and (inner_w - sample_w) % 2 == 0
    sr0 = ir0 + (inner_h - sample_h) // 2
    sc0 = ic0 + (inner_w - sample_w) // 2
    return (ir0, ic0, ir1, ic1), (sr0, sc0, sr0 + sample_h - 1, sc0 + sample_w - 1)


def _apply_frames(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for frame in frames:
        (ir0, ic0, ir1, ic1), (sr0, sc0, sr1, sc1) = _interior_and_sample(frame)
        sample = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
        enlarged = _zoom(sample, int(frame["t"]))
        assert enlarged.shape == (ir1 - ir0 + 1, ic1 - ic0 + 1)
        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = enlarged
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each thick rectangular frame's interior by a zoomed copy of its centered inner sample.

    Input structure:
    - The whole grid is a multicolor pixel scene.
    - One or two hollow monochrome rectangular frames are overlaid on the scene.
    - A frame's uniform ring thickness t is greater than one; if two frames are present, their thicknesses differ.
    - Each frame's interior height and width are divisible by that frame's thickness.

    Rule:
    - Preserve every frame ring and every pixel outside all frames.
    - For each frame, take the centered sub-rectangle inside its interior whose height and width are 1/t of the interior.
    - Read that sub-rectangle from the original input and pixel-upscale it by factor t.
    - Replace exactly the whole frame interior with the enlarged centered sample.

    Key insight:
    - The border thickness is the zoom factor, and the source image is the visible concentric sample at the frame center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _apply_frames(g, _find_frames(g))


# --------------- generator ---------------

_GEN_RETRIES = 500
_GEN_CALL_COUNT = 0


def _valid_inner_lengths(t: int, max_inner: int) -> list[int]:
    vals = []
    for n in range(int(t), int(max_inner) + 1):
        if n % int(t) != 0:
            continue
        sample = n // int(t)
        if sample < 2:
            continue
        if (n - sample) % 2 != 0:
            continue
        vals.append(n)
    return vals


def _sample_frame_size(t: int, grid_h: int, grid_w: int, frame_count: int) -> tuple[int, int] | None:
    if frame_count == 2:
        max_outer_h = min(grid_h - 4, 15 if t == 3 else 12)
        max_outer_w = min(grid_w - 4, 15 if t == 3 else 12)
    else:
        max_outer_h = min(grid_h - 4, 24)
        max_outer_w = min(grid_w - 4, 24)

    hs = _valid_inner_lengths(t, max_outer_h - 2 * t)
    ws = _valid_inner_lengths(t, max_outer_w - 2 * t)
    if not hs or not ws:
        return None
    return int(random.choice(hs) + 2 * t), int(random.choice(ws) + 2 * t)


def _separated(bb: tuple[int, int, int, int], bboxes: list[tuple[int, int, int, int]], gap: int) -> bool:
    r0, c0, r1, c1 = bb
    for ar0, ac0, ar1, ac1 in bboxes:
        if not (r1 + gap < ar0 or ar1 + gap < r0 or c1 + gap < ac0 or ac1 + gap < c0):
            return False
    return True


def _place_frame_specs(
    grid_h: int,
    grid_w: int,
    specs: list[dict],
) -> list[dict] | None:
    placed: list[dict] = []
    bboxes: list[tuple[int, int, int, int]] = []
    for spec in sorted(specs, key=lambda x: x["outer_h"] * x["outer_w"], reverse=True):
        oh, ow = int(spec["outer_h"]), int(spec["outer_w"])
        candidates = []
        for r0 in range(1, grid_h - oh):
            for c0 in range(1, grid_w - ow):
                bb = (r0, c0, r0 + oh - 1, c0 + ow - 1)
                if _separated(bb, bboxes, gap=1):
                    candidates.append((r0, c0, bb))
        if not candidates:
            return None
        r0, c0, bb = random.choice(candidates)
        item = dict(spec)
        item.update({"r0": r0, "c0": c0, "r1": bb[2], "c1": bb[3]})
        placed.append(item)
        bboxes.append(bb)
    placed.sort(key=lambda f: (f["r0"], f["c0"]))
    return placed


def _build_scene(height: int, width: int, palette: list[int]) -> np.ndarray:
    k = len(palette)
    row_bias = [random.randrange(k) for _ in range(height)]
    col_bias = [random.randrange(k) for _ in range(width)]
    a = random.randrange(1, k + 1)
    b = random.randrange(1, k + 1)

    grid = np.empty((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            idx = (a * r + b * c + row_bias[r] + col_bias[c] + (r // 2) + 2 * (c // 3)) % k
            grid[r, c] = int(palette[idx])

    for _ in range(random.randint(height * width // 12, height * width // 6)):
        r = random.randrange(height)
        c = random.randrange(width)
        grid[r, c] = int(random.choice(palette))
    return grid


def _paint_frame(grid: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = (int(frame[k]) for k in ("r0", "c0", "r1", "c1"))
    t = int(frame["t"])
    color = int(frame["color"])
    grid[r0 : r0 + t, c0 : c1 + 1] = color
    grid[r1 - t + 1 : r1 + 1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0 : c0 + t] = color
    grid[r0 : r1 + 1, c1 - t + 1 : c1 + 1] = color


def _sample_mode() -> str:
    global _GEN_CALL_COUNT
    if _GEN_CALL_COUNT == 0:
        mode = "both"
    elif _GEN_CALL_COUNT == 1:
        mode = "single2"
    elif _GEN_CALL_COUNT == 2:
        mode = "single3"
    else:
        mode = random.choices(["both", "single2", "single3"], weights=[5, 2, 2], k=1)[0]
    _GEN_CALL_COUNT += 1
    return mode


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, frames: list[dict]) -> bool:
    try:
        found = _find_frames(inp)
    except AssertionError:
        return False

    key = lambda f: (int(f["r0"]), int(f["c0"]), int(f["r1"]), int(f["c1"]), int(f["t"]), int(f["color"]))
    if sorted(map(key, found)) != sorted(map(key, frames)):
        return False

    if not np.array_equal(_apply_frames(inp, frames), expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    for frame in frames:
        (ir0, ic0, ir1, ic1), (sr0, sc0, sr1, sc1) = _interior_and_sample(frame)
        t = int(frame["t"])
        interior_in = inp[ir0 : ir1 + 1, ic0 : ic1 + 1]
        interior_out = expected[ir0 : ir1 + 1, ic0 : ic1 + 1]
        sample = inp[sr0 : sr1 + 1, sc0 : sc1 + 1]
        if np.array_equal(interior_in, interior_out):
            return False
        if not np.array_equal(_zoom(sample, t), interior_out):
            return False

        inner_h, inner_w = interior_in.shape
        sample_h, sample_w = sample.shape
        alt_starts = {
            (ir0, ic0),
            (ir0, ic1 - sample_w + 1),
            (ir1 - sample_h + 1, ic0),
            (ir1 - sample_h + 1, ic1 - sample_w + 1),
        }
        alt_starts.discard((sr0, sc0))
        if not alt_starts:
            return False
        for ar0, ac0 in alt_starts:
            alt = inp[ar0 : ar0 + sample_h, ac0 : ac0 + sample_w]
            if np.array_equal(_zoom(alt, t), interior_out):
                return False

        for wrong_t in (2, 3):
            if wrong_t == t:
                continue
            if inner_h % wrong_t != 0 or inner_w % wrong_t != 0:
                continue
            wh, ww = inner_h // wrong_t, inner_w // wrong_t
            if (inner_h - wh) % 2 != 0 or (inner_w - ww) % 2 != 0:
                continue
            wr0 = ir0 + (inner_h - wh) // 2
            wc0 = ic0 + (inner_w - ww) // 2
            wrong = _zoom(inp[wr0 : wr0 + wh, wc0 : wc0 + ww], wrong_t)
            if wrong.shape == interior_out.shape and np.array_equal(wrong, interior_out):
                return False
    return True


def _gen_attempt(mode: str) -> np.ndarray | None:
    if mode == "both":
        height = random.randint(24, 30)
        width = random.randint(24, 30)
        thicknesses = [2, 3]
        random.shuffle(thicknesses)
    else:
        height = random.randint(18, 30)
        width = random.randint(18, 30)
        thicknesses = [2 if mode == "single2" else 3]

    specs = []
    for t in thicknesses:
        size = _sample_frame_size(t, height, width, len(thicknesses))
        if size is None:
            return None
        outer_h, outer_w = size
        specs.append({"t": int(t), "outer_h": outer_h, "outer_w": outer_w})

    frames = _place_frame_specs(height, width, specs)
    if frames is None:
        return None

    colors = list(range(10))
    random.shuffle(colors)
    frame_colors = colors[: len(frames)]
    scene_palette = colors[len(frames) : len(frames) + random.randint(4, min(6, 10 - len(frames)))]
    if len(scene_palette) < 4:
        return None

    scene = _build_scene(height, width, scene_palette)
    inp = scene.copy()
    for frame, color in zip(frames, frame_colors):
        frame["color"] = int(color)
        _paint_frame(inp, frame)

    expected = _apply_frames(inp, frames)
    if not _semantic_checks(inp, expected, frames):
        return None
    return inp


def generate_input() -> np.ndarray:
    mode = _sample_mode()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid zoom-lens input")
