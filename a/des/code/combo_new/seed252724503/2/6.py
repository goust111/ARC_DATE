from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
PALETTE = tuple(range(10))
GENERATE_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _ring_thickness(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> int | None:
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = sub.shape
    for thickness in range(min(3, h // 2, w // 2), 0, -1):
        if h <= 2 * thickness or w <= 2 * thickness:
            continue
        ring = np.zeros((h, w), dtype=bool)
        ring[:thickness, :] = True
        ring[-thickness:, :] = True
        ring[:, :thickness] = True
        ring[:, -thickness:] = True
        if np.array_equal(sub == color, ring):
            return thickness
    return None


def _strictly_contains(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 < br0 and ac0 < bc0 and ar1 > br1 and ac1 > bc1


def _find_frames(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    colors = [int(c) for c in np.unique(grid) if int(c) != BG]
    candidates: list[dict[str, int | tuple[int, int, int, int]]] = []
    for color in colors:
        for comp in _components(grid == color):
            bbox = _bbox(comp)
            thickness = _ring_thickness(grid, bbox, color)
            if thickness is None:
                continue
            candidates.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "thickness": thickness,
                }
            )

    frames = [
        cand
        for i, cand in enumerate(candidates)
        if not any(
            j != i and _strictly_contains(other["bbox"], cand["bbox"])
            for j, other in enumerate(candidates)
        )
    ]
    assert frames, "expected at least one rectangular frame"
    frame_colors = {int(frame["color"]) for frame in frames}
    assert len(frame_colors) == 1, "all outer frames must share one color"
    frames.sort(key=lambda frame: (frame["bbox"][0], frame["bbox"][1]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge each top-left seed inside a rectangular frame by the frame's border thickness.

    Input structure:
    - The grid has a black background.
    - Several non-overlapping rectangular frames share one frame color, but their border thicknesses may differ.
    - Inside each frame, a small colored seed pattern sits in the interior's top-left corner; the rest of the interior is background.

    Rule:
    - Detect every rectangular frame and read its border thickness T.
    - Take the smallest top-left interior rectangle that contains all non-background seed pixels, keeping any background gaps inside that prefix.
    - Scale that prefix by replacing each cell with a T x T block of the same color, then place it back from the interior's top-left corner.
    - If the scaled pattern would extend past the frame interior, clip it at the interior boundary, while leaving the frame border unchanged.

    Key insight:
    - The frame thickness is not decorative; it encodes the magnification factor that must be applied to the seed pattern inside that frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    for frame in _find_frames(g):
        r0, c0, r1, c1 = frame["bbox"]
        thickness = int(frame["thickness"])
        inner = out[r0 + thickness : r1 - thickness + 1, c0 + thickness : c1 - thickness + 1]
        seed_pixels = np.argwhere(inner != BG)
        if len(seed_pixels) == 0:
            continue

        max_r = int(seed_pixels[:, 0].max())
        max_c = int(seed_pixels[:, 1].max())
        seed = inner[: max_r + 1, : max_c + 1].copy()
        scaled = np.kron(seed, np.ones((thickness, thickness), dtype=int))

        clip_h = min(inner.shape[0], scaled.shape[0])
        clip_w = min(inner.shape[1], scaled.shape[1])
        inner[:clip_h, :clip_w] = scaled[:clip_h, :clip_w]

    return out


def _sample_patch_size(
    inner_h: int,
    inner_w: int,
    thickness: int,
    clip: bool,
    rng: random.Random,
) -> tuple[int, int] | None:
    fit_options: list[tuple[int, int]] = []
    clip_options: list[tuple[int, int]] = []
    for ph in range(1, min(inner_h, 6) + 1):
        for pw in range(1, min(inner_w, 6) + 1):
            fits = ph * thickness <= inner_h and pw * thickness <= inner_w
            if fits:
                fit_options.append((ph, pw))
            else:
                clip_options.append((ph, pw))
    options = clip_options if clip else fit_options
    if not options:
        return None
    return rng.choice(options)


def _is_ring_mask(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for thickness in range(1, min(3, h // 2, w // 2) + 1):
        if h <= 2 * thickness or w <= 2 * thickness:
            continue
        ring = np.zeros((h, w), dtype=bool)
        ring[:thickness, :] = True
        ring[-thickness:, :] = True
        ring[:, :thickness] = True
        ring[:, -thickness:] = True
        if np.array_equal(mask, ring):
            return True
    return False


def _make_seed_patch(ph: int, pw: int, frame_color: int, rng: random.Random) -> np.ndarray:
    colors = [c for c in PALETTE if c not in {BG, frame_color}]
    n_colors = rng.randint(1, min(3, len(colors)))
    active_colors = rng.sample(colors, n_colors)

    for _ in range(100):
        patch = np.full((ph, pw), BG, dtype=int)
        density = rng.uniform(0.35, 0.8)
        filled: set[tuple[int, int]] = set()
        for r in range(ph):
            for c in range(pw):
                if rng.random() < density:
                    filled.add((r, c))

        if not filled:
            filled.add((0, 0))
        if not any(r == 0 for r, _ in filled):
            filled.add((0, rng.randrange(pw)))
        if not any(c == 0 for _, c in filled):
            filled.add((rng.randrange(ph), 0))

        if ph * pw > 1 and len(filled) == ph * pw:
            removable = [cell for cell in filled if cell != (0, 0)]
            if removable:
                filled.remove(rng.choice(removable))

        for r, c in filled:
            patch[r, c] = rng.choice(active_colors)

        mask = patch != BG
        if _is_ring_mask(mask):
            continue
        return patch

    patch = np.full((ph, pw), BG, dtype=int)
    patch[0, 0] = active_colors[0]
    if ph > 1 or pw > 1:
        patch[ph - 1, pw - 1] = active_colors[-1]
    return patch


def _can_place(
    occupied: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    pad: int = 1,
) -> bool:
    hh, ww = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _draw_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    thickness: int,
    frame_color: int,
    seed: np.ndarray,
) -> None:
    outer_h = inner_h + 2 * thickness
    outer_w = inner_w + 2 * thickness
    grid[r0 : r0 + outer_h, c0 : c0 + outer_w] = frame_color
    grid[
        r0 + thickness : r0 + thickness + inner_h,
        c0 + thickness : c0 + thickness + inner_w,
    ] = BG
    ph, pw = seed.shape
    grid[
        r0 + thickness : r0 + thickness + ph,
        c0 + thickness : c0 + thickness + pw,
    ] = seed


def _generate_one(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    frame_color = rng.choice([c for c in PALETTE if c != BG])
    n_frames = rng.choices([1, 2, 3, 4], weights=[1, 3, 3, 2], k=1)[0]
    specs: list[dict[str, int | tuple[int, int] | np.ndarray | bool]] = []

    for _ in range(n_frames):
        placed = False
        for _ in range(200):
            thickness = rng.choice([1, 2, 3])
            clip = thickness > 1 and rng.random() < 0.5

            inner_h = rng.randint(3, min(14, h - 2 * thickness - 2))
            inner_w = rng.randint(3, min(14, w - 2 * thickness - 2))
            patch_size = _sample_patch_size(inner_h, inner_w, thickness, clip, rng)
            if patch_size is None:
                continue
            ph, pw = patch_size
            outer_h = inner_h + 2 * thickness
            outer_w = inner_w + 2 * thickness
            if outer_h >= h or outer_w >= w:
                continue

            r0 = rng.randint(0, h - outer_h)
            c0 = rng.randint(0, w - outer_w)
            if not _can_place(occupied, r0, c0, outer_h, outer_w):
                continue

            seed = _make_seed_patch(ph, pw, frame_color, rng)
            specs.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "inner_h": inner_h,
                    "inner_w": inner_w,
                    "thickness": thickness,
                    "seed": seed,
                    "clip": clip,
                }
            )
            occupied[r0 : r0 + outer_h, c0 : c0 + outer_w] = True
            placed = True
            break
        if not placed:
            return None

    if all(int(spec["thickness"]) == 1 for spec in specs):
        return None

    saw_clip = False
    saw_fit = False
    for spec in specs:
        seed = spec["seed"]
        thickness = int(spec["thickness"])
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        if seed.shape[0] * thickness > inner_h or seed.shape[1] * thickness > inner_w:
            saw_clip = True
        else:
            saw_fit = True

        _draw_frame(
            grid,
            int(spec["r0"]),
            int(spec["c0"]),
            inner_h,
            inner_w,
            thickness,
            frame_color,
            seed,
        )

    if not saw_fit:
        return None
    if rng.random() < 0.5 and not saw_clip:
        return None

    frames = _find_frames(grid)
    if len(frames) != len(specs):
        return None

    out = main(grid)
    if out.shape != grid.shape or not np.any(out != grid):
        return None
    if not np.array_equal(out == frame_color, grid == frame_color):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _generate_one(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
