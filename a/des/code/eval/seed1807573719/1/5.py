from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
SYM_CLASSES = ("lr_only", "ud_only", "both", "none")
INTERIOR_DIMS = ((3, 3), (3, 5), (5, 3), (5, 5))
MAX_PATTERN_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 400

_RNG = random.Random()


def _components_of_color(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    mask = grid == color
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": int(color),
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _rect_frame_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_rect_frame_component(grid: np.ndarray, comp: dict[str, object]) -> bool:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 5 or width < 5:
        return False
    window = grid[r0 : r1 + 1, c0 : c1 + 1] == color
    return np.array_equal(window, _rect_frame_mask(height, width))


def _find_windows(grid: np.ndarray) -> list[dict[str, object]]:
    windows: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        for comp in _components_of_color(grid, color):
            if _is_rect_frame_component(grid, comp):
                windows.append(comp)
    windows.sort(key=lambda item: tuple(item["bbox"]))
    assert windows, "expected at least one framed window"
    return windows


def _classify_symmetry(mask: np.ndarray) -> str:
    lr_sym = np.array_equal(mask, mask[:, ::-1])
    ud_sym = np.array_equal(mask, mask[::-1, :])
    if lr_sym and ud_sym:
        return "both"
    if lr_sym:
        return "lr_only"
    if ud_sym:
        return "ud_only"
    return "none"


def _draw_arrow(out: np.ndarray, start_r: int, start_c: int, dr: int, dc: int, color: int) -> None:
    cells = [(start_r, start_c), (start_r + dr, start_c + dc)]
    tip_r, tip_c = cells[-1]
    if dr != 0:
        cells.extend([(tip_r, tip_c - 1), (tip_r, tip_c + 1)])
    else:
        cells.extend([(tip_r - 1, tip_c), (tip_r + 1, tip_c)])

    h, w = out.shape
    for r, c in cells:
        assert 0 <= r < h and 0 <= c < w, "arrow would go out of bounds"
        assert out[r, c] in (BG, color), "arrows must not overlap unrelated objects"
        out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect colored rectangular windows, classify the interior pattern symmetry,
    and draw frame-colored arrows outside the window.

    The puzzle wording uses the mirror directions as the key:
    - left-right mirror symmetry -> draw up/down arrows
    - up-down mirror symmetry -> draw left/right arrows
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for window in _find_windows(g):
        r0, c0, r1, c1 = window["bbox"]
        frame_color = int(window["color"])
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        pattern_colors = [int(v) for v in np.unique(interior) if int(v) != BG]
        assert len(pattern_colors) == 1, "each window should contain one pattern color"
        pattern_mask = interior == pattern_colors[0]
        assert np.any(pattern_mask), "pattern must not be empty"

        sym = _classify_symmetry(pattern_mask)
        mid_r = (r0 + r1) // 2
        mid_c = (c0 + c1) // 2

        if sym in ("lr_only", "both"):
            _draw_arrow(out, r0 - 1, mid_c, -1, 0, frame_color)
            _draw_arrow(out, r1 + 1, mid_c, 1, 0, frame_color)
        if sym in ("ud_only", "both"):
            _draw_arrow(out, mid_r, c0 - 1, 0, -1, frame_color)
            _draw_arrow(out, mid_r, c1 + 1, 0, 1, frame_color)

    return out


def _random_mask(rng: random.Random, height: int, width: int, symmetry: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)

    if symmetry == "both":
        for r in range((height + 1) // 2):
            for c in range((width + 1) // 2):
                value = rng.random() < 0.45
                for rr in {r, height - 1 - r}:
                    for cc in {c, width - 1 - c}:
                        mask[rr, cc] = value
        return mask

    if symmetry == "lr_only":
        for r in range(height):
            for c in range((width + 1) // 2):
                value = rng.random() < 0.45
                mask[r, c] = value
                mask[r, width - 1 - c] = value
        return mask

    if symmetry == "ud_only":
        for r in range((height + 1) // 2):
            for c in range(width):
                value = rng.random() < 0.45
                mask[r, c] = value
                mask[height - 1 - r, c] = value
        return mask

    for r in range(height):
        for c in range(width):
            mask[r, c] = rng.random() < 0.45
    return mask


def _is_inner_frame(mask: np.ndarray) -> bool:
    height, width = mask.shape
    if height < 3 or width < 3:
        return False
    return np.array_equal(mask, _rect_frame_mask(height, width))


def _generate_pattern_mask(rng: random.Random, height: int, width: int, symmetry: str) -> np.ndarray:
    for _ in range(MAX_PATTERN_ATTEMPTS):
        mask = _random_mask(rng, height, width, symmetry)
        filled = int(mask.sum())
        if filled < 2 or filled >= height * width:
            continue
        if _is_inner_frame(mask):
            continue
        if _classify_symmetry(mask) != symmetry:
            continue
        return mask
    raise RuntimeError(f"failed to build pattern for symmetry={symmetry}")


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _place_windows(
    rng: random.Random, height: int, width: int, specs: list[dict[str, object]]
) -> list[dict[str, object]] | None:
    placed: list[dict[str, object]] = []
    reserved: list[tuple[int, int, int, int]] = []

    order = sorted(specs, key=lambda item: item["frame_h"] * item["frame_w"], reverse=True)
    for spec in order:
        frame_h = int(spec["frame_h"])
        frame_w = int(spec["frame_w"])
        positions = [
            (r0, c0)
            for r0 in range(2, height - frame_h - 1)
            for c0 in range(2, width - frame_w - 1)
        ]
        rng.shuffle(positions)

        picked = None
        for r0, c0 in positions:
            bbox = (r0 - 2, c0 - 2, r0 + frame_h + 1, c0 + frame_w + 1)
            if any(_boxes_overlap(bbox, other) for other in reserved):
                continue
            picked = (r0, c0, bbox)
            break

        if picked is None:
            return None

        r0, c0, bbox = picked
        spec = dict(spec)
        spec["top"] = r0
        spec["left"] = c0
        placed.append(spec)
        reserved.append(bbox)

    return placed


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_windows = rng.randint(1, 4)
    min_side = 14 + 3 * n_windows
    height = rng.randint(min_side, 33)
    width = rng.randint(min_side, 33)

    symmetries = [rng.choice(SYM_CLASSES) for _ in range(n_windows)]
    if all(sym == "none" for sym in symmetries):
        symmetries[rng.randrange(n_windows)] = rng.choice(SYM_CLASSES[:-1])

    specs: list[dict[str, object]] = []
    for sym in symmetries:
        inner_h, inner_w = rng.choice(INTERIOR_DIMS)
        specs.append(
            {
                "symmetry": sym,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "frame_h": inner_h + 2,
                "frame_w": inner_w + 2,
                "mask": _generate_pattern_mask(rng, inner_h, inner_w, sym),
            }
        )

    placements = _place_windows(rng, height, width, specs)
    if placements is None:
        return None

    palette = rng.sample(COLORS, 2 * n_windows)
    frame_colors = palette[:n_windows]
    pattern_colors = palette[n_windows:]

    grid = np.zeros((height, width), dtype=int)
    for idx, spec in enumerate(placements):
        r0 = int(spec["top"])
        c0 = int(spec["left"])
        frame_h = int(spec["frame_h"])
        frame_w = int(spec["frame_w"])
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1
        frame_color = frame_colors[idx]
        pattern_color = pattern_colors[idx]

        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color
        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        inner[np.asarray(spec["mask"], dtype=bool)] = pattern_color

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid window-symmetry input")
