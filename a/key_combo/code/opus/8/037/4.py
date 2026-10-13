from __future__ import annotations

import hashlib
from collections import deque

import numpy as np


BACKGROUND = 0
TOP_COLOR = 2
RIGHT_COLOR = 1
BOTTOM_COLOR = 3
LEFT_COLOR = 4
SIDE_COLORS = (TOP_COLOR, RIGHT_COLOR, BOTTOM_COLOR, LEFT_COLOR)
MOVE_BY_COLOR = {
    TOP_COLOR: (-1, 0),
    RIGHT_COLOR: (0, 1),
    BOTTOM_COLOR: (1, 0),
    LEFT_COLOR: (0, -1),
}

_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_component_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _repeat_axes(mask: np.ndarray) -> list[str]:
    axes: list[str] = []
    h, w = mask.shape
    if h % 2 == 0 and np.array_equal(mask[: h // 2, :], mask[h // 2 :, :]):
        axes.append("h")
    if w % 2 == 0 and np.array_equal(mask[:, : w // 2], mask[:, w // 2 :]):
        axes.append("v")
    return axes


def _find_color_components(grid: np.ndarray, colors: tuple[int, ...]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in colors or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            mask, bbox = _crop_component_mask(coords)
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "mask": mask,
                    "area": len(coords),
                }
            )
    return out


def _frame_grid(h: int, w: int) -> np.ndarray:
    g = np.zeros((h, w), dtype=int)
    g[0, 1:-1] = TOP_COLOR
    g[1:-1, -1] = RIGHT_COLOR
    g[-1, 1:-1] = BOTTOM_COLOR
    g[1:-1, 0] = LEFT_COLOR
    return g


def _signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    coords = np.argwhere(mask)
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _can_place(blocked: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > blocked.shape[0] or left + pw > blocked.shape[1]:
        return False
    return not np.any(blocked[top : top + ph, left : left + pw] & patch)


def _stamp_bool(mask: np.ndarray, patch: np.ndarray, top: int, left: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    ph, pw = patch.shape
    out[top : top + ph, left : left + pw] |= patch
    return out


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> None:
    ph, pw = patch.shape
    view = grid[top : top + ph, left : left + pw]
    view[patch] = int(color)


def _random_connected_mask(
    rng: np.random.Generator,
    h: int,
    w: int,
    area: int,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start_r = int(rng.integers(h))
    start_c = int(rng.integers(w))
    mask[start_r, start_c] = True

    while int(mask.sum()) < area:
        frontier: list[tuple[int, int]] = []
        for r_raw, c_raw in np.argwhere(mask):
            r = int(r_raw)
            c = int(c_raw)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = frontier[int(rng.integers(len(frontier)))]
        mask[nr, nc] = True
    return mask


def _has_full_bbox(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return (r0, c0, r1, c1) == (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)


def _make_half_mask(axis: str, rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        if axis == "h":
            h = int(rng.integers(2, 4))
            w = int(rng.integers(2, 5))
        else:
            h = int(rng.integers(2, 5))
            w = int(rng.integers(2, 4))
        max_area = min(6, h * w)
        min_area = min(3, max_area)
        area = int(rng.integers(min_area, max_area + 1))
        mask = _random_connected_mask(rng, h, w, area)
        if not _has_full_bbox(mask):
            continue
        full = np.vstack([mask, mask]) if axis == "h" else np.hstack([mask, mask])
        if _repeat_axes(full) != [axis]:
            continue
        return mask
    raise RuntimeError("failed to build repeated half-mask")


def _make_noise_mask(rng: np.random.Generator, force_even_dim: bool) -> np.ndarray:
    for _ in range(300):
        if force_even_dim:
            h = int(rng.choice([2, 3, 4]))
            w = int(rng.choice([2, 3, 4]))
            if h % 2 and w % 2:
                continue
        else:
            h = int(rng.integers(2, 5))
            w = int(rng.integers(2, 5))
        max_area = min(12, h * w)
        area = int(rng.integers(4, max_area + 1))
        mask = _random_connected_mask(rng, h, w, area)
        if not _has_full_bbox(mask):
            continue
        if _repeat_axes(mask):
            continue
        return mask
    raise RuntimeError("failed to build non-repeating mask")


def _validate_input_structure(grid: np.ndarray, expected_count: int) -> list[dict]:
    h, w = grid.shape
    assert 10 <= h <= 30 and 10 <= w <= 30
    assert np.all(grid[0, 1:-1] == TOP_COLOR)
    assert np.all(grid[1:-1, -1] == RIGHT_COLOR)
    assert np.all(grid[-1, 1:-1] == BOTTOM_COLOR)
    assert np.all(grid[1:-1, 0] == LEFT_COLOR)
    assert int(grid[0, 0]) == int(grid[0, -1]) == int(grid[-1, 0]) == int(grid[-1, -1]) == 0

    inner = grid.copy()
    inner[0, :] = 0
    inner[-1, :] = 0
    inner[:, 0] = 0
    inner[:, -1] = 0
    comps = _find_color_components(inner, SIDE_COLORS)
    assert len(comps) == expected_count
    for comp in comps:
        assert 4 <= comp["area"] <= 12
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect self-repeating interior sprites, keep their first half, and shift them toward the matching frame side.

    Input structure:
    - The grid has a black background with a one-cell rectangular frame whose top/right/bottom/left sides are red/blue/green/yellow and whose corners are black.
    - Inside the frame are several non-overlapping monochromatic 4-connected sprites using only those four side colors.
    - Some sprites have a bounding box that splits into two identical top/bottom halves or two identical left/right halves.

    Rule:
    - Ignore the frame itself and examine each interior sprite inside its own minimal bounding box.
    - If exactly one repeat axis exists, keep only the first half along that axis: the top half for a horizontal split or the left half for a vertical split.
    - Shift that kept half by one cell toward the frame side whose color matches the sprite color; erase sprites with no repeat axis.
    - Keep the frame unchanged.

    Key insight:
    - The frame colors do not move every matching sprite: they only provide directions after a separate per-sprite repeated-half test decides which sprites are real and which are noise.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 5 and w >= 5

    out = _frame_grid(h, w)
    inner = g.copy()
    inner[0, :] = 0
    inner[-1, :] = 0
    inner[:, 0] = 0
    inner[:, -1] = 0

    for comp in _find_color_components(inner, SIDE_COLORS):
        axes = _repeat_axes(comp["mask"])
        if not axes:
            continue
        assert len(axes) == 1, "expected at most one valid repeat axis per sprite"

        axis = axes[0]
        color = int(comp["color"])
        dr, dc = MOVE_BY_COLOR[color]
        r0, c0, _r1, _c1 = comp["bbox"]
        mask = comp["mask"]
        if axis == "h":
            kept = mask[: mask.shape[0] // 2, :]
        else:
            kept = mask[:, : mask.shape[1] // 2]

        top = r0 + dr
        left = c0 + dc
        for rr_raw, cc_raw in np.argwhere(kept):
            rr = top + int(rr_raw)
            cc = left + int(cc_raw)
            assert 1 <= rr < h - 1 and 1 <= cc < w - 1
            assert out[rr, cc] == BACKGROUND
            out[rr, cc] = color

    return out


def _attempt_generate(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    h = int(rng.integers(15, 20))
    w = int(rng.integers(15, 20))
    total_objects = int(rng.integers(4, 7))
    repeated_count = int(rng.choice([2, 3] if total_objects >= 5 else [2]))
    noise_count = total_objects - repeated_count

    repeated_axes = ["h", "v"]
    while len(repeated_axes) < repeated_count:
        repeated_axes.append("h" if rng.random() < 0.5 else "v")
    rng.shuffle(repeated_axes)

    colors = list(SIDE_COLORS)
    rng.shuffle(colors)
    while len(colors) < total_objects:
        colors.append(int(rng.choice(SIDE_COLORS)))
    rng.shuffle(colors)
    repeated_colors = colors[:repeated_count]
    noise_colors = colors[repeated_count:]

    inp = _frame_grid(h, w)
    target = _frame_grid(h, w)
    in_blocked = np.zeros((h, w), dtype=bool)
    out_blocked = np.zeros((h, w), dtype=bool)

    repeated_specs = []
    for axis, color in zip(repeated_axes, repeated_colors):
        half = _make_half_mask(axis, rng)
        full = np.vstack([half, half]) if axis == "h" else np.hstack([half, half])
        repeated_specs.append(
            {
                "axis": axis,
                "color": int(color),
                "half": half,
                "full": full,
                "size": int(full.sum()),
            }
        )
    repeated_specs.sort(key=lambda item: item["size"], reverse=True)

    repeated_meta = []
    for spec in repeated_specs:
        axis = str(spec["axis"])
        color = int(spec["color"])
        half = spec["half"]
        full = spec["full"]
        fh, fw = full.shape
        kh, kw = half.shape
        dr, dc = MOVE_BY_COLOR[color]

        placed = False
        for _ in range(300):
            min_top = 2
            max_top = h - 2 - fh
            min_left = 2
            max_left = w - 2 - fw
            if min_top > max_top or min_left > max_left:
                break
            top = int(rng.integers(min_top, max_top + 1))
            left = int(rng.integers(min_left, max_left + 1))
            out_top = top + dr
            out_left = left + dc
            if out_top < 1 or out_left < 1 or out_top + kh > h - 1 or out_left + kw > w - 1:
                continue
            if not _can_place(in_blocked, full, top, left):
                continue
            if not _can_place(out_blocked, half, out_top, out_left):
                continue

            _paint_patch(inp, full, top, left, int(color))
            _paint_patch(target, half, out_top, out_left, int(color))
            in_blocked |= _dilate8(_stamp_bool(in_blocked, full, top, left))
            out_blocked |= _dilate8(_stamp_bool(out_blocked, half, out_top, out_left))
            repeated_meta.append({"axis": axis, "color": color})
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place repeated sprite")

    noise_specs = []
    for idx, color in enumerate(noise_colors):
        patch = _make_noise_mask(rng, force_even_dim=(idx == 0))
        noise_specs.append({"color": int(color), "patch": patch, "size": int(patch.sum())})
    noise_specs.sort(key=lambda item: item["size"], reverse=True)

    for spec in noise_specs:
        color = int(spec["color"])
        patch = spec["patch"]
        ph, pw = patch.shape
        placed = False
        for _ in range(300):
            min_top = 2
            max_top = h - 2 - ph
            min_left = 2
            max_left = w - 2 - pw
            if min_top > max_top or min_left > max_left:
                break
            top = int(rng.integers(min_top, max_top + 1))
            left = int(rng.integers(min_left, max_left + 1))
            if not _can_place(in_blocked, patch, top, left):
                continue
            _paint_patch(inp, patch, top, left, int(color))
            in_blocked |= _dilate8(_stamp_bool(in_blocked, patch, top, left))
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place noise sprite")

    comps = _validate_input_structure(inp, total_objects)
    repeated = [comp for comp in comps if len(_repeat_axes(comp["mask"])) == 1]
    noise = [comp for comp in comps if not _repeat_axes(comp["mask"])]
    assert len(repeated) == repeated_count
    assert len(noise) == noise_count
    assert {"h", "v"}.issubset({axis["axis"] for axis in repeated_meta})

    solved = main(inp)
    assert np.array_equal(solved, target)
    return inp, target


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(600):
        try:
            inp, out = _attempt_generate(rng)
        except (AssertionError, RuntimeError):
            continue
        out_sig = _signature(out)
        if out_sig in _SEEN_OUTPUT_SIGNATURES:
            continue
        _SEEN_OUTPUT_SIGNATURES.add(out_sig)
        return inp
    raise RuntimeError("failed to generate a unique valid sample")
