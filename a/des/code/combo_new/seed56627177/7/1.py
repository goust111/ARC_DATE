from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")
GEN_MAX_ATTEMPTS = 300
GEN_COUNT_RANGE = (2, 4)
GEN_FRAME_H_RANGE = (5, 10)
GEN_FRAME_W_RANGE = (5, 10)
GEN_COLORS = tuple(range(1, 10))
GEN_PADDING = 1
GEN_GRID_MIN = 18
GEN_GRID_MAX = 30


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(g: np.ndarray) -> list[dict]:
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return comps


def _is_consecutive(values: list[int]) -> bool:
    ordered = sorted(int(v) for v in values)
    return all(b == a + 1 for a, b in zip(ordered, ordered[1:]))


def _analyze_frame(comp: dict) -> dict | None:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 4 or w < 4:
        return None

    coords_set = {(int(r), int(c)) for r, c in comp["coords"]}
    if any(r not in (r0, r1) and c not in (c0, c1) for r, c in coords_set):
        return None

    side_gaps: dict[str, list[int]] = {}
    top = [c for c in range(c0, c1 + 1) if (r0, c) not in coords_set]
    bottom = [c for c in range(c0, c1 + 1) if (r1, c) not in coords_set]
    left = [r for r in range(r0, r1 + 1) if (r, c0) not in coords_set]
    right = [r for r in range(r0, r1 + 1) if (r, c1) not in coords_set]
    if top:
        side_gaps["top"] = top
    if bottom:
        side_gaps["bottom"] = bottom
    if left:
        side_gaps["left"] = left
    if right:
        side_gaps["right"] = right
    if len(side_gaps) != 1:
        return None

    side, values = next(iter(side_gaps.items()))
    if len(values) < 2 or not _is_consecutive(values):
        return None

    gap_len = len(values)
    perimeter = 2 * (h + w) - 4
    if int(comp["area"]) != perimeter - gap_len:
        return None

    if side == "top":
        anchor = (r0, min(values))
    elif side == "bottom":
        anchor = (r1, min(values))
    elif side == "left":
        anchor = (min(values), c0)
    else:
        anchor = (min(values), c1)

    return {
        **comp,
        "side": side,
        "gap_len": gap_len,
        "anchor": anchor,
    }


def _paste_line(out: np.ndarray, color: int, length: int, anchor: tuple[int, int], vertical: bool) -> None:
    r0, c0 = map(int, anchor)
    if vertical:
        out[r0 : r0 + length, c0] = int(color)
    else:
        out[r0, c0 : c0 + length] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored fragment to its same-colored frame and plug the frame's missing border segment.

    Input structure:
    - The background is black.
    - Several monochrome rectangular frames with distinct border colors are scattered in the grid.
    - Each frame is a one-pixel-thick rectangle with exactly one contiguous border gap on one side.
    - Separate solid fragments appear elsewhere; each fragment shares its color with exactly one frame.

    Rule:
    - Detect every hollow rectangle and identify which border side contains its missing segment.
    - Pair each frame with the fragment of the same color.
    - Remove the fragment from its original location, keep it horizontal for top/bottom gaps, and use a 90-degree rotation for left/right gaps.
    - Paste the fragment into the missing border segment so the frame becomes complete, while leaving all other pixels unchanged.

    Key insight:
    Color determines which fragment belongs to which frame, but the gap side determines whether the fragment stays horizontal or must become vertical.
    """
    g = np.asarray(grid, dtype=int)
    comps = _components(g)

    frames: dict[int, dict] = {}
    fragments: dict[int, dict] = {}
    for comp in comps:
        frame = _analyze_frame(comp)
        if frame is not None:
            color = int(frame["color"])
            assert color not in frames, "frame colors must be unique"
            frames[color] = frame
        else:
            color = int(comp["color"])
            assert color not in fragments, "fragment colors must be unique"
            fragments[color] = comp

    assert frames and len(frames) == len(fragments), "expected one fragment per frame"

    out = g.copy()
    for color, frame in frames.items():
        frag = fragments.get(int(color))
        assert frag is not None, "every frame must have a fragment with the same color"
        fr0, fc0, fr1, fc1 = map(int, frag["bbox"])
        frag_h = int(fr1 - fr0 + 1)
        frag_w = int(fc1 - fc0 + 1)
        assert min(frag_h, frag_w) == 1, "fragments must be one-pixel-thick strips"
        assert int(frag["area"]) == int(frame["gap_len"]), "fragment length must match gap length"

        for r, c in frag["coords"]:
            out[int(r), int(c)] = BG

        vertical = frame["side"] in ("left", "right")
        _paste_line(out, int(color), int(frame["gap_len"]), frame["anchor"], vertical)

    return out


def _make_frame_patch(
    color: int, height: int, width: int, side: str, gap_start: int, gap_len: int
) -> tuple[np.ndarray, tuple[int, int]]:
    patch = np.zeros((int(height), int(width)), dtype=int)
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)

    if side == "top":
        patch[0, gap_start : gap_start + gap_len] = BG
        anchor = (0, int(gap_start))
    elif side == "bottom":
        patch[-1, gap_start : gap_start + gap_len] = BG
        anchor = (int(height - 1), int(gap_start))
    elif side == "left":
        patch[gap_start : gap_start + gap_len, 0] = BG
        anchor = (int(gap_start), 0)
    elif side == "right":
        patch[gap_start : gap_start + gap_len, -1] = BG
        anchor = (int(gap_start), int(width - 1))
    else:
        raise ValueError(side)
    return patch, anchor


def _sample_sides(count: int) -> list[str]:
    if count == 2:
        sides = [random.choice(("top", "bottom")), random.choice(("left", "right"))]
    elif count == 3:
        sides = [
            random.choice(("top", "bottom")),
            random.choice(("left", "right")),
            random.choice(SIDES),
        ]
    else:
        sides = random.sample(list(SIDES), 4)
    random.shuffle(sides)
    return sides


def _sample_specs() -> list[dict] | None:
    count = random.randint(*GEN_COUNT_RANGE)
    colors = random.sample(list(GEN_COLORS), count)
    sides = _sample_sides(count)
    specs: list[dict] = []
    for color, side in zip(colors, sides):
        for _ in range(50):
            height = random.randint(*GEN_FRAME_H_RANGE)
            width = random.randint(*GEN_FRAME_W_RANGE)
            max_gap = (width - 2) if side in ("top", "bottom") else (height - 2)
            if max_gap < 2:
                continue
            gap_len = random.randint(2, max_gap)
            gap_start = random.randint(1, max_gap - gap_len + 1)
            frame_patch, anchor = _make_frame_patch(color, height, width, side, gap_start, gap_len)
            frag_patch = np.full((1, gap_len), int(color), dtype=int)
            specs.append(
                {
                    "color": int(color),
                    "side": side,
                    "gap_len": int(gap_len),
                    "frame_patch": frame_patch,
                    "fragment_patch": frag_patch,
                    "gap_anchor": anchor,
                }
            )
            break
        else:
            return None
    return specs


def _sample_canvas(specs: list[dict]) -> tuple[int, int] | None:
    padded_area = 0
    max_h = 0
    max_w = 0
    for spec in specs:
        fh, fw = map(int, spec["frame_patch"].shape)
        gh, gw = map(int, spec["fragment_patch"].shape)
        max_h = max(max_h, fh, gh)
        max_w = max(max_w, fw, gw)
        padded_area += (fh + 2 * GEN_PADDING) * (fw + 2 * GEN_PADDING)
        padded_area += (gh + 2 * GEN_PADDING) * (gw + 2 * GEN_PADDING)

    base = max(
        GEN_GRID_MIN,
        max_h + 2 * GEN_PADDING,
        max_w + 2 * GEN_PADDING,
        int(np.ceil(np.sqrt(float(padded_area)))) + 1,
    )
    if base > GEN_GRID_MAX:
        return None

    grid_h = random.randint(base, GEN_GRID_MAX)
    min_w = max(base, int(np.ceil(float(padded_area) / grid_h)))
    if min_w > GEN_GRID_MAX:
        return None
    grid_w = random.randint(min_w, GEN_GRID_MAX)
    if random.random() < 0.5:
        grid_h, grid_w = grid_w, grid_h
    return int(grid_h), int(grid_w)


def _can_place_bbox(blocked: np.ndarray, shape: tuple[int, int], r0: int, c0: int) -> bool:
    h, w = map(int, shape)
    H, W = map(int, blocked.shape)
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, int(r0 - GEN_PADDING))
    rr1 = min(H, int(r0 + h + GEN_PADDING))
    cc0 = max(0, int(c0 - GEN_PADDING))
    cc1 = min(W, int(c0 + w + GEN_PADDING))
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _mark_bbox(blocked: np.ndarray, shape: tuple[int, int], r0: int, c0: int) -> None:
    h, w = map(int, shape)
    H, W = map(int, blocked.shape)
    rr0 = max(0, int(r0 - GEN_PADDING))
    rr1 = min(H, int(r0 + h + GEN_PADDING))
    cc0 = max(0, int(c0 - GEN_PADDING))
    cc1 = min(W, int(c0 + w + GEN_PADDING))
    blocked[rr0:rr1, cc0:cc1] = True


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = map(int, patch.shape)
    mask = patch != BG
    target = grid[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    target[mask] = patch[mask]


def _place_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    h, w = map(int, patch.shape)
    H, W = map(int, grid.shape)
    choices = [
        (r, c)
        for r in range(H - h + 1)
        for c in range(W - w + 1)
        if _can_place_bbox(blocked, patch.shape, r, c)
    ]
    if not choices:
        return None
    r0, c0 = random.choice(choices)
    _paste_patch(grid, patch, int(r0), int(c0))
    _mark_bbox(blocked, patch.shape, int(r0), int(c0))
    return int(r0), int(c0)


def _build_expected(inp: np.ndarray, placed_specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in placed_specs:
        frag_r, frag_c = map(int, spec["fragment_pos"])
        gap_len = int(spec["gap_len"])
        out[frag_r, frag_c : frag_c + gap_len] = BG

        frame_r, frame_c = map(int, spec["frame_pos"])
        rel_r, rel_c = map(int, spec["gap_anchor"])
        color = int(spec["color"])
        if spec["side"] in ("top", "bottom"):
            out[frame_r + rel_r, frame_c + rel_c : frame_c + rel_c + gap_len] = color
        else:
            out[frame_r + rel_r : frame_r + rel_r + gap_len, frame_c + rel_c] = color
    return out


def _generate_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    if specs is None:
        return None

    canvas = _sample_canvas(specs)
    if canvas is None:
        return None
    grid_h, grid_w = canvas
    inp = np.zeros((int(grid_h), int(grid_w)), dtype=int)
    blocked = np.zeros_like(inp, dtype=bool)

    placed_specs: list[dict] = []
    for spec in sorted(specs, key=lambda d: d["frame_patch"].size, reverse=True):
        frame_pos = _place_patch(inp, blocked, spec["frame_patch"])
        if frame_pos is None:
            return None
        fragment_pos = _place_patch(inp, blocked, spec["fragment_patch"])
        if fragment_pos is None:
            return None
        placed_specs.append({**spec, "frame_pos": frame_pos, "fragment_pos": fragment_pos})

    expected = _build_expected(inp, placed_specs)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
