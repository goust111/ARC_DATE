from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_SIDES = ("top", "bottom", "left", "right")
_COLORS = tuple(range(10))
_MAX_GENERATION_ATTEMPTS = 500
_PLACEMENT_ATTEMPTS = 120


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in _DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _contiguous(indices: list[int]) -> bool:
    return bool(indices) and indices[-1] - indices[0] + 1 == len(indices)


def _parse_frame_component(cells: list[tuple[int, int]]) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    sub = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        rr = r - r0
        cc = c - c0
        if rr not in (0, h - 1) and cc not in (0, w - 1):
            return None
        sub[rr, cc] = True

    if not (sub[0, 0] and sub[0, -1] and sub[-1, 0] and sub[-1, -1]):
        return None

    top_missing = [c for c in range(1, w - 1) if not sub[0, c]]
    bottom_missing = [c for c in range(1, w - 1) if not sub[h - 1, c]]
    left_missing = [r for r in range(1, h - 1) if not sub[r, 0]]
    right_missing = [r for r in range(1, h - 1) if not sub[r, w - 1]]

    missing = {
        "top": top_missing,
        "bottom": bottom_missing,
        "left": left_missing,
        "right": right_missing,
    }
    gap_sides = [side for side, idxs in missing.items() if idxs]
    if len(gap_sides) != 1:
        return None

    side = gap_sides[0]
    gap = missing[side]
    if not _contiguous(gap):
        return None

    if side == "top":
        gap_span = (c0 + gap[0], c0 + gap[-1])
    elif side == "bottom":
        gap_span = (c0 + gap[0], c0 + gap[-1])
    elif side == "left":
        gap_span = (r0 + gap[0], r0 + gap[-1])
    else:
        gap_span = (r0 + gap[0], r0 + gap[-1])

    return {
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "gap_span": gap_span,
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict[str, object]] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for cells in _components(g == color):
            info = _parse_frame_component(cells)
            if info is None:
                continue
            info["color"] = color
            frames.append(info)
    frames.sort(key=lambda item: tuple(int(v) for v in item["bbox"]))  # stable, deterministic
    return frames


def _move_offset(shape: tuple[int, int], bbox: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if side == "top":
        return -(r0 + 1), 0
    if side == "bottom":
        return h - r1, 0
    if side == "left":
        return 0, -(c0 + 1)
    return 0, w - c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pour each frame's interior fragments out toward the grid edge indicated by its border gap, then seal the frame.

    Input structure:
    - The background is the dominant color.
    - One or more one-pixel-thick rectangular hollow frames appear in the grid; each frame is missing one contiguous segment on exactly one side.
    - Inside each frame are scattered fragment pixels whose colors differ from both the background and the frame color.
    - Outside the frame, along the gap direction out to the grid edge, the area is background.

    Rule:
    - Detect each rectangular frame, identify which side contains the contiguous gap, and recover the frame's full bbox.
    - Move every interior fragment pixel by a pure translation in that gap direction so that its distance from the frame's inner wall becomes the same distance from the corresponding grid edge.
    - Clear the original fragment pixels from the frame interior.
    - Repair the missing border segment so the frame becomes a complete rectangle.

    Key insight:
    - The gap is not damage to keep; it is a directional cue telling you which edge-aligned copy of the interior coordinate system should receive the fragments.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()

    for frame in _find_frames(g, bg):
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[index]
        side = frame["side"]  # type: ignore[index]
        color = int(frame["color"])  # type: ignore[index]
        dr, dc = _move_offset(g.shape, (r0, c0, r1, c1), str(side))

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        fragment_mask = (interior != bg) & (interior != color)
        frag_rows, frag_cols = np.where(fragment_mask)
        frag_colors = interior[fragment_mask].tolist()

        out[r0 + 1 : r1, c0 + 1 : c1][fragment_mask] = bg

        gap_start, gap_end = frame["gap_span"]  # type: ignore[index]
        if side == "top":
            out[r0, int(gap_start) : int(gap_end) + 1] = color
        elif side == "bottom":
            out[r1, int(gap_start) : int(gap_end) + 1] = color
        elif side == "left":
            out[int(gap_start) : int(gap_end) + 1, c0] = color
        else:
            out[int(gap_start) : int(gap_end) + 1, c1] = color

        for rr, cc, val in zip(frag_rows.tolist(), frag_cols.tolist(), frag_colors, strict=False):
            src_r = r0 + 1 + int(rr)
            src_c = c0 + 1 + int(cc)
            out[src_r + dr, src_c + dc] = int(val)

    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rs = slice(max(0, dr), min(h, h + dr))
            cs = slice(max(0, dc), min(w, w + dc))
            rt = slice(max(0, -dr), min(h, h - dr))
            ct = slice(max(0, -dc), min(w, w - dc))
            out[rt, ct] |= mask[rs, cs]
    return out


def _reserve_mask(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    side: str,
) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = bbox
    mask = np.zeros((h, w), dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    if side == "top" and r0 > 0:
        mask[:r0, c0 : c1 + 1] = True
    elif side == "bottom" and r1 + 1 < h:
        mask[r1 + 1 :, c0 : c1 + 1] = True
    elif side == "left" and c0 > 0:
        mask[r0 : r1 + 1, :c0] = True
    elif side == "right" and c1 + 1 < w:
        mask[r0 : r1 + 1, c1 + 1 :] = True
    return mask


def _sample_fragments(
    rng: random.Random,
    inner_h: int,
    inner_w: int,
    frame_color: int,
) -> list[tuple[int, int, int]]:
    area = inner_h * inner_w
    min_count = 2 if area >= 2 else 1
    max_count = min(5, area)
    count = rng.randint(min_count, max_count)
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    chosen_cells = rng.sample(cells, count)
    color_choices = [color for color in _COLORS if color != 0 and color != frame_color]
    chosen_colors = rng.sample(color_choices, count)
    return [(r, c, color) for (r, c), color in zip(chosen_cells, chosen_colors, strict=False)]


def _frame_candidates(
    shape: tuple[int, int],
    side: str,
    n_frames: int,
) -> tuple[range, range] | None:
    h, w = shape
    size_cap = 5 if n_frames >= 3 else 7
    if side in ("top", "bottom"):
        inner_h_max = min(size_cap, (h - 2) // 2)
        inner_w_max = min(size_cap + 2, w - 4)
    else:
        inner_h_max = min(size_cap + 2, h - 4)
        inner_w_max = min(size_cap, (w - 2) // 2)
    if inner_h_max < 2 or inner_w_max < 2:
        return None
    return range(2, inner_h_max + 1), range(2, inner_w_max + 1)


def _place_one_frame(
    rng: random.Random,
    shape: tuple[int, int],
    side: str,
    n_frames: int,
    blocked: np.ndarray,
) -> dict[str, object] | None:
    dims = _frame_candidates(shape, side, n_frames)
    if dims is None:
        return None
    inner_h_choices, inner_w_choices = dims
    h, w = shape

    for _ in range(_PLACEMENT_ATTEMPTS):
        inner_h = rng.choice(list(inner_h_choices))
        inner_w = rng.choice(list(inner_w_choices))
        frame_h = inner_h + 2
        frame_w = inner_w + 2

        if side == "top":
            r0_low, r0_high = inner_h, h - frame_h
            c0_low, c0_high = 0, w - frame_w
        elif side == "bottom":
            r0_low, r0_high = 0, h - 2 * inner_h - 2
            c0_low, c0_high = 0, w - frame_w
        elif side == "left":
            r0_low, r0_high = 0, h - frame_h
            c0_low, c0_high = inner_w, w - frame_w
        else:
            r0_low, r0_high = 0, h - frame_h
            c0_low, c0_high = 0, w - 2 * inner_w - 2

        if r0_low > r0_high or c0_low > c0_high:
            continue

        r0 = rng.randint(r0_low, r0_high)
        c0 = rng.randint(c0_low, c0_high)
        bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
        reserve = _reserve_mask(shape, bbox, side)
        if np.any(blocked & reserve):
            continue

        if side in ("top", "bottom"):
            gap_len = rng.randint(1, inner_w)
            gap_start = rng.randint(0, inner_w - gap_len)
        else:
            gap_len = rng.randint(1, inner_h)
            gap_start = rng.randint(0, inner_h - gap_len)

        return {
            "bbox": bbox,
            "side": side,
            "reserve": reserve,
            "inner_h": inner_h,
            "inner_w": inner_w,
            "gap_start": gap_start,
            "gap_len": gap_len,
        }
    return None


def _draw_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    side: str,
    gap_start: int,
    gap_len: int,
    bg: int,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    if side == "top":
        start = c0 + 1 + gap_start
        grid[r0, start : start + gap_len] = bg
    elif side == "bottom":
        start = c0 + 1 + gap_start
        grid[r1, start : start + gap_len] = bg
    elif side == "left":
        start = r0 + 1 + gap_start
        grid[start : start + gap_len, c0] = bg
    else:
        start = r0 + 1 + gap_start
        grid[start : start + gap_len, c1] = bg


def _render_output_from_specs(
    shape: tuple[int, int],
    frame_color: int,
    specs: list[dict[str, object]],
) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    for spec in specs:
        bbox = spec["bbox"]  # type: ignore[index]
        side = spec["side"]  # type: ignore[index]
        r0, c0, r1, c1 = bbox
        out[r0, c0 : c1 + 1] = frame_color
        out[r1, c0 : c1 + 1] = frame_color
        out[r0 : r1 + 1, c0] = frame_color
        out[r0 : r1 + 1, c1] = frame_color

        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        if side == "top":
            base_r, base_c = 0, c0 + 1
        elif side == "bottom":
            base_r, base_c = h - inner_h, c0 + 1
        elif side == "left":
            base_r, base_c = r0 + 1, 0
        else:
            base_r, base_c = r0 + 1, w - inner_w

        for rr, cc, color in spec["fragments"]:  # type: ignore[index]
            out[base_r + int(rr), base_c + int(cc)] = int(color)
    return out


def _generate_candidate(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    n_frames = rng.randint(1, 4)
    base_min = 12 + 3 * n_frames
    h = rng.randint(base_min, 30)
    w = rng.randint(base_min, 30)
    if rng.random() < 0.5:
        delta = rng.randint(-4, 4)
        w = min(30, max(base_min, h + delta))

    frame_color = rng.choice([color for color in _COLORS if color != 0])
    chosen_sides = rng.sample(list(_SIDES), n_frames)
    blocked = np.zeros((h, w), dtype=bool)
    specs: list[dict[str, object]] = []

    for side in chosen_sides:
        spec = _place_one_frame(rng, (h, w), side, n_frames, blocked)
        if spec is None:
            return None
        spec["fragments"] = _sample_fragments(
            rng,
            int(spec["inner_h"]),
            int(spec["inner_w"]),
            frame_color,
        )
        specs.append(spec)
        blocked |= _dilate(spec["reserve"])  # type: ignore[arg-type]

    inp = np.zeros((h, w), dtype=int)
    for spec in specs:
        _draw_frame(
            inp,
            spec["bbox"],  # type: ignore[arg-type]
            frame_color,
            str(spec["side"]),
            int(spec["gap_start"]),
            int(spec["gap_len"]),
            0,
        )
        r0, c0, _r1, _c1 = spec["bbox"]  # type: ignore[index]
        for rr, cc, color in spec["fragments"]:  # type: ignore[index]
            inp[r0 + 1 + int(rr), c0 + 1 + int(cc)] = int(color)

    expected = _render_output_from_specs((h, w), frame_color, specs)

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[int(np.argmax(counts))]) != 0:
        return None
    if np.any(inp != 0) and Counter(inp[inp != 0].tolist())[frame_color] >= counts.max():
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        candidate = _generate_candidate(rng)
        if candidate is not None:
            inp, _expected = candidate
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
