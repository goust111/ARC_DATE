from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


WORKSPACE = 30
INNER_SIZES = (3, 5, 7)
QUADRANTS = ("tl", "tr", "bl", "br")
MAX_ATTEMPTS = 500
PAIR_ATTEMPTS = 600
FRAME_ATTEMPTS = 300
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass(frozen=True)
class FrameSpec:
    r0: int
    c0: int
    inner_h: int
    inner_w: int
    quadrant: str
    seed_color: int
    seed_r: int
    seed_c: int

    @property
    def outer_h(self) -> int:
        return self.inner_h + 2

    @property
    def outer_w(self) -> int:
        return self.inner_w + 2

    @property
    def r1(self) -> int:
        return self.r0 + self.outer_h - 1

    @property
    def c1(self) -> int:
        return self.c0 + self.outer_w - 1

    @property
    def mid_r(self) -> int:
        return self.r0 + self.outer_h // 2

    @property
    def mid_c(self) -> int:
        return self.c0 + self.outer_w // 2


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _detect_frame(grid: np.ndarray, color: int, coords: np.ndarray) -> tuple[int, int, int, int] | None:
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None
    if int(coords.shape[0]) != 2 * h + 2 * w - 4:
        return None
    for br, bc in _border_cells(r0, c0, r1, c1):
        if int(grid[br, bc]) != color:
            return None
    if h > 2 and w > 2 and np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == color):
        return None
    return r0, c0, r1, c1


def _paint_ray(
    out: np.ndarray,
    frame_mask: np.ndarray,
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
    length: int,
    color: int,
) -> None:
    h, w = out.shape
    r = start_r
    c = start_c
    for _ in range(length):
        if not (0 <= r < h and 0 <= c < w):
            break
        if frame_mask[r, c]:
            break
        out[r, c] = color
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow two outward branches from each framed seed, using the seed color and the seed's quadrant inside the frame.

    Input structure:
    - The grid has a dominant background color plus several one-pixel-thick hollow rectangular frames.
    - Every valid frame is a monochrome rectangular border with exactly one non-background seed pixel inside it.
    - Different seeds may have different colors, while the frame color is separate from both the background and the seeds.

    Rule:
    - Keep every frame and every seed unchanged.
    - For each frame, compare the seed position with the frame center to determine one of four quadrants.
    - Draw one vertical ray from the matching top or bottom edge midpoint, and one horizontal ray from the matching left or right edge midpoint; both rays start outside the frame, use the seed color, and have lengths set by the perpendicular interior size.
    - Stop a ray early if the next cell would be another detected frame border.

    Key insight:
    - The seed color controls what to paint, the seed quadrant controls which two edge midpoints activate, and the frame's interior width/height swap determines the two ray lengths.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    frames: list[tuple[int, int, int, int, int]] = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for comp in _components_of_color(g, color):
            bbox = _detect_frame(g, color, comp)
            if bbox is not None:
                frames.append((color, *bbox))

    assert frames, "expected at least one hollow frame"
    frames.sort(key=lambda item: (item[1], item[2], item[3], item[4], item[0]))

    frame_mask = np.zeros_like(g, dtype=bool)
    for _color, r0, c0, r1, c1 in frames:
        for br, bc in _border_cells(r0, c0, r1, c1):
            frame_mask[br, bc] = True

    out = g.copy()
    for frame_color, r0, c0, r1, c1 in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_pos = np.argwhere((inner != bg) & (inner != frame_color))
        assert seed_pos.shape[0] == 1, "each frame must contain exactly one seed pixel"
        sr = int(seed_pos[0, 0] + r0 + 1)
        sc = int(seed_pos[0, 1] + c0 + 1)
        seed_color = int(g[sr, sc])

        top_half = 2 * sr < r0 + r1
        left_half = 2 * sc < c0 + c1
        mid_r = (r0 + r1) // 2
        mid_c = (c0 + c1) // 2
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1

        if top_half:
            _paint_ray(out, frame_mask, r0 - 1, mid_c, -1, 0, inner_w, seed_color)
        else:
            _paint_ray(out, frame_mask, r1 + 1, mid_c, 1, 0, inner_w, seed_color)

        if left_half:
            _paint_ray(out, frame_mask, mid_r, c0 - 1, 0, -1, inner_h, seed_color)
        else:
            _paint_ray(out, frame_mask, mid_r, c1 + 1, 0, 1, inner_h, seed_color)

    return out


def _sample_seed_local(inner_h: int, inner_w: int, quadrant: str) -> tuple[int, int]:
    center_r = inner_h // 2
    center_c = inner_w // 2
    row_choices = list(range(center_r)) if quadrant[0] == "t" else list(range(center_r + 1, inner_h))
    col_choices = list(range(center_c)) if quadrant[1] == "l" else list(range(center_c + 1, inner_w))
    return random.choice(row_choices), random.choice(col_choices)


def _make_frame(r0: int, c0: int, inner_h: int, inner_w: int, quadrant: str, seed_color: int) -> FrameSpec:
    seed_lr, seed_lc = _sample_seed_local(inner_h, inner_w, quadrant)
    return FrameSpec(
        r0=r0,
        c0=c0,
        inner_h=inner_h,
        inner_w=inner_w,
        quadrant=quadrant,
        seed_color=seed_color,
        seed_r=r0 + 1 + seed_lr,
        seed_c=c0 + 1 + seed_lc,
    )


def _ray_specs(frame: FrameSpec) -> list[tuple[str, int, int, int, int, int]]:
    vertical = (
        "v",
        frame.r0 - 1 if frame.quadrant[0] == "t" else frame.r1 + 1,
        frame.mid_c,
        -1 if frame.quadrant[0] == "t" else 1,
        0,
        frame.inner_w,
    )
    horizontal = (
        "h",
        frame.mid_r,
        frame.c0 - 1 if frame.quadrant[1] == "l" else frame.c1 + 1,
        0,
        -1 if frame.quadrant[1] == "l" else 1,
        frame.inner_h,
    )
    return [vertical, horizontal]


def _bbox_cells(frame: FrameSpec) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(frame.r0, frame.r1 + 1)
        for c in range(frame.c0, frame.c1 + 1)
    }


def _free_footprint(frame: FrameSpec) -> set[tuple[int, int]] | None:
    cells = _bbox_cells(frame)
    cells.add((frame.seed_r, frame.seed_c))
    for _axis, start_r, start_c, dr, dc, length in _ray_specs(frame):
        r = start_r
        c = start_c
        for _ in range(length):
            if not (0 <= r < WORKSPACE and 0 <= c < WORKSPACE):
                return None
            cells.add((r, c))
            r += dr
            c += dc
    return cells


def _render_input(shape: tuple[int, int], bg: int, frame_color: int, frames: list[FrameSpec]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for frame in frames:
        for br, bc in _border_cells(frame.r0, frame.c0, frame.r1, frame.c1):
            grid[br, bc] = frame_color
        grid[frame.seed_r, frame.seed_c] = frame.seed_color
    return grid


def _simulate_specs(
    shape: tuple[int, int],
    bg: int,
    frame_color: int,
    frames: list[FrameSpec],
) -> dict[str, object]:
    inp = _render_input(shape, bg, frame_color, frames)
    out = inp.copy()
    frame_mask = np.zeros(shape, dtype=bool)
    for frame in frames:
        for br, bc in _border_cells(frame.r0, frame.c0, frame.r1, frame.c1):
            frame_mask[br, bc] = True

    ray_cells: dict[tuple[int, str], list[tuple[int, int]]] = {}
    blocked_tags: set[tuple[int, str]] = set()
    painted: dict[tuple[int, int], tuple[int, str]] = {}
    boundary_hit = False
    intersection = False

    for idx, frame in enumerate(frames):
        for axis, start_r, start_c, dr, dc, length in _ray_specs(frame):
            cells: list[tuple[int, int]] = []
            r = start_r
            c = start_c
            blocked = False
            for _ in range(length):
                if not (0 <= r < shape[0] and 0 <= c < shape[1]):
                    boundary_hit = True
                    break
                if frame_mask[r, c]:
                    blocked = True
                    break
                if (r, c) in painted:
                    intersection = True
                cells.append((r, c))
                painted[(r, c)] = (idx, axis)
                out[r, c] = frame.seed_color
                r += dr
                c += dc
            if blocked:
                blocked_tags.add((idx, axis))
            elif len(cells) < length:
                boundary_hit = True
            ray_cells[(idx, axis)] = cells

    return {
        "input": inp,
        "output": out,
        "ray_cells": ray_cells,
        "blocked_tags": blocked_tags,
        "boundary_hit": boundary_hit,
        "intersection": intersection,
    }


def _stamp_blocked(mask: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                mask[nr, nc] = True


def _place_free_frame(blocked: np.ndarray, seed_color: int, quadrant: str | None = None) -> tuple[FrameSpec, set[tuple[int, int]]] | None:
    for _ in range(FRAME_ATTEMPTS):
        inner_h = random.choice(INNER_SIZES)
        inner_w = random.choice(INNER_SIZES)
        quad = quadrant or random.choice(QUADRANTS)
        r0 = random.randint(0, WORKSPACE - (inner_h + 2))
        c0 = random.randint(0, WORKSPACE - (inner_w + 2))
        frame = _make_frame(r0, c0, inner_h, inner_w, quad, seed_color)
        footprint = _free_footprint(frame)
        if footprint is None:
            continue
        if any(blocked[r, c] for r, c in footprint):
            continue
        return frame, footprint
    return None


def _pair_target_quadrant(mode: str) -> str:
    if mode == "vertical_block":
        return random.choice(("tl", "tr")) if random.random() < 0.5 else random.choice(("bl", "br"))
    return random.choice(("tl", "bl")) if random.random() < 0.5 else random.choice(("tr", "br"))


def _place_blocked_pair(
    bg: int,
    frame_color: int,
    mode: str,
    seed_colors: list[int],
) -> tuple[list[FrameSpec], set[tuple[int, str]], set[tuple[int, int]]] | None:
    expected_axis = "v" if mode == "vertical_block" else "h"

    for _ in range(PAIR_ATTEMPTS):
        target_quad = _pair_target_quadrant(mode)
        blocker_quad = random.choice(QUADRANTS)

        target_inner_h = random.choice(INNER_SIZES)
        target_inner_w = random.choice(INNER_SIZES)
        blocker_inner_h = random.choice(INNER_SIZES)
        blocker_inner_w = random.choice(INNER_SIZES)
        target_oh = target_inner_h + 2
        target_ow = target_inner_w + 2
        blocker_oh = blocker_inner_h + 2
        blocker_ow = blocker_inner_w + 2

        gap_limit = target_inner_w if expected_axis == "v" else target_inner_h
        if gap_limit <= 1:
            continue
        gap = random.randint(1, gap_limit - 1)

        if mode == "vertical_block":
            if target_quad[0] == "t":
                min_r0 = blocker_oh + gap
                max_r0 = WORKSPACE - target_oh
            else:
                min_r0 = 0
                max_r0 = WORKSPACE - target_oh - blocker_oh - gap
            if min_r0 > max_r0:
                continue
            target_r0 = random.randint(min_r0, max_r0)
            target_c0 = random.randint(0, WORKSPACE - target_ow)
            mid_c = target_c0 + target_ow // 2
            if target_quad[0] == "t":
                blocker_r0 = target_r0 - gap - blocker_oh
            else:
                blocker_r0 = target_r0 + target_oh + gap
            col_lo = max(0, mid_c - (blocker_ow - 1))
            col_hi = min(mid_c, WORKSPACE - blocker_ow)
            if col_lo > col_hi:
                continue
            blocker_c0 = random.randint(col_lo, col_hi)
        else:
            if target_quad[1] == "l":
                min_c0 = blocker_ow + gap
                max_c0 = WORKSPACE - target_ow
            else:
                min_c0 = 0
                max_c0 = WORKSPACE - target_ow - blocker_ow - gap
            if min_c0 > max_c0:
                continue
            target_c0 = random.randint(min_c0, max_c0)
            target_r0 = random.randint(0, WORKSPACE - target_oh)
            mid_r = target_r0 + target_oh // 2
            if target_quad[1] == "l":
                blocker_c0 = target_c0 - gap - blocker_ow
            else:
                blocker_c0 = target_c0 + target_ow + gap
            row_lo = max(0, mid_r - (blocker_oh - 1))
            row_hi = min(mid_r, WORKSPACE - blocker_oh)
            if row_lo > row_hi:
                continue
            blocker_r0 = random.randint(row_lo, row_hi)

        target = _make_frame(target_r0, target_c0, target_inner_h, target_inner_w, target_quad, seed_colors[0])
        blocker = _make_frame(blocker_r0, blocker_c0, blocker_inner_h, blocker_inner_w, blocker_quad, seed_colors[1])
        specs = [target, blocker]
        sim = _simulate_specs((WORKSPACE, WORKSPACE), bg, frame_color, specs)
        expected_tag = {(0, expected_axis)}
        if sim["boundary_hit"] or sim["intersection"]:
            continue
        if sim["blocked_tags"] != expected_tag:
            continue

        footprint = _bbox_cells(target) | _bbox_cells(blocker)
        for cells in sim["ray_cells"].values():
            footprint.update(cells)
        return specs, expected_tag, footprint

    return None


def _sample_pad(cap: int) -> int:
    if cap <= 0:
        return 0
    lo = 1 if cap == 1 else 2
    return random.randint(lo, min(6, cap))


def _crop_with_padding(inp: np.ndarray, out: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    non_bg = np.argwhere(out != bg)
    if non_bg.size == 0:
        return None
    r0, c0, r1, c1 = _bbox(non_bg)

    for _ in range(40):
        top = _sample_pad(r0)
        left = _sample_pad(c0)
        bottom = _sample_pad(inp.shape[0] - 1 - r1)
        right = _sample_pad(inp.shape[1] - 1 - c1)
        rr0 = r0 - top
        cc0 = c0 - left
        rr1 = r1 + bottom
        cc1 = c1 + right
        cropped_inp = inp[rr0 : rr1 + 1, cc0 : cc1 + 1]
        cropped_out = out[rr0 : rr1 + 1, cc0 : cc1 + 1]
        if _most_common_color(cropped_inp) == bg:
            return cropped_inp, cropped_out

    if _most_common_color(inp) == bg:
        return inp.copy(), out.copy()
    return None


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(2, 4)
    mode = random.choices(
        population=["free", "vertical_block", "horizontal_block"],
        weights=[2, 1, 1],
        k=1,
    )[0]
    colors = random.sample(range(10), 2 + n_frames)
    bg = colors[0]
    frame_color = colors[1]
    seed_colors = colors[2:]

    blocked = np.zeros((WORKSPACE, WORKSPACE), dtype=bool)
    frames: list[FrameSpec] = []
    expected_blocked: set[tuple[int, str]] = set()

    if mode == "free":
        for seed_color in seed_colors:
            placed = _place_free_frame(blocked, seed_color)
            if placed is None:
                return None
            frame, footprint = placed
            frames.append(frame)
            _stamp_blocked(blocked, footprint)
    else:
        pair = _place_blocked_pair(bg, frame_color, mode, seed_colors[:2])
        if pair is None:
            return None
        pair_frames, expected_blocked, footprint = pair
        frames.extend(pair_frames)
        _stamp_blocked(blocked, footprint)
        for seed_color in seed_colors[2:]:
            placed = _place_free_frame(blocked, seed_color)
            if placed is None:
                return None
            frame, free_footprint = placed
            frames.append(frame)
            _stamp_blocked(blocked, free_footprint)

    sim = _simulate_specs((WORKSPACE, WORKSPACE), bg, frame_color, frames)
    if sim["boundary_hit"] or sim["intersection"]:
        return None
    if mode == "free":
        if sim["blocked_tags"]:
            return None
    elif sim["blocked_tags"] != expected_blocked:
        return None

    crop = _crop_with_padding(sim["input"], sim["output"], bg)
    if crop is None:
        return None
    inp, out = crop
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    if np.array_equal(inp, out):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
