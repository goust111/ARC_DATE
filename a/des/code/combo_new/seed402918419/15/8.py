import random
from typing import Dict, List, Optional, Tuple

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
DIR_HORIZONTAL = "h"
DIR_VERTICAL = "v"
SIDES = ("top", "bottom", "left", "right")
LEGEND_STYLES = ("centered", "anchored")
GEN_RETRIES = 500


def main(grid: np.ndarray) -> np.ndarray:
    """
    Refill each hollow frame interior with key-colored stripes read from matching edge legends.

    Input structure:
    - The background is black.
    - Several large hollow rectangular frames with distinct border colors sit away from the edge.
    - Inside each frame, the interior is otherwise black except for a 2x2 solid key block at the interior top-left corner.
    - Small 3x3 legend patches lie on the grid edge; each legend uses one non-background color and shows either a horizontal or vertical stripe sample for that color.

    Rule:
    - For each frame, read the color C of its interior 2x2 key block.
    - Find the edge legend drawn with the same color C and determine whether its sample stripe is horizontal or vertical.
    - Clear the whole frame interior, then refill it with color C using every other interior row for a horizontal legend or every other interior column for a vertical legend, always starting from the first interior row or column.
    - Keep the frame borders and all legends unchanged.

    Key insight:
    - The 2x2 block is only an index into the legend set; the legend's stripe direction, not the frame border color, determines the repainting pattern.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    legends = _find_legends(g, bg)
    frames = _find_frames(g, bg)

    assert legends, "main: expected at least one edge legend"
    assert frames, "main: expected at least one hollow frame"

    out = g.copy()
    for r0, c0, r1, c1, key_color in frames:
        assert key_color in legends, "main: every frame key color must have a legend"
        out[r0 + 1 : r1, c0 + 1 : c1] = bg
        if legends[key_color] == DIR_HORIZONTAL:
            out[r0 + 1 : r1 : 2, c0 + 1 : c1] = key_color
        else:
            out[r0 + 1 : r1, c0 + 1 : c1 : 2] = key_color
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid instance")


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    height = rng.randint(16, 30)
    width = rng.randint(16, 30)

    max_frames = 4 if min(height, width) >= 24 else 3 if min(height, width) >= 18 else 2
    frame_count = rng.randint(1, max_frames)

    spare_colors = 9 - 2 * frame_count
    distractor_count = 1 if spare_colors > 0 and rng.random() < 0.35 else 0
    total_legends = frame_count + distractor_count
    if 2 * frame_count + distractor_count > len(COLORS):
        return None

    palette = rng.sample(COLORS, 2 * frame_count + distractor_count)
    frame_colors = palette[:frame_count]
    key_colors = palette[frame_count : 2 * frame_count]
    distractor_colors = palette[2 * frame_count :]

    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    legend_specs: Dict[int, str] = {}
    all_legend_colors = list(key_colors) + list(distractor_colors)
    rng.shuffle(all_legend_colors)
    for color in all_legend_colors:
        direction = rng.choice((DIR_HORIZONTAL, DIR_VERTICAL))
        if not _place_legend(grid, occupied, color, direction, rng):
            return None
        legend_specs[color] = direction

    frame_specs = []
    for idx, (frame_color, key_color) in enumerate(zip(frame_colors, key_colors)):
        if frame_count == 1:
            min_side, max_side = 6, 12
        elif frame_count == 2:
            min_side, max_side = 5, 10
        else:
            min_side, max_side = 5, 8

        max_h = min(max_side, height - 6)
        max_w = min(max_side, width - 6)
        if max_h < min_side or max_w < min_side:
            return None

        placed = False
        for _ in range(80):
            fh = rng.randint(min_side, max_h)
            fw = rng.randint(min_side, max_w)
            if height - fh - 3 < 3 or width - fw - 3 < 3:
                continue
            r0 = rng.randint(3, height - fh - 3)
            c0 = rng.randint(3, width - fw - 3)
            r1 = r0 + fh - 1
            c1 = c0 + fw - 1
            if not _bbox_is_free(occupied, r0, c0, r1, c1, pad=1):
                continue
            _draw_frame(grid, r0, c0, r1, c1, frame_color, key_color)
            _mark_bbox(occupied, r0, c0, r1, c1, pad=1)
            frame_specs.append((r0, c0, r1, c1, key_color))
            placed = True
            break
        if not placed:
            return None

    detected_legends = _find_legends(grid, BG)
    if set(detected_legends) != set(all_legend_colors):
        return None
    if any(detected_legends[color] != direction for color, direction in legend_specs.items()):
        return None

    detected_frames = _find_frames(grid, BG)
    expected_frames = sorted(frame_specs)
    if detected_frames != expected_frames:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def _place_legend(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    direction: str,
    rng: random.Random,
) -> bool:
    height, width = grid.shape
    candidates: List[Tuple[str, int, int]] = []
    for side in SIDES:
        if side in ("top", "bottom"):
            if width < 5:
                continue
            for c0 in range(1, width - 3):
                r0 = 0 if side == "top" else height - 3
                candidates.append((side, r0, c0))
        else:
            if height < 5:
                continue
            for r0 in range(1, height - 3):
                c0 = 0 if side == "left" else width - 3
                candidates.append((side, r0, c0))
    rng.shuffle(candidates)

    for side, r0, c0 in candidates:
        r1, c1 = r0 + 2, c0 + 2
        if not _bbox_is_free(occupied, r0, c0, r1, c1, pad=1):
            continue
        style = rng.choice(LEGEND_STYLES)
        patch = _legend_patch(color, direction, side, style)
        grid[r0 : r0 + 3, c0 : c0 + 3] = patch
        _mark_bbox(occupied, r0, c0, r1, c1, pad=1)
        return True
    return False


def _legend_patch(color: int, direction: str, side: str, style: str) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    if style == "centered":
        if direction == DIR_HORIZONTAL:
            patch[1, :] = color
        else:
            patch[:, 1] = color
        return patch

    if direction == DIR_HORIZONTAL:
        patch[1, :] = color
        if side == "top":
            patch[0, :] = color
        elif side == "bottom":
            patch[2, :] = color
        elif side == "left":
            patch[0, 0] = color
            patch[2, 0] = color
        else:
            patch[0, 2] = color
            patch[2, 2] = color
        return patch

    patch[:, 1] = color
    if side == "top":
        patch[0, 0] = color
        patch[0, 2] = color
    elif side == "bottom":
        patch[2, 0] = color
        patch[2, 2] = color
    elif side == "left":
        patch[:, 0] = color
    else:
        patch[:, 2] = color
    return patch


def _draw_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    frame_color: int,
    key_color: int,
) -> None:
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    grid[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3] = key_color


def _find_legends(grid: np.ndarray, bg: int) -> Dict[int, str]:
    height, width = grid.shape
    legends: Dict[int, str] = {}
    for r0 in range(height - 2):
        for c0 in range(width - 2):
            if r0 not in (0, height - 3) and c0 not in (0, width - 3):
                continue
            patch = grid[r0 : r0 + 3, c0 : c0 + 3]
            colors = [int(v) for v in np.unique(patch) if int(v) != bg]
            if len(colors) != 1:
                continue
            color = colors[0]
            mask = patch == color
            if not bool(mask[1, 1]):
                continue
            row_counts = mask.sum(axis=1)
            col_counts = mask.sum(axis=0)
            direction: Optional[str]
            if int(row_counts[1]) == 3 and int(col_counts[1]) < 3:
                direction = DIR_HORIZONTAL
            elif int(col_counts[1]) == 3 and int(row_counts[1]) < 3:
                direction = DIR_VERTICAL
            else:
                direction = None
            if direction is None:
                continue
            if color in legends:
                assert legends[color] == direction, "legend color must not encode two directions"
            legends[color] = direction
    return legends


def _find_frames(grid: np.ndarray, bg: int) -> List[Tuple[int, int, int, int, int]]:
    frames: List[Tuple[int, int, int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            rows = [p[0] for p in comp]
            cols = [p[1] for p in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 4 or width < 4:
                continue

            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((height, width), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not bool((patch[border] == color).all()):
                continue
            if int(np.count_nonzero(patch == color)) != int(border.sum()):
                continue

            inner = patch[1:-1, 1:-1]
            if inner.shape[0] < 2 or inner.shape[1] < 2:
                continue
            key_patch = inner[:2, :2]
            key_colors = [int(v) for v in np.unique(key_patch) if int(v) != bg]
            if len(key_colors) != 1:
                continue
            key_color = key_colors[0]
            if key_color == color or bool((key_patch == bg).any()):
                continue

            if inner.size > 4:
                rest = inner.copy()
                rest[:2, :2] = bg
                if bool((rest != bg).any()):
                    continue

            frames.append((r0, c0, r1, c1, key_color))
    frames.sort()
    return frames


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_is_free(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> bool:
    height, width = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(height - 1, r1 + pad)
    cc1 = min(width - 1, c1 + pad)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_bbox(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> None:
    height, width = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(height - 1, r1 + pad)
    cc1 = min(width - 1, c1 + pad)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))
