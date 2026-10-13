import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


BG = 0
RED = 2
GREY = 5
FRAME_COLORS = (1, 3, 4, 6, 7, 8, 9)
CORNERS = ("tl", "tr", "bl", "br")
SIDES = ("top", "right", "bottom", "left")
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: List[Tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: Sequence[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_ring(points: Sequence[Tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    pts = set(points)
    if len(pts) != 2 * (h + w) - 4:
        return False
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in pts):
                return False
    return True


def _find_source_box(grid: np.ndarray) -> Tuple[int, int, int, int]:
    candidates: List[Tuple[int, Tuple[int, int, int, int]]] = []
    for pts in _components(grid == GREY):
        if not _is_rect_ring(pts):
            continue
        r0, c0, r1, c1 = _bbox(pts)
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0:
            continue
        if np.any(inner == BG):
            continue
        if len(np.unique(inner)) < 2:
            continue
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        candidates.append((area, (r0, c0, r1, c1)))
    if not candidates:
        raise ValueError("source frame not found")
    candidates.sort()
    return candidates[-1][1]


def _find_red_side(grid: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[str]:
    h, w = grid.shape
    r0, c0, r1, c1 = box
    matches: List[str] = []
    if r0 - 1 >= 0 and np.all(grid[r0 - 1, c0 : c1 + 1] == RED):
        matches.append("top")
    if r1 + 1 < h and np.all(grid[r1 + 1, c0 : c1 + 1] == RED):
        matches.append("bottom")
    if c0 - 1 >= 0 and np.all(grid[r0 : r1 + 1, c0 - 1] == RED):
        matches.append("left")
    if c1 + 1 < w and np.all(grid[r0 : r1 + 1, c1 + 1] == RED):
        matches.append("right")
    if len(matches) != 1:
        return None
    return matches[0]


def _find_frames(grid: np.ndarray) -> List[Dict[str, object]]:
    frames: List[Dict[str, object]] = []
    for color in np.unique(grid):
        col = int(color)
        if col in (BG, RED, GREY):
            continue
        for pts in _components(grid == col):
            if not _is_rect_ring(pts):
                continue
            box = _bbox(pts)
            r0, c0, r1, c1 = box
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0 or not np.all(inner == BG):
                continue
            side = _find_red_side(grid, box)
            if side is None:
                continue
            frames.append({"color": col, "box": box, "side": side})
    return frames


def _source_corner_colors(grid: np.ndarray, source_box: Tuple[int, int, int, int]) -> Dict[int, str]:
    h, w = grid.shape
    r0, c0, r1, c1 = source_box
    checks = {
        "tl": (r0 - 1, c0 - 1),
        "tr": (r0 - 1, c1 + 1),
        "bl": (r1 + 1, c0 - 1),
        "br": (r1 + 1, c1 + 1),
    }
    result: Dict[int, str] = {}
    for corner, (r, c) in checks.items():
        if not (0 <= r < h and 0 <= c < w):
            continue
        color = int(grid[r, c])
        if color in (BG, RED, GREY):
            continue
        result[color] = corner
    return result


def _crop_from_corner(src: np.ndarray, corner: str, height: int, width: int) -> np.ndarray:
    if corner == "tl":
        return src[:height, :width].copy()
    if corner == "tr":
        return src[:height, src.shape[1] - width :].copy()
    if corner == "bl":
        return src[src.shape[0] - height :, :width].copy()
    if corner == "br":
        return src[src.shape[0] - height :, src.shape[1] - width :].copy()
    raise ValueError(f"unknown corner: {corner}")


def _rotation_k(side: str) -> int:
    # The red strip says which canvas side is the frame's current "up".
    return {"top": 0, "right": 3, "bottom": 2, "left": 1}[side]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each colored sampling frame with a rotated corner crop from the grey-framed source pattern.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[out == RED] = BG

    source_box = _find_source_box(g)
    corner_by_color = _source_corner_colors(g, source_box)
    frames = _find_frames(g)

    sr0, sc0, sr1, sc1 = source_box
    src = g[sr0 + 1 : sr1, sc0 + 1 : sc1]

    for frame in frames:
        color = int(frame["color"])
        if color not in corner_by_color:
            raise ValueError(f"missing source marker for frame color {color}")
        r0, c0, r1, c1 = frame["box"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        patch = _crop_from_corner(src, corner_by_color[color], inner_h, inner_w)
        rotated = np.rot90(patch, _rotation_k(str(frame["side"])))
        if rotated.shape != (inner_h, inner_w):
            raise ValueError("rotated patch does not fit frame interior")
        out[r0 + 1 : r1, c0 + 1 : c1] = rotated

    return out


def _make_source_pattern(rng: random.Random, height: int, width: int, palette: Sequence[int]) -> np.ndarray:
    n = len(palette)
    a = rng.randint(1, 3)
    b = rng.randint(1, 3)
    phase = rng.randint(0, n - 1)
    src = np.empty((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            idx = (a * r + b * c + (r // 2) + (c // 2) + ((r ^ c) & 1) + phase) % n
            src[r, c] = palette[idx]

    for _ in range(rng.randint(2, 4)):
        col = rng.choice(tuple(palette))
        rh = rng.randint(1, max(1, height // 2))
        rw = rng.randint(1, max(1, width // 2))
        r0 = rng.randint(0, height - rh)
        c0 = rng.randint(0, width - rw)
        src[r0 : r0 + rh, c0 : c0 + rw] = col
    return src


def _patch_is_informative(patch: np.ndarray, side: str) -> bool:
    if len(np.unique(patch)) < 2:
        return False
    k = _rotation_k(side)
    rotated = np.rot90(patch, k)
    if rotated.shape != patch.shape:
        return False
    if k and np.array_equal(rotated, patch):
        return False
    return True


def _footprint_size(inner_h: int, inner_w: int, side: str) -> Tuple[int, int]:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    foot_h = outer_h + (1 if side in ("top", "bottom") else 0)
    foot_w = outer_w + (1 if side in ("left", "right") else 0)
    return foot_h, foot_w


def _box_clear(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _mark_box(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _place_frame(
    grid: np.ndarray,
    occupied: np.ndarray,
    rng: random.Random,
    source_left: bool,
    source_box: Tuple[int, int, int, int],
    spec: Dict[str, object],
) -> bool:
    h, w = grid.shape
    sr0, sc0, sr1, sc1 = source_box
    side = str(spec["side"])
    color = int(spec["color"])
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    foot_h, foot_w = _footprint_size(inner_h, inner_w, side)

    if source_left:
        foot_c_min = sc1 + 3
        foot_c_max = w - foot_w - 2
    else:
        foot_c_min = 1
        foot_c_max = sc0 - foot_w - 3
    foot_r_min = 1
    foot_r_max = h - foot_h - 2
    if foot_c_min > foot_c_max or foot_r_min > foot_r_max:
        return False

    for _ in range(250):
        foot_r0 = rng.randint(foot_r_min, foot_r_max)
        foot_c0 = rng.randint(foot_c_min, foot_c_max)
        foot_r1 = foot_r0 + foot_h - 1
        foot_c1 = foot_c0 + foot_w - 1
        if not _box_clear(occupied, foot_r0, foot_c0, foot_r1, foot_c1, pad=1):
            continue

        frame_r0 = foot_r0 + (1 if side == "top" else 0)
        frame_c0 = foot_c0 + (1 if side == "left" else 0)
        frame_r1 = frame_r0 + outer_h - 1
        frame_c1 = frame_c0 + outer_w - 1

        grid[frame_r0, frame_c0 : frame_c1 + 1] = color
        grid[frame_r1, frame_c0 : frame_c1 + 1] = color
        grid[frame_r0 : frame_r1 + 1, frame_c0] = color
        grid[frame_r0 : frame_r1 + 1, frame_c1] = color

        if side == "top":
            grid[frame_r0 - 1, frame_c0 : frame_c1 + 1] = RED
        elif side == "bottom":
            grid[frame_r1 + 1, frame_c0 : frame_c1 + 1] = RED
        elif side == "left":
            grid[frame_r0 : frame_r1 + 1, frame_c0 - 1] = RED
        else:
            grid[frame_r0 : frame_r1 + 1, frame_c1 + 1] = RED

        _mark_box(occupied, foot_r0, foot_c0, foot_r1, foot_c1)
        spec["box"] = (frame_r0, frame_c0, frame_r1, frame_c1)
        return True
    return False


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    frame_count = rng.randint(2, 4)
    corners = rng.sample(CORNERS, frame_count)
    sides = rng.sample(SIDES, frame_count)
    colors = rng.sample(FRAME_COLORS, frame_count)

    source_colors = [c for c in FRAME_COLORS if c not in colors]
    if len(source_colors) < 3:
        return None

    src_h = rng.randint(6, 9)
    src_w = rng.randint(6, 9)

    specs: List[Dict[str, object]] = []
    for corner, side, color in zip(corners, sides, colors):
        if side in ("left", "right"):
            size = rng.randint(2, min(4, src_h, src_w))
            inner_h = size
            inner_w = size
        else:
            inner_h = rng.randint(2, min(4, src_h))
            inner_w = rng.randint(2, min(4, src_w))
            if inner_h == inner_w and min(src_h, src_w) >= 3 and rng.random() < 0.6:
                if inner_w < min(4, src_w):
                    inner_w += 1
                elif inner_h > 2:
                    inner_h -= 1
        specs.append(
            {
                "corner": corner,
                "side": side,
                "color": color,
                "inner_h": inner_h,
                "inner_w": inner_w,
            }
        )

    src = None
    for _ in range(60):
        candidate = _make_source_pattern(rng, src_h, src_w, source_colors)
        signatures = set()
        ok = True
        for spec in specs:
            patch = _crop_from_corner(
                candidate,
                str(spec["corner"]),
                int(spec["inner_h"]),
                int(spec["inner_w"]),
            )
            if not _patch_is_informative(patch, str(spec["side"])):
                ok = False
                break
            rotated = np.rot90(patch, _rotation_k(str(spec["side"])))
            key = (rotated.shape, rotated.tobytes())
            if key in signatures:
                ok = False
                break
            signatures.add(key)
        if ok:
            src = candidate
            break
    if src is None:
        return None

    max_foot_h = max(_footprint_size(int(s["inner_h"]), int(s["inner_w"]), str(s["side"]))[0] for s in specs)
    max_foot_w = max(_footprint_size(int(s["inner_h"]), int(s["inner_w"]), str(s["side"]))[1] for s in specs)
    min_h = max(20, src_h + 6, max_foot_h * (2 if frame_count >= 3 else 1) + 5)
    min_w = max(24, src_w + 10, max_foot_w * (2 if frame_count >= 3 else 1) + src_w + 8)
    if min_h > 28 or min_w > 34:
        return None
    height = rng.randint(min_h, 28)
    width = rng.randint(min_w, 34)

    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    source_left = rng.choice([True, False])
    outer_h = src_h + 2
    outer_w = src_w + 2
    gr0 = rng.randint(2, height - outer_h - 3)
    if source_left:
        gc0 = rng.randint(2, 4)
    else:
        gc0 = rng.randint(width - outer_w - 5, width - outer_w - 3)
    gr1 = gr0 + outer_h - 1
    gc1 = gc0 + outer_w - 1
    source_box = (gr0, gc0, gr1, gc1)

    grid[gr0, gc0 : gc1 + 1] = GREY
    grid[gr1, gc0 : gc1 + 1] = GREY
    grid[gr0 : gr1 + 1, gc0] = GREY
    grid[gr0 : gr1 + 1, gc1] = GREY
    grid[gr0 + 1 : gr1, gc0 + 1 : gc1] = src
    _mark_box(occupied, gr0 - 1, gc0 - 1, gr1 + 1, gc1 + 1)

    marker_pos = {
        "tl": (gr0 - 1, gc0 - 1),
        "tr": (gr0 - 1, gc1 + 1),
        "bl": (gr1 + 1, gc0 - 1),
        "br": (gr1 + 1, gc1 + 1),
    }
    for spec in specs:
        mr, mc = marker_pos[str(spec["corner"])]
        grid[mr, mc] = int(spec["color"])

    order = sorted(
        specs,
        key=lambda s: _footprint_size(int(s["inner_h"]), int(s["inner_w"]), str(s["side"]))[0]
        * _footprint_size(int(s["inner_h"]), int(s["inner_w"]), str(s["side"]))[1],
        reverse=True,
    )
    for spec in order:
        if not _place_frame(grid, occupied, rng, source_left, source_box, spec):
            return None

    if len(_find_frames(grid)) != frame_count:
        return None
    if set(_source_corner_colors(grid, source_box)) != set(colors):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.any(out == RED):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
