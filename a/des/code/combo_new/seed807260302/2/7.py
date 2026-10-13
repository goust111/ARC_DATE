from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
TEXTURE_COLORS = list(range(3, 10))
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ARC_MAX = 30
MAX_GENERATION_ATTEMPTS = 500


def _perimeter_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _side_cells(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _missing_runs(
    missing: set[tuple[int, int]], cells: list[tuple[int, int]]
) -> tuple[list[list[tuple[int, int]]], set[tuple[int, int]]] | None:
    runs: list[list[tuple[int, int]]] = []
    used: set[tuple[int, int]] = set()
    i = 0
    while i < len(cells):
        if cells[i] not in missing:
            i += 1
            continue
        start = i
        while i < len(cells) and cells[i] in missing:
            i += 1
        run = cells[start:i]
        if len(run) != 2:
            return None
        runs.append(run)
        used.update(run)
    return runs, used


def _rectangle_info(g: np.ndarray, bbox: tuple[int, int, int, int]) -> dict | None:
    r0, c0, r1, c1 = bbox
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return None

    missing: set[tuple[int, int]] = set()
    for r, c in _perimeter_cells(bbox):
        value = int(g[r, c])
        if value == BLUE:
            continue
        if value != BG:
            return None
        missing.add((r, c))

    if len(missing) not in (0, 2, 4):
        return None
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if missing & corners:
        return None

    window_dirs: list[str] = []
    if missing:
        used: set[tuple[int, int]] = set()
        for side, direction in (
            ("top", "horizontal"),
            ("bottom", "horizontal"),
            ("left", "vertical"),
            ("right", "vertical"),
        ):
            found = _missing_runs(missing, _side_cells(bbox, side))
            if found is None:
                return None
            runs, side_used = found
            used.update(side_used)
            window_dirs.extend([direction] * len(runs))
        if used != missing or not (1 <= len(window_dirs) <= 2):
            return None

    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    if missing:
        if not np.all(interior == BG):
            return None
        return {"kind": "target", "bbox": bbox, "window_dirs": window_dirs}

    if np.all(interior == BG):
        return None
    if np.any(interior == BLUE) or np.any(interior == RED):
        return None
    if len(set(map(int, interior.reshape(-1)))) < 2:
        return None
    return {"kind": "sample", "bbox": bbox, "pattern": interior.copy()}


def _find_rectangles(g: np.ndarray) -> list[dict]:
    h, w = map(int, g.shape)
    blue = g == BLUE
    found: list[dict] = []
    seen: set[tuple[int, int, int, int]] = set()

    for r0 in range(h - 4):
        for r1 in range(r0 + 4, h):
            cols = [int(c) for c in np.flatnonzero(blue[r0] & blue[r1])]
            for i, c0 in enumerate(cols):
                for c1 in cols[i + 1 :]:
                    if c1 - c0 + 1 < 5:
                        continue
                    bbox = (r0, c0, r1, c1)
                    if bbox in seen:
                        continue
                    info = _rectangle_info(g, bbox)
                    if info is not None:
                        seen.add(bbox)
                        found.append(info)

    found.sort(key=lambda item: item["bbox"])
    return found


def _red_segments(g: np.ndarray) -> list[dict]:
    mask = g == RED
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    segments: list[dict] = []

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for dr, dc in DIRS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))

        if len(coords) != 2:
            continue
        (r0, c0), (r1, c1) = coords
        if r0 == r1 and abs(c0 - c1) == 1:
            orientation = "horizontal"
        elif c0 == c1 and abs(r0 - r1) == 1:
            orientation = "vertical"
        else:
            continue
        segments.append({"coords": set(coords), "orientation": orientation})

    return segments


def _touches_frame(segment: dict, bbox: tuple[int, int, int, int]) -> bool:
    frame = set(_perimeter_cells(bbox))
    for r, c in segment["coords"]:
        for dr, dc in DIRS4:
            if (r + dr, c + dc) in frame:
                return True
    return False


def _read_texture_samples(samples: list[dict], red_segments: list[dict]) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    by_direction: dict[str, np.ndarray] = {}
    assigned: set[int] = set()

    for i, sample in enumerate(samples):
        marker_dirs = {
            segment["orientation"]
            for segment in red_segments
            if _touches_frame(segment, sample["bbox"])
        }
        if len(marker_dirs) != 1:
            continue
        direction = next(iter(marker_dirs))
        assert direction not in by_direction, "each marked direction must have one texture sample"
        by_direction[direction] = sample["pattern"]
        assigned.add(i)

    assert "horizontal" in by_direction, "missing horizontal texture sample"
    assert "vertical" in by_direction, "missing vertical texture sample"

    unused = [sample["pattern"] for i, sample in enumerate(samples) if i not in assigned]
    assert len(unused) <= 1, "mixed texture sample must be the unique unused sample"
    mixed = unused[0] if unused else None
    return by_direction, mixed


def _target_direction(window_dirs: list[str]) -> str:
    unique = set(window_dirs)
    if unique == {"horizontal"}:
        return "horizontal"
    if unique == {"vertical"}:
        return "vertical"
    assert unique == {"horizontal", "vertical"}, "window directions must be horizontal, vertical, or mixed"
    return "mixed"


def _tile_fill(out: np.ndarray, bbox: tuple[int, int, int, int], pattern: np.ndarray) -> None:
    r0, c0, r1, c1 = bbox
    ph, pw = map(int, pattern.shape)
    for rr, r in enumerate(range(r0 + 1, r1)):
        for cc, c in enumerate(range(c0 + 1, c1)):
            out[r, c] = int(pattern[rr % ph, cc % pw])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each windowed blue region with the texture indicated by its window direction.

    Input structure:
    - The background is black.
    - In one corner, complete blue sample frames contain two-color repeated textures; red length-2 markers beside marked samples encode horizontal and vertical directions.
    - Elsewhere, blue rectangular region borders have one or two length-2 black window gaps on horizontal or vertical sides, and their interiors are black.

    Rule:
    - Read the marked sample frames to map horizontal and vertical red markers to texture tiles; if a third sample is present, it is the unused mixed texture.
    - For each windowed blue region, classify its window direction as horizontal, vertical, or mixed when both directions occur.
    - Tile the matching texture over the region interior, aligned at that interior's top-left cell.
    - Leave blue borders, red markers, samples, and black window gaps unchanged.

    Key insight:
    - The small boundary gaps are symbolic windows: their side orientation selects a texture through the nearby red-marker sample legend before the texture is tiled into the enclosed region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= ARC_MAX, "ARC grids must be at most 30 by 30"

    rectangles = _find_rectangles(g)
    samples = [item for item in rectangles if item["kind"] == "sample"]
    targets = [item for item in rectangles if item["kind"] == "target"]
    assert len(samples) >= 2, "expected at least two texture samples"
    assert targets, "expected at least one windowed region"

    direction_patterns, mixed_pattern = _read_texture_samples(samples, _red_segments(g))

    out = g.copy()
    for target in targets:
        direction = _target_direction(target["window_dirs"])
        if direction == "mixed":
            assert mixed_pattern is not None, "mixed windows require a third unused texture sample"
            pattern = mixed_pattern
        else:
            pattern = direction_patterns[direction]
        _tile_fill(out, target["bbox"], pattern)

    return out


# --------------- generator ---------------


def _draw_frame(g: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    g[r0, c0 : c1 + 1] = BLUE
    g[r1, c0 : c1 + 1] = BLUE
    g[r0 : r1 + 1, c0] = BLUE
    g[r0 : r1 + 1, c1] = BLUE


def _texture(kind: str, height: int, width: int, colors: tuple[int, int]) -> np.ndarray:
    a, b = colors
    pat = np.full((height, width), a, dtype=int)
    for r in range(height):
        for c in range(width):
            if kind == "horizontal_stripes":
                pat[r, c] = a if r % 2 == 0 else b
            elif kind == "vertical_stripes":
                pat[r, c] = a if c % 2 == 0 else b
            else:
                pat[r, c] = a if (r + c) % 2 == 0 else b
    return pat


def _expanded(bbox: tuple[int, int, int, int], h: int, w: int, pad: int = 1) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _reserve(reserved: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = map(int, reserved.shape)
    r0, c0, r1, c1 = _expanded(bbox, h, w, pad)
    reserved[r0 : r1 + 1, c0 : c1 + 1] = True


def _free(reserved: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    h, w = map(int, reserved.shape)
    r0, c0, r1, c1 = bbox
    if r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
        return False
    er0, ec0, er1, ec1 = _expanded(bbox, h, w, pad)
    return not bool(np.any(reserved[er0 : er1 + 1, ec0 : ec1 + 1]))


def _draw_sample(
    g: np.ndarray,
    reserved: np.ndarray,
    r0: int,
    c0: int,
    role: str,
    pattern: np.ndarray,
) -> None:
    ih, iw = map(int, pattern.shape)
    height = ih + 2
    width = iw + 2
    _draw_frame(g, r0, c0, height, width)
    g[r0 + 1 : r0 + 1 + ih, c0 + 1 : c0 + 1 + iw] = pattern

    r1 = r0 + height - 1
    c1 = c0 + width - 1
    marker_bbox = (r0, c0, r1, c1)
    if role == "horizontal":
        mr = r0 + 2
        mc = c1 + 1
        g[mr, mc : mc + 2] = RED
        marker_bbox = (r0, c0, r1, mc + 1)
    elif role == "vertical":
        mr = r0 + 2
        mc = c1 + 1
        g[mr : mr + 2, mc] = RED
        marker_bbox = (r0, c0, r1, mc)

    _reserve(reserved, marker_bbox, pad=1)


def _sample_windows(role: str, height: int, width: int) -> list[tuple[str, int]]:
    windows: list[tuple[str, int]] = []
    if role == "horizontal":
        sides = ["top", "bottom"]
        random.shuffle(sides)
        for side in sides[: random.randint(1, 2)]:
            windows.append((side, random.randint(1, width - 3)))
    elif role == "vertical":
        sides = ["left", "right"]
        random.shuffle(sides)
        for side in sides[: random.randint(1, 2)]:
            windows.append((side, random.randint(1, height - 3)))
    else:
        hside = random.choice(["top", "bottom"])
        vside = random.choice(["left", "right"])
        windows.append((hside, random.randint(1, width - 3)))
        windows.append((vside, random.randint(1, height - 3)))
    return windows


def _erase_window(g: np.ndarray, r0: int, c0: int, height: int, width: int, window: tuple[str, int]) -> None:
    side, offset = window
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    if side == "top":
        g[r0, c0 + offset : c0 + offset + 2] = BG
    elif side == "bottom":
        g[r1, c0 + offset : c0 + offset + 2] = BG
    elif side == "left":
        g[r0 + offset : r0 + offset + 2, c0] = BG
    else:
        g[r0 + offset : r0 + offset + 2, c1] = BG


def _draw_target(g: np.ndarray, r0: int, c0: int, height: int, width: int, role: str) -> None:
    _draw_frame(g, r0, c0, height, width)
    for window in _sample_windows(role, height, width):
        _erase_window(g, r0, c0, height, width, window)


def _gen_attempt() -> np.ndarray | None:
    include_mixed = random.random() < 0.7
    roles = ["horizontal", "vertical"] + (["mixed"] if include_mixed else [])

    height = random.randint(24 if include_mixed else 21, ARC_MAX)
    width = random.randint(24, ARC_MAX)
    g = np.full((height, width), BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    sample_roles = roles[:]
    random.shuffle(sample_roles)
    texture_kinds = random.sample(["horizontal_stripes", "vertical_stripes", "checkerboard"], len(sample_roles))
    colors = random.sample(TEXTURE_COLORS, 2 * len(sample_roles))

    row = 1
    for idx, role in enumerate(sample_roles):
        int_h = 4
        int_w = random.choice([4, 6])
        color_pair = (colors[2 * idx], colors[2 * idx + 1])
        pattern = _texture(texture_kinds[idx], int_h, int_w, color_pair)
        _draw_sample(g, reserved, row, 1, role, pattern)
        row += int_h + 3

    target_roles = roles[:]
    extra_count = random.randint(0, 2)
    for _ in range(extra_count):
        target_roles.append(random.choice(roles))
    random.shuffle(target_roles)

    for role in target_roles:
        placed = False
        for _ in range(200):
            th = random.randint(5, 8)
            tw = random.randint(5, 9)
            r0 = random.randint(1, height - th - 1)
            c0 = random.randint(1, width - tw - 1)
            bbox = (r0, c0, r0 + th - 1, c0 + tw - 1)
            if not _free(reserved, bbox, pad=1):
                continue
            _draw_target(g, r0, c0, th, tw, role)
            _reserve(reserved, bbox, pad=1)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(g)
    except AssertionError:
        return None
    if out.shape != g.shape or max(out.shape) > ARC_MAX:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
