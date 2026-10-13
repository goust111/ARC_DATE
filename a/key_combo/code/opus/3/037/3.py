from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5

FRAME_SIZE = 7
INNER_SIZE = 5
SPRITE_SIZE = 3

ALL_CORNERS = ("NW", "NE", "SE", "SW")
FRAME_CORNERS = ("NW", "NE", "SW", "SE")
CORNER_ORDER_CW = ("NW", "NE", "SE", "SW")
CORNER_TO_INNER = {
    "NW": (0, 0),
    "NE": (0, INNER_SIZE - 1),
    "SW": (INNER_SIZE - 1, 0),
    "SE": (INNER_SIZE - 1, INNER_SIZE - 1),
}

GEN_BODY_COLORS = (1, 2, 3, 6, 7, 8, 9)
GEN_FRAME_COUNT_CHOICES = (3, 4)
GEN_GRID_ROWS = tuple(range(20, 29))
GEN_GRID_COLS = tuple(range(20, 29))
GEN_FRAME_BORDER = 1
GEN_FRAME_GAP = 2
GEN_MAX_ATTEMPTS = 300


def get_bg(grid: np.ndarray) -> int:
    """Return the most frequent color in the grid."""
    return int(Counter(map(int, np.ravel(grid))).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    """Yield 4-neighbors using the ARC (row, col) convention."""
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Return the inclusive bounding box of an (N,2) coord array."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
    monochromatic: bool = True,
    colors: list[int] | tuple[int, ...] | None = None,
) -> list[dict]:
    """Small self-contained subset of arc_utils.find_components."""
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = get_bg(g)
    bg = int(bg)
    if connectivity != 4:
        raise ValueError("this task only uses 4-connectivity")

    def component_dicts(mask: np.ndarray, color: int | None) -> list[dict]:
        visited = np.zeros(mask.shape, dtype=bool)
        comps: list[dict] = []
        rows, cols = mask.shape
        for r in range(rows):
            for c in range(cols):
                if not mask[r, c] or visited[r, c]:
                    continue
                stack = [(r, c)]
                visited[r, c] = True
                coords: list[tuple[int, int]] = []
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for nr, nc in neighbors4(rr, cc):
                        if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                coord_arr = np.array(coords, dtype=int)
                comps.append(
                    {
                        "color": color,
                        "coords": coord_arr,
                        "bbox": bbox_coords(coord_arr),
                        "area": int(coord_arr.shape[0]),
                    }
                )
        return comps

    if monochromatic:
        if colors is None:
            colors_iter = [int(color) for color in np.unique(g) if int(color) != bg]
        else:
            colors_iter = [int(color) for color in colors if int(color) != bg]
        out: list[dict] = []
        for color in colors_iter:
            out.extend(component_dicts(g == color, color))
        return out

    fg_mask = g != bg
    if colors is not None:
        fg_mask &= np.isin(g, np.array(colors, dtype=int))
    return component_dicts(fg_mask, None)


def _corner_coords_from_bbox(bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {
        "NW": (int(r0), int(c0)),
        "NE": (int(r0), int(c1)),
        "SW": (int(r1), int(c0)),
        "SE": (int(r1), int(c1)),
    }


def _outside_diagonals(bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    corners = _corner_coords_from_bbox(bbox)
    return {
        "NW": (corners["NW"][0] - 1, corners["NW"][1] - 1),
        "NE": (corners["NE"][0] - 1, corners["NE"][1] + 1),
        "SW": (corners["SW"][0] + 1, corners["SW"][1] - 1),
        "SE": (corners["SE"][0] + 1, corners["SE"][1] + 1),
    }


def _l_marker_coords_for_corner(bbox: tuple[int, int, int, int], corner: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if corner == "NW":
        return {(r0 - 1, c0 - 1), (r0 - 1, c0), (r0, c0 - 1)}
    if corner == "NE":
        return {(r0 - 1, c1 + 1), (r0 - 1, c1), (r0, c1 + 1)}
    if corner == "SW":
        return {(r1 + 1, c0 - 1), (r1 + 1, c0), (r1, c0 - 1)}
    if corner == "SE":
        return {(r1 + 1, c1 + 1), (r1 + 1, c1), (r1, c1 + 1)}
    raise ValueError(f"unknown corner {corner}")


def _rotation_cw_degrees(src_corner: str, dst_corner: str) -> int:
    src_idx = CORNER_ORDER_CW.index(src_corner)
    dst_idx = CORNER_ORDER_CW.index(dst_corner)
    return 90 * ((dst_idx - src_idx) % 4)


def _corner_before_cw_rotation(dst_corner: str, degrees: int) -> str:
    steps = (degrees // 90) % 4
    dst_idx = CORNER_ORDER_CW.index(dst_corner)
    return CORNER_ORDER_CW[(dst_idx - steps) % 4]


def _rotate_patch_cw(patch: np.ndarray, degrees: int) -> np.ndarray:
    steps = (degrees // 90) % 4
    return np.rot90(np.asarray(patch, dtype=int), k=(-steps) % 4)


def _is_valid_frame_patch(frame: np.ndarray, bg: int) -> bool:
    if frame.shape != (FRAME_SIZE, FRAME_SIZE):
        return False
    border_mask = np.zeros_like(frame, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    border = frame[border_mask]
    if np.count_nonzero(border == bg) != 1:
        return False
    if np.any((border != GREY) & (border != bg)):
        return False
    if np.any(frame[1:-1, 1:-1] == GREY):
        return False
    return True


def _missing_frame_corner(frame: np.ndarray, bg: int) -> str:
    corners = {
        "NW": int(frame[0, 0]),
        "NE": int(frame[0, -1]),
        "SW": int(frame[-1, 0]),
        "SE": int(frame[-1, -1]),
    }
    missing = [corner for corner, value in corners.items() if value == bg]
    assert len(missing) == 1, "expected one missing frame corner"
    return missing[0]


def _analyze_inner_patch(inner: np.ndarray, bg: int) -> tuple[str, str]:
    assert inner.shape == (INNER_SIZE, INNER_SIZE), "inner frame crop must be 5x5"

    yellow_components = find_components(inner, bg=bg, colors=[YELLOW], monochromatic=True)
    singleton_components = [comp for comp in yellow_components if comp["area"] == 1]
    l_components = [comp for comp in yellow_components if comp["area"] == 3]
    assert len(singleton_components) == 1, "expected one singleton yellow marker"
    assert len(l_components) == 1, "expected one L-shaped yellow marker"

    sprite_coords = np.argwhere((inner != bg) & (inner != YELLOW))
    assert sprite_coords.shape[0] == 9, "expected a filled 3x3 two-color sprite"
    sprite_bbox = bbox_coords(sprite_coords)
    r0, c0, r1, c1 = sprite_bbox
    assert (r1 - r0 + 1, c1 - c0 + 1) == (SPRITE_SIZE, SPRITE_SIZE), "sprite bbox must be 3x3"
    sprite_patch = inner[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(sprite_patch != bg), "sprite bbox must be fully filled"
    assert len(set(map(int, sprite_patch.flat))) == 2, "sprite must use exactly two colors"

    singleton = tuple(map(int, singleton_components[0]["coords"][0]))
    source_matches = [
        corner for corner, coord in _outside_diagonals(sprite_bbox).items() if coord == singleton
    ]
    assert len(source_matches) == 1, "singleton marker must identify one source corner"
    source_corner = source_matches[0]

    l_coords = {tuple(map(int, rc)) for rc in l_components[0]["coords"]}
    l_matches = [
        corner for corner in FRAME_CORNERS if l_coords == _l_marker_coords_for_corner(sprite_bbox, corner)
    ]
    assert len(l_matches) == 1, "expected one corner-touching L marker"
    return source_corner, l_matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Crop each marked frame interior, rotate it by the frame-defined quarter-turn, then stack the results by increasing rotation.

    Input structure:
    - The background is black and the foreground contains 3 to 4 grey hollow 7x7 frames.
    - Each frame border has exactly one missing grey corner pixel, and the 5x5 interior contains one filled 3x3 two-color sprite plus yellow control markers.
    - Inside each frame, one yellow singleton lies diagonally outside a sprite corner and three yellow pixels form an L touching another sprite corner.

    Rule:
    - Find every grey 7x7 frame and crop its 5x5 interior.
    - Use the singleton marker to identify the sprite source corner, and use the frame's missing corner as the destination corner in the interior's corner system.
    - Clear all yellow markers from the 5x5 interior, rotate the whole interior by the unique clockwise quarter-turn that maps source to destination, and keep the rotation angle as metadata.
    - Sort the rotated 5x5 interiors by rotation angle 0, 90, 180, 270 and stack them vertically without gaps.

    Key insight:
    - The missing frame corner is not decoration: together with the singleton corner marker it determines both how each crop rotates and where that crop appears in the final sorted stack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    bg = get_bg(g)
    grey_components = find_components(g, bg=bg, colors=[GREY], monochromatic=True)
    frames = []
    for comp in grey_components:
        if comp["area"] != 23:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        if (r1 - r0 + 1, c1 - c0 + 1) != (FRAME_SIZE, FRAME_SIZE):
            continue
        frame = g[r0 : r1 + 1, c0 : c1 + 1]
        if not _is_valid_frame_patch(frame, bg):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        target_corner = _missing_frame_corner(frame, bg)
        source_corner, _l_corner = _analyze_inner_patch(inner, bg)
        degrees = _rotation_cw_degrees(source_corner, target_corner)
        cleaned = inner.copy()
        cleaned[cleaned == YELLOW] = bg
        frames.append((degrees, _rotate_patch_cw(cleaned, degrees)))

    assert len(frames) == len(grey_components), "all grey components must be valid frames"
    assert 3 <= len(frames) <= 4, "expected 3 to 4 frames"

    ordered = [patch for degrees, patch in sorted(frames, key=lambda item: item[0])]
    return np.vstack(ordered).astype(int)


def _sprite_is_rot_asymmetric(sprite: np.ndarray) -> bool:
    return all(not np.array_equal(sprite, np.rot90(sprite, k=k)) for k in (1, 2, 3))


def _make_sprite(body_color: int, accent_color: int) -> np.ndarray:
    positions = [(r, c) for r in range(SPRITE_SIZE) for c in range(SPRITE_SIZE)]
    for _ in range(200):
        accent_count = random.randint(1, 3)
        accent_positions = set(random.sample(positions, accent_count))
        sprite = np.full((SPRITE_SIZE, SPRITE_SIZE), body_color, dtype=int)
        for r, c in accent_positions:
            sprite[r, c] = accent_color
        if _sprite_is_rot_asymmetric(sprite):
            return sprite
    raise RuntimeError("failed to sample a rotation-asymmetric 3x3 sprite")


def _build_inner_patch(sprite: np.ndarray, source_corner: str, l_corner: str) -> np.ndarray:
    inner = np.full((INNER_SIZE, INNER_SIZE), BLACK, dtype=int)
    inner[1 : 1 + SPRITE_SIZE, 1 : 1 + SPRITE_SIZE] = sprite
    sr, sc = CORNER_TO_INNER[source_corner]
    inner[sr, sc] = YELLOW
    for r, c in _l_marker_coords_for_corner((1, 1, 3, 3), l_corner):
        inner[r, c] = YELLOW
    return inner


def _build_frame(inner: np.ndarray, target_corner: str) -> np.ndarray:
    frame = np.full((FRAME_SIZE, FRAME_SIZE), BLACK, dtype=int)
    frame[0, :] = GREY
    frame[-1, :] = GREY
    frame[:, 0] = GREY
    frame[:, -1] = GREY
    mr, mc = {
        "NW": (0, 0),
        "NE": (0, FRAME_SIZE - 1),
        "SW": (FRAME_SIZE - 1, 0),
        "SE": (FRAME_SIZE - 1, FRAME_SIZE - 1),
    }[target_corner]
    frame[mr, mc] = BLACK
    frame[1:-1, 1:-1] = inner
    return frame


def _sample_positions(rows: int, cols: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [
        (r, c)
        for r in range(GEN_FRAME_BORDER, rows - FRAME_SIZE - GEN_FRAME_BORDER + 1)
        for c in range(GEN_FRAME_BORDER, cols - FRAME_SIZE - GEN_FRAME_BORDER + 1)
    ]
    random.shuffle(candidates)
    blocked = np.zeros((rows, cols), dtype=bool)
    positions: list[tuple[int, int]] = []
    for r, c in candidates:
        if blocked[r : r + FRAME_SIZE, c : c + FRAME_SIZE].any():
            continue
        positions.append((r, c))
        rr0 = max(0, r - GEN_FRAME_GAP)
        rr1 = min(rows, r + FRAME_SIZE + GEN_FRAME_GAP)
        cc0 = max(0, c - GEN_FRAME_GAP)
        cc1 = min(cols, c + FRAME_SIZE + GEN_FRAME_GAP)
        blocked[rr0:rr1, cc0:cc1] = True
        if len(positions) == count:
            return positions
    return None


def _attempt_generate() -> np.ndarray | None:
    rows = random.choice(GEN_GRID_ROWS)
    cols = random.choice(GEN_GRID_COLS)
    frame_count = random.choice(GEN_FRAME_COUNT_CHOICES)
    positions = _sample_positions(rows, cols, frame_count)
    if positions is None:
        return None

    target_corners = random.sample(FRAME_CORNERS, frame_count)
    degrees = random.sample([0, 90, 180, 270], frame_count)
    body_colors = random.sample(GEN_BODY_COLORS, frame_count)

    grid = np.full((rows, cols), BLACK, dtype=int)
    for (r0, c0), target_corner, degree, body_color in zip(positions, target_corners, degrees, body_colors):
        accent_choices = [color for color in GEN_BODY_COLORS if color != body_color]
        accent_color = random.choice(accent_choices)
        sprite = _make_sprite(body_color, accent_color)
        source_corner = _corner_before_cw_rotation(target_corner, degree)
        l_corner = random.choice([corner for corner in FRAME_CORNERS if corner != source_corner])
        inner = _build_inner_patch(sprite, source_corner, l_corner)
        frame = _build_frame(inner, target_corner)
        grid[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE] = frame

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != (frame_count * INNER_SIZE, INNER_SIZE):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid frame-rotation stacking puzzle")
