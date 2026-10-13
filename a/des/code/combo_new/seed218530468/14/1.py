from __future__ import annotations

import random
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
FRAME_COLOR = 5
EDGES = ("top", "right", "bottom", "left")
CORNERS = ("TL", "TR", "BR", "BL")
EDGE_TO_SOURCE_CORNER = {
    "top": "TL",
    "right": "TR",
    "bottom": "BR",
    "left": "BL",
}
INNER_SIZES = (5, 7, 9)
MAX_GENERATION_ATTEMPTS = 400
MAX_LAYOUT_ATTEMPTS = 120


def _most_common_color(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_rectangular_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    coord_set = set(coords)
    if len(coords) != 2 * (h + w) - 4:
        return False
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in coord_set):
                return False
    return True


def _find_frames(g: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for comp in _components(g == color):
            if _is_rectangular_frame(comp):
                r0, c0, r1, c1 = _bbox(comp)
                frames.append(
                    {
                        "color": int(color),
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    }
                )
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _source_edge_from_marker(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = marker
    r0, c0, r1, c1 = bbox
    if r == r0 - 1 and c0 <= c <= c1:
        return "top"
    if r == r1 + 1 and c0 <= c <= c1:
        return "bottom"
    if c == c0 - 1 and r0 <= r <= r1:
        return "left"
    if c == c1 + 1 and r0 <= r <= r1:
        return "right"
    raise AssertionError("marker is not attached to any frame edge")


def _corner_from_fragment(mask: np.ndarray) -> str:
    h, w = map(int, mask.shape)
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    touch_top = r0 == 0
    touch_bottom = r1 == h - 1
    touch_left = c0 == 0
    touch_right = c1 == w - 1
    if touch_top and touch_left and not touch_bottom and not touch_right:
        return "TL"
    if touch_top and touch_right and not touch_bottom and not touch_left:
        return "TR"
    if touch_bottom and touch_right and not touch_top and not touch_left:
        return "BR"
    if touch_bottom and touch_left and not touch_top and not touch_right:
        return "BL"
    raise AssertionError("fragment must occupy exactly one interior corner")


def _restore_mask(fragment_mask: np.ndarray, source_edge: str, current_corner: str) -> np.ndarray:
    source_corner = EDGE_TO_SOURCE_CORNER[source_edge]
    rotation = (CORNERS.index(current_corner) - CORNERS.index(source_corner)) % 4
    assert rotation in (1, 2, 3)

    unrotated = np.rot90(fragment_mask, rotation)
    r0, c0, r1, c1 = _bbox_from_mask(unrotated)
    restored = np.zeros_like(unrotated, dtype=bool)
    if source_edge == "top":
        restored[: r1 + 1, c0 : c1 + 1] = True
    elif source_edge == "right":
        restored[r0 : r1 + 1, c0:] = True
    elif source_edge == "bottom":
        restored[r0:, c0 : c1 + 1] = True
    else:
        restored[r0 : r1 + 1, : c1 + 1] = True
    return restored


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each framed fragment's rotation from its corner and side marker, then restore the full edge-aligned rectangle.

    Input structure:
    - The grid has a dominant background and several one-pixel-thick grey hollow rectangular frames.
    - Each frame contains one monochrome colored fragment inside its interior corner area.
    - A same-colored singleton marker sits just outside one frame edge and identifies the fragment's original edge before rotation.

    Rule:
    - For each grey frame, match the interior fragment with the exterior marker of the same color.
    - The marker's edge gives the fragment's original attachment side; the fragment's current corner gives its rotated orientation.
    - Inverse-rotate the fragment back to that source side, then complete it into the full side-aligned rectangle inside the frame.
    - Remove the outside marker and leave the grey frame unchanged.

    Key insight:
    - The outside pixel is not a destination marker: it names the fragment's pre-rotation side, while the fragment's corner reveals how far it has been quarter-turned.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    frames = _find_frames(g, bg)
    assert frames

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = [int(v) for v in np.unique(interior) if int(v) not in {bg, frame["color"]}]
        assert len(colors) == 1
        obj_color = colors[0]

        all_cells = [tuple(map(int, xy)) for xy in np.argwhere(g == obj_color)]
        inside_cells = [
            (r, c)
            for r, c in all_cells
            if r0 < r < r1 and c0 < c < c1
        ]
        marker_cells = [cell for cell in all_cells if cell not in inside_cells]
        assert len(marker_cells) == 1
        marker = marker_cells[0]

        fragment_mask = interior == obj_color
        assert fragment_mask.any()
        source_edge = _source_edge_from_marker(marker, frame["bbox"])
        current_corner = _corner_from_fragment(fragment_mask)
        restored_mask = _restore_mask(fragment_mask, source_edge, current_corner)

        patch = np.full(interior.shape, bg, dtype=int)
        patch[restored_mask] = obj_color
        out[r0 + 1 : r1, c0 + 1 : c1] = patch
        out[marker[0], marker[1]] = bg

    return out


def _full_rectangle_mask(size: int, edge: str, height: int, width: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    if edge == "top":
        mask[:height, :width] = True
    elif edge == "right":
        mask[:height, size - width :] = True
    elif edge == "bottom":
        mask[size - height :, size - width :] = True
    else:
        mask[size - height :, :width] = True
    return mask


def _marker_position(r0: int, c0: int, size: int, edge: str) -> tuple[int, int]:
    mid = size // 2 + 1
    if edge == "top":
        return r0 - 1, c0 + mid
    if edge == "right":
        return r0 + mid, c0 + size + 2
    if edge == "bottom":
        return r0 + size + 2, c0 + mid
    return r0 + mid, c0 - 1


def _draw_frame(canvas: np.ndarray, r0: int, c0: int, size: int, color: int) -> None:
    outer = size + 2
    canvas[r0, c0 : c0 + outer] = color
    canvas[r0 + outer - 1, c0 : c0 + outer] = color
    canvas[r0 : r0 + outer, c0] = color
    canvas[r0 : r0 + outer, c0 + outer - 1] = color


def _place_layout(
    height: int,
    width: int,
    sizes: list[int],
) -> list[tuple[int, int]] | None:
    occupancy = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int]] = []
    for size in sorted(sizes, reverse=True):
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, height - size - 2):
            for c0 in range(1, width - size - 2):
                rr0 = r0 - 1
                cc0 = c0 - 1
                rr1 = r0 + size + 3
                cc1 = c0 + size + 3
                if occupancy[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        choice = random.choice(candidates)
        placements.append((size, choice[0], choice[1]))
        rr0 = choice[0] - 1
        cc0 = choice[1] - 1
        rr1 = choice[0] + size + 3
        cc1 = choice[1] + size + 3
        occupancy[rr0:rr1, cc0:cc1] = True

    by_size: dict[int, list[tuple[int, int]]] = {}
    for size, r0, c0 in placements:
        by_size.setdefault(size, []).append((r0, c0))

    ordered: list[tuple[int, int]] = []
    used: dict[int, int] = {}
    for size in sizes:
        idx = used.get(size, 0)
        ordered.append(by_size[size][idx])
        used[size] = idx + 1
    return ordered


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    pairs = sorted(((int(n), int(v)) for v, n in zip(vals, counts)), reverse=True)
    return pairs[0][1] == bg and (len(pairs) == 1 or pairs[0][0] > pairs[1][0])


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(1, 4)
    sizes = [random.choice(INNER_SIZES) for _ in range(frame_count)]
    min_dim = max(max(sizes) + 4, 16)
    height = random.randint(min_dim, 30)
    width = random.randint(min_dim, 30)

    placements = None
    for _ in range(MAX_LAYOUT_ATTEMPTS):
        placements = _place_layout(height, width, sizes)
        if placements is not None:
            break
    if placements is None:
        return None

    bg = random.choice([c for c in range(10) if c != FRAME_COLOR])
    obj_colors = random.sample([c for c in range(10) if c not in {bg, FRAME_COLOR}], frame_count)

    inp = np.full((height, width), bg, dtype=int)
    expected = inp.copy()

    for (r0, c0), size, obj_color in zip(placements, sizes, obj_colors):
        _draw_frame(inp, r0, c0, size, FRAME_COLOR)
        _draw_frame(expected, r0, c0, size, FRAME_COLOR)

        edge = random.choice(EDGES)
        rotation = random.randint(1, 3)
        rect_h = random.randint(2, size - 1)
        rect_w = random.randint(2, size - 1)

        source_mask = _full_rectangle_mask(size, edge, rect_h, rect_w)
        fragment_mask = np.rot90(source_mask, -rotation)

        in_patch = inp[r0 + 1 : r0 + 1 + size, c0 + 1 : c0 + 1 + size]
        in_patch[fragment_mask] = obj_color
        inp[r0 + 1 : r0 + 1 + size, c0 + 1 : c0 + 1 + size] = in_patch

        out_patch = expected[r0 + 1 : r0 + 1 + size, c0 + 1 : c0 + 1 + size]
        out_patch[source_mask] = obj_color
        expected[r0 + 1 : r0 + 1 + size, c0 + 1 : c0 + 1 + size] = out_patch

        mr, mc = _marker_position(r0, c0, size, edge)
        inp[mr, mc] = obj_color

    if not _bg_is_dominant(inp, bg):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
