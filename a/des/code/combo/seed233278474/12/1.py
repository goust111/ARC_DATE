from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _is_border_rectangle(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False
    pts = set(coords)
    for r in range(r0, r1 + 1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return False
    return True


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for comp in _color_components(grid, bg):
        if not _is_border_rectangle(comp["coords"]):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        inner = grid[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3]
        if inner.shape != (2, 2):
            continue
        colors = {int(v) for v in inner.reshape(-1)}
        if bg in colors or len(colors) != 2:
            continue
        frames.append(
            {
                "color": int(comp["color"]),
                "bbox": tuple(map(int, comp["bbox"])),
                "fingerprint": inner.copy(),
            }
        )
    return frames


def _quadrant_of_point(rel_r: int, rel_c: int, h: int, w: int) -> int:
    top = (2 * rel_r) < h
    left = (2 * rel_c) < w
    if top and left:
        return 0
    if top and not left:
        return 1
    if not top and left:
        return 2
    return 3


def _anchor_for_quadrant(
    frame_bbox: tuple[int, int, int, int], patch_shape: tuple[int, int], quadrant: int
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    ph, pw = patch_shape
    inner_r0 = r0 + 1
    inner_c0 = c0 + 1
    inner_r1 = r1 - 1
    inner_c1 = c1 - 1
    if quadrant == 0:
        return inner_r0, inner_c0
    if quadrant == 1:
        return inner_r0, inner_c1 - pw + 1
    if quadrant == 2:
        return inner_r1 - ph + 1, inner_c0
    return inner_r1 - ph + 1, inner_c1 - pw + 1


def _extract_fragment(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    frame_colors: set[int],
) -> dict:
    counts = Counter(int(grid[r, c]) for r, c in coords)
    marker_candidates = [color for color, n in counts.items() if n == 1 and color in frame_colors]
    assert len(marker_candidates) == 1, "each fragment must have exactly one marker color matching a frame"
    marker_color = int(marker_candidates[0])

    other_colors = [(n, color) for color, n in counts.items() if color != marker_color]
    assert other_colors, "fragment must have a body color"
    base_color = int(max(other_colors)[1])

    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    marker_pos: tuple[int, int] | None = None
    for r, c in coords:
        rr = int(r - r0)
        cc = int(c - c0)
        mask[rr, cc] = True
        if int(grid[r, c]) == marker_color:
            marker_pos = (rr, cc)
    assert marker_pos is not None

    quadrant = _quadrant_of_point(marker_pos[0], marker_pos[1], h, w)
    return {
        "coords": [(int(r), int(c)) for r, c in coords],
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "mask": mask,
        "base_color": base_color,
        "marker_color": marker_color,
        "quadrant": quadrant,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match floating fragments to colored rectangular frames by marker color, then place
    each fragment into the frame corner implied by the marker's quadrant within the
    fragment bbox. The marker itself is restored to the fragment body color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    frames = _find_frames(g, bg)
    assert frames, "expected at least one frame"
    frame_by_color: dict[int, dict] = {}
    frame_bboxes = set()
    for frame in frames:
        color = int(frame["color"])
        assert color not in frame_by_color, "frame colors must be unique"
        frame_by_color[color] = frame
        frame_bboxes.add(tuple(frame["bbox"]))

    fragments: list[dict] = []
    for coords in _mask_components(g != bg):
        bbox = _bbox(coords)
        if bbox in frame_bboxes:
            continue
        fragments.append(_extract_fragment(g, coords, set(frame_by_color)))

    out = g.copy()
    used_markers: set[int] = set()
    for fragment in fragments:
        marker_color = int(fragment["marker_color"])
        assert marker_color in frame_by_color
        assert marker_color not in used_markers, "each frame should receive exactly one fragment"
        used_markers.add(marker_color)

        for r, c in fragment["coords"]:
            out[r, c] = bg

        frame = frame_by_color[marker_color]
        mask = fragment["mask"]
        dest_r0, dest_c0 = _anchor_for_quadrant(frame["bbox"], mask.shape, int(fragment["quadrant"]))
        inner_h = frame["bbox"][2] - frame["bbox"][0] - 1
        inner_w = frame["bbox"][3] - frame["bbox"][1] - 1
        assert mask.shape[0] <= inner_h and mask.shape[1] <= inner_w
        for rr in range(mask.shape[0]):
            for cc in range(mask.shape[1]):
                if mask[rr, cc]:
                    out[dest_r0 + rr, dest_c0 + cc] = int(fragment["base_color"])
    return out


def _fingerprint_block(rng: random.Random, colors: tuple[int, int]) -> np.ndarray:
    a, b = colors
    patterns = (
        np.array([[a, b], [b, a]], dtype=int),
        np.array([[a, a], [b, b]], dtype=int),
        np.array([[a, b], [a, b]], dtype=int),
        np.array([[b, a], [a, b]], dtype=int),
    )
    return patterns[rng.randrange(len(patterns))].copy()


def _random_shape(rng: random.Random, quadrant: int) -> tuple[np.ndarray, tuple[int, int]]:
    for _ in range(300):
        box_h = rng.randint(2, 4)
        box_w = rng.randint(2, 4)
        target_area = rng.randint(3, min(8, box_h * box_w))

        start = (rng.randrange(box_h), rng.randrange(box_w))
        cells = {start}
        frontier = set()
        for nr, nc in _neighbors4(*start):
            if 0 <= nr < box_h and 0 <= nc < box_w:
                frontier.add((nr, nc))

        while len(cells) < target_area and frontier:
            pick = rng.choice(tuple(frontier))
            frontier.remove(pick)
            cells.add(pick)
            for nr, nc in _neighbors4(*pick):
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 2 or w < 2:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True

        quadrant_cells = [
            (rr, cc)
            for rr in range(h)
            for cc in range(w)
            if mask[rr, cc] and _quadrant_of_point(rr, cc, h, w) == quadrant
        ]
        if not quadrant_cells:
            continue
        marker = rng.choice(quadrant_cells)
        return mask, marker
    raise RuntimeError("failed to sample a fragment shape")


def _make_frame_patch(border_color: int, fingerprint: np.ndarray, inner_h: int, inner_w: int) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    patch[1:3, 1:3] = fingerprint
    return patch


def _make_fragment_patch(
    mask: np.ndarray,
    marker_pos: tuple[int, int],
    base_color: int,
    marker_color: int,
) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = base_color
    patch[marker_pos] = marker_color
    return patch


def _can_place(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    grid_h, grid_w = reserved.shape
    if r0 < 0 or c0 < 0 or r0 + h > grid_h or c0 + w > grid_w:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(grid_h, r0 + h + pad)
    cc1 = min(grid_w, c0 + w + pad)
    return not reserved[rr0:rr1, cc0:cc1].any()


def _reserve(reserved: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    reserved[r0 : r0 + h, c0 : c0 + w] = True


def _random_place(rng: random.Random, reserved: np.ndarray, h: int, w: int, pad: int = 1) -> tuple[int, int] | None:
    grid_h, grid_w = reserved.shape
    if h > grid_h or w > grid_w:
        return None
    for _ in range(500):
        r0 = rng.randint(0, grid_h - h)
        c0 = rng.randint(0, grid_w - w)
        if _can_place(reserved, r0, c0, h, w, pad=pad):
            _reserve(reserved, r0, c0, h, w)
            return r0, c0
    return None


def _draw_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    sub = patch != BG
    grid[r0 : r0 + h, c0 : c0 + w][sub] = patch[sub]


def _draw_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    for rr in range(mask.shape[0]):
        for cc in range(mask.shape[1]):
            if mask[rr, cc]:
                grid[r0 + rr, c0 + cc] = color


def _build_expected(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.array(inp, copy=True)
    for spec in specs:
        frag_r0, frag_c0 = spec["fragment_pos"]
        frag_patch = spec["fragment_patch"]
        fh, fw = frag_patch.shape
        for rr in range(fh):
            for cc in range(fw):
                if frag_patch[rr, cc] != BG:
                    out[frag_r0 + rr, frag_c0 + cc] = BG

        frame_r0, frame_c0 = spec["frame_pos"]
        frame_bbox = (
            frame_r0,
            frame_c0,
            frame_r0 + spec["frame_patch"].shape[0] - 1,
            frame_c0 + spec["frame_patch"].shape[1] - 1,
        )
        dest_r0, dest_c0 = _anchor_for_quadrant(frame_bbox, spec["mask"].shape, int(spec["quadrant"]))
        _draw_mask(out, spec["mask"], dest_r0, dest_c0, int(spec["base_color"]))
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_frames = rng.randint(2, 4)
    all_colors = list(range(1, 10))
    rng.shuffle(all_colors)
    fingerprint_colors = (all_colors[0], all_colors[1])
    border_colors = all_colors[2 : 2 + num_frames]
    if len(border_colors) < num_frames:
        return None
    base_palette = [color for color in all_colors if color not in fingerprint_colors and color not in border_colors]
    if not base_palette:
        return None

    quadrants = [0, 1, 2, 3]
    rng.shuffle(quadrants)
    fingerprint = _fingerprint_block(rng, fingerprint_colors)

    specs: list[dict] = []
    for i in range(num_frames):
        border_color = int(border_colors[i])
        base_color = int(rng.choice(base_palette))
        quadrant = int(quadrants[i])
        mask, marker_pos = _random_shape(rng, quadrant)
        inner_h = rng.randint(max(5, 2 * mask.shape[0]), max(6, 2 * mask.shape[0] + 4))
        inner_w = rng.randint(max(5, 2 * mask.shape[1]), max(6, 2 * mask.shape[1] + 4))
        frame_patch = _make_frame_patch(border_color, fingerprint, inner_h, inner_w)
        fragment_patch = _make_fragment_patch(mask, marker_pos, base_color, border_color)
        specs.append(
            {
                "border_color": border_color,
                "base_color": base_color,
                "quadrant": quadrant,
                "mask": mask,
                "frame_patch": frame_patch,
                "fragment_patch": fragment_patch,
            }
        )

    max_frame_h = max(spec["frame_patch"].shape[0] for spec in specs)
    max_frame_w = max(spec["frame_patch"].shape[1] for spec in specs)
    total_frame_w = sum(spec["frame_patch"].shape[1] for spec in specs)
    grid_h = rng.randint(max(24, max_frame_h + 10), max(30, max_frame_h + 18))
    grid_w = rng.randint(max(26, max(max_frame_w + 12, total_frame_w // 2 + 10)), max(34, total_frame_w + 18))

    grid = np.zeros((grid_h, grid_w), dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    for spec in specs:
        fh, fw = spec["frame_patch"].shape
        pos = _random_place(rng, reserved, fh, fw, pad=1)
        if pos is None:
            return None
        spec["frame_pos"] = pos
        _draw_patch(grid, spec["frame_patch"], pos[0], pos[1])

    for spec in specs:
        ph, pw = spec["fragment_patch"].shape
        pos = _random_place(rng, reserved, ph, pw, pad=1)
        if pos is None:
            return None
        spec["fragment_pos"] = pos
        _draw_patch(grid, spec["fragment_patch"], pos[0], pos[1])

    expected = _build_expected(grid, specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC grid")
