from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
RED = 2
BLUE = 1
GREEN = 3
BORDER_TRANSFORMS = {RED: "h", BLUE: "v", GREEN: "n"}
KEY_COLORS = (4, 6, 7, 8, 9)
FRAME_SUBSETS = (
    (RED, BLUE),
    (RED, GREEN),
    (BLUE, GREEN),
    (RED, BLUE, GREEN),
)
GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _bool_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _monochrome_components(grid: np.ndarray, active: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not active[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if active[nr, nc] and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _is_rect_frame(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    frame = np.zeros_like(mask)
    frame[0, :] = True
    frame[-1, :] = True
    frame[:, 0] = True
    frame[:, -1] = True
    if np.array_equal(mask, frame):
        return r0, c0, r1, c1
    return None


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    comps = _monochrome_components(grid, grid != bg)
    frames: list[dict] = []
    for comp in comps:
        bbox = _is_rect_frame(comp["coords"])
        if bbox is not None:
            frames.append({"color": int(comp["color"]), "bbox": bbox})
    return frames


def _extract_hole_info(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> dict:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    colors = {int(v) for v in np.unique(interior) if int(v) != bg}
    assert len(colors) == 1, "each frame interior must contain one lock color"
    lock_color = colors.pop()
    bg_comps = _bool_components(interior == bg)
    holes = []
    ih, iw = interior.shape
    for coords in bg_comps:
        if any(r == 0 or c == 0 or r == ih - 1 or c == iw - 1 for r, c in coords):
            continue
        holes.append(coords)
    assert len(holes) == 1, "expected a unique enclosed hole"
    hole_coords = holes[0]
    hr0, hc0, hr1, hc1 = _bbox(hole_coords)
    mask = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
    for r, c in hole_coords:
        mask[r - hr0, c - hc0] = True
    return {"lock_color": lock_color, "bbox": (hr0, hc0, hr1, hc1), "mask": mask}


def _apply_transform(arr: np.ndarray, border_color: int) -> np.ndarray:
    mode = BORDER_TRANSFORMS.get(int(border_color), "n")
    if mode == "h":
        return np.fliplr(arr)
    if mode == "v":
        return np.flipud(arr)
    return arr.copy()


def _find_keys(grid: np.ndarray, frames: list[dict], bg: int) -> dict[int, dict]:
    h, w = grid.shape
    outside = np.ones((h, w), dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        outside[r0 : r1 + 1, c0 : c1 + 1] = False

    comps = _monochrome_components(grid, outside & (grid != bg))
    labels = np.full((h, w), -1, dtype=int)
    for idx, comp in enumerate(comps):
        for r, c in comp["coords"]:
            labels[r, c] = idx

    frame_colors = {int(frame["color"]) for frame in frames}
    used_keys: set[int] = set()
    found: dict[int, dict] = {}
    for idx, comp in enumerate(comps):
        if len(comp["coords"]) != 1 or int(comp["color"]) not in frame_colors:
            continue
        r, c = comp["coords"][0]
        neighbor_ids = {
            int(labels[nr, nc])
            for nr, nc in _neighbors4(r, c, h, w)
            if labels[nr, nc] >= 0 and labels[nr, nc] != idx
        }
        neighbor_ids = {j for j in neighbor_ids if int(comps[j]["color"]) != int(comp["color"])}
        assert len(neighbor_ids) == 1, "marker must touch exactly one key"
        key_idx = next(iter(neighbor_ids))
        assert key_idx not in used_keys, "a key cannot belong to two markers"
        key_comp = comps[key_idx]
        found[int(comp["color"])] = {
            "marker": (int(r), int(c)),
            "marker_coords": comp["coords"],
            "key_coords": key_comp["coords"],
            "key_mask": _crop_mask(key_comp["coords"]),
        }
        used_keys.add(key_idx)

    assert len(found) == len(frames), "each frame needs exactly one key-marker pair"
    return found


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each outside key to the same-colored frame marker, fill the frame's
    enclosed lock hole with the key, then flip the frame interior by border
    color: red = left-right, blue = top-bottom, green = unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _bg_color(g)

    frames = _find_frames(g, bg)
    assert frames, "expected framed regions"
    frame_by_color: dict[int, dict] = {}
    for frame in frames:
        color = int(frame["color"])
        assert color not in frame_by_color, "frame colors must be unique"
        frame_by_color[color] = frame

    keys = _find_keys(g, frames, bg)
    out = g.copy()

    for info in keys.values():
        for r, c in info["key_coords"]:
            out[r, c] = bg
        for r, c in info["marker_coords"]:
            out[r, c] = bg

    for color, frame in frame_by_color.items():
        hole = _extract_hole_info(g, frame["bbox"], bg)
        assert np.array_equal(hole["mask"], keys[color]["key_mask"]), "key must match its hole"
        r0, c0, r1, c1 = frame["bbox"]
        hr0, hc0, hr1, hc1 = hole["bbox"]
        lock_color = int(hole["lock_color"])
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        patch = interior[hr0 : hr1 + 1, hc0 : hc1 + 1]
        patch[hole["mask"]] = lock_color
        interior[:, :] = _apply_transform(interior, color)

    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    coords = np.argwhere(mask)
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r) + dr
                cc = int(c) + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = True
    return out


def _has_enclosed_false(mask: np.ndarray) -> bool:
    h, w = mask.shape
    inv = ~mask
    for coords in _bool_components(inv):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords):
            continue
        return True
    return False


def _random_hole_mask(rng: random.Random) -> np.ndarray | None:
    for _ in range(80):
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        target = rng.randint(max(3, (h * w + 2) // 3), h * w - 1)
        cells = {(rng.randrange(h), rng.randrange(w))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, h, w):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != target:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        mask = _crop_mask(coords)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if mask.all():
            continue
        if _has_enclosed_false(mask):
            continue
        return mask
    return None


def _build_frame_case(rng: random.Random, border_color: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    for _ in range(120):
        hole = _random_hole_mask(rng)
        if hole is None:
            return None
        hh, hw = hole.shape
        hole_canvas = np.zeros((hh + 2, hw + 2), dtype=bool)
        hole_canvas[1 : hh + 1, 1 : hw + 1] = hole
        filled = _dilate8(hole_canvas)
        ring = filled & ~hole_canvas

        top = rng.randint(1, 3)
        bottom = rng.randint(1, 3)
        left = rng.randint(1, 3)
        right = rng.randint(1, 3)
        ih = filled.shape[0] + top + bottom
        iw = filled.shape[1] + left + right

        input_interior = np.full((ih, iw), BG, dtype=int)
        output_interior = np.full((ih, iw), BG, dtype=int)
        input_interior[top : top + filled.shape[0], left : left + filled.shape[1]][ring] = GREY
        output_interior[top : top + filled.shape[0], left : left + filled.shape[1]][filled] = GREY
        transformed = _apply_transform(output_interior, border_color)
        if border_color in (RED, BLUE) and np.array_equal(transformed, output_interior):
            continue

        frame_h = ih + 2
        frame_w = iw + 2
        inp = np.full((frame_h, frame_w), BG, dtype=int)
        out = np.full((frame_h, frame_w), BG, dtype=int)
        inp[0, :] = border_color
        inp[-1, :] = border_color
        inp[:, 0] = border_color
        inp[:, -1] = border_color
        out[0, :] = border_color
        out[-1, :] = border_color
        out[:, 0] = border_color
        out[:, -1] = border_color
        inp[1:-1, 1:-1] = input_interior
        out[1:-1, 1:-1] = transformed
        return inp, out, hole
    return None


def _build_key_patch(
    rng: random.Random,
    hole: np.ndarray,
    marker_color: int,
    key_color: int,
) -> np.ndarray:
    hh, hw = hole.shape
    side = rng.choice(("left", "right", "top", "bottom"))
    if side == "left":
        patch = np.full((hh, hw + 1), BG, dtype=int)
        patch[:, 1:] = np.where(hole, key_color, BG)
        rows = [r for r in range(hh) if hole[r, 0]]
        patch[rng.choice(rows), 0] = marker_color
        return patch
    if side == "right":
        patch = np.full((hh, hw + 1), BG, dtype=int)
        patch[:, :hw] = np.where(hole, key_color, BG)
        rows = [r for r in range(hh) if hole[r, -1]]
        patch[rng.choice(rows), hw] = marker_color
        return patch
    if side == "top":
        patch = np.full((hh + 1, hw), BG, dtype=int)
        patch[1:, :] = np.where(hole, key_color, BG)
        cols = [c for c in range(hw) if hole[0, c]]
        patch[0, rng.choice(cols)] = marker_color
        return patch
    patch = np.full((hh + 1, hw), BG, dtype=int)
    patch[:hh, :] = np.where(hole, key_color, BG)
    cols = [c for c in range(hw) if hole[-1, c]]
    patch[hh, rng.choice(cols)] = marker_color
    return patch


def _find_free_position(
    occupied: np.ndarray,
    patch_shape: tuple[int, int],
    pad: int,
    rng: random.Random,
) -> tuple[int, int] | None:
    h, w = occupied.shape
    ph, pw = patch_shape
    positions = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            rr0 = max(0, r0 - pad)
            cc0 = max(0, c0 - pad)
            rr1 = min(h, r0 + ph + pad)
            cc1 = min(w, c0 + pw + pad)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    if not positions:
        return None
    return rng.choice(positions)


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = patch.shape
    sub = grid[r0 : r0 + ph, c0 : c0 + pw]
    mask = patch != BG
    sub[mask] = patch[mask]


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, shape: tuple[int, int], pad: int) -> None:
    h, w = occupied.shape
    ph, pw = shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + ph + pad)
    cc1 = min(w, c0 + pw + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_colors = list(rng.choice(FRAME_SUBSETS))
    h = rng.randint(24, 34)
    w = rng.randint(24, 34)
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    frame_infos = []
    for border_color in frame_colors:
        built = _build_frame_case(rng, border_color)
        if built is None:
            return None
        frame_patch, out_patch, hole = built
        frame_infos.append((border_color, frame_patch, out_patch, hole))

    frame_infos.sort(key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True)
    placed_frames = []
    for border_color, frame_patch, out_patch, hole in frame_infos:
        pos = _find_free_position(occupied, frame_patch.shape, pad=2, rng=rng)
        if pos is None:
            return None
        r0, c0 = pos
        _paste(inp, frame_patch, r0, c0)
        _paste(out, out_patch, r0, c0)
        _mark_occupied(occupied, r0, c0, frame_patch.shape, pad=2)
        placed_frames.append((border_color, hole))

    for border_color, hole in placed_frames:
        key_color = rng.choice([c for c in KEY_COLORS if c not in frame_colors])
        key_patch = _build_key_patch(rng, hole, border_color, key_color)
        pos = _find_free_position(occupied, key_patch.shape, pad=1, rng=rng)
        if pos is None:
            return None
        r0, c0 = pos
        _paste(inp, key_patch, r0, c0)
        _mark_occupied(occupied, r0, c0, key_patch.shape, pad=1)

    if np.count_nonzero(inp == BG) <= inp.size // 2:
        return None
    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
