from __future__ import annotations

from collections import deque
import random

import numpy as np


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == int(color)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(coords)
    return components


def _single_gap(edge: np.ndarray) -> tuple[int, int] | None | bool:
    inner = ~edge[1:-1]
    gap_idx = np.flatnonzero(inner)
    if gap_idx.size == 0:
        return None
    if int(gap_idx[-1]) - int(gap_idx[0]) + 1 != int(gap_idx.size):
        return False
    return int(gap_idx[0]) + 1, int(gap_idx.size)


def _enclosed_mask(component_mask: np.ndarray) -> np.ndarray:
    h, w = component_mask.shape
    open_cells = ~component_mask
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if open_cells[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if open_cells[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if open_cells[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return open_cells & ~outside


def _is_bar(coords: np.ndarray) -> bool:
    if coords.shape[0] < 2:
        return False
    rows = coords[:, 0]
    cols = coords[:, 1]
    if np.all(rows == rows[0]):
        ordered = np.sort(cols)
        return int(ordered[-1]) - int(ordered[0]) + 1 == int(coords.shape[0])
    if np.all(cols == cols[0]):
        ordered = np.sort(rows)
        return int(ordered[-1]) - int(ordered[0]) + 1 == int(coords.shape[0])
    return False


def _notch_signature(component_mask: np.ndarray) -> tuple[str, int, int] | None:
    h, w = component_mask.shape
    if h < 4 or w < 4:
        return None
    if not (
        component_mask[0, 0]
        and component_mask[0, -1]
        and component_mask[-1, 0]
        and component_mask[-1, -1]
    ):
        return None

    edges = {
        "top": component_mask[0, :],
        "bottom": component_mask[-1, :],
        "left": component_mask[:, 0],
        "right": component_mask[:, -1],
    }
    candidates: list[tuple[str, int, int]] = []
    for side, edge in edges.items():
        gap = _single_gap(edge)
        if gap is False:
            return None
        if gap is not None:
            candidates.append((side, gap[0], gap[1]))
    if len(candidates) != 1:
        return None

    side, start, length = candidates[0]
    if side == "top":
        if h < 3 or start < 2 or start + length > w - 3:
            return None
        if not np.all(component_mask[1, start - 1 : start + length + 1]):
            return None
    elif side == "bottom":
        if h < 3 or start < 2 or start + length > w - 3:
            return None
        if not np.all(component_mask[-2, start - 1 : start + length + 1]):
            return None
    elif side == "left":
        if w < 3 or start < 2 or start + length > h - 3:
            return None
        if not np.all(component_mask[start - 1 : start + length + 1, 1]):
            return None
    else:
        if w < 3 or start < 2 or start + length > h - 3:
            return None
        if not np.all(component_mask[start - 1 : start + length + 1, -2]):
            return None
    return side, start, length


def _analyze_frame(
    grid: np.ndarray, coords: list[tuple[int, int]], color: int, bg: int
) -> dict[str, object] | None:
    pts = np.asarray(coords, dtype=int)
    r0, c0 = np.min(pts, axis=0)
    r1, c1 = np.max(pts, axis=0)
    component_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    component_mask[pts[:, 0] - r0, pts[:, 1] - c0] = True

    signature = _notch_signature(component_mask)
    if signature is None:
        return None

    inside = _enclosed_mask(component_mask)
    if not inside.any():
        return None

    subgrid = grid[r0 : r1 + 1, c0 : c1 + 1]
    payload_mask = inside & (subgrid != int(bg))
    payload_color: int | None = None
    if payload_mask.any():
        payload_colors = {int(v) for v in np.unique(subgrid[payload_mask]) if int(v) != int(color)}
        if len(payload_colors) != 1:
            return None
        payload_coords = np.argwhere(payload_mask)
        if not _is_bar(payload_coords):
            return None
        payload_color = int(next(iter(payload_colors)))

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "inside": inside,
        "signature": signature,
        "payload_color": payload_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    colors = sorted(set(int(v) for v in g.flat) - {int(bg)})

    frames: list[dict[str, object]] = []
    for color in colors:
        for coords in _components_of_color(g, color):
            info = _analyze_frame(g, coords, color, bg)
            if info is not None:
                frames.append(info)

    key_by_signature: dict[tuple[str, int, int], int] = {}
    locks: list[dict[str, object]] = []
    for info in frames:
        signature = info["signature"]
        payload_color = info["payload_color"]
        if payload_color is None:
            locks.append(info)
        else:
            key_by_signature[signature] = int(payload_color)

    out = g.copy()
    for info in locks:
        fill_color = key_by_signature.get(info["signature"])
        if fill_color is None:
            continue
        r0, c0, r1, c1 = info["bbox"]
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        sub[info["inside"]] = int(fill_color)
    return out


# --------------- generator ---------------

BG_COLOR = 0
FRAME_COLOR = 5
GRID_RANGES = {
    2: ((18, 24), (20, 28)),
    3: ((22, 28), (24, 32)),
    4: ((26, 32), (28, 36)),
}
PAIR_COUNT_RANGE = (2, 4)
FRAME_H_RANGE = (6, 9)
FRAME_W_RANGE = (7, 11)
NOTCH_LEN_MAX = 3
OBJECT_PADDING = 2
MAX_GEN_ATTEMPTS = 300


def _candidate_signatures(h: int, w: int) -> list[tuple[str, int, int]]:
    candidates: list[tuple[str, int, int]] = []
    for length in range(1, min(NOTCH_LEN_MAX, w - 5) + 1):
        for start in range(2, w - length - 2):
            candidates.append(("top", start, length))
            candidates.append(("bottom", start, length))
    for length in range(1, min(NOTCH_LEN_MAX, h - 5) + 1):
        for start in range(2, h - length - 2):
            candidates.append(("left", start, length))
            candidates.append(("right", start, length))
    return candidates


def _frame_mask(h: int, w: int, signature: tuple[str, int, int]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    side, start, length = signature

    if side == "top":
        mask[0, start : start + length] = False
        mask[1, start - 1 : start + length + 1] = True
    elif side == "bottom":
        mask[-1, start : start + length] = False
        mask[-2, start - 1 : start + length + 1] = True
    elif side == "left":
        mask[start : start + length, 0] = False
        mask[start - 1 : start + length + 1, 1] = True
    else:
        mask[start : start + length, -1] = False
        mask[start - 1 : start + length + 1, -2] = True
    return mask


def _bar_candidates(inside: np.ndarray) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    h, w = inside.shape
    horizontal: list[tuple[int, int, int]] = []
    vertical: list[tuple[int, int, int]] = []

    for r in range(h):
        c = 0
        while c < w:
            while c < w and not inside[r, c]:
                c += 1
            start = c
            while c < w and inside[r, c]:
                c += 1
            run_len = c - start
            if run_len >= 2:
                max_len = min(run_len, 4)
                for length in range(2, max_len + 1):
                    for seg_start in range(start, c - length + 1):
                        horizontal.append((r, seg_start, length))

    for c in range(w):
        r = 0
        while r < h:
            while r < h and not inside[r, c]:
                r += 1
            start = r
            while r < h and inside[r, c]:
                r += 1
            run_len = r - start
            if run_len >= 2:
                max_len = min(run_len, 4)
                for length in range(2, max_len + 1):
                    for seg_start in range(start, r - length + 1):
                        vertical.append((seg_start, c, length))

    return horizontal, vertical


def _make_key_patch(
    h: int, w: int, signature: tuple[str, int, int], key_color: int, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    mask = _frame_mask(h, w, signature)
    inside = _enclosed_mask(mask)
    horiz, vert = _bar_candidates(inside)
    if not horiz and not vert:
        raise ValueError("frame interior has no valid key bar placement")

    patch = np.full((h, w), BG_COLOR, dtype=int)
    patch[mask] = FRAME_COLOR
    if horiz and vert:
        orientation = rng.choice(["h", "v"])
    else:
        orientation = "h" if horiz else "v"

    if orientation == "h":
        row, start, length = rng.choice(horiz)
        patch[row, start : start + length] = int(key_color)
    else:
        start, col, length = rng.choice(vert)
        patch[start : start + length, col] = int(key_color)
    return patch, inside


def _lock_patches(
    h: int, w: int, signature: tuple[str, int, int], key_color: int
) -> tuple[np.ndarray, np.ndarray]:
    mask = _frame_mask(h, w, signature)
    inside = _enclosed_mask(mask)
    inp = np.full((h, w), BG_COLOR, dtype=int)
    inp[mask] = FRAME_COLOR
    out = inp.copy()
    out[inside] = int(key_color)
    return inp, out


def _free_locations(
    occupied: np.ndarray, patch_shape: tuple[int, int], padding: int
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    ph, pw = patch_shape
    locations: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            rs = max(0, r0 - padding)
            re = min(h, r0 + ph + padding)
            cs = max(0, c0 - padding)
            ce = min(w, c0 + pw + padding)
            if not occupied[rs:re, cs:ce].any():
                locations.append((r0, c0))
    return locations


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    pair_count = rng.randint(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1])
    (h_range, w_range) = GRID_RANGES[pair_count]
    grid_h = rng.randint(h_range[0], h_range[1])
    grid_w = rng.randint(w_range[0], w_range[1])

    key_colors = rng.sample([1, 2, 3, 4, 6, 7, 8, 9], pair_count)
    used_signatures: set[tuple[str, int, int]] = set()
    objects: list[dict[str, object]] = []

    for key_color in key_colors:
        frame_h = rng.randint(FRAME_H_RANGE[0], FRAME_H_RANGE[1])
        frame_w = rng.randint(FRAME_W_RANGE[0], FRAME_W_RANGE[1])
        signatures = [sig for sig in _candidate_signatures(frame_h, frame_w) if sig not in used_signatures]
        if not signatures:
            return None
        signature = rng.choice(signatures)
        used_signatures.add(signature)

        try:
            key_patch, _inside = _make_key_patch(frame_h, frame_w, signature, key_color, rng)
        except ValueError:
            return None
        lock_inp, lock_out = _lock_patches(frame_h, frame_w, signature, key_color)

        objects.append(
            {
                "shape": key_patch.shape,
                "input_patch": key_patch,
                "output_patch": key_patch.copy(),
            }
        )
        objects.append(
            {
                "shape": lock_inp.shape,
                "input_patch": lock_inp,
                "output_patch": lock_out,
            }
        )

    rng.shuffle(objects)
    objects.sort(key=lambda obj: int(np.prod(obj["shape"])), reverse=True)

    inp = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    out = inp.copy()
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    for obj in objects:
        patch_shape = obj["shape"]
        locations = _free_locations(occupied, patch_shape, OBJECT_PADDING)
        if not locations:
            return None
        r0, c0 = rng.choice(locations)
        input_patch = obj["input_patch"]
        output_patch = obj["output_patch"]
        _paste(inp, input_patch, r0, c0)
        _paste(out, output_patch, r0, c0)
        ph, pw = patch_shape
        occupied[
            max(0, r0 - OBJECT_PADDING) : min(grid_h, r0 + ph + OBJECT_PADDING),
            max(0, c0 - OBJECT_PADDING) : min(grid_w, c0 + pw + OBJECT_PADDING),
        ] = True

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
