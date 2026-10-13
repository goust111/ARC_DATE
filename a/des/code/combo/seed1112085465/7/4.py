from __future__ import annotations

import random

import numpy as np


BG = 8
FRAME = 5
KNOWN = 9

WINDOW_SIZE = 5
INNER_SIZE = 3

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 32)
WINDOW_COUNT_RANGE = (2, 5)
OBJECT_PAD = 1
GEN_RETRIES = 400

FRAGMENT_COLORS = [0, 1, 2, 3, 4, 6, 7]

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

WINDOW_BORDER_MASK = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=bool)
WINDOW_BORDER_MASK[0, :] = True
WINDOW_BORDER_MASK[-1, :] = True
WINDOW_BORDER_MASK[:, 0] = True
WINDOW_BORDER_MASK[:, -1] = True


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIRS4 if connectivity == 4 else DIRS8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in dirs:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append(np.array(coords, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _mask_from_coords(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _mask_key(mask: np.ndarray) -> bytes:
    return mask.astype(np.uint8).tobytes()


def _find_windows(grid: np.ndarray) -> list[dict[str, object]]:
    windows: list[dict[str, object]] = []

    for coords in _components(grid == FRAME, connectivity=4):
        bbox = _bbox(coords)
        if bbox[2] - bbox[0] != WINDOW_SIZE - 1 or bbox[3] - bbox[1] != WINDOW_SIZE - 1:
            continue

        border_mask = _mask_from_coords(coords, bbox)
        if not np.array_equal(border_mask, WINDOW_BORDER_MASK):
            continue

        r0, c0, _, _ = bbox
        inner = grid[r0 + 1 : r0 + 1 + INNER_SIZE, c0 + 1 : c0 + 1 + INNER_SIZE]
        if inner.shape != (INNER_SIZE, INNER_SIZE):
            continue
        if not np.all(np.isin(inner, [BG, KNOWN])):
            continue

        missing_mask = inner == BG
        missing_count = int(missing_mask.sum())
        if not 3 <= missing_count <= 5:
            continue

        windows.append({"bbox": bbox, "missing_mask": missing_mask.copy()})

    windows.sort(key=lambda item: tuple(item["bbox"]))  # type: ignore[index]
    assert windows, "expected at least one valid 5x5 grey window"
    return windows


def _find_fragments(grid: np.ndarray, window_bboxes: list[tuple[int, int, int, int]]) -> list[dict[str, object]]:
    outside = np.array(grid, copy=True)
    for r0, c0, r1, c1 in window_bboxes:
        outside[r0 : r1 + 1, c0 : c1 + 1] = BG

    h, w = outside.shape
    fragments: list[dict[str, object]] = []

    for r0 in range(h - INNER_SIZE + 1):
        for c0 in range(w - INNER_SIZE + 1):
            sub = outside[r0 : r0 + INNER_SIZE, c0 : c0 + INNER_SIZE]
            colors = [int(v) for v in np.unique(sub) if int(v) != BG]
            if len(colors) != 1:
                continue

            color = colors[0]
            if color in (FRAME, KNOWN):
                continue

            mask = sub == color
            count = int(mask.sum())
            if not 3 <= count <= 5:
                continue

            rows, cols = np.where(mask)
            if rows.min() != 0 or rows.max() != INNER_SIZE - 1:
                continue
            if cols.min() != 0 or cols.max() != INNER_SIZE - 1:
                continue

            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + INNER_SIZE + 1)
            cc1 = min(w, c0 + INNER_SIZE + 1)
            ring = outside[rr0:rr1, cc0:cc1].copy()
            ring[r0 - rr0 : r0 - rr0 + INNER_SIZE, c0 - cc0 : c0 - cc0 + INNER_SIZE] = BG
            if np.any(ring != BG):
                continue

            fragments.append(
                {
                    "bbox": (r0, c0, r0 + INNER_SIZE - 1, c0 + INNER_SIZE - 1),
                    "mask": mask.copy(),
                    "color": color,
                }
            )

    fragments.sort(key=lambda item: tuple(item["bbox"]))  # type: ignore[index]
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each window's missing 3x3 cells with the uniquely matching outside fragment color.
    """
    g = np.asarray(grid, dtype=int)
    out = np.array(g, copy=True)

    windows = _find_windows(g)
    fragments = _find_fragments(g, [item["bbox"] for item in windows])  # type: ignore[index]

    by_shape: dict[bytes, list[int]] = {}
    for i, fragment in enumerate(fragments):
        by_shape.setdefault(_mask_key(fragment["mask"]), []).append(i)  # type: ignore[index]

    used: set[int] = set()
    for window in windows:
        missing_mask = window["missing_mask"]  # type: ignore[assignment]
        key = _mask_key(missing_mask)
        candidates = [idx for idx in by_shape.get(key, []) if idx not in used]
        assert len(candidates) == 1, "each window must have a unique matching fragment"

        fragment = fragments[candidates[0]]
        used.add(candidates[0])

        wr0, wc0, _, _ = window["bbox"]  # type: ignore[index]
        inner = out[wr0 + 1 : wr0 + 1 + INNER_SIZE, wc0 + 1 : wc0 + 1 + INNER_SIZE]
        inner[missing_mask] = int(fragment["color"])  # type: ignore[index]

        fr0, fc0, fr1, fc1 = fragment["bbox"]  # type: ignore[index]
        patch = out[fr0 : fr1 + 1, fc0 : fc1 + 1]
        patch[fragment["mask"]] = BG  # type: ignore[index]

    return out


def _is_connected8(mask: np.ndarray) -> bool:
    comps = _components(mask, connectivity=8)
    return len(comps) == 1


def _build_fragment_library() -> dict[int, list[np.ndarray]]:
    library: dict[int, list[np.ndarray]] = {3: [], 4: [], 5: []}
    for bits in range(1 << (INNER_SIZE * INNER_SIZE)):
        mask = np.array([(bits >> i) & 1 for i in range(INNER_SIZE * INNER_SIZE)], dtype=bool)
        mask = mask.reshape(INNER_SIZE, INNER_SIZE)
        count = int(mask.sum())
        if count not in library:
            continue

        rows, cols = np.where(mask)
        if rows.min() != 0 or rows.max() != INNER_SIZE - 1:
            continue
        if cols.min() != 0 or cols.max() != INNER_SIZE - 1:
            continue
        if not _is_connected8(mask):
            continue

        library[count].append(mask.copy())

    return library


FRAGMENT_LIBRARY = _build_fragment_library()


def _reserve(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = OBJECT_PAD) -> None:
    hh, ww = blocked.shape
    blocked[max(0, r0 - pad) : min(hh, r0 + h + pad), max(0, c0 - pad) : min(ww, c0 + w + pad)] = True


def _place_box(blocked: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    hh, ww = blocked.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(hh - h + 1):
        for c0 in range(ww - w + 1):
            if not blocked[r0 : r0 + h, c0 : c0 + w].any():
                candidates.append((r0, c0))

    if not candidates:
        return None

    r0, c0 = random.choice(candidates)
    _reserve(blocked, r0, c0, h, w)
    return r0, c0


def _sample_fragment_masks(count: int) -> list[np.ndarray] | None:
    chosen: list[np.ndarray] = []
    used: set[bytes] = set()
    for _ in range(count):
        area = random.choice([3, 4, 5])
        candidates = [mask for mask in FRAGMENT_LIBRARY[area] if _mask_key(mask) not in used]
        if not candidates:
            return None
        mask = random.choice(candidates).copy()
        used.add(_mask_key(mask))
        chosen.append(mask)
    return chosen


def _draw_window(grid: np.ndarray, r0: int, c0: int, fragment_mask: np.ndarray) -> None:
    grid[r0, c0 : c0 + WINDOW_SIZE] = FRAME
    grid[r0 + WINDOW_SIZE - 1, c0 : c0 + WINDOW_SIZE] = FRAME
    grid[r0 : r0 + WINDOW_SIZE, c0] = FRAME
    grid[r0 : r0 + WINDOW_SIZE, c0 + WINDOW_SIZE - 1] = FRAME

    inner = grid[r0 + 1 : r0 + 1 + INNER_SIZE, c0 + 1 : c0 + 1 + INNER_SIZE]
    inner[:] = BG
    inner[~fragment_mask] = KNOWN


def _draw_fragment(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int) -> None:
    patch = grid[r0 : r0 + INNER_SIZE, c0 : c0 + INNER_SIZE]
    patch[mask] = color


def _validate_generated(grid: np.ndarray, expected_windows: int) -> bool:
    windows = _find_windows(grid)
    if len(windows) != expected_windows:
        return False

    fragments = _find_fragments(grid, [item["bbox"] for item in windows])  # type: ignore[index]
    if len(fragments) != expected_windows:
        return False

    counts: dict[bytes, int] = {}
    for fragment in fragments:
        key = _mask_key(fragment["mask"])  # type: ignore[index]
        counts[key] = counts.get(key, 0) + 1
    if any(v != 1 for v in counts.values()):
        return False

    for window in windows:
        if counts.get(_mask_key(window["missing_mask"]), 0) != 1:  # type: ignore[index]
            return False

    out = main(grid)
    for window in windows:
        wr0, wc0, _, _ = window["bbox"]  # type: ignore[index]
        missing_mask = window["missing_mask"]  # type: ignore[assignment]
        inner = out[wr0 + 1 : wr0 + 1 + INNER_SIZE, wc0 + 1 : wc0 + 1 + INNER_SIZE]
        if np.any(inner == BG):
            return False
        if not np.all(inner[~missing_mask] == KNOWN):
            return False

    for fragment in fragments:
        fr0, fc0, fr1, fc1 = fragment["bbox"]  # type: ignore[index]
        patch = out[fr0 : fr1 + 1, fc0 : fc1 + 1]
        if np.any(patch[fragment["mask"]] != BG):  # type: ignore[index]
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    window_count = random.randint(*WINDOW_COUNT_RANGE)
    min_h = max(GRID_H_RANGE[0], 12 + 2 * window_count)
    min_w = max(GRID_W_RANGE[0], 12 + 2 * window_count)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    h = random.randint(min_h, GRID_H_RANGE[1])
    w = random.randint(min_w, GRID_W_RANGE[1])
    grid = np.full((h, w), BG, dtype=int)

    fragment_masks = _sample_fragment_masks(window_count)
    if fragment_masks is None:
        return None
    fragment_colors = random.sample(FRAGMENT_COLORS, window_count)

    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True

    window_positions: list[tuple[int, int]] = []
    for _ in range(window_count):
        pos = _place_box(blocked, WINDOW_SIZE, WINDOW_SIZE)
        if pos is None:
            return None
        window_positions.append(pos)

    fragment_positions: list[tuple[int, int]] = []
    for _ in range(window_count):
        pos = _place_box(blocked, INNER_SIZE, INNER_SIZE)
        if pos is None:
            return None
        fragment_positions.append(pos)

    for (r0, c0), mask in zip(window_positions, fragment_masks):
        _draw_window(grid, r0, c0, mask)

    for (r0, c0), mask, color in zip(fragment_positions, fragment_masks, fragment_colors):
        _draw_fragment(grid, r0, c0, mask, color)

    if not _validate_generated(grid, window_count):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
