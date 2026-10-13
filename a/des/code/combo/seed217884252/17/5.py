from __future__ import annotations

import numpy as np


BG_COLOR = 7
PALETTE = (1, 2, 3, 4, 5, 6, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
RNG = np.random.default_rng()


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            bbox = (min(rows), min(cols), max(rows), max(cols))
            components.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "bbox": bbox,
                }
            )

    return components


def _component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _horizontal_score(mask: np.ndarray) -> int:
    half = mask.shape[0] // 2
    if half == 0:
        return 0
    return int(np.logical_and(mask[:half, :], mask[::-1, :][:half, :]).sum())


def _vertical_score(mask: np.ndarray) -> int:
    half = mask.shape[1] // 2
    if half == 0:
        return 0
    return int(np.logical_and(mask[:, :half], mask[:, ::-1][:, :half]).sum())


def _fold_horizontal(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    half = mask.shape[0] // 2
    if half > 0:
        out[:half, :] = np.logical_and(mask[:half, :], mask[::-1, :][:half, :])
    return out


def _fold_vertical(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    half = mask.shape[1] // 2
    if half > 0:
        out[:, :half] = np.logical_and(mask[:, :half], mask[:, ::-1][:, :half])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each foreground object across its stronger bbox symmetry axis and keep only overlaps.

    Input structure:
    - The background is orange.
    - Foreground objects are monochrome 4-connected components.
    - Each object is almost symmetric across either the bbox horizontal midline
      or the bbox vertical midline.

    Rule:
    - For each object, compare horizontal and vertical reflected overlap counts.
    - Choose the axis with the larger overlap count.
    - Keep only pixels whose mirrored counterpart also exists.
    - Show the folded result in place using the upper half or left half only.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG_COLOR if np.any(g == BG_COLOR) else _majority_color(g)
    out = np.full_like(g, bg)

    for comp in _find_components(g, bg):
        coords = comp["coords"]
        bbox = comp["bbox"]
        color = int(comp["color"])
        mask = _component_mask(coords, bbox)

        horizontal_score = _horizontal_score(mask)
        vertical_score = _vertical_score(mask)
        folded = _fold_horizontal(mask) if horizontal_score >= vertical_score else _fold_vertical(mask)

        r0, c0, r1, c1 = bbox
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[folded] = color

    return out


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False

    seen = {tuple(map(int, cells[0]))}
    stack = [tuple(map(int, cells[0]))]
    h, w = mask.shape

    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not mask[nr, nc]:
                continue
            key = (nr, nc)
            if key in seen:
                continue
            seen.add(key)
            stack.append(key)

    return len(seen) == int(mask.sum())


def _grow_connected_mask(height: int, width: int, target: int, anchor: str) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    if anchor == "bottom":
        seed = (height - 1, int(RNG.integers(0, width)))
    elif anchor == "right":
        seed = (int(RNG.integers(0, height)), width - 1)
    else:
        raise AssertionError(f"unknown anchor: {anchor}")
    mask[seed] = True

    while int(mask.sum()) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            for dr, dc in DIR4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        pick = frontier[int(RNG.integers(0, len(frontier)))]
        mask[pick] = True

    return mask


def _add_horizontal_noise(mask: np.ndarray, side: str, desired: int) -> int:
    h, w = mask.shape
    half = h // 2
    rows = range(0, half) if side == "top" else range(h - half, h)
    added = 0

    for _ in range(desired * 10):
        candidates: list[tuple[int, int]] = []
        for r in rows:
            for c in range(w):
                if mask[r, c] or mask[h - 1 - r, c]:
                    continue
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        candidates.append((r, c))
                        break
        if not candidates:
            break
        pick = candidates[int(RNG.integers(0, len(candidates)))]
        mask[pick] = True
        added += 1
        if added >= desired:
            break

    return added


def _add_vertical_noise(mask: np.ndarray, side: str, desired: int) -> int:
    h, w = mask.shape
    half = w // 2
    cols = range(0, half) if side == "left" else range(w - half, w)
    added = 0

    for _ in range(desired * 10):
        candidates: list[tuple[int, int]] = []
        for r in range(h):
            for c in cols:
                if mask[r, c] or mask[r, w - 1 - c]:
                    continue
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        candidates.append((r, c))
                        break
        if not candidates:
            break
        pick = candidates[int(RNG.integers(0, len(candidates)))]
        mask[pick] = True
        added += 1
        if added >= desired:
            break

    return added


def _make_horizontal_object() -> np.ndarray | None:
    for _ in range(120):
        h = int(RNG.integers(4, 9))
        w = int(RNG.integers(4, 9))
        half = h // 2
        area = int(RNG.integers(max(3, half), max(4, half * w - 1) + 1))
        keep = _grow_connected_mask(half, w, area, anchor="bottom")
        if keep is None or not np.any(keep[-1, :]):
            continue

        mask = np.zeros((h, w), dtype=bool)
        mask[:half, :] = keep
        mask[h - half :, :] |= keep[::-1, :]

        if h % 2 == 1:
            bridge_cols = np.flatnonzero(keep[-1, :])
            count = int(RNG.integers(1, min(3, len(bridge_cols)) + 1))
            chosen = RNG.choice(bridge_cols, size=count, replace=False)
            mask[half, chosen] = True

        side = "top" if int(RNG.integers(0, 2)) == 0 else "bottom"
        if _add_horizontal_noise(mask, side, int(RNG.integers(1, 4))) == 0:
            continue
        if not _is_connected(mask):
            continue

        horizontal_score = _horizontal_score(mask)
        vertical_score = _vertical_score(mask)
        if horizontal_score < 3 or horizontal_score <= vertical_score:
            continue
        return mask

    return None


def _make_vertical_object() -> np.ndarray | None:
    for _ in range(120):
        h = int(RNG.integers(4, 9))
        w = int(RNG.integers(4, 9))
        half = w // 2
        area = int(RNG.integers(max(3, half), max(4, h * half - 1) + 1))
        keep = _grow_connected_mask(h, half, area, anchor="right")
        if keep is None or not np.any(keep[:, -1]):
            continue

        mask = np.zeros((h, w), dtype=bool)
        mask[:, :half] = keep
        mask[:, w - half :] |= keep[:, ::-1]

        if w % 2 == 1:
            bridge_rows = np.flatnonzero(keep[:, -1])
            count = int(RNG.integers(1, min(3, len(bridge_rows)) + 1))
            chosen = RNG.choice(bridge_rows, size=count, replace=False)
            mask[chosen, half] = True

        side = "left" if int(RNG.integers(0, 2)) == 0 else "right"
        if _add_vertical_noise(mask, side, int(RNG.integers(1, 4))) == 0:
            continue
        if not _is_connected(mask):
            continue

        horizontal_score = _horizontal_score(mask)
        vertical_score = _vertical_score(mask)
        if vertical_score < 3 or vertical_score <= horizontal_score:
            continue
        return mask

    return None


def _make_object(axis: str) -> np.ndarray | None:
    if axis == "H":
        return _make_horizontal_object()
    if axis == "V":
        return _make_vertical_object()
    raise AssertionError(f"unknown axis: {axis}")


def _reserve_free(reserved: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = reserved.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + height + 1)
    c1 = min(w, left + width + 1)
    return not reserved[r0:r1, c0:c1].any()


def _mark_reserved(reserved: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    h, w = reserved.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + height + 1)
    c1 = min(w, left + width + 1)
    reserved[r0:r1, c0:c1] = True


def _validate_scene(grid: np.ndarray) -> bool:
    comps = _find_components(grid, BG_COLOR)
    if len(comps) < 2:
        return False

    axes = set()
    for comp in comps:
        mask = _component_mask(comp["coords"], comp["bbox"])
        horizontal_score = _horizontal_score(mask)
        vertical_score = _vertical_score(mask)
        if horizontal_score == vertical_score:
            return False
        axes.add("H" if horizontal_score > vertical_score else "V")

    return axes == {"H", "V"} and not np.array_equal(main(grid), grid)


def _gen_attempt() -> np.ndarray | None:
    height = int(RNG.integers(17, 25))
    width = int(RNG.integers(17, 25))
    grid = np.full((height, width), BG_COLOR, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    count = int(RNG.integers(2, 5))
    axes = ["H", "V"]
    while len(axes) < count:
        axes.append("H" if int(RNG.integers(0, 2)) == 0 else "V")
    RNG.shuffle(axes)

    colors = list(RNG.choice(np.array(PALETTE, dtype=int), size=count, replace=False))

    for axis, color in zip(axes, colors):
        mask = _make_object(axis)
        if mask is None:
            return None
        obj_h, obj_w = mask.shape

        positions = [
            (r0, c0)
            for r0 in range(0, height - obj_h + 1)
            for c0 in range(0, width - obj_w + 1)
            if _reserve_free(reserved, r0, c0, obj_h, obj_w)
        ]
        if not positions:
            return None

        r0, c0 = positions[int(RNG.integers(0, len(positions)))]
        patch = grid[r0 : r0 + obj_h, c0 : c0 + obj_w]
        patch[mask] = int(color)
        _mark_reserved(reserved, r0, c0, obj_h, obj_w)

    if not _validate_scene(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
