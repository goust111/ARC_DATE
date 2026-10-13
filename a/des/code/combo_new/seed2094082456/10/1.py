from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
FRAME_COLOR = 5
NON_SPECIAL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("left", "right", "top", "bottom")
MAX_GENERATION_ATTEMPTS = 400

PATTERN_BANK = {
    2: [
        np.array([[0, 1], [1, 0]], dtype=int),
        np.array([[0, 1], [0, 1]], dtype=int),
        np.array([[0, 0], [1, 1]], dtype=int),
        np.array([[0, 0], [0, 1]], dtype=int),
        np.array([[0, 1], [1, 1]], dtype=int),
    ],
    3: [
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
        np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=int),
        np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=int),
        np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=int),
        np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=int),
    ],
}


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_rect_frame(local_mask: np.ndarray) -> bool:
    h, w = local_mask.shape
    if h < 3 or w < 3:
        return False
    if int(local_mask.sum()) != 2 * (h + w) - 4:
        return False
    if not local_mask[0, :].all() or not local_mask[-1, :].all():
        return False
    if not local_mask[:, 0].all() or not local_mask[:, -1].all():
        return False
    return not local_mask[1:-1, 1:-1].any()


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[comp[:, 0] - r0, comp[:, 1] - c0] = True
            if _is_rect_frame(local):
                frames.append({"color": color, "bbox": (r0, c0, r1, c1), "coords": comp})
    return frames


def _parse_legend_component(
    grid: np.ndarray, comp: np.ndarray
) -> tuple[int, np.ndarray, np.ndarray] | None:
    r0, c0, r1, c1 = _bbox(comp)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    occ = np.zeros((h, w), dtype=bool)
    occ[comp[:, 0] - r0, comp[:, 1] - c0] = True
    total = int(occ.sum())

    if h in (2, 3) and w == h + 1 and total == h * h + 1:
        col_counts = occ.sum(axis=0)
        sparse = [idx for idx, count in enumerate(col_counts) if int(count) == 1]
        dense = [idx for idx, count in enumerate(col_counts) if int(count) == h]
        if (
            len(sparse) == 1
            and len(dense) == h
            and sparse[0] in (0, w - 1)
            and dense == list(range(min(dense), min(dense) + h))
        ):
            patch_c0 = min(dense)
            patch = grid[r0 : r0 + h, c0 + patch_c0 : c0 + patch_c0 + h].copy()
            if len(np.unique(patch)) != 2:
                return None
            key_r = int(np.flatnonzero(occ[:, sparse[0]])[0])
            key = int(grid[r0 + key_r, c0 + sparse[0]])
            return key, patch, comp

    if w in (2, 3) and h == w + 1 and total == w * w + 1:
        row_counts = occ.sum(axis=1)
        sparse = [idx for idx, count in enumerate(row_counts) if int(count) == 1]
        dense = [idx for idx, count in enumerate(row_counts) if int(count) == w]
        if (
            len(sparse) == 1
            and len(dense) == w
            and sparse[0] in (0, h - 1)
            and dense == list(range(min(dense), min(dense) + w))
        ):
            patch_r0 = min(dense)
            patch = grid[r0 + patch_r0 : r0 + patch_r0 + w, c0 : c0 + w].copy()
            if len(np.unique(patch)) != 2:
                return None
            key_c = int(np.flatnonzero(occ[sparse[0], :])[0])
            key = int(grid[r0 + sparse[0], c0 + key_c])
            return key, patch, comp

    return None


def _find_legend_mapping(
    grid: np.ndarray, bg: int, frame_bbox_mask: np.ndarray
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    outside = (grid != bg) & (~frame_bbox_mask)
    mapping: dict[int, np.ndarray] = {}
    legend_mask = np.zeros_like(grid, dtype=bool)
    for comp in _components(outside):
        parsed = _parse_legend_component(grid, comp)
        if parsed is None:
            continue
        key, patch, coords = parsed
        assert key not in mapping, "legend keys must be unique"
        mapping[key] = patch
        legend_mask[coords[:, 0], coords[:, 1]] = True
    return mapping, legend_mask


def _tile_patch(patch: np.ndarray, h: int, w: int) -> np.ndarray:
    ph, pw = patch.shape
    reps = ((h + ph - 1) // ph, (w + pw - 1) // pw)
    return np.tile(patch, reps)[:h, :w]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed region by looking up its marker color in a nearby texture legend and tiling the matched texture.

    Input structure:
    - The background is black.
    - Several one-pixel-thick gray rectangular frames enclose empty interiors; each interior contains exactly one non-black, non-gray marker pixel.
    - Along one grid edge there is a legend made of disconnected entries, each consisting of one color key pixel attached to a 2x2 or 3x3 texture block.

    Rule:
    - Detect the gray rectangular frames and keep their borders unchanged.
    - For each frame, read the interior marker color and find the legend entry with the same key color.
    - Tile that entry's texture from the interior's top-left corner to cover the whole framed interior, replacing the marker pixel.
    - Erase the legend itself to black in the output.

    Key insight:
    - The interior marker is only a lookup key; the actual fill comes from a separate edge legend that encodes a color-to-texture mapping.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    frames = _find_frames(g, bg)
    assert frames, "expected at least one rectangular frame"

    frame_bbox_mask = np.zeros_like(g, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        frame_bbox_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    mapping, legend_mask = _find_legend_mapping(g, bg, frame_bbox_mask)
    assert mapping, "expected at least one legend entry"

    out = g.copy()
    out[legend_mask] = bg

    for frame in frames:
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_coords = np.argwhere((interior != bg) & (interior != color))
        assert marker_coords.shape == (1, 2), "each frame must contain exactly one marker pixel"
        mr, mc = map(int, marker_coords[0])
        key = int(interior[mr, mc])
        assert key in mapping, "every marker color must appear in the legend"
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_patch(mapping[key], r1 - r0 - 1, c1 - c0 - 1)

    return out


def _colorize_pattern(binary_pattern: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    patch = np.full(binary_pattern.shape, color_a, dtype=int)
    patch[binary_pattern.astype(bool)] = color_b
    return patch


def _build_legend_entry(key_color: int, patch: np.ndarray, side: str, rng: random.Random) -> np.ndarray:
    n = int(patch.shape[0])
    if side == "left":
        entry = np.full((n, n + 1), BG, dtype=int)
        entry[:, 1:] = patch
        entry[rng.randrange(n), 0] = key_color
        return entry
    if side == "right":
        entry = np.full((n, n + 1), BG, dtype=int)
        entry[:, :n] = patch
        entry[rng.randrange(n), n] = key_color
        return entry
    if side == "top":
        entry = np.full((n + 1, n), BG, dtype=int)
        entry[1:, :] = patch
        entry[0, rng.randrange(n)] = key_color
        return entry
    if side == "bottom":
        entry = np.full((n + 1, n), BG, dtype=int)
        entry[:n, :] = patch
        entry[n, rng.randrange(n)] = key_color
        return entry
    raise ValueError(f"unsupported side: {side}")


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    target = grid[top : top + h, left : left + w]
    target[mask] = patch[mask]


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR


def _bboxes_conflict(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1
) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _legend_layout(
    entries: list[np.ndarray], side: str, h: int, w: int, rng: random.Random
) -> tuple[list[tuple[int, int]], int]:
    if side in ("left", "right"):
        gaps = [rng.randint(1, 2) for _ in range(max(0, len(entries) - 1))]
        span = sum(entry.shape[0] for entry in entries) + sum(gaps)
        if span + 2 > h:
            raise ValueError("legend entries do not fit vertically")
        strip = max(entry.shape[1] for entry in entries)
        top = rng.randint(1, h - span - 1)
        col = 0 if side == "left" else w - strip
        positions: list[tuple[int, int]] = []
        cursor = top
        for idx, entry in enumerate(entries):
            if side == "left":
                positions.append((cursor, 0))
            else:
                positions.append((cursor, w - entry.shape[1]))
            cursor += entry.shape[0]
            if idx < len(gaps):
                cursor += gaps[idx]
        return positions, strip

    gaps = [rng.randint(1, 2) for _ in range(max(0, len(entries) - 1))]
    span = sum(entry.shape[1] for entry in entries) + sum(gaps)
    if span + 2 > w:
        raise ValueError("legend entries do not fit horizontally")
    strip = max(entry.shape[0] for entry in entries)
    left = rng.randint(1, w - span - 1)
    positions = []
    cursor = left
    for idx, entry in enumerate(entries):
        if side == "top":
            positions.append((0, cursor))
        else:
            positions.append((h - entry.shape[0], cursor))
        cursor += entry.shape[1]
        if idx < len(gaps):
            cursor += gaps[idx]
    return positions, strip


def _sample_entries(
    rng: random.Random,
) -> tuple[list[dict[str, object]], dict[int, np.ndarray], str]:
    entry_count = rng.randint(2, 4)
    key_colors = rng.sample(NON_SPECIAL_COLORS, entry_count)
    texture_pool = [color for color in NON_SPECIAL_COLORS if color not in key_colors]
    side = rng.choice(SIDES)
    used_patterns: set[tuple[int, int]] = set()
    entries: list[dict[str, object]] = []
    mapping: dict[int, np.ndarray] = {}

    for key in key_colors:
        for _ in range(50):
            size = rng.choice((2, 3))
            pattern_idx = rng.randrange(len(PATTERN_BANK[size]))
            if (size, pattern_idx) in used_patterns:
                continue
            used_patterns.add((size, pattern_idx))
            c1, c2 = rng.sample(texture_pool, 2)
            if rng.random() < 0.5:
                c1, c2 = c2, c1
            patch = _colorize_pattern(PATTERN_BANK[size][pattern_idx], c1, c2)
            entry = _build_legend_entry(key, patch, side, rng)
            entries.append({"key": key, "patch": patch, "entry": entry})
            mapping[key] = patch
            break
        else:
            raise RuntimeError("failed to sample a unique legend pattern")
    rng.shuffle(entries)
    return entries, mapping, side


def _frame_zone(h: int, w: int, side: str, strip: int) -> tuple[int, int, int, int]:
    if side == "left":
        return 1, strip + 1, h - 2, w - 2
    if side == "right":
        return 1, 1, h - 2, w - strip - 2
    if side == "top":
        return strip + 1, 1, h - 2, w - 2
    if side == "bottom":
        return 1, 1, h - strip - 2, w - 2
    raise ValueError(f"unsupported side: {side}")


def _place_frames(
    rng: random.Random, zone: tuple[int, int, int, int]
) -> list[dict[str, object]] | None:
    zr0, zc0, zr1, zc1 = zone
    zone_h = zr1 - zr0 + 1
    zone_w = zc1 - zc0 + 1
    if zone_h < 5 or zone_w < 5:
        return None

    area = zone_h * zone_w
    max_regions = min(4, max(1, area // 45))
    region_count = rng.randint(1, max_regions)
    frames: list[dict[str, object]] = []

    for _ in range(region_count):
        placed = False
        for _ in range(200):
            inner_h = rng.randint(2, min(8, zone_h - 2))
            inner_w = rng.randint(2, min(8, zone_w - 2))
            outer_h, outer_w = inner_h + 2, inner_w + 2
            if outer_h > zone_h or outer_w > zone_w:
                continue
            r0 = rng.randint(zr0, zr1 - outer_h + 1)
            c0 = rng.randint(zc0, zc1 - outer_w + 1)
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if any(_bboxes_conflict(bbox, frame["bbox"], pad=1) for frame in frames):
                continue
            frames.append({"bbox": bbox, "inner_shape": (inner_h, inner_w)})
            placed = True
            break
        if not placed:
            return None

    return frames


def _expected_output(
    grid: np.ndarray,
    legend_positions: list[tuple[int, int]],
    entries: list[dict[str, object]],
    region_specs: list[dict[str, object]],
    mapping: dict[int, np.ndarray],
) -> np.ndarray:
    out = np.full_like(grid, BG)
    for spec in region_specs:
        bbox = spec["bbox"]
        key = int(spec["key"])
        _draw_frame(out, bbox)
        r0, c0, r1, c1 = bbox
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_patch(mapping[key], r1 - r0 - 1, c1 - c0 - 1)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    entries, mapping, side = _sample_entries(rng)
    strip = max(entry["entry"].shape[1] if side in ("left", "right") else entry["entry"].shape[0] for entry in entries)

    for _ in range(80):
        h = rng.randint(14, 30)
        w = rng.randint(14, 30)
        try:
            legend_positions, strip = _legend_layout([entry["entry"] for entry in entries], side, h, w, rng)
        except ValueError:
            continue

        zone = _frame_zone(h, w, side, strip)
        frames = _place_frames(rng, zone)
        if frames is None:
            continue

        grid = np.full((h, w), BG, dtype=int)
        for (top, left), entry in zip(legend_positions, entries):
            _paste(grid, entry["entry"], top, left)

        key_sequence = [int(entry["key"]) for entry in entries]
        rng.shuffle(key_sequence)
        region_specs: list[dict[str, object]] = []
        for idx, frame in enumerate(frames):
            bbox = frame["bbox"]
            r0, c0, r1, c1 = bbox
            _draw_frame(grid, bbox)
            inner_h, inner_w = frame["inner_shape"]
            if idx < len(key_sequence):
                key = key_sequence[idx]
            else:
                key = rng.choice([int(entry["key"]) for entry in entries])
            mr = rng.randint(r0 + 1, r1 - 1)
            mc = rng.randint(c0 + 1, c1 - 1)
            grid[mr, mc] = key
            region_specs.append({"bbox": bbox, "key": key, "inner_shape": (inner_h, inner_w)})

        expected = _expected_output(grid, legend_positions, entries, region_specs, mapping)
        if np.array_equal(grid, expected):
            continue
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
