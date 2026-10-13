from __future__ import annotations

import math
import random
from collections import Counter, defaultdict, deque

import numpy as np


INPUT_BG = 8  # azure
OUTPUT_BG = 0  # black
PALETTE_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
GEN_RETRIES = 300

_MASK_STRINGS = {
    (5, 5): [
        (
            "11000",
            "11000",
            "01000",
            "01000",
            "00000",
        ),
        (
            "11000",
            "11100",
            "01100",
            "01100",
            "00000",
        ),
        (
            "00011",
            "11001",
            "11001",
            "11011",
            "11111",
        ),
        (
            "11100",
            "01000",
            "01100",
            "01110",
            "00000",
        ),
        (
            "00011",
            "10111",
            "10111",
            "10011",
            "11111",
        ),
    ],
    (5, 6): [
        (
            "110000",
            "111000",
            "001100",
            "000000",
            "000000",
        ),
        (
            "011111",
            "010001",
            "000011",
            "010001",
            "111111",
        ),
        (
            "000111",
            "110001",
            "100101",
            "101111",
            "111111",
        ),
        (
            "111110",
            "111110",
            "111110",
            "101110",
            "000000",
        ),
        (
            "111000",
            "101100",
            "101100",
            "101000",
            "000000",
        ),
    ],
    (6, 5): [
        (
            "00111",
            "10001",
            "11101",
            "11001",
            "11111",
            "11111",
        ),
        (
            "11110",
            "01110",
            "01010",
            "01010",
            "00000",
            "00000",
        ),
        (
            "11100",
            "10100",
            "00110",
            "01100",
            "01100",
            "00000",
        ),
        (
            "11110",
            "01000",
            "01110",
            "00000",
            "00000",
            "00000",
        ),
        (
            "11110",
            "10000",
            "10110",
            "10100",
            "11100",
            "00000",
        ),
    ],
}


def _parse_mask(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


MASK_LIBRARY = {
    shape: tuple(_parse_mask(rows) for rows in masks)
    for shape, masks in _MASK_STRINGS.items()
}


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _full_mask(coords: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((frame_h, frame_w), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    if h > frame_h or w > frame_w:
        raise AssertionError("fragment exceeds inferred frame")
    return mask


def _parse_legend_component(g: np.ndarray, coords: np.ndarray) -> tuple[tuple[int, int], int] | None:
    if int(coords.shape[0]) != 3:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None
    local = g[r0 : r1 + 1, c0 : c1 + 1]
    occupied = local != INPUT_BG
    if not bool(occupied[0, 0] and occupied[0, 1]):
        return None
    if int(occupied[1].sum()) != 1:
        return None
    pair = tuple(sorted((int(local[0, 0]), int(local[0, 1]))))
    target = int(local[1, 0] if occupied[1, 0] else local[1, 1])
    return pair, target


def _render_output(frame_h: int, frame_w: int, ordered_targets: list[int]) -> np.ndarray:
    width = len(ordered_targets) * frame_w + max(0, len(ordered_targets) - 1)
    out = np.full((frame_h, width), OUTPUT_BG, dtype=int)
    for idx, color in enumerate(ordered_targets):
        c0 = idx * (frame_w + 1)
        out[:, c0 : c0 + frame_w] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble complementary scattered fragments into full rectangles, then recolor each rectangle via a legend lookup.

    Input structure:
    - The background is azure.
    - Several monochrome 4-connected fragments are scattered across the grid; each fragment occupies some cells inside a shared invisible top-left-anchored frame.
    - Fragments come in complementary pairs: padding each fragment to the common frame size at its own top-left anchor makes the two masks exact complements.
    - The lower-right corner contains a color legend made of 2-cell horizontal bars with one target pixel directly below the left or right bar cell.

    Rule:
    - Read every legend entry as an unordered fragment-color pair mapped to one output color.
    - Extract all non-legend components, infer the common frame size from the maximum fragment extent, and pad every fragment mask to that shared frame.
    - Pair fragments whose padded masks are exact complements, then sort the pairs by the earlier top-left fragment position in the input.
    - For each pair, output one solid full rectangle of the legend-mapped color; place all rectangles in one compact row on a black background with a 1-cell gap.

    Key insight:
    - The pairing cue is geometric complementarity inside a shared hidden frame, while the final rectangle color comes from a separate legend lookup rather than either fragment's visible color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend: dict[tuple[int, int], int] = {}
    fragments: list[dict[str, object]] = []

    for coords in _components(g != INPUT_BG):
        parsed = _parse_legend_component(g, coords)
        if parsed is not None:
            pair_key, target = parsed
            legend[pair_key] = int(target)
            continue

        r0, c0, r1, c1 = _bbox(coords)
        window = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = np.unique(window[window != INPUT_BG])
        assert colors.size == 1, "fragments must be monochrome"
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        rel = coords - np.array([r0, c0], dtype=int)
        mask[rel[:, 0], rel[:, 1]] = True
        fragments.append(
            {
                "coords": coords,
                "anchor": (int(r0), int(c0)),
                "local_mask": mask,
                "color": int(colors[0]),
            }
        )

    assert legend, "legend must be present"
    assert fragments and len(fragments) % 2 == 0, "fragments must form pairs"

    frame_h = max(int(rec["local_mask"].shape[0]) for rec in fragments)
    frame_w = max(int(rec["local_mask"].shape[1]) for rec in fragments)

    mask_to_indices: defaultdict[bytes, list[int]] = defaultdict(list)
    for idx, rec in enumerate(fragments):
        full = np.zeros((frame_h, frame_w), dtype=bool)
        local = rec["local_mask"]
        full[: local.shape[0], : local.shape[1]] = local
        rec["full_mask"] = full
        mask_to_indices[full.tobytes()].append(idx)

    used: set[int] = set()
    solved: list[tuple[tuple[int, int], int]] = []
    for idx, rec in enumerate(fragments):
        if idx in used:
            continue
        comp_key = np.logical_not(rec["full_mask"]).tobytes()
        candidates = [
            j for j in mask_to_indices.get(comp_key, [])
            if j not in used and j != idx
        ]
        assert len(candidates) == 1, "complementary pairing must be unique"
        other = fragments[candidates[0]]
        color_key = tuple(sorted((int(rec["color"]), int(other["color"]))))
        assert color_key in legend, "missing legend mapping for fragment colors"
        pair_anchor = min(rec["anchor"], other["anchor"])
        solved.append((pair_anchor, int(legend[color_key])))
        used.add(idx)
        used.add(candidates[0])

    assert len(used) == len(fragments)
    solved.sort(key=lambda item: item[0])
    ordered_targets = [color for _anchor, color in solved]
    return _render_output(frame_h, frame_w, ordered_targets)


def _background_is_mode(grid: np.ndarray) -> bool:
    counts = Counter(int(v) for v in grid.ravel())
    bg_count = counts.pop(INPUT_BG, 0)
    return bg_count > max(counts.values(), default=0)


def _place_legend(
    grid: np.ndarray,
    occupied: np.ndarray,
    mappings: list[tuple[int, int, int]],
    rng: random.Random,
) -> None:
    entries = mappings[:]
    rng.shuffle(entries)
    legend_h = 3 * len(entries) - 1
    legend_w = 2
    top = grid.shape[0] - legend_h
    left = grid.shape[1] - legend_w

    occupied[max(0, top - 1) :, max(0, left - 1) :] = True
    for idx, (c1, c2, target) in enumerate(entries):
        r = top + idx * 3
        if rng.random() < 0.5:
            grid[r, left] = int(c1)
            grid[r, left + 1] = int(c2)
        else:
            grid[r, left] = int(c2)
            grid[r, left + 1] = int(c1)
        if rng.random() < 0.5:
            grid[r + 1, left] = int(target)
        else:
            grid[r + 1, left + 1] = int(target)


def _candidate_positions(
    occupied: np.ndarray,
    frame_h: int,
    frame_w: int,
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions: list[tuple[int, int]] = []
    for r in range(h - frame_h + 1):
        for c in range(w - frame_w + 1):
            rr0 = max(0, r - 1)
            rr1 = min(h, r + frame_h + 1)
            cc0 = max(0, c - 1)
            cc1 = min(w, c + frame_w + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r, c))
    return positions


def _paint_fragment(
    grid: np.ndarray,
    occupied: np.ndarray,
    top: int,
    left: int,
    mask: np.ndarray,
    frame_h: int,
    frame_w: int,
    color: int,
) -> None:
    rr0 = max(0, top - 1)
    rr1 = min(grid.shape[0], top + frame_h + 1)
    cc0 = max(0, left - 1)
    cc1 = min(grid.shape[1], left + frame_w + 1)
    occupied[rr0:rr1, cc0:cc1] = True

    pts = np.argwhere(mask)
    for r, c in pts:
        grid[top + int(r), left + int(c)] = int(color)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_h, frame_w = rng.choice(tuple(MASK_LIBRARY.keys()))
    mask_options = list(MASK_LIBRARY[(frame_h, frame_w)])
    pair_count = rng.randint(2, 4)
    if pair_count > len(mask_options):
        return None
    rng.shuffle(mask_options)
    masks = mask_options[:pair_count]

    fragment_colors = rng.sample(PALETTE_COLORS, 2 * pair_count)
    target_colors = [rng.choice(PALETTE_COLORS) for _ in range(pair_count)]

    slot_cols = min(3, max(2, math.ceil(math.sqrt(2 * pair_count))))
    slot_rows = math.ceil((2 * pair_count) / slot_cols)
    legend_h = 3 * pair_count - 1
    min_h = max(slot_rows * frame_h + (slot_rows - 1) + 2, legend_h + 2, 16)
    min_w = max(slot_cols * frame_w + (slot_cols - 1) + 3, 18)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = rng.randint(min_h, min(30, min_h + 6))
    grid_w = rng.randint(min_w, min(30, min_w + 8))
    grid = np.full((grid_h, grid_w), INPUT_BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    mappings: list[tuple[int, int, int]] = []
    pieces: list[dict[str, object]] = []
    expected_pairs: list[tuple[tuple[int, int], int]] = []

    for idx, mask_a in enumerate(masks):
        color_a = fragment_colors[2 * idx]
        color_b = fragment_colors[2 * idx + 1]
        target = target_colors[idx]
        mappings.append((color_a, color_b, target))
        pieces.append({"pair": idx, "mask": mask_a, "color": color_a})
        pieces.append({"pair": idx, "mask": np.logical_not(mask_a), "color": color_b})
        expected_pairs.append(((10**9, 10**9), target))

    _place_legend(grid, occupied, mappings, rng)
    rng.shuffle(pieces)

    anchors: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for piece in pieces:
        candidates = _candidate_positions(occupied, frame_h, frame_w)
        if not candidates:
            return None
        top, left = rng.choice(candidates)
        _paint_fragment(
            grid,
            occupied,
            top,
            left,
            piece["mask"],
            frame_h,
            frame_w,
            int(piece["color"]),
        )
        anchors[int(piece["pair"])].append((top, left))

    ordered = []
    for pair_idx, (_c1, _c2, target) in enumerate(mappings):
        if len(anchors[pair_idx]) != 2:
            return None
        ordered.append((min(anchors[pair_idx]), int(target)))
    ordered.sort(key=lambda item: item[0])
    expected = _render_output(frame_h, frame_w, [target for _anchor, target in ordered])

    if not _background_is_mode(grid):
        return None
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
