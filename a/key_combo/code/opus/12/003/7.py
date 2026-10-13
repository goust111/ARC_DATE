from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np


BG = 0
MARKER = 5
FRAGMENT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
FOUR_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _find_components(grid: np.ndarray, *, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in FOUR_DIRS:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _component_sprite(grid: np.ndarray, comp: dict, *, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    coords = comp["coords"]
    sprite[coords[:, 0] - r0, coords[:, 1] - c0] = int(comp["color"])
    return sprite


def _orientations(arr: np.ndarray) -> list[np.ndarray]:
    outs = []
    base = np.asarray(arr, dtype=int)
    for k in range(4):
        rot = np.rot90(base, k)
        outs.append(rot.copy())
        outs.append(np.fliplr(rot).copy())
    unique: list[np.ndarray] = []
    for cand in outs:
        if not any(cand.shape == prev.shape and np.array_equal(cand, prev) for prev in unique):
            unique.append(cand)
    return unique


def _extract_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    comps = _find_components(g, bg=bg)
    markers = [comp for comp in comps if int(comp["color"]) == MARKER and int(comp["area"]) == 1]
    fragments = [comp for comp in comps if int(comp["color"]) != MARKER]
    assert 3 <= len(fragments) <= 4, "expected 3-4 fragments"
    assert len(markers) == len(fragments), "expected one marker per fragment"
    assert all(int(comp["area"]) >= 3 for comp in fragments), "fragments must have area >= 3"
    return bg, fragments, markers


def _match_markers_to_fragments(
    grid: np.ndarray, fragments: list[dict], markers: list[dict], *, bg: int
) -> dict[int, int]:
    coord_to_fragment: dict[tuple[int, int], int] = {}
    bottom_rows = []
    for idx, frag in enumerate(fragments):
        bottom = int(frag["bbox"][2])
        bottom_rows.append(bottom)
        for rr, cc in frag["coords"]:
            coord_to_fragment[(int(rr), int(cc))] = idx

    gap_by_fragment: dict[int, int] = {}
    used_fragments = set()
    for marker in markers:
        mr, mc = map(int, marker["coords"][0])
        hit = None
        for rr in range(mr - 1, -1, -1):
            if int(grid[rr, mc]) != bg:
                hit = (rr, mc)
                break
        assert hit is not None, "every marker must see a fragment upward"
        frag_idx = coord_to_fragment.get(hit)
        assert frag_idx is not None, "marker sightline must hit a fragment"
        assert hit[0] == bottom_rows[frag_idx], "marker must hit the fragment's bottom row"
        assert frag_idx not in used_fragments, "marker-fragment pairing must be one-to-one"
        used_fragments.add(frag_idx)
        gap_by_fragment[frag_idx] = int(mr - hit[0] - 1)

    assert len(gap_by_fragment) == len(fragments), "every fragment must be matched"
    gaps = list(gap_by_fragment.values())
    assert len(gaps) == len(set(gaps)), "marker gaps must be distinct"
    return gap_by_fragment


def _pack_fragments(
    fragments: list[dict],
    gap_by_fragment: dict[int, int],
    *,
    bg: int,
) -> np.ndarray:
    enriched = []
    for idx, frag in enumerate(fragments):
        sprite = _component_sprite(frag["grid"], frag, bg=bg) if "grid" in frag else frag["sprite"]
        r0, c0, r1, c1 = frag["bbox"]
        enriched.append(
            {
                "idx": idx,
                "area": int(frag["area"]),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "sprite": sprite,
                "gap": int(gap_by_fragment[idx]),
            }
        )

    ordered = sorted(
        enriched,
        key=lambda item: (item["area"], item["bbox"][1], item["bbox"][0]),
    )
    out_h = max(item["sprite"].shape[0] for item in ordered)
    out_w = sum(item["sprite"].shape[1] + item["gap"] for item in ordered)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    cursor = 0
    for item in ordered:
        cursor += item["gap"]
        sprite = item["sprite"]
        sh, sw = sprite.shape
        _paste(out, sprite, out_h - sh, cursor, transparent=bg)
        cursor += sw
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered fragments by sorting them by area and using marker sightline gaps as left padding.

    Input structure:
    - The background is black.
    - The upper part contains 3-4 monochrome 4-connected fragments with distinct areas.
    - The lower part contains the same number of grey single-pixel markers.
    - Each marker has a clear vertical sightline upward to a bottom-row pixel of exactly one fragment.

    Rule:
    - Pair every grey marker with the fragment first seen directly above it and measure the background gap length.
    - Sort fragments by increasing area; if areas tie, use the fragment bbox left coordinate.
    - Build a new grid whose height is the tallest fragment sprite, bottom-align every cropped fragment sprite, and
      insert a number of black columns before each fragment equal to its own marker gap.
    - Preserve fragment colors and internal holes; omit the grey markers from the output.

    Key insight:
    - The marker does not move or recolor anything: its vertical distance to the fragment encodes how much blank
      space must appear immediately to that fragment's left after the fragments are reordered by area.
    """
    g = np.asarray(grid, dtype=int)
    bg, fragments, markers = _extract_scene(g)
    for frag in fragments:
        frag["grid"] = g
        frag["sprite"] = _component_sprite(g, frag, bg=bg)
    gap_by_fragment = _match_markers_to_fragments(g, fragments, markers, bg=bg)
    return _pack_fragments(fragments, gap_by_fragment, bg=bg)


def _is_connected(mask: np.ndarray) -> bool:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not cells:
        return False
    seen = {cells[0]}
    stack = [cells[0]]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in FOUR_DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _random_shape(area: int) -> np.ndarray:
    dims = [
        (h, w)
        for h in range(2, 6)
        for w in range(2, 6)
        if area <= h * w and area * 10 >= 5 * h * w
    ]
    random.shuffle(dims)
    for h, w in dims:
        for _ in range(120):
            seed = (random.randrange(h), random.randrange(w))
            cells = {seed}
            while len(cells) < area:
                frontier = []
                for r, c in cells:
                    for dr, dc in FOUR_DIRS:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                            frontier.append((nr, nc))
                if not frontier:
                    break
                cells.add(random.choice(frontier))
            if len(cells) != area:
                continue
            mask = np.zeros((h, w), dtype=bool)
            for r, c in cells:
                mask[r, c] = True
            coords = np.argwhere(mask)
            r0, c0, r1, c1 = _bbox_coords(coords)
            mask = mask[r0 : r1 + 1, c0 : c1 + 1]
            if min(mask.shape) < 2 or not _is_connected(mask):
                continue
            if mask.sum() == mask.size and random.random() < 0.75:
                continue
            return mask
    raise RuntimeError(f"failed to build shape of area {area}")


def _make_fragment(area: int, color: int) -> np.ndarray:
    mask = _random_shape(area)
    sprite = np.full(mask.shape, BG, dtype=int)
    sprite[mask] = int(color)
    return random.choice(_orientations(sprite))


def _bottom_columns(sprite: np.ndarray, *, bg: int) -> list[int]:
    rows, cols = np.where(sprite != bg)
    bottom = int(rows.max())
    return [int(c) for c in cols[rows == bottom]]


def _input_order(n: int, sorted_order: list[int]) -> list[int]:
    orders = [list(order) for order in itertools.permutations(range(n)) if list(order) != sorted_order]
    return random.choice(orders)


def _sample_gaps(n: int) -> list[int]:
    if random.random() < 0.6:
        gaps = [0, *random.sample(range(1, 6), n - 1)]
    else:
        gaps = random.sample(range(1, 6), n)
    random.shuffle(gaps)
    return gaps


def _expected_output(fragments: list[dict], gaps: list[int]) -> np.ndarray:
    gap_by_idx = {idx: int(gaps[idx]) for idx in range(len(fragments))}
    pseudo = []
    for idx, frag in enumerate(fragments):
        h, w = frag["sprite"].shape
        pseudo.append(
            {
                "bbox": (0, idx * 10, h - 1, idx * 10 + w - 1),
                "area": frag["area"],
                "sprite": frag["sprite"],
            }
        )
    return _pack_fragments(pseudo, gap_by_idx, bg=BG)


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    n = random.choice((3, 4))
    areas = random.sample(range(3, 9), n)
    colors = random.sample(FRAGMENT_COLORS, n)
    gaps = _sample_gaps(n)

    fragments = []
    for idx, (area, color) in enumerate(zip(areas, colors)):
        sprite = _make_fragment(area, color)
        h, w = sprite.shape
        bottom_cols = _bottom_columns(sprite, bg=BG)
        local_bottom = int(np.max(np.argwhere(sprite != BG)[:, 0]))
        fragments.append(
            {
                "idx": idx,
                "area": area,
                "color": color,
                "sprite": sprite,
                "h": h,
                "w": w,
                "local_bottom": local_bottom,
                "bottom_cols": bottom_cols,
            }
        )

    expected = _expected_output(fragments, gaps)
    if max(expected.shape) > 30:
        return None

    sorted_order = sorted(range(n), key=lambda idx: areas[idx])
    order = _input_order(n, sorted_order)

    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    inter_gaps = [random.randint(1, 3) for _ in range(n - 1)]
    width = left_margin + right_margin + sum(fragments[idx]["w"] for idx in order) + sum(inter_gaps)
    width += random.randint(0, 3)
    if width > 30:
        return None

    height = random.randint(14, 18)
    if height > 30:
        return None

    placements = [None] * n
    c = left_margin
    for pos, frag_idx in enumerate(order):
        frag = fragments[frag_idx]
        gap = gaps[frag_idx]
        lo = max(height // 2, frag["local_bottom"] + gap + 2)
        hi = min(height - 2, height // 2 + gap)
        if lo > hi:
            return None
        marker_r = random.randint(lo, hi)
        top = marker_r - gap - 1 - frag["local_bottom"]
        if top < 1:
            return None
        marker_c = c + random.choice(frag["bottom_cols"])
        placements[frag_idx] = {
            "top": top,
            "left": c,
            "marker_r": marker_r,
            "marker_c": marker_c,
        }
        c += frag["w"]
        if pos < n - 1:
            c += inter_gaps[pos]

    grid = np.full((height, width), BG, dtype=int)
    for frag_idx, frag in enumerate(fragments):
        place = placements[frag_idx]
        _paste(grid, frag["sprite"], place["top"], place["left"], transparent=BG)

    for place in placements:
        if int(grid[place["marker_r"], place["marker_c"]]) != BG:
            return None
        grid[place["marker_r"], place["marker_c"]] = MARKER

    for frag_idx, frag in enumerate(fragments):
        sprite = frag["sprite"]
        place = placements[frag_idx]
        coords = np.argwhere(sprite != BG)
        coords = coords + np.array([place["top"], place["left"]], dtype=int)
        frag["bbox"] = _bbox_coords(coords)
        frag["coords"] = coords

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None

    bg, found_frags, found_markers = _extract_scene(grid)
    gap_by_fragment = _match_markers_to_fragments(grid, found_frags, found_markers, bg=bg)
    if len(set(gap_by_fragment.values())) != n:
        return None
    if any(int(frag["bbox"][2]) >= height // 2 for frag in found_frags):
        return None
    if any(int(marker["coords"][0, 0]) < height // 2 for marker in found_markers):
        return None

    left_to_right_input = [idx for idx, _frag in sorted(enumerate(found_frags), key=lambda item: item[1]["bbox"][1])]
    sorted_by_area = [idx for idx, _frag in sorted(enumerate(found_frags), key=lambda item: item[1]["area"])]
    if left_to_right_input == sorted_by_area:
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _build_candidate()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid puzzle 7 input")
