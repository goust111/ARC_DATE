from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 400
SPRITE_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 80


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_patch(grid: np.ndarray, coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch


def _find_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != BG:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _parse_reference_pairs(grid: np.ndarray) -> list[tuple[int, int]]:
    row = np.asarray(grid[-1], dtype=int)
    non_bg = [idx for idx, value in enumerate(row) if int(value) != BG]
    assert non_bg and len(non_bg) % 2 == 0, "reference bar must contain an even number of colored cells"
    pairs: list[tuple[int, int]] = []
    i = 0
    while i < len(non_bg):
        c0 = non_bg[i]
        c1 = non_bg[i + 1]
        assert c1 == c0 + 1, "each reference pair must occupy two adjacent cells"
        if i + 2 < len(non_bg):
            assert non_bg[i + 2] >= c1 + 2, "reference pairs must be separated by background"
        pairs.append((int(row[c0]), int(row[c1])))
        i += 2
    return pairs


def _sprite_records(grid: np.ndarray) -> list[dict]:
    body = np.asarray(grid[:-1], dtype=int)
    components = _find_components(body)
    assert 4 <= len(components) <= 5, "expected 4 or 5 sprites"
    records: list[dict] = []
    for coords in components:
        patch = _component_patch(body, coords)
        counts = Counter(int(body[r, c]) for r, c in coords)
        colors = list(counts)
        assert len(colors) == 2, "each sprite must use exactly two non-background colors"
        major, minor = counts.most_common(2)
        assert major[1] != minor[1], "major/minor colors must be unambiguous"
        records.append(
            {
                "coords": coords,
                "patch": patch,
                "colors": frozenset(colors),
                "major": int(major[0]),
                "minor": int(minor[0]),
                "area": len(coords),
            }
        )
    return records


def _solve(grid: np.ndarray, rotate_mode: str = "rule") -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    refs = _parse_reference_pairs(g)
    records = _sprite_records(g)
    assert len(refs) == len(records), "reference-pair count must equal sprite count"
    ref_map: dict[frozenset[int], tuple[int, int]] = {}
    for pair in refs:
        key = frozenset(pair)
        assert len(key) == 2, "each reference pair must contain two distinct colors"
        assert key not in ref_map, "reference color pairs must be unique"
        ref_map[key] = pair

    prepared: list[tuple[int, np.ndarray]] = []
    for rec in records:
        assert rec["colors"] in ref_map, "every sprite color pair must appear in the reference bar"
        standard_major, standard_minor = ref_map[rec["colors"]]
        if rotate_mode == "always":
            rotate = True
        elif rotate_mode == "never":
            rotate = False
        else:
            if rec["major"] == standard_major and rec["minor"] == standard_minor:
                rotate = False
            else:
                assert rec["major"] == standard_minor and rec["minor"] == standard_major
                rotate = True
        patch = np.rot90(rec["patch"], 2) if rotate else rec["patch"]
        prepared.append((rec["area"], patch))

    areas = [area for area, _patch in prepared]
    assert len(areas) == len(set(areas)), "sprite areas must be unique"
    prepared.sort(key=lambda item: item[0])

    out_h = max(patch.shape[0] for _area, patch in prepared)
    out_w = sum(patch.shape[1] for _area, patch in prepared)
    out = np.full((out_h, out_w), BG, dtype=int)
    c0 = 0
    for _area, patch in prepared:
        h, w = patch.shape
        out[out_h - h : out_h, c0 : c0 + w] = patch
        c0 += w
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the color-swapped sprites, then sort and bottom-pack them by area.

    Input structure:
    - The background is black, and the bottom row is a reference bar made of 2-cell color pairs.
    - Above that row are 4-5 disconnected 4-connected sprites, each using exactly two non-black colors.
    - Each sprite's unordered color pair appears exactly once in the reference bar, and its two color counts differ.

    Rule:
    - Read each reference-bar pair as (standard major color, standard minor color).
    - For each sprite, count its two colors to determine the actual major/minor colors.
    - If the actual ordering matches the reference ordering, keep the sprite unchanged; if it is reversed, rotate that sprite 180 degrees.
    - Sort all resulting sprites by pixel area from smallest to largest, then concatenate their bounding-box patches left-to-right with bottom alignment.

    Key insight:
    - The reference bar does not tell you where to place colors directly; it encodes a hidden boolean that says whether each sprite must be turned around before the final area-based packing.
    """
    return _solve(grid, rotate_mode="rule")


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier = {start}
    while int(mask.sum()) < area:
        growth: list[tuple[int, int]] = []
        for r, c in frontier:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    growth.append((nr, nc))
        if not growth:
            return _random_connected_mask(h, w, area)
        cell = random.choice(growth)
        mask[cell] = True
        frontier.add(cell)
    rows, cols = np.where(mask)
    cropped = mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    return cropped


def _build_sprite(area: int, standard_pair: tuple[int, int], swapped: bool) -> np.ndarray | None:
    color_major, color_minor = standard_pair
    major_color, minor_color = (color_minor, color_major) if swapped else (color_major, color_minor)
    for _ in range(SPRITE_ATTEMPTS):
        h = random.randint(3, 4)
        w = random.randint(3, 4)
        if area > h * w or area < 5:
            continue
        mask = _random_connected_mask(h, w, area)
        h2, w2 = mask.shape
        if h2 > 4 or w2 > 4:
            continue
        coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        major_count = random.randint(area // 2 + 1, area - 2)
        major_cells = set(random.sample(coords, major_count))
        patch = np.full(mask.shape, BG, dtype=int)
        for r, c in coords:
            patch[r, c] = major_color if (r, c) in major_cells else minor_color
        if np.array_equal(patch, np.rot90(patch, 2)):
            continue
        return patch
    return None


def _generate_reference_pairs(count: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    used: set[frozenset[int]] = set()
    while len(pairs) < count:
        a, b = random.sample(PALETTE, 2)
        key = frozenset((a, b))
        if key in used:
            continue
        used.add(key)
        pairs.append((a, b))
    random.shuffle(pairs)
    return pairs


def _can_place(occ: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    hh, ww = occ.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            if int(patch[r - r0, c - c0]) == BG:
                continue
            for rr in range(max(0, r - 1), min(hh, r + 2)):
                for cc in range(max(0, c - 1), min(ww, c + 2)):
                    if occ[rr, cc]:
                        return False
    return True


def _place_patch(grid: np.ndarray, occ: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    for r in range(h):
        for c in range(w):
            if int(patch[r, c]) != BG:
                grid[r0 + r, c0 + c] = int(patch[r, c])
                occ[r0 + r, c0 + c] = True


def _bar_width(pair_count: int) -> int:
    return 3 * pair_count - 1


def _pack_reference_row(grid: np.ndarray, pairs: list[tuple[int, int]]) -> None:
    width = _bar_width(len(pairs))
    max_start = grid.shape[1] - width
    start = random.randint(0, max_start)
    c = start
    for idx, (a, b) in enumerate(pairs):
        grid[-1, c] = a
        grid[-1, c + 1] = b
        c += 2
        if idx + 1 < len(pairs):
            c += 1


def _semantic_checks(grid: np.ndarray) -> None:
    refs = _parse_reference_pairs(grid)
    records = _sprite_records(grid)
    assert len(refs) == len(records)
    states = []
    for rec in records:
        pair = next(pair for pair in refs if frozenset(pair) == rec["colors"])
        states.append(rec["major"] != pair[0])
        assert not np.array_equal(rec["patch"], np.rot90(rec["patch"], 2))
    assert any(states) and not all(states), "generator must cover both rotate and keep branches"
    out = _solve(grid, "rule")
    assert out.shape[0] <= 30 and out.shape[1] <= 30
    assert not np.array_equal(out, _solve(grid, "always"))
    assert not np.array_equal(out, _solve(grid, "never"))


def _attempt_input() -> np.ndarray | None:
    sprite_count = random.randint(4, 5)
    areas = random.sample(range(5, 11), sprite_count)
    pairs = _generate_reference_pairs(sprite_count)
    branch_mask = [True] * random.randint(1, sprite_count - 1) + [False] * random.randint(1, sprite_count - 1)
    while len(branch_mask) < sprite_count:
        branch_mask.append(random.choice([False, True]))
    branch_mask = branch_mask[:sprite_count]
    random.shuffle(branch_mask)
    if all(branch_mask) or not any(branch_mask):
        branch_mask[0] = True
        branch_mask[-1] = False

    patches: list[np.ndarray] = []
    for area, pair, swapped in zip(areas, pairs, branch_mask):
        patch = _build_sprite(area, pair, swapped)
        if patch is None:
            return None
        patches.append(patch)

    min_width = max(18, _bar_width(sprite_count) + 2)
    width = random.randint(min_width, 26)
    height = random.randint(11, 15)
    body_h = height - 1

    max_sprite_h = max(p.shape[0] for p in patches)
    total_bbox_area = sum(p.shape[0] * p.shape[1] for p in patches)
    if max_sprite_h > body_h - 1 or total_bbox_area > width * (body_h - 1):
        return None

    grid = np.full((height, width), BG, dtype=int)
    occ = np.zeros((body_h, width), dtype=bool)
    order = list(range(sprite_count))
    random.shuffle(order)
    for idx in order:
        patch = patches[idx]
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, body_h - patch.shape[0] - 1)
            c0 = random.randint(0, width - patch.shape[1])
            if _can_place(occ, patch, r0, c0):
                _place_patch(grid[:-1], occ, patch, r0, c0)
                placed = True
                break
        if not placed:
            return None

    _pack_reference_row(grid, pairs)
    _semantic_checks(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_ATTEMPTS} attempts")
