from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg, in_bounds  # noqa: E402


MAX_GENERATION_ATTEMPTS = 600
BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _cell_colors(grid: np.ndarray, coords: np.ndarray) -> Counter[int]:
    return Counter(int(grid[int(r), int(c)]) for r, c in coords)


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _sort_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted((int(r), int(c)) for r, c in cells)


def _infer_sprites(grid: np.ndarray) -> tuple[dict[int, dict], dict[int, list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=8, monochromatic=False)

    source_components = []
    marker_components = []
    for comp in components:
        colors = _cell_colors(g, comp["coords"])
        if len(colors) >= 2:
            source_components.append(comp)
        else:
            marker_components.append(comp)

    assert 3 <= len(source_components) <= 4, "expected 3-4 multicolor source sprites"
    source_presence: Counter[int] = Counter()
    per_component_counts: list[Counter[int]] = []
    for comp in source_components:
        counts = _cell_colors(g, comp["coords"])
        per_component_counts.append(counts)
        for color in counts:
            source_presence[int(color)] += 1

    sprites: dict[int, dict] = {}
    for comp, counts in zip(source_components, per_component_counts):
        signature_candidates = [
            int(color)
            for color, count in counts.items()
            if int(count) == 1 and int(source_presence[int(color)]) == 1
        ]
        assert len(signature_candidates) == 1, "each source sprite must expose one signature color"
        signature_color = int(signature_candidates[0])
        signature_coords = comp["coords"][
            g[comp["coords"][:, 0], comp["coords"][:, 1]] == signature_color
        ]
        assert len(signature_coords) == 1
        sr, sc = map(int, signature_coords[0])

        rel_cells: list[tuple[int, int, int]] = []
        abs_cells: list[tuple[int, int]] = []
        for r, c in comp["coords"]:
            rr, cc = int(r), int(c)
            abs_cells.append((rr, cc))
            color = int(g[rr, cc])
            if color == signature_color:
                continue
            rel_cells.append((rr - sr, cc - sc, color))

        sprites[signature_color] = {
            "signature_color": signature_color,
            "signature_abs": (sr, sc),
            "rel_cells": tuple(rel_cells),
            "abs_cells": tuple(abs_cells),
            "area": int(comp["area"]),
        }

    markers_by_signature: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for comp in marker_components:
        assert int(comp["area"]) == 1, "targets must be single-pixel markers"
        r, c = map(int, comp["coords"][0])
        color = int(g[r, c])
        if color in sprites:
            markers_by_signature[color].append((r, c))

    for signature_color in sprites:
        markers = _sort_cells(markers_by_signature[signature_color])
        assert markers, "each source sprite must have at least one target marker"
        markers_by_signature[signature_color] = markers

    return sprites, markers_by_signature


def _apply_rule(
    grid: np.ndarray,
    sprites: dict[int, dict],
    markers_by_signature: dict[int, list[tuple[int, int]]],
    *,
    ascending: bool,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    ordered = sorted(sprites.values(), key=lambda sprite: int(sprite["area"]), reverse=not ascending)
    for sprite in ordered:
        signature_color = int(sprite["signature_color"])
        for ar, ac in markers_by_signature[signature_color]:
            for dr, dc, color in sprite["rel_cells"]:
                rr = int(ar + dr)
                cc = int(ac + dc)
                if in_bounds(out.shape, rr, cc):
                    out[rr, cc] = int(color)
            out[ar, ac] = signature_color

    for sprite in sprites.values():
        for r, c in sprite["abs_cells"]:
            out[int(r), int(c)] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each signature-matched multicolor sprite onto its markers, resolve overlaps by sprite area, then erase the originals.

    Input structure:
    - A black background contains 3-4 8-connected multicolor source sprites in the upper half and scattered single-pixel target markers in the lower half.
    - Each source sprite has 4-10 cells, uses 2-3 non-black colors, and contains exactly one signature color that appears once in that sprite and in no other source sprite.
    - Every target marker is a single pixel whose color matches one source sprite's signature color.

    Rule:
    - Identify each source sprite and its signature pixel, then match every target marker to the source sprite with the same signature color.
    - For each source sprite, count its total non-black area S and sort the sprites from smaller S to larger S.
    - In that order, stamp every matched sprite onto all of its target markers by aligning the source signature pixel to the marker; copy every non-signature sprite cell, keep the marker color at the anchor, and discard stamped cells that fall outside the grid.
    - Later stamps from larger sprites overwrite earlier stamps in overlapping cells, then all original upper-half source sprites are erased to black.

    Key insight:
    - The signature color determines which sprite to stamp, while the sprite's own pixel count determines which overlapping stamp stays visible on top.
    """
    sprites, markers_by_signature = _infer_sprites(np.asarray(grid, dtype=int))
    return _apply_rule(np.asarray(grid, dtype=int), sprites, markers_by_signature, ascending=True)


def _normalize_cells(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _random_connected_mask(area: int) -> tuple[tuple[int, int], ...]:
    for _ in range(300):
        max_dim = 4 if area <= 8 else 5
        h = random.randint(3, max_dim)
        w = random.randint(3, max_dim)
        if area > h * w:
            continue
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                            frontier.add((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue
        norm = _normalize_cells(cells)
        rows = {r for r, _ in norm}
        cols = {c for _, c in norm}
        if len(rows) < 2 or len(cols) < 2:
            continue
        return norm
    raise RuntimeError("failed to build connected sprite mask")


def _build_sprite(
    area: int,
    signature_color: int,
    body_primary: int,
    body_secondary: int,
    use_secondary: bool,
) -> dict:
    mask = _random_connected_mask(area)
    signature_cell = random.choice(mask)
    non_signature_cells = [cell for cell in mask if cell != signature_cell]
    assert len(non_signature_cells) >= 3

    body_colors: dict[tuple[int, int], int] = {}
    if use_secondary:
        secondary_count = random.randint(1, len(non_signature_cells) - 1)
        secondary_cells = set(random.sample(non_signature_cells, secondary_count))
    else:
        secondary_cells = set()

    for cell in non_signature_cells:
        body_colors[cell] = body_secondary if cell in secondary_cells else body_primary

    sr, sc = signature_cell
    local_cells: list[tuple[int, int, int]] = []
    rel_cells: list[tuple[int, int, int]] = []
    for r, c in mask:
        color = signature_color if (r, c) == signature_cell else int(body_colors[(r, c)])
        local_cells.append((int(r), int(c), int(color)))
        if (r, c) != signature_cell:
            rel_cells.append((int(r - sr), int(c - sc), int(color)))

    bbox_h = 1 + max(r for r, _, _ in local_cells)
    bbox_w = 1 + max(c for _, c, _ in local_cells)
    return {
        "signature_color": int(signature_color),
        "local_cells": tuple(local_cells),
        "rel_cells": tuple(rel_cells),
        "signature_local": (int(sr), int(sc)),
        "area": int(area),
        "bbox": (int(bbox_h), int(bbox_w)),
        "uses_secondary": bool(use_secondary),
    }


def _sprite_full_rel(sprite: dict) -> list[tuple[int, int, int]]:
    return [(0, 0, int(sprite["signature_color"]))] + [
        (int(dr), int(dc), int(color)) for dr, dc, color in sprite["rel_cells"]
    ]


def _placed_cells(sprite: dict, anchor: tuple[int, int], shape: tuple[int, int]) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    ar, ac = map(int, anchor)
    for dr, dc, color in _sprite_full_rel(sprite):
        rr = ar + int(dr)
        cc = ac + int(dc)
        if in_bounds(shape, rr, cc):
            out[(rr, cc)] = int(color)
    return out


def _placed_non_signature(sprite: dict, anchor: tuple[int, int], shape: tuple[int, int]) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    ar, ac = map(int, anchor)
    for dr, dc, color in sprite["rel_cells"]:
        rr = ar + int(dr)
        cc = ac + int(dc)
        if in_bounds(shape, rr, cc):
            out[(rr, cc)] = int(color)
    return out


def _all_non_signature_in_bounds(sprite: dict, anchor: tuple[int, int], shape: tuple[int, int]) -> bool:
    ar, ac = map(int, anchor)
    return all(in_bounds(shape, ar + int(dr), ac + int(dc)) for dr, dc, _ in sprite["rel_cells"])


def _has_clipped_non_signature(sprite: dict, anchor: tuple[int, int], shape: tuple[int, int]) -> bool:
    ar, ac = map(int, anchor)
    return any(not in_bounds(shape, ar + int(dr), ac + int(dc)) for dr, dc, _ in sprite["rel_cells"])


def _place_sources(sprites: list[dict], shape: tuple[int, int]) -> list[dict] | None:
    h, w = map(int, shape)
    upper_limit = h // 2 - 2
    occupied: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()
    placed: list[dict | None] = [None] * len(sprites)

    order = sorted(range(len(sprites)), key=lambda idx: sprites[idx]["area"], reverse=True)
    for idx in order:
        sprite = sprites[idx]
        bbox_h, bbox_w = map(int, sprite["bbox"])
        max_r0 = upper_limit - bbox_h
        max_c0 = w - bbox_w - 1
        if max_r0 < 1 or max_c0 < 1:
            return None

        success = False
        for _ in range(300):
            r0 = random.randint(1, max_r0)
            c0 = random.randint(1, max_c0)
            abs_cells = [(r0 + r, c0 + c, color) for r, c, color in sprite["local_cells"]]
            if any((r, c) in forbidden for r, c, _ in abs_cells):
                continue

            placed[idx] = {
                **sprite,
                "source_top_left": (int(r0), int(c0)),
                "source_abs_cells": tuple((int(r), int(c), int(color)) for r, c, color in abs_cells),
                "signature_abs": tuple(
                    int(v)
                    for v in (r0 + sprite["signature_local"][0], c0 + sprite["signature_local"][1])
                ),
            }
            for r, c, _ in abs_cells:
                occupied.add((int(r), int(c)))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr = int(r + dr)
                        cc = int(c + dc)
                        if in_bounds(shape, rr, cc):
                            forbidden.add((rr, cc))
            success = True
            break
        if not success:
            return None

    return [sprite for sprite in placed if sprite is not None]


def _marker_anchor_ok(
    anchor: tuple[int, int],
    lower_start: int,
    shape: tuple[int, int],
    existing_markers: list[tuple[int, int]],
) -> bool:
    r, c = map(int, anchor)
    h, w = map(int, shape)
    if not (lower_start <= r < h and 0 <= c < w):
        return False
    return all(_chebyshev(anchor, other) >= 2 for other in existing_markers)


def _find_forced_overlap(
    sprites: list[dict],
    shape: tuple[int, int],
    lower_start: int,
) -> tuple[int, tuple[int, int], int, tuple[int, int]] | None:
    candidates: list[tuple[int, int]] = []
    with_secondary = [idx for idx, sprite in enumerate(sprites) if sprite["uses_secondary"]]
    without_secondary = [idx for idx, sprite in enumerate(sprites) if not sprite["uses_secondary"]]
    for a in without_secondary:
        for b in with_secondary:
            if a != b:
                candidates.append((a, b))
                candidates.append((b, a))
    random.shuffle(candidates)

    for idx_a, idx_b in candidates:
        sprite_a = sprites[idx_a]
        sprite_b = sprites[idx_b]
        cells_a = [(dr, dc, color) for dr, dc, color in sprite_a["rel_cells"]]
        cells_b = [(dr, dc, color) for dr, dc, color in sprite_b["rel_cells"]]
        deltas = list(
            {
                (int(dra - drb), int(dca - dcb))
                for dra, dca, color_a in cells_a
                for drb, dcb, color_b in cells_b
                if int(color_a) != int(color_b)
            }
        )
        random.shuffle(deltas)
        for delta_r, delta_c in deltas:
            if max(abs(delta_r), abs(delta_c)) <= 1:
                continue
            for _ in range(120):
                anchor_a = (random.randint(lower_start, shape[0] - 1), random.randint(0, shape[1] - 1))
                anchor_b = (anchor_a[0] + delta_r, anchor_a[1] + delta_c)
                if not _marker_anchor_ok(anchor_a, lower_start, shape, []):
                    continue
                if not _marker_anchor_ok(anchor_b, lower_start, shape, [anchor_a]):
                    continue
                if not _all_non_signature_in_bounds(sprite_a, anchor_a, shape):
                    continue
                if not _all_non_signature_in_bounds(sprite_b, anchor_b, shape):
                    continue

                non_sig_a = _placed_non_signature(sprite_a, anchor_a, shape)
                non_sig_b = _placed_non_signature(sprite_b, anchor_b, shape)
                overlap = set(non_sig_a) & set(non_sig_b)
                if not (1 <= len(overlap) <= 3):
                    continue
                if not any(int(non_sig_a[cell]) != int(non_sig_b[cell]) for cell in overlap):
                    continue

                test_grid = np.zeros(shape, dtype=int)
                order_ab = test_grid.copy()
                for cell, color in non_sig_a.items():
                    order_ab[cell] = int(color)
                order_ab[anchor_a] = int(sprite_a["signature_color"])
                for cell, color in non_sig_b.items():
                    order_ab[cell] = int(color)
                order_ab[anchor_b] = int(sprite_b["signature_color"])

                order_ba = test_grid.copy()
                for cell, color in non_sig_b.items():
                    order_ba[cell] = int(color)
                order_ba[anchor_b] = int(sprite_b["signature_color"])
                for cell, color in non_sig_a.items():
                    order_ba[cell] = int(color)
                order_ba[anchor_a] = int(sprite_a["signature_color"])
                if np.array_equal(order_ab, order_ba):
                    continue
                return idx_a, anchor_a, idx_b, anchor_b
    return None


def _add_extra_markers(
    sprites: list[dict],
    shape: tuple[int, int],
    lower_start: int,
    placements: list[dict],
    desired_counts: list[int],
) -> bool:
    existing_markers = [tuple(map(int, placement["anchor"])) for placement in placements]
    occupied_sets = [set(_placed_cells(sprites[p["sprite_index"]], p["anchor"], shape)) for p in placements]

    clip_needed = random.random() < 0.4 and sum(desired_counts) > len(sprites)
    clip_used = False

    for sprite_index, target_count in enumerate(desired_counts):
        while sum(1 for placement in placements if placement["sprite_index"] == sprite_index) < target_count:
            sprite = sprites[sprite_index]
            require_clip = clip_needed and not clip_used
            success = False
            for _ in range(400):
                anchor = (random.randint(lower_start, shape[0] - 1), random.randint(0, shape[1] - 1))
                if not _marker_anchor_ok(anchor, lower_start, shape, existing_markers):
                    continue
                if require_clip:
                    if not _has_clipped_non_signature(sprite, anchor, shape):
                        continue
                else:
                    if not _all_non_signature_in_bounds(sprite, anchor, shape):
                        continue

                full_cells = set(_placed_cells(sprite, anchor, shape))
                if any(full_cells & other for other in occupied_sets):
                    continue
                same_sprite = [
                    placement for placement in placements if placement["sprite_index"] == sprite_index
                ]
                if any(full_cells & set(_placed_cells(sprite, placement["anchor"], shape)) for placement in same_sprite):
                    continue

                placements.append({"sprite_index": int(sprite_index), "anchor": tuple(map(int, anchor))})
                existing_markers.append(tuple(map(int, anchor)))
                occupied_sets.append(full_cells)
                if require_clip:
                    clip_used = True
                success = True
                break
            if not success:
                return False
    if clip_needed and not clip_used:
        return False
    return True


def _build_input_grid(shape: tuple[int, int], sprites: list[dict], placements: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for sprite in sprites:
        for r, c, color in sprite["source_abs_cells"]:
            grid[int(r), int(c)] = int(color)
    for placement in placements:
        sprite = sprites[placement["sprite_index"]]
        r, c = placement["anchor"]
        grid[int(r), int(c)] = int(sprite["signature_color"])
    return grid


def _validate_generated(
    grid: np.ndarray,
    sprites: list[dict],
    placements: list[dict],
) -> bool:
    if max(grid.shape) > 30:
        return False
    if not np.any(grid):
        return False

    solver_out = main(grid)
    inferred_sprites, markers_by_signature = _infer_sprites(grid)
    expected = _apply_rule(grid, inferred_sprites, markers_by_signature, ascending=True)
    reversed_order = _apply_rule(grid, inferred_sprites, markers_by_signature, ascending=False)
    if not np.array_equal(solver_out, expected):
        return False
    if np.array_equal(solver_out, grid):
        return False
    if np.array_equal(solver_out, reversed_order):
        return False

    source_cells = {(int(r), int(c)) for sprite in sprites for r, c, _ in sprite["source_abs_cells"]}
    marker_cells = [tuple(map(int, placement["anchor"])) for placement in placements]
    if any(cell in source_cells for cell in marker_cells):
        return False
    if any(_chebyshev(a, b) <= 1 for i, a in enumerate(marker_cells) for b in marker_cells[i + 1 :]):
        return False

    areas = [int(sprite["area"]) for sprite in sprites]
    if len(set(areas)) != len(areas):
        return False
    if not any(sprite["uses_secondary"] for sprite in sprites):
        return False
    if not any(not sprite["uses_secondary"] for sprite in sprites):
        return False

    overlap_sensitive = False
    for i, placement_a in enumerate(placements):
        sprite_a = sprites[placement_a["sprite_index"]]
        cells_a = _placed_non_signature(sprite_a, placement_a["anchor"], grid.shape)
        for placement_b in placements[i + 1 :]:
            sprite_b = sprites[placement_b["sprite_index"]]
            if placement_a["sprite_index"] == placement_b["sprite_index"]:
                continue
            cells_b = _placed_non_signature(sprite_b, placement_b["anchor"], grid.shape)
            overlap = set(cells_a) & set(cells_b)
            if overlap and any(int(cells_a[cell]) != int(cells_b[cell]) for cell in overlap):
                overlap_sensitive = True
                break
        if overlap_sensitive:
            break
    if not overlap_sensitive:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(18, 24), random.randint(18, 24))
    sprite_count = random.choice([3, 4])
    colors = random.sample(PALETTE, sprite_count + 2)
    signature_colors = colors[:sprite_count]
    body_primary, body_secondary = colors[-2:]

    areas = random.sample(list(range(4, 11)), sprite_count)
    secondary_users = set(random.sample(range(sprite_count), random.randint(2, sprite_count - 1)))
    sprites = [
        _build_sprite(
            area=int(areas[idx]),
            signature_color=int(signature_colors[idx]),
            body_primary=int(body_primary),
            body_secondary=int(body_secondary),
            use_secondary=idx in secondary_users,
        )
        for idx in range(sprite_count)
    ]
    placed_sources = _place_sources(sprites, shape)
    if placed_sources is None:
        return None
    sprites = placed_sources

    lower_start = shape[0] // 2 + 1
    forced = _find_forced_overlap(sprites, shape, lower_start)
    if forced is None:
        return None
    idx_a, anchor_a, idx_b, anchor_b = forced
    placements = [
        {"sprite_index": int(idx_a), "anchor": tuple(map(int, anchor_a))},
        {"sprite_index": int(idx_b), "anchor": tuple(map(int, anchor_b))},
    ]

    desired_counts = [1] * sprite_count
    extras = random.choice([1, 1, 2])
    for sprite_index in random.sample(range(sprite_count), extras):
        desired_counts[sprite_index] += 1
    desired_counts[idx_a] = max(desired_counts[idx_a], 1)
    desired_counts[idx_b] = max(desired_counts[idx_b], 1)

    if not _add_extra_markers(sprites, shape, lower_start, placements, desired_counts):
        return None

    grid = _build_input_grid(shape, sprites, placements)
    if not _validate_generated(grid, sprites, placements):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
