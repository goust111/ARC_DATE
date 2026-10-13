from __future__ import annotations

import random
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg, paste, random_free_location


BG = 0
SEP = 5
LABEL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

SPRITE_TEMPLATES = (
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]], dtype=int),
)

ANCHOR_TEMPLATES = (
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=int),
)

GEN_ATTEMPTS = 500


def _crop_binary(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask != 0)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_orientation(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    out = np.rot90(out, k=random.randrange(4))
    return _crop_binary(out)


def _normalize_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    rel = sorted((int(r - r0), int(c - c0)) for r, c in coords)
    return tuple(rel)


def _component_patch(component: dict, color: int) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    coords = component["coords"]
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(color)
    return patch


def _point_component_sqdist(point: tuple[int, int], coords: np.ndarray) -> int:
    pr, pc = point
    deltas = coords - np.array([[pr, pc]], dtype=int)
    return int(np.min(np.sum(deltas * deltas, axis=1)))


def _point_component_cheb(point: tuple[int, int], coords: np.ndarray) -> int:
    pr, pc = point
    deltas = np.abs(coords - np.array([[pr, pc]], dtype=int))
    return int(np.min(np.max(deltas, axis=1)))


def _separator_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    rows = [
        r
        for r in range(grid.shape[0])
        if np.all(grid[r, :] == grid[r, 0]) and int(grid[r, 0]) != int(bg)
    ]
    assert len(rows) == 1, "expected exactly one full-width separator row"
    r = int(rows[0])
    return r, int(grid[r, 0])


def _sorted_component_indices(components: list[dict]) -> list[int]:
    return sorted(
        range(len(components)),
        key=lambda idx: (int(components[idx]["bbox"][0]), int(components[idx]["bbox"][1])),
    )


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    sep_r, sep_color = _separator_row(g, bg)

    upper = g[:sep_r, :]
    lower = g[sep_r + 1 :, :]
    assert upper.size > 0 and lower.size > 0, "separator must split the grid into two non-empty panels"

    anchor_components = find_components(
        upper,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert len(anchor_components) == 3, "expected three upper-panel anchors"
    anchors_by_color: dict[int, dict] = {}
    for component in anchor_components:
        color = int(component["color"])
        assert color not in {bg, sep_color}, "anchor colors must differ from background and separator"
        assert 3 <= int(component["area"]) <= 5, "anchor size must be 3..5 pixels"
        assert color not in anchors_by_color, "anchor colors must be unique"
        anchors_by_color[color] = component

    sprite_components = find_components(
        lower,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[sep_color],
        return_format="dict",
    )
    assert len(sprite_components) == 3, "expected three lower-panel sprites"
    sprite_signature = _normalize_coords(sprite_components[0]["coords"])
    for component in sprite_components:
        assert 4 <= int(component["area"]) <= 6, "sprite size must be 4..6 pixels"
        assert _normalize_coords(component["coords"]) == sprite_signature, "sprites must be identical"

    marker_points: dict[int, tuple[int, int]] = {}
    for color in sorted(set(map(int, np.unique(lower))) - {bg, sep_color}):
        coords = np.argwhere(lower == color)
        assert coords.shape == (1, 2), "every lower-panel label must be a singleton marker"
        marker_points[int(color)] = (int(coords[0, 0]), int(coords[0, 1]))

    assert set(marker_points) == set(anchors_by_color), "marker colors must match anchor colors exactly"

    sprite_to_color: dict[int, int] = {}
    color_to_sprite: dict[int, int] = {}
    for color, point in marker_points.items():
        dists = [_point_component_sqdist(point, comp["coords"]) for comp in sprite_components]
        best = min(dists)
        winners = [idx for idx, dist in enumerate(dists) if dist == best]
        assert len(winners) == 1, "marker-to-sprite nearest match must be unique"
        sprite_idx = int(winners[0])
        assert sprite_idx not in sprite_to_color, "two markers cannot claim the same sprite"
        assert _point_component_cheb(point, sprite_components[sprite_idx]["coords"]) <= 2, "marker must stay within 8-neighbor distance 2 of its sprite"
        sprite_to_color[sprite_idx] = color
        color_to_sprite[color] = sprite_idx

    assert len(sprite_to_color) == len(sprite_components) == 3
    return {
        "bg": bg,
        "sep_r": sep_r,
        "sep_color": sep_color,
        "upper": upper,
        "lower": lower,
        "anchors_by_color": anchors_by_color,
        "sprite_components": sprite_components,
        "sprite_to_color": sprite_to_color,
        "color_to_sprite": color_to_sprite,
        "marker_points": marker_points,
        "sprite_signature": sprite_signature,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Match each lower sprite to its nearest color marker, then move and recolor it at the upper anchor of the same color.

    Input structure:
    - A black background is split by one full-width grey separator row into upper and lower panels.
    - The upper panel contains exactly three small monochrome anchor components of distinct colors.
    - The lower panel contains exactly three identical grey 4-connected sprites plus three singleton color markers.

    Rule:
    - In the lower panel, pair each singleton marker with the unique nearest grey sprite using Euclidean distance from the marker to the sprite's nearest pixel.
    - Clear the entire lower panel back to black while preserving the separator and the upper anchors.
    - For each sprite, find the upper anchor with the matched color and place the sprite so its bbox top-left lands one cell below and one cell right of that anchor bbox's bottom-right corner.
    - Recolor the moved sprite with the matched marker color and keep all anchor pixels unchanged.

    Key insight:
    - The lower-panel nearest-neighbor labels create the color lookup table; color, not fixed position, decides which upper anchor receives each moved sprite.
    """
    parsed = _parse_input(grid)
    g = np.asarray(grid, dtype=int)
    bg = int(parsed["bg"])
    sep_r = int(parsed["sep_r"])
    sep_color = int(parsed["sep_color"])
    upper = np.asarray(parsed["upper"], dtype=int)
    anchors_by_color = parsed["anchors_by_color"]
    sprite_components = parsed["sprite_components"]
    sprite_to_color = parsed["sprite_to_color"]

    out = np.full_like(g, bg)
    out[sep_r, :] = sep_color
    out[:sep_r, :] = upper

    for sprite_idx, component in enumerate(sprite_components):
        color = int(sprite_to_color[sprite_idx])
        anchor = anchors_by_color[color]
        _ar0, _ac0, ar1, ac1 = anchor["bbox"]
        patch = _component_patch(component, color)
        ph, pw = patch.shape
        dst_r = int(ar1) + 1
        dst_c = int(ac1) + 1
        assert dst_r + ph <= sep_r, "moved sprite must stay inside the upper panel"
        assert dst_c + pw <= g.shape[1], "moved sprite must stay inside the grid"
        mask = patch != bg
        view = out[dst_r : dst_r + ph, dst_c : dst_c + pw]
        assert not np.any(view[mask] != bg), "moved sprites must not overwrite anchors or each other"
        paste(out, patch, dst_r, dst_c, transparent=bg)

    return out


def _sprite_patch() -> np.ndarray:
    mask = _random_orientation(random.choice(SPRITE_TEMPLATES))
    return np.where(mask != 0, SEP, BG).astype(int)


def _anchor_patch(color: int) -> np.ndarray:
    mask = _random_orientation(random.choice(ANCHOR_TEMPLATES))
    return np.where(mask != 0, int(color), BG).astype(int)


def _group_patch(anchor_patch: np.ndarray, sprite_patch: np.ndarray, color: int) -> np.ndarray:
    ah, aw = anchor_patch.shape
    sh, sw = sprite_patch.shape
    patch = np.full((ah + sh, aw + sw), BG, dtype=int)
    paste(patch, anchor_patch, 0, 0, transparent=BG)
    color_sprite = np.where(sprite_patch == SEP, int(color), BG).astype(int)
    paste(patch, color_sprite, ah, aw, transparent=BG)
    return patch


def _sprite_coords_from_loc(loc: tuple[int, int], sprite_patch: np.ndarray) -> np.ndarray:
    r0, c0 = loc
    coords = np.argwhere(sprite_patch != BG)
    coords = coords.copy()
    coords[:, 0] += int(r0)
    coords[:, 1] += int(c0)
    return coords


def _marker_candidates(
    sprite_idx: int,
    sprite_coords_list: list[np.ndarray],
    lower_scene: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = lower_scene.shape
    target = sprite_coords_list[sprite_idx]
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(lower_scene[r, c]) != BG:
                continue
            point = (r, c)
            if _point_component_cheb(point, target) > 2:
                continue
            dists = [_point_component_sqdist(point, coords) for coords in sprite_coords_list]
            best = min(dists)
            winners = [idx for idx, dist in enumerate(dists) if dist == best]
            if winners != [sprite_idx]:
                continue
            candidates.append(point)
    random.shuffle(candidates)
    return candidates


def _choose_markers(
    sprite_coords_list: list[np.ndarray],
    lower_scene: np.ndarray,
    lower_order: list[int],
) -> dict[int, tuple[int, int]] | None:
    candidate_map = {idx: _marker_candidates(idx, sprite_coords_list, lower_scene) for idx in range(3)}
    if any(not points for points in candidate_map.values()):
        return None

    order_rank = {sprite_idx: rank for rank, sprite_idx in enumerate(lower_order)}
    search_order = sorted(range(3), key=lambda idx: len(candidate_map[idx]))
    chosen: dict[int, tuple[int, int]] = {}
    used: set[tuple[int, int]] = set()

    def rec(pos: int) -> dict[int, tuple[int, int]] | None:
        if pos == len(search_order):
            marker_order = [sprite_idx for sprite_idx, _point in sorted(chosen.items(), key=lambda item: item[1])]
            marker_perm = [order_rank[sprite_idx] for sprite_idx in marker_order]
            if marker_perm == [0, 1, 2]:
                return None
            return dict(chosen)

        sprite_idx = search_order[pos]
        for point in candidate_map[sprite_idx]:
            if point in used:
                continue
            used.add(point)
            chosen[sprite_idx] = point
            result = rec(pos + 1)
            if result is not None:
                return result
            del chosen[sprite_idx]
            used.remove(point)
        return None

    return rec(0)


def _build_expected_output(
    shape: tuple[int, int],
    sep_r: int,
    upper_output: np.ndarray,
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[sep_r, :] = SEP
    out[:sep_r, :] = upper_output
    return out


def _semantic_check(inp: np.ndarray, expected_out: np.ndarray) -> bool:
    try:
        parsed = _parse_input(inp)
        out = main(inp)
    except Exception:
        return False

    if not np.array_equal(out, expected_out):
        return False

    sep_r = int(parsed["sep_r"])
    if np.any(out[sep_r + 1 :, :] != BG):
        return False

    anchors_by_color = parsed["anchors_by_color"]
    sprite_components = parsed["sprite_components"]
    sprite_to_color = parsed["sprite_to_color"]
    sprite_signature = parsed["sprite_signature"]
    lower_order = _sorted_component_indices(sprite_components)
    lower_colors = [int(sprite_to_color[idx]) for idx in lower_order]
    anchor_order = [
        color
        for color, _comp in sorted(
            anchors_by_color.items(),
            key=lambda item: (int(item[1]["bbox"][0]), int(item[1]["bbox"][1])),
        )
    ]
    if lower_colors == anchor_order:
        return False

    marker_order = [
        color
        for color, _point in sorted(parsed["marker_points"].items(), key=lambda item: item[1])
    ]
    if marker_order == lower_colors:
        return False

    upper_out = out[:sep_r, :]
    for color, anchor in anchors_by_color.items():
        comps = find_components(
            upper_out,
            bg=BG,
            connectivity=4,
            monochromatic=True,
            colors=[color],
            return_format="dict",
        )
        if len(comps) != 2:
            return False
        anchor_bbox = tuple(map(int, anchor["bbox"]))
        matching = [comp for comp in comps if tuple(map(int, comp["bbox"])) == anchor_bbox]
        if len(matching) != 1:
            return False
        moved = [comp for comp in comps if tuple(map(int, comp["bbox"])) != anchor_bbox]
        if len(moved) != 1:
            return False
        moved = moved[0]
        mr0, mc0, _mr1, _mc1 = moved["bbox"]
        ar0, ac0, ar1, ac1 = anchor_bbox
        if (int(mr0), int(mc0)) != (int(ar1) + 1, int(ac1) + 1):
            return False
        if _normalize_coords(moved["coords"]) != sprite_signature:
            return False

    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(18, 24)
    width = random.randint(13, 18)
    sep_r = random.randint(7, height - 8)
    upper_h = sep_r
    lower_h = height - sep_r - 1

    sprite_patch = _sprite_patch()
    if sprite_patch.shape[0] > lower_h - 2:
        return None

    colors = random.sample(LABEL_COLORS, 3)

    upper_input = np.full((upper_h, width), BG, dtype=int)
    upper_output = np.full((upper_h, width), BG, dtype=int)
    upper_reserved = np.full((upper_h, width), BG, dtype=int)
    anchor_positions: dict[int, tuple[int, int]] = {}

    for color in random.sample(colors, len(colors)):
        anchor_patch = _anchor_patch(color)
        group_patch = _group_patch(anchor_patch, sprite_patch, color)
        reserve_patch = np.ones(group_patch.shape, dtype=int)
        try:
            loc = random_free_location(
                upper_reserved,
                reserve_patch,
                transparent=BG,
                border=0,
                padding=1,
            )
        except ValueError:
            return None
        paste(upper_reserved, reserve_patch, loc[0], loc[1])
        paste(upper_input, anchor_patch, loc[0], loc[1], transparent=BG)
        paste(upper_output, group_patch, loc[0], loc[1], transparent=BG)
        anchor_positions[int(color)] = (int(loc[0]), int(loc[1]))

    lower_scene = np.full((lower_h, width), BG, dtype=int)
    lower_reserved = np.full((lower_h, width), BG, dtype=int)
    sprite_locs: list[tuple[int, int]] = []
    reserve_sprite = np.ones(sprite_patch.shape, dtype=int)
    for _ in range(3):
        try:
            loc = random_free_location(
                lower_reserved,
                reserve_sprite,
                transparent=BG,
                border=1,
                padding=2,
            )
        except ValueError:
            return None
        paste(lower_reserved, reserve_sprite, loc[0], loc[1])
        paste(lower_scene, sprite_patch, loc[0], loc[1], transparent=BG)
        sprite_locs.append((int(loc[0]), int(loc[1])))

    sprite_components = [
        {
            "bbox": (
                int(loc[0]),
                int(loc[1]),
                int(loc[0] + sprite_patch.shape[0] - 1),
                int(loc[1] + sprite_patch.shape[1] - 1),
            ),
            "coords": _sprite_coords_from_loc(loc, sprite_patch),
        }
        for loc in sprite_locs
    ]
    lower_order = _sorted_component_indices(sprite_components)
    upper_order_colors = [color for color, _pos in sorted(anchor_positions.items(), key=lambda item: item[1])]
    color_perms = [list(perm) for perm in permutations(upper_order_colors) if list(perm) != upper_order_colors]
    if not color_perms:
        return None
    assigned_order_colors = random.choice(color_perms)
    sprite_color_by_idx: dict[int, int] = {}
    for rank, sprite_idx in enumerate(lower_order):
        sprite_color_by_idx[sprite_idx] = int(assigned_order_colors[rank])

    sprite_coords_list = [component["coords"] for component in sprite_components]
    marker_points = _choose_markers(sprite_coords_list, lower_scene, lower_order)
    if marker_points is None:
        return None

    lower_input = lower_scene.copy()
    for sprite_idx, point in marker_points.items():
        lower_input[point] = int(sprite_color_by_idx[sprite_idx])

    inp = np.full((height, width), BG, dtype=int)
    inp[:upper_h, :] = upper_input
    inp[sep_r, :] = SEP
    inp[sep_r + 1 :, :] = lower_input

    expected_out = _build_expected_output((height, width), sep_r, upper_output)
    if not _semantic_check(inp, expected_out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
