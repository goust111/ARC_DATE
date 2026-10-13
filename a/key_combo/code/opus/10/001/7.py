from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
YELLOW = 4
FRAME = 9
PALETTE = (1, 2, 3, 5, 6, 7, 8)
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mono_components(grid: np.ndarray, colors: set[int] | None = None) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            if colors is not None and color not in colors:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS8:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            r0, c0, r1, c1 = _bbox(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )
    return comps


def _component_patch(comp: dict, color: int) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in comp["coords"]:
        patch[r - r0, c - c0] = color
    return patch


def _detect_plus_markers(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    markers: list[dict] = []
    for r, c in np.argwhere(grid == YELLOW):
        r = int(r)
        c = int(c)
        if not (0 < r < h - 1 and 0 < c < w - 1):
            continue
        arms = [
            int(grid[r - 1, c]),
            int(grid[r + 1, c]),
            int(grid[r, c - 1]),
            int(grid[r, c + 1]),
        ]
        if len(set(arms)) != 1:
            continue
        arm_color = arms[0]
        if arm_color in (BG, YELLOW):
            continue
        cells = {
            (r, c),
            (r - 1, c),
            (r + 1, c),
            (r, c - 1),
            (r, c + 1),
        }
        markers.append({"center": (r, c), "arm_color": arm_color, "cells": cells})
    return markers


def _is_sprite_like(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return 3 <= h <= 5 and h == w and int(comp["area"]) >= 5


def _find_candidate_sprites(grid: np.ndarray, ignore_cells: set[tuple[int, int]]) -> list[dict]:
    comps = _mono_components(grid, colors=set(PALETTE))
    sprites: list[dict] = []
    for comp in comps:
        coords = set(comp["coords"])
        if coords & ignore_cells:
            continue
        if _is_sprite_like(comp):
            patch = _component_patch(comp, int(comp["color"]))
            sprites.append({**comp, "patch": patch})
    return sprites


def _find_frames(grid: np.ndarray, side_delta: int) -> list[dict]:
    h, w = grid.shape
    marker_coords = {(int(r), int(c)) for r, c in np.argwhere(grid == FRAME)}
    rows = sorted({r for r, _ in marker_coords})
    cols = sorted({c for _, c in marker_coords})
    frames: list[dict] = []
    seen: set[frozenset[tuple[int, int]]] = set()
    for r0 in rows:
        for c0 in cols:
            r1 = r0 + side_delta
            c1 = c0 + side_delta
            if r1 >= h or c1 >= w:
                continue
            corners = {
                "tl": (r0, c0),
                "tr": (r0, c1),
                "bl": (r1, c0),
                "br": (r1, c1),
            }
            present = [kind for kind, rc in corners.items() if rc in marker_coords]
            if len(present) != 3:
                continue
            marker_set = frozenset(corners[kind] for kind in present)
            if marker_set in seen:
                continue
            missing = next(kind for kind in ("tl", "tr", "bl", "br") if kind not in present)
            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "missing_kind": missing,
                    "missing": corners[missing],
                    "markers": marker_set,
                }
            )
            seen.add(marker_set)
    return frames


def _orient_patch_top_left_reference(patch: np.ndarray, missing_kind: str) -> np.ndarray:
    out = patch.copy()
    if missing_kind in ("bl", "br"):
        out = out[::-1, :]
    if missing_kind in ("tr", "br"):
        out = out[:, ::-1]
    return out


def _paste_corner_aligned(out: np.ndarray, patch: np.ndarray, color: int, missing: tuple[int, int], missing_kind: str) -> None:
    h, w = patch.shape
    mr, mc = missing
    if missing_kind == "tl":
        r0, c0 = mr, mc
    elif missing_kind == "tr":
        r0, c0 = mr, mc - w + 1
    elif missing_kind == "bl":
        r0, c0 = mr - h + 1, mc
    else:
        r0, c0 = mr - h + 1, mc - w + 1
    target = out[r0 : r0 + h, c0 : c0 + w]
    mask = patch == color
    target[mask & (target == BG)] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Use the repeated plus-arm color to choose one sprite, then mirror-stamp it into every three-corner frame.

    Input structure:
    - The background is black.
    - Several yellow-centered plus markers appear, each with four orthogonal arms
      of one shared color; exactly one arm color is repeated across markers.
    - There are 2-3 monochrome candidate sprites and several maroon three-corner
      square frames whose missing corner indicates a stamp anchor.
    - Extra colored noise pixels may appear and must be ignored.

    Rule:
    - Detect the valid yellow-centered plus markers and count their arm colors.
    - Let T be the unique arm color that appears on more than one marker.
    - Find the candidate sprite whose color is T and keep that sprite as the template;
      erase the other candidate sprites from the output.
    - For each maroon frame, align the template's corresponding bbox corner to the
      frame's missing corner and stamp the template inside the frame. Treat the
      observed template orientation as the top-left reference, flipping vertically
      and/or horizontally when the missing corner is in another quadrant.
    - Only paint background cells when stamping; frame markers and any other
      existing foreground pixels remain unchanged.

    Key insight:
    - The repeated plus-arm color is not the answer itself: it selects which sprite
      to use, and the frame missing corners then control how that sprite is mirrored.
    """
    g = np.asarray(grid, dtype=int)
    markers = _detect_plus_markers(g)
    counts = Counter(int(marker["arm_color"]) for marker in markers)
    repeated = [color for color, count in counts.items() if count > 1]
    assert len(repeated) == 1
    target_color = repeated[0]

    plus_cells = set()
    for marker in markers:
        plus_cells |= set(marker["cells"])
    frame_cells = {(int(r), int(c)) for r, c in np.argwhere(g == FRAME)}
    candidate_sprites = _find_candidate_sprites(g, plus_cells | frame_cells)
    assert 2 <= len(candidate_sprites) <= 3

    target_candidates = [sprite for sprite in candidate_sprites if int(sprite["color"]) == target_color]
    assert len(target_candidates) == 1
    target_sprite = target_candidates[0]
    template = target_sprite["patch"]
    h, w = template.shape
    assert h == w

    frames = _find_frames(g, side_delta=h - 1)
    assert 2 <= len(frames) <= 4

    out = g.copy()
    for sprite in candidate_sprites:
        if int(sprite["color"]) != target_color:
            for r, c in sprite["coords"]:
                out[r, c] = BG

    for frame in frames:
        oriented = _orient_patch_top_left_reference(template, frame["missing_kind"])
        _paste_corner_aligned(out, oriented, target_color, frame["missing"], frame["missing_kind"])

    return out


def _random_connected_mask(n: int, require_asymmetric: bool) -> np.ndarray | None:
    for _ in range(200):
        target_area = random.randint(n + 2, n * n - 2)
        mask = np.zeros((n, n), dtype=bool)
        seed = (random.randrange(n), random.randrange(n))
        mask[seed] = True
        while int(mask.sum()) < target_area:
            frontier = set()
            for r, c in np.argwhere(mask):
                r = int(r)
                c = int(c)
                for dr, dc in NEIGHBORS8:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(list(frontier))
            mask[nr, nc] = True
        coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        if not coords:
            continue
        if _bbox(coords) != (0, 0, n - 1, n - 1):
            continue
        if require_asymmetric:
            if np.array_equal(mask, np.fliplr(mask)):
                continue
            if np.array_equal(mask, np.flipud(mask)):
                continue
            if np.array_equal(mask, np.flipud(np.fliplr(mask))):
                continue
        return mask
    return None


def _rect_free(used: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    H, W = used.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    return not used[rr0:rr1, cc0:cc1].any()


def _mark_rect(used: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    H, W = used.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    used[rr0:rr1, cc0:cc1] = True


def _sample_position(used: np.ndarray, h: int, w: int, pad: int) -> tuple[int, int] | None:
    H, W = used.shape
    candidates = [
        (r0, c0)
        for r0 in range(H - h + 1)
        for c0 in range(W - w + 1)
        if _rect_free(used, r0, c0, h, w, pad)
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _draw_plus(grid: np.ndarray, center: tuple[int, int], color: int) -> None:
    r, c = center
    grid[r, c] = YELLOW
    grid[r - 1, c] = color
    grid[r + 1, c] = color
    grid[r, c - 1] = color
    grid[r, c + 1] = color


def _draw_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    target = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    target[mask] = patch[mask]


def _frame_patch(n: int, missing_kind: str) -> np.ndarray:
    patch = np.zeros((n, n), dtype=int)
    corners = {
        "tl": (0, 0),
        "tr": (0, n - 1),
        "bl": (n - 1, 0),
        "br": (n - 1, n - 1),
    }
    for kind, (r, c) in corners.items():
        if kind != missing_kind:
            patch[r, c] = FRAME
    return patch


def _build_expected(grid: np.ndarray, sprites: list[dict], target_color: int, frames: list[dict]) -> np.ndarray:
    out = grid.copy()
    target_sprite = next(sprite for sprite in sprites if sprite["color"] == target_color)
    template = target_sprite["patch"]
    for sprite in sprites:
        if sprite["color"] != target_color:
            r0, c0 = sprite["pos"]
            h, w = sprite["patch"].shape
            mask = sprite["patch"] != BG
            out[r0 : r0 + h, c0 : c0 + w][mask] = BG
    for frame in frames:
        oriented = _orient_patch_top_left_reference(template, frame["missing_kind"])
        _paste_corner_aligned(out, oriented, target_color, frame["missing"], frame["missing_kind"])
    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(24, 30)
    W = random.randint(24, 30)
    grid = np.zeros((H, W), dtype=int)
    used = np.zeros((H, W), dtype=bool)

    sprite_count = random.choice((2, 3))
    frame_count = random.choice((2, 3, 3, 4))
    plus_count = random.choice((4, 5))
    n = random.choice((4, 5))

    sprite_colors = random.sample(PALETTE, sprite_count)
    target_color = random.choice(sprite_colors)
    other_sprite_colors = [color for color in sprite_colors if color != target_color]

    marker_colors = [target_color, target_color]
    if other_sprite_colors:
        marker_colors.append(random.choice(other_sprite_colors))
    remaining_pool = [color for color in PALETTE if color not in set(marker_colors)]
    while len(marker_colors) < plus_count:
        marker_colors.append(remaining_pool.pop(random.randrange(len(remaining_pool))))
    random.shuffle(marker_colors)

    sprite_masks: list[np.ndarray] = []
    sprite_keys: set[tuple[tuple[int, ...], ...]] = set()
    for index, color in enumerate(sprite_colors):
        require_asymmetric = color == target_color
        mask = _random_connected_mask(n, require_asymmetric=require_asymmetric)
        if mask is None:
            return None
        key = tuple(tuple(int(v) for v in row) for row in mask)
        if key in sprite_keys:
            return None
        sprite_keys.add(key)
        patch = np.where(mask, color, BG)
        pos = _sample_position(used, n, n, pad=1)
        if pos is None:
            return None
        _mark_rect(used, pos[0], pos[1], n, n, pad=1)
        _draw_patch(grid, patch, pos[0], pos[1])
        sprite_masks.append({"color": color, "patch": patch, "pos": pos})

    missing_kinds = random.sample(("tl", "tr", "bl", "br"), frame_count)
    frames: list[dict] = []
    for missing_kind in missing_kinds:
        pos = _sample_position(used, n, n, pad=1)
        if pos is None:
            return None
        _mark_rect(used, pos[0], pos[1], n, n, pad=1)
        patch = _frame_patch(n, missing_kind)
        _draw_patch(grid, patch, pos[0], pos[1])
        missing_map = {
            "tl": (pos[0], pos[1]),
            "tr": (pos[0], pos[1] + n - 1),
            "bl": (pos[0] + n - 1, pos[1]),
            "br": (pos[0] + n - 1, pos[1] + n - 1),
        }
        frames.append({"pos": pos, "missing_kind": missing_kind, "missing": missing_map[missing_kind]})

    centers: list[tuple[int, int]] = []
    for color in marker_colors:
        pos = _sample_position(used, 3, 3, pad=2)
        if pos is None:
            return None
        _mark_rect(used, pos[0], pos[1], 3, 3, pad=2)
        center = (pos[0] + 1, pos[1] + 1)
        centers.append(center)
        _draw_plus(grid, center, color)

    noise_colors = list(set(marker_colors + sprite_colors + random.sample(list(PALETTE), 2)))
    random.shuffle(noise_colors)
    noise_count = random.randint(6, 12)
    for i in range(noise_count):
        pos = _sample_position(used, 1, 1, pad=1)
        if pos is None:
            break
        _mark_rect(used, pos[0], pos[1], 1, 1, pad=1)
        grid[pos[0], pos[1]] = noise_colors[i % len(noise_colors)]

    plus_cells = set()
    detected_markers = _detect_plus_markers(grid)
    if len(detected_markers) != plus_count:
        return None
    for marker in detected_markers:
        plus_cells |= set(marker["cells"])
    detected_sprites = _find_candidate_sprites(grid, plus_cells | {(int(r), int(c)) for r, c in np.argwhere(grid == FRAME)})
    if len(detected_sprites) != sprite_count:
        return None
    if sum(int(sprite["color"]) == target_color for sprite in detected_sprites) != 1:
        return None
    detected_frames = _find_frames(grid, n - 1)
    if len(detected_frames) != frame_count:
        return None

    expected = _build_expected(grid, sprite_masks, target_color, frames)

    assert Counter(marker_colors)[target_color] == 2
    assert len(set(missing_kinds)) >= 2
    target_mask = next(sprite["patch"] == target_color for sprite in sprite_masks if sprite["color"] == target_color)
    assert not np.array_equal(target_mask, np.fliplr(target_mask))
    assert not np.array_equal(target_mask, np.flipud(target_mask))
    assert any(color in other_sprite_colors for color in marker_colors)
    assert not np.array_equal(grid, expected)

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    for sprite in sprite_masks:
        if sprite["color"] == target_color:
            continue
        r0, c0 = sprite["pos"]
        h, w = sprite["patch"].shape
        assert not np.any(actual[r0 : r0 + h, c0 : c0 + w] == sprite["color"])

    target_patch = next(sprite["patch"] for sprite in sprite_masks if sprite["color"] == target_color)
    for frame in frames:
        oriented = _orient_patch_top_left_reference(target_patch, frame["missing_kind"])
        mr, mc = frame["missing"]
        if frame["missing_kind"] == "tl":
            r0, c0 = mr, mc
        elif frame["missing_kind"] == "tr":
            r0, c0 = mr, mc - n + 1
        elif frame["missing_kind"] == "bl":
            r0, c0 = mr - n + 1, mc
        else:
            r0, c0 = mr - n + 1, mc - n + 1
        target = actual[r0 : r0 + n, c0 : c0 + n]
        assert np.all(target[_frame_patch(n, frame["missing_kind"]) == FRAME] == FRAME)
        mask = oriented == target_color
        assert np.any(target[mask] == target_color)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 7")
