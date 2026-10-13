from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import ndimage


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
STRUCT8 = np.ones((3, 3), dtype=bool)

GRID_H_RANGE = (16, 26)
GRID_W_RANGE = (16, 26)
CHAIN_RELAY_CHOICES = (0, 1, 2, 3)
DECOY_COUNT_CHOICES = (0, 1, 1, 2)
CONTENT_COUNT_MAX = 6
GEN_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(np.asarray(grid))))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def components_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    structure = np.ones((3, 3), dtype=bool) if connectivity == 8 else np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        dtype=bool,
    )
    labeled, count = ndimage.label(mask.astype(bool), structure=structure)
    return [np.argwhere(labeled == idx) for idx in range(1, int(count) + 1)]


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _flood_from_bbox_side(border_set: set[tuple[int, int]], box: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    if side == "U":
        seeds = [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in border_set]
    elif side == "D":
        seeds = [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in border_set]
    elif side == "L":
        seeds = [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in border_set]
    else:
        seeds = [(r, c1) for r in range(r0, r1 + 1) if (r, c1) not in border_set]

    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in visited or (nr, nc) in border_set:
                continue
            visited.add((nr, nc))
            stack.append((nr, nc))
    return visited


def _ray_cells(shape: tuple[int, int], spike: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = DIR4[direction]
    rr, cc = spike[0] + dr, spike[1] + dc
    cells = []
    while 0 <= rr < h and 0 <= cc < w:
        cells.append((int(rr), int(cc)))
        rr += dr
        cc += dc
    return cells


def _ray_hit(spike: tuple[int, int], direction: str, mouth_cells: set[tuple[int, int]]) -> tuple[int, tuple[int, int]] | None:
    sr, sc = spike
    if direction == "R":
        options = [(c - sc, (r, c)) for r, c in mouth_cells if r == sr and c > sc]
    elif direction == "L":
        options = [(sc - c, (r, c)) for r, c in mouth_cells if r == sr and c < sc]
    elif direction == "D":
        options = [(r - sr, (r, c)) for r, c in mouth_cells if c == sc and r > sr]
    else:
        options = [(sr - r, (r, c)) for r, c in mouth_cells if c == sc and r < sr]
    if not options:
        return None
    return min(options, key=lambda item: item[0])


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    assert L >= 1 and R >= 1 and hL >= 1 and hR >= 1
    w = L + R + 1
    h = max(L + hL, R + hR) + 1
    patch = np.zeros((h, w), dtype=int)
    patch[0, L] = color
    patch[np.arange(1, L + 1), L - np.arange(1, L + 1)] = color
    patch[np.arange(1, R + 1), L + np.arange(1, R + 1)] = color
    patch[L + 1:L + hL + 1, 0] = color
    patch[R + 1:R + hR + 1, w - 1] = color
    return patch, (0, L)


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
    h, w = patch.shape
    border_mask = patch == border_color
    if opposite_side == "U":
        seeds = [(0, c) for c in range(w) if not border_mask[0, c]]
    elif opposite_side == "D":
        seeds = [(h - 1, c) for c in range(w) if not border_mask[h - 1, c]]
    elif opposite_side == "L":
        seeds = [(r, 0) for r in range(h) if not border_mask[r, 0]]
    else:
        seeds = [(r, w - 1) for r in range(h) if not border_mask[r, w - 1]]

    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and not border_mask[nr, nc]:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return sorted(visited)


def _rotate_to_direction(
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    h, w = patch.shape
    sr, sc = spike
    if direction == "U":
        return patch.copy(), (sr, sc), list(mouth)
    if direction == "D":
        return np.flipud(patch).copy(), (h - 1 - sr, sc), [(h - 1 - r, c) for r, c in mouth]
    if direction == "L":
        return np.rot90(patch, k=1).copy(), (w - 1 - sc, sr), [(w - 1 - c, r) for r, c in mouth]
    if direction == "R":
        return np.rot90(patch, k=-1).copy(), (sc, h - 1 - sr), [(c, h - 1 - r) for r, c in mouth]
    raise ValueError(direction)


def _sample_rotated_cup(rng: np.random.Generator, border_color: int, direction: str) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]] | None:
    for _ in range(60):
        L = int(rng.integers(1, 4))
        R = int(rng.integers(1, 4))
        hL = int(rng.integers(1, 4))
        hR = int(rng.integers(1, 4))
        patch_canon, spike_canon = _build_canonical_cup(L, R, hL, hR, border_color)
        mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
        patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
        if len(mouth) >= 2 and max(patch.shape) <= 8:
            return patch, spike, mouth
    return None


def _detect_cups(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    cups = []
    for border_color in colors_present(g, exclude={bg}):
        for border_coords in components_mask(g == int(border_color), connectivity=8):
            if border_coords.shape[0] < 5:
                continue
            r0, c0, r1, c1 = bbox_coords(border_coords)
            if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
                continue
            border_set = {(int(r), int(c)) for r, c in border_coords}
            corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
            edges = (
                ("U", [p for p in border_set if p[0] == r0]),
                ("D", [p for p in border_set if p[0] == r1]),
                ("L", [p for p in border_set if p[1] == c0]),
                ("R", [p for p in border_set if p[1] == c1]),
            )
            candidates = [(d, cells[0]) for d, cells in edges if len(cells) == 1 and cells[0] not in corners]
            if len(candidates) != 1:
                continue
            direction, spike = candidates[0]
            mouth = _flood_from_bbox_side(border_set, (r0, c0, r1, c1), OPPOSITE[direction])
            if not mouth:
                continue
            content_cells = [(r, c) for r, c in mouth if int(g[r, c]) not in (bg, int(border_color))]
            content_colors = {int(g[r, c]) for r, c in content_cells}
            if len(content_colors) > 1:
                continue
            cups.append(
                {
                    "border_color": int(border_color),
                    "direction": direction,
                    "spike": spike,
                    "mouth": set(mouth),
                    "content_cells": content_cells,
                    "content_color": next(iter(content_colors)) if content_colors else None,
                }
            )
    return bg, cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relay the unique loaded cup's contents through any cups entered from their mouths, then draw the final straight ejection.

    Input structure:
    - The background is black, and several hollow cup frames of distinct border colors sit apart on the canvas.
    - Each cup has one single-cell tail spike on one bbox edge and an open mouth cavity on the opposite side.
    - Exactly one cup contains several same-colored content pixels inside its mouth cavity; all other cups are empty.

    Rule:
    - Find the unique loaded cup and erase its content pixels.
    - Starting from that cup, follow the tail direction ray from its spike; if the ray first enters an empty cup through that cup's mouth cavity, transfer the full content count to that cup and continue from its own tail.
    - When the current cup's ray reaches no further mouth cavity, draw a straight line of the original content color from the spike outward, one pixel per original content pixel.

    Key insight:
    - A cup only participates if the previous ray enters its mouth, so the answer is determined by tracing the nearest mouth-capture chain from the loaded cup, not by firing every cup independently.
    """
    g = np.asarray(grid, dtype=int)
    bg, cups = _detect_cups(g)
    loaded = [idx for idx, cup in enumerate(cups) if cup["content_cells"]]
    assert len(loaded) == 1, "expected exactly one loaded cup"

    start = cups[loaded[0]]
    count = len(start["content_cells"])
    color = int(start["content_color"])

    out = g.copy()
    for r, c in start["content_cells"]:
        out[r, c] = bg

    current_idx = loaded[0]
    visited = set()
    while True:
        if current_idx in visited:
            raise AssertionError("relay chain must not cycle")
        visited.add(current_idx)
        current = cups[current_idx]
        direction = current["direction"]
        next_options = []
        for idx, cup in enumerate(cups):
            if idx == current_idx or idx in visited:
                continue
            if cup["direction"] != direction:
                continue
            hit = _ray_hit(current["spike"], direction, cup["mouth"])
            if hit is None:
                continue
            dist, cell = hit
            next_options.append((dist, idx, cell))

        if not next_options:
            for rr, cc in _ray_cells(g.shape, current["spike"], direction)[:count]:
                out[rr, cc] = color
            return out

        next_options.sort(key=lambda item: item[0])
        current_idx = int(next_options[0][1])


def _paint_patch(canvas: np.ndarray, patch: np.ndarray, anchor: tuple[int, int]) -> None:
    r0, c0 = anchor
    mask = patch != 0
    canvas[r0:r0 + patch.shape[0], c0:c0 + patch.shape[1]][mask] = patch[mask]


def _anchor_bbox(anchor: tuple[int, int], patch: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = anchor
    return int(r0), int(c0), int(r0 + patch.shape[0] - 1), int(c0 + patch.shape[1] - 1)


def _place_chain_cup(
    rng: np.random.Generator,
    shape: tuple[int, int],
    patch: np.ndarray,
    spike_local: tuple[int, int],
    mouth_local: list[tuple[int, int]],
    direction: str,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
    prev_spike: tuple[int, int] | None,
) -> dict | None:
    h, w = shape
    ph, pw = patch.shape
    dilated = ndimage.binary_dilation(occ_cup, structure=STRUCT8)
    anchors = [(r0, c0) for r0 in range(h - ph + 1) for c0 in range(w - pw + 1)]
    rng.shuffle(anchors)

    patch_border = {(int(r), int(c)) for r, c in np.argwhere(patch != 0)}
    for anchor in anchors:
        r0, c0 = anchor
        if dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue

        spike_abs = (int(r0 + spike_local[0]), int(c0 + spike_local[1]))
        mouth_abs = {(int(r0 + r), int(c0 + c)) for r, c in mouth_local}
        ray = _ray_cells(shape, spike_abs, direction)
        if not ray:
            continue

        incoming_cells = []
        if prev_spike is not None:
            hit = _ray_hit(prev_spike, direction, mouth_abs)
            if hit is None:
                continue
            dist, capture_cell = hit
            incoming_cells = _ray_cells(shape, prev_spike, direction)[:dist]
            if not incoming_cells or incoming_cells[-1] != capture_cell:
                continue
            blocked = False
            for rr, cc in incoming_cells[:-1]:
                if occ_cup[rr, cc] or occ_path[rr, cc]:
                    blocked = True
                    break
                if (rr - r0, cc - c0) in patch_border:
                    blocked = True
                    break
            if blocked:
                continue

        return {
            "patch": patch,
            "anchor": anchor,
            "spike": spike_abs,
            "mouth": mouth_abs,
            "direction": direction,
            "bbox": _anchor_bbox(anchor, patch),
            "ray": ray,
            "incoming_cells": incoming_cells,
        }
    return None


def _place_decoy_cup(
    rng: np.random.Generator,
    shape: tuple[int, int],
    border_color: int,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
    preferred_direction: str,
) -> dict | None:
    directions = [preferred_direction, "U", "D", "L", "R"]
    rng.shuffle(directions)
    for direction in directions:
        sampled = _sample_rotated_cup(rng, border_color, direction)
        if sampled is None:
            continue
        patch, spike_local, mouth_local = sampled
        cup = _place_chain_cup(
            rng,
            shape,
            patch,
            spike_local,
            mouth_local,
            direction,
            occ_cup,
            occ_path,
            prev_spike=None,
        )
        if cup is None:
            continue
        if not cup["ray"]:
            continue
        return cup
    return None


def _commit_cup(meta: dict, occ_cup: np.ndarray, occ_path: np.ndarray) -> None:
    r0, c0, r1, c1 = meta["bbox"]
    occ_cup[r0:r1 + 1, c0:c1 + 1] = True
    for rr, cc in meta["incoming_cells"]:
        occ_path[rr, cc] = True


def _build_input_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    shape = (h, w)
    relay_count = int(rng.choice(CHAIN_RELAY_CHOICES))
    chain_len = relay_count + 1
    decoy_count = int(rng.choice(DECOY_COUNT_CHOICES))
    total_cups = chain_len + decoy_count
    if total_cups > 7:
        return None

    palette = list(range(1, 10))
    rng.shuffle(palette)
    content_color = int(palette.pop())
    border_colors = [int(palette.pop()) for _ in range(total_cups)]
    chain_direction = str(rng.choice(["U", "D", "L", "R"]))

    occ_cup = np.zeros(shape, dtype=bool)
    occ_path = np.zeros(shape, dtype=bool)
    chain = []
    prev_spike = None

    for border_color in border_colors[:chain_len]:
        placed = None
        for _ in range(80):
            sampled = _sample_rotated_cup(rng, border_color, chain_direction)
            if sampled is None:
                continue
            patch, spike_local, mouth_local = sampled
            placed = _place_chain_cup(
                rng,
                shape,
                patch,
                spike_local,
                mouth_local,
                chain_direction,
                occ_cup,
                occ_path,
                prev_spike,
            )
            if placed is not None:
                break
        if placed is None:
            return None
        _commit_cup(placed, occ_cup, occ_path)
        chain.append(placed)
        prev_spike = placed["spike"]

    terminal_ray = chain[-1]["ray"]
    max_content = min(CONTENT_COUNT_MAX, len(chain[0]["mouth"]), len(terminal_ray))
    if max_content < 1:
        return None
    count = int(rng.integers(1, max_content + 1))

    for rr, cc in terminal_ray:
        occ_path[rr, cc] = True

    decoys = []
    for border_color in border_colors[chain_len:]:
        placed = _place_decoy_cup(
            rng,
            shape,
            border_color,
            occ_cup,
            occ_path,
            preferred_direction=OPPOSITE[chain_direction] if rng.random() < 0.5 else chain_direction,
        )
        if placed is None:
            return None
        _commit_cup(placed, occ_cup, occ_path)
        for rr, cc in placed["ray"]:
            occ_path[rr, cc] = True
        decoys.append(placed)

    grid = np.zeros(shape, dtype=int)
    for meta, border_color in zip(chain + decoys, border_colors):
        patch = meta["patch"].copy()
        patch[patch != 0] = int(border_color)
        _paint_patch(grid, patch, meta["anchor"])

    load_cells = list(chain[0]["mouth"])
    rng.shuffle(load_cells)
    for r, c in load_cells[:count]:
        grid[r, c] = content_color

    try:
        out = main(grid)
    except Exception:
        return None

    produced_bg, cups = _detect_cups(grid)
    if produced_bg != 0:
        return None
    if len([cup for cup in cups if cup["content_cells"]]) != 1:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_input_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-cup input")
