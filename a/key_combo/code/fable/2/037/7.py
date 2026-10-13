from __future__ import annotations

from dataclasses import dataclass
import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, in_bounds, paste


BG = Color.BLACK
CANVAS_LIMIT = 30
MAX_GENERATION_ATTEMPTS = 500
CHAIN_RETRIES = 160
PALETTE = tuple(int(color) for color in Color.NOT_BLACK)
CORNERS = ("tl", "tr", "bl", "br")
CORNER_TO_DIR = {
    "tl": (-1, -1),
    "tr": (-1, 1),
    "bl": (1, -1),
    "br": (1, 1),
}
DIR_TO_ARRIVAL = {
    (-1, -1): "br",
    (-1, 1): "bl",
    (1, -1): "tr",
    (1, 1): "tl",
}


@dataclass(frozen=True)
class RectSpec:
    bbox: tuple[int, int, int, int]
    fill: int
    missing_corner: str | None
    marker_local: tuple[int, int] | None = None
    marker_color: int | None = None


def _bbox_hw(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _corner_abs(bbox: tuple[int, int, int, int], corner: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if corner == "tl":
        return int(r0), int(c0)
    if corner == "tr":
        return int(r0), int(c1)
    if corner == "bl":
        return int(r1), int(c0)
    if corner == "br":
        return int(r1), int(c1)
    raise ValueError(f"unknown corner {corner}")


def _corner_local(shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = shape
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, int(w - 1)
    if corner == "bl":
        return int(h - 1), 0
    if corner == "br":
        return int(h - 1), int(w - 1)
    raise ValueError(f"unknown corner {corner}")


def _bbox_from_corner(
    corner_coord: tuple[int, int], corner: str, height: int, width: int
) -> tuple[int, int, int, int]:
    rr, cc = map(int, corner_coord)
    height = int(height)
    width = int(width)
    if corner == "tl":
        r0, c0 = rr, cc
    elif corner == "tr":
        r0, c0 = rr, cc - width + 1
    elif corner == "bl":
        r0, c0 = rr - height + 1, cc
    elif corner == "br":
        r0, c0 = rr - height + 1, cc - width + 1
    else:
        raise ValueError(f"unknown corner {corner}")
    return int(r0), int(c0), int(r0 + height - 1), int(c0 + width - 1)


def _rect_patch(spec: RectSpec) -> np.ndarray:
    h, w = _bbox_hw(spec.bbox)
    patch = np.full((h, w), int(spec.fill), dtype=int)
    if spec.missing_corner is not None:
        mr, mc = _corner_local((h, w), spec.missing_corner)
        patch[mr, mc] = BG
    if spec.marker_local is not None:
        assert spec.marker_color is not None
        patch[spec.marker_local] = int(spec.marker_color)
    return patch


def _occupied_coords(spec: RectSpec) -> np.ndarray:
    patch = _rect_patch(spec)
    coords = np.argwhere(patch != BG)
    coords[:, 0] += int(spec.bbox[0])
    coords[:, 1] += int(spec.bbox[1])
    return coords


def _terminal_signature(spec: RectSpec) -> tuple[tuple[int, int], int]:
    assert spec.missing_corner is None
    return _bbox_hw(spec.bbox), int(spec.fill)


def _analyze_component(grid: np.ndarray, obj: dict, bg: int) -> dict:
    bbox = tuple(map(int, obj["bbox"]))
    patch = crop_bbox(grid, bbox)
    h, w = patch.shape
    present = patch != int(bg)
    holes = np.argwhere(~present)
    assert len(holes) in (0, 1), "rectangle may miss at most one cell"

    missing_corner = None
    if len(holes) == 1:
        hole = tuple(map(int, holes[0]))
        hole_to_corner = {
            (0, 0): "tl",
            (0, w - 1): "tr",
            (h - 1, 0): "bl",
            (h - 1, w - 1): "br",
        }
        assert hole in hole_to_corner, "the only hole must be a corner"
        missing_corner = hole_to_corner[hole]

    colors, counts = np.unique(patch[present], return_counts=True)
    assert len(colors) >= 1, "rectangle must contain foreground"
    fill = int(colors[int(np.argmax(counts))])
    marker_mask = present & (patch != fill)
    assert int(marker_mask.sum()) <= 1, "only the start rectangle may carry one marker pixel"

    marker_abs = None
    marker_color = None
    if marker_mask.any():
        mr, mc = map(int, np.argwhere(marker_mask)[0])
        assert 0 < mr < h - 1 and 0 < mc < w - 1, "marker must lie inside the rectangle"
        marker_abs = (int(bbox[0] + mr), int(bbox[1] + mc))
        marker_color = int(patch[mr, mc])
        assert missing_corner is not None, "the marked rectangle must also have a missing corner"

    return {
        "bbox": bbox,
        "coords": np.asarray(obj["coords"], dtype=int),
        "fill": int(fill),
        "missing_corner": missing_corner,
        "marker_abs": marker_abs,
        "marker_color": marker_color,
    }


def _first_hit(
    owner: np.ndarray, start: tuple[int, int], direction: tuple[int, int], self_idx: int
) -> int:
    dr, dc = map(int, direction)
    r, c = int(start[0] + dr), int(start[1] + dc)
    while in_bounds(owner.shape, r, c):
        hit = int(owner[r, c])
        if hit >= 0:
            assert hit != int(self_idx), "ray must leave the current rectangle before the next hit"
            return hit
        r += dr
        c += dc
    raise AssertionError("every missing-corner ray must hit another rectangle")


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    objects = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert len(objects) >= 4, "expected several separated rectangles"

    owner = np.full(g.shape, -1, dtype=int)
    rects: list[dict] = []
    start_candidates: list[int] = []
    for idx, obj in enumerate(objects):
        info = _analyze_component(g, obj, bg)
        coords = info["coords"]
        assert bool((owner[coords[:, 0], coords[:, 1]] < 0).all()), "objects must not overlap"
        owner[coords[:, 0], coords[:, 1]] = int(idx)
        if info["marker_abs"] is not None:
            start_candidates.append(int(idx))
        rects.append(info)

    assert len(start_candidates) == 1, "expected exactly one marked start rectangle"
    start_idx = int(start_candidates[0])

    terminal_count = 0
    for idx, info in enumerate(rects):
        if info["missing_corner"] is None:
            terminal_count += 1
            info["next_idx"] = None
            continue
        notch = _corner_abs(info["bbox"], info["missing_corner"])
        direction = CORNER_TO_DIR[info["missing_corner"]]
        info["next_idx"] = _first_hit(owner, notch, direction, idx)
    assert terminal_count >= 2, "there must be multiple intact rectangles as chain terminals"

    return {
        "bg": bg,
        "grid": g,
        "rects": rects,
        "start_idx": start_idx,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the marked missing-corner rectangle chain and crop the intact rectangle where that chain ends.

    Input structure:
    - The background contains several separated single-color solid rectangles, some with exactly one missing corner.
    - Exactly one missing-corner rectangle contains a single interior marker pixel of another color; that marked rectangle is the start.
    - Every missing corner points diagonally toward exactly one later rectangle, while multiple independent chains and multiple intact terminal rectangles coexist.

    Rule:
    - Detect the marked start rectangle, read its missing corner, and cast a diagonal ray outward from that corner to the first rectangle it hits.
    - If that hit rectangle also has a missing corner, keep following the new diagonal pointer from its own missing corner.
    - Stop at the first rectangle with all four corners present and return only that rectangle's tight crop.

    Key insight:
    - The missing corner is not damage; it is a directional pointer, so the correct output is only recoverable by tracing the full start-marked pointer chain and ignoring decoy chains.
    """
    scene = _extract_scene(np.asarray(grid, dtype=int))
    rects = scene["rects"]
    idx = int(scene["start_idx"])
    seen: set[int] = set()
    while rects[idx]["missing_corner"] is not None:
        assert idx not in seen, "pointer chain must not cycle"
        seen.add(idx)
        idx = int(rects[idx]["next_idx"])
    return crop_bbox(scene["grid"], rects[idx]["bbox"])


def _random_fill(marker_color: int) -> int:
    return int(random.choice([color for color in PALETTE if int(color) != int(marker_color)]))


def _sample_marker_local(height: int, width: int) -> tuple[int, int]:
    cells = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    return random.choice(cells)


def _can_place(spec: RectSpec, blocked: np.ndarray, reserved: np.ndarray) -> bool:
    r0, c0, r1, c1 = spec.bbox
    if r0 < 0 or c0 < 0 or r1 >= CANVAS_LIMIT or c1 >= CANVAS_LIMIT:
        return False
    coords = _occupied_coords(spec)
    if coords.size == 0:
        return False
    return not bool(blocked[coords[:, 0], coords[:, 1]].any() or reserved[coords[:, 0], coords[:, 1]].any())


def _commit_spec(spec: RectSpec, occupied: np.ndarray, blocked: np.ndarray) -> None:
    coords = _occupied_coords(spec)
    occupied[coords[:, 0], coords[:, 1]] = True
    r0, c0, r1, c1 = spec.bbox
    rr0 = max(0, int(r0 - 1))
    cc0 = max(0, int(c0 - 1))
    rr1 = min(CANVAS_LIMIT - 1, int(r1 + 1))
    cc1 = min(CANVAS_LIMIT - 1, int(c1 + 1))
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _reserve_link(
    source: RectSpec, arrival: tuple[int, int], reserved: np.ndarray
) -> None:
    assert source.missing_corner is not None
    dr, dc = CORNER_TO_DIR[source.missing_corner]
    r, c = _corner_abs(source.bbox, source.missing_corner)
    r += dr
    c += dc
    while (r, c) != tuple(map(int, arrival)):
        reserved[r, c] = True
        r += dr
        c += dc


def _sample_first_spec(is_start: bool, marker_color: int) -> RectSpec:
    height = random.randint(3, 5 if is_start else 6)
    width = random.randint(3, 5 if is_start else 6)
    r0 = random.randint(0, CANVAS_LIMIT - height)
    c0 = random.randint(0, CANVAS_LIMIT - width)
    marker_local = _sample_marker_local(height, width) if is_start else None
    return RectSpec(
        bbox=(r0, c0, r0 + height - 1, c0 + width - 1),
        fill=_random_fill(marker_color),
        missing_corner=random.choice(CORNERS),
        marker_local=marker_local,
        marker_color=int(marker_color) if is_start else None,
    )


def _sample_next_spec(
    source: RectSpec,
    occupied: np.ndarray,
    blocked: np.ndarray,
    reserved: np.ndarray,
    marker_color: int,
    terminal_signatures: set[tuple[tuple[int, int], int]],
    terminal: bool,
) -> tuple[RectSpec, tuple[int, int]] | None:
    assert source.missing_corner is not None
    dr, dc = CORNER_TO_DIR[source.missing_corner]
    notch = _corner_abs(source.bbox, source.missing_corner)
    arrival_corner = DIR_TO_ARRIVAL[(dr, dc)]

    for _ in range(120):
        height = random.randint(3, 6)
        width = random.randint(3, 6)
        gap = random.randint(3, 8)
        arrival = (int(notch[0] + dr * gap), int(notch[1] + dc * gap))
        bbox = _bbox_from_corner(arrival, arrival_corner, height, width)

        if terminal:
            missing_corner = None
        else:
            candidates = [corner for corner in CORNERS if corner != arrival_corner]
            missing_corner = random.choice(candidates)

        spec = RectSpec(
            bbox=bbox,
            fill=_random_fill(marker_color),
            missing_corner=missing_corner,
        )
        if terminal and _terminal_signature(spec) in terminal_signatures:
            continue
        if not _can_place(spec, blocked, reserved):
            continue

        r, c = int(notch[0] + dr), int(notch[1] + dc)
        blocked_by_existing = False
        while (r, c) != arrival:
            if not in_bounds(occupied.shape, r, c):
                blocked_by_existing = True
                break
            if occupied[r, c]:
                blocked_by_existing = True
                break
            r += dr
            c += dc
        if blocked_by_existing:
            continue
        return spec, arrival
    return None


def _place_chain(
    occupied: np.ndarray,
    blocked: np.ndarray,
    reserved: np.ndarray,
    marker_color: int,
    terminal_signatures: set[tuple[tuple[int, int], int]],
    *,
    is_start: bool,
    missing_count: int,
) -> tuple[list[RectSpec], np.ndarray, np.ndarray, np.ndarray] | None:
    for _ in range(CHAIN_RETRIES):
        occ = occupied.copy()
        blk = blocked.copy()
        res = reserved.copy()

        first = _sample_first_spec(is_start, marker_color)
        if not _can_place(first, blk, res):
            continue
        chain = [first]
        _commit_spec(first, occ, blk)

        current = first
        ok = True
        for hop in range(int(missing_count)):
            terminal = hop == int(missing_count) - 1
            nxt = _sample_next_spec(
                current,
                occ,
                blk,
                res,
                marker_color,
                terminal_signatures,
                terminal=terminal,
            )
            if nxt is None:
                ok = False
                break
            next_spec, arrival = nxt
            _reserve_link(current, arrival, res)
            _commit_spec(next_spec, occ, blk)
            chain.append(next_spec)
            current = next_spec

        if not ok:
            continue
        if is_start:
            directions = [CORNER_TO_DIR[spec.missing_corner] for spec in chain[:-1]]
            if len(directions) < 2 or len(set(directions)) < 2:
                continue
        return chain, occ, blk, res
    return None


def _render_canvas(chains: list[list[RectSpec]]) -> np.ndarray:
    canvas = np.full((CANVAS_LIMIT, CANVAS_LIMIT), BG, dtype=int)
    for chain in chains:
        for spec in chain:
            r0, c0, _r1, _c1 = spec.bbox
            paste(canvas, _rect_patch(spec), r0, c0, transparent=BG)
    return canvas


def _crop_canvas(canvas: np.ndarray) -> np.ndarray:
    coords = np.argwhere(canvas != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = random.randint(0, min(2, int(r0)))
    left = random.randint(0, min(2, int(c0)))
    bottom = random.randint(0, min(2, int(CANVAS_LIMIT - 1 - r1)))
    right = random.randint(0, min(2, int(CANVAS_LIMIT - 1 - c1)))
    return canvas[int(r0 - top) : int(r1 + bottom) + 1, int(c0 - left) : int(c1 + right) + 1].copy()


def _validate_generated_grid(grid: np.ndarray, expected: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert max(grid.shape) <= CANVAS_LIMIT
    assert int(get_bg(grid)) == BG, "background must remain the most common color"

    scene = _extract_scene(grid)
    rects = scene["rects"]
    assert rects[scene["start_idx"]]["missing_corner"] is not None

    incoming = [0] * len(rects)
    pointer_targets: list[int] = []
    for info in rects:
        if info["next_idx"] is None:
            continue
        nxt = int(info["next_idx"])
        incoming[nxt] += 1
        pointer_targets.append(nxt)
    assert len(pointer_targets) == len(set(pointer_targets)), "chains must stay disjoint"

    heads = [
        idx
        for idx, info in enumerate(rects)
        if info["missing_corner"] is not None and incoming[idx] == 0
    ]
    assert len(heads) >= 2, "there must be at least one decoy chain"
    assert int(scene["start_idx"]) in heads

    start_path: list[int] = []
    idx = int(scene["start_idx"])
    while rects[idx]["missing_corner"] is not None:
        start_path.append(idx)
        idx = int(rects[idx]["next_idx"])
    assert len(start_path) >= 2, "the marked chain must require multiple jumps"
    directions = [CORNER_TO_DIR[rects[path_idx]["missing_corner"]] for path_idx in start_path]
    assert len(set(directions)) >= 2, "the marked chain must turn at least once"

    terminals = [i for i, info in enumerate(rects) if info["missing_corner"] is None]
    terminal_crops = {
        (crop_bbox(grid, rects[i]["bbox"]).shape, crop_bbox(grid, rects[i]["bbox"]).tobytes())
        for i in terminals
    }
    assert len(terminal_crops) == len(terminals), "terminal crops should be pairwise distinct"

    actual = main(grid)
    assert np.array_equal(actual, expected), "solver must match the generator's intended terminal"


def _gen_attempt() -> np.ndarray | None:
    marker_color = int(random.choice(PALETTE))
    chain_count = random.choice((2, 3))
    missing_counts = [random.choice((2, 3, 4))]
    for _ in range(chain_count - 1):
        missing_counts.append(random.choices((1, 2, 3), weights=(2, 5, 3), k=1)[0])

    occupied = np.zeros((CANVAS_LIMIT, CANVAS_LIMIT), dtype=bool)
    blocked = np.zeros((CANVAS_LIMIT, CANVAS_LIMIT), dtype=bool)
    reserved = np.zeros((CANVAS_LIMIT, CANVAS_LIMIT), dtype=bool)
    terminal_signatures: set[tuple[tuple[int, int], int]] = set()
    chains: list[list[RectSpec]] = []
    expected: np.ndarray | None = None

    for chain_index, missing_count in enumerate(missing_counts):
        placed = _place_chain(
            occupied,
            blocked,
            reserved,
            marker_color,
            terminal_signatures,
            is_start=chain_index == 0,
            missing_count=missing_count,
        )
        if placed is None:
            return None
        chain, occupied, blocked, reserved = placed
        terminal_signatures.add(_terminal_signature(chain[-1]))
        chains.append(chain)
        if chain_index == 0:
            expected = _rect_patch(chain[-1])

    assert expected is not None
    grid = _crop_canvas(_render_canvas(chains))
    try:
        _validate_generated_grid(grid, expected)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid missing-corner pointer-chain input")
