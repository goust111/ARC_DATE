from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np


for _parent in Path(__file__).resolve().parents:
    _candidate = _parent / "new" / "solvers"
    if (_candidate / "arc_utils.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise ImportError("could not locate new/solvers/arc_utils.py")

from arc_utils import crop_bbox, find_components, get_bg, paste  # noqa: E402


BACKGROUND = 0
PALETTE = (1, 2, 3)
GRID_SIZE_RANGE = (18, 24)
OBJECT_COUNT_OPTIONS = (2, 3)
SPRITE_SIDE_OPTIONS = (4, 5)
BAR_GAP_RANGE = (2, 6)
MAX_GENERATION_ATTEMPTS = 400
EDGE_MARGIN = 1

MOTIF_TEMPLATES = (
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0], [0, 0]], dtype=bool),
    np.array([[0, 1], [1, 1], [0, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 0]], dtype=bool),
    np.array([[0, 1], [1, 1], [0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [0, 1, 0], [1, 1, 0]], dtype=bool),
)


def _bbox_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int], pad: int = 0) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    h, w = shape
    r0, c0, r1, c1 = bbox
    r0 = max(0, int(r0) - pad)
    c0 = max(0, int(c0) - pad)
    r1 = min(h - 1, int(r1) + pad)
    c1 = min(w - 1, int(c1) + pad)
    out[r0 : r1 + 1, c0 : c1 + 1] = True
    return out


def _solid_rectangle_component(comp: dict, grid: np.ndarray) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h <= 1 or w <= 1 or int(comp["area"]) != h * w:
        return False
    patch = crop_bbox(grid, comp["bbox"])
    return bool(np.all(patch == int(comp["color"])))


def _hole_offsets(tile_h: int, tile_w: int, motif_h: int, motif_w: int, direction: str) -> tuple[int, int]:
    off_r = (tile_h - motif_h) // 2
    off_c = (tile_w - motif_w) // 2
    if direction == "down" and (tile_h - motif_h) % 2 == 1:
        off_r += 1
    if direction == "right" and (tile_w - motif_w) % 2 == 1:
        off_c += 1
    return int(off_r), int(off_c)


def _tile_anchors(sprite_bbox: tuple[int, int, int, int], bar_bbox: tuple[int, int, int, int]) -> tuple[str, list[tuple[int, int]]]:
    sr0, sc0, sr1, sc1 = map(int, sprite_bbox)
    br0, bc0, br1, bc1 = map(int, bar_bbox)
    tile_h = sr1 - sr0 + 1
    tile_w = sc1 - sc0 + 1

    anchors: list[tuple[int, int]] = []
    if br0 == br1:
        assert (bc0, bc1) == (sc0, sc1)
        if br0 < sr0:
            direction = "up"
            r = sr0
            while r + tile_h - 1 >= br0:
                anchors.append((int(r), int(sc0)))
                r -= tile_h
        else:
            direction = "down"
            r = sr0
            while r <= br0:
                anchors.append((int(r), int(sc0)))
                r += tile_h
    else:
        assert (br0, br1) == (sr0, sr1)
        if bc0 < sc0:
            direction = "left"
            c = sc0
            while c + tile_w - 1 >= bc0:
                anchors.append((int(sr0), int(c)))
                c -= tile_w
        else:
            direction = "right"
            c = sc0
            while c <= bc0:
                anchors.append((int(sr0), int(c)))
                c += tile_w
    return direction, anchors


def _render_group(
    out: np.ndarray,
    color: int,
    sprite_bbox: tuple[int, int, int, int],
    bar_bbox: tuple[int, int, int, int],
    motif_mask: np.ndarray,
    bg: int,
) -> None:
    sr0, sc0, sr1, sc1 = map(int, sprite_bbox)
    tile_h = sr1 - sr0 + 1
    tile_w = sc1 - sc0 + 1
    motif_h, motif_w = motif_mask.shape
    tile = np.full((tile_h, tile_w), int(color), dtype=int)
    direction, anchors = _tile_anchors(sprite_bbox, bar_bbox)
    off_r, off_c = _hole_offsets(tile_h, tile_w, motif_h, motif_w, direction)

    for ar, ac in anchors:
        paste(out, tile, int(ar), int(ac), transparent=bg, clip=True)
        for rr, cc in np.argwhere(motif_mask):
            tr = int(ar + off_r + int(rr))
            tc = int(ac + off_c + int(cc))
            if 0 <= tr < out.shape[0] and 0 <= tc < out.shape[1]:
                out[tr, tc] = int(bg)


def _parse_scene(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(comp)

    groups: list[dict] = []
    for color in sorted(by_color):
        items = by_color[color]
        sprite_candidates = []
        bar_candidates = []
        motif_candidates = []
        for comp in items:
            r0, c0, r1, c1 = map(int, comp["bbox"])
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            area = int(comp["area"])
            if _solid_rectangle_component(comp, grid) and h in SPRITE_SIDE_OPTIONS and w in SPRITE_SIDE_OPTIONS:
                sprite_candidates.append(comp)
            elif ((h == 1) ^ (w == 1)) and area == h * w:
                bar_candidates.append(comp)
            else:
                motif_candidates.append(comp)

        assert len(sprite_candidates) == 1, f"expected one sprite for color {color}"
        assert len(bar_candidates) == 1, f"expected one control bar for color {color}"
        assert len(motif_candidates) == 1, f"expected one motif for color {color}"

        sprite = sprite_candidates[0]
        bar = bar_candidates[0]
        motif = motif_candidates[0]

        motif_patch = crop_bbox(grid, motif["bbox"])
        motif_mask = motif_patch == int(color)
        assert 2 <= int(motif_mask.sum()) <= 4
        assert motif_mask.shape[0] >= 2 and motif_mask.shape[1] >= 2

        direction, anchors = _tile_anchors(sprite["bbox"], bar["bbox"])
        assert len(anchors) >= 2

        groups.append(
            {
                "color": int(color),
                "sprite_bbox": tuple(map(int, sprite["bbox"])),
                "bar_bbox": tuple(map(int, bar["bbox"])),
                "motif_bbox": tuple(map(int, motif["bbox"])),
                "motif_mask": motif_mask.astype(bool),
                "direction": direction,
            }
        )
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the control bars, tile each matched solid rectangle toward its bar, and punch a same-colored motif through every copy.

    Input structure:
    - The background is black.
    - There are 2-3 solid monochrome rectangle sprites, each 4x4 or 5x5, colored blue, red, or green.
    - Each sprite has a same-colored one-cell-thick control bar on one side with matching row span or column span.
    - Each sprite color also has one remote same-colored irregular motif component of area 2-4.

    Rule:
    - Match every bar to the same-colored sprite whose span agrees with the bar orientation.
    - Erase all control bars and all original motif components.
    - Starting from the sprite's original position, repeat that solid rectangle toward the bar with period equal to the sprite height or width.
    - In every repeated rectangle, including the original, center the same-colored motif inside the rectangle and erase the covered cells; if perfect centering is impossible, bias the tie by one cell toward the control bar.

    Key insight:
    - The bar determines where copies go, while the remote motif determines which cells are removed from every copy; both cues are required to reconstruct the final wallpaper-like strip.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    groups = _parse_scene(g)

    out = g.copy()
    for group in groups:
        br0, bc0, br1, bc1 = group["bar_bbox"]
        mr0, mc0, mr1, mc1 = group["motif_bbox"]
        out[br0 : br1 + 1, bc0 : bc1 + 1] = int(bg)
        out[mr0 : mr1 + 1, mc0 : mc1 + 1] = int(bg)

    for group in groups:
        _render_group(
            out,
            int(group["color"]),
            group["sprite_bbox"],
            group["bar_bbox"],
            group["motif_mask"],
            int(bg),
        )
    return out


def _choose_motif() -> np.ndarray:
    return random.choice(MOTIF_TEMPLATES).copy()


def _sample_group(shape: tuple[int, int], reserved: np.ndarray) -> dict | None:
    h, w = shape
    for _ in range(500):
        tile_h = random.choice(SPRITE_SIDE_OPTIONS)
        tile_w = random.choice(SPRITE_SIDE_OPTIONS)
        direction = random.choice(("up", "down", "left", "right"))
        gap = random.randint(*BAR_GAP_RANGE)
        copies = 2 + int(gap >= (tile_h if direction in {"up", "down"} else tile_w))

        if direction == "down":
            max_r0 = h - copies * tile_h - EDGE_MARGIN
            if max_r0 < EDGE_MARGIN:
                continue
            sr0 = random.randint(EDGE_MARGIN, max_r0)
            sc0 = random.randint(EDGE_MARGIN, w - tile_w - EDGE_MARGIN)
            br0 = sr0 + tile_h + gap
            sprite_bbox = (sr0, sc0, sr0 + tile_h - 1, sc0 + tile_w - 1)
            bar_bbox = (br0, sc0, br0, sc0 + tile_w - 1)
            output_bbox = (sr0, sc0, sr0 + copies * tile_h - 1, sc0 + tile_w - 1)
        elif direction == "up":
            min_r0 = (copies - 1) * tile_h + EDGE_MARGIN
            max_r0 = h - tile_h - EDGE_MARGIN
            if min_r0 > max_r0:
                continue
            sr0 = random.randint(min_r0, max_r0)
            sc0 = random.randint(EDGE_MARGIN, w - tile_w - EDGE_MARGIN)
            br0 = sr0 - gap - 1
            sprite_bbox = (sr0, sc0, sr0 + tile_h - 1, sc0 + tile_w - 1)
            bar_bbox = (br0, sc0, br0, sc0 + tile_w - 1)
            output_bbox = (sr0 - (copies - 1) * tile_h, sc0, sr0 + tile_h - 1, sc0 + tile_w - 1)
        elif direction == "right":
            max_c0 = w - copies * tile_w - EDGE_MARGIN
            if max_c0 < EDGE_MARGIN:
                continue
            sr0 = random.randint(EDGE_MARGIN, h - tile_h - EDGE_MARGIN)
            sc0 = random.randint(EDGE_MARGIN, max_c0)
            bc0 = sc0 + tile_w + gap
            sprite_bbox = (sr0, sc0, sr0 + tile_h - 1, sc0 + tile_w - 1)
            bar_bbox = (sr0, bc0, sr0 + tile_h - 1, bc0)
            output_bbox = (sr0, sc0, sr0 + tile_h - 1, sc0 + copies * tile_w - 1)
        else:
            min_c0 = (copies - 1) * tile_w + EDGE_MARGIN
            max_c0 = w - tile_w - EDGE_MARGIN
            if min_c0 > max_c0:
                continue
            sr0 = random.randint(EDGE_MARGIN, h - tile_h - EDGE_MARGIN)
            sc0 = random.randint(min_c0, max_c0)
            bc0 = sc0 - gap - 1
            sprite_bbox = (sr0, sc0, sr0 + tile_h - 1, sc0 + tile_w - 1)
            bar_bbox = (sr0, bc0, sr0 + tile_h - 1, bc0)
            output_bbox = (sr0, sc0 - (copies - 1) * tile_w, sr0 + tile_h - 1, sc0 + tile_w - 1)

        if np.any(reserved & _bbox_mask(shape, sprite_bbox, pad=1)):
            continue
        if np.any(reserved & _bbox_mask(shape, bar_bbox, pad=1)):
            continue
        if np.any(reserved & _bbox_mask(shape, output_bbox, pad=1)):
            continue

        return {
            "sprite_bbox": tuple(map(int, sprite_bbox)),
            "bar_bbox": tuple(map(int, bar_bbox)),
            "output_bbox": tuple(map(int, output_bbox)),
            "direction": direction,
        }
    return None


def _place_motifs(shape: tuple[int, int], groups: list[dict], reserved: np.ndarray) -> bool:
    h, w = shape
    order = list(range(len(groups)))
    random.shuffle(order)
    for idx in order:
        motif = _choose_motif()
        mh, mw = motif.shape
        positions = [(r, c) for r in range(EDGE_MARGIN, h - mh - EDGE_MARGIN + 1) for c in range(EDGE_MARGIN, w - mw - EDGE_MARGIN + 1)]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            bbox = (r0, c0, r0 + mh - 1, c0 + mw - 1)
            if np.any(reserved & _bbox_mask(shape, bbox, pad=1)):
                continue
            groups[idx]["motif_bbox"] = bbox
            groups[idx]["motif_mask"] = motif
            reserved |= _bbox_mask(shape, bbox, pad=1)
            placed = True
            break
        if not placed:
            return False
    return True


def _render_input(shape: tuple[int, int], groups: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for group in groups:
        color = int(group["color"])
        sr0, sc0, sr1, sc1 = group["sprite_bbox"]
        br0, bc0, br1, bc1 = group["bar_bbox"]
        mr0, mc0, _mr1, _mc1 = group["motif_bbox"]
        motif_mask = group["motif_mask"]

        grid[sr0 : sr1 + 1, sc0 : sc1 + 1] = color
        grid[br0 : br1 + 1, bc0 : bc1 + 1] = color
        patch = np.where(motif_mask, color, BACKGROUND).astype(int)
        paste(grid, patch, mr0, mc0, transparent=BACKGROUND)
    return grid


def _expected_from_specs(shape: tuple[int, int], groups: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for group in groups:
        _render_group(
            out,
            int(group["color"]),
            group["sprite_bbox"],
            group["bar_bbox"],
            group["motif_mask"],
            BACKGROUND,
        )
    return out


def _validate_specs(inp: np.ndarray, expected: np.ndarray, groups: list[dict]) -> None:
    occupied = np.zeros(inp.shape, dtype=bool)
    tiled = np.zeros(inp.shape, dtype=bool)

    for group in groups:
        color = int(group["color"])
        sprite_bbox = group["sprite_bbox"]
        bar_bbox = group["bar_bbox"]
        motif_bbox = group["motif_bbox"]
        motif_mask = group["motif_mask"]
        direction = group["direction"]
        sr0, sc0, sr1, sc1 = sprite_bbox
        br0, bc0, br1, bc1 = bar_bbox
        mr0, mc0, mr1, mc1 = motif_bbox
        tile_h = sr1 - sr0 + 1
        tile_w = sc1 - sc0 + 1

        assert tile_h in SPRITE_SIDE_OPTIONS and tile_w in SPRITE_SIDE_OPTIONS
        assert 2 <= int(motif_mask.sum()) <= 4
        assert motif_mask.shape[0] >= 2 and motif_mask.shape[1] >= 2
        assert np.count_nonzero(inp[sr0 : sr1 + 1, sc0 : sc1 + 1] == color) == tile_h * tile_w
        assert not np.any(occupied & _bbox_mask(inp.shape, sprite_bbox))
        assert not np.any(occupied & _bbox_mask(inp.shape, bar_bbox))
        assert not np.any(occupied & _bbox_mask(inp.shape, motif_bbox))
        occupied |= _bbox_mask(inp.shape, sprite_bbox)
        occupied |= _bbox_mask(inp.shape, bar_bbox)
        occupied |= _bbox_mask(inp.shape, motif_bbox)

        if direction in {"up", "down"}:
            assert (bc0, bc1) == (sc0, sc1)
            gap = (sr0 - br0 - 1) if direction == "up" else (br0 - sr1 - 1)
        else:
            assert (br0, br1) == (sr0, sr1)
            gap = (sc0 - bc0 - 1) if direction == "left" else (bc0 - sc1 - 1)
        assert BAR_GAP_RANGE[0] <= gap <= BAR_GAP_RANGE[1]

        solid_unit = np.full((tile_h, tile_w), color, dtype=int)
        carved_unit = solid_unit.copy()
        off_r, off_c = _hole_offsets(tile_h, tile_w, motif_mask.shape[0], motif_mask.shape[1], direction)
        for rr, cc in np.argwhere(motif_mask):
            carved_unit[off_r + int(rr), off_c + int(cc)] = BACKGROUND
        assert np.any(carved_unit != solid_unit)

        anchors = _tile_anchors(sprite_bbox, bar_bbox)[1]
        assert len(anchors) >= 2
        for ar, ac in anchors:
            bbox = (ar, ac, ar + tile_h - 1, ac + tile_w - 1)
            mask = _bbox_mask(inp.shape, bbox)
            assert not np.any(tiled & mask)
            tiled |= mask

    solved = main(inp)
    assert np.array_equal(solved, expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        h = random.randint(*GRID_SIZE_RANGE)
        w = random.randint(*GRID_SIZE_RANGE)
        n_groups = random.choice(OBJECT_COUNT_OPTIONS)
        colors = random.sample(PALETTE, n_groups)
        reserved = np.zeros((h, w), dtype=bool)
        groups: list[dict] = []

        ok = True
        for color in colors:
            spec = _sample_group((h, w), reserved)
            if spec is None:
                ok = False
                break
            spec["color"] = int(color)
            groups.append(spec)
            reserved |= _bbox_mask((h, w), spec["sprite_bbox"], pad=1)
            reserved |= _bbox_mask((h, w), spec["bar_bbox"], pad=1)
            reserved |= _bbox_mask((h, w), spec["output_bbox"], pad=1)
        if not ok:
            continue

        if not _place_motifs((h, w), groups, reserved):
            continue

        inp = _render_input((h, w), groups)
        expected = _expected_from_specs((h, w), groups)
        try:
            _validate_specs(inp, expected, groups)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a valid input")
