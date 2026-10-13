from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BLUE = 1
GREEN = 3
GREY = 5
ORANGE = 7

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

BASE_TEMPLATES = [
    [
        ".O.M.",
        "MMMEM",
        "GGGGG",
        "MMM.M",
        ".M...",
    ],
    [
        "O.MM.",
        "MMEMM",
        "GGGGG",
        "MMEM.",
        ".MM..",
    ],
    [
        ".M.O.",
        "MEMMM",
        "GGGGG",
        ".MMEM",
        "...M.",
    ],
]

GEN_ATTEMPTS = 500


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def parse_template(rows: list[str], main_color: int) -> np.ndarray:
    mapping = {".": BG, "M": main_color, "G": GREY, "O": ORANGE, "E": ORANGE}
    patch = np.array([[mapping[ch] for ch in row] for row in rows], dtype=int)
    return patch


def build_patch(main_color: int, *, vertical_separator: bool) -> np.ndarray:
    rows = random.choice(BASE_TEMPLATES)
    patch = parse_template(rows, main_color)
    if vertical_separator:
        patch = patch.T
    if random.random() < 0.5:
        patch = np.flipud(patch)
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    return patch.copy()


def find_main_color(patch: np.ndarray) -> int:
    colors = [int(v) for v in patch.flatten() if int(v) not in {BG, GREY, ORANGE}]
    assert colors
    return Counter(colors).most_common(1)[0][0]


def marker_kind(patch: np.ndarray, r: int, c: int, main_color: int) -> str:
    main_neighbors = 0
    bg_neighbors = 0
    for nr, nc in neighbors4(r, c):
        if in_bounds(patch.shape, nr, nc):
            val = int(patch[nr, nc])
        else:
            val = BG
        if val == main_color:
            main_neighbors += 1
        if val == BG:
            bg_neighbors += 1
    if main_neighbors >= 3:
        return "embedded"
    assert bg_neighbors >= 3
    return "isolated"


def find_separator(patch: np.ndarray) -> tuple[str, int]:
    row_axes = [r for r in range(patch.shape[0]) if np.all(patch[r, :] == GREY)]
    col_axes = [c for c in range(patch.shape[1]) if np.all(patch[:, c] == GREY)]
    if row_axes and not col_axes:
        return "horizontal", row_axes[0]
    if col_axes and not row_axes:
        return "vertical", col_axes[0]
    raise AssertionError("expected exactly one full internal separator line")


def mirror_patch(patch: np.ndarray, orient: str, axis: int) -> np.ndarray:
    h, w = patch.shape
    out = np.full_like(patch, BG)
    if orient == "horizontal":
        for r in range(h):
            src_r = 2 * axis - r
            if 0 <= src_r < h:
                out[r, :] = patch[src_r, :]
    else:
        for c in range(w):
            src_c = 2 * axis - c
            if 0 <= src_c < w:
                out[:, c] = patch[:, src_c]
    return out


def repair_patch(patch: np.ndarray, main_color: int) -> np.ndarray:
    out = patch.copy()
    for r, c in np.argwhere(patch == ORANGE):
        if marker_kind(patch, int(r), int(c), main_color) == "embedded":
            out[int(r), int(c)] = main_color
    return out


def find_isolated_marker(patch: np.ndarray, main_color: int) -> tuple[int, int]:
    markers = [(int(r), int(c)) for r, c in np.argwhere(patch == ORANGE)]
    isolated = [pos for pos in markers if marker_kind(patch, pos[0], pos[1], main_color) == "isolated"]
    assert len(isolated) == 1, f"expected exactly one isolated marker, got {isolated}"
    return isolated[0]


def extract_sprite_info(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    main_color = find_main_color(patch)
    orient, axis = find_separator(patch)
    return {
        "bbox": (r0, c0, r1, c1),
        "coords": coords,
        "patch": patch,
        "main_color": main_color,
        "orient": orient,
        "axis": axis,
    }


def find_anchors(grid: np.ndarray) -> dict[int, tuple[int, int]]:
    h, w = grid.shape
    anchors: dict[int, tuple[int, int]] = {}
    for r in range(h):
        for c in range(w):
            if r not in {0, h - 1} and c not in {0, w - 1}:
                continue
            val = int(grid[r, c])
            if val == BG:
                continue
            if any(in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != BG for nr, nc in neighbors4(r, c)):
                continue
            anchors[val] = (r, c)
    return anchors


def solve_with_options(
    grid: np.ndarray,
    *,
    do_repair: bool,
    do_mirror: bool,
    strict: bool = True,
) -> np.ndarray | None:
    g = np.array(grid, dtype=int)
    anchors = find_anchors(g)
    comps = find_components(g != BG)
    sprite_infos = []
    for comp in comps:
        vals = {int(g[r, c]) for r, c in comp}
        if GREY in vals and ORANGE in vals and any(v in {BLUE, GREEN} for v in vals):
            sprite_infos.append(extract_sprite_info(g, comp))
    assert len(sprite_infos) == 2

    out = g.copy()
    for info in sprite_infos:
        for r, c in info["coords"]:
            out[r, c] = BG

    for info in sprite_infos:
        patch = info["patch"]
        main_color = info["main_color"]
        work = repair_patch(patch, main_color) if do_repair else patch.copy()
        transformed = mirror_patch(work, info["orient"], info["axis"]) if do_mirror else work
        mr, mc = find_isolated_marker(transformed, main_color)
        r0, c0, r1, c1 = info["bbox"]
        ar, ac = anchors[main_color]
        if ar in {0, g.shape[0] - 1}:
            dr = 0
            dc = ac - (c0 + mc)
        else:
            dr = ar - (r0 + mr)
            dc = 0
        nr0 = r0 + dr
        nc0 = c0 + dc
        h, w = transformed.shape
        if not (0 <= nr0 and nr0 + h <= g.shape[0] and 0 <= nc0 and nc0 + w <= g.shape[1]):
            if strict:
                raise AssertionError("translated sprite must stay in bounds")
            return None
        for rr in range(h):
            for cc in range(w):
                val = int(transformed[rr, cc])
                if val != BG:
                    out[nr0 + rr, nc0 + cc] = val
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair embedded orange defects, mirror each sprite across its internal grey separator, then slide it to its same-colored border anchor.

    Input structure:
    - A black background contains exactly two interior multi-color sprite components and two isolated border anchors.
    - Each sprite uses one main color (blue or green), some orange marker pixels, and a full internal grey separator row or column.
    - Orange markers are either embedded inside the main-colored body or isolated protrusions attached by one 4-neighbor.

    Rule:
    - For each sprite, identify its majority non-black, non-grey, non-orange color as the main color.
    - Recolor every orange marker with at least three 4-neighbor main-color cells into the main color.
    - Mirror the repaired sprite across its full grey separator line, keeping the separator fixed and moving the remaining isolated marker with its side.
    - Find the border anchor of the same color: top/bottom anchors force horizontal sliding to align marker columns, left/right anchors force vertical sliding to align marker rows.
    - Clear the original sprite location to black and place the transformed sprite at its slid position; anchors stay unchanged.

    Key insight:
    - The orange pixels must first be split into embedded defects versus true protrusion markers; only after repair does the grey separator define the correct local mirror, and the anchor edge then selects which marker coordinate controls the final slide.
    """
    out = solve_with_options(grid, do_repair=True, do_mirror=True, strict=True)
    assert out is not None
    return out


def rects_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def swept_box(start: tuple[int, int], end: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    sr, sc = start
    er, ec = end
    h, w = shape
    return min(sr, er), min(sc, ec), max(sr, er) + h - 1, max(sc, ec) + w - 1


def place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w][patch != BG] = patch[patch != BG]


def build_expected(
    shape: tuple[int, int],
    sprites: dict[int, dict],
    anchors: dict[int, tuple[int, int]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for color, (r, c) in anchors.items():
        out[r, c] = color
    for color, info in sprites.items():
        top, left = info["final_tl"]
        place_patch(out, info["mirrored_patch"], top, left)
    return out


def sample_layout(shape: tuple[int, int], sprites: dict[int, dict], motion_by_color: dict[int, str]):
    h, w = shape
    patch_h, patch_w = next(iter(sprites.values()))["input_patch"].shape
    margin = 2
    final_rows = list(range(margin, h - patch_h - margin + 1))
    final_cols = list(range(margin, w - patch_w - margin + 1))
    for _ in range(300):
        final_tl: dict[int, tuple[int, int]] = {}
        anchors: dict[int, tuple[int, int]] = {}
        ok = True
        colors = [BLUE, GREEN]
        random.shuffle(colors)
        for color in colors:
            info = sprites[color]
            mr, mc = info["post_marker"]
            candidates = [(r, c) for r in final_rows for c in final_cols]
            random.shuffle(candidates)
            placed = False
            for r0, c0 in candidates:
                box = (r0, c0, r0 + patch_h - 1, c0 + patch_w - 1)
                if any(rects_intersect(box, (rr, cc, rr + patch_h - 1, cc + patch_w - 1), pad=1) for rr, cc in final_tl.values()):
                    continue
                if motion_by_color[color] == "horizontal":
                    if not (1 <= c0 + mc <= w - 2):
                        continue
                    edge_row = random.choice([0, h - 1])
                    anchor = (edge_row, c0 + mc)
                else:
                    if not (1 <= r0 + mr <= h - 2):
                        continue
                    edge_col = random.choice([0, w - 1])
                    anchor = (r0 + mr, edge_col)
                if anchor in anchors.values():
                    continue
                final_tl[color] = (r0, c0)
                anchors[color] = anchor
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        deltas_by_color: dict[int, int] = {}
        input_tl: dict[int, tuple[int, int]] = {}
        delta_choices = [d for d in range(-6, 7) if abs(d) >= 2]
        random.shuffle(colors)
        for color in colors:
            fr, fc = final_tl[color]
            motion = motion_by_color[color]
            choices = delta_choices[:]
            random.shuffle(choices)
            placed = False
            for delta in choices:
                if motion == "horizontal":
                    ir, ic = fr, fc + delta
                else:
                    ir, ic = fr + delta, fc
                if not (margin <= ir <= h - patch_h - margin and margin <= ic <= w - patch_w - margin):
                    continue
                box = (ir, ic, ir + patch_h - 1, ic + patch_w - 1)
                if any(rects_intersect(box, (rr, cc, rr + patch_h - 1, cc + patch_w - 1), pad=1) for other, (rr, cc) in input_tl.items() if other != color):
                    continue
                input_tl[color] = (ir, ic)
                deltas_by_color[color] = delta
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        sweep_boxes = {
            color: swept_box(input_tl[color], final_tl[color], (patch_h, patch_w))
            for color in sprites
        }
        if rects_intersect(sweep_boxes[BLUE], sweep_boxes[GREEN], pad=0):
            continue

        return final_tl, input_tl, anchors
    return None


def gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 24)
    width = random.randint(18, 24)
    motion_by_color = {}
    if random.random() < 0.5:
        motion_by_color[BLUE] = "horizontal"
        motion_by_color[GREEN] = "vertical"
    else:
        motion_by_color[BLUE] = "vertical"
        motion_by_color[GREEN] = "horizontal"

    sprites: dict[int, dict] = {}
    for color in (BLUE, GREEN):
        vertical_separator = motion_by_color[color] == "horizontal"
        input_patch = build_patch(color, vertical_separator=vertical_separator)
        main_color = find_main_color(input_patch)
        orient, axis = find_separator(input_patch)
        repaired = repair_patch(input_patch, main_color)
        mirrored = mirror_patch(repaired, orient, axis)
        pre_marker = find_isolated_marker(repaired, main_color)
        post_marker = find_isolated_marker(mirrored, main_color)
        if motion_by_color[color] == "horizontal" and pre_marker[1] == post_marker[1]:
            return None
        if motion_by_color[color] == "vertical" and pre_marker[0] == post_marker[0]:
            return None
        if np.array_equal(mirrored, mirror_patch(input_patch, orient, axis)):
            return None
        sprites[color] = {
            "input_patch": input_patch,
            "repaired_patch": repaired,
            "mirrored_patch": mirrored,
            "pre_marker": pre_marker,
            "post_marker": post_marker,
            "orient": orient,
        }

    layout = sample_layout((height, width), sprites, motion_by_color)
    if layout is None:
        return None
    final_tl, input_tl, anchors = layout

    for color in sprites:
        sprites[color]["final_tl"] = final_tl[color]
        sprites[color]["input_tl"] = input_tl[color]

    grid = np.zeros((height, width), dtype=int)
    for color, (r, c) in anchors.items():
        grid[r, c] = color
    for color, info in sprites.items():
        place_patch(grid, info["input_patch"], info["input_tl"][0], info["input_tl"][1])

    expected = build_expected(grid.shape, sprites, anchors)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None

    no_repair = solve_with_options(grid, do_repair=False, do_mirror=True, strict=False)
    if no_repair is not None and np.array_equal(no_repair, expected):
        return None

    no_mirror = solve_with_options(grid, do_repair=True, do_mirror=False, strict=False)
    if no_mirror is not None and np.array_equal(no_mirror, expected):
        return None

    anchors_found = find_anchors(grid)
    if anchors_found.keys() != {BLUE, GREEN}:
        return None

    comps = find_components(grid != BG)
    sprite_count = 0
    for comp in comps:
        vals = {int(grid[r, c]) for r, c in comp}
        if GREY in vals and ORANGE in vals:
            sprite_count += 1
    if sprite_count != 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
