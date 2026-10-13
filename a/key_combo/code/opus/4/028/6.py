from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox, crop_bbox, find_components, get_bg, paste, random_free_location


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_SIDE_BAG: list[str] = []
_RECENT_OUTPUT_SIGS: list[str] = []


def _detect_marker_side(panel: np.ndarray, panel_color: int) -> tuple[str, int]:
    sides = {
        "top": panel[0, :],
        "bottom": panel[-1, :],
        "left": panel[:, 0],
        "right": panel[:, -1],
    }
    hits = []
    for side, line in sides.items():
        vals = np.unique(line)
        if len(vals) == 1 and int(vals[0]) != int(panel_color):
            hits.append((side, int(vals[0])))
    if len(hits) != 1:
        raise ValueError("expected exactly one marked border side")
    return hits[0]


def _extract_pattern(panel: np.ndarray, panel_color: int) -> np.ndarray:
    inner = panel[1:-1, 1:-1]
    pattern_mask = inner != int(panel_color)
    bb = bbox(pattern_mask)
    if bb is None:
        raise ValueError("panel is missing an interior pattern")
    return crop_bbox(inner, bb)


def _build_piece(pattern: np.ndarray, side: str, marker_color: int) -> np.ndarray:
    if side in ("top", "bottom"):
        mirrored = np.flipud(pattern)
        seam = np.full((1, pattern.shape[1]), int(marker_color), dtype=int)
        if side == "top":
            return np.vstack([mirrored, seam, pattern])
        return np.vstack([pattern, seam, mirrored])

    mirrored = np.fliplr(pattern)
    seam = np.full((pattern.shape[0], 1), int(marker_color), dtype=int)
    if side == "left":
        return np.hstack([mirrored, seam, pattern])
    return np.hstack([pattern, seam, mirrored])


def _assemble_output(items: list[tuple[tuple[int, int, int, int], np.ndarray]]) -> np.ndarray:
    ordered = sorted(items, key=lambda item: (int(item[0][1]), int(item[0][0])))
    out_h = max(int(piece.shape[0]) for _bbox0, piece in ordered)
    out_w = sum(int(piece.shape[1]) for _bbox0, piece in ordered) + len(ordered) - 1
    out = np.full((out_h, out_w), BG, dtype=int)
    c0 = 0
    for _bbox0, piece in ordered:
        paste(out, piece, 0, c0)
        c0 += int(piece.shape[1]) + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each panel's marked edge to choose the mirror axis, stitch direction, and seam color.

    Input structure:
    - Black background contains 2-3 separated multicolor rectangular panels.
    - Each panel is a solid rectangle of one dominant color, with exactly one full border side replaced by a marker color.
    - Inside each panel is a small non-symmetric 4-connected pattern whose colors differ from the panel fill and the marker strip.

    Rule:
    - Find each panel as a non-black multicolor connected component whose bbox is fully filled.
    - The unique fully uniform non-panel border side is the marker: top/bottom means mirror across a horizontal axis (vertical flip), left/right means mirror across a vertical axis (horizontal flip).
    - Crop the interior pattern to its minimal bbox on the panel-color background, mirror it using that axis, and place the mirrored copy on the marked side with a 1-pixel seam colored by the marker.
    - Sort panel results by the input panel top-left position from left to right, place them on black with 1 black column between them, and top-align shorter results.

    Key insight:
    - A single side marker simultaneously encodes mirror axis, which side receives the mirrored copy, and the seam color, so the marked border must be read geometrically rather than as a repaint instruction.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    pieces: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    for comp in comps:
        bb = tuple(int(x) for x in comp["bbox"])
        panel = crop_bbox(g, bb)
        if panel.shape[0] < 5 or panel.shape[1] < 5:
            continue
        if np.any(panel == bg):
            continue

        counts = np.bincount(panel.ravel(), minlength=10)
        panel_color = int(np.argmax(counts))
        side, marker_color = _detect_marker_side(panel, panel_color)
        pattern = _extract_pattern(panel, panel_color)
        pieces.append((bb, _build_piece(pattern, side, marker_color)))

    if not pieces:
        raise ValueError("no valid panels found")
    return _assemble_output(pieces)


def _next_sides(count: int) -> list[str]:
    global _SIDE_BAG
    chosen: list[str] = []
    while len(chosen) < count:
        if not _SIDE_BAG:
            _SIDE_BAG = list(SIDES)
            random.shuffle(_SIDE_BAG)
        side = _SIDE_BAG.pop()
        if side not in chosen:
            chosen.append(side)
    return chosen


def _grow_connected_cells(h: int, w: int, n_cells: int) -> list[tuple[int, int]]:
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < n_cells:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return []
        cells.add(random.choice(frontier))
    return sorted(cells)


def _make_pattern_crop(panel_color: int, marker_color: int, interior_h: int, interior_w: int) -> np.ndarray:
    usable_colors = [c for c in PALETTE if c not in {panel_color, marker_color}]
    max_h = min(4, interior_h - 1)
    max_w = min(4, interior_w - 1)
    if max_h < 2 or max_w < 2:
        raise ValueError("interior is too small for a cropped asymmetric pattern")

    for _ in range(300):
        base_h = random.randint(2, max_h)
        base_w = random.randint(2, max_w)
        n_cells = random.randint(3, min(5, base_h * base_w))
        cells = _grow_connected_cells(base_h, base_w, n_cells)
        if not cells:
            continue

        patch = np.full((base_h, base_w), int(panel_color), dtype=int)
        n_cols = random.randint(2, min(3, n_cells, len(usable_colors)))
        colors = random.sample(usable_colors, n_cols)
        assigned = colors[:] + random.choices(colors, k=n_cells - n_cols)
        random.shuffle(assigned)
        for (r, c), color in zip(cells, assigned):
            patch[r, c] = int(color)

        bb = bbox(patch != int(panel_color))
        if bb is None:
            continue
        crop = crop_bbox(patch, bb)
        if crop.shape[0] >= interior_h or crop.shape[1] >= interior_w:
            continue
        if len({int(v) for v in crop.ravel() if int(v) != int(panel_color)}) < 2:
            continue
        if np.array_equal(crop, np.flipud(crop)):
            continue
        if np.array_equal(crop, np.fliplr(crop)):
            continue
        return crop

    raise RuntimeError("failed to build a non-symmetric pattern crop")


def _make_panel(side: str, panel_color: int, marker_color: int) -> dict:
    for _ in range(200):
        h = random.randint(5, 7)
        w = random.randint(5, 7)
        interior_h, interior_w = h - 2, w - 2
        crop = _make_pattern_crop(panel_color, marker_color, interior_h, interior_w)

        panel = np.full((h, w), int(panel_color), dtype=int)
        max_r = interior_h - crop.shape[0]
        max_c = interior_w - crop.shape[1]
        off_r = random.randint(0, max_r)
        off_c = random.randint(0, max_c)
        paste(panel, crop, 1 + off_r, 1 + off_c)

        if side == "top":
            panel[0, :] = int(marker_color)
        elif side == "bottom":
            panel[-1, :] = int(marker_color)
        elif side == "left":
            panel[:, 0] = int(marker_color)
        else:
            panel[:, -1] = int(marker_color)

        return {
            "panel": panel,
            "side": side,
            "panel_color": int(panel_color),
            "marker_color": int(marker_color),
            "crop": crop,
        }
    raise RuntimeError("failed to build panel")


def _register_output_signature(out: np.ndarray) -> bool:
    sig = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
    if sig in _RECENT_OUTPUT_SIGS:
        return False
    _RECENT_OUTPUT_SIGS.append(sig)
    if len(_RECENT_OUTPUT_SIGS) > 256:
        del _RECENT_OUTPUT_SIGS[0]
    return True


def _validate_instance(grid: np.ndarray, specs: list[dict]) -> bool:
    if len(specs) not in (2, 3):
        return False
    if len({spec["side"] for spec in specs}) != len(specs):
        return False

    pieces = []
    piece_sigs = set()
    for spec in specs:
        panel = spec["panel"]
        crop = spec["crop"]
        side = spec["side"]
        marker_color = spec["marker_color"]
        panel_color = spec["panel_color"]
        bbox0 = spec["bbox"]

        if bbox0[0] < 2 or bbox0[1] < 2:
            return False
        if bbox0[2] > grid.shape[0] - 3 or bbox0[3] > grid.shape[1] - 3:
            return False
        if marker_color == panel_color:
            return False
        if np.any(crop == marker_color):
            return False
        if crop.shape[0] >= panel.shape[0] - 2 or crop.shape[1] >= panel.shape[1] - 2:
            return False
        if np.array_equal(crop, np.flipud(crop)) or np.array_equal(crop, np.fliplr(crop)):
            return False

        piece = _build_piece(crop, side, marker_color)
        wrong_side = "left" if side in ("top", "bottom") else "top"
        if np.array_equal(piece, _build_piece(crop, wrong_side, marker_color)):
            return False
        piece_sig = hashlib.md5(piece.tobytes() + str(piece.shape).encode()).hexdigest()
        if piece_sig in piece_sigs:
            return False
        piece_sigs.add(piece_sig)
        pieces.append((bbox0, piece))

    expected = _assemble_output(pieces)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    return _register_output_signature(expected)


def _gen_attempt() -> np.ndarray | None:
    n_panels = random.choice([2, 3])
    sides = _next_sides(n_panels)
    random.shuffle(sides)

    colors = random.sample(PALETTE, 2 * n_panels)
    panel_colors = colors[:n_panels]
    marker_colors = colors[n_panels:]
    specs = [_make_panel(side, panel_color, marker_color) for side, panel_color, marker_color in zip(sides, panel_colors, marker_colors)]

    max_panel_h = max(int(spec["panel"].shape[0]) for spec in specs)
    min_h = max(16, max_panel_h + 6)
    max_h = min(26, min_h + 6)
    base_w = sum(int(spec["panel"].shape[1]) for spec in specs) + 2 * n_panels + 4
    min_w = max(16, min(29, base_w - 3))
    max_w = min(30, max(min_w, base_w + 1))
    if min_h > max_h or min_w > max_w:
        return None

    for _ in range(120):
        h = random.randint(min_h, max_h)
        w = random.randint(min_w, max_w)
        grid = np.full((h, w), BG, dtype=int)
        order = list(range(n_panels))
        order.sort(key=lambda idx: int(specs[idx]["panel"].size), reverse=True)

        placed = True
        for idx in order:
            spec = specs[idx]
            panel = spec["panel"]
            try:
                r0, c0 = random_free_location(grid, panel, transparent=BG, border=2, padding=1, padding_connectivity=8)
            except ValueError:
                placed = False
                break
            paste(grid, panel, r0, c0)
            spec["bbox"] = (int(r0), int(c0), int(r0 + panel.shape[0] - 1), int(c0 + panel.shape[1] - 1))
        if placed and _validate_instance(grid, specs):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
