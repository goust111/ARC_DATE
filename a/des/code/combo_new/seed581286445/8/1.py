from __future__ import annotations

import numpy as np


BG = 0
FRAME_COLOR = 5
FRAME_SIZE = 5
INNER_SIZE = 3
STAMP_SIZE = 2
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 300
CELL_SIZE_CHOICES = [12, 13]
FRAME_COUNT_CHOICES = [2, 3, 4]
CELL_ANCHORS = [(0, 0), (0, 1), (1, 0), (1, 1)]

MISSING_CORNERS = {
    "tl": ((0, 0), [(0, 1), (1, 0), (1, 1)]),
    "tr": ((0, 2), [(0, 1), (1, 1), (1, 2)]),
    "bl": ((2, 0), [(1, 0), (1, 1), (2, 1)]),
    "br": ((2, 2), [(1, 1), (1, 2), (2, 1)]),
}


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            pts = np.asarray(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            comps.append(
                {
                    "coords": pts,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return comps


def _frame_components(g: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        for comp in _components(g == color):
            r0, c0, r1, c1 = comp["bbox"]
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h != FRAME_SIZE or w != FRAME_SIZE or comp["area"] != 16:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if np.all(sub[border] == color) and not np.any(sub[1:-1, 1:-1] == color):
                frames.append({"color": color, "bbox": (r0, c0, r1, c1)})
    assert frames, "expected at least one frame"
    return frames


def _stamp_components(g: np.ndarray, frame_colors: set[int]) -> dict[int, list[dict]]:
    stamps_by_color: dict[int, list[dict]] = {}
    for color in np.unique(g):
        color = int(color)
        if color == BG or color in frame_colors:
            continue
        for comp in _components(g == color):
            r0, c0, r1, c1 = comp["bbox"]
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h != STAMP_SIZE or w != STAMP_SIZE or comp["area"] != 4:
                continue
            if np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color):
                stamps_by_color.setdefault(color, []).append({"bbox": (r0, c0, r1, c1)})
    return stamps_by_color


def _missing_corner(interior: np.ndarray, color: int) -> tuple[str, tuple[int, int]]:
    matches: list[tuple[str, tuple[int, int]]] = []
    for name, (missing, support) in MISSING_CORNERS.items():
        mr, mc = missing
        if interior[mr, mc] == color:
            continue
        if all(interior[r, c] == color for r, c in support):
            matches.append((name, (mr, mc)))
    assert len(matches) == 1, "inner shape must expose exactly one missing corner"
    return matches[0]


def _stamp_matches_quadrant(
    frame_bbox: tuple[int, int, int, int],
    stamp_bbox: tuple[int, int, int, int],
    corner_name: str,
) -> bool:
    fr0, fc0, fr1, fc1 = frame_bbox
    sr0, sc0, sr1, sc1 = stamp_bbox
    if corner_name == "tl":
        return sr1 < fr0 and sc1 < fc0
    if corner_name == "tr":
        return sr1 < fr0 and sc0 > fc1
    if corner_name == "bl":
        return sr0 > fr1 and sc1 < fc0
    assert corner_name == "br"
    return sr0 > fr1 and sc0 > fc1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed missing-corner block by consuming the same-colored outside stamp from the matching quadrant.

    Input structure:
    - The background is black.
    - Several grey square frames with 1-pixel borders enclose 3x3 interiors.
    - Inside each frame, one color forms an L-shaped corner cluster whose local 2x2 corner is missing exactly one interior corner cell.
    - Outside the frames, the same color appears as four separate 2x2 stamps placed in the four diagonal quadrants around that frame.

    Rule:
    - For each frame, detect which interior corner cell is the missing corner of the colored cluster.
    - Select the same-colored 2x2 stamp lying in the matching outside quadrant: top-left, top-right, bottom-left, or bottom-right.
    - Copy the selected stamp's top-left pixel color into the missing interior corner cell and erase that chosen stamp back to black.
    - Leave the frame, the other three same-colored stamps, and all unrelated pixels unchanged.

    Key insight:
    - The direction of the missing interior corner determines two things at once: where the hole is filled and which one of the four same-colored outside stamps must be consumed.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = sorted(_frame_components(g), key=lambda item: item["bbox"])
    frame_colors = {int(frame["color"]) for frame in frames}
    stamps_by_color = _stamp_components(g, frame_colors)

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        frame_color = int(frame["color"])
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        inner_colors = set(map(int, np.unique(interior))) - {BG, frame_color}
        assert len(inner_colors) == 1, "each frame must contain exactly one colored inner object"
        color = inner_colors.pop()

        corner_name, (mr, mc) = _missing_corner(interior, color)
        candidates = [
            stamp
            for stamp in stamps_by_color.get(color, [])
            if _stamp_matches_quadrant((r0, c0, r1, c1), stamp["bbox"], corner_name)
        ]
        assert len(candidates) == 1, "missing corner must choose a unique same-colored stamp"
        sr0, sc0, sr1, sc1 = candidates[0]["bbox"]
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = BG
        out[r0 + 1 + mr, c0 + 1 + mc] = int(g[sr0, sc0])

    return out


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(int(lo), int(hi) + 1))


def _place_frame_and_stamps(
    grid: np.ndarray,
    *,
    rng: np.random.Generator,
    cell_anchor: tuple[int, int],
    cell_size: int,
    color: int,
    corner_name: str,
) -> dict:
    cell_r, cell_c = cell_anchor
    base_r = cell_r * cell_size
    base_c = cell_c * cell_size

    local_r = _randint(rng, 3, cell_size - 8)
    local_c = _randint(rng, 3, cell_size - 8)
    r0 = base_r + local_r
    c0 = base_c + local_c

    grid[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE] = BG
    grid[r0, c0 : c0 + FRAME_SIZE] = FRAME_COLOR
    grid[r0 + FRAME_SIZE - 1, c0 : c0 + FRAME_SIZE] = FRAME_COLOR
    grid[r0 : r0 + FRAME_SIZE, c0] = FRAME_COLOR
    grid[r0 : r0 + FRAME_SIZE, c0 + FRAME_SIZE - 1] = FRAME_COLOR

    mr, mc = MISSING_CORNERS[corner_name][0]
    for rr, cc in MISSING_CORNERS[corner_name][1]:
        grid[r0 + 1 + rr, c0 + 1 + cc] = color

    tl_r = base_r + _randint(rng, 0, local_r - 3)
    tl_c = base_c + _randint(rng, 0, local_c - 3)
    tr_r = base_r + _randint(rng, 0, local_r - 3)
    tr_c = base_c + _randint(rng, local_c + 6, cell_size - 2)
    bl_r = base_r + _randint(rng, local_r + 6, cell_size - 2)
    bl_c = base_c + _randint(rng, 0, local_c - 3)
    br_r = base_r + _randint(rng, local_r + 6, cell_size - 2)
    br_c = base_c + _randint(rng, local_c + 6, cell_size - 2)

    stamp_positions = {
        "tl": (tl_r, tl_c),
        "tr": (tr_r, tr_c),
        "bl": (bl_r, bl_c),
        "br": (br_r, br_c),
    }
    for sr, sc in stamp_positions.values():
        grid[sr : sr + STAMP_SIZE, sc : sc + STAMP_SIZE] = color

    return {
        "frame_bbox": (r0, c0, r0 + FRAME_SIZE - 1, c0 + FRAME_SIZE - 1),
        "missing_cell": (r0 + 1 + mr, c0 + 1 + mc),
        "selected_stamp": stamp_positions[corner_name],
    }


def _crop_bounds(grid: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != BG)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = _randint(rng, 0, min(2, int(r0)))
    left = _randint(rng, 0, min(2, int(c0)))
    bottom = _randint(rng, 0, min(2, int(grid.shape[0] - 1 - r1)))
    right = _randint(rng, 0, min(2, int(grid.shape[1] - 1 - c1)))
    return int(r0 - top), int(r1 + bottom + 1), int(c0 - left), int(c1 + right + 1)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    cell_size = CELL_SIZE_CHOICES[_randint(rng, 0, len(CELL_SIZE_CHOICES) - 1)]
    frame_count = FRAME_COUNT_CHOICES[_randint(rng, 0, len(FRAME_COUNT_CHOICES) - 1)]

    colors = list(rng.choice(PALETTE, size=frame_count, replace=False))
    anchors = [CELL_ANCHORS[i] for i in rng.choice(len(CELL_ANCHORS), size=frame_count, replace=False)]
    corner_names = ["tl", "tr", "bl", "br"]
    rng.shuffle(corner_names)
    chosen_corners = corner_names[:frame_count]

    grid = np.full((2 * cell_size, 2 * cell_size), BG, dtype=int)
    placements = []
    for color, anchor, corner_name in zip(colors, anchors, chosen_corners):
        placements.append(
            _place_frame_and_stamps(
                grid,
                rng=rng,
                cell_anchor=anchor,
                cell_size=cell_size,
                color=int(color),
                corner_name=corner_name,
            )
        )

    expected = grid.copy()
    for placement in placements:
        sr, sc = placement["selected_stamp"]
        expected[sr : sr + STAMP_SIZE, sc : sc + STAMP_SIZE] = BG
        rr, cc = placement["missing_cell"]
        expected[rr, cc] = int(grid[sr, sc])

    rr0, rr1, cc0, cc1 = _crop_bounds(grid, rng)
    cropped_inp = grid[rr0:rr1, cc0:cc1]
    cropped_out = expected[rr0:rr1, cc0:cc1]
    if max(cropped_inp.shape) > 30:
        return None

    try:
        solved = main(cropped_inp)
    except Exception:
        return None
    if np.array_equal(solved, cropped_out):
        return cropped_inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_ATTEMPTS} attempts")
