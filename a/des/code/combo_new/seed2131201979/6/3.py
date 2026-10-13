import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = ["top", "bottom", "left", "right"]
CORNERS = ["tl", "tr", "bl", "br"]
GAP_POSITIONS = {
    "top": (0, 2),
    "bottom": (4, 2),
    "left": (2, 0),
    "right": (2, 4),
}
CORNER_POSITIONS = {
    "tl": (0, 0),
    "tr": (0, 4),
    "bl": (4, 0),
    "br": (4, 4),
}
SHAPES = {
    "dot": np.array([[1]], dtype=bool),
    "h2": np.array([[1, 1]], dtype=bool),
    "v2": np.array([[1], [1]], dtype=bool),
    "h3": np.array([[1, 1, 1]], dtype=bool),
    "v3": np.array([[1], [1], [1]], dtype=bool),
    "l_tl": np.array([[1, 1], [1, 0]], dtype=bool),
    "l_tr": np.array([[1, 1], [0, 1]], dtype=bool),
    "l_bl": np.array([[1, 0], [1, 1]], dtype=bool),
    "l_br": np.array([[0, 1], [1, 1]], dtype=bool),
}
GEN_ATTEMPTS = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(pts)
    return parts


def _coords_to_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _scan_containers(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    border = [(r, c) for r in range(5) for c in range(5) if r in (0, 4) or c in (0, 4)]
    found: list[dict] = []
    for r0 in range(h - 4):
        for c0 in range(w - 4):
            patch = grid[r0 : r0 + 5, c0 : c0 + 5]
            if not np.all(patch[1:4, 1:4] == bg):
                continue

            bg_border = [(r, c) for r, c in border if int(patch[r, c]) == bg]
            if len(bg_border) != 1:
                continue
            gap_rel = bg_border[0]
            gap_side = next((name for name, pos in GAP_POSITIONS.items() if pos == gap_rel), None)
            if gap_side is None:
                continue

            counts = Counter(int(patch[r, c]) for r, c in border if int(patch[r, c]) != bg)
            if sorted(counts.values()) != [1, 14]:
                continue
            marker_color = next(color for color, count in counts.items() if count == 1)
            frame_color = next(color for color, count in counts.items() if count == 14)
            marker_positions = [(r, c) for r, c in border if int(patch[r, c]) == marker_color]
            if len(marker_positions) != 1:
                continue
            marker_rel = marker_positions[0]
            marker_corner = next((name for name, pos in CORNER_POSITIONS.items() if pos == marker_rel), None)
            if marker_corner is None:
                continue

            ok = True
            for rr, cc in border:
                val = int(patch[rr, cc])
                if (rr, cc) == gap_rel:
                    ok &= val == bg
                elif (rr, cc) == marker_rel:
                    ok &= val == marker_color
                else:
                    ok &= val == frame_color
            if not ok:
                continue

            found.append(
                {
                    "top": int(r0),
                    "left": int(c0),
                    "gap": gap_side,
                    "corner": marker_corner,
                    "marker_color": int(marker_color),
                    "frame_color": int(frame_color),
                }
            )
    found.sort(key=lambda item: (item["top"], item["left"]))
    return found


def _center_start(piece_span: int, prefer_low: bool) -> int:
    slack = 3 - piece_span
    assert 0 <= slack <= 2
    if slack % 2 == 0:
        return slack // 2
    return 0 if prefer_low else 1


def _paint_piece(out: np.ndarray, container: dict, piece_mask: np.ndarray, color: int) -> None:
    piece_h, piece_w = piece_mask.shape
    if container["gap"] in ("top", "bottom"):
        top = 3 - piece_h if container["gap"] == "top" else 0
        prefer_low = container["corner"] in ("tl", "bl")
        left = _center_start(piece_w, prefer_low)
    else:
        left = 3 - piece_w if container["gap"] == "left" else 0
        prefer_low = container["corner"] in ("tl", "tr")
        top = _center_start(piece_h, prefer_low)

    base_r = container["top"] + 1 + top
    base_c = container["left"] + 1 + left
    for rr in range(piece_h):
        for cc in range(piece_w):
            if piece_mask[rr, cc]:
                out[base_r + rr, base_c + cc] = int(color)


def _apply_rule(grid: np.ndarray, containers: list[dict], pieces: dict[int, np.ndarray], bg: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    inside_any_container = np.zeros_like(out, dtype=bool)
    for container in containers:
        r0, c0 = container["top"], container["left"]
        inside_any_container[r0 : r0 + 5, c0 : c0 + 5] = True

    marker_colors = {container["marker_color"] for container in containers}
    for color in marker_colors:
        out[(grid == int(color)) & (~inside_any_container)] = bg

    for container in containers:
        _paint_piece(out, container, pieces[container["marker_color"]], container["marker_color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move each colored fragment into the container whose corner marker has the same color.

    Input structure:
    - The background is black.
    - Several 5x5 square containers appear as one-pixel-thick grey frames with a 3x3 empty interior.
    - Each container has exactly one missing border pixel at the midpoint of one side, and exactly one colored corner marker whose color is unique across containers.
    - Outside the containers, each marker color appears as one small 4-connected fragment of 1 to 3 same-colored pixels.

    Rule:
    - Match every fragment to the container whose colored corner uses the same color.
    - Use the frame gap to determine the slide axis: top gap slides downward, bottom upward, left rightward, and right leftward.
    - Place the fragment inside the 3x3 interior without rotation, translated until its farthest pixel touches the inner wall opposite the gap.
    - Center the fragment on the perpendicular axis; when exact centering is impossible, bias toward the side indicated by the marker corner. Remove the original scattered fragment.

    Key insight:
    - Color identifies the destination container, the gap identifies which interior wall the fragment must press against, and the marker corner resolves the only remaining centering ambiguity.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    containers = _scan_containers(g, bg)
    assert containers, "expected at least one valid container"

    marker_colors = [container["marker_color"] for container in containers]
    assert len(marker_colors) == len(set(marker_colors)), "marker colors must be unique"

    container_mask = np.zeros_like(g, dtype=bool)
    for container in containers:
        r0, c0 = container["top"], container["left"]
        container_mask[r0 : r0 + 5, c0 : c0 + 5] = True

    outside_colors = set(map(int, np.unique(g[~container_mask]))) - {bg}
    assert outside_colors == set(marker_colors), "outside colors must match container marker colors"

    pieces: dict[int, np.ndarray] = {}
    for color in marker_colors:
        comps = _components((g == int(color)) & (~container_mask))
        assert len(comps) == 1, "each marker color should have exactly one fragment"
        assert 1 <= len(comps[0]) <= 3, "fragment size must be between 1 and 3 pixels"
        pieces[int(color)] = _coords_to_mask(comps[0])
        assert pieces[int(color)].shape[0] <= 3 and pieces[int(color)].shape[1] <= 3

    return _apply_rule(g, containers, pieces, bg)


def _reserve(mask: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = mask.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    mask[r0:r1, c0:c1] = True


def _free_positions(
    occupied: np.ndarray, height: int, width: int, pad: int = 1, edge_margin: int = 1
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    spots: list[tuple[int, int]] = []
    for top in range(edge_margin, h - height - edge_margin + 1):
        for left in range(edge_margin, w - width - edge_margin + 1):
            r0 = max(0, top - pad)
            c0 = max(0, left - pad)
            r1 = min(h, top + height + pad)
            c1 = min(w, left + width + pad)
            if not occupied[r0:r1, c0:c1].any():
                spots.append((top, left))
    return spots


def _choose_shape(rng: random.Random, direction: str, force_ambiguous: bool | None = None) -> np.ndarray:
    if direction in ("top", "bottom"):
        ambiguous = ["h2", "l_tl", "l_tr", "l_bl", "l_br"]
        exact = ["dot", "v2", "v3", "h3"]
    else:
        ambiguous = ["v2", "l_tl", "l_tr", "l_bl", "l_br"]
        exact = ["dot", "h2", "h3", "v3"]
    if force_ambiguous is None:
        force_ambiguous = rng.random() < 0.5
    pool = ambiguous if force_ambiguous else exact
    return SHAPES[rng.choice(pool)].copy()


def _render_container(grid: np.ndarray, top: int, left: int, frame_color: int, marker_color: int, gap: str, corner: str) -> None:
    for rr in range(5):
        for cc in range(5):
            if rr in (0, 4) or cc in (0, 4):
                grid[top + rr, left + cc] = frame_color
    gap_r, gap_c = GAP_POSITIONS[gap]
    grid[top + gap_r, left + gap_c] = BG
    marker_r, marker_c = CORNER_POSITIONS[corner]
    grid[top + marker_r, left + marker_c] = marker_color
    grid[top + 1 : top + 4, left + 1 : left + 4] = BG


def _render_piece(grid: np.ndarray, top: int, left: int, piece_mask: np.ndarray, color: int) -> None:
    rows, cols = piece_mask.shape
    for rr in range(rows):
        for cc in range(cols):
            if piece_mask[rr, cc]:
                grid[top + rr, left + cc] = int(color)


def _manual_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for spec in specs:
        piece = spec["piece"]
        top, left = spec["piece_top"], spec["piece_left"]
        for rr in range(piece.shape[0]):
            for cc in range(piece.shape[1]):
                if piece[rr, cc]:
                    out[top + rr, left + cc] = BG
        _paint_piece(out, spec["container"], piece, spec["container"]["marker_color"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_containers = rng.randint(2, 4)
    min_side = {2: 16, 3: 20, 4: 23}[n_containers]
    height = rng.randint(min_side, min(30, min_side + 5))
    width = rng.randint(min_side, min(30, min_side + 5))
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    colors = rng.sample(PALETTE, n_containers)
    if n_containers == 4 and rng.random() < 0.5:
        directions = rng.sample(DIRS, 4)
    else:
        while True:
            directions = [rng.choice(DIRS) for _ in range(n_containers)]
            if len(set(directions)) >= 2:
                break
    corners = [rng.choice(CORNERS) for _ in range(n_containers)]

    ambiguous_index = rng.randrange(n_containers)
    exact_index = rng.randrange(n_containers)
    if exact_index == ambiguous_index and n_containers > 1:
        exact_index = (exact_index + 1) % n_containers

    specs: list[dict] = []
    for idx in range(n_containers):
        spots = _free_positions(occupied, 5, 5, pad=1, edge_margin=1)
        if not spots:
            return None
        top, left = rng.choice(spots)
        _reserve(occupied, top, left, 5, 5, pad=1)

        piece = _choose_shape(
            rng,
            directions[idx],
            force_ambiguous=True if idx == ambiguous_index else False if idx == exact_index else None,
        )
        specs.append(
            {
                "container": {
                    "top": int(top),
                    "left": int(left),
                    "gap": directions[idx],
                    "corner": corners[idx],
                    "marker_color": int(colors[idx]),
                    "frame_color": FRAME,
                },
                "piece": piece,
            }
        )

    for spec in specs:
        piece = spec["piece"]
        spots = _free_positions(occupied, piece.shape[0], piece.shape[1], pad=1, edge_margin=1)
        if not spots:
            return None
        top, left = rng.choice(spots)
        _reserve(occupied, top, left, piece.shape[0], piece.shape[1], pad=1)
        spec["piece_top"] = int(top)
        spec["piece_left"] = int(left)

    rng.shuffle(specs)
    for spec in specs:
        container = spec["container"]
        _render_container(
            grid,
            container["top"],
            container["left"],
            container["frame_color"],
            container["marker_color"],
            container["gap"],
            container["corner"],
        )
        _render_piece(grid, spec["piece_top"], spec["piece_left"], spec["piece"], container["marker_color"])

    expected = _manual_expected(grid, specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt(random.Random())
        if result is not None:
            return result
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
