import hashlib
import random
from collections import Counter, deque

import numpy as np


BLACK = 0
RED = 2
GREY = 5
MARKER_COLORS = (3, 4, 6, 8)

LOCAL_TO_CORNER = {
    frozenset({(0, 0), (0, 1), (1, 0)}): "top_left",
    frozenset({(0, 0), (0, 1), (1, 1)}): "top_right",
    frozenset({(0, 0), (1, 0), (1, 1)}): "bottom_left",
    frozenset({(0, 1), (1, 0), (1, 1)}): "bottom_right",
}
CORNER_TO_LOCAL = {corner: local for local, corner in LOCAL_TO_CORNER.items()}
CORNER_ORDER = ("top_left", "top_right", "bottom_left", "bottom_right")
CORNER_TO_DIAGONAL_SLOT = {
    "top_left": "nw",
    "top_right": "ne",
    "bottom_left": "sw",
    "bottom_right": "se",
}

_SEEN_OUTPUT_SIGS: set[str] = set()


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != int(color):
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _extract_sprite(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    grey_components = _components_of_color(grid, GREY)
    assert len(grey_components) == 1, "expected exactly one grey sprite"
    coords = grey_components[0]
    assert 5 <= len(coords) <= 10, "sprite area must be 5..10"
    r0, c0, r1, c1 = bbox_coords(coords)
    assert 3 <= (r1 - r0 + 1) <= 4 and 3 <= (c1 - c0 + 1) <= 4, "sprite bbox must be 3..4"
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = GREY
    return patch, (r0, c0, r1, c1)


def _extract_markers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(v) for v in np.unique(g) if int(v) not in {BLACK, GREY})
    markers: list[dict] = []
    for color in colors:
        for coords in _components_of_color(g, color):
            assert len(coords) == 3, "marker must have 3 pixels"
            r0, c0, r1, c1 = bbox_coords(coords)
            assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "marker must occupy a 2x2 box"
            local = frozenset((r - r0, c - c0) for r, c in coords)
            assert local in LOCAL_TO_CORNER, "invalid L marker"
            markers.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "target_corner": LOCAL_TO_CORNER[local],
                }
            )
    assert 2 <= len(markers) <= 4, "expected 2..4 markers"
    seen_corners = {marker["target_corner"] for marker in markers}
    seen_colors = {marker["color"] for marker in markers}
    assert len(seen_corners) == len(markers), "marker orientations must be distinct"
    assert len(seen_colors) == len(markers), "marker colors must be distinct"
    return markers


def _corner_origin(grid_shape: tuple[int, int], patch_shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = grid_shape
    ph, pw = patch_shape
    if corner == "top_left":
        return 0, 0
    if corner == "top_right":
        return 0, w - pw
    if corner == "bottom_left":
        return h - ph, 0
    assert corner == "bottom_right"
    return h - ph, w - pw


def _render_output_from_layout(
    grid_shape: tuple[int, int],
    sprite_patch: np.ndarray,
    corner_to_color: dict[str, int],
) -> np.ndarray:
    h, w = grid_shape
    ph, pw = sprite_patch.shape
    out = np.zeros((h, w), dtype=int)
    stamp_mask = np.zeros((h, w), dtype=bool)
    placements: list[tuple[str, int, int, int]] = []

    for corner, color in corner_to_color.items():
        top, left = _corner_origin((h, w), (ph, pw), corner)
        mask = sprite_patch == GREY
        out[top : top + ph, left : left + pw][mask] = GREY
        stamp_mask[top : top + ph, left : left + pw] |= mask
        placements.append((corner, color, top, left))

    source_sets = [[set() for _ in range(w)] for _ in range(h)]
    source_color: dict[str, int] = {}

    for corner, color, top, left in placements:
        source_color[corner] = color
        for r in range(top, top + ph):
            for c in range(0, left):
                if not stamp_mask[r, c]:
                    source_sets[r][c].add(corner)
            for c in range(left + pw, w):
                if not stamp_mask[r, c]:
                    source_sets[r][c].add(corner)
        for c in range(left, left + pw):
            for r in range(0, top):
                if not stamp_mask[r, c]:
                    source_sets[r][c].add(corner)
            for r in range(top + ph, h):
                if not stamp_mask[r, c]:
                    source_sets[r][c].add(corner)

    for r in range(h):
        for c in range(w):
            if stamp_mask[r, c]:
                continue
            covers = source_sets[r][c]
            if len(covers) == 1:
                only = next(iter(covers))
                out[r, c] = source_color[only]
            elif len(covers) >= 2:
                out[r, c] = RED
    return out


def _parse_layout(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    sprite_patch, _sprite_bbox = _extract_sprite(grid)
    markers = _extract_markers(grid)
    corner_to_color = {marker["target_corner"]: marker["color"] for marker in markers}
    ph, pw = sprite_patch.shape
    h, w = grid.shape
    assert ph <= h and pw <= w
    for corner in CORNER_ORDER:
        top, left = _corner_origin((h, w), (ph, pw), corner)
        assert 0 <= top <= h - ph and 0 <= left <= w - pw
    return sprite_patch, corner_to_color


def main(grid: np.ndarray) -> np.ndarray:
    """Place the central grey sprite into marker-directed corners, then emit marker-colored cross rays whose cross-source intersections turn red.

    Input structure:
    - Black background with one central grey 4-connected sprite whose tight bbox is 3x3 or 4x4.
    - Around the sprite are 2 to 4 colored 3-pixel L markers, each occupying three cells of a 2x2 box.
    - Each marker's L opening direction encodes one target corner, while its own color is reserved for later ray coloring.

    Rule:
    - Crop the grey sprite's tight bbox to obtain the stamp template.
    - Map each L marker orientation to one grid corner, and stamp a grey copy of the template flush to that corner.
    - For every stamped copy, extend rays along every bbox row to the left and right edges and along every bbox column to the top and bottom edges, coloring those rays with the marker's color.
    - Do not overwrite any grey stamp pixels; when rays from two different markers cover the same cell, color that cell red; erase the original central sprite and all input markers.

    Key insight:
    - Each L marker carries two independent signals: geometry chooses where the grey stamp goes, while color chooses how that corner's rays are painted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    sprite_patch, corner_to_color = _parse_layout(g)
    return _render_output_from_layout(g.shape, sprite_patch, corner_to_color)


def _signature(grid: np.ndarray) -> str:
    arr = np.asarray(grid, dtype=int)
    return hashlib.md5(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(int(v) for v in coords[0])
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _non_symmetric(mask: np.ndarray) -> bool:
    return not (
        np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.rot90(mask, 2))
    )


def _random_sprite_patch(rng: random.Random) -> np.ndarray:
    for _ in range(500):
        h = rng.choice((3, 4))
        w = rng.choice((3, 4))
        min_area = max(5, h + w - 1)
        max_area = min(10, h * w - 1)
        area = rng.randint(min_area, max_area)
        mask = np.zeros((h, w), dtype=bool)
        all_cells = [(r, c) for r in range(h) for c in range(w)]
        first = rng.choice(all_cells)
        mask[first] = True
        frontier = {first}
        while int(mask.sum()) < area:
            frontier_cells = set()
            for r, c in frontier:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier_cells.add((nr, nc))
            if not frontier_cells:
                break
            chosen = rng.choice(sorted(frontier_cells))
            mask[chosen] = True
            frontier.add(chosen)
        if int(mask.sum()) != area:
            continue
        if not _is_connected(mask):
            continue
        if not mask.any(axis=0).all() or not mask.any(axis=1).all():
            continue
        if not _non_symmetric(mask):
            continue
        if np.array_equal(mask, np.ones_like(mask, dtype=bool)):
            continue
        patch = np.zeros((h, w), dtype=int)
        patch[mask] = GREY
        return patch
    raise RuntimeError("failed to generate a non-symmetric sprite patch")


def _candidate_marker_slots(sr: int, sc: int, sh: int, sw: int) -> list[tuple[str, int, int]]:
    mid_r = sr + max(0, sh // 2 - 1)
    mid_c = sc + max(0, sw // 2 - 1)
    return [
        ("nw", sr - 4, sc - 4),
        ("n", sr - 4, mid_c),
        ("ne", sr - 4, sc + sw + 2),
        ("w", mid_r, sc - 4),
        ("e", mid_r, sc + sw + 2),
        ("sw", sr + sh + 2, sc - 4),
        ("s", sr + sh + 2, mid_c),
        ("se", sr + sh + 2, sc + sw + 2),
    ]


def _slot_set_fits(slots: list[tuple[str, int, int]], grid_shape: tuple[int, int], sprite_bbox: tuple[int, int, int, int]) -> bool:
    h, w = grid_shape
    sr0, sc0, sr1, sc1 = sprite_bbox
    used: list[tuple[int, int, int, int]] = [(sr0, sc0, sr1, sc1)]
    for _name, r0, c0 in slots:
        r1, c1 = r0 + 1, c0 + 1
        if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
            return False
        for ur0, uc0, ur1, uc1 in used:
            if not (r1 < ur0 or ur1 < r0 or c1 < uc0 or uc1 < c0):
                return False
        used.append((r0, c0, r1, c1))
    return True


def _choose_corner_subset(rng: random.Random) -> list[str]:
    count = rng.choices((2, 3), weights=(3, 5), k=1)[0]
    if count == 2:
        return list(rng.choice((("top_left", "bottom_right"), ("top_right", "bottom_left"))))
    return sorted(rng.sample(CORNER_ORDER, 3))


def _build_input_from_layout(
    grid_shape: tuple[int, int],
    sprite_patch: np.ndarray,
    sprite_top_left: tuple[int, int],
    marker_layout: list[dict],
) -> np.ndarray:
    h, w = grid_shape
    out = np.zeros((h, w), dtype=int)
    sr, sc = sprite_top_left
    sh, sw = sprite_patch.shape
    sprite_mask = sprite_patch == GREY
    out[sr : sr + sh, sc : sc + sw][sprite_mask] = GREY
    for marker in marker_layout:
        top, left = marker["top_left"]
        color = marker["color"]
        for dr, dc in CORNER_TO_LOCAL[marker["target_corner"]]:
            out[top + dr, left + dc] = color
    return out


def _validate_example(
    inp: np.ndarray,
    expected_out: np.ndarray,
    sprite_patch: np.ndarray,
    corner_to_color: dict[str, int],
    marker_layout: list[dict],
) -> bool:
    actual_out = main(inp)
    if not np.array_equal(actual_out, expected_out):
        return False

    parsed_patch, parsed_corner_to_color = _parse_layout(inp)
    if not np.array_equal(parsed_patch, sprite_patch):
        return False
    if parsed_corner_to_color != corner_to_color:
        return False

    if int(np.count_nonzero(actual_out == RED)) == 0:
        return False

    h, w = inp.shape
    ph, pw = sprite_patch.shape
    sprite_mask = sprite_patch == GREY
    for corner in CORNER_ORDER:
        top, left = _corner_origin((h, w), (ph, pw), corner)
        patch = actual_out[top : top + ph, left : left + pw]
        if corner in corner_to_color:
            if not np.array_equal(patch == GREY, sprite_mask):
                return False
        elif int(np.count_nonzero(patch == GREY)) != 0:
            return False

    for color in corner_to_color.values():
        if int(np.count_nonzero(actual_out == color)) == 0:
            return False

    if all(marker["slot_name"] == CORNER_TO_DIAGONAL_SLOT[marker["target_corner"]] for marker in marker_layout):
        return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 20)
    w = rng.randint(14, 20)
    sprite_patch = _random_sprite_patch(rng)
    sh, sw = sprite_patch.shape
    sr = (h - sh) // 2
    sc = (w - sw) // 2
    sprite_bbox = (sr, sc, sr + sh - 1, sc + sw - 1)

    selected_corners = _choose_corner_subset(rng)
    colors = rng.sample(MARKER_COLORS, len(selected_corners))
    corner_to_color = dict(zip(selected_corners, colors))

    candidate_slots = _candidate_marker_slots(sr, sc, sh, sw)
    valid_slots = [slot for slot in candidate_slots if _slot_set_fits([slot], (h, w), sprite_bbox)]
    if len(valid_slots) < len(selected_corners):
        return None

    for _ in range(80):
        chosen_slots = rng.sample(valid_slots, len(selected_corners))
        if not _slot_set_fits(chosen_slots, (h, w), sprite_bbox):
            continue
        marker_layout = []
        for corner, color, (slot_name, top, left) in zip(selected_corners, colors, chosen_slots):
            marker_layout.append(
                {
                    "target_corner": corner,
                    "color": color,
                    "slot_name": slot_name,
                    "top_left": (top, left),
                }
            )
        if all(marker["slot_name"] == CORNER_TO_DIAGONAL_SLOT[marker["target_corner"]] for marker in marker_layout):
            continue
        inp = _build_input_from_layout((h, w), sprite_patch, (sr, sc), marker_layout)
        expected_out = _render_output_from_layout((h, w), sprite_patch, corner_to_color)
        if not _validate_example(inp, expected_out, sprite_patch, corner_to_color, marker_layout):
            continue
        out_sig = _signature(expected_out)
        if out_sig in _SEEN_OUTPUT_SIGS:
            continue
        _SEEN_OUTPUT_SIGS.add(out_sig)
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corner-stamp ray puzzle")
