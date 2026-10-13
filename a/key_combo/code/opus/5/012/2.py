from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _ensure_arc_utils_on_path() -> None:
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for parent in here.parents:
        candidates.append(parent / "new" / "solvers")
    candidates.append(Path("/home/lixinyu/snap/new/solvers"))
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            sys.path.insert(0, key)
            return
    raise ImportError("could not locate arc_utils.py")


_ensure_arc_utils_on_path()

from arc_utils import bbox_coords, crop_bbox, find_components, get_bg, tile_fill


BG = 0
BLUE = 1
RED = 2
TEMPLATE_COLORS = {5, 6}
CONNECTOR_POOL = (3, 4, 8)

SIDE_TO_VEC = {
    "T": (-1, 0),
    "B": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
OPPOSITE = {"T": "B", "B": "T", "L": "R", "R": "L"}

ENDPOINT_MASK = np.array([[1, 1, 1], [0, 1, 0]], dtype=bool)
STRAIGHT_MASK = np.array([[1, 1, 1], [1, 1, 1]], dtype=bool)
CORNER_MASK = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool)

GEN_ATTEMPTS = 400


def _occupied_mask(patch: np.ndarray) -> np.ndarray:
    return patch != BG


def _unique_side_cells(mask: np.ndarray) -> dict[str, tuple[int, int]]:
    h, w = mask.shape
    out: dict[str, tuple[int, int]] = {}
    for r, c in np.argwhere(mask):
        exposed: list[str] = []
        for side, (dr, dc) in SIDE_TO_VEC.items():
            nr, nc = int(r) + dr, int(c) + dc
            if not (0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc])):
                exposed.append(side)
        if len(exposed) == 1:
            out[exposed[0]] = (int(r), int(c))
    return out


def _shape_for_sides(sides: set[str]) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    if len(sides) == 1:
        base = ENDPOINT_MASK
    elif len(sides) == 2 and any({a, b} == sides for a, b in ({"T", "B"}, {"L", "R"})):
        base = STRAIGHT_MASK
    else:
        base = CORNER_MASK
    for k in range(4):
        mask = np.rot90(base, k)
        side_cells = _unique_side_cells(mask)
        if set(side_cells) == set(sides):
            return mask.copy(), side_cells
    raise ValueError(f"no prototype matches port sides {sorted(sides)}")


def _make_fragment(body_color: int, side_to_marker_color: dict[str, int]) -> tuple[np.ndarray, dict[str, dict[str, object]]]:
    mask, side_cells = _shape_for_sides(set(side_to_marker_color))
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(body_color)
    markers: dict[str, dict[str, object]] = {}
    for side, color in side_to_marker_color.items():
        r, c = side_cells[side]
        patch[r, c] = int(color)
        markers[side] = {"pos": (int(r), int(c)), "dir": SIDE_TO_VEC[side], "color": int(color)}
    return patch, markers


def _extract_template(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(np.isin(grid, list(TEMPLATE_COLORS)))
    assert coords.size > 0, "expected a 3x3 template tile"
    r0, c0, r1, c1 = bbox_coords(coords)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "template tile must be 3x3"
    tile = crop_bbox(grid, (r0, c0, r1, c1))
    assert set(map(int, np.unique(tile))) <= TEMPLATE_COLORS, "template tile colors must be 5/6"
    return tile


def _fragment_components(grid: np.ndarray) -> list[dict[str, object]]:
    fragment_colors = [int(v) for v in np.unique(grid) if int(v) not in {BG, *TEMPLATE_COLORS}]
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=False,
        colors=fragment_colors,
        return_format="dict",
    )
    fragments: list[dict[str, object]] = []
    for comp in comps:
        patch = crop_bbox(grid, comp["bbox"])
        body_colors = [int(c) for c in np.unique(patch) if int(c) in {BLUE, RED}]
        assert len(body_colors) == 1, "each fragment must have exactly one body color"
        body_color = body_colors[0]
        abs_r0, abs_c0, _abs_r1, _abs_c1 = map(int, comp["bbox"])
        occupied = _occupied_mask(patch)
        marker_cells = np.argwhere((patch != BG) & (patch != body_color))
        assert 1 <= len(marker_cells) <= 2, "each fragment must have one or two connector markers"
        markers: dict[int, dict[str, object]] = {}
        side_cells = _unique_side_cells(occupied)
        pos_to_side = {pos: side for side, pos in side_cells.items()}
        for rr, cc in marker_cells:
            pos = (int(rr), int(cc))
            assert pos in pos_to_side, "marker must expose exactly one boundary side"
            side = pos_to_side[pos]
            color = int(patch[rr, cc])
            markers[color] = {
                "local_pos": pos,
                "abs_pos": (abs_r0 + int(rr), abs_c0 + int(cc)),
                "dir": SIDE_TO_VEC[side],
                "side": side,
            }
        assert len(markers) == len(marker_cells), "marker colors inside one fragment must be distinct"
        fragments.append(
            {
                "patch": patch,
                "body_color": body_color,
                "markers": markers,
            }
        )
    assert 3 <= len(fragments) <= 4, "expected three or four fragments"
    return fragments


def _docking_pairs(fragments: list[dict[str, object]]) -> list[tuple[tuple[int, int], int, tuple[int, dict[str, object]], tuple[int, dict[str, object]]]]:
    occurrences: dict[int, list[tuple[int, dict[str, object]]]] = defaultdict(list)
    for idx, fragment in enumerate(fragments):
        for color, info in fragment["markers"].items():
            occurrences[int(color)].append((idx, info))
    pairs = []
    for color, occ in occurrences.items():
        assert len(occ) == 2, f"connector color {color} must appear exactly twice"
        sort_key = min((tuple(map(int, item[1]["abs_pos"])) for item in occ))
        pairs.append((sort_key, int(color), occ[0], occ[1]))
    pairs.sort(key=lambda item: item[0])
    assert len(pairs) == len(fragments) - 1, "fragments must form a connected docking tree"
    return pairs


def _assemble_positions(fragments: list[dict[str, object]]) -> dict[int, tuple[int, int]]:
    pairs = _docking_pairs(fragments)
    placements: dict[int, tuple[int, int]] = {}
    group_of: dict[int, int] = {}
    group_members: dict[int, set[int]] = {}
    next_group = 0

    def add_new_group(a: int, b: int, delta: tuple[int, int]) -> None:
        nonlocal next_group
        gid = next_group
        next_group += 1
        placements[a] = (0, 0)
        placements[b] = delta
        group_of[a] = gid
        group_of[b] = gid
        group_members[gid] = {a, b}

    def add_to_group(idx: int, gid: int, pos: tuple[int, int]) -> None:
        placements[idx] = pos
        group_of[idx] = gid
        group_members[gid].add(idx)

    def shift_group(gid: int, dr: int, dc: int) -> None:
        for member in list(group_members[gid]):
            mr, mc = placements[member]
            placements[member] = (mr + int(dr), mc + int(dc))

    for _sort_key, _color, (a_idx, a_info), (b_idx, b_info) in pairs:
        a_pos = tuple(map(int, a_info["local_pos"]))
        b_pos = tuple(map(int, b_info["local_pos"]))
        a_dir = tuple(map(int, a_info["dir"]))
        b_dir = tuple(map(int, b_info["dir"]))
        assert b_dir == (-a_dir[0], -a_dir[1]), "matched connectors must face each other"
        delta = (a_pos[0] + a_dir[0] - b_pos[0], a_pos[1] + a_dir[1] - b_pos[1])

        a_placed = a_idx in placements
        b_placed = b_idx in placements
        if not a_placed and not b_placed:
            add_new_group(a_idx, b_idx, delta)
            continue
        if a_placed and not b_placed:
            ar, ac = placements[a_idx]
            add_to_group(b_idx, group_of[a_idx], (ar + delta[0], ac + delta[1]))
            continue
        if b_placed and not a_placed:
            br, bc = placements[b_idx]
            add_to_group(a_idx, group_of[b_idx], (br - delta[0], bc - delta[1]))
            continue

        desired_b = (placements[a_idx][0] + delta[0], placements[a_idx][1] + delta[1])
        ga = group_of[a_idx]
        gb = group_of[b_idx]
        if ga == gb:
            assert placements[b_idx] == desired_b, "existing docking constraints must be consistent"
            continue
        shift_group(gb, desired_b[0] - placements[b_idx][0], desired_b[1] - placements[b_idx][1])
        for member in list(group_members[gb]):
            group_of[member] = ga
        group_members[ga].update(group_members[gb])
        del group_members[gb]

    assert len(placements) == len(fragments), "all fragments must be placed by the docking pairs"
    return placements


def _assembled_counts(fragments: list[dict[str, object]], placements: dict[int, tuple[int, int]]) -> tuple[int, int]:
    occupied: dict[tuple[int, int], int] = {}
    red_total = 0
    blue_total = 0
    for idx, fragment in enumerate(fragments):
        patch = np.asarray(fragment["patch"], dtype=int)
        br, bc = placements[idx]
        for rr, cc in np.argwhere(patch != BG):
            pos = (br + int(rr), bc + int(cc))
            assert pos not in occupied, "assembled fragments must not overlap"
            color = int(patch[rr, cc])
            occupied[pos] = color
            if color == RED:
                red_total += 1
            elif color == BLUE:
                blue_total += 1
    return red_total, blue_total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock the marked fragments into one assembly, use its red/blue body counts as output size, then tile the corner stencil.

    Input structure:
    - The background is black.
    - The upper part contains three or four 4-connected multicolor fragments whose body color is either blue or red.
    - Each fragment also carries one or two boundary connector pixels of non-body colors; every connector color appears on exactly two fragments.
    - A separate 3x3 stencil tile in the lower-right corner is filled entirely with colors 5 and 6.

    Rule:
    - Crop the 3x3 lower-right stencil tile.
    - Identify every fragment and, for each repeated connector color, the two fragments that own that color.
    - Process connector pairs in the order of the lexicographically top-left occurrence among the two marker pixels, translating fragments so the matched marker cells touch edge-to-edge along their exposed boundary normals.
    - After all dockings, count red body pixels to obtain the output width and blue body pixels to obtain the output height.
    - Return an output grid of that height and width, filled by periodically tiling the 3x3 stencil and cropping the excess.

    Key insight:
    - The marker colors define a docking tree, while each marker's exposed side fixes the translation direction needed to recover the assembly that encodes the tiled output size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert int(get_bg(g)) == BG, "this task uses black background"

    tile = _extract_template(g)
    fragments = _fragment_components(g)
    placements = _assemble_positions(fragments)
    width, height = _assembled_counts(fragments, placements)
    assert 1 <= width <= 30 and 1 <= height <= 30, "decoded output size must stay within ARC bounds"
    return tile_fill((height, width), tile)


def _choose_body_colors(body_counts: list[int]) -> list[int] | None:
    valid: list[list[int]] = []
    n = len(body_counts)
    for mask in range(1, 1 << n):
        colors = [RED if (mask >> i) & 1 else BLUE for i in range(n)]
        if len(set(colors)) < 2:
            continue
        red_total = sum(body_counts[i] for i in range(n) if colors[i] == RED)
        blue_total = sum(body_counts[i] for i in range(n) if colors[i] == BLUE)
        if 3 <= red_total <= 8 and 3 <= blue_total <= 8 and red_total != blue_total:
            valid.append(colors)
    if not valid:
        return None
    return random.choice(valid)


def _choose_edge_sides(n_fragments: int) -> list[str]:
    sides = [random.choice(list(SIDE_TO_VEC))]
    for _ in range(1, n_fragments - 1):
        incoming = OPPOSITE[sides[-1]]
        opposite = OPPOSITE[incoming]
        candidates = [side for side in SIDE_TO_VEC if side != incoming]
        if random.random() < 0.5:
            sides.append(opposite)
        else:
            corner_candidates = [side for side in candidates if side != opposite]
            sides.append(random.choice(corner_candidates))
    return sides


def _build_chain_fragments(n_fragments: int) -> tuple[list[dict[str, object]], list[int]]:
    connector_colors = list(CONNECTOR_POOL[: n_fragments - 1])
    random.shuffle(connector_colors)
    edge_sides = _choose_edge_sides(n_fragments)
    body_counts = [3] + [4] * (n_fragments - 2) + [3]
    body_colors = _choose_body_colors(body_counts)
    if body_colors is None:
        raise ValueError("failed to choose valid red/blue body totals")

    fragments: list[dict[str, object]] = []
    for idx in range(n_fragments):
        side_to_marker: dict[str, int] = {}
        if idx > 0:
            side_to_marker[OPPOSITE[edge_sides[idx - 1]]] = connector_colors[idx - 1]
        if idx < n_fragments - 1:
            side_to_marker[edge_sides[idx]] = connector_colors[idx]
        patch, markers = _make_fragment(body_colors[idx], side_to_marker)
        fragments.append(
            {
                "patch": patch,
                "body_color": body_colors[idx],
                "markers": {
                    info["color"]: {
                        "local_pos": tuple(map(int, info["pos"])),
                        "dir": tuple(map(int, info["dir"])),
                        "side": side,
                    }
                    for side, info in markers.items()
                },
            }
        )
    return fragments, connector_colors


def _assembled_chain_positions(fragments: list[dict[str, object]], connector_colors: list[int]) -> dict[int, tuple[int, int]] | None:
    placements: dict[int, tuple[int, int]] = {0: (0, 0)}
    for color, left_idx in zip(connector_colors, range(len(connector_colors))):
        right_idx = left_idx + 1
        left_info = fragments[left_idx]["markers"][color]
        right_info = fragments[right_idx]["markers"][color]
        left_pos = tuple(map(int, left_info["local_pos"]))
        left_dir = tuple(map(int, left_info["dir"]))
        right_pos = tuple(map(int, right_info["local_pos"]))
        right_dir = tuple(map(int, right_info["dir"]))
        if right_dir != (-left_dir[0], -left_dir[1]):
            return None
        base_left = placements[left_idx]
        placements[right_idx] = (
            base_left[0] + left_pos[0] + left_dir[0] - right_pos[0],
            base_left[1] + left_pos[1] + left_dir[1] - right_pos[1],
        )

    occupied: set[tuple[int, int]] = set()
    for idx, fragment in enumerate(fragments):
        br, bc = placements[idx]
        for rr, cc in np.argwhere(fragment["patch"] != BG):
            pos = (br + int(rr), bc + int(cc))
            if pos in occupied:
                return None
            occupied.add(pos)
    return placements


def _random_tile() -> np.ndarray:
    for _ in range(100):
        tile = np.random.choice([5, 6], size=(3, 3))
        if len(set(map(int, tile.flatten()))) < 2:
            continue
        if all(np.array_equal(tile[r], tile[0]) for r in range(1, 3)):
            continue
        if all(np.array_equal(tile[:, c], tile[:, 0]) for c in range(1, 3)):
            continue
        return tile.astype(int)
    raise RuntimeError("failed to generate a non-degenerate 3x3 tile")


def _can_place(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, pad: int = 1) -> bool:
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    if np.any(grid[rr0:rr1, cc0:cc1] != BG):
        return False
    return True


def _place_fragments_on_input(fragments: list[dict[str, object]], tile: np.ndarray) -> np.ndarray | None:
    H = random.randint(14, 18)
    W = random.randint(14, 18)
    grid = np.zeros((H, W), dtype=int)
    grid[H - 3 : H, W - 3 : W] = tile

    order = list(range(len(fragments)))
    random.shuffle(order)
    top_limit = H // 2 + random.randint(0, 1)
    reserved_col_limit = W - 4

    for idx in order:
        patch = np.asarray(fragments[idx]["patch"], dtype=int)
        h, w = patch.shape
        max_r = top_limit - h
        max_c = reserved_col_limit - w
        if max_r < 0 or max_c < 0:
            return None
        positions = [(r, c) for r in range(0, max_r + 1) for c in range(0, max_c + 1)]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            if not _can_place(grid, patch, r0, c0, pad=1):
                continue
            region = grid[r0 : r0 + h, c0 : c0 + w]
            if np.any((region != BG) & (patch != BG)):
                continue
            region[patch != BG] = patch[patch != BG]
            for color, info in fragments[idx]["markers"].items():
                info["abs_pos"] = (r0 + int(info["local_pos"][0]), c0 + int(info["local_pos"][1]))
            placed = True
            break
        if not placed:
            return None
    return grid


def _pair_order_keys(fragments: list[dict[str, object]]) -> list[tuple[int, int]]:
    keys = []
    seen: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for fragment in fragments:
        for color, info in fragment["markers"].items():
            seen[int(color)].append(tuple(map(int, info["abs_pos"])))
    for occ in seen.values():
        keys.append(min(occ))
    return keys


def _gen_attempt() -> np.ndarray | None:
    n_fragments = random.choice([3, 4])
    try:
        fragments, connector_colors = _build_chain_fragments(n_fragments)
    except ValueError:
        return None

    assembled = _assembled_chain_positions(fragments, connector_colors)
    if assembled is None:
        return None
    red_total, blue_total = _assembled_counts(fragments, assembled)
    if not (3 <= red_total <= 8 and 3 <= blue_total <= 8):
        return None
    if red_total == blue_total:
        return None

    tile = _random_tile()
    inp = _place_fragments_on_input(fragments, tile)
    if inp is None:
        return None

    keys = _pair_order_keys(fragments)
    if len(keys) != len(set(keys)):
        return None

    # Semantic checks beyond self-consistency.
    try:
        out = main(inp)
    except Exception:
        return None
    expected = tile_fill((blue_total, red_total), tile)
    if not np.array_equal(out, expected):
        return None
    if out.shape[0] == out.shape[1]:
        return None
    if np.array_equal(out, tile_fill((red_total, blue_total), tile)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
