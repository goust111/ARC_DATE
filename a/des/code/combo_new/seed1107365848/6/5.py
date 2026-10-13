from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
RED = 2
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CORNER_TO_INDEX = {"tl": 0, "tr": 1, "br": 2, "bl": 3}


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _corner_of_point(r: int, c: int, h: int, w: int) -> str | None:
    if (r, c) == (0, 0):
        return "tl"
    if (r, c) == (0, w - 1):
        return "tr"
    if (r, c) == (h - 1, w - 1):
        return "br"
    if (r, c) == (h - 1, 0):
        return "bl"
    return None


def _corner_after_rot90_ccw(corner: str, k: int) -> str:
    order = ["tl", "bl", "br", "tr"]
    idx = order.index(corner)
    return order[(idx + k) % 4]


def _corner_cell(patch: np.ndarray, corner: str) -> tuple[int, int]:
    h, w = patch.shape
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - 1
    if corner == "br":
        return h - 1, w - 1
    return h - 1, 0


def _shape_matches_under_rotation(shape: tuple[int, int], target: tuple[int, int]) -> bool:
    h, w = shape
    return target == (h, w) or target == (w, h)


def _analyze_stamp(patch: np.ndarray, bg: int) -> dict | None:
    if np.any(patch == bg):
        return None
    red_pts = np.argwhere(patch == RED)
    if red_pts.shape[0] != 1:
        return None
    rr, rc = map(int, red_pts[0])
    marker_corner = _corner_of_point(rr, rc, *patch.shape)
    if marker_corner is None:
        return None

    edge_mask = np.zeros(patch.shape, dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    edge_vals = patch[edge_mask]
    edge_non_red = edge_vals[edge_vals != RED]
    if edge_non_red.size == 0:
        return None
    vals, counts = np.unique(edge_non_red, return_counts=True)
    border = int(vals[np.argmax(counts)])
    if np.any((edge_vals != border) & (edge_vals != RED)):
        return None

    return {
        "kind": "stamp",
        "color": border,
        "patch": patch.copy(),
        "marker_corner": marker_corner,
        "shape": tuple(map(int, patch.shape)),
    }


def _corner_mask(mask: np.ndarray, corner: str) -> tuple[int, int] | None:
    h, w = mask.shape
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    if corner == "tl":
        cut_h = int(coords[:, 0].max()) + 1
        cut_w = int(coords[:, 1].max()) + 1
        exp = np.zeros_like(mask)
        exp[:cut_h, :cut_w] = True
    elif corner == "tr":
        cut_h = int(coords[:, 0].max()) + 1
        cut_w = int(w - coords[:, 1].min())
        exp = np.zeros_like(mask)
        exp[:cut_h, w - cut_w :] = True
    elif corner == "br":
        cut_h = int(h - coords[:, 0].min())
        cut_w = int(w - coords[:, 1].min())
        exp = np.zeros_like(mask)
        exp[h - cut_h :, w - cut_w :] = True
    else:
        cut_h = int(h - coords[:, 0].min())
        cut_w = int(coords[:, 1].max()) + 1
        exp = np.zeros_like(mask)
        exp[h - cut_h :, :cut_w] = True
    if not np.array_equal(mask, exp):
        return None
    return cut_h, cut_w


def _analyze_envelope(patch: np.ndarray, color: int, bg: int) -> dict | None:
    if np.any((patch != color) & (patch != bg)):
        return None
    cut_mask = patch == bg
    if not cut_mask.any():
        return None

    corners = {
        "tl": bool(cut_mask[0, 0]),
        "tr": bool(cut_mask[0, -1]),
        "br": bool(cut_mask[-1, -1]),
        "bl": bool(cut_mask[-1, 0]),
    }
    if sum(corners.values()) != 1:
        return None
    corner = next(name for name, present in corners.items() if present)
    dims = _corner_mask(cut_mask, corner)
    if dims is None:
        return None
    cut_h, cut_w = dims
    h, w = patch.shape
    if cut_h >= h or cut_w >= w:
        return None

    if corner == "tl":
        hole_origin = (0, 0)
    elif corner == "tr":
        hole_origin = (0, w - cut_w)
    elif corner == "br":
        hole_origin = (h - cut_h, w - cut_w)
    else:
        hole_origin = (h - cut_h, 0)

    return {
        "kind": "envelope",
        "color": color,
        "corner": corner,
        "hole_shape": (cut_h, cut_w),
        "hole_origin_local": hole_origin,
        "patch_shape": tuple(map(int, patch.shape)),
    }


def _rotate_to_target(
    patch: np.ndarray, marker_corner: str, target_corner: str, target_shape: tuple[int, int]
) -> np.ndarray:
    for k in range(4):
        rotated = np.rot90(patch, k)
        if rotated.shape != target_shape:
            continue
        if _corner_after_rot90_ccw(marker_corner, k) == target_corner:
            return rotated
    raise AssertionError("no valid stamp rotation for target corner")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each color-matched stamp so its red-corner orientation matches the envelope's missing corner, then embed it into that corner cutout.

    Input structure:
    - The background is black.
    - Each envelope is a large monochrome rectangle with one smaller corner rectangle removed to background.
    - Each stamp is a smaller full rectangle whose border color matches one envelope, whose interior contains a second-color motif, and whose single red pixel sits at one corner.

    Rule:
    - Find every stamp from its red corner marker and read which corner currently holds that marker.
    - Find every envelope by detecting a monochrome component whose bounding box is a full rectangle except for one missing corner block, and read which corner is missing.
    - Match stamps to envelopes by their shared border color, rotate each stamp by 0/90/180/270 so the red corner would land on the missing envelope corner, and paste it into the cutout.
    - Erase the original stamp, and after pasting replace the red marker cell with the envelope color so the completed envelope corner is solid.

    Key insight:
    - Color determines which stamp belongs to which envelope, but the actual rotation comes from combining the envelope's missing-corner direction with the stamp's current red-corner direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    envelopes_by_color: dict[int, list[dict]] = defaultdict(list)
    stamps_by_color: dict[int, list[dict]] = defaultdict(list)

    for coords in _components(g != bg):
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = {int(v) for v in np.unique(patch[patch != bg])}
        if RED in colors:
            stamp = _analyze_stamp(patch, bg)
            assert stamp is not None, "red component must be a valid stamp"
            stamp["bbox"] = (r0, c0, r1, c1)
            stamps_by_color[stamp["color"]].append(stamp)
            continue
        assert len(colors) == 1, "non-stamp foreground components must be monochrome envelopes"
        color = next(iter(colors))
        env = _analyze_envelope(patch, color, bg)
        assert env is not None, "monochrome component must be a valid envelope"
        env["bbox"] = (r0, c0, r1, c1)
        env["hole_origin"] = (r0 + env["hole_origin_local"][0], c0 + env["hole_origin_local"][1])
        envelopes_by_color[color].append(env)

    assert envelopes_by_color, "expected at least one envelope"
    assert stamps_by_color, "expected at least one stamp"

    for color, envs in envelopes_by_color.items():
        stamps = stamps_by_color.get(color, [])
        assert len(stamps) == len(envs), f"color {color} should have equal envelope/stamp counts"
        unused = set(range(len(stamps)))
        envs_sorted = sorted(envs, key=lambda item: (item["hole_shape"], item["bbox"]))
        for env in envs_sorted:
            candidates = [
                i
                for i in unused
                if _shape_matches_under_rotation(stamps[i]["shape"], env["hole_shape"])
            ]
            assert len(candidates) == 1, f"color {color} must pair to exactly one compatible stamp"
            idx = candidates[0]
            unused.remove(idx)
            stamp = stamps[idx]

            sr0, sc0, sr1, sc1 = stamp["bbox"]
            out[sr0 : sr1 + 1, sc0 : sc1 + 1] = bg

            rotated = _rotate_to_target(
                stamp["patch"], stamp["marker_corner"], env["corner"], env["hole_shape"]
            ).copy()
            mr, mc = _corner_cell(rotated, env["corner"])
            rotated[mr, mc] = color

            hr, hc = env["hole_origin"]
            hh, hw = env["hole_shape"]
            out[hr : hr + hh, hc : hc + hw] = rotated

    return out


GEN_COLOR_POOL = [1, 3, 4, 5, 6, 7, 8, 9]
PAIR_COUNT_RANGE = (1, 3)
GRID_SIDE_RANGE = (18, 30)
ENVELOPE_SIDE_RANGE = (7, 12)
CUT_SIDE_RANGE = (4, 6)
PLACE_PAD = 1
GEN_RETRIES = 400


def _random_connected_mask(size: int) -> np.ndarray | None:
    max_area = size * size - 1
    target = random.randint(1, min(max_area, size + 2))
    cells = {(random.randrange(size), random.randrange(size))}
    frontier = set()
    for r, c in list(cells):
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                frontier.add((nr, nc))
    while len(cells) < target and frontier:
        nr, nc = random.choice(list(frontier))
        frontier.remove((nr, nc))
        cells.add((nr, nc))
        for dr, dc in DIR4:
            rr, cc = nr + dr, nc + dc
            if 0 <= rr < size and 0 <= cc < size and (rr, cc) not in cells:
                frontier.add((rr, cc))
    if len(cells) != target:
        return None
    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    for k in range(1, 4):
        if np.array_equal(mask, np.rot90(mask, k)):
            return None
    return mask


def _build_stamp(side: int, border: int, fill: int, marker_corner: str) -> np.ndarray:
    patch = np.full((side, side), border, dtype=int)
    inner_side = side - 2
    inner = None
    for _ in range(100):
        inner = _random_connected_mask(inner_side)
        if inner is not None:
            break
    if inner is None:
        raise RuntimeError("failed to build asymmetric inner mask")
    patch[1:-1, 1:-1][inner] = fill
    mr, mc = _corner_cell(patch, marker_corner)
    patch[mr, mc] = RED
    return patch


def _can_place(occupied: np.ndarray, h: int, w: int, pad: int) -> list[tuple[int, int]]:
    H, W = occupied.shape
    positions = []
    for r0 in range(H - h + 1):
        rr0 = max(0, r0 - pad)
        rr1 = min(H, r0 + h + pad)
        for c0 in range(W - w + 1):
            cc0 = max(0, c0 - pad)
            cc1 = min(W, c0 + w + pad)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            positions.append((r0, c0))
    return positions


def _hole_origin(corner: str, env_h: int, env_w: int, cut: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, env_w - cut
    if corner == "br":
        return env_h - cut, env_w - cut
    return env_h - cut, 0


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    border_colors = random.sample(GEN_COLOR_POOL, n_pairs)
    remaining = [c for c in GEN_COLOR_POOL if c not in border_colors]
    fill_colors = random.sample(remaining, n_pairs)

    specs = []
    for i in range(n_pairs):
        cut = random.randint(*CUT_SIDE_RANGE)
        env_h = random.randint(max(cut + 2, ENVELOPE_SIDE_RANGE[0]), ENVELOPE_SIDE_RANGE[1])
        env_w = random.randint(max(cut + 2, ENVELOPE_SIDE_RANGE[0]), ENVELOPE_SIDE_RANGE[1])
        target_corner = random.choice(["tl", "tr", "br", "bl"])
        marker_corner = random.choice(["tl", "tr", "br", "bl"])
        stamp = _build_stamp(cut, border_colors[i], fill_colors[i], marker_corner)
        specs.append(
            {
                "border": border_colors[i],
                "fill": fill_colors[i],
                "cut": cut,
                "env_h": env_h,
                "env_w": env_w,
                "target_corner": target_corner,
                "marker_corner": marker_corner,
                "stamp": stamp,
            }
        )

    H = random.randint(*GRID_SIDE_RANGE)
    W = random.randint(*GRID_SIDE_RANGE)
    input_grid = np.full((H, W), BG, dtype=int)
    expected = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    for spec in sorted(specs, key=lambda item: -(item["env_h"] * item["env_w"])):
        env_positions = _can_place(occupied, spec["env_h"], spec["env_w"], PLACE_PAD)
        if not env_positions:
            return None
        er0, ec0 = random.choice(env_positions)
        occupied[er0 : er0 + spec["env_h"], ec0 : ec0 + spec["env_w"]] = True
        spec["env_pos"] = (er0, ec0)

        stamp_positions = _can_place(occupied, spec["cut"], spec["cut"], PLACE_PAD)
        if not stamp_positions:
            return None
        sr0, sc0 = random.choice(stamp_positions)
        occupied[sr0 : sr0 + spec["cut"], sc0 : sc0 + spec["cut"]] = True
        spec["stamp_pos"] = (sr0, sc0)

    for spec in specs:
        border = spec["border"]
        cut = spec["cut"]
        env_h = spec["env_h"]
        env_w = spec["env_w"]
        er0, ec0 = spec["env_pos"]

        expected[er0 : er0 + env_h, ec0 : ec0 + env_w] = border
        rotated = _rotate_to_target(
            spec["stamp"], spec["marker_corner"], spec["target_corner"], (cut, cut)
        ).copy()
        rr, rc = _corner_cell(rotated, spec["target_corner"])
        rotated[rr, rc] = border
        hr_local, hc_local = _hole_origin(spec["target_corner"], env_h, env_w, cut)
        expected[
            er0 + hr_local : er0 + hr_local + cut,
            ec0 + hc_local : ec0 + hc_local + cut,
        ] = rotated

        input_grid[er0 : er0 + env_h, ec0 : ec0 + env_w] = border
        input_grid[
            er0 + hr_local : er0 + hr_local + cut,
            ec0 + hc_local : ec0 + hc_local + cut,
        ] = BG

        sr0, sc0 = spec["stamp_pos"]
        input_grid[sr0 : sr0 + cut, sc0 : sc0 + cut] = spec["stamp"]

    try:
        actual = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid example")
