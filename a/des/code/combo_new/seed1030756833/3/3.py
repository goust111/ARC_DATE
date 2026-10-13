from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SIDES = ("top", "right", "bottom", "left")
SIDE_CENTER = {
    "top": (0, 2),
    "right": (2, 4),
    "bottom": (4, 2),
    "left": (2, 0),
}
SIDE_TO_CORNER = {
    "top": (0, 0),
    "right": (0, 2),
    "bottom": (2, 2),
    "left": (2, 0),
}
CORNER_TO_INDEX = {
    (0, 0): 0,
    (0, 2): 1,
    (2, 2): 2,
    (2, 0): 3,
}
CORNER_POSITIONS = tuple(CORNER_TO_INDEX)
GEN_RETRIES = 1000


def _border_positions() -> list[tuple[int, int]]:
    cells = []
    for r in range(5):
        for c in range(5):
            if r in (0, 4) or c in (0, 4):
                cells.append((r, c))
    return cells


BORDER_POSITIONS = _border_positions()


def _find_envelopes(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    envelopes = []
    for r0 in range(h - 4):
        for c0 in range(w - 4):
            patch = g[r0 : r0 + 5, c0 : c0 + 5]
            if not np.all(patch[1:4, 1:4] == BG):
                continue

            vals = [int(patch[r, c]) for r, c in BORDER_POSITIONS]
            counts = Counter(vals)
            if len(counts) != 2:
                continue

            frame_color = marker_color = None
            for color, count in counts.items():
                if color == BG:
                    break
                if count == 15:
                    frame_color = int(color)
                elif count == 1:
                    marker_color = int(color)
            else:
                if frame_color is None or marker_color is None:
                    continue
                marker_cells = [
                    (rr, cc)
                    for rr, cc in BORDER_POSITIONS
                    if int(patch[rr, cc]) == marker_color
                ]
                if len(marker_cells) != 1:
                    continue
                marker_cell = marker_cells[0]
                side = None
                for name, pos in SIDE_CENTER.items():
                    if marker_cell == pos:
                        side = name
                        break
                if side is None:
                    continue
                envelopes.append(
                    {
                        "color": frame_color,
                        "marker": marker_color,
                        "side": side,
                        "r0": r0,
                        "c0": c0,
                    }
                )
    return envelopes


def _find_stamps(g: np.ndarray, color: int) -> list[dict]:
    h, w = g.shape
    stamps = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            patch = g[r0 : r0 + 3, c0 : c0 + 3]
            mask = patch == color
            if int(mask.sum()) != 8:
                continue
            missing = np.argwhere(~mask)
            if missing.shape != (1, 2):
                continue
            mr, mc = map(int, missing[0])
            corner = (mr, mc)
            if corner not in CORNER_TO_INDEX:
                continue
            if int(patch[mr, mc]) != BG:
                continue
            stamps.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "corner": corner,
                    "mask": mask.copy(),
                }
            )
    return stamps


def _rotated_stamp_mask(mask: np.ndarray, current_corner: tuple[int, int], target_side: str) -> np.ndarray:
    target_corner = SIDE_TO_CORNER[target_side]
    current_index = CORNER_TO_INDEX[current_corner]
    target_index = CORNER_TO_INDEX[target_corner]
    k_ccw = (current_index - target_index) % 4
    return np.rot90(mask, k_ccw)


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each paired stamp into its matching envelope and recolor it with the envelope marker.

    Input structure:
    - The background is black.
    - Each pair has a 3x3 same-color stamp with exactly one missing corner and a separated 5x5 same-color hollow envelope frame.
    - Each envelope has one side-center border cell replaced by a globally unique marker color.

    Rule:
    - Match every stamp to the envelope whose frame has the same color.
    - Interpret the missing-corner cycle as upper-left, upper-right, lower-right, lower-left corresponding to up, right, down, left.
    - Rotate the stamp mask until its missing corner faces the marked side of the envelope.
    - Erase the original stamp and fill the envelope's 3x3 cavity with the rotated stamp mask using the marker color.
    - Preserve the envelope border, including the side-center marker cell.

    Key insight:
    - The shared color gives the pair, the stamp corner and envelope marker give the rotation, and the marker color gives the fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    envelopes = _find_envelopes(g)
    assert envelopes, "expected at least one marked envelope"

    by_color: dict[int, dict] = {}
    for env in envelopes:
        color = int(env["color"])
        assert color not in by_color, "each envelope frame color must be unique"
        by_color[color] = env

    out = g.copy()
    for color, env in by_color.items():
        stamps = _find_stamps(g, color)
        assert len(stamps) == 1, "each frame color must have exactly one stamp"
        stamp = stamps[0]
        rotated = _rotated_stamp_mask(stamp["mask"], stamp["corner"], env["side"])

        sr0, sc0 = int(stamp["r0"]), int(stamp["c0"])
        out[sr0 : sr0 + 3, sc0 : sc0 + 3][stamp["mask"]] = BG

        er0, ec0 = int(env["r0"]), int(env["c0"])
        cavity = out[er0 + 1 : er0 + 4, ec0 + 1 : ec0 + 4]
        cavity[:, :] = BG
        cavity[rotated] = int(env["marker"])

    return out


def _stamp_patch(color: int, missing_corner: tuple[int, int]) -> np.ndarray:
    patch = np.full((3, 3), int(color), dtype=int)
    patch[missing_corner] = BG
    return patch


def _envelope_patch(color: int, marker: int, side: str) -> np.ndarray:
    patch = np.zeros((5, 5), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    patch[SIDE_CENTER[side]] = marker
    return patch


def _expanded_box(box: tuple[int, int, int, int], gap: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 - gap, c0 - gap, r1 + gap, c1 + gap


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _valid_positions(shape: tuple[int, int], patch_shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    h, w = shape
    ph, pw = patch_shape
    positions = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            padded = _expanded_box(box, 1)
            if any(_boxes_overlap(padded, old) for old in boxes):
                continue
            positions.append((r0, c0))
    return positions


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BG
    view = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    view[mask] = patch[mask]


def _build_expected(grid: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for pair in pairs:
        stamp_r, stamp_c = pair["stamp_pos"]
        env_r, env_c = pair["envelope_pos"]
        missing = pair["missing"]
        side = pair["side"]
        marker = pair["marker"]

        mask = _stamp_patch(1, missing) == 1
        rotated = _rotated_stamp_mask(mask, missing, side)
        out[stamp_r : stamp_r + 3, stamp_c : stamp_c + 3][mask] = BG
        cavity = out[env_r + 1 : env_r + 4, env_c + 1 : env_c + 4]
        cavity[:, :] = BG
        cavity[rotated] = marker
    return out


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.choice((2, 2, 3, 3, 4))
    min_size = 12 + 3 * n_pairs
    h = random.randint(min_size, 30)
    w = random.randint(min_size, 30)
    grid = np.zeros((h, w), dtype=int)

    palette = random.sample(range(1, 10), 2 * n_pairs)
    frame_colors = palette[:n_pairs]
    marker_colors = palette[n_pairs:]

    if n_pairs == 4:
        sides = list(SIDES)
        corners = list(CORNER_POSITIONS)
        random.shuffle(sides)
        random.shuffle(corners)
    else:
        sides = [random.choice(SIDES) for _ in range(n_pairs)]
        corners = [random.choice(CORNER_POSITIONS) for _ in range(n_pairs)]

    objects = []
    pairs = []
    for i in range(n_pairs):
        color = frame_colors[i]
        marker = marker_colors[i]
        side = sides[i]
        missing = corners[i]
        stamp = _stamp_patch(color, missing)
        envelope = _envelope_patch(color, marker, side)

        pairs.append(
            {
                "color": color,
                "marker": marker,
                "side": side,
                "missing": missing,
                "stamp_pos": None,
                "envelope_pos": None,
            }
        )
        objects.append({"pair": i, "kind": "stamp", "patch": stamp})
        objects.append({"pair": i, "kind": "envelope", "patch": envelope})

    random.shuffle(objects)
    boxes: list[tuple[int, int, int, int]] = []
    for obj in objects:
        patch = obj["patch"]
        positions = _valid_positions(grid.shape, patch.shape, boxes)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        _paint_patch(grid, patch, r0, c0)
        boxes.append((r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1))
        if obj["kind"] == "stamp":
            pairs[obj["pair"]]["stamp_pos"] = (r0, c0)
        else:
            pairs[obj["pair"]]["envelope_pos"] = (r0, c0)

    if any(pair["stamp_pos"] is None or pair["envelope_pos"] is None for pair in pairs):
        return None

    expected = _build_expected(grid, pairs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
