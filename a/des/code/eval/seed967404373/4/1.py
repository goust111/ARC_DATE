from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIDE_TO_ROT = {"top": 0, "right": 3, "bottom": 2, "left": 1}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _square_border(side: int) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _seed_positions(size: int) -> dict[tuple[int, int], str]:
    mid = size // 2
    return {
        (0, mid): "top",
        (mid, size - 1): "right",
        (size - 1, mid): "bottom",
        (mid, 0): "left",
    }


def _parse_frame_component(
    grid: np.ndarray, coords: np.ndarray, color: int, bg: int
) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h != w or h < 5:
        return None
    if int(coords.shape[0]) != 4 * h - 4:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = _square_border(h)
    if not bool(np.all(patch[border] == int(color))):
        return None
    if bool(np.any(patch[~border] == int(color))):
        return None

    inner = patch[1:-1, 1:-1]
    inner_side = int(h - 2)
    if inner_side < 3 or inner_side % 2 == 0:
        return None
    non_bg = np.argwhere(inner != int(bg))
    if int(non_bg.shape[0]) != 1:
        return None

    sr, sc = map(int, non_bg[0])
    seed_side = _seed_positions(inner_side).get((sr, sc))
    if seed_side is None:
        return None
    seed_color = int(inner[sr, sc])
    if seed_color == int(color):
        return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "frame_color": int(color),
        "size": int(inner_side),
        "seed_color": int(seed_color),
        "seed_side": seed_side,
        "seed_abs": (int(r0 + 1 + sr), int(c0 + 1 + sc)),
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        for comp in _components(grid == int(color)):
            frame = _parse_frame_component(grid, comp, int(color), int(bg))
            if frame is not None:
                frames.append(frame)

    assert frames, "expected at least one hollow square frame"
    sizes = {int(frame["size"]) for frame in frames}
    assert len(sizes) == 1, "all frames must share one interior size"
    frame_colors = {int(frame["frame_color"]) for frame in frames}
    assert len(frame_colors) == 1, "all frames must use one border color"
    seed_colors = [int(frame["seed_color"]) for frame in frames]
    assert len(seed_colors) == len(set(seed_colors)), "seed colors must uniquely identify pieces"
    return frames


def _point_in_any_frame(point: tuple[int, int], frames: Iterable[dict]) -> bool:
    r, c = point
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        if int(r0) <= r <= int(r1) and int(c0) <= c <= int(c1):
            return True
    return False


def _extract_piece_patch(
    grid: np.ndarray, bg: int, frames: list[dict], frame: dict
) -> tuple[np.ndarray, np.ndarray]:
    color = int(frame["seed_color"])
    seed_abs = tuple(map(int, frame["seed_abs"]))
    piece_cells: list[tuple[int, int]] = []
    seed_hits = 0
    for r, c in np.argwhere(grid == int(color)):
        rr, cc = int(r), int(c)
        if (rr, cc) == seed_abs:
            seed_hits += 1
            continue
        assert not _point_in_any_frame((rr, cc), frames), "piece cells must stay outside frames"
        piece_cells.append((rr, cc))

    assert seed_hits == 1, "each seed color must appear exactly once inside its frame"
    assert piece_cells, "each seed color needs an external piece"

    coords = np.array(piece_cells, dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    side = int(frame["size"])
    assert int(r1 - r0 + 1) == side and int(c1 - c0 + 1) == side, "piece bbox must match frame interior"

    patch = np.full((side, side), int(bg), dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(color)
    mid = side // 2
    assert int(patch[0, mid]) == int(color), "piece must expose the canonical top-mid anchor"
    return patch, coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each outer colored piece to the frame whose seed pixel shares its color, rotate the
    piece according to the seed side, and place it into the frame interior.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    frames = _find_frames(g, bg)

    piece_patches: dict[int, np.ndarray] = {}
    piece_coords: dict[int, np.ndarray] = {}
    for frame in frames:
        patch, coords = _extract_piece_patch(g, bg, frames, frame)
        piece_patches[int(frame["seed_color"])] = patch
        piece_coords[int(frame["seed_color"])] = coords

    out = g.copy()
    for coords in piece_coords.values():
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        color = int(frame["seed_color"])
        out[r0 + 1 : r1, c0 + 1 : c1] = int(bg)
        rotated = np.rot90(piece_patches[color], int(SIDE_TO_ROT[frame["seed_side"]]))
        seed_r = int(frame["seed_abs"][0] - (r0 + 1))
        seed_c = int(frame["seed_abs"][1] - (c0 + 1))
        assert int(rotated[seed_r, seed_c]) == color, "rotated piece must cover the seed pixel"
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[rotated != int(bg)] = color

    return out


def _rotation_count(mask: np.ndarray) -> int:
    return len({np.rot90(mask, k).tobytes() for k in range(4)})


def _grow_piece_mask(size: int, rng: random.Random) -> np.ndarray | None:
    mid = size // 2
    min_cells = max(4, size + 1)
    max_cells = max(min_cells, size * size - 2)
    target = rng.randint(min_cells, max_cells)

    cells = {(0, mid)}
    frontier = {(1, mid)}
    if mid > 0:
        frontier.add((0, mid - 1))
    if mid + 1 < size:
        frontier.add((0, mid + 1))

    while len(cells) < target and frontier:
        options = [p for p in frontier if 0 <= p[0] < size and 0 <= p[1] < size and p not in cells]
        if not options:
            return None
        weights = []
        for r, c in options:
            weight = 1
            if r == size - 1:
                weight += 5
            if c == 0 or c == size - 1:
                weight += 4
            if r > 0:
                weight += 1
            weights.append(weight)
        nr, nc = rng.choices(options, weights=weights, k=1)[0]
        frontier.discard((nr, nc))
        cells.add((nr, nc))
        for dr, dc in DIR4:
            frontier.add((nr + dr, nc + dc))

    if len(cells) != target:
        return None

    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _piece_mask(size: int, rng: random.Random) -> np.ndarray:
    for _ in range(500):
        mask = _grow_piece_mask(size, rng)
        if mask is None:
            continue
        if not bool(mask[0, size // 2]):
            continue
        if not bool(mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if bool(mask.all()):
            continue
        if _rotation_count(mask) != 4:
            continue
        return mask
    raise RuntimeError("failed to generate a non-symmetric piece")


def _rect_candidates(forbidden: np.ndarray, h: int, w: int, pad: int) -> list[tuple[int, int]]:
    H, W = forbidden.shape
    out: list[tuple[int, int]] = []
    for r in range(1, H - h):
        for c in range(1, W - w):
            r0 = max(0, r - pad)
            r1 = min(H, r + h + pad)
            c0 = max(0, c - pad)
            c1 = min(W, c + w + pad)
            if not bool(forbidden[r0:r1, c0:c1].any()):
                out.append((int(r), int(c)))
    return out


def _reserve(forbidden: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> None:
    H, W = forbidden.shape
    r0 = max(0, top - pad)
    r1 = min(H, top + h + pad)
    c0 = max(0, left - pad)
    c1 = min(W, left + w + pad)
    forbidden[r0:r1, c0:c1] = True


def _draw_frame(grid: np.ndarray, top: int, left: int, size: int, color: int) -> None:
    outer = size + 2
    grid[top, left : left + outer] = FRAME
    grid[top + outer - 1, left : left + outer] = FRAME
    grid[top : top + outer, left] = FRAME
    grid[top : top + outer, left + outer - 1] = FRAME
    mid = size // 2
    seed_map = {
        "top": (top + 1, left + 1 + mid),
        "right": (top + 1 + mid, left + outer - 2),
        "bottom": (top + outer - 2, left + 1 + mid),
        "left": (top + 1 + mid, left + 1),
    }
    return seed_map


def _build_expected(
    inp: np.ndarray,
    frames: list[dict],
    pieces: list[dict],
) -> np.ndarray:
    out = inp.copy()
    for piece in pieces:
        coords = piece["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        out[r0 + 1 : r1, c0 + 1 : c1] = BG
        patch = next(piece["patch"] for piece in pieces if int(piece["color"]) == int(frame["seed_color"]))
        rotated = np.rot90(patch, int(SIDE_TO_ROT[frame["seed_side"]]))
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[rotated != BG] = int(frame["seed_color"])
    return out


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    piece_size = rng.choice((3, 5, 7))
    frame_outer = piece_size + 2
    n_pairs = rng.randint(2, 4)
    colors = rng.sample(PALETTE, n_pairs)
    seed_sides = rng.sample(list(SIDE_TO_ROT), n_pairs)

    total_area = n_pairs * (frame_outer * frame_outer + piece_size * piece_size)
    base = int(np.ceil(np.sqrt(total_area * 2.4)))
    min_dim = max(18, base - 2)
    max_dim = max(min_dim, min(30, base + 6))
    H = rng.randint(min_dim, max_dim)
    W = rng.randint(min_dim, max_dim)

    grid = np.full((H, W), BG, dtype=int)
    forbidden = np.zeros((H, W), dtype=bool)

    frame_specs: list[dict] = []
    for color, side in zip(colors, seed_sides):
        candidates = _rect_candidates(forbidden, frame_outer, frame_outer, pad=1)
        if not candidates:
            return None
        top, left = rng.choice(candidates)
        _reserve(forbidden, top, left, frame_outer, frame_outer, pad=1)
        grid[top, left : left + frame_outer] = FRAME
        grid[top + frame_outer - 1, left : left + frame_outer] = FRAME
        grid[top : top + frame_outer, left] = FRAME
        grid[top : top + frame_outer, left + frame_outer - 1] = FRAME
        mid = piece_size // 2
        seed_map = {
            "top": (top + 1, left + 1 + mid),
            "right": (top + 1 + mid, left + frame_outer - 2),
            "bottom": (top + frame_outer - 2, left + 1 + mid),
            "left": (top + 1 + mid, left + 1),
        }
        sr, sc = seed_map[side]
        grid[sr, sc] = int(color)
        frame_specs.append(
            {
                "bbox": (int(top), int(left), int(top + frame_outer - 1), int(left + frame_outer - 1)),
                "frame_color": FRAME,
                "size": int(piece_size),
                "seed_color": int(color),
                "seed_side": side,
                "seed_abs": (int(sr), int(sc)),
            }
        )

    piece_specs: list[dict] = []
    for color in colors:
        mask = _piece_mask(piece_size, rng)
        patch = np.full((piece_size, piece_size), BG, dtype=int)
        patch[mask] = int(color)
        candidates = _rect_candidates(forbidden, piece_size, piece_size, pad=1)
        if not candidates:
            return None
        top, left = rng.choice(candidates)
        _reserve(forbidden, top, left, piece_size, piece_size, pad=1)
        coords = np.argwhere(mask)
        abs_coords = coords + np.array([[top, left]], dtype=int)
        grid[abs_coords[:, 0], abs_coords[:, 1]] = int(color)
        piece_specs.append(
            {
                "color": int(color),
                "patch": patch,
                "coords": abs_coords.astype(int),
            }
        )

    expected = _build_expected(grid, frame_specs, piece_specs)
    if not bool(np.array_equal(main(grid), expected)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
