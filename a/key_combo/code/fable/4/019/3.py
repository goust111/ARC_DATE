from __future__ import annotations

import importlib.util
from pathlib import Path
import random
from typing import Final

import numpy as np

try:
    from arc_utils import Color, find_components, get_bg
except Exception:  # pragma: no cover
    _UTIL_PATH: Path | None = None
    for _parent in Path(__file__).resolve().parents:
        candidate = _parent / "new" / "solvers" / "arc_utils.py"
        if candidate.exists():
            _UTIL_PATH = candidate
            break
    if _UTIL_PATH is None:
        raise
    _spec = importlib.util.spec_from_file_location("arc_utils", _UTIL_PATH)
    if _spec is None or _spec.loader is None:
        raise RuntimeError("cannot load arc_utils module")
    _arc_utils = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_arc_utils)
    Color = _arc_utils.Color
    find_components = _arc_utils.find_components
    get_bg = _arc_utils.get_bg


_DIR_TO_DRC: Final = {
    int(Color.BLUE): (0, 1),
    int(Color.RED): (0, -1),
    int(Color.GREEN): (1, 0),
    int(Color.PINK): (-1, 0),
}


def _collect_framed_tiles(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    tiles: list[dict[str, object]] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color == bg:
            continue

        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        area = int(comp["area"])
        if area != 2 * (h + w) - 4:
            continue

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        if not np.all(patch[border] == color):
            continue

        inner = patch[1:-1, 1:-1]
        if inner.size == 0:
            continue
        inner_color = int(inner[0, 0])
        if inner_color == bg or inner_color == color:
            continue
        if not np.all(inner == inner_color):
            continue

        tiles.append(
            {
                "frame_coords": comp["coords"],
                "inner_color": inner_color,
                "inner_rows": slice(r0 + 1, r1),
                "inner_cols": slice(c0 + 1, c1),
            }
        )
    return tiles


def _collect_marker_pixels(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    for color, _drdc in _DIR_TO_DRC.items():
        for comp in find_components(grid, bg=bg, connectivity=8, monochromatic=True, colors=[color], return_format="dict"):
            if int(comp["area"]) != 1:
                continue
            r, c = map(int, comp["coords"][0])
            markers.append((r, c, color))
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """Swap each framed tile's interior with its direction-marker color through a hit test.

    Input structure
    - A grid with a dominant background color.
    - Several separated rectangular framed tiles (single-color frame, single-color interior).
    - Several isolated background marker pixels in fixed direction colors:
      blue=right, red=left, green=down, pink=up.

    Rule
    - For each marker, cast a ray from the marker along its direction until the first
      framed tile frame pixel is reached.
    - Replace the whole tile interior with the marker color, and replace the marker
      with that tile's original interior color.
    - Keep all frames, unhit tiles, and background unchanged; draw nothing for the ray path.

    Key insight
    - Markers are not moved and do not leave traces; they encode the target via
      directional ray order, so only one interior-to-marker color swap occurs per hit.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    tiles = _collect_framed_tiles(g, bg)
    assert tiles, "expected at least one framed tile"

    frame_of = np.full(g.shape, -1, dtype=int)
    for i, tile in enumerate(tiles):
        for r, c in tile["frame_coords"]:
            frame_of[int(r), int(c)] = i

    markers = _collect_marker_pixels(g, bg)
    assert markers, "expected at least one marker pixel"

    H, W = g.shape
    hits: list[tuple[int, int, int, int]] = []
    used_tiles: set[int] = set()

    for mr, mc, mcolor in markers:
        dr, dc = _DIR_TO_DRC[mcolor]
        rr = mr + dr
        cc = mc + dc

        hit: int = -1
        while 0 <= rr < H and 0 <= cc < W:
            idx = int(frame_of[rr, cc])
            if idx != -1:
                hit = idx
                break
            if g[rr, cc] != bg:
                break
            rr += dr
            cc += dc

        assert hit >= 0, "each marker must hit a framed tile first"
        assert hit not in used_tiles, "a tile must be hit by at most one marker"
        used_tiles.add(hit)
        hits.append((mr, mc, mcolor, hit))

    out = g.copy()
    for mr, mc, mcolor, tile_idx in hits:
        tile = tiles[tile_idx]
        inner_color = int(tile["inner_color"])
        out[tile["inner_rows"], tile["inner_cols"]] = mcolor
        out[mr, mc] = inner_color
    return out


def _has_room(mask: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    H, W = mask.shape
    r0 = int(r0)
    c0 = int(c0)
    h = int(h)
    w = int(w)
    if r0 < 1 or c0 < 1 or r0 + h >= H or c0 + w >= W:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(H, r0 + h + 1)
    cc1 = min(W, c0 + w + 1)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _is_isolated(mask: np.ndarray, r: int, c: int) -> bool:
    H, W = mask.shape
    for rr in range(r - 1, r + 2):
        if rr < 0 or rr >= H:
            continue
        for cc in range(c - 1, c + 2):
            if cc < 0 or cc >= W:
                continue
            if mask[rr, cc]:
                return False
    return True


def _candidates_right(grid: np.ndarray, tile: dict[str, object], blocked: np.ndarray, bg: int) -> list[tuple[int, int]]:
    r0 = int(tile["r0"])
    r1 = int(tile["r1"])
    c0 = int(tile["c0"])
    H, W = grid.shape
    out: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0):
            if not (grid[r, c] == bg and _is_isolated(blocked, r, c)):
                continue
            if np.any(blocked[r, c + 1 : c0]):
                continue
            if np.any(grid[r, c + 1 : c0] != bg):
                continue
            out.append((r, c))
    return out


def _candidates_left(grid: np.ndarray, tile: dict[str, object], blocked: np.ndarray, bg: int) -> list[tuple[int, int]]:
    r0 = int(tile["r0"])
    r1 = int(tile["r1"])
    c1 = int(tile["c1"])
    H, W = grid.shape
    out: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c1 + 1, W):
            if not (grid[r, c] == bg and _is_isolated(blocked, r, c)):
                continue
            if np.any(blocked[r, c1 + 1 : c]):
                continue
            if np.any(grid[r, c1 + 1 : c] != bg):
                continue
            out.append((r, c))
    return out


def _candidates_down(grid: np.ndarray, tile: dict[str, object], blocked: np.ndarray, bg: int) -> list[tuple[int, int]]:
    c0 = int(tile["c0"])
    c1 = int(tile["c1"])
    r1 = int(tile["r1"])
    H, W = grid.shape
    out: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        for r in range(r1 + 1, H):
            if not (grid[r, c] == bg and _is_isolated(blocked, r, c)):
                continue
            if np.any(blocked[r1 + 1 : r, c]):
                continue
            if np.any(grid[r1 + 1 : r, c] != bg):
                continue
            out.append((r, c))
    return out


def _candidates_up(grid: np.ndarray, tile: dict[str, object], blocked: np.ndarray, bg: int) -> list[tuple[int, int]]:
    c0 = int(tile["c0"])
    c1 = int(tile["c1"])
    r0 = int(tile["r0"])
    out: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        for r in range(r0):
            if not (grid[r, c] == bg and _is_isolated(blocked, r, c)):
                continue
            if np.any(blocked[r + 1 : r0, c]):
                continue
            if np.any(grid[r + 1 : r0, c] != bg):
                continue
            out.append((r, c))
    return out


def _gen_tile_info(
    h: int,
    w: int,
    frame_color: int,
    inner_color: int,
    top: int,
    left: int,
) -> dict[str, object]:
    r0, c0 = top, left
    r1, c1 = top + h - 1, left + w - 1
    return {
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "frame_color": int(frame_color),
        "inner_color": int(inner_color),
    }


def _paint_tile(grid: np.ndarray, tile: dict[str, object]) -> None:
    patch = np.full(
        (int(tile["r1"]) - int(tile["r0"]) + 1, int(tile["c1"]) - int(tile["c0"]) + 1),
        int(tile["frame_color"]),
        dtype=int,
    )
    patch[1:-1, 1:-1] = int(tile["inner_color"])
    grid[int(tile["r0"]) : int(tile["r1"]) + 1, int(tile["c0"]) : int(tile["c1"]) + 1] = patch


def _make_tiles(H: int, W: int, bg: int) -> tuple[np.ndarray, list[dict[str, object]]]:
    occupied = np.zeros((H, W), dtype=bool)
    grid = np.full((H, W), bg, dtype=int)
    tile_count = random.randint(1, 4)

    frame_palette = [4, 5, 7, 8, 9]
    inner_palette = [1, 4, 5, 7, 9]
    frame_colors = random.sample(frame_palette, min(tile_count, len(frame_palette)))
    inner_colors = random.sample(inner_palette, tile_count)

    tiles: list[dict[str, object]] = []
    for idx in range(tile_count):
        frame_color = frame_colors[idx]
        inner_color = inner_colors[idx % len(inner_colors)]
        if inner_color == frame_color:
            inner_color = random.choice([c for c in inner_palette if c != frame_color])

        for _ in range(200):
            h = random.randint(3, 9)
            w = random.randint(3, 10)
            top = random.randint(0, H - h - 1)
            left = random.randint(0, W - w - 1)
            if not _has_room(occupied, top, left, h, w):
                continue
            tile = _gen_tile_info(h, w, frame_color, inner_color, top, left)
            occupied[top - 1 : top + h + 1, left - 1 : left + w + 1] = True
            _paint_tile(grid, tile)
            tiles.append(tile)
            break
        else:
            return np.empty((0, 0), dtype=int), []
    return grid, tiles


def generate_input() -> np.ndarray:
    """Generate multiple-framed-tile inputs with isolated directional markers in a valid layout."""
    max_trials = 300
    for _ in range(max_trials):
        H = random.randint(12, 30)
        W = random.randint(12, 30)
        bg = 0
        base, tiles = _make_tiles(H, W, bg)
        if base.size == 0:
            continue
        occupied = base != bg

        marker_specs = []
        for tile in tiles:
            tile_candidates: dict[int, list[tuple[int, int]]] = {
                int(Color.BLUE): _candidates_right(base, tile, occupied, bg),
                int(Color.RED): _candidates_left(base, tile, occupied, bg),
                int(Color.GREEN): _candidates_down(base, tile, occupied, bg),
                int(Color.PINK): _candidates_up(base, tile, occupied, bg),
            }
            tile_candidates = {color: cand for color, cand in tile_candidates.items() if cand}
            if tile_candidates:
                marker_specs.append(tile_candidates)

        if not marker_specs:
            continue

        for tile_idx, tile in enumerate(tiles):
            tile_key = tuple(
                (int(tile["r0"]), int(tile["c0"]), int(tile["r1"]), int(tile["c1"]))
            )
            _ = tile_key
            # random chance of keeping some tiles unhit for branch diversity
            if random.random() > 0.8:
                continue
            tile_candidates = {
                int(Color.BLUE): _candidates_right(base, tile, occupied, bg),
                int(Color.RED): _candidates_left(base, tile, occupied, bg),
                int(Color.GREEN): _candidates_down(base, tile, occupied, bg),
                int(Color.PINK): _candidates_up(base, tile, occupied, bg),
            }
            tile_candidates = {color: cand for color, cand in tile_candidates.items() if cand}
            if not tile_candidates:
                continue
            color, choices = random.choice(list(tile_candidates.items()))
            r, c = random.choice(choices)
            base[r, c] = color
            occupied[r, c] = True

        if not np.any(np.isin(base, list(_DIR_TO_DRC.keys()))):
            all_tiles = list(range(len(tiles)))
            if not all_tiles:
                continue
            choose = random.choice(all_tiles)
            tile = tiles[choose]
            fallback = {
                int(Color.BLUE): _candidates_right(base, tile, occupied, bg),
                int(Color.RED): _candidates_left(base, tile, occupied, bg),
                int(Color.GREEN): _candidates_down(base, tile, occupied, bg),
                int(Color.PINK): _candidates_up(base, tile, occupied, bg),
            }
            fallback = {color: cand for color, cand in fallback.items() if cand}
            if not fallback:
                continue
            color, choices = random.choice(list(fallback.items()))
            r, c = random.choice(choices)
            base[r, c] = color
            occupied[r, c] = True

        # validate exact self-consistency
        try:
            main(base)
        except Exception:
            continue
        return base.astype(int)
    raise RuntimeError("failed to generate valid 019-3 input")

