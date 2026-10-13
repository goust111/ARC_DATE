from __future__ import annotations

from collections import Counter, defaultdict, deque

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
RNG = np.random.default_rng()
MAX_RETRIES = 400

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NOTCH_TO_CELL = {
    "up": (0, 1),
    "down": (2, 1),
    "left": (1, 0),
    "right": (1, 2),
}
CELL_TO_NOTCH = {v: k for k, v in NOTCH_TO_CELL.items()}
COMPATIBLE = {
    "h": ("up", "down"),
    "v": ("left", "right"),
}


def _mode_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue

            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != int(color):
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )

    return comps


def _make_piece(color: int, notch: str, bg: int = BG) -> np.ndarray:
    patch = np.full((3, 3), int(color), dtype=int)
    rr, cc = NOTCH_TO_CELL[notch]
    patch[rr, cc] = int(bg)
    return patch


def _detect_notch(patch: np.ndarray, color: int, bg: int) -> str | None:
    if patch.shape != (3, 3):
        return None
    if int(np.sum(patch == int(bg))) != 1:
        return None

    missing = tuple(map(int, np.argwhere(patch == int(bg))[0]))
    if missing not in CELL_TO_NOTCH:
        return None

    filled = patch.copy()
    filled[missing] = int(color)
    if not bool((filled == int(color)).all()):
        return None
    return CELL_TO_NOTCH[missing]


def _paste(out: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int | None = None) -> None:
    h, w = patch.shape
    assert 0 <= r0 <= out.shape[0] - h
    assert 0 <= c0 <= out.shape[1] - w

    view = out[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        mask = np.ones((h, w), dtype=bool)
    else:
        mask = patch != int(transparent)

    clash = mask & (view != int(BG)) & (view != patch)
    assert not bool(clash.any())
    view[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock each 3x3-minus-one fragment onto the same-colored magnetic bar.

    The notch direction selects which side of the bar the fragment approaches from,
    and fragments sharing one bar are packed from the top/left endpoint in the order
    induced by their original row/column position.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    comps = _find_components(g, bg)

    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(comp)

    out = g.copy()

    for color, parts in by_color.items():
        bars: list[tuple[str, dict]] = []
        pieces: list[dict] = []

        for comp in parts:
            r0, c0, r1, c1 = comp["bbox"]
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            area = int(comp["area"])

            if h == 1 and area == w and area >= 4:
                bars.append(("h", comp))
                continue
            if w == 1 and area == h and area >= 4:
                bars.append(("v", comp))
                continue

            if area == 8 and h == 3 and w == 3:
                patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
                notch = _detect_notch(patch, color, bg)
                if notch is not None:
                    pieces.append(
                        {
                            "bbox": comp["bbox"],
                            "coords": comp["coords"],
                            "patch": patch,
                            "notch": notch,
                        }
                    )

        assert len(bars) == 1
        assert 1 <= len(pieces) <= 2

        orient, bar = bars[0]
        if orient == "h":
            pieces.sort(key=lambda p: (int(p["bbox"][1]), int(p["bbox"][0])))
        else:
            pieces.sort(key=lambda p: (int(p["bbox"][0]), int(p["bbox"][1])))

        for piece in pieces:
            rows = piece["coords"][:, 0]
            cols = piece["coords"][:, 1]
            out[rows, cols] = int(bg)

        if orient == "h":
            bar_row = int(bar["bbox"][0])
            bar_c0 = int(bar["bbox"][1])
            for idx, piece in enumerate(pieces):
                assert piece["notch"] in COMPATIBLE["h"]
                rr = int(bar_row if piece["notch"] == "up" else bar_row - 2)
                cc = int(bar_c0 + 3 * idx)
                _paste(out, piece["patch"], rr, cc, transparent=bg)
        else:
            bar_col = int(bar["bbox"][1])
            bar_r0 = int(bar["bbox"][0])
            for idx, piece in enumerate(pieces):
                assert piece["notch"] in COMPATIBLE["v"]
                rr = int(bar_r0 + 3 * idx)
                cc = int(bar_col if piece["notch"] == "left" else bar_col - 2)
                _paste(out, piece["patch"], rr, cc, transparent=bg)

    return out


def _rect_is_free(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 0) -> bool:
    rr0 = max(0, int(r0 - pad))
    cc0 = max(0, int(c0 - pad))
    rr1 = min(mask.shape[0], int(r0 + h + pad))
    cc1 = min(mask.shape[1], int(c0 + w + pad))
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _mark_rect(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 0) -> None:
    rr0 = max(0, int(r0 - pad))
    cc0 = max(0, int(c0 - pad))
    rr1 = min(mask.shape[0], int(r0 + h + pad))
    cc1 = min(mask.shape[1], int(c0 + w + pad))
    mask[rr0:rr1, cc0:cc1] = True


def _draw_bar(grid: np.ndarray, spec: dict) -> None:
    color = int(spec["color"])
    if spec["orient"] == "h":
        r = int(spec["bar_row"])
        c0 = int(spec["bar_c0"])
        c1 = int(spec["bar_c1"])
        grid[r, c0 : c1 + 1] = color
    else:
        c = int(spec["bar_col"])
        r0 = int(spec["bar_r0"])
        r1 = int(spec["bar_r1"])
        grid[r0 : r1 + 1, c] = color


def _place_piece_slot(spec: dict, idx: int, notch: str) -> tuple[int, int]:
    if spec["orient"] == "h":
        r = int(spec["bar_row"] if notch == "up" else spec["bar_row"] - 2)
        c = int(spec["bar_c0"] + 3 * idx)
    else:
        r = int(spec["bar_r0"] + 3 * idx)
        c = int(spec["bar_col"] if notch == "left" else spec["bar_col"] - 2)
    return r, c


def _sample_bar_specs(h: int, w: int) -> list[dict]:
    n_bars = int(RNG.integers(2, 5))
    colors = [int(x) for x in RNG.choice(PALETTE, size=n_bars, replace=False)]

    if n_bars == 2 and RNG.random() < 0.7:
        orientations = list(RNG.permutation(["h", "v"]))
    else:
        orientations = [("h" if int(RNG.integers(0, 2)) == 0 else "v") for _ in range(n_bars)]
        if n_bars >= 2 and len(set(orientations)) == 1 and RNG.random() < 0.7:
            flip_idx = int(RNG.integers(0, n_bars))
            orientations[flip_idx] = "v" if orientations[flip_idx] == "h" else "h"

    specs: list[dict] = []
    for color, orient in zip(colors, orientations):
        piece_count = int(RNG.integers(1, 3))
        min_len = max(4, 3 * piece_count)
        max_len = min(11, (w if orient == "h" else h) - 2)
        if max_len < min_len:
            raise ValueError("grid too small for bar spec")
        length = int(RNG.integers(min_len, max_len + 1))
        specs.append(
            {
                "color": color,
                "orient": orient,
                "piece_count": piece_count,
                "length": length,
            }
        )
    return specs


def _place_bars(h: int, w: int, specs: list[dict]) -> bool:
    reserved = np.zeros((h, w), dtype=bool)

    for spec in specs:
        placed = False
        for _ in range(300):
            length = int(spec["length"])
            if spec["orient"] == "h":
                if h < 5 or w < length:
                    return False
                z_r = int(RNG.integers(0, h - 5 + 1))
                z_c = int(RNG.integers(0, w - length + 1))
                if not _rect_is_free(reserved, z_r, z_c, 5, length, pad=1):
                    continue
                _mark_rect(reserved, z_r, z_c, 5, length, pad=1)
                spec["bar_row"] = int(z_r + 2)
                spec["bar_c0"] = int(z_c)
                spec["bar_c1"] = int(z_c + length - 1)
                placed = True
                break
            else:
                if h < length or w < 5:
                    return False
                z_r = int(RNG.integers(0, h - length + 1))
                z_c = int(RNG.integers(0, w - 5 + 1))
                if not _rect_is_free(reserved, z_r, z_c, length, 5, pad=1):
                    continue
                _mark_rect(reserved, z_r, z_c, length, 5, pad=1)
                spec["bar_r0"] = int(z_r)
                spec["bar_r1"] = int(z_r + length - 1)
                spec["bar_col"] = int(z_c + 2)
                placed = True
                break
        if not placed:
            return False

    return True


def _place_scattered_pieces(inp: np.ndarray, specs: list[dict]) -> np.ndarray | None:
    h, w = inp.shape
    expected = np.full_like(inp, BG)

    for spec in specs:
        _draw_bar(inp, spec)
        _draw_bar(expected, spec)

    occupied = np.zeros((h, w), dtype=bool)
    for spec in specs:
        if spec["orient"] == "h":
            _mark_rect(occupied, int(spec["bar_row"]), int(spec["bar_c0"]), 1, int(spec["length"]), pad=1)
        else:
            _mark_rect(occupied, int(spec["bar_r0"]), int(spec["bar_col"]), int(spec["length"]), 1, pad=1)

    for spec in specs:
        positions: list[tuple[int, int]] = []
        for _ in range(int(spec["piece_count"])):
            found = None
            for _ in range(400):
                r0 = int(RNG.integers(0, h - 3 + 1))
                c0 = int(RNG.integers(0, w - 3 + 1))
                if not _rect_is_free(occupied, r0, c0, 3, 3, pad=1):
                    continue
                _mark_rect(occupied, r0, c0, 3, 3, pad=1)
                found = (int(r0), int(c0))
                break
            if found is None:
                return None
            positions.append(found)

        if spec["orient"] == "h":
            positions.sort(key=lambda rc: (int(rc[1]), int(rc[0])))
        else:
            positions.sort(key=lambda rc: (int(rc[0]), int(rc[1])))

        notches = [COMPATIBLE[spec["orient"]][int(RNG.integers(0, 2))] for _ in positions]
        spec["notches"] = list(notches)

        for idx, ((r0, c0), notch) in enumerate(zip(positions, notches)):
            patch = _make_piece(int(spec["color"]), notch, bg=BG)
            _paste(inp, patch, int(r0), int(c0), transparent=BG)
            tr, tc = _place_piece_slot(spec, idx, notch)
            _paste(expected, patch, int(tr), int(tc), transparent=BG)

    return expected


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(20, 31))
    w = int(RNG.integers(20, 31))

    try:
        specs = _sample_bar_specs(h, w)
    except ValueError:
        return None

    if not _place_bars(h, w, specs):
        return None

    inp = np.full((h, w), BG, dtype=int)
    expected = _place_scattered_pieces(inp, specs)
    if expected is None:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid magnetic-bar sample")
