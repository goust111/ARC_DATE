from __future__ import annotations

from collections import Counter
import numpy as np


Coord = tuple[int, int]


def _bg_color(grid: np.ndarray) -> int:
    return 0 if np.any(grid == 0) else int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(comp)
    return parts


def _shape_key(coords: list[Coord]) -> tuple[Coord, ...]:
    r0 = min(r for r, _c in coords)
    c0 = min(c for _r, c in coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[Coord]:
    coords: list[Coord] = []
    for c in range(c0, c1 + 1):
        coords.append((int(r0), int(c)))
        coords.append((int(r1), int(c)))
    for r in range(r0 + 1, r1):
        coords.append((int(r), int(c0)))
        coords.append((int(r), int(c1)))
    return coords


def _frame_bbox_if_valid(grid: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(grid == int(color))
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    mask = grid == int(color)
    rows = coords[:, 0]
    cols = coords[:, 1]
    if not np.all((rows == r0) | (rows == r1) | (cols == c0) | (cols == c1)):
        return None
    if not (mask[r0, c0] and mask[r0, c1] and mask[r1, c0] and mask[r1, c1]):
        return None
    return r0, c0, r1, c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble keyed line fragments back into their matching broken rectangle borders.

    Input structure:
    - Black background with several single-cell-thick rectangular borders in different colors.
    - Each border color forms an incomplete rectangle: one or more straight edge segments are missing.
    - Outside the frames are line fragments in other colors, and each fragment color appears once more as a single marker pixel inside exactly one frame.

    Rule:
    - Identify each frame by its incomplete rectangular border geometry.
    - Use the interior marker pixel to pair one fragment color with one frame, then compare the outside fragment shapes with the missing perimeter segments of that frame.
    - Remove the marker and the outside fragments, and fill the missing perimeter cells with the frame's border color.

    Key insight:
    - The marker pixel is the ownership key, while the fragment geometry reveals which border gaps must be restored.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    colors = [int(color) for color in np.unique(g) if int(color) != int(bg)]
    frames: list[dict[str, object]] = []
    frame_by_color: dict[int, int] = {}
    for color in colors:
        bbox = _frame_bbox_if_valid(g, int(color))
        if bbox is None:
            continue
        frame_by_color[int(color)] = len(frames)
        frames.append({"frame_color": int(color), "bbox": bbox})

    assert frames, "expected at least one incomplete frame"

    owner_by_piece_color: dict[int, int] = {}
    marker_by_piece_color: dict[int, Coord] = {}
    fragments_by_piece_color: dict[int, list[list[Coord]]] = {}

    for color in colors:
        if color in frame_by_color:
            continue
        coords = [tuple(map(int, coord)) for coord in np.argwhere(g == int(color))]
        owner: int | None = None
        marker: Coord | None = None
        outside_mask = np.zeros(g.shape, dtype=bool)

        for r, c in coords:
            inside = []
            for idx, frame in enumerate(frames):
                r0, c0, r1, c1 = frame["bbox"]  # type: ignore[index]
                if int(r0) < r < int(r1) and int(c0) < c < int(c1):
                    inside.append(idx)
            if inside:
                assert len(inside) == 1, "marker cannot belong to multiple frames"
                if owner is None:
                    owner = int(inside[0])
                assert owner == int(inside[0]), "a fragment color must map to one frame"
                assert marker is None, "expected exactly one marker pixel"
                marker = (int(r), int(c))
            else:
                outside_mask[int(r), int(c)] = True

        assert owner is not None and marker is not None, "piece color must have one interior marker"
        fragment_comps = _components(outside_mask)
        assert fragment_comps, "piece color must have outside fragments"
        owner_by_piece_color[int(color)] = int(owner)
        marker_by_piece_color[int(color)] = marker
        fragments_by_piece_color[int(color)] = fragment_comps

    piece_color_for_frame: dict[int, int] = {}
    for piece_color, owner in owner_by_piece_color.items():
        assert owner not in piece_color_for_frame, "each frame must have a unique marker color"
        piece_color_for_frame[int(owner)] = int(piece_color)

    for idx, frame in enumerate(frames):
        assert idx in piece_color_for_frame, "every frame must have a marker/piece color"
        piece_color = int(piece_color_for_frame[idx])
        frame_color = int(frame["frame_color"])  # type: ignore[index]
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[index]

        border = _border_coords(int(r0), int(c0), int(r1), int(c1))
        missing_mask = np.zeros(g.shape, dtype=bool)
        missing_coords: list[Coord] = []
        for r, c in border:
            if int(g[r, c]) != frame_color:
                missing_mask[int(r), int(c)] = True
                missing_coords.append((int(r), int(c)))

        missing_comps = _components(missing_mask)
        need = Counter(_shape_key(comp) for comp in missing_comps)
        have = Counter(_shape_key(comp) for comp in fragments_by_piece_color[piece_color])
        assert need == have, "outside fragments must match the frame gaps"

        mr, mc = marker_by_piece_color[piece_color]
        out[int(mr), int(mc)] = bg
        for comp in fragments_by_piece_color[piece_color]:
            for r, c in comp:
                out[int(r), int(c)] = bg
        for r, c in missing_coords:
            out[int(r), int(c)] = frame_color

    return out


# --------------- generator ---------------

GEN_BG = 0
GEN_CANVAS = 30
GEN_FRAME_COUNT_RANGE = (2, 5)
GEN_FRAME_H_RANGE = (5, 10)
GEN_FRAME_W_RANGE = (5, 10)
GEN_FRAME_PAD = 1
GEN_FRAGMENT_PAD = 1
GEN_GAPS_PER_FRAME = (1, 3)
GEN_CROP_MARGIN = 3
GEN_ATTEMPTS = 400


def _draw_frame(
    canvas: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    color: int,
    gap_cells: set[Coord] | None = None,
) -> None:
    gaps = gap_cells or set()
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for r, c in _border_coords(r0, c0, r1, c1):
        if (int(r), int(c)) not in gaps:
            canvas[int(r), int(c)] = int(color)


def _line_patch(length: int, vertical: bool, color: int) -> np.ndarray:
    if vertical:
        patch = np.full((length, 1), int(color), dtype=int)
    else:
        patch = np.full((1, length), int(color), dtype=int)
    return patch


def _reserve_spot(
    blocked: np.ndarray,
    height: int,
    width: int,
    pad: int,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    h, w = blocked.shape
    candidates: list[Coord] = []
    for r0 in range(h - height + 1):
        for c0 in range(w - width + 1):
            rr0 = max(0, r0 - pad)
            cc0 = max(0, c0 - pad)
            rr1 = min(h, r0 + height + pad)
            cc1 = min(w, c0 + width + pad)
            if not blocked[rr0:rr1, cc0:cc1].any():
                candidates.append((int(r0), int(c0)))
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))]


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, int(r0) - pad)
    cc0 = max(0, int(c0) - pad)
    rr1 = min(blocked.shape[0], int(r0) + h + pad)
    cc1 = min(blocked.shape[1], int(c0) + w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _sample_gaps(
    rng: np.random.Generator,
    h: int,
    w: int,
    required_orientation: str | None,
) -> list[list[Coord]]:
    all_sides = ["top", "bottom", "left", "right"]
    if required_orientation == "H":
        mandatory = "top" if int(rng.integers(0, 2)) == 0 else "bottom"
    elif required_orientation == "V":
        mandatory = "left" if int(rng.integers(0, 2)) == 0 else "right"
    else:
        mandatory = None

    total = int(rng.integers(GEN_GAPS_PER_FRAME[0], GEN_GAPS_PER_FRAME[1] + 1))
    chosen: list[str] = []
    if mandatory is not None:
        chosen.append(mandatory)
    remaining = [side for side in all_sides if side not in chosen]
    order = list(np.asarray(remaining, dtype=object)[rng.permutation(len(remaining))])
    chosen.extend(order[: max(0, total - len(chosen))])

    gaps: list[list[Coord]] = []
    for side in chosen:
        if side in ("top", "bottom"):
            inner_len = w - 2
            length = int(rng.integers(1, inner_len + 1))
            offset = int(rng.integers(0, inner_len - length + 1))
            row = 0 if side == "top" else h - 1
            gaps.append([(int(row), int(1 + offset + k)) for k in range(length)])
        else:
            inner_len = h - 2
            length = int(rng.integers(1, inner_len + 1))
            offset = int(rng.integers(0, inner_len - length + 1))
            col = 0 if side == "left" else w - 1
            gaps.append([(int(1 + offset + k), int(col)) for k in range(length)])
    return gaps


def _crop_pair(
    inp: np.ndarray,
    out: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != GEN_BG)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    top = int(rng.integers(0, min(GEN_CROP_MARGIN, r0) + 1))
    left = int(rng.integers(0, min(GEN_CROP_MARGIN, c0) + 1))
    bottom = int(rng.integers(0, min(GEN_CROP_MARGIN, inp.shape[0] - 1 - r1) + 1))
    right = int(rng.integers(0, min(GEN_CROP_MARGIN, inp.shape[1] - 1 - c1) + 1))
    rs = slice(r0 - top, r1 + bottom + 1)
    cs = slice(c0 - left, c1 + right + 1)
    return inp[rs, cs].copy(), out[rs, cs].copy()


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    grid_in = np.full((GEN_CANVAS, GEN_CANVAS), GEN_BG, dtype=int)
    grid_out = np.full_like(grid_in, GEN_BG)
    blocked = np.zeros_like(grid_in, dtype=bool)

    n_frames = int(rng.integers(*GEN_FRAME_COUNT_RANGE))
    palette = rng.choice(np.arange(1, 10), size=2 * n_frames, replace=False).astype(int)
    border_colors = [int(x) for x in palette[:n_frames]]
    piece_colors = [int(x) for x in palette[n_frames:]]

    required_orientations: list[str | None] = ["H", "V"] + [None] * max(0, n_frames - 2)
    frame_specs: list[dict[str, object]] = []

    for i in range(n_frames):
        h = int(rng.integers(GEN_FRAME_H_RANGE[0], GEN_FRAME_H_RANGE[1] + 1))
        w = int(rng.integers(GEN_FRAME_W_RANGE[0], GEN_FRAME_W_RANGE[1] + 1))
        pos = _reserve_spot(blocked, h, w, GEN_FRAME_PAD, rng)
        if pos is None:
            return None
        r0, c0 = pos
        _mark_blocked(blocked, r0, c0, h, w, GEN_FRAME_PAD)

        gaps_rel = _sample_gaps(rng, h, w, required_orientations[i])
        gaps_abs = [
            [(int(r0 + rr), int(c0 + cc)) for rr, cc in rel_gap]
            for rel_gap in gaps_rel
        ]
        marker_r = int(r0 + rng.integers(1, h - 1))
        marker_c = int(c0 + rng.integers(1, w - 1))
        frame_specs.append(
            {
                "frame_color": int(border_colors[i]),
                "piece_color": int(piece_colors[i]),
                "bbox": (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)),
                "marker": (int(marker_r), int(marker_c)),
                "gaps": gaps_abs,
            }
        )

    for spec in frame_specs:
        r0, c0, r1, c1 = spec["bbox"]  # type: ignore[index]
        h = int(r1) - int(r0) + 1
        w = int(c1) - int(c0) + 1
        gap_cells = {cell for gap in spec["gaps"] for cell in gap}  # type: ignore[index]
        _draw_frame(grid_out, int(r0), int(c0), h, w, int(spec["frame_color"]))
        _draw_frame(grid_in, int(r0), int(c0), h, w, int(spec["frame_color"]), gap_cells=gap_cells)
        mr, mc = spec["marker"]  # type: ignore[index]
        grid_in[int(mr), int(mc)] = int(spec["piece_color"])

    for spec in frame_specs:
        piece_color = int(spec["piece_color"])  # type: ignore[index]
        for gap in spec["gaps"]:  # type: ignore[index]
            rows = {int(r) for r, _c in gap}
            cols = {int(c) for _r, c in gap}
            vertical = len(cols) == 1 and len(rows) > 1
            patch = _line_patch(len(gap), vertical, piece_color)
            pos = _reserve_spot(blocked, patch.shape[0], patch.shape[1], GEN_FRAGMENT_PAD, rng)
            if pos is None:
                return None
            pr0, pc0 = pos
            _mark_blocked(blocked, pr0, pc0, patch.shape[0], patch.shape[1], GEN_FRAGMENT_PAD)
            grid_in[pr0 : pr0 + patch.shape[0], pc0 : pc0 + patch.shape[1]] = patch

    inp, expected = _crop_pair(grid_in, grid_out, rng)
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
